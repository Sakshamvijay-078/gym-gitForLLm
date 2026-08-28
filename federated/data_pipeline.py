"""
data_pipeline.py — MoE-aware data bucketing pipeline.

Pipeline:
  1. Load a HuggingFace MoE model (frozen) with its tokenizer.
  2. Stream tokenized sentence-level chunks through the router.
  3. Record the top-K expert indices (the "path") for each chunk.
  4. Compute path-purity: fraction of routing probability mass
     captured by exactly those top-K experts.
  5. Group chunks into buckets keyed by sorted expert path.
  6. Rank + preferentially sample high-purity chunks inside each bucket.
  7. Serialize buckets to disk as HuggingFace Arrow datasets.

Usage:
  python data_pipeline.py \
      --model_name_or_path "mistralai/Mixtral-8x7B-Instruct-v0.1" \
      --dataset_name "HuggingFaceH4/ultrachat_200k" \
      --output_dir ./buckets \
      --top_k 2 \
      --purity_threshold 0.80 \
      --max_chunks 100000 \
      --batch_size 32
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.nn import functional as F

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ──────────────────────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ChunkRecord:
    text:          str
    expert_path:   tuple[int, ...]   # sorted top-K expert indices
    purity_score:  float             # ∈ [0, 1]
    routing_probs: list[float]       # full distribution over all experts


@dataclass
class ExpertBucket:
    path:           tuple[int, ...]
    records:        list[ChunkRecord] = field(default_factory=list)

    @property
    def mean_purity(self) -> float:
        if not self.records:
            return 0.0
        return float(np.mean([r.purity_score for r in self.records]))

    def ranked_sample(self, n: int | None = None) -> list[ChunkRecord]:
        """Return records sorted by purity descending; optionally truncate to n."""
        ranked = sorted(self.records, key=lambda r: r.purity_score, reverse=True)
        return ranked[:n] if n is not None else ranked


# ──────────────────────────────────────────────────────────────────────────────
# Router Hook
# ──────────────────────────────────────────────────────────────────────────────

class RouterCapture:
    """
    Forward hook that captures MoE routing decisions.

    Compatible with Mixtral-style MoE where each MoE layer has a
    `gate` sub-module that returns logits over `num_experts`.
    """

    def __init__(self, top_k: int):
        self.top_k = top_k
        self.routing_decisions: list[tuple[tuple[int, ...], list[float]]] = []
        self._hooks: list = []

    def attach(self, model: torch.nn.Module) -> None:
        """Attach hooks to all MoE gate modules."""
        attached = 0
        for name, module in model.named_modules():
            # Mixtral: MixtralSparseMoeBlock has a .gate attribute
            if hasattr(module, "gate") and isinstance(module.gate, torch.nn.Linear):
                h = module.gate.register_forward_hook(self._make_hook(name))
                self._hooks.append(h)
                attached += 1
            # Fallback: any module named "router" or "gate"
            elif any(kw in name.lower() for kw in ("router", ".gate")) and \
                 isinstance(module, torch.nn.Linear) and module.out_features > 4:
                h = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(h)
                attached += 1
        log.info("RouterCapture: attached %d hooks", attached)
        if attached == 0:
            raise RuntimeError("No MoE gate modules found. Check model architecture.")

    def _make_hook(self, layer_name: str):
        def hook(module, _inputs, output):
            # output: (batch, num_experts) logits  — or (batch, seq, experts)
            logits = output.detach().float()
            if logits.ndim == 3:
                logits = logits.mean(dim=1)   # avg over tokens in chunk
            probs = F.softmax(logits, dim=-1)  # (batch, num_experts)
            top_vals, top_idx = torch.topk(probs, self.top_k, dim=-1)
            for b in range(probs.shape[0]):
                path   = tuple(sorted(top_idx[b].tolist()))
                purity = top_vals[b].sum().item()   # mass on top-K
                self.routing_decisions.append((path, probs[b].tolist()))
        return hook

    def flush(self) -> list[tuple[tuple[int, ...], list[float]]]:
        """Pop and return all captured decisions."""
        out = self.routing_decisions[:]
        self.routing_decisions.clear()
        return out

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ──────────────────────────────────────────────────────────────────────────────
# Core Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def load_frozen_model(
    model_name: str,
    use_4bit: bool = True,
    device: str = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a quantized, frozen model for routing-only inference."""
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tok.pad_token = tok.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) if use_4bit else None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map=device,
        torch_dtype=torch.bfloat16 if not use_4bit else None,
        trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log.info("Loaded frozen model: %s", model_name)
    return model, tok


def _stream_chunks(
    dataset_name: str,
    split: str,
    text_col: str,
    max_chunks: int,
) -> Iterator[str]:
    """Yield raw text strings from a HuggingFace dataset."""
    ds = load_dataset(dataset_name, split=split, streaming=True)
    count = 0
    for sample in ds:
        text = sample.get(text_col) or sample.get("text") or sample.get("content", "")
        if isinstance(text, list):
            text = " ".join(str(t) for t in text)
        if text and isinstance(text, str):
            yield text.strip()
            count += 1
            if count >= max_chunks:
                break


@torch.inference_mode()
def route_batch(
    texts: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    router: RouterCapture,
    max_length: int = 256,
    device: str = "cuda",
) -> list[tuple[tuple[int, ...], list[float]]]:
    """Forward pass through the frozen model to capture routing decisions."""
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    _ = model(**enc)
    return router.flush()


def compute_purity(routing_probs: list[float], path: tuple[int, ...]) -> float:
    """Fraction of routing probability mass captured by the top-K experts."""
    return sum(routing_probs[i] for i in path)


def build_buckets(
    model_name: str,
    dataset_name: str,
    output_dir: Path,
    top_k: int = 2,
    purity_threshold: float = 0.0,
    max_chunks: int = 50_000,
    batch_size: int = 32,
    split: str = "train",
    text_col: str = "prompt",
    use_4bit: bool = True,
) -> dict[tuple[int, ...], ExpertBucket]:
    """
    Main pipeline entry point. Returns a dict of {expert_path: ExpertBucket}.
    Also serializes each bucket as a HuggingFace Arrow dataset.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tok = load_frozen_model(model_name, use_4bit=use_4bit, device="auto")
    router = RouterCapture(top_k=top_k)
    router.attach(model)

    buckets: dict[tuple[int, ...], ExpertBucket] = defaultdict(ExpertBucket)
    text_iter = _stream_chunks(dataset_name, split, text_col, max_chunks)

    batch: list[str] = []
    processed = 0

    def _flush_batch(b: list[str]) -> None:
        nonlocal processed
        decisions = route_batch(b, model, tok, router, device=device)
        for text, (path, probs) in zip(b, decisions):
            purity = compute_purity(probs, path)
            if purity < purity_threshold:
                continue
            if path not in buckets:
                buckets[path] = ExpertBucket(path=path)
            buckets[path].records.append(ChunkRecord(
                text=text,
                expert_path=path,
                purity_score=purity,
                routing_probs=probs,
            ))
        processed += len(b)
        if processed % 1000 == 0:
            log.info("Processed %d chunks, %d buckets", processed, len(buckets))

    for text in text_iter:
        batch.append(text)
        if len(batch) == batch_size:
            _flush_batch(batch)
            batch.clear()
    if batch:
        _flush_batch(batch)

    router.detach()
    log.info("Routing complete: %d total chunks → %d unique paths", processed, len(buckets))

    # Serialize buckets
    manifest = {}
    for path, bucket in buckets.items():
        ranked = bucket.ranked_sample()
        path_str = "_".join(map(str, path))
        ds = Dataset.from_dict({
            "text":         [r.text         for r in ranked],
            "purity_score": [r.purity_score for r in ranked],
            "expert_path":  [list(r.expert_path) for r in ranked],
        })
        bucket_dir = output_dir / f"path_{path_str}"
        ds.save_to_disk(str(bucket_dir))
        manifest[path_str] = {
            "path":           list(path),
            "num_chunks":     len(ranked),
            "mean_purity":    bucket.mean_purity,
            "dataset_dir":    str(bucket_dir),
        }
        log.info("  Bucket %s: %d chunks, mean_purity=%.3f",
                 path_str, len(ranked), bucket.mean_purity)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Manifest written to %s", manifest_path)
    return dict(buckets)


def load_bucket_manifest(output_dir: Path) -> dict:
    p = output_dir / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"No manifest found at {p}. Run build_buckets first.")
    return json.loads(p.read_text())


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MoE-aware data bucketing pipeline")
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--dataset_name",       required=True)
    p.add_argument("--output_dir",         default="./buckets")
    p.add_argument("--top_k",              type=int,   default=2)
    p.add_argument("--purity_threshold",   type=float, default=0.0,
                   help="Discard chunks below this purity (0 = keep all)")
    p.add_argument("--max_chunks",         type=int,   default=50_000)
    p.add_argument("--batch_size",         type=int,   default=32)
    p.add_argument("--split",              default="train")
    p.add_argument("--text_col",           default="prompt")
    p.add_argument("--no_4bit",            action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    build_buckets(
        model_name=args.model_name_or_path,
        dataset_name=args.dataset_name,
        output_dir=Path(args.output_dir),
        top_k=args.top_k,
        purity_threshold=args.purity_threshold,
        max_chunks=args.max_chunks,
        batch_size=args.batch_size,
        split=args.split,
        text_col=args.text_col,
        use_4bit=not args.no_4bit,
    )
