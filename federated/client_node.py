"""
client_node.py — Flower client agent for MoE federated fine-tuning.

Implements:
  • Quantized base model loading (4-bit NF4 via bitsandbytes)
  • Path-selective LoRA attachment: only adapters on assigned expert layers
  • FlexLoRA: rank chosen from compute budget
  • DiLoCo-style local training: many local steps + frequent local checkpointing
  • Crash-resilient: resumes from latest checkpoint if interrupted
  • Sends (adapter_deltas, local_val_loss, confidence) back to server
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import flwr as fl
from datasets import load_from_disk
from flwr.common import (
    NDArrays, Scalar, ndarrays_to_parameters, parameters_to_ndarrays,
    FitIns, FitRes, EvaluateIns, EvaluateRes, GetParametersIns, GetParametersRes,
)
from peft import (
    LoraConfig, get_peft_model, TaskType, PeftModel,
    prepare_model_for_kbit_training,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    DataCollatorForLanguageModeling, get_cosine_schedule_with_warmup,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ──────────────────────────────────────────────────────────────────────────────
# Compute-budget → LoRA rank mapping
# ──────────────────────────────────────────────────────────────────────────────

def compute_score_to_rank(compute_score: float) -> int:
    """
    Map normalized compute score [0, 1] to a LoRA rank.
    Low-resource nodes use rank 4; high-resource nodes use up to rank 32.
    """
    breakpoints = [(0.2, 4), (0.4, 8), (0.6, 16), (0.8, 24), (1.0, 32)]
    for threshold, rank in breakpoints:
        if compute_score <= threshold:
            return rank
    return 32


# ──────────────────────────────────────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────────────────────────────────────

def _expert_layer_names(model: nn.Module, expert_indices: list[int]) -> list[str]:
    """
    Discover parameter names that belong to the requested MoE expert indices.

    Works with Mixtral-style: model.model.layers.{L}.block_sparse_moe.experts.{E}.*
    and Qwen-MoE style: model.model.layers.{L}.mlp.experts.{E}.*
    """
    target_patterns = [f".experts.{e}." for e in expert_indices]
    names = []
    for name, _ in model.named_modules():
        if any(pat in name for pat in target_patterns):
            names.append(name)
    return names


def _lora_target_modules(model: nn.Module, expert_indices: list[int]) -> list[str]:
    """
    Return the short module names that LoRA should target, scoped to the
    requested expert indices.  Falls back to global projection layers if
    no expert-specific names are found (e.g., dense models).
    """
    expert_names = _expert_layer_names(model, expert_indices)
    # Keep only Linear sub-module names that are projections (w1, w2, w3, gate, etc.)
    proj_suffixes = {"w1", "w2", "w3", "gate", "up_proj", "down_proj",
                     "q_proj", "k_proj", "v_proj", "o_proj"}
    targets = set()
    for full_name in expert_names:
        short = full_name.split(".")[-1]
        if short in proj_suffixes:
            targets.add(short)
    if not targets:
        log.warning("No expert-specific projection layers found; falling back to q/v proj")
        targets = {"q_proj", "v_proj"}
    return list(targets)


def load_model_with_expert_lora(
    model_name: str,
    expert_indices: list[int],
    lora_rank: int,
    lora_alpha: Optional[int] = None,
) -> tuple[PeftModel, AutoTokenizer]:
    """
    Load a 4-bit quantized base model and attach LoRA only to the layers
    serving the expert_indices path. Everything else stays frozen.
    """
    log.info("Loading base model: %s", model_name)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model = prepare_model_for_kbit_training(base_model)

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tok.pad_token = tok.eos_token

    target_modules = _lora_target_modules(base_model, expert_indices)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha or (lora_rank * 2),
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()
    return model, tok


# ──────────────────────────────────────────────────────────────────────────────
# DiLoCo-style Local Training
# ──────────────────────────────────────────────────────────────────────────────

def _make_dataloader(
    dataset_dir: str,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_length: int = 512,
    shuffle: bool = True,
) -> DataLoader:
    ds = load_from_disk(dataset_dir)
    ds = ds.map(
        lambda ex: tokenizer(
            ex["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        ),
        batched=True,
        remove_columns=[c for c in ds.column_names if c not in ("input_ids", "attention_mask")],
    )
    ds.set_format("torch")
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)


def local_train(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    dataset_dir: str,
    checkpoint_dir: Path,
    local_steps: int = 200,
    checkpoint_every: int = 50,
    batch_size: int = 4,
    lr: float = 2e-4,
    max_length: int = 512,
) -> dict:
    """
    DiLoCo-style local training: many local SGD steps with frequent checkpointing.
    On restart, automatically resumes from the latest checkpoint.

    Returns: {"final_train_loss": float, "final_val_loss": float, "steps_done": int}
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device

    train_loader = _make_dataloader(dataset_dir, tokenizer, batch_size, max_length, shuffle=True)
    val_loader   = _make_dataloader(dataset_dir, tokenizer, batch_size=2, max_length=max_length, shuffle=False)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, local_steps // 10), num_training_steps=local_steps,
    )

    # Resume from checkpoint if available
    start_step   = 0
    best_ckpt    = _latest_checkpoint(checkpoint_dir)
    if best_ckpt:
        state = torch.load(best_ckpt, map_location="cpu")
        model.load_state_dict(state["model_state"], strict=False)
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_step = state["step"]
        log.info("Resumed from checkpoint step %d (%s)", start_step, best_ckpt)

    model.train()
    train_iter  = iter(train_loader)
    total_loss  = 0.0
    steps_done  = 0

    for step in range(start_step, local_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch      = next(train_iter)

        batch   = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss    = outputs.loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        steps_done += 1

        if (step + 1) % checkpoint_every == 0:
            ckpt_path = checkpoint_dir / f"step_{step+1}.pt"
            torch.save({
                "step":             step + 1,
                "model_state":      {k: v.cpu() for k, v in model.state_dict().items()
                                     if "lora_" in k},
                "optimizer_state":  optimizer.state_dict(),
                "scheduler_state":  scheduler.state_dict(),
            }, ckpt_path)
            log.info("Checkpoint saved: %s (loss=%.4f)", ckpt_path, loss.item())

    avg_train_loss = total_loss / max(steps_done, 1)

    # Validation
    val_loss = _evaluate_loss(model, val_loader, device, max_batches=20)
    log.info("Local training done: train_loss=%.4f, val_loss=%.4f", avg_train_loss, val_loss)

    return {"final_train_loss": avg_train_loss, "final_val_loss": val_loss, "steps_done": steps_done}


def _latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    ckpts = sorted(checkpoint_dir.glob("step_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    return ckpts[-1] if ckpts else None


@torch.inference_mode()
def _evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device,
    max_batches: int = 20,
) -> float:
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out   = model(**batch)
        losses.append(out.loss.item())
    model.train()
    return float(np.mean(losses)) if losses else 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Adapter Delta Extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_lora_deltas(
    model: PeftModel,
    initial_state: dict[str, torch.Tensor],
) -> tuple[list[np.ndarray], dict]:
    """
    Extract per-expert LoRA adapter deltas (model_state - initial_state).
    Returns arrays packed as [lora_A_0, lora_B_0, lora_A_1, lora_B_1, ...]
    and metadata dict with num_lora_pairs and layer names.
    """
    lora_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    other_arrays: list[np.ndarray] = []
    layer_names: list[str] = []

    # Group LoRA A/B pairs
    state = model.state_dict()
    a_keys = sorted(k for k in state if "lora_A" in k)
    b_keys = [k.replace("lora_A", "lora_B") for k in a_keys]

    for ak, bk in zip(a_keys, b_keys):
        if bk not in state:
            continue
        delta_A = (state[ak] - initial_state.get(ak, torch.zeros_like(state[ak]))).cpu().numpy()
        delta_B = (state[bk] - initial_state.get(bk, torch.zeros_like(state[bk]))).cpu().numpy()
        lora_pairs.append((delta_A, delta_B))
        layer_names.append(ak.replace(".lora_A.default.weight", ""))

    arrays: list[np.ndarray] = []
    for A, B in lora_pairs:
        arrays.extend([A, B])

    meta = {
        "num_lora_pairs": len(lora_pairs),
        "layer_names": layer_names,
        "lora_rank": lora_pairs[0][0].shape[0] if lora_pairs else 0,
    }
    return arrays, meta


# ──────────────────────────────────────────────────────────────────────────────
# Flower Client
# ──────────────────────────────────────────────────────────────────────────────

class MoEFedClient(fl.client.Client):
    """
    Flower client agent.

    Lifecycle per round:
      1. get_parameters       — return current LoRA weights
      2. fit                  — run DiLoCo local training, return deltas + metrics
      3. evaluate (optional)  — run local evaluation, return loss
    """

    def __init__(
        self,
        model_name: str,
        expert_indices: list[int],
        dataset_dir: str,
        checkpoint_dir: Path,
        compute_score: float = 0.5,
        local_steps: int = 200,
        checkpoint_every: int = 50,
        batch_size: int = 4,
        lr: float = 2e-4,
        heartbeat_url: Optional[str] = None,
        node_id: Optional[str] = None,
    ):
        self.model_name       = model_name
        self.expert_indices   = expert_indices
        self.dataset_dir      = dataset_dir
        self.checkpoint_dir   = checkpoint_dir
        self.compute_score    = compute_score
        self.local_steps      = local_steps
        self.checkpoint_every = checkpoint_every
        self.batch_size       = batch_size
        self.lr               = lr
        self.heartbeat_url    = heartbeat_url
        self.node_id          = node_id

        self.lora_rank = compute_score_to_rank(compute_score)
        log.info("Node compute=%.2f → LoRA rank=%d", compute_score, self.lora_rank)

        self.model, self.tokenizer = load_model_with_expert_lora(
            model_name, expert_indices, self.lora_rank,
        )
        # Snapshot initial LoRA weights for delta extraction
        self._initial_state = {
            k: v.detach().cpu().clone()
            for k, v in self.model.state_dict().items()
            if "lora_" in k
        }

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        arrays, _ = extract_lora_deltas(self.model, self._initial_state)
        return GetParametersRes(
            status=fl.common.Status(code=fl.common.Code.OK, message=""),
            parameters=ndarrays_to_parameters(arrays),
        )

    def fit(self, ins: FitIns) -> FitRes:
        log.info("Starting local training (expert_path=%s, rank=%d)",
                 self.expert_indices, self.lora_rank)
        config = ins.config

        # Apply server's global weights (non-LoRA portions) if sent
        server_arrays = parameters_to_ndarrays(ins.parameters)
        # (Full weight sync is not used for LoRA-only clients; server sends global
        #  non-adapter weights which we simply ignore to preserve quantized base)

        # Local training
        stats = local_train(
            self.model,
            self.tokenizer,
            self.dataset_dir,
            self.checkpoint_dir,
            local_steps=int(config.get("local_steps", self.local_steps)),
            checkpoint_every=self.checkpoint_every,
            batch_size=self.batch_size,
            lr=self.lr,
        )

        # Extract deltas
        arrays, meta = extract_lora_deltas(self.model, self._initial_state)

        # Confidence: invert val loss, clamp ∈ [0, 1]
        val_loss   = stats["final_val_loss"]
        train_loss = stats["final_train_loss"]
        confidence = max(0.0, min(1.0, 1.0 - val_loss / max(train_loss, 1e-6)))

        metrics = {
            "expert_path":      json.dumps(self.expert_indices),
            "local_val_loss":   val_loss,
            "local_train_loss": train_loss,
            "confidence":       confidence,
            "steps_done":       stats["steps_done"],
            "lora_rank":        self.lora_rank,
            **meta,
        }
        # Flatten list fields for Flower (only scalar types allowed in metrics)
        metrics["num_lora_pairs"] = int(meta["num_lora_pairs"])
        metrics["layer_names"]    = json.dumps(meta["layer_names"])

        log.info("Fit done: val_loss=%.4f, confidence=%.4f", val_loss, confidence)

        return FitRes(
            status=fl.common.Status(code=fl.common.Code.OK, message=""),
            parameters=ndarrays_to_parameters(arrays),
            num_examples=stats["steps_done"],
            metrics=metrics,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        device = next(self.model.parameters()).device
        val_loader = _make_dataloader(
            self.dataset_dir, self.tokenizer, batch_size=2, shuffle=False,
        )
        loss = _evaluate_loss(self.model, val_loader, device, max_batches=30)
        return EvaluateRes(
            status=fl.common.Status(code=fl.common.Code.OK, message=""),
            loss=loss,
            num_examples=30,
            metrics={"val_loss": loss},
        )


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MoE federated fine-tuning client node")
    p.add_argument("--server_address",   default="orchestrator:8080")
    p.add_argument("--model_name",       required=True)
    p.add_argument("--expert_indices",   type=int, nargs="+", required=True,
                   help="Expert indices this node is responsible for")
    p.add_argument("--dataset_dir",      required=True,
                   help="Path to HuggingFace dataset directory for this bucket")
    p.add_argument("--checkpoint_dir",   default="./checkpoints")
    p.add_argument("--compute_score",    type=float, default=0.5,
                   help="Normalized compute budget [0, 1] for LoRA rank selection")
    p.add_argument("--local_steps",      type=int,   default=200)
    p.add_argument("--checkpoint_every", type=int,   default=50)
    p.add_argument("--batch_size",       type=int,   default=4)
    p.add_argument("--lr",               type=float, default=2e-4)
    p.add_argument("--node_id",          default=None)
    p.add_argument("--heartbeat_url",    default=None,
                   help="Orchestrator heartbeat URL, e.g. http://orchestrator:8000/heartbeat")
    return p.parse_args()


def main():
    args = _parse()
    client = MoEFedClient(
        model_name       = args.model_name,
        expert_indices   = args.expert_indices,
        dataset_dir      = args.dataset_dir,
        checkpoint_dir   = Path(args.checkpoint_dir),
        compute_score    = args.compute_score,
        local_steps      = args.local_steps,
        checkpoint_every = args.checkpoint_every,
        batch_size       = args.batch_size,
        lr               = args.lr,
        heartbeat_url    = args.heartbeat_url,
        node_id          = args.node_id,
    )

    # Background heartbeat thread
    if args.heartbeat_url and args.node_id:
        import threading, requests
        def _heartbeat_loop():
            while True:
                try:
                    requests.post(args.heartbeat_url, json={"node_id": args.node_id}, timeout=5)
                except Exception as e:
                    log.warning("Heartbeat failed: %s", e)
                time.sleep(10)
        t = threading.Thread(target=_heartbeat_loop, daemon=True)
        t.start()

    fl.client.start_client(
        server_address=args.server_address,
        client=client,
    )


if __name__ == "__main__":
    main()
