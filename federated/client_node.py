"""Flower client for federated MoE LoRA fine-tuning.

Fixes compared with the original prototype:
- Checkpoints are scoped to a federation round, so a completed round is never
  accidentally resumed as if it were the next round.
- Server global LoRA parameters are actually applied before local training.
- The LoRA baseline is refreshed at the beginning of every federation round.
- Variable client ranks are supported by reconstructing the server's canonical
  full-rank delta and projecting it to the client's rank.
- Delta extraction is relative to the beginning-of-round global state.
- Empty-training rounds cannot silently report train_loss=0.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import flwr as fl
from datasets import load_from_disk
from flwr.common import (
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    GetParametersIns,
    GetParametersRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def compute_score_to_rank(compute_score: float) -> int:
    breakpoints = [(0.2, 4), (0.4, 8), (0.6, 16), (0.8, 24), (1.0, 32)]
    for threshold, rank in breakpoints:
        if compute_score <= threshold:
            return rank
    return 32


def _maybe_register_toy_moe(model_name: str) -> None:
    config_path = Path(model_name) / "config.json"
    if not config_path.exists():
        return

    try:
        cfg_data = json.loads(config_path.read_text())
    except Exception:
        return

    if cfg_data.get("model_type") == "toy_moe":
        dev_dir = Path(__file__).parent / "dev"
        if str(dev_dir) not in sys.path:
            sys.path.insert(0, str(dev_dir))
        from toy_moe import register_toy_moe
        register_toy_moe()
        log.info("Registered ToyMoE custom AutoClasses")


def _expert_layer_names(model: nn.Module, expert_indices: list[int]) -> list[str]:
    patterns = [f".experts.{e}." for e in expert_indices]
    return [
        name
        for name, _ in model.named_modules()
        if any(pattern in name for pattern in patterns)
    ]


def _lora_target_modules(model: nn.Module, expert_indices: list[int]) -> list[str]:
    expert_names = _expert_layer_names(model, expert_indices)
    proj_suffixes = {
        "w1", "w2", "w3", "gate",
        "up_proj", "down_proj",
        "q_proj", "k_proj", "v_proj", "o_proj",
    }

    # IMPORTANT: return full module paths, not only suffixes such as ``w1``.
    # PEFT treats a short suffix as a global match, which would attach LoRA to
    # every expert. Full paths keep the adapter restricted to this node's experts.
    targets = {
        name
        for name in expert_names
        if name.split(".")[-1] in proj_suffixes
    }

    if not targets:
        log.warning("No expert-specific projection layers found; falling back to q_proj/v_proj")
        targets = {"q_proj", "v_proj"}

    log.info("LoRA target modules: %s", sorted(targets))
    return sorted(targets)


def load_model_with_expert_lora(
    model_name: str,
    expert_indices: list[int],
    lora_rank: int,
    lora_alpha: Optional[int] = None,
) -> tuple[PeftModel, AutoTokenizer]:
    use_4bit = (
        torch.cuda.is_available()
        and os.getenv("NO_4BIT", "").lower() != "true"
    )

    log.info("Loading base model: %s (4bit=%s)", model_name, use_4bit)
    _maybe_register_toy_moe(model_name)

    bnb_cfg = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        if use_4bit
        else None
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto" if use_4bit else "cpu",
        torch_dtype=None if use_4bit else torch.float32,
        trust_remote_code=True,
    )

    if use_4bit:
        base_model = prepare_model_for_kbit_training(base_model)

    tok = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
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
        remove_columns=[
            c for c in ds.column_names
            if c not in ("input_ids", "attention_mask")
        ],
    )

    ds.set_format("torch")
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
    )


def _checkpoint_path(
    checkpoint_dir: Path,
    session_id: str,
    round_id: int,
    step: int,
) -> Path:
    return (
        checkpoint_dir
        / session_id
        / f"round_{round_id}"
        / f"step_{step}.pt"
    )


def _latest_checkpoint(
    checkpoint_dir: Path,
    session_id: str,
    round_id: int,
) -> Optional[Path]:
    round_dir = checkpoint_dir / session_id / f"round_{round_id}"
    if not round_dir.exists():
        return None

    ckpts = list(round_dir.glob("step_*.pt"))
    if not ckpts:
        return None

    return max(ckpts, key=lambda p: int(p.stem.split("_")[1]))


def local_train(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    dataset_dir: str,
    checkpoint_dir: Path,
    round_id: int,
    session_id: str = "default",
    local_steps: int = 200,
    checkpoint_every: int = 50,
    batch_size: int = 4,
    lr: float = 2e-4,
    max_length: int = 512,
) -> dict:
    """Run local SGD and resume only within the same federation round."""

    if local_steps <= 0:
        raise ValueError("local_steps must be > 0")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / session_id / f"round_{round_id}").mkdir(
        parents=True, exist_ok=True
    )

    device = next(model.parameters()).device

    train_loader = _make_dataloader(
        dataset_dir, tokenizer, batch_size, max_length, shuffle=True
    )
    val_loader = _make_dataloader(
        dataset_dir, tokenizer, batch_size=2, max_length=max_length, shuffle=False
    )

    # Measure the model before this round. Confidence must compare the
    # post-training validation loss against the loss at the start of the
    # round, not against training loss.
    initial_val_loss = _evaluate_loss(model, val_loader, device, max_batches=20)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, local_steps // 10),
        num_training_steps=local_steps,
    )

    start_step = 0
    latest = _latest_checkpoint(checkpoint_dir, session_id, round_id)

    if latest is not None:
        state = torch.load(latest, map_location="cpu")
        model.load_state_dict(state["model_state"], strict=False)
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_step = int(state["step"])
        log.info(
            "Resumed round %d from checkpoint step %d (%s)",
            round_id, start_step, latest,
        )

    if start_step >= local_steps:
        # A completed round must not be reported as a new zero-step round.
        # This normally means the same round was restarted after completion.
        log.info(
            "Round %d already reached %d/%d steps; evaluating checkpoint.",
            round_id, start_step, local_steps,
        )
        val_loss = _evaluate_loss(model, val_loader, device, max_batches=20)
        return {
            "initial_val_loss": float(initial_val_loss),
            "final_train_loss": float("nan"),
            "final_val_loss": val_loss,
            "steps_done": 0,
            "resumed_completed": True,
        }

    model.train()
    train_iter = iter(train_loader)
    total_loss = 0.0
    steps_done = 0

    for step in range(start_step, local_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs.loss

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at step {step + 1}: {loss.item()}")

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += float(loss.item())
        steps_done += 1

        if (step + 1) % checkpoint_every == 0 or (step + 1) == local_steps:
            ckpt_path = _checkpoint_path(
                checkpoint_dir, session_id, round_id, step + 1
            )
            torch.save(
                {
                    "round_id": round_id,
                    "step": step + 1,
                    "model_state": {
                        k: v.detach().cpu()
                        for k, v in model.state_dict().items()
                        if "lora_" in k
                    },
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                },
                ckpt_path,
            )
            log.info(
                "Checkpoint saved: %s (loss=%.4f)",
                ckpt_path, loss.item(),
            )

    if steps_done == 0:
        raise RuntimeError(
            f"Round {round_id} performed zero local training steps. "
            "This should never be silently treated as successful training."
        )

    avg_train_loss = total_loss / steps_done
    val_loss = _evaluate_loss(model, val_loader, device, max_batches=20)

    log.info(
        "Round %d local training done: train_loss=%.4f, val_loss=%.4f, steps=%d",
        round_id, avg_train_loss, val_loss, steps_done,
    )

    return {
        "initial_val_loss": float(initial_val_loss),
        "final_train_loss": avg_train_loss,
        "final_val_loss": val_loss,
        "steps_done": steps_done,
        "resumed_completed": False,
    }


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
        out = model(**batch)
        losses.append(float(out.loss.item()))

    model.train()

    return float(np.mean(losses)) if losses else float("inf")


def _lora_keys(model: PeftModel) -> list[tuple[str, str]]:
    state = model.state_dict()
    a_keys = sorted(k for k in state if "lora_A" in k)

    pairs = []
    for ak in a_keys:
        bk = ak.replace("lora_A", "lora_B")
        if bk in state:
            pairs.append((ak, bk))

    return pairs


def extract_lora_deltas(
    model: PeftModel,
    initial_state: dict[str, torch.Tensor],
) -> tuple[list[np.ndarray], dict]:
    """
    Extract an exact effective LoRA weight delta while retaining low-rank factors.

    For each layer we send [A_before, B_before, A_after, B_after]. The server
    reconstructs the true update as:

        delta_W = B_after @ A_after - B_before @ A_before

    Using delta_A @ delta_B would be mathematically incorrect because LoRA
    factors are not additive in weight space.
    """
    state = model.state_dict()
    pairs = _lora_keys(model)

    arrays: list[np.ndarray] = []
    layer_names: list[str] = []

    for ak, bk in pairs:
        A_before = initial_state.get(ak, torch.zeros_like(state[ak]))
        B_before = initial_state.get(bk, torch.zeros_like(state[bk]))

        arrays.extend([
            A_before.detach().cpu().numpy().astype(np.float32),
            B_before.detach().cpu().numpy().astype(np.float32),
            state[ak].detach().cpu().numpy().astype(np.float32),
            state[bk].detach().cpu().numpy().astype(np.float32),
        ])
        layer_names.append(ak.replace(".lora_A.default.weight", ""))

    rank = int(arrays[0].shape[0]) if arrays else 0

    return arrays, {
        "num_lora_pairs": len(pairs),
        "layer_names": layer_names,
        "lora_rank": rank,
        "payload_format": "before_after_lora_factors",
    }


def extract_current_lora(model: PeftModel) -> tuple[list[np.ndarray], dict]:
    state = model.state_dict()
    pairs = _lora_keys(model)

    arrays = []
    layer_names = []

    for ak, bk in pairs:
        arrays.extend([
            state[ak].detach().cpu().numpy(),
            state[bk].detach().cpu().numpy(),
        ])
        layer_names.append(ak.replace(".lora_A.default.weight", ""))

    rank = int(arrays[0].shape[0]) if arrays else 0

    return arrays, {
        "num_lora_pairs": len(pairs),
        "layer_names": layer_names,
        "lora_rank": rank,
    }


def _project_delta_to_rank(
    A_global: np.ndarray,
    B_global: np.ndarray,
    target_rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a canonical B@A delta into target-rank LoRA factors."""

    if target_rank <= 0:
        raise ValueError("target_rank must be positive")

    delta = B_global.astype(np.float32) @ A_global.astype(np.float32)

    U, S, Vh = np.linalg.svd(delta, full_matrices=False)
    r = min(target_rank, len(S))

    U_r = U[:, :r]
    S_r = np.maximum(S[:r], 0.0)
    Vh_r = Vh[:r, :]

    sqrt_s = np.sqrt(S_r)

    B = U_r * sqrt_s[None, :]
    A = sqrt_s[:, None] * Vh_r

    if r < target_rank:
        A_pad = np.zeros(
            (target_rank, A.shape[1]),
            dtype=A.dtype,
        )
        B_pad = np.zeros(
            (B.shape[0], target_rank),
            dtype=B.dtype,
        )
        A_pad[:r] = A
        B_pad[:, :r] = B
        A, B = A_pad, B_pad

    return A.astype(np.float32), B.astype(np.float32)


def apply_global_lora_parameters(
    model: PeftModel,
    global_arrays: list[np.ndarray],
    global_meta: dict,
) -> None:
    """Apply only the global LoRA layers that exist on this client's path."""

    if not global_arrays:
        return

    global_rank = int(global_meta.get("global_rank", 0))
    global_layers = list(global_meta.get("layer_names", []) or [])

    if global_rank <= 0:
        raise ValueError("Server returned invalid global_rank")
    if len(global_arrays) % 2 != 0:
        raise ValueError("Global parameter payload must contain A/B pairs")

    local_pairs = _lora_keys(model)

    # During Flower initialization, the strategy has not received any fit
    # result yet, so it cannot know the layer names. The initial parameters
    # were obtained from one client and therefore follow that client's local
    # LoRA ordering. Infer the names from our local ordering for round 1.
    #
    # From round 2 onward the strategy supplies explicit layer_names.
    if not global_layers:
        pair_count = len(global_arrays) // 2

        if pair_count > len(local_pairs):
            raise ValueError(
                "Initial global LoRA payload contains more layers than "
                f"this client: payload_pairs={pair_count}, "
                f"local_pairs={len(local_pairs)}"
            )

        global_layers = [
            ak.replace(".lora_A.default.weight", "")
            for ak, _ in local_pairs[:pair_count]
        ]

        log.info(
            "Initial global LoRA metadata has no layer names; "
            "mapped %d parameter pairs using local ordering",
            pair_count,
        )

    if len(global_arrays) // 2 != len(global_layers):
        raise ValueError(
            f"Global LoRA metadata mismatch: arrays={len(global_arrays)}, "
            f"layers={len(global_layers)}"
        )
    local_state = model.state_dict()
    local_names = {
        ak.replace(".lora_A.default.weight", ""): (ak, bk)
        for ak, bk in local_pairs
    }

    applied = 0
    skipped_zero = 0
    with torch.no_grad():
        for i, global_name in enumerate(global_layers):
            if global_name not in local_names:
                # This global layer belongs to another expert/path.
                continue

            ak, bk = local_names[global_name]
            A_global = np.asarray(global_arrays[2 * i], dtype=np.float32)
            B_global = np.asarray(global_arrays[2 * i + 1], dtype=np.float32)

            # A numerically-zero global delta (e.g. the server's bootstrap
            # "zero effective LoRA" sent before any round has trained
            # anything) must NOT be applied by overwriting this layer's A/B.
            # SVD of a zero delta necessarily reconstructs A=0 AND B=0. That
            # differs from PEFT's normal init (A=random, B=0) in a way that
            # matters: with output = B @ (A @ x), d(loss)/dA is proportional
            # to B and d(loss)/dB is proportional to (A @ x). If BOTH A and B
            # are zero, BOTH gradients are exactly zero on every step,
            # forever - the layer can never start learning. Leaving the
            # client's own (already correctly initialized) LoRA weights
            # untouched is the correct behavior when there is genuinely
            # nothing trained yet to apply.
            if not np.any(A_global) and not np.any(B_global):
                skipped_zero += 1
                continue

            A_local, B_local = _project_delta_to_rank(
                A_global,
                B_global,
                int(model.peft_config["default"].r),
            )

            local_state[ak].copy_(
                torch.from_numpy(A_local).to(
                    device=local_state[ak].device,
                    dtype=local_state[ak].dtype,
                )
            )
            local_state[bk].copy_(
                torch.from_numpy(B_local).to(
                    device=local_state[bk].device,
                    dtype=local_state[bk].dtype,
                )
            )
            applied += 1

    log.info(
        "Applied global LoRA: canonical_rank=%d -> local_rank=%d (%d/%d layers, %d skipped as zero-delta)",
        global_rank,
        int(model.peft_config["default"].r),
        applied,
        len(local_pairs),
        skipped_zero,
    )


class MoEFedClient(fl.client.Client):
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
        self.model_name = model_name
        self.expert_indices = expert_indices
        self.dataset_dir = dataset_dir
        self.checkpoint_dir = checkpoint_dir
        self.compute_score = compute_score
        self.local_steps = local_steps
        self.checkpoint_every = checkpoint_every
        self.batch_size = batch_size
        self.lr = lr
        self.heartbeat_url = heartbeat_url
        self.node_id = node_id

        self.lora_rank = compute_score_to_rank(compute_score)
        log.info(
            "Node compute=%.2f -> LoRA rank=%d",
            compute_score, self.lora_rank,
        )

        self.model, self.tokenizer = load_model_with_expert_lora(
            model_name,
            expert_indices,
            self.lora_rank,
        )

        self._round_id = 0
        self._initial_state = self._snapshot_lora_state()

    def _snapshot_lora_state(self) -> dict[str, torch.Tensor]:
        return {
            k: v.detach().cpu().clone()
            for k, v in self.model.state_dict().items()
            if "lora_" in k
        }

    def _set_round_baseline(self) -> None:
        self._initial_state = self._snapshot_lora_state()

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        # Flower asks one client for the initial global parameters. The local
        # PEFT adapter contains random A weights by design, but those random
        # factors do NOT represent a useful global update. Start federation
        # from a zero effective LoRA delta instead.
        arrays, _ = extract_current_lora(self.model)
        zero_arrays = [np.zeros_like(arr, dtype=np.float32) for arr in arrays]

        return GetParametersRes(
            status=fl.common.Status(
                code=fl.common.Code.OK,
                message="",
            ),
            parameters=ndarrays_to_parameters(zero_arrays),
        )

    def fit(self, ins: FitIns) -> FitRes:
        config = ins.config
        self._round_id = int(config.get("server_round", self._round_id + 1))

        log.info(
            "Starting local training: session=%s round=%d path=%s rank=%d",
            config.get("session_id", "unknown"),
            self._round_id,
            self.expert_indices,
            self.lora_rank,
        )

        # 1. Receive and apply the current global LoRA model.
        global_arrays = parameters_to_ndarrays(ins.parameters)

        global_meta_raw = config.get("global_meta", "{}")
        if isinstance(global_meta_raw, str):
            global_meta = json.loads(global_meta_raw)
        else:
            global_meta = dict(global_meta_raw or {})

        if global_arrays:
            apply_global_lora_parameters(
                self.model,
                global_arrays,
                global_meta,
            )

        # 2. Snapshot the GLOBAL state. The returned delta is local - global.
        self._set_round_baseline()

        # 3. Train NEW local steps for this round.
        stats = local_train(
            self.model,
            self.tokenizer,
            self.dataset_dir,
            self.checkpoint_dir,
            round_id=self._round_id,
            session_id=str(config.get("session_id", "default")),
            local_steps=int(config.get("local_steps", self.local_steps)),
            checkpoint_every=self.checkpoint_every,
            batch_size=self.batch_size,
            lr=self.lr,
        )

        if stats["steps_done"] <= 0:
            raise RuntimeError(
                f"Round {self._round_id} returned zero training steps"
            )

        arrays, meta = extract_lora_deltas(
            self.model,
            self._initial_state,
        )

        val_loss = float(stats["final_val_loss"])
        train_loss = float(stats["final_train_loss"])
        initial_val_loss = float(stats.get("initial_val_loss", val_loss))

        # Confidence measures actual validation improvement during THIS
        # round. Comparing validation loss to training loss is misleading
        # because they are different estimates.
        if (
            not np.isfinite(initial_val_loss)
            or initial_val_loss <= 0
            or not np.isfinite(val_loss)
        ):
            confidence = 0.0
        else:
            confidence = (initial_val_loss - val_loss) / initial_val_loss
            confidence = float(np.clip(confidence, 0.0, 1.0))

        metrics = {
            "expert_path": json.dumps(self.expert_indices),
            "local_val_loss": val_loss,
            "local_train_loss": train_loss,
            "initial_val_loss": initial_val_loss,
            "confidence": confidence,
            "steps_done": int(stats["steps_done"]),
            "lora_rank": int(self.lora_rank),
            "num_lora_pairs": int(meta["num_lora_pairs"]),
            "layer_names": json.dumps(meta["layer_names"]),
            "round_id": int(self._round_id),
            "payload_format": "before_after_lora_factors",
        }

        log.info(
            "Fit done: round=%d train_loss=%.4f val_loss=%.4f confidence=%.4f",
            self._round_id,
            train_loss,
            val_loss,
            confidence,
        )

        return FitRes(
            status=fl.common.Status(
                code=fl.common.Code.OK,
                message="",
            ),
            parameters=ndarrays_to_parameters(arrays),
            num_examples=int(stats["steps_done"]),
            metrics=metrics,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        device = next(self.model.parameters()).device
        val_loader = _make_dataloader(
            self.dataset_dir,
            self.tokenizer,
            batch_size=2,
            shuffle=False,
        )
        loss = _evaluate_loss(
            self.model,
            val_loader,
            device,
            max_batches=30,
        )

        return EvaluateRes(
            status=fl.common.Status(
                code=fl.common.Code.OK,
                message="",
            ),
            loss=loss,
            num_examples=30,
            metrics={"val_loss": loss},
        )


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MoE federated fine-tuning client node"
    )

    p.add_argument("--server_address", default="orchestrator:8080")
    p.add_argument("--model_name", required=True)
    p.add_argument(
        "--expert_indices",
        type=int,
        nargs="+",
        required=True,
    )
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--checkpoint_dir", default="./checkpoints")
    p.add_argument("--compute_score", type=float, default=0.5)
    p.add_argument("--local_steps", type=int, default=200)
    p.add_argument("--checkpoint_every", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--node_id", default=None)
    p.add_argument("--heartbeat_url", default=None)

    return p.parse_args()


def main() -> None:
    args = _parse()

    client = MoEFedClient(
        model_name=args.model_name,
        expert_indices=args.expert_indices,
        dataset_dir=args.dataset_dir,
        checkpoint_dir=Path(args.checkpoint_dir),
        compute_score=args.compute_score,
        local_steps=args.local_steps,
        checkpoint_every=args.checkpoint_every,
        batch_size=args.batch_size,
        lr=args.lr,
        heartbeat_url=args.heartbeat_url,
        node_id=args.node_id,
    )

    if args.heartbeat_url and args.node_id:
        import threading
        import requests

        def _heartbeat_loop() -> None:
            while True:
                try:
                    requests.post(
                        args.heartbeat_url,
                        json={"node_id": args.node_id},
                        timeout=5,
                    )
                except Exception as exc:
                    log.warning("Heartbeat failed: %s", exc)

                time.sleep(10)

        threading.Thread(
            target=_heartbeat_loop,
            daemon=True,
        ).start()

    # Kept compatible with the Flower version used by the existing prototype.
    fl.client.start_client(
        server_address=args.server_address,
        client=client,
    )


if __name__ == "__main__":
    main()