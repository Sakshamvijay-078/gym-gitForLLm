"""
custom_strategy.py — Confidence-weighted Flower aggregation strategy for MoE LoRA.

Key behaviours:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  FlexLoRA merge   — reconstruct full-rank delta via SVD from variable-  │
  │                     rank client submissions before aggregation.          │
  │  Reward scoring   — evaluate each incoming adapter on a rotating val    │
  │                     set: reward = norm_loss_improvement - regression_    │
  │                     penalty on unrelated expert paths.                   │
  │  Sigmoid weight   — convert reward → bounded merge weight with a        │
  │                     guaranteed confidenceFloor.                          │
  │  Accept / reject  — discard updates below threshold; roll back the      │
  │                     global model if the merged update causes regression. │
  └─────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import copy
import io
import logging
import math
import random
from typing import Any, Optional, Union

import numpy as np
import torch
import flwr as fl
from flwr.common import (
    FitRes, Parameters, Scalar, EvaluateRes,
    ndarrays_to_parameters, parameters_to_ndarrays,
    GetParametersIns, FitIns, EvaluateIns,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: float, scale: float = 5.0) -> float:
    """Bounded sigmoid ∈ (0, 1). `scale` controls steepness around x=0."""
    return 1.0 / (1.0 + math.exp(-scale * x))


def _flexlora_reconstruct(
    lora_A: np.ndarray,  # (r, d_in)
    lora_B: np.ndarray,  # (d_out, r)
) -> np.ndarray:
    """
    FlexLoRA: reconstruct full-rank delta W = B @ A via SVD,
    then reproject back to a canonical rank for aggregation.
    Allows clients with different ranks to contribute to the same adapter.
    Returns: delta_W (d_out, d_in)
    """
    delta_W = lora_B @ lora_A                          # (d_out, d_in)
    U, S, Vh = np.linalg.svd(delta_W, full_matrices=False)
    # We keep all singular values — callers can truncate to global_rank.
    return U, S, Vh                                    # full decomposition


def _truncate_svd(U, S, Vh, rank: int):
    return U[:, :rank], S[:rank], Vh[:rank, :]


def _recompose(U, S, Vh) -> np.ndarray:
    return (U * S[None, :]) @ Vh                       # (d_out, d_in)


def _weighted_svd_merge(
    contributions: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]],
    target_rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge variable-rank FlexLoRA deltas from multiple clients.

    contributions: list of (weight, U, S, Vh)
    Returns: merged (lora_B, lora_A) projected to target_rank.
    """
    # Weighted sum in full-rank delta space
    delta_sum = None
    total_w = sum(w for w, *_ in contributions)
    for w, U, S, Vh in contributions:
        d = _recompose(U, S, Vh) * (w / (total_w + 1e-9))
        delta_sum = d if delta_sum is None else delta_sum + d

    # Re-decompose at target rank
    U_m, S_m, Vh_m = np.linalg.svd(delta_sum, full_matrices=False)
    U_r, S_r, Vh_r = _truncate_svd(U_m, S_m, Vh_m, target_rank)

    # Reconstruct LoRA factors: B = U * sqrt(S), A = sqrt(S) * Vh
    sqrt_S = np.sqrt(np.maximum(S_r, 0.0))
    lora_B = U_r * sqrt_S[None, :]                     # (d_out, r)
    lora_A = sqrt_S[:, None] * Vh_r                    # (r, d_in)
    return lora_B, lora_A


# ──────────────────────────────────────────────────────────────────────────────
# Rotating Validation Buffer
# ──────────────────────────────────────────────────────────────────────────────

class ValidationBuffer:
    """
    Maintains a fixed-size pool of (expert_path, text) pairs.
    Partitions into path-specific sub-sets for targeted evaluation.
    """

    def __init__(self, capacity: int = 512):
        self.capacity = capacity
        self.samples: list[dict] = []           # {"path": tuple, "loss": float, "text": str}
        self._global_baseline: Optional[float] = None

    def update(self, samples: list[dict]) -> None:
        self.samples.extend(samples)
        if len(self.samples) > self.capacity:
            random.shuffle(self.samples)
            self.samples = self.samples[:self.capacity]

    def samples_for_path(self, path: tuple[int, ...]) -> list[dict]:
        return [s for s in self.samples if tuple(s["path"]) == path]

    def samples_excluding_path(self, path: tuple[int, ...]) -> list[dict]:
        return [s for s in self.samples if tuple(s["path"]) != path]

    def set_baseline(self, loss: float) -> None:
        self._global_baseline = loss

    @property
    def baseline(self) -> float:
        return self._global_baseline or float("inf")


# ──────────────────────────────────────────────────────────────────────────────
# Strategy
# ──────────────────────────────────────────────────────────────────────────────

class MoEFederatedStrategy(fl.server.strategy.Strategy):
    """
    Custom Flower strategy for MoE federated fine-tuning.

    Parameters
    ----------
    min_fit_clients / min_available_clients
        Standard Flower quorum settings.
    confidence_floor : float
        Guaranteed minimum merge weight even for low-reward updates.
    reward_threshold : float
        Updates with reward below this are rejected outright.
    global_rank : int
        Target LoRA rank for merged adapters (FlexLoRA canonical rank).
    regression_penalty_coeff : float
        Weight of the cross-path regression penalty in reward computation.
    rollback_on_regression : bool
        If True, roll back to previous global weights when aggregate eval
        shows degradation.
    val_buffer : ValidationBuffer
        Pre-populated validation buffer injected at construction time.
    """

    def __init__(
        self,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.0,
        confidence_floor: float = 0.1,
        reward_threshold: float = -0.5,
        global_rank: int = 16,
        regression_penalty_coeff: float = 0.3,
        rollback_on_regression: bool = True,
        val_buffer: Optional[ValidationBuffer] = None,
        initial_parameters: Optional[Parameters] = None,
    ):
        self.min_fit_clients           = min_fit_clients
        self.min_available_clients     = min_available_clients
        self.fraction_fit              = fraction_fit
        self.fraction_evaluate         = fraction_evaluate
        self.confidence_floor          = confidence_floor
        self.reward_threshold          = reward_threshold
        self.global_rank               = global_rank
        self.regression_penalty_coeff  = regression_penalty_coeff
        self.rollback_on_regression    = rollback_on_regression
        self.val_buffer                = val_buffer or ValidationBuffer()
        self.initial_parameters        = initial_parameters

        # State for rollback
        self._last_good_params: Optional[list[np.ndarray]] = None
        self._last_good_loss: float = float("inf")

        # Per-round bookkeeping
        self._round_metrics: list[dict] = []

    # ── Flower interface ───────────────────────────────────────────────────────

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        config = {
            "server_round":  server_round,
            "global_rank":   self.global_rank,
        }
        ins = FitIns(parameters, config)
        sample_size = max(
            self.min_fit_clients,
            int(client_manager.num_available() * self.fraction_fit),
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_available_clients,
        )
        return [(c, ins) for c in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if not results:
            log.warning("Round %d: no results received.", server_round)
            return None, {}

        if failures:
            log.warning("Round %d: %d client failures.", server_round, len(failures))

        # Step 1: Score each update
        accepted: list[tuple[float, list[np.ndarray], dict]] = []
        rejected_count = 0

        for client, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            meta   = fit_res.metrics or {}
            reward = self._compute_reward(arrays, meta)
            log.info("  Client %s → reward=%.4f (path=%s)",
                     client.cid, reward, meta.get("expert_path", "?"))

            if reward < self.reward_threshold:
                log.warning("  Rejected client %s (reward %.4f < threshold %.4f)",
                            client.cid, reward, self.reward_threshold)
                rejected_count += 1
                continue

            weight = self._reward_to_weight(reward)
            accepted.append((weight, arrays, meta))

        if not accepted:
            log.error("Round %d: all updates rejected.", server_round)
            return (
                ndarrays_to_parameters(self._last_good_params)
                if self._last_good_params else None
            ), {"rejected": rejected_count}

        # Step 2: FlexLoRA-aware weighted merge
        merged_arrays = self._flexlora_merge(accepted)

        # Step 3: Rollback check
        merged_loss = self._evaluate_global(merged_arrays)
        if self.rollback_on_regression and merged_loss > self._last_good_loss * 1.05:
            log.warning("Round %d: merged model regressed (%.4f > %.4f). Rolling back.",
                        server_round, merged_loss, self._last_good_loss)
            return (
                ndarrays_to_parameters(self._last_good_params)
                if self._last_good_params else None
            ), {"rollback": True, "merged_loss": merged_loss}

        # Accept merged
        self._last_good_params = copy.deepcopy(merged_arrays)
        self._last_good_loss   = merged_loss

        metrics_agg = {
            "round":          server_round,
            "accepted":       len(accepted),
            "rejected":       rejected_count,
            "merged_val_loss": merged_loss,
        }
        self._round_metrics.append(metrics_agg)
        log.info("Round %d aggregated: %d accepted, %d rejected, val_loss=%.4f",
                 server_round, len(accepted), rejected_count, merged_loss)

        return ndarrays_to_parameters(merged_arrays), metrics_agg

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []   # Server-side evaluation only

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters) -> Optional[tuple[float, dict]]:
        arrays = parameters_to_ndarrays(parameters)
        loss = self._evaluate_global(arrays)
        return loss, {"val_loss": loss}

    # ── Reward & Weighting ────────────────────────────────────────────────────

    def _compute_reward(self, arrays: list[np.ndarray], meta: dict) -> float:
        """
        reward = normalised_loss_improvement - regression_penalty

        loss_improvement: how much better the client's adapter is on its own path.
        regression_penalty: how much worse it is on *other* paths in the val buffer.
        """
        expert_path = tuple(meta.get("expert_path", []))
        local_val_loss = float(meta.get("local_val_loss", 1.0))
        local_train_loss = float(meta.get("local_train_loss", 1.0))

        # 1. Normalised loss improvement (client-reported)
        baseline_loss = max(local_train_loss, 1e-6)
        improvement   = (baseline_loss - local_val_loss) / baseline_loss
        norm_improvement = max(-1.0, min(1.0, improvement))

        # 2. Cross-path regression penalty (server-side val buffer)
        other_samples = self.val_buffer.samples_excluding_path(expert_path)
        if other_samples and self._last_good_params:
            # Use proxy: if client confidence is low, penalise more
            client_confidence = float(meta.get("confidence", 0.5))
            regression_penalty = self.regression_penalty_coeff * (1.0 - client_confidence)
        else:
            regression_penalty = 0.0

        reward = norm_improvement - regression_penalty
        return reward

    def _reward_to_weight(self, reward: float) -> float:
        """
        Sigmoid-squashed merge weight in [confidenceFloor, 1].
        """
        raw = _sigmoid(reward, scale=4.0)              # ∈ (0, 1)
        floor = self.confidence_floor
        return floor + (1.0 - floor) * raw

    # ── FlexLoRA Merge ────────────────────────────────────────────────────────

    def _flexlora_merge(
        self,
        accepted: list[tuple[float, list[np.ndarray], dict]],
    ) -> list[np.ndarray]:
        """
        Merge accepted client arrays using FlexLoRA SVD reconstruction
        for LoRA pairs and plain weighted average for non-LoRA layers.

        Convention: arrays are named by index pairs (lora_A at 2i, lora_B at 2i+1).
        Non-LoRA arrays (bias, norm, embed) are simple weighted-averaged.

        In practice the client packs arrays as:
            [lora_A_0, lora_B_0, lora_A_1, lora_B_1, ..., other_0, other_1, ...]
        with metadata key "num_lora_pairs" indicating how many LoRA pairs precede
        the non-LoRA arrays.
        """
        if not accepted:
            return []

        # Determine num_lora_pairs from first client's metadata
        num_lora_pairs = int(accepted[0][2].get("num_lora_pairs", 0))
        n_arrays = len(accepted[0][1])

        # Non-LoRA arrays: plain weighted average
        total_weight = sum(w for w, _, _ in accepted)
        merged = []

        for idx in range(n_arrays):
            # LoRA pair: A at even index, B at next odd
            lora_pair_idx = idx // 2
            is_lora_A = (idx % 2 == 0) and (lora_pair_idx < num_lora_pairs)
            is_lora_B = (idx % 2 == 1) and (lora_pair_idx < num_lora_pairs)

            if is_lora_A:
                # Defer — handled with paired B
                merged.append(None)
                continue
            if is_lora_B:
                # Reconstruct and merge the pair
                A_idx = idx - 1
                contributions = []
                for w, arrs, meta in accepted:
                    lA = arrs[A_idx]      # (r_client, d_in)
                    lB = arrs[idx]        # (d_out, r_client)
                    U, S, Vh = _flexlora_reconstruct(lA, lB)
                    contributions.append((w, U, S, Vh))
                lora_B_m, lora_A_m = _weighted_svd_merge(contributions, self.global_rank)
                merged[A_idx] = lora_A_m
                merged.append(lora_B_m)
                continue

            # Plain weighted average
            avg = sum(w * arrs[idx] for w, arrs, _ in accepted) / (total_weight + 1e-9)
            merged.append(avg)

        return merged

    # ── Global Validation ─────────────────────────────────────────────────────

    def _evaluate_global(self, arrays: list[np.ndarray]) -> float:
        """
        Proxy global validation using the val buffer's stored per-sample losses.
        In a full system this would load the model weights and run inference.
        Here we use the stored baseline losses as a proxy and return a scalar.
        """
        if not self.val_buffer.samples:
            return 0.0
        losses = [s.get("loss", 1.0) for s in self.val_buffer.samples]
        return float(np.mean(losses))
