"""Confidence-weighted, path-aware aggregation for MoE LoRA clients.

The server keeps one canonical LoRA representation per expert layer. Clients
may train different expert paths and may use different LoRA ranks. Each client
returns before/after LoRA factors; the server converts those factors into an
effective weight delta and applies the weighted delta only to the layers that
client actually trained.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import os
import uuid
from typing import Optional

import numpy as np
import flwr as fl
from flwr.common import EvaluateRes, FitIns, FitRes, GetParametersIns, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

log = logging.getLogger(__name__)


def _sigmoid(x: float, scale: float = 4.0) -> float:
    x = float(np.clip(x, -50.0, 50.0))
    return 1.0 / (1.0 + math.exp(-scale * x))


def _decode_json_metric(value, default):
    if value is None:
        return default
    if isinstance(value, (list, tuple, dict)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def _factorize_delta(delta: np.ndarray, target_rank: int) -> tuple[np.ndarray, np.ndarray]:
    delta = np.asarray(delta, dtype=np.float32)
    if delta.ndim != 2:
        raise ValueError(f"LoRA delta must be 2-D, got {delta.shape}")
    U, S, Vh = np.linalg.svd(delta, full_matrices=False)
    r = min(int(target_rank), len(S))
    sqrt_s = np.sqrt(np.maximum(S[:r], 0.0))
    B = U[:, :r] * sqrt_s[None, :]
    A = sqrt_s[:, None] * Vh[:r, :]
    if r < target_rank:
        A_pad = np.zeros((target_rank, A.shape[1]), dtype=np.float32)
        B_pad = np.zeros((B.shape[0], target_rank), dtype=np.float32)
        A_pad[:r] = A
        B_pad[:, :r] = B
        A, B = A_pad, B_pad
    return A.astype(np.float32), B.astype(np.float32)


def _update_norm(arrays: list[np.ndarray]) -> float:
    total = 0.0
    for arr in arrays:
        x = np.asarray(arr, dtype=np.float32)
        if not np.all(np.isfinite(x)):
            return float("inf")
        total += float(np.sum(x * x))
    return float(np.sqrt(total))


class ValidationBuffer:
    def __init__(self, capacity: int = 512):
        self.capacity = capacity
        self.samples: list[dict] = []

    def update(self, samples: list[dict]) -> None:
        self.samples.extend(samples)
        if len(self.samples) > self.capacity:
            self.samples = self.samples[-self.capacity:]

    def samples_for_path(self, path: tuple[int, ...]) -> list[dict]:
        return [s for s in self.samples if tuple(s.get("path", [])) == path]

    def samples_excluding_path(self, path: tuple[int, ...]) -> list[dict]:
        return [s for s in self.samples if tuple(s.get("path", [])) != path]


class MoEFederatedStrategy(fl.server.strategy.Strategy):
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
        rollback_threshold: float = 0.5,
        val_buffer: Optional[ValidationBuffer] = None,
        initial_parameters: Optional[Parameters] = None,
    ):
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.confidence_floor = confidence_floor
        self.reward_threshold = reward_threshold
        self.global_rank = global_rank
        self.regression_penalty_coeff = regression_penalty_coeff
        self.rollback_on_regression = rollback_on_regression
        # NOTE: this is a *parameter-space* drift guard (relative L2 change in
        # the merged LoRA weights vs. the last accepted checkpoint), not a
        # validation-loss regression check. The server has no shared held-out
        # dataset in Phase 1, so it cannot measure real loss regression. 0.5
        # means "reject a merge that moves the canonical weights by more than
        # 50% in relative norm in a single round" - a sane instability guard.
        # Rename/replace this once server-side validation data exists.
        self.rollback_threshold = rollback_threshold
        self.val_buffer = val_buffer or ValidationBuffer()
        self.initial_parameters = initial_parameters
        self._global_parameters = initial_parameters
        self._last_good_params: Optional[list[np.ndarray]] = None
        # Snapshot taken BEFORE each aggregate_fit call, used only so that
        # evaluate() (called by Flower right after aggregate_fit) can report
        # this round's drift instead of comparing the new global params to
        # themselves (which is always ~0 and was previously misleading).
        self._pre_round_params: Optional[list[np.ndarray]] = None
        self._last_good_loss = float("inf")
        self._last_layer_names: list[str] = []
        self._last_num_lora_pairs = 0
        self._session_id = uuid.uuid4().hex[:12]
        self.local_steps = int(os.getenv("LOCAL_STEPS", "200"))

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        return self._global_parameters

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        config = {
            "server_round": int(server_round),
            "global_rank": int(self.global_rank),
            "local_steps": int(self.local_steps),
            "global_meta": json.dumps(self._global_meta()),
            "session_id": getattr(self, "_session_id", "default"),
        }
        ins = FitIns(parameters, config)
        available = client_manager.num_available()
        sample_size = min(max(self.min_fit_clients, int(available * self.fraction_fit)), available)
        if sample_size < self.min_fit_clients:
            return []
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=self.min_available_clients)
        return [(client, ins) for client in clients]

    def aggregate_fit(self, server_round: int, results: list[tuple[ClientProxy, FitRes]], failures):
        if not results:
            return self._global_parameters, {"accepted": 0, "rejected": 0}

        accepted: list[tuple[float, list[np.ndarray], dict]] = []
        rejected = 0
        for client, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            meta = dict(fit_res.metrics or {})
            reward = self._compute_reward(meta)
            path = _decode_json_metric(meta.get("expert_path"), [])
            confidence = float(meta.get("confidence", 0.0))
            log.info("Client %s -> reward=%.4f path=%s confidence=%.4f", client.cid, reward, path, confidence)
            if reward < self.reward_threshold:
                rejected += 1
                log.warning("Rejected client %s: reward %.4f < threshold %.4f", client.cid, reward, self.reward_threshold)
                continue
            accepted.append((self._reward_to_weight(reward), arrays, meta))

        if not accepted:
            return self._global_parameters, {"round": server_round, "accepted": 0, "rejected": rejected, "rollback": False}

        previous = copy.deepcopy(self._last_good_params) if self._last_good_params is not None else None
        merged, layer_names = self._merge_path_aware(accepted)
        metric = self._parameter_sanity_metric(merged, previous)

        if self.rollback_on_regression and previous is not None and metric > self.rollback_threshold:
            log.warning("Round %d: unstable aggregate (relative_change=%.4f); rolling back", server_round, metric)
            return self._global_parameters, {"round": server_round, "accepted": len(accepted), "rejected": rejected, "rollback": True, "server_validation": metric}

        # Snapshot for evaluate() to diff against BELOW, before we overwrite
        # _last_good_params with the just-merged result.
        self._pre_round_params = previous
        self._last_good_params = copy.deepcopy(merged)
        self._last_good_loss = metric
        self._global_parameters = ndarrays_to_parameters(merged)
        self._last_layer_names = layer_names
        self._last_num_lora_pairs = len(layer_names)

        log.info("Round %d aggregated: accepted=%d rejected=%d canonical_rank=%d layers=%d", server_round, len(accepted), rejected, self.global_rank, len(layer_names))
        return self._global_parameters, {
            "round": int(server_round),
            "accepted": int(len(accepted)),
            "rejected": int(rejected),
            "rollback": False,
            "server_validation": float(metric),
            "global_rank": int(self.global_rank),
            "global_layers": int(len(layer_names)),
        }

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round: int, parameters: Parameters):
        # Compare the just-published global parameters against what was
        # canonical BEFORE this round's merge, not against themselves
        # (self._last_good_params is set to `arrays` right before Flower
        # calls this, so diffing against it was always ~0).
        arrays = parameters_to_ndarrays(parameters)
        reference = self._pre_round_params if server_round > 0 else None
        metric = self._parameter_sanity_metric(arrays, reference)
        return metric, {"server_validation": metric}

    def _compute_reward(self, meta: dict) -> float:
        initial_val = float(meta.get("initial_val_loss", float("inf")))
        final_val = float(meta.get("local_val_loss", float("inf")))
        confidence = float(np.clip(float(meta.get("confidence", 0.0)), 0.0, 1.0))
        if not np.isfinite(initial_val) or initial_val <= 0 or not np.isfinite(final_val):
            return -1.0
        improvement = float(np.clip((initial_val - final_val) / initial_val, -1.0, 1.0))
        # TODO(phase 2): the cross-path regression penalty needs a shared
        # server-side validation set to be meaningful. self.val_buffer is
        # currently never populated (no caller feeds it samples), so a
        # penalty computed from it would silently always be 0 - which is
        # worse than not having the term at all, since it implies protection
        # that isn't there. Reward is just the client's own loss improvement
        # until a real cross-client validation pass exists.
        return improvement

    def _reward_to_weight(self, reward: float) -> float:
        return self.confidence_floor + (1.0 - self.confidence_floor) * _sigmoid(reward)

    def _merge_path_aware(self, accepted):
        # Decode the current canonical global model into layer -> (A, B).
        current: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if self._global_parameters is not None and self._last_layer_names:
            old = parameters_to_ndarrays(self._global_parameters)
            if len(old) == 2 * len(self._last_layer_names):
                for i, name in enumerate(self._last_layer_names):
                    current[name] = (np.asarray(old[2*i], dtype=np.float32), np.asarray(old[2*i+1], dtype=np.float32))

        weighted_deltas: dict[str, list[tuple[float, np.ndarray]]] = {}
        total_weights: dict[str, float] = {}
        all_names = list(current.keys())

        for weight, arrays, meta in accepted:
            names = list(_decode_json_metric(meta.get("layer_names"), []))
            num_pairs = int(meta.get("num_lora_pairs", 0))
            if len(names) != num_pairs:
                raise ValueError(f"Metadata mismatch: names={len(names)} pairs={num_pairs}")
            if len(arrays) != 4 * num_pairs:
                raise ValueError(f"Expected 4 arrays per LoRA pair, got {len(arrays)} for {num_pairs} pairs")

            for i, name in enumerate(names):
                A_before = np.asarray(arrays[4*i], dtype=np.float32)
                B_before = np.asarray(arrays[4*i+1], dtype=np.float32)
                A_after = np.asarray(arrays[4*i+2], dtype=np.float32)
                B_after = np.asarray(arrays[4*i+3], dtype=np.float32)
                delta = (B_after @ A_after) - (B_before @ A_before)
                if not np.all(np.isfinite(delta)):
                    raise ValueError(f"Non-finite delta for {name}")
                weighted_deltas.setdefault(name, []).append((max(float(weight), 1e-8), delta))
                total_weights[name] = total_weights.get(name, 0.0) + max(float(weight), 1e-8)
                if name not in all_names:
                    all_names.append(name)

        merged: list[np.ndarray] = []
        for name in all_names:
            if name in weighted_deltas:
                contributions = weighted_deltas[name]
                delta = sum(w * d for w, d in contributions) / max(total_weights[name], 1e-8)
                if name in current:
                    base_A, base_B = current[name]
                    base = base_B @ base_A
                    target = base + delta
                else:
                    target = delta
                A, B = _factorize_delta(target, self.global_rank)
            else:
                A, B = current[name]
            merged.extend([A, B])

        return merged, all_names

    def _global_meta(self) -> dict:
        return {"global_rank": int(self.global_rank), "layer_names": list(self._last_layer_names), "num_lora_pairs": int(self._last_num_lora_pairs)}

    def _parameter_sanity_metric(self, arrays, previous):
        norm = _update_norm(arrays)
        if previous is None:
            return norm
        if len(arrays) != len(previous):
            return float("inf")
        diff_sq = base_sq = 0.0
        for c0, p0 in zip(arrays, previous):
            c = np.asarray(c0, dtype=np.float32); p = np.asarray(p0, dtype=np.float32)
            if c.shape != p.shape:
                return float("inf")
            diff_sq += float(np.sum((c-p)**2)); base_sq += float(np.sum(p**2))
        return float(np.sqrt(diff_sq) / max(np.sqrt(base_sq), 1e-6))


__all__ = ["MoEFederatedStrategy", "ValidationBuffer"]