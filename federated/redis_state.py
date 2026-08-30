"""
redis_state.py — Centralized Redis state manager.

Responsibilities:
  - Node registry: capacity, last-seen heartbeat.
  - Task lifecycle FSM: pending → assigned → training → done | failed.
  - Heartbeat watchdog: requeue tasks for timed-out nodes.
  - Atomic task assignment sized to node compute budget.
"""

from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, asdict, field
from typing import Optional
from enum import Enum

import redis

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
HEARTBEAT_TTL_S = 30          # Node considered dead after this many seconds.
TASK_ASSIGN_TIMEOUT_S = 60    # Give up on assigned-but-not-started tasks.

_KEY_NODE       = "node:{nid}"
_KEY_TASK       = "task:{tid}"
_KEY_QUEUE      = "queue:{state}"       # Lists for pending / assigned / …
_KEY_NODE_TASK  = "node_task:{nid}"     # Which task is a node currently doing
_KEY_ALL_NODES  = "all_nodes"
_KEY_ALL_TASKS  = "all_tasks"


class TaskState(str, Enum):
    PENDING   = "pending"
    ASSIGNED  = "assigned"
    TRAINING  = "training"
    DONE      = "done"
    FAILED    = "failed"


@dataclass
class NodeInfo:
    node_id:       str
    ram_gb:        float          # Reported RAM available (GB)
    compute_score: float          # Arbitrary compute budget 0-1
    last_seen:     float = field(default_factory=time.time)
    current_task:  Optional[str] = None

    def to_dict(self) -> dict:
        return {k: ("" if v is None else v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> "NodeInfo":
        d = {k: (None if v == "" else v) for k, v in d.items()}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TaskInfo:
    task_id:        str
    bucket_path:    list[int]       # Expert indices that define this bucket
    bucket_size:    int             # Number of samples in the bucket
    state:          str = TaskState.PENDING
    assigned_node:  Optional[str] = None
    assigned_at:    Optional[float] = None
    started_at:     Optional[float] = None
    completed_at:   Optional[float] = None
    retries:        int = 0
    result_ref:     Optional[str] = None    # Redis key / blob path for adapter delta

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bucket_path"] = json.dumps(self.bucket_path)
        return {k: ("" if v is None else v) for k, v in d.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "TaskInfo":
        d = dict(d)
        if isinstance(d.get("bucket_path"), str):
            d["bucket_path"] = json.loads(d["bucket_path"])
        if "retries" in d and d["retries"] != "":
            d["retries"] = int(d["retries"])
        if "bucket_size" in d and d["bucket_size"] != "":
            d["bucket_size"] = int(d["bucket_size"])
        d = {k: (None if v == "" else v) for k, v in d.items()}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class RedisStateManager:
    """Thread-safe, Redis-backed state manager for the federated orchestrator."""

    def __init__(self, host: str = "redis", port: int = 6379, db: int = 0):
        self.r = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        log.info("RedisStateManager connected to %s:%d", host, port)

    # ── Node Registry ─────────────────────────────────────────────────────────

    def register_node(self, node: NodeInfo) -> None:
        key = _KEY_NODE.format(nid=node.node_id)
        self.r.hset(key, mapping=node.to_dict())
        self.r.sadd(_KEY_ALL_NODES, node.node_id)
        log.info("Registered node %s (ram=%.1f GB, compute=%.2f)",
                 node.node_id, node.ram_gb, node.compute_score)

    def heartbeat(self, node_id: str) -> bool:
        """Update last_seen. Returns False if node is unknown."""
        key = _KEY_NODE.format(nid=node_id)
        if not self.r.exists(key):
            return False
        self.r.hset(key, "last_seen", time.time())
        return True

    def get_node(self, node_id: str) -> Optional[NodeInfo]:
        raw = self.r.hgetall(_KEY_NODE.format(nid=node_id))
        if not raw:
            return None
        raw["ram_gb"]        = float(raw["ram_gb"])
        raw["compute_score"] = float(raw["compute_score"])
        raw["last_seen"]     = float(raw["last_seen"])
        return NodeInfo.from_dict(raw)

    def list_nodes(self) -> list[NodeInfo]:
        nids = self.r.smembers(_KEY_ALL_NODES)
        return [n for nid in nids if (n := self.get_node(nid)) is not None]

    # ── Task Lifecycle ────────────────────────────────────────────────────────

    def enqueue_task(self, task: TaskInfo) -> None:
        """Push a new task onto the pending queue (idempotent)."""
        key = _KEY_TASK.format(tid=task.task_id)
        if self.r.exists(key):
            return  # Already known, skip re-add
        task.state = TaskState.PENDING
        self.r.hset(key, mapping=task.to_dict())
        self.r.sadd(_KEY_ALL_TASKS, task.task_id)
        self.r.rpush(_KEY_QUEUE.format(state=TaskState.PENDING), task.task_id)
        log.debug("Enqueued task %s (path=%s, size=%d)",
                  task.task_id, task.bucket_path, task.bucket_size)

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        raw = self.r.hgetall(_KEY_TASK.format(tid=task_id))
        return TaskInfo.from_dict(raw) if raw else None

    def assign_task_to_node(self, node_id: str) -> Optional[TaskInfo]:
        """
        Pop from pending, transition → assigned, record ownership.
        Capacity-aware: skips tasks whose bucket is too large for the node.
        """
        node = self.get_node(node_id)
        if node is None:
            return None

        # Max samples a node can handle (heuristic: 1 GB RAM ~ 500 samples at fp16)
        max_samples = int(node.ram_gb * 500 * node.compute_score)

        pending_key = _KEY_QUEUE.format(state=TaskState.PENDING)
        # Inspect the queue without permanently consuming
        queue_len = self.r.llen(pending_key)
        for _ in range(queue_len):
            task_id = self.r.lpop(pending_key)
            if task_id is None:
                break
            task = self.get_task(task_id)
            if task is None:
                continue
            if task.bucket_size <= max_samples:
                # Fits — assign it
                task.state         = TaskState.ASSIGNED
                task.assigned_node = node_id
                task.assigned_at   = time.time()
                self._save_task(task)
                self.r.rpush(_KEY_QUEUE.format(state=TaskState.ASSIGNED), task_id)
                self.r.hset(_KEY_NODE.format(nid=node_id), "current_task", task_id)
                self.r.set(_KEY_NODE_TASK.format(nid=node_id), task_id)
                log.info("Assigned task %s → node %s", task_id, node_id)
                return task
            else:
                # Too big — put back and keep looking
                self.r.rpush(pending_key, task_id)

        log.debug("No suitable task found for node %s (max_samples=%d)", node_id, max_samples)
        return None

    def mark_training(self, task_id: str, node_id: str) -> None:
        task = self.get_task(task_id)
        if task and task.assigned_node == node_id:
            task.state      = TaskState.TRAINING
            task.started_at = time.time()
            self._save_task(task)
            self._move_queue(task_id, TaskState.ASSIGNED, TaskState.TRAINING)

    def mark_done(self, task_id: str, node_id: str, result_ref: str) -> None:
        task = self.get_task(task_id)
        if task and task.assigned_node == node_id:
            task.state        = TaskState.DONE
            task.completed_at = time.time()
            task.result_ref   = result_ref
            self._save_task(task)
            self._move_queue(task_id, TaskState.TRAINING, TaskState.DONE)
            self.r.delete(_KEY_NODE_TASK.format(nid=node_id))
            log.info("Task %s done by node %s, result_ref=%s", task_id, node_id, result_ref)

    def mark_failed(self, task_id: str, node_id: str, requeue: bool = True) -> None:
        task = self.get_task(task_id)
        if task is None:
            return
        task.retries += 1
        if requeue and task.retries < 3:
            task.state         = TaskState.PENDING
            task.assigned_node = None
            task.assigned_at   = None
            task.started_at    = None
            self._save_task(task)
            # Remove from wherever it currently sits
            for state in [TaskState.ASSIGNED, TaskState.TRAINING, TaskState.FAILED]:
                self.r.lrem(_KEY_QUEUE.format(state=state), 0, task_id)
            self.r.rpush(_KEY_QUEUE.format(state=TaskState.PENDING), task_id)
            log.warning("Task %s failed (node %s), requeued (retry %d)",
                        task_id, node_id, task.retries)
        else:
            task.state = TaskState.FAILED
            self._save_task(task)
            for state in [TaskState.ASSIGNED, TaskState.TRAINING, TaskState.PENDING]:
                self.r.lrem(_KEY_QUEUE.format(state=state), 0, task_id)
            self.r.rpush(_KEY_QUEUE.format(state=TaskState.FAILED), task_id)
            log.error("Task %s permanently failed after %d retries", task_id, task.retries)
        self.r.delete(_KEY_NODE_TASK.format(nid=node_id))

    # ── Heartbeat Watchdog ────────────────────────────────────────────────────

    def reap_dead_nodes(self) -> list[str]:
        """
        Scan all nodes; requeue tasks for those that missed heartbeats.
        Returns list of reaped node IDs.
        """
        reaped = []
        now = time.time()
        for node in self.list_nodes():
            age = now - node.last_seen
            if age > HEARTBEAT_TTL_S:
                log.warning("Node %s timed out (last_seen %.0fs ago)", node.node_id, age)
                task_id = self.r.get(_KEY_NODE_TASK.format(nid=node.node_id))
                if task_id:
                    self.mark_failed(task_id, node.node_id, requeue=True)
                reaped.append(node.node_id)
        return reaped

    def queue_lengths(self) -> dict[str, int]:
        return {
            state.value: self.r.llen(_KEY_QUEUE.format(state=state))
            for state in TaskState
        }

    def all_task_states(self) -> dict[str, str]:
        tids = self.r.smembers(_KEY_ALL_TASKS)
        out = {}
        for tid in tids:
            task = self.get_task(tid)
            if task:
                out[tid] = task.state
        return out

    # ── Internals ─────────────────────────────────────────────────────────────

    def _save_task(self, task: TaskInfo) -> None:
        self.r.hset(_KEY_TASK.format(tid=task.task_id), mapping=task.to_dict())

    def _move_queue(self, task_id: str, from_state: TaskState, to_state: TaskState) -> None:
        self.r.lrem(_KEY_QUEUE.format(state=from_state), 0, task_id)
        self.r.rpush(_KEY_QUEUE.format(state=to_state), task_id)
