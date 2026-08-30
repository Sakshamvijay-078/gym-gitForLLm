"""
orchestrator.py — FastAPI service hosting the Flower server + admin endpoints.

Responsibilities:
  • POST /register       — Node registration (reports RAM/compute capacity)
  • POST /heartbeat      — Liveness signal from client nodes
  • GET  /assign_task    — Capacity-aware task assignment
  • POST /task/training  — Node signals it has started training
  • POST /task/done      — Node submits completed adapter delta ref
  • POST /task/failed    — Node reports failure
  • GET  /status         — Dashboard: queue depths, node states
  • Background watchdog  — Reaps dead nodes every HEARTBEAT_TTL_S seconds

The Flower gRPC server runs on its own thread alongside FastAPI.
"""

from __future__ import annotations

import signal
import threading
_original_signal = signal.signal
def _patched_signal(signum, handler):
    if threading.current_thread() is threading.main_thread():
        return _original_signal(signum, handler)
    return None
signal.signal = _patched_signal

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import redis as redis_lib
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import flwr as fl

from redis_state import (
    RedisStateManager,
    NodeInfo,
    TaskInfo,
    TaskState,
    HEARTBEAT_TTL_S,
)
from custom_strategy import MoEFederatedStrategy, ValidationBuffer
from data_pipeline import load_bucket_manifest

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

REDIS_HOST         = os.getenv("REDIS_HOST", "redis")
REDIS_PORT         = int(os.getenv("REDIS_PORT", 6379))
FLOWER_PORT        = int(os.getenv("FLOWER_PORT", 8080))
API_PORT           = int(os.getenv("API_PORT",    8000))
BUCKET_DIR         = Path(os.getenv("BUCKET_DIR", "/data/buckets"))
WATCHDOG_INTERVAL  = int(os.getenv("WATCHDOG_INTERVAL_S", 15))
FL_ROUNDS          = int(os.getenv("FL_ROUNDS", 10))
FL_MIN_CLIENTS     = int(os.getenv("FL_MIN_CLIENTS", 2))
ROLLBACK_THRESHOLD = float(os.getenv("ROLLBACK_THRESHOLD", 0.5))

# ──────────────────────────────────────────────────────────────────────────────
# Global state (initialized in lifespan)
# ──────────────────────────────────────────────────────────────────────────────

state_mgr: Optional[RedisStateManager] = None
strategy:  Optional[MoEFederatedStrategy] = None


# ──────────────────────────────────────────────────────────────────────────────
# Lifespan: start Flower server + watchdog as background threads
# ──────────────────────────────────────────────────────────────────────────────

def _start_flower_server():
    """Run the Flower gRPC server in a daemon thread."""
    log.info("Starting Flower server on port %d", FLOWER_PORT)
    fl.server.start_server(
        server_address=f"0.0.0.0:{FLOWER_PORT}",
        config=fl.server.ServerConfig(num_rounds=FL_ROUNDS),
        strategy=strategy,
    )


def _watchdog_loop():
    """Periodically reap dead nodes and requeue their tasks."""
    log.info("Watchdog started (interval=%ds, TTL=%ds)", WATCHDOG_INTERVAL, HEARTBEAT_TTL_S)
    while True:
        time.sleep(WATCHDOG_INTERVAL)
        try:
            reaped = state_mgr.reap_dead_nodes()
            if reaped:
                log.warning("Watchdog reaped %d dead nodes: %s", len(reaped), reaped)
        except Exception as e:
            log.error("Watchdog error: %s", e)


def _populate_tasks_from_manifest():
    """Seed the Redis pending queue from the bucket manifest produced by data_pipeline.py."""
    if not BUCKET_DIR.exists():
        log.warning("BUCKET_DIR %s does not exist; no tasks seeded", BUCKET_DIR)
        return
    try:
        manifest = load_bucket_manifest(BUCKET_DIR)
    except FileNotFoundError:
        log.warning("No bucket manifest found; no tasks seeded")
        return
    for path_str, info in manifest.items():
        # IMPORTANT: task_id must be deterministic per bucket. It previously
        # included a random uuid suffix, which meant enqueue_task()'s
        # idempotency check (`if self.r.exists(key): return`) could never
        # find a match on restart - every orchestrator restart re-seeded ALL
        # buckets as brand-new duplicate tasks, even ones already completed,
        # so the pending queue only ever grew across dev runs. A deterministic
        # id keyed on the bucket path lets Redis correctly recognize "this
        # bucket already has a task" (pending, assigned, or done) and skip it.
        task = TaskInfo(
            task_id     = f"task_{path_str}",
            bucket_path = info["path"],
            bucket_size = info["num_chunks"],
        )
        state_mgr.enqueue_task(task)
    log.info("Seeded %d tasks from manifest", len(manifest))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global state_mgr, strategy

    # Init Redis state manager
    state_mgr = RedisStateManager(host=REDIS_HOST, port=REDIS_PORT)
    val_buffer = ValidationBuffer(capacity=512)

    # Init Flower strategy
    strategy = MoEFederatedStrategy(
        min_fit_clients        = FL_MIN_CLIENTS,
        min_available_clients  = FL_MIN_CLIENTS,
        confidence_floor       = 0.1,
        reward_threshold       = -0.5,
        global_rank            = 16,
        regression_penalty_coeff = 0.3,
        rollback_on_regression = True,
        rollback_threshold     = ROLLBACK_THRESHOLD,
        val_buffer             = val_buffer,
    )

    # Seed task queue from bucket manifest
    _populate_tasks_from_manifest()

    # Start Flower gRPC server in daemon thread
    flower_thread = threading.Thread(target=_start_flower_server, daemon=True)
    flower_thread.start()

    # Start heartbeat watchdog
    watchdog_thread = threading.Thread(target=_watchdog_loop, daemon=True)
    watchdog_thread.start()

    log.info("Orchestrator ready. Flower on :%d, API on :%d", FLOWER_PORT, API_PORT)
    yield
    # Cleanup (graceful shutdown handled by Docker / uvicorn)


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MoE Federated Fine-Tuning Orchestrator",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Pydantic request/response models ──────────────────────────────────────────

class RegisterRequest(BaseModel):
    node_id:       str
    ram_gb:        float = Field(..., gt=0, description="Available RAM in GB")
    compute_score: float = Field(..., ge=0.0, le=1.0, description="Normalized compute budget")


class HeartbeatRequest(BaseModel):
    node_id: str


class TaskStartedRequest(BaseModel):
    task_id: str
    node_id: str


class TaskDoneRequest(BaseModel):
    task_id:    str
    node_id:    str
    result_ref: str   # Path or Redis key to adapter delta blob


class TaskFailedRequest(BaseModel):
    task_id: str
    node_id: str
    reason:  Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/register", status_code=201)
def register_node(req: RegisterRequest):
    node = NodeInfo(
        node_id       = req.node_id,
        ram_gb        = req.ram_gb,
        compute_score = req.compute_score,
    )
    state_mgr.register_node(node)
    return {"status": "registered", "node_id": req.node_id}


@app.post("/heartbeat")
def heartbeat(req: HeartbeatRequest):
    ok = state_mgr.heartbeat(req.node_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Unknown node: {req.node_id}")
    return {"status": "ok", "timestamp": time.time()}


@app.get("/assign_task/{node_id}")
def assign_task(node_id: str):
    """
    Capacity-aware task assignment.
    Skips tasks whose bucket_size exceeds what the node can handle.
    """
    task = state_mgr.assign_task_to_node(node_id)
    if task is None:
        return {"status": "no_task_available"}
    return {
        "status":       "assigned",
        "task_id":      task.task_id,
        "bucket_path":  task.bucket_path,
        "bucket_size":  task.bucket_size,
        "dataset_dir":  str(BUCKET_DIR / f"path_{'_'.join(map(str, task.bucket_path))}"),
    }


@app.post("/task/training")
def task_training(req: TaskStartedRequest):
    state_mgr.mark_training(req.task_id, req.node_id)
    return {"status": "training", "task_id": req.task_id}


@app.post("/task/done")
def task_done(req: TaskDoneRequest):
    state_mgr.mark_done(req.task_id, req.node_id, req.result_ref)
    return {"status": "done", "task_id": req.task_id}


@app.post("/task/failed")
def task_failed(req: TaskFailedRequest):
    log.warning("Task %s failed on node %s: %s", req.task_id, req.node_id, req.reason)
    state_mgr.mark_failed(req.task_id, req.node_id, requeue=True)
    return {"status": "requeued", "task_id": req.task_id}


@app.get("/status")
def status():
    return {
        "queue_lengths": state_mgr.queue_lengths(),
        "task_states":   state_mgr.all_task_states(),
        "nodes":         [
            {"node_id": n.node_id, "ram_gb": n.ram_gb,
             "compute_score": n.compute_score, "last_seen_ago": time.time() - n.last_seen,
             "current_task": n.current_task}
            for n in state_mgr.list_nodes()
        ],
    }


@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": time.time()}


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "orchestrator:app",
        host="0.0.0.0",
        port=API_PORT,
        log_level="info",
        reload=False,
    )