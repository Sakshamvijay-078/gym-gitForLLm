#!/usr/bin/env bash
set -euo pipefail

info() { echo "[INFO] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

ORCH_PID=""
TASK_ID=""
NODE_ID="node_dev"

cleanup() {
  if [[ -n "${ORCH_PID}" ]] && kill -0 "${ORCH_PID}" 2>/dev/null; then
    kill "${ORCH_PID}" 2>/dev/null || true
    info "Orchestrator stopped."
  fi
}
trap cleanup EXIT INT TERM

# ── 1. Redis ────────────────────────────────────────────────────────
python3 - <<'PY'
import redis
r = redis.Redis(host="localhost", port=6379, decode_responses=True)
r.ping()
print("[INFO] Redis: OK")
PY

# ── 2. ToyMoE / buckets ────────────────────────────────────────────
[[ -d ./toy_moe_model ]] || die "./toy_moe_model not found"
info "ToyMoE model already exists at ./toy_moe_model"
if [[ -d ./buckets && -f ./buckets/manifest.json ]]; then
  info "Buckets already exist. Skipping pipeline."
else
  info "Buckets are missing; run your pipeline first."
  die "No bucket manifest found"
fi

# ── 3. Start orchestrator ──────────────────────────────────────────
info "Starting orchestrator (FastAPI + Flower server)..."
REDIS_HOST=localhost \
BUCKET_DIR=./buckets \
FL_ROUNDS=3 \
FL_MIN_CLIENTS=1 \
FLOWER_PORT=8080 \
API_PORT=8000 \
LOCAL_STEPS=200 \
python3 orchestrator.py &
ORCH_PID=$!

for i in $(seq 1 15); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    info "Orchestrator ready at http://localhost:8000"
    break
  fi
  sleep 1
  [[ $i -eq 15 ]] && die "Orchestrator failed to start"
done

# ── 4. Register node and get task assignment ───────────────────────
info "Registering dev node..."
curl -sf -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d '{"node_id":"node_dev","ram_gb":5.0,"compute_score":0.3}' | python3 -m json.tool

TASK_JSON=$(curl -sf http://localhost:8000/assign_task/node_dev)
echo "Task assigned: $TASK_JSON"

TASK_ID=$(echo "$TASK_JSON" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("task_id", ""))')
[[ -n "$TASK_ID" ]] || die "No task was assigned"

DATASET_DIR=$(echo "$TASK_JSON" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("dataset_dir", "./buckets/path_0_1"))')
EXPERTS=$(echo "$TASK_JSON" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(" ".join(map(str,d.get("bucket_path", [0,1]))))')

info "Assigned task: $TASK_ID"
info "Assigned dataset: $DATASET_DIR"
info "Assigned experts: $EXPERTS"

curl -sf -X POST http://localhost:8000/task/training \
  -H "Content-Type: application/json" \
  -d "{\"task_id\":\"$TASK_ID\",\"node_id\":\"$NODE_ID\"}" >/dev/null

# ── 5. Start Flower client ─────────────────────────────────────────
info "Starting Flower client (20 local steps per round)..."
info "This dev run executes 3 federation rounds."

set +e
NO_4BIT=true python3 client_node.py \
  --server_address localhost:8080 \
  --model_name ./toy_moe_model \
  --expert_indices $EXPERTS \
  --dataset_dir "$DATASET_DIR" \
  --checkpoint_dir ./checkpoints/node_dev \
  --compute_score 0.3 \
  --local_steps 200 \
  --checkpoint_every 5 \
  --batch_size 2 \
  --lr 3e-3 \
  --node_id "$NODE_ID" \
  --heartbeat_url http://localhost:8000/heartbeat
CLIENT_RC=$?
set -e

if [[ $CLIENT_RC -eq 0 ]]; then
  curl -sf -X POST http://localhost:8000/task/done \
    -H "Content-Type: application/json" \
    -d "{\"task_id\":\"$TASK_ID\",\"node_id\":\"$NODE_ID\",\"result_ref\":\"./checkpoints/node_dev\"}" >/dev/null
  info "Task $TASK_ID marked DONE."
else
  curl -sf -X POST http://localhost:8000/task/failed \
    -H "Content-Type: application/json" \
    -d "{\"task_id\":\"$TASK_ID\",\"node_id\":\"$NODE_ID\",\"reason\":\"Flower client exited with code $CLIENT_RC\"}" >/dev/null || true
  die "Flower client failed with exit code $CLIENT_RC"
fi

# ── 6. Final status ────────────────────────────────────────────────
info "Federation complete. Final status:"
curl -sf http://localhost:8000/status | python3 -m json.tool
info "Done. Checkpoints saved in ./checkpoints/node_dev/"
