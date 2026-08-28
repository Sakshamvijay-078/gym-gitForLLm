#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_dev_local.sh — Run the full federated system NATIVELY (no Docker)
#                    Optimised for U-series CPU + 8 GB RAM
#
# Prerequisites (install once):
#   pip install flwr fastapi uvicorn redis pydantic numpy requests \
#               torch transformers datasets peft accelerate safetensors \
#               tokenizers tqdm
#   sudo apt install redis-server   # or: brew install redis
#
# Estimated time on U-series (Intel Core i5/i7 U, 8 GB RAM):
#   Step 1 (model create):     ~10 seconds
#   Step 2 (data pipeline):    ~3-6 minutes (200 chunks, CPU)
#   Step 3 (orchestrator up):  ~5 seconds
#   Step 4 (1 training round): ~3-8 minutes (20 steps, batch=2)
#   Step 5 (3 rounds total):   ~10-25 minutes end-to-end
#   ──────────────────────────────────────────────────
#   TOTAL for a full 3-round test: ~15-35 minutes
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()   { echo -e "${RED}[ERR]${NC}  $*" >&2; exit 1; }

# ── 0. Sanity checks ─────────────────────────────────────────────
python3 -c "import torch, flwr, fastapi, redis, peft" 2>/dev/null || \
  die "Missing Python packages. Run: pip install flwr fastapi uvicorn redis pydantic numpy requests torch transformers datasets peft accelerate safetensors tokenizers tqdm"

redis-cli ping &>/dev/null || {
  warn "Redis not running. Starting redis-server in background..."
  redis-server --daemonize yes --loglevel warning
  sleep 1
}
info "Redis: OK"

# ── 1. Create ToyMoE model ────────────────────────────────────────
if [ ! -d "./toy_moe_model" ]; then
  info "Creating ToyMoE model (~5M params, ~20MB)..."
  NO_4BIT=true python3 dev/toy_moe.py --save_path ./toy_moe_model
else
  info "ToyMoE model already exists at ./toy_moe_model"
fi

# ── 2. Run data bucketing pipeline ───────────────────────────────
if [ ! -f "./buckets/manifest.json" ]; then
  info "Running data pipeline (200 chunks of wikitext-2, CPU)..."
  info "Estimated time: 3-6 minutes..."
  NO_4BIT=true python3 data_pipeline.py \
    --model_name_or_path ./toy_moe_model \
    --dataset_name wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --output_dir ./buckets \
    --top_k 2 \
    --max_chunks 200 \
    --batch_size 4 \
    --no_4bit
  info "Data pipeline complete. Buckets written to ./buckets/"
else
  info "Buckets already exist. Skipping pipeline."
fi

# ── 3. Start orchestrator ─────────────────────────────────────────
info "Starting orchestrator (FastAPI + Flower server)..."
REDIS_HOST=localhost \
BUCKET_DIR=./buckets \
FL_ROUNDS=3 \
FL_MIN_CLIENTS=1 \
FLOWER_PORT=8080 \
API_PORT=8000 \
  python3 orchestrator.py &
ORCH_PID=$!
sleep 4   # Let it initialise

# Health check
for i in $(seq 1 10); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    info "Orchestrator ready at http://localhost:8000"
    break
  fi
  sleep 2
  [ $i -eq 10 ] && die "Orchestrator failed to start"
done

# ── 4. Register node and get task assignment ──────────────────────
info "Registering dev node..."
curl -sf -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d '{"node_id": "node_dev", "ram_gb": 5.0, "compute_score": 0.3}' | python3 -m json.tool

TASK_JSON=$(curl -sf http://localhost:8000/assign_task/node_dev)
echo "Task assigned: $TASK_JSON"

DATASET_DIR=$(echo "$TASK_JSON" | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(d.get('dataset_dir','./buckets/path_0_1'))")
EXPERTS=$(echo "$TASK_JSON" | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(' '.join(map(str,d.get('bucket_path',[0,1]))))")

info "Assigned dataset: $DATASET_DIR"
info "Assigned experts: $EXPERTS"

# ── 5. Start Flower client ────────────────────────────────────────
info "Starting Flower client (20 local steps, ~3-8 min per round)..."
info "This will run 3 federation rounds. Estimated total: 10-25 minutes."
NO_4BIT=true python3 client_node.py \
  --server_address   localhost:8080 \
  --model_name       ./toy_moe_model \
  --expert_indices   $EXPERTS \
  --dataset_dir      "$DATASET_DIR" \
  --checkpoint_dir   ./checkpoints/node_dev \
  --compute_score    0.3 \
  --local_steps      20 \
  --checkpoint_every 5 \
  --batch_size       2 \
  --lr               2e-4 \
  --node_id          node_dev \
  --heartbeat_url    http://localhost:8000/heartbeat

# ── 6. Final status ───────────────────────────────────────────────
info "Federation complete. Final status:"
curl -sf http://localhost:8000/status | python3 -m json.tool

# Cleanup
kill $ORCH_PID 2>/dev/null && info "Orchestrator stopped."
info "Done. Checkpoints saved in ./checkpoints/node_dev/"
