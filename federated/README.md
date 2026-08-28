# Federated MoE Fine-Tuning — Phase 1

> **Decentralized Federated Fine-Tuning of Mixture-of-Experts LLMs on Heterogeneous, Fault-Prone Infrastructure**

This subdirectory implements Phase 1: a robust prototype that lets a heterogeneous, unreliable fleet of machines collaboratively fine-tune a shared MoE model (e.g., Mixtral-8x7B) via a custom [Flower](https://flower.ai) strategy.

---

## 📂 Directory Tree

```
federated/
├── data_pipeline.py          # Offline bucketing: routes chunks → expert paths, scores purity
├── redis_state.py            # Redis-backed state machine: nodes, tasks, heartbeats, FSM
├── orchestrator.py           # FastAPI service + Flower gRPC server + watchdog
├── custom_strategy.py        # Flower Strategy: FlexLoRA merge, reward scoring, rollback
├── client_node.py            # Flower client: quantized MoE + path-selective LoRA + DiLoCo
│
├── docker-compose.yml        # Compose file orchestrating all 4 services
├── Dockerfile.orchestrator   # Lightweight CPU image for orchestrator
├── Dockerfile.pipeline       # CUDA 12.1 image for one-shot data pipeline
├── Dockerfile.client         # CUDA 12.1 image for each client node
│
├── requirements.txt          # CPU-only deps (orchestrator)
├── requirements-gpu.txt      # Full GPU deps (pipeline + client)
└── .env.example              # Template for environment variables
```

### Role of Each File

| File | Role |
|---|---|
| `data_pipeline.py` | Routes sentence chunks through a **frozen** MoE router via forward-hook capture; groups by top-K expert path; scores path-purity; emits ranked Arrow datasets |
| `redis_state.py` | Atomic task FSM (`pending→assigned→training→done/failed`), node registry, heartbeat TTL watchdog, capacity-aware assignment |
| `orchestrator.py` | FastAPI REST API (`/register`, `/heartbeat`, `/assign_task`, `/task/*`, `/status`); launches Flower gRPC server + watchdog in daemon threads |
| `custom_strategy.py` | Custom `fl.server.strategy.Strategy`; FlexLoRA SVD reconstruction + weighted merge; sigmoid-bounded confidence weights with `confidenceFloor`; rollback on global regression |
| `client_node.py` | Flower `Client`; 4-bit quantized base model; path-selective LoRA (rank from compute score); DiLoCo local training loop; crash-resume from checkpoints; sends delta + confidence |
| `docker-compose.yml` | Wires all services; GPU reservations; named volumes; health-check-gated startup |
| `Dockerfile.*` | Separate images for orchestrator (slim), pipeline (CUDA), client (CUDA+devel) |

---

## 🏗️ Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                         Docker Network                              │
│                                                                    │
│   ┌──────────────┐    ┌─────────────────────────────────────────┐  │
│   │    Redis     │◄───│           Orchestrator                  │  │
│   │  (task FSM   │    │  ┌─────────────────┐  ┌─────────────┐  │  │
│   │  heartbeats  │    │  │  FastAPI :8000  │  │ Flower gRPC │  │  │
│   │  node reg.)  │    │  │  /register      │  │   :8080     │  │  │
│   └──────────────┘    │  │  /heartbeat     │  │             │  │  │
│                        │  │  /assign_task   │  │ MoEFederated│  │  │
│   ┌──────────────┐    │  │  /task/done     │  │  Strategy   │  │  │
│   │ Data Pipeline│    │  │  /status        │  │  (FlexLoRA) │  │  │
│   │ (one-shot)   │    │  └─────────────────┘  └─────────────┘  │  │
│   │              │    └─────────────────────────────────────────┘  │
│   │ Routes text →│                  ▲  ▲  ▲                       │
│   │ expert paths │        ┌─────────┘  │  └──────────┐            │
│   │ Writes Arrow │        │            │             │             │
│   │ datasets     │   ┌────┴───┐  ┌────┴───┐  ┌─────┴───┐        │
│   └──────────────┘   │Client 0│  │Client 1│  │Client N │        │
│                       │Path[0,1]  │Path[2,3]  │Path[4,5]│        │
│                       │rank=4  │  │rank=16 │  │rank=32  │        │
│                       │DiLoCo  │  │DiLoCo  │  │DiLoCo   │        │
│                       │ckpt/   │  │ckpt/   │  │ckpt/    │        │
│                       └────────┘  └────────┘  └─────────┘        │
└────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Execution Instructions

### Prerequisites

- Docker ≥ 24 + Docker Compose plugin
- NVIDIA Container Toolkit (for GPU services)
- A HuggingFace token with access to the gated model
- At least 1 GPU per pipeline/client container

### 1. Configure environment

```bash
cd federated/
cp .env.example .env
# Edit .env: set MODEL_NAME, HF_TOKEN, etc.
```

### 2. Run the data bucketing pipeline (one-time)

This step routes your dataset through the frozen model's MoE router to create per-expert-path datasets. It exits when done.

```bash
docker compose --profile pipeline up --build pipeline
# Datasets written to the `bucket_data` Docker volume
```

### 3. Start the orchestrator and Redis

```bash
docker compose up --build orchestrator redis -d
```

Wait for the health check:
```bash
docker compose ps        # orchestrator should show "healthy"
curl http://localhost:8000/status
```

### 4. Launch client nodes

Each client is a separate worker. Override `NODE_ID`, `EXPERT_INDICES`, and `COMPUTE_SCORE` per instance.

**Single client (quick test):**
```bash
NODE_ID=node_0 EXPERT_INDICES="0 1" COMPUTE_SCORE=0.5 \
  docker compose up --build client
```

**Multiple heterogeneous clients (different machines or terminals):**
```bash
# High-resource node, experts 0+1
NODE_ID=node_gpu0 EXPERT_INDICES="0 1" COMPUTE_SCORE=0.9 \
  docker compose run --rm client

# Low-resource node, experts 2+3
NODE_ID=node_gpu1 EXPERT_INDICES="2 3" COMPUTE_SCORE=0.3 \
  docker compose run --rm client
```

**Scale identically-configured clients:**
```bash
EXPERT_INDICES="0 1" COMPUTE_SCORE=0.5 \
  docker compose up --build --scale client=4 client
```

### 5. Monitor

```bash
# Task queue depths and node health
curl http://localhost:8000/status | python3 -m json.tool

# Orchestrator logs (watchdog, rounds, rollbacks)
docker compose logs -f orchestrator
```

---

## ⚙️ Key Configuration Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `mistralai/Mixtral-8x7B-Instruct-v0.1` | HuggingFace model ID |
| `DATASET_NAME` | `HuggingFaceH4/ultrachat_200k` | Dataset for bucketing |
| `TOP_K` | `2` | Number of top experts per routing decision |
| `PURITY_THRESHOLD` | `0.0` | Min routing purity to include a chunk |
| `MAX_CHUNKS` | `50000` | Max chunks to route in pipeline |
| `FL_ROUNDS` | `10` | Number of Flower federation rounds |
| `FL_MIN_CLIENTS` | `2` | Minimum clients required per round |
| `COMPUTE_SCORE` | `0.5` | Node compute budget `[0,1]` → LoRA rank |
| `LOCAL_STEPS` | `200` | DiLoCo local SGD steps per round |
| `CHECKPOINT_EVERY` | `50` | Steps between crash-recovery checkpoints |

---

## 🔬 Technical Highlights

### Path-Purity Bucketing (`data_pipeline.py`)
Chunks are routed through the frozen MoE gate layers via `register_forward_hook`. Purity = Σ routing_prob for top-K experts. Only high-purity chunks are assigned to fine-tune those experts, ensuring the adapter learns a clean signal.

### FlexLoRA (`custom_strategy.py` + `client_node.py`)
Clients submit LoRA adapters at their compute-appropriate rank (4–32). The server reconstructs full-rank deltas via SVD (`W = B @ A`), merges them as weighted sums in delta space, then re-decomposes at the global canonical rank (16). This enables rank-heterogeneous federated learning.

### Confidence-Weighted Aggregation (`custom_strategy.py`)
```
reward = normalised_loss_improvement − regression_penalty
weight = confidenceFloor + (1 − confidenceFloor) × sigmoid(reward)
```
Updates with `reward < threshold` are rejected. If the merged global model degrades on the validation buffer by >5%, the orchestrator rolls back to the last known-good parameters.

### DiLoCo Local Training (`client_node.py`)
Nodes run 200 local SGD steps before communicating (vs. FedAvg's 1 step). Local checkpoints are saved every 50 steps — on restart, training resumes from the latest checkpoint without losing progress.

### Fault Handling (`redis_state.py` + `orchestrator.py`)
A background watchdog checks all node `last_seen` timestamps every 15 s. Nodes that exceed the `HEARTBEAT_TTL_S=30 s` TTL have their tasks automatically re-enqueued (up to 3 retries before marking permanently failed).

---

## 🗂️ Task FSM

```
          ┌──────────┐
          │  PENDING │◄──────────────────────────────────┐
          └────┬─────┘                                   │
               │ assign_task_to_node()                   │
               ▼                                         │
          ┌──────────┐                          requeue (retry < 3)
          │ ASSIGNED │──── node timeout ─────────────────┘
          └────┬─────┘
               │ /task/training
               ▼
          ┌──────────┐
          │ TRAINING │──── node timeout ─────────────────┐
          └────┬─────┘                                   │
          ┌────┴─────┐                                   │
     ┌────┤          ├────┐                              │
     ▼    └──────────┘    ▼                              │
  ┌──────┐           ┌──────────┐                        │
  │ DONE │           │  FAILED  │◄───────────────────────┘
  └──────┘           └──────────┘  (retries ≥ 3)
```
