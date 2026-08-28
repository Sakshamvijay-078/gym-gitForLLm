# Gym: Git for LLMs & Confidence-Weighted Model Merging

Gym is a prototype version control and model merging system—essentially **"Git for LLMs"**. It provides a content-addressable blob store, manifest lineage graph, and pluggable model merging strategies built from the ground up to be format-agnostic. 

Currently, the project focuses heavily on solving **catastrophic performance degradation in imbalanced model merging**. It implements an advanced **Confidence-Weighted Merging** strategy (specifically using a "norm-equalized" task-vector scaling approach). This ensures that starved branches (models trained on very little data) can still contribute meaningful signals rather than being drowned out by dominant branches with larger parameter updates.

---

## 📂 Project Structure: What Things Do What

The repository is divided into TypeScript core packages for the version control system and Python scripts for training, testing, and benchmarking the models.

### 1. `packages/core-versioning` (The Engine)
This is the core library that powers the version control and merging math. 
* **BlobStore & ManifestStore**: Manages content-addressed storage (like Git's `.git/objects`) and the lineage graph (commits, parents, branches).
* **Model Codecs**: Converts binary files (currently heavily supporting `.safetensors` for HuggingFace compatibility) into in-memory `ModelWeights` (tensors) that can be merged.
* **Merge Strategies**: Implements various pluggable algorithms to merge weights:
  * `average`: Plain elementwise mean.
  * `task-arithmetic`: `base + λ·Σ(model − base)`.
  * `ties`: Trims to top-k magnitude, elects signs, and merges agreeing values.
  * `slerp`: Spherical interpolation for exactly two models.
  * `confidence-weighted`: Uses branch training signals (dataset size, metrics) to weight branches. Handles extreme data imbalances gracefully.

### 2. `packages/cli` (The Interface)
A Git-like command-line wrapper over `core-versioning`. It allows you to run commands anywhere inside the project.
* Commands include: `gym init`, `gym commit`, `gym log`, `gym checkout`, `gym merge`, `gym status`, etc.
* Allows passing training signals during commits (e.g., `--dataset-size`, `--metric`) which the `confidence-weighted` strategy later uses during merges.

### 3. `gym_colab_cw_v3.py` (The Benchmark Harness)
An end-to-end Python script that tests the Gym CLI and merge strategies on a **Split-MNIST** dataset with severe data imbalance (e.g., 30,596 samples vs. 300 samples). 
* It automatically initializes the store, trains sub-models on different shards of MNIST, commits them via the CLI, merges them using various strategies, and prints a comparative benchmark of the results.

### 4. `testing/` (Reference Tools)
* `train_and_export.py`: A minimal, clean example of how to train a PyTorch `TinyMLP` on synthetic data shards, carefully masking classes to prevent suppression, and saving the outputs as `.safetensors`.
* `evaluate.py`: A script to evaluate merged `.safetensors` checkpoints on synthetic test data to verify that the merge successfully preserved knowledge from both branches.

---

## 🚀 How it Works: A Process Walkthrough

Here is a step-by-step walkthrough of the typical Gym workflow, demonstrating how models are versioned and merged:

### Step 1: Initialization and the Base Model
Just like Git, you start by initializing a repository. A base (untrained or pre-trained) model is created and saved.
1. Run `gym init` to create the `.gym_colab_store` (the underlying blob storage).
2. Train or define a base model (e.g., a PyTorch MLP) and save it as `root.safetensors`.
3. Commit the base model: 
   `gym commit --file root.safetensors --node root --round 0`

### Step 2: Branching and Decentralized Training
Imagine two different users fine-tuning the base model on different, highly imbalanced datasets.
1. **Branch A** loads the `root` model and fine-tunes it on a massive dataset (e.g., MNIST digits 0-4). Saves as `branch_a.safetensors`.
2. **Branch B** loads the same `root` model but fine-tunes on a highly "starved" dataset (e.g., just 300 samples of MNIST digits 5-9). Saves as `branch_b.safetensors`.

### Step 3: Committing with Confidence Signals
When committing these new branches, we provide metadata that the confidence-weighted merger will use later.
* **Commit Branch A**:
  `gym commit --file branch_a.safetensors --node branch_a --round 1 --parent <root_hash> --dataset-size 30000`
* **Commit Branch B**:
  `gym commit --file branch_b.safetensors --node branch_b --round 1 --parent <root_hash> --dataset-size 300`

### Step 4: The Merge
Now, we want to combine both branches into a single model that knows all classes (0-9). Standard averaging would cause Branch A's massive updates to completely destroy Branch B's fragile knowledge.

Instead, we use Gym's advanced merge strategy:
* `gym merge <hash_A> <hash_B> --strategy confidence-weighted --score-mode norm-equalized --base <root_hash> --out merged.safetensors`

**What happens under the hood?**
The `core-versioning` library loads the `.safetensors`, calculates the task vectors (Model - Base), and applies **Norm-Equalized Task Vector Scaling**. It rescales the task vectors so that the "starved" Branch B contributes meaningful signal, electing parameter signs intelligently (TIES-style) while weighting by dataset size and metrics.

### Step 5: Evaluation
Finally, the resulting `merged.safetensors` is loaded back into Python (via `evaluate.py` or the benchmark script). Because we used confidence-weighted merging, the resulting model successfully maintains high accuracy on both the dominant classes (0-4) and the starved classes (5-9), effectively bypassing the catastrophic forgetting typically seen in naive model merging!
