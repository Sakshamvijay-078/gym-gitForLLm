#!/usr/bin/env python3
"""
Split-MNIST Benchmark Harness for Gym Confidence-Weighted Model Merging
========================================================================
Tests model merging strategies under severe data imbalance (30,596 vs 300 samples).
Includes the new Norm-Equalized Task Vector Merging (`--score-mode norm-equalized`),
which prevents high-sample branch magnitude dominance over starved branches.
"""

import os
import sys
import subprocess
import shutil
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths & Setup
WORK_DIR = os.path.abspath(".")
STORE_DIR = os.path.join(WORK_DIR, ".gym_colab_store")
OUT_DIR = os.path.join(WORK_DIR, "merged_models")

if os.path.exists(STORE_DIR):
    shutil.rmtree(STORE_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# CLI Runner Helper
def run_gym(args):
    cmd = ["npx", "tsx", "packages/cli/src/index.ts"] + args
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=WORK_DIR)
    if res.returncode != 0:
        print(f"Error running gym CLI: {' '.join(args)}")
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
        sys.exit(1)
    return res.stdout.strip()

# Initialize Gym Store
os.environ["GYM_DIR"] = STORE_DIR
run_gym(["init"])
print(f"Initialized Gym store at {STORE_DIR}")

# Data Loading
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_full  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

def indices_for_classes(dataset, classes):
    targets = dataset.targets
    mask = torch.isin(targets, torch.tensor(classes))
    return torch.where(mask)[0].tolist()

shard_a_idx = indices_for_classes(train_full, [0, 1, 2, 3, 4])
shard_b_idx = indices_for_classes(train_full, [5, 6, 7, 8, 9])

test_a_idx = indices_for_classes(test_full, [0, 1, 2, 3, 4])
test_b_idx = indices_for_classes(test_full, [5, 6, 7, 8, 9])

shard_a = Subset(train_full, shard_a_idx)
shard_b_full = Subset(train_full, shard_b_idx)

# Starved branch B: 300 samples
torch.manual_seed(42)
starved_b_idx = torch.randperm(len(shard_b_full))[:300].tolist()
shard_b_starved = Subset(shard_b_full, starved_b_idx)

test_a = Subset(test_full, test_a_idx)
test_b = Subset(test_full, test_b_idx)

print(f"Shard A (digits 0-4): {len(shard_a)} samples")
print(f"Shard B Starved (digits 5-9): {len(shard_b_starved)} samples")

# Architecture
class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc2(torch.relu(self.fc1(x)))

def train_on_shard(model, dataset, epochs=3, lr=1e-3, batch_size=128):
    model.to(device)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
    return model

@torch.no_grad()
def eval_acc(model, dataset):
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=256)
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total

# Training
torch.manual_seed(0)
root_model = TinyMLP()

# Train Branch A
branch_a = TinyMLP()
branch_a.load_state_dict(root_model.state_dict())
train_on_shard(branch_a, shard_a, epochs=3)

# Train Branch B (Starved)
branch_b = TinyMLP()
branch_b.load_state_dict(root_model.state_dict())
train_on_shard(branch_b, shard_b_starved, epochs=5)

# Save safetensors & Commit
os.makedirs("models", exist_ok=True)
from safetensors.torch import save_file, load_file

def save_model(model, path):
    save_file(model.state_dict(), path)

save_model(root_model, "models/root.safetensors")
save_model(branch_a, "models/branch_a.safetensors")
save_model(branch_b, "models/branch_b.safetensors")

def parse_hash(stdout):
    for line in stdout.splitlines():
        if "[" in line and "]" in line:
            parts = line.split()
            if len(parts) >= 3:
                return parts[2]
    return stdout.strip().split()[-1]

root_hash = parse_hash(run_gym(["commit", "--file", "models/root.safetensors", "--node", "root", "--round", "0"]))
hash_a = parse_hash(run_gym(["commit", "--file", "models/branch_a.safetensors", "--node", "branch_a", "--round", "1", "--parent", root_hash, "--dataset-size", str(len(shard_a)), "--metric", f"{eval_acc(branch_a, test_a):.4f}"]))
hash_b = parse_hash(run_gym(["commit", "--file", "models/branch_b.safetensors", "--node", "branch_b", "--round", "1", "--parent", root_hash, "--dataset-size", str(len(shard_b_starved)), "--metric", f"{eval_acc(branch_b, test_b):.4f}"]))

print(f"Committed Root: {root_hash[:8]}")
print(f"Committed Branch A: {hash_a[:8]}")
print(f"Committed Branch B: {hash_b[:8]}")

# Merging Experiments
experiments = [
    ("average (imbalanced)", ["--strategy", "average"]),
    ("confidence-weighted (sqrt)", ["--strategy", "confidence-weighted", "--score-mode", "sqrt"]),
    ("ties (imbalanced)", ["--strategy", "ties", "--base", root_hash]),
    ("cw-ties (imbalanced)", ["--strategy", "confidence-weighted", "--base", root_hash, "--score-mode", "sqrt"]),
    ("cw-tv-sqrt (imbalanced)", ["--strategy", "confidence-weighted", "--base", root_hash, "--ties", "false", "--score-mode", "sqrt"]),
    ("cw-tv-equal (imbalanced)", ["--strategy", "confidence-weighted", "--base", root_hash, "--ties", "false", "--score-mode", "equal"]),
    ("cw-tv-fisher (imbalanced)", ["--strategy", "confidence-weighted", "--base", root_hash, "--ties", "false", "--score-mode", "delta-norm"]),
    ("cw-tv-norm-equalized (imbalanced)", ["--strategy", "confidence-weighted", "--base", root_hash, "--ties", "false", "--score-mode", "norm-equalized", "--norm-equalize-power", "0.6"]),
]

results = []

# Evaluate standalone models
acc_a_04 = eval_acc(branch_a, test_a)
acc_a_59 = eval_acc(branch_a, test_b)
acc_a_full = eval_acc(branch_a, test_full)
results.append(("branch A alone", acc_a_04, acc_a_59, acc_a_full))

acc_b_04 = eval_acc(branch_b, test_a)
acc_b_59 = eval_acc(branch_b, test_b)
acc_b_full = eval_acc(branch_b, test_full)
results.append(("branch B starved alone", acc_b_04, acc_b_59, acc_b_full))

for name, flags in experiments:
    out_path = os.path.join(OUT_DIR, f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.safetensors")
    args = [hash_a, hash_b, "--node", "merged", "--round", "2", "--out", out_path] + flags
    run_gym(["merge"] + args)
    
    # Load merged model & eval
    m = TinyMLP()
    m.load_state_dict(load_file(out_path))
    
    acc_04 = eval_acc(m, test_a)
    acc_59 = eval_acc(m, test_b)
    acc_full = eval_acc(m, test_full)
    results.append((name, acc_04, acc_59, acc_full))

print("\n" + "="*70)
print("KEY COMPARISON — Imbalanced case (30,596 vs 300 samples)")
print("="*70)
print(f"{'Strategy':<38} {'digits 0-4':>12} {'digits 5-9':>12} {'full':>10}")
print("-" * 70)
for name, a, b, f in results:
    print(f"{name:<38} {a:>11.2%} {b:>11.2%} {f:>9.2%}")
print("="*70)
