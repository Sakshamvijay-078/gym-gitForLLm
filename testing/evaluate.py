"""
Evaluate a merged (or any) checkpoint on both synthetic shards, to see
whether the merge actually combined what each branch learned — the real
test, not just "the pipeline ran without crashing."

Usage: python evaluate.py <path-to-safetensors-file>
"""

import sys
import torch
from safetensors.torch import load_file
from train_and_export import TinyMLP

path = sys.argv[1] if len(sys.argv) > 1 else "merged.safetensors"

model = TinyMLP()
model.load_state_dict(load_file(path))
model.eval()

torch.manual_seed(1)  # different seed than training, held-out-ish data
x_a = torch.randn(128, 784)
y_a = torch.randint(0, 5, (128,))
x_b = torch.randn(128, 784)
y_b = torch.randint(5, 10, (128,))

with torch.no_grad():
    acc_a = (model(x_a).argmax(dim=1) == y_a).float().mean().item()
    acc_b = (model(x_b).argmax(dim=1) == y_b).float().mean().item()

print(f"{path}")
print(f"  accuracy on shard A (classes 0-4): {acc_a:.2%}")
print(f"  accuracy on shard B (classes 5-9): {acc_b:.2%}")
print(f"  (a merge that only learned from one branch will show a big gap here)")
