"""
Minimal reference for testing gym's merge pipeline against a REAL trained
model instead of synthetic vectors. Not part of the gym codebase itself —
this is what runs on your machine to produce the .safetensors files that
`gym commit` / `gym merge` then operate on.

Requires: pip install torch safetensors torchvision
"""

import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file

torch.manual_seed(0)


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def save_model(model: nn.Module, path: str):
    # safetensors requires contiguous tensors; state_dict() values from a
    # freshly trained model already are, but .contiguous() is cheap insurance.
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state, path)
    print(f"saved {path}")


def load_model(path: str) -> TinyMLP:
    model = TinyMLP()
    model.load_state_dict(load_file(path))
    return model


def train_briefly(model: nn.Module, x: torch.Tensor, y: torch.Tensor, steps: int = 200):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    return model


if __name__ == "__main__":
    # --- Step 1: root checkpoint, untrained ---
    root = TinyMLP()
    save_model(root, "root.safetensors")

    # --- Step 2: fine-tune on synthetic "shard A" (replace with real MNIST subset) ---
    x_a = torch.randn(256, 784)
    y_a = torch.randint(0, 5, (256,))  # pretend this shard only has classes 0-4
    branch_a = TinyMLP()
    branch_a.load_state_dict(root.state_dict())
    train_briefly(branch_a, x_a, y_a)
    save_model(branch_a, "branch_a.safetensors")

    # --- Step 3: fine-tune the SAME root on synthetic "shard B" (classes 5-9) ---
    x_b = torch.randn(256, 784)
    y_b = torch.randint(5, 10, (256,))
    branch_b = TinyMLP()
    branch_b.load_state_dict(root.state_dict())
    train_briefly(branch_b, x_b, y_b)
    save_model(branch_b, "branch_b.safetensors")

    print("\nNow run these through the gym CLI:")
    print("  gym init")
    print("  gym commit --file root.safetensors --node seed --round 0")
    print("  gym commit --file branch_a.safetensors --node nodeA --round 1")
    print("  gym commit --file branch_b.safetensors --node nodeB --round 1 --parent <root-hash>")
    print("  gym merge <branchA-hash> <branchB-hash> --strategy ties --node merger --round 2 --out merged.safetensors")
    print("\nThen evaluate the merged model:")
    print("  python evaluate.py merged.safetensors")
