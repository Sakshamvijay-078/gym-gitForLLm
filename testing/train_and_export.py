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


def train_briefly(model: nn.Module, x: torch.Tensor, y: torch.Tensor, classes=None, steps: int = 200):
    """
    If `classes` is given, the loss is MASKED to only those classes: the
    softmax is restricted to `classes` and `y` is remapped into that local
    index space before computing cross-entropy.

    Why this matters: a branch that only ever sees a subset of classes but
    is trained with a plain full-width CrossEntropyLoss doesn't just learn
    its own classes -- it actively pushes every OTHER class's logit down at
    every step, since softmax normalizes over all outputs. Over enough steps
    that suppresses the classes this branch has never seen. When you later
    merge that branch back in, even a heavily confidence-weighted partner
    branch can't out-vote an actively suppressed output unit, and the merge
    result silently loses that partner's classes (this was the root cause of
    a 0% accuracy bug on the held-out classes in earlier versions of the
    Split-MNIST test notebook -- see the notebook's markdown header). Masking
    the loss means a branch's gradient never touches classes it has no data
    for, so it can't suppress them.
    """
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    classes_t = torch.tensor(sorted(classes)) if classes is not None else None
    for _ in range(steps):
        opt.zero_grad()
        logits = model(x)
        if classes_t is not None:
            logits = logits[:, classes_t]
            y_step = torch.searchsorted(classes_t, y)
        else:
            y_step = y
        loss = loss_fn(logits, y_step)
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
    train_briefly(branch_a, x_a, y_a, classes=[0, 1, 2, 3, 4])
    save_model(branch_a, "branch_a.safetensors")

    # --- Step 3: fine-tune the SAME root on synthetic "shard B" (classes 5-9) ---
    x_b = torch.randn(256, 784)
    y_b = torch.randint(5, 10, (256,))
    branch_b = TinyMLP()
    branch_b.load_state_dict(root.state_dict())
    train_briefly(branch_b, x_b, y_b, classes=[5, 6, 7, 8, 9])
    save_model(branch_b, "branch_b.safetensors")

    print("\nNow run these through the gym CLI:")
    print("  gym init")
    print("  gym commit --file root.safetensors --node seed --round 0")
    print("  gym commit --file branch_a.safetensors --node nodeA --round 1")
    print("  gym commit --file branch_b.safetensors --node nodeB --round 1 --parent <root-hash>")
    print("  gym merge <branchA-hash> <branchB-hash> --strategy ties --node merger --round 2 --out merged.safetensors")
    print("\nThen evaluate the merged model:")
    print("  python evaluate.py merged.safetensors")
