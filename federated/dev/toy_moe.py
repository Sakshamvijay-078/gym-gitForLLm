"""
dev/toy_moe.py — Tiny HuggingFace-compatible MoE model for CPU testing.

• ~5M parameters, ~20MB in fp32, loads in <2s on any CPU.
• Has `.gate` Linear modules on each MoE block — RouterCapture hooks
  attach identically to how they attach on real Mixtral.
• Fully compatible with PEFT LoRA (standard nn.Linear internals).
• Registers with AutoModelForCausalLM so existing pipeline code is unchanged.

Usage:
  python toy_moe.py --save_path ./toy_moe_model
  # Then point MODEL_NAME=./toy_moe_model in .env.dev
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    PretrainedConfig, PreTrainedModel,
    AutoConfig, AutoModelForCausalLM,
    AutoTokenizer, GPT2Tokenizer,
)
from transformers.modeling_outputs import CausalLMOutput


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

class ToyMoEConfig(PretrainedConfig):
    model_type = "toy_moe"

    def __init__(
        self,
        vocab_size: int   = 32_000,    # Same as Mixtral tokenizer for drop-in compat
        hidden_size: int  = 256,
        num_layers: int   = 2,
        num_heads: int    = 4,
        num_experts: int  = 8,         # Mimic Mixtral's 8 experts
        top_k: int        = 2,
        expert_hidden: int = 512,
        max_position_embeddings: int = 512,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        self.vocab_size               = vocab_size
        self.hidden_size              = hidden_size
        self.num_layers               = num_layers
        self.num_heads                = num_heads
        self.num_experts              = num_experts
        self.top_k                    = top_k
        self.expert_hidden            = expert_hidden
        self.max_position_embeddings  = max_position_embeddings
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Model Modules
# ──────────────────────────────────────────────────────────────────────────────

class ToyExpert(nn.Module):
    """Single FFN expert: up → gelu → down."""
    def __init__(self, hidden: int, expert_hidden: int):
        super().__init__()
        self.w1 = nn.Linear(hidden, expert_hidden, bias=False)
        self.w2 = nn.Linear(expert_hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


class ToyMoEBlock(nn.Module):
    """
    Sparse MoE layer with a top-K router.

    The `.gate` attribute is a plain nn.Linear — RouterCapture hooks
    it exactly as it hooks Mixtral's gate modules.
    """
    def __init__(self, config: ToyMoEConfig):
        super().__init__()
        self.top_k = config.top_k
        # ── This is the module RouterCapture will hook ──
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            ToyExpert(config.hidden_size, config.expert_hidden)
            for _ in range(config.num_experts)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        B, T, H = x.shape

        # Gate: (B, T, num_experts)
        gate_logits = self.gate(x)
        gate_probs  = F.softmax(gate_logits, dim=-1)
        top_vals, top_idx = torch.topk(gate_probs, self.top_k, dim=-1)
        top_vals = top_vals / (top_vals.sum(dim=-1, keepdim=True) + 1e-9)

        out = torch.zeros_like(x)
        for k in range(self.top_k):
            idx = top_idx[..., k]          # (B, T)
            w   = top_vals[..., k:k+1]     # (B, T, 1)
            # Route each token to its assigned expert (simplified: loop over experts)
            for e_idx in range(len(self.experts)):
                mask = (idx == e_idx).unsqueeze(-1).float()  # (B, T, 1)
                if mask.sum() == 0:
                    continue
                expert_out = self.experts[e_idx](x)           # (B, T, H)
                out += mask * w * expert_out

        return residual + out


class ToyAttention(nn.Module):
    """Minimal multi-head self-attention."""
    def __init__(self, config: ToyMoEConfig):
        super().__init__()
        H, nh = config.hidden_size, config.num_heads
        assert H % nh == 0
        self.nh   = nh
        self.head = H // nh
        self.q_proj = nn.Linear(H, H, bias=False)
        self.k_proj = nn.Linear(H, H, bias=False)
        self.v_proj = nn.Linear(H, H, bias=False)
        self.o_proj = nn.Linear(H, H, bias=False)
        self.norm   = nn.LayerNorm(H)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        B, T, H = x.shape
        def split_heads(t):
            return t.view(B, T, self.nh, self.head).transpose(1, 2)
        Q, K, V = split_heads(self.q_proj(x)), split_heads(self.k_proj(x)), split_heads(self.v_proj(x))
        scores = (Q @ K.transpose(-2, -1)) / (self.head ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        # Causal mask
        causal = torch.ones(T, T, device=x.device).tril()
        scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out  = (attn @ V).transpose(1, 2).reshape(B, T, H)
        return residual + self.o_proj(out)


class ToyMoELayer(nn.Module):
    """One transformer layer = attention + MoE FFN."""
    def __init__(self, config: ToyMoEConfig):
        super().__init__()
        self.attn = ToyAttention(config)
        self.moe  = ToyMoEBlock(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.moe(x)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Full Model (HuggingFace PreTrainedModel)
# ──────────────────────────────────────────────────────────────────────────────

class ToyMoEForCausalLM(PreTrainedModel):
    """
    Tiny MoE causal LM compatible with:
      • AutoModelForCausalLM.from_pretrained(path)
      • PEFT LoRA (attaches to q_proj, v_proj, w1, w2 inside MoE layers)
      • RouterCapture (hooks .gate modules)
      • HuggingFace Trainer / DataCollatorForLanguageModeling
    """
    config_class = ToyMoEConfig
    supports_gradient_checkpointing = False

    def __init__(self, config: ToyMoEConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_pos    = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layers       = nn.ModuleList([ToyMoELayer(config) for _ in range(config.num_layers)])
        self.norm         = nn.LayerNorm(config.hidden_size)
        self.lm_head      = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutput:
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x    = self.embed_tokens(input_ids) + self.embed_pos(pos)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}


# ──────────────────────────────────────────────────────────────────────────────
# Registration + Save
# ──────────────────────────────────────────────────────────────────────────────

def register_toy_moe():
    """Register ToyMoE with HuggingFace AutoClasses so from_pretrained() works."""
    AutoConfig.register("toy_moe", ToyMoEConfig)
    AutoModelForCausalLM.register(ToyMoEConfig, ToyMoEForCausalLM)


def create_and_save(save_path: str) -> None:
    """Create a tiny toy model + GPT-2 tokenizer and save to disk."""
    register_toy_moe()

    cfg = ToyMoEConfig(
        vocab_size   = 32_000,
        hidden_size  = 256,
        num_layers   = 2,
        num_heads    = 4,
        num_experts  = 8,
        top_k        = 2,
        expert_hidden = 512,
        max_position_embeddings = 512,
    )
    model = ToyMoEForCausalLM(cfg)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"ToyMoE: {param_count:,} parameters ({param_count*4/1e6:.1f} MB fp32)")

    path = Path(save_path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    cfg.save_pretrained(str(path))

    # Reuse GPT-2 tokenizer (same BPE interface, compatible vocab size wrapper)
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.save_pretrained(str(path))
    print(f"Saved ToyMoE to: {path.resolve()}")
    print("Set MODEL_NAME=./toy_moe_model in your .env.dev")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--save_path", default="./toy_moe_model")
    args = p.parse_args()
    create_and_save(args.save_path)
