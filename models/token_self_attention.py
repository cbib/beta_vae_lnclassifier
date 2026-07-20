#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/token_self_attention.py

Multi-head self-attention layer operating among the 20 sub-group tokens,
applied before cross-modal attention in BetaVAESubgroup.

Motivation
----------
The current architecture processes each sub-group independently through
its projector MLP, with inter-sub-group interaction only through the
cross-modal attention (where the sequence encoder queries the tokens).
The cross-modal attention is a weighted sum — it routes through tokens
but does not allow tokens to exchange information with each other.

The M* within-class covariance analysis revealed class-specific TE
sub-group covariance structure:
  - lncRNA: TE_PSEUDO + TE_UNKNOWN + TE_GLOBAL co-vary
  - mRNA:   TE_CORE + TE_LCTR + TE_GLOBAL co-vary

This structure is invisible to the current architecture. A token
self-attention layer allows sub-group tokens to attend to each other
before the sequence queries them, enabling the model to learn
inter-sub-group relationships from data — including the class-specific
TE covariance structure — without hard-coding a biological prior.

Architecture (per layer)
------------------------
  tokens_in  : (B, 20, d_proj)
  tokens_out = LayerNorm(tokens_in + MHA(tokens_in, tokens_in, tokens_in))
  tokens_out = LayerNorm(tokens_out + FFN(tokens_out))

Standard pre-norm transformer block. One or two layers configurable.

Relationship to existing architecture
--------------------------------------
The existing CrossModalAttentionMH handles sequence→token attention.
TokenSelfAttention handles token→token attention and runs upstream of it.
They are separate modules with no shared weights.

References
----------
Attention is All You Need: Vaswani et al. (2017)
  https://arxiv.org/abs/1706.03762
Perceiver: Jaegle et al. (2021) — cross-attention preceded by self-attention
  https://arxiv.org/abs/2103.03206
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Single transformer block for token self-attention
# ---------------------------------------------------------------------------

class TokenTransformerBlock(nn.Module):
    """
    One pre-norm transformer block: MHA + FFN with residual connections.

    Parameters
    ----------
    d_model  : token dimensionality (= d_proj)
    n_heads  : number of self-attention heads
    ffn_mult : FFN hidden dim multiplier (default 2 → hidden = 2 * d_model)
    dropout  : dropout rate on attention weights and FFN
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        ffn_mult: int   = 2,
        dropout:  float = 0.1,
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.d_model = d_model

        # Pre-norm MHA
        self.norm1  = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        # Pre-norm FFN
        ffn_hidden  = d_model * ffn_mult
        self.norm2  = nn.LayerNorm(d_model)
        self.ffn    = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model),
            nn.Dropout(dropout),
        )

        # Near-zero init for query/key to start with uniform attention
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, d_model)  — N = 20 sub-group tokens

        Returns
        -------
        x : (B, N, d_model)
        """
        B, N, D = x.shape
        H, d_h  = self.n_heads, self.d_head

        # ── MHA block ────────────────────────────────────────────────────────
        x_norm = self.norm1(x)
        Q = self.q_proj(x_norm).view(B, N, H, d_h).transpose(1, 2)  # (B,H,N,d_h)
        K = self.k_proj(x_norm).view(B, N, H, d_h).transpose(1, 2)
        V = self.v_proj(x_norm).view(B, N, H, d_h).transpose(1, 2)

        scale   = d_h ** 0.5
        scores  = torch.matmul(Q, K.transpose(-2, -1)) / scale      # (B,H,N,N)
        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        context = torch.matmul(weights, V)                           # (B,H,N,d_h)
        context = context.transpose(1, 2).contiguous().view(B, N, D)
        x       = x + self.o_proj(context)

        # ── FFN block ─────────────────────────────────────────────────────────
        x = x + self.ffn(self.norm2(x))

        return x


# ---------------------------------------------------------------------------
# TokenSelfAttention: stacked transformer blocks
# ---------------------------------------------------------------------------

class TokenSelfAttention(nn.Module):
    """
    Stacked self-attention among sub-group tokens.

    Applied after token_norm and before cross-modal attention in
    BetaVAESubgroup.encode(). Allows sub-group tokens to exchange
    information and learn inter-sub-group relationships from data.

    Parameters
    ----------
    d_proj    : token dimensionality (must match SubgroupProjectionLayer d_proj)
    n_heads   : number of self-attention heads per layer (default 4)
    n_layers  : number of transformer blocks (1 or 2 recommended)
    ffn_mult  : FFN hidden dim multiplier (default 2)
    dropout   : dropout rate

    Usage in BetaVAESubgroup
    ------------------------
    # In __init__ (when use_token_sa=True):
    self.token_sa = TokenSelfAttention(
        d_proj   = d_proj,
        n_heads  = token_sa_heads,
        n_layers = token_sa_layers,
        dropout  = fusion_dropout,
    )

    # In encode(), after token_norm:
    feature_tokens = self.token_norm(feature_tokens)
    if self.use_token_sa:
        feature_tokens = self.token_sa(feature_tokens)
    h_attended, attn_weights = self.cross_attn(h_seq, feature_tokens)
    """

    def __init__(
        self,
        d_proj:   int,
        n_heads:  int   = 4,
        n_layers: int   = 1,
        ffn_mult: int   = 2,
        dropout:  float = 0.1,
    ) -> None:
        super().__init__()

        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        self.d_proj   = d_proj
        self.n_heads  = n_heads
        self.n_layers = n_layers

        self.blocks = nn.ModuleList([
            TokenTransformerBlock(
                d_model  = d_proj,
                n_heads  = n_heads,
                ffn_mult = ffn_mult,
                dropout  = dropout,
            )
            for _ in range(n_layers)
        ])

        # Final LayerNorm after all blocks
        self.norm_out = nn.LayerNorm(d_proj)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens : (B, N_tokens, d_proj)

        Returns
        -------
        tokens : (B, N_tokens, d_proj)  — updated token representations
        """
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm_out(tokens)

    def extra_repr(self) -> str:
        return (f"d_proj={self.d_proj}, n_heads={self.n_heads}, "
                f"n_layers={self.n_layers}")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, N, D = 8, 20, 64

    print("=" * 55)
    print("TokenSelfAttention self-test")
    print("=" * 55)

    for n_layers in [1, 2]:
        print(f"\n── {n_layers} layer(s) ──")
        tsa = TokenSelfAttention(d_proj=D, n_heads=4, n_layers=n_layers)

        params = sum(p.numel() for p in tsa.parameters())
        print(f"  Parameters: {params:,}")

        x   = torch.randn(B, N, D)
        out = tsa(x)
        assert out.shape == (B, N, D), f"Shape mismatch: {out.shape}"
        print(f"  Output shape: {out.shape}  ")

        # Residual — output should differ from input (not identity)
        diff = (out - x).abs().mean().item()
        assert diff > 1e-4, "Output identical to input — check residual"
        print(f"  Output differs from input (mean |diff|={diff:.4f})  ")

        # Gradient flow
        out.sum().backward()
        no_grad = [n for n, p in tsa.named_parameters() if p.grad is None]
        assert not no_grad, f"No gradient for: {no_grad}"
        print(f"  Gradients: all parameters  ")

    print("\nAll tests passed.")