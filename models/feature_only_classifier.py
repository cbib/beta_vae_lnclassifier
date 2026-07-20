#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/feature_only_classifier.py

Sequence-free classifier using only sub-group feature tokens.

Purpose
-------
Ablation of BetaVAESubgroup that removes the CNN sequence encoder entirely.
Used to test whether subgroup dominance in the full model is caused by
sequence-feature synergy or by an intrinsic feature routing bias in the
attention mechanism.

Architecture
------------
No sequence input. No CNN encoder. No VAE. No decoder.

  {block}_genomic + {block}_processed  → SubgroupProjectionLayer per block
                                                              ↓
                               token_norm → (B, N_tokens, d_proj)
                                                              ↓
         class_token (learned) → CrossModalAttentionMH ← feature tokens
              (B, 1, d_proj)     query=class_token          key/value=tokens
                                                              ↓
                               attended class token (B, 1, d_proj)
                                                              ↓
                    attention-weighted per-block summaries (one per block)
                                                              ↓
                               classifier MLP → logits (B, num_classes)

The class token replaces the sequence encoder output as the attention query.

Output dict keys
----------------
  logits                  (B, num_classes)
  feat_logits_{block}     (B, num_classes)   one per registry block
  feat_logits_nonb        alias for feat_logits_nonb (if "nonb" in registry)
  feat_logits_te          alias for feat_logits_te   (if "te" in registry)
  attn_weights            (B, num_heads, 1, N_tokens)
  mu                      (B, d_proj * (1 + n_blocks))  attended_cls + block summaries
  z                       same as mu
  reconstruction          zeros (B, 1, 1) — placeholder for trainer compat
  logvar                  zeros (B, mu_dim) — placeholder
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.feature_registry import REGISTRY, FeatureRegistry, BLOCK_DEFS
from models.subgroup_projector import SubgroupProjectionLayer
from models.beta_vae_subgroup import CrossModalAttentionMH


# ---------------------------------------------------------------------------
# FeatureOnlyClassifier
# ---------------------------------------------------------------------------

class FeatureOnlyClassifier(nn.Module):
    """
    Sequence-free sub-group token classifier with dynamic feature blocks.

    Parameters
    ----------
    num_classes        : int
    d_proj              : int    token dimensionality (default 64)
    fusion_dropout      : float  dropout inside projector MLPs
    attn_heads          : int    number of attention heads (default 4)
    attn_dropout        : float
    attn_temperature    : float  initial softmax temperature
    attn_mode           : str    "standard" | "role_mask" | "orthogonal"
    classifier_hidden   : List[int]
    dropout_rate        : float  dropout in classifier MLP
    registry            : FeatureRegistry
    te_features_dim      : Optional[int]  accepted for config backward compat, unused
    nonb_features_dim    : Optional[int]  accepted for config backward compat, unused
    """

    def __init__(
        self,
        num_classes:        int,
        d_proj:              int             = 64,
        fusion_dropout:      float           = 0.1,
        attn_heads:          int             = 4,
        attn_dropout:        float           = 0.1,
        attn_temperature:    float           = 4.0,
        attn_mode:           str             = "standard",
        classifier_hidden:   List[int]       = None,
        dropout_rate:        float           = 0.3,
        registry:            FeatureRegistry = REGISTRY,
        # Accepted for backward compat with old configs; not used internally
        te_features_dim:     Optional[int]   = None,
        nonb_features_dim:   Optional[int]   = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if classifier_hidden is None:
            classifier_hidden = [128]

        self.num_classes = num_classes
        self.d_proj      = d_proj
        self.attn_mode   = attn_mode
        self._registry   = registry

        # ── Dynamic sub-group projectors — one per registry block ─────────────
        self.projectors = nn.ModuleDict({
            block_name: SubgroupProjectionLayer(
                block    = block_name,
                d_proj   = d_proj,
                dropout  = fusion_dropout,
                registry = registry,
            )
            for block_name in registry.block_names
        })

        self.n_tokens = registry.total_tokens   # dynamic (15 → 20 with nonb2)

        # Per-block token count for attention summary split
        self._block_token_counts: Dict[str, int] = {
            b: len(registry.block_subgroups(b))
            for b in registry.block_names
        }

        # Token LayerNorm
        self.token_norm = nn.LayerNorm(d_proj)

        # ── Learned class token (replaces CNN encoder output as query) ─────────
        self.class_token = nn.Parameter(torch.zeros(1, 1, d_proj))
        nn.init.trunc_normal_(self.class_token, std=0.02)

        # ── Cross-modal attention ─────────────────────────────────────────────
        # d_model = d_proj so the class token and feature tokens share the
        # same space — no dimension mismatch
        self.cross_attn = CrossModalAttentionMH(
            d_model     = d_proj,
            d_feat      = d_proj,
            num_heads   = attn_heads,
            dropout     = attn_dropout,
            temperature = attn_temperature,
            attn_mode   = attn_mode,
        )

        # ── Classifier ────────────────────────────────────────────────────────
        # attended_cls (d_proj) + one summary per block (d_proj each)
        n_blocks     = len(registry.block_names)
        combined_dim = d_proj * (1 + n_blocks)

        def _make_head(in_dim: int) -> nn.Sequential:
            layers: List[nn.Module] = []
            prev = in_dim
            for h in classifier_hidden:
                layers += [
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
                prev = h
            layers.append(nn.Linear(prev, num_classes))
            return nn.Sequential(*layers)

        self.classifier = _make_head(combined_dim)

        # Per-block auxiliary classifiers
        self.feat_classifiers = nn.ModuleDict({
            block_name: _make_head(d_proj)
            for block_name in registry.block_names
        })

        # Backward compat shims
        if "nonb" in registry.block_names:
            self.feat_classifier_nonb = self.feat_classifiers["nonb"]
        if "te" in registry.block_names:
            self.feat_classifier_te = self.feat_classifiers["te"]

        # ── Compatibility shims for probing and analysis scripts ───────────────
        # extract_representations.py hooks model.fc_mu to capture z.
        self.fc_mu = nn.Linear(combined_dim, combined_dim, bias=False)
        nn.init.eye_(self.fc_mu.weight)
        self.fc_mu.weight.requires_grad = False   # frozen — should stay identity

        # Expose encoded_length=1 and latent_dim for trainer/script compat
        self.encoded_length = 1
        self.latent_dim     = combined_dim
        self.conv_channels  = d_proj   # used in some analysis scripts

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        x:               Optional[torch.Tensor] = None,   # ignored — no seq encoder
        te_genomic:      Optional[torch.Tensor] = None,
        te_processed:    Optional[torch.Tensor] = None,
        nonb_genomic:    Optional[torch.Tensor] = None,
        nonb_processed:  Optional[torch.Tensor] = None,
        nonb2:           Optional[torch.Tensor] = None,
        te_features:     Optional[torch.Tensor] = None,   # legacy fallback
        nonb_features:   Optional[torch.Tensor] = None,
        deterministic:   bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        # Input resolution — legacy fallback
        if te_genomic   is None: te_genomic   = te_features
        if te_processed is None: te_processed = te_genomic
        if nonb_genomic is None: nonb_genomic = nonb_features
        if nonb_processed is None: nonb_processed = nonb_genomic

        if te_genomic is None or nonb_genomic is None:
            raise ValueError(
                "FeatureOnlyClassifier requires te_genomic and nonb_genomic inputs."
            )

        B = te_genomic.shape[0]

        # ── Build block_inputs dict for all registry blocks ───────────────────
        block_inputs: Dict[str, Tuple[Optional[torch.Tensor],
                                      Optional[torch.Tensor]]] = {}
        for block_name in self._registry.block_names:
            if block_name == "nonb":
                block_inputs[block_name] = (nonb_genomic, nonb_processed)
            elif block_name == "te":
                block_inputs[block_name] = (te_genomic, te_processed)
            elif block_name == "nonb2":
                block_inputs[block_name] = (None, nonb2) if nonb2 is not None else None
            else:
                g = kwargs.get(f"{block_name}_genomic")
                p = kwargs.get(f"{block_name}_processed",
                               kwargs.get(block_name))
                block_inputs[block_name] = (g, p)

        # ── Project all blocks, skipping absent ones ──────────────────────────
        all_tokens:   list[torch.Tensor]       = []
        block_tokens: dict[str, torch.Tensor]  = {}

        for block_name in self._registry.block_names:
            inputs = block_inputs[block_name]
            if inputs is None:
                # Block absent from this run — inject zero tokens
                n_sg = len(self._registry.block_subgroups(block_name))
                zero = torch.zeros(B, n_sg, self.d_proj,
                                   device=te_genomic.device)
                block_tokens[block_name] = zero
                all_tokens.append(zero)
                continue
            x_g, x_p = inputs
            tok = self.projectors[block_name](x_g, x_p)
            block_tokens[block_name] = tok
            all_tokens.append(tok)

        feature_tokens = torch.cat(all_tokens, dim=1)   # (B, N_total, d_proj)
        feature_tokens = self.token_norm(feature_tokens)

        # ── Class token query ─────────────────────────────────────────────────
        query = self.class_token.expand(B, -1, -1)   # (B, 1, d_proj)

        # ── Cross-modal attention ─────────────────────────────────────────────
        attended, attn_weights = self.cross_attn(query, feature_tokens)
        # attended: (B, 1, d_proj)  attn_weights: (B, H, 1, N_total)

        # ── Per-block attention-weighted summaries ────────────────────────────
        attn_mean = attn_weights.mean(dim=(1, 2))   # (B, N_total)
        block_summaries: Dict[str, torch.Tensor] = {}
        offset = 0
        for block_name in self._registry.block_names:
            n_sg   = self._block_token_counts[block_name]
            b_attn = attn_mean[:, offset:offset + n_sg].unsqueeze(-1)
            b_tok  = block_tokens[block_name]
            block_summaries[block_name] = (b_tok * b_attn).sum(dim=1)
            offset += n_sg

        # Attended class token
        attended_cls = attended.squeeze(1)   # (B, d_proj)

        # ── z for probing scripts (via fc_mu hook) ────────────────────────────
        combined = torch.cat([attended_cls] + list(block_summaries.values()), dim=1)
        z        = self.fc_mu(combined)   # identity

        # ── Classification ────────────────────────────────────────────────────
        logits = self.classifier(combined)

        out: Dict[str, torch.Tensor] = {
            "logits":         logits,
            "reconstruction": torch.zeros(B, 1, 1, device=te_genomic.device),
            "mu":             z,
            "logvar":         torch.zeros_like(z),
            "z":              z,
            "attn_weights":   attn_weights,
            "projection":     z,
        }

        for block_name, summary in block_summaries.items():
            out[f"feat_logits_{block_name}"] = self.feat_classifiers[block_name](summary)

        return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data.feature_registry import REGISTRY

    B      = 4
    D_PROJ = 64

    print("=" * 65)
    print("FeatureOnlyClassifier self-test")
    print("=" * 65)
    print(f"Registry blocks: {REGISTRY.block_names}")
    print(f"Total tokens   : {REGISTRY.total_tokens}")

    model = FeatureOnlyClassifier(
        num_classes       = 2,
        d_proj            = D_PROJ,
        attn_heads        = 4,
        attn_mode         = "standard",
        classifier_hidden = [128],
    )

    total = sum(p.numel() for p in model.parameters())
    print(f"\nParameters : {total:,}")
    print(f"n_tokens   : {model.n_tokens}")
    print(f"latent_dim : {model.latent_dim}")

    te_g  = torch.randn(B, REGISTRY.te_dim)
    te_p  = torch.randn(B, REGISTRY.te_dim)
    nb_g  = torch.randn(B, REGISTRY.nonb_dim)
    nb_p  = torch.randn(B, REGISTRY.nonb_dim)
    nb2   = torch.randn(B, REGISTRY.nonb2_dim)

    out = model(te_genomic=te_g, te_processed=te_p,
                nonb_genomic=nb_g, nonb_processed=nb_p,
                nonb2=nb2)

    assert out["logits"].shape == (B, 2)
    assert out["attn_weights"].shape == (B, 4, 1, model.n_tokens)
    assert "feat_logits_nonb"  in out
    assert "feat_logits_te"    in out
    assert "feat_logits_nonb2" in out

    print(f"\nlogits           : {out['logits'].shape}  ")
    print(f"attn_weights     : {out['attn_weights'].shape}    "
          f"(B, H, 1 query, {model.n_tokens} tokens)")
    print(f"mu               : {out['mu'].shape}  ")
    print(f"feat_logits_nonb2: {out['feat_logits_nonb2'].shape}  ")

    loss = (out["logits"].sum()
            + out["feat_logits_nonb"].sum()
            + out["feat_logits_te"].sum()
            + out["feat_logits_nonb2"].sum())
    loss.backward()
    no_grad = [n for n, p in model.named_parameters()
               if p.grad is None and p.requires_grad]
    assert not no_grad, f"No gradient for: {no_grad}"
    print("gradients        : all trainable parameters have grad  ")

    with torch.no_grad():
        x_test = torch.randn(2, model.latent_dim)
        diff   = (model.fc_mu(x_test) - x_test).abs().max().item()
        assert diff < 1e-5, f"fc_mu is not identity (max diff={diff:.2e})"
    print("fc_mu            : identity (frozen)  ")

    print("\nSelf-test passed.")