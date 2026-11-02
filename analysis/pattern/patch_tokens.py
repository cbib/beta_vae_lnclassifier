#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis/pattern/patch_tokens.py

Token-level activation patching for BetaVAESubgroup: matched-pair interchange
interventions over several scopes, with partner and noise controls.

Patch modes (--patch_modes, default single,block,all)
-----------------------------------------------------
  single    one subgroup token.
  block     all tokens of one block (te / nonb / nonb2).
  all       all 20 subgroup tokens (ceiling case; scope recorded as "ALL").
  shuffled  partner = random opposite-class transcript (same confidence
            filter, no length/GC match); tests partner matching.
  sameclass partner = another transcript of the baseline's own class; its
            flip rate and |mean delta_logit| are the noise floor.
  noise     partner = isotropic random vector with the per-token L2 norm of
            the shuffled partner (same baselines -> paired comparison).
Matched pairs are shared by single/block/all; shuffled, sameclass and noise
use independent RNG streams. --null_scopes (default single,block,all) sets
the scopes the three controls cover.

Patch site (--patch_site)
-------------------------
  projector   (default) patch model.projectors[block], i.e. the raw token
              before it is stored for the per-block summaries and passed to
              token_norm, so the patch reaches both the cross-attention/z
              path and the summary path.
  token_norm  legacy single-path hook, for comparison only; do not mix its
              output with projector-site output.
Every row carries patch_site. An existing fold CSV is reused only if its
patch_site matches; legacy CSVs (no column) are refused, so write to a
fresh --output_dir.

Output (<output_dir>/)
----------------------
  fold_N_patching.csv, all_folds_patching.csv (long format), and
  patching_summary.csv (cross-fold mean/std per patch_mode, scope, direction).
  Columns: patch_mode, patch_scope (subgroup | block | "ALL"), scope_level,
  partner_dataset_idx, patch_site.

Usage
-----
python analysis/pattern/patch_tokens.py \\
    --experiment_dir EXP --config CFG --output_dir OUT \\
    --n_pairs 200 --patch_modes single,block,all,shuffled,sameclass,noise
"""

from __future__ import annotations
from data.cv_utils import get_cv_splits_from_config

import argparse
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from configs.load_config import load_config
from models.model_builder import create_model_builder
from data.cv_utils import load_sequences_in_order, create_length_stratified_groups
from data.gated_feature_dataset import SequenceFeatureDataset
from data.feature_registry import REGISTRY

# Where interchange interventions are applied (see module docstring).
PATCH_SITES = ("projector", "token_norm")
DEFAULT_PATCH_SITE = "projector"


# ---------------------------------------------------------------------------
# Pair selection (identical logic to patch_tokens.py)
# ---------------------------------------------------------------------------

def select_pairs(
    val_indices:    np.ndarray,
    labels:         list,
    lengths:        np.ndarray,
    confidences:    np.ndarray,
    n_pairs:        int,
    min_confidence: float,
    max_length_diff: float,
    max_gc_diff:    float,
    gc_content:     Optional[np.ndarray],
    rng:            np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Select matched (lncRNA, mRNA) pairs from the val set. Same semantics
    as patch_tokens.py::select_pairs."""
    label_arr = np.array([labels[i] for i in val_indices])
    conf_arr  = confidences
    len_arr   = lengths

    lnc_mask  = (label_arr == "lnc") & (conf_arr >= min_confidence)
    mrna_mask = (label_arr == "pc")  & (conf_arr >= min_confidence)

    lnc_pool  = np.where(lnc_mask)[0]
    mrna_pool = np.where(mrna_mask)[0]

    if len(lnc_pool) < 10 or len(mrna_pool) < 10:
        warnings.warn(
            f"Very few high-confidence samples: lnc={len(lnc_pool)}, "
            f"mrna={len(mrna_pool)}. Lowering min_confidence or checking "
            f"fold checkpoint."
        )

    rng.shuffle(lnc_pool)
    matched_lnc  = []
    matched_mrna = []
    mrna_used    = set()

    for li in lnc_pool:
        if len(matched_lnc) >= n_pairs:
            break

        l_len = len_arr[li]
        l_gc  = gc_content[li] if gc_content is not None else None

        len_diffs = np.abs(len_arr[mrna_pool] - l_len) / max(l_len, 1.0)
        candidates = mrna_pool[len_diffs <= max_length_diff]

        if l_gc is not None and gc_content is not None and len(candidates) > 0:
            gc_diffs   = np.abs(gc_content[candidates] - l_gc)
            candidates = candidates[gc_diffs <= max_gc_diff]

        candidates = [c for c in candidates if c not in mrna_used]

        if len(candidates) == 0:
            continue

        chosen = rng.choice(candidates)
        matched_lnc.append(li)
        matched_mrna.append(chosen)
        mrna_used.add(chosen)

    if len(matched_lnc) < n_pairs:
        warnings.warn(
            f"Only {len(matched_lnc)} pairs matched (requested {n_pairs}). "
            f"Consider relaxing --max_length_diff or --max_gc_diff."
        )

    return np.array(matched_lnc), np.array(matched_mrna)


def select_shuffled_partners(
    val_indices:    np.ndarray,
    labels:         list,
    confidences:    np.ndarray,
    min_confidence: float,
    target_label:   str,
    n:              int,
    rng:            np.random.Generator,
) -> np.ndarray:
    """
    Draw n indices (local positions into val_indices) of high-confidence
    transcripts of target_label ("lnc" or "pc"), with REPLACEMENT, ignoring
    length/GC matching entirely. Used as the null-control patch source.
    """
    label_arr = np.array([labels[i] for i in val_indices])
    mask = (label_arr == target_label) & (confidences >= min_confidence)
    pool = np.where(mask)[0]
    if len(pool) == 0:
        warnings.warn(f"No high-confidence '{target_label}' transcripts for "
                       f"shuffled control.")
        return np.array([], dtype=int)
    return rng.choice(pool, size=n, replace=True)


def select_same_class_partners(
    val_indices:    np.ndarray,
    labels:         list,
    confidences:    np.ndarray,
    min_confidence: float,
    label:          str,
    baseline_idx:   np.ndarray,
    rng:            np.random.Generator,
) -> np.ndarray:
    """Per baseline index (class `label`), a different high-confidence
    same-class transcript, drawn with replacement, no length/GC match."""
    local = select_shuffled_partners(
        val_indices=val_indices, labels=labels, confidences=confidences,
        min_confidence=min_confidence, target_label=label,
        n=len(baseline_idx), rng=rng)
    if len(local) == 0:
        return np.array([], dtype=int)
    partners = val_indices[local]
    clash = partners == np.asarray(baseline_idx)
    if clash.any():  # a self-patch is the identity; shift to a neighbour draw
        partners = np.where(clash, np.roll(partners, 1), partners)
    return partners


def make_noise_vectors(partner_vectors: torch.Tensor,
                       generator: torch.Generator) -> torch.Tensor:
    """Isotropic random vectors with the per-token L2 norm of
    `partner_vectors` (n_positions, d_proj); same shape/device/dtype."""
    g = torch.randn(partner_vectors.shape, generator=generator)
    g = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    norms = partner_vectors.detach().norm(dim=-1, keepdim=True).cpu()
    return (g * norms).to(partner_vectors.device, partner_vectors.dtype)


# ---------------------------------------------------------------------------
# Token cache hook
# ---------------------------------------------------------------------------

class TokenCache:
    """Captures token representations after token_norm via forward hook."""

    def __init__(self):
        self.tokens: Optional[torch.Tensor] = None
        self._handle = None

    def register(self, model):
        self._handle = model.token_norm.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self.tokens = output.detach()

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ---------------------------------------------------------------------------
# Projector-site capture / patching
# ---------------------------------------------------------------------------
# Hooking model.projectors[block] intercepts the raw token upstream of both
# the cross-attention path (via token_norm) and the per-block summaries.

def build_token_layout(
    model,
    token_names: list[str],
    block_map:   dict[str, str],
    block_names: list[str],
) -> list[tuple[str, int]]:
    """Map each token position to (block_name, row_within_block_output).
    Raises ValueError if token_names does not follow encode()'s concatenation
    order or the per-block counts disagree."""
    counts = {b: len(REGISTRY.block_subgroups(b)) for b in block_names}
    offsets, running = {}, 0
    for b in block_names:
        offsets[b] = running
        running += counts[b]
    if running != len(token_names):
        raise ValueError(
            f"token_names has {len(token_names)} entries but the registry "
            f"blocks sum to {running} tokens."
        )

    layout: list[tuple[str, int]] = []
    for pos, sg in enumerate(token_names):
        block = block_map[sg]
        row   = list(REGISTRY.block_subgroups(block)).index(sg)
        if offsets[block] + row != pos:
            raise ValueError(
                f"Token '{sg}' sits at position {pos} in token_names but "
                f"encode() concatenates it at {offsets[block] + row} "
                f"(block '{block}', row {row})."
            )
        layout.append((block, row))

    model_counts = getattr(model, "_block_token_counts", None)
    if model_counts is not None:
        got = {b: model_counts[b] for b in block_names}
        if got != counts:
            raise ValueError(
                f"Model block token counts {got} != registry counts {counts}."
            )
    missing = [b for b in block_names if b not in model.projectors]
    if missing:
        raise ValueError(f"model.projectors has no entry for blocks: {missing}")
    return layout


class ProjectorTokenCache:
    """Captures raw (pre-token_norm) tokens from every model.projectors[block],
    concatenated in registry order; `.tokens` is (B, N_tokens, d_proj)."""

    def __init__(self):
        self.tokens: Optional[torch.Tensor] = None
        self._handles: list = []
        self._per_block: dict[str, torch.Tensor] = {}

    def register(self, model):
        self._per_block = {}
        for block in REGISTRY.block_names:
            self._handles.append(
                model.projectors[block].register_forward_hook(
                    self._make_hook(block)))

    def _make_hook(self, block):
        def _hook(module, input, output):
            # output: (B, n_subgroups_in_block, d_proj)
            self._per_block[block] = output.detach()
            if len(self._per_block) == len(REGISTRY.block_names):
                self.tokens = torch.cat(
                    [self._per_block[b] for b in REGISTRY.block_names], dim=1)
                # reset per forward pass so stale blocks are never mixed in
                self._per_block = {}
        return _hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def register_projector_patch_hooks(model, layout, positions, vectors) -> list:
    """
    Register one forward hook per block touched by `positions`, replacing the
    corresponding rows of that block's projector output with `vectors`.

    positions : global token positions (indices into the layout).
    vectors   : (len(positions), d_proj)            shared across the batch
                (batch-size-1 callers) or
                (B, len(positions), d_proj)         one source vector set per
                batch item (batched callers).

    Returns the list of hook handles; the caller must .remove() them.
    """
    by_block: dict[str, tuple[list[int], list[int]]] = {}
    for j, pos in enumerate(positions):
        block, row = layout[pos]
        rows, idxs = by_block.setdefault(block, ([], []))
        rows.append(row)
        idxs.append(j)

    handles = []
    for block, (rows, idxs) in by_block.items():
        n_rows = len(REGISTRY.block_subgroups(block))

        def _hook(module, input, output, _rows=rows, _idxs=idxs,
                  _vecs=vectors, _n=n_rows):
            if output.shape[1] != _n:
                raise RuntimeError(
                    f"Projector output has {output.shape[1]} token rows, "
                    f"expected {_n}.")
            patched = output.clone()
            for r, j in zip(_rows, _idxs):
                patched[:, r, :] = (_vecs[j] if _vecs.dim() == 2
                                    else _vecs[:, j, :])
            return patched

        handles.append(model.projectors[block].register_forward_hook(_hook))
    return handles


# ---------------------------------------------------------------------------
# Generic multi-position patch hook
# ---------------------------------------------------------------------------

def make_patch_hook(positions: list[int], vectors: torch.Tensor):
    """
    Returns a forward-hook function that replaces `output[:, positions, :]`
    with `vectors` (shape (len(positions), d_proj)) at the given token
    positions. Used for single (len(positions)==1), block, and all-token
    (len(positions)==n_tokens) patching alike — one code path for all scopes.
    """
    def _hook(module, input, output):
        patched = output.clone()
        for pos, vec in zip(positions, vectors):
            patched[:, pos, :] = vec
        return patched
    return _hook


# ---------------------------------------------------------------------------
# Per-sample forward helpers
# ---------------------------------------------------------------------------

def build_fwd_kwargs(sample_or_batch, device, has_nonb2, batched: bool):
    """Build the forward() kwargs dict from a dataset sample (unbatched,
    batched=False -> unsqueeze) or an already-batched dict (batched=True)."""
    def prep(x):
        t = x.to(device)
        return t.unsqueeze(0) if not batched else t

    fwd = dict(
        te_genomic     = prep(sample_or_batch["te_genomic"]),
        te_processed   = prep(sample_or_batch["te_processed"]),
        nonb_genomic   = prep(sample_or_batch["nonb_genomic"]),
        nonb_processed = prep(sample_or_batch["nonb_processed"]),
        deterministic  = True,
    )
    if has_nonb2 and "nonb2" in sample_or_batch:
        fwd["nonb2"] = prep(sample_or_batch["nonb2"])
    return fwd


def get_tokens_single(model, dataset, idx: int, device, has_nonb2,
                      patch_site: str = DEFAULT_PATCH_SITE) -> torch.Tensor:
    """(N_tokens, d_proj) tokens for dataset index idx: raw for "projector",
    post-norm for "token_norm"."""
    sample = dataset[idx]
    cache = TokenCache() if patch_site == "token_norm" else ProjectorTokenCache()
    cache.register(model)

    seq = sample["sequence"].unsqueeze(0).to(device)
    fwd = build_fwd_kwargs(sample, device, has_nonb2, batched=False)

    with torch.no_grad():
        model(seq, **fwd)

    cache.remove()
    return cache.tokens[0]


def get_logits_single(model, dataset, idx: int, device, has_nonb2) -> tuple[float, float]:
    """Return (logit_mRNA, prob_mRNA) for dataset index idx, unpatched."""
    sample = dataset[idx]
    seq = sample["sequence"].unsqueeze(0).to(device)
    fwd = build_fwd_kwargs(sample, device, has_nonb2, batched=False)

    with torch.no_grad():
        out   = model(seq, **fwd)
        logit = out["logits"][0, 1].item()
        prob  = torch.softmax(out["logits"], dim=1)[0, 1].item()
    return logit, prob


def get_patched_logits(model, dataset, idx: int, device, has_nonb2,
                        positions: list[int], vectors: torch.Tensor,
                        patch_site: str = DEFAULT_PATCH_SITE,
                        layout=None) -> tuple[float, float]:
    """One forward pass with `positions` replaced by `vectors`
    (len(positions), d_proj); returns (logit_mRNA, prob_mRNA). "projector"
    needs `layout` and raw-token vectors."""
    sample = dataset[idx]
    seq = sample["sequence"].unsqueeze(0).to(device)
    fwd = build_fwd_kwargs(sample, device, has_nonb2, batched=False)

    if patch_site == "token_norm":
        hook_fn = make_patch_hook(positions, vectors)
        handles = [model.token_norm.register_forward_hook(hook_fn)]
    else:
        if layout is None:
            raise ValueError("patch_site='projector' requires `layout` "
                             "(see build_token_layout).")
        handles = register_projector_patch_hooks(model, layout, positions, vectors)
    try:
        with torch.no_grad():
            out = model(seq, **fwd)
    finally:
        for handle in handles:
            handle.remove()

    logit = out["logits"][0, 1].item()
    prob  = torch.softmax(out["logits"], dim=1)[0, 1].item()
    return logit, prob


# ---------------------------------------------------------------------------
# Hook sanity check
# ---------------------------------------------------------------------------

def self_patch_sanity_check(model, dataset, indices, device, has_nonb2,
                            token_names, patch_site, layout, atol=1e-4):
    """Patching a transcript with its own cached tokens must leave its logit
    unchanged (single and all scopes); raises RuntimeError otherwise."""
    all_pos = list(range(len(token_names)))
    for idx in indices:
        toks = get_tokens_single(model, dataset, idx, device, has_nonb2,
                                 patch_site=patch_site)
        base, _ = get_logits_single(model, dataset, idx, device, has_nonb2)
        for positions in ([0], all_pos):
            patched, _ = get_patched_logits(
                model, dataset, idx, device, has_nonb2, positions,
                toks[positions], patch_site=patch_site, layout=layout)
            if abs(patched - base) > atol:
                raise RuntimeError(
                    f"Self-patch check failed (patch_site={patch_site}, "
                    f"idx={idx}, n_positions={len(positions)}): "
                    f"baseline logit {base:.6f} vs self-patched {patched:.6f}.")


# ---------------------------------------------------------------------------
# Per-fold patching, all scopes
# ---------------------------------------------------------------------------

def patch_fold(
    model,
    dataset,
    val_indices:     np.ndarray,
    labels:          list,
    device:          torch.device,
    has_nonb2:       bool,
    n_pairs:         int,
    min_confidence:  float,
    max_length_diff: float,
    max_gc_diff:     float,
    fold_idx:        int,
    token_names:     list[str],
    block_map:       dict[str, str],
    block_names:     list[str],
    patch_modes:     list[str],
    n_shuffled:      int,
    patch_site:      str = DEFAULT_PATCH_SITE,
    null_scopes:     tuple = ("single", "block", "all"),
) -> pd.DataFrame:
    model.eval()

    # token -> (block, row) map for projector-site hooks
    layout = (build_token_layout(model, token_names, block_map, block_names)
              if patch_site == "projector" else None)

    # ── Step 1: metadata pass (confidence, length, GC) ──────────────────────
    print(f"  Collecting val set metadata ({len(val_indices):,} samples)...")

    meta_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=256, shuffle=False, num_workers=0,
    )

    cache = TokenCache()
    cache.register(model)

    all_confs, all_lengths, all_gc = [], [], []

    with torch.no_grad():
        for batch in tqdm(meta_loader, desc="    Metadata pass", leave=False):
            fwd = build_fwd_kwargs(batch, device, has_nonb2, batched=True)
            seq = batch["sequence"].to(device)
            out = model(seq, **fwd)
            probs = torch.softmax(out["logits"], dim=1)
            confs = probs.max(1).values.cpu().numpy()
            all_confs.append(confs)

            seq_np = seq.cpu().numpy()
            lengths = seq_np[:, :4, :].sum(axis=(1, 2)).astype(np.float32)
            all_lengths.append(lengths)

            gc = ((seq[:, 1, :] + seq[:, 2, :]).sum(dim=1)
                  / seq[:, :4, :].sum(dim=(1, 2)).clamp(min=1)).cpu().numpy()
            all_gc.append(gc.astype(np.float32))

    cache.remove()

    confidences = np.concatenate(all_confs)
    lengths     = np.concatenate(all_lengths)
    gc_content  = np.concatenate(all_gc)

    print(f"  Confidence stats: min={confidences.min():.3f}  "
          f"max={confidences.max():.3f}  mean={confidences.mean():.3f}  "
          f">=0.7: {(confidences >= 0.7).sum()}")

    # ── Step 2: matched pairs (shared across single/block/all) ─────────────
    rng = np.random.default_rng(seed=42 + fold_idx)
    lnc_local, mrna_local = select_pairs(
        val_indices=val_indices, labels=labels, lengths=lengths,
        confidences=confidences, n_pairs=n_pairs,
        min_confidence=min_confidence, max_length_diff=max_length_diff,
        max_gc_diff=max_gc_diff, gc_content=gc_content, rng=rng,
    )
    n_actual = len(lnc_local)
    print(f"  Selected {n_actual} matched pairs")
    if n_actual == 0:
        warnings.warn(f"  No pairs matched for fold {fold_idx} — skipping")
        return pd.DataFrame()

    lnc_dataset_idx  = val_indices[lnc_local]
    mrna_dataset_idx = val_indices[mrna_local]

    # ── Step 3: shuffled-null partners (independent RNG stream) ────────────
    shuffled_rng = np.random.default_rng(seed=9042 + fold_idx)
    n_shuf = n_shuffled if n_shuffled is not None else n_pairs
    shuf_mrna_local = select_shuffled_partners(
        val_indices=val_indices, labels=labels, confidences=confidences,
        min_confidence=min_confidence, target_label="pc",
        n=n_shuf, rng=shuffled_rng,
    )
    shuf_lnc_local = select_shuffled_partners(
        val_indices=val_indices, labels=labels, confidences=confidences,
        min_confidence=min_confidence, target_label="lnc",
        n=n_shuf, rng=shuffled_rng,
    )
    shuf_mrna_dataset_idx = (val_indices[shuf_mrna_local]
                              if len(shuf_mrna_local) else np.array([], dtype=int))
    shuf_lnc_dataset_idx  = (val_indices[shuf_lnc_local]
                              if len(shuf_lnc_local) else np.array([], dtype=int))

    # same-class partners (independent RNG); baselines match the shuffled rows
    same_lnc_dataset_idx  = np.array([], dtype=int)
    same_mrna_dataset_idx = np.array([], dtype=int)
    if "sameclass" in patch_modes:
        same_rng = np.random.default_rng(seed=9142 + fold_idx)
        base_l = lnc_dataset_idx[np.arange(n_shuf) % n_actual]
        base_m = mrna_dataset_idx[np.arange(n_shuf) % n_actual]
        same_lnc_dataset_idx = select_same_class_partners(
            val_indices, labels, confidences, min_confidence, "lnc", base_l, same_rng)
        same_mrna_dataset_idx = select_same_class_partners(
            val_indices, labels, confidences, min_confidence, "pc", base_m, same_rng)

    # ── Step 4: cache token representations ─────────────────────────────────
    # Union of every dataset index we'll ever need a token vector from:
    # matched pairs + shuffled partners.
    needed_idx = set(lnc_dataset_idx.tolist()) | set(mrna_dataset_idx.tolist())
    needed_idx |= set(shuf_mrna_dataset_idx.tolist()) | set(shuf_lnc_dataset_idx.tolist())
    needed_idx |= set(same_lnc_dataset_idx.tolist()) | set(same_mrna_dataset_idx.tolist())
    needed_idx = sorted(needed_idx)

    print(f"  Caching token representations for {len(needed_idx)} transcripts...")
    token_cache_map: dict[int, torch.Tensor] = {}
    for idx in tqdm(needed_idx, desc="    Caching tokens", leave=False):
        token_cache_map[idx] = get_tokens_single(model, dataset, idx, device, has_nonb2,
                                                 patch_site=patch_site)

    # verify the hook site / cache pair before any real patching
    self_patch_sanity_check(
        model, dataset, [int(i) for i in lnc_dataset_idx[:2]]
        + [int(i) for i in mrna_dataset_idx[:2]],
        device, has_nonb2, token_names, patch_site, layout)
    print(f"  Self-patch sanity check passed (patch_site={patch_site})")

    n_tokens = len(token_names)
    all_positions = list(range(n_tokens))
    block_positions = {
        b: [i for i, sg in enumerate(token_names) if block_map.get(sg) == b]
        for b in block_names
    }

    rows = []

    scopes: list[tuple[str, str, list[int]]] = []  # (patch_mode, scope_label, positions)
    if "single" in patch_modes:
        for i, sg in enumerate(token_names):
            scopes.append(("single", sg, [i]))
    if "block" in patch_modes:
        for b in block_names:
            if block_positions[b]:
                scopes.append(("block", b, block_positions[b]))
    if "all" in patch_modes:
        scopes.append(("all", "ALL", all_positions))

    print(f"  Patching {len(scopes)} scopes × 2 directions × {n_actual} pairs "
          f"(matched modes)...")

    for patch_mode, scope_label, positions in tqdm(scopes, desc="    Scope loop", leave=False):
        block_label = (block_map.get(scope_label, "unknown")
                       if patch_mode == "single" else scope_label)

        for pair_i in range(n_actual):
            lnc_idx  = lnc_dataset_idx[pair_i]
            mrna_idx = mrna_dataset_idx[pair_i]

            lnc_tok  = token_cache_map[lnc_idx]
            mrna_tok = token_cache_map[mrna_idx]

            # ── lnc_to_mrna ──
            baseline_logit_lnc, baseline_prob_lnc = get_logits_single(
                model, dataset, lnc_idx, device, has_nonb2)
            vectors = mrna_tok[positions]  # (len(positions), d_proj)
            patched_logit_lnc, patched_prob_lnc = get_patched_logits(
                model, dataset, lnc_idx, device, has_nonb2, positions, vectors,
                patch_site=patch_site, layout=layout)

            iia_baseline_valid_lnc = baseline_prob_lnc <= 0.5
            iia_success_lnc = bool(iia_baseline_valid_lnc and patched_prob_lnc > 0.5)

            rows.append({
                "patch_mode": patch_mode, "patch_scope": scope_label,
                "block": block_label, "direction": "lnc_to_mrna",
                "pair_i": pair_i,
                "lnc_dataset_idx": int(lnc_idx), "mrna_dataset_idx": int(mrna_idx),
                "baseline_logit": baseline_logit_lnc, "patched_logit": patched_logit_lnc,
                "delta_logit": patched_logit_lnc - baseline_logit_lnc,
                "baseline_prob": baseline_prob_lnc, "patched_prob": patched_prob_lnc,
                "delta_prob": patched_prob_lnc - baseline_prob_lnc,
                "iia_baseline_valid": iia_baseline_valid_lnc,
                "iia_success": iia_success_lnc,
                "fold": fold_idx,
            })

            # ── mrna_to_lnc ──
            baseline_logit_mrna, baseline_prob_mrna = get_logits_single(
                model, dataset, mrna_idx, device, has_nonb2)
            vectors_r = lnc_tok[positions]
            patched_logit_mrna, patched_prob_mrna = get_patched_logits(
                model, dataset, mrna_idx, device, has_nonb2, positions, vectors_r,
                patch_site=patch_site, layout=layout)

            iia_baseline_valid_mrna = baseline_prob_mrna > 0.5
            iia_success_mrna = bool(iia_baseline_valid_mrna and patched_prob_mrna <= 0.5)

            rows.append({
                "patch_mode": patch_mode, "patch_scope": scope_label,
                "block": block_label, "direction": "mrna_to_lnc",
                "pair_i": pair_i,
                "lnc_dataset_idx": int(lnc_idx), "mrna_dataset_idx": int(mrna_idx),
                "baseline_logit": baseline_logit_mrna, "patched_logit": patched_logit_mrna,
                "delta_logit": patched_logit_mrna - baseline_logit_mrna,
                "baseline_prob": baseline_prob_mrna, "patched_prob": patched_prob_mrna,
                "delta_prob": patched_prob_mrna - baseline_prob_mrna,
                "iia_baseline_valid": iia_baseline_valid_mrna,
                "iia_success": iia_success_mrna,
                "fold": fold_idx,
            })

    # control scopes (--null_scopes)
    null_scope_list: list[tuple[str, str, str, list[int]]] = []  # (label, block, level, positions)
    if "single" in null_scopes:
        for i, sg in enumerate(token_names):
            null_scope_list.append((sg, block_map.get(sg, "unknown"), "single", [i]))
    if "block" in null_scopes:
        for b in block_names:
            if block_positions[b]:
                null_scope_list.append((b, b, "block", block_positions[b]))
    if "all" in null_scopes:
        null_scope_list.append(("ALL", "ALL", "all", all_positions))

    # ── Shuffled-null control: random partner (was: single-token scope only)
    if "shuffled" in patch_modes and len(shuf_mrna_dataset_idx) and len(shuf_lnc_dataset_idx):
        print(f"  Shuffled-null control: {len(null_scope_list)} scopes × 2 directions × "
              f"{len(shuf_mrna_dataset_idx)} draws...")

        for sg, block_label, level, positions in tqdm(
                null_scope_list, desc="    Shuffled scope loop", leave=False):

            for pair_i in range(len(shuf_mrna_dataset_idx)):
                # lnc_to_mrna: baseline is a matched-pool lncRNA (reuse the
                # same lnc transcripts as the matched run for comparability),
                # patch source is a RANDOM mRNA transcript, not its partner.
                lnc_idx = lnc_dataset_idx[pair_i % n_actual]
                rand_mrna_idx = shuf_mrna_dataset_idx[pair_i]

                if rand_mrna_idx not in token_cache_map:
                    token_cache_map[rand_mrna_idx] = get_tokens_single(
                        model, dataset, rand_mrna_idx, device, has_nonb2,
                        patch_site=patch_site)
                rand_mrna_tok = token_cache_map[rand_mrna_idx]

                baseline_logit_lnc, baseline_prob_lnc = get_logits_single(
                    model, dataset, lnc_idx, device, has_nonb2)
                vectors = rand_mrna_tok[positions]
                patched_logit_lnc, patched_prob_lnc = get_patched_logits(
                    model, dataset, lnc_idx, device, has_nonb2, positions, vectors,
                    patch_site=patch_site, layout=layout)

                iia_baseline_valid_lnc = baseline_prob_lnc <= 0.5
                iia_success_lnc = bool(iia_baseline_valid_lnc and patched_prob_lnc > 0.5)

                rows.append({
                    "patch_mode": "shuffled", "patch_scope": sg, "scope_level": level,
                    "block": block_label, "direction": "lnc_to_mrna",
                    "pair_i": pair_i,
                    "lnc_dataset_idx": int(lnc_idx),
                    "mrna_dataset_idx": int(rand_mrna_idx),
                    "baseline_logit": baseline_logit_lnc, "patched_logit": patched_logit_lnc,
                    "delta_logit": patched_logit_lnc - baseline_logit_lnc,
                    "baseline_prob": baseline_prob_lnc, "patched_prob": patched_prob_lnc,
                    "delta_prob": patched_prob_lnc - baseline_prob_lnc,
                    "iia_baseline_valid": iia_baseline_valid_lnc,
                    "iia_success": iia_success_lnc,
                    "fold": fold_idx,
                })

                # mrna_to_lnc: baseline is a matched-pool mRNA, patch source
                # is a random lncRNA transcript.
                mrna_idx = mrna_dataset_idx[pair_i % n_actual]
                rand_lnc_idx = shuf_lnc_dataset_idx[pair_i]

                if rand_lnc_idx not in token_cache_map:
                    token_cache_map[rand_lnc_idx] = get_tokens_single(
                        model, dataset, rand_lnc_idx, device, has_nonb2,
                        patch_site=patch_site)
                rand_lnc_tok = token_cache_map[rand_lnc_idx]

                baseline_logit_mrna, baseline_prob_mrna = get_logits_single(
                    model, dataset, mrna_idx, device, has_nonb2)
                vectors_r = rand_lnc_tok[positions]
                patched_logit_mrna, patched_prob_mrna = get_patched_logits(
                    model, dataset, mrna_idx, device, has_nonb2, positions, vectors_r,
                    patch_site=patch_site, layout=layout)

                iia_baseline_valid_mrna = baseline_prob_mrna > 0.5
                iia_success_mrna = bool(iia_baseline_valid_mrna and patched_prob_mrna <= 0.5)

                rows.append({
                    "patch_mode": "shuffled", "patch_scope": sg, "scope_level": level,
                    "block": block_label, "direction": "mrna_to_lnc",
                    "pair_i": pair_i,
                    "lnc_dataset_idx": int(rand_lnc_idx),
                    "mrna_dataset_idx": int(mrna_idx),
                    "baseline_logit": baseline_logit_mrna, "patched_logit": patched_logit_mrna,
                    "delta_logit": patched_logit_mrna - baseline_logit_mrna,
                    "baseline_prob": baseline_prob_mrna, "patched_prob": patched_prob_mrna,
                    "delta_prob": patched_prob_mrna - baseline_prob_mrna,
                    "iia_baseline_valid": iia_baseline_valid_mrna,
                    "iia_success": iia_success_mrna,
                    "fold": fold_idx,
                })

    # ── same-class and norm-matched-noise controls ──────────────────────────
    # sameclass: iia_success is the spurious flip rate; direction keeps the
    #   baseline class. noise: random vectors with the shuffled partner's
    #   norms (same baselines -> paired with the shuffled rows).
    def _run_control(mode: str):
        if mode == "sameclass":
            part_l, part_m = same_lnc_dataset_idx, same_mrna_dataset_idx
        else:
            part_l, part_m = shuf_mrna_dataset_idx, shuf_lnc_dataset_idx
        n_draw = min(len(part_l), len(part_m))
        if n_draw == 0:
            warnings.warn(f"No partners available for '{mode}' control — skipped.")
            return
        gen = torch.Generator().manual_seed(9242 + fold_idx)
        print(f"  {mode} control: {len(null_scope_list)} scopes × 2 directions × "
              f"{n_draw} draws...")
        for label, block_label, level, positions in tqdm(
                null_scope_list, desc=f"    {mode} scope loop", leave=False):
            for pair_i in range(n_draw):
                for direction in ("lnc_to_mrna", "mrna_to_lnc"):
                    if direction == "lnc_to_mrna":
                        base_idx, part_idx = lnc_dataset_idx[pair_i % n_actual], part_l[pair_i]
                    else:
                        base_idx, part_idx = mrna_dataset_idx[pair_i % n_actual], part_m[pair_i]
                    if part_idx not in token_cache_map:
                        token_cache_map[part_idx] = get_tokens_single(
                            model, dataset, part_idx, device, has_nonb2,
                            patch_site=patch_site)
                    vectors = token_cache_map[part_idx][positions]
                    if mode == "noise":
                        vectors = make_noise_vectors(vectors, gen)

                    bl, bp = get_logits_single(model, dataset, base_idx, device, has_nonb2)
                    pl, pp = get_patched_logits(
                        model, dataset, base_idx, device, has_nonb2, positions,
                        vectors, patch_site=patch_site, layout=layout)
                    if direction == "lnc_to_mrna":
                        valid = bp <= 0.5
                        success = bool(valid and pp > 0.5)
                    else:
                        valid = bp > 0.5
                        success = bool(valid and pp <= 0.5)
                    rows.append({
                        "patch_mode": mode, "patch_scope": label,
                        "scope_level": level, "block": block_label,
                        "direction": direction, "pair_i": pair_i,
                        "lnc_dataset_idx": int(base_idx if direction == "lnc_to_mrna" else part_idx),
                        "mrna_dataset_idx": int(part_idx if direction == "lnc_to_mrna" else base_idx),
                        "partner_dataset_idx": int(part_idx),
                        "baseline_logit": bl, "patched_logit": pl,
                        "delta_logit": pl - bl,
                        "baseline_prob": bp, "patched_prob": pp,
                        "delta_prob": pp - bp,
                        "iia_baseline_valid": bool(valid),
                        "iia_success": success,
                        "fold": fold_idx,
                    })

    for _mode in ("sameclass", "noise"):
        if _mode in patch_modes:
            _run_control(_mode)

    df = pd.DataFrame(rows)
    df["patch_site"] = patch_site
    # scope level and patch-source index on every row
    if "scope_level" not in df.columns:
        df["scope_level"] = np.nan
    df["scope_level"] = df["scope_level"].fillna(df["patch_mode"])
    if "partner_dataset_idx" not in df.columns:
        df["partner_dataset_idx"] = np.nan
    df["partner_dataset_idx"] = df["partner_dataset_idx"].fillna(
        df["mrna_dataset_idx"].where(df["direction"] == "lnc_to_mrna",
                                     df["lnc_dataset_idx"])).astype(int)
    return df


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary(all_folds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-fold mean +/- std of delta_logit / delta_prob, and IIA, grouped by
    (patch_mode, patch_scope, block, direction), plus symmetry_score /
    causal_effect / iia_overall collapsed across direction for each
    (patch_mode, patch_scope).
    """
    agg = (all_folds_df
           .groupby(["patch_mode", "patch_scope", "block", "direction"])
           .agg(
               mean_delta_logit = ("delta_logit", "mean"),
               std_delta_logit  = ("delta_logit", "std"),
               mean_delta_prob  = ("delta_prob",  "mean"),
               std_delta_prob   = ("delta_prob",  "std"),
               n_pairs          = ("delta_logit", "count"),
               n_valid_baseline = ("iia_baseline_valid", "sum"),
               n_iia_success    = ("iia_success", "sum"),
           )
           .reset_index())

    agg["iia"] = agg.apply(
        lambda r: (r["n_iia_success"] / r["n_valid_baseline"])
                  if r["n_valid_baseline"] > 0 else float("nan"),
        axis=1,
    )

    n_folds = all_folds_df["fold"].nunique()
    agg["n_folds"] = n_folds

    pivot = (agg.pivot(index=["patch_mode", "patch_scope", "block"],
                        columns="direction",
                        values="mean_delta_logit")
             .reset_index())
    pivot.columns.name = None
    if "lnc_to_mrna" in pivot.columns and "mrna_to_lnc" in pivot.columns:
        pivot["symmetry_score"] = (
            pivot["lnc_to_mrna"].abs() + pivot["mrna_to_lnc"].abs()
        ) / 2
        pivot["causal_effect"] = (
            pivot["lnc_to_mrna"] - pivot["mrna_to_lnc"]
        ) / 2
        agg = agg.merge(
            pivot[["patch_mode", "patch_scope", "symmetry_score", "causal_effect"]],
            on=["patch_mode", "patch_scope"], how="left",
        )

    iia_pivot = (agg.pivot(index=["patch_mode", "patch_scope", "block"],
                            columns="direction",
                            values="iia")
                 .reset_index())
    iia_pivot.columns.name = None
    if "lnc_to_mrna" in iia_pivot.columns and "mrna_to_lnc" in iia_pivot.columns:
        iia_pivot["iia_overall"] = iia_pivot[
            ["lnc_to_mrna", "mrna_to_lnc"]
        ].mean(axis=1, skipna=True)
        agg = agg.merge(
            iia_pivot[["patch_mode", "patch_scope", "iia_overall"]],
            on=["patch_mode", "patch_scope"], how="left",
        )

    # self-describing summary: downstream scripts refuse mixed patch sites
    if "patch_site" in all_folds_df.columns:
        sites = all_folds_df["patch_site"].dropna().unique()
        agg["patch_site"] = sites[0] if len(sites) == 1 else "mixed"

    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-scope token activation patching (single/block/all) "
                    "+ shuffled-null control for β-LNC causal interpretability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--experiment_dir", required=True,
                        help="Experiment directory containing models/fold_N_best.pt")
    parser.add_argument("--config",         required=True,
                        help="Training config JSON")
    parser.add_argument("--output_dir",     required=True,
                        help="Where to write patching results")
    parser.add_argument("--device",
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_pairs",         type=int,   default=200,
                        help="Matched pairs to sample per fold (default: 200)")
    parser.add_argument("--min_confidence",  type=float, default=0.7,
                        help="Minimum classification confidence for pair selection")
    parser.add_argument("--max_length_diff", type=float, default=0.2,
                        help="Max fractional length difference between pairs")
    parser.add_argument("--max_gc_diff",     type=float, default=0.05,
                        help="Max absolute GC content difference between pairs")
    parser.add_argument("--batch_size",      type=int,   default=None)
    parser.add_argument("--patch_modes",     type=str,   default="single,block,all",
                        help="Comma-separated subset of "
                             "{single,block,all,shuffled,sameclass,noise} "
                             "(default: single,block,all)")
    parser.add_argument("--n_shuffled",      type=int,   default=None,
                        help="Draws per token for the shuffled-null control "
                             "(default: same as --n_pairs)")
    parser.add_argument("--patch_site",      choices=PATCH_SITES,
                        default=DEFAULT_PATCH_SITE,
                        help="Where interventions are applied. 'projector' "
                             "(default) patches model.projectors[block] "
                             "outputs, reaching both the cross-attention/z "
                             "path and the per-block summary path. "
                             "'token_norm' is the LEGACY single-path hook, "
                             "kept for comparison only.")
    parser.add_argument("--null_scopes",     type=str,   default="single,block,all",
                        help="Scopes covered by the control modes (shuffled, "
                             "sameclass, noise), comma-separated subset of "
                             "{single,block,all} (default: all three)")
    args = parser.parse_args()

    null_scopes = [m.strip() for m in args.null_scopes.split(",") if m.strip()]
    if set(null_scopes) - {"single", "block", "all"}:
        raise ValueError(f"Unknown --null_scopes entries: "
                         f"{set(null_scopes) - {'single', 'block', 'all'}}")
    patch_modes = [m.strip() for m in args.patch_modes.split(",") if m.strip()]
    valid_modes = {"single", "block", "all", "shuffled", "sameclass", "noise"}
    unknown = set(patch_modes) - valid_modes
    if unknown:
        raise ValueError(f"Unknown --patch_modes entries: {unknown}. "
                          f"Valid: {valid_modes}")

    config     = load_config(args.config)
    exp_dir    = Path(args.experiment_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device     = torch.device(args.device)

    print("=" * 70)
    print("BetaVAESubgroup — Multi-Scope Token Activation Patching")
    print("=" * 70)
    print(f"Experiment  : {exp_dir}")
    print(f"Config      : {args.config}")
    print(f"Output dir  : {output_dir}")
    print(f"Device      : {args.device}")
    print(f"Pairs/fold  : {args.n_pairs}")
    print(f"Patch modes : {patch_modes}")
    print(f"Patch site  : {args.patch_site}")
    print(f"Min conf    : {args.min_confidence}")
    print(f"Max Δlen    : {args.max_length_diff:.0%}")
    print(f"Max ΔGC     : {args.max_gc_diff}")
    print("=" * 70)

    all_subgroups = REGISTRY.all_subgroups
    token_names   = all_subgroups
    block_map     = {sg: REGISTRY.token_block(sg) for sg in all_subgroups}
    block_names   = list(REGISTRY.block_names)

    print(f"\nRegistry: {REGISTRY.total_tokens} tokens")
    for b in block_names:
        print(f"  {b}: {REGISTRY.block_subgroups(b)}")

    print("\nLoading sequences and dataset...")
    all_sequences, labels = load_sequences_in_order(
        lnc_fasta=config.get("data", "lnc_fasta"),
        pc_fasta= config.get("data", "pc_fasta"),
    )

    has_nonb2    = config.get("data", "nonb2_csv", default=None) is not None
    nonb2_csv    = config.get("data", "nonb2_csv",              default=None)
    nonb2_scaler = config.get("data", "nonb2_scaler_bank_path", default=None)

    if nonb2_csv is not None and nonb2_scaler is None:
        raise ValueError("Config has nonb2_csv but no nonb2_scaler_bank_path.")

    dataset = SequenceFeatureDataset(
        lnc_fasta              = config.get("data", "lnc_fasta"),
        pc_fasta               = config.get("data", "pc_fasta"),
        te_genomic_csv         = config.get("data", "te_genomic_csv"),
        te_processed_csv       = config.get("data", "te_processed_csv",   default=None),
        nonb_genomic_csv       = config.get("data", "nonb_genomic_csv"),
        nonb_processed_csv     = config.get("data", "nonb_processed_csv", default=None),
        nonb2_csv              = nonb2_csv,
        te_scaler_bank_path    = config.get("data", "te_scaler_bank_path"),
        nonb_scaler_bank_path  = config.get("data", "nonb_scaler_bank_path"),
        nonb2_scaler_bank_path = nonb2_scaler,
        max_length             = config.get("model", "max_length"),
    )
    print(f"Dataset: {len(dataset):,} samples  NonB2: {has_nonb2}")

    splits = get_cv_splits_from_config(config, all_sequences, labels)

    model_builder = create_model_builder(config)

    model_dir  = exp_dir / "models"
    fold_files = sorted(model_dir.glob("fold_*_best.pt"))
    if not fold_files:
        raise FileNotFoundError(f"No fold checkpoints found in {model_dir}")
    print(f"\nFound {len(fold_files)} fold checkpoint(s)")

    all_fold_dfs = []

    for ckpt_path in fold_files:
        fold_idx  = int(ckpt_path.stem.split("_")[1])
        save_path = output_dir / f"fold_{fold_idx}_patching.csv"

        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx}  ({ckpt_path.name})")
        print(f"{'='*60}")

        if save_path.exists():
            print(f"   Already done — loading from {save_path}")
            df = pd.read_csv(save_path)
            # never mix patch sites across resumed folds
            if ("patch_site" not in df.columns
                    or set(df["patch_site"].unique()) != {args.patch_site}):
                found = (sorted(df["patch_site"].unique())
                         if "patch_site" in df.columns
                         else "none (legacy token_norm output)")
                raise RuntimeError(
                    f"{save_path} was produced with patch_site={found}, but "
                    f"this run uses patch_site='{args.patch_site}'. Use a "
                    f"fresh --output_dir (or delete the stale CSVs).")
            all_fold_dfs.append(df)
            continue

        ckpt  = torch.load(ckpt_path, map_location=device)
        model = model_builder()
        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
        if missing:
            print(f"  Missing keys  : {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
        model.to(device)
        model.eval()
        print(f"  Loaded epoch={ckpt['epoch']}  val_acc={ckpt['val_acc']:.4f}")

        _, val_idx = splits[fold_idx]
        print(f"  Val set: {len(val_idx):,} samples")

        fold_df = patch_fold(
            model           = model,
            dataset         = dataset,
            val_indices     = val_idx,
            labels          = labels,
            device          = device,
            has_nonb2       = has_nonb2,
            n_pairs         = args.n_pairs,
            min_confidence  = args.min_confidence,
            max_length_diff = args.max_length_diff,
            max_gc_diff     = args.max_gc_diff,
            fold_idx        = fold_idx,
            token_names     = token_names,
            block_map       = block_map,
            block_names     = block_names,
            patch_modes     = patch_modes,
            n_shuffled      = args.n_shuffled,
            patch_site      = args.patch_site,
            null_scopes     = tuple(null_scopes),
        )

        if fold_df.empty:
            print(f"    No results for fold {fold_idx} — skipping")
            continue

        fold_df.to_csv(save_path, index=False)
        print(f"  Saved fold results → {save_path}  ({len(fold_df):,} rows)")

        all_fold_dfs.append(fold_df)

        del model
        torch.cuda.empty_cache()

    if not all_fold_dfs:
        print("\nERROR: No fold results collected — check above for errors")
        return

    all_df = pd.concat(all_fold_dfs, ignore_index=True)
    all_path = output_dir / "all_folds_patching.csv"
    all_df.to_csv(all_path, index=False)
    print(f"\nAll-folds CSV → {all_path}  ({len(all_df):,} rows)")

    summary_df = build_summary(all_df)
    summary_path = output_dir / "patching_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary CSV   → {summary_path}")

    # ── Print ranked summary per patch_mode ──────────────────────────────
    print("\n" + "=" * 70)
    print("PATCHING SUMMARY — symmetry score & IIA, ranked within mode")
    print("=" * 70)
    for mode in patch_modes:
        sub = (summary_df[(summary_df["patch_mode"] == mode)
                          & (summary_df["direction"] == "lnc_to_mrna")]
               .sort_values("symmetry_score", ascending=False)
               [["patch_scope", "block", "symmetry_score", "causal_effect", "iia_overall"]]
               .drop_duplicates())
        if sub.empty:
            continue
        print(f"\n--- mode: {mode} ---")
        for _, row in sub.iterrows():
            iia_str = (f"  iia={row['iia_overall']:.3f}"
                      if pd.notna(row["iia_overall"]) else "")
            print(f"  {row['patch_scope']:<15}  {row['block']:<8}  "
                  f"symmetry={row['symmetry_score']:.4f}  "
                  f"causal_effect={row['causal_effect']:+.4f}{iia_str}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")
    print(f"  all_folds_patching.csv → {all_path}")
    print(f"  patching_summary.csv   → {summary_path}")


if __name__ == "__main__":
    main()