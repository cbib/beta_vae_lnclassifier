#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis/pattern/patch_tokens_v2.py

Token-level activation patching for causal interpretability of BetaVAESubgroup —
extended version supporting multiple patch SCOPES and a shuffled-pairing NULL
CONTROL, on top of the original single-token patching in patch_tokens.py.

This file is additive / non-destructive: it does not modify patch_tokens.py.
It re-implements the pair-selection and token-caching steps identically (same
RNG seeding, same matching constraints) so results are directly comparable to
the original single-token run, and adds three things:

Patch scopes (--patch_modes, comma-separated, default: single,block,all)
--------------------------------------------------------------------------
  single : original behaviour — patch exactly one subgroup token.
  block  : patch every subgroup token belonging to one block (te / nonb /
           nonb2) simultaneously — one row per (block, pair, direction).
  all    : patch all 20 subgroup tokens simultaneously — the ceiling case.
           One row per (pair, direction), block/subgroup recorded as "ALL".

This gives three points on a sufficiency curve: single token -> block ->
full token pathway. If single-token IIA is low but all-token IIA is high,
no individual subgroup is causally SUFFICIENT alone, even where symmetry
score suggested strong reliance — motivates joint/multi-token analysis
(Track A, A5). If block IIA is close to all-token IIA, the block is doing
essentially all the work and other blocks are close to causally inert;
if block IIA is well below single-token IIA of its best member, that's
evidence of interference/redundancy rather than superadditivity.

Shuffled-pairing null control (--patch_modes shuffled)
--------------------------------------------------------------------------
  For each subgroup and direction, instead of the length/GC-matched partner
  transcript, the patch vector is drawn from a RANDOMLY CHOSEN transcript of
  the target class elsewhere in the val set (same confidence filter, no
  length/GC matching). Same IIA computation. This tests whether IIA under
  matched patching reflects genuine class-identity content in the token
  (matched >> shuffled) or just generic sensitivity to any perturbation of
  that token position (matched ~= shuffled). Uses a separate RNG stream
  (seed offset) so it doesn't disturb the matched-pair selection.

Output
------
<output_dir>/
  fold_N_patching.csv       per-pair results for fold N, all modes
  all_folds_patching.csv    concatenated long-format results
  patching_summary.csv      cross-fold mean +/- std per
                                (patch_mode, scope, direction)

Schema additions vs. the original all_folds_patching.csv:
  patch_mode   : "single" | "block" | "all" | "shuffled"
  patch_scope  : subgroup name (single/shuffled) | block name (block) | "ALL"

Usage
-----
python analysis/pattern/patch_tokens_v2.py \\
    --experiment_dir  gencode_v49_experiments/beta_vae_subgroup_base_g49_new \\
    --config          configs/beta_vae_subgroup_base_g49.json \\
    --output_dir      gencode_v49_experiments/beta_vae_subgroup_base_g49_new/patching_v2 \\
    --device          cuda:0 \\
    --n_pairs         200 \\
    --min_confidence  0.7 \\
    --max_length_diff 0.2 \\
    --max_gc_diff     0.05 \\
    --patch_modes     single,block,all,shuffled

Notes
-----
- Hook placement, z-freezing, pair-matching constraints, and RNG seeding for
  matched pairs are identical to patch_tokens.py, so single-mode results in
  this file should closely reproduce (up to negligible RNG-path differences
  introduced by shared code execution order) the original all_folds output.
  Treat this file as the source of truth going forward for anything that
  needs block/all/shuffled scopes; keep patch_tokens.py for the archived
  single-token-only run referenced in the poster.
- "single" mode is included here (not just imported) so that a single command
  produces all four modes together with consistent pairing across modes
  within a fold — the same matched pairs are reused for single/block/all,
  and only "shuffled" draws an independent partner.
"""

from __future__ import annotations

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


def get_tokens_single(model, dataset, idx: int, device, has_nonb2) -> torch.Tensor:
    """Return (N_tokens, d_proj) cached token tensor for dataset index idx."""
    sample = dataset[idx]
    cache = TokenCache()
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
                        positions: list[int], vectors: torch.Tensor) -> tuple[float, float]:
    """Run one forward pass with token positions `positions` replaced by
    `vectors` (shape (len(positions), d_proj)). Returns (logit_mRNA, prob_mRNA)."""
    sample = dataset[idx]
    seq = sample["sequence"].unsqueeze(0).to(device)
    fwd = build_fwd_kwargs(sample, device, has_nonb2, batched=False)

    hook_fn = make_patch_hook(positions, vectors)
    handle = model.token_norm.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            out = model(seq, **fwd)
    finally:
        handle.remove()

    logit = out["logits"][0, 1].item()
    prob  = torch.softmax(out["logits"], dim=1)[0, 1].item()
    return logit, prob


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
) -> pd.DataFrame:
    model.eval()

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

    # ── Step 4: cache token representations ─────────────────────────────────
    # Union of every dataset index we'll ever need a token vector from:
    # matched pairs + shuffled partners.
    needed_idx = set(lnc_dataset_idx.tolist()) | set(mrna_dataset_idx.tolist())
    needed_idx |= set(shuf_mrna_dataset_idx.tolist()) | set(shuf_lnc_dataset_idx.tolist())
    needed_idx = sorted(needed_idx)

    print(f"  Caching token representations for {len(needed_idx)} transcripts...")
    token_cache_map: dict[int, torch.Tensor] = {}
    for idx in tqdm(needed_idx, desc="    Caching tokens", leave=False):
        token_cache_map[idx] = get_tokens_single(model, dataset, idx, device, has_nonb2)

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
                model, dataset, lnc_idx, device, has_nonb2, positions, vectors)

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
                model, dataset, mrna_idx, device, has_nonb2, positions, vectors_r)

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

    # ── Shuffled-null control: single-token scope only, random partner ─────
    if "shuffled" in patch_modes and len(shuf_mrna_dataset_idx) and len(shuf_lnc_dataset_idx):
        print(f"  Shuffled-null control: {n_tokens} tokens × 2 directions × "
              f"{len(shuf_mrna_dataset_idx)} draws...")

        for i, sg in enumerate(tqdm(token_names, desc="    Shuffled token loop", leave=False)):
            block_label = block_map.get(sg, "unknown")
            positions = [i]

            for pair_i in range(len(shuf_mrna_dataset_idx)):
                # lnc_to_mrna: baseline is a matched-pool lncRNA (reuse the
                # same lnc transcripts as the matched run for comparability),
                # patch source is a RANDOM mRNA transcript, not its partner.
                lnc_idx = lnc_dataset_idx[pair_i % n_actual]
                rand_mrna_idx = shuf_mrna_dataset_idx[pair_i]

                if rand_mrna_idx not in token_cache_map:
                    token_cache_map[rand_mrna_idx] = get_tokens_single(
                        model, dataset, rand_mrna_idx, device, has_nonb2)
                rand_mrna_tok = token_cache_map[rand_mrna_idx]

                baseline_logit_lnc, baseline_prob_lnc = get_logits_single(
                    model, dataset, lnc_idx, device, has_nonb2)
                vectors = rand_mrna_tok[positions]
                patched_logit_lnc, patched_prob_lnc = get_patched_logits(
                    model, dataset, lnc_idx, device, has_nonb2, positions, vectors)

                iia_baseline_valid_lnc = baseline_prob_lnc <= 0.5
                iia_success_lnc = bool(iia_baseline_valid_lnc and patched_prob_lnc > 0.5)

                rows.append({
                    "patch_mode": "shuffled", "patch_scope": sg,
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
                        model, dataset, rand_lnc_idx, device, has_nonb2)
                rand_lnc_tok = token_cache_map[rand_lnc_idx]

                baseline_logit_mrna, baseline_prob_mrna = get_logits_single(
                    model, dataset, mrna_idx, device, has_nonb2)
                vectors_r = rand_lnc_tok[positions]
                patched_logit_mrna, patched_prob_mrna = get_patched_logits(
                    model, dataset, mrna_idx, device, has_nonb2, positions, vectors_r)

                iia_baseline_valid_mrna = baseline_prob_mrna > 0.5
                iia_success_mrna = bool(iia_baseline_valid_mrna and patched_prob_mrna <= 0.5)

                rows.append({
                    "patch_mode": "shuffled", "patch_scope": sg,
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

    return pd.DataFrame(rows)


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
                             "{single,block,all,shuffled} (default: single,block,all)")
    parser.add_argument("--n_shuffled",      type=int,   default=None,
                        help="Draws per token for the shuffled-null control "
                             "(default: same as --n_pairs)")
    args = parser.parse_args()

    patch_modes = [m.strip() for m in args.patch_modes.split(",") if m.strip()]
    valid_modes = {"single", "block", "all", "shuffled"}
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
    print("BetaVAESubgroup — Multi-Scope Token Activation Patching ")
    print("=" * 70)
    print(f"Experiment  : {exp_dir}")
    print(f"Config      : {args.config}")
    print(f"Output dir  : {output_dir}")
    print(f"Device      : {args.device}")
    print(f"Pairs/fold  : {args.n_pairs}")
    print(f"Patch modes : {patch_modes}")
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

    strat_groups = create_length_stratified_groups(
        all_sequences, labels,
        n_bins=config.get("training", "n_bins", default=5)
    )
    skf = StratifiedKFold(
        n_splits     = config.get("training", "n_folds"),
        shuffle      = True,
        random_state = config.get("training", "random_state", default=42)
    )
    splits = list(skf.split(all_sequences, strat_groups))

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