#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis/pattern/joint_patching_search.py

Post-training multi-token causal search: greedy forward selection (Option 1)
+ targeted top-k combinations (Option 3) over ARBITRARY subgroup subsets,
built on top of patch_tokens_v2.py's infrastructure (TokenCache, select_pairs)
but with BATCHED forward passes throughout, since this script calls the
patching routine ~100+ times (once per candidate subgroup set) rather than
once per fold like patch_tokens_v2.py.

Does NOT retrain anything and does NOT modify patch_tokens_v2.py.

Batching notes (see "Efficiency" below for what changed vs. a naive port)
---------------------------------------------------------------------------
patch_tokens_v2.py's per-sample helpers (get_logits_single,
get_patched_logits, get_tokens_single) run one transcript per forward call
— fine for a script that patches each scope once, but the wrong shape here:
greedy search alone issues on the order of 10^5 individual forward passes
at batch size 1, which badly under-utilizes the GPU (kernel-launch and
host->device transfer overhead dominates over actual compute).

This version instead:
  1. Batches the initial token-caching pass over all transcripts needed
     for the selected pairs (DataLoader, batch_size configurable), instead
     of one-sequence-at-a-time caching.
  2. Batches the baseline (unpatched) forward pass over all n_pairs at once
     per direction, instead of one baseline call per pair.
  3. Batches the patched forward pass: builds one batch of n_pairs
     sequences, and patches each batch item with ITS OWN source token
     vector (not a single shared vector — patch_tokens_v2.py's original
     make_patch_hook assumes one shared vector per call, which only works
     at batch size 1; this version's hook indexes per-sample vectors from
     a pre-stacked (n_pairs, len(positions), d_proj) tensor).

Net effect: run_joint_patch does ~4 batched forward passes (2 directions x
{baseline, patched}) per candidate subgroup set, chunked to fit GPU memory,
instead of ~4 * n_pairs individual forward passes.

Modes
-----
  greedy : start empty, at each step add whichever remaining subgroup
           most increases joint IIA, stop at --max_set_size or on
           plateau. One point per step -> a search-derived sufficiency
           curve, directly comparable to the fixed single/block/all curve.

  topk   : take the --topk strongest subgroups by symmetry_score (from an
           existing patching_summary.csv) and exhaustively test every
           multi-element combination among them.

Null model
----------
For a subgroup set S, the additive/independence-null expected IIA is:

    IIA_additive(S) = 1 - prod_{i in S} (1 - IIA(i))

IIA_joint(S) >> IIA_additive(S) is evidence of synergy; IIA_joint(S) ~=
IIA_additive(S) is consistent with independent, non-synergistic
contributions (matching the existing block/all finding in
patch_tokens_v2.py).

Output
------
<output_dir>/
  joint_patching_greedy.csv   one row per greedy step
  joint_patching_topk.csv     one row per tested top-k combination

Usage
-----
python analysis/pattern/joint_patching_search.py \\
    --experiment_dir  gencode_v49_experiments/beta_vae_subgroup_base_g49_new \\
    --config          configs/beta_vae_subgroup_base_g49.json \\
    --fold            0 \\
    --single_summary  gencode_v49_experiments/.../patching_summary.csv \\
    --output_dir      gencode_v49_experiments/.../joint_patching \\
    --device          cuda:0 \\
    --n_pairs         1000 \\
    --min_confidence  0.7 \\
    --max_length_diff 0.2 \\
    --max_gc_diff     0.05 \\
    --mode            both \\
    --max_set_size    6 \\
    --topk            5 \\
    --forward_batch_size 128
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Callable, Dict, List, Sequence

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

from analysis.pattern.patch_tokens import (
    select_pairs,
    TokenCache,
    build_fwd_kwargs,
)


# ---------------------------------------------------------------------------
# Additive-null model
# ---------------------------------------------------------------------------

def iia_additive_expected(single_iias: Sequence[float]) -> float:
    """IIA_additive(S) = 1 - prod (1 - IIA(i)) — see module docstring."""
    p_none_flip = 1.0
    for iia in single_iias:
        p_none_flip *= (1.0 - iia)
    return 1.0 - p_none_flip


# ---------------------------------------------------------------------------
# Batched token caching
# ---------------------------------------------------------------------------

def cache_tokens_batched(
    model,
    dataset,
    indices:    List[int],
    device:     torch.device,
    has_nonb2:  bool,
    batch_size: int = 128,
) -> Dict[int, torch.Tensor]:
    """
    Batched replacement for patch_tokens_v2.py's get_tokens_single called in
    a loop. Returns {dataset_idx: (N_tokens, d_proj) tensor}, same contract
    as before, but computed via a DataLoader over `indices` at `batch_size`
    instead of one forward call per index.
    """
    cache = TokenCache()
    cache.register(model)

    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size,
                        shuffle=False, num_workers=0)

    token_map: Dict[int, torch.Tensor] = {}
    cursor = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Caching tokens (batched)", leave=False):
            fwd = build_fwd_kwargs(batch, device, has_nonb2, batched=True)
            seq = batch["sequence"].to(device)
            model(seq, **fwd)  # TokenCache hook captures cache.tokens here
            batch_tokens = cache.tokens  # (B, N_tokens, d_proj)

            for i in range(batch_tokens.shape[0]):
                token_map[indices[cursor]] = batch_tokens[i].clone()
                cursor += 1

    cache.remove()
    assert cursor == len(indices), f"Cached {cursor} != requested {len(indices)}"
    return token_map


# ---------------------------------------------------------------------------
# Batched forward passes (baseline + patched, per direction)
# ---------------------------------------------------------------------------

def _gather_batch(dataset, idx_array: np.ndarray, device: torch.device,
                   has_nonb2: bool) -> dict:
    """
    Build one batched forward-kwargs dict for a set of dataset indices,
    by stacking individual dataset[idx] samples. Mirrors what a DataLoader
    collate_fn would do, done explicitly here since idx_array is an
    arbitrary (possibly repeated) index list rather than a Subset.
    """
    samples = [dataset[int(i)] for i in idx_array]
    seq = torch.stack([s["sequence"] for s in samples]).to(device)

    batch = {
        "te_genomic":     torch.stack([s["te_genomic"]     for s in samples]).to(device),
        "te_processed":   torch.stack([s["te_processed"]   for s in samples]).to(device),
        "nonb_genomic":   torch.stack([s["nonb_genomic"]   for s in samples]).to(device),
        "nonb_processed": torch.stack([s["nonb_processed"] for s in samples]).to(device),
        "deterministic":  True,
    }
    if has_nonb2 and "nonb2" in samples[0]:
        batch["nonb2"] = torch.stack([s["nonb2"] for s in samples]).to(device)

    return seq, batch


def batched_baseline_forward(
    model, dataset, idx_array: np.ndarray, device: torch.device, has_nonb2: bool,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (logits_mRNA, probs_mRNA) arrays, one per item in idx_array,
    chunked to fit GPU memory. Batched replacement for repeated
    get_logits_single calls."""
    all_logits, all_probs = [], []
    with torch.no_grad():
        for start in range(0, len(idx_array), chunk_size):
            chunk = idx_array[start:start + chunk_size]
            seq, fwd = _gather_batch(dataset, chunk, device, has_nonb2)
            out = model(seq, **fwd)
            logits = out["logits"][:, 1].cpu().numpy()
            probs  = torch.softmax(out["logits"], dim=1)[:, 1].cpu().numpy()
            all_logits.append(logits)
            all_probs.append(probs)
    return np.concatenate(all_logits), np.concatenate(all_probs)


def batched_patched_forward(
    model, dataset, idx_array: np.ndarray, device: torch.device, has_nonb2: bool,
    positions: List[int], patch_vectors: torch.Tensor, chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Batched patched forward pass. `patch_vectors` has shape
    (len(idx_array), len(positions), d_proj) — ONE source vector set per
    batch item, unlike patch_tokens_v2.py's make_patch_hook which assumes a
    single shared vector for the whole (batch-size-1) call.

    Returns (patched_logits_mRNA, patched_probs_mRNA), one per idx_array item.
    """
    all_logits, all_probs = [], []

    with torch.no_grad():
        for start in range(0, len(idx_array), chunk_size):
            chunk       = idx_array[start:start + chunk_size]
            chunk_vecs  = patch_vectors[start:start + chunk_size].to(device)
            seq, fwd    = _gather_batch(dataset, chunk, device, has_nonb2)

            def _hook(module, inp, output, _positions=positions, _vecs=chunk_vecs):
                patched = output.clone()
                for pos_i, pos in enumerate(_positions):
                    patched[:, pos, :] = _vecs[:, pos_i, :]
                return patched

            handle = model.token_norm.register_forward_hook(_hook)
            try:
                out = model(seq, **fwd)
            finally:
                handle.remove()

            logits = out["logits"][:, 1].cpu().numpy()
            probs  = torch.softmax(out["logits"], dim=1)[:, 1].cpu().numpy()
            all_logits.append(logits)
            all_probs.append(probs)

    return np.concatenate(all_logits), np.concatenate(all_probs)


# ---------------------------------------------------------------------------
# Core: joint patching over an arbitrary subgroup SET, for pre-selected pairs
# ---------------------------------------------------------------------------

def run_joint_patch(
    model,
    dataset,
    lnc_dataset_idx:  np.ndarray,
    mrna_dataset_idx: np.ndarray,
    token_cache_map:  Dict[int, torch.Tensor],
    subgroup_set:     List[str],
    token_names:      List[str],
    device:           torch.device,
    has_nonb2:        bool,
    chunk_size:       int,
) -> Dict[str, float]:
    """
    Patch every token belonging to `subgroup_set` simultaneously, across all
    pre-selected matched pairs, in both directions — batched. Mirrors
    patch_tokens_v2.py's "block"/"all" scope logic exactly (same hook
    semantics, same IIA definition), generalized to an arbitrary subgroup
    list and computed via chunked batched forward passes instead of one
    forward call per pair.
    """
    positions = [token_names.index(sg) for sg in subgroup_set]

    # ── lnc_to_mrna: baseline on lncRNA source, patch in matched mRNA's tokens
    baseline_logit_l, baseline_prob_l = batched_baseline_forward(
        model, dataset, lnc_dataset_idx, device, has_nonb2, chunk_size)
    patch_vecs_l = torch.stack(
        [token_cache_map[int(i)][positions] for i in mrna_dataset_idx]
    )  # (n_pairs, len(positions), d_proj)
    patched_logit_l, patched_prob_l = batched_patched_forward(
        model, dataset, lnc_dataset_idx, device, has_nonb2,
        positions, patch_vecs_l, chunk_size)

    valid_l = baseline_prob_l <= 0.5
    iia_l2m_val = (float(np.mean(patched_prob_l[valid_l] > 0.5))
                   if valid_l.any() else float("nan"))
    deltas_l2m  = (patched_logit_l[valid_l] - baseline_logit_l[valid_l])
    mean_delta_l2m = float(np.mean(deltas_l2m)) if valid_l.any() else 0.0

    # ── mrna_to_lnc: baseline on mRNA source, patch in matched lncRNA's tokens
    baseline_logit_m, baseline_prob_m = batched_baseline_forward(
        model, dataset, mrna_dataset_idx, device, has_nonb2, chunk_size)
    patch_vecs_m = torch.stack(
        [token_cache_map[int(i)][positions] for i in lnc_dataset_idx]
    )
    patched_logit_m, patched_prob_m = batched_patched_forward(
        model, dataset, mrna_dataset_idx, device, has_nonb2,
        positions, patch_vecs_m, chunk_size)

    valid_m = baseline_prob_m > 0.5
    iia_m2l_val = (float(np.mean(patched_prob_m[valid_m] <= 0.5))
                   if valid_m.any() else float("nan"))
    deltas_m2l  = (patched_logit_m[valid_m] - baseline_logit_m[valid_m])
    mean_delta_m2l = float(np.mean(deltas_m2l)) if valid_m.any() else 0.0

    iia_overall    = float(np.nanmean([iia_l2m_val, iia_m2l_val]))
    symmetry_score = (abs(mean_delta_l2m) + abs(mean_delta_m2l)) / 2

    return {
        "symmetry_score": symmetry_score,
        "iia":            iia_overall,
        "iia_l2m":        iia_l2m_val,
        "iia_m2l":        iia_m2l_val,
    }


# ---------------------------------------------------------------------------
# Option 1: greedy forward selection
# ---------------------------------------------------------------------------

def greedy_forward_search(
    all_subgroups: List[str],
    single_iia:    Dict[str, float],
    max_set_size:  int,
    patch_fn:      Callable[[List[str]], Dict[str, float]],
) -> pd.DataFrame:
    current_set: List[str] = []
    remaining    = list(all_subgroups)
    rows         = []

    for step in range(1, max_set_size + 1):
        best_candidate, best_result = None, None
        for candidate in tqdm(remaining, desc=f"    Greedy step {step}", leave=False):
            result = patch_fn(current_set + [candidate])
            if best_result is None or result["iia"] > best_result["iia"]:
                best_candidate, best_result = candidate, result

        if best_candidate is None:
            break

        current_set.append(best_candidate)
        remaining.remove(best_candidate)

        expected = iia_additive_expected([single_iia[sg] for sg in current_set])
        rows.append({
            "step":                  step,
            "added_subgroup":        best_candidate,
            "current_set":           "+".join(current_set),
            "set_size":              len(current_set),
            "iia_joint":             best_result["iia"],
            "symmetry_joint":        best_result["symmetry_score"],
            "iia_additive_expected": expected,
            "delta_vs_additive":     best_result["iia"] - expected,
        })

        prev_iia = rows[-2]["iia_joint"] if len(rows) > 1 else 0.0
        if best_result["iia"] <= prev_iia:
            break  # plateaued

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Option 3: targeted top-k combinations
# ---------------------------------------------------------------------------

def topk_combinations_search(
    single_summary: pd.DataFrame,
    single_iia:     Dict[str, float],
    k:              int,
    patch_fn:       Callable[[List[str]], Dict[str, float]],
) -> pd.DataFrame:
    top_subgroups = (
        single_summary.sort_values("symmetry_score", ascending=False)
        ["patch_scope"].head(k).tolist()
    )
    print(f"    Top-{k} subgroups by symmetry_score: {top_subgroups}")

    rows = []
    combos = [c for size in range(2, k + 1)
              for c in itertools.combinations(top_subgroups, size)]
    for combo in tqdm(combos, desc="    Top-k combinations"):
        combo = list(combo)
        result   = patch_fn(combo)
        expected = iia_additive_expected([single_iia[sg] for sg in combo])
        rows.append({
            "combo":                 "+".join(combo),
            "set_size":              len(combo),
            "iia_joint":             result["iia"],
            "symmetry_joint":        result["symmetry_score"],
            "iia_additive_expected": expected,
            "delta_vs_additive":     result["iia"] - expected,
        })

    return pd.DataFrame(rows).sort_values("delta_vs_additive", ascending=False)


# ---------------------------------------------------------------------------
# Per-fold orchestration
# ---------------------------------------------------------------------------

def normalize_single_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts EITHER shape of single-mode patching output:
      - summary shape (patching_summary.csv): already has one row per
        (patch_scope, direction[, fold]) with iia_overall, symmetry_score
        columns pre-computed by patch_tokens_v2.py's build_summary().
      - raw per-pair shape (all_folds_patching.csv): one row per
        (patch_scope, direction, pair_i, fold), with iia_success /
        iia_baseline_valid / delta_logit but no iia_overall or
        symmetry_score column at all.

    Returns a normalized dataframe with exactly one row per
    (patch_scope, fold) — direction already collapsed — carrying
    'iia_overall' and 'symmetry_score', built by aggregating the raw shape
    if needed, or passed through (deduplicated across direction) if the
    summary shape was given.
    """
    df = raw_df[raw_df["patch_mode"] == "single"].copy()
    has_fold = "fold" in df.columns

    if "iia_overall" in df.columns and "symmetry_score" in df.columns:
        # Already summary-shaped: one row per (patch_scope, direction[, fold]).
        # Collapse direction by taking the first value (iia_overall and
        # symmetry_score are already direction-independent aggregates in
        # patch_tokens_v2.py's build_summary()).
        group_cols = ["patch_scope"] + (["fold"] if has_fold else [])
        out = (df.groupby(group_cols, as_index=False)
                 .agg(iia_overall=("iia_overall", "first"),
                      symmetry_score=("symmetry_score", "first")))
        return out

    # Raw per-pair shape: compute iia_overall and symmetry_score ourselves.
    required = {"iia_success", "iia_baseline_valid", "delta_logit", "direction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"single_summary file has neither ('iia_overall','symmetry_score') "
            f"nor the raw per-pair columns needed to compute them "
            f"(missing: {missing}). Got columns: {list(df.columns)}"
        )

    df["iia_baseline_valid"] = df["iia_baseline_valid"].astype(bool)
    df["iia_success"]        = df["iia_success"].astype(bool)

    group_cols = ["patch_scope", "direction"] + (["fold"] if has_fold else [])
    per_direction = (
        df[df["iia_baseline_valid"]]
        .groupby(group_cols, as_index=False)
        .agg(iia=("iia_success", "mean"))
    )
    delta_mean = (
        df.groupby(group_cols, as_index=False)
        .agg(mean_delta_logit=("delta_logit", "mean"))
    )
    per_direction = per_direction.merge(
        delta_mean, on=group_cols, how="outer"
    )

    id_cols = ["patch_scope"] + (["fold"] if has_fold else [])
    pivot_iia = per_direction.pivot_table(
        index=id_cols, columns="direction", values="iia"
    ).reset_index()
    pivot_delta = per_direction.pivot_table(
        index=id_cols, columns="direction", values="mean_delta_logit"
    ).reset_index()

    out = pivot_iia.merge(pivot_delta, on=id_cols, suffixes=("_iia", "_delta"))

    l2m_iia = "lnc_to_mrna_iia" if "lnc_to_mrna_iia" in out.columns else "lnc_to_mrna"
    m2l_iia = "mrna_to_lnc_iia" if "mrna_to_lnc_iia" in out.columns else "mrna_to_lnc"
    l2m_delta = "lnc_to_mrna_delta" if "lnc_to_mrna_delta" in out.columns else "lnc_to_mrna"
    m2l_delta = "mrna_to_lnc_delta" if "mrna_to_lnc_delta" in out.columns else "mrna_to_lnc"

    out["iia_overall"] = out[[l2m_iia, m2l_iia]].mean(axis=1, skipna=True)
    out["symmetry_score"] = (
        out[l2m_delta].abs().fillna(0) + out[m2l_delta].abs().fillna(0)
    ) / 2

    return out[id_cols + ["iia_overall", "symmetry_score"]]


def load_single_iia_for_fold(single_summary_normalized: pd.DataFrame, fold: int) -> Dict[str, float]:
    """
    Extract the single-mode IIA lookup for one fold from a NORMALIZED
    dataframe (output of normalize_single_summary — already one row per
    (patch_scope[, fold]) with iia_overall). Falls back to a fold-agnostic
    lookup if the summary has no per-fold breakdown, in which case the SAME
    single_iia dict is reused for every fold's additive-null calculation.
    """
    sub = single_summary_normalized
    if "fold" in sub.columns and sub["fold"].nunique() > 1:
        sub = sub[sub["fold"] == fold]
    sub = sub.drop_duplicates(subset="patch_scope")

    single_iia = dict(zip(sub["patch_scope"], sub["iia_overall"]))
    return {k: (v if pd.notna(v) else 0.0) for k, v in single_iia.items()}


def run_one_fold(
    fold:            int,
    exp_dir:         Path,
    dataset,
    all_sequences,
    labels,
    strat_groups,
    n_folds_config:  int,
    random_state:    int,
    has_nonb2:       bool,
    device:          torch.device,
    args,
    all_subgroups:   List[str],
    single_iia:      Dict[str, float],
    single_summary:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run greedy + top-k search for a single fold. Returns (greedy_df, topk_df),
    either of which may be empty depending on --mode. Both dataframes get a
    'fold' column added here so the caller can concatenate across folds.
    """
    skf = StratifiedKFold(n_splits=n_folds_config, shuffle=True,
                          random_state=random_state)
    splits = list(skf.split(all_sequences, strat_groups))
    _, val_idx = splits[fold]
    print(f"\nFold {fold}: val set {len(val_idx):,} samples")

    ckpt_path = exp_dir / "models" / f"fold_{fold}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt  = torch.load(ckpt_path, map_location=device)
    model = create_model_builder(load_config(str(args.config)))()
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded epoch={ckpt['epoch']}  val_acc={ckpt['val_acc']:.4f}")

    # ── Metadata pass — batched ──────────────────────────────────────────
    print("  Collecting val set metadata...")
    meta_loader = DataLoader(Subset(dataset, val_idx),
                             batch_size=args.forward_batch_size,
                             shuffle=False, num_workers=0)
    cache = TokenCache()
    cache.register(model)
    all_confs, all_lengths, all_gc = [], [], []
    with torch.no_grad():
        for batch in tqdm(meta_loader, desc="    Metadata pass", leave=False):
            fwd = build_fwd_kwargs(batch, device, has_nonb2, batched=True)
            seq = batch["sequence"].to(device)
            out = model(seq, **fwd)
            probs = torch.softmax(out["logits"], dim=1)
            all_confs.append(probs.max(1).values.cpu().numpy())
            seq_np = seq.cpu().numpy()
            all_lengths.append(seq_np[:, :4, :].sum(axis=(1, 2)).astype(np.float32))
            gc = ((seq[:, 1, :] + seq[:, 2, :]).sum(dim=1)
                  / seq[:, :4, :].sum(dim=(1, 2)).clamp(min=1)).cpu().numpy()
            all_gc.append(gc.astype(np.float32))
    cache.remove()
    confidences = np.concatenate(all_confs)
    lengths     = np.concatenate(all_lengths)
    gc_content  = np.concatenate(all_gc)

    rng = np.random.default_rng(seed=42 + fold)
    lnc_local, mrna_local = select_pairs(
        val_indices=val_idx, labels=labels, lengths=lengths,
        confidences=confidences, n_pairs=args.n_pairs,
        min_confidence=args.min_confidence, max_length_diff=args.max_length_diff,
        max_gc_diff=args.max_gc_diff, gc_content=gc_content, rng=rng,
    )
    n_actual = len(lnc_local)
    print(f"  Selected {n_actual} matched pairs "
          f"(reproduces patch_tokens_v2.py's fold-{fold} pairing)")
    lnc_dataset_idx  = val_idx[lnc_local]
    mrna_dataset_idx = val_idx[mrna_local]

    # ── Token caching — batched ──────────────────────────────────────────
    needed_idx = sorted(set(lnc_dataset_idx.tolist()) | set(mrna_dataset_idx.tolist()))
    print(f"  Caching token representations for {len(needed_idx)} transcripts "
          f"(batch_size={args.forward_batch_size})...")
    token_cache_map = cache_tokens_batched(
        model, dataset, needed_idx, device, has_nonb2,
        batch_size=args.forward_batch_size,
    )

    def patch_fn(subgroup_set: List[str]) -> Dict[str, float]:
        return run_joint_patch(
            model, dataset, lnc_dataset_idx, mrna_dataset_idx,
            token_cache_map, subgroup_set, token_names_for(all_subgroups), device,
            has_nonb2, chunk_size=args.forward_batch_size,
        )

    greedy_df = pd.DataFrame()
    topk_df   = pd.DataFrame()

    if args.mode in ("greedy", "both"):
        print(f"  Running greedy forward selection (max_set_size={args.max_set_size})...")
        greedy_df = greedy_forward_search(
            all_subgroups, single_iia, args.max_set_size, patch_fn
        )
        greedy_df["fold"] = fold
        fold_path = Path(args.output_dir) / f"joint_patching_greedy_fold{fold}.csv"
        greedy_df.to_csv(fold_path, index=False)
        print(f"  Saved: {fold_path}")

    if args.mode in ("topk", "both"):
        print(f"  Running top-{args.topk} combinations search...")
        topk_df = topk_combinations_search(
            single_summary, single_iia, args.topk, patch_fn
        )
        topk_df["fold"] = fold
        fold_path = Path(args.output_dir) / f"joint_patching_topk_fold{fold}.csv"
        topk_df.to_csv(fold_path, index=False)
        print(f"  Saved: {fold_path}")

    del model
    torch.cuda.empty_cache()

    return greedy_df, topk_df


def token_names_for(all_subgroups: List[str]) -> List[str]:
    """REGISTRY.all_subgroups defines both the token order and the name list
    — kept as a tiny named helper so run_one_fold's patch_fn closure reads
    clearly rather than repeating REGISTRY.all_subgroups inline."""
    return all_subgroups


def parse_fold_arg(fold_arg: str, n_folds_config: int) -> List[int]:
    """--fold accepts: 'all', a single int ('0'), or a comma list ('0,2,4')."""
    if fold_arg == "all":
        return list(range(n_folds_config))
    return [int(f) for f in fold_arg.split(",")]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--config",         required=True)
    parser.add_argument("--fold",           type=str, default="0",
                        help="'all' for every fold, a single fold index "
                             "('0'), or a comma-separated list ('0,2,4'). "
                             "Default: '0'.")
    parser.add_argument("--single_summary", required=True,
                        help="patching_summary.csv or all_folds_patching.csv "
                             "(single-mode rows; per-fold IIA used if present)")
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--device",
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_pairs",         type=int,   default=1000)
    parser.add_argument("--min_confidence",  type=float, default=0.7)
    parser.add_argument("--max_length_diff", type=float, default=0.2)
    parser.add_argument("--max_gc_diff",     type=float, default=0.05)
    parser.add_argument("--mode",            choices=["greedy", "topk", "both"],
                        default="both")
    parser.add_argument("--max_set_size",    type=int, default=6)
    parser.add_argument("--topk",            type=int, default=5)
    parser.add_argument("--forward_batch_size", type=int, default=256,
                        help="Chunk size for batched forward passes. Lower "
                             "if you hit GPU OOM; raise if you have headroom.")
    args = parser.parse_args()

    config     = load_config(args.config)
    exp_dir    = Path(args.experiment_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device     = torch.device(args.device)

    print("=" * 70)
    print("BetaVAESubgroup — Joint / Multi-Token Causal Search (batched, multi-fold)")
    print("=" * 70)

    single_summary_raw = pd.read_csv(args.single_summary)
    single_summary_all = normalize_single_summary(single_summary_raw)
    print(f"Normalized single-mode summary: {len(single_summary_all)} rows "
          f"(from {len(single_summary_raw)} raw rows in {args.single_summary})")
    all_subgroups = REGISTRY.all_subgroups
    print(f"Registry: {REGISTRY.total_tokens} tokens: {all_subgroups}")

    n_folds_config = config.get("training", "n_folds")
    random_state   = config.get("training", "random_state", default=42)
    fold_list = parse_fold_arg(args.fold, n_folds_config)
    print(f"Folds to run: {fold_list}")

    # ── Load dataset ONCE, reused across every fold ─────────────────────────
    all_sequences, labels = load_sequences_in_order(
        lnc_fasta=config.get("data", "lnc_fasta"),
        pc_fasta= config.get("data", "pc_fasta"),
    )
    has_nonb2    = config.get("data", "nonb2_csv", default=None) is not None
    nonb2_csv    = config.get("data", "nonb2_csv",              default=None)
    nonb2_scaler = config.get("data", "nonb2_scaler_bank_path", default=None)

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
        all_sequences, labels, n_bins=config.get("training", "n_bins", default=5)
    )

    all_greedy_dfs, all_topk_dfs = [], []

    for fold in fold_list:
        single_iia = load_single_iia_for_fold(single_summary_all, fold)
        single_summary_fold = single_summary_all.copy()
        if "fold" in single_summary_fold.columns and single_summary_fold["fold"].nunique() > 1:
            single_summary_fold = single_summary_fold[single_summary_fold["fold"] == fold]
        single_summary_fold = single_summary_fold.drop_duplicates(subset="patch_scope")

        greedy_df, topk_df = run_one_fold(
            fold=fold, exp_dir=exp_dir, dataset=dataset,
            all_sequences=all_sequences, labels=labels, strat_groups=strat_groups,
            n_folds_config=n_folds_config, random_state=random_state,
            has_nonb2=has_nonb2, device=device, args=args,
            all_subgroups=all_subgroups, single_iia=single_iia,
            single_summary=single_summary_fold,
        )
        if not greedy_df.empty:
            all_greedy_dfs.append(greedy_df)
        if not topk_df.empty:
            all_topk_dfs.append(topk_df)

    # ── Concatenate across folds ─────────────────────────────────────────
    if all_greedy_dfs:
        greedy_all = pd.concat(all_greedy_dfs, ignore_index=True)
        out_path = output_dir / "joint_patching_greedy_all_folds.csv"
        greedy_all.to_csv(out_path, index=False)
        print(f"\nCross-fold greedy CSV → {out_path}  ({len(greedy_all)} rows, "
              f"{greedy_all['fold'].nunique()} folds)")

        # Cross-fold mean/std per step, for a plottable summary curve
        greedy_summary = (
            greedy_all.groupby("step")
            .agg(
                mean_iia_joint             = ("iia_joint", "mean"),
                std_iia_joint              = ("iia_joint", "std"),
                mean_iia_additive_expected = ("iia_additive_expected", "mean"),
                std_iia_additive_expected  = ("iia_additive_expected", "std"),
                mean_delta_vs_additive     = ("delta_vs_additive", "mean"),
                std_delta_vs_additive      = ("delta_vs_additive", "std"),
                n_folds                    = ("fold", "nunique"),
            )
            .reset_index()
        )
        summary_path = output_dir / "joint_patching_greedy_summary.csv"
        greedy_summary.to_csv(summary_path, index=False)
        print(f"Cross-fold greedy summary → {summary_path}")
        print(greedy_summary.to_string(index=False))

    if all_topk_dfs:
        topk_all = pd.concat(all_topk_dfs, ignore_index=True)
        out_path = output_dir / "joint_patching_topk_all_folds.csv"
        topk_all.to_csv(out_path, index=False)
        print(f"\nCross-fold top-k CSV → {out_path}  ({len(topk_all)} rows, "
              f"{topk_all['fold'].nunique()} folds)")

        topk_summary = (
            topk_all.groupby("combo")
            .agg(
                mean_iia_joint         = ("iia_joint", "mean"),
                std_iia_joint          = ("iia_joint", "std"),
                mean_delta_vs_additive = ("delta_vs_additive", "mean"),
                std_delta_vs_additive  = ("delta_vs_additive", "std"),
                n_folds                = ("fold", "nunique"),
            )
            .reset_index()
            .sort_values("mean_delta_vs_additive", ascending=False)
        )
        summary_path = output_dir / "joint_patching_topk_summary.csv"
        topk_summary.to_csv(summary_path, index=False)
        print(f"Cross-fold top-k summary → {summary_path}")

    print("\nDONE")


if __name__ == "__main__":
    main()