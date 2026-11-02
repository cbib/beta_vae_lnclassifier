#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis/pattern/joint_patching_search.py

Multi-token causal search over subgroup sets, built on patch_tokens.py
(TokenCache, select_pairs) with batched forward passes. No retraining.

Modes
-----
  greedy   forward selection to --max_search_depth; strict-decrease stop.
  topk     all combinations among the --topk strongest subgroups.
  pairs    interaction test for all 190 subgroup pairs (BH within fold,
           Fisher-combined and BH across folds).
  both     greedy + topk.   all: greedy + pairs.
  controls replay greedy sets / pairs with norm-matched noise and
           same-class sources (needs --greedy_csv).

Design
------
- Patch site (--patch_site): "projector" (default) patches model.projectors[block]
  with the source's raw tokens, reaching both the cross-attention and the
  per-block summary path. "token_norm" is the legacy site, kept for comparison;
  --single_summary must come from the same site.
- Held-out selection: matched pairs are split (--heldout_frac, default 0.5)
  into a selection half (greedy / top-k choice) and a held-out half; every
  *_heldout column is scored on the held-out half only.
- Nulls: independence IIA_add(S) = 1 - prod_i(1 - IIA_i), and the logit-additive
  flip rate (single-token logit shifts summed). logit_interaction is the joint
  shift minus the summed single shifts, per transcript.
- Checks: per-sample and batched self-patch identity run before any search.
- Every mode writes pair_balance_*.csv (SMD of length / GC).

Output (<output_dir>/)
----------------------
  joint_patching_{greedy,topk,pairs,controls}_{fold<N>,all_folds,summary}.csv
  pair_balance_*.csv

Usage
-----
python analysis/pattern/joint_patching_search.py \\
    --experiment_dir EXP --config CFG --fold all --output_dir OUT \\
    --mode all --max_search_depth 10 --n_pairs 1000 --heldout_frac 0.5
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
from data.cv_utils import load_sequences_in_order, get_cv_splits_from_config
from data.gated_feature_dataset import SequenceFeatureDataset
from data.feature_registry import REGISTRY

# Reuse patch_tokens.py's pair selection and metadata-pass conventions.
# NOTE: the per-sample forward helpers (get_logits_single, get_patched_logits,
# get_tokens_single) are intentionally NOT imported — this script replaces
# them with batched equivalents below. TokenCache and build_fwd_kwargs are
# still reused since they're already collection/dict-shaped, not per-sample.
from analysis.pattern.patch_tokens import (
    select_pairs,
    make_noise_vectors,
    TokenCache,
    build_fwd_kwargs,
    PATCH_SITES,
    DEFAULT_PATCH_SITE,
    ProjectorTokenCache,
    build_token_layout,
    register_projector_patch_hooks,
    self_patch_sanity_check,
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
    patch_site: str = DEFAULT_PATCH_SITE,
) -> Dict[int, torch.Tensor]:
    """Returns {dataset_idx: (N_tokens, d_proj)}; "projector" caches raw
    pre-token_norm tokens, "token_norm" the post-norm ones."""
    cache = TokenCache() if patch_site == "token_norm" else ProjectorTokenCache()
    cache.register(model)

    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size,
                        shuffle=False, num_workers=0)

    token_map: Dict[int, torch.Tensor] = {}
    cursor = 0
    try:  # always detach the cache hooks
        with torch.no_grad():
            for batch in tqdm(loader, desc="  Caching tokens (batched)", leave=False):
                fwd = build_fwd_kwargs(batch, device, has_nonb2, batched=True)
                seq = batch["sequence"].to(device)
                model(seq, **fwd)  # TokenCache hook captures cache.tokens here
                batch_tokens = cache.tokens  # (B, N_tokens, d_proj)

                for i in range(batch_tokens.shape[0]):
                    token_map[indices[cursor]] = batch_tokens[i].clone()
                    cursor += 1
    finally:
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
    chunk_size: int, return_diff: bool = False,
) -> tuple[np.ndarray, ...]:
    """(logits_mRNA, probs_mRNA) per item; return_diff=True appends the logit
    difference (mRNA - lnc), whose sign decides the class."""
    all_logits, all_probs, all_diffs = [], [], []
    with torch.no_grad():
        for start in range(0, len(idx_array), chunk_size):
            chunk = idx_array[start:start + chunk_size]
            seq, fwd = _gather_batch(dataset, chunk, device, has_nonb2)
            out = model(seq, **fwd)
            logits = out["logits"][:, 1].cpu().numpy()
            probs  = torch.softmax(out["logits"], dim=1)[:, 1].cpu().numpy()
            all_logits.append(logits)
            all_probs.append(probs)
            all_diffs.append((out["logits"][:, 1] - out["logits"][:, 0]).cpu().numpy())
    res = (np.concatenate(all_logits), np.concatenate(all_probs))
    return res + (np.concatenate(all_diffs),) if return_diff else res


def batched_patched_forward(
    model, dataset, idx_array: np.ndarray, device: torch.device, has_nonb2: bool,
    positions: List[int], patch_vectors: torch.Tensor, chunk_size: int,
    patch_site: str = DEFAULT_PATCH_SITE, layout=None, return_diff: bool = False,
) -> tuple[np.ndarray, ...]:
    """Patched forward; `patch_vectors` is (len(idx_array), len(positions),
    d_proj), one source set per item. "projector" needs `layout` and raw
    tokens; "token_norm" is the legacy hook. Returns as in
    batched_baseline_forward."""
    all_logits, all_probs, all_diffs = [], [], []

    with torch.no_grad():
        for start in range(0, len(idx_array), chunk_size):
            chunk       = idx_array[start:start + chunk_size]
            chunk_vecs  = patch_vectors[start:start + chunk_size].to(device)
            seq, fwd    = _gather_batch(dataset, chunk, device, has_nonb2)

            if patch_site == "token_norm":
                def _hook(module, inp, output, _positions=positions, _vecs=chunk_vecs):
                    patched = output.clone()
                    for pos_i, pos in enumerate(_positions):
                        patched[:, pos, :] = _vecs[:, pos_i, :]
                    return patched

                handles = [model.token_norm.register_forward_hook(_hook)]
            else:
                if layout is None:
                    raise ValueError("patch_site='projector' requires `layout` "
                                     "(see build_token_layout).")
                handles = register_projector_patch_hooks(
                    model, layout, positions, chunk_vecs)
            try:
                out = model(seq, **fwd)
            finally:
                for handle in handles:
                    handle.remove()

            logits = out["logits"][:, 1].cpu().numpy()
            probs  = torch.softmax(out["logits"], dim=1)[:, 1].cpu().numpy()
            all_logits.append(logits)
            all_probs.append(probs)
            all_diffs.append((out["logits"][:, 1] - out["logits"][:, 0]).cpu().numpy())

    res = (np.concatenate(all_logits), np.concatenate(all_probs))
    return res + (np.concatenate(all_diffs),) if return_diff else res


# ---------------------------------------------------------------------------
# Batched hook sanity check
# ---------------------------------------------------------------------------

def batched_self_patch_check(
    model, dataset, idx_array: np.ndarray, token_cache_map: Dict[int, torch.Tensor],
    positions: List[int], device: torch.device, has_nonb2: bool, chunk_size: int,
    patch_site: str, layout, atol: float = 1e-4,
) -> None:
    """Patching every item with its OWN cached tokens through the BATCHED hook
    must reproduce the unpatched logits. Guards the per-sample-vector hook
    (a different code path from patch_tokens.py's single-sample check)."""
    base_l, _ = batched_baseline_forward(
        model, dataset, idx_array, device, has_nonb2, chunk_size)
    own = torch.stack([token_cache_map[int(i)][positions] for i in idx_array])
    patched_l, _ = batched_patched_forward(
        model, dataset, idx_array, device, has_nonb2, positions, own,
        chunk_size, patch_site=patch_site, layout=layout)
    err = float(np.max(np.abs(patched_l - base_l)))
    if err > atol:
        raise RuntimeError(
            f"Batched self-patch check failed (patch_site={patch_site}, "
            f"n_positions={len(positions)}): max |Δlogit| = {err:.2e} > {atol}.")


# ---------------------------------------------------------------------------
# Core: joint patching over an arbitrary subgroup SET, for pre-selected pairs
# ---------------------------------------------------------------------------

def split_pairs(n_pairs: int, heldout_frac: float, fold: int):
    """(selection_idx, heldout_idx): disjoint, sorted, deterministic per fold.
    heldout_frac == 0 returns the same index set twice (no split)."""
    all_idx = np.arange(n_pairs)
    if heldout_frac <= 0:
        return all_idx, all_idx
    perm = np.random.default_rng(seed=7000 + fold).permutation(n_pairs)
    n_eval = int(round(n_pairs * heldout_frac))
    return np.sort(perm[n_eval:]), np.sort(perm[:n_eval])


def compute_baselines(
    model, dataset, lnc_dataset_idx: np.ndarray, mrna_dataset_idx: np.ndarray,
    device: torch.device, has_nonb2: bool, chunk_size: int,
) -> dict:
    """Unpatched passes for both directions, computed once per pair set."""
    return {
        "l2m": batched_baseline_forward(model, dataset, lnc_dataset_idx, device,
                                        has_nonb2, chunk_size, return_diff=True),
        "m2l": batched_baseline_forward(model, dataset, mrna_dataset_idx, device,
                                        has_nonb2, chunk_size, return_diff=True),
    }


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
    patch_site:       str = DEFAULT_PATCH_SITE,
    layout=None,
    baseline:         dict | None = None,
    return_arrays:    bool = False,
    vecs_override:    dict | None = None,
) -> Dict[str, float]:
    """Patch all tokens of `subgroup_set` at once over the matched pairs, both
    directions. `baseline` (compute_baselines) skips the unpatched passes;
    return_arrays=True adds per-pair diff_*/valid_* arrays; vecs_override
    ({"l2m": T, "m2l": T}, each (n_pairs, len(set), d_proj)) replaces the
    matched partners as patch sources."""
    positions = [token_names.index(sg) for sg in subgroup_set]
    if baseline is None:
        baseline = compute_baselines(model, dataset, lnc_dataset_idx,
                                     mrna_dataset_idx, device, has_nonb2, chunk_size)

    # ── lnc_to_mrna: baseline on lncRNA source, patch in matched mRNA's tokens
    baseline_logit_l, baseline_prob_l, baseline_diff_l = baseline["l2m"]
    patch_vecs_l = (vecs_override["l2m"] if vecs_override is not None else torch.stack(
        [token_cache_map[int(i)][positions] for i in mrna_dataset_idx]
    ))  # (n_pairs, len(positions), d_proj)
    patched_logit_l, patched_prob_l, patched_diff_l = batched_patched_forward(
        model, dataset, lnc_dataset_idx, device, has_nonb2,
        positions, patch_vecs_l, chunk_size,
        patch_site=patch_site, layout=layout, return_diff=True)

    valid_l = baseline_prob_l <= 0.5
    iia_l2m_val = (float(np.mean(patched_prob_l[valid_l] > 0.5))
                   if valid_l.any() else float("nan"))
    deltas_l2m  = (patched_logit_l[valid_l] - baseline_logit_l[valid_l])
    mean_delta_l2m = float(np.mean(deltas_l2m)) if valid_l.any() else 0.0

    # ── mrna_to_lnc: baseline on mRNA source, patch in matched lncRNA's tokens
    baseline_logit_m, baseline_prob_m, baseline_diff_m = baseline["m2l"]
    patch_vecs_m = (vecs_override["m2l"] if vecs_override is not None else torch.stack(
        [token_cache_map[int(i)][positions] for i in lnc_dataset_idx]
    ))
    patched_logit_m, patched_prob_m, patched_diff_m = batched_patched_forward(
        model, dataset, mrna_dataset_idx, device, has_nonb2,
        positions, patch_vecs_m, chunk_size,
        patch_site=patch_site, layout=layout, return_diff=True)

    valid_m = baseline_prob_m > 0.5
    iia_m2l_val = (float(np.mean(patched_prob_m[valid_m] <= 0.5))
                   if valid_m.any() else float("nan"))
    deltas_m2l  = (patched_logit_m[valid_m] - baseline_logit_m[valid_m])
    mean_delta_m2l = float(np.mean(deltas_m2l)) if valid_m.any() else 0.0

    iia_overall    = float(np.nanmean([iia_l2m_val, iia_m2l_val]))
    symmetry_score = (abs(mean_delta_l2m) + abs(mean_delta_m2l)) / 2

    out = {
        "symmetry_score": symmetry_score,
        "iia":            iia_overall,
        "iia_l2m":        iia_l2m_val,
        "iia_m2l":        iia_m2l_val,
    }
    if return_arrays:
        out.update({
            "diff_base_l2m": baseline_diff_l, "diff_patched_l2m": patched_diff_l,
            "diff_base_m2l": baseline_diff_m, "diff_patched_m2l": patched_diff_m,
            "valid_l2m": valid_l, "valid_m2l": valid_m,
        })
    return out


# ---------------------------------------------------------------------------
# Joint-set controls (--mode controls) and pair-balance table
# ---------------------------------------------------------------------------

def control_patch_vectors(mode: str, token_cache_map: Dict[int, torch.Tensor],
                          lnc_idx: np.ndarray, mrna_idx: np.ndarray,
                          positions: List[int], generator: torch.Generator,
                          rng: np.random.Generator) -> Dict[str, torch.Tensor]:
    """Patch sources for run_joint_patch(vecs_override=...).
    sameclass: each baseline gets another transcript of ITS OWN class (cyclic
               shift by a random non-zero offset within the pair set).
    noise:     isotropic random vectors with the L2 norm of the matched
               partner's token at each patched position."""
    def stack(idx):
        return torch.stack([token_cache_map[int(i)][positions] for i in idx])

    if mode == "sameclass":
        n = len(lnc_idx)
        if n < 2:
            raise ValueError("sameclass control needs at least 2 pairs.")
        return {"l2m": stack(np.roll(lnc_idx, int(rng.integers(1, n)))),
                "m2l": stack(np.roll(mrna_idx, int(rng.integers(1, n))))}
    if mode == "noise":
        return {"l2m": make_noise_vectors(stack(mrna_idx), generator),
                "m2l": make_noise_vectors(stack(lnc_idx), generator)}
    raise ValueError(f"unknown control mode {mode!r}")


def run_control_replay(
    sets: Sequence[Dict], lnc_p: np.ndarray, mrna_p: np.ndarray, patch_one,
    vecs_fn, n_draws: int, fold: int,
) -> pd.DataFrame:
    """Replay each set (dicts: kind, step, subgroups) with matched, noise and
    same-class sources; controls are averaged over n_draws. patch_one(set,
    vecs_override=None) -> run_joint_patch result; vecs_fn(mode, set, gen,
    rng) -> vecs_override."""
    rows = []
    for si, item in enumerate(sets):
        members = list(item["subgroups"])
        matched = patch_one(members)
        row = {
            "fold": fold, "kind": item["kind"], "step": item.get("step"),
            "set": "+".join(members), "set_size": len(members),
            "iia_matched": matched["iia"], "iia_l2m_matched": matched["iia_l2m"],
            "iia_m2l_matched": matched["iia_m2l"],
            "symmetry_matched": matched["symmetry_score"], "n_draws": n_draws,
        }
        for mode in ("noise", "sameclass"):
            res = []
            for d in range(n_draws):
                gen = torch.Generator().manual_seed(9342 + 100000 * fold + 100 * si + d)
                rng = np.random.default_rng(9442 + 100000 * fold + 100 * si + d)
                res.append(patch_one(members, vecs_fn(mode, members, gen, rng)))
            for key, col in (("iia", "iia"), ("iia_l2m", "iia_l2m"),
                             ("iia_m2l", "iia_m2l"), ("symmetry_score", "symmetry")):
                vals = np.array([r[key] for r in res], dtype=float)
                row[f"{col}_{mode}_mean"] = float(np.nanmean(vals))
                row[f"{col}_{mode}_std"] = (float(np.nanstd(vals, ddof=1))
                                            if len(vals) > 1 else float("nan"))
        row["excess_iia_over_noise"] = row["iia_matched"] - row["iia_noise_mean"]
        row["excess_iia_over_sameclass"] = row["iia_matched"] - row["iia_sameclass_mean"]
        row["excess_symmetry_over_noise"] = (row["symmetry_matched"]
                                             - row["symmetry_noise_mean"])
        rows.append(row)
    return pd.DataFrame(rows)


def _smd(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference (pooled SD); NaN when undefined."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def pair_balance_table(lengths: np.ndarray, gc: np.ndarray, confidences: np.ndarray,
                       labels_local: np.ndarray, lnc_local: np.ndarray,
                       mrna_local: np.ndarray, min_confidence: float,
                       rng: np.random.Generator) -> pd.DataFrame:
    """Balance of matched / random_opposite (mRNA partners permuted) /
    all_highconf_unmatched pairings: SMD of length and GC (lnc - mRNA),
    mean confidence per class, and mean |rel. length| and |GC| gaps for
    paired groups. Arrays are indexed by val position."""
    rows = []

    def add(group, l_idx, m_idx, paired):
        l_idx, m_idx = np.asarray(l_idx), np.asarray(m_idx)
        row = {"group": group, "n_lnc": len(l_idx), "n_mrna": len(m_idx),
               "smd_length": _smd(lengths[l_idx], lengths[m_idx]),
               "smd_gc": _smd(gc[l_idx], gc[m_idx]),
               "mean_conf_lnc": float(confidences[l_idx].mean()) if len(l_idx) else float("nan"),
               "mean_conf_mrna": float(confidences[m_idx].mean()) if len(m_idx) else float("nan")}
        if paired and len(l_idx):
            row["mean_abs_rel_length_diff"] = float(np.mean(
                np.abs(lengths[l_idx] - lengths[m_idx]) / np.maximum(lengths[l_idx], 1.0)))
            row["mean_abs_gc_diff"] = float(np.mean(np.abs(gc[l_idx] - gc[m_idx])))
        rows.append(row)

    add("matched", lnc_local, mrna_local, True)
    add("random_opposite", lnc_local, rng.permutation(mrna_local), True)
    hc_l = np.where((labels_local == "lnc") & (confidences >= min_confidence))[0]
    hc_m = np.where((labels_local == "pc") & (confidences >= min_confidence))[0]
    add("all_highconf_unmatched", hc_l, hc_m, False)
    return pd.DataFrame(rows)



# ---------------------------------------------------------------------------
# Output-space interaction test and multiplicity correction
# ---------------------------------------------------------------------------
# The independence null 1 - prod(1 - IIA_i) treats each token's flip as an
# independent event. It is a reference level, not a test of interaction:
# effects that ADD in logit-difference space already give a joint flip rate
# above it once their sum crosses the decision threshold. The statistics
# below compare the joint shift with the sum of the single-token shifts,
# per patched transcript, on the same pairs.

def logit_interaction_stats(joint: Dict, members: Sequence[Dict]) -> Dict[str, float]:
    """
    Output-space interaction of a subgroup set.

    joint   : run_joint_patch(..., return_arrays=True) result for the set.
    members : same, one result per single-token member, on the SAME pairs.

    Per patched transcript, additive prediction of the logit-difference
    shift = sum of the members' single-token shifts. Sign convention:
    positive = shift toward the flip direction (lnc->mrna up, mrna->lnc down).
    Returns iia_logit_additive (flip rate if shifts add), logit_interaction
    (mean of sign * (joint shift - additive shift), both directions pooled),
    logit_interaction_se, n_units and the per-unit array 'units'.
    """
    units, flips = [], []
    for d, sign in (("l2m", 1.0), ("m2l", -1.0)):
        base  = joint[f"diff_base_{d}"]
        valid = joint[f"valid_{d}"]
        joint_shift = joint[f"diff_patched_{d}"] - base
        add_shift   = sum((m[f"diff_patched_{d}"] - base) for m in members)
        units.append((sign * (joint_shift - add_shift))[valid])
        pred = base + add_shift
        flipped = (pred > 0) if d == "l2m" else (pred <= 0)
        flips.append(float(np.mean(flipped[valid])) if valid.any() else float("nan"))
    u = np.concatenate(units)
    n = len(u)
    se = float(u.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {
        "iia_logit_additive":   float(np.nanmean(flips)),
        "logit_interaction":    float(u.mean()) if n else float("nan"),
        "logit_interaction_se": se,
        "n_units":              n,
        "units":                u,
    }


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """BH-adjusted q-values in input order; NaN p-values stay NaN and are
    excluded from the number of tests."""
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return q
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q_ok = np.empty(m)
    q_ok[order] = np.minimum(ranked, 1.0)
    q[ok] = q_ok
    return q


def _heldout_columns(res: Dict, subgroup_set: Sequence[str],
                     single_iia: Dict[str, float],
                     single_results: Dict[str, Dict]) -> Dict[str, float]:
    """Held-out columns shared by the greedy, top-k and pairwise tables."""
    expected = iia_additive_expected([single_iia[sg] for sg in subgroup_set])
    st = logit_interaction_stats(res, [single_results[sg] for sg in subgroup_set])
    return {
        "iia_joint_heldout":               res["iia"],
        "symmetry_joint_heldout":          res["symmetry_score"],
        "iia_additive_expected_heldout":   expected,
        "delta_vs_additive_heldout":       res["iia"] - expected,
        "iia_logit_additive_heldout":      st["iia_logit_additive"],
        "delta_vs_logit_additive_heldout": res["iia"] - st["iia_logit_additive"],
        "logit_interaction_heldout":       st["logit_interaction"],
        "logit_interaction_se_heldout":    st["logit_interaction_se"],
    }


# ---------------------------------------------------------------------------
# Option 1: greedy forward selection
# ---------------------------------------------------------------------------

def greedy_forward_search(
    all_subgroups:    List[str],
    single_iia:       Dict[str, float],
    max_search_depth: int,
    patch_fn:         Callable[[List[str]], Dict[str, float]],
    eval_patch_fn:    Callable[[List[str]], Dict] | None = None,
    eval_single_iia:  Dict[str, float] | None = None,
    eval_single_results: Dict[str, Dict] | None = None,
) -> pd.DataFrame:
    """
    Greedy forward selection on the SELECTION pairs (patch_fn, single_iia).
    If eval_patch_fn / eval_single_iia / eval_single_results are all given,
    each step's set is also scored on held-out pairs (columns *_heldout, see
    _heldout_columns); selection never sees them. Stops after a strict
    decrease of the selection IIA; a plateau continues (matches triframe).
    """
    given = [eval_patch_fn is not None, eval_single_iia is not None,
             eval_single_results is not None]
    if any(given) and not all(given):
        raise ValueError("eval_patch_fn, eval_single_iia and "
                         "eval_single_results must be given together.")
    current_set: List[str] = []
    remaining    = list(all_subgroups)
    rows         = []

    for step in range(1, max_search_depth + 1):
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
        row = {
            "step":                  step,
            "added_subgroup":        best_candidate,
            "current_set":           "+".join(current_set),
            "set_size":              len(current_set),
            "iia_joint":             best_result["iia"],
            "symmetry_joint":        best_result["symmetry_score"],
            "iia_additive_expected": expected,
            "delta_vs_additive":     best_result["iia"] - expected,
        }
        if eval_patch_fn is not None:
            row.update(_heldout_columns(eval_patch_fn(current_set), current_set,
                                        eval_single_iia, eval_single_results))
        rows.append(row)

        prev_iia = rows[-2]["iia_joint"] if len(rows) > 1 else 0.0
        # strict decrease only, as in triframe.layer3.greedy_token_search
        if best_result["iia"] < prev_iia:
            break

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Option 3: targeted top-k combinations
# ---------------------------------------------------------------------------

def topk_combinations_search(
    single_summary: pd.DataFrame,
    single_iia:     Dict[str, float],
    k:              int,
    patch_fn:       Callable[[List[str]], Dict[str, float]],
    eval_patch_fn:  Callable[[List[str]], Dict] | None = None,
    eval_single_iia: Dict[str, float] | None = None,
    eval_single_results: Dict[str, Dict] | None = None,
) -> pd.DataFrame:
    """
    Exhaustive combinations of the k strongest subgroups. `single_summary`
    (patch_scope, symmetry_score) and `single_iia` should come from the
    SELECTION pairs; held-out columns as in greedy_forward_search.
    """
    given = [eval_patch_fn is not None, eval_single_iia is not None,
             eval_single_results is not None]
    if any(given) and not all(given):
        raise ValueError("eval_patch_fn, eval_single_iia and "
                         "eval_single_results must be given together.")
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
        row = {
            "combo":                 "+".join(combo),
            "set_size":              len(combo),
            "iia_joint":             result["iia"],
            "symmetry_joint":        result["symmetry_score"],
            "iia_additive_expected": expected,
            "delta_vs_additive":     result["iia"] - expected,
        }
        if eval_patch_fn is not None:
            row.update(_heldout_columns(eval_patch_fn(combo), combo,
                                        eval_single_iia, eval_single_results))
        rows.append(row)

    sort_col = "delta_vs_logit_additive_heldout" if eval_patch_fn is not None \
        else "delta_vs_additive"
    return pd.DataFrame(rows).sort_values(sort_col, ascending=False)


# ---------------------------------------------------------------------------
# Pairwise interaction test (all token pairs, BH-corrected)
# ---------------------------------------------------------------------------

def pairwise_interaction_search(
    all_subgroups:       List[str],
    eval_patch_fn:       Callable[[List[str]], Dict],
    eval_single_iia:     Dict[str, float],
    eval_single_results: Dict[str, Dict],
) -> pd.DataFrame:
    """
    Output-space interaction for every unordered subgroup pair on the
    held-out pairs: two-sided one-sample t-test of the per-transcript
    interaction (see logit_interaction_stats) against 0, BH-adjusted over
    all pairs within the fold. C(20, 2) = 190 tests for the 20 subgroups.
    """
    from scipy import stats as sp_stats

    rows = []
    for a, b in tqdm(list(itertools.combinations(all_subgroups, 2)),
                     desc="    Pairwise interactions"):
        res = eval_patch_fn([a, b])
        st  = logit_interaction_stats(res, [eval_single_results[a],
                                            eval_single_results[b]])
        u = st["units"]
        if len(u) > 1 and u.std(ddof=1) > 0:
            t_stat, p_val = sp_stats.ttest_1samp(u, 0.0)
        elif len(u) > 1:
            t_stat, p_val = float("nan"), (1.0 if u.mean() == 0 else 0.0)
        else:
            t_stat, p_val = float("nan"), float("nan")
        row = {"token_a": a, "token_b": b, "n_units": st["n_units"],
               "t": float(t_stat), "p": float(p_val)}
        row.update(_heldout_columns(res, [a, b], eval_single_iia, eval_single_results))
        rows.append(row)

    df = pd.DataFrame(rows)
    df["q"] = benjamini_hochberg(df["p"].to_numpy())
    return df.sort_values("q").reset_index(drop=True)


def benjamini_hochberg_log(log_p: Sequence[float]) -> np.ndarray:
    """BH q-values from natural-log p-values, returned as plain q in [0, 1].
    Ranks on the log scale, so p-values below float underflow still order."""
    lp = np.asarray(log_p, dtype=float)
    q = np.full(lp.shape, np.nan)
    ok = ~np.isnan(lp)
    m = int(ok.sum())
    if m == 0:
        return q
    order = np.argsort(lp[ok])
    ranked = lp[ok][order] + np.log(m) - np.log(np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q_ok = np.empty(m)
    q_ok[order] = np.exp(np.minimum(ranked, 0.0))
    q[ok] = q_ok
    return q


def fisher_log_p(p: Sequence[float]) -> float:
    """Natural-log Fisher combined p of per-fold p-values (nan if none).
    Exact closed form for even df (chi2 survival = Poisson CDF) in log space,
    so it stays finite where scipy's logsf returns -inf."""
    from scipy.special import gammaln, logsumexp

    p = np.asarray(p, dtype=float)
    p = p[~np.isnan(p)]
    k = len(p)
    if k == 0:
        return float("nan")
    half = -np.log(np.clip(p, 1e-300, 1.0)).sum()   # = stat / 2
    if half <= 0:
        return 0.0
    i = np.arange(k)
    return float(-half + logsumexp(i * np.log(half) - gammaln(i + 1)))


def summarize_pairwise_folds(all_pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-fold summary of the pairwise table: mean interaction, fold-level
    significance counts, and a Fisher-combined p-value (validation sets are
    disjoint across folds) BH-adjusted over all pairs. Combination and BH run
    in log space; fisher_log10p is exact, fisher_p may underflow to 0.
    """
    rows = []
    for (a, b), g in all_pairs.groupby(["token_a", "token_b"]):
        lp = fisher_log_p(g["p"].dropna().to_numpy())
        rows.append({
            "token_a": a, "token_b": b, "n_folds": g["fold"].nunique(),
            "mean_logit_interaction": g["logit_interaction_heldout"].mean(),
            "std_logit_interaction":  g["logit_interaction_heldout"].std(),
            "n_folds_q_lt_0.05":      int((g["q"] < 0.05).sum()),
            "mean_delta_vs_logit_additive": g["delta_vs_logit_additive_heldout"].mean(),
            "fisher_p": float(np.exp(lp)) if not np.isnan(lp) else float("nan"),
            "fisher_log10p": lp / np.log(10.0) if not np.isnan(lp) else float("nan"),
        })
    out = pd.DataFrame(rows)
    out["fisher_q"] = benjamini_hochberg_log(
        (out["fisher_log10p"] * np.log(10.0)).to_numpy())
    return out.sort_values(["fisher_q", "fisher_log10p"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-fold orchestration
# ---------------------------------------------------------------------------

def normalize_single_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts EITHER shape of single-mode patching output:
      - summary shape (patching_summary.csv): already has one row per
        (patch_scope, direction[, fold]) with iia_overall, symmetry_score
        columns pre-computed by patch_tokens.py's build_summary().
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
        # patch_tokens.py's build_summary()).
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


def _run_controls_for_fold(fold, args, all_subgroups, token_names, model, dataset,
                           device, has_nonb2, layout, token_cache_map,
                           lnc_p, mrna_p) -> pd.DataFrame:
    """Sets: this fold's greedy sets (from --greedy_csv) and/or all subgroup
    pairs, replayed on the HELD-OUT pairs (lnc_p, mrna_p)."""
    scope = getattr(args, "control_scope", "greedy")
    sets = []
    if scope in ("greedy", "both"):
        gpath = getattr(args, "greedy_csv", None)
        if not gpath:
            raise ValueError("--mode controls with --control_scope greedy|both "
                             "needs --greedy_csv (joint_patching_greedy_all_folds.csv).")
        g = pd.read_csv(gpath)
        missing = {"step", "current_set"} - set(g.columns)
        if missing or "fold" not in g.columns:
            raise ValueError(
                f"{gpath} lacks columns {sorted(missing | ({'fold'} - set(g.columns)))}. "
                "Pass joint_patching_greedy_all_folds.csv (or a per-fold "
                "joint_patching_greedy_fold<N>.csv), not the *_summary.csv.")
        g = g[g["fold"] == fold].sort_values("step")
        if g.empty:
            raise ValueError(f"{gpath} has no rows for fold {fold}.")
        for _, r in g.iterrows():
            sets.append({"kind": "greedy_step", "step": int(r["step"]),
                         "subgroups": str(r["current_set"]).split("+")})
    if scope in ("pairs", "both"):
        for a_, b_ in itertools.combinations(all_subgroups, 2):
            sets.append({"kind": "pair", "step": None, "subgroups": [a_, b_]})
    print(f"  Control replays: {len(sets)} sets x (matched + "
          f"{args.n_control_draws} noise + {args.n_control_draws} same-class)")

    base = compute_baselines(model, dataset, lnc_p, mrna_p, device, has_nonb2,
                             args.forward_batch_size)

    def patch_one(members, vecs=None):
        return run_joint_patch(
            model, dataset, lnc_p, mrna_p, token_cache_map, members, token_names,
            device, has_nonb2, chunk_size=args.forward_batch_size,
            patch_site=args.patch_site, layout=layout, baseline=base,
            vecs_override=vecs)

    def vecs_fn(mode, members, gen, rng):
        return control_patch_vectors(
            mode, token_cache_map, lnc_p, mrna_p,
            [token_names.index(m) for m in members], gen, rng)

    df = run_control_replay(sets, lnc_p, mrna_p, patch_one, vecs_fn,
                            args.n_control_draws, fold)
    df["patch_site"] = args.patch_site
    df["heldout_frac"] = args.heldout_frac
    path = Path(args.output_dir) / f"joint_patching_controls_fold{fold}.csv"
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return df


def run_one_fold(
    fold:            int,
    exp_dir:         Path,
    dataset,
    all_sequences,
    labels,
    splits,
    n_folds_config:  int,
    random_state:    int,
    has_nonb2:       bool,
    device:          torch.device,
    args,
    all_subgroups:   List[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One fold per --mode; returns (greedy, topk, pairs, controls) frames with
    a 'fold' column (empty if not run). Selection uses the selection half of
    the pairs; *_heldout columns and the pair test use the held-out half."""
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
          f"(reproduces patch_tokens.py's fold-{fold} pairing)")
    lnc_dataset_idx  = val_idx[lnc_local]
    mrna_dataset_idx = val_idx[mrna_local]

    # confound balance of the matched pairs
    balance_df = pair_balance_table(
        lengths, gc_content, confidences,
        np.array([labels[i] for i in val_idx]), lnc_local, mrna_local,
        args.min_confidence, np.random.default_rng(seed=4242 + fold))
    balance_df.insert(0, "fold", fold)
    balance_path = Path(args.output_dir) / f"pair_balance_fold{fold}.csv"
    balance_df.to_csv(balance_path, index=False)
    print(f"  Pair balance → {balance_path}")
    print(balance_df.round(3).to_string(index=False))

    # ── Token caching — batched ──────────────────────────────────────────
    needed_idx = sorted(set(lnc_dataset_idx.tolist()) | set(mrna_dataset_idx.tolist()))
    print(f"  Caching token representations for {len(needed_idx)} transcripts "
          f"(batch_size={args.forward_batch_size})...")
    token_cache_map = cache_tokens_batched(
        model, dataset, needed_idx, device, has_nonb2,
        batch_size=args.forward_batch_size,
        patch_site=args.patch_site,
    )

    # token -> (block, row) map + hook sanity checks, before any search
    token_names = token_names_for(all_subgroups)
    block_map   = {sg: REGISTRY.token_block(sg) for sg in token_names}
    layout = (build_token_layout(model, token_names, block_map,
                                 list(REGISTRY.block_names))
              if args.patch_site == "projector" else None)
    check_idx = np.concatenate([lnc_dataset_idx[:16], mrna_dataset_idx[:16]])
    self_patch_sanity_check(
        model, dataset, [int(i) for i in check_idx[:2]]
        + [int(i) for i in check_idx[-2:]],
        device, has_nonb2, token_names, args.patch_site, layout)
    for chk_positions in ([0], list(range(len(token_names)))):
        batched_self_patch_check(
            model, dataset, check_idx, token_cache_map, chk_positions, device,
            has_nonb2, args.forward_batch_size, args.patch_site, layout)
    print(f"  Self-patch sanity checks passed "
          f"(per-sample + batched, patch_site={args.patch_site})")

    # selection / held-out split of the matched pairs
    sel_pairs, eval_pairs = split_pairs(n_actual, args.heldout_frac, fold)
    if args.heldout_frac > 0:
        print(f"  Pair split: {len(sel_pairs)} selection / {len(eval_pairs)} held-out")
    else:
        print("  WARNING: --heldout_frac 0 -> selection and evaluation share the "
              "same pairs; joint IIAs are optimistically biased.")

    def make_patch_fn(pair_idx: np.ndarray, with_arrays: bool):
        lnc_p, mrna_p = lnc_dataset_idx[pair_idx], mrna_dataset_idx[pair_idx]
        base = compute_baselines(model, dataset, lnc_p, mrna_p, device,
                                 has_nonb2, args.forward_batch_size)

        def patch_fn(subgroup_set: List[str]) -> Dict[str, float]:
            return run_joint_patch(
                model, dataset, lnc_p, mrna_p, token_cache_map, subgroup_set,
                token_names_for(all_subgroups), device, has_nonb2,
                chunk_size=args.forward_batch_size,
                patch_site=args.patch_site, layout=layout,
                baseline=base, return_arrays=with_arrays,
            )
        return patch_fn

    patch_fn      = make_patch_fn(sel_pairs, with_arrays=False)
    eval_patch_fn = make_patch_fn(eval_pairs, with_arrays=True)

    if args.mode == "controls":
        controls_df = _run_controls_for_fold(
            fold, args, all_subgroups, token_names, model, dataset, device,
            has_nonb2, layout, token_cache_map, lnc_dataset_idx[eval_pairs],
            mrna_dataset_idx[eval_pairs])
        del model
        torch.cuda.empty_cache()
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), controls_df

    # Single-token results on both halves (null IIAs, top-k ranking, and the
    # per-transcript shifts the logit-additive null is built from).
    print("  Single-token results on selection / held-out halves...")
    sel_single  = {sg: patch_fn([sg]) for sg in all_subgroups}
    eval_single = {sg: eval_patch_fn([sg]) for sg in all_subgroups}
    single_iia_sel  = {sg: (r["iia"] if np.isfinite(r["iia"]) else 0.0)
                       for sg, r in sel_single.items()}
    single_iia_eval = {sg: (r["iia"] if np.isfinite(r["iia"]) else 0.0)
                       for sg, r in eval_single.items()}
    single_rank = pd.DataFrame({
        "patch_scope":    list(sel_single),
        "symmetry_score": [r["symmetry_score"] for r in sel_single.values()],
    })

    greedy_df = pd.DataFrame()
    topk_df   = pd.DataFrame()
    pairs_df  = pd.DataFrame()
    out_dir = Path(args.output_dir)

    if args.mode in ("greedy", "both", "all"):
        print(f"  Running greedy forward selection (max_search_depth={args.max_search_depth})...")
        greedy_df = greedy_forward_search(
            all_subgroups, single_iia_sel, args.max_search_depth, patch_fn,
            eval_patch_fn=eval_patch_fn, eval_single_iia=single_iia_eval,
            eval_single_results=eval_single,
        )
        greedy_df["fold"] = fold
        greedy_df["patch_site"] = args.patch_site
        greedy_df["heldout_frac"] = args.heldout_frac
        fold_path = out_dir / f"joint_patching_greedy_fold{fold}.csv"
        greedy_df.to_csv(fold_path, index=False)
        print(f"  Saved: {fold_path}")

    if args.mode in ("topk", "both"):
        print(f"  Running top-{args.topk} combinations search...")
        topk_df = topk_combinations_search(
            single_rank, single_iia_sel, args.topk, patch_fn,
            eval_patch_fn=eval_patch_fn, eval_single_iia=single_iia_eval,
            eval_single_results=eval_single,
        )
        topk_df["fold"] = fold
        topk_df["patch_site"] = args.patch_site
        topk_df["heldout_frac"] = args.heldout_frac
        fold_path = out_dir / f"joint_patching_topk_fold{fold}.csv"
        topk_df.to_csv(fold_path, index=False)
        print(f"  Saved: {fold_path}")

    if args.mode in ("pairs", "all"):
        print(f"  Running pairwise interaction test over "
              f"{len(all_subgroups) * (len(all_subgroups) - 1) // 2} pairs...")
        pairs_df = pairwise_interaction_search(
            all_subgroups, eval_patch_fn, single_iia_eval, eval_single)
        pairs_df["fold"] = fold
        pairs_df["patch_site"] = args.patch_site
        pairs_df["heldout_frac"] = args.heldout_frac
        fold_path = out_dir / f"joint_patching_pairs_fold{fold}.csv"
        pairs_df.to_csv(fold_path, index=False)
        print(f"  Saved: {fold_path}")

    del model
    torch.cuda.empty_cache()

    return greedy_df, topk_df, pairs_df, pd.DataFrame()


def token_names_for(all_subgroups: List[str]) -> List[str]:
    """REGISTRY.all_subgroups defines both the token order and the name list
    — kept as a tiny named helper so run_one_fold's patch_fn closure reads
    clearly rather than repeating REGISTRY.all_subgroups inline."""
    return all_subgroups


def check_single_summary_site(raw_df: pd.DataFrame, patch_site: str,
                              allow_legacy: bool) -> None:
    """--single_summary must come from the same patch site; legacy files (no
    patch_site column) are refused unless allow_legacy."""
    if "patch_site" not in raw_df.columns:
        if allow_legacy:
            print("WARNING: --single_summary has no patch_site column (legacy "
                  "token_norm output); continuing because "
                  "--allow_legacy_single_summary was given. delta_vs_additive "
                  "will mix patch sites — do not report it.")
            return
        raise RuntimeError(
            "--single_summary has no patch_site column: it predates the "
            "projector-site fix (legacy token_norm patching). Re-run "
            "patch_tokens.py, or pass --allow_legacy_single_summary to "
            "override (not for reporting).")
    sites = set(raw_df["patch_site"].dropna().unique())
    if sites != {patch_site}:
        raise RuntimeError(
            f"--single_summary patch_site={sorted(sites)} does not match "
            f"--patch_site={patch_site}.")


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
    parser.add_argument("--single_summary", default=None,
                        help="OPTIONAL patching_summary.csv / all_folds_patching.csv. "
                             "Only its patch site is validated; the null and "
                             "top-k ranking are recomputed in-script.")
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--device",
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_pairs",         type=int,   default=1000)
    parser.add_argument("--min_confidence",  type=float, default=0.7)
    parser.add_argument("--max_length_diff", type=float, default=0.2)
    parser.add_argument("--max_gc_diff",     type=float, default=0.05)
    parser.add_argument("--mode",            choices=["greedy", "topk", "pairs", "both", "all", "controls"],
                        default="both",
                        help="greedy | topk | pairs (all-pairs interaction test) | "
                             "both (= greedy + topk) | all (= greedy + pairs; top-k only via "
                             "topk/both) | "
                             "controls (noise / same-class replays of the greedy sets "
                             "and/or all pairs on the held-out half; needs "
                             "--greedy_csv from an earlier greedy run)")
    parser.add_argument("--greedy_csv", default=None,
                        help="joint_patching_greedy_all_folds.csv of a previous "
                             "run (same --n_pairs/--heldout_frac/--fold) — "
                             "--mode controls only.")
    parser.add_argument("--control_scope", choices=["greedy", "pairs", "both"],
                        default="greedy", help="--mode controls: which sets to replay.")
    parser.add_argument("--n_control_draws", type=int, default=5,
                        help="--mode controls: independent noise / same-class "
                             "draws per set (mean and std reported).")
    parser.add_argument("--max_search_depth", "--max_set_size", dest="max_search_depth",
                        type=int, default=10,
                        help="Maximum greedy set size (name matches triframe; "
                             "--max_set_size is the old alias). Default 10; "
                             "was 6, which truncated rising curves.")
    parser.add_argument("--heldout_frac",    type=float, default=0.5,
                        help="Fraction of matched pairs held out from selection "
                             "and used for every *_heldout column and the "
                             "pairwise test (default 0.5; 0 = no split, "
                             "optimistically biased).")
    parser.add_argument("--topk",            type=int, default=5)
    parser.add_argument("--forward_batch_size", type=int, default=128,
                        help="Chunk size for batched forward passes. Lower "
                             "if you hit GPU OOM; raise if you have headroom.")
    parser.add_argument("--patch_site",      choices=PATCH_SITES,
                        default=DEFAULT_PATCH_SITE,
                        help="Where interventions are applied. 'projector' "
                             "(default) reaches both the cross-attention/z "
                             "path and the per-block summary path; "
                             "'token_norm' is the LEGACY single-path hook, "
                             "kept for comparison only. Must match the site "
                             "used to produce --single_summary.")
    parser.add_argument("--allow_legacy_single_summary", action="store_true",
                        help="Accept a --single_summary file with no "
                             "patch_site column (legacy token_norm output). "
                             "Not for reporting: delta_vs_additive would mix "
                             "patch sites.")
    args = parser.parse_args()

    config     = load_config(args.config)
    exp_dir    = Path(args.experiment_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device     = torch.device(args.device)

    print("=" * 70)
    print("BetaVAESubgroup — Joint / Multi-Token Causal Search (batched, multi-fold)")
    print("=" * 70)

    print(f"Patch site  : {args.patch_site}")
    if not 0.0 <= args.heldout_frac < 1.0:
        raise ValueError("--heldout_frac must be in [0, 1).")
    if args.single_summary:   # validation only
        single_summary_raw = pd.read_csv(args.single_summary)
        check_single_summary_site(single_summary_raw, args.patch_site,
                                  args.allow_legacy_single_summary)
        print(f"Single-mode summary patch site OK ({args.single_summary})")
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

    splits = get_cv_splits_from_config(config, all_sequences, labels)

    all_greedy_dfs, all_topk_dfs, all_pairs_dfs, all_ctrl_dfs = [], [], [], []

    for fold in fold_list:
        greedy_df, topk_df, pairs_df, ctrl_df = run_one_fold(
            fold=fold, exp_dir=exp_dir, dataset=dataset,
            all_sequences=all_sequences, labels=labels, splits=splits,
            n_folds_config=n_folds_config, random_state=random_state,
            has_nonb2=has_nonb2, device=device, args=args,
            all_subgroups=all_subgroups,
        )
        if not greedy_df.empty:
            all_greedy_dfs.append(greedy_df)
        if not topk_df.empty:
            all_topk_dfs.append(topk_df)
        if not pairs_df.empty:
            all_pairs_dfs.append(pairs_df)
        if not ctrl_df.empty:
            all_ctrl_dfs.append(ctrl_df)

    # ── Concatenate across folds ─────────────────────────────────────────
    if all_greedy_dfs:
        greedy_all = pd.concat(all_greedy_dfs, ignore_index=True)
        out_path = output_dir / "joint_patching_greedy_all_folds.csv"
        greedy_all.to_csv(out_path, index=False)
        print(f"\nCross-fold greedy CSV → {out_path}  ({len(greedy_all)} rows, "
              f"{greedy_all['fold'].nunique()} folds)")

        # Cross-fold mean/std per step, for a plottable summary curve
        agg_spec = dict(
            mean_iia_joint             = ("iia_joint", "mean"),
            std_iia_joint              = ("iia_joint", "std"),
            mean_iia_additive_expected = ("iia_additive_expected", "mean"),
            std_iia_additive_expected  = ("iia_additive_expected", "std"),
            mean_delta_vs_additive     = ("delta_vs_additive", "mean"),
            std_delta_vs_additive      = ("delta_vs_additive", "std"),
            n_folds                    = ("fold", "nunique"),
        )
        # held-out columns (the ones to report) when present
        for col in ("iia_joint_heldout", "iia_additive_expected_heldout",
                    "delta_vs_additive_heldout", "iia_logit_additive_heldout",
                    "delta_vs_logit_additive_heldout", "logit_interaction_heldout"):
            if col in greedy_all.columns:
                agg_spec[f"mean_{col}"] = (col, "mean")
                agg_spec[f"std_{col}"]  = (col, "std")
        greedy_summary = (
            greedy_all.groupby("step").agg(**agg_spec).reset_index()
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

        topk_spec = dict(
            mean_iia_joint         = ("iia_joint", "mean"),
            std_iia_joint          = ("iia_joint", "std"),
            mean_delta_vs_additive = ("delta_vs_additive", "mean"),
            std_delta_vs_additive  = ("delta_vs_additive", "std"),
            n_folds                = ("fold", "nunique"),
        )
        for col in ("iia_joint_heldout", "delta_vs_additive_heldout",
                    "delta_vs_logit_additive_heldout", "logit_interaction_heldout"):
            if col in topk_all.columns:
                topk_spec[f"mean_{col}"] = (col, "mean")
                topk_spec[f"std_{col}"]  = (col, "std")
        sort_col = ("mean_delta_vs_logit_additive_heldout"
                    if "mean_delta_vs_logit_additive_heldout" in topk_spec
                    else "mean_delta_vs_additive")
        topk_summary = (
            topk_all.groupby("combo").agg(**topk_spec).reset_index()
            .sort_values(sort_col, ascending=False)
        )
        summary_path = output_dir / "joint_patching_topk_summary.csv"
        topk_summary.to_csv(summary_path, index=False)
        print(f"Cross-fold top-k summary → {summary_path}")

    if all_pairs_dfs:   # pairwise interaction test
        pairs_all = pd.concat(all_pairs_dfs, ignore_index=True)
        out_path = output_dir / "joint_patching_pairs_all_folds.csv"
        pairs_all.to_csv(out_path, index=False)
        pairs_summary = summarize_pairwise_folds(pairs_all)
        summary_path = output_dir / "joint_patching_pairs_summary.csv"
        pairs_summary.to_csv(summary_path, index=False)
        print(f"\nPairwise interaction CSV → {out_path}  ({len(pairs_all)} rows)")
        print(f"Cross-fold pairwise summary → {summary_path}")
        print(pairs_summary.head(15).to_string(index=False))

    if all_ctrl_dfs:   # joint-set controls
        ctrl_all = pd.concat(all_ctrl_dfs, ignore_index=True)
        out_path = output_dir / "joint_patching_controls_all_folds.csv"
        ctrl_all.to_csv(out_path, index=False)
        metric_cols = [c for c in ctrl_all.columns
                       if c.startswith(("iia_", "symmetry_", "excess_"))]
        ctrl_summary = (ctrl_all.groupby(["kind", "set", "set_size"], dropna=False)
                        [metric_cols].mean()
                        .join(ctrl_all.groupby(["kind", "set", "set_size"], dropna=False)
                              .agg(n_folds=("fold", "nunique")))
                        .reset_index())
        summary_path = output_dir / "joint_patching_controls_summary.csv"
        ctrl_summary.to_csv(summary_path, index=False)
        print(f"\nJoint-set controls CSV → {out_path}  ({len(ctrl_all)} rows)")
        print(f"Cross-fold controls summary → {summary_path}")

    bal_files = [output_dir / f"pair_balance_fold{f}.csv" for f in fold_list]
    bal_files = [b for b in bal_files if b.exists()]
    if bal_files:      # pair-balance summary
        bal = pd.concat([pd.read_csv(b) for b in bal_files], ignore_index=True)
        bal_summary = bal.drop(columns=["fold"]).groupby("group").mean().reset_index()
        bal_path = output_dir / "pair_balance_summary.csv"
        bal_summary.to_csv(bal_path, index=False)
        print(f"\nPair balance summary → {bal_path}")
        print(bal_summary.round(3).to_string(index=False))

    print("\nDONE")


if __name__ == "__main__":
    main()