#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis/pattern/patch_tokens.py

Token-level activation patching for causal interpretability of BetaVAESubgroup.

For each subgroup token (position 0–19 in the cross-modal attention input),
replaces the token representation from a source class transcript into a
target class transcript's forward pass — keeping z and all other tokens fixed —
and measures the resulting shift in classification logit (Δlogit).

Two directions are run per subgroup:
  lnc_to_mrna : inject mRNA token into lncRNA context → Δlogit > 0 = causal shift toward mRNA
  mrna_to_lnc : inject lncRNA token into mRNA context → Δlogit < 0 = causal shift toward lncRNA

Asymmetry between directions is informative for suppressor variables:
a subgroup that is causally relied upon will show consistent Δlogit in both
directions; a subgroup that carries signal but is not causally relied upon
(e.g. SS_COUNT/SS_RELPOS) will show near-zero Δlogit in both directions.

Cross-fold design: one run per fold checkpoint, using the held-out val set
of each fold. Final summary is the cross-fold mean ± std of Δlogit per subgroup,
matching the format of the ablation and pattern analysis outputs.

Output
------
<output_dir>/
  fold_N_patching.csv          per-pair results for fold N
  all_folds_patching.csv       concatenated long-format results
  patching_summary.csv         cross-fold mean ± std per (subgroup, direction)

patching_summary.csv columns:
  subgroup, block, direction, mean_delta_logit, std_delta_logit,
  mean_delta_prob, std_delta_prob, n_pairs, n_folds

Usage
-----
python analysis/pattern/patch_tokens.py \\
    --experiment_dir  gencode_v49_experiments/beta_vae_subgroup_base_g49_new \\
    --config          configs/beta_vae_subgroup_base_g49.json \\
    --output_dir      gencode_v49_experiments/beta_vae_subgroup_base_g49_new/patching \\
    --device          cuda:0 \\
    --n_pairs         200 \\
    --min_confidence  0.7 \\
    --max_length_diff 0.2 \\
    --max_gc_diff     0.05

Notes
-----
- Hook placement: model.token_norm output captures tokens after normalisation
  and before cross-modal attention — the same location as extract_representations.py.
  Patching here is equivalent to swapping the subgroup's projected + normalised
  token representation, which is the natural intervention point.
- z is NOT patched: this isolates the token pathway contribution independently
  of the sequence encoder, which is the core causal claim.
- Pair matching is done within each fold's val set to avoid data leakage.
- Pairs are sampled without replacement per fold; the RNG seed is fixed per fold.
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
# Pair selection
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
    """
    Select matched (lncRNA, mRNA) pairs from the val set.

    Matching criteria:
      - Both correctly classified with confidence >= min_confidence
      - Transcript lengths within max_length_diff (fractional) of each other
      - GC content within max_gc_diff (if available)

    Returns
    -------
    lnc_pairs : (n_pairs,) indices into val_indices
    mrna_pairs : (n_pairs,) indices into val_indices
    """
    # labels are strings ("lnc" / "pc") from load_sequences_in_order
    label_arr = np.array([labels[i] for i in val_indices])
    conf_arr  = confidences   # already aligned to val_indices (local positions)
    len_arr   = lengths

    # Correct classification mask — confidence acts as a proxy
    # (high-confidence correct = model is using the right signal)
    # labels from load_sequences_in_order are strings: "lnc" / "pc"
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

    # Greedy matching: for each sampled lncRNA, find a mRNA within constraints
    rng.shuffle(lnc_pool)
    matched_lnc  = []
    matched_mrna = []
    mrna_used    = set()

    for li in lnc_pool:
        if len(matched_lnc) >= n_pairs:
            break

        l_len = len_arr[li]
        l_gc  = gc_content[li] if gc_content is not None else None

        # Filter mRNA pool by length match
        len_diffs = np.abs(len_arr[mrna_pool] - l_len) / max(l_len, 1.0)
        candidates = mrna_pool[len_diffs <= max_length_diff]

        # Further filter by GC if available
        if l_gc is not None and gc_content is not None and len(candidates) > 0:
            gc_diffs   = np.abs(gc_content[candidates] - l_gc)
            candidates = candidates[gc_diffs <= max_gc_diff]

        # Exclude already-used mRNA transcripts
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
        # output: (B, N_tokens, d_proj)
        self.tokens = output.detach()

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ---------------------------------------------------------------------------
# Single forward pass helper
# ---------------------------------------------------------------------------

def forward_single(
    model,
    batch,
    device: torch.device,
    has_nonb2: bool,
    patch_token_idx: Optional[int] = None,
    patch_vector:    Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run a single forward pass, optionally patching one token position.

    Parameters
    ----------
    patch_token_idx : if not None, replace this token position after token_norm
    patch_vector    : (1, d_proj) tensor to inject at patch_token_idx

    Returns
    -------
    logits : (1, 2)
    probs  : (1, 2)
    """
    seq  = batch["sequence"].to(device)
    te_g = batch["te_genomic"].to(device)
    te_p = batch["te_processed"].to(device)
    nb_g = batch["nonb_genomic"].to(device)
    nb_p = batch["nonb_processed"].to(device)

    fwd_kwargs = dict(
        te_genomic     = te_g,
        te_processed   = te_p,
        nonb_genomic   = nb_g,
        nonb_processed = nb_p,
        deterministic  = True,
    )
    if has_nonb2 and "nonb2" in batch:
        fwd_kwargs["nonb2"] = batch["nonb2"].to(device)

    patch_handle = None
    if patch_token_idx is not None and patch_vector is not None:
        def _patch_hook(module, input, output):
            patched = output.clone()
            patched[:, patch_token_idx, :] = patch_vector
            return patched
        patch_handle = model.token_norm.register_forward_hook(_patch_hook)

    try:
        out = model(seq, **fwd_kwargs)
    finally:
        if patch_handle is not None:
            patch_handle.remove()

    logits = out["logits"]          # (1, 2)
    probs  = torch.softmax(logits, dim=1)
    return logits, probs


# ---------------------------------------------------------------------------
# Per-fold patching
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
) -> pd.DataFrame:
    """
    Run activation patching for all subgroup tokens on one fold.

    Returns a long-format DataFrame with one row per (pair, subgroup, direction).
    """
    model.eval()

    # ── Step 1: extract confidences, lengths, GC for the val set ─────────────
    print(f"  Collecting val set metadata ({len(val_indices):,} samples)...")

    # Single-sample DataLoader for metadata pass (batch_size=256 for speed)
    meta_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=256, shuffle=False, num_workers=0,
    )

    cache = TokenCache()
    cache.register(model)

    all_confs   = []
    all_lengths = []
    all_gc      = []

    with torch.no_grad():
        for batch in tqdm(meta_loader, desc="    Metadata pass", leave=False):
            seq  = batch["sequence"].to(device)
            te_g = batch["te_genomic"].to(device)
            te_p = batch["te_processed"].to(device)
            nb_g = batch["nonb_genomic"].to(device)
            nb_p = batch["nonb_processed"].to(device)
            fwd  = dict(
                te_genomic=te_g, te_processed=te_p,
                nonb_genomic=nb_g, nonb_processed=nb_p,
                deterministic=True,
            )
            if has_nonb2 and "nonb2" in batch:
                fwd["nonb2"] = batch["nonb2"].to(device)

            out   = model(seq, **fwd)
            probs = torch.softmax(out["logits"], dim=1)
            confs = probs.max(1).values.cpu().numpy()
            all_confs.append(confs)

            # Sequence length from one-hot: sum of non-N positions
            seq_np = seq.cpu().numpy()
            lengths = seq_np[:, :4, :].sum(axis=(1, 2)).astype(np.float32)
            all_lengths.append(lengths)

            # GC fraction
            gc = ((seq[:, 1, :] + seq[:, 2, :]).sum(dim=1)
                  / seq[:, :4, :].sum(dim=(1, 2)).clamp(min=1)).cpu().numpy()
            all_gc.append(gc.astype(np.float32))

    cache.remove()

    confidences = np.concatenate(all_confs)
    lengths     = np.concatenate(all_lengths)
    gc_content  = np.concatenate(all_gc)

    # Diagnostics — print confidence distribution to help debug pair selection
    print(f"  Confidence stats: min={confidences.min():.3f}  "
          f"max={confidences.max():.3f}  "
          f"mean={confidences.mean():.3f}  "
          f">=0.7: {(confidences >= 0.7).sum()}  "
          f">=0.5: {(confidences >= 0.5).sum()}")
    label_arr_diag = np.array([labels[i] for i in val_indices])
    print(f"  Label distribution: lnc={(label_arr_diag=='lnc').sum()}  "
          f"mrna={(label_arr_diag=='pc').sum()}")
    lnc_conf  = confidences[label_arr_diag == "lnc"]
    mrna_conf = confidences[label_arr_diag == "pc"]
    print(f"  lnc  conf: min={lnc_conf.min():.3f}  max={lnc_conf.max():.3f}  "
          f"mean={lnc_conf.mean():.3f}  >=0.7: {(lnc_conf  >= 0.7).sum()}")
    print(f"  mrna conf: min={mrna_conf.min():.3f}  max={mrna_conf.max():.3f}  "
          f"mean={mrna_conf.mean():.3f}  >=0.7: {(mrna_conf >= 0.7).sum()}")

    # ── Step 2: select matched pairs ─────────────────────────────────────────
    rng = np.random.default_rng(seed=42 + fold_idx)
    lnc_local, mrna_local = select_pairs(
        val_indices     = val_indices,
        labels          = labels,
        lengths         = lengths,
        confidences     = confidences,
        n_pairs         = n_pairs,
        min_confidence  = min_confidence,
        max_length_diff = max_length_diff,
        max_gc_diff     = max_gc_diff,
        gc_content      = gc_content,
        rng             = rng,
    )
    n_actual = len(lnc_local)
    print(f"  Selected {n_actual} matched pairs")
    if n_actual == 0:
        warnings.warn(f"  No pairs matched for fold {fold_idx} — skipping")
        return pd.DataFrame()

    # Map local val-set indices back to dataset indices
    lnc_dataset_idx  = val_indices[lnc_local]
    mrna_dataset_idx = val_indices[mrna_local]

    # ── Step 3: cache token representations for all pairs ────────────────────
    # Pre-compute token representations for every lnc and mrna transcript
    # to avoid redundant forward passes in the patching loop.
    print(f"  Caching token representations for {n_actual * 2} transcripts...")

    def get_tokens_single(idx: int) -> torch.Tensor:
        """Return (N_tokens, d_proj) token tensor for dataset index idx."""
        sample = dataset[idx]
        batch  = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v]
                  for k, v in sample.items()}

        cache_inner = TokenCache()
        cache_inner.register(model)

        seq  = sample["sequence"].unsqueeze(0).to(device)
        te_g = sample["te_genomic"].unsqueeze(0).to(device)
        te_p = sample["te_processed"].unsqueeze(0).to(device)
        nb_g = sample["nonb_genomic"].unsqueeze(0).to(device)
        nb_p = sample["nonb_processed"].unsqueeze(0).to(device)
        fwd  = dict(
            te_genomic=te_g, te_processed=te_p,
            nonb_genomic=nb_g, nonb_processed=nb_p,
            deterministic=True,
        )
        if has_nonb2 and "nonb2" in sample:
            fwd["nonb2"] = sample["nonb2"].unsqueeze(0).to(device)

        with torch.no_grad():
            model(seq, **fwd)

        cache_inner.remove()
        return cache_inner.tokens[0]   # (N_tokens, d_proj)

    def get_logits_single(idx: int) -> tuple[float, float]:
        """Return (logit_mRNA, prob_mRNA) for dataset index idx."""
        sample = dataset[idx]
        seq  = sample["sequence"].unsqueeze(0).to(device)
        te_g = sample["te_genomic"].unsqueeze(0).to(device)
        te_p = sample["te_processed"].unsqueeze(0).to(device)
        nb_g = sample["nonb_genomic"].unsqueeze(0).to(device)
        nb_p = sample["nonb_processed"].unsqueeze(0).to(device)
        fwd  = dict(
            te_genomic=te_g, te_processed=te_p,
            nonb_genomic=nb_g, nonb_processed=nb_p,
            deterministic=True,
        )
        if has_nonb2 and "nonb2" in sample:
            fwd["nonb2"] = sample["nonb2"].unsqueeze(0).to(device)

        with torch.no_grad():
            out   = model(seq, **fwd)
            logit = out["logits"][0, 1].item()
            prob  = torch.softmax(out["logits"], dim=1)[0, 1].item()
        return logit, prob

    # Cache all token tensors upfront (avoids re-running token_norm per patch)
    lnc_tokens  = []   # list of (N_tokens, d_proj) tensors
    mrna_tokens = []

    for i in tqdm(range(n_actual), desc="    Caching lnc tokens",  leave=False):
        lnc_tokens.append(get_tokens_single(lnc_dataset_idx[i]))
    for i in tqdm(range(n_actual), desc="    Caching mRNA tokens", leave=False):
        mrna_tokens.append(get_tokens_single(mrna_dataset_idx[i]))

    # ── Step 4: activation patching loop ─────────────────────────────────────
    # For each subgroup token and each direction, run patched forward passes.
    # We patch AFTER token_norm so z is completely unaffected.

    n_tokens = len(token_names)
    rows     = []

    print(f"  Patching {n_tokens} tokens × 2 directions × {n_actual} pairs...")

    for token_idx, sg_name in enumerate(
        tqdm(token_names, desc="    Token loop", leave=False)
    ):
        block = block_map.get(sg_name, "unknown")

        for pair_i in range(n_actual):
            lnc_idx  = lnc_dataset_idx[pair_i]
            mrna_idx = mrna_dataset_idx[pair_i]

            lnc_tok  = lnc_tokens[pair_i]    # (N_tokens, d_proj)
            mrna_tok = mrna_tokens[pair_i]

            # ── Direction 1: lnc_to_mrna ──────────────────────────────────
            # Inject mRNA token into lncRNA forward pass
            # Baseline: lncRNA transcript (label=0), logit for class mRNA
            baseline_logit_lnc, baseline_prob_lnc = get_logits_single(lnc_idx)

            # Patched: replace token_idx with mRNA's token
            patch_vec = mrna_tok[token_idx].unsqueeze(0)  # (1, d_proj)

            sample_lnc = dataset[lnc_idx]
            seq  = sample_lnc["sequence"].unsqueeze(0).to(device)
            te_g = sample_lnc["te_genomic"].unsqueeze(0).to(device)
            te_p = sample_lnc["te_processed"].unsqueeze(0).to(device)
            nb_g = sample_lnc["nonb_genomic"].unsqueeze(0).to(device)
            nb_p = sample_lnc["nonb_processed"].unsqueeze(0).to(device)
            fwd_lnc = dict(
                te_genomic=te_g, te_processed=te_p,
                nonb_genomic=nb_g, nonb_processed=nb_p,
                deterministic=True,
            )
            if has_nonb2 and "nonb2" in sample_lnc:
                fwd_lnc["nonb2"] = sample_lnc["nonb2"].unsqueeze(0).to(device)

            def _patch_hook_lnc(module, input, output,
                                ti=token_idx, pv=patch_vec):
                patched = output.clone()
                patched[:, ti, :] = pv
                return patched

            ph = model.token_norm.register_forward_hook(_patch_hook_lnc)
            with torch.no_grad():
                out_p = model(seq, **fwd_lnc)
            ph.remove()

            patched_logit_lnc = out_p["logits"][0, 1].item()
            patched_prob_lnc  = torch.softmax(
                out_p["logits"], dim=1)[0, 1].item()

            rows.append({
                "subgroup":          sg_name,
                "block":             block,
                "token_idx":         token_idx,
                "direction":         "lnc_to_mrna",
                "pair_i":            pair_i,
                "lnc_dataset_idx":   int(lnc_idx),
                "mrna_dataset_idx":  int(mrna_idx),
                "baseline_logit":    baseline_logit_lnc,
                "patched_logit":     patched_logit_lnc,
                "delta_logit":       patched_logit_lnc - baseline_logit_lnc,
                "baseline_prob":     baseline_prob_lnc,
                "patched_prob":      patched_prob_lnc,
                "delta_prob":        patched_prob_lnc  - baseline_prob_lnc,
                "fold":              fold_idx,
            })

            # ── Direction 2: mrna_to_lnc ──────────────────────────────────
            # Inject lncRNA token into mRNA forward pass
            baseline_logit_mrna, baseline_prob_mrna = get_logits_single(mrna_idx)

            patch_vec_mrna = lnc_tok[token_idx].unsqueeze(0)  # (1, d_proj)

            sample_mrna = dataset[mrna_idx]
            seq  = sample_mrna["sequence"].unsqueeze(0).to(device)
            te_g = sample_mrna["te_genomic"].unsqueeze(0).to(device)
            te_p = sample_mrna["te_processed"].unsqueeze(0).to(device)
            nb_g = sample_mrna["nonb_genomic"].unsqueeze(0).to(device)
            nb_p = sample_mrna["nonb_processed"].unsqueeze(0).to(device)
            fwd_mrna = dict(
                te_genomic=te_g, te_processed=te_p,
                nonb_genomic=nb_g, nonb_processed=nb_p,
                deterministic=True,
            )
            if has_nonb2 and "nonb2" in sample_mrna:
                fwd_mrna["nonb2"] = sample_mrna["nonb2"].unsqueeze(0).to(device)

            def _patch_hook_mrna(module, input, output,
                                 ti=token_idx, pv=patch_vec_mrna):
                patched = output.clone()
                patched[:, ti, :] = pv
                return patched

            ph = model.token_norm.register_forward_hook(_patch_hook_mrna)
            with torch.no_grad():
                out_p2 = model(seq, **fwd_mrna)
            ph.remove()

            patched_logit_mrna = out_p2["logits"][0, 1].item()
            patched_prob_mrna  = torch.softmax(
                out_p2["logits"], dim=1)[0, 1].item()

            rows.append({
                "subgroup":          sg_name,
                "block":             block,
                "token_idx":         token_idx,
                "direction":         "mrna_to_lnc",
                "pair_i":            pair_i,
                "lnc_dataset_idx":   int(lnc_idx),
                "mrna_dataset_idx":  int(mrna_idx),
                "baseline_logit":    baseline_logit_mrna,
                "patched_logit":     patched_logit_mrna,
                "delta_logit":       patched_logit_mrna - baseline_logit_mrna,
                "baseline_prob":     baseline_prob_mrna,
                "patched_prob":      patched_prob_mrna,
                "delta_prob":        patched_prob_mrna  - baseline_prob_mrna,
                "fold":              fold_idx,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary(all_folds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-fold mean ± std of delta_logit and delta_prob per
    (subgroup, direction). Also computes a symmetry score:
        symmetry = (|mean_lnc_to_mrna| + |mean_mrna_to_lnc|) / 2
    High symmetry = consistent causal effect in both directions.
    """
    agg = (all_folds_df
           .groupby(["subgroup", "block", "direction"])
           .agg(
               mean_delta_logit = ("delta_logit", "mean"),
               std_delta_logit  = ("delta_logit", "std"),
               mean_delta_prob  = ("delta_prob",  "mean"),
               std_delta_prob   = ("delta_prob",  "std"),
               n_pairs          = ("delta_logit", "count"),
           )
           .reset_index())

    n_folds = all_folds_df["fold"].nunique()
    agg["n_folds"] = n_folds

    # Compute symmetry score: mean |Δ| across both directions per subgroup
    pivot = (agg.pivot(index=["subgroup", "block"],
                       columns="direction",
                       values="mean_delta_logit")
             .reset_index())
    pivot.columns.name = None
    if "lnc_to_mrna" in pivot.columns and "mrna_to_lnc" in pivot.columns:
        pivot["symmetry_score"] = (
            pivot["lnc_to_mrna"].abs() + pivot["mrna_to_lnc"].abs()
        ) / 2
        # lnc_to_mrna should be positive, mrna_to_lnc should be negative
        # sign-corrected: flip mrna_to_lnc so both point in same direction
        pivot["causal_effect"] = (
            pivot["lnc_to_mrna"] - pivot["mrna_to_lnc"]
        ) / 2
        pivot = pivot.sort_values("symmetry_score", ascending=False)
        agg = agg.merge(
            pivot[["subgroup", "symmetry_score", "causal_effect"]],
            on="subgroup", how="left"
        )

    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Token activation patching for β-LNC causal interpretability",
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
                        help="Pairs to sample per fold per direction (default: 200)")
    parser.add_argument("--min_confidence",  type=float, default=0.7,
                        help="Minimum classification confidence for pair selection")
    parser.add_argument("--max_length_diff", type=float, default=0.2,
                        help="Max fractional length difference between pairs")
    parser.add_argument("--max_gc_diff",     type=float, default=0.05,
                        help="Max absolute GC content difference between pairs")
    parser.add_argument("--batch_size",      type=int,   default=None)
    args = parser.parse_args()

    config     = load_config(args.config)
    exp_dir    = Path(args.experiment_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device     = torch.device(args.device)

    print("=" * 70)
    print("BetaVAESubgroup — Token Activation Patching")
    print("=" * 70)
    print(f"Experiment : {exp_dir}")
    print(f"Config     : {args.config}")
    print(f"Output dir : {output_dir}")
    print(f"Device     : {args.device}")
    print(f"Pairs/fold : {args.n_pairs}")
    print(f"Min conf   : {args.min_confidence}")
    print(f"Max Δlen   : {args.max_length_diff:.0%}")
    print(f"Max ΔGC    : {args.max_gc_diff}")
    print("=" * 70)

    # ── Token metadata from registry ──────────────────────────────────────────
    all_subgroups = REGISTRY.all_subgroups
    token_names   = all_subgroups
    block_map     = {sg: REGISTRY.token_block(sg) for sg in all_subgroups}

    print(f"\nRegistry: {REGISTRY.total_tokens} tokens")
    for b in REGISTRY.block_names:
        print(f"  {b}: {REGISTRY.block_subgroups(b)}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("\nLoading sequences and dataset...")
    all_sequences, labels = load_sequences_in_order(
        lnc_fasta=config.get("data", "lnc_fasta"),
        pc_fasta= config.get("data", "pc_fasta"),
    )

    has_nonb2    = config.get("data", "nonb2_csv", default=None) is not None
    nonb2_csv    = config.get("data", "nonb2_csv",              default=None)
    nonb2_scaler = config.get("data", "nonb2_scaler_bank_path", default=None)

    if nonb2_csv is not None and nonb2_scaler is None:
        raise ValueError(
            "Config has nonb2_csv but no nonb2_scaler_bank_path."
        )

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

    # ── Reconstruct CV splits ─────────────────────────────────────────────────
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

    # ── Find fold checkpoints ─────────────────────────────────────────────────
    model_dir  = exp_dir / "models"
    fold_files = sorted(model_dir.glob("fold_*_best.pt"))
    if not fold_files:
        raise FileNotFoundError(
            f"No fold checkpoints found in {model_dir}"
        )
    print(f"\nFound {len(fold_files)} fold checkpoint(s)")

    # ── Per-fold patching ─────────────────────────────────────────────────────
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
        )

        if fold_df.empty:
            print(f"    No results for fold {fold_idx} — skipping")
            continue

        fold_df.to_csv(save_path, index=False)
        print(f"  Saved fold results → {save_path}")
        print(f"  Rows: {len(fold_df):,}  "
              f"({fold_df['direction'].nunique()} directions × "
              f"{fold_df['subgroup'].nunique()} subgroups × "
              f"{fold_df['pair_i'].nunique()} pairs)")

        # Quick sanity print: top-5 subgroups by |Δlogit| in lnc_to_mrna
        lnc_df = fold_df[fold_df["direction"] == "lnc_to_mrna"]
        top5   = (lnc_df.groupby("subgroup")["delta_logit"]
                  .mean().abs().nlargest(5))
        print(f"  Top-5 |Δlogit| lnc→mRNA: {top5.to_dict()}")

        all_fold_dfs.append(fold_df)

        # Free GPU memory between folds
        del model
        torch.cuda.empty_cache()

    # ── Aggregate and save ────────────────────────────────────────────────────
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

    # ── Print ranked summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PATCHING SUMMARY — lnc_to_mrna direction (ranked by |Δlogit|)")
    print("=" * 70)
    lnc_summary = (summary_df[summary_df["direction"] == "lnc_to_mrna"]
                   .sort_values("mean_delta_logit", ascending=False))
    print(f"{'Subgroup':<15}  {'Block':<8}  {'Mean Δlogit':>12}  "
          f"{'Std':>8}  {'Mean ΔP(mRNA)':>13}  {'n_pairs':>8}")
    print("-" * 70)
    for _, row in lnc_summary.iterrows():
        print(f"  {row['subgroup']:<13}  {row['block']:<8}  "
              f"{row['mean_delta_logit']:>+12.4f}  "
              f"{row['std_delta_logit']:>8.4f}  "
              f"{row['mean_delta_prob']:>+13.4f}  "
              f"{int(row['n_pairs']):>8}")

    if "symmetry_score" in summary_df.columns:
        print("\n" + "=" * 70)
        print("PATCHING SUMMARY — symmetry score (both directions, ranked)")
        print("(Higher = more consistent causal effect across both directions)")
        print("=" * 70)
        sym = (summary_df[summary_df["direction"] == "lnc_to_mrna"]
               .sort_values("symmetry_score", ascending=False)
               [["subgroup", "block", "symmetry_score", "causal_effect"]]
               .drop_duplicates())
        for _, row in sym.iterrows():
            print(f"  {row['subgroup']:<15}  {row['block']:<8}  "
                  f"symmetry={row['symmetry_score']:.4f}  "
                  f"causal_effect={row['causal_effect']:+.4f}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")
    print(f"  all_folds_patching.csv → {all_path}")
    print(f"  patching_summary.csv   → {summary_path}")


if __name__ == "__main__":
    main()