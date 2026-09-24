#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_mstar.py

Constructs the M* co-discrimination matrix for lncRNA/mRNA classification
using processed transcript sub-group features, then diagonalises it to find
the theoretically optimal discriminative directions in sub-group space.

Motivation
----------
Inspired by the word2vec theory (Karkada et al. 2025): the features a model
learns are the top eigenvectors of a matrix capturing normalised excess
co-occurrence between class membership and feature values. Here we construct
the biological analogue:

  M*[class, sub-group] = (mu_class[s] - mu_all[s]) / std_all[s]

where mu_class[s] is the mean value of sub-group s for transcripts of that
class. Diagonalising M*^T M* gives the sub-group combinations that maximally
discriminate lncRNA from mRNA, independent of any model.

This provides:
  1. A model-free ground truth for sub-group importance ranking
  2. Effect sizes for synthetic benchmark design (eigenvalue magnitudes)
  3. A reference for evaluating whether our model recovers the optimal
     discriminative structure (comparison with ablation and Pattern results)

Usage
-----
python analysis/pattern/analyze_mstar.py \
    --config         configs/beta_vae_subgroup_base_g49.json \
    --output_dir     gencode_v49_experiments/mstar_analysis \
    --n_top_eigen    5

Output
------
mstar_analysis/
  mstar_matrix.csv            M* (2 × n_subgroups) — class × sub-group excess
  eigenvalues.csv             eigenvalue spectrum with cumulative variance
  eigenvectors.csv            top eigenvector loadings per sub-group
  class_profiles.csv          class-conditional mean ± std per sub-group
  mstar_heatmap.png           M* visualised as heatmap
  eigenvalue_spectrum.png     scree plot
  eigenvector_loadings.png    top-k eigenvectors as bar charts
  class_profile_comparison.png  lncRNA vs mRNA sub-group profiles
  summary.txt                 key findings in plain text
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.linalg import eigh as scipy_eigh, sqrtm as scipy_sqrtm
from sklearn.preprocessing import StandardScaler

from configs.load_config import load_config
from data.cv_utils import load_sequences_in_order
from data.feature_registry import (REGISTRY, display_block,
                                    display_subgroup, display_subgroups)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

# ── Sub-group lists and block colour maps (filled from REGISTRY at runtime) ──
NONB_SUBGROUPS = None
TE_SUBGROUPS   = None
ALL_SUBGROUPS  = None   # ordered across all blocks

# Block colour palette — extend when new blocks are added
BLOCK_COLORS = {
    "nonb":  "#4A90D9AA",
    "te":    "#EF9F27AA",
    "nonb2": "#7B68EEAA",
}
BLOCK_EDGES = {
    "nonb":  "#185FA5",
    "te":    "#BA7517",
    "nonb2": "#4B3F9E",
}

def _sg_color(sg: str, alpha: bool = True) -> str:
    block = REGISTRY.token_block(sg)
    return BLOCK_COLORS[block] if alpha else BLOCK_COLORS[block].rstrip("A")[:7]

def _sg_edge(sg: str) -> str:
    return BLOCK_EDGES[REGISTRY.token_block(sg)]


# ---------------------------------------------------------------------------
# Feature loading and sub-group aggregation
# ---------------------------------------------------------------------------

def load_features(config) -> tuple[dict, np.ndarray]:
    """
    Load processed feature CSVs and return:
      block_dfs : {block_name: pd.DataFrame} aligned to sequence order
      labels    : (N,) int — 0=lncRNA 1=mRNA
    """
    print("Loading sequences and labels...")
    all_sequences, labels_str = load_sequences_in_order(
        lnc_fasta=config.get("data", "lnc_fasta"),
        pc_fasta= config.get("data", "pc_fasta"),
    )
    labels = np.array([0 if l == "lnc" else 1 for l in labels_str])
    print(f"  lncRNA: {(labels==0).sum():,}  mRNA: {(labels==1).sum():,}")
 
    # Transcript IDs for alignment — strip pipe-suffix AND version suffix.
    # Must match however the feature CSV indices are normalised below.
    ids = [str(seq.id).split("|")[0].split(".")[0] for seq in all_sequences]
 
    def _load_and_dedupe(path: str, label: str) -> pd.DataFrame:
        print(f"Loading {label} features...")
        df = pd.read_csv(path, index_col=0)
        print(f"  Raw shape: {df.shape}")
 
        # Normalise index the same way as the sequence IDs above
        df.index = (df.index.astype(str)
                    .str.split("|").str[0]
                    .str.split(".").str[0])
 
        n_before = len(df)
        n_unique = df.index.nunique()
        if n_unique < n_before:
            n_dupes = n_before - n_unique
            print(f"  WARNING: {n_dupes:,} duplicate transcript_id rows "
                  f"({n_before:,} rows, {n_unique:,} unique) — "
                  f"keeping first occurrence")
            df = df[~df.index.duplicated(keep="first")]
 
        print(f"  Deduped shape: {df.shape}")
        return df
 
    # ── NonB ──────────────────────────────────────────────────────────────
    nonb_path = config.get("data", "nonb_processed_csv", default=None)
    if nonb_path is None:
        nonb_path = config.get("data", "nonb_genomic_csv")
        print("  (nonb_processed_csv not set, using genomic)")
    nonb_df = _load_and_dedupe(nonb_path, "NonB processed")
 
    # ── TE ────────────────────────────────────────────────────────────────
    te_path = config.get("data", "te_processed_csv", default=None)
    if te_path is None:
        te_path = config.get("data", "te_genomic_csv")
        print("  (te_processed_csv not set, using genomic)")
    te_df = _load_and_dedupe(te_path, "TE processed")
 
    # ── NonB2 (optional) ─────────────────────────────────────────────────
    nonb2_path = config.get("data", "nonb2_csv", default=None)
    if nonb2_path is None:
        nonb2_path = config.get("data", "nonb2_processed_csv", default=None)
        if nonb2_path is not None:
            print("  (nonb2_csv not set, using nonb2_processed_csv)")
    nonb2_df = (_load_and_dedupe(nonb2_path, "NonB2")
                if nonb2_path is not None else None)
 
    # ── Align all blocks to sequence order ──────────────────────────────
    # Safe now — every DataFrame has a unique index after dedup above
    nonb_df = nonb_df.reindex(ids)
    te_df   = te_df.reindex(ids)
    if nonb2_df is not None:
        nonb2_df = nonb2_df.reindex(ids)
 
    # ── Drop rows with all-NaN in ANY block (unmatched transcripts) ────
    valid = (~nonb_df.isnull().all(axis=1)) & (~te_df.isnull().all(axis=1))
    if nonb2_df is not None:
        valid = valid & (~nonb2_df.isnull().all(axis=1))
 
    print(f"  Valid transcripts after alignment: {valid.sum():,} / {len(valid):,}")
 
    # Use a numpy boolean array, not the pandas Series, to avoid any
    # remaining index-alignment surprises when filtering each DataFrame
    valid_arr = valid.values
 
    nonb_df = nonb_df[valid_arr].fillna(0.0)
    te_df   = te_df[valid_arr].fillna(0.0)
    labels  = labels[valid_arr]
 
    block_dfs = {"nonb": nonb_df, "te": te_df}
    if nonb2_df is not None:
        nonb2_df = nonb2_df[valid_arr].fillna(0.0)
        block_dfs["nonb2"] = nonb2_df
 
    return block_dfs, labels


def aggregate_to_subgroups(
    block_dfs: dict,   # {block_name: pd.DataFrame}
) -> pd.DataFrame:
    """
    Aggregate feature columns to sub-group level (mean per sub-group).
    Handles all blocks defined in REGISTRY dynamically.

    Returns
    -------
    subgroup_df : (N, n_tokens) DataFrame — one column per sub-group
    """
    global NONB_SUBGROUPS, TE_SUBGROUPS, ALL_SUBGROUPS
    NONB_SUBGROUPS = list(REGISTRY.nonb_subgroups)
    TE_SUBGROUPS   = list(REGISTRY.te_subgroups)
    ALL_SUBGROUPS  = REGISTRY.all_subgroups

    # Use first block df as index reference
    ref_df = next(iter(block_dfs.values()))

    print("\nAggregating features to sub-group level...")
    subgroup_data = {}

    for block_name in REGISTRY.block_names:
        if block_name not in block_dfs:
            print(f"  WARNING: block '{block_name}' not in block_dfs — skipping")
            continue
        df = block_dfs[block_name]
        for sg in REGISTRY.block_subgroups(block_name):
            idx = REGISTRY.indices_for_subgroup(sg)
            if idx:
                cols = df.columns[idx].tolist()
            else:
                print(f"  WARNING: no indices for sub-group {sg}")
                cols = []

            if cols:
                subgroup_data[sg] = df[cols].mean(axis=1).values
            else:
                subgroup_data[sg] = np.zeros(len(df))

            print(f"  {sg:15s} [{block_name}] ({len(cols):3d} features) -> mean scalar")

    subgroup_df = pd.DataFrame(subgroup_data, index=ref_df.index)
    print(f"\nSub-group matrix: {subgroup_df.shape}")
    return subgroup_df


# ---------------------------------------------------------------------------
# M* construction
# ---------------------------------------------------------------------------

def construct_mstar(
    X:      np.ndarray,   # (N, n_subgroups) z-scored sub-group matrix
    labels: np.ndarray,   # (N,)    0=lncRNA 1=mRNA
) -> np.ndarray:
    """
    Construct M* — normalised excess class-conditional sub-group deviation.

    M*[c, s] = (mu_class[c, s] - mu_all[s]) / std_all[s]

    Returns
    -------
    M_star : (2, n_subgroups)
    """
    mu_all  = X.mean(axis=0)          # (n_subgroups,)
    std_all = X.std(axis=0).clip(min=1e-8)

    M_star = np.zeros((2, len(ALL_SUBGROUPS)))
    for c in [0, 1]:
        mask       = labels == c
        mu_class   = X[mask].mean(axis=0)
        M_star[c]  = (mu_class - mu_all) / std_all

    return M_star


# ---------------------------------------------------------------------------
# Eigendecomposition
# ---------------------------------------------------------------------------

def decompose_mstar(M_star: np.ndarray) -> dict:
    """
    Diagonalise M*^T M* to find optimal discriminative directions.

    Returns dict with eigenvalues, eigenvectors, explained variance.
    """
    C = M_star.T @ M_star   # (n_subgroups, n_subgroups) — sub-group co-discrimination matrix

    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # Sort descending
    idx          = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]   # columns are eigenvectors

    # Explained variance
    total_var       = eigenvalues.sum()
    explained_var   = eigenvalues / total_var
    cumulative_var  = np.cumsum(explained_var)

    # Sign convention: largest absolute loading is positive
    for i in range(eigenvectors.shape[1]):
        if eigenvectors[:, i][np.abs(eigenvectors[:, i]).argmax()] < 0:
            eigenvectors[:, i] *= -1

    return {
        "eigenvalues":   eigenvalues,
        "eigenvectors":  eigenvectors,
        "explained_var": explained_var,
        "cumulative_var":cumulative_var,
        "C":             C,
    }


# ---------------------------------------------------------------------------
# LDA generalised eigendecomposition
# ---------------------------------------------------------------------------

def lda_analysis(
    X:      np.ndarray,   # (N, n_subgroups) z-scored
    labels: np.ndarray,   # (N,)
) -> dict:
    """
    Linear Discriminant Analysis via generalised eigenvalue problem.

    Solves: Σ_b w = λ Σ_w w
    Equivalent to: Σ_w^{-1} Σ_b w = λ w

    Fisher's criterion: J(w) = (wᵀ Σ_b w) / (wᵀ Σ_w w)

    Unlike M*ᵀM* which only captures between-class differences,
    LDA accounts for within-class spread. A sub-group with a large
    between-class difference but high within-class variance is
    down-weighted relative to one with a smaller difference but
    tighter within-class distribution.

    Returns
    -------
    dict with:
      Sigma_w        : (n_subgroups, n_subgroups) pooled within-class covariance
      Sigma_b        : (n_subgroups, n_subgroups) between-class scatter
      Sigma_lnc      : (n_subgroups, n_subgroups) lncRNA within-class covariance
      Sigma_mrna     : (n_subgroups, n_subgroups) mRNA within-class covariance
      lda_direction  : (n_subgroups,) optimal LDA direction
      fisher_ratio   : scalar — between/within variance ratio
      within_std     : (n_subgroups,) within-class std per sub-group (pooled)
      snr            : (n_subgroups,) signal-to-noise ratio per sub-group
                       = |between-class mean diff| / within-class std
    """
    from scipy.linalg import eigh, lstsq

    X_lnc  = X[labels == 0]
    X_mrna = X[labels == 1]
    p_lnc  = len(X_lnc)  / len(X)
    p_mrna = len(X_mrna) / len(X)

    # Within-class covariances
    Sigma_lnc  = np.cov(X_lnc.T)
    Sigma_mrna = np.cov(X_mrna.T)
    Sigma_w    = p_lnc * Sigma_lnc + p_mrna * Sigma_mrna

    # Between-class scatter
    delta    = X_lnc.mean(0) - X_mrna.mean(0)   # (n_subgroups,)
    Sigma_b  = np.outer(delta, delta)             # (n_subgroups, n_subgroups)

    # Generalised eigenvalue problem: Sigma_b w = lambda Sigma_w w
    # Use regularised Sigma_w for numerical stability
    reg      = 1e-6 * np.eye(Sigma_w.shape[0])
    try:
        eigenvalues, eigenvectors = eigh(Sigma_b, Sigma_w + reg)
    except Exception:
        # Fallback: direct inversion
        Sigma_w_inv  = np.linalg.pinv(Sigma_w + reg)
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma_w_inv @ Sigma_b)

    # Sort descending
    idx          = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # LDA direction (first eigenvector)
    lda_dir = eigenvectors[:, 0]
    if lda_dir[np.abs(lda_dir).argmax()] < 0:
        lda_dir *= -1

    # Per sub-group signal-to-noise ratio
    within_std = np.sqrt(np.diag(Sigma_w)).clip(min=1e-8)
    snr        = np.abs(delta) / within_std

    return {
        "Sigma_w":       Sigma_w,
        "Sigma_b":       Sigma_b,
        "Sigma_lnc":     Sigma_lnc,
        "Sigma_mrna":    Sigma_mrna,
        "lda_direction": lda_dir,
        "fisher_ratio":  float(eigenvalues[0]),
        "within_std":    within_std,
        "snr":           snr,
        "delta":         delta,
    }


# ---------------------------------------------------------------------------
# Within-class covariance plots
# ---------------------------------------------------------------------------

def plot_within_class_covariance(
    lda_result:  dict,
    output_path: Path,
) -> None:
    """Two-panel: within-class covariance heatmaps for lncRNA and mRNA."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    vmax = max(np.abs(lda_result["Sigma_lnc"]).max(),
               np.abs(lda_result["Sigma_mrna"]).max())

    tick_colors = [_sg_edge(sg) for sg in ALL_SUBGROUPS]

    for ax, mat, title in zip(
        axes,
        [lda_result["Sigma_lnc"], lda_result["Sigma_mrna"]],
        ["lncRNA — Within-class Covariance", "mRNA — Within-class Covariance"],
    ):
        im = ax.imshow(mat, cmap="RdBu_r", aspect="auto",
                       vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(ALL_SUBGROUPS)))
        ax.set_yticks(range(len(ALL_SUBGROUPS)))
        ax.set_xticklabels(display_subgroups(ALL_SUBGROUPS), rotation=90, fontsize=8)
        ax.set_yticklabels(display_subgroups(ALL_SUBGROUPS), fontsize=8)
        for tick, color in zip(ax.get_xticklabels(), tick_colors):
            tick.set_color(color)
        for tick, color in zip(ax.get_yticklabels(), tick_colors):
            tick.set_color(color)
        ax.set_title(title, fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Mark boundary — must be inside the panel loop so both
        # panels get the boundary lines, not just the last one drawn
        offset = 0
        for bn in REGISTRY.block_names[:-1]:
            offset += len(REGISTRY.block_subgroups(bn))
            ax.axhline(offset - 0.5, color="white", lw=1.5)
            ax.axvline(offset - 0.5, color="white", lw=1.5)

    fig.suptitle("Within-class Sub-group Covariance Structure",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=1000, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_snr_ranking(
    lda_result:  dict,
    output_path: Path,
) -> None:
    """
    Bar chart of signal-to-noise ratio per sub-group.
    SNR = |between-class mean diff| / within-class std (pooled)

    This is the Fisher-corrected discrimination score — accounts for
    within-class variance unlike the raw M* difference.
    """
    snr    = lda_result["snr"]
    order  = np.argsort(snr)[::-1]
    n_nonb = len(NONB_SUBGROUPS)

    ordered_names = [ALL_SUBGROUPS[i] for i in order]
    ordered_snr   = snr[order]
    colors = [_sg_color(ALL_SUBGROUPS[i]) for i in order]
    edges  = [_sg_edge(ALL_SUBGROUPS[i]) for i in order]

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("white")
    ax.bar(range(len(ordered_names)), ordered_snr,
           color=colors, edgecolor=edges, linewidth=0.8)
    ax.set_xticks(range(len(ordered_names)))
    ax.set_xticklabels(display_subgroups(ordered_names), rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("|Δμ| / σ_within  (signal-to-noise ratio)", fontsize=11)
    ax.set_title(
        "Sub-group Signal-to-Noise Ratio — Fisher-Corrected Discrimination\n"
        "(between-class mean difference / pooled within-class std)",
        fontsize=12, fontweight="bold"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(handles=[
        mpatches.Patch(facecolor=BLOCK_COLORS[b], edgecolor=BLOCK_EDGES[b],
                       label=display_block(b))
        for b in REGISTRY.block_names
    ], fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_lda_vs_mstar(
    lda_result:  dict,
    profiles_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Side-by-side comparison: M* ranking (raw diff) vs SNR ranking
    (Fisher-corrected). Shows how within-class variance reshuffles the ranking.
    """
    # Align both by M* order
    mstar_order = profiles_df["subgroup"].tolist()
    snr_vals    = {sg: lda_result["snr"][ALL_SUBGROUPS.index(sg)]
                   for sg in mstar_order}
    diff_vals   = {sg: profiles_df.set_index("subgroup").loc[sg, "abs_diff"]
                   for sg in mstar_order}

    x     = np.arange(len(mstar_order))
    width = 0.35
    n_nonb = len(NONB_SUBGROUPS)

    colors_raw = [_sg_color(sg) for sg in mstar_order]
    colors_snr = [_sg_color(sg) for sg in mstar_order]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("white")

    ax.bar(x - width/2,
           [diff_vals[sg] for sg in mstar_order],
           width, label="M* |diff| (raw)", color=colors_raw,
           edgecolor="none", alpha=0.85)
    ax.bar(x + width/2,
           [snr_vals[sg] for sg in mstar_order],
           width, label="SNR (Fisher-corrected)", color=colors_snr,
           edgecolor="none", alpha=0.85)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(display_subgroups(mstar_order), rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Discrimination score", fontsize=11)
    ax.set_title(
        "M* Raw Difference vs Fisher SNR — Effect of Within-class Variance\n"
        "(sub-groups ordered by M* ranking)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_within_std(
    lda_result:  dict,
    output_path: Path,
) -> None:
    """
    Within-class standard deviation per sub-group — effect size reference
    for synthetic benchmark design.
    """
    within_std = lda_result["within_std"]
    order      = np.argsort(within_std)[::-1]
    n_nonb     = len(NONB_SUBGROUPS)

    ordered_names = [ALL_SUBGROUPS[i] for i in order]
    ordered_std   = within_std[order]
    colors = [_sg_color(ALL_SUBGROUPS[i]) for i in order]
    edges  = [_sg_edge(ALL_SUBGROUPS[i]) for i in order]

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("white")
    ax.bar(range(len(ordered_names)), ordered_std,
           color=colors, edgecolor=edges, linewidth=0.8)
    ax.axhline(1.0, color="#aaa", linestyle="--", lw=1)
    ax.set_xticks(range(len(ordered_names)))
    ax.set_xticklabels(display_subgroups(ordered_names), rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Pooled within-class std (z-scored units)", fontsize=11)
    ax.set_title(
        "Within-class Sub-group Variance\n"
        "(higher = more heterogeneous within class — harder to discriminate)",
        fontsize=12, fontweight="bold"
    )
    ax.text(len(ordered_names) - 0.5, 1.02, "population std = 1.0",
            fontsize=8, color="#999", ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(handles=[
        mpatches.Patch(facecolor=BLOCK_COLORS[b], edgecolor=BLOCK_EDGES[b],
                       label=display_block(b))
        for b in REGISTRY.block_names
    ], fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")



# ---------------------------------------------------------------------------
# Fréchet distance per sub-group
# ---------------------------------------------------------------------------

def frechet_analysis(
    block_dfs: dict,           # {block_name: pd.DataFrame} raw features per block
    labels:    np.ndarray,     # (N,) 0=lncRNA 1=mRNA
    reg:       float = 1e-4,   # diagonal regularisation
) -> dict:
    """
    Compute per-sub-group Fréchet distance between lncRNA and mRNA
    class-conditional distributions.

    The Fréchet (Wasserstein-2) distance between two Gaussians
    N(μ₁, Σ₁) and N(μ₂, Σ₂) is:

        FD² = ||μ₁ - μ₂||² + trace(Σ₁ + Σ₂ - 2·(Σ₁^{1/2} Σ₂ Σ₁^{1/2})^{1/2})

    The first term captures mean shift (equivalent to M* squared).
    The second term captures covariance differences — invisible to M*.

    For each sub-group sg with k raw features:
      - Fit N(μ_lnc, Σ_lnc) and N(μ_mrna, Σ_mrna) on the k raw features
      - Compute FD(sg) = Wasserstein-2 distance between the two Gaussians
      - Decompose into mean_term and cov_term to show M* vs covariance contribution

    Returns
    -------
    dict with:
      frechet_scores : (n_subgroups,) — FD per sub-group
      mean_terms     : (n_subgroups,) — ||μ_lnc - μ_mrna||² per sub-group
      cov_terms      : (n_subgroups,) — covariance contribution per sub-group
      subgroup_names : list of sub-group names in order
    """
    from sklearn.preprocessing import StandardScaler

    frechet_scores = []
    mean_terms     = []
    cov_terms      = []

    for block_name in REGISTRY.block_names:
        df = block_dfs.get(block_name)
        if df is None:
            continue
        for sg in REGISTRY.block_subgroups(block_name):
            idx  = REGISTRY.indices_for_subgroup(sg)
            if not idx:
                frechet_scores.append(0.0)
                mean_terms.append(0.0)
                cov_terms.append(0.0)
                continue

            X_sg = df.iloc[:, idx].values.astype(np.float64)

            # Z-score within this sub-group for numerical stability
            sc   = StandardScaler()
            X_z  = sc.fit_transform(X_sg)

            X_lnc  = X_z[labels == 0]
            X_mrna = X_z[labels == 1]

            mu1 = X_lnc.mean(axis=0)
            mu2 = X_mrna.mean(axis=0)

            S1 = np.cov(X_lnc.T) + reg * np.eye(X_z.shape[1])
            S2 = np.cov(X_mrna.T) + reg * np.eye(X_z.shape[1])

            # Handle 1-feature sub-groups (scalar covariance)
            if S1.ndim == 0:
                S1 = np.array([[float(S1)]])
                S2 = np.array([[float(S2)]])

            # Mean term: ||μ₁ - μ₂||²
            mean_term = float(np.sum((mu1 - mu2) ** 2))

            # Covariance term: trace(S1 + S2 - 2·(S1^{1/2} S2 S1^{1/2})^{1/2})
            try:
                S1_sqrt     = scipy_sqrtm(S1).real
                inner       = S1_sqrt @ S2 @ S1_sqrt
                inner_sqrt  = scipy_sqrtm(inner).real
                cov_term    = float(np.trace(S1 + S2 - 2 * inner_sqrt))
                cov_term    = max(cov_term, 0.0)   # numerical guard
            except Exception:
                cov_term = 0.0

            fd = mean_term + cov_term

            frechet_scores.append(fd)
            mean_terms.append(mean_term)
            cov_terms.append(cov_term)

    return {
        "frechet_scores": np.array(frechet_scores),
        "mean_terms":     np.array(mean_terms),
        "cov_terms":      np.array(cov_terms),
        "subgroup_names": ALL_SUBGROUPS,
    }


def plot_frechet_ranking(
    frechet_result: dict,
    profiles_df:    pd.DataFrame,
    output_path:    Path,
) -> None:
    """
    Stacked bar chart showing Fréchet distance per sub-group, decomposed
    into mean term (equivalent to M* squared) and covariance term.
    Sub-groups ordered by total Fréchet distance descending.
    """
    scores = frechet_result["frechet_scores"]
    means  = frechet_result["mean_terms"]
    covs   = frechet_result["cov_terms"]
    order  = np.argsort(scores)[::-1]
    n_nonb = len(NONB_SUBGROUPS)

    ordered_names  = [ALL_SUBGROUPS[i] for i in order]
    ordered_means  = means[order]
    ordered_covs   = covs[order]
    edge_colors    = ["#185FA5" if i < n_nonb else "#BA7517" for i in order]

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("white")

    x = np.arange(len(ordered_names))
    ax.bar(x, ordered_means, label="Mean shift (||Δμ||²)",
           color="#4A90D9AA", edgecolor="none")
    ax.bar(x, ordered_covs, bottom=ordered_means,
           label="Covariance term", color="#EF9F27AA", edgecolor="none")

    # Edge colours by block
    for xi, edge in zip(x, edge_colors):
        ax.get_children()[xi].set_edgecolor(edge)
        ax.get_children()[xi].set_linewidth(0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(display_subgroups(ordered_names), rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Fréchet distance (W₂² between class Gaussians)", fontsize=11)
    ax.set_title(
        "Sub-group Fréchet Distance — Multi-dimensional Discrimination "
        "(mean shift + covariance difference)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_frechet_vs_pattern(
    fd_df:              pd.DataFrame,   # subgroup, block, frechet_dist, mean_term, cov_term
    pattern_raw_df:     pd.DataFrame,   # subgroup, mean_align, std_align
    output_path:        Path,
    divergence_frechet_quantile: float = 0.6,
    divergence_align_threshold: float = 0.1,
) -> None:
    """
    Two-panel comparison, shared x-axis sorted by Fréchet distance:
    top = Fréchet distance (mean shift + covariance term, stacked),
    bottom = raw pattern-weight alignment (cosine similarity).

    A subgroup is marked with an asterisk in the bottom panel when its
    Fréchet distance is in the top (1 - divergence_frechet_quantile)
    fraction AND its mean raw alignment is below divergence_align_threshold
    — i.e. a subgroup the associational layer's more rigorous method
    (Fréchet) flags as discriminative, but the cheaper pattern-alignment
    check would have missed.
    """
    df = fd_df.merge(pattern_raw_df, on="subgroup", how="left")
    df = df.sort_values("frechet_dist", ascending=False).reset_index(drop=True)
    names = df["subgroup"].tolist()

    edge_colors = [BLOCK_EDGES[REGISTRY.token_block(sg)] for sg in names]
    face_colors = [BLOCK_COLORS[REGISTRY.token_block(sg)] for sg in names]

    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1]},
    )
    fig.patch.set_facecolor("white")

    ax1.bar(x, df["mean_term"], label="Mean shift (||Δμ||²)",
            color="#4A90D9AA", edgecolor="none")
    ax1.bar(x, df["cov_term"], bottom=df["mean_term"],
            label="Covariance term", color="#EF9F27AA", edgecolor="none")
    for xi, edge in zip(x, edge_colors):
        ax1.get_children()[xi].set_edgecolor(edge)
        ax1.get_children()[xi].set_linewidth(0.8)
    ax1.set_ylabel("Fréchet distance (W₂²)", fontsize=11)
    ax1.set_title(
        "Associational Layer: Fréchet Distance vs Pattern Alignment",
        fontsize=13, fontweight="bold"
    )
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.bar(x, df["mean_align"], color=face_colors, edgecolor=edge_colors,
            linewidth=0.8, alpha=0.9)
    if "std_align" in df.columns:
        ax2.errorbar(x, df["mean_align"], yerr=df["std_align"],
                    fmt="none", color="black", capsize=3, linewidth=1.0)
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_ylabel("Pattern alignment\n(cosine similarity, raw)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(display_subgroups(names), rotation=45, ha="right", fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fd_threshold = df["frechet_dist"].quantile(divergence_frechet_quantile)
    for xi, row in zip(x, df.itertuples()):
        if (row.frechet_dist > fd_threshold
                and abs(row.mean_align) < divergence_align_threshold):
            y = row.mean_align + (row.std_align if "std_align" in df.columns else 0)
            ax2.annotate("*", (xi, y), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=15, fontweight="bold", color="#C0392B")

    fig.text(
        0.5, -0.02,
        "* high Fréchet distance, low pattern alignment — flagged by the "
        "associational layer, missed by naive pattern alignment",
        fontsize=9, color="#C0392B", ha="center",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_frechet_vs_mstar(
    frechet_result: dict,
    profiles_df:    pd.DataFrame,
    output_path:    Path,
) -> None:
    """
    Rank comparison: M* scalar ranking vs Fréchet distance ranking.
    Shows which sub-groups are promoted or demoted when covariance
    differences are accounted for.
    """
    fd_scores = frechet_result["frechet_scores"]
    fd_order  = np.argsort(fd_scores)[::-1]
    fd_ranks  = {ALL_SUBGROUPS[i]: r + 1
                 for r, i in enumerate(fd_order)}

    mstar_ranks = {row["subgroup"]: r + 1
                   for r, (_, row) in enumerate(profiles_df.iterrows())}

    # Rank change: positive = promoted by Fréchet, negative = demoted
    rank_changes = {sg: mstar_ranks[sg] - fd_ranks[sg]
                    for sg in ALL_SUBGROUPS}

    # Sort by absolute rank change descending
    sgs_sorted = sorted(ALL_SUBGROUPS,
                        key=lambda sg: abs(rank_changes[sg]), reverse=True)
    changes    = [rank_changes[sg] for sg in sgs_sorted]
    n_nonb     = len(NONB_SUBGROUPS)
    colors     = ["#2ECC71" if c > 0 else "#E74C3C" if c < 0 else "#95A5A6"
                  for c in changes]

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("white")

    bars = ax.bar(range(len(sgs_sorted)), changes,
                  color=colors, edgecolor="none", alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(sgs_sorted)))
    ax.set_xticklabels(display_subgroups(sgs_sorted), rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("M* rank − Fréchet rank(positive = promoted by Fréchet)",
                  fontsize=11)
    ax.set_title(
        "Rank Change: M* Scalar vs Fréchet Distance"
        "(sub-groups promoted when covariance differences add information)",
        fontsize=12, fontweight="bold"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate ranks
    for i, sg in enumerate(sgs_sorted):
        ax.text(i, changes[i] + (0.2 if changes[i] >= 0 else -0.4),
                f"M*:{mstar_ranks[sg]}→FD:{fd_ranks[sg]}",
                ha="center", fontsize=7, color="#333")

    promoted_patch = mpatches.Patch(color="#2ECC71", alpha=0.85,
                                    label="Promoted by Fréchet")
    demoted_patch  = mpatches.Patch(color="#E74C3C", alpha=0.85,
                                    label="Demoted by Fréchet")
    ax.legend(handles=[promoted_patch, demoted_patch], fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_mstar_heatmap(
    M_star:      np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 3.5))
    fig.patch.set_facecolor("white")

    im = ax.imshow(M_star, cmap="RdBu_r", aspect="auto",
                   vmin=-np.abs(M_star).max(),
                   vmax= np.abs(M_star).max())

    ax.set_xticks(range(len(ALL_SUBGROUPS)))
    ax.set_xticklabels(display_subgroups(ALL_SUBGROUPS), rotation=45, ha="right", fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["lncRNA", "mRNA"], fontsize=11)

    # Divider between NonB and TE
    n_nonb = len(NONB_SUBGROUPS)
    ax.axvline(x=n_nonb - 0.5, color="white", linewidth=2)

    plt.colorbar(im, ax=ax, label="Normalised class excess (M*)", shrink=0.8)
    ax.set_title("M* — Normalised Class-Conditional Sub-group Deviation\n"
                 "(red = enriched in this class, blue = depleted)",
                 fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_eigenvalue_spectrum(
    result:      dict,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor("white")

    # Left: eigenvalue magnitudes
    ax = axes[0]
    x  = np.arange(1, len(result["eigenvalues"]) + 1)
    ax.bar(x, result["eigenvalues"], color="#4A90D9", edgecolor="#185FA5",
           linewidth=0.8, alpha=0.9)
    ax.set_xlabel("Eigenvalue rank", fontsize=11)
    ax.set_ylabel("Eigenvalue magnitude", fontsize=11)
    ax.set_title("M*ᵀM* Eigenvalue Spectrum", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: cumulative explained variance
    ax2 = axes[1]
    ax2.plot(x, result["cumulative_var"] * 100, "o-",
             color="#D85A30", lw=2, markersize=6)
    ax2.axhline(80, color="#aaa", linestyle="--", lw=1)
    ax2.axhline(95, color="#aaa", linestyle="--", lw=1)
    ax2.set_xlabel("Number of eigenvectors", fontsize=11)
    ax2.set_ylabel("Cumulative variance explained (%)", fontsize=11)
    ax2.set_title("Cumulative Explained Variance", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for threshold in [80, 95]:
        ax2.text(len(x) - 0.5, threshold + 1.5, f"{threshold}%",
                 fontsize=9, color="#888", ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_eigenvector_loadings(
    result:      dict,
    n_top:       int,
    output_path: Path,
) -> None:
    n_top    = min(n_top, result["eigenvectors"].shape[1])
    n_nonb   = len(NONB_SUBGROUPS)
    fig, axes = plt.subplots(1, n_top, figsize=(4 * n_top, 5), sharey=True)
    if n_top == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for i, ax in enumerate(axes):
        loadings = result["eigenvectors"][:, i]
        colors = [_sg_color(sg, alpha=False) for sg in ALL_SUBGROUPS]
        edge   = [_sg_edge(sg) for sg in ALL_SUBGROUPS]

        ax.barh(ALL_SUBGROUPS, loadings, color=colors, edgecolor=edge,
                linewidth=0.8, alpha=0.9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"Eigenvector {i+1}\n"
                     f"({result['explained_var'][i]*100:.1f}% var)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Loading", fontsize=10)
        ax.grid(True, axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Mark top-loading sub-group via tick color
        top_idx   = int(np.abs(loadings).argmax())
        top_label = ALL_SUBGROUPS[top_idx]
        ax.set_yticklabels([
            f"► {display_subgroup(sg)}" if sg == top_label else display_subgroup(sg)
            for sg in ALL_SUBGROUPS
        ], fontsize=9)

    nonb_patch = mpatches.Patch(facecolor="#4A90D9AA", edgecolor="#185FA5",
                                label="NonB")
    te_patch   = mpatches.Patch(facecolor="#EF9F27AA", edgecolor="#BA7517",
                                label="TE")
    fig.legend(handles=[
        mpatches.Patch(facecolor=BLOCK_COLORS[b], edgecolor=BLOCK_EDGES[b],
                       label=display_block(b))
        for b in REGISTRY.block_names
    ], fontsize=10, loc="lower center", ncol=len(REGISTRY.block_names),
    bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Top Eigenvectors of M*ᵀM* — Optimal Discriminative Directions",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_class_profiles(
    X:           np.ndarray,   # (N, n_subgroups) z-scored
    labels:      np.ndarray,
    output_path: Path,
) -> None:
    n_nonb = len(NONB_SUBGROUPS)
    x      = np.arange(len(ALL_SUBGROUPS))
    width  = 0.35

    mu_lnc  = X[labels == 0].mean(axis=0)
    mu_mrna = X[labels == 1].mean(axis=0)
    se_lnc  = X[labels == 0].std(axis=0) / np.sqrt((labels == 0).sum())
    se_mrna = X[labels == 1].std(axis=0) / np.sqrt((labels == 1).sum())

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("white")

    ax.bar(x - width/2, mu_lnc,  width, label="lncRNA",
           color="#4A90D9AA", edgecolor="#185FA5", linewidth=0.8)
    ax.errorbar(x - width/2, mu_lnc,  yerr=se_lnc,
                fmt="none", color="#185FA5", capsize=3, linewidth=1)
    ax.bar(x + width/2, mu_mrna, width, label="mRNA",
           color="#EF9F27AA", edgecolor="#BA7517", linewidth=0.8)
    ax.errorbar(x + width/2, mu_mrna, yerr=se_mrna,
                fmt="none", color="#BA7517", capsize=3, linewidth=1)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(x=n_nonb - 0.5, color="#ccc", linestyle="--",
               linewidth=1.5, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(display_subgroups(ALL_SUBGROUPS), rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Mean z-scored sub-group value (± SE)", fontsize=11)
    ax.set_title("Class-Conditional Sub-group Profiles\n"
                 "(lncRNA vs mRNA, z-scored features, mean ± SE)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_discrimination_ranking(
    M_star:      np.ndarray,
    output_path: Path,
) -> None:
    """
    Bar chart of |M*[lnc] - M*[mrna]| per sub-group — direct univariate
    discrimination score, model-free.
    """
    diff   = np.abs(M_star[0] - M_star[1])
    order  = np.argsort(diff)[::-1]
    n_nonb = len(NONB_SUBGROUPS)

    ordered_names = [ALL_SUBGROUPS[i] for i in order]
    ordered_diff  = diff[order]
    colors = [_sg_color(ALL_SUBGROUPS[i]) for i in order]
    edges  = [_sg_edge(ALL_SUBGROUPS[i]) for i in order]

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("white")

    ax.bar(range(len(ordered_names)), ordered_diff,
           color=colors, edgecolor=edges, linewidth=0.8)
    ax.set_xticks(range(len(ordered_names)))
    ax.set_xticklabels(display_subgroups(ordered_names), rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("|M*[lncRNA] - M*[mRNA]|", fontsize=11)
    ax.set_title("Sub-group Discrimination Score — Model-Free Ranking\n"
                 "(absolute class-conditional excess difference, z-scored)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(handles=[
        mpatches.Patch(facecolor=BLOCK_COLORS[b], edgecolor=BLOCK_EDGES[b],
                       label=display_block(b))
        for b in REGISTRY.block_names
    ], fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="M* matrix analysis for lncRNA/mRNA sub-group discrimination"
    )
    parser.add_argument("--config",       required=True)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--n_top_eigen",  type=int, default=5,
                        help="Number of top eigenvectors to plot")
    parser.add_argument("--pattern_raw_csv", default=None,
                        help="Path to cross_fold_pattern_raw.csv "
                             "(from analyze_latent_probing.py) — if given, "
                             "also produces frechet_vs_pattern.png")
    args = parser.parse_args()

    config     = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("M* Sub-group Co-discrimination Analysis")
    print("=" * 70)
    print(f"Config    : {args.config}")
    print(f"Output    : {output_dir}")
    print("=" * 70)

    # ── Load and aggregate ────────────────────────────────────────────────────
    block_dfs, labels = load_features(config)
    subgroup_df             = aggregate_to_subgroups(block_dfs)

    # ── Z-score ───────────────────────────────────────────────────────────────
    print("\nZ-scoring sub-group matrix...")
    scaler = StandardScaler()
    X      = scaler.fit_transform(subgroup_df.values)   # (N, n_subgroup)
    print(f"  X shape: {X.shape}")

    # Save z-scored matrix for synthetic data generation (Stage 2)
    subgroup_scaled_df = pd.DataFrame(
        X, columns=ALL_SUBGROUPS, index=subgroup_df.index
    )
    subgroup_scaled_df["label"] = labels   # 0=lncRNA 1=mRNA
    subgroup_scaled_df.to_csv(output_dir / "subgroup_matrix_zscored.csv")
    print(f"  Saved z-scored sub-group matrix → "
          f"{output_dir / 'subgroup_matrix_zscored.csv'}")

    # ── Class profiles ────────────────────────────────────────────────────────
    print("\nComputing class-conditional profiles...")
    profile_rows = []
    for i, sg in enumerate(ALL_SUBGROUPS):
        lnc_vals  = X[labels == 0, i]
        mrna_vals = X[labels == 1, i]
        profile_rows.append({
            "subgroup":   sg,
            "block":      "nonb" if sg in NONB_SUBGROUPS else "te",
            "lnc_mean":   float(lnc_vals.mean()),
            "lnc_std":    float(lnc_vals.std()),
            "mrna_mean":  float(mrna_vals.mean()),
            "mrna_std":   float(mrna_vals.std()),
            "diff":       float(lnc_vals.mean() - mrna_vals.mean()),
            "abs_diff":   float(abs(lnc_vals.mean() - mrna_vals.mean())),
        })
    profiles_df = pd.DataFrame(profile_rows).sort_values(
        "abs_diff", ascending=False
    )
    profiles_df.to_csv(output_dir / "class_profiles.csv", index=False)
    print("\nSub-group discrimination ranking (model-free):")
    print(f"{'Rank':<5} {'Sub-group':<15} {'lncRNA mean':>12} "
          f"{'mRNA mean':>10} {'|diff|':>8}")
    print("-" * 55)
    for rank, (_, row) in enumerate(profiles_df.iterrows(), 1):
        direction = "lnc↑" if row["diff"] > 0 else "mrna↑"
        print(f"{rank:<5} {row['subgroup']:<15} {row['lnc_mean']:>+12.4f} "
              f"{row['mrna_mean']:>+10.4f} {row['abs_diff']:>8.4f}  "
              f"{direction}")

    # ── M* construction ───────────────────────────────────────────────────────
    print("\nConstructing M*...")
    M_star = construct_mstar(X, labels)
    mstar_df = pd.DataFrame(
        M_star,
        index=["lncRNA", "mRNA"],
        columns=ALL_SUBGROUPS
    )
    mstar_df.to_csv(output_dir / "mstar_matrix.csv")
    print("M* matrix:")
    print(mstar_df.round(4).to_string())

    # ── Eigendecomposition ────────────────────────────────────────────────────
    print("\nDecomposing M*ᵀM*...")
    result = decompose_mstar(M_star)

    eigen_df = pd.DataFrame({
        "rank":             range(1, len(result["eigenvalues"]) + 1),
        "eigenvalue":       result["eigenvalues"],
        "explained_var":    result["explained_var"],
        "cumulative_var":   result["cumulative_var"],
    })
    eigen_df.to_csv(output_dir / "eigenvalues.csv", index=False)

    print("\nEigenvalue spectrum:")
    print(f"{'Rank':<6} {'Eigenvalue':>12} {'Var %':>8} {'Cumul %':>9}")
    print("-" * 40)
    for _, row in eigen_df.iterrows():
        print(f"{int(row['rank']):<6} {row['eigenvalue']:>12.4f} "
              f"{row['explained_var']*100:>7.1f}% "
              f"{row['cumulative_var']*100:>8.1f}%")

    # Save eigenvectors
    evec_df = pd.DataFrame(
        result["eigenvectors"],
        index=ALL_SUBGROUPS,
        columns=[f"EV{i+1}" for i in range(len(ALL_SUBGROUPS))]
    )
    evec_df.to_csv(output_dir / "eigenvectors.csv")

    # ── LDA and within-class covariance ──────────────────────────────────────
    print("\nRunning LDA and within-class covariance analysis...")
    lda_result = lda_analysis(X, labels)

    # SNR ranking
    snr_rows = []
    for i, sg in enumerate(ALL_SUBGROUPS):
        snr_rows.append({
            "subgroup":   sg,
            "block":      "nonb" if sg in NONB_SUBGROUPS else "te",
            "snr":        float(lda_result["snr"][i]),
            "within_std": float(lda_result["within_std"][i]),
            "delta":      float(lda_result["delta"][i]),
        })
    snr_df = pd.DataFrame(snr_rows).sort_values("snr", ascending=False)
    snr_df.to_csv(output_dir / "snr_ranking.csv", index=False)

    print("\nFisher SNR ranking (accounts for within-class variance):")
    print(f"{'Rank':<5} {'Sub-group':<15} {'SNR':>8} {'within_std':>12} "
          f"{'raw |diff|':>11}")
    print("-" * 55)
    for rank, (_, row) in enumerate(snr_df.iterrows(), 1):
        raw_diff = abs(profiles_df.set_index("subgroup")
                       .loc[row["subgroup"], "diff"])
        print(f"{rank:<5} {row['subgroup']:<15} {row['snr']:>8.4f} "
              f"{row['within_std']:>12.4f}  {raw_diff:>10.4f}")

    print(f"\n  Fisher ratio (LDA): {lda_result['fisher_ratio']:.4f}")
    print(f"  LDA direction top sub-group: "
          f"{ALL_SUBGROUPS[np.abs(lda_result['lda_direction']).argmax()]}")

    # Save within-class covariances
    pd.DataFrame(lda_result["Sigma_w"],
                 index=ALL_SUBGROUPS,
                 columns=ALL_SUBGROUPS).to_csv(
        output_dir / "within_class_covariance_pooled.csv"
    )
    pd.DataFrame(lda_result["Sigma_lnc"],
                 index=ALL_SUBGROUPS,
                 columns=ALL_SUBGROUPS).to_csv(
        output_dir / "within_class_covariance_lncrna.csv"
    )
    pd.DataFrame(lda_result["Sigma_mrna"],
                 index=ALL_SUBGROUPS,
                 columns=ALL_SUBGROUPS).to_csv(
        output_dir / "within_class_covariance_mrna.csv"
    )

    # ── Fréchet distance analysis ─────────────────────────────────────────────
    print("\nRunning Fréchet distance analysis...")
    frechet_result = frechet_analysis(block_dfs, labels)

    fd_rows = []
    for i, sg in enumerate(ALL_SUBGROUPS):
        fd_rows.append({
            "subgroup":       sg,
            "block":          "nonb" if sg in NONB_SUBGROUPS else "te",
            "frechet_dist":   float(frechet_result["frechet_scores"][i]),
            "mean_term":      float(frechet_result["mean_terms"][i]),
            "cov_term":       float(frechet_result["cov_terms"][i]),
            "cov_fraction":   float(frechet_result["cov_terms"][i] /
                              max(frechet_result["frechet_scores"][i], 1e-8)),
        })
    fd_df = pd.DataFrame(fd_rows).sort_values("frechet_dist", ascending=False)
    fd_df.to_csv(output_dir / "frechet_ranking.csv", index=False)

    print("\nFréchet distance ranking (mean shift + covariance):")
    print(f"  {'Rank':<5} {'Sub-group':<15} {'FD':>8}  "
          f"{'Mean term':>10}  {'Cov term':>9}  {'Cov %':>6}")
    print("  " + "-" * 60)
    for rank, (_, row) in enumerate(fd_df.iterrows(), 1):
        print(f"  {rank:<5} {row['subgroup']:<15} {row['frechet_dist']:>8.4f}  "
              f"{row['mean_term']:>10.4f}  {row['cov_term']:>9.4f}  "
              f"{row['cov_fraction']*100:>5.1f}%")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_mstar_heatmap(M_star, output_dir / "mstar_heatmap.png")
    plot_eigenvalue_spectrum(result, output_dir / "eigenvalue_spectrum.png")
    plot_eigenvector_loadings(result, args.n_top_eigen,
                              output_dir / "eigenvector_loadings.png")
    plot_class_profiles(X, labels,
                        output_dir / "class_profile_comparison.png")
    plot_discrimination_ranking(M_star,
                                output_dir / "discrimination_ranking.png")
    plot_within_class_covariance(lda_result,
                                 output_dir / "within_class_covariance.png")
    plot_snr_ranking(lda_result,
                     output_dir / "snr_ranking.png")
    plot_lda_vs_mstar(lda_result, profiles_df,
                      output_dir / "lda_vs_mstar_comparison.png")
    plot_within_std(lda_result,
                    output_dir / "within_class_std.png")
    plot_frechet_ranking(frechet_result, profiles_df,
                         output_dir / "frechet_ranking.png")
    plot_frechet_vs_mstar(frechet_result, profiles_df,
                          output_dir / "frechet_vs_mstar.png")

    if args.pattern_raw_csv is not None:
        pattern_raw_path = Path(args.pattern_raw_csv)
        if pattern_raw_path.exists():
            pattern_raw_df = pd.read_csv(pattern_raw_path)
            plot_frechet_vs_pattern(
                fd_df, pattern_raw_df,
                output_dir / "frechet_vs_pattern.png",
            )
        else:
            print(f"\n--pattern_raw_csv given but not found at "
                  f"{pattern_raw_path} — skipping frechet_vs_pattern plot")

    # ── Summary ───────────────────────────────────────────────────────────────
    top_snr_sg  = snr_df.iloc[0]["subgroup"]
    top_mstar_sg = profiles_df.iloc[0]["subgroup"]
    ranking_agrees = top_snr_sg == top_mstar_sg

    summary_lines = [
        "M* Sub-group Co-discrimination Analysis — Key Findings",
        "=" * 60,
        "",
        "NOTE: M*ᵀM* is rank-1 by construction for binary classification.",
        "The eigenvalue result (100% in EV1) is mathematically guaranteed,",
        "not a data finding. The discrimination ranking below is the result.",
        "",
        "1. Model-free discrimination ranking (|M*[lnc] - M*[mrna]|):",
    ]
    for rank, (_, row) in enumerate(profiles_df.iterrows(), 1):
        direction = "lncRNA > mRNA" if row["diff"] > 0 else "mRNA > lncRNA"
        summary_lines.append(
            f"   {rank:2d}. {row['subgroup']:<15} |diff|={row['abs_diff']:.4f}  "
            f"({direction})"
        )

    summary_lines += [
        "",
        "2. Fisher SNR ranking (accounts for within-class variance):",
    ]
    for rank, (_, row) in enumerate(snr_df.iterrows(), 1):
        raw = abs(profiles_df.set_index("subgroup")
                  .loc[row["subgroup"], "diff"])
        summary_lines.append(
            f"   {rank:2d}. {row['subgroup']:<15} SNR={row['snr']:.4f}  "
            f"within_std={row['within_std']:.4f}  |diff|={raw:.4f}"
        )

    fd_top = fd_df.iloc[0]["subgroup"]
    fd_top_cov_pct = fd_df.iloc[0]["cov_fraction"] * 100
    summary_lines += [
        "",
        f"3. Fréchet distance top sub-group: {fd_top}",
        f"   Covariance contribution: {fd_top_cov_pct:.1f}% of total FD",
        "   Fréchet ranking (FD = mean shift + covariance term):",
    ]
    for rank, (_, row) in enumerate(fd_df.iterrows(), 1):
        summary_lines.append(
            f"   {rank:2d}. {row['subgroup']:<15} FD={row['frechet_dist']:.4f}  "
            f"(mean={row['mean_term']:.4f}, cov={row['cov_term']:.4f}, "
            f"cov%={row['cov_fraction']*100:.1f}%)"
        )
    summary_lines += [
        "",
        f"4. M* top vs SNR top: {'AGREE' if ranking_agrees else 'DISAGREE'}",
        f"   M* rank 1: {top_mstar_sg}",
        f"   SNR rank 1: {top_snr_sg}",
        "",
        "5. Synthetic benchmark — effect sizes (within-class std):",
    ]
    for _, row in snr_df.head(5).iterrows():
        sep = row["snr"] * 2
        summary_lines.append(
            f"   {row['subgroup']:<15} within_std={row['within_std']:.4f}  "
            f"→ need Δ ≥ {sep:.3f} z-score units for 2σ separation"
        )

    summary_lines += [
        "",
        "6. Comparison with model ablation ranking (feature_zero):",
        "   Sub-group     | M* rank | SNR rank | Ablation rank",
        "   " + "-" * 50,
    ]
    ablation_ranks = {
        "IR": 1, "TE_CORE": 2, "GQ": 3, "GLOBAL": 4, "STR": 5,
        "DR": 6, "APR": 7, "Z": 8, "TRI": 9,
        "TE_LCTR": 11, "TE_UNKNOWN": 12, "TE_GLOBAL": 13,
        "TE_QUALITY": 10, "MR": 14, "TE_PSEUDO": 15,
    }
    mstar_ranks = {row["subgroup"]: r+1
                   for r, (_, row) in enumerate(profiles_df.iterrows())}
    snr_ranks   = {row["subgroup"]: r+1
                   for r, (_, row) in enumerate(snr_df.iterrows())}
    for sg in ALL_SUBGROUPS:
        summary_lines.append(
            f"   {sg:<15}  M*={mstar_ranks.get(sg,'?'):>2}  "
            f"SNR={snr_ranks.get(sg,'?'):>2}  "
            f"Ablation={ablation_ranks.get(sg,'?'):>2}"
        )

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    with open(output_dir / "summary.txt", "w") as f:
        f.write(summary_text)

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()