#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_feature_concentration.py

Per-feature decomposition of sub-group discrimination signal.

Motivation
----------
Sub-group-level metrics (M*, Fréchet distance, SNR, ablation drop) collapse
all features within a sub-group into a single scalar. This conflates two
very different situations:

  concentrated : a few features within the sub-group carry strong signal,
                  the rest are near-noise — the aggregate score reflects a
                  real, specific biological mechanism
  diffuse       : signal is spread thinly and roughly evenly across many
                  features — the aggregate score may be inflated mainly by
                  dimensionality (more features = more chances for small
                  real or spurious differences to accumulate in a distance
                  metric), without any single feature being individually
                  compelling

This matters most for large sub-groups (SS_BINNED: 40 features, TE_CORE:
69 features combined genomic+processed) where aggregate metrics have much
more "surface area" than small sub-groups (RG4: 4 features, SS_RELPOS: 3
features) even if true per-feature effect sizes are comparable.

What this script computes, per sub-group
-------------------------------------------
1. Per-feature univariate discrimination:
     - Cohen's d (lncRNA vs mRNA, z-scored)
     - Mann-Whitney U p-value (rank-based, robust to non-normality)
     - Individual contribution to the sub-group's aggregate Fréchet
       distance (mean-term decomposition is additive across features
       in z-scored space; covariance-term contribution approximated
       via leave-one-out Fréchet recomputation)

2. Concentration metrics (per sub-group):
     - Gini coefficient of |Cohen's d| across the sub-group's features
       (0 = perfectly diffuse/uniform, 1 = perfectly concentrated in one
       feature)
     - Participation ratio: (Σd_i²)² / Σd_i⁴ — effective number of features
       carrying signal (intuition: if N features have equal |d|, participation
       ratio ≈ N; if one dominates, ratio ≈ 1)
     - Top-3 feature share: fraction of total Σ|d_i| held by the top 3
       features

3. Leave-one-out Fréchet sensitivity:
     - Recompute the sub-group's Fréchet distance with each feature
       individually removed; report the % drop. A feature whose removal
       causes a large % drop is doing disproportionate work.

Outputs
-------
  feature_discrimination.csv   — per-feature d, p-value, rank within subgroup
                                  (p-value retained for completeness, but see
                                  NOTE below — not used for the verdict)
  concentration_summary.csv    — per-subgroup Gini, participation ratio,
                                  elbow metrics (n/% features needed for
                                  80%/90% of cumulative |d|), top feature
  loo_frechet_sensitivity.csv  — per-feature % drop in subgroup FD when removed
  figures/
    concentration_summary.png  — 3-panel: elbow %, Gini, participation ratio
    top_features_<subgroup>.png — per-subgroup horizontal bar of |d| ranked,
                                  coloured by effect-size tier (NOT p-value —
                                  see NOTE), generated for top N subgroups

NOTE on significance vs effect size
-------------------------------------
At the sample sizes used here (N ~ 10^5-10^6 transcripts), Bonferroni-
corrected p-values are saturated: almost any nonzero population difference
reaches significance regardless of practical magnitude. The p_value /
p_bonferroni columns are retained in feature_discrimination.csv for
completeness but should NOT be used to judge "real vs noise" — use |Cohen's
d| and the elbow/concentration metrics instead throughout this script's
console output and verdict.

Usage
-----
python analysis/pattern/analyze_feature_concentration.py \\
    --config       configs/beta_vae_subgroup_base_g49.json \\
    --mstar_dir    gencode_v49_experiments/mstar_analysis \\
    --output_dir   gencode_v49_experiments/feature_concentration \\
    --top_n_subgroups 6
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.linalg import sqrtm as scipy_sqrtm
from sklearn.preprocessing import StandardScaler

from configs.load_config import load_config
from data.cv_utils import load_sequences_in_order
from data.feature_registry import (REGISTRY, display_block,
                                    display_subgroup, display_subgroups)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

BLOCK_COLORS = {"nonb": "#4A90D9", "te": "#F4B942", "nonb2": "#9B7FD4"}


# ---------------------------------------------------------------------------
# Feature loading (mirrors analyze_mstar.py's load_features, deduped)
# ---------------------------------------------------------------------------

def load_block_features(config) -> tuple[dict, np.ndarray]:
    print("Loading sequences and labels...")
    all_sequences, labels_str = load_sequences_in_order(
        lnc_fasta=config.get("data", "lnc_fasta"),
        pc_fasta= config.get("data", "pc_fasta"),
    )
    labels = np.array([0 if l == "lnc" else 1 for l in labels_str])
    print(f"  lncRNA: {(labels==0).sum():,}  mRNA: {(labels==1).sum():,}")

    ids = [str(seq.id).split("|")[0].split(".")[0] for seq in all_sequences]

    def _load_and_dedupe(path: str, label: str) -> pd.DataFrame:
        print(f"Loading {label} features...")
        df = pd.read_csv(path, index_col=0)
        print(f"  Raw shape: {df.shape}")
        df.index = (df.index.astype(str)
                    .str.split("|").str[0]
                    .str.split(".").str[0])
        n_before = len(df)
        n_unique = df.index.nunique()
        if n_unique < n_before:
            print(f"  WARNING: {n_before - n_unique:,} duplicate transcript_id "
                  f"rows — keeping first occurrence")
            df = df[~df.index.duplicated(keep="first")]
        print(f"  Deduped shape: {df.shape}")
        return df

    nonb_path = config.get("data", "nonb_processed_csv", default=None) \
                or config.get("data", "nonb_genomic_csv")
    te_path   = config.get("data", "te_processed_csv", default=None) \
                or config.get("data", "te_genomic_csv")
    nonb_df = _load_and_dedupe(nonb_path, "NonB")
    te_df   = _load_and_dedupe(te_path,   "TE")

    nonb2_path = config.get("data", "nonb2_csv", default=None)
    nonb2_df = _load_and_dedupe(nonb2_path, "NonB2") if nonb2_path else None

    nonb_df = nonb_df.reindex(ids)
    te_df   = te_df.reindex(ids)
    if nonb2_df is not None:
        nonb2_df = nonb2_df.reindex(ids)

    valid = (~nonb_df.isnull().all(axis=1)) & (~te_df.isnull().all(axis=1))
    if nonb2_df is not None:
        valid = valid & (~nonb2_df.isnull().all(axis=1))
    print(f"  Valid transcripts after alignment: {valid.sum():,} / {len(valid):,}")

    valid_arr = valid.values
    nonb_df = nonb_df[valid_arr].fillna(0.0)
    te_df   = te_df[valid_arr].fillna(0.0)
    labels  = labels[valid_arr]

    block_dfs = {"nonb": nonb_df, "te": te_df}
    if nonb2_df is not None:
        block_dfs["nonb2"] = nonb2_df[valid_arr].fillna(0.0)

    return block_dfs, labels


# ---------------------------------------------------------------------------
# Per-feature Cohen's d and Mann-Whitney
# ---------------------------------------------------------------------------

def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * a.std(ddof=1)**2 +
                           (nb - 1) * b.std(ddof=1)**2) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0


def per_feature_discrimination(
    block_dfs: dict, labels: np.ndarray, registry=REGISTRY
) -> pd.DataFrame:
    """
    Per-feature Cohen's d and Mann-Whitney U test, lncRNA vs mRNA,
    for every feature in every registry block. Z-scores within each
    feature's own column before computing d (so d is comparable across
    features of different raw scales).
    """
    rows = []
    for block_name in registry.block_names:
        df = block_dfs.get(block_name)
        if df is None:
            continue
        for sg in registry.block_subgroups(block_name):
            idx = registry.indices_for_subgroup(sg)
            for rank_in_sg, feat_idx in enumerate(idx):
                col = df.columns[feat_idx]
                vals = StandardScaler().fit_transform(
                    df.iloc[:, feat_idx].values.reshape(-1, 1)
                ).ravel()
                v_lnc  = vals[labels == 0]
                v_mrna = vals[labels == 1]
                d = cohen_d(v_lnc, v_mrna)
                try:
                    _, p = mannwhitneyu(v_lnc, v_mrna, alternative="two-sided")
                except ValueError:
                    p = np.nan

                rows.append(dict(
                    feature      = col,
                    subgroup     = sg,
                    block        = block_name,
                    cohen_d      = round(float(d), 4),
                    abs_d        = round(abs(float(d)), 4),
                    p_value      = float(p) if not np.isnan(p) else None,
                ))

    df_out = pd.DataFrame(rows)
    # Bonferroni across all features
    n_tests = len(df_out)
    df_out["p_bonferroni"] = (df_out["p_value"] * n_tests).clip(upper=1.0)
    # Rank within subgroup
    df_out["rank_in_subgroup"] = (
        df_out.groupby("subgroup")["abs_d"]
        .rank(ascending=False, method="first").astype(int)
    )
    return df_out.sort_values(["subgroup", "rank_in_subgroup"])


# ---------------------------------------------------------------------------
# Concentration metrics
# ---------------------------------------------------------------------------

def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative array. 0=uniform, 1=concentrated."""
    v = np.sort(np.abs(values))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    cum = np.cumsum(v)
    return float((2 * np.sum((np.arange(1, n + 1)) * v) - (n + 1) * cum[-1])
                 / (n * cum[-1]))


def participation_ratio(values: np.ndarray) -> float:
    """
    (Σd_i²)² / Σd_i⁴ — effective number of features carrying signal.
    Equal to N if all |d_i| equal; approaches 1 if one feature dominates.
    """
    v2 = np.abs(values) ** 2
    num = v2.sum() ** 2
    den = (v2 ** 2).sum()
    return float(num / den) if den > 0 else 0.0


def elbow_n_features(values: np.ndarray, threshold: float = 0.80) -> int:
    """
    Number of top-ranked features (by |d|, descending) needed to reach
    `threshold` fraction of the subgroup's total Σ|d|.

    This is the most directly interpretable concentration metric: "this
    subgroup's signal lives in its top N features" — exactly what's visible
    by eye on a sorted bar chart, and unaffected by the large-N significance
    saturation that makes p-values uninformative at this sample size.
    """
    v = np.sort(np.abs(values))[::-1]
    total = v.sum()
    if total <= 0:
        return 0
    cum = np.cumsum(v) / total
    return int(np.searchsorted(cum, threshold) + 1)


def effect_size_tier(d: float) -> str:
    """Conventional Cohen's d magnitude bands — used instead of p-value
    significance for per-feature plot coloring, since at N~10^5-10^6 the
    Bonferroni-corrected p-value is saturated (near-zero for almost any
    nonzero effect) and carries no discriminating information."""
    ad = abs(d)
    if ad < 0.10:
        return "negligible"
    elif ad < 0.30:
        return "small"
    elif ad < 0.50:
        return "medium"
    else:
        return "large"


def concentration_summary(feat_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sg, grp in feat_df.groupby("subgroup"):
        d_vals = grp["abs_d"].values
        n      = len(d_vals)
        sorted_d = np.sort(d_vals)[::-1]
        top3_share = (sorted_d[:min(3, n)].sum() / sorted_d.sum()
                     if sorted_d.sum() > 0 else 0.0)

        n_for_80 = elbow_n_features(d_vals, threshold=0.80)
        n_for_90 = elbow_n_features(d_vals, threshold=0.90)

        top5 = grp.sort_values("abs_d", ascending=False).head(5)
        rest = grp.sort_values("abs_d", ascending=False).iloc[5:]

        rows.append(dict(
            subgroup             = sg,
            block                 = grp["block"].iloc[0],
            n_features            = n,
            n_features_for_80pct  = n_for_80,
            n_features_for_90pct  = n_for_90,
            pct_features_for_80   = round(100 * n_for_80 / n, 1),
            gini                  = round(gini_coefficient(d_vals), 4),
            participation_ratio   = round(participation_ratio(d_vals), 2),
            participation_pct     = round(100 * participation_ratio(d_vals) / n, 1),
            top3_share            = round(float(top3_share), 4),
            mean_abs_d            = round(float(d_vals.mean()), 4),
            mean_abs_d_top5       = round(float(top5["abs_d"].mean()), 4),
            mean_abs_d_rest       = (round(float(rest["abs_d"].mean()), 4)
                                     if len(rest) > 0 else None),
            max_abs_d             = round(float(d_vals.max()), 4),
            top_feature           = grp.sort_values("abs_d", ascending=False)
                                       ["feature"].iloc[0],
            top_feature_d         = round(float(grp.sort_values(
                                       "abs_d", ascending=False)
                                       ["abs_d"].iloc[0]), 4),
        ))
    return pd.DataFrame(rows).sort_values("pct_features_for_80")


# ---------------------------------------------------------------------------
# Leave-one-out Fréchet sensitivity
# ---------------------------------------------------------------------------

def frechet_distance_raw(X_lnc: np.ndarray, X_mrna: np.ndarray, reg=1e-4) -> float:
    """Fréchet distance between two Gaussians fit to raw (already z-scored) data."""
    if X_lnc.shape[1] == 0:
        return 0.0
    mu1, mu2 = X_lnc.mean(0), X_mrna.mean(0)
    S1 = np.cov(X_lnc.T) + reg * np.eye(X_lnc.shape[1])
    S2 = np.cov(X_mrna.T) + reg * np.eye(X_mrna.shape[1])
    if S1.ndim == 0:
        S1, S2 = np.array([[float(S1)]]), np.array([[float(S2)]])
    mean_term = float(np.sum((mu1 - mu2) ** 2))
    try:
        S1_sqrt = scipy_sqrtm(S1).real
        inner   = scipy_sqrtm(S1_sqrt @ S2 @ S1_sqrt).real
        cov_term = max(float(np.trace(S1 + S2 - 2 * inner)), 0.0)
    except Exception:
        cov_term = 0.0
    return mean_term + cov_term


def loo_frechet_sensitivity(
    block_dfs: dict, labels: np.ndarray, top_subgroups: List[str],
    registry=REGISTRY,
) -> pd.DataFrame:
    """
    For each subgroup in top_subgroups, compute full-subgroup Fréchet
    distance, then recompute with each feature individually removed.
    Report % drop — features causing large drops are doing disproportionate
    work; features with near-zero drop are along for the ride.

    Restricted to top_subgroups (by full FD) since this is O(k) Fréchet
    computations per subgroup with k features — expensive for k=40.
    """
    rows = []
    for sg in top_subgroups:
        block_name = registry.token_block(sg)
        df = block_dfs.get(block_name)
        if df is None:
            continue
        idx = registry.indices_for_subgroup(sg)
        if len(idx) < 2:
            continue   # LOO undefined for single-feature subgroups

        X = StandardScaler().fit_transform(df.iloc[:, idx].values)
        X_lnc, X_mrna = X[labels == 0], X[labels == 1]

        full_fd = frechet_distance_raw(X_lnc, X_mrna)
        if full_fd <= 1e-8:
            continue

        print(f"  LOO sensitivity for {sg} ({len(idx)} features, "
              f"full FD={full_fd:.4f})...")

        for i, feat_idx in enumerate(idx):
            keep = [j for j in range(len(idx)) if j != i]
            loo_fd = frechet_distance_raw(X_lnc[:, keep], X_mrna[:, keep])
            pct_drop = 100 * (full_fd - loo_fd) / full_fd

            rows.append(dict(
                subgroup    = sg,
                feature     = df.columns[feat_idx],
                full_fd     = round(full_fd, 4),
                loo_fd      = round(loo_fd, 4),
                pct_drop    = round(pct_drop, 2),
            ))

    return pd.DataFrame(rows).sort_values(["subgroup", "pct_drop"], ascending=[True, False])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_concentration_summary(conc_df: pd.DataFrame, output_path: Path) -> None:
    """
    Three-panel: elbow %, Gini coefficient, and participation % per subgroup,
    ordered by pct_features_for_80 ascending (most concentrated first).
    The elbow panel (n features needed for 80% of cumulative |d|) is the
    most directly interpretable of the three — it answers "how many features
    actually carry this subgroup's signal" without needing a definitional
    detour through Gini or participation ratio.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor("white")

    order = conc_df.sort_values("pct_features_for_80")
    colors = [BLOCK_COLORS.get(b, "#999") for b in order["block"]]

    ax0 = axes[0]
    ax0.barh(display_subgroups(order["subgroup"]), order["pct_features_for_80"], color=colors,
             edgecolor="black", linewidth=0.6, alpha=0.9)
    ax0.set_xlabel("% of features needed for 80% of Σ|d|", fontsize=11)
    ax0.set_title("Elbow Concentration\n"
                  "(low % = few features carry 80% of the subgroup's signal)",
                  fontsize=11, fontweight="bold")
    ax0.grid(True, axis="x", alpha=0.3)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    ax = axes[1]
    ax.barh(display_subgroups(order["subgroup"]), order["gini"], color=colors, edgecolor="black",
            linewidth=0.6, alpha=0.9)
    ax.set_xlabel("Gini coefficient (|Cohen's d| across features)", fontsize=11)
    ax.set_title("Signal Concentration — Gini\n"
                 "(0 = diffuse across all features, 1 = one feature dominates)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2 = axes[2]
    ax2.barh(display_subgroups(order["subgroup"]), order["participation_pct"], color=colors,
             edgecolor="black", linewidth=0.6, alpha=0.9)
    ax2.set_xlabel("Participation ratio (% of n_features)", fontsize=11)
    ax2.set_title("Effective Number of Discriminative Features\n"
                  "(low % = few features carry the subgroup's signal)",
                  fontsize=11, fontweight="bold")
    ax2.grid(True, axis="x", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    for ax_i in (ax0, ax, ax2):
        for i, (_, row) in enumerate(order.iterrows()):
            ax_i.text(0.5, i, f"n={row['n_features']}", fontsize=7,
                     color="white", va="center", ha="left", fontweight="bold")

    legend_handles = [
        mpatches.Patch(facecolor=c_col, label=display_block(b))
        for b, c_col in BLOCK_COLORS.items()
        if b in conc_df["block"].values
    ]
    fig.legend(handles=legend_handles, fontsize=10, loc="lower center",
               ncol=len(legend_handles), bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_top_features(
    feat_df: pd.DataFrame, subgroup: str, output_path: Path
) -> None:
    """
    Horizontal bar of |Cohen's d| per feature within one subgroup, ranked.

    Coloured by effect-size tier (negligible/small/medium/large), not by
    p-value significance — at the sample sizes used here (N ~ 10^5-10^6)
    the Bonferroni-corrected p-value is saturated and near-zero for almost
    any nonzero effect, so it carries no discriminating information. Effect
    size is the metric that actually distinguishes real signal from noise.
    """
    sub = feat_df[feat_df["subgroup"] == subgroup].sort_values(
        "abs_d", ascending=True
    )
    if len(sub) == 0:
        return

    tier_colors = {
        "negligible": "#BBBBBB",
        "small":      "#F4B942",
        "medium":     "#E8743B",
        "large":      "#C0392B",
    }
    bar_colors = [tier_colors[effect_size_tier(d)] for d in sub["cohen_d"]]

    fig, ax = plt.subplots(figsize=(9, max(3, 0.3 * len(sub))))
    fig.patch.set_facecolor("white")

    ax.barh(sub["feature"], sub["abs_d"], color=bar_colors,
            edgecolor="black", linewidth=0.5, alpha=0.9)
    ax.set_xlabel("|Cohen's d|  (lncRNA vs mRNA, z-scored)", fontsize=10)
    ax.set_title(f"{display_subgroup(subgroup)} — Per-feature discrimination\n"
                 "(coloured by effect size — negligible <0.1, small 0.1-0.3, "
                 "medium 0.3-0.5, large >0.5)",
                 fontsize=10.5, fontweight="bold")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    present_tiers = [t for t in ["large", "medium", "small", "negligible"]
                     if t in {effect_size_tier(d) for d in sub["cohen_d"]}]
    ax.legend(handles=[Patch(facecolor=tier_colors[t], label=t)
                       for t in present_tiers],
              fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Per-feature concentration analysis within sub-groups"
    )
    parser.add_argument("--config",       required=True)
    parser.add_argument("--mstar_dir",    default=None,
                        help="Directory with frechet_ranking.csv from "
                             "analyze_mstar.py, used to pick top subgroups "
                             "for LOO sensitivity (optional — if absent, "
                             "top subgroups picked from this script's own "
                             "Fréchet computation)")
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--top_n_subgroups", type=int, default=6,
                        help="Number of top subgroups (by Fréchet distance) "
                             "to run LOO sensitivity + plot on (default 6)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)

    print("=" * 70)
    print("Per-feature concentration analysis")
    print("=" * 70)

    block_dfs, labels = load_block_features(config)

    print("\nComputing per-feature discrimination (Cohen's d, Mann-Whitney)...")
    feat_df = per_feature_discrimination(block_dfs, labels)
    feat_df.to_csv(output_dir / "feature_discrimination.csv", index=False)
    print(f"  {len(feat_df)} features across {feat_df['subgroup'].nunique()} subgroups")
    print(f"  NOTE: at N~10^5-10^6 transcripts, Bonferroni-corrected p-values "
          f"are saturated (near-zero for almost any nonzero effect) and are "
          f"NOT used below to judge real vs noise signal — effect size "
          f"(|Cohen's d|) is used instead throughout.")

    print("\nComputing concentration metrics per subgroup...")
    conc_df = concentration_summary(feat_df)
    conc_df.to_csv(output_dir / "concentration_summary.csv", index=False)

    print(f"\n  {'Subgroup':<14} {'n_feat':>6} {'n@80%':>6} {'%@80%':>6} "
          f"{'top_d':>6} {'d_top5':>7} {'d_rest':>7} {'top_feature':<28}")
    print("  " + "-" * 98)
    for _, row in conc_df.iterrows():
        d_rest = f"{row['mean_abs_d_rest']:.3f}" if row['mean_abs_d_rest'] is not None else "  n/a"
        print(f"  {row['subgroup']:<14} {row['n_features']:>6} "
              f"{row['n_features_for_80pct']:>6} {row['pct_features_for_80']:>5.1f}% "
              f"{row['top_feature_d']:>6.3f} {row['mean_abs_d_top5']:>7.3f} "
              f"{d_rest:>7} {row['top_feature']:<28}")

    # Pick top subgroups by full Fréchet distance for LOO sensitivity
    if args.mstar_dir and Path(args.mstar_dir, "frechet_ranking.csv").exists():
        fd_df = pd.read_csv(Path(args.mstar_dir) / "frechet_ranking.csv")
        top_subgroups = (fd_df.sort_values("frechet_dist", ascending=False)
                         ["subgroup"].head(args.top_n_subgroups).tolist())
        print(f"\nTop {args.top_n_subgroups} subgroups by Fréchet distance "
              f"(from {args.mstar_dir}): {top_subgroups}")
    else:
        # Fall back: rank subgroups by mean_abs_d as a proxy
        top_subgroups = (conc_df.sort_values("mean_abs_d", ascending=False)
                         ["subgroup"].head(args.top_n_subgroups).tolist())
        print(f"\nNo mstar_dir provided — using mean |d| proxy ranking: "
              f"{top_subgroups}")

    print("\nRunning leave-one-out Fréchet sensitivity...")
    loo_df = loo_frechet_sensitivity(block_dfs, labels, top_subgroups)
    loo_df.to_csv(output_dir / "loo_frechet_sensitivity.csv", index=False)

    for sg in top_subgroups:
        sub = loo_df[loo_df["subgroup"] == sg]
        if len(sub) == 0:
            continue
        print(f"\n  {sg} — top 5 most sensitive features (LOO % FD drop):")
        for _, row in sub.head(5).iterrows():
            print(f"    {row['feature']:<35} drop={row['pct_drop']:>6.2f}%")

    print("\nGenerating plots...")
    plot_concentration_summary(conc_df, output_dir / "figures" / "concentration_summary.png")
    for sg in top_subgroups:
        plot_top_features(
            feat_df, sg,
            output_dir / "figures" / f"top_features_{sg}.png"
        )

    print("\n" + "=" * 70)
    print("Verdict")
    print("=" * 70)
    print(f"\n  (Ranked by % of features needed to reach 80% of cumulative |d| — "
          f"lower = more concentrated)")
    concentrated = conc_df[conc_df["pct_features_for_80"] < 30]
    diffuse      = conc_df[conc_df["pct_features_for_80"] >= 70]
    no_signal    = conc_df[conc_df["max_abs_d"] < 0.10]

    print(f"\n  Concentrated subgroups (<30% of features carry 80% of signal):")
    for _, row in concentrated.iterrows():
        print(f"    {row['subgroup']:<14} {row['pct_features_for_80']:.1f}% "
              f"({row['n_features_for_80pct']}/{row['n_features']} features)  "
              f"top feature: {row['top_feature']} (d={row['top_feature_d']:.3f})")

    print(f"\n  Diffuse subgroups (≥70% of features needed for 80% of signal):")
    for _, row in diffuse.iterrows():
        print(f"    {row['subgroup']:<14} {row['pct_features_for_80']:.1f}%  "
              f"mean|d| top5={row['mean_abs_d_top5']:.3f}  "
              f"mean|d| rest={row['mean_abs_d_rest']}")

    if len(no_signal) > 0:
        print(f"\n    Near-zero signal subgroups (max |d| < 0.10 — essentially "
              f"no individual feature is discriminative, regardless of "
              f"aggregate metrics or p-values):")
        for _, row in no_signal.iterrows():
            print(f"    {row['subgroup']:<14} max|d|={row['max_abs_d']:.4f}")

    print(f"\n  Diffuse subgroups with large n_features are most at risk of "
          f"aggregate metrics (Fréchet, M*) reflecting broadly-distributed "
          f"real signal (verify via per-bin/per-measure inspection) OR "
          f"dimensionality accumulation — check whether the diffuse signal "
          f"is structured (e.g. consistent across a biological sub-measure) "
          f"or unstructured before treating the aggregate score as a clean "
          f"single finding.")

    print(f"\n  Outputs: {output_dir}/")
    print("  Done.")


if __name__ == "__main__":
    main()