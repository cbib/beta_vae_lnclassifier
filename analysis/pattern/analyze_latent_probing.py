#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_latent_probing.py

Three-stage interpretability analysis for BetaVAESubgroup.

Reads per-fold .npz files produced by extract_representations.py and runs:

Stage 1 — Latent probing of z
  Fits linear probes from the VAE latent z to known confounds (transcript
  length, GC content) and to class labels. Reports R² per confound to
  characterise what the sequence encoder has learned.

Stage 2 — Cross-modal correlation
  For each sub-group token, measures how much of its representation variance
  is already explained by z (sequence-feature redundancy). High redundancy
  means the feature token adds nothing beyond what the sequence encoder
  already captured.

Stage 3 — Pattern analysis
  Computes activation patterns (Haufe et al. 2014) from linear probes on
  token representations, both raw and after projecting out z-explained
  variance. Measures pattern-weight alignment per sub-group.

  raw alignment     : class signal in the full token representation
  residual alignment: class signal in the z-orthogonal token representation
                      — the genuinely independent feature contribution

The drop from raw to residual alignment reveals how much of the apparent
class signal in each sub-group's tokens is actually reflected sequence signal.

Usage
-----
python analysis/pattern/analyze_latent_probing.py \\
    --repr_dir   gencode_v49_experiments/beta_vae_subgroup_base_g49/representations \\
    --output_dir gencode_v49_experiments/beta_vae_subgroup_base_g49/latent_probing \\
    --model_label "β-VAE Standard" \\
    --gencode_version v49

Input .npz keys (from extract_representations.py)
--------------------------------------------------
    z              (N, latent_dim)
    tokens         (N, N_tokens, d_proj)
    labels         (N,)
    lengths        (N,)
    gc_content     (N,)
    token_names    (N_tokens,)   string array — sub-group name per token
    block_names    (N_tokens,)   string array — block name per token
    nonb_subgroups (legacy alias, kept for backward compat)
    te_subgroups   (legacy alias, kept for backward compat)
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from data.feature_registry import (REGISTRY, display_block,
                                    display_subgroup, display_subgroups)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Block colour palette (shared convention with analyze_mstar.py)
# ---------------------------------------------------------------------------

BLOCK_COLORS = {
    "nonb":  "#4A90D9",
    "te":    "#F4B942",
    "nonb2": "#9B7FD4",
}
BLOCK_EDGES = {
    "nonb":  "#185FA5",
    "te":    "#BA7517",
    "nonb2": "#5B3FA0",
}


def _token_block_map(token_names: List[str], block_names: List[str]) -> Dict[str, str]:
    """Map token name → block name from the npz arrays (or registry fallback)."""
    if block_names:
        return dict(zip(token_names, block_names))
    # Fallback: derive from registry directly
    return {sg: REGISTRY.token_block(sg) for sg in token_names}


def _block_colors_for(
    token_names: List[str], block_map: Dict[str, str]
) -> List[str]:
    return [BLOCK_COLORS.get(block_map.get(sg, "nonb"), "#999999")
            for sg in token_names]


def _block_boundaries(token_names: List[str], block_map: Dict[str, str]) -> List[float]:
    """Return x-positions (token_names order) where block membership changes."""
    boundaries = []
    prev_block = None
    for i, sg in enumerate(token_names):
        block = block_map.get(sg)
        if prev_block is not None and block != prev_block:
            boundaries.append(i - 0.5)
        prev_block = block
    return boundaries


def _block_legend_handles(token_names: List[str], block_map: Dict[str, str]):
    from matplotlib.patches import Patch
    seen_blocks = list(dict.fromkeys(block_map.get(sg) for sg in token_names))
    return [
        Patch(facecolor=BLOCK_COLORS.get(b, "#999999"), label=display_block(b))
        for b in seen_blocks
    ]


# ---------------------------------------------------------------------------
# Stage 1 — Latent probing
# ---------------------------------------------------------------------------

def probe_latent(
    z:          np.ndarray,   # (N, latent_dim)
    labels:     np.ndarray,   # (N,)
    lengths:    np.ndarray,   # (N,)
    gc_content: np.ndarray,   # (N,)
) -> pd.DataFrame:
    """
    Probe z for confounds and class using linear models.

    Returns DataFrame with columns: confound, r2, type
    """
    rows = []
    z_scaled = StandardScaler().fit_transform(z)

    for name, target in [("length", lengths), ("gc_content", gc_content)]:
        reg = Ridge(alpha=1.0)
        reg.fit(z_scaled, target)
        pred = reg.predict(z_scaled)
        r2   = r2_score(target, pred)
        rows.append({"confound": name, "r2": float(r2), "type": "regression"})

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(z_scaled, labels)
    acc = accuracy_score(labels, clf.predict(z_scaled))
    rows.append({"confound": "class (train acc)", "r2": float(acc), "type": "classification"})

    return pd.DataFrame(rows)


def plot_latent_probing(
    df:          pd.DataFrame,
    output_path: Path,
    fig_tag:     str = "",
) -> None:
    """Bar chart of R² (or accuracy) per confound for z probing."""
    fig, ax = plt.subplots(figsize=(8, 4))

    colors = {"regression": "#4A90D9", "classification": "#E74C3C"}
    bar_colors = [colors[t] for t in df["type"]]

    ax.bar(df["confound"], df["r2"], color=bar_colors,
           edgecolor="black", linewidth=0.6, alpha=0.9)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("R² (regression) / Accuracy (classification)", fontsize=11)
    ax.set_title(
        f"{fig_tag}Latent z — Confound Probing\n"
        "(R² = how much confound variance is encoded in z)",
        fontsize=13, fontweight="bold"
    )
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=colors["regression"],     label="Regression (R²)"),
        Patch(facecolor=colors["classification"], label="Classification (accuracy)"),
    ], fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Stage 2 — Cross-modal correlation (sequence-feature redundancy)
# ---------------------------------------------------------------------------

def compute_redundancy(
    z:           np.ndarray,   # (N, latent_dim)
    tokens:      np.ndarray,   # (N, N_tokens, d_proj)
    token_names: List[str],
) -> pd.DataFrame:
    """
    For each sub-group, measure how much of its token representation variance
    is explained by z via Ridge regression (z → T_sg).

    Returns DataFrame: subgroup, r2_mean, r2_std
    """
    z_scaled = StandardScaler().fit_transform(z)
    rows = []

    for i, sg in enumerate(token_names):
        T_sg = tokens[:, i, :]

        reg = Ridge(alpha=1.0)
        reg.fit(z_scaled, T_sg)
        T_pred = reg.predict(z_scaled)

        r2_per_dim = np.array([
            r2_score(T_sg[:, d], T_pred[:, d])
            for d in range(T_sg.shape[1])
        ])
        r2_per_dim = np.clip(r2_per_dim, 0, 1)

        rows.append({
            "subgroup": sg,
            "r2_mean":  float(r2_per_dim.mean()),
            "r2_std":   float(r2_per_dim.std()),
        })

    return pd.DataFrame(rows)


def plot_redundancy(
    df:          pd.DataFrame,
    block_map:   Dict[str, str],
    output_path: Path,
    fig_tag:     str = "",
) -> None:
    """Bar chart of sequence-feature redundancy per sub-group."""
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(df))

    token_names = df["subgroup"].tolist()
    colors = _block_colors_for(token_names, block_map)

    ax.bar(x, df["r2_mean"], color=colors, edgecolor="black",
           linewidth=0.6, alpha=0.9)
    ax.errorbar(x, df["r2_mean"], yerr=df["r2_std"],
                fmt="none", color="black", capsize=3, linewidth=1.2)

    for boundary in _block_boundaries(token_names, block_map):
        ax.axvline(x=boundary, color="gray", linestyle="--",
                   linewidth=1.5, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(display_subgroups(df["subgroup"]), rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Mean R² (z → token representation)", fontsize=11)
    ax.set_title(
        f"{fig_tag}Sequence–Feature Redundancy\n"
        "(higher = token representation already captured by sequence encoder z)",
        fontsize=13, fontweight="bold"
    )
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=_block_legend_handles(token_names, block_map), fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

def plot_redundancy_crossfold(
    df:          pd.DataFrame,   # subgroup, mean_r2, std_r2
    block_map:   Dict[str, str],
    output_path: Path,
    fig_tag:     str = "",
) -> None:
    """Cross-fold bar chart of sequence-feature redundancy, mean ± std over folds."""
    df = df.sort_values("mean_r2", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(df))

    token_names = df["subgroup"].tolist()
    colors = _block_colors_for(token_names, block_map)

    ax.bar(x, df["mean_r2"], color=colors, edgecolor="black",
           linewidth=0.6, alpha=0.9)
    ax.errorbar(x, df["mean_r2"], yerr=df["std_r2"],
                fmt="none", color="black", capsize=3, linewidth=1.2)

    for boundary in _block_boundaries(token_names, block_map):
        ax.axvline(x=boundary, color="gray", linestyle="--",
                   linewidth=1.5, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(display_subgroups(df["subgroup"]), rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Mean R² (z → token representation), cross-fold ± std", fontsize=11)
    ax.set_title(
        f"{fig_tag}Sequence–Feature Redundancy — Cross-Fold\n"
        "(mean ± std over folds; higher = captured by sequence encoder z)",
        fontsize=13, fontweight="bold"
    )
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=_block_legend_handles(token_names, block_map), fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

# ---------------------------------------------------------------------------
# Stage 3 — Pattern analysis
# ---------------------------------------------------------------------------

def compute_pattern(
    T_sg:   np.ndarray,   # (N, d_proj)
    labels: np.ndarray,   # (N,)
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit linear probe y ~ T_sg and compute activation pattern.
    """
    T_scaled = StandardScaler().fit_transform(T_sg)

    probe = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    probe.fit(T_scaled, labels)
    w = probe.coef_[0]

    Sigma = np.cov(T_scaled.T)
    denom = w @ Sigma @ w
    if abs(denom) < 1e-12:
        a = np.zeros_like(w)
    else:
        a = (Sigma @ w) / denom

    alignment = float(cosine_similarity(
        w.reshape(1, -1), a.reshape(1, -1)
    )[0, 0])

    return w, a, alignment


def compute_residual_tokens(
    z:      np.ndarray,   # (N, latent_dim)
    tokens: np.ndarray,   # (N, N_tokens, d_proj)
) -> np.ndarray:
    """
    Project out z-explained variance from each token representation.
    """
    z_scaled        = StandardScaler().fit_transform(z)
    tokens_residual = tokens.copy()

    for i in range(tokens.shape[1]):
        T_sg = tokens[:, i, :]
        reg  = Ridge(alpha=1.0)
        reg.fit(z_scaled, T_sg)
        tokens_residual[:, i, :] = T_sg - reg.predict(z_scaled)

    return tokens_residual


def run_pattern_analysis(
    tokens:      np.ndarray,   # (N, N_tokens, d_proj)  raw or residual
    labels:      np.ndarray,   # (N,)
    token_names: List[str],
) -> pd.DataFrame:
    """
    Run Pattern analysis for all sub-groups.
    """
    rows = []
    for i, sg in enumerate(token_names):
        T_sg = tokens[:, i, :]
        w, a, alignment = compute_pattern(T_sg, labels)

        T_scaled = StandardScaler().fit_transform(T_sg)
        probe    = LogisticRegression(max_iter=1000, C=1.0)
        probe.fit(T_scaled, labels)
        acc = accuracy_score(labels, probe.predict(T_scaled))

        rows.append({
            "subgroup":        sg,
            "alignment":       alignment,
            "probe_train_acc": float(acc),
        })

    return pd.DataFrame(rows)


def plot_pattern_comparison(
    raw_df:      pd.DataFrame,
    resid_df:    pd.DataFrame,
    block_map:   Dict[str, str],
    output_path: Path,
    fig_tag:     str = "",
) -> None:
    """
    Side-by-side bar chart of pattern-weight alignment:
    raw tokens vs z-orthogonal residual tokens.
    """
    subgroups = raw_df["subgroup"].tolist()
    x         = np.arange(len(subgroups))
    width     = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar(x - width/2, raw_df["alignment"],   width,
           label="Raw tokens",     color="#4A90D9",
           edgecolor="black", linewidth=0.6, alpha=0.9)
    ax.bar(x + width/2, resid_df["alignment"], width,
           label="Residual tokens\n(z-orthogonal)", color="#E74C3C",
           edgecolor="black", linewidth=0.6, alpha=0.9)

    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(1, color="gray",  linewidth=0.8, linestyle="--", alpha=0.5)
    for boundary in _block_boundaries(subgroups, block_map):
        ax.axvline(x=boundary, color="gray", linestyle="--",
                   linewidth=1.5, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(display_subgroups(subgroups), rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Pattern–weight alignment (cosine similarity)", fontsize=11)
    ax.set_title(
        f"{fig_tag}Pattern Analysis — Raw vs Residual Token Alignment\n"
        "(1.0 = genuine class signal, 0.0 = suppressor, <0 = denoising)",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.set_ylim(-0.3, 1.1)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_alignment_bar(
    df:          pd.DataFrame,
    block_map:   Dict[str, str],
    output_path: Path,
    title:       str = "",
) -> None:
    """Simple alignment bar chart for one condition (raw or residual)."""
    token_names = df["subgroup"].tolist()
    x      = np.arange(len(df))
    colors = _block_colors_for(token_names, block_map)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x, df["alignment"], color=colors, edgecolor="black",
           linewidth=0.6, alpha=0.9)
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(1, color="gray",  linewidth=0.8, linestyle="--", alpha=0.4)
    for boundary in _block_boundaries(token_names, block_map):
        ax.axvline(x=boundary, color="gray", linestyle="--",
                   linewidth=1.5, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(display_subgroups(df["subgroup"]), rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Pattern–weight alignment", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(-0.3, 1.1)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=_block_legend_handles(token_names, block_map), fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Cross-fold summary
# ---------------------------------------------------------------------------

def cross_fold_summary(
    fold_results: List[dict],
    token_names:  List[str],
    block_map:    Dict[str, str],
    output_dir:   Path,
    fig_tag:      str = "",
) -> None:
    """Aggregate per-fold results and produce cross-fold summary plots."""
    print("\n" + "=" * 70)
    print("CROSS-FOLD SUMMARY")
    print("=" * 70)

    probe_dfs = [r["probe_df"] for r in fold_results]
    probe_all = pd.concat(probe_dfs, ignore_index=True)
    probe_summary = (probe_all.groupby("confound")["r2"]
                     .agg(["mean", "std"])
                     .reset_index())
    probe_summary.columns = ["confound", "mean_r2", "std_r2"]
    probe_summary.to_csv(output_dir / "cross_fold_latent_probing.csv", index=False)

    print("\nLatent z confound probing (cross-fold mean ± std):")
    for _, row in probe_summary.iterrows():
        print(f"  {row['confound']:25s}: R²={row['mean_r2']:.3f} ± {row['std_r2']:.3f}")

    redund_dfs = [r["redundancy_df"] for r in fold_results]
    redund_all = pd.concat(redund_dfs, ignore_index=True)
    redund_summary = (redund_all.groupby("subgroup")["r2_mean"]
                      .agg(["mean", "std"])
                      .reset_index())
    redund_summary.columns = ["subgroup", "mean_r2", "std_r2"]
    redund_summary = redund_summary.set_index("subgroup").reindex(token_names).reset_index()
    redund_summary.to_csv(output_dir / "cross_fold_redundancy.csv", index=False)

    print("\nSequence-feature redundancy (cross-fold mean R²):")
    for _, row in redund_summary.sort_values("mean_r2", ascending=False).iterrows():
        bar = "█" * int(row["mean_r2"] * 20)
        print(f"  {row['subgroup']:15s}: {row['mean_r2']:.3f} ± {row['std_r2']:.3f}  {bar}")

    plot_redundancy_crossfold(
        redund_summary, block_map,
        output_dir / "cross_fold_redundancy.png",
        fig_tag=fig_tag
    )

    for condition in ("raw", "residual"):
        align_dfs = [r[f"pattern_{condition}_df"] for r in fold_results]
        align_all = pd.concat(align_dfs, ignore_index=True)
        align_summary = (align_all.groupby("subgroup")["alignment"]
                         .agg(["mean", "std"])
                         .reset_index())
        align_summary.columns = ["subgroup", "mean_align", "std_align"]
        align_summary = (align_summary.set_index("subgroup")
                         .reindex(token_names).reset_index())
        align_summary.to_csv(
            output_dir / f"cross_fold_pattern_{condition}.csv", index=False
        )

        print(f"\nPattern alignment — {condition} tokens (cross-fold mean):")
        for _, row in align_summary.sort_values("mean_align", ascending=False).iterrows():
            bar = "█" * max(0, int(row["mean_align"] * 20))
            print(f"  {row['subgroup']:15s}: {row['mean_align']:+.3f} "
                  f"± {row['std_align']:.3f}  {bar}")

    raw_summary   = pd.read_csv(output_dir / "cross_fold_pattern_raw.csv")
    resid_summary = pd.read_csv(output_dir / "cross_fold_pattern_residual.csv")

    subgroups = token_names
    x         = np.arange(len(subgroups))
    width     = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2,
           [raw_summary.set_index("subgroup").loc[sg, "mean_align"] for sg in subgroups],
           width, label="Raw tokens",
           color="#4A90D9", edgecolor="black", linewidth=0.6, alpha=0.9)
    ax.errorbar(
        x - width/2,
        [raw_summary.set_index("subgroup").loc[sg, "mean_align"] for sg in subgroups],
        yerr=[raw_summary.set_index("subgroup").loc[sg, "std_align"] for sg in subgroups],
        fmt="none", color="black", capsize=3, linewidth=1.0
    )
    ax.bar(x + width/2,
           [resid_summary.set_index("subgroup").loc[sg, "mean_align"] for sg in subgroups],
           width, label="Residual tokens\n(z-orthogonal)",
           color="#E74C3C", edgecolor="black", linewidth=0.6, alpha=0.9)
    ax.errorbar(
        x + width/2,
        [resid_summary.set_index("subgroup").loc[sg, "mean_align"] for sg in subgroups],
        yerr=[resid_summary.set_index("subgroup").loc[sg, "std_align"] for sg in subgroups],
        fmt="none", color="black", capsize=3, linewidth=1.0
    )

    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(1, color="gray",  linewidth=0.8, linestyle="--", alpha=0.4)
    for boundary in _block_boundaries(subgroups, block_map):
        ax.axvline(x=boundary, color="gray", linestyle="--",
                   linewidth=1.5, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(display_subgroups(subgroups), rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Pattern–weight alignment (cosine similarity)", fontsize=11)
    ax.set_title(
        f"{fig_tag}Cross-Fold Pattern Analysis — Raw vs Residual\n"
        "(mean ± std over folds)",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.set_ylim(-0.5, 1.2)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / "cross_fold_pattern_comparison.png",
                dpi=350, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {output_dir / 'cross_fold_pattern_comparison.png'}")

    with open(output_dir / "key_findings.txt", "w") as f:
        f.write("β-LNC Latent Probing — Key Findings\n")
        f.write("=" * 50 + "\n\n")

        f.write("1. Latent z confound encoding (cross-fold mean R²)\n")
        for _, row in probe_summary.iterrows():
            f.write(f"   {row['confound']:25s}: {row['mean_r2']:.3f}\n")

        f.write("\n2. Sequence-feature redundancy (z explains token variance)\n")
        for _, row in redund_summary.sort_values("mean_r2", ascending=False).iterrows():
            f.write(f"   {row['subgroup']:15s}: {row['mean_r2']:.3f}\n")

        f.write("\n3. Pattern alignment — raw vs residual\n")
        raw_idx   = raw_summary.set_index("subgroup")
        resid_idx = resid_summary.set_index("subgroup")
        for sg in token_names:
            raw_a   = raw_idx.loc[sg, "mean_align"]
            resid_a = resid_idx.loc[sg, "mean_align"]
            drop    = raw_a - resid_a
            f.write(f"   {sg:15s}: raw={raw_a:+.3f}  "
                    f"residual={resid_a:+.3f}  "
                    f"drop={drop:+.3f}\n")

    print(f"  Saved: {output_dir / 'key_findings.txt'}")


# ---------------------------------------------------------------------------
# npz loading helper
# ---------------------------------------------------------------------------

def _load_token_metadata(data) -> Tuple[List[str], List[str]]:
    """
    Read token_names + block_names from an npz, with fallback to the
    legacy nonb_subgroups/te_subgroups arrays for older extraction outputs.
    """
    if "token_names" in data:
        token_names = list(data["token_names"])
        block_names = (list(data["block_names"])
                       if "block_names" in data else None)
        if block_names is None:
            block_names = [REGISTRY.token_block(sg) for sg in token_names]
        return token_names, block_names

    # Legacy fallback
    nonb_sgs = list(data["nonb_subgroups"])
    te_sgs   = list(data["te_subgroups"])
    token_names = nonb_sgs + te_sgs
    block_names = ["nonb"] * len(nonb_sgs) + ["te"] * len(te_sgs)
    return token_names, block_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Latent probing and Pattern analysis for BetaVAESubgroup"
    )
    parser.add_argument("--repr_dir",        required=True,
                        help="Directory containing fold_*_repr.npz files")
    parser.add_argument("--output_dir",      required=True)
    parser.add_argument("--model_label",     default="β-VAE")
    parser.add_argument("--gencode_version", default="v49")
    args = parser.parse_args()

    repr_dir   = Path(args.repr_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag_parts = [p for p in [args.model_label,
                 f"GENCODE {args.gencode_version}"] if p]
    fig_tag   = " | ".join(tag_parts) + " — " if tag_parts else ""

    fold_files = sorted(repr_dir.glob("fold_*_repr.npz"))
    if not fold_files:
        print(f"ERROR: No fold_*_repr.npz found in {repr_dir}")
        return

    print("=" * 70)
    print("LATENT PROBING + PATTERN ANALYSIS")
    print("=" * 70)
    print(f"Found {len(fold_files)} fold(s)")

    fold_results = []
    final_token_names = None
    final_block_map   = None

    for fold_file in fold_files:
        fold_name = fold_file.stem.replace("_repr", "")
        print(f"\n{'='*60}")
        print(f"PROCESSING {fold_name.upper()}")
        print(f"{'='*60}")

        fold_out = output_dir / fold_name
        fold_out.mkdir(exist_ok=True)

        data        = np.load(fold_file, allow_pickle=True)
        z           = data["z"].astype(np.float32)
        tokens      = data["tokens"].astype(np.float32)
        labels      = data["labels"].astype(int)
        lengths     = data["lengths"].astype(np.float32)
        gc_content  = data["gc_content"].astype(np.float32)

        token_names, block_names = _load_token_metadata(data)
        block_map = dict(zip(token_names, block_names))
        final_token_names = token_names
        final_block_map   = block_map

        print(f"  N={len(labels):,}  z={z.shape}  tokens={tokens.shape}")
        print(f"  lncRNA={( labels==0).sum():,}  mRNA={(labels==1).sum():,}")
        print(f"  Blocks: {dict.fromkeys(block_names)}")

        # ── Stage 1: Latent probing ───────────────────────────────────────────
        print("\n  Stage 1: Latent probing of z...")
        probe_df = probe_latent(z, labels, lengths, gc_content)
        probe_df.to_csv(fold_out / "latent_probing.csv", index=False)
        plot_latent_probing(
            probe_df,
            fold_out / "latent_probing.png",
            fig_tag=f"{fig_tag}{fold_name} — "
        )
        for _, row in probe_df.iterrows():
            print(f"    {row['confound']:25s}: {row['r2']:.3f}")

        # ── Stage 2: Sequence-feature redundancy ──────────────────────────────
        print("\n  Stage 2: Sequence-feature redundancy...")
        redundancy_df = compute_redundancy(z, tokens, token_names)
        redundancy_df.to_csv(fold_out / "redundancy.csv", index=False)
        plot_redundancy(
            redundancy_df, block_map,
            fold_out / "redundancy.png",
            fig_tag=f"{fig_tag}{fold_name} — "
        )
        for _, row in redundancy_df.sort_values("r2_mean", ascending=False).iterrows():
            print(f"    {row['subgroup']:15s}: R²={row['r2_mean']:.3f} "
                  f"± {row['r2_std']:.3f}")

        # ── Stage 3: Pattern analysis ─────────────────────────────────────────
        print("\n  Stage 3: Pattern analysis...")

        print("    Raw tokens:")
        raw_df = run_pattern_analysis(tokens, labels, token_names)
        raw_df.to_csv(fold_out / "pattern_raw.csv", index=False)
        plot_alignment_bar(
            raw_df, block_map,
            fold_out / "pattern_raw.png",
            title=f"{fig_tag}{fold_name} — Pattern Alignment (Raw Tokens)"
        )
        for _, row in raw_df.sort_values("alignment", ascending=False).iterrows():
            print(f"    {row['subgroup']:15s}: alignment={row['alignment']:+.3f}  "
                  f"probe_acc={row['probe_train_acc']:.3f}")

        print("    Residual tokens (z-orthogonal):")
        tokens_residual = compute_residual_tokens(z, tokens)
        resid_df        = run_pattern_analysis(tokens_residual, labels, token_names)
        resid_df.to_csv(fold_out / "pattern_residual.csv", index=False)
        plot_alignment_bar(
            resid_df, block_map,
            fold_out / "pattern_residual.png",
            title=(f"{fig_tag}{fold_name} — "
                   f"Pattern Alignment (Residual / z-Orthogonal Tokens)")
        )
        for _, row in resid_df.sort_values("alignment", ascending=False).iterrows():
            drop = (raw_df.set_index("subgroup")
                    .loc[row["subgroup"], "alignment"] - row["alignment"])
            print(f"    {row['subgroup']:15s}: alignment={row['alignment']:+.3f}  "
                  f"drop={drop:+.3f}")

        plot_pattern_comparison(
            raw_df, resid_df, block_map,
            fold_out / "pattern_comparison.png",
            fig_tag=f"{fig_tag}{fold_name} — "
        )

        fold_results.append({
            "fold":               fold_name,
            "probe_df":           probe_df,
            "redundancy_df":      redundancy_df,
            "pattern_raw_df":     raw_df,
            "pattern_residual_df":resid_df,
        })
        print(f"\n  {fold_name} complete → {fold_out}/")

    # ── Cross-fold summary ────────────────────────────────────────────────────
    if len(fold_results) > 1:
        cross_fold_summary(
            fold_results, final_token_names, final_block_map,
            output_dir, fig_tag
        )
    else:
        print("\nOnly one fold — skipping cross-fold summary.")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir}/")
    print(f"\nPer-fold outputs (fold_N/):")
    print(f"  latent_probing.csv / .png")
    print(f"  redundancy.csv / .png")
    print(f"  pattern_raw.csv / .png")
    print(f"  pattern_residual.csv / .png")
    print(f"  pattern_comparison.png")
    print(f"\nCross-fold outputs:")
    print(f"  cross_fold_latent_probing.csv")
    print(f"  cross_fold_redundancy.csv")
    print(f"  cross_fold_pattern_raw.csv")
    print(f"  cross_fold_pattern_residual.csv")
    print(f"  cross_fold_pattern_comparison.png")
    print(f"  key_findings.txt")


if __name__ == "__main__":
    main()