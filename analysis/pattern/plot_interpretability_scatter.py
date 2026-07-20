#!/usr/bin/env python3
"""
plot_interpretability_scatter.py

Four-dimensional interpretability scatter for β-LNC.
All three layers of the interpretability framework encoded simultaneously:

  x    — log10(Fréchet distance)           [Layer 2: associational]
  y    — feature-zero ablation Δacc        [Layer 1: model-functional]
  size — patching symmetry score           [Layer 3: causal]
  edge — residual pattern alignment        [Layer 1: z-independent signal]
  color — block (NonB / REP / NonB2)

Usage
-----
python analysis/benchmark/plot_interpretability_scatter.py \\
    --frechet_csv          data/frechet_results_g49.csv \\
    --ablation_csv         <exp>/ablation/analysis/all_folds_ablation.csv \\
    --pattern_raw_csv      <exp>/latent_probing/cross_fold_pattern_raw/raw.csv \\
    --pattern_residual_csv <exp>/latent_probing/cross_fold_pattern_raw/residual.csv \\
    --patching_csv         <exp>/patching/patching_summary.csv \\
    --output_dir           gencode_v49_experiments/benchmark_comparison \\
    --release              v49 \\
    --display_names

Expected CSV schemas
--------------------
frechet_csv          : subgroup, block, frechet_dist, [...]
ablation_csv         : subgroup, block, mode, acc_drop, fold, ...
pattern_raw_csv      : subgroup, mean_align, std_align
pattern_residual_csv : subgroup, mean_align, std_align
patching_csv         : subgroup, block, direction, mean_delta_logit,
                       std_delta_logit, symmetry_score, causal_effect, ...
                       (output of patch_tokens.py patching_summary.csv)

All CSVs must share the same subgroup key convention (internal TE_* keys).
Pass --display_names to remap TE_* → REP_* in figure labels only.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Inter", "Helvetica Neue", "Arial", "Liberation Sans"],
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
})

# ── Constants ─────────────────────────────────────────────────────────────────

BLOCK_COLORS = {
    "nonb":  "#2a78d6",
    "te":    "#1baf7a",
    "nonb2": "#eda100",
}
BLOCK_LABELS = {
    "nonb":  "NonB",
    "te":    "REP",
    "nonb2": "NonB2",
}

DISPLAY_NAMES = {
    "TE_CORE":    "REP_CORE",
    "TE_LCTR":    "REP_LCTR",
    "TE_QUALITY": "REP_QUALITY",
    "TE_PSEUDO":  "REP_PSEUDO",
    "TE_UNKNOWN": "REP_UNKNOWN",
    "TE_GLOBAL":  "REP_GLOBAL",
}

# All 20 subgroups labeled — offsets tuned to avoid overlap in dense clusters
ALWAYS_LABEL = None   # None = label all

FD_DIVIDER  = 5.0
ABL_DIVIDER = 0.003

QUADRANT_LABELS = {
    "TL": "data signal\nnot used",
    "TR": "detected by\nboth layers",
    "BL": "neither layer",
    "BR": "model-dependent\n(not by Fréchet)",
}

# Per-subgroup label offsets — tuned to spread the dense bottom-left cluster
# Format: (dx in log10-FD units, dy in Δacc units, horizontal alignment)
LABEL_OFFSETS = {
    # Top-right: anchors
    "SS_BINNED":    ( 0.08,  0.0005,  "left"),
    "TE_CORE":      ( 0.06,  0.0003,  "left"),
    "REP_CORE":     ( 0.06,  0.0003,  "left"),
    # Mid-right
    "IR":           (-0.07,  0.0003,  "right"),
    "TE_LCTR":      ( 0.06,  0.0002,  "left"),
    "TE_QUALITY":   ( 0.06, -0.0003,  "left"),
    "TE_PSEUDO":    ( 0.06,  0.0003,  "left"),
    "TE_UNKNOWN":   ( 0.06, -0.0004,  "left"),
    "TE_GLOBAL":    (-0.07,  0.0003,  "right"),
    # Mid cluster (FD 3-10) — staggered to reduce overlap
    "GQ":           ( 0.06,  0.0003,  "left"),
    "STR":          (-0.07,  0.0004,  "right"),
    "DR":           ( 0.06, -0.0002,  "left"),
    "MR":           ( 0.06,  0.00005, "left"),
    "TRI":          (-0.07,  0.0002,  "right"),
    "Z":            (-0.07, -0.0002,  "right"),
    "APR":          ( 0.06, -0.0003,  "left"),
    "GLOBAL":       (-0.08,  0.0003,  "right"),
    # Bottom-left: NonB2 independent channel — spread vertically
    "SS_COUNT":     (-0.10, -0.00015, "right"),
    "SS_RELPOS":    (-0.10,  0.00015, "right"),
    "RG4":          ( 0.07,  0.00010, "left"),
    "SS_STAB":      ( 0.07, -0.00010, "left"),
}

LABEL_MIN_ABL = 0.0   # label all subgroups regardless of ablation value


# ── Helpers ───────────────────────────────────────────────────────────────────

def norm_cols(df):
    df.columns = df.columns.str.strip().str.lower()
    if "subgroup" not in df.columns:
        df.rename(columns={df.columns[0]: "subgroup"}, inplace=True)
    return df


def load_pattern(path, suffix):
    df = norm_cols(pd.read_csv(path))
    # Handle both "mean_align" (real model) and "alignment" (synthetic)
    align_col = "mean_align" if "mean_align" in df.columns else "alignment"
    std_col   = "std_align"  if "std_align"  in df.columns else None
    df = df.rename(columns={align_col: f"mean_align_{suffix}"})
    if std_col and std_col in df.columns:
        df = df.rename(columns={std_col: f"std_align_{suffix}"})
        return df[["subgroup", f"mean_align_{suffix}", f"std_align_{suffix}"]]
    return df[["subgroup", f"mean_align_{suffix}"]]


def load_patching(path):
    """
    Load patching_summary.csv — keep one row per subgroup (lnc_to_mrna direction)
    with symmetry_score and causal_effect columns.
    """
    df = norm_cols(pd.read_csv(path))
    # Keep lnc_to_mrna direction for mean_delta_logit; symmetry_score is
    # direction-agnostic so either row works — just deduplicate by subgroup.
    if "direction" in df.columns:
        df = df[df["direction"] == "lnc_to_mrna"].copy()
    # Ensure symmetry_score exists — compute from abs mean_delta_logit if absent
    if "symmetry_score" not in df.columns:
        df["symmetry_score"] = df["mean_delta_logit"].abs()
    return df[["subgroup", "symmetry_score",
               "mean_delta_logit", "std_delta_logit"]].copy()


def load_csvs(frechet_path, ablation_path, raw_path,
              residual_path, patching_path=None):
    fd  = norm_cols(pd.read_csv(frechet_path))
    abl = norm_cols(pd.read_csv(ablation_path))
    raw = load_pattern(raw_path,      "raw")
    res = load_pattern(residual_path, "residual")

    # Ablation: keep feature_zero, average acc_drop over folds
    if "mode" in abl.columns:
        abl = abl[abl["mode"] == "feature_zero"]
    abl = (abl.groupby("subgroup", as_index=False)["acc_drop"]
              .agg(mean_acc_drop="mean", std_acc_drop="std"))

    merged = (fd
              .merge(abl, on="subgroup", how="inner")
              .merge(raw, on="subgroup", how="inner")
              .merge(res, on="subgroup", how="inner"))

    if patching_path is not None:
        pat = load_patching(patching_path)
        merged = merged.merge(pat, on="subgroup", how="left")
        # Fill missing patching values with 0 (subgroups not in patching CSV)
        for col in ["symmetry_score", "mean_delta_logit", "std_delta_logit"]:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.0)
    else:
        merged["symmetry_score"]   = 0.0
        merged["mean_delta_logit"] = 0.0

    if len(merged) == 0:
        sys.exit("ERROR: no rows after merging CSVs")

    print(f"  Fréchet: {len(fd)}  Ablation: {len(abl)}  "
          f"Pattern: {len(res)}  →  Merged: {len(merged)}"
          + (f"  Patching: {len(pat)}" if patching_path else "  (no patching)"))
    return merged


def resolve_block(row):
    sg = row["subgroup"].upper()
    if sg.startswith(("SS_", "RG4")):
        return "nonb2"
    if sg.startswith(("TE_", "REP_")):
        return "te"
    raw = str(row.get("block", "")).lower()
    if raw == "nonb":
        return "nonb"
    return "nonb"


def bubble_size(sym, scale=2800):
    """Map patching symmetry score → bubble area. Minimum size for visibility."""
    return np.clip(sym, 0.005, 1.0) ** 0.9 * scale





# ── Plot ──────────────────────────────────────────────────────────────────────

def make_scatter(df, release, output_dir, display_names=False,
                 has_patching=False):
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    log_fd  = np.log10(np.maximum(df["frechet_dist"].values, 0.05))
    abl     = np.maximum(df["mean_acc_drop"].values, 0.0)
    sym     = df["symmetry_score"].values
    blocks  = df.apply(resolve_block, axis=1).values
    abl     = np.maximum(abl, 0.0)   # clip negative noise-floor ablation values

    # Quadrant dividers
    xdiv = np.log10(FD_DIVIDER)
    ax.axvline(xdiv,        color="#c8c7c0", lw=0.8, ls="--", zorder=0)
    ax.axhline(ABL_DIVIDER, color="#c8c7c0", lw=0.8, ls="--", zorder=0)

    # Draw points: size = patching symmetry, edge = residual alignment
    for i in range(len(df)):
        c  = BLOCK_COLORS[blocks[i]]
        ax.scatter(log_fd[i], abl[i],
                   s          = bubble_size(sym[i]),
                   color      = c,
                   alpha      = 0.65,
                   linewidths = 0.6,
                   edgecolors = "white",
                   zorder     = 3)

    # Labels — all subgroups labeled; ALWAYS_LABEL=None means label all
    for i, row in df.iterrows():
        sg = row["subgroup"]
        if ALWAYS_LABEL is not None and sg not in ALWAYS_LABEL:
            continue
        iloc = df.index.get_loc(i)
        if abl[iloc] < LABEL_MIN_ABL and sym[iloc] < LABEL_MIN_ABL:
            continue  # only skip if truly zero on all axes
        label = DISPLAY_NAMES.get(sg, sg) if display_names else sg
        c     = BLOCK_COLORS[blocks[iloc]]
        dx, dy, ha = LABEL_OFFSETS.get(sg, (0.06, 0.0002, "left"))
        x, y = log_fd[iloc], abl[iloc]
        ax.annotate(
            label, xy=(x, y), xytext=(x + dx, y + dy),
            fontsize=8.5, color=c, ha=ha, va="center",
            arrowprops=dict(arrowstyle="-", color=c, lw=0.6, alpha=0.6),
            zorder=5,
        )

    # Quadrant text
    ax.autoscale()
    xl, xr = ax.get_xlim()
    yl, yr = ax.get_ylim()
    ax.set_xlim(xl, xr + 0.25)
    ax.set_ylim(bottom=-0.0008, top=0.025)  # headroom above SS_BINNED (~0.015)
    qkw = dict(fontsize=8, color="#b0afa8", va="top", linespacing=1.5)
    ax.text(xdiv - 0.05, yr * 0.97, QUADRANT_LABELS["TL"], ha="right", **qkw)
    ax.text(xdiv + 0.05, yr * 0.97, QUADRANT_LABELS["TR"], ha="left",  **qkw)
    ax.text(xdiv - 0.05, ABL_DIVIDER * 0.6,
            QUADRANT_LABELS["BL"], ha="right", **qkw)
    ax.text(xdiv + 0.05, ABL_DIVIDER * 0.6,
            QUADRANT_LABELS["BR"], ha="left",  **qkw)

    # Axes
    ax.set_yscale("symlog", linthresh=0.001, linscale=0.5)
    import matplotlib.ticker as mticker
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("log₁₀(Fréchet distance)", fontsize=12, color="#000000", fontweight="bold")
    ax.set_ylabel("Ablation Δacc (symlog)", fontsize=12, color="#000000", fontweight="bold")
    ax.tick_params(labelsize=10, colors="#000000")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.spines["left"].set_color("#c3c2b7")
    xticks = ax.get_xticks()
    ax.set_xticklabels(
        [f"{t:.1f}\n(FD={10**t:.1f})" if t >= 0 else f"{t:.1f}" for t in xticks],
        fontsize=9, color="#000000"
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    # Block colors
    block_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=BLOCK_COLORS[k], markersize=8,
               label=BLOCK_LABELS[k])
        for k in ("nonb", "te", "nonb2")
    ]

    # Bubble size = patching symmetry
    size_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#aaa",
               markersize=np.sqrt(bubble_size(v) / np.pi) * 0.9,
               label=f"patching = {v:.2f}", alpha=0.7)
        for v in (0.05, 0.3, 0.7)
    ] if has_patching else []

    separator = [Line2D([0], [0], color="none", label=" ")]
    all_handles = (block_handles + separator + size_handles)

    leg = ax.legend(
        handles    = all_handles,
        fontsize   = 9,
        frameon    = True,
        framealpha = 0.92,
        edgecolor  = "#e1e0d9",
        loc        = "upper left",
        ncol       = 1,
        handlelength = 1.8,
        labelspacing = 0.4,
    )
    leg.get_frame().set_linewidth(0.5)

    # Title
    causal_note = ("  ·  size: causal patching symmetry" if has_patching
                   else "  ·  size: patching (not yet run)")
    ax.set_title(
        f"β-LNC GENCODE {release} — three-layer interpretability\n"
        f"x: Fréchet (associational)  ·  y: ablation (model-functional)"
        f"{causal_note}",
        fontsize=12, color="#000000", fontweight="bold", pad=8,
    )

    plt.tight_layout()

    out_dir = Path(output_dir)
    suffix  = "_with_patching" if has_patching else ""
    for ext in (".pdf", ".png"):
        p = out_dir / f"interpretability_scatter_{release}{suffix}{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=1500)
        print(f"Saved: {p}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--frechet_csv",          required=True)
    p.add_argument("--ablation_csv",         required=True)
    p.add_argument("--pattern_raw_csv",      required=True)
    p.add_argument("--pattern_residual_csv", required=True)
    p.add_argument("--patching_csv",         default=None,
                   help="patching_summary.csv from patch_tokens.py "
                        "(optional — omit to plot without causal layer)")
    p.add_argument("--output_dir",           required=True)
    p.add_argument("--release",              default="v49")
    p.add_argument("--display_names",        action="store_true",
                   help="Remap TE_* → REP_* in figure labels")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    has_patching = args.patching_csv is not None

    df = load_csvs(
        args.frechet_csv, args.ablation_csv,
        args.pattern_raw_csv, args.pattern_residual_csv,
        args.patching_csv,
    )
    print(f"  Subgroups: {sorted(df['subgroup'].tolist())}")
    make_scatter(df, args.release, args.output_dir,
                 args.display_names, has_patching)