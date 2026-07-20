#!/usr/bin/env python3
"""
plot_patching_symmetry.py

Reads patching_summary.csv (output of patch_tokens.py) and produces a
horizontal, log-scale bar chart ranking subgroups by causal patching
symmetry score.

Usage
-----
python plot_patching_symmetry.py \
    --patching_csv gencode_v49_experiments/beta_vae_subgroup_base_g49_new/patching/patching_summary.csv \
    --output_dir   gencode_v49_experiments/benchmark_comparison \
    --release      v49 \
    --display_names
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Inter", "Helvetica Neue", "Arial", "Liberation Sans"],
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
})

BLOCK_COLORS = {"nonb": "#2a78d6", "te": "#1baf7a", "nonb2": "#eda100"}
BLOCK_LABELS = {"nonb": "NonB", "te": "REP", "nonb2": "NonB2"}
DISPLAY_NAMES = {
    "TE_CORE": "REP_CORE", "TE_LCTR": "REP_LCTR", "TE_QUALITY": "REP_QUALITY",
    "TE_PSEUDO": "REP_PSEUDO", "TE_UNKNOWN": "REP_UNKNOWN", "TE_GLOBAL": "REP_GLOBAL",
}


def load_symmetry(patching_csv: Path) -> pd.DataFrame:
    """
    Load patching_summary.csv and reduce to one row per subgroup with its
    symmetry_score (direction-agnostic — dedupe on lnc_to_mrna row).
    """
    df = pd.read_csv(patching_csv)
    df.columns = df.columns.str.strip().str.lower()

    if "direction" in df.columns:
        df = df[df["direction"] == "lnc_to_mrna"].copy()

    if "symmetry_score" not in df.columns:
        df["symmetry_score"] = df["mean_delta_logit"].abs()

    df = df[["subgroup", "block", "symmetry_score"]].drop_duplicates("subgroup")
    return df.sort_values("symmetry_score", ascending=False).reset_index(drop=True)


def make_plot(df: pd.DataFrame, release: str, output_dir: Path,
              display_names: bool = False, threshold: float = None,
              annotate_top_n: int = 2):
    df_plot = df.sort_values("symmetry_score").reset_index(drop=True)  # ascending for barh

    labels = df_plot["subgroup"].map(lambda s: DISPLAY_NAMES.get(s, s)) if display_names else df_plot["subgroup"]
    colors = df_plot["block"].map(lambda b: BLOCK_COLORS.get(b, "#999999"))

    # Vertical bars: sort descending left-to-right (largest first)
    df_plot = df_plot.sort_values("symmetry_score", ascending=False).reset_index(drop=True)
    labels = df_plot["subgroup"].map(lambda s: DISPLAY_NAMES.get(s, s)) if display_names else df_plot["subgroup"]
    colors = df_plot["block"].map(lambda b: BLOCK_COLORS.get(b, "#999999"))

    fig, ax = plt.subplots(figsize=(0.55 * len(df_plot) + 2, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(df_plot))
    ax.bar(x, df_plot["symmetry_score"], width=0.65, color=colors, edgecolor="none")

    ax.set_yscale("log")
    ymax = max(df_plot["symmetry_score"].max() * 1.8, 1e-3)
    ax.set_ylim(1e-5, ymax)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5, rotation=45, ha="right")
    for tick_label, block in zip(ax.get_xticklabels(), df_plot["block"]):
        tick_label.set_color(BLOCK_COLORS.get(block, "#333333"))

    if threshold is not None:
        ax.axhline(threshold, color="#aaaaaa", lw=1.2, ls="--", zorder=0)

    ax.set_ylabel("Patching symmetry score (log scale)", fontsize=12,
                  color="#000000", fontweight="bold")
    ax.tick_params(axis="y", labelsize=9.5, colors="#000000")
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.yaxis.grid(True, which="both", color="#e1e0d9", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Annotate top-N bars with their exact value
    top_n_subgroups = df.head(annotate_top_n)["subgroup"].tolist()
    for sg in top_n_subgroups:
        idx = df_plot.index[df_plot["subgroup"] == sg][0]
        val = df_plot.loc[idx, "symmetry_score"]
        ax.text(idx, val * 1.15, f"{val:.3f}", ha="center", fontsize=9.5,
                color="#000000", fontweight="bold")

    present_blocks = [b for b in ("nonb", "te", "nonb2") if b in df_plot["block"].unique()]
    patches = [mpatches.Patch(color=BLOCK_COLORS[b], label=BLOCK_LABELS[b])
               for b in present_blocks]
    ax.legend(handles=patches, fontsize=9.5, frameon=True, framealpha=0.9,
              edgecolor="#e1e0d9", loc="lower right")

    ax.set_title(f"Causal patching symmetry — GENCODE {release}\n"
                 f"log scale, ranked by symmetry score",
                 fontsize=12.5, color="#000000", fontweight="bold", pad=10)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = output_dir / f"patching_symmetry_bar_{release}.{ext}"
        fig.savefig(p, dpi=1000, bbox_inches="tight", facecolor="white")
        print(f"Saved: {p}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patching_csv", required=True,
                   help="Path to patching_summary.csv from patch_tokens.py")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--release", default="v49")
    p.add_argument("--display_names", action="store_true",
                   help="Remap TE_* -> REP_* in labels")
    p.add_argument("--threshold", type=float, default=None,
                   help="Optional vertical dashed line at this symmetry score")
    p.add_argument("--annotate_top_n", type=int, default=2,
                   help="Number of top bars to annotate with exact value (default: 2)")
    args = p.parse_args()

    df = load_symmetry(Path(args.patching_csv))
    print(df.to_string(index=False))

    make_plot(
        df, args.release, Path(args.output_dir),
        display_names=args.display_names,
        threshold=args.threshold,
        annotate_top_n=args.annotate_top_n,
    )


if __name__ == "__main__":
    main()