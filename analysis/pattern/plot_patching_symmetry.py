#!/usr/bin/env python3
"""
plot_patching_symmetry.py

Reads a patching summary CSV (from patch_tokens.py or, filtered to
patch_mode == "single", from patch_tokens_v2.py) and produces:
  1. patching_symmetry_bar_<release>.{png,pdf} — horizontal, log-scale
     bar chart ranking subgroups by symmetry score (unchanged from before).
  2. symmetry_vs_iia_<release>.{png,pdf} — two-panel comparison, shared
     x-axis sorted by symmetry score: symmetry score (log) on top,
     IIA (linear, 0-1) below. Only produced if the CSV has an
     iia_overall column.

Usage
-----
python plot_patching_symmetry.py \
    --patching_csv gencode_v49_experiments/beta_vae_subgroup_base_g49/patching/patching_summary.csv \
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
    Load a patching summary CSV (from patch_tokens.py or patch_tokens_v2.py)
    and reduce to one row per subgroup with symmetry_score and iia_overall
    (direction-agnostic — both are already collapsed across direction in
    the source CSV; dedupe on lnc_to_mrna row where direction is still
    present per-row).

    patch_tokens_v2.py's summary contains multiple patch_mode/patch_scope
    rows (single/block/all/shuffled) per subgroup — filtered here to
    patch_mode == "single" to match patch_tokens.py's single-token-only
    scope. Block/all-scope data is used separately for the sufficiency
    curve, not this plot.
    """
    df = pd.read_csv(patching_csv)
    df.columns = df.columns.str.strip().str.lower()

    if "patch_mode" in df.columns:
        df = df[df["patch_mode"] == "single"].copy()
        df = df.rename(columns={"patch_scope": "subgroup"})

    if "direction" in df.columns:
        df = df[df["direction"] == "lnc_to_mrna"].copy()

    if "symmetry_score" not in df.columns:
        df["symmetry_score"] = df["mean_delta_logit"].abs()

    cols = ["subgroup", "block", "symmetry_score"]
    if "iia_overall" in df.columns:
        cols.append("iia_overall")

    df = df[cols].drop_duplicates("subgroup")
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


def make_symmetry_vs_iia_plot(df: pd.DataFrame, release: str, output_dir: Path,
                              display_names: bool = False, annotate_top_n: int = 2):
    """
    Two-panel comparison, shared x-axis sorted by symmetry score:
    top = symmetry score (continuous effect size, log scale),
    bottom = IIA (discrete intervention sufficiency, linear scale).

    Symmetry and IIA answer different questions (effect size vs whether
    the intervention alone flips the class) and can diverge sharply for
    the same subgroup — this figure makes that divergence visible.
    """
    if "iia_overall" not in df.columns:
        print("  No iia_overall column found — skipping symmetry-vs-IIA plot")
        return

    df_plot = df.sort_values("symmetry_score", ascending=False).reset_index(drop=True)
    names = df_plot["subgroup"].tolist()
    labels = [DISPLAY_NAMES.get(s, s) for s in names] if display_names else names
    colors = df_plot["block"].map(lambda b: BLOCK_COLORS.get(b, "#999999"))
    edge_colors = colors

    x = np.arange(len(df_plot))
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(0.55 * len(df_plot) + 2, 9),
        sharex=True, gridspec_kw={"height_ratios": [1.3, 1]},
    )
    fig.patch.set_facecolor("white")

    ax1.bar(x, df_plot["symmetry_score"], width=0.65, color=colors, edgecolor="none")
    ax1.set_yscale("log")
    ymax = max(df_plot["symmetry_score"].max() * 1.8, 1e-3)
    ax1.set_ylim(1e-5, ymax)
    ax1.set_ylabel("Symmetry score\n(log scale)", fontsize=11, fontweight="bold")
    ax1.yaxis.grid(True, which="both", color="#e1e0d9", linewidth=0.6, zorder=0)
    ax1.set_axisbelow(True)
    ax1.spines["left"].set_color("#c3c2b7")
    ax1.spines["bottom"].set_color("#c3c2b7")
    ax1.set_title(f"Causal Layer: Symmetry Score vs IIA — GENCODE {release}",
                  fontsize=13, fontweight="bold")

    top_n_subgroups = df_plot.head(annotate_top_n)["subgroup"].tolist()
    for sg in top_n_subgroups:
        idx = df_plot.index[df_plot["subgroup"] == sg][0]
        val = df_plot.loc[idx, "symmetry_score"]
        ax1.text(idx, val * 1.15, f"{val:.3f}", ha="center", fontsize=9.5,
                 color="#000000", fontweight="bold")

    ax2.bar(x, df_plot["iia_overall"], width=0.65, color=colors, edgecolor="none")
    ax2.set_ylim(0, min(df_plot["iia_overall"].max() * 1.3 + 0.05, 1.05))
    ax2.set_ylabel("IIA (fraction of trials\nflipped by patching)", fontsize=11, fontweight="bold")
    ax2.yaxis.grid(True, color="#e1e0d9", linewidth=0.6, zorder=0)
    ax2.set_axisbelow(True)
    ax2.spines["left"].set_color("#c3c2b7")
    ax2.spines["bottom"].set_color("#c3c2b7")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10.5, rotation=45, ha="right")
    for tick_label, block in zip(ax2.get_xticklabels(), df_plot["block"]):
        tick_label.set_color(BLOCK_COLORS.get(block, "#333333"))

    for sg in top_n_subgroups:
        idx = df_plot.index[df_plot["subgroup"] == sg][0]
        val = df_plot.loc[idx, "iia_overall"]
        ax2.text(idx, val + 0.02, f"{val:.3f}", ha="center", fontsize=9.5,
                 color="#000000", fontweight="bold")

    present_blocks = [b for b in ("nonb", "te", "nonb2") if b in df_plot["block"].unique()]
    patches = [mpatches.Patch(color=BLOCK_COLORS[b], label=BLOCK_LABELS[b])
               for b in present_blocks]
    ax1.legend(handles=patches, fontsize=9.5, frameon=True, framealpha=0.9,
              edgecolor="#e1e0d9", loc="upper right")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = output_dir / f"symmetry_vs_iia_{release}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved: {p}")
    plt.close(fig)


def load_sufficiency_curve(patching_csv: Path) -> pd.DataFrame:
    """
    Load a patch_tokens.py summary CSV and reduce to one row per
    (patch_mode, patch_scope) with symmetry_score and iia_overall,
    covering single/block/all scopes — the sufficiency-curve data.
    Requires a patch_tokens.py-format CSV (patch_mode column present); returns an
    empty DataFrame otherwise.
    """
    df = pd.read_csv(patching_csv)
    df.columns = df.columns.str.strip().str.lower()

    if "patch_mode" not in df.columns:
        return pd.DataFrame()

    df = df[df["patch_mode"].isin(["single", "block", "all"])].copy()
    if "direction" in df.columns:
        df = df[df["direction"] == "lnc_to_mrna"].copy()

    cols = ["patch_mode", "patch_scope", "block", "symmetry_score"]
    if "iia_overall" in df.columns:
        cols.append("iia_overall")

    df = df[cols].drop_duplicates(["patch_mode", "patch_scope"])
    return df.reset_index(drop=True)


def make_sufficiency_curve_plot(df: pd.DataFrame, release: str, output_dir: Path,
                                display_names: bool = False, top_n_single: int = 6):
    """
    Sufficiency curve: single-token -> block -> all-token IIA, showing
    whether causal effect concentrates in individual subgroups, is
    roughly additive within a block, or requires the whole token pathway.
    x-axis is scope, grouped single/block/all — NOT sorted by symmetry
    (unlike the other two plots in this script), since the point here is
    the progression across scope size, not a subgroup ranking.

    Only the top_n_single single-scope subgroups (by symmetry score) are
    shown — with 20 subgroups, most have near-zero IIA and showing all of
    them wastes most of the plot's width on empty bars while pushing the
    single/block/all section labels out of visual alignment with their bars.
    """
    if df.empty or "iia_overall" not in df.columns:
        print("  No sufficiency-curve data (need a v2-format CSV with "
              "block/all scopes and iia_overall) — skipping")
        return

    single_df = df[df["patch_mode"] == "single"].sort_values(
        "symmetry_score", ascending=False
    ).head(top_n_single)
    block_df = df[df["patch_mode"] == "block"].sort_values(
        "symmetry_score", ascending=False
    )
    all_df = df[df["patch_mode"] == "all"]

    ordered = pd.concat([single_df, block_df, all_df], ignore_index=True)
    names = ordered["patch_scope"].tolist()
    labels = [DISPLAY_NAMES.get(s, s) for s in names] if display_names else names

    def _scope_color(row):
        if row["patch_mode"] == "all":
            return "#555555"
        return BLOCK_COLORS.get(row["block"], "#999999")

    colors = ordered.apply(_scope_color, axis=1)

    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(0.55 * len(ordered) + 2, 6.5))
    fig.patch.set_facecolor("white")

    ax.bar(x, ordered["iia_overall"], width=0.65, color=colors, edgecolor="none")

    for boundary_mode in ["block", "all"]:
        first_idx = ordered.index[ordered["patch_mode"] == boundary_mode]
        if len(first_idx) > 0:
            ax.axvline(first_idx[0] - 0.5, color="#aaaaaa", lw=1.2, ls="--", zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=45, ha="right")
    ax.set_ylabel("IIA (fraction of trials flipped by patching)", fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, color="#e1e0d9", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")

    ax.set_title(
        f"Causal Layer: Sufficiency Curve — GENCODE {release}\n"
        "single token → block → all tokens",
        fontsize=12.5, fontweight="bold", pad=28
    )

    # Reserve headroom above the tallest bar, then place scope-section
    # labels at a fixed axes-fraction y within that headroom. Sections
    # are contiguous in `ordered` by construction (single, then block,
    # then all), so each section's midpoint is computed directly from
    # its length and running offset rather than a lookup.
    ymax_data = ordered["iia_overall"].max()
    ax.set_ylim(0, ymax_data * 1.18)

    offset = 0
    for mode, sub in [("single", single_df), ("block", block_df), ("all", all_df)]:
        if len(sub) == 0:
            continue
        mid = offset + (len(sub) - 1) / 2
        ax.text(mid, ymax_data * 1.10, mode, ha="center", fontsize=10,
                color="#555555", style="italic")
        offset += len(sub)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = output_dir / f"sufficiency_curve_{release}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved: {p}")
    plt.close(fig)


def make_cross_release_iia_plot(
    df_a: pd.DataFrame, release_a: str,
    df_b: pd.DataFrame, release_b: str,
    output_dir: Path, display_names: bool = False,
    top_n: int = 8,
):
    """
    Paired bar comparison of single-token IIA across two releases,
    for the union of each release's top_n subgroups by IIA — showing
    whether the dominant causally-relied-upon subgroup's identity is
    stable across releases (see A2 sec 7 cross-release finding: the
    dominant subgroup shifted between G49 and G47 while the general
    sufficiency-curve shape replicated).
    """
    if "iia_overall" not in df_a.columns or "iia_overall" not in df_b.columns:
        print("  Missing iia_overall in one or both releases — skipping "
              "cross-release IIA plot")
        return

    top_a = set(df_a.sort_values("iia_overall", ascending=False).head(top_n)["subgroup"])
    top_b = set(df_b.sort_values("iia_overall", ascending=False).head(top_n)["subgroup"])
    subgroups = top_a | top_b

    a_indexed = df_a.set_index("subgroup")
    b_indexed = df_b.set_index("subgroup")
    block_map = {**a_indexed["block"].to_dict(), **b_indexed["block"].to_dict()}

    # Order by the larger of the two releases' IIA, descending
    def _max_iia(sg):
        va = a_indexed["iia_overall"].get(sg, 0.0)
        vb = b_indexed["iia_overall"].get(sg, 0.0)
        return max(va, vb)

    ordered_subgroups = sorted(subgroups, key=_max_iia, reverse=True)
    labels = [DISPLAY_NAMES.get(s, s) for s in ordered_subgroups] if display_names else ordered_subgroups

    vals_a = [a_indexed["iia_overall"].get(sg, 0.0) for sg in ordered_subgroups]
    vals_b = [b_indexed["iia_overall"].get(sg, 0.0) for sg in ordered_subgroups]

    x = np.arange(len(ordered_subgroups))
    width = 0.38
    fig, ax = plt.subplots(figsize=(0.7 * len(ordered_subgroups) + 2, 6.5))
    fig.patch.set_facecolor("white")

    edge_colors = [BLOCK_COLORS.get(block_map.get(sg), "#999999") for sg in ordered_subgroups]

    ax.bar(x - width/2, vals_a, width, label=f"GENCODE {release_a}",
           color="#4A90D9", edgecolor=edge_colors, linewidth=1.2)
    ax.bar(x + width/2, vals_b, width, label=f"GENCODE {release_b}",
           color="#F4B942", edgecolor=edge_colors, linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=45, ha="right")
    ax.set_ylabel("IIA (fraction of trials flipped by patching)", fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, color="#e1e0d9", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.set_title(
        f"Causal Layer: Cross-Release IIA — GENCODE {release_a} vs {release_b}\n"
        f"top {top_n} subgroups by IIA, either release",
        fontsize=12.5, fontweight="bold"
    )
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = output_dir / f"cross_release_iia_{release_a}_vs_{release_b}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
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
    p.add_argument("--top_n_single", type=int, default=6,
                   help="Number of single-scope subgroups to show in the "
                        "sufficiency curve (default: 6)")
    p.add_argument("--patching_csv2", default=None,
                   help="Optional second release's patching summary CSV — "
                        "if given (with --release2), also produces a "
                        "cross-release IIA comparison plot")
    p.add_argument("--release2", default=None)
    p.add_argument("--top_n_cross_release", type=int, default=8,
                   help="Number of top subgroups (union across both "
                        "releases) shown in the cross-release IIA plot "
                        "(default: 8)")
    args = p.parse_args()

    df = load_symmetry(Path(args.patching_csv))
    print(df.to_string(index=False))

    make_plot(
        df, args.release, Path(args.output_dir),
        display_names=args.display_names,
        threshold=args.threshold,
        annotate_top_n=args.annotate_top_n,
    )

    make_symmetry_vs_iia_plot(
        df, args.release, Path(args.output_dir),
        display_names=args.display_names,
        annotate_top_n=args.annotate_top_n,
    )

    suff_df = load_sufficiency_curve(Path(args.patching_csv))
    make_sufficiency_curve_plot(
        suff_df, args.release, Path(args.output_dir),
        display_names=args.display_names,
        top_n_single=args.top_n_single,
    )

    if args.patching_csv2 is not None:
        if args.release2 is None:
            raise SystemExit("--patching_csv2 given but --release2 is missing")
        df2 = load_symmetry(Path(args.patching_csv2))
        make_cross_release_iia_plot(
            df, args.release, df2, args.release2,
            Path(args.output_dir), display_names=args.display_names,
            top_n=args.top_n_cross_release,
        )


if __name__ == "__main__":
    main()