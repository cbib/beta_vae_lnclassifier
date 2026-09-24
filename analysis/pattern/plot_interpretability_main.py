#!/usr/bin/env python3
"""
plot_interpretability_main.py

The consolidated MAIN-TEXT interpretability figure for β-LNC, given the
7-page limit. Three panels, one figure:

  (A) Three-layer scatter (unchanged from plot_interpretability_scatter.py):
        x    — log10(Fréchet distance)      [Layer 2: associational]
        y    — feature-zero ablation Δacc   [Layer 1: model-functional]
        size — patching symmetry score      [Layer 3: causal]
        color — block (NonB / REP / NonB2)
      The single-figure summary of all three layers, per-subgroup.

  (B) Single-subgroup symmetry score (bars, log) vs. IIA (line) — shows
      that even the strongest individual subgroups (SS_BINNED, TE_CORE)
      are causally insufficient alone (IIA far below what symmetry score
      would suggest).

  (C) Cross-fold greedy multi-token search: joint IIA vs. the additive-
      independence null as a function of subgroup-set size, mean ± std
      across folds — shows that a searched COMBINATION (anchored by
      SS_BINNED + TE_CORE, selected first in every fold) is synergistic,
      which (A) and (B) alone cannot show since both are single-subgroup
      views.

Together (A)+(B)+(C) carry the paper's full Layer 1/2/3 argument in one
figure. Supplementary should hold: the fixed block/all sufficiency curve,
shuffled-null control, cross-release comparison, concentration metrics,
z-redundancy, and full per-fold greedy search table.

Usage
-----
python analysis/pattern/plot_interpretability_main.py \\
    --frechet_csv          gencode_v49_experiments/mstar_analysis/frechet_ranking.csv \\
    --ablation_csv         gencode_v49_experiments/beta_vae_subgroup_base_g49/ablation_analysis/all_folds_ablation.csv \\
    --pattern_raw_csv      gencode_v49_experiments/beta_vae_subgroup_base_g49/latent_probing/cross_fold_pattern_raw.csv \\
    --pattern_residual_csv gencode_v49_experiments/beta_vae_subgroup_base_g49/latent_probing/cross_fold_pattern_residual.csv \\
    --patching_csv         gencode_v49_experiments/beta_vae_subgroup_base_g49/patching_v2/all_folds_patching.csv \\
    --greedy_all_folds     gencode_v49_experiments/beta_vae_subgroup_base_g49/joint_patching/joint_patching_greedy_all_folds.csv \\
    --output_dir           gencode_v49_experiments/benchmark_comparison \\
    --release              v49

Expected CSV schemas
--------------------
frechet_csv          : subgroup, block, frechet_dist, [...]
ablation_csv          : subgroup, block, mode, acc_drop, fold, ...
pattern_raw_csv       : subgroup, mean_align, std_align
pattern_residual_csv  : subgroup, mean_align, std_align
patching_csv          : either patching_summary.csv (subgroup,
                         symmetry_score, iia_overall, ...) or the raw
                         per-pair all_folds_patching.csv (patch_mode,
                         patch_scope, direction, delta_logit, iia_success,
                         iia_baseline_valid, ...) — auto-detected.
greedy_all_folds       : step, added_subgroup, current_set, set_size,
                         iia_joint, symmetry_joint, iia_additive_expected,
                         delta_vs_additive, fold  (joint_patching_search.py
                         output)

All CSVs must share the same subgroup key convention (internal TE_* keys).
Pass --display_names to remap TE_* → REP_* in figure labels only.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Cantarell", "DejaVu Sans"],
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "figure.dpi":       150,
})


# ── Shared constants ─────────────────────────────────────────────────────────

BLOCK_COLORS = {"nonb": "#2a78d6", "te": "#1baf7a", "nonb2": "#eda100"}
BLOCK_LABELS = {"nonb": "NonB", "te": "REP", "nonb2": "NonB2"}

DISPLAY_NAMES = {
    "TE_CORE":    "REP_CORE",
    "TE_LCTR":    "REP_LCTR",
    "TE_QUALITY": "REP_QUALITY",
    "TE_PSEUDO":  "REP_PSEUDO",
    "TE_UNKNOWN": "REP_UNKNOWN",
    "TE_GLOBAL":  "REP_GLOBAL",
}

def display_name(sg: str) -> str:
    """Always rename TE_* subgroup keys to REP_* for figure labels. Data
    stays keyed as TE_* internally (feature registry, CSVs); this is the
    only place the REP_ naming appears, applied unconditionally in every
    panel — there is no raw-TE display mode."""
    return DISPLAY_NAMES.get(sg, sg)

FD_DIVIDER  = 5.0
ABL_DIVIDER = 0.003

QUADRANT_LABELS = {
    "TL": "data signal\nnot used",
    "TR": "detected by\nboth layers",
    "BL": "neither layer",
    "BR": "model-dependent\n(not by Fréchet)",
}

LABEL_OFFSETS = {
    "SS_BINNED":    ( 0.08,  0.0005,  "left"),
    "TE_CORE":      ( 0.06,  0.0003,  "left"),
    "REP_CORE":     ( 0.06,  0.0003,  "left"),
    "IR":           (-0.07,  0.0003,  "right"),
    "TE_LCTR":      ( 0.06,  0.0002,  "left"),
    "TE_QUALITY":   ( 0.06, -0.0003,  "left"),
    "TE_PSEUDO":    ( 0.06,  0.0003,  "left"),
    "TE_UNKNOWN":   ( 0.06, -0.0004,  "left"),
    "TE_GLOBAL":    (-0.07,  0.0003,  "right"),
    "GQ":           ( 0.06,  0.0003,  "left"),
    "STR":          (-0.07,  0.0004,  "right"),
    "DR":           ( 0.06, -0.0002,  "left"),
    "MR":           ( 0.06,  0.00005, "left"),
    "TRI":          (-0.07,  0.0002,  "right"),
    "Z":            (-0.07, -0.0002,  "right"),
    "APR":          ( 0.06, -0.0003,  "left"),
    "GLOBAL":       (-0.08,  0.0003,  "right"),
    "SS_COUNT":     (-0.10, -0.00015, "right"),
    "SS_RELPOS":    ( 0.10,  0.00015, "left"),
    "RG4":          ( 0.07,  0.00010, "left"),
    "SS_STAB":      ( 0.07, -0.00010, "left"),
}

LABEL_MIN_ABL = 0.0


# ── Shared helpers ────────────────────────────────────────────────────────────

def norm_cols(df):
    df.columns = df.columns.str.strip().str.lower()
    if "subgroup" not in df.columns:
        df.rename(columns={df.columns[0]: "subgroup"}, inplace=True)
    return df


def resolve_block(subgroup: str) -> str:
    sg = str(subgroup).upper()
    if sg.startswith(("SS_", "RG4")):
        return "nonb2"
    if sg.startswith(("TE_", "REP_")):
        return "te"
    return "nonb"


def bubble_size(sym, scale=2800):
    return np.clip(sym, 0.005, 1.0) ** 0.9 * scale


# ── Panel A data: Fréchet / ablation / pattern / patching merge ──────────────

def load_pattern(path, suffix):
    df = norm_cols(pd.read_csv(path))
    align_col = "mean_align" if "mean_align" in df.columns else "alignment"
    std_col   = "std_align"  if "std_align"  in df.columns else None
    df = df.rename(columns={align_col: f"mean_align_{suffix}"})
    if std_col and std_col in df.columns:
        df = df.rename(columns={std_col: f"std_align_{suffix}"})
        return df[["subgroup", f"mean_align_{suffix}", f"std_align_{suffix}"]]
    return df[["subgroup", f"mean_align_{suffix}"]]


def normalize_patching_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts EITHER the pre-aggregated summary shape (subgroup, symmetry_score,
    iia_overall, ...) or the raw per-pair shape (patch_mode, patch_scope,
    direction, delta_logit, iia_success, iia_baseline_valid, ...) and returns
    a normalized (subgroup, symmetry_score, iia_overall) frame, single-mode
    only. Same logic as joint_patching_search.py's normalize_single_summary,
    duplicated here so this plotting script has no cross-script import
    dependency.

    """
    df = raw_df.copy()
    df.columns = df.columns.str.strip().str.lower()

    if "patch_mode" in df.columns:
        df = df[df["patch_mode"] == "single"].copy()

    if "subgroup" not in df.columns and "patch_scope" in df.columns:
        df = df.rename(columns={"patch_scope": "subgroup"})

    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        sys.exit(f"ERROR: patching_csv has duplicate columns after "
                 f"normalization: {dupes}. Check the raw file's header.")

    if "symmetry_score" in df.columns and "iia_overall" in df.columns:
        out = (df.groupby("subgroup", as_index=False)
                 .agg(symmetry_score=("symmetry_score", "first"),
                      iia_overall=("iia_overall", "first")))
        return out

    required = {"iia_success", "iia_baseline_valid", "delta_logit", "direction"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: patching_csv has neither ('symmetry_score','iia_overall') "
                 f"nor the raw columns needed to compute them (missing: {missing}). "
                 f"Columns found: {list(df.columns)}")

    df["iia_baseline_valid"] = df["iia_baseline_valid"].astype(bool)
    df["iia_success"]        = df["iia_success"].astype(bool)

    per_direction = (
        df[df["iia_baseline_valid"]]
        .groupby(["subgroup", "direction"], as_index=False)
        .agg(iia=("iia_success", "mean"))
    )
    delta_mean = (
        df.groupby(["subgroup", "direction"], as_index=False)
        .agg(mean_delta_logit=("delta_logit", "mean"))
    )
    merged = per_direction.merge(delta_mean, on=["subgroup", "direction"], how="outer")

    pivot_iia   = merged.pivot(index="subgroup", columns="direction", values="iia")
    pivot_delta = merged.pivot(index="subgroup", columns="direction", values="mean_delta_logit")

    out = pd.DataFrame({
        "subgroup": pivot_iia.index,
        "iia_overall": pivot_iia[["lnc_to_mrna", "mrna_to_lnc"]].mean(axis=1, skipna=True).values,
        "symmetry_score": (
            (pivot_delta["lnc_to_mrna"].abs().fillna(0)
             + pivot_delta["mrna_to_lnc"].abs().fillna(0)) / 2
        ).values,
    }).reset_index(drop=True)
    return out


def load_csvs(frechet_path, ablation_path, raw_path, residual_path, patching_path):
    fd  = norm_cols(pd.read_csv(frechet_path))
    abl = norm_cols(pd.read_csv(ablation_path))
    raw = load_pattern(raw_path,      "raw")
    res = load_pattern(residual_path, "residual")

    if "mode" in abl.columns:
        abl = abl[abl["mode"] == "feature_zero"]
    abl = (abl.groupby("subgroup", as_index=False)["acc_drop"]
              .agg(mean_acc_drop="mean", std_acc_drop="std"))

    merged = (fd
              .merge(abl, on="subgroup", how="inner")
              .merge(raw, on="subgroup", how="inner")
              .merge(res, on="subgroup", how="inner"))

    pat = normalize_patching_summary(pd.read_csv(patching_path))
    merged = merged.merge(pat, on="subgroup", how="left")
    for col in ["symmetry_score", "iia_overall"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)

    if len(merged) == 0:
        sys.exit("ERROR: no rows after merging CSVs")

    print(f"  Fréchet: {len(fd)}  Ablation: {len(abl)}  "
          f"Pattern: {len(res)}  Patching: {len(pat)}  →  Merged: {len(merged)}")
    return merged, pat


# ── Panel A: three-layer scatter ─────────────────────────────────────────────

def plot_panel_a(ax, df):
    log_fd = np.log10(np.maximum(df["frechet_dist"].values, 0.05))
    abl    = np.maximum(df["mean_acc_drop"].values, 0.0)
    sym    = df["symmetry_score"].values
    blocks = df["subgroup"].apply(resolve_block).values

    xdiv = np.log10(FD_DIVIDER)
    ax.axvline(xdiv,        color="#c8c7c0", lw=0.8, ls="--", zorder=0)
    ax.axhline(ABL_DIVIDER, color="#c8c7c0", lw=0.8, ls="--", zorder=0)

    for i in range(len(df)):
        c = BLOCK_COLORS[blocks[i]]
        ax.scatter(log_fd[i], abl[i], s=bubble_size(sym[i]), color=c,
                  alpha=0.65, linewidths=0.6, edgecolors="white", zorder=3)

    for i, row in df.iterrows():
        sg = row["subgroup"]
        iloc = df.index.get_loc(i)
        if abl[iloc] < LABEL_MIN_ABL and sym[iloc] < LABEL_MIN_ABL:
            continue
        label = display_name(sg)
        c = BLOCK_COLORS[blocks[iloc]]
        dx, dy, ha = LABEL_OFFSETS.get(sg, (0.04, 0.0002, "left"))
        x, y = log_fd[iloc], abl[iloc]
        ax.annotate(label, xy=(x, y), xytext=(x + dx, y + dy),
                   fontsize=7, color=c, ha=ha, va="center",
                   arrowprops=dict(arrowstyle="-", color=c, lw=0.5, alpha=0.6),
                   zorder=5)

    ax.autoscale()
    xl, xr = ax.get_xlim()
    ax.set_xlim(xl - 0.05, xr + 0.45)
    ax.set_ylim(bottom=-0.0004, top=0.028)
    yr = ax.get_ylim()[1]
    qkw = dict(fontsize=7, color="#b0afa8", va="top", linespacing=1.4)
    ax.text(xdiv - 0.05, yr * 0.97, QUADRANT_LABELS["TL"], ha="right", **qkw)
    ax.text(xdiv + 0.05, yr * 0.97, QUADRANT_LABELS["TR"], ha="left",  **qkw)
    ax.text(xdiv - 0.05, ABL_DIVIDER * 0.6, QUADRANT_LABELS["BL"], ha="right", **qkw)
    ax.text(xdiv + 0.05, ABL_DIVIDER * 0.6, QUADRANT_LABELS["BR"], ha="left",  **qkw)

    ax.set_yscale("symlog", linthresh=0.001, linscale=0.5)
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("Layer 2: log₁₀(Fréchet distance)", fontsize=9, fontweight="bold")
    ax.set_ylabel("Layer 1: Ablation Δacc (symlog)", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=8)
    xticks = ax.get_xticks()
    ax.set_xticklabels(
        [f"{t:.1f}\n(FD={10**t:.1f})" if t >= 0 else f"{t:.1f}" for t in xticks],
        fontsize=7)

    block_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BLOCK_COLORS[k],
              markersize=7, label=BLOCK_LABELS[k])
        for k in ("nonb", "te", "nonb2")
    ]
    size_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#aaa",
              markersize=np.sqrt(bubble_size(v) / np.pi) * 0.9,
              label=f"L3 symmetry = {v:.2f}", alpha=0.7)
        for v in (0.05, 0.3, 0.7)
    ]
    leg = ax.legend(handles=block_handles + size_handles, fontsize=6.5,
                    frameon=True, framealpha=0.92, edgecolor="#e1e0d9",
                    loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2,
                    handlelength=1.4, labelspacing=0.3, columnspacing=0.7)
    leg.get_frame().set_linewidth(0.5)

    ax.set_title("A", loc="left", fontsize=12, fontweight="bold")


# ── Panel B: single-subgroup symmetry vs IIA ─────────────────────────────────

def plot_panel_b(ax, patching_df):
    df = patching_df.sort_values("symmetry_score", ascending=False).reset_index(drop=True)
    df["block"] = df["subgroup"].apply(resolve_block)

    x = np.arange(len(df))
    colors = [BLOCK_COLORS[b] for b in df["block"]]

    ax2 = ax.twinx()
    ax.bar(x, df["symmetry_score"], color=colors, alpha=0.35, width=0.6, zorder=2)
    ax2.plot(x, df["iia_overall"], color="#1c1a17", marker="o", markersize=3.5,
             linewidth=1.3, zorder=3)

    labels = [display_name(sg) for sg in df["subgroup"]]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6.5)

    ax.set_yscale("log")
    ax.set_ylabel("Symmetry score (log)", fontsize=9)
    ax2.set_ylabel("IIA", fontsize=9, color="#1c1a17")
    ax2.set_ylim(bottom=0)
    ax2.tick_params(axis="y", colors="#1c1a17", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)

    ax.set_title("B", loc="left", fontsize=12, fontweight="bold")

    iia_handle = Line2D([0], [0], color="#1c1a17", marker="o", markersize=3.5,
                        linewidth=1.3, label="IIA")
    sym_handle = plt.Rectangle((0, 0), 1, 1, color="#8a8478", alpha=0.35,
                               label="Symmetry")
    ax.legend(handles=[sym_handle, iia_handle], fontsize=7, frameon=True,
             loc="upper right", handlelength=1.2, labelspacing=0.3)

    for i, row in df.head(2).iterrows():
        ax2.annotate(f"{row['iia_overall']:.3f}",
                    xy=(i, row["iia_overall"]),
                    xytext=(0, 8), textcoords="offset points",
                    fontsize=7, color="#1c1a17", ha="center")


# ── Panel C: cross-fold greedy synergy ───────────────────────────────────────

def plot_panel_c(ax, greedy_all: pd.DataFrame):
    summary = (
        greedy_all.groupby("step")
        .agg(mean_iia_joint=("iia_joint", "mean"),
             std_iia_joint=("iia_joint", "std"),
             mean_iia_additive=("iia_additive_expected", "mean"),
             std_iia_additive=("iia_additive_expected", "std"))
        .reset_index()
    )
    steps = summary["step"].values

    ax.errorbar(steps, summary["mean_iia_joint"], yerr=summary["std_iia_joint"],
               color="#a34c32", marker="o", markersize=4.5, linewidth=1.6,
               capsize=2.5, label="Joint IIA (searched)", zorder=3)
    ax.errorbar(steps, summary["mean_iia_additive"], yerr=summary["std_iia_additive"],
               color="#8a8478", marker="s", markersize=4.5, linewidth=1.2,
               linestyle="--", capsize=2.5, label="Additive null", zorder=2)
    ax.fill_between(steps, summary["mean_iia_joint"], summary["mean_iia_additive"],
                    where=(summary["mean_iia_joint"] >= summary["mean_iia_additive"]),
                    color="#a34c32", alpha=0.12, zorder=1)

    ax.set_xlabel("Subgroup set size", fontsize=9)
    ax.set_ylabel("IIA", fontsize=9)
    ax.set_xticks(steps)
    ax.tick_params(labelsize=7)
    ax.set_title("C", loc="left", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, frameon=True, loc="upper left")

    last = summary.iloc[-1]
    gap = last["mean_iia_joint"] - last["mean_iia_additive"]
    ax.annotate(f"+{gap:.3f}",
               xy=(last["step"], (last["mean_iia_joint"] + last["mean_iia_additive"]) / 2),
               xytext=(last["step"] + 0.15, (last["mean_iia_joint"] + last["mean_iia_additive"]) / 2),
               fontsize=7.5, color="#a34c32", fontweight="bold", va="center")
    anchor_label = " + ".join(display_name(sg) for sg in ("SS_BINNED", "TE_CORE"))
    ax.annotate(f"{anchor_label}\nselected 1st/2nd, every fold",
               xy=(2, summary.loc[summary["step"] == 2, "mean_iia_joint"].values[0]),
               xytext=(3.1, 0.335),
               fontsize=6.5, color="#a34c32", ha="left", va="top",
               arrowprops=dict(arrowstyle="-", color="#a34c32", lw=0.5,
                                connectionstyle="arc3,rad=0.2"))


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frechet_csv",          required=True)
    p.add_argument("--ablation_csv",         required=True)
    p.add_argument("--pattern_raw_csv",      required=True)
    p.add_argument("--pattern_residual_csv", required=True)
    p.add_argument("--patching_csv",         required=True,
                   help="patching_summary.csv or all_folds_patching.csv "
                        "(single-mode; either shape auto-detected)")
    p.add_argument("--greedy_all_folds",     required=True,
                   help="joint_patching_greedy_all_folds.csv")
    p.add_argument("--output_dir",           required=True)
    p.add_argument("--release",              default="v49")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, patching_df = load_csvs(
        args.frechet_csv, args.ablation_csv,
        args.pattern_raw_csv, args.pattern_residual_csv,
        args.patching_csv,
    )
    greedy_all = pd.read_csv(args.greedy_all_folds)

    print(f"  Subgroups: {sorted(df['subgroup'].tolist())}")
    print(f"  Greedy search: {greedy_all['fold'].nunique()} folds, "
          f"max step {greedy_all['step'].max()}")

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.suptitle(
        f"β-LNC GENCODE {args.release} — three-layer interpretability: "
        f"per-subgroup summary, single-subgroup insufficiency, and searched synergy",
        fontsize=11.5, fontweight="bold", y=1.04,
    )

    plot_panel_a(ax_a, df)
    plot_panel_b(ax_b, patching_df)
    plot_panel_c(ax_c, greedy_all)

    plt.tight_layout()

    for ext in (".pdf", ".png"):
        p = out_dir / f"interpretability_main_{args.release}{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=300)
        print(f"Saved: {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()