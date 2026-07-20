#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_hard_for_all_cpat.py

Publication-quality figure comparing CPAT features between hard-for-all
transcripts and the full test set background, split by biotype class
(mRNA and lncRNA) and release (v47, v49).

Usage
-----
# Both releases, portrait (original behaviour — features as rows, releases as columns)
python analysis/benchmark/plot_hard_for_all_cpat.py \
    --summary_csv    gencode_v47_experiments/benchmark_comparison/cpat_feature_summary_combined.csv \
    --cpat_v47       gencode_v47_experiments/benchmark_comparison/hard_for_all/hard_for_all_cpat.csv \
    --cpat_v49       gencode_v49_experiments/benchmark_comparison/hard_for_all/hard_for_all_cpat.csv \
    --cpat_full_v47  gencode_v47_experiments/benchmark_tools/cpat/predictions_with_cpat.csv \
    --cpat_full_v49  gencode_v49_experiments/benchmark_tools/cpat/predictions_with_cpat.csv \
    --output_dir     figures/

# Single release (v49 only), landscape — for a wide poster row
# (2 rows: mRNA / lncRNA  x  4 columns: one per CPAT feature)
python analysis/benchmark/plot_hard_for_all_cpat.py \
    --summary_csv    gencode_v49_experiments/benchmark_comparison/cpat_feature_summary_combined.csv \
    --cpat_v49       gencode_v49_experiments/benchmark_comparison/hard_for_all/hard_for_all_cpat.csv \
    --cpat_full_v49  gencode_v49_experiments/benchmark_tools/cpat/predictions_with_cpat.csv \
    --releases       v49 \
    --orientation    landscape \
    --output_dir     figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    "bg_mrna":    "#F1948A",
    "hard_mrna":  "#922B21",
    "bg_lnc":   "#85C1E9",
    "hard_lnc": "#1A5276",
}

FEATURE_ORDER = ["coding_prob", "orf_size", "fickett_score", "hexamer_score"]
FEATURE_META  = {
    "coding_prob":   dict(label="CPAT coding\nprobability", ylim=(-0.05, 1.15)),
    "orf_size":      dict(label="ORF size (nt)",             ylim=(-50,   3600)),
    "fickett_score": dict(label="Fickett score",             ylim=(0.25,  1.65)),
    "hexamer_score": dict(label="Hexamer score",             ylim=(-0.6,  0.95)),
}

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth":  0.7,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})

# ── Data loaders ──────────────────────────────────────────────────────────────

def norm_label(series):
    return series.astype(str).str.lower().str.strip().map(
        {"lnc":"lnc","lncrna":"lnc","pc":"mrna","pcrna":"mrna",
         "noncoding":"lnc","coding":"mrna"}
    )

def load_per_transcript(path, release):
    if not path or not Path(path).exists():
        return None
    df = pd.read_csv(path)
    df["transcript_id"] = df["transcript_id"].astype(str).str.split("|").str[0]
    df["true_label"] = norm_label(df["true_label"])
    df["release"] = release
    return df

def load_full_cpat(path, hard_ids):
    if not path or not Path(path).exists():
        return None
    df = pd.read_csv(path)
    df["transcript_id"] = df["transcript_id"].astype(str).str.split("|").str[0]
    df["true_label"] = norm_label(df["true_label"])
    df["is_hard_all"] = df["transcript_id"].isin(hard_ids)
    return df

# ── Drawing ───────────────────────────────────────────────────────────────────

def boxplot(ax, x, q25, med, q75, color, width=0.38, alpha=0.55,
            lw=0.8, zorder=2):
    """Simple box-and-whisker from quartiles."""
    iqr    = q75 - q25
    whislo = q25 - 1.5 * iqr
    whishi = q75 + 1.5 * iqr
    rect = mpatches.Rectangle(
        (x - width/2, q25), width, iqr,
        linewidth=0, edgecolor="none", facecolor=color,
        alpha=alpha, zorder=zorder,
    )
    ax.add_patch(rect)
    edge = mpatches.Rectangle(
        (x - width/2, q25), width, iqr,
        linewidth=0.8, edgecolor="black", facecolor="none",
        alpha=1.0, zorder=zorder+1,
    )
    ax.add_patch(edge)
    import matplotlib.colors as mc
    try:
        rgb = mc.to_rgb(color)
        dark = tuple(max(0, c * 0.4) for c in rgb)
    except Exception:
        dark = "black"
    ax.plot([x - width/2, x + width/2], [med, med],
            color=dark, lw=2.0, zorder=zorder+2, solid_capstyle="round")
    for y0, y1 in [(whislo, q25), (q75, whishi)]:
        ax.plot([x, x], [y0, y1], color=color, lw=lw,
                alpha=min(alpha+0.2, 1.0), zorder=zorder)
    for y in [whislo, whishi]:
        ax.plot([x-width/4, x+width/4], [y, y], color=color, lw=lw,
                alpha=min(alpha+0.2, 1.0), zorder=zorder)

def strip(ax, x, vals, color, width=0.20, size=2.8, alpha=0.75, zorder=4):
    rng = np.random.default_rng(0)
    jx  = x + rng.uniform(-width/2, width/2, size=len(vals))
    ax.scatter(jx, vals, s=size**2, color=color, alpha=alpha,
               linewidths=0, zorder=zorder)

def sig_bracket(ax, x1, x2, p, ylim, fontsize=7.5):
    if np.isnan(p) or p >= 0.05:
        return
    stars = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
    yrange = ylim[1] - ylim[0]
    bar_h  = yrange * 0.04
    bar_y  = ylim[0] + yrange * 0.88
    ax.plot([x1, x1, x2, x2],
            [bar_y - bar_h*0.3, bar_y, bar_y, bar_y - bar_h*0.3],
            lw=0.8, color="#333333", clip_on=True)
    ax.text((x1+x2)/2, bar_y + bar_h*0.1, stars,
            ha="center", va="bottom", fontsize=fontsize,
            color="#333333", clip_on=True)


def _draw_panel(ax, summary, per_trans, cpat_full, feature, release,
                 lbl, ylim, show_ylabel, show_title, title_text):
    """
    Draw one (feature, release, class) panel — bg + hard boxplots for a
    single biotype class, used by both orientations.
    """
    ax.set_xlim(0.4, 1.6)
    ax.set_ylim(ylim)

    sub = summary[
        (summary["feature"] == feature) &
        (summary["release"] == release) &
        (summary["true_label"] == lbl)
    ]
    if sub.empty:
        ax.set_xticks([])
        return

    row = sub.iloc[0]
    x_bg, x_hard = 0.75, 1.25
    c_bg, c_hard = C[f"bg_{lbl}"], C[f"hard_{lbl}"]

    boxplot(ax, x_bg, row["bg_q25"], row["bg_median"], row["bg_q75"],
            c_bg, alpha=0.50)

    pt = per_trans.get(release)
    cf = cpat_full.get(release)
    if pt is not None and feature in pt.columns:
        vals = pt[pt["true_label"] == lbl][feature].dropna().values
    elif cf is not None and feature in cf.columns:
        vals = cf[cf["is_hard_all"] & (cf["true_label"] == lbl)][feature].dropna().values
    else:
        vals = np.array([])

    if len(vals):
        q25, med, q75 = (np.percentile(vals, 25), np.median(vals), np.percentile(vals, 75))
        boxplot(ax, x_hard, q25, med, q75, c_hard, alpha=0.85)
        strip(ax, x_hard, vals, c_hard)
    else:
        boxplot(ax, x_hard, row["hard_q25"], row["hard_median"], row["hard_q75"],
                c_hard, alpha=0.85)

    sig_bracket(ax, x_bg, x_hard, row["p_mannwhitney"], ylim)

    n_hard = int(row["hard_n"])
    y_n = ylim[0] + (ylim[1]-ylim[0]) * 0.02
    ax.text(x_hard, y_n, f"n={n_hard}", ha="center", va="bottom",
            fontsize=6.0, color="#555555")

    ax.set_xticks([x_bg, x_hard])
    ax.set_xticklabels(["bg", "hard"], fontsize=7.5)
    ax.tick_params(axis="x", length=0, pad=6)

    if show_ylabel:
        ax.set_ylabel(FEATURE_META[feature]["label"], fontsize=8.5, labelpad=3)
    else:
        ax.set_yticklabels([])

    if show_title:
        ax.set_title(title_text, fontsize=9.5, fontweight="bold", pad=8)

    ax.tick_params(axis="y", labelsize=7.5)


# ── Figure — portrait (original): rows=features, cols=releases×class-pairs ──

def make_figure_portrait(summary, per_trans, cpat_full, releases, output_dir):
    n_feat = len(FEATURE_ORDER)
    n_rel  = len(releases)

    fig = plt.figure(figsize=(3.6 * max(n_rel, 1) + 1.0, 8.2))
    fig.patch.set_facecolor("white")
    gs = GridSpec(n_feat, n_rel, figure=fig,
                  hspace=0.62, wspace=0.30,
                  left=0.13, right=0.97, top=0.90, bottom=0.05)

    POS = {("mrna", "bg"): 1.00, ("mrna", "hard"): 1.90,
           ("lnc", "bg"): 3.30, ("lnc", "hard"): 4.20}
    XLIM = (0.35, 4.85)

    for ri, feature in enumerate(FEATURE_ORDER):
        meta = FEATURE_META[feature]
        ylim = meta["ylim"]

        for ci, release in enumerate(releases):
            ax = fig.add_subplot(gs[ri, ci])
            ax.set_xlim(XLIM)
            ax.set_ylim(ylim)

            sub = summary[(summary["feature"] == feature) &
                          (summary["release"] == release)]

            for _, row in sub.iterrows():
                lbl    = row["true_label"]
                x_bg   = POS[(lbl, "bg")]
                x_hard = POS[(lbl, "hard")]
                c_bg, c_hard = C[f"bg_{lbl}"], C[f"hard_{lbl}"]

                boxplot(ax, x_bg, row["bg_q25"], row["bg_median"], row["bg_q75"],
                        c_bg, alpha=0.50)

                pt = per_trans.get(release)
                cf = cpat_full.get(release)
                if pt is not None and feature in pt.columns:
                    vals = pt[pt["true_label"] == lbl][feature].dropna().values
                elif cf is not None and feature in cf.columns:
                    vals = cf[cf["is_hard_all"] & (cf["true_label"] == lbl)][feature].dropna().values
                else:
                    vals = np.array([])

                if len(vals):
                    q25, med, q75 = (np.percentile(vals, 25), np.median(vals), np.percentile(vals, 75))
                    boxplot(ax, x_hard, q25, med, q75, c_hard, alpha=0.85)
                    strip(ax, x_hard, vals, c_hard)
                else:
                    boxplot(ax, x_hard, row["hard_q25"], row["hard_median"], row["hard_q75"],
                            c_hard, alpha=0.85)

                sig_bracket(ax, x_bg, x_hard, row["p_mannwhitney"], ylim)
                n_hard = int(row["hard_n"])
                y_n = ylim[0] + (ylim[1]-ylim[0]) * 0.02
                ax.text(x_hard, y_n, f"n={n_hard}", ha="center", va="bottom",
                        fontsize=6.0, color="#555555")

            ax.axvline(2.6, color="#cccccc", lw=0.6, ls="--", zorder=0)
            ax.set_xticks([1.45, 3.75])
            ax.set_xticklabels(["mRNA", "lncRNA"], fontsize=8.5)
            ax.tick_params(axis="x", length=0, pad=10)

            if ci == 0:
                ax.set_ylabel(meta["label"], fontsize=8.5, labelpad=3)
            else:
                ax.set_yticklabels([])

            if ri == 0:
                ax.set_title(f"GENCODE {release}", fontsize=9.5,
                             fontweight="bold", pad=8)

            ax.tick_params(axis="y", labelsize=7.5)

    _add_legend_and_save(fig, output_dir, "cpat_hard_for_all_figure")


# ── Figure — landscape: rows=class (mRNA/lncRNA), cols=features ───────────────
# Intended for single-release, wide poster layouts.

def make_figure_landscape(summary, per_trans, cpat_full, release, output_dir):
    n_feat = len(FEATURE_ORDER)

    TARGET_RATIO = 14 / 6   # 2.333
    fig_width  = 2.6 * n_feat + 1.2
    fig_height = fig_width / TARGET_RATIO
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("white")
    gs = GridSpec(2, n_feat, figure=fig,
                  hspace=0.55, wspace=0.35,
                  left=0.09, right=0.98, top=0.85, bottom=0.08)

    classes = [("mrna", "mRNA"), ("lnc", "lncRNA")]

    for ri, (lbl, lbl_display) in enumerate(classes):
        for ci, feature in enumerate(FEATURE_ORDER):
            ax = fig.add_subplot(gs[ri, ci])
            _draw_panel(
                ax, summary, per_trans, cpat_full,
                feature=feature, release=release, lbl=lbl,
                ylim=FEATURE_META[feature]["ylim"],
                show_ylabel=(ci == 0),
                show_title=(ri == 0),
                title_text=FEATURE_META[feature]["label"].replace("\n", " "),
            )
        # Row label on the far left, vertically centered
        ax_first = fig.axes[ri * n_feat]
        ax_first.annotate(
            lbl_display, xy=(-0.55, 0.5), xycoords="axes fraction",
            fontsize=10, fontweight="bold", ha="center", va="center",
            rotation=90,
        )

    _add_legend_and_save(fig, output_dir, f"cpat_hard_for_all_figure_{release}_landscape")


def _add_legend_and_save(fig, output_dir, filename_stem):
    patches = [
        mpatches.Patch(facecolor=C["bg_mrna"],   alpha=0.55, label="Background mRNA"),
        mpatches.Patch(facecolor=C["hard_mrna"],  alpha=0.85, label="Hard-for-all mRNA"),
        mpatches.Patch(facecolor=C["bg_lnc"],  alpha=0.55, label="Background lncRNA"),
        mpatches.Patch(facecolor=C["hard_lnc"], alpha=0.85, label="Hard-for-all lncRNA"),
    ]
    fig.legend(handles=patches, loc="upper center",
               bbox_to_anchor=(0.5, 0.99), ncol=4,
               fontsize=7.5, frameon=False,
               handlelength=1.1, handletextpad=0.4, columnspacing=1.2)
    fig.suptitle("CPAT sequence features: hard-for-all vs background",
                 fontsize=10.5, fontweight="bold", y=1.03)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png"]:
        p = output_dir / f"{filename_stem}.{ext}"
        fig.savefig(p, dpi=2000, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {p}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv",   required=True)
    parser.add_argument("--cpat_v47",      default=None)
    parser.add_argument("--cpat_v49",      default=None)
    parser.add_argument("--cpat_full_v47", default=None)
    parser.add_argument("--cpat_full_v49", default=None)
    parser.add_argument("--output_dir",    required=True)
    parser.add_argument(
        "--releases", nargs="+", default=None,
        help="Which release(s) to plot, e.g. --releases v49. "
             "Default: all releases found in summary_csv."
    )
    parser.add_argument(
        "--orientation", choices=["portrait", "landscape"], default="portrait",
        help="portrait: rows=features, cols=releases (original layout). "
             "landscape: rows=class (mRNA/lncRNA), cols=features — "
             "for a single release in a wide poster row. "
             "landscape requires exactly one release (via --releases)."
    )
    args = parser.parse_args()

    summary = pd.read_csv(args.summary_csv)
    summary["true_label"] = summary["true_label"].astype(str).str.lower().str.strip().map(
        {"lnc": "lnc", "lncrna": "lnc", "pc": "mrna", "pcrna": "mrna",
        "noncoding": "lnc", "coding": "mrna", "mrna": "mrna"}
    )

    available_releases = sorted(summary["release"].unique())
    releases = args.releases if args.releases else available_releases
    for r in releases:
        if r not in available_releases:
            raise ValueError(f"Release {r!r} not found in summary_csv "
                             f"(available: {available_releases})")
    summary = summary[summary["release"].isin(releases)]

    pt_v47 = load_per_transcript(args.cpat_v47, "v47")
    pt_v49 = load_per_transcript(args.cpat_v49, "v49")
    ids_v47 = set(pt_v47["transcript_id"]) if pt_v47 is not None else set()
    ids_v49 = set(pt_v49["transcript_id"]) if pt_v49 is not None else set()
    cf_v47 = load_full_cpat(args.cpat_full_v47, ids_v47)
    cf_v49 = load_full_cpat(args.cpat_full_v49, ids_v49)

    per_trans = {"v47": pt_v47, "v49": pt_v49}
    cpat_full = {"v47": cf_v47, "v49": cf_v49}

    if args.orientation == "landscape":
        if len(releases) != 1:
            raise ValueError(
                f"--orientation landscape requires exactly one release "
                f"(got {releases}). Pass --releases v49 (or v47)."
            )
        make_figure_landscape(
            summary=summary, per_trans=per_trans, cpat_full=cpat_full,
            release=releases[0], output_dir=args.output_dir,
        )
    else:
        make_figure_portrait(
            summary=summary, per_trans=per_trans, cpat_full=cpat_full,
            releases=releases, output_dir=args.output_dir,
        )

    print("Done.")

if __name__ == "__main__":
    main()