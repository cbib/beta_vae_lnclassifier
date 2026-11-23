"""Supplement figures for layer 2, from frechet_analysis/ and the latent-probing pattern CSV.

frechet: Fréchet distance against pattern alignment
  A  Fréchet distance, split into mean term and covariance term
  B  raw pattern alignment (mean and std over folds); * marks a subgroup in the top 40% by
     Fréchet distance whose mean alignment is below 0.1 in absolute value
  Subgroups are sorted by Fréchet distance; the squares under B give the feature block.
frechet_within_class: within-class structure behind the covariance term
  A, B  covariance across subgroups of the z-scored subgroup means, lncRNA and mRNA
  C     within-class standard deviation per subgroup and class
"""
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import style
from .interpretability_main import BLOCK_COLORS, BLOCK_LABELS, display_name, input_paths, resolve_block

MEAN_COLOR, COV_COLOR, ALIGN_COLOR = "#68737f", "#a34c32", "#b9c0c7"
FD_QUANTILE, ALIGN_FLOOR = 0.6, 0.1  # flag: FD above this quantile and |mean_align| below the floor
DIVERGING = LinearSegmentedColormap.from_list("cov", ["#2a5d8a", "#f2f2f2", "#b4532a"])
CLASS_COLORS = {"lncRNA": "#3a4652", "mRNA": "#aab4be"}


def load(frechet_csv, pattern_csv):
    """Fréchet table merged with pattern alignment, sorted by Fréchet distance (descending).

    Stops when any Fréchet computation failed (fd_status is not "ok")."""
    fd = pd.read_csv(frechet_csv)
    bad = fd.loc[fd["fd_status"] != "ok", "subgroup"].tolist()
    if bad:
        raise ValueError(f"{frechet_csv}: Fréchet distance failed for {', '.join(bad)}")
    pat = pd.read_csv(pattern_csv)[["subgroup", "mean_align", "std_align"]]
    df = fd.merge(pat, on="subgroup", how="left")
    if df["mean_align"].isna().any():
        raise ValueError(f"{pattern_csv}: no pattern alignment for "
                         f"{', '.join(df.loc[df['mean_align'].isna(), 'subgroup'])}")
    return df.sort_values("frechet_dist", ascending=False).reset_index(drop=True)


def flagged(df):
    """Boolean mask: high Fréchet distance but low pattern alignment."""
    return (df["frechet_dist"] > df["frechet_dist"].quantile(FD_QUANTILE)) & \
           (df["mean_align"].abs() < ALIGN_FLOOR)


def draw(ax_a, ax_b, df):
    x = np.arange(len(df))
    ax_a.bar(x, df["mean_term"], width=0.7, color=MEAN_COLOR, label="Mean term")
    ax_a.bar(x, df["cov_term"], width=0.7, bottom=df["mean_term"], color=COV_COLOR,
             label="Covariance term")
    ax_a.set_ylabel("Fréchet distance")
    ax_a.legend(loc="upper right", handlelength=1, ncols=2)
    ax_b.bar(x, df["mean_align"], width=0.7, color=ALIGN_COLOR)
    ax_b.errorbar(x, df["mean_align"], yerr=df["std_align"], fmt="none", ecolor=style.INK,
                  elinewidth=0.5, capsize=1.5, capthick=0.5)
    ax_b.axhline(0, color=style.INK, linewidth=0.6)
    ax_b.set_ylabel("Pattern alignment\n(cosine, raw)")
    for xi, row in zip(x, df.itertuples()):
        if flagged(df).iloc[xi]:
            ax_b.text(xi, row.mean_align + row.std_align, "*", ha="center", va="bottom",
                      fontsize=9, color=style.ACCENT, fontweight="bold")
    blocks = [resolve_block(s) for s in df["subgroup"]]
    ax_b.scatter(x, np.full(len(x), -0.075), s=14, marker="s", c=[BLOCK_COLORS[b] for b in blocks],
                 transform=ax_b.get_xaxis_transform(), clip_on=False, linewidths=0)
    ax_b.set_xticks(x, [display_name(s) for s in df["subgroup"]], rotation=45, ha="right",
                    rotation_mode="anchor")
    ax_b.tick_params(axis="x", pad=8)
    ax_b.set_xlim(-0.7, len(df) - 0.3)
    ax_b.set_ylim(top=1.4 * (df["mean_align"] + df["std_align"]).max())  # headroom for the legend
    handles = [Patch(color=BLOCK_COLORS[b], label=BLOCK_LABELS[b]) for b in dict.fromkeys(blocks)]
    handles.append(Line2D([], [], ls="", marker="$*$", color=style.ACCENT, label="high Fréchet, low alignment"))
    ax_b.legend(handles=handles, loc="upper right", handlelength=1, ncols=len(handles))


def load_covariance(frechet_dir):
    """(lncRNA, mRNA) covariance DataFrames from frechet_dir; the two must share their subgroups."""
    cov = [pd.read_csv(Path(frechet_dir) / f"within_class_covariance_{c}.csv", index_col=0)
           for c in ("lncrna", "mrna")]
    if list(cov[0].index) != list(cov[1].index):
        raise ValueError(f"{frechet_dir}: the two covariance tables have different subgroups")
    return cov


def draw_within_class(axes, cov):
    lnc, mrna = cov
    names = list(lnc.index)
    vmax = max(np.abs(lnc.values).max(), np.abs(mrna.values).max())
    for key, mat, title in (("A", lnc, "lncRNA"), ("B", mrna, "mRNA")):
        ax = axes[key]
        im = ax.imshow(mat.values, cmap=DIVERGING, vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=6.5, pad=3)
        ax.set_xticks(range(len(names)), [display_name(n) for n in names], rotation=90, fontsize=5)
        ax.set_yticks(range(len(names)), [display_name(n) for n in names] if key == "A" else [],
                      fontsize=5)
        ax.tick_params(length=1.5)
        ax.spines[:].set_visible(False)
        blocks = [resolve_block(n) for n in names]
        for k in range(1, len(names)):
            if blocks[k] != blocks[k - 1]:
                ax.axhline(k - 0.5, color="white", linewidth=0.8)
                ax.axvline(k - 0.5, color="white", linewidth=0.8)
    cbar = axes["A"].figure.colorbar(im, ax=[axes["A"], axes["B"]], shrink=0.6, aspect=25, pad=0.02)
    cbar.set_label("Covariance (z-scored units)")
    cbar.outline.set_visible(False)
    ax = axes["C"]
    x, w = np.arange(len(names)), 0.38
    for off, (label, mat) in zip((-w / 2, w / 2), (("lncRNA", lnc), ("mRNA", mrna))):
        ax.bar(x + off, np.sqrt(np.diag(mat.values)), width=w, color=CLASS_COLORS[label], label=label)
    ax.axhline(1.0, color=style.MUTED, linewidth=0.5, linestyle=(0, (3, 2)))
    ax.set_ylabel("Within-class std")
    ax.set_xticks(x, [display_name(n) for n in names], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.7, len(names) - 0.3)
    ax.legend(loc="upper right", handlelength=1, ncols=2)


def build(exp_dir, out_dir, captions=None):
    """Render both figures from exp_dir's CSVs into out_dir; return (inputs, outputs).

    captions: optional {figure name: (caption, alt text)}. A missing input stops the run."""
    paths = input_paths(exp_dir)
    frechet_dir = paths["frechet"].parent
    cov_paths = [frechet_dir / f"within_class_covariance_{c}.csv" for c in ("lncrna", "mrna")]
    inputs = [paths["frechet"], paths["pattern_raw"], *cov_paths]
    for p in inputs:
        if not p.exists():
            raise FileNotFoundError(p)
    cap = captions or {}
    df, cov = load(*inputs[:2]), load_covariance(frechet_dir)

    fig, (ax_a, ax_b) = style.new_figure("double", 90, nrows=2, sharex=True,
                                         gridspec_kw={"height_ratios": [1.2, 1]})
    draw(ax_a, ax_b, df)
    style.panel_label(ax_a, "A")
    style.panel_label(ax_b, "B")
    outputs = style.save(fig, out_dir, "frechet", *cap.get("frechet", (None, None)))

    fig, axes = style.new_figure("double", 125, mosaic=[["A", "B"], ["C", "C"]],
                                 height_ratios=[1.7, 1])
    draw_within_class(axes, cov)
    for key in "ABC":
        style.panel_label(axes[key], key)
    outputs += style.save(fig, out_dir, "frechet_within_class",
                          *cap.get("frechet_within_class", (None, None)))
    return inputs, outputs
