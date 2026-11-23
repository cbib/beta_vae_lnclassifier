"""Supplement figure `latent_probing`: what z encodes, redundancy and pattern alignment, from the cross-fold CSVs of 1.4.

A what z encodes: R2 of length and GC content, accuracy of the class probe (scored on its own training data)
B redundancy per subgroup: R2 from z to the token representation; * = q < 0.05 (Fisher-combined permutation test, BH-FDR)
C raw against residual pattern alignment per subgroup; * = q < 0.05 (paired t-test over folds, BH-FDR)
Points are the mean over folds, whiskers the standard deviation over folds.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from . import style
from .ablations import BLOCKS, group_labels, subgroup_order
from .interpretability_main import display_name

RAW = ("raw", "Raw tokens", "o", style.INK)
RESIDUAL = ("residual", "Residual tokens (z-orthogonal)", "^", "#a34c32")
CONFOUNDS = {"length": "Length", "gc_content": "GC content", "class (train acc)": "Class\n(training accuracy)"}
NOTE = "Length and GC content: R².\nClass: accuracy on the data the probe was fit on."


def probing_paths(exp_dir):
    """The five input CSVs of exp_dir (a beta_vae_subgroup_base_gNN folder)."""
    d = Path(exp_dir) / "interpretability" / "1.4_latent_probing"
    return {"probing": d / "cross_fold_latent_probing.csv", "redundancy": d / "cross_fold_redundancy.csv",
            "raw": d / "cross_fold_pattern_raw.csv", "residual": d / "cross_fold_pattern_residual.csv",
            "stats": d / "cross_fold_pattern_comparison_stats.csv"}


def read(path, cols):
    """The CSV with stripped lower-case headers; stops on a missing column."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    if set(cols) - set(df.columns):
        raise ValueError(f"{path}: lacks columns {sorted(set(cols) - set(df.columns))}")
    return df


def starred(df):
    """Subgroups with q < 0.05; none when the file has no q-values."""
    if "q_value_bh" not in df.columns:
        return set()
    return set(df.loc[df["q_value_bh"] < 0.05, "subgroup"])


def draw_probing(ax, probing):
    """Mean and sd of the three probes, regression and classification in two markers."""
    probing = probing.assign(k=probing["confound"].map(lambda c: list(CONFOUNDS).index(c) if c in CONFOUNDS else 99))
    probing = probing.sort_values("k", kind="stable").reset_index(drop=True)
    x = np.arange(len(probing))
    for i, row in enumerate(probing.itertuples()):
        cls = row.confound.startswith("class")
        ax.errorbar(i, row.mean_r2, row.std_r2, fmt="^" if cls else "o", color="#a34c32" if cls else style.INK,
                    markersize=3.2, elinewidth=0.5, capsize=0, markeredgewidth=0)
    ax.set_xticks(x, [CONFOUNDS.get(c, c) for c in probing["confound"]])
    ax.set_xlim(-0.6, len(probing) - 0.4)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R² / accuracy")


def draw_subgroups(ax, order, x, series, stars=None):
    """Mean and sd of each series (key, label, marker, colour, frame) at the subgroup positions."""
    spread = np.linspace(-0.15, 0.15, len(series)) if len(series) > 1 else [0.0]
    tops = []
    for off, (_, label, marker, color, f) in zip(spread, series):
        f = f.set_index("subgroup").loc[order["subgroup"]]
        ax.errorbar(x + off, f["mean"], f["sd"], fmt=marker, color=color, markersize=2.6, elinewidth=0.4,
                    capsize=0, markeredgewidth=0)
        tops.append((f["mean"] + f["sd"]).to_numpy())
    for xi, t, sg in zip(x, np.nanmax(tops, axis=0), order["subgroup"]):
        if stars and sg in stars:
            ax.annotate("*", (xi, t), xytext=(0, 1), textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")
    for b in BLOCKS[1:]:
        ax.axvline(x[(order["block"] == b).to_numpy()].min() - 1.0, color=style.HAIR, lw=0.6, zorder=0)
    ax.set_xlim(x.min() - 0.8, x.max() + 0.8)
    ax.set_xticks(x, [display_name(s) for s in order["subgroup"]], rotation=45, ha="right", rotation_mode="anchor")


def alignment_frame(df):
    """Rows (subgroup, mean, sd) of a cross-fold pattern CSV."""
    return df.rename(columns={"mean_align": "mean", "std_align": "sd"})[["subgroup", "mean", "sd"]]


def build(exp_dir, out_dir, captions=None):
    """Render the figure from exp_dir's CSVs into out_dir; return (inputs, outputs).

    captions: optional {figure name: (caption, alt text)}. A missing input stops the run."""
    paths = probing_paths(exp_dir)
    for p in paths.values():
        if not p.exists():
            raise FileNotFoundError(p)
    probing = read(paths["probing"], ["confound", "mean_r2", "std_r2"])
    red = read(paths["redundancy"], ["subgroup", "mean_r2", "std_r2"])
    raw = alignment_frame(read(paths["raw"], ["subgroup", "mean_align", "std_align"]))
    res = alignment_frame(read(paths["residual"], ["subgroup", "mean_align", "std_align"]))
    stats = read(paths["stats"], ["subgroup", "q_value_bh"])
    if not (set(red["subgroup"]) == set(raw["subgroup"]) == set(res["subgroup"])):
        raise ValueError("redundancy, raw and residual alignment cover different subgroups")
    order, x = subgroup_order(raw.dropna(subset=["mean"]))
    red = red.rename(columns={"mean_r2": "mean", "std_r2": "sd"})

    fig, axes = style.new_figure("double", 185, mosaic=[["A", "T"], ["B", "B"], ["C", "C"]],
                                 width_ratios=[1, 2], height_ratios=[0.6, 1, 1])
    draw_probing(axes["A"], probing)
    axes["T"].axis("off")
    axes["T"].text(0.04, 0.5, NOTE, fontsize=6, va="center", color=style.MUTED,
                   transform=axes["T"].transAxes).set_in_layout(False)
    draw_subgroups(axes["B"], order, x, [("red", "", "o", style.INK, red)])
    axes["B"].set_ylabel("Redundancy, R² (z → token)")
    axes["B"].set_ylim(0, 1.05)
    draw_subgroups(axes["C"], order, x, [RAW[:4] + (raw,), RESIDUAL[:4] + (res,)], starred(stats))
    axes["C"].axhline(0, color=style.MUTED, lw=0.5, zorder=0)
    axes["C"].set_ylim(top=axes["C"].get_ylim()[1] + 0.12)  # room for the stars below the legend
    axes["C"].set_ylabel("Pattern–weight alignment (cosine)")
    axes["C"].legend(handles=[Line2D([], [], ls="", marker=m, color=c, markersize=3.5, label=lab)
                              for _, lab, m, c in (RAW, RESIDUAL)],
                     loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=2, handletextpad=0.3,
                     columnspacing=1.5, borderaxespad=0.1)
    group_labels(axes["B"], order, x)
    for key in "ABC":
        style.panel_label(axes[key], key)
    outputs = style.save(fig, out_dir, "latent_probing", *(captions or {}).get("latent_probing", (None, None)))
    return list(paths.values()), outputs
