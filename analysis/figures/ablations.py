"""Supplement figure `ablations`: the three ablation views of layer 1, from the CSVs of 1.2, 1.3 and 1.2b.

A subgroup ablation: feature_zero and token_zero
B feature_zero against residual_zero; paired t-test over folds, BH-FDR, * = q < 0.05 (descriptive: folds share transcripts)
C input-level ablation: zero, mean, permute
Points are the mean accuracy drop over folds, whiskers the standard deviation over folds.
Also writes ablations_stats.csv.
"""
from pathlib import Path

import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

from . import style
from .interpretability_main import BLOCK_LABELS, display_name, resolve_block

FEATURE = ("feature_zero", "Feature zero", "o", style.INK)
TOKEN = ("token_zero", "Token zero", "s", "#4a6fa5")
RESIDUAL = ("residual_zero", "Residual zero", "^", "#a34c32")
INPUT = (("zero", "Zero", "o", style.INK), ("mean", "Mean", "s", "#4a6fa5"), ("permute", "Permute", "D", "#8a8478"))
PANELS = {"A": (FEATURE, TOKEN), "B": (FEATURE, RESIDUAL), "C": INPUT}
BLOCKS = ("nonb", "te", "nonb2")
LINTHRESH = 0.001  # symlog, as in main panel A


def ablation_paths(exp_dir):
    """The three input CSVs of exp_dir (a beta_vae_subgroup_base_gNN folder)."""
    out = Path(exp_dir) / "interpretability"
    return {"subgroup": out / "1.2_ablation" / "all_folds_ablation.csv",
            "residual": out / "1.3_residual_ablation" / "all_folds_residual_ablation.csv",
            "input": out / "1.2b_input_ablation" / "all_folds_input_ablation.csv"}


def load_drops(path, modes):
    """Rows (fold, subgroup, mode, acc_drop) of the given modes; stops on a missing column or mode."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    need = {"fold", "subgroup", "mode", "acc_drop"}
    if need - set(df.columns):
        raise ValueError(f"{path}: lacks columns {sorted(need - set(df.columns))}")
    missing = [m for m in modes if m not in set(df["mode"])]
    if missing:
        raise ValueError(f"{path}: lacks modes {missing}")
    return df[df["mode"].isin(modes)][["fold", "subgroup", "mode", "acc_drop"]].reset_index(drop=True)


def check_same(frames):
    """Stops unless every frame covers the same subgroups and folds."""
    (ref, first), *rest = frames
    for name, f in rest:
        for col in ("subgroup", "fold"):
            a, b = set(first[col]), set(f[col])
            if a != b:
                raise ValueError(f"{name} and {ref} differ in {col}: {sorted(a ^ b)}")


def summarize(df):
    """Per (mode, subgroup): mean, sd over folds, number of folds."""
    g = df.groupby(["mode", "subgroup"])["acc_drop"]
    return pd.concat([g.mean().rename("mean"), g.std().fillna(0).rename("sd"), g.count().rename("n_folds")],
                     axis=1).reset_index()


def paired_test(a, b):
    """Per subgroup: paired t-test of a against b over folds (rows fold, subgroup, acc_drop), BH-FDR q."""
    pa = a.pivot(index="subgroup", columns="fold", values="acc_drop")
    pb = b.pivot(index="subgroup", columns="fold", values="acc_drop").reindex(pa.index)
    p = pd.Series({sg: stats.ttest_rel(pa.loc[sg], pb.loc[sg]).pvalue if pa.shape[1] > 1 else np.nan
                   for sg in pa.index}, name="p_value")
    q = pd.Series(np.nan, index=p.index, name="q_value_bh")
    ok = p.notna()
    if ok.any():
        q[ok] = stats.false_discovery_control(p[ok].to_numpy(), method="bh")
    return pd.concat([p, q], axis=1).rename_axis("subgroup").reset_index()


def subgroup_order(feature):
    """Subgroups by block (NonB, REP, NonB2), then by descending feature_zero drop; with the x positions."""
    m = feature.assign(block=lambda d: d["subgroup"].map(resolve_block))
    order = pd.concat([m[m["block"] == b].sort_values("mean", ascending=False) for b in BLOCKS], ignore_index=True)
    shift = order["block"].map({b: i for i, b in enumerate(BLOCKS)}).to_numpy()
    return order[["subgroup", "block"]], np.arange(len(order)) + shift


def draw_panel(ax, summary, modes, order, x, starred=None):
    """Mean and sd of each mode at the subgroup positions; starred: subgroups marked * in B."""
    spread = np.linspace(-0.2, 0.2, len(modes))
    tops = {}
    for off, (key, label, marker, color) in zip(spread, modes):
        s = summary[summary["mode"] == key].set_index("subgroup").loc[order["subgroup"]]
        ax.errorbar(x + off, s["mean"], s["sd"], fmt=marker, color=color, markersize=2.6, elinewidth=0.4,
                    capsize=0, markeredgewidth=0, zorder=3 if key == "feature_zero" else 2)
        tops[key] = (s["mean"] + s["sd"]).to_numpy()
    if starred is not None:
        top = np.max(list(tops.values()), axis=0)
        for xi, t, sg in zip(x, top, order["subgroup"]):
            if sg in starred:
                ax.annotate("*", (xi, t), xytext=(0, 1), textcoords="offset points", ha="center", va="bottom",
                            fontsize=8, fontweight="bold")
    ax.axhline(0, color=style.MUTED, lw=0.5, zorder=0)
    for b in BLOCKS[1:]:
        ax.axvline(x[(order["block"] == b).to_numpy()].min() - 1.0, color=style.HAIR, lw=0.6, zorder=0)
    ax.set_yscale("symlog", linthresh=LINTHRESH, linscale=0.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(x.min() - 0.8, x.max() + 0.8)
    ax.legend(handles=[Line2D([], [], ls="", marker=m, color=c, markersize=3.5, label=lab) for _, lab, m, c in modes],
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=len(modes), handletextpad=0.3, columnspacing=1.5,
              borderaxespad=0.1)


def group_labels(ax, order, x):
    """Block names along the top of ax."""
    for b in BLOCKS:
        xs = x[(order["block"] == b).to_numpy()]
        ax.text((xs.min() + xs.max()) / 2, 0.98, BLOCK_LABELS[b], transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6, color=style.MUTED)


def stats_table(panel, summary, test=None):
    """Rows of ablations_stats.csv for one panel; the p and q of B ride on the residual_zero rows."""
    t = summary.assign(panel=panel, mean_acc_drop=summary["mean"], sd_acc_drop=summary["sd"])
    t = t[["panel", "mode", "subgroup", "mean_acc_drop", "sd_acc_drop", "n_folds"]]
    if test is not None:
        t = t.merge(test.assign(mode="residual_zero"), on=["subgroup", "mode"], how="left")
    return t


def build(exp_dir, out_dir, captions=None):
    """Render the figure from exp_dir's CSVs into out_dir; return (inputs, outputs).

    captions: optional {figure name: (caption, alt text)}. A missing input stops the run."""
    paths = ablation_paths(exp_dir)
    for p in paths.values():
        if not p.exists():
            raise FileNotFoundError(p)
    sub = load_drops(paths["subgroup"], [FEATURE[0], TOKEN[0]])
    res = load_drops(paths["residual"], [RESIDUAL[0]])
    inp = load_drops(paths["input"], [m[0] for m in INPUT])
    check_same([("1.2", sub), ("1.3", res), ("1.2b", inp)])
    fz = sub[sub["mode"] == FEATURE[0]]
    summaries = {"A": summarize(sub), "B": summarize(pd.concat([fz, res])), "C": summarize(inp)}
    test = paired_test(fz, res)
    order, x = subgroup_order(summaries["A"][summaries["A"]["mode"] == FEATURE[0]])
    starred = set(test.loc[test["q_value_bh"] < 0.05, "subgroup"])

    fig, axes = style.new_figure("double", 185, mosaic=[["A"], ["B"], ["C"]])
    ylabels = {"A": "Accuracy drop", "B": "Accuracy drop", "C": "Accuracy drop"}
    for key, ax in axes.items():
        draw_panel(ax, summaries[key], PANELS[key], order, x, starred if key == "B" else None)
        ax.set_ylabel(ylabels[key] + " (symlog)")
        style.panel_label(ax, key)
    axes["B"].sharex(axes["A"])
    axes["C"].sharex(axes["A"])
    group_labels(axes["A"], order, x)
    names = [display_name(s) for s in order["subgroup"]]
    for ax in axes.values():
        ax.set_xticks(x, names, rotation=45, ha="right", rotation_mode="anchor")
    outputs = style.save(fig, out_dir, "ablations", *(captions or {}).get("ablations", (None, None)))

    table = pd.concat([stats_table("A", summaries["A"]), stats_table("B", summaries["B"], test),
                       stats_table("C", summaries["C"])], ignore_index=True)
    csv = Path(out_dir) / "ablations_stats.csv"
    table.to_csv(csv, index=False)
    return list(paths.values()), outputs + [csv]
