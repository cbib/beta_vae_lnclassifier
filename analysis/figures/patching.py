"""Supplement figures for layer 3, from the patching CSVs of 3.1, 3.2 and 3.4.

patching_sufficiency: IIA of matched patching, single subgroup -> block -> all tokens
patching_controls: matched patching against its three controls
  A symmetry, B IIA, C two-way swap effect, D IIA in both directions (each per scope),
  E greedy joint sets (IIA in both directions)
  Controls: shuffled partner (random opposite-class transcript), same-class partner
  (its IIA is the spurious flip rate), norm-matched noise vectors.
All points are the mean over folds; bars and whiskers are the standard deviation over folds.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from . import style
from .interpretability_main import BLOCK_COLORS, BLOCK_LABELS, display_name, resolve_block

MATCHED = ("matched", "Matched pairs", "o", style.INK)
CONTROLS = (("shuffled", "Shuffled partner", "^", "#a34c32"),
            ("sameclass", "Same-class partner", "s", "#4a6fa5"),
            ("noise", "Norm-matched noise", "D", "#8a8478"))
NOTE = ("Symmetry: mean over directions of |mean shift in logit|.\n"
        "Two-way swap effect: the smaller of the two shifts toward the\n"
        "swapped class (0 if either direction moves away).\n"
        "IIA, both directions: the smaller of the two flip rates.")
ALL_COLOR = "#68737f"
TOP_SINGLE = 6  # sufficiency curve: strongest single subgroups shown


def patching_paths(exp_dir):
    """The three input CSVs of exp_dir (a beta_vae_subgroup_base_gNN folder)."""
    out = Path(exp_dir) / "interpretability"
    return {"matched": out / "3.1_patching" / "all_folds_patching.csv",
            "controls": out / "3.2_patching_controls" / "all_folds_patching.csv",
            "joint": out / "3.4_joint_controls" / "joint_patching_controls_all_folds.csv"}


def load_scopes(path):
    """One row per (patch_mode, patch_scope): four scores, each mean and std over folds.

    Per fold and direction: mean delta_logit (d) and the flip rate among valid baselines.
    sym  = mean over directions of |d| (the symmetry score of the main figure)
    iia  = mean over directions of the flip rate (the IIA of the main figure)
    swap = min(d_lnc_to_mrna, -d_mrna_to_lnc), 0 if negative: the shift toward the swapped class
           in both directions
    iia_both = min over directions of the flip rate"""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    sites = set(df["patch_site"].dropna().unique()) if "patch_site" in df.columns else set()
    if sites != {"projector"}:
        raise ValueError(f"{path}: patch_site is {sorted(sites) or 'missing'}, expected projector")
    keys = ["patch_mode", "patch_scope", "fold", "direction"]
    valid = df[df["iia_baseline_valid"].astype(bool)].assign(ok=lambda d: d["iia_success"].astype(float))
    w = pd.concat([df.groupby(keys)["delta_logit"].mean().rename("d"),
                   valid.groupby(keys)["ok"].mean().rename("iia")], axis=1).unstack("direction")
    d, iia = w["d"], w["iia"]
    fold = pd.DataFrame({"sym": d.abs().mean(axis=1), "iia": iia.mean(axis=1),
                         "swap": np.minimum(d["lnc_to_mrna"], -d["mrna_to_lnc"]).clip(lower=0),
                         "iia_both": iia.min(axis=1, skipna=False)}).reset_index()
    g = fold.groupby(["patch_mode", "patch_scope"])[["sym", "iia", "swap", "iia_both"]]
    out = g.mean().join(g.std().add_suffix("_sd").fillna(0)).reset_index()
    block = df.groupby(["patch_mode", "patch_scope"])["block"].first().reset_index()
    return out.merge(block, on=["patch_mode", "patch_scope"])


def scope_block(row):
    """Block key of a scope row: a block scope is its own block, ALL has none."""
    if row["patch_mode"] == "all":
        return "all"
    return row["patch_scope"] if row["patch_mode"] == "block" else resolve_block(row["patch_scope"])


def draw_sufficiency(ax, m):
    """Bars of matched IIA: top single subgroups, the blocks, then all tokens."""
    single = m[m["patch_mode"] == "single"].sort_values("sym", ascending=False).head(TOP_SINGLE)
    block = m[m["patch_mode"] == "block"].sort_values("sym", ascending=False)
    rows = pd.concat([single, block, m[m["patch_mode"] == "all"]], ignore_index=True)
    x = np.arange(len(rows)) + np.r_[np.zeros(len(single)), np.ones(len(block)), 2 * np.ones(len(rows) - len(single) - len(block))]
    colors = [ALL_COLOR if k == "all" else BLOCK_COLORS[k] for k in rows.apply(scope_block, axis=1)]
    ax.bar(x, rows["iia"], width=0.7, color=colors)
    ax.errorbar(x, rows["iia"], rows["iia_sd"], fmt="none", ecolor=style.INK, elinewidth=0.5,
                capsize=1.5, capthick=0.5)
    names = [display_name(s) if mode == "single" else (BLOCK_LABELS[s] if mode == "block" else "All")
             for s, mode in zip(rows["patch_scope"], rows["patch_mode"])]
    ax.set_xticks(x, names, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("IIA")
    top = (rows["iia"] + rows["iia_sd"]).max() * 1.25
    ax.set_ylim(0, top)
    for lo, hi, label in ((x[0], x[len(single) - 1], "single subgroup"),
                          (x[len(single)], x[len(single) + len(block) - 1], "block"),
                          (x[-1], x[-1], "all")):
        ax.text((lo + hi) / 2, top * 0.97, label, ha="center", va="top", fontsize=6, color=style.MUTED)
    for cut in (x[len(single)] - 1.0, x[-1] - 1.0):
        ax.axvline(cut, color=style.HAIR, lw=0.6, zorder=0)


def scope_order(m):
    """Matched scopes: single subgroups, blocks, then all, each by descending symmetry."""
    parts = [m[m["patch_mode"] == k].sort_values("sym", ascending=False) for k in ("single", "block", "all")]
    return pd.concat(parts, ignore_index=True)


def control_frame(matched, controls, mode):
    """Rows of one control mode aligned to the matched scope order (stops on a missing scope)."""
    sub = controls[controls["patch_mode"] == mode].set_index("patch_scope")
    order = scope_order(matched)
    missing = [s for s in order["patch_scope"] if s not in sub.index]
    if missing:
        raise ValueError(f"control {mode!r} lacks scopes {missing}")
    return sub.loc[order["patch_scope"]].reset_index()


def positions(order):
    """x position per scope, with one empty slot between the three groups."""
    shift = order["patch_mode"].map({"single": 0, "block": 1, "all": 2}).to_numpy()
    return np.arange(len(order)) + shift


def draw_scopes(ax, order, x, frames, col, log=False):
    """Matched and control points of one metric (col: sym or iia) at the scope positions."""
    spread = np.linspace(-0.27, 0.27, 1 + len(CONTROLS))
    for off, (key, label, marker, color) in zip(spread, (MATCHED, *CONTROLS)):
        f = order if key == "matched" else frames[key]
        y, sd = f[col].to_numpy(), f[col + "_sd"].to_numpy()
        lower = np.minimum(sd, 0.9 * y) if log else sd
        ax.errorbar(x + off, y, [lower, sd], fmt=marker, color=color, markersize=2.6,
                    elinewidth=0.4, capsize=0, markeredgewidth=0, zorder=3 if key == "matched" else 2)
    for cut in (x[order["patch_mode"] == "block"].min() - 1.0, x[order["patch_mode"] == "all"].min() - 1.0):
        ax.axvline(cut, color=style.HAIR, lw=0.6, zorder=0)
    ax.set_xlim(-0.8, x.max() + 0.8)


def group_labels(ax, order, x):
    """Group names ("single subgroup", "block", "all") along the top of ax."""
    for mode, label in (("single", "single subgroup"), ("block", "block"), ("all", "all")):
        xs = x[(order["patch_mode"] == mode).to_numpy()]
        ax.text((xs.min() + xs.max()) / 2, 0.98, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6, color=style.MUTED)


def load_joint(path):
    """Greedy-step rows: IIA in both directions (the smaller of the two flip rates) for matched,
    noise and same-class sources per set size, mean and std over folds."""
    df = pd.read_csv(path)
    df = df[df["kind"] == "greedy_step"]
    if df.empty:
        raise ValueError(f"{path}: no greedy_step rows")
    cols = {"matched": ("iia_l2m_matched", "iia_m2l_matched"),
            "noise": ("iia_l2m_noise_mean", "iia_m2l_noise_mean"),
            "sameclass": ("iia_l2m_sameclass_mean", "iia_m2l_sameclass_mean")}
    missing = [c for pair in cols.values() for c in pair if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    both = pd.DataFrame({k: df[list(c)].min(axis=1, skipna=False) for k, c in cols.items()})
    g = both.groupby(df["set_size"]).agg(["mean", "std"]).fillna(0)
    return {k: (g[(k, "mean")], g[(k, "std")]) for k in cols}, g.index


def draw_joint(ax, joint, sizes):
    for key, label, marker, color in (MATCHED, *CONTROLS):
        if key == "shuffled":
            continue  # the joint replay has no shuffled partner
        mean, sd = joint[key]
        ax.errorbar(sizes, mean, sd, color=color, marker=marker, markersize=2.6, lw=0.8,
                    elinewidth=0.5, capsize=1.2, capthick=0.5, label=label)
    ax.set_xticks(sizes)
    ax.set_xlabel("Subgroup set size")
    ax.set_ylabel("IIA, both directions")
    ax.set_ylim(bottom=0)


def build(exp_dir, out_dir, captions=None):
    """Render both figures from exp_dir's CSVs into out_dir; return (inputs, outputs).

    captions: optional {figure name: (caption, alt text)}. A missing input stops the run."""
    paths = patching_paths(exp_dir)
    for p in paths.values():
        if not p.exists():
            raise FileNotFoundError(p)
    cap = captions or {}
    matched, controls = load_scopes(paths["matched"]), load_scopes(paths["controls"])
    matched = matched[matched["patch_mode"].isin(["single", "block", "all"])]

    fig, ax = style.new_figure("single", 65)
    draw_sufficiency(ax, matched)
    outputs = style.save(fig, out_dir, "patching_sufficiency", *cap.get("patching_sufficiency", (None, None)))

    order = scope_order(matched)
    frames = {k: control_frame(matched, controls, k) for k, *_ in CONTROLS}
    x = positions(order)
    joint, sizes = load_joint(paths["joint"])
    fig, axes = style.new_figure("double", 215, mosaic=[["A", "A"], ["B", "B"], ["C", "C"], ["D", "D"], ["E", "T"]],
                                 height_ratios=[1.15, 0.8, 0.8, 0.95, 1.1])
    draw_scopes(axes["A"], order, x, frames, "sym", log=True)
    axes["A"].set_yscale("log")
    axes["A"].set_ylabel("Symmetry score (log)")
    axes["A"].set_ylim(top=axes["A"].get_ylim()[1] * 2.2)  # headroom for the group labels
    group_labels(axes["A"], order, x)
    draw_scopes(axes["B"], order, x, frames, "iia")
    axes["B"].set_ylabel("IIA")
    draw_scopes(axes["C"], order, x, frames, "swap")
    axes["C"].set_ylabel("Two-way swap effect (logit)")
    draw_scopes(axes["D"], order, x, frames, "iia_both")
    axes["D"].set_ylabel("IIA, both directions")
    for key in "BCD":
        axes[key].sharex(axes["A"])
        axes[key].set_ylim(bottom=0)
    for key in "ABC":
        axes[key].tick_params(labelbottom=False)
    names = [display_name(s) if m == "single" else (BLOCK_LABELS[s] if m == "block" else "All")
             for s, m in zip(order["patch_scope"], order["patch_mode"])]
    axes["D"].set_xticks(x, names, rotation=45, ha="right", rotation_mode="anchor")
    fig.legend(handles=[Line2D([], [], ls="", marker=m, color=c, markersize=3.5, label=lab)
                        for _, lab, m, c in (MATCHED, *CONTROLS)],
               loc="outside upper center", ncols=4, handletextpad=0.3, columnspacing=2.0)
    axes["T"].axis("off")
    axes["T"].text(0.04, 0.5, NOTE, fontsize=6, va="center", color=style.MUTED,
                transform=axes["T"].transAxes).set_in_layout(False)  # keep it out of the margins
    draw_joint(axes["E"], joint, sizes)
    for key in "ABCDE":
        style.panel_label(axes[key], key)
    outputs += style.save(fig, out_dir, "patching_controls", *cap.get("patching_controls", (None, None)))
    return list(paths.values()), outputs
