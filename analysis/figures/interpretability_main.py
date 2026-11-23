"""Main-text interpretability figure: three layers in one double-column figure.

A  per subgroup: Fréchet distance (layer 2), ablation drop (layer 1), symmetry (layer 3)
B  single-subgroup symmetry score and IIA
C  greedy joint IIA against the independence nulls, mean and std over folds
Subgroup keys stay TE_* in the data; labels read REP_*.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.text import Text

from . import style

BLOCK_COLORS = {"nonb": "#2a78d6", "te": "#1baf7a", "nonb2": "#eda100"}
BLOCK_LABELS = {"nonb": "NonB", "te": "REP", "nonb2": "NonB2"}
JOINT_COLOR, NULL_COLOR, LOGIT_COLOR = "#a34c32", "#8a8478", "#4a6fa5"
FD_DIVIDER, ABL_DIVIDER = 5.0, 0.003
LABEL_MIN_ABL, LABEL_MIN_SYM = 0.001, 0.05  # a point is labelled above either threshold
QUADRANTS = {"TL": "data signal\nnot used", "TR": "detected by\nboth layers",
             "BL": "neither layer", "BR": "model-dependent\n(not by Fréchet)"}
SMALL = 5.5  # annotation size, pt
BUBBLE = 170  # scatter area in pt^2 at symmetry 1.0


def input_paths(exp_dir):
    """The six input CSVs of exp_dir (a beta_vae_subgroup_base_gNN folder)."""
    exp = Path(exp_dir)
    out = exp / "interpretability"
    return {
        "frechet": exp.parent / "frechet_analysis" / "frechet_ranking.csv",
        "ablation": out / "1.2_ablation" / "all_folds_ablation.csv",
        "pattern_raw": out / "1.4_latent_probing" / "cross_fold_pattern_raw.csv",
        "pattern_residual": out / "1.4_latent_probing" / "cross_fold_pattern_residual.csv",
        "patching": out / "3.1_patching" / "all_folds_patching.csv",
        "greedy": out / "3.3_joint_search" / "joint_patching_greedy_all_folds.csv",
    }


def display_name(sg):
    return "REP_" + sg[3:] if sg.startswith("TE_") else sg


def resolve_block(sg):
    sg = str(sg).upper()
    if sg.startswith(("SS_", "RG4")):
        return "nonb2"
    return "te" if sg.startswith(("TE_", "REP_")) else "nonb"


def bubble_size(sym):
    return np.clip(sym, 0.005, 1.0) ** 0.9 * BUBBLE


def _require_projector(df, path):
    sites = set(df["patch_site"].dropna().unique()) if "patch_site" in df.columns else set()
    if sites != {"projector"}:
        raise ValueError(f"{path}: patch_site is {sorted(sites) or 'missing'}, expected "
                         "projector-site patching output; do not report it as final")


def _lower(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    return df


def _pattern(path, suffix):
    df = _lower(pd.read_csv(path))
    return df[["subgroup", "mean_align"]].rename(columns={"mean_align": f"mean_align_{suffix}"})


def single_summary(raw):
    """(subgroup, symmetry_score, iia_overall) from the raw per-pair patching rows."""
    df = _lower(raw)
    df = df[df["patch_mode"] == "single"].rename(columns={"patch_scope": "subgroup"})
    valid = df[df["iia_baseline_valid"].astype(bool)]
    iia = (valid.assign(ok=valid["iia_success"].astype(bool))
           .groupby(["subgroup", "direction"])["ok"].mean().unstack("direction"))
    delta = df.groupby(["subgroup", "direction"])["delta_logit"].mean().unstack("direction")
    both = ["lnc_to_mrna", "mrna_to_lnc"]
    return pd.DataFrame({
        "iia_overall": iia[both].mean(axis=1, skipna=True),
        "symmetry_score": delta[both].abs().fillna(0).mean(axis=1),
    }).reset_index()


def load_inputs(paths):
    """Read the six CSVs; return (per-subgroup table for A, single-patching summary, greedy rows).
    Raises FileNotFoundError naming a missing file."""
    for p in paths.values():
        if not Path(p).exists():
            raise FileNotFoundError(p)
    fd = _lower(pd.read_csv(paths["frechet"]))
    abl = _lower(pd.read_csv(paths["ablation"]))
    abl = abl[abl["mode"] == "feature_zero"].groupby("subgroup", as_index=False)["acc_drop"].mean()
    abl = abl.rename(columns={"acc_drop": "mean_acc_drop"})
    raw = pd.read_csv(paths["patching"])
    greedy = pd.read_csv(paths["greedy"])
    _require_projector(raw, paths["patching"])
    _require_projector(greedy, paths["greedy"])
    pat = single_summary(raw)
    df = (fd.merge(abl, on="subgroup").merge(_pattern(paths["pattern_raw"], "raw"), on="subgroup")
          .merge(_pattern(paths["pattern_residual"], "residual"), on="subgroup")
          .merge(pat, on="subgroup", how="left").fillna({"symmetry_score": 0.0, "iia_overall": 0.0}))
    if df.empty:
        raise ValueError("no subgroup left after merging the interpretability CSVs")
    return df, pat, greedy


def place_labels(ax, xs, ys, radii, texts, colors, labelled, obstacles=()):
    """Annotate every point with a label at one of 40 candidate offsets (8 directions,
    5 distances) that overlaps no other point, label, leader line, obstacle text or axes
    edge; if none is free, the least overlapping one. Two passes: the second re-places
    each label against all the others. radii: marker radii in points. Call after limits,
    scales and layout are final. Only points with labelled[i] get a label; all are obstacles."""
    fig = ax.figure
    fig.canvas.draw()
    rend, k = fig.canvas.get_renderer(), fig.dpi / 72
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    discs = [(x - r * k, y - r * k, x + r * k, y + r * k) for (x, y), r in zip(pts, radii)]
    edge = ax.get_window_extent()
    fixed = [(lambda b: (b.x0, b.y0, b.x1, b.y1))(Text.get_window_extent(t, rend)) for t in obstacles]
    n = len(texts)
    state = [None] * n  # per point: (annotation, box, leader sample points)

    def inside(pt, b):
        return b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]

    def cost(i, box, leader):
        boxes = [state[j][1] for j in range(n) if j != i and state[j]] + fixed
        others = [discs[j] for j in range(n) if j != i]
        c = sum(not (box[2] < o[0] or o[2] < box[0] or box[3] < o[1] or o[3] < box[1])
                for o in boxes + others)
        c += sum(inside(pt, box) for j in range(n) if j != i and state[j] for pt in state[j][2])
        c += sum(inside(pt, o) for pt in leader for o in boxes + others)
        return c + 5 * (box[0] < edge.x0 or box[2] > edge.x1 or box[1] < edge.y0 or box[3] > edge.y1)

    candidates = [(d, a) for d in (2.5, 8, 14, 22, 30) for a in (0, 180, 90, 270, 45, 135, 315, 225)]
    for _ in range(2):
        for i in [i for i in np.argsort(-np.asarray(radii)) if labelled[i]]:
            if state[i]:
                state[i][0].remove()
                state[i] = None
            best = None
            for dist, ang in candidates:
                d = radii[i] + dist
                dx, dy = d * np.cos(np.radians(ang)), d * np.sin(np.radians(ang))
                ann = ax.annotate(
                    texts[i], (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=SMALL, color=colors[i], zorder=5,
                    ha="left" if dx > 1 else "right" if dx < -1 else "center",
                    va="bottom" if dy > 1 else "top" if dy < -1 else "center",
                    arrowprops=None if dist == 2.5 else dict(
                        arrowstyle="-", color=colors[i], lw=0.4, alpha=0.6, shrinkA=0, shrinkB=0))
                bb = Text.get_window_extent(ann, rend)
                box = (bb.x0, bb.y0, bb.x1, bb.y1)
                leader = [] if dist == 2.5 else [(pts[i][0] + t * dx * k, pts[i][1] + t * dy * k)
                                                 for t in np.linspace(0.2, 1, 6)]
                c = cost(i, box, leader)
                if best is None or c < best[0]:
                    if best:
                        best[1].remove()
                    best = (c, ann, box, leader)
                else:
                    ann.remove()
                if c == 0:
                    break
            state[i] = best[1:]


def panel_a(ax, df):
    """Draw the scatter; return the arguments of place_labels. Points below both label
    thresholds stay unlabelled (their names are in panel B)."""
    log_fd = np.log10(np.maximum(df["frechet_dist"].values, 0.05))
    abl = np.maximum(df["mean_acc_drop"].values, 0.0)
    sizes = bubble_size(df["symmetry_score"].values)
    colors = [BLOCK_COLORS[resolve_block(s)] for s in df["subgroup"]]
    xdiv = np.log10(FD_DIVIDER)
    ax.axvline(xdiv, color=style.HAIR, lw=0.8, ls="--", zorder=0)
    ax.axhline(ABL_DIVIDER, color=style.HAIR, lw=0.8, ls="--", zorder=0)
    ax.scatter(log_fd, abl, s=sizes, c=colors, alpha=0.65, linewidths=0.4,
               edgecolors="white", zorder=3)
    ax.autoscale()
    xl, xr = ax.get_xlim()
    ax.set_xlim(xl - 0.1, xr + 0.3)
    ax.set_ylim(bottom=-0.0012, top=0.028)
    ax.set_yscale("symlog", linthresh=0.001, linscale=0.5)
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_yticks([0, 1e-3, 1e-2])
    ax.set_yticklabels(["0", "10$^{-3}$", "10$^{-2}$"])
    q = dict(fontsize=SMALL, color=style.MUTED, linespacing=1.3, transform=ax.transAxes)
    quad = [ax.text(0.02, 0.98, QUADRANTS["TL"], ha="left", va="top", **q),
            ax.text(0.98, 0.98, QUADRANTS["TR"], ha="right", va="top", **q),
            ax.text(0.02, 0.02, QUADRANTS["BL"], ha="left", va="bottom", **q),
            ax.text(0.98, 0.02, QUADRANTS["BR"], ha="right", va="bottom", **q)]
    ax.set_xlabel("Layer 2: log$_{10}$(Fréchet distance)")
    ax.set_ylabel("Layer 1: ablation Δacc (symlog)")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda t, _: f"{t:.0f}\n({10 ** t:g})" if t >= 0 else f"{t:.0f}"))
    blocks_h = [Line2D([0], [0], marker="o", color="w", markerfacecolor=BLOCK_COLORS[k],
                       markersize=4.5, label=BLOCK_LABELS[k]) for k in ("nonb", "te", "nonb2")]
    size_h = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#aaaaaa", alpha=0.7,
                     markersize=np.sqrt(bubble_size(v)), label=lab)
              for v, lab in ((0.05, "L3 symmetry 0.05"), (0.3, "0.30"), (0.7, "0.70"))]
    ax.legend(handles=blocks_h + size_h, fontsize=SMALL, loc="lower left",
              bbox_to_anchor=(0.0, 1.0), ncol=6, handlelength=2, columnspacing=0.8,
              handletextpad=0.3, borderaxespad=0.3)
    style.panel_label(ax, "A")
    names = df["subgroup"].map(display_name).tolist()
    labelled = (abl >= LABEL_MIN_ABL) | (df["symmetry_score"].values >= LABEL_MIN_SYM)
    return dict(xs=log_fd, ys=abl, radii=np.sqrt(sizes) / 2, texts=names, colors=colors,
                labelled=labelled, obstacles=quad)


def panel_b(ax, pat):
    df = pat.sort_values("symmetry_score", ascending=False).reset_index(drop=True)
    x = np.arange(len(df))
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax.bar(x, df["symmetry_score"], color=[BLOCK_COLORS[resolve_block(s)] for s in df["subgroup"]],
           alpha=0.4, width=0.65, zorder=2)
    ax2.plot(x, df["iia_overall"], color=style.INK, marker="o", markersize=2.5, lw=0.9, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([display_name(s) for s in df["subgroup"]], rotation=90, fontsize=SMALL)
    ax.set_yscale("log")
    ax.set_ylabel("Symmetry score (log)")
    ax2.set_ylabel("IIA")
    ax2.set_ylim(bottom=0)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=NULL_COLOR, alpha=0.4, label="Symmetry"),
                       Line2D([0], [0], color=style.INK, marker="o", markersize=2.5, lw=0.9,
                              label="IIA")], fontsize=SMALL, loc="upper right", handlelength=1.2)
    for i, row in df.head(2).iterrows():  # IIA of the two strongest subgroups
        ax2.annotate(f"{row['iia_overall']:.3f}", xy=(i, row["iia_overall"]),
                     xytext=(4, 2) if i == 0 else (0, -3), textcoords="offset points",
                     fontsize=SMALL, ha="left" if i == 0 else "center",
                     va="bottom" if i == 0 else "top")
    style.panel_label(ax, "B")


def selection_note(greedy, steps=(1, 2)):
    """'selected 1st/2nd' text: the subgroup added at each step and the folds choosing it."""
    n_folds, parts = greedy["fold"].nunique(), []
    for st in steps:
        counts = greedy.loc[greedy["step"] == st, "added_subgroup"].value_counts()
        if not counts.empty:
            parts.append(f"{display_name(counts.index[0])} ({int(counts.iloc[0])}/{n_folds} folds)")
    return "selected 1st / 2nd:\n" + "\n".join(parts)


def panel_c(ax, greedy):
    held = "iia_joint_heldout" in greedy.columns
    joint = "iia_joint_heldout" if held else "iia_joint"
    null = "iia_additive_expected_heldout" if held else "iia_additive_expected"
    spec = {"joint": (joint, "mean"), "joint_sd": (joint, "std"),
            "null": (null, "mean"), "null_sd": (null, "std")}
    logit = "iia_logit_additive_heldout" in greedy.columns
    if logit:
        spec.update(logit=("iia_logit_additive_heldout", "mean"),
                    logit_sd=("iia_logit_additive_heldout", "std"))
    s = greedy.groupby("step").agg(**spec).reset_index()
    kw = dict(markersize=3, capsize=1.5, elinewidth=0.6)
    ax.errorbar(s["step"], s["joint"], s["joint_sd"], color=JOINT_COLOR, marker="o", lw=1.2,
                label="Joint IIA (held-out pairs)" if held else "Joint IIA (searched)", zorder=3, **kw)
    ax.errorbar(s["step"], s["null"], s["null_sd"], color=NULL_COLOR, marker="s", lw=0.9, ls="--",
                label="Independence null", zorder=2, **kw)
    ref = "null"
    if logit:
        ax.errorbar(s["step"], s["logit"], s["logit_sd"], color=LOGIT_COLOR, marker="^", lw=0.9,
                    ls=":", label="Logit-additive null", zorder=2, **kw)
        ref = "logit"
    ax.fill_between(s["step"], s["joint"], s[ref], where=s["joint"] >= s[ref],
                    color=JOINT_COLOR, alpha=0.12, zorder=1)
    ax.set_xlabel("Subgroup set size")
    ax.set_ylabel("IIA")
    ax.set_xticks(s["step"])
    ax.set_ylim(0, s["joint"].max() + 0.12)
    ax.legend(fontsize=SMALL, loc="upper left", handlelength=1.8)
    last = s.iloc[-1]
    mid = (last["joint"] + last[ref]) / 2
    ax.annotate(f"{last['joint'] - last[ref]:+.3f}", xy=(last["step"], mid),
                xytext=(3, 0), textcoords="offset points", fontsize=SMALL + 0.5,
                color=JOINT_COLOR, fontweight="bold", va="center")
    ax.text(0.98, 0.97, selection_note(greedy), transform=ax.transAxes, fontsize=SMALL,
            color=JOINT_COLOR, ha="right", va="top")
    style.panel_label(ax, "C")


def build(exp_dir, out_dir, caption=None, alt=None, height_mm=95):
    """Render the figure from exp_dir's CSVs into out_dir; return (inputs, outputs).
    A missing input or non-projector patching output stops the run."""
    paths = input_paths(exp_dir)
    df, pat, greedy = load_inputs(paths)
    fig, ax = style.new_figure("double", height_mm, mosaic=[["A", "B"], ["A", "C"]],
                               width_ratios=[1.25, 1])
    labels = panel_a(ax["A"], df)
    panel_b(ax["B"], pat)
    panel_c(ax["C"], greedy)
    place_labels(ax["A"], **labels)
    return list(paths.values()), style.save(fig, out_dir, "interpretability_main", caption, alt)