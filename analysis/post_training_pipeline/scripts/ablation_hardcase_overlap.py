#!/usr/bin/env python3
"""
ablation_hardcase_overlap.py

Computes hard case set overlaps (Jaccard) between ablation variants
and the full model, and exports per-transcript membership CSVs.
Produces a 4-set Venn diagram as well, per gencode release

Hard case definition: misclassified OR mean_confidence < threshold.
"""

import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations

# ── matplotlib style ──────────────────────────────────────────────────────────
# ── at the top, replace COLORS and rcParams ───────────────────────────────────
plt.rcParams.update({
    "font.family":  "sans-serif",
    "font.size":    9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

COLORS = {
    "Full model": "#000000",
    "Seq. only":  "#000000",
    "Seq.+TE":    "#000000",
    "Seq.+NonB":  "#000000",
}

INACTIVE_COLOR = "#cccccc"
ACTIVE_COLOR   = "#000000"
BAR_COLOR      = "#333333"
SINGLETON_COLOR = "#333333"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--full_model",  required=True,
                   help="test_predictions.csv of the full model")
    p.add_argument("--ablations",   required=True, nargs="+",
                   metavar="LABEL:PATH",
                   help="label:path pairs for ablation test_predictions.csv")
    p.add_argument("--confidence_threshold", type=float, default=0.6)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--gencode_version", required=True,
                   help="e.g. v47 or v49")
    return p.parse_args()


def load_hard_cases(csv_path: str, threshold: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df[
        (df["consensus_prediction"] != df["true_label"]) |
        (df["mean_confidence"] < threshold)
    ].copy()


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _region_counts(sets: dict[str, set]) -> dict[frozenset, int]:
    """
    Compute exact cardinality for every non-empty intersection region
    of an arbitrary number of sets using inclusion-exclusion on the
    power set.  Each transcript is assigned to exactly one region
    (the maximal subset it belongs to).
    """
    labels = list(sets.keys())
    all_ids = set().union(*sets.values())
    region_map: dict[str, frozenset] = {}
    for tid in all_ids:
        membership = frozenset(l for l in labels if tid in sets[l])
        region_map[tid] = membership
    counts: dict[frozenset, int] = {}
    for membership in region_map.values():
        counts[membership] = counts.get(membership, 0) + 1
    return counts


def make_upset(sets: dict[str, set], title: str, out_path: Path):
    labels = list(sets.keys())
    n_sets = len(labels)

    counts = _region_counts(sets)
    regions = sorted(
        [(reg, cnt) for reg, cnt in counts.items() if cnt > 0],
        key=lambda x: -x[1]
    )
    n_regions = len(regions)

    fig = plt.figure(figsize=(max(9, n_regions * 0.62), 4.5))

    ax_bar  = fig.add_axes([0.10, 0.32, 0.88, 0.58])
    ax_dots = fig.add_axes([0.10, 0.04, 0.88, 0.26])

    xs = list(range(n_regions))

    # ── intersection size bars ────────────────────────────────────────────
    max_cnt = max(cnt for _, cnt in regions)
    ax_bar.bar(xs, [cnt for _, cnt in regions],
               color=BAR_COLOR, width=0.6, zorder=3)
    for xi, (_, cnt) in enumerate(regions):
        ax_bar.text(xi, cnt + max_cnt * 0.02, str(cnt),
                    ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color="black")
    ax_bar.set_xlim(-0.7, n_regions - 0.3)
    ax_bar.set_ylim(0, max_cnt * 1.18)
    ax_bar.set_ylabel("")
    ax_bar.text(0, max_cnt * 1.20, "Intersection size", fontsize=9,
            ha="left", va="bottom", transform=ax_bar.transData)
    ax_bar.set_xticks([])
    ax_bar.spines["bottom"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.set_title(title, fontsize=10, fontweight="bold", pad=10)
    ax_bar.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0, color="grey")

    # ── dot matrix ────────────────────────────────────────────────────────
    ax_dots.set_xlim(-0.7, n_regions - 0.3)
    ax_dots.set_ylim(-0.6, n_sets - 0.4)
    ax_dots.set_xticks([])
    ax_dots.set_yticks(range(n_sets))
    ax_dots.set_yticklabels(
        [f"{l}  (n={len(sets[l])})" for l in labels[::-1]],
        fontsize=8.5
    )    
    ax_dots.tick_params(axis="y", length=0, pad=6)
    ax_dots.spines[:].set_visible(False)

    for yi in range(n_sets):
        ax_dots.axhline(yi, color="#e8e8e8", linewidth=0.8, zorder=0)

    for xi, (reg, _) in enumerate(regions):
        active_ys   = [n_sets - 1 - labels.index(l) for l in reg]
        inactive_ys = [y for y in range(n_sets) if y not in active_ys]
        for yi in inactive_ys:
            ax_dots.scatter(xi, yi, s=55, color=INACTIVE_COLOR,
                            zorder=2, linewidths=0)
        for yi in active_ys:
            ax_dots.scatter(xi, yi, s=80, color=ACTIVE_COLOR,
                            zorder=3, linewidths=0)
        if len(active_ys) > 1:
            ax_dots.plot([xi, xi], [min(active_ys), max(active_ys)],
                         color=ACTIVE_COLOR, linewidth=2.2, zorder=2)

    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  UpSet plot saved: {out_path}")


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    v = args.gencode_version

    # ── Load sets ─────────────────────────────────────────────────────────
    full_df  = load_hard_cases(args.full_model, args.confidence_threshold)
    full_ids = set(full_df["transcript_id"])

    ablations: dict[str, pd.DataFrame] = {}
    for item in args.ablations:
        label, path = item.split(":", 1)
        ablations[label] = load_hard_cases(path, args.confidence_threshold)

    all_sets: dict[str, set] = {"Full model": full_ids}
    all_sets |= {lbl: set(df["transcript_id"])
                 for lbl, df in ablations.items()}

    # ── Jaccard summary ───────────────────────────────────────────────────
    rows = []
    for label, ids in ablations.items():
        s = set(ids["transcript_id"]) if isinstance(ids, pd.DataFrame) \
            else ids
        abl_ids = set(ablations[label]["transcript_id"])
        rows.append({
            "gencode_version":  v,
            "ablation":         label,
            "n_hard_ablation":  len(abl_ids),
            "n_hard_full":      len(full_ids),
            "n_intersection":   len(abl_ids & full_ids),
            "n_union":          len(abl_ids | full_ids),
            "jaccard":          round(jaccard(abl_ids, full_ids), 4),
            "rescued_by_full":  len(abl_ids - full_ids),
            "new_in_full":      len(full_ids - abl_ids),
        })

    summary = pd.DataFrame(rows)
    summary_path = out / f"hardcase_jaccard_{v}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n── Jaccard summary ({v}) ──")
    print(summary.to_string(index=False))

    # ── Pairwise Jaccard table ────────────────────────────────────────────
    pair_rows = []
    set_labels = list(all_sets.keys())
    for a, b in combinations(set_labels, 2):
        pair_rows.append({
            "set_a": a, "set_b": b,
            "jaccard": round(jaccard(all_sets[a], all_sets[b]), 4),
        })
    pairs = pd.DataFrame(pair_rows)
    pairs_path = out / f"hardcase_jaccard_pairwise_{v}.csv"
    pairs.to_csv(pairs_path, index=False)

    # ── Membership CSV (Zenodo) ───────────────────────────────────────────
    all_ids = set().union(*all_sets.values())
    membership = pd.DataFrame({"transcript_id": sorted(all_ids)})
    for label, s in all_sets.items():
        membership[label] = membership["transcript_id"].isin(s)
    membership_path = out / f"hardcase_membership_{v}.csv"
    membership.to_csv(membership_path, index=False)

    print(f"\nSaved:\n  {summary_path}\n  {pairs_path}\n  {membership_path}")

    upset_path = out / f"hardcase_upset_{v}.png"
    make_upset(
        all_sets,
        title=f"Hard case overlap — GENCODE{v}",
        out_path=upset_path,
    )


if __name__ == "__main__":
    main()