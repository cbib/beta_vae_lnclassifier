#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis/pattern/compare_frechet_releases.py

Quantifies the cross-release shift in raw (model-free) feature structure
that motivated it, by diffing two `analyze_mstar.py` output directories —
one per GENCODE release. Written to follow up on the token activation
patching cross-release finding (Track A, A4): the identity of the
dominant causally-relied-upon subgroup shifted between G49 (SS_BINNED,
TE_CORE) and G47 (TE_CORE, IR), while a pair-level success-overlap check
on the G49 patching data ruled out "redundant pathway, different
selection within one release" (IR was essentially inert in G49, not
quietly active on a different subset of pairs). This script tests the
next-simplest explanation: a genuine shift in the raw feature-level
covariance structure between releases, independent of any trained model.

This is NOT a new analysis method — `analyze_mstar.py` already computes
everything needed (within-class covariance via `lda_analysis`, Frechet
distance ranking via `frechet_analysis`). This script only *diffs* two
runs of it. Run `analyze_mstar.py` once per release first:

    python src/analyze_mstar.py --config configs/beta_vae_subgroup_base_g49.json \\
        --output_dir gencode_v49_experiments/mstar_analysis
    python src/analyze_mstar.py --config configs/beta_vae_subgroup_base_g47.json \\
        --output_dir gencode_v47_experiments/mstar_analysis

Then:

    python analysis/pattern/compare_frechet_releases.py \\
        --dir_a gencode_v49_experiments/mstar_analysis --label_a G49 \\
        --dir_b gencode_v47_experiments/mstar_analysis --label_b G47 \\
        --focus_subgroups IR,MR,STR,TRI,GLOBAL \\
        --output_dir gencode_v49_experiments/mstar_analysis/release_diff

Output
------
release_diff/
  frechet_diff.csv           per-subgroup Frechet dist/rank in both releases + delta
  covariance_block_diff.csv  mean off-diagonal within-class covariance for the
                              focus sub-block, both classes, both releases
  covariance_diff_heatmap.png  (mRNA_B − mRNA_A) and (lncRNA_B − lncRNA_A)
                              covariance difference heatmaps, full subgroup set
  summary.txt                 plain-text summary of the key numbers

Notes
-----
- Requires the two `analyze_mstar.py` runs to have used the SAME subgroup
  ordering (i.e. the same REGISTRY / feature set) — true for a same-
  architecture, different-release comparison like G49 vs G47. If the
  subgroup sets differ, this script will error rather than silently
  misalign columns.
- The mean-off-diagonal statistic is deliberately simple (unweighted mean
  of the strictly-upper-triangular entries of the focus sub-block) so it
  is easy to sanity-check by eye against the heatmap images. It is not a
  claim about statistical significance — no null distribution or
  resampling is computed here. If a reviewer needs a significance test on
  the covariance shift itself, that is a separate, harder problem
  (comparing empirical covariance matrices estimated from two different,
  non-nested samples) and is out of scope for this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_mstar_outputs(output_dir: Path) -> dict:
    """Load the subset of analyze_mstar.py's output files this script needs."""
    frechet_path = output_dir / "frechet_ranking.csv"
    cov_lnc_path = output_dir / "within_class_covariance_lncrna.csv"
    cov_mrna_path = output_dir / "within_class_covariance_mrna.csv"

    for p in (frechet_path, cov_lnc_path, cov_mrna_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Expected analyze_mstar.py output not found: {p}\n"
                f"Run analyze_mstar.py with --output_dir {output_dir} first."
            )

    frechet_df = pd.read_csv(frechet_path)
    frechet_df["rank"] = frechet_df["frechet_dist"].rank(
        ascending=False, method="min"
    ).astype(int)

    cov_lnc = pd.read_csv(cov_lnc_path, index_col=0)
    cov_mrna = pd.read_csv(cov_mrna_path, index_col=0)

    return {
        "frechet": frechet_df.set_index("subgroup"),
        "cov_lnc": cov_lnc,
        "cov_mrna": cov_mrna,
    }


# ---------------------------------------------------------------------------
# Frechet diff
# ---------------------------------------------------------------------------

def build_frechet_diff(
    data_a: dict, label_a: str,
    data_b: dict, label_b: str,
) -> pd.DataFrame:
    fa = data_a["frechet"]
    fb = data_b["frechet"]

    common = fa.index.intersection(fb.index)
    if len(common) < len(fa.index) or len(common) < len(fb.index):
        missing_in_b = set(fa.index) - set(fb.index)
        missing_in_a = set(fb.index) - set(fa.index)
        raise ValueError(
            f"Subgroup sets differ between releases — cannot align.\n"
            f"  In {label_a} but not {label_b}: {missing_in_b}\n"
            f"  In {label_b} but not {label_a}: {missing_in_a}\n"
            f"Both releases must use the same REGISTRY / feature set for "
            f"this comparison to be meaningful."
        )

    rows = []
    for sg in common:
        row_a = fa.loc[sg]
        row_b = fb.loc[sg]
        rows.append({
            "subgroup": sg,
            "block": row_a.get("block", ""),
            f"frechet_dist_{label_a}": row_a["frechet_dist"],
            f"frechet_dist_{label_b}": row_b["frechet_dist"],
            "frechet_delta": row_b["frechet_dist"] - row_a["frechet_dist"],
            f"rank_{label_a}": int(row_a["rank"]),
            f"rank_{label_b}": int(row_b["rank"]),
            "rank_delta": int(row_a["rank"]) - int(row_b["rank"]),  # positive = moved up (more important) in B
            f"cov_term_{label_a}": row_a.get("cov_term", float("nan")),
            f"cov_term_{label_b}": row_b.get("cov_term", float("nan")),
        })

    df = pd.DataFrame(rows).sort_values("frechet_delta", ascending=False)
    return df


# ---------------------------------------------------------------------------
# Covariance sub-block diff
# ---------------------------------------------------------------------------

def mean_off_diagonal(cov: pd.DataFrame, subgroups: list[str]) -> float:
    """Mean of the strictly-upper-triangular entries of the sub-block of
    `cov` restricted to `subgroups`. NaN-safe (skips missing subgroups)."""
    present = [sg for sg in subgroups if sg in cov.index and sg in cov.columns]
    missing = set(subgroups) - set(present)
    if missing:
        print(f"    WARNING: subgroups not found in covariance matrix, "
              f"skipping: {missing}")
    if len(present) < 2:
        return float("nan")

    sub = cov.loc[present, present].values
    iu = np.triu_indices_from(sub, k=1)
    return float(sub[iu].mean())


def build_covariance_block_diff(
    data_a: dict, label_a: str,
    data_b: dict, label_b: str,
    focus_subgroups: list[str],
) -> pd.DataFrame:
    rows = []
    for cls_name, key in [("lncRNA", "cov_lnc"), ("mRNA", "cov_mrna")]:
        mean_a = mean_off_diagonal(data_a[key], focus_subgroups)
        mean_b = mean_off_diagonal(data_b[key], focus_subgroups)
        rows.append({
            "class": cls_name,
            "focus_subgroups": ",".join(focus_subgroups),
            f"mean_offdiag_cov_{label_a}": mean_a,
            f"mean_offdiag_cov_{label_b}": mean_b,
            "delta": (mean_b - mean_a) if pd.notna(mean_a) and pd.notna(mean_b) else float("nan"),
            "ratio_B_over_A": (mean_b / mean_a) if (pd.notna(mean_a) and mean_a != 0 and pd.notna(mean_b)) else float("nan"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Covariance difference heatmaps (full subgroup set, both classes)
# ---------------------------------------------------------------------------

def plot_covariance_diff(
    data_a: dict, label_a: str,
    data_b: dict, label_b: str,
    output_path: Path,
) -> None:
    common_sg = data_a["cov_lnc"].index.intersection(data_b["cov_lnc"].index)
    # Preserve the ordering from release A's output (should match REGISTRY order)
    ordered_sg = [sg for sg in data_a["cov_lnc"].index if sg in common_sg]

    diff_lnc = (data_b["cov_lnc"].loc[ordered_sg, ordered_sg]
                - data_a["cov_lnc"].loc[ordered_sg, ordered_sg])
    diff_mrna = (data_b["cov_mrna"].loc[ordered_sg, ordered_sg]
                 - data_a["cov_mrna"].loc[ordered_sg, ordered_sg])

    vmax = max(diff_lnc.abs().values.max(), diff_mrna.abs().values.max())
    vmax = max(vmax, 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle(
        f"Within-class Covariance Shift: {label_b} \u2212 {label_a}",
        fontsize=18, fontweight="bold",
    )

    for ax, diff_df, title in [
        (axes[0], diff_lnc, f"lncRNA — Covariance Shift ({label_b} \u2212 {label_a})"),
        (axes[1], diff_mrna, f"mRNA — Covariance Shift ({label_b} \u2212 {label_a})"),
    ]:
        sns.heatmap(
            diff_df, ax=ax, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
            square=True, linewidths=0.5, linecolor="lightgray",
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", rotation=90)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved covariance diff heatmap → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Diff two analyze_mstar.py output directories to quantify "
                    "cross-release shift in raw feature covariance structure "
                    "and Frechet ranking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dir_a", required=True, type=Path,
                        help="analyze_mstar.py output_dir for release A (baseline)")
    parser.add_argument("--label_a", default="A", help="Short label for release A, e.g. G49")
    parser.add_argument("--dir_b", required=True, type=Path,
                        help="analyze_mstar.py output_dir for release B (comparison)")
    parser.add_argument("--label_b", default="B", help="Short label for release B, e.g. G47")
    parser.add_argument("--focus_subgroups", default="IR,MR,STR,TRI,GLOBAL",
                        help="Comma-separated subgroup list for the mean-"
                             "off-diagonal sub-block statistic (default: the "
                             "block that visibly tightened in the G47 vs G49 "
                             "mRNA covariance heatmap)")
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()

    focus_subgroups = [s.strip() for s in args.focus_subgroups.split(",") if s.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("M* / Frechet cross-release diff")
    print("=" * 70)
    print(f"Release A ({args.label_a}): {args.dir_a}")
    print(f"Release B ({args.label_b}): {args.dir_b}")
    print(f"Focus sub-block: {focus_subgroups}")
    print("=" * 70)

    print(f"\nLoading {args.label_a} outputs...")
    data_a = load_mstar_outputs(args.dir_a)
    print(f"Loading {args.label_b} outputs...")
    data_b = load_mstar_outputs(args.dir_b)

    # ── Frechet diff ──────────────────────────────────────────────────────
    print("\nComputing Frechet ranking diff...")
    frechet_diff_df = build_frechet_diff(data_a, args.label_a, data_b, args.label_b)
    frechet_diff_path = args.output_dir / "frechet_diff.csv"
    frechet_diff_df.to_csv(frechet_diff_path, index=False)
    print(f"  Saved → {frechet_diff_path}")

    print(f"\nFrechet distance, focus sub-block ({', '.join(focus_subgroups)}):")
    focus_rows = frechet_diff_df[frechet_diff_df["subgroup"].isin(focus_subgroups)]
    for _, row in focus_rows.iterrows():
        print(f"  {row['subgroup']:<10} "
              f"{args.label_a}: FD={row[f'frechet_dist_{args.label_a}']:.4f} "
              f"(rank {row[f'rank_{args.label_a}']:>2d})   "
              f"{args.label_b}: FD={row[f'frechet_dist_{args.label_b}']:.4f} "
              f"(rank {row[f'rank_{args.label_b}']:>2d})   "
              f"delta={row['frechet_delta']:+.4f}  rank_delta={row['rank_delta']:+d}")

    # ── Covariance sub-block diff ────────────────────────────────────────
    print(f"\nComputing within-class covariance sub-block diff...")
    cov_diff_df = build_covariance_block_diff(
        data_a, args.label_a, data_b, args.label_b, focus_subgroups
    )
    cov_diff_path = args.output_dir / "covariance_block_diff.csv"
    cov_diff_df.to_csv(cov_diff_path, index=False)
    print(f"  Saved → {cov_diff_path}")
    print(f"\nMean off-diagonal within-class covariance, "
          f"{', '.join(focus_subgroups)} sub-block:")
    for _, row in cov_diff_df.iterrows():
        a_val = row[f"mean_offdiag_cov_{args.label_a}"]
        b_val = row[f"mean_offdiag_cov_{args.label_b}"]
        print(f"  {row['class']:<8} {args.label_a}={a_val:.4f}  "
              f"{args.label_b}={b_val:.4f}  delta={row['delta']:+.4f}  "
              f"ratio={row['ratio_B_over_A']:.2f}x")

    # ── Full covariance diff heatmap ─────────────────────────────────────
    print("\nGenerating covariance diff heatmap...")
    plot_covariance_diff(
        data_a, args.label_a, data_b, args.label_b,
        args.output_dir / "covariance_diff_heatmap.png",
    )

    # ── Summary ───────────────────────────────────────────────────────────
    summary_path = args.output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"M* / Frechet cross-release diff: {args.label_a} vs {args.label_b}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Focus sub-block: {', '.join(focus_subgroups)}\n\n")
        f.write("Frechet distance (focus sub-block):\n")
        for _, row in focus_rows.iterrows():
            f.write(f"  {row['subgroup']:<10} "
                    f"{args.label_a} FD={row[f'frechet_dist_{args.label_a}']:.4f} "
                    f"(rank {row[f'rank_{args.label_a}']}) -> "
                    f"{args.label_b} FD={row[f'frechet_dist_{args.label_b}']:.4f} "
                    f"(rank {row[f'rank_{args.label_b}']})  "
                    f"delta={row['frechet_delta']:+.4f}\n")
        f.write("\nMean off-diagonal within-class covariance (focus sub-block):\n")
        for _, row in cov_diff_df.iterrows():
            a_val = row[f"mean_offdiag_cov_{args.label_a}"]
            b_val = row[f"mean_offdiag_cov_{args.label_b}"]
            f.write(f"  {row['class']:<8} {args.label_a}={a_val:.4f}  "
                    f"{args.label_b}={b_val:.4f}  delta={row['delta']:+.4f}  "
                    f"ratio={row['ratio_B_over_A']:.2f}x\n")
        f.write(
            "\nInterpretation note: a positive covariance delta together "
            "with a Frechet rank increase (rank_delta > 0, meaning the "
            "subgroup ranked lower/less-important number in B) for the "
            "focus sub-block would support the hypothesis that these "
            "subgroups became a more exploitable joint signal in release "
            "B, independent of any trained model — consistent with, but "
            "not proof of, the token-patching finding that the model's "
            "dominant causal subgroup shifted between releases (Track A, "
            "A4). This is descriptive evidence, not a significance test.\n"
        )
    print(f"\nSummary written → {summary_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"  {frechet_diff_path}")
    print(f"  {cov_diff_path}")
    print(f"  {args.output_dir / 'covariance_diff_heatmap.png'}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()