#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_hard_for_all_cpat.py

CPAT-based feature analysis for the hard-for-all transcript groups,
comparing them against the full test set background by biotype class
(PC and lncRNA separately).

Produces a clean summary table and per-transcript CSV for downstream use.

Inputs
------
  hard_for_all_transcripts.csv  — from analyze_hard_for_all.py
  predictions_with_cpat.csv     — from run_cpat_hard_cases.py
                                   (has coding_prob, orf_size, fickett_score,
                                    hexamer_score, true_label per transcript)

Outputs (per release, written to --output_dir)
----------------------------------------------
  cpat_feature_summary.csv      — median / mean / std per feature per group
  hard_for_all_cpat.csv         — per-transcript CPAT features for hard-for-all
  cpat_feature_summary.txt      — human-readable table (same content)

Usage
-----
# Single release
python analysis/benchmark/analyze_hard_for_all_cpat.py \\
    --hard_csv   gencode_v47_experiments/benchmark_comparison/hard_for_all/hard_for_all_transcripts.csv \\
    --cpat_csv   gencode_v47_experiments/benchmark_tools/cpat/predictions_with_cpat.csv \\
    --output_dir gencode_v47_experiments/benchmark_comparison/hard_for_all \\
    --release    v47

# Both releases in one call
python analysis/benchmark/analyze_hard_for_all_cpat.py \\
    --hard_csv   gencode_v47_experiments/benchmark_comparison/hard_for_all/hard_for_all_transcripts.csv \\
    --cpat_csv   gencode_v47_experiments/benchmark_tools/cpat/predictions_with_cpat.csv \\
    --output_dir gencode_v47_experiments/benchmark_comparison/hard_for_all \\
    --release    v47 \\
    --hard_csv2  gencode_v49_experiments/benchmark_comparison/hard_for_all/hard_for_all_transcripts.csv \\
    --cpat_csv2  gencode_v49_experiments/benchmark_tools/cpat/predictions_with_cpat.csv \\
    --output_dir2 gencode_v49_experiments/benchmark_comparison/hard_for_all \\
    --release2   v49
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


CPAT_FEATURES = ["coding_prob", "orf_size", "fickett_score", "hexamer_score"]

FEATURE_LABELS = {
    "coding_prob":    "CPAT coding probability",
    "orf_size":       "ORF size (nt)",
    "fickett_score":  "Fickett score",
    "hexamer_score":  "Hexamer score",
}

GROUP_ORDER = [
    ("hard_all",  "pc",  "Hard-for-all PC"),
    ("easy_all",  "pc",  "Easy-for-all PC"),
    ("hard_all",  "lnc", "Hard-for-all lncRNA"),
    ("easy_all",  "lnc", "Easy-for-all lncRNA"),
]


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def load_and_merge(hard_csv: str, cpat_csv: str) -> pd.DataFrame:
    hard = pd.read_csv(hard_csv)
    hard["transcript_id"] = (hard["transcript_id"].astype(str)
                             .str.split("|").str[0])
    hard_ids = set(hard["transcript_id"])

    cpat = pd.read_csv(cpat_csv)
    cpat["transcript_id"] = (cpat["transcript_id"].astype(str)
                             .str.split("|").str[0])

    # Merge group label from hard_csv
    cpat = cpat.merge(
        hard[["transcript_id", "group", "n_hard_methods",
              "biotype"]].rename(columns={"biotype": "biotype_hard"}),
        on="transcript_id", how="left",
    )
    # Transcripts not in hard_csv are "hard_some" or "easy_all" —
    # we don't have their group label here so mark as background
    cpat["group"] = cpat["group"].fillna("background")
    cpat["is_hard_all"] = cpat["transcript_id"].isin(hard_ids)

    # Normalise true_label
    cpat["true_label"] = cpat["true_label"].astype(str).str.lower().str.strip()
    cpat["true_label"] = cpat["true_label"].map(
        {"lnc": "lnc", "lncrna": "lnc", "pc": "pc", "pcrna": "pc",
         "noncoding": "lnc", "coding": "pc"}
    )

    return cpat


def feature_stats(series: pd.Series) -> dict:
    s = series.dropna()
    if len(s) == 0:
        return dict(n=0, mean=np.nan, median=np.nan, std=np.nan,
                    q25=np.nan, q75=np.nan)
    return dict(
        n      = len(s),
        mean   = float(s.mean()),
        median = float(s.median()),
        std    = float(s.std()),
        q25    = float(s.quantile(0.25)),
        q75    = float(s.quantile(0.75)),
    )


def run_analysis(df: pd.DataFrame, release: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    summary_df   : one row per (group, true_label, feature) with stats
    hard_all_df  : per-transcript CPAT features for the hard-for-all group
    """
    # ── Per-transcript table for hard-for-all ─────────────────────────────────
    hard_all_df = df[df["is_hard_all"]].copy()
    available   = [f for f in CPAT_FEATURES if f in df.columns]

    # ── Summary stats ─────────────────────────────────────────────────────────
    rows = []
    for feature in available:
        # Hard-for-all vs full background, split by true_label
        for label in ["pc", "lnc"]:
            hard = df[df["is_hard_all"] & (df["true_label"] == label)][feature]
            bg   = df[df["true_label"] == label][feature]

            hard_stats = feature_stats(hard)
            bg_stats   = feature_stats(bg)

            # Mann-Whitney U (one-sided: hard != background)
            if hard_stats["n"] >= 3 and bg_stats["n"] >= 3:
                _, p = mannwhitneyu(
                    hard.dropna(), bg.dropna(),
                    alternative="two-sided",
                )
            else:
                p = np.nan

            rows.append(dict(
                release       = release,
                feature       = feature,
                feature_label = FEATURE_LABELS.get(feature, feature),
                true_label    = label,
                group         = "hard_for_all",
                **{f"hard_{k}": v for k, v in hard_stats.items()},
                **{f"bg_{k}":   v for k, v in bg_stats.items()},
                p_mannwhitney = float(p) if not np.isnan(p) else np.nan,
            ))

    summary_df = pd.DataFrame(rows)
    return summary_df, hard_all_df


def print_summary(summary_df: pd.DataFrame, release: str) -> str:
    """Format summary as a readable table. Returns string for file writing."""
    lines = []
    lines.append("=" * 75)
    lines.append(f"CPAT feature analysis — hard-for-all  [{release}]")
    lines.append("=" * 75)

    for label, label_name in [("pc", "Protein-coding"), ("lnc", "lncRNA")]:
        lines.append(f"\n{label_name} transcripts")
        lines.append("-" * 75)
        lines.append(f"  {'Feature':<30} {'Hard-all':>10} {'Background':>12} "
                     f"{'Fold':>7}  {'p (MWU)':>10}")
        lines.append("  " + "-" * 65)

        sub = summary_df[
            (summary_df["true_label"] == label) &
            (summary_df["release"]    == release)
        ]

        for _, row in sub.iterrows():
            if row["bg_median"] != 0 and not np.isnan(row["bg_median"]):
                fold = row["hard_median"] / row["bg_median"]
                fold_str = f"{fold:>7.2f}×"
            else:
                fold_str = "     N/A"

            p_str = f"{row['p_mannwhitney']:.2e}" if not np.isnan(
                row["p_mannwhitney"]) else "       N/A"
            sig = " *" if (not np.isnan(row["p_mannwhitney"]) and
                           row["p_mannwhitney"] < 0.05) else ""

            lines.append(
                f"  {row['feature_label']:<30} "
                f"{row['hard_median']:>10.3f} "
                f"{row['bg_median']:>12.3f} "
                f"{fold_str}  {p_str}{sig}"
            )
            lines.append(
                f"  {'':30} "
                f"  n={row['hard_n']:<6,} "
                f"  n={row['bg_n']:<6,}"
            )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_release(
    hard_csv:   str,
    cpat_csv:   str,
    output_dir: str,
    release:    str,
) -> pd.DataFrame:
    """Run analysis for one release. Returns summary_df."""
    print(f"\n{'='*65}")
    print(f"Release: {release}")
    print(f"{'='*65}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_and_merge(hard_csv, cpat_csv)
    n_hard = df["is_hard_all"].sum()
    print(f"  Total transcripts   : {len(df):,}")
    print(f"  Hard-for-all        : {n_hard:,}")
    print(f"  Features available  : {[f for f in CPAT_FEATURES if f in df.columns]}")

    summary_df, hard_all_df = run_analysis(df, release)

    # Print and save text summary
    text = print_summary(summary_df, release)
    print(text)
    (out / "cpat_feature_summary.txt").write_text(text)

    # Save CSVs
    summary_df.to_csv(out / "cpat_feature_summary.csv", index=False)
    hard_all_df.to_csv(out / "hard_for_all_cpat.csv",   index=False)

    print(f"  Saved to {out}/")
    print(f"    cpat_feature_summary.csv  ({len(summary_df)} rows)")
    print(f"    cpat_feature_summary.txt")
    print(f"    hard_for_all_cpat.csv     ({len(hard_all_df)} transcripts)")

    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description="CPAT feature analysis for hard-for-all transcripts"
    )
    # Release 1
    parser.add_argument("--hard_csv",    required=True,
                        help="hard_for_all_transcripts.csv from analyze_hard_for_all.py")
    parser.add_argument("--cpat_csv",    required=True,
                        help="predictions_with_cpat.csv from run_cpat_hard_cases.py")
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--release",     default="unknown")
    # Release 2 (optional)
    parser.add_argument("--hard_csv2",   default=None)
    parser.add_argument("--cpat_csv2",   default=None)
    parser.add_argument("--output_dir2", default=None)
    parser.add_argument("--release2",    default="unknown2")
    args = parser.parse_args()

    print("=" * 65)
    print("CPAT feature analysis — hard-for-all transcripts")
    print("=" * 65)

    all_summaries = []

    s1 = process_release(
        hard_csv   = args.hard_csv,
        cpat_csv   = args.cpat_csv,
        output_dir = args.output_dir,
        release    = args.release,
    )
    all_summaries.append(s1)

    if args.hard_csv2:
        for attr, name in [("cpat_csv2", "--cpat_csv2"),
                           ("output_dir2", "--output_dir2")]:
            if not getattr(args, attr):
                print(f"ERROR: --hard_csv2 provided but {name} is missing")
                raise SystemExit(1)
        s2 = process_release(
            hard_csv   = args.hard_csv2,
            cpat_csv   = args.cpat_csv2,
            output_dir = args.output_dir2,
            release    = args.release2,
        )
        all_summaries.append(s2)

        # Combined CSV across both releases
        combined = pd.concat(all_summaries, ignore_index=True)
        combined_path = Path(args.output_dir).parent / "cpat_feature_summary_combined.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n  Combined summary → {combined_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()