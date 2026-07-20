#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_upset_benchmark.py

UpSet plot showing the intersection structure of misclassified transcripts
across all benchmarked methods.

Reads predictions_merged.csv produced by compare_models.py, which contains
one boolean {method}_hard column per method. Works with any number of methods
— no hardcoded method names.

Two plots:

1. upset_hard_cases.png
   All transcripts that are hard (misclassified or low-confidence) for at
   least one method. Each bar = transcripts hard for exactly that combination.

2. upset_blnc_advantage.png
   Transcripts that are easy for β-LNC but hard for at least one other method.
   Shows which combinations of other methods β-LNC uniquely outperforms.
   Skipped if blnc_hard column is absent.

Requires upsetplot >= 0.9:
    pip install upsetplot

Usage
-----
python analysis/plot_upset_benchmark.py \
    --predictions_csv  gencode_v49_experiments/benchmark_comparison/predictions_merged.csv \
    --output_dir       gencode_v49_experiments/benchmark_comparison \
    [--min_set_size    50] \
    [--method_labels   "blnc=β-LNC,lncrnabert=lncRNA-BERT"]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from upsetplot import UpSet, from_indicators
    HAS_UPSET = True
except ImportError:
    HAS_UPSET = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_upsetplot():
    if not HAS_UPSET:
        print("ERROR: upsetplot not installed.")
        print("  pip install upsetplot")
        raise SystemExit(1)


KNOWN_LABELS = {
    "blnc":        "β-LNC",
    "feature_only":"β-LNC (feature-only)",
    "orthrus":     "Orthrus 4-track",
    "cpat":        "CPAT",
    "cpc2":        "CPC2",
    "lncdc":       "LncDC",
    "feelnc":      "FEELnc",
    "lncrnabert":  "lncRNA-BERT",
}


def discover_methods(df: pd.DataFrame, label_map: dict[str, str]) -> dict[str, str]:
    """
    Return {display_label: column_name} for all *_hard columns in df.
    Uses KNOWN_LABELS for default display names; label_map overrides both.
    """
    methods = {}
    for col in df.columns:
        if not col.endswith("_hard"):
            continue
        stem  = col[: -len("_hard")]
        label = label_map.get(stem) or KNOWN_LABELS.get(stem, stem.replace("_", " ").title())
        methods[label] = col
    return methods


def build_memberships(
    df:      pd.DataFrame,
    methods: dict[str, str],
    subset:  pd.Series | None = None,
):
    """
    Build upsetplot data using from_indicators — more reliable than
    from_memberships across upsetplot versions.
    NaN hard flags are treated as False.
    """
    from upsetplot import from_indicators

    source = df if subset is None else df[subset]

    # Boolean DataFrame: columns = display labels
    hard_matrix = pd.DataFrame({
        label: source[col].fillna(False).astype(bool)
        for label, col in methods.items()
        if col in source.columns
    }, index=source.index)

    # Keep only rows where at least one method is hard
    hard_matrix = hard_matrix[hard_matrix.any(axis=1)]

    if len(hard_matrix) == 0:
        return None

    data = from_indicators(hard_matrix.columns, data=hard_matrix)
    return data

def _filter_data(data, min_set_size: int, top_n: int = 20):
    """
    Filter upsetplot data to intersections >= min_set_size.
    Works with both Series (from_memberships) and DataFrame (from_indicators).
    Falls back to top_n by count if nothing passes the threshold.
    """
    # from_indicators returns a DataFrame; we need the count column
    if hasattr(data, "columns"):
        counts = data.iloc[:, 0] if len(data.columns) == 1 else data.sum(axis=1)
    else:
        counts = data  # Series from from_memberships

    mask = counts >= min_set_size
    if mask.any():
        return data[mask], min_set_size

    # Fallback: top_n by count
    top_idx = counts.nlargest(min(top_n, len(counts))).index
    return data.loc[top_idx], int(counts.loc[top_idx].min())


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

PALETTE = {
    "primary":   "#2C3E50",
    "highlight": "#E74C3C",
    "easy":      "#27AE60",
}


def _save_upset(data, title: str, output_path: Path) -> None:
    fig = plt.figure(figsize=(max(12, len(data) // 2), 7))
    fig.patch.set_facecolor("white")

    upset = UpSet(
        data,
        subset_size          = "count",
        show_counts          = True,
        sort_by              = "cardinality",
        sort_categories_by   = "cardinality",
        totals_plot_elements = 3,
        facecolor            = PALETTE["primary"],
    )
    upset.plot(fig)
    plt.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path.name}")


# ---------------------------------------------------------------------------
# Plot 1 — all hard cases
# ---------------------------------------------------------------------------

def plot_hard_case_upset(
    df:           pd.DataFrame,
    methods:      dict[str, str],
    output_path:  Path,
    min_set_size: int = 5,
) -> None:
    check_upsetplot()

    data = build_memberships(df, methods)
    if data is None:
        print("  No hard cases found — skipping")
        return

    filtered, actual_min = _filter_data(data, min_set_size)
    if actual_min < min_set_size:
        print(f"  No intersections >= {min_set_size} — "
              f"showing top 20 by count (min size={actual_min})")

    n_methods = len(methods)
    _save_upset(
        filtered,
        title       = (f"Hard Case Intersections Across {n_methods} Methods\n"
                       f"(misclassified or low-confidence; "
                       f"intersections ≥ {actual_min} shown)"),
        output_path = output_path,
    )


# ---------------------------------------------------------------------------
# Plot 2 — β-LNC advantage
# ---------------------------------------------------------------------------

def plot_blnc_advantage_upset(
    df:           pd.DataFrame,
    methods:      dict[str, str],
    blnc_col:     str,
    output_path:  Path,
    min_set_size: int = 5,
) -> None:
    check_upsetplot()

    other_methods = {k: v for k, v in methods.items() if v != blnc_col}
    if not other_methods:
        print("  No other methods — skipping advantage plot")
        return

    # β-LNC correct, at least one other method wrong
    blnc_easy = ~df[blnc_col].astype(bool)
    other_hard_cols = [col for col in other_methods.values() if col in df.columns]
    any_other_hard  = df[other_hard_cols].any(axis=1)
    mask = blnc_easy & any_other_hard

    n_advantage = int(mask.sum())
    if n_advantage == 0:
        print("  No β-LNC advantage cases found — skipping")
        return

    data = build_memberships(df, other_methods, subset=mask)
    if data is None:
        return

    filtered, actual_min = _filter_data(data, min_set_size)
    if actual_min < min_set_size:
        print(f"  No intersections >= {min_set_size} — "
              f"showing top 20 by count (min size={actual_min})")

    blnc_label = next(k for k, v in methods.items() if v == blnc_col)
    _save_upset(
        filtered,
        title       = (f"{blnc_label} Advantage: easy for {blnc_label}, "
                       f"hard for other methods\n"
                       f"(n={n_advantage:,} transcripts; "
                       f"intersections ≥ {actual_min} shown)"),
        output_path = output_path,
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, methods: dict[str, str]) -> pd.DataFrame:
    n = len(df)
    print(f"\nHard case summary  (N={n:,})")
    print(f"  {'Method':<28}  {'Hard':>7}  {'%':>6}")
    print("  " + "-" * 44)

    rows = []
    for label, col in methods.items():
        if col not in df.columns:
            continue
        n_hard = int(df[col].astype(bool).sum())
        print(f"  {label:<28}  {n_hard:>7,}  {100*n_hard/n:>5.1f}%")
        rows.append({"method": label, "column": col,
                     "n_hard": n_hard, "pct_hard": round(100*n_hard/n, 1)})

    hard_cols = [col for col in methods.values() if col in df.columns]
    if hard_cols:
        n_all  = int(df[hard_cols].all(axis=1).sum())
        n_none = int((~df[hard_cols].any(axis=1)).sum())
        print(f"\n  Hard for ALL  : {n_all:>7,}  {100*n_all/n:>5.1f}%  ← biological ambiguity")
        print(f"  Easy for ALL  : {n_none:>7,}  {100*n_none/n:>5.1f}%")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="UpSet plots of hard case intersections across benchmark methods"
    )
    parser.add_argument("--predictions_csv", required=True,
                        help="predictions_merged.csv from compare_models.py")
    parser.add_argument("--output_dir",      required=True)
    parser.add_argument("--min_set_size",    type=int, default=5,
                        help="Minimum intersection size to display (default 5)")
    parser.add_argument("--method_labels",   default="",
                        help="Override display labels: 'stem=Label,stem2=Label2'")
    parser.add_argument("--error_only",      action="store_true",
                        help="Redefine hard as misclassified only (error_rate > 0), "
                             "ignoring low-confidence correct predictions. "
                             "Requires --blnc_csv to recompute β-LNC's flag.")
    parser.add_argument("--blnc_csv",        default=None,
                        help="test_predictions.csv from evaluate_cv_subgroup.py — "
                             "required with --error_only to recompute β-LNC hard flag "
                             "from error_rate rather than is_hard_case")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse optional label overrides
    label_map: dict[str, str] = {}
    if args.method_labels:
        for pair in args.method_labels.split(","):
            pair = pair.strip()
            if "=" in pair:
                stem, label = pair.split("=", 1)
                label_map[stem.strip()] = label.strip()

    print("=" * 65)
    suffix = " [error-only mode]" if args.error_only else ""
    print(f"UpSet plot — hard case intersections{suffix}")
    print("=" * 65)

    df = pd.read_csv(args.predictions_csv)
    df["transcript_id"] = df["transcript_id"].astype(str).str.split("|").str[0]

    # ── Error-only mode: recompute blnc_hard from error_rate ─────────────────
    if args.error_only:
        if args.blnc_csv is None:
            print("ERROR: --error_only requires --blnc_csv")
            raise SystemExit(1)

        blnc_raw = pd.read_csv(args.blnc_csv)
        blnc_raw["transcript_id"] = (blnc_raw["transcript_id"].astype(str)
                                     .str.split("|").str[0])
        # error_rate > 0 means genuinely misclassified
        error_map = blnc_raw.set_index("transcript_id")["error_rate"].gt(0)

        if "blnc_hard" in df.columns:
            df["blnc_hard"] = df["transcript_id"].map(error_map).fillna(False)
            print("  β-LNC hard flag recomputed from error_rate "
                  f"({df['blnc_hard'].sum()} errors, "
                  f"down from {df['blnc_hard'].sum()} is_hard_case)")

        # For methods that already use pure error (CPAT, CPC2 etc.) no change needed
        # lncRNABERT also uses is_hard_case — recompute from error_rate too
        if "lncrnabert_hard" in df.columns and args.blnc_csv:
            # lncRNABERT CSV has same schema — try to find it via sibling path
            lncrnabert_path = Path(args.blnc_csv).parent.parent / \
                              "benchmark_tools" / "lncrnabert_test_predictions.csv"
            if lncrnabert_path.exists():
                bert_raw = pd.read_csv(lncrnabert_path)
                bert_raw["transcript_id"] = (bert_raw["transcript_id"].astype(str)
                                             .str.split("|").str[0])
                bert_error_map = bert_raw.set_index("transcript_id")["error_rate"].gt(0)
                df["lncrnabert_hard"] = (df["transcript_id"]
                                         .map(bert_error_map).fillna(False))
                print(f"  lncRNA-BERT hard flag recomputed from error_rate "
                      f"({df['lncrnabert_hard'].sum()} errors)")
            else:
                print(f"  WARNING: lncRNABERT CSV not found at {lncrnabert_path} "
                      f"— lncRNA-BERT hard flag unchanged")

        out_suffix = "_error_only"
    else:
        out_suffix = ""

    methods = discover_methods(df, label_map)
    if not methods:
        print("ERROR: no *_hard columns found in CSV.")
        print(f"  Columns present: {df.columns.tolist()}")
        raise SystemExit(1)

    print(f"\nMethods detected ({len(methods)}):")
    for label, col in methods.items():
        n_hard = df[col].astype(bool).sum()
        print(f"  {label:<28} ← {col}  ({n_hard:,} hard)")
    print(f"Transcripts: {len(df):,}")

    if len(methods) < 2:
        print("ERROR: need ≥ 2 methods for an UpSet plot.")
        raise SystemExit(1)

    # Summary table
    summary_df = print_summary(df, methods)
    summary_df.to_csv(output_dir / f"hard_case_summary{out_suffix}.csv", index=False)

    # Plot 1 — all hard cases
    print("\nGenerating hard case UpSet plot...")
    plot_hard_case_upset(
        df           = df,
        methods      = methods,
        output_path  = output_dir / f"upset_hard_cases{out_suffix}.png",
        min_set_size = args.min_set_size,
    )

    # Plot 2 — β-LNC advantage (look for blnc_hard column)
    blnc_col = next(
        (col for col in methods.values()
         if col == "blnc_hard" or "blnc" in col),
        None,
    )
    if blnc_col:
        print("\nGenerating β-LNC advantage UpSet plot...")
        plot_blnc_advantage_upset(
            df           = df,
            methods      = methods,
            blnc_col     = blnc_col,
            output_path  = output_dir / f"upset_blnc_advantage{out_suffix}.png",
            min_set_size = args.min_set_size,
        )
    else:
        print("\n  No blnc_hard column found — skipping advantage plot")

    print(f"\nOutputs saved to {output_dir}/")
    print(f"  upset_hard_cases{out_suffix}.png")
    if blnc_col:
        print(f"  upset_blnc_advantage{out_suffix}.png")
    print(f"  hard_case_summary{out_suffix}.csv")


if __name__ == "__main__":
    main()