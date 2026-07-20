#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_hard_for_all.py

Biotype enrichment analysis for transcripts that are misclassified by ALL
benchmarked methods simultaneously ("hard-for-all").

Inputs
------
  predictions_merged.csv   — from compare_models.py (has *_hard columns)
  biotype CSV              — transcript_id, biotype, gene_id, gene_name, gene_biotype
  blnc_csv                 — test_predictions.csv (for error_rate, confidence)

Three comparison groups
-----------------------
  hard_all   : hard for every method present in predictions_merged.csv
  hard_some  : hard for at least one but not all methods
  easy_all   : easy for every method

Outputs
-------
  hard_for_all_transcripts.csv      — full per-transcript table for the hard-all group
  biotype_enrichment.csv            — fold-enrichment + Fisher p-value per biotype
  gene_biotype_enrichment.csv       — same at gene biotype level
  summary.txt                       — printed summary

Usage
-----
python analysis/benchmark/analyze_hard_for_all.py \\
    --predictions_csv  gencode_v47_experiments/benchmark_comparison/predictions_merged.csv \\
    --biotype_csv      data/dataset_biotypes/g47_dataset_biotypes_cdhit.csv \\
    --blnc_csv         gencode_v47_experiments/beta_vae_subgroup_base_g47/evaluation_csvs/test_predictions.csv \\
    --output_dir       gencode_v47_experiments/benchmark_comparison/hard_for_all \\
    [--error_only]     # use error_rate > 0 instead of is_hard_case for β-LNC
    [--min_biotype_count  10]  # minimum background count to include a biotype
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_merged(predictions_csv: str, blnc_csv: str | None,
                error_only: bool) -> pd.DataFrame:
    df = pd.read_csv(predictions_csv)
    df["transcript_id"] = df["transcript_id"].astype(str).str.split("|").str[0]

    hard_cols = [c for c in df.columns if c.endswith("_hard")]
    # Fill NaN → False (unmatched transcripts not hard)
    df[hard_cols] = df[hard_cols].fillna(False).astype(bool)

    # Optionally recompute blnc_hard from error_rate
    if error_only and blnc_csv and "blnc_hard" in df.columns:
        blnc_raw = pd.read_csv(blnc_csv)
        blnc_raw["transcript_id"] = (blnc_raw["transcript_id"].astype(str)
                                     .str.split("|").str[0])
        error_map = blnc_raw.set_index("transcript_id")["error_rate"].gt(0)
        df["blnc_hard"] = df["transcript_id"].map(error_map).fillna(False)
        print(f"  β-LNC hard recomputed from error_rate: "
              f"{df['blnc_hard'].sum()} errors")

    # Add confidence from blnc_csv if available
    if blnc_csv:
        blnc_raw = pd.read_csv(blnc_csv)
        blnc_raw["transcript_id"] = (blnc_raw["transcript_id"].astype(str)
                                     .str.split("|").str[0])
        conf_map = blnc_raw.set_index("transcript_id")["mean_confidence"]
        err_map  = blnc_raw.set_index("transcript_id")["error_rate"]
        df["blnc_confidence"] = df["transcript_id"].map(conf_map)
        df["blnc_error_rate"] = df["transcript_id"].map(err_map)

    # Group labels
    df["n_hard_methods"] = df[hard_cols].sum(axis=1)
    n_methods = len(hard_cols)
    df["group"] = "hard_some"
    df.loc[df["n_hard_methods"] == 0,          "group"] = "easy_all"
    df.loc[df["n_hard_methods"] == n_methods,  "group"] = "hard_all"

    return df, hard_cols


def load_biotypes(biotype_csv: str) -> pd.DataFrame:
    bio = pd.read_csv(biotype_csv)
    bio["transcript_id"] = bio["transcript_id"].astype(str).str.split("|").str[0]
    return bio.set_index("transcript_id")


def enrichment_table(
    focal:      pd.Series,   # biotype values for the focal group
    background: pd.Series,   # biotype values for the full test set
    min_bg_count: int = 10,
) -> pd.DataFrame:
    """
    Compute fold-enrichment and Fisher's exact p-value for each biotype.

    For each biotype B:
        focal_B     = count in focal group
        focal_not_B = count in focal group not B
        bg_B        = count in background
        bg_not_B    = count in background not B
    """
    bg_counts    = background.value_counts()
    focal_counts = focal.value_counts()
    n_focal      = len(focal)
    n_bg         = len(background)

    rows = []
    for biotype, bg_count in bg_counts.items():
        if bg_count < min_bg_count:
            continue
        f_count  = focal_counts.get(biotype, 0)
        bg_pct   = bg_count / n_bg * 100
        f_pct    = f_count  / n_focal * 100 if n_focal > 0 else 0
        fold_enr = (f_pct / bg_pct) if bg_pct > 0 else np.nan

        # 2×2 contingency: focal vs rest-of-background
        table = [[f_count,       n_focal - f_count],
                 [bg_count - f_count, n_bg - n_focal - (bg_count - f_count)]]
        # Clip negatives that arise from focal being a subset of background
        table[1][0] = max(table[1][0], 0)
        table[1][1] = max(table[1][1], 0)
        _, p = fisher_exact(table, alternative="greater")

        rows.append(dict(
            biotype       = biotype,
            n_focal       = int(f_count),
            n_background  = int(bg_count),
            pct_focal     = round(f_pct, 2),
            pct_background= round(bg_pct, 2),
            fold_enrichment = round(fold_enr, 3) if not np.isnan(fold_enr) else None,
            p_value_fisher= float(p),
        ))

    if not rows:
        return pd.DataFrame(columns=["biotype", "n_hard", "n_background",
                                    "pct_hard", "pct_background", "fold_enrichment",
                                    "p_value"])
    result = pd.DataFrame(rows).sort_values("fold_enrichment", ascending=False)
    # Bonferroni correction
    n_tests = len(result)
    result["p_bonferroni"] = (result["p_value_fisher"] * n_tests).clip(upper=1.0)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Biotype enrichment analysis for hard-for-all transcripts"
    )
    parser.add_argument("--predictions_csv",   required=True)
    parser.add_argument("--biotype_csv",       required=True)
    parser.add_argument("--blnc_csv",          default=None)
    parser.add_argument("--output_dir",        required=True)
    parser.add_argument("--error_only",        action="store_true",
                        help="Recompute β-LNC hard flag from error_rate only")
    parser.add_argument("--min_biotype_count", type=int, default=10,
                        help="Min background count to include biotype (default 10)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Hard-for-all biotype enrichment analysis")
    print("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading predictions...")
    df, hard_cols = load_merged(
        args.predictions_csv, args.blnc_csv, args.error_only
    )
    n_methods = len(hard_cols)
    print(f"  {len(df):,} transcripts, {n_methods} methods")
    print(f"  Hard cols: {hard_cols}")

    g = df.groupby("group").size()
    print(f"\n  easy_all   : {g.get('easy_all',  0):>6,}  ({100*g.get('easy_all',  0)/len(df):.1f}%)")
    print(f"  hard_some  : {g.get('hard_some', 0):>6,}  ({100*g.get('hard_some', 0)/len(df):.1f}%)")
    print(f"  hard_all   : {g.get('hard_all',  0):>6,}  ({100*g.get('hard_all',  0)/len(df):.1f}%)")

    hard_all_df  = df[df["group"] == "hard_all"].copy()
    easy_all_df  = df[df["group"] == "easy_all"].copy()

    print("\nLoading biotypes...")
    bio = load_biotypes(args.biotype_csv)
    print(f"  {len(bio):,} transcripts with biotype annotation")

    # Join biotypes
    df = df.join(bio[["biotype", "gene_biotype", "gene_name"]], on="transcript_id", how="left")
    n_no_bio = df["biotype"].isna().sum()
    if n_no_bio:
        print(f"  WARNING: {n_no_bio} transcripts without biotype annotation")

    hard_all_df = df[df["group"] == "hard_all"].copy()
    print(f"\n  hard_all group: {len(hard_all_df):,} transcripts")

    # ── Biotype distribution in hard-all ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("Biotype distribution — hard-for-all")
    print("=" * 65)

    bio_counts = hard_all_df["biotype"].value_counts()
    bg_counts  = df["biotype"].value_counts()
    print(f"\n  {'Biotype':<40} {'Hard-all':>8}  {'%hard-all':>10}  {'%background':>12}")
    print("  " + "-" * 74)
    for bt, count in bio_counts.items():
        bg  = bg_counts.get(bt, 0)
        pct_h = 100 * count / len(hard_all_df)
        pct_b = 100 * bg / len(df)
        print(f"  {bt:<40} {count:>8,}  {pct_h:>9.1f}%  {pct_b:>11.1f}%")

    # ── Enrichment vs full test set ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Biotype enrichment (hard-all vs full test set)")
    print("=" * 65)

    enrich = enrichment_table(
        focal      = hard_all_df["biotype"].dropna(),
        background = df["biotype"].dropna(),
        min_bg_count = args.min_biotype_count,
    )
    print(f"\n  {'Biotype':<35} {'n_focal':>8} {'fold':>7} {'p_fisher':>10} {'p_bonf':>10}")
    print("  " + "-" * 74)
    for _, row in enrich.iterrows():
        sig = " *" if row["p_bonferroni"] < 0.05 else ""
        print(f"  {row['biotype']:<35} {row['n_focal']:>8,} "
              f"{row['fold_enrichment']:>7.2f} "
              f"{row['p_value_fisher']:>10.2e} "
              f"{row['p_bonferroni']:>10.2e}{sig}")

    # ── Gene biotype enrichment ───────────────────────────────────────────────
    if "gene_biotype" in df.columns and df["gene_biotype"].notna().any():
        print("\n" + "=" * 65)
        print("Gene biotype enrichment (hard-all vs full test set)")
        print("=" * 65)

        gene_enrich = enrichment_table(
            focal      = hard_all_df["gene_biotype"].dropna(),
            background = df["gene_biotype"].dropna(),
            min_bg_count = args.min_biotype_count,
        )
        print(f"\n  {'Gene biotype':<35} {'n_focal':>8} {'fold':>7} {'p_fisher':>10} {'p_bonf':>10}")
        print("  " + "-" * 74)
        for _, row in gene_enrich.iterrows():
            sig = " *" if row["p_bonferroni"] < 0.05 else ""
            print(f"  {row['biotype']:<35} {row['n_focal']:>8,} "
                  f"{row['fold_enrichment']:>7.2f} "
                  f"{row['p_value_fisher']:>10.2e} "
                  f"{row['p_bonferroni']:>10.2e}{sig}")
    else:
        print("\n  gene_biotype column not available — skipping gene-level enrichment")
        gene_enrich = pd.DataFrame()

    # ── β-LNC confidence for hard-all ────────────────────────────────────────
    if "blnc_confidence" in df.columns:
        print("\n" + "=" * 65)
        print("β-LNC confidence distribution by group")
        print("=" * 65)
        for grp in ["easy_all", "hard_some", "hard_all"]:
            sub = df[df["group"] == grp]["blnc_confidence"].dropna()
            if len(sub) == 0:
                continue
            print(f"\n  {grp} (n={len(sub):,})")
            print(f"    mean={sub.mean():.4f}  median={sub.median():.4f}  "
                  f"std={sub.std():.4f}  "
                  f"min={sub.min():.4f}  max={sub.max():.4f}")

    # ── n_hard_methods distribution for hard-all ──────────────────────────────
    print("\n" + "=" * 65)
    print("Number of methods flagging each transcript as hard")
    print("=" * 65)
    dist = df["n_hard_methods"].value_counts().sort_index()
    for n, count in dist.items():
        bar = "█" * (count // max(1, len(df) // 200))
        print(f"  {n}/{n_methods} methods hard: {count:>6,}  {bar}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    hard_all_df.to_csv(output_dir / "hard_for_all_transcripts.csv", index=False)
    enrich.to_csv(output_dir / "biotype_enrichment.csv", index=False)
    if len(gene_enrich):
        gene_enrich.to_csv(output_dir / "gene_biotype_enrichment.csv", index=False)

    print(f"\nOutputs saved to {output_dir}/")
    print(f"  hard_for_all_transcripts.csv  ({len(hard_all_df):,} transcripts)")
    print(f"  biotype_enrichment.csv")
    print(f"  gene_biotype_enrichment.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()