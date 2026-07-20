#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_tsl_from_gtf.py

Extract transcript support level (TSL) and related annotation metadata
from a GENCODE GTF file for a given set of transcript IDs.

Joins against the hard-for-all transcript CSV and produces an enrichment
summary of TSL distribution vs the full test set background.

Usage
-----
python analysis/benchmark/hard_for_all/extract_tsl_from_gtf.py \\
    --gtf          /mnt/cbib/LNClassifier/paper/nonb-pipeline.old/resources/gencode.v47.annotation.gtf \\
    --hard_csv     gencode_v47_experiments/benchmark_comparison/hard_for_all/hard_for_all_transcripts.csv \\
    --background_csv  gencode_v47_experiments/benchmark_comparison/predictions_merged.csv \\
    --output_dir   gencode_v47_experiments/benchmark_comparison/hard_for_all
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import fisher_exact


# ---------------------------------------------------------------------------
# GTF parser — transcript lines only
# ---------------------------------------------------------------------------

def parse_gtf_transcripts(gtf_path: str) -> pd.DataFrame:
    """
    Parse transcript-level entries from a GENCODE GTF.
    Extracts: transcript_id, transcript_support_level, transcript_biotype,
              gene_id, gene_name, tag (fields like 'basic', 'CCDS', etc.)
    """
    print(f"Parsing GTF: {gtf_path}")
    print("  (reading transcript lines only — this may take ~30s for a full GTF)")

    records = []
    attr_re = re.compile(r'(\w+) "([^"]+)"')

    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "transcript":
                continue

            attr_str = fields[8]
            attrs    = dict(attr_re.findall(attr_str))

            tid = attrs.get("transcript_id", "")
            # Strip version suffix for matching
            tid_bare = tid.split(".")[0]

            tsl = attrs.get("transcript_support_level", "NA")
            # GENCODE encodes TSL as e.g. "1", "2", ..., "5", "NA"
            # Sometimes wrapped as "tsl1 (assigned to previous version N)"
            tsl_match = re.match(r"(\d+|NA)", tsl)
            tsl_clean = tsl_match.group(1) if tsl_match else "NA"

            records.append(dict(
                transcript_id      = tid_bare,
                transcript_id_full = tid,
                transcript_biotype = attrs.get("transcript_biotype", ""),
                gene_id            = attrs.get("gene_id",   "").split(".")[0],
                gene_name          = attrs.get("gene_name", ""),
                gene_biotype       = attrs.get("gene_type", ""),
                tsl                = tsl_clean,
                tags               = attrs.get("tag", ""),
            ))

    df = pd.DataFrame(records)
    print(f"  Parsed {len(df):,} transcript entries")
    return df


# ---------------------------------------------------------------------------
# TSL enrichment
# ---------------------------------------------------------------------------

def tsl_enrichment(
    focal_tsl:  pd.Series,
    bg_tsl:     pd.Series,
) -> pd.DataFrame:
    """Fisher's exact enrichment for each TSL level vs background."""
    all_levels = sorted(
        set(focal_tsl.dropna()) | set(bg_tsl.dropna()),
        key=lambda x: (x == "NA", x)
    )
    n_focal = len(focal_tsl)
    n_bg    = len(bg_tsl)
    rows = []
    for level in all_levels:
        f  = (focal_tsl == level).sum()
        b  = (bg_tsl   == level).sum()
        fp = f / n_focal * 100 if n_focal else 0
        bp = b / n_bg    * 100 if n_bg    else 0
        fold = (fp / bp) if bp > 0 else np.nan
        table = [[f, n_focal - f],
                 [max(b - f, 0), max(n_bg - n_focal - (b - f), 0)]]
        _, p = fisher_exact(table, alternative="greater")
        rows.append(dict(
            tsl            = level,
            n_focal        = int(f),
            n_background   = int(b),
            pct_focal      = round(fp, 1),
            pct_background = round(bp, 1),
            fold_enrichment= round(fold, 3) if not np.isnan(fold) else None,
            p_fisher       = float(p),
        ))
    result = pd.DataFrame(rows)
    n = len(result)
    result["p_bonferroni"] = (result["p_fisher"] * n).clip(upper=1.0)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract TSL from GENCODE GTF for hard-for-all transcripts"
    )
    parser.add_argument("--gtf",            required=True)
    parser.add_argument("--hard_csv",       required=True,
                        help="hard_for_all_transcripts.csv")
    parser.add_argument("--background_csv", required=True,
                        help="predictions_merged.csv (all test transcripts)")
    parser.add_argument("--output_dir",     required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load transcript sets ──────────────────────────────────────────────────
    hard = pd.read_csv(args.hard_csv)
    hard["transcript_id"] = hard["transcript_id"].astype(str).str.split(".").str[0]
    hard_ids = set(hard["transcript_id"])

    bg = pd.read_csv(args.background_csv)
    bg["transcript_id"] = bg["transcript_id"].astype(str).str.split(".").str[0]
    bg_ids = set(bg["transcript_id"])

    print(f"Hard-for-all  : {len(hard_ids):,}")
    print(f"Background    : {len(bg_ids):,}")

    # ── Parse GTF ─────────────────────────────────────────────────────────────
    gtf = parse_gtf_transcripts(args.gtf)

    # ── Join ──────────────────────────────────────────────────────────────────
    hard_gtf = hard.merge(
        gtf[["transcript_id", "tsl", "transcript_biotype",
             "gene_name", "gene_biotype", "tags"]],
        on="transcript_id", how="left",
    )
    bg_gtf = bg.merge(
        gtf[["transcript_id", "tsl"]],
        on="transcript_id", how="left",
    )

    n_matched_hard = hard_gtf["tsl"].notna().sum()
    n_matched_bg   = bg_gtf["tsl"].notna().sum()
    print(f"\nGTF match rate:")
    print(f"  Hard-for-all : {n_matched_hard} / {len(hard_gtf)} matched")
    print(f"  Background   : {n_matched_bg} / {len(bg_gtf)} matched")

    # ── TSL distribution ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("TSL distribution — hard-for-all vs background")
    print("=" * 65)

    tsl_levels = ["1", "2", "3", "4", "5", "NA"]
    hard_tsl   = hard_gtf["tsl"].fillna("NA")
    bg_tsl     = bg_gtf["tsl"].fillna("NA")

    print(f"\n  {'TSL':<6} {'Hard-all':>10} {'%hard':>8}  {'%bg':>8}  {'fold':>7}")
    print("  " + "-" * 44)
    for level in tsl_levels:
        n_h   = (hard_tsl == level).sum()
        n_b   = (bg_tsl   == level).sum()
        pct_h = 100 * n_h / len(hard_tsl) if len(hard_tsl) else 0
        pct_b = 100 * n_b / len(bg_tsl)   if len(bg_tsl)   else 0
        fold  = (pct_h / pct_b) if pct_b > 0 else float("nan")
        print(f"  {level:<6} {n_h:>10,}  {pct_h:>7.1f}%  {pct_b:>7.1f}%  {fold:>7.2f}×")

    # ── Split by biotype ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("TSL distribution by biotype class")
    print("=" * 65)
    for biotype_label, biotype_vals in [
        ("Protein-coding", ["protein_coding"]),
        ("lncRNA",         ["lncRNA", "lncrna"]),
        ("NMD",            ["nonsense_mediated_decay"]),
    ]:
        subset = hard_gtf[hard_gtf["transcript_biotype"].isin(biotype_vals)]
        if len(subset) == 0:
            continue
        print(f"\n  {biotype_label} (n={len(subset)})")
        counts = subset["tsl"].fillna("NA").value_counts()
        for level in tsl_levels:
            n = counts.get(level, 0)
            if n > 0:
                print(f"    TSL {level}: {n:>3}  ({100*n/len(subset):.0f}%)")

    # ── Enrichment table ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Fisher enrichment — hard-for-all vs background")
    print("=" * 65)
    enrich = tsl_enrichment(hard_tsl, bg_tsl)
    print(f"\n  {'TSL':<6} {'n_focal':>8} {'fold':>7} {'p_fisher':>10} {'p_bonf':>10}")
    print("  " + "-" * 44)
    for _, row in enrich.iterrows():
        sig = " *" if row["p_bonferroni"] < 0.05 else ""
        fold_str = f"{row['fold_enrichment']:.2f}×" if row["fold_enrichment"] else "  N/A"
        print(f"  {row['tsl']:<6} {row['n_focal']:>8,} {fold_str:>7} "
              f"{row['p_fisher']:>10.2e} {row['p_bonferroni']:>10.2e}{sig}")

    # ── Notable transcripts ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Hard-for-all transcripts by TSL — notable cases")
    print("=" * 65)
    for tsl_val in ["5", "4", "NA"]:
        subset = hard_gtf[hard_gtf["tsl"] == tsl_val]
        if len(subset) == 0:
            continue
        print(f"\n  TSL {tsl_val} (n={len(subset)}):")
        for _, row in subset.iterrows():
            print(f"    {row['transcript_id']:<20} "
                  f"{row.get('transcript_biotype','?'):<30} "
                  f"{row.get('gene_name_y', row.get('gene_name_x', '?'))}"
                )

    # ── Save ──────────────────────────────────────────────────────────────────
    hard_gtf.to_csv(output_dir / "hard_for_all_tsl.csv",    index=False)
    enrich.to_csv(  output_dir / "tsl_enrichment.csv",      index=False)

    print(f"\nSaved:")
    print(f"  hard_for_all_tsl.csv   ({len(hard_gtf)} transcripts)")
    print(f"  tsl_enrichment.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()