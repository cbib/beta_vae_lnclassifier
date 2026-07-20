#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_lncrnabert_predictions.py

Converts lncRNABERT prediction CSVs (id, P(pcRNA), class) to the standard
benchmark format used by compare_models.py (same schema as
evaluate_cv_subgroup.py test_predictions.csv).

Output columns
--------------
  transcript_id       — bare ENST ID (pipe-suffix stripped)
  true_label          — lnc / pc  (from FASTA pair ordering)
  consensus_prediction— lnc / pc  (from lncRNABERT 'class' column)
  mean_confidence     — P(pcRNA) if pred=pc, else 1-P(pcRNA)
                        i.e. probability assigned to the predicted class
  error_rate          — 0.0 or 1.0 (misclassified)
  is_hard_case        — True if error_rate > 0 OR mean_confidence < thresh

Usage
-----
# Single release
python analysis/convert_lncrnabert_predictions.py \
    --pred_csv      data/lncRNABERT_results/g47_lncRNABERT_results.csv \
    --lnc_fasta     data/split_gencode_47/lnc_test.fa \
    --pc_fasta      data/split_gencode_47/pc_test.fa \
    --output_csv    gencode_v47_experiments/benchmark_tools/lncrnabert_test_predictions.csv

# Both releases in one call
python analysis/convert_lncrnabert_predictions.py \
    --pred_csv      data/lncRNABERT_results/g47_lncRNABERT_results.csv \
    --lnc_fasta     data/split_gencode_47/lnc_test.fa \
    --pc_fasta      data/split_gencode_47/pc_test.fa \
    --output_csv    gencode_v47_experiments/benchmark_tools/lncrnabert_test_predictions.csv \
    --pred_csv2     data/lncRNABERT_results/g49_lncRNABERT_results.csv \
    --lnc_fasta2    data/split_gencode_49/lnc_test.fa \
    --pc_fasta2     data/split_gencode_49/pc_test.fa \
    --output_csv2   gencode_v49_experiments/benchmark_tools/lncrnabert_test_predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO


LABEL_MAP = {"pcrna": "pc", "pc": "pc", "pcrna": "pc",
             "lncrna": "lnc", "lnc": "lnc", "ncrna": "lnc"}


def load_true_labels(lnc_fasta: str, pc_fasta: str) -> pd.DataFrame:
    """
    Build a transcript_id → true_label mapping from the two FASTAs.
    pc comes first in lncRNABERT's Data() call, then lnc — but we just
    build a lookup so order doesn't matter.
    """
    rows = []
    for path, label in [(pc_fasta, "pc"), (lnc_fasta, "lnc")]:
        for rec in SeqIO.parse(path, "fasta"):
            tid = rec.id.split("|")[0]
            rows.append({"transcript_id": tid, "true_label": label,
                         "sequence_length": len(str(rec.seq))})
    df = pd.DataFrame(rows)
    # Sanity check for duplicate IDs across files
    dups = df["transcript_id"].duplicated().sum()
    if dups:
        print(f"  WARNING: {dups} duplicate transcript IDs across FASTAs")
    return df.set_index("transcript_id")


def convert(
    pred_csv:   str,
    lnc_fasta:  str,
    pc_fasta:   str,
    output_csv: str,
    hard_conf_thresh: float = 0.6,
) -> pd.DataFrame:
    print(f"\nConverting: {pred_csv}")

    # ── Load predictions ──────────────────────────────────────────────────────
    pred_df = pd.read_csv(pred_csv)
    print(f"  Raw columns : {pred_df.columns.tolist()}")
    print(f"  Rows        : {len(pred_df):,}")

    # Normalise column names
    pred_df.columns = [c.strip().lower().replace("(", "").replace(")", "")
                       for c in pred_df.columns]
    # Expected: id, ppcrna, class
    id_col    = next(c for c in pred_df.columns if c in ("id", "transcript_id", "name"))
    prob_col  = next(c for c in pred_df.columns if "pcrna" in c or "prob" in c or c == "p")
    class_col = next(c for c in pred_df.columns if c in ("class", "label", "prediction"))

    pred_df = pred_df.rename(columns={
        id_col:    "raw_id",
        prob_col:  "prob_pc",
        class_col: "pred_class",
    })

    # Strip pipe-suffixed FASTA headers to bare ENST ID
    pred_df["transcript_id"] = pred_df["raw_id"].astype(str).str.split("|").str[0]
    pred_df["prob_pc"]       = pred_df["prob_pc"].astype(float)
    pred_df["pred_class"]    = pred_df["pred_class"].astype(str).str.lower().str.strip()
    # Normalise class labels
    pred_df["pred_label"]    = pred_df["pred_class"].map(
        lambda x: "pc" if "pc" in x else "lnc"
    )

    # ── Load true labels ──────────────────────────────────────────────────────
    print(f"  Loading true labels from FASTAs...")
    true_df = load_true_labels(lnc_fasta, pc_fasta)
    print(f"  True label map: {len(true_df):,} transcripts")

    # ── Join ──────────────────────────────────────────────────────────────────
    pred_df = pred_df.join(true_df, on="transcript_id", how="left")

    n_missing = pred_df["true_label"].isna().sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} transcripts not found in FASTAs — dropped")
        pred_df = pred_df.dropna(subset=["true_label"])

    # ── Confidence = probability assigned to the predicted class ──────────────
    pred_df["mean_confidence"] = np.where(
        pred_df["pred_label"] == "pc",
        pred_df["prob_pc"],
        1.0 - pred_df["prob_pc"],
    )

    # ── Error rate and hard case flag ─────────────────────────────────────────
    pred_df["error_rate"] = (
        pred_df["pred_label"] != pred_df["true_label"]
    ).astype(float)

    pred_df["is_hard_case"] = (
        (pred_df["error_rate"] > 0) |
        (pred_df["mean_confidence"] < hard_conf_thresh)
    )

    # ── Build output ──────────────────────────────────────────────────────────
    out = pred_df[[
        "transcript_id",
        "true_label",
        "pred_label",
        "mean_confidence",
        "prob_pc",
        "sequence_length",
        "error_rate",
        "is_hard_case",
    ]].rename(columns={"pred_label": "consensus_prediction"})

    # ── Summary ───────────────────────────────────────────────────────────────
    n_total = len(out)
    n_hard  = out["is_hard_case"].sum()
    acc     = 1.0 - out["error_rate"].mean()
    print(f"  Samples     : {n_total:,}")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Hard cases  : {n_hard:,} ({100*n_hard/n_total:.1f}%)")
    for lab in ["lnc", "pc"]:
        n_c   = (out["true_label"] == lab).sum()
        n_c_h = ((out["true_label"] == lab) & out["is_hard_case"]).sum()
        print(f"    {lab}: {n_c_h:,} / {n_c:,} hard ({100*n_c_h/n_c:.1f}%)")

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"  Saved → {output_csv}")

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Convert lncRNABERT predictions to benchmark-compatible CSV"
    )
    # Release 1 (required)
    parser.add_argument("--pred_csv",    required=True,
                        help="lncRNABERT predictions CSV (id, P(pcRNA), class)")
    parser.add_argument("--lnc_fasta",   required=True)
    parser.add_argument("--pc_fasta",    required=True)
    parser.add_argument("--output_csv",  required=True)
    # Release 2 (optional)
    parser.add_argument("--pred_csv2",   default=None)
    parser.add_argument("--lnc_fasta2",  default=None)
    parser.add_argument("--pc_fasta2",   default=None)
    parser.add_argument("--output_csv2", default=None)
    # Options
    parser.add_argument("--hard_conf_thresh", type=float, default=0.6)
    args = parser.parse_args()

    print("=" * 65)
    print("lncRNABERT prediction converter")
    print("=" * 65)

    convert(
        pred_csv         = args.pred_csv,
        lnc_fasta        = args.lnc_fasta,
        pc_fasta         = args.pc_fasta,
        output_csv       = args.output_csv,
        hard_conf_thresh = args.hard_conf_thresh,
    )

    if args.pred_csv2:
        for attr, name in [("lnc_fasta2", "--lnc_fasta2"),
                           ("pc_fasta2",  "--pc_fasta2"),
                           ("output_csv2","--output_csv2")]:
            if not getattr(args, attr):
                print(f"ERROR: --pred_csv2 provided but {name} is missing")
                raise SystemExit(1)
        convert(
            pred_csv         = args.pred_csv2,
            lnc_fasta        = args.lnc_fasta2,
            pc_fasta         = args.pc_fasta2,
            output_csv       = args.output_csv2,
            hard_conf_thresh = args.hard_conf_thresh,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()