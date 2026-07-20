#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_models.py

Unified benchmark comparison across any number of methods on the held-out
test set.

Produces:
  benchmark_table.csv          — accuracy / precision / recall / F1 per method
  predictions_merged.csv       — per-sample predictions from all methods joined
  hard_case_summary.csv        — per-method hard case counts (for upset plot)

The UpSet plot analysis (intersection structure across methods) is handled
separately by plot_upset_benchmark.py, which reads predictions_merged.csv.

Tool output format notes
------------------------
CPAT        : predictions_with_cpat.csv (from run_cpat_hard_cases.py)
              columns: transcript_id, true_label, coding_prob
              threshold: 0.364 (human)

CPC2        : tsv with columns ID, coding_probability, label
              label: "coding" or "noncoding"

FEELnc      : tsv with columns transcript_id, type (lncRNA / mRNA)

LncDC       : csv with columns Description/transcript_id, predict

RNAsamba    : tsv with columns sequence_name, coding_score, classification
              classification: "coding" or "noncoding"

Orthrus     : orthrus_test_predictions.csv from evaluate_orthrus.py
              columns: transcript_id, true_label, prediction, confidence

lncRNABERT  : lncrnabert_test_predictions.csv from convert_lncrnabert_predictions.py
              columns: transcript_id, true_label, consensus_prediction,
                       mean_confidence, error_rate, is_hard_case
              (same schema as β-LNC — loaded via load_blnc)

β-LNC       : test_predictions.csv from evaluate_cv_subgroup.py
              columns: transcript_id, true_label, consensus_prediction,
                       mean_confidence, error_rate, is_hard_case

Usage
-----
python analysis/compare_models.py \\
    --blnc_csv      .../test_predictions.csv \\
    --cpat_csv      .../cpat/predictions_with_cpat.csv \\
    --cpc2_tsv      .../cpc2_results.tsv \\
    --lncdc_tsv     .../lncdc_results.csv \\
    --orthrus_csv   .../orthrus/orthrus_test_predictions.csv \
    --lncrnabert_csv .../lncrnabert_test_predictions.csv \\
    --output_dir    gencode_v49_experiments/benchmark_comparison

Any subset of tools can be passed — only provided files are loaded.
At least --blnc_csv is required to provide reference labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


CPAT_THRESHOLD = 0.364


def _norm_id(series: "pd.Series") -> "pd.Series":
    """Strip pipe-suffix AND version suffix from transcript IDs.
    e.g. ENST00000810508.1|ENSG... → ENST00000810508
         ENST00000810508.1         → ENST00000810508
         ENST00000810508           → ENST00000810508
    """
    return series.astype(str).str.split("|").str[0].str.split(".").str[0]
CPC2_THRESHOLD = 0.5

LABEL_MAP = {
    "lnc": 0, "pc": 1,
    "lncrna": 0, "mrna": 1,
    "noncoding": 0, "coding": 1,
    "0": 0, "1": 1,
}


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name:   str,
) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    return dict(
        method      = name,
        n_samples   = int(len(y_true)),
        accuracy    = float(acc),
        precision   = float(prec),
        recall      = float(rec),
        f1_macro    = float(f1),
        f1_lncrna   = float(f1_per[0]) if len(f1_per) > 0 else 0.0,
        f1_pcrna    = float(f1_per[1]) if len(f1_per) > 1 else 0.0,
        tp          = int(tp),
        tn          = int(tn),
        fp          = int(fp),
        fn          = int(fn),
    )


def print_metrics_table(rows: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("Benchmark table")
    print("=" * 90)
    print(f"  {'Method':<28} {'N':>7}  {'Acc':>7}  {'Prec':>7}  "
          f"{'Rec':>7}  {'F1':>7}  {'F1-lnc':>8}  {'F1-pc':>7}")
    print("  " + "-" * 88)
    for r in rows:
        print(f"  {r['method']:<28} {r['n_samples']:>7,}  "
              f"{r['accuracy']:>7.4f}  {r['precision']:>7.4f}  "
              f"{r['recall']:>7.4f}  {r['f1_macro']:>7.4f}  "
              f"{r['f1_lncrna']:>8.4f}  {r['f1_pcrna']:>7.4f}")
    print("=" * 90)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_blnc(path: str) -> pd.DataFrame:
    """Load β-LNC test_predictions.csv (also used for feature-only variant)."""
    df = pd.read_csv(path)
    df["transcript_id"] = _norm_id(df["transcript_id"])
    df["true_int"] = df["true_label"].str.lower().map(LABEL_MAP)
    df["pred_int"] = df["consensus_prediction"].str.lower().map(LABEL_MAP)
    df["is_hard"]  = df["is_hard_case"].astype(bool)
    return df[["transcript_id", "true_int", "pred_int",
               "mean_confidence", "is_hard"]]


def load_orthrus(path: str) -> pd.DataFrame:
    """Load orthrus_test_predictions.csv from evaluate_orthrus.py."""
    df = pd.read_csv(path)
    df["transcript_id"] = _norm_id(df["transcript_id"])
    df["true_int"] = df["true_label"].str.lower().map(LABEL_MAP)
    df["pred_int"] = df["prediction"].str.lower().map(LABEL_MAP)
    df["is_hard"]  = df["is_hard_case"].astype(bool)
    return df[["transcript_id", "true_int", "pred_int", "confidence", "is_hard"]]


def load_cpat(path: str, threshold: float = CPAT_THRESHOLD) -> pd.DataFrame:
    """Load predictions_with_cpat.csv from run_cpat_hard_cases.py."""
    df = pd.read_csv(path)
    df["transcript_id"] = _norm_id(df["transcript_id"])
    df["true_int"] = df["true_label"].str.lower().map(LABEL_MAP)
    df["pred_int"] = (df["coding_prob"] >= threshold).astype(int)
    df["is_hard"]  = df["pred_int"] != df["true_int"]
    return df[["transcript_id", "true_int", "pred_int", "coding_prob", "is_hard"]]


def load_cpc2(path: str, threshold: float = CPC2_THRESHOLD) -> pd.DataFrame:
    """
    Load CPC2 output TSV.

    CPC2 sometimes writes no header. The canonical column order is:
        ID  mRNA_size  ORF_size  Fickett_score  pI  ORFP  coding_probability  label
    We detect a missing header by checking whether the first column looks like
    a transcript ID (starts with ENST / NM_ / XM_ / or any non-numeric string).
    """
    CPC2_COLS = ["transcript_id", "mrna_size", "orf_size", "fickett_score",
                 "pI", "orfp", "cpc2_prob", "label"]

    # CPC2 output may or may not have a header. Read without assuming one,
    # then check if the first column of row 0 looks like a header token.
    # comment="#" is intentionally omitted so we can inspect the raw first line.
    df = pd.read_csv(path, sep="\t", header=None, dtype=str)
    first_val = str(df.iloc[0, 0]).lstrip("#").strip().lower()
    if first_val in ("id", "transcript_id", "seq_id", "name", "seqid"):
        # First row is a header — drop it and use CPC2_COLS by position
        df = df.iloc[1:].reset_index(drop=True)
    # Skip any remaining comment lines
    df = df[~df.iloc[:, 0].astype(str).str.startswith("#")].reset_index(drop=True)
    n_cols = len(df.columns)
    df.columns = CPC2_COLS[:n_cols]

    print(f"  CPC2 columns (after normalisation): {df.columns.tolist()}")
    df["transcript_id"] = _norm_id(df["transcript_id"])

    if "cpc2_prob" in df.columns:
        df["pred_int"] = (df["cpc2_prob"].astype(float) >= threshold).astype(int)
    elif "label" in df.columns:
        df["pred_int"] = df["label"].astype(str).str.lower().map(LABEL_MAP)
        n_missing = df["pred_int"].isna().sum()
        if n_missing > 0:
            print(f"  WARNING: {n_missing} CPC2 labels unmapped. "
                  f"Unique values: {df['label'].unique()[:5]}")
    else:
        raise ValueError(f"Cannot find probability or label column. "
                         f"Columns: {df.columns.tolist()}")

    return df[["transcript_id", "pred_int"]
              + (["cpc2_prob"] if "cpc2_prob" in df.columns else [])]


def load_feelNc(path: str) -> pd.DataFrame:
    """Load FEELnc classifier output TSV."""
    df = pd.read_csv(path, sep="\t", comment="#")
    print(f"  FEELnc columns: {df.columns.tolist()}")

    id_col = next((c for c in df.columns
                   if c.lower() in ("transcript_id", "query", "id", "name", "txid")),
                  df.columns[0])
    df = df.rename(columns={id_col: "transcript_id"})
    df["transcript_id"] = _norm_id(df["transcript_id"])

    type_col = next((c for c in df.columns
                     if c.lower() in ("type", "class", "label", "biotype",
                                      "prediction")), None)
    if type_col is None:
        raise ValueError(f"Cannot find type/class column. "
                         f"Columns: {df.columns.tolist()}")

    df["pred_int"] = df[type_col].str.lower().map(LABEL_MAP)
    n_missing = df["pred_int"].isna().sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} FEELnc labels unmapped. "
              f"Unique values: {df[type_col].unique()[:10]}")

    return df[["transcript_id", "pred_int"]].dropna()


def load_lncdc(path: str) -> pd.DataFrame:
    """Load LncDC output CSV."""
    df = pd.read_csv(path, comment="#")
    print(f"  LncDC columns: {df.columns.tolist()}")

    id_col = next((c for c in df.columns
                   if c.lower() in ("description", "transcript_id", "id",
                                    "name", "seq_id", "rna_id")),
                  df.columns[0])
    df = df.rename(columns={id_col: "transcript_id"})
    df["transcript_id"] = _norm_id(df["transcript_id"])

    pred_col = next((c for c in df.columns
                     if c.lower() in ("predict", "prediction", "label",
                                      "class", "type", "result")), None)
    if pred_col is None:
        remaining = [c for c in df.columns if c != "transcript_id"]
        pred_col = remaining[-1] if remaining else None
    if pred_col is None:
        raise ValueError(f"Cannot find prediction column. "
                         f"Columns: {df.columns.tolist()}")

    print(f"  LncDC using id='{id_col}' pred='{pred_col}'")
    df["pred_int"] = df[pred_col].astype(str).str.lower().map(LABEL_MAP)
    n_missing = df["pred_int"].isna().sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} LncDC labels unmapped. "
              f"Unique values: {df[pred_col].unique()[:5]}")

    return df[["transcript_id", "pred_int"]].dropna()


def load_rnasamba(path: str) -> pd.DataFrame:
    """
    Load RNAsamba classify output TSV.

    Columns: sequence_name, coding_score, classification
    classification: "coding" or "noncoding"
    """
    df = pd.read_csv(path, sep="\t")
    print(f"  RNAsamba columns: {df.columns.tolist()}")

    df = df.rename(columns={"sequence_name": "transcript_id"})
    df["transcript_id"] = _norm_id(df["transcript_id"])

    df["pred_int"] = df["classification"].astype(str).str.lower().map(LABEL_MAP)
    n_missing = df["pred_int"].isna().sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} RNAsamba labels unmapped. "
              f"Unique values: {df['classification'].unique()[:5]}")

    return df[["transcript_id", "pred_int", "coding_score"]].dropna(
        subset=["pred_int"])
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified benchmark comparison — any number of methods"
    )
    # β-LNC variants
    parser.add_argument("--blnc_csv",         required=True,
                        help="test_predictions.csv from evaluate_cv_subgroup.py")
    parser.add_argument("--feature_only_csv", default=None,
                        help="test_predictions.csv for feature-only variant")
    # External tools
    parser.add_argument("--cpat_csv",         default=None,
                        help="predictions_with_cpat.csv from run_cpat_hard_cases.py")
    parser.add_argument("--cpc2_tsv",         default=None,
                        help="CPC2 output TSV")
    parser.add_argument("--feelNc_tsv",       default=None,
                        help="FEELnc classifier output TSV")
    parser.add_argument("--lncdc_tsv",        default=None,
                        help="LncDC output CSV")
    parser.add_argument("--orthrus_csv",      default=None,
                        help="orthrus_test_predictions.csv from evaluate_orthrus.py")
    parser.add_argument("--lncrnabert_csv",   default=None,
                        help="lncrnabert_test_predictions.csv from convert_lncrnabert_predictions.py")
    parser.add_argument("--rnasamba_tsv",     default=None,
                        help="rnasamba_results.tsv from rnasamba classify")
    # Options
    parser.add_argument("--output_dir",       required=True)
    parser.add_argument("--cpat_threshold",   type=float, default=CPAT_THRESHOLD)
    parser.add_argument("--cpc2_threshold",   type=float, default=CPC2_THRESHOLD)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Benchmark comparison")
    print("=" * 65)

    # ── Load β-LNC (required — provides reference labels) ────────────────────
    print("\nLoading β-LNC predictions...")
    blnc_df = load_blnc(args.blnc_csv)
    print(f"  {len(blnc_df):,} samples")

    ref = blnc_df[["transcript_id", "true_int"]].copy()

    # ── Accumulate results ────────────────────────────────────────────────────
    metrics_rows = []
    # merged_df: one row per transcript, one column per method's hard flag
    merged_df = ref.copy()

    def add_method(name: str, df: pd.DataFrame,
                   pred_col: str = "pred_int",
                   hard_col: str | None = None,
                   flag_col: str | None = None):
        """
        Align df to reference transcripts, compute metrics, add hard flag
        column to merged_df.

        flag_col: column name to use in merged_df for is_hard (default: name)
        """
        nonlocal merged_df

        m = ref.merge(df[["transcript_id", pred_col]
                          + ([hard_col] if hard_col and hard_col in df.columns
                             else [])],
                      on="transcript_id", how="inner")

        n_matched = len(m)
        if n_matched < len(ref) * 0.9:
            print(f"  WARNING: {name} matched only {n_matched:,} / "
                  f"{len(ref):,} reference transcripts")

        y_true = m["true_int"].values
        y_pred = m[pred_col].values
        valid  = ~pd.isna(y_pred)
        if valid.sum() < len(y_pred):
            print(f"  WARNING: {name} has {(~valid).sum()} NaN predictions — dropped")
        y_true = y_true[valid].astype(int)
        y_pred = y_pred[valid].astype(int)

        row = compute_metrics(y_true, y_pred, name)
        metrics_rows.append(row)
        print(f"  {name:<28} n={row['n_samples']:,}  "
              f"acc={row['accuracy']:.4f}  f1={row['f1_macro']:.4f}")

        # Derive is_hard flag for upset plot
        col_name = flag_col or (name.lower()
                                .replace(" ", "_")
                                .replace("(", "").replace(")", "")
                                .replace("-", "_")
                                .replace("/", "_"))
        col_name = f"{col_name}_hard"

        if hard_col and hard_col in df.columns:
            hard_series = df.set_index("transcript_id")[hard_col]
        else:
            # Derive from prediction vs true label
            hard_series = (m.set_index("transcript_id")[pred_col].astype(float)
                           != m.set_index("transcript_id")["true_int"].astype(float))

        merged_df = merged_df.merge(
            hard_series.rename(col_name).reset_index(),
            on="transcript_id", how="left",
        )

    # β-LNC full model
    add_method("β-LNC (full)", blnc_df,
               hard_col="is_hard", flag_col="blnc")

    # Feature-only
    if args.feature_only_csv:
        print("\nLoading feature-only predictions...")
        fo_df = load_blnc(args.feature_only_csv)
        add_method("β-LNC (feature-only)", fo_df,
                   hard_col="is_hard", flag_col="feature_only")

    # CPAT
    if args.cpat_csv:
        print("\nLoading CPAT predictions...")
        cpat_df = load_cpat(args.cpat_csv, threshold=args.cpat_threshold)
        add_method(f"CPAT (t={args.cpat_threshold})", cpat_df,
                   hard_col="is_hard", flag_col="cpat")

    # CPC2
    if args.cpc2_tsv:
        print("\nLoading CPC2 predictions...")
        cpc2_df = load_cpc2(args.cpc2_tsv, threshold=args.cpc2_threshold)
        add_method(f"CPC2 (t={args.cpc2_threshold})", cpc2_df,
                   flag_col="cpc2")

    # FEELnc
    if args.feelNc_tsv:
        print("\nLoading FEELnc predictions...")
        feelNc_df = load_feelNc(args.feelNc_tsv)
        add_method("FEELnc", feelNc_df, flag_col="feelnc")

    # LncDC
    if args.lncdc_tsv:
        print("\nLoading LncDC predictions...")
        lncdc_df = load_lncdc(args.lncdc_tsv)
        add_method("LncDC", lncdc_df, flag_col="lncdc")

    # RNAsamba
    if args.rnasamba_tsv:
        print("\nLoading RNAsamba predictions...")
        rnasamba_df = load_rnasamba(args.rnasamba_tsv)
        add_method("RNAsamba", rnasamba_df, flag_col="rnasamba")

    # Orthrus
    if args.orthrus_csv:
        print("\nLoading Orthrus predictions...")
        orth_df = load_orthrus(args.orthrus_csv)
        add_method("Orthrus 4-track", orth_df,
                   hard_col="is_hard", flag_col="orthrus")

    # lncRNABERT — same schema as β-LNC, load via load_blnc
    if args.lncrnabert_csv:
        print("\nLoading lncRNABERT predictions...")
        bert_df = load_blnc(args.lncrnabert_csv)
        add_method("lncRNA-BERT (3-mer)", bert_df,
                   hard_col="is_hard", flag_col="lncrnabert")

    # ── Print and save benchmark table ────────────────────────────────────────
    print_metrics_table(metrics_rows)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "benchmark_table.csv", index=False)
    print(f"\n  benchmark_table.csv → {output_dir}")

    # ── Hard case summary per method ──────────────────────────────────────────
    hard_cols = [c for c in merged_df.columns if c.endswith("_hard")]
    n = len(merged_df)
    print("\nHard case counts per method:")
    summary_rows = []
    for col in hard_cols:
        n_hard = merged_df[col].astype(bool).sum()
        method = col.replace("_hard", "")
        print(f"  {method:<28}: {n_hard:>6,} ({100*n_hard/n:.1f}%)")
        summary_rows.append({"method": method, "n_hard": int(n_hard),
                              "pct_hard": round(100*n_hard/n, 1)})

    if hard_cols:
        n_all_hard  = merged_df[hard_cols].all(axis=1).sum()
        n_all_easy  = (~merged_df[hard_cols].any(axis=1)).sum()
        print(f"\n  Hard for ALL  : {n_all_hard:>6,} ({100*n_all_hard/n:.1f}%)"
              f"  ← biological ambiguity")
        print(f"  Easy for ALL  : {n_all_easy:>6,} ({100*n_all_easy/n:.1f}%)")

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "hard_case_summary.csv", index=False
    )

    # ── Save merged predictions (input for upset plot) ────────────────────────
    # Fill NaN hard flags with False — NaN means the method had no prediction
    # for that transcript (ID mismatch), not that it was hard
    hard_cols = [c for c in merged_df.columns if c.endswith("_hard")]
    merged_df[hard_cols] = merged_df[hard_cols].fillna(False)

    merged_df.to_csv(output_dir / "predictions_merged.csv", index=False)
    print(f"\n  predictions_merged.csv  → {output_dir}"
          f"  (use with plot_upset_benchmark.py)")
    print(f"  hard_case_summary.csv   → {output_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()