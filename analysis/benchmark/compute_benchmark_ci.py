#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis/benchmark/compute_benchmark_ci.py

Adds 95% bootstrap CI (F1, n=10,000 resamples by default) and 95% DeLong CI
(AUC) to the benchmark comparison, matching the reporting standard used in
the submitted classification paper's Table 1. compare_models.py computes
only point estimates (accuracy/precision/recall/F1); this script is
additive — it does not modify compare_models.py — and mirrors its argument
names and ID-normalization convention exactly, so it can be grafted into
run_full_benchmark.sh as an additional stage right after compare_models.py,
reusing the same per-method CLI flags and file paths that stage already
derives.

Canonical labels
-----------------
Reuses --blnc_csv as the label source, exactly as compare_models.py does
(per its own docstring: "at least --blnc_csv is required to provide
reference labels"). This is the ONLY required argument; every other
--*_csv / --*_tsv is optional, and only provided methods are evaluated —
same pattern as compare_models.py.

Per-method score conventions (probability of the POSITIVE class, pc/mRNA)
---------------------------------------------------------------------------
  CPAT        : coding_prob is P(coding) directly. Threshold 0.364 (human).
  CPC2        : coding_probability is P(coding). Threshold 0.5.
  LncDC       : 1 - Noncoding_prob = P(coding). Threshold 0.5.
  RNAsamba    : coding_score is P(coding). Threshold 0.5 (tool default).
  Orthrus     : prob_pc is P(pc) directly. Threshold 0.5.
  lncRNA-BERT : prob_pc is P(pc) directly, same schema as β-LNC. Threshold 0.5.
  β-LNC       : mean_confidence is P(predicted class), not P(pc) directly —
                reconstructed as P(pc) = mean_confidence if predicted pc,
                else 1 - mean_confidence. Threshold 0.5.

Pipeline integration (run_full_benchmark.sh)
---------------------------------------------
Intended to be called as Stage 2.5, immediately after compare_models.py and
before upset_benchmark.py, with the SAME COMPARE_ARGS-derived file paths
already assembled by that stage's per-method [[ -f ... ]] checks — i.e. the
shell script only needs to pass this script whichever --*_csv/--*_tsv flags
it already decided to pass to compare_models.py, unchanged.

Output
------
<output_dir>/benchmark_table_with_ci.csv
  method, n_samples, accuracy, precision, recall,
  f1, f1_ci_low, f1_ci_high,
  auc, auc_ci_low, auc_ci_high
Sorted by F1 descending. Exits non-zero if zero methods could be evaluated.

Usage
-----
python analysis/benchmark/compute_benchmark_ci.py \\
    --blnc_csv        gencode_v49_experiments/beta_vae_subgroup_base_g49/evaluation_csvs/test_predictions.csv \\
    --feature_only_csv gencode_v49_experiments/feature_only_g49/evaluation_csvs/test_predictions.csv \\
    --cpat_csv        gencode_v49_experiments/benchmark_tools/cpat/predictions_with_cpat.csv \\
    --cpc2_tsv        gencode_v49_experiments/benchmark_tools/cpc2_results.tsv.txt \\
    --lncdc_csv       gencode_v49_experiments/benchmark_tools/lncdc_results.csv \\
    --rnasamba_tsv    gencode_v49_experiments/benchmark_tools/rnasamba_results.tsv \\
    --orthrus_csv     gencode_v49_experiments/benchmark_tools/orthrus/orthrus_test_predictions.csv \\
    --lncrnabert_csv  gencode_v49_experiments/benchmark_tools/lncrnabert_test_predictions.csv \\
    --n_bootstrap     10000 \\
    --output_dir      gencode_v49_experiments/benchmark_comparison
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------------
# Shared ID normalization — identical to compare_models.py's _norm_id
# ---------------------------------------------------------------------------

def norm_id(series: pd.Series) -> pd.Series:
    """Strip pipe-suffix AND version suffix from transcript IDs.
    e.g. ENST00000810508.1|ENSG... -> ENST00000810508
         ENST00000810508.1         -> ENST00000810508
         ENST00000810508           -> ENST00000810508
    """
    return series.astype(str).str.split("|").str[0].str.split(".").str[0]


LABEL_MAP = {
    "lnc": 0, "pc": 1,
    "lncrna": 0, "mrna": 1,
    "noncoding": 0, "coding": 1,
    "0": 0, "1": 1,
}

DEFAULT_THRESHOLDS = {
    "cpat":               0.364,
    "cpc2":                0.5,
    "lncdc":                0.5,
    "rnasamba":             0.5,
    "orthrus":              0.5,
    "lncrnabert":           0.5,
    "blnc":                 0.5,
    "feature_only":         0.5,
}


# ---------------------------------------------------------------------------
# Canonical labels — from --blnc_csv, same reference role as compare_models.py
# ---------------------------------------------------------------------------

def load_canonical_labels(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "transcript_id" not in df.columns or "true_label" not in df.columns:
        sys.exit(f"ERROR: --blnc_csv ({path}) must have transcript_id and "
                 f"true_label columns (compare_models.py's β-LNC schema)")
    df["id"] = norm_id(df["transcript_id"])
    df["y_true"] = df["true_label"].astype(str).str.lower().map(LABEL_MAP)
    missing = df["y_true"].isna().sum()
    if missing:
        print(f"  WARNING: {missing} rows in --blnc_csv had unrecognized "
              f"true_label values and will be dropped from labels")
    return df.dropna(subset=["y_true"])[["id", "y_true"]].drop_duplicates(subset="id")


# ---------------------------------------------------------------------------
# Gene ids from the test FASTA headers (for gene-clustered resampling)
# ---------------------------------------------------------------------------

_ENSG = re.compile(r"ENSG\d+")


def load_gene_map(fastas: list[str]) -> pd.DataFrame:
    rows = []
    for f in fastas:
        with open(f) as fh:
            for line in fh:
                if line.startswith(">"):
                    m = _ENSG.search(line)
                    tid = line[1:].split()[0].split("|")[0].split(".")[0]
                    rows.append((tid, m.group(0) if m else ""))
    return pd.DataFrame(rows, columns=["id", "gene"]).drop_duplicates("id")


def sha256_short(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Per-method loaders — return (id, y_score) with y_score = P(positive/pc)
# ---------------------------------------------------------------------------

def load_cpat(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["id"] = norm_id(df["transcript_id"])
    df["y_score"] = df["coding_prob"].astype(float)
    return df[["id", "y_score"]].dropna()


def load_cpc2(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.lstrip("#")  # CPC2 writes header as "#ID\t..."
    df["id"] = norm_id(df["ID"])
    df["y_score"] = df["coding_probability"].astype(float)
    return df[["id", "y_score"]].dropna()


def load_lncdc(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    id_col = "Description" if "Description" in df.columns else "transcript_id"
    df["id"] = norm_id(df[id_col])
    df["y_score"] = 1.0 - df["Noncoding_prob"].astype(float)
    return df[["id", "y_score"]].dropna()


def load_rnasamba(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df["id"] = norm_id(df["sequence_name"])
    df["y_score"] = df["coding_score"].astype(float)
    return df[["id", "y_score"]].dropna()


def load_orthrus(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["id"] = norm_id(df["transcript_id"])
    df["y_score"] = df["prob_pc"].astype(float)
    return df[["id", "y_score"]].dropna()


def load_lncrnabert(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["id"] = norm_id(df["transcript_id"])
    df["y_score"] = df["prob_pc"].astype(float)
    return df[["id", "y_score"]].dropna()


def load_blnc_style(path: str) -> pd.DataFrame:
    """Shared loader for --blnc_csv and --feature_only_csv (same schema:
    transcript_id, true_label, consensus_prediction, mean_confidence, ...).
    mean_confidence is P(predicted class); reconstructs P(pc)."""
    df = pd.read_csv(path)
    df["id"] = norm_id(df["transcript_id"])
    pred_is_pc = df["consensus_prediction"].astype(str).str.lower() == "pc"
    df["y_score"] = np.where(pred_is_pc, df["mean_confidence"],
                             1.0 - df["mean_confidence"])
    return df[["id", "y_score"]].dropna()


LOADERS = {
    "cpat":         load_cpat,
    "cpc2":         load_cpc2,
    "lncdc":        load_lncdc,
    "rnasamba":     load_rnasamba,
    "orthrus":      load_orthrus,
    "lncrnabert":   load_lncrnabert,
    "blnc":         load_blnc_style,
    "feature_only": load_blnc_style,
}


# ---------------------------------------------------------------------------
# Bootstrap CI (F1) — n_bootstrap resamples with replacement, percentile CI
# ---------------------------------------------------------------------------

def bootstrap_f1_ci(y_true: np.ndarray, y_pred: np.ndarray,
                    n_bootstrap: int, rng: np.random.Generator,
                    ci: float = 0.95) -> tuple[float, float, float]:
    n = len(y_true)
    point = f1_score(y_true, y_pred, average="macro", zero_division=0)

    boot_scores = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_scores[b] = f1_score(y_true[idx], y_pred[idx],
                                   average="macro", zero_division=0)

    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_scores, [alpha, 1 - alpha])
    return point, float(lo), float(hi)


def cluster_bootstrap(y_true, y_pred, y_score, genes, n_bootstrap, rng, ci=0.95):
    """Resample GENES with replacement (all isoforms of a drawn gene come
    along), so isoform correlation within a gene is respected. Returns
    percentile CIs for macro-F1, accuracy and AUC."""
    from sklearn.metrics import roc_auc_score
    codes, uniq = pd.factorize(genes)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    G = len(uniq)
    f1s, accs, aucs = np.empty(n_bootstrap), np.empty(n_bootstrap), np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        draw = rng.integers(0, G, size=G)
        idx = np.concatenate([order[bounds[g]:bounds[g + 1]] for g in draw])
        yt, yp = y_true[idx], y_pred[idx]
        f1s[b] = f1_score(yt, yp, average="macro", zero_division=0)
        accs[b] = (yt == yp).mean()
        aucs[b] = roc_auc_score(yt, y_score[idx]) if 0 < yt.sum() < len(yt) else np.nan
    a = (1 - ci) / 2
    q = lambda v: tuple(float(x) for x in np.nanquantile(v, [a, 1 - a]))
    return q(f1s), q(accs), q(aucs), G


# ---------------------------------------------------------------------------
# DeLong CI (AUC) — analytic, matches submitted paper's "95% DeLong CI"
# ---------------------------------------------------------------------------

def _delong_variance(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """
    DeLong variance estimator for a single ROC AUC via the structural
    components / placement-values method (Sun & Xu, 2014; equivalent to
    the original DeLong et al. 1988 covariance approach for one
    classifier). Returns (auc, variance).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return float("nan"), float("nan")

    def midrank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty(len(x))
        sorted_x = x[order]
        i = 0
        while i < len(x):
            j = i
            while j < len(x) - 1 and sorted_x[j + 1] == sorted_x[i]:
                j += 1
            ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
            i = j + 1
        return ranks

    combined = np.concatenate([pos, neg])
    tx = midrank(pos)
    ty = midrank(neg)
    tz = midrank(combined)

    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m

    auc = float(np.mean(v01))
    s01 = np.var(v01, ddof=1) if m > 1 else 0.0
    s10 = np.var(v10, ddof=1) if n > 1 else 0.0
    var = s01 / m + s10 / n
    return auc, var


def delong_auc_ci(y_true: np.ndarray, y_score: np.ndarray,
                  ci: float = 0.95) -> tuple[float, float, float]:
    from scipy.stats import norm
    auc, var = _delong_variance(y_true, y_score)
    if np.isnan(auc):
        return float("nan"), float("nan"), float("nan")
    se = np.sqrt(max(var, 0.0))
    z = norm.ppf(1 - (1 - ci) / 2)
    lo = max(0.0, auc - z * se)
    hi = min(1.0, auc + z * se)
    return auc, float(lo), float(hi)


# ---------------------------------------------------------------------------
# Per-method pipeline
# ---------------------------------------------------------------------------

def evaluate_method(name: str, scores_df: pd.DataFrame, labels_df: pd.DataFrame,
                    threshold: float, n_bootstrap: int,
                    rng: np.random.Generator,
                    gene_df: pd.DataFrame | None = None) -> dict | None:
    merged = scores_df.merge(labels_df, on="id", how="inner")
    if gene_df is not None:
        merged = merged.merge(gene_df, on="id", how="left")
        miss = merged["gene"].isna() | (merged["gene"] == "")
        if miss.any():
            # unknown gene -> its own cluster (conservative: no pooling)
            merged.loc[miss, "gene"] = "NOGENE_" + merged.loc[miss, "id"]
    n_before, n_after = len(scores_df), len(merged)
    if n_after == 0:
        print(f"  ERROR: [{name}] zero overlap with canonical labels "
              f"(ID mismatch?) — skipping")
        return None
    if n_after < n_before:
        print(f"  [{name}] {n_before - n_after} predictions dropped "
              f"(no matching canonical label)")

    y_true  = merged["y_true"].astype(int).values
    y_score = merged["y_score"].astype(float).values
    y_pred  = (y_score >= threshold).astype(int)

    f1, f1_lo, f1_hi = bootstrap_f1_ci(y_true, y_pred, n_bootstrap, rng)
    auc, auc_lo, auc_hi = delong_auc_ci(y_true, y_score)

    extra = {}
    if gene_df is not None:
        (fl, fh), (al, ah), (ul, uh), G = cluster_bootstrap(
            y_true, y_pred, y_score, merged["gene"].values, n_bootstrap, rng)
        extra = dict(n_genes=G, f1_gene_ci_low=fl, f1_gene_ci_high=fh,
                     acc_gene_ci_low=al, acc_gene_ci_high=ah,
                     auc_gene_ci_low=ul, auc_gene_ci_high=uh)
    per_class = {}
    for cname, cval in (("lnc", 0), ("pc", 1)):
        m = y_true == cval
        per_class[f"n_{cname}"] = int(m.sum())
        per_class[f"recall_{cname}"] = float((y_pred[m] == cval).mean()) if m.any() else float("nan")
        per_class[f"precision_{cname}"] = (float((y_true[y_pred == cval] == cval).mean())
                                           if (y_pred == cval).any() else float("nan"))
    return dict(
        method=name, n_samples=n_after, **per_class, **extra,
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, average="macro", zero_division=0),
        recall=recall_score(y_true, y_pred, average="macro", zero_division=0),
        f1=f1, f1_ci_low=f1_lo, f1_ci_high=f1_hi,
        auc=auc, auc_ci_low=auc_lo, auc_ci_high=auc_hi,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Mirrors compare_models.py's argument names exactly, for pipeline reuse
    p.add_argument("--blnc_csv",         required=True,
                   help="Required: also the canonical true-label source")
    p.add_argument("--feature_only_csv", default=None)
    p.add_argument("--cpat_csv",         default=None)
    p.add_argument("--cpc2_tsv",         default=None)
    p.add_argument("--lncdc_csv",        default=None)
    p.add_argument("--lncdc_tsv",        default=None,
                   help="alias of --lncdc_csv (compare_models.py name)")
    p.add_argument("--test_lnc_fasta",   default=None,
                   help="test lncRNA FASTA; with --test_pc_fasta enables "
                        "gene-clustered CIs (ENSG from headers)")
    p.add_argument("--test_pc_fasta",    default=None)
    p.add_argument("--feelNc_tsv",       default=None, help="accepted and ignored")
    p.add_argument("--rnasamba_tsv",     default=None)
    p.add_argument("--orthrus_csv",      default=None)
    p.add_argument("--lncrnabert_csv",   default=None)

    p.add_argument("--cpat_threshold",   type=float, default=None)
    p.add_argument("--cpc2_threshold",   type=float, default=None)

    p.add_argument("--common_subset",    action="store_true",
                   help="score every method on the transcripts ALL methods returned; "
                        "the all-available table is also written (..._allavailable.csv)")
    p.add_argument("--n_bootstrap",      type=int, default=10000)
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--output_dir",       required=True)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Canonical labels: --blnc_csv ({args.blnc_csv})")
    labels_df = load_canonical_labels(args.blnc_csv)
    print(f"  {len(labels_df)} transcripts with valid labels")

    gene_df = None
    fastas = [f for f in (args.test_lnc_fasta, args.test_pc_fasta) if f]
    if len(fastas) == 2:
        gene_df = load_gene_map(fastas)
        ng = gene_df.loc[gene_df.gene != "", "gene"].nunique()
        print(f"  gene map: {len(gene_df)} test transcripts, {ng} genes")
        ids = set(labels_df["id"])
        miss = len(ids - set(gene_df["id"]))
        if miss:
            sys.exit(f"ERROR: {miss} labelled transcripts are not in the test "
                     f"FASTAs -- predictions and FASTAs are from different splits")
    else:
        print("  WARNING: no test FASTAs given -> isoform-naive CIs only "
              "(treats isoforms of one gene as independent)")

    thresholds = dict(DEFAULT_THRESHOLDS)
    if args.cpat_threshold is not None:
        thresholds["cpat"] = args.cpat_threshold
    if args.cpc2_threshold is not None:
        thresholds["cpc2"] = args.cpc2_threshold

    file_args = {
        "blnc":         args.blnc_csv,
        "feature_only": args.feature_only_csv,
        "cpat":         args.cpat_csv,
        "cpc2":         args.cpc2_tsv,
        "lncdc":        args.lncdc_csv or args.lncdc_tsv,
        "rnasamba":     args.rnasamba_tsv,
        "orthrus":      args.orthrus_csv,
        "lncrnabert":   args.lncrnabert_csv,
    }

    # load every method's scores once
    scores = {}
    for name, path in file_args.items():
        if path is None:
            continue
        if not Path(path).is_file():
            print(f"  WARNING: [{name}] file not found ({path}) — skipping")
            continue
        try:
            scores[name] = LOADERS[name](path)
        except (KeyError, ValueError) as e:
            print(f"  ERROR: [{name}] failed to load ({e}) — skipping")

    def run_table(lab, tag):
        out = []
        for name, sdf in scores.items():
            print(f"\nEvaluating {name} [{tag}]...")
            row = evaluate_method(name, sdf, lab, thresholds[name],
                                  args.n_bootstrap, rng, gene_df)
            if row is None:
                continue
            row["coverage"] = len(set(sdf["id"]) & set(labels_df["id"])) / len(labels_df)
            out.append(row)
            print(f"  F1={row['f1']:.4f} [{row['f1_ci_low']:.4f}, {row['f1_ci_high']:.4f}]  "
                  f"AUC={row['auc']:.4f} [{row['auc_ci_low']:.4f}, {row['auc_ci_high']:.4f}]")
        return out

    common_info = None
    all_rows = run_table(labels_df, "own available set")
    rows = all_rows
    if args.common_subset:
        common = set(labels_df["id"])
        for sdf in scores.values():
            common &= set(sdf["id"])
        dropped = labels_df[~labels_df["id"].isin(common)].copy()
        dropped["missing_from"] = [",".join(n for n, sdf in scores.items() if i not in set(sdf["id"]))
                                   for i in dropped["id"]]
        dropped.to_csv(out_dir / "excluded_ids_common_subset.csv", index=False)
        lab_c = labels_df[labels_df["id"].isin(common)]
        common_info = dict(n_common=int(len(common)), n_excluded=int(len(dropped)),
                           excluded_lnc=int((dropped.y_true == 0).sum()),
                           excluded_pc=int((dropped.y_true == 1).sum()),
                           excluded_missing_from=dropped["missing_from"].value_counts().to_dict())
        print(f"\nCommon subset: {len(common)} of {len(labels_df)} transcripts "
              f"({len(dropped)} excluded; lnc={common_info['excluded_lnc']}, pc={common_info['excluded_pc']})")
        pd.DataFrame(all_rows).sort_values("f1", ascending=False).to_csv(
            out_dir / "benchmark_table_with_ci_allavailable.csv", index=False)
        rows = run_table(lab_c, "common subset")

    if not rows:
        sys.exit("ERROR: no methods evaluated — check that at least one "
                 "--*_csv/--*_tsv argument points to a valid file")

    out_df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    prov = dict(
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        n_test_records=int(len(labels_df)),
        n_lnc=int((labels_df.y_true == 0).sum()), n_pc=int((labels_df.y_true == 1).sum()),
        n_genes=int(gene_df.loc[gene_df.id.isin(labels_df.id), "gene"].nunique()) if gene_df is not None else None,
        ci="gene-clustered percentile bootstrap" if gene_df is not None else "isoform-naive",
        n_bootstrap=args.n_bootstrap, seed=args.seed,
        thresholds=thresholds,
        files={k: dict(path=v, sha256_16=sha256_short(v))
               for k, v in list(file_args.items()) + [("test_lnc_fasta", args.test_lnc_fasta),
                                                       ("test_pc_fasta", args.test_pc_fasta)]
               if v and Path(v).is_file()},
        methods=[r["method"] for r in rows],
        common_subset=common_info,
    )
    (out_dir / "benchmark_provenance.json").write_text(json.dumps(prov, indent=2))
    out_path = out_dir / "benchmark_table_with_ci.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()