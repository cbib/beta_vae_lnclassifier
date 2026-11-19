#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis/benchmark/compute_benchmark_ci.py

Adds 95% gene-clustered percentile bootstrap CIs (macro-F1, accuracy, AUC;
10,000 resamples by default; all isoforms of a drawn gene come along) to the
benchmark comparison. compare_models.py computes only point estimates
(accuracy/precision/recall/F1); this script is additive — it does not modify compare_models.py — and mirrors its argument
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
  method, n_samples, n_genes, accuracy, precision, recall, f1, auc,
  f1_gene_ci_low/high, acc_gene_ci_low/high, auc_gene_ci_low/high
  (plus per-class counts, recall and precision)
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
    --test_lnc_fasta  data/split_gencode_49/lnc_test.fa \\
    --test_pc_fasta   data/split_gencode_49/pc_test.fa \\
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
# Gene-clustered percentile bootstrap (macro-F1, accuracy, AUC)
# ---------------------------------------------------------------------------

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
# Per-method pipeline
# ---------------------------------------------------------------------------

def evaluate_method(name: str, scores_df: pd.DataFrame, labels_df: pd.DataFrame,
                    threshold: float, n_bootstrap: int,
                    rng: np.random.Generator,
                    gene_df: pd.DataFrame) -> dict | None:
    merged = scores_df.merge(labels_df, on="id", how="inner")
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

    from sklearn.metrics import roc_auc_score
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    auc = float(roc_auc_score(y_true, y_score)) if 0 < y_true.sum() < len(y_true) else float("nan")
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
        f1=f1, auc=auc,
    )


# ---------------------------------------------------------------------------
# Paired gene-clustered bootstrap of the macro-F1 difference
# ---------------------------------------------------------------------------

def _gene_confusion(y_true, y_pred, codes, n_genes):
    """Per-gene [tn, fp, fn, tp] counts (class 1 = positive)."""
    k = y_true * 2 + y_pred
    return np.bincount(codes * 4 + k, minlength=n_genes * 4).reshape(n_genes, 4)


def _macro_f1_counts(c):
    """Macro-F1 from [..., 4] confusion counts; 0 where undefined
    (sklearn zero_division=0)."""
    tn, fp, fn, tp = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    d1, d0 = 2 * tp + fp + fn, 2 * tn + fn + fp
    f1 = np.where(d1 > 0, 2 * tp / np.maximum(d1, 1), 0.0)
    f0 = np.where(d0 > 0, 2 * tn / np.maximum(d0, 1), 0.0)
    return (f1 + f0) / 2


def paired_f1_bootstrap(y_true, pred_a, pred_b, genes, n_bootstrap, rng,
                        ci=0.95, chunk=500):
    """Macro-F1(a) - macro-F1(b) on the SAME resampled genes. Genes are drawn
    with replacement (a multinomial count per gene), so isoforms stay together
    and both methods see identical resamples. Returns point, ci_low, ci_high,
    p (two-sided bootstrap p, resolution 1/n_bootstrap)."""
    codes, uniq = pd.factorize(genes)
    G = len(uniq)
    ca = _gene_confusion(y_true, pred_a, codes, G)
    cb = _gene_confusion(y_true, pred_b, codes, G)
    point = float(_macro_f1_counts(ca.sum(0)) - _macro_f1_counts(cb.sum(0)))
    diffs = np.empty(n_bootstrap)
    probs = np.full(G, 1.0 / G)
    for s in range(0, n_bootstrap, chunk):
        m = min(chunk, n_bootstrap - s)
        W = rng.multinomial(G, probs, size=m)
        diffs[s:s + m] = _macro_f1_counts(W @ ca) - _macro_f1_counts(W @ cb)
    a = (1 - ci) / 2
    lo, hi = np.quantile(diffs, [a, 1 - a])
    p = min(1.0, 2 * min((diffs <= 0).sum() + 1, (diffs >= 0).sum() + 1) / (n_bootstrap + 1))
    return point, float(lo), float(hi), float(p), G


def run_paired(ref, scores, labels, thresholds, gene_df, n_bootstrap, rng):
    if ref not in scores:
        print(f"  WARNING: paired reference '{ref}' not available; skipping paired table")
        return []
    rows = []
    for name, sdf in scores.items():
        if name == ref:
            continue
        m = (scores[ref].rename(columns={"y_score": "s_ref"})
             .merge(sdf.rename(columns={"y_score": "s_oth"}), on="id")
             .merge(labels, on="id"))
        if gene_df is not None:
            m = m.merge(gene_df, on="id", how="left")
            miss = m["gene"].isna() | (m["gene"] == "")
            m.loc[miss, "gene"] = "NOGENE_" + m.loc[miss, "id"]
            genes = m["gene"].values
        else:
            genes = m["id"].values          # no gene map: every transcript its own cluster
        if m.empty:
            continue
        y = m["y_true"].astype(int).values
        pr = (m["s_ref"].values >= thresholds[ref]).astype(int)
        po = (m["s_oth"].values >= thresholds[name]).astype(int)
        d, lo, hi, p, G = paired_f1_bootstrap(y, pr, po, genes, n_bootstrap, rng)
        rows.append(dict(reference=ref, method=name, n_samples=len(m), n_genes=G,
                         n_lnc=int((y == 0).sum()), n_pc=int((y == 1).sum()),
                         f1_reference=float(_macro_f1_counts(
                             _gene_confusion(y, pr, np.zeros(len(y), int), 1)[0])),
                         f1_method=float(_macro_f1_counts(
                             _gene_confusion(y, po, np.zeros(len(y), int), 1)[0])),
                         diff=d, diff_ci_low=lo, diff_ci_high=hi, p_boot=p))
        print(f"  {ref} - {name}: dF1={d:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.4g} "
              f"(n={len(m)}, genes={G})")
    return rows


# ---------------------------------------------------------------------------
# Contrast between two subsets: is the gap to the reference the same in both?
# ---------------------------------------------------------------------------

_METRICS = {
    "f1": _macro_f1_counts,
    # confusion layout: [tn, fp, fn, tp], class 1 (pc) positive
    "recall_lnc": lambda c: np.where(c[..., 0] + c[..., 1] > 0,
                                     c[..., 0] / np.maximum(c[..., 0] + c[..., 1], 1), 0.0),
    "recall_pc":  lambda c: np.where(c[..., 2] + c[..., 3] > 0,
                                     c[..., 3] / np.maximum(c[..., 2] + c[..., 3], 1), 0.0),
}


def contrast_bootstrap(y, pred_ref, pred_oth, genes, in_a, metrics, n_bootstrap,
                       rng, ci=0.95, chunk=500):
    """For each metric: (ref - other) on subset A, the same on subset B, and
    the swing = diff_A - diff_B. Genes are resampled over A and B together
    (a gene can have transcripts in both), the same resample for every
    quantity. Returns {metric: dict}."""
    codes, uniq = pd.factorize(genes)
    G = len(uniq)

    def counts(pred, mask):
        return _gene_confusion(y[mask], pred[mask], codes[mask], G)

    cr_a, cr_b = counts(pred_ref, in_a), counts(pred_ref, ~in_a)
    co_a, co_b = counts(pred_oth, in_a), counts(pred_oth, ~in_a)
    point = {}
    for k in metrics:
        f = _METRICS[k]
        point[k] = (float(f(cr_a.sum(0)) - f(co_a.sum(0))),
                    float(f(cr_b.sum(0)) - f(co_b.sum(0))))
    draws = {k: np.empty((n_bootstrap, 2)) for k in metrics}
    probs = np.full(G, 1.0 / G)
    for s in range(0, n_bootstrap, chunk):
        m = min(chunk, n_bootstrap - s)
        W = rng.multinomial(G, probs, size=m)
        for k in metrics:
            f = _METRICS[k]
            draws[k][s:s + m, 0] = f(W @ cr_a) - f(W @ co_a)
            draws[k][s:s + m, 1] = f(W @ cr_b) - f(W @ co_b)
    a = (1 - ci) / 2
    out = {}
    for k in metrics:
        da, db = draws[k][:, 0], draws[k][:, 1]
        sw = da - db
        qa, qb, qs = (np.quantile(v, [a, 1 - a]) for v in (da, db, sw))
        p = min(1.0, 2 * min((sw <= 0).sum() + 1, (sw >= 0).sum() + 1) / (n_bootstrap + 1))
        out[k] = dict(diff_focus=point[k][0], diff_focus_ci_low=float(qa[0]), diff_focus_ci_high=float(qa[1]),
                      diff_contrast=point[k][1], diff_contrast_ci_low=float(qb[0]), diff_contrast_ci_high=float(qb[1]),
                      swing=point[k][0] - point[k][1], swing_ci_low=float(qs[0]), swing_ci_high=float(qs[1]),
                      p_swing=float(p))
    return out, G


def run_contrast(ref, scores, labels_all, ids_focus, ids_contrast, thresholds,
                 gene_df, metrics, n_bootstrap, rng, common_only):
    if ref not in scores:
        print(f"  WARNING: contrast reference '{ref}' not available; skipping")
        return []
    ids_focus, ids_contrast = set(ids_focus), set(ids_contrast)
    if ids_focus & ids_contrast:
        sys.exit("ERROR: --id_list and --contrast_id_list overlap")
    lab = labels_all[labels_all["id"].isin(ids_focus | ids_contrast)]
    if common_only:
        for sdf in scores.values():
            lab = lab[lab["id"].isin(set(sdf["id"]))]
    rows = []
    for name, sdf in scores.items():
        if name == ref:
            continue
        m = (scores[ref].rename(columns={"y_score": "s_ref"})
             .merge(sdf.rename(columns={"y_score": "s_oth"}), on="id")
             .merge(lab, on="id"))
        if gene_df is not None:
            m = m.merge(gene_df, on="id", how="left")
            miss = m["gene"].isna() | (m["gene"] == "")
            m.loc[miss, "gene"] = "NOGENE_" + m.loc[miss, "id"]
            genes = m["gene"].values
        else:
            genes = m["id"].values
        in_a = m["id"].isin(ids_focus).values
        if in_a.all() or (~in_a).all():
            print(f"  [{name}] one of the two subsets is empty after merging; skipped")
            continue
        y = m["y_true"].astype(int).values
        pr = (m["s_ref"].values >= thresholds[ref]).astype(int)
        po = (m["s_oth"].values >= thresholds[name]).astype(int)
        stats, G = contrast_bootstrap(y, pr, po, genes, in_a, metrics, n_bootstrap, rng)
        for k, v in stats.items():
            rows.append(dict(reference=ref, method=name, metric=k, n_focus=int(in_a.sum()),
                             n_contrast=int((~in_a).sum()), n_genes=G, **v))
            print(f"  {ref} - {name} [{k}]: focus {v['diff_focus']:+.4f}, contrast "
                  f"{v['diff_contrast']:+.4f}, swing {v['swing']:+.4f} "
                  f"[{v['swing_ci_low']:+.4f}, {v['swing_ci_high']:+.4f}] p={v['p_swing']:.4g}")
    return rows


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
    p.add_argument("--test_lnc_fasta",   required=True,
                   help="test lncRNA FASTA (ENSG from the headers drives the "
                        "gene-clustered resampling)")
    p.add_argument("--test_pc_fasta",    required=True, help="test mRNA FASTA")
    p.add_argument("--feelNc_tsv",       default=None, help="accepted and ignored")
    p.add_argument("--rnasamba_tsv",     default=None)
    p.add_argument("--orthrus_csv",      default=None)
    p.add_argument("--lncrnabert_csv",   default=None)

    p.add_argument("--cpat_threshold",   type=float, default=None)
    p.add_argument("--cpc2_threshold",   type=float, default=None)

    p.add_argument("--common_subset",    action="store_true",
                   help="score every method on the transcripts ALL methods returned; "
                        "the all-available table is also written (..._allavailable.csv)")
    p.add_argument("--id_list",          default=None,
                   help="text file of transcript IDs (one per line, version "
                        "suffix optional); restrict every method to them "
                        "(e.g. unseen_subset.py output)")
    p.add_argument("--paired_ref",       default="blnc",
                   help="reference method for the paired gene-clustered bootstrap "
                        "of the macro-F1 difference (written to "
                        "benchmark_paired_vs_<ref>.csv); the same transcripts "
                        "and gene resamples are used for both methods")
    p.add_argument("--no_paired",        action="store_true")
    p.add_argument("--contrast_id_list", default=None,
                   help="second ID list (disjoint from --id_list). Adds "
                        "benchmark_contrast_vs_<ref>.csv: for each method, the "
                        "reference-minus-method difference on --id_list (focus) "
                        "and on this list (contrast), and their swing "
                        "(focus - contrast), gene-clustered over both lists")
    p.add_argument("--contrast_metrics", nargs="+", default=["f1", "recall_lnc"],
                   choices=["f1", "recall_lnc", "recall_pc"])
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
    labels_all = labels_df
    if args.contrast_id_list and not args.id_list:
        sys.exit("ERROR: --contrast_id_list needs --id_list (the focus subset)")
    if args.id_list:
        keep = {l.strip().split(".")[0] for l in open(args.id_list) if l.strip()}
        labels_df = labels_df[labels_df["id"].isin(keep)]
        print(f"  --id_list ({args.id_list}): {len(labels_df)} of {len(keep)} "
              f"listed IDs are in the labelled test set")
        if labels_df.empty:
            sys.exit("ERROR: --id_list leaves no labelled transcripts")

    gene_df = load_gene_map([args.test_lnc_fasta, args.test_pc_fasta])
    ng = gene_df.loc[gene_df.gene != "", "gene"].nunique()
    print(f"  gene map: {len(gene_df)} test transcripts, {ng} genes")
    miss = len(set(labels_df["id"]) - set(gene_df["id"]))
    if miss:
        sys.exit(f"ERROR: {miss} labelled transcripts are not in the test "
                 f"FASTAs -- predictions and FASTAs are from different splits")

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
            print(f"  F1={row['f1']:.4f} [{row['f1_gene_ci_low']:.4f}, {row['f1_gene_ci_high']:.4f}]  "
                  f"AUC={row['auc']:.4f} [{row['auc_gene_ci_low']:.4f}, {row['auc_gene_ci_high']:.4f}]")
        return out

    common_info = None
    eval_labels = labels_df
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
        eval_labels = lab_c
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
        id_list=dict(path=args.id_list, sha256_16=sha256_short(args.id_list)) if args.id_list else None,
        n_test_records=int(len(labels_df)),
        n_lnc=int((labels_df.y_true == 0).sum()), n_pc=int((labels_df.y_true == 1).sum()),
        n_genes=int(gene_df.loc[gene_df.id.isin(labels_df.id), "gene"].nunique()),
        ci="gene-clustered percentile bootstrap",
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
    if not args.no_paired:
        print(f"\nPaired gene-clustered bootstrap vs {args.paired_ref} "
              f"({'common subset' if args.common_subset else 'own available set'}):")
        prow = run_paired(args.paired_ref, scores, eval_labels, thresholds,
                          gene_df, args.n_bootstrap, rng)
        if prow:
            pd.DataFrame(prow).to_csv(out_dir / f"benchmark_paired_vs_{args.paired_ref}.csv",
                                      index=False)
    if args.contrast_id_list:
        ids_c = {l.strip().split(".")[0] for l in open(args.contrast_id_list) if l.strip()}
        print(f"\nContrast: focus = {args.id_list} ({labels_df['id'].nunique()} labelled), "
              f"contrast = {args.contrast_id_list} ({len(ids_c)} listed)")
        crow = run_contrast(args.paired_ref, scores, labels_all, set(labels_df["id"]), ids_c,
                            thresholds, gene_df, args.contrast_metrics, args.n_bootstrap,
                            rng, args.common_subset)
        if crow:
            pd.DataFrame(crow).to_csv(out_dir / f"benchmark_contrast_vs_{args.paired_ref}.csv",
                                      index=False)
    print(f"\nSaved: {out_path}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()