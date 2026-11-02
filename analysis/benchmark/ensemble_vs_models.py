#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis/benchmark/ensemble_vs_models.py

Macro F1 of each fold model, the 5-model ensemble and
lncRNA-BERT (3-mer, zero-shot) on the SAME test transcripts, scored with the
benchmark's macro F1 (threshold 0.5 on P(pc)). Percentile bootstrap CIs and
paired bootstrap differences (same resamples for every method).

Inputs
------
--fold_probs   test_fold_probs.csv from evaluate_cv_fold_subgroup.py
--lncrnabert   lncRNA-BERT predictions CSV (transcript_id, prob_pc)

python ensemble_vs_models.py --fold_probs EXP/evaluation_csvs/test_fold_probs.csv \
    --lncrnabert PATH/lncrnabert.csv --out_dir OUT \
    --test_lnc_fasta lnc_test.fa --test_pc_fasta pc_test.fa

With both test FASTAs the bootstrap resamples genes (ENSG from headers), not
isoforms; without them it falls back to isoform-level resampling and warns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from compute_benchmark_ci import LABEL_MAP, load_gene_map, load_lncrnabert, norm_id


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    """Macro F1 over classes {0, 1}; equals sklearn f1_score(average='macro',
    zero_division=0)."""
    tp = np.array([np.sum((y == c) & (pred == c)) for c in (0, 1)], dtype=float)
    fp = np.array([np.sum((y != c) & (pred == c)) for c in (0, 1)], dtype=float)
    fn = np.array([np.sum((y == c) & (pred != c)) for c in (0, 1)], dtype=float)
    den = 2 * tp + fp + fn
    present = den > 0                    # classes seen in y or pred, as sklearn
    return float(np.mean(2 * tp[present] / den[present])) if present.any() else 0.0


def paired_bootstrap(y: np.ndarray, preds: dict, n_boot: int, seed: int,
                     genes: np.ndarray | None = None) -> tuple[dict, dict]:
    """Per-method macro F1 samples, same resample for all methods.
    genes given -> resample genes (all isoforms of a drawn gene), else records."""
    rng = np.random.default_rng(seed)
    n = len(y)
    names = list(preds)
    boot = {k: np.empty(n_boot) for k in names}
    if genes is not None:
        codes, uniq = pd.factorize(genes)
        order = np.argsort(codes, kind="stable")
        bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    for b in range(n_boot):
        if genes is None:
            idx = rng.integers(0, n, size=n)
        else:
            draw = rng.integers(0, len(uniq), size=len(uniq))
            idx = np.concatenate([order[bounds[g]:bounds[g + 1]] for g in draw])
        for k in names:
            boot[k][b] = macro_f1(y[idx], preds[k][idx])
    return boot, {k: macro_f1(y, preds[k]) for k in names}


def q(x, a):
    return float(np.quantile(x, a))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold_probs", required=True)
    ap.add_argument("--lncrnabert", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--test_lnc_fasta", default=None)
    ap.add_argument("--test_pc_fasta", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--n_bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.fold_probs)
    df["id"] = norm_id(df["transcript_id"])
    df["y"] = df["true_label"].astype(str).str.lower().map(LABEL_MAP)
    df = df.dropna(subset=["y"]).drop_duplicates("id")
    fold_cols = sorted(c for c in df.columns if c.startswith("p_pc_fold"))

    scores = {f"fold{c[len('p_pc_fold'):]}": df.set_index("id")[c] for c in fold_cols}
    scores["ensemble"] = df.set_index("id")["p_pc_ensemble"]
    if args.lncrnabert:
        lb = load_lncrnabert(args.lncrnabert).drop_duplicates("id").set_index("id")["y_score"]
        scores["lncRNA-BERT"] = lb

    ids = df.set_index("id").index
    for s in scores.values():
        ids = ids.intersection(s.index)
    print(f"Common test transcripts: {len(ids):,} of {len(df):,}")
    y = df.set_index("id").loc[ids, "y"].astype(int).to_numpy()
    preds = {k: (v.loc[ids].to_numpy() >= args.threshold).astype(int)
             for k, v in scores.items()}

    genes = None
    if args.test_lnc_fasta and args.test_pc_fasta:
        gm = load_gene_map([args.test_lnc_fasta, args.test_pc_fasta]).set_index("id")["gene"]
        g = gm.reindex(ids)
        if g.isna().any():
            raise SystemExit(f"{int(g.isna().sum())} transcripts missing from the test FASTAs "
                             "-- fold probabilities and FASTAs come from different splits")
        g = g.where(g != "", "NOGENE_" + pd.Series(ids, index=ids))
        genes = g.to_numpy()
        print(f"Gene-clustered bootstrap over {len(set(genes)):,} genes")
    else:
        print("WARNING: no test FASTAs -> record-level bootstrap (isoforms treated as independent)")
    boot, point = paired_bootstrap(y, preds, args.n_bootstrap, args.seed, genes)
    rows = [dict(method=k, n=len(y), macro_f1=point[k],
                 ci_low=q(boot[k], 0.025), ci_high=q(boot[k], 0.975)) for k in preds]
    table = pd.DataFrame(rows)

    singles = [k for k in preds if k.startswith("fold")]
    diffs = []

    def add(label, point_d, boot_d):
        diffs.append(dict(comparison=label, diff=point_d,
                          ci_low=q(boot_d, 0.025), ci_high=q(boot_d, 0.975),
                          frac_boot_le_0=float(np.mean(boot_d <= 0))))

    mean_single_pt = np.mean([point[k] for k in singles])
    mean_single_bt = np.mean([boot[k] for k in singles], axis=0)
    add("ensemble - mean(single models)", point["ensemble"] - mean_single_pt,
        boot["ensemble"] - mean_single_bt)
    for k in singles:
        add(f"ensemble - {k}", point["ensemble"] - point[k], boot["ensemble"] - boot[k])
    if "lncRNA-BERT" in preds:
        add("ensemble - lncRNA-BERT", point["ensemble"] - point["lncRNA-BERT"],
            boot["ensemble"] - boot["lncRNA-BERT"])
        add("mean(single models) - lncRNA-BERT", mean_single_pt - point["lncRNA-BERT"],
            mean_single_bt - boot["lncRNA-BERT"])
    dtab = pd.DataFrame(diffs)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "ensemble_vs_models_f1.csv", index=False)
    dtab.to_csv(out / "ensemble_vs_models_diffs.csv", index=False)
    pd.set_option("display.width", 140)
    print("\nMacro F1 (95% bootstrap CI)\n", table.round(4).to_string(index=False))
    print("\nPaired differences\n", dtab.round(4).to_string(index=False))


if __name__ == "__main__":
    main()