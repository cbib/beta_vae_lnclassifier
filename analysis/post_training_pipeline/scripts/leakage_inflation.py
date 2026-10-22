#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
leakage_inflation.py

Evaluate an existing 5-fold ensemble on a test FASTA pair and stratify the
result by overlap with the models' training set (config data.lnc_fasta /
data.pc_fasta):

    seen_record   test transcript is in the training FASTAs  (invalid test)
    seen_gene     transcript unseen, but another isoform of its gene is seen
    novel_gene    no isoform of its gene is in the training FASTAs

Accuracy on novel_gene vs seen_gene is the within-model measure of how much
gene sharing inflates performance. Run it on the OLD test set with the OLD
models.

Place next to evaluate_cv_fold_subgroup.py. Example:
    python leakage_inflation.py \\
        --config configs/beta_vae_subgroup_base_g49.json \\
        --experiment_dir gencode_v49_experiments/beta_vae_subgroup_base_g49 \\
        --lnc_fasta data/split_gencode_49/lnc_test.fa \\
        --pc_fasta  data/split_gencode_49/pc_test.fa \\
        --output_dir gencode_v49_experiments/beta_vae_subgroup_base_g49/leakage_inflation
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO

from configs.load_config import load_config
from models.model_builder import create_model_builder

from evaluate_cv_fold_subgroup import evaluate_test_ensemble

_ENSG = re.compile(r"ENSG\d+")


def _norm(x: str) -> str:
    return str(x).split("|")[0].split(".")[0]


def read_ids(fasta: str) -> dict[str, str]:
    """transcript id (no version) -> gene id (no version, '' if absent)."""
    out = {}
    for rec in SeqIO.parse(fasta, "fasta"):
        m = _ENSG.search(rec.description)
        out[_norm(rec.id)] = m.group(0) if m else ""
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--experiment_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--lnc_fasta", required=True)
    ap.add_argument("--pc_fasta", required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    config = load_config(args.config)
    out = Path(args.output_dir)
    n_folds = config.get("training", "n_folds")

    # 1. ensemble predictions on the test FASTAs (existing routine)
    evaluate_test_ensemble(
        config=config, experiment_dir=Path(args.experiment_dir),
        model_builder=create_model_builder(config),
        test_lnc_fasta=args.lnc_fasta, test_pc_fasta=args.pc_fasta,
        n_folds=n_folds, batch_size=args.batch_size,
        device=torch.device(args.device), output_dir=out,
    )
    pred = pd.read_csv(out / "test_predictions.csv")
    pred["tid"] = pred["transcript_id"].map(_norm)

    # 2. overlap with the training FASTAs
    train = {}
    for f in (config.get("data", "lnc_fasta"), config.get("data", "pc_fasta")):
        train.update(read_ids(f))
    train_genes = {g for g in train.values() if g}
    test = {}
    for f in (args.lnc_fasta, args.pc_fasta):
        test.update(read_ids(f))

    pred["gene"] = pred["tid"].map(test).fillna("")
    if (pred["gene"] == "").any():
        print(f"WARNING: {(pred['gene'] == '').sum()} test records have no ENSG in "
              f"the header; they are excluded from the gene strata.")

    def stratum(r):
        if r.tid in train:
            return "seen_record"
        if r.gene and r.gene in train_genes:
            return "seen_gene"
        return "novel_gene" if r.gene else "unknown"

    pred["stratum"] = pred.apply(stratum, axis=1)
    pred["correct"] = pred["true_label"] == pred["consensus_prediction"]

    rows = []
    for name, d in [("all", pred)] + list(pred.groupby("stratum")):
        n, k = len(d), int(d["correct"].sum())
        lo, hi = wilson(k, n)
        rows.append(dict(stratum=name, n=n, n_lnc=int((d.true_label == "lnc").sum()),
                         n_pc=int((d.true_label == "pc").sum()),
                         accuracy=k / n if n else float("nan"),
                         ci95_low=lo, ci95_high=hi))
    res = pd.DataFrame(rows)
    res.to_csv(out / "leakage_inflation.csv", index=False)
    pred.drop(columns=["tid"]).to_csv(out / "leakage_inflation_records.csv", index=False)
    print("\n" + res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    frac_seen = (pred.stratum == "seen_record").mean()
    if frac_seen > 0.05:
        print(f"\nWARNING: {frac_seen:.1%} of test records are in the models' "
              f"training set. Accuracy on them is not a test result.")
    (out / "leakage_inflation.json").write_text(json.dumps(
        dict(rows=rows, frac_seen_record=float(frac_seen)), indent=2))


if __name__ == "__main__":
    main()