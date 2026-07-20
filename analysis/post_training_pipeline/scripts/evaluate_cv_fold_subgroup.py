#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_cv_fold_subgroup.py

Cross-validation evaluation and hard case extraction for BetaVAESubgroup.

1.  Loads each fold's best checkpoint and evaluates on its validation set.
2.  Aggregates per-sample predictions across folds (each sample is seen
    exactly once as a validation sample, in the fold where it was held out).
3.  Computes hard case scores: error_rate, mean_confidence, std_confidence,
    min_confidence, cross-fold agreement.
4.  Writes:
      evaluation_csvs/all_sample_predictions.csv  — all N samples
      evaluation_csvs/hard_cases.csv              — hard cases only
5.  Optionally evaluates on an independent held-out test set using a
    5-fold ensemble (average predicted probabilities across all folds).

Hard case definition
--------------------
A sample is flagged as a hard case if:
    error_rate > 0          (misclassified in at least one fold)
  OR
    mean_confidence < 0.6   (ensemble probability < 0.6 for the true class)

Usage
-----
python src/evaluate_cv_subgroup.py \\
    --config         configs/beta_vae_subgroup_base_g49.json \\
    --experiment_dir gencode_v49_experiments/beta_vae_subgroup_base_g49 \\
    --output_dir     gencode_v49_experiments/beta_vae_subgroup_base_g49/evaluation_csvs \\
    --lnc_fasta      data/split_gencode_49/lnc_test.fa \\
    --pc_fasta       data/split_gencode_49/pc_test.fa
"""

from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from configs.load_config import load_config
from data.cv_utils import (
    create_length_stratified_groups,
    load_sequences_in_order,
)
from data.gated_feature_dataset import SequenceFeatureDataset
from models.model_builder import create_model_builder
from sklearn.model_selection import StratifiedKFold


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _build_dataset(config, lnc_fasta=None, pc_fasta=None) -> SequenceFeatureDataset:
    nonb2_csv    = config.get("data", "nonb2_csv",              default=None)
    nonb2_scaler = config.get("data", "nonb2_scaler_bank_path", default=None)
    if nonb2_csv is not None and nonb2_scaler is None:
        raise ValueError(
            "Config has data.nonb2_csv set but no data.nonb2_scaler_bank_path."
        )
    return SequenceFeatureDataset(
        lnc_fasta              = lnc_fasta or config.get("data", "lnc_fasta"),
        pc_fasta                = pc_fasta  or config.get("data", "pc_fasta"),
        te_genomic_csv          = config.get("data", "te_genomic_csv"),
        te_processed_csv        = config.get("data", "te_processed_csv",   default=None),
        nonb_genomic_csv        = config.get("data", "nonb_genomic_csv"),
        nonb_processed_csv      = config.get("data", "nonb_processed_csv", default=None),
        nonb2_csv                = nonb2_csv,
        te_scaler_bank_path     = config.get("data", "te_scaler_bank_path"),
        nonb_scaler_bank_path   = config.get("data", "nonb_scaler_bank_path"),
        nonb2_scaler_bank_path  = nonb2_scaler,
        max_length               = config.get("model", "max_length"),
    )


def _get_splits(config, all_sequences, labels):
    strat_groups = create_length_stratified_groups(
        all_sequences, labels,
        n_bins=config.get("training", "n_bins", default=5),
    )
    skf = StratifiedKFold(
        n_splits     = config.get("training", "n_folds"),
        shuffle      = True,
        random_state = config.get("training", "random_state", default=42),
    )
    return list(skf.split(all_sequences, strat_groups))


def _forward_batch(model, batch, device):
    """Forward pass, resolving nonb2 from batch dynamically when present."""
    seq = batch["sequence"].to(device)
    fwd_kw = dict(
        te_genomic     = batch["te_genomic"].to(device),
        te_processed   = batch["te_processed"].to(device),
        nonb_genomic   = batch["nonb_genomic"].to(device),
        nonb_processed = batch["nonb_processed"].to(device),
    )
    if "nonb2" in batch:
        fwd_kw["nonb2"] = batch["nonb2"].to(device)
    out = model(seq, deterministic=True, **fwd_kw)
    return torch.softmax(out["logits"], dim=1).cpu().numpy()


# ---------------------------------------------------------------------------
# Per-fold evaluation — returns only scalars, no stored probs array
# ---------------------------------------------------------------------------

def evaluate_fold_into_accumulator(
    model,
    loader:       DataLoader,
    val_idx:      np.ndarray,
    device:       torch.device,
    accumulator:  dict,
) -> dict:
    """
    Run inference on one fold's validation set and accumulate per-sample
    running totals directly into `accumulator` (a defaultdict of lists).

    Only scalar per-sample values are stored (prediction, confidence, label,
    transcript_id) — the full (N, 2) probs array is never accumulated,
    keeping memory O(N) rather than O(N × n_folds).

    Returns fold-level metrics dict.
    """
    model.eval()
    all_probs, all_labels, all_ids = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating", leave=False):
            probs = _forward_batch(model, batch, device)
            all_probs.append(probs)
            all_labels.append(batch["label"].numpy())
            all_ids.extend(batch["transcript_id"])

    probs       = np.vstack(all_probs)       # (N_val, 2) — temporary
    labels      = np.concatenate(all_labels)
    predictions = probs.argmax(axis=1)
    confidences = probs.max(axis=1)

    # Fold-level metrics
    acc                       = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0,
    )

    # Accumulate per-sample scalars only — then free the probs array
    for i, sample_idx in enumerate(val_idx):
        d = accumulator[int(sample_idx)]
        d["predictions"].append(int(predictions[i]))
        d["confidences"].append(float(confidences[i]))
        d["labels"].append(int(labels[i]))
        d["ids"].append(all_ids[i])

    # Explicitly free the large arrays before returning
    del probs, all_probs
    gc.collect()

    return dict(
        accuracy  = float(acc),
        precision = float(precision),
        recall    = float(recall),
        f1        = float(f1),
    )


# ---------------------------------------------------------------------------
# Accumulator → hard case CSV
# ---------------------------------------------------------------------------

def accumulator_to_df(
    accumulator:    dict,
    dataset:        SequenceFeatureDataset,
    hard_conf_thresh: float = 0.6,
) -> pd.DataFrame:
    """
    Convert per-sample running totals into the predictions DataFrame.

    transcript_id is taken from the original FASTA sequence record
    (dataset.sequences[sample_idx].id) rather than from the batch's
    normalised index, preserving the version suffix (e.g. ENST00000810508.1)
    so that downstream benchmark joining against external tool CSVs works
    without any ID normalisation on either side.
    """
    idx_to_label = {0: "lnc", 1: "pc"}
    records = []

    for sample_idx in sorted(accumulator.keys()):
        d = accumulator[sample_idx]

        predictions   = np.array(d["predictions"])
        confidences   = np.array(d["confidences"])
        true_label    = int(d["labels"][0])
        # Use original FASTA record ID to preserve version suffix
        transcript_id = str(dataset.sequences[sample_idx].id).split("|")[0]
        seq_length    = len(str(dataset.sequences[sample_idx].seq))

        consensus_pred = int(np.bincount(predictions).argmax())
        error_rate     = float((predictions != true_label).mean())
        mean_conf      = float(confidences.mean())
        std_conf       = float(confidences.std()) if len(confidences) > 1 else 0.0
        min_conf       = float(confidences.min())
        agreement      = float((predictions == consensus_pred).mean())
        is_hard        = (error_rate > 0) or (mean_conf < hard_conf_thresh)

        records.append(dict(
            sample_idx           = sample_idx,
            transcript_id        = transcript_id,
            true_label           = idx_to_label[true_label],
            consensus_prediction = idx_to_label[consensus_pred],
            sequence_length      = seq_length,
            n_folds              = len(predictions),
            error_rate           = error_rate,
            mean_confidence      = mean_conf,
            std_confidence       = std_conf,
            min_confidence       = min_conf,
            agreement            = agreement,
            is_hard_case         = bool(is_hard),
        ))

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Independent test set ensemble
# ---------------------------------------------------------------------------

def evaluate_test_ensemble(
    config,
    experiment_dir:  Path,
    model_builder,
    test_lnc_fasta:  str,
    test_pc_fasta:   str,
    n_folds:         int,
    batch_size:      int,
    device:          torch.device,
    output_dir:      Path,
    hard_conf_thresh: float = 0.6,
) -> dict:
    """
    Ensemble evaluation on a held-out test set.

    Accumulates sum of softmax probabilities across folds rather than
    stacking the full (F, N, 2) array — memory is O(N) regardless of
    the number of folds.
    """
    print("\n" + "=" * 65)
    print("Independent test set evaluation (ensemble)")
    print("=" * 65)

    test_dataset = _build_dataset(
        config, lnc_fasta=test_lnc_fasta, pc_fasta=test_pc_fasta
    )
    print(f"  Test samples: {len(test_dataset):,}")

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=1,
    )

    n_samples  = len(test_dataset)
    sum_probs  = np.zeros((n_samples, 2), dtype=np.float64)
    # Per-sample fold confidence tracking: stored as list-of-lists to avoid
    # stacking the full (F, N) array in memory
    fold_conf_lists = [[] for _ in range(n_samples)]
    fold_pred_lists = [[] for _ in range(n_samples)]
    labels_arr = None
    all_ids    = []
    cv_rows    = []

    for fold_idx in range(n_folds):
        ckpt_path = experiment_dir / "models" / f"fold_{fold_idx}_best.pt"
        ckpt      = torch.load(ckpt_path, map_location=device)
        model     = model_builder()
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.to(device)
        model.eval()
        print(f"  Fold {fold_idx}: epoch={ckpt['epoch']}  "
              f"val_acc={ckpt['val_acc']:.4f}")
        cv_rows.append(dict(fold=fold_idx, val_acc=float(ckpt["val_acc"])))

        fold_probs_list, fold_labels_list = [], []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"  Fold {fold_idx}", leave=False):
                probs = _forward_batch(model, batch, device)
                fold_probs_list.append(probs)
                fold_labels_list.append(batch["label"].numpy())
                if fold_idx == 0:
                    all_ids.extend(batch["transcript_id"])

        fp         = np.vstack(fold_probs_list)   # (N, 2) — temporary per fold
        labels_arr = np.concatenate(fold_labels_list)
        sum_probs += fp

        # Accumulate per-sample confidence/prediction scalars
        fold_preds = fp.argmax(axis=1)
        fold_confs = fp.max(axis=1)
        for i in range(n_samples):
            fold_conf_lists[i].append(float(fold_confs[i]))
            fold_pred_lists[i].append(int(fold_preds[i]))

        # Free fold arrays immediately
        del fp, fold_probs_list, fold_preds, fold_confs
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    avg_probs   = sum_probs / n_folds
    predictions = avg_probs.argmax(axis=1)

    acc                       = accuracy_score(labels_arr, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_arr, predictions, average="macro", zero_division=0,
    )
    cm = confusion_matrix(labels_arr, predictions)

    print(f"\n  Test ensemble results ({n_folds} folds):")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {precision:.4f}")
    print(f"    Recall    : {recall:.4f}")
    print(f"    F1        : {f1:.4f}")

    metrics = dict(
        accuracy=float(acc), precision=float(precision),
        recall=float(recall), f1=float(f1),
        confusion_matrix=cm.tolist(),
        n_samples=int(n_samples), n_folds=n_folds,
    )

    idx_to_label = {0: "lnc", 1: "pc"}
    records = []
    for i in range(n_samples):
        prob       = avg_probs[i]
        pred       = int(predictions[i])
        true       = int(labels_arr[i])
        confidence = float(prob.max())
        error_rate = float(pred != true)
        is_hard    = error_rate > 0 or confidence < hard_conf_thresh
        f_confs    = np.array(fold_conf_lists[i])
        f_preds    = np.array(fold_pred_lists[i])

        records.append(dict(
            sample_idx           = i,
            transcript_id        = str(test_dataset.sequences[i].id).split("|")[0],
            true_label           = idx_to_label[true],
            consensus_prediction = idx_to_label[pred],
            sequence_length      = len(str(test_dataset.sequences[i].seq)),
            n_folds              = n_folds,
            error_rate           = error_rate,
            mean_confidence      = confidence,
            std_confidence       = float(f_confs.std()),
            min_confidence       = float(f_confs.min()),
            agreement            = float((f_preds == pred).mean()),
            is_hard_case         = bool(is_hard),
        ))

    output_dir.mkdir(parents=True, exist_ok=True)
    df      = pd.DataFrame(records)
    hard_df = df[df["is_hard_case"]]
    df.to_csv(output_dir / "test_predictions.csv", index=False)
    hard_df.to_csv(output_dir / "test_hard_cases.csv", index=False)
    print(f"\n  test_predictions.csv : {len(df):,} samples")
    print(f"  test_hard_cases.csv  : {len(hard_df):,} hard cases "
          f"({100*len(hard_df)/len(df):.1f}%)")

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CV evaluation and hard case extraction for BetaVAESubgroup"
    )
    parser.add_argument("--config",           required=True)
    parser.add_argument("--experiment_dir",   required=True)
    parser.add_argument("--output_dir",       required=True)
    parser.add_argument("--device",
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size",       type=int, default=256)
    parser.add_argument("--hard_conf_thresh", type=float, default=0.6)
    parser.add_argument("--lnc_fasta",        default=None,
                        help="lncRNA FASTA for held-out test ensemble (optional)")
    parser.add_argument("--pc_fasta",         default=None,
                        help="mRNA FASTA for held-out test ensemble (optional)")
    args = parser.parse_args()

    config     = load_config(args.config)
    exp_dir    = Path(args.experiment_dir)
    output_dir = Path(args.output_dir)
    device     = torch.device(args.device)
    n_folds    = config.get("training", "n_folds")
    bs         = args.batch_size or config.get("training", "batch_size")
    nw         = config.get("training", "num_workers", default=1)

    print("=" * 65)
    print("BetaVAESubgroup — CV Evaluation + Hard Case Extraction")
    print("=" * 65)
    print(f"Config      : {args.config}")
    print(f"Experiment  : {exp_dir}")
    print(f"Output      : {output_dir}")
    print(f"Device      : {device}")
    print(f"Hard thresh : {args.hard_conf_thresh}")
    print("=" * 65)

    print("\nLoading sequences...")
    all_sequences, labels = load_sequences_in_order(
        lnc_fasta = config.get("data", "lnc_fasta"),
        pc_fasta  = config.get("data", "pc_fasta"),
    )
    print(f"  {len(all_sequences):,} sequences loaded")

    print("Building dataset...")
    dataset = _build_dataset(config)
    print(f"  Dataset: {len(dataset):,} samples")

    splits        = _get_splits(config, all_sequences, labels)
    model_builder = create_model_builder(config)

    # Single shared accumulator — populated fold by fold, never holding
    # more than one fold's probs array in memory at a time
    accumulator: dict = defaultdict(lambda: dict(
        predictions=[], confidences=[], labels=[], ids=[]
    ))

    print("\n" + "=" * 65)
    print("Per-fold evaluation")
    print("=" * 65)

    cv_metrics = []

    for fold_idx, (_, val_idx) in enumerate(splits):
        print(f"\nFold {fold_idx}  ({len(val_idx):,} val samples)")

        ckpt_path = exp_dir / "models" / f"fold_{fold_idx}_best.pt"
        if not ckpt_path.exists():
            print(f"  ERROR: checkpoint not found: {ckpt_path}")
            continue

        ckpt  = torch.load(ckpt_path, map_location=device)
        model = model_builder()
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.to(device)
        print(f"  Loaded epoch={ckpt['epoch']}  val_acc={ckpt['val_acc']:.4f}")

        loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=bs, shuffle=False,
            num_workers=nw, pin_memory=True,
        )

        fold_metrics = evaluate_fold_into_accumulator(
            model, loader, val_idx, device, accumulator
        )
        fold_metrics["fold"] = fold_idx
        cv_metrics.append(fold_metrics)

        print(f"  acc={fold_metrics['accuracy']:.4f}  "
              f"prec={fold_metrics['precision']:.4f}  "
              f"rec={fold_metrics['recall']:.4f}  "
              f"f1={fold_metrics['f1']:.4f}")

        # Free model before loading the next fold
        del model, ckpt
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── CV summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("CV summary")
    print("=" * 65)
    cv_df = pd.DataFrame(cv_metrics)
    for col in ["accuracy", "precision", "recall", "f1"]:
        print(f"  {col:<12}: {cv_df[col].mean():.4f} ± {cv_df[col].std():.4f}")

    # ── Hard case aggregation ─────────────────────────────────────────────
    print("\nAggregating hard cases...")
    all_preds_df = accumulator_to_df(
        accumulator, dataset,
        hard_conf_thresh=args.hard_conf_thresh,
    )

    n_hard  = all_preds_df["is_hard_case"].sum()
    n_total = len(all_preds_df)
    print(f"  Total samples : {n_total:,}")
    print(f"  Hard cases    : {n_hard:,} ({100*n_hard/n_total:.1f}%)")
    for lab in ["lnc", "pc"]:
        n_c   = (all_preds_df["true_label"] == lab).sum()
        n_c_h = ((all_preds_df["true_label"] == lab) &
                 all_preds_df["is_hard_case"]).sum()
        print(f"    {lab}: {n_c_h:,} / {n_c:,} hard ({100*n_c_h/n_c:.1f}%)")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_preds_df.to_csv(output_dir / "all_sample_predictions.csv", index=False)
    all_preds_df[all_preds_df["is_hard_case"]].to_csv(
        output_dir / "hard_cases.csv", index=False
    )
    cv_df.to_csv(output_dir / "cv_fold_metrics.csv", index=False)
    print(f"\n  all_sample_predictions.csv → {output_dir}")
    print(f"  hard_cases.csv             → {output_dir}")

    # ── Optional test set ensemble ────────────────────────────────────────
    if args.lnc_fasta and args.pc_fasta:
        evaluate_test_ensemble(
            config          = config,
            experiment_dir  = exp_dir,
            model_builder   = model_builder,
            test_lnc_fasta  = args.lnc_fasta,
            test_pc_fasta   = args.pc_fasta,
            n_folds         = n_folds,
            batch_size      = bs,
            device          = device,
            output_dir      = output_dir,
            hard_conf_thresh= args.hard_conf_thresh,
        )

    print("\n" + "=" * 65)
    print("Done.")
    print("=" * 65)


if __name__ == "__main__":
    main()