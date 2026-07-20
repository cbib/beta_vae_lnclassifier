#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_cv_feature_only.py

CV evaluation and hard case extraction for FeatureOnlyClassifier.

Produces the same per-sample predictions CSV schema as
evaluate_cv_subgroup.

Usage
-----
# CV evaluation
python src/evaluate_cv_feature_only.py \\
    --config         configs/beta_vae_feature_only_base_g49.json \\
    --experiment_dir gencode_v49_experiments/feature_only_base_g49 \\
    --output_dir     gencode_v49_experiments/feature_only_base_g49/evaluation_csvs

# With test set ensemble
python src/evaluate_cv_feature_only.py \\
    --config          configs/beta_vae_feature_only_base_g49.json \\
    --experiment_dir  gencode_v49_experiments/feature_only_base_g49 \\
    --output_dir      gencode_v49_experiments/feature_only_base_g49/evaluation_csvs \\
    --test_lnc_fasta  data/split_gencode_49/lnc_test.fa \\
    --test_pc_fasta   data/split_gencode_49/pc_test.fa
"""

from __future__ import annotations

import argparse
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
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from configs.load_config import load_config
from data.cv_utils import (
    create_length_stratified_groups,
    load_sequences_in_order,
)
from data.gated_feature_dataset import SequenceFeatureDataset
from data.feature_registry import REGISTRY
from models.feature_only_classifier import FeatureOnlyClassifier


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_feature_only_model(config) -> FeatureOnlyClassifier:
    return FeatureOnlyClassifier(
        num_classes       = config.get("model", "num_classes",       default=2),
        d_proj            = config.get("model", "d_proj",            default=64),
        fusion_dropout    = config.get("model", "fusion_dropout",    default=0.1),
        attn_heads        = config.get("model", "attn_heads",        default=4),
        attn_dropout      = config.get("model", "attn_dropout",      default=0.1),
        attn_temperature  = config.get("model", "attn_temperature",  default=4.0),
        attn_mode         = config.get("model", "attn_mode",         default="standard"),
        classifier_hidden = config.get("model", "classifier_hidden", default=[128]),
        dropout_rate      = config.get("model", "dropout_rate",      default=0.3),
        registry          = REGISTRY,
    )


# ---------------------------------------------------------------------------
# Dataset construction (registry-aware: passes nonb2 when configured)
# ---------------------------------------------------------------------------

def _build_dataset(config, lnc_fasta=None, pc_fasta=None) -> SequenceFeatureDataset:
    nonb2_csv    = config.get("data", "nonb2_csv",              default=None)
    nonb2_scaler = config.get("data", "nonb2_scaler_bank_path", default=None)
    if nonb2_csv is not None and nonb2_scaler is None:
        raise ValueError(
            "Config has data.nonb2_csv set but no data.nonb2_scaler_bank_path."
        )
    return SequenceFeatureDataset(
        lnc_fasta               = lnc_fasta or config.get("data", "lnc_fasta"),
        pc_fasta                 = pc_fasta  or config.get("data", "pc_fasta"),
        te_genomic_csv           = config.get("data", "te_genomic_csv"),
        te_processed_csv         = config.get("data", "te_processed_csv",   default=None),
        nonb_genomic_csv         = config.get("data", "nonb_genomic_csv"),
        nonb_processed_csv       = config.get("data", "nonb_processed_csv", default=None),
        nonb2_csv                 = nonb2_csv,
        te_scaler_bank_path      = config.get("data", "te_scaler_bank_path"),
        nonb_scaler_bank_path    = config.get("data", "nonb_scaler_bank_path"),
        nonb2_scaler_bank_path   = nonb2_scaler,
        max_length                = config.get("model", "max_length"),
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


# ---------------------------------------------------------------------------
# Inference — feature-only forward (no sequence input)
# ---------------------------------------------------------------------------

def _forward_batch(model, batch, device) -> np.ndarray:
    """Forward pass ignoring sequence input, resolving nonb2 when present."""
    fwd_kw = dict(
        te_genomic     = batch["te_genomic"].to(device),
        te_processed   = batch["te_processed"].to(device),
        nonb_genomic   = batch["nonb_genomic"].to(device),
        nonb_processed = batch["nonb_processed"].to(device),
    )
    if "nonb2" in batch:
        fwd_kw["nonb2"] = batch["nonb2"].to(device)

    out = model(deterministic=True, **fwd_kw)
    return torch.softmax(out["logits"], dim=1).cpu().numpy()


def evaluate_fold(model, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_probs, all_labels, all_ids = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating", leave=False):
            all_probs.append(_forward_batch(model, batch, device))
            all_labels.append(batch["label"].numpy())
            all_ids.extend(batch["transcript_id"])

    probs       = np.vstack(all_probs)
    labels      = np.concatenate(all_labels)
    predictions = probs.argmax(axis=1)
    confidences = probs.max(axis=1)

    acc                       = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0,
    )
    return dict(
        probs=probs, labels=labels, predictions=predictions,
        confidences=confidences, transcript_ids=all_ids,
        accuracy=float(acc), precision=float(precision),
        recall=float(recall), f1=float(f1),
    )


# ---------------------------------------------------------------------------
# Aggregation → predictions CSV
# ---------------------------------------------------------------------------

def aggregate_predictions(
    fold_eval_results: list[dict],
    splits:            list,
    dataset:           SequenceFeatureDataset,
    hard_conf_thresh:  float = 0.6,
) -> pd.DataFrame:
    idx_to_label = {0: "lnc", 1: "pc"}
    sample_data: dict[int, dict] = defaultdict(lambda: dict(
        predictions=[], confidences=[], labels=[], ids=[],
    ))

    for fold_result, (_, val_idx) in zip(fold_eval_results, splits):
        preds  = fold_result["predictions"]
        confs  = fold_result["confidences"]
        labels = fold_result["labels"]
        ids    = fold_result["transcript_ids"]
        for i, sample_idx in enumerate(val_idx):
            d = sample_data[int(sample_idx)]
            d["predictions"].append(int(preds[i]))
            d["confidences"].append(float(confs[i]))
            d["labels"].append(int(labels[i]))
            d["ids"].append(ids[i])

    records = []
    for sample_idx in sorted(sample_data.keys()):
        d = sample_data[sample_idx]
        predictions = np.array(d["predictions"])
        confidences = np.array(d["confidences"])
        true_label  = int(d["labels"][0])
        transcript_id = str(dataset.sequences[sample_idx].id).split("|")[0]
        seq_length = len(str(dataset.sequences[sample_idx].seq))

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
# Test set ensemble
# ---------------------------------------------------------------------------

def evaluate_test_ensemble(
    config, exp_dir: Path, n_folds: int, batch_size: int,
    test_lnc_fasta: str, test_pc_fasta: str,
    device: torch.device, output_dir: Path,
    hard_conf_thresh: float = 0.6,
) -> dict:
    print("\n" + "=" * 65)
    print("Test set ensemble (feature-only)")
    print("=" * 65)

    test_dataset = _build_dataset(config, lnc_fasta=test_lnc_fasta, pc_fasta=test_pc_fasta)
    print(f"  Test samples: {len(test_dataset):,}")

    loader = DataLoader(test_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=1)

    n          = len(test_dataset)
    sum_probs  = np.zeros((n, 2), dtype=np.float64)
    fold_stack = []
    labels_arr = None
    all_ids    = []

    for fold_idx in range(n_folds):
        ckpt  = torch.load(exp_dir / "models" / f"fold_{fold_idx}_best.pt",
                           map_location=device)
        model = build_feature_only_model(config)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.to(device)
        model.eval()
        print(f"  Fold {fold_idx}: epoch={ckpt['epoch']}  "
              f"val_acc={ckpt['val_acc']:.4f}")

        fold_probs, fold_labels = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"  Fold {fold_idx}", leave=False):
                fold_probs.append(_forward_batch(model, batch, device))
                fold_labels.append(batch["label"].numpy())
                if fold_idx == 0:
                    all_ids.extend(batch["transcript_id"])

        fp           = np.vstack(fold_probs)
        fold_stack.append(fp)
        sum_probs   += fp
        labels_arr   = np.concatenate(fold_labels)

    avg_probs   = sum_probs / n_folds
    predictions = avg_probs.argmax(axis=1)

    acc                       = accuracy_score(labels_arr, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_arr, predictions, average="macro", zero_division=0,
    )
    print(f"\n  Accuracy  : {acc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")

    idx_to_label     = {0: "lnc", 1: "pc"}
    fold_probs_stack = np.stack(fold_stack, axis=0)     # (F, N, 2)
    fold_confidences = fold_probs_stack.max(axis=2)     # (F, N)

    records = []
    for i in range(n):
        prob       = avg_probs[i]
        pred       = int(predictions[i])
        true       = int(labels_arr[i])
        confidence = float(prob.max())
        error_rate = float(pred != true)
        is_hard    = error_rate > 0 or confidence < hard_conf_thresh

        records.append(dict(
            sample_idx           = i,
            transcript_id        = all_ids[i],
            true_label           = idx_to_label[true],
            consensus_prediction = idx_to_label[pred],
            sequence_length      = len(str(test_dataset.sequences[i].seq)),
            n_folds              = n_folds,
            error_rate           = error_rate,
            mean_confidence      = confidence,
            std_confidence       = float(fold_confidences[:, i].std()),
            min_confidence       = float(fold_confidences[:, i].min()),
            agreement            = float(
                (fold_probs_stack[:, i].argmax(axis=1) == pred).mean()
            ),
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

    metrics = dict(accuracy=float(acc), precision=float(precision),
                   recall=float(recall), f1=float(f1), n_samples=int(n))
    with open(output_dir / "test_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CV evaluation for FeatureOnlyClassifier"
    )
    parser.add_argument("--config",           required=True)
    parser.add_argument("--experiment_dir",   required=True)
    parser.add_argument("--output_dir",       required=True)
    parser.add_argument("--device",
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size",       type=int, default=512)
    parser.add_argument("--hard_conf_thresh", type=float, default=0.6)
    parser.add_argument("--lnc_fasta",   default=None)
    parser.add_argument("--pc_fasta",    default=None)
    args = parser.parse_args()

    config     = load_config(args.config)
    exp_dir    = Path(args.experiment_dir)
    output_dir = Path(args.output_dir)
    device     = torch.device(args.device)
    n_folds    = config.get("training", "n_folds")
    bs         = args.batch_size
    nw         = config.get("training", "num_workers", default=1)

    print("=" * 65)
    print("FeatureOnlyClassifier — CV Evaluation")
    print("=" * 65)
    print(f"Experiment   : {exp_dir}")
    print(f"Output       : {output_dir}")
    print(f"Hard thresh  : {args.hard_conf_thresh}")
    print("=" * 65)
    print("Feature blocks (from registry):")
    for block_name in REGISTRY.block_names:
        dim = REGISTRY.block_dim(block_name)
        print(f"  {block_name:8s}: {dim:4d} features")
    print(f"  TOTAL TOKENS: {REGISTRY.total_tokens}")
    print("=" * 65)

    print("\nLoading sequences and dataset...")
    all_sequences, labels = load_sequences_in_order(
        lnc_fasta = config.get("data", "lnc_fasta"),
        pc_fasta  = config.get("data", "pc_fasta"),
    )
    dataset = _build_dataset(config)
    splits  = _get_splits(config, all_sequences, labels)
    print(f"  {len(dataset):,} samples  |  {n_folds} folds")

    print("\nPer-fold evaluation:")
    fold_results = []
    cv_rows      = []

    for fold_idx, (_, val_idx) in enumerate(splits):
        ckpt_path = exp_dir / "models" / f"fold_{fold_idx}_best.pt"
        if not ckpt_path.exists():
            print(f"  Fold {fold_idx}: checkpoint not found — skipping")
            continue

        ckpt  = torch.load(ckpt_path, map_location=device)
        model = build_feature_only_model(config)
        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
        safe_prefixes = ("lambda_adv", "_sg_scales", "token_sa", "length_head")
        real_missing  = [k for k in missing
                         if not any(k.startswith(p) for p in safe_prefixes)]
        if real_missing:
            print(f"  WARNING — unexpected missing keys: {real_missing}")
        model.to(device)
        print(f"\n  Fold {fold_idx}: epoch={ckpt['epoch']}  "
              f"val_acc={ckpt['val_acc']:.4f}  "
              f"({len(val_idx):,} val samples)")

        loader = DataLoader(
            Subset(dataset, val_idx), batch_size=bs,
            shuffle=False, num_workers=nw, pin_memory=True,
        )
        result         = evaluate_fold(model, loader, device)
        result["fold"] = fold_idx
        fold_results.append(result)

        print(f"  acc={result['accuracy']:.4f}  f1={result['f1']:.4f}")
        cv_rows.append(dict(fold=fold_idx, accuracy=result["accuracy"],
                            precision=result["precision"],
                            recall=result["recall"], f1=result["f1"]))

    cv_df = pd.DataFrame(cv_rows)
    print("\n" + "=" * 65)
    print("CV summary (feature-only)")
    print("=" * 65)
    for col in ["accuracy", "precision", "recall", "f1"]:
        print(f"  {col:<12}: {cv_df[col].mean():.4f} ± {cv_df[col].std():.4f}")

    print("\nAggregating predictions...")
    preds_df = aggregate_predictions(
        fold_results, splits, dataset, args.hard_conf_thresh,
    )
    n_hard = preds_df["is_hard_case"].sum()
    print(f"  Hard cases: {n_hard:,} / {len(preds_df):,} "
          f"({100*n_hard/len(preds_df):.1f}%)")

    output_dir.mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(output_dir / "all_sample_predictions.csv", index=False)
    preds_df[preds_df["is_hard_case"]].to_csv(
        output_dir / "hard_cases.csv", index=False)
    cv_df.to_csv(output_dir / "cv_fold_metrics.csv", index=False)
    print(f"  Saved → {output_dir}/")

    if args.lnc_fasta and args.pc_fasta:
        evaluate_test_ensemble(
            config         = config,
            exp_dir        = exp_dir,
            n_folds        = n_folds,
            batch_size     = bs,
            test_lnc_fasta = args.lnc_fasta,
            test_pc_fasta  = args.pc_fasta,
            device         = device,
            output_dir     = output_dir,
            hard_conf_thresh = args.hard_conf_thresh,
        )

    print("\n" + "=" * 65)
    print("Done.")
    print("=" * 65)


if __name__ == "__main__":
    main()