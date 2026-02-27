#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone script to re-run attention extraction from a completed experiment.

Usage:
    python extract_attention.py --experiment_dir <path> --config <path>

Example:
    python extract_attention.py \
        --experiment_dir gencode_v49_experiments/beta_vae_features_attn_g49 \
        --config configs/beta_vae_features_attn_g49_holdout.json

The script reconstructs cv_trainer state from model_paths.csv and the original
config, then calls extract_attention_all_folds().
"""

import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path

from configs.load_config import load_config
from models.model_builder import create_model_builder
from data.cv_utils import load_sequences_in_order, verify_sequence_order
from data.feature_dataset import SequenceFeatureDataset
from trainers.features_attn_trainer import BetaVAEFeaturesAttentionTrainer


def main():
    parser = argparse.ArgumentParser(
        description='Re-run attention extraction from a completed experiment directory.'
    )
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='Path to the experiment output directory (contains model_paths.csv)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the config JSON used for the original training run')
    parser.add_argument('--device', type=str,
                        default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    model_paths_csv = experiment_dir / 'model_paths.csv'

    if not model_paths_csv.exists():
        raise FileNotFoundError(f"model_paths.csv not found in {experiment_dir}. "
                                f"Was cross-validation completed?")

    config = load_config(args.config)

    print("=" * 80)
    print("Attention Extraction — Standalone")
    print("=" * 80)
    print(f"Experiment dir: {experiment_dir}")
    print(f"Config:         {args.config}")
    print(f"Device:         {args.device}")

    # -------------------------------------------------------------------------
    # Reconstruct dataset (needed to recover val splits)
    # -------------------------------------------------------------------------
    # Use lnc_trainval/pc_trainval if present (holdout run), else fall back to
    # the original lnc_fasta/pc_fasta keys
    lnc_fasta = (config.get('data', 'lnc_test_fasta', default=None)
                 and config.get('data', 'lnc_fasta'))
    lnc_fasta = config.get('data', 'lnc_fasta')
    pc_fasta  = config.get('data', 'pc_fasta')

    print(f"\nLoading sequences from:")
    print(f"  lnc: {lnc_fasta}")
    print(f"  pc:  {pc_fasta}")

    all_sequences, labels = load_sequences_in_order(
        lnc_fasta=lnc_fasta,
        pc_fasta=pc_fasta
    )

    full_dataset = SequenceFeatureDataset(
        pc_fasta=pc_fasta,
        lnc_fasta=lnc_fasta,
        te_features_csv=config.get('data', 'te_features_csv'),
        nonb_features_csv=config.get('data', 'nonb_features_csv'),
        te_scaler_path=config.get('data', 'te_scaler'),
        nonb_scaler_path=config.get('data', 'nonb_scaler'),
        max_length=config.get('model', 'max_length')
    )

    verify_sequence_order(all_sequences, labels, full_dataset.sequences)
    print(f"Dataset: {len(full_dataset):,} samples")

    # -------------------------------------------------------------------------
    # Rebuild cv_trainer and restore state from model_paths.csv
    # -------------------------------------------------------------------------
    model_builder = create_model_builder(config)

    cv_trainer = BetaVAEFeaturesAttentionTrainer(
        model_builder=model_builder,
        dataset=full_dataset,
        config=config,
        n_folds=config.get('training', 'n_folds'),
        device=args.device
    )

    # Restore model_paths list and best fold from CSV
    model_paths_df = pd.read_csv(model_paths_csv)
    cv_trainer.model_paths = model_paths_df.to_dict('records')

    best_fold_idx            = model_paths_df['val_loss'].idxmin()
    cv_trainer.best_fold_idx = best_fold_idx
    cv_trainer.best_model_path = model_paths_df.iloc[best_fold_idx]['path']

    print(f"\nRestored {len(cv_trainer.model_paths)} fold models from model_paths.csv")
    print(f"Best fold: {best_fold_idx + 1} "
          f"(val_loss={model_paths_df.iloc[best_fold_idx]['val_loss']:.4f})")

    # Also pass the correct label list to the trainer so extract_attention_all_folds
    # can reconstruct splits without relying on sequence annotations
    cv_trainer._labels = labels   # used by patched extract below

    # -------------------------------------------------------------------------
    # Run extraction — also patch the label-reconstruction bug in-place
    # -------------------------------------------------------------------------
    # The original extract_attention_all_folds reads labels from
    # s.annotations.get('label', 'unknown') which gives KeyError in
    # create_length_stratified_groups. We monkey-patch the dataset's label
    # access so the trainer uses the correct label list instead.

    # Simplest safe fix: attach labels to dataset so the method can find them
    full_dataset._label_list = labels

    attn_dir = experiment_dir / 'fold_attention'

    print(f"\nExtracting attention weights -> {attn_dir}/")
    cv_trainer.extract_attention_all_folds(
        output_dir=attn_dir,
        labels_override=labels        # see note below
    )

    print(f"\nAttention weights saved to: {attn_dir}/")
    print(f"Run interpretability analysis with:")
    print(f"  python analyze_attention.py \\")
    print(f"    --attn_dir {attn_dir} \\")
    print(f"    --output_dir {experiment_dir}/attention_analysis")


if __name__ == '__main__':
    main()