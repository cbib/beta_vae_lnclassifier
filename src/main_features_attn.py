#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for β-VAE with cross-attention genomic features training.

Changes vs main_features.py:
  - Imports BetaVAEFeaturesAttentionTrainer instead of BetaVAEFeaturesTrainer
  - Calls extract_attention_all_folds() after CV to save per-fold attn weights
  - Config expects attn_dropout (optional, defaults apply)
  - Reads optional lnc_test_fasta / pc_test_fasta from config [data]
  - After CV, evaluates best-fold model on the independent test set
  - Saves test metrics to test_results.json in the output directory
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import pickle
import pandas as pd
from Bio import SeqIO

from configs.load_config import load_config
from models.model_builder import create_model_builder
from data.cv_utils import load_sequences_in_order, verify_sequence_order
from data.feature_dataset import SequenceFeatureDataset
from trainers.features_attn_trainer import BetaVAEFeaturesAttentionTrainer
from torch.utils.data import DataLoader


def load_features_and_scalers(config):
    """Load feature data and scalers."""
    print("\nLoading features and scalers...")
    
    te_df = pd.read_csv(config.get('data', 'te_features_csv'), index_col='transcript_id')
    nonb_df = pd.read_csv(config.get('data', 'nonb_features_csv'), index_col='transcript_id')
    
    with open(config.get('data', 'te_scaler'), 'rb') as f:
        te_scaler = pickle.load(f)
    with open(config.get('data', 'nonb_scaler'), 'rb') as f:
        nonb_scaler = pickle.load(f)
    
    print(f"  TE features: {te_df.shape}")
    print(f"  Non-B features: {nonb_df.shape}")
    
    return te_df, nonb_df, te_scaler, nonb_scaler


def evaluate_on_test_set(cv_trainer, config, output_dir):
    """
    Load the independent test FASTAs, build a SequenceFeatureDataset,
    run inference with the best-fold model, and save metrics.
    Returns empty dict if no test FASTAs are configured.
    """
    lnc_test_fasta = config.get('data', 'lnc_test_fasta', default=None)
    pc_test_fasta  = config.get('data', 'pc_test_fasta',  default=None)

    if not lnc_test_fasta or not pc_test_fasta:
        print("\n[Test set] No test FASTAs configured — skipping independent evaluation.")
        return {}

    print("\n" + "=" * 80)
    print("INDEPENDENT TEST SET EVALUATION")
    print("=" * 80)
    print(f"  lnc test FASTA: {lnc_test_fasta}")
    print(f"  pc  test FASTA: {pc_test_fasta}")

    test_dataset = SequenceFeatureDataset(
        pc_fasta=pc_test_fasta,
        lnc_fasta=lnc_test_fasta,
        te_features_csv=config.get('data', 'te_features_csv'),
        nonb_features_csv=config.get('data', 'nonb_features_csv'),
        te_scaler_path=config.get('data', 'te_scaler'),
        nonb_scaler_path=config.get('data', 'nonb_scaler'),
        max_length=config.get('model', 'max_length')
    )

    print(f"  Test samples: {len(test_dataset):,}")
    lnc_count = sum(1 for v in test_dataset.label_dict.values() if v == 0)
    pc_count  = sum(1 for v in test_dataset.label_dict.values() if v == 1)
    print(f"    lncRNA: {lnc_count:,}")
    print(f"    pcRNA:  {pc_count:,}")
    test_metrics = cv_trainer.evaluate_on_test_set(test_dataset)

    print(f"\n  Test results:")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"    {k:20s}: {v:.4f}")
        else:
            print(f"    {k:20s}: {v}")

    test_results_path = output_dir / 'test_results.json'
    with open(test_results_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\n  Saved to: {test_results_path}")

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description='β-VAE with cross-attention genomic features')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', type=str,
                        default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("=" * 80)
    print("β-VAE with Cross-Attention Genomic Features")
    print("=" * 80)
    print(f"Architecture:    {config.get('model', 'architecture')}")
    print(f"Latent dim:      {config.get('model', 'latent_dim')}")
    print(f"Beta:            {config.get('model', 'beta')}")
    print(f"Use CSE:         {config.get('model', 'use_cse')}")
    print(f"TE features:     {config.get('model', 'te_features_dim')} dims")
    print(f"Non-B features:  {config.get('model', 'nonb_features_dim')} dims")
    print(f"Attn dropout:    {config.get('model', 'attn_dropout', default=0.1)}")
    print(f"Device:          {args.device}")
    print(f"N folds:         {config.get('training', 'n_folds')}")

    # Report test set config upfront
    lnc_test = config.get('data', 'lnc_test_fasta', default=None)
    pc_test  = config.get('data', 'pc_test_fasta',  default=None)
    if lnc_test and pc_test:
        print(f"Test set:        ENABLED")
        print(f"  lnc_test:      {lnc_test}")
        print(f"  pc_test:       {pc_test}")
    else:
        print(f"Test set:        NOT configured (CV only)")
    print("=" * 80)
    
    # Load data
    all_sequences, labels = load_sequences_in_order(
        lnc_fasta=config.get('data', 'lnc_fasta'),
        pc_fasta=config.get('data', 'pc_fasta')
    )
    
    # Create full dataset
    print("\n" + "=" * 80)
    print("Creating Dataset")
    print("=" * 80)
    
    full_dataset = SequenceFeatureDataset(
        pc_fasta=config.get('data', 'pc_fasta'),
        lnc_fasta=config.get('data', 'lnc_fasta'),
        te_features_csv=config.get('data', 'te_features_csv'),
        nonb_features_csv=config.get('data', 'nonb_features_csv'),
        te_scaler_path=config.get('data', 'te_scaler'),
        nonb_scaler_path=config.get('data', 'nonb_scaler'),
        max_length=config.get('model', 'max_length')
    )

    # Sanity check
    verify_sequence_order(all_sequences, labels, full_dataset.sequences)
    print(f"\nDataset: {len(full_dataset):,} samples")

    # Output directory
    output_dir = Path(config.get('output', 'experiment_name'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model builder
    model_builder = create_model_builder(config)
    sample_model  = model_builder()
    total_params  = sum(p.numel() for p in sample_model.parameters())
    trainable     = sum(p.numel() for p in sample_model.parameters() if p.requires_grad)
    print(f"\nModel: {total_params:,} params ({trainable:,} trainable)")
    print(f"L_encoded (attention positions): {sample_model.encoded_length}")

    with open(output_dir / 'model_architecture.txt', 'w') as f:
        f.write(str(sample_model))
    del sample_model

    cv_trainer = BetaVAEFeaturesAttentionTrainer(
        model_builder=model_builder,
        dataset=full_dataset,
        config=config,
        n_folds=config.get('training', 'n_folds'),
        device=args.device
    )
    
    # Run cross-validation
    print("\n" + "=" * 80)
    print(f"STARTING {config.get('training', 'n_folds')}-FOLD CROSS-VALIDATION")
    print("=" * 80)
    
    cv_results = cv_trainer.cross_validate(
        sequences=all_sequences,
        labels=labels,
        stratify_by_length=True
    )
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    cv_trainer.save_results(output_dir)

    """ # Extract embeddings from best fold
    print("\n" + "=" * 80)
    print("EXTRACTING EMBEDDINGS FROM BEST MODEL")
    print("=" * 80)
    
    embeddings_path = cv_trainer.extract_embeddings_from_best_fold(
        output_path=output_dir / 'embeddings_best_fold.npz'
    ) """

    # extract attention weights for ALL folds → input for analyze_attention.py
    attn_dir = output_dir / 'fold_attention'
    cv_trainer.extract_attention_all_folds(output_dir=attn_dir, labels_override=labels)
    print(f"\nAttention weights saved to: {attn_dir}/")
    print(f"Run interpretability analysis with:")
    print(f"  python analyze_attention.py \\")
    print(f"    --attn_dir {attn_dir} \\")
    print(f"    --output_dir {output_dir}/attention_analysis")

    # -----------------------------------------------------------------------
    # Independent test set evaluation
    # -----------------------------------------------------------------------
    test_metrics = evaluate_on_test_set(cv_trainer, config, output_dir)

    # Save config + hyperparams
    config.save(str(output_dir / 'config.json'))

    hyperparams = {
        'model': {
            'architecture':      config.get('model', 'architecture'),
            'total_params':      total_params,
            'trainable_params':  trainable,
            'latent_dim':        config.get('model', 'latent_dim'),
            'beta':              config.get('model', 'beta'),
            'use_cse':           config.get('model', 'use_cse'),
            'cse_d_model':       config.get('model', 'cse_d_model'),
            'cse_kernel_size':   config.get('model', 'cse_kernel_size'),
            'te_features_dim':   config.get('model', 'te_features_dim'),
            'nonb_features_dim': config.get('model', 'nonb_features_dim'),
            'attn_dropout':      config.get('model', 'attn_dropout', default=0.1),
        },
        'data': {
            'total_samples':  len(full_dataset),
            'lnc_samples':    labels.count('lncRNA'),
            'pc_samples':     labels.count('protein_coding'),
            'lnc_test_fasta': lnc_test,
            'pc_test_fasta':  pc_test,
        },
        'training': {
            'n_folds':              config.get('training', 'n_folds'),
            'n_bins':               config.get('training', 'n_bins', default=5),
            'batch_size':           config.get('training', 'batch_size'),
            'learning_rate':        config.get('training', 'learning_rate'),
            'weight_decay':         config.get('training', 'weight_decay'),
            'alpha':                config.get('training', 'alpha'),
            'beta':                 config.get('training', 'beta'),
            'gamma_classification': config.get('training', 'gamma_classification'),
            'lambda_recon':         config.get('training', 'lambda_recon')
        }
    }
    
    with open(output_dir / 'hyperparameters.json', 'w') as f:
        json.dump(hyperparams, f, indent=2)
    
    print(f"Saved hyperparameters to {output_dir / 'hyperparameters.json'}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")

    print(f"\nCross-Validation ({config.get('training', 'n_folds')}-fold):")
    for metric in ['val_acc', 'precision', 'recall', 'f1']:
        m = cv_results[metric]['mean']
        s = cv_results[metric]['std']
        print(f"  {metric:12s}: {m:.4f} ± {s:.4f}")

    if test_metrics:
        print(f"\nIndependent Test Set:")
        for k, v in test_metrics.items():
            if isinstance(v, float):
                print(f"  {k:20s}: {v:.4f}")

    print("=" * 80)


if __name__ == '__main__':
    main()