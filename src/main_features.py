#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for β-VAE with genomic features training
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
from data.cv_utils import create_length_stratified_groups, load_sequences_in_order, verify_sequence_order
from data.feature_dataset import SequenceFeatureDataset
from trainers.beta_vae_features_trainer import BetaVAEFeaturesTrainer
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

def main():
    parser = argparse.ArgumentParser(
        description='β-VAE with genomic features training'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--device', type=str,
                       default='cuda:0' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("=" * 80)
    print("β-VAE with Genomic Features Training")
    print("=" * 80)
    print(f"Architecture:    {config.get('model', 'architecture')}")
    print(f"Latent dim:      {config.get('model', 'latent_dim')}")
    print(f"Beta:            {config.get('model', 'beta')}")
    print(f"Use CSE:         {config.get('model', 'use_cse')}")
    if config.get('model', 'use_cse'):
        print(f"  CSE d_model:   {config.get('model', 'cse_d_model')}")
        print(f"  CSE kernel:    {config.get('model', 'cse_kernel_size')}")
    print(f"TE features:     {config.get('model', 'te_features_dim')} dims")
    print(f"Non-B features:  {config.get('model', 'nonb_features_dim')} dims")
    print(f"Device:          {args.device}")
    print(f"N folds:         {config.get('training', 'n_folds')}")
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
    
    print(f"\nDataset created: {len(full_dataset):,} samples")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = config.get('output', 'experiment_name')
    output_dir = Path(experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Build sample model to show architecture
    print("\n" + "=" * 80)
    print("Model Architecture")
    print("=" * 80)
    
    model_builder = create_model_builder(config)
    sample_model = model_builder()
    total_params = sum(p.numel() for p in sample_model.parameters())
    trainable_params = sum(p.numel() for p in sample_model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Save model architecture
    with open(output_dir / 'model_architecture.txt', 'w') as f:
        f.write(str(sample_model))
    
    del sample_model  # Free memory
    
    # Create trainer
    print("\n" + "=" * 80)
    print("Cross-Validation Training Setup")
    print("=" * 80)
    
    cv_trainer = BetaVAEFeaturesTrainer(
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

    # Extract embeddings from best fold
    print("\n" + "=" * 80)
    print("EXTRACTING EMBEDDINGS FROM BEST MODEL")
    print("=" * 80)
    
    embeddings_path = cv_trainer.extract_embeddings_from_best_fold(
        output_path=output_dir / 'embeddings_best_fold.npz'
    )
    
    print(f"\n  Embeddings saved to: {embeddings_path}")
    
    # Save config
    config_save_path = output_dir / 'config.json'
    config.save(str(config_save_path))
    print(f"Saved config to {config_save_path}")
    
    # Save hyperparameters
    hyperparams = {
        'model': {
            'architecture': config.get('model', 'architecture'),
            'total_params': total_params,
            'trainable_params': trainable_params,
            'latent_dim': config.get('model', 'latent_dim'),
            'beta': config.get('model', 'beta'),
            'use_cse': config.get('model', 'use_cse'),
            'cse_d_model': config.get('model', 'cse_d_model'),
            'cse_kernel_size': config.get('model', 'cse_kernel_size'),
            'te_features_dim': config.get('model', 'te_features_dim'),
            'nonb_features_dim': config.get('model', 'nonb_features_dim')
        },
        'data': {
            'total_samples': len(full_dataset),
            'lnc_samples': labels.count('lncRNA'),
            'pc_samples': labels.count('protein_coding')
        },
        'training': {
            'n_folds': config.get('training', 'n_folds'),
            'n_bins': config.get('training', 'n_bins', default=5),
            'batch_size': config.get('training', 'batch_size'),
            'learning_rate': config.get('training', 'learning_rate'),
            'weight_decay': config.get('training', 'weight_decay'),
            'alpha': config.get('training', 'alpha'),
            'beta': config.get('training', 'beta'),
            'gamma_classification': config.get('training', 'gamma_classification'),
            'lambda_recon': config.get('training', 'lambda_recon')
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
    print(f"\nCross-Validation Summary:")
    for metric in ['val_acc', 'precision', 'recall', 'f1']:
        mean = cv_results[metric]['mean']
        std = cv_results[metric]['std']
        print(f"  {metric.upper():12s}: {mean:.4f} ± {std:.4f}")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()