#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for cross-validation training with contrastive learning
"""
import os
import sys
import argparse
import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path

from torch.utils.data import DataLoader
from data.loaders import load_sequences_with_labels
from data.preprocessing import SequencePreprocessor
from data.cv_utils import (
    load_biotype_mapping,
    extract_biotypes_from_sequences,
    group_rare_biotypes
)
from models import get_contrastive_model
from configs.load_config import load_config
from trainers.contrastive_trainer import ContrastiveTrainer

def print_biotype_statistics(biotypes, labels):
    """Print detailed biotype statistics"""
    df = pd.DataFrame({'biotype': biotypes, 'label': labels})
    
    print("\nBiotype distribution:")
    biotype_counts = df['biotype'].value_counts()
    
    # Print top 20
    for i, (biotype, count) in enumerate(biotype_counts.head(20).items()):
        label_dist = df[df['biotype'] == biotype]['label'].value_counts()
        label_str = ', '.join([f"{k}={v}" for k, v in label_dist.items()])
        print(f"  {biotype:30s}: {count:6d} ({label_str})")
    
    if len(biotype_counts) > 20:
        print(f"  ... and {len(biotype_counts) - 20} more biotypes")
    
    print(f"\nTotal unique biotypes: {len(biotype_counts)}")

def create_model_builder(config):
    """
    Create a function that builds a fresh model instance with contrastive learning
    This is needed because CV requires multiple model instances
    """
    architecture = config.get('model', 'architecture', default='cnn')
    
    def build_model():
        # Base parameters (all models need these)
        model_params = {
            'num_classes': config.get('model', 'num_classes'),
            'input_dim': 5  # Default for one-hot
        }
        model_params['input_length'] = config.get('model', 'max_length')
        encoding_type = config.get('model', 'encoding_type', default='one_hot')
        
        # Add encoding-specific parameters
        if encoding_type == 'kmer':
            kmer_k = config.get('model', 'kmer_k', default=6)
            model_params['input_dim'] = 4 ** kmer_k
        elif encoding_type == 'cse':
            model_params['cse_d_model'] = config.get('model', 'cse_d_model', default=128)
            model_params['cse_kernel_size'] = config.get('model', 'cse_kernel_size', default=6)
        
        # Architecture-specific parameters
        if architecture == 'beta_vae':
            # beta-VAE specific parameters ONLY
            model_params.update({
                'latent_dim': config.get('model', 'latent_dim', default=128),
                'beta': config.get('model', 'beta', default=4.0),
                'dropout_rate': config.get('model', 'dropout_rate', default=0.3),
                'use_cse': config.get('model', 'use_cse', default=False),
                'cse_d_model': config.get('model', 'cse_d_model', default=128),
                'cse_kernel_size': config.get('model', 'cse_kernel_size', default=9)
            })
        
        else:
            # For other architectures (CNN, transformer, etc.)
            
            # Contrastive-specific parameters (NOT for beta-VAE)
            model_params['embedding_dim'] = config.get('model', 'embedding_dim', default=256)
            model_params['projection_dim'] = config.get('model', 'projection_dim', default=128)
            model_params['dropout_rate'] = config.get('model', 'dropout_rate', default=0.3)
            model_params['use_cse'] = config.get('model', 'use_cse', default=False)
            
            if architecture == 'autoencoder':
                model_params.update({
                    'latent_dim': config.get('model', 'latent_dim', default=128),
                    'hidden_dims': config.get('model', 'hidden_dims', default=[256, 128])
                })
            
            elif architecture == 'cnn_cse':
                model_params.update({
                    'num_filters': config.get('model', 'num_filters', default=[256, 256]),
                    'kernel_sizes': config.get('model', 'kernel_sizes', default=[8, 8]),
                    'pool_sizes': config.get('model', 'pool_sizes', default=[4, 4])
                })
            
            elif architecture == 'transformer':
                model_params.update({
                    'd_model': config.get('model', 'd_model', default=128),
                    'nhead': config.get('model', 'nhead', default=8),
                    'num_layers': config.get('model', 'num_layers', default=4)
                })
            
            elif architecture == 'bert':
                model_params.update({
                    'pretrained_model': config.get('model', 'pretrained_model', 
                                                   default='multimolecule/rnafm'),
                    'freeze_backbone': config.get('model', 'freeze_backbone', default=False)
                })
        
        return get_contrastive_model(architecture, **model_params)
    
    return build_model


def main():
    parser = argparse.ArgumentParser(
        description='Cross-validation training with contrastive learning for RNA classification'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--device', type=str,
                       default='cuda:0' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    parser.add_argument('--biotype_csv', type=str,
                       default='${DATA_ROOT}/data/dataset_biotypes.csv',
                       help='Path to biotype CSV file')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of CV folds (default: 5)')
    parser.add_argument('--misc_threshold', type=float, default=0.6,
                       help='Confidence threshold for misc class assignment')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("=" * 80)
    print("Contrastive Learning Cross-Validation Configuration")
    print("=" * 80)
    print(f"Architecture:    {config.get('model', 'architecture', default='cnn')}")
    print(f"Encoding:        {config.get('model', 'encoding_type', default='one_hot')}")
    print(f"Use CSE:         {config.get('model', 'use_cse', default=False)}")
    if config.get('model', 'use_cse', default=False):
        print(f"  CSE d_model:   {config.get('model', 'cse_d_model', default=128)}")
        print(f"  CSE kernel:    {config.get('model', 'cse_kernel_size', default=6)}")
    print(f"Device:          {args.device}")
    print(f"N folds:         {args.n_folds}")
    print(f"Misc threshold:  {args.misc_threshold}")
    print("=" * 80)
    
    # Load sequences
    print("\nLoading sequences from FASTA files...")
    matched_seqs, matched_labels = load_sequences_with_labels(
        config.get('data', 'lnc_fasta'),
        config.get('data', 'pc_fasta')
    )
    print(f"Total sequences: {len(matched_seqs)}")
    print(f"  lncRNA: {matched_labels.count('lnc')}")
    print(f"  PC: {matched_labels.count('pc')}")
    
    # Load biotype information
    biotype_lookup = load_biotype_mapping(args.biotype_csv)
    biotypes = extract_biotypes_from_sequences(matched_seqs, biotype_lookup)
    
    print(f"\nBiotype extraction complete:")
    print(f"  Total biotypes assigned: {len(biotypes)}")
    print(f"  Unknown biotypes: {biotypes.count('unknown')}")
    
    # Group rare biotypes for contrastive learning
    grouped_biotypes = group_rare_biotypes(biotypes, min_count=500)
    # Print biotype statistics
    print_biotype_statistics(biotypes, matched_labels)
    
    # Create output directory
    experiment_name = config.get('output', 'experiment_name', default='gencode_v47_experiments')
    output_dir = Path(experiment_name)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Save biotype mapping
    unique_biotypes = sorted(set(biotypes))
    biotype_mapping = {bt: idx for idx, bt in enumerate(unique_biotypes)}
    
    with open(output_dir / 'biotype_mapping.json', 'w') as f:
        json.dump(biotype_mapping, f, indent=2)
    print(f"Saved biotype mapping to {output_dir / 'biotype_mapping.json'}")
    
    # Initialize preprocessor
    print("\n" + "=" * 80)
    print("Preprocessing Setup")
    print("=" * 80)
    
    max_length = config.get('model', 'max_length')
    encoding_type = config.get('model', 'encoding_type', default='one_hot')
    
    print(f"Encoding type: {encoding_type}")
    print(f"Max length: {max_length}")
    
    if encoding_type == 'kmer':
        kmer_k = config.get('model', 'kmer_k', default=6)
        print(f"K-mer size: {kmer_k}")
        preprocessor = SequencePreprocessor(
            max_length=max_length,
            encoding_type=encoding_type,
            kmer_k=kmer_k
        )
    elif encoding_type == 'cse':
        cse_d_model = config.get('model', 'cse_d_model', default=128)
        cse_kernel_size = config.get('model', 'cse_kernel_size', default=6)
        print(f"CSE d_model: {cse_d_model}")
        print(f"CSE kernel size: {cse_kernel_size}")
        preprocessor = SequencePreprocessor(
            max_length=max_length,
            encoding_type=encoding_type,
            cse_d_model=cse_d_model,
            cse_kernel_size=cse_kernel_size
        )
    else:
        preprocessor = SequencePreprocessor(
            max_length=max_length,
            encoding_type=encoding_type
        )
    
    # Create model builder
    print("\n" + "=" * 80)
    print("Model Setup")
    print("=" * 80)
    
    model_builder = create_model_builder(config)
    
    # Build a sample model to show architecture
    sample_model = model_builder()
    total_params = sum(p.numel() for p in sample_model.parameters())
    trainable_params = sum(p.numel() for p in sample_model.parameters() if p.requires_grad)
    
    print(f"Architecture: {config.get('model', 'architecture')}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Save model architecture
    with open(output_dir / 'model_architecture.txt', 'w') as f:
        f.write(str(sample_model))
    
    del sample_model  # Free memory
    
    # Create cross-validation trainer
    print("\n" + "=" * 80)
    print("Cross-Validation Training Setup")
    print("=" * 80)
    
    lambda_contrastive = config.get('training', 'lambda_contrastive', default=1.0)
    lambda_classification = config.get('training', 'lambda_classification', default=1.0)
    temperature = config.get('training', 'temperature', default=0.07)
    max_oversample = config.get('training', 'max_oversample', default=None)

    if config.get('model', 'architecture') == 'beta_vae':
        max_oversample = None  # Disable oversampling for beta-VAE
        
    cv_trainer = ContrastiveTrainer(
        model_builder=model_builder,
        preprocessor=preprocessor,
        config=config,
        n_folds=args.n_folds,
        misc_threshold=args.misc_threshold,
        device=args.device,
        max_oversample=max_oversample
    )
    
    if args.n_folds == 1:
        # ======================================================================
        # SINGLE TRAIN/VAL SPLIT MODE (for debugging and quick experiments)
        # ======================================================================
        print("\n" + "=" * 80)
        print("SINGLE TRAIN/VAL SPLIT MODE (n_folds=1)")
        print("=" * 80)
        
        from sklearn.model_selection import train_test_split
        from sklearn.utils.class_weight import compute_class_weight
        from data.preprocessing import RNASequenceBiotypeDataset
        
        # Create stratification groups
        strat_groups = cv_trainer._create_length_stratified_groups(matched_seqs, matched_labels, n_bins=5)
        
        # Split indices
        train_idx, val_idx = train_test_split(
            range(len(matched_seqs)),
            test_size=0.2,
            stratify=strat_groups,
            random_state=config.get('training', 'random_state', default=42)
        )
        
        # Get train/val data
        train_seqs = [matched_seqs[i] for i in train_idx]
        val_seqs = [matched_seqs[i] for i in val_idx]
        train_labels = [matched_labels[i] for i in train_idx]
        val_labels = [matched_labels[i] for i in val_idx]
        train_biotypes = [grouped_biotypes[i] for i in train_idx]
        val_biotypes = [grouped_biotypes[i] for i in val_idx]
        
        print(f"\nData split (80/20):")
        print(f"  Train: {len(train_seqs):,} samples")
        print(f"  Val:   {len(val_seqs):,} samples")
        
        # Create biotype mapping
        unique_biotypes = sorted(set(grouped_biotypes))
        biotype_to_idx = {bt: idx for idx, bt in enumerate(unique_biotypes)}
        cv_trainer.biotype_to_idx = biotype_to_idx
        
        # Compute class weights
        label_to_idx = {'lnc': 0, 'pc': 1}
        train_labels_numeric = [label_to_idx[l] for l in train_labels]
        class_weights = compute_class_weight(
            'balanced',
            classes=np.array([0, 1]),
            y=train_labels_numeric
        )
        print(f"  Class weights: lnc={class_weights[0]:.3f}, pc={class_weights[1]:.3f}")
        
        # Create datasets
        train_dataset = RNASequenceBiotypeDataset(
            train_seqs, train_labels, preprocessor, 
            biotype_labels=train_biotypes, 
            biotype_to_idx=biotype_to_idx
        )
        val_dataset = RNASequenceBiotypeDataset(
            val_seqs, val_labels, preprocessor,
            biotype_labels=val_biotypes,
            biotype_to_idx=biotype_to_idx
        )
        
        # Train single fold
        print("\n" + "=" * 80)
        print("TRAINING")
        print("=" * 80)
        
        model, fold_metrics, model_path = cv_trainer.train_fold(
            fold_idx=0,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_biotypes=train_biotypes,
            class_weights=class_weights
        )
        
        # Store for embedding extraction
        cv_trainer.best_model_path = model_path
        cv_trainer.best_fold_idx = 0

        print("\nGenerating predictions for hard case analysis...")

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('training', 'batch_size'),
            shuffle=False,
            num_workers=config.get('training', 'num_workers', default=1)
        )

        # Get predictions with confidence
        predictions, confidences, probs = cv_trainer._predict_with_confidence(model, val_loader)

        # Create hard cases dataframe
        hard_cases = []
        for i in range(len(val_seqs)):
            true_label = label_to_idx[val_labels[i]]
            true_label_str = val_labels[i]
            biotype = val_biotypes[i]
            
            # Mark as hard case if low confidence or incorrect
            is_hard_case = (
                confidences[i] < cv_trainer.misc_threshold or
                predictions[i] != true_label
            )
            
            seq_length = len(str(val_seqs[i].seq))
            
            hard_cases.append({
                'sample_idx': val_idx[i],
                'transcript_id': val_seqs[i].id,
                'true_label': true_label_str,
                'biotype': biotype,
                'sequence_length': seq_length,
                'avg_confidence': confidences[i],
                'std_confidence': 0.0,
                'max_agreement': 1.0,
                'misc_frequency': 1.0 if confidences[i] < cv_trainer.misc_threshold else 0.0,
                'error_rate': 1.0 if predictions[i] != true_label else 0.0,
                'is_hard_case': is_hard_case,
                'fold_predictions': [{
                    'fold': 0,
                    'prediction': predictions[i],
                    'confidence': confidences[i],
                    'probs': probs[i],
                    'true_label': true_label,
                    'biotype': biotype
                }]
            })

        cv_trainer.hard_cases = pd.DataFrame(hard_cases)

        # Print summary
        n_hard = cv_trainer.hard_cases['is_hard_case'].sum()
        print(f"\nHard Cases Summary:")
        print(f"  Total validation samples: {len(val_seqs):,}")
        print(f"  Hard cases: {n_hard:,} ({100*n_hard/len(val_seqs):.1f}%)")
        print(f"  Easy cases: {len(val_seqs) - n_hard:,} ({100*(1 - n_hard/len(val_seqs)):.1f}%)")

        if n_hard > 0:
            print(f"\n  Top 5 biotypes in hard cases:")
            hard_biotypes = cv_trainer.hard_cases[cv_trainer.hard_cases['is_hard_case']]['biotype'].value_counts().head(5)
            for biotype, count in hard_biotypes.items():
                print(f"    {biotype}: {count}")
        
        # Create minimal results for consistency with CV format
        cv_results = {
            'val_acc': {'mean': fold_metrics['val_acc'], 'std': 0.0, 'values': [fold_metrics['val_acc']]},
            'precision': {'mean': fold_metrics['precision'], 'std': 0.0, 'values': [fold_metrics['precision']]},
            'recall': {'mean': fold_metrics['recall'], 'std': 0.0, 'values': [fold_metrics['recall']]},
            'f1': {'mean': fold_metrics['f1'], 'std': 0.0, 'values': [fold_metrics['f1']]}
        }
        
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Val Acc:  {fold_metrics['val_acc']:.4f}")
        print(f"Precision: {fold_metrics['precision']:.4f}")
        print(f"Recall:    {fold_metrics['recall']:.4f}")
        print(f"F1 Score:  {fold_metrics['f1']:.4f}")
        print(f"Model saved: {model_path}")
        print(f"{'='*80}\n")
        
    else:
        # ======================================================================
        # FULL CROSS-VALIDATION MODE (n_folds >= 2)
        # ======================================================================
        print("\n" + "=" * 80)
        print(f"STARTING {args.n_folds}-FOLD CROSS-VALIDATION")
        print("=" * 80)
        
        cv_results = cv_trainer.cross_validate(
            sequences=matched_seqs,
            labels=matched_labels,
            biotypes=grouped_biotypes,
            stratify_by_length=True
        )

    # Extract embeddings from best fold (works for both modes)
    print("\n" + "=" * 80)
    print("EXTRACTING EMBEDDINGS FROM BEST MODEL")
    print("=" * 80)
    
    embeddings_path = cv_trainer.extract_embeddings_from_best_fold(
        sequences=matched_seqs,
        labels=matched_labels,
        biotypes=grouped_biotypes,
        output_path=output_dir / 'embeddings_best_fold.npz'
    )

    print(f"\n  Embeddings saved to: {embeddings_path}")
    
    # Analyze hard cases
    print("\n" + "=" * 80)
    print("ANALYZING HARD CASES")
    print("=" * 80)
    
    cv_trainer.analyze_hard_cases(output_dir)
    
    # Save all results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    cv_trainer.save_results(output_dir)
    
    # Save config
    config_save_path = output_dir / 'config.json'
    config.save(str(config_save_path))
    print(f"Saved config to {config_save_path}")
    
    # Save hyperparameters summary
    hyperparams = {
        'model': {
            'architecture': config.get('model', 'architecture'),
            'total_params': total_params,
            'trainable_params': trainable_params,
            'use_cse': config.get('model', 'use_cse', default=False),
            'cse_d_model': config.get('model', 'cse_d_model', default=128) if config.get('model', 'use_cse', default=False) else None,
            'cse_kernel_size': config.get('model', 'cse_kernel_size', default=6) if config.get('model', 'use_cse', default=False) else None
        },
        'data': {
            'total_samples': len(matched_seqs),
            'num_biotypes': len(unique_biotypes)
        },
        'training': {
            'n_folds': args.n_folds,
            'misc_threshold': args.misc_threshold,
            'batch_size': config.get('training', 'batch_size'),
            'learning_rate': config.get('training', 'learning_rate'),
            'weight_decay': config.get('training', 'weight_decay', default=1e-5),
            'lambda_contrastive': lambda_contrastive,
            'lambda_classification': lambda_classification,
            'temperature': temperature
        }
    }
    
    with open(output_dir / 'hyperparameters.json', 'w') as f:
        json.dump(hyperparams, f, indent=2)
    print(f"Saved hyperparameters to {output_dir / 'hyperparameters.json'}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION COMPLETE")
    print("=" * 80)
    
    print(f"\nResults saved to: {output_dir}/")
    # Print CV summary
    print(f"\nCross-Validation Summary:")
    for metric in ['val_acc', 'precision', 'recall', 'f1']:
        mean = cv_results[metric]['mean']
        std = cv_results[metric]['std']
        print(f"  {metric.upper():12s}: {mean:.4f} ± {std:.4f}")
    
    # Hard cases summary
    if cv_trainer.hard_cases is not None:
        n_hard = cv_trainer.hard_cases['is_hard_case'].sum()
        n_total = len(cv_trainer.hard_cases)
        print(f"\nHard Cases:")
        print(f"  Total: {n_hard}/{n_total} ({100*n_hard/n_total:.1f}%)")
        
        # Top biotypes in hard cases
        hard_cases_only = cv_trainer.hard_cases[cv_trainer.hard_cases['is_hard_case']]
        if len(hard_cases_only) > 0:
            print(f"\n  Top 5 biotypes in hard cases:")
            top_biotypes = hard_cases_only['biotype'].value_counts().head(5)
            for biotype, count in top_biotypes.items():
                print(f"    {biotype}: {count}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()