#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation study for β-VAE with cross-modal attention (BetaVAEFeaturesAttentionTrainer)

Ablation axes:
  1. Modality: TE only / NonB only / Both (full)
  2. Entropy regularization: gamma_attn=0.1 (default) vs gamma_attn=0.0

Variants:
  full          — TE + NonB + entropy regularization           (baseline)
  seq_te        — TE only  + entropy regularization           (modality ablation)
  seq_nonb      — NonB only + entropy regularization          (modality ablation)

Usage:
    # All variants, all folds
    python main_attn_ablation.py \\
        --base_config configs/beta_vae_features_attn_g49.json \\
        --device cuda:0

    # Specific variants only
    python main_attn_ablation.py \\
        --base_config configs/beta_vae_features_attn_g49.json \\
        --variants full no_entropy seq_te seq_nonb \\
        --device cuda:0

    # Single fold (quick test)
    python main_attn_ablation.py \\
        --base_config configs/beta_vae_features_attn_g49.json \\
        --fold 0 \\
        --device cuda:0
"""

import sys
import json
import numpy as np
import torch
import argparse
from pathlib import Path
from copy import deepcopy

from configs.load_config import load_config
from models.model_builder import create_model_builder
from data.cv_utils import load_sequences_in_order, verify_sequence_order
from data.feature_dataset import SequenceFeatureDataset
from training.features_attn_trainer import BetaVAEFeaturesAttentionTrainer


# ─────────────────────────────────────────────────────────────────────────────
# Variant definitions
# ─────────────────────────────────────────────────────────────────────────────

VARIANTS = {
    'seq_te': {
        'use_te_features':   True,
        'use_nonb_features': False,
        'gamma_attn':        0.1,
        'description':       'TE features only + entropy regularization',
    },
    'seq_nonb': {
        'use_te_features':   False,
        'use_nonb_features': True,
        'gamma_attn':        0.1,
        'description':       'NonB features only + entropy regularization',
    },
    'seq_only': {
        'use_te_features':   False,
        'use_nonb_features': False,
        'gamma_attn':        0.1,
        'description':       'Sequence only : entropy regularization',
    }
}


# ─────────────────────────────────────────────────────────────────────────────
# Config manipulation
# ─────────────────────────────────────────────────────────────────────────────

def create_ablation_config(base_config, variant_name, base_experiment_name):
    """Build a modified config for a given ablation variant."""
    settings = VARIANTS[variant_name]
    config = deepcopy(base_config)

    # Redirect output
    config.config['output']['experiment_name'] = (
        f"{base_experiment_name}/ablations/{variant_name}"
    )

    # Entropy regularization weight
    config.config['training']['gamma_attn'] = settings['gamma_attn']

    # Store ablation metadata for logging / saving
    config.config['ablation'] = {
        'variant':           variant_name,
        'use_te_features':   settings['use_te_features'],
        'use_nonb_features': settings['use_nonb_features'],
        'gamma_attn':        settings['gamma_attn'],
        'description':       settings['description'],
    }

    return config


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(variant_config):
    """Load sequences and build SequenceFeatureDataset for this variant."""
    abl = variant_config.config['ablation']

    all_sequences, labels = load_sequences_in_order(
        lnc_fasta=variant_config.get('data', 'lnc_fasta'),
        pc_fasta=variant_config.get('data', 'pc_fasta'),
    )

    full_dataset = SequenceFeatureDataset(
        lnc_fasta=variant_config.get('data', 'lnc_fasta'),
        pc_fasta=variant_config.get('data', 'pc_fasta'),
        te_features_csv=variant_config.get('data', 'te_features_csv'),
        nonb_features_csv=variant_config.get('data', 'nonb_features_csv'),
        te_scaler_path=variant_config.get('data', 'te_scaler'),
        nonb_scaler_path=variant_config.get('data', 'nonb_scaler'),
        max_length=variant_config.get('model', 'max_length'),
        use_te_features=abl['use_te_features'],
        use_nonb_features=abl['use_nonb_features'],
    )

    verify_sequence_order(all_sequences, labels, full_dataset.sequences)
    return all_sequences, labels, full_dataset


def print_variant_header(variant_name, variant_config, fold=None, n_folds=5):
    settings = VARIANTS[variant_name]
    print("\n" + "=" * 80)
    print(f"ABLATION VARIANT: {variant_name.upper()}"
          + (f"  (FOLD {fold}/{n_folds-1})" if fold is not None else f"  (ALL {n_folds} FOLDS)"))
    print("=" * 80)
    print(f"  Description : {settings['description']}")
    print(f"  TE features : {settings['use_te_features']}")
    print(f"  NonB features: {settings['use_nonb_features']}")
    print(f"  gamma_attn  : {settings['gamma_attn']}")
    print(f"  Output      : {variant_config.get('output', 'experiment_name')}")
    print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# Single-fold training
# ─────────────────────────────────────────────────────────────────────────────

def train_single_fold(base_config, variant_name, fold_idx, n_folds, device):
    from sklearn.utils.class_weight import compute_class_weight
    from data.cv_utils import create_length_stratified_groups, get_cv_splitter
    from torch.utils.data import Subset

    base_experiment_name = base_config.get('output', 'experiment_name')
    variant_config = create_ablation_config(base_config, variant_name,
                                            base_experiment_name)
    print_variant_header(variant_name, variant_config, fold=fold_idx,
                         n_folds=n_folds)

    output_dir = Path(variant_config.get('output', 'experiment_name'))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'models').mkdir(exist_ok=True)
    variant_config.save(str(output_dir / 'config.json'))

    # Data
    all_sequences, labels, full_dataset = build_dataset(variant_config)
    print(f"  Dataset: {len(full_dataset):,} samples")

    # Fold split (identical seed/strategy as main training)
    n_bins = variant_config.get('training', 'n_bins', default=5)
    random_state = variant_config.get('training', 'random_state', default=42)
    strat_groups = create_length_stratified_groups(
        all_sequences, labels, n_bins=n_bins)
    skf = get_cv_splitter(n_folds=n_folds, random_state=random_state)
    train_idx, val_idx = list(skf.split(all_sequences, strat_groups))[fold_idx]

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset   = Subset(full_dataset, val_idx)

    # Class weights
    train_labels_numeric = [{'lnc': 0, 'pc': 1}[labels[i]] for i in train_idx]
    class_weights = compute_class_weight(
        'balanced', classes=np.array([0, 1]), y=train_labels_numeric)
    print(f"  Class weights: lnc={class_weights[0]:.3f}, "
          f"pc={class_weights[1]:.3f}")

    # Model
    model_builder = create_model_builder(variant_config)

    # Trainer
    cv_trainer = BetaVAEFeaturesAttentionTrainer(
        model_builder=model_builder,
        dataset=full_dataset,
        config=variant_config,
        n_folds=n_folds,
        device=device,
    )

    model, fold_metrics, model_path = cv_trainer.train_fold(
        fold_idx=fold_idx,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        class_weights=class_weights,
    )

    # Save fold results
    result = {
        'fold':             int(fold_idx),
        'variant':          variant_name,
        'val_loss':         float(fold_metrics['val_loss']),
        'val_acc':          float(fold_metrics['val_acc']),
        'precision':        float(fold_metrics['precision']),
        'recall':           float(fold_metrics['recall']),
        'f1':               float(fold_metrics['f1']),
        'confusion_matrix': fold_metrics['confusion_matrix'].tolist(),
        'model_path':       str(model_path),
    }
    with open(output_dir / f'fold_{fold_idx}_results.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n  Fold {fold_idx} complete:")
    print(f"    F1={fold_metrics['f1']:.4f}  "
          f"Acc={fold_metrics['val_acc']:.4f}  "
          f"Loss={fold_metrics['val_loss']:.4f}")
    print(f"    Saved: {output_dir}/fold_{fold_idx}_results.json")

    return fold_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Full cross-validation training
# ─────────────────────────────────────────────────────────────────────────────

def train_ablation_variant(base_config, variant_name, n_folds, device):
    base_experiment_name = base_config.get('output', 'experiment_name')
    variant_config = create_ablation_config(base_config, variant_name,
                                            base_experiment_name)
    print_variant_header(variant_name, variant_config, n_folds=n_folds)

    output_dir = Path(variant_config.get('output', 'experiment_name'))
    output_dir.mkdir(parents=True, exist_ok=True)
    variant_config.save(str(output_dir / 'config.json'))

    # Data
    all_sequences, labels, full_dataset = build_dataset(variant_config)
    print(f"  Dataset: {len(full_dataset):,} samples")

    # Model
    model_builder = create_model_builder(variant_config)
    sample_model = model_builder()
    total_params     = sum(p.numel() for p in sample_model.parameters())
    trainable_params = sum(p.numel() for p in sample_model.parameters()
                           if p.requires_grad)
    with open(output_dir / 'model_architecture.txt', 'w') as f:
        f.write(str(sample_model))
    del sample_model

    # Trainer
    cv_trainer = BetaVAEFeaturesAttentionTrainer(
        model_builder=model_builder,
        dataset=full_dataset,
        config=variant_config,
        n_folds=n_folds,
        device=device,
    )

    cv_results = cv_trainer.cross_validate(
        sequences=all_sequences,
        labels=labels,
        stratify_by_length=True,
    )

    cv_trainer.save_results(output_dir)

    # Save hyperparameters + ablation metadata
    hyperparams = {
        'ablation': variant_config.config['ablation'],
        'model': {
            'architecture':   variant_config.get('model', 'architecture'),
            'total_params':   total_params,
            'trainable_params': trainable_params,
            'latent_dim':     variant_config.get('model', 'latent_dim'),
            'beta':           variant_config.get('model', 'beta'),
            'attn_heads':     variant_config.get('model', 'attn_heads'),
        },
        'data': {
            'total_samples': len(full_dataset),
            'lnc_samples':   labels.count('lnc'),
            'pc_samples':    labels.count('pc'),
        },
        'training': {
            'n_folds':       n_folds,
            'batch_size':    variant_config.get('training', 'batch_size'),
            'learning_rate': variant_config.get('training', 'learning_rate'),
            'gamma_attn':    variant_config.get('training', 'gamma_attn'),
        },
    }
    with open(output_dir / 'hyperparameters.json', 'w') as f:
        json.dump(hyperparams, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print(f"VARIANT '{variant_name}' COMPLETE")
    print(f"{'='*80}")
    for metric in ['val_acc', 'precision', 'recall', 'f1']:
        m = cv_results[metric]
        print(f"  {metric.upper():12s}: {m['mean']:.4f} ± {m['std']:.4f}")

    return cv_results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Ablation study for β-VAE + cross-modal attention',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--base_config', required=True,
                        help='Path to base attention model config JSON')
    parser.add_argument('--variants', nargs='+',
                        default=list(VARIANTS.keys()),
                        choices=list(VARIANTS.keys()),
                        help='Variants to train (default: all)')
    parser.add_argument('--fold', type=int, default=None,
                        help='Train a single fold only (default: all folds)')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--device', type=str,
                        default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print("=" * 80)
    print("ABLATION STUDY: β-VAE + CROSS-MODAL ATTENTION")
    print("=" * 80)
    print(f"  Base config : {args.base_config}")
    print(f"  Variants    : {args.variants}")
    print(f"  Device      : {args.device}")
    print(f"  Fold        : {args.fold if args.fold is not None else 'all'}")
    print("=" * 80)

    base_config = load_config(args.base_config)
    base_experiment_name = base_config.get('output', 'experiment_name')

    all_results = {}

    for variant_name in args.variants:
        try:
            if args.fold is not None:
                results = train_single_fold(
                    base_config=base_config,
                    variant_name=variant_name,
                    fold_idx=args.fold,
                    n_folds=args.n_folds,
                    device=args.device,
                )
            else:
                results = train_ablation_variant(
                    base_config=base_config,
                    variant_name=variant_name,
                    n_folds=args.n_folds,
                    device=args.device,
                )
            all_results[variant_name] = results

        except Exception as e:
            print(f"\n  ERROR in variant '{variant_name}': {e}")
            import traceback
            traceback.print_exc()
            continue

    # ── Cross-variant summary ──
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("ABLATION STUDY COMPLETE — SUMMARY")
        print("=" * 80)
        print(f"\n{'Variant':<20} {'Accuracy':<22} {'F1':<22} "
              f"{'gamma_attn':<12} {'TE':<6} {'NonB'}")
        print("-" * 85)

        for vname in args.variants:
            if vname not in all_results:
                continue
            res = all_results[vname]
            s   = VARIANTS[vname]

            # fold results have scalar metrics; cv results have dicts
            if isinstance(res.get('f1'), dict):
                acc_str = f"{res['val_acc']['mean']:.4f} ± {res['val_acc']['std']:.4f}"
                f1_str  = f"{res['f1']['mean']:.4f} ± {res['f1']['std']:.4f}"
            else:
                acc_str = f"{res.get('val_acc', 0):.4f}"
                f1_str  = f"{res.get('f1', 0):.4f}"

            print(f"{vname:<20} {acc_str:<22} {f1_str:<22} "
                  f"{s['gamma_attn']:<12} {str(s['use_te_features']):<6} "
                  f"{s['use_nonb_features']}")

        # Save comparison JSON
        ablation_dir = Path(base_experiment_name) / 'ablations'
        comparison = {
            'variants': args.variants,
            'n_folds':  args.n_folds,
            'results':  {},
        }
        for vname, res in all_results.items():
            if isinstance(res.get('f1'), dict):
                comparison['results'][vname] = {
                    'accuracy': {'mean': res['val_acc']['mean'],
                                 'std':  res['val_acc']['std'],
                                 'values': res['val_acc'].get('values', [])},
                    'f1':       {'mean': res['f1']['mean'],
                                 'std':  res['f1']['std'],
                                 'values': res['f1'].get('values', [])},
                    'settings': VARIANTS[vname],
                }
            else:
                comparison['results'][vname] = {
                    'fold_metrics': res,
                    'settings':     VARIANTS[vname],
                }

        with open(ablation_dir / 'attn_ablation_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\n  Saved: {ablation_dir}/attn_ablation_comparison.json")

    print("\n  Ablation study complete.\n")


if __name__ == '__main__':
    main()