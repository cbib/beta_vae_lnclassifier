#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate all trained folds for β-VAE with Genomic Features and compute cross-validation statistics

This script:
1. Loads each fold's checkpoint
2. Evaluates on its validation set with proper feature loading
3. Computes overall CV statistics (mean ± std)
4. Extracts and saves embeddings from best fold (or all folds)
5. Generates hard case CSVs for downstream analysis (Optional)
6. Generates comprehensive results for β-VAE with TE/Non-B features

Usage:
    # Evaluate all folds
    python evaluate_cv_fold_features.py \
        --config configs/beta_vae_features_g47.json \
        --experiment_dir gencode_v47_experiments/beta_vae_features_g47 \
        --n_folds 5
    
    # Extract embeddings from ALL folds and generate hard case CSVs
    python evaluate_cv_fold_features.py \
        --config configs/beta_vae_features_g47.json \
        --experiment_dir gencode_v47_experiments/beta_vae_features_g47 \
        --n_folds 5 \
        --extract_all_folds \
        --generate_hard_case_csvs
"""

import argparse
import torch
import torch.nn.functional as F
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve, 
    auc, 
    precision_recall_curve, 
    average_precision_score
)
import matplotlib.pyplot as plt

from data.feature_dataset import SequenceFeatureDataset
from data.cv_utils import create_length_stratified_groups, get_cv_splitter, load_sequences_in_order, verify_sequence_order
from models.beta_vae_features import BetaVAEWithFeatures
from models.beta_vae_features_attn import BetaVAEWithFeaturesAttention
from configs.load_config import load_config


def create_model_builder(config):
    """Create model builder function for β-VAE with features"""
    
    def builder():
        model_params = {
            'num_classes': config.get('model', 'num_classes', default=2),
            'input_dim': config.get('model', 'input_dim'),
            'input_length': config.get('model', 'max_length'),
            'latent_dim': config.get('model', 'latent_dim', default=128),
            'beta': config.get('model', 'beta', default=4.0),
            'dropout_rate': config.get('model', 'dropout_rate', default=0.3),
            'use_cse': config.get('model', 'use_cse', default=True),
            'cse_d_model': config.get('model', 'cse_d_model', default=512),
            'cse_kernel_size': config.get('model', 'cse_kernel_size', default=9),
            'te_features_dim': config.get('model', 'te_features_dim'),
            'nonb_features_dim': config.get('model', 'nonb_features_dim'),
            'te_processor_hidden': config.get('model', 'te_processor_hidden', default=[128, 64, 32]),
            'nonb_processor_hidden': config.get('model', 'nonb_processor_hidden', default=[128, 64, 32]),
            'classifier_hidden': config.get('model', 'classifier_hidden', default=[256, 128]),
        }

        if config.get('model', 'architecture') == 'beta_vae_features':
            model = BetaVAEWithFeatures(**model_params)
        elif config.get('model', 'architecture') == 'beta_vae_features_attn':
            model_params.update({
            'attn_heads': config.get('model', 'attn_heads', default=4),
            'attn_dropout': config.get('model', 'attn_dropout', default=0.1),
            })
            model = BetaVAEWithFeaturesAttention(**model_params)
        
        return model
    
    return builder

def evaluate_deterministic(model, dataloader, device):
    """
    Evaluate model in deterministic mode (use mu, no sampling)
    
    Returns:
        Dict with predictions, confidences, probs, and metrics
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    all_confidences = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            sequences = batch['sequence'].to(device)
            te_features = batch['te_features'].to(device)
            nonb_features = batch['nonb_features'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass (deterministic - use mu)
            outputs = model(sequences, te_features, nonb_features, deterministic=True)
            logits = outputs['logits']
            
            # Get predictions and probabilities
            probs = F.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)
            confidences = probs.max(dim=1)[0]
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_confidences.append(confidences.cpu().numpy())
    
    # Concatenate
    predictions = np.concatenate(all_predictions)
    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)
    confidences = np.concatenate(all_confidences)

    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    cm = confusion_matrix(labels, predictions)
    
    return {
        'predictions': predictions,
        'labels': labels,
        'probs': probs,
        'confidences': confidences,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def extract_embeddings(model, dataloader, device, biotype_csv=None, lnc_fasta=None, pc_fasta=None):
    """
    Extract latent embeddings (mu) from model
    
    Args:
        biotype_csv: Optional path to biotype CSV for loading real biotype names
        lnc_fasta, pc_fasta: Paths to FASTA files (needed for biotype loading)
    
    Returns:
        Dict with embeddings, labels, predictions, biotypes, etc.
    """
    model.eval()
    
    all_embeddings = []
    all_labels = []
    all_logits = []
    all_transcript_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings", leave=False):
            sequences = batch['sequence'].to(device)
            te_features = batch['te_features'].to(device)
            nonb_features = batch['nonb_features'].to(device)
            labels = batch['label']
            
            outputs = model(sequences, te_features, nonb_features, deterministic=True)
            
            embeddings = outputs['mu']
            logits = outputs['logits']
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())
            all_transcript_ids.extend(batch['transcript_id'])
    
    embeddings = np.vstack(all_embeddings)
    logits = np.vstack(all_logits)
    labels = np.concatenate(all_labels)
    
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    predictions = logits.argmax(axis=1)
    confidences = probs.max(axis=1)
    
    # Load biotypes if available
    biotypes = np.array(['unknown'] * len(labels))
    
    if biotype_csv and lnc_fasta and pc_fasta:
        try:
            from data.cv_utils import (
                load_biotype_mapping,
                extract_biotypes_from_sequences,
                group_rare_biotypes
            )
            from data.loaders import load_sequences_with_labels
            
            print("  Loading biotypes from CSV...")
            sequences, _ = load_sequences_with_labels(lnc_fasta, pc_fasta)
            biotype_lookup = load_biotype_mapping(biotype_csv)
            all_biotypes = extract_biotypes_from_sequences(sequences, biotype_lookup)
            grouped_biotypes = group_rare_biotypes(all_biotypes, min_count=500)
            
            # Create transcript_id -> biotype mapping
            biotype_map = {}
            for seq, biotype in zip(sequences, grouped_biotypes):
                biotype_map[seq.id.split('|')[0]] = biotype
            
            # Map to extracted transcripts
            biotypes = np.array([
                biotype_map.get(tid.split('|')[0], 'unknown')
                for tid in all_transcript_ids
            ])
            
            print(f"  Loaded biotypes for {(biotypes != 'unknown').sum()}/{len(biotypes)} transcripts")
            
        except Exception as e:
            print(f"  Could not load biotypes: {e}")
    
    return {
        'embeddings': embeddings,
        'logits': logits,
        'probs': probs,
        'predictions': predictions,
        'confidences': confidences,
        'labels': labels,
        'transcript_ids': np.array(all_transcript_ids),
        'biotypes': biotypes
    }


def aggregate_fold_predictions(fold_results, all_sequences, labels):
    """
    Aggregate predictions from all folds to create CSV outputs
    
    Args:
        fold_results: List of dicts with fold evaluation results
        all_sequences: Original sequences (in order)
        labels: Original labels (in order)
    
    Returns:
        DataFrame with all sample predictions
    """
    label_to_idx = {'lnc': 0, 'pc': 1}
    
    # Create a dictionary to collect predictions per sample
    sample_data = defaultdict(lambda: {
        'predictions': [],
        'confidences': [],
        'probs': [],
        'folds': [],
        'sample_idx': None
    })
    
    # Collect predictions from each fold
    for fold_result in fold_results:
        fold_idx = fold_result['fold']
        
        # Get predictions from deterministic evaluation
        det = fold_result['deterministic']
        predictions = det['predictions']
        confidences = det['confidences']
        probs = det['probs']
        sample_indices = det['sample_indices']
        
        for idx, pred, conf, prob in zip(sample_indices, predictions, confidences, probs):
            sample_data[idx]['predictions'].append(int(pred))
            sample_data[idx]['confidences'].append(float(conf))
            sample_data[idx]['probs'].append(prob.tolist() if hasattr(prob, 'tolist') else prob)
            sample_data[idx]['folds'].append(fold_idx)
            sample_data[idx]['sample_idx'] = int(idx)
    
    # Create records for DataFrame
    records = []
    
    for sample_idx in sorted(sample_data.keys()):
        data = sample_data[sample_idx]
        
        # Get sequence metadata
        seq = all_sequences[sample_idx]
        transcript_id = seq.id.split('|')[0]
        true_label_str = labels[sample_idx]
        true_label_idx = label_to_idx[true_label_str]
        seq_length = len(str(seq.seq))
        
        # Aggregate predictions
        predictions = np.array(data['predictions'])
        confidences = np.array(data['confidences'])
        
        # Determine consensus prediction (majority vote)
        consensus_pred = int(np.bincount(predictions).argmax())
        consensus_pred_str = 'lnc' if consensus_pred == 0 else 'pc'
        
        # Calculate statistics
        n_folds = len(predictions)
        error_rate = (predictions != true_label_idx).mean()
        mean_confidence = confidences.mean()
        std_confidence = confidences.std() if len(confidences) > 1 else 0.0
        min_confidence = confidences.min()
        
        # Agreement: fraction of folds that agree with consensus
        agreement = (predictions == consensus_pred).mean()
        
        # Hard case definition: error_rate > 0 OR mean_confidence < 0.6
        is_hard_case = (error_rate > 0) or (mean_confidence < 0.6)
        
        # Create fold-level predictions
        fold_predictions_list = []
        for fold_idx, pred, conf, prob in zip(data['folds'], predictions, confidences, data['probs']):
            fold_predictions_list.append({
                'fold': int(fold_idx),
                'prediction': int(pred),
                'confidence': float(conf),
                'probs': prob,
                'true_label': int(true_label_idx),
            })
        
        record = {
            'sample_idx': int(sample_idx),
            'transcript_id': transcript_id,
            'true_label': true_label_str,
            'consensus_prediction': consensus_pred_str,
            'sequence_length': int(seq_length),
            'n_folds': int(n_folds),
            'error_rate': float(error_rate),
            'mean_confidence': float(mean_confidence),
            'std_confidence': float(std_confidence),
            'min_confidence': float(min_confidence),
            'agreement': float(agreement),
            'is_hard_case': bool(is_hard_case),
            'fold_predictions': fold_predictions_list
        }
        
        records.append(record)
    
    return pd.DataFrame(records)


def plot_roc_pr_curves(fold_results, output_dir):
    """
    Generate ROC and PR curves for all folds with mean curves
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('white')
    
    fold_colors = plt.cm.tab10(np.linspace(0, 1, len(fold_results)))
    
    # Storage for interpolated curves
    all_fpr_interp = []
    all_tpr_interp = []
    all_recall_interp = []
    all_precision_interp = []
    
    # =========================================================================
    # Panel 1: ROC Curves
    # =========================================================================
    ax_roc = axes[0]
    base_fpr = np.linspace(0, 1, 100)
    
    for i, fold_result in enumerate(fold_results):
        fold_idx = fold_result['fold']
        det = fold_result['deterministic']
        
        y_true = det['labels']
        y_scores = det['probs'][:, 1]  # Probability of class 1 (protein-coding)
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Interpolate
        tpr_interp = np.interp(base_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        all_fpr_interp.append(base_fpr)
        all_tpr_interp.append(tpr_interp)
        
        # Plot fold curve
        ax_roc.plot(fpr, tpr, color=fold_colors[i], alpha=0.4, linewidth=1.5,
                   label=f'Fold {fold_idx} (AUC = {roc_auc:.3f})')
    
    # Mean ROC curve
    mean_tpr = np.mean(all_tpr_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(base_fpr, mean_tpr)
    std_auc = np.std([auc(base_fpr, t) for t in all_tpr_interp])
    std_tpr = np.std(all_tpr_interp, axis=0)
    
    ax_roc.plot(base_fpr, mean_tpr, color='#2C3E50', linewidth=3,
               label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})',
               zorder=10)
    
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax_roc.fill_between(base_fpr, tpr_lower, tpr_upper, color='#2C3E50', alpha=0.2,
                        label='± 1 std. dev.')
    
    ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Chance')
    
    ax_roc.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax_roc.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax_roc.set_title('ROC Curves (5-Fold Cross-Validation)', fontsize=15, fontweight='bold', pad=15)
    ax_roc.legend(loc='lower right', fontsize=9, framealpha=0.95)
    ax_roc.grid(True, alpha=0.3, linestyle='--')
    ax_roc.set_xlim([0, 1])
    ax_roc.set_ylim([0, 1])
    
    # =========================================================================
    # Panel 2: Precision-Recall Curves
    # =========================================================================
    ax_pr = axes[1]
    base_recall = np.linspace(0, 1, 100)
    
    all_ap = []
    for i, fold_result in enumerate(fold_results):
        fold_idx = fold_result['fold']
        det = fold_result['deterministic']
        
        y_true = det['labels']
        y_scores = det['probs'][:, 1]
        
        # Compute PR curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        all_ap.append(avg_precision)
        
        # Interpolate
        precision_interp = np.interp(base_recall, recall[::-1], precision[::-1])
        all_recall_interp.append(base_recall)
        all_precision_interp.append(precision_interp)
        
        # Plot fold curve
        ax_pr.plot(recall, precision, color=fold_colors[i], alpha=0.4, linewidth=1.5,
                  label=f'Fold {fold_idx} (AP = {avg_precision:.3f})')
    
    # Mean PR curve
    mean_precision = np.mean(all_precision_interp, axis=0)
    std_precision = np.std(all_precision_interp, axis=0)
    mean_ap = np.mean(all_ap)
    std_ap = np.std(all_ap)
    
    ax_pr.plot(base_recall, mean_precision, color='#2C3E50', linewidth=3,
              label=f'Mean PR (AP = {mean_ap:.3f} ± {std_ap:.3f})',
              zorder=10)
    
    precision_upper = np.minimum(mean_precision + std_precision, 1)
    precision_lower = np.maximum(mean_precision - std_precision, 0)
    ax_pr.fill_between(base_recall, precision_lower, precision_upper, 
                       color='#2C3E50', alpha=0.2, label='± 1 std. dev.')
    
    # Baseline
    n_pc = sum(fold_results[0]['deterministic']['labels'] == 1)
    n_total = len(fold_results[0]['deterministic']['labels'])
    baseline = n_pc / n_total
    ax_pr.axhline(y=baseline, color='k', linestyle='--', linewidth=2, alpha=0.5,
                 label=f'Baseline ({baseline:.3f})')
    
    ax_pr.set_xlabel('Recall', fontsize=13, fontweight='bold')
    ax_pr.set_ylabel('Precision', fontsize=13, fontweight='bold')
    ax_pr.set_title('Precision-Recall Curves (5-Fold CV)', fontsize=15, fontweight='bold', pad=15)
    ax_pr.legend(loc='lower left', fontsize=9, framealpha=0.95)
    ax_pr.grid(True, alpha=0.3, linestyle='--')
    ax_pr.set_xlim([0, 1])
    ax_pr.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_pr_curves.png', dpi=350, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n  Saved ROC/PR curves: {output_dir / 'roc_pr_curves.png'}")
    print(f"   Mean AUC-ROC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"   Mean AP (PR): {mean_ap:.4f} ± {std_ap:.4f}")
    
    return {
        'mean_auc_roc': mean_auc,
        'std_auc_roc': std_auc,
        'mean_ap_pr': mean_ap,
        'std_ap_pr': std_ap
    }


def convert_to_serializable(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj

def evaluate_test_set(config, experiment_dir, model_builder,
                      test_lnc_fasta, test_pc_fasta, n_folds,
                      batch_size, device):
    """Ensemble evaluation on independent held-out test set."""
    test_dataset = SequenceFeatureDataset(
        lnc_fasta=test_lnc_fasta,
        pc_fasta=test_pc_fasta,
        max_length=config.get('model', 'max_length'),
        te_features_csv=config.get('data', 'te_features_csv'),
        nonb_features_csv=config.get('data', 'nonb_features_csv'),
        te_scaler_path=config.get('data', 'te_scaler'),
        nonb_scaler_path=config.get('data', 'nonb_scaler')
    )
    print(f"  Test samples: {len(test_dataset):,}")

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=1)

    n_samples      = len(test_dataset)
    sum_probs      = np.zeros((n_samples, 2), dtype=np.float64)
    labels_arr     = None
    fold_probs_all = []  # per-fold probabilities for std/min confidence
    all_transcript_ids = []  # collected on fold 0

    for fold_idx in range(n_folds):
        ckpt_path  = experiment_dir / 'models' / f'fold_{fold_idx}_best.pt'
        checkpoint = torch.load(ckpt_path, map_location=device)
        model      = model_builder()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"  Fold {fold_idx} | epoch={checkpoint['epoch']} "
              f"val_acc={checkpoint['val_acc']:.4f}")

        fold_probs, fold_labels = [], []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"  Fold {fold_idx}", leave=False):
                seq  = batch['sequence'].to(device)
                te   = batch['te_features'].to(device)
                nonb = batch['nonb_features'].to(device)
                out  = model(seq, te, nonb, deterministic=True)
                fold_probs.append(torch.softmax(out['logits'], dim=1).cpu().numpy())
                fold_labels.append(batch['label'].numpy())
                if fold_idx == 0:
                    all_transcript_ids.extend(batch['transcript_id'])

        fp         = np.vstack(fold_probs)
        fold_probs_all.append(fp)
        sum_probs += fp
        labels_arr = np.concatenate(fold_labels)

    avg_probs   = sum_probs / n_folds
    predictions = avg_probs.argmax(axis=1)

    acc                       = accuracy_score(labels_arr, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_arr, predictions, average='binary', zero_division=0)
    cm = confusion_matrix(labels_arr, predictions)

    print(f"\n  Test Results (ensemble of {n_folds} folds):")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1:        {f1:.4f}")

    test_metrics = {
        'accuracy': float(acc), 'precision': float(precision),
        'recall': float(recall), 'f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'n_samples': int(n_samples),
        'n_lncrna': int((labels_arr == 0).sum()),
        'n_pcrna':  int((labels_arr == 1).sum()),
        'n_folds_ensembled': n_folds
    }
    with open(experiment_dir / 'test_results.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"  Saved to: {experiment_dir / 'test_results.json'}")

    # --- per-sample test predictions CSV ---
    idx_to_label = {0: 'lnc', 1: 'pc'}
    # stack fold probs: shape (n_folds, n_samples, 2)
    fold_probs_stack = np.stack(fold_probs_all, axis=0)
    # per-fold max confidence per sample: shape (n_folds, n_samples)
    fold_confidences = fold_probs_stack.max(axis=2)

    records = []
    for i in range(n_samples):
        prob         = avg_probs[i]
        pred         = int(predictions[i])
        true         = int(labels_arr[i])
        confidence   = float(prob.max())
        true_str     = idx_to_label[true]
        pred_str     = idx_to_label[pred]
        error_rate   = float(pred != true)
        is_hard_case = error_rate > 0 or confidence < 0.6

        records.append({
            'sample_idx':           i,
            'transcript_id':        all_transcript_ids[i],
            'true_label':           true_str,
            'consensus_prediction': pred_str,
            'sequence_length':      int(len(str(test_dataset.sequences[i].seq))),
            'n_folds':              n_folds,
            'error_rate':           error_rate,
            'mean_confidence':      confidence,
            'std_confidence':       float(fold_confidences[:, i].std()),
            'min_confidence':       float(fold_confidences[:, i].min()),
            'agreement':            float((fold_probs_stack[:, i].argmax(axis=1) == pred).mean()),
            'is_hard_case':         bool(is_hard_case),
        })

    csv_dir = experiment_dir / 'evaluation_csvs'
    csv_dir.mkdir(exist_ok=True)
    test_pred_df = pd.DataFrame(records)
    test_pred_df.to_csv(csv_dir / 'test_predictions.csv', index=False)
    hard_df = test_pred_df[test_pred_df['is_hard_case']]
    hard_df.to_csv(csv_dir / 'test_hard_cases.csv', index=False)
    print(f"  Saved test_predictions.csv ({len(test_pred_df):,} samples, "
          f"{len(hard_df):,} hard cases)")

    return test_metrics

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate all CV folds for β-VAE with Genomic Features'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='Path to experiment directory with trained models')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of folds')
    parser.add_argument('--device', type=str,
                       default='cuda:0' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--test_lnc_fasta', type=str, default=None,
                   help='Independent test set lncRNA FASTA (holdout evaluation)')
    parser.add_argument('--test_pc_fasta', type=str, default=None,
                   help='Independent test set pcRNA FASTA (holdout evaluation)')
    parser.add_argument('--biotype_csv', type=str, required=True,
                       help='Path to biotype CSV for biotype labeling in embeddings')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for evaluation')
    parser.add_argument('--extract_all_folds', action='store_true',
                       help='Extract embeddings from ALL folds (not just best)')
    parser.add_argument('--generate_hard_case_csvs', action='store_true',
                       help='Generate all_sample_predictions.csv and hard_cases.csv')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    experiment_dir = Path(args.experiment_dir)
    device = torch.device(args.device)
    
    print("=" * 80)
    print("β-VAE WITH GENOMIC FEATURES - CROSS-VALIDATION EVALUATION")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Experiment: {experiment_dir}")
    print(f"Folds: {args.n_folds}")
    print(f"Device: {device}")
    print(f"Extract all folds: {args.extract_all_folds}")
    print(f"Generate hard case CSVs: {args.generate_hard_case_csvs}")
    print("=" * 80)
    
    # Check fold checkpoints
    print("\nChecking fold checkpoints...")
    missing_folds = []
    for fold_idx in range(args.n_folds):
        checkpoint_path = experiment_dir / 'models' / f'fold_{fold_idx}_best.pt'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f" Fold {fold_idx}: {checkpoint_path}")
            print(f"      Epoch {checkpoint.get('epoch', '?')}, "
                  f"Val Acc: {checkpoint.get('val_acc', '?'):.4f}")
        else:
            print(f" Fold {fold_idx}: NOT FOUND")
            missing_folds.append(fold_idx)
    
    if missing_folds:
        print(f"\nERROR: Missing checkpoints for folds: {missing_folds}")
        return
    
    # Load sequences in SAME ORDER as training (pc first, then lnc)
    print("\nLoading sequences from FASTA files...")
    from Bio import SeqIO
    
    lnc_seqs = list(SeqIO.parse(config.get('data', 'lnc_fasta'), 'fasta'))
    pc_seqs = list(SeqIO.parse(config.get('data', 'pc_fasta'), 'fasta'))
    
    print(f"  lncRNA: {len(lnc_seqs):,}")
    print(f"  Protein-coding: {len(pc_seqs):,}")
    
    # Combine in same order as training
    all_sequences, labels = load_sequences_in_order(
        config.get('data', 'lnc_fasta'),
        config.get('data', 'pc_fasta')
    )
    
    print(f"  Total sequences: {len(all_sequences):,}")
    
    # Create CV splits
    n_bins = config.get('training', 'n_bins', default=5)
    random_state = config.get('training', 'random_state', default=42)
    
    print(f"\nCreating CV splits (n_bins={n_bins}, random_state={random_state})...")
    strat_groups = create_length_stratified_groups(all_sequences, labels, n_bins=n_bins)
    print(f"  Created {len(set(strat_groups))} stratification groups")
    
    skf = get_cv_splitter(n_folds=args.n_folds, random_state=random_state)
    
    # Create the dataset (after we've established the sequence order)
    print("\nLoading full dataset...")
    full_dataset = SequenceFeatureDataset(
        lnc_fasta=config.get('data', 'lnc_fasta'),
        pc_fasta=config.get('data', 'pc_fasta'),
        max_length=config.get('model', 'max_length'),
        te_features_csv=config.get('data', 'te_features_csv'),
        nonb_features_csv=config.get('data', 'nonb_features_csv'),
        te_scaler_path=config.get('data', 'te_scaler'),
        nonb_scaler_path=config.get('data', 'nonb_scaler')
    )
    
    print(f"  Dataset created: {len(full_dataset):,} samples")

    # Sanity Check: Verify sequence order matches training
    verify_sequence_order(all_sequences, labels, full_dataset.sequences)
    
    # Create model builder
    model_builder = create_model_builder(config)
    
    # Evaluate each fold
    print("\n" + "=" * 80)
    print("EVALUATING FOLDS")
    print("=" * 80)
    
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_sequences, strat_groups)):
        print(f"\n{'='*80}")
        print(f"Fold {fold_idx}")
        print(f"{'='*80}")
        
        print(f"  Validation samples: {len(val_idx):,}")
        
        # Create validation dataset
        val_dataset = Subset(full_dataset, val_idx)
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )
        
        # Load model
        checkpoint_path = experiment_dir / 'models' / f'fold_{fold_idx}_best.pt'
        print(f"\n  Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"    Saved at epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"    Val acc at save: {checkpoint.get('val_acc', 'unknown')}")
        
        model = model_builder()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        print(f"\n{'='*80}")
        print(f"VAL_LOADER DIAGNOSTIC - Fold {fold_idx}")
        print(f"{'='*80}")

        # Count label distribution in val_loader
        all_val_labels = []
        for batch in val_loader:
            all_val_labels.extend(batch['label'].tolist())

        from collections import Counter
        val_label_counts = Counter(all_val_labels)
        print(f"\nVal loader label distribution:")
        print(f"  Label 0 (lnc): {val_label_counts[0]:,}")
        print(f"  Label 1 (pc): {val_label_counts[1]:,}")
        print(f"  Total: {len(all_val_labels):,}")

        # Compare to expected from val_idx
        expected_label_counts = Counter([0 if labels[i] == 'lnc' else 1 for i in val_idx])
        print(f"\nExpected label distribution (from val_idx):")
        print(f"  Label 0 (lnc): {expected_label_counts[0]:,}")
        print(f"  Label 1 (pc): {expected_label_counts[1]:,}")

        if val_label_counts == expected_label_counts:
            print(f"  ✓ Label distributions match!")
        else:
            print(f"  ✗ Label distributions DO NOT match!")

        print(f"{'='*80}\n")
        
        # Evaluate
        print(f"\n  Evaluating...")
        
        # Deterministic evaluation
        det_results = evaluate_deterministic(model, val_loader, device)
        
        # Print results
        print(f"\n  Results (Deterministic):")
        print(f"    Accuracy:  {det_results['accuracy']:.4f}")
        print(f"    Precision: {det_results['precision']:.4f}")
        print(f"    Recall:    {det_results['recall']:.4f}")
        print(f"    F1 Score:  {det_results['f1']:.4f}")
        print(f"    Mean Conf: {det_results['confidences'].mean():.4f} ± {det_results['confidences'].std():.4f}")
        
        # Store results
        fold_results.append({
            'fold': fold_idx,
            'deterministic': {
                'accuracy': det_results['accuracy'],
                'precision': det_results['precision'],
                'recall': det_results['recall'],
                'f1': det_results['f1'],
                'confusion_matrix': det_results['confusion_matrix'],
                'mean_confidence': det_results['confidences'].mean(),
                'std_confidence': det_results['confidences'].std(),
                'predictions': det_results['predictions'],
                'labels': det_results['labels'],
                'probs': det_results['probs'],
                'confidences': det_results['confidences'],
                'sample_indices': val_idx  # IMPORTANT: Add this for hard case CSV generation
            },
            'n_samples': len(val_idx),
            'val_indices': val_idx
        })
    
    # Compute overall statistics
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 80)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    summary = {}
    
    for metric in metrics:
        values = [r['deterministic'][metric] for r in fold_results]
        summary[f"deterministic_{metric}"] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values
        }
    
    # Print summary
    print(f"\nOVERALL STATISTICS:")
    print(f"{'Metric':<15} {'Mean ± Std':<20} {'Min':<10} {'Max':<10}")
    print("-" * 60)
    
    for metric in metrics:
        mean = summary[f"deterministic_{metric}"]['mean']
        std = summary[f"deterministic_{metric}"]['std']
        min_val = summary[f"deterministic_{metric}"]['min']
        max_val = summary[f"deterministic_{metric}"]['max']
        
        print(f"{metric.capitalize():<15} {mean:.4f} ± {std:.4f}     {min_val:.4f}    {max_val:.4f}")
    
    # Generate hard case CSVs if requested
    if args.generate_hard_case_csvs:
        print("\n" + "=" * 80)
        print("GENERATING HARD CASE CSVs")
        print("=" * 80)
        
        all_predictions_df = aggregate_fold_predictions(
            fold_results, all_sequences, labels
        )
        
        # Summary statistics
        n_hard = all_predictions_df['is_hard_case'].sum()
        n_total = len(all_predictions_df)
        
        print(f"\nHard Case Statistics:")
        print(f"  Total samples: {n_total:,}")
        print(f"  Hard cases: {n_hard:,} ({100*n_hard/n_total:.1f}%)")
        print(f"  Easy cases: {n_total - n_hard:,} ({100*(1 - n_hard/n_total):.1f}%)")
        
        # By class
        for label in ['lnc', 'pc']:
            n_class = (all_predictions_df['true_label'] == label).sum()
            n_class_hard = ((all_predictions_df['true_label'] == label) & all_predictions_df['is_hard_case']).sum()
            print(f"  {label}:")
            print(f"    Total: {n_class:,}")
            print(f"    Hard: {n_class_hard:,} ({100*n_class_hard/n_class:.1f}%)")
        
        # Save CSVs
        csv_dir = experiment_dir / 'evaluation_csvs'
        csv_dir.mkdir(exist_ok=True)
        
        # 1. All sample predictions
        all_predictions_csv = csv_dir / 'all_sample_predictions.csv'
        all_predictions_df.to_csv(all_predictions_csv, index=False)
        print(f"\n  Saved: {all_predictions_csv}")
        
        # 2. Hard cases only
        hard_cases_df = all_predictions_df[all_predictions_df['is_hard_case']].copy()
        hard_cases_csv = csv_dir / 'hard_cases.csv'
        hard_cases_df.to_csv(hard_cases_csv, index=False)
        print(f"  Saved: {hard_cases_csv}")
        
        print(f"\nCSVs contain the following fields:")
        print(f"  - sample_idx, transcript_id, true_label, consensus_prediction")
        print(f"  - sequence_length, n_folds")
        print(f"  - error_rate, mean_confidence, std_confidence, min_confidence")
        print(f"  - agreement, is_hard_case, fold_predictions")
    
    # Generate ROC/PR curves
    print("\n" + "=" * 80)
    print("GENERATING ROC/PR CURVES")
    print("=" * 80)
    
    figures_dir = experiment_dir / 'performance_figures'
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    roc_pr_metrics = plot_roc_pr_curves(fold_results, figures_dir)
    
    # Extract embeddings
    if args.extract_all_folds:
        print("\n" + "=" * 80)
        print("EXTRACTING EMBEDDINGS FROM ALL FOLDS")
        print("=" * 80)
        
        all_fold_data = {}
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_sequences, strat_groups)):
            print(f"\nExtracting embeddings from fold {fold_idx}...")
            
            val_dataset = Subset(full_dataset, val_idx)
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4
            )
            
            checkpoint_path = experiment_dir / 'models' / f'fold_{fold_idx}_best.pt'
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            model = model_builder()
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            
            embeddings_dict = extract_embeddings(
                model, val_loader, device,
                biotype_csv=args.biotype_csv,
                lnc_fasta=config.get('data', 'lnc_fasta'),
                pc_fasta=config.get('data', 'pc_fasta')
            )
            
            # Store with fold prefix (multi-fold format for UMAP compatibility)
            for key, value in embeddings_dict.items():
                all_fold_data[f'fold_{fold_idx}_{key}'] = value
            
            print(f"  Extracted {embeddings_dict['embeddings'].shape}")
        
        # Save all fold embeddings in multi-fold format
        save_path = experiment_dir / 'embeddings_all_folds.npz'
        np.savez(save_path, **all_fold_data)
        print(f"\n  Saved all-fold embeddings to: {save_path}")
        
        # Print what was saved
        print(f"\n  Saved keys (sample):")
        for key in sorted(list(all_fold_data.keys()))[:10]:
            print(f"    {key}")
        
    else:
        print("\n" + "=" * 80)
        print("EXTRACTING EMBEDDINGS FROM BEST FOLD")
        print("=" * 80)
        
        # Find best fold
        best_fold_idx = max(range(len(fold_results)), 
                           key=lambda i: fold_results[i]['deterministic']['accuracy'])
        best_fold = fold_results[best_fold_idx]
        
        print(f"\nBest fold: {best_fold_idx} (accuracy: {best_fold['deterministic']['accuracy']:.4f})")
        
        # Extract embeddings
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_sequences, strat_groups)):
            if fold_idx != best_fold_idx:
                continue
            
            print(f"Extracting embeddings from fold {fold_idx}...")
            
            val_dataset = Subset(full_dataset, val_idx)
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4
            )
            
            checkpoint_path = experiment_dir / 'models' / f'fold_{fold_idx}_best.pt'
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            model = model_builder()
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            
            embeddings_dict = extract_embeddings(
                model, val_loader, device,
                biotype_csv=args.biotype_csv,
                lnc_fasta=config.get('data', 'lnc_fasta'),
                pc_fasta=config.get('data', 'pc_fasta')
            )
            
            # Save with fold prefix
            save_data = {}
            for key, value in embeddings_dict.items():
                save_data[f'fold_{fold_idx}_{key}'] = value
            
            save_path = experiment_dir / 'embeddings_best_fold.npz'
            np.savez(save_path, **save_data)
            
            print(f"  Saved embeddings to: {save_path}")
            print(f"  Shape: {embeddings_dict['embeddings'].shape}")
            
            break
    
    # Save evaluation results
    output_file = experiment_dir / 'cv_evaluation_results.json'
    
    # Prepare data for JSON (remove large arrays)
    fold_results_for_json = []
    for r in fold_results:
        r_copy = {
            'fold': r['fold'],
            'n_samples': r['n_samples'],
            'deterministic': {
                'accuracy': r['deterministic']['accuracy'],
                'precision': r['deterministic']['precision'],
                'recall': r['deterministic']['recall'],
                'f1': r['deterministic']['f1'],
                'confusion_matrix': r['deterministic']['confusion_matrix'],
                'mean_confidence': r['deterministic']['mean_confidence'],
                'std_confidence': r['deterministic']['std_confidence']
            }
        }
        fold_results_for_json.append(r_copy)
    
    data_to_save = {
        'summary': summary,
        'roc_pr_metrics': roc_pr_metrics,
        'fold_results': fold_results_for_json,
        'n_folds': args.n_folds,
        'total_samples': len(full_dataset),
        'config': {
            'n_bins': n_bins,
            'random_state': random_state
        }
    }
    
    data_serializable = convert_to_serializable(data_to_save)
    
    with open(output_file, 'w') as f:
        json.dump(data_serializable, f, indent=2)
    
    print(f"\n  Results saved to: {output_file}")
    
    # Save per-fold CSV
    csv_rows = []
    for r in fold_results:
        csv_rows.append({
            'fold': r['fold'],
            'accuracy': r['deterministic']['accuracy'],
            'precision': r['deterministic']['precision'],
            'recall': r['deterministic']['recall'],
            'f1': r['deterministic']['f1'],
            'n_samples': r['n_samples']
        })
    
    csv_file = experiment_dir / 'cv_fold_results.csv'
    pd.DataFrame(csv_rows).to_csv(csv_file, index=False)
    print(f"  Per-fold CSV saved to: {csv_file}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nKey Results (Deterministic - REPORT THESE):")
    print(f"  Accuracy:  {summary['deterministic_accuracy']['mean']:.4f} ± {summary['deterministic_accuracy']['std']:.4f}")
    print(f"  Precision: {summary['deterministic_precision']['mean']:.4f} ± {summary['deterministic_precision']['std']:.4f}")
    print(f"  Recall:    {summary['deterministic_recall']['mean']:.4f} ± {summary['deterministic_recall']['std']:.4f}")
    print(f"  F1 Score:  {summary['deterministic_f1']['mean']:.4f} ± {summary['deterministic_f1']['std']:.4f}")
    print(f"  AUC-ROC:   {roc_pr_metrics['mean_auc_roc']:.4f} ± {roc_pr_metrics['std_auc_roc']:.4f}")
    print(f"  AP (PR):   {roc_pr_metrics['mean_ap_pr']:.4f} ± {roc_pr_metrics['std_ap_pr']:.4f}")
    
    if args.generate_hard_case_csvs:
        print(f"\nHard case CSVs saved to: {experiment_dir / 'evaluation_csvs'}/")

    if args.test_lnc_fasta and args.test_pc_fasta:
        print("\n" + "=" * 80)
        print("INDEPENDENT TEST SET EVALUATION (ENSEMBLE)")
        print("=" * 80)
        evaluate_test_set(config, experiment_dir, model_builder,
                        args.test_lnc_fasta, args.test_pc_fasta,
                        args.n_folds, args.batch_size, device)

if __name__ == '__main__':
    main()