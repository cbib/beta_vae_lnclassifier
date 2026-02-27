#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate all trained folds and compute cross-validation statistics

This script:
1. Loads each fold's checkpoint
2. Evaluates on its validation set using shared evaluation utilities
3. Computes overall CV statistics (mean ± std)
4. Extracts and saves embeddings from best fold (or all folds)
5. Generates hard case CSVs for downstream analysis (Optional)
6. Generates comprehensive results

Usage:
    # Evaluate all folds and extract best fold embeddings
    python evaluate_cv_fold.py \
        --config configs/beta_vae_contrastive_g47.json \
        --experiment_dir gencode_v47_experiments/beta_vae_g47 \
        --n_folds 5
    
    # Extract embeddings from ALL folds and generate hard case CSVs
    python evaluate_cv_fold.py \
        --config configs/beta_vae_contrastive_g47.json \
        --experiment_dir gencode_v47_experiments/beta_vae_g47 \
        --n_folds 5 \
        --extract_all_folds \
        --generate_hard_case_csvs
"""
import sys
import argparse
import torch
import torch.nn.functional as F
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from collections import defaultdict
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

from data.loaders import load_sequences_with_labels
from data.preprocessing import SequencePreprocessor, RNASequenceBiotypeDataset
from data.cv_utils import (
    create_length_stratified_groups,
    load_biotype_mapping,
    extract_biotypes_from_sequences,
    group_rare_biotypes,
    get_cv_splitter,
    DEFAULT_N_BINS,
    DEFAULT_RANDOM_STATE
)
from analysis.embeddings.embedding_utils import (
    extract_embeddings_from_model,
    extract_embeddings_all_folds,
    save_embeddings
)
from models import get_contrastive_model
from configs.load_config import load_config


def create_model_builder(config):
    """Create model builder function"""
    
    def builder():
        from models.beta_vae_contrastive_model import BetaVAE_Contrastive
        from models.cnn_contrastive import CNNContrastive
        
        if config.get('model', 'architecture') == 'cnn':
            model = CNNContrastive(
                input_length=config.get('model', 'max_length'),
                input_dim=config.get('model', 'input_dim'),
                embedding_dim=config.get('model', 'embedding_dim'),
                projection_dim=config.get('model', 'projection_dim'),
                num_classes=2,
                dropout_rate=config.get('model', 'dropout_rate', default=0.3),
                use_cse=config.get('model', 'use_cse', default=True),
                cse_d_model=config.get('model', 'cse_d_model', default=128),
                cse_kernel_size=config.get('model', 'cse_kernel_size', default=9)
            )
        else:
            model = BetaVAE_Contrastive(
                input_dim=config.get('model', 'input_dim'),
                input_length=config.get('model', 'max_length'),
                latent_dim=config.get('model', 'latent_dim'),
                beta=config.get('training', 'beta', default=4.0),
                num_classes=config.get('model', 'num_classes', default=2),
                dropout_rate=config.get('model', 'dropout_rate', default=0.3),
                use_cse=config.get('model', 'use_cse', default=True),
                cse_d_model=config.get('model', 'cse_d_model', default=128),
                cse_kernel_size=config.get('model', 'cse_kernel_size', default=9)
            )
        return model
    
    return builder


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


def aggregate_fold_predictions(fold_results, matched_seqs, matched_labels, grouped_biotypes):
    """
    Aggregate predictions from all folds to create CSV outputs
    
    Args:
        fold_results: List of dicts with fold evaluation results
        matched_seqs: Original sequences
        matched_labels: Original labels
        grouped_biotypes: Biotype labels
    
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
        seq = matched_seqs[sample_idx]
        transcript_id = seq.id
        true_label_str = matched_labels[sample_idx]
        true_label_idx = label_to_idx[true_label_str]
        biotype = grouped_biotypes[sample_idx]
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
                'biotype': biotype
            })
        
        record = {
            'sample_idx': int(sample_idx),
            'transcript_id': transcript_id,
            'true_label': true_label_str,
            'consensus_prediction': consensus_pred_str,
            'biotype': biotype,
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

def plot_roc_pr_curves(fold_results, matched_labels, output_dir):
    """
    Generate ROC and PR curves for all folds with mean curves
    
    Args:
        fold_results: List of fold evaluation results with predictions and probs
        output_dir: Directory to save figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('white')
    
    # Colors for folds
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
        
        # Get probabilities for positive class (protein-coding = class 1)
        y_true = []
        y_scores = []
        
        for sample_idx, prob in zip(det['sample_indices'], det['probs']):
            true_label_str = matched_labels[sample_idx]
            true_label = 1 if true_label_str == 'pc' else 0
            y_true.append(true_label)
            # prob is [p_lnc, p_pc], we want p_pc
            y_scores.append(prob[1] if isinstance(prob, (list, np.ndarray)) else prob)
        
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Interpolate for mean calculation
        tpr_interp = np.interp(base_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        all_fpr_interp.append(base_fpr)
        all_tpr_interp.append(tpr_interp)
        
        # Plot fold curve
        ax_roc.plot(fpr, tpr, color=fold_colors[i], alpha=0.4, linewidth=1.5,
                   label=f'Fold {fold_idx} (AUC = {roc_auc:.3f})')
    
    # Plot mean ROC curve
    mean_tpr = np.mean(all_tpr_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(base_fpr, mean_tpr)
    std_tpr = np.std(all_tpr_interp, axis=0)
    
    ax_roc.plot(base_fpr, mean_tpr, color='#2C3E50', linewidth=3,
               label=f'Mean ROC (AUC = {mean_auc:.3f} ± {np.std([auc(base_fpr, t) for t in all_tpr_interp]):.3f})',
               zorder=10)
    
    # Confidence interval
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax_roc.fill_between(base_fpr, tpr_lower, tpr_upper, color='#2C3E50', alpha=0.2,
                        label='± 1 std. dev.')
    
    # Diagonal reference line
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
    
    for i, fold_result in enumerate(fold_results):
        fold_idx = fold_result['fold']
        det = fold_result['deterministic']
        
        # Get probabilities
        y_true = []
        y_scores = []
        
        for sample_idx, prob in zip(det['sample_indices'], det['probs']):
            true_label_str = matched_labels[sample_idx]
            true_label = 1 if true_label_str == 'pc' else 0
            y_true.append(true_label)
            y_scores.append(prob[1] if isinstance(prob, (list, np.ndarray)) else prob)
        
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        # Compute PR curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        
        # Interpolate (reverse because recall goes high to low)
        precision_interp = np.interp(base_recall, recall[::-1], precision[::-1])
        all_recall_interp.append(base_recall)
        all_precision_interp.append(precision_interp)
        
        # Plot fold curve
        ax_pr.plot(recall, precision, color=fold_colors[i], alpha=0.4, linewidth=1.5,
                  label=f'Fold {fold_idx} (AP = {avg_precision:.3f})')
    
    # Plot mean PR curve
    mean_precision = np.mean(all_precision_interp, axis=0)
    std_precision = np.std(all_precision_interp, axis=0)
    mean_ap = np.mean([average_precision_score(
        np.array([1 if matched_labels[si] == 'pc' else 0 for si in fold_result['deterministic']['sample_indices']]),
        np.array([p[1] if isinstance(p, (list, np.ndarray)) else p for p in fold_result['deterministic']['probs']])
    ) for fold_result in fold_results])
    std_ap = np.std([average_precision_score(
        np.array([1 if matched_labels[si] == 'pc' else 0 for si in fold_result['deterministic']['sample_indices']]),
        np.array([p[1] if isinstance(p, (list, np.ndarray)) else p for p in fold_result['deterministic']['probs']])
    ) for fold_result in fold_results])
    
    ax_pr.plot(base_recall, mean_precision, color='#2C3E50', linewidth=3,
              label=f'Mean PR (AP = {mean_ap:.3f} ± {std_ap:.3f})',
              zorder=10)
    
    # Confidence interval
    precision_upper = np.minimum(mean_precision + std_precision, 1)
    precision_lower = np.maximum(mean_precision - std_precision, 0)
    ax_pr.fill_between(base_recall, precision_lower, precision_upper, 
                       color='#2C3E50', alpha=0.2, label='± 1 std. dev.')
    
    # Baseline (random classifier based on class distribution)
    n_pc = sum(1 for label in matched_labels if label == 'pc')
    baseline = n_pc / len(matched_labels)
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
    plt.savefig(output_dir / 'roc_pr_curves.png', dpi=1200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n  Saved ROC/PR curves: {output_dir / 'roc_pr_curves.png'}")
    print(f"   Mean AUC-ROC: {mean_auc:.4f}")
    print(f"   Mean AP (PR): {mean_ap:.4f}")
    
    return {
        'mean_auc_roc': mean_auc,
        'std_auc_roc': np.std([auc(base_fpr, t) for t in all_tpr_interp]),
        'mean_ap_pr': mean_ap,
        'std_ap_pr': std_ap
    }

def evaluate_test_set(config, experiment_dir, model_builder,
                      test_lnc_fasta, test_pc_fasta, n_folds,
                      batch_size, device, preprocessor,
                      biotype_to_idx):
    """Ensemble evaluation on independent held-out test set."""
    from data.loaders import load_sequences_with_labels

    test_seqs, test_labels = load_sequences_with_labels(test_lnc_fasta, test_pc_fasta)
    print(f"  Test samples: {len(test_seqs):,}")

    # Use the first available biotype as a safe placeholder
    # It doesn't affect metrics, only needs to be a valid key
    fallback_biotype = next(iter(biotype_to_idx.keys()))
    test_biotypes = [fallback_biotype] * len(test_seqs)

    test_dataset = RNASequenceBiotypeDataset(
        test_seqs, test_labels, preprocessor,
        biotype_labels=test_biotypes,
        biotype_to_idx=biotype_to_idx
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=1)

    n_samples     = len(test_dataset)
    sum_probs     = np.zeros((n_samples, 2), dtype=np.float64)
    labels_arr    = None
    fold_probs_all = []  # per-fold probabilities for std/min confidence

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
                sequences = batch['sequence'].to(device)
                out       = model(sequences)
                logits    = out['logits'] if isinstance(out, dict) else out
                fold_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                fold_labels.append(batch['label'].numpy())

        fp          = np.vstack(fold_probs)
        fold_probs_all.append(fp)
        sum_probs  += fp
        labels_arr  = np.concatenate(fold_labels)

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
        prob        = avg_probs[i]
        pred        = int(predictions[i])
        true        = int(labels_arr[i])
        confidence  = float(prob.max())
        true_str    = idx_to_label[true]
        pred_str    = idx_to_label[pred]
        error_rate  = float(pred != true)
        is_hard_case = error_rate > 0 or confidence < 0.6
        transcript_id = test_seqs[i].id.split('|')[0]

        records.append({
            'sample_idx':           i,
            'transcript_id':        transcript_id,
            'true_label':           true_str,
            'consensus_prediction': pred_str,
            'sequence_length':      int(len(str(test_seqs[i].seq))),
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
    parser = argparse.ArgumentParser(description='Evaluate all CV folds with proper embedding extraction')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='Path to experiment directory with trained models')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of folds')
    parser.add_argument('--device', type=str,
                       default='cuda:0' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--biotype_csv', type=str,
                       default='${DATA_ROOT}/data/dataset_biotypes/g49_dataset_biotypes_cdhit.csv',
                       help='Path to biotype CSV')
    parser.add_argument('--test_lnc_fasta', type=str, default=None,
                   help='Independent test set lncRNA FASTA (holdout evaluation)')
    parser.add_argument('--test_pc_fasta', type=str, default=None,
                   help='Independent test set pcRNA FASTA (holdout evaluation)')
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
    print("CROSS-VALIDATION EVALUATION (WITH PROPER EMBEDDINGS)")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Experiment: {experiment_dir}")
    print(f"Folds: {args.n_folds}")
    print(f"Device: {device}")
    print(f"CV Config: n_bins={DEFAULT_N_BINS}, random_state={DEFAULT_RANDOM_STATE}")
    print(f"Extract all folds: {args.extract_all_folds}")
    print(f"Generate hard case CSVs: {args.generate_hard_case_csvs}")
    print("=" * 80)
    
    # Check that all fold checkpoints exist
    print("\nChecking fold checkpoints...")
    missing_folds = []
    for fold_idx in range(args.n_folds):
        checkpoint_path = experiment_dir / 'models' / f'fold_{fold_idx}_best.pt'
        if checkpoint_path.exists():
            print(f"    Fold {fold_idx}: {checkpoint_path}")
        else:
            print(f"  ✗ Fold {fold_idx}: NOT FOUND")
            missing_folds.append(fold_idx)
    
    if missing_folds:
        print(f"\nERROR: Missing checkpoints for folds: {missing_folds}")
        print("Cannot proceed with evaluation.")
        return
    
    # Load sequences
    print("\nLoading sequences...")
    matched_seqs, matched_labels = load_sequences_with_labels(
        config.get('data', 'lnc_fasta'),
        config.get('data', 'pc_fasta')
    )
    print(f"  Total: {len(matched_seqs):,} sequences")
    
    # Load biotypes
    biotype_lookup = load_biotype_mapping(args.biotype_csv)
    biotypes = extract_biotypes_from_sequences(matched_seqs, biotype_lookup)
    grouped_biotypes = group_rare_biotypes(biotypes, min_count=500)
    
    # Create biotype mapping
    unique_biotypes = sorted(set(grouped_biotypes))
    biotype_to_idx = {bt: idx for idx, bt in enumerate(unique_biotypes)}
    
    print(f"\nBiotypes:")
    print(f"  Unique biotypes (after grouping): {len(unique_biotypes)}")
    from collections import Counter
    biotype_counts = Counter(grouped_biotypes)
    print(f"  Top 5: {biotype_counts.most_common(5)}")
    
    # Create preprocessor
    max_length = config.get('model', 'max_length')
    encoding_type = config.get('model', 'encoding_type', default='one_hot')
    
    if encoding_type == 'cse':
        preprocessor = SequencePreprocessor(
            max_length=max_length,
            encoding_type=encoding_type,
            cse_d_model=config.get('model', 'cse_d_model', default=128),
            cse_kernel_size=config.get('model', 'cse_kernel_size', default=9)
        )
    else:
        preprocessor = SequencePreprocessor(max_length=max_length, encoding_type=encoding_type)
    
    # Create model builder
    model_builder = create_model_builder(config)
    
    # Create CV splits
    print(f"\nCreating CV splits (n_bins={DEFAULT_N_BINS}, random_state={DEFAULT_RANDOM_STATE})...")
    strat_groups = create_length_stratified_groups(matched_seqs, matched_labels, n_bins=DEFAULT_N_BINS)
    skf = get_cv_splitter(
        n_folds=args.n_folds,
        random_state=config.get('training', 'random_state', default=DEFAULT_RANDOM_STATE)
    )
    
    # Evaluate each fold
    print("\n" + "=" * 80)
    print("EVALUATING FOLDS")
    print("=" * 80)
    
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(matched_seqs, strat_groups)):
        print(f"\n{'='*80}")
        print(f"Fold {fold_idx}")
        print(f"{'='*80}")
        
        # Get validation data
        val_seqs = [matched_seqs[i] for i in val_idx]
        val_labels = [matched_labels[i] for i in val_idx]
        val_biotypes = [grouped_biotypes[i] for i in val_idx]
        
        print(f"  Validation samples: {len(val_seqs):,}")
        print(f"  Val indices (first 5): {list(val_idx[:5])}")  # For verification
        
        # Create dataset
        val_dataset = RNASequenceBiotypeDataset(
            val_seqs, val_labels, preprocessor,
            biotype_labels=val_biotypes,
            biotype_to_idx=biotype_to_idx
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=1
        )
        
        # Load model
        checkpoint_path = experiment_dir / 'models' / f'fold_{fold_idx}_best.pt'
        print(f"  Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"    Saved at epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"    Val acc at save: {checkpoint.get('val_acc', 'unknown')}")
        
        model = model_builder()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Evaluate using shared utilities
        print(f"  Evaluating...")
        
        # Deterministic evaluation
        det_results = evaluate_deterministic(model, val_loader, device)
         
        # Print results
        print(f"\n  Results (Deterministic - μ, dropout OFF):")
        print(f"    Accuracy:  {det_results['accuracy']:.4f}")
        print(f"    Precision: {det_results['precision']:.4f}")
        print(f"    Recall:    {det_results['recall']:.4f}")
        print(f"    F1 Score:  {det_results['f1']:.4f}")
        
        # Store results (including sample indices for CSV generation)
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
                # Store these for CSV generation
                'predictions': det_results['predictions'],
                'confidences': det_results['confidences'],
                'probs': det_results['probs'],
                'sample_indices': val_idx
            },
            'n_samples': len(val_seqs)
        })
    
    # Compute overall statistics
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 80)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    summary = {}
    
    for mode in ['deterministic']:
        for metric in metrics:
            values = [r[mode][metric] for r in fold_results]
            summary[f"{mode}_{metric}"] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
    
    # Print summary
    print(f"\nOVERALL STATISTICS:")
    print(f"{'Mode':<30} {'Accuracy':<20} {'F1':<20}")
    print("-" * 80)
    
    for mode, mode_label in [
        ('deterministic', 'Deterministic (μ)'),
    ]:
        acc_mean = summary[f"{mode}_accuracy"]['mean']
        acc_std = summary[f"{mode}_accuracy"]['std']
        f1_mean = summary[f"{mode}_f1"]['mean']
        f1_std = summary[f"{mode}_f1"]['std']
        
        print(f"{mode_label:<30} {acc_mean:.4f} ± {acc_std:.4f}     {f1_mean:.4f} ± {f1_std:.4f}")
    
    # Generate hard case CSVs if requested
    if args.generate_hard_case_csvs:
        print("\n" + "=" * 80)
        print("GENERATING HARD CASE CSVs")
        print("=" * 80)
        
        all_predictions_df = aggregate_fold_predictions(
            fold_results, matched_seqs, matched_labels, grouped_biotypes
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
        print(f"  - biotype, sequence_length, n_folds")
        print(f"  - error_rate, mean_confidence, std_confidence, min_confidence")
        print(f"  - agreement, is_hard_case, fold_predictions")
    
    # Extract embeddings
    if args.extract_all_folds:
        print("\n" + "=" * 80)
        print("EXTRACTING EMBEDDINGS FROM ALL FOLDS")
        print("=" * 80)
        
        embeddings_dict = extract_embeddings_all_folds(
            model_builder, experiment_dir, matched_seqs, matched_labels,
            grouped_biotypes, biotype_to_idx, preprocessor, config,
            device, use_deterministic=True
        )
        
        # Save
        save_path = experiment_dir / 'embeddings_all_folds.npz'
        save_embeddings(embeddings_dict, save_path)
        print(f"\n  Saved all-fold embeddings to: {save_path}")
    else:
        print("\n" + "=" * 80)
        print("EXTRACTING EMBEDDINGS FROM BEST FOLD")
        print("=" * 80)
        
        # Find best fold based on deterministic accuracy
        best_fold_idx = max(range(len(fold_results)), 
                           key=lambda i: fold_results[i]['deterministic']['accuracy'])
        best_fold = fold_results[best_fold_idx]
        
        print(f"\nBest fold: {best_fold_idx} (deterministic accuracy: {best_fold['deterministic']['accuracy']:.4f})")
        
        # Extract embeddings from best fold
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(matched_seqs, strat_groups)):
            if fold_idx != best_fold_idx:
                continue
            
            print(f"Extracting embeddings from fold {fold_idx}...")
            
            # Get validation data
            val_seqs = [matched_seqs[i] for i in val_idx]
            val_labels = [matched_labels[i] for i in val_idx]
            val_biotypes = [grouped_biotypes[i] for i in val_idx]
            
            # Create dataset
            val_dataset = RNASequenceBiotypeDataset(
                val_seqs, val_labels, preprocessor,
                biotype_labels=val_biotypes,
                biotype_to_idx=biotype_to_idx
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=1
            )
            
            # Load model
            checkpoint_path = experiment_dir / 'models' / f'fold_{fold_idx}_best.pt'
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            model = model_builder()
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            
            # Extract embeddings using shared utility
            embeddings_dict = extract_embeddings_from_model(
                model, val_loader, device, use_deterministic=True
            )
            
            # Add fold metadata
            embeddings_dict['fold_idx'] = fold_idx
            embeddings_dict['val_indices'] = val_idx
            
            # Save
            save_path = experiment_dir / 'embeddings_best_fold.npz'
            save_embeddings(embeddings_dict, save_path)
            
            print(f"  Saved embeddings to: {save_path}")
            print(f"  Shape: {embeddings_dict['embeddings'].shape}")
            print(f"  Latent dim: {embeddings_dict['embeddings'].shape[1]}")
            
            break
    
    # Save evaluation results
    output_file = experiment_dir / 'cv_evaluation_results.json'
    
    # Remove large arrays from fold_results before saving
    fold_results_for_json = []
    for r in fold_results:
        r_copy = {
            'fold': r['fold'],
            'n_samples': r['n_samples']
        }
        for mode in ['deterministic']:
            r_copy[mode] = {
                'accuracy': r[mode]['accuracy'],
                'precision': r[mode]['precision'],
                'recall': r[mode]['recall'],
                'f1': r[mode]['f1']
            }
            if mode == 'deterministic':
                r_copy[mode]['confusion_matrix'] = r[mode]['confusion_matrix']
                r_copy[mode]['mean_confidence'] = r[mode]['mean_confidence']
                r_copy[mode]['std_confidence'] = r[mode]['std_confidence']
        fold_results_for_json.append(r_copy)
    
    data_to_save = {
        'summary': summary,
        'fold_results': fold_results_for_json,
        'n_folds': args.n_folds,
        'total_samples': len(matched_seqs),
        'config': {
            'n_bins': DEFAULT_N_BINS,
            'random_state': DEFAULT_RANDOM_STATE
        }
    }
    
    data_serializable = convert_to_serializable(data_to_save)
    
    with open(output_file, 'w') as f:
        json.dump(data_serializable, f, indent=2)
    
    print(f"\n  Results saved to: {output_file}")
    
    # Save per-fold CSV
    csv_rows = []
    for r in fold_results:
        for mode in ['deterministic']:
            csv_rows.append({
                'fold': r['fold'],
                'mode': mode,
                'accuracy': r[mode]['accuracy'],
                'precision': r[mode]['precision'],
                'recall': r[mode]['recall'],
                'f1': r[mode]['f1'],
                'n_samples': r['n_samples']
            })
    
    csv_file = experiment_dir / 'cv_fold_results.csv'
    pd.DataFrame(csv_rows).to_csv(csv_file, index=False)
    print(f"  Per-fold CSV saved to: {csv_file}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nKey Results:")
    print(f"  Deterministic Accuracy: {summary['deterministic_accuracy']['mean']:.4f} ± {summary['deterministic_accuracy']['std']:.4f}")
    print(f"  Deterministic F1:       {summary['deterministic_f1']['mean']:.4f} ± {summary['deterministic_f1']['std']:.4f}")
    print(f"\nNote: Report 'Deterministic' metrics in papers (standard VAE inference with μ)")
    
    if args.generate_hard_case_csvs:
        print(f"\nHard case CSVs saved to: {experiment_dir / 'evaluation_csvs'}/")
        print(f"  Use these with visualize_umap.py or other downstream analysis")

    # =========================================================================
    # GENERATE ROC/PR CURVES
    # =========================================================================
    print("\\n" + "=" * 80)
    print("GENERATING ROC/PR CURVES")
    print("=" * 80)
    
    figures_dir = experiment_dir / 'performance_figures'
    figures_dir.mkdir(exist_ok=True)
    
    roc_pr_metrics = plot_roc_pr_curves(fold_results, matched_labels, figures_dir)  

    if args.test_lnc_fasta and args.test_pc_fasta:
        print("\n" + "=" * 80)
        print("INDEPENDENT TEST SET EVALUATION (ENSEMBLE)")
        print("=" * 80)
        evaluate_test_set(config, experiment_dir, model_builder,
                  args.test_lnc_fasta, args.test_pc_fasta,
                  args.n_folds, args.batch_size, device,
                  preprocessor, biotype_to_idx)


if __name__ == '__main__':
    main()