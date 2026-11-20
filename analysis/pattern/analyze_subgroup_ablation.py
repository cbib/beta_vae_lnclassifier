#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_subgroup_ablation.py

Sub-group ablation analysis for BetaVAESubgroup.

Two modes
---------
token_zero   : zeros the projected token vector after the projector MLP
               (removes the token's contribution to attention entirely)
feature_zero : zeros the raw input features before the projector
               (removes the input signal, but the projector still runs
               on zeros — isolates whether the projector's bias term or
               learned transformation contributes residual signal)

Usage
-----
python analysis/pattern/analyze_subgroup_ablation.py \
    --experiment_dir gencode_v49_experiments/beta_vae_subgroup_base_g49 \
    --config         configs/beta_vae_subgroup_base_g49.json \
    --output_dir     gencode_v49_experiments/beta_vae_subgroup_base_g49/ablation_analysis
"""
from data.cv_utils import get_cv_splits_from_config

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset

from data.feature_registry import REGISTRY


def _forward_kwargs(batch: dict, device: torch.device) -> dict:
    kwargs = dict(
        te_genomic     = batch['te_genomic'].to(device),
        te_processed   = batch['te_processed'].to(device),
        nonb_genomic   = batch['nonb_genomic'].to(device),
        nonb_processed = batch['nonb_processed'].to(device),
    )
    if 'nonb2' in batch:
        kwargs['nonb2'] = batch['nonb2'].to(device)
    return kwargs


# ---------------------------------------------------------------------------
# Ablation hook
# ---------------------------------------------------------------------------

class SubgroupAblationHook:
    """
    Registers forward hooks on the relevant block's SubgroupProjectionLayer
    to zero out specific sub-group tokens before they enter cross-modal
    attention.

    token_zero   : zeros the projected token vector after the projector MLP
    feature_zero : zeros the raw input features before the projector

    The block and token position are resolved dynamically via the registry,
    so this works for any block defined in BLOCK_DEFS (nonb, te, nonb2, ...).
    """

    def __init__(self, model, subgroup: str, mode: str = 'token_zero',
                 registry = REGISTRY):
        self.model     = model
        self.subgroup  = subgroup
        self.mode      = mode
        self._handles  = []
        self._registry = registry

        block_name = registry.token_block(subgroup)
        if block_name not in model.projectors:
            raise ValueError(
                f"Block '{block_name}' (for subgroup '{subgroup}') not found "
                f"in model.projectors. Available: {list(model.projectors.keys())}"
            )
        self.proj_layer = model.projectors[block_name]
        self.block_name = block_name

        block_subgroups = registry.block_subgroups(block_name)
        if subgroup not in block_subgroups:
            raise ValueError(
                f"Sub-group '{subgroup}' not in block '{block_name}' "
                f"subgroups: {block_subgroups}"
            )
        self.token_idx = block_subgroups.index(subgroup)

        # Feature indices within the block's raw feature vector — used for
        # feature_zero mode
        self.feature_indices = registry.indices_for_subgroup(subgroup)

    def __enter__(self):
        if self.mode == 'token_zero':
            self._register_token_hook()
        elif self.mode == 'feature_zero':
            self._register_feature_hook()
        return self

    def __exit__(self, *args):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _register_token_hook(self):
        token_idx = self.token_idx

        def hook(module, input, output):
            # output is (B, n_subgroups_in_block, d_proj)
            output = output.clone()
            output[:, token_idx, :] = 0.0
            return output

        handle = self.proj_layer.register_forward_hook(hook)
        self._handles.append(handle)

    def _register_feature_hook(self):
        idx = self.feature_indices
        source = self._registry.block_source(self.block_name)

        def hook(module, inputs):
            # inputs: (x_genomic, x_processed) as received by
            # SubgroupProjectionLayer.forward(). For source="processed" or
            # "genomic" blocks one of these may be None.
            x_g, x_p = inputs[0], inputs[1]

            if source in ("both", "genomic") and x_g is not None:
                x_g = x_g.clone()
                x_g[:, idx] = 0.0
            if source in ("both", "processed") and x_p is not None:
                x_p = x_p.clone()
                x_p[:, idx] = 0.0

            return x_g, x_p

        handle = self.proj_layer.register_forward_pre_hook(hook)
        self._handles.append(handle)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, loader, device, ablation_hook=None):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    if ablation_hook is not None:
        ablation_hook.__enter__()

    try:
        with torch.no_grad():
            for batch in loader:
                seq    = batch['sequence'].to(device)
                labels = batch['label']
                fwd_kw = _forward_kwargs(batch, device)

                out = model(seq, deterministic=True, **fwd_kw)

                probs = torch.softmax(out['logits'], dim=1)
                preds = out['logits'].argmax(1)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.numpy())
                all_probs.append(probs.cpu().numpy())
    finally:
        if ablation_hook is not None:
            ablation_hook.__exit__(None, None, None)

    return (np.concatenate(all_preds),
            np.concatenate(all_labels),
            np.vstack(all_probs))

def compute_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average='binary', zero_division=0)
    return {'accuracy': float(acc), 'f1': float(f1)}


# ---------------------------------------------------------------------------
# Per-fold ablation
# ---------------------------------------------------------------------------

def ablate_fold(model, loader, device, token_names: List[str]):
    """
    Run full ablation study for one fold model.
    Returns DataFrame with results for all sub-groups × modes.
    """
    base_preds, base_labels, _ = run_inference(model, loader, device)
    base = compute_metrics(base_preds, base_labels)
    print(f"    Baseline: acc={base['accuracy']:.4f} f1={base['f1']:.4f}")

    rows = []
    for sg in token_names:
        block = REGISTRY.token_block(sg)

        for mode in ['token_zero', 'feature_zero']:
            hook  = SubgroupAblationHook(model, sg, mode=mode)
            preds, labels, _ = run_inference(model, loader, device,
                                             ablation_hook=hook)
            metrics = compute_metrics(preds, labels)

            acc_drop = base['accuracy'] - metrics['accuracy']
            f1_drop  = base['f1']       - metrics['f1']

            rows.append({
                'subgroup':       sg,
                'block':          block,
                'mode':           mode,
                'baseline_acc':   base['accuracy'],
                'baseline_f1':    base['f1'],
                'ablated_acc':    metrics['accuracy'],
                'ablated_f1':     metrics['f1'],
                'acc_drop':       float(acc_drop),
                'f1_drop':        float(f1_drop),
            })
            print(f"    {sg:15s} [{block:5s}] [{mode:13s}]: "
                  f"acc={metrics['accuracy']:.4f} "
                  f"(Δ={acc_drop:+.4f}) "
                  f"f1={metrics['f1']:.4f} "
                  f"(Δ={f1_drop:+.4f})")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Sub-group ablation analysis for BetaVAESubgroup',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--experiment_dir', required=True,
                        help='Experiment directory containing models/ subfolder')
    parser.add_argument('--config',         required=True,
                        help='Path to model config JSON')
    parser.add_argument('--output_dir',     default='ablation_analysis')
    parser.add_argument('--batch_size',     type=int, default=256)
    parser.add_argument('--device',
                        default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    exp_dir    = Path(args.experiment_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    from configs.load_config import load_config
    from models.model_builder import create_model_builder
    from data.cv_utils import load_sequences_in_order, create_length_stratified_groups
    from data.gated_feature_dataset import SequenceFeatureDataset
    from sklearn.model_selection import StratifiedKFold

    config = load_config(args.config)

    print("=" * 70)
    print("SUB-GROUP ABLATION ANALYSIS")
    print("=" * 70)
    print(f"Registry blocks: {REGISTRY.block_names} "
          f"({REGISTRY.total_tokens} total tokens)")

    print("\nLoading dataset...")
    nonb2_csv    = config.get('data', 'nonb2_csv',              default=None)
    nonb2_scaler = config.get('data', 'nonb2_scaler_bank_path', default=None)
    if nonb2_csv is not None and nonb2_scaler is None:
        raise ValueError(
            "Config has data.nonb2_csv set but no data.nonb2_scaler_bank_path."
        )

    dataset = SequenceFeatureDataset(
        lnc_fasta              = config.get('data', 'lnc_fasta'),
        pc_fasta                = config.get('data', 'pc_fasta'),
        te_genomic_csv          = config.get('data', 'te_genomic_csv'),
        te_processed_csv        = config.get('data', 'te_processed_csv',   default=None),
        nonb_genomic_csv        = config.get('data', 'nonb_genomic_csv'),
        nonb_processed_csv      = config.get('data', 'nonb_processed_csv', default=None),
        nonb2_csv                = nonb2_csv,
        te_scaler_bank_path     = config.get('data', 'te_scaler_bank_path'),
        nonb_scaler_bank_path   = config.get('data', 'nonb_scaler_bank_path'),
        nonb2_scaler_bank_path  = nonb2_scaler,
        max_length                = config.get('model', 'max_length'),
    )

    all_sequences, labels = load_sequences_in_order(
        lnc_fasta=config.get('data', 'lnc_fasta'),
        pc_fasta= config.get('data', 'pc_fasta'),
    )

    splits = get_cv_splits_from_config(config, all_sequences, labels)

    model_builder = create_model_builder(config)
    token_names   = REGISTRY.all_subgroups

    model_dir = exp_dir / 'models'
    fold_files = sorted(model_dir.glob('fold_*_best.pt'))
    if not fold_files:
        print(f"ERROR: No fold checkpoints found in {model_dir}")
        return

    print(f"Found {len(fold_files)} fold checkpoint(s)")

    all_fold_results = []

    for ckpt_path in fold_files:
        fold_idx = int(ckpt_path.stem.split('_')[1])
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx}")
        print(f"{'='*60}")

        ckpt  = torch.load(ckpt_path, map_location=device)
        model = model_builder()
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"  Loaded epoch={ckpt['epoch']} val_acc={ckpt['val_acc']:.4f}")

        if model.n_tokens != len(token_names):
            print(f"    WARNING: model.n_tokens ({model.n_tokens}) ≠ "
                  f"registry token count ({len(token_names)}) — "
                  f"checkpoint may predate current feature blocks")

        _, val_idx  = splits[fold_idx]
        val_loader  = DataLoader(
            Subset(dataset, val_idx),
            batch_size=args.batch_size, shuffle=False, num_workers=1
        )
        print(f"  Val samples: {len(val_idx):,}")

        fold_out = output_dir / f'fold_{fold_idx}'
        fold_out.mkdir(exist_ok=True)
        results_csv = fold_out / 'ablation_results.csv'

        if results_csv.exists():
            print(f"   Already completed — loading from {results_csv}")
            fold_df = pd.read_csv(results_csv)
            all_fold_results.append(fold_df)
            print(f"\n  Fold {fold_idx} skipped (cached) → {fold_out}/")
            continue

        fold_df = ablate_fold(model, val_loader, device, token_names)
        fold_df['fold'] = fold_idx

        fold_df.to_csv(results_csv, index=False)

        all_fold_results.append(fold_df)
        print(f"\n  Fold {fold_idx} complete → {fold_out}/")

    if len(all_fold_results) > 1:
        print(f"\n{'='*60}")
        print("CROSS-FOLD SUMMARY")
        print(f"{'='*60}")

        combined = pd.concat(all_fold_results, ignore_index=True)
        combined.to_csv(output_dir / 'all_folds_ablation.csv', index=False)

        print("\nSub-group importance ranking (token_zero, mean acc drop):")
        token_summary = (combined[combined['mode'] == 'token_zero']
                         .groupby('subgroup')['acc_drop']
                         .agg(['mean', 'std'])
                         .sort_values('mean', ascending=False))
        print(f"\n{'Rank':<5} {'Sub-group':<15} {'Mean Δacc':>10} {'Std':>8}")
        print("-" * 42)
        for rank, (sg, row) in enumerate(token_summary.iterrows(), 1):
            bar = '█' * max(0, int(row['mean'] * 1000))
            print(f"{rank:<5} {sg:<15} {row['mean']:>+10.4f}  "
                  f"±{row['std']:.4f}  {bar}")
    else:
        combined = all_fold_results[0]
        combined.to_csv(output_dir / 'all_folds_ablation.csv', index=False)

    print(f"\nAll outputs saved to: {output_dir}/")


if __name__ == '__main__':
    main()