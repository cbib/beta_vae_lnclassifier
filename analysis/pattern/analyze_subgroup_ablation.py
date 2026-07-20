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
python analysis/pattern/analyze_subgroup_ablation.py \\
    --experiment_dir gencode_v49_experiments/beta_vae_subgroup_base_g49 \\
    --config         configs/beta_vae_subgroup_base_g49.json \\
    --output_dir     gencode_v49_experiments/beta_vae_subgroup_base_g49/ablation_analysis \\
    --model_label    "β-VAE Standard" \\
    --gencode_version v49
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset

from data.feature_registry import (REGISTRY, display_block,
                                    display_subgroup, display_subgroups)

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100


# ---------------------------------------------------------------------------
# Block colour palette (shared convention across interpretability scripts)
# ---------------------------------------------------------------------------

BLOCK_COLORS = {
    "nonb":  "#4A90D9",
    "te":    "#F4B942",
    "nonb2": "#9B7FD4",
}


def _sg_color(sg: str) -> str:
    return BLOCK_COLORS.get(REGISTRY.token_block(sg), "#999999")


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
# Visualisation
# ---------------------------------------------------------------------------

def plot_ablation_results(df: pd.DataFrame, output_path: Path,
                          fig_tag: str = '') -> None:
    """
    Ranked bar chart of accuracy drop per sub-group for both ablation modes.
    Sub-groups sorted by mean drop across modes.
    """
    mean_drop = (df.groupby('subgroup')['acc_drop']
                 .mean()
                 .sort_values(ascending=False))
    subgroups_ordered = mean_drop.index.tolist()

    token_df   = df[df['mode'] == 'token_zero'].set_index('subgroup')
    feature_df = df[df['mode'] == 'feature_zero'].set_index('subgroup')

    x     = np.arange(len(subgroups_ordered))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))

    token_drops   = [token_df.loc[sg, 'acc_drop']   if sg in token_df.index   else 0
                     for sg in subgroups_ordered]
    feature_drops = [feature_df.loc[sg, 'acc_drop'] if sg in feature_df.index else 0
                     for sg in subgroups_ordered]

    ax.bar(x - width/2, token_drops,   width, label='Token zero',
           color='#4A90D9', edgecolor='black', linewidth=0.6, alpha=0.9)
    ax.bar(x + width/2, feature_drops, width, label='Feature zero',
           color='#F4B942', edgecolor='black', linewidth=0.6, alpha=0.9)

    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(display_subgroups(subgroups_ordered), rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Accuracy drop (baseline − ablated)', fontsize=12)
    ax.set_title(
        f'{fig_tag}Sub-group Ablation — Classification Accuracy Drop\n'
        '(larger = sub-group more important for classification)',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Block boundary annotations — registry-driven, supports any block count
    block_of = {sg: REGISTRY.token_block(sg) for sg in subgroups_ordered}
    blocks_in_order = list(dict.fromkeys(block_of[sg] for sg in subgroups_ordered))
    # Note: subgroups_ordered is sorted by drop magnitude, not block, so
    # boundary lines/labels here annotate composition rather than draw a
    # clean partition. We instead annotate each block's mean x-position.
    ymax = max(max(token_drops, default=0.01), max(feature_drops, default=0.01))
    for block_name in blocks_in_order:
        xs = [i for i, sg in enumerate(subgroups_ordered) if block_of[sg] == block_name]
        if xs:
            ax.text(np.mean(xs), ymax * 1.02, f'← {block_name.upper()} →',
                    ha='center', fontsize=9, color=BLOCK_COLORS.get(block_name, '#333'),
                    style='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cross_fold_summary(summary_df: pd.DataFrame, output_path: Path,
                            fig_tag: str = '') -> None:
    """Cross-fold mean ± std accuracy drop per sub-group."""
    for mode in ['token_zero', 'feature_zero']:
        mode_df = summary_df[summary_df['mode'] == mode]
        if mode_df.empty:
            continue

        pivot = mode_df.pivot(index='subgroup', columns='fold', values='acc_drop')
        mean  = pivot.mean(axis=1).sort_values(ascending=False)
        std   = pivot.std(axis=1).reindex(mean.index)

        fig, ax = plt.subplots(figsize=(13, 5))
        x = np.arange(len(mean))

        mode_color = '#4A90D9' if mode == 'token_zero' else '#F4B942'
        ax.bar(x, mean.values, color=mode_color, edgecolor='black',
               linewidth=0.6, alpha=0.9)
        ax.errorbar(x, mean.values, yerr=std.values,
                    fmt='none', color='black', capsize=3, linewidth=1.2)

        ax.axhline(0, color='black', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(display_subgroups(list(mean.index)), rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Mean accuracy drop (mean ± std over folds)', fontsize=11)
        ax.set_title(
            f'{fig_tag}Cross-Fold Ablation — {mode.replace("_", " ").title()}\n'
            'Sub-groups ranked by mean importance',
            fontsize=13, fontweight='bold'
        )
        ax.grid(True, axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        out = output_path.parent / f'cross_fold_ablation_{mode}.png'
        plt.savefig(out, dpi=350, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out}")


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
    parser.add_argument('--model_label',     default='β-VAE Standard')
    parser.add_argument('--gencode_version', default='v49')
    parser.add_argument('--batch_size',     type=int, default=256)
    parser.add_argument('--device',
                        default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    exp_dir    = Path(args.experiment_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag_parts = [p for p in [args.model_label,
                 f'GENCODE {args.gencode_version}'] if p]
    fig_tag   = ' | '.join(tag_parts) + ' — ' if tag_parts else ''

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

    strat_groups = create_length_stratified_groups(
        all_sequences, labels,
        n_bins=config.get('training', 'n_bins', default=5)
    )
    skf = StratifiedKFold(
        n_splits   = config.get('training', 'n_folds'),
        shuffle    = True,
        random_state=config.get('training', 'random_state', default=42)
    )
    splits = list(skf.split(all_sequences, strat_groups))

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

        plot_ablation_results(
            fold_df,
            fold_out / 'ablation_bar.png',
            fig_tag=f'{fig_tag}fold_{fold_idx} — '
        )

        all_fold_results.append(fold_df)
        print(f"\n  Fold {fold_idx} complete → {fold_out}/")

    if len(all_fold_results) > 1:
        print(f"\n{'='*60}")
        print("CROSS-FOLD SUMMARY")
        print(f"{'='*60}")

        combined = pd.concat(all_fold_results, ignore_index=True)
        combined.to_csv(output_dir / 'all_folds_ablation.csv', index=False)

        plot_cross_fold_summary(
            combined,
            output_dir / 'cross_fold_ablation.png',
            fig_tag=fig_tag
        )

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