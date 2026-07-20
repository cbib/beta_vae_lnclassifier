#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subgroup attention analysis for BetaVAEGated.

Analyses the 20-token multi-head cross-modal attention weights saved by
BetaVAEGatedTrainer.extract_attention_all_folds().

Two primary views
-----------------
1. Token importance
   Mean attention weight per sub-group token, stratified by:
     - lncRNA vs mRNA
     - hard vs easy cases
   Produces bar charts and a heatmap showing which repeat sub-groups the
   model relies on most, and whether that reliance differs by class or
   classification difficulty.

2. Head specialisation
   Per-head mean attention weight over all 20 tokens, as a (head × token)
   heatmap.  Reveals whether different heads have learned to focus on
   different sub-groups (e.g. NonB motif types vs TE classes).

Processing
----------
Each fold is processed independently (per-fold outputs), then a cross-fold
summary aggregates mean ± std across folds.

Usage
-----
python analyze_subgroup_attention.py \\
    --attn_dir  gencode_v49_experiments/beta_vae_gated_g49/fold_attention \\
    --output_dir gencode_v49_experiments/beta_vae_gated_g49/attention_analysis \\
    --model_label "β-VAE Gated" \\
    --gencode_version v49

Input .npz keys used
--------------------
    attn_weights       (N, L_encoded, 20)   head-averaged
    attn_weights_full  (N, num_heads, L_encoded, 20)
    nonb_subgroups     (9,)  string array
    te_subgroups       (6,)  string array
    labels             (N,)  0=lnc, 1=mRNA
    is_hard_case       (N,)  bool
    predictions        (N,)
    confidences        (N,)
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100


# ---------------------------------------------------------------------------
# Colour palette — consistent across all figures
# ---------------------------------------------------------------------------

# Per-class colours
CLASS_COLORS  = {'lncRNA': '#FF6B6B', 'mRNA': '#4ECDC4'}
# Hard/easy colours
HARDNESS_COLORS = {'hard': '#E74C3C', 'easy': '#2ECC71'}
# NonB subgroups get blues, TE subgroups get oranges
NONB_CMAP = plt.cm.Blues
TE_CMAP   = plt.cm.Oranges


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_fold(npz_path: Path) -> Dict:
    """Load a single fold .npz and return a dict of arrays."""
    data = np.load(npz_path, allow_pickle=True)

    # Head-averaged attention: (N, L, 20)
    attn_mean = data['attn_weights']

    # Full multi-head attention: (N, H, L, 20) — may be absent in older files
    attn_full = data['attn_weights_full'] if 'attn_weights_full' in data else None

    # Sub-group names
    nonb_sgs = list(data['nonb_subgroups'])
    te_sgs   = list(data['te_subgroups'])
    token_names = nonb_sgs + te_sgs   # 20 names in order

    labels       = data['labels'].astype(int)
    is_hard      = data['is_hard_case'].astype(bool)
    predictions  = data['predictions'].astype(int)
    confidences  = data['confidences'].astype(float)

    return {
        'attn_mean':    attn_mean,     # (N, L, 20)
        'attn_full':    attn_full,     # (N, H, L, 20) or None
        'token_names':  token_names,   # list[str] len=20
        'nonb_sgs':     nonb_sgs,
        'te_sgs':       te_sgs,
        'labels':       labels,
        'is_hard':      is_hard,
        'predictions':  predictions,
        'confidences':  confidences,
        'n_samples':    len(labels),
    }


def token_importance(attn_mean: np.ndarray) -> np.ndarray:
    """
    Compute per-token importance as mean attention weight averaged over
    sequence positions and samples.

    Parameters
    ----------
    attn_mean : (N, L, T)

    Returns
    -------
    importance : (T,)  values sum to ~1 (softmax was applied over T)
    """
    # Average over L (positions) → (N, T), then over N → (T,)
    return attn_mean.mean(axis=(0, 1))


def token_importance_by_group(
    attn_mean:  np.ndarray,
    labels:     np.ndarray,
    is_hard:    np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute token importance for four strata:
      lnc_easy, lnc_hard, mrna_easy, mrna_hard

    Returns dict: stratum_name → (T,) importance vector
    """
    masks = {
        'lnc_easy':  (labels == 0) & ~is_hard,
        'lnc_hard':  (labels == 0) &  is_hard,
        'mrna_easy': (labels == 1) & ~is_hard,
        'mrna_hard': (labels == 1) &  is_hard,
    }
    result = {}
    for name, mask in masks.items():
        if mask.sum() == 0:
            result[name] = np.zeros(attn_mean.shape[-1])
        else:
            result[name] = attn_mean[mask].mean(axis=(0, 1))
    return result


def head_specialisation(attn_full: np.ndarray) -> np.ndarray:
    """
    Compute mean attention weight per (head, token) averaged over
    positions and samples.

    Parameters
    ----------
    attn_full : (N, H, L, T)

    Returns
    -------
    heatmap : (H, T)
    """
    return attn_full.mean(axis=(0, 2))   # average over N and L → (H, T)


# ---------------------------------------------------------------------------
# Per-fold visualisations
# ---------------------------------------------------------------------------

def plot_token_importance(
    importance_by_stratum: Dict[str, np.ndarray],
    token_names: List[str],
    n_nonb: int,
    output_path: Path,
    fig_tag: str = '',
) -> None:
    """
    Bar chart of token importance for each stratum, grouped by token.
    Tokens are coloured by block (NonB = blue, TE = orange).
    """
    T = len(token_names)
    strata = list(importance_by_stratum.keys())

    # Colour per token: NonB = blue shades, TE = orange shades
    token_colors = (
        [NONB_CMAP(0.5 + 0.4 * i / max(n_nonb - 1, 1)) for i in range(n_nonb)] +
        [TE_CMAP(0.5 + 0.4 * i / max(T - n_nonb - 1, 1))
         for i in range(T - n_nonb)]
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
    axes = axes.flatten()

    stratum_labels = {
        'lnc_easy':  'lncRNA — Easy cases',
        'lnc_hard':  'lncRNA — Hard cases',
        'mrna_easy': 'mRNA — Easy cases',
        'mrna_hard': 'mRNA — Hard cases',
    }

    x = np.arange(T)

    for ax, stratum in zip(axes, strata):
        imp = importance_by_stratum[stratum]
        bars = ax.bar(x, imp, color=token_colors, edgecolor='black',
                      linewidth=0.5, width=0.7)

        # Vertical separator between NonB and TE blocks
        ax.axvline(x=n_nonb - 0.5, color='gray', linestyle='--',
                   linewidth=1.5, alpha=0.7, label='NonB | TE boundary')

        ax.set_xticks(x)
        ax.set_xticklabels(token_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Mean attention weight', fontsize=11)
        ax.set_title(stratum_labels[stratum], fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Annotate max token
        max_idx = imp.argmax()
        ax.annotate(
            f'max: {token_names[max_idx]}\n({imp[max_idx]:.3f})',
            xy=(max_idx, imp[max_idx]),
            xytext=(max_idx + 0.5, imp[max_idx] + imp.max() * 0.05),
            fontsize=8, color='black',
            arrowprops=dict(arrowstyle='->', color='black', lw=1),
        )

    # Block legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=NONB_CMAP(0.65), edgecolor='black', label='NonB DNA subgroups'),
        Patch(facecolor=TE_CMAP(0.65),   edgecolor='black', label='TE subgroups'),
    ]
    fig.legend(handles=legend_elements, loc='upper right',
               bbox_to_anchor=(0.99, 0.99), fontsize=10, framealpha=0.9)

    fig.suptitle(f'{fig_tag}Sub-group Token Importance by Class and Difficulty',
                 fontsize=20, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_token_importance_heatmap(
    importance_by_stratum: Dict[str, np.ndarray],
    token_names: List[str],
    n_nonb: int,
    output_path: Path,
    fig_tag: str = '',
) -> None:
    """
    Heatmap of token importance across strata — compact summary view.
    Rows = strata, columns = tokens.
    """
    strata_order = ['lnc_easy', 'lnc_hard', 'mrna_easy', 'mrna_hard']
    row_labels   = ['lncRNA easy', 'lncRNA hard', 'mRNA easy', 'mRNA hard']

    matrix = np.stack([importance_by_stratum[s] for s in strata_order], axis=0)
    # (4, T)

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0)

    ax.set_xticks(range(len(token_names)))
    ax.set_xticklabels(token_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(strata_order)))
    ax.set_yticklabels(row_labels, fontsize=11)

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                    fontsize=8, color='white' if v > matrix.max() * 0.6 else 'black')

    # Block separator
    ax.axvline(x=n_nonb - 0.5, color='white', linewidth=2.5)

    # Block labels above
    ax.annotate('← NonB DNA subgroups →',
                xy=((n_nonb - 1) / 2, -0.8), xycoords=('data', 'axes fraction'),
                ha='center', fontsize=10, fontweight='bold', color=NONB_CMAP(0.8))
    ax.annotate('← TE subgroups →',
                xy=(n_nonb + (len(token_names) - n_nonb - 1) / 2, -0.8),
                xycoords=('data', 'axes fraction'),
                ha='center', fontsize=10, fontweight='bold', color=TE_CMAP(0.8))

    plt.colorbar(im, ax=ax, label='Mean attention weight', shrink=0.8)
    ax.set_title(f'{fig_tag}Token Importance Heatmap — lncRNA vs mRNA, Easy vs Hard',
                 fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_head_specialisation(
    heatmap: np.ndarray,
    token_names: List[str],
    n_nonb: int,
    output_path: Path,
    fig_tag: str = '',
) -> None:
    """
    Heatmap of mean attention weight per (head, token).
    Each row is one attention head; columns are the 20 sub-group tokens.
    """
    H, T = heatmap.shape

    fig, ax = plt.subplots(figsize=(14, 1.2 * H + 1.5))
    im = ax.imshow(heatmap, cmap='viridis', aspect='auto', vmin=0)

    ax.set_xticks(range(T))
    ax.set_xticklabels(token_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(H))
    ax.set_yticklabels([f'Head {h+1}' for h in range(H)], fontsize=11)

    # Annotate cells
    for h in range(H):
        for t in range(T):
            v = heatmap[h, t]
            ax.text(t, h, f'{v:.3f}', ha='center', va='center',
                    fontsize=8,
                    color='white' if v > heatmap.max() * 0.6 else 'black')

    # Block separator
    ax.axvline(x=n_nonb - 0.5, color='white', linewidth=2.5)

    plt.colorbar(im, ax=ax, label='Mean attention weight', shrink=0.7)
    ax.set_title(f'{fig_tag}Head Specialisation — Mean Attention per Head × Sub-group Token',
                 fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_lnc_vs_mrna_delta(
    importance_by_stratum: Dict[str, np.ndarray],
    token_names: List[str],
    n_nonb: int,
    output_path: Path,
    fig_tag: str = '',
) -> None:
    """
    Diverging bar chart: (mRNA importance - lncRNA importance) per token,
    for easy and hard cases separately.
    Positive = model attends more for mRNA, negative = more for lncRNA.
    """
    lnc_easy  = importance_by_stratum['lnc_easy']
    lnc_hard  = importance_by_stratum['lnc_hard']
    mrna_easy = importance_by_stratum['mrna_easy']
    mrna_hard = importance_by_stratum['mrna_hard']

    delta_easy = mrna_easy - lnc_easy
    delta_hard = mrna_hard - lnc_hard

    T = len(token_names)
    x = np.arange(T)
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar(x - width/2, delta_easy, width, label='Easy cases',
           color=[CLASS_COLORS['mRNA'] if d >= 0 else CLASS_COLORS['lncRNA']
                  for d in delta_easy],
           edgecolor='black', linewidth=0.5, alpha=0.85)
    ax.bar(x + width/2, delta_hard, width, label='Hard cases',
           color=[CLASS_COLORS['mRNA'] if d >= 0 else CLASS_COLORS['lncRNA']
                  for d in delta_hard],
           edgecolor='black', linewidth=0.5, alpha=0.55)

    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(x=n_nonb - 0.5, color='gray', linestyle='--', linewidth=1.5,
               alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(token_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Δ attention (mRNA − lncRNA)', fontsize=12)
    ax.set_title(
        f'{fig_tag}Differential Token Attention: mRNA vs lncRNA\n'
        '(positive = mRNA attends more; negative = lncRNA attends more)',
        fontsize=13, fontweight='bold'
    )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CLASS_COLORS['mRNA'],   alpha=0.85, label='mRNA > lncRNA'),
        Patch(facecolor=CLASS_COLORS['lncRNA'], alpha=0.85, label='lncRNA > mRNA'),
        plt.Rectangle((0,0), 1, 1, fc='gray', alpha=0.85, label='Easy cases'),
        plt.Rectangle((0,0), 1, 1, fc='gray', alpha=0.45, label='Hard cases'),
    ]
    ax.legend(handles=legend_elements, fontsize=10, framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Cross-fold summary
# ---------------------------------------------------------------------------

def create_cross_fold_summary(
    fold_importances: List[Dict[str, np.ndarray]],
    fold_head_maps:   List[np.ndarray],
    token_names:      List[str],
    n_nonb:           int,
    output_dir:       Path,
    fig_tag:          str = '',
) -> None:
    """
    Aggregate per-fold token importance and head specialisation across folds.
    Produces mean ± std ribbon plots for token importance and a mean head
    specialisation heatmap.
    """
    print("\n" + "=" * 80)
    print("CROSS-FOLD SUMMARY")
    print("=" * 80)

    strata = ['lnc_easy', 'lnc_hard', 'mrna_easy', 'mrna_hard']
    T      = len(token_names)

    # Stack per-stratum importances across folds: (n_folds, T) per stratum
    stacked = {
        s: np.stack([fi[s] for fi in fold_importances], axis=0)
        for s in strata
    }
    mean_imp = {s: stacked[s].mean(axis=0) for s in strata}
    std_imp  = {s: stacked[s].std(axis=0)  for s in strata}

    # ── Mean token importance ribbon plot ────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
    axes = axes.flatten()

    stratum_titles = {
        'lnc_easy':  'lncRNA — Easy',
        'lnc_hard':  'lncRNA — Hard',
        'mrna_easy': 'mRNA — Easy',
        'mrna_hard': 'mRNA — Hard',
    }

    x = np.arange(T)
    token_colors = (
        [NONB_CMAP(0.5 + 0.4 * i / max(n_nonb - 1, 1)) for i in range(n_nonb)] +
        [TE_CMAP(0.5 + 0.4 * i / max(T - n_nonb - 1, 1))
         for i in range(T - n_nonb)]
    )

    for ax, stratum in zip(axes, strata):
        m = mean_imp[stratum]
        s = std_imp[stratum]

        ax.bar(x, m, color=token_colors, edgecolor='black',
               linewidth=0.5, width=0.7, alpha=0.85)
        ax.errorbar(x, m, yerr=s, fmt='none', color='black',
                    capsize=3, linewidth=1.2)

        ax.axvline(x=n_nonb - 0.5, color='gray', linestyle='--',
                   linewidth=1.5, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(token_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Mean attention weight', fontsize=11)
        ax.set_title(stratum_titles[stratum], fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(
        f'{fig_tag}Cross-Fold Token Importance (mean ± std over folds)',
        fontsize=20, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    out = output_dir / 'cross_fold_token_importance.png'
    plt.savefig(out, dpi=350, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")

    # ── Cross-fold head specialisation ───────────────────────────────────────
    if fold_head_maps and fold_head_maps[0] is not None:
        mean_head = np.stack(fold_head_maps, axis=0).mean(axis=0)   # (H, T)
        H = mean_head.shape[0]

        fig, ax = plt.subplots(figsize=(14, 1.2 * H + 1.5))
        im = ax.imshow(mean_head, cmap='viridis', aspect='auto', vmin=0)

        ax.set_xticks(range(T))
        ax.set_xticklabels(token_names, rotation=45, ha='right', fontsize=10)
        ax.set_yticks(range(H))
        ax.set_yticklabels([f'Head {h+1}' for h in range(H)], fontsize=11)

        for h in range(H):
            for t in range(T):
                v = mean_head[h, t]
                ax.text(t, h, f'{v:.3f}', ha='center', va='center',
                        fontsize=8,
                        color='white' if v > mean_head.max() * 0.6 else 'black')

        ax.axvline(x=n_nonb - 0.5, color='white', linewidth=2.5)
        plt.colorbar(im, ax=ax, label='Mean attention weight', shrink=0.7)
        ax.set_title(
            f'{fig_tag}Cross-Fold Head Specialisation (mean over folds)',
            fontsize=13, fontweight='bold', pad=20
        )
        plt.tight_layout()
        out = output_dir / 'cross_fold_head_specialisation.png'
        plt.savefig(out, dpi=350, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out}")

    # ── Save summary CSV ─────────────────────────────────────────────────────
    rows = []
    for stratum in strata:
        for t, name in enumerate(token_names):
            rows.append({
                'stratum':    stratum,
                'token':      name,
                'block':      'nonb' if t < n_nonb else 'te',
                'mean':       float(mean_imp[stratum][t]),
                'std':        float(std_imp[stratum][t]),
            })
    df = pd.DataFrame(rows)
    csv_out = output_dir / 'cross_fold_token_importance.csv'
    df.to_csv(csv_out, index=False)
    print(f"  Saved: {csv_out}")

    # Print top tokens per stratum
    print("\nTop 3 tokens per stratum (cross-fold mean):")
    for stratum in strata:
        sub = df[df['stratum'] == stratum].nlargest(3, 'mean')
        tops = ', '.join(
            f"{r['token']} ({r['mean']:.3f}±{r['std']:.3f})"
            for _, r in sub.iterrows()
        )
        print(f"  {stratum:12s}: {tops}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Subgroup attention analysis for BetaVAEGated',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--attn_dir',   required=True,
                        help='Directory containing fold_N_attn.npz files')
    parser.add_argument('--output_dir', default='attention_analysis')
    parser.add_argument('--model_label',    default='β-VAE Gated')
    parser.add_argument('--gencode_version',default='v49')
    args = parser.parse_args()

    attn_dir   = Path(args.attn_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag_parts = [p for p in [args.model_label,
                 f'GENCODE {args.gencode_version}'
                 if args.gencode_version else ''] if p]
    fig_tag   = ' | '.join(tag_parts) + ' — ' if tag_parts else ''

    # Find fold files
    fold_files = sorted(attn_dir.glob('fold_*_attn.npz'))
    if not fold_files:
        print(f"ERROR: No fold_*_attn.npz files found in {attn_dir}")
        return

    print("=" * 80)
    print("SUBGROUP ATTENTION ANALYSIS")
    print("=" * 80)
    print(f"Found {len(fold_files)} fold(s): {[f.name for f in fold_files]}")

    fold_importances: List[Dict[str, np.ndarray]] = []
    fold_head_maps:   List[np.ndarray]            = []

    for fold_file in fold_files:
        fold_name = fold_file.stem.replace('_attn', '')
        print(f"\n{'=' * 80}")
        print(f"PROCESSING {fold_name.upper()}")
        print(f"{'=' * 80}")

        fold_out = output_dir / fold_name
        fold_out.mkdir(exist_ok=True)

        data = load_fold(fold_file)

        print(f"  Samples: {data['n_samples']:,} "
              f"(lnc={( data['labels']==0).sum():,}, "
              f"mrna={(data['labels']==1).sum():,})")
        print(f"  Hard cases: {data['is_hard'].sum():,} "
              f"({100*data['is_hard'].mean():.1f}%)")
        print(f"  Tokens: {data['token_names']}")
        print(f"  attn_mean shape: {data['attn_mean'].shape}")
        if data['attn_full'] is not None:
            print(f"  attn_full shape: {data['attn_full'].shape}")

        token_names = data['token_names']
        n_nonb      = len(data['nonb_sgs'])

        # ── Token importance by stratum ───────────────────────────────────────
        imp_by_stratum = token_importance_by_group(
            data['attn_mean'], data['labels'], data['is_hard']
        )

        plot_token_importance(
            imp_by_stratum, token_names, n_nonb,
            output_path = fold_out / 'token_importance_bars.png',
            fig_tag     = f'{fig_tag}{fold_name} — ',
        )
        plot_token_importance_heatmap(
            imp_by_stratum, token_names, n_nonb,
            output_path = fold_out / 'token_importance_heatmap.png',
            fig_tag     = f'{fig_tag}{fold_name} — ',
        )
        plot_lnc_vs_mrna_delta(
            imp_by_stratum, token_names, n_nonb,
            output_path = fold_out / 'token_importance_delta.png',
            fig_tag     = f'{fig_tag}{fold_name} — ',
        )

        # Save per-fold importance CSV
        rows = []
        for stratum, imp in imp_by_stratum.items():
            for t, name in enumerate(token_names):
                rows.append({
                    'fold':    fold_name,
                    'stratum': stratum,
                    'token':   name,
                    'block':   'nonb' if t < n_nonb else 'te',
                    'importance': float(imp[t]),
                })
        pd.DataFrame(rows).to_csv(
            fold_out / 'token_importance.csv', index=False
        )

        fold_importances.append(imp_by_stratum)

        # ── Head specialisation ───────────────────────────────────────────────
        head_map = None
        if data['attn_full'] is not None:
            head_map = head_specialisation(data['attn_full'])
            plot_head_specialisation(
                head_map, token_names, n_nonb,
                output_path = fold_out / 'head_specialisation.png',
                fig_tag     = f'{fig_tag}{fold_name} — ',
            )
            # Save CSV
            H = head_map.shape[0]
            rows = []
            for h in range(H):
                for t, name in enumerate(token_names):
                    rows.append({
                        'fold':  fold_name,
                        'head':  h + 1,
                        'token': name,
                        'block': 'nonb' if t < n_nonb else 'te',
                        'mean_attn': float(head_map[h, t]),
                    })
            pd.DataFrame(rows).to_csv(
                fold_out / 'head_specialisation.csv', index=False
            )
        else:
            warnings.warn(
                f"{fold_name}: attn_weights_full not found — "
                "head specialisation analysis skipped."
            )

        fold_head_maps.append(head_map)
        print(f"\n  {fold_name} complete → {fold_out}/")

    # ── Cross-fold summary ────────────────────────────────────────────────────
    if len(fold_importances) > 1:
        create_cross_fold_summary(
            fold_importances, fold_head_maps,
            token_names = fold_files and load_fold(fold_files[0])['token_names'],
            n_nonb      = len(load_fold(fold_files[0])['nonb_sgs']),
            output_dir  = output_dir,
            fig_tag     = fig_tag,
        )
    else:
        print("\nOnly one fold — skipping cross-fold summary.")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}/")
    print(f"\nStructure:")
    print(f"  {output_dir}/")
    print(f"    fold_0/")
    print(f"      token_importance_bars.png")
    print(f"      token_importance_heatmap.png")
    print(f"      token_importance_delta.png")
    print(f"      head_specialisation.png")
    print(f"      token_importance.csv")
    print(f"      head_specialisation.csv")
    print(f"    fold_1/ ...")
    print(f"    cross_fold_token_importance.png")
    print(f"    cross_fold_head_specialisation.png")
    print(f"    cross_fold_token_importance.csv")


if __name__ == '__main__':
    main()