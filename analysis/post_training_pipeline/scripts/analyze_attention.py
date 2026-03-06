#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attention weight distribution analysis for beta-VAE with cross-attention.

Key outputs
-----------
  attention_analysis/
    per_transcript_attention.csv
    group_attention_stats.csv
    attention_by_group.png
    dominant_modality.png
    attention_entropy.png
    cross_fold_consistency.png
    mean_positional_profile.png
    fold_N/positional_heatmap.png
    class_analysis/
      attention_by_class.png
      dominant_modality_by_class.png
      misclassified_direction.png
      positional_profile_by_class.png
      misclassified_direction_stats.csv
      class_dominance_chi2.csv
      class_attention_stats.csv

Usage
-----
    python analyze_attention.py --attn_dir path/to/fold_attn_npz/ \
                                 --output_dir attention_analysis/ \
                                 --gencode_version v47
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu, chi2_contingency
from scipy.special import entr
from scipy import stats as scipy_stats
from matplotlib.patches import Patch

# =============================================================================
# Figure style constants  (OUP double-column: 7 in total width at 350 DPI)
# =============================================================================
# Single-panel (main text): 3.54in * 350dpi = 1239px; 2 assembled = 2478px = OUP double-column
FIG_1      = 3.54  # single panel width
FIG_H      = 2.5   # single panel height — compact
# Two-panel supplementary
FIG_2      = 7.0   # two-panel total width
FIG_H2     = 3.0   # two-panel height
# Three-panel supplementary
FIG_3      = 9.5   # three-panel total width
# Heatmap (3x2 grid)
FIG_HM_W   = 7.0
FIG_HM_H   = 3.0   # per row
FONT_S     = 6     # tick labels, annotations, n= labels
FONT_M     = 7     # axis labels, subplot titles
FONT_L     = 8     # suptitle
LW_MAIN    = 1.0   # main plot lines
LW_THIN    = 0.6   # whiskers, grid, reference lines
LW_BOX     = 0.7   # boxstrip box edges
MARKER_S   = 1.0   # strip/scatter point size (s = MARKER_S**2)
STRIP_A    = 0.2   # strip alpha
LEGEND_S   = 6     # legend font size
LEG_KW     = dict(fontsize=LEGEND_S, handlelength=1.2,
                  borderpad=0.3, labelspacing=0.3)
DPI        = 350

sns.set_style('whitegrid')
plt.rcParams['figure.dpi']       = DPI
plt.rcParams['font.size']        = FONT_S
plt.rcParams['axes.titlesize']   = FONT_M
plt.rcParams['axes.labelsize']   = FONT_M
plt.rcParams['xtick.labelsize']  = FONT_S
plt.rcParams['ytick.labelsize']  = FONT_S
plt.rcParams['legend.fontsize']  = LEGEND_S
plt.rcParams['lines.linewidth']  = LW_MAIN
plt.rcParams['axes.linewidth']   = 0.6
plt.rcParams['xtick.major.width']= 0.6
plt.rcParams['ytick.major.width']= 0.6

PALETTE = {'easy': '#2ECC71', 'hard_correct': '#F39C12', 'misclassified': '#E74C3C'}
ORDER   = ['easy', 'hard_correct', 'misclassified']
LABELS  = {'easy': 'Easy', 'hard_correct': 'Hard (correct)', 'misclassified': 'Misclassified'}

CLASS_PALETTE = {
    'easy_lnc':          '#27AE60',
    'easy_pc':           '#A9DFBF',
    'hard_correct_lnc':  '#E67E22',
    'hard_correct_pc':   '#FAD7A0',
    'misclassified_lnc': '#C0392B',
    'misclassified_pc':  '#F1948A',
    'lnc_as_pc':         '#8E44AD',
    'pc_as_lnc':         '#2980B9',
}

CLASS_LABELS = {0: 'lncRNA', 1: 'mRNA'}

CLASS_GROUP_ORDER = [
    'easy_lnc', 'easy_pc',
    'hard_correct_lnc', 'hard_correct_pc',
    'misclassified_lnc', 'misclassified_pc',
]
CLASS_GROUP_LABELS = {
    'easy_lnc':          'Easy\nlncRNA',
    'easy_pc':           'Easy\nmRNA',
    'hard_correct_lnc':  'Hard\nlncRNA',
    'hard_correct_pc':   'Hard\nmRNA',
    'misclassified_lnc': 'Misclass\nlncRNA',
    'misclassified_pc':  'Misclass\nmRNA',
}


# =============================================================================
# Helper: safe boxplot+strip
# =============================================================================

def _boxstrip(ax, data, x_col, y_col, order, palette_list,
              strip_n=5000, strip_size=MARKER_S, strip_alpha=STRIP_A):
    rng = np.random.default_rng(42)
    for i, (group, color) in enumerate(zip(order, palette_list)):
        vals = data[data[x_col] == group][y_col].dropna().values
        if len(vals) == 0:
            continue
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        iqr  = q75 - q25
        wlo  = max(vals.min(), q25 - 1.5 * iqr)
        whi  = min(vals.max(), q75 + 1.5 * iqr)
        w    = 0.4
        ax.bar(i, q75 - q25, width=w, bottom=q25,
               color=color, edgecolor='black', linewidth=LW_BOX, zorder=2)
        ax.plot([i - w/2, i + w/2], [q50, q50], color='black', lw=LW_MAIN, zorder=3)
        ax.plot([i, i], [wlo, q25], color='black', lw=LW_THIN, zorder=2)
        ax.plot([i, i], [q75, whi], color='black', lw=LW_THIN, zorder=2)
        ax.plot([i - w/4, i + w/4], [wlo, wlo], color='black', lw=LW_THIN, zorder=2)
        ax.plot([i - w/4, i + w/4], [whi, whi], color='black', lw=LW_THIN, zorder=2)
        n_strip = min(strip_n, len(vals))
        idx     = rng.choice(len(vals), n_strip, replace=False)
        jitter  = rng.uniform(-0.15, 0.15, n_strip)
        ax.scatter(i + jitter, vals[idx], color=color, s=strip_size**2,
                   alpha=strip_alpha, zorder=1, edgecolors='none')

    ax.set_xticks(range(len(order)))
    ax.set_xlim(-0.5, len(order) - 0.5)
    return ax


# =============================================================================
# Data loading
# =============================================================================

def load_fold_data(attn_dir):
    attn_dir   = Path(attn_dir)
    fold_files = sorted(attn_dir.glob('fold_*_attn.npz'))
    if not fold_files:
        raise FileNotFoundError(f"No fold_*_attn.npz files in {attn_dir}")

    print(f"Found {len(fold_files)} fold file(s):")
    folds = {}
    for f in fold_files:
        fold_id = int(f.stem.split('_')[1])
        data    = np.load(f, allow_pickle=True)
        folds[fold_id] = {k: data[k] for k in
                  ['attn_weights', 'predictions', 'confidences',
                   'labels', 'transcript_ids', 'is_hard_case', 'seq_lengths']}
        n      = len(data['attn_weights'])
        n_hard = int(data['is_hard_case'].sum())
        L      = data['attn_weights'].shape[1]
        print(f"  fold_{fold_id}: {n:,} samples | {n_hard:,} hard "
              f"({100*n_hard/n:.1f}%) | L_encoded={L}")
    return folds


# =============================================================================
# Feature extraction
# =============================================================================

def assign_group(pred, label, is_hard):
    if pred != label:
        return 'misclassified'
    return 'hard_correct' if is_hard else 'easy'


def assign_class_group(pred, label, is_hard):
    group = assign_group(pred, label, is_hard)
    cls   = 'lnc' if label == 0 else 'pc'
    return f'{group}_{cls}'


def assign_misclass_direction(pred, label, correct):
    if correct:
        return None
    return 'lnc_as_pc' if label == 0 else 'pc_as_lnc'


def attention_entropy(attn_vec):
    p    = attn_vec / (attn_vec.sum() + 1e-12)
    L    = len(p)
    H    = float(entr(p).sum())
    Hmax = np.log(L) if L > 1 else 1.0
    return H / Hmax if Hmax > 0 else 0.0


def extract_attention_features(fold_data):
    attn    = fold_data['attn_weights']
    preds   = fold_data['predictions']
    confs   = fold_data['confidences']
    labels  = fold_data['labels']
    t_ids   = fold_data['transcript_ids']
    is_hard = fold_data['is_hard_case']

    rows = []
    for i in range(len(attn)):
        te_v   = attn[i, :, 0].astype(float)
        nonb_v = attn[i, :, 1].astype(float)

        mean_te   = float(te_v.mean())
        mean_nonb = float(nonb_v.mean())
        total     = mean_te + mean_nonb + 1e-12

        label   = int(labels[i])
        pred    = int(preds[i])
        correct = pred == label
        group   = assign_group(pred, label, bool(is_hard[i]))

        rows.append({
            'transcript_id':      str(t_ids[i]),
            'label':              label,
            'class_name':         CLASS_LABELS[label],
            'seq_length': int(fold_data['seq_lengths'][i]) if 'seq_lengths' in fold_data else None,
            'prediction':         pred,
            'confidence':         float(confs[i]),
            'is_hard':            bool(is_hard[i]),
            'correct':            correct,
            'group':              group,
            'class_group':        assign_class_group(pred, label, bool(is_hard[i])),
            'misclass_direction': assign_misclass_direction(pred, label, correct),
            'mean_te':        mean_te,
            'max_te':         float(te_v.max()),
            'std_te':         float(te_v.std()),
            'entropy_te':     attention_entropy(te_v),
            'mean_nonb':      mean_nonb,
            'max_nonb':       float(nonb_v.max()),
            'std_nonb':       float(nonb_v.std()),
            'entropy_nonb':   attention_entropy(nonb_v),
            'te_fraction':    mean_te / total,
            'dominant':       'TE' if mean_te >= mean_nonb else 'NonB',
        })

    return pd.DataFrame(rows)


# =============================================================================
# Statistics
# =============================================================================

def summarize_by_group(df, metrics, fold_id):
    rows = []
    for metric in metrics:
        print(f"\n  [{metric}] Fold {fold_id}:")
        print(f"  {'Group':20s}  {'N':>6}  {'mean':>8}  {'std':>8}  {'median':>8}")
        print("  " + "-" * 56)
        for g in ORDER:
            sub = df[df['group'] == g][metric].dropna()
            if not len(sub):
                continue
            rows.append({'fold': fold_id, 'metric': metric, 'group': g,
                         'n': len(sub), 'mean': sub.mean(), 'std': sub.std(),
                         'median': sub.median(), 'q25': sub.quantile(0.25),
                         'q75': sub.quantile(0.75)})
            print(f"  {g:20s}  {len(sub):6d}  {sub.mean():8.4f}  "
                  f"{sub.std():8.4f}  {sub.median():8.4f}")

        easy = df[df['group'] == 'easy'][metric].dropna()
        hard = df[df['group'] == 'hard_correct'][metric].dropna()
        if len(easy) > 10 and len(hard) > 10:
            _, p = mannwhitneyu(hard, easy, alternative='greater')
            print(f"  MWU (hard > easy): p={p:.4e}")
    return pd.DataFrame(rows)


# =============================================================================
# Masked positional mean
# =============================================================================

def masked_positional_mean(attn_arr, seq_lengths, stride=9, mod_idx=0):
    L        = attn_arr.shape[1]
    enc_lens = np.ceil(seq_lengths / stride).astype(int).clip(1, L)
    sum_attn = np.zeros(L)
    count    = np.zeros(L)
    for i in range(len(attn_arr)):
        elen = enc_lens[i]
        sum_attn[:elen] += attn_arr[i, :elen, mod_idx]
        count[:elen]    += 1
    return sum_attn / np.maximum(count, 1)


def _savefig(fig, path, suptitle='', top=None):
    if suptitle:
        fig.suptitle(suptitle, fontweight='bold')
    plt.tight_layout()
    if top is not None:
        fig.subplots_adjust(top=top)  # AFTER tight_layout
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Group-level figures
# =============================================================================

def plot_attention_by_group(all_df, output_dir, gc_tag=''):
    metrics = [
        ('mean_te',      'Mean TE attention',        'TE'),
        ('mean_nonb',    'Mean NonB attention',       'NonB'),
        ('entropy_te',   'Positional entropy (TE)',   'TE'),
        ('entropy_nonb', 'Positional entropy (NonB)', 'NonB'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(FIG_2, FIG_H2 * 2),
                             gridspec_kw={'hspace': 0.55, 'wspace': 0.35})
    for ax, (col, ylabel, modality) in zip(axes.flat, metrics):
        order = [g for g in ORDER if g in all_df['group'].values]

        sns.violinplot(data=all_df, x='group', y=col, order=order,
                       hue='group', palette=PALETTE, legend=False,
                       inner=None, ax=ax, cut=0, linewidth=LW_THIN, alpha=0.7)
        sns.stripplot(data=all_df.sample(min(3000, len(all_df)), random_state=42),
                      x='group', y=col, order=order, hue='group', palette=PALETTE,
                      legend=False, size=MARKER_S, alpha=STRIP_A,
                      jitter=True, ax=ax, zorder=2)

        easy = all_df[all_df['group'] == 'easy'][col].dropna()
        hard = all_df[all_df['group'] == 'hard_correct'][col].dropna()
        if len(easy) > 10 and len(hard) > 10:
            _, p = mannwhitneyu(hard, easy, alternative='greater')
            sig  = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
            ymax = all_df[col].quantile(0.97)
            ax.annotate('', xy=(1, ymax), xytext=(0, ymax),
                        arrowprops=dict(arrowstyle='-', color='black', lw=LW_THIN))
            ax.text(0.5, ymax * 1.01, sig, ha='center', va='bottom',
                    fontsize=FONT_M, fontweight='bold')

        ax.set_xlabel('')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{modality} — {ylabel}', fontweight='bold')
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([LABELS.get(g, g) for g in order], rotation=20, ha='right')
        for j, g in enumerate(order):
            n = (all_df['group'] == g).sum()
            ax.text(j, ax.get_ylim()[0], f'n={n:,}', ha='center',
                    va='top', fontsize=FONT_S - 1, color='gray')
    _savefig(fig, Path(output_dir) / 'attention_by_group.png',
             suptitle=f'{gc_tag}Attention Weight Distribution by Transcript Group')


def plot_dominant_modality(all_df, output_dir, gc_tag=''):
    counts = (all_df.groupby(['group', 'dominant'])
              .size().reset_index(name='n'))
    pivot  = counts.pivot(index='group', columns='dominant', values='n').fillna(0)
    pct    = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pct    = pct.reindex([g for g in ORDER if g in pct.index])

    groups_present = [g for g in ORDER if g in pivot.index]
    contingency    = np.array([
        [pivot.loc[g, 'TE']   if 'TE'   in pivot.columns else 0,
         pivot.loc[g, 'NonB'] if 'NonB' in pivot.columns else 0]
        for g in groups_present
    ], dtype=int)

    chi2_global, p_global, dof, _ = chi2_contingency(contingency)
    print(f"\n  Chi-squared (dominance, all groups): chi2={chi2_global:.2f} p={p_global:.3e}")

    pairs = [('easy vs hard_correct', 0, 1),
             ('easy vs misclassified', 0, 2),
             ('hard_correct vs misclassified', 1, 2)]
    alpha_corrected = 0.05 / len(pairs)
    pairwise_rows   = []
    for name, i, j in pairs:
        if i >= len(groups_present) or j >= len(groups_present):
            continue
        chi2_p, p_p, _, _ = chi2_contingency(contingency[[i, j], :])
        sig = '***' if p_p < alpha_corrected else ('*' if p_p < 0.05 else 'ns')
        print(f"    {name}: chi2={chi2_p:.2f}  p={p_p:.3e}  {sig}")
        pairwise_rows.append({'comparison': name, 'chi2': round(chi2_p, 4),
                              'p_value': p_p, 'significance_str': sig,
                              'alpha_bonferroni': alpha_corrected})

    pd.DataFrame([{'comparison': 'global', 'chi2': round(chi2_global, 4),
                   'p_value': p_global, 'significance_str':
                   '***' if p_global < 0.001 else 'ns'}] + pairwise_rows
                ).to_csv(Path(output_dir) / 'dominant_modality_chi2.csv', index=False)

    # --- Panel A: stacked bar ---
    fig, ax = plt.subplots(1, 1, figsize=(FIG_1, FIG_H))
    x         = np.arange(len(pct))
    width     = 0.5
    te_vals   = pct.get('TE',   pd.Series(np.zeros(len(pct)))).values
    nonb_vals = pct.get('NonB', pd.Series(np.zeros(len(pct)))).values

    bars_te   = ax.bar(x, te_vals,   width, label='TE dominant',
                       color='#3498DB', edgecolor='black', linewidth=LW_BOX)
    bars_nonb = ax.bar(x, nonb_vals, width, bottom=te_vals,
                       label='NonB dominant', color='#9B59B6',
                       edgecolor='black', linewidth=LW_BOX)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(g, g) for g in pct.index], rotation=20, ha='right')
    ax.set_ylabel('% of transcripts')
    ax.set_ylim(0, 115)
    ax.set_title(f'{gc_tag}Dominant Modality by Group', fontweight='bold')
    ax.legend(**LEG_KW)
    for bar, val in zip(bars_te, te_vals):
        if val > 5:
            ax.text(bar.get_x() + bar.get_width()/2, val/2,
                    f'{val:.0f}%', ha='center', va='center',
                    fontsize=FONT_S, color='white', fontweight='bold')
    for bar, nval, tval in zip(bars_nonb, nonb_vals, te_vals):
        if nval > 5:
            ax.text(bar.get_x() + bar.get_width()/2, tval + nval/2,
                    f'{nval:.0f}%', ha='center', va='center',
                    fontsize=FONT_S, color='white', fontweight='bold')
    sig_str = '***' if p_global < 0.001 else ('*' if p_global < 0.05 else 'ns')
    ax.text(0.98, 0.98, f'chi2({dof})={chi2_global:.1f}\np={p_global:.2e} {sig_str}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=FONT_S, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    _savefig(fig, Path(output_dir) / 'dominant_modality_bar.png')

    # --- Panel B: TE fraction violin ---
    fig, ax = plt.subplots(1, 1, figsize=(FIG_1, FIG_H))
    order = [g for g in ORDER if g in all_df['group'].values]
    sns.violinplot(data=all_df, x='group', y='te_fraction', order=order,
                   hue='group', palette=PALETTE, legend=False,
                   inner='box', ax=ax, cut=0, linewidth=LW_THIN)
    ax.axhline(0.5, color='black', linestyle='--', lw=LW_THIN,
               alpha=0.6, label='Equal weight')
    ax.set_xlabel('')
    ax.set_ylabel('TE fraction')
    ax.set_title(f'{gc_tag}TE Attention Fraction', fontweight='bold')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABELS.get(g, g) for g in order], rotation=20, ha='right')
    ax.legend(**LEG_KW)
    _savefig(fig, Path(output_dir) / 'dominant_modality_violin.png')


def plot_mean_positional_profile(all_feat, attn_arrays, output_dir, gc_tag=''):
    """Two single-panel PNGs (TE and NonB) for main-text figure assembly."""
    positions    = np.arange(58)
    group_styles = {
        'easy':          {'color': '#2ECC71', 'label': 'Easy',           'lw': LW_MAIN},
        'hard_correct':  {'color': '#E67E22', 'label': 'Hard (correct)', 'lw': LW_MAIN},
        'misclassified': {'color': '#E74C3C', 'label': 'Misclassified',  'lw': LW_MAIN},
    }
    for mod_name, mod_idx, fname in [('TE', 0, 'mean_positional_profile_TE.png'),
                                      ('NonB', 1, 'mean_positional_profile_NonB.png')]:
        fig, ax = plt.subplots(1, 1, figsize=(FIG_1, FIG_H))
        for group, style in group_styles.items():
            if group not in attn_arrays:
                continue
            arr, lengths = attn_arrays[group]
            mean = masked_positional_mean(arr, lengths, mod_idx=mod_idx)
            ci   = 1.96 * scipy_stats.sem(arr[:, :, mod_idx], axis=0)
            ax.plot(positions, mean, color=style['color'], lw=style['lw'],
                    label=f"{style['label']} (n={len(arr):,})")
            ax.fill_between(positions, mean - ci, mean + ci,
                            color=style['color'], alpha=0.15)
        ax.axhline(0.5, color='black', linestyle='--', lw=LW_THIN,
                   alpha=0.5, label='Equal weight')
        ax.set_xlabel('Encoded position')
        ax.set_ylabel('Mean attention weight')
        ax.set_title(f'{gc_tag}{mod_name} attention — positional profile', fontweight='bold')
        ax.set_xlim(0, 57)
        ax.legend(**LEG_KW)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=LW_THIN)
        _savefig(fig, Path(output_dir) / fname)


def plot_attention_entropy(all_df, output_dir, gc_tag=''):
    """1x2 supplementary: positional entropy by group."""
    fig, axes = plt.subplots(1, 2, figsize=(FIG_2, FIG_H2),
                             gridspec_kw={'wspace': 0.35})
    for ax, (col, title) in zip(axes, [
            ('entropy_te',   'Positional Entropy — TE'),
            ('entropy_nonb', 'Positional Entropy — NonB')]):
        order = [g for g in ORDER if g in all_df['group'].values]
        sns.violinplot(data=all_df, x='group', y=col, order=order,
                       hue='group', palette=PALETTE, legend=False,
                       inner='box', ax=ax, cut=0, linewidth=LW_THIN)
        easy = all_df[all_df['group'] == 'easy'][col].dropna()
        hard = all_df[all_df['group'] == 'hard_correct'][col].dropna()
        if len(easy) > 10 and len(hard) > 10:
            _, p = mannwhitneyu(hard, easy, alternative='greater')
            ax.set_title(f'{title}\np={p:.2e}', fontweight='bold')
        else:
            ax.set_title(title, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Norm. entropy (0=focused, 1=uniform)')
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([LABELS.get(g, g) for g in order], rotation=20, ha='right')
        ax.set_ylim(-0.05, 1.1)
    _savefig(fig, Path(output_dir) / 'attention_entropy.png',
             suptitle=f'{gc_tag}Attention Entropy by Group', top=0.75)


def plot_positional_heatmap(fold_data, output_dir, n_per_group=50, gc_tag=''):
    attn    = fold_data['attn_weights']
    preds   = fold_data['predictions']
    labels  = fold_data['labels']
    is_hard = fold_data['is_hard_case']
    correct = preds == labels
    rng     = np.random.default_rng(42)
    groups  = [
        ('Easy cases',     correct & ~is_hard, PALETTE['easy']),
        ('Hard (correct)', correct & is_hard,  PALETTE['hard_correct']),
        ('Misclassified',  ~correct,            PALETTE['misclassified']),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(FIG_HM_W, FIG_HM_H * 3),
                             gridspec_kw={'hspace': 0.5, 'wspace': 0.35})
    for row, (label_str, mask_bool, color) in enumerate(groups):
        idx = np.where(mask_bool)[0]
        idx = rng.choice(idx, min(n_per_group, len(idx)), replace=False) \
              if len(idx) else np.array([], dtype=int)
        for col_i, (feat_col, feat_name) in enumerate(
                [(0, 'TE attention'), (1, 'NonB attention')]):
            ax = axes[row, col_i]
            if not len(idx):
                ax.text(0.5, 0.5, 'No samples', ha='center', va='center',
                        transform=ax.transAxes)
                ax.axis('off')
                continue
            data_map  = attn[idx, :, feat_col].astype(float)
            order_idx = np.argsort(data_map.mean(axis=1))[::-1]
            data_map  = data_map[order_idx]
            vmax = max(data_map.max(), 1e-4)
            im   = ax.imshow(data_map, cmap='YlOrRd', aspect='auto',
                             vmin=0, vmax=vmax, interpolation='nearest')
            ax.set_title(f'{label_str} — {feat_name}',
                         fontweight='bold', color=color, pad=4)
            ax.set_xlabel('Encoded position')
            ax.set_ylabel('Transcript (sorted)')
            plt.colorbar(im, ax=ax, label='Attention weight',
                         fraction=0.046, pad=0.04)
    _savefig(fig, Path(output_dir) / 'positional_heatmap.png',
             suptitle=f'{gc_tag}Positional Attention Heatmap  |  Left: TE  |  Right: NonB')


def plot_cross_fold_consistency(stats_df, output_dir, gc_tag=''):
    """1x2 supplementary: cross-fold mean attention per group."""
    fig, axes = plt.subplots(1, 2, figsize=(FIG_2, FIG_H2),
                             gridspec_kw={'wspace': 0.35})
    for ax, metric in zip(axes, ['mean_te', 'mean_nonb']):
        sub_m = stats_df[stats_df['metric'] == metric]
        for group in ORDER:
            sub_g = sub_m[sub_m['group'] == group].sort_values('fold')
            if sub_g.empty:
                continue
            ax.plot(sub_g['fold'], sub_g['mean'], marker='o',
                    markersize=3, linewidth=LW_MAIN,
                    label=LABELS.get(group, group), color=PALETTE[group])
            ax.fill_between(sub_g['fold'],
                            sub_g['mean'] - sub_g['std'],
                            sub_g['mean'] + sub_g['std'],
                            alpha=0.15, color=PALETTE[group])
        ax.set_xlabel('Fold')
        ax.set_ylabel('Mean attention weight')
        title = 'TE attention' if metric == 'mean_te' else 'NonB attention'
        ax.set_title(f'{title} — cross-fold consistency', fontweight='bold')
        ax.legend(**LEG_KW)
        ax.grid(True, alpha=0.3, linewidth=LW_THIN)
    _savefig(fig, Path(output_dir) / 'cross_fold_consistency.png',
             suptitle=f'{gc_tag}Attention Pattern Stability Across Folds')


# =============================================================================
# Class-level figures
# =============================================================================

def plot_dominant_modality_by_class(all_df, class_out, gc_tag=''):
    """1x2 supplementary: stacked bar + TE fraction boxstrip by group x class."""
    fig, axes = plt.subplots(1, 2, figsize=(FIG_2, FIG_H2),
                             gridspec_kw={'wspace': 0.4})
    ax    = axes[0]
    order = [g for g in CLASS_GROUP_ORDER if g in all_df['class_group'].values]

    counts = (all_df.groupby(['class_group', 'dominant'])
              .size().reset_index(name='n'))
    pivot  = counts.pivot(index='class_group', columns='dominant', values='n').fillna(0)
    pivot  = pivot.reindex([g for g in order if g in pivot.index])
    pct    = pivot.div(pivot.sum(axis=1), axis=0) * 100

    x          = np.arange(len(pct))
    width      = 0.6
    te_vals    = pct.get('TE',   pd.Series(np.zeros(len(pct)), index=pct.index)).values
    nonb_vals  = pct.get('NonB', pd.Series(np.zeros(len(pct)), index=pct.index)).values
    bar_colors = [CLASS_PALETTE.get(g, '#AAAAAA') for g in pct.index]

    ax.bar(x, te_vals,   width, color=bar_colors,
           edgecolor='black', linewidth=LW_BOX, alpha=0.9)
    ax.bar(x, nonb_vals, width, bottom=te_vals, color=bar_colors,
           edgecolor='black', linewidth=LW_BOX, alpha=0.5, hatch='//')
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_GROUP_LABELS.get(g, g) for g in pct.index],
                       rotation=30, ha='right')
    ax.set_ylabel('% of transcripts')
    ax.set_ylim(0, 120)
    ax.set_title('Dominant Modality by Group × Class', fontweight='bold')
    ax.legend(handles=[
        Patch(facecolor='gray', edgecolor='black', alpha=0.9, label='TE dominant'),
        Patch(facecolor='gray', edgecolor='black', alpha=0.5,
              hatch='//', label='NonB dominant'),
    ], **LEG_KW)
    for i, (val, nval, tval) in enumerate(zip(te_vals, nonb_vals, te_vals)):
        if val > 8:
            ax.text(x[i], val/2, f'{val:.0f}%', ha='center', va='center',
                    fontsize=FONT_S, color='white', fontweight='bold')
        if nval > 8:
            ax.text(x[i], tval + nval/2, f'{nval:.0f}%', ha='center', va='center',
                    fontsize=FONT_S, color='white', fontweight='bold')

    ax           = axes[1]
    cg_order     = [g for g in CLASS_GROUP_ORDER if g in all_df['class_group'].values]
    palette_list = [CLASS_PALETTE.get(g, '#AAAAAA') for g in cg_order]

    _boxstrip(ax, all_df, 'class_group', 'te_fraction', cg_order, palette_list)
    ax.axhline(0.5, color='black', linestyle='--', lw=LW_THIN,
               alpha=0.6, label='Equal weight')
    ax.set_xlabel('')
    ax.set_ylabel('TE fraction')
    ax.set_title('TE Fraction by Group × Class', fontweight='bold')
    ax.set_xticklabels([CLASS_GROUP_LABELS.get(g, g) for g in cg_order],
                       rotation=30, ha='right')
    ax.legend(**LEG_KW)

    print("\n  Chi-squared (lnc vs pc dominance within each group):")
    chi2_rows  = []
    alpha_corr = 0.05 / len(ORDER)
    for g_base in ORDER:
        lnc_key = f'{g_base}_lnc'
        pc_key  = f'{g_base}_pc'
        if lnc_key not in pivot.index or pc_key not in pivot.index:
            continue
        ct = pivot.loc[[lnc_key, pc_key], ['TE', 'NonB']].values.astype(int)
        chi2_v, p_v, _, _ = chi2_contingency(ct)
        sig = '***' if p_v < alpha_corr else ('*' if p_v < 0.05 else 'ns')
        print(f"    {g_base}: chi2={chi2_v:.2f}  p={p_v:.3e}  {sig}")
        chi2_rows.append({'group': g_base, 'chi2': round(chi2_v, 4),
                          'p_value': p_v, 'significance_str': sig,
                          'alpha_bonferroni': alpha_corr})
    pd.DataFrame(chi2_rows).to_csv(class_out / 'class_dominance_chi2.csv', index=False)

    _savefig(fig, class_out / 'dominant_modality_by_class.png',
             suptitle=f'{gc_tag}Modality Dominance by Group and Transcript Class')


def plot_misclassified_direction(all_df, attn_arrays_by_direction, class_out, gc_tag=''):
    misclass   = all_df[all_df['group'] == 'misclassified'].copy()
    directions = ['lnc_as_pc', 'pc_as_lnc']
    dir_labels = {
        'lnc_as_pc': 'lncRNA→mRNA',
        'pc_as_lnc': 'mRNA→lncRNA',
    }
    dir_colors = {
        'lnc_as_pc': CLASS_PALETTE['lnc_as_pc'],
        'pc_as_lnc': CLASS_PALETTE['pc_as_lnc'],
    }

    present      = [d for d in directions if d in misclass['misclass_direction'].values]
    palette_list = [dir_colors[d] for d in present]

    fig, axes = plt.subplots(1, 3, figsize=(FIG_3, FIG_H2),
                             gridspec_kw={'wspace': 0.4})
    for ax, (col, ylabel, mod) in zip(axes[:2], [
            ('mean_te',   'Mean TE attention',   'TE'),
            ('mean_nonb', 'Mean NonB attention', 'NonB')]):

        _boxstrip(ax, misclass, 'misclass_direction', col,
                  present, palette_list,
                  strip_n=len(misclass), strip_size=MARKER_S * 1.5, strip_alpha=0.3)

        ax.axhline(0.5, color='black', linestyle='--', lw=LW_THIN, alpha=0.5)
        ax.set_xlabel('')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{mod} attention\nby misclass. direction', fontweight='bold')
        ax.set_xticklabels([dir_labels.get(d, d) for d in present],
                           rotation=15, ha='right')
        d0 = misclass[misclass['misclass_direction'] == 'lnc_as_pc'][col].dropna()
        d1 = misclass[misclass['misclass_direction'] == 'pc_as_lnc'][col].dropna()
        if len(d0) > 10 and len(d1) > 10:
            _, p = mannwhitneyu(d0, d1, alternative='two-sided')
            sig  = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
            ymax = misclass[col].quantile(0.96)
            ax.annotate('', xy=(1, ymax), xytext=(0, ymax),
                        arrowprops=dict(arrowstyle='-', color='black', lw=LW_THIN))
            ax.text(0.5, ymax * 1.005, f'{sig}\np={p:.2e}',
                    ha='center', va='bottom', fontsize=FONT_S)

        for j, d in enumerate(present):
            n = (misclass['misclass_direction'] == d).sum()
            ax.text(j, ax.get_ylim()[0], f'n={n:,}', ha='center',
                    va='top', fontsize=FONT_S - 1, color='gray')

    ax        = axes[2]
    positions = np.arange(58)
    for direction in directions:
        if direction not in attn_arrays_by_direction:
            continue
        color       = dir_colors[direction]
        short_label = dir_labels[direction]
        for mod_idx, (mod_name, ls) in enumerate([('TE', '-'), ('NonB', '--')]):
            arr, lengths = attn_arrays_by_direction[direction]
            mean = masked_positional_mean(arr, lengths, mod_idx=mod_idx)
            ci   = 1.96 * scipy_stats.sem(arr[:, :, mod_idx], axis=0)
            ax.plot(positions, mean, color=color, lw=LW_MAIN, ls=ls,
                    label=f'{short_label} — {mod_name}')
            ax.fill_between(positions, mean - ci, mean + ci,
                            color=color, alpha=0.10)
    ax.axhline(0.5, color='black', linestyle=':', lw=LW_THIN, alpha=0.5)
    ax.set_xlabel('Encoded position')
    ax.set_ylabel('Mean attention weight')
    ax.set_title('Positional profile\nby misclass. direction', fontweight='bold')
    ax.set_xlim(0, 57)
    ax.legend(ncol=2, **LEG_KW)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=LW_THIN)

    dir_stats = []
    for col in ['mean_te', 'mean_nonb', 'te_fraction']:
        d0 = misclass[misclass['misclass_direction'] == 'lnc_as_pc'][col].dropna()
        d1 = misclass[misclass['misclass_direction'] == 'pc_as_lnc'][col].dropna()
        if len(d0) > 10 and len(d1) > 10:
            _, p = mannwhitneyu(d0, d1, alternative='two-sided')
            dir_stats.append({
                'metric':         col,
                'lnc_as_pc_mean': d0.mean(), 'lnc_as_pc_std': d0.std(),
                'pc_as_lnc_mean': d1.mean(), 'pc_as_lnc_std': d1.std(),
                'mwu_p':          p,
                'significance':   '***' if p < 0.001 else ('**' if p < 0.01
                                  else ('*' if p < 0.05 else 'ns')),
            })
    pd.DataFrame(dir_stats).to_csv(class_out / 'misclassified_direction_stats.csv', index=False)
    print(f"  Saved: misclassified_direction_stats.csv")

    _savefig(fig, class_out / 'misclassified_direction.png',
             suptitle=f'{gc_tag}Misclassified Transcripts: lncRNA→mRNA vs mRNA→lncRNA', top=0.75)


def plot_positional_profile_by_class(attn_arrays_by_class, class_out, gc_tag=''):
    """1x2 supplementary: positional profile split by group x class."""
    fig, axes = plt.subplots(1, 2, figsize=(FIG_2, FIG_H2), sharey=False,
                             gridspec_kw={'wspace': 0.35})
    positions = np.arange(58)
    styles = {
        'easy_lnc':          {'color': CLASS_PALETTE['easy_lnc'],          'ls': '-',  'lw': LW_MAIN},
        'easy_pc':           {'color': CLASS_PALETTE['easy_pc'],           'ls': '--', 'lw': LW_THIN},
        'hard_correct_lnc':  {'color': CLASS_PALETTE['hard_correct_lnc'],  'ls': '-',  'lw': LW_MAIN},
        'hard_correct_pc':   {'color': CLASS_PALETTE['hard_correct_pc'],   'ls': '--', 'lw': LW_THIN},
        'misclassified_lnc': {'color': CLASS_PALETTE['misclassified_lnc'], 'ls': '-',  'lw': LW_MAIN},
        'misclassified_pc':  {'color': CLASS_PALETTE['misclassified_pc'],  'ls': '--', 'lw': LW_THIN},
    }
    for ax, (mod_name, mod_idx) in zip(axes, [('TE', 0), ('NonB', 1)]):
        for cg, style in styles.items():
            if cg not in attn_arrays_by_class:
                continue
            arr, lengths = attn_arrays_by_class[cg]
            mean = masked_positional_mean(arr, lengths, mod_idx=mod_idx)
            ci   = 1.96 * scipy_stats.sem(arr[:, :, mod_idx], axis=0)
            label = f"{CLASS_GROUP_LABELS.get(cg, cg).replace(chr(10), ' ')} (n={len(arr):,})"
            ax.plot(positions, mean, color=style['color'],
                    lw=style['lw'], ls=style['ls'], label=label)
            ax.fill_between(positions, mean - ci, mean + ci,
                            color=style['color'], alpha=0.08)
        ax.axhline(0.5, color='black', linestyle=':', lw=LW_THIN, alpha=0.5)
        ax.set_xlabel('Encoded position')
        ax.set_ylabel('Mean attention weight')
        ax.set_title(f'{mod_name} — profile by class', fontweight='bold')
        ax.set_xlim(0, 57)
        ax.legend(ncol=2, **LEG_KW)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=LW_THIN)
    _savefig(fig, class_out / 'positional_profile_by_class.png',
             suptitle=f'{gc_tag}Positional Profile by Group × Class  |  Solid=lncRNA  Dashed=mRNA')

def save_class_attention_stats(all_df, class_out):
    rows = []
    for cg in CLASS_GROUP_ORDER:
        sub = all_df[all_df['class_group'] == cg]
        if sub.empty:
            continue
        for metric in ['mean_te', 'mean_nonb', 'entropy_te', 'entropy_nonb', 'te_fraction']:
            rows.append({'class_group': cg, 'metric': metric, 'n': len(sub),
                         'mean': sub[metric].mean(), 'std': sub[metric].std(),
                         'median': sub[metric].median()})
    pd.DataFrame(rows).to_csv(class_out / 'class_attention_stats.csv', index=False)
    print(f"  Saved: class_attention_stats.csv")


# =============================================================================
# Verdict
# =============================================================================

def print_verdict(all_df, output_dir=None):
    print("\n" + "=" * 80)
    print("INTERPRETABILITY VERDICT")
    print("=" * 80)

    mwu_rows = []
    for metric, direction, desc in [
        ('mean_te',      'greater', 'Hard cases use MORE TE attention'),
        ('mean_nonb',    'greater', 'Hard cases use MORE NonB attention'),
        ('entropy_te',   'greater', 'Hard cases show MORE diffuse TE attention'),
        ('entropy_nonb', 'greater', 'Hard cases show MORE diffuse NonB attention'),
    ]:
        easy = all_df[all_df['group'] == 'easy'][metric].dropna()
        hard = all_df[all_df['group'] == 'hard_correct'][metric].dropna()
        if len(easy) < 10 or len(hard) < 10:
            continue
        _, p = mannwhitneyu(hard, easy, alternative=direction)
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        print(f"\n  {desc}")
        print(f"     easy: {easy.mean():.4f} +/- {easy.std():.4f}  "
              f"hard: {hard.mean():.4f} +/- {hard.std():.4f}  p={p:.4e}")
        mwu_rows.append({
            'metric':      metric,
            'description': desc,
            'easy_mean':   easy.mean(),
            'easy_std':    easy.std(),
            'easy_n':      len(easy),
            'hard_mean':   hard.mean(),
            'hard_std':    hard.std(),
            'hard_n':      len(hard),
            'mwu_p':       p,
            'significance': sig,
            'alternative': direction,
        })

    easy_dom = all_df[all_df['group'] == 'easy']['dominant'].value_counts(normalize=True)
    hard_dom = all_df[all_df['group'] == 'hard_correct']['dominant'].value_counts(normalize=True)
    print(f"\n  Modality dominance (TE%):")
    print(f"    Easy:         {easy_dom.get('TE', 0)*100:.1f}%")
    print(f"    Hard correct: {hard_dom.get('TE', 0)*100:.1f}%")

    dominance_rows = []
    for group_name, dom in [('easy', easy_dom), ('hard_correct', hard_dom)]:
        dominance_rows.append({
            'group':        group_name,
            'te_dominant_pct': dom.get('TE', 0) * 100,
            'nonb_dominant_pct': dom.get('NonB', 0) * 100,
            'n': len(all_df[all_df['group'] == group_name]),
        })

    print(f"\n  Misclassification direction breakdown:")
    misclass_rows = []
    for direction in ['lnc_as_pc', 'pc_as_lnc']:
        sub = all_df[all_df['misclass_direction'] == direction]
        if not sub.empty:
            print(f"    {direction}: n={len(sub):,}  "
                  f"mean_te={sub['mean_te'].mean():.4f}  "
                  f"mean_nonb={sub['mean_nonb'].mean():.4f}  "
                  f"TE_dominant={sub['dominant'].eq('TE').mean()*100:.1f}%")
            misclass_rows.append({
                'direction':        direction,
                'n':                len(sub),
                'mean_te':          sub['mean_te'].mean(),
                'std_te':           sub['mean_te'].std(),
                'mean_nonb':        sub['mean_nonb'].mean(),
                'std_nonb':         sub['mean_nonb'].std(),
                'te_dominant_pct':  sub['dominant'].eq('TE').mean() * 100,
            })

    print("=" * 80)

    # --- Save CSVs ---
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if mwu_rows:
            mwu_df = pd.DataFrame(mwu_rows)
            mwu_path = output_dir / 'group_mwu_results.csv'
            mwu_df.to_csv(mwu_path, index=False)
            print(f"\n  Saved: {mwu_path}")

        if dominance_rows:
            dom_df = pd.DataFrame(dominance_rows)
            dom_path = output_dir / 'modality_dominance_summary.csv'
            dom_df.to_csv(dom_path, index=False)
            print(f"  Saved: {dom_path}")

        if misclass_rows:
            mis_df = pd.DataFrame(misclass_rows)
            mis_path = output_dir / 'misclassified_direction_summary.csv'
            mis_df.to_csv(mis_path, index=False)
            print(f"  Saved: {mis_path}")

    return (
        pd.DataFrame(mwu_rows) if mwu_rows else None,
        pd.DataFrame(dominance_rows) if dominance_rows else None,
        pd.DataFrame(misclass_rows) if misclass_rows else None,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Attention distribution analysis')
    parser.add_argument('--attn_dir',        required=True)
    parser.add_argument('--output_dir',      default='attention_analysis')
    parser.add_argument('--n_heatmap',       type=int, default=50)
    parser.add_argument('--gencode_version', default='',
                        help='GENCODE version label, e.g. "v47" or "v49"')
    parser.add_argument('--model_label',      default='',
                        help='Model label for figure titles (e.g. "bvae+attn")')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    class_out = output_dir / 'class_analysis'
    class_out.mkdir(exist_ok=True)

    gc_tag = ''
    if args.model_label:
        gc_tag += args.model_label + ' | '
    if args.gencode_version:
        gc_tag += f'GENCODE {args.gencode_version} — '

    print("=" * 80)
    print("ATTENTION WEIGHT DISTRIBUTION ANALYSIS")
    print("=" * 80)

    folds = load_fold_data(args.attn_dir)

    all_feat_dfs  = []
    all_stats_dfs = []
    key_metrics   = ['mean_te', 'mean_nonb', 'entropy_te', 'entropy_nonb', 'te_fraction']

    for fold_id, fold_data in sorted(folds.items()):
        print(f"\n{'='*60}\nFold {fold_id}\n{'='*60}")
        feat_df         = extract_attention_features(fold_data)
        feat_df['fold'] = fold_id
        all_feat_dfs.append(feat_df)
        all_stats_dfs.append(summarize_by_group(feat_df, key_metrics, fold_id))
        fold_out = output_dir / f'fold_{fold_id}'
        fold_out.mkdir(exist_ok=True)
        plot_positional_heatmap(fold_data, fold_out,
                                n_per_group=args.n_heatmap, gc_tag=gc_tag)

    all_feat  = pd.concat(all_feat_dfs,  ignore_index=True)
    all_stats = pd.concat(all_stats_dfs, ignore_index=True)
    all_feat.to_csv(output_dir  / 'per_transcript_attention.csv',  index=False)
    all_stats.to_csv(output_dir / 'group_attention_stats.csv',     index=False)
    print(f"\nSaved CSVs ({len(all_feat):,} transcripts total)")

    fold_list = sorted(folds.keys())

    def _gather(mask_fn):
        arrays  = []
        lengths = []
        for fold_id in fold_list:
            feat_df   = all_feat_dfs[fold_list.index(fold_id)]
            fold_mask = mask_fn(feat_df)
            if fold_mask.any():
                arrays.append(folds[fold_id]['attn_weights'][fold_mask.values])
                lengths.append(feat_df[fold_mask]['seq_length'].values)
        if not arrays:
            return None, None
        return np.concatenate(arrays, axis=0), np.concatenate(lengths, axis=0)

    attn_arrays = {g: _gather(lambda df, g=g: df['group'] == g) for g in ORDER}

    attn_arrays_by_class = {}
    for cg in CLASS_GROUP_ORDER:
        arr, lens = _gather(lambda df, cg=cg: df['class_group'] == cg)
        if arr is not None:
            attn_arrays_by_class[cg] = (arr, lens)

    attn_arrays_by_direction = {}
    for d in ['lnc_as_pc', 'pc_as_lnc']:
        arr, lens = _gather(lambda df, d=d: df['misclass_direction'] == d)
        if arr is not None:
            attn_arrays_by_direction[d] = (arr, lens)

    print("\nGenerating group-level figures...")
    plot_attention_by_group(all_feat, output_dir, gc_tag)
    plot_dominant_modality(all_feat, output_dir, gc_tag)
    plot_attention_entropy(all_feat, output_dir, gc_tag)
    plot_cross_fold_consistency(all_stats, output_dir, gc_tag)
    plot_mean_positional_profile(all_feat, attn_arrays, output_dir, gc_tag)

    print("\nGenerating class-level figures...")
    plot_dominant_modality_by_class(all_feat, class_out, gc_tag)
    plot_misclassified_direction(all_feat, attn_arrays_by_direction, class_out, gc_tag)
    plot_positional_profile_by_class(attn_arrays_by_class, class_out, gc_tag)
    save_class_attention_stats(all_feat, class_out)

    print_verdict(all_feat, output_dir=output_dir / 'verdict')
    print(f"\nAll outputs -> {output_dir}/")


if __name__ == '__main__':
    main()