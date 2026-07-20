#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_attention.py

Tests whether the sub-groups highlighted by attention (GQ, TE_CORE)
are statistically different between lncRNA and mRNA in the raw features,
independently of any model training.

For each sub-group identified as attention-dominant, computes:
  - Mann-Whitney U test (lncRNA vs mRNA) per feature within the sub-group
  - Effect size (rank-biserial correlation)
  - % of features in the sub-group that are significantly different
  - Comparison against low-attention sub-groups (e.g. APR, DR) as controls

This provides model-independent evidence for or against the attention findings.

Usage
-----
python src/validate_attention.py \
    --te_csv   data/processed_features_genomic/g49_te_features_clean.csv \
    --nonb_csv data/processed_features_genomic/g49_nonb_features_clean.csv \
    --lnc_fasta data/split_gencode_49/lnc_trainval.fa \
    --pc_fasta  data/split_gencode_49/pc_trainval.fa \
    --output_dir gencode_v49_experiments/beta_vae_gated_g49_debug/feature_validation
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from Bio import SeqIO

# Import registry for sub-group index lookups
import sys
sys.path.insert(0, str(Path(__file__).parent))
from data.feature_registry import REGISTRY

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

# Sub-groups to validate — ordered by attention weight (high to low)
# from the training results
ATTENTION_ORDER = ['GQ', 'TE_CORE', 'TE_UNKNOWN', 'TE_QUALITY',
                   'APR', 'DR', 'IR', 'STR', 'TRI', 'Z', 'GLOBAL',
                   'TE_LCTR', 'TE_PSEUDO', 'TE_GLOBAL', 'MR']

# Control sub-groups (low attention) for comparison
CONTROL_SUBGROUPS = ['APR', 'DR', 'MR']


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ids_from_fasta(fasta_path: Path) -> set:
    ids = set()
    for record in SeqIO.parse(fasta_path, 'fasta'):
        ids.add(record.id.split('|')[0])
    return ids


def load_features_with_labels(te_csv, nonb_csv, lnc_fasta, pc_fasta):
    print("Loading features...")
    te_df   = pd.read_csv(te_csv,   index_col='transcript_id')
    nonb_df = pd.read_csv(nonb_csv, index_col='transcript_id')

    # Drop metadata columns
    for col in ['transcript_type', 'coding_class', 'transcript_length']:
        te_df.drop(columns=[col], errors='ignore', inplace=True)
        nonb_df.drop(columns=[col], errors='ignore', inplace=True)

    print(f"  TE features:   {te_df.shape}")
    print(f"  NonB features: {nonb_df.shape}")

    # Build label series
    lnc_ids = load_ids_from_fasta(Path(lnc_fasta))
    pc_ids  = load_ids_from_fasta(Path(pc_fasta))

    all_ids = set(te_df.index) & set(nonb_df.index) & (lnc_ids | pc_ids)
    labels  = pd.Series(
        {tid: 0 if tid in lnc_ids else 1 for tid in all_ids},
        name='label'
    )

    te_df   = te_df.loc[labels.index]
    nonb_df = nonb_df.loc[labels.index]

    n_lnc = (labels == 0).sum()
    n_pc  = (labels == 1).sum()
    print(f"  lncRNA: {n_lnc:,} | mRNA: {n_pc:,} | total: {len(labels):,}")

    return te_df, nonb_df, labels


# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------

def test_subgroup(feature_df: pd.DataFrame,
                  labels: pd.Series,
                  subgroup: str,
                  block: str,
                  alpha: float = 0.05) -> pd.DataFrame:
    """
    Mann-Whitney U test for each feature in a sub-group.
    Returns DataFrame with test results per feature.
    """
    if block == 'nonb':
        indices = REGISTRY.nonb_indices_for_subgroup(subgroup)
        cols    = [REGISTRY.nonb_features[i].name for i in indices]
    else:
        indices = REGISTRY.te_indices_for_subgroup(subgroup)
        cols    = [REGISTRY.te_features[i].name for i in indices]

    # Filter to columns present in df
    cols = [c for c in cols if c in feature_df.columns]
    if not cols:
        return pd.DataFrame()

    lnc_df = feature_df.loc[labels == 0, cols]
    pc_df  = feature_df.loc[labels == 1, cols]

    rows = []
    for col in cols:
        lnc_vals = lnc_df[col].values
        pc_vals  = pc_df[col].values

        # Skip constant features
        if lnc_vals.std() == 0 and pc_vals.std() == 0:
            continue

        u_stat, p_val = stats.mannwhitneyu(
            lnc_vals, pc_vals, alternative='two-sided'
        )
        # Rank-biserial correlation as effect size
        n1, n2 = len(lnc_vals), len(pc_vals)
        r = 1 - (2 * u_stat) / (n1 * n2)

        rows.append({
            'subgroup':    subgroup,
            'block':       block,
            'feature':     col,
            'u_stat':      float(u_stat),
            'p_value':     float(p_val),
            'effect_size': float(r),
            'lnc_mean':    float(lnc_vals.mean()),
            'pc_mean':     float(pc_vals.mean()),
            'lnc_median':  float(np.median(lnc_vals)),
            'pc_median':   float(np.median(pc_vals)),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Bonferroni correction within sub-group
    n_tests = len(df)
    df['p_adjusted'] = (df['p_value'] * n_tests).clip(upper=1.0)
    df['significant'] = df['p_adjusted'] < alpha
    df = df.sort_values('p_value')
    return df


def subgroup_summary(test_df: pd.DataFrame) -> dict:
    """Summarise test results for a sub-group."""
    if test_df.empty:
        return {'n_features': 0, 'n_significant': 0, 'pct_significant': 0,
                'mean_effect_size': 0, 'max_effect_size': 0}
    return {
        'n_features':       len(test_df),
        'n_significant':    int(test_df['significant'].sum()),
        'pct_significant':  float(100 * test_df['significant'].mean()),
        'mean_effect_size': float(test_df['effect_size'].abs().mean()),
        'max_effect_size':  float(test_df['effect_size'].abs().max()),
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_subgroup_comparison(summaries: dict, output_path: Path,
                              fig_tag: str = '') -> None:
    """
    Bar chart comparing % significant features per sub-group,
    ordered by attention weight. Highlights attention-dominant sub-groups.
    """
    subgroups = [sg for sg in ATTENTION_ORDER if sg in summaries]
    pct_sig   = [summaries[sg]['pct_significant'] for sg in subgroups]
    eff_sizes = [summaries[sg]['mean_effect_size'] for sg in subgroups]

    # Colour: high attention = gold, control = gray, others = steelblue
    high_attn = {'GQ', 'TE_CORE', 'TE_UNKNOWN'}
    colors = ['#F4B942' if sg in high_attn
              else '#AAAAAA' if sg in CONTROL_SUBGROUPS
              else '#4A90D9'
              for sg in subgroups]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Top: % significant features
    bars = ax1.bar(subgroups, pct_sig, color=colors, edgecolor='black',
                   linewidth=0.7, width=0.7)
    ax1.axvline(x=8.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
    ax1.set_ylabel('% features significant\n(Bonferroni-corrected, lncRNA vs mRNA)',
                   fontsize=12)
    ax1.set_title(f'{fig_tag}Statistical Validation of Attention Findings\n'
                  'Are attention-dominant sub-groups actually different between lncRNA and mRNA?',
                  fontsize=13, fontweight='bold')
    ax1.set_xticklabels(subgroups, rotation=45, ha='right', fontsize=10)
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Annotate bars
    for bar, val in zip(bars, pct_sig):
        if val > 2:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{val:.0f}%', ha='center', va='bottom', fontsize=9)

    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(facecolor='#F4B942', edgecolor='black', label='High attention (model finding)'),
        Patch(facecolor='#4A90D9', edgecolor='black', label='Medium/low attention'),
        Patch(facecolor='#AAAAAA', edgecolor='black', label='Control (low attention)'),
    ], fontsize=10, loc='upper right')

    # Bottom: mean effect size
    bars2 = ax2.bar(subgroups, eff_sizes, color=colors, edgecolor='black',
                    linewidth=0.7, width=0.7)
    ax2.axvline(x=8.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
    ax2.set_ylabel('Mean |effect size|\n(rank-biserial correlation)',
                   fontsize=12)
    ax2.set_xticklabels(subgroups, rotation=45, ha='right', fontsize=10)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Block labels
    for ax in [ax1, ax2]:
        ax.text(4, ax.get_ylim()[1] * 0.95, '← NonB DNA →',
                ha='center', fontsize=10, color='navy', style='italic')
        ax.text(12, ax.get_ylim()[1] * 0.95, '← TE →',
                ha='center', fontsize=10, color='darkorange', style='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_top_features(all_results: dict, output_path: Path,
                      top_n: int = 10, fig_tag: str = '') -> None:
    """
    Show the top-N most significant features from high-attention sub-groups,
    with their effect sizes and direction (lncRNA > mRNA or vice versa).
    """
    # Collect all results from high-attention sub-groups
    high_attn_dfs = []
    for sg in ['GQ', 'TE_CORE', 'TE_UNKNOWN']:
        if sg in all_results and not all_results[sg].empty:
            high_attn_dfs.append(all_results[sg])

    if not high_attn_dfs:
        print("  No results for high-attention sub-groups")
        return

    combined = pd.concat(high_attn_dfs).sort_values('p_adjusted')
    top = combined.head(top_n)

    if top.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#FF6B6B' if r < 0 else '#4ECDC4'
              for r in top['effect_size']]
    bars = ax.barh(range(len(top)), top['effect_size'].values,
                   color=colors, edgecolor='black', linewidth=0.5)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(
        [f"{row['feature']} [{row['subgroup']}]"
         for _, row in top.iterrows()],
        fontsize=10
    )
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Effect size (rank-biserial correlation)\n'
                  'negative = lncRNA higher | positive = mRNA higher',
                  fontsize=11)
    ax.set_title(f'{fig_tag}Top {top_n} Most Significant Features\n'
                 'from Attention-Dominant Sub-groups (GQ, TE_CORE, TE_UNKNOWN)',
                 fontsize=13, fontweight='bold')

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='#FF6B6B', label='lncRNA > mRNA'),
        Patch(facecolor='#4ECDC4', label='mRNA > lncRNA'),
    ], fontsize=10)
    ax.grid(True, axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Validate attention findings against raw feature statistics'
    )
    parser.add_argument('--te_csv',     required=True)
    parser.add_argument('--nonb_csv',   required=True)
    parser.add_argument('--lnc_fasta',  required=True)
    parser.add_argument('--pc_fasta',   required=True)
    parser.add_argument('--output_dir', default='feature_validation')
    parser.add_argument('--model_label',     default='β-VAE Gated')
    parser.add_argument('--gencode_version', default='v49')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag_parts = [p for p in [args.model_label,
                 f'GENCODE {args.gencode_version}'] if p]
    fig_tag   = ' | '.join(tag_parts) + ' — ' if tag_parts else ''

    # Load data
    te_df, nonb_df, labels = load_features_with_labels(
        args.te_csv, args.nonb_csv, args.lnc_fasta, args.pc_fasta
    )

    # Test all sub-groups
    print("\nRunning statistical tests...")
    all_results = {}
    summaries   = {}

    nonb_subgroups = REGISTRY.nonb_subgroups
    te_subgroups   = REGISTRY.te_subgroups

    for sg in nonb_subgroups:
        print(f"  NonB/{sg}...", end=' ')
        result = test_subgroup(nonb_df, labels, sg, 'nonb')
        all_results[sg] = result
        summaries[sg]   = subgroup_summary(result)
        s = summaries[sg]
        print(f"{s['n_significant']}/{s['n_features']} sig "
              f"({s['pct_significant']:.0f}%), "
              f"mean |r|={s['mean_effect_size']:.3f}")

    for sg in te_subgroups:
        print(f"  TE/{sg}...", end=' ')
        result = test_subgroup(te_df, labels, sg, 'te')
        all_results[sg] = result
        summaries[sg]   = subgroup_summary(result)
        s = summaries[sg]
        print(f"{s['n_significant']}/{s['n_features']} sig "
              f"({s['pct_significant']:.0f}%), "
              f"mean |r|={s['mean_effect_size']:.3f}")

    # Summary table
    summary_df = pd.DataFrame(summaries).T.reset_index()
    summary_df.columns = ['subgroup'] + list(summary_df.columns[1:])
    summary_df['attention_rank'] = summary_df['subgroup'].map(
        {sg: i for i, sg in enumerate(ATTENTION_ORDER)}
    )
    summary_df = summary_df.sort_values('attention_rank')
    summary_df.to_csv(output_dir / 'subgroup_significance_summary.csv', index=False)

    # Print ranked comparison
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY — sorted by attention rank")
    print("=" * 70)
    print(f"{'Rank':<5} {'Subgroup':<15} {'%Sig':>6} {'MeanR':>7} {'MaxR':>7}")
    print("-" * 45)
    for _, row in summary_df.iterrows():
        rank = int(row['attention_rank']) + 1
        marker = ' ◄ HIGH ATTENTION' if row['subgroup'] in {'GQ','TE_CORE','TE_UNKNOWN'} else ''
        print(f"{rank:<5} {row['subgroup']:<15} "
              f"{row['pct_significant']:>5.0f}% "
              f"{row['mean_effect_size']:>7.3f} "
              f"{row['max_effect_size']:>7.3f}"
              f"{marker}")

    # Save all test results
    all_df = pd.concat(
        [df for df in all_results.values() if not df.empty],
        ignore_index=True
    )
    all_df.to_csv(output_dir / 'all_feature_tests.csv', index=False)
    print(f"\nSaved: all_feature_tests.csv ({len(all_df):,} features tested)")

    # Visualisations
    print("\nGenerating figures...")
    plot_subgroup_comparison(
        summaries, output_dir / 'subgroup_significance.png', fig_tag=fig_tag
    )
    plot_top_features(
        all_results, output_dir / 'top_features_high_attention.png',
        fig_tag=fig_tag
    )

    # Conclusion
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    gq_pct  = summaries.get('GQ',       {}).get('pct_significant', 0)
    tc_pct  = summaries.get('TE_CORE',  {}).get('pct_significant', 0)
    apr_pct = summaries.get('APR',      {}).get('pct_significant', 0)
    dr_pct  = summaries.get('DR',       {}).get('pct_significant', 0)
    ctrl_pct = (apr_pct + dr_pct) / 2

    print(f"\nHigh-attention sub-groups:")
    print(f"  GQ:       {gq_pct:.0f}% of features significant")
    print(f"  TE_CORE:  {tc_pct:.0f}% of features significant")
    print(f"\nControl sub-groups (low attention):")
    print(f"  APR:      {apr_pct:.0f}% of features significant")
    print(f"  DR:       {dr_pct:.0f}% of features significant")
    print(f"  Mean:     {ctrl_pct:.0f}%")

    if gq_pct > ctrl_pct * 1.5 or tc_pct > ctrl_pct * 1.5:
        print("\n VALIDATED: High-attention sub-groups show substantially more")
        print("  significant differences than controls — the model found real signal.")
    else:
        print("\n INCONCLUSIVE: High-attention sub-groups are not more significant")
        print("  than controls — attention may reflect arbitrary convergence.")


if __name__ == '__main__':
    main()