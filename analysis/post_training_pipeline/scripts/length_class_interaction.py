#!/usr/bin/env python3
"""
Length-Class Interaction Analysis for Hard Cases

Works directly with all_sample_predictions.csv or hard_cases.csv
Expected columns: transcript_id, true_label, consensus_prediction, sequence_length
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100  # Display only, savefig uses 1200


def load_predictions_csv(predictions_csv):
    """Load predictions CSV with sequence_length column"""
    print(f"Loading predictions from {predictions_csv}...")
    df = pd.read_csv(predictions_csv)
    
    # Validate required columns
    required = ['transcript_id', 'true_label', 'consensus_prediction', 'sequence_length']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Compute is_hard_case if not present
    if 'is_hard_case' not in df.columns:
        if 'is_correct' in df.columns:
            df['is_hard_case'] = ~df['is_correct']
        else:
            df['is_hard_case'] = (df['true_label'] != df['consensus_prediction'])
    
    df['length'] = df['sequence_length']
    
    print(f"  Loaded {len(df):,} transcripts")
    print(f"  Hard cases: {df['is_hard_case'].sum():,} ({100*df['is_hard_case'].mean():.2f}%)")
    print(f"  lncRNAs: {(df['true_label'] == 'lnc').sum():,}")
    print(f"  Protein-coding: {(df['true_label'] == 'pc').sum():,}")
    
    return df


def analyze_length_class_interaction(df):
    """Analyze hard case rates by length bin and class"""
    # Create length bins
    bins = [0, 500, 1000, 2000, 5000, np.inf]
    labels = ['<500', '500-1k', '1-2k', '2-5k', '>5k']
    df['length_bin'] = pd.cut(df['length'], bins=bins, labels=labels, include_lowest=True)
    
    results = []
    
    for length_bin in labels:
        bin_df = df[df['length_bin'] == length_bin]
        if len(bin_df) == 0:
            continue
        
        for class_label in ['lnc', 'pc']:
            class_name = 'lncRNA' if class_label == 'lnc' else 'protein_coding'
            class_df = bin_df[bin_df['true_label'] == class_label]
            
            if len(class_df) == 0:
                continue
            
            n_total = len(class_df)
            n_hard = class_df['is_hard_case'].sum()
            hard_rate = n_hard / n_total
            
            results.append({
                'length_bin': length_bin,
                'class': class_name,
                'n_total': n_total,
                'n_hard': n_hard,
                'hard_rate': hard_rate,
                'hard_rate_pct': hard_rate * 100
            })
    
    return pd.DataFrame(results)


def compute_relative_difficulty(results_df):
    """Compute PC/lncRNA ratio (>1 = PCs harder, <1 = lncRNAs harder)"""
    relative_diff = []
    
    for length_bin in results_df['length_bin'].unique():
        bin_df = results_df[results_df['length_bin'] == length_bin]
        
        lnc_rate = bin_df[bin_df['class'] == 'lncRNA']['hard_rate'].values
        pc_rate = bin_df[bin_df['class'] == 'protein_coding']['hard_rate'].values
        
        if len(lnc_rate) > 0 and len(pc_rate) > 0 and lnc_rate[0] > 0:
            ratio = pc_rate[0] / lnc_rate[0]
            relative_diff.append({
                'length_bin': length_bin,
                'ratio': ratio,
                'lnc_rate': lnc_rate[0],
                'pc_rate': pc_rate[0]
            })
    
    return pd.DataFrame(relative_diff)


def create_dual_panel_figure(results_df, relative_diff_df, output_path):
    """Create dual-panel figure"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define proper length bin order
    length_order = ['<500', '500-1k', '1-2k', '2-5k', '>5k']
    
    # Panel 1: Hard Case Rates
    ax = axes[0]
    
    # Sort by length bin order, not alphabetically
    lnc_data = results_df[results_df['class'] == 'lncRNA'].copy()
    pc_data = results_df[results_df['class'] == 'protein_coding'].copy()
    
    # Create categorical with proper order
    lnc_data['length_bin'] = pd.Categorical(lnc_data['length_bin'], categories=length_order, ordered=True)
    pc_data['length_bin'] = pd.Categorical(pc_data['length_bin'], categories=length_order, ordered=True)
    
    lnc_data = lnc_data.sort_values('length_bin')
    pc_data = pc_data.sort_values('length_bin')
    
    x_pos = np.arange(len(lnc_data))
    
    ax.plot(x_pos, lnc_data['hard_rate_pct'], 'o-', color='#E74C3C', 
           linewidth=3, markersize=10, label='lncRNA', alpha=0.8)
    ax.plot(x_pos, pc_data['hard_rate_pct'], 's-', color='#3498DB', 
           linewidth=3, markersize=10, label='Protein-coding', alpha=0.8)
    
    # Highlight balanced zone (1-2k is at index 2 in proper order)
    balanced_idx = length_order.index('1-2k') if '1-2k' in length_order else None
    if balanced_idx is not None and balanced_idx < len(x_pos):
        ax.axvspan(balanced_idx - 0.5, balanced_idx + 0.5, 
                  color='yellow', alpha=0.2, zorder=0)
        y_max = max(results_df['hard_rate_pct']) if len(results_df) > 0 else 20
        ax.text(balanced_idx, y_max * 0.95, 
               'Balanced difficulty zone', 
               ha='center', fontsize=10, style='italic', 
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Annotations
    if len(lnc_data) >= 5:
        # Long lncRNAs (>5k is last in order)
        long_lnc_rate = lnc_data.iloc[-1]['hard_rate_pct']
        ax.annotate(f'Long lncRNAs\ndifficult ({long_lnc_rate:.1f}%)',
                   xy=(len(lnc_data)-1, long_lnc_rate),
                   xytext=(len(lnc_data)-1.5, long_lnc_rate + 2),
                   ha='right', fontsize=10, color='#E74C3C', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2))
    
    if len(pc_data) >= 1:
        # Short PCs (<500 is first in order)
        short_pc_rate = pc_data.iloc[0]['hard_rate_pct']
        ax.annotate(f'Short PCs\ndifficult ({short_pc_rate:.1f}%)',
                   xy=(0, short_pc_rate),
                   xytext=(0.5, short_pc_rate + 2),
                   ha='left', fontsize=10, color='#3498DB', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='#3498DB', lw=2))
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(lnc_data['length_bin'], fontsize=11)
    ax.set_xlabel('Transcript Length (bp)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Hard Case Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Length-Class Interaction: Hard Case Rates\n(Opposite effects for lncRNA vs PC)',
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    ax.set_ylim(0, max(results_df['hard_rate_pct']) * 1.1)
    
    # Panel 2: Relative Difficulty
    ax = axes[1]
    
    # Sort relative_diff by length order too
    relative_diff_sorted = relative_diff_df.copy()
    relative_diff_sorted['length_bin'] = pd.Categorical(
        relative_diff_sorted['length_bin'], 
        categories=length_order, 
        ordered=True
    )
    relative_diff_sorted = relative_diff_sorted.sort_values('length_bin')
    
    x_pos_ratio = np.arange(len(relative_diff_sorted))
    colors = ['#3498DB' if r > 1 else '#E74C3C' for r in relative_diff_sorted['ratio']]
    
    bars = ax.bar(x_pos_ratio, relative_diff_sorted['ratio'], 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.6)
    ax.text(len(relative_diff_sorted) * 0.5, 1.05, 'Equal difficulty', 
           ha='center', fontsize=10, style='italic')
    
    for i, (idx, row) in enumerate(relative_diff_sorted.iterrows()):
        ratio = row['ratio']
        y_pos = ratio + max(relative_diff_sorted['ratio']) * 0.03
        ax.text(i, y_pos, f'{ratio:.1f}×', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_xticks(x_pos_ratio)
    ax.set_xticklabels(relative_diff_sorted['length_bin'], fontsize=11)
    ax.set_xlabel('Transcript Length (bp)', fontsize=13, fontweight='bold')
    ax.set_ylabel('PC / lncRNA Hard Rate Ratio', fontsize=13, fontweight='bold')
    ax.set_title('Relative Difficulty by Length\n(>1 = PCs harder, <1 = lncRNAs harder)',
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1, axis='y')
    ax.set_ylim(0, max(relative_diff_sorted['ratio']) * 1.15)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498DB', alpha=0.8, label='PCs harder'),
        Patch(facecolor='#E74C3C', alpha=0.8, label='lncRNAs harder')
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=1200, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Figure saved: {output_path}")


def print_summary(results_df, relative_diff_df):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("LENGTH-CLASS INTERACTION ANALYSIS")
    print("="*80)
    
    print("\nHard Case Rates by Length and Class:")
    print("-" * 80)
    print(f"{'Length':<12} {'lncRNA':<15} {'PC':<15} {'Ratio':<15} {'Samples (lnc/PC)'}")
    print("-" * 80)
    
    for length_bin in results_df['length_bin'].unique():
        bin_df = results_df[results_df['length_bin'] == length_bin]
        
        lnc = bin_df[bin_df['class'] == 'lncRNA']
        pc = bin_df[bin_df['class'] == 'protein_coding']
        
        lnc_rate = lnc['hard_rate_pct'].values[0] if len(lnc) > 0 else 0
        pc_rate = pc['hard_rate_pct'].values[0] if len(pc) > 0 else 0
        lnc_n = lnc['n_total'].values[0] if len(lnc) > 0 else 0
        pc_n = pc['n_total'].values[0] if len(pc) > 0 else 0
        
        ratio_row = relative_diff_df[relative_diff_df['length_bin'] == length_bin]
        ratio = ratio_row['ratio'].values[0] if len(ratio_row) > 0 else 0
        
        print(f"{length_bin:<12} {lnc_rate:>6.1f}%          {pc_rate:>6.1f}%          {ratio:>6.1f}×        {lnc_n:>6,}/{pc_n:<6,}")
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    lnc_df = results_df[results_df['class'] == 'lncRNA']
    pc_df = results_df[results_df['class'] == 'protein_coding']
    
    hardest_lnc = lnc_df.loc[lnc_df['hard_rate_pct'].idxmax()]
    hardest_pc = pc_df.loc[pc_df['hard_rate_pct'].idxmax()]
    
    print(f"\nHardest lncRNAs: {hardest_lnc['length_bin']} ({hardest_lnc['hard_rate_pct']:.1f}% hard)")
    print(f"Hardest PCs: {hardest_pc['length_bin']} ({hardest_pc['hard_rate_pct']:.1f}% hard)")
    
    max_pc_harder = relative_diff_df.loc[relative_diff_df['ratio'].idxmax()]
    max_lnc_harder = relative_diff_df.loc[relative_diff_df['ratio'].idxmin()]
    
    print(f"\nGreatest PC difficulty: {max_pc_harder['length_bin']} ({max_pc_harder['ratio']:.1f}× harder than lncRNAs)")
    print(f"Greatest lncRNA difficulty: {max_lnc_harder['length_bin']} ({1/max_lnc_harder['ratio']:.1f}× harder than PCs)")
    
    balanced = relative_diff_df[relative_diff_df['ratio'].between(0.8, 1.2)]
    if len(balanced) > 0:
        print(f"\nBalanced difficulty zone: {', '.join(balanced['length_bin'])}")


def main():
    parser = argparse.ArgumentParser(description='Length-class interaction analysis')
    parser.add_argument('--predictions_csv', required=True,
                       help='Path to all_sample_predictions.csv (with sequence_length column)')
    parser.add_argument('--output_dir', default='length_class_analysis',
                       help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("LENGTH-CLASS INTERACTION ANALYSIS")
    print("="*80)
    
    # Load data
    df = load_predictions_csv(args.predictions_csv)
    
    # Analyze
    print("\nAnalyzing length-class interaction...")
    results_df = analyze_length_class_interaction(df)
    results_df.to_csv(output_dir / 'length_class_hard_rates.csv', index=False)
    print(f"    Saved: length_class_hard_rates.csv")
    
    # Compute relative difficulty
    relative_diff_df = compute_relative_difficulty(results_df)
    relative_diff_df.to_csv(output_dir / 'relative_difficulty_by_length.csv', index=False)
    print(f"    Saved: relative_difficulty_by_length.csv")
    
    # Create figure
    print("\nCreating visualization...")
    fig_path = output_dir / 'length_class_interaction_figure.png'
    create_dual_panel_figure(results_df, relative_diff_df, fig_path)
    
    # Print summary
    print_summary(results_df, relative_diff_df)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()