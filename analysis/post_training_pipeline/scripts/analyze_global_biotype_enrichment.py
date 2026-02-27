#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global Biotype Enrichment Analysis

Computes biotype enrichment against the global baseline hard case rate,
answering: "Which biotypes are more/less likely to be misclassified?"

"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.5


def compute_global_enrichment(spatial_dir, min_biotype_count=10):
    """
    Compute global biotype enrichment against baseline hard case rate
    
    Returns DataFrame with enrichment statistics
    """
    print("Loading sample data from all folds...")
    
    # Load all samples from all folds
    all_samples = []
    fold_dirs = sorted([d for d in Path(spatial_dir).iterdir() 
                       if d.is_dir() and d.name.startswith('fold_')])
    
    for fold_dir in fold_dirs:
        samples_csv = fold_dir / 'samples_with_regions.csv'
        if samples_csv.exists():
            df = pd.read_csv(samples_csv)
            all_samples.append(df)
    
    if not all_samples:
        raise ValueError("No sample data found!")
    
    # Combine all samples (each fold has different validation samples)
    combined = pd.concat(all_samples, ignore_index=True)
    
    # Count unique transcripts
    if 'transcript_id' in combined.columns:
        # Deduplicate by transcript_id (in case same transcript in multiple folds)
        combined = combined.drop_duplicates(subset='transcript_id')

    # Filter unknown biotype before any computation
    n_before = len(combined)
    combined = combined[combined['biotype'] != 'unknown'].copy()
    n_filtered = n_before - len(combined)
    if n_filtered > 0:
        print(f"  Filtered {n_filtered:,} samples with unknown biotype")
    
    print(f"  Total unique samples: {len(combined):,}")
    
    print(f"  Hard cases: {combined['is_hard_case'].sum():,}")
    
    # Global baseline hard case rate
    global_hard_rate = combined['is_hard_case'].mean()
    print(f"  Global hard rate: {global_hard_rate:.2%}")
    
    # Compute per-biotype statistics
    results = []
    
    for biotype in sorted(combined['biotype'].unique()):
        biotype_df = combined[combined['biotype'] == biotype]
        
        n_total = len(biotype_df)
        n_hard = biotype_df['is_hard_case'].sum()
        
        if n_total < min_biotype_count:
            continue
        
        hard_rate = n_hard / n_total
        
        # Enrichment vs global baseline
        fold_enrichment = hard_rate / global_hard_rate if global_hard_rate > 0 else 1.0
        log2_enrichment = np.log2(fold_enrichment) if fold_enrichment > 0 else 0
        
        # Statistical test: binomial test
        # H0: This biotype has the same hard rate as global
        # HA: This biotype has different hard rate
        
        # Use two-sided binomial test
        p_value = stats.binomtest(
            k=n_hard,
            n=n_total, 
            p=global_hard_rate,
            alternative='two-sided'
        ).pvalue
        
        # Direction
        direction = 'enriched' if fold_enrichment > 1 else 'depleted'
        
        results.append({
            'biotype': biotype,
            'n_total': n_total,
            'n_hard': n_hard,
            'hard_rate': hard_rate,
            'baseline_rate': global_hard_rate,
            'fold_enrichment': fold_enrichment,
            'log2_enrichment': log2_enrichment,
            'p_value': p_value,
            'direction': direction
        })
    
    results_df = pd.DataFrame(results)
    
    # FDR correction (Benjamini-Hochberg)
    from scipy.stats import false_discovery_control
    results_df['fdr'] = false_discovery_control(results_df['p_value'].values)
    
    # Significant at FDR < 0.05
    results_df['significant'] = results_df['fdr'] < 0.05
    
    # Sort by fold enrichment
    results_df = results_df.sort_values('fold_enrichment', ascending=False)
    
    return results_df


def create_enrichment_figure(enrichment_df, output_path, model_label='', gencode_version=''):
    """Create publication-quality horizontal bar chart"""
    tag_parts = [p for p in [model_label,
                 f'GENCODE {gencode_version}' if gencode_version else ''] if p]
    fig_tag = ' | '.join(tag_parts) + ' — ' if tag_parts else ''
    
    # Filter to significant only
    sig_df = enrichment_df[enrichment_df['significant']].copy()
    
    if len(sig_df) == 0:
        print("No significant biotypes found!")
        return
    
    print(f"\nFound {len(sig_df)} significant biotypes (FDR < 0.05)")
    
    # Sort for plotting (ascending for horizontal bars)
    sig_df = sig_df.sort_values('fold_enrichment', ascending=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, max(4, len(sig_df) * 0.5)))
    
    # Colors: enriched (red) vs depleted (blue)
    colors = ['#E63946' if x > 1.0 else '#457B9D' for x in sig_df['fold_enrichment']]
    
    # Horizontal bars
    bars = ax.barh(
        range(len(sig_df)),
        sig_df['fold_enrichment'],
        color=colors,
        alpha=0.85,
        edgecolor='black',
        linewidth=1.2
    )
    
    # Reference line at 1.0
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, zorder=0)
    
    # Add value labels
    for i, (idx, row) in enumerate(sig_df.iterrows()):
        value = row['fold_enrichment']
        
        if value < 1.5:
            x_pos = value + 0.1
            ha = 'left'
            color = 'black'
        else:
            x_pos = value - 0.1
            ha = 'right'
            color = 'white'
            
        ax.text(x_pos, i, f"{value:.2f}×", va='center', ha=ha,
               fontsize=11, fontweight='bold', color=color)
    
    # Y-axis: biotype names
    ax.set_yticks(range(len(sig_df)))
    ax.set_yticklabels(sig_df['biotype'], fontsize=11)
    
    # X-axis
    ax.set_xlabel('Fold Enrichment in Hard Cases (vs Baseline)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, max(sig_df['fold_enrichment']) * 1.15)
    
    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle=':', linewidth=1)
    ax.set_axisbelow(True)
    
    # Title
    ax.set_title(f'{fig_tag}Biotype Enrichment in Misclassified Cases', 
                 fontsize=13, fontweight='bold', pad=15)
    
    # Sample size annotations
    max_x = max(sig_df['fold_enrichment']) * 1.12
    for i, (idx, row) in enumerate(sig_df.iterrows()):
        ax.text(max_x, i, f"n={row['n_hard']}/{row['n_total']}",
               va='center', ha='right', fontsize=8, color='gray')
    
    # Significance note
    fig.text(0.99, 0.01, '*** FDR < 0.05 (Binomial test)', 
             ha='right', va='bottom', fontsize=9, style='italic')
    
    # Clean up
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=1200, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def print_summary(enrichment_df):
    """Print summary table"""
    
    print("\n" + "=" * 100)
    print("GLOBAL BIOTYPE ENRICHMENT")
    print("=" * 100)
    
    print(f"\n{'Biotype':<30} {'n_hard/n_total':<18} {'Hard Rate':<12} {'Baseline':<12} "
          f"{'Fold Enr.':<12} {'FDR':<12} {'Sig'}")
    print("-" * 100)
    
    for _, row in enrichment_df.iterrows():
        sig_marker = '***' if row['significant'] else ''
        counts = f"{row['n_hard']:,}/{row['n_total']:,}"
        
        print(f"{row['biotype']:<30} {counts:<18} {row['hard_rate']:>10.1%}  "
              f"{row['baseline_rate']:>10.1%}  {row['fold_enrichment']:>10.2f}×  "
              f"{row['fdr']:>10.2e}  {sig_marker}")


def main():
    parser = argparse.ArgumentParser(
        description='Global biotype enrichment analysis (vs baseline hard rate)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--spatial_dir', required=True,
                       help='Directory containing fold_N/samples_with_regions.csv')
    parser.add_argument('--output_dir', default='global_biotype_enrichment',
                       help='Output directory')
    parser.add_argument('--min_count', type=int, default=10,
                       help='Minimum biotype count to include (default: 10)')
    parser.add_argument('--model_label', default='',
                        help='Model variant label')
    parser.add_argument('--gencode_version', default='',
                        help='GENCODE version, e.g. "v47"')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 80)
    print("GLOBAL BIOTYPE ENRICHMENT ANALYSIS")
    print("=" * 80)
    print("\nComparing each biotype's hard case rate to global baseline")
    print("Question: Which biotypes are more/less likely to be misclassified?")
    print("=" * 80)
    
    # Compute enrichment
    enrichment_df = compute_global_enrichment(args.spatial_dir, args.min_count)
    
    # Save results
    output_csv = output_dir / 'global_biotype_enrichment.csv'
    enrichment_df.to_csv(output_csv, index=False)
    print(f"\n  Saved: {output_csv}")
    
    # Print summary
    print_summary(enrichment_df)
    
    # Create figure
    output_fig = output_dir / 'global_biotype_enrichment.png'
    create_enrichment_figure(enrichment_df, output_fig, args.model_label, args.gencode_version)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - global_biotype_enrichment.csv")
    print(f"  - global_biotype_enrichment.png")


if __name__ == '__main__':
    main()