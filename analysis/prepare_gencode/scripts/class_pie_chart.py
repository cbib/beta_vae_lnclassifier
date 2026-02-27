#!/usr/bin/env python3
"""
Create plots for GENCODE biotype distribution
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import re

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']


def create_class_distribution(
    biotype_csv: str,
    output_path: str,
    title: str = "GENCODE v49 Dataset"
):
    """
    Create a simple, clean visualization focusing on the main classes
    """
    df = pd.read_csv(biotype_csv)
    
    # Check if we have gene_biotype column
    if 'gene_biotype' in df.columns:
        class_col = 'gene_biotype'
    else:
        class_col = 'biotype'
    
    # Get main classes
    lnc_count = (df[class_col] == 'lncRNA').sum()
    pc_count = (df[class_col] == 'protein_coding').sum()
    total = len(df)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.6)
    
    # =========================================================================
    # LEFT: Simple pie chart for main classes
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    counts = [lnc_count, pc_count]
    labels = ['lncRNA', 'Protein-coding']
    colors = ['#E74C3C', '#3498DB']
    
    wedges, texts, autotexts = ax1.pie(
        counts,
        labels=labels,
        colors=colors,
        autopct='',
        startangle=90,
        textprops={'fontsize': 14, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 3}
    )
    
    # Add custom percentage labels inside
    for i, (wedge, count) in enumerate(zip(wedges, counts)):
        angle = (wedge.theta2 - wedge.theta1) / 2 + wedge.theta1
        x = 0.7 * np.cos(np.radians(angle))
        y = 0.7 * np.sin(np.radians(angle))
        
        pct = 100 * count / total
        ax1.text(x, y, f'{pct:.1f}%\n({count:,})',
                ha='center', va='center',
                fontsize=13, fontweight='bold', color='white')
    
    # Add ratio text below
    ratio = lnc_count / pc_count
    ax1.text(0, -1.4, f'Ratio (lnc:pc) = {ratio:.3f}:1',
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax1.set_title(f'{title}\nMain Classes', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # =========================================================================
    # RIGHT: Horizontal bar chart for transcript biotypes
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Get top 8 transcript biotypes
    biotype_counts = df['biotype'].value_counts().head(8)
    
    # Assign colors based on category
    def get_biotype_color(biotype):
        biotype_lower = biotype.lower()
        if biotype == 'lncRNA':
            return '#E74C3C'
        elif 'protein_coding' in biotype_lower:
            return '#3498DB'
        elif 'retained_intron' in biotype_lower:
            return '#E67E22'
        elif 'nonsense_mediated_decay' in biotype_lower or 'nmd' in biotype_lower:
            return '#9B59B6'
        elif 'non_stop_decay' in biotype_lower:
            return '#8E44AD'
        else:
            return '#95A5A6'
    
    colors = [get_biotype_color(bt) for bt in biotype_counts.index]
    
    # Create horizontal bars
    y_pos = np.arange(len(biotype_counts))
    bars = ax2.barh(y_pos, biotype_counts.values, color=colors, 
                    edgecolor='black', linewidth=1, alpha=0.8)
    
    # Customize
    ax2.set_yticks(y_pos)
    
    # Create clean labels
    labels_clean = []
    for biotype in biotype_counts.index:
        if len(biotype) <= 30:
            labels_clean.append(biotype)
        else:
            # Abbreviate long names
            labels_clean.append(biotype.replace('protein_coding_', 'PC_'))
    
    ax2.set_yticklabels(labels_clean, fontsize=11)
    ax2.set_xlabel('Number of Transcripts', fontsize=12, fontweight='bold')
    ax2.set_title('Transcript Biotype Breakdown', fontsize=16, fontweight='bold', pad=20)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, biotype_counts.values)):
        pct = 100 * count / total
        ax2.text(count + max(biotype_counts.values) * 0.01, i,
                f' {count:,} ({pct:.1f}%)',
                va='center', fontsize=10, fontweight='bold')
    
    # Add total count in corner
    ax2.text(0.98, 0.02, f'Total: {total:,} transcripts',
            transform=ax2.transAxes,
            ha='right', va='bottom',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', 
                     edgecolor='black', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=1200, bbox_inches='tight')
    print(f"  Saved clean visualization to: {output_path}")
    plt.close()


def create_stacked_bar_comparison(
    v47_csv: str,
    v49_csv: str,
    output_path: str
):
    """
    Create side-by-side stacked bar chart comparing v47 and v49
    """
    df_v47 = pd.read_csv(v47_csv)
    df_v49 = pd.read_csv(v49_csv)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get counts for both versions
    versions = ['GENCODE v47', 'GENCODE v49']
    
    data = []
    for df in [df_v47, df_v49]:
        class_col = 'gene_biotype' if 'gene_biotype' in df.columns else 'biotype'
        lnc = (df[class_col] == 'lncRNA').sum()
        pc = (df[class_col] == 'protein_coding').sum()
        total = len(df)
        data.append({
            'lnc': lnc,
            'pc': pc,
            'total': total,
            'lnc_pct': 100 * lnc / total,
            'pc_pct': 100 * pc / total,
            'ratio': lnc / pc
        })
    
    # Create grouped bars
    x = np.arange(len(versions))
    width = 0.35
    
    # Plot absolute counts
    lnc_counts = [d['lnc'] for d in data]
    pc_counts = [d['pc'] for d in data]
    
    bars1 = ax.bar(x - width/2, lnc_counts, width, label='lncRNA',
                   color='#E74C3C', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, pc_counts, width, label='Protein-coding',
                   color='#3498DB', edgecolor='black', linewidth=1.5)
    
    # Customize
    ax.set_ylabel('Number of Transcripts', fontsize=12, fontweight='bold')
    ax.set_title('GENCODE Version Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(versions, fontsize=12, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    def add_labels(bars, values):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{val:,}',
                   ha='center', va='bottom',
                   fontsize=10, fontweight='bold')
    
    add_labels(bars1, lnc_counts)
    add_labels(bars2, pc_counts)
    
    # Add ratio text below each version
    for i, d in enumerate(data):
        ax.text(i, -max(lnc_counts + pc_counts) * 0.15,
               f"Ratio: {d['ratio']:.3f}:1\nTotal: {d['total']:,}",
               ha='center', va='top',
               fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=1200, bbox_inches='tight')
    print(f"  Saved comparison plot to: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Create clean visualizations for GENCODE biotype distribution'
    )
    parser.add_argument('--biotype_csv', required=True,
                       help='Path to biotype CSV file')
    parser.add_argument('--output_dir', default='.',
                       help='Output directory for plots')
    parser.add_argument('--title', default='GENCODE v49 Dataset',
                       help='Title for visualizations')
    parser.add_argument('--compare_with', default=None,
                       help='Path to v47 CSV for comparison')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get gencode version through biotype_csv parsing
    gencode_version = r'g(\d+)' 
    match = re.search(gencode_version, args.biotype_csv)[0]
    
    print("=" * 80)
    print("Creating Plots")
    print("=" * 80)
    
    # Main visualization
    main_output = output_dir / f'{match}_dataset_distribution.png'
    create_class_distribution(args.biotype_csv, main_output, args.title)
    
    # Comparison if requested
    if args.compare_with:
        comparing_version = r'g(\d+)'
        match_compare = re.search(comparing_version, args.compare_with)[0]
        comparison_output = output_dir / f'{match}_vs_{match_compare}_version_comparison.png'
        create_stacked_bar_comparison(args.compare_with, args.biotype_csv, comparison_output)
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  - {main_output.name}")
    if args.compare_with:
        print(f"  - {comparison_output.name}")


if __name__ == '__main__':
    main()