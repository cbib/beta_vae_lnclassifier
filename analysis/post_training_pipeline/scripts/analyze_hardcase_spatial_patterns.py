#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial region analysis for beta-VAE hard cases

Automatically processes all folds in a UMAP visualization directory.
Identifies spatial regions of hard cases using K-means clustering on UMAP coordinates,
then analyzes biotype composition and class balance in each region.

Processes ALL folds in a directory automatically.

Usage:
    # Process all folds in the UMAP directory
    python analyze_hardcase_spatial_patterns.py \
        --umap_dir umap_visualizations_with_RI \
        --n_regions 5 \
        --output_dir spatial_analysis
    
    # The script will automatically find and process:
    #   umap_dir/fold_0/umap_embeddings.csv
    #   umap_dir/fold_1/umap_embeddings.csv
    #   ...
    
    # Creates per-fold outputs plus cross-fold summary:
    #   spatial_analysis/fold_0/...
    #   spatial_analysis/fold_1/...
    #   spatial_analysis/cross_fold_summary.png
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100  # Display only, savefig uses 1200


def load_and_validate_data(umap_csv):
    """Load UMAP CSV and validate it has required columns"""
    
    print(f"\nLoading data from: {umap_csv}")
    df = pd.read_csv(umap_csv)
    n_before = len(df)
    df = df[df['biotype'] != 'unknown'].copy()
    if len(df) < n_before:
        print(f"  Filtered {n_before - len(df):,} unknown biotype samples")
    
    print(f"  Loaded {len(df):,} samples")
    print(f"  Columns: {list(df.columns)}")
    
    # Check required columns
    required_cols = ['UMAP1', 'UMAP2', 'is_hard_case', 'biotype', 'true_label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Validate true_label format
    if df['true_label'].dtype == 'object':
        # Convert 'lnc'/'pc' to 0/1 for consistency
        print("  Converting true_label from strings to numeric (lnc=0, pc=1)")
        df['true_label_numeric'] = df['true_label'].map({'lnc': 0, 'pc': 1, 'lncRNA': 0, 'protein_coding': 1})
    else:
        df['true_label_numeric'] = df['true_label']
    
    # Check biotype quality
    unique_biotypes = df['biotype'].unique()
    if all(bt.startswith('biotype_') or bt == 'unknown' for bt in unique_biotypes):
        print("    WARNING: Biotypes are generic labels (biotype_0, biotype_1, etc.)")
        print("    Consider running fix_embedding_biotypes.py first for real biotype names")
    else:
        print(f"   Real biotype names detected")
        print(f"    Top biotypes: {list(unique_biotypes[:5])}")
    
    # Report hard cases
    n_hard = df['is_hard_case'].sum()
    print(f"\n  Hard cases: {n_hard:,} ({100*n_hard/len(df):.2f}%)")
    
    return df


def identify_spatial_regions(df, n_regions=5):
    """
    Identify spatial regions using K-means on UMAP coordinates of hard cases
    
    Returns:
        df: DataFrame with 'spatial_region' column added
        stats_df: DataFrame with region statistics
    """
    print(f"\n{'='*80}")
    print(f"IDENTIFYING {n_regions} SPATIAL REGIONS")
    print(f"{'='*80}")
    
    # Focus on hard cases only
    hard_mask = df['is_hard_case']
    hard_coords = df.loc[hard_mask, ['UMAP1', 'UMAP2']].values
    
    print(f"\nClustering {len(hard_coords):,} hard cases...")
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_regions, random_state=42, n_init=20)
    region_labels = kmeans.fit_predict(hard_coords)
    
    # Assign regions to full dataset (-1 for easy cases)
    full_regions = np.full(len(df), -1, dtype=int)
    full_regions[hard_mask] = region_labels
    
    df['spatial_region'] = full_regions
    
    # Compute region statistics
    print("\nComputing region statistics...")
    region_stats = []
    
    for region_id in range(n_regions):
        region_mask = df['spatial_region'] == region_id
        region_df = df[region_mask]
        
        # Spatial characteristics
        umap1_mean = region_df['UMAP1'].mean()
        umap2_mean = region_df['UMAP2'].mean()
        umap1_std = region_df['UMAP1'].std()
        umap2_std = region_df['UMAP2'].std()
        
        # Class distribution
        if region_df['true_label_numeric'].isna().any():
            print(f"  WARNING: {region_df['true_label_numeric'].isna().sum()} samples have invalid true_label values")
            print(f"  Unique values: {region_df['true_label'].unique()}")

        lnc_count = (region_df['true_label_numeric'] == 0).sum()
        pc_count = (region_df['true_label_numeric'] == 1).sum()
        lnc_pct = 100 * lnc_count / len(region_df) if len(region_df) > 0 else 0
        pc_pct = 100 * pc_count / len(region_df) if len(region_df) > 0 else 0
        
        # Classify region position based on TRUE LABELS
        if pc_pct > 70:
            position = "pc_dominated"
        elif lnc_pct > 70:
            position = "lnc_dominated"
        elif abs(lnc_pct - pc_pct) < 20:
            position = "frontier"
        elif pc_pct > lnc_pct:
            position = "pc_majority"
        else:
            position = "lnc_majority"
        
        # Biotype composition
        biotype_counts = region_df['biotype'].value_counts()
        top_biotypes = biotype_counts.head(3)
        
        # Special tracking for interesting biotypes
        retained_intron_count = biotype_counts.get('retained_intron', 0)
        retained_intron_pct = 100 * retained_intron_count / len(region_df) if len(region_df) > 0 else 0
        
        nmd_count = biotype_counts.get('nonsense_mediated_decay', 0)
        nmd_pct = 100 * nmd_count / len(region_df) if len(region_df) > 0 else 0
        
        region_stats.append({
            'region_id': region_id,
            'n_samples': len(region_df),
            'umap1_center': umap1_mean,
            'umap2_center': umap2_mean,
            'umap1_spread': umap1_std,
            'umap2_spread': umap2_std,
            'position_label': position,
            'top_biotype': top_biotypes.index[0] if len(top_biotypes) > 0 else 'unknown',
            'top_biotype_count': top_biotypes.iloc[0] if len(top_biotypes) > 0 else 0,
            'top_biotype_pct': 100 * top_biotypes.iloc[0] / len(region_df) if len(top_biotypes) > 0 else 0,
            'n_lnc': int(lnc_count),
            'n_pc': int(pc_count),
            'pct_lnc': lnc_pct,
            'pct_pc': pc_pct,
            'n_retained_intron': int(retained_intron_count),
            'pct_retained_intron': retained_intron_pct,
            'n_nmd': int(nmd_count),
            'pct_nmd': nmd_pct
        })
    
    stats_df = pd.DataFrame(region_stats)
    
    # Sort by spatial position (left to right)
    stats_df = stats_df.sort_values('umap1_center')
    
    print("\n" + "="*80)
    print("REGION SUMMARY")
    print("="*80)
    print(stats_df[['region_id', 'n_samples', 'position_label', 
                    'pct_lnc', 'pct_pc', 'top_biotype']].to_string(index=False))
    
    return df, stats_df


def compare_spatial_regions(df, stats_df, output_dir, fig_tag=None):
    """Compare biotype composition across spatial regions"""
    
    print("\n" + "=" * 80)
    print("BIOTYPE COMPOSITION ANALYSIS")
    print("=" * 80)
    
    # Get hard cases by region
    hard_cases = df[df['is_hard_case'] & (df['spatial_region'] >= 0)]
    
    if len(hard_cases) == 0:
        print("No hard cases found!")
        return
    
    # Identify important biotypes (those that appear in >1% of hard cases)
    biotype_global_counts = hard_cases['biotype'].value_counts()
    biotype_global_pct = 100 * biotype_global_counts / len(hard_cases)
    important_biotypes = biotype_global_pct[biotype_global_pct > 1.0].index.tolist()
    
    print(f"\nTracking {len(important_biotypes)} biotypes (>1% of hard cases):")
    print(f"  {important_biotypes[:10]}")
    
    # Create biotype x region matrix
    regions = sorted(hard_cases['spatial_region'].unique())
    
    region_biotype_matrix = []
    
    for region_id in regions:
        region_mask = hard_cases['spatial_region'] == region_id
        region_df = hard_cases[region_mask]
        
        row = {'region_id': region_id, 'n_samples': len(region_df)}
        
        for biotype in important_biotypes:
            count = (region_df['biotype'] == biotype).sum()
            pct = 100 * count / len(region_df)
            row[f'{biotype}_count'] = count
            row[f'{biotype}_pct'] = pct
        
        region_biotype_matrix.append(row)
    
    matrix_df = pd.DataFrame(region_biotype_matrix)
    
    # Merge with position labels
    matrix_df = matrix_df.merge(
        stats_df[['region_id', 'position_label', 'umap1_center', 'pct_lnc', 'pct_pc']],
        on='region_id',
        how='left'
    ).sort_values('umap1_center')
    
    print("\nBiotype composition by spatial region:")
    
    # Display key biotypes
    display_cols = ['region_id', 'position_label', 'n_samples', 'pct_lnc', 'pct_pc']
    for biotype in important_biotypes[:6]:  # Top 6
        display_cols.append(f'{biotype}_pct')
    
    print(matrix_df[display_cols].to_string(index=False, float_format='%.1f'))
    
    # Save full matrix
    matrix_df.to_csv(output_dir / 'region_biotype_composition.csv', index=False)
    print(f"\n  Saved: {output_dir / 'region_biotype_composition.csv'}")
    
    # Visualization: Heatmap of top biotypes
    print("\nGenerating biotype composition heatmap...")
    
    # Select top biotypes for visualization (max 8)
    plot_biotypes = important_biotypes[:min(8, len(important_biotypes))]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data
    matrix_df_sorted = matrix_df.sort_values('region_id')  # Sort by region_id, not umap1_center
    plot_data = matrix_df_sorted[[f'{bt}_pct' for bt in plot_biotypes]].values
    region_labels = [f"R{int(r)}\n({matrix_df_sorted.iloc[i]['position_label'][:12]})" 
                    for i, r in enumerate(matrix_df_sorted['region_id'])]
    
    # Create heatmap
    im = ax.imshow(plot_data.T, cmap='YlOrRd', aspect='auto', vmin=0)
    
    ax.set_xticks(range(len(region_labels)))
    ax.set_xticklabels(region_labels, rotation=0, fontsize=10)
    ax.set_yticks(range(len(plot_biotypes)))
    ax.set_yticklabels(plot_biotypes, fontsize=11)
    
    # Add percentage values
    for i in range(len(plot_biotypes)):
        for j in range(len(region_labels)):
            value = plot_data[j, i]
            color = 'white' if value > 50 else 'black'
            ax.text(j, i, f'{value:.1f}%',
                   ha="center", va="center", color=color, fontsize=9, fontweight='bold')
    
    ax.set_title(f'{fig_tag}Biotype Composition Across Spatial Regions\n(% of hard cases in each region)', 
                fontsize=14, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(im, ax=ax, label='% of region')
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'region_biotype_heatmap.png', dpi=350, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_dir / 'region_biotype_heatmap.png'}")


def visualize_regions(df, stats_df, output_dir, fig_tag=None):
    """Visualize spatial regions with class balance"""
    
    print("\nGenerating spatial regions visualization...")
    
    fig = plt.figure(figsize=(18, 10))
    
    # Create grid: main plot (70%) + bar chart (30%)
    gs = fig.add_gridspec(10, 10, hspace=0.3, wspace=0.3)
    ax_main = fig.add_subplot(gs[:, :7])
    ax_bars = fig.add_subplot(gs[6:, 7:])
    
    # === Main UMAP Plot ===
    # Background: easy cases
    easy_mask = df['spatial_region'] == -1
    ax_main.scatter(df.loc[easy_mask, 'UMAP1'], df.loc[easy_mask, 'UMAP2'],
                   c='lightblue', s=1, alpha=0.15, label='Easy cases', rasterized=True)
    
    # Hard case regions
    regions = sorted([r for r in df['spatial_region'].unique() if r >= 0])
    colors = sns.color_palette('tab10', n_colors=len(regions))
    
    for i, region_id in enumerate(regions):
        region_mask = df['spatial_region'] == region_id
        region_df = df[region_mask]
        
        region_info = stats_df[stats_df['region_id'] == region_id].iloc[0]
        
        label = (f"Region {region_id} ({region_info['position_label']})\n"
                f"n={region_info['n_samples']}, {region_info['top_biotype']}")
        
        ax_main.scatter(region_df['UMAP1'], region_df['UMAP2'],
                       c=[colors[i]], s=12, alpha=0.65, 
                       edgecolors='black', linewidths=0.3,
                       label=label, rasterized=True)
        
        # Add region center marker
        ax_main.scatter(region_info['umap1_center'], region_info['umap2_center'],
                       c='black', marker='X', s=400, edgecolors='white', 
                       linewidths=3, zorder=100)
    
    ax_main.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    ax_main.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
    ax_main.set_title(f'{fig_tag}Hard Case Spatial Regions in β-VAE Embedding Space', 
                     fontsize=16, fontweight='bold', pad=15)
    
    legend = ax_main.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
                           fontsize=9, framealpha=0.95, edgecolor='black')
    legend.get_frame().set_linewidth(1.5)
    
    # === Bar Chart: Class Balance ===
    ax_bars.set_title(f'{fig_tag}Class Balance per Region', fontsize=12, fontweight='bold', pad=10)
    
    stats_sorted = stats_df.sort_values('region_id')
    n_regions = len(stats_sorted)
    y_positions = np.arange(n_regions)
    
    for i, row in enumerate(stats_sorted.itertuples()):
        # lncRNA segment
        ax_bars.barh(i, row.pct_lnc, 0.6,
                    label='lncRNA' if i == 0 else '',
                    color='#FF6B6B', edgecolor='black', linewidth=1)
        
        # Protein-coding segment
        ax_bars.barh(i, row.pct_pc, 0.6, left=row.pct_lnc,
                    label='Protein-coding' if i == 0 else '',
                    color='#4ECDC4', edgecolor='black', linewidth=1)
        
        # Add label
        label_text = f'n={int(row.n_samples)}: {row.pct_lnc:.1f}% lnc, {row.pct_pc:.1f}% PC'
        ax_bars.text(102, i, label_text, va='center', ha='left', fontsize=9)
    
    ax_bars.set_yticks(y_positions)
    ax_bars.set_yticklabels([f'R{int(r)}' for r in stats_sorted['region_id']], 
                           fontsize=11, fontweight='bold')
    ax_bars.set_xlabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax_bars.set_xlim(0, 140)
    ax_bars.set_ylim(-0.5, n_regions - 0.5)
    ax_bars.axvline(x=50, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
    ax_bars.axvline(x=100, color='black', linestyle='-', linewidth=1.5, alpha=0.3, zorder=0)
    ax_bars.grid(True, alpha=0.2, axis='x')
    
    ax_bars.legend(loc='upper left', bbox_to_anchor=(0, 1.25), 
                  fontsize=9, framealpha=0.95, ncol=2)
    
    ax_bars.spines['right'].set_visible(False)
    ax_bars.spines['top'].set_visible(False)
    
    plt.tight_layout()
    
    output_path = output_dir / 'spatial_regions_labeled.png'
    plt.savefig(output_path, dpi=350, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def create_cross_fold_summary(output_dir, fold_dirs, fig_tag=None):
    """Create summary comparing regions across all folds"""
    
    print("\n" + "=" * 80)
    print("CREATING CROSS-FOLD SUMMARY")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    
    # Collect region statistics from all folds
    all_fold_stats = []
    
    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        stats_file = output_dir / fold_name / 'region_statistics.csv'
        
        if not stats_file.exists():
            continue
        
        fold_stats = pd.read_csv(stats_file)
        fold_stats['fold'] = fold_name
        all_fold_stats.append(fold_stats)
    
    if not all_fold_stats:
        print("  No fold statistics found, skipping cross-fold summary")
        return
    
    combined_stats = pd.concat(all_fold_stats, ignore_index=True)
    
    # Save combined statistics
    combined_stats.to_csv(output_dir / 'all_folds_region_statistics.csv', index=False)
    print(f"    Saved: all_folds_region_statistics.csv")
    
    # Create summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Region sizes across folds
    ax = axes[0, 0]
    pivot = combined_stats.pivot(index='fold', columns='region_id', values='n_samples')
    pivot.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title(f'{fig_tag}Region Sizes Across Folds', fontsize=14, fontweight='bold')
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Class balance consistency
    ax = axes[0, 1]
    for region_id in sorted(combined_stats['region_id'].unique()):
        region_data = combined_stats[combined_stats['region_id'] == region_id]
        ax.plot(region_data['fold'], region_data['pct_lnc'], 
               marker='o', label=f'Region {region_id}', linewidth=2)
    
    ax.set_title(f'{fig_tag}lncRNA Percentage by Region (Across Folds)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('% lncRNA', fontsize=12)
    ax.set_ylim([0, 105])
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 3. Top biotype consistency
    ax = axes[1, 0]
    biotype_positions = combined_stats.groupby('top_biotype')['position_label'].value_counts()
    biotype_positions.plot(kind='barh', ax=ax)
    ax.set_title(f'{fig_tag}Top Biotypes by Region Position', fontsize=14, fontweight='bold')
    ax.set_xlabel('Count', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    # 4. Retained intron enrichment (if present)
    ax = axes[1, 1]
    if 'pct_retained_intron' in combined_stats.columns:
        pivot_ri = combined_stats.pivot(index='fold', columns='region_id', 
                                        values='pct_retained_intron')
        pivot_ri.plot(kind='bar', ax=ax, width=0.8, cmap='YlOrRd')
        ax.set_title(f'{fig_tag}Retained Intron % Across Folds', fontsize=14, fontweight='bold')
        ax.set_xlabel('Fold', fontsize=12)
        ax.set_ylabel('% Retained Intron', fontsize=12)
        ax.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'Retained intron data not available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_fold_summary.png', dpi=350, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved: cross_fold_summary.png")
    
    # Print summary statistics
    print("\n" + "-" * 80)
    print("CROSS-FOLD CONSISTENCY")
    print("-" * 80)
    
    for region_id in sorted(combined_stats['region_id'].unique()):
        region_data = combined_stats[combined_stats['region_id'] == region_id]
        
        print(f"\nRegion {region_id}:")
        print(f"  Position labels: {region_data['position_label'].value_counts().to_dict()}")
        print(f"  Size range: {region_data['n_samples'].min()}-{region_data['n_samples'].max()} samples")
        print(f"  lncRNA %: {region_data['pct_lnc'].mean():.1f}% ± {region_data['pct_lnc'].std():.1f}%")
        
        if 'pct_retained_intron' in region_data.columns:
            ri_mean = region_data['pct_retained_intron'].mean()
            ri_std = region_data['pct_retained_intron'].std()
            if ri_mean > 0:
                print(f"  Retained intron %: {ri_mean:.1f}% ± {ri_std:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Spatial region analysis for beta-VAE hard cases',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--umap_dir', required=True,
                       help='Directory containing fold_N/umap_embeddings.csv files')
    parser.add_argument('--n_regions', type=int, default=5,
                       help='Number of spatial regions to identify (default: 5)')
    parser.add_argument('--output_dir', default='spatial_analysis',
                       help='Output directory (default: spatial_analysis)')
    parser.add_argument('--model_label', default='β-VAE + Attn',
                        help='Label for the model/fold in visualizations (default: β-VAE + Attn)')
    parser.add_argument('--gencode_version', default='v47')
    
    args = parser.parse_args()

    # Build tag after parse:
    tag_parts = [p for p in [args.model_label,
                f'GENCODE {args.gencode_version}' if args.gencode_version else ''] if p]
    fig_tag = ' | '.join(tag_parts) + ' — ' if tag_parts else ''
    
    umap_dir = Path(args.umap_dir)
    
    if not umap_dir.exists():
        print(f"ERROR: Directory not found: {umap_dir}")
        return
    
    # Find all fold directories
    fold_dirs = sorted([d for d in umap_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('fold_')])
    
    if not fold_dirs:
        print(f"ERROR: No fold_N directories found in {umap_dir}")
        print(f"Expected structure: {umap_dir}/fold_0/umap_embeddings.csv")
        return
    
    print("=" * 80)
    print("SPATIAL REGION ANALYSIS: β-VAE HARD CASES")
    print("=" * 80)
    print(f"\nFound {len(fold_dirs)} folds:")
    for fold_dir in fold_dirs:
        print(f"  - {fold_dir.name}")
    
    # Process each fold
    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        umap_csv = fold_dir / 'umap_embeddings.csv'
        
        if not umap_csv.exists():
            print(f"\n  WARNING: {fold_name}/umap_embeddings.csv not found, skipping")
            continue
        
        print("\n" + "=" * 80)
        print(f"PROCESSING {fold_name.upper()}")
        print("=" * 80)
        
        # Create fold-specific output directory
        fold_output_dir = Path(args.output_dir) / fold_name
        fold_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load and validate data
        df = load_and_validate_data(umap_csv)
        
        # Identify spatial regions
        df, stats_df = identify_spatial_regions(df, n_regions=args.n_regions)
        
        # Save annotated data
        print(f"\nSaving annotated data...")
        df.to_csv(fold_output_dir / 'samples_with_regions.csv', index=False)
        stats_df.to_csv(fold_output_dir / 'region_statistics.csv', index=False)
        print(f"    samples_with_regions.csv ({len(df):,} samples)")
        print(f"    region_statistics.csv ({len(stats_df)} regions)")
        
        # Compare regions
        compare_spatial_regions(df, stats_df, fold_output_dir, fig_tag=fig_tag)
        
        # Visualize
        visualize_regions(df, stats_df, fold_output_dir, fig_tag=fig_tag)
        
        print(f"\n  {fold_name} complete: {fold_output_dir}/")
    
    print("\n" + "=" * 80)
    print("ALL FOLDS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"\nStructure:")
    print(f"  {args.output_dir}/")
    print(f"    fold_0/")
    print(f"      spatial_regions_labeled.png")
    print(f"      region_biotype_heatmap.png")
    print(f"      samples_with_regions.csv")
    print(f"      region_statistics.csv")
    print(f"      region_biotype_composition.csv")
    print(f"    fold_1/")
    print(f"      ...")
    print(f"    ...")
    
    # Optionally create cross-fold summary
    create_cross_fold_summary(args.output_dir, fold_dirs, fig_tag=fig_tag)


if __name__ == '__main__':
    main()