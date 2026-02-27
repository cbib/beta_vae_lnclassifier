#!/usr/bin/env python3
"""
Analyze GENCODE novelty patterns in beta-VAE embedding spaces.

This script:
1. Loads CV model embeddings from NPZ and per-fold UMAP CSV files
2. Tags transcripts by GENCODE novelty status (novel, common, reannotated)
3. Visualizes embedding space colored by novelty category
4. Quantifies spatial clustering and hard case enrichment
5. Compares patterns across all folds

Usage:
    python gencode_novelty_embeddings.py \
        --embeddings_npz exp_dir/embeddings_all_folds.npz \
        --fold_embeddings_dir exp_dir/per_fold_embeddings \
        --hard_cases exp_dir/evaluation_csvs/hard_cases.csv \
        --novel_fasta gencode.v49.new_with_class_transcripts.fa \
        --common_fasta gencode.v49.common_no_class_change_transcripts.fa \
        --reannotated_fasta gencode.v49.common_class_change_transcripts.fa \
        --old-version 47 \
        --new-version 49 \
        --output_dir gencode_novelty_analysis
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import Counter
from scipy.stats import chi2_contingency, fisher_exact, ttest_rel, ttest_1samp
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100  # Display only, savefig uses 1200
plt.rcParams['savefig.dpi'] = 1200
plt.rcParams['font.size'] = 10

def parse_fasta_ids(fasta_path: Path) -> Set[str]:
    """Extract transcript IDs from FASTA file (without version numbers)."""
    transcript_ids = set()
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                # Extract clean ID (first field before |)
                full_id = line[1:].strip().split('|')[0]
                # Remove version number
                clean_id = full_id.split('.')[0]
                transcript_ids.add(clean_id)
    return transcript_ids

def categorize_transcripts(transcript_ids: List[str],
                          novel_ids: Set[str],
                          common_ids: Set[str],
                          reannotated_ids: Set[str],
                          novel_label: str = 'novel') -> np.ndarray:
    """
    Categorize each transcript as novel, common, or reannotated.
    
    Args:
        transcript_ids: List of transcript IDs
        novel_ids: Set of novel transcript IDs
        common_ids: Set of common transcript IDs
        reannotated_ids: Set of reannotated transcript IDs
        novel_label: Label for novel category (e.g., 'novel_49')
    
    Returns:
        Array of category labels
    """
    categories = []
    for tid in transcript_ids:
        # Remove version number for matching
        clean_tid = tid.split('.')[0]
        
        if clean_tid in novel_ids:
            categories.append(novel_label)
        elif clean_tid in reannotated_ids:
            categories.append('reannotated')
        elif clean_tid in common_ids:
            categories.append('common')
        else:
            categories.append('unknown')
    
    return np.array(categories)

def load_embeddings_npz(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load all embeddings from NPZ file with per-fold structure.
    
    Handles format: fold_0_embeddings, fold_1_embeddings, etc.
    Returns concatenated data from all folds.
    """
    data = np.load(npz_path, allow_pickle=True)
    
    keys = list(data.keys())
    print(f"  NPZ keys: {keys[:5]}...")
    
    # Check for per-fold format
    fold_keys = [k for k in keys if k.startswith('fold_') and '_embeddings' in k]
    
    if fold_keys:
        # Per-fold format: concatenate all folds
        print(f"  Detected per-fold format with {len(fold_keys)} folds")
        
        all_embeddings = []
        all_labels = []
        all_transcript_ids = []
        
        # Extract fold numbers
        fold_nums = sorted(set([int(k.split('_')[1]) for k in fold_keys]))
        
        for fold_num in fold_nums:
            prefix = f'fold_{fold_num}_'
            
            fold_embeddings = data[f'{prefix}embeddings']
            fold_labels = data[f'{prefix}labels']
            fold_transcript_ids = data[f'{prefix}transcript_ids']
            
            all_embeddings.append(fold_embeddings)
            all_labels.append(fold_labels)
            
            # Handle bytes vs strings
            if isinstance(fold_transcript_ids[0], bytes):
                fold_transcript_ids = [tid.decode() for tid in fold_transcript_ids]
            all_transcript_ids.extend(fold_transcript_ids)
            
            print(f"    Fold {fold_num}: {len(fold_embeddings):,} samples")
        
        # Concatenate
        embeddings = np.concatenate(all_embeddings, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        print(f"  Total: {len(embeddings):,} samples")
        
    else:
        # Old flat format (fallback)
        embeddings = data['embeddings']
        labels = data['labels']
        transcript_ids = data['transcript_ids']
        
        if isinstance(transcript_ids[0], bytes):
            all_transcript_ids = [tid.decode() for tid in transcript_ids]
        else:
            all_transcript_ids = list(transcript_ids)
    
    return embeddings, labels, all_transcript_ids

def load_fold_umap(fold_csv_path: Path) -> pd.DataFrame:
    """Load UMAP coordinates and predictions for a specific fold."""
    df = pd.read_csv(fold_csv_path)
    return df

def calculate_spatial_metrics(umap_coords: np.ndarray, 
                              categories: np.ndarray,
                              category_name: str,
                              n_neighbors: int = 50) -> Dict:
    """
    Calculate spatial clustering metrics for a specific category.
    
    Metrics:
    - Local density: % of k-nearest neighbors in same category
    """
    mask = (categories == category_name)
    if mask.sum() == 0:
        return {'count': 0, 'local_density_mean': np.nan, 'local_density_std': np.nan, 'local_density_median': np.nan}
    
    # Fit k-NN on all points
    nbrs = NearestNeighbors(n_neighbors=min(n_neighbors, len(umap_coords)-1))
    nbrs.fit(umap_coords)
    
    # For points in this category, find their neighbors
    category_coords = umap_coords[mask]
    distances, indices = nbrs.kneighbors(category_coords)
    
    # Calculate local density: what % of neighbors are same category?
    neighbor_categories = categories[indices]
    same_category_neighbors = (neighbor_categories == category_name).sum(axis=1)
    local_density = same_category_neighbors / n_neighbors
    
    metrics = {
        'count': mask.sum(),
        'local_density_mean': local_density.mean(),
        'local_density_std': local_density.std(),
        'local_density_median': np.median(local_density),
    }
    
    return metrics

def plot_embedding_by_novelty(umap_coords: np.ndarray,
                              categories: np.ndarray,
                              true_labels: np.ndarray,
                              fold_id: int,
                              output_path: Path,
                              novel_label: str,
                              new_version: str):
    """Create visualization of embedding space colored by novelty category."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    # Color schemes
    category_colors = {
        novel_label: '#E74C3C',      # Red
        'common': '#3498DB',          # Blue
        'reannotated': '#F39C12',     # Orange
        'unknown': '#95A5A6'          # Gray
    }
    
    class_names = {0: 'lncRNA', 1: 'protein_coding'}
    
    # Panel 1: Novel vs Common only (show dispersion with different alphas)
    ax = axes[0]
    common_mask = (categories == 'common')
    novel_mask = (categories == novel_label)
    
    # Plot common first as background with lower alpha
    ax.scatter(umap_coords[common_mask, 0], umap_coords[common_mask, 1],
              c=category_colors['common'], label=f'Common (n={common_mask.sum():,})',
              alpha=0.15, s=3, rasterized=True, edgecolors='none')
    # Plot novel on top with higher alpha to show where they are
    ax.scatter(umap_coords[novel_mask, 0], umap_coords[novel_mask, 1],
              c=category_colors[novel_label], label=f'Novel (n={novel_mask.sum():,})',
              alpha=0.4, s=3, rasterized=True, edgecolors='none')
    
    ax.set_xlabel('UMAP 1', fontsize=11)
    ax.set_ylabel('UMAP 2', fontsize=11)
    ax.set_title(f'Fold {fold_id}: Novel vs Common Spatial Distribution', fontsize=12, fontweight='bold')
    ax.legend(markerscale=3, loc='upper right', framealpha=0.9)
    
    # Panel 2: Density heatmap comparison - Common
    ax = axes[1]
    
    common_coords = umap_coords[common_mask]
    if len(common_coords) > 1000:  # Subsample for speed
        idx = np.random.choice(len(common_coords), 5000, replace=False)
        common_coords = common_coords[idx]
    
    ax.hexbin(common_coords[:, 0], common_coords[:, 1], 
              gridsize=50, cmap='Blues', mincnt=1, alpha=0.8)
    ax.set_xlabel('UMAP 1', fontsize=11)
    ax.set_ylabel('UMAP 2', fontsize=11)
    ax.set_title(f'Common Transcripts: Density Distribution (n={common_mask.sum():,})', 
                 fontsize=12, fontweight='bold')
    
    # Panel 3: Density heatmap comparison - Novel
    ax = axes[2]
    novel_coords = umap_coords[novel_mask]
    if len(novel_coords) > 1000:
        idx = np.random.choice(len(novel_coords), 5000, replace=False)
        novel_coords = novel_coords[idx]
    
    ax.hexbin(novel_coords[:, 0], novel_coords[:, 1],
              gridsize=50, cmap='Reds', mincnt=1, alpha=0.8)
    ax.set_xlabel('UMAP 1', fontsize=11)
    ax.set_ylabel('UMAP 2', fontsize=11)
    ax.set_title(f'Novel Transcripts: Density Distribution (n={novel_mask.sum():,})', 
                 fontsize=12, fontweight='bold')
    
    # Panel 4: By true class, faceted by novelty
    ax = axes[3]
    for class_label in [0, 1]:
        for cat in ['common', novel_label]:
            mask = (true_labels == class_label) & (categories == cat)
            if mask.sum() > 0:
                marker = 'o' if class_label == 0 else '^'
                alpha = 0.2 if cat == 'common' else 0.4
                size = 2 if cat == 'common' else 3
                label = f"{class_names[class_label]} - {cat.replace('_', ' ')}"
                ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                          c=category_colors[cat], marker=marker,
                          label=label, alpha=alpha, s=size, rasterized=True,
                          edgecolors='none')
    
    ax.set_xlabel('UMAP 1', fontsize=11)
    ax.set_ylabel('UMAP 2', fontsize=11)
    ax.set_title(f'Fold {fold_id}: Class × Novelty Interaction', fontsize=12, fontweight='bold')
    ax.legend(markerscale=2, fontsize=9, loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=350)
    plt.close()
    print(f"  Saved visualization: {output_path}")

def analyze_hard_case_enrichment(categories: np.ndarray,
                                 hard_case_ids: Set[str],
                                 transcript_ids: List[str],
                                 fold_id: int,
                                 novel_label: str) -> pd.DataFrame:
    """Test if hard cases are enriched in specific novelty categories."""
    # Mark hard cases
    is_hard_case = np.array([tid.split('.')[0] in hard_case_ids 
                             for tid in transcript_ids])
    
    results = []
    # Only analyze novel and common (exclude reannotated due to small n)
    for cat in [novel_label, 'common']:
        cat_mask = (categories == cat)
        if cat_mask.sum() == 0:
            continue
        
        # Contingency table
        hard_in_cat = (is_hard_case & cat_mask).sum()
        hard_not_in_cat = (is_hard_case & ~cat_mask).sum()
        not_hard_in_cat = (~is_hard_case & cat_mask).sum()
        not_hard_not_in_cat = (~is_hard_case & ~cat_mask).sum()
        
        contingency = np.array([[hard_in_cat, hard_not_in_cat],
                               [not_hard_in_cat, not_hard_not_in_cat]])
        
        # Fisher's exact test
        odds_ratio, p_value = fisher_exact(contingency)
        
        # Calculate percentages
        hard_case_rate_in_cat = hard_in_cat / cat_mask.sum() if cat_mask.sum() > 0 else 0
        hard_case_rate_overall = is_hard_case.sum() / len(is_hard_case)
        
        results.append({
            'fold': fold_id,
            'category': cat,
            'n_transcripts': cat_mask.sum(),
            'n_hard_cases': hard_in_cat,
            'hard_case_rate': hard_case_rate_in_cat,
            'baseline_rate': hard_case_rate_overall,
            'fold_enrichment': hard_case_rate_in_cat / hard_case_rate_overall if hard_case_rate_overall > 0 else np.nan,
            'odds_ratio': odds_ratio,
            'p_value': p_value,
            'significant': p_value < 0.05
        })
    
    return pd.DataFrame(results)

def analyze_classification_performance(categories: np.ndarray,
                                       true_labels: np.ndarray,
                                       pred_labels: np.ndarray,
                                       fold_id: int,
                                       novel_label: str) -> pd.DataFrame:
    """Compare classification accuracy across novelty categories."""
    results = []
    
    # Only analyze novel and common (exclude reannotated due to small n)
    for cat in [novel_label, 'common']:
        cat_mask = (categories == cat)
        if cat_mask.sum() == 0:
            continue
        
        # Per-class accuracy
        for class_label in [0, 1]:
            class_mask = cat_mask & (true_labels == class_label)
            if class_mask.sum() == 0:
                continue
            
            class_correct = (true_labels[class_mask] == pred_labels[class_mask]).sum()
            class_total = class_mask.sum()
            class_acc = class_correct / class_total
            
            results.append({
                'fold': fold_id,
                'category': cat,
                'class': 'lncRNA' if class_label == 0 else 'protein_coding',
                'n_samples': class_total,
                'accuracy': class_acc
            })
    
    return pd.DataFrame(results)

def create_comprehensive_summary_figures(all_fold_results: Dict, output_dir: Path, 
                                        old_version: str, new_version: str, novel_label: str):
    """
    Create publication-quality summary figures showing:
    1. Hard case enrichment (novel vs common) - MAIN FIGURE
    2. Classification accuracy comparison
    3. Spatial clustering patterns
    """
    
    def get_sig_stars(p):
        if p < 0.001: return '***'
        elif p < 0.01: return '**'
        elif p < 0.05: return '*'
        else: return 'ns'
    
    # =========================================================================
    # Figure 1: Hard Case Enrichment - Main Result
    # =========================================================================
    enrichment_df = pd.concat([df for df in all_fold_results['hard_case_enrichment']])
    
    # Only analyze novel vs common (exclude reannotated if present)
    enrichment_df = enrichment_df[enrichment_df['category'].isin([novel_label, 'common'])]
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Panel A: Fold Enrichment per Category (BAR CHART)
    ax = fig.add_subplot(gs[0, :2])

    # Compute mean and SEM across folds
    summary_stats = enrichment_df.groupby('category').agg({
        'fold_enrichment': ['mean', 'std', 'count'],
        'hard_case_rate': 'mean',
        'baseline_rate': 'first'
    }).reset_index()

    summary_stats.columns = ['category', 'mean_enrichment', 'std_enrichment', 
                            'n_folds', 'mean_hard_rate', 'baseline_rate']
    summary_stats['sem_enrichment'] = summary_stats['std_enrichment'] / np.sqrt(summary_stats['n_folds'])

    # CRITICAL FIX: Force correct order and build bars explicitly
    category_order = [novel_label, 'common']
    summary_stats['category'] = pd.Categorical(
        summary_stats['category'],
        categories=category_order,
        ordered=True
    )
    summary_stats = summary_stats.sort_values('category').reset_index(drop=True)

    # Now summary_stats is GUARANTEED to be [novel, common]
    x_pos = np.arange(len(summary_stats))

    # Assign colors based on actual category
    colors = []
    x_labels = []
    for _, row in summary_stats.iterrows():
        cat = row['category']
        colors.append('#E74C3C' if cat == novel_label else '#3498DB')
        x_labels.append(f'Novel (GENCODE v{new_version})' if cat == novel_label 
                    else f'Common (v{old_version} & v{new_version})')

    bars = ax.bar(x_pos, summary_stats['mean_enrichment'], 
                yerr=summary_stats['sem_enrichment'],
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.5,
                capsize=8, error_kw={'linewidth': 2})

    # Reference line at 1.0 (no enrichment)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.6, 
            label='No enrichment (baseline)')

    # Add individual fold points
    for i, row in summary_stats.iterrows():
        cat = row['category']
        cat_data = enrichment_df[enrichment_df['category'] == cat]
        y_vals = cat_data['fold_enrichment'].values
        x_vals = np.random.normal(i, 0.04, size=len(y_vals))  # Jitter
        ax.scatter(x_vals, y_vals, color='black', alpha=0.5, s=50, 
                edgecolors='white', linewidths=1, zorder=10)

    # Statistical test
    novel_enrichments = enrichment_df[enrichment_df['category'] == novel_label]['fold_enrichment'].values
    common_enrichments = enrichment_df[enrichment_df['category'] == 'common']['fold_enrichment'].values

    # Test if each is different from baseline (1.0)
    novel_t, novel_p = ttest_1samp(novel_enrichments, 1.0)
    common_t, common_p = ttest_1samp(common_enrichments, 1.0)

    # Add significance stars
    for i, row in summary_stats.iterrows():
        cat = row['category']
        p_val = novel_p if cat == novel_label else common_p
        sig = get_sig_stars(p_val)
        y_pos = row['mean_enrichment'] + row['sem_enrichment'] + 0.1
        ax.text(i, y_pos, sig, ha='center', fontsize=16, fontweight='bold')

    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=12, fontweight='bold')
    ax.set_ylabel('Hard Case Fold Enrichment', fontsize=13, fontweight='bold')
    ax.set_title('A. Hard Case Enrichment by GENCODE Novelty Status', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=1)
    ax.set_ylim(0, max(summary_stats['mean_enrichment']) * 1.3)
    
    # Panel B: Hard Case Rate (ACTUAL PERCENTAGES)
    ax = fig.add_subplot(gs[0, 2])

    # Show actual hard case rates
    baseline = summary_stats['baseline_rate'].iloc[0]

    # summary_stats is already ordered as [novel_label, 'common']
    bar_data = []
    colors_rates = []

    for i, row in summary_stats.iterrows():
        cat = row['category']
        bar_data.append({
            'category': 'Novel' if cat == novel_label else 'Common',
            'mean_hard_rate': row['mean_hard_rate']
        })
        colors_rates.append('#E74C3C' if cat == novel_label else '#3498DB')

    # Add baseline
    bar_data.append({'category': 'Overall\nBaseline', 'mean_hard_rate': baseline})
    colors_rates.append('#95A5A6')

    bar_df = pd.DataFrame(bar_data)

    x_pos_rates = np.arange(len(bar_df))

    ax.bar(x_pos_rates, bar_df['mean_hard_rate'] * 100, 
        color=colors_rates, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add percentage labels on bars
    for i, val in enumerate(bar_df['mean_hard_rate']):
        ax.text(i, val * 100 + 0.3, f'{val*100:.1f}%', 
            ha='center', fontsize=11, fontweight='bold')

    ax.set_xticks(x_pos_rates)
    ax.set_xticklabels(bar_df['category'].tolist(), fontsize=10)
    ax.set_ylabel('Hard Case Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('B. Hard Case Rates', fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=1)
    ax.set_ylim(0, max(bar_df['mean_hard_rate']) * 120)
    
    # Panel C: Per-Fold Heatmap
    ax = fig.add_subplot(gs[1, 0])
    
    pivot_data = enrichment_df.pivot_table(
        values='hard_case_rate', 
        index='fold', 
        columns='category'
    )
    
    # Reorder columns: novel, common
    if novel_label in pivot_data.columns and 'common' in pivot_data.columns:
        pivot_data = pivot_data[[novel_label, 'common']]
    
    sns.heatmap(pivot_data * 100, annot=True, fmt='.2f', cmap='YlOrRd', 
                ax=ax, cbar_kws={'label': 'Hard Case Rate (%)'}, 
                linewidths=1, linecolor='black', vmin=0)
    ax.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fold', fontsize=12, fontweight='bold')
    ax.set_title('C. Hard Case Rate Heatmap', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticklabels(['Novel', 'Common'], fontsize=11)
    
    # Panel D: Fold Enrichment Heatmap
    ax = fig.add_subplot(gs[1, 1])
    
    pivot_enrichment = enrichment_df.pivot_table(
        values='fold_enrichment',
        index='fold',
        columns='category'
    )
    
    if novel_label in pivot_enrichment.columns and 'common' in pivot_enrichment.columns:
        pivot_enrichment = pivot_enrichment[[novel_label, 'common']]
    
    sns.heatmap(pivot_enrichment, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=1.0, vmin=0.5, vmax=1.5,
                ax=ax, cbar_kws={'label': 'Fold Enrichment'}, 
                linewidths=1, linecolor='black')
    ax.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fold', fontsize=12, fontweight='bold')
    ax.set_title('D. Fold Enrichment Heatmap', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticklabels(['Novel', 'Common'], fontsize=11)
    
    # Panel E: Statistical Summary Table
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    
    # Create summary statistics text
    stats_text = "Statistical Summary\n" + "="*40 + "\n\n"
    
    for cat in [novel_label, 'common']:
        cat_data = enrichment_df[enrichment_df['category'] == cat]
        mean_enrich = cat_data['fold_enrichment'].mean()
        std_enrich = cat_data['fold_enrichment'].std()
        mean_rate = cat_data['hard_case_rate'].mean()
        n_folds = len(cat_data)
        
        # Test against baseline
        t_stat, p_val = ttest_1samp(cat_data['fold_enrichment'].values, 1.0)
        
        cat_label = f"Novel (v{new_version})" if cat == novel_label else "Common"
        stats_text += f"{cat_label}:\n"
        stats_text += f"  Enrichment: {mean_enrich:.3f} ± {std_enrich:.3f}\n"
        stats_text += f"  Hard rate: {mean_rate*100:.2f}%\n"
        stats_text += f"  vs Baseline: p={p_val:.4f} {get_sig_stars(p_val)}\n"
        stats_text += f"  n={n_folds} folds\n\n"
    
    # Compare novel vs common
    if len(novel_enrichments) == len(common_enrichments):
        t_comp, p_comp = ttest_rel(novel_enrichments, common_enrichments)
        stats_text += f"Novel vs Common:\n"
        stats_text += f"  Paired t-test: t={t_comp:.3f}\n"
        stats_text += f"  p={p_comp:.4f} {get_sig_stars(p_comp)}\n\n"
    
    stats_text += "\nSignificance:\n"
    stats_text += "*** p<0.001  ** p<0.01  * p<0.05"
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Save figure
    plt.tight_layout()
    output_path = output_dir / 'summary_hard_case_enrichment_comprehensive.png'
    plt.savefig(output_path, dpi=350, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Main summary figure saved: {output_path}")
    
    # =========================================================================
    # Figure 2: Classification Accuracy Comparison
    # =========================================================================
    accuracy_df = pd.concat([df for df in all_fold_results['classification_performance']])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Boxplot by category and class
    ax = axes[0]
    sns.boxplot(data=accuracy_df, x='category', y='accuracy', hue='class', ax=ax,
               palette={'lncRNA': '#FF6B6B', 'protein_coding': '#4ECDC4'})
    ax.set_xlabel('GENCODE Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Classification Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Classification Performance by Novelty Status', 
                fontsize=13, fontweight='bold')
    ax.set_xticklabels(['Novel', 'Common'])
    ax.set_ylim(0.7, 1.0)
    ax.legend(title='True Class', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Panel B: Mean accuracy comparison
    ax = axes[1]

    # Explicitly order categories
    category_order = [novel_label, 'common']  # ← Force correct order

    acc_summary = accuracy_df.groupby(['category', 'class'])['accuracy'].agg(['mean', 'std', 'count']).reset_index()
    acc_summary['sem'] = acc_summary['std'] / np.sqrt(acc_summary['count'])

    # Ensure data is in correct order
    acc_summary['category'] = pd.Categorical(acc_summary['category'], categories=category_order, ordered=True)
    acc_summary = acc_summary.sort_values(['category', 'class'])  # ← Sort explicitly

    x = np.arange(len(category_order))  # ← Use explicit order
    width = 0.35

    # Split by class (now guaranteed to be in correct order)
    lnc_data = acc_summary[acc_summary['class'] == 'lncRNA'].reset_index(drop=True)
    pc_data = acc_summary[acc_summary['class'] == 'protein_coding'].reset_index(drop=True)

    # Verify alignment
    assert len(lnc_data) == len(x), f"lnc_data length mismatch: {len(lnc_data)} vs {len(x)}"
    assert len(pc_data) == len(x), f"pc_data length mismatch: {len(pc_data)} vs {len(x)}"

    # Plot bars
    ax.bar(x - width/2, lnc_data['mean'], width, 
        yerr=lnc_data['sem'], label='lncRNA',
        color='#FF6B6B', alpha=0.8, capsize=5)
    ax.bar(x + width/2, pc_data['mean'], width,
        yerr=pc_data['sem'], label='Protein-coding',
        color='#4ECDC4', alpha=0.8, capsize=5)

    # Update x-tick labels
    ax.set_xticks(x)
    ax.set_xticklabels(['Novel', 'Common'])  # ← Match category_order
    
    ax.set_xlabel('GENCODE Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Accuracy ± SEM', fontsize=12, fontweight='bold')
    ax.set_title('Mean Classification Accuracy', fontsize=13, fontweight='bold')
    ax.set_ylim(0.7, 1.0)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_classification_accuracy.png', dpi=1200, bbox_inches='tight')
    plt.close()
    
    print(f"  Accuracy summary figure saved")
    
    # =========================================================================
    # Figure 3: Spatial Clustering
    # =========================================================================
    spatial_df = pd.concat([df for df in all_fold_results['spatial_metrics']])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Local density comparison
    ax = axes[0]
    spatial_plot = spatial_df[spatial_df['count'] > 0].copy()
    
    sns.violinplot(data=spatial_plot, x='category', y='local_density_mean', ax=ax,
                  palette={novel_label: '#E74C3C', 'common': '#3498DB'})
    ax.set_xlabel('GENCODE Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Local Density (k=50)', fontsize=12, fontweight='bold')
    ax.set_title('Spatial Clustering: Are Novel Transcripts Clustered?', 
                fontsize=13, fontweight='bold')
    ax.set_xticklabels(['Novel', 'Common'])
    ax.grid(axis='y', alpha=0.3)
    
    # Panel B: Count by category
    ax = axes[1]
    count_summary = spatial_plot.groupby('category')['count'].mean()
    
    ax.bar(range(len(count_summary)), count_summary.values,
          color=['#E74C3C', '#3498DB'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(count_summary)))
    ax.set_xticklabels(['Novel', 'Common'], fontsize=11)
    ax.set_ylabel('Mean Sample Count per Fold', fontsize=12, fontweight='bold')
    ax.set_title('Sample Size Distribution', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_spatial_clustering.png', dpi=350, bbox_inches='tight')
    plt.close()
    
    print(f"  Spatial clustering figure saved")
    
    # Print key findings
    print("\n" + "="*80)
    print("KEY FINDINGS SUMMARY")
    print("="*80)
    
    novel_mean = enrichment_df[enrichment_df['category'] == novel_label]['fold_enrichment'].mean()
    novel_rate = enrichment_df[enrichment_df['category'] == novel_label]['hard_case_rate'].mean()
    common_rate = enrichment_df[enrichment_df['category'] == 'common']['hard_case_rate'].mean()
    
    print(f"\n1. Hard Case Enrichment:")
    print(f"   Novel transcripts: {novel_mean:.2f}× enriched (p={novel_p:.4f} {get_sig_stars(novel_p)})")
    print(f"   Novel hard rate: {novel_rate*100:.2f}%")
    print(f"   Common hard rate: {common_rate*100:.2f}%")
    print(f"   Difference: {(novel_rate - common_rate)*100:.2f} percentage points")
    
    if len(novel_enrichments) == len(common_enrichments):
        print(f"\n2. Novel vs Common Comparison:")
        print(f"   Paired t-test: t={t_comp:.3f}, p={p_comp:.4f} {get_sig_stars(p_comp)}")
        if p_comp < 0.05:
            print(f"   → Novel transcripts are SIGNIFICANTLY more likely to be hard cases")
        else:
            print(f"   → No significant difference between novel and common")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze GENCODE novelty in embedding spaces',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python gencode_novelty_embeddings.py \\
        --embeddings_npz experiments/beta_vae_g47/embeddings_all_folds.npz \\
        --fold_embeddings_dir experiments/beta_vae_g47/umap_visualizations \\
        --hard_cases experiments/beta_vae_g47/evaluation_csvs/hard_cases.csv \\
        --novel_fasta resources/gencode_comparison_v47_v49/gencode.v49.new_with_class_transcripts.fa \\
        --common_fasta resources/gencode_comparison_v47_v49/gencode.v49.common_no_class_change_transcripts.fa \\
        --reannotated_fasta resources/gencode_comparison_v47_v49/gencode.v49.common_class_change_transcripts.fa \\
        --old-version 47 \\
        --new-version 49 \\
        --output_dir experiments/beta_vae_g47/gencode_novelty_analysis
        """
    )
    parser.add_argument('--embeddings_npz', type=str, required=True,
                       help='Path to embeddings_all_folds.npz file')
    parser.add_argument('--fold_embeddings_dir', type=str, required=True,
                       help='Directory containing umap_visualizations/fold_X subdirectories')
    parser.add_argument('--hard_cases', type=str, required=True,
                       help='Path to hard_cases.csv file')
    parser.add_argument('--novel_fasta', type=str, required=True,
                       help='Path to gencode.vXX.new_with_class_transcripts.fa')
    parser.add_argument('--common_fasta', type=str, required=True,
                       help='Path to gencode.vXX.common_no_class_change_transcripts.fa')
    parser.add_argument('--reannotated_fasta', type=str, required=True,
                       help='Path to gencode.vXX.common_class_change_transcripts.fa')
    parser.add_argument('--old-version', type=str, required=True,
                       help='Old GENCODE version number (e.g., 47)')
    parser.add_argument('--new-version', type=str, required=True,
                       help='New GENCODE version number (e.g., 49)')
    parser.add_argument('--output_dir', type=str, default='gencode_novelty_analysis',
                       help='Output directory for results')
    parser.add_argument('--folds', type=str, default='0,1,2,3,4',
                       help='Comma-separated fold IDs to analyze')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create dynamic novel label
    novel_label = f'novel_{args.new_version}'
    
    print("="*80)
    print(f"GENCODE v{args.old_version} vs v{args.new_version} Novelty Analysis in CV Model Embedding Spaces")
    print("="*80)
    
    # Load GENCODE category IDs
    print("\n  Loading GENCODE category IDs...")
    novel_ids = parse_fasta_ids(Path(args.novel_fasta))
    common_ids = parse_fasta_ids(Path(args.common_fasta))
    reannotated_ids = parse_fasta_ids(Path(args.reannotated_fasta))
    
    print(f"  Novel (v{args.new_version} only): {len(novel_ids):,} transcripts")
    print(f"  Common (no change): {len(common_ids):,} transcripts")
    print(f"  Reannotated (class change): {len(reannotated_ids):,} transcripts")
    
    # Load hard cases
    print(f"\n  Loading hard cases from {args.hard_cases}...")
    hard_cases_df = pd.read_csv(args.hard_cases)
    hard_case_ids = set(hard_cases_df['transcript_id'].str.split('.').str[0])
    print(f"  Hard cases: {len(hard_case_ids):,} transcripts")
    
    # Load all embeddings from NPZ
    print(f"\n  Loading embeddings from {args.embeddings_npz}...")
    embeddings_npz_path = Path(args.embeddings_npz)
    all_embeddings, all_labels, all_transcript_ids = load_embeddings_npz(embeddings_npz_path)
    print(f"  Loaded {len(all_transcript_ids):,} total transcripts")
    
    # Process each fold
    folds = [int(f) for f in args.folds.split(',')]
    fold_embeddings_dir = Path(args.fold_embeddings_dir)
    
    all_fold_results = {
        'hard_case_enrichment': [],
        'classification_performance': [],
        'spatial_metrics': []
    }
    
    for fold_id in folds:
        print(f"\n{'='*80}")
        print(f"Processing Fold {fold_id}")
        print(f"{'='*80}")
        
        # Load fold-specific UMAP embeddings
        fold_csv_path = fold_embeddings_dir / f'fold_{fold_id}' / 'umap_embeddings.csv'
        if not fold_csv_path.exists():
            print(f"     Fold CSV not found: {fold_csv_path}")
            continue
        
        print(f"  Loading fold data from {fold_csv_path}...")
        fold_df = load_fold_umap(fold_csv_path)

        # Validate required columns
        required_cols = ['UMAP1', 'UMAP2', 'transcript_id', 'true_label', 'predicted_label']
        missing = [c for c in required_cols if c not in fold_df.columns]
        if missing:
            print(f"     Missing columns: {missing}")
            print(f"     Available: {list(fold_df.columns)}")
            continue
        
        # Extract UMAP coordinates and transcript info
        umap_coords = fold_df[['UMAP1', 'UMAP2']].values
        
        # Extract clean transcript IDs (first field before |)
        transcript_ids = fold_df['transcript_id'].str.split('|').str[0].tolist()
        
        # Convert true_label string to binary (lnc=0, pc=1)
        if fold_df['true_label'].dtype == 'object':
            true_labels = (fold_df['true_label'] == 'pc').astype(int).values
            pred_labels = (fold_df['predicted_label'] == 'pc').astype(int).values
        else:
            true_labels = fold_df['true_label'].values
            pred_labels = fold_df['predicted_label'].values
        
        print(f"  Loaded {len(transcript_ids):,} transcripts from fold {fold_id}")
        
        # Categorize transcripts
        print("  Categorizing transcripts by GENCODE novelty...")
        categories = categorize_transcripts(transcript_ids, novel_ids, common_ids, reannotated_ids, novel_label)
        
        category_counts = Counter(categories)
        print(f"    Novel: {category_counts[novel_label]:,}")
        print(f"    Common: {category_counts['common']:,}")
        print(f"    Reannotated: {category_counts['reannotated']:,} (excluded from statistical analysis due to small n)")
        print(f"    Unknown: {category_counts['unknown']:,}")
        
        # Skip fold if reannotated is the only category with substantial counts
        if category_counts['reannotated'] > category_counts[novel_label] + category_counts['common']:
            print(f"     Fold {fold_id} has unusual category distribution, skipping...")
            continue
        
        # Visualizations
        print("  Creating visualizations...")
        plot_path = output_dir / f'fold_{fold_id}_novelty_embedding.png'
        plot_embedding_by_novelty(umap_coords, categories, true_labels, fold_id, plot_path, 
                                  novel_label, args.new_version)
        
        # Spatial clustering analysis
        print("  Analyzing spatial clustering...")
        spatial_results = []
        # Only analyze novel and common (exclude reannotated due to small n)
        for cat in [novel_label, 'common']:
            metrics = calculate_spatial_metrics(umap_coords, categories, cat)
            metrics['fold'] = fold_id
            metrics['category'] = cat
            spatial_results.append(metrics)
        
        spatial_df = pd.DataFrame(spatial_results)
        all_fold_results['spatial_metrics'].append(spatial_df)
        
        print("\n  Spatial Metrics:")
        print(spatial_df.to_string(index=False))
        
        # Hard case enrichment
        print("\n  Analyzing hard case enrichment...")
        enrichment_df = analyze_hard_case_enrichment(
            categories, hard_case_ids, transcript_ids, fold_id, novel_label
        )
        all_fold_results['hard_case_enrichment'].append(enrichment_df)
        
        print("\n  Hard Case Enrichment:")
        print(enrichment_df.to_string(index=False))
        
        # Classification performance
        print("\n  Analyzing classification performance...")
        performance_df = analyze_classification_performance(
            categories, true_labels, pred_labels, fold_id, novel_label
        )
        all_fold_results['classification_performance'].append(performance_df)
        
        print("\n  Classification Performance:")
        print(performance_df.to_string(index=False))
        
        # Save fold-specific results
        fold_output = output_dir / f'fold_{fold_id}_results.csv'
        fold_summary = pd.concat([
            enrichment_df.assign(metric_type='hard_case_enrichment'),
            performance_df.assign(metric_type='classification_performance'),
            spatial_df.assign(metric_type='spatial_clustering')
        ], ignore_index=True)
        fold_summary.to_csv(fold_output, index=False)
        print(f"  Saved fold results: {fold_output}")
    
    # Create summary plots across all folds
    if len(all_fold_results['hard_case_enrichment']) > 0:
        print(f"\n{'='*80}")
        print("Creating Summary Visualizations Across All Folds")
        print(f"{'='*80}")
        create_comprehensive_summary_figures(all_fold_results, output_dir, 
                                            args.old_version, args.new_version, novel_label)
            
    # Save combined results
    combined_enrichment = pd.concat(all_fold_results['hard_case_enrichment'])
    combined_enrichment.to_csv(output_dir / 'all_folds_hard_case_enrichment.csv', index=False)
    
    combined_performance = pd.concat(all_fold_results['classification_performance'])
    combined_performance.to_csv(output_dir / 'all_folds_classification_performance.csv', index=False)
    
    combined_spatial = pd.concat(all_fold_results['spatial_metrics'])
    combined_spatial.to_csv(output_dir / 'all_folds_spatial_metrics.csv', index=False)
    
    print(f"\n  Analysis complete! Results saved to {output_dir}")

if __name__ == '__main__':
    main()