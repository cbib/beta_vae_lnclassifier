#!/usr/bin/env python3
"""
Enhanced cross-model comparison with automatic deep statistical analysis.

Version-agnostic: automatically detects which GENCODE versions are being compared.
Performs comprehensive sequence and biotype analysis for each performance category.

Usage:
    python compare_models_deep_analysis.py \
        --model1_predictions gencode_v47_experiments/.../all_sample_predictions.csv \
        --model1_name "G47" \
        --model1_biotypes data/dataset_biotypes/g47_dataset_biotypes_cdhit.csv \
        --model1_lnc_fasta data/gencode_v47_lnc_cdhit.fa \
        --model1_pc_fasta data/gencode_v47_pc_cdhit.fa \
        --model2_predictions gencode_v49_experiments/.../all_sample_predictions.csv \
        --model2_name "G49" \
        --model2_biotypes data/dataset_biotypes/g49_dataset_biotypes_cdhit.csv \
        --model2_lnc_fasta data/gencode_v49_lnc_cdhit.fa \
        --model2_pc_fasta data/gencode_v49_pc_cdhit.fa \
        --common_fasta resources/gencode_comparison_v47_v49/gencode.v49.common_no_class_change_transcripts.fa \
        --output_dir cross_model_deep_analysis
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, fisher_exact
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 350
plt.rcParams['font.size'] = 10

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_predictions(predictions_csv, model_name):
    """Load model predictions CSV."""
    print(f"\n  Loading {model_name} predictions...")
    df = pd.read_csv(predictions_csv)
    df['transcript_id_clean'] = df['transcript_id'].str.split('.').str[0]
    print(f"    Loaded {len(df):,} predictions")
    return df

def parse_fasta_ids(fasta_path):
    """Extract clean transcript IDs from FASTA."""
    ids = set()
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                full_id = line[1:].strip().split('|')[0]
                clean_id = full_id.split('.')[0]
                ids.add(clean_id)
    return ids

def load_sequence_features(lnc_fasta, pc_fasta, model_name):
    """Load sequences and calculate features (length, GC content)."""
    print(f"\n  Loading {model_name} sequence features...")
    
    seq_features = {}
    
    for fasta_path, seq_type in [(lnc_fasta, 'lnc'), (pc_fasta, 'pc')]:
        for record in SeqIO.parse(fasta_path, 'fasta'):
            tid = record.id.split('|')[0].split('.')[0]
            seq = str(record.seq).upper()
            
            seq_features[tid] = {
                'length': len(seq),
                'gc_content': (seq.count('G') + seq.count('C')) / len(seq) if len(seq) > 0 else 0,
                'true_class': seq_type
            }
    
    print(f"    Loaded {len(seq_features):,} sequences")
    return seq_features

def load_biotypes(biotype_csv, model_name):
    """Load biotype information."""
    print(f"\n  Loading {model_name} biotypes...")
    df = pd.read_csv(biotype_csv)
    df['transcript_id_clean'] = df['transcript_id'].str.split('.').str[0]
    biotype_dict = dict(zip(df['transcript_id_clean'], df['biotype']))
    print(f"    Loaded {len(biotype_dict):,} biotype annotations")
    return biotype_dict

# ============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# ============================================================================

def compare_distributions(group1, group2, feature_name, group1_name, group2_name):
    """
    Compare distributions between two groups.
    Returns dict with test statistics and interpretation.
    """
    # Remove NaN values
    g1 = group1.dropna()
    g2 = group2.dropna()
    
    if len(g1) < 10 or len(g2) < 10:
        return {
            'feature': feature_name,
            'group1_mean': np.nan,
            'group2_mean': np.nan,
            'difference': np.nan,
            'test': 'insufficient_data',
            'statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'interpretation': 'Insufficient data for comparison'
        }
    
    # Compute descriptive statistics
    g1_mean = g1.mean()
    g2_mean = g2.mean()
    difference = g1_mean - g2_mean
    
    # Choose appropriate test
    # For continuous features: Mann-Whitney U (non-parametric)
    stat, p_value = mannwhitneyu(g1, g2, alternative='two-sided')
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(g1)-1)*g1.std()**2 + (len(g2)-1)*g2.std()**2) / (len(g1)+len(g2)-2))
    cohens_d = difference / pooled_std if pooled_std > 0 else 0
    
    # Interpretation
    significant = p_value < 0.05
    if significant:
        direction = "higher" if difference > 0 else "lower"
        interpretation = f"{group1_name} has significantly {direction} {feature_name} (p={p_value:.4f}, d={cohens_d:.2f})"
    else:
        interpretation = f"No significant difference in {feature_name} (p={p_value:.4f})"
    
    return {
        'feature': feature_name,
        'group1_name': group1_name,
        'group2_name': group2_name,
        'group1_mean': g1_mean,
        'group1_std': g1.std(),
        'group1_n': len(g1),
        'group2_mean': g2_mean,
        'group2_std': g2.std(),
        'group2_n': len(g2),
        'difference': difference,
        'percent_difference': (difference / g2_mean * 100) if g2_mean != 0 else 0,
        'test': 'Mann-Whitney U',
        'statistic': stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': significant,
        'interpretation': interpretation
    }

def analyze_biotype_enrichment(group_biotypes, reference_biotypes, group_name, reference_name):
    """
    Test if specific biotypes are enriched in one group vs reference.
    Uses chi-squared for large samples, Fisher's exact for small samples.
    """
    group_counts = Counter(group_biotypes)
    reference_counts = Counter(reference_biotypes)
    
    all_biotypes = set(group_counts.keys()) | set(reference_counts.keys())
    
    results = []
    for biotype in all_biotypes:
        # Contingency table
        in_group_with_biotype = group_counts.get(biotype, 0)
        in_group_without_biotype = len(group_biotypes) - in_group_with_biotype
        in_ref_with_biotype = reference_counts.get(biotype, 0)
        in_ref_without_biotype = len(reference_biotypes) - in_ref_with_biotype
        
        # Skip if too few samples
        if in_group_with_biotype < 5:
            continue
        
        contingency = np.array([
            [in_group_with_biotype, in_group_without_biotype],
            [in_ref_with_biotype, in_ref_without_biotype]
        ])
        
        # Choose test based on sample size
        # Use chi-squared if all expected frequencies > 5
        expected = contingency.sum(axis=0) * contingency.sum(axis=1)[:, None] / contingency.sum()
        
        if (expected >= 5).all():
            # Chi-squared test (more powerful for large samples)
            chi2, p_value, dof, _ = chi2_contingency(contingency)
            test_used = 'chi-squared'
            odds_ratio = (contingency[0,0] * contingency[1,1]) / (contingency[0,1] * contingency[1,0]) if contingency[0,1] > 0 and contingency[1,0] > 0 else np.inf
        else:
            # Fisher's exact test (for small samples)
            odds_ratio, p_value = fisher_exact(contingency)
            test_used = 'fisher'
        
        # Calculate rates
        group_rate = in_group_with_biotype / len(group_biotypes)
        ref_rate = in_ref_with_biotype / len(reference_biotypes)
        fold_enrichment = group_rate / ref_rate if ref_rate > 0 else np.inf
        
        results.append({
            'biotype': biotype,
            'group_name': group_name,
            'reference_name': reference_name,
            'n_in_group': in_group_with_biotype,
            'n_in_reference': in_ref_with_biotype,
            'group_rate': group_rate,
            'reference_rate': ref_rate,
            'fold_enrichment': fold_enrichment,
            'log2_enrichment': np.log2(fold_enrichment) if fold_enrichment > 0 and np.isfinite(fold_enrichment) else np.nan,
            'odds_ratio': odds_ratio,
            'test_used': test_used,  # NEW: track which test was used
            'p_value': p_value,
            'significant': p_value < 0.05
        })
    
    return pd.DataFrame(results).sort_values('p_value')

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_sequence_feature_plots(categories_df, output_dir, model1_name, model2_name):
    """Create comprehensive sequence feature comparison plots."""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    categories = ['both_correct', f'{model1_name.lower()}_only', f'{model2_name.lower()}_only', 'both_wrong']
    category_labels = ['Both Correct', f'{model1_name} Only', f'{model2_name} Only', 'Both Wrong']
    colors = ['#2ECC71', '#3498DB', '#E74C3C', '#95A5A6']
    
    # Row 1: Length distributions
    for i, (cat, label, color) in enumerate(zip(categories, category_labels, colors)):
        ax = fig.add_subplot(gs[0, i])
        
        cat_data = categories_df[categories_df['performance_category'] == cat]
        
        if len(cat_data) > 0:
            ax.hist(cat_data['length'].dropna(), bins=50, color=color, alpha=0.7, edgecolor='black')
            ax.axvline(cat_data['length'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {cat_data["length"].mean():.0f}')
            ax.set_xlabel('Sequence Length (nt)', fontsize=9)
            ax.set_ylabel('Count', fontsize=9)
            ax.set_xlim(0, 10000)
            ax.set_title(f'{label}\n(n={len(cat_data):,})', fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    
    # Row 2: GC content distributions
    for i, (cat, label, color) in enumerate(zip(categories, category_labels, colors)):
        ax = fig.add_subplot(gs[1, i])
        
        cat_data = categories_df[categories_df['performance_category'] == cat]
        
        if len(cat_data) > 0:
            ax.hist(cat_data['gc_content'].dropna() * 100, bins=50, color=color, alpha=0.7, edgecolor='black')
            ax.axvline(cat_data['gc_content'].mean() * 100, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {cat_data["gc_content"].mean()*100:.1f}%')
            ax.set_xlabel('GC Content (%)', fontsize=9)
            ax.set_ylabel('Count', fontsize=9)
            ax.set_title(f'{label}', fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    
    # Row 3: Class distribution
    for i, (cat, label, color) in enumerate(zip(categories, category_labels, colors)):
        ax = fig.add_subplot(gs[2, i])
        
        cat_data = categories_df[categories_df['performance_category'] == cat]
        
        if len(cat_data) > 0:
            class_counts = cat_data['true_class'].value_counts()
            ax.bar(range(len(class_counts)), class_counts.values, color=[color], alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(class_counts)))
            ax.set_xticklabels(class_counts.index, fontsize=9)
            ax.set_ylabel('Count', fontsize=9)
            ax.set_title(f'{label}', fontsize=10, fontweight='bold')
            
            # Add percentage labels
            for j, (cls, count) in enumerate(class_counts.items()):
                pct = count / len(cat_data) * 100
                ax.text(j, count + len(cat_data)*0.02, f'{pct:.1f}%', ha='center', fontsize=8, fontweight='bold')
            
            ax.grid(alpha=0.3, axis='y')
    
    plt.suptitle(f'Sequence Feature Analysis by Performance Category\n{model1_name} vs {model2_name}', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(output_dir / 'sequence_features_by_category.png', bbox_inches='tight', dpi=1200)
    plt.close()
    print(f"  Saved: sequence_features_by_category.png")

def create_biotype_enrichment_plot(enrichment_df, output_dir, model1_name, model2_name):
    """Create biotype enrichment visualization."""
    
    # Filter to significant and top biotypes
    sig_df = enrichment_df[enrichment_df['significant']].copy()
    
    if len(sig_df) == 0:
        print("  No significant biotype enrichments found")
        return
    
    # Take top 20 by fold enrichment
    top_df = sig_df.nlargest(20, 'fold_enrichment')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Panel A: Fold enrichment
    ax = axes[0]
    
    y_pos = np.arange(len(top_df))
    colors = ['#E74C3C' if x > 1 else '#3498DB' for x in top_df['fold_enrichment']]
    
    ax.barh(y_pos, top_df['fold_enrichment'], color=colors, alpha=0.7, edgecolor='black')
    ax.axvline(1.0, color='black', linestyle='--', linewidth=2, label='No enrichment')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_df['biotype'], fontsize=9)
    ax.set_xlabel('Fold Enrichment', fontsize=11, fontweight='bold')
    ax.set_title(f'A. Biotype Enrichment in {top_df.iloc[0]["group_name"]}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='x')
    
    # Add p-value stars
    for i, (idx, row) in enumerate(top_df.iterrows()):
        p = row['p_value']
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*'
        x_pos = row['fold_enrichment'] + 0.1
        ax.text(x_pos, i, sig, fontsize=12, fontweight='bold', va='center')
    
    # Panel B: Rates comparison
    ax = axes[1]
    
    x = np.arange(len(top_df))
    width = 0.35
    
    ax.bar(x - width/2, top_df['group_rate'] * 100, width, label=top_df.iloc[0]['group_name'], 
           color='#E74C3C', alpha=0.7, edgecolor='black')
    ax.bar(x + width/2, top_df['reference_rate'] * 100, width, label=top_df.iloc[0]['reference_name'],
           color='#3498DB', alpha=0.7, edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(top_df['biotype'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Prevalence (%)', fontsize=11, fontweight='bold')
    ax.set_title('B. Biotype Prevalence Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'biotype_enrichment_analysis.png', bbox_inches='tight', dpi=1200)
    plt.close()
    print(f"  Saved: biotype_enrichment_analysis.png")

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced cross-model comparison with deep statistical analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model 1 arguments
    parser.add_argument('--model1_predictions', default='gencode_v47_experiments/beta_vae_features_g47/evaluation_csvs/all_sample_predictions.csv', help='Model 1 predictions CSV')
    parser.add_argument('--model1_name', default='G47', help='Model 1 name (e.g., G47)')
    parser.add_argument('--model1_biotypes', default='data/dataset_biotypes/g47_dataset_biotypes_cdhit.csv', help='Model 1 biotypes CSV')
    parser.add_argument('--model1_lnc_fasta', default='data/cdhit_clusters/g47_lncRNA_clustered.fa', help='Model 1 lncRNA FASTA')
    parser.add_argument('--model1_pc_fasta', default='data/cdhit_clusters/g47_pc_clustered.fa', help='Model 1 protein-coding FASTA')
    
    # Model 2 arguments
    parser.add_argument('--model2_predictions', default='gencode_v49_experiments/beta_vae_features_g49/evaluation_csvs/all_sample_predictions.csv', help='Model 2 predictions CSV')
    parser.add_argument('--model2_name', default='G49', help='Model 2 name (e.g., G49)')
    parser.add_argument('--model2_biotypes', default='data/dataset_biotypes/g49_dataset_biotypes_cdhit.csv', help='Model 2 biotypes CSV')
    parser.add_argument('--model2_lnc_fasta', default='data/cdhit_clusters/g49_lncRNA_clustered.fa', help='Model 2 lncRNA FASTA')
    parser.add_argument('--model2_pc_fasta', default='data/cdhit_clusters/g49_pc_clustered.fa', help='Model 2 protein-coding FASTA')
    
    # Shared transcripts
    parser.add_argument('--common_fasta', default='data/gencode_comparison_v47_vs_v49/gencode.v49.common_no_class_change_transcripts.fa', help='Common transcripts FASTA')
    
    # Output
    parser.add_argument('--output_dir', default='cross_model_deep_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print(f"ENHANCED CROSS-MODEL COMPARISON: {args.model1_name} vs {args.model2_name}")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Load all data
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: Loading Data")
    print("="*80)
    
    # Load common transcript IDs
    print("\nLoading common transcript IDs...")
    common_ids = parse_fasta_ids(args.common_fasta)
    print(f"  Common transcripts: {len(common_ids):,}")
    
    # Load predictions
    model1_df = load_predictions(args.model1_predictions, args.model1_name)
    model2_df = load_predictions(args.model2_predictions, args.model2_name)
    
    # Load sequence features
    model1_seq_features = load_sequence_features(args.model1_lnc_fasta, args.model1_pc_fasta, args.model1_name)
    model2_seq_features = load_sequence_features(args.model2_lnc_fasta, args.model2_pc_fasta, args.model2_name)
    
    # Load biotypes
    model1_biotypes = load_biotypes(args.model1_biotypes, args.model1_name)
    model2_biotypes = load_biotypes(args.model2_biotypes, args.model2_name)
    
    # ========================================================================
    # STEP 2: Filter to shared transcripts and merge
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: Merging Shared Transcripts")
    print("="*80)
    
    model1_common = model1_df[model1_df['transcript_id_clean'].isin(common_ids)].copy()
    model2_common = model2_df[model2_df['transcript_id_clean'].isin(common_ids)].copy()
    
    print(f"\n  {args.model1_name} common: {len(model1_common):,}")
    print(f"  {args.model2_name} common: {len(model2_common):,}")
    
    # Merge
    merged = model1_common.merge(
        model2_common,
        on='transcript_id_clean',
        suffixes=(f'_{args.model1_name.lower()}', f'_{args.model2_name.lower()}')
    )
    
    print(f"\n  Shared transcripts in both models: {len(merged):,}")
    
    # ========================================================================
    # STEP 3: Categorize performance
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: Categorizing Performance")
    print("="*80)
    
    merged['correct_m1'] = (merged[f'consensus_prediction_{args.model1_name.lower()}'] == merged[f'true_label_{args.model1_name.lower()}'])
    merged['correct_m2'] = (merged[f'consensus_prediction_{args.model2_name.lower()}'] == merged[f'true_label_{args.model2_name.lower()}'])
    
    merged['performance_category'] = 'both_correct'
    merged.loc[merged['correct_m1'] & ~merged['correct_m2'], 'performance_category'] = f'{args.model1_name.lower()}_only'
    merged.loc[~merged['correct_m1'] & merged['correct_m2'], 'performance_category'] = f'{args.model2_name.lower()}_only'
    merged.loc[~merged['correct_m1'] & ~merged['correct_m2'], 'performance_category'] = 'both_wrong'
    
    category_counts = merged['performance_category'].value_counts()
    
    print("\n  Performance Categories:")
    for cat, count in category_counts.items():
        pct = count / len(merged) * 100
        print(f"    {cat:20s}: {count:7,} ({pct:5.2f}%)")
    
    # ========================================================================
    # STEP 4: Add sequence features and biotypes
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: Adding Sequence Features and Biotypes")
    print("="*80)
    
    # Use model2 sequence features (both should be same for common transcripts)
    merged['length'] = merged['transcript_id_clean'].map(lambda x: model2_seq_features.get(x, {}).get('length', np.nan))
    merged['gc_content'] = merged['transcript_id_clean'].map(lambda x: model2_seq_features.get(x, {}).get('gc_content', np.nan))
    merged['true_class'] = merged['transcript_id_clean'].map(lambda x: model2_seq_features.get(x, {}).get('true_class', 'unknown'))
    merged['biotype_m1'] = merged['transcript_id_clean'].map(lambda x: model1_biotypes.get(x, 'unknown'))
    merged['biotype_m2'] = merged['transcript_id_clean'].map(lambda x: model2_biotypes.get(x, 'unknown'))
    
    print(f"\n  Added sequence features for {(~merged['length'].isna()).sum():,} transcripts")
    print(f"  Added {args.model1_name} biotypes for {(merged['biotype_m1'] != 'unknown').sum():,} transcripts")
    print(f"  Added {args.model2_name} biotypes for {(merged['biotype_m2'] != 'unknown').sum():,} transcripts")
    
    # ========================================================================
    # STEP 5: Statistical comparisons
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: Statistical Analysis")
    print("="*80)
    
    # Reference group: both_correct
    reference_data = merged[merged['performance_category'] == 'both_correct']
    
    comparison_results = []
    
    for cat in [f'{args.model1_name.lower()}_only', f'{args.model2_name.lower()}_only', 'both_wrong']:
        cat_data = merged[merged['performance_category'] == cat]
        
        if len(cat_data) < 10:
            continue
        
        print(f"\n  Analyzing: {cat} vs both_correct")
        
        # Length comparison
        length_result = compare_distributions(
            cat_data['length'],
            reference_data['length'],
            'sequence_length',
            cat,
            'both_correct'
        )
        comparison_results.append(length_result)
        print(f"    Length: {length_result['interpretation']}")
        
        # GC content comparison
        gc_result = compare_distributions(
            cat_data['gc_content'],
            reference_data['gc_content'],
            'gc_content',
            cat,
            'both_correct'
        )
        comparison_results.append(gc_result)
        print(f"    GC: {gc_result['interpretation']}")
    
    # Save comparison results
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(output_dir / 'statistical_comparisons.csv', index=False)
    print(f"\n  Saved statistical comparisons to: statistical_comparisons.csv")
    
    # ========================================================================
    # STEP 6: Biotype enrichment analysis
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: Biotype Enrichment Analysis")
    print("="*80)
    
    # Use model2 biotypes (should be same for common transcripts)
    reference_biotypes = reference_data['biotype_m2'].tolist()
    
    all_enrichment_results = []
    
    for cat in [f'{args.model1_name.lower()}_only', f'{args.model2_name.lower()}_only', 'both_wrong']:
        cat_data = merged[merged['performance_category'] == cat]
        
        if len(cat_data) < 10:
            continue
        
        print(f"\n  Analyzing biotype enrichment: {cat}")
        
        cat_biotypes = cat_data['biotype_m2'].tolist()
        enrichment_df = analyze_biotype_enrichment(cat_biotypes, reference_biotypes, cat, 'both_correct')
        
        if len(enrichment_df) > 0:
            print(f"    Found {len(enrichment_df[enrichment_df['significant']]):,} significantly enriched biotypes")
            all_enrichment_results.append(enrichment_df)
            
            # Print top 5
            top5 = enrichment_df.head(5)
            for _, row in top5.iterrows():
                print(f"      {row['biotype']:30s}: {row['fold_enrichment']:6.2f}× (p={row['p_value']:.4f})")
    
    # Save all enrichment results
    if all_enrichment_results:
        all_enrichment_df = pd.concat(all_enrichment_results, ignore_index=True)
        all_enrichment_df.to_csv(output_dir / 'biotype_enrichment_results.csv', index=False)
        print(f"\n  Saved biotype enrichment results to: biotype_enrichment_results.csv")
    
    # ========================================================================
    # STEP 7: Visualizations
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 7: Creating Visualizations")
    print("="*80)
    
    create_sequence_feature_plots(merged, output_dir, args.model1_name, args.model2_name)
    
    if all_enrichment_results:
        # Create separate plots for each category
        for enrichment_df in all_enrichment_results:
            if len(enrichment_df[enrichment_df['significant']]) > 0:
                group_name = enrichment_df.iloc[0]['group_name']
                create_biotype_enrichment_plot(enrichment_df, output_dir, args.model1_name, args.model2_name)
    
    # ========================================================================
    # STEP 8: Save categorized transcripts
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 8: Saving Categorized Transcripts")
    print("="*80)
    
    for cat in merged['performance_category'].unique():
        cat_data = merged[merged['performance_category'] == cat]
        filename = f'{cat}_transcripts.csv'
        cat_data.to_csv(output_dir / filename, index=False)
        print(f"  Saved: {filename} ({len(cat_data):,} transcripts)")
    
    # ========================================================================
    # STEP 9: Generate summary report
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    summary_text = f"""
# Cross-Model Comparison: {args.model1_name} vs {args.model2_name}

## Overall Performance

Total shared transcripts: {len(merged):,}

Performance breakdown:
- Both correct: {category_counts.get('both_correct', 0):,} ({category_counts.get('both_correct', 0)/len(merged)*100:.2f}%)
- {args.model1_name} only correct: {category_counts.get(f'{args.model1_name.lower()}_only', 0):,} ({category_counts.get(f'{args.model1_name.lower()}_only', 0)/len(merged)*100:.2f}%)
- {args.model2_name} only correct: {category_counts.get(f'{args.model2_name.lower()}_only', 0):,} ({category_counts.get(f'{args.model2_name.lower()}_only', 0)/len(merged)*100:.2f}%)
- Both wrong: {category_counts.get('both_wrong', 0):,} ({category_counts.get('both_wrong', 0)/len(merged)*100:.2f}%)

{args.model1_name} accuracy on shared: {merged['correct_m1'].mean()*100:.2f}%
{args.model2_name} accuracy on shared: {merged['correct_m2'].mean()*100:.2f}%
Difference: {(merged['correct_m2'].mean() - merged['correct_m1'].mean())*100:.2f} percentage points

## Sequence Feature Analysis

### {args.model1_name}-only correct transcripts:
Mean length: {merged[merged['performance_category']==f'{args.model1_name.lower()}_only']['length'].mean():.1f} nt
Mean GC: {merged[merged['performance_category']==f'{args.model1_name.lower()}_only']['gc_content'].mean()*100:.2f}%

### {args.model2_name}-only correct transcripts:
Mean length: {merged[merged['performance_category']==f'{args.model2_name.lower()}_only']['length'].mean():.1f} nt
Mean GC: {merged[merged['performance_category']==f'{args.model2_name.lower()}_only']['gc_content'].mean()*100:.2f}%

### Both correct transcripts:
Mean length: {merged[merged['performance_category']=='both_correct']['length'].mean():.1f} nt
Mean GC: {merged[merged['performance_category']=='both_correct']['gc_content'].mean()*100:.2f}%

## Files Generated

- statistical_comparisons.csv: Detailed statistical test results
- biotype_enrichment_results.csv: Biotype enrichment analysis
- sequence_features_by_category.png: Comprehensive sequence feature visualizations
- biotype_enrichment_analysis.png: Biotype enrichment plots
- *_transcripts.csv: Individual files for each performance category

## Interpretation

See statistical_comparisons.csv for detailed p-values and effect sizes.
See biotype_enrichment_results.csv for significantly enriched biotypes in each category.
"""
    
    with open(output_dir / 'SUMMARY.md', 'w') as f:
        f.write(summary_text)
    
    print(summary_text)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}/")

if __name__ == '__main__':
    main()