#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize beta-VAE embeddings with UMAP

Generates comprehensive UMAP visualizations and CSVs for:
1. Binary classification (lncRNA vs Protein-coding)
2. Biotype distribution (if biotypes provided)
3. Hard case identification

Compatible with embeddings from evaluate_cv_folds.py (with --extract_all_folds)

When using --per_fold, UMAP is computed INDEPENDENTLY for each fold
to avoid fold-separation artifacts that occur when concatenating all folds first.

Usage:
    # Visualize per-fold
    python visualize_embeddings.py \
        --embeddings gencode_v47_experiments/beta_vae_g47/embeddings_all_folds.npz \
        --output_dir umap_visualizations \
        --per_fold
    
    # With hard cases CSV from evaluate_cv_folds.py
    python visualize_embeddings.py \
        --embeddings gencode_v47_experiments/beta_vae_g47/embeddings_all_folds.npz \
        --hard_cases gencode_v47_experiments/beta_vae_g47/evaluation_csvs/hard_cases.csv \
        --output_dir umap_visualizations \
        --per_fold
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import umap
import json

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100  # Display only, savefig uses 1200

def detect_model_type(experiment_dir):
    """Detect if experiment used biotype or features model"""
    config_path = Path(experiment_dir) / 'config.json'
    
    if not config_path.exists():
        return 'unknown'
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Check for features-specific fields
    if 'te_features_csv' in config.get('data', {}):
        return 'features'
    elif 'architecture' in config.get('model', {}):
        arch = config['model']['architecture']
        if 'features' in arch.lower():
            return 'features'
    
    return 'biotype'

def load_feature_metadata(experiment_dir):
    """
    Load TE and Non-B feature data for features-based models
    
    Returns dict with transcript_id -> {te_density, nonb_density}
    """
    config_path = Path(experiment_dir) / 'config.json'
    
    if not config_path.exists():
        return None
    
    with open(config_path) as f:
        config = json.load(f)
    
    te_csv = config.get('data', {}).get('te_features_csv')
    nonb_csv = config.get('data', {}).get('nonb_features_csv')
    
    if not te_csv or not nonb_csv:
        return None
    
    print(f"\nLoading feature metadata for features model...")
    
    try:
        import pandas as pd
        
        te_df = pd.read_csv(te_csv, index_col='transcript_id')
        nonb_df = pd.read_csv(nonb_csv, index_col='transcript_id')
        
        # Compute density metrics
        feature_meta = {}
        
        for transcript_id in te_df.index:
            te_density = te_df.loc[transcript_id].values.sum()  # Total TE content
            nonb_density = nonb_df.loc[transcript_id].values.sum()  # Total Non-B content
            
            feature_meta[transcript_id] = {
                'te_density': te_density,
                'nonb_density': nonb_density
            }
        
        print(f"  Loaded feature metadata for {len(feature_meta):,} transcripts")
        
        return feature_meta
    
    except Exception as e:
        print(f"  Could not load feature metadata: {e}")
        return None

def load_embeddings(embeddings_path):
    """
    Load embeddings from npz file
    
    Returns embeddings WITHOUT concatenating - keeps fold structure intact
    """
    print(f"Loading embeddings from: {embeddings_path}")
    data = np.load(embeddings_path, allow_pickle=True)
    
    keys = list(data.keys())
    print(f"  Available keys: {keys[:5]}... ({len(keys)} total)")
    
    # Check for fold-based format
    fold_keys = [k for k in keys if k.startswith('fold_') and '_embeddings' in k]
    
    if fold_keys:
        # Multi-fold format - return fold structure, DON'T concatenate yet
        print(f"  Format: Multi-fold (fold_N_* structure)")
        
        fold_nums = sorted(set([int(k.split('_')[1]) for k in fold_keys]))
        print(f"  Found {len(fold_nums)} folds: {fold_nums}")
        
        # Store per-fold data
        folds_data = {}
        
        for fold_num in fold_nums:
            prefix = f'fold_{fold_num}_'
            
            fold_embeddings = data[f'{prefix}embeddings']
            fold_labels = data[f'{prefix}labels']
            fold_predictions = data[f'{prefix}predictions']
            fold_probs = data[f'{prefix}probs']
            fold_transcript_ids = data[f'{prefix}transcript_ids']
            #fold_biotype_indices = data[f'{prefix}biotype_indices']

            # Try to load biotypes (strings) first, fallback to biotype_indices (integers)
            if f'{prefix}biotypes' in data:
                fold_biotypes = data[f'{prefix}biotypes']
            elif f'{prefix}biotype_indices' in data:
                # Old format - has indices instead of strings
                fold_biotype_indices = data[f'{prefix}biotype_indices']
                fold_biotypes = np.array([f'biotype_{idx}' for idx in fold_biotype_indices])
            else:
                raise KeyError(f"Neither biotypes nor biotype_indices found for {prefix}")
            
            folds_data[fold_num] = {
                'embeddings': fold_embeddings,
                'labels': fold_labels,
                'predictions': fold_predictions,
                'probs': fold_probs,
                'transcript_ids': fold_transcript_ids,
                'biotypes': fold_biotypes,  # <-- Already loaded as strings above
                'fold_id': fold_num
            }
            
            print(f"  Fold {fold_num}: {len(fold_embeddings):,} samples")
        
        return {
            'format': 'multi_fold',
            'folds': folds_data,
            'n_folds': len(fold_nums)
        }
    
    else:
        raise ValueError(f"Unknown embeddings format! Expected fold_N_* keys. "
                        f"Found: {keys[:10]}...")


def compute_umap(embeddings, n_neighbors=30, min_dist=0.1, metric='euclidean', random_seed=42):
    """Compute UMAP projection"""
    print(f"  Computing UMAP...")
    print(f"    n_neighbors: {n_neighbors}, min_dist: {min_dist}, metric: {metric}")
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_seed,
        n_jobs=-1
    )
    
    umap_embedding = reducer.fit_transform(embeddings)
    
    print(f"    UMAP shape: {umap_embedding.shape}")
    print(f"    Range: [{umap_embedding.min():.2f}, {umap_embedding.max():.2f}]")
    
    return umap_embedding


def load_biotype_mapping_from_csv(biotype_csv, lnc_fasta=None, pc_fasta=None):
    """Load biotype mapping from CSV to convert indices to names"""
    if biotype_csv is None:
        return None
    
    print(f"\nLoading biotype mapping from: {biotype_csv}")
    
    try:
        import sys
        from pathlib import Path
        
        from data.cv_utils import (
            load_biotype_mapping,
            extract_biotypes_from_sequences,
            group_rare_biotypes
        )
        from data.loaders import load_sequences_with_labels
        
        print("  Note: Loading full dataset to create biotype mapping...")
        
        if lnc_fasta is None or pc_fasta is None:
            print("  WARNING: lncRNA or protein-coding FASTA paths not provided, "
                  "using default paths (may fail if files not found)")
            default_lnc = '${DATA_ROOT}/data/cdhit_clusters/g47_lncRNA_clustered.fa'
            default_pc = '${DATA_ROOT}/data/cdhit_clusters/g47_pc_clustered.fa'
        
            if Path(default_lnc).exists() and Path(default_pc).exists():
                sequences, _ = load_sequences_with_labels(default_lnc, default_pc)
                biotype_lookup = load_biotype_mapping(biotype_csv)
                biotypes = extract_biotypes_from_sequences(sequences, biotype_lookup)
                
                # CRITICAL: Must match the grouping used during embedding creation!
                # evaluate_cv_folds_enhanced.py uses group_rare_biotypes with min_count=500
                grouped_biotypes = group_rare_biotypes(biotypes, min_count=500)
                
                unique_biotypes = sorted(set(grouped_biotypes))
                idx_to_biotype = {idx: bt for idx, bt in enumerate(unique_biotypes)}
                
                print(f"    Created mapping for {len(idx_to_biotype)} biotypes")
                
                # Show biotype distribution
                from collections import Counter
                biotype_counts = Counter(grouped_biotypes)
                print(f"  Biotype distribution:")
                for bt, count in biotype_counts.most_common(10):
                    print(f"    {bt}: {count:,}")
                return idx_to_biotype
            else:
                print(f"    Could not find FASTA files at default paths")
                return None
        else:
            sequences, _ = load_sequences_with_labels(lnc_fasta, pc_fasta)
            biotype_lookup = load_biotype_mapping(biotype_csv)
            biotypes = extract_biotypes_from_sequences(sequences, biotype_lookup)
            
            # CRITICAL: Must match the grouping used during embedding creation!
            # evaluate_cv_folds_enhanced.py uses group_rare_biotypes with min_count=500
            grouped_biotypes = group_rare_biotypes(biotypes, min_count=500)
            
            unique_biotypes = sorted(set(grouped_biotypes))
            idx_to_biotype = {idx: bt for idx, bt in enumerate(unique_biotypes)}
            
            print(f"    Created mapping for {len(idx_to_biotype)} biotypes")
            
            # Show biotype distribution
            from collections import Counter
            biotype_counts = Counter(grouped_biotypes)
            print(f"  Biotype distribution:")
            for bt, count in biotype_counts.most_common(10):
                print(f"    {bt}: {count:,}")
            return idx_to_biotype
    
    except Exception as e:
        print(f"    Could not load biotype mapping: {e}")
        return None


def convert_biotype_indices(biotype_array, idx_to_biotype):
    """Convert biotype index strings to actual biotype names"""
    if idx_to_biotype is None:
        return biotype_array
    
    converted = []
    for b in biotype_array:
        if isinstance(b, str) and b.startswith('biotype_'):
            try:
                idx = int(b.split('_')[1])
                converted.append(idx_to_biotype.get(idx, b))
            except:
                converted.append(b)
        else:
            converted.append(str(b))
    
    return np.array(converted)


def load_hard_cases(hard_cases_path):
    """Load hard cases CSV"""
    if hard_cases_path is None:
        return None
    
    print(f"\nLoading hard cases from: {hard_cases_path}")
    df = pd.read_csv(hard_cases_path)
    
    if 'transcript_id' in df.columns:
        hard_case_ids = set(df['transcript_id'].values)
    elif 'id' in df.columns:
        hard_case_ids = set(df['id'].values)
    else:
        print(f"  WARNING: Could not find transcript ID column")
        return None
    
    print(f"  Loaded {len(hard_case_ids):,} hard case IDs")
    return hard_case_ids


def identify_hard_cases(labels, predictions, transcript_ids, hard_case_ids=None):
    """Identify hard cases based on prediction mismatch and/or provided IDs"""
    
    is_hard_mismatch = labels != predictions
    
    if hard_case_ids is not None:
        # Clean transcript IDs
        clean_ids_emb = np.array([tid.split('|')[0] for tid in transcript_ids])
        clean_ids_no_version = np.array([tid.split('.')[0] for tid in clean_ids_emb])
        
        clean_hard_ids = set()
        clean_hard_ids_no_version = set()
        
        for hid in hard_case_ids:
            base_id = hid.split('|')[0]
            clean_hard_ids.add(base_id)
            clean_hard_ids_no_version.add(base_id.split('.')[0])
        
        is_hard_ids = np.array([
            (tid in clean_hard_ids) or (tid_no_ver in clean_hard_ids_no_version)
            for tid, tid_no_ver in zip(clean_ids_emb, clean_ids_no_version)
        ])
        
        is_hard = is_hard_mismatch | is_hard_ids
        
        print(f"    Hard case identification:")
        print(f"      From prediction mismatch: {is_hard_mismatch.sum():,}")
        print(f"      From provided IDs: {is_hard_ids.sum():,}")
        print(f"      Total: {is_hard.sum():,} ({100*is_hard.sum()/len(labels):.2f}%)")
    else:
        is_hard = is_hard_mismatch
        print(f"    Hard cases (prediction mismatch): {is_hard.sum():,} ({100*is_hard.sum()/len(labels):.2f}%)")
    
    return is_hard


def visualize_umap(umap_embedding, labels, predictions, biotypes, is_hard, output_dir, fold_id, fig_tag=None):
    """Create UMAP visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    label_names = ['lncRNA', 'Protein-coding']
    
    # 1. All samples by class
    print("    Generating visualization: All samples by class...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for label_idx, label_name in enumerate(label_names):
        mask = labels == label_idx
        color = '#FF6B6B' if label_idx == 0 else '#4ECDC4'
        
        ax.scatter(umap_embedding[mask, 0], umap_embedding[mask, 1],
                  c=color, s=2, alpha=0.3, label=label_name, rasterized=True)
    
    ax.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
    ax.set_title(f'{fig_tag}Fold {fold_id}: β-VAE Embeddings (All Samples)\nn={len(labels):,}', 
                fontsize=16, fontweight='bold')
    ax.legend(markerscale=5, loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'umap_all_samples.png', dpi=350, bbox_inches='tight')
    plt.close()
    
    # 2. Hard cases highlighted
    print("    Generating visualization: Hard cases...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    easy_mask = ~is_hard
    ax.scatter(umap_embedding[easy_mask, 0], umap_embedding[easy_mask, 1],
              c='lightgray', s=1, alpha=0.2, label='Easy cases', rasterized=True)
    
    hard_lnc = is_hard & (labels == 0)
    hard_pc = is_hard & (labels == 1)
    
    ax.scatter(umap_embedding[hard_lnc, 0], umap_embedding[hard_lnc, 1],
              c='#FF6B6B', s=8, alpha=0.7, edgecolors='black', linewidths=0.3,
              label=f'Hard lncRNA (n={hard_lnc.sum():,})', rasterized=True)
    
    ax.scatter(umap_embedding[hard_pc, 0], umap_embedding[hard_pc, 1],
              c='#4ECDC4', s=8, alpha=0.7, edgecolors='black', linewidths=0.3,
              label=f'Hard PC (n={hard_pc.sum():,})', rasterized=True)
    
    ax.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
    ax.set_title(f'{fig_tag}Fold {fold_id}: Hard Cases (n={is_hard.sum():,})', 
                fontsize=16, fontweight='bold')
    ax.legend(markerscale=2, loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'umap_hard_cases.png', dpi=350, bbox_inches='tight')
    plt.close()
    
    # 3. Biotype distribution (if meaningful)
    unique_biotypes = np.unique(biotypes)
    all_generic = all(bt.startswith('biotype_') or bt == 'unknown' for bt in unique_biotypes)
    
    if not all_generic:
        print("    Generating visualization: Biotypes...")
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Get biotype counts
        biotype_counts_dict = {}
        for bt in unique_biotypes:
            count = (biotypes == bt).sum()
            if count > 0:  # Only include biotypes that actually exist
                biotype_counts_dict[bt] = count
        
        # Sort by count
        sorted_biotypes_items = sorted(biotype_counts_dict.items(), 
                                       key=lambda x: x[1], reverse=True)
        
        # Show top 8 by count (rare biotypes already grouped into 'other' during index creation)
        n_to_show = min(8, len(sorted_biotypes_items))
        selected_biotypes = [bt for bt, _ in sorted_biotypes_items[:n_to_show]]
        
        print(f"      Showing top {n_to_show} biotypes:")
        for bt in selected_biotypes:
            print(f"        {bt}: {biotype_counts_dict[bt]:,}")
        
        # Plot selected biotypes
        colors = sns.color_palette('husl', len(selected_biotypes))
        for biotype, color in zip(selected_biotypes, colors):
            mask = biotypes == biotype
            count = mask.sum()
            ax.scatter(umap_embedding[mask, 0], umap_embedding[mask, 1],
                      c=[color], s=3, alpha=0.6, label=f'{biotype} (n={count:,})',
                      rasterized=True)
        
        ax.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
        ax.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
        ax.set_title(f'{fig_tag}Fold {fold_id}: Biotype Distribution', fontsize=16, fontweight='bold')
        ax.legend(markerscale=3, loc='best', fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'umap_by_biotype.png', dpi=350, bbox_inches='tight')
        plt.close()

        # Minor biotype UMAP
        visualize_minor_biotypes_only(umap_embedding, biotypes, output_dir, fold_id, fig_tag=fig_tag)

    else:
        print("    Skipping biotype visualization (generic labels)")

def visualize_minor_biotypes_only(umap_embedding, biotypes, output_dir, fold_id, fig_tag):
    """
    Create UMAP visualization highlighting ONLY minor biotypes.
    Major biotypes (lncRNA, protein_coding) shown as gray background.
    
    This makes it easy to see clustering patterns of rare/interesting biotypes.
    """
    unique_biotypes = np.unique(biotypes)
    
    # Define major biotypes to exclude from coloring
    major_biotypes = {'lncRNA', 'protein_coding'}
    
    # Identify minor biotypes
    minor_biotypes = [bt for bt in unique_biotypes 
                     if bt not in major_biotypes and bt != 'other']
    
    if len(minor_biotypes) == 0:
        print("      No minor biotypes found (only lncRNA/protein_coding)")
        return
    
    print("    Generating visualization: Minor biotypes only...")
    
    # Count minor biotypes
    minor_counts = {bt: (biotypes == bt).sum() for bt in minor_biotypes}
    sorted_minor = sorted(minor_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Show all minor biotypes (or top 15 if too many)
    n_minor_show = min(15, len(sorted_minor))
    minor_to_show = [bt for bt, _ in sorted_minor[:n_minor_show]]
    
    print(f"      Found {len(minor_biotypes)} minor biotypes, showing top {n_minor_show}:")
    for bt in minor_to_show:
        print(f"        {bt}: {minor_counts[bt]:,}")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot major biotypes as gray background
    major_mask = np.isin(biotypes, list(major_biotypes))
    ax.scatter(umap_embedding[major_mask, 0], umap_embedding[major_mask, 1],
              c='lightgray', s=1, alpha=0.15, label='lncRNA/protein_coding',
              rasterized=True)
    
    # Plot each minor biotype in color
    colors_minor = sns.color_palette('husl', len(minor_to_show))
    for biotype, color in zip(minor_to_show, colors_minor):
        mask = biotypes == biotype
        count = mask.sum()
        ax.scatter(umap_embedding[mask, 0], umap_embedding[mask, 1],
                  c=[color], s=15, alpha=0.8, edgecolors='black', linewidths=0.3,
                  label=f'{biotype} (n={count:,})', rasterized=True)
    
    ax.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
    ax.set_title(f'{fig_tag}Fold {fold_id}: Minor Biotype Clustering\n'
                f'(lncRNA/protein_coding shown as gray background)',
                fontsize=16, fontweight='bold')
    ax.legend(markerscale=1.5, loc='best', fontsize=10, ncol=2, 
             framealpha=0.9, edgecolor='black')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'umap_minor_biotypes_only.png', dpi=350, bbox_inches='tight')
    plt.close()
    
    print(f"      Saved: umap_minor_biotypes_only.png")

def visualize_feature_overlays(umap_embedding, transcript_ids, feature_meta, 
                               output_dir, fold_id, fig_tag=None):
    """
    Create UMAP visualizations with TE/Non-B density overlays
    (Only for features-based models)
    """
    if feature_meta is None:
        return
    
    print("    Generating visualization: Feature density overlays...")
    
    # Extract densities for this fold's transcripts
    te_densities = []
    nonb_densities = []
    
    for tid in transcript_ids:
        clean_tid = tid.split('|')[0]
        
        if clean_tid in feature_meta:
            te_densities.append(feature_meta[clean_tid]['te_density'])
            nonb_densities.append(feature_meta[clean_tid]['nonb_density'])
        else:
            te_densities.append(0.0)
            nonb_densities.append(0.0)
    
    te_densities = np.array(te_densities)
    nonb_densities = np.array(nonb_densities)
    
    # 1. TE density overlay
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1],
                        c=te_densities, s=3, alpha=0.6, cmap='YlOrRd',
                        rasterized=True)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('TE Density', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
    ax.set_title(f'{fig_tag}Fold {fold_id}: TE Feature Density\n'
                f'(Mean: {te_densities.mean():.2f}, Max: {te_densities.max():.2f})',
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'umap_te_density.png', dpi=350, bbox_inches='tight')
    plt.close()
    
    # 2. Non-B DNA density overlay
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1],
                        c=nonb_densities, s=3, alpha=0.6, cmap='viridis',
                        rasterized=True)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Non-B DNA Density', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
    ax.set_title(f'{fig_tag}Fold {fold_id}: Non-B DNA Feature Density\n'
                f'(Mean: {nonb_densities.mean():.2f}, Max: {nonb_densities.max():.2f})',
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'umap_nonb_density.png', dpi=350, bbox_inches='tight')
    plt.close()
    
    # 3. Combined TE + Non-B
    fig, ax = plt.subplots(figsize=(12, 10))
    
    combined = te_densities + nonb_densities
    
    scatter = ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1],
                        c=combined, s=3, alpha=0.6, cmap='plasma',
                        rasterized=True)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Combined Feature Density (TE + Non-B)', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
    ax.set_title(f'{fig_tag}Fold {fold_id}: Combined Genomic Features\n'
                f'(Mean: {combined.mean():.2f}, Max: {combined.max():.2f})',
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'umap_combined_features.png', dpi=350, bbox_inches='tight')
    plt.close()
    
    print(f"      Saved feature density overlays")

def save_umap_csv(umap_embedding, labels, predictions, transcript_ids, biotypes, 
                  probs, is_hard, output_dir, fold_id):
    """Save UMAP coordinates with metadata to CSV"""
    output_dir = Path(output_dir)
    
    df = pd.DataFrame({
        'UMAP1': umap_embedding[:, 0],
        'UMAP2': umap_embedding[:, 1],
        'transcript_id': transcript_ids,
        'biotype': biotypes,
        'true_label': ['lnc' if l == 0 else 'pc' for l in labels],
        'predicted_label': ['lnc' if p == 0 else 'pc' for p in predictions],
        'is_hard_case': is_hard,
        'prob_lnc': probs[:, 0],
        'prob_pc': probs[:, 1],
        'fold_id': fold_id
    })
    
    output_file = output_dir / 'umap_embeddings.csv'
    df.to_csv(output_file, index=False)
    
    print(f"      Saved CSV: {output_file.name} ({len(df):,} rows)")


def process_all_folds(emb_data, args, hard_case_ids, idx_to_biotype, fig_tag=None, feature_meta=None):
    """Process each fold independently with its own UMAP"""
    
    print("\n" + "=" * 80)
    print("PROCESSING FOLDS INDEPENDENTLY")
    print("=" * 80)
    print("IMPORTANT: Computing UMAP separately for each fold")
    print("This avoids fold-separation artifacts!")
    print("=" * 80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for fold_id in sorted(emb_data['folds'].keys()):
        fold_data = emb_data['folds'][fold_id]
        
        print(f"\n{'='*80}")
        print(f"Fold {fold_id} ({len(fold_data['embeddings']):,} samples)")
        print(f"{'='*80}")
        
        # Create fold directory
        fold_dir = output_dir / f'fold_{fold_id}'
        fold_dir.mkdir(exist_ok=True)
        
        # Convert biotypes if possible
        biotypes = convert_biotype_indices(fold_data['biotypes'], idx_to_biotype)
        
        # Identify hard cases
        is_hard = identify_hard_cases(
            fold_data['labels'],
            fold_data['predictions'],
            fold_data['transcript_ids'],
            hard_case_ids
        )
        
        umap_embedding = compute_umap(
            fold_data['embeddings'],
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric
        )
        
        # Visualize
        visualize_umap(
            umap_embedding,
            fold_data['labels'],
            fold_data['predictions'],
            biotypes,
            is_hard,
            fold_dir,
            fold_id,
            fig_tag=fig_tag
        )

        # Features model: add TE/Non-B density overlays
        if feature_meta is not None:
            visualize_feature_overlays(
                umap_embedding,
                fold_data['transcript_ids'],
                feature_meta,
                fold_dir,
                fold_id,
                fig_tag=fig_tag
            )
        
        # Save CSV
        save_umap_csv(
            umap_embedding,
            fold_data['labels'],
            fold_data['predictions'],
            fold_data['transcript_ids'],
            biotypes,
            fold_data['probs'],
            is_hard,
            fold_dir,
            fold_id
        )
        
        print(f"    Fold {fold_id} complete: {fold_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize beta-VAE embeddings with UMAP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--embeddings', type=str, required=True,
                       help='Path to embeddings .npz file')
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='Path to experiment directory (for detecting model type)')
    parser.add_argument('--hard_cases', type=str, default=None,
                       help='Path to hard cases CSV (optional)')
    parser.add_argument('--lnc_fasta', type=str, required=True,
                       help='Path to lncRNA FASTA (for biotype mapping, optional)')
    parser.add_argument('--pc_fasta', type=str, required=True,
                          help='Path to protein-coding FASTA (for biotype mapping, optional)')
    parser.add_argument('--biotype_csv', type=str,
                       default=None,
                       help='Path to biotype CSV for converting indices to names')
    parser.add_argument('--output_dir', type=str, default='umap_visualizations',
                       help='Output directory')
    parser.add_argument('--n_neighbors', type=int, default=30,
                       help='UMAP n_neighbors parameter (default: 30)')
    parser.add_argument('--min_dist', type=float, default=0.1,
                       help='UMAP min_dist parameter (default: 0.1)')
    parser.add_argument('--metric', type=str, default='euclidean',
                       choices=['euclidean', 'manhattan', 'cosine', 'correlation'],
                       help='UMAP distance metric (default: euclidean)')
    parser.add_argument('--per_fold', action='store_true', default=True,
                       help='Process each fold independently (default: True, RECOMMENDED)')
    parser.add_argument('--model_label', type=str, default=None )
    parser.add_argument('--gencode_version', type=str, default=None )
    
    
    args = parser.parse_args()
    tag_parts = [p for p in [args.model_label,
             f'GENCODE {args.gencode_version}' if args.gencode_version else ''] if p]
    fig_tag = ' | '.join(tag_parts) + ' — ' if tag_parts else ''
    
    print("=" * 80)
    print("BETA-VAE EMBEDDING VISUALIZATION")
    print("=" * 80)

    model_type = detect_model_type(args.experiment_dir)
    print(f"\nDetected model type: {model_type}")
    
    # Load embeddings
    emb_data = load_embeddings(args.embeddings)
    
    if emb_data['format'] != 'multi_fold':
        print("ERROR: This script requires multi-fold embeddings format!")
        return
    
    print(f"\nTotal folds: {emb_data['n_folds']}")
    total_samples = sum(len(fold['embeddings']) for fold in emb_data['folds'].values())
    print(f"Total samples: {total_samples:,}")
    
    # Load biotype mapping
    idx_to_biotype = None
    if args.biotype_csv:
        idx_to_biotype = load_biotype_mapping_from_csv(
            args.biotype_csv, 
            args.lnc_fasta, 
            args.pc_fasta
        )
    
    # Load feature metadata (only for features models)
    feature_meta = None
    if model_type == 'features':
        feature_meta = load_feature_metadata(args.experiment_dir)
    
    # Load hard cases
    hard_case_ids = load_hard_cases(args.hard_cases)
    
    # Process all folds
    process_all_folds(emb_data, args, hard_case_ids, idx_to_biotype, fig_tag=fig_tag, feature_meta=feature_meta)
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"\nGenerated per fold:")
    print(f"  - umap_all_samples.png      (binary classification)")
    print(f"  - umap_hard_cases.png        (hard case identification)")
    if idx_to_biotype:
        print(f"  - umap_by_biotype.png        (biotype distribution)")
        print(f"  - umap_minor_biotypes_only.png")
    
    if model_type == 'features' and feature_meta:
        print(f"  - umap_te_density.png        (TE feature overlay)")
        print(f"  - umap_nonb_density.png      (Non-B DNA overlay)")
        print(f"  - umap_combined_features.png (TE + Non-B combined)")
    
    print(f"  - umap_embeddings.csv        (coordinates + metadata)")


if __name__ == '__main__':
    main()