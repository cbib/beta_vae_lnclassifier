# experiments/data/prepare_features.py
"""
Prepare TE and non-B DNA features for integration with β-VAE.
Ultra memory-efficient version.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle
import gc
import re


def prepare_features(te_csv_path, nonb_csv_path, output_dir):
    """
    Load, clean, and normalize TE and non-B DNA features.
    Memory-efficient implementation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("PREPARING FEATURES (Memory-Efficient)")
    print("="*80)
    
    # =========================================================================
    # Load TE features
    # =========================================================================
    print("\n[1/5] Loading TE features...")
    te_df = pd.read_csv(te_csv_path, index_col='transcript_id')
    
    print(f"  Shape: {te_df.shape}")
    print(f"  Memory: {te_df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    
    # Remove metadata columns
    metadata_cols = ['transcript_type', 'coding_class', 'transcript_length']
    te_feature_cols = [c for c in te_df.columns if c not in metadata_cols]
    
    print(f"  Removing metadata: {metadata_cols}")
    te_features = te_df[te_feature_cols].copy()
    del te_df  # Free memory immediately
    gc.collect()
    
    print(f"  Features: {te_features.shape}")
    
    # =========================================================================
    # Load Non-B features
    # =========================================================================
    print("\n[2/5] Loading Non-B DNA features...")
    nonb_df = pd.read_csv(nonb_csv_path, index_col='transcript_id')
    
    print(f"  Shape: {nonb_df.shape}")
    print(f"  Memory: {nonb_df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    
    # Remove redundant columns
    redundant_cols = [c for c in nonb_df.columns if c.endswith('_transcript_id')]
    nonb_feature_cols = [c for c in nonb_df.columns 
                        if c not in metadata_cols and c not in redundant_cols]
    
    print(f"  Removing {len(redundant_cols)} redundant transcript_id columns")
    nonb_features = nonb_df[nonb_feature_cols].copy()
    del nonb_df
    gc.collect()
    
    print(f"  Features: {nonb_features.shape}")
    
    # =========================================================================
    # Convert boolean columns efficiently
    # =========================================================================
    print("\n[3/5] Converting data types...")
    
    # TE features - convert column by column to avoid memory spike
    te_bool_cols = te_features.select_dtypes(include=['object', 'bool']).columns.tolist()
    
    if te_bool_cols:
        print(f"  Converting {len(te_bool_cols)} TE boolean columns...")
        for i, col in enumerate(te_bool_cols):
            if (i + 1) % 5 == 0:
                print(f"    Progress: {i+1}/{len(te_bool_cols)}")
            
            # Convert directly without .map() to save memory
            if te_features[col].dtype == 'object':
                # String booleans
                te_features[col] = (te_features[col] == 'True').astype(np.int8)
            else:
                # Python booleans
                te_features[col] = te_features[col].astype(np.int8)
            
            if i % 10 == 0:
                gc.collect()  # Collect garbage periodically
    
    # Non-B features
    nonb_bool_cols = nonb_features.select_dtypes(include=['object', 'bool']).columns.tolist()
    
    if nonb_bool_cols:
        print(f"  Converting {len(nonb_bool_cols)} Non-B boolean columns...")
        for i, col in enumerate(nonb_bool_cols):
            if (i + 1) % 5 == 0:
                print(f"    Progress: {i+1}/{len(nonb_bool_cols)}")
            
            if nonb_features[col].dtype == 'object':
                nonb_features[col] = (nonb_features[col] == 'True').astype(np.int8)
            else:
                nonb_features[col] = nonb_features[col].astype(np.int8)
            
            if i % 10 == 0:
                gc.collect()
    
    print("  ✓ Boolean conversion complete")
    
    # Convert remaining columns to numeric (should already be numeric)
    print("  Converting remaining columns to numeric...")
    te_features = te_features.apply(pd.to_numeric, errors='coerce')
    nonb_features = nonb_features.apply(pd.to_numeric, errors='coerce')
    gc.collect()
    
    # =========================================================================
    # Clean missing/invalid values
    # =========================================================================
    print("\n[4/5] Cleaning features...")
    
    te_missing = te_features.isnull().sum().sum()
    nonb_missing = nonb_features.isnull().sum().sum()
    
    print(f"  Missing values: TE={te_missing:,}, Non-B={nonb_missing:,}")
    
    if te_missing > 0:
        te_features.fillna(0, inplace=True)
        gc.collect()
    
    if nonb_missing > 0:
        nonb_features.fillna(0, inplace=True)
        gc.collect()
    
    # Replace inf
    print("  Replacing inf values...")
    te_features.replace([np.inf, -np.inf], 0, inplace=True)
    nonb_features.replace([np.inf, -np.inf], 0, inplace=True)
    gc.collect()
    
    print(f"  ✓ Features cleaned")
    print(f"    TE: {te_features.shape}")
    print(f"    Non-B: {nonb_features.shape}")
    
    # =========================================================================
    # Fit scalers in chunks
    # =========================================================================
    print("\n[5/5] Fitting scalers...")
    
    te_scaler = StandardScaler()
    nonb_scaler = StandardScaler()
    
    chunk_size = 50000
    
    print(f"  Fitting TE scaler (chunk_size={chunk_size})...")
    for i in range(0, len(te_features), chunk_size):
        end_idx = min(i + chunk_size, len(te_features))
        chunk = te_features.iloc[i:end_idx].values.astype(np.float32)  # Use float32
        te_scaler.partial_fit(chunk)
        print(f"    {end_idx:,}/{len(te_features):,}")
        del chunk
        gc.collect()
    
    print(f"  Fitting Non-B scaler (chunk_size={chunk_size})...")
    for i in range(0, len(nonb_features), chunk_size):
        end_idx = min(i + chunk_size, len(nonb_features))
        chunk = nonb_features.iloc[i:end_idx].values.astype(np.float32)
        nonb_scaler.partial_fit(chunk)
        print(f"    {end_idx:,}/{len(nonb_features):,}")
        del chunk
        gc.collect()
    
    print(f"  ✓ Scalers fitted")
    
    # =========================================================================
    # Save outputs (in chunks for CSV to avoid memory spike)
    # =========================================================================
    print("\nSaving outputs...")

    gencode_version = re.search(r'g\d+', te_csv_path).group(0) if re.search(r'g\d+', te_csv_path) else 'unknown'
    
    print("  Writing te_features_clean.csv...")
    te_features.to_csv(output_dir / f'{gencode_version}_te_features_clean.csv', chunksize=50000)
    
    print("  Writing nonb_features_clean.csv...")
    nonb_features.to_csv(output_dir / f'{gencode_version}_nonb_features_clean.csv', chunksize=50000)
    
    print("  Writing scalers...")
    with open(output_dir / f'{gencode_version}_te_scaler.pkl', 'wb') as f:
        pickle.dump(te_scaler, f)
    with open(output_dir / f'{gencode_version}_nonb_scaler.pkl', 'wb') as f:
        pickle.dump(nonb_scaler, f)
    
    print("  Writing feature names...")
    with open(output_dir / f'{gencode_version}_te_feature_names.txt', 'w') as f:
        f.write('\n'.join(te_features.columns))
    with open(output_dir / f'{gencode_version}_nonb_feature_names.txt', 'w') as f:
        f.write('\n'.join(nonb_features.columns))
    
    print(f"\n{'='*80}")
    print("FEATURE PREPARATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutput directory: {output_dir}")
    print(f"  TE features: {te_features.shape[1]} dimensions")
    print(f"  Non-B features: {nonb_features.shape[1]} dimensions")
    print(f"\nFiles created:")
    for fname in ['te_features_clean.csv', 'nonb_features_clean.csv',
                  'te_scaler.pkl', 'nonb_scaler.pkl',
                  'te_feature_names.txt', 'nonb_feature_names.txt']:
        print(f"  - {fname}")
    print(f"{'='*80}\n")
    
    return te_features, nonb_features, te_scaler, nonb_scaler


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--te_csv', required=True)
    parser.add_argument('--nonb_csv', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    
    prepare_features(args.te_csv, args.nonb_csv, args.output_dir)