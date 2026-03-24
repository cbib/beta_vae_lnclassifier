# data/prepare_features.py
"""
Prepare TE and non-B DNA features for integration with β-VAE.
Scalers are fit on trainval transcripts only.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle
import gc
import re
from Bio import SeqIO


def load_trainval_ids(split_dir):
    split_dir = Path(split_dir)
    ids = set()
    for fa_name in ['lnc_trainval.fa', 'pc_trainval.fa']:
        fa_path = split_dir / fa_name
        if not fa_path.exists():
            raise FileNotFoundError(f"Expected trainval FASTA not found: {fa_path}")
        for record in SeqIO.parse(fa_path, 'fasta'):
            # GENCODE headers: ENST00000810660.1|ENSG...|...
            # take only the transcript ID before the first pipe
            transcript_id = record.id.split('|')[0]
            ids.add(transcript_id)
    print(f"  Loaded {len(ids):,} trainval transcript IDs from {split_dir}")
    return ids


def prepare_features(te_csv_path, nonb_csv_path, split_dir, output_dir):
    """
    Load, clean, and normalize TE and non-B DNA features.
    Scalers are fit on trainval split only, then applied to all transcripts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("PREPARING FEATURES (Memory-Efficient)")
    print("="*80)

    # =========================================================================
    # Load trainval IDs
    # =========================================================================
    print("\n[0/5] Loading trainval split IDs...")
    trainval_ids = load_trainval_ids(split_dir)

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
    te_features = te_df[te_feature_cols].copy()
    del te_df
    gc.collect()
    print(f"  Features: {te_features.shape}")

    # =========================================================================
    # Load Non-B features
    # =========================================================================
    print("\n[2/5] Loading Non-B DNA features...")
    nonb_df = pd.read_csv(nonb_csv_path, index_col='transcript_id')
    
    print(f"  Shape: {nonb_df.shape}")

    redundant_cols = [c for c in nonb_df.columns if c.endswith('_transcript_id')]
    nonb_feature_cols = [c for c in nonb_df.columns
                         if c not in metadata_cols and c not in redundant_cols]
    nonb_features = nonb_df[nonb_feature_cols].copy()
    del nonb_df
    gc.collect()
    print(f"  Features: {nonb_features.shape}")

    # =========================================================================
    # Convert boolean columns
    # =========================================================================
    print("\n[3/5] Converting data types...")

    for name, df in [('TE', te_features), ('Non-B', nonb_features)]:
        bool_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
        if bool_cols:
            print(f"  Converting {len(bool_cols)} {name} boolean columns...")
            for i, col in enumerate(bool_cols):
                if df[col].dtype == 'object':
                    df[col] = (df[col] == 'True').astype(np.int8)
                else:
                    df[col] = df[col].astype(np.int8)
                if i % 10 == 0:
                    gc.collect()

    te_features = te_features.apply(pd.to_numeric, errors='coerce')
    nonb_features = nonb_features.apply(pd.to_numeric, errors='coerce')
    gc.collect()

    # =========================================================================
    # Clean missing/invalid values
    # =========================================================================
    print("\n[4/5] Cleaning features...")

    for name, df in [('TE', te_features), ('Non-B', nonb_features)]:
        missing = df.isnull().sum().sum()
        print(f"  {name} missing values: {missing:,}")
        if missing > 0:
            df.fillna(0, inplace=True)
        df.replace([np.inf, -np.inf], 0, inplace=True)

    gc.collect()

    # =========================================================================
    # Fit scalers on trainval only, transform all
    # =========================================================================
    print("\n[5/5] Fitting scalers on trainval transcripts, transforming all...")

    te_trainval_mask   = te_features.index.isin(trainval_ids)
    nonb_trainval_mask = nonb_features.index.isin(trainval_ids)

    print(f"  TE   trainval: {te_trainval_mask.sum():,} / {len(te_features):,}")
    print(f"  NonB trainval: {nonb_trainval_mask.sum():,} / {len(nonb_features):,}")

    # Warn if coverage is unexpectedly low
    for label, mask, df in [('TE', te_trainval_mask, te_features),
                             ('NonB', nonb_trainval_mask, nonb_features)]:
        missing_from_split = (~mask).sum()
        if missing_from_split > 0:
            pct = 100 * missing_from_split / len(df)
            print(f"  WARNING: {missing_from_split:,} {label} transcripts "
                  f"({pct:.1f}%) not found in trainval split — "
                  f"will be transformed but not used for scaler fitting.")

    chunk_size = 50000
    te_scaler   = StandardScaler()
    nonb_scaler = StandardScaler()

    # Fit TE scaler on trainval rows only
    print("  Fitting TE scaler...")
    te_trainval = te_features[te_trainval_mask]
    for i in range(0, len(te_trainval), chunk_size):
        chunk = te_trainval.iloc[i:i+chunk_size].values.astype(np.float32)
        te_scaler.partial_fit(chunk)
        del chunk
        gc.collect()
    del te_trainval
    gc.collect()

    # Fit NonB scaler on trainval rows only
    print("  Fitting NonB scaler...")
    nonb_trainval = nonb_features[nonb_trainval_mask]
    for i in range(0, len(nonb_trainval), chunk_size):
        chunk = nonb_trainval.iloc[i:i+chunk_size].values.astype(np.float32)
        nonb_scaler.partial_fit(chunk)
        del chunk
        gc.collect()
    del nonb_trainval
    gc.collect()

    print("  ✓ Scalers fitted on trainval.")

    # =========================================================================
    # Save outputs
    # =========================================================================
    print("\nSaving outputs...")

    gencode_version = re.search(r'g\d+', str(te_csv_path)).group(0) \
        if re.search(r'g\d+', str(te_csv_path)) else 'unknown'

    print("  Writing te_features_clean.csv...")
    te_features.to_csv(output_dir / f'{gencode_version}_te_features_clean.csv',
                       chunksize=50000)

    print("  Writing nonb_features_clean.csv...")
    nonb_features.to_csv(output_dir / f'{gencode_version}_nonb_features_clean.csv',
                         chunksize=50000)

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
    print(f"  TE features  : {te_features.shape[1]} dimensions")
    print(f"  NonB features: {nonb_features.shape[1]} dimensions")
    print(f"  Output dir   : {output_dir}")
    print(f"{'='*80}\n")

    return te_features, nonb_features, te_scaler, nonb_scaler


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--te_csv',     required=True)
    parser.add_argument('--nonb_csv',   required=True)
    parser.add_argument('--split_dir',  required=True,
                        help='Directory containing lnc_trainval.fa and pc_trainval.fa')
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    prepare_features(args.te_csv, args.nonb_csv, args.split_dir, args.output_dir)