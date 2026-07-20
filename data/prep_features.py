"""
data/prepare_features.py

Prepare TE, NonB, and NonB2 (RNA secondary structure + rG4) features
for integration with β-VAE.

- Replaces single global StandardScaler with FeatureScalerBank
  (per-category scaling: RobustScaler / MinMaxScaler / pass-through)
- Scalers are fit on trainval transcripts only
- Chunked fitting preserved for memory efficiency

NonB2-specific preprocessing
-----------------------------
  Excluded columns (length-dependent or metadata):
    length            — transcript length, excluded to avoid length confound
    n_paired          — absolute count, use frac_paired instead
    n_competition     — absolute count, use frac_competition instead
    n_z_lt_minus1     — absolute count, use frac_z_lt_minus1 instead
    n_z_lt_minus2     — absolute count, use frac_z_lt_minus2 instead
    n_*_bin{1..N}     — absolute bin counts, use prop_*_bin instead
    simple_ID         — metadata column from ScanFold2 output

  Sentinel imputation:
    rg4_peak_rel_position = -1.0 when has_peak = False (no peaks detected)
    Imputed to 0.5 (neutral midpoint) before scaling.
    has_peak separately encodes peak absence as a binary feature.

  Boolean cast:
    has_peak (bool string or bool) → int8 (0/1) before scaling

Output files (per release, written to output_dir)
--------------------------------------------------
  {release}_te_features_clean.csv
  {release}_nonb_features_clean.csv
  {release}_nonb2_features_clean.csv
  {release}_te_scaler_bank.pkl
  {release}_nonb_scaler_bank.pkl
  {release}_nonb2_scaler_bank.pkl
  {release}_{block}_feature_names.txt  (for each block)
"""

import argparse
import gc
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from Bio import SeqIO

from data.feature_registry import REGISTRY, FeatureRegistry
from data.feature_scaler import FeatureScalerBank


# ---------------------------------------------------------------------------
# Columns to exclude from NonB2 (length-dependent or metadata)
# ---------------------------------------------------------------------------

_NONB2_EXCLUDE_PREFIXES = ("n_paired", "n_competition", "n_z_lt_minus1",
                            "n_z_lt_minus2")
_NONB2_EXCLUDE_EXACT    = {"length", "simple_ID", "transcript_id"}

# Absolute bin count columns: n_{feat}_bin{N}
_NONB2_BIN_PATTERN = re.compile(
    r"^n_(paired|competition|z_lt_minus1|z_lt_minus2)_bin\d+$"
)


def _nonb2_exclude_col(col: str) -> bool:
    """Return True if a column should be excluded from NonB2 features."""
    if col in _NONB2_EXCLUDE_EXACT:
        return True
    if col.startswith(_NONB2_EXCLUDE_PREFIXES) and not col.startswith("n_paired_bin"):
        # Exclude n_paired, n_competition etc. but not via the bin pattern
        # (bin pattern handled below)
        if col in ("n_paired", "n_competition", "n_z_lt_minus1", "n_z_lt_minus2"):
            return True
    if _NONB2_BIN_PATTERN.match(col):
        return True
    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_trainval_ids(split_dir: Path) -> set:
    ids = set()
    for fa_name in ["lnc_trainval.fa", "pc_trainval.fa"]:
        fa_path = split_dir / fa_name
        if not fa_path.exists():
            raise FileNotFoundError(
                f"Expected trainval FASTA not found: {fa_path}"
            )
        for record in SeqIO.parse(fa_path, "fasta"):
            ids.add(record.id.split("|")[0].split(".")[0])
    print(f"  Loaded {len(ids):,} trainval transcript IDs from {split_dir}")
    return ids


def _clean_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Convert dtypes, handle booleans, fill NaN/Inf."""
    bool_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()
    if bool_cols:
        print(f"  Converting {len(bool_cols)} {name} boolean/object columns...")
        for i, col in enumerate(bool_cols):
            if df[col].dtype == "object":
                df[col] = (df[col].astype(str).str.strip().str.lower()
                           .isin(["true", "1", "yes"])).astype(np.int8)
            else:
                df[col] = df[col].astype(np.int8)
            if i % 10 == 0:
                gc.collect()

    df = df.apply(pd.to_numeric, errors="coerce")
    gc.collect()

    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"  {name} missing values: {missing:,} — filling with 0")
        df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    return df


def _preprocess_nonb2(df: pd.DataFrame) -> pd.DataFrame:
    """
    NonB2-specific preprocessing before generic cleaning:

    1. Strip transcript_id from index (pipe-separated + version suffix)
    2. Drop excluded columns (metadata, length-dependent absolute counts)
    3. Impute rg4_peak_rel_position sentinel (-1.0) → 0.5
    4. Cast has_peak to int8

    Called BEFORE _clean_df so sentinel imputation precedes numeric coercion.
    """
    # ── Normalise transcript_id index ────────────────────────────────────────
    # Handle both pipe-separated FASTA IDs and plain versioned ENST IDs
    df.index = (df.index.astype(str)
                .str.split("|").str[0]
                .str.split(".").str[0])

    # ── Drop excluded columns ─────────────────────────────────────────────────
    cols_to_drop = [c for c in df.columns if _nonb2_exclude_col(c)]
    if cols_to_drop:
        print(f"  Dropping {len(cols_to_drop)} excluded NonB2 columns "
              f"(length-dependent / metadata)")
        df = df.drop(columns=cols_to_drop)

    # ── Sentinel imputation: rg4_peak_rel_position ────────────────────────────
    if "rg4_peak_rel_position" in df.columns:
        n_sentinel = (df["rg4_peak_rel_position"] == -1.0).sum()
        if n_sentinel > 0:
            print(f"  Imputing {n_sentinel:,} rg4_peak_rel_position sentinel "
                  f"values (-1.0 → 0.5)")
            df["rg4_peak_rel_position"] = df["rg4_peak_rel_position"].replace(
                -1.0, 0.5
            )

    # ── Cast has_peak to int8 before generic cleaning ─────────────────────────
    if "has_peak" in df.columns:
        df["has_peak"] = (df["has_peak"].astype(str).str.lower()
                          .isin(["true", "1", "yes"])).astype(np.int8)

    return df


def _normalise_index(df: pd.DataFrame) -> pd.DataFrame:
    """Strip pipe-suffix and version suffix from transcript_id index."""
    df.index = (df.index.astype(str)
                .str.split("|").str[0]
                .str.split(".").str[0])
    return df


def _fit_bank_chunked(
    bank:          FeatureScalerBank,
    df:            pd.DataFrame,
    trainval_mask: np.ndarray,
    chunk_size:    int = 50_000,
) -> None:
    """
    Fit a FeatureScalerBank on trainval rows using chunked loading.

    RobustScaler and MinMaxScaler do not support partial_fit, so we
    concatenate per-category column slices from all trainval chunks
    before calling fit().
    """
    trainval_df = df[trainval_mask]
    n_trainval  = len(trainval_df)
    print(f"  Fitting [{bank.block}] on {n_trainval:,} trainval rows...")

    cats_to_fit = {
        cat: (scaler, bank._cat_indices[cat])
        for cat, scaler in bank._scalers.items()
        if scaler is not None
    }

    if not cats_to_fit:
        print(f"  [{bank.block}] No categories require fitting (all pass-through).")
        bank._fitted = True
        return

    cat_chunks: dict = {cat: [] for cat in cats_to_fit}

    for start in range(0, n_trainval, chunk_size):
        chunk = trainval_df.iloc[start:start + chunk_size].values.astype(
            np.float64
        )
        for cat, (_, indices) in cats_to_fit.items():
            cat_chunks[cat].append(chunk[:, indices])
        del chunk
        gc.collect()

    for cat, (scaler, _) in cats_to_fit.items():
        X_cat = np.concatenate(cat_chunks[cat], axis=0)
        scaler.fit(X_cat)
        del X_cat
        cat_chunks[cat] = []
        gc.collect()

    bank._fitted = True
    print(f"  FeatureScalerBank [{bank.block}] fitted on {n_trainval:,} samples.")
    bank._print_summary()


def _warn_missing_trainval(df: pd.DataFrame, mask: np.ndarray,
                            label: str) -> None:
    n_missing = (~mask).sum()
    if n_missing > 0:
        pct = 100 * n_missing / len(df)
        warnings.warn(
            f"{n_missing:,} {label} transcripts ({pct:.1f}%) not in trainval "
            f"split — will be transformed but not used for scaler fitting.",
            UserWarning,
        )


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def prepare_features(
    te_csv_path:    str | Path,
    nonb_csv_path:  str | Path,
    split_dir:      str | Path,
    output_dir:     str | Path,
    release:        str,
    nonb2_csv_path: Optional[str | Path] = None,
    chunk_size:     int = 50_000,
    registry:       FeatureRegistry = REGISTRY,
) -> tuple:
    """
    Load, clean, and scale TE, NonB, and optionally NonB2 features.

    Parameters
    ----------
    te_csv_path    : path to TE features CSV (processed transcript-level)
    nonb_csv_path  : path to NonB features CSV (processed transcript-level)
    split_dir      : directory containing lnc_trainval.fa / pc_trainval.fa
    output_dir     : directory for output files
    release        : version string used in output filenames (e.g. "g47", "g49")
    nonb2_csv_path : path to NonB2 features CSV (ScanFold2 + rG4); optional
    chunk_size     : rows per chunk for scaler fitting
    registry       : FeatureRegistry instance

    Returns
    -------
    Tuple of (te_features, nonb_features, te_bank, nonb_bank)
    or (te_features, nonb_features, nonb2_features, te_bank, nonb_bank, nonb2_bank)
    if nonb2_csv_path is provided.
    """
    te_csv_path   = Path(te_csv_path)
    nonb_csv_path = Path(nonb_csv_path)
    split_dir     = Path(split_dir)
    output_dir    = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    has_nonb2 = nonb2_csv_path is not None
    if has_nonb2:
        nonb2_csv_path = Path(nonb2_csv_path)

    n_steps = 7 if has_nonb2 else 5
    step    = [0]

    def next_step(label: str) -> None:
        step[0] += 1
        print(f"\n[{step[0]}/{n_steps}] {label}")

    print("=" * 80)
    print(f"PREPARING FEATURES  [release={release}]")
    print("=" * 80)

    # ── Trainval IDs ──────────────────────────────────────────────────────────
    next_step("Loading trainval split IDs...")
    trainval_ids = load_trainval_ids(split_dir)

    # ── Load TE ───────────────────────────────────────────────────────────────
    next_step("Loading TE features...")
    te_df = pd.read_csv(te_csv_path, index_col="transcript_id")
    te_df = _normalise_index(te_df)
    print(f"  Shape: {te_df.shape}")
    meta_cols    = {"transcript_type", "coding_class", "transcript_length"}
    te_features  = te_df[[c for c in te_df.columns
                           if c not in meta_cols]].copy()
    del te_df; gc.collect()
    print(f"  Feature columns: {te_features.shape[1]}")

    # ── Load NonB ─────────────────────────────────────────────────────────────
    next_step("Loading NonB features...")
    nonb_df = pd.read_csv(nonb_csv_path, index_col="transcript_id")
    nonb_df = _normalise_index(nonb_df)
    print(f"  Shape: {nonb_df.shape}")
    redundant     = [c for c in nonb_df.columns
                     if c.endswith("_transcript_id")]
    nonb_features = nonb_df[[c for c in nonb_df.columns
                              if c not in meta_cols
                              and c not in redundant]].copy()
    del nonb_df; gc.collect()
    print(f"  Feature columns: {nonb_features.shape[1]}")

    # ── Load NonB2 (optional) ─────────────────────────────────────────────────
    nonb2_features = None
    if has_nonb2:
        next_step("Loading NonB2 features (ScanFold2 + rG4)...")
        nonb2_df = pd.read_csv(nonb2_csv_path, index_col="transcript_id")
        nonb2_df = _normalise_index(nonb2_df)
        print(f"  Shape (raw): {nonb2_df.shape}")
        nonb2_df = _preprocess_nonb2(nonb2_df)
        nonb2_features = nonb2_df.copy()
        del nonb2_df; gc.collect()
        print(f"  Feature columns after preprocessing: {nonb2_features.shape[1]}")

    # ── Validate registry ─────────────────────────────────────────────────────
    next_step("Validating feature registry...")
    registry.validate(
        nonb_columns  = list(nonb_features.columns),
        te_columns    = list(te_features.columns),
        nonb2_columns = (list(nonb2_features.columns)
                         if nonb2_features is not None else None),
        strict        = True,
    )

    # ── Clean ─────────────────────────────────────────────────────────────────
    next_step("Cleaning features...")
    te_features   = _clean_df(te_features,   "TE")
    nonb_features = _clean_df(nonb_features, "NonB")
    if nonb2_features is not None:
        nonb2_features = _clean_df(nonb2_features, "NonB2")
    gc.collect()

    # ── Fit scaler banks ──────────────────────────────────────────────────────
    next_step("Fitting scaler banks on trainval transcripts...")

    te_mask   = te_features.index.isin(trainval_ids)
    nonb_mask = nonb_features.index.isin(trainval_ids)
    print(f"  TE   trainval: {te_mask.sum():,} / {len(te_features):,}")
    print(f"  NonB trainval: {nonb_mask.sum():,} / {len(nonb_features):,}")
    _warn_missing_trainval(te_features,   te_mask,   "TE")
    _warn_missing_trainval(nonb_features, nonb_mask, "NonB")

    te_bank   = FeatureScalerBank("te",   registry)
    nonb_bank = FeatureScalerBank("nonb", registry)
    _fit_bank_chunked(te_bank,   te_features,   te_mask,   chunk_size)
    _fit_bank_chunked(nonb_bank, nonb_features, nonb_mask, chunk_size)

    nonb2_bank = None
    if nonb2_features is not None:
        nonb2_mask = nonb2_features.index.isin(trainval_ids)
        print(f"  NonB2 trainval: {nonb2_mask.sum():,} / {len(nonb2_features):,}")
        _warn_missing_trainval(nonb2_features, nonb2_mask, "NonB2")
        nonb2_bank = FeatureScalerBank("nonb2", registry)
        _fit_bank_chunked(nonb2_bank, nonb2_features, nonb2_mask, chunk_size)

    # ── Save ──────────────────────────────────────────────────────────────────
    next_step("Saving outputs...")

    def _save(df: pd.DataFrame, block: str, bank: FeatureScalerBank) -> None:
        csv_path = output_dir / f"{release}_{block}_features_clean.csv"
        df.to_csv(csv_path, chunksize=50_000)
        print(f"  Wrote {csv_path.name}")

        bank_path = output_dir / f"{release}_{block}_scaler_bank.pkl"
        bank.save(bank_path)
        print(f"  Wrote {bank_path.name}")

        names_path = output_dir / f"{release}_{block}_feature_names.txt"
        names_path.write_text("\n".join(df.columns))
        print(f"  Wrote {names_path.name}")

    _save(te_features,   "te",   te_bank)
    _save(nonb_features, "nonb", nonb_bank)
    if nonb2_features is not None and nonb2_bank is not None:
        _save(nonb2_features, "nonb2", nonb2_bank)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("FEATURE PREPARATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"  Release      : {release}")
    print(f"  TE features  : {te_features.shape[1]} dimensions")
    print(f"  NonB features: {nonb_features.shape[1]} dimensions")
    if nonb2_features is not None:
        print(f"  NonB2 features: {nonb2_features.shape[1]} dimensions")
    print(f"  Output dir   : {output_dir}")
    print(f"{'=' * 80}\n")

    if nonb2_features is not None:
        return te_features, nonb_features, nonb2_features, te_bank, nonb_bank, nonb2_bank
    return te_features, nonb_features, te_bank, nonb_bank


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare TE, NonB, and NonB2 features with per-category scaling."
    )
    parser.add_argument("--te_csv",     required=True,
                        help="TE features CSV (processed transcript-level)")
    parser.add_argument("--nonb_csv",   required=True,
                        help="NonB DNA features CSV")
    parser.add_argument("--split_dir",  required=True,
                        help="Directory with lnc_trainval.fa / pc_trainval.fa")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--release",    required=True,
                        help="Version string for output filenames, e.g. g47 or g49")
    parser.add_argument("--nonb2_csv",  default=None,
                        help="NonB2 features CSV (ScanFold2 + rG4); optional")
    parser.add_argument("--chunk_size", type=int, default=50_000)
    args = parser.parse_args()

    prepare_features(
        te_csv_path    = args.te_csv,
        nonb_csv_path  = args.nonb_csv,
        split_dir      = args.split_dir,
        output_dir     = args.output_dir,
        release        = args.release,
        nonb2_csv_path = args.nonb2_csv,
        chunk_size     = args.chunk_size,
    )