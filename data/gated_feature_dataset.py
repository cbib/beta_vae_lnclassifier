"""
data/gated_feature_dataset.py

Dataset combining RNA sequences with TE, NonB, and NonB2 features.

------------------------------
* Accepts nonb2_csv + nonb2_scaler_bank_path for RNA secondary structure
  and rG4 features (ScanFold2 + rg4detector).
* NonB2 has a single source ("processed" — RNA/transcript-level only). 
  Returns as "nonb2" key in __getitem__.
* NonB2 ID normalisation: strips pipe-suffix then version suffix, matching
  prepare_features.py convention.

__getitem__ keys
----------------
    sequence          (5, max_length)    float32  one-hot encoded
    te_genomic        (te_dim,)          float32  scaled
    te_processed      (te_dim,)          float32  scaled
    nonb_genomic      (nonb_dim,)        float32  scaled
    nonb_processed    (nonb_dim,)        float32  scaled
    nonb2             (nonb2_dim,)       float32  scaled  [new — if loaded]
    te_raw_diff       (te_dim,)          float32  |genomic - processed| unscaled
    nonb_raw_diff     (nonb_dim,)        float32  |genomic - processed| unscaled
    label             ()                 long
    transcript_id     str
    length            int

Backward compatibility
----------------------
* If nonb2_csv is not supplied, "nonb2" key is absent from __getitem__
  and the model must not request it.
* Legacy single-CSV / scaler args (te_features_csv, nonb_features_csv,
  te_scaler_path, nonb_scaler_path) still accepted.
* If only a single TE/NonB CSV is supplied, both *_genomic and *_processed
  return the same scaled vector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from torch.utils.data import Dataset

from data.cv_utils import load_sequences_in_order
from data.feature_registry import REGISTRY, FeatureRegistry
from data.feature_scaler import FeatureScalerBank


# ---------------------------------------------------------------------------
# ID normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_id(raw_id: str) -> str:
    """Strip pipe-suffix then version suffix from a transcript ID."""
    return raw_id.split("|")[0].split(".")[0]


def _normalise_index(df: pd.DataFrame) -> pd.DataFrame:
    """Apply _normalise_id to a DataFrame's index in-place."""
    df.index = df.index.map(_normalise_id)
    return df


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SequenceFeatureDataset(Dataset):
    """
    Dataset combining RNA sequences with TE, NonB, and NonB2 features.

    Parameters
    ----------
    lnc_fasta, pc_fasta     : paths to lncRNA / protein-coding FASTA files
    te_genomic_csv          : TE features — genomic (unspliced) version
    te_processed_csv        : TE features — processed (spliced) version
                              (if None, te_genomic_csv is used for both)
    nonb_genomic_csv        : NonB features — genomic version
    nonb_processed_csv      : NonB features — processed version
                              (if None, nonb_genomic_csv is used for both)
    nonb2_csv               : NonB2 features — RNA secondary structure + rG4
                              (single source, processed/transcript-level only)
    te_scaler_bank_path     : fitted FeatureScalerBank pickle for TE
    nonb_scaler_bank_path   : fitted FeatureScalerBank pickle for NonB
    nonb2_scaler_bank_path  : fitted FeatureScalerBank pickle for NonB2
    max_length              : sequence truncation length
    use_te_features         : if False, zero out all TE feature vectors
    use_nonb_features       : if False, zero out all NonB feature vectors
    use_nonb2_features      : if False, zero out NonB2 feature vector
    registry                : FeatureRegistry (defaults to module singleton)
    """

    def __init__(
        self,
        lnc_fasta:              Optional[str | Path] = None,
        pc_fasta:               Optional[str | Path] = None,
        # TE
        te_genomic_csv:         Optional[str | Path] = None,
        te_processed_csv:       Optional[str | Path] = None,
        # NonB
        nonb_genomic_csv:       Optional[str | Path] = None,
        nonb_processed_csv:     Optional[str | Path] = None,
        # NonB2 (new)
        nonb2_csv:              Optional[str | Path] = None,
        # Scaler banks
        te_scaler_bank_path:    Optional[str | Path] = None,
        nonb_scaler_bank_path:  Optional[str | Path] = None,
        nonb2_scaler_bank_path: Optional[str | Path] = None,
        # Options
        max_length:             int  = 6000,
        use_te_features:        bool = True,
        use_nonb_features:      bool = True,
        use_nonb2_features:     bool = True,
        registry:               FeatureRegistry = REGISTRY,
        # Legacy / compatibility
        fasta_file:             Optional[str | Path] = None,
        te_features_csv:        Optional[str | Path] = None,
        nonb_features_csv:      Optional[str | Path] = None,
        te_scaler_path:         Optional[str | Path] = None,
        nonb_scaler_path:       Optional[str | Path] = None,
    ) -> None:
        self.max_length         = max_length
        self.use_te_features    = use_te_features
        self.use_nonb_features  = use_nonb_features
        self.use_nonb2_features = use_nonb2_features
        self._reg               = registry
        self._has_nonb2         = nonb2_csv is not None

        # ── Legacy single-CSV fallback ────────────────────────────────────────
        if te_genomic_csv is None and te_features_csv is not None:
            te_genomic_csv = te_features_csv
        if nonb_genomic_csv is None and nonb_features_csv is not None:
            nonb_genomic_csv = nonb_features_csv

        # ── Load sequences ────────────────────────────────────────────────────
        print("Loading sequences...")
        if fasta_file is not None:
            sequences_raw  = list(SeqIO.parse(fasta_file, "fasta"))
            self.sequences = sequences_raw
            self.label_dict = {_normalise_id(s.id): 0 for s in sequences_raw}
        elif lnc_fasta is not None and pc_fasta is not None:
            self.sequences, sequence_labels = load_sequences_in_order(
                lnc_fasta, pc_fasta
            )
            self.label_dict = {
                _normalise_id(seq.id): (0 if lbl == "lnc" else 1)
                for seq, lbl in zip(self.sequences, sequence_labels)
            }
        else:
            raise ValueError(
                "Provide either fasta_file or both lnc_fasta and pc_fasta."
            )

        n_lnc = sum(v == 0 for v in self.label_dict.values())
        n_pc  = sum(v == 1 for v in self.label_dict.values())
        print(f"  Labels: {n_lnc:,} lncRNA, {n_pc:,} protein-coding")

        # ── Load feature DataFrames ───────────────────────────────────────────
        print("Loading features...")

        if te_genomic_csv is None or nonb_genomic_csv is None:
            raise ValueError(
                "At least te_genomic_csv and nonb_genomic_csv must be provided."
            )

        self._te_genomic   = self._load_csv(te_genomic_csv,   "TE genomic")
        self._nonb_genomic = self._load_csv(nonb_genomic_csv, "NonB genomic")

        self._te_processed = (
            self._load_csv(te_processed_csv, "TE processed")
            if te_processed_csv is not None
            else self._te_genomic
        )
        self._nonb_processed = (
            self._load_csv(nonb_processed_csv, "NonB processed")
            if nonb_processed_csv is not None
            else self._nonb_genomic
        )

        # NonB2 — single source
        self._nonb2: Optional[pd.DataFrame] = None
        if self._has_nonb2:
            self._nonb2 = self._load_csv(nonb2_csv, "NonB2")  # type: ignore[arg-type]

        # ── Validate registry ─────────────────────────────────────────────────
        print("Validating feature registry...")
        registry.validate(
            nonb_columns  = list(self._nonb_genomic.columns),
            te_columns    = list(self._te_genomic.columns),
            nonb2_columns = (list(self._nonb2.columns)
                             if self._nonb2 is not None else None),
            strict        = True,
        )

        # ── Load scaler banks ─────────────────────────────────────────────────
        print("Loading scaler banks...")

        # TE scaler
        if te_scaler_bank_path is not None:
            self._te_bank = FeatureScalerBank.load(te_scaler_bank_path)
        elif te_scaler_path is not None:
            import pickle
            with open(te_scaler_path, "rb") as f:
                self._te_legacy_scaler = pickle.load(f)
            self._te_bank = None
            print("  WARNING: using legacy TE scaler. "
                  "Re-run prepare_features.py to generate a FeatureScalerBank.")
        else:
            raise ValueError(
                "Provide te_scaler_bank_path (or legacy te_scaler_path)."
            )

        # NonB scaler
        if nonb_scaler_bank_path is not None:
            self._nonb_bank = FeatureScalerBank.load(nonb_scaler_bank_path)
        elif nonb_scaler_path is not None:
            import pickle
            with open(nonb_scaler_path, "rb") as f:
                self._nonb_legacy_scaler = pickle.load(f)
            self._nonb_bank = None
            print("  WARNING: using legacy NonB scaler. "
                  "Re-run prepare_features.py to generate a FeatureScalerBank.")
        else:
            raise ValueError(
                "Provide nonb_scaler_bank_path (or legacy nonb_scaler_path)."
            )

        # NonB2 scaler
        self._nonb2_bank: Optional[FeatureScalerBank] = None
        if self._has_nonb2:
            if nonb2_scaler_bank_path is None:
                raise ValueError(
                    "nonb2_csv provided but nonb2_scaler_bank_path is missing. "
                    "Run prepare_features.py with --nonb2_csv to generate it."
                )
            self._nonb2_bank = FeatureScalerBank.load(nonb2_scaler_bank_path)

        # ── Filter to valid IDs ───────────────────────────────────────────────
        valid_ids = (
            set(self.label_dict.keys())
            & set(self._te_genomic.index)
            & set(self._nonb_genomic.index)
        )
        if te_processed_csv is not None:
            valid_ids &= set(self._te_processed.index)
        if nonb_processed_csv is not None:
            valid_ids &= set(self._nonb_processed.index)
        if self._nonb2 is not None:
            valid_ids &= set(self._nonb2.index)

        n_before = len(self.sequences)
        self.sequences = [
            s for s in self.sequences
            if _normalise_id(s.id) in valid_ids
        ]
        n_after = len(self.sequences)
        if n_after < n_before:
            print(f"  WARNING: {n_before - n_after:,} sequences dropped "
                  f"(missing from one or more feature CSVs)")
        print(f"  Sequences after filtering: {n_after:,}")

    # -------------------------------------------------------------------------
    # Dataset protocol
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq_record    = self.sequences[idx]
        transcript_id = _normalise_id(seq_record.id)

        # Sequence
        seq_tensor = torch.from_numpy(
            self._encode_sequence(str(seq_record.seq))
        )

        # Raw (unscaled) TE / NonB for diff signal
        te_g_raw   = self._raw(self._te_genomic,    transcript_id)
        te_p_raw   = self._raw(self._te_processed,  transcript_id)
        nonb_g_raw = self._raw(self._nonb_genomic,  transcript_id)
        nonb_p_raw = self._raw(self._nonb_processed, transcript_id)

        # Scaled TE / NonB
        te_g   = self._scale_te(transcript_id, genomic=True)
        te_p   = self._scale_te(transcript_id, genomic=False)
        nonb_g = self._scale_nonb(transcript_id, genomic=True)
        nonb_p = self._scale_nonb(transcript_id, genomic=False)

        if not self.use_te_features:
            te_g = te_p = te_g_raw = te_p_raw = np.zeros_like(te_g)
        if not self.use_nonb_features:
            nonb_g = nonb_p = nonb_g_raw = nonb_p_raw = np.zeros_like(nonb_g)

        item = {
            "sequence":       seq_tensor,
            "te_genomic":     torch.from_numpy(te_g),
            "te_processed":   torch.from_numpy(te_p),
            "nonb_genomic":   torch.from_numpy(nonb_g),
            "nonb_processed": torch.from_numpy(nonb_p),
            "te_raw_diff":    torch.from_numpy(
                                  np.abs(te_g_raw - te_p_raw).astype(np.float32)),
            "nonb_raw_diff":  torch.from_numpy(
                                  np.abs(nonb_g_raw - nonb_p_raw).astype(np.float32)),
            "label":          torch.tensor(
                                  self.label_dict[transcript_id], dtype=torch.long),
            "transcript_id":  transcript_id,
            "length":         len(str(seq_record.seq)),
        }

        # NonB2 — only present if loaded
        if self._has_nonb2 and self._nonb2 is not None and self._nonb2_bank is not None:
            raw_nonb2 = self._raw(self._nonb2, transcript_id)
            if not self.use_nonb2_features:
                raw_nonb2 = np.zeros_like(raw_nonb2)
            item["nonb2"] = torch.from_numpy(
                self._nonb2_bank.transform(raw_nonb2)
            )

        return item

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _load_csv(path: str | Path, label: str) -> pd.DataFrame:
        path = Path(path)
        print(f"  Loading {label} from {path.name}...")
        df = pd.read_csv(path, index_col="transcript_id")
        # Normalise index
        df = _normalise_index(df)
        # Drop any residual metadata columns
        meta = {"transcript_type", "coding_class", "transcript_length",
                "simple_ID", "length"}
        df = df[[c for c in df.columns if c not in meta]]

        # Deduplicate — duplicate transcript_id rows break .loc[id] lookups
        # in __getitem__ (returns 2D instead of 1D, crashes default_collate)
        n_before = len(df)
        n_unique = df.index.nunique()
        if n_unique < n_before:
            n_dupes = n_before - n_unique
            print(f"    WARNING: {n_dupes:,} duplicate transcript_id rows "
                  f"({n_before:,} rows, {n_unique:,} unique) — "
                  f"keeping first occurrence")
            df = df[~df.index.duplicated(keep="first")]

        print(f"    Shape: {df.shape}")
        return df

    @staticmethod
    def _raw(df: pd.DataFrame, transcript_id: str) -> np.ndarray:
        return df.loc[transcript_id].values.astype(np.float32)

    def _scale_te(self, transcript_id: str, genomic: bool) -> np.ndarray:
        df  = self._te_genomic if genomic else self._te_processed
        raw = self._raw(df, transcript_id)
        if self._te_bank is not None:
            return self._te_bank.transform(raw)
        return self._te_legacy_scaler.transform(  # type: ignore[attr-defined]
            raw.reshape(1, -1)
        ).flatten().astype(np.float32)

    def _scale_nonb(self, transcript_id: str, genomic: bool) -> np.ndarray:
        df  = self._nonb_genomic if genomic else self._nonb_processed
        raw = self._raw(df, transcript_id)
        if self._nonb_bank is not None:
            return self._nonb_bank.transform(raw)
        return self._nonb_legacy_scaler.transform(  # type: ignore[attr-defined]
            raw.reshape(1, -1)
        ).flatten().astype(np.float32)

    @staticmethod
    def _encode_sequence(seq_str: str) -> np.ndarray:
        """One-hot encode RNA/DNA sequence → (5, max_length) float32."""
        encoding = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 4}
        seq_str  = seq_str.upper().replace("T", "U")
        one_hot  = np.zeros((5, 15000), dtype=np.float32)
        for i, nt in enumerate(seq_str[:15000]):
            one_hot[encoding.get(nt, 4), i] = 1.0
        return one_hot