"""
Shared utilities for cross-validation splits.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


# This value must match across ALL scripts for reproducibility (training, evaluation, ablation)
DEFAULT_N_BINS = 5
DEFAULT_RANDOM_STATE = 42

def create_length_stratified_groups(sequences, labels, n_bins=DEFAULT_N_BINS):
    """
    Create stratification groups based on class and sequence length.
    All scripts (training, evaluation, ablation) must use this function.
    Args:
        sequences: List of SeqRecord objects
        labels: List of labels ('lnc' or 'pc')
        n_bins: Number of length bins (default: 5)
            WARNING: Changing this will create different validation splits!
    
    Returns:
        List of stratification group labels (e.g., ['0_2', '1_1', ...])
    """
    label_to_idx = {'lnc': 0, 'pc': 1}
    label_indices = [label_to_idx[l] for l in labels]
    
    # Get sequence lengths
    lengths = np.array([len(str(seq.seq)) for seq in sequences])
    
    # Create length bins
    length_bins = pd.qcut(lengths, q=n_bins, labels=False, duplicates='drop')
    
    # Combine label and length bin for stratification
    strat_groups = [f"{label}_{bin}" for label, bin in zip(label_indices, length_bins)]
    
    return strat_groups


def get_cv_splitter(n_folds=5, random_state=DEFAULT_RANDOM_STATE):
    """
    Get configured StratifiedKFold splitter.
    
    Args:
        n_folds: Number of CV folds
        random_state: Random seed for reproducibility
    
    Returns:
        StratifiedKFold object
    """
    return StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state
    )

def load_biotype_mapping(biotype_csv_path):
    """Load biotype information from CSV"""
    print(f"Loading biotype information from {biotype_csv_path}...")
    df = pd.read_csv(biotype_csv_path)
    
    biotype_lookup = {}
    for _, row in df.iterrows():
        transcript_id = row['transcript_id']
        biotype_lookup[transcript_id] = {
            'biotype': row['biotype'],
            'gene_id': row['gene_id'],
            'gene_name': row['gene_name'],
            'gene_biotype': row['gene_biotype']
        }
    
    return biotype_lookup


def extract_biotypes_from_sequences(sequences, biotype_lookup):
    """Extract biotype labels for sequences"""
    biotypes = []
    
    for seq in sequences:
        transcript_id = seq.id.split('|')[0]
        if transcript_id in biotype_lookup:
            biotypes.append(biotype_lookup[transcript_id]['biotype'])
        else:
            biotypes.append('unknown')
    
    return biotypes


def group_rare_biotypes(biotypes, min_count=500):
    """Group rare biotypes into 'other' category"""
    from collections import Counter
    counts = Counter(biotypes)
    
    grouped = []
    for bt in biotypes:
        if counts[bt] < min_count:
            grouped.append('other')
        else:
            grouped.append(bt)
    
    return grouped

from Bio import SeqIO
from typing import List, Tuple
from Bio.SeqRecord import SeqRecord

def load_sequences_in_order(lnc_fasta: str, pc_fasta: str) -> Tuple[List[SeqRecord], List[str]]:
    """
    Load sequences in canonical order (lnc + pc).
    
    This is the SINGLE function that defines sequence ordering.
    ALL scripts must use this to ensure consistency.
    
    Args:
        lnc_fasta: Path to lncRNA FASTA file
        pc_fasta: Path to protein-coding FASTA file
    
    Returns:
        all_sequences: List of SeqRecord objects (lnc first, then pc)
        labels: List of string labels ('lnc' or 'pc')
    
    Example:
        >>> sequences, labels = load_sequences_in_order(
        ...     'data/lnc.fa', 'data/pc.fa'
        ... )
        >>> print(f"First sequence is: {labels[0]}")  # Should be 'lnc'
    """
    print("Loading sequences in CANONICAL order (lnc + pc)...")
    
    # Load FASTA files
    lnc_seqs = list(SeqIO.parse(lnc_fasta, 'fasta'))
    pc_seqs = list(SeqIO.parse(pc_fasta, 'fasta'))
    
    print(f"  lncRNA sequences: {len(lnc_seqs):,}")
    print(f"  Protein-coding sequences: {len(pc_seqs):,}")
    
    # lnc first, then pc
    all_sequences = lnc_seqs + pc_seqs
    labels = ['lnc'] * len(lnc_seqs) + ['pc'] * len(pc_seqs)
    
    print(f"  Total sequences: {len(all_sequences):,}")
    print(f"  Order: lnc (indices 0-{len(lnc_seqs)-1}), pc (indices {len(lnc_seqs)}-{len(all_sequences)-1})")
    
    # Sanity check
    assert labels[0] == 'lnc', "First sequence should be lncRNA!"
    assert labels[-1] == 'pc', "Last sequence should be protein-coding!"
    
    return all_sequences, labels


def verify_sequence_order(sequences: List[SeqRecord], 
                         labels: List[str],
                         dataset_sequences: List[SeqRecord]) -> bool:
    """
    Verify that sequence order matches between CV splits and dataset.
    
    This should be called as a sanity check in all scripts.
    
    Args:
        sequences: Sequences used for CV splits
        labels: Labels for CV splits
        dataset_sequences: Sequences from SequenceFeatureDataset
    
    Returns:
        True if orders match, False otherwise
    
    Raises:
        ValueError: If mismatch detected
    """
    print("\n" + "="*80)
    print("VERIFYING SEQUENCE ORDER CONSISTENCY")
    print("="*80)
    
    # Check first 100 sequences
    mismatches = 0
    for i in range(min(100, len(sequences))):
        cv_tid = sequences[i].id.split('|')[0]
        dataset_tid = dataset_sequences[i].id.split('|')[0]
        
        if cv_tid != dataset_tid:
            mismatches += 1
            if mismatches <= 5:
                print(f"  ⚠️  MISMATCH at index {i}:")
                print(f"      CV splits: {cv_tid} ({labels[i]})")
                print(f"      Dataset:   {dataset_tid}")
    
    if mismatches == 0:
        print("  ✓ Sequence order verified (first 100 sequences match)")
        print("="*80 + "\n")
        return True
    else:
        print(f"\n  ✗ FATAL ERROR: Found {mismatches} mismatches in first 100 sequences!")
        print("  This means CV splits and dataset are using DIFFERENT orderings!")
        print("  Training/evaluation results will be INVALID!")
        print("="*80 + "\n")
        raise ValueError(
            f"Sequence order mismatch detected! "
            f"CV splits and dataset must use the same ordering. "
            f"All scripts should import load_sequences_canonical_order() from data.sequence_utils"
        )


def get_label_mapping() -> dict:
    """
    Get the canonical label-to-index mapping.
    
    Returns:
        Dictionary mapping string labels to numeric indices
    """
    return {'lnc': 0, 'pc': 1}


def get_index_to_label_mapping() -> dict:
    """
    Get the canonical index-to-label mapping.
    
    Returns:
        Dictionary mapping numeric indices to string labels
    """
    return {0: 'lnc', 1: 'pc'}