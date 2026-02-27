#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loading utilities
"""
import pandas as pd
from Bio import SeqIO


def load_metadata(tsv_path):
    """Load TSV metadata file"""
    return pd.read_csv(tsv_path, sep='\t')


def load_fasta_sequences(lnc_path, pc_path):
    """Load FASTA files for lnc and pc sequences"""
    lnc_seqs = list(SeqIO.parse(lnc_path, "fasta"))
    pc_seqs = list(SeqIO.parse(pc_path, "fasta"))
    
    print(f"Loaded {len(lnc_seqs)} lncRNA sequences and {len(pc_seqs)} protein-coding sequences.")
    return lnc_seqs, pc_seqs


def match_sequences_to_dataframe(df, lnc_seqs, pc_seqs):
    """Match sequences from FASTA files to dataframe entries"""
    # Create dictionaries for fast lookup
    lnc_dict = {seq.id.split('|')[0]: seq for seq in lnc_seqs}
    pc_dict = {seq.id.split('|')[0]: seq for seq in pc_seqs}
    
    # Match sequences to dataframe
    matched_sequences = []
    matched_labels = []
    unmatched = []
    
    for idx, row in df.iterrows():
        transcript_id = row['transcript_id']
        label = row['label']
        
        seq_dict = lnc_dict if label == 'lnc' else pc_dict
        if transcript_id in seq_dict:
            matched_sequences.append(seq_dict[transcript_id])
            matched_labels.append(label)
        else:
            unmatched.append((transcript_id, label))
    
    print(f"Matched {len(matched_sequences)} sequences out of {len(df)}")
    if unmatched:
        print(f"Unmatched: {len(unmatched)} sequences")
    
    return matched_sequences, matched_labels, unmatched

def extract_label_from_fasta_id(seq_id):
    """
    Extract label from FASTA sequence ID based on CDS presence.
    
    Args:
        seq_id: FASTA sequence ID (e.g., ENST...|...|CDS:...|)
    
    Returns:
        'pc' if has CDS annotations, 'lnc' otherwise
    """
    return 'pc' if 'CDS:' in seq_id else 'lnc'


def load_sequences_with_labels(lnc_fasta_path, pc_fasta_path):
    """
    Load sequences from FASTA files and infer labels from IDs.
    
    Returns:
        sequences: List of SeqRecord objects
        labels: List of labels ('lnc' or 'pc')
    """
    from Bio import SeqIO
    
    # Load sequences
    lnc_seqs = list(SeqIO.parse(lnc_fasta_path, 'fasta'))
    pc_seqs = list(SeqIO.parse(pc_fasta_path, 'fasta'))
    
    print(f"Loaded {len(lnc_seqs)} sequences from lncRNA FASTA")
    print(f"Loaded {len(pc_seqs)} sequences from protein-coding FASTA")
    
    # Combine all sequences
    all_sequences = lnc_seqs + pc_seqs
    
    # Extract labels from sequence IDs
    labels = [extract_label_from_fasta_id(seq.id) for seq in all_sequences]
    
    # Verify labels match expected files
    lnc_count = labels.count('lnc')
    pc_count = labels.count('pc')
    
    print(f"\nLabel distribution (inferred from IDs):")
    print(f"  lnc: {lnc_count}")
    print(f"  pc:  {pc_count}")
    
    # Sanity check
    lnc_from_lnc_file = sum(1 for i, seq in enumerate(lnc_seqs) if extract_label_from_fasta_id(seq.id) == 'lnc')
    pc_from_pc_file = sum(1 for i, seq in enumerate(pc_seqs) if extract_label_from_fasta_id(seq.id) == 'pc')
    
    print(f"\nSanity check:")
    print(f"  lnc sequences with lnc label: {lnc_from_lnc_file}/{len(lnc_seqs)} ({100*lnc_from_lnc_file/len(lnc_seqs):.1f}%)")
    print(f"  pc sequences with pc label: {pc_from_pc_file}/{len(pc_seqs)} ({100*pc_from_pc_file/len(pc_seqs):.1f}%)")
    
    return all_sequences, labels