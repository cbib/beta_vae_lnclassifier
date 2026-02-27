# experiments/data/feature_dataset.py
"""
Dataset class that combines sequence data with TE and non-B DNA features.
Binary classification only (lncRNA vs protein_coding).
No contrastive loss - features provide the biological priors instead.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from Bio import SeqIO
import pickle
from pathlib import Path

from data.cv_utils import load_sequences_in_order


class SequenceFeatureDataset(Dataset):
    """
    Dataset combining RNA sequences with TE and non-B DNA features.
    Extracts binary labels from feature CSVs (coding_class column).
    """
    
    def __init__(self, fasta_file=None, pc_fasta=None, lnc_fasta=None,
             te_features_csv=None, nonb_features_csv=None,
             te_scaler_path=None, nonb_scaler_path=None, max_length=6000,
             use_te_features=True, use_nonb_features=True):
        """
        Args:
            fasta_file: Path to combined FASTA (legacy support)
            pc_fasta: Path to protein-coding FASTA
            lnc_fasta: Path to lncRNA FASTA
            te_features_csv: Path to cleaned TE features
            nonb_features_csv: Path to cleaned non-B features
            te_scaler_path: Path to fitted TE scaler
            nonb_scaler_path: Path to fitted non-B scaler
            max_length: Maximum sequence length
            use_te_features: If False, return zeros for TE features (deactivation)
            use_nonb_features: If False, return zeros for Non-B features (deactivation)

        """
        self.max_length = max_length
        self.use_te_features = use_te_features
        self.use_nonb_features = use_nonb_features

        # Load sequences from either combined or separate files
        print("Loading sequences...")
        if fasta_file is not None:
            self.sequences = list(SeqIO.parse(fasta_file, 'fasta'))
        elif pc_fasta is not None and lnc_fasta is not None:
            self.sequences, sequence_labels = load_sequences_in_order(lnc_fasta, pc_fasta)
        else:
            raise ValueError("Must provide either fasta_file or both pc_fasta and lnc_fasta")
                
        # Create labels from FASTA files directly
        self.label_dict = {}
        for seq, label in zip(self.sequences, sequence_labels):
            transcript_id = seq.id.split('|')[0]
            self.label_dict[transcript_id] = 0 if label == 'lnc' else 1
        
        print(f"  Labels: {sum(v == 0 for v in self.label_dict.values())} lncRNA, "
              f"{sum(v == 1 for v in self.label_dict.values())} protein-coding")
        
        # Load cleaned features
        print("Loading features...")
        self.te_features = pd.read_csv(te_features_csv, index_col='transcript_id')
        self.nonb_features = pd.read_csv(nonb_features_csv, index_col='transcript_id')

        print("Index name:", self.te_features.index.name)
        print("Number of data columns (TE):", len(self.te_features.columns))
        print("First five columns (TE):", self.te_features.columns[:5])
        print("Sample row shape (TE):", self.te_features.iloc[0].shape)
        print("Sample row values (TE):", self.te_features.iloc[0].values[:5])
        
        print(f"  TE features: {self.te_features.shape}")
        print(f"  Non-B features: {self.nonb_features.shape}")
        
        print(f"  TE features: {self.te_features.shape[1]} dimensions")
        print(f"  Non-B features: {self.nonb_features.shape[1]} dimensions")
        
        # Load scalers
        print("Loading scalers...")
        with open(te_scaler_path, 'rb') as f:
            self.te_scaler = pickle.load(f)
        with open(nonb_scaler_path, 'rb') as f:
            self.nonb_scaler = pickle.load(f)
        
        valid_ids = set(self.label_dict.keys()) & \
            set(self.te_features.index) & \
            set(self.nonb_features.index)

        self.sequences = [s for s in self.sequences if s.id.split('|')[0] in valid_ids]
        print(f"  Sequences after filtering: {len(self.sequences)}")
        
    def __len__(self):
        return len(self.sequences)
    
    def encode_sequence(self, seq_str):
        """One-hot encode RNA sequence."""
        encoding = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3, 'N': 4}
        seq_str = seq_str.upper().replace('T', 'U')
        
        # Truncate or pad
        if len(seq_str) > self.max_length:
            seq_str = seq_str[:self.max_length]
        
        # One-hot encode
        one_hot = np.zeros((5, self.max_length), dtype=np.float32)
        for i, nucleotide in enumerate(seq_str):
            if nucleotide in encoding:
                one_hot[encoding[nucleotide], i] = 1.0
            else:
                one_hot[4, i] = 1.0  # 'N' or unknown
        
        return one_hot
    
    def __getitem__(self, idx):
        seq_record = self.sequences[idx]
        transcript_id = seq_record.id.split('|')[0]
        
        # Encode sequence
        seq_tensor = torch.from_numpy(self.encode_sequence(str(seq_record.seq)))
        
        # Get and normalize features
        te_feats = self.te_features.loc[transcript_id].values.astype(np.float32)
        nonb_feats = self.nonb_features.loc[transcript_id].values.astype(np.float32)
        
        te_feats = self.te_scaler.transform(te_feats.reshape(1, -1)).flatten()
        nonb_feats = self.nonb_scaler.transform(nonb_feats.reshape(1, -1)).flatten()
        
        te_tensor = torch.from_numpy(te_feats)
        nonb_tensor = torch.from_numpy(nonb_feats)
        
        # Get label (binary classification)
        label = self.label_dict[transcript_id]

        # Apply ablation if enabled
        if not self.use_te_features:
            te_feats = np.zeros_like(te_feats)
        if not self.use_nonb_features:
            nonb_feats = np.zeros_like(nonb_feats)
        
        return {
            'sequence': seq_tensor,
            'te_features': te_tensor,
            'nonb_features': nonb_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'transcript_id': transcript_id,
            'length': len(str(seq_record.seq))
        }