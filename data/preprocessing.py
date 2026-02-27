#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sequence preprocessing and dataset classes
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from itertools import product

class SequencePreprocessor:
    """Preprocessor for RNA sequences"""
    
    def __init__(self, max_length=4000, encoding_type='one_hot', kmer_k=6, cse_d_model=128, cse_kernel_size=6):
        self.max_length = max_length
        self.encoding_type = encoding_type
        self.kmer_k = kmer_k
        self.cse_d_model = cse_d_model
        self.cse_kernel_size = cse_kernel_size

        self.nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    
        if self.encoding_type == 'kmer':
            self.kmer_to_idx = self._generate_kmer_dict(kmer_k)
            self.kmer_dim = len(self.kmer_to_idx)
            print(f"K-mer encoding with k={kmer_k}, vocab size={self.kmer_dim}")

        elif self.encoding_type == 'cse':
            print(f"CSE encoding with d_model={cse_d_model}, kernel_size={cse_kernel_size}")
            self.cse_output_length = self._calculate_cse_output_length(max_length, cse_kernel_size)
            print(f"  Input length: {max_length}")
            print(f"  Output length after CSE: {self.cse_output_length}")
            print(f"  Length reduction: {max_length / self.cse_output_length:.1f}x")

    def _calculate_cse_output_length(self, input_length, kernel_size, stride=None, padding=0):
        """Calculate output length after CSE convolution"""
        if stride is None:
            stride = kernel_size  # Non-overlapping by default
        return (input_length - kernel_size + 2 * padding) // stride + 1

    def _generate_kmer_dict(self, k):
        """Generate a dictionary mapping k-mers to indices"""
        nucleotides = ['A', 'C', 'G', 'T']
        kmers = [''.join(p) for p in product(nucleotides, repeat=k)]
        return {kmer: idx for idx, kmer in enumerate(kmers)}

    def one_hot_encode(self, sequence):
        """Convert sequence to one-hot encoding"""
        seq_str = str(sequence).upper()[:self.max_length]
        encoded = np.zeros((self.max_length, 5), dtype=np.float32)
        
        for i, nucleotide in enumerate(seq_str):
            if nucleotide in self.nucleotide_map:
                encoded[i, self.nucleotide_map[nucleotide]] = 1.0
        
        return encoded
    
    def kmer_encode(self, sequence):
        """
        Convert sequence to k-mer frequency vector
        
        Args:
            sequence: DNA/RNA sequence
        
        Returns:
            numpy array of shape (4^k,) with normalized k-mer frequencies
        """
        seq_str = str(sequence).upper().replace('U', 'T')
        freq = np.zeros(self.kmer_dim, dtype=np.float32)

        # Count k-mers
        valid_kmers = 0
        for i in range(len(seq_str) - self.kmer_k + 1):
            kmer = seq_str[i:i + self.kmer_k]
            if kmer in self.kmer_to_idx:
                idx = self.kmer_to_idx[kmer]
                freq[idx] += 1
                valid_kmers += 1

        # Normalize frequencies by total number of valid k-mers
        if valid_kmers > 0:
            freq /= valid_kmers

        return freq
    
    def pwm_encode(self, sequence):
        """
        Convert sequence to Position Weight Matrix (PWM) for CSE
        
        Args:
            sequence: DNA/RNA sequence
        
        Returns:
            PWM of shape (4, length) where:
            - Channel 0: A positions
            - Channel 1: C positions
            - Channel 2: G positions
            - Channel 3: T/U positions
        """
        seq_str = str(sequence).upper().replace('U', 'T')
        
        # Truncate if too long
        if len(seq_str) > self.max_length:
            seq_str = seq_str[:self.max_length]
        
        #length = len(seq_str)
        pwm = np.zeros((5, self.max_length), dtype=np.float32)
        
        for i, nucleotide in enumerate(seq_str):
            if nucleotide == 'N':
                # N is ambiguous: distribute equally across A, C, G, T (not the N channel) (cf. CSE method)
                pwm[0:4, i] = 0.25
            elif nucleotide in self.nucleotide_map:
                idx = self.nucleotide_map[nucleotide]
                pwm[idx, i] = 1.0
        
        return pwm
    
    def encode(self, sequence):
        if self.encoding_type == 'one_hot':
            return self.one_hot_encode(sequence)
        elif self.encoding_type == 'kmer':
            return self.kmer_encode(sequence)
        elif self.encoding_type == 'cse':
            return self.pwm_encode(sequence)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")

    def get_truncation_stats(self, sequences):
        """Analyze how many sequences will be truncated"""
        lengths = [len(str(seq.seq)) for seq in sequences]
        truncated = sum(1 for l in lengths if l > self.max_length)
        
        return {
            'total': len(sequences),
            'truncated': truncated,
            'truncated_pct': (truncated / len(sequences)) * 100,
            'mean_length': np.mean(lengths),
            'median_length': np.median(lengths)
        }


class RNASequenceDataset(Dataset):
    """PyTorch Dataset for RNA sequences"""
    
    def __init__(self, sequences, labels, preprocessor, augment=False):
        self.sequences = sequences
        self.labels = labels
        self.preprocessor = preprocessor
        self.label_map = {'lnc': 0, 'pc': 1}
        self.augment = augment
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_record = self.sequences[idx]
        seq = seq_record.seq
        
        # Simple augmentation: reverse complement with 50% probability
        if self.augment and np.random.random() > 0.5:
            seq = seq.reverse_complement()
        
        encoded_seq = self.preprocessor.encode(seq)
        
        label_str = self.labels[idx]
        label = self.label_map[label_str]
        
        seq_tensor = torch.FloatTensor(encoded_seq)
        label_tensor = torch.LongTensor([label])[0]
        
        return seq_tensor, label_tensor
    
class RNASequenceBiotypeDataset(Dataset):
    def __init__(self, sequences, labels, preprocessor, biotype_labels=None, biotype_to_idx=None):
        """
        Args:
            sequences: List of Bio.SeqRecord objects
            labels: List of 'lnc' or 'pc' strings
            preprocessor: SequencePreprocessor instance
            biotype_labels: Optional list of detailed biotype strings
            biotype_to_idx: Optional dict mapping biotype strings to indices
        """
        self.sequences = sequences
        self.labels = labels
        self.preprocessor = preprocessor
        self.biotype_labels = biotype_labels
        self.biotype_to_idx = biotype_to_idx
        
        # Encode labels
        self.label_to_idx = {'lnc': 0, 'pc': 1}
        self.encoded_labels = [self.label_to_idx[l] for l in labels]
        
        if biotype_labels is not None:
            if biotype_to_idx is None:
                raise ValueError("biotype_to_idx must be provided when biotype_labels is provided")
            
            self.biotype_indices = [self.biotype_to_idx[bt] for bt in biotype_labels]
            print(f"[RNASequenceBiotypeDataset] Encoded {len(set(biotype_labels))} unique biotypes")
            print(f"[RNASequenceBiotypeDataset] Biotype index range: {min(self.biotype_indices)} - {max(self.biotype_indices)}")
        else:
            self.biotype_indices = None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """Get one sample"""
        sequence = self.sequences[idx]
        label = self.encoded_labels[idx]
        
        # Preprocess
        processed = self.preprocessor.encode(str(sequence.seq))
        
        sample = {
            'sequence': processed,
            'label': label,
            'transcript_id': sequence.id
        }
        
        # Add biotype if available
        if self.biotype_indices is not None:
            sample['biotype_label'] = torch.LongTensor([self.biotype_indices[idx]])[0]
        
        return sample
    

# Utility functions for k-mer analysis
def analyze_kmer_distribution(sequences, labels, k=6):
    """
    Analyze k-mer distribution differences between classes
    
    Args:
        sequences: List of sequence records
        labels: List of labels ('lnc' or 'pc')
        k: k-mer size
    
    Returns:
        Dictionary with k-mer statistics
    """
    from collections import defaultdict
    
    preprocessor = SequencePreprocessor(encoding_type='kmer', kmer_k=k)
    
    lnc_kmers = []
    pc_kmers = []
    
    for seq, label in zip(sequences, labels):
        kmer_freq = preprocessor.kmer_encode(seq.seq)
        
        if label == 'lnc':
            lnc_kmers.append(kmer_freq)
        else:
            pc_kmers.append(kmer_freq)
    
    lnc_mean = np.mean(lnc_kmers, axis=0)
    pc_mean = np.mean(pc_kmers, axis=0)
    
    # Find most discriminative k-mers
    diff = np.abs(lnc_mean - pc_mean)
    top_indices = np.argsort(diff)[-20:][::-1]  # Top 20 discriminative k-mers
    
    # Reverse k-mer dictionary
    idx_to_kmer = {v: k for k, v in preprocessor.kmer_to_idx.items()}
    
    top_kmers = []
    for idx in top_indices:
        top_kmers.append({
            'kmer': idx_to_kmer[idx],
            'lnc_freq': lnc_mean[idx],
            'pc_freq': pc_mean[idx],
            'difference': diff[idx]
        })
    
    return {
        'lnc_mean': lnc_mean,
        'pc_mean': pc_mean,
        'top_discriminative': top_kmers
    }


def print_kmer_analysis(kmer_stats):
    """Print k-mer analysis results"""
    print("\n=== Top Discriminative K-mers ===")
    print(f"{'K-mer':<10} {'lncRNA freq':<15} {'PC freq':<15} {'Difference':<12}")
    print("-" * 55)
    
    for kmer_info in kmer_stats['top_discriminative']:
        print(f"{kmer_info['kmer']:<10} "
              f"{kmer_info['lnc_freq']:<15.6f} "
              f"{kmer_info['pc_freq']:<15.6f} "
              f"{kmer_info['difference']:<12.6f}")