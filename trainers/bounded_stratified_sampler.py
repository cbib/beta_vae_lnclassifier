#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bounded Stratified Sampler for Contrastive Learning

Balances two competing goals:
1. Ensure rare biotypes have enough positive pairs for contrastive learning
2. Don't over-represent rare biotypes to the point of distorting embeddings

Strategy:
- Common biotypes: sample at natural frequency (no oversampling)
- Rare biotypes: oversample up to max_oversample factor
"""
import torch
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import numpy as np


def create_bounded_stratified_sampler(biotype_labels, max_oversample=10, verbose=True):
    """
    Create stratified sampler with bounded oversampling
    
    Args:
        biotype_labels: List of biotype strings for each sample
        max_oversample: Maximum oversampling factor for rare biotypes
        verbose: Print sampling statistics
    
    Returns:
        sampler: WeightedRandomSampler instance
    """
    # Count biotype frequencies
    biotype_counts = Counter(biotype_labels)
    
    # Find most common biotype count (upper bound)
    max_count = max(biotype_counts.values())
    
    # Calculate sampling weights with bounded oversampling
    weights = []
    for biotype in biotype_labels:
        count = biotype_counts[biotype]
        
        # Target frequency: min(max_count, count * max_oversample)
        # This means:
        # - Common biotypes (count close to max_count): no oversampling
        # - Rare biotypes: oversample up to max_oversample factor
        if max_oversample is not None:
            target_freq = min(max_count, count * max_oversample)
        
        # Weight = target_freq / actual_count
        # Higher weight = more likely to be sampled
        weight = target_freq / count
        weights.append(weight)
    
    weights = torch.DoubleTensor(weights)
    
    # Create sampler
    # replacement=True allows samples to appear multiple times per epoch
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(biotype_labels),
        replacement=True
    )
    
    # Print sampling statistics
    if verbose:
        print(f"\n{'='*80}")
        print(f"Bounded Stratified Sampling Configuration")
        print(f"{'='*80}")
        print(f"Max oversample factor: {max_oversample}x")
        print(f"Total samples: {len(biotype_labels)}")
        print(f"\nPer-biotype sampling:")
        print(f"{'Biotype':<30s} {'Count':>8s} {'Target':>10s} {'Factor':>8s} {'Per Batch':>10s}")
        print(f"{'-'*80}")
        
        # Calculate expected samples per batch (assuming batch_size=256)
        batch_size = 256
        num_batches = len(biotype_labels) // batch_size
        
        # Sort by original count (descending)
        sorted_biotypes = sorted(
            biotype_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for biotype, count in sorted_biotypes:
            # Calculate effective samples after weighting
            target_freq = min(max_count, count * max_oversample)
            oversample_factor = target_freq / count
            
            # Expected samples per batch
            samples_per_batch = target_freq / num_batches
            
            print(f"{biotype:<30s} {count:8d} {int(target_freq):10d} "
                  f"{oversample_factor:8.1f}x {samples_per_batch:9.1f}")
        
        print(f"{'='*80}\n")
    
    return sampler


def analyze_batch_composition(loader, biotype_labels, num_batches=10):
    """
    Analyze actual batch composition to verify stratified sampling works
    
    Args:
        loader: DataLoader with stratified sampler
        biotype_labels: List of all biotype labels
        num_batches: Number of batches to analyze
    """
    print(f"\n{'='*80}")
    print(f"Analyzing Batch Composition (first {num_batches} batches)")
    print(f"{'='*80}\n")
    
    batch_stats = []
    
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break
        
        # Get biotypes in this batch
        batch_indices = batch.get('index', None)
        if batch_indices is None:
            print("Warning: batch doesn't contain 'index' field, cannot analyze composition")
            break
        
        batch_biotypes = [biotype_labels[idx] for idx in batch_indices]
        batch_counts = Counter(batch_biotypes)
        
        batch_stats.append(batch_counts)
        
        print(f"Batch {batch_idx + 1}:")
        for biotype, count in sorted(batch_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {biotype:<30s}: {count:3d} samples")
        print()
    
    # Aggregate statistics
    print(f"\n{'='*80}")
    print(f"Aggregate Statistics (avg per batch)")
    print(f"{'='*80}")
    
    all_biotypes = set()
    for batch_count in batch_stats:
        all_biotypes.update(batch_count.keys())
    
    avg_counts = {}
    for biotype in all_biotypes:
        counts = [batch_count.get(biotype, 0) for batch_count in batch_stats]
        avg_counts[biotype] = np.mean(counts)
    
    print(f"{'Biotype':<30s} {'Avg per batch':>15s} {'Std':>10s}")
    print(f"{'-'*80}")
    for biotype in sorted(avg_counts.keys(), key=lambda x: avg_counts[x], reverse=True):
        counts = [batch_count.get(biotype, 0) for batch_count in batch_stats]
        avg = np.mean(counts)
        std = np.std(counts)
        print(f"{biotype:<30s} {avg:15.1f} {std:10.1f}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    # Test the sampler
    print("Testing Bounded Stratified Sampler\n")
    
    # Create mock biotype data matching distribution
    biotype_labels = (
        ['lncRNA'] * 159825 +
        ['protein_coding'] * 72757 +
        ['nonsense_mediated_decay'] * 18746 +
        ['TEC'] * 1026 +
        ['retained_intron'] * 798 +
        ['other'] * 421  # Sum of all rare biotypes
    )
    
    print(f"Total samples: {len(biotype_labels)}")
    print(f"Unique biotypes: {len(set(biotype_labels))}")
    
    # Test different max_oversample values
    for max_oversample in [5, 10, 20]:
        sampler = create_bounded_stratified_sampler(
            biotype_labels,
            max_oversample=max_oversample,
            verbose=True
        )
        print("\n" + "="*80 + "\n")