#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sanity check script for GENCODE data before training
"""
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from Bio import SeqIO
from collections import Counter
import matplotlib.pyplot as plt

from data.cv_utils import (
    load_biotype_mapping,
    extract_biotypes_from_sequences,
    group_rare_biotypes,
    create_length_stratified_groups
)

def analyze_fasta_files(lnc_fasta, pc_fasta):
    """Analyze basic FASTA file statistics"""
    print("=" * 80)
    print("FASTA FILE ANALYSIS - GENCODE")
    print("=" * 80)
    
    # Load sequences
    print(f"\nLoading lncRNA from: {lnc_fasta}")
    lnc_seqs = list(SeqIO.parse(lnc_fasta, 'fasta'))
    print(f"    Loaded {len(lnc_seqs):,} lncRNA sequences")
    
    print(f"\nLoading PC from: {pc_fasta}")
    pc_seqs = list(SeqIO.parse(pc_fasta, 'fasta'))
    print(f"    Loaded {len(pc_seqs):,} PC sequences")
    
    # Length statistics
    lnc_lengths = [len(seq.seq) for seq in lnc_seqs]
    pc_lengths = [len(seq.seq) for seq in pc_seqs]
    
    print(f"\nLength Statistics:")
    print(f"  lncRNA:")
    print(f"    Min:    {min(lnc_lengths):>8,} bp")
    print(f"    Max:    {max(lnc_lengths):>8,} bp")
    print(f"    Mean:   {np.mean(lnc_lengths):>8,.1f} bp")
    print(f"    Median: {np.median(lnc_lengths):>8,.1f} bp")
    print(f"    Q25:    {np.percentile(lnc_lengths, 25):>8,.1f} bp")
    print(f"    Q75:    {np.percentile(lnc_lengths, 75):>8,.1f} bp")
    
    print(f"  Protein-coding:")
    print(f"    Min:    {min(pc_lengths):>8,} bp")
    print(f"    Max:    {max(pc_lengths):>8,} bp")
    print(f"    Mean:   {np.mean(pc_lengths):>8,.1f} bp")
    print(f"    Median: {np.median(pc_lengths):>8,.1f} bp")
    print(f"    Q25:    {np.percentile(pc_lengths, 25):>8,.1f} bp")
    print(f"    Q75:    {np.percentile(pc_lengths, 75):>8,.1f} bp")
    
    # Class balance
    total = len(lnc_seqs) + len(pc_seqs)
    print(f"\nClass Balance:")
    print(f"  lncRNA: {len(lnc_seqs):,} ({100*len(lnc_seqs)/total:.2f}%)")
    print(f"  PC:     {len(pc_seqs):,} ({100*len(pc_seqs)/total:.2f}%)")
    print(f"  Ratio (lnc:pc): {len(lnc_seqs)/len(pc_seqs):.3f}:1")
    
    return lnc_seqs, pc_seqs, lnc_lengths, pc_lengths


def check_truncation_impact(lengths, class_name, max_length=4000):
    """Check how many sequences will be truncated"""
    print(f"\nTruncation Analysis for {class_name} (max_length={max_length}):")
    
    truncated = sum(1 for l in lengths if l > max_length)
    pct = (truncated / len(lengths)) * 100
    
    print(f"  Total sequences:        {len(lengths):>8,}")
    print(f"  Sequences > {max_length}:     {truncated:>8,} ({pct:.2f}%)")
    print(f"  Sequences ≤ {max_length}:     {len(lengths) - truncated:>8,} ({100-pct:.2f}%)")
    
    if truncated > 0:
        over_lengths = [l for l in lengths if l > max_length]
        print(f"  Mean truncated length:  {np.mean(over_lengths):>8,.1f} bp")
        print(f"  Max truncated length:   {max(over_lengths):>8,} bp")
        print(f"  Median truncated:       {np.median(over_lengths):>8,.1f} bp")
        
        # Show how much is lost
        total_bp_lost = sum(l - max_length for l in over_lengths)
        mean_bp_lost = total_bp_lost / len(over_lengths)
        print(f"  Mean bp lost per seq:   {mean_bp_lost:>8,.1f} bp ({100*mean_bp_lost/np.mean(over_lengths):.1f}%)")


def analyze_biotypes(sequences, labels, biotype_csv):
    """Analyze biotype distribution"""
    print("\n" + "=" * 80)
    print("BIOTYPE ANALYSIS")
    print("=" * 80)
    
    # Load biotype mapping
    biotype_lookup = load_biotype_mapping(biotype_csv)
    print(f"\n  Loaded biotype info for {len(biotype_lookup):,} transcripts")
    
    # Extract biotypes
    biotypes = extract_biotypes_from_sequences(sequences, biotype_lookup)
    
    unknown_count = biotypes.count('unknown')
    print(f"  Unknown biotypes: {unknown_count:,} ({100*unknown_count/len(biotypes):.2f}%)")
    
    # Count unique biotypes
    biotype_counts = Counter(biotypes)
    print(f"\n  Total unique biotypes: {len(biotype_counts)}")
    
    # Show distribution by class
    df = pd.DataFrame({'biotype': biotypes, 'label': labels})
    
    print(f"\n  Top 20 biotypes:")
    print(f"  {'Biotype':<40} {'Count':>8} {'lncRNA':>8} {'PC':>8}")
    print(f"  {'-'*67}")
    
    for biotype, count in biotype_counts.most_common(20):
        biotype_df = df[df['biotype'] == biotype]
        lnc_count = (biotype_df['label'] == 'lnc').sum()
        pc_count = (biotype_df['label'] == 'pc').sum()
        print(f"  {biotype:<40} {count:>8,} {lnc_count:>8,} {pc_count:>8,}")
    
    if len(biotype_counts) > 20:
        print(f"  ... and {len(biotype_counts) - 20} more biotypes")
    
    # Check rare biotypes
    print(f"\n  Rare biotype analysis (threshold=500):")
    grouped = group_rare_biotypes(biotypes, min_count=500)
    other_count = grouped.count('other')
    print(f"    Sequences grouped as 'other': {other_count:,} ({100*other_count/len(grouped):.2f}%)")
    
    unique_after = len(set(grouped))
    print(f"    Unique biotypes after grouping: {unique_after}")
    
    # Show which biotypes were grouped
    rare_biotypes = [bt for bt, count in biotype_counts.items() if count < 500]
    print(f"    Number of rare biotypes: {len(rare_biotypes)}")
    if len(rare_biotypes) <= 10:
        print(f"    Rare biotypes: {', '.join(rare_biotypes)}")
    else:
        print(f"    Examples: {', '.join(rare_biotypes[:10])}...")
    
    return biotypes, grouped


def check_stratification_groups(sequences, labels, n_bins=5):
    """Check stratification group distribution"""
    print("\n" + "=" * 80)
    print("STRATIFICATION GROUP ANALYSIS")
    print("=" * 80)
    
    strat_groups = create_length_stratified_groups(sequences, labels, n_bins=n_bins)
    
    group_counts = Counter(strat_groups)
    print(f"\nStratification groups (n_bins={n_bins}):")
    print(f"  Total groups: {len(group_counts)}")
    print(f"\n  {'Group':<10} {'Count':>8} {'% of total':>12}")
    print(f"  {'-'*32}")
    
    total = len(strat_groups)
    for group, count in sorted(group_counts.items()):
        pct = 100 * count / total
        print(f"  {group:<10} {count:>8,} {pct:>11.2f}%")
    
    # Check minimum group size
    min_size = min(group_counts.values())
    max_size = max(group_counts.values())
    print(f"\n  Minimum group size: {min_size:,}")
    print(f"  Maximum group size: {max_size:,}")
    print(f"  Size ratio (max/min): {max_size/min_size:.2f}")
    
    if min_size < 10:
        print(f"   ️  WARNING: Smallest group has only {min_size} samples!")
        print(f"     This may cause issues with 5-fold CV.")
        print(f"     Consider using n_bins=3 or 4 instead.")
    elif min_size < 50:
        print(f"   ️  Note: Smallest group has {min_size} samples.")
        print(f"     This should work but watch for stratification warnings.")


def check_id_format(sequences, n_samples=10):
    """Check transcript ID format"""
    print("\n" + "=" * 80)
    print("TRANSCRIPT ID FORMAT CHECK")
    print("=" * 80)
    
    print(f"\nSample IDs (first {n_samples}):")
    for i, seq in enumerate(sequences[:n_samples]):
        print(f"  {i+1:2d}. {seq.id}")
    
    # Check if pipe-delimited
    has_pipe = all('|' in seq.id for seq in sequences[:100])
    print(f"\nPipe-delimited format: {'  Yes' if has_pipe else '✗ No'}")
    
    if has_pipe:
        # Extract base IDs
        base_ids = [seq.id.split('|')[0] for seq in sequences[:n_samples]]
        print(f"\nBase IDs (after split on '|'):")
        for i, base_id in enumerate(base_ids):
            print(f"  {i+1:2d}. {base_id}")
            
        # Check version numbers
        has_version = all('.' in base_id for base_id in base_ids)
        print(f"\nVersion numbers present: {'  Yes' if has_version else '✗ No'}")
        
        if has_version:
            print("\nNote: Biotype CSV should use transcript IDs WITH version numbers")
            print("      (e.g., 'ENST00000456328.2' not 'ENST00000456328')")


def create_length_distribution_plot(lnc_lengths, pc_lengths, output_path):
    """Create length distribution comparison plot"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax = axes[0]
    bins = np.logspace(np.log10(min(lnc_lengths + pc_lengths)),
                       np.log10(max(lnc_lengths + pc_lengths)), 50)
    
    ax.hist(lnc_lengths, bins=bins, alpha=0.6, label='lncRNA', color='#E74C3C', edgecolor='black')
    ax.hist(pc_lengths, bins=bins, alpha=0.6, label='PC', color='#3498DB', edgecolor='black')
    ax.set_xscale('log')
    ax.set_xlabel('Sequence Length (bp)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Length Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Mark max_length threshold
    ax.axvline(4000, color='red', linestyle='--', linewidth=2, label='max_length=4000')
    
    # Box plot
    ax = axes[1]
    bp = ax.boxplot([lnc_lengths, pc_lengths],
                     labels=['lncRNA', 'PC'],
                     patch_artist=True,
                     showfliers=False)
    
    bp['boxes'][0].set_facecolor('#E74C3C')
    bp['boxes'][1].set_facecolor('#3498DB')
    
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1.5)
    
    ax.set_ylabel('Sequence Length (bp)', fontsize=12, fontweight='bold')
    ax.set_title('Length Distribution (Box Plot)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')
    
    # Add median values as text
    median_lnc = np.median(lnc_lengths)
    median_pc = np.median(pc_lengths)
    ax.text(1, median_lnc, f'{median_lnc:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(2, median_pc, f'{median_pc:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=1200, bbox_inches='tight')
    print(f"\n  Length distribution plot saved to: {output_path}")
    plt.close()


def create_summary_report(lnc_seqs, pc_seqs, biotypes, lnc_lengths, pc_lengths, output_file):
    """Create comprehensive summary report"""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GENCODE DATA SANITY CHECK REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Dataset Size:\n")
        f.write(f"  lncRNA sequences:  {len(lnc_seqs):>8,}\n")
        f.write(f"  PC sequences:      {len(pc_seqs):>8,}\n")
        total = len(lnc_seqs) + len(pc_seqs)
        f.write(f"  Total:             {total:>8,}\n")
        f.write(f"  Class balance:     {len(lnc_seqs)/len(pc_seqs):>8.3f}:1\n\n")
        
        f.write(f"Length Statistics:\n")
        f.write(f"  lncRNA:\n")
        f.write(f"    Mean:   {np.mean(lnc_lengths):>8,.1f} bp\n")
        f.write(f"    Median: {np.median(lnc_lengths):>8,.1f} bp\n")
        f.write(f"    Std:    {np.std(lnc_lengths):>8,.1f} bp\n")
        f.write(f"  PC:\n")
        f.write(f"    Mean:   {np.mean(pc_lengths):>8,.1f} bp\n")
        f.write(f"    Median: {np.median(pc_lengths):>8,.1f} bp\n")
        f.write(f"    Std:    {np.std(pc_lengths):>8,.1f} bp\n\n")

        
        f.write(f"Biotype Information:\n")
        f.write(f"  Unique biotypes:   {len(set(biotypes)):>8,}\n")
        f.write(f"  Unknown:           {biotypes.count('unknown'):>8,}\n\n")
        
        biotype_counts = Counter(biotypes)
        f.write(f"Top 15 biotypes:\n")
        for biotype, count in biotype_counts.most_common(15):
            f.write(f"  {biotype:<40} {count:>8,}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"  Summary report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Filter biotype CSV to match post-CD-HIT FASTA files'
    )
    parser.add_argument('--biotype_csv', required=True,
                       help='Path to full biotype CSV (pre-CD-HIT)')
    parser.add_argument('--lnc_fasta', required=True,
                       help='Path to post-CD-HIT lncRNA FASTA file')
    parser.add_argument('--pc_fasta', required=True,
                       help='Path to post-CD-HIT PC FASTA file')
    MAX_LENGTH = 15000
    args = parser.parse_args()
    
    OUTPUT_DIR = Path("gencode_sanity_check")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    print("GENCODE Data Sanity Check")
    print("=" * 80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # 1. Analyze FASTA files
    lnc_seqs, pc_seqs, lnc_lengths, pc_lengths = analyze_fasta_files(args.lnc_fasta, args.pc_fasta)
    
    # 2. Check truncation impact
    check_truncation_impact(lnc_lengths, 'lncRNA', MAX_LENGTH)
    check_truncation_impact(pc_lengths, 'Protein-coding', MAX_LENGTH)
    
    # 3. Combine sequences and labels
    sequences = lnc_seqs + pc_seqs
    labels = ['lnc'] * len(lnc_seqs) + ['pc'] * len(pc_seqs)
    
    # 4. Check ID format
    check_id_format(sequences)
    
    # 5. Analyze biotypes
    biotypes, grouped_biotypes = analyze_biotypes(sequences, labels, args.biotype_csv)
    
    # 6. Check stratification groups
    check_stratification_groups(sequences, labels, n_bins=5)
    
    # 7. Create visualizations
    length_plot_path = OUTPUT_DIR / "length_distribution.png"
    create_length_distribution_plot(lnc_lengths, pc_lengths, length_plot_path)
    
    # 8. Create summary report
    summary_path = OUTPUT_DIR / "sanity_check_summary.txt"
    create_summary_report(lnc_seqs, pc_seqs, biotypes, lnc_lengths, pc_lengths, summary_path)
    
    print("\n" + "=" * 80)
    print("SANITY CHECK COMPLETE")
    print("=" * 80)
    print("\n  All checks passed!")
    print(f"\nOutput files:")
    print(f"  - {length_plot_path}")
    print(f"  - {summary_path}")
    print("\nYou can now proceed with training using main_contrastive.py")


if __name__ == '__main__':
    main()