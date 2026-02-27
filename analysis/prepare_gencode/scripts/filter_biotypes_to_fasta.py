#!/usr/bin/env python3
"""
Analyze post-CD-HIT FASTA files and create biotype distributions
"""
import pandas as pd
import argparse
from pathlib import Path
from Bio import SeqIO
from collections import Counter


def extract_transcript_ids_from_fasta(fasta_path):
    """Extract transcript IDs from FASTA file"""
    transcript_ids = []
    for record in SeqIO.parse(fasta_path, 'fasta'):
        # Extract base transcript ID (before pipe or space)
        transcript_id = record.id.split('|')[0].split()[0]
        transcript_ids.append(transcript_id)
    return transcript_ids


def filter_biotypes_to_fasta(
    biotype_csv: str,
    lnc_fasta: str,
    pc_fasta: str,
    output_csv: str
):
    """
    Filter biotype CSV to only include transcripts present in post-CD-HIT FASTA files
    """
    print("=" * 80)
    print("FILTERING BIOTYPES TO POST-CD-HIT FASTA FILES")
    print("=" * 80)
    
    # Load biotype CSV
    print(f"\nLoading biotype CSV: {biotype_csv}")
    df_all = pd.read_csv(biotype_csv)
    print(f"  Total transcripts in CSV: {len(df_all):,}")
    
    # Extract transcript IDs from FASTA files
    print(f"\nExtracting transcript IDs from lncRNA FASTA: {lnc_fasta}")
    lnc_ids = extract_transcript_ids_from_fasta(lnc_fasta)
    print(f"  lncRNA transcripts: {len(lnc_ids):,}")
    
    print(f"\nExtracting transcript IDs from PC FASTA: {pc_fasta}")
    pc_ids = extract_transcript_ids_from_fasta(pc_fasta)
    print(f"  PC transcripts: {len(pc_ids):,}")
    
    # Combine all FASTA IDs
    fasta_ids = set(lnc_ids + pc_ids)
    print(f"\nTotal unique transcripts in FASTA files: {len(fasta_ids):,}")
    
    # Try exact match first
    print("\nAttempting exact match...")
    df_matched = df_all[df_all['transcript_id'].isin(fasta_ids)].copy()
    print(f"  Exact matches: {len(df_matched):,}")
    
    # If poor match rate, try without version numbers
    if len(df_matched) < len(fasta_ids) * 0.9:
        print("\nExact match rate low, trying without version numbers...")
        
        # Strip versions from FASTA IDs
        fasta_ids_no_version = {tid.split('.')[0] for tid in fasta_ids}
        
        # Strip versions from CSV
        df_all['transcript_id_no_version'] = df_all['transcript_id'].apply(lambda x: x.split('.')[0])
        
        # Match without version
        df_matched = df_all[df_all['transcript_id_no_version'].isin(fasta_ids_no_version)].copy()
        df_matched = df_matched.drop(columns=['transcript_id_no_version'])
        
        print(f"  Matches without version: {len(df_matched):,}")
    
    # Calculate statistics
    match_rate = len(df_matched) / len(fasta_ids) * 100
    reduction = len(df_all) - len(df_matched)
    
    print(f"\n{'='*80}")
    print("FILTERING RESULTS")
    print(f"{'='*80}")
    print(f"Original CSV transcripts:     {len(df_all):>10,}")
    print(f"Post-CD-HIT FASTA transcripts: {len(fasta_ids):>10,}")
    print(f"Matched transcripts:          {len(df_matched):>10,}")
    print(f"Match rate:                   {match_rate:>10.1f}%")
    print(f"Removed by CD-HIT:            {reduction:>10,} ({100*reduction/len(df_all):.1f}%)")
    
    if match_rate < 95:
        print(f"\n ️  WARNING: Low match rate ({match_rate:.1f}%)")
        print("   Sample FASTA IDs:")
        for i, tid in enumerate(list(fasta_ids)[:3]):
            print(f"     {tid}")
        print("   Sample CSV IDs:")
        for i, tid in enumerate(df_all['transcript_id'].head(3)):
            print(f"     {tid}")
    
    # Show biotype distribution
    print(f"\n{'='*80}")
    print("POST-CD-HIT BIOTYPE DISTRIBUTION")
    print(f"{'='*80}")
    
    if 'gene_biotype' in df_matched.columns:
        print("\nGene-level classes:")
        for gene_biotype in ['lncRNA', 'protein_coding']:
            count = (df_matched['gene_biotype'] == gene_biotype).sum()
            pct = 100 * count / len(df_matched)
            print(f"  {gene_biotype:<20s}: {count:>8,} ({pct:>5.2f}%)")
        
        lnc_count = (df_matched['gene_biotype'] == 'lncRNA').sum()
        pc_count = (df_matched['gene_biotype'] == 'protein_coding').sum()
        if lnc_count > 0 and pc_count > 0:
            print(f"  Balance (lnc:pc):      {lnc_count/pc_count:.3f}:1")
    
    print("\nTranscript biotypes (top 10):")
    biotype_counts = df_matched['biotype'].value_counts().head(10)
    for biotype, count in biotype_counts.items():
        pct = 100 * count / len(df_matched)
        print(f"  {biotype:<45s}: {count:>8,} ({pct:>5.2f}%)")
    
    # Save filtered CSV
    output_path = Path(output_csv)
    df_matched.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"Saved post-CD-HIT biotype CSV to: {output_path}")
    print(f"{'='*80}")
    
    return df_matched


def create_comparison_report(df_before, df_after, output_path):
    """Create report comparing pre and post CD-HIT"""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CD-HIT FILTERING IMPACT REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Before CD-HIT:  {len(df_before):>10,} transcripts\n")
        f.write(f"After CD-HIT:   {len(df_after):>10,} transcripts\n")
        f.write(f"Removed:        {len(df_before) - len(df_after):>10,} transcripts "
                f"({100*(len(df_before) - len(df_after))/len(df_before):.1f}%)\n\n")
        
        if 'gene_biotype' in df_after.columns:
            f.write("GENE-LEVEL CLASS DISTRIBUTION\n")
            f.write("-" * 80 + "\n\n")
            
            for gene_biotype in ['lncRNA', 'protein_coding']:
                before_count = (df_before['gene_biotype'] == gene_biotype).sum()
                after_count = (df_after['gene_biotype'] == gene_biotype).sum()
                removed = before_count - after_count
                
                before_pct = 100 * before_count / len(df_before)
                after_pct = 100 * after_count / len(df_after)
                
                f.write(f"{gene_biotype}:\n")
                f.write(f"  Before: {before_count:>10,} ({before_pct:>5.2f}%)\n")
                f.write(f"  After:  {after_count:>10,} ({after_pct:>5.2f}%)\n")
                f.write(f"  Removed: {removed:>9,} ({100*removed/before_count:>5.2f}% of class)\n\n")
            
            lnc_before = (df_before['gene_biotype'] == 'lncRNA').sum()
            pc_before = (df_before['gene_biotype'] == 'protein_coding').sum()
            lnc_after = (df_after['gene_biotype'] == 'lncRNA').sum()
            pc_after = (df_after['gene_biotype'] == 'protein_coding').sum()
            
            f.write(f"Class balance:\n")
            f.write(f"  Before: {lnc_before/pc_before:.3f}:1\n")
            f.write(f"  After:  {lnc_after/pc_after:.3f}:1\n\n")
        
        f.write("TRANSCRIPT BIOTYPE DISTRIBUTION (after CD-HIT)\n")
        f.write("-" * 80 + "\n\n")
        
        biotype_counts = df_after['biotype'].value_counts()
        for biotype, count in biotype_counts.items():
            pct = 100 * count / len(df_after)
            f.write(f"  {biotype:<45s}: {count:>8,} ({pct:>5.2f}%)\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"\nComparison report saved to: {output_path}")


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
    parser.add_argument('--output', required=True,
                       help='Output CSV for post-CD-HIT biotypes')
    parser.add_argument('--report', default=None,
                       help='Output comparison report (default: <output>.cdhit_report.txt)')
    
    args = parser.parse_args()
    
    # Load original biotype CSV
    df_before = pd.read_csv(args.biotype_csv)
    
    # Filter to FASTA files
    df_after = filter_biotypes_to_fasta(
        args.biotype_csv,
        args.lnc_fasta,
        args.pc_fasta,
        args.output
    )
    
    # Create comparison report
    report_path = args.report if args.report else Path(args.output).with_suffix('.cdhit_report.txt')
    create_comparison_report(df_before, df_after, report_path)
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"  1. Use {args.output} as biotype reference")
    print(f"  2. Run visualization script on this post-CD-HIT CSV")
    print(f"  3. Run sanity check with the post-CD-HIT FASTA files")


if __name__ == '__main__':
    main()