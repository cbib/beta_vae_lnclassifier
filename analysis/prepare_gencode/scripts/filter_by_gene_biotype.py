#!/usr/bin/env python3
"""
Filter GENCODE Biotypes by Gene Biotype
========================================
Filters GENCODE biotype CSV to keep only transcripts from lncRNA and protein_coding genes.

This ensures you're working with transcripts from genes that are definitively
lncRNA or protein-coding at the gene level, regardless of transcript-level biotype.

Usage:
    python filter_by_gene_biotype.py \
        --input gencode49_transcript_biotypes.csv \
        --output gencode49_dataset_biotypes.csv \
        --keep_gene_biotypes lncRNA protein_coding
"""

import pandas as pd
import argparse
from pathlib import Path


def filter_by_gene_biotype(
    input_csv: str,
    keep_gene_biotypes: list = None,
    output_csv: str = None
) -> pd.DataFrame:
    """
    Filter biotype CSV to keep only specific gene biotypes.
    
    Args:
        input_csv: Path to full biotype CSV
        keep_gene_biotypes: List of gene_biotype values to keep
        output_csv: Optional output path
    
    Returns:
        Filtered DataFrame
    """
    if keep_gene_biotypes is None:
        keep_gene_biotypes = ['lncRNA', 'protein_coding']
    
    print("=" * 80)
    print("FILTERING BY GENE BIOTYPE")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Total transcripts: {len(df):,}")
    
    # Show available columns
    print(f"\nColumns: {', '.join(df.columns)}")
    
    # Check if gene_biotype column exists
    if 'gene_biotype' not in df.columns:
        print(f"\n ️  ERROR: 'gene_biotype' column not found!")
        print(f"   Available columns: {', '.join(df.columns)}")
        return None
    
    # Show distribution before filtering
    print(f"\n{'='*80}")
    print("ORIGINAL GENE BIOTYPE DISTRIBUTION")
    print(f"{'='*80}")
    gene_biotype_counts = df['gene_biotype'].value_counts()
    print(f"\nTotal unique gene biotypes: {len(gene_biotype_counts)}")
    print(f"\nTop 15 gene biotypes:")
    for gene_biotype, count in gene_biotype_counts.head(15).items():
        pct = 100 * count / len(df)
        marker = " " if gene_biotype in keep_gene_biotypes else "✗"
        print(f"  {marker} {gene_biotype:45s}: {count:>8,} ({pct:>5.2f}%)")
    
    # Filter
    print(f"\n{'='*80}")
    print(f"FILTERING (keeping: {', '.join(keep_gene_biotypes)})")
    print(f"{'='*80}")
    
    df_filtered = df[df['gene_biotype'].isin(keep_gene_biotypes)].copy()
    
    print(f"\nFiltered transcripts: {len(df_filtered):,}")
    print(f"Removed transcripts: {len(df) - len(df_filtered):,}")
    print(f"Retention rate: {100 * len(df_filtered) / len(df):.2f}%")
    
    # Show gene biotype distribution after filtering
    print(f"\n{'='*80}")
    print("FILTERED GENE BIOTYPE DISTRIBUTION")
    print(f"{'='*80}")
    
    for gene_biotype in keep_gene_biotypes:
        count = (df_filtered['gene_biotype'] == gene_biotype).sum()
        pct = 100 * count / len(df_filtered)
        print(f"  {gene_biotype:45s}: {count:>8,} ({pct:>5.2f}%)")
    
    print(f"\nClass balance: {(df_filtered['gene_biotype'] == 'lncRNA').sum()}/{(df_filtered['gene_biotype'] == 'protein_coding').sum()}")
    print(f"               = {(df_filtered['gene_biotype'] == 'lncRNA').sum()/(df_filtered['gene_biotype'] == 'protein_coding').sum():.3f}:1 (lnc:pc)")
    
    # Show transcript biotype distribution within filtered dataset
    print(f"\n{'='*80}")
    print("TRANSCRIPT BIOTYPE DISTRIBUTION (within filtered dataset)")
    print(f"{'='*80}")
    
    transcript_biotype_counts = df_filtered['biotype'].value_counts()
    print(f"\nTotal unique transcript biotypes: {len(transcript_biotype_counts)}")
    print(f"\nTop 20 transcript biotypes:")
    for biotype, count in transcript_biotype_counts.head(20).items():
        pct = 100 * count / len(df_filtered)
        print(f"  {biotype:45s}: {count:>8,} ({pct:>5.2f}%)")
    
    # Save if output path provided
    if output_csv:
        output_path = Path(output_csv)
        df_filtered.to_csv(output_path, index=False)
        print(f"\n{'='*80}")
        print(f"Saved filtered data to: {output_path}")
        print(f"{'='*80}")
    
    return df_filtered


def create_comparison_report(df_original, df_filtered, output_path):
    """Create detailed comparison report"""
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GENE BIOTYPE FILTERING REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary statistics
        f.write("SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Original transcripts:  {len(df_original):>10,}\n")
        f.write(f"Filtered transcripts:  {len(df_filtered):>10,}\n")
        f.write(f"Removed transcripts:   {len(df_original) - len(df_filtered):>10,}\n")
        f.write(f"Retention rate:        {100 * len(df_filtered) / len(df_original):>10.2f}%\n\n")
        
        # Gene biotype comparison
        f.write("GENE BIOTYPE DISTRIBUTION\n")
        f.write("-" * 80 + "\n\n")
        
        orig_gene_biotypes = df_original['gene_biotype'].value_counts()
        filt_gene_biotypes = df_filtered['gene_biotype'].value_counts()
        
        f.write(f"{'Gene Biotype':45s} {'Original':>15s} {'Filtered':>15s} {'Status':>10s}\n")
        f.write("-" * 90 + "\n")
        
        all_gene_biotypes = set(orig_gene_biotypes.index)
        for gene_biotype in sorted(all_gene_biotypes, 
                                   key=lambda x: orig_gene_biotypes[x], 
                                   reverse=True):
            orig_count = orig_gene_biotypes[gene_biotype]
            filt_count = filt_gene_biotypes.get(gene_biotype, 0)
            
            orig_pct = 100 * orig_count / len(df_original)
            
            if filt_count > 0:
                filt_pct = 100 * filt_count / len(df_filtered)
                status = "KEPT"
                f.write(f"{gene_biotype:45s} {orig_count:>7,} ({orig_pct:>5.2f}%) "
                       f"{filt_count:>7,} ({filt_pct:>5.2f}%) {status:>10s}\n")
            else:
                status = "REMOVED"
                f.write(f"{gene_biotype:45s} {orig_count:>7,} ({orig_pct:>5.2f}%) "
                       f"{'0':>7s} {'(0.00%)':>9s} {status:>10s}\n")
        
        f.write("\n")
        
        # Transcript biotype breakdown for filtered dataset
        f.write("TRANSCRIPT BIOTYPE BREAKDOWN (filtered dataset only)\n")
        f.write("-" * 80 + "\n\n")
        
        transcript_biotypes = df_filtered['biotype'].value_counts()
        f.write(f"Total unique transcript biotypes: {len(transcript_biotypes)}\n\n")
        
        for biotype, count in transcript_biotypes.items():
            pct = 100 * count / len(df_filtered)
            f.write(f"  {biotype:45s}: {count:>8,} ({pct:>5.2f}%)\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"\nComparison report saved to: {output_path}")


def analyze_removed_transcripts(df_original, df_filtered):
    """Analyze what was removed"""
    print(f"\n{'='*80}")
    print("REMOVED TRANSCRIPTS ANALYSIS")
    print(f"{'='*80}")
    
    # Get removed transcripts
    removed_ids = set(df_original['transcript_id']) - set(df_filtered['transcript_id'])
    df_removed = df_original[df_original['transcript_id'].isin(removed_ids)]
    
    print(f"\nTotal removed: {len(df_removed):,}")
    
    # Gene biotypes of removed transcripts
    print(f"\nRemoved by gene biotype:")
    removed_gene_biotypes = df_removed['gene_biotype'].value_counts()
    for gene_biotype, count in removed_gene_biotypes.head(10).items():
        pct = 100 * count / len(df_removed)
        print(f"  {gene_biotype:45s}: {count:>8,} ({pct:>5.2f}%)")
    
    # Transcript biotypes of removed transcripts
    print(f"\nTop transcript biotypes in removed set:")
    removed_transcript_biotypes = df_removed['biotype'].value_counts()
    for biotype, count in removed_transcript_biotypes.head(10).items():
        pct = 100 * count / len(df_removed)
        print(f"  {biotype:45s}: {count:>8,} ({pct:>5.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Filter GENCODE biotypes by gene biotype',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--input', required=True,
                       help='Input biotype CSV file')
    parser.add_argument('--output', required=True,
                       help='Output filtered CSV file')
    parser.add_argument('--keep_gene_biotypes', nargs='+',
                       default=['lncRNA', 'protein_coding'],
                       help='Gene biotypes to keep (default: lncRNA protein_coding)')
    parser.add_argument('--report', default=None,
                       help='Output comparison report (default: <output>.report.txt)')
    
    args = parser.parse_args()
    
    # Load original data
    df_original = pd.read_csv(args.input)
    
    # Filter
    df_filtered = filter_by_gene_biotype(
        args.input,
        args.keep_gene_biotypes,
        args.output
    )
    
    if df_filtered is None:
        return
    
    # Analyze removed transcripts
    analyze_removed_transcripts(df_original, df_filtered)
    
    # Create comparison report
    report_path = args.report if args.report else Path(args.output).with_suffix('.report.txt')
    create_comparison_report(df_original, df_filtered, report_path)
    
    print(f"\n{'='*80}")
    print("FILTERING COMPLETE")
    print(f"{'='*80}")
    print(f"\nFiltered CSV: {args.output}")
    print(f"Report:       {report_path}")
    print(f"\nNext steps:")
    print(f"  1. Use this filtered CSV as biotype reference")
    print(f"  2. Extract sequences for these transcript IDs from FASTA")
    print(f"  3. Proceed with training on clean lncRNA vs protein_coding dataset")


if __name__ == '__main__':
    main()