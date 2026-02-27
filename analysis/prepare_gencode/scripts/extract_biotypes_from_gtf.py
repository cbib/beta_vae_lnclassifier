#!/usr/bin/env python3
"""
Extract Biotypes from GENCODE GTF
==================================
Parses GENCODE GTF annotation file to extract transcript biotypes.

Downloads GENCODE47 GTF (usable for 49 as well, simply change version number):
    wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/gencode.v47.annotation.gtf.gz
    gunzip gencode.v47.annotation.gtf.gz

Usage:
    python extract_biotypes_from_gtf.py \
        --gtf gencode.v47.annotation.gtf \
        --output gencode47_transcript_biotypes.csv
"""

import pandas as pd
import argparse
import re
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def parse_gtf_attributes(attribute_string: str) -> dict:
    """
    Parse GTF attribute string into dictionary.
    
    Example attribute string:
        gene_id "ENSG00000223972.5"; transcript_id "ENST00000456328.2"; 
        gene_type "transcribed_unprocessed_pseudogene"; 
        transcript_type "processed_transcript";
    
    Returns:
        Dictionary with parsed attributes
    """
    attributes = {}
    
    # Split by semicolon
    pairs = attribute_string.strip().split(';')
    
    for pair in pairs:
        pair = pair.strip()
        if not pair:
            continue
        
        # Split by first space to separate key and value
        parts = pair.split(' ', 1)
        if len(parts) != 2:
            continue
        
        key, value = parts
        # Remove quotes from value
        value = value.strip('"')
        attributes[key] = value
    
    return attributes


def extract_transcript_biotypes(gtf_path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Extract transcript biotypes from GENCODE GTF file.
    
    Args:
        gtf_path: Path to GTF file
        verbose: Print progress
    
    Returns:
        DataFrame with columns: transcript_id, biotype, gene_id, gene_name, gene_biotype
    """
    if verbose:
        print("="*80)
        print("EXTRACTING TRANSCRIPT BIOTYPES FROM GTF")
        print("="*80)
        print(f"Reading: {gtf_path}")
    
    transcript_data = defaultdict(dict)
    
    # Count lines for progress bar
    if verbose:
        print("\nCounting lines...")
        with open(gtf_path, 'r') as f:
            total_lines = sum(1 for line in f if not line.startswith('#'))
        print(f"Total annotation lines: {total_lines:,}")
    
    # Parse GTF
    if verbose:
        print("\nParsing GTF file...")
    
    with open(gtf_path, 'r') as f:
        iterator = tqdm(f, total=total_lines, desc="Processing") if verbose else f
        
        for line in iterator:
            # Skip comments
            if line.startswith('#'):
                continue
            
            # Split line
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            
            feature_type = fields[2]
            attributes_str = fields[8]
            
            # We only need transcript-level features
            if feature_type != 'transcript':
                continue
            
            # Parse attributes
            attributes = parse_gtf_attributes(attributes_str)
            
            # Extract transcript ID (with version)
            transcript_id = attributes.get('transcript_id', None)
            if not transcript_id:
                continue
            
            # Store transcript information
            transcript_data[transcript_id] = {
                'transcript_id': transcript_id,
                'biotype': attributes.get('transcript_type', 
                                         attributes.get('transcript_biotype', 'unknown')),
                'gene_id': attributes.get('gene_id', 'unknown'),
                'gene_name': attributes.get('gene_name', 'unknown'),
                'gene_biotype': attributes.get('gene_type', 
                                              attributes.get('gene_biotype', 'unknown'))
            }
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(transcript_data, orient='index')
    
    if verbose:
        print(f"\n{'='*80}")
        print("EXTRACTION COMPLETE")
        print(f"{'='*80}")
        print(f"Total transcripts: {len(df):,}")
        print(f"\nBiotype distribution (top 20):")
        biotype_counts = df['biotype'].value_counts().head(20)
        for biotype, count in biotype_counts.items():
            print(f"  {biotype:40s}: {count:6,} ({100*count/len(df):5.2f}%)")
        
        print(f"\nGene biotype distribution (top 10):")
        gene_biotype_counts = df['gene_biotype'].value_counts().head(10)
        for biotype, count in gene_biotype_counts.items():
            print(f"  {biotype:40s}: {count:6,} ({100*count/len(df):5.2f}%)")
    
    return df


def create_summary_report(df: pd.DataFrame, output_path: str):
    """Create a summary report of biotype statistics."""
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("GENCODE TRANSCRIPT BIOTYPE SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total transcripts: {len(df):,}\n\n")
        
        # Transcript biotypes
        f.write("TRANSCRIPT BIOTYPES:\n")
        f.write("-"*80 + "\n\n")
        biotype_counts = df['biotype'].value_counts()
        for biotype, count in biotype_counts.items():
            pct = 100 * count / len(df)
            f.write(f"  {biotype:45s}: {count:7,} ({pct:5.2f}%)\n")
        
        # Gene biotypes
        f.write("\n\nGENE BIOTYPES:\n")
        f.write("-"*80 + "\n\n")
        gene_biotype_counts = df['gene_biotype'].value_counts()
        for biotype, count in gene_biotype_counts.items():
            pct = 100 * count / len(df)
            f.write(f"  {biotype:45s}: {count:7,} ({pct:5.2f}%)\n")
        
        # Key categories
        f.write("\n\nKEY CATEGORIES:\n")
        f.write("-"*80 + "\n\n")
        
        # Protein coding
        pc_count = df[df['biotype'] == 'protein_coding'].shape[0]
        f.write(f"Protein coding: {pc_count:,} ({100*pc_count/len(df):.2f}%)\n")
        
        # lncRNA
        lnc_count = df[df['biotype'] == 'lncRNA'].shape[0]
        f.write(f"lncRNA: {lnc_count:,} ({100*lnc_count/len(df):.2f}%)\n")
        
        # Pseudogenes (all types)
        pseudo_count = df[df['biotype'].str.contains('pseudogene', case=False, na=False)].shape[0]
        f.write(f"Pseudogenes (all types): {pseudo_count:,} ({100*pseudo_count/len(df):.2f}%)\n")
        
        # NMD
        nmd_count = df[df['biotype'].str.contains('nonsense_mediated_decay', case=False, na=False)].shape[0]
        f.write(f"Nonsense-mediated decay: {nmd_count:,} ({100*nmd_count/len(df):.2f}%)\n")
        
        # IG genes
        ig_count = df[df['biotype'].str.contains('IG_', case=False, na=False)].shape[0]
        f.write(f"Immunoglobulin genes: {ig_count:,} ({100*ig_count/len(df):.2f}%)\n")
        
        # TR genes
        tr_count = df[df['biotype'].str.contains('TR_', case=False, na=False)].shape[0]
        f.write(f"T-cell receptor genes: {tr_count:,} ({100*tr_count/len(df):.2f}%)\n")
        
        # Retained intron
        ri_count = df[df['biotype'].str.contains('retained_intron', case=False, na=False)].shape[0]
        f.write(f"Retained intron: {ri_count:,} ({100*ri_count/len(df):.2f}%)\n")
        
        f.write("\n" + "="*80 + "\n")


def create_visualization(df: pd.DataFrame, output_path: str):
    """Create visualization of biotype distribution."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Top transcript biotypes
    ax = axes[0]
    top_biotypes = df['biotype'].value_counts().head(15)
    colors = plt.cm.tab20(range(len(top_biotypes)))
    
    ax.barh(range(len(top_biotypes)), top_biotypes.values, color=colors, edgecolor='black')
    ax.set_yticks(range(len(top_biotypes)))
    ax.set_yticklabels(top_biotypes.index, fontsize=9)
    ax.set_xlabel('Count', fontsize=11)
    ax.set_title('Top 15 Transcript Biotypes', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add count labels
    for i, v in enumerate(top_biotypes.values):
        ax.text(v + 500, i, f'{v:,}', va='center', fontsize=8)
    
    # 2. Pie chart of major categories
    ax = axes[1]
    
    # Group into major categories
    categories = {
        'Protein coding': df[df['biotype'] == 'protein_coding'].shape[0],
        'lncRNA': df[df['biotype'] == 'lncRNA'].shape[0],
        'Pseudogenes': df[df['biotype'].str.contains('pseudogene', case=False, na=False)].shape[0],
        'IG/TR genes': df[df['biotype'].str.contains('IG_|TR_', case=False, na=False)].shape[0],
        'NMD': df[df['biotype'].str.contains('nonsense_mediated_decay', case=False, na=False)].shape[0],
        'Retained intron': df[df['biotype'].str.contains('retained_intron', case=False, na=False)].shape[0],
    }
    
    # Add "Other" category
    accounted = sum(categories.values())
    categories['Other'] = len(df) - accounted
    
    # Filter out zero counts and sort
    categories = {k: v for k, v in categories.items() if v > 0}
    sorted_categories = dict(sorted(categories.items(), key=lambda x: x[1], reverse=True))
    
    colors_pie = plt.cm.Set3(range(len(sorted_categories)))
    wedges, texts, autotexts = ax.pie(
        sorted_categories.values(),
        labels=sorted_categories.keys(),
        autopct='%1.1f%%',
        colors=colors_pie,
        startangle=90,
        textprops={'fontsize': 10}
    )
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    ax.set_title('Major Transcript Categories', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=1200, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract transcript biotypes from GENCODE GTF file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--gtf', required=True,
                        help='Path to GENCODE GTF file')
    parser.add_argument('--output', required=True,
                        help='Output CSV file for transcript biotypes')
    parser.add_argument('--summary', default=None,
                        help='Output summary report (optional, default: <output>.summary.txt)')
    parser.add_argument('--plot', default=None,
                        help='Output visualization (optional, default: <output>.png)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Extract biotypes
    df = extract_transcript_biotypes(args.gtf, verbose=not args.quiet)
    
    # Save CSV
    output_path = Path(args.output)
    df.to_csv(output_path, index=False)
    
    if not args.quiet:
        print(f"\nSaved biotype data to: {output_path}")
        print(f"Columns: {', '.join(df.columns)}")
    
    # Create summary report
    if args.summary:
        summary_path = args.summary
    else:
        summary_path = output_path.with_suffix('.summary.txt')
    
    create_summary_report(df, summary_path)
    
    if not args.quiet:
        print(f"Saved summary report to: {summary_path}")
    
    # Create visualization
    if args.plot:
        plot_path = args.plot
    else:
        plot_path = output_path.with_suffix('.png')
    
    try:
        create_visualization(df, plot_path)
    except Exception as e:
        if not args.quiet:
            print(f"Warning: Could not create visualization: {e}")
    
    if not args.quiet:
        print("\n" + "="*80)
        print("EXTRACTION COMPLETE")
        print("="*80)


if __name__ == '__main__':
    main()