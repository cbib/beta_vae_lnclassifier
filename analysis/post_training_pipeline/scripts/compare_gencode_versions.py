import os
import sys
from pathlib import Path
import argparse
from Bio import SeqIO
import pandas as pd
import time

script_dir = Path(__file__).parent
workflow_dir = script_dir.parent
sys.path.insert(0, str(workflow_dir))

def load_sequences(fasta_file, as_dict=False):
    """
    Load sequences from a FASTA file.

    Args:
        fasta_file: Path to FASTA file
        as_dict: If True, return a dictionary with sequence IDs as keys. Otherwise, return a list.
        
    Returns:
        List of SeqRecord objects
    """
    if as_dict:
        return SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
    else:
        return list(SeqIO.parse(fasta_file, "fasta"))


def simple_load_ids(fasta_file):
    """
    Load sequences from a FASTA file and return a list of ids.
    
    Args:
        fasta_file: Path to FASTA file
        
    Returns:
        List sequence ids
    """
    ids = []
    with open(fasta_file) as handle:
        for title, sequence in SeqIO.FastaIO.SimpleFastaParser(handle):
            ids.append(title)
    return ids


def parse_gencode_ids(record_ids: list, to_df=False):
    """
    Parse GENCODE transcript IDs to extract relevant information.
        
    Args:
        records: List of SeqRecord objects
    Returns:
        List of dictionaries with parsed information
    """
    full_ids = []
    seq_ids = []
    clean_ids = []
    id_versions = []
    biotypes = []
    
    for full_id in record_ids:
        parts = full_id.split("|")
        full_ids.append(full_id)
        seq_ids.append(parts[0])
        id_parts = parts[0].split(".")
        clean_ids.append(id_parts[0])
        id_versions.append(id_parts[1] if len(id_parts) > 1 else "")
        biotypes.append(parts[-2])
    
    if to_df:
        df = pd.DataFrame({
            'full_id': full_ids,
            'seq_id': seq_ids,
            'clean_id': clean_ids,
            'id_version': id_versions,
            'biotype': biotypes
        })
        df.set_index('clean_id', inplace=True)
        return df
    
    return full_ids, seq_ids, clean_ids, biotypes


def write_sequences(sequences, output_file):
    """
    Write sequences to a FASTA file.
    
    Args:
        sequences: List of SeqRecord objects
        output_file: Path to output FASTA file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    SeqIO.write(sequences, output_file, "fasta")


def process_version_data(fasta_file, pc_file, lnc_file, version_label):
    """Process data for a single version."""
    print(f"--- Processing {version_label.upper()} dataset ---")
    loop_start = time.time()
    
    # Load full transcript dataset
    start = time.time()
    all_transcripts = load_sequences(fasta_file, as_dict=True)
    print(f"  ✓ Loaded {version_label}_fasta: {len(all_transcripts)} sequences in {time.time()-start:.2f}s")

    # Create dataframe with info extracted from IDs
    start = time.time()
    df = parse_gencode_ids(list(all_transcripts.keys()), to_df=True)
    df.rename(columns={col: f"{version_label}_{col}" for col in df.columns}, inplace=True)
    print(f"  ✓ Parsed IDs and created dataframe in {time.time()-start:.2f}s")

    # Load pc and lncRNA transcripts
    start = time.time()
    pc_transcripts = simple_load_ids(pc_file)
    _, _, pc_clean_ids, _ = parse_gencode_ids(list(pc_transcripts))
    print(f"  ✓ Loaded {version_label}_pc: {len(pc_transcripts)} sequences in {time.time()-start:.2f}s")
    
    start = time.time()
    lnc_transcripts = simple_load_ids(lnc_file)
    _, _, lnc_clean_ids, _ = parse_gencode_ids(list(lnc_transcripts))
    print(f"  ✓ Loaded {version_label}_lnc: {len(lnc_transcripts)} sequences in {time.time()-start:.2f}s")

    # Classify transcripts
    start = time.time()
    df[f"{version_label}_class"] = "NA"
    df.loc[df.index.isin(pc_clean_ids), f"{version_label}_class"] = "pc"
    df.loc[df.index.isin(lnc_clean_ids), f"{version_label}_class"] = "lncRNA"
    print(f"  ✓ Classified transcripts in {time.time()-start:.2f}s")
    
    print(f"  Total for {version_label}: {time.time()-loop_start:.2f}s")
    
    return df


def analyze_differences(combined_df):
    """Analyze differences between old and new versions."""
    # Check presence in old and new versions
    combined_df["is_common"] = combined_df["old_full_id"].notna() & combined_df["new_full_id"].notna()
    # If not common:
    combined_df["transcript_added"] = combined_df["old_full_id"].isna() & combined_df["new_full_id"].notna()
    combined_df["transcript_removed"] = combined_df["new_full_id"].isna() & combined_df["old_full_id"].notna()

    # All transcript have assigned versions and biotypes. Check changes.
    combined_df["version_changed"] = combined_df["old_id_version"] != combined_df["new_id_version"]
    combined_df["biotype_changed"] = combined_df["old_biotype"] != combined_df["new_biotype"]

    # Not all transcripts have assigned coding class.
    combined_df["both_have_class"] = combined_df["is_common"] & (combined_df["old_class"] != "NA") & (combined_df["new_class"] != "NA")
    # Same class only if both old and new classes are assigned and equal
    combined_df["same_class"] = (combined_df["both_have_class"]) & (combined_df["old_class"] == combined_df["new_class"])
    # Class changed only if both old and new classes are assigned and different
    combined_df["class_changed"] = (combined_df["both_have_class"]) & (combined_df["old_class"] != combined_df["new_class"])
    
    # If not both have assigned class, mark as class added or removed (not changed!)
    combined_df["class_added"] = combined_df["is_common"] & (combined_df["old_class"] == "NA") & (combined_df["new_class"] != "NA")
    combined_df["class_removed"] = combined_df["is_common"] & (combined_df["old_class"] != "NA") & (combined_df["new_class"] == "NA")

    # Define categories
    ## 1. Common transcripts with no class change (both versions have a coding class assigned)
    combined_df["common_same_class"] = (combined_df["is_common"]) & (combined_df["same_class"])
    combined_df["common_class_change"] = (combined_df["is_common"]) & (combined_df["class_changed"])
    ### Subcategories (not exported to fasta)
    combined_df["pc_to_lncRNA"] = combined_df['common_class_change'] & (combined_df['old_class'] == 'pc') & (combined_df['new_class'] == 'lncRNA')
    combined_df["lncRNA_to_pc"] = combined_df['common_class_change'] & (combined_df['old_class'] == 'lncRNA') & (combined_df['new_class'] == 'pc')
    ## 3. New transcripts with assigned class
    combined_df["added_with_class"] = (combined_df["transcript_added"]) & (combined_df['new_class'] != 'NA')
    ### Subcategories (not exported to fasta)
    combined_df["added_pc"] = combined_df['added_with_class'] & (combined_df['new_class'] == 'pc')
    combined_df["added_lncRNA"] = combined_df['added_with_class'] & (combined_df['new_class'] == 'lncRNA')
    ### 3b. Removed transcripts with assigned class
    combined_df["removed_with_class"] = (combined_df["transcript_removed"]) & (combined_df['old_class'] != 'NA')
    combined_df["removed_pc"] = combined_df['removed_with_class'] & (combined_df['old_class'] == 'pc')
    combined_df["removed_lncRNA"] = combined_df['removed_with_class'] & (combined_df['old_class'] == 'lncRNA')
    ## 4. All transcripts present in new version without assigned class
    combined_df["no_class"] = (combined_df["new_full_id"].notna()) & ((combined_df["new_class"] == "NA"))

    # Log descriptive statistics
    print("=" * 60)
    print("TRANSCRIPT ANALYSIS SUMMARY")
    print("=" * 60)
    
    total_transcripts = len(combined_df)
    print(f"Total transcripts analyzed: {total_transcripts}")
    print(f"In old version: {combined_df['old_full_id'].notna().sum()}")
    print(f"In new version: {combined_df['new_full_id'].notna().sum()}")
    print(f"Common transcripts: {combined_df['is_common'].sum()}")
    print(f"Transcripts added in new version: {combined_df['transcript_added'].sum()}")
    print(f"Transcripts removed in new version: {combined_df['transcript_removed'].sum()}")

    print("")
    print("Changes among common transcripts:")
    print(f"  - Version changed: {combined_df.loc[combined_df['is_common'], 'version_changed'].sum()}")
    print(f"  - Biotype changed: {combined_df.loc[combined_df['is_common'], 'biotype_changed'].sum()}")
    print("")

    print("Transcript groups regarding coding class:")
    print(f"  - Common with same class (DATASET): {combined_df['common_same_class'].sum()}")
    print(f"     - pc: {combined_df.loc[combined_df['common_same_class'] & (combined_df['old_class'] == 'pc')].shape[0]}")
    print(f"     - lncRNA: {combined_df.loc[combined_df['common_same_class'] & (combined_df['old_class'] == 'lncRNA')].shape[0]}")
    print(f"  - Common with class change: {combined_df['common_class_change'].sum()}")
    print(f"     - pc to lncRNA: {combined_df.loc[combined_df['common_class_change'] & (combined_df['old_class'] == 'pc') & (combined_df['new_class'] == 'lncRNA')].shape[0]}")
    print(f"     - lncRNA to pc: {combined_df.loc[combined_df['common_class_change'] & (combined_df['old_class'] == 'lncRNA') & (combined_df['new_class'] == 'pc')].shape[0]}")
    print(f"  - New transcripts with assigned class: {combined_df['added_with_class'].sum()}")
    print(f"     - pc: {combined_df.loc[combined_df['added_with_class'] & (combined_df['new_class'] == 'pc')].shape[0]}")
    print(f"     - lncRNA: {combined_df.loc[combined_df['added_with_class'] & (combined_df['new_class'] == 'lncRNA')].shape[0]}")
    print(f"  - Removed transcripts with assigned class: {combined_df['removed_with_class'].sum()}")
    print(f"     - pc: {combined_df.loc[combined_df['removed_with_class'] & (combined_df['old_class'] == 'pc')].shape[0]}")
    print(f"     - lncRNA: {combined_df.loc[combined_df['removed_with_class'] & (combined_df['old_class'] == 'lncRNA')].shape[0]}")
    print(f"  - Transcripts with class removed in new version: {(combined_df['class_removed']).sum()}")
    print(f"     - pc: {(combined_df['class_removed'] & (combined_df['old_class'] == 'pc')).sum()}")
    print(f"     - lncRNA: {(combined_df['class_removed'] & (combined_df['old_class'] == 'lncRNA')).sum()}")
    print(f"  - Transcripts with class added in new version: {(combined_df['class_added']).sum()}")
    print(f"     - pc: {(combined_df['class_added'] & (combined_df['new_class'] == 'pc')).sum()}")
    print(f"     - lncRNA: {(combined_df['class_added'] & (combined_df['new_class'] == 'lncRNA')).sum()}")
    print(f"  - Transcripts in new version (common + added) without assigned class: {combined_df['no_class'].sum()}")

    print("=" * 60)

    return combined_df


def export_categories(combined_df, new_fasta, output_dir, old_version, new_version):
    """Export transcript categories to separate files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Update TSV filename to include version numbers
    comparison_file = os.path.join(output_dir, f"gencode_v{old_version}_v{new_version}_comparison.tsv")
    combined_df.to_csv(comparison_file, sep="\t")
    print(f"✓ Saved comparison to {comparison_file}")
    
    # Load new sequences for extraction
    new_sequences = load_sequences(new_fasta, as_dict=True)
    print(f"Loaded new FASTA sequences for extraction: {len(new_sequences)} sequences")
    
    # Map internal names to output filenames (matching gencode_novelty_embeddings.py expectations)
    categories = {
        "common_same_class": "common_no_class_change",
        "common_class_change": "common_class_change", 
        "added_with_class": "new_with_class",
        "no_class": "no_class"
    }

    for internal_name, output_name in categories.items():
        df = combined_df[combined_df[internal_name] == True]
        print(f"Processing category: {internal_name} → {output_name} ({df.shape[0]} transcripts)")
        
        # Add 'v' before version number to match expected format
        output_fasta = os.path.join(output_dir, f"gencode.v{new_version}.{output_name}_transcripts.fa")
        
        # Extract sequences
        seq_ids = df["new_full_id"].tolist()
        seqs_to_write = []
        for seq_id in seq_ids:
            if seq_id in new_sequences:
                seqs_to_write.append(new_sequences[seq_id])
        
        write_sequences(seqs_to_write, output_fasta)
        print(f"  ✓ Written {len(seqs_to_write)} sequences to {output_fasta}")


def main():
    """Main function to handle command-line execution."""
    parser = argparse.ArgumentParser(
        description="Compare GENCODE transcript versions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Example usage:
            python compare_gencode_versions.py \\
                --old-fasta data/raw/gencode.v46.transcripts.fa \\
                --old-pc data/raw/gencode.v46.pc_transcripts.fa \\
                --old-lnc data/raw/gencode.v46.lncRNA_transcripts.fa \\
                --new-fasta data/raw/gencode.v47.transcripts.fa \\
                --new-pc data/raw/gencode.v47.pc_transcripts.fa \\
                --new-lnc data/raw/gencode.v47.lncRNA_transcripts.fa \\
                --old-version 46 \\
                --new-version 47 \\
                --output-dir resources/gencode_comparison_v46_v47
            """
    )
    parser.add_argument("--old-fasta", required=True, help="Old version FASTA file")
    parser.add_argument("--old-pc", required=True, help="Old version protein-coding file")
    parser.add_argument("--old-lnc", required=True, help="Old version lncRNA file")
    parser.add_argument("--new-fasta", required=True, help="New version FASTA file")
    parser.add_argument("--new-pc", required=True, help="New version protein-coding file")
    parser.add_argument("--new-lnc", required=True, help="New version lncRNA file")
    parser.add_argument("--old-version", required=True, help="Old version number (e.g., 46)")
    parser.add_argument("--new-version", required=True, help="New version number (e.g., 47)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    
    
    args = parser.parse_args()
    print("Running in command-line mode")

    print("=" * 60)
    print("Starting data loading and processing...")
    print("=" * 60)
    
    # Process old version
    old_df = process_version_data(args.old_fasta, args.old_pc, args.old_lnc, "old")
    
    # Process new version
    new_df = process_version_data(args.new_fasta, args.new_pc, args.new_lnc, "new")
    
    # Combine dataframes
    combined_df = old_df.join(new_df, how="outer")
    
    print(f"{'=' * 60}")
    print(f"✓ Processing complete! Combined dataframe shape: {combined_df.shape}")
    print(f"{'=' * 60}")
    
    # Analyze differences
    combined_df = analyze_differences(combined_df)
    
    # Export categories
    export_categories(combined_df, args.new_fasta, args.output_dir, args.old_version, args.new_version)
    
    print(f"{'=' * 60}")
    print("All tasks completed successfully!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
