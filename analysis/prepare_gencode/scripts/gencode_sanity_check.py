#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sanity check script for GENCODE data before training
"""
import sys
import json
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
    create_length_stratified_groups,
    gene_overlap,
    get_cv_splits,
    get_gene_ids,
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
    print(f"\nPipe-delimited format: {'  Yes' if has_pipe else ' No'}")
    
    if has_pipe:
        # Extract base IDs
        base_ids = [seq.id.split('|')[0] for seq in sequences[:n_samples]]
        print(f"\nBase IDs (after split on '|'):")
        for i, base_id in enumerate(base_ids):
            print(f"  {i+1:2d}. {base_id}")
            
        # Check version numbers
        has_version = all('.' in base_id for base_id in base_ids)
        print(f"\nVersion numbers present: {'  Yes' if has_version else ' No'}")
        
        if has_version:
            print("\nNote: Biotype CSV should use transcript IDs WITH version numbers")
            print("      (e.g., 'ENST00000456328.2' not 'ENST00000456328')")


def create_length_distribution_plot(lnc_lengths, pc_lengths, output_path, max_length=15000):
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
    ax.axvline(max_length, color='red', linestyle='--', linewidth=2, label=f'max_length={max_length}')
    
    # Box plot
    ax = axes[1]
    bp = ax.boxplot([lnc_lengths, pc_lengths], patch_artist=True, showfliers=False)
    ax.set_xticks([1, 2], ['lncRNA', 'PC'])
    
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


# ---------------------------------------------------------------------------
# Gene-level split audit (--split_dir)
# ---------------------------------------------------------------------------

def check_gene_overlap(split_dir, n_folds=5, random_state=42, n_bins=5,
                       group_by_gene=False):
    """Genes shared between test and train/val, and between CV train/val
    folds (legacy and configured fold construction). Returns a dict."""
    print("\n" + "=" * 80)
    print("GENE-LEVEL SPLIT AUDIT")
    print("=" * 80)
    sd = Path(split_dir)
    parts = {k: list(SeqIO.parse(str(sd / f"{k}.fa"), "fasta"))
             for k in ("lnc_trainval", "pc_trainval", "lnc_test", "pc_test")}
    trainval = parts["lnc_trainval"] + parts["pc_trainval"]
    tv_labels = (["lnc"] * len(parts["lnc_trainval"])
                 + ["pc"] * len(parts["pc_trainval"]))
    tv_genes = set(get_gene_ids(trainval))

    out = {"test": {}, "cv": {}}
    print("\nTest vs train/val (records whose gene also occurs in train/val):")
    n_all = shared_all = 0
    for cls in ("lnc", "pc"):
        g = get_gene_ids(parts[f"{cls}_test"])
        shared = sum(x in tv_genes for x in g)
        out["test"][cls] = {"n": len(g), "n_shared": shared,
                            "frac_shared": shared / len(g),
                            "n_genes": len(set(g))}
        n_all += len(g)
        shared_all += shared
        print(f"  {cls:<4} {shared:>7,} / {len(g):>7,}  ({100*shared/len(g):.1f}%)"
              f"   test genes: {len(set(g)):,}")
    out["test"]["all"] = {"n": n_all, "n_shared": shared_all,
                          "frac_shared": shared_all / n_all}
    print(f"  all  {shared_all:>7,} / {n_all:>7,}  ({100*shared_all/n_all:.1f}%)")

    genes = get_gene_ids(trainval)
    print("\nCV folds on train/val (val records whose gene occurs in fold train):")
    for mode in (False, True):
        splits = get_cv_splits(trainval, tv_labels, n_folds, random_state,
                               n_bins, group_by_gene=mode)
        fr = [gene_overlap(genes, tr, va)[0] / len(va) for tr, va in splits]
        key = "gene_grouped" if mode else "legacy"
        out["cv"][key] = {"frac_per_fold": fr, "mean_frac": float(np.mean(fr))}
        tag = "  <- configured" if mode == group_by_gene else ""
        print(f"  {key:<13} mean {100*np.mean(fr):5.1f}%   folds: "
              f"{', '.join(f'{100*x:.1f}' for x in fr)}{tag}")
    return out


# ---------------------------------------------------------------------------
# Genomic vs processed feature audit (--config)
# ---------------------------------------------------------------------------

_META_COLS = {"transcript_type", "coding_class", "transcript_length",
              "simple_ID", "length"}


def _norm_tid(x):
    return str(x).split("|")[0].split(".")[0]


def _load_origin_csv(path):
    """(df first-occurrence per id, n duplicate rows, n ids whose duplicate
    rows differ)."""
    df = pd.read_csv(path, index_col="transcript_id")
    df.index = df.index.map(_norm_tid)
    df = df[[c for c in df.columns if c not in _META_COLS]]
    n_dup = int(df.index.duplicated(keep="first").sum())
    n_conflict = 0
    if n_dup:
        d = df[df.index.duplicated(keep=False)]
        n_conflict = int((d.groupby(level=0).nunique(dropna=False)
                          .max(axis=1) > 1).sum())
    return df[~df.index.duplicated(keep="first")], n_dup, n_conflict


def _gene_eta2(df, genes):
    """(eta2, adjusted eta2) of gene identity per column, over genes with
    >= 2 records. Raw eta2 has a chance baseline of about (G-1)/(N-1); the
    adjusted value is ~0 when genes explain nothing, 1 when isoforms are
    interchangeable."""
    counts = genes.map(genes.value_counts())
    keep = (counts >= 2).to_numpy()
    sub, g = df[keep], genes[keep].to_numpy()
    if len(sub) == 0:
        nan = pd.Series(np.nan, index=df.columns)
        return nan, nan
    n, k = len(sub), len(set(g))
    ssw = ((sub - sub.groupby(g).transform("mean")) ** 2).sum()
    sst = ((sub - sub.mean()) ** 2).sum().where(lambda x: x > 0)
    return 1 - ssw / sst, 1 - (ssw / (n - k)) / (sst / (n - 1))


_NO_SCALED = {"scaled_std_g": np.nan, "scaled_std_p": np.nan,
              "scaled_median_p": np.nan, "frac_p_outside01": np.nan}


def _scaled_diagnostics(config, block, g, p):
    """Per-feature effect of the shared (genomic-fit) scaler on both origins,
    for robust/minmax features: {name: dict}. Empty if no bank is configured."""
    from data.feature_registry import REGISTRY
    from data.feature_scaler import FeatureScalerBank
    path = config.get("data", f"{block}_scaler_bank_path", default=None)
    metas = REGISTRY.block_features(block)
    names = [m.name for m in metas]
    if not path or not Path(path).exists() or not set(names) <= set(g.columns):
        return {}
    bank = FeatureScalerBank.load(path)
    sg_ = bank.transform(g[names].to_numpy(np.float64))
    sp_ = bank.transform(p[names].to_numpy(np.float64))
    out = {}
    for i, m in enumerate(metas):
        if m.scale == "none":
            continue
        d = {"scaled_std_g": float(sg_[:, i].std()),
             "scaled_std_p": float(sp_[:, i].std()),
             "scaled_median_p": float(np.median(sp_[:, i])),
             "frac_p_outside01": (float(((sp_[:, i] < 0) | (sp_[:, i] > 1)).mean())
                                  if m.scale == "minmax" else np.nan)}
        out[m.name] = d
    return out


def check_origin_redundancy(config, records, labels, out_dir):
    """Per TE / NonB feature: agreement of the genomic and processed
    versions across transcripts and how gene-determined each version is.
    NonB2 has one (processed) source and is skipped."""
    from data.feature_registry import REGISTRY

    print("\n" + "=" * 80)
    print("GENOMIC vs PROCESSED FEATURE AUDIT")
    print("=" * 80)
    ids = [_norm_tid(r.id) for r in records]
    gene_of = dict(zip(ids, get_gene_ids(records)))
    label_of = dict(zip(ids, labels))
    feat_rows, summary = [], {}

    for block in ("te", "nonb"):
        g_path = config.get("data", f"{block}_genomic_csv")
        p_path = config.get("data", f"{block}_processed_csv")
        g_df, g_dup, g_conf = _load_origin_csv(g_path)
        p_df, p_dup, p_conf = _load_origin_csv(p_path)
        common = [i for i in dict.fromkeys(ids) if i in g_df.index and i in p_df.index]
        cols = [c for c in g_df.columns if c in p_df.columns]
        g = g_df.loc[common, cols].astype(np.float64)
        p = p_df.loc[common, cols].astype(np.float64)
        eq = np.isclose(g.to_numpy(), p.to_numpy(), rtol=1e-6, atol=1e-9)
        row_same = eq.all(axis=1)
        cls = np.array([label_of[i] for i in common])
        genes = pd.Series([gene_of[i] for i in common], index=common)
        sp = g.rank().corrwith(p.rank())
        (eta_g, adj_g), (eta_p, adj_p) = _gene_eta2(g, genes), _gene_eta2(p, genes)
        sd_g, sd_p = g.std(), p.std()
        sg_of = {m.name: m.subgroup for m in REGISTRY.block_features(block)}
        sc = _scaled_diagnostics(config, block, g, p)
        for j, c in enumerate(cols):
            feat_rows.append({"block": block, "feature": c,
                              "subgroup": sg_of.get(c, "?"),
                              "frac_equal": float(eq[:, j].mean()),
                              "spearman": float(sp[c]),
                              "eta2_genomic": float(eta_g[c]),
                              "eta2_processed": float(eta_p[c]),
                              "eta2_adj_genomic": float(adj_g[c]),
                              "eta2_adj_processed": float(adj_p[c]),
                              "mean_genomic": float(g[c].mean()),
                              "mean_processed": float(p[c].mean()),
                              "std_genomic": float(sd_g[c]),
                              "std_processed": float(sd_p[c]),
                              **sc.get(c, _NO_SCALED)})
        summary[block] = {
            "n_transcripts": len(common), "n_features": len(cols),
            "dup_rows_genomic": g_dup, "dup_ids_conflicting_genomic": g_conf,
            "dup_rows_processed": p_dup, "dup_ids_conflicting_processed": p_conf,
            "frac_identical_vector": float(row_same.mean()),
            "frac_identical_vector_lnc": float(row_same[cls == "lnc"].mean()),
            "frac_identical_vector_pc": float(row_same[cls == "pc"].mean()),
        }
        s = summary[block]
        print(f"\n[{block}] {s['n_transcripts']:,} transcripts, {s['n_features']} features")
        print(f"  duplicate rows (genomic/processed): {g_dup:,}/{p_dup:,}; "
              f"ids with differing duplicates: {g_conf:,}/{p_conf:,}")
        print(f"  transcripts with identical genomic and processed vectors: "
              f"{100*s['frac_identical_vector']:.1f}%  "
              f"(lnc {100*s['frac_identical_vector_lnc']:.1f}%, "
              f"pc {100*s['frac_identical_vector_pc']:.1f}%)")

    feat = pd.DataFrame(feat_rows)
    sub = (feat.groupby(["block", "subgroup"])
           .agg(n_features=("feature", "size"), frac_equal=("frac_equal", "mean"),
                spearman_median=("spearman", "median"),
                eta2_genomic=("eta2_genomic", "mean"),
                eta2_processed=("eta2_processed", "mean"),
                adj_genomic=("eta2_adj_genomic", "mean"),
                adj_processed=("eta2_adj_processed", "mean")).reset_index())
    ratio = (feat.assign(r=feat.std_processed / feat.std_genomic.where(feat.std_genomic > 0),
                         rs=feat.scaled_std_p / feat.scaled_std_g.where(feat.scaled_std_g > 0))
             .groupby(["block", "subgroup"])
             .agg(std_ratio_p_over_g=("r", "median"),
                  scaled_std_ratio=("rs", "median"),
                  pct_p_outside01=("frac_p_outside01", "mean")))
    sub = sub.merge(ratio.reset_index(), on=["block", "subgroup"])
    print("\nPer subgroup (token): mean share of equal values, median Spearman,")
    print("gene-determined variance (eta2 raw / adj) of each version, median")
    print("raw std ratio processed/genomic; scaled_std_ratio = same after the shared")
    print("genomic-fit scaler (robust/minmax features only); pct_p_outside01 = share of")
    print("processed minmax values outside [0, 1]")
    print(sub.round(3).to_string(index=False))
    out_dir = Path(out_dir)
    feat.to_csv(out_dir / "origin_redundancy_features.csv", index=False)
    sub.to_csv(out_dir / "origin_redundancy_subgroups.csv", index=False)
    return summary


class _Tee:
    """Mirror stdout into a log file so the full report is kept."""
    def __init__(self, path):
        self.f, self.out = open(path, "w"), sys.stdout
    def write(self, x):
        self.out.write(x); self.f.write(x)
    def flush(self):
        self.out.flush(); self.f.flush()


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
    parser.add_argument('--split_dir', default=None,
                       help='Dir with {lnc,pc}_{trainval,test}.fa: gene-overlap audit')
    parser.add_argument('--config', default=None,
                       help='Model config: adds the genomic-vs-processed feature audit')
    parser.add_argument('--require_gene_disjoint', action='store_true',
                       help='Exit 1 if test or the configured CV folds share genes')
    MAX_LENGTH = 15000
    args = parser.parse_args()
    
    OUTPUT_DIR = Path("gencode_sanity_check")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    sys.stdout = _Tee(OUTPUT_DIR / "sanity_check_full.log")
    
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
    create_length_distribution_plot(lnc_lengths, pc_lengths, length_plot_path, MAX_LENGTH)
    
    # 8. Create summary report
    summary_path = OUTPUT_DIR / "sanity_check_summary.txt"
    create_summary_report(lnc_seqs, pc_seqs, biotypes, lnc_lengths, pc_lengths, summary_path)

    # 9. Gene-level split audit and genomic-vs-processed feature audit
    audit, cfg = {}, None
    if args.config:
        from configs.load_config import load_config
        cfg = load_config(args.config)
    if args.split_dir:
        kw = {}
        if cfg is not None:
            kw = dict(n_folds=cfg.get("training", "n_folds"),
                      random_state=cfg.get("training", "random_state", default=42),
                      n_bins=cfg.get("training", "n_bins", default=5),
                      group_by_gene=bool(cfg.get("training", "group_by_gene",
                                                 default=False)))
        audit["gene_overlap"] = check_gene_overlap(args.split_dir, **kw)
        go = audit["gene_overlap"]
        cv_key = "gene_grouped" if kw.get("group_by_gene") else "legacy"
        leaks = (go["test"]["all"]["n_shared"] > 0
                 or go["cv"][cv_key]["mean_frac"] > 0)
        if args.require_gene_disjoint and leaks:
            print("\n  FAIL: test or configured CV folds share genes with train/val.")
            sys.exit(1)
    if cfg is not None:
        tv_l, tv_p = (list(SeqIO.parse(cfg.get("data", k), "fasta"))
                      for k in ("lnc_fasta", "pc_fasta"))
        audit["origin"] = check_origin_redundancy(
            cfg, tv_l + tv_p, ["lnc"] * len(tv_l) + ["pc"] * len(tv_p), OUTPUT_DIR)
    if audit:
        (OUTPUT_DIR / "gene_origin_audit.json").write_text(json.dumps(audit, indent=2))
    
    print("\n" + "=" * 80)
    print("SANITY CHECK COMPLETE")
    print("=" * 80)
    print("\nSanity check finished; see the audit sections above for pass/fail.")
    print(f"\nOutput files:")
    print(f"  - {length_plot_path}")
    print(f"  - {summary_path}")
    print(f"  - {OUTPUT_DIR / 'sanity_check_full.log'}  (everything printed above)")


if __name__ == '__main__':
    main()