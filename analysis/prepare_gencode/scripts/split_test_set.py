#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split CD-HIT filtered lncRNA and mRNA FASTA files into:
  - 95% train/val  → used as input to existing CV training scripts
  -  5% test       → held-out independent test set

Stratified by class (lnc / pc) to preserve class ratio.
Outputs four FASTA files and a JSON manifest with split statistics.

Usage:
    python split_test_set.py \
        --lnc_fasta data/lnc_cdhit.fa \
        --pc_fasta  data/pc_cdhit.fa  \
        --output_dir data/splits/      \
        --test_size 0.05               \
        --seed 42
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

from Bio import SeqIO


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_fasta(path: str) -> list:
    """Return list of SeqRecord objects."""
    records = list(SeqIO.parse(path, "fasta"))
    if not records:
        raise ValueError(f"No sequences found in {path}")
    return records


def stratified_split(records: list, test_size: float, seed: int):
    """
    Simple stratified split on a single class (all records same label).
    Returns (train_val, test) as lists of SeqRecord.
    """
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    n_test = max(1, round(len(shuffled) * test_size))
    test = shuffled[:n_test]
    train_val = shuffled[n_test:]
    return train_val, test


def write_fasta(records: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        SeqIO.write(records, f, "fasta")
    print(f"  Written: {path}  ({len(records):,} sequences)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Split lnc/pc FASTAs into train/val and independent test sets"
    )
    parser.add_argument("--lnc_fasta",   required=True,  help="CD-HIT filtered lncRNA FASTA")
    parser.add_argument("--pc_fasta",    required=True,  help="CD-HIT filtered mRNA FASTA")
    parser.add_argument("--output_dir",  required=True,  help="Directory for output files")
    parser.add_argument("--test_size",   type=float, default=0.05,
                        help="Fraction of each class to hold out as test (default: 0.05)")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    assert 0 < args.test_size < 1, "--test_size must be between 0 and 1"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Loading FASTA files...")
    lnc_records = load_fasta(args.lnc_fasta)
    pc_records  = load_fasta(args.pc_fasta)
    print(f"  lncRNA sequences:  {len(lnc_records):,}")
    print(f"  mRNA  sequences:  {len(pc_records):,}")
    print(f"  Total:             {len(lnc_records) + len(pc_records):,}")

    # ------------------------------------------------------------------
    # Split each class independently (stratified by class)
    # ------------------------------------------------------------------
    print(f"\nSplitting with test_size={args.test_size:.0%}, seed={args.seed}...")

    lnc_trainval, lnc_test = stratified_split(lnc_records, args.test_size, seed=args.seed)
    pc_trainval,  pc_test  = stratified_split(pc_records,  args.test_size, seed=args.seed)

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    print("\nWriting train/val sets (input to CV scripts):")
    write_fasta(lnc_trainval, output_dir / "lnc_trainval.fa")
    write_fasta(pc_trainval,  output_dir / "pc_trainval.fa")

    print("\nWriting independent test sets:")
    write_fasta(lnc_test, output_dir / "lnc_test.fa")
    write_fasta(pc_test,  output_dir / "pc_test.fa")

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    manifest = {
        "seed": args.seed,
        "test_size": args.test_size,
        "input": {
            "lnc_fasta": str(args.lnc_fasta),
            "pc_fasta":  str(args.pc_fasta),
        },
        "trainval": {
            "lnc": len(lnc_trainval),
            "pc":  len(pc_trainval),
            "total": len(lnc_trainval) + len(pc_trainval),
            "lnc_file": "lnc_trainval.fa",
            "pc_file":  "pc_trainval.fa",
        },
        "test": {
            "lnc": len(lnc_test),
            "pc":  len(pc_test),
            "total": len(lnc_test) + len(pc_test),
            "lnc_file": "lnc_test.fa",
            "pc_file":  "pc_test.fa",
        },
        "class_ratio": {
            "trainval_lnc_frac": round(len(lnc_trainval) / (len(lnc_trainval) + len(pc_trainval)), 4),
            "test_lnc_frac":     round(len(lnc_test)    / (len(lnc_test)     + len(pc_test)),     4),
        }
    }

    manifest_path = output_dir / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Split summary")
    print("=" * 60)
    print(f"  Train/val — lnc: {len(lnc_trainval):>6,}  pc: {len(pc_trainval):>6,}"
          f"  total: {len(lnc_trainval)+len(pc_trainval):>7,}"
          f"  (lnc frac: {manifest['class_ratio']['trainval_lnc_frac']:.3f})")
    print(f"  Test      — lnc: {len(lnc_test):>6,}  pc: {len(pc_test):>6,}"
          f"  total: {len(lnc_test)+len(pc_test):>7,}"
          f"  (lnc frac: {manifest['class_ratio']['test_lnc_frac']:.3f})")
    print(f"\n  Manifest saved to: {manifest_path}")
    print("=" * 60)
    print("\nNext steps:")
    print(f"  1. Point your CV configs to:  {output_dir}/lnc_trainval.fa")
    print(f"                                {output_dir}/pc_trainval.fa")
    print(f"  2. Keep test files sealed:    {output_dir}/lnc_test.fa")
    print(f"                                {output_dir}/pc_test.fa")
    print(f"  3. Evaluate on test only AFTER all CV training is complete.")


if __name__ == "__main__":
    main()