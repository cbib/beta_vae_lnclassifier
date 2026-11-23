#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_nonb2.py

Merges ScanFold2 and rG4detector CSVs into a single NonB2 feature file
ready for prep_features.py.

Steps
-----
1. Load both CSVs
2. Normalise transcript_id: pipe-split → strip version suffix
3. Drop excluded columns from ScanFold2:
     length, simple_ID — metadata
     n_paired, n_competition, n_z_lt_minus1, n_z_lt_minus2 — absolute counts
     n_*_bin{1..10} — length-dependent absolute bin counts
4. Inner join on normalised transcript_id
5. Write merged CSV with transcript_id as first column
6. Write <output_csv stem>_report.json (and, with --fasta, _wrong_length_ids.txt): per input file, duplicate rows (and
   the IDs whose duplicates disagree), and, with --fasta, rows whose reported
   length differs from the FASTA sequence length (a keep-first rule downstream
   keeps the wrong row for the IDs counted in n_ids_first_row_wrong_length)

The output is used as --nonb2_csv in prep_features.py.

Usage
-----
python data/merge_nonb2.py \\
    --scanfold_csv  data/nonb2_te_raw/scanfold_features_v49.csv \\
    --rg4_csv       data/nonb2_te_raw/rg4detector_features_v49.csv \\
    --output_csv    data/nonb2_te_raw/nonb2_features_v49.csv \\
    [--fasta        lncRNA.fa mRNA.fa]   # enables the wrong-length check
"""

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np


# Columns to drop from ScanFold2 (length-dependent or metadata)
_SF_DROP_EXACT = {"length", "simple_ID"}
_SF_DROP_ABS_COUNT = {"n_paired", "n_competition", "n_z_lt_minus1", "n_z_lt_minus2"}
_SF_BIN_PATTERN = re.compile(
    r"^n_(paired|competition|z_lt_minus1|z_lt_minus2)_bin\d+$"
)

_KINDS = ("paired", "competition", "z_lt_minus1", "z_lt_minus2")


def fill_prop_from_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Rows from the later ScanFold2 run carry n_<kind>_bin1..10 counts and NaN prop_<kind>_bin1..10.
    prop_<kind>_binK = n_<kind>_binK / sum_K n_<kind>_binK (share of the transcript's <kind> windows in bin K;
    0 when it has none). Must run BEFORE the n_* columns are dropped."""
    for k in _KINDS:
        pc = [f"prop_{k}_bin{b}" for b in range(1, 11)]
        nc = [f"n_{k}_bin{b}" for b in range(1, 11)]
        if not (set(pc) <= set(df.columns) and set(nc) <= set(df.columns)):
            print(f"  [{k}] prop or n bin columns missing; not filled")
            continue
        n = df[nc].astype(float)
        tot = n.sum(axis=1, min_count=1)
        derived = n.div(tot.where(tot > 0), axis=0).fillna(0.0).where(n.notna().all(axis=1))
        derived.columns = pc
        has_p = df[pc].notna().all(axis=1)
        has_n = derived.notna().all(axis=1)
        # agreement of the two runs on transcripts that have both formats
        a = df.loc[has_p, ["transcript_id"] + pc].groupby("transcript_id").first()
        b = derived.loc[has_n].assign(transcript_id=df.loc[has_n, "transcript_id"].values).groupby("transcript_id").first()
        common = a.index.intersection(b.index)
        agree = ""
        if len(common):
            x, y = a.loc[common].values.ravel(), b.loc[common].values.ravel()
            agree = (f"; agreement between runs on {len(common):,} shared ids: "
                     f"r = {np.corrcoef(x, y)[0, 1]:.3f}, mean |diff| = {np.abs(x - y).mean():.4f}")
        need = ~has_p & has_n
        df.loc[need, pc] = derived.loc[need].values
        left = int(df[pc].isna().all(axis=1).sum())
        print(f"  [{k}] filled {int(need.sum()):,} rows from counts; rows still without bins: {left:,}{agree}")
    return df


# Columns to drop from rG4 (metadata)
_RG4_DROP_EXACT = {"transcript_length"}


def normalise_id(series: pd.Series) -> pd.Series:
    """Strip pipe-suffix then version suffix from a transcript ID series."""
    return (series.astype(str)
            .str.split("|").str[0]
            .str.split(".").str[0])


def _print_report(name: str, r: dict) -> None:
    print(f"  {name}: {r['n_duplicate_rows']:,} duplicate rows over {r['n_unique_ids']:,} ids; "
          f"{r['n_ids_with_conflicting_features']:,} ids with differing features, "
          f"{r['n_ids_with_conflicting_length']:,} with differing length")
    lc = r.get("length_check")
    if lc:
        print(f"  {name}: {lc['n_rows_wrong_length']:,} of {lc['n_rows_checked']:,} rows have a length "
              f"different from the FASTA; {lc['n_ids_first_row_wrong_length']:,} ids would keep a "
              f"wrong-length first row")


def read_fasta_lengths(paths) -> dict:
    """{normalised transcript id: sequence length} over the given FASTA files."""
    out, tid, n = {}, None, 0
    for p in paths:
        with open(p) as fh:
            for line in fh:
                if line.startswith(">"):
                    if tid is not None:
                        out[tid] = n
                    tid, n = line[1:].split()[0].split("|")[0].split(".")[0], 0
                else:
                    n += len(line.strip())
        if tid is not None:
            out[tid] = n
        tid, n = None, 0
    return out


def table_report(df: pd.DataFrame, fasta_len: dict | None = None, tol: float = 1e-6) -> dict:
    """Duplicate and length report for a raw table (transcript_id already normalised,
    before any column is dropped). Rows keep file order: downstream cleaning keeps
    the FIRST row of a duplicated id."""
    ids = df["transcript_id"]
    lcol = next((c for c in ("length", "transcript_length") if c in df.columns), None)
    dup = ids.duplicated(keep=False)
    conflict_ids = conflict_len_ids = 0
    if dup.any():
        num = df.loc[dup].drop(columns=[c for c in (lcol,) if c], errors="ignore").select_dtypes("number")
        g = num.groupby(ids[dup])
        conflict_ids = int(((g.max() - g.min()).abs() > tol * (1 + num.abs().groupby(ids[dup]).max())
                            ).any(axis=1).sum())
        if lcol:
            gl = df.loc[dup, lcol].groupby(ids[dup])
            conflict_len_ids = int((gl.nunique(dropna=True) > 1).sum())
    rep = dict(n_rows=int(len(df)), n_unique_ids=int(ids.nunique()),
               n_duplicate_rows=int(ids.duplicated().sum()),
               n_ids_with_conflicting_features=conflict_ids,
               n_ids_with_conflicting_length=conflict_len_ids,
               length_column=lcol, length_check=None)
    if lcol and fasta_len:
        fl = ids.map(fasta_len)
        known = fl.notna() & df[lcol].notna()
        diff = (df[lcol] - fl)[known]
        wk, idk = (diff != 0), ids[known]
        anyw, allw = wk.groupby(idk).any(), wk.groupby(idk).all()
        bad = diff[wk]
        rep["length_check"] = dict(
            n_rows_checked=int(known.sum()),
            n_rows_not_in_fasta=int(fl.isna().sum()),
            n_rows_wrong_length=int(wk.sum()),
            n_rows_longer_than_fasta=int((diff > 0).sum()),
            n_rows_shorter_than_fasta=int((diff < 0).sum()),
            n_ids_wrong_length=int(anyw.sum()),
            n_ids_only_wrong_rows=int(allw.sum()),
            n_ids_wrong_and_correct_rows=int((anyw & ~allw).sum()),
            n_ids_first_row_wrong_length=int(wk.groupby(idk).first().sum()),
            first_row_wrong_ids=sorted(wk.groupby(idk).first().loc[lambda x: x].index),
            examples=[dict(id=ids[i], length=int(df.at[i, lcol]), fasta_length=int(fl[i]))
                      for i in bad.index[:10]],
        )
    return rep


def load_scanfold(path: str | Path, report: dict | None = None,
                  fasta_len: dict | None = None) -> pd.DataFrame:
    print(f"Loading ScanFold2: {path}")
    df = pd.read_csv(path)
    print(f"  Raw shape: {df.shape}")

    # Normalise transcript_id
    df["transcript_id"] = normalise_id(df["transcript_id"])

    # Fill prop_*_bin* from n_*_bin* where the run only wrote counts (before n_* is dropped)
    df = fill_prop_from_counts(df)
    if report is not None:
        report.update(table_report(df, fasta_len))
        _print_report("ScanFold2", report)

    # Drop excluded columns
    drop_cols = set()
    for col in df.columns:
        if col == "transcript_id":
            continue
        if col in _SF_DROP_EXACT:
            drop_cols.add(col)
        elif col in _SF_DROP_ABS_COUNT:
            drop_cols.add(col)
        elif _SF_BIN_PATTERN.match(col):
            drop_cols.add(col)

    if drop_cols:
        print(f"  Dropping {len(drop_cols)} excluded columns "
              f"(metadata + length-dependent): {sorted(drop_cols)[:5]}{'...' if len(drop_cols)>5 else ''}")
        df = df.drop(columns=list(drop_cols))

    # Drop fully-empty columns (trailing commas in ScanFold2 output)
    empty_cols = [c for c in df.columns if df[c].isna().all() and c != "transcript_id"]
    if empty_cols:
        print(f"  Dropping {len(empty_cols)} fully-empty columns")
        df = df.drop(columns=empty_cols)

    df = df.set_index("transcript_id")
    print(f"  Clean shape: {df.shape}  ({len(df):,} transcripts, "
          f"{len(df.columns)} features)")
    return df


def load_rg4(path: str | Path, report: dict | None = None,
             fasta_len: dict | None = None) -> pd.DataFrame:
    print(f"Loading rG4detector: {path}")
    df = pd.read_csv(path)
    print(f"  Raw shape: {df.shape}")

    # Normalise transcript_id
    df["transcript_id"] = normalise_id(df["transcript_id"])
    if report is not None:
        report.update(table_report(df, fasta_len))
        _print_report("rG4detector", report)

    # Drop metadata columns
    drop_cols = [c for c in df.columns
                 if c in _RG4_DROP_EXACT and c != "transcript_id"]
    if drop_cols:
        print(f"  Dropping {len(drop_cols)} metadata columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    df = df.set_index("transcript_id")
    print(f"  Clean shape: {df.shape}  ({len(df):,} transcripts, "
          f"{len(df.columns)} features)")
    return df


def merge_nonb2(
    scanfold_csv: str | Path,
    rg4_csv:      str | Path,
    output_csv:   str | Path,
    fasta_paths=None,
) -> pd.DataFrame:
    fasta_len = read_fasta_lengths(fasta_paths) if fasta_paths else None
    if fasta_len is not None:
        print(f"FASTA lengths: {len(fasta_len):,} sequences from {len(fasta_paths)} file(s)")
    rep_sf, rep_rg4 = {}, {}
    sf  = load_scanfold(scanfold_csv, rep_sf, fasta_len)
    rg4 = load_rg4(rg4_csv, rep_rg4, fasta_len)

    # Coverage check
    sf_ids  = set(sf.index)
    rg4_ids = set(rg4.index)
    common  = sf_ids & rg4_ids
    only_sf  = sf_ids  - rg4_ids
    only_rg4 = rg4_ids - sf_ids

    print(f"\nCoverage:")
    print(f"  ScanFold2 transcripts : {len(sf_ids):,}")
    print(f"  rG4detector transcripts: {len(rg4_ids):,}")
    print(f"  Common (inner join)   : {len(common):,}")
    if only_sf:
        print(f"    Only in ScanFold2  : {len(only_sf):,}")
    if only_rg4:
        print(f"    Only in rG4        : {len(only_rg4):,}")
    if len(only_sf) > 0 or len(only_rg4) > 0:
        print("  Using inner join — transcripts not in both files will be excluded")
        print("  If coverage differs substantially, check Diego's export notebooks")

    # Inner join
    merged = sf.join(rg4, how="inner")
    print(f"\nMerged shape: {merged.shape}  "
          f"({len(merged):,} transcripts, {len(merged.columns)} features)")

    # Sanity: check for duplicate column names
    dupes = merged.columns[merged.columns.duplicated()].tolist()
    if dupes:
        print(f"    Duplicate columns after merge: {dupes}")
        print("  Removing duplicates (keeping first occurrence)")
        merged = merged.loc[:, ~merged.columns.duplicated()]

    # Validate column order matches registry expectation
    try:
        from data.feature_registry import REGISTRY
        expected = [m.name for m in REGISTRY.nonb2_features]
        actual   = list(merged.columns)
        if expected == actual:
            print(f"   Column order matches registry ({len(expected)} features)")
        else:
            missing_from_csv = set(expected) - set(actual)
            extra_in_csv     = set(actual) - set(expected)
            order_only       = not missing_from_csv and not extra_in_csv
            if missing_from_csv:
                print(f"    In registry but not in merged CSV: "
                      f"{sorted(missing_from_csv)[:5]}")
            if extra_in_csv:
                print(f"    In merged CSV but not in registry: "
                      f"{sorted(extra_in_csv)[:5]}")
            if order_only:
                print("    Column ORDER differs from registry — "
                      "reordering to match registry")
                merged = merged[expected]
    except ImportError:
        print("  Registry not available — skipping column order validation")

    # Write output with transcript_id as first column
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv)   # index=True keeps transcript_id as index column
    used_wrong = sorted(
        {i for r in (rep_sf, rep_rg4) for i in ((r.get("length_check") or {}).get("first_row_wrong_ids") or [])}
        & (sf_ids | rg4_ids))
    report = dict(
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        inputs=dict(scanfold_csv=str(scanfold_csv), rg4_csv=str(rg4_csv),
                    fasta=[str(p) for p in (fasta_paths or [])]),
        scanfold=rep_sf, rg4detector=rep_rg4,
        join=dict(n_scanfold_ids=len(sf_ids), n_rg4_ids=len(rg4_ids), n_common=len(common),
                  n_only_scanfold=len(only_sf), n_only_rg4=len(only_rg4),
                  n_merged_rows=int(len(merged)), n_features=int(len(merged.columns)),
                  n_ids_used_row_wrong_length=len(used_wrong) if fasta_paths else None),
    )
    if fasta_paths:
        # ids whose row kept by the keep-first cleaning has a length different from the FASTA,
        # in either input file (the union is what a model trained on the cleaned table saw)
        output_csv.with_name(output_csv.stem + "_wrong_length_ids.txt").write_text(
            "".join(f"{i}\n" for i in used_wrong))
    rep_path = output_csv.with_name(output_csv.stem + "_report.json")
    rep_path.write_text(json.dumps(report, indent=2))
    print(f"\nWritten: {output_csv}")
    print(f"Report : {rep_path}")
    print(f"  Transcripts: {len(merged):,}")
    print(f"  Features   : {len(merged.columns)}")
    print(f"  First 5 features: {list(merged.columns[:5])}")
    print(f"  Last 5 features : {list(merged.columns[-5:])}")

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Merge ScanFold2 and rG4detector CSVs into NonB2 feature file"
    )
    parser.add_argument("--scanfold_csv", required=True,
                        help="ScanFold2 features CSV")
    parser.add_argument("--rg4_csv",      required=True,
                        help="rG4detector features CSV")
    parser.add_argument("--output_csv",   required=True,
                        help="Output merged NonB2 CSV")
    parser.add_argument("--fasta", nargs="+", default=None,
                        help="FASTA file(s) of the same transcripts; enables the "
                             "wrong-length check (reported length vs sequence length)")
    args = parser.parse_args()

    merge_nonb2(args.scanfold_csv, args.rg4_csv, args.output_csv, args.fasta)
    print("\nDone.")


if __name__ == "__main__":
    main()