#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis/benchmark/unseen_subset.py

Which test transcripts could a GENCODE-trained comparator already have seen?

For every test transcript, find the first GENCODE release (scanned range
[--min_release, --test_release - 1]) whose annotation contains it, then flag
the transcripts that appear only AFTER a given release (the cutoff). The
resulting ID lists feed compute_benchmark_ci.py --id_list, so no model is
retrained: existing predictions are simply re-scored on the unseen subset.
The seen list (seen_ids_L2_by_v<c>.txt: determined transcripts that are not
unseen at L2) is the contrast subset; run_unseen_benchmark.sh scores both.

Three exposure levels, from loosest to strictest "unseen":
  L1  versioned transcript ID absent from every release <= cutoff
  L2  unversioned transcript ID absent   (primary: a re-versioned isoform
      is treated as seen)
  L3  L2 and the unversioned gene ID absent (no isoform of the gene seen)

By default only the cutoff releases are scanned (a re-added retired ID is
the only thing a full scan would add); --all_releases scans every release for
the first-seen age distribution. Release indexes are streamed from the GENCODE annotation GTFs (only
transcript lines are kept) and cached as small TSVs in --cache_dir; the
caches can be deposited so the analysis reruns offline. Use --local_gtf_dir
for GTFs already on disk (gencode.v{N}.annotation.gtf.gz).

Not covered: RefSeq-derived training data (mRNA side of LncDC, CPAT, CPC2,
lncRNA-BERT) cannot be matched by ENST ID; the ID-level result is a proxy
for sequence overlap, not a sequence-level check.

Usage
-----
python analysis/benchmark/unseen_subset.py \\
    --test_lnc_fasta data/split_gencode_49/lnc_test.fa \\
    --test_pc_fasta  data/split_gencode_49/pc_test.fa \\
    --test_release 49 --cutoffs 34 43 46 \\
    --cache_dir gencode_release_index \\
    --output_dir gencode_v49_experiments/unseen_subset
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import io
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

URL_TEMPLATE = ("https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/"
                "release_{r}/gencode.v{r}.annotation.gtf.gz")
GTF_NAME = "gencode.v{r}.annotation.gtf.gz"

_ENST = re.compile(r"ENST\d+(?:\.\d+)?")
_ENSG = re.compile(r"ENSG\d+(?:\.\d+)?")
_TX = re.compile(r'transcript_id "([^"]+)"')
_GN = re.compile(r'gene_id "([^"]+)"')


def unver(s: str) -> str:
    return s.split(".")[0]


def clean(s: str) -> str:
    """Drop the _PAR_Y suffix GENCODE adds to chrY PAR copies."""
    return s.split("_")[0]


# ---------------------------------------------------------------------------
# Test set
# ---------------------------------------------------------------------------

def read_test_table(lnc_fa: str, pc_fa: str) -> pd.DataFrame:
    rows = []
    for path, cls in ((lnc_fa, "lnc"), (pc_fa, "pc")):
        with open(path) as fh:
            for line in fh:
                if not line.startswith(">"):
                    continue
                t, g = _ENST.search(line), _ENSG.search(line)
                if not t:
                    sys.exit(f"ERROR: no ENST id in header of {path}: {line[:80]}")
                rows.append((t.group(0), g.group(0) if g else "", cls))
    df = pd.DataFrame(rows, columns=["transcript_id", "gene_id", "class"])
    dup = int(df.duplicated("transcript_id").sum())
    if dup:
        print(f"  WARNING: {dup} duplicate transcript IDs in the test FASTAs "
              f"(kept once)")
    df = df.drop_duplicates("transcript_id").reset_index(drop=True)
    df["id"] = df["transcript_id"].map(unver)
    df["gene"] = df["gene_id"].map(unver)
    return df


# ---------------------------------------------------------------------------
# Release index: set of (transcript_id, gene_id), versioned, from one GTF
# ---------------------------------------------------------------------------

def read_gtf_ids(handle) -> set[tuple[str, str]]:
    out = set()
    for line in handle:
        if line[0] == "#":
            continue
        c = line.split("\t", 8)
        if len(c) < 9 or c[2] != "transcript":
            continue
        t, g = _TX.search(c[8]), _GN.search(c[8])
        if t and g:
            out.add((clean(t.group(1)), clean(g.group(1))))
    return out


def open_gtf(r: int, local_dir: Path | None, url_tpl: str):
    if local_dir is not None:
        p = local_dir / GTF_NAME.format(r=r)
        if p.is_file():
            return gzip.open(p, "rt", encoding="utf-8"), str(p)
    url = url_tpl.format(r=r)
    resp = urllib.request.urlopen(url, timeout=120)
    return io.TextIOWrapper(gzip.GzipFile(fileobj=resp), encoding="utf-8"), url


def release_index(r: int, cache_dir: Path, local_dir: Path | None,
                  url_tpl: str) -> tuple[set, str] | None:
    """Returns (ids, source) or None when the release cannot be obtained."""
    cache = cache_dir / f"gencode_v{r}.ids.tsv.gz"
    if cache.is_file():
        df = pd.read_csv(cache, sep="\t", dtype=str)
        return set(zip(df["transcript_id"], df["gene_id"])), f"cache:{cache.name}"
    try:
        handle, source = open_gtf(r, local_dir, url_tpl)
        with handle:
            ids = read_gtf_ids(handle)
    except (urllib.error.URLError, OSError, EOFError) as e:
        print(f"  v{r}: unavailable ({e})")
        return None
    if not ids:
        print(f"  v{r}: no transcript lines parsed from {source}")
        return None
    tmp = cache.with_suffix(".tmp.gz")
    pd.DataFrame(sorted(ids), columns=["transcript_id", "gene_id"]).to_csv(
        tmp, sep="\t", index=False, compression="gzip")
    tmp.replace(cache)
    return ids, source


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def scan(test: pd.DataFrame, releases: list[int], test_release: int,
         cache_dir: Path, local_dir: Path | None, url_tpl: str,
         allow_missing: bool):
    want_tv = set(test["transcript_id"])
    want_tu = set(test["id"])
    want_gu = set(test["gene"]) - {""}
    first_tv, first_tu, first_gu = {}, {}, {}
    in_test_release = set()
    sources, missing = {}, []

    for r in sorted(set(releases) | {test_release}):
        got = release_index(r, cache_dir, local_dir, url_tpl)
        if got is None:
            if r == test_release:
                print(f"  WARNING: test release v{r} unavailable; "
                      f"skipping the presence sanity check")
            else:
                missing.append(r)
            continue
        ids, src = got
        sources[r] = src
        tv = {t for t, _ in ids}
        if r == test_release:
            in_test_release = want_tv & tv
            print(f"  v{r} (test release): {len(in_test_release)}/{len(want_tv)} "
                  f"test transcript IDs present")
            continue
        tu = {unver(t) for t in tv}
        gu = {unver(g) for _, g in ids}
        for t in want_tv & tv:
            first_tv.setdefault(t, r)
        for t in want_tu & tu:
            first_tu.setdefault(t, r)
        for g in want_gu & gu:
            first_gu.setdefault(g, r)
        print(f"  v{r}: {len(ids)} transcripts ({src})")
    if missing and not allow_missing:
        sys.exit(f"ERROR: releases unavailable: {missing}. A gap weakens every "
                 f"'unseen' claim; supply the GTFs (--local_gtf_dir) or pass "
                 f"--allow_missing.")
    return first_tv, first_tu, first_gu, in_test_release, sources, missing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test_lnc_fasta", required=True)
    p.add_argument("--test_pc_fasta", required=True)
    p.add_argument("--test_release", type=int, required=True,
                   help="GENCODE release the test set was built from (e.g. 49)")
    p.add_argument("--cutoffs", type=int, nargs="+", default=[34, 43, 46],
                   help="training-data releases to test against: a transcript "
                        "is unseen at cutoff c if first seen after c")
    p.add_argument("--releases", type=int, nargs="+", default=None,
                   help="releases to scan (default: the cutoffs only). A "
                        "transcript is 'seen at c' if present in any scanned "
                        "release <= c")
    p.add_argument("--all_releases", action="store_true",
                   help="scan every release from --min_release to test_release-1 "
                        "(adds the first-seen age distribution)")
    p.add_argument("--min_release", type=int, default=20,
                   help="oldest release for --all_releases (GRCh38 starts at v20)")
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--local_gtf_dir", default=None)
    p.add_argument("--url_template", default=URL_TEMPLATE)
    p.add_argument("--allow_missing", action="store_true")
    p.add_argument("--min_class_n", type=int, default=200,
                   help="warn when an unseen subset has fewer transcripts per class")
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    if max(args.cutoffs) >= args.test_release:
        sys.exit("ERROR: every cutoff must be < --test_release")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_dir = Path(args.local_gtf_dir) if args.local_gtf_dir else None

    test = read_test_table(args.test_lnc_fasta, args.test_pc_fasta)
    print(f"Test set: {len(test)} transcripts "
          f"({(test['class'] == 'lnc').sum()} lnc, {(test['class'] == 'pc').sum()} pc), "
          f"{test['gene'].nunique()} genes")

    if args.all_releases:
        releases = list(range(args.min_release, args.test_release))
    else:
        releases = sorted(set(args.releases or args.cutoffs) | set(args.cutoffs))
    if max(releases) >= args.test_release:
        sys.exit("ERROR: scanned releases must be < --test_release")
    print(f"Scanning {['v%d' % r for r in releases]} (+ test release v{args.test_release})")
    first_tv, first_tu, first_gu, in_test, sources, missing = scan(
        test, releases, args.test_release, cache_dir, local_dir,
        args.url_template, args.allow_missing)

    test["first_seen_tx_versioned"] = test["transcript_id"].map(first_tv)
    test["first_seen_tx"] = test["id"].map(first_tu)
    test["first_seen_gene"] = test["gene"].map(first_gu)
    test["in_test_release"] = test["transcript_id"].isin(in_test)
    # IDs missing from the test release's main GTF (e.g. patch/haplotype contigs
    # live in the chr_patch_hapl_scaff GTF) have an unreliable history here:
    # exclude them from the unseen sets instead of calling them unseen.
    determined = test["in_test_release"] if in_test else pd.Series(True, index=test.index)
    test["exposure_determined"] = determined
    if not determined.all():
        print(f"  NOTE: {int((~determined).sum())} test transcripts are absent from "
              f"v{args.test_release}'s main annotation; excluded from the unseen "
              f"sets (see exposure_table.csv, exposure_determined == False)")

    def seen(col, c):
        return test[col].le(c).fillna(False)

    summary = []
    for c in sorted(args.cutoffs):
        test[f"unseen_L1_v{c}"] = ~seen("first_seen_tx_versioned", c) & determined
        test[f"unseen_L2_v{c}"] = ~seen("first_seen_tx", c) & determined
        test[f"unseen_L3_v{c}"] = test[f"unseen_L2_v{c}"] & ~seen("first_seen_gene", c)
        for level in ("L1", "L2", "L3"):
            flag = test[f"unseen_{level}_v{c}"]
            for cls in ("lnc", "pc", "all"):
                m = (test["class"].eq(cls) if cls != "all"
                     else pd.Series(True, index=test.index)) & determined
                n, k = int(m.sum()), int((m & flag).sum())
                summary.append(dict(cutoff=c, level=level, cls=cls, n_test=n,
                                    n_unseen=k, pct_unseen=100 * k / n if n else float("nan"),
                                    pct_seen=100 * (1 - k / n) if n else float("nan")))
        for level in ("L2", "L3"):
            ids = test.loc[test[f"unseen_{level}_v{c}"], "id"]
            (out / f"unseen_ids_{level}_after_v{c}.txt").write_text(
                "\n".join(sorted(ids)) + ("\n" if len(ids) else ""))
            by = test.loc[test[f"unseen_{level}_v{c}"], "class"].value_counts()
            for cls in ("lnc", "pc"):
                if by.get(cls, 0) < args.min_class_n:
                    print(f"  WARNING: {level} after v{c}: only {by.get(cls, 0)} "
                          f"{cls} transcripts (< {args.min_class_n}); CIs will be wide")
        # the complement at the primary level: transcripts whose history is
        # determined and that appear in a release <= cutoff (contrast subset)
        seen_ids = test.loc[determined & ~test[f"unseen_L2_v{c}"], "id"]
        (out / f"seen_ids_L2_by_v{c}.txt").write_text(
            "\n".join(sorted(seen_ids)) + ("\n" if len(seen_ids) else ""))

    summ = pd.DataFrame(summary)
    test.to_csv(out / "exposure_table.csv", index=False)
    summ.to_csv(out / "exposure_summary.csv", index=False)
    manifest = dict(
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        test_release=args.test_release, cutoffs=sorted(args.cutoffs),
        scanned=sorted(r for r in sources if r != args.test_release),
        missing_releases=missing, sources=sources,
        cache_sha256_16={f"v{r}": sha16(cache_dir / f"gencode_v{r}.ids.tsv.gz")
                         for r in sources if (cache_dir / f"gencode_v{r}.ids.tsv.gz").is_file()},
        n_test=int(len(test)),
        test_in_test_release=int(test["in_test_release"].sum()) if in_test else None,
        levels=dict(L1="versioned transcript ID", L2="unversioned transcript ID (primary)",
                    L3="L2 and unversioned gene ID"),
        url_template=args.url_template)
    (out / "unseen_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\nShare of test transcripts NOT seen in any release <= cutoff (%)")
    print(summ[summ.cls != "all"].pivot_table(
        index=["level", "cutoff"], columns="cls", values="pct_unseen").round(1).to_string())
    print(f"\nWrote {out}/ (exposure_table.csv, exposure_summary.csv, "
          f"unseen_ids_*.txt, seen_ids_L2_by_v*.txt, unseen_manifest.json)")


if __name__ == "__main__":
    main()