#!/usr/bin/env python3
"""Offline test for unseen_subset.py: fake GTF releases, known exposure."""
import gzip
import subprocess
import sys
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "benchmark" / "unseen_subset.py"

# (transcript_id, gene_id) -> releases in which it is annotated
SPEC = {
    ("ENST00000000001.1", "ENSG00000000001.1"): range(20, 50),   # old
    ("ENST00000000002.1", "ENSG00000000002.1"): range(40, 50),   # new in 40, old gene
    ("ENST00000000003.1", "ENSG00000000001.1"): range(47, 50),   # new isoform of old gene
    ("ENST00000000004.1", "ENSG00000000004.1"): range(30, 47),   # old version
    ("ENST00000000004.2", "ENSG00000000004.1"): range(47, 50),   # re-versioned in 47
    ("ENST00000000005.1", "ENSG00000000005.1"): range(48, 50),   # all new in 48
    ("ENST00000000006.1", "ENSG00000000006.1"): range(49, 50),   # new in test release
    ("ENST00000000007.1", "ENSG00000000007.1"): range(20, 50),   # old pc
    ("ENST00000000099.1", "ENSG00000000099.1"): range(20, 30),   # retired, not in test
    ("ENST00000000008.1", "ENSG00000000008.1"): range(0, 0),     # in FASTA only (patch contig)
}
LNC = ["ENST00000000001.1", "ENST00000000002.1", "ENST00000000003.1",
       "ENST00000000004.2", "ENST00000000005.1", "ENST00000000008.1"]
PC = ["ENST00000000006.1", "ENST00000000007.1"]
GENE = {t: g for (t, g) in SPEC}


def write_gtf(path, r):
    with gzip.open(path, "wt") as fh:
        fh.write("##header\n")
        for (t, g), rels in SPEC.items():
            if r in rels:
                fh.write(f'chr1\tHAVANA\tgene\t1\t9\t.\t+\t.\tgene_id "{g}";\n')
                fh.write(f'chr1\tHAVANA\ttranscript\t1\t9\t.\t+\t.\t'
                         f'gene_id "{g}"; transcript_id "{t}"; gene_type "x";\n')
                fh.write(f'chr1\tHAVANA\texon\t1\t9\t.\t+\t.\t'
                         f'gene_id "{g}"; transcript_id "{t}";\n')


def write_fa(path, ids):
    with open(path, "w") as fh:
        for t in ids:
            fh.write(f">{t}|{GENE[t]}|OTT|-|{t.split('.')[0]}|x|100|\nACGT\n")


def test_unseen_subset(tmp_path):
    d = tmp_path
    gtfs = d / "gtf"
    gtfs.mkdir()
    for r in range(20, 50):
        write_gtf(gtfs / f"gencode.v{r}.annotation.gtf.gz", r)
    write_fa(d / "lnc.fa", LNC)
    write_fa(d / "pc.fa", PC)
    cmd = [sys.executable, str(SCRIPT),
           "--test_lnc_fasta", str(d / "lnc.fa"), "--test_pc_fasta", str(d / "pc.fa"),
           "--test_release", "49", "--cutoffs", "34", "43", "46",
           "--cache_dir", str(d / "cache"), "--local_gtf_dir", str(gtfs),
           "--all_releases", "--min_class_n", "0", "--output_dir", str(d / "out")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    t = pd.read_csv(d / "out/exposure_table.csv").set_index("transcript_id")

    def unseen(level, c):
        return set(t.index[t[f"unseen_{level}_v{c}"]])

    # cutoff 46
    assert unseen("L2", 46) == {"ENST00000000003.1", "ENST00000000005.1", "ENST00000000006.1"}
    assert unseen("L1", 46) == {"ENST00000000003.1", "ENST00000000004.2",
                                "ENST00000000005.1", "ENST00000000006.1"}
    assert unseen("L3", 46) == {"ENST00000000005.1", "ENST00000000006.1"}
    # cutoff 34: transcript 2 (first in v40) and 4.x (first v30, seen) differ
    assert "ENST00000000002.1" in unseen("L2", 34)
    assert "ENST00000000004.2" not in unseen("L2", 34)
    # cutoff 43: transcript 2 seen
    assert "ENST00000000002.1" not in unseen("L2", 43)
    assert t.loc["ENST00000000002.1", "first_seen_tx"] == 40
    assert t.loc["ENST00000000004.2", "first_seen_tx"] == 30
    assert t.loc["ENST00000000004.2", "first_seen_tx_versioned"] == 47
    assert not t.loc["ENST00000000008.1", "in_test_release"]
    assert not t.loc["ENST00000000008.1", "exposure_determined"]
    assert all("ENST00000000008.1" not in unseen(l, c) for l in ("L1", "L2", "L3") for c in (34, 43, 46))
    assert t.drop("ENST00000000008.1").in_test_release.all()
    ids = (d / "out/unseen_ids_L2_after_v46.txt").read_text().split()
    assert ids == ["ENST00000000003", "ENST00000000005", "ENST00000000006"], ids
    # the seen list is the complement among the transcripts with a known history
    seen46 = (d / "out/seen_ids_L2_by_v46.txt").read_text().split()
    assert seen46 == ["ENST00000000001", "ENST00000000002", "ENST00000000004",
                      "ENST00000000007"], seen46
    for c in (34, 43, 46):
        un = set((d / f"out/unseen_ids_L2_after_v{c}.txt").read_text().split())
        se = set((d / f"out/seen_ids_L2_by_v{c}.txt").read_text().split())
        known = set(t.index[t.exposure_determined].map(lambda x: x.split(".")[0]))
        assert un.isdisjoint(se) and un | se == known, c

    # cutoff-only scan gives the same unseen sets as the full scan
    cmdc = [c for c in cmd]
    cmdc[cmdc.index("--output_dir") + 1] = str(d / "outc")
    cmdc[cmdc.index("--cache_dir") + 1] = str(d / "cachec")
    rc = subprocess.run(cmdc, capture_output=True, text=True)
    assert rc.returncode == 0, rc.stdout + rc.stderr
    tc = pd.read_csv(d / "outc/exposure_table.csv").set_index("transcript_id")
    for lvl in ("L1", "L2", "L3"):
        for c in (34, 43, 46):
            assert (tc[f"unseen_{lvl}_v{c}"] == t[f"unseen_{lvl}_v{c}"]).all(), (lvl, c)

    # cache-only rerun (no GTFs, no network) reproduces the table
    cmd2 = [c for c in cmd]
    cmd2[cmd2.index("--local_gtf_dir") + 1] = str(d / "nonexistent")
    cmd2[cmd2.index("--output_dir") + 1] = str(d / "out2")
    r2 = subprocess.run(cmd2, capture_output=True, text=True)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    t2 = pd.read_csv(d / "out2/exposure_table.csv")
    pd.testing.assert_frame_equal(t.reset_index(), t2)

    # a missing release is fatal unless allowed
    (d / "cache/gencode_v40.ids.tsv.gz").unlink()
    (gtfs / "gencode.v40.annotation.gtf.gz").unlink()
    cmd3 = list(cmd2)
    cmd3[cmd3.index("--output_dir") + 1] = str(d / "out3")
    r3 = subprocess.run(cmd3 + ["--url_template", "http://127.0.0.1:9/{r}"],
                        capture_output=True, text=True)
    assert r3.returncode != 0 and "unavailable" in (r3.stdout + r3.stderr)