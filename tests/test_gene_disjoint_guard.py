"""Gene-disjoint guards: CV folds, the test split, and the benchmark preflight.

Protects the claim that no gene appears on both sides of a split. In-memory
FASTA files only; no data files.
"""
import importlib.util
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest
from Bio import SeqIO

from data.cv_utils import (gene_overlap, get_cv_splits, get_gene_ids,
                           load_sequences_in_order)

ROOT = Path(__file__).resolve().parents[1]


def _fasta(path, entries, seed=0):
    """entries: list of (transcript_number, gene_number)."""
    rng = random.Random(seed)
    lines = []
    for tid, gid in entries:
        seq = "".join(rng.choice("ACGT") for _ in range(rng.randint(200, 3000)))
        lines += [f">ENST{tid:011d}.1|ENSG{gid:011d}.5|x|y|n|n|{len(seq)}|biotype|", seq]
    Path(path).write_text("\n".join(lines) + "\n")
    return path


def _genes(first_gene, n_genes, first_tid, isoforms=3):
    return [(first_tid + g * isoforms + i, first_gene + g)
            for g in range(n_genes) for i in range(isoforms)]


@pytest.fixture
def records(tmp_path):
    lnc = _fasta(tmp_path / "lnc.fa", _genes(0, 60, 0), seed=1)
    pc = _fasta(tmp_path / "pc.fa", _genes(1000, 60, 100000), seed=2)
    return load_sequences_in_order(str(lnc), str(pc))


# ---------------------------------------------------------------- helpers

def test_gene_id_is_versionless_ensg(tmp_path):
    fa = _fasta(tmp_path / "a.fa", [(1, 7)])
    rec = next(SeqIO.parse(fa, "fasta"))
    assert get_gene_ids([rec]) == ["ENSG00000000007"]


def test_gene_id_missing_raises(tmp_path):
    fa = tmp_path / "a.fa"
    fa.write_text(">ENST00000000001.1|no_gene_here\nACGT\n")
    rec = next(SeqIO.parse(fa, "fasta"))
    with pytest.raises(ValueError):
        get_gene_ids([rec])


def test_gene_overlap_counts_validation_records():
    genes = ["A", "A", "B", "C"]
    assert gene_overlap(genes, train_idx=[0, 2], val_idx=[1, 3]) == (1, 2)


# --------------------------------------------------------------- CV folds

def test_grouped_cv_folds_share_no_gene(records):
    seqs, labels = records
    genes = get_gene_ids(seqs)
    splits = get_cv_splits(seqs, labels, n_folds=5, group_by_gene=True)
    assert len(splits) == 5
    seen = []
    for train_idx, val_idx in splits:
        assert gene_overlap(genes, train_idx, val_idx)[0] == 0
        seen += list(val_idx)
    assert sorted(seen) == list(range(len(seqs)))


def test_legacy_folds_do_share_genes(records):
    """The guard is meaningful: without grouping, isoforms leak across folds."""
    seqs, labels = records
    genes = get_gene_ids(seqs)
    splits = get_cv_splits(seqs, labels, n_folds=5, group_by_gene=False)
    assert any(gene_overlap(genes, tr, va)[0] > 0 for tr, va in splits)


# ------------------------------------------------------------- test split

def _load_split_script():
    path = ROOT / "analysis" / "prepare_gencode" / "scripts" / "split_test_set.py"
    spec = importlib.util.spec_from_file_location("split_test_set", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_test_split_is_gene_disjoint(tmp_path):
    mod = _load_split_script()
    lnc = list(SeqIO.parse(_fasta(tmp_path / "l.fa", _genes(0, 80, 0), seed=3), "fasta"))
    pc = list(SeqIO.parse(_fasta(tmp_path / "p.fa", _genes(1000, 80, 100000), seed=4), "fasta"))
    lnc_tv, lnc_te, pc_tv, pc_te, stats = mod.gene_grouped_split(lnc, pc, test_size=0.2, seed=0)
    tv_genes = set(get_gene_ids(lnc_tv + pc_tv))
    te_genes = set(get_gene_ids(lnc_te + pc_te))
    assert tv_genes.isdisjoint(te_genes)
    assert stats["test_records_sharing_gene_with_trainval"] == 0
    assert len(lnc_tv) + len(lnc_te) == len(lnc)
    assert len(pc_tv) + len(pc_te) == len(pc)
    assert len(te_genes) > 0


# ----------------------------------------------- benchmark preflight guard

def _preflight_code():
    text = (ROOT / "analysis" / "benchmark" / "run_full_benchmark.sh").read_text()
    m = re.search(r"python - [^\n]*<<'PY'\n(.*?)\nPY\n", text, re.S)
    assert m, "preflight block not found in run_full_benchmark.sh"
    return m.group(1)


def _run_preflight(tmp_path, train_extra=(), require="true", drop_prediction=False):
    test_lnc = _fasta(tmp_path / "te_lnc.fa", _genes(0, 5, 0), seed=5)
    test_pc = _fasta(tmp_path / "te_pc.fa", _genes(1000, 5, 100000), seed=6)
    tr_lnc = _fasta(tmp_path / "tr_lnc.fa", _genes(10, 8, 1000) + list(train_extra), seed=7)
    tr_pc = _fasta(tmp_path / "tr_pc.fa", _genes(1010, 8, 200000), seed=8)
    ids = [f"ENST{t:011d}.1" for t, _ in _genes(0, 5, 0) + _genes(1000, 5, 100000)]
    if drop_prediction:
        ids = ids[:-1]
    pred = tmp_path / "pred.csv"
    pred.write_text("transcript_id,p_pc\n" + "".join(f"{i}|ENSG,0.5\n" for i in ids))
    args = [test_lnc, test_pc, tr_lnc, tr_pc, pred, require]
    return subprocess.run([sys.executable, "-", *map(str, args)], input=_preflight_code(),
                          capture_output=True, text=True)


def test_preflight_passes_when_disjoint(tmp_path):
    r = _run_preflight(tmp_path)
    assert r.returncode == 0, r.stderr
    assert "[preflight] OK" in r.stdout


def test_preflight_fails_on_one_shared_gene(tmp_path):
    shared = [(999999, 0)]          # a new isoform of test gene 0 inside trainval
    r = _run_preflight(tmp_path, train_extra=shared, require="true")
    assert r.returncode != 0
    assert "share 1 genes" in r.stderr


def test_preflight_only_warns_when_not_required(tmp_path):
    r = _run_preflight(tmp_path, train_extra=[(999999, 0)], require="false")
    assert r.returncode == 0, r.stderr
    assert "WARNING" in r.stdout


def test_preflight_fails_when_predictions_do_not_match_test_fastas(tmp_path):
    r = _run_preflight(tmp_path, drop_prediction=True)
    assert r.returncode != 0
    assert "differ" in r.stderr
