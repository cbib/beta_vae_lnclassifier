"""merge_nonb2.py writes a report of duplicate and wrong-length rows.

Protects: duplicate rows and conflicting duplicates are counted per input file,
rows whose reported length differs from the FASTA are counted (and the ids whose
FIRST row is wrong, which a keep-first rule would retain), and the merge itself
is unchanged when --fasta is omitted.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "data" / "merge_nonb2.py"


def _setup(tmp):
    # FASTA lengths: T1=10, T2=20, T3=30, T4=40
    with open(tmp / "t.fa", "w") as fh:
        for i, n in enumerate((10, 20, 30, 40), 1):
            fh.write(f">ENST0000000000{i}.1|ENSG0000000000{i}.1|x\n{'A' * n}\n")
    sf = pd.DataFrame({
        "transcript_id": ["ENST00000000001.1|g", "ENST00000000002.1|g",   # T1 ok, T2 first row wrong
                          "ENST00000000002.1|g",                          # T2 duplicate, correct, features differ
                          "ENST00000000003.1|g", "ENST00000000003.1|g",   # T3 exact duplicate, correct
                          "ENST00000000004.1|g"],                         # T4 wrong length only
        "length": [10, 21, 20, 30, 30, 39],
        "simple_ID": list("abcdef"),
        "f_a": [0.1, 0.2, 0.3, 0.4, 0.4, 0.5],
    })
    sf.to_csv(tmp / "sf.csv", index=False)
    pd.DataFrame({"transcript_id": [f"ENST0000000000{i}.1" for i in (1, 2, 3, 4)],
                  "transcript_length": [10, 20, 30, 40], "g": [1.0, 2.0, 3.0, 4.0]}
                 ).to_csv(tmp / "rg4.csv", index=False)


def _run(tmp, *extra):
    return subprocess.run([sys.executable, str(SCRIPT), "--scanfold_csv", str(tmp / "sf.csv"),
                           "--rg4_csv", str(tmp / "rg4.csv"), "--output_csv", str(tmp / "m.csv"),
                           *extra], capture_output=True, text=True, cwd=ROOT)


def test_report_counts(tmp_path):
    _setup(tmp_path)
    r = _run(tmp_path, "--fasta", str(tmp_path / "t.fa"))
    assert r.returncode == 0, r.stderr
    rep = json.loads((tmp_path / "m_report.json").read_text())
    sf = rep["scanfold"]
    assert (sf["n_rows"], sf["n_unique_ids"], sf["n_duplicate_rows"]) == (6, 4, 2)
    assert sf["n_ids_with_conflicting_features"] == 1      # T2 (0.2 vs 0.3); T3 duplicates agree
    assert sf["n_ids_with_conflicting_length"] == 1        # T2 (21 vs 20)
    lc = sf["length_check"]
    assert lc["n_rows_checked"] == 6 and lc["n_rows_wrong_length"] == 2
    assert lc["n_rows_longer_than_fasta"] == 1 and lc["n_rows_shorter_than_fasta"] == 1
    assert lc["n_ids_wrong_length"] == 2                   # T2 and T4
    assert lc["n_ids_only_wrong_rows"] == 1                # T4
    assert lc["n_ids_wrong_and_correct_rows"] == 1         # T2
    assert lc["n_ids_first_row_wrong_length"] == 2         # T2 (first row 21) and T4
    assert {e["id"] for e in lc["examples"]} == {"ENST00000000002", "ENST00000000004"}
    assert lc["first_row_wrong_ids"] == ["ENST00000000002", "ENST00000000004"]
    assert rep["join"]["n_ids_used_row_wrong_length"] == 2
    ids = (tmp_path / "m_wrong_length_ids.txt").read_text().split()
    assert ids == ["ENST00000000002", "ENST00000000004"]
    assert rep["rg4detector"]["length_check"]["n_rows_wrong_length"] == 0
    assert rep["join"]["n_common"] == 4


def test_report_without_fasta_has_no_length_check(tmp_path):
    _setup(tmp_path)
    r = _run(tmp_path)
    assert r.returncode == 0, r.stderr
    rep = json.loads((tmp_path / "m_report.json").read_text())
    assert rep["scanfold"]["length_check"] is None
    assert rep["scanfold"]["n_duplicate_rows"] == 2
    assert rep["inputs"]["fasta"] == []
    assert rep["join"]["n_ids_used_row_wrong_length"] is None
    assert not (tmp_path / "m_wrong_length_ids.txt").exists()
