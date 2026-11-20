"""compute_benchmark_ci.py reports gene-clustered percentile bootstrap CIs only.

Protects: no DeLong or isoform-naive columns, point AUC equal to sklearn's,
the test FASTAs being required, and the end-to-end CSV columns.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from analysis.benchmark import compute_benchmark_ci as ci

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "analysis" / "benchmark" / "compute_benchmark_ci.py"
N_GENES, ISOFORMS = 120, 3
NAIVE = ("f1_ci_low", "f1_ci_high", "auc_ci_low", "auc_ci_high")


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ci")
    rng = np.random.default_rng(0)
    rows = []                                     # (tid, gene, label)
    for g in range(N_GENES):
        lab = g % 2                               # 0 = lnc, 1 = pc; one class per gene
        rows += [(f"ENST{g * 10 + i:011d}", f"ENSG{g:011d}", lab) for i in range(ISOFORMS)]
    for name, lab in (("lnc", 0), ("pc", 1)):
        with open(tmp / f"{name}.fa", "w") as fh:
            for tid, gene, l in rows:
                if l == lab:
                    fh.write(f">{tid}.1|{gene}.2|x\nACGT\n")
    y = np.array([r[2] for r in rows])
    p_pc = np.clip(np.where(y == 1, 0.75, 0.25) + rng.normal(0, 0.2, len(y)), 0.001, 0.999)
    pred_pc = p_pc >= 0.5
    pd.DataFrame({
        "transcript_id": [f"{r[0]}.1|{r[1]}.2" for r in rows],
        "true_label": np.where(y == 1, "pc", "lnc"),
        "consensus_prediction": np.where(pred_pc, "pc", "lnc"),
        "mean_confidence": np.where(pred_pc, p_pc, 1 - p_pc),
    }).to_csv(tmp / "blnc.csv", index=False)
    pd.DataFrame({"transcript_id": [r[0] for r in rows], "prob_pc": p_pc}
                 ).to_csv(tmp / "bert.csv", index=False)
    return dict(tmp=tmp, y=y, p=p_pc)


def _run(args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          capture_output=True, text=True, cwd=ROOT)


def test_end_to_end_columns_are_gene_clustered_only(data):
    t = data["tmp"]
    out = t / "out"
    r = _run(["--blnc_csv", t / "blnc.csv", "--lncrnabert_csv", t / "bert.csv",
              "--test_lnc_fasta", t / "lnc.fa", "--test_pc_fasta", t / "pc.fa",
              "--n_bootstrap", 200, "--output_dir", out])
    assert r.returncode == 0, r.stderr
    df = pd.read_csv(out / "benchmark_table_with_ci.csv")
    assert set(df.method) == {"blnc", "lncrnabert"}
    for col in ("f1_gene_ci_low", "f1_gene_ci_high", "acc_gene_ci_low",
                "auc_gene_ci_low", "auc_gene_ci_high", "n_genes"):
        assert col in df.columns
    assert not [c for c in NAIVE if c in df.columns]
    assert (df.n_genes == N_GENES).all()
    assert ((df.f1_gene_ci_low <= df.f1) & (df.f1 <= df.f1_gene_ci_high)).all()
    prov = pd.read_json(out / "benchmark_provenance.json", typ="series")
    assert prov["ci"] == "gene-clustered percentile bootstrap"


def test_point_auc_equals_sklearn(data):
    t = data["tmp"]
    labels = ci.load_canonical_labels(str(t / "blnc.csv"))
    scores = ci.load_blnc_style(str(t / "blnc.csv"))
    genes = ci.load_gene_map([str(t / "lnc.fa"), str(t / "pc.fa")])
    row = ci.evaluate_method("blnc", scores, labels, 0.5, 100,
                             np.random.default_rng(0), genes)
    merged = scores.merge(labels, on="id")
    assert abs(row["auc"] - roc_auc_score(merged.y_true, merged.y_score)) < 1e-12
    assert not [c for c in NAIVE if c in row]


def test_script_requires_the_test_fastas(data):
    t = data["tmp"]
    r = _run(["--blnc_csv", t / "blnc.csv", "--output_dir", t / "out2"])
    assert r.returncode != 0
    assert "--test_lnc_fasta" in r.stderr


def test_no_delong_code_left():
    assert not hasattr(ci, "delong_auc_ci") and not hasattr(ci, "bootstrap_f1_ci")


def test_provenance_records_the_cpat_threshold_rule(data):
    t = data["tmp"]
    pd.DataFrame({"transcript_id": [f"ENST{g * 10:011d}" for g in range(N_GENES)],
                  "coding_prob": data["p"][::ISOFORMS][:N_GENES]}).to_csv(t / "cpat.csv", index=False)
    notes = {}
    for tag, extra in (("default", []), ("override", ["--cpat_threshold", 0.41])):
        out = t / f"out_cpat_{tag}"
        r = _run(["--blnc_csv", t / "blnc.csv", "--cpat_csv", t / "cpat.csv",
                  "--test_lnc_fasta", t / "lnc.fa", "--test_pc_fasta", t / "pc.fa",
                  "--n_bootstrap", 50, "--no_paired", "--output_dir", out, *extra])
        assert r.returncode == 0, r.stderr
        notes[tag] = json.loads((out / "benchmark_provenance.json").read_text())["threshold_notes"]
    assert "0.364" in notes["default"]["cpat"] and "not tuned on the test set" in notes["default"]["cpat"]
    assert "--cpat_threshold" in notes["override"]["cpat"] and "0.41" in notes["override"]["cpat"]
