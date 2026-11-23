"""run_unseen_benchmark.sh: B2 and B3 are scored from the full-test run's provenance.

Builds fake GENCODE releases, a tiny test set with predictions, runs
compute_benchmark_ci.py (the full-test run), then run_unseen_benchmark.sh, and
checks the seen/unseen outputs against the exposure lists.
"""
import gzip
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / "analysis" / "benchmark" / "compute_benchmark_ci.py"
RUN = ROOT / "analysis" / "benchmark" / "run_unseen_benchmark.sh"
N_GENES = 160
FIRST = [20, 40, 47, 49]          # first release of a gene's transcripts


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("unseen")
    rng = np.random.default_rng(3)
    rows = []                     # (tid, gene, label, first_release)
    for g in range(N_GENES):
        first = FIRST[g % 4]
        for i in range(3):
            rows.append((f"ENST{g * 10 + i:011d}.1", f"ENSG{g:011d}.1", g % 2, first))
    gtfs = d / "gtf"
    gtfs.mkdir()
    for r in range(20, 50):
        with gzip.open(gtfs / f"gencode.v{r}.annotation.gtf.gz", "wt") as fh:
            for t, g, _l, first in rows:
                if r >= first:
                    fh.write(f'chr1\tH\ttranscript\t1\t9\t.\t+\t.\t'
                             f'gene_id "{g}"; transcript_id "{t}";\n')
    for name, lab in (("lnc", 0), ("pc", 1)):
        with open(d / f"{name}.fa", "w") as fh:
            for t, g, l, _f in rows:
                if l == lab:
                    fh.write(f">{t}|{g}|x\nACGT\n")
    y = np.array([r[2] for r in rows])
    p = np.clip(np.where(y == 1, 0.7, 0.3) + rng.normal(0, 0.25, len(y)), 0.001, 0.999)
    pred = p >= 0.5
    exp = d / "gencode_v49_experiments"
    (exp / "benchmark_comparison").mkdir(parents=True)
    pd.DataFrame({"transcript_id": [f"{r[0]}|{r[1]}" for r in rows],
                  "true_label": np.where(y == 1, "pc", "lnc"),
                  "consensus_prediction": np.where(pred, "pc", "lnc"),
                  "mean_confidence": np.where(pred, p, 1 - p)}).to_csv(d / "blnc.csv", index=False)
    pd.DataFrame({"transcript_id": [r[0] for r in rows], "prob_pc": p}
                 ).to_csv(d / "bert.csv", index=False)
    cmp_dir = exp / "benchmark_comparison"
    r = subprocess.run([sys.executable, str(CI), "--blnc_csv", d / "blnc.csv",
                        "--lncrnabert_csv", d / "bert.csv",
                        "--test_lnc_fasta", d / "lnc.fa", "--test_pc_fasta", d / "pc.fa",
                        "--common_subset", "--n_bootstrap", "100", "--output_dir", cmp_dir],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    return dict(d=d, exp=exp, cmp=cmp_dir, gtfs=gtfs, rows=rows)


def _run(world, *extra):
    return subprocess.run(
        ["bash", str(RUN), "--comparison_dir", str(world["cmp"]), "--release", "49",
         "--cache_dir", str(world["d"] / "cache"), "--local_gtf_dir", str(world["gtfs"]), *extra],
        capture_output=True, text=True, cwd=ROOT, env={"PATH": "/usr/bin:/bin", "PYTHON": sys.executable})


def test_dry_run_prints_the_three_commands(world):
    r = _run(world, "--dry_run", "true")
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("compute_benchmark_ci.py") == 2 and "unseen_subset.py" in r.stdout
    assert "--common_subset" in r.stdout and "--contrast_id_list" in r.stdout
    assert not (world["exp"] / "unseen_subset").exists()


def test_b2_and_b3_are_scored_on_the_exposure_lists(world):
    r = _run(world)
    assert r.returncode == 0, r.stdout + r.stderr
    exp = world["exp"]
    unseen = set((exp / "unseen_subset/unseen_ids_L2_after_v46.txt").read_text().split())
    seen = set((exp / "unseen_subset/seen_ids_L2_by_v46.txt").read_text().split())
    assert unseen and seen and unseen.isdisjoint(seen)
    assert len(unseen) + len(seen) == len(world["rows"])
    b2 = pd.read_csv(exp / "benchmark_unseen_v46/benchmark_table_with_ci_allavailable.csv")
    b3 = pd.read_csv(exp / "benchmark_seen_v46/benchmark_table_with_ci_allavailable.csv")
    assert (b2.n_samples == len(unseen)).all() and (b3.n_samples == len(seen)).all()
    assert (exp / "benchmark_unseen_v46/benchmark_contrast_vs_blnc.csv").is_file()
    assert not (exp / "benchmark_seen_v46/benchmark_contrast_vs_blnc.csv").exists()
    assert (exp / "benchmark_unseen_ids_L2_after_v46.log").is_file()
    assert (exp / "benchmark_seen_ids_L2_by_v46.log").is_file()


def test_same_predictions_as_the_full_run(world):
    import json
    full = json.load(open(world["cmp"] / "benchmark_provenance.json"))
    for name in ("benchmark_unseen_v46", "benchmark_seen_v46"):
        sub = json.load(open(world["exp"] / name / "benchmark_provenance.json"))
        assert sub["files"] == full["files"] and sub["thresholds"] == full["thresholds"]
        assert sub["n_bootstrap"] == full["n_bootstrap"] and sub["seed"] == full["seed"]
        assert sub["id_list"] is not None


def test_missing_provenance_stops(tmp_path):
    r = subprocess.run(["bash", str(RUN), "--comparison_dir", str(tmp_path), "--release", "49"],
                       capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    assert r.returncode != 0 and "benchmark_provenance.json" in r.stdout
