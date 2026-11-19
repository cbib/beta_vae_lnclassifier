"""Tests for the paired and contrast (swing) gene-clustered bootstraps in compute_benchmark_ci.py."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

from analysis.benchmark import compute_benchmark_ci as ci


def _data():
    rng = np.random.default_rng(1)
    n = 3000
    genes = rng.integers(0, 600, n)
    y = rng.integers(0, 2, n)
    good = np.where(rng.random(n) < 0.93, y, 1 - y)
    bad = np.where(rng.random(n) < 0.85, y, 1 - y)
    return n, genes, y, good, bad


def test_fast_macro_f1_matches_sklearn():
    n, genes, y, good, bad = _data()
    codes, _ = pd.factorize(genes)
    for pred in (good, bad, np.zeros(n, int), np.ones(n, int)):
        c = ci._gene_confusion(y, pred, codes, codes.max() + 1).sum(0)
        ref = f1_score(y, pred, average="macro", zero_division=0)
        assert abs(ci._macro_f1_counts(c) - ref) < 1e-12


def test_identical_methods_have_zero_difference():
    _, genes, y, good, _ = _data()
    d, lo, hi, p, _G = ci.paired_f1_bootstrap(y, good, good, genes, 500, np.random.default_rng(0))
    assert d == 0 and lo == 0 and hi == 0 and p == 1.0


def test_better_method_has_positive_significant_difference():
    _, genes, y, good, bad = _data()
    d, lo, hi, p, G = ci.paired_f1_bootstrap(y, good, bad, genes, 2000, np.random.default_rng(0))
    ref = f1_score(y, good, average="macro") - f1_score(y, bad, average="macro")
    assert abs(d - ref) < 1e-12
    assert lo > 0 and p < 0.01
    assert G == len(np.unique(genes))


def test_swapping_methods_flips_the_point_estimate():
    _, genes, y, good, bad = _data()
    d = ci.paired_f1_bootstrap(y, good, bad, genes, 200, np.random.default_rng(0))[0]
    d2 = ci.paired_f1_bootstrap(y, bad, good, genes, 200, np.random.default_rng(0))[0]
    assert abs(d + d2) < 1e-12


def _contrast_data():
    rng0 = np.random.default_rng(7)
    n = 8000
    genes = rng0.integers(0, 1500, n)  # genes span both subsets
    in_a = rng0.random(n) < 0.5
    y = rng0.integers(0, 2, n)
    ref = np.where(rng0.random(n) < 0.93, y, 1 - y)

    def method(acc_a, acc_b):
        acc = np.where(in_a, acc_a, acc_b)
        return np.where(rng0.random(n) < acc, y, 1 - y)

    return genes, in_a, y, ref, method


def test_contrast_recovers_planted_swing():
    genes, in_a, y, ref, method = _contrast_data()
    oth = method(0.88, 0.97)  # worse than ref on A, better on B
    out, _G = ci.contrast_bootstrap(y, ref, oth, genes, in_a, ["f1", "recall_lnc", "recall_pc"],
                                    2000, np.random.default_rng(0))
    f = out["f1"]
    da = f1_score(y[in_a], ref[in_a], average="macro") - f1_score(y[in_a], oth[in_a], average="macro")
    db = f1_score(y[~in_a], ref[~in_a], average="macro") - f1_score(y[~in_a], oth[~in_a], average="macro")
    assert abs(f["diff_focus"] - da) < 1e-12 and abs(f["diff_contrast"] - db) < 1e-12
    assert abs(f["swing"] - (da - db)) < 1e-12
    assert f["swing"] > 0.05 and f["swing_ci_low"] > 0 and f["p_swing"] < 0.01
    for key, pos in (("recall_lnc", 0), ("recall_pc", 1)):
        r = recall_score(y[in_a], ref[in_a], pos_label=pos) - recall_score(y[in_a], oth[in_a], pos_label=pos)
        assert abs(out[key]["diff_focus"] - r) < 1e-12


def test_contrast_null_covers_zero():
    genes, in_a, y, ref, method = _contrast_data()
    oth0 = method(0.90, 0.90)
    out0, _ = ci.contrast_bootstrap(y, ref, oth0, genes, in_a, ["f1"], 2000, np.random.default_rng(0))
    z = out0["f1"]
    assert z["swing_ci_low"] < 0 < z["swing_ci_high"] and z["p_swing"] > 0.05


def test_swapping_subsets_flips_the_swing():
    genes, in_a, y, ref, method = _contrast_data()
    oth = method(0.88, 0.97)
    out, _ = ci.contrast_bootstrap(y, ref, oth, genes, in_a, ["f1"], 300, np.random.default_rng(0))
    outs, _ = ci.contrast_bootstrap(y, ref, oth, genes, ~in_a, ["f1"], 300, np.random.default_rng(0))
    assert abs(outs["f1"]["swing"] + out["f1"]["swing"]) < 1e-12