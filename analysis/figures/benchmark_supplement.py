"""Supplement benchmark tables: all-available, per-class, paired, contrast, exposure."""
from pathlib import Path

import pandas as pd

from . import tables
from .benchmark_table import CSV, DISPLAY

METRICS = {"f1": "macro-F1", "recall_lnc": "lncRNA recall"}
CLASSES = {"lnc": "lncRNA", "pc": "mRNA", "all": "all"}


def _read(path):
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _name(method):
    if method not in DISPLAY:
        raise KeyError(f"no display name for '{method}'; add it to DISPLAY in benchmark_table.py")
    return DISPLAY[method]


def _f(x):
    return f"{x:.3f}".replace("-0.000", "0.000")


def _ci(r, v, lo, hi):
    return f"{_f(r[v])} [{_f(r[lo])}, {_f(r[hi])}]"


def _p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def build(b1, b2, b3, exposure_dir, out_dir, cutoff=46):
    """Write the supplement tables (5 CSV + LaTeX pairs) into out_dir; return (inputs, outputs).

    b1, b2, b3: benchmark folders (full, unseen, seen); exposure_dir: unseen_subset folder.
    Methods are ordered by their full-test F1. A missing file stops the run, naming it."""
    subsets = [("Full test", Path(b1)), (f"Unseen by v{cutoff}", Path(b2)), (f"Seen by v{cutoff}", Path(b3))]
    inputs, out = [], []

    def read(path):
        inputs.append(Path(path))
        return _read(path)

    common = {lab: read(d / CSV).set_index("method") for lab, d in subsets}
    allav = {lab: read(d / CSV.replace(".csv", "_allavailable.csv")).set_index("method")
             for lab, d in subsets}
    paired = {lab: read(d / "benchmark_paired_vs_blnc.csv") for lab, d in subsets}
    contrast = read(Path(b2) / "benchmark_contrast_vs_blnc.csv")
    exposure = read(Path(exposure_dir) / "exposure_summary.csv")
    order = list(common["Full test"]["f1"].sort_values(ascending=False).index)
    label = {lab: f"{lab} (n = {int(common[lab]['n_samples'].iloc[0]):,})" for lab, _ in subsets}

    rows = [[lab, _name(m), f"{int(r['n_samples']):,}", _ci(r, "f1", "f1_gene_ci_low", "f1_gene_ci_high"),
             _ci(r, "auc", "auc_gene_ci_low", "auc_gene_ci_high"),
             _ci(r, "accuracy", "acc_gene_ci_low", "acc_gene_ci_high"), f"{r['coverage']:.3f}"]
            for lab, _ in subsets for m in order for r in [allav[lab].loc[m]]]
    out += tables.write(out_dir, "supp_allavailable",
                        ["Subset", "Method", "n", "Macro-F1", "AUC", "Accuracy", "Coverage"], rows)

    rows = [[label[lab], _name(m), f"{r['recall_lnc']:.3f}", f"{r['precision_lnc']:.3f}",
             f"{r['recall_pc']:.3f}", f"{r['precision_pc']:.3f}"]
            for lab, _ in subsets for m in order for r in [common[lab].loc[m]]]
    out += tables.write(out_dir, "supp_perclass",
                        ["Subset", "Method", "lncRNA recall", "lncRNA precision", "mRNA recall",
                         "mRNA precision"], rows)

    rows = []
    for lab, _ in subsets:
        df = paired[lab].set_index("method")
        rows += [[lab, _name(m), f"{r['f1_reference']:.3f}", f"{r['f1_method']:.3f}",
                  _ci(r, "diff", "diff_ci_low", "diff_ci_high"), _p(r["p_boot"])]
                 for m in order if m in df.index for r in [df.loc[m]]]
    out += tables.write(out_dir, "supp_paired",
                        ["Subset", "Method", "Macro-F1 β-LNC", "Macro-F1 method",
                         "Difference β-LNC minus method", "p"], rows)

    rows = [[_name(r["method"]), METRICS.get(r["metric"], r["metric"]),
             _ci(r, "diff_focus", "diff_focus_ci_low", "diff_focus_ci_high"),
             _ci(r, "diff_contrast", "diff_contrast_ci_low", "diff_contrast_ci_high"),
             _ci(r, "swing", "swing_ci_low", "swing_ci_high"), _p(r["p_swing"])]
            for m in order for _, r in contrast[contrast["method"] == m].iterrows()]
    out += tables.write(out_dir, "supp_contrast",
                        ["Method", "Metric", f"Unseen by v{cutoff}: β-LNC minus method",
                         f"Seen by v{cutoff}: β-LNC minus method", "Swing (unseen minus seen)", "p"], rows)

    rows = [[f"v{r['cutoff']}", r["level"], CLASSES[r["cls"]], f"{int(r['n_test']):,}",
             f"{int(r['n_unseen']):,}", f"{r['pct_unseen']:.1f}", f"{r['pct_seen']:.1f}"]
            for _, r in exposure.iterrows()]
    out += tables.write(out_dir, "supp_exposure",
                        ["Cutoff", "Level", "Class", "Test transcripts", "Unseen", "% unseen", "% seen"],
                        rows)
    return inputs, out