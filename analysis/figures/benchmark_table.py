"""Main benchmark table: macro-F1 on three subsets, AUC on the full test set."""
from pathlib import Path

import pandas as pd

CSV = "benchmark_table_with_ci.csv"  # common-subset table with gene-clustered CIs
DISPLAY = {"lncrnabert": "lncRNA-BERT", "blnc": "β-LNC", "rnasamba": "RNAsamba",
           "cpat": "CPAT", "feature_only": "Feature-only", "lncdc": "lncDC",
           "cpc2": "CPC2", "orthrus": "Orthrus"}
# (folder key, value column, CI low, CI high, header)
SPECS = [("b1", "f1", "f1_gene_ci_low", "f1_gene_ci_high", "Full test"),
         ("b2", "f1", "f1_gene_ci_low", "f1_gene_ci_high", "Unseen by v46"),
         ("b3", "f1", "f1_gene_ci_low", "f1_gene_ci_high", "Seen by v46"),
         ("b1", "auc", "auc_gene_ci_low", "auc_gene_ci_high", "AUC, full test")]


def _load(path):
    if not Path(path).exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path).set_index("method")
    if df["n_samples"].nunique() != 1:
        raise ValueError(f"{path}: n_samples differs between methods (not the common subset)")
    return df


def _tex(s):
    return s.replace("β", r"$\beta$")


def build(b1, b2, b3, out_dir):
    """Write benchmark_table.csv and benchmark_table.tex; return (inputs, outputs).

    b1, b2, b3: benchmark folders (full, unseen, seen), each holding CSV.
    Raises if a file is missing, the method sets differ or a method has no DISPLAY name."""
    folders = {"b1": Path(b1), "b2": Path(b2), "b3": Path(b3)}
    inputs = [folders[k] / CSV for k in folders]
    frames = {k: _load(p) for k, p in zip(folders, inputs)}
    methods = set(frames["b1"].index)
    for k, df in frames.items():
        if set(df.index) != methods:
            raise ValueError(f"{folders[k] / CSV}: methods differ from the full-test table")
    unnamed = sorted(methods - set(DISPLAY))
    if unnamed:
        raise KeyError(f"no display name for {unnamed}; add them to DISPLAY in benchmark_table.py")
    order = frames["b1"]["f1"].sort_values(ascending=False).index
    heads = []
    for key, val, lo, hi, label in SPECS:
        n = int(frames[key]["n_samples"].iloc[0])
        heads.append((label if val == "auc" else f"{label} (n = {n:,})",
                      label if val == "auc" else f"{label} ($n={n:,}$)".replace(",", "{,}")))
    best = [round(frames[k][v].max(), 3) for k, v, *_ in SPECS]
    plain, tex = [], []
    for m in order:
        cells, tcells = [], []
        for (k, v, lo, hi, _), b in zip(SPECS, best):
            r = frames[k].loc[m]
            s = f"{r[v]:.3f} [{r[lo]:.3f}, {r[hi]:.3f}]"
            cells.append(s)
            tcells.append(rf"\textbf{{{s}}}" if round(r[v], 3) == b else s)
        plain.append([DISPLAY[m]] + cells)
        tcells.insert(0, _tex(DISPLAY[m]))
        tex.append(" & ".join(tcells) + r" \\")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(plain, columns=["Method"] + [h[0] for h in heads]).to_csv(
        out / "benchmark_table.csv", index=False, encoding="utf-8")
    lines = [r"\begin{tabular}{l" + "c" * len(SPECS) + "}", r"\toprule",
             " & ".join(["Method"] + [h[1] for h in heads]) + r" \\", r"\midrule",
             *tex, r"\bottomrule", r"\end{tabular}"]
    (out / "benchmark_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return inputs, [out / "benchmark_table.csv", out / "benchmark_table.tex"]