#!/usr/bin/env python3
"""Paired gene-clustered bootstrap of the macro-F1 / accuracy difference between two methods.

Input, one of:
  --merged predictions_merged.csv   benchmark_comparison file: true_int + <method>_hard error flags
                                    (*_hard = misclassified; prediction = truth flipped where hard)
  --blnc F --bert F                 per-method test_predictions.csv (transcript_id, true_label, consensus_prediction)
Gene ids: --fasta test FASTAs (GENCODE '>ENST|ENSG|...') or --gene_map CSV/TSV (transcript_id, gene_id).
Resamples whole genes with replacement and scores both methods on the same resample.
Reports delta = metric(a) - metric(b), percentile 95% CI, two-sided bootstrap p.

usage: python paired_gene_bootstrap.py --merged predictions_merged.csv \
         --fasta lnc_test.fa pc_test.fa [--a_col blnc_hard --b_col lncrnabert_hard] --out diff.json"""
import argparse, json, sys
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--merged"); ap.add_argument("--a_col", default="blnc_hard"); ap.add_argument("--b_col", default="lncrnabert_hard")
ap.add_argument("--blnc"); ap.add_argument("--bert")
ap.add_argument("--fasta", nargs="*", default=[]); ap.add_argument("--gene_map")
ap.add_argument("--n_boot", type=int, default=10000); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--save_merged", help="optional: write the merged per-transcript table")
ap.add_argument("--out")
a = ap.parse_args()

strip = lambda s: s.astype(str).str.replace(r"\.\d+$", "", regex=True)

# ---- 1. predictions (y = truth, pa / pb = hard predictions) -----------------
if a.merged:
    d = pd.read_csv(a.merged).drop_duplicates("transcript_id").dropna(subset=["true_int", a.a_col, a.b_col])
    y = d.true_int.astype(int)
    for c in (a.a_col, a.b_col):
        print(f"{c}: error rate {d[c].astype(bool).mean():.4f} (compare 1 - accuracy in benchmark_table.csv)")
    m = pd.DataFrame({"transcript_id": strip(d.transcript_id), "y": y,
                      "pa": y.where(~d[a.a_col].astype(bool), 1 - y),
                      "pb": y.where(~d[a.b_col].astype(bool), 1 - y)})
elif a.blnc and a.bert:
    lab = {"lnc": 0, "pc": 1}
    def read(p):
        d = pd.read_csv(p).drop_duplicates("transcript_id")
        d["y"], d["p"] = d.true_label.map(lab), d.consensus_prediction.map(lab)
        if d[["y", "p"]].isna().any().any(): sys.exit("unexpected label values (expected lnc/pc)")
        return d[["transcript_id", "y", "p"]]
    m = read(a.blnc).merge(read(a.bert), on="transcript_id", suffixes=("_a", "_b"))
    if (m.y_a != m.y_b).any(): sys.exit("conflicting true labels between the two files")
    m = m.rename(columns={"y_a": "y", "p_a": "pa", "p_b": "pb"})[["transcript_id", "y", "pa", "pb"]]
    m["transcript_id"] = strip(m.transcript_id)
else:
    sys.exit("give --merged, or --blnc and --bert")

# ---- 2. gene ids -------------------------------------------------------------
if a.gene_map:
    g = pd.read_csv(a.gene_map, sep=None, engine="python")[["transcript_id", "gene_id"]]
else:
    rows = [tuple(l[1:].strip().split("|")[:2]) for f in a.fasta for l in open(f) if l.startswith(">")]
    g = pd.DataFrame(rows, columns=["transcript_id", "gene_id"])
g["transcript_id"] = strip(g.transcript_id)
m = m.merge(g.drop_duplicates("transcript_id"), on="transcript_id", how="left")
print(f"transcripts {len(m)}, missing gene id {m.gene_id.isna().sum()}")
m = m.dropna(subset=["gene_id"])
if a.save_merged: m.to_csv(a.save_merged, index=False)

# ---- 3. paired gene-clustered bootstrap ---------------------------------------
y, pa, pb = m.y.to_numpy(int), m.pa.to_numpy(int), m.pb.to_numpy(int)
codes, genes = pd.factorize(m.gene_id); G = len(genes)

def per_gene(pred):  # rows = gene; cols = tp, tn, fp, fn
    c = np.zeros((G, 4))
    for j, k in enumerate([(y == 1) & (pred == 1), (y == 0) & (pred == 0), (y == 0) & (pred == 1), (y == 1) & (pred == 0)]):
        np.add.at(c[:, j], codes, k)
    return c

def metrics(c):  # summed counts -> macro F1, accuracy
    tp, tn, fp, fn = c
    return ((2 * tp / max(2 * tp + fp + fn, 1) + 2 * tn / max(2 * tn + fn + fp, 1)) / 2,
            (tp + tn) / max(tp + tn + fp + fn, 1))

CA, CB = per_gene(pa), per_gene(pb)
pt = np.array(metrics(CA.sum(0))) - np.array(metrics(CB.sum(0)))
rng = np.random.default_rng(a.seed)
dl = np.empty((a.n_boot, 2))
for i in range(a.n_boot):
    idx = rng.integers(0, G, G)
    dl[i] = np.array(metrics(CA[idx].sum(0))) - np.array(metrics(CB[idx].sum(0)))
res = {"n_transcripts": int(len(m)), "n_genes": int(G), "n_boot": a.n_boot, "a": a.a_col if a.merged else "blnc", "b": a.b_col if a.merged else "bert"}
for k, name in enumerate(["macro_f1", "accuracy"]):
    lo, hi = np.percentile(dl[:, k], [2.5, 97.5])
    p = min(2 * min((dl[:, k] <= 0).mean(), (dl[:, k] >= 0).mean()), 1.0)
    res[name] = {"delta_a_minus_b": float(pt[k]), "ci95": [float(lo), float(hi)], "p_two_sided": float(p)}
print(json.dumps(res, indent=2))
if a.out: json.dump(res, open(a.out, "w"), indent=2)