#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_audit.py - one-shot, multi-layer audit of a beta-LNC experiment.

Stages (each prints PASS / WARN / FAIL lines; exit code 1 if any FAIL):
  config    paths exist; one release; group_by_gene on; split FASTAs in one folder
  split     ID/version collisions; gene + sequence overlap train/val vs test
  features  ID coverage; duplicates; NaN; structured zeros (position / ENST order /
            split); proportion groups summing to 1; column order vs *_feature_names.txt
  leakage   single-feature AUC vs class, length correlation, zero-pattern AUC
  scalers   stored scaler bank == scaler re-fit on trainval rows of the CSV;
            bank/CSV/raw file age
  model     checkpoints newer than features; test_predictions vs test FASTA and
            test_results.json; benchmark predictions_merged vs test_predictions
  layers    run manifest vs current checkpoints/config; result CSVs finite, IIA in
            [0,1], newer than checkpoints; results touching degenerate subgroups

Usage:
  python scripts/pipeline_audit.py --config configs/beta_vae_subgroup_base_g49.json \
      [--exp_root .] [--benchmark_dir gencode_v49_experiments/benchmark_comparison] \
      [--raw_nonb2 data/nonb2_te_raw/nonb2_features_v49_fixed.csv] \
      [--stages config,split,features,leakage,scalers,model,layers] [--out audit_g49] \
      [--path_map OLD=NEW ...]
--path_map rewrites OLD to NEW (plain substring) in every path the audit reads from the config,
from checkpoint configs and from the layers manifest, so results written under old folder
names are audited under the current ones. Repeat it for several renames; applied in order.
Stages that need absent inputs (no checkpoints yet, no layers yet) are SKIPPED, not failed.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ALL_STAGES = ["config", "split", "features", "leakage", "scalers", "model", "layers"]
RES: list[dict] = []
DEGENERATE: dict[str, dict] = {}   # family -> info, shared with the layers stage


# ------------------------------------------------------------------ reporting
def rec(level: str, stage: str, check: str, msg: str) -> None:
    RES.append({"level": level, "stage": stage, "check": check, "msg": msg})
    tag = {"PASS": "  ok  ", "WARN": " WARN ", "FAIL": " FAIL ", "SKIP": " skip ", "INFO": " info "}[level]
    print(f"[{tag}] {stage}/{check}: {msg}")


def head(t: str) -> None:
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ------------------------------------------------------------------ helpers
def norm_id(x) -> str:
    return str(x).split("|")[0].split(".")[0]


def read_fasta(path):
    """[(raw_id, header, length, sha1(seq))]."""
    out, hdr, h, n = [], None, None, 0
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if hdr is not None:
                    out.append((hdr.split()[0], hdr, n, h.hexdigest()))
                hdr, h, n = line[1:].strip(), hashlib.sha1(), 0
            else:
                s = line.strip().upper()
                h.update(s.encode()); n += len(s)
        if hdr is not None:
            out.append((hdr.split()[0], hdr, n, h.hexdigest()))
    return out


def gene_of(header: str):
    parts = header.split("|")
    if len(parts) > 1 and parts[1].startswith("ENSG"):
        return parts[1].split(".")[0]
    m = re.search(r"ENSG\d+", header)
    return m.group(0) if m else None


def family(col: str) -> str:
    return re.sub(r"\d+$", "", col)


def sha16(p) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def auc_matrix(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """Rank AUC of every column against binary y (1 = positive)."""
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return pd.Series(np.nan, index=X.columns)
    r = X.rank(method="average").to_numpy()
    return pd.Series((r[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0), index=X.columns)


def jget(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def find_key(obj, names):
    """First value whose key is in names, searching nested dicts."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in names and isinstance(v, (int, float)):
                return v
        for v in obj.values():
            r = find_key(v, names)
            if r is not None:
                return r
    return None


# ------------------------------------------------------------------ context
def parse_path_map(items) -> list:
    out = []
    for it in items or []:
        old, sep, new = it.partition("=")
        if not sep or not old:
            sys.exit(f"ERROR: --path_map expects OLD=NEW, got {it!r}")
        out.append((old, new))
    return out


class Ctx:
    def __init__(self, a):
        self.a = a
        self.cfg_path = a.config
        self.path_map = parse_path_map(getattr(a, "path_map", None))
        self.path_map_hits = {old: 0 for old, _ in self.path_map}
        self.cfg = self.remap(json.load(open(a.config)))
        self.d = self.cfg.get("data", {})
        self.t = self.cfg.get("training", {})
        self.root = Path(a.exp_root)
        self.fa = {}          # name -> list of records
        self.label, self.split, self.length, self.gene = {}, {}, {}, {}
        self.trainval_ids: set = set()
        self.exp_dir = self.root / jget(self.cfg, "output", "experiment_name", default="__none__")
        self.n_folds = int(self.t.get("n_folds", 5))
        self.release = None
        self.feat_csvs = {}   # label -> path
        self.banks = {}       # block -> (csv label, bank path)
        self.frames = {}      # label -> DataFrame (cached)

    def remap(self, x):
        """Apply --path_map to every string (keys and values) of a JSON-like object."""
        if isinstance(x, str):
            for old, new in self.path_map:
                if old in x:
                    x = x.replace(old, new)
                    self.path_map_hits[old] += 1
            return x
        if isinstance(x, dict):
            return {self.remap(k): self.remap(v) for k, v in x.items()}
        if isinstance(x, list):
            return [self.remap(v) for v in x]
        return x

    def load_fastas(self):
        keys = {"lnc_trainval": "lnc_fasta", "pc_trainval": "pc_fasta",
                "lnc_test": "lnc_test_fasta", "pc_test": "pc_test_fasta"}
        for name, k in keys.items():
            p = self.d.get(k)
            if p and Path(p).exists():
                self.fa[name] = read_fasta(p)
        for name, recs in self.fa.items():
            cls, sp = name.split("_")
            for raw, hdr, ln, _ in recs:
                i = norm_id(raw)
                self.label[i] = cls; self.split[i] = sp
                self.length[i] = ln; self.gene[i] = gene_of(hdr)
                if sp == "trainval":
                    self.trainval_ids.add(i)

    def frame(self, label):
        if label not in self.frames:
            df = pd.read_csv(self.feat_csvs[label], index_col=0)
            df.index = df.index.map(norm_id)
            self.frames[label] = df
        return self.frames[label]


# ------------------------------------------------------------------ stage: config
def stage_config(c: Ctx):
    head("STAGE config")
    missing = [(k, v) for k, v in c.d.items() if isinstance(v, str) and not Path(v).exists()]
    for k, v in missing:
        rec("FAIL", "config", "path_exists", f"data.{k} missing: {v}")
    if not missing:
        rec("PASS", "config", "path_exists", f"all {len(c.d)} data paths exist")

    rels = defaultdict(set)
    for k, v in c.d.items():
        if isinstance(v, str):
            for m in re.finditer(r"(?:split_gencode_|gencode_v|/g|^g)(\d{2})(?=[_/])", v):
                rels[m.group(1)].add(k)
    en = jget(c.cfg, "output", "experiment_name", default="")
    for m in re.finditer(r"gencode_v(\d{2})|_g(\d{2})", en):
        rels[m.group(1) or m.group(2)].add("output.experiment_name")
    if len(rels) == 1:
        c.release = next(iter(rels))
        rec("PASS", "config", "one_release", f"all paths refer to release {c.release}")
    elif rels:
        rec("FAIL", "config", "one_release",
            "mixed releases: " + "; ".join(f"{r}: {sorted(v)[:3]}" for r, v in rels.items()))
    gg = bool(c.t.get("group_by_gene"))
    rec("PASS" if gg else "FAIL", "config", "group_by_gene", f"training.group_by_gene = {gg}")
    # Gene-grouping is verified by group_by_gene and by the split-stage gene-overlap checks,
    # not by file names. Here: the four split FASTAs must come from one folder.
    split_dirs = {str(Path(c.d[k]).parent) for k in ("lnc_fasta", "pc_fasta", "lnc_test_fasta", "pc_test_fasta")
                  if c.d.get(k)}
    rec("PASS" if len(split_dirs) == 1 else "FAIL", "config", "split_dir",
        f"train/val and test FASTAs come from {len(split_dirs)} folder(s): {sorted(split_dirs)}")
    dirs = {Path(v).parent for k, v in c.d.items()
            if isinstance(v, str) and (k.endswith("_genomic_csv") or k == "nonb2_csv" or k.endswith("scaler_bank_path"))}
    if len(dirs) > 1:
        rec("WARN", "config", "feature_dir", f"feature CSVs/banks come from {len(dirs)} dirs: {sorted(map(str, dirs))}")
    bs = jget(c.cfg, "model", "block_sources")
    rec("INFO", "config", "block_sources", f"{bs}")

    # register feature CSVs and banks used by later stages
    for k, v in c.d.items():
        if isinstance(v, str) and k.endswith("_csv") and "fasta" not in k:
            c.feat_csvs[k] = v
    for b in ("te", "nonb", "nonb2"):
        bp = c.d.get(f"{b}_scaler_bank_path")
        csv = c.d.get("nonb2_csv") if b == "nonb2" else c.d.get(f"{b}_genomic_csv")
        if bp and csv:
            c.banks[b] = (bp, csv)


# ------------------------------------------------------------------ stage: split
def stage_split(c: Ctx):
    head("STAGE split")
    if len(c.fa) < 4:
        rec("SKIP", "split", "fastas", "need lnc/pc trainval+test FASTAs"); return
    c.load_fastas()
    allr = [(n, r) for n, rs in c.fa.items() for r in rs]
    raw = defaultdict(set)
    for _, r in allr:
        raw[norm_id(r[0])].add(r[0].split("|")[0])
    coll = {k: v for k, v in raw.items() if len(v) > 1}
    rec("FAIL" if coll else "PASS", "split", "version_collisions",
        f"{len(coll)} normalised IDs map to >1 raw ID (version stripped)")
    cnt = defaultdict(list)
    for n, r in allr:
        cnt[norm_id(r[0])].append(n)
    dup_within = sum(1 for v in cnt.values() if len(v) > 1 and len(set(v)) == 1)
    dup_across = [k for k, v in cnt.items() if len(set(x.split("_")[1] for x in v)) > 1]
    rec("FAIL" if dup_across else "PASS", "split", "id_across_splits",
        f"{len(dup_across)} transcript IDs in both trainval and test")
    rec("WARN" if dup_within else "PASS", "split", "id_within_split", f"{dup_within} IDs repeated inside one split")
    miss_gene = sum(1 for k in cnt if c.gene.get(k) is None)
    rec("FAIL" if miss_gene else "PASS", "split", "gene_in_header", f"{miss_gene} records without an ENSG id")

    tv_genes = {c.gene[i] for i in c.trainval_ids}
    te = [i for i, s in c.split.items() if s == "test"]
    sh = sum(c.gene[i] in tv_genes for i in te)
    rec("FAIL" if sh else "PASS", "split", "gene_overlap_test",
        f"{sh:,}/{len(te):,} test records share a gene with train/val ({100*sh/max(len(te),1):.2f}%)")
    seqs = defaultdict(set)
    for n, r in allr:
        seqs[r[3]].add(n.split("_")[1])
    sd = sum(len(v) > 1 for v in seqs.values())
    rec("WARN" if sd else "PASS", "split", "identical_sequences", f"{sd} sequences occur in both train/val and test")
    for sp in ("trainval", "test"):
        n_l, n_p = len(c.fa[f"lnc_{sp}"]), len(c.fa[f"pc_{sp}"])
        rec("INFO", "split", f"balance_{sp}", f"lnc {n_l:,}  pc {n_p:,}  lnc fraction {n_l/(n_l+n_p):.3f}")
    try:
        sys.path.insert(0, os.getcwd())
        from data.cv_utils import get_cv_splits, gene_overlap, get_gene_ids
        from Bio import SeqIO
        tv = list(SeqIO.parse(c.d["lnc_fasta"], "fasta")) + list(SeqIO.parse(c.d["pc_fasta"], "fasta"))
        lab = ["lnc"] * len(c.fa["lnc_trainval"]) + ["pc"] * len(c.fa["pc_trainval"])
        sp = get_cv_splits(tv, lab, c.n_folds, c.t.get("random_state", 42), c.t.get("n_bins", 5),
                           group_by_gene=bool(c.t.get("group_by_gene")))
        genes = get_gene_ids(tv)
        fr = [gene_overlap(genes, a, b)[0] / len(b) for a, b in sp]
        rec("FAIL" if max(fr) > 0 else "PASS", "split", "gene_overlap_cv",
            "configured CV folds, share of val records with a train gene: " + ", ".join(f"{100*x:.2f}%" for x in fr))
    except Exception as e:  # repo not importable from here
        rec("SKIP", "split", "gene_overlap_cv", f"needs repo (data.cv_utils) and Biopython: {type(e).__name__}")


# ------------------------------------------------------------------ stage: features
def family_stats(df: pd.DataFrame, c: Ctx, tag: str, stage="features"):
    """Zero / position structure per column family; fills DEGENERATE."""
    X = df.apply(pd.to_numeric, errors="coerce")
    fams = defaultdict(list)
    for col in X.columns:
        fams[family(col)].append(col)
    pos_dec = pd.qcut(np.arange(len(X)), 10, labels=False, duplicates="drop")
    enst = pd.Series([int(m.group(1)) if (m := re.match(r"ENST(\d+)", str(i))) else -1 for i in X.index])
    enst_dec = pd.qcut(enst.rank(method="first"), 10, labels=False) if (enst >= 0).all() else None
    spl = np.array([c.split.get(i, "?") for i in X.index])
    cls = np.array([c.label.get(i, "?") for i in X.index])
    n_bad = 0
    for fam, cols in fams.items():
        if len(cols) < 3:
            continue
        z = (X[cols].fillna(0).abs().sum(axis=1) == 0).to_numpy()
        rate = z.mean()
        if rate < 0.10:
            continue
        pr = pd.Series(z).groupby(pos_dec).mean()
        pos_rng = float(pr.max() - pr.min())
        enst_rng = float(pd.Series(z).groupby(enst_dec.values).mean().agg(lambda s: s.max() - s.min())) if enst_dec is not None else 0.0
        by_sp = {str(s): float(z[spl == s].mean()) for s in set(spl) if (spl == s).sum() > 50}
        by_cl = {str(s): float(z[cls == s].mean()) for s in set(cls) if (cls == s).sum() > 50}
        bad = max(pos_rng, enst_rng) >= 0.30
        sg = "SS_BINNED" if re.match(r"^prop_.*_bin$", fam) else fam
        info = dict(file=tag, all_zero_rate=float(rate), position_range=pos_rng, enst_range=enst_rng,
                    by_split=by_sp, by_class=by_cl, n_cols=len(cols), subgroup=sg)
        if bad:
            DEGENERATE[f"{tag}:{fam}"] = info
            n_bad += 1
        rec("FAIL" if bad else "WARN", stage, "zero_structure",
            f"{tag} family '{fam}*' ({len(cols)} cols): {100*rate:.1f}% rows all-zero; "
            f"rate range over file position {pos_rng:.2f}, over ENST order {enst_rng:.2f}; "
            f"by split {({k: round(v, 2) for k, v in by_sp.items()})}, by class {({k: round(v, 2) for k, v in by_cl.items()})}"
            + ("  -> missing values were written/filled as 0 for a contiguous part of the data" if bad else ""))
    if not n_bad:
        rec("PASS", stage, "zero_structure", f"{tag}: no column family is zero in a position/ID-ordered block")

    # proportion groups
    kinds = defaultdict(list)
    for col in X.columns:
        m = re.match(r"^(prop_.+)_bin\d+$", col)
        if m:
            kinds[m.group(1)].append(col)
    for k, cols in kinds.items():
        s = X[cols].sum(axis=1)
        zero, off = float((s < 1e-9).mean()), float(((s > 1e-9) & ((s - 1).abs() > 1e-3)).mean())
        lvl = "FAIL" if zero > 0.30 else ("WARN" if zero > 0.05 or off > 0.01 else "PASS")
        rec(lvl, stage, "proportions_sum", f"{tag} {k}_bin1..{len(cols)}: {100*zero:.1f}% rows sum to 0, {100*off:.2f}% sum to neither 0 nor 1")
    return X


def stage_features(c: Ctx):
    head("STAGE features")
    if not c.fa:
        c.load_fastas()
    ids_all = set(c.label)
    targets = dict(c.feat_csvs)
    if c.a.raw_nonb2:
        targets["raw_nonb2"] = c.a.raw_nonb2
    for lab, path in targets.items():
        if lab.endswith("_processed_csv") and not c.a.include_processed:
            continue
        if not Path(path).exists():
            rec("FAIL", "features", "exists", f"{lab}: {path} missing"); continue
        c.feat_csvs[lab] = path
        df = c.frame(lab)
        rec("INFO", "features", "shape", f"{lab}: {df.shape[0]:,} rows x {df.shape[1]} cols  ({path})")
        dup = int(df.index.duplicated().sum())
        is_clean = lab != "raw_nonb2"
        if dup:
            d = df[df.index.duplicated(keep=False)].apply(pd.to_numeric, errors="coerce").astype(float)
            g = d.groupby(level=0)
            rng = (g.max() - g.min()).abs()
            tol = 1e-6 * (1 + d.abs().groupby(level=0).max())
            diff = rng > tol
            conf = int(diff.any(axis=1).sum())
            cols = diff.sum().sort_values(ascending=False)
            cols = cols[cols > 0].head(4)
            detail = "; ".join(f"{k}: {int(v)} ids, max diff {rng[k].max():.3g}" for k, v in cols.items())
            lvl = "WARN" if (not is_clean or conf == 0) else "FAIL"
            rec(lvl, "features", "duplicate_ids",
                f"{lab}: {dup:,} duplicate rows; {conf:,} IDs differ beyond 1e-6 ({detail or 'none'})"
                + (" - clean step keeps the first row" if not is_clean else ""))
        else:
            rec("PASS", "features", "duplicate_ids", f"{lab}: IDs unique")
        if ids_all:
            have = set(df.index)
            for cl in ("lnc", "pc"):
                want = {i for i, l in c.label.items() if l == cl}
                miss = len(want - have)
                rec("FAIL" if miss / max(len(want), 1) > 0.005 else ("WARN" if miss else "PASS"),
                    "features", "coverage", f"{lab}: {miss:,}/{len(want):,} {cl} transcripts missing")
        num = df.apply(pd.to_numeric, errors="coerce")
        n_nan, n_inf = int(num.isna().sum().sum()), int(np.isinf(num.to_numpy(float)).sum())
        rec("FAIL" if (n_nan or n_inf) and is_clean else ("WARN" if n_nan or n_inf else "PASS"),
            "features", "nan_inf", f"{lab}: {n_nan:,} NaN, {n_inf:,} inf" + ("" if is_clean else " (raw file: NaN expected only where a format is absent)"))
        const = [col for col in num.columns if num[col].nunique(dropna=True) <= 1]
        rec("WARN" if const else "PASS", "features", "constant_cols", f"{lab}: {len(const)} constant columns {const[:5]}")
        # NaN structure in raw file
        if not is_clean and n_nan:
            fams = defaultdict(list)
            for col in num.columns:
                fams[family(col)].append(col)
            for fam, cols in fams.items():
                if len(cols) >= 3:
                    r = float(num[cols].isna().all(axis=1).mean())
                    if r > 0.01:
                        rec("FAIL", "features", "raw_nan_family", f"{lab}: {100*r:.1f}% rows have '{fam}*' entirely NaN (a later clean step would zero-fill)")
        family_stats(num.fillna(0) if not is_clean else num, c, lab)
        nm = Path(path).name.replace("_features_clean.csv", "_feature_names.txt")
        npth = Path(path).with_name(nm)
        if is_clean and nm != Path(path).name and npth.exists():
            names = npth.read_text().split("\n")
            same = names == list(df.columns)
            rec("PASS" if same else "FAIL", "features", "feature_names_order", f"{npth.name} matches CSV header order" if same else f"{npth.name} differs from CSV header")
        if is_clean:
            try:
                sys.path.insert(0, os.getcwd())
                from data.feature_registry import REGISTRY
                blk = "nonb2" if lab == "nonb2_csv" else lab.split("_")[0]
                exp = [m.name for m in REGISTRY.block_features(blk)]
                ok = exp == list(df.columns)
                rec("PASS" if ok else "FAIL", "features", "registry_order",
                    f"{lab}: columns {'match' if ok else 'DO NOT match'} registry ({len(exp)} expected, {df.shape[1]} found)")
            except Exception:
                rec("SKIP", "features", "registry_order", "feature_registry not importable from cwd")
        del num


# ------------------------------------------------------------------ stage: leakage
def stage_leakage(c: Ctx):
    head("STAGE leakage (feature-class shortcuts)")
    if not c.label:
        c.load_fastas()
    rng = np.random.default_rng(0)
    for lab in [k for k in c.feat_csvs if k != "raw_nonb2" and not k.endswith("_processed_csv")]:
        if not Path(c.feat_csvs[lab]).exists():
            continue
        df = c.frame(lab)
        df = df[~df.index.duplicated()]
        ids = [i for i in df.index if i in c.label]
        if len(ids) < 100:
            rec("SKIP", "leakage", lab, "too few labelled rows"); continue
        if len(ids) > 60000:
            ids = list(rng.choice(ids, 60000, replace=False))
        X = df.loc[ids].apply(pd.to_numeric, errors="coerce").fillna(0)
        y = np.array([c.label[i] == "lnc" for i in ids]).astype(int)
        ln = np.array([c.length[i] for i in ids], float)
        auc = auc_matrix(X, y)
        top = (auc - 0.5).abs().sort_values(ascending=False).head(3)
        worst = float(top.iloc[0]) + 0.5 if len(top) else np.nan
        rec("FAIL" if worst > 0.99 else ("WARN" if worst > 0.95 else "PASS"), "leakage", "single_feature_auc",
            f"{lab}: strongest single feature {top.index[0]} AUC {auc[top.index[0]]:.3f}")
        rho = X.rank().corrwith(pd.Series(ln, index=X.index).rank())
        hi = rho[rho.abs() > 0.8]
        rec("WARN" if len(hi) else "PASS", "leakage", "length_confound", f"{lab}: {len(hi)} features with |Spearman vs length| > 0.8 {list(hi.index[:4])}")
        rec("INFO", "leakage", "length_only_auc", f"{lab}: length alone gives AUC {auc_matrix(pd.DataFrame({'len': ln}), y)['len']:.3f} (reference)")
        fams = defaultdict(list)
        for col in X.columns:
            fams[family(col)].append(col)
        pres = {}
        for fam, cols in fams.items():
            if len(cols) >= 3:
                zi = (X[cols].abs().sum(axis=1) == 0).astype(float)
                if 0.02 < zi.mean() < 0.98:
                    pres[fam] = zi
        if pres:
            a = auc_matrix(pd.DataFrame(pres), y)
            for fam, v in a.items():
                lvl = "WARN" if abs(v - 0.5) > 0.10 else "INFO"
                rec(lvl, "leakage", "zero_pattern_auc", f"{lab}: 'is {fam}* all-zero' predicts class with AUC {v:.3f}")
        del X


# ------------------------------------------------------------------ stage: scalers
def stage_scalers(c: Ctx):
    head("STAGE scalers")
    if not c.trainval_ids:
        c.load_fastas()
    for b, (bp, csv) in c.banks.items():
        if not (Path(bp).exists() and Path(csv).exists()):
            rec("SKIP", "scalers", b, "bank or CSV missing"); continue
        mt_b, mt_c = os.path.getmtime(bp), os.path.getmtime(csv)
        rec("FAIL" if mt_b < mt_c - 1 else "PASS", "scalers", f"{b}_age",
            "bank is OLDER than its CSV (CSV rebuilt after the scaler was fit)" if mt_b < mt_c - 1 else "bank is not older than its CSV")
        raw = c.a.raw_nonb2
        if b == "nonb2" and raw and Path(raw).exists() and os.path.getmtime(raw) > mt_c + 1:
            rec("FAIL", "scalers", "nonb2_raw_age", "raw merged NonB2 CSV is newer than the clean CSV: clean step not rerun")
        try:
            sys.path.insert(0, os.getcwd())
            bank = pickle.load(open(bp, "rb"))
        except Exception as e:
            rec("SKIP", "scalers", f"{b}_refit", f"cannot unpickle bank (run from repo root): {type(e).__name__}"); continue
        sc, ci = getattr(bank, "_scalers", None), getattr(bank, "_cat_indices", None)
        if not sc or ci is None:
            rec("SKIP", "scalers", f"{b}_refit", "bank layout unknown"); continue
        df = c.frame(f"{b}_csv" if f"{b}_csv" in c.feat_csvs else next(k for k, v in c.feat_csvs.items() if v == csv))
        X = df[df.index.isin(c.trainval_ids)].to_numpy(np.float64)
        tot = sum(len(v) for v in ci.values())
        if tot != df.shape[1]:
            rec("FAIL", "scalers", f"{b}_columns", f"bank indexes {tot} columns, CSV has {df.shape[1]}"); continue
        bad, checked = [], 0
        names = np.array(df.columns)
        for cat, scaler in sc.items():
            if scaler is None:
                continue
            idx = np.array(ci[cat]); sub = X[:, idx]
            if hasattr(scaler, "center_") and hasattr(scaler, "scale_"):
                qr = getattr(scaler, "quantile_range", (25, 75))
                q = np.nanpercentile(sub, [qr[0], 50, qr[1]], axis=0)
                s = q[2] - q[0]; s[s == 0] = 1.0
                e1, e2, g1, g2 = q[1], s, scaler.center_, scaler.scale_
            elif hasattr(scaler, "data_min_"):
                e1, e2, g1, g2 = sub.min(0), sub.max(0), scaler.data_min_, scaler.data_max_
            else:
                continue
            ok = np.isclose(e1, g1, rtol=1e-4, atol=1e-6) & np.isclose(e2, g2, rtol=1e-4, atol=1e-6)
            checked += len(idx); bad += list(names[idx[~ok]])
        if checked == 0:
            rec("SKIP", "scalers", f"{b}_refit", "no robust/minmax scalers to check"); continue
        rec("FAIL" if bad else "PASS", "scalers", f"{b}_refit",
            f"{len(bad)}/{checked} scaled columns differ from a re-fit on the trainval rows of {Path(csv).name}"
            + (f" e.g. {bad[:4]} -> bank was fit on different data/rows" if bad else ""))
        # scaled output sanity
        try:
            Z = bank.transform(df.to_numpy(np.float64)[:50000])
            big = float((np.abs(Z) > 50).mean())
            rec("WARN" if big > 0.001 else "PASS", "scalers", f"{b}_extremes", f"{100*big:.3f}% of scaled values have |z| > 50")
        except Exception:
            pass


# ------------------------------------------------------------------ stage: model
def stage_model(c: Ctx):
    head("STAGE model")
    ck = sorted(glob.glob(str(c.exp_dir / "models" / "fold_*_best.pt")))
    if not ck:
        rec("SKIP", "model", "checkpoints", f"no checkpoints under {c.exp_dir}/models (not trained yet)"); return
    rec("PASS" if len(ck) == c.n_folds else "FAIL", "model", "n_checkpoints", f"{len(ck)} checkpoints, config expects {c.n_folds}")
    newest_input = 0.0
    for k in ("nonb2_csv", "te_genomic_csv", "nonb_genomic_csv", "te_scaler_bank_path", "nonb_scaler_bank_path", "nonb2_scaler_bank_path"):
        p = c.d.get(k)
        if p and Path(p).exists():
            newest_input = max(newest_input, os.path.getmtime(p))
    oldest_ck = min(os.path.getmtime(p) for p in ck)
    rec("FAIL" if oldest_ck < newest_input else "PASS", "model", "trained_after_features",
        "a checkpoint is OLDER than the feature CSVs/scalers it should have been trained on" if oldest_ck < newest_input
        else "all checkpoints are newer than every feature CSV and scaler bank")
    try:
        import torch
        bad = 0
        for p in ck:
            obj = torch.load(p, map_location="cpu", weights_only=False)
            sd = obj.get("model_state_dict", obj.get("state_dict", obj)) if isinstance(obj, dict) else obj
            if isinstance(sd, dict):
                for v in sd.values():
                    if hasattr(v, "isfinite") and v.dtype.is_floating_point and not bool(v.isfinite().all()):
                        bad += 1; break
            if isinstance(obj, dict) and isinstance(obj.get("config"), dict):
                cd = c.remap(obj["config"].get("data", {}))
                diff = [k for k in ("nonb2_csv", "lnc_fasta", "nonb2_scaler_bank_path") if k in cd and cd[k] != c.d.get(k)]
                if diff:
                    rec("FAIL", "model", "ckpt_config", f"{Path(p).name} was trained with different {diff}")
        rec("FAIL" if bad else "PASS", "model", "finite_weights", f"{bad} checkpoints contain NaN/inf")
    except Exception as e:
        rec("SKIP", "model", "ckpt_load", f"torch load skipped: {type(e).__name__}")

    tp = c.exp_dir / "evaluation_csvs" / "test_predictions.csv"
    tr = next(iter(glob.glob(str(c.exp_dir / "**" / "test_results.json"), recursive=True)), None)
    if tp.exists():
        pr = pd.read_csv(tp)
        pr["tid"] = pr["transcript_id"].map(norm_id)
        test_ids = {i for i, s in c.split.items() if s == "test"}
        if test_ids:
            cov = len(set(pr.tid) & test_ids) / len(test_ids)
            extra = len(set(pr.tid) - test_ids)
            rec("FAIL" if cov < 0.999 or extra else "PASS", "model", "pred_ids", f"test_predictions covers {100*cov:.2f}% of the test FASTA; {extra} IDs not in it")
        n_err = int((pr["true_label"] != pr["consensus_prediction"]).sum())
        rec("INFO", "model", "pred_errors", f"{n_err:,} errors of {len(pr):,} (accuracy {1-n_err/len(pr):.6f})")
        rec("FAIL" if tp.stat().st_mtime < max(os.path.getmtime(p) for p in ck) else "PASS", "model", "pred_age",
            "test_predictions.csv is OLDER than the newest checkpoint (stale)" if tp.stat().st_mtime < max(os.path.getmtime(p) for p in ck) else "test_predictions.csv is newer than all checkpoints")
        if tr:
            acc = find_key(json.load(open(tr)), {"accuracy", "test_accuracy", "ensemble_accuracy"})
            if acc is not None:
                same = abs(acc - (1 - n_err / len(pr))) < 1e-6
                rec("PASS" if same else "FAIL", "model", "pred_vs_results", f"test_results.json accuracy {acc:.6f} vs recomputed {1-n_err/len(pr):.6f}")
    else:
        rec("SKIP", "model", "test_predictions", f"{tp} not found")
    if c.a.benchmark_dir:
        bm = Path(c.a.benchmark_dir) / "predictions_merged.csv"
        if bm.exists() and tp.exists():
            m = pd.read_csv(bm)
            m["tid"] = m["transcript_id"].map(norm_id)
            j = m.merge(pr[["tid", "true_label", "consensus_prediction"]], on="tid", how="inner")
            pred_err = (j["true_label"] != j["consensus_prediction"])
            mism = int((pred_err != j["blnc_hard"].astype(bool)).sum())
            rec("FAIL" if mism else "PASS", "model", "benchmark_vs_predictions",
                f"{mism:,}/{len(j):,} transcripts where benchmark blnc_hard disagrees with test_predictions errors"
                + (" -> benchmark folder belongs to a different model run" if mism else ""))
            rec("INFO", "model", "benchmark_age", f"benchmark newer than checkpoints: {bm.stat().st_mtime >= max(os.path.getmtime(p) for p in ck)}")
        else:
            rec("SKIP", "model", "benchmark_vs_predictions", "predictions_merged.csv or test_predictions.csv missing")


# ------------------------------------------------------------------ stage: layers
def stage_layers(c: Ctx):
    head("STAGE layers")
    d = c.exp_dir / "interpretability"
    if not d.exists():
        rec("SKIP", "layers", "dir", f"{d} not found (layers not run yet)"); return
    ck = sorted(glob.glob(str(c.exp_dir / "models" / "fold_*_best.pt")))
    mans = sorted(glob.glob(str(d / "run_manifest_*.json")))
    if not mans:
        rec("FAIL", "layers", "manifest", "no run_manifest_*.json: results cannot be tied to checkpoints")
    else:
        m = c.remap(json.load(open(mans[-1])))
        stale = [Path(p).name for p in ck if m.get("checkpoints", {}).get(p) != sha16(p) and m.get("checkpoints", {}).get(str(Path(p))) != sha16(p)]
        rec("FAIL" if stale else "PASS", "layers", "manifest_checkpoints",
            f"{len(stale)} checkpoints differ from those the layers ran on {stale[:3]}" if stale else f"{Path(mans[-1]).name}: all {len(ck)} checkpoint hashes match")
        if Path(m.get("config", "")).exists():
            rec("PASS" if sha16(m["config"]) == m.get("config_sha") else "FAIL", "layers", "manifest_config", "config unchanged since the layers ran" if sha16(m["config"]) == m.get("config_sha") else "config changed after layers ran")
        elif m.get("config"):
            rec("WARN", "layers", "manifest_config", f"config named in the manifest not found: {m['config']} (renamed folder? add --path_map OLD=NEW)")
        if m.get("git_dirty"):
            rec("WARN", "layers", "git_dirty", "layers ran on an uncommitted working tree")
    newest_ck = max((os.path.getmtime(p) for p in ck), default=0)
    csvs = glob.glob(str(d / "**" / "*.csv"), recursive=True)
    old, nonfin, iia_bad = [], [], []
    for p in csvs:
        if os.path.getmtime(p) < newest_ck:
            old.append(Path(p).name)
        try:
            df = pd.read_csv(p, nrows=200000)
        except Exception:
            continue
        num = df.select_dtypes("number")
        if "group" in df.columns:             # unmatched baselines have no pairs: pair-difference means are undefined
            um = df["group"].astype(str).str.contains("unmatched").to_numpy()[:, None]
            pc = np.array([c_.startswith("mean_abs_") for c_ in num.columns])[None, :]
            num = num.mask(pd.DataFrame(um & pc, index=num.index, columns=num.columns), 0.0)
        bad_cols = [c_ for c_ in num.columns if not np.isfinite(num[c_].to_numpy(float)).all()]
        if bad_cols:
            nonfin.append(f"{Path(p).name}:" + ",".join(
                f"{c_}({int((~np.isfinite(num[c_].to_numpy(float))).sum())})" for c_ in bad_cols[:3]))
        for col in num.columns:
            cl = col.lower()
            if "iia" not in cl or cl.startswith(("n_", "num_")) or cl.endswith(("_count", "_n")):
                continue                      # counts are not rates
            # a difference of two IIA values (excess over noise, delta, ...) lies in [-1, 1]; IIA itself in [0, 1]
            lo = -1.0 if any(w in cl for w in ("excess", "delta", "diff", "over_noise", "minus")) else 0.0
            if ((num[col] < lo - 1e-9) | (num[col] > 1 + 1e-9)).any():
                iia_bad.append(f"{Path(p).name}:{col}")
    rec("INFO", "layers", "n_csv", f"{len(csvs)} result CSVs")
    rec("FAIL" if old else "PASS", "layers", "results_age", f"{len(old)} result CSVs older than the newest checkpoint {old[:3]}" if old else "all result CSVs are newer than the checkpoints")
    rec("FAIL" if nonfin else "PASS", "layers", "finite", f"non-finite numbers in {nonfin[:4]} (column(count))" if nonfin else "all numeric results finite")
    rec("FAIL" if iia_bad else "PASS", "layers", "iia_range", f"IIA outside its range ([0,1]; [-1,1] for differences): {iia_bad[:3]}" if iia_bad else "IIA columns within range ([0,1]; [-1,1] for differences)")
    if DEGENERATE:
        subs = {v["subgroup"] for v in DEGENERATE.values()}
        hit = defaultdict(list)
        for p in csvs + glob.glob(str(d / "**" / "*.txt"), recursive=True):
            try:
                txt = open(p, errors="ignore").read(5_000_000)
            except Exception:
                continue
            for s in subs:
                if s in txt:
                    hit[s].append(Path(p).name)
        for s in subs:
            rec("WARN", "layers", "degenerate_subgroup",
                f"subgroup {s} has structurally zero inputs (see features stage); {len(hit[s])} result files mention it: do not interpret biologically {hit[s][:4]}")
    else:
        rec("INFO", "layers", "degenerate_subgroup", "no degenerate subgroup recorded (run the features stage in the same call to cross-check)")


def report_path_map(c: Ctx) -> None:
    for old, new in c.path_map:
        n = c.path_map_hits[old]
        rec("INFO" if n else "WARN", "config", "path_map",
            f"{old} -> {new}: {n} strings rewritten" if n else f"{old} -> {new}: matched no path (typo?)")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp_root", default=".", help="folder that contains output.experiment_name")
    ap.add_argument("--benchmark_dir", default=None)
    ap.add_argument("--raw_nonb2", default=None, help="merged (pre-clean) NonB2 CSV from merge_nonb2.py")
    ap.add_argument("--include_processed", action="store_true", help="also audit *_processed_csv files")
    ap.add_argument("--stages", default=",".join(ALL_STAGES))
    ap.add_argument("--out", default="pipeline_audit")
    ap.add_argument("--fail_on_warn", action="store_true")
    ap.add_argument("--path_map", action="append", default=[], metavar="OLD=NEW",
                    help="rewrite OLD to NEW in every path read from the config, checkpoint "
                         "configs and the layers manifest (repeatable)")
    a = ap.parse_args()
    c = Ctx(a)
    c.load_fastas()
    fns = {"config": stage_config, "split": stage_split, "features": stage_features, "leakage": stage_leakage,
           "scalers": stage_scalers, "model": stage_model, "layers": stage_layers}
    want = [s for s in a.stages.split(",") if s]
    stage_config(c)   # always: registers CSVs and banks
    for s in ALL_STAGES:
        if s in want and s != "config":
            try:
                fns[s](c)
            except Exception as e:
                rec("FAIL", s, "crashed", f"{type(e).__name__}: {e}")
    report_path_map(c)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps({"config": a.config, "path_map": c.path_map,
                                                "results": RES, "degenerate": DEGENERATE}, indent=2))
    n = defaultdict(int)
    for r in RES:
        n[r["level"]] += 1
    with open(out / "audit.md", "w") as f:
        f.write(f"# Pipeline audit: {a.config}\n\nFAIL {n['FAIL']} | WARN {n['WARN']} | PASS {n['PASS']} | SKIP {n['SKIP']}\n\n")
        for lvl in ("FAIL", "WARN"):
            for r in RES:
                if r["level"] == lvl:
                    f.write(f"- **{lvl}** `{r['stage']}/{r['check']}`: {r['msg']}\n")
    head(f"SUMMARY  FAIL {n['FAIL']} | WARN {n['WARN']} | PASS {n['PASS']} | SKIP {n['SKIP']}   -> {out}/audit.md")
    for r in RES:
        if r["level"] == "FAIL":
            print(f"  FAIL {r['stage']}/{r['check']}: {r['msg'][:200]}")
    sys.exit(1 if n["FAIL"] or (a.fail_on_warn and n["WARN"]) else 0)


if __name__ == "__main__":
    main()