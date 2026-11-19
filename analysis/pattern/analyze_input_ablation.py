#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_input_ablation.py

Architecture-independent sub-group ablation at the INPUT level.
Works for BetaVAESubgroup and feature-only models (no projector hooks).

Modes (applied to the sub-group's feature columns of the val set)
  zero     : set columns to 0
  mean     : set columns to the val-set column mean
  permute  : shuffle the group's rows jointly across samples (keeps the
             marginal and within-group correlations, e.g. bins still sum to 1;
             breaks the link to the label). Averaged over --n_perm seeds.

Metrics: accuracy and macro-F1 (same definition as the benchmark tables).
`f1_drop` is the macro-F1 drop. Mean replacement uses the val-fold column mean.
Cross-fold stars are descriptive: the 5 folds share training data.
Checkpoints must be numbered from 0 (fold_0_best.pt ...).

Usage (repo root)
  python -m analysis.pattern.analyze_input_ablation \
    --experiment_dir EXP --config CFG --output_dir EXP/interpretability/1.2b_input_ablation \
    [--verify]   # full model only: zero must reproduce feature_zero of
                 # analyze_subgroup_ablation.py
"""
import argparse
import inspect
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset

from data.cv_utils import get_cv_splits_from_config
from data.feature_registry import REGISTRY, display_subgroups

MODE_COLORS = {'zero': '#4A90D9', 'mean': '#F4B942', 'permute': '#9B7FD4'}

FEATURE_KEYS = ("te_genomic", "te_processed", "nonb_genomic",
                "nonb_processed", "nonb2")
BLOCK_KEYS = {"te": ("te_genomic", "te_processed"),
              "nonb": ("nonb_genomic", "nonb_processed"),
              "nonb2": ("nonb2",)}


def keys_for(block):
    keys = BLOCK_KEYS[block]
    if len(keys) == 1:
        return keys
    src = REGISTRY.block_source(block)
    return {"both": keys, "genomic": keys[:1], "processed": keys[1:]}[src]


def collect_features(loader):
    """One pass: stack feature tensors (cheap) and labels for the val set."""
    feats = {k: [] for k in FEATURE_KEYS}
    for b in loader:
        for k in FEATURE_KEYS:
            if k in b:
                feats[k].append(b[k].float())
    return {k: torch.cat(v) for k, v in feats.items() if v}


def make_edit(feats, sg, mode, perm=None):
    """Return {key: (idx, replacement_full_rows)} for the group's columns."""
    block = REGISTRY.token_block(sg)
    idx = list(REGISTRY.indices_for_subgroup(sg))
    edit = {}
    for k in keys_for(block):
        if k not in feats:
            continue
        col = feats[k][:, idx]
        if mode == "zero":
            rep = torch.zeros_like(col)
        elif mode == "mean":
            rep = col.mean(0, keepdim=True).expand_as(col).clone()
        else:
            rep = col[perm]
        edit[k] = (idx, rep)
    return edit


@torch.no_grad()
def run(model, loader, device, edit=None):
    model.eval()
    params = inspect.signature(model.forward).parameters
    preds, labels, off = [], [], 0
    for b in loader:
        n = len(b["label"])
        kw = {}
        for k in FEATURE_KEYS:
            if k in b:
                x = b[k].float()
                if edit and k in edit:
                    idx, rep = edit[k]
                    x = x.clone()
                    x[:, idx] = rep[off:off + n]
                kw[k] = x.to(device)
        # Sequence goes positionally if forward's first parameter is not a
        # feature key (e.g. `x`), matching analyze_subgroup_ablation.py.
        names = list(params)
        args = []
        if names and names[0] not in FEATURE_KEYS and names[0] != "deterministic":
            args.append(b["sequence"].to(device))
            names = names[1:]
        if "deterministic" in params:
            kw["deterministic"] = True
        var_kw = any(q.kind == q.VAR_KEYWORD for q in params.values())
        kw = {k: v for k, v in kw.items() if var_kw or k in names}
        out = model(*args, **kw)
        logits = out["logits"] if isinstance(out, dict) else out
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(b["label"].numpy())
        off += n
    return np.concatenate(preds), np.concatenate(labels)


def metrics(p, y):
    return accuracy_score(y, p), f1_score(y, p, average="macro", zero_division=0)


def plot_fold(df, path, tag=''):
    """Grouped bars of acc_drop per sub-group, one bar per mode."""
    modes = [m for m in MODE_COLORS if m in set(df['mode'])]
    piv = df.pivot(index='subgroup', columns='mode', values='acc_drop')
    order = piv.mean(axis=1).sort_values(ascending=False).index
    x, w = np.arange(len(order)), 0.8 / max(len(modes), 1)
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, m in enumerate(modes):
        ax.bar(x + (i - (len(modes) - 1) / 2) * w, piv.loc[order, m], w,
               label=m, color=MODE_COLORS[m], edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(display_subgroups(list(order)), rotation=45, ha='right')
    ax.set_ylabel('Accuracy drop (baseline - ablated)')
    ax.set_title(f'{tag}Input ablation per fold')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout(); plt.savefig(path, dpi=350, bbox_inches='tight'); plt.close()


def plot_cross_fold(df, out_dir, tag=''):
    """Per mode: mean +/- sd over folds, BH-FDR one-sample t-test vs 0."""
    for mode in [m for m in MODE_COLORS if m in set(df['mode'])]:
        piv = df[df['mode'] == mode].pivot(index='subgroup', columns='fold',
                                           values='acc_drop')
        mean = piv.mean(axis=1).sort_values(ascending=False)
        std = piv.std(axis=1).reindex(mean.index)
        p = piv.reindex(mean.index).apply(
            lambda r: stats.ttest_1samp(r.dropna(), 0.0).pvalue
            if r.dropna().size > 1 else np.nan, axis=1)
        q = pd.Series(np.nan, index=p.index)
        ok = p.notna()
        if ok.any():
            q[ok] = stats.false_discovery_control(p[ok].values, method='bh')
        sig = q < 0.05
        fig, ax = plt.subplots(figsize=(13, 5))
        x = np.arange(len(mean))
        ax.bar(x, mean.values, color=MODE_COLORS[mode], edgecolor='black',
               linewidth=0.6)
        ax.errorbar(x, mean.values, yerr=std.values, fmt='none', color='black',
                    capsize=3)
        yr = mean.max() - mean.min()
        for xi, m, s_, f in zip(x, mean.values, std.values, sig.values):
            if f:
                ax.annotate('*', (xi, max(m, 0) + s_ + 0.06 * yr), ha='center',
                            fontsize=14, fontweight='bold')
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.08 * yr)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(display_subgroups(list(mean.index)), rotation=45,
                           ha='right')
        ax.set_ylabel('Mean accuracy drop (mean +/- std over folds)')
        ax.set_title(f'{tag}Cross-fold input ablation - {mode} '
                     f'(* q<0.05, BH-FDR; descriptive, folds not independent)')
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.savefig(out_dir / f'cross_fold_input_ablation_{mode}.png',
                    dpi=350, bbox_inches='tight'); plt.close()
        pd.DataFrame({'subgroup': mean.index, 'mean_acc_drop': mean.values,
                      'std_acc_drop': std.values, 'p_value': p.values,
                      'q_value_bh': q.values, 'significant_fdr': sig.values}
                     ).to_csv(out_dir / f'cross_fold_input_ablation_{mode}_stats.csv',
                              index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--modes", default="zero,mean,permute")
    ap.add_argument("--n_perm", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--device",
                    default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    from configs.load_config import load_config
    from models.model_builder import create_model_builder
    from data.cv_utils import load_sequences_in_order
    from data.gated_feature_dataset import SequenceFeatureDataset

    cfg = load_config(a.config)
    g = lambda k, d=None: cfg.get("data", k, default=d)
    dataset = SequenceFeatureDataset(
        lnc_fasta=cfg.get("data", "lnc_fasta"),
        pc_fasta=cfg.get("data", "pc_fasta"),
        te_genomic_csv=g("te_genomic_csv"),
        te_processed_csv=g("te_processed_csv"),
        nonb_genomic_csv=g("nonb_genomic_csv"),
        nonb_processed_csv=g("nonb_processed_csv"),
        nonb2_csv=g("nonb2_csv"),
        te_scaler_bank_path=g("te_scaler_bank_path"),
        nonb_scaler_bank_path=g("nonb_scaler_bank_path"),
        nonb2_scaler_bank_path=g("nonb2_scaler_bank_path"),
        max_length=cfg.get("model", "max_length"))
    seqs, labels = load_sequences_in_order(
        lnc_fasta=cfg.get("data", "lnc_fasta"),
        pc_fasta=cfg.get("data", "pc_fasta"))
    splits = get_cv_splits_from_config(cfg, seqs, labels)
    builder = create_model_builder(cfg)
    device = torch.device(a.device)
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = a.modes.split(",")
    rows = []

    ckpts = sorted((Path(a.experiment_dir) / "models").glob("fold_*_best.pt"))
    folds = [int(c.stem.split("_")[1]) for c in ckpts]
    if not ckpts or min(folds) != 0 or max(folds) >= len(splits):
        raise SystemExit(f"checkpoints {folds} do not match the {len(splits)} CV splits: "
                         "expected fold_0_best.pt ... (0-based); refusing to evaluate on the wrong folds")

    for ck in ckpts:
        fold = int(ck.stem.split("_")[1])
        model = builder()
        model.load_state_dict(torch.load(ck, map_location=device,
                                         weights_only=False)["model_state_dict"])
        model.to(device).eval()
        _, val_idx = splits[fold]
        loader = DataLoader(Subset(dataset, val_idx), batch_size=a.batch_size,
                            shuffle=False, num_workers=1)
        feats = collect_features(loader)
        n = len(val_idx)
        bp, by = run(model, loader, device)
        bacc, bf1 = metrics(bp, by)
        print(f"fold {fold}: n={n} baseline acc={bacc:.4f} f1={bf1:.4f}")

        for sg in REGISTRY.all_subgroups:
            block = REGISTRY.token_block(sg)
            for mode in modes:
                reps = a.n_perm if mode == "permute" else 1
                accs, f1s = [], []
                for r in range(reps):
                    perm = (torch.from_numpy(
                        np.random.default_rng(1000 * fold + r).permutation(n))
                        if mode == "permute" else None)
                    p, y = run(model, loader, device,
                               make_edit(feats, sg, mode, perm))
                    acc, f1 = metrics(p, y)
                    accs.append(acc); f1s.append(f1)
                rows.append(dict(fold=fold, subgroup=sg, block=block, mode=mode,
                                 baseline_acc=bacc, ablated_acc=np.mean(accs),
                                 acc_drop=bacc - np.mean(accs),
                                 f1_drop=bf1 - np.mean(f1s),
                                 acc_drop_sd_perm=float(np.std(accs)) if reps > 1 else 0.0))
                print(f"  {sg:14s} {mode:8s} Δacc={bacc-np.mean(accs):+.4f}")

        if a.verify and hasattr(model, "projectors"):
            from analysis.pattern.analyze_subgroup_ablation import (
                SubgroupAblationHook, run_inference)
            for sg in REGISTRY.all_subgroups[:3]:
                hp, hl, _ = run_inference(model, loader, device,
                                          SubgroupAblationHook(model, sg, "feature_zero"))
                ref = accuracy_score(hl, hp)
                mine = [r["ablated_acc"] for r in rows if r["fold"] == fold
                        and r["subgroup"] == sg and r["mode"] == "zero"][0]
                print(f"VERIFY {sg}: hook={ref:.6f} input={mine:.6f} "
                      f"{'OK' if abs(ref-mine) < 1e-9 else 'MISMATCH'}")
            a.verify = False

        pd.DataFrame(rows).to_csv(out_dir / "all_folds_input_ablation.csv", index=False)
        fd = out_dir / f"fold_{fold}"; fd.mkdir(exist_ok=True)
        plot_fold(pd.DataFrame([r for r in rows if r["fold"] == fold]),
                  fd / "input_ablation_bar.png", tag=f"fold {fold} - ")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "all_folds_input_ablation.csv", index=False)
    if df["fold"].nunique() > 1:
        plot_cross_fold(df, out_dir)
    s = (df.groupby(["mode", "subgroup"])["acc_drop"].agg(["mean", "std"])
         .sort_values("mean", ascending=False))
    print(s.to_string())


if __name__ == "__main__":
    main()