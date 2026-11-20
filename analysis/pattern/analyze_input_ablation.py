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

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset

from data.cv_utils import get_cv_splits_from_config
from data.feature_registry import REGISTRY

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

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "all_folds_input_ablation.csv", index=False)
    s = (df.groupby(["mode", "subgroup"])["acc_drop"].agg(["mean", "std"])
         .sort_values("mean", ascending=False))
    print(s.to_string())


if __name__ == "__main__":
    main()