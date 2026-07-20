#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full 5-fold cross-validation training for FeatureOnlyClassifier.

Sequence-free ablation of BetaVAESubgroup — tests whether subgroup
dominance in the full model is caused by sequence-feature synergy or by
an intrinsic feature routing bias in the cross-modal attention mechanism.

Distinct from main_subgroup.py
--------------------------------
FeatureOnlyClassifier has no VAE (no encoder, no decoder, no z sampling),
so it needs a lighter trainer loop: no KL term, no reconstruction loss, no
KL annealing schedule, no adversarial length head. Previously this was run
through main_subgroup.py / BetaVAESubgroupTrainer with alpha=0, beta=0,
lambda_recon=0 to neutralise the VAE-specific loss terms — that works
because FeatureOnlyClassifier returns placeholder reconstruction/mu/logvar
tensors, but it's fragile and obscures which architecture is actually being
trained.

Usage
-----
python src/main_feature_only.py \
    --config  configs/beta_vae_feature_only_base_g49.json \
    --device  cuda:0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from configs.load_config import load_config
from models.model_builder import create_model_builder
from data.cv_utils import load_sequences_in_order, create_length_stratified_groups
from data.gated_feature_dataset import SequenceFeatureDataset
from data.feature_registry import REGISTRY


# ---------------------------------------------------------------------------
# Forward-call helper (same convention as beta_vae_subgroup_trainer.py)
# ---------------------------------------------------------------------------

def _forward_kwargs(batch: dict, device: torch.device) -> dict:
    kwargs = dict(
        te_genomic     = batch["te_genomic"].to(device),
        te_processed   = batch["te_processed"].to(device),
        nonb_genomic   = batch["nonb_genomic"].to(device),
        nonb_processed = batch["nonb_processed"].to(device),
    )
    if "nonb2" in batch:
        kwargs["nonb2"] = batch["nonb2"].to(device)
    return kwargs


# ---------------------------------------------------------------------------
# Loss — classification only, no VAE terms
# ---------------------------------------------------------------------------

class FeatureOnlyLoss:
    """
    Classification loss for FeatureOnlyClassifier.

    Main classification loss + per-block auxiliary classification losses,
    weighted by lambda_feat (dict, one entry per registry block).
    No reconstruction, no KL — there is no VAE in this architecture.
    """

    def __init__(
        self,
        class_weights: torch.Tensor = None,
        lambda_feat:   dict = None,
    ) -> None:
        self.class_weights = class_weights
        self.lambda_feat: dict = {b: 0.0 for b in REGISTRY.block_names}
        if lambda_feat:
            self.lambda_feat.update(lambda_feat)

    def _cls(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return (
            F.cross_entropy(logits, labels, weight=self.class_weights)
            if self.class_weights is not None
            else F.cross_entropy(logits, labels)
        )

    def __call__(self, logits, labels, feat_logits: dict = None):
        cls   = self._cls(logits, labels)
        total = cls

        feat_losses = {}
        if feat_logits is not None:
            for block_name, block_logits in feat_logits.items():
                if block_logits is None:
                    feat_losses[block_name] = None
                    continue
                loss_val = self._cls(block_logits, labels)
                feat_losses[block_name] = loss_val
                w = self.lambda_feat.get(block_name, 0.0)
                if w > 0:
                    total = total + w * loss_val

        loss_dict = {"loss": total.item(), "classification": cls.item()}
        for block_name, loss_val in feat_losses.items():
            loss_dict[f"feat_{block_name}"] = (
                loss_val.item() if loss_val is not None else None
            )
        return total, loss_dict


# ---------------------------------------------------------------------------
# Single-fold training loop
# ---------------------------------------------------------------------------

def train_one_fold(
    model, train_loader, val_loader, config, device, class_weights, save_path,
):
    model.to(device)

    lambda_feat = {
        b: config.get("training", f"lambda_feat_{b}", default=0.0)
        for b in REGISTRY.block_names
    }
    cw = (torch.FloatTensor(class_weights).to(device)
          if class_weights is not None else None)
    criterion = FeatureOnlyLoss(class_weights=cw, lambda_feat=lambda_feat)

    proj_params  = [p for n, p in model.named_parameters()
                    if n.startswith("projectors.") and "cross_attn" not in n]
    other_params = [p for n, p in model.named_parameters()
                    if "cross_attn" not in n and not n.startswith("projectors.")]
    attn_params  = [p for n, p in model.named_parameters() if "cross_attn" in n]

    lr = config.get("training", "learning_rate")
    optimizer = torch.optim.Adam([
        {"params": other_params, "lr": lr},
        {"params": proj_params,  "lr": lr * 2},
        {"params": attn_params,  "lr": lr},
    ], weight_decay=config.get("training", "weight_decay"))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7
    )

    num_epochs = config.get("training", "num_epochs")
    patience   = config.get("training", "early_stopping_patience")

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    for b in REGISTRY.block_names:
        history[f"train_feat_{b}"] = []

    best_val_loss    = float("inf")
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        totals = {"loss": 0.0, "classification": 0.0}
        for b in REGISTRY.block_names:
            totals[f"feat_{b}"] = 0.0
        counts = {k: 0 for k in totals}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for batch in pbar:
            labels = batch["label"].to(device)
            fwd_kw = _forward_kwargs(batch, device)

            optimizer.zero_grad()
            out = model(**fwd_kw)

            feat_logits = {b: out.get(f"feat_logits_{b}") for b in REGISTRY.block_names}
            loss, loss_dict = criterion(out["logits"], labels, feat_logits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in totals:
                v = loss_dict.get(k)
                if v is not None:
                    totals[k] += v
                    counts[k] += 1
            pbar.set_postfix({k: f"{v:.4f}" for k, v in loss_dict.items() if v is not None})

        train_m = {k: (totals[k] / counts[k] if counts[k] > 0 else None) for k in totals}

        # ── Validation ──
        model.eval()
        all_logits, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                labels = batch["label"].to(device)
                fwd_kw = _forward_kwargs(batch, device)
                out    = model(**fwd_kw)
                val_loss += F.cross_entropy(out["logits"], labels).item()
                all_logits.append(out["logits"].cpu())
                all_labels.append(labels.cpu())

        val_loss /= len(val_loader)
        preds = torch.cat(all_logits).argmax(dim=1)
        val_acc = accuracy_score(torch.cat(all_labels).numpy(), preds.numpy())

        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        for b in REGISTRY.block_names:
            v = train_m.get(f"feat_{b}")
            if v is not None:
                history[f"train_feat_{b}"].append(v)

        aux_str = " ".join(
            f"f_{b}={train_m[f'feat_{b}']:.3f}" for b in REGISTRY.block_names
            if train_m.get(f"feat_{b}") is not None
        )
        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"Train: {train_m['loss']:.4f} (cls={train_m['classification']:.4f} "
              f"{aux_str}) | Val: {val_loss:.4f} acc={val_acc:.4f}")

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != old_lr:
            print(f"  → LR: {old_lr:.2e} → {new_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch, "val_loss": best_val_loss, "val_acc": val_acc,
                "history": history,
            }, save_path)
            print(f"  → Saved checkpoint → {save_path}")
        else:
            patience_counter += 1
            print(f"  → Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    return model, {
        "val_loss": best_val_loss,
        "val_acc": max(history["val_acc"]),
        "train_loss": min(history["train_loss"]),
    }, history


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------

def _build_dataset(config, lnc_fasta, pc_fasta):
    nonb2_csv    = config.get("data", "nonb2_csv",              default=None)
    nonb2_scaler = config.get("data", "nonb2_scaler_bank_path", default=None)
    if nonb2_csv is not None and nonb2_scaler is None:
        raise ValueError(
            "Config has data.nonb2_csv set but no data.nonb2_scaler_bank_path."
        )
    return SequenceFeatureDataset(
        lnc_fasta              = lnc_fasta,
        pc_fasta                = pc_fasta,
        te_genomic_csv          = config.get("data", "te_genomic_csv"),
        te_processed_csv        = config.get("data", "te_processed_csv",   default=None),
        nonb_genomic_csv        = config.get("data", "nonb_genomic_csv"),
        nonb_processed_csv      = config.get("data", "nonb_processed_csv", default=None),
        nonb2_csv                = nonb2_csv,
        te_scaler_bank_path     = config.get("data", "te_scaler_bank_path"),
        nonb_scaler_bank_path   = config.get("data", "nonb_scaler_bank_path"),
        nonb2_scaler_bank_path  = nonb2_scaler,
        max_length               = config.get("model", "max_length"),
    )


# ---------------------------------------------------------------------------
# Main — 5-fold CV
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="5-fold CV training for FeatureOnlyClassifier"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available()
                                             else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config.get("output", "experiment_name"))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    arch = config.get("model", "architecture", default="feature_only")
    if arch != "feature_only":
        raise ValueError(
            f"main_feature_only.py requires model.architecture='feature_only', "
            f"got '{arch}'. Use main_subgroup.py for beta_vae_subgroup configs."
        )

    print("=" * 80)
    print("FeatureOnlyClassifier Training")
    print("=" * 80)
    print(f"d_proj     : {config.get('model', 'd_proj', default=64)}")
    print(f"attn_heads : {config.get('model', 'attn_heads', default=4)}")
    print(f"Device     : {device}")
    print(f"N folds    : {config.get('training', 'n_folds')}")
    print("=" * 80)
    print("Feature blocks (from registry):")
    for block_name in REGISTRY.block_names:
        dim    = REGISTRY.block_dim(block_name)
        source = REGISTRY.block_source(block_name)
        n_sg   = len(REGISTRY.block_subgroups(block_name))
        lf     = config.get("training", f"lambda_feat_{block_name}", default=0.0)
        print(f"  {block_name:8s}: {dim:4d} features  ({n_sg} subgroups, "
              f"source={source}, lambda_feat={lf})")
    print(f"  TOTAL TOKENS: {REGISTRY.total_tokens}")
    print("=" * 80)

    all_sequences, labels = load_sequences_in_order(
        lnc_fasta=config.get("data", "lnc_fasta"),
        pc_fasta= config.get("data", "pc_fasta"),
    )

    dataset = _build_dataset(
        config,
        lnc_fasta=config.get("data", "lnc_fasta"),
        pc_fasta=config.get("data", "pc_fasta"),
    )
    print(f"\nDataset: {len(dataset):,} samples")

    model_builder = create_model_builder(config)
    _model = model_builder()
    total_params = sum(p.numel() for p in _model.parameters())
    print(f"\nModel: {total_params:,} params")
    print(f"n_tokens   : {_model.n_tokens}")
    if _model.n_tokens != REGISTRY.total_tokens:
        print(f"    WARNING: model.n_tokens ({_model.n_tokens}) ≠ "
              f"registry.total_tokens ({REGISTRY.total_tokens})")
    del _model

    n_folds = config.get("training", "n_folds")
    strat_groups = create_length_stratified_groups(
        all_sequences, labels, n_bins=config.get("training", "n_bins", default=5)
    )
    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True,
        random_state=config.get("training", "random_state", default=42)
    )

    bs = config.get("training", "batch_size")
    nw = config.get("training", "num_workers", default=1)

    fold_results = []
    model_paths  = []
    save_dir = output_dir / "models"
    save_dir.mkdir(exist_ok=True, parents=True)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_sequences, strat_groups)):
        print(f"\n{'='*60}\nFold {fold_idx+1}/{n_folds}\n{'='*60}")

        train_dataset = Subset(dataset, train_idx)
        val_dataset   = Subset(dataset, val_idx)

        train_labels_n = [{"lnc": 0, "pc": 1}[labels[i]] for i in train_idx]
        class_weights  = compute_class_weight(
            "balanced", classes=np.array([0, 1]), y=train_labels_n
        )

        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                                  num_workers=nw, pin_memory=True, drop_last=True)
        val_loader   = DataLoader(val_dataset, batch_size=bs, shuffle=False,
                                  num_workers=nw, pin_memory=True)

        model_save_path = save_dir / f"fold_{fold_idx}_best.pt"
        model = model_builder()

        model, best_metrics, history = train_one_fold(
            model, train_loader, val_loader, config, device,
            class_weights, model_save_path,
        )

        ckpt  = torch.load(model_save_path, map_location=device)
        model = model_builder()
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        print(f"Loaded epoch={ckpt['epoch']} val_acc={ckpt['val_acc']:.4f}")

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=nw):
                fwd_kw = _forward_kwargs(batch, device)
                out    = model(**fwd_kw)
                all_preds.append(out["logits"].argmax(1).cpu())
                all_labels.append(batch["label"].cpu())
        val_preds  = torch.cat(all_preds).numpy()
        val_labels = torch.cat(all_labels).numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average="binary"
        )

        fold_metrics = {
            "fold": fold_idx, "val_loss": best_metrics["val_loss"],
            "val_acc": best_metrics["val_acc"],
            "precision": precision, "recall": recall, "f1": f1,
            "confusion_matrix": confusion_matrix(val_labels, val_preds),
        }
        print(f"Fold {fold_idx+1}: acc={best_metrics['val_acc']:.4f} f1={f1:.4f}")

        model_paths.append({
            "fold": fold_idx, "path": str(model_save_path),
            "val_acc": fold_metrics["val_acc"], "val_loss": fold_metrics["val_loss"],
        })
        fold_results.append(fold_metrics)

    # ── Aggregate ──
    print(f"\n{'='*80}\nCV Results\n{'='*80}")
    cv_results = {}
    for m in ["val_acc", "precision", "recall", "f1"]:
        vals = [f[m] for f in fold_results]
        cv_results[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        print(f"  {m:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    pd.DataFrame(model_paths).to_csv(output_dir / "model_paths.csv", index=False)
    serializable = []
    for fold in fold_results:
        f = fold.copy()
        f["confusion_matrix"] = f["confusion_matrix"].tolist()
        serializable.append(f)
    with open(output_dir / "fold_results.json", "w") as fh:
        json.dump(serializable, fh, indent=2)

    # ── Test set evaluation ──
    lnc_test_fasta = config.get("data", "lnc_test_fasta", default=None)
    pc_test_fasta  = config.get("data", "pc_test_fasta",  default=None)

    test_results = None
    if lnc_test_fasta and pc_test_fasta:
        print("\n" + "=" * 80)
        print("INDEPENDENT TEST SET EVALUATION")
        print("=" * 80)
        test_dataset = _build_dataset(config, lnc_test_fasta, pc_test_fasta)
        print(f"  Samples: {len(test_dataset):,}")

        loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=nw)
        sum_probs = np.zeros((len(test_dataset), 2), dtype=np.float64)
        labels_arr = None
        for fold_info in model_paths:
            ckpt = torch.load(fold_info["path"], map_location=device)
            model = model_builder()
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device).eval()
            all_probs, all_labels = [], []
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"Fold {fold_info['fold']}", leave=False):
                    fwd_kw = _forward_kwargs(batch, device)
                    out    = model(**fwd_kw)
                    all_probs.append(torch.softmax(out["logits"], dim=1).cpu().numpy())
                    all_labels.append(batch["label"].cpu().numpy())
            sum_probs  += np.vstack(all_probs)
            labels_arr  = np.concatenate(all_labels)

        avg_probs   = sum_probs / len(model_paths)
        predictions = avg_probs.argmax(axis=1)
        acc = accuracy_score(labels_arr, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_arr, predictions, average="binary"
        )
        test_results = {
            "accuracy": float(acc), "precision": float(precision),
            "recall": float(recall), "f1": float(f1),
            "confusion_matrix": confusion_matrix(labels_arr, predictions).tolist(),
            "n_samples": int(len(labels_arr)),
            "n_folds_ensembled": len(model_paths),
        }
        with open(output_dir / "test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)
        print(f"  Saved → {output_dir / 'test_results.json'}")

    # ── Save hyperparameters ──
    lambda_feat = {
        b: config.get("training", f"lambda_feat_{b}", default=0.0)
        for b in REGISTRY.block_names
    }
    hp = {
        "architecture": "feature_only",
        "d_proj": config.get("model", "d_proj", default=64),
        "attn_heads": config.get("model", "attn_heads", default=4),
        "attn_mode": config.get("model", "attn_mode", default="standard"),
        "learning_rate": config.get("training", "learning_rate"),
        "batch_size": config.get("training", "batch_size"),
        "num_epochs": config.get("training", "num_epochs"),
        "lambda_feat": lambda_feat,
        "n_folds": n_folds,
        "registry_blocks": REGISTRY.block_names,
        "total_tokens": REGISTRY.total_tokens,
    }
    with open(output_dir / "hyperparameters.json", "w") as f:
        json.dump(hp, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    for metric, values in cv_results.items():
        print(f"  {metric:<12}: {values['mean']:.4f} ± {values['std']:.4f}")
    if test_results:
        print(f"\nIndependent Test Set:")
        for k in ["accuracy", "precision", "recall", "f1"]:
            print(f"  {k:<12}: {test_results[k]:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()