#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast single-fold training for BetaVAESubgroup.

Uses an 80/20 stratified split instead of 5-fold CV.
Intended for validating architecture variants before full CV.

Usage
-----
# Baseline
python src/main_subgroup_debug.py \
    --config configs/beta_vae_subgroup_base_g49.json \
    --device cuda:0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Subset

from configs.load_config import load_config
from models.model_builder import create_model_builder
from data.cv_utils import load_sequences_in_order, create_length_stratified_groups
from data.gated_feature_dataset import SequenceFeatureDataset
from data.feature_registry import REGISTRY
from trainers.beta_vae_subgroup_trainer import SingleFoldSubgroupTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Single-fold debug training for BetaVAESubgroup"
    )
    parser.add_argument("--config",      required=True)
    parser.add_argument("--device",      default="cuda:0" if torch.cuda.is_available()
                                                 else "cpu")
    parser.add_argument("--max_epochs",  type=int, default=None)
    parser.add_argument("--val_size",    type=float, default=0.2)
    args = parser.parse_args()

    config     = load_config(args.config)
    num_epochs = args.max_epochs or config.get("training", "num_epochs")
    attn_mode  = config.get("model", "attn_mode", default="standard")

    base_dir   = Path(config.get("output", "experiment_name"))
    output_dir = Path(str(base_dir) + f"_{attn_mode}_debug")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BetaVAESubgroup — Single-Fold Debug Training")
    print("=" * 70)
    print(f"Config     : {args.config}")
    print(f"attn_mode  : {attn_mode}")
    print(f"Output dir : {output_dir}")
    print(f"Val frac   : {args.val_size:.0%}")
    print(f"Max epochs : {num_epochs}")
    print(f"Device     : {args.device}")
    print(f"d_proj     : {config.get('model', 'd_proj', default=64)}")
    print(f"lambda_ortho: {config.get('training', 'lambda_ortho', default=0.0)}")
    print("=" * 70)
    print("Feature blocks (from registry):")
    for block_name in REGISTRY.block_names:
        dim    = REGISTRY.block_dim(block_name)
        source = REGISTRY.block_source(block_name)
        n_sg   = len(REGISTRY.block_subgroups(block_name))
        lf     = config.get("training", f"lambda_feat_{block_name}", default=0.0)
        print(f"  {block_name:8s}: {dim:4d} features  ({n_sg} subgroups, "
              f"source={source}, lambda_feat={lf})")
    print(f"  TOTAL TOKENS: {REGISTRY.total_tokens}")
    print("=" * 70)

    # ── Load sequences ────────────────────────────────────────────────────────
    all_sequences, labels = load_sequences_in_order(
        lnc_fasta=config.get("data", "lnc_fasta"),
        pc_fasta= config.get("data", "pc_fasta"),
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    nonb2_csv    = config.get("data", "nonb2_csv",              default=None)
    nonb2_scaler = config.get("data", "nonb2_scaler_bank_path", default=None)

    if nonb2_csv is not None and nonb2_scaler is None:
        raise ValueError(
            "Config has data.nonb2_csv set but no data.nonb2_scaler_bank_path. "
            "Run prepare_features.py with --nonb2_csv to generate the scaler bank."
        )

    dataset = SequenceFeatureDataset(
        lnc_fasta              = config.get("data", "lnc_fasta"),
        pc_fasta                = config.get("data", "pc_fasta"),
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
    print(f"Dataset: {len(dataset):,} samples")

    # ── 80/20 stratified split ────────────────────────────────────────────────
    strat_groups = create_length_stratified_groups(
        all_sequences, labels,
        n_bins=config.get("training", "n_bins", default=5)
    )
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=args.val_size,
        random_state=config.get("training", "random_state", default=42)
    )
    train_idx, val_idx = next(splitter.split(all_sequences, strat_groups))
    print(f"Split: {len(train_idx):,} train / {len(val_idx):,} val")

    train_labels_n = [{"lnc": 0, "pc": 1}[labels[i]] for i in train_idx]
    class_weights  = compute_class_weight(
        "balanced", classes=np.array([0, 1]), y=train_labels_n
    )

    # ── DataLoaders ───────────────────────────────────────────────────────────
    bs = config.get("training", "batch_size")
    nw = config.get("training", "num_workers", default=1)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=bs,
                              shuffle=True,  num_workers=nw,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=bs,
                              shuffle=False, num_workers=nw, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model_builder = create_model_builder(config)
    model         = model_builder()
    total_params  = sum(p.numel() for p in model.parameters())
    print(f"\nModel : {total_params:,} params")
    print(f"  attn_mode  : {model.attn_mode}")
    print(f"  L_encoded  : {model.encoded_length}")
    print(f"  N tokens   : {model.n_tokens}")
    print(f"  d_proj     : {model.cross_attn.d_feat}")
    print(f"  num_heads  : {model.cross_attn.num_heads}")
    if model.n_tokens != REGISTRY.total_tokens:
        print(f"    WARNING: model.n_tokens ({model.n_tokens}) ≠ "
              f"registry.total_tokens ({REGISTRY.total_tokens})")

    with open(output_dir / "model_architecture.txt", "w") as f:
        f.write(str(model))

    # ── Train ─────────────────────────────────────────────────────────────────
    save_dir = output_dir / "models"
    save_dir.mkdir(exist_ok=True)

    lambda_feat = {
        b: config.get("training", f"lambda_feat_{b}", default=0.0)
        for b in REGISTRY.block_names
    }

    trainer = SingleFoldSubgroupTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        learning_rate        = config.get("training", "learning_rate"),
        weight_decay         = config.get("training", "weight_decay"),
        alpha                = config.get("training", "alpha"),
        beta                 = config.get("training", "beta"),
        gamma_classification = config.get("training", "gamma_classification"),
        lambda_recon         = config.get("training", "lambda_recon"),
        reconstruction_loss  = config.get("training", "reconstruction_loss"),
        class_weights        = class_weights,
        gamma_attn           = config.get("training", "gamma_attn", default=0.0),
        lambda_feat          = lambda_feat,
        lambda_conc          = config.get("training", "lambda_conc",  default=0.1),
        lambda_ortho         = config.get("training", "lambda_ortho", default=0.0),
        kl_anneal_epochs     = config.get("training", "kl_anneal_epochs", default=20),
        kl_anneal_end        = config.get("training", "kl_anneal_end",    default=4.0),
        device               = args.device,
    )

    model, best_metrics, history = trainer.train(
        num_epochs              = num_epochs,
        early_stopping_patience = config.get("training", "early_stopping_patience"),
        save_path               = save_dir / "fold_0_best.pt",
    )

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "attn_mode":            attn_mode,
        "val_loss":             best_metrics["val_loss"],
        "val_acc":              best_metrics["val_acc"],
        "train_loss":           best_metrics["train_loss"],
        "n_train":              len(train_idx),
        "n_val":                len(val_idx),
        "num_epochs_trained":   len(history["train_loss"]),
        "final_attn_entropy":   history["attn_entropy"][-1] if history["attn_entropy"] else None,
        "min_attn_entropy":     min(history["attn_entropy"]) if history["attn_entropy"] else None,
        "registry_blocks":      REGISTRY.block_names,
        "total_tokens":         REGISTRY.total_tokens,
    }
    with open(output_dir / "debug_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Attention entropy plot ────────────────────────────────────────────────
    if history["attn_entropy"]:
        try:
            import matplotlib.pyplot as plt

            n_tokens    = model.n_tokens
            max_ent     = float(np.log(n_tokens))
            epochs      = range(1, len(history["attn_entropy"]) + 1)
            entropy_pct = [100 * e / max_ent for e in history["attn_entropy"]]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(epochs, entropy_pct, color="steelblue", linewidth=2)
            ax.axhline(100, color="green", linestyle="--", alpha=0.5,
                       label="Max entropy (uniform)")
            ax.axhline(20,  color="red",   linestyle="--", alpha=0.5,
                       label="Collapse threshold (20%)")
            ax.fill_between(epochs, 0, 20, alpha=0.08, color="red")
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Attention entropy (% of max)", fontsize=12)
            ax.set_title(
                f"Attention Entropy During Training — attn_mode='{attn_mode}'\n"
                f"(max = log({n_tokens}) = {max_ent:.2f} nats)",
                fontsize=13, fontweight="bold"
            )
            ax.legend(fontsize=10)
            ax.set_ylim(0, 110)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "attn_entropy_training.png",
                        dpi=200, bbox_inches="tight")
            plt.close()
            print(f"\nSaved entropy plot → {output_dir}/attn_entropy_training.png")
        except Exception as e:
            print(f"  (Could not save entropy plot: {e})")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DEBUG TRAINING COMPLETE")
    print("=" * 70)
    print(f"  attn_mode   : {attn_mode}")
    print(f"  val_acc     : {best_metrics['val_acc']:.4f}")
    print(f"  val_loss    : {best_metrics['val_loss']:.4f}")
    if history["attn_entropy"]:
        max_ent = float(np.log(model.n_tokens))
        final_e = history["attn_entropy"][-1]
        print(f"  Final attn entropy : {final_e:.3f} ({100*final_e/max_ent:.0f}% of max)")
        if 100 * final_e / max_ent < 30:
            print("    Attention collapsed — increase gamma_attn")
        else:
            print("    Attention entropy healthy")
    print(f"\nOutputs: {output_dir}/")


if __name__ == "__main__":
    main()