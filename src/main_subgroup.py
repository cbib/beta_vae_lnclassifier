#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full 5-fold cross-validation training for BetaVAESubgroup.

Usage
-----
python src/main_subgroup.py \
    --config  configs/beta_vae_subgroup_base_g49.json \
    --device  cuda:0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from configs.load_config import load_config
from models.model_builder import create_model_builder
from data.cv_utils import load_sequences_in_order
from data.gated_feature_dataset import SequenceFeatureDataset
from data.feature_registry import REGISTRY
from trainers.beta_vae_subgroup_trainer import BetaVAESubgroupTrainer


def _build_dataset(config, lnc_fasta, pc_fasta):
    """Construct SequenceFeatureDataset from config, including NonB2 if present."""
    nonb2_csv    = config.get("data", "nonb2_csv",              default=None)
    nonb2_scaler = config.get("data", "nonb2_scaler_bank_path", default=None)

    if nonb2_csv is not None and nonb2_scaler is None:
        raise ValueError(
            "Config has data.nonb2_csv set but no data.nonb2_scaler_bank_path. "
            "Run prepare_features.py with --nonb2_csv to generate the scaler bank."
        )

    return SequenceFeatureDataset(
        lnc_fasta              = lnc_fasta,
        pc_fasta                = pc_fasta,
        te_genomic_csv          = config.get("data", "te_genomic_csv"),
        te_processed_csv        = config.get("data", "te_processed_csv",   default=None),
        nonb_genomic_csv        = config.get("data", "nonb_genomic_csv"),
        nonb_processed_csv      = config.get("data", "nonb_processed_csv", default=None),
        nonb2_csv               = nonb2_csv,
        te_scaler_bank_path     = config.get("data", "te_scaler_bank_path"),
        nonb_scaler_bank_path   = config.get("data", "nonb_scaler_bank_path"),
        nonb2_scaler_bank_path  = nonb2_scaler,
        max_length               = config.get("model", "max_length"),
    )


def main():
    parser = argparse.ArgumentParser(
        description="5-fold CV training for BetaVAESubgroup"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available()
                                             else "cpu")
    args = parser.parse_args()

    config    = load_config(args.config)
    attn_mode = config.get("model", "attn_mode", default="standard")

    output_dir = Path(config.get("output", "experiment_name"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BetaVAESubgroup Training")
    print("=" * 80)
    print(f"Architecture : beta_vae_subgroup")
    print(f"attn_mode    : {attn_mode}")
    print(f"Latent dim   : {config.get('model', 'latent_dim')}")
    print(f"Beta         : {config.get('model', 'beta')}")
    print(f"d_proj       : {config.get('model', 'd_proj', default=64)}")
    print(f"attn_heads   : {config.get('model', 'attn_heads', default=4)}")
    print(f"Device       : {args.device}")
    print(f"N folds      : {config.get('training', 'n_folds')}")
    print(f"Test set     : ENABLED")
    print("=" * 80)
    print("Feature blocks (from registry):")
    for block_name in REGISTRY.block_names:
        dim    = REGISTRY.block_dim(block_name)
        source = REGISTRY.block_source(block_name)
        n_sg   = len(REGISTRY.block_subgroups(block_name))
        print(f"  {block_name:8s}: {dim:4d} features  "
              f"({n_sg} subgroups, source={source})")
    print(f"  TOTAL TOKENS: {REGISTRY.total_tokens}")
    print("=" * 80)

    # ── Load sequences ────────────────────────────────────────────────────────
    print("Loading sequences in CANONICAL order (lnc + pc)...")
    all_sequences, labels = load_sequences_in_order(
        lnc_fasta=config.get("data", "lnc_fasta"),
        pc_fasta= config.get("data", "pc_fasta"),
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Creating Dataset")
    print("=" * 80)

    dataset = _build_dataset(
        config,
        lnc_fasta=config.get("data", "lnc_fasta"),
        pc_fasta=config.get("data", "pc_fasta"),
    )
    print(f"\nDataset: {len(dataset):,} samples")

    # ── Model builder ─────────────────────────────────────────────────────────
    model_builder = create_model_builder(config)

    # Print model summary from a fresh instance
    _model = model_builder()
    total_params = sum(p.numel() for p in _model.parameters())
    print(f"\nModel: {total_params:,} params "
          f"({sum(p.numel() for p in _model.parameters() if p.requires_grad):,} trainable)")
    print(f"L_encoded (attention positions): {_model.encoded_length}")
    print(f"N tokens  (sub-group tokens)   : {_model.n_tokens}")
    if _model.n_tokens != REGISTRY.total_tokens:
        print(f"    WARNING: model.n_tokens ({_model.n_tokens}) ≠ "
              f"registry.total_tokens ({REGISTRY.total_tokens})")
    del _model

    # ── Cross-validation ──────────────────────────────────────────────────────
    trainer = BetaVAESubgroupTrainer(
        model_builder = model_builder,
        dataset       = dataset,
        config        = config,
        n_folds       = config.get("training", "n_folds"),
        device        = args.device,
    )

    cv_results = trainer.cross_validate(
        sequences          = all_sequences,
        labels             = labels,
        stratify_by_length = True,
    )

    # ── Save CV results ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    trainer.save_results(output_dir)

    # ── Extract attention weights ─────────────────────────────────────────────
    attn_dir = output_dir / "fold_attention"
    trainer.extract_attention_all_folds(
        output_dir      = attn_dir,
        labels_override = labels,
    )

    # ── Independent test set evaluation ──────────────────────────────────────
    lnc_test_fasta = config.get("data", "lnc_test_fasta", default=None)
    pc_test_fasta  = config.get("data", "pc_test_fasta",  default=None)

    if lnc_test_fasta and pc_test_fasta:
        print("\n" + "=" * 80)
        print("INDEPENDENT TEST SET EVALUATION")
        print("=" * 80)

        test_dataset = _build_dataset(
            config,
            lnc_fasta=lnc_test_fasta,
            pc_fasta=pc_test_fasta,
        )
        print(f"  Samples: {len(test_dataset):,}")

        for fold_info in trainer.model_paths:
            ckpt = torch.load(fold_info["path"],
                              map_location=torch.device(args.device))
            print(f"  Fold {fold_info['fold']} | "
                  f"epoch={ckpt['epoch']} "
                  f"val_acc={ckpt['val_acc']:.4f}")

        test_results = trainer.evaluate_on_test_set(test_dataset)

        print(f"\n  Test results:")
        for k, v in test_results.items():
            if k != "confusion_matrix":
                print(f"    {k:<20}: {v}")

        test_results_path = output_dir / "test_results.json"
        with open(test_results_path, "w") as f:
            json.dump(test_results, f, indent=2)
        print(f"\n  Saved → {test_results_path}")
    else:
        print("\n  No test FASTAs specified — skipping test set evaluation.")

    # ── Save config and hyperparameters ───────────────────────────────────────
    config_out = output_dir / "config.json"
    with open(config_out, "w") as f:
        json.dump(config._config if hasattr(config, "_config") else {}, f, indent=2)
    print(f"Configuration saved to {config_out}")

    # lambda_feat: one entry per registry block, read dynamically
    lambda_feat = {
        b: config.get("training", f"lambda_feat_{b}", default=0.0)
        for b in REGISTRY.block_names
    }

    hp = {
        "architecture":     "beta_vae_subgroup",
        "attn_mode":        attn_mode,
        "latent_dim":       config.get("model", "latent_dim"),
        "beta":             config.get("model", "beta"),
        "d_proj":           config.get("model", "d_proj", default=64),
        "attn_heads":       config.get("model", "attn_heads", default=4),
        "attn_temperature": config.get("model", "attn_temperature", default=4.0),
        "dropout_rate":     config.get("model", "dropout_rate"),
        "learning_rate":    config.get("training", "learning_rate"),
        "batch_size":       config.get("training", "batch_size"),
        "num_epochs":       config.get("training", "num_epochs"),
        "lambda_feat":      lambda_feat,
        "lambda_conc":      config.get("training", "lambda_conc",  default=0.1),
        "lambda_ortho":     config.get("training", "lambda_ortho", default=0.0),
        "n_folds":          config.get("training", "n_folds"),
        "registry_blocks":  REGISTRY.block_names,
        "total_tokens":     REGISTRY.total_tokens,
    }
    hp_path = output_dir / "hyperparameters.json"
    with open(hp_path, "w") as f:
        json.dump(hp, f, indent=2)
    print(f"Saved hyperparameters → {hp_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    print(f"\nCross-Validation (5-fold):")
    for metric, values in cv_results.items():
        print(f"  {metric:<12}: {values['mean']:.4f} ± {values['std']:.4f}")

    if lnc_test_fasta and pc_test_fasta:
        print(f"\nIndependent Test Set:")
        for k, v in test_results.items():
            if k not in ("confusion_matrix", "n_samples", "n_lncrna",
                         "n_pcrna", "n_folds_ensembled"):
                print(f"  {k:<20}: {v:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()