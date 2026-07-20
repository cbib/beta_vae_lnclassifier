#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_representations.py

Extracts latent representations (z), sub-group token vectors, and sequence-level
properties from BetaVAESubgroup fold checkpoints for latent probing and
Pattern analysis.

Output .npz keys
----------------
    z              (N, latent_dim)       VAE latent mean (deterministic)
    tokens         (N, N_tokens, d_proj) sub-group tokens after token_norm
    labels         (N,)
    lengths        (N,)
    gc_content     (N,)
    predictions    (N,)
    confidences    (N,)
    transcript_ids (N,)
    token_names    (N_tokens,)           sub-group name per token position
    block_names    (N_tokens,)           block name per token position
    nonb_subgroups (n_nonb,)             legacy alias
    te_subgroups   (n_te,)               legacy alias

Usage
-----
python analysis/pattern/extract_representations.py \\
    --experiment_dir gencode_v49_experiments/beta_vae_subgroup_base_g49 \\
    --config         configs/beta_vae_subgroup_base_g49.json \\
    --output_dir     gencode_v49_experiments/beta_vae_subgroup_base_g49/representations \\
    --device         cuda:0
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from configs.load_config import load_config
from models.model_builder import create_model_builder
from data.cv_utils import load_sequences_in_order, create_length_stratified_groups
from data.gated_feature_dataset import SequenceFeatureDataset
from data.feature_registry import REGISTRY


# ---------------------------------------------------------------------------
# GC content from one-hot batch
# ---------------------------------------------------------------------------

def compute_gc(seq_batch: torch.Tensor) -> np.ndarray:
    """
    Compute GC fraction from one-hot encoded sequence tensor.

    Parameters
    ----------
    seq_batch : (B, 5, L)  channels: A=0 C=1 G=2 U/T=3 N=4
    """
    gc_counts    = (seq_batch[:, 1, :].sum(dim=-1)
                    + seq_batch[:, 2, :].sum(dim=-1))
    total_counts = seq_batch[:, :4, :].sum(dim=-1).sum(dim=-1)
    return (gc_counts / total_counts.clamp(min=1)).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Representation extraction for one fold
# ---------------------------------------------------------------------------

def extract_fold(
    model,
    val_loader: DataLoader,
    device:     torch.device,
    has_nonb2:  bool,
) -> dict:
    """
    Run inference with forward hooks to capture z and token representations.
    Returns dict of numpy arrays for all val samples.
    """
    model.eval()

    z_store      = []
    tokens_store = []

    def hook_z(module, input, output):
        z_store.append(output.detach().cpu())

    def hook_tokens(module, input, output):
        # output: (B, N_tokens, d_proj)
        tokens_store.append(output.detach().cpu())

    handle_z      = model.fc_mu.register_forward_hook(hook_z)
    handle_tokens = model.token_norm.register_forward_hook(hook_tokens)

    all_preds   = []
    all_confs   = []
    all_labels  = []
    all_ids     = []
    all_lengths = []
    all_gc      = []

    try:
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="  Extracting", leave=False):
                seq  = batch["sequence"].to(device)
                te_g = batch["te_genomic"].to(device)
                te_p = batch["te_processed"].to(device)
                nb_g = batch["nonb_genomic"].to(device)
                nb_p = batch["nonb_processed"].to(device)

                all_gc.append(compute_gc(seq))

                fwd_kwargs = dict(
                    te_genomic    = te_g,
                    te_processed  = te_p,
                    nonb_genomic  = nb_g,
                    nonb_processed= nb_p,
                    deterministic = True,
                )
                if has_nonb2 and "nonb2" in batch:
                    fwd_kwargs["nonb2"] = batch["nonb2"].to(device)

                out   = model(seq, **fwd_kwargs)
                probs = torch.softmax(out["logits"], dim=1)
                preds = out["logits"].argmax(1)
                confs = probs.max(1).values

                all_preds.append(preds.cpu().numpy())
                all_confs.append(confs.cpu().numpy())
                all_labels.append(batch["label"].numpy())
                all_ids.extend(batch["transcript_id"])
                all_lengths.append(batch["length"].numpy())

    finally:
        handle_z.remove()
        handle_tokens.remove()

    return {
        "z":             torch.cat(z_store,      dim=0).numpy(),
        "tokens":        torch.cat(tokens_store, dim=0).numpy(),
        "predictions":   np.concatenate(all_preds),
        "confidences":   np.concatenate(all_confs),
        "labels":        np.concatenate(all_labels),
        "lengths":       np.concatenate(all_lengths).astype(np.float32),
        "gc_content":    np.concatenate(all_gc),
        "transcript_ids":np.array(all_ids, dtype=object),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract z and token representations from BetaVAESubgroup checkpoints"
    )
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--config",         required=True)
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--device",
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    config     = load_config(args.config)
    exp_dir    = Path(args.experiment_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device     = torch.device(args.device)

    print("=" * 70)
    print("BetaVAESubgroup — Representation Extraction")
    print("=" * 70)
    print(f"Experiment : {exp_dir}")
    print(f"Config     : {args.config}")
    print(f"Output dir : {output_dir}")
    print(f"Device     : {args.device}")
    print("=" * 70)

    # ── Token metadata from registry ──────────────────────────────────────────
    # Build token_names and block_names in registry order
    all_subgroups = REGISTRY.all_subgroups   # ordered list across all blocks
    token_names   = np.array(all_subgroups, dtype=object)
    block_names   = np.array(
        [REGISTRY.token_block(sg) for sg in all_subgroups], dtype=object
    )

    # Legacy aliases for backward compatibility
    nonb_subgroups = np.array(REGISTRY.nonb_subgroups, dtype=object)
    te_subgroups   = np.array(REGISTRY.te_subgroups,   dtype=object)

    print(f"\nRegistry: {REGISTRY.total_tokens} tokens across "
          f"{len(REGISTRY.block_names)} blocks")
    for b in REGISTRY.block_names:
        sgs = REGISTRY.block_subgroups(b)
        print(f"  {b}: {len(sgs)} subgroups — {sgs}")

    # ── Load sequences ────────────────────────────────────────────────────────
    print("\nLoading sequences...")
    all_sequences, labels = load_sequences_in_order(
        lnc_fasta=config.get("data", "lnc_fasta"),
        pc_fasta= config.get("data", "pc_fasta"),
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("Loading dataset...")
    has_nonb2      = config.get("data", "nonb2_csv", default=None) is not None
    nonb2_csv      = config.get("data", "nonb2_csv",              default=None)
    nonb2_scaler   = config.get("data", "nonb2_scaler_bank_path", default=None)

    dataset = SequenceFeatureDataset(
        lnc_fasta              = config.get("data", "lnc_fasta"),
        pc_fasta               = config.get("data", "pc_fasta"),
        te_genomic_csv         = config.get("data", "te_genomic_csv"),
        te_processed_csv       = config.get("data", "te_processed_csv",   default=None),
        nonb_genomic_csv       = config.get("data", "nonb_genomic_csv"),
        nonb_processed_csv     = config.get("data", "nonb_processed_csv", default=None),
        nonb2_csv              = nonb2_csv,
        te_scaler_bank_path    = config.get("data", "te_scaler_bank_path"),
        nonb_scaler_bank_path  = config.get("data", "nonb_scaler_bank_path"),
        nonb2_scaler_bank_path = nonb2_scaler,
        max_length             = config.get("model", "max_length"),
    )
    print(f"Dataset: {len(dataset):,} samples")
    print(f"NonB2 features: {'loaded' if has_nonb2 else 'not available'}")

    # ── Reconstruct CV splits ─────────────────────────────────────────────────
    strat_groups = create_length_stratified_groups(
        all_sequences, labels,
        n_bins=config.get("training", "n_bins", default=5)
    )
    skf = StratifiedKFold(
        n_splits     = config.get("training", "n_folds"),
        shuffle      = True,
        random_state = config.get("training", "random_state", default=42)
    )
    splits = list(skf.split(all_sequences, strat_groups))

    model_builder = create_model_builder(config)

    # ── Find fold checkpoints ─────────────────────────────────────────────────
    model_dir  = exp_dir / "models"
    fold_files = sorted(model_dir.glob("fold_*_best.pt"))
    if not fold_files:
        print(f"ERROR: No fold checkpoints found in {model_dir}")
        return
    print(f"\nFound {len(fold_files)} fold checkpoint(s)")

    bs = args.batch_size or config.get("training", "batch_size")
    nw = config.get("training", "num_workers", default=1)

    # ── Per-fold extraction ───────────────────────────────────────────────────
    for ckpt_path in fold_files:
        fold_idx  = int(ckpt_path.stem.split("_")[1])
        save_path = output_dir / f"fold_{fold_idx}_repr.npz"

        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx}")
        print(f"{'='*60}")

        if save_path.exists():
            print(f"   Already extracted — skipping (delete to re-run)")
            continue

        ckpt  = torch.load(ckpt_path, map_location=device)
        model = model_builder()
        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
        model.to(device)
        model.eval()
        print(f"  Loaded epoch={ckpt['epoch']}  val_acc={ckpt['val_acc']:.4f}")
        print(f"  n_tokens={model.n_tokens}")

        _, val_idx = splits[fold_idx]
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=bs, shuffle=False,
            num_workers=nw, pin_memory=True,
        )
        print(f"  Val samples: {len(val_idx):,}")

        result = extract_fold(model, val_loader, device, has_nonb2)

        # Sanity check
        acc = (result["predictions"] == result["labels"]).mean()
        print(f"  Extracted val_acc={acc:.4f}  "
              f"(checkpoint: {ckpt['val_acc']:.4f})")
        if abs(acc - float(ckpt["val_acc"])) > 0.002:
            print(f"    Accuracy mismatch — check split reconstruction")

        # Verify token dimension matches registry
        n_tokens_actual = result["tokens"].shape[1]
        if n_tokens_actual != REGISTRY.total_tokens:
            print(f"    tokens dim {n_tokens_actual} ≠ registry "
                  f"{REGISTRY.total_tokens} — checkpoint may predate NonB2 block")

        np.savez(
            save_path,
            # Core representations
            z              = result["z"],
            tokens         = result["tokens"],
            labels         = result["labels"],
            lengths        = result["lengths"],
            gc_content     = result["gc_content"],
            predictions    = result["predictions"],
            confidences    = result["confidences"],
            transcript_ids = result["transcript_ids"],
            # Token metadata — primary (dynamic)
            token_names    = token_names[:n_tokens_actual],
            block_names    = block_names[:n_tokens_actual],
            # Legacy aliases for backward compatibility
            nonb_subgroups = nonb_subgroups,
            te_subgroups   = te_subgroups,
        )

        print(f"  z shape     : {result['z'].shape}")
        print(f"  tokens shape: {result['tokens'].shape}")
        print(f"  token_names : {token_names[:n_tokens_actual].tolist()}")
        print(f"  Saved → {save_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    saved = sorted(output_dir.glob("fold_*_repr.npz"))
    print(f"  {len(saved)} fold(s) saved to {output_dir}/")
    for p in saved:
        d = np.load(p, allow_pickle=True)
        print(f"  {p.name}: N={len(d['labels']):,}  "
              f"z={d['z'].shape}  tokens={d['tokens'].shape}  "
              f"blocks={list(d['block_names'])}")


if __name__ == "__main__":
    main()