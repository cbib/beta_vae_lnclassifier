#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_residual_ablation.py

Residual token ablation for BetaVAESubgroup.

Extends the standard feature_zero ablation by zeroing only the z-orthogonal
(sequence-independent) component of each sub-group token representation.

This answers: after controlling for what the sequence encoder already captures,
which sub-groups carry independent classification signal?

Comparison with feature_zero ablation
--------------------------------------
feature_zero  : zeros all feature content (raw + sequence-correlated)
residual_zero : zeros only the z-orthogonal residual component
                (sequence-correlated component left intact)

A sub-group where residual_zero drop > feature_zero drop has MORE independent
signal than the raw ablation suggested. A sub-group where residual_zero ≈ 0
but feature_zero > 0 is contributing primarily through sequence-correlated
variance — a length/GC proxy rather than independent repeat biology.

Procedure
---------
For each fold:
1. Load pre-extracted z and token representations (fold_N_repr.npz)
2. Fit Ridge regressor per sub-group: z → T_sg (offline, full val set)
3. At inference time, hook intercepts token output from the relevant
   block's SubgroupProjectionLayer:
   - Captures z from fc_mu in the same forward pass
   - Computes z-explained component: T_hat = Ridge.predict(z)
   - Zeros the residual: T_ablated = T_hat  (residual set to zero)
   - Returns T_ablated to the attention mechanism
4. Measure accuracy drop vs baseline

Output
------
Same format as analyze_subgroup_ablation.py — compatible with existing
comparison scripts.

Usage
-----
python analysis/pattern/analyze_residual_ablation.py \
    --experiment_dir gencode_v49_experiments/beta_vae_subgroup_base_g49 \
    --repr_dir       gencode_v49_experiments/beta_vae_subgroup_base_g49/representations \
    --config         configs/beta_vae_subgroup_base_g49.json \
    --output_dir     gencode_v49_experiments/beta_vae_subgroup_base_g49/residual_ablation \
    --device         cuda:0
"""
from data.cv_utils import get_cv_splits_from_config

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
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
# Ridge regressor fitting (offline, per sub-group)
# ---------------------------------------------------------------------------

def fit_regressors(
    z:           np.ndarray,   # (N, latent_dim)
    tokens:      np.ndarray,   # (N, N_tokens, d_proj)
    token_names: List[str],
    alpha:       float = 1.0,
) -> Dict[str, Dict]:
    """
    Fit one Ridge regressor per sub-group: z → T_sg.

    Returns dict keyed by sub-group name containing:
        reg     : fitted Ridge regressor
        z_scaler: StandardScaler fitted on z
    """
    z_scaler = StandardScaler()
    z_scaled = z_scaler.fit_transform(z)

    regressors = {}
    for i, sg in enumerate(token_names):
        T_sg = tokens[:, i, :]
        reg  = Ridge(alpha=alpha)
        reg.fit(z_scaled, T_sg)
        regressors[sg] = {
            "reg":      reg,
            "z_scaler": z_scaler,
        }

    return regressors


# ---------------------------------------------------------------------------
# Residual ablation hook
# ---------------------------------------------------------------------------

class ResidualAblationHook:
    """
    Zeros the z-orthogonal (residual) component of a sub-group token.

    Resolves the relevant block's SubgroupProjectionLayer via
    model.projectors[block_name] (registry-driven nn.ModuleDict), and the
    token's position within that block's projector output via the
    registry's block_subgroups() ordering — works for any block, not just
    nonb/te.

    At forward time:
      1. Captures z from fc_mu via a secondary hook
      2. Computes z-explained component of the target token: T_hat = Ridge(z)
      3. Zeros the residual: token[:, token_idx, :] = T_hat
         (only the z-explained component remains — independent signal removed)

    This is the complement of feature_zero: instead of removing all feature
    content, we remove only the z-orthogonal component — leaving the
    sequence-explained component intact.

    Parameters
    ----------
    model    : BetaVAESubgroup instance
    subgroup : sub-group name to ablate
    reg_info : dict with 'reg' (Ridge) and 'z_scaler' (StandardScaler)
    device   : torch device
    registry : FeatureRegistry (defaults to module singleton)
    """

    def __init__(
        self,
        model,
        subgroup:  str,
        reg_info:  Dict,
        device:    torch.device,
        registry  = REGISTRY,
    ) -> None:
        self.model    = model
        self.subgroup = subgroup
        self.reg_info = reg_info
        self.device   = device
        self._handles = []
        self._registry = registry

        # Resolve block + position within that block's projector output
        block_name = registry.token_block(subgroup)
        if block_name not in model.projectors:
            raise ValueError(
                f"Block '{block_name}' (for subgroup '{subgroup}') not found "
                f"in model.projectors. Available blocks: "
                f"{list(model.projectors.keys())}"
            )
        self.proj_layer = model.projectors[block_name]

        block_subgroups = registry.block_subgroups(block_name)
        if subgroup not in block_subgroups:
            raise ValueError(
                f"Sub-group '{subgroup}' not found in block '{block_name}' "
                f"subgroups: {block_subgroups}"
            )
        self.token_idx = block_subgroups.index(subgroup)

        self._z_buffer: Optional[np.ndarray] = None
        self._pass: int = 1

    def __enter__(self):
        self._register_hooks()
        return self

    def __exit__(self, *args):
        for h in self._handles:
            h.remove()
        self._handles  = []
        self._z_buffer = None
        self._pass     = 1

    def set_pass(self, pass_num: int) -> None:
        """Switch between pass 1 (capture z) and pass 2 (apply residual hook)."""
        self._pass = pass_num

    def _register_hooks(self):
        # Hook on fc_mu: always capture z (used in pass 2)
        def hook_z(module, input, output):
            self._z_buffer = output.detach().cpu().numpy()

        handle_z = self.model.fc_mu.register_forward_hook(hook_z)
        self._handles.append(handle_z)

        # Hook on proj_layer: only modify output in pass 2
        token_idx = self.token_idx
        reg_info  = self.reg_info

        def hook_tokens(module, input, output):
            if self._pass != 2 or self._z_buffer is None:
                return output   # pass 1: no modification

            z_np     = self._z_buffer                      # (B, latent_dim)
            z_scaled = reg_info["z_scaler"].transform(z_np)
            T_hat    = reg_info["reg"].predict(z_scaled)   # (B, d_proj)
            T_hat_t  = torch.tensor(T_hat, dtype=output.dtype,
                                    device=output.device)

            output = output.clone()
            output[:, token_idx, :] = T_hat_t              # residual zeroed
            return output

        handle_t = self.proj_layer.register_forward_hook(hook_tokens)
        self._handles.append(handle_t)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model,
    loader:          DataLoader,
    device:          torch.device,
    ablation_hook    = None,
) -> tuple:
    """
    Run inference with optional residual ablation hook.

    When ablation_hook is provided, uses two-pass inference per batch:
      Pass 1 (hook set to pass=1): clean forward — captures z in hook buffer
      Pass 2 (hook set to pass=2): ablated forward — uses captured z to
                                    apply residual zeroing via token hook

    Pass 1 output is discarded; Pass 2 output is collected.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    if ablation_hook is not None:
        ablation_hook.__enter__()

    try:
        with torch.no_grad():
            for batch in loader:
                seq    = batch["sequence"].to(device)
                labels = batch["label"]
                fwd_kw = _forward_kwargs(batch, device)

                if ablation_hook is not None:
                    # Pass 1: capture z (token hook inactive)
                    ablation_hook.set_pass(1)
                    _ = model(seq, deterministic=True, **fwd_kw)

                    # Pass 2: apply residual zeroing using captured z
                    ablation_hook.set_pass(2)
                    out = model(seq, deterministic=True, **fwd_kw)
                else:
                    out = model(seq, deterministic=True, **fwd_kw)

                probs = torch.softmax(out["logits"], dim=1)
                preds = out["logits"].argmax(1)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.numpy())
                all_probs.append(probs.cpu().numpy())
    finally:
        if ablation_hook is not None:
            ablation_hook.__exit__(None, None, None)

    return (np.concatenate(all_preds),
            np.concatenate(all_labels),
            np.vstack(all_probs))


def compute_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="binary", zero_division=0)
    return {"accuracy": float(acc), "f1": float(f1)}


# ---------------------------------------------------------------------------
# Per-fold ablation
# ---------------------------------------------------------------------------

def ablate_fold(
    model,
    loader:      DataLoader,
    device:      torch.device,
    regressors:  Dict,
    token_names: List[str],
) -> pd.DataFrame:
    """Run residual_zero ablation for all sub-groups."""

    # Baseline
    base_preds, base_labels, _ = run_inference(model, loader, device)
    base = compute_metrics(base_preds, base_labels)
    print(f"    Baseline: acc={base['accuracy']:.4f} f1={base['f1']:.4f}")

    rows = []
    for sg in token_names:
        block = REGISTRY.token_block(sg)
        hook  = ResidualAblationHook(model, sg, regressors[sg], device)

        preds, labels, _ = run_inference(model, loader, device,
                                         ablation_hook=hook)
        metrics  = compute_metrics(preds, labels)
        acc_drop = base["accuracy"] - metrics["accuracy"]
        f1_drop  = base["f1"]       - metrics["f1"]

        rows.append({
            "subgroup":     sg,
            "block":        block,
            "mode":         "residual_zero",
            "baseline_acc": base["accuracy"],
            "baseline_f1":  base["f1"],
            "ablated_acc":  metrics["accuracy"],
            "ablated_f1":   metrics["f1"],
            "acc_drop":     float(acc_drop),
            "f1_drop":      float(f1_drop),
        })
        print(f"    {sg:15s} [{block:5s}] [residual_zero]: "
              f"acc={metrics['accuracy']:.4f} "
              f"(Δ={acc_drop:+.4f}) "
              f"f1={metrics['f1']:.4f} "
              f"(Δ={f1_drop:+.4f})")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Residual token ablation for BetaVAESubgroup"
    )
    parser.add_argument("--experiment_dir", required=True,
                        help="Experiment dir containing models/ subfolder")
    parser.add_argument("--repr_dir",       required=True,
                        help="Dir containing fold_*_repr.npz files")
    parser.add_argument("--config",         required=True)
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--batch_size",     type=int, default=256)
    parser.add_argument("--device",
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config     = load_config(args.config)
    exp_dir    = Path(args.experiment_dir)
    repr_dir   = Path(args.repr_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device     = torch.device(args.device)

    print("=" * 70)
    print("RESIDUAL TOKEN ABLATION")
    print("=" * 70)
    print(f"Registry blocks: {REGISTRY.block_names} "
          f"({REGISTRY.total_tokens} total tokens)")

    # ── Load sequences and reconstruct splits ─────────────────────────────────
    print("\nLoading sequences...")
    all_sequences, labels = load_sequences_in_order(
        lnc_fasta=config.get("data", "lnc_fasta"),
        pc_fasta= config.get("data", "pc_fasta"),
    )

    print("Loading dataset...")
    nonb2_csv    = config.get("data", "nonb2_csv",              default=None)
    nonb2_scaler = config.get("data", "nonb2_scaler_bank_path", default=None)
    if nonb2_csv is not None and nonb2_scaler is None:
        raise ValueError(
            "Config has data.nonb2_csv set but no data.nonb2_scaler_bank_path."
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
        max_length                = config.get("model", "max_length"),
    )
    print(f"Dataset: {len(dataset):,} samples")

    splits = get_cv_splits_from_config(config, all_sequences, labels)

    model_builder = create_model_builder(config)
    token_names   = REGISTRY.all_subgroups

    # ── Find fold checkpoints ─────────────────────────────────────────────────
    model_dir  = exp_dir / "models"
    fold_files = sorted(model_dir.glob("fold_*_best.pt"))
    if not fold_files:
        print(f"ERROR: No fold checkpoints in {model_dir}")
        return
    print(f"\nFound {len(fold_files)} fold checkpoint(s)")

    bs = args.batch_size
    nw = config.get("training", "num_workers", default=1)

    all_fold_results = []

    for ckpt_path in fold_files:
        fold_idx = int(ckpt_path.stem.split("_")[1])
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx}")
        print(f"{'='*60}")

        fold_out = output_dir / f"fold_{fold_idx}"
        fold_out.mkdir(exist_ok=True)
        results_csv = fold_out / "residual_ablation_results.csv"

        # Resume
        if results_csv.exists():
            print(f"   Already completed — loading from {results_csv}")
            fold_df = pd.read_csv(results_csv)
            all_fold_results.append(fold_df)
            continue

        # Load representations
        repr_path = repr_dir / f"fold_{fold_idx}_repr.npz"
        if not repr_path.exists():
            print(f"  ERROR: {repr_path} not found — run extract_representations.py first")
            continue

        repr_data = np.load(repr_path, allow_pickle=True)
        z_val     = repr_data["z"].astype(np.float32)
        tok_val   = repr_data["tokens"].astype(np.float32)

        # Use token_names from npz if available — must match the order tokens
        # were extracted in, which may differ from REGISTRY.all_subgroups
        # ordering if the checkpoint predates a registry change.
        if "token_names" in repr_data:
            fold_token_names = list(repr_data["token_names"])
        else:
            fold_token_names = token_names

        print(f"  Loaded representations: z={z_val.shape} tokens={tok_val.shape}")
        if tok_val.shape[1] != len(fold_token_names):
            print(f"    WARNING: tokens dim {tok_val.shape[1]} ≠ "
                  f"len(token_names) {len(fold_token_names)} — skipping fold")
            continue

        # Fit Ridge regressors offline on full val set
        print("  Fitting Ridge regressors (z → T_sg per sub-group)...")
        regressors = fit_regressors(z_val, tok_val, fold_token_names)

        # Load model
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = model_builder()
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        print(f"  Loaded epoch={ckpt['epoch']} val_acc={ckpt['val_acc']:.4f}")

        # Val loader
        _, val_idx = splits[fold_idx]
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=bs, shuffle=False,
            num_workers=nw, pin_memory=True,
        )
        print(f"  Val samples: {len(val_idx):,}")

        # Run ablation
        fold_df = ablate_fold(
            model, val_loader, device, regressors, fold_token_names
        )
        fold_df["fold"] = fold_idx

        fold_df.to_csv(results_csv, index=False)
        all_fold_results.append(fold_df)
        print(f"\n  Fold {fold_idx} complete → {fold_out}/")

    if not all_fold_results:
        print("No fold results — exiting.")
        return

    # ── Cross-fold summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("CROSS-FOLD SUMMARY")
    print(f"{'='*60}")

    combined = pd.concat(all_fold_results, ignore_index=True)
    combined.to_csv(output_dir / "all_folds_residual_ablation.csv", index=False)

    # Ranked summary
    print("\nSub-group importance ranking (residual_zero, mean acc drop):")
    summary = (combined.groupby("subgroup")["acc_drop"]
               .agg(["mean", "std"])
               .sort_values("mean", ascending=False))
    print(f"\n{'Rank':<5} {'Sub-group':<15} {'Mean Δacc':>10} {'Std':>8}")
    print("-" * 42)
    for rank, (sg, row) in enumerate(summary.iterrows(), 1):
        bar = "█" * max(0, int(row["mean"] * 1000))
        print(f"{rank:<5} {sg:<15} {row['mean']:>+10.4f}  "
              f"±{row['std']:.4f}  {bar}")

    print(f"\nAll outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()