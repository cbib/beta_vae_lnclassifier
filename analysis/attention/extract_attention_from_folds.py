#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis/attention/extract_attention_from_folds.py

Layer 1 attention extraction for both architectures:
- beta_vae_features_attn (legacy): attn_weights in forward() output
  dict, shape (B, L_encoded, 2), over [REP, NonB] as whole modalities.
- beta_vae_subgroup (current): attn_weights as the second element of
  CrossModalAttentionMH.forward()'s (out, attn_weights) return, shape
  (B, H, L, N=20), over individual subgroup tokens. Captured via a
  forward hook on model.cross_attn.

Architecture is detected from each experiment's config
(model.architecture). Checkpoints are discovered via model_paths.csv
at the experiment directory root (columns: fold,path,val_acc,val_loss),
resolved against --repo_root.

Attention is aggregated by mean over batch, heads (current arch only),
and sequence position L, down to one weight per subgroup/modality per
fold. Raw per-transcript arrays are saved as .npz alongside the
aggregated summary.

Legacy and current outputs are kept in separate files and are not
merged into one table (N=2 vs N=20, different quantities).

Usage
-----
python analysis/attention/extract_attention_from_folds.py \
    --experiment_dir gencode_v49_experiments/beta_vae_subgroup_base_g49 \
    --config configs/beta_vae_subgroup_base_g49.json \
    --output_dir gencode_v49_experiments/beta_vae_subgroup_base_g49/attention \
    --device cuda:0

python analysis/attention/extract_attention_from_folds.py \
    --experiment_dir gencode_v49_experiments/beta_vae_features_attn_g49 \
    --config configs/beta_vae_features_attn_g49.json \
    --output_dir gencode_v49_experiments/beta_vae_features_attn_g49/attention \
    --device cuda:0

Output
------
<output_dir>/
  fold_N_attn_weights.npz     raw per-transcript aggregated weights, fold N
                               (legacy: shape (n_transcripts, 2);
                                current: shape (n_transcripts, 20))
  attn_summary_per_fold.csv   long-format: fold, subgroup/modality, mean, std
  attn_summary_cross_fold.csv cross-fold mean +/- std per subgroup/modality,
                               sorted descending by mean attention weight
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.load_config import load_config
from models.model_builder import create_model_builder
from data.cv_utils import load_sequences_in_order
from data.feature_registry import REGISTRY, display_block, display_subgroups

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

BLOCK_COLORS = {
    "nonb":  "#4A90D9",
    "te":    "#F4B942",
    "nonb2": "#9B7FD4",
}


def _block_colors_for(subgroup_names, block_map):
    return [BLOCK_COLORS.get(block_map.get(sg, ""), "#999999")
            for sg in subgroup_names]


def _block_boundaries(subgroup_names, block_map):
    boundaries = []
    prev_block = None
    for i, sg in enumerate(subgroup_names):
        block = block_map.get(sg)
        if prev_block is not None and block != prev_block:
            boundaries.append(i - 0.5)
        prev_block = block
    return boundaries


def _block_legend_handles(subgroup_names, block_map):
    from matplotlib.patches import Patch
    seen = []
    for sg in subgroup_names:
        block = block_map.get(sg)
        if block is not None and block not in seen:
            seen.append(block)
    return [Patch(facecolor=BLOCK_COLORS.get(b, "#999999"), label=display_block(b))
            for b in seen]


# ---------------------------------------------------------------------------
# Checkpoint manifest
# ---------------------------------------------------------------------------

def load_model_paths(experiment_dir: Path) -> pd.DataFrame:
    manifest = experiment_dir / "model_paths.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"model_paths.csv not found at {manifest}. Both architecture "
            f"vintages write this manifest at the experiment directory "
            f"root — confirm --experiment_dir points at the right place."
        )
    return pd.read_csv(manifest)


def resolve_checkpoint_path(raw_path: str, repo_root: Path) -> Path:
    """model_paths.csv stores paths relative to the repo root, not the
    experiment directory — resolve against repo_root, falling back to
    treating raw_path as already-absolute/already-correct if that fails."""
    candidate = repo_root / raw_path
    if candidate.exists():
        return candidate
    candidate2 = Path(raw_path)
    if candidate2.exists():
        return candidate2
    raise FileNotFoundError(
        f"Could not resolve checkpoint path '{raw_path}' against repo "
        f"root '{repo_root}' or as a standalone path. Pass --repo_root "
        f"explicitly if the manifest's paths are relative to something else."
    )


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

LEGACY_ARCHITECTURES = {"beta_vae_features_attn"}
CURRENT_ARCHITECTURES = {"beta_vae_subgroup"}


def detect_architecture(config) -> str:
    arch = config.get("model", "architecture")
    if arch in LEGACY_ARCHITECTURES:
        return "legacy"
    if arch in CURRENT_ARCHITECTURES:
        return "current"
    raise ValueError(
        f"Unrecognized architecture '{arch}' for attention extraction. "
        f"Known legacy: {LEGACY_ARCHITECTURES}. Known current: "
        f"{CURRENT_ARCHITECTURES}. If this is a new architecture, add it "
        f"to one of these sets after confirming how it exposes attn_weights."
    )


# ---------------------------------------------------------------------------
# Dataset construction, per architecture
# ---------------------------------------------------------------------------

def build_dataset_legacy(config):
    from data.feature_dataset import SequenceFeatureDataset
    return SequenceFeatureDataset(
        lnc_fasta          = config.get("data", "lnc_fasta"),
        pc_fasta           = config.get("data", "pc_fasta"),
        te_features_csv    = config.get("data", "te_features_csv"),
        nonb_features_csv  = config.get("data", "nonb_features_csv"),
        te_scaler_path     = config.get("data", "te_scaler"),
        nonb_scaler_path   = config.get("data", "nonb_scaler"),
        max_length         = config.get("model", "max_length"),
    )


def build_dataset_current(config):
    from data.gated_feature_dataset import SequenceFeatureDataset
    nonb2_csv    = config.get("data", "nonb2_csv", default=None)
    nonb2_scaler = config.get("data", "nonb2_scaler_bank_path", default=None)
    return SequenceFeatureDataset(
        lnc_fasta               = config.get("data", "lnc_fasta"),
        pc_fasta                = config.get("data", "pc_fasta"),
        te_genomic_csv          = config.get("data", "te_genomic_csv"),
        te_processed_csv        = config.get("data", "te_processed_csv",   default=None),
        nonb_genomic_csv        = config.get("data", "nonb_genomic_csv"),
        nonb_processed_csv      = config.get("data", "nonb_processed_csv", default=None),
        nonb2_csv               = nonb2_csv,
        te_scaler_bank_path     = config.get("data", "te_scaler_bank_path"),
        nonb_scaler_bank_path   = config.get("data", "nonb_scaler_bank_path"),
        nonb2_scaler_bank_path  = nonb2_scaler,
        max_length              = config.get("model", "max_length"),
    )


def build_fwd_kwargs_legacy(batch, device):
    return dict(
        te_features   = batch["te_features"].to(device),
        nonb_features = batch["nonb_features"].to(device),
    )


def build_fwd_kwargs_current(batch, device, has_nonb2):
    kwargs = dict(
        te_genomic     = batch["te_genomic"].to(device),
        te_processed   = batch["te_processed"].to(device),
        nonb_genomic   = batch["nonb_genomic"].to(device),
        nonb_processed = batch["nonb_processed"].to(device),
    )
    if has_nonb2 and "nonb2" in batch:
        kwargs["nonb2"] = batch["nonb2"].to(device)
    return kwargs


# ---------------------------------------------------------------------------
# Attention capture, per architecture
# ---------------------------------------------------------------------------

class AttentionCache:
    """Forward hook capturing the second element of CrossModalAttentionMH's
    (out, attn_weights) tuple return, for the current architecture."""

    def __init__(self):
        self.attn_weights = None
        self._handle = None

    def register(self, model):
        self._handle = model.cross_attn.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        # output is (out, attn_weights)
        self.attn_weights = output[1].detach()

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def extract_fold_legacy(model, loader, device) -> np.ndarray:
    """Returns (n_transcripts, 2) — mean-over-L attention to [REP, NonB]."""
    all_weights = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="    Extracting (legacy)", leave=False):
            seq = batch["sequence"].to(device)
            fwd = build_fwd_kwargs_legacy(batch, device)
            out = model(seq, deterministic=True, **fwd)
            attn = out["attn_weights"]              # (B, L_encoded, 2)
            attn_mean = attn.mean(dim=1)             # (B, 2)
            all_weights.append(attn_mean.cpu().numpy())
    return np.concatenate(all_weights, axis=0)


def extract_fold_current(model, loader, device, has_nonb2) -> np.ndarray:
    """Returns (n_transcripts, N=20) — mean-over-heads-and-L attention
    to each subgroup token, in REGISTRY.all_subgroups order."""
    cache = AttentionCache()
    cache.register(model)

    all_weights = []
    model.eval()
    try:
        with torch.no_grad():
            for batch in tqdm(loader, desc="    Extracting (current)", leave=False):
                seq = batch["sequence"].to(device)
                fwd = build_fwd_kwargs_current(batch, device, has_nonb2)
                model(seq, deterministic=True, **fwd)
                attn = cache.attn_weights             # (B, H, L, N)
                attn_mean = attn.mean(dim=(1, 2))      # (B, N) — mean over heads, L
                all_weights.append(attn_mean.cpu().numpy())
    finally:
        cache.remove()

    return np.concatenate(all_weights, axis=0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_attention_crossfold(
    cross_fold_df: pd.DataFrame,   # subgroup, mean_attn, std_attn, n_folds
    output_path:   Path,
    block_map:     dict = None,
    fig_tag:       str = "",
) -> None:
    """Cross-fold bar chart of mean attention weight per subgroup/modality.
    Block-colored and boundary-lined when block_map is provided (current
    architecture); plain single-color bars otherwise (legacy architecture,
    N=2, no block structure to show)."""
    df = cross_fold_df.sort_values("mean_attn", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(13, 5) if len(df) > 4 else (6, 4))
    x = np.arange(len(df))
    names = df["subgroup"].tolist()

    if block_map:
        colors = _block_colors_for(names, block_map)
    else:
        colors = "#4A90D9"

    ax.bar(x, df["mean_attn"], color=colors, edgecolor="black",
           linewidth=0.6, alpha=0.9)
    ax.errorbar(x, df["mean_attn"], yerr=df["std_attn"],
                fmt="none", color="black", capsize=3, linewidth=1.2)

    if block_map:
        for boundary in _block_boundaries(names, block_map):
            ax.axvline(x=boundary, color="gray", linestyle="--",
                       linewidth=1.5, alpha=0.6)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    labels = display_subgroups(names) if block_map else names
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Mean attention weight, cross-fold ± std", fontsize=11)
    ax.set_title(
        f"{fig_tag}Layer 1 — Attention Weights — Cross-Fold\n"
        f"(mean ± std over {int(df['n_folds'].iloc[0])} folds)",
        fontsize=13, fontweight="bold"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if block_map:
        ax.legend(handles=_block_legend_handles(names, block_map), fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Layer 1 attention extraction — supports both legacy "
                    "(beta_vae_features_attn, 2-modality) and current "
                    "(beta_vae_subgroup, 20-subgroup) architectures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--experiment_dir", required=True, type=Path,
                        help="Experiment directory containing model_paths.csv")
    parser.add_argument("--config",         required=True)
    parser.add_argument("--output_dir",     required=True, type=Path)
    parser.add_argument("--repo_root",      type=Path, default=Path("."),
                        help="Root to resolve model_paths.csv's relative "
                             "checkpoint paths against (default: cwd)")
    parser.add_argument("--batch_size",     type=int, default=256)
    parser.add_argument("--device",
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    architecture_kind = detect_architecture(config)
    arch_name = config.get("model", "architecture")

    print("=" * 70)
    print("Layer 1 — Attention Extraction")
    print("=" * 70)
    print(f"Experiment    : {args.experiment_dir}")
    print(f"Architecture  : {arch_name}  ({architecture_kind})")
    print(f"Output dir    : {args.output_dir}")
    if architecture_kind == "legacy":
        print("Legacy architecture: attention is over 2 modalities (REP, NonB), N=2.")
    print("=" * 70)

    # ── Dataset ──────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    if architecture_kind == "legacy":
        dataset = build_dataset_legacy(config)
        subgroup_names = ["TE", "NonB"]
        has_nonb2 = False
    else:
        dataset = build_dataset_current(config)
        subgroup_names = list(REGISTRY.all_subgroups)
        has_nonb2 = config.get("data", "nonb2_csv", default=None) is not None
        print(f"Registry: {len(subgroup_names)} subgroups: {subgroup_names}")
    print(f"Dataset: {len(dataset):,} samples")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0)

    # ── Checkpoints ──────────────────────────────────────────────────────
    manifest = load_model_paths(args.experiment_dir)
    print(f"\nFound {len(manifest)} fold checkpoint(s) in model_paths.csv")

    model_builder = create_model_builder(config)

    all_summary_rows = []

    for _, row in manifest.iterrows():
        fold_idx = int(row["fold"])
        ckpt_path = resolve_checkpoint_path(row["path"], args.repo_root)

        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx}  ({ckpt_path.name})  val_acc={row.get('val_acc', 'n/a')}")
        print(f"{'='*60}")

        out_path = args.output_dir / f"fold_{fold_idx}_attn_weights.npz"
        if out_path.exists():
            print(f"  Already extracted — loading from {out_path}")
            weights = np.load(out_path)["attn_weights"]
        else:
            ckpt = torch.load(ckpt_path, map_location=device)
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

            if architecture_kind == "legacy":
                weights = extract_fold_legacy(model, loader, device)
            else:
                weights = extract_fold_current(model, loader, device, has_nonb2)

            np.savez_compressed(out_path, attn_weights=weights,
                                subgroup_names=np.array(subgroup_names))
            print(f"  Saved raw weights -> {out_path}  (shape {weights.shape})")

            del model
            torch.cuda.empty_cache()

        # Per-fold summary
        fold_mean = weights.mean(axis=0)
        fold_std  = weights.std(axis=0)
        for sg, m, s in zip(subgroup_names, fold_mean, fold_std):
            all_summary_rows.append({
                "fold": fold_idx, "subgroup": sg,
                "mean_attn": m, "std_attn": s,
                "n_transcripts": weights.shape[0],
            })

    # ── Summaries ────────────────────────────────────────────────────────
    per_fold_df = pd.DataFrame(all_summary_rows)
    per_fold_path = args.output_dir / "attn_summary_per_fold.csv"
    per_fold_df.to_csv(per_fold_path, index=False)
    print(f"\nPer-fold summary -> {per_fold_path}")

    cross_fold_df = (
        per_fold_df.groupby("subgroup")
        .agg(mean_attn=("mean_attn", "mean"),
             std_attn=("mean_attn", "std"),
             n_folds=("fold", "nunique"))
        .reset_index()
        .sort_values("mean_attn", ascending=False)
    )
    cross_fold_path = args.output_dir / "attn_summary_cross_fold.csv"
    cross_fold_df.to_csv(cross_fold_path, index=False)

    print(f"\n{'='*70}")
    print(f"CROSS-FOLD ATTENTION SUMMARY  ({architecture_kind})")
    print(f"{'='*70}")
    for _, r in cross_fold_df.iterrows():
        print(f"  {r['subgroup']:<12} mean={r['mean_attn']:.4f}  "
              f"std={r['std_attn']:.4f}  (n_folds={r['n_folds']})")
    print(f"\nSaved -> {cross_fold_path}")

    if architecture_kind == "current":
        block_map = {sg: REGISTRY.token_block(sg) for sg in subgroup_names}
    else:
        block_map = None
    plot_attention_crossfold(
        cross_fold_df,
        args.output_dir / "attn_summary_cross_fold.png",
        block_map=block_map,
    )

    if architecture_kind == "legacy":
        print("\nLegacy output is 2-modality (REP/NonB), not per-subgroup.")


if __name__ == "__main__":
    main()