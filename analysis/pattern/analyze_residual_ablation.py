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
comparison scripts and plots.

Usage
-----
python analysis/pattern/analyze_residual_ablation.py \
    --experiment_dir gencode_v49_experiments/beta_vae_subgroup_base_g49 \
    --repr_dir       gencode_v49_experiments/beta_vae_subgroup_base_g49/representations \
    --config         configs/beta_vae_subgroup_base_g49.json \
    --output_dir     gencode_v49_experiments/beta_vae_subgroup_base_g49/residual_ablation \
    --model_label    "β-VAE" \
    --gencode_version v49 \
    --device         cuda:0
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
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
from data.feature_registry import (REGISTRY, display_block,
                                    display_subgroup, display_subgroups)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100


# ---------------------------------------------------------------------------
# Block colour palette (shared convention with analyze_mstar.py / latent probing)
# ---------------------------------------------------------------------------

BLOCK_COLORS = {
    "nonb":  "#4A90D9",
    "te":    "#F4B942",
    "nonb2": "#9B7FD4",
}


def _sg_color(sg: str) -> str:
    return BLOCK_COLORS.get(REGISTRY.token_block(sg), "#999999")


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
# Visualisation
# ---------------------------------------------------------------------------

def plot_ablation_results(
    df:          pd.DataFrame,
    output_path: Path,
    fig_tag:     str = "",
) -> None:
    mean_drop = (df.groupby("subgroup")["acc_drop"]
                 .mean().sort_values(ascending=False))
    subgroups_ordered = mean_drop.index.tolist()

    x     = np.arange(len(subgroups_ordered))
    drops = [df[df["subgroup"] == sg]["acc_drop"].values[0]
             for sg in subgroups_ordered]
    colors = [_sg_color(sg) for sg in subgroups_ordered]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x, drops, color=colors, edgecolor="black", linewidth=0.6, alpha=0.9)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(display_subgroups(subgroups_ordered), rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Accuracy drop (baseline − ablated)", fontsize=12)
    ax.set_title(
        f"{fig_tag}Residual Token Ablation — Classification Accuracy Drop\n"
        "(z-orthogonal component zeroed — sequence-independent signal removed)",
        fontsize=13, fontweight="bold"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    seen_blocks = list(dict.fromkeys(REGISTRY.token_block(sg) for sg in subgroups_ordered))
    ax.legend(handles=[
        Patch(facecolor=BLOCK_COLORS.get(b, "#999999"), label=display_block(b))
        for b in seen_blocks
    ], fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cross_fold_summary(
    summary_df:  pd.DataFrame,
    output_path: Path,
    fig_tag:     str = "",
) -> None:
    pivot = summary_df.pivot(index="subgroup", columns="fold", values="acc_drop")
    mean  = pivot.mean(axis=1).sort_values(ascending=False)
    std   = pivot.std(axis=1).reindex(mean.index)

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(mean))
    colors = [_sg_color(sg) for sg in mean.index]
    ax.bar(x, mean.values, color=colors, edgecolor="black",
           linewidth=0.6, alpha=0.9)
    ax.errorbar(x, mean.values, yerr=std.values,
                fmt="none", color="black", capsize=3, linewidth=1.2)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(display_subgroups(list(mean.index)), rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Mean accuracy drop (mean ± std over folds)", fontsize=11)
    ax.set_title(
        f"{fig_tag}Cross-Fold Residual Ablation\n"
        "Sub-groups ranked by mean independent contribution",
        fontsize=13, fontweight="bold"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_comparison(
    feature_zero_df: pd.DataFrame,
    residual_df:     pd.DataFrame,
    output_path:     Path,
    fig_tag:         str = "",
) -> None:
    """
    Side-by-side comparison: feature_zero vs residual_zero cross-fold means,
    with a Benjamini-Hochberg FDR-corrected paired t-test per subgroup
    (feature_zero acc_drop vs residual_zero acc_drop, matched by fold).
    """
    from scipy import stats

    fz_pivot = (feature_zero_df[feature_zero_df["mode"] == "feature_zero"]
                .pivot(index="subgroup", columns="fold", values="acc_drop"))
    rz_pivot = (residual_df.pivot(index="subgroup", columns="fold", values="acc_drop"))

    fz = fz_pivot.mean(axis=1).rename("feature_zero")
    rz = rz_pivot.mean(axis=1).rename("residual_zero")
    fz_std = fz_pivot.std(axis=1).rename("fz_std")
    rz_std = rz_pivot.std(axis=1).rename("rz_std")

    combined = pd.concat([fz, rz, fz_std, rz_std], axis=1).fillna(0)
    combined = combined.sort_values("residual_zero", ascending=False)

    common_subgroups = [sg for sg in combined.index
                        if sg in fz_pivot.index and sg in rz_pivot.index]

    raw_pvals = {}
    for sg in common_subgroups:
        fz_vals = fz_pivot.loc[sg].dropna()
        rz_vals = rz_pivot.loc[sg].dropna()
        common_folds = fz_vals.index.intersection(rz_vals.index)
        if len(common_folds) > 1:
            raw_pvals[sg] = stats.ttest_rel(fz_vals[common_folds],
                                            rz_vals[common_folds]).pvalue
        else:
            raw_pvals[sg] = np.nan
    raw_pval_series = pd.Series(raw_pvals).reindex(combined.index)

    valid = raw_pval_series.notna()
    qval_series = pd.Series(np.nan, index=raw_pval_series.index)
    if valid.sum() > 0:
        qval_series[valid] = stats.false_discovery_control(
            raw_pval_series[valid].values, method='bh'
        )
    sig = qval_series < 0.05

    x     = np.arange(len(combined))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, combined["feature_zero"], width,
           label="Feature zero (all content)",
           color="#F4B942", edgecolor="black", linewidth=0.6, alpha=0.9)
    ax.errorbar(x - width/2, combined["feature_zero"],
                yerr=combined["fz_std"],
                fmt="none", color="black", capsize=3, linewidth=1.0)
    ax.bar(x + width/2, combined["residual_zero"], width,
           label="Residual zero (z-orthogonal only)",
           color="#9B59B6", edgecolor="black", linewidth=0.6, alpha=0.9)
    ax.errorbar(x + width/2, combined["residual_zero"],
                yerr=combined["rz_std"],
                fmt="none", color="black", capsize=3, linewidth=1.0)

    combined_vals = combined[["feature_zero", "residual_zero"]].values
    y_range = combined_vals.max() - combined_vals.min()
    fixed_offset = 0.06 * y_range
    for xi, m1, s1, m2, s2, is_sig in zip(
        x, combined["feature_zero"], combined["fz_std"],
        combined["residual_zero"], combined["rz_std"], sig.values,
    ):
        if is_sig:
            top = max(m1 + s1, m2 + s2)
            y_star = top + fixed_offset
            ax.annotate('*', (xi, y_star), ha='center', va='bottom',
                        fontsize=14, fontweight='bold')

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.08 * y_range)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(display_subgroups(list(combined.index)), rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Mean accuracy drop (cross-fold)", fontsize=12)
    ax.set_title(
        f"{fig_tag}Feature Zero vs Residual Zero — Cross-Fold Comparison\n"
        f"(residual_zero isolates sequence-independent contribution)  "
        f"(* q < 0.05, Benjamini-Hochberg FDR paired t-test, n={len(common_subgroups)})",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

    stats_out = output_path.parent / "feature_vs_residual_comparison_stats.csv"
    stats_df = pd.DataFrame({
        "subgroup": combined.index,
        "feature_zero_mean": combined["feature_zero"].values,
        "residual_zero_mean": combined["residual_zero"].values,
        "p_value_paired": raw_pval_series.values,
        "q_value_bh": qval_series.values,
        "significant_fdr": sig.values,
    })
    stats_df.to_csv(stats_out, index=False)
    print(f"  Saved: {stats_out}")


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
    parser.add_argument("--model_label",     default="β-VAE Standard")
    parser.add_argument("--gencode_version", default="v49")
    parser.add_argument("--batch_size",     type=int, default=256)
    parser.add_argument("--device",
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feature_zero_csv", default=None,
                        help="Path to all_folds_ablation.csv from feature_zero "
                             "ablation for comparison plot (optional)")
    args = parser.parse_args()

    config     = load_config(args.config)
    exp_dir    = Path(args.experiment_dir)
    repr_dir   = Path(args.repr_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device     = torch.device(args.device)

    tag_parts = [p for p in [args.model_label,
                 f"GENCODE {args.gencode_version}"] if p]
    fig_tag   = " | ".join(tag_parts) + " — " if tag_parts else ""

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
        plot_ablation_results(
            fold_df,
            fold_out / "residual_ablation_bar.png",
            fig_tag=f"{fig_tag}fold_{fold_idx} — "
        )

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

    plot_cross_fold_summary(
        combined,
        output_dir / "cross_fold_residual_ablation.png",
        fig_tag=fig_tag
    )

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

    # ── Comparison plot with feature_zero ─────────────────────────────────────
    if args.feature_zero_csv and Path(args.feature_zero_csv).exists():
        print("\nGenerating feature_zero vs residual_zero comparison plot...")
        fz_df = pd.read_csv(args.feature_zero_csv)
        plot_comparison(
            fz_df, combined,
            output_dir / "feature_vs_residual_comparison.png",
            fig_tag=fig_tag
        )
    else:
        print("\n  --feature_zero_csv not provided — skipping comparison plot")
        print("  To generate: rerun with "
              "--feature_zero_csv path/to/all_folds_ablation.csv")

    print(f"\nAll outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()