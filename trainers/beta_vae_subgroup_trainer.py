"""
trainers/beta_vae_subgroup_trainer.py

Trainer for BetaVAESubgroup.

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

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

from data.cv_utils import create_length_stratified_groups
from data.feature_registry import REGISTRY


# ---------------------------------------------------------------------------
# Forward-call helper — resolves nonb2 (and any future block) from batch
# ---------------------------------------------------------------------------

def _forward_kwargs(batch: dict, device: torch.device) -> dict:
    """
    Build model.forward() kwargs from a dataset batch, handling the
    fixed nonb/te keys plus any optional dynamic blocks (e.g. nonb2).

    Only includes keys that are actually present in the batch, so models
    built against a registry without nonb2 still work against batches
    that happen to include it (and vice versa is handled by the model's
    own validation).
    """
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
# Orthogonality loss helper (unchanged)
# ---------------------------------------------------------------------------

def orthogonality_loss(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    Penalise cosine similarity between attention distributions across heads.
    """
    h_dist = attn_weights.mean(dim=(0, 2))   # (H, N)
    h_norm = F.normalize(h_dist, p=2, dim=-1)
    sim_matrix = torch.matmul(h_norm, h_norm.t())
    H = sim_matrix.shape[0]
    mask = ~torch.eye(H, dtype=torch.bool, device=sim_matrix.device)
    off_diag = sim_matrix[mask]
    return off_diag.mean()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class BetaVAESubgroupLoss:
    """
    Combined loss for BetaVAESubgroup.

    lambda_feat is a dict {block_name: weight}. Blocks not present default
    to 0.0 (auxiliary head computed but not added to total loss — still
    useful for monitoring, controlled by compute_unweighted_aux).
    """

    def __init__(
        self,
        alpha:                float = 0.001,
        beta:                 float = 4.0,
        gamma_classification: float = 1.0,
        lambda_recon:         float = 1.0,
        reconstruction_loss:  str   = "mse",
        class_weights:        Optional[torch.Tensor] = None,
        gamma_attn:           float = 0.0,
        lambda_feat:          Optional[Dict[str, float]] = None,
        lambda_conc:          float = 0.1,
        lambda_ortho:         float = 0.0,
    ) -> None:
        self.alpha                = alpha
        self.beta                 = beta
        self.gamma_classification = gamma_classification
        self.lambda_recon         = lambda_recon
        self.reconstruction_loss  = reconstruction_loss
        self.class_weights        = class_weights
        self.gamma_attn           = gamma_attn
        # Default: every registry block gets lambda 0.0 unless overridden
        self.lambda_feat: Dict[str, float] = {
            b: 0.0 for b in REGISTRY.block_names
        }
        if lambda_feat:
            self.lambda_feat.update(lambda_feat)
        self.lambda_conc  = lambda_conc
        self.lambda_ortho = lambda_ortho

    def _recon_loss(self, x_recon: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        if x_recon.shape != x_true.shape:
            return torch.tensor(0.0, device=x_recon.device, requires_grad=False)
        if self.reconstruction_loss == "bce":
            return F.binary_cross_entropy(x_recon, x_true, reduction="mean")
        return F.mse_loss(x_recon, x_true, reduction="mean")

    def _kl(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return (-0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=1
        )).mean() / mu.size(1)

    def _cls(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return (
            F.cross_entropy(logits, labels, weight=self.class_weights)
            if self.class_weights is not None
            else F.cross_entropy(logits, labels)
        )

    def __call__(
        self,
        x_true:       torch.Tensor,
        x_recon:      torch.Tensor,
        mu:           torch.Tensor,
        logvar:       torch.Tensor,
        logits:       torch.Tensor,
        labels:       torch.Tensor,
        attn_weights: Optional[torch.Tensor]              = None,
        feat_logits:  Optional[Dict[str, torch.Tensor]]   = None,
    ):
        recon = self._recon_loss(x_recon, x_true)
        kl    = self._kl(mu, logvar)
        cls   = self._cls(logits, labels)

        total = (self.alpha * self.lambda_recon * recon
                 + self.beta * kl
                 + self.gamma_classification * cls)

        # ── Auxiliary feature heads — one per registry block ────────────────
        feat_losses: Dict[str, Optional[torch.Tensor]] = {}
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

        # ── Entropy regularisation ───────────────────────────────────────────
        entropy_val = None
        if attn_weights is not None and self.gamma_attn > 0:
            aw          = attn_weights.mean(dim=1)
            entropy_val = -(aw * torch.log(aw + 1e-8)).sum(-1).mean()
            total       = total - self.gamma_attn * entropy_val

        # ── Concentration loss ───────────────────────────────────────────────
        conc_val = None
        if attn_weights is not None and self.lambda_conc > 0:
            aw_mean  = attn_weights.mean(dim=(1, 2))
            top3     = aw_mean.topk(k=min(3, aw_mean.shape[-1]), dim=-1).values
            conc_val = top3.sum(-1).mean()
            total    = total - self.lambda_conc * conc_val

        # ── Orthogonality loss ───────────────────────────────────────────────
        ortho_val = None
        if attn_weights is not None and self.lambda_ortho > 0:
            ortho_val = orthogonality_loss(attn_weights)
            total     = total + self.lambda_ortho * ortho_val

        loss_dict = {
            "loss":           total.item(),
            "reconstruction": recon.item(),
            "kl":             kl.item(),
            "classification": cls.item(),
            "attn_entropy":   entropy_val.item() if entropy_val is not None else None,
            "conc":           conc_val.item()    if conc_val    is not None else None,
            "ortho":          ortho_val.item()   if ortho_val   is not None else None,
        }
        # Per-block auxiliary loss entries: feat_{block_name}
        for block_name, loss_val in feat_losses.items():
            loss_dict[f"feat_{block_name}"] = (
                loss_val.item() if loss_val is not None else None
            )

        return total, loss_dict


# ---------------------------------------------------------------------------
# SingleFoldSubgroupTrainer
# ---------------------------------------------------------------------------

class SingleFoldSubgroupTrainer:
    """Single-fold trainer for BetaVAESubgroup."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        learning_rate:        float = 1e-4,
        weight_decay:         float = 1e-5,
        alpha:                float = 0.001,
        beta:                 float = 1.0,
        gamma_classification: float = 1.0,
        lambda_recon:         float = 1.0,
        reconstruction_loss:  str   = "mse",
        class_weights         = None,
        gamma_attn:           float = 0.0,
        lambda_feat:          Optional[Dict[str, float]] = None,
        lambda_conc:          float = 0.1,
        lambda_ortho:         float = 0.0,
        kl_anneal_epochs:     int   = 20,
        kl_anneal_end:        float = 4.0,
        device                = None,
    ) -> None:

        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = (torch.device(device) if device
                             else torch.device("cuda" if torch.cuda.is_available()
                                               else "cpu"))
        self.model.to(self.device)

        self.kl_anneal_start  = 0.0
        self.kl_anneal_end    = kl_anneal_end
        self.kl_anneal_epochs = kl_anneal_epochs

        cw = (torch.FloatTensor(class_weights).to(self.device)
              if class_weights is not None else None)

        self.criterion = BetaVAESubgroupLoss(
            alpha=alpha, beta=beta,
            gamma_classification=gamma_classification,
            lambda_recon=lambda_recon,
            reconstruction_loss=reconstruction_loss,
            class_weights=cw,
            gamma_attn=gamma_attn,
            lambda_feat=lambda_feat,
            lambda_conc=lambda_conc,
            lambda_ortho=lambda_ortho,
        )

        # ── LR groups ─────────────────────────────────────────────────────────
        # Projectors live under model.projectors.{block_name}.* now (ModuleDict)
        # rather than the old nonb_proj/te_proj attribute names.
        attn_params  = [p for n, p in model.named_parameters() if "cross_attn" in n]
        proj_params  = [p for n, p in model.named_parameters()
                        if n.startswith("projectors.") and "cross_attn" not in n]
        other_params = [p for n, p in model.named_parameters()
                        if "cross_attn" not in n and not n.startswith("projectors.")]

        n_proj = sum(p.numel() for p in proj_params)
        n_attn = sum(p.numel() for p in attn_params)
        n_other = sum(p.numel() for p in other_params)
        print(f"  LR groups: other={n_other:,}  proj={n_proj:,}  attn={n_attn:,}")
        if n_proj == 0:
            print("    WARNING: proj_params is empty — check model.projectors "
                  "naming matches 'projectors.' prefix")

        self.optimizer = torch.optim.Adam([
            {"params": other_params, "lr": learning_rate},
            {"params": proj_params,  "lr": learning_rate * 2},
            {"params": attn_params,  "lr": learning_rate},
        ], weight_decay=weight_decay)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7
        )

        # History keys built dynamically per registry block
        self.history: Dict[str, List] = {
            "train_loss": [], "train_reconstruction": [], "train_kl": [],
            "train_classification": [], "val_loss": [], "val_acc": [],
            "attn_entropy": [], "train_conc": [], "train_ortho": [],
        }
        for block_name in REGISTRY.block_names:
            self.history[f"train_feat_{block_name}"] = []

        self._check_attention_gradients(train_loader)

    # ── gradient health check ────────────────────────────────────────────────

    def _check_attention_gradients(self, train_loader, n_batches: int = 5) -> None:
        print("\n--- Attention gradient check (first 5 batches) ---")
        self.model.train()
        qp_init = self.model.cross_attn.query_proj.weight.data.clone()

        for i, batch in enumerate(train_loader):
            if i >= n_batches:
                break
            seq    = batch["sequence"].to(self.device)
            labels = batch["label"].to(self.device)
            fwd_kw = _forward_kwargs(batch, self.device)

            self.optimizer.zero_grad()
            out = self.model(seq, **fwd_kw)

            feat_logits = {
                b: out.get(f"feat_logits_{b}") for b in REGISTRY.block_names
            }
            loss, _ = self.criterion(
                seq, out["reconstruction"], out["mu"], out["logvar"],
                out["logits"], labels,
                attn_weights=out["attn_weights"],
                feat_logits=feat_logits,
            )
            loss.backward()

            grad = self.model.cross_attn.query_proj.weight.grad
            if grad is None:
                raise RuntimeError("query_proj.weight.grad is None — attention disconnected.")
            grad_std = grad.std().item()
            print(f"  Batch {i}: query_proj grad std = {grad_std:.2e}")
            if grad_std < 1e-7:
                raise RuntimeError(f"query_proj gradient vanishing (std={grad_std:.2e}).")
            self.optimizer.step()

        delta = (self.model.cross_attn.query_proj.weight.data - qp_init).std().item()
        print(f"  query_proj delta after {n_batches} steps: std={delta:.2e}")
        if delta < 1e-7:
            raise RuntimeError(f"query_proj weights did not move (delta={delta:.2e}).")
        print("   Attention gradients healthy.\n")

    # ── KL / entropy annealing ────────────────────────────────────────────────

    def get_current_beta(self, epoch: int) -> float:
        if epoch >= self.kl_anneal_epochs:
            return self.kl_anneal_end
        return self.kl_anneal_start + (epoch / self.kl_anneal_epochs) * (
            self.kl_anneal_end - self.kl_anneal_start)

    def get_current_gamma_attn(self, epoch: int) -> float:
        base = self.criterion.gamma_attn
        if base == 0.0:
            return 0.0
        if epoch >= self.kl_anneal_epochs:
            return base * 0.1
        return base * (1.0 - 0.9 * (epoch / self.kl_anneal_epochs))

    # ── train / eval ─────────────────────────────────────────────────────────

    def train_epoch(self, epoch_num: int) -> Dict:
        self.model.train()

        tracked_keys = ["loss", "reconstruction", "kl", "classification",
                        "conc", "ortho"] + [f"feat_{b}" for b in REGISTRY.block_names]
        totals       = {k: 0.0 for k in tracked_keys}
        totals_count = {k: 0   for k in tracked_keys}
        attn_entropy_batches = []

        self.criterion.beta       = self.get_current_beta(epoch_num)
        self.criterion.gamma_attn = self.get_current_gamma_attn(epoch_num)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_num} [Train]", leave=False)
        for batch in pbar:
            seq    = batch["sequence"].to(self.device)
            labels = batch["label"].to(self.device)
            fwd_kw = _forward_kwargs(batch, self.device)

            self.optimizer.zero_grad()
            out = self.model(seq, **fwd_kw)

            feat_logits = {
                b: out.get(f"feat_logits_{b}") for b in REGISTRY.block_names
            }
            loss, loss_dict = self.criterion(
                seq, out["reconstruction"], out["mu"], out["logvar"],
                out["logits"], labels,
                attn_weights=out["attn_weights"],
                feat_logits=feat_logits,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for k in totals:
                v = loss_dict.get(k)
                if v is not None:
                    totals[k]       += v
                    totals_count[k] += 1

            with torch.no_grad():
                aw      = out["attn_weights"].mean(dim=1)
                entropy = -(aw * torch.log(aw + 1e-8)).sum(-1).mean().item()
                attn_entropy_batches.append(entropy)

            pbar.set_postfix({k: f"{v:.4f}" for k, v in loss_dict.items()
                              if v is not None})

        self.history["attn_entropy"].append(float(np.mean(attn_entropy_batches)))
        return {k: (totals[k] / totals_count[k] if totals_count[k] > 0 else None)
                for k in totals}

    def evaluate(self, epoch_num: int) -> Dict:
        self.model.eval()
        all_logits, all_labels = [], []
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {epoch_num} [Val]",
                              leave=False):
                seq    = batch["sequence"].to(self.device)
                labels = batch["label"].to(self.device)
                fwd_kw = _forward_kwargs(batch, self.device)

                out = self.model(seq, deterministic=True, **fwd_kw)
                total_loss += F.cross_entropy(out["logits"], labels).item()
                all_logits.append(out["logits"].cpu())
                all_labels.append(labels.cpu())

        preds = torch.cat(all_logits).argmax(dim=1)
        acc   = accuracy_score(torch.cat(all_labels).numpy(), preds.numpy())
        return {"loss": total_loss / len(self.val_loader), "accuracy": acc}

    def train(
        self,
        num_epochs:              int,
        early_stopping_patience: int   = 10,
        save_path                      = None,
    ):
        best_val_loss    = float("inf")
        patience_counter = 0

        print(f"\nTraining {num_epochs} epochs (patience={early_stopping_patience})")

        for epoch in range(1, num_epochs + 1):
            train_m = self.train_epoch(epoch)
            val_m   = self.evaluate(epoch)

            self.history["train_loss"].append(train_m["loss"])
            self.history["train_reconstruction"].append(train_m["reconstruction"])
            self.history["train_kl"].append(train_m["kl"])
            self.history["train_classification"].append(train_m["classification"])
            self.history["val_loss"].append(val_m["loss"])
            self.history["val_acc"].append(val_m["accuracy"])
            for aux_key in (["conc", "ortho"]
                            + [f"feat_{b}" for b in REGISTRY.block_names]):
                v = train_m.get(aux_key)
                if v is not None:
                    self.history[f"train_{aux_key}"].append(v)

            epoch_entropy = self.history["attn_entropy"][-1]
            max_entropy   = float(np.log(self.model.n_tokens))
            entropy_pct   = 100 * epoch_entropy / max_entropy

            # Build aux string for non-None terms — dynamic per block
            aux_parts = []
            for block_name in REGISTRY.block_names:
                v = train_m.get(f"feat_{block_name}")
                if v is not None:
                    short = block_name[:4]  # nonb→nonb, te→te, nonb2→nonb
                    aux_parts.append(f"f_{block_name}={v:.3f}")
            for key, short in [("conc", "conc"), ("ortho", "ortho")]:
                v = train_m.get(key)
                if v is not None:
                    aux_parts.append(f"{short}={v:.3f}")
            aux_str = (" " + " ".join(aux_parts)) if aux_parts else ""

            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train: {train_m['loss']:.4f} "
                  f"(recon={train_m['reconstruction']:.4f} "
                  f"kl={train_m['kl']:.4f} "
                  f"cls={train_m['classification']:.4f}{aux_str}) | "
                  f"Val: {val_m['loss']:.4f} acc={val_m['accuracy']:.4f} | "
                  f"AttnH: {epoch_entropy:.3f} ({entropy_pct:.0f}% max)")

            if epoch <= 10 and entropy_pct < 20:
                print(f"   COLLAPSE WARNING: entropy={entropy_pct:.0f}% of max.")

            old_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_m["loss"])
            new_lr = self.optimizer.param_groups[0]["lr"]
            if new_lr != old_lr:
                print(f"  → LR: {old_lr:.2e} → {new_lr:.2e}")

            if val_m["loss"] < best_val_loss:
                best_val_loss    = val_m["loss"]
                patience_counter = 0
                if save_path:
                    torch.save({
                        "model_state_dict":     self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epoch":    epoch,
                        "val_loss": best_val_loss,
                        "val_acc":  val_m["accuracy"],
                        "history":  self.history,
                    }, save_path)
                    print(f"  → Saved checkpoint → {save_path}")
                print(f"  → New best val loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"  → Patience: {patience_counter}/{early_stopping_patience}")
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        return self.model, {
            "val_loss":   best_val_loss,
            "val_acc":    max(self.history["val_acc"]),
            "train_loss": min(self.history["train_loss"]),
        }, self.history


# ---------------------------------------------------------------------------
# BetaVAESubgroupTrainer  (cross-validation)
# ---------------------------------------------------------------------------

class BetaVAESubgroupTrainer:
    """Cross-validation trainer for BetaVAESubgroup."""

    def __init__(self, model_builder, dataset, config, n_folds: int = 5,
                 device=None) -> None:
        self.model_builder = model_builder
        self.dataset       = dataset
        self.config        = config
        self.n_folds       = n_folds
        self.device        = (torch.device(device) if device
                              else torch.device("cuda" if torch.cuda.is_available()
                                                else "cpu"))
        self.fold_results  = []
        self.model_paths   = []
        print(f"BetaVAESubgroupTrainer | {n_folds} folds | device={self.device}")
        print(f"Registry blocks: {REGISTRY.block_names} "
              f"({REGISTRY.total_tokens} total tokens)")

    @staticmethod
    def _lambda_feat_from_config(config) -> Dict[str, float]:
        """Read lambda_feat_{block_name} for every registry block."""
        return {
            b: config.get("training", f"lambda_feat_{b}", default=0.0)
            for b in REGISTRY.block_names
        }

    # ── fold training ─────────────────────────────────────────────────────────

    def train_fold(self, fold_idx, train_dataset, val_dataset, class_weights=None):
        print(f"\n{'='*60}\nFold {fold_idx + 1}/{self.n_folds}\n{'='*60}")

        train_ids = {self.dataset.sequences[i].id.split("|")[0]
                     for i in train_dataset.indices}
        val_ids   = {self.dataset.sequences[i].id.split("|")[0]
                     for i in val_dataset.indices}
        overlap   = train_ids & val_ids
        print(f"Train={len(train_ids)} Val={len(val_ids)} Overlap={len(overlap)}")
        if overlap:
            print(f"  WARNING: {len(overlap)} overlapping transcripts!")

        bs  = self.config.get("training", "batch_size")
        nw  = self.config.get("training", "num_workers", default=1)

        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                                  num_workers=nw, pin_memory=True, drop_last=True)
        val_loader   = DataLoader(val_dataset,   batch_size=bs, shuffle=False,
                                  num_workers=nw, pin_memory=True)

        save_dir = Path(self.config.get("output", "experiment_name")) / "models"
        save_dir.mkdir(exist_ok=True, parents=True)
        model_save_path = save_dir / f"fold_{fold_idx}_best.pt"

        model   = self.model_builder()
        trainer = SingleFoldSubgroupTrainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            learning_rate        = self.config.get("training", "learning_rate"),
            weight_decay         = self.config.get("training", "weight_decay"),
            alpha                = self.config.get("training", "alpha"),
            beta                 = self.config.get("training", "beta"),
            gamma_classification = self.config.get("training", "gamma_classification"),
            lambda_recon         = self.config.get("training", "lambda_recon"),
            reconstruction_loss  = self.config.get("training", "reconstruction_loss"),
            class_weights        = class_weights,
            gamma_attn           = self.config.get("training", "gamma_attn", default=0.0),
            lambda_feat          = self._lambda_feat_from_config(self.config),
            lambda_conc          = self.config.get("training", "lambda_conc",  default=0.1),
            lambda_ortho         = self.config.get("training", "lambda_ortho", default=0.0),
            kl_anneal_epochs     = self.config.get("training", "kl_anneal_epochs", default=20),
            kl_anneal_end        = self.config.get("training", "kl_anneal_end",    default=4.0),
            device               = self.device,
        )

        model, best_metrics, history = trainer.train(
            num_epochs              = self.config.get("training", "num_epochs"),
            early_stopping_patience = self.config.get("training", "early_stopping_patience"),
            save_path               = model_save_path,
        )

        ckpt  = torch.load(model_save_path, map_location=self.device)
        model = self.model_builder()
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device)
        print(f"Loaded epoch={ckpt['epoch']} val_acc={ckpt['val_acc']:.4f}")

        val_loader_eval = DataLoader(val_dataset, batch_size=bs,
                                     shuffle=False, num_workers=nw)
        val_preds, val_labels = self._get_predictions(model, val_loader_eval)
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average="binary"
        )

        fold_metrics = {
            "fold":     fold_idx,
            "val_loss": best_metrics["val_loss"],
            "val_acc":  best_metrics["val_acc"],
            "precision": precision, "recall": recall, "f1": f1,
            "confusion_matrix": confusion_matrix(val_labels, val_preds),
        }
        print(f"Fold {fold_idx+1}: acc={best_metrics['val_acc']:.4f} f1={f1:.4f}")
        return model, fold_metrics, str(model_save_path)

    # ── inference helpers ─────────────────────────────────────────────────────

    def _get_predictions(self, model, loader):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating", leave=False):
                seq    = batch["sequence"].to(self.device)
                fwd_kw = _forward_kwargs(batch, self.device)
                out    = model(seq, deterministic=True, **fwd_kw)
                all_preds.append(out["logits"].argmax(1).cpu())
                all_labels.append(batch["label"].cpu())
        return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()

    def _get_probs(self, model, loader):
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="  Ensemble pass", leave=False):
                seq    = batch["sequence"].to(self.device)
                fwd_kw = _forward_kwargs(batch, self.device)
                out    = model(seq, deterministic=True, **fwd_kw)
                all_probs.append(torch.softmax(out["logits"], dim=1).cpu().numpy())
                all_labels.append(batch["label"].cpu().numpy())
        return np.vstack(all_probs), np.concatenate(all_labels)

    # ── cross-validation ──────────────────────────────────────────────────────

    def cross_validate(self, sequences, labels, stratify_by_length: bool = True):
        print(f"\n{'='*80}\n{self.n_folds}-Fold CV | {len(sequences)} samples\n{'='*80}")

        strat_groups = (
            create_length_stratified_groups(
                sequences, labels,
                n_bins=self.config.get("training", "n_bins", default=5)
            ) if stratify_by_length else labels
        )

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True,
            random_state=self.config.get("training", "random_state", default=42)
        )

        for fold_idx, (train_idx, val_idx) in enumerate(
                skf.split(sequences, strat_groups)):
            train_dataset = Subset(self.dataset, train_idx)
            val_dataset   = Subset(self.dataset, val_idx)

            train_labels_n = [{"lnc": 0, "pc": 1}[labels[i]] for i in train_idx]
            class_weights  = compute_class_weight(
                "balanced", classes=np.array([0, 1]), y=train_labels_n
            )

            model, fold_metrics, model_path = self.train_fold(
                fold_idx, train_dataset, val_dataset, class_weights
            )
            self.model_paths.append({
                "fold": fold_idx, "path": model_path,
                "val_acc": fold_metrics["val_acc"],
                "val_loss": fold_metrics["val_loss"],
            })
            self.fold_results.append(fold_metrics)

        cv_results = self._aggregate_results()
        output_dir = Path(self.config.get("output", "experiment_name"))
        pd.DataFrame(self.model_paths).to_csv(output_dir / "model_paths.csv", index=False)
        return cv_results

    def _aggregate_results(self):
        metrics = ["val_acc", "precision", "recall", "f1"]
        aggregated = {}
        print(f"\n{'='*80}\nCV Results\n{'='*80}")
        for m in metrics:
            vals = [f[m] for f in self.fold_results]
            aggregated[m] = {"mean": np.mean(vals), "std": np.std(vals)}
            print(f"  {m:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        return aggregated

    def save_results(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        serializable = []
        for fold in self.fold_results:
            f = fold.copy()
            f["confusion_matrix"] = f["confusion_matrix"].tolist()
            serializable.append(f)
        with open(output_dir / "fold_results.json", "w") as fh:
            json.dump(serializable, fh, indent=2)
        print(f"Saved fold results → {output_dir / 'fold_results.json'}")

    # ── test set evaluation ───────────────────────────────────────────────────

    def evaluate_on_test_set(self, test_dataset):
        bs  = self.config.get("training", "batch_size")
        nw  = self.config.get("training", "num_workers", default=1)
        loader     = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=nw)
        n_samples  = len(test_dataset)
        sum_probs  = np.zeros((n_samples, 2), dtype=np.float64)
        labels_arr = None

        for fold_info in self.model_paths:
            ckpt  = torch.load(fold_info["path"], map_location=self.device)
            model = self.model_builder()
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(self.device)
            print(f"  Fold {fold_info['fold']} | epoch={ckpt['epoch']} "
                  f"val_acc={ckpt['val_acc']:.4f}")
            fold_probs, fold_labels = self._get_probs(model, loader)
            sum_probs  += fold_probs
            labels_arr  = fold_labels

        avg_probs   = sum_probs / len(self.model_paths)
        predictions = avg_probs.argmax(axis=1)
        acc                      = accuracy_score(labels_arr, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_arr, predictions, average="binary"
        )
        return {
            "accuracy":          float(acc),
            "precision":         float(precision),
            "recall":            float(recall),
            "f1":                float(f1),
            "confusion_matrix":  confusion_matrix(labels_arr, predictions).tolist(),
            "n_samples":         int(n_samples),
            "n_lncrna":          int((labels_arr == 0).sum()),
            "n_pcrna":           int((labels_arr == 1).sum()),
            "n_folds_ensembled": len(self.model_paths),
        }

    # ── attention extraction ──────────────────────────────────────────────────

    def extract_attention_all_folds(self, output_dir, labels_override=None):
        """
        Extract attention weights from every fold model on its validation set.

        Saves per-fold .npz files with:
            attn_weights      (N, L_encoded, N_tokens)   averaged over heads
            attn_weights_full (N, num_heads, L_encoded, N_tokens)
            predictions, confidences, labels, transcript_ids,
            is_hard_case, seq_lengths,
            token_names  (N_tokens,)  sub-group name per token position
            block_names  (N_tokens,)  block name per token position
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        sequences   = self.dataset.sequences
        labels_list = (labels_override if labels_override is not None
                       else [s.annotations.get("label") for s in sequences])

        strat_groups = create_length_stratified_groups(
            sequences, labels_list,
            n_bins=self.config.get("training", "n_bins", default=5)
        )
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True,
            random_state=self.config.get("training", "random_state", default=42)
        )
        splits = list(skf.split(sequences, strat_groups))

        bs = self.config.get("training", "batch_size")
        nw = self.config.get("training", "num_workers", default=1)

        # Token metadata — computed once, registry-driven
        all_subgroups = REGISTRY.all_subgroups
        token_names = np.array(all_subgroups, dtype=object)
        block_names = np.array(
            [REGISTRY.token_block(sg) for sg in all_subgroups], dtype=object
        )

        for fold_info in self.model_paths:
            fold_idx   = fold_info["fold"]
            _, val_idx = splits[fold_idx]
            print(f"\nFold {fold_idx} — extracting from {len(val_idx)} val samples")

            ckpt  = torch.load(fold_info["path"], map_location=self.device)
            model = self.model_builder()
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(self.device)
            model.eval()

            val_loader = DataLoader(
                Subset(self.dataset, val_idx),
                batch_size=bs, shuffle=False, num_workers=nw
            )

            all_attn_full, all_preds, all_confs = [], [], []
            all_labels, all_ids, all_lengths    = [], [], []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Fold {fold_idx}", leave=False):
                    seq    = batch["sequence"].to(self.device)
                    fwd_kw = _forward_kwargs(batch, self.device)

                    out   = model(seq, deterministic=True, **fwd_kw)
                    probs = torch.softmax(out["logits"], dim=1)
                    all_attn_full.append(out["attn_weights"].cpu().numpy())
                    all_preds.append(out["logits"].argmax(1).cpu().numpy())
                    all_confs.append(probs.max(1).values.cpu().numpy())
                    all_labels.append(batch["label"].numpy())
                    all_ids.extend(batch["transcript_id"])
                    all_lengths.append(batch["length"].numpy())

            attn_full   = np.concatenate(all_attn_full, axis=0)  # (N, H, L, T)
            predictions = np.concatenate(all_preds)
            confidences = np.concatenate(all_confs)
            labels_arr  = np.concatenate(all_labels)
            lengths_arr = np.concatenate(all_lengths)
            is_hard     = (predictions != labels_arr) | (confidences < 0.6)

            n_tokens_actual = attn_full.shape[-1]
            if n_tokens_actual != len(token_names):
                print(f"   attn token dim {n_tokens_actual} ≠ registry "
                      f"{len(token_names)} — checkpoint may predate current "
                      f"feature blocks")

            save_path = output_dir / f"fold_{fold_idx}_attn.npz"
            np.savez(
                save_path,
                attn_weights      = attn_full.mean(axis=1),   # (N, L, T)
                attn_weights_full = attn_full,                 # (N, H, L, T)
                predictions       = predictions,
                confidences       = confidences,
                labels            = labels_arr,
                transcript_ids    = np.array(all_ids, dtype=object),
                is_hard_case      = is_hard,
                seq_lengths       = lengths_arr,
                token_names       = token_names[:n_tokens_actual],
                block_names       = block_names[:n_tokens_actual],
                # Legacy aliases for scripts not yet updated
                nonb_subgroups    = np.array(REGISTRY.nonb_subgroups, dtype=object),
                te_subgroups      = np.array(REGISTRY.te_subgroups,   dtype=object),
            )

            print(f"  Saved → {save_path}  "
                  f"(hard: {is_hard.sum()}/{len(is_hard)}, "
                  f"attn: {attn_full.shape})")

        print(f"\nAll fold outputs saved to {output_dir}/")