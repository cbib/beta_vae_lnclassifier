# trainers/features_attn_trainer.py
"""
Cross-validation trainer for β-VAE with cross-attention genomic features.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import json
from pathlib import Path
from tqdm import tqdm

from data.cv_utils import create_length_stratified_groups

from models.beta_vae_features_attn import BetaVAEWithFeaturesAttention

import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


# =============================================================================
# Loss — identical to previous version, copied here for self-containment
# =============================================================================

class BetaVAE_Features_Loss:
    """Combined loss for β-VAE with genomic features."""

    def __init__(self, alpha=0.001, beta=4.0, gamma_classification=1.0,
                 lambda_recon=1.0, reconstruction_loss='bce', class_weights=None, gamma_attn=0.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma_classification = gamma_classification
        self.lambda_recon = lambda_recon
        self.reconstruction_loss = reconstruction_loss
        self.class_weights = class_weights
        self.gamma_attn = gamma_attn

    def reconstruction_loss_fn(self, x_recon, x_true):
        """
        Reconstruction loss (x_recon is already sigmoid output).
        Returns per-sequence loss for better scaling.
        """
        if self.reconstruction_loss == 'bce':
            return F.binary_cross_entropy(x_recon, x_true, reduction='mean')
        elif self.reconstruction_loss == 'mse':
            return F.mse_loss(x_recon, x_true, reduction='mean')
        raise ValueError(f"Unknown loss type: {self.reconstruction_loss}")

    def kl_divergence(self, mu, logvar):
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl.mean() / mu.size(1)

    def __call__(self, x_true, x_recon, mu, logvar, logits, labels, attn_weights=None):
        """
        Compute combined loss.
        
        Args:
            x_true: Original sequences (batch, input_dim, seq_len)
            x_recon: Reconstructed sequences (batch, input_dim, seq_len)
            mu: Latent mean (batch, latent_dim)
            logvar: Latent log variance (batch, latent_dim)
            logits: Classification logits (batch, num_classes)
            labels: Class labels (batch,)
        
        Returns:
            total_loss: Combined loss tensor
            loss_dict: Dictionary of individual loss components
        """
        # Reconstruction loss
        recon_loss = self.reconstruction_loss_fn(x_recon, x_true)
        
        # KL divergence
        kl_loss = self.kl_divergence(mu, logvar)
        
        # Classification loss
        if self.class_weights is not None:
            cls_loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            cls_loss = F.cross_entropy(logits, labels)

        total = (self.alpha * self.lambda_recon * recon_loss +
                 self.beta * kl_loss +
                 self.gamma_classification * cls_loss)
        
        # Attention entropy penalty
        if attn_weights is not None and self.gamma_attn > 0:
            entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(-1).mean()
            total = total + self.gamma_attn * entropy

        return total, {
            'loss': total.item(),
            'reconstruction': recon_loss.item(),
            'kl': kl_loss.item(),
            'classification': cls_loss.item(),
            'attn_entropy': entropy.item() if attn_weights is not None else None
        }


# =============================================================================
# Single-fold trainer — identical to previous version
# =============================================================================

class SingleFoldFeaturesAttentionTrainer:
    """Single-fold trainer. Identical to SingleFoldBetaVAEFeaturesTrainer."""

    def __init__(self, model, train_loader, val_loader,
                 learning_rate=1e-4, weight_decay=1e-5,
                 alpha=0.001, beta=1.0, gamma_classification=1.0,
                 lambda_recon=1.0, reconstruction_loss='bce', 
                 class_weights=None, gamma_attn=0.0, device=None):

        self.model = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.kl_anneal_start  = 0.0
        self.kl_anneal_end    = 4.0
        self.kl_anneal_epochs = 20

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(self.device)

        self.gamma_attn = gamma_attn

        self.criterion = BetaVAE_Features_Loss(
            alpha=alpha, beta=beta,
            gamma_classification=gamma_classification,
            lambda_recon=lambda_recon,
            reconstruction_loss=reconstruction_loss,
            class_weights=class_weights,
            gamma_attn=gamma_attn
        )

        attn_params  = [p for n, p in self.model.named_parameters() if 'cross_attn' in n]
        other_params = [p for n, p in self.model.named_parameters() if 'cross_attn' not in n]

        self.optimizer = torch.optim.Adam([
            {'params': other_params, 'lr': learning_rate},
            {'params': attn_params,  'lr': learning_rate * 50}
        ], weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
        )
        # In SingleFoldFeaturesAttentionTrainer.__init__, after self.scheduler = ...
        self._check_attention_gradients(train_loader, n_batches=5)

        self.history = {
            'train_loss': [], 'train_reconstruction': [], 'train_kl': [],
            'train_classification': [], 'val_loss': [], 'val_acc': []
        }

    def _check_attention_gradients(self, train_loader, n_batches=5):
        print("\n--- Attention gradient check (first 5 batches) ---")
        self.model.train()
        qp_init = self.model.cross_attn.query_proj.weight.data.clone()

        for i, batch in enumerate(train_loader):
            if i >= n_batches:
                break
            seq  = batch['sequence'].to(self.device)
            te   = batch['te_features'].to(self.device)
            nonb = batch['nonb_features'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(seq, te_features=te, nonb_features=nonb, deterministic=False)
            loss, _ = self.criterion(
                seq, outputs['reconstruction'], outputs['mu'],
                outputs['logvar'], outputs['logits'], labels
            )
            loss.backward()

            grad = self.model.cross_attn.query_proj.weight.grad
            if grad is None:
                raise RuntimeError("query_proj.weight.grad is None — attention disconnected from graph.")
            grad_std = grad.std().item()
            print(f"  Batch {i}: query_proj grad std = {grad_std:.2e}")
            if grad_std < 1e-6:
                raise RuntimeError(
                    f"query_proj gradient vanishing (std={grad_std:.2e}). "
                    f"Attention will collapse."
                )
            self.optimizer.step()

        qp_after  = self.model.cross_attn.query_proj.weight.data
        delta_std = (qp_after - qp_init).std().item()
        print(f"  query_proj delta after {n_batches} steps: std={delta_std:.2e}")
        if delta_std < 1e-6:
            raise RuntimeError(f"query_proj weights did not move (delta std={delta_std:.2e}).")
        print(f"  ✓ Attention gradients healthy — proceeding with training.\n")

    def get_current_beta(self, epoch):
        if epoch >= self.kl_anneal_epochs:
            return self.kl_anneal_end
        return self.kl_anneal_start + (epoch / self.kl_anneal_epochs) * (
            self.kl_anneal_end - self.kl_anneal_start)

    def train_epoch(self, epoch_num):
        """Train for one epoch"""
        self.model.train()
        totals = {'loss': 0, 'reconstruction': 0, 'kl': 0, 'classification': 0}
        current_beta = self.get_current_beta(epoch_num)

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch_num} [Train]', leave=False)
        for batch in pbar:
            seq        = batch['sequence'].to(self.device)
            te         = batch['te_features'].to(self.device)
            nonb       = batch['nonb_features'].to(self.device)
            labels     = batch['label'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(seq, te_features=te, nonb_features=nonb, deterministic=False)

            self.criterion.beta = current_beta
            loss, loss_dict = self.criterion(
                seq, outputs['reconstruction'], outputs['mu'],
                outputs['logvar'], outputs['logits'], labels, attn_weights=outputs['attn_weights']
            )
            # Shared loss class returns 'total'; remap to 'loss' here
            # so the loss class is not modified and other trainers are unaffected
            if 'total' in loss_dict and 'loss' not in loss_dict:
                loss_dict['loss'] = loss_dict.pop('total')

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for k in totals:
                totals[k] += loss_dict[k]
            pbar.set_postfix({k: f"{v:.4f}" for k, v in loss_dict.items()})

        n = len(self.train_loader)
        return {k: v / n for k, v in totals.items()}

    def evaluate(self, epoch_num):
        """Evaluate on validation set"""
        self.model.eval()
        all_logits, all_labels = [], []
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f'Epoch {epoch_num} [Val]', leave=False):
                seq    = batch['sequence'].to(self.device)
                te     = batch['te_features'].to(self.device)
                nonb   = batch['nonb_features'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(seq, te_features=te, nonb_features=nonb, deterministic=True)
                loss = F.cross_entropy(outputs['logits'], labels)
                total_loss += loss.item()

                all_logits.append(outputs['logits'].cpu())
                all_labels.append(labels.cpu())

        preds = torch.cat(all_logits).argmax(dim=1)
        acc   = accuracy_score(torch.cat(all_labels), preds)
        return {'loss': total_loss / len(self.val_loader), 'accuracy': acc}

    def train(self, num_epochs, early_stopping_patience=10, save_path=None):
        """Full training loop with model saving"""
        best_val_loss = float('inf')
        patience_counter = 0

        print(f"\nTraining {num_epochs} epochs (patience={early_stopping_patience})")
        if save_path:
            print(f"Saving best model to: {save_path}")

        for epoch in range(1, num_epochs + 1):
            train_m = self.train_epoch(epoch)
            val_m   = self.evaluate(epoch)

            for k in ['loss', 'reconstruction', 'kl', 'classification']:
                self.history[f'train_{k}' if k != 'loss' else 'train_loss'].append(
                    train_m.get(k, train_m.get('loss'))
                )
            self.history['val_loss'].append(val_m['loss'])
            self.history['val_acc'].append(val_m['accuracy'])

            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train: {train_m['loss']:.4f} "
                  f"(recon={train_m['reconstruction']:.4f} "
                  f"kl={train_m['kl']:.4f} "
                  f"cls={train_m['classification']:.4f}) | "
                  f"Val: {val_m['loss']:.4f} acc={val_m['accuracy']:.4f}")

            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_m['loss'])
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"  → LR: {old_lr:.2e} → {new_lr:.2e}")

            if val_m['loss'] < best_val_loss:
                best_val_loss    = val_m['loss']
                patience_counter = 0
                
                # Save best model checkpoint
                if save_path:
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch,
                        'val_loss': best_val_loss,
                        'val_acc': val_m['accuracy'],
                        'history': self.history
                    }, save_path)
                    print(f"  → Saved checkpoint → {save_path}")
                print(f"  → New best val loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"  → Patience: {patience_counter}/{early_stopping_patience}")
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        best_metrics = {
            'val_loss': best_val_loss,
            'val_acc':  max(self.history['val_acc']),
            'train_loss': min(self.history['train_loss'])
        }
        return self.model, best_metrics, self.history


# =============================================================================
# Cross-validation trainer
# =============================================================================

class BetaVAEFeaturesAttentionTrainer:
    """Cross-validation trainer for β-VAE with cross-attention genomic features."""

    def __init__(self, model_builder, dataset, config, n_folds=5, device=None):
        """
        Args:
            model_builder: Function that returns a new model instance
            dataset: Full SequenceFeatureDataset
            config: Configuration object
            n_folds: Number of cross-validation folds
            device: torch device
        """
        self.model_builder = model_builder
        self.dataset       = dataset
        self.config        = config
        self.n_folds       = n_folds
        self.device        = torch.device(device) if device else \
                             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fold_results  = []
        self.model_paths   = []

        print(f"BetaVAEFeaturesAttentionTrainer | {n_folds} folds | device={self.device}")

    def train_fold(self, fold_idx, train_dataset, val_dataset, class_weights=None):
        print(f"\n{'='*60}\nFold {fold_idx + 1}/{self.n_folds}\n{'='*60}")

        # Leakage check
        train_ids = set([self.dataset.sequences[i].id.split('|')[0]
                         for i in train_dataset.indices])
        val_ids   = set([self.dataset.sequences[i].id.split('|')[0]
                         for i in val_dataset.indices])
        overlap   = train_ids & val_ids
        print(f"Train={len(train_ids)} Val={len(val_ids)} Overlap={len(overlap)}")
        if overlap:
            print(f"⚠️  WARNING: {len(overlap)} overlapping transcripts!")

        batch_size  = self.config.get('training', 'batch_size')
        num_workers = self.config.get('training', 'num_workers', default=1)

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  pin_memory=True, drop_last=True)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers,
                                  pin_memory=True)

        save_dir = Path(self.config.get('output', 'experiment_name')) / 'models'
        save_dir.mkdir(exist_ok=True, parents=True)
        model_save_path = save_dir / f'fold_{fold_idx}_best.pt'

        model   = self.model_builder()
        trainer = SingleFoldFeaturesAttentionTrainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            learning_rate=self.config.get('training', 'learning_rate'),
            weight_decay=self.config.get('training', 'weight_decay'),
            alpha=self.config.get('training', 'alpha'),
            beta=self.config.get('training', 'beta'),
            gamma_classification=self.config.get('training', 'gamma_classification'),
            lambda_recon=self.config.get('training', 'lambda_recon'),
            reconstruction_loss=self.config.get('training', 'reconstruction_loss'),
            class_weights=class_weights, device=self.device,
            gamma_attn=self.config.get('training', 'gamma_attn', default=0.0)
        )
                
        # Train
        model, best_metrics, history = trainer.train(
            num_epochs=self.config.get('training', 'num_epochs'),
            early_stopping_patience=self.config.get('training', 'early_stopping_patience'),
            save_path=model_save_path
        )

        # Reload best checkpoint
        checkpoint = torch.load(model_save_path, map_location=self.device)
        model = self.model_builder()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        print(f"Loaded epoch={checkpoint['epoch']} val_acc={checkpoint['val_acc']:.4f}")

        val_loader_eval = DataLoader(val_dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=num_workers)
        val_preds, val_labels = self._get_predictions(model, val_loader_eval)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average='binary'
        )
        cm = confusion_matrix(val_labels, val_preds)

        fold_metrics = {
            'fold': fold_idx,
            'val_loss': best_metrics['val_loss'],
            'val_acc':  best_metrics['val_acc'],
            'precision': precision, 'recall': recall, 'f1': f1,
            'confusion_matrix': cm
        }
        print(f"Fold {fold_idx+1}: acc={best_metrics['val_acc']:.4f} f1={f1:.4f}")
        return model, fold_metrics, str(model_save_path)
    
    def _get_predictions(self, model, loader):
        """Get predictions from model"""
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating", leave=False):
                seq    = batch['sequence'].to(self.device)
                te     = batch['te_features'].to(self.device)
                nonb   = batch['nonb_features'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = model(seq, te_features=te, nonb_features=nonb, deterministic=True)
                all_preds.append(outputs['logits'].argmax(1).cpu())
                all_labels.append(labels.cpu())
        return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()

    def evaluate_on_test_set(self, test_dataset):
        """
        Evaluate on independent test set using ensemble of all fold models.
        Averages softmax probabilities across folds, then thresholds at 0.5.
        """
        if not self.model_paths:
            raise ValueError("No model paths found. Run cross_validate() first.")

        batch_size  = self.config.get('training', 'batch_size')
        num_workers = self.config.get('training', 'num_workers', default=1)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)

        n_samples    = len(test_dataset)
        sum_probs    = np.zeros((n_samples, 2), dtype=np.float64)
        labels_arr   = None

        for fold_info in self.model_paths:
            checkpoint = torch.load(fold_info['path'], map_location=self.device)
            model = self.model_builder()
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            print(f"  Fold {fold_info['fold']} | epoch={checkpoint['epoch']} "
                f"val_acc={checkpoint['val_acc']:.4f}")

            fold_probs, fold_labels = self._get_probs(model, test_loader)
            sum_probs  += fold_probs
            labels_arr  = fold_labels   # same every fold

        avg_probs  = sum_probs / len(self.model_paths)
        predictions = avg_probs.argmax(axis=1)

        acc                       = accuracy_score(labels_arr, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_arr, predictions, average='binary'
        )
        cm = confusion_matrix(labels_arr, predictions)

        return {
            'accuracy':         float(acc),
            'precision':        float(precision),
            'recall':           float(recall),
            'f1':               float(f1),
            'confusion_matrix': cm.tolist(),
            'n_samples':        int(n_samples),
            'n_lncrna':         int((labels_arr == 0).sum()),
            'n_pcrna':          int((labels_arr == 1).sum()),
            'n_folds_ensembled': len(self.model_paths),
        }
    
    def _get_probs(self, model, loader):
        """Returns softmax probabilities and labels as numpy arrays."""
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="  Ensemble pass", leave=False):
                seq    = batch['sequence'].to(self.device)
                te     = batch['te_features'].to(self.device)
                nonb   = batch['nonb_features'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = model(seq, te_features=te, nonb_features=nonb, deterministic=True)
                probs   = torch.softmax(outputs['logits'], dim=1)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        return np.vstack(all_probs), np.concatenate(all_labels)

    def cross_validate(self, sequences, labels, stratify_by_length=True):
        print(f"\n{'='*80}\n{self.n_folds}-Fold CV | {len(sequences)} samples\n{'='*80}")

        strat_groups = create_length_stratified_groups(
            sequences, labels,
            n_bins=self.config.get('training', 'n_bins', default=5)
        ) if stratify_by_length else labels

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True,
            random_state=self.config.get('training', 'random_state', default=42)
        )

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(sequences, strat_groups)):
            train_dataset = Subset(self.dataset, train_idx)
            val_dataset   = Subset(self.dataset, val_idx)

            label_to_idx   = {'lnc': 0, 'pc': 1}
            train_labels_n = [label_to_idx[labels[i]] for i in train_idx]
            class_weights  = compute_class_weight(
                'balanced', classes=np.array([0, 1]), y=train_labels_n)

            model, fold_metrics, model_path = self.train_fold(
                fold_idx, train_dataset, val_dataset, class_weights)

            self.model_paths.append({
                'fold': fold_idx, 'path': model_path,
                'val_acc': fold_metrics['val_acc'],
                'val_loss': fold_metrics['val_loss']
            })
            self.fold_results.append(fold_metrics)

        cv_results = self._aggregate_results()

        output_dir      = Path(self.config.get('output', 'experiment_name'))
        model_paths_df  = pd.DataFrame(self.model_paths)
        model_paths_df.to_csv(output_dir / 'model_paths.csv', index=False)

        best_fold_idx       = model_paths_df['val_loss'].idxmin()
        self.best_model_path = model_paths_df.iloc[best_fold_idx]['path']
        self.best_fold_idx   = best_fold_idx

        print(f"\nBest fold: {best_fold_idx+1} | "
              f"val_loss={model_paths_df.iloc[best_fold_idx]['val_loss']:.4f} | "
              f"val_acc={model_paths_df.iloc[best_fold_idx]['val_acc']:.4f}")
        return cv_results
    
    def _aggregate_results(self):
        metrics = ['val_acc', 'precision', 'recall', 'f1']
        aggregated = {}
        print(f"\n{'='*80}\nCV Results\n{'='*80}")
        for m in metrics:
            vals = [f[m] for f in self.fold_results]
            aggregated[m] = {'mean': np.mean(vals), 'std': np.std(vals), 'values': vals}
            print(f"  {m:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        return aggregated
    
    def save_results(self, output_dir):
        """Save all CV results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save fold results
        fold_results_serializable = []
        for fold in self.fold_results:
            f = fold.copy()
            f['confusion_matrix'] = f['confusion_matrix'].tolist()
            fold_results_serializable.append(f)
        with open(output_dir / 'fold_results.json', 'w') as fh:
            json.dump(fold_results_serializable, fh, indent=2)
        print(f"Saved fold results → {output_dir / 'fold_results.json'}")

    # =========================================================================
    # extract_embeddings_from_best_fold — now also saves attn_weights
    # =========================================================================

    def extract_embeddings_from_best_fold(self, output_path):
        """
        Extract embeddings AND attention weights from best fold model.

        Saves .npz with:
            embeddings:     (N, latent_dim)    — latent mean z
            logits:         (N, num_classes)
            probs:          (N, num_classes)
            predictions:    (N,)
            confidences:    (N,)               — max class probability
            labels:         (N,)
            transcript_ids: (N,)
            attn_weights:   (N, L_encoded, 2)  — col0=TE, col1=NonB  ← NEW
        """
        if not hasattr(self, 'best_model_path'):
            raise ValueError("Run cross_validate() first.")

        print(f"\nLoading best model: {self.best_model_path}")
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        
        # Rebuild model
        model = self.model_builder()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        print(f"  epoch={checkpoint['epoch']} val_acc={checkpoint['val_acc']:.4f}")

        loader = DataLoader(self.dataset, batch_size=256, shuffle=False, num_workers=1)

        all_embeddings, all_logits, all_labels   = [], [], []
        all_transcript_ids, all_attn_weights      = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting"):
                seq  = batch['sequence'].to(self.device)
                te   = batch['te_features'].to(self.device)
                nonb = batch['nonb_features'].to(self.device)

                outputs = model(seq, te_features=te, nonb_features=nonb, deterministic=True)

                all_embeddings.append(outputs['mu'].cpu().numpy())
                all_logits.append(outputs['logits'].cpu().numpy())
                all_labels.append(batch['label'].numpy())
                all_transcript_ids.extend(batch['transcript_id'])
                all_attn_weights.append(outputs['attn_weights'].cpu().numpy())

        embeddings   = np.vstack(all_embeddings)
        logits       = np.vstack(all_logits)
        labels       = np.concatenate(all_labels)
        attn_weights = np.vstack(all_attn_weights)

        probs       = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        predictions = logits.argmax(axis=1)
        confidences = probs.max(axis=1)

        print(f"\nExtracted:")
        print(f"  embeddings:   {embeddings.shape}")
        print(f"  attn_weights: {attn_weights.shape}  (L_encoded={attn_weights.shape[1]})")

        output_path = Path(output_path)
        np.savez(
            output_path,
            embeddings=embeddings,
            logits=logits,
            probs=probs,
            predictions=predictions,
            confidences=confidences,
            labels=labels,
            transcript_ids=np.array(all_transcript_ids, dtype=object),
            attn_weights=attn_weights
        )
        print(f"  Saved -> {output_path}")
        return str(output_path)

    def extract_attention_all_folds(self, output_dir, labels_override=None):
        """
        Extract attention weights from every fold model on its validation set.

        Args:
            output_dir: Directory to save per-fold .npz files
            labels_override: Optional list of labels ['lncRNA'/'protein_coding'] in
                             dataset order. If provided, used instead of reading from
                             sequence annotations (which can fail with KeyError: 'unknown').

        Creates one .npz per fold: output_dir/fold_N_attn.npz

        Each .npz contains:
            attn_weights:   (N_val, L_encoded, 2)
            predictions:    (N_val,)
            confidences:    (N_val,)
            labels:         (N_val,)
            transcript_ids: (N_val,)
            is_hard_case:   (N_val,)  — misclassified OR confidence < 0.6
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        if not self.model_paths:
            raise ValueError("No model paths found. Run cross_validate() first.")

        sequences = self.dataset.sequences

        # Use override if provided, else try annotations (original behaviour)
        if labels_override is not None:
            labels_list = labels_override
        else:
            labels_list = [s.annotations.get('label', None) for s in sequences]
            if any(l is None for l in labels_list):
                raise ValueError(
                    "Some sequences have no 'label' annotation. "
                    "Pass labels_override= to extract_attention_all_folds()."
                )

        strat_groups = create_length_stratified_groups(
            sequences, labels_list,
            n_bins=self.config.get('training', 'n_bins', default=5)
        )
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True,
            random_state=self.config.get('training', 'random_state', default=42)
        )
        splits = list(skf.split(sequences, strat_groups))

        batch_size  = self.config.get('training', 'batch_size')
        num_workers = self.config.get('training', 'num_workers', default=1)

        for fold_info in self.model_paths:
            fold_idx   = fold_info['fold']
            model_path = fold_info['path']
            _, val_idx = splits[fold_idx]

            print(f"\nFold {fold_idx} — extracting attention from {len(val_idx)} val samples")

            checkpoint = torch.load(model_path, map_location=self.device)
            model = self.model_builder()
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()

            val_dataset = Subset(self.dataset, val_idx)
            val_loader  = DataLoader(val_dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=num_workers)

            all_attn, all_preds, all_labels, all_ids, all_lengths = [], [], [], [], []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Fold {fold_idx}", leave=False):
                    seq  = batch['sequence'].to(self.device)
                    te   = batch['te_features'].to(self.device)
                    nonb = batch['nonb_features'].to(self.device)

                    outputs     = model(seq, te_features=te, nonb_features=nonb, deterministic=True)
                    probs_batch = torch.softmax(outputs['logits'], dim=1)
                    preds       = outputs['logits'].argmax(1)
                    confs       = probs_batch.max(1).values

                    all_attn.append(outputs['attn_weights'].cpu().numpy())
                    all_preds.append(np.stack([preds.cpu().numpy(),
                                               confs.cpu().numpy()], axis=1))
                    all_labels.append(batch['label'].numpy())
                    all_ids.extend(batch['transcript_id'])
                    all_lengths.append(batch['length'].numpy())

            attn_weights = np.vstack(all_attn)
            preds_confs  = np.vstack(all_preds)
            predictions  = preds_confs[:, 0].astype(int)
            confidences  = preds_confs[:, 1]
            labels_arr   = np.concatenate(all_labels)

            is_hard = (predictions != labels_arr) | (confidences < 0.6)

            save_path = output_dir / f'fold_{fold_idx}_attn.npz'
            np.savez(
                save_path,
                attn_weights=attn_weights,
                predictions=predictions,
                confidences=confidences,
                labels=labels_arr,
                transcript_ids=np.array(all_ids, dtype=object),
                is_hard_case=is_hard,
                seq_lengths=np.concatenate(all_lengths)
            )
            print(f"  Saved -> {save_path}  "
                  f"(hard cases: {is_hard.sum()} / {len(is_hard)})")

        print(f"\nAll fold attention weights saved to {output_dir}/")