# trainers/beta_vae_features_trainer.py
"""
Cross-validation trainer for β-VAE with genomic features
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

# trainers/beta_vae_features_single_fold.py
"""
Single fold trainer for β-VAE with genomic features
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


class BetaVAE_Features_Loss:
    """
    Combined loss for β-VAE with genomic features.
    No contrastive component (features provide biological priors instead).
    """
    
    def __init__(self,
                 alpha=0.001,
                 beta=4.0,
                 gamma_classification=1.0,
                 lambda_recon=1.0,
                 reconstruction_loss='bce',
                 class_weights=None):
        """
        Args:
            alpha: Weight for reconstruction loss
            beta: Weight for KL divergence (β-VAE parameter)
            gamma_classification: Weight for classification loss
            lambda_recon: Additional scaling for reconstruction
            reconstruction_loss: 'bce' or 'mse'
            class_weights: Tensor of class weights for classification loss
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma_classification = gamma_classification
        self.lambda_recon = lambda_recon
        self.reconstruction_loss = reconstruction_loss
        self.class_weights = class_weights

    def reconstruction_loss_fn(self, x_recon, x_true):
        """
        Reconstruction loss (x_recon is already sigmoid output).
        Returns per-sequence loss for better scaling.
        """
        if self.reconstruction_loss == 'bce':
            # x_recon already has sigmoid applied
            loss = F.binary_cross_entropy(x_recon, x_true, reduction='mean')
        elif self.reconstruction_loss == 'mse':
            # Direct MSE (no extra sigmoid!)
            loss = F.mse_loss(x_recon, x_true, reduction='mean')
        else:
            raise ValueError(f"Unknown loss type: {self.reconstruction_loss}")
        
        return loss
    
    def kl_divergence(self, mu, logvar):
        """KL divergence between latent distribution and standard normal."""
        kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_per_sample.mean() / mu.size(1)  # Normalize by latent_dim
    
    def __call__(self, x_true, x_recon, mu, logvar, logits, labels):
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
            classification_loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            classification_loss = F.cross_entropy(logits, labels)
        
        # Combined loss (no contrastive component)
        total_loss = (self.alpha * self.lambda_recon * recon_loss + 
                     self.beta * kl_loss + 
                     self.gamma_classification * classification_loss)
        
        loss_dict = {
            'total': total_loss.item(),
            'reconstruction': recon_loss.item(),
            'kl': kl_loss.item(),
            'classification': classification_loss.item()
        }
        
        return total_loss, loss_dict


class SingleFoldBetaVAEFeaturesTrainer:
    """
    Trainer for a single fold with β-VAE + genomic features (no contrastive loss).
    """
    
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 learning_rate=1e-4,
                 weight_decay=1e-5,
                 alpha=0.001,
                 beta=1.0,
                 gamma_classification=1.0,
                 lambda_recon=1.0,
                 reconstruction_loss='bce',
                 class_weights=None,
                 device=None):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # KL annealing
        self.kl_anneal_start = 0.0
        self.kl_anneal_end = 4.0
        self.kl_anneal_epochs = 20
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # Loss function
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        self.criterion = BetaVAE_Features_Loss(
            alpha=alpha,
            beta=beta,
            gamma_classification=gamma_classification,
            lambda_recon=lambda_recon,
            reconstruction_loss=reconstruction_loss,
            class_weights=class_weights
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # History
        self.history = {
            'train_loss': [],
            'train_recon': [],
            'train_kl': [],
            'train_classification': [],
            'val_loss': [],
            'val_acc': []
        }

    def get_current_beta(self, epoch):
            if epoch >= self.kl_anneal_epochs:
                return self.kl_anneal_end
            
            progress = epoch / self.kl_anneal_epochs
            return self.kl_anneal_start + progress * (self.kl_anneal_end - self.kl_anneal_start)
        
    
    def train_epoch(self, epoch_num):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_classification = 0

        current_beta = self.get_current_beta(epoch_num)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch_num} [Train]', leave=False)
        for batch in pbar:
            sequences = batch['sequence'].to(self.device)
            te_features = batch['te_features'].to(self.device)
            nonb_features = batch['nonb_features'].to(self.device)
            class_labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(sequences, te_features=te_features, 
                               nonb_features=nonb_features, deterministic=False)
            
            x_recon = outputs['reconstruction']
            mu = outputs['mu']
            logvar = outputs['logvar']
            logits = outputs['logits']
            
            self.criterion.beta = current_beta

            # Compute loss
            loss, loss_dict = self.criterion(
                sequences, x_recon, mu, logvar, logits, class_labels
            )
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track
            total_loss += loss_dict['total']
            total_recon += loss_dict['reconstruction']
            total_kl += loss_dict['kl']
            total_classification += loss_dict['classification']
            
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'recon': f"{loss_dict['reconstruction']:.4f}",
                'kl': f"{loss_dict['kl']:.4f}",
                'cls': f"{loss_dict['classification']:.4f}"
            })
        
        n_batches = len(self.train_loader)
        return {
            'loss': total_loss / n_batches,
            'reconstruction': total_recon / n_batches,
            'kl': total_kl / n_batches,
            'classification': total_classification / n_batches
        }
    
    def evaluate(self, epoch_num):
        """Evaluate on validation set"""
        self.model.eval()
        
        all_logits = []
        all_labels = []
        total_loss = 0
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch_num} [Val]', leave=False)
        with torch.no_grad():
            for batch in pbar:
                sequences = batch['sequence'].to(self.device)
                te_features = batch['te_features'].to(self.device)
                nonb_features = batch['nonb_features'].to(self.device)
                class_labels = batch['label'].to(self.device)
                
                # Forward pass (deterministic for evaluation)
                outputs = self.model(sequences, te_features=te_features,
                                   nonb_features=nonb_features, deterministic=True)
                
                logits = outputs['logits']
                
                loss = F.cross_entropy(logits, class_labels)
                total_loss += loss.item()
                
                # Move to CPU and delete GPU tensors
                all_logits.append(logits.cpu())
                all_labels.append(class_labels.cpu())
                
                del sequences, te_features, nonb_features, class_labels, logits
                
                pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
        
        # Compute metrics
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        predictions = all_logits.argmax(dim=1)
        
        acc = accuracy_score(all_labels, predictions)
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': acc
        }
    
    def train(self, num_epochs, early_stopping_patience=10, save_path=None):
        """Full training loop with model saving"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\nTraining for {num_epochs} epochs (patience={early_stopping_patience})")
        if save_path:
            print(f"Will save best model to: {save_path}")
        print("-" * 70)
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_recon'].append(train_metrics['reconstruction'])
            self.history['train_kl'].append(train_metrics['kl'])
            self.history['train_classification'].append(train_metrics['classification'])
            
            # Evaluate
            val_metrics = self.evaluate(epoch)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            # Print epoch summary
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} "
                  f"(recon: {train_metrics['reconstruction']:.4f}, "
                  f"kl: {train_metrics['kl']:.4f}, "
                  f"cls: {train_metrics['classification']:.4f}) | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f}")
            
            # Learning rate scheduling
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_metrics['loss'])
            new_lr = self.optimizer.param_groups[0]['lr']
            
            if new_lr != current_lr:
                print(f"  → Learning rate reduced: {current_lr:.2e} → {new_lr:.2e}")
            
            # Early stopping and model saving
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                # Save best model checkpoint
                if save_path:
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch,
                        'val_loss': best_val_loss,
                        'val_acc': val_metrics['accuracy'],
                        'history': self.history
                    }, save_path)
                    print(f"  → Saved checkpoint to {save_path}")
                
                print(f"  → New best validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"  → Patience: {patience_counter}/{early_stopping_patience}")
                
                if patience_counter >= early_stopping_patience:
                    print(f"\n  Early stopping triggered at epoch {epoch}")
                    break
        
        print("-" * 70)
        print(f"Training complete. Best val loss: {best_val_loss:.4f}\n")
        
        # Return best model state, metrics, and history
        best_metrics = {
            'val_loss': best_val_loss,
            'val_acc': max(self.history['val_acc']) if self.history['val_acc'] else 0.0,
            'train_loss': min(self.history['train_loss']) if self.history['train_loss'] else float('inf')
        }
        
        return self.model, best_metrics, self.history

class BetaVAEFeaturesTrainer:
    """
    Cross-validation trainer for β-VAE with genomic features
    """
    
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
        self.dataset = dataset
        self.config = config
        self.n_folds = n_folds
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Storage for results
        self.fold_results = []
        self.model_paths = []
        
        print(f"Initialized BetaVAEFeaturesTrainer with {n_folds} folds")
        print(f"Device: {self.device}")
    
    def train_fold(self, fold_idx, train_dataset, val_dataset, class_weights=None):
        """
        Train a single fold
        
        Returns:
            trained_model: The trained model
            fold_metrics: Dictionary of metrics
            model_save_path: Path to saved model checkpoint
        """
        print(f"\n{'='*60}")
        print(f"Training Fold {fold_idx + 1}/{self.n_folds}")
        print(f"{'='*60}")

        # DEBUG: Check for leakage
        print("\n=== CHECKING FOR DATA LEAKAGE ===")
        # Get actual indices from Subset
        train_indices = train_dataset.indices
        val_indices = val_dataset.indices
        
        # Get transcript IDs
        train_ids = set([self.dataset.sequences[i].id.split('|')[0] for i in train_indices])
        val_ids = set([self.dataset.sequences[i].id.split('|')[0] for i in val_indices])
        
        overlap = train_ids & val_ids
        print(f"Train: {len(train_ids)} unique transcripts")
        print(f"Val: {len(val_ids)} unique transcripts")
        print(f"Overlap: {len(overlap)} transcripts")
        
        if len(overlap) > 0:
            print(f"⚠️ WARNING: {len(overlap)} transcripts in BOTH train and val!")
            print(f"Sample overlaps: {list(overlap)[:5]}")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('training', 'batch_size'),
            shuffle=True,
            num_workers=self.config.get('training', 'num_workers', default=4),
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('training', 'batch_size'),
            shuffle=False,
            num_workers=self.config.get('training', 'num_workers', default=4),
            pin_memory=True
        )
        
        print(f"\nDatasets:")
        print(f"  Train: {len(train_dataset):,} samples")
        print(f"  Val:   {len(val_dataset):,} samples")

        # Create model save directory
        save_dir = Path(self.config.get('output', 'experiment_name')) / 'models'
        save_dir.mkdir(exist_ok=True, parents=True)
        model_save_path = save_dir / f'fold_{fold_idx}_best.pt'
        
        # Create new model
        model = self.model_builder()
        
        # Create trainer
        trainer = SingleFoldBetaVAEFeaturesTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=self.config.get('training', 'learning_rate'),
            weight_decay=self.config.get('training', 'weight_decay'),
            alpha=self.config.get('training', 'alpha'),
            beta=self.config.get('training', 'beta'),
            gamma_classification=self.config.get('training', 'gamma_classification'),
            lambda_recon=self.config.get('training', 'lambda_recon'),
            reconstruction_loss=self.config.get('training', 'reconstruction_loss'),
            class_weights=class_weights,
            device=self.device
        )
        
        # Train
        model, best_metrics, history = trainer.train(
            num_epochs=self.config.get('training', 'num_epochs'),
            early_stopping_patience=self.config.get('training', 'early_stopping_patience'),
            save_path=model_save_path
        )

        # Reload best model for evaluation
        checkpoint = torch.load(model_save_path, map_location=self.device)
        model = self.model_builder()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        print(f"  Loaded checkpoint from epoch {checkpoint['epoch']} (val_acc: {checkpoint['val_acc']:.4f})")
    
        # Get detailed metrics on validation set
        val_loader_eval = DataLoader(
            val_dataset,
            batch_size=self.config.get('training', 'batch_size'),
            shuffle=False,
            num_workers=self.config.get('training', 'num_workers', default=4)
        )
        
        val_preds, val_labels = self._get_predictions(model, val_loader_eval)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average='binary'
        )
        cm = confusion_matrix(val_labels, val_preds)
        
        fold_metrics = {
            'fold': fold_idx,
            'val_loss': best_metrics['val_loss'],
            'val_acc': best_metrics['val_acc'],
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
        
        print(f"\nFold {fold_idx + 1} Results:")
        print(f"  Val Loss: {best_metrics['val_loss']:.4f}")
        print(f"  Val Acc:  {best_metrics['val_acc']:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Model saved to: {model_save_path}")
        
        return model, fold_metrics, str(model_save_path)
    
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
    
    def _get_predictions(self, model, loader):
        """Get predictions from model"""
        model.eval()
        all_preds = []
        all_labels = []

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                print(f"{name}: running_mean={module.running_mean[:3]}, running_var={module.running_var[:3]}")
                break  # Just print first one
        
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Prediction...", leave=False):
                sequences = batch['sequence'].to(self.device)
                te_features = batch['te_features'].to(self.device)
                nonb_features = batch['nonb_features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = model(sequences, te_features=te_features,
                              nonb_features=nonb_features, deterministic=True)
                logits = outputs['logits']
                preds = logits.argmax(dim=1)
                
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        return all_preds, all_labels
    
    def cross_validate(self, sequences, labels, stratify_by_length=True):
        """
        Perform k-fold cross-validation
        
        Args:
            sequences: List of sequence records
            labels: List of labels ('lncRNA' or 'protein_coding')
            stratify_by_length: Whether to stratify by both class and length
        
        Returns:
            cv_results: Dictionary with aggregated results
        """
        print(f"\n{'='*80}")
        print(f"Starting {self.n_folds}-Fold Cross-Validation")
        print(f"{'='*80}")
        print(f"Total samples: {len(sequences)}")
        print(f"Stratify by length: {stratify_by_length}")

        from collections import Counter
        label_counts = Counter(labels)
        print(f"Class distribution: {label_counts}")
        print(f" lnc: {label_counts.get('lnc', 0)} ({label_counts.get('lnc', 0)/len(labels)*100:.1f}%)")
        print(f" pc:  {label_counts.get('pc', 0)} ({label_counts.get('pc', 0)/len(labels)*100:.1f}%)")
        
        # Create stratification groups
        if stratify_by_length:
            n_bins = self.config.get('training', 'n_bins', default=5)
            strat_groups = create_length_stratified_groups(sequences, labels, n_bins=n_bins)
            print(f"Created {len(set(strat_groups))} stratification groups ({n_bins} length bins)")
        else:
            strat_groups = labels
        
        # Initialize k-fold splitter
        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.config.get('training', 'random_state', default=42)
        )
        
        # Perform cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(sequences, strat_groups)):
            print(f"\nFold {fold_idx + 1} split:")
            print(f"  Train: {len(train_idx)} samples")
            print(f"  Val:   {len(val_idx)} samples")

            print(f"\nValidation split diagnostic:")
            print(f"  First 10 val indices: {list(val_idx[:10])}")
            print(f"  Last 10 val indices: {list(val_idx[-10:])}")
            print(f"  Val set size: {len(val_idx)}")
            
            # Create subset datasets
            train_dataset = Subset(self.dataset, train_idx)
            val_dataset = Subset(self.dataset, val_idx)
            
            # Compute class weights
            train_labels = [labels[i] for i in train_idx]
            label_to_idx = {'lnc': 0, 'pc': 1}
            train_labels_numeric = [label_to_idx[l] for l in train_labels]
            class_weights = compute_class_weight(
                'balanced',
                classes=np.array([0, 1]),
                y=train_labels_numeric
            )
            print(f"  Class weights: lnc={class_weights[0]:.3f}, pc={class_weights[1]:.3f}")
            
            # Train fold
            model, fold_metrics, model_path = self.train_fold(
                fold_idx, train_dataset, val_dataset, class_weights
            )
            
            self.model_paths.append({
                'fold': fold_idx,
                'path': model_path,
                'val_acc': fold_metrics['val_acc'],
                'val_loss': fold_metrics['val_loss']
            })
            
            self.fold_results.append(fold_metrics)
        
        # Aggregate results
        cv_results = self._aggregate_results()
        
        # Save model paths
        model_paths_df = pd.DataFrame(self.model_paths)
        output_dir = Path(self.config.get('output', 'experiment_name'))
        model_paths_df.to_csv(output_dir / 'model_paths.csv', index=False)
        
        # Identify best fold
        best_fold_idx = model_paths_df['val_loss'].idxmin()
        self.best_model_path = model_paths_df.iloc[best_fold_idx]['path']
        self.best_fold_idx = best_fold_idx
        
        print(f"\n{'='*80}")
        print(f"BEST FOLD IDENTIFIED")
        print(f"{'='*80}")
        print(f"Fold {best_fold_idx + 1}:")
        print(f"  Val Loss: {model_paths_df.iloc[best_fold_idx]['val_loss']:.4f}")
        print(f"  Val Acc:  {model_paths_df.iloc[best_fold_idx]['val_acc']:.4f}")
        print(f"  Model: {self.best_model_path}")
        print(f"{'='*80}\n")
        
        return cv_results
    
    def _aggregate_results(self):
        """Aggregate results across all folds"""
        metrics_names = ['val_acc', 'precision', 'recall', 'f1']
        
        aggregated = {}
        for metric in metrics_names:
            values = [fold[metric] for fold in self.fold_results]
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
        
        print(f"\n{'='*80}")
        print(f"Cross-Validation Results Summary")
        print(f"{'='*80}")
        for metric in metrics_names:
            mean = aggregated[metric]['mean']
            std = aggregated[metric]['std']
            print(f"{metric.upper():12s}: {mean:.4f} ± {std:.4f}")
        
        return aggregated
    
    def save_results(self, output_dir):
        """Save all CV results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save fold results
        fold_results_serializable = []
        for fold in self.fold_results:
            fold_copy = fold.copy()
            fold_copy['confusion_matrix'] = fold_copy['confusion_matrix'].tolist()
            fold_results_serializable.append(fold_copy)
        
        with open(output_dir / 'fold_results.json', 'w') as f:
            json.dump(fold_results_serializable, f, indent=2)
        
        print(f"\nSaved fold results to: {output_dir / 'fold_results.json'}")

    # Add to trainers/beta_vae_features_trainer.py

    def extract_embeddings_from_best_fold(self, output_path):
        """
        Extract embeddings from the best trained model
        
        Args:
            output_path: Where to save embeddings (.npz file)
        
        Returns:
            Path to saved embeddings
        """        
        if not hasattr(self, 'best_model_path'):
            raise ValueError("No best model found. Run cross_validate() first.")
        
        print("\n" + "="*80)
        print("EXTRACTING EMBEDDINGS FROM BEST MODEL")
        print("="*80)
        print(f"Loading model from: {self.best_model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        
        # Rebuild model
        model = self.model_builder()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded successfully:")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
        print(f"  Val Acc: {checkpoint['val_acc']:.4f}")
        
        # Create dataloader for full dataset
        loader = DataLoader(
            self.dataset,
            batch_size=256,
            shuffle=False,
            num_workers=1
        )
        
        print(f"\nExtracting embeddings for {len(self.dataset)} samples...")
        
        # Extract embeddings
        all_embeddings = []
        all_logits = []
        all_labels = []
        all_transcript_ids = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting"):
                sequences = batch['sequence'].to(self.device)
                te_features = batch['te_features'].to(self.device)
                nonb_features = batch['nonb_features'].to(self.device)
                
                # Get outputs
                outputs = model(sequences, te_features=te_features,
                              nonb_features=nonb_features, deterministic=True)
                
                # Use latent mean as embedding
                embeddings = outputs['mu']
                logits = outputs['logits']
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_logits.append(logits.cpu().numpy())
                all_labels.append(batch['label'].numpy())
                all_transcript_ids.extend(batch['transcript_id'])
        
        # Concatenate
        embeddings = np.vstack(all_embeddings)
        logits = np.vstack(all_logits)
        labels = np.concatenate(all_labels)
        
        # Get predictions and confidences
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        predictions = logits.argmax(axis=1)
        confidences = probs.max(axis=1)
        
        print(f"\nExtracted:")
        print(f"  Embeddings: {embeddings.shape}")
        print(f"  Predictions: {predictions.shape}")
        
        # Save everything
        output_path = Path(output_path)
        np.savez(
            output_path,
            embeddings=embeddings,
            logits=logits,
            probs=probs,
            predictions=predictions,
            confidences=confidences,
            labels=labels,
            transcript_ids=np.array(all_transcript_ids, dtype=object)
        )
        
        print(f"\n  Saved embeddings to: {output_path}")
        
        return str(output_path)