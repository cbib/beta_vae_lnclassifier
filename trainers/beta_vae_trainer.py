#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainer for beta-VAE with Contrastive Learning

Supports:
- Cross-validation (n_folds >= 2)
- Single train/val split (n_folds = 1)
- Model checkpointing
- Hard case tracking
- Embedding extraction
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from models.beta_vae_contrastive_model import BetaVAE_Contrastive_Loss


class SingleFoldBetaVAETrainer:
    """
    Trainer for a single fold with beta-VAE + contrastive learning
    """
    
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 learning_rate=5e-4,
                 weight_decay=1e-4,
                 alpha=1.0,
                 beta=4.0,
                 gamma_contrastive=0.05,
                 gamma_classification=1.0,
                 temperature=0.15,
                 reconstruction_loss='mse',
                 class_weights=None,
                 device=None):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # Loss function
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        self.criterion = BetaVAE_Contrastive_Loss(
            alpha=alpha,
            beta=beta,
            gamma_contrastive=gamma_contrastive,
            gamma_classification=gamma_classification,
            temperature=temperature,
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
            patience=3
        )
        
        # History
        self.history = {
            'train_loss': [],
            'train_recon': [],
            'train_kl': [],
            'train_contrastive': [],
            'train_classification': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, epoch_num):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_contrastive = 0
        total_classification = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch_num} [Train]', leave=False)
        for batch in pbar:
            sequences = batch['sequence'].to(self.device)
            biotype_labels = batch['biotype_label'].to(self.device)
            class_labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            x_recon, mu, logvar, z, z_proj, logits = self.model(sequences, return_all=True)
            
            # Compute loss
            loss, loss_dict = self.criterion(
                sequences, x_recon, mu, logvar, z_proj, logits,
                biotype_labels, class_labels
            )
            
            # Backward
            loss.backward()
            
            # Gradient clipping (important for VAE stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track
            total_loss += loss_dict['total']
            total_recon += loss_dict['reconstruction']
            total_kl += loss_dict['kl']
            total_contrastive += loss_dict['contrastive']
            total_classification += loss_dict['classification']
            
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'recon': f"{loss_dict['reconstruction']:.4f}",
                'kl': f"{loss_dict['kl']:.4f}",
                'cont': f"{loss_dict['contrastive']:.4f}",
                'cls': f"{loss_dict['classification']:.4f}"
            })
        
        n_batches = len(self.train_loader)
        return {
            'loss': total_loss / n_batches,
            'reconstruction': total_recon / n_batches,
            'kl': total_kl / n_batches,
            'contrastive': total_contrastive / n_batches,
            'classification': total_classification / n_batches
        }
    
    def evaluate(self, epoch_num):
        self.model.eval()
        
        all_logits = []
        all_labels = []
        total_loss = 0
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch_num} [Val]', leave=False)
        with torch.no_grad():
            for batch in pbar:
                sequences = batch['sequence'].to(self.device)
                class_labels = batch['label'].to(self.device)

                #self.optimizer.zero_grad(set_to_none=True)
                
                logits = self.model(sequences, return_embeddings=False)
                
                loss = F.cross_entropy(logits, class_labels)
                total_loss += loss.item()
                
                # Move to CPU and delete GPU tensors immediately
                all_logits.append(logits.cpu())
                all_labels.append(class_labels.cpu())
                
                # Explicitly delete GPU tensors
                del sequences, class_labels, logits
                
                pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
        
        # Rest stays the same
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
            self.history['train_contrastive'].append(train_metrics['contrastive'])
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
                  f"cont: {train_metrics['contrastive']:.4f}, "
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


# The BetaVAE_Trainer class will be similar to ContrastiveTrainer
# but using SingleFoldBetaVAETrainer instead of SingleFoldTrainer