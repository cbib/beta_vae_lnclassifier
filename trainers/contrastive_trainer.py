#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-validation trainer with contrastive learning
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torch.nn import functional as F
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from collections import defaultdict
from pathlib import Path

from data.preprocessing import RNASequenceBiotypeDataset
from data.cv_utils import create_length_stratified_groups, get_cv_splitter
from trainers.contrastive_loss import CombinedLoss
from trainers.bounded_stratified_sampler import create_bounded_stratified_sampler

from .beta_vae_trainer import SingleFoldBetaVAETrainer

class SingleFoldTrainer:
    """
    Trainer for a single fold with contrastive learning
    """
    
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 learning_rate=1e-4,
                 weight_decay=1e-5,
                 lambda_contrastive=1.0,
                 lambda_classification=1.0,
                 temperature=0.07,
                 class_weights=None,
                 device=None,
                 max_oversample=10):
        
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
        
        self.criterion = CombinedLoss(
            temperature=temperature,
            lambda_contrastive=lambda_contrastive,
            lambda_classification=lambda_classification,
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
            'train_contrastive': [],
            'train_classification': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, epoch_num):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_contrastive = 0
        total_classification = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch_num} [Train]', leave=False)
        for batch in pbar:
            sequences = batch['sequence'].to(self.device)
            biotype_labels = batch['biotype_label'].to(self.device)
            class_labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            logits, embeddings, projections = self.model(sequences, return_embeddings=True)
            
            # Compute loss
            loss, loss_dict = self.criterion(
                projections,
                logits,
                biotype_labels,
                class_labels
            )
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Track
            total_loss += loss_dict['total']
            total_contrastive += loss_dict['contrastive']
            total_classification += loss_dict['classification']
            
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'cont': f"{loss_dict['contrastive']:.4f}",
                'cls': f"{loss_dict['classification']:.4f}"
            })
        
        n_batches = len(self.train_loader)
        return {
            'loss': total_loss / n_batches,
            'contrastive': total_contrastive / n_batches,
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
                class_labels = batch['label'].to(self.device)
                
                # Forward (classification only for eval)
                logits = self.model(sequences, return_embeddings=False)
                
                # Loss
                loss = F.cross_entropy(logits, class_labels)
                total_loss += loss.item()
                
                # Store predictions
                all_logits.append(logits.cpu())
                all_labels.append(class_labels.cpu())
                
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
        """Full training loop"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\nTraining for {num_epochs} epochs (patience={early_stopping_patience})")
        print("-" * 70)
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_contrastive'].append(train_metrics['contrastive'])
            self.history['train_classification'].append(train_metrics['classification'])
            
            # Evaluate
            val_metrics = self.evaluate(epoch)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            # Print epoch summary
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} "
                  f"(cont: {train_metrics['contrastive']:.4f}, "
                  f"cls: {train_metrics['classification']:.4f}) | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f}")
            
            # Learning rate scheduling
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_metrics['loss'])
            new_lr = self.optimizer.param_groups[0]['lr']
            
            if new_lr != current_lr:
                print(f"  → Learning rate reduced: {current_lr:.2e} → {new_lr:.2e}")
            
            # Early stopping
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


class ContrastiveTrainer:
    """
    Cross-validation trainer with contrastive learning and hard case tracking
    """
    
    def __init__(self, model_builder, preprocessor, config,
                 n_folds=5, misc_threshold=0.6, device=None,
                 max_oversample=10):
        """
        Args:
            model_builder: Function that returns a new model instance
            preprocessor: SequencePreprocessor instance
            config: Configuration object
            n_folds: Number of cross-validation folds
            misc_threshold: Confidence threshold for misc assignment
            device: torch device
        """
        self.model_builder = model_builder
        self.preprocessor = preprocessor
        self.config = config
        self.n_folds = n_folds
        self.misc_threshold = misc_threshold
        self.max_oversample = max_oversample
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Storage for results
        self.fold_results = []
        self.sample_predictions = defaultdict(list)
        self.hard_cases = None
        self.model_paths = []
        
        print(f"Initialized ContrastiveTrainer with {n_folds} folds")
        print(f"Misc threshold: {misc_threshold}")
        print(f"Device: {self.device}")
    
    def train_fold(self, fold_idx, train_dataset, val_dataset, train_biotypes, class_weights=None):
        """
        Train a single fold with contrastive learning
        
        Returns:
            trained_model: The trained model
            fold_metrics: Dictionary of metrics
            model_save_path: Path to saved model checkpoint
        """
        print(f"\n{'='*60}")
        print(f"Training Fold {fold_idx + 1}/{self.n_folds}")
        print(f"{'='*60}")
        
        # Create dataloaders
        if self.max_oversample is not None:
            train_sampler = create_bounded_stratified_sampler(
            train_biotypes,
            max_oversample=self.max_oversample,
            verbose=(fold_idx == 0)
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.get('training', 'batch_size'),
                num_workers=self.config.get('training', 'num_workers', default=1),
                sampler=train_sampler
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.get('training', 'batch_size'),
                shuffle=True,
                num_workers=self.config.get('training', 'num_workers', default=1)
            )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('training', 'batch_size'),
            shuffle=False,
            num_workers=self.config.get('training', 'num_workers', default=1)
        )

        save_dir = Path(self.config.get('output', 'experiment_name')) / 'models'
        save_dir.mkdir(exist_ok=True, parents=True)
        model_save_path = save_dir / f'fold_{fold_idx}_best.pt'
        
        # Create new model
        model = self.model_builder()
        
        # Get architecture type from config
        architecture = self.config.get('model', 'architecture', default='cnn')
        
        # Choose trainer based on architecture
        if architecture == 'beta_vae':
            # Use beta-VAE trainer
            trainer = SingleFoldBetaVAETrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                learning_rate=self.config.get('training', 'learning_rate'),
                weight_decay=self.config.get('training', 'weight_decay', default=0.0001),
                alpha=self.config.get('training', 'alpha', default=1.0),
                beta=self.config.get('training', 'beta', default=4.0),
                gamma_contrastive=self.config.get('training', 'gamma_contrastive', default=0.05),
                gamma_classification=self.config.get('training', 'gamma_classification', default=1.0),
                temperature=self.config.get('training', 'temperature', default=0.15),
                reconstruction_loss=self.config.get('training', 'reconstruction_loss', default='mse'),
                class_weights=class_weights,
                device=self.device
            )
        else:
            # Use standard contrastive trainer (existing code)
            trainer = SingleFoldTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                learning_rate=self.config.get('training', 'learning_rate'),
                weight_decay=self.config.get('training', 'weight_decay', default=0.0001),
                lambda_contrastive=self.config.get('training', 'lambda_contrastive', default=0.5),
                lambda_classification=self.config.get('training', 'lambda_classification', default=1.0),
                temperature=self.config.get('training', 'temperature', default=0.1),
                max_oversample=self.config.get('training', 'max_oversample', default=None),
                class_weights=class_weights,
                device=self.device
            )
        
        # Train
        trainer.train(
            num_epochs=self.config.get('training', 'num_epochs'),
            early_stopping_patience=self.config.get('training', 'early_stopping_patience', default=5),
            save_path=model_save_path
        )
        
        # Get validation predictions
        val_loss, val_acc, val_preds, val_labels = self._evaluate_fold(trainer.model, val_loader)
        
        # Get detailed metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average='binary'
        )
        cm = confusion_matrix(val_labels, val_preds)
        
        fold_metrics = {
            'fold': fold_idx,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
        
        print(f"\nFold {fold_idx + 1} Results:")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Acc:  {val_acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Model saved to: {model_save_path}")
        
        return trainer.model, fold_metrics, str(model_save_path)
    
    def _get_probs(self, model, loader):
        """Returns softmax probabilities and labels as numpy arrays."""
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="  Ensemble pass", leave=False):
                sequences = batch['sequence'].to(self.device)
                labels    = batch['label'].to(self.device)
                logits    = model(sequences)          # contrastive model interface
                probs     = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        return np.vstack(all_probs), np.concatenate(all_labels)

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

        n_samples = len(test_dataset)
        sum_probs = np.zeros((n_samples, 2), dtype=np.float64)
        labels_arr = None

        for fold_info in self.model_paths:
            checkpoint = torch.load(fold_info['path'], map_location=self.device)
            model = self.model_builder()
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            print(f"  Fold {fold_info['fold']} | epoch={checkpoint['epoch']} "
                f"val_acc={checkpoint['val_acc']:.4f}")

            fold_probs, fold_labels = self._get_probs(model, test_loader)
            sum_probs  += fold_probs
            labels_arr  = fold_labels

        avg_probs   = sum_probs / len(self.model_paths)
        predictions = avg_probs.argmax(axis=1)

        acc                       = accuracy_score(labels_arr, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_arr, predictions, average='binary'
        )
        cm = confusion_matrix(labels_arr, predictions)

        return {
            'accuracy':          float(acc),
            'precision':         float(precision),
            'recall':            float(recall),
            'f1':                float(f1),
            'confusion_matrix':  cm.tolist(),
            'n_samples':         int(n_samples),
            'n_lncrna':          int((labels_arr == 0).sum()),
            'n_pcrna':           int((labels_arr == 1).sum()),
            'n_folds_ensembled': len(self.model_paths),
        }
    
    def _evaluate_fold(self, model, loader):
        """Evaluate model on validation set"""
        model.eval()
        
        all_logits = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in loader:
                sequences = batch['sequence'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass (no contrastive loss during eval)
                logits = model(sequences)
                
                # Classification loss only
                loss = F.cross_entropy(logits, labels)
                total_loss += loss.item()
                
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
        
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        predictions = all_logits.argmax(dim=1)
        
        acc = accuracy_score(all_labels, predictions)
        
        return total_loss / len(loader), acc, predictions.numpy(), all_labels.numpy()
    
    def _predict_with_confidence(self, model, loader):
        """Get predictions with confidence scores"""
        model.eval()
        predictions = []
        confidences = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc='Predicting with confidence...', leave=False):
                sequences = batch['sequence'].to(self.device)
                
                outputs = model(sequences)
                
                # Get probabilities
                probs = torch.softmax(outputs, dim=1)
                max_probs, preds = torch.max(probs, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                confidences.extend(max_probs.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return predictions, confidences, all_probs
    
    def cross_validate(self, sequences, labels, biotypes, stratify_by_length=True):
        """
        Perform k-fold cross-validation with contrastive learning
        
        Args:
            sequences: List of sequence records
            labels: List of labels ('lnc' or 'pc')
            biotypes: List of biotype strings
            stratify_by_length: Whether to stratify by both class and length
        
        Returns:
            cv_results: Dictionary with aggregated results
        """
        print(f"\n{'='*80}")
        print(f"Starting {self.n_folds}-Fold Cross-Validation with Contrastive Learning")
        print(f"{'='*80}")
        print(f"Total samples: {len(sequences)}")
        print(f"Stratify by length: {stratify_by_length}")
        
        unique_biotypes = sorted(set(biotypes))
        self.biotype_to_idx = {bt: idx for idx, bt in enumerate(unique_biotypes)}
        print(f"\nCreated global biotype mapping with {len(unique_biotypes)} unique biotypes")
        print(f"Biotype index range: 0 - {len(unique_biotypes)-1}")

        print(f"\nExample biotype mappings:")
        for i, (bt, idx) in enumerate(list(self.biotype_to_idx.items())[:10]):
            print(f"  {bt}: {idx}")
        if len(unique_biotypes) > 10:
            print(f"  ... and {len(unique_biotypes)-10} more")

        # Create stratification groups
        if stratify_by_length:
            strat_groups = create_length_stratified_groups(sequences, labels)
            print(f"Created {len(set(strat_groups))} stratification groups")
        else:
            strat_groups = labels
        
        # Initialize k-fold splitter
        skf = get_cv_splitter(
            n_folds=self.n_folds,
            random_state=self.config.get('training', 'random_state', default=42)
        )
        
        # Perform cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(sequences, strat_groups)):
            # Get train/val data
            train_seqs = [sequences[i] for i in train_idx]
            val_seqs = [sequences[i] for i in val_idx]
            train_labels = [labels[i] for i in train_idx]
            val_labels = [labels[i] for i in val_idx]
            train_biotypes = [biotypes[i] for i in train_idx]
            val_biotypes = [biotypes[i] for i in val_idx]
            
            print(f"\nFold {fold_idx + 1} split:")
            print(f"  Train: {len(train_seqs)} samples")
            print(f"  Val:   {len(val_seqs)} samples")
            
            # Compute class weights
            label_to_idx = {'lnc': 0, 'pc': 1}
            train_labels_numeric = [label_to_idx[l] for l in train_labels]
            class_weights = compute_class_weight(
                'balanced',
                classes=np.array([0, 1]),
                y=train_labels_numeric
            )
            print(f"  Class weights: lnc={class_weights[0]:.3f}, pc={class_weights[1]:.3f}")
            
            # Create datasets
            train_dataset = RNASequenceBiotypeDataset(
                train_seqs, train_labels, self.preprocessor, biotype_labels=train_biotypes, biotype_to_idx=self.biotype_to_idx
            )
            val_dataset = RNASequenceBiotypeDataset(
                val_seqs, val_labels, self.preprocessor, biotype_labels=val_biotypes, biotype_to_idx=self.biotype_to_idx
            )
            
            # Train fold
            model, fold_metrics, model_path = self.train_fold(
                fold_idx, train_dataset, val_dataset, train_biotypes, class_weights
            )

            self.model_paths.append({
                'fold': fold_idx,
                'path': model_path,
                'val_acc': fold_metrics['val_acc'],
                'val_loss': fold_metrics['val_loss']
            })
            
            # Get predictions with confidence
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.get('training', 'batch_size'),
                shuffle=False,
                num_workers=self.config.get('training', 'num_workers', default=1)
            )
            predictions, confidences, probs = self._predict_with_confidence(model, val_loader)
            
            # Store predictions for each validation sample
            for i, val_sample_idx in enumerate(val_idx):
                self.sample_predictions[val_sample_idx].append({
                    'fold': fold_idx,
                    'prediction': predictions[i],
                    'confidence': confidences[i],
                    'probs': probs[i],
                    'true_label': label_to_idx[val_labels[i]],
                    'biotype': val_biotypes[i]
                })
            
            # Store fold results
            self.fold_results.append(fold_metrics)
        
         # Save model paths and identify best fold
        model_paths_df = pd.DataFrame(self.model_paths)
        output_dir = Path(self.config.get('output', 'experiment_name'))
        model_paths_df.to_csv(output_dir / 'model_paths.csv', index=False)
        
        # Find best fold
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

        # Aggregate results
        cv_results = self._aggregate_results()
        
        # Identify hard cases
        self.hard_cases = self._identify_hard_cases(sequences, labels, biotypes)
        
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
    
    def _identify_hard_cases(self, sequences, labels, biotypes, consistency_threshold=0.6):
        """Identify hard cases based on single-fold predictions"""
        
        hard_cases = []
        label_to_idx = {'lnc': 0, 'pc': 1}
        
        for sample_idx, predictions in self.sample_predictions.items():
            # With single predictions, should only have 1 entry
            assert len(predictions) == 1, f"Expected 1 prediction, got {len(predictions)}"
            
            pred = predictions[0]
            confidence = pred['confidence']
            prediction = pred['prediction']
            true_label = pred['true_label']
            biotype = pred['biotype']
            
            # Simple hard case criteria for single predictions
            is_hard_case = (
                confidence < consistency_threshold or  # Low confidence
                prediction != true_label               # OR incorrect
            )
            
            seq_length = len(str(sequences[sample_idx].seq))
            true_label_str = 'lnc' if true_label == 0 else 'pc'
            
            hard_cases.append({
                'sample_idx': sample_idx,
                'transcript_id': sequences[sample_idx].id,
                'true_label': true_label_str,
                'biotype': biotype,
                'sequence_length': seq_length,
                'confidence': confidence,  # Not avg_confidence
                'prediction': prediction,
                'is_correct': prediction == true_label,
                'is_hard_case': is_hard_case
            })
        
        df = pd.DataFrame(hard_cases)
        
        # Summary
        n_hard = df['is_hard_case'].sum()
        n_errors = (~df['is_correct']).sum()
        n_low_conf = df[df['confidence'] < consistency_threshold].shape[0]
        
        print(f"\nHard Cases Summary:")
        print(f"  Total samples: {len(df):,}")
        print(f"  Hard cases: {n_hard:,} ({100*n_hard/len(df):.1f}%)")
        print(f"    Errors: {n_errors:,} ({100*n_errors/len(df):.1f}%)")
        print(f"    Low confidence: {n_low_conf:,} ({100*n_low_conf/len(df):.1f}%)")
        
        return df
    
    def analyze_hard_cases(self, output_dir):
        """Comprehensive analysis of hard cases"""
        if self.hard_cases is None:
            print("No hard cases to analyze. Run cross_validate() first.")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save hard cases
        hard_cases_only = self.hard_cases[self.hard_cases['is_hard_case']].copy()
        hard_cases_only.to_csv(output_dir / 'hard_cases.csv', index=False)
        
        self.hard_cases.to_csv(output_dir / 'all_sample_predictions.csv', index=False)
        
        print(f"\nSaved hard cases to: {output_dir / 'hard_cases.csv'}")
        
        # Generate plots
        self._plot_hard_case_analysis(hard_cases_only, output_dir)
    
    def _plot_hard_case_analysis(self, hard_cases_df, output_dir):
        """Generate analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        easy_cases = self.hard_cases[~self.hard_cases['is_hard_case']]
        
        # 1. Confidence distribution
        ax = axes[0, 0]
        ax.hist(easy_cases['avg_confidence'], bins=30, alpha=0.5, label='Easy', color='green')
        ax.hist(hard_cases_df['avg_confidence'], bins=30, alpha=0.5, label='Hard', color='red')
        ax.axvline(self.misc_threshold, color='black', linestyle='--',
                   label=f'Threshold ({self.misc_threshold})')
        ax.set_xlabel('Average Confidence')
        ax.set_ylabel('Count')
        ax.set_title('Confidence Distribution')
        ax.legend()
        
        # 2. Length distribution
        ax = axes[0, 1]
        ax.hist(easy_cases['sequence_length'], bins=30, alpha=0.5, label='Easy', color='green')
        ax.hist(hard_cases_df['sequence_length'], bins=30, alpha=0.5, label='Hard', color='red')
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Count')
        ax.set_title('Length Distribution')
        ax.legend()
        
        # 3. Error rate
        ax = axes[1, 0]
        ax.hist(hard_cases_df['error_rate'], bins=20, color='orange', edgecolor='black')
        ax.set_xlabel('Error Rate Across Folds')
        ax.set_ylabel('Count')
        ax.set_title('Error Rate Distribution (Hard Cases)')
        
        # 4. Hard cases by class
        ax = axes[1, 1]
        class_counts = hard_cases_df['true_label'].value_counts()
        ax.bar(class_counts.index, class_counts.values, color=['blue', 'orange'])
        ax.set_xlabel('True Label')
        ax.set_ylabel('Count')
        ax.set_title('Hard Cases by Class')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'hard_case_analysis.png', dpi=1200, bbox_inches='tight')
        plt.close()
    
    def save_results(self, output_dir):
        """Save all CV results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save fold results
        with open(output_dir / 'fold_results.json', 'w') as f:
            fold_results_serializable = []
            for fold in self.fold_results:
                fold_copy = fold.copy()
                fold_copy['confusion_matrix'] = fold_copy['confusion_matrix'].tolist()
                fold_results_serializable.append(fold_copy)
            
            json.dump(fold_results_serializable, f, indent=2)
        
        print(f"\nSaved fold results to: {output_dir / 'fold_results.json'}")

    def extract_embeddings_from_best_fold(self, sequences, labels, biotypes, output_path):
        """
        Extract embeddings from the best trained model
        
        Args:
            sequences: All sequences
            labels: All labels
            biotypes: All biotype strings
            output_path: Where to save embeddings (.npz file)
        
        Returns:
            Path to saved embeddings
        """
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        
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
        
        # Create dataset for ALL samples
        dataset = RNASequenceBiotypeDataset(
            sequences=sequences,
            labels=labels,
            preprocessor=self.preprocessor,
            biotype_labels=biotypes,
            biotype_to_idx=self.biotype_to_idx
        )
        
        loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
            num_workers=1
        )
        
        print(f"\nExtracting embeddings for {len(dataset)} samples...")
        
        # Extract embeddings
        all_embeddings = []
        all_projections = []
        all_logits = []
        all_labels = []
        all_biotypes = []
        all_transcript_ids = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting"):
                sequences_batch = batch['sequence'].to(self.device)
                
                # Get all outputs
                #logits, embeddings, projections = model(sequences_batch, return_embeddings=True)
                
                mu, logvar = model.encode(sequences)
                logits = model.classifier(mu)
                embeddings = mu  # ← Use deterministic mean
                projections = mu
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_projections.append(projections.cpu().numpy())
                all_logits.append(logits.cpu().numpy())
                all_labels.append(batch['label'].numpy())
                all_biotypes.append(batch['biotype_label'].numpy())
                all_transcript_ids.extend(batch['transcript_id'])
        
        # Concatenate
        embeddings = np.vstack(all_embeddings)
        projections = np.vstack(all_projections)
        logits = np.vstack(all_logits)
        labels = np.concatenate(all_labels)
        biotype_indices = np.concatenate(all_biotypes)
        
        # Get predictions and confidences
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        predictions = logits.argmax(axis=1)
        confidences = probs.max(axis=1)
        
        print(f"\nExtracted:")
        print(f"  Embeddings: {embeddings.shape}")
        print(f"  Projections: {projections.shape}")
        print(f"  Predictions: {predictions.shape}")
        
        # Save everything
        output_path = Path(output_path)
        np.savez(
            output_path,
            embeddings=embeddings,
            projections=projections,
            logits=logits,
            probs=probs,
            predictions=predictions,
            confidences=confidences,
            labels=labels,
            biotype_indices=biotype_indices,
            transcript_ids=np.array(all_transcript_ids, dtype=object)
        )
        
        print(f"\n  Saved embeddings to: {output_path}")
        
        return str(output_path)