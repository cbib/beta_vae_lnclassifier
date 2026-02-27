#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contrastive Loss Functions for RNA Classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.first_call = True
    
    def forward(self, features, labels):
        """
        Compute contrastive loss, gracefully handling samples without positives
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs (same label)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove diagonal (self-similarity)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask
        
        # Count positives per sample
        positives_per_sample = mask.sum(dim=1)
        
        # Debug on first call
        if self.first_call:
            print(f"[SupervisedContrastiveLoss] First call (debug):")
            print(f"  Features shape: {features.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Labels dtype: {labels.dtype}")
            print(f"  Unique labels: {torch.unique(labels).tolist()}")
            print(f"  Num unique: {len(torch.unique(labels))}")
            print(f"  Temperature: {self.temperature}")
            print(f"  Positives per sample (first 10): {positives_per_sample[:10].tolist()}")
            self.first_call = False
        
        # Identify samples with no positives to avoid batch effect
        valid_samples = positives_per_sample > 0
        num_invalid = (valid_samples == 0).sum().item()
        
        # if num_invalid > 0:
            #print(f"  WARNING: {num_invalid} samples have no positive pairs!")
        
        # Only compute loss for samples with enough positives within batch
        if valid_samples.sum() == 0:
            # Entire batch has no positives (rare, but possible)
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Mask out invalid samples
        similarity_matrix = similarity_matrix[valid_samples]
        mask = mask[valid_samples]
        logits_mask = logits_mask[valid_samples]
        positives_per_sample = positives_per_sample[valid_samples]
        
        # Compute exp for numerical stability
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        
        # Log probabilities
        log_prob = similarity_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True))
        
        # Compute mean of log-likelihood over positives
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / positives_per_sample
        
        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos.mean()
        
        return loss
    
class CombinedLoss(nn.Module):
    """
    Combines contrastive learning with classification
    
    Loss = λ_contrastive * L_contrastive + λ_classification * L_classification
    """

    def __init__(self, 
                 temperature=0.07,
                 lambda_contrastive=1.0,
                 lambda_classification=1.0,
                 class_weights=None):
        """
        Args:
            temperature: Temperature for contrastive loss
            lambda_contrastive: Weight for contrastive loss
            lambda_classification: Weight for classification loss
            class_weights: Optional weights for classification (for imbalance)
        """
        super(CombinedLoss, self).__init__()
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=temperature)
        self.classification_loss = nn.CrossEntropyLoss(weight=class_weights)

        self.lambda_contrastive = lambda_contrastive
        self.lambda_classification = lambda_classification

    def forward(self, embeddings, logits, biotype_labels, class_labels):
        """
        Args:
            embeddings: (batch_size, embedding_dim) - L2 normalized embeddings from encoder
            logits: (batch_size, num_classes) - classification logits
            biotype_labels: (batch_size,) - biotype labels for contrastive loss
            class_labels: (batch_size,) - true class labels for classification loss (binary lnc/pc)
        
        Returns:
            total_loss, loss_dict
        """
        loss_contrastive = self.contrastive_loss(embeddings, biotype_labels)
        loss_classification = self.classification_loss(logits, class_labels)

        total_loss = (self.lambda_contrastive * loss_contrastive +
                self.lambda_classification * loss_classification)

        loss_dict = {
            'total' : total_loss.item(),
            'contrastive' : loss_contrastive.item() * self.lambda_contrastive,
            'classification' : loss_classification.item() * self.lambda_classification
        }

        return total_loss, loss_dict
    