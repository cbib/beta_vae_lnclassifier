#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN with Contrastive Learning and optional CSE encoding
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CSELayer(nn.Module):
    """
    Convolutional Sequence Encoding layer
    Converts nucleotide sequences to embeddings via convolution
    """
    def __init__(self, d_model=128, kernel_size=9):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        
        # Convolution: 4 input channels (ACGT) -> d_model dimensions
        self.conv = nn.Conv1d(
            in_channels=4,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=kernel_size,  # Non-overlapping windows
            padding=0
        )
        self.activation = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, length) - ALREADY in Conv1d format
        Returns:
            (batch, d_model, reduced_length)
        """
        # x is already (batch, channels, length) from preprocessor
        
        # If channels=5 (one-hot with padding), take only ACGT (first 4)
        if x.shape[1] == 5:
            x = x[:, :4, :]  # (batch, 4, length)
        
        # Apply convolution - no transpose needed!
        x = self.conv(x)  # (batch, d_model, reduced_length)
        x = self.activation(x)
        
        return x


class CNNContrastive(nn.Module):
    """
    CNN with Contrastive Learning
    Supports both standard one-hot encoding and CSE encoding
    """
    
    def __init__(self,
                 input_length=5000,
                 input_dim=5,
                 embedding_dim=256,
                 projection_dim=128,
                 num_classes=2,
                 dropout_rate=0.3,
                 use_cse=False,
                 cse_d_model=128,
                 cse_kernel_size=9):
        super().__init__()
        
        self.use_cse = use_cse
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        if use_cse:
            # CSE encoding path
            self.cse = CSELayer(d_model=cse_d_model, kernel_size=cse_kernel_size)
            
            # After CSE, sequence length is reduced by kernel_size
            reduced_length = input_length // cse_kernel_size
            
            # CNN encoder on top of CSE embeddings
            self.encoder = nn.Sequential(
                # Conv block 1
                nn.Conv1d(cse_d_model, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                # Conv block 2
                nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                # Global pooling
                nn.AdaptiveAvgPool1d(1)
            )
            
            encoder_output_dim = 512
            
        else:
            # Standard one-hot encoding path
            self.cse = None
            
            self.encoder = nn.Sequential(
                # Conv block 1
                nn.Conv1d(input_dim, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                # Conv block 2
                nn.Conv1d(128, 256, kernel_size=6, stride=5, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                # Global pooling
                nn.AdaptiveAvgPool1d(1)
            )
            
            encoder_output_dim = 256
        
        # Project to embedding dimension
        self.to_embedding = nn.Linear(encoder_output_dim, embedding_dim)
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate + 0.2),  # Higher than 0.5
            nn.Linear(embedding_dim, projection_dim)
        )
        
        # Classification head (lnc vs pc)
        self.classification_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def encode(self, x):
        """
        Encode sequence to embedding
        
        Args:
            x: (batch, channels, length) - already in Conv1d format from preprocessor
        
        Returns:
            embeddings: (batch, embedding_dim)
        """
        if self.use_cse:
            # CSE encoding path
            # x is already (batch, channels, length)
            x = self.cse(x)  # (batch, cse_d_model, reduced_length)
        else:
            # Standard path - x is already in correct format
            # No transpose needed!
            pass
        
        # CNN encoding
        h = self.encoder(x)  # (batch, encoder_output_dim, 1)
        h = h.squeeze(-1)     # (batch, encoder_output_dim)
        
        # Project to embedding
        embeddings = self.to_embedding(h)
        
        return embeddings
    
    def forward(self, x, return_embeddings=False):
        """
        Forward pass
        
        Args:
            x: (batch, channels, length) - already in Conv1d format
            return_embeddings: If True, return embeddings for contrastive loss
        
        Returns:
            If return_embeddings:
                logits, embeddings, projections
            Else:
                logits
        """
        # Get embeddings
        embeddings = self.encode(x)
        
        # Classification logits
        logits = self.classification_head(embeddings)
        
        if return_embeddings:
            # Projection for contrastive learning
            projections = self.projection_head(embeddings)
            projections = F.normalize(projections, dim=1)  # L2 normalize
            
            return logits, embeddings, projections
        else:
            return logits