#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
beta-VAE with Contrastive Learning for RNA Classification

Combines:
1. Reconstruction loss (forces learning of sequence features)
2. KL divergence with beta weighting (encourages disentanglement)
3. Contrastive loss (biotype-aware clustering in latent space)
4. Classification loss (binary lnc vs pc)

Architecture: Conv1D encoder/decoder optimized for 1D sequences
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        
    def forward(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(h))
        return out

class BetaVAE_Contrastive(nn.Module):
    """
    beta-VAE with contrastive learning for RNA sequences
    
    Args:
        input_dim: Number of input channels (5 for one-hot RNA)
        input_length: Maximum sequence length
        latent_dim: Dimension of latent space (default: 128)
        beta: beta parameter for KL weighting (default: 4.0)
        num_classes: Number of classification classes (default: 2)
        dropout_rate: Dropout rate for regularization (default: 0.3)
        use_cse: Whether to use CSE encoding/decoding (default: False)
        cse_d_model: CSE model dimension (if use_cse is True)
        cse_kernel_size: CSE kernel size (if use_cse is True)
    """
    def __init__(self, input_dim=5, input_length=10000, latent_dim=128, 
                 beta=4.0, num_classes=2, dropout_rate=0.3, use_cse=False, cse_d_model=128, cse_kernel_size=9):
        super().__init__()
        
        self.input_dim = input_dim
        self.input_length = input_length
        self.latent_dim = latent_dim
        self.beta = beta
        self.num_classes = num_classes
        self.use_cse = use_cse
        
        if use_cse:
            self.cse_encoder = nn.Sequential(
                nn.Conv1d(input_dim, cse_d_model, kernel_size=cse_kernel_size, padding='same'),
                nn.ReLU(),
                nn.BatchNorm1d(cse_d_model)
            )
            encoder_input_dim = cse_d_model
        else:
            self.cse_encoder = None
            encoder_input_dim = input_dim

        self.encoder_input_dim = encoder_input_dim

        # Encoder: (batch, 5, input_length) -> (batch, latent_dim)
        self.encoder = nn.Sequential(
            # (5, input_length) -> (64, input_length/4)
            nn.Conv1d(encoder_input_dim, 64, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate),
            
            # -> (128, input_length/16)
            nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate),
            
            # -> (256, input_length/64)
            nn.Conv1d(128, 256, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate),
            
            # -> (256, input_length/256)
            nn.Conv1d(256, 256, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Flatten()
        )
        
        # Calculate flattened size and encoded length dynamically
        self.flatten_size, self.encoded_length = self._get_flatten_size()
        
        # Latent space (mean and log-variance)
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder: (batch, latent_dim) -> (batch, 5, input_length)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)

        if use_cse:
            decoder_output_dim = cse_d_model  # Output 128 channels for CSE decoder
        else:
            decoder_output_dim = input_dim  # Output 5 channels (one-hot) directly
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, self.encoded_length)),
            
            # Upsample by 4x each time (total 4^4 = 256x)
            nn.ConvTranspose1d(256, 256, kernel_size=8, stride=4, padding=2, output_padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.ConvTranspose1d(256, 128, kernel_size=8, stride=4, padding=2, output_padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.ConvTranspose1d(128, 64, kernel_size=8, stride=4, padding=2, output_padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.ConvTranspose1d(64, decoder_output_dim, kernel_size=8, stride=4, padding=2, output_padding=0),
            nn.Sigmoid()
        )

        # CSE Decoder (if using CSE, decode back to one-hot)
        if use_cse:
            self.cse_decoder = nn.Sequential(
                nn.Conv1d(cse_d_model, input_dim, kernel_size=cse_kernel_size, padding='same'),
                nn.Sigmoid()
            )
        else:
            self.cse_decoder = None
        
        # Contrastive projection head (from latent space)
        self.projection = ProjectionHead(latent_dim, 128, 64)
        
        # Classifier (from latent space)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
    
    def _get_flatten_size(self):
        """Calculate flattened size and encoded length after encoder"""
        with torch.no_grad():
            x = torch.zeros(1, self.encoder_input_dim, self.input_length)
            x = self.encoder(x)
            flatten_size = x.shape[1]
            # Calculate encoded_length from flatten_size (since last encoder has 256 channels)
            encoded_length = flatten_size // 256
            return flatten_size, encoded_length
    
    def encode(self, x):
        """Encode input to latent distribution parameters"""
        # Apply CSE encoding if enabled
        if self.use_cse:
            x = self.cse_encoder(x)
        
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + eps * sigma"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to reconstruction"""
        h = self.fc_decode(z)
        x_recon = self.decoder(h)
        
        # Apply CSE decoder if enabled (decode to one-hot)
        if self.use_cse:
            x_recon = self.cse_decoder(x_recon)
        
        # Ensure output length exactly matches input_length
        current_length = x_recon.shape[2]
        if current_length != self.input_length:
            # Use adaptive padding/truncation to match exact length
            if current_length < self.input_length:
                # Pad to reach input_length
                pad_amount = self.input_length - current_length
                x_recon = F.pad(x_recon, (0, pad_amount), mode='constant', value=0)
            else:
                # Truncate to input_length
                x_recon = x_recon[:, :, :self.input_length]
        
        return x_recon
    
    def forward(self, x, return_all=False, return_embeddings=False):
        """
        Forward pass
        
        Args:
            x: Input sequences (batch, input_dim, input_length)
            return_all: If True, return all intermediate outputs
        
        Returns:
            If return_all=False: logits only (for inference)
            If return_all=True: (x_recon, mu, logvar, z, z_proj, logits)
            If return_embeddings=True: (logits, z, z_proj)
        """
        # Encode
        mu, logvar = self.encode(x)
        
        # Sample latent
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decode(z)
        
        # Project for contrastive
        # z_proj = self.projection(z)
        # Use mu directly for contrastive (no projection to avoid collapse)
        z_proj = mu  # ← Simple and stable

        # Classify
        logits = self.classifier(z)
        
        if return_all:
            return x_recon, mu, logvar, z, z_proj, logits
        if return_embeddings:
            return logits, z, z_proj
        return logits


class BetaVAE_Contrastive_Loss(nn.Module):
    """
    Combined loss: Reconstruction + beta*KL + Contrastive + Classification
    
    Args:
        alpha: Weight for reconstruction loss (default: 1.0)
        beta: Weight for KL divergence (default: 4.0)
        gamma_contrastive: Weight for contrastive loss (default: 0.05)
        gamma_classification: Weight for classification loss (default: 1.0)
        temperature: Temperature for contrastive loss (default: 0.15)
        reconstruction_loss: 'bce' or 'mse' (default: 'mse')
    """
    def __init__(self, alpha=1.0, beta=4.0, gamma_contrastive=0.05, 
                 gamma_classification=1.0, temperature=0.15, 
                 reconstruction_loss='mse', class_weights=None):
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma_contrastive = gamma_contrastive
        self.gamma_classification = gamma_classification
        self.temperature = temperature
        self.reconstruction_loss_type = reconstruction_loss
        
        # Classification loss with optional class weights
        if class_weights is not None:
            self.classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.classification_criterion = nn.CrossEntropyLoss()
    
    def reconstruction_loss(self, x_recon, x):
        """Compute reconstruction loss"""
        if self.reconstruction_loss_type == 'bce':
            loss_per_element = F.binary_cross_entropy(x_recon, x, reduction='mean')
            # Scale back to per-sequence (beta-VAE uses sum)
            n_elements = x.size(1) * x.size(2)  # 5 * input_length
            return loss_per_element * n_elements
        else:
            loss_per_element = F.mse_loss(x_recon, x, reduction='mean')
            n_elements = x.size(1) * x.size(2)
            return loss_per_element * n_elements
    
    def kl_divergence(self, mu, logvar):
        """
        KL divergence between N(mu, sigma^2) and N(0, 1)
        KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        """
        # Sum over latent dimensions, then average over batch
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl.mean()
    
    def contrastive_loss(self, z_proj, biotype_labels):
        """
        Supervised contrastive loss (biotype-aware)
        
        Args:
            z_proj: Projected features (batch, projection_dim)
            biotype_labels: Biotype labels (batch,)
        """
        device = z_proj.device
        batch_size = z_proj.size(0)
        
        # Normalize projections
        z_proj = F.normalize(z_proj, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(z_proj, z_proj.T) / self.temperature
        
        # Create mask for positive pairs (same biotype)
        biotype_labels = biotype_labels.contiguous().view(-1, 1)
        mask = torch.eq(biotype_labels, biotype_labels.T).float().to(device)
        
        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(batch_size, device=device)
        
        # Check for positive pairs
        positives_per_sample = mask.sum(dim=1)
        if positives_per_sample.min() == 0:
            # Some samples have no positive pairs, avoid batch effect
            valid_samples = positives_per_sample > 0
            similarity = similarity[valid_samples]
            mask = mask[valid_samples]
            positives_per_sample = positives_per_sample[valid_samples]
            if similarity.size(0) == 0:
                return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute log probabilities
        exp_sim = torch.exp(similarity)
        
        # Sum of similarities excluding self
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / positives_per_sample
        
        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos.mean()
        
        return loss
    
    def forward(self, x, x_recon, mu, logvar, z_proj, logits, 
            biotype_labels, class_labels):
        """
        Compute total loss (only computing terms with non-zero weights)
        
        Returns:
            total_loss, loss_dict
        """
        total = 0.0
        loss_dict = {}
        
        # 1. Reconstruction loss (only if alpha > 0)
        if self.alpha > 0:
            recon_loss = self.reconstruction_loss(x_recon, x)
            loss_dict['reconstruction'] = recon_loss.item()
            total += self.alpha * recon_loss
        else:
            loss_dict['reconstruction'] = 0.0
        
        # 2. KL divergence (only if beta > 0)
        if self.beta > 0:
            kl_loss = self.kl_divergence(mu, logvar)
            loss_dict['kl'] = kl_loss.item()
            total += self.beta * kl_loss
        else:
            loss_dict['kl'] = 0.0
        
        # 3. Contrastive loss (only if gamma_contrastive > 0)
        if self.gamma_contrastive > 0:
            cont_loss = self.contrastive_loss(z_proj, biotype_labels)
            loss_dict['contrastive'] = cont_loss.item()
            total += self.gamma_contrastive * cont_loss
        else:
            loss_dict['contrastive'] = 0.0
        
        # 4. Classification loss (only if gamma_classification > 0)
        if self.gamma_classification > 0:
            cls_loss = self.classification_criterion(logits, class_labels)
            loss_dict['classification'] = cls_loss.item()
            total += self.gamma_classification * cls_loss
        else:
            loss_dict['classification'] = 0.0
        
        # Compute vae_loss for logging (only if components exist)
        vae_loss = 0.0
        if self.alpha > 0:
            vae_loss += loss_dict['reconstruction']
        if self.beta > 0:
            vae_loss += self.beta * (loss_dict['kl'] / max(self.alpha, 1e-8))  # Rescale if needed
        
        loss_dict['total'] = total.item()
        loss_dict['vae_loss'] = vae_loss
        
        return total, loss_dict


def get_beta_vae_contrastive_model(input_dim=5, input_length=10000, latent_dim=128,
                                   beta=4.0, num_classes=2, dropout_rate=0.3, **kwargs):
    """
    Factory function to create beta-VAE with contrastive learning model
    
    Compatible with existing model builder pattern
    """
    return BetaVAE_Contrastive(
        input_dim=input_dim,
        input_length=input_length,
        latent_dim=latent_dim,
        beta=beta,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )


if __name__ == '__main__':
    # Test model with different input lengths
    print("Testing beta-VAE + Contrastive model with adaptive decoder...")
    
    for input_length in [10000, 15000, 20000]:
        print(f"\n{'='*60}")
        print(f"Testing with input_length={input_length}")
        print('='*60)
        
        model = BetaVAE_Contrastive(
            input_dim=5,
            input_length=input_length,
            latent_dim=128,
            beta=4.0
        )
        
        # Test forward pass
        batch_size = 4
        x = torch.randn(batch_size, 5, input_length)
        
        # Inference mode
        logits = model(x, return_all=False)
        print(f"Logits shape: {logits.shape}")  # Should be (4, 2)
        
        # Training mode
        x_recon, mu, logvar, z, z_proj, logits = model(x, return_all=True)
        print(f"Input shape: {x.shape}")
        print(f"Reconstruction shape: {x_recon.shape}")
        
        # CRITICAL: Check that shapes match
        assert x.shape == x_recon.shape, f"Shape mismatch! Input: {x.shape}, Recon: {x_recon.shape}"
        print(f"  Shape match confirmed!")
        
        print(f"Latent (mu) shape: {mu.shape}")
        print(f"Projection shape: {z_proj.shape}")
        
        # Test loss
        criterion = BetaVAE_Contrastive_Loss(
            alpha=1.0,
            beta=4.0,
            gamma_contrastive=0.05,
            gamma_classification=1.0
        )
        
        biotype_labels = torch.randint(0, 5, (batch_size,))
        class_labels = torch.randint(0, 2, (batch_size,))
        
        total_loss, loss_dict = criterion(
            x, x_recon, mu, logvar, z_proj, logits,
            biotype_labels, class_labels
        )
        
        print(f"\nLoss components:")
        for key, value in loss_dict.items():
            print(f"  {key}: {value:.4f}")
    
    print(f"\n{'='*60}")
    print(f"  All tests passed for all input lengths!")
    print('='*60)
    
    # Count parameters for 10000 length
    model = BetaVAE_Contrastive(input_length=10000)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")