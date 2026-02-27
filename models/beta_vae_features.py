# experiments/models/beta_vae_features.py
"""
β-VAE with integrated TE and non-B DNA features.
Features are concatenated to latent representation before classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalSequenceEncoder(nn.Module):
    """CSE encoder with learnable convolutions."""
    
    def __init__(self, input_dim=4, d_model=512, kernel_size=9, max_length=6000):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.kernel_size = kernel_size
        
        # Convolutional encoding
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        
        # Adaptive pooling to handle variable lengths
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        """
        Args:
            x: (batch, input_dim, seq_len)
        Returns:
            h: (batch, d_model)
        """
        h = self.conv(x)  # (batch, d_model, seq_len)
        h = F.relu(h)
        h = self.pool(h)  # (batch, d_model, 1)
        h = h.squeeze(-1)  # (batch, d_model)
        return h


class FeatureProcessor(nn.Module):
    """MLP for processing genomic features."""
    
    def __init__(self, input_dim, hidden_dims, dropout=0.3):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions (e.g., [128, 64, 32])
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout if i < len(hidden_dims) - 1 else dropout * 1.5)  # More dropout on last layer
            ])
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        
    def forward(self, x):
        return self.network(x)


class BetaVAEWithFeatures(nn.Module):
    """
    β-VAE with genomic features, matching the encoder/decoder complexity
    of the contrastive model.
    """
    
    def __init__(self, num_classes, input_dim, input_length, latent_dim=128, 
                 beta=4.0, dropout_rate=0.3, use_cse=True, cse_d_model=512, 
                 cse_kernel_size=9, te_features_dim=170, nonb_features_dim=178,
                 te_processor_hidden=[64, 32], nonb_processor_hidden=[64, 32],
                 classifier_hidden=[128, 64], **kwargs):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.input_length = input_length
        self.latent_dim = latent_dim
        self.beta = beta
        self.use_cse = use_cse
        
        # =====================================================================
        # CSE Pre-encoder (optional, like contrastive model)
        # =====================================================================
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
        
        # =====================================================================
        # Convolutional Encoder (matching contrastive model)
        # =====================================================================
        self.encoder = nn.Sequential(
            # (encoder_input_dim, input_length) -> (64, input_length/4)
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
        
        # Calculate flattened size
        self.flatten_size, self.encoded_length = self._get_flatten_size(encoder_input_dim)
        
        # VAE bottleneck
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # =====================================================================
        # Feature Processors (keep yours, maybe slightly smaller)
        # =====================================================================
        self.te_processor = FeatureProcessor(
            input_dim=te_features_dim,
            hidden_dims=te_processor_hidden,
            dropout=0.0
        )
        
        self.nonb_processor = FeatureProcessor(
            input_dim=nonb_features_dim,
            hidden_dims=nonb_processor_hidden,
            dropout=0.0
        )
        
        # =====================================================================
        # Decoder (matching contrastive model)
        # =====================================================================
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)
        
        if use_cse:
            decoder_output_dim = cse_d_model
        else:
            decoder_output_dim = input_dim
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, self.encoded_length)),
            
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
        
        # CSE Decoder
        if use_cse:
            self.cse_decoder = nn.Sequential(
                nn.Conv1d(cse_d_model, input_dim, kernel_size=cse_kernel_size, padding='same'),
                nn.Sigmoid()
            )
        else:
            self.cse_decoder = None
        
        # =====================================================================
        # Classifier (with combined features)
        # =====================================================================
        combined_dim = (latent_dim + 
                       self.te_processor.output_dim + 
                       self.nonb_processor.output_dim)
        
        classifier_layers = []
        prev_dim = combined_dim
        
        for hidden_dim in classifier_hidden:
            classifier_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        classifier_layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
        
        # Projection head (identity, for compatibility)
        self.projection = nn.Identity()
    
    def _get_flatten_size(self, encoder_input_dim):
        """Calculate flattened size after encoder"""
        with torch.no_grad():
            x = torch.zeros(1, encoder_input_dim, self.input_length)
            x = self.encoder(x)
            flatten_size = x.shape[1]
            encoded_length = flatten_size // 256
            return flatten_size, encoded_length
    
    def encode(self, x):
        """Encode sequence to latent distribution parameters."""
        if self.use_cse:
            x = self.cse_encoder(x)
        
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent to sequence."""
        h = self.fc_decode(z)
        x_recon = self.decoder(h)
        
        if self.use_cse:
            x_recon = self.cse_decoder(x_recon)
        
        # Ensure exact length match
        current_length = x_recon.shape[2]
        if current_length != self.input_length:
            if current_length < self.input_length:
                pad_amount = self.input_length - current_length
                x_recon = F.pad(x_recon, (0, pad_amount), mode='constant', value=0)
            else:
                x_recon = x_recon[:, :, :self.input_length]
        
        return x_recon
    
    def forward(self, x, te_features=None, nonb_features=None, 
                biotypes=None, return_latent=False, deterministic=False):
        """Forward pass."""
        if te_features is None or nonb_features is None:
            raise ValueError("te_features and nonb_features are required")
        
        # Encode
        mu, logvar = self.encode(x)
        
        # Sample or use mean
        if deterministic:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decode(z)
        
        # Process features
        te_embed = self.te_processor(te_features)
        nonb_embed = self.nonb_processor(nonb_features)
        
        # Combine for classification
        combined = torch.cat([z, te_embed, nonb_embed], dim=1)
        logits = self.classifier(combined)
        
        output = {
            'logits': logits,
            'reconstruction': x_recon,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'projection': z  # For compatibility
        }
        
        return output