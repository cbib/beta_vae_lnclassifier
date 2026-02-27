# experiments/models/beta_vae_features.py
"""
β-VAE with integrated TE and non-B DNA features.

Cross-attention fusion: sequence positions (query) attend to TE and NonB
feature tokens (key/value), producing per-position attention weights that
reveal which part of the sequence relied on which genomic feature.

Attention weights shape per transcript: (L_encoded, 2)
    [:, 0] = positional attention to TE features
    [:, 1] = positional attention to NonB features

These are stored in output['attn_weights'] and used for interpretability.
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

class CrossModalAttention(nn.Module):
    def __init__(self, d_model, d_feat, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_feat  = d_feat
        self.scale   = d_feat ** 0.5
        # Project sequence positions DOWN to feature space (32-dim)
        # Much easier to learn than projecting 32→256
        self.query_proj = nn.Linear(d_model, d_feat)
        self.norm       = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)
        # Value projection: features → d_model for residual addition
        self.value_proj = nn.Linear(d_feat, d_model)

    def forward(self, seq_positions, feature_tokens):
        # seq_positions:  (B, L, d_model=256)
        # feature_tokens: (B, 2, d_feat=32)

        # Project queries to low-dim space — scores will have real variance
        q = self.query_proj(seq_positions)          # (B, L, 32)
        scores = torch.bmm(q, feature_tokens.transpose(1, 2)) / self.scale
        # scores: (B, L, 2) — now with meaningful variance
        attn_weights = torch.softmax(scores, dim=-1)

        # Weighted sum of feature tokens, projected back to d_model
        context  = torch.bmm(attn_weights, feature_tokens)  # (B, L, d_feat)
        attended = self.value_proj(context)                  # (B, L, d_model)

        out = self.norm(seq_positions + self.dropout(attended))
        return out, attn_weights

class BetaVAEWithFeaturesAttention(nn.Module):
    """
    β-VAE with cross-attention fusion of sequence and genomic features.

    Architecture changes vs. previous version:
    - CSE encoder no longer pools to (B, d) — retains spatial dim (B, d, L_encoded)
    - CrossModalAttention fuses sequence positions with TE/NonB tokens
    - Attention weights (B, L_encoded, 2) returned in output dict
    - Decoder unchanged: takes z (B, latent_dim) as before
    - Classifier takes pooled attended representation + original feature embeddings
    """
    
    def __init__(self, num_classes, input_dim, input_length, latent_dim=128, 
                 beta=4.0, dropout_rate=0.3, use_cse=True, cse_d_model=512, 
                 cse_kernel_size=9, te_features_dim=170, nonb_features_dim=178,
                 te_processor_hidden=[64, 32], nonb_processor_hidden=[64, 32],
                 classifier_hidden=[128, 64], attn_heads=4, attn_dropout=0.1,
                 **kwargs):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.input_length = input_length
        self.latent_dim = latent_dim
        self.beta = beta
        self.use_cse = use_cse
        self.cse_d_model = cse_d_model

        # =====================================================================
        # CSE Pre-encoder — same as before
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
        # Convolutional Encoder — unchanged, but we intercept BEFORE flatten
        # to keep spatial dimension for attention
        # =====================================================================
        self.encoder_convs = nn.Sequential(
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
            # NOTE: no Flatten here — kept for attention
        )

        # Calculate spatial length after convolutions
        self.encoded_length = self._get_encoded_length(encoder_input_dim)
        self.conv_channels = 256
        self.flatten_size = self.conv_channels * self.encoded_length

        # VAE bottleneck — takes pooled+attended sequence
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # =====================================================================
        # Feature Processors — unchanged MLPs
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

        # d_feat: TE and NonB embeddings must match for the feature token
        # We project both to the same dim (conv_channels=256) for attention
        te_out_dim = te_processor_hidden[-1]
        nonb_out_dim = nonb_processor_hidden[-1]

        # If TE and NonB outputs differ in size, project both to max
        d_feat = max(te_out_dim, nonb_out_dim)
        #self.te_feat_proj  = nn.Linear(te_out_dim, d_feat)   if te_out_dim != d_feat   else nn.Identity()
        #self.nonb_feat_proj = nn.Linear(nonb_out_dim, d_feat) if nonb_out_dim != d_feat else nn.Identity()

        # =====================================================================
        # Cross-Modal Attention
        # Query  = sequence positions (B, L_encoded, conv_channels=256)
        # Key/V  = feature tokens     (B, 2, d_feat)
        # =====================================================================
        self.cross_attn = CrossModalAttention(
            d_model=self.conv_channels,   # 256
            d_feat=d_feat,
            num_heads=attn_heads,
            dropout=attn_dropout
        )

        # =====================================================================
        # Decoder — unchanged
        # =====================================================================
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)

        decoder_output_dim = cse_d_model if use_cse else input_dim

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
        # Classifier — takes latent z + original feature embeddings
        # (not the attended sequence — z already incorporates attention signal)
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

    # =========================================================================
    # Helpers
    # =========================================================================

    def _get_encoded_length(self, encoder_input_dim):
        """Calculate spatial length after encoder convolutions."""
        with torch.no_grad():
            x = torch.zeros(1, encoder_input_dim, self.input_length)
            x = self.encoder_convs(x)
            return x.shape[2]   # (1, 256, L_encoded) → L_encoded

    # =========================================================================
    # Forward components
    # =========================================================================

    def encode(self, x, te_features, nonb_features):
        """
        Encode sequence with cross-attention fusion.

        Returns:
            mu, logvar: VAE parameters  (B, latent_dim)
            attn_weights:               (B, L_encoded, 2)
        """
        # CSE pre-encoding
        if self.use_cse:
            x = self.cse_encoder(x)                    # (B, cse_d_model, L)

        # Convolutional encoding — keep spatial dim
        h = self.encoder_convs(x)                      # (B, 256, L_encoded)

        # Feature tokens
        te_embed   = self.te_processor(te_features)    # (B, te_out_dim)
        nonb_embed = self.nonb_processor(nonb_features) # (B, nonb_out_dim)

        te_token   = te_embed.unsqueeze(1)    # (B, 1, te_out_dim)
        nonb_token = nonb_embed.unsqueeze(1)  # (B, 1, nonb_out_dim)
        feature_tokens = torch.cat([te_token, nonb_token], dim=1) # (B, 2, d_feat)

        # Cross-attention: sequence positions attend to feature tokens
        h_seq = h.permute(0, 2, 1)                    # (B, L_encoded, 256)
        h_attended, attn_weights = self.cross_attn(h_seq, feature_tokens)
        # h_attended:   (B, L_encoded, 256)
        # attn_weights: (B, L_encoded, 2)   col0=TE, col1=NonB

        # Flatten for VAE bottleneck
        h_flat = h_attended.reshape(h_attended.size(0), -1)  # (B, flatten_size)

        mu     = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)

        self._te_embed   = te_embed
        self._nonb_embed = nonb_embed

        # Gate raw embeddings by their mean attention weight → puts attn on grad path
        attn_mean          = attn_weights.mean(dim=1)          # (B, 2)
        self._te_gated     = te_embed   * attn_mean[:, 0:1]   # (B, te_out_dim)
        self._nonb_gated   = nonb_embed * attn_mean[:, 1:2]   # (B, nonb_out_dim)

        return mu, logvar, attn_weights

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

    # =========================================================================
    # Main forward
    # =========================================================================

    def forward(self, x, te_features=None, nonb_features=None,
                biotypes=None, return_latent=False, deterministic=False):
        """
        Forward pass.

        Returns dict with:
            logits:        (B, num_classes)
            reconstruction:(B, input_dim, input_length)
            mu, logvar, z: VAE parameters and sample
            attn_weights:  (B, L_encoded, 2)  ← NEW: col0=TE, col1=NonB
            projection:    z  (for training compatibility)
        """
        if te_features is None or nonb_features is None:
            raise ValueError("te_features and nonb_features are required")

        # Encode with cross-attention
        mu, logvar, attn_weights = self.encode(x, te_features, nonb_features)

        # Sample
        z = mu if deterministic else self.reparameterize(mu, logvar)

        # Decode
        x_recon = self.decode(z)

        # Classify: z + raw feature embeddings (same as before)
        combined = torch.cat([z, self._te_gated, self._nonb_gated], dim=1)
        logits   = self.classifier(combined)

        return {
            'logits':         logits,
            'reconstruction': x_recon,
            'mu':             mu,
            'logvar':         logvar,
            'z':              z,
            'attn_weights':   attn_weights,   # (B, L_encoded, 2)
            'projection':     z
        }