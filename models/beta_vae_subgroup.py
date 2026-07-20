"""
models/beta_vae_subgroup.py

β-VAE with dynamic sub-group tokenisation and diversified multi-head
cross-modal attention.

Auxiliary heads
---------------
One feat_classifier per block, keyed by block name in self.feat_classifiers
(nn.ModuleDict). Output dict contains feat_logits_{block_name} for each.
The trainer should sum auxiliary losses over all blocks weighted by
lambda_feat_{block_name} from the config.

Role-mask (legacy 15-token partition — disabled for n_tokens ≠ 15)
--------------------------------------------------------------------
  Head 0 — NonB structural  : GQ, IR, DR, STR
  Head 1 — NonB global/rep  : APR, MR, TRI, Z, GLOBAL
  Head 2 — TE content       : TE_QUALITY, TE_CORE, TE_LCTR
  Head 3 — TE annotation    : TE_PSEUDO, TE_UNKNOWN, TE_GLOBAL
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.feature_registry import REGISTRY, FeatureRegistry, BLOCK_DEFS
from models.subgroup_projector import SubgroupProjectionLayer
from models.gradient_reversal import GradientReversalLayer, LengthPredictionHead
from models.token_self_attention import TokenSelfAttention


# ---------------------------------------------------------------------------
# Legacy role-mask (15 tokens, 4 heads) — used only when n_tokens == 15
# ---------------------------------------------------------------------------

_LEGACY_ROLE_MASK_INDICES: List[List[int]] = [
    [2, 3, 1, 5],       # Head 0 — NonB structural  (GQ, IR, DR, STR)
    [0, 4, 6, 7, 8],    # Head 1 — NonB global/rep  (APR, MR, TRI, Z, GLOBAL)
    [9, 10, 11],        # Head 2 — TE content       (TE_QUALITY, TE_CORE, TE_LCTR)
    [12, 13, 14],       # Head 3 — TE annotation    (TE_PSEUDO, TE_UNKNOWN, TE_GLOBAL)
]


def _build_role_mask(
    n_tokens:  int,
    num_heads: int,
    device:    torch.device,
    indices:   List[List[int]],
) -> torch.Tensor:
    assert num_heads == len(indices), (
        f"role_mask requires exactly {len(indices)} heads, got {num_heads}"
    )
    mask = torch.full((num_heads, n_tokens), float("-inf"), device=device)
    for h, idx in enumerate(indices):
        mask[h, idx] = 0.0
    return mask


# ---------------------------------------------------------------------------
# CrossModalAttentionMH  (unchanged from previous version)
# ---------------------------------------------------------------------------

class CrossModalAttentionMH(nn.Module):
    def __init__(
        self,
        d_model:     int,
        d_feat:      int,
        num_heads:   int   = 4,
        dropout:     float = 0.1,
        temperature: float = 4.0,
        attn_mode:   str   = "standard",
    ) -> None:
        super().__init__()

        if d_feat % num_heads != 0:
            raise ValueError(
                f"d_feat ({d_feat}) must be divisible by num_heads ({num_heads})."
            )
        if attn_mode not in ("standard", "role_mask", "orthogonal"):
            raise ValueError(f"Unknown attn_mode '{attn_mode}'")

        self.d_feat    = d_feat
        self.num_heads = num_heads
        self.d_head    = d_feat // num_heads
        self.attn_mode = attn_mode

        self.log_temp = nn.Parameter(
            torch.full((num_heads, 1, 1), math.log(temperature))
        )

        self.query_proj = nn.Linear(d_model, d_feat, bias=False)
        self.key_proj   = nn.Linear(d_feat,  d_feat, bias=False)
        nn.init.normal_(self.query_proj.weight, std=0.01)
        nn.init.normal_(self.key_proj.weight,   std=0.01)

        self.value_proj = nn.Linear(d_feat, d_feat, bias=False)
        self.out_proj   = nn.Linear(d_feat, d_model)

        self.attn_drop  = nn.Dropout(dropout)
        self.norm       = nn.LayerNorm(d_model)
        self.resid_drop = nn.Dropout(dropout)

        self._role_mask: Optional[torch.Tensor] = None

    def _get_role_mask(
        self, n_tokens: int, device: torch.device
    ) -> torch.Tensor:
        if self._role_mask is None or self._role_mask.device != device:
            self._role_mask = _build_role_mask(
                n_tokens, self.num_heads, device, _LEGACY_ROLE_MASK_INDICES
            )
        return self._role_mask

    def forward(
        self,
        seq_positions:  torch.Tensor,
        feature_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = seq_positions.shape
        _, N, _ = feature_tokens.shape
        H, d_h  = self.num_heads, self.d_head

        Q = self.query_proj(seq_positions).view(B, L, H, d_h).transpose(1, 2)
        K = self.key_proj(feature_tokens).view(B, N, H, d_h).transpose(1, 2)
        V = self.value_proj(feature_tokens).view(B, N, H, d_h).transpose(1, 2)

        scale  = torch.exp(self.log_temp) * (d_h ** 0.5)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        if self.attn_mode == "role_mask":
            role_mask = self._get_role_mask(N, scores.device)
            scores = scores + role_mask.unsqueeze(0).unsqueeze(2)

        attn_weights = self.attn_drop(torch.softmax(scores, dim=-1))

        context  = torch.matmul(attn_weights, V)
        context  = context.transpose(1, 2).contiguous().view(B, L, self.d_feat)
        attended = self.out_proj(context)
        out      = self.norm(seq_positions + self.resid_drop(attended))

        return out, attn_weights


# ---------------------------------------------------------------------------
# BetaVAESubgroup
# ---------------------------------------------------------------------------

class BetaVAESubgroup(nn.Module):
    """
    β-VAE with dynamic sub-group tokenisation and MH cross-modal attention.

    All feature blocks defined in REGISTRY.block_names are automatically
    included. The classifier input dim is latent_dim + d_proj * n_blocks.

    forward() inputs
    ----------------
    x                        : (B, input_dim, input_length)
    {block_name}_genomic     : (B, block_dim)  for source="both" blocks
    {block_name}_processed   : (B, block_dim)  for source="both" blocks
    {block_name}             : (B, block_dim)  for single-source blocks
                               (passed as x_processed to projector)

    For convenience, legacy keyword names te_genomic, te_processed,
    nonb_genomic, nonb_processed are still accepted and mapped to
    the "te" and "nonb" blocks.

    forward() output dict
    ---------------------
    logits                   (B, num_classes)
    feat_logits_{block}      (B, num_classes)  one per block
    reconstruction           (B, input_dim, input_length)
    mu, logvar, z            VAE parameters
    attn_weights             (B, num_heads, L_encoded, N_tokens)
    projection               z
    length_pred              (B,)  from GRL adversarial head
    """

    def __init__(
        self,
        num_classes:       int,
        input_dim:         int,
        input_length:      int,
        latent_dim:        int             = 128,
        beta:              float           = 4.0,
        dropout_rate:      float           = 0.3,
        use_cse:           bool            = True,
        cse_d_model:       int             = 512,
        cse_kernel_size:   int             = 9,
        d_proj:            int             = 64,
        fusion_dropout:    float           = 0.1,
        attn_heads:        int             = 4,
        attn_dropout:      float           = 0.1,
        attn_temperature:  float           = 4.0,
        attn_mode:         str             = "standard",
        use_token_sa:      bool            = False,
        token_sa_heads:    int             = 4,
        token_sa_layers:   int             = 1,
        classifier_hidden: List[int]       = None,
        registry:          FeatureRegistry = REGISTRY,
        # Accepted for backward compat with old configs; not used internally
        te_features_dim:   Optional[int]   = None,
        nonb_features_dim: Optional[int]   = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if classifier_hidden is None:
            classifier_hidden = [128]

        self.num_classes  = num_classes
        self.input_dim    = input_dim
        self.input_length = input_length
        self.latent_dim   = latent_dim
        self.beta         = beta
        self.use_cse      = use_cse
        self.attn_mode    = attn_mode
        self._registry    = registry
        self.d_proj       = d_proj

        # Validate role_mask compatibility
        n_tokens = registry.total_tokens
        if attn_mode == "role_mask" and n_tokens != 15:
            warnings.warn(
                f"attn_mode='role_mask' is defined for 15 tokens but registry "
                f"has {n_tokens} tokens. Falling back to 'standard'. "
                f"Update _LEGACY_ROLE_MASK_INDICES or provide a new partition "
                f"to re-enable role_mask.",
                UserWarning,
            )
            self.attn_mode = "standard"

        # ── CSE pre-encoder ───────────────────────────────────────────────────
        if use_cse:
            self.cse_encoder = nn.Sequential(
                nn.Conv1d(input_dim, cse_d_model,
                          kernel_size=cse_kernel_size, padding="same"),
                nn.ReLU(),
                nn.BatchNorm1d(cse_d_model),
            )
            encoder_input_dim = cse_d_model
        else:
            self.cse_encoder  = None
            encoder_input_dim = input_dim

        # ── CNN encoder ───────────────────────────────────────────────────────
        self.encoder_convs = nn.Sequential(
            nn.Conv1d(encoder_input_dim, 64,  kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout1d(dropout_rate),

            nn.Conv1d(64,  128, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout1d(dropout_rate),

            nn.Conv1d(128, 256, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout1d(dropout_rate),

            nn.Conv1d(256, 256, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(),
        )

        self.encoded_length = self._get_encoded_length(encoder_input_dim)
        self.conv_channels  = 256
        self.flatten_size   = self.conv_channels * self.encoded_length

        # ── VAE bottleneck ────────────────────────────────────────────────────
        self.fc_mu     = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

        # ── Dynamic sub-group projectors ──────────────────────────────────────
        # One SubgroupProjectionLayer per block, keyed by block name
        self.projectors = nn.ModuleDict({
            block_name: SubgroupProjectionLayer(
                block    = block_name,
                d_proj   = d_proj,
                dropout  = fusion_dropout,
                registry = registry,
            )
            for block_name in registry.block_names
        })

        self.n_tokens = registry.total_tokens   # dynamic (15 → 20 with nonb2)

        # Per-block token count for attention summary split
        self._block_token_counts: Dict[str, int] = {
            b: len(registry.block_subgroups(b))
            for b in registry.block_names
        }

        # Token LayerNorm
        self.token_norm = nn.LayerNorm(d_proj)

        # Optional token self-attention
        self.use_token_sa = use_token_sa
        if use_token_sa:
            self.token_sa = TokenSelfAttention(
                d_proj   = d_proj,
                n_heads  = token_sa_heads,
                n_layers = token_sa_layers,
                dropout  = fusion_dropout,
            )
        else:
            self.token_sa = None

        # ── Cross-modal attention ─────────────────────────────────────────────
        self.cross_attn = CrossModalAttentionMH(
            d_model     = self.conv_channels,
            d_feat      = d_proj,
            num_heads   = attn_heads,
            dropout     = attn_dropout,
            temperature = attn_temperature,
            attn_mode   = self.attn_mode,
        )

        # ── Decoder ───────────────────────────────────────────────────────────
        self.fc_decode     = nn.Linear(latent_dim, self.flatten_size)
        decoder_output_dim = cse_d_model if use_cse else input_dim

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, self.encoded_length)),
            nn.ConvTranspose1d(256, 256, kernel_size=8, stride=4,
                               padding=2, output_padding=0),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=8, stride=4,
                               padding=2, output_padding=0),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.ConvTranspose1d(128, 64,  kernel_size=8, stride=4,
                               padding=2, output_padding=0),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64, decoder_output_dim, kernel_size=8, stride=4,
                               padding=2, output_padding=0),
            nn.Sigmoid(),
        )

        if use_cse:
            self.cse_decoder = nn.Sequential(
                nn.Conv1d(cse_d_model, input_dim,
                          kernel_size=cse_kernel_size, padding="same"),
                nn.Sigmoid(),
            )
        else:
            self.cse_decoder = None

        # ── Classifiers ───────────────────────────────────────────────────────
        # Main: z + one d_proj summary per block
        n_blocks     = len(registry.block_names)
        combined_dim = latent_dim + d_proj * n_blocks

        def _make_head(in_dim: int) -> nn.Sequential:
            layers: List[nn.Module] = []
            prev = in_dim
            for h in classifier_hidden:
                layers += [
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
                prev = h
            layers.append(nn.Linear(prev, num_classes))
            return nn.Sequential(*layers)

        self.classifier = _make_head(combined_dim)

        # Per-block auxiliary classifiers
        self.feat_classifiers = nn.ModuleDict({
            block_name: _make_head(d_proj)
            for block_name in registry.block_names
        })

        # Backward compat shims for trainer code that references
        # self.feat_classifier_nonb / self.feat_classifier_te directly
        if "nonb" in registry.block_names:
            self.feat_classifier_nonb = self.feat_classifiers["nonb"]
        if "te" in registry.block_names:
            self.feat_classifier_te = self.feat_classifiers["te"]

        # ── Adversarial length disentanglement ────────────────────────────────
        self.grl         = GradientReversalLayer()
        self.length_head = LengthPredictionHead(
            latent_dim = latent_dim,
            hidden_dim = 64,
        )
        self.register_buffer(
            "lambda_adv",
            torch.tensor(0.0, dtype=torch.float32),
        )

        self.projection = nn.Identity()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _get_encoded_length(self, encoder_input_dim: int) -> int:
        with torch.no_grad():
            x = torch.zeros(1, encoder_input_dim, self.input_length)
            return self.encoder_convs(x).shape[2]

    # -------------------------------------------------------------------------
    # Encode
    # -------------------------------------------------------------------------

    def encode(
        self,
        x:           torch.Tensor,
        block_inputs: Dict[str, Tuple[Optional[torch.Tensor],
                                      Optional[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x            : (B, input_dim, input_length)
        block_inputs : {block_name: (x_genomic, x_processed)}
                       For single-source blocks, x_genomic may be None.

        Returns
        -------
        mu, logvar   : (B, latent_dim)
        attn_weights : (B, num_heads, L_encoded, N_tokens)
        """
        if self.use_cse:
            x = self.cse_encoder(x)

        h = self.encoder_convs(x)   # (B, 256, L_encoded)

        # Build feature tokens from all blocks in registry order
        all_tokens: List[torch.Tensor] = []
        self._block_tokens: Dict[str, torch.Tensor] = {}

        for block_name in self._registry.block_names:
            x_g, x_p = block_inputs[block_name]
            tokens    = self.projectors[block_name](x_g, x_p)
            # tokens: (B, n_sg_in_block, d_proj)
            self._block_tokens[block_name] = tokens
            all_tokens.append(tokens)

        feature_tokens = torch.cat(all_tokens, dim=1)   # (B, N_total, d_proj)
        feature_tokens = self.token_norm(feature_tokens)

        if self.use_token_sa and self.token_sa is not None:
            feature_tokens = self.token_sa(feature_tokens)

        # Cross-modal attention
        h_seq = h.permute(0, 2, 1)
        h_attended, attn_weights = self.cross_attn(h_seq, feature_tokens)

        # VAE bottleneck
        h_flat = h_attended.reshape(h_attended.size(0), -1)
        mu     = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)

        # Per-block attention-weighted summaries
        attn_mean    = attn_weights.mean(dim=(1, 2))   # (B, N_total)
        self._block_summaries: Dict[str, torch.Tensor] = {}
        offset = 0
        for block_name in self._registry.block_names:
            n_sg   = self._block_token_counts[block_name]
            b_attn = attn_mean[:, offset:offset + n_sg].unsqueeze(-1)  # (B, n_sg, 1)
            b_tok  = self._block_tokens[block_name]                    # (B, n_sg, d_proj)
            self._block_summaries[block_name] = (b_tok * b_attn).sum(dim=1)  # (B, d_proj)
            offset += n_sg

        return mu, logvar, attn_weights

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h       = self.fc_decode(z)
        x_recon = self.decoder(h)
        if self.use_cse:
            x_recon = self.cse_decoder(x_recon)
        L = x_recon.shape[2]
        if L < self.input_length:
            x_recon = F.pad(x_recon, (0, self.input_length - L))
        elif L > self.input_length:
            x_recon = x_recon[:, :, :self.input_length]
        return x_recon

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        x:              torch.Tensor,
        # Named block inputs — accepted via **kwargs and resolved below
        te_genomic:     Optional[torch.Tensor] = None,
        te_processed:   Optional[torch.Tensor] = None,
        nonb_genomic:   Optional[torch.Tensor] = None,
        nonb_processed: Optional[torch.Tensor] = None,
        nonb2:          Optional[torch.Tensor] = None,
        # Legacy single-version fallbacks
        te_features:    Optional[torch.Tensor] = None,
        nonb_features:  Optional[torch.Tensor] = None,
        deterministic:  bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        # ── Resolve legacy fallbacks ──────────────────────────────────────────
        if te_genomic is None:
            te_genomic = te_processed = te_features
        if te_processed is None:
            te_processed = te_genomic
        if nonb_genomic is None:
            nonb_genomic = nonb_processed = nonb_features
        if nonb_processed is None:
            nonb_processed = nonb_genomic

        # ── Build block_inputs dict for all registry blocks ───────────────────
        block_inputs: Dict[str, Tuple[Optional[torch.Tensor],
                                      Optional[torch.Tensor]]] = {}

        for block_name in self._registry.block_names:
            source = BLOCK_DEFS[block_name].source
            if block_name == "nonb":
                block_inputs[block_name] = (nonb_genomic, nonb_processed)
            elif block_name == "te":
                block_inputs[block_name] = (te_genomic, te_processed)
            elif block_name == "nonb2":
                # single-source: pass as x_processed, x_genomic=None
                block_inputs[block_name] = (None, nonb2)
            else:
                # Future blocks — look up from kwargs by block_name
                g = kwargs.get(f"{block_name}_genomic")
                p = kwargs.get(f"{block_name}_processed",
                               kwargs.get(block_name))
                block_inputs[block_name] = (g, p)

        mu, logvar, attn_weights = self.encode(x, block_inputs)

        z       = mu if deterministic else self.reparameterize(mu, logvar)
        x_recon = self.decode(z)

        # Adversarial length
        z_reversed  = self.grl(mu, lambda_adv=float(self.lambda_adv))
        length_pred = self.length_head(z_reversed)

        # Main classifier
        summaries    = list(self._block_summaries.values())
        combined     = torch.cat([z] + summaries, dim=1)
        logits       = self.classifier(combined)

        # Auxiliary classifiers
        out: Dict[str, torch.Tensor] = {
            "logits":         logits,
            "reconstruction": x_recon,
            "mu":             mu,
            "logvar":         logvar,
            "z":              z,
            "attn_weights":   attn_weights,
            "projection":     z,
            "length_pred":    length_pred,
        }

        for block_name, summary in self._block_summaries.items():
            key = f"feat_logits_{block_name}"
            out[key] = self.feat_classifiers[block_name](summary)

        # Backward compat aliases
        if "feat_logits_nonb" not in out and "nonb" in self._block_summaries:
            out["feat_logits_nonb"] = out["feat_logits_nonb"]
        if "feat_logits_te" not in out and "te" in self._block_summaries:
            out["feat_logits_te"] = out["feat_logits_te"]

        return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data.feature_registry import REGISTRY

    B, INPUT_DIM, INPUT_LENGTH = 2, 5, 1000
    D_PROJ, ATTN_HEADS         = 64, 4

    print(f"Registry blocks : {REGISTRY.block_names}")
    print(f"Total tokens    : {REGISTRY.total_tokens}")

    model = BetaVAESubgroup(
        num_classes      = 2,
        input_dim        = INPUT_DIM,
        input_length     = INPUT_LENGTH,
        latent_dim       = 128,
        beta             = 4.0,
        dropout_rate     = 0.3,
        use_cse          = True,
        cse_d_model      = 512,
        cse_kernel_size  = 9,
        d_proj           = D_PROJ,
        fusion_dropout   = 0.1,
        attn_heads       = ATTN_HEADS,
        attn_dropout     = 0.1,
        attn_mode        = "standard",
        classifier_hidden= [128],
    )

    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters  : {total:,}")
    print(f"L_encoded   : {model.encoded_length}")
    print(f"N tokens    : {model.n_tokens}")

    x      = torch.randn(B, INPUT_DIM, INPUT_LENGTH)
    te_g   = torch.randn(B, REGISTRY.te_dim)
    te_p   = torch.randn(B, REGISTRY.te_dim)
    nb_g   = torch.randn(B, REGISTRY.nonb_dim)
    nb_p   = torch.randn(B, REGISTRY.nonb_dim)
    nb2    = torch.randn(B, REGISTRY.nonb2_dim)

    out = model(x,
                te_genomic=te_g, te_processed=te_p,
                nonb_genomic=nb_g, nonb_processed=nb_p,
                nonb2=nb2)

    assert out["logits"].shape == (B, 2)
    assert out["mu"].shape     == (B, 128)
    assert out["attn_weights"].shape == (
        B, ATTN_HEADS, model.encoded_length, model.n_tokens
    )
    assert "feat_logits_nonb"  in out
    assert "feat_logits_te"    in out
    assert "feat_logits_nonb2" in out

    print(f"\nlogits          : {out['logits'].shape}  ")
    print(f"attn_weights    : {out['attn_weights'].shape}  ")
    print(f"feat_logits_nonb2: {out['feat_logits_nonb2'].shape}  ")

    loss = out["logits"].sum() + out["mu"].sum()
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert not no_grad, f"No gradient for: {no_grad}"
    print("gradients       : all parameters have grad  ")

    print("\nAll self-tests passed.")