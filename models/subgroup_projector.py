"""
models/subgroup_projector.py

Concat-based sub-group projection for BetaVAESubgroup.

Each sub-group receives feature slices and projects them to d_proj dimensions
via an independent MLP. Supports two input modes controlled by block source:

  source="both"      — cat([x_genomic_sg, x_processed_sg]) → MLP in_features = 2 * sg_dim
  source="processed" — x_processed_sg only                 → MLP in_features = 1 * sg_dim
  source="genomic"   — x_genomic_sg only                   → MLP in_features = 1 * sg_dim

The source is read from BLOCK_DEFS in the registry at init time.
SubgroupProjectionLayer.forward() accepts both x_genomic and x_processed
for all modes; for single-source blocks the unused tensor is simply ignored.

Architecture (per sub-group, source="both")
--------------------------------------------
  x_concat = cat([x_genomic_sg, x_processed_sg], dim=-1)   # (B, 2 * sg_dim)
  r        = MLP(x_concat)                                   # (B, d_proj)

Architecture (per sub-group, source="processed" or "genomic")
--------------------------------------------------------------
  r = MLP(x_sg)   # (B, d_proj)    in_features = 1 * sg_dim

MLP: in_dim → max(in_dim, d_proj) → d_proj
     LayerNorm + GELU + Dropout between layers.

1/√k token norm scaling is applied per sub-group after projection,
where k = number of input features in that sub-group. This prevents
high-feature-count sub-groups (e.g. TE_CORE: 69) from dominating
attention purely through token norm magnitude.

"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from data.feature_registry import FeatureRegistry, REGISTRY, BLOCK_DEFS


# ---------------------------------------------------------------------------
# SubgroupProjector  (one per sub-group)
# ---------------------------------------------------------------------------

class SubgroupProjector(nn.Module):
    """
    Projects feature slices for a single sub-group to d_proj dimensions.

    Parameters
    ----------
    in_features : number of features in this sub-group (per source)
    d_proj      : output dimensionality
    dropout     : dropout rate
    dual_input  : if True, concatenates genomic + processed (in_features * 2)
                  if False, uses a single input (in_features)
    """

    def __init__(
        self,
        in_features: int,
        d_proj:      int,
        dropout:     float = 0.1,
        dual_input:  bool  = True,
    ) -> None:
        super().__init__()
        in_dim   = 2 * in_features if dual_input else in_features
        mid      = max(in_dim, d_proj)
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid),
            nn.LayerNorm(mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, d_proj),
        )
        self.dual_input = dual_input

    def forward(
        self,
        x_primary:   torch.Tensor,             # (B, in_features)  always provided
        x_secondary: Optional[torch.Tensor],   # (B, in_features)  or None
    ) -> torch.Tensor:
        """
        Returns (B, d_proj).
        x_secondary is used only when dual_input=True.
        """
        if self.dual_input and x_secondary is not None:
            x = torch.cat([x_primary, x_secondary], dim=-1)
        else:
            x = x_primary
        return self.net(x)


# ---------------------------------------------------------------------------
# SubgroupProjectionLayer  (one per block)
# ---------------------------------------------------------------------------

class SubgroupProjectionLayer(nn.Module):
    """
    Full sub-group projection for one feature block (nonb / te / nonb2 / ...).

    Source handling
    ---------------
    The block's source type is read from BLOCK_DEFS at init:
      "both"      → dual_input=True  for all sub-group projectors
      "processed" → dual_input=False; forward uses x_processed only
      "genomic"   → dual_input=False; forward uses x_genomic only

    Parameters
    ----------
    block    : block name as defined in BLOCK_DEFS (e.g. "nonb", "te", "nonb2")
    d_proj   : output projection dimension per sub-group token
    dropout  : dropout rate inside SubgroupProjector MLPs
    registry : FeatureRegistry instance

    Inputs (forward)
    ----------------
    x_genomic   : (B, block_dim)  genomic-version features  (or None for processed-only blocks)
    x_processed : (B, block_dim)  processed-version features

    For source="both" blocks, both inputs are required.
    For source="processed" blocks, x_processed is used; x_genomic is ignored.
    For source="genomic"   blocks, x_genomic  is used; x_processed is ignored.

    Output
    ------
    tokens : (B, n_subgroups, d_proj)
    """

    def __init__(
        self,
        block:    str,
        d_proj:   int             = 64,
        dropout:  float           = 0.1,
        registry: FeatureRegistry = REGISTRY,
    ) -> None:
        super().__init__()

        if block not in BLOCK_DEFS:
            raise ValueError(
                f"Unknown block '{block}'. "
                f"Known blocks: {list(BLOCK_DEFS.keys())}"
            )

        self.block    = block
        self.d_proj   = d_proj
        self._reg     = registry
        self.source   = BLOCK_DEFS[block].source   # "both" | "processed" | "genomic"

        subgroups  = registry.block_subgroups(block)
        block_dim  = registry.block_dim(block)

        self.subgroups   = subgroups
        self.n_subgroups = len(subgroups)
        self.block_dim   = block_dim

        # Build per-subgroup flat index lists into the full block vector
        # Uses the generic indices_for_subgroup() rather than block-specific methods
        self._sg_indices: Dict[str, List[int]] = {
            sg: registry.indices_for_subgroup(sg)
            for sg in subgroups
        }

        dual = (self.source == "both")

        # One independent projector per sub-group
        self.projectors = nn.ModuleDict({
            sg: SubgroupProjector(
                in_features = len(self._sg_indices[sg]),
                d_proj      = d_proj,
                dropout     = dropout,
                dual_input  = dual,
            )
            for sg in subgroups
        })

        # 1/√k token norm scaling per sub-group
        scales = torch.tensor(
            [1.0 / (len(self._sg_indices[sg]) ** 0.5) for sg in subgroups],
            dtype=torch.float32,
        )
        self.register_buffer("_sg_scales", scales)

    def forward(
        self,
        x_genomic:   Optional[torch.Tensor],   # (B, block_dim) or None
        x_processed: Optional[torch.Tensor],   # (B, block_dim) or None
    ) -> torch.Tensor:
        """
        Returns
        -------
        tokens : (B, n_subgroups, d_proj)
        """
        # Resolve primary / secondary inputs based on source type
        if self.source == "both":
            if x_genomic is None or x_processed is None:
                raise ValueError(
                    f"Block '{self.block}' has source='both' and requires "
                    f"both x_genomic and x_processed."
                )
            primary   = x_genomic
            secondary = x_processed
        elif self.source == "genomic":
            if x_genomic is None:
                raise ValueError(
                    f"Block '{self.block}' has source='genomic' and requires x_genomic."
                )
            primary   = x_genomic
            secondary = None
        else:  # "processed"
            if x_processed is None:
                # Fall back to x_genomic if only one was supplied
                if x_genomic is None:
                    raise ValueError(
                        f"Block '{self.block}' has source='processed' but neither "
                        f"x_genomic nor x_processed was provided."
                    )
                primary = x_genomic
            else:
                primary = x_processed
            secondary = None

        token_list = []
        for i, sg in enumerate(self.subgroups):
            idx   = self._sg_indices[sg]
            p_sg  = primary[:, idx]
            s_sg  = secondary[:, idx] if secondary is not None else None
            token = self.projectors[sg](p_sg, s_sg)
            token = token * self._sg_scales[i]
            token_list.append(token.unsqueeze(1))

        return torch.cat(token_list, dim=1)   # (B, n_subgroups, d_proj)

    def extra_repr(self) -> str:
        return (
            f"block={self.block}, source={self.source}, "
            f"n_subgroups={self.n_subgroups}, "
            f"d_proj={self.d_proj}, block_dim={self.block_dim}"
        )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data.feature_registry import REGISTRY

    print("=" * 65)
    print("SubgroupProjectionLayer self-test")
    print("=" * 65)

    B = 8

    test_cases = [
        ("nonb",  REGISTRY.nonb_dim,  len(REGISTRY.nonb_subgroups),  "both"),
        ("te",    REGISTRY.te_dim,    len(REGISTRY.te_subgroups),    "both"),
        ("nonb2", REGISTRY.nonb2_dim, len(REGISTRY.nonb2_subgroups), "processed"),
    ]

    for block, dim, expected_tokens, expected_source in test_cases:
        print(f"\n── {block.upper()} (dim={dim}, "
              f"subgroups={expected_tokens}, source={expected_source}) ──")

        layer = SubgroupProjectionLayer(block=block, d_proj=64, dropout=0.1)
        print(layer)

        assert layer.source == expected_source, (
            f"Expected source={expected_source}, got {layer.source}"
        )

        x_g = torch.randn(B, dim)
        x_p = torch.randn(B, dim)

        if expected_source == "both":
            tokens = layer(x_g, x_p)
        else:
            # Single-source: pass x_processed, x_genomic can be None
            tokens = layer(None, x_p)

        assert tokens.shape == (B, expected_tokens, 64), (
            f"Expected ({B}, {expected_tokens}, 64), got {tokens.shape}"
        )

        tokens.sum().backward()
        for name, param in layer.named_parameters():
            assert param.grad is not None, f"No gradient for: {name}"

        n_params = sum(p.numel() for p in layer.parameters())
        print(f"  tokens shape : {tokens.shape}  ")
        print(f"  source       : {layer.source}  ")
        print(f"  gradients    : all parameters have grad  ")
        print(f"  parameters   : {n_params:,}")

    print("\nAll tests passed.")