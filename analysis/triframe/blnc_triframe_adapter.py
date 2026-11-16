"""blnc_triframe_adapter.py — triframe ModelAdapter for BetaVAESubgroup.

Independent re-implementation of the projector-site hooks in patch_tokens.py
(deliberately shares no code with it, so agreement is a real check).
Tokens are the raw model.projectors[block] outputs, concatenated in
REGISTRY block order: (B, N_tokens, d_proj).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
from triframe import ModelAdapter

FEATURE_KEYS = ("te_genomic", "te_processed", "nonb_genomic", "nonb_processed")


def batch_from_indices(dataset, indices: Sequence[int], has_nonb2: bool) -> Dict[str, torch.Tensor]:
    """Stack dataset samples into one batch dict (CPU tensors)."""
    keys = ["sequence", *FEATURE_KEYS] + (["nonb2"] if has_nonb2 else [])
    samples = [dataset[int(i)] for i in indices]
    return {k: torch.stack([s[k] for s in samples]) for k in keys}


class BlncAdapter(ModelAdapter):
    """Patch site: model.projectors[block] outputs (reaches both the
    cross-attention and per-block summary paths)."""

    def __init__(self, model, registry, device, has_nonb2: bool):
        self.model, self.device, self.has_nonb2 = model, device, has_nonb2
        self.blocks: List[str] = list(registry.block_names)
        self.offset: Dict[str, int] = {}
        self.size: Dict[str, int] = {}
        run = 0
        for b in self.blocks:
            self.offset[b] = run
            self.size[b] = len(registry.block_subgroups(b))
            run += self.size[b]
        self.n_tokens = run

    def _forward(self, batch: Dict[str, torch.Tensor]):
        kw = {k: batch[k].to(self.device) for k in FEATURE_KEYS}
        if self.has_nonb2 and "nonb2" in batch:
            kw["nonb2"] = batch["nonb2"].to(self.device)
        with torch.no_grad():
            return self.model(batch["sequence"].to(self.device),
                              deterministic=True, **kw)

    def run_forward(self, batch: Any) -> Any:
        return self._forward(batch)

    def get_output_value(self, model_output: Any) -> Any:
        return model_output["logits"]

    def capture_tokens(self, batch: Any) -> torch.Tensor:
        got: Dict[str, torch.Tensor] = {}
        handles = [self.model.projectors[b].register_forward_hook(
            lambda m, i, o, _b=b: got.__setitem__(_b, o.detach()))
            for b in self.blocks]
        try:
            self._forward(batch)
        finally:
            for h in handles:
                h.remove()
        return torch.cat([got[b] for b in self.blocks], dim=1)

    def run_forward_patched(self, batch: Any, positions: Any, values: Any) -> Any:
        positions = list(positions)
        if values.shape[1] != len(positions):
            raise ValueError("values must have one column per position")
        per_block: Dict[str, List[tuple]] = {}
        for j, pos in enumerate(positions):
            for b in self.blocks:
                lo = self.offset[b]
                if lo <= pos < lo + self.size[b]:
                    per_block.setdefault(b, []).append((pos - lo, j))
                    break
            else:
                raise IndexError(f"position {pos} outside 0..{self.n_tokens - 1}")

        def make_hook(block, pairs):
            def hook(module, inp, out):
                if out.shape[1] != self.size[block]:
                    raise RuntimeError(f"{block}: projector output has "
                                       f"{out.shape[1]} rows, expected {self.size[block]}")
                patched = out.clone()
                for row, j in pairs:
                    patched[:, row, :] = values[:, j, :].to(out.dtype)
                return patched
            return hook

        handles = [self.model.projectors[b].register_forward_hook(make_hook(b, p))
                   for b, p in per_block.items()]
        try:
            return self._forward(batch)
        finally:
            for h in handles:
                h.remove()