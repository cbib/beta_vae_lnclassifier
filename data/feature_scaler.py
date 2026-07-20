"""
data/feature_scaler.py

Per-category scaler bank for NonB, TE, and NonB2 feature blocks.

Replaces the single global StandardScaler with category-aware scaling:
  - presence / count / diversity  → pass-through (no scaling)
  - rel_length (_pct features)    → MinMaxScaler to [0, 1]
  - abs_length / gap / density /
    quality / stability           → RobustScaler (median/IQR)
  - fraction / binned_prop        → MinMaxScaler to [0, 1]

One FeatureScalerBank is fit per block (NonB, TE, NonB2) on trainval data only,
then applied to all transcripts.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from data.feature_registry import FeatureRegistry, REGISTRY


def _make_scaler(strategy: str):
    if strategy == "none":
        return None
    elif strategy == "robust":
        return RobustScaler(quantile_range=(5.0, 95.0))
    elif strategy == "minmax":
        return MinMaxScaler(feature_range=(0.0, 1.0))
    else:
        raise ValueError(f"Unknown scaling strategy: '{strategy}'. "
                         f"Expected one of: none, robust, minmax")


class FeatureScalerBank:
    """Per-category scaler bank for a single feature block (nonb, te, or nonb2)."""

    def __init__(self, block: str, registry: FeatureRegistry = REGISTRY) -> None:
        if block not in ("nonb", "te", "nonb2"):
            raise ValueError(f"block must be 'nonb', 'te', or 'nonb2', got '{block}'")

        self.block    = block
        self._reg     = registry
        self._fitted  = False

        # Build category → index list and category → scaler maps
        if block == "nonb":
            self._cat_indices: Dict[str, List[int]] = registry.nonb_category_indices()
            self._strategies:  Dict[str, str]       = registry.nonb_scale_strategy()
            self._dim = registry.nonb_dim
        elif block == "nonb2":
            self._cat_indices = registry.nonb2_category_indices()
            self._strategies  = registry.nonb2_scale_strategy()
            self._dim = registry.nonb2_dim
        else:  # te
            self._cat_indices = registry.te_category_indices()
            self._strategies  = registry.te_scale_strategy()
            self._dim = registry.te_dim

        self._scalers: Dict[str, object] = {
            cat: _make_scaler(strat)
            for cat, strat in self._strategies.items()
        }

    def fit(self, X: np.ndarray) -> "FeatureScalerBank":
        X = self._coerce_2d(X)
        self._check_dim(X)

        for cat, scaler in self._scalers.items():
            if scaler is None:
                continue

            indices = self._cat_indices[cat]
            X_cat   = X[:, indices].astype(np.float64)

            col_var = X_cat.var(axis=0)
            n_zero_var = (col_var == 0).sum()
            if n_zero_var > 0:
                warnings.warn(
                    f"[FeatureScalerBank/{self.block}] Category '{cat}': "
                    f"{n_zero_var}/{len(indices)} features have zero variance "
                    f"in trainval — will be zero-filled after transform.",
                    UserWarning, stacklevel=2,
                )

            scaler.fit(X_cat)

        self._fitted = True
        print(f"  FeatureScalerBank [{self.block}] fitted on {X.shape[0]:,} samples.")
        self._print_summary()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FeatureScalerBank has not been fitted. Call .fit() first.")

        squeeze = X.ndim == 1
        X = self._coerce_2d(X).astype(np.float64)
        self._check_dim(X)

        X_out = X.copy()

        for cat, scaler in self._scalers.items():
            if scaler is None:
                continue

            indices = self._cat_indices[cat]
            X_cat_scaled = scaler.transform(X[:, indices])

            bad = ~np.isfinite(X_cat_scaled)
            if bad.any():
                X_cat_scaled[bad] = 0.0

            X_out[:, indices] = X_cat_scaled

        result = X_out.astype(np.float32)
        return result.squeeze(0) if squeeze else result

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  Saved FeatureScalerBank [{self.block}] → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureScalerBank":
        import sys
        current = sys.modules.get(__name__)
        if current is not None:
            sys.modules.setdefault('feature_scaler', current)
            sys.modules.setdefault('data.feature_scaler', current)
        with open(path, "rb") as f:
            bank = pickle.load(f)
        if not isinstance(bank, cls):
            raise TypeError(f"Expected FeatureScalerBank, got {type(bank)}")
        return bank

    def _print_summary(self) -> None:
        lines = [f"    Category scaling summary [{self.block}]:"]
        for cat, strat in self._strategies.items():
            n = len(self._cat_indices[cat])
            lines.append(f"      {cat:12s}: {strat:7s}  ({n} features)")
        print("\n".join(lines))

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return (f"FeatureScalerBank(block='{self.block}', "
                f"dim={self._dim}, status={status})")

    def _coerce_2d(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[np.newaxis, :]
        elif X.ndim != 2:
            raise ValueError(f"Expected 1-D or 2-D array, got shape {X.shape}")
        return X

    def _check_dim(self, X: np.ndarray) -> None:
        if X.shape[1] != self._dim:
            raise ValueError(
                f"FeatureScalerBank [{self.block}]: expected {self._dim} features, "
                f"got {X.shape[1]}."
            )