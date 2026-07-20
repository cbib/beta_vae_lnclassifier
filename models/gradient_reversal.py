#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/gradient_reversal.py

Gradient Reversal Layer (GRL) and adversarial length prediction head
for length disentanglement in BetaVAESubgroup.

The GRL is an identity function during the forward pass. During the
backward pass it negates and scales the gradient before it reaches
the encoder, creating an adversarial signal that penalises the encoder
for encoding length-predictive information in z.

Integration into BetaVAESubgroup
---------------------------------
Add to __init__:
    self.grl        = GradientReversalLayer()
    self.length_head = LengthPredictionHead(latent_dim, hidden_dim=64)

Add to forward / encode output:
    length_pred = self.length_head(self.grl(mu, lambda_adv))
    # return length_pred in output dict

The adversarial loss is computed in the trainer:
    L_adv = MSE(length_pred, true_lengths)
    # length_head minimises L_adv (standard gradient)
    # encoder minimises -L_adv (negated via GRL)
    # net effect: encoder loses incentive to encode length

Training dynamics
-----------------
lambda_adv controls reversal strength. Start small (0.01–0.1) and
increase if length R² does not drop. Too large will destabilise
classification. The GRL strength can also be annealed:
    lambda_adv = 2 / (1 + exp(-10 * progress)) - 1   (DANN schedule)
where progress = current_epoch / total_epochs ∈ [0, 1].
"""

import torch
import torch.nn as nn
from torch.autograd import Function


# ---------------------------------------------------------------------------
# Gradient reversal autograd function
# ---------------------------------------------------------------------------

class _GradientReversalFunction(Function):
    """
    Forward: identity.
    Backward: negate and scale gradient by lambda_adv.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_adv: float) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor(lambda_adv))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        lambda_adv, = ctx.saved_tensors
        # Return negated gradient for x, None for lambda_adv (not a tensor param)
        return -lambda_adv.item() * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer.

    Usage
    -----
    grl = GradientReversalLayer()
    z_reversed = grl(z, lambda_adv=0.1)

    Parameters
    ----------
    lambda_adv : float
        Reversal strength. Passed at call time so it can be annealed.
        Default 1.0 (full reversal).
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, lambda_adv: float = 1.0) -> torch.Tensor:
        return _GradientReversalFunction.apply(x, lambda_adv)


# ---------------------------------------------------------------------------
# Length prediction head
# ---------------------------------------------------------------------------

class LengthPredictionHead(nn.Module):
    """
    Small MLP predicting log-transcript-length from z.

    Predicts log(length) rather than raw length for numerical stability —
    lengths span 12 to 347,561 nt (GENCODE v49), so raw MSE would be
    dominated by the longest transcripts.

    The adversarial loss is MSE(log_pred, log_true).

    Parameters
    ----------
    latent_dim : int
        Dimensionality of z (128 in current architecture).
    hidden_dim : int
        Hidden layer size. Default 64.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (B, latent_dim)

        Returns
        -------
        log_length_pred : (B,)  predicted log-length
        """
        return self.net(z).squeeze(-1)


# ---------------------------------------------------------------------------
# DANN annealing schedule for lambda_adv
# ---------------------------------------------------------------------------

def dann_lambda(current_epoch: int, total_epochs: int,
                gamma: float = 10.0, max_lambda: float = 1.0) -> float:
    """
    DANN progressive annealing schedule for lambda_adv.

    Starts near 0 and saturates toward max_lambda over training.
    Prevents early destabilisation when the encoder has not yet learned
    useful representations.

    L(p) = 2 / (1 + exp(-gamma * p)) - 1
    where p = current_epoch / total_epochs ∈ [0, 1]

    Parameters
    ----------
    current_epoch : int
    total_epochs  : int
    gamma         : float   Controls ramp speed. Default 10.
    max_lambda    : float   Ceiling for lambda. Default 1.0.

    Returns
    -------
    lambda_adv : float ∈ [0, max_lambda]
    """
    import math
    p = current_epoch / max(total_epochs, 1)
    return max_lambda * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)