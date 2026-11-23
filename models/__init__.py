#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model factory
"""
from .beta_vae_subgroup import BetaVAESubgroup
from .feature_only_classifier import FeatureOnlyClassifier


def get_model(architecture, **kwargs):
    """
    Factory function to get model by name
    
    Args:
        architecture: str, one of ['cnn', 'vae']
        **kwargs: model-specific parameters
    
    Returns:
        model: nn.Module
    """
    models = {
        'beta_vae_subgroup': BetaVAESubgroup,
        'feature_only': FeatureOnlyClassifier,
    }
    
    if architecture not in models:
        raise ValueError(f"Unknown architecture: {architecture}. Available: {list(models.keys())}")
    
    return models[architecture](**kwargs)