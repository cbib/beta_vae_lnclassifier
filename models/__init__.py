#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model factory
"""
""" from .cnn import CNN
from .vae import VAE
from .transformer import TransformerClassifier, LightweightTransformer
from .cnn_transformer import CNNTransformerHybrid """
from .autoencoder import DeterministicAE, LightweightDeterministicAE
from .cnn_cse import CNN_CSE
from .beta_vae_contrastive_model import BetaVAE_Contrastive
from .beta_vae_features import BetaVAEWithFeatures


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
        """ 'cnn': CNN,
        'vae': VAE,
        'transformer': TransformerClassifier,
        'lightweight_transformer': LightweightTransformer,
        'cnn_transformer': CNNTransformerHybrid, """
        'autoencoder': DeterministicAE,
        'cnn_cse': CNN_CSE,
        'beta_vae_contrastive': BetaVAE_Contrastive,
        'beta_vae_features': BetaVAEWithFeatures
    }
    
    if architecture not in models:
        raise ValueError(f"Unknown architecture: {architecture}. Available: {list(models.keys())}")
    
    return models[architecture](**kwargs)

def get_contrastive_model(architecture, **kwargs):
    """
    Factory function to get contrastive learning models
    
    Args:
        architecture: Model architecture name
        **kwargs: Model-specific parameters
    
    Returns:
        Model instance with contrastive learning support
    """
    """
    if architecture == 'autoencoder':
        from models.autoencoder_contrastive import AutoencoderContrastive
        return AutoencoderContrastive(**kwargs)
    
    elif architecture == 'transformer':
        from models.transformer_contrastive import TransformerContrastive
        return TransformerContrastive(**kwargs)
    
    elif architecture == 'bert':
        from models.bert_contrastive import BERTContrastive
        return BERTContrastive(**kwargs) """
    
    if architecture == 'beta_vae':
        from models.beta_vae_contrastive_model import BetaVAE_Contrastive
        return BetaVAE_Contrastive(**kwargs)
    
    elif architecture == 'cnn':
        from models.cnn_contrastive import CNNContrastive
        return CNNContrastive(**kwargs)
    
    elif architecture == 'beta_vae_features':
        from models.beta_vae_features import BetaVAEWithFeatures
        return BetaVAEWithFeatures(**kwargs)
    
    elif architecture == 'beta_vae_features_attn':
        from models.beta_vae_features_attn import BetaVAEWithFeaturesAttention
        return BetaVAEWithFeaturesAttention(**kwargs)
    
    else:
        raise ValueError(f"Unknown architecture for contrastive learning: {architecture}")