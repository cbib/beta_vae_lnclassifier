from models import get_contrastive_model

def create_model_builder(config):
    """Create model builder"""
    architecture = config.get('model', 'architecture', default='cnn')
    
    def build_model():
        model_params = {
            'num_classes': config.get('model', 'num_classes'),
            'input_dim': 5,
            'input_length': config.get('model', 'max_length')
        }
        
        if architecture == 'beta_vae':
            model_params.update({
                'latent_dim': config.get('model', 'latent_dim', default=128),
                'beta': config.get('model', 'beta', default=4.0),
                'dropout_rate': config.get('model', 'dropout_rate', default=0.3),
                'use_cse': config.get('model', 'use_cse', default=False),
                'cse_d_model': config.get('model', 'cse_d_model', default=128),
                'cse_kernel_size': config.get('model', 'cse_kernel_size', default=9)
            })
        elif architecture == 'beta_vae_features':
            model_params.update({
                'latent_dim': config.get('model', 'latent_dim', default=128),
                'beta': config.get('model', 'beta', default=4.0),
                'dropout_rate': config.get('model', 'dropout_rate', default=0.3),
                'use_cse': config.get('model', 'use_cse', default=True),
                'cse_d_model': config.get('model', 'cse_d_model', default=512),
                'cse_kernel_size': config.get('model', 'cse_kernel_size', default=9),
                # Feature-specific parameters
                'te_features_dim': config.get('model', 'te_features_dim'),
                'nonb_features_dim': config.get('model', 'nonb_features_dim'),
                'te_processor_hidden': config.get('model', 'te_processor_hidden', default=[128, 64, 32]),
                'nonb_processor_hidden': config.get('model', 'nonb_processor_hidden', default=[128, 64, 32]),
                'classifier_hidden': config.get('model', 'classifier_hidden', default=[256, 128])
            })
        elif architecture == 'beta_vae_features_attn':
            model_params.update({
                'latent_dim': config.get('model', 'latent_dim', default=128),
                'beta': config.get('model', 'beta', default=4.0),
                'dropout_rate': config.get('model', 'dropout_rate', default=0.3),
                'use_cse': config.get('model', 'use_cse', default=True),
                'cse_d_model': config.get('model', 'cse_d_model', default=512),
                'cse_kernel_size': config.get('model', 'cse_kernel_size', default=9),
                # Feature-specific parameters
                'te_features_dim': config.get('model', 'te_features_dim'),
                'nonb_features_dim': config.get('model', 'nonb_features_dim'),
                'te_processor_hidden': config.get('model', 'te_processor_hidden', default=[128, 64, 32]),
                'nonb_processor_hidden': config.get('model', 'nonb_processor_hidden', default=[128, 64, 32]),
                'classifier_hidden': config.get('model', 'classifier_hidden', default=[256, 128]),
                'attn_heads': config.get('model', 'attn_heads', default=4),
                'attn_dropout': config.get('model', 'attn_dropout', default=0.1)
            })
        else:
            model_params['embedding_dim'] = config.get('model', 'embedding_dim', default=256)
            model_params['projection_dim'] = config.get('model', 'projection_dim', default=128)
            model_params['dropout_rate'] = config.get('model', 'dropout_rate', default=0.3)
            model_params['use_cse'] = config.get('model', 'use_cse', default=False)
        
        return get_contrastive_model(architecture, **model_params)
    
    return build_model
