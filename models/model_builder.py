from models import get_model
from data.feature_registry import REGISTRY


def create_model_builder(config):
    """Create model builder."""
    architecture = config.get('model', 'architecture', default='cnn')

    def build_model():
        model_params = {
            'num_classes':  config.get('model', 'num_classes'),
            'input_dim':    5,
            'input_length': config.get('model', 'max_length'),
        }

        if architecture == 'beta_vae_subgroup':
            # Current architecture — dims derived from registry, not config
            model_params.update({
                'attn_mode':        config.get('model', 'attn_mode',        default='standard'),
                'latent_dim':       config.get('model', 'latent_dim',       default=128),
                'beta':             config.get('model', 'beta',             default=4.0),
                'dropout_rate':     config.get('model', 'dropout_rate',     default=0.3),
                'use_cse':          config.get('model', 'use_cse',          default=True),
                'cse_d_model':      config.get('model', 'cse_d_model',      default=512),
                'cse_kernel_size':  config.get('model', 'cse_kernel_size',  default=9),
                'd_proj':           config.get('model', 'd_proj',           default=64),
                'fusion_dropout':   config.get('model', 'fusion_dropout',   default=0.1),
                'classifier_hidden':config.get('model', 'classifier_hidden',default=[128]),
                'attn_heads':       config.get('model', 'attn_heads',       default=4),
                'attn_dropout':     config.get('model', 'attn_dropout',     default=0.1),
                'attn_temperature': config.get('model', 'attn_temperature', default=4.0),
                'block_sources':    config.get('model', 'block_sources',    default=None),
                # Registry drives feature dims — old config keys silently ignored
                'registry':         REGISTRY,
            })

        elif architecture == 'feature_only':
            # Current architecture — dims derived from registry, not config
            model_params.update({
                'd_proj':           config.get('model', 'd_proj',           default=64),
                'fusion_dropout':   config.get('model', 'fusion_dropout',   default=0.1),
                'attn_heads':       config.get('model', 'attn_heads',       default=4),
                'attn_dropout':     config.get('model', 'attn_dropout',     default=0.1),
                'attn_temperature': config.get('model', 'attn_temperature', default=4.0),
                'attn_mode':        config.get('model', 'attn_mode',        default='standard'),
                'classifier_hidden':config.get('model', 'classifier_hidden',default=[128]),
                'dropout_rate':     config.get('model', 'dropout_rate',     default=0.3),
                # Registry drives feature dims — old config keys silently ignored
                'registry':         REGISTRY,
            })

        else:
            model_params['embedding_dim']  = config.get('model', 'embedding_dim',  default=256)
            model_params['projection_dim'] = config.get('model', 'projection_dim', default=128)
            model_params['dropout_rate']   = config.get('model', 'dropout_rate',   default=0.3)
            model_params['use_cse']        = config.get('model', 'use_cse',        default=False)

        return get_model(architecture, **model_params)

    return build_model