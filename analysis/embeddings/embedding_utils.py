"""
Shared embedding extraction utilities.

Provides consistent embedding extraction across:
- Single fold evaluation
- Multi-fold analysis
- Visualization scripts
- Spatial analysis
"""
import torch
import numpy as np
from tqdm import tqdm


def extract_embeddings_from_model(model, data_loader, device, use_deterministic=True, biotype_mapping=None):
    """
    Extract embeddings from a trained model.
    
    For β-VAE models, can extract either:
    - Deterministic embeddings (μ) - recommended for visualization
    - Stochastic embeddings (sampled z)
    
    Args:
        model: Trained model with encode() method
        data_loader: DataLoader for the data
        device: torch device
        use_deterministic: If True, use μ; if False, sample z
        biotype_mapping: Optional mapping from biotype indices to strings
    
    Returns:
        Dictionary containing:
            - embeddings: (N, latent_dim) array
            - labels: (N,) array of class labels (0=lnc, 1=pc)
            - predictions: (N,) array of predicted classes
            - probs: (N, num_classes) array of class probabilities
            - confidences: (N,) array of max probabilities
            - biotype_indices: (N,) array of biotype indices
            - transcript_ids: (N,) array of transcript IDs
    """
    model.eval()
    
    all_embeddings = []
    all_labels = []
    all_predictions = []
    all_probs = []
    all_biotype_indices = []
    all_transcript_ids = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Extracting embeddings'):
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            
            # Extract embeddings
            if hasattr(model, 'encode') and hasattr(model, 'reparameterize'):
                mu, logvar = model.encode(sequences)
                
                if use_deterministic:
                    # Use deterministic mean
                    embeddings = mu
                else:
                    # Sample from latent distribution
                    embeddings = model.reparameterize(mu, logvar)
                
                # Get predictions
                logits = model.classifier(embeddings if not use_deterministic else mu)
            else:
                logits, embeddings, projections = model(sequences, return_embeddings=True)  # Ensure forward pass works
                
            # Compute predictions and probabilities
            probs = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)
            
            # Store results
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_biotype_indices.append(batch['biotype_label'].numpy())
            all_transcript_ids.extend(batch['transcript_id'])
    
    # Concatenate all batches
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    predictions = np.concatenate(all_predictions)
    probs = np.vstack(all_probs)
    biotype_indices = np.concatenate(all_biotype_indices)
    transcript_ids = np.array(all_transcript_ids, dtype=object)
    
    # Compute confidences
    confidences = probs.max(axis=1)
    
    # Compute correctness
    is_correct = (predictions == labels)

    result = {
        'embeddings': embeddings,
        'labels': labels,
        'predictions': predictions,
        'probs': probs,
        'confidences': confidences,
        'transcript_ids': transcript_ids,
        'is_correct': is_correct,
        'extraction_mode': 'deterministic' if use_deterministic else 'stochastic'
    }

    # Convert biotype indices to strings if mapping provided
    if biotype_mapping is not None:
        biotypes = np.array([biotype_mapping.get(int(idx), f'unknown_{idx}') 
                            for idx in biotype_indices])
        result['biotypes'] = biotypes
    else:
        # Fallback: save indices
        result['biotype_indices'] = biotype_indices
    
    return result


def extract_embeddings_all_folds(model_builder, experiment_dir, sequences, labels, 
                                 biotypes, biotype_to_idx, preprocessor, config, 
                                 device, use_deterministic=True):
    """
    Extract embeddings from all trained folds.
    
    Args:
        model_builder: Function that creates a new model instance
        experiment_dir: Path to experiment directory containing fold checkpoints
        sequences: All sequences
        labels: All labels
        biotypes: All biotype labels
        biotype_to_idx: Biotype to index mapping
        preprocessor: SequencePreprocessor instance
        config: Configuration object
        device: torch device
        use_deterministic: If True, use μ; if False, sample z
    
    Returns:
        Dictionary with keys:
            - fold_0, fold_1, ...: Embedding dictionaries for each fold
            - metadata: Information about the extraction
    """
    from pathlib import Path
    from torch.utils.data import DataLoader
    from sklearn.model_selection import StratifiedKFold
    from data.preprocessing import RNASequenceBiotypeDataset
    from data.cv_utils import create_length_stratified_groups, get_cv_splitter
    
    experiment_dir = Path(experiment_dir)

    idx_to_biotype = {idx: bt for bt, idx in biotype_to_idx.items()}

    results = {'metadata': {
        'experiment_dir': str(experiment_dir),
        'extraction_mode': 'deterministic' if use_deterministic else 'stochastic',
        'n_folds': config.get('training', 'n_folds', default=5)
    }}
    
    # Create CV splits (must match training)
    strat_groups = create_length_stratified_groups(sequences, labels)
    skf = get_cv_splitter(
        n_folds=config.get('training', 'n_folds', default=5),
        random_state=config.get('training', 'random_state', default=42)
    )
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(sequences, strat_groups)):
        print(f"\n{'='*80}")
        print(f"Extracting embeddings from Fold {fold_idx}")
        print(f"{'='*80}")
        
        # Load checkpoint
        checkpoint_path = experiment_dir / 'models' / f'fold_{fold_idx}_best.pt'
        if not checkpoint_path.exists():
            print(f"  Checkpoint not found: {checkpoint_path}")
            continue
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create and load model
        model = model_builder()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"  Val acc at save: {checkpoint.get('val_acc', 'unknown')}")
        
        # Create validation dataset
        val_seqs = [sequences[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        val_biotypes = [biotypes[i] for i in val_idx]
        
        val_dataset = RNASequenceBiotypeDataset(
            val_seqs, val_labels, preprocessor,
            biotype_labels=val_biotypes,
            biotype_to_idx=biotype_to_idx
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=1
        )
        
        # Extract embeddings
        fold_embeddings = extract_embeddings_from_model(
            model, val_loader, device, use_deterministic, biotype_mapping=idx_to_biotype
        )
        
        # Add fold-specific information
        fold_embeddings['fold_idx'] = fold_idx
        fold_embeddings['val_indices'] = val_idx
        fold_embeddings['n_samples'] = len(val_idx)
        fold_embeddings['checkpoint_epoch'] = checkpoint.get('epoch', None)
        fold_embeddings['checkpoint_val_acc'] = checkpoint.get('val_acc', None)
        
        results[f'fold_{fold_idx}'] = fold_embeddings
        
        print(f"  Extracted {len(fold_embeddings['embeddings'])} embeddings")
        print(f"  Embedding shape: {fold_embeddings['embeddings'].shape}")
        print(f"  Accuracy: {fold_embeddings['is_correct'].mean():.4f}")

        if 'biotypes' in fold_embeddings:
            print(f"  Biotypes saved as strings: {fold_embeddings['biotypes'][:3]}")
        else:
            print(f"  Biotypes saved as indices")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    return results


def save_embeddings(embeddings_dict, output_path, compress=True):
    """
    Save embeddings to disk using flattened key structure.
    
    Args:
        embeddings_dict: Dictionary from extract_embeddings_from_model() or
                        extract_embeddings_all_folds()
        output_path: Path to save (will use .npz format)
        compress: If True, use compressed format
    """
    from pathlib import Path
    
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Convert all numpy arrays and handle nested dicts with flattened keys
    def prepare_for_save(d, prefix=''):
        result = {}
        for key, value in d.items():
            # Create flattened key
            if prefix:
                full_key = f"{prefix}_{key}"
            else:
                full_key = key
            
            if isinstance(value, dict):
                # Recursively handle nested dicts
                result.update(prepare_for_save(value, prefix=full_key))
            elif isinstance(value, np.ndarray):
                result[full_key] = value
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                if isinstance(value[0], (int, float, str, bool, np.integer, np.floating)):
                    result[full_key] = np.array(value)
            elif isinstance(value, (int, float, str, bool)):
                # Store scalars as 0-d arrays
                result[full_key] = np.array(value)
            elif value is not None:
                # Try to convert to numpy array
                try:
                    result[full_key] = np.array(value)
                except:
                    print(f"  Warning: Could not save key '{full_key}' with type {type(value)}")
        
        return result
    
    save_dict = prepare_for_save(embeddings_dict)
    
    if compress:
        np.savez_compressed(output_path, **save_dict)
    else:
        np.savez(output_path, **save_dict)
    
    print(f"  Saved embeddings to: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Keys saved: {len(save_dict)}")


def load_embeddings(embeddings_path):
    """
    Load embeddings from disk.
    
    Args:
        embeddings_path: Path to .npz file
    
    Returns:
        Dictionary with embeddings and metadata
    """
    from pathlib import Path
    
    embeddings_path = Path(embeddings_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    
    data = np.load(embeddings_path, allow_pickle=True)
    
    # Reconstruct nested dictionary structure
    result = {}
    
    for key in data.files:
        # Handle metadata keys (e.g., 'metadata_experiment_dir', 'metadata_extraction_mode')
        if key.startswith('metadata_'):
            metadata_key = key.replace('metadata_', '')
            if 'metadata' not in result:
                result['metadata'] = {}
            result['metadata'][metadata_key] = data[key].item() if data[key].shape == () else data[key]
        
        # Handle fold keys (e.g., 'fold_0_embeddings', 'fold_1_labels')
        elif key.startswith('fold_') and '_' in key[6:]:  # fold_X_something
            parts = key.split('_', 2)  # Split into ['fold', 'X', 'rest']
            fold_key = f"{parts[0]}_{parts[1]}"  # 'fold_X'
            sub_key = parts[2] if len(parts) > 2 else ''
            
            if fold_key not in result:
                result[fold_key] = {}
            
            if sub_key:
                result[fold_key][sub_key] = data[key]
            else:
                result[fold_key] = data[key]
        
        # Handle direct fold keys (e.g., just 'fold_0' as a dict)
        elif key.startswith('fold_') and key.replace('fold_', '').isdigit():
            # This is a fold dict stored as a single object
            result[key] = data[key].item() if hasattr(data[key], 'item') else data[key]
        
        # Handle single-fold embeddings (no fold prefix)
        else:
            result[key] = data[key]
    
    print(f"  Loaded embeddings from: {embeddings_path}")
    
    # Print summary
    fold_keys = [k for k in result.keys() if k.startswith('fold_')]
    if fold_keys:
        n_folds = len(fold_keys)
        print(f"  Multi-fold embeddings: {n_folds} folds")
        for fold_key in sorted(fold_keys):
            if isinstance(result[fold_key], dict):
                n_samples = len(result[fold_key].get('embeddings', []))
                print(f"    {fold_key}: {n_samples} samples")
            else:
                print(f"    {fold_key}: (stored as object)")
    else:
        n_samples = len(result.get('embeddings', []))
        print(f"  Single-fold embeddings: {n_samples} samples")
    
    # Print metadata if available
    if 'metadata' in result:
        print(f"  Metadata keys: {list(result['metadata'].keys())}")
    
    return result