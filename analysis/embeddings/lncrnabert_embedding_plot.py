#!/usr/bin/env python3
"""
Visualize lncRNA-BERT Embeddings using t-SNE

Creates t-SNE visualization of BERT embeddings for GENCODE data.

Usage:
    python visualize_tsne_embeddings.py --gencode_version 49
    python visualize_tsne_embeddings.py --gencode_version 47 --output_dir figures/
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def load_embeddings(gencode_version: int, embeddings_dir: str = '.') -> pd.DataFrame:
    """
    Load lncRNA-BERT embeddings for specified GENCODE version.
    
    Args:
        gencode_version: GENCODE version number (e.g., 47, 49)
        embeddings_dir: Directory containing embeddings files
    
    Returns:
        DataFrame with embeddings and labels
    """
    embeddings_dir = Path(embeddings_dir)
    
    # Try different possible filenames
    possible_paths = [
        embeddings_dir / f'g{gencode_version}_lncRNABERT_embeddings.h5',
        embeddings_dir / f'gencode{gencode_version}_lncRNABERT_embeddings.h5',
        embeddings_dir / f'lncRNABERT_embeddings_g{gencode_version}.h5',
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"Loading embeddings from: {path}")
            df = pd.read_hdf(path, key='data')
            return df
    
    # If not found, show error with expected locations
    raise FileNotFoundError(
        f"Could not find embeddings file for GENCODE v{gencode_version}.\n"
        f"Tried:\n" + "\n".join(f"  - {p}" for p in possible_paths)
    )


def create_tsne_visualization(
    df: pd.DataFrame,
    gencode_version: int,
    output_dir: str = '.',
    perplexity: int = 30,
    random_state: int = 42,
    dpi: int = 350
):
    """
    Create t-SNE visualization of embeddings.
    
    Args:
        df: DataFrame with embeddings (columns L0-L767) and labels
        gencode_version: GENCODE version for title
        output_dir: Where to save figure
        perplexity: t-SNE perplexity parameter
        random_state: Random seed for reproducibility
        dpi: Figure DPI (default 350 for scatter plots)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract embedding columns (L0, L1, ..., L767)
    embedding_cols = [col for col in df.columns if col.startswith('L')]
    X = df[embedding_cols].values
    
    print(f"\nEmbedding shape: {X.shape}")
    print(f"Expected: ({len(df)}, 768)")
    
    if X.shape[1] != 768:
        print(f"⚠️  WARNING: Expected 768 dimensions, got {X.shape[1]}")
    
    # Check label column
    if 'label' not in df.columns:
        raise ValueError("DataFrame must have 'label' column")
    
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    
    # Apply t-SNE
    print(f"\nApplying t-SNE (perplexity={perplexity}, random_state={random_state})...")
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    X_tsne = tsne.fit_transform(X)
    print("✓ t-SNE complete")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color mapping
    colors = {
        'ncRNA': '#FF8C00',  # Orange (or use 'orange')
        'lncRNA': '#FF8C00',  # In case labels say 'lncRNA'
        'pcRNA': '#1f77b4'   # Default matplotlib blue
    }    
    # Plot each class
    for label in sorted(df['label'].unique()):
        mask = df['label'] == label
        n_samples = mask.sum()
        
        ax.scatter(
            X_tsne[mask, 0], 
            X_tsne[mask, 1],
            c=colors.get(label, 'gray'),
            label=f'{label} (n={n_samples:,})',
            alpha=0.5,
            s=1
        )
    
    # Formatting
    ax.legend(loc='best', frameon=True, framealpha=0.9)
    ax.set_title(f't-SNE of lncRNA-BERT Embeddings on GENCODE v{gencode_version}',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('t-SNE 2', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / f'tsne_embeddings_gencode{gencode_version}.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"\n✓ Figure saved to: {output_path}")
    
    plt.close()
    
    # Also save t-SNE coordinates for future use
    tsne_df = df[['label']].copy()
    tsne_df['tsne_1'] = X_tsne[:, 0]
    tsne_df['tsne_2'] = X_tsne[:, 1]
    
    tsne_coords_path = output_dir / f'tsne_coordinates_gencode{gencode_version}.csv'
    tsne_df.to_csv(tsne_coords_path, index=True)
    print(f"✓ t-SNE coordinates saved to: {tsne_coords_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize lncRNA-BERT embeddings using t-SNE',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--gencode_version',
        type=int,
        required=True,
        help='GENCODE version number (e.g., 47, 49)'
    )
    
    parser.add_argument(
        '--embeddings_dir',
        type=str,
        default='.',
        help='Directory containing embeddings files (default: current directory)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='figures',
        help='Output directory for figures (default: figures/)'
    )
    
    parser.add_argument(
        '--perplexity',
        type=int,
        default=30,
        help='t-SNE perplexity parameter (default: 30)'
    )
    
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=350,
        help='Figure DPI (default: 350 for scatter plots)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("t-SNE VISUALIZATION OF lncRNA-BERT EMBEDDINGS")
    print("=" * 80)
    print(f"GENCODE version: {args.gencode_version}")
    print(f"Embeddings directory: {args.embeddings_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Perplexity: {args.perplexity}")
    print(f"Random state: {args.random_state}")
    print(f"DPI: {args.dpi}")
    print("=" * 80)
    
    # Load embeddings
    df = load_embeddings(args.gencode_version, args.embeddings_dir)
    
    # Create visualization
    create_tsne_visualization(
        df,
        args.gencode_version,
        output_dir=args.output_dir,
        perplexity=args.perplexity,
        random_state=args.random_state,
        dpi=args.dpi
    )
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == '__main__':
    main()