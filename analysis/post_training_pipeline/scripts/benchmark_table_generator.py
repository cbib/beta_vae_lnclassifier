#!/usr/bin/env python3
"""
Comprehensive Benchmark Table Generator

Includes all models: CNN, Deterministic AE, β-VAE (Contrastive/Features), lncRNA-BERT
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.table import Table


# ============================================================================
# QUICK START CONFIGURATION
# ============================================================================

# Set base directory here
BASE_DIR = Path('.')  # or Path('/path/to/your/DL_benchmark')

CONFIG = {
    'experiments': {
        # ===== GENCODE v47 Models =====
        'CNN (CSE, k=9)': {
            '47': BASE_DIR / 'gencode_v47_experiments/cnn_cse_k9_g47',
            '49': BASE_DIR / 'gencode_v49_experiments/cnn_g49'
        },
        'β-VAE (Contrastive)': {
            '47': BASE_DIR / 'gencode_v47_experiments/beta_vae_contrastive_g47',
            '49': BASE_DIR / 'gencode_v49_experiments/beta_vae_contrastive_g49'
        },
        'β-VAE (Features)': {
            '47': BASE_DIR / 'gencode_v47_experiments/beta_vae_features_g47',
            '49': BASE_DIR / 'gencode_v49_experiments/beta_vae_features_g49'
        },
        'β-VAE (Features + Attention)': {
        '47': BASE_DIR / 'gencode_v47_experiments/beta_vae_features_attn_g47',
        '49': BASE_DIR / 'gencode_v49_experiments/beta_vae_features_attn_g49'
        }
    },
    
    'lncrnabert': {
        '47': {
            'pred_csv': Path('g47_lncRNABERT_results/g47_lncRNABERT_results.csv'),
            'pc_fasta': BASE_DIR / 'data/split_gencode_47/pc_test.fa',
            'lnc_fasta': BASE_DIR / 'data/split_gencode_47/lnc_test.fa'
        },
        '49': {
            'pred_csv': Path('g49_lncRNABERT_results/g49_lncRNABERT_results.csv'),
            'pc_fasta': BASE_DIR / 'data/split_gencode_49/pc_test.fa',
            'lnc_fasta': BASE_DIR / 'data/split_gencode_49/lnc_test.fa'
        }
    },
    
    'output': 'benchmark_comparison_full.csv'
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def parse_fasta_ids(fasta_path):
    """Parse transcript IDs from FASTA file."""
    ids = []
    if not Path(fasta_path).exists():
        print(f"Warning: FASTA file not found: {fasta_path}")
        return ids
    
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                transcript_id = line[1:].split('|')[0].strip()
                ids.append(transcript_id)
    return ids

def load_test_results(exp_dir):
    """Load test_results.json from experiment directory."""
    exp_path = Path(exp_dir)
    if not exp_path.exists():
        raise FileNotFoundError(f"Directory not found: {exp_dir}")
    json_path = exp_path / 'test_results.json'
    if not json_path.exists():
        raise FileNotFoundError(f"No test_results.json in {exp_dir}")
    with open(json_path, 'r') as f:
        return json.load(f)

# ============================================================================
# TABLE FIGURE GENERATION
# ============================================================================

def extract_mean_value(metric_str):
    """Extract mean value from formatted string (e.g., '0.9506 ± 0.0005' -> 0.9506)."""
    if '±' in metric_str:
        return float(metric_str.split('±')[0].strip())
    return float(metric_str)


def create_table_figure(df, output_path='benchmark_table.png', dpi=1200):
    """
    Create a high-resolution publication-quality table figure.
    
    Args:
        df: DataFrame with benchmark results
        output_path: Output file path for the figure
        dpi: Resolution (default 1200 for publication quality)
    """
    print(f"\nCreating high-resolution table figure (dpi={dpi})...")
    
    # Sort by dataset and accuracy for proper ranking
    df_sorted = df.copy()
    df_sorted['_acc_value'] = df_sorted['Accuracy'].apply(extract_mean_value)
    
    # Prepare display columns
    display_cols = ['Model', 'Training', 'Dataset', 'Accuracy', 
                    'Precision (macro)', 'Recall (macro)', 'F1 (macro)']
    df_display = df_sorted[display_cols].copy()
    
    # Identify best and second-best values per dataset per metric
    metric_cols = ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1 (macro)']
    formatting = {}  # {(row_idx, col_name): 'bold' or 'italic'}
    
    for dataset in df_display['Dataset'].unique():
        mask = df_display['Dataset'] == dataset
        df_subset = df_display[mask].copy()
        
        for metric in metric_cols:
            # Extract numeric values
            values = df_subset[metric].apply(extract_mean_value).values
            indices = df_subset.index.values
            
            # Get top 2 indices
            if len(values) >= 2:
                sorted_idx = np.argsort(values)[::-1]  # Descending
                best_idx = indices[sorted_idx[0]]
                second_idx = indices[sorted_idx[1]]
                
                formatting[(best_idx, metric)] = 'bold'
                formatting[(second_idx, metric)] = 'italic'
    
    # Create figure
    n_rows, n_cols = df_display.shape
    
    # Calculate figure size based on content
    col_widths = [2.5, 2.5, 2.0, 2.2, 2.2, 2.2, 2.2]  # Column widths in inches
    fig_width = sum(col_widths) + 0.5
    fig_height = (n_rows + 2) * 0.35  # Header + rows + padding
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
    table_data = []
    table_data.append(list(df_display.columns))  # Header
    for idx, row in df_display.iterrows():
        table_data.append([str(row[col]) for col in df_display.columns])
    
    # Create table
    table = ax.table(cellText=table_data, loc='center', cellLoc='left',
                     colWidths=[w/fig_width for w in col_widths])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)  # Scale row height
    
    # Style header row
    for j in range(n_cols):
        cell = table[(0, j)]
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(weight='bold', color='white', ha='left')
        cell.set_edgecolor('white')
        cell.set_linewidth(1.5)
    
    # Style data rows with alternating colors
    for i in range(1, n_rows + 1):
        row_idx = df_display.index[i - 1]
        
        # Alternate row colors
        bg_color = '#ECF0F1' if i % 2 == 0 else 'white'

         # Track last G47 row
        if 'v47' in df_display.loc[row_idx, 'Dataset']:
            if i < n_rows:
                next_idx = df_display.index[i]
                if 'v49' in df_display.loc[next_idx, 'Dataset']:
                    g47_last_row_num = i

        dataset = df_display.loc[row_idx, 'Dataset']
        
        for j, col_name in enumerate(df_display.columns):
            cell = table[(i, j)]
            cell.set_facecolor(bg_color)
            cell.set_edgecolor('#BDC3C7')
            cell.set_linewidth(0.5)
            
            # Apply formatting (bold for best, italic for second-best)
            if (row_idx, col_name) in formatting:
                fmt = formatting[(row_idx, col_name)]
                if fmt == 'bold':
                    cell.set_text_props(weight='bold', style='normal')
                elif fmt == 'italic':
                    cell.set_text_props(weight='normal', style='italic')
    
    plt.tight_layout()

     # Draw separator line between G47 and G49 (after layout is done)
    if g47_last_row_num is not None:
        # Force render to get actual positions
        fig.canvas.draw()
        
        # Get cell bounding box
        cell = table[(g47_last_row_num, 0)]
        bbox = cell.get_window_extent(fig.canvas.get_renderer())
        bbox_fig = bbox.transformed(fig.transFigure.inverted())
        
        # Draw line at bottom of last G47 cell
        line = plt.Line2D([0.025, 0.975], [bbox_fig.y0, bbox_fig.y0],
                         transform=fig.transFigure,
                         color='#34495E', linewidth=2.5, zorder=100)
        fig.add_artist(line)
    
    # Save with high DPI
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Table figure saved: {output_path} ({dpi} dpi)")
    print(f"  - Best values per column: BOLD")
    print(f"  - Second-best values: italic")
    
    return output_path


# ============================================================================
# MODEL PROCESSORS
# ============================================================================

def process_cv_model(model_name, dataset, exp_dir):
    """Process a model's test set results."""
    try:
        r = load_test_results(exp_dir)

        if 'β-VAE' in model_name or 'VAE' in model_name or 'CNN' in model_name:
            training = f"5-fold CV ensemble (n={r.get('n_folds_ensembled', 5)})"
        else:
            training = '5-fold CV'

        return {
            'Model':              model_name,
            'Training':           training,
            'Dataset':            f'GENCODE v{dataset} Test (5%)',
            'Accuracy':           f"{r['accuracy']:.4f}",
            'Precision (macro)':  f"{r['precision']:.4f}",
            'Recall (macro)':     f"{r['recall']:.4f}",
            'F1 (macro)':         f"{r['f1']:.4f}",
            '_acc_mean':          r['accuracy'],
            '_n_folds':           r.get('n_folds_ensembled', 1),
        }
    except Exception as e:
        print(f"  ✗ Error processing {model_name} - {dataset}: {e}")
        return None


def process_lncrnabert(dataset, config):
    """Process lncRNA-BERT zero-shot results."""
    try:
        # Load predictions
        pred_df = pd.read_csv(config['pred_csv'])
        pred_df['transcript_id'] = pred_df['id'].str.split('|').str[0]
        
        # Load ground truth
        pc_ids = parse_fasta_ids(config['pc_fasta'])
        lnc_ids = parse_fasta_ids(config['lnc_fasta'])
        
        if not pc_ids or not lnc_ids:
            raise ValueError("Could not load FASTA files")
        
        # Create label mapping
        true_labels = {tid: 1 for tid in pc_ids}  # PC = 1
        true_labels.update({tid: 0 for tid in lnc_ids})  # lncRNA = 0
        
        # Match predictions
        pred_df['true_label'] = pred_df['transcript_id'].map(true_labels)
        pred_df['pred_label'] = (pred_df['class'] == 'pcRNA').astype(int)
        pred_df = pred_df.dropna(subset=['true_label'])
        
        # Calculate metrics
        y_true = pred_df['true_label'].values.astype(int)
        y_pred = pred_df['pred_label'].values
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        return {
            'Model': 'lncRNA-BERT (3-mer)',
            'Training': 'Zero-shot Inference',
            'Dataset': f'GENCODE v{dataset} Test (5%)',
            'Accuracy': f"{accuracy:.4f}",
            'Precision (macro)': f"{precision:.4f}",
            'Recall (macro)': f"{recall:.4f}",
            'F1 (macro)': f"{f1:.4f}",
            '_acc_mean': accuracy,
            '_n_folds': 1
        }
    except Exception as e:
        print(f"  ✗ Error processing lncRNA-BERT - {dataset}: {e}")
        return None


# ============================================================================
# MAIN TABLE GENERATOR
# ============================================================================

def create_benchmark_table(generate_figure=True, figure_dpi=1200):
    """Generate the complete benchmark comparison table."""
    print("="*80)
    print("BENCHMARK TABLE GENERATOR")
    print("="*80 + "\n")
    
    results = []
    
    # Process all CV models
    print("Processing cross-validated models...")
    for model_name, datasets in CONFIG['experiments'].items():
        for dataset, exp_dir in datasets.items():
            print(f"  Processing {model_name} - {dataset}...", end=" ")
            result = process_cv_model(model_name, dataset, exp_dir)
            if result:
                results.append(result)
                print("✓")
            else:
                print("✗")
    
    # Process lncRNA-BERT
    print("\nProcessing lncRNA-BERT (zero-shot)...")
    for dataset, config in CONFIG['lncrnabert'].items():
        print(f"  Processing lncRNA-BERT - {dataset}...", end=" ")
        result = process_lncrnabert(dataset, config)
        if result:
            results.append(result)
            print("✓")
        else:
            print("✗")
    
    if not results:
        print("\n✗ No results processed! Check paths in CONFIG.")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by dataset then accuracy
    model_order = {
    'CNN (CSE, k=9)':               1,
    'β-VAE (Contrastive)':          2,
    'β-VAE (Features)':             3,
    'β-VAE (Features + Attention)': 4,
    'lncRNA-BERT (3-mer)':          5,
    }
    df['_model_order'] = df['Model'].map(model_order)
    df = df.sort_values(by=['Dataset', '_model_order'], ascending=[True, True])    
    # Display columns
    display_cols = [
        'Model', 'Training', 'Dataset', 'Accuracy',
        'Precision (macro)', 'Recall (macro)', 'F1 (macro)'
    ]
    df_display = df[display_cols].copy()
    
    # Save to CSV
    output_path = Path(CONFIG['output'])
    df_display.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"✓ Benchmark table saved to: {output_path}")
    print(f"{'='*80}\n")
    
    # Display table
    print("TABLE PREVIEW:")
    print(df_display.to_string(index=False))
    
    # Generate high-resolution figure if requested
    if generate_figure:
        figure_path = output_path.parent / f"{output_path.stem}_figure.png"
        create_table_figure(df, figure_path, dpi=figure_dpi)
    
    # Print Markdown version
    print("\n" + "="*80)
    print("MARKDOWN VERSION (copy-paste into paper/report):")
    print("="*80 + "\n")
    
    for dataset in sorted(df_display['Dataset'].unique()):
        df_subset = df_display[df_display['Dataset'] == dataset]
        print(f"\n### {dataset}\n")
        print("| Model | Training | Accuracy | Precision | Recall | F1 |")
        print("|-------|----------|----------|-----------|--------|-----|")
        for _, row in df_subset.iterrows():
            print(f"| {row['Model']} | {row['Training']} | {row['Accuracy']} | "
                  f"{row['Precision (macro)']} | {row['Recall (macro)']} | "
                  f"{row['F1 (macro)']} |")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS:")
    print("="*80)
    
    for dataset in sorted(df_display['Dataset'].unique()):
        df_subset = df[df['Dataset'] == dataset]
        print(f"\n{dataset}:")
        print(f"  Best model: {df_subset.iloc[0]['Model']}")
        print(f"  Best accuracy: {df_subset.iloc[0]['Accuracy']}")
        print(f"  Models evaluated: {len(df_subset)}")
    
    return df_display


# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == '__main__':
    import sys
    
    # Parse simple command-line arguments
    generate_figure = True
    figure_dpi = 1200
    
    if '--no-figure' in sys.argv:
        generate_figure = False
    
    if '--dpi' in sys.argv:
        try:
            dpi_idx = sys.argv.index('--dpi')
            figure_dpi = int(sys.argv[dpi_idx + 1])
        except (ValueError, IndexError):
            print("Warning: Invalid --dpi value, using default 1200")
    
    df = create_benchmark_table(generate_figure=generate_figure, figure_dpi=figure_dpi)
    
    if df is not None:
        print("\n" + "="*80)
        print("✓ Benchmark table generation complete!")
        if generate_figure:
            print(f"✓ High-resolution figure created at {figure_dpi} dpi")
        print("="*80)
        print("\nUsage:")
        print("  python benchmark_table_generator.py              # Generate CSV + figure (1200 dpi)")
        print("  python benchmark_table_generator.py --no-figure  # Generate CSV only")
        print("  python benchmark_table_generator.py --dpi 300    # Custom DPI")
    else:
        print("\n✗ Table generation failed. Check paths and file locations.")