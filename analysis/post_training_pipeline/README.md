# Post-Training Analysis Pipeline

Complete pipeline for analyzing trained β-VAE models, from evaluation to biotype enrichment analysis. Proof-tested for Beta-VAE and GENCODE v47/49.

## Quick Start
```bash
# Run entire pipeline
bash run_all.sh --experiment_dir experiments/beta_vae_g47 \
                --config configs/beta_vae_contrastive_g47.json \
                --biotype_csv data/processed/gencode47_dataset_biotypes_cdhit.csv

# Or run steps individually
bash 01_evaluate_cv_folds.sh --experiment_dir experiments/beta_vae_g47 ...
bash 02_generate_umap.sh --experiment_dir experiments/beta_vae_g47 ...
bash 03_spatial_clustering.sh --umap_dir experiments/beta_vae_g47/umap_visualizations ...
bash 04_biotype_enrichment.sh --spatial_dir experiments/beta_vae_g47/spatial_analysis ...
```

## Pipeline Overview
```
Training (main_contrastive.py)
    ↓
[1] Evaluate CV Folds
    • Load trained models (fold_N_best.pt)
    • Compute CV statistics (mean ± std)
    • Generate hard_cases.csv
    • Extract embeddings (embeddings_all_folds.npz)
    ↓
[2] Generate UMAP
    • Compute UMAP per fold
    • Create visualizations (all samples, hard cases, biotypes)
    • Save umap_embeddings.csv per fold
    ↓
[3] Spatial Clustering
    • K-means clustering on hard cases (default: 5 regions)
    • Analyze class balance per region
    • Create biotype composition heatmaps
    • Generate samples_with_regions.csv
    ↓
[4] Biotype Enrichment
    • Chi-squared test per biotype vs global baseline
    • FDR correction (Benjamini-Hochberg)
    • Visualize enriched/depleted biotypes
    ↓
[5] GENCODE Comparison (Optional)
    • Compare novel vs common transcripts
    • Hard case enrichment by novelty status
    • Spatial clustering analysis
```

---

## Requirements #TODO

### Software
- Python 3.10+
- PyTorch (with trained models)
- Models were trained using CUDA 12.1

### Input Files
1. **Trained models**: `experiment_dir/models/fold_N_best.pt` (N=0,1,2,3,4)
2. **Config file**: Training configuration (JSON format)
3. **Biotype CSV**: Biotype annotations matching training data
4. **FASTA files**: lncRNA and protein-coding sequences (for evaluation)

### Python Packages

It is recommended to create a conda environment and then install requirements:
```bash
conda create env -f environment.yml
conda activate beta_lncrna
```

---

## Detailed Steps (Example with Gencode47)

### Step 0: Prerequisites Check

Verifies all required post-training files and dependencies exist.
```bash
bash 00_prerequisites.sh --experiment_dir experiments/beta_vae_g47
```

**Checks**:
- Trained model checkpoints (fold_0_best.pt ... fold_4_best.pt)
- Config file
- Biotype CSV
- FASTA files
- Python packages

---

### Step 1: Evaluate CV Folds

Evaluates all trained folds, computes statistics, and generates hard cases.
```bash
bash 01_evaluate_cv_folds.sh \
    --experiment_dir experiments/beta_vae_g47 \
    --config configs/beta_vae.json \
    --biotype_csv data/processed/gencode47_dataset_biotypes_cdhit90.csv \
    --n_folds 5
```

**Outputs**:
- `cv_evaluation_results.json`: Overall CV statistics
- `cv_fold_results.csv`: Per-fold metrics
- `evaluation_csvs/`
  - `all_sample_predictions.csv`: Predictions for all samples across folds
  - `hard_cases.csv`: Samples with incorrect predictions or confidence < 0.6
- `embeddings_all_folds.npz`: Latent embeddings from all folds
- `performance_figures/`
  - `roc_pr_curves.png`: ROC and PR curves with mean ± std
- `length_class_analysis/`
  - `length_class_interaction_figure.png`: Two-panel figure of hard case rates per class and transcript length relative difficulty, alongside its data.

**Key Metrics**:
- Per-fold confusion matrices and metrics
- Hard case statistics

---

### Step 2: Generate UMAP

Creates UMAP visualizations per fold.
```bash
bash 02_generate_umap.sh \
    --embeddings experiments/beta_vae_g47/embeddings_all_folds.npz \
    --hard_cases experiments/beta_vae_g47/evaluation_csvs/hard_cases.csv \
    --biotype_csv data/processed/gencode47_dataset_biotypes_cdhit90.csv \
    --output_dir experiments/beta_vae_g47/umap_visualizations
```

**Parameters**:
- `--n_neighbors 30`: UMAP neighborhood size (default: 30)
- `--min_dist 0.1`: UMAP minimum distance (default: 0.1)
- `--metric euclidean`: Distance metric (euclidean, manhattan, cosine, correlation)

**Outputs** (per fold):
```
umap_visualizations/
├── fold_0/
│   ├── umap_all_samples.png        # Binary classification view
│   ├── umap_hard_cases.png         # Hard cases highlighted
│   ├── umap_by_biotype.png         # Biotype distribution
│   └── umap_embeddings.csv         # Coordinates + metadata
├── fold_1/
│   └── ...
└── fold_N/
```

**Note**: Each fold's UMAP is computed independently.

**Time**: 5-15 minutes per fold

---

### Step 3: Spatial Clustering Analysis

Identifies spatial regions of hard cases using K-means clustering.
```bash
bash 03_spatial_clustering.sh \
    --umap_dir experiments/beta_vae_g47/umap_visualizations \
    --n_regions 5 \
    --output_dir experiments/beta_vae_g47/spatial_analysis
```

**Method**:
1. K-means clustering on UMAP coordinates of **hard cases only**
2. Classify regions by class balance:
   - `lnc_dominated`: >70% lncRNA
   - `pc_dominated`: >70% protein-coding
   - `frontier`: Mixed (within 20%)
   - `lnc_majority` / `pc_majority`: 50-70%

**Outputs** (per fold):
```
spatial_analysis/
├── fold_0/
│   ├── spatial_regions_labeled.png      # Main UMAP with regions
│   ├── region_biotype_heatmap.png       # Biotype composition
│   ├── samples_with_regions.csv         # Annotated samples
│   ├── region_statistics.csv            # Region summaries
│   └── region_biotype_composition.csv   # Detailed composition
├── fold_1/
│   └── ...
├── all_folds_region_statistics.csv      # Combined statistics
└── cross_fold_summary.png                # Cross-fold consistency
```

**Key Questions**:
- Are hard cases clustered or dispersed?
- Do specific biotypes concentrate in certain regions?
- Are regions consistent across folds?

**Time**: 2-5 minutes per fold

---

### Step 4: Biotype Enrichment Analysis

Tests which biotypes are enriched/depleted in hard cases using binomial test.
```bash
bash 04_biotype_enrichment.sh \
    --spatial_dir experiments/beta_vae_g47/spatial_analysis \
    --output_dir experiments/beta_vae_g47/biotype_enrichment \
    --min_count 10
```

**Method**:
1. Compute global baseline hard case rate
2. For each biotype: Test if hard case rate differs from baseline
3. Binomial test (two-sided)
4. FDR correction (Benjamini-Hochberg)

**Outputs**:
```
biotype_enrichment/
├── global_biotype_enrichment.csv     # Full results table
└── global_biotype_enrichment.png     # Horizontal bar chart
```

**CSV Columns**:
- `biotype`: Biotype name
- `n_total`: Total samples
- `n_hard`: Hard cases
- `hard_rate`: Hard case rate for this biotype
- `baseline_rate`: Global hard case rate
- `fold_enrichment`: Enrichment vs baseline (>1 = enriched)
- `p_value`: Binomial test p-value
- `fdr`: FDR-corrected p-value
- `significant`: FDR < 0.05

**Interpretation**:
- `fold_enrichment > 1`: Biotype more likely to be misclassified
- `fold_enrichment < 1`: Biotype easier to classify
- `significant = True`: Statistically significant (FDR < 0.05)

---

### Step 5: GENCODE Comparison (Optional)

Analyzes novelty patterns (novel vs common transcripts) in embedding space.

**Note**: This step is **optional** and only relevant if comparing GENCODE versions. FASTA files with common and novel characterization for GENCODE are provided in Zenodo.
```bash
bash 05_gencode_comparison.sh \
    --embeddings_npz experiments/beta_vae_g47/embeddings_all_folds.npz \
    --fold_embeddings_dir experiments/beta_vae_g47/umap_visualizations \
    --hard_cases experiments/beta_vae_g47/evaluation_csvs/hard_cases.csv \
    --novel_fasta resources/gencode.v47.new_with_class_transcripts.fa \
    --common_fasta resources/gencode.v47.common_no_class_change_transcripts.fa \
    --reannotated_fasta resources/gencode.v47.common_class_change_transcripts.fa \
    --output_dir experiments/beta_vae_g47/gencode_novelty_analysis
```

**Outputs**:
- Per-fold novelty visualizations
- Hard case enrichment by novelty status
- Spatial clustering metrics
- Classification performance comparison
- Cross-fold summary figures

**Time**: 10-20 minutes

---

### Step 6: Generate Summary Report

Aggregates all results into a comprehensive summary.
```bash
bash 06_generate_summary_report.sh \
    --experiment_dir experiments/beta_vae_g47 \
    --output_file experiments/beta_vae_g47/ANALYSIS_SUMMARY.md
```

**Outputs**:
- `ANALYSIS_SUMMARY.md`: Markdown report with:
  - CV performance metrics
  - Hard case statistics
  - Spatial clustering summary
  - Enriched/depleted biotypes
  - Links to all figures

---

## Output Directory Structure

After running the complete pipeline, you should have:
```
experiments/beta_vae_g47/
├── models/
│   ├── fold_0_best.pt
│   ├── fold_1_best.pt
│   └── ...
├── cv_evaluation_results.json
├── cv_fold_results.csv
├── embeddings_all_folds.npz
├── evaluation_csvs/
│   ├── all_sample_predictions.csv
│   └── hard_cases.csv
├── performance_figures/
│   └── roc_pr_curves.png
├── umap_visualizations/
│   ├── fold_0/
│   │   ├── umap_all_samples.png
│   │   ├── umap_hard_cases.png
│   │   ├── umap_by_biotype.png
│   │   └── umap_embeddings.csv
│   └── ...
├── spatial_analysis/
│   ├── fold_0/
│   │   ├── spatial_regions_labeled.png
│   │   ├── region_biotype_heatmap.png
│   │   ├── samples_with_regions.csv
│   │   └── region_statistics.csv
│   ├── all_folds_region_statistics.csv
│   └── cross_fold_summary.png
├── biotype_enrichment/
│   ├── global_biotype_enrichment.csv
│   └── global_biotype_enrichment.png
├── length_class_analysis/
│   ├── length_class_hard_rates.csv
│   └── length_class_interaction_figure.png
│   └── relative_difficulty_by_length.csv
└── ANALYSIS_SUMMARY.md
```

---

## Key Findings to Report

### 1. Model Performance
- **Accuracy**: mean ± std across folds
- **F1 score (macro)**: Balanced performance metric
- **AUC-ROC**: Discrimination ability
- **AP (PR-AUC)**: Precision-recall balance

### 2. Hard Cases
- **Overall hard case rate**: What % of samples are consistently misclassified?
- **Per-class rates**: Are lncRNA or PC harder to classify?
- **Hard case characteristics**: Length distribution, biotype patterns

### 3. Spatial Organization
- **Number of regions**: How many distinct hard case clusters?
- **Region characteristics**: Class balance, biotype composition
- **Cross-fold consistency**: Are regions reproducible?

### 4. Biotype Patterns
- **Enriched biotypes**: Which are harder to classify? (e.g., retained_intron, NMD)
- **Depleted biotypes**: Which are easier? (e.g., canonical protein_coding, lncRNA)
- **Statistical significance**: FDR-corrected p-values

---

## Troubleshooting

### Issue: "Missing fold checkpoints"
**Solution**: Ensure all fold_N_best.pt files exist in `models/` directory, as created during training. Original model weights are published through Zenodo. 

### Issue: "Biotypes are generic labels (biotype_0, biotype_1)"
**Solution**: Run with correct `--biotype_csv` that matches training data grouping (min_count=500)

### Issue: "UMAP shows fold separation"
**Problem**: Folds were concatenated before UMAP
**Solution**: Use `--per_fold` flag (default in this pipeline)

### Issue: "No significant biotypes found"
**Possible causes**:
- Small dataset (increase sample size)
- Low hard case rate (model too good)
- Uniform biotype distribution
**Solution**: Check `global_biotype_enrichment.csv` for trends even if not significant

### Issue: "Memory error during UMAP"
**Solution**: Reduce dataset size or use fewer neighbors (`--n_neighbors 15`)

---

## Customization

### Change Number of Spatial Regions
```bash
bash 03_spatial_clustering.sh --n_regions 3  # Use 3 instead of 5
```

### Adjust UMAP Parameters
```bash
bash 02_generate_umap.sh --n_neighbors 50 --min_dist 0.05
```

### Lower Biotype Count Threshold
```bash
bash 04_biotype_enrichment.sh --min_count 5  # Include biotypes with ≥5 samples
```

### Run Single Fold Only
```bash
# Modify scripts to process specific fold
python scripts/visualize_embeddings.py \
    --embeddings embeddings_all_folds.npz \
    --output_dir umap_fold_0 \
    --folds 0  # Only process fold 0
```
---

## Contact

For issues or questions, please open an issue on GitHub or contact mikael.georges@ibgc.cnrs.fr.