# β-VAE LNClassifier

Training and analysis code for β-VAE-based lncRNA classification, as described in:


Three model architectures are provided:

- **β-VAE + Contrastive Learning** — sequence-only model with biotype-aware contrastive loss
- **β-VAE + Genomic Features** — integrates TE and non-B DNA features alongside sequence
- **β-VAE + Genomic Features + Cross-Attention** — extends the above with a cross-modal attention mechanism for interpretable feature-sequence fusion

---

## Repository Structure

```
beta_vae_lnclassifier/
├── src/                    # main training scripts
├── analysis/               # post-training pipeline (run_all.sh + step scripts)
├── scripts/                # SLURM submission shells + data utilities
├── models/                 # model definitions (β-VAE variants)
├── trainers/               # trainers, CV utilities, loss functions
├── data/                   # data loading and preprocessing code
├── configs/                # JSON config files (one per experiment)
├── environment.yml
├── cdhit_env.yml
└── lncrnabert_environment.yml
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/[username]/lncrna-classifier.git
cd lncrna-classifier
```

### 2. Create the main environment

```bash
conda env create -f environment.yml
conda activate beta_lncrna
```

### 3. Set PYTHONPATH

Add this to your `~/.bashrc` (once):

```bash
echo 'export PYTHONPATH="/path/to/lncrna-classifier:$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc
```

---

## Data

Processed datasets (CD-HIT filtered FASTA files, TE feature CSVs, non-B DNA feature CSVs,
biotype annotation CSVs, and train/val/test split manifests) are deposited on Zenodo:

> **Zenodo DOI: [DOI — to be added upon publication]**

### Download and setup

```bash
bash scripts/setup_data.sh
```

This downloads and extracts all required files into `data/` and prints the
`DATA_ROOT` export command to add to your environment.

### Reproduce from raw GENCODE (optional)

If you want to reproduce the full preprocessing pipeline from raw GENCODE files:

```bash
# Requires the CD-HIT environment
conda env create -f cdhit_env.yml
conda activate cdhit_env

cd analysis/prepare_gencode/
GENCODE_VERSION=47 bash run_all.sh       # or 49
```

This downloads GENCODE, filters by biotype, runs CD-HIT clustering (submitted
as a SLURM job), and optionally creates the train/val + test split:

```bash
DO_SPLIT=1 TEST_SIZE=0.05 GENCODE_VERSION=47 bash run_all.sh
```

---

## Training

### Configuration

Each experiment is defined by a JSON config in `configs/`. Path placeholders
use the `DATA_ROOT` environment variable — set it before training:

```bash
export DATA_ROOT=/path/to/root
```

### Run training

Training scripts are in `src/`. Each handles 5-fold cross-validation
and optionally evaluates on the held-out test set if test FASTA paths are
provided in the config.

```bash
# β-VAE + Contrastive
python -m src.main_contrastive --config configs/beta_vae_contrastive_g47.json

# β-VAE + Genomic Features
python -m src.main_features --config configs/beta_vae_features_g47.json

# β-VAE + Genomic Features + Cross-Attention
python -m src.main_features_attn --config configs/beta_vae_features_attn_g47.json
```

SLURM submission scripts for each model type are provided in `scripts/`. If SLURM isn't available, you can simply run the commands within the shell scripts as standalone commands in the conda environment.

---

## Post-Training Analysis

The full post-training pipeline (CV evaluation, UMAP, spatial clustering,
biotype enrichment, optional GENCODE version comparison, summary report) is
driven by a single script. Example for GENCODE 47:

```bash
cd analysis/post_training_pipeline/

bash run_all.sh \
    --experiment_dir path/to/experiment \
    --config configs/your_config.json \
    --biotype_csv data/dataset_biotypes/g47_dataset_biotypes_cdhit.csv \
    --lnc_fasta data/split_gencode_47/lnc_trainval.fa \
    --pc_fasta  data/split_gencode_47/pc_trainval.fa \
    --lnc_test_fasta data/split_gencode_47/lnc_test.fa \
    --pc_test_fasta  data/split_gencode_47/pc_test.fa \
    --model_label βVAE+Attn \
    --gencode_version 47
```

Individual steps can be re-run in isolation or the pipeline can resume from
any step using `--start-from N`. Each step script is self-documented.

Outputs are saved under the experiment directory:
- `cv_evaluation_results.json` — fold-level metrics
- `test_results.json` — independent test set metrics (ensemble)
- `evaluation_csvs/` — per-sample predictions and hard case CSVs
- `umap_visualizations/` — per-fold UMAP plots
- `spatial_analysis/` — hard case spatial clustering
- `global_biotype_enrichment/` — biotype enrichment in hard cases
- `fold_attention/` — per-fold attention weight `.npz` files (attention model only)
- `attention_analysis/` — attention plots and statistics
- `ANALYSIS_SUMMARY.md` — aggregated report

---

## lncRNA-BERT Baseline

Reproducing the lncRNA-BERT baseline requires their package:

```bash
git clone https://github.com/luukromeijn/lncRNA-Py
conda env create -f lncrnabert_environment.yml
conda activate lncrnabert_env
pip install -e lncRNA-Py/
```

Run inference and compute metrics:

```bash
sbatch scripts/lncRNABERT_slurm.sh # or run individual commands
python analysis/lncrnabert_inference.py
```

Results are written to `g47_lncRNABERT_results/` and `g49_lncRNABERT_results/`.

---

## Citation

If you use this code, please cite:

```bibtex
@article{[citekey],
  title   = {[Title]},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {[Year]},
  doi     = {[DOI]}
}
```
