# β-VAE LNClassifier

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18833347.svg)](https://doi.org/10.5281/zenodo.18833347)

Training and analysis code for β-VAE-based lncRNA classification, as described in:

*[citation pending — see [Citation](#citation) below]*

---

## Model architectures

The central architecture is **BetaVAESubgroup**: a β-VAE with cross-modal
multi-head attention over explicit, registry-driven feature-group tokens.
Sequence is encoded by a CNN into a latent representation; locus-derived
features are projected into a set of named tokens and fused with the
sequence representation via cross-modal attention before classification.

Feature tokens are organised into three biological blocks, defined in
`data/feature_registry.py`:

| Block | Subgroups | Features | Source |
|-------|-----------|----------|--------|
| NonB  | 9  (APR, DR, GQ, IR, MR, STR, TRI, Z, GLOBAL) | 178 | Genomic non-B DNA motifs |
| REP (internal key `te`) | 6 (REP_CORE, REP_LCTR, REP_QUALITY, REP_PSEUDO, REP_UNKNOWN, REP_GLOBAL) | 170 | Repeat element composition |
| NonB2 | 5 (RG4, SS_STAB, SS_COUNT, SS_RELPOS, SS_BINNED) | 63 | RNA secondary structure / G-quadruplex, spliced transcript |

Adding or removing a block requires only a registry entry and matching
config keys — model code (`models/beta_vae_subgroup.py`,
`models/feature_only_classifier.py`) reads block structure dynamically via
`FeatureRegistry`.

A **Feature-Only** variant (`models/feature_only_classifier.py`) removes the
sequence encoder entirely, using a learned class token as the attention
query over the same feature tokens. This isolates locus-feature contribution
from sequence-derived signal and is used as an interpretability baseline.

### Earlier architectures (still available, not under active development)

Three earlier single-modality/pre-registry architectures remain in the
repository and are still functional:

- **β-VAE + Contrastive Learning** — sequence-only model with biotype-aware
  contrastive loss (`src/main_contrastive.py`)
- **β-VAE + Genomic Features** — integrates TE and non-B DNA features
  alongside sequence, without cross-modal attention (`src/main_features.py`)
- **β-VAE + Genomic Features + Cross-Attention** — earlier fixed two-block
  (TE/NonB) predecessor of BetaVAESubgroup (`src/main_features_attn.py`)

These are not being extended further, but their post-training analysis
pipeline (UMAP, spatial clustering, biotype enrichment — see
`analysis/post_training_pipeline/README.md`) remains available for anyone
reproducing that earlier line of work.

---

## Repository structure

```
beta_vae_lnclassifier/
├── src/                          # Training entry points
│   ├── main_subgroup.py          # BetaVAESubgroup, 5-fold CV
│   ├── main_subgroup_debug.py    # BetaVAESubgroup, single 80/20 split
│   ├── main_feature_only.py      # Feature-Only variant
│   ├── main_contrastive.py       # earlier: sequence-only contrastive
│   ├── main_features.py          # earlier: fixed TE/NonB, no attention
│   └── main_features_attn.py     # earlier: fixed TE/NonB, with attention
├── models/                       # Model definitions
│   ├── beta_vae_subgroup.py
│   ├── feature_only_classifier.py
│   ├── subgroup_projector.py
│   ├── gradient_reversal.py
│   └── token_self_attention.py
├── trainers/                     # Trainers, CV utilities, loss functions
├── data/                         # Data loading, preprocessing, feature registry
│   └── feature_registry.py       # single source of truth for block/subgroup structure
├── configs/                      # JSON config files (one per experiment)
├── analysis/
│   ├── pattern/                  # interpretability pipeline (see README below)
│   ├── benchmark/                # cross-method comparison + hard-for-all analysis
│   └── post_training_pipeline/   # earlier UMAP/spatial-clustering pipeline
├── scripts/                      # SLURM submission shells + data utilities
├── environment.yml
├── cdhit_env.yml
└── lncrnabert_environment.yml
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/cbib/beta_vae_lnclassifier.git
cd beta_vae_lnclassifier
```

### 2. Create the main environment

```bash
conda env create -f environment.yml
conda activate beta_lncrna
```

### 3. Set PYTHONPATH

Add this to your `~/.bashrc` (once):

```bash
echo 'export PYTHONPATH="/path/to/beta_vae_lnclassifier:$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc
```

---

## Data

Processed datasets (CD-HIT filtered FASTA files, TE/NonB/NonB2 feature CSVs,
biotype annotation CSVs, and train/val/test split manifests) are deposited
on Zenodo:

> **Zenodo DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18849718.svg)](https://doi.org/10.5281/zenodo.18849718)**

The NonB2 (RNA secondary structure / G-quadruplex) feature CSVs are a recent
addition; the linked Zenodo deposit will be updated with a new DOI once
finalised — check the repository for updates if you need this block
specifically.

### Download and setup

```bash
bash scripts/setup_data.sh
```

This downloads and extracts all required files into `data/` and prints the
`DATA_ROOT` export command to add to your environment.

### Reproduce from raw GENCODE (optional)

```bash
conda env create -f cdhit_env.yml
conda activate cdhit_env

cd analysis/prepare_gencode/
GENCODE_VERSION=47 bash run_all.sh       # or 49
```

```bash
DO_SPLIT=1 TEST_SIZE=0.05 GENCODE_VERSION=47 bash run_all.sh
```

---

## Training

### Configuration

Each experiment is defined by a JSON config in `configs/`. Path placeholders
use the `DATA_ROOT` environment variable:

```bash
export DATA_ROOT=/path/to/root
```

### BetaVAESubgroup (primary architecture)

```bash
# Full 5-fold CV
python src/main_subgroup.py --config configs/beta_vae_subgroup_base_g49.json --device cuda:0

# Fast single 80/20-split validation run
python src/main_subgroup_debug.py --config configs/beta_vae_subgroup_base_g49.json --device cuda:0

# Feature-Only variant (no sequence encoder)
python src/main_feature_only.py --config configs/beta_vae_feature_only_base_g49.json --device cuda:0
```

### Earlier architectures

```bash
python -m src.main_contrastive   --config configs/beta_vae_contrastive_g47.json
python -m src.main_features      --config configs/beta_vae_features_g47.json
python -m src.main_features_attn --config configs/beta_vae_features_attn_g47.json
```

SLURM submission scripts for each model type are in `scripts/`; run the
commands inside them directly if SLURM isn't available.

---

## Post-training interpretability pipeline

BetaVAESubgroup's post-training analysis implements a three-layer
interpretability framework. See `analysis/pattern/README.md` for the full pipeline and script-by-script
usage.

### Earlier post-training pipeline (UMAP / spatial clustering / biotype enrichment)

The earlier three architectures use a separate post-training pipeline
(CV evaluation, UMAP, spatial clustering, biotype enrichment). See
`analysis/post_training_pipeline/README.md` for full usage — unchanged from
prior releases.

---

## Benchmark comparison

```bash
bash analysis/benchmark/run_full_benchmark.sh \
    --release v49 \
    --base_dir gencode_v49_experiments \
    --lnc_fasta data/split_gencode_49/lnc_test.fa \
    --pc_fasta  data/split_gencode_49/pc_test.fa \
    --blnc_predictions gencode_v49_experiments/beta_vae_subgroup_base_g49_new/evaluation_csvs/test_predictions.csv
```

Compares against CPAT, CPC2, LncDC, RNAsamba, Orthrus (4-track), and
lncRNA-BERT. See `analysis/benchmark/hard_for_all/` for the cross-method
hard-case analysis (transcripts misclassified by every benchmarked method).

---

## lncRNA-BERT baseline

```bash
git clone https://github.com/luukromeijn/lncRNA-Py
conda env create -f lncrnabert_environment.yml
conda activate lncrnabert_env
pip install -e lncRNA-Py/
```

```bash
sbatch scripts/lncRNABERT_slurm.sh
python analysis/lncrnabert_inference.py
```

---

## Citation

Citation details will be added once the associated manuscript is available.

```bibtex
@article{[citekey],
  title   = {[Title]},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {[Year]},
  doi     = {[DOI]}
}
```