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
| NonB  | 9  (APR, DR, GQ, IR, MR, STR, TRI, Z, GLOBAL) | 178 | Genomic non-B DNA motifs (`genomic` source in all reported models) |
| REP (internal key `te`) | 6 (REP_CORE, REP_LCTR, REP_QUALITY, REP_PSEUDO, REP_UNKNOWN, REP_GLOBAL) | 170 | Repeat element composition |
| NonB2 | 5 (RG4, SS_STAB, SS_COUNT, SS_RELPOS, SS_BINNED) | 63 | RNA secondary structure / G-quadruplex, spliced transcript |

The feature source of each block can be overridden per experiment with
`model.block_sources` (e.g. `{"nonb": "genomic"}`). Omitted entries fall back
to the registry default; for NonB the registry default is `both`, so reported
models set `nonb: genomic` explicitly.

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

## Gene-grouped splits (current release)

All reported models are trained and evaluated on **gene-grouped** splits:
every transcript of a gene falls in the same train/validation/test partition
(`group_by_gene`), so no gene appears on both sides of a split. The earlier
transcript-level splits shared genes between train and test (about 90% of
test transcripts for GENCODE v47 and 92% for v49), which inflated test
performance. The released splits have 0% gene overlap.

- Splits live in `data/split_gencode_<v>_genegrouped/`; experiment outputs use
  the `_genegrouped` suffix (e.g. `gencode_v49_experiments_genegrouped/`).
- Configs ending in `_genegrouped.json` select these splits. CV folds are built
  with `get_cv_splits_from_config`, which groups by gene.
- Each model is scored on its own test set. Cross-method benchmarks
  additionally report a common subset of transcripts scored by every method.
- Pre-audit transcript-level configs and results are kept for provenance only
  (`configs/legacy/`) and are not used for any reported number.

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

Processed datasets (CD-HIT filtered FASTA files, REP/NonB/NonB2 feature CSVs,
biotype annotation CSVs, and gene-grouped train/val/test split manifests) are
deposited on Zenodo:

> **Zenodo DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18849718.svg)](https://doi.org/10.5281/zenodo.18849718)**

The linked deposit predates the gene-grouped splits and the NonB2 features;
a new DOI covering both will be added here at publication.

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
DO_SPLIT=1 TEST_SIZE=0.05 GENCODE_VERSION=47 bash run_all.sh   # gene-grouped split
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
# Full 5-fold gene-grouped CV
python src/main_subgroup.py --config configs/beta_vae_subgroup_g49_genegrouped.json --device cuda:0

# Fast single 80/20-split validation run
python src/main_subgroup_debug.py --config configs/beta_vae_subgroup_g49_genegrouped.json --device cuda:0

# Feature-Only variant (no sequence encoder)
python src/main_feature_only.py --config configs/beta_vae_feature_only_g49_genegrouped.json --device cuda:0
```

For GENCODE v47 use `configs/beta_vae_subgroup_g47_genegrouped.json`.

### Earlier architectures

These use the original transcript-level splits and are not gene-disjoint.

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
interpretability framework:

1. **Diagnostics** — attention and ablation.
2. **Associational** — Fréchet distance and pattern alignment.
3. **Interventional** — activation patching at projector sites, scored with a
   continuous effect-size metric and Interchange Intervention Accuracy (IIA),
   the stricter test of whether an intervention flips the model's decision.
   Includes length/GC-matched pairs, controls (shuffled, same-class, noise) at
   single/block/all scope, a held-out-half joint search over subgroup sets,
   a synergy test against a logit-additive null, and a pairwise test over all
   190 subgroup pairs (BH-corrected within fold, Fisher-combined across folds).

Run the whole pipeline for one experiment with:

```bash
bash scripts/layers.sh <experiment_dir> --config configs/<config>.json
```

Stages (1.1–1.4, 2.1, 3.1–3.5, S.1–S.2) write to `<experiment_dir>/interpretability/`
and are resumable via `.done` markers keyed on the command signature.
`--force` reruns everything, `--only <ids>` selects stages, `--dry-run` prints
the plan. See `analysis/pattern/README.md` for script-by-script usage.

### Earlier post-training pipeline (UMAP / spatial clustering / biotype enrichment)

The earlier three architectures use a separate post-training pipeline
(CV evaluation, UMAP, spatial clustering, biotype enrichment). See
`analysis/post_training_pipeline/README.md` for full usage — unchanged from
prior releases.

---

## Benchmark comparison

On SLURM, `scripts/submit_benchmarks.sh` submits the per-release benchmark job
(`scripts/benchmark_job.sh`). Direct invocation:

```bash
bash analysis/benchmark/run_full_benchmark.sh \
    --release v49 \
    --base_dir gencode_v49_experiments_genegrouped \
    --lnc_fasta data/split_gencode_49_genegrouped/lnc_test.fa \
    --pc_fasta  data/split_gencode_49_genegrouped/pc_test.fa \
    --blnc_predictions gencode_v49_experiments_genegrouped/beta_vae_subgroup_base_g49_genegrouped/evaluation_csvs/test_predictions.csv
```

Compares against CPAT, CPC2, LncDC, RNAsamba, Orthrus (4-track), and
lncRNA-BERT, reporting accuracy/precision/recall/F1 with 95% gene-clustered
bootstrap confidence intervals (10,000 resamples) and AUC with 95% DeLong CI
per method. Results are given on the full test set and on the common subset
scored by every method. See `analysis/benchmark/README.md` for the full pipeline and
`analysis/benchmark/hard_for_all/` for the cross-method hard-case
analysis (transcripts hard for every benchmarked method, or a
configurable subset — see that README for the threshold option).

---

## lncRNA-BERT baseline

Used zero-shot (pretrained weights, no fine-tuning).

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