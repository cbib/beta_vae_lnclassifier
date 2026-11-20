# β-VAE LNClassifier

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18833347.svg)](https://doi.org/10.5281/zenodo.18833347)

Training and analysis code for β-VAE-based lncRNA classification, as described in:

*[citation pending — see [Citation](#citation) below]*

---

## Model architectures

The central architecture is **BetaVAESubgroup**: a β-VAE with cross-modal
multi-head attention over explicit, registry-driven feature-group tokens.
Sequence is encoded by a CNN into a latent representation; feature tables are
projected into a set of named tokens and fused with the sequence representation
via cross-modal attention before classification.

Feature tokens are organised into three biological blocks, defined in
`data/feature_registry.py`:

| Block | Subgroups | Features | Source |
|-------|-----------|----------|--------|
| NonB  | 9  (APR, DR, GQ, IR, MR, STR, TRI, Z, GLOBAL) | 178 | Genomic non-B DNA motifs (`genomic` source in all reported models) |
| REP (internal key `te`) | 6 (REP_CORE, REP_LCTR, REP_QUALITY, REP_PSEUDO, REP_UNKNOWN, REP_GLOBAL) | 170 | Repeat element composition, genomic locus and spliced transcript (`both`) |
| NonB2 | 5 (RG4, SS_STAB, SS_COUNT, SS_RELPOS, SS_BINNED) | 63 | RNA secondary structure / G-quadruplex, spliced transcript |

**Feature origins.** Some features differ when a tool runs on the genomic locus
or on the spliced transcript, so NonB and REP are computed on both.
`data/processed_features_genomic/` holds the locus runs (and the NonB2 tables),
`data/processed_features/` the spliced-transcript runs. NonB2 is
transcript-level only. Reported models read NonB from the locus runs
(`nonb: genomic`) and REP from both.

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
query over the same feature tokens. This isolates the feature contribution
from sequence-derived signal and is used as an interpretability baseline.

### Earlier architectures

The sequence-only contrastive model, the two fixed-block feature models and
their UMAP and spatial-clustering analysis are no longer in the main branch.
The tagged releases `v1.0.0` and `v2.0.0` reproduce them.

---

## Repository structure

```
beta_vae_lnclassifier/
├── src/                          # Training entry points
│   ├── main_subgroup.py          # BetaVAESubgroup, 5-fold CV
│   └── main_feature_only.py      # Feature-Only variant
├── models/                       # Model definitions
│   ├── beta_vae_subgroup.py
│   ├── feature_only_classifier.py
│   ├── subgroup_projector.py
│   └── model_builder.py
├── trainers/                     # Trainer and CV loop
├── data/                         # Data loading, feature scaling, feature registry
│   └── feature_registry.py       # single source of truth for block/subgroup structure
├── configs/                      # One config per release and model
├── analysis/
│   ├── evaluation/               # per-fold evaluation of trained checkpoints
│   ├── pattern/                  # interpretability pipeline (see README below)
│   ├── benchmark/                # cross-method comparison
│   ├── prepare_gencode/          # GENCODE download, CD-HIT, splits
│   └── triframe/                 # equivalence checks
├── scripts/                      # prep, train, submit, layers, benchmark
├── tests/
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

- Splits live in `data/split_gencode_<v>/`; experiment outputs in
  `gencode_v<v>_experiments/`.
- CV folds are built with `get_cv_splits_from_config`, which groups by gene.
- Each model is scored on its own test set. Cross-method benchmarks
  additionally report a common subset of transcripts scored by every method.
- The transcript-level splits and results of the earlier releases are in the
  tags `v1.0.0` and `v2.0.0`; no reported number uses them.

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

### Pipeline for one release

Run from the repository root with `DATA_ROOT` set.

```bash
# Stage 0: fit the scaler banks and write the cleaned feature tables
bash scripts/prep_features.sh 49 --te_csv FILE --nonb_csv FILE [--nonb2_csv FILE]

# Stages 1-2: BetaVAESubgroup (5-fold CV) and the Feature-Only variant
sbatch scripts/train_beta_vae_subgroup.sh 49
sbatch scripts/train_beta_vae_feature_only.sh 49

# Everything with job dependencies (training, layers, benchmark)
bash scripts/submit_pipeline.sh --release 49 [--skip train,feature_only_train,layers,benchmark] [--retrain] [--dry-run]
```

`submit_pipeline.sh` refuses to retrain a model that already has checkpoints
unless `--retrain` is given. For GENCODE v47 use release `47`. Without SLURM,
run the commands inside the scripts directly. The entry points are
`src/main_subgroup.py` and `src/main_feature_only.py` (`--config`, `--device`).

### Tests

```bash
python -m pytest -q          # CPU only, no data files or checkpoints
```

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

Stages (1.1–1.4 including 1.2b input ablation, 2.1, 3.1–3.5) write to `<experiment_dir>/interpretability/`
and are resumable via `.done` markers keyed on the command signature.
`--force` reruns everything, `--only <ids>` selects stages, `--dry-run` prints
the plan. See `analysis/pattern/README.md` for script-by-script usage.

---

## Benchmark comparison

On SLURM, `scripts/submit_pipeline.sh` submits the benchmark stage
(`scripts/benchmark_job.sh`). Direct invocation:

```bash
bash analysis/benchmark/run_full_benchmark.sh \
    --release v49 \
    --base_dir gencode_v49_experiments \
    --lnc_fasta data/split_gencode_49/lnc_test.fa \
    --pc_fasta  data/split_gencode_49/pc_test.fa \
    --blnc_predictions gencode_v49_experiments/beta_vae_subgroup_base_g49/evaluation_csvs/test_predictions.csv
```

Compares against CPAT, CPC2, LncDC, RNAsamba, Orthrus (4-track), and
lncRNA-BERT, reporting accuracy/precision/recall/F1 with 95% gene-clustered
bootstrap confidence intervals (10,000 resamples) and AUC with 95% DeLong CI
per method. Results are given on the full test set and on the common subset
scored by every method. See `analysis/benchmark/README.md` for the full pipeline.

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
bash scripts/run_lncrnabert.sh      # usage in the script header
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