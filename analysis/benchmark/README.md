# Benchmark Comparison Pipeline

Cross-method classification benchmark for BetaVAESubgroup, evaluated on
GENCODE v47 and v49 held-out test sets.

## Methods compared

| Method | Type |
|--------|------|
| **β-LNC (BetaVAESubgroup)** | Sequence + 20 feature tokens, cross-modal attention |
| β-LNC Feature-Only | Feature tokens only, no sequence encoder |
| lncRNA-BERT (3-mer) | Sequence foundation model |
| CPAT | ORF/hexamer/Fickett scalar features + logistic regression |
| CPC2 | ORF/Fickett/isoelectric point features |
| LncDC | Sequence + structural features, ML classifier |
| RNAsamba | CNN on sequence, ORF-aware |
| Orthrus (4-track) | Mamba-based sequence foundation model, linear probe |

---

## Pipeline overview

```
[1] run_benchmark_tools.sh
    • Merges lncRNA + mRNA test FASTAs
    • Runs CPAT, CPC2, LncDC, RNAsamba (each in its own conda env / container)
    • Runs Orthrus 4-track linear probing (evaluate_orthrus.py)
    • Runs lncRNA-BERT inference (convert_lncrnabert_predictions.py)
    ↓
[2] compare_models.py
    • Loads all method predictions, normalises transcript IDs
      (_norm_id: strips pipe-suffix and version suffix from every loader)
    • Computes accuracy / precision / recall / F1 per method (point estimates)
    • Identifies hard cases per method (misclassified or confidence < threshold)
    • Writes benchmark_table.csv, predictions_merged.csv, hard_case_summary.csv
    ↓
[2.5] compute_benchmark_ci.py
    • 95% bootstrap CI on F1 (10,000 resamples) and 95% DeLong CI on AUC,
      per method — the point estimates from step 2 with confidence intervals
    • Writes benchmark_table_with_ci.csv
```

Stage 1 is a full benchmark tool run and can take a long time (each
tool's own environment/dependencies apply — see `run_benchmark_tools.sh`
header for per-tool notes on RNAsamba's TF1/Keras2.1 pinning via Singularity,
and Orthrus's conda env activation). Stages 2–3 are fast once Stage 1's
outputs exist (stage 2.5 runs a 10,000-resample bootstrap per method).

On SLURM, `scripts/submit_pipeline.sh` submits the benchmark stage
(`scripts/benchmark_job.sh`).

`run_full_benchmark.sh` orchestrates all stages with per-method
enable/disable toggles (`--use_cpat`, `--use_orthrus`, etc.) and derives all
standard paths from `--release` / `--base_dir`.

---

## Usage

### Full pipeline (all methods, one release)

```bash
bash analysis/benchmark/run_full_benchmark.sh \
    --release v49 \
    --base_dir gencode_v49_experiments \
    --lnc_fasta       data/split_gencode_49/lnc_test.fa \
    --pc_fasta        data/split_gencode_49/pc_test.fa \
    --train_lnc_fasta data/split_gencode_49/lnc_trainval.fa \
    --train_pc_fasta  data/split_gencode_49/pc_trainval.fa \
    --blnc_predictions gencode_v49_experiments/beta_vae_subgroup_base_g49/evaluation_csvs/test_predictions.csv \
    --feature_only_csv gencode_v49_experiments/feature_only_g49/evaluation_csvs/test_predictions.csv \
    --lncrnabert_csv  gencode_v49_experiments/benchmark_tools/lncrnabert_test_predictions.csv \
    --orthrus_env     $HOME/.conda/envs/orthrus \
    --rnasamba_sif    rnasamba.sif \
    --error_only \
    --min_set_size 5
```

### Individual tool run (if you only need to add/refresh one method)

```bash
bash analysis/benchmark/run_benchmark_tools.sh \
    --lnc_fasta data/split_gencode_49/lnc_test.fa \
    --pc_fasta  data/split_gencode_49/pc_test.fa \
    --output_dir gencode_v49_experiments/benchmark_tools \
    --release v49 \
    --skip_cpat --skip_cpc2 --skip_lncdc   # only re-run Orthrus/RNAsamba, e.g.
```

### Bootstrap CI only (e.g. re-running after adding a method)

```bash
python analysis/benchmark/compute_benchmark_ci.py \
    --blnc_csv        gencode_v49_experiments/beta_vae_subgroup_base_g49/evaluation_csvs/test_predictions.csv \
    --feature_only_csv gencode_v49_experiments/feature_only_g49/evaluation_csvs/test_predictions.csv \
    --cpat_csv        gencode_v49_experiments/benchmark_tools/cpat/predictions_with_cpat.csv \
    --cpc2_tsv        gencode_v49_experiments/benchmark_tools/cpc2_results.tsv.txt \
    --lncdc_csv       gencode_v49_experiments/benchmark_tools/lncdc_results.csv \
    --rnasamba_tsv    gencode_v49_experiments/benchmark_tools/rnasamba_results.tsv \
    --orthrus_csv     gencode_v49_experiments/benchmark_tools/orthrus/orthrus_test_predictions.csv \
    --lncrnabert_csv  gencode_v49_experiments/benchmark_tools/lncrnabert_test_predictions.csv \
    --n_bootstrap     10000 \
    --output_dir      gencode_v49_experiments/benchmark_comparison
```

`--blnc_csv` is the only required argument and also serves as the
canonical true-label source; every other `--*_csv`/`--*_tsv` is optional —
only the methods you provide files for are evaluated.

---

## Output reference

```
benchmark_tools/
  cpat/predictions_with_cpat.csv
  cpc2_results.tsv.txt
  lncdc_results.csv
  orthrus/orthrus_test_predictions.csv
  rnasamba_results.tsv
  lncrnabert_test_predictions.csv

benchmark_comparison/
  benchmark_table.csv              method, n_samples, accuracy, precision, recall, f1_macro, ...
  benchmark_table_with_ci.csv      method, n_samples, accuracy, precision, recall,
                                   f1, f1_ci_low, f1_ci_high, auc, auc_ci_low, auc_ci_high
  predictions_merged.csv           per-transcript predictions across all methods
  hard_case_summary.csv
```