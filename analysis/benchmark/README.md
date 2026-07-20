# Benchmark Comparison Pipeline

Cross-method classification benchmark and hard-for-all analysis for
BetaVAESubgroup, evaluated on GENCODE v47 and v49 held-out test sets.

## Methods compared

| Method | Type |
|--------|------|
| **β-LNC (BetaVAESubgroup)** | Sequence + 20-token locus features, cross-modal attention |
| β-LNC Feature-Only | Locus features only, no sequence encoder |
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
    • Computes accuracy / precision / recall / F1 per method
    • Identifies hard cases per method (misclassified or confidence < threshold)
    • Writes benchmark_table.csv, predictions_merged.csv, hard_case_summary.csv
    ↓
[3] upset_benchmark.py
    • UpSet plot of hard-case intersections across all methods
    • Reveals whether hard cases are shared (biological ambiguity) or
      method-specific (model failure)
    ↓
[4] hard_for_all/run_hard_for_all.sh
    • Identifies the intersection: transcripts hard for every method
    • extract_tsl_gtf.py — transcript support level annotation for hard-for-all set
    • cpat_scores.py — CPAT sequence features for hard-for-all vs background
    • analyze_hard_for_all.py — statistical comparison (Mann-Whitney U),
      writes cpat_feature_summary_combined.csv
    • plot_hard_for_all_cpat.py — publication figure
```

Stage 1 is a full benchmark tool run and can take a long time (each
tool's own environment/dependencies apply — see `run_benchmark_tools.sh`
header for per-tool notes on RNAsamba's TF1/Keras2.1 pinning via Singularity,
and Orthrus's conda env activation). Stages 2–4 are fast once Stage 1's
outputs exist.

`run_full_benchmark.sh` orchestrates all three stages with per-method
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
    --blnc_predictions gencode_v49_experiments/beta_vae_subgroup_base_g49_new/evaluation_csvs/test_predictions.csv \
    --feature_only_csv gencode_v49_experiments/feature_only_g49_new/evaluation_csvs/test_predictions.csv \
    --lncrnabert_csv  gencode_v49_experiments/benchmark_tools/lncrnabert_test_predictions.csv \
    --orthrus_env     {$HOME}.conda/envs/orthrus \
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

### Hard-for-all analysis

```bash
bash analysis/benchmark/hard_for_all/run_hard_for_all.sh \
    --predictions_merged gencode_v49_experiments/benchmark_comparison/predictions_merged.csv \
    --release v49 \
    --output_dir gencode_v49_experiments/benchmark_comparison/hard_for_all
```

### CPAT hard-for-all figure

```bash
# Both releases, portrait (features as rows, releases as columns)
python analysis/benchmark/hard_for_all/plot_hard_for_all_cpat.py \
    --summary_csv    gencode_v47_experiments/benchmark_comparison/cpat_feature_summary_combined.csv \
    --cpat_v47       gencode_v47_experiments/benchmark_comparison/hard_for_all/hard_for_all_cpat.csv \
    --cpat_v49       gencode_v49_experiments/benchmark_comparison/hard_for_all/hard_for_all_cpat.csv \
    --cpat_full_v47  gencode_v47_experiments/benchmark_tools/cpat/predictions_with_cpat.csv \
    --cpat_full_v49  gencode_v49_experiments/benchmark_tools/cpat/predictions_with_cpat.csv \
    --output_dir     figures/

# Single release, landscape (class as rows, feature as columns) — for wide poster/slide layouts
python analysis/benchmark/hard_for_all/plot_hard_for_all_cpat.py \
    --summary_csv    gencode_v49_experiments/benchmark_comparison/cpat_feature_summary_combined.csv \
    --cpat_v49       gencode_v49_experiments/benchmark_comparison/hard_for_all/hard_for_all_cpat.csv \
    --cpat_full_v49  gencode_v49_experiments/benchmark_tools/cpat/predictions_with_cpat.csv \
    --releases       v49 \
    --orientation    landscape \
    --output_dir     figures/
```

`--orientation landscape` requires exactly one release. Figure aspect ratio
is fixed to match the companion within-class covariance heatmap figure
(2.333:1) so the two sit cleanly in the same layout without stretching.

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
  predictions_merged.csv           per-transcript predictions across all methods
  hard_case_summary.csv
  upset_hard_cases[_error_only].png
  upset_blnc_advantage[_error_only].png

benchmark_comparison/hard_for_all/
  hard_for_all_cpat.csv            per-transcript CPAT features for the all-methods-hard set
  cpat_feature_summary_combined.csv
  tsl_annotation.csv

figures/
  cpat_hard_for_all_figure.png              (portrait, all releases)
  cpat_hard_for_all_figure_{release}_landscape.png
```