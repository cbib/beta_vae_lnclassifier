#!/bin/bash
#SBATCH --job-name=bench
#SBATCH --output=logs/bench_%j.log
#SBATCH --error=logs/bench_%j.err
#SBATCH --gres=gpu:nvidia_h100_nvl_4g.47gb:1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#
# Gene-grouped benchmark for one release (tools + CIs + ensemble_vs_models + UpSet).
#
#   sbatch scripts/benchmark_job.sh --release 47 \
#       --exp_dir gencode_v47_experiments_genegrouped/<experiment> --evaluate
#   sbatch scripts/benchmark_job.sh --release 49 \
#       --exp_dir gencode_v49_experiments_genegrouped/beta_vae_subgroup_base_g49_genegrouped
#
# Options
#   --release N        47 | 49
#   --exp_dir DIR      trained experiment folder (holds models/ and evaluation_csvs/)
#   --config FILE      training config (default configs/beta_vae_subgroup_g<N>_genegrouped.json)
#   --split_dir DIR    default data/split_gencode_<N>_genegrouped
#   --base_dir DIR     benchmark output root (default gencode_v<N>_experiments_genegrouped);
#                      must be where finished tool outputs live, else they are rerun
#   --evaluate         run evaluate_cv_fold_subgroup.py first if test_predictions.csv /
#                      test_fold_probs.csv are missing (also for the feature-only model)
#   --feature_exp_dir DIR   trained feature-only experiment folder -> adds it to the
#                      benchmark (--use_feature_only true)
#   --feature_config FILE   its training config (needed with --evaluate)
# Finished tool steps are skipped on rerun, so resubmitting is safe.

set -euo pipefail
echo "benchmark_job args: $*"
REL=""; EXP=""; CFG=""; SPLIT=""; BASE=""; EVAL=0; FEXP=""; FCFG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --release) REL="$2"; shift 2;; --exp_dir) EXP="${2%/}"; shift 2;;
    --config) CFG="$2"; shift 2;; --split_dir) SPLIT="${2%/}"; shift 2;;
    --base_dir) BASE="${2%/}"; shift 2;; --evaluate) EVAL=1; shift;;
    --feature_exp_dir) FEXP="${2%/}"; shift 2;; --feature_config) FCFG="$2"; shift 2;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
[[ -n "$REL" && -n "$EXP" ]] || { echo "need --release and --exp_dir" >&2; exit 2; }
CFG="${CFG:-configs/beta_vae_subgroup_g${REL}_genegrouped.json}"
SPLIT="${SPLIT:-data/split_gencode_${REL}_genegrouped}"
BASE="${BASE:-gencode_v${REL}_experiments_genegrouped}"
EVAL_SCRIPT=analysis/post_training_pipeline/scripts/evaluate_cv_fold_subgroup.py   # adjust if elsewhere

module load nvidia/cuda/12.1
eval "$(conda shell.bash hook)"
conda activate beta_lncrna
cd ${DATA_ROOT}
mkdir -p logs

fail() { echo "PREFLIGHT FAILED: $*" >&2; exit 1; }
for f in lnc_test pc_test lnc_trainval pc_trainval; do
  [[ -f "$SPLIT/$f.fa" ]] || fail "missing $SPLIT/$f.fa"
done
[[ -d "$EXP" ]] || fail "experiment dir not found: $EXP"
EV="$EXP/evaluation_csvs"
PRED="$EV/test_predictions.csv"; FOLD="$EV/test_fold_probs.csv"

if [[ ! -f "$PRED" || ! -f "$FOLD" ]]; then
  (( EVAL )) || fail "$PRED or $FOLD missing (use --evaluate)"
  [[ -f "$CFG" ]] || fail "config not found: $CFG"
  [[ ! -f "$PRED" ]] || fail "$PRED exists but $FOLD is missing; refusing to overwrite $EV. Rerun into a new dir and copy test_fold_probs.csv"
  echo "[$(date +%T)] evaluating $EXP -> $EV"
  PYTHONPATH=".:$(dirname "$EVAL_SCRIPT")" python "$EVAL_SCRIPT" \
      --config "$CFG" --experiment_dir "$EXP" --output_dir "$EV" \
      --lnc_fasta "$SPLIT/lnc_test.fa" --pc_fasta "$SPLIT/pc_test.fa" --device cuda:0
fi

FEAT_ARGS=(--use_feature_only false)
if [[ -n "$FEXP" ]]; then
  [[ -d "$FEXP" ]] || fail "feature-only experiment dir not found: $FEXP"
  FPRED="$FEXP/evaluation_csvs/test_predictions.csv"
  if [[ ! -f "$FPRED" ]]; then
    (( EVAL )) || fail "$FPRED missing (use --evaluate)"
    [[ -n "$FCFG" && -f "$FCFG" ]] || fail "--feature_config required and must exist"
    echo "[$(date +%T)] evaluating feature-only $FEXP"
    PYTHONPATH=".:$(dirname "$EVAL_SCRIPT")" python "$EVAL_SCRIPT" \
        --config "$FCFG" --experiment_dir "$FEXP" --output_dir "$FEXP/evaluation_csvs" \
        --lnc_fasta "$SPLIT/lnc_test.fa" --pc_fasta "$SPLIT/pc_test.fa" --device cuda:0
  fi
  FEAT_ARGS=(--use_feature_only true --feature_only_csv "$FPRED")
fi

echo "[$(date +%T)] benchmark g$REL base=$BASE split=$SPLIT"
bash analysis/benchmark/run_full_benchmark.sh --release "v$REL" --base_dir "$BASE" \
    --lnc_fasta "$SPLIT/lnc_test.fa" --pc_fasta "$SPLIT/pc_test.fa" \
    --train_lnc_fasta "$SPLIT/lnc_trainval.fa" --train_pc_fasta "$SPLIT/pc_trainval.fa" \
    --blnc_predictions "$PRED" \
    --require_gene_disjoint true "${FEAT_ARGS[@]}" --use_lncrnabert true \
    --common_subset true --hard_def "${HARD_DEF:-error}"

# exit non-zero if a method is missing so dependent jobs / the user notice
T="$BASE/benchmark_tools"; C="$BASE/benchmark_comparison"; MISS=0
for f in "$T/cpat/predictions_with_cpat.csv" "$T/cpc2_results.tsv.txt" "$T/lncdc_results.csv" \
         "$T/rnasamba_results.tsv" "$T/orthrus_test_predictions.csv" \
         "$T/lncrnabert_test_predictions.csv" "$C/benchmark_table_with_ci.csv" \
         "$C/ensemble_vs_models_f1.csv" "$C/benchmark_provenance.json"; do
  [[ -s "$f" ]] || { echo "MISSING OUTPUT: $f" >&2; MISS=1; }
done
exit $MISS