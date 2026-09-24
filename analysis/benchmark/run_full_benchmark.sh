#!/bin/bash
# =============================================================================
# run_full_benchmark.sh
#
# End-to-end orchestration: run_benchmark_tools.sh → compare_models.py →
# plot_upset_benchmark.py, for one GENCODE release.
#
# Each method is toggled with a true/false flag. The script derives all
# standard paths from --release and --base_dir, and only passes arguments
# for enabled methods to each downstream stage.
#
# Usage
# -----
# bash analysis/run_full_benchmark.sh \
#     --release v49 \
#     --base_dir gencode_v49_experiments \
#     --lnc_fasta       data/split_gencode_49/lnc_test.fa \
#     --pc_fasta        data/split_gencode_49/pc_test.fa \
#     --train_lnc_fasta data/split_gencode_49/lnc_trainval.fa \
#     --train_pc_fasta  data/split_gencode_49/pc_trainval.fa \
#     --blnc_predictions gencode_v49_experiments/beta_vae_subgroup_base_g49/evaluation_csvs/test_predictions.csv \
#     [--feature_only_csv ...] \
#     [--lncrnabert_csv  ...] \
#     [--use_blnc          true]  \
#     [--use_feature_only  false] \
#     [--use_cpat          true]  \
#     [--use_cpc2          true]  \
#     [--use_lncdc         true]  \
#     [--use_orthrus       true]  \
#     [--use_rnasamba      true]  \
#     [--use_lncrnabert    true]  \
#     [--error_only] \
#     [--min_set_size 5] \
#     [--orthrus_env  {$HOME}.conda/envs/orthrus] \
#     [--rnasamba_sif rnasamba.sif] \
#     [--max_length   10000]
#
# Notes
# -----
# - β-LNC is the reference method; --use_blnc=false is allowed (no reference
#   metrics computed) but --blnc_predictions is still required since
#   compare_models.py needs it for ground-truth labels.
# - Methods with use_*=false are skipped entirely in run_benchmark_tools.sh
#   (via --skip_*) AND excluded from the compare_models.py / upset arguments,
#   even if their output files already exist from a previous run.
# =============================================================================

set -eo pipefail
cd /mnt/cbib/LNClassifier/beta_vae_lnclassifier

SCRIPT_DIR="analysis/benchmark"

# ── Defaults ──────────────────────────────────────────────────────────────────
RELEASE="v49"
BASE_DIR="gencode_v49_experiments"
LNC_FASTA="data/split_gencode_49/lnc_test.fa"
PC_FASTA="data/split_gencode_49/pc_test.fa"
TRAIN_LNC_FASTA="data/split_gencode_49/lnc_trainval.fa"
TRAIN_PC_FASTA="data/split_gencode_49/pc_trainval.fa"
BLNC_PREDICTIONS="gencode_v49_experiments/beta_vae_subgroup_base_g49/evaluation_csvs/test_predictions.csv"
FEATURE_ONLY_CSV="gencode_v49_experiments/beta_vae_feature_only_g49/evaluation_csvs/test_predictions.csv"
LNCRNABERT_CSV="gencode_v49_experiments/benchmark_tools/lncrnabert_test_predictions.csv"

# Per-method toggles (true/false)
USE_BLNC=true
USE_FEATURE_ONLY=false
USE_CPAT=true
USE_CPC2=true
USE_LNCDC=true
USE_ORTHRUS=true
USE_RNASAMBA=true
USE_LNCRNABERT=true

# Tool paths / options (passed through to run_benchmark_tools.sh)
CPAT_HEX="cpat/data/Human_Hexamer.tsv"
CPAT_MODEL="cpat/data/Human_logitModel.RData"
CPC2_DIR="cpc2/CPC2_standalone-1.0.1"
LNCDC_DIR="LncDC-1.3.6"
R_MODULE="R/4.1.0"
ORTHRUS_MODEL=""
ORTHRUS_ENV="{$HOME}.conda/envs/orthrus"
RNASAMBA_WEIGHTS="rnasamba/full_length_weights.hdf5"
RNASAMBA_SIF="rnasamba.sif"
THREADS=8
MAX_LENGTH=15000

# compare_models / upset options
ERROR_ONLY=false
MIN_SET_SIZE=6
CPAT_THRESHOLD=""
CPC2_THRESHOLD=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --release)            RELEASE="$2";            shift 2 ;;
        --base_dir)           BASE_DIR="$2";           shift 2 ;;
        --lnc_fasta)          LNC_FASTA="$2";          shift 2 ;;
        --pc_fasta)           PC_FASTA="$2";           shift 2 ;;
        --train_lnc_fasta)    TRAIN_LNC_FASTA="$2";    shift 2 ;;
        --train_pc_fasta)     TRAIN_PC_FASTA="$2";     shift 2 ;;
        --blnc_predictions)   BLNC_PREDICTIONS="$2";   shift 2 ;;
        --feature_only_csv)   FEATURE_ONLY_CSV="$2";   shift 2 ;;
        --lncrnabert_csv)     LNCRNABERT_CSV="$2";     shift 2 ;;

        --use_blnc)           USE_BLNC="$2";           shift 2 ;;
        --use_feature_only)   USE_FEATURE_ONLY="$2";   shift 2 ;;
        --use_cpat)           USE_CPAT="$2";           shift 2 ;;
        --use_cpc2)           USE_CPC2="$2";           shift 2 ;;
        --use_lncdc)          USE_LNCDC="$2";          shift 2 ;;
        --use_orthrus)        USE_ORTHRUS="$2";        shift 2 ;;
        --use_rnasamba)       USE_RNASAMBA="$2";       shift 2 ;;
        --use_lncrnabert)     USE_LNCRNABERT="$2";     shift 2 ;;

        --cpat_hex)           CPAT_HEX="$2";           shift 2 ;;
        --cpat_model)         CPAT_MODEL="$2";         shift 2 ;;
        --cpc2_dir)           CPC2_DIR="$2";           shift 2 ;;
        --lncdc_dir)          LNCDC_DIR="$2";          shift 2 ;;
        --r_module)           R_MODULE="$2";           shift 2 ;;
        --orthrus_model)      ORTHRUS_MODEL="$2";      shift 2 ;;
        --orthrus_env)        ORTHRUS_ENV="$2";        shift 2 ;;
        --rnasamba_weights)   RNASAMBA_WEIGHTS="$2";   shift 2 ;;
        --rnasamba_sif)       RNASAMBA_SIF="$2";       shift 2 ;;
        --threads)            THREADS="$2";            shift 2 ;;
        --max_length)         MAX_LENGTH="$2";         shift 2 ;;

        --error_only)         ERROR_ONLY=true;         shift 1 ;;
        --min_set_size)       MIN_SET_SIZE="$2";       shift 2 ;;
        --cpat_threshold)     CPAT_THRESHOLD="$2";     shift 2 ;;
        --cpc2_threshold)     CPC2_THRESHOLD="$2";     shift 2 ;;

        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Validate required args ────────────────────────────────────────────────────
for var in RELEASE BASE_DIR LNC_FASTA PC_FASTA BLNC_PREDICTIONS; do
    if [[ -z "${!var}" ]]; then
        echo "ERROR: --${var,,} is required"
        exit 1
    fi
done

# ── Derived paths ─────────────────────────────────────────────────────────────
TOOLS_DIR="${BASE_DIR}/benchmark_tools"
COMPARISON_DIR="${BASE_DIR}/benchmark_comparison"

CPAT_HARD_CSV="${TOOLS_DIR}/cpat/predictions_with_cpat.csv"
CPC2_RESULT="${TOOLS_DIR}/cpc2_results.tsv.txt"
LNCDC_RESULT="${TOOLS_DIR}/lncdc_results.csv"
ORTHRUS_RESULT="${TOOLS_DIR}/orthrus/orthrus_test_predictions.csv"
RNASAMBA_RESULT="${TOOLS_DIR}/rnasamba_results.tsv"

mkdir -p "$COMPARISON_DIR"

LOG_FILE="${COMPARISON_DIR}/full_benchmark_${RELEASE}_$(date +%Y%m%d_%H%M%S).log"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
log_section() {
    echo "" | tee -a "$LOG_FILE"
    echo "$(printf '=%.0s' {1..70})" | tee -a "$LOG_FILE"
    echo "  $*" | tee -a "$LOG_FILE"
    echo "$(printf '=%.0s' {1..70})" | tee -a "$LOG_FILE"
}

log_section "Full benchmark pipeline — GENCODE ${RELEASE}"
log "Base dir     : ${BASE_DIR}"
log "Tools dir    : ${TOOLS_DIR}"
log "Comparison   : ${COMPARISON_DIR}"
log ""
log "Methods enabled:"
for pair in "blnc:$USE_BLNC" "feature_only:$USE_FEATURE_ONLY" "cpat:$USE_CPAT" \
            "cpc2:$USE_CPC2" "lncdc:$USE_LNCDC" "orthrus:$USE_ORTHRUS" \
            "rnasamba:$USE_RNASAMBA" "lncrnabert:$USE_LNCRNABERT"; do
    name="${pair%%:*}"; val="${pair##*:}"
    log "  $(printf '%-14s' "$name"): ${val}"
done

# =============================================================================
# Stage 1 — run_benchmark_tools.sh
# =============================================================================
log_section "Stage 1/3 — run_benchmark_tools.sh"

TOOLS_ARGS=(
    --lnc_fasta   "$LNC_FASTA"
    --pc_fasta    "$PC_FASTA"
    --output_dir  "$TOOLS_DIR"
    --release     "$RELEASE"
    --threads     "$THREADS"
    --max_length  "$MAX_LENGTH"
)

[[ -n "$TRAIN_LNC_FASTA" ]] && TOOLS_ARGS+=(--train_lnc_fasta "$TRAIN_LNC_FASTA")
[[ -n "$TRAIN_PC_FASTA"  ]] && TOOLS_ARGS+=(--train_pc_fasta  "$TRAIN_PC_FASTA")
[[ -n "$BLNC_PREDICTIONS" ]] && TOOLS_ARGS+=(--blnc_predictions "$BLNC_PREDICTIONS")
[[ -n "$FEATURE_ONLY_CSV" ]] && TOOLS_ARGS+=(--feature_only_csv "$FEATURE_ONLY_CSV")

[[ -n "$CPAT_HEX"         ]] && TOOLS_ARGS+=(--cpat_hex "$CPAT_HEX")
[[ -n "$CPAT_MODEL"       ]] && TOOLS_ARGS+=(--cpat_model "$CPAT_MODEL")
[[ -n "$CPC2_DIR"         ]] && TOOLS_ARGS+=(--cpc2_dir "$CPC2_DIR")
[[ -n "$LNCDC_DIR"        ]] && TOOLS_ARGS+=(--lncdc_dir "$LNCDC_DIR")
[[ -n "$R_MODULE"         ]] && TOOLS_ARGS+=(--r_module "$R_MODULE")
[[ -n "$ORTHRUS_MODEL"    ]] && TOOLS_ARGS+=(--orthrus_model "$ORTHRUS_MODEL")
[[ -n "$ORTHRUS_ENV"      ]] && TOOLS_ARGS+=(--orthrus_env "$ORTHRUS_ENV")
[[ -n "$RNASAMBA_WEIGHTS" ]] && TOOLS_ARGS+=(--rnasamba_weights "$RNASAMBA_WEIGHTS")
[[ -n "$RNASAMBA_SIF"     ]] && TOOLS_ARGS+=(--rnasamba_sif "$RNASAMBA_SIF")

# Skip flags for disabled methods
[[ "$USE_CPAT"     != "true" ]] && TOOLS_ARGS+=(--skip_cpat)
[[ "$USE_CPC2"     != "true" ]] && TOOLS_ARGS+=(--skip_cpc2)
[[ "$USE_LNCDC"    != "true" ]] && TOOLS_ARGS+=(--skip_lncdc)
[[ "$USE_ORTHRUS"  != "true" ]] && TOOLS_ARGS+=(--skip_orthrus)
[[ "$USE_RNASAMBA" != "true" ]] && TOOLS_ARGS+=(--skip_rnasamba)

log "Command: bash ${SCRIPT_DIR}/run_benchmark_tools.sh ${TOOLS_ARGS[*]}"
bash "${SCRIPT_DIR}/run_benchmark_tools.sh" "${TOOLS_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

# =============================================================================
# Stage 2 — compare_models.py
# =============================================================================
log_section "Stage 2/3 — compare_models.py"

COMPARE_ARGS=(
    --blnc_csv   "$BLNC_PREDICTIONS"
    --output_dir "$COMPARISON_DIR"
)

if [[ "$USE_FEATURE_ONLY" == "true" ]]; then
    if [[ -n "$FEATURE_ONLY_CSV" ]]; then
        COMPARE_ARGS+=(--feature_only_csv "$FEATURE_ONLY_CSV")
    else
        log "  WARNING: use_feature_only=true but --feature_only_csv not provided — skipping"
    fi
fi

if [[ "$USE_CPAT" == "true" ]]; then
    if [[ -f "$CPAT_HARD_CSV" ]]; then
        COMPARE_ARGS+=(--cpat_csv "$CPAT_HARD_CSV")
    else
        log "  WARNING: use_cpat=true but ${CPAT_HARD_CSV} not found — skipping CPAT"
    fi
fi

if [[ "$USE_CPC2" == "true" ]]; then
    if [[ -f "$CPC2_RESULT" ]]; then
        COMPARE_ARGS+=(--cpc2_tsv "$CPC2_RESULT")
    else
        log "  WARNING: use_cpc2=true but ${CPC2_RESULT} not found — skipping CPC2"
    fi
fi

if [[ "$USE_LNCDC" == "true" ]]; then
    if [[ -f "$LNCDC_RESULT" ]]; then
        COMPARE_ARGS+=(--lncdc_tsv "$LNCDC_RESULT")
    else
        log "  WARNING: use_lncdc=true but ${LNCDC_RESULT} not found — skipping LncDC"
    fi
fi

if [[ "$USE_ORTHRUS" == "true" ]]; then
    if [[ -f "$ORTHRUS_RESULT" ]]; then
        COMPARE_ARGS+=(--orthrus_csv "$ORTHRUS_RESULT")
    else
        log "  WARNING: use_orthrus=true but ${ORTHRUS_RESULT} not found — skipping Orthrus"
    fi
fi

if [[ "$USE_RNASAMBA" == "true" ]]; then
    if [[ -f "$RNASAMBA_RESULT" ]]; then
        COMPARE_ARGS+=(--rnasamba_tsv "$RNASAMBA_RESULT")
    else
        log "  WARNING: use_rnasamba=true but ${RNASAMBA_RESULT} not found — skipping RNAsamba"
    fi
fi

if [[ "$USE_LNCRNABERT" == "true" ]]; then
    if [[ -n "$LNCRNABERT_CSV" ]] && [[ -f "$LNCRNABERT_CSV" ]]; then
        COMPARE_ARGS+=(--lncrnabert_csv "$LNCRNABERT_CSV")
    else
        log "  WARNING: use_lncrnabert=true but --lncrnabert_csv missing or not found — skipping"
    fi
fi

[[ -n "$CPAT_THRESHOLD" ]] && COMPARE_ARGS+=(--cpat_threshold "$CPAT_THRESHOLD")
[[ -n "$CPC2_THRESHOLD" ]] && COMPARE_ARGS+=(--cpc2_threshold "$CPC2_THRESHOLD")

log "Command: python ${SCRIPT_DIR}/compare_models.py ${COMPARE_ARGS[*]}"
python "${SCRIPT_DIR}/compare_models.py" "${COMPARE_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

PREDICTIONS_MERGED="${COMPARISON_DIR}/predictions_merged.csv"
if [[ ! -f "$PREDICTIONS_MERGED" ]]; then
    log "ERROR: ${PREDICTIONS_MERGED} not produced — aborting before upset plots"
    exit 1
fi

# =============================================================================
# Stage 2.5 — compute_benchmark_ci.py
# =============================================================================
log_section "Stage 2.5/3 — compute_benchmark_ci.py"

log "Command: python ${SCRIPT_DIR}/compute_benchmark_ci.py ${COMPARE_ARGS[*]} --n_bootstrap 10000"
python "${SCRIPT_DIR}/compute_benchmark_ci.py" "${COMPARE_ARGS[@]}" --n_bootstrap 10000 2>&1 | tee -a "$LOG_FILE"

# =============================================================================
# Stage 3 — upset_benchmark.py
# =============================================================================
log_section "Stage 3/3 — upset_benchmark.py"

UPSET_ARGS=(
    --predictions_csv "$PREDICTIONS_MERGED"
    --output_dir      "$COMPARISON_DIR"
    --min_set_size    "$MIN_SET_SIZE"
)

if $ERROR_ONLY; then
    UPSET_ARGS+=(--error_only --blnc_csv "$BLNC_PREDICTIONS")
fi

log "Command: python ${SCRIPT_DIR}/upset_benchmark.py ${UPSET_ARGS[*]}"
python "${SCRIPT_DIR}/upset_benchmark.py" "${UPSET_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

# =============================================================================
# Summary
# =============================================================================
log_section "Done — GENCODE ${RELEASE}"
log "Benchmark table     : ${COMPARISON_DIR}/benchmark_table.csv"
log "Merged predictions  : ${PREDICTIONS_MERGED}"
log "Hard case summary   : ${COMPARISON_DIR}/hard_case_summary*.csv"
SUFFIX=""
$ERROR_ONLY && SUFFIX="_error_only"
log "UpSet plots         : ${COMPARISON_DIR}/upset_hard_cases${SUFFIX}.png"
log "                      ${COMPARISON_DIR}/upset_blnc_advantage${SUFFIX}.png"
log ""
log "Full log: ${LOG_FILE}"