#!/bin/bash
# =============================================================================
# run_posttrain.sh
#
# Post-training pipeline for β-LNC — runs after all SLURM training jobs
# complete for both GENCODE releases.
#
# Steps:
#   1. evaluate_cv_subgroup.py  — test predictions + CV metrics (per experiment)
#   2. interpretability         — ablation, latent probing, residual ablation
#   3. run_full_benchmark.sh    — 7-method benchmark comparison + upset plots
#   4. run_hard_for_all.sh      — biotype enrichment, CPAT features, TSL, figure
#
# Usage
# -----
# bash analysis/run_posttrain.sh \
#     --blnc_exp_v47    gencode_v47_experiments/beta_vae_subgroup_base_g47 \
#     --blnc_exp_v49    gencode_v49_experiments/beta_vae_subgroup_base_g49 \
#     --feat_exp_v47    gencode_v47_experiments/beta_vae_feature_only_g47 \
#     --feat_exp_v49    gencode_v49_experiments/beta_vae_feature_only_g49 \
#     --config_v47      configs/beta_vae_subgroup_base_g47.json \
#     --config_v49      configs/beta_vae_subgroup_base_g49.json \
#     --lnc_fasta_v47   data/split_gencode_47/lnc_test.fa \
#     --pc_fasta_v47    data/split_gencode_47/pc_test.fa \
#     --lnc_fasta_v49   data/split_gencode_49/lnc_test.fa \
#     --pc_fasta_v49    data/split_gencode_49/pc_test.fa \
#     --train_lnc_v47   data/split_gencode_47/lnc_trainval.fa \
#     --train_pc_v47    data/split_gencode_47/pc_trainval.fa \
#     --train_lnc_v49   data/split_gencode_49/lnc_trainval.fa \
#     --train_pc_v49    data/split_gencode_49/pc_trainval.fa \
#     --biotype_csv_v47 data/dataset_biotypes/g47_dataset_biotypes_cdhit.csv \
#     --biotype_csv_v49 data/dataset_biotypes/g49_dataset_biotypes_cdhit.csv \
#     --cpat_csv_v47    gencode_v47_experiments/benchmark_tools/cpat/predictions_with_cpat.csv \
#     --cpat_csv_v49    gencode_v49_experiments/benchmark_tools/cpat/predictions_with_cpat.csv \
#     --lncrnabert_csv_v47 gencode_v47_experiments/benchmark_tools/lncrnabert_test_predictions.csv \
#     --lncrnabert_csv_v49 gencode_v49_experiments/benchmark_tools/lncrnabert_test_predictions.csv \
#     --gtf_v47         /mnt/cbib/LNClassifier/paper/nonb-pipeline.old/resources/gencode.v47.annotation.gtf \
#     --orthrus_env     {$HOME}.conda/envs/orthrus \
#     --rnasamba_sif    rnasamba.sif \
#     --figures_dir     figures/ \
#     [--n_folds        5] \
#     [--device         cuda:0] \
#     [--min_set_size   5]
# =============================================================================

set -uo pipefail
cd /mnt/cbib/LNClassifier/beta_vae_lnclassifier

SCRIPT_DIR="analysis/benchmark"
HARD_DIR="analysis/benchmark/hard_for_all"
EVAL_SCRIPT="analysis/post_training_pipeline/scripts/evaluate_cv_fold_subgroup.py"
EVAL_FEAT_SCRIPT="analysis/post_training_pipeline/scripts/evaluate_cv_fold_featureonly.py"
ABLATION_SCRIPT="analysis/pattern/analyze_subgroup_ablation.py"
LATENT_SCRIPT="analysis/pattern/analyze_latent_probing.py"
RESIDUAL_SCRIPT="analysis/pattern/analyze_residual_ablation.py"
REPR_SCRIPT="analysis/pattern/extract_representations.py"

# ── Defaults ──────────────────────────────────────────────────────────────────
BLNC_EXP_V47="" ; BLNC_EXP_V49=""
FEAT_EXP_V47="" ; FEAT_EXP_V49=""
CONFIG_V47=""   ; CONFIG_V49=""

LNC_FASTA_V47="" ; PC_FASTA_V47=""
LNC_FASTA_V49="" ; PC_FASTA_V49=""
TRAIN_LNC_V47="" ; TRAIN_PC_V47=""
TRAIN_LNC_V49="" ; TRAIN_PC_V49=""

BIOTYPE_CSV_V47="" ; BIOTYPE_CSV_V49=""
CPAT_CSV_V47=""    ; CPAT_CSV_V49=""
LNCRNABERT_CSV_V47="" ; LNCRNABERT_CSV_V49=""
GTF_V47=""   # GTF_V49 optional — not required
GTF_V49=""

ORTHRUS_ENV="{$HOME}.conda/envs/orthrus"
RNASAMBA_SIF="rnasamba.sif"
FIGURES_DIR="figures"
N_FOLDS=5
DEVICE="cuda:0"
MIN_SET_SIZE=5

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --blnc_exp_v47)       BLNC_EXP_V47="$2";       shift 2 ;;
        --blnc_exp_v49)       BLNC_EXP_V49="$2";       shift 2 ;;
        --feat_exp_v47)       FEAT_EXP_V47="$2";       shift 2 ;;
        --feat_exp_v49)       FEAT_EXP_V49="$2";       shift 2 ;;
        --config_v47)         CONFIG_V47="$2";          shift 2 ;;
        --config_v49)         CONFIG_V49="$2";          shift 2 ;;
        --lnc_fasta_v47)      LNC_FASTA_V47="$2";      shift 2 ;;
        --pc_fasta_v47)       PC_FASTA_V47="$2";       shift 2 ;;
        --lnc_fasta_v49)      LNC_FASTA_V49="$2";      shift 2 ;;
        --pc_fasta_v49)       PC_FASTA_V49="$2";       shift 2 ;;
        --train_lnc_v47)      TRAIN_LNC_V47="$2";      shift 2 ;;
        --train_pc_v47)       TRAIN_PC_V47="$2";       shift 2 ;;
        --train_lnc_v49)      TRAIN_LNC_V49="$2";      shift 2 ;;
        --train_pc_v49)       TRAIN_PC_V49="$2";       shift 2 ;;
        --biotype_csv_v47)    BIOTYPE_CSV_V47="$2";    shift 2 ;;
        --biotype_csv_v49)    BIOTYPE_CSV_V49="$2";    shift 2 ;;
        --cpat_csv_v47)       CPAT_CSV_V47="$2";       shift 2 ;;
        --cpat_csv_v49)       CPAT_CSV_V49="$2";       shift 2 ;;
        --lncrnabert_csv_v47) LNCRNABERT_CSV_V47="$2"; shift 2 ;;
        --lncrnabert_csv_v49) LNCRNABERT_CSV_V49="$2"; shift 2 ;;
        --gtf_v47)            GTF_V47="$2";            shift 2 ;;
        --gtf_v49)            GTF_V49="$2";            shift 2 ;;
        --orthrus_env)        ORTHRUS_ENV="$2";        shift 2 ;;
        --rnasamba_sif)       RNASAMBA_SIF="$2";       shift 2 ;;
        --figures_dir)        FIGURES_DIR="$2";        shift 2 ;;
        --n_folds)            N_FOLDS="$2";            shift 2 ;;
        --device)             DEVICE="$2";             shift 2 ;;
        --min_set_size)       MIN_SET_SIZE="$2";       shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Validate required args ────────────────────────────────────────────────────
for var in BLNC_EXP_V47 BLNC_EXP_V49 CONFIG_V47 CONFIG_V49 \
           LNC_FASTA_V47 PC_FASTA_V47 LNC_FASTA_V49 PC_FASTA_V49 \
           BIOTYPE_CSV_V47 BIOTYPE_CSV_V49 CPAT_CSV_V47 CPAT_CSV_V49; do
    if [[ -z "${!var}" ]]; then
        echo "ERROR: --${var,,} is required"
        exit 1
    fi
done

# ── Derived paths ─────────────────────────────────────────────────────────────
BLNC_PREDS_V47="${BLNC_EXP_V47}/evaluation_csvs/test_predictions.csv"
BLNC_PREDS_V49="${BLNC_EXP_V49}/evaluation_csvs/test_predictions.csv"
FEAT_PREDS_V47="${FEAT_EXP_V47}/evaluation_csvs/test_predictions.csv"
FEAT_PREDS_V49="${FEAT_EXP_V49}/evaluation_csvs/test_predictions.csv"

BENCHMARK_DIR_V47="$(dirname "$BLNC_EXP_V47")/benchmark_tools_new"
BENCHMARK_DIR_V49="$(dirname "$BLNC_EXP_V49")/benchmark_tools_new"
COMPARISON_DIR_V47="$(dirname "$BLNC_EXP_V47")/benchmark_comparison_new"
COMPARISON_DIR_V49="$(dirname "$BLNC_EXP_V49")/benchmark_comparison_new"
HARD_OUT_V47="${COMPARISON_DIR_V47}/hard_for_all"
HARD_OUT_V49="${COMPARISON_DIR_V49}/hard_for_all"

mkdir -p "$FIGURES_DIR"
LOG_FILE="posttrain_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
log_section() {
    echo "" | tee -a "$LOG_FILE"
    echo "$(printf '=%.0s' {1..70})" | tee -a "$LOG_FILE"
    echo "  $*" | tee -a "$LOG_FILE"
    echo "$(printf '=%.0s' {1..70})" | tee -a "$LOG_FILE"
}
log_warn() { echo "[$(date '+%H:%M:%S')]   WARNING: $*" | tee -a "$LOG_FILE"; }
log_fail() { echo "[$(date '+%H:%M:%S')]   FAILED:  $*" | tee -a "$LOG_FILE"; }
log_ok()   { echo "[$(date '+%H:%M:%S')]   OK:      $*" | tee -a "$LOG_FILE"; }

STATUS_V47=0
STATUS_V49=0
STATUS_HARD=0
STATUS_INTERP_V47=0
STATUS_INTERP_V49=0
STATUS_BENCH_V47=0
STATUS_BENCH_V49=0

log_section "β-LNC post-training pipeline"
log "Log: ${LOG_FILE}"

# =============================================================================
# Stage 1 — evaluate_cv_subgroup.py (per experiment per release)
# =============================================================================
run_evaluation() {
    local EXP_DIR="$1"
    local CONFIG="$2"
    local LABEL="$3"
    local LNC_FA="$4"
    local PC_FA="$5"
    local EVAL_PY="$6"

    log_section "Stage 1 — Evaluation: ${LABEL}"

    if [[ ! -d "${EXP_DIR}/fold_0" ]] && [[ ! -d "${EXP_DIR}/fold_attention" ]]; then
        log_warn "No fold outputs found in ${EXP_DIR} — training may not have completed"
    fi

    if [[ -f "${EXP_DIR}/evaluation_csvs/test_predictions.csv" ]] && [[ -f "${EXP_DIR}/evaluation_csvs/test_hard_cases.csv" ]]; then
        log_warn "Evaluation CSV already exists — skipping"
        return 0
    fi

    if conda run -n beta_lncrna python "$EVAL_PY" \
            --experiment_dir "$EXP_DIR" \
            --config         "$CONFIG" \
            --lnc_fasta      "$LNC_FA" \
            --pc_fasta       "$PC_FA" \
            --output_dir     "${EXP_DIR}/evaluation_csvs" \
            --device         "$DEVICE" \
        2>&1 | tee -a "$LOG_FILE"; then
        log_ok "${LABEL}: test_predictions.csv written"
        return 0
    else
        log_fail "${LABEL}: ${EVAL_PY} failed"
        return 1
    fi
}

# G47 — Subgroup Base
run_evaluation "$BLNC_EXP_V47" "$CONFIG_V47" "β-LNC Subgroup Base G47" \
    "$LNC_FASTA_V47" "$PC_FASTA_V47" "$EVAL_SCRIPT" || STATUS_V47=1

# G47 — Feature-Only (optional)
if [[ -n "$FEAT_EXP_V47" ]]; then
    run_evaluation "$FEAT_EXP_V47" "$CONFIG_V47" "β-LNC Feature-Only G47" \
        "$LNC_FASTA_V47" "$PC_FASTA_V47" "$EVAL_FEAT_SCRIPT" || STATUS_V47=1
fi

# G49 — Subgroup Base
run_evaluation "$BLNC_EXP_V49" "$CONFIG_V49" "β-LNC Subgroup Base G49" \
    "$LNC_FASTA_V49" "$PC_FASTA_V49" "$EVAL_SCRIPT" || STATUS_V49=1

# G49 — Feature-Only (optional)
if [[ -n "$FEAT_EXP_V49" ]]; then
    run_evaluation "$FEAT_EXP_V49" "$CONFIG_V49" "β-LNC Feature-Only G49" \
        "$LNC_FASTA_V49" "$PC_FASTA_V49" "$EVAL_FEAT_SCRIPT" || STATUS_V49=1
fi

STATUS_INTERP_V47=0
STATUS_INTERP_V49=0

# =============================================================================
# Stage 2 — Interpretability analyses (per experiment)
# =============================================================================
run_interpretability() {
    local EXP_DIR="$1"
    local CONFIG="$2"
    local LABEL="$3"
    local FEAT_ZERO_CSV="${EXP_DIR}/ablation_analysis/all_folds_ablation.csv"

    log_section "Stage 2 — Interpretability: ${LABEL}"

    # ── 2a: extract_representations.py ───────────────────────────────────────
    local REPR_DIR="${EXP_DIR}/representations"
    log "  2a. extract_representations.py → ${REPR_DIR}"
    if [[ -d "$REPR_DIR" ]] && ls "${REPR_DIR}"/fold_*_repr.npz &>/dev/null 2>&1; then
        log "      Already exists — skipping"
    else
        if conda run -n beta_lncrna python "$REPR_SCRIPT" \
                --experiment_dir "$EXP_DIR" \
                --config         "$CONFIG" \
                --output_dir     "$REPR_DIR" \
                --device         "$DEVICE" \
            2>&1 | tee -a "$LOG_FILE"; then
            log_ok "Representations extracted → ${REPR_DIR}"
        else
            log_fail "extract_representations.py failed for ${LABEL}"
            return 1
        fi
    fi

    # ── 2b: analyze_subgroup_ablation.py ─────────────────────────────────────
    local ABLATION_DIR="${EXP_DIR}/ablation_analysis"
    log "  2b. analyze_subgroup_ablation.py → ${ABLATION_DIR}"
    if [[ -f "${ABLATION_DIR}/all_folds_ablation.csv" ]]; then
        log "      Already exists — skipping"
    else
        if conda run -n beta_lncrna python "$ABLATION_SCRIPT" \
                --experiment_dir "$EXP_DIR" \
                --config         "$CONFIG" \
                --output_dir     "$ABLATION_DIR" \
                --model_label    "$LABEL" \
                --device         "$DEVICE" \
            2>&1 | tee -a "$LOG_FILE"; then
            log_ok "Ablation complete → ${ABLATION_DIR}"
        else
            log_fail "analyze_subgroup_ablation.py failed for ${LABEL}"
        fi
    fi

    # ── 2c: analyze_latent_probing.py ────────────────────────────────────────
    local LATENT_DIR="${EXP_DIR}/latent_probing"
    log "  2c. analyze_latent_probing.py → ${LATENT_DIR}"
    if [[ -f "${LATENT_DIR}/cross_fold_pattern_comparison.png" ]]; then
        log "      Already exists — skipping"
    else
        if conda run -n beta_lncrna python "$LATENT_SCRIPT" \
                --repr_dir   "$REPR_DIR" \
                --output_dir "$LATENT_DIR" \
                --model_label "$LABEL" \
            2>&1 | tee -a "$LOG_FILE"; then
            log_ok "Latent probing + pattern analysis complete → ${LATENT_DIR}"
        else
            log_fail "analyze_latent_probing.py failed for ${LABEL}"
        fi
    fi

    # ── 2d: analyze_residual_ablation.py ─────────────────────────────────────
    local RESIDUAL_DIR="${EXP_DIR}/residual_ablation"
    log "  2d. analyze_residual_ablation.py → ${RESIDUAL_DIR}"
    if [[ -f "${RESIDUAL_DIR}/all_folds_residual_ablation.csv" ]]; then
        log "      Already exists — skipping"
    else
        local RESIDUAL_ARGS=(
            --experiment_dir "$EXP_DIR"
            --repr_dir       "$REPR_DIR"
            --config         "$CONFIG"
            --output_dir     "$RESIDUAL_DIR"
            --model_label    "$LABEL"
            --device         "$DEVICE"
        )
        [[ -f "$FEAT_ZERO_CSV" ]] && \
            RESIDUAL_ARGS+=(--feature_zero_csv "$FEAT_ZERO_CSV")

        if conda run -n beta_lncrna python "$RESIDUAL_SCRIPT" \
                "${RESIDUAL_ARGS[@]}" \
            2>&1 | tee -a "$LOG_FILE"; then
            log_ok "Residual ablation complete → ${RESIDUAL_DIR}"
        else
            log_fail "analyze_residual_ablation.py failed for ${LABEL}"
        fi
    fi

    return 0
}

run_interpretability "$BLNC_EXP_V47" "$CONFIG_V47" \
    "β-LNC Subgroup Base G47" || STATUS_INTERP_V47=1
run_interpretability "$BLNC_EXP_V49" "$CONFIG_V49" \
    "β-LNC Subgroup Base G49" || STATUS_INTERP_V49=1
run_benchmark() {
    local REL="$1"
    local BASE_DIR="$2"
    local BLNC_PREDS="$3"
    local FEAT_PREDS="$4"
    local LNC_FA="$5"
    local PC_FA="$6"
    local TRAIN_LNC="$7"
    local TRAIN_PC="$8"
    local LNCRNABERT_CSV="$9"

    log_section "Stage 3 — Benchmark [${REL}]"

    if [[ ! -f "$BLNC_PREDS" ]]; then
        log_fail "[${REL}] ${BLNC_PREDS} not found — skipping benchmark (evaluation failed?)"
        return 1
    fi

    local BENCH_ARGS=(
        --release        "$REL"
        --base_dir       "$BASE_DIR"
        --lnc_fasta      "$LNC_FA"
        --pc_fasta       "$PC_FA"
        --train_lnc_fasta "$TRAIN_LNC"
        --train_pc_fasta  "$TRAIN_PC"
        --blnc_predictions "$BLNC_PREDS"
        --orthrus_env    "$ORTHRUS_ENV"
        --rnasamba_sif   "$RNASAMBA_SIF"
        --error_only
        --min_set_size   "$MIN_SET_SIZE"
    )
    [[ -n "$FEAT_PREDS" ]] && [[ -f "$FEAT_PREDS" ]] && \
        BENCH_ARGS+=(--feature_only_csv "$FEAT_PREDS")
    [[ -n "$LNCRNABERT_CSV" ]] && [[ -f "$LNCRNABERT_CSV" ]] && \
        BENCH_ARGS+=(--lncrnabert_csv "$LNCRNABERT_CSV")

    if bash "${SCRIPT_DIR}/run_full_benchmark.sh" "${BENCH_ARGS[@]}" \
        2>&1 | tee -a "$LOG_FILE"; then
        log_ok "[${REL}] Benchmark complete → ${BASE_DIR}/benchmark_comparison/"
        return 0
    else
        log_fail "[${REL}] run_full_benchmark.sh failed"
        return 1
    fi
}

BASE_DIR_V47="$(dirname "$BLNC_EXP_V47")"
BASE_DIR_V49="$(dirname "$BLNC_EXP_V49")"

run_benchmark "v47" "$BASE_DIR_V47" "$BLNC_PREDS_V47" "$FEAT_PREDS_V47" \
    "$LNC_FASTA_V47" "$PC_FASTA_V47" "$TRAIN_LNC_V47" "$TRAIN_PC_V47" \
    "$LNCRNABERT_CSV_V47" || STATUS_BENCH_V47=1

run_benchmark "v49" "$BASE_DIR_V49" "$BLNC_PREDS_V49" "$FEAT_PREDS_V49" \
    "$LNC_FASTA_V49" "$PC_FASTA_V49" "$TRAIN_LNC_V49" "$TRAIN_PC_V49" \
    "$LNCRNABERT_CSV_V49" || STATUS_BENCH_V49=1

# =============================================================================
# Stage 4 — run_hard_for_all.sh (both releases, depends on both benchmarks)
# =============================================================================
log_section "Stage 4 — Hard-for-all analysis [v47 + v49]"

if [[ $STATUS_BENCH_V47 -ne 0 ]] || [[ $STATUS_BENCH_V49 -ne 0 ]]; then
    log_warn "One or both benchmark stages failed — skipping hard-for-all analysis"
    log_warn "Fix benchmark errors and rerun run_hard_for_all.sh manually"
    STATUS_HARD=1
else
    PRED_MERGED_V47="${COMPARISON_DIR_V47}/predictions_merged.csv"
    PRED_MERGED_V49="${COMPARISON_DIR_V49}/predictions_merged.csv"

    HARD_ARGS=(
        --predictions_csv_v47 "$PRED_MERGED_V47"
        --predictions_csv_v49 "$PRED_MERGED_V49"
        --blnc_csv_v47        "$BLNC_PREDS_V47"
        --blnc_csv_v49        "$BLNC_PREDS_V49"
        --biotype_csv_v47     "$BIOTYPE_CSV_V47"
        --biotype_csv_v49     "$BIOTYPE_CSV_V49"
        --cpat_csv_v47        "$CPAT_CSV_V47"
        --cpat_csv_v49        "$CPAT_CSV_V49"
        --output_dir_v47      "$HARD_OUT_V47"
        --output_dir_v49      "$HARD_OUT_V49"
        --figures_dir         "$FIGURES_DIR"
        --error_only
    )
    [[ -n "$GTF_V47" ]] && HARD_ARGS+=(--gtf_v47 "$GTF_V47")
    [[ -n "$GTF_V49" ]] && HARD_ARGS+=(--gtf_v49 "$GTF_V49")

    if bash "${HARD_DIR}/run_hard_for_all.sh" "${HARD_ARGS[@]}" \
        2>&1 | tee -a "$LOG_FILE"; then
        log_ok "Hard-for-all analysis complete → ${FIGURES_DIR}/"
    else
        log_fail "run_hard_for_all.sh failed"
        STATUS_HARD=1
    fi
fi

# =============================================================================
# Summary
# =============================================================================
log_section "Post-training pipeline — Summary"

print_status() {
    local LABEL="$1" ; local STATUS="$2"
    if [[ $STATUS -eq 0 ]]; then
        log_ok  "${LABEL}"
    else
        log_fail "${LABEL}"
    fi
}

print_status "v47 — evaluation"           $STATUS_V47
print_status "v49 — evaluation"           $STATUS_V49
print_status "v47 — interpretability"     $STATUS_INTERP_V47
print_status "v49 — interpretability"     $STATUS_INTERP_V49
print_status "v47 — benchmark"            $STATUS_BENCH_V47
print_status "v49 — benchmark"            $STATUS_BENCH_V49
print_status "hard-for-all analysis"      $STATUS_HARD

echo "" | tee -a "$LOG_FILE"

TOTAL=$((STATUS_V47 + STATUS_V49 + STATUS_INTERP_V47 + STATUS_INTERP_V49 + STATUS_BENCH_V47 + STATUS_BENCH_V49 + STATUS_HARD))
if [[ $TOTAL -eq 0 ]]; then
    log "All stages completed successfully."
else
    log_warn "${TOTAL} stage(s) failed — review the log above for details."
    log_warn "Failed stages can be rerun individually:"
    log_warn "  evaluation  : python ${EVAL_SCRIPT} / ${EVAL_FEAT_SCRIPT} --experiment_dir ..."
log_warn "  interpretability: python ${ABLATION_SCRIPT} / ${LATENT_SCRIPT} / ${RESIDUAL_SCRIPT} ..."
    log_warn "  benchmark   : bash ${SCRIPT_DIR}/run_full_benchmark.sh ..."
    log_warn "  hard-for-all: bash ${HARD_DIR}/run_hard_for_all.sh ..."
fi

log ""
log "Full log: ${LOG_FILE}"

[[ $TOTAL -eq 0 ]] && exit 0 || exit 1