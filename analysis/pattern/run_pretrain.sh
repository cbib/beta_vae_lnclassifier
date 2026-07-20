#!/bin/bash
# =============================================================================
# run_pretrain.sh
#
# Pre-training pipeline for β-LNC — runs for both GENCODE releases in sequence.
# CPU-only. Must complete before launching SLURM training jobs.
#
# Steps per release:
#   0. Feature validation + scaler bank generation (prepare_features.py)
#   1. M* / Fisher LDA / Fréchet distance analysis (analyze_mstar.py)
#   2. Per-feature concentration analysis (analyze_feature_concentration.py)
#      — diagnostic/non-blocking; decomposes subgroup-level signal into
#        per-feature contributions to flag concentrated vs diffuse subgroups
#
# Usage
# -----
# bash analysis/run_pretrain.sh \
#     --te_csv_v47      data/processed_features_genomic/g47_te_features_clean.csv \
#     --nonb_csv_v47    data/processed_features_genomic/g47_nonb_features_clean.csv \
#     --te_csv_v49      data/processed_features_genomic/g49_te_features_clean.csv \
#     --nonb_csv_v49    data/processed_features_genomic/g49_nonb_features_clean.csv \
#     --split_dir_v47   data/split_gencode_47 \
#     --split_dir_v49   data/split_gencode_49 \
#     --feature_dir     data/processed_features_genomic \
#     --output_dir_v47  gencode_v47_experiments/mstar \
#     --output_dir_v49  gencode_v49_experiments/mstar
# =============================================================================

set -uo pipefail
cd /mnt/cbib/LNClassifier/beta_vae_lnclassifier

# ── Defaults ──────────────────────────────────────────────────────────────────
TE_CSV_V47=""
NONB_CSV_V47=""
TE_CSV_V49=""
NONB_CSV_V49=""
SPLIT_DIR_V47="data/split_gencode_47"
SPLIT_DIR_V49="data/split_gencode_49"
FEATURE_DIR="data/processed_features_genomic"
OUTPUT_DIR_V47="gencode_v47_experiments/mstar"
OUTPUT_DIR_V49="gencode_v49_experiments/mstar"
OLD_TE_CSV_V47=""
OLD_NONB_CSV_V47=""
OLD_TE_CSV_V49=""
OLD_NONB_CSV_V49=""
OLD_TE_SCALER_V47=""
OLD_NONB_SCALER_V47=""
OLD_TE_SCALER_V49=""
OLD_NONB_SCALER_V49=""
NONB2_CSV_V47=""
NONB2_CSV_V49=""
CONFIG_V47="configs/beta_vae_subgroup_base_g47.json"
CONFIG_V49="configs/beta_vae_subgroup_base_g49.json"
CHUNK_SIZE=50000
CONCENTRATION_TOP_N=8   # number of top-Fréchet subgroups to run LOO sensitivity on

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --te_csv_v47)     TE_CSV_V47="$2";     shift 2 ;;
        --nonb_csv_v47)   NONB_CSV_V47="$2";   shift 2 ;;
        --te_csv_v49)     TE_CSV_V49="$2";     shift 2 ;;
        --nonb_csv_v49)   NONB_CSV_V49="$2";   shift 2 ;;
        --split_dir_v47)  SPLIT_DIR_V47="$2";  shift 2 ;;
        --split_dir_v49)  SPLIT_DIR_V49="$2";  shift 2 ;;
        --feature_dir)    FEATURE_DIR="$2";    shift 2 ;;
        --output_dir_v47) OUTPUT_DIR_V47="$2"; shift 2 ;;
        --output_dir_v49) OUTPUT_DIR_V49="$2"; shift 2 ;;
        --chunk_size)     CHUNK_SIZE="$2";     shift 2 ;;
        --concentration_top_n) CONCENTRATION_TOP_N="$2"; shift 2 ;;
        --old_te_csv_v47)      OLD_TE_CSV_V47="$2";      shift 2 ;;
        --old_nonb_csv_v47)    OLD_NONB_CSV_V47="$2";    shift 2 ;;
        --old_te_csv_v49)      OLD_TE_CSV_V49="$2";      shift 2 ;;
        --old_nonb_csv_v49)    OLD_NONB_CSV_V49="$2";    shift 2 ;;
        --old_te_scaler_v47)   OLD_TE_SCALER_V47="$2";   shift 2 ;;
        --old_nonb_scaler_v47) OLD_NONB_SCALER_V47="$2"; shift 2 ;;
        --old_te_scaler_v49)   OLD_TE_SCALER_V49="$2";   shift 2 ;;
        --old_nonb_scaler_v49) OLD_NONB_SCALER_V49="$2"; shift 2 ;;
        --nonb2_csv_v47)       NONB2_CSV_V47="$2";       shift 2 ;;
        --nonb2_csv_v49)       NONB2_CSV_V49="$2";       shift 2 ;;
        --config_v47)          CONFIG_V47="$2";          shift 2 ;;
        --config_v49)          CONFIG_V49="$2";          shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Validate required args ────────────────────────────────────────────────────
for var in TE_CSV_V47 NONB_CSV_V47 TE_CSV_V49 NONB_CSV_V49; do
    if [[ -z "${!var}" ]]; then
        echo "ERROR: --${var,,} is required"
        exit 1
    fi
done

# ── Setup logging ─────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR_V47" "$OUTPUT_DIR_V49"
LOG_FILE="pretrain_$(date +%Y%m%d_%H%M%S).log"

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

# Track per-release status
STATUS_V47=0
STATUS_V49=0

log_section "β-LNC pre-training pipeline"
log "Releases : v47 + v49"
log "Log      : ${LOG_FILE}"

# =============================================================================
# Helper — run one release
# =============================================================================
run_release() {
    local REL="$1"
    local TE_CSV="$2"
    local NONB_CSV="$3"
    local SPLIT_DIR="$4"
    local OUTPUT_DIR="$5"
    local OLD_TE_CSV="$6"
    local OLD_NONB_CSV="$7"
    local OLD_TE_SCALER="$8"
    local OLD_NONB_SCALER="$9"
    local NONB2_CSV="${10}"
    local CONFIG="${11}"
    local REL_TAG="${12}"   # e.g. "g47" or "g49"
    local RELEASE_STATUS=0

    log_section "GENCODE ${REL}"

    # ── Validate input files ──────────────────────────────────────────────────
    log "Checking input files..."
    local MISSING=0
    for f in "$TE_CSV" "$NONB_CSV" "$SPLIT_DIR"; do
        if [[ ! -e "$f" ]]; then
            log_fail "Not found: $f"
            MISSING=1
        else
            log "  Found: $f"
        fi
    done
    if [[ $MISSING -eq 1 ]]; then
        log_fail "[${REL}] Input files missing — skipping this release"
        return 1
    fi

    # ── Stage 0a: Feature version comparison (optional) ───────────────────────
    if [[ -n "$OLD_TE_CSV" ]] && [[ -f "$OLD_TE_CSV" ]] && \
       [[ -n "$OLD_NONB_CSV" ]] && [[ -f "$OLD_NONB_CSV" ]]; then
        log_section "Stage 0a/2 — Feature version comparison [${REL}]"
        local COMPARE_OUT="${OUTPUT_DIR}/feature_comparison"
        mkdir -p "$COMPARE_OUT"

        local COMPARE_ARGS=(
            --old_te_csv   "$OLD_TE_CSV"
            --old_nonb_csv "$OLD_NONB_CSV"
            --new_te_csv   "$TE_CSV"
            --new_nonb_csv "$NONB_CSV"
            --output_dir   "$COMPARE_OUT"
            --split_dir    "$SPLIT_DIR"
        )
        [[ -n "$OLD_TE_SCALER"   ]] && COMPARE_ARGS+=(--old_te_scaler_bank   "$OLD_TE_SCALER")
        [[ -n "$OLD_NONB_SCALER" ]] && COMPARE_ARGS+=(--old_nonb_scaler_bank "$OLD_NONB_SCALER")

        if conda run -n beta_lncrna python analysis/compare_feature_versions.py \
                "${COMPARE_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"; then
            log_ok "[${REL}] Feature comparison complete → ${COMPARE_OUT}/"
        else
            log_warn "[${REL}] Feature comparison failed — continuing anyway"
            log_warn "[${REL}] Review ${COMPARE_OUT}/ before proceeding with training"
        fi
    else
        log "  Stage 0a: Skipped — no --old_*_csv_${REL,,} provided"
        log "            (pass old CSVs to enable differential comparison)"
    fi

    # ── Stage 0: prepare_features.py ─────────────────────────────────────────
    log_section "Stage 0/2 — Feature validation + scaler banks [${REL}]"
    log "  TE CSV    : ${TE_CSV}"
    log "  NonB CSV  : ${NONB_CSV}"
    log "  NonB2 CSV : ${NONB2_CSV:-<not provided>}"
    log "  Split dir : ${SPLIT_DIR}"
    log "  Out dir   : ${FEATURE_DIR}"
    log "  Release   : ${REL_TAG}"

    local PREP_ARGS=(
        --te_csv     "$TE_CSV"
        --nonb_csv   "$NONB_CSV"
        --split_dir  "$SPLIT_DIR"
        --output_dir "$FEATURE_DIR"
        --release    "$REL_TAG"
        --chunk_size "$CHUNK_SIZE"
    )
    [[ -n "$NONB2_CSV" ]] && PREP_ARGS+=(--nonb2_csv "$NONB2_CSV")

    if conda run -n beta_lncrna python data/prepare_features.py \
            "${PREP_ARGS[@]}" \
        2>&1 | tee -a "$LOG_FILE"; then
        log_ok "[${REL}] Scaler banks written to ${FEATURE_DIR}"
    else
        log_fail "[${REL}] prepare_features.py failed — see log"
        log_warn "[${REL}] Skipping M* analysis (depends on Stage 0)"
        return 1
    fi

    # ── Stage 1: analyze_mstar.py ─────────────────────────────────────────────
    # analyze_mstar.py takes --config to load data paths (not raw CSV paths)
    log_section "Stage 1/2 — M* / Fisher LDA / Frechet analysis [${REL}]"
    log "  Config    : ${CONFIG}"
    log "  Output dir: ${OUTPUT_DIR}"
    mkdir -p "$OUTPUT_DIR"

    if conda run -n beta_lncrna python analysis/pattern/analyze_mstar.py \
            --config     "$CONFIG" \
            --output_dir "$OUTPUT_DIR" \
        2>&1 | tee -a "$LOG_FILE"; then
        log_ok "[${REL}] M* analysis complete → ${OUTPUT_DIR}"
    else
        log_fail "[${REL}] analyze_mstar.py failed — see log"
        return 1
    fi

    # ── Stage 2: analyze_feature_concentration.py ─────────────────────────────
    # Non-blocking: diagnostic/optional, depends on Stage 1's frechet_ranking.csv
    # but a failure here should not prevent training from proceeding.
    log_section "Stage 2/2 — Per-feature concentration analysis [${REL}]"
    local CONCENTRATION_DIR="${OUTPUT_DIR}/feature_concentration"
    log "  Output dir   : ${CONCENTRATION_DIR}"
    log "  Top N subgrp : ${CONCENTRATION_TOP_N}"
    mkdir -p "$CONCENTRATION_DIR"

    if conda run -n beta_lncrna python analysis/pattern/analyze_feature_concentration.py \
            --config           "$CONFIG" \
            --mstar_dir        "$OUTPUT_DIR" \
            --output_dir       "$CONCENTRATION_DIR" \
            --top_n_subgroups  "$CONCENTRATION_TOP_N" \
        2>&1 | tee -a "$LOG_FILE"; then
        log_ok "[${REL}] Feature concentration analysis complete → ${CONCENTRATION_DIR}"
    else
        log_warn "[${REL}] analyze_feature_concentration.py failed — see log"
        log_warn "[${REL}] Continuing — this stage is diagnostic, not blocking"
    fi

    return 0
}

# =============================================================================
# Run both releases
# =============================================================================
run_release "v47" "$TE_CSV_V47" "$NONB_CSV_V47" "$SPLIT_DIR_V47" "$OUTPUT_DIR_V47" \
    "$OLD_TE_CSV_V47" "$OLD_NONB_CSV_V47" "$OLD_TE_SCALER_V47" "$OLD_NONB_SCALER_V47" \
    "$NONB2_CSV_V47" "$CONFIG_V47" "g47" \
    || STATUS_V47=1

run_release "v49" "$TE_CSV_V49" "$NONB_CSV_V49" "$SPLIT_DIR_V49" "$OUTPUT_DIR_V49" \
    "$OLD_TE_CSV_V49" "$OLD_NONB_CSV_V49" "$OLD_TE_SCALER_V49" "$OLD_NONB_SCALER_V49" \
    "$NONB2_CSV_V49" "$CONFIG_V49" "g49" \
    || STATUS_V49=1

# =============================================================================
# Summary
# =============================================================================
log_section "Pre-training pipeline — Summary"
if [[ $STATUS_V47 -eq 0 ]]; then
    log_ok  "v47 — all steps completed"
    log     "      Scaler banks : ${FEATURE_DIR}/g47_*scaler_bank*"
    log     "      M* results   : ${OUTPUT_DIR_V47}/"
else
    log_fail "v47 — one or more steps FAILED"
    log_warn "v47 — do NOT launch training until pre-training pipeline passes"
fi

echo "" | tee -a "$LOG_FILE"

if [[ $STATUS_V49 -eq 0 ]]; then
    log_ok  "v49 — all steps completed"
    log     "      Scaler banks : ${FEATURE_DIR}/g49_*scaler_bank*"
    log     "      M* results   : ${OUTPUT_DIR_V49}/"
else
    log_fail "v49 — one or more steps FAILED"
    log_warn "v49 — do NOT launch training until pre-training pipeline passes"
fi

echo "" | tee -a "$LOG_FILE"
log "Full log: ${LOG_FILE}"

# Exit non-zero if either release failed
[[ $STATUS_V47 -eq 0 && $STATUS_V49 -eq 0 ]] && exit 0 || exit 1