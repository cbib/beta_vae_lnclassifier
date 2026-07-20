#!/bin/bash
# =============================================================================
# run_hard_for_all.sh
#
# End-to-end "hard-for-all" analysis pipeline:
#   1. analyze_hard_for_all.py       — biotype enrichment, group assignment
#   2. analyze_hard_for_all_cpat.py  — CPAT feature comparison (both releases)
#   3. extract_tsl_from_gtf.py       — TSL enrichment (per release, requires GTF)
#   4. plot_hard_for_all_cpat.py     — combined figure (both releases)
#
# Each release is processed independently for steps 1 and 3 (TSL requires a
# release-specific GTF). Steps 2 and 4 combine both releases into a single
# summary CSV / figure.
#
# Usage
# -----
# bash analysis/benchmark/hard_for_all/run_hard_for_all.sh \
#     --predictions_csv_v47 gencode_v47_experiments/benchmark_comparison/predictions_merged.csv \
#     --predictions_csv_v49 gencode_v49_experiments/benchmark_comparison/predictions_merged.csv \
#     --blnc_csv_v47 gencode_v47_experiments/beta_vae_subgroup_base_g47/evaluation_csvs/test_predictions.csv \
#     --blnc_csv_v49 gencode_v49_experiments/beta_vae_subgroup_base_g49/evaluation_csvs/test_predictions.csv \
#     --biotype_csv_v47 data/dataset_biotypes/g47_dataset_biotypes_cdhit.csv \
#     --biotype_csv_v49 data/dataset_biotypes/g49_dataset_biotypes_cdhit.csv \
#     --cpat_csv_v47 gencode_v47_experiments/benchmark_tools/cpat/predictions_with_cpat.csv \
#     --cpat_csv_v49 gencode_v49_experiments/benchmark_tools/cpat/predictions_with_cpat.csv \
#     --gtf_v47 /mnt/cbib/LNClassifier/paper/nonb-pipeline.old/resources/gencode.v47.annotation.gtf \
#     [--gtf_v49 ...] \
#     --output_dir_v47 gencode_v47_experiments/benchmark_comparison/hard_for_all \
#     --output_dir_v49 gencode_v49_experiments/benchmark_comparison/hard_for_all \
#     --figures_dir figures/ \
#     [--error_only] \
#     [--min_biotype_count 10]
#
# Notes
# -----
# - --gtf_v49 is optional. If omitted, TSL analysis is skipped for v49 (the
#   step prints a warning and continues — the rest of the pipeline does not
#   depend on TSL output).
# - All steps are independent; if one fails the script reports it and
#   continues to the next, so a missing GTF doesn't block the figure.
# =============================================================================

set -uo pipefail
cd /mnt/cbib/LNClassifier/beta_vae_lnclassifier

SCRIPT_DIR="analysis/benchmark/hard_for_all"

# ── Defaults ──────────────────────────────────────────────────────────────────
PRED_V47="" ; PRED_V49=""
BLNC_V47="" ; BLNC_V49=""
BIOTYPE_V47="" ; BIOTYPE_V49=""
CPAT_V47="" ; CPAT_V49=""
GTF_V47="" ; GTF_V49=""
OUT_V47="" ; OUT_V49=""
FIGURES_DIR="figures"
ERROR_ONLY=false
MIN_BIOTYPE_COUNT=10

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --predictions_csv_v47) PRED_V47="$2";    shift 2 ;;
        --predictions_csv_v49) PRED_V49="$2";    shift 2 ;;
        --blnc_csv_v47)        BLNC_V47="$2";    shift 2 ;;
        --blnc_csv_v49)        BLNC_V49="$2";    shift 2 ;;
        --biotype_csv_v47)     BIOTYPE_V47="$2"; shift 2 ;;
        --biotype_csv_v49)     BIOTYPE_V49="$2"; shift 2 ;;
        --cpat_csv_v47)        CPAT_V47="$2";    shift 2 ;;
        --cpat_csv_v49)        CPAT_V49="$2";    shift 2 ;;
        --gtf_v47)             GTF_V47="$2";     shift 2 ;;
        --gtf_v49)             GTF_V49="$2";     shift 2 ;;
        --output_dir_v47)      OUT_V47="$2";     shift 2 ;;
        --output_dir_v49)      OUT_V49="$2";     shift 2 ;;
        --figures_dir)         FIGURES_DIR="$2"; shift 2 ;;
        --error_only)          ERROR_ONLY=true;  shift 1 ;;
        --min_biotype_count)   MIN_BIOTYPE_COUNT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Validate required args ────────────────────────────────────────────────────
for var in PRED_V47 PRED_V49 BLNC_V47 BLNC_V49 BIOTYPE_V47 BIOTYPE_V49 \
           CPAT_V47 CPAT_V49 OUT_V47 OUT_V49; do
    if [[ -z "${!var}" ]]; then
        echo "ERROR: --${var,,} is required" | tr 'A-Z_' 'a-z_'
        exit 1
    fi
done

mkdir -p "$OUT_V47" "$OUT_V49" "$FIGURES_DIR"

LOG_FILE="${OUT_V47}/../hard_for_all_$(date +%Y%m%d_%H%M%S).log"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
log_section() {
    echo "" | tee -a "$LOG_FILE"
    echo "$(printf '=%.0s' {1..70})" | tee -a "$LOG_FILE"
    echo "  $*" | tee -a "$LOG_FILE"
    echo "$(printf '=%.0s' {1..70})" | tee -a "$LOG_FILE"
}

ERROR_FLAG=""
$ERROR_ONLY && ERROR_FLAG="--error_only"

log_section "Hard-for-all analysis pipeline"
log "Error-only mode : ${ERROR_ONLY}"
log "Figures dir     : ${FIGURES_DIR}"

STATUS=0

# =============================================================================
# Step 1 — analyze_hard_for_all.py (per release)
# =============================================================================
for REL in v47 v49; do
    PRED_VAR="PRED_${REL^^}"
    BLNC_VAR="BLNC_${REL^^}"
    BIOTYPE_VAR="BIOTYPE_${REL^^}"
    OUT_VAR="OUT_${REL^^}"

    log_section "Step 1/4 — analyze_hard_for_all.py [${REL}]"
    CMD=(python "${SCRIPT_DIR}/analyze_hard_for_all.py"
        --predictions_csv "${!PRED_VAR}"
        --biotype_csv     "${!BIOTYPE_VAR}"
        --blnc_csv        "${!BLNC_VAR}"
        --output_dir      "${!OUT_VAR}"
        --min_biotype_count "$MIN_BIOTYPE_COUNT"
    )
    [[ -n "$ERROR_FLAG" ]] && CMD+=("$ERROR_FLAG")

    log "Command: ${CMD[*]}"
    if "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
        log "   [${REL}] hard_for_all_transcripts.csv written to ${!OUT_VAR}"
    else
        log "   [${REL}] analyze_hard_for_all.py failed — see log"
        STATUS=1
    fi
done

# =============================================================================
# Step 2 — analyze_hard_for_all_cpat.py (both releases in one call)
# =============================================================================
log_section "Step 2/4 — analyze_hard_for_all_cpat.py [v47 + v49]"

HARD_V47="${OUT_V47}/hard_for_all_transcripts.csv"
HARD_V49="${OUT_V49}/hard_for_all_transcripts.csv"

if [[ -f "$HARD_V47" ]] && [[ -f "$HARD_V49" ]]; then
    CMD=(python "${SCRIPT_DIR}/analyze_hard_for_all.py"
        --hard_csv    "$HARD_V47"
        --cpat_csv    "$CPAT_V47"
        --output_dir  "$OUT_V47"
        --release     v47
        --hard_csv2   "$HARD_V49"
        --cpat_csv2   "$CPAT_V49"
        --output_dir2 "$OUT_V49"
        --release2    v49
    )
    log "Command: ${CMD[*]}"
    if "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
        log "   cpat_feature_summary_combined.csv written"
    else
        log "   analyze_hard_for_all.py failed — see log"
        STATUS=1
    fi
else
    log "   Skipping — hard_for_all_transcripts.csv missing for one or both releases"
    log "    v47: ${HARD_V47} ($([[ -f "$HARD_V47" ]] && echo found || echo missing))"
    log "    v49: ${HARD_V49} ($([[ -f "$HARD_V49" ]] && echo found || echo missing))"
    STATUS=1
fi

# =============================================================================
# Step 3 — extract_tsl_from_gtf.py (per release, optional)
# =============================================================================
for REL in v47 v49; do
    GTF_VAR="GTF_${REL^^}"
    PRED_VAR="PRED_${REL^^}"
    OUT_VAR="OUT_${REL^^}"
    HARD_CSV="${!OUT_VAR}/hard_for_all_transcripts.csv"

    log_section "Step 3/4 — extract_tsl_gtf.py [${REL}]"

    if [[ -z "${!GTF_VAR}" ]]; then
        log "  Skipped — no --gtf_${REL} provided"
        continue
    fi
    if [[ ! -f "${!GTF_VAR}" ]]; then
        log "   GTF not found at ${!GTF_VAR} — skipping"
        continue
    fi
    if [[ ! -f "$HARD_CSV" ]]; then
        log "   ${HARD_CSV} not found (Step 1 may have failed) — skipping"
        continue
    fi

    CMD=(python "${SCRIPT_DIR}/extract_tsl_from_gtf.py"
        --gtf            "${!GTF_VAR}"
        --hard_csv       "$HARD_CSV"
        --background_csv "${!PRED_VAR}"
        --output_dir     "${!OUT_VAR}"
    )
    log "Command: ${CMD[*]}"
    if "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
        log "   [${REL}] tsl_enrichment.csv written"
    else
        log "   [${REL}] extract_tsl_from_gtf.py failed — see log"
        STATUS=1
    fi
done

# =============================================================================
# Step 4 — plot_hard_for_all_cpat.py (both releases)
# =============================================================================
log_section "Step 4/4 — plot_hard_for_all_cpat.py [v47 + v49]"

SUMMARY_CSV="${OUT_V47}/../cpat_feature_summary_combined.csv"
HARD_CPAT_V47="${OUT_V47}/hard_for_all_cpat.csv"
HARD_CPAT_V49="${OUT_V49}/hard_for_all_cpat.csv"

if [[ -f "$SUMMARY_CSV" ]]; then
    CMD=(python "${SCRIPT_DIR}/plot_hard_for_all_cpat.py"
        --summary_csv   "$SUMMARY_CSV"
        --output_dir    "$FIGURES_DIR"
    )
    [[ -f "$HARD_CPAT_V47" ]] && CMD+=(--cpat_v47 "$HARD_CPAT_V47")
    [[ -f "$HARD_CPAT_V49" ]] && CMD+=(--cpat_v49 "$HARD_CPAT_V49")
    [[ -f "$CPAT_V47"      ]] && CMD+=(--cpat_full_v47 "$CPAT_V47")
    [[ -f "$CPAT_V49"      ]] && CMD+=(--cpat_full_v49 "$CPAT_V49")

    log "Command: ${CMD[*]}"
    if "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
        log "   cpat_hard_for_all_figure.{pdf,png} written to ${FIGURES_DIR}"
    else
        log "   plot_hard_for_all_cpat.py failed — see log"
        STATUS=1
    fi
else
    log "   Skipping — ${SUMMARY_CSV} not found (Step 2 may have failed)"
    STATUS=1
fi

# =============================================================================
# Summary
# =============================================================================
log_section "Done"
log "v47 outputs : ${OUT_V47}/"
log "v49 outputs : ${OUT_V49}/"
log "Combined    : $(dirname "$SUMMARY_CSV")/cpat_feature_summary_combined.csv"
log "Figure      : ${FIGURES_DIR}/cpat_hard_for_all_figure.{pdf,png}"
log ""
if [[ $STATUS -eq 0 ]]; then
    log "All steps completed successfully."
else
    log "One or more steps failed or were skipped — see above for details."
fi
log "Full log: ${LOG_FILE}"

exit $STATUS