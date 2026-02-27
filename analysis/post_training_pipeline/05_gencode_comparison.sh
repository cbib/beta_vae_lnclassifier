#!/bin/bash
set -e

# =============================================================================
# Step 5: GENCODE Version Comparison (OPTIONAL)
# =============================================================================
# Compares two GENCODE versions and analyzes novelty patterns in embedding space
#
# This script runs in two phases:
#   Phase 1: Compare GENCODE versions and generate category FASTAs
#   Phase 2: Analyze embedding space by novelty category
# 
# Prerequisites:
#   - Step 1 completed (embeddings_all_folds.npz exists)
#   - Step 2 completed (UMAP embeddings exist)
#   - GENCODE FASTA files for both versions (all transcripts, PC, lncRNA)
#
# Outputs:
#   - Comparison TSV and category FASTAs (novel, common, reannotated)
#   - Per-fold novelty visualizations
#   - Hard case enrichment statistics
#   - Cross-fold summary figures
# =============================================================================

echo "=============================================="
echo "Step 5: GENCODE Version Comparison (OPTIONAL)"
echo "=============================================="
echo ""
echo "   This step is OPTIONAL - only run if you're comparing GENCODE versions"
echo ""

# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------

EXPERIMENT_DIR=""
OLD_VERSION=""
NEW_VERSION=""
OLD_FASTA=""
OLD_PC=""
OLD_LNC=""
NEW_FASTA=""
NEW_PC=""
NEW_LNC=""
COMPARISON_OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment_dir)
            EXPERIMENT_DIR="$2"
            shift 2
            ;;
        --old-version)
            OLD_VERSION="$2"
            shift 2
            ;;
        --new-version)
            NEW_VERSION="$2"
            shift 2
            ;;
        --old-fasta)
            OLD_FASTA="$2"
            shift 2
            ;;
        --old-pc)
            OLD_PC="$2"
            shift 2
            ;;
        --old-lnc)
            OLD_LNC="$2"
            shift 2
            ;;
        --new-fasta)
            NEW_FASTA="$2"
            shift 2
            ;;
        --new-pc)
            NEW_PC="$2"
            shift 2
            ;;
        --new-lnc)
            NEW_LNC="$2"
            shift 2
            ;;
        --comparison-output-dir)
            COMPARISON_OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "❌ Unknown argument: $1"
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

if [ -z "${EXPERIMENT_DIR}" ]; then
    echo " ERROR: --experiment_dir is required"
    echo ""
    echo "Usage:"
    echo "  bash 05_gencode_comparison.sh \\"
    echo "    --experiment_dir experiments/beta_vae_g47 \\"
    echo "    --old-version 46 \\"
    echo "    --new-version 47 \\"
    echo "    --old-fasta data/raw/gencode.v46.transcripts.fa \\"
    echo "    --old-pc data/raw/gencode.v46.pc_transcripts.fa \\"
    echo "    --old-lnc data/raw/gencode.v46.lncRNA_transcripts.fa \\"
    echo "    --new-fasta data/raw/gencode.v47.transcripts.fa \\"
    echo "    --new-pc data/raw/gencode.v47.pc_transcripts.fa \\"
    echo "    --new-lnc data/raw/gencode.v47.lncRNA_transcripts.fa \\"
    echo "    [--comparison-output-dir resources/gencode_comparison_v46_v47]"
    exit 1
fi

# Check required arguments
REQUIRED_ARGS=(
    "OLD_VERSION:--old-version"
    "NEW_VERSION:--new-version"
    "OLD_FASTA:--old-fasta"
    "OLD_PC:--old-pc"
    "OLD_LNC:--old-lnc"
    "NEW_FASTA:--new-fasta"
    "NEW_PC:--new-pc"
    "NEW_LNC:--new-lnc"
)

for arg_pair in "${REQUIRED_ARGS[@]}"; do
    arg_name="${arg_pair%%:*}"
    arg_flag="${arg_pair##*:}"
    if [ -z "${!arg_name}" ]; then
        echo "❌ ERROR: ${arg_flag} is required"
    exit 1
    fi
done

# Set default comparison output dir if not provided
if [ -z "${COMPARISON_OUTPUT_DIR}" ]; then
    COMPARISON_OUTPUT_DIR="resources/gencode_comparison_v${OLD_VERSION}_v${NEW_VERSION}"
fi

# Check prerequisites from previous steps
EMBEDDINGS_NPZ="${EXPERIMENT_DIR}/embeddings_all_folds.npz"
FOLD_EMBEDDINGS_DIR="${EXPERIMENT_DIR}/umap_visualizations"
HARD_CASES_CSV="${EXPERIMENT_DIR}/evaluation_csvs/hard_cases.csv"

if [ ! -f "${EMBEDDINGS_NPZ}" ]; then
    echo " ERROR: Embeddings not found: ${EMBEDDINGS_NPZ}"
    echo "   Run Step 1 (01_evaluate_cv_folds.sh) first"
    exit 1
fi

if [ ! -d "${FOLD_EMBEDDINGS_DIR}" ]; then
    echo " ERROR: UMAP visualizations not found: ${FOLD_EMBEDDINGS_DIR}"
    echo "   Run Step 2 (02_generate_umap.sh) first"
    exit 1
fi

if [ ! -f "${HARD_CASES_CSV}" ]; then
    echo " ERROR: Hard cases CSV not found: ${HARD_CASES_CSV}"
    echo "   Run Step 1 (01_evaluate_cv_folds.sh) first"
    exit 1
fi

# Check FASTA files exist
for FASTA in "${OLD_FASTA}" "${OLD_PC}" "${OLD_LNC}" "${NEW_FASTA}" "${NEW_PC}" "${NEW_LNC}"; do
    if [ ! -f "${FASTA}" ]; then
        echo " ERROR: FASTA file not found: ${FASTA}"
        exit 1
    fi
done

# -----------------------------------------------------------------------------
# Configuration Display
# -----------------------------------------------------------------------------

ANALYSIS_OUTPUT_DIR="${EXPERIMENT_DIR}/gencode_novelty_analysis"

echo "Configuration:"
echo "  Experiment directory: ${EXPERIMENT_DIR}"
echo "  Comparing: GENCODE v${OLD_VERSION} → v${NEW_VERSION}"
echo ""
echo "  Old version files:"
echo "    All transcripts: ${OLD_FASTA}"
echo "    Protein-coding: ${OLD_PC}"
echo "    lncRNA: ${OLD_LNC}"
echo ""
echo "  New version files:"
echo "    All transcripts: ${NEW_FASTA}"
echo "    Protein-coding: ${NEW_PC}"
echo "    lncRNA: ${NEW_LNC}"
echo ""
echo "  Comparison output: ${COMPARISON_OUTPUT_DIR}"
echo "  Analysis output: ${ANALYSIS_OUTPUT_DIR}"
echo ""

# -----------------------------------------------------------------------------
# Phase 1: Compare GENCODE Versions
# -----------------------------------------------------------------------------

echo "=============================================="
echo "Phase 1: Comparing GENCODE Versions"
echo "=============================================="
echo ""

# Check if comparison already exists
COMPARISON_TSV="${COMPARISON_OUTPUT_DIR}/gencode_v${OLD_VERSION}_v${NEW_VERSION}_comparison.tsv"
NOVEL_FASTA="${COMPARISON_OUTPUT_DIR}/gencode.v${NEW_VERSION}.new_with_class_transcripts.fa"
COMMON_FASTA="${COMPARISON_OUTPUT_DIR}/gencode.v${NEW_VERSION}.common_no_class_change_transcripts.fa"
REANNOTATED_FASTA="${COMPARISON_OUTPUT_DIR}/gencode.v${NEW_VERSION}.common_class_change_transcripts.fa"

if [ -f "${COMPARISON_TSV}" ] && [ -f "${NOVEL_FASTA}" ] && [ -f "${COMMON_FASTA}" ] && [ -f "${REANNOTATED_FASTA}" ]; then
    echo " Comparison files already exist, skipping Phase 1"
    echo "  To regenerate, delete: ${COMPARISON_OUTPUT_DIR}"
else
    echo "Running GENCODE comparison script..."
    echo ""
    
    python analysis/post_training_pipeline/scripts/compare_gencode_versions.py \
        --old-fasta "${OLD_FASTA}" \
        --old-pc "${OLD_PC}" \
        --old-lnc "${OLD_LNC}" \
        --new-fasta "${NEW_FASTA}" \
        --new-pc "${NEW_PC}" \
        --new-lnc "${NEW_LNC}" \
        --old-version "${OLD_VERSION}" \
        --new-version "${NEW_VERSION}" \
        --output-dir "${COMPARISON_OUTPUT_DIR}"
    
    if [ $? -ne 0 ]; then
        echo ""
        echo " Phase 1 failed: GENCODE comparison"
        exit 1
    fi
    
    echo ""
    echo " Phase 1 complete: GENCODE comparison"
fi

echo ""

# -----------------------------------------------------------------------------
# Phase 2: Analyze Embedding Space by Novelty
# -----------------------------------------------------------------------------

echo "=============================================="
echo "Phase 2: Analyzing Embedding Space"
echo "=============================================="
echo ""

echo "Running novelty analysis on embeddings..."
echo ""

python analysis/post_training_pipeline/scripts/gencode_novelty_embeddings.py \
    --embeddings_npz "${EMBEDDINGS_NPZ}" \
    --fold_embeddings_dir "${FOLD_EMBEDDINGS_DIR}" \
    --hard_cases "${HARD_CASES_CSV}" \
    --novel_fasta "${NOVEL_FASTA}" \
    --common_fasta "${COMMON_FASTA}" \
    --reannotated_fasta "${REANNOTATED_FASTA}" \
    --old-version "${OLD_VERSION}" \
    --new-version "${NEW_VERSION}" \
    --output_dir "${ANALYSIS_OUTPUT_DIR}"

# -----------------------------------------------------------------------------
# Completion
# -----------------------------------------------------------------------------

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo " Step 5 Complete: GENCODE Comparison"
    echo "=============================================="
    echo ""
    echo "Generated outputs:"
    echo ""
    echo "Comparison files (Phase 1):"
    echo "  ${COMPARISON_TSV}"
    echo "  ${NOVEL_FASTA}"
    echo "  ${COMMON_FASTA}"
    echo "  ${REANNOTATED_FASTA}"
    echo ""
    echo "Analysis results (Phase 2):"
    ls -lh "${ANALYSIS_OUTPUT_DIR}"/*.png 2>/dev/null || true
    ls -lh "${ANALYSIS_OUTPUT_DIR}"/*.csv 2>/dev/null || true
    echo ""
    echo "Next step: Generate summary report"
    echo "  bash 06_generate_summary_report.sh --experiment_dir ${EXPERIMENT_DIR}"
else
    echo ""
    echo " Step 5 failed"
    exit 1
fi