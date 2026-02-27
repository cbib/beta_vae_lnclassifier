#!/bin/bash
set -e

# Get the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set project root (adjust based on script location)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Add to PYTHONPATH so Python can find modules
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# =============================================================================
# Post-Training Analysis Pipeline - Master Script
# =============================================================================
# Runs the complete post-training analysis pipeline:
#   1. Evaluate CV folds → hard cases + embeddings
#   2. Generate UMAP visualizations
#   3. Spatial clustering of hard cases
#   4. Biotype enrichment analysis
#   5. (Optional) GENCODE novelty comparison
#   6. Generate summary report
#
# Requires conda environment activated prior to running.
# Usage:
#   bash run_all.sh \
#     --experiment_dir experiments/beta_vae_g47 \
#     --config configs/beta_vae.json \
#     --biotype_csv data/processed/gencode47_dataset_biotypes_cdhit90.csv \
#     --n_folds 5
#     --start_from 0 (default, index of which step to start the pipeline from)
#
# With GENCODE comparison:
#   bash run_all.sh \
#     --experiment_dir experiments/beta_vae_g47 \
#     --config configs/beta_vae.json \
#     --biotype_csv data/processed/gencode47_dataset_biotypes_cdhit90.csv \
#     --include_gencode \
#     --old-version 46 \
#     --new-version 47 \
#     --old-fasta data/raw/gencode.v46.transcripts.fa \
#     --old-pc data/raw/gencode.v46.pc_transcripts.fa \
#     --old-lnc data/raw/gencode.v46.lncRNA_transcripts.fa \
#     --new-fasta data/raw/gencode.v47.transcripts.fa \
#     --new-pc data/raw/gencode.v47.pc_transcripts.fa \
#     --new-lnc data/raw/gencode.v47.lncRNA_transcripts.fa
# =============================================================================

echo "=============================================="
echo "Post-Training Analysis Pipeline"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# Defaults (override via CLI or edit here for hardcoded runs)
# -----------------------------------------------------------------------------

# G49 + GENCODE comparison
EXPERIMENT_DIR="gencode_v49_experiments/beta_vae_features_attn_g49_split"
CONFIG="g49_configs/beta_vae_features_attn_g49_split.json"
BIOTYPE_CSV="data/g49_dataset_biotypes_cdhit.csv"
LNC_FASTA="data/split_gencode_49/lnc_trainval.fa"
PC_FASTA="data/split_gencode_49/pc_trainval.fa"
LNC_TEST_FASTA="data/split_gencode_49/lnc_test.fa"
PC_TEST_FASTA="data/split_gencode_49/pc_test.fa"
MODEL_LABEL=""          # e.g. "βVAE+Attn" — passed to all figure-generating scripts
N_FOLDS=5
INCLUDE_GENCODE=0
OLD_VERSION="47"
NEW_VERSION="49"
OLD_FASTA="data/raw/gencode.v47.transcripts.fa"
OLD_PC="data/cdhit_clusters/g47_pc_clustered.fa"
OLD_LNC="data/cdhit_clusters/g47_lncRNA_clustered.fa"
NEW_FASTA="data/raw/gencode.v49.transcripts.fa"
NEW_PC="data/cdhit_clusters/g49_pc_clustered.fa"
NEW_LNC="data/cdhit_clusters/g49_lncRNA_clustered.fa"
COMPARISON_OUTPUT_DIR="data/gencode_comparison_v47_vs_v49_features"

# G47
#EXPERIMENT_DIR="gencode_v47_experiments/beta_vae_features_attn_g47_split"
#CONFIG="configs/beta_vae_features_attn_g47.json"
#BIOTYPE_CSV="data/g47_dataset_biotypes_cdhit.csv"
#LNC_FASTA="data/split_gencode_47/lnc_trainval.fa"
#PC_FASTA="data/split_gencode_47/pc_trainval.fa"
#LNC_TEST_FASTA="data/split_gencode_47/lnc_test.fa"
#PC_TEST_FASTA="data/split_gencode_47/pc_test.fa"
#MODEL_LABEL="βVAE+Attn"
#N_FOLDS=5
#INCLUDE_GENCODE=0

START_FROM=0

# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment_dir)       EXPERIMENT_DIR="$2";       shift 2 ;;
        --config)               CONFIG="$2";               shift 2 ;;
        --biotype_csv)          BIOTYPE_CSV="$2";          shift 2 ;;
        --lnc_fasta)            LNC_FASTA="$2";            shift 2 ;;
        --pc_fasta)             PC_FASTA="$2";             shift 2 ;;
        --lnc_test_fasta)       LNC_TEST_FASTA="$2";       shift 2 ;;
        --pc_test_fasta)        PC_TEST_FASTA="$2";        shift 2 ;;
        --model_label)          MODEL_LABEL="$2";          shift 2 ;;
        --n_folds)              N_FOLDS="$2";              shift 2 ;;
        --include_gencode)      INCLUDE_GENCODE=1;         shift 1 ;;
        --old-version)          OLD_VERSION="$2";          shift 2 ;;
        --new-version)          NEW_VERSION="$2";          shift 2 ;;
        --old-fasta)            OLD_FASTA="$2";            shift 2 ;;
        --old-pc)               OLD_PC="$2";               shift 2 ;;
        --old-lnc)              OLD_LNC="$2";              shift 2 ;;
        --new-fasta)            NEW_FASTA="$2";            shift 2 ;;
        --new-pc)               NEW_PC="$2";               shift 2 ;;
        --new-lnc)              NEW_LNC="$2";              shift 2 ;;
        --comparison-output-dir) COMPARISON_OUTPUT_DIR="$2"; shift 2 ;;
        --start-from)           START_FROM="$2";           shift 2 ;;
        *)
            echo " Unknown argument: $1"
            echo ""
            echo "Usage:"
            echo "  bash run_all.sh \\"
            echo "    --experiment_dir experiments/beta_vae_g47 \\"
            echo "    --config configs/beta_vae.json \\"
            echo "    --biotype_csv data/processed/gencode47_dataset_biotypes_cdhit90.csv \\"
            echo "    --lnc_fasta data/g47_cdhit_clusters/g47_lncRNA_clustered.fa \\"
            echo "    --pc_fasta data/g47_cdhit_clusters/g47_pc_clustered.fa \\"
            echo "    --model_label \"βVAE+Attn\" \\"
            echo "    [--n_folds 5] \\"
            echo "    [--include_gencode] \\"
            echo "    [--start-from N]"
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Infer GENCODE version from experiment dir (produces e.g. "v47")
# -----------------------------------------------------------------------------

GENCODE_VERSION=$(echo "$EXPERIMENT_DIR" | grep -oP '(?<=_g)\d+' | head -1)
[ -n "$GENCODE_VERSION" ] && GENCODE_VERSION="v${GENCODE_VERSION}"

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

[ -z "${EXPERIMENT_DIR}" ] && { echo " ERROR: --experiment_dir is required"; exit 1; }
[ -z "${CONFIG}" ]         && { echo " ERROR: --config is required";          exit 1; }
[ -z "${BIOTYPE_CSV}" ]    && { echo " ERROR: --biotype_csv is required";     exit 1; }
[ -z "${LNC_FASTA}" ]      && { echo " ERROR: --lnc_fasta is required";       exit 1; }
[ -z "${PC_FASTA}" ]       && { echo " ERROR: --pc_fasta is required";        exit 1; }
[ ! -f "${CONFIG}" ]       && { echo " ERROR: Config file not found: ${CONFIG}"; exit 1; }
[ ! -f "${BIOTYPE_CSV}" ]  && { echo " ERROR: Biotype CSV not found: ${BIOTYPE_CSV}"; exit 1; }

if [ ${INCLUDE_GENCODE} -eq 1 ]; then
    GENCODE_REQUIRED=("OLD_VERSION" "NEW_VERSION" "OLD_FASTA" "OLD_PC" "OLD_LNC" "NEW_FASTA" "NEW_PC" "NEW_LNC")
    MISSING_GENCODE_ARGS=()
    for arg_name in "${GENCODE_REQUIRED[@]}"; do
        [ -z "${!arg_name}" ] && MISSING_GENCODE_ARGS+=("--${arg_name,,}")
    done
    if [ ${#MISSING_GENCODE_ARGS[@]} -gt 0 ]; then
        echo "  ERROR: --include_gencode requires: ${MISSING_GENCODE_ARGS[*]}"
        exit 1
    fi
    
    # Check FASTA files exist
    for FASTA in "${OLD_FASTA}" "${OLD_PC}" "${OLD_LNC}" "${NEW_FASTA}" "${NEW_PC}" "${NEW_LNC}"; do
        [ ! -f "${FASTA}" ] && { echo "  ERROR: FASTA file not found: ${FASTA}"; exit 1; }
    done
fi
# -----------------------------------------------------------------------------
# Configuration Display
# -----------------------------------------------------------------------------

echo "Pipeline Configuration:"
echo "  Experiment directory: ${EXPERIMENT_DIR}"
echo "  Config:               ${CONFIG}"
echo "  Biotype CSV:          ${BIOTYPE_CSV}"
echo "  lncRNA FASTA:         ${LNC_FASTA}"
echo "  pcRNA FASTA:          ${PC_FASTA}"
echo "  Model label:          ${MODEL_LABEL:-none}"
echo "  GENCODE version:      ${GENCODE_VERSION:-unknown}"
echo "  Number of folds:      ${N_FOLDS}"
echo "  GENCODE comparison:   $([ ${INCLUDE_GENCODE} -eq 1 ] && echo 'Yes' || echo 'No')"
if [ ${INCLUDE_GENCODE} -eq 1 ]; then
    echo ""
    echo "  GENCODE Comparison: v${OLD_VERSION} → v${NEW_VERSION}"
fi

echo ""
echo "=============================================="
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Track start time
START_TIME=$(date +%s)
STEP_START_TIME=${START_TIME}

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

print_step_header() {
    echo ""
    echo "=============================================="
    echo "Step $1: $2"
    echo "=============================================="
    echo ""
    STEP_START_TIME=$(date +%s)
}

print_step_footer() {
    local DURATION=$(( $(date +%s) - STEP_START_TIME ))
    echo ""
    echo " Step $1 completed in ${DURATION}s"
    echo ""
}

print_step_skip() {
    echo ""
    echo "  Skipping Step $1: $2"
    echo ""
}

print_error() {
    echo ""
    echo "=============================================="
    echo " PIPELINE FAILED AT STEP $1"
    echo "=============================================="
    echo "Error: $2"
    echo ""
    exit 1
}

# if conda env not activated, print warning
if [ -z "$CONDA_PREFIX" ]; then
    echo "  Warning: No conda environment detected"
    echo "  Please activate 'beta_lncrna' before running this pipeline"
    echo ""
fi
# -----------------------------------------------------------------------------
# Step 1: Evaluate CV Folds
# -----------------------------------------------------------------------------

if [ ${START_FROM} -le 1 ]; then
    print_step_header 1 "Evaluate CV Folds"
    
    STEP1_CMD=(
        bash "${SCRIPT_DIR}/01_evaluate_cv_folds.sh"
        --experiment_dir "${EXPERIMENT_DIR}"
        --config "${CONFIG}"
        --n_folds ${N_FOLDS}
        --test_lnc_fasta "${LNC_TEST_FASTA}"
        --test_pc_fasta "${PC_TEST_FASTA}"
        --biotype_csv "${BIOTYPE_CSV}"
    )
    # model_label passed so analyze_attention.py inside step 1 gets the right title
    [ -n "${MODEL_LABEL}" ]     && STEP1_CMD+=(--model_label "${MODEL_LABEL}")
    [ -n "${GENCODE_VERSION}" ] && STEP1_CMD+=(--gencode_version "${GENCODE_VERSION}")

    if ! "${STEP1_CMD[@]}"; then
        print_error 1 "Failed to evaluate CV folds"
    fi
    
    print_step_footer 1
else
    print_step_skip 1 "Evaluate CV Folds"
fi

# -----------------------------------------------------------------------------
# Step 2: Generate UMAP
# -----------------------------------------------------------------------------

if [ ${START_FROM} -le 2 ]; then
    print_step_header 2 "Generate UMAP Visualizations"
    
    STEP2_CMD=(
        bash "${SCRIPT_DIR}/02_generate_umap.sh"
        --experiment_dir "${EXPERIMENT_DIR}"
        --biotype_csv "${BIOTYPE_CSV}"
        --lnc_fasta "${LNC_FASTA}"
        --pc_fasta "${PC_FASTA}"
    )
    [ -n "${MODEL_LABEL}" ]     && STEP2_CMD+=(--model_label "${MODEL_LABEL}")
    [ -n "${GENCODE_VERSION}" ] && STEP2_CMD+=(--gencode_version "${GENCODE_VERSION}")

    if ! "${STEP2_CMD[@]}"; then
        print_error 2 "Failed to generate UMAP visualizations"
    fi
    
    print_step_footer 2
else
    print_step_skip 2 "Generate UMAP Visualizations"
fi

# -----------------------------------------------------------------------------
# Step 3: Spatial Clustering
# -----------------------------------------------------------------------------

if [ ${START_FROM} -le 3 ]; then
    print_step_header 3 "Spatial Clustering"
    
    STEP3_CMD=(
        bash "${SCRIPT_DIR}/03_spatial_clustering.sh"
        --umap_dir "${EXPERIMENT_DIR}/umap_visualizations"
        --output_dir "${EXPERIMENT_DIR}/spatial_analysis"
    )
    [ -n "${MODEL_LABEL}" ]     && STEP3_CMD+=(--model_label "${MODEL_LABEL}")
    [ -n "${GENCODE_VERSION}" ] && STEP3_CMD+=(--gencode_version "${GENCODE_VERSION}")

    if ! "${STEP3_CMD[@]}"; then
        print_error 3 "Failed to perform spatial clustering"
    fi
    
    print_step_footer 3
else
    print_step_skip 3 "Spatial Clustering"
fi

# -----------------------------------------------------------------------------
# Step 4: Biotype Enrichment
# -----------------------------------------------------------------------------

if [ ${START_FROM} -le 4 ]; then
    print_step_header 4 "Biotype Enrichment Analysis"
    
    STEP4_CMD=(
        bash "${SCRIPT_DIR}/04_biotype_enrichment.sh"
        --spatial_dir "${EXPERIMENT_DIR}/spatial_analysis"
        --output_dir "${EXPERIMENT_DIR}/global_biotype_enrichment"
        --min_count 500
    )
    [ -n "${MODEL_LABEL}" ]     && STEP4_CMD+=(--model_label "${MODEL_LABEL}")
    [ -n "${GENCODE_VERSION}" ] && STEP4_CMD+=(--gencode_version "${GENCODE_VERSION}")

    if ! "${STEP4_CMD[@]}"; then
        print_error 4 "Failed to analyze biotype enrichment"
    fi
    
    print_step_footer 4
else
    print_step_skip 4 "Biotype Enrichment Analysis"
fi

# -----------------------------------------------------------------------------
# Step 5: GENCODE Comparison (Optional)
# -----------------------------------------------------------------------------

if [ ${INCLUDE_GENCODE} -eq 1 ]; then
    if [ ${START_FROM} -le 5 ]; then
        print_step_header 5 "GENCODE Version Comparison"
        
        GENCODE_ARGS=(
            --experiment_dir "${EXPERIMENT_DIR}"
            --old-version "${OLD_VERSION}"
            --new-version "${NEW_VERSION}"
            --old-fasta "${OLD_FASTA}"
            --old-pc "${OLD_PC}"
            --old-lnc "${OLD_LNC}"
            --new-fasta "${NEW_FASTA}"
            --new-pc "${NEW_PC}"
            --new-lnc "${NEW_LNC}"
        )
        [ -n "${COMPARISON_OUTPUT_DIR}" ] && GENCODE_ARGS+=(--comparison-output-dir "${COMPARISON_OUTPUT_DIR}")
        
        if ! "${SCRIPT_DIR}/05_gencode_comparison.sh" "${GENCODE_ARGS[@]}"; then
            print_error 5 "Failed to perform GENCODE comparison"
        fi
        
        print_step_footer 5
    else
        print_step_skip 5 "GENCODE Version Comparison"
    fi
else
    echo ""
    echo "  Skipping Step 5 (GENCODE comparison not requested)"
    echo ""
fi

# -----------------------------------------------------------------------------
# Step 6: Generate Summary Report
# -----------------------------------------------------------------------------

if [ ${START_FROM} -le 6 ]; then
    print_step_header 6 "Generate Summary Report"
    
    if ! bash "${SCRIPT_DIR}/06_generate_summary_report.sh" \
        --experiment_dir "${EXPERIMENT_DIR}"; then
        print_error 6 "Failed to generate summary report"
    fi
    
    print_step_footer 6
else
    print_step_skip 6 "Generate Summary Report"
fi

# -----------------------------------------------------------------------------
# Pipeline Complete
# -----------------------------------------------------------------------------

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
MINUTES=$((TOTAL_DURATION / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "=============================================="
echo " PIPELINE COMPLETE!"
echo "=============================================="
echo ""
echo "Total execution time: ${MINUTES}m ${SECONDS}s"
echo ""
echo "Generated outputs in: ${EXPERIMENT_DIR}/"
echo ""
echo " Key Results:"
echo "  - Performance metrics:  cv_evaluation_results.json"
echo "  - Hard cases:           evaluation_csvs/hard_cases.csv"
echo "  - UMAP visualizations:  umap_visualizations/"
echo "  - Spatial analysis:     spatial_analysis/"
echo "  - Biotype enrichment:   biotype_enrichment/"
echo "  - Attention analysis:   attention_analysis/"
[ ${INCLUDE_GENCODE} -eq 1 ] && echo "  - GENCODE comparison:   gencode_novelty_analysis/"
echo ""
echo " Summary Report: ${EXPERIMENT_DIR}/ANALYSIS_SUMMARY.md"
echo ""
echo "To view the report:"
echo "  cat ${EXPERIMENT_DIR}/ANALYSIS_SUMMARY.md"
echo ""