#!/bin/bash

# =============================================================================
# Dry-Run Validator for run_all.sh
# =============================================================================
# Simulates run_all.sh execution to verify:
# - All scripts exist and are executable
# - Arguments are correctly passed to each step
# - File paths are valid
# - No missing dependencies
#
# Usage (same arguments as run_all.sh):
#   bash dry_run_check.sh \
#     --experiment_dir experiments/beta_vae_g47 \
#     --config configs/beta_vae.json \
#     --biotype_csv data/processed/gencode47_dataset_biotypes_cdhit90.csv \
#     --n_folds 5
# =============================================================================

set +e  # Don't exit on errors during validation

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "                    DRY-RUN VALIDATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# -----------------------------------------------------------------------------
# Parse Arguments (identical to run_all.sh)
# -----------------------------------------------------------------------------

EXPERIMENT_DIR=""
CONFIG=""
BIOTYPE_CSV=""
N_FOLDS=5
INCLUDE_GENCODE=0
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
        --experiment_dir) EXPERIMENT_DIR="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --biotype_csv) BIOTYPE_CSV="$2"; shift 2 ;;
        --n_folds) N_FOLDS="$2"; shift 2 ;;
        --include_gencode) INCLUDE_GENCODE=1; shift 1 ;;
        --old-version) OLD_VERSION="$2"; shift 2 ;;
        --new-version) NEW_VERSION="$2"; shift 2 ;;
        --old-fasta) OLD_FASTA="$2"; shift 2 ;;
        --old-pc) OLD_PC="$2"; shift 2 ;;
        --old-lnc) OLD_LNC="$2"; shift 2 ;;
        --new-fasta) NEW_FASTA="$2"; shift 2 ;;
        --new-pc) NEW_PC="$2"; shift 2 ;;
        --new-lnc) NEW_LNC="$2"; shift 2 ;;
        --comparison-output-dir) COMPARISON_OUTPUT_DIR="$2"; shift 2 ;;
        *) echo -e "${RED}❌ Unknown argument: $1${NC}"; exit 1 ;;
    esac
done

SCRIPT_DIR="analysis/post_training_pipeline"

# Set default comparison output dir if not provided
if [ ${INCLUDE_GENCODE} -eq 1 ] && [ -z "${COMPARISON_OUTPUT_DIR}" ]; then
    COMPARISON_OUTPUT_DIR="resources/gencode_comparison_v${OLD_VERSION}_v${NEW_VERSION}"
fi

# -----------------------------------------------------------------------------
# Section 1: Validate Required Arguments
# -----------------------------------------------------------------------------

echo -e "${BOLD}[1/7] Validating Required Arguments${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -z "$EXPERIMENT_DIR" ]; then
    echo -e "${RED}❌ Missing: --experiment_dir${NC}"
    ((ERRORS++))
else
    echo -e "${GREEN}✓${NC} --experiment_dir: $EXPERIMENT_DIR"
fi

if [ -z "$CONFIG" ]; then
    echo -e "${RED}❌ Missing: --config${NC}"
    ((ERRORS++))
else
    echo -e "${GREEN}✓${NC} --config: $CONFIG"
fi

if [ -z "$BIOTYPE_CSV" ]; then
    echo -e "${RED}❌ Missing: --biotype_csv${NC}"
    ((ERRORS++))
else
    echo -e "${GREEN}✓${NC} --biotype_csv: $BIOTYPE_CSV"
fi

echo -e "${GREEN}✓${NC} --n_folds: $N_FOLDS"

echo ""

# -----------------------------------------------------------------------------
# Section 2: Validate Input Files
# -----------------------------------------------------------------------------

echo -e "${BOLD}[2/7] Validating Input Files${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}❌ Config file not found: $CONFIG${NC}"
    ((ERRORS++))
else
    echo -e "${GREEN}✓${NC} Config file exists"
fi

if [ ! -f "$BIOTYPE_CSV" ]; then
    echo -e "${RED}❌ Biotype CSV not found: $BIOTYPE_CSV${NC}"
    ((ERRORS++))
else
    echo -e "${GREEN}✓${NC} Biotype CSV exists"
fi

if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo -e "${YELLOW}⚠️  Experiment directory will be created: $EXPERIMENT_DIR${NC}"
    ((WARNINGS++))
else
    echo -e "${GREEN}✓${NC} Experiment directory exists"
fi

echo ""

# -----------------------------------------------------------------------------
# Section 3: Validate GENCODE Arguments (if applicable)
# -----------------------------------------------------------------------------

echo -e "${BOLD}[3/7] Validating GENCODE Comparison Arguments${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ ${INCLUDE_GENCODE} -eq 1 ]; then
    echo -e "${BLUE}GENCODE comparison enabled${NC}"
    echo ""
    
    GENCODE_REQUIRED=("OLD_VERSION" "NEW_VERSION" "OLD_FASTA" "OLD_PC" "OLD_LNC" "NEW_FASTA" "NEW_PC" "NEW_LNC")
    for arg in "${GENCODE_REQUIRED[@]}"; do
        if [ -z "${!arg}" ]; then
            echo -e "${RED}❌ Missing GENCODE argument: --${arg,,}${NC}"
            ((ERRORS++))
        else
            echo -e "${GREEN}✓${NC} --${arg,,}: ${!arg}"
        fi
    done
    
    echo ""
    
    # Check FASTA files exist
    for FASTA in "$OLD_FASTA" "$OLD_PC" "$OLD_LNC" "$NEW_FASTA" "$NEW_PC" "$NEW_LNC"; do
        if [ -n "$FASTA" ] && [ ! -f "$FASTA" ]; then
            echo -e "${RED}❌ FASTA file not found: $FASTA${NC}"
            ((ERRORS++))
        fi
    done
    
    if [ $ERRORS -eq 0 ]; then
        echo -e "${GREEN}✓${NC} All GENCODE FASTA files exist"
    fi
else
    echo -e "${YELLOW}GENCODE comparison disabled (use --include_gencode to enable)${NC}"
fi

echo ""

# -----------------------------------------------------------------------------
# Section 4: Validate Pipeline Scripts
# -----------------------------------------------------------------------------

echo -e "${BOLD}[4/7] Validating Pipeline Scripts${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

SCRIPTS=(
    "01_evaluate_cv_folds.sh"
    "02_generate_umap.sh"
    "03_spatial_clustering.sh"
    "04_biotype_enrichment.sh"
    "05_gencode_comparison.sh"
    "06_generate_summary_report.sh"
)

for script in "${SCRIPTS[@]}"; do
    SCRIPT_PATH="$SCRIPT_DIR/$script"
    if [ ! -f "$SCRIPT_PATH" ]; then
        echo -e "${RED}❌ Script not found: $SCRIPT_PATH${NC}"
        ((ERRORS++))
    elif [ ! -x "$SCRIPT_PATH" ]; then
        echo -e "${YELLOW}⚠️  Script not executable: $SCRIPT_PATH${NC}"
        echo -e "   Run: chmod +x $SCRIPT_PATH"
        ((WARNINGS++))
    else
        echo -e "${GREEN}✓${NC} $script"
    fi
done

echo ""

# -----------------------------------------------------------------------------
# Section 5: Validate Python Scripts (symlinks)
# -----------------------------------------------------------------------------

echo -e "${BOLD}[5/7] Validating Python Scripts${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

PYTHON_SCRIPTS=(
    "evaluate_cv_fold.py:experiments/evaluation/"
    "visualize_embeddings.py:analysis/embeddings/"
    "analyze_hardcase_spatial_patterns.py:analysis/spatial/"
    "analyze_global_biotype_enrichment.py:analysis/biotypes/"
    "compare_gencode_versions.py:scripts/"
    "gencode_novelty_embeddings.py:scripts/"
)

for item in "${PYTHON_SCRIPTS[@]}"; do
    SCRIPT_NAME="${item%%:*}"
    EXPECTED_LOCATION="${item##*:}"
    
    # Check in scripts/ directory (symlink)
    SYMLINK_PATH="$SCRIPT_DIR/scripts/$SCRIPT_NAME"
    
    if [ -L "$SYMLINK_PATH" ]; then
        TARGET=$(readlink "$SYMLINK_PATH")
        if [ -f "$TARGET" ]; then
            echo -e "${GREEN}✓${NC} $SCRIPT_NAME (symlink → $TARGET)"
        else
            echo -e "${RED}❌ Broken symlink: $SYMLINK_PATH → $TARGET${NC}"
            ((ERRORS++))
        fi
    elif [ -f "$SYMLINK_PATH" ]; then
        echo -e "${GREEN}✓${NC} $SCRIPT_NAME (direct file)"
    else
        # Check original location
        ORIGINAL_PATH="$EXPECTED_LOCATION$SCRIPT_NAME"
        if [ -f "$ORIGINAL_PATH" ]; then
            echo -e "${YELLOW}⚠️  $SCRIPT_NAME exists at $ORIGINAL_PATH but symlink missing${NC}"
            echo -e "   Run: ln -s ../../$ORIGINAL_PATH $SYMLINK_PATH"
            ((WARNINGS++))
        else
            echo -e "${RED}❌ $SCRIPT_NAME not found (expected at $ORIGINAL_PATH)${NC}"
            ((ERRORS++))
        fi
    fi
done

echo ""

# -----------------------------------------------------------------------------
# Section 6: Simulate Script Calls
# -----------------------------------------------------------------------------

echo -e "${BOLD}[6/7] Simulating Script Calls${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo -e "${BLUE}Step 1: Evaluate CV Folds${NC}"
echo "  Command: 01_evaluate_cv_folds.sh \\"
echo "    --experiment_dir \"$EXPERIMENT_DIR\" \\"
echo "    --config \"$CONFIG\" \\"
echo "    --n_folds \"$N_FOLDS\""
echo ""

echo -e "${BLUE}Step 2: Generate UMAP${NC}"
echo "  Command: 02_generate_umap.sh \\"
echo "    --experiment_dir \"$EXPERIMENT_DIR\" \\"
echo "    --biotype_csv \"$BIOTYPE_CSV\" \\"
echo "    --n_folds \"$N_FOLDS\""
echo ""

echo -e "${BLUE}Step 3: Spatial Clustering${NC}"
echo "  Command: 03_spatial_clustering.sh \\"
echo "    --experiment_dir \"$EXPERIMENT_DIR\" \\"
echo "    --n_folds \"$N_FOLDS\""
echo ""

echo -e "${BLUE}Step 4: Biotype Enrichment${NC}"
echo "  Command: 04_biotype_enrichment.sh \\"
echo "    --experiment_dir \"$EXPERIMENT_DIR\" \\"
echo "    --biotype_csv \"$BIOTYPE_CSV\""
echo ""

if [ ${INCLUDE_GENCODE} -eq 1 ]; then
    echo -e "${BLUE}Step 5: GENCODE Comparison${NC}"
    echo "  Command: 05_gencode_comparison.sh \\"
    echo "    --experiment_dir \"$EXPERIMENT_DIR\" \\"
    echo "    --old-version \"$OLD_VERSION\" \\"
    echo "    --new-version \"$NEW_VERSION\" \\"
    echo "    --old-fasta \"$OLD_FASTA\" \\"
    echo "    --old-pc \"$OLD_PC\" \\"
    echo "    --old-lnc \"$OLD_LNC\" \\"
    echo "    --new-fasta \"$NEW_FASTA\" \\"
    echo "    --new-pc \"$NEW_PC\" \\"
    echo "    --new-lnc \"$NEW_LNC\""
    [ -n "$COMPARISON_OUTPUT_DIR" ] && echo "    --comparison-output-dir \"$COMPARISON_OUTPUT_DIR\""
    echo ""
else
    echo -e "${YELLOW}Step 5: GENCODE Comparison (SKIPPED)${NC}"
    echo ""
fi

echo -e "${BLUE}Step 6: Generate Summary Report${NC}"
echo "  Command: 06_generate_summary_report.sh \\"
echo "    --experiment_dir \"$EXPERIMENT_DIR\""
echo ""

# -----------------------------------------------------------------------------
# Section 7: Check Expected Outputs
# -----------------------------------------------------------------------------

echo -e "${BOLD}[7/7] Checking Expected Output Locations${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "After Step 1, expecting:"
echo "  - $EXPERIMENT_DIR/cv_evaluation_results.json"
echo "  - $EXPERIMENT_DIR/evaluation_csvs/hard_cases.csv"
echo "  - $EXPERIMENT_DIR/embeddings_all_folds.npz"
echo ""

echo "After Step 2, expecting:"
echo "  - $EXPERIMENT_DIR/umap_visualizations/fold_0/umap_embeddings.csv"
echo "  - $EXPERIMENT_DIR/umap_visualizations/fold_0/umap_*.png"
echo ""

echo "After Step 3, expecting:"
echo "  - $EXPERIMENT_DIR/spatial_analysis/fold_0/samples_with_regions.csv"
echo ""

echo "After Step 4, expecting:"
echo "  - $EXPERIMENT_DIR/biotype_enrichment/global_biotype_enrichment.csv"
echo ""

if [ ${INCLUDE_GENCODE} -eq 1 ]; then
    echo "After Step 5, expecting:"
    echo "  - $COMPARISON_OUTPUT_DIR/gencode_v${OLD_VERSION}_v${NEW_VERSION}_comparison.tsv"
    echo "  - $COMPARISON_OUTPUT_DIR/gencode.v${NEW_VERSION}.new_with_class_transcripts.fa"
    echo "  - $EXPERIMENT_DIR/gencode_novelty_analysis/*.png"
    echo ""
fi

echo "After Step 6, expecting:"
echo "  - $EXPERIMENT_DIR/ANALYSIS_SUMMARY.md"
echo ""

# -----------------------------------------------------------------------------
# Final Summary
# -----------------------------------------------------------------------------

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "                      VALIDATION SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}${BOLD}✓ ALL CHECKS PASSED${NC}"
    echo ""
    echo "You can now run:"
    echo "  bash analysis/post_training_pipeline/run_all.sh [same arguments]"
    EXIT_CODE=0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}${BOLD}⚠️  PASSED WITH WARNINGS${NC}"
    echo ""
    echo "Warnings: $WARNINGS"
    echo ""
    echo "You can run the pipeline, but consider fixing warnings first."
    EXIT_CODE=0
else
    echo -e "${RED}${BOLD}❌ VALIDATION FAILED${NC}"
    echo ""
    echo "Errors: $ERRORS"
    echo "Warnings: $WARNINGS"
    echo ""
    echo "Please fix the errors above before running the pipeline."
    EXIT_CODE=1
fi

echo ""
exit $EXIT_CODE