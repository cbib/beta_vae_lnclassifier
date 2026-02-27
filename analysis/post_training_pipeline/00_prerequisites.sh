#!/bin/bash
set -e

echo "=========================================="
echo "Post-Training Pipeline: Prerequisites Check"
echo "=========================================="

# Parse arguments
EXPERIMENT_DIR=""
CONFIG=""
BIOTYPE_CSV=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment_dir)
            EXPERIMENT_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --biotype_csv)
            BIOTYPE_CSV="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$EXPERIMENT_DIR" ]; then
    echo "ERROR: --experiment_dir required"
    exit 1
fi

if [ -z "$CONFIG" ]; then
    echo "WARNING: --config not provided, will skip config check"
fi

if [ -z "$BIOTYPE_CSV" ]; then
    echo "WARNING: --biotype_csv not provided, will skip biotype check"
fi

echo ""
echo "Checking: $EXPERIMENT_DIR"
echo ""

ERRORS=0
WARNINGS=0

# Check experiment directory exists
if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo " Experiment directory not found: $EXPERIMENT_DIR"
    ERRORS=$((ERRORS + 1))
else
    echo " Experiment directory exists"
fi

# Check models directory
if [ ! -d "$EXPERIMENT_DIR/models" ]; then
    echo " Models directory not found: $EXPERIMENT_DIR/models"
    ERRORS=$((ERRORS + 1))
else
    echo " Models directory exists"
    
    # Check for fold checkpoints
    echo ""
    echo "Checking fold checkpoints:"
    for fold in {0..4}; do
        checkpoint="$EXPERIMENT_DIR/models/fold_${fold}_best.pt"
        if [ -f "$checkpoint" ]; then
            echo "   fold_${fold}_best.pt"
        else
            echo "   fold_${fold}_best.pt MISSING"
            ERRORS=$((ERRORS + 1))
        fi
    done
fi

# Check config file
echo ""
if [ -n "$CONFIG" ]; then
    if [ -f "$CONFIG" ]; then
        echo " Config file exists: $CONFIG"
    else
        echo " Config file not found: $CONFIG"
        ERRORS=$((ERRORS + 1))
    fi
fi

# Check biotype CSV
if [ -n "$BIOTYPE_CSV" ]; then
    if [ -f "$BIOTYPE_CSV" ]; then
        echo " Biotype CSV exists: $BIOTYPE_CSV"
        
        # Check it has content
        line_count=$(wc -l < "$BIOTYPE_CSV")
        if [ $line_count -gt 1 ]; then
            echo "  Lines: $line_count"
        else
            echo "  ⚠ WARNING: Biotype CSV appears empty"
            WARNINGS=$((WARNINGS + 1))
        fi
    else
        echo " Biotype CSV not found: $BIOTYPE_CSV"
        ERRORS=$((ERRORS + 1))
    fi
fi

# Check analysis scripts exist
echo ""
echo "Checking analysis scripts:"
SCRIPT_DIR="$(dirname "$0")/scripts"

scripts=(
    "evaluate_cv_fold.py"
    "visualize_embeddings.py"
    "analyze_hardcase_spatial_patterns.py"
    "analyze_global_biotype_enrichment.py"
)

for script in "${scripts[@]}"; do
    if [ -f "$SCRIPT_DIR/$script" ]; then
        echo "   $script"
    else
        echo "   $script MISSING"
        WARNINGS=$((WARNINGS + 1))
    fi
done

# Summary
echo ""
echo "=========================================="
echo "PREREQUISITE CHECK SUMMARY"
echo "=========================================="
echo "Errors:   $ERRORS"
echo "Warnings: $WARNINGS"
echo ""

if [ $ERRORS -gt 0 ]; then
    echo " FAILED: Please fix errors before proceeding"
    exit 1
elif [ $WARNINGS -gt 0 ]; then
    echo "  PASSED WITH WARNINGS: Some optional components missing"
    exit 0
else
    echo " PASSED: All prerequisites met"
    exit 0
fi