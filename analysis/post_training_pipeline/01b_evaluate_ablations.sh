#!/bin/bash
# evaluate_ablations.sh
# Discovers and evaluates all ablation variants under an ablations/ directory.
# Reads shared config (fasta paths, biotype CSV) from the parent experiment config.
#
# Usage:
#   bash 01b_evaluate_ablations.sh \
#       --ablations_dir <path/to/ablations> \
#       --eval_script   <path/to/01_evaluate_cv_folds.sh> \
#       --biotype_csv   <path/to/biotype.csv> \
#       --gencode_version v47|v49 \
#       [--n_folds 5] \
#       [--device cuda:0]

set -euo pipefail

ABLATIONS_DIR=""
EVAL_SCRIPT=""
BIOTYPE_CSV=""
GENCODE_VERSION=""
N_FOLDS=5
DEVICE="cuda:0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --ablations_dir)    ABLATIONS_DIR="$2";    shift 2 ;;
        --eval_script)      EVAL_SCRIPT="$2";      shift 2 ;;
        --biotype_csv)      BIOTYPE_CSV="$2";      shift 2 ;;
        --gencode_version)  GENCODE_VERSION="$2";  shift 2 ;;
        --n_folds)          N_FOLDS="$2";          shift 2 ;;
        --device)           DEVICE="$2";           shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Validate required args
for var in ABLATIONS_DIR EVAL_SCRIPT BIOTYPE_CSV GENCODE_VERSION; do
    [[ -z "${!var}" ]] && { echo "ERROR: --${var,,} required"; exit 1; }
done
[[ -d "$ABLATIONS_DIR" ]] || { echo "ERROR: $ABLATIONS_DIR not found"; exit 1; }
[[ -f "$EVAL_SCRIPT"   ]] || { echo "ERROR: $EVAL_SCRIPT not found";   exit 1; }

# Read fasta paths from parent experiment config (one level above ablations/)
PARENT_CONFIG="$(dirname "$ABLATIONS_DIR")/config.json"
[[ -f "$PARENT_CONFIG" ]] || { echo "ERROR: parent config not found at $PARENT_CONFIG"; exit 1; }

LNC_TEST_FASTA=$(python3 -c "import json,sys; c=json.load(open('$PARENT_CONFIG')); print(c['data']['lnc_test_fasta'])")
PC_TEST_FASTA=$(python3  -c "import json,sys; c=json.load(open('$PARENT_CONFIG')); print(c['data']['pc_test_fasta'])")

echo "=== Ablation evaluation ==="
echo "  Ablations dir  : $ABLATIONS_DIR"
echo "  Eval script    : $EVAL_SCRIPT"
echo "  GENCODE version: $GENCODE_VERSION"
echo "  lnc test fasta : $LNC_TEST_FASTA"
echo "  pc  test fasta : $PC_TEST_FASTA"
echo "  Biotype CSV    : $BIOTYPE_CSV"
echo "  Device         : $DEVICE"
echo "==========================="

n_total=0; n_skipped=0; n_done=0; n_failed=0

for variant_dir in "$ABLATIONS_DIR"/*/; do
    [[ -d "$variant_dir" ]] || continue
    variant=$(basename "$variant_dir")

    if [[ ! -f "$variant_dir/fold_results.json" ]]; then
        echo "  [SKIP] $variant — training incomplete"
        ((n_skipped++)); ((n_total++)); continue
    fi

    if [[ -f "$variant_dir/evaluation_csv/test_predictions.csv" ]]; then
        echo "  [SKIP] $variant — already evaluated"
        ((n_skipped++)); ((n_total++)); continue
    fi

    echo "  [RUN]  $variant"
    if bash "$EVAL_SCRIPT" \
        --experiment_dir    "$variant_dir" \
        --config            "$variant_dir/config.json" \
        --biotype_csv       "$BIOTYPE_CSV" \
        --lnc_test_fasta    "$LNC_TEST_FASTA" \
        --pc_test_fasta     "$PC_TEST_FASTA" \
        --n_folds           "$N_FOLDS" \
        --device            "$DEVICE" \
        --model_label       "$variant" \
        --gencode_version   "$GENCODE_VERSION" \
        >> "$variant_dir/eval.log" 2>&1; then
        echo "  [OK]   $variant"
        ((n_done++))
    else
        echo "  [FAIL] $variant — see $variant_dir/eval.log"
        ((n_failed++))
    fi
    ((n_total++))
done

echo ""
echo "=== Summary: $n_total variants | $n_done evaluated | $n_skipped skipped | $n_failed failed ==="