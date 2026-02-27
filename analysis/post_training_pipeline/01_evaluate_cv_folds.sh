#!/bin/bash
set -e

echo "=========================================="
echo "Step 1: Evaluate CV Folds"
echo "=========================================="

# Default values
EXPERIMENT_DIR=""
CONFIG=""
BIOTYPE_CSV=""
LNC_TEST_FASTA=""
PC_TEST_FASTA=""
N_FOLDS=5
DEVICE="cuda:0"
MODEL_LABEL=""
GENCODE_VERSION=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment_dir)   EXPERIMENT_DIR="$2"; shift 2 ;;
        --config)           CONFIG="$2";         shift 2 ;;
        --biotype_csv)      BIOTYPE_CSV="$2";    shift 2 ;;
        --test_lnc_fasta)   LNC_TEST_FASTA="$2"; shift 2 ;;
        --test_pc_fasta)    PC_TEST_FASTA="$2";  shift 2 ;;
        --n_folds)          N_FOLDS="$2";        shift 2 ;;
        --device)           DEVICE="$2";         shift 2 ;;
        --model_label)      MODEL_LABEL="$2";    shift 2 ;;
        --gencode_version)  GENCODE_VERSION="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --experiment_dir DIR --config FILE --biotype_csv FILE [--n_folds 5] [--device cuda:0] [--model_label LABEL] [--gencode_version VERSION]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$EXPERIMENT_DIR" ] || [ -z "$CONFIG" ]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: $0 --experiment_dir DIR --config FILE [--biotype_csv FILE]"
    exit 1
fi

# Check experiment directory exists
if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "ERROR: Experiment directory not found: $EXPERIMENT_DIR"
    exit 1
fi

# Check config exists
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

# Infer gencode version from experiment dir if not explicitly provided
# Matches _g47 or _g49 suffixes → produces "v47" / "v49"
if [ -z "$GENCODE_VERSION" ]; then
    GENCODE_VERSION=$(echo "$EXPERIMENT_DIR" | grep -oP '(?<=_g)\d+' | head -1)
    [ -n "$GENCODE_VERSION" ] && GENCODE_VERSION="v${GENCODE_VERSION}"
fi

echo "Configuration:"
echo "  Experiment:      $EXPERIMENT_DIR"
echo "  Config:          $CONFIG"
echo "  N_folds:         $N_FOLDS"
echo "  Device:          $DEVICE"
echo "  Model label:     ${MODEL_LABEL:-none}"
echo "  GENCODE version: ${GENCODE_VERSION:-unknown}"
echo ""

# Get script directory
SCRIPT_DIR="$(dirname "$0")/scripts"
ATTENTION_SCRIPT_DIR="${DATA_ROOT}/analysis/attention"

# Detect model architecture from config
echo "Detecting model architecture..."

# Check if config has 'architecture' field or feature-specific fields
if grep -q '"beta_vae_features_attn"' "$CONFIG" 2>/dev/null; then
    ARCHITECTURE="beta_vae_features_attn"
    EVAL_SCRIPT="evaluate_cv_fold_features.py"
    HAS_ATTENTION=1
elif grep -q '"te_features_csv"' "$CONFIG" 2>/dev/null; then
    ARCHITECTURE="beta_vae_features"
    EVAL_SCRIPT="evaluate_cv_fold_features.py"
    HAS_ATTENTION=0
else
    ARCHITECTURE="contrastive"
    EVAL_SCRIPT="evaluate_cv_fold.py"
    HAS_ATTENTION=0
fi

echo "  Architecture: $ARCHITECTURE"
echo ""

if [ ! -f "$SCRIPT_DIR/$EVAL_SCRIPT" ]; then
    echo "ERROR: $EVAL_SCRIPT not found in $SCRIPT_DIR"
    exit 1
fi

CMD=(
    python "$SCRIPT_DIR/$EVAL_SCRIPT"
    --config "$CONFIG"
    --experiment_dir "$EXPERIMENT_DIR"
    --biotype_csv "$BIOTYPE_CSV"
    --test_lnc_fasta "$LNC_TEST_FASTA"
    --test_pc_fasta "$PC_TEST_FASTA"
    --n_folds "$N_FOLDS"
    --device "$DEVICE"
    --batch_size 512 # default batch size for evaluation; adjust if needed
    --extract_all_folds
    --generate_hard_case_csvs
)

EXPECTED_OUTPUTS=(
    "$EXPERIMENT_DIR/cv_evaluation_results.json"
    "$EXPERIMENT_DIR/cv_fold_results.csv"
    "$EXPERIMENT_DIR/embeddings_all_folds.npz"
    "$EXPERIMENT_DIR/evaluation_csvs/all_sample_predictions.csv"
    "$EXPERIMENT_DIR/evaluation_csvs/hard_cases.csv"
    "$EXPERIMENT_DIR/performance_figures/roc_pr_curves.png"
    # if attention is present, also expect attention .npz files for each fold
    "$EXPERIMENT_DIR/fold_attention/"
    "$EXPERIMENT_DIR/attention_analysis/"
)

echo "Running evaluation..."
echo "Command: ${CMD[*]}"
echo ""

"${CMD[@]}"

if [ "${HAS_ATTENTION}" = "1" ]; then
    MISSING_NPZ=0
    for fold_idx in $(seq 0 $((N_FOLDS - 1))); do
        if [ ! -f "${EXPERIMENT_DIR}/fold_attention/fold_${fold_idx}_attn.npz" ]; then
            MISSING_NPZ=1
            break
        fi
    done

    if [ "${MISSING_NPZ}" = "1" ]; then
        echo "Attention .npz files incomplete — running extraction..."
        python "$ATTENTION_SCRIPT_DIR/extract_attention_from_folds.py" \
            --experiment_dir "$EXPERIMENT_DIR" \
            --config "$CONFIG" \
            || { echo "Attention extraction failed"; exit 1; }
    else
        echo "Attention .npz files already present — skipping extraction."
    fi

    ATTN_OUT="${EXPERIMENT_DIR}/attention_analysis"
    echo "Running attention analysis..."

    ATTN_CMD=(
        python "$SCRIPT_DIR/analyze_attention.py"
        --attn_dir "${EXPERIMENT_DIR}/fold_attention"
        --output_dir "$ATTN_OUT"
    )
    [ -n "$GENCODE_VERSION" ] && ATTN_CMD+=(--gencode_version "$GENCODE_VERSION")
    [ -n "$MODEL_LABEL" ]     && ATTN_CMD+=(--model_label "$MODEL_LABEL")

    "${ATTN_CMD[@]}" || { echo "Attention analysis failed"; exit 1; }
    echo "Attention analysis complete → $ATTN_OUT"
fi

# Length-Class interaction

CMD=(
    python "$SCRIPT_DIR/length_class_interaction.py"
    --predictions_csv "$EXPERIMENT_DIR/evaluation_csvs/all_sample_predictions.csv"
    --output_dir "$EXPERIMENT_DIR/length_class_analysis/"
)

echo "Running length-class interaction analysis..."
"${CMD[@]}"
echo "Length-class interaction analysis complete."

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo " Step 1 Complete"
    echo "=========================================="
    echo ""
    echo "Outputs:"
    for output in "${EXPECTED_OUTPUTS[@]}"; do
        if [ -e "$output" ]; then
            echo "   $output"
        else
            echo "   $output (not found)"
        fi
    done
    echo ""
        echo "Next step:"
        echo "  bash 02_generate_umap.sh --experiment_dir $EXPERIMENT_DIR"
else
    echo ""
    echo " Step 1 Failed (exit code: $exit_code)"
    exit $exit_code
fi