#!/bin/bash
set -e

echo "=========================================="
echo "Step 2: Generate UMAP Visualizations"
echo "=========================================="

# Default values
EXPERIMENT_DIR=""
LNC_FASTA=""
PC_FASTA=""
BIOTYPE_CSV=""
N_NEIGHBORS=30
MIN_DIST=0.1
METRIC="euclidean"
MODEL_LABEL=""
GENCODE_VERSION=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment_dir)
            EXPERIMENT_DIR="$2"
            shift 2
            ;;
        --lnc_fasta)
            LNC_FASTA="$2"
            shift 2
            ;;
        --pc_fasta)
            PC_FASTA="$2"
            shift 2
            ;;
        --biotype_csv)
            BIOTYPE_CSV="$2"
            shift 2
            ;;
        --n_neighbors)
            N_NEIGHBORS="$2"
            shift 2
            ;;
        --min_dist)
            MIN_DIST="$2"
            shift 2
            ;;
        --metric)
            METRIC="$2"
            shift 2
            ;;
        --model_label)
            MODEL_LABEL="$2"
            shift 2
            ;;
        --gencode_version)
            GENCODE_VERSION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --experiment_dir DIR --biotype_csv FILE [--n_neighbors 30] [--min_dist 0.1] [--metric euclidean] [--model_label LABEL] [--gencode_version VERSION]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$EXPERIMENT_DIR" ]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: $0 --experiment_dir DIR --biotype_csv FILE"
    exit 1
fi

# Infer gencode version from experiment dir if not provided
if [ -z "$GENCODE_VERSION" ]; then
    GENCODE_VERSION=$(echo "$EXPERIMENT_DIR" | grep -oP '(?<=_g)\d+' | head -1)
    [ -n "$GENCODE_VERSION" ] && GENCODE_VERSION="v${GENCODE_VERSION}"
fi

MODEL_TYPE=$(python3 -c "
import json
try:
    with open('$CONFIG') as f:
        config = json.load(f)
    if 'te_features_csv' in config.get('data', {}):
        print('features')
    else:
        print('biotype')
except:
    print('unknown')
")

echo "Detected model type: $MODEL_TYPE"

# Derived paths
EMBEDDINGS="$EXPERIMENT_DIR/embeddings_all_folds.npz"
HARD_CASES="$EXPERIMENT_DIR/evaluation_csvs/hard_cases.csv"
OUTPUT_DIR="$EXPERIMENT_DIR/umap_visualizations"

# Check inputs exist
if [ ! -f "$EMBEDDINGS" ]; then
    echo "ERROR: Embeddings not found: $EMBEDDINGS"
    echo "Please run 01_evaluate_cv_folds.sh first"
    exit 1
fi

if [ ! -f "$HARD_CASES" ]; then
    echo "WARNING: Hard cases CSV not found: $HARD_CASES"
    echo "Proceeding without hard case highlighting"
    HARD_CASES=""
fi

echo "Configuration:"
echo "  Embeddings: $EMBEDDINGS"
echo "  LncRNA FASTA: ${LNC_FASTA:-none}"
echo "  Protein-coding FASTA: ${PC_FASTA:-none}"
echo "  Hard cases: ${HARD_CASES:-none}"
echo "  Biotype CSV: ${BIOTYPE_CSV:-optional}"
echo "  Output: $OUTPUT_DIR"
echo "  UMAP params: n_neighbors=$N_NEIGHBORS, min_dist=$MIN_DIST, metric=$METRIC"
echo "  Model label: ${MODEL_LABEL:-none}"
echo "  GENCODE version: ${GENCODE_VERSION:-none}"
echo ""

# Get script directory
SCRIPT_DIR="$(dirname "$0")/scripts"

# Check script exists
if [ ! -f "$SCRIPT_DIR/visualize_embeddings.py" ]; then
    echo "ERROR: visualize_embeddings.py not found in $SCRIPT_DIR"
    exit 1
fi

# Build command as array (avoids special character quoting issues with eval)
CMD=(
    python "$SCRIPT_DIR/visualize_embeddings.py"
    --embeddings "$EMBEDDINGS"
    --experiment_dir "$EXPERIMENT_DIR"
    --output_dir "$OUTPUT_DIR"
    --n_neighbors "$N_NEIGHBORS"
    --min_dist "$MIN_DIST"
    --metric "$METRIC"
    --per_fold
)

[ -n "$HARD_CASES" ]      && CMD+=(--hard_cases "$HARD_CASES")
[ -n "$LNC_FASTA" ]       && CMD+=(--lnc_fasta "$LNC_FASTA")
[ -n "$PC_FASTA" ]        && CMD+=(--pc_fasta "$PC_FASTA")
[ -n "$BIOTYPE_CSV" ]     && CMD+=(--biotype_csv "$BIOTYPE_CSV")
[ -n "$MODEL_LABEL" ]     && CMD+=(--model_label "$MODEL_LABEL")
[ -n "$GENCODE_VERSION" ] && CMD+=(--gencode_version "$GENCODE_VERSION")

# Run UMAP generation
echo "Generating UMAP visualizations (per fold)..."
echo ""
echo "Command: ${CMD[*]}"
echo ""

"${CMD[@]}"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo " Step 2 Complete"
    echo "=========================================="
    echo ""
    echo "Outputs:"
    echo "  $OUTPUT_DIR/fold_0/"
    echo "    - umap_all_samples.png"
    echo "    - umap_hard_cases.png"
    
    if [ "$MODEL_TYPE" = "features" ]; then
        echo "    - umap_te_density.png         (features model)"
        echo "    - umap_nonb_density.png       (features model)"
        echo "    - umap_combined_features.png  (features model)"
    fi
    
    if [ -n "$BIOTYPE_CSV" ]; then
        echo "    - umap_by_biotype.png"
        echo "    - umap_minor_biotypes_only.png"
    fi
    
    echo "    - umap_embeddings.csv"
    echo "  (... same for fold_1 through fold_4)"
    echo ""
    echo "Next step:"
    echo "  bash 03_spatial_clustering.sh --umap_dir $OUTPUT_DIR"
else
    echo ""
    echo " Step 2 Failed (exit code: $exit_code)"
    exit $exit_code
fi