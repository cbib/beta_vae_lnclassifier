#!/bin/bash
set -e

echo "=========================================="
echo "Step 3: Spatial Clustering Analysis"
echo "=========================================="

# Default values
UMAP_DIR=""
N_REGIONS=5
OUTPUT_DIR=""
MODEL_LABEL=""
GENCODE_VERSION=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --umap_dir)
            UMAP_DIR="$2"
            shift 2
            ;;
        --n_regions)
            N_REGIONS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
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
            echo "Usage: $0 --umap_dir DIR [--n_regions 5] [--output_dir DIR] [--model_label LABEL] [--gencode_version VERSION]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$UMAP_DIR" ]; then
    echo "ERROR: --umap_dir required"
    echo "Usage: $0 --umap_dir DIR [--n_regions 5]"
    exit 1
fi

# Check UMAP directory exists
if [ ! -d "$UMAP_DIR" ]; then
    echo "ERROR: UMAP directory not found: $UMAP_DIR"
    echo "Please run 02_generate_umap.sh first"
    exit 1
fi

# Infer gencode version from umap_dir path if not provided
if [ -z "$GENCODE_VERSION" ]; then
    GENCODE_VERSION=$(echo "$UMAP_DIR" | grep -oP '(?<=_g)\d+' | head -1)
    [ -n "$GENCODE_VERSION" ] && GENCODE_VERSION="v${GENCODE_VERSION}"
fi

# Set default output directory if not specified
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$(dirname "$UMAP_DIR")/spatial_analysis"
fi

echo "Configuration:"
echo "  UMAP dir: $UMAP_DIR"
echo "  N_regions: $N_REGIONS"
echo "  Output: $OUTPUT_DIR"
echo "  Model label: ${MODEL_LABEL:-none}"
echo "  GENCODE version: ${GENCODE_VERSION:-none}"
echo ""

# Get script directory
SCRIPT_DIR="$(dirname "$0")/scripts"

# Check script exists
if [ ! -f "$SCRIPT_DIR/analyze_hardcase_spatial_patterns.py" ]; then
    echo "ERROR: analyze_hardcase_spatial_patterns.py not found in $SCRIPT_DIR"
    exit 1
fi

# Check that fold directories exist
fold_count=$(find "$UMAP_DIR" -maxdepth 1 -type d -name "fold_*" | wc -l)
if [ $fold_count -eq 0 ]; then
    echo "ERROR: No fold_* directories found in $UMAP_DIR"
    exit 1
fi

echo "Found $fold_count folds to analyze"
echo ""

# Build command
CMD=(
    python "$SCRIPT_DIR/analyze_hardcase_spatial_patterns.py"
    --umap_dir "$UMAP_DIR"
    --n_regions "$N_REGIONS"
    --output_dir "$OUTPUT_DIR"
)
[ -n "$MODEL_LABEL" ]     && CMD+=(--model_label "$MODEL_LABEL")
[ -n "$GENCODE_VERSION" ] && CMD+=(--gencode_version "$GENCODE_VERSION")

# Run spatial analysis
echo "Running spatial clustering analysis..."
echo ""

"${CMD[@]}"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo " Step 3 Complete"
    echo "=========================================="
    echo ""
    echo "Outputs:"
    echo "  $OUTPUT_DIR/fold_0/"
    echo "    - spatial_regions_labeled.png"
    echo "    - region_biotype_heatmap.png"
    echo "    - samples_with_regions.csv"
    echo "    - region_statistics.csv"
    echo "    - region_biotype_composition.csv"
    echo "  (... same for fold_1 through fold_4)"
    echo "  $OUTPUT_DIR/all_folds_region_statistics.csv"
    echo "  $OUTPUT_DIR/cross_fold_summary.png"
    echo ""
    echo "Next step:"
    echo "  bash 04_biotype_enrichment.sh --spatial_dir $OUTPUT_DIR"
else
    echo ""
    echo " Step 3 Failed (exit code: $exit_code)"
    exit $exit_code
fi