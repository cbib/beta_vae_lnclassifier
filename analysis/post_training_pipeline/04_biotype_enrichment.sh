#!/bin/bash
set -e

echo "=========================================="
echo "Step 4: Biotype Enrichment Analysis"
echo "=========================================="

# Default values
SPATIAL_DIR=""
OUTPUT_DIR=""
MIN_COUNT=500
MODEL_LABEL=""
GENCODE_VERSION=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --spatial_dir)
            SPATIAL_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --min_count)
            MIN_COUNT="$2"
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
            echo "Usage: $0 --spatial_dir DIR [--output_dir DIR] [--min_count 500] [--model_label LABEL] [--gencode_version VERSION]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$SPATIAL_DIR" ]; then
    echo "ERROR: --spatial_dir required"
    echo "Usage: $0 --spatial_dir DIR [--output_dir DIR]"
    exit 1
fi

# Check spatial directory exists
if [ ! -d "$SPATIAL_DIR" ]; then
    echo "ERROR: Spatial analysis directory not found: $SPATIAL_DIR"
    echo "Please run 03_spatial_clustering.sh first"
    exit 1
fi

# Infer gencode version from spatial_dir path if not provided
if [ -z "$GENCODE_VERSION" ]; then
    GENCODE_VERSION=$(echo "$SPATIAL_DIR" | grep -oP '(?<=_g)\d+' | head -1)
    [ -n "$GENCODE_VERSION" ] && GENCODE_VERSION="v${GENCODE_VERSION}"
fi

# Set default output directory
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$(dirname "$SPATIAL_DIR")/biotype_enrichment"
fi

echo "Configuration:"
echo "  Spatial dir: $SPATIAL_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Min count: $MIN_COUNT"
echo "  Model label: ${MODEL_LABEL:-none}"
echo "  GENCODE version: ${GENCODE_VERSION:-none}"
echo ""

# Get script directory
SCRIPT_DIR="$(dirname "$0")/scripts"

# Check script exists
if [ ! -f "$SCRIPT_DIR/analyze_global_biotype_enrichment.py" ]; then
    echo "ERROR: analyze_global_biotype_enrichment.py not found in $SCRIPT_DIR"
    exit 1
fi

# Check that samples_with_regions.csv files exist
sample_count=$(find "$SPATIAL_DIR" -name "samples_with_regions.csv" | wc -l)
if [ $sample_count -eq 0 ]; then
    echo "ERROR: No samples_with_regions.csv files found in $SPATIAL_DIR"
    echo "Please run 03_spatial_clustering.sh first"
    exit 1
fi

echo "Found $sample_count fold(s) with region data"
echo ""

# Build command
CMD=(
    python "$SCRIPT_DIR/analyze_global_biotype_enrichment.py"
    --spatial_dir "$SPATIAL_DIR"
    --output_dir "$OUTPUT_DIR"
    --min_count "$MIN_COUNT"
)
[ -n "$MODEL_LABEL" ]     && CMD+=(--model_label "$MODEL_LABEL")
[ -n "$GENCODE_VERSION" ] && CMD+=(--gencode_version "$GENCODE_VERSION")

# Run enrichment analysis
echo "Running biotype enrichment analysis..."
echo ""

"${CMD[@]}"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo " Step 4 Complete"
    echo "=========================================="
    echo ""
    echo "Outputs:"
    echo "  $OUTPUT_DIR/global_biotype_enrichment.csv"
    echo "  $OUTPUT_DIR/global_biotype_enrichment.png"
    echo ""
    echo "Next step (optional):"
    echo "  bash 05_gencode_comparison.sh ..."
    echo ""
    echo "Or generate summary:"
    echo "  bash 06_generate_summary_report.sh --experiment_dir $(dirname "$SPATIAL_DIR")"
else
    echo ""
    echo " Step 4 Failed (exit code: $exit_code)"
    exit $exit_code
fi