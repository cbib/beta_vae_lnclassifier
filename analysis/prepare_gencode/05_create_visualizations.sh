#!/bin/bash
set -e

GENCODE_VERSION=${GENCODE_VERSION:-49}
BIOTYPE_CSV="data/processed/g${GENCODE_VERSION}_dataset_biotypes_cdhit.csv"
OUTPUT_DIR="figures"
V47_CSV=${V47_BIOTYPE_CSV:-""}  # Optional: set to compare with v47 if available

echo "=========================================="
echo "Step 5: Create Visualizations"
echo "=========================================="
echo ""

# Check input exists
if [ ! -f "${BIOTYPE_CSV}" ]; then
    echo "ERROR: Biotype CSV not found: ${BIOTYPE_CSV}"
    echo "Please run 04_filter_to_cdhit.sh first"
    exit 1
fi

mkdir -p ${OUTPUT_DIR}

echo "Input CSV:    ${BIOTYPE_CSV}"
echo "Output dir:   ${OUTPUT_DIR}"

# Build command
CMD="python analysis/prepare_gencode/scripts/class_pie_chart.py \
    --biotype_csv ${BIOTYPE_CSV} \
    --output_dir ${OUTPUT_DIR} \
    --title \"GENCODE v${GENCODE_VERSION} Dataset (post CD-HIT 90%)\""

# Add comparison if v47 CSV provided
if [ -n "${V47_CSV}" ] && [ -f "${V47_CSV}" ]; then
    echo "v47 CSV:      ${V47_CSV}"
    CMD="${CMD} --compare_with ${V47_CSV}"
fi

echo ""
eval ${CMD}

echo ""
echo "Step 5 complete!"
echo "Generated figures:"
ls -lh ${OUTPUT_DIR}/*.png