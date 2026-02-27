#!/bin/bash
set -e

GENCODE_VERSION=${GENCODE_VERSION:-49}
INPUT_GTF="data/raw/gencode.v${GENCODE_VERSION}.annotation.gtf"
OUTPUT_CSV="data/processed/g${GENCODE_VERSION}_transcript_biotypes.csv"

echo "=========================================="
echo "Step 1: Extract Biotypes from GTF"
echo "=========================================="
echo ""

if [ ! -f "${INPUT_GTF}" ]; then
    echo "ERROR: GTF file not found: ${INPUT_GTF}"
    echo "Please run 00_download_gencode.sh first"
    exit 1
fi

echo "Input GTF: ${INPUT_GTF}"
echo "Output CSV: ${OUTPUT_CSV}"
echo ""

python analysis/prepare_gencode/scripts/extract_biotypes_from_gtf.py \
    --gtf ${INPUT_GTF} \
    --output ${OUTPUT_CSV}

echo ""
echo "Step 1 complete!"
echo "Output files:"
ls -lh data/processed/gencode${GENCODE_VERSION}_transcript_biotypes.*