#!/bin/bash
set -e

GENCODE_VERSION=${GENCODE_VERSION:-49}
INPUT_CSV="data/processed/g${GENCODE_VERSION}_transcript_biotypes.csv"
OUTPUT_CSV="data/processed/g${GENCODE_VERSION}_dataset_biotypes.csv"

echo "=========================================="
echo "Step 2: Filter by Gene Biotype"
echo "=========================================="
echo ""

if [ ! -f "${INPUT_CSV}" ]; then
    echo "ERROR: Input CSV not found: ${INPUT_CSV}"
    echo "Please run 01_extract_biotypes.sh first"
    exit 1
fi

echo "Input CSV: ${INPUT_CSV}"
echo "Output CSV: ${OUTPUT_CSV}"
echo ""

python analysis/prepare_gencode/scripts/filter_by_gene_biotype.py \
    --input ${INPUT_CSV} \
    --output ${OUTPUT_CSV} \
    --keep_gene_biotypes lncRNA protein_coding

echo ""
echo "Step 2 complete!"
echo "Output files:"
ls -lh data/processed/gencode${GENCODE_VERSION}_dataset_biotypes.*