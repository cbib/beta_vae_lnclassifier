#!/bin/bash
set -e

GENCODE_VERSION=${GENCODE_VERSION:-49}
LNC_FASTA="data/cdhit_clusters/lnc_clustered.fa"
PC_FASTA="data/cdhit_clusters/g47_pc_clustered.fa"
BIOTYPE_CSV="data/processed/g${GENCODE_VERSION}_dataset_biotypes_cdhit.csv"
OUTPUT_DIR="data/processed/sanity_check"

echo "=========================================="
echo "Step 6: Sanity Check"
echo "=========================================="
echo ""

# Check inputs exist
if [ ! -f "${LNC_FASTA}" ] || [ ! -f "${PC_FASTA}" ]; then
    echo "ERROR: CD-HIT FASTA files not found!"
    echo "Expected:"
    echo "  ${LNC_FASTA}"
    echo "  ${PC_FASTA}"
    echo ""
    echo "Please run 03_run_cdhit.sh first"
    exit 1
fi

if [ ! -f "${BIOTYPE_CSV}" ]; then
    echo "ERROR: Biotype CSV not found: ${BIOTYPE_CSV}"
    echo "Please run 04_filter_to_cdhit.sh first"
    exit 1
fi

mkdir -p ${OUTPUT_DIR}

echo "lncRNA FASTA:  ${LNC_FASTA}"
echo "PC FASTA:      ${PC_FASTA}"
echo "Biotype CSV:   ${BIOTYPE_CSV}"
echo "Output dir:    ${OUTPUT_DIR}"
echo ""

# Quick FASTA checks
echo "Quick FASTA checks:"
LNC_COUNT=$(grep -c '^>' ${LNC_FASTA})
PC_COUNT=$(grep -c '^>' ${PC_FASTA})
echo "  lncRNA sequences: ${LNC_COUNT}"
echo "  PC sequences:     ${PC_COUNT}"
echo "  Total:            $((LNC_COUNT + PC_COUNT))"
echo ""

# Run full sanity check
python analysis/prepare_gencode/scripts/gencode_sanity_check.py \
    --lnc_fasta ${LNC_FASTA} \
    --pc_fasta ${PC_FASTA} \
    --biotype_csv ${BIOTYPE_CSV} \
    --output_dir ${OUTPUT_DIR}

echo ""
echo "=========================================="
echo "Step 6 complete!"
echo "=========================================="
echo ""
echo "Output files:"
ls -lh ${OUTPUT_DIR}/
echo ""

# Display summary
if [ -f "${OUTPUT_DIR}/sanity_check_summary.txt" ]; then
    echo "Summary:"
    cat ${OUTPUT_DIR}/sanity_check_summary.txt
fi