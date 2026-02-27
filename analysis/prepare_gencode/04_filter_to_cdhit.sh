#!/bin/bash
set -e

GENCODE_VERSION=${GENCODE_VERSION:-49}
BIOTYPE_CSV="data/processed/g${GENCODE_VERSION}_dataset_biotypes.csv"
LNC_FASTA="data/cdhit_clusters/lnc_clustered.fa"
PC_FASTA="data/cdhit_clusters/g47_pc_clustered.fa"
OUTPUT_CSV="data/processed/g${GENCODE_VERSION}_dataset_biotypes_cdhit.csv"

echo "=========================================="
echo "Step 4: Filter Biotypes to CD-HIT Output"
echo "=========================================="
echo ""

# Check inputs exist
if [ ! -f "${BIOTYPE_CSV}" ]; then
    echo "ERROR: Biotype CSV not found: ${BIOTYPE_CSV}"
    echo "Please run 02_filter_gene_biotypes.sh first"
    exit 1
fi

if [ ! -f "${LNC_FASTA}" ] || [ ! -f "${PC_FASTA}" ]; then
    echo "ERROR: CD-HIT FASTA files not found!"
    echo "Expected:"
    echo "  ${LNC_FASTA}"
    echo "  ${PC_FASTA}"
    echo ""
    echo "Please run 03_run_cdhit.sh first"
    exit 1
fi

echo "Input biotype CSV: ${BIOTYPE_CSV}"
echo "lncRNA FASTA:      ${LNC_FASTA}"
echo "PC FASTA:          ${PC_FASTA}"
echo "Output CSV:        ${OUTPUT_CSV}"
echo ""

python analysis/prepare_gencode/scripts/filter_biotypes_to_fasta.py \
    --biotype_csv ${BIOTYPE_CSV} \
    --lnc_fasta ${LNC_FASTA} \
    --pc_fasta ${PC_FASTA} \
    --output ${OUTPUT_CSV}

echo ""
echo "Step 4 complete!"
echo "Output files:"
ls -lh data/processed/gencode${GENCODE_VERSION}_dataset_biotypes_cdhit90.*