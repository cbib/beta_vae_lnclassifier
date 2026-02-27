#!/bin/bash
set -e

GENCODE_VERSION=${GENCODE_VERSION:-49}
BASE_URL="https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_${GENCODE_VERSION}"
OUTPUT_DIR="data/raw"

mkdir -p ${OUTPUT_DIR}

echo "Downloading GENCODE v${GENCODE_VERSION}..."
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Download GTF
echo "Downloading GTF annotation..."
wget -O ${OUTPUT_DIR}/gencode.v${GENCODE_VERSION}.annotation.gtf.gz \
    ${BASE_URL}/gencode.v${GENCODE_VERSION}.annotation.gtf.gz

echo "Decompressing GTF..."
gunzip -f ${OUTPUT_DIR}/gencode.v${GENCODE_VERSION}.annotation.gtf.gz

# Download lncRNA transcripts
echo "Downloading lncRNA transcripts..."
wget -O ${OUTPUT_DIR}/gencode.v${GENCODE_VERSION}.lncRNA_transcripts.fa.gz \
    ${BASE_URL}/gencode.v${GENCODE_VERSION}.lncRNA_transcripts.fa.gz

echo "Decompressing lncRNA FASTA..."
gunzip -f ${OUTPUT_DIR}/gencode.v${GENCODE_VERSION}.lncRNA_transcripts.fa.gz

# Download PC transcripts
echo "Downloading protein-coding transcripts..."
wget -O ${OUTPUT_DIR}/gencode.v${GENCODE_VERSION}.pc_transcripts.fa.gz \
    ${BASE_URL}/gencode.v${GENCODE_VERSION}.pc_transcripts.fa.gz

echo "Decompressing PC FASTA..."
gunzip -f ${OUTPUT_DIR}/gencode.v${GENCODE_VERSION}.pc_transcripts.fa.gz

echo ""
echo "Download complete!"
echo "Files:"
ls -lh ${OUTPUT_DIR}/