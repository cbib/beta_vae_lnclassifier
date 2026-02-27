#!/bin/bash
set -e

GENCODE_VERSION=${GENCODE_VERSION:-49}
INPUT_DIR="data/raw"
OUTPUT_DIR="data/cdhit_clusters"
CPUS=${CDHIT_CPUS:-16}  # Allow override via environment variable

echo "=========================================="
echo "Step 3: CD-HIT Clustering"
echo "=========================================="
echo ""

# Check conda environment
if ! command -v cd-hit-est &> /dev/null; then
    echo "ERROR: cd-hit-est not found!"
    echo ""
    echo "Please create and activate the conda environment:"
    echo "  conda env create -f cdhit_env.yml"
    echo "  conda activate cdhit_env"
    exit 1
fi

echo "CD-HIT version:"
cd-hit-est -h | head -3
echo ""

# Create output directory
mkdir -p ${OUTPUT_DIR}
mkdir -p logs

# Define input files
LNC_INPUT="${INPUT_DIR}/gencode.v${GENCODE_VERSION}.lncRNA_transcripts.fa"
PC_INPUT="${INPUT_DIR}/gencode.v${GENCODE_VERSION}.pc_transcripts.fa"
LNC_OUTPUT="${OUTPUT_DIR}/lnc_clustered.fa"
PC_OUTPUT="${OUTPUT_DIR}/pc_clustered.fa"

# Check input files exist
if [ ! -f "${LNC_INPUT}" ] || [ ! -f "${PC_INPUT}" ]; then
    echo "ERROR: Input FASTA files not found!"
    echo "Expected:"
    echo "  ${LNC_INPUT}"
    echo "  ${PC_INPUT}"
    echo ""
    echo "Please run 00_download_gencode.sh first"
    exit 1
fi

# ============================================================================
# Process lncRNA
# ============================================================================
echo "Processing lncRNA..."
echo "Input:  ${LNC_INPUT}"
echo "Output: ${LNC_OUTPUT}"

LNC_INPUT_COUNT=$(grep -c '^>' ${LNC_INPUT})
echo "Input sequences: ${LNC_INPUT_COUNT}"
echo ""

echo "Running CD-HIT-EST (this may take 30-60 minutes)..."
time cd-hit-est \
    -i ${LNC_INPUT} \
    -o ${LNC_OUTPUT} \
    -c 0.9 \
    -n 8 \
    -M 8000 \
    -T ${CPUS} \
    -d 0 \
    -aS 0.8 \
    -g 1 \
    -b 100 \
    -s 0.9 \
    -B 1 \
    -p 1 \
    > logs/cdhit_lnc.log 2>&1

LNC_OUTPUT_COUNT=$(grep -c '^>' ${LNC_OUTPUT})
LNC_REMOVED=$((LNC_INPUT_COUNT - LNC_OUTPUT_COUNT))
LNC_PERCENT=$((100 * LNC_REMOVED / LNC_INPUT_COUNT))

echo ""
echo "lncRNA clustering complete!"
echo "  Input:   ${LNC_INPUT_COUNT}"
echo "  Output:  ${LNC_OUTPUT_COUNT}"
echo "  Removed: ${LNC_REMOVED} (${LNC_PERCENT}%)"
echo ""

# ============================================================================
# Process Protein-Coding
# ============================================================================
echo "Processing Protein-coding..."
echo "Input:  ${PC_INPUT}"
echo "Output: ${PC_OUTPUT}"

PC_INPUT_COUNT=$(grep -c '^>' ${PC_INPUT})
echo "Input sequences: ${PC_INPUT_COUNT}"
echo ""

echo "Running CD-HIT-EST (this may take 2-4 hours)..."
time cd-hit-est \
    -i ${PC_INPUT} \
    -o ${PC_OUTPUT} \
    -c 0.9 \
    -n 8 \
    -M 32000 \
    -T ${CPUS} \
    -d 0 \
    -aS 0.8 \
    -g 1 \
    -b 100 \
    -s 0.9 \
    -B 1 \
    -p 1 \
    > logs/cdhit_pc.log 2>&1

PC_OUTPUT_COUNT=$(grep -c '^>' ${PC_OUTPUT})
PC_REMOVED=$((PC_INPUT_COUNT - PC_OUTPUT_COUNT))
PC_PERCENT=$((100 * PC_REMOVED / PC_INPUT_COUNT))

echo ""
echo "Protein-coding clustering complete!"
echo "  Input:   ${PC_INPUT_COUNT}"
echo "  Output:  ${PC_OUTPUT_COUNT}"
echo "  Removed: ${PC_REMOVED} (${PC_PERCENT}%)"
echo ""

# ============================================================================
# Summary
# ============================================================================
TOTAL_INPUT=$((LNC_INPUT_COUNT + PC_INPUT_COUNT))
TOTAL_OUTPUT=$((LNC_OUTPUT_COUNT + PC_OUTPUT_COUNT))
TOTAL_REMOVED=$((TOTAL_INPUT - TOTAL_OUTPUT))
TOTAL_PERCENT=$((100 * TOTAL_REMOVED / TOTAL_INPUT))

echo "=========================================="
echo "CD-HIT Clustering Complete"
echo "=========================================="
echo ""
echo "Summary:"
echo "  Total input:   ${TOTAL_INPUT}"
echo "  Total output:  ${TOTAL_OUTPUT}"
echo "  Total removed: ${TOTAL_REMOVED} (${TOTAL_PERCENT}%)"
echo ""
echo "Output files:"
ls -lh ${OUTPUT_DIR}/*.fa
echo ""
echo "Cluster files:"
ls -lh ${OUTPUT_DIR}/*.clstr
echo ""
echo "Log files:"
ls -lh logs/cdhit_*.log