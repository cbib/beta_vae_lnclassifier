#!/bin/bash
set -e  # Exit on error

# Get the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set project root (adjust based on script location)
# For scripts in pipelines/prepare_gencode/, go up 2 levels
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Add to PYTHONPATH so Python can find modules
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=========================================="
echo "GENCODE Preprocessing Pipeline"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Configuration
GENCODE_VERSION=${GENCODE_VERSION:-49}
DO_SPLIT=${DO_SPLIT:-0}        # set to 1 to create trainval/test split
TEST_SIZE=${TEST_SIZE:-0.05}
SPLIT_SEED=${SPLIT_SEED:-42}
export GENCODE_VERSION

# Create directories
mkdir -p data/{raw,cdhit,processed,processed/sanity_check}
mkdir -p figures
mkdir -p logs

# Step 0: Download
echo "[Step 0/6] Downloading GENCODE v${GENCODE_VERSION}..."
bash 00_download_gencode.sh
echo ""

# Step 1: Extract biotypes
echo "[Step 1/6] Extracting biotypes from GTF..."
bash 01_extract_biotypes.sh
echo ""

# Step 2: Filter by gene biotype
echo "[Step 2/6] Filtering by gene biotype..."
bash 02_filter_gene_biotypes.sh
echo ""

# Step 3: CD-HIT clustering
echo "[Step 3/6] Submitting CD-HIT clustering job..."
JOB_ID=$(bash 03_run_cdhit.sh)
echo "CD-HIT job submitted: ${JOB_ID}"
echo "Waiting for CD-HIT to complete..."
echo "(This may take 2-6 hours depending on cluster load)"
echo ""
echo "You can monitor progress with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/v49_cdhit_*.log"
echo ""
echo "Once CD-HIT completes, run:"
echo "  bash 04_filter_to_cdhit.sh"
echo "  bash 05_create_visualizations.sh"
echo "  bash 06_sanity_check.sh"
echo ""
echo "Or re-run this script to continue from Step 4"
exit 0

# Steps 4-6 will run after CD-HIT completes
# Uncomment below to run automatically (requires CD-HIT to finish)

# Wait for CD-HIT to complete
while squeue -u $USER | grep -q "v49_cdhit"; do
    sleep 60
done

# # Step 4: Filter to CD-HIT
echo "[Step 4/6] Filtering biotypes to CD-HIT output..."
bash 04_filter_to_cdhit.sh
echo ""

# # Step 5: Visualizations
echo "[Step 5/6] Creating visualizations..."
bash 05_create_visualizations.sh
echo ""

# # Step 6: Sanity check
echo "[Step 6/6] Running sanity check..."
bash 06_sanity_check.sh
echo ""

# Step 7: Train/val + test split (optional)
if [ "${DO_SPLIT}" = "1" ]; then
    echo "[Step 7/7] Creating train/val and test split..."
    python "${PROJECT_ROOT}/scripts/split_test_set.py" \
        --lnc_fasta data/cdhit/lnc_clustered.fa \
        --pc_fasta  data/cdhit/pc_clustered.fa \
        --output_dir data/splits/ \
        --test_size ${TEST_SIZE} \
        --seed ${SPLIT_SEED}
    echo ""
else
    echo "[Step 7/7] Skipping split (DO_SPLIT not set)"
    echo "  To create a train/val + test split, run:"
    echo "  python scripts/split_test_set.py \\"
    echo "    --lnc_fasta data/cdhit/lnc_clustered.fa \\"
    echo "    --pc_fasta  data/cdhit/pc_clustered.fa \\"
    echo "    --output_dir data/splits/"
    echo ""
fi

# echo "=========================================="
# echo "Pipeline Complete!"
# echo "=========================================="
# echo "End time: $(date)"
# echo ""
# echo "Output files:"
# echo "  - data/cdhit/lnc_clustered.fa"
# echo "  - data/cdhit/pc_clustered.fa"
# echo "  - data/processed/gencode49_dataset_biotypes_cdhit90.csv"
# echo "  - figures/dataset_distribution.png"
# echo "  - data/processed/sanity_check/sanity_check_report.txt"
# echo "  - data/split_gencode_49/lnc_test.fa (if split)"
# echo "  - data/split_gencode_49/pc_test.fa (if split)"
# echo ""
# echo "Ready for training!"