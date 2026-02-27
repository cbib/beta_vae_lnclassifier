#!/bin/bash
#SBATCH --job-name=v49_pc_cdhit    # Job name
#SBATCH --output=logs/v49_pc_cdhit.log # Standard output log
#SBATCH --error=logs/v49_pc_cdhit.err  # Standard error log
#SBATCH --partition=compute              # Partition to submit to (e.g., GPU queue)
#SBATCH --cpus-per-task=60          # Number of CPU cores per task
#SBATCH --mem=64G                     # Memory per node
#SBATCH --time=24:00:00    # Time limit day-hrs:min:sec

# PC clustering only

OUTPUT_DIR="data/g49_cdhit_clusters"
mkdir -p ${OUTPUT_DIR}
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate cdhit_env

cd-hit-est \
    -i data/raw/gencode.v49.pc_transcripts.fa \
    -o ${OUTPUT_DIR}/g49_pc_clustered.fa \
    -c 0.9 \
    -n 8 \
    -M 0 \
    -T ${SLURM_CPUS_PER_TASK} \
    -d 0 \
    -aS 0.8 \
    -g 0 \
    -b 20 \
    -s 0.9

echo "PC clustering complete!"