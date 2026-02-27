#!/bin/bash
#SBATCH --job-name=v49_lnc_cdhit    # Job name
#SBATCH --output=logs/v49_lnc_cdhit.log # Standard output log
#SBATCH --error=logs/v49_lnc_cdhit.err  # Standard error log
#SBATCH --gres=gpu:nvidia_h100_nvl_1g.24gb:1            # Request 1 GPU
#SBATCH --partition=gpu              # Partition to submit to (e.g., GPU queue)
#SBATCH --time=24:00:00    # Time limit day-hrs:min:sec

# lncRNA clustering only for GENCODE 49

OUTPUT_DIR="data/cdhit_clusters"
mkdir -p ${OUTPUT_DIR}
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate cdhit_env

cd-hit-est \
    -i data/raw/gencode.v49.lncRNA_transcripts.fa \
    -o ${OUTPUT_DIR}/g49_lncRNA_clustered.fa \
    -c 0.9 \
    -n 8 \
    -M 0 \
    -T ${SLURM_CPUS_PER_TASK} \
    -d 0 \
    -aS 0.8 \
    -g 0 \
    -b 20 \
    -s 0.9

echo "lncRNA clustering complete!"