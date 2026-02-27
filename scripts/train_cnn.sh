#!/bin/bash
#SBATCH --job-name=train_cnn_g47    # Job name
#SBATCH --output=logs/train_cnn_g47.out # Standard output log
#SBATCH --error=logs/train_cnn_g47.err  # Standard error log
#SBATCH --gres=gpu:nvidia_h100_nvl_1g.12gb:1             # Request 1 GPU
#SBATCH --partition=gpu              # Partition to submit to (e.g., GPU queue)
#SBATCH --time=48:00:00    # Time limit day:hrs:min:sec

# Load modules (if necessary)
# module load cuda/12.8                # Load CUDA module (adjust version as needed)

module load nvidia/cuda/12.1                # Load CUDA module (adjust version as needed)
eval "$(conda shell.bash hook)"
conda activate beta_lncrna

cd ${DATA_ROOT}

# Run from project root
# python -m src.main_contrastive --config configs/cnn_contrastive_g49.json --biotype_csv ${DATA_ROOT}/data/dataset_biotypes/g49_dataset_biotypes_cdhit.csv

# Gencode 47 version
python -m src.main_contrastive --config configs/cnn_contrastive_g47.json --biotype_csv ${DATA_ROOT}/data/dataset_biotypes/g47_dataset_biotypes_cdhit.csv


echo "Training complete at $(date)"
echo "Job ID: $SLURM_JOB_ID"
