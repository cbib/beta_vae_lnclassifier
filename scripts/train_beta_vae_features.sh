#!/bin/bash
#SBATCH --job-name=train_bvae_g47_features   # Job name
#SBATCH --output=logs/train_bvae_g47_features.out # Standard output log
#SBATCH --error=logs/train_bvae_g47_features.err  # Standard error log
#SBATCH --gres=gpu:nvidia_h100_nvl             # Request 1 GPU
#SBATCH --partition=gpu              # Partition to submit to (e.g., GPU queue)
#SBATCH --time=48:00:00    # Time limit day:hrs:min:sec
#SBATCH --mem=64G

# Load modules (if necessary)
module load nvidia/cuda/12.1                # Load CUDA module (adjust version as needed)
eval "$(conda shell.bash hook)"
conda activate beta_lncrna

cd ${DATA_ROOT}

# Run from project root
# python -m src.main_features --config configs/beta_vae_features_g49.json --device cuda:0

# g47 version
python -m src.main_features --config configs/beta_vae_features_g47.json --device cuda:0

echo "Training complete at $(date)"
echo "Job ID: $SLURM_JOB_ID"
