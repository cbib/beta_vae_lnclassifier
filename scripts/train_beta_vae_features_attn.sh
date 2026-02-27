#!/bin/bash
#SBATCH --job-name=bvae_g47_features_attention    # Job name
#SBATCH --output=logs/bvae_g47_features_attention.out # Standard output log
#SBATCH --error=logs/bvae_g47_features_attention.err  # Standard error log
#SBATCH --gres=gpu:nvidia_h100_nvl            # Request 1 GPU
#SBATCH --partition=gpu              # Partition to submit to (e.g., GPU queue)
#SBATCH --time=48:00:00    # Time limit day:hrs:min:sec
#SBATCH --mem=64G

# Load modules (if necessary)
module load nvidia/cuda/12.1                # Load CUDA module (adjust version as needed)
eval "$(conda shell.bash hook)"
conda activate beta_lncrna

cd ${DATA_ROOT}

# Run from project root
# python -m src.main_features_attn --config configs/beta_vae_features_attn_g49.json --device cuda:0

# Gencode 47 version
python -m src.main_features_attn --config configs/beta_vae_features_attn_g47.json --device cuda:0

echo "Training complete at $(date)"
echo "Job ID: $SLURM_JOB_ID"
