#!/bin/bash
#SBATCH --job-name=train_bvae_g47_features_attn_ablation_seqonly    # Job name
#SBATCH --output=logs/train_bvae_g47_features_attn_ablation_seqonly.out # Standard output log
#SBATCH --error=logs/train_bvae_g47_features_attn_ablation_seqonly.err  # Standard error log
#SBATCH --gres=gpu:nvidia_h100_nvl             # Request 1 GPU
#SBATCH --partition=gpu              # Partition to submit to (e.g., GPU queue)
#SBATCH --time=64:00:00    # Time limit day:hrs:min:sec
#SBATCH --mem=64G

# Load modules (if necessary)
module load nvidia/cuda/12.1                # Load CUDA module (adjust version as needed)
eval "$(conda shell.bash hook)"
conda activate beta_lncrna

cd /mnt/cbib/LNClassifier/DL_benchmark

# Run from project root
python -m experiments.train.main_features_attn_ablation --base_config configs/beta_vae_features_attn_g47.json --variants seq_only seq_te seq_nonb --device cuda:0 --n_folds 5
#python -m experiments.train.main_features_attn_ablation --base_config configs/beta_vae_features_attn_g49.json --variants seq_only seq_te seq_nonb --device cuda:0 --n_folds 5


echo "Training complete at $(date)"
echo "Job ID: $SLURM_JOB_ID"
