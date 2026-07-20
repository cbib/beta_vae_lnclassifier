#!/bin/bash
#SBATCH --job-name=train_subgroup_no_nonb2
#SBATCH --output=logs/subgroup_no_nonb2_%j.log
#SBATCH --error=logs/subgroup_no_nonb2_%j.err
#SBATCH --gres=gpu:nvidia_h100_nvl            # Request 1 GPU
#SBATCH --partition=gpu              # Partition to submit to (e.g., GPU queue)
#SBATCH --time=70:00:00    # Time limit day:hrs:min:sec
#SBATCH --mem=64G

# Load modules (if necessary)
module load nvidia/cuda/12.1                # Load CUDA module (adjust version as needed)
eval "$(conda shell.bash hook)"
conda activate beta_lncrna

cd /mnt/cbib/LNClassifier/beta_vae_lnclassifier

python src/main_subgroup.py \
    --config configs/beta_vae_subgroup_base_g49.json \
    --device cuda:0 \

#python src/main_subgroup.py \
#     --config configs/beta_vae_subgroup_token_sa_g49.json \
#     --device cuda:0

#python src/main_feature_only.py \
#    --config configs/beta_vae_feature_only_base_g47_nononb2.json \
#    --device cuda:0

echo "Training complete at $(date)"
echo "Job ID: $SLURM_JOB_ID"
