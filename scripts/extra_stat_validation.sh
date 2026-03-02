#!/bin/bash
#SBATCH --job-name=stat_validation
#SBATCH --output=logs/stat_validation.out
#SBATCH --error=logs/stat_validation.err
#SBATCH --gres=gpu:nvidia_h100_nvl             # Request 1 GPU
#SBATCH --partition=gpu              # Partition to submit to (e.g., GPU queue)
#SBATCH --time=64:00:00    # Time limit day:hrs:min:sec
#SBATCH --mem=64G

module load nvidia/cuda/12.1                # Load CUDA module (adjust version as needed)
eval "$(conda shell.bash hook)"
conda activate beta_lncrna

cd /mnt/cbib/LNClassifier/DL_benchmark

# ─────────────────────────────────────────────
# GENCODEv47
# ─────────────────────────────────────────────
python analysis/post_training_pipeline/scripts/extra_stat_validation.py \
    --fold_results \
        gencode_v47_experiments/cnn_g47/cv_evaluation_results.json \
        gencode_v47_experiments/beta_vae_contrastive_g47/cv_evaluation_results.json \
        gencode_v47_experiments/beta_vae_features_g47/cv_evaluation_results.json \
        gencode_v47_experiments/beta_vae_features_attn_g47/cv_evaluation_results.json \
    --test_preds \
        gencode_v47_experiments/cnn_g47/evaluation_csvs/test_predictions.csv \
        gencode_v47_experiments/beta_vae_contrastive_g47/evaluation_csvs/test_predictions.csv \
        gencode_v47_experiments/beta_vae_features_g47/evaluation_csvs/test_predictions.csv \
        gencode_v47_experiments/beta_vae_features_attn_g47/evaluation_csvs/test_predictions.csv \
    --labels "CNN" "βVAE+Contr." "βVAE+Feat." "βVAE+Attn" \
    --prob_col mean_confidence \
    --output_dir gencode_v47_experiments/stat_results/

# ─────────────────────────────────────────────
# GENCODEv49
# ─────────────────────────────────────────────
python analysis/post_training_pipeline/scripts/extra_stat_validation.py \
    --fold_results \
        gencode_v49_experiments/cnn_g49/cv_evaluation_results.json \
        gencode_v49_experiments/beta_vae_contrastive_g49/cv_evaluation_results.json \
        gencode_v49_experiments/beta_vae_features_g49/cv_evaluation_results.json \
        gencode_v49_experiments/beta_vae_features_attn_g49/cv_evaluation_results.json \
    --test_preds \
        gencode_v49_experiments/cnn_g49/evaluation_csvs/test_predictions.csv \
        gencode_v49_experiments/beta_vae_contrastive_g49/evaluation_csvs/test_predictions.csv \
        gencode_v49_experiments/beta_vae_features_g49/evaluation_csvs/test_predictions.csv \
        gencode_v49_experiments/beta_vae_features_attn_g49/evaluation_csvs/test_predictions.csv \
    --labels "CNN" "βVAE+Contr." "βVAE+Feat." "βVAE+Attn" \
    --prob_col mean_confidence \
    --output_dir gencode_v49_experiments/stat_results/