#!/bin/bash
#SBATCH --job-name=hardcase_overlap
#SBATCH --output=logs/hardcase_overlap.out
#SBATCH --error=logs/hardcase_overlap.err
#SBATCH --gres=gpu:nvidia_h100_nvl_1g.24gb:1             # Request 1 GPU
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --mem=8G

eval "$(conda shell.bash hook)"
conda activate beta_lncrna

cd /mnt/cbib/LNClassifier/DL_benchmark

ATTN_G47=gencode_v47_experiments/beta_vae_features_attn_g47
ATTN_G49=gencode_v49_experiments/beta_vae_features_attn_g49

for VERSION in v47 v49; do
    [[ $VERSION == v47 ]] && ATTN=$ATTN_G47 || ATTN=$ATTN_G49

    python analysis/post_training_pipeline/scripts/ablation_hardcase_overlap.py \
        --full_model    "$ATTN/evaluation_csvs/test_predictions.csv" \
        --ablations \
            "Seq. only:$ATTN/ablations/seq_only/evaluation_csvs/test_predictions.csv" \
            "Seq.+TE:$ATTN/ablations/seq_te/evaluation_csvs/test_predictions.csv" \
            "Seq.+NonB:$ATTN/ablations/seq_nonb/evaluation_csvs/test_predictions.csv" \
        --confidence_threshold 0.6 \
        --output_dir    "stat_results/ablations_$VERSION" \
        --gencode_version "$VERSION"
done