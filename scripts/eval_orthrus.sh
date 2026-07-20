#!/bin/bash
#SBATCH --job-name=orthrus_4track
#SBATCH --output=logs/orthrus_%j.log
#SBATCH --error=logs/orthrus_%j.err
#SBATCH --gres=gpu:nvidia_h100_nvl            # Request 1 GPU
#SBATCH --partition=gpu              # Partition to submit to (e.g., GPU queue)
#SBATCH --time=70:00:00    # Time limit day:hrs:min:sec
#SBATCH --mem=64G

# Load modules (if necessary)
module load nvidia/cuda/12.1                # Load CUDA module (adjust version as needed)
eval "$(conda shell.bash hook)"
conda activate orthrus

cd /mnt/cbib/LNClassifier/beta_vae_lnclassifier

python analysis/benchmark/evaluate_orthrus.py --model_name Orthrus/models/orthrus-4-track --train_lnc_fasta data/split_gencode_49/lnc_trainval.fa --train_pc_fasta data/split_gencode_49/pc_trainval.fa --test_lnc_fasta data/split_gencode_49/lnc_test.fa --test_pc_fasta data/split_gencode_49/pc_test.fa --output_dir gencode_v49_experiments/orthrus_evaluation/ --batch_size 16 --max_length 15000 --save_unpooled

# G47 version
python analysis/benchmark/evaluate_orthrus.py --model_name Orthrus/models/orthrus-4-track --train_lnc_fasta data/split_gencode_47/lnc_trainval.fa --train_pc_fasta data/split_gencode_47/pc_trainval.fa --test_lnc_fasta data/split_gencode_47/lnc_test.fa --test_pc_fasta data/split_gencode_47/pc_test.fa --output_dir gencode_v47_experiments/orthrus_evaluation/ --batch_size 16 --max_length 15000 --save_unpooled