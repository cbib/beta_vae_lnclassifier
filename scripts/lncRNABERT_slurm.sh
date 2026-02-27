#!/bin/bash
#SBATCH --job-name=lncRNA_two_gencodes_emb_testset    # Job name
#SBATCH --output=logs/lncRNA_two_gencodes_emb_testset.out # Standard output log
#SBATCH --error=logs/lncRNA_two_gencodes_emb_testset.err  # Standard error log
#SBATCH --gres=gpu:nvidia_h100_nvl_1g.24gb:1            # Request 1 GPU
#SBATCH --partition=gpu              # Partition to submit to (e.g., GPU queue)
#SBATCH --time=48:00:00    # Time limit day:hrs:min:sec
#SBATCH --mem=80G


# Load modules (if necessary)
module load nvidia/cuda/12.1                # Load CUDA module (adjust version as needed)
eval "$(conda shell.bash hook)"
conda activate lncrnabert

cd ${DATA_ROOT}

# if not done, clone the repository to get the scripts and get data from Zenodo

# Run the Python classification
python -m lncrnapy.scripts.classify data/split_gencode_47/pc_test.fa data/split_gencode_47/lnc_test.fa --output_file g47_lncRNABERT_results.csv --batch_size 8 --k 3 --model_file luukromeijn/lncRNA-BERT-kmer-k3-finetuned --encoding_method kmer --results_dir g47_lncRNABERT_results/

python -m lncrnapy.scripts.classify data/split_gencode_49/pc_test.fa data/split_gencode_49/lnc_test.fa --output_file g49_lncRNABERT_results.csv --batch_size 8 --k 3 --model_file luukromeijn/lncRNA-BERT-kmer-k3-finetuned --encoding_method kmer --results_dir g49_lncRNABERT_results/

# Embeddings/t-SNE
# G47
python -m lncrnapy.scripts.embeddings data/split_gencode_47/pc_test.fa data/split_gencode_47/lnc_test.fa --output_file g47_lncRNABERT_embeddings.h5 --batch_size 8 --k 3 --model_file luukromeijn/lncRNA-BERT-kmer-k3-finetuned --encoding_method kmer --results_dir g47_lncRNABERT_results/ --dim_red None

# G49
python -m lncrnapy.scripts.embeddings data/split_gencode_49/pc_test.fa data/split_gencode_49/lnc_test.fa --output_file g49_lncRNABERT_embeddings.h5 --batch_size 8 --k 3 --model_file luukromeijn/lncRNA-BERT-kmer-k3-finetuned --encoding_method kmer --results_dir g49_lncRNABERT_results/ --dim_red None

# Run evaluation for performance metrics
python -m analysis.lncrnabert_inference.py