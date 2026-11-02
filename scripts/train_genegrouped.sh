#!/bin/bash
#SBATCH --job-name=train_subgroup_g47_genegrouped
#SBATCH --output=logs/subgroup_g47_genegrouped_%j.log
#SBATCH --error=logs/subgroup_g47_genegrouped_%j.err
#SBATCH --gres=gpu:nvidia_h100_nvl_4g.47gb:1            # Request 1 GPU
#SBATCH --partition=gpu              # Partition to submit to (e.g., GPU queue)
#SBATCH --time=70:00:00    # Time limit day:hrs:min:sec
#SBATCH --mem=24G

# Load modules (if necessary)
module load nvidia/cuda/12.1                # Load CUDA module (adjust version as needed)
eval "$(conda shell.bash hook)"
conda activate beta_lncrna

cd ${DATA_ROOT}
set -euo pipefail

REL=47                                                    # 47 for the other release
CFG=configs/beta_vae_subgroup_g${REL}_genegrouped.json

# Pre-flight: every data path in the config exists; key settings are explicit
python - "$CFG" <<'PY'
import json, sys, pathlib
c = json.load(open(sys.argv[1]))
missing = [v for v in c["data"].values() if isinstance(v, str) and not pathlib.Path(v).exists()]
assert not missing, f"missing files: {missing}"
t, m = c["training"], c.get("model", {})
assert t.get("group_by_gene") is True, "group_by_gene must be true"
assert "lambda_conc" in t, "lambda_conc must be set explicitly"
print("group_by_gene:", t["group_by_gene"], "| lambda_conc:", t["lambda_conc"],
      "| block_sources:", m.get("block_sources"), "| out:", c["output"]["experiment_name"])
PY

python src/main_subgroup.py \
    --config "$CFG" \
    --device cuda:0

echo "Training complete at $(date)"
echo "Job ID: $SLURM_JOB_ID"