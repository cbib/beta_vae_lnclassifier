#!/bin/bash
#SBATCH --job-name=train_subgroup
#SBATCH --output=logs/train_subgroup_%j.log
#SBATCH --error=logs/train_subgroup_%j.err
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --partition=gpu
#SBATCH --time=70:00:00                 # hh:mm:ss
#SBATCH --mem=24G
#
# Train the full beta-LNC model (src/main_subgroup.py): gene-grouped 5-fold CV for one release.
#
#   sbatch scripts/train_beta_vae_subgroup.sh <release>          # 47 or 49
#
# Config: configs/beta_vae_subgroup_base_g<release>.json
# Needs DATA_ROOT (repository root: data/, configs/ and the experiment folders).
# Job name and log files can be overridden on the sbatch command line (submit_pipeline.sh does).

module load nvidia/cuda/12.1
eval "$(conda shell.bash hook)"
conda activate beta_lncrna

set -euo pipefail
cd "${DATA_ROOT:?DATA_ROOT is not set}"

REL="${1:-}"
[[ "$REL" =~ ^[0-9]{2}$ ]] || { echo "usage: sbatch scripts/train_beta_vae_subgroup.sh <release, e.g. 47 or 49>" >&2; exit 2; }
CFG=configs/beta_vae_subgroup_base_g${REL}.json
[[ -f "$CFG" ]] || { echo "config not found: $CFG" >&2; exit 1; }

# Pre-flight: every data path in the config exists; key settings are explicit
python - "$CFG" <<'PY'
import json, sys, pathlib
c = json.load(open(sys.argv[1]))
missing = [v for v in c["data"].values() if isinstance(v, str) and not pathlib.Path(v).exists()]
assert not missing, f"missing files: {missing}"
t, m = c["training"], c.get("model", {})
assert t.get("group_by_gene") is True, "group_by_gene must be true"
print("group_by_gene:", t["group_by_gene"],
      "| block_sources:", m.get("block_sources"), "| out:", c["output"]["experiment_name"])
PY

python src/main_subgroup.py \
    --config "$CFG" \
    --device cuda:0

echo "Training complete at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-none}"