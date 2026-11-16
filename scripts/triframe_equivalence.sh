#!/bin/bash
#SBATCH --job-name=triframe_equivalence
#SBATCH --output=logs/triframe_equivalence_%j.log
#SBATCH --error=logs/triframe_equivalence_%j.err
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --partition=gpu
#SBATCH --time=72:00:00
#SBATCH --mem=16G

module load nvidia/cuda/12.1                # Load CUDA module (adjust version as needed)
eval "$(conda shell.bash hook)"
conda activate beta_lncrna

cd ${DATA_ROOT}
set -euo pipefail

# Usage: sbatch <this script> [LAYER ...]     LAYER = 1, 2 and/or 3 (also "1,2" or "12"); default: 1 2 3
#   1 = probing (1.4) + ablation (1.2)   2 = Fréchet (2.1)   3 = patching (3.1) + greedy (3.3)
want=""
for arg in "${@:-1 2 3}"; do
  for l in $(echo "$arg" | grep -o '[0-9]'); do
    case "$l" in 1|2|3) want+="$l" ;; *) echo "unknown layer '$l' (use 1, 2 and/or 3)" >&2; exit 2 ;; esac
  done
done
has() { [[ "$want" == *"$1"* ]]; }

# Reports of a partial run go to <tag>_L<layers> so they do not overwrite a full report.
suffix=""
for l in 1 2 3; do has "$l" && suffix+="$l"; done
[[ "$suffix" == "123" ]] && suffix="" || suffix="_L$suffix"

EXP47=gencode_v47_experiments_genegrouped/beta_vae_subgroup_base_g47_genegrouped_binfix
EXP49=gencode_v49_experiments_genegrouped/beta_vae_subgroup_base_g49_genegrouped_binfix

# Selected layers against the released β-LNC outputs of one experiment.
#   compare <tag> <experiment_dir> <config>
#
# --batch_size 1 (patching, 3.1): each pair goes through the model alone, so triframe and
# β-LNC see identical inputs. Batched forwards differ at ~1e-4 in the logits (float32
# reduction order depends on batch size); batch size 1 brings the gap under 1e-5, which is
# what --atol_logit checks. Other stages: --forward_batch_size only affects speed.
compare() {
  local tag="$1" exp="${2%/}" cfg="$3"
  local interp="$exp/interpretability"
  local mstar="$(dirname "$exp")/mstar_analysis"
  local args=(--experiment_dir "$exp" --config "$cfg"
              --folds all --output_dir "analysis/triframe/reports/${tag}${suffix}")
  if has 1; then
    args+=(--repr_dir "$interp/1.1_representations"
           --probing_dir "$interp/1.4_latent_probing" --probing_atol 1e-5
           --ablation_reference "$interp/1.2_ablation/all_folds_ablation.csv"
           --ablation_modes token_zero,feature_zero --ablation_atol 1e-9
           --num_workers 2 --forward_batch_size 128)
  fi
  if has 2; then
    args+=(--frechet_reference "$mstar/frechet_ranking.csv"
           --frechet_rtol 1e-6 --frechet_atol 1e-8)
  fi
  if has 3; then
    args+=(--reference "$interp/3.1_patching/all_folds_patching.csv"
           --greedy_reference "$interp/3.3_joint_search/joint_patching_greedy_all_folds.csv"
           --max_pairs 0 --batch_size 1 --atol_logit 1e-5)
  fi
  python -m analysis.triframe.compare_with_triframe "${args[@]}"
}

echo "layers: ${want}  (reports: analysis/triframe/reports/*${suffix})"
rc=0
compare g47 "$EXP47" configs/beta_vae_subgroup_g47_genegrouped.json || rc=1
compare g49 "$EXP49" configs/beta_vae_subgroup_g49_genegrouped.json || rc=1
exit $rc