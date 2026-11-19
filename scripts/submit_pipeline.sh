#!/bin/bash
# Submit the pipeline for one release, with job dependencies.
#
#   bash scripts/submit_pipeline.sh --release <47|49> [--skip STAGES] [--retrain] [--dry-run]
#
# Stages (comma list for --skip):
#   train               scripts/train_beta_vae_subgroup.sh      full model
#   feature_only_train  scripts/train_beta_vae_feature_only.sh  feature-only baseline
#   layers              scripts/layers.sh on both models, each after its own training
#                       (the feature-only model runs stage 1.2b only)
#   benchmark           scripts/benchmark_job.sh, after both trainings (evaluates, then benchmarks)
#
# Dependencies are afterok. A skipped stage must already be finished: its checkpoints are checked here.
# A model that already has checkpoints is not retrained unless --retrain is given.
# Experiment folders, split folder and benchmark root are read from the release's configs.
# DRY=1 or --dry-run prints the sbatch commands without submitting anything.
# Needs DATA_ROOT (repository root).

set -euo pipefail
cd "${DATA_ROOT:?DATA_ROOT is not set}"

REL=""; SKIP=""; RETRAIN=0; DRY="${DRY:+1}"; DRY="${DRY:-0}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --release) REL="$2"; shift 2;;
    --skip) SKIP="$2"; shift 2;;
    --retrain) RETRAIN=1; shift;;
    --dry-run) DRY=1; shift;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
[[ "$REL" =~ ^[0-9]{2}$ ]] || { echo "usage: submit_pipeline.sh --release <47|49> [--skip STAGES] [--retrain] [--dry-run]" >&2; exit 2; }
for s in ${SKIP//,/ }; do
  case "$s" in train|feature_only_train|layers|benchmark) ;; *) echo "unknown stage in --skip: $s" >&2; exit 2;; esac
done
skip() { [[ ",$SKIP," == *",$1,"* ]]; }
problem() { if (( DRY )); then echo "WARNING (dry run): $*" >&2; else echo "ERROR: $*" >&2; exit 1; fi; }

CFG_FULL=configs/beta_vae_subgroup_base_g${REL}.json
CFG_FEAT=configs/beta_vae_feature_only_base_g${REL}.json
for f in "$CFG_FULL" "$CFG_FEAT"; do [[ -f "$f" ]] || { echo "config not found: $f" >&2; exit 1; }; done
cfg() { python3 - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
c = json.load(open(sys.argv[1]))
if sys.argv[2] == "exp":
    print(c["output"]["experiment_name"])
elif sys.argv[2] == "split":
    print(Path(c["data"]["lnc_test_fasta"]).parent)
PY
}
EXP_FULL=$(cfg "$CFG_FULL" exp); EXP_FEAT=$(cfg "$CFG_FEAT" exp)
SPLIT=$(cfg "$CFG_FULL" split); BASE=$(dirname "$EXP_FULL")
has_ckpt() { compgen -G "$1/models/fold_*_best.pt" >/dev/null; }

# ---------------------------------------------------------------- checks before anything is queued
if skip train;              then has_ckpt "$EXP_FULL" || problem "train skipped but no checkpoints in $EXP_FULL/models"
elif has_ckpt "$EXP_FULL" && (( ! RETRAIN )); then problem "$EXP_FULL already has checkpoints (use --skip train, or --retrain)"; fi
if skip feature_only_train; then has_ckpt "$EXP_FEAT" || problem "feature_only_train skipped but no checkpoints in $EXP_FEAT/models"
elif has_ckpt "$EXP_FEAT" && (( ! RETRAIN )); then problem "$EXP_FEAT already has checkpoints (use --skip feature_only_train, or --retrain)"; fi

mkdir -p logs
sub() {   # sub <label> <sbatch args...>  -> prints the job id
  local label="$1"; shift
  if (( DRY )); then echo "sbatch $*" >&2; echo "<$label>"; else sbatch --parsable "$@"; fi
}
dep() {   # dep <job ids...>  -> sbatch dependency options (nothing when there are no ids)
  local d=""; for j in "$@"; do [[ -n "$j" ]] && d+=":$j"; done
  [[ -n "$d" ]] && echo "--dependency=afterok$d --kill-on-invalid-dep=yes" || true
}
logs() { echo "--job-name=g${REL}_$1 --output=logs/g${REL}_$1_%j.log --error=logs/g${REL}_$1_%j.err"; }

J_TRAIN=""; J_FEAT=""; J_LAY=""; J_LAY_FO=""; J_BENCH=""
skip train              || J_TRAIN=$(sub train $(logs train) scripts/train_beta_vae_subgroup.sh "$REL")
skip feature_only_train || J_FEAT=$(sub feature_only_train $(logs feature_only) scripts/train_beta_vae_feature_only.sh "$REL")
if ! skip layers; then
  J_LAY=$(sub layers $(dep "$J_TRAIN") $(logs layers) scripts/layers.sh "$EXP_FULL" --config "$CFG_FULL")
  J_LAY_FO=$(sub layers_feature_only $(dep "$J_FEAT") $(logs layers_feature_only) scripts/layers.sh "$EXP_FEAT" --config "$CFG_FEAT")
fi
skip benchmark || J_BENCH=$(sub benchmark $(dep "$J_TRAIN" "$J_FEAT") $(logs benchmark) scripts/benchmark_job.sh \
    --release "$REL" --exp_dir "$EXP_FULL" --config "$CFG_FULL" --split_dir "$SPLIT" --base_dir "$BASE" \
    --feature_exp_dir "$EXP_FEAT" --feature_config "$CFG_FEAT" --evaluate)

echo "release g$REL: train=${J_TRAIN:--} feature_only_train=${J_FEAT:--} layers=${J_LAY:--} layers_feature_only=${J_LAY_FO:--} benchmark=${J_BENCH:--}"