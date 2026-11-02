#!/bin/bash
# Submit the G49 then G47 benchmarks once the feature-only models exist.
#   FEAT49_EXP=<dir> FEAT47_EXP=<dir> FEAT49_CFG=<cfg> FEAT47_CFG=<cfg> \
#   [FEAT49_JOB=<id> FEAT47_JOB=<id> TRAIN47_JOB=<id>] [EXP47=<dir>] [EXP49=<dir>] \
#   bash scripts/submit_benchmarks.sh        (DRY=1 to print only)
# *_JOB ids are optional afterok dependencies on still-running trainings.
set -euo pipefail
cd ${DATA_ROOT}; mkdir -p logs
: "${FEAT49_EXP:?}" "${FEAT47_EXP:?}" "${FEAT49_CFG:?}" "${FEAT47_CFG:?}"
CFG47=configs/beta_vae_subgroup_g47_genegrouped.json
EXP49=${EXP49:-gencode_v49_experiments_genegrouped/beta_vae_subgroup_base_g49_genegrouped}
EXP47=${EXP47:-$(python3 -c "import json;print(json.load(open('$CFG47'))['output']['experiment_name'])")}
sub() { if [[ -n "${DRY:-}" ]]; then echo "sbatch $*" >&2; echo "JOB_$RANDOM"; else sbatch --parsable "$@"; fi; }
dep() { local d=""; for x in "$@"; do [[ -n "$x" ]] && d+="${d:+,}$x"; done; [[ -n "$d" ]] && echo "--dependency=$d" || true; }

#D49=$(dep ${FEAT49_JOB:+afterok:$FEAT49_JOB})
#J1=$(sub $D49 scripts/benchmark_job.sh --release 49 --exp_dir "$EXP49" \
#        --feature_exp_dir "$FEAT49_EXP" --feature_config "$FEAT49_CFG" --evaluate)
D47=$(dep ${TRAIN47_JOB:+afterok:$TRAIN47_JOB} ${FEAT47_JOB:+afterok:$FEAT47_JOB} afterany:$J1)
J2=$(sub $D47 scripts/benchmark_job.sh --release 47 --exp_dir "$EXP47" \
        --feature_exp_dir "$FEAT47_EXP" --feature_config "$FEAT47_CFG" --evaluate)
#echo "bench49=$J1 bench47=$J2"
echo "bench47=$J2"