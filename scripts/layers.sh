#!/bin/bash
#SBATCH --job-name=layers
#SBATCH --output=logs/layers_input_%j.log
#SBATCH --error=logs/layers_input_%j.err
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#
# Layers 1-3 interpretability analyses for one trained experiment folder.
#
#   sbatch scripts/layers.sh <experiment_dir> [options]
#   bash   scripts/layers.sh <experiment_dir> --dry-run     # prints the plan
#
# Options
#   --config FILE     config used for training (default: configs/beta_vae_subgroup_<tag>.json
#                     where <tag> is the experiment folder name minus "beta_vae_subgroup_")
#   --only IDS        comma list of stage ids to run, e.g. 1.2,3.3
#   --force           rerun stages even if their .done marker matches
#   --dry-run         print commands, run nothing
#   --allow-legacy    do not fail when group_by_gene is false
#
# Feature-only experiments (architecture feature_only, or 'feature_only' in the config/experiment
# name) run stage 1.2b only; every other stage needs the token architecture.
#
# Resume: resubmit the same command. A stage is skipped when its marker holds the
# same command signature; changed parameters or --force rerun it.
#
# Stage ids (layer numbering as in the manuscript)
#   1.1 representations  1.2 subgroup ablation  1.3 residual ablation  1.4 latent probing
#   1.2b input-level ablation (zero/mean/permute; also runs on feature-only models)
#   2.1 Frechet          (per release, shared across experiments of that release)
#   3.1 patching matched 3.2 patching controls  3.3 joint search  3.4 joint controls
#   3.5 interventional results

set -euo pipefail

# ---------------------------------------------------------------- arguments
EXP=""; CFG=""; ONLY=""; FORCE=0; DRY=0; LEGACY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CFG="$2"; shift 2;;
    --only) ONLY="$2"; shift 2;;
    --force) FORCE=1; shift;;
    --dry-run) DRY=1; shift;;
    --allow-legacy) LEGACY=1; shift;;
    -*) echo "unknown option $1" >&2; exit 2;;
    *) EXP="${1%/}"; shift;;
  esac
done
[[ -n "$EXP" ]] || { echo "usage: layers.sh <experiment_dir> [options]" >&2; exit 2; }

# ---------------------------------------------------------------- environment
if (( ! DRY )); then
  mkdir -p logs
  module load nvidia/cuda/12.1
  eval "$(conda shell.bash hook)"
  conda activate beta_lncrna
fi
cd ${DATA_ROOT} 2>/dev/null || true

# ---------------------------------------------------------------- parameters
DEVICE=cuda:0
N_PAIRS_SINGLE=200        # single/block/all patching
N_SHUFFLED=200            # controls
N_PAIRS_JOINT=1000
MIN_CONF=0.7; MAX_LEN_DIFF=0.2; MAX_GC_DIFF=0.05
HELDOUT=0.5; DEPTH=10; CONTROL_DRAWS=5; FWD_BS=128
N_PERMUTATIONS=1000

REL=$(sed -n 's/.*gencode_v\([0-9]*\)_.*/\1/p' <<<"$EXP" | head -1)
[[ -n "$REL" ]] || { echo "cannot infer release from $EXP (expected gencode_v<NN>_...)" >&2; exit 2; }
if [[ -z "$CFG" ]]; then
  TAG=$(basename "$EXP" | sed 's/^beta_vae_subgroup_base_//')
  CFG="configs/beta_vae_subgroup_${TAG}.json"
fi

OUT="$EXP/interpretability"
REPR="$OUT/1.1_representations"
ABL="$OUT/1.2_ablation"
INABL="$OUT/1.2b_input_ablation"
RABL="$OUT/1.3_residual_ablation"
PROBE="$OUT/1.4_latent_probing"
PATCH="$OUT/3.1_patching"
PCTRL="$OUT/3.2_patching_controls"
JOINT="$OUT/3.3_joint_search"
JCTRL="$OUT/3.4_joint_controls"
FRECHET="$(dirname "$EXP")/frechet_analysis"
DONE="$OUT/.done"; LOGS="$OUT/logs"
GREEDY="$JOINT/joint_patching_greedy_all_folds.csv"

# ---------------------------------------------------------------- preflight
fail() { echo "PREFLIGHT FAILED: $*" >&2; exit 1; }
[[ -d "$EXP" ]] || fail "experiment dir not found: $EXP"
[[ -f "$CFG" ]] || fail "config not found: $CFG"
NF=$(python - "$CFG" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
t = c.get("training", {})
print(t.get("n_folds", 5), t.get("group_by_gene", False))
m = c.get("model", {})
print(str(m.get("architecture", m.get("type", c.get("architecture", "")))))
PY
)
read -r N_FOLDS GBG <<<"$(head -1 <<<"$NF")"
ARCH=$(sed -n 2p <<<"$NF")
FEATONLY=0
[[ "$ARCH" == *feature_only* || "$CFG" == *feature_only* || "$EXP" == *feature_only* ]] && FEATONLY=1
BASE=1; [[ -f "$EXP/models/fold_0_best.pt" ]] && BASE=0
for ((i=BASE; i<BASE+N_FOLDS; i++)); do
  [[ -f "$EXP/models/fold_${i}_best.pt" ]] || fail "missing checkpoint $EXP/models/fold_${i}_best.pt ($N_FOLDS folds expected)"
done
if [[ "$GBG" != "True" ]]; then
  (( LEGACY )) || fail "group_by_gene is $GBG in $CFG (use --allow-legacy for old, leaky models)"
  echo "WARNING: group_by_gene is $GBG: results are on gene-leaky folds" >&2
fi
echo "release=v$REL config=$CFG folds=$N_FOLDS group_by_gene=$GBG feature_only=$FEATONLY"
echo "out=$OUT  frechet=$FRECHET"

if (( ! DRY )); then
  mkdir -p "$DONE" "$LOGS"
  python - "$EXP" "$CFG" "$OUT" <<'PY'
import hashlib, json, os, subprocess, sys, time, glob
exp, cfg, out = sys.argv[1:4]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
m = dict(started=time.strftime("%FT%TZ", time.gmtime()), slurm_job=os.environ.get("SLURM_JOB_ID"),
         host=os.uname().nodename, git_commit=git("rev-parse", "HEAD"),
         git_dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
         config=cfg, config_sha=sha(cfg),
         checkpoints={p: sha(p) for p in sorted(glob.glob(f"{exp}/models/fold_*_best.pt"))})
json.dump(m, open(f"{out}/run_manifest_" + time.strftime("%Y%m%dT%H%M%S") + ".json", "w"), indent=2)
PY
fi

# ---------------------------------------------------------------- stage runner
stage() {  # stage <id> <name> <command...>
  local id="$1" name="$2"; shift 2
  if (( FEATONLY )) && [[ "$id" != "1.2b" ]]; then return 0; fi
  if [[ -n "$ONLY" && ",$ONLY," != *",$id,"* ]]; then return 0; fi
  local sig; sig=$(printf '%s ' "$@" | sha256sum | cut -c1-16)
  if (( ! FORCE )) && [[ -f "$DONE/$id" && "$(cat "$DONE/$id")" == "$sig" ]]; then
    echo "[skip ] $id $name"; return 0
  fi
  echo "[run  ] $id $name"
  if (( DRY )); then printf '      '; printf ' %q' "$@"; printf '\n'; return 0; fi
  local t0=$SECONDS
  "$@" 2>&1 | tee "$LOGS/$id.log"
  echo "$sig" > "$DONE/$id"
  echo "[done ] $id $name ($((SECONDS - t0))s)"
}
selected() { (( FEATONLY )) && [[ "$1" != "1.2b" ]] && return 1
  [[ -z "$ONLY" || ",$ONLY," == *",$1,"* ]]; }
# need <stage-id> <files...>: fail early if a selected stage lacks inputs
need() { local id="$1"; shift; selected "$id" || return 0
  for f in "$@"; do [[ -e "$f" || $DRY -eq 1 ]] || { echo "stage $id: missing input $f" >&2; exit 1; }; done; }

PAIR_ARGS=(--experiment_dir "$EXP" --config "$CFG" --fold all --patch_site projector
           --n_pairs "$N_PAIRS_JOINT" --heldout_frac "$HELDOUT"
           --min_confidence "$MIN_CONF" --max_length_diff "$MAX_LEN_DIFF" --max_gc_diff "$MAX_GC_DIFF"
           --forward_batch_size "$FWD_BS" --device "$DEVICE")
GATE=(--min_confidence "$MIN_CONF" --max_length_diff "$MAX_LEN_DIFF" --max_gc_diff "$MAX_GC_DIFF")

# ---------------------------------------------------------------- Layer 1
stage 1.1 "extract representations" \
  python -m analysis.pattern.extract_representations --experiment_dir "$EXP" --config "$CFG" \
    --output_dir "$REPR" --device "$DEVICE"

stage 1.2 "subgroup ablation" \
  python -m analysis.pattern.analyze_subgroup_ablation --experiment_dir "$EXP" --config "$CFG" \
    --output_dir "$ABL" --device "$DEVICE"

stage 1.2b "input-level ablation (zero/mean/permute)" \
  python -m analysis.pattern.analyze_input_ablation --experiment_dir "$EXP" --config "$CFG" \
    --output_dir "$INABL" --device "$DEVICE" --verify

stage 1.3 "residual ablation" \
  python -m analysis.pattern.analyze_residual_ablation --experiment_dir "$EXP" --repr_dir "$REPR" \
    --config "$CFG" --output_dir "$RABL" --device "$DEVICE"

stage 1.4 "latent probing + pattern alignment" \
  python -m analysis.pattern.analyze_latent_probing --repr_dir "$REPR" --output_dir "$PROBE" \
    --n_permutations "$N_PERMUTATIONS"

# ---------------------------------------------------------------- Layer 2
# Uses the trainval FASTAs/features named in the config; model-independent.
stage 2.1 "Frechet distance" \
  python -m analysis.pattern.analyze_frechet --config "$CFG" --output_dir "$FRECHET"

# ---------------------------------------------------------------- Layer 3
stage 3.1 "patching: matched (single, block, all) + shuffled" \
  python -m analysis.pattern.patch_tokens --experiment_dir "$EXP" --config "$CFG" \
    --output_dir "$PATCH" --device "$DEVICE" --n_pairs "$N_PAIRS_SINGLE" "${GATE[@]}" \
    --patch_site projector --patch_modes single,block,all,shuffled

stage 3.2 "patching: controls (shuffled, same-class, noise)" \
  python -m analysis.pattern.patch_tokens --experiment_dir "$EXP" --config "$CFG" \
    --output_dir "$PCTRL" --device "$DEVICE" --n_pairs "$N_PAIRS_SINGLE" "${GATE[@]}" \
    --patch_site projector --patch_modes shuffled,sameclass,noise --n_shuffled "$N_SHUFFLED"

stage 3.3 "joint search (greedy + 190 pairs)" \
  python -m analysis.pattern.joint_patching_search "${PAIR_ARGS[@]}" --output_dir "$JOINT" \
    --single_summary "$PATCH/patching_summary.csv" --mode all --max_search_depth "$DEPTH"

need 3.4 "$GREEDY"
stage 3.4 "joint controls (noise, same-class replays)" \
  python -m analysis.pattern.joint_patching_search "${PAIR_ARGS[@]}" --output_dir "$JCTRL" \
    --mode controls --greedy_csv "$GREEDY" --control_scope greedy --n_control_draws "$CONTROL_DRAWS"

stage 3.5 "collect interventional results" \
  python -m analysis.pattern.interventional_results --joint_dir "$JOINT" --controls_dir "$PCTRL" \
    --matched_summary "$PATCH/patching_summary.csv" --joint_controls_dir "$JCTRL" \
    --log "$LOGS/3.4.log" --out "$OUT/g${REL}_results.txt"

(( DRY )) && echo "dry run only: nothing executed" || echo "requested stages finished; results: $OUT/g${REL}_results.txt"