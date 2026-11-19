#!/bin/bash
# Stage 0 of the pipeline: fit the feature scaler banks and write the cleaned feature tables
# for one release (data/prep_features.py), then audit the scalers.
#
#   bash scripts/prep_features.sh <release> --te_csv FILE --nonb_csv FILE [--nonb2_csv FILE] [--chunk_size N]
#
#   <release>      47 or 49
#   --te_csv       raw transcript-level TE feature table
#   --nonb_csv     raw NonB DNA feature table
#   --nonb2_csv    merged NonB2 table (data/merge_nonb2.py output); optional
#
# Everything else is read from configs/beta_vae_subgroup_base_g<release>.json:
#   split folder  = folder of data.lnc_fasta (lnc_trainval.fa and pc_trainval.fa; scalers are
#                   fitted on these transcripts only, never on the test set)
#   output folder = folder of data.te_genomic_csv (cleaned tables and scaler banks are written there)
# CPU only. Needs DATA_ROOT (repository root). Run it on a compute node (srun or sbatch).

set -euo pipefail
cd "${DATA_ROOT:?DATA_ROOT is not set}"

REL="${1:-}"
[[ "$REL" =~ ^[0-9]{2}$ ]] || { echo "usage: bash scripts/prep_features.sh <release> --te_csv FILE --nonb_csv FILE [--nonb2_csv FILE]" >&2; exit 2; }
shift
TE=""; NONB=""; NONB2=""; CHUNK=50000
while [[ $# -gt 0 ]]; do
  case "$1" in
    --te_csv) TE="$2"; shift 2;;
    --nonb_csv) NONB="$2"; shift 2;;
    --nonb2_csv) NONB2="$2"; shift 2;;
    --chunk_size) CHUNK="$2"; shift 2;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
[[ -f "$TE" && -f "$NONB" ]] || { echo "--te_csv and --nonb_csv must be existing files" >&2; exit 2; }
[[ -z "$NONB2" || -f "$NONB2" ]] || { echo "--nonb2_csv not found: $NONB2" >&2; exit 2; }

CFG=configs/beta_vae_subgroup_base_g${REL}.json
[[ -f "$CFG" ]] || { echo "config not found: $CFG" >&2; exit 1; }
read -r SPLIT_DIR OUT_DIR < <(python3 - "$CFG" <<'PY'
import json, sys
from pathlib import Path
d = json.load(open(sys.argv[1]))["data"]
split = {str(Path(d[k]).parent) for k in ("lnc_fasta", "pc_fasta")}
out = {str(Path(d[k]).parent) for k in ("te_genomic_csv", "nonb_genomic_csv", "nonb2_csv") if d.get(k)}
assert len(split) == 1 and len(out) == 1, f"config points to several folders: split={split} out={out}"
print(split.pop(), out.pop())
PY
)
for f in lnc_trainval.fa pc_trainval.fa; do
  [[ -f "$SPLIT_DIR/$f" ]] || { echo "missing $SPLIT_DIR/$f" >&2; exit 1; }
done

# never overwrite an input table with its own cleaned version
for blk in te nonb nonb2; do
  tgt="$OUT_DIR/g${REL}_${blk}_features_clean.csv"
  for src in "$TE" "$NONB" "$NONB2"; do
    [[ -n "$src" && -e "$tgt" && "$(realpath "$src")" == "$(realpath "$tgt")" ]] \
      && { echo "input $src would be overwritten by $tgt: write the cleaned tables elsewhere" >&2; exit 1; }
  done
done

echo "release=g$REL split=$SPLIT_DIR out=$OUT_DIR"
ARGS=(--te_csv "$TE" --nonb_csv "$NONB" --split_dir "$SPLIT_DIR" --output_dir "$OUT_DIR"
      --release "g$REL" --chunk_size "$CHUNK")
[[ -n "$NONB2" ]] && ARGS+=(--nonb2_csv "$NONB2")
python -m data.prep_features "${ARGS[@]}"

# scalers stage: stored banks must equal a re-fit on the trainval rows
python scripts/pipeline_audit.py --config "$CFG" --stages config,scalers --out "audit_prep_g${REL}"