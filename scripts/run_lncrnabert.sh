#!/bin/bash
# =============================================================================
# run_lncrnabert.sh  --  lncRNA-BERT (3-mer, zero-shot): inference + conversion
#
# 1. lncrnapy.scripts.classify on <pc_test.fa> <lnc_test.fa>   (GPU)
# 2. convert_lncrnabert_predictions.py -> benchmark schema
# 3. validate: one row per test record, no unmatched ids
# Optional: lncrnapy.scripts.embeddings (--embeddings), not used by the benchmark.
#
# Output (under --output_dir, the benchmark_tools dir):
#   lncrnabert/lncrnabert_raw.csv               raw classify output
#   lncrnabert_test_predictions.csv             benchmark schema (compare_models.py)
#   lncrnabert/lncrnabert_provenance.json       model, fasta hashes, counts
#
# Usage
#   bash analysis/benchmark/run_lncrnabert.sh \
#       --lnc_fasta <split>/lnc_test.fa --pc_fasta <split>/pc_test.fa \
#       --output_dir <base>/benchmark_tools [--release v49] [--env lncrnabert] \
#       [--batch_size 8] [--model luukromeijn/lncRNA-BERT-kmer-k3-finetuned] \
#       [--workdir .] [--embeddings] [--force]
# =============================================================================
set -eo pipefail

ENV_NAME="lncrnabert"; CONVERT_ENV=""; BATCH=8; K=3; RELEASE="unknown"
MODEL="luukromeijn/lncRNA-BERT-kmer-k3-finetuned"; WORKDIR="."
EMBEDDINGS=false; FORCE=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lnc_fasta)    LNC_FASTA="$2"; shift 2 ;;
    --pc_fasta)     PC_FASTA="$2"; shift 2 ;;
    --output_dir)   OUTPUT_DIR="$2"; shift 2 ;;
    --release)      RELEASE="$2"; shift 2 ;;
    --env)          ENV_NAME="$2"; shift 2 ;;
    --convert_env)  CONVERT_ENV="$2"; shift 2 ;;
    --batch_size)   BATCH="$2"; shift 2 ;;
    --model)        MODEL="$2"; shift 2 ;;
    --workdir)      WORKDIR="$2"; shift 2 ;;
    --embeddings)   EMBEDDINGS=true; shift 1 ;;
    --force)        FORCE=true; shift 1 ;;
    --convert_script) CONVERT_SCRIPT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done
CONVERT_ENV="${CONVERT_ENV:-$ENV_NAME}"; CONVERT_SCRIPT="${CONVERT_SCRIPT:-}"
for v in LNC_FASTA PC_FASTA OUTPUT_DIR; do
  [[ -n "${!v:-}" ]] || { echo "ERROR: --${v,,} is required"; exit 1; }
done
for f in "$LNC_FASTA" "$PC_FASTA"; do [[ -f "$f" ]] || { echo "ERROR: not found: $f"; exit 1; }; done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# converter: explicit flag, else next to this script, else analysis/benchmark (repo root or workdir)
if [[ -z "$CONVERT_SCRIPT" ]]; then
  for c in "$HERE/convert_lncrnabert_predictions.py" "$HERE/../analysis/benchmark/convert_lncrnabert_predictions.py" \
           "$WORKDIR/analysis/benchmark/convert_lncrnabert_predictions.py" "analysis/benchmark/convert_lncrnabert_predictions.py"; do
    [[ -f "$c" ]] && { CONVERT_SCRIPT="$c"; break; }
  done
fi
[[ -f "$CONVERT_SCRIPT" ]] || { echo "ERROR: convert_lncrnabert_predictions.py not found (pass --convert_script)"; exit 1; }
CONVERT_SCRIPT="$(realpath "$CONVERT_SCRIPT")"
LNC_FASTA="$(realpath "$LNC_FASTA")"; PC_FASTA="$(realpath "$PC_FASTA")"
mkdir -p "$OUTPUT_DIR"; OUT="$(realpath "$OUTPUT_DIR")"
RAWDIR="$OUT/lncrnabert"; mkdir -p "$RAWDIR"
RAW="$RAWDIR/lncrnabert_raw.csv"
FINAL="$OUT/lncrnabert_test_predictions.csv"
PROV="$RAWDIR/lncrnabert_provenance.json"
LOG="$RAWDIR/lncrnabert_${RELEASE}_$(date +%Y%m%d_%H%M%S).log"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
cr() { # conda run helper; -p for path envs
  local e="$1"; shift
  if [[ "$e" == /* || "$e" == ./* ]]; then conda run --no-capture-output -p "$e" "$@"
  else conda run --no-capture-output -n "$e" "$@"; fi
}

if [[ -f "$FINAL" && "$FORCE" != "true" ]]; then
  log "lncRNA-BERT: already done ($FINAL) -- skipping (use --force to redo)"; exit 0
fi

# ── 1. inference ────────────────────────────────────────────────────────────
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  log "WARNING: no GPU visible; inference will be very slow. Run on a GPU node."
fi
if [[ -f "$RAW" && "$FORCE" != "true" ]]; then
  log "Raw predictions exist ($RAW) -- skipping inference"
else
  log "Step 1: classify (model=$MODEL, k=$K, batch=$BATCH, env=$ENV_NAME)"
  rm -f "$RAW"
  # lncrnapy prefixes './' to every path it is given, so absolute paths break
  # ('.//mnt/...'). Pass paths relative to the working directory instead.
  REL_PC="$(realpath --relative-to="$WORKDIR" "$PC_FASTA")"
  REL_LNC="$(realpath --relative-to="$WORKDIR" "$LNC_FASTA")"
  REL_RES="$(realpath --relative-to="$WORKDIR" "$RAWDIR")/"
  ( cd "$WORKDIR" && cr "$ENV_NAME" python -m lncrnapy.scripts.classify \
      "$REL_PC" "$REL_LNC" \
      --output_file "$(basename "$RAW")" --batch_size "$BATCH" --k "$K" \
      --model_file "$MODEL" --encoding_method kmer --results_dir "$REL_RES" ) 2>&1 | tee -a "$LOG"
  if [[ ! -f "$RAW" ]]; then
    # classify may place the file relative to its own results root; look for it
    FOUND=$(find "$RAWDIR" "$WORKDIR" -maxdepth 4 -name "$(basename "$RAW")" -newer "$LNC_FASTA" 2>/dev/null | head -1 || true)
    [[ -n "$FOUND" ]] && { log "  found output at $FOUND"; cp "$FOUND" "$RAW"; }
  fi
  [[ -f "$RAW" ]] || { log "ERROR: classify produced no $(basename "$RAW")"; exit 1; }
fi

# ── 2. convert ──────────────────────────────────────────────────────────────
log "Step 2: convert to benchmark schema (env=$CONVERT_ENV)"
cr "$CONVERT_ENV" python "$CONVERT_SCRIPT" \
    --pred_csv "$RAW" --lnc_fasta "$LNC_FASTA" --pc_fasta "$PC_FASTA" \
    --output_csv "$FINAL" 2>&1 | tee -a "$LOG"

# ── 3. validate + provenance ────────────────────────────────────────────────
log "Step 3: validate"
python3 - "$LNC_FASTA" "$PC_FASTA" "$RAW" "$FINAL" "$PROV" "$MODEL" "$BATCH" "$K" <<'PY' 2>&1 | tee -a "$LOG" || { rm -f "$FINAL"; log "ERROR: validation failed; removed $FINAL"; exit 1; }
import csv, hashlib, json, sys, datetime
lnc, pc, raw, final, prov, model, batch, k = sys.argv[1:9]
def ids(f):
    return [l[1:].split()[0].split("|")[0] for l in open(f) if l.startswith(">")]
exp = ids(lnc) + ids(pc)
got = [r["transcript_id"] for r in csv.DictReader(open(final))]
rawn = sum(1 for _ in csv.reader(open(raw))) - 1
bad = []
if len(set(exp)) != len(exp): bad.append("duplicate ids across test FASTAs")
if rawn != len(exp): bad.append(f"raw rows {rawn} != FASTA records {len(exp)}")
if set(got) != set(exp): bad.append(f"converted ids differ from FASTA ids: {len(set(exp)-set(got))} missing, {len(set(got)-set(exp))} extra")
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
json.dump(dict(created=datetime.datetime.now().isoformat(timespec="seconds"), model=model,
               k=int(k), batch_size=int(batch), encoding="kmer", n_records=len(exp),
               n_converted=len(got), sha256_16=dict(lnc_fasta=h(lnc), pc_fasta=h(pc), raw=h(raw), final=h(final)),
               problems=bad), open(prov, "w"), indent=2)
if bad:
    sys.exit("VALIDATION FAILED: " + "; ".join(bad))
print(f"  OK: {len(got)} records, ids match both FASTAs")
PY

# ── optional embeddings (not used by the benchmark) ─────────────────────────
if $EMBEDDINGS; then
  log "Embeddings (optional)"
  ( cd "$WORKDIR" && cr "$ENV_NAME" python -m lncrnapy.scripts.embeddings \
      "$(realpath --relative-to="$WORKDIR" "$PC_FASTA")" "$(realpath --relative-to="$WORKDIR" "$LNC_FASTA")" \
      --output_file "lncrnabert_embeddings_${RELEASE}.h5" \
      --batch_size "$BATCH" --k "$K" --model_file "$MODEL" --encoding_method kmer \
      --results_dir "$(realpath --relative-to="$WORKDIR" "$RAWDIR")/" --dim_red None ) 2>&1 | tee -a "$LOG"
fi
log "lncRNA-BERT done: $FINAL"