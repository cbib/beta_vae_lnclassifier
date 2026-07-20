
#!/bin/bash
# =============================================================================
# run_benchmark_tools.sh
#
# Runs all benchmark tools on a given test set FASTA pair (lncRNA + mRNA)
# and produces outputs ready for compare_models.py.
#
# Tools and environments:
#   CPAT              — beta_lncrna conda env (+ R module)
#   CPC2              — beta_lncrna conda env
#   LncDC             — lncdc conda env
#   Orthrus 4-track   — orthrus conda env
#
# Usage
# -----
# bash analysis/run_benchmark_tools.sh \
#     --lnc_fasta          data/split_gencode_49/lnc_test.fa \
#     --pc_fasta           data/split_gencode_49/pc_test.fa \
#     --train_lnc_fasta    data/split_gencode_49/lnc_trainval.fa \
#     --train_pc_fasta     data/split_gencode_49/pc_trainval.fa \
#     --blnc_predictions   gencode_v49_experiments/beta_vae_subgroup_base_g49/evaluation_csvs/test_predictions.csv \
#     --feature_only_csv   gencode_v49_experiments/beta_vae_feature_only_g49/evaluation_csvs/test_predictions.csv \
#     --output_dir         gencode_v49_experiments/benchmark_tools \
#     --release            v49 \
#     [--cpat_hex          cpat/data/Human_Hexamer.tsv] \
#     [--cpat_model        cpat/data/Human_logitModel.RData] \
#     [--cpc2_dir          cpc2/CPC2_standalone-1.0.1] \
#     [--lncdc_dir         LncDC-1.3.6] \
#     [--r_module          R/4.1.0] \
#     [--orthrus_model     Orthrus/models/orthrus-4-track] \
#     [--orthrus_env       orthrus] \
#     [--threads           8] \
#     [--max_length        15000] \
#     [--rnasamba_weights  rnasamba/full_length_weights.hdf5] \
#     [--rnasamba_sif      rnasamba.sif]  # Singularity image, see note below \
#     [--skip_cpat] [--skip_cpc2] [--skip_lncdc] [--skip_orthrus] [--skip_rnasamba]
# =============================================================================

cd /mnt/cbib/LNClassifier/beta_vae_lnclassifier
set -eo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
CPAT_HEX="cpat/data/Human_Hexamer.tsv"
CPAT_MODEL="cpat/data/Human_logitModel.RData"
CPC2_DIR="cpc2/CPC2_standalone-1.0.1"
LNCDC_DIR="LncDC-1.3.6"
R_MODULE="R/4.1.0"
ORTHRUS_MODEL="Orthrus/models/orthrus-4-track"
ORTHRUS_ENV="orthrus"
THREADS=8
RELEASE="unknown"
MAX_LENGTH=15000
SKIP_CPAT=false
SKIP_CPC2=false
SKIP_LNCDC=false
SKIP_ORTHRUS=false
SKIP_RNASAMBA=false
RNASAMBA_WEIGHTS="rnasamba/full_length_weights.hdf5"
TRAIN_LNC_FASTA=""
TRAIN_PC_FASTA=""
BLNC_PREDICTIONS=""
FEATURE_ONLY_PREDICTIONS=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --lnc_fasta)          LNC_FASTA="$2";          shift 2 ;;
        --pc_fasta)           PC_FASTA="$2";           shift 2 ;;
        --train_lnc_fasta)    TRAIN_LNC_FASTA="$2";    shift 2 ;;
        --train_pc_fasta)     TRAIN_PC_FASTA="$2";     shift 2 ;;
        --blnc_predictions)   BLNC_PREDICTIONS="$2";   shift 2 ;;
        --feature_only_csv)   FEATURE_ONLY_PREDICTIONS="$2"; shift 2 ;;
        --output_dir)         OUTPUT_DIR="$2";         shift 2 ;;
        --release)            RELEASE="$2";            shift 2 ;;
        --cpat_hex)           CPAT_HEX="$2";           shift 2 ;;
        --cpat_model)         CPAT_MODEL="$2";         shift 2 ;;
        --cpc2_dir)           CPC2_DIR="$2";           shift 2 ;;
        --lncdc_dir)          LNCDC_DIR="$2";          shift 2 ;;
        --r_module)           R_MODULE="$2";           shift 2 ;;
        --orthrus_model)      ORTHRUS_MODEL="$2";      shift 2 ;;
        --orthrus_env)        ORTHRUS_ENV="$2";        shift 2 ;;
        --threads)            THREADS="$2";            shift 2 ;;
        --max_length)         MAX_LENGTH="$2";         shift 2 ;;
        --skip_cpat)          SKIP_CPAT=true;          shift 1 ;;
        --skip_cpc2)          SKIP_CPC2=true;          shift 1 ;;
        --skip_lncdc)         SKIP_LNCDC=true;         shift 1 ;;
        --skip_orthrus)       SKIP_ORTHRUS=true;       shift 1 ;;
        --skip_rnasamba)      SKIP_RNASAMBA=true;      shift 1 ;;
        --rnasamba_weights)   RNASAMBA_WEIGHTS="$2";   shift 2 ;;
        --rnasamba_sif)       RNASAMBA_SIF="$2";       shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Validate required args ────────────────────────────────────────────────────
for var in LNC_FASTA PC_FASTA OUTPUT_DIR; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: --${var,,} is required"
        exit 1
    fi
done

for f in "$LNC_FASTA" "$PC_FASTA"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: FASTA file not found: $f"
        exit 1
    fi
done

# Orthrus requires train FASTAs — warn if missing
if ! $SKIP_ORTHRUS; then
    if [[ -z "$TRAIN_LNC_FASTA" ]] || [[ -z "$TRAIN_PC_FASTA" ]]; then
        echo "WARNING: --train_lnc_fasta / --train_pc_fasta not provided — skipping Orthrus"
        SKIP_ORTHRUS=true
    elif [[ ! -f "$TRAIN_LNC_FASTA" ]] || [[ ! -f "$TRAIN_PC_FASTA" ]]; then
        echo "WARNING: Train FASTAs not found — skipping Orthrus"
        SKIP_ORTHRUS=true
    fi
fi

# CPAT hard cases step requires β-LNC predictions — warn if missing
RUN_CPAT_HARD=false
if ! $SKIP_CPAT && [[ -n "$BLNC_PREDICTIONS" ]] && [[ -f "$BLNC_PREDICTIONS" ]]; then
    RUN_CPAT_HARD=true
fi

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${OUTPUT_DIR}/benchmark_${RELEASE}_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log_section() {
    echo "" | tee -a "$LOG_FILE"
    echo "$(printf '=%.0s' {1..65})" | tee -a "$LOG_FILE"
    echo "  $*" | tee -a "$LOG_FILE"
    echo "$(printf '=%.0s' {1..65})" | tee -a "$LOG_FILE"
}

log_section "Benchmark tools — GENCODE ${RELEASE}"
log "lncRNA FASTA    : ${LNC_FASTA}"
log "mRNA FASTA      : ${PC_FASTA}"
log "Output dir      : ${OUTPUT_DIR}"
log "Threads         : ${THREADS}"
log "Log file        : ${LOG_FILE}"
[[ -n "$BLNC_PREDICTIONS" ]] && log "β-LNC preds     : ${BLNC_PREDICTIONS}"

# ── Step 0: Merge FASTAs into one input file ──────────────────────────────────
log_section "Step 0: Merging FASTAs"

MERGED_FASTA="${OUTPUT_DIR}/test_sequences.fa"
if [[ -f "$MERGED_FASTA" ]]; then
    log "Merged FASTA already exists — skipping"
else
    conda run -n beta_lncrna python3 -c "
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
out = []
for path in ['${LNC_FASTA}', '${PC_FASTA}']:
    for rec in SeqIO.parse(path, 'fasta'):
        dna = str(rec.seq).upper().replace('U', 'T')
        out.append(SeqRecord(Seq(dna), id=rec.id.split('|')[0], name='', description=''))
SeqIO.write(out, '${MERGED_FASTA}', 'fasta')
print(f'Written {len(out):,} sequences')
" >> "\$LOG_FILE" 2>&1

if [[ ! -f "\$MERGED_FASTA" ]]; then
    log "   FASTA merge failed — \${MERGED_FASTA} not created"
    exit 1
fi
log "Merged FASTA written: \${MERGED_FASTA}"
fi

# ── Step 1: CPAT ──────────────────────────────────────────────────────────────
CPAT_DIR="${OUTPUT_DIR}/cpat"
CPAT_OUT="${CPAT_DIR}/cpat_output"
CPAT_RESULT="${CPAT_OUT}.ORF_prob.best.tsv"
CPAT_HARD_CSV="${CPAT_DIR}/predictions_with_cpat.csv"

if $SKIP_CPAT; then
    log_section "CPAT: skipped"
elif [[ -f "$CPAT_RESULT" ]]; then
    log_section "CPAT: already done — skipping raw predictions"
    log "  Found: ${CPAT_RESULT}"
else
    log_section "Step 1: CPAT"
    mkdir -p "$CPAT_DIR"
    log "  Hex table : ${CPAT_HEX}"
    log "  Model     : ${CPAT_MODEL}"

    CPAT_CMD="source /etc/profile.d/modules.sh && \
module load ${R_MODULE} && \
conda run -n beta_lncrna cpat \
    --gene       ${MERGED_FASTA} \
    --hex        ${CPAT_HEX} \
    --logitModel ${CPAT_MODEL} \
    --out        ${CPAT_OUT}"

    log "  Running CPAT..."
    if bash -c "$CPAT_CMD" >> "$LOG_FILE" 2>&1; then
        if [[ -f "$CPAT_RESULT" ]]; then
            N=$(wc -l < "$CPAT_RESULT")
            log "   CPAT done: ${CPAT_RESULT} (${N} lines)"
        else
            log "   CPAT finished but output not found — check log"
            SKIP_CPAT=true
        fi
    else
        log "   CPAT failed — see ${LOG_FILE}"
        SKIP_CPAT=true
    fi
fi

# ── Step 1b: CPAT hard cases (requires β-LNC predictions) ────────────────────
if ! $SKIP_CPAT && $RUN_CPAT_HARD; then
    if [[ -f "$CPAT_HARD_CSV" ]]; then
        log "  CPAT hard cases already done — skipping"
        log "  Found: ${CPAT_HARD_CSV}"
    else
        log_section "Step 1b: CPAT hard cases (run_cpat_hard_cases.py)"
        log "  β-LNC predictions : ${BLNC_PREDICTIONS}"
        log "  CPAT output dir   : ${CPAT_DIR}"

        CPAT_HARD_CMD="conda run -n beta_lncrna python cpat/run_cpat_hard_cases.py \
    --predictions_csv  ${BLNC_PREDICTIONS} \
    --lnc_fasta        ${LNC_FASTA} \
    --pc_fasta         ${PC_FASTA} \
    --hexamer_table    ${CPAT_HEX} \
    --logit_model      ${CPAT_MODEL} \
    --output_dir       ${CPAT_DIR} \
    --r_module         ${R_MODULE}"

        log "  Running run_cpat_hard_cases.py..."
        if bash -c "$CPAT_HARD_CMD" >> "$LOG_FILE" 2>&1; then
            if [[ -f "$CPAT_HARD_CSV" ]]; then
                N=$(wc -l < "$CPAT_HARD_CSV")
                log "   CPAT hard cases done: ${CPAT_HARD_CSV} (${N} lines)"
            else
                log "   run_cpat_hard_cases.py finished but output not found — check log"
            fi
        else
            log "   run_cpat_hard_cases.py failed — see ${LOG_FILE}"
        fi
    fi
elif ! $SKIP_CPAT && [[ -z "$BLNC_PREDICTIONS" ]]; then
    log "  Skipping CPAT hard cases: --blnc_predictions not provided"
fi

# ── Step 2: CPC2 ──────────────────────────────────────────────────────────────
CPC2_RESULT="${OUTPUT_DIR}/cpc2_results.tsv.txt"

if $SKIP_CPC2; then
    log_section "CPC2: skipped"
elif [[ -f "$CPC2_RESULT" ]]; then
    log_section "CPC2: already done — skipping"
    log "  Found: ${CPC2_RESULT}"
else
    log_section "Step 2: CPC2"
    CPC2_PY="${CPC2_DIR}/bin/CPC2.py"

    if [[ ! -f "$CPC2_PY" ]]; then
        log "   CPC2 not found at ${CPC2_PY} — skipping"
    else
        CPC2_CMD="conda run -n beta_lncrna python ${CPC2_PY} \
    -i ${MERGED_FASTA} \
    -o ${OUTPUT_DIR}/cpc2_results.tsv"

        log "  Running CPC2..."
        if bash -c "$CPC2_CMD" >> "$LOG_FILE" 2>&1; then
            if [[ -f "$CPC2_RESULT" ]]; then
                N=$(wc -l < "$CPC2_RESULT")
                log "   CPC2 done: ${CPC2_RESULT} (${N} lines)"
            else
                log "   CPC2 finished but output not found — check log"
            fi
        else
            log "   CPC2 failed — see ${LOG_FILE}"
        fi
    fi
fi

# ── Step 3: LncDC ─────────────────────────────────────────────────────────────
LNCDC_RESULT="${OUTPUT_DIR}/lncdc_results.csv"

if $SKIP_LNCDC; then
    log_section "LncDC: skipped"
elif [[ -f "$LNCDC_RESULT" ]]; then
    log_section "LncDC: already done — skipping"
    log "  Found: ${LNCDC_RESULT}"
else
    log_section "Step 3: LncDC"
    LNCDC_PY="${LNCDC_DIR}/bin/lncDC.py"

    if [[ ! -f "$LNCDC_PY" ]]; then
        LNCDC_PY=$(conda run -n lncdc which lncDC.py 2>/dev/null || echo "")
    fi

    if [[ -z "$LNCDC_PY" ]] || [[ ! -f "$LNCDC_PY" ]]; then
        log "   LncDC not found — skipping"
        log "    Tried: ${LNCDC_DIR}/bin/lncDC.py"
    else
        LNCDC_CMD="conda run -n lncdc python ${LNCDC_PY} \
    -i ${MERGED_FASTA} \
    -o ${LNCDC_RESULT} \
    -t ${THREADS}"

        log "  Running LncDC (env: lncdc)..."
        if bash -c "$LNCDC_CMD" >> "$LOG_FILE" 2>&1; then
            if [[ -f "$LNCDC_RESULT" ]]; then
                N=$(wc -l < "$LNCDC_RESULT")
                log "   LncDC done: ${LNCDC_RESULT} (${N} lines)"
            else
                log "   LncDC finished but output not found — check log"
            fi
        else
            log "   LncDC failed — see ${LOG_FILE}"
        fi
    fi
fi

# ── Step 4: Orthrus ───────────────────────────────────────────────────────────
ORTHRUS_DIR="${OUTPUT_DIR}"
ORTHRUS_RESULT="${ORTHRUS_DIR}/orthrus/orthrus_test_predictions.csv"

if $SKIP_ORTHRUS; then
    log_section "Orthrus: skipped"
elif [[ -f "$ORTHRUS_RESULT" ]]; then
    log_section "Orthrus: already done — skipping"
    log "  Found: ${ORTHRUS_RESULT}"
else
    log_section "Step 4: Orthrus 4-track (linear probing)"
    mkdir -p "$ORTHRUS_DIR"
    log "  Model       : ${ORTHRUS_MODEL}"
    log "  Env         : ${ORTHRUS_ENV}"
    log "  Train lnc   : ${TRAIN_LNC_FASTA}"
    log "  Train pc    : ${TRAIN_PC_FASTA}"
    log "  Max length  : ${MAX_LENGTH}"

    # Use -p for path-based env (contains /), -n for named env
    if [[ "${ORTHRUS_ENV}" == /* ]] || [[ "${ORTHRUS_ENV}" == ./* ]]; then
        ORTHRUS_ENV_FLAG="-p ${ORTHRUS_ENV}"
    else
        ORTHRUS_ENV_FLAG="-n ${ORTHRUS_ENV}"
    fi

    # Unset LD_LIBRARY_PATH and set MKL threading layer to avoid MKL conflicts
    ORTHRUS_CMD="unset LD_LIBRARY_PATH && \
export MKL_THREADING_LAYER=GNU && \
conda run ${ORTHRUS_ENV_FLAG} python analysis/benchmark/evaluate_orthrus.py \
    --model_name        ${ORTHRUS_MODEL} \
    --train_lnc_fasta   ${TRAIN_LNC_FASTA} \
    --train_pc_fasta    ${TRAIN_PC_FASTA} \
    --test_lnc_fasta    ${LNC_FASTA} \
    --test_pc_fasta     ${PC_FASTA} \
    --output_dir        ${ORTHRUS_DIR} \
    --max_length        ${MAX_LENGTH} \
    --num_workers       0"

    log "  Running evaluate_orthrus.py..."
    if bash -c "$ORTHRUS_CMD" >> "$LOG_FILE" 2>&1; then
        if [[ -f "$ORTHRUS_RESULT" ]]; then
            N=$(wc -l < "$ORTHRUS_RESULT")
            log "   Orthrus done: ${ORTHRUS_RESULT} (${N} lines)"
            # Also report metrics if available
            METRICS_FILE="${ORTHRUS_DIR}/orthrus_test_metrics.json"
            if [[ -f "$METRICS_FILE" ]]; then
                ACC=$(python3 -c "import json; d=json.load(open('${METRICS_FILE}')); print(f\"{d['accuracy']:.4f}\")" 2>/dev/null || echo "?")
                F1=$(python3  -c "import json; d=json.load(open('${METRICS_FILE}')); print(f\"{d['f1']:.4f}\")"       2>/dev/null || echo "?")
                log "  Orthrus accuracy=${ACC}  F1=${F1}"
            fi
        else
            log "   evaluate_orthrus.py finished but output not found — check log"
        fi
    else
        log "   Orthrus failed — see ${LOG_FILE}"
    fi
fi

# ── Step 5: RNAsamba ──────────────────────────────────────────────────────────
# RNAsamba's published conda/pip packages are pinned to TF1/Keras2.1-era
# dependencies (Python <3.8, numpy<=1.16.5, TF<2.0) that conflict with modern
# environments. The official Docker image ships the exact pinned environment
# RNAsamba was built and tested against.
#
#   Repo users with Docker available can run RNAsamba directly without
#   Singularity, e.g.:
#     docker run --rm -u $(id -u) -v "$(pwd):/app" antoniopcamargo/rnasamba \
#         classify /app/<output>.tsv /app/<input_fasta> /app/<weights>.hdf5
#
#   On HPC clusters without Docker daemon access, build a reusable .sif once:
#     singularity pull rnasamba.sif docker://antoniopcamargo/rnasamba
#   and pass it via --rnasamba_sif (default: rnasamba.sif in the cwd).
#   If no .sif is found, this script falls back to a conda env named
#   'beta_lncrna' — only works if that env happens to have a compatible
#   TF1/Keras2.1 stack installed.
RNASAMBA_RESULT="${OUTPUT_DIR}/rnasamba_results.tsv"
RNASAMBA_SIF="${RNASAMBA_SIF:-rnasamba.sif}"

if $SKIP_RNASAMBA; then
    log_section "RNAsamba: skipped"
elif [[ -f "$RNASAMBA_RESULT" ]]; then
    log_section "RNAsamba: already done — skipping"
    log "  Found: ${RNASAMBA_RESULT}"
else
    log_section "Step 5: RNAsamba"
    log "  Weights : ${RNASAMBA_WEIGHTS}"

    if [[ ! -f "$RNASAMBA_WEIGHTS" ]]; then
        log "   RNAsamba weights not found at ${RNASAMBA_WEIGHTS} — skipping"
        log "    Download with:"
        log "    curl -O https://raw.githubusercontent.com/apcamargo/RNAsamba/master/data/full_length_weights.hdf5"
    elif [[ -f "$RNASAMBA_SIF" ]]; then
        # ── Singularity path (preferred — RNAsamba's pinned TF1/Keras2.1 deps
        #    are not installable in modern conda envs) ──────────────────────
        log "  Execution : Singularity container (${RNASAMBA_SIF})"

        ABS_RESULT=$(realpath "$RNASAMBA_RESULT" 2>/dev/null || echo "${PWD}/${RNASAMBA_RESULT}")
        ABS_FASTA=$(realpath "$MERGED_FASTA")
        ABS_WEIGHTS=$(realpath "$RNASAMBA_WEIGHTS")
        ABS_SIF=$(realpath "$RNASAMBA_SIF")

        log "  Container : ${ABS_SIF}"
        log "  Input     : ${ABS_FASTA}"
        log "  Weights   : ${ABS_WEIGHTS}"
        log "  Output    : ${ABS_RESULT}"

        RNASAMBA_CMD="singularity exec --bind $(pwd) ${ABS_SIF} rnasamba classify \
    ${ABS_RESULT} \
    ${ABS_FASTA} \
    ${ABS_WEIGHTS}"

        log "  Running RNAsamba via Singularity..."
        if bash -c "$RNASAMBA_CMD" >> "$LOG_FILE" 2>&1; then
            if [[ -f "$RNASAMBA_RESULT" ]]; then
                N=$(wc -l < "$RNASAMBA_RESULT")
                log "   RNAsamba done (Singularity): ${RNASAMBA_RESULT} (${N} lines)"
            else
                log "   RNAsamba (Singularity) finished but output not found — check log"
            fi
        else
            log "   RNAsamba (Singularity) failed — see ${LOG_FILE}"
        fi
    else
        # ── Conda fallback — only works if rnasamba env has compatible
        #    TF1/Keras2.1 stack installed (see RNAsamba README) ─────────────
        log "  Execution : conda env 'beta_lncrna' (Singularity image not found at ${RNASAMBA_SIF})"
        log "  WARNING: RNAsamba 0.2.5 requires TF<2.0, Keras<2.3, numpy<=1.16.5."
        log "           If this fails, build rnasamba.sif via:"
        log "             singularity pull rnasamba.sif docker://antoniopcamargo/rnasamba"

        RNASAMBA_CMD="conda run -n beta_lncrna rnasamba classify \
    ${RNASAMBA_RESULT} \
    ${MERGED_FASTA} \
    ${RNASAMBA_WEIGHTS}"

        log "  Running RNAsamba via conda..."
        if bash -c "$RNASAMBA_CMD" >> "$LOG_FILE" 2>&1; then
            if [[ -f "$RNASAMBA_RESULT" ]]; then
                N=$(wc -l < "$RNASAMBA_RESULT")
                log "   RNAsamba done (conda): ${RNASAMBA_RESULT} (${N} lines)"
            else
                log "   RNAsamba (conda) finished but output not found — check log"
            fi
        else
            log "   RNAsamba (conda) failed — see ${LOG_FILE}"
        fi
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
log_section "Summary — GENCODE ${RELEASE}"

for label in \
    "CPAT raw:${CPAT_RESULT}" \
    "CPAT hard cases:${CPAT_HARD_CSV}" \
             "CPC2:${CPC2_RESULT}" \
    "LncDC:${LNCDC_RESULT}" \
    "Orthrus:${ORTHRUS_RESULT}" \
    "RNAsamba:${RNASAMBA_RESULT}"; do
    name="${label%%:*}"
    path="${label##*:}"
    if [[ -f "$path" ]]; then
        N=$(wc -l < "$path")
        log "   ${name}: ${path}  (${N} lines)"
    else
        log "   ${name}: NOT FOUND"
    fi
done

log ""
log "Next step — run compare_models.py:"
log ""
log "  python analysis/compare_models.py \\"
log "      --blnc_csv      ${BLNC_PREDICTIONS:-<test_predictions.csv>} \\"
log "      --cpat_csv      ${CPAT_HARD_CSV} \\"
log "      --cpc2_tsv      ${CPC2_RESULT} \\"
log "      --lncdc_tsv     ${LNCDC_RESULT} \\"
log "      --orthrus_csv   ${ORTHRUS_RESULT} \\"
log "      --rnasamba_tsv  ${RNASAMBA_RESULT} \\"
[[ -n "$FEATURE_ONLY_PREDICTIONS" ]] && log "      --feature_only_csv ${FEATURE_ONLY_PREDICTIONS} \\"
log "      --output_dir    ${OUTPUT_DIR}/comparison"
log ""
log "Full log: ${LOG_FILE}"