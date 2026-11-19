#!/bin/bash
# =============================================================================
# run_unseen_benchmark.sh
#
# Seen/unseen benchmark for one release: which test transcripts could a
# GENCODE-trained comparator already have seen? Runs unseen_subset.py, then
# scores the predictions of an existing benchmark run (no model is rerun) on
#   B2  benchmark_unseen_v<C>   transcripts in no release <= C (L2), with the
#                               contrast against the seen subset
#   B3  benchmark_seen_v<C>     transcripts in a release <= C (L2)
#
# The method files, thresholds, common-subset flag, bootstrap size and seed are
# read from <comparison_dir>/benchmark_provenance.json, so B2 and B3 score
# exactly the predictions (same file hashes) and settings of the full-test run.
#
# Usage
# -----
# bash analysis/benchmark/run_unseen_benchmark.sh \
#     --comparison_dir gencode_v49_experiments/benchmark_comparison \
#     --release 49 [--cutoff 46] [--cutoffs "34 43 46"] \
#     [--cache_dir gencode_release_index] [--local_gtf_dir DIR] \
#     [--out_root gencode_v49_experiments] [--dry_run true]
#
# Outputs under <out_root> (default: the folder above --comparison_dir)
#   unseen_subset/                  ID lists, exposure tables, unseen_manifest.json
#   benchmark_unseen_v<C>/          B2 tables, paired and contrast CSVs, provenance
#   benchmark_seen_v<C>/            B3 tables, paired CSV, provenance
#   benchmark_unseen_ids_L2_after_v<C>.log, benchmark_seen_ids_L2_by_v<C>.log
# Set PYTHON to choose the interpreter (default: python).
# =============================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

COMPARISON_DIR=""; RELEASE=""; CUTOFF=46; CUTOFFS="34 43 46"
CACHE_DIR="gencode_release_index"; LOCAL_GTF_DIR=""; OUT_ROOT=""; DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --comparison_dir) COMPARISON_DIR="$2"; shift 2 ;;
        --release)        RELEASE="$2";        shift 2 ;;
        --cutoff)         CUTOFF="$2";         shift 2 ;;
        --cutoffs)        CUTOFFS="$2";        shift 2 ;;
        --cache_dir)      CACHE_DIR="$2";      shift 2 ;;
        --local_gtf_dir)  LOCAL_GTF_DIR="$2";  shift 2 ;;
        --out_root)       OUT_ROOT="$2";       shift 2 ;;
        --dry_run)        DRY_RUN="$2";        shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -n "$COMPARISON_DIR" && -n "$RELEASE" ]] || { echo "ERROR: --comparison_dir and --release are required"; exit 1; }
PROV="${COMPARISON_DIR}/benchmark_provenance.json"
[[ -f "$PROV" ]] || { echo "ERROR: ${PROV} not found (run compute_benchmark_ci.py first)"; exit 1; }
OUT_ROOT="${OUT_ROOT:-$(dirname "${COMPARISON_DIR%/}")}"
case " $CUTOFFS " in *" $CUTOFF "*) ;; *) CUTOFFS="$CUTOFFS $CUTOFF" ;; esac

# Method files and settings of the full-test run, one argument per line
mapfile -t CI_ARGS < <("$PYTHON" - "$PROV" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
flag = {"blnc": "--blnc_csv", "feature_only": "--feature_only_csv", "cpat": "--cpat_csv",
        "cpc2": "--cpc2_tsv", "lncdc": "--lncdc_csv", "rnasamba": "--rnasamba_tsv",
        "orthrus": "--orthrus_csv", "lncrnabert": "--lncrnabert_csv",
        "test_lnc_fasta": "--test_lnc_fasta", "test_pc_fasta": "--test_pc_fasta"}
out = []
for k, v in p["files"].items():
    out += [flag[k], v["path"]]
for m in ("cpat", "cpc2"):
    if m in p["thresholds"]:
        out += [f"--{m}_threshold", str(p["thresholds"][m])]
out += ["--n_bootstrap", str(p["n_bootstrap"]), "--seed", str(p["seed"])]
if p.get("common_subset"):
    out.append("--common_subset")
print("\n".join(out))
PY
)
for i in "${!CI_ARGS[@]}"; do   # test FASTAs for unseen_subset.py
    [[ "${CI_ARGS[$i]}" == "--test_lnc_fasta" ]] && LNC_FASTA="${CI_ARGS[$((i+1))]}"
    [[ "${CI_ARGS[$i]}" == "--test_pc_fasta"  ]] && PC_FASTA="${CI_ARGS[$((i+1))]}"
done

SUBSET_DIR="${OUT_ROOT}/unseen_subset"
UNSEEN_IDS="${SUBSET_DIR}/unseen_ids_L2_after_v${CUTOFF}.txt"
SEEN_IDS="${SUBSET_DIR}/seen_ids_L2_by_v${CUTOFF}.txt"
B2="${OUT_ROOT}/benchmark_unseen_v${CUTOFF}"
B3="${OUT_ROOT}/benchmark_seen_v${CUTOFF}"

SUBSET_CMD=("$PYTHON" "${SCRIPT_DIR}/unseen_subset.py"
    --test_lnc_fasta "$LNC_FASTA" --test_pc_fasta "$PC_FASTA"
    --test_release "$RELEASE" --cutoffs $CUTOFFS
    --cache_dir "$CACHE_DIR" --output_dir "$SUBSET_DIR")
[[ -n "$LOCAL_GTF_DIR" ]] && SUBSET_CMD+=(--local_gtf_dir "$LOCAL_GTF_DIR")
B2_CMD=("$PYTHON" "${SCRIPT_DIR}/compute_benchmark_ci.py" "${CI_ARGS[@]}"
    --id_list "$UNSEEN_IDS" --contrast_id_list "$SEEN_IDS" --output_dir "$B2")
B3_CMD=("$PYTHON" "${SCRIPT_DIR}/compute_benchmark_ci.py" "${CI_ARGS[@]}"
    --id_list "$SEEN_IDS" --output_dir "$B3")

echo "[1/3] unseen_subset.py:        ${SUBSET_CMD[*]}"
echo "[2/3] B2 unseen (+ contrast):  ${B2_CMD[*]}"
echo "[3/3] B3 seen:                 ${B3_CMD[*]}"
[[ "$DRY_RUN" == "true" ]] && { echo "dry run: nothing executed"; exit 0; }

"${SUBSET_CMD[@]}"
[[ -s "$UNSEEN_IDS" && -s "$SEEN_IDS" ]] || { echo "ERROR: ${UNSEEN_IDS} or ${SEEN_IDS} is empty"; exit 1; }
"${B2_CMD[@]}" 2>&1 | tee "${OUT_ROOT}/benchmark_unseen_ids_L2_after_v${CUTOFF}.log"
"${B3_CMD[@]}" 2>&1 | tee "${OUT_ROOT}/benchmark_seen_ids_L2_by_v${CUTOFF}.log"
echo "Done: ${B2}  ${B3}"
