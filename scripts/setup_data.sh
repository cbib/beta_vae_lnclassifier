#!/bin/bash
# Fetch the Zenodo deposit of the paper and check that every file the configs name exists.
#
#   bash scripts/setup_data.sh [--releases "47 49"] [--download-dir DIR] [--check-only]
#
# Run it from the repository root (DATA_ROOT, default: the current directory).
#   1. downloads the three zip files of the deposit into the download folder (default downloads/);
#      a file that is already there with the right MD5 is not downloaded again
#   2. verifies every MD5, then unzips data.zip into data/ and each experiments zip into the
#      repository root (each holds the folder gencode_v<release>_experiments/ at its top level)
#   3. checks, per release, that every file named in data.* of configs/beta_vae_subgroup_base_g<v>.json
#      and configs/beta_vae_feature_only_base_g<v>.json exists; exits 1 and lists what is missing
# --check-only skips steps 1 and 2 and only runs the check (for a data/ folder built another way).
# Needs curl or wget, unzip, python3 and md5sum (or openssl).

set -euo pipefail

# ---- the v3 deposit: replace the placeholders once the DOI exists (MD5 as listed by Zenodo)
ZENODO_RECORD_ID="PLACEHOLDER"
ZENODO_DOI="PLACEHOLDER"
DEPOSIT=(
  "data.zip|PLACEHOLDER"
  "gencode_v47_experiments.zip|PLACEHOLDER"
  "gencode_v49_experiments.zip|PLACEHOLDER"
)
# folders the final pipeline writes under gencode_v<release>_experiments/ (@R = release);
# a missing one is only a note, the pipeline creates it
EXP_SUBDIRS=(beta_vae_subgroup_base_g@R feature_only_g@R frechet_analysis benchmark_tools
             benchmark_comparison benchmark_unseen_v46 benchmark_seen_v46)

RELEASES="47 49"; DOWNLOADS="downloads"; CHECK_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --releases) RELEASES="$2"; shift 2;;
    --download-dir) DOWNLOADS="${2%/}"; shift 2;;
    --check-only) CHECK_ONLY=1; shift;;
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
for r in $RELEASES; do
  [[ "$r" =~ ^[0-9]{2}$ ]] || { echo "bad release '$r' (use 47 or 49)" >&2; exit 2; }
done

cd "${DATA_ROOT:-$(pwd)}"
[[ -d configs ]] || { echo "configs/ not found in $(pwd): run from the repository root or set DATA_ROOT" >&2; exit 2; }

md5_of() {
  if command -v md5sum >/dev/null 2>&1; then md5sum "$1" | cut -d' ' -f1
  else openssl md5 -r "$1" | cut -d' ' -f1; fi
}

fetch() {  # fetch <file name> <target path>
  local url="https://zenodo.org/records/${ZENODO_RECORD_ID}/files/$1?download=1"
  if command -v curl >/dev/null 2>&1; then curl -fL --retry 3 -C - -o "$2" "$url"
  elif command -v wget >/dev/null 2>&1; then wget -c -O "$2" "$url"
  else echo "neither curl nor wget found" >&2; return 1; fi
}

if (( ! CHECK_ONLY )); then
  [[ "$ZENODO_RECORD_ID" != "PLACEHOLDER" ]] || { echo "ZENODO_RECORD_ID is not set: the v3 deposit does not exist yet (edit the top of this script)" >&2; exit 2; }
  for entry in "${DEPOSIT[@]}"; do
    [[ "${entry#*|}" != "PLACEHOLDER" ]] || { echo "MD5 of ${entry%%|*} is not set (edit the top of this script)" >&2; exit 2; }
  done
  command -v unzip >/dev/null 2>&1 || { echo "unzip not found" >&2; exit 1; }
  mkdir -p "$DOWNLOADS" data

  echo "Zenodo record ${ZENODO_RECORD_ID} (DOI ${ZENODO_DOI})"
  for entry in "${DEPOSIT[@]}"; do
    name="${entry%%|*}"; md5="${entry#*|}"; file="$DOWNLOADS/$name"
    if [[ -f "$file" && "$(md5_of "$file")" == "$md5" ]]; then
      echo "have $name"
    else
      echo "downloading $name"
      fetch "$name" "$file" || true          # a failed or interrupted transfer is caught by the MD5 below
      [[ -f "$file" && "$(md5_of "$file")" == "$md5" ]] || { echo "MD5 mismatch for $file: delete it and rerun" >&2; exit 1; }
    fi
  done
  for entry in "${DEPOSIT[@]}"; do
    name="${entry%%|*}"
    [[ "$name" == data.zip ]] && dest=data || dest=.
    echo "unzipping $name -> $dest/"
    unzip -q -o "$DOWNLOADS/$name" -d "$dest"
  done
fi

STATUS=0
for r in $RELEASES; do
  echo "== GENCODE ${r}"
  python3 - "$r" <<'PY' || STATUS=1
import json, os, sys

rel = sys.argv[1]
missing, n = {}, 0
for kind in ("subgroup", "feature_only"):
    cfg = f"configs/beta_vae_{kind}_base_g{rel}.json"
    if not os.path.isfile(cfg):
        missing[cfg] = "config file"
        continue
    for key, val in json.load(open(cfg))["data"].items():
        if isinstance(val, str) and "/" in val:
            n += 1
            path = os.path.expandvars(val)
            if not os.path.exists(path):
                missing.setdefault(path, f"{os.path.basename(cfg)}: data.{key}")
if missing:
    print(f"  FAILED: {len(missing)} of {n} configured paths missing:")
    print("\n".join(f"    {p}   ({w})" for p, w in sorted(missing.items())))
    sys.exit(1)
print(f"  ok: all {n} configured data paths exist")
PY
  for sub in "${EXP_SUBDIRS[@]}"; do
    d="gencode_v${r}_experiments/${sub//@R/$r}"
    [[ -d "$d" ]] || echo "  note: $d does not exist yet"
  done
done

if (( STATUS )); then echo "data check FAILED" >&2; exit 1; fi
echo "data check passed for: $RELEASES"