#!/bin/bash

# setup_data.sh
# Downloads and extracts dataset from Zenodo for lncRNA classification project
# Usage: ./setup_data.sh

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Zenodo record ID and DOI
ZENODO_RECORD_ID="18849718"  # e.g., "1234567"
ZENODO_DOI="10.5281/zenodo.18849718"

# File information
DATA_ZIP="data.zip"
V47_ZIP="gencode_v47_experiments.zip"
V49_ZIP="gencode_v49_experiments.zip"

# Zenodo download URLs (constructed from record ID)
DATA_URL="https://zenodo.org/record/${ZENODO_RECORD_ID}/files/${DATA_ZIP}"
V47_URL="https://zenodo.org/record/${ZENODO_RECORD_ID}/files/${V47_ZIP}"
V49_URL="https://zenodo.org/record/${ZENODO_RECORD_ID}/files/${V49_ZIP}"

# Target directories
PROJECT_ROOT="$(pwd)"
DATA_DIR="${PROJECT_ROOT}/data"
EXPERIMENTS_DIR="${PROJECT_ROOT}/experiments"

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}lncRNA Classification Dataset Setup${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Zenodo DOI: ${ZENODO_DOI}"
echo "Project root: ${PROJECT_ROOT}"
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to download file with progress
download_file() {
    local url=$1
    local output=$2
    
    if command_exists wget; then
        echo -e "${GREEN}Downloading with wget...${NC}"
        wget --no-check-certificate -O "$output" "$url"
    elif command_exists curl; then
        echo -e "${GREEN}Downloading with curl...${NC}"
        curl -L -o "$output" "$url"
    else
        echo -e "${RED}Error: Neither wget nor curl found.${NC}"
        echo -e "${YELLOW}Please install wget or curl, or download manually.${NC}"
        return 1
    fi
}

# Check for unzip
if ! command_exists unzip; then
    echo -e "${RED}Error: unzip not found.${NC}"
    echo -e "${YELLOW}Please install unzip: sudo apt install unzip${NC}"
    exit 1
fi

# Create directories
echo -e "${GREEN}Creating directory structure...${NC}"
mkdir -p "${DATA_DIR}"
mkdir -p "${EXPERIMENTS_DIR}"
mkdir -p "${PROJECT_ROOT}/downloads"

cd "${PROJECT_ROOT}/downloads"

# Download data.zip
echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}[1/3] Downloading data.zip${NC}"
echo -e "${GREEN}================================================${NC}"
if [ -f "${DATA_ZIP}" ]; then
    echo -e "${YELLOW}${DATA_ZIP} already exists. Skipping download.${NC}"
else
    if [ "${ZENODO_RECORD_ID}" = "[RECORD_ID]" ]; then
        echo -e "${YELLOW}Zenodo record ID not set.${NC}"
        echo -e "${YELLOW}Please download manually from Zenodo:${NC}"
        echo -e "${YELLOW}https://zenodo.org/record/YOUR_RECORD_ID/files/${DATA_ZIP}${NC}"
        echo -e "${YELLOW}And place it in: ${PROJECT_ROOT}/downloads/${NC}"
        read -p "Press Enter once you've downloaded ${DATA_ZIP}..."
    else
        download_file "$DATA_URL" "$DATA_ZIP" || {
            echo -e "${RED}Download failed.${NC}"
            echo -e "${YELLOW}Please download manually:${NC}"
            echo -e "${YELLOW}${DATA_URL}${NC}"
            echo -e "${YELLOW}And place it in: ${PROJECT_ROOT}/downloads/${NC}"
            read -p "Press Enter once you've downloaded ${DATA_ZIP}..."
        }
    fi
fi

# Download gencode_v47_experiments.zip
echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}[2/3] Downloading gencode_v47_experiments.zip${NC}"
echo -e "${GREEN}================================================${NC}"
if [ -f "${V47_ZIP}" ]; then
    echo -e "${YELLOW}${V47_ZIP} already exists. Skipping download.${NC}"
else
    if [ "${ZENODO_RECORD_ID}" = "[RECORD_ID]" ]; then
        echo -e "${YELLOW}Zenodo record ID not set.${NC}"
        echo -e "${YELLOW}Please download manually from Zenodo and place in: ${PROJECT_ROOT}/downloads/${NC}"
        read -p "Press Enter once you've downloaded ${V47_ZIP}..."
    else
        download_file "$V47_URL" "$V47_ZIP" || {
            echo -e "${RED}Download failed.${NC}"
            echo -e "${YELLOW}Please download manually and place in: ${PROJECT_ROOT}/downloads/${NC}"
            read -p "Press Enter once you've downloaded ${V47_ZIP}..."
        }
    fi
fi

# Download gencode_v49_experiments.zip
echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}[3/3] Downloading gencode_v49_experiments.zip${NC}"
echo -e "${GREEN}================================================${NC}"
if [ -f "${V49_ZIP}" ]; then
    echo -e "${YELLOW}${V49_ZIP} already exists. Skipping download.${NC}"
else
    if [ "${ZENODO_RECORD_ID}" = "[RECORD_ID]" ]; then
        echo -e "${YELLOW}Zenodo record ID not set.${NC}"
        echo -e "${YELLOW}Please download manually from Zenodo and place in: ${PROJECT_ROOT}/downloads/${NC}"
        read -p "Press Enter once you've downloaded ${V49_ZIP}..."
    else
        download_file "$V49_URL" "$V49_ZIP" || {
            echo -e "${RED}Download failed.${NC}"
            echo -e "${YELLOW}Please download manually and place in: ${PROJECT_ROOT}/downloads/${NC}"
            read -p "Press Enter once you've downloaded ${V49_ZIP}..."
        }
    fi
fi

# Verify all files exist
echo ""
echo -e "${GREEN}Verifying downloaded files...${NC}"
if [ ! -f "${DATA_ZIP}" ] || [ ! -f "${V47_ZIP}" ] || [ ! -f "${V49_ZIP}" ]; then
    echo -e "${RED}Error: Not all required files are present.${NC}"
    echo "Expected files in ${PROJECT_ROOT}/downloads/:"
    echo "  - ${DATA_ZIP}"
    echo "  - ${V47_ZIP}"
    echo "  - ${V49_ZIP}"
    exit 1
fi
echo -e "${GREEN}✓ All files present${NC}"

# Extract data.zip
echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Extracting data.zip to ${DATA_DIR}/${NC}"
echo -e "${GREEN}================================================${NC}"
unzip -q -o "${DATA_ZIP}" -d "${DATA_DIR}"
echo -e "${GREEN}✓ data.zip extracted${NC}"

# Extract gencode_v47_experiments.zip
echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Extracting gencode_v47_experiments.zip${NC}"
echo -e "${GREEN}================================================${NC}"
unzip -q -o "${V47_ZIP}" -d "${EXPERIMENTS_DIR}"
echo -e "${GREEN}✓ gencode_v47_experiments.zip extracted${NC}"

# Extract gencode_v49_experiments.zip
echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Extracting gencode_v49_experiments.zip${NC}"
echo -e "${GREEN}================================================${NC}"
unzip -q -o "${V49_ZIP}" -d "${EXPERIMENTS_DIR}"
echo -e "${GREEN}✓ gencode_v49_experiments.zip extracted${NC}"

# Verify directory structure
echo ""
echo -e "${GREEN}Verifying directory structure...${NC}"

# Check data subdirectories
DATA_SUBDIRS=("cdhit_clusters" "dataset_biotypes" "lncRNABERT_results" "processed_features" "split_gencode_47" "split_gencode_49")
for subdir in "${DATA_SUBDIRS[@]}"; do
    if [ -d "${DATA_DIR}/${subdir}" ]; then
        echo -e "${GREEN}✓${NC} data/${subdir}/"
    else
        echo -e "${RED}✗${NC} data/${subdir}/ not found"
    fi
done

# Check experiment directories
EXP_SUBDIRS=("beta_vae_contrastive_g47" "beta_vae_features_attn_g47" "beta_vae_features_g47" "cnn_g47" "stat_results")
for subdir in "${EXP_SUBDIRS[@]}"; do
    if [ -d "${EXPERIMENTS_DIR}/${subdir}" ] || [ -d "${EXPERIMENTS_DIR}/gencode_v47_experiments/${subdir}" ]; then
        echo -e "${GREEN}✓${NC} experiments/${subdir}/"
    else
        echo -e "${YELLOW}?${NC} experiments/${subdir}/ (check manually)"
    fi
done

# Print summary
echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Directory structure:"
echo "  ${PROJECT_ROOT}/"
echo "  ├── data/"
echo "  │   ├── cdhit_clusters/"
echo "  │   ├── dataset_biotypes/"
echo "  │   ├── lncRNABERT_results/"
echo "  │   ├── processed_features/"
echo "  │   ├── split_gencode_47/"
echo "  │   └── split_gencode_49/"
echo "  ├── experiments/"
echo "  │   ├── gencode_v47_experiments/"
echo "  │   └── gencode_v49_experiments/"
echo "  └── downloads/ (can be deleted)"
echo ""
echo -e "${YELLOW}Note: You can safely delete the downloads/ directory to save space${NC}"
echo ""
echo -e "${GREEN}Data is ready for use!${NC}"