# GENCODE Preprocessing Pipeline

Complete pipeline to prepare GENCODE data for lncRNA vs protein-coding classification. Proof-tested for v47 and v49.

## Quick Start
```bash
# Run entire pipeline
bash run_all.sh

# Or run steps individually
bash 00_download_gencode.sh
bash 01_extract_biotypes.sh
bash 02_filter_gene_biotypes.sh
bash 03_run_cdhit.sh          # Submit to SLURM or
bash 04_filter_to_cdhit.sh
bash 05_create_visualizations.sh
bash 06_sanity_check.sh
```

## Pipeline Overview
```
GENCODE GTF + FASTA
    ↓ [00] Download
    ↓ [01] Extract biotypes from GTF
    ↓ [02] Filter by gene_biotype (lncRNA, protein_coding)
    ↓ [03] CD-HIT clustering (90% identity)
    ↓ [04] Filter biotype CSV to match CD-HIT output
    ↓ [05] Create visualizations
    ↓ [06] Sanity check
Final dataset ready for training
```

### CD-HIT environment (optional, if you want to reproduce the CDHIT fasta files. Otherwise use the cdhit fasta files from Zenodo)
```bash
conda env create -f cdhit_env.yml
conda activate cdhit_env
```

## Detailed Steps (Example with GENCODE v49)

### Step 0: Download GENCODE v49

Downloads GTF annotation and transcript FASTA files from GENCODE FTP.
```bash
bash 00_download_gencode.sh
```

**Output**:
- `data/raw/gencode.v49.annotation.gtf`
- `data/raw/gencode.v49.lncRNA_transcripts.fa`
- `data/raw/gencode.v49.pc_transcripts.fa`

---

### Step 1: Extract Biotypes from GTF

Parses GTF to create a CSV mapping transcript IDs to biotypes.
```bash
bash 01_extract_biotypes.sh
```

**Output**:
- `data/processed/gencode49_transcript_biotypes.csv` (507,365 transcripts)
- `data/processed/gencode49_transcript_biotypes.summary.txt`
- `data/processed/gencode49_transcript_biotypes.png`

---

### Step 2: Filter by Gene Biotype

Filters to transcripts from lncRNA or protein_coding genes only.
```bash
bash 02_filter_gene_biotypes.sh
```

**Output**:
- `data/processed/gencode49_dataset_biotypes.csv` (483,608 transcripts)
- `data/processed/gencode49_dataset_biotypes.report.txt`

**Removed**: 23,757 transcripts (4.7%) from pseudogenes, small RNAs, IG/TR genes

---

### Step 3: CD-HIT Clustering

Removes redundant sequences at 90% identity threshold.

**Prerequisites**:
```bash
# Create conda environment (one-time setup)
conda env create -f cdhit_env.yml

# Activate environment
conda activate cdhit_env

# Verify CD-HIT is available
cd-hit-est -h
```

**Run clustering**:
```bash
# With default settings (16 CPUs)
bash 03_run_cdhit.sh

# Or specify number of CPUs
CDHIT_CPUS=32 bash 03_run_cdhit.sh
```

**Parameters**:
- Identity threshold: 90% (`-c 0.9`)
- Alignment coverage: 80% of shorter sequence (`-aS 0.8`)
- Word size: 8 (`-n 8`)
- Greedy mode: enabled (`-g 1`) for speed

**Output**:
- `data/cdhit_clusters/lnc_clustered.fa` (~160k sequences)
- `data/cdhit_clusters/g47_pc_clustered.fa` (~145k sequences)
- `data/cdhit_clusters/*.clstr` (cluster information)
- `logs/cdhit_lnc.log` (lncRNA clustering log)
- `logs/cdhit_pc.log` (PC clustering log)

**Note**: PC takes longer due to longer sequences (max ~109kb vs ~50kb for lncRNA)

---

### Step 4: Filter Biotypes to CD-HIT Output

Matches biotype CSV to transcripts retained after CD-HIT.
```bash
bash 04_filter_to_cdhit.sh
```

**Output**:
- `data/processed/gencode49_dataset_biotypes_cdhit90.csv` (305,967 transcripts)
- `data/processed/gencode49_dataset_biotypes_cdhit90.cdhit_report.txt`

---

### Step 5: Create Visualizations

Generates the figures of dataset composition.
```bash
bash 05_create_visualizations.sh
```

**Output**:
- `figures/dataset_distribution.png`
- `figures/version_comparison.png` (if other version data available)

---

### Step 6: Sanity Check

Validates final dataset before training.
```bash
bash 06_sanity_check.sh
```

**Checks**:
- FASTA file integrity
- Length distributions
- Truncation rates
- Biotype consistency
- Stratification group sizes

**Output**:
- `data/processed/sanity_check/`
  - `sanity_check_summary.txt`
  - `length_distribution.png`

---

## Output Files (Example for v49)
```
data/
├── raw/                                    # Downloaded files
│   ├── gencode.v49.annotation.gtf
│   ├── gencode.v49.lncRNA_transcripts.fa
│   └── gencode.v49.pc_transcripts.fa
├── cdhit_clusters/                                  # CD-HIT output
│   ├── lnc_clustered.fa
│   ├── lnc_clustered.fa.clstr
│   ├── pc_clustered.fa
│   └── pc_clustered.fa.clstr
└── processed/                              # Final dataset
    ├── g49_dataset_biotypes_cdhit.csv
    └── sanity_check/

figures/                                    # Visualizations
├── dataset_distribution.png
└── version_comparison.png
```

## Customization

### Different GENCODE Version
```bash
# Edit version number in scripts
GENCODE_VERSION=49  # Change this

# Or pass as environment variable
GENCODE_VERSION=49 bash run_all.sh
```

### Different CD-HIT Threshold
```bash
# Edit 03_run_cdhit.sh
cd-hit-est \
    -c 0.95 \  # Change from 0.9 to 0.95 for 95% identity
    # ... rest of params
```

### Local Execution (no SLURM)
```bash
# Edit 03_run_cdhit.sh - remove SBATCH directives
# Run directly:
bash 03_run_cdhit.sh
```

## Troubleshooting

### CD-HIT not found
```bash
# Ensure conda environment is activated
conda activate cdhit_env
cd-hit-est -h
```

### Memory issues during CD-HIT
```bash
# Increase memory limit in 03_run_cdhit.sh
#SBATCH --mem=128G  # Increase from 64G
# You can also increase cpus per thread
#SBATCH --cpus-per-task=16
```

### Biotype matching failures
```bash
# Check transcript ID format matches between FASTA and CSV
head -1 data/cdhit/lnc_clustered.fa
head -2 data/processed/gencode49_dataset_biotypes.csv
```

## Citation #CHECK

If you use this preprocessing pipeline, please cite:

- GENCODE: Frankish et al. (2021) Nucleic Acids Research
- CD-HIT: Fu et al. (2012) Bioinformatics

## Contact

For questions or issues, please open an issue on GitHub or contact mikael.georges@ibgc.cnrs.fr