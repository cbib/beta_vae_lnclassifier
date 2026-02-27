#!/bin/bash
set -e

# =============================================================================
# Step 6: Generate Summary Report
# =============================================================================
# Aggregates all analysis results into a comprehensive markdown report
#
# Prerequisites:
#   - Steps 1-4 completed
#   - (Optional) Step 5 completed if comparing GENCODE versions
#
# Outputs:
#   - ANALYSIS_SUMMARY.md with all key findings
# =============================================================================

echo "=============================================="
echo "Step 6: Generate Summary Report"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------

EXPERIMENT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment_dir)
            EXPERIMENT_DIR="$2"
            shift 2
            ;;
        *)
            echo " Unknown argument: $1"
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

if [ -z "${EXPERIMENT_DIR}" ]; then
    echo " ERROR: --experiment_dir is required"
    echo ""
    echo "Usage:"
    echo "  bash 06_generate_summary_report.sh --experiment_dir experiments/beta_vae_g47"
    exit 1
fi

if [ ! -d "${EXPERIMENT_DIR}" ]; then
    echo " ERROR: Experiment directory not found: ${EXPERIMENT_DIR}"
    exit 1
fi

# -----------------------------------------------------------------------------
# Check Prerequisites
# -----------------------------------------------------------------------------

echo "Checking for required files from previous steps..."
echo ""

MISSING_FILES=0

# Step 1 outputs
CV_RESULTS="${EXPERIMENT_DIR}/cv_evaluation_results.json"
HARD_CASES="${EXPERIMENT_DIR}/evaluation_csvs/hard_cases.csv"
EMBEDDINGS="${EXPERIMENT_DIR}/embeddings_all_folds.npz"
TEST_RESULTS_DIR="${EXPERIMENT_DIR}/test_results.json"

if [ ! -f "${CV_RESULTS}" ]; then
    echo "  Missing: ${CV_RESULTS} (Step 1)"
    MISSING_FILES=1
fi

if [ ! -f "${HARD_CASES}" ]; then
    echo "  Missing: ${HARD_CASES} (Step 1)"
    MISSING_FILES=1
fi

if [ ! -f "${EMBEDDINGS}" ]; then
    echo "  Missing: ${EMBEDDINGS} (Step 1)"
    MISSING_FILES=1
fi

if [ ! -f "${TEST_RESULTS_DIR}" ]; then
    echo "  Missing: ${TEST_RESULTS_DIR} (Step 1)"
    MISSING_FILES=1
fi

# Step 2 outputs
UMAP_DIR="${EXPERIMENT_DIR}/umap_visualizations"
if [ ! -d "${UMAP_DIR}" ] || [ -z "$(ls -A ${UMAP_DIR})" ]; then
    echo "  Missing or empty: ${UMAP_DIR} (Step 2)"
    MISSING_FILES=1
fi

# Step 3 outputs
SPATIAL_DIR="${EXPERIMENT_DIR}/spatial_analysis"
SPATIAL_STATS="${SPATIAL_DIR}/all_folds_region_statistics.csv"
if [ ! -f "${SPATIAL_STATS}" ]; then
    echo "  Missing: ${SPATIAL_STATS} (Step 3)"
    MISSING_FILES=1
fi

# Step 4 outputs
BIOTYPE_DIR="${EXPERIMENT_DIR}/global_biotype_enrichment"
BIOTYPE_CSV="${BIOTYPE_DIR}/global_biotype_enrichment.csv"
if [ ! -f "${BIOTYPE_CSV}" ]; then
    echo "  Missing: ${BIOTYPE_CSV} (Step 4)"
    MISSING_FILES=1
fi

# Step 5 outputs (optional)
GENCODE_DIR="${EXPERIMENT_DIR}/gencode_novelty_analysis"
GENCODE_AVAILABLE=0
if [ -d "${GENCODE_DIR}" ] && [ -n "$(ls -A ${GENCODE_DIR} 2>/dev/null)" ]; then
    GENCODE_AVAILABLE=1
    echo " GENCODE comparison results found (Step 5)"
fi

if [ ${MISSING_FILES} -eq 1 ]; then
    echo ""
    echo " ERROR: Some required files are missing"
    echo "   Please complete Steps 1-4 before generating the summary report"
    exit 1
fi

echo ""
echo " All required files found"
echo ""

# -----------------------------------------------------------------------------
# Configuration Display
# -----------------------------------------------------------------------------

OUTPUT_FILE="${EXPERIMENT_DIR}/ANALYSIS_SUMMARY.md"

echo "Configuration:"
echo "  Experiment directory: ${EXPERIMENT_DIR}"
echo "  Output file: ${OUTPUT_FILE}"
echo "  GENCODE comparison: $([ ${GENCODE_AVAILABLE} -eq 1 ] && echo 'Yes' || echo 'No')"
echo ""

# -----------------------------------------------------------------------------
# Generate Report
# -----------------------------------------------------------------------------

echo "Generating summary report..."
echo ""

export EXPERIMENT_DIR="${EXPERIMENT_DIR}"
export OUTPUT_FILE="${OUTPUT_FILE}"
export GENCODE_AVAILABLE="${GENCODE_AVAILABLE}"

# Create Python script inline for report generation
python3 << 'PYTHON_SCRIPT'
import json
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Get arguments from environment
experiment_dir = os.environ['EXPERIMENT_DIR']
output_file = os.environ['OUTPUT_FILE']
gencode_available = int(os.environ['GENCODE_AVAILABLE'])

# Read CV evaluation results
cv_results_path = Path(experiment_dir) / "cv_evaluation_results.json"
with open(cv_results_path) as f:
    cv_results = json.load(f)

test_results_path = Path(experiment_dir) / "test_results.json"
if test_results_path.exists():
    with open(test_results_path) as f:
        test_results = json.load(f)

# Read hard cases
hard_cases_path = Path(experiment_dir) / "evaluation_csvs" / "hard_cases.csv"
hard_cases_df = pd.read_csv(hard_cases_path)

# Read spatial statistics
spatial_stats_path = Path(experiment_dir) / "spatial_analysis" / "all_folds_region_statistics.csv"
spatial_df = pd.read_csv(spatial_stats_path)

# Read biotype enrichment
biotype_path = Path(experiment_dir) / "global_biotype_enrichment" / "global_biotype_enrichment.csv"
biotype_df = pd.read_csv(biotype_path)

# Extract experiment name from path
experiment_name = Path(experiment_dir).name

# Start building report
report = []
report.append(f"# Analysis Summary: {experiment_name}")
report.append("")
report.append(f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
report.append("")
report.append("---")
report.append("")

# ============================================================================
# Section 1: Model Performance
# ============================================================================
report.append("## 1A. Model Performance")
report.append("")

# Extract metrics from summary (deterministic results)
summary = cv_results['summary']
accuracy_mean = summary['deterministic_accuracy']['mean']
accuracy_std = summary['deterministic_accuracy']['std']
precision_mean = summary['deterministic_precision']['mean']
precision_std = summary['deterministic_precision']['std']
recall_mean = summary['deterministic_recall']['mean']
recall_std = summary['deterministic_recall']['std']
f1_mean = summary['deterministic_f1']['mean']
f1_std = summary['deterministic_f1']['std']

report.append("### Cross-Validation Metrics (5-fold, Deterministic)")
report.append("")
report.append("| Metric | Mean ± Std |")
report.append("|--------|------------|")
report.append(f"| **Accuracy** | {accuracy_mean:.4f} ± {accuracy_std:.4f} |")
report.append(f"| **Precision** | {precision_mean:.4f} ± {precision_std:.4f} |")
report.append(f"| **Recall** | {recall_mean:.4f} ± {recall_std:.4f} |")
report.append(f"| **F1-Score** | {f1_mean:.4f} ± {f1_std:.4f} |")
report.append("")

report.append("### Model Variants Comparison")
report.append("")
report.append("| Variant | Accuracy |")
report.append("|---------|----------|")
report.append(f"| Deterministic | {accuracy_mean:.4f} ± {accuracy_std:.4f} |")
report.append("")

# Per-fold performance
report.append("### Per-Fold Performance (Deterministic)")
report.append("")
report.append("| Fold | Accuracy | Precision | Recall | F1-Score | Samples |")
report.append("|------|----------|-----------|--------|----------|---------|")
for fold_result in cv_results['fold_results']:
    fold_num = fold_result['fold']
    det = fold_result['deterministic']
    n_samples = fold_result['n_samples']
    report.append(f"| {fold_num} | {det['accuracy']:.4f} | {det['precision']:.4f} | "
                  f"{det['recall']:.4f} | {det['f1']:.4f} | {n_samples:,} |")
report.append("")

# Confusion matrix (aggregated from fold 0 as example)
report.append("### Example Confusion Matrix (Fold 0)")
report.append("")
cm = cv_results['fold_results'][0]['deterministic']['confusion_matrix']
report.append("```")
report.append("                Predicted")
report.append("              lncRNA    PC")
report.append(f"Actual lncRNA  {cm[0][0]:5d}  {cm[0][1]:5d}")
report.append(f"       PC      {cm[1][0]:5d}  {cm[1][1]:5d}")
report.append("```")
report.append("")

# ============================================================================
# Section 1b: Independent Test Set Results
# ============================================================================

test_results_path = Path(experiment_dir) / "test_results.json"

if test_results_path.exists():
    with open(test_results_path) as f:
        test_results = json.load(f)

    report.append("## 1B. Independent Test Set Evaluation")
    report.append("")
    report.append("> These are the **reportable generalisation metrics** — evaluated on the")
    report.append("> held-out test set using an ensemble of all fold models.")
    report.append("")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| **Accuracy** | {test_results['accuracy']:.4f} |")
    report.append(f"| **Precision** | {test_results['precision']:.4f} |")
    report.append(f"| **Recall** | {test_results['recall']:.4f} |")
    report.append(f"| **F1-Score** | {test_results['f1']:.4f} |")
    report.append(f"| **N samples** | {test_results['n_samples']:,} |")
    report.append(f"| **N lncRNA** | {test_results['n_lncrna']:,} |")
    report.append(f"| **N protein-coding** | {test_results['n_pcrna']:,} |")
    report.append(f"| **Folds ensembled** | {test_results['n_folds_ensembled']} |")
    report.append("")

    cm = test_results['confusion_matrix']
    report.append("### Confusion Matrix (Test Set)")
    report.append("")
    report.append("```")
    report.append("                Predicted")
    report.append("              lncRNA    PC")
    report.append(f"Actual lncRNA  {cm[0][0]:5d}  {cm[0][1]:5d}")
    report.append(f"       PC      {cm[1][0]:5d}  {cm[1][1]:5d}")
    report.append("```")
    report.append("")
else:
    report.append("## 2. Independent Test Set Evaluation")
    report.append("")
    report.append("> `test_results.json` not found — holdout evaluation was not run.")
    report.append("> Re-run Step 1 with `--test_lnc_fasta` and `--test_pc_fasta` to generate.")
    report.append("")

# ============================================================================
# Section 2: Hard Cases
# ============================================================================
report.append("## 2. Hard Cases Analysis")
report.append("")

total_samples = len(hard_cases_df)
hard_case_count = hard_cases_df['is_hard_case'].sum()
hard_case_rate = (hard_case_count / total_samples) * 100

report.append(f"- **Total samples**: {total_samples:,}")
report.append(f"- **Hard cases**: {hard_case_count:,} ({hard_case_rate:.2f}%)")
report.append("")

# Hard cases by class
lnc_hard = hard_cases_df[hard_cases_df['true_label'] == 'lnc']['is_hard_case'].sum()
pc_hard = hard_cases_df[hard_cases_df['true_label'] == 'pc']['is_hard_case'].sum()
lnc_total = (hard_cases_df['true_label'] == 'lnc').sum()
pc_total = (hard_cases_df['true_label'] == 'pc').sum()

report.append("### By Class")
report.append("")
report.append("| Class | Hard Cases | Total | Hard Rate |")
report.append("|-------|------------|-------|-----------|")
report.append(f"| lncRNA | {lnc_hard:,} | {lnc_total:,} | {(lnc_hard/lnc_total)*100:.2f}% |")
report.append(f"| Protein-coding | {pc_hard:,} | {pc_total:,} | {(pc_hard/pc_total)*100:.2f}% |")
report.append("")

# ============================================================================
# Section 3: Spatial Clustering
# ============================================================================
report.append("## 3. Spatial Clustering in Embedding Space")
report.append("")

# Get unique regions (exclude -1 for easy cases)
regions = spatial_df[spatial_df['region_id'] != -1]['region_id'].unique()
n_regions = len(regions)

report.append(f"- **Number of spatial regions identified**: {n_regions}")
report.append("")

# Region characteristics
report.append("### Region Characteristics")
report.append("")
report.append("| Region | Sample Count | Dominant Class |")
report.append("|--------|--------------|----------------|")

for region_id in sorted(regions):
    region_data = spatial_df[spatial_df['region_id'] == region_id].iloc[0]
    sample_count = region_data['n_samples']
    
    # Determine dominant class
    if 'pct_lnc' in region_data:
        lnc_frac = region_data['pct_lnc']
        dominant = "lncRNA" if lnc_frac > 0.5 else "Protein-coding"
    else:
        dominant = "N/A"
    
    report.append(f"| Region {region_id} | {sample_count:,} | {dominant} |")

report.append("")

# ============================================================================
# Section 4: Biotype Enrichment
# ============================================================================
report.append("## 4. Biotype Enrichment in Hard Cases")
report.append("")

# Get significant biotypes
sig_biotypes = biotype_df[biotype_df['significant'] == True].sort_values('fold_enrichment', ascending=False)

if len(sig_biotypes) > 0:
    report.append(f"- **Significantly enriched/depleted biotypes**: {len(sig_biotypes)}")
    report.append("")
    
    report.append("### Top Enriched/Depleted Biotypes")
    report.append("")
    report.append("| Biotype | Hard Rate | Fold Enrichment | FDR | Status |")
    report.append("|---------|-----------|-----------------|-----|--------|")
    
    for _, row in sig_biotypes.head(10).iterrows():
        biotype = row['biotype']
        hard_rate = row['hard_rate'] * 100
        fold_enrich = row['fold_enrichment']
        fdr = row['fdr']
        status = "Enriched" if fold_enrich > 1.0 else "Depleted"
        
        report.append(f"| {biotype} | {hard_rate:.2f}% | {fold_enrich:.2f}× | {fdr:.4f} | {status} |")
    
    report.append("")
else:
    report.append("*No significantly enriched or depleted biotypes found.*")
    report.append("")

# ============================================================================
# Section 5: GENCODE Comparison (if available)
# ============================================================================
if gencode_available:
    report.append("## 5. GENCODE Version Comparison")
    report.append("")
    report.append("Novelty analysis results available in:")
    report.append(f"- `{experiment_name}/gencode_novelty_analysis/`")
    report.append("")

# ============================================================================
# Section 6: Generated Figures
# ============================================================================
report.append("## 6. Generated Visualizations")
report.append("")

report.append("### Performance Metrics")
report.append("")
report.append(f"- [ROC/PR Curves](performance_figures/roc_pr_curves.png)")
report.append("")

report.append("### Embedding Space Analysis")
report.append("")
for fold in range(5):
    report.append(f"- Fold {fold}:")
    report.append(f"  - [UMAP by Class](umap_visualizations/fold_{fold}/umap_by_class.png)")
    report.append(f"  - [UMAP with Hard Cases](umap_visualizations/fold_{fold}/umap_hard_cases.png)")
    report.append(f"  - [UMAP by Biotype](umap_visualizations/fold_{fold}/umap_by_biotype.png)")
report.append("")

report.append("### Spatial Clustering")
report.append("")
for fold in range(5):
    report.append(f"- [Fold {fold} Spatial Regions](spatial_analysis/fold_{fold}/spatial_regions_labeled.png)")
report.append("")

report.append("### Biotype Analysis")
report.append("")
report.append(f"- [Global Biotype Enrichment](global_biotype_enrichment/global_biotype_enrichment.png)")
report.append("")

if gencode_available:
    report.append("### GENCODE Novelty")
    report.append("")
    report.append(f"- [Novelty Analysis Figures](gencode_novelty_analysis/)")
    report.append("")

# ============================================================================
# Write report
# ============================================================================
with open(output_file, 'w') as f:
    f.write('\n'.join(report))

print(f" Report generated: {output_file}")

PYTHON_SCRIPT

# -----------------------------------------------------------------------------
# Completion
# -----------------------------------------------------------------------------

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo " Step 6 Complete: Summary Report"
    echo "=============================================="
    echo ""
    echo "Generated report:"
    echo "  ${OUTPUT_FILE}"
    echo ""
    echo " Analysis pipeline complete!"
    echo ""
    echo "Review the summary report for key findings:"
    echo "  cat ${OUTPUT_FILE}"
else
    echo ""
    echo " Step 6 failed"
    exit 1
fi