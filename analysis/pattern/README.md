# Interpretability Pipeline — BetaVAESubgroup

Three-layer interpretability analysis for the BetaVAESubgroup architecture,
proof-tested on GENCODE v47/v49 with the 20-token registry (NonB / REP /
NonB2 blocks).

---

## Pipeline overview

```
Training (main_subgroup.py / main_subgroup_debug.py)
    ↓
[1] extract_representations.py
    • Load trained fold checkpoints
    • Forward pass with hooks on fc_mu (z) and token_norm (feature tokens)
    • Save fold_N_repr.npz: z, tokens, labels, lengths, gc_content, token_names, block_names
    ↓
[2] analyze_mstar.py                          (associational layer, pre-training)
    • M* (scalar SNR) and Fréchet distance decomposition per subgroup
    • Mean term vs covariance term — reveals distributional suppressor candidates
    ↓
[3] analyze_latent_probing.py                 (model-functional layer)
    Stage 1 — probe z for length / GC / class confounds
    Stage 2 — sequence-feature redundancy: R² from z to each token's variance
    Stage 3 — pattern-weight alignment (Haufe et al. 2014), raw and z-residual
    ↓
[4] analyze_subgroup_ablation.py               (model-functional layer)
    • Token-zero and feature-zero ablation, per subgroup, per fold
    ↓
[5] analyze_residual_ablation.py               (model-functional layer)
    • Feature-zero ablation on z-orthogonalized tokens
    • Isolates sequence-independent ablation importance
    ↓
[6] patch_tokens.py                            (causal layer)
    • In-distribution token activation patching
    • Matched lncRNA/mRNA pairs, both patch directions
    • Symmetry score: causal reliance, independent of ablation's OOD intervention problem
    ↓
[7] plot_interpretability_scatter.py           (synthesis)
    • Fréchet (x) × ablation (y) × patching symmetry (bubble size)
    • All three layers in one figure
    ↓
plot_patching_symmetry.py                      (standalone patching summary chart)
```

Steps 1–6 write CSV + NPZ outputs; step 7 (and the standalone patching plot)
read those outputs and produce the summary figures.

---

## Usage

### Step 1 — Extract representations

```bash
python analysis/pattern/extract_representations.py \
    --experiment_dir gencode_v49_experiments/beta_vae_subgroup_base_g49_new \
    --config         configs/beta_vae_subgroup_base_g49.json \
    --output_dir     gencode_v49_experiments/beta_vae_subgroup_base_g49_new/representations \
    --device         cuda:0
```

### Step 2 — Fréchet / M* decomposition (associational layer)

```bash
python analysis/pattern/analyze_mstar.py \
    --features_dir data/processed_features_genomic \
    --release      v49 \
    --output_dir   gencode_v49_experiments/mstar_analysis
```

### Step 3 — Latent probing + pattern analysis (model-functional layer)

```bash
python analysis/pattern/analyze_latent_probing.py \
    --repr_dir        gencode_v49_experiments/beta_vae_subgroup_base_g49_new/representations \
    --output_dir      gencode_v49_experiments/beta_vae_subgroup_base_g49_new/latent_probing \
    --model_label     "β-LNC" \
    --gencode_version v49
```

With more than one fold present, this also writes cross-fold summaries
(`cross_fold_redundancy.csv`, `cross_fold_pattern_raw.csv`,
`cross_fold_pattern_residual.csv`, `key_findings.txt`).

### Step 4 — Subgroup ablation

```bash
python analysis/post_training_pipeline/scripts/analyze_subgroup_ablation.py \
    --experiment_dir  gencode_v49_experiments/beta_vae_subgroup_base_g49_new \
    --config          configs/beta_vae_subgroup_base_g49.json \
    --output_dir      gencode_v49_experiments/beta_vae_subgroup_base_g49_new/ablation \
    --gencode_version v49 \
    --device          cuda:0
```

### Step 5 — Residual ablation

```bash
python analysis/pattern/analyze_residual_ablation.py \
    --experiment_dir  gencode_v49_experiments/beta_vae_subgroup_base_g49_new \
    --repr_dir        gencode_v49_experiments/beta_vae_subgroup_base_g49_new/representations \
    --config          configs/beta_vae_subgroup_base_g49.json \
    --output_dir      gencode_v49_experiments/beta_vae_subgroup_base_g49_new/residual_ablation \
    --gencode_version v49 \
    --device          cuda:0
```

### Step 6 — Token activation patching (causal layer)

```bash
python analysis/pattern/patch_tokens.py \
    --experiment_dir  gencode_v49_experiments/beta_vae_subgroup_base_g49_new \
    --config          configs/beta_vae_subgroup_base_g49.json \
    --output_dir      gencode_v49_experiments/beta_vae_subgroup_base_g49_new/patching \
    --device          cuda:0 \
    --n_pairs         200 \
    --min_confidence  0.7 \
    --max_length_diff 0.2 \
    --max_gc_diff     0.05
```

Cross-fold by default (one run per `fold_N_best.pt` checkpoint). Outputs
`all_folds_patching.csv` and `patching_summary.csv` (mean ± std symmetry
score per subgroup).

### Step 7 — Synthesis figures

```bash
python analysis/benchmark/plot_interpretability_scatter.py \
    --frechet_csv          gencode_v49_experiments/mstar_analysis/frechet_results.csv \
    --ablation_csv          gencode_v49_experiments/beta_vae_subgroup_base_g49_new/ablation/analysis/all_folds_ablation.csv \
    --pattern_raw_csv      gencode_v49_experiments/beta_vae_subgroup_base_g49_new/latent_probing/cross_fold_pattern_raw.csv \
    --pattern_residual_csv gencode_v49_experiments/beta_vae_subgroup_base_g49_new/latent_probing/cross_fold_pattern_residual.csv \
    --patching_csv         gencode_v49_experiments/beta_vae_subgroup_base_g49_new/patching/patching_summary.csv \
    --output_dir           gencode_v49_experiments/benchmark_comparison \
    --release              v49 \
    --display_names
```

```bash
python analysis/pattern/plot_patching_symmetry.py \
    --patching_csv gencode_v49_experiments/beta_vae_subgroup_base_g49_new/patching/patching_summary.csv \
    --output_dir   gencode_v49_experiments/benchmark_comparison \
    --release      v49 \
    --display_names
```

---

## Key formulas

**Residual (z-orthogonalized) token:**

$$T_{\text{resid}} = T_{sg} - \widehat{T}_{sg}(z), \quad \widehat{T}_{sg} = \text{Ridge}(z \rightarrow T_{sg})$$

**Pattern-weight alignment** (Haufe et al. 2014) — filters suppressor
variance out of a linear probe's weight vector $W$, using the input
covariance $\Sigma$:

$$A = \Sigma W, \qquad \text{alignment} = \cos(A, W)$$

Alignment near 1 indicates genuine class signal in the probe direction;
near 0 indicates the raw weight $W$ was largely suppressor-driven. This is
an approximately SAP-compliant attribution for linear probes, under the
assumption that token representations and class are approximately linearly
related.

**Fréchet distance decomposition:**

$$FD^2 = \underbrace{\|\mu_1-\mu_2\|_2^2}_{\text{mean term}} \; + \; \underbrace{\text{Tr}\!\left(\Sigma_1+\Sigma_2-2(\Sigma_1\Sigma_2)^{1/2}\right)}_{\text{covariance term}}$$

Unlike scalar SNR/M*, this captures class-conditional covariance
differences — the source of signal for distributional suppressor
candidates such as REP_CORE, which score near zero on scalar metrics but
rank highly by total FD.

**Patching symmetry score** — mean absolute logit shift across both patch
directions, for subgroup $s$:

$$\Delta_{\text{lnc}\to\text{mRNA}} = \text{logit}_{\text{patched}} - \text{logit}_{\text{baseline}}$$

$$\text{symmetry}_s = \tfrac{1}{2}\left(|\Delta_{\text{lnc}\to\text{mRNA}}| + |\Delta_{\text{mRNA}\to\text{lnc}}|\right)$$

A high, consistent symmetry score indicates the model causally relies on
subgroup $s$; a near-zero score indicates the model does not, regardless of
how much independent signal that subgroup carries by other measures.

---

## Naming note

Internal code, CSV columns, and checkpoint keys use `te`/`TE_*` for the
repeat-element block for historical reasons. Display labels throughout
plots and documents use `REP`/`REP_*` (the scientifically accurate name).
Pass `--display_names` to the plotting scripts to apply this remapping in
figure labels only — do not rename the internal keys.

---

## Output reference

```
representations/
  fold_N_repr.npz            z, tokens, labels, token_names, block_names, ...

mstar_analysis/
  frechet_results.csv        subgroup, block, frechet_dist, mean_term, cov_term, cov_fraction

latent_probing/
  fold_N/
    latent_probing.csv / .png
    redundancy.csv / .png
    pattern_raw.csv / .png
    pattern_residual.csv / .png
    pattern_comparison.png
  cross_fold_latent_probing.csv
  cross_fold_redundancy.csv
  cross_fold_pattern_raw.csv
  cross_fold_pattern_residual.csv
  cross_fold_pattern_comparison.png
  key_findings.txt

ablation/analysis/
  all_folds_ablation.csv     subgroup, block, mode, acc_drop, fold, ...

residual_ablation/
  all_folds_residual_ablation.csv

patching/
  fold_N_patching.csv
  all_folds_patching.csv
  patching_summary.csv       subgroup, block, direction, mean_delta_logit,
                             symmetry_score, causal_effect, ...

benchmark_comparison/
  interpretability_scatter_{release}.png / .pdf
  patching_symmetry_bar_{release}.png / .pdf
```