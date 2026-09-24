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
    • Optional: --pattern_raw_csv also produces frechet_vs_pattern.png, flagging
      subgroups the associational layer catches that naive pattern alignment misses
    ↓
[3] analyze_latent_probing.py                 (model-functional layer)
    Stage 1 — probe z for length / GC / class confounds
    Stage 2 — sequence-feature redundancy: R² from z to each token's variance,
              with an optional permutation-test significance check (BH-FDR)
    Stage 3 — pattern-weight alignment (Haufe et al. 2014), raw and z-residual,
              with a paired t-test (raw vs residual, BH-FDR) per subgroup
    ↓
[4] analyze_subgroup_ablation.py               (model-functional layer)
    • Token-zero and feature-zero ablation, per subgroup, per fold
    • One-sample t-test (acc_drop vs 0, BH-FDR) per subgroup
    ↓
[5] analyze_residual_ablation.py               (model-functional layer)
    • Feature-zero ablation on z-orthogonalized tokens
    • Isolates sequence-independent ablation importance
    • Paired t-test (feature_zero vs residual_zero, BH-FDR) per subgroup
    ↓
[6] patch_tokens.py                            (causal layer)
    • In-distribution token activation patching, matched lncRNA/mRNA pairs
    • Scopes: single subgroup / block / all tokens / shuffled-pair null control
    • Symmetry score: continuous effect size, both patch directions
    • IIA (Interchange Intervention Accuracy): stricter, discrete — did
      patching actually flip the predicted class, not just shift the logit
    ↓
[7] joint_patching_search.py                   (causal layer, follow-up)
    • Greedy and top-k multi-token search: does a COMBINATION of subgroups
      patch jointly better than the fixed single/block/all scopes predict
      from an independence baseline? Reveals synergy the fixed scopes can't.
    ↓
[8] plot_interpretability_main.py               (synthesis, main-text figure)
    • Three-panel consolidated figure: (A) the three-layer scatter
      (Fréchet × ablation × symmetry, replaces the retired standalone
      plot_interpretability_scatter.py), (B) single-subgroup symmetry vs
      IIA, (C) cross-fold greedy-search synergy curve. Designed for the
      paper's main text under a page-limit constraint — carries the full
      Layer 1/2/3 argument in one figure.
    ↓
plot_patching_symmetry.py                      (supplementary patching figures)
    • Symmetry-score bar chart, symmetry-vs-IIA comparison, sufficiency
      curve, optional cross-release IIA comparison — each also usable
      standalone for slides/appendix figures outside the main-text panel.
```

Steps 1–7 write CSV + NPZ outputs; step 8 and the standalone patching plot
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
    --output_dir   gencode_v49_experiments/mstar_analysis \
    --pattern_raw_csv gencode_v49_experiments/beta_vae_subgroup_base_g49_new/latent_probing/cross_fold_pattern_raw.csv
```

`--pattern_raw_csv` is optional — omit it to skip `frechet_vs_pattern.png`
(requires step 3 to have run first, since it reads that step's output).

### Step 3 — Latent probing + pattern analysis (model-functional layer)

```bash
python analysis/pattern/analyze_latent_probing.py \
    --repr_dir        gencode_v49_experiments/beta_vae_subgroup_base_g49_new/representations \
    --output_dir      gencode_v49_experiments/beta_vae_subgroup_base_g49_new/latent_probing \
    --model_label     "β-LNC" \
    --gencode_version v49 \
    --n_permutations  1000
```

`--n_permutations` (default 1000, set 0 to skip) controls the Stage 2
redundancy significance test — each permutation refits a Ridge regression
per subgroup per fold, so this is the slow part of the step; reduce it for
a quick pass.

With more than one fold present, this also writes cross-fold summaries
(`cross_fold_redundancy.csv`, `cross_fold_pattern_raw.csv`,
`cross_fold_pattern_residual.csv`, `cross_fold_pattern_comparison_stats.csv`,
`key_findings.txt`).

### Step 4 — Subgroup ablation

```bash
python analysis/pattern/analyze_subgroup_ablation.py \
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
    --max_gc_diff     0.05 \
    --patch_modes     single,block,all
```

Cross-fold by default (one run per `fold_N_best.pt` checkpoint). Outputs
`all_folds_patching.csv` and `patching_summary.csv` (mean ± std symmetry
score and IIA per subgroup, per scope).

`--patch_modes` accepts any comma-separated subset of `single,block,all,shuffled`:
- `single,block,all` (default) — the sufficiency curve's three fixed scopes
- add `shuffled` for the null control — same single-token scope, but the
  patch source is drawn from a random same-class transcript instead of the
  length/GC-matched partner, testing whether IIA reflects genuine
  pair-specific structure or just class-conditional signal

### Step 7 — Joint multi-token search (causal layer, follow-up)

```bash
python analysis/pattern/joint_patching_search.py \
    --experiment_dir  gencode_v49_experiments/beta_vae_subgroup_base_g49_new \
    --config          configs/beta_vae_subgroup_base_g49.json \
    --fold            all \
    --single_summary  gencode_v49_experiments/beta_vae_subgroup_base_g49_new/patching/all_folds_patching.csv \
    --output_dir      gencode_v49_experiments/beta_vae_subgroup_base_g49_new/joint_patching \
    --device          cuda:0 \
    --n_pairs         1000 \
    --mode            both \
    --max_set_size    6 \
    --topk            5
```

Requires step 6's output as `--single_summary` (accepts either the raw
per-pair CSV or the aggregated summary — auto-detected). Runs independently
per fold (`--fold all` for cross-fold, or a specific fold index). Outputs a
searched sufficiency curve (`joint_patching_greedy_*.csv`) directly
comparable to step 6's fixed single/block/all curve, plus an
independence-null baseline per step — a searched-set IIA well above the
null at a given set size indicates synergy the fixed scopes cannot detect,
since block membership constrains which subgroups can be patched together.

### Step 8 — Synthesis figures

```bash
python analysis/pattern/plot_interpretability_main.py \
    --frechet_csv          gencode_v49_experiments/mstar_analysis/frechet_ranking.csv \
    --ablation_csv         gencode_v49_experiments/beta_vae_subgroup_base_g49_new/ablation/analysis/all_folds_ablation.csv \
    --pattern_raw_csv      gencode_v49_experiments/beta_vae_subgroup_base_g49_new/latent_probing/cross_fold_pattern_raw.csv \
    --pattern_residual_csv gencode_v49_experiments/beta_vae_subgroup_base_g49_new/latent_probing/cross_fold_pattern_residual.csv \
    --patching_csv         gencode_v49_experiments/beta_vae_subgroup_base_g49_new/patching/all_folds_patching.csv \
    --greedy_all_folds     gencode_v49_experiments/beta_vae_subgroup_base_g49_new/joint_patching/joint_patching_greedy_all_folds.csv \
    --output_dir           gencode_v49_experiments/benchmark_comparison \
    --release              v49
```

All CSV inputs must share the same subgroup key
convention (internal `TE_*` keys) — the figure always renders `TE_*` as
`REP_*` in labels, unconditionally (no `--display_names` flag here, unlike
the other plotting scripts).

For standalone supplementary/appendix figures instead of the consolidated
panel:

```bash
python analysis/pattern/plot_patching_symmetry.py \
    --patching_csv gencode_v49_experiments/beta_vae_subgroup_base_g49_new/patching/patching_summary.csv \
    --output_dir   gencode_v49_experiments/benchmark_comparison \
    --release      v49 \
    --display_names \
    --top_n_single 6
```

Produces, from one call: the symmetry-score bar chart, the symmetry-vs-IIA
comparison, and (if the input CSV has block/all scopes) the sufficiency
curve. For a cross-release comparison, add a second release's summary:

```bash
python analysis/pattern/plot_patching_symmetry.py \
    --patching_csv  gencode_v49_experiments/beta_vae_subgroup_base_g49_new/patching/patching_summary.csv \
    --patching_csv2 gencode_v47_experiments/beta_vae_subgroup_base_g47_new/patching/patching_summary.csv \
    --release       v49 --release2 v47 \
    --output_dir    gencode_v49_experiments/benchmark_comparison \
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

A high, consistent symmetry score indicates the model's output is
sensitive to subgroup $s$; a near-zero score indicates it is not, regardless
of how much independent signal that subgroup carries by other measures.

**Interchange Intervention Accuracy (IIA)** — stricter than symmetry score:
the fraction of valid trials (baseline correctly classified on the source
side) where patching actually **flipped** the predicted class, not merely
shifted the logit toward it:

$$\text{IIA}_s = \frac{\#\{\text{trials where patching flipped the prediction}\}}{\#\{\text{trials with a valid baseline}\}}$$

Reported per direction and as an overall mean (`iia_overall`). A subgroup
can show a nonzero symmetry score (partial shift) with near-zero IIA
(rarely changes the actual decision) — the two metrics answer different
questions: "does this move the needle" vs. "does this change the verdict."
`symmetry_vs_iia_{release}.png` makes this divergence visible directly.

**Joint-search independence null** — for a searched token set $S$, the
independence-null IIA is:

$$\text{IIA}_{\text{additive}}(S) = 1 - \prod_{i \in S} \left(1 - \text{IIA}(\{i\})\right)$$

i.e. the probability that at least one subgroup's single-token IIA would
have flipped the prediction, if each contributed independently. Searched
IIA well above this null indicates synergy; searched IIA close to the null
is consistent with independent, non-synergistic contributions.

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
  frechet_vs_pattern.png     (only if --pattern_raw_csv given)

latent_probing/
  fold_N/
    latent_probing.csv / .png
    redundancy.csv / .png
    pattern_raw.csv / .png
    pattern_residual.csv / .png
    pattern_comparison.png
  cross_fold_latent_probing.csv
  cross_fold_redundancy.csv                  includes p_value_fisher_combined, q_value_bh, significant_fdr
  cross_fold_pattern_raw.csv
  cross_fold_pattern_residual.csv
  cross_fold_pattern_comparison.png
  cross_fold_pattern_comparison_stats.csv    p_value_paired, q_value_bh, significant_fdr
  key_findings.txt

ablation/analysis/
  all_folds_ablation.csv                     subgroup, block, mode, acc_drop, fold, ...
  cross_fold_ablation_{mode}_stats.csv       p_value, q_value_bh, significant_fdr

residual_ablation/
  all_folds_residual_ablation.csv
  feature_vs_residual_comparison_stats.csv   p_value_paired, q_value_bh, significant_fdr

patching/
  fold_N_patching.csv
  all_folds_patching.csv
  patching_summary.csv       subgroup, block, patch_mode, patch_scope, direction,
                             mean_delta_logit, n_valid_baseline, n_iia_success,
                             iia, symmetry_score, causal_effect, iia_overall, ...

joint_patching/
  joint_patching_greedy_fold{N}.csv
  joint_patching_greedy_all_folds.csv
  joint_patching_greedy_summary.csv          mean/std iia_joint and additive-null per step
  joint_patching_topk_fold{N}.csv            (if --mode topk or both)
  joint_patching_topk_all_folds.csv
  joint_patching_topk_summary.csv

benchmark_comparison/
  patching_symmetry_bar_{release}.png / .pdf
  symmetry_vs_iia_{release}.png / .pdf
  sufficiency_curve_{release}.png / .pdf
  cross_release_iia_{release_a}_vs_{release_b}.png / .pdf   (if --patching_csv2 given)
  interpretability_main_{release}.png / .pdf                 (plot_interpretability_main.py,
                                                              main-text 3-panel figure)
```