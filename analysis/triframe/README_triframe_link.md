# β-LNC ↔ triframe link (branch `link/triframe-equivalence`)

Question: does triframe's Layer 3, driven through an independent adapter,
reproduce the published β-LNC patching rows? **β-LNC outputs are the reference.**

- `blnc_triframe_adapter.py` — `BlncAdapter`: projector-site capture/patch,
  written independently of `patch_tokens.py`.
- `compare_with_triframe.py` — reads a released `all_folds_patching.csv`,
  re-runs one fold's pairs/partners through triframe, compares per pair
  (baseline, patched, delta logit) and in aggregate (IIA via
  `interchange_intervention_accuracy`, mean |Δ| via `symmetry_score`).
  Writes `equivalence_report.md` + CSVs; exit code 0 = PASS.

Five parts, one script (any subset; each gated separately):
- **Probing (1.4, Layer 1):** `--repr_dir 1.1_representations --probing_dir 1.4_latent_probing`.
  Per fold, triframe's `representation_redundancy` (alpha=1, clip_per_dim, in-sample) and
  `pattern_alignment` (raw, and with z regressed out) vs `redundancy.csv`, `pattern_raw.csv`,
  `pattern_residual.csv` (tol `--probing_atol`, 1e-5). No model needed. Permutation p-values are
  not covered: triframe has no null.
- **Ablation (1.2, Layer 1):** `--ablation_reference all_folds_ablation.csv` (+ `--experiment_dir`).
  `run_ablation` with its own token_zero / feature_zero hooks on each fold's validation set;
  baseline, acc_drop and f1_drop vs the release (tol `--ablation_atol`, 1e-9: counts, so exact).
  Cost = same forward passes as stage 1.2; restrict with `--folds` / `--ablation_modes`.
- **Fréchet (2.1, Layer 2):** `triframe.layer2.frechet_by_token` vs the released
  `frechet_ranking.csv`, on features/labels from β-LNC's own loader. Per sub-group
  frechet_dist/mean_term/cov_term within `--frechet_rtol/atol` (1e-6/1e-8), NaN must
  match NaN, same sub-groups and ranking. Needs no model.
- **Patching (3.1):** every group of every fold in `all_folds_patching.csv`
  (`--max_pairs 0` = all 200 pairs; `--folds all` = the full release).
- **Greedy (3.3):** pairs rebuilt with β-LNC's own `select_pairs`/`split_pairs`
  (checked to extend the 3.1 pairs), search run by
  `triframe.layer3.greedy_token_search`; compared step by step (token order, IIA
  selection + held-out, additive null, symmetry). Pass the release's settings
  (`--greedy_n_pairs`, `--max_search_depth`, ...; defaults match `joint_patch.sh`).
  `logit_interaction_*` columns are not covered.

Pass rule: all |Δ logit| ≤ `--atol_logit` (use 1e-5 with `--batch_size 1`; batched
runs show ~1e-4 float noise), IIA equal in every group (a flip is only accepted
when the reference probability is within 1e-4 of 0.5), greedy token order
identical and IIA gaps ≤ 1e-9.
Requires triframe 0.1.0: `git clone https://github.com/cbib/triframe.git && pip install -e triframe`
(or the repo on PYTHONPATH).

Definitions (β-LNC is the reference; triframe now exposes each variant):
1. Noise control: `noise_null_control(mode="norm_matched", source_batches=…)`
   = random isotropic direction with the partner token's L2 norm (β-LNC).
   `mode="gaussian"` (default) is triframe's original. Random vectors differ by
   RNG, so noise is compared in distribution, not per pair.
2. Symmetry: β-LNC's symmetry_score/causal_effect = `net_effect_symmetry`
   (|mean| per direction); triframe's `symmetry_score` is mean |Δ| (also
   checked against the reference rows).
3. IIA: `interchange_intervention_accuracy(valid_fn=…)` drops pairs whose
   baseline is already on the target side, as β-LNC's iia_baseline_valid does.
4. Pair selection: β-LNC picks random partners within confidence/length/GC
   gates; triframe's `nearest_neighbor` picks the nearest. Pairs are supplied
   from the reference, not re-selected.