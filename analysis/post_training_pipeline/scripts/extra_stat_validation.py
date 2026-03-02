#!/usr/bin/env python3
"""
Statistical validation of classification performance.
Computes:
  1. Fold-wise summary: mean ± SD, 95% CI (t-dist) for F1/precision/recall
  2. Bootstrap 95% CI for test set macro F1
  3. DeLong 95% CI for test set AUC (requires probability scores)
  4. Wilcoxon signed-rank + Holm correction for model pairs (requires all models)

Usage:
    python extra_stat_validation.py \
        --fold_results model_a/fold_results.json model_b/fold_results.json \
        --test_preds   model_a/test_predictions.csv model_b/test_predictions.csv \
        --labels       "βVAE+Attn" "βVAE+Feat" \
        --output_dir   stat_results/
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, roc_auc_score
from itertools import combinations


# ─────────────────────────────────────────────
# 1. Fold-wise summary
# ─────────────────────────────────────────────
def fold_summary(fold_results_path: str, label: str) -> list:
    with open(fold_results_path) as f:
        data = json.load(f)

    # Handle both formats
    if isinstance(data, list):
        # fold_results.json — plain list of fold dicts
        folds = data
    elif 'fold_results' in data:
        # cv_evaluation_results.json — nested under fold_results key
        folds = data['fold_results']
    else:
        raise ValueError(f"Unrecognised format in {fold_results_path}")

    # Extract per-fold metrics — handle nested 'deterministic' key if present
    def get_metric(fold, metric):
        if 'deterministic' in fold:
            return fold['deterministic'][metric]
        return fold[metric]

    metrics = ['f1', 'precision', 'recall']
    rows = []
    for metric in metrics:
        values = np.array([get_metric(f, metric) for f in folds])
        n = len(values)
        mean, std = values.mean(), values.std(ddof=1)
        se = std / np.sqrt(n)
        ci_lo, ci_hi = stats.t.interval(0.95, df=n-1, loc=mean, scale=se)
        rows.append({
            'model':  label,
            'metric': metric,
            'mean':   round(mean, 4),
            'std':    round(std, 4),
            'ci_lo':  round(ci_lo, 4),
            'ci_hi':  round(ci_hi, 4),
            'values': values.tolist(),
        })
    return rows


# ─────────────────────────────────────────────
# 2. Bootstrap CI for test macro F1
# ─────────────────────────────────────────────
def bootstrap_f1_ci(y_true, y_pred, n_bootstrap=10000, seed=42):
    """Paired bootstrap 95% CI for macro F1."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores.append(f1_score(y_true[idx], y_pred[idx],
                               average='macro', zero_division=0))
    scores = np.array(scores)
    return {
        'f1_observed': round(f1_score(y_true, y_pred,
                                      average='macro', zero_division=0), 4),
        'ci_lo': round(np.percentile(scores, 2.5), 4),
        'ci_hi': round(np.percentile(scores, 97.5), 4),
        'n_bootstrap': n_bootstrap,
    }


# ─────────────────────────────────────────────
# 3. DeLong AUC CI
# ─────────────────────────────────────────────
def delong_auc_ci(y_true, y_score, alpha=0.05):
    """
    DeLong et al. (1988) method for AUC confidence interval.
    y_score: predicted probability of positive class.
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos, n_neg = len(pos), len(neg)

    # Placement values
    psi_pos = np.array([np.mean(p > neg) + 0.5 * np.mean(p == neg) for p in pos])
    psi_neg = np.array([np.mean(n < pos) + 0.5 * np.mean(n == pos) for n in neg])

    auc = psi_pos.mean()

    # Variance via structural components
    var_pos = np.var(psi_pos, ddof=1) / n_pos
    var_neg = np.var(psi_neg, ddof=1) / n_neg
    se = np.sqrt(var_pos + var_neg)

    z = stats.norm.ppf(1 - alpha / 2)
    return {
        'auc': round(auc, 4),
        'se': round(se, 6),
        'ci_lo': round(max(0, auc - z * se), 4),
        'ci_hi': round(min(1, auc + z * se), 4),
    }


# ─────────────────────────────────────────────
# 4. Wilcoxon signed-rank + Holm correction
# ─────────────────────────────────────────────
def pairwise_wilcoxon(fold_data: dict, metric='f1'):
    """
    fold_data: {label: [fold_0_score, fold_1_score, ...]}
    Returns DataFrame with pairwise Wilcoxon results + Holm correction.
    """
    labels = list(fold_data.keys())
    rows = []
    for a, b in combinations(labels, 2):
        xa = np.array(fold_data[a])
        xb = np.array(fold_data[b])
        diff = xa - xb
        if np.all(diff == 0):
            stat, p = np.nan, 1.0
        else:
            stat, p = wilcoxon(xa, xb, alternative='two-sided')
        rows.append({
            'model_a': a,
            'model_b': b,
            'mean_diff': round((xa - xb).mean(), 4),
            'statistic': stat,
            'p_value': p,
        })

    df = pd.DataFrame(rows).sort_values('p_value')

    # Holm correction
    n = len(df)
    df['p_holm'] = [min(1.0, p * (n - i))
                    for i, p in enumerate(df['p_value'])]
    df['significant'] = df['p_holm'] < 0.05
    df['p_holm'] = df['p_holm'].round(4)
    return df


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold_results', nargs='+', required=True,
                        help='fold_results.json files, one per model')
    parser.add_argument('--test_preds', nargs='+', required=True,
                        help='test_predictions.csv files, one per model')
    parser.add_argument('--labels', nargs='+', required=True,
                        help='Model labels (same order as above)')
    parser.add_argument('--prob_col', default='mean_confidence',
                        help='Column with predicted probability for DeLong')
    parser.add_argument('--output_dir', default='stat_results')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assert len(args.fold_results) == len(args.test_preds) == len(args.labels), \
        "Must provide equal numbers of fold_results, test_preds, and labels"

    # ── 1 & 2 & 3: per-model analysis ──
    all_fold_summary = []
    all_bootstrap = []
    all_delong = []
    fold_f1_data = {}  # for Wilcoxon

    for fold_path, pred_path, label in zip(
            args.fold_results, args.test_preds, args.labels):

        # Fold summary
        with open(fold_path) as f:
            fold_results = json.load(f)
        rows = fold_summary(fold_path, label)
        all_fold_summary.extend(rows)
        fold_f1_data[label] = [r['values'] for r in rows
                                if r['metric'] == 'f1'][0]

        # Test set
        df = pd.read_csv(pred_path)
        y_true = (df['true_label'] == 'pc').astype(int).values
        y_pred = (df['consensus_prediction'] == 'pc').astype(int).values

        # Bootstrap F1
        boot = bootstrap_f1_ci(y_true, y_pred)
        boot['model'] = label
        all_bootstrap.append(boot)

        # DeLong AUC — only if probability column is meaningful
        if args.prob_col in df.columns:
            # mean_confidence is max(P(lnc), P(pc)) — need to check direction
            # If consensus_prediction == 'pc', confidence is P(pc); else P(lnc)
            y_score = np.where(
                df['consensus_prediction'] == 'pc',
                df[args.prob_col],
                1 - df[args.prob_col]
            )
            dl = delong_auc_ci(y_true, y_score)
            dl['model'] = label
            all_delong.append(dl)
        else:
            print(f"  WARNING: {args.prob_col} not found in {pred_path}, "
                  f"skipping DeLong for {label}")

    # ── 4: Wilcoxon pairwise ──
    wilcoxon_df = pairwise_wilcoxon(fold_f1_data, metric='f1')

    # ── Save outputs ──
    fold_df = pd.DataFrame(all_fold_summary).drop(columns='values')
    fold_df.to_csv(output_dir / 'fold_summary.csv', index=False)

    boot_df = pd.DataFrame(all_bootstrap)
    boot_df.to_csv(output_dir / 'bootstrap_f1_ci.csv', index=False)

    if all_delong:
        delong_df = pd.DataFrame(all_delong)
        delong_df.to_csv(output_dir / 'delong_auc_ci.csv', index=False)

    wilcoxon_df.to_csv(output_dir / 'wilcoxon_pairwise.csv', index=False)

    # ── Print summary ──
    print("\n── Fold-wise summary ──")
    print(fold_df.to_string(index=False))

    print("\n── Bootstrap F1 CI (test set) ──")
    print(boot_df.to_string(index=False))

    if all_delong:
        print("\n── DeLong AUC CI (test set) ──")
        print(delong_df.to_string(index=False))

    print("\n── Wilcoxon pairwise (fold F1, Holm-corrected) ──")
    print(wilcoxon_df.to_string(index=False))


if __name__ == '__main__':
    main()