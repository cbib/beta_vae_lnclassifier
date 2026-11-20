#!/usr/bin/env python3
"""Layer 2: Fréchet distance between the lncRNA and mRNA feature distributions, per subgroup.

Per subgroup the raw features are z-scored, both classes are modelled as Gaussians, and
FD = ||mu1 - mu2||^2 + trace(S1 + S2 - 2 (S1^1/2 S2 S1^1/2)^1/2) (mean term + covariance term).
The mean term generalises the class-mean shift of Karkada et al. 2025 (word2vec theory);
the covariance term adds the difference in spread. Model-independent: trainval data only.

Usage:
  python -m analysis.pattern.analyze_frechet --config CFG --output_dir DIR

Writes in DIR:
  frechet_ranking.csv  subgroup, block, frechet_dist, mean_term, cov_term, cov_fraction,
                       fd_status. A failed computation gives NaN and fd_status "failed:...", never 0.
  within_class_covariance_{lncrna,mrna}.csv  covariance across subgroups of the z-scored
                       subgroup means (each the mean of its raw features), per class.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm as scipy_sqrtm
from sklearn.preprocessing import StandardScaler

from configs.load_config import load_config
from data.cv_utils import load_sequences_in_order
from data.feature_registry import REGISTRY


def load_features(config) -> tuple[dict, np.ndarray]:
    """
    Load processed feature CSVs and return:
      block_dfs : {block_name: pd.DataFrame} aligned to sequence order
      labels    : (N,) int — 0=lncRNA 1=mRNA
    """
    print("Loading sequences and labels...")
    all_sequences, labels_str = load_sequences_in_order(
        lnc_fasta=config.get("data", "lnc_fasta"),
        pc_fasta= config.get("data", "pc_fasta"),
    )
    labels = np.array([0 if l == "lnc" else 1 for l in labels_str])
    print(f"  lncRNA: {(labels==0).sum():,}  mRNA: {(labels==1).sum():,}")

    # Transcript IDs for alignment — strip pipe-suffix AND version suffix.
    # Must match however the feature CSV indices are normalised below.
    ids = [str(seq.id).split("|")[0].split(".")[0] for seq in all_sequences]

    def _load_and_dedupe(path: str, label: str) -> pd.DataFrame:
        print(f"Loading {label} features...")
        df = pd.read_csv(path, index_col=0)
        print(f"  Raw shape: {df.shape}")

        # Normalise index the same way as the sequence IDs above
        df.index = (df.index.astype(str)
                    .str.split("|").str[0]
                    .str.split(".").str[0])

        n_before = len(df)
        n_unique = df.index.nunique()
        if n_unique < n_before:
            n_dupes = n_before - n_unique
            print(f"  WARNING: {n_dupes:,} duplicate transcript_id rows "
                  f"({n_before:,} rows, {n_unique:,} unique) — "
                  f"keeping first occurrence")
            df = df[~df.index.duplicated(keep="first")]

        print(f"  Deduped shape: {df.shape}")
        return df

    # ── NonB ──────────────────────────────────────────────────────────────
    nonb_path = config.get("data", "nonb_processed_csv", default=None)
    if nonb_path is None:
        nonb_path = config.get("data", "nonb_genomic_csv")
        print("  (nonb_processed_csv not set, using genomic)")
    nonb_df = _load_and_dedupe(nonb_path, "NonB processed")

    # ── TE ────────────────────────────────────────────────────────────────
    te_path = config.get("data", "te_processed_csv", default=None)
    if te_path is None:
        te_path = config.get("data", "te_genomic_csv")
        print("  (te_processed_csv not set, using genomic)")
    te_df = _load_and_dedupe(te_path, "TE processed")

    # ── NonB2 (optional) ─────────────────────────────────────────────────
    nonb2_path = config.get("data", "nonb2_csv", default=None)
    if nonb2_path is None:
        nonb2_path = config.get("data", "nonb2_processed_csv", default=None)
        if nonb2_path is not None:
            print("  (nonb2_csv not set, using nonb2_processed_csv)")
    nonb2_df = (_load_and_dedupe(nonb2_path, "NonB2")
                if nonb2_path is not None else None)

    # ── Align all blocks to sequence order ──────────────────────────────
    # Safe now — every DataFrame has a unique index after dedup above
    nonb_df = nonb_df.reindex(ids)
    te_df   = te_df.reindex(ids)
    if nonb2_df is not None:
        nonb2_df = nonb2_df.reindex(ids)

    # ── Drop rows with all-NaN in ANY block (unmatched transcripts) ────
    valid = (~nonb_df.isnull().all(axis=1)) & (~te_df.isnull().all(axis=1))
    if nonb2_df is not None:
        valid = valid & (~nonb2_df.isnull().all(axis=1))

    print(f"  Valid transcripts after alignment: {valid.sum():,} / {len(valid):,}")

    # Use a numpy boolean array, not the pandas Series, to avoid any
    # remaining index-alignment surprises when filtering each DataFrame
    valid_arr = valid.values

    nonb_df = nonb_df[valid_arr].fillna(0.0)
    te_df   = te_df[valid_arr].fillna(0.0)
    labels  = labels[valid_arr]

    block_dfs = {"nonb": nonb_df, "te": te_df}
    if nonb2_df is not None:
        nonb2_df = nonb2_df[valid_arr].fillna(0.0)
        block_dfs["nonb2"] = nonb2_df

    return block_dfs, labels



def frechet_cov_term(S1, S2, tag="", rel_tol=1e-6):
    """Covariance term of W2^2 between two Gaussians.

    Returns (value, status). On numerical failure returns (NaN, "failed:<why>")
    and warns; never substitutes 0.
    """
    scale = float(np.trace(S1) + np.trace(S2)) or 1.0
    try:
        if not (np.isfinite(S1).all() and np.isfinite(S2).all()):
            raise FloatingPointError("non-finite covariance")
        r1 = scipy_sqrtm(S1)
        r2 = scipy_sqrtm(r1 @ S2 @ r1)
        if not (np.isfinite(r1).all() and np.isfinite(r2).all()):
            raise FloatingPointError("non-finite square root")
        imag = max(float(np.abs(np.imag(r1)).max()),
                   float(np.abs(np.imag(r2)).max()))
        if imag > rel_tol * scale:
            raise FloatingPointError(f"imaginary part {imag:.2e}")
        val = float(np.trace(S1 + S2 - 2.0 * np.real(r2)))
    except Exception as exc:
        warnings.warn(f"Frechet covariance term failed for {tag or 'sub-group'}: {exc}")
        return float("nan"), f"failed:{type(exc).__name__}"
    if val < -rel_tol * scale:
        warnings.warn(f"Frechet covariance term negative for {tag or 'sub-group'}: {val:.3e}")
        return float("nan"), "failed:negative_cov_term"
    return max(val, 0.0), "ok"




def frechet_analysis(block_dfs, labels, reg=1e-4):
    """Ranking DataFrame, one row per subgroup in REGISTRY order, sorted by frechet_dist.

    block_dfs: {block: raw feature DataFrame}; labels: 0 = lncRNA, 1 = mRNA;
    reg: diagonal added to both covariances. Failures give NaN (sorted last), never 0."""
    rows, nan = [], float("nan")
    for block in REGISTRY.block_names:
        df = block_dfs.get(block)
        if df is None:
            continue
        for sg in REGISTRY.block_subgroups(block):
            idx = REGISTRY.indices_for_subgroup(sg)
            row = {"subgroup": sg, "block": block}
            if not idx:
                warnings.warn(f"Frechet: no features for sub-group {sg}")
                rows.append({**row, "frechet_dist": nan, "mean_term": nan, "cov_term": nan,
                             "cov_fraction": nan, "fd_status": "failed:no_features"})
                continue
            X = StandardScaler().fit_transform(df.iloc[:, idx].values.astype(np.float64))
            lnc, mrna = X[labels == 0], X[labels == 1]
            eye = reg * np.eye(X.shape[1])
            mean_term = float(np.sum((lnc.mean(axis=0) - mrna.mean(axis=0)) ** 2))
            cov_term, status = frechet_cov_term(np.cov(lnc.T) + eye, np.cov(mrna.T) + eye, tag=sg)
            fd = mean_term + cov_term  # NaN propagates
            rows.append({**row, "frechet_dist": fd, "mean_term": mean_term, "cov_term": cov_term,
                         "cov_fraction": cov_term / max(fd, 1e-8) if np.isfinite(fd) else nan,
                         "fd_status": status})
    names = [r["subgroup"] for r in rows]
    if names != list(REGISTRY.all_subgroups):
        raise ValueError(f"Frechet sub-groups differ from the registry (missing block?): {names}")
    return pd.DataFrame(rows).sort_values("frechet_dist", ascending=False, na_position="last")


def subgroup_covariance(block_dfs, labels):
    """(lncRNA, mRNA) DataFrames: covariance across subgroups of the z-scored subgroup means.

    A subgroup's value is the mean of its raw feature columns; z-scoring uses all transcripts.
    Rows and columns follow REGISTRY order."""
    cols = {}
    for block in REGISTRY.block_names:
        df = block_dfs.get(block)
        if df is None:
            continue
        for sg in REGISTRY.block_subgroups(block):
            idx = REGISTRY.indices_for_subgroup(sg)
            cols[sg] = df.iloc[:, idx].mean(axis=1).values if idx else np.zeros(len(df))
    names = list(cols)
    if names != list(REGISTRY.all_subgroups):
        raise ValueError(f"Subgroups differ from the registry (missing block?): {names}")
    X = StandardScaler().fit_transform(np.column_stack(list(cols.values())))
    return tuple(pd.DataFrame(np.cov(X[labels == c].T), index=names, columns=names) for c in (0, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    block_dfs, labels = load_features(load_config(args.config))
    ranking = frechet_analysis(block_dfs, labels)
    ranking.to_csv(out / "frechet_ranking.csv", index=False)
    for name, cov in zip(("lncrna", "mrna"), subgroup_covariance(block_dfs, labels)):
        cov.to_csv(out / f"within_class_covariance_{name}.csv")

    failed = ranking.loc[ranking["fd_status"] != "ok", "subgroup"].tolist()
    if failed:
        print(f"WARNING: Frechet distance failed for {', '.join(failed)} (NaN, see fd_status)")
    print(ranking.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
