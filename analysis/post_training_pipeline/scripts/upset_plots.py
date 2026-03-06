"""
UpSet plots for lncRNA classification model comparison.

Produces 4 figures:
  - G47 all-test (5 models: 4 VAE/CNN variants + lncRNA-BERT)
  - G47 hard-cases (4 models, union hard-case universe)
  - G49 all-test (5 models)
  - G49 hard-cases (4 models)

Usage:
    python upset_plots.py --base_dir /path/to/experiments --output_dir ./figures
"""

import argparse
import ast
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from upsetplot import UpSet, from_memberships


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_SLUGS = {
    "beta_vae_features_attn": "βVAE+Attn",
    "beta_vae_contrastive":   "βVAE+Contr.",
    "beta_vae_features":      "βVAE+Feat.",
    "cnn":                    "CNN",
}

BERT_LABEL = "lncRNA-BERT"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_fasta_ids(fa_path: Path) -> set:
    ids = set()
    with open(fa_path) as f:
        for line in f:
            if line.startswith(">"):
                # split on pipe first, then whitespace
                ids.add(line[1:].split("|")[0].strip())
    return ids


def extract_enst(raw_id: str) -> str:
    """Extract ENST accession from lncRNA-BERT pipe-delimited ID field."""
    return raw_id.split("|")[0].strip()


def load_model_predictions(csv_path: Path, test_ids: set) -> pd.DataFrame:
    """
    Load test_predictions.csv for one model, filter to test IDs.
    Returns DataFrame with columns: transcript_id, true_label, pred_label, correct.
    """
    df = pd.read_csv(csv_path)
    df = df[df["transcript_id"].isin(test_ids)].copy()
    df["correct"] = df["true_label"] == df["consensus_prediction"]
    return df[["transcript_id", "true_label", "consensus_prediction", "correct"]]


def load_bert_predictions(csv_path: Path, test_ids: set,
                          ground_truth: dict) -> pd.DataFrame:
    """
    Load lncRNA-BERT results CSV, filter to test IDs, derive correctness.
    ground_truth: dict {transcript_id -> 'lnc'|'pc'}
    """
    df = pd.read_csv(csv_path)
    df["transcript_id"] = df["id"].apply(extract_enst)
    df = df[df["transcript_id"].isin(test_ids)].copy()
    df["pred_label"] = df["class"]   # 'lncRNA' or 'mRNA' — normalise below
    # normalise BERT class strings to match your model labels
    df["pred_label"] = df["pred_label"].map({"pcRNA": "pc", "ncRNA": "lnc"})
    # now pred_label is 'lnc' or 'pc'
    df["true_label"] = df["transcript_id"].map(ground_truth)
    df = df.dropna(subset=["true_label"])
    df["correct"] = df["true_label"] == df["pred_label"]
    return df[["transcript_id", "true_label", "pred_label", "correct"]]


def load_hard_cases(csv_path: Path, test_ids: set) -> set:
    """Return set of transcript IDs that are hard cases for this model."""
    df = pd.read_csv(csv_path)
    df = df[df["transcript_id"].isin(test_ids)]
    return set(df["transcript_id"])


def build_membership_df(correct_dict: dict) -> pd.DataFrame:
    """
    correct_dict: {model_name -> Series(index=transcript_id, values=bool)}
    Returns a DataFrame with one row per transcript, bool columns per model.
    """
    master = pd.DataFrame(correct_dict)
    master.index.name = "transcript_id"
    return master


def make_upset(bool_df: pd.DataFrame, title: str, out_path: Path):
    """
    bool_df: rows = transcripts, columns = model names, values = bool membership.
    For all-test plot: membership = correctly classified.
    For hard-cases plot: membership = is a hard case for that model.
    """
    memberships = []
    for _, row in bool_df.iterrows():
        memberships.append([col for col in bool_df.columns if row[col]])

    upset_data = from_memberships(memberships)

    fig = plt.figure(figsize=(14, 5))
    upset = UpSet(
        upset_data,
        subset_size="count",
        show_counts=True,
        sort_by="cardinality",
        sort_categories_by=None,
        totals_plot_elements=4,
        min_subset_size=20,
    )

    axes = upset.plot(fig)
    axes['intersections'].set_yscale('log')

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    plt.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_version(version: int, base_dir: Path, bert_csv: Path,
                    output_dir: Path):
    """Generate all-test and hard-cases UpSet plots for one GENCODE version."""
    v = str(version)
    print(f"\n=== GENCODE v{v} ===")

    # --- Test IDs and ground truth ---
    lnc_fa = base_dir / f"data/split_gencode_{v}/lnc_test.fa"
    pc_fa  = base_dir / f"data/split_gencode_{v}/pc_test.fa"
    lnc_ids = parse_fasta_ids(lnc_fa)
    pc_ids  = parse_fasta_ids(pc_fa)
    test_ids = lnc_ids | pc_ids
    ground_truth = {tid: "lnc" for tid in lnc_ids}
    ground_truth.update({tid: "pc" for tid in pc_ids})
    print(f"  Test set: {len(lnc_ids)} lnc + {len(pc_ids)} pc = {len(test_ids)} total")

    # --- Load per-model predictions ---
    correct_series = {}   # model_label -> Series(transcript_id -> bool correct)
    hard_sets = {}        # model_label -> set of hard-case transcript_ids

    for slug, label in MODEL_SLUGS.items():
        exp_dir = base_dir / f"gencode_v{v}_experiments/{slug}_g{v}_split/evaluation_csvs"
        test_csv = exp_dir / "test_predictions.csv"
        hard_csv = exp_dir / "test_hard_cases.csv"

        if not test_csv.exists():
            print(f"  WARNING: missing {test_csv}")
            continue

        df = load_model_predictions(test_csv, test_ids)
        correct_series[label] = df.set_index("transcript_id")["correct"]
        print(f"  {label}: {len(df)} test transcripts loaded, "
              f"{df['correct'].sum()} correct ({df['correct'].mean()*100:.1f}%)")

        if hard_csv.exists():
            hard_sets[label] = load_hard_cases(hard_csv, test_ids)
            print(f"    Hard cases: {len(hard_sets[label])}")
        else:
            print(f"  WARNING: missing {hard_csv}")

    # --- Load lncRNA-BERT ---
    if bert_csv and bert_csv.exists():
        bert_df = load_bert_predictions(bert_csv, test_ids, ground_truth)
        correct_series[BERT_LABEL] = bert_df.set_index("transcript_id")["correct"]
        print(f"  {BERT_LABEL}: {len(bert_df)} test transcripts loaded, "
              f"{bert_df['correct'].sum()} correct ({bert_df['correct'].mean()*100:.1f}%)")
    else:
        print(f"  WARNING: lncRNA-BERT CSV not found at {bert_csv}")

    # -----------------------------------------------------------------------
    # Plot 1: All-test UpSet (correct classification membership)
    # -----------------------------------------------------------------------
    # Align on common transcript IDs
    all_models = list(correct_series.keys())
    combined = pd.DataFrame(correct_series).dropna()  # only transcripts present in all models
    print(f"\n  All-test UpSet: {len(combined)} transcripts with predictions from all models")

    make_upset(
        combined,
        title=f"GENCODE v{v} — Test set: correctly classified transcripts per model",
        out_path=output_dir / f"upset_alltest_g{v}.pdf",
    )

    # -----------------------------------------------------------------------
    # Plot 2: Hard-cases UpSet (hard case membership per model, 4 models only)
    # -----------------------------------------------------------------------
    if len(hard_sets) < len(MODEL_SLUGS):
        print("  Skipping hard-cases UpSet: missing hard_cases.csv for some models")
        return

    # Universe = union of all models' hard cases (test-set only)
    hard_universe = set.union(*hard_sets.values())
    print(f"\n  Hard-cases UpSet universe: {len(hard_universe)} unique transcripts "
          f"(union across {len(hard_sets)} models)")

    hard_bool = pd.DataFrame(
        {label: pd.Series({tid: tid in hard_sets[label] for tid in hard_universe})
         for label in MODEL_SLUGS.values() if label in hard_sets}
    )
    hard_bool.index.name = "transcript_id"

    make_upset(
        hard_bool,
        title=f"GENCODE v{v} — Hard cases: model-specific vs. shared difficulty",
        out_path=output_dir / f"upset_hardcases_g{v}.pdf",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default="${DATA_ROOT}",
                        help="Root directory containing gencode_vXX_experiments/ and data/")
    parser.add_argument("--bert_g47", type=Path, default=None,
                        help="Path to lncRNA-BERT G47 results CSV")
    parser.add_argument("--bert_g49", type=Path, default=None,
                        help="Path to lncRNA-BERT G49 results CSV")
    parser.add_argument("--output_dir", type=Path, default=Path("./figures/upset"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    process_version(47, args.base_dir, args.bert_g47, args.output_dir)
    process_version(49, args.base_dir, args.bert_g49, args.output_dir)

    print("\nDone. All figures saved to", args.output_dir)


if __name__ == "__main__":
    main()