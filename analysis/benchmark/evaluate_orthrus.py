#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_orthrus.py

Linear probing evaluation of Orthrus 4-track on the β-LNC benchmark.

Protocol
--------
1.  Load Orthrus 4-track from HuggingFace (frozen, no gradient).
2.  Extract pooled embeddings (B, 512) for the full training FASTA pair.
3.  Fit a logistic regression (L2, liblinear) on training embeddings.
4.  Evaluate on the held-out test FASTA pair.
5.  Optionally save per-position (unpooled) embeddings as a ragged .npz
    for downstream positional analysis (TE/NonB locus probing, etc.).

Outputs written to --output_dir
--------------------------------
  orthrus_test_predictions.csv   — per-sample predictions + confidence
  orthrus_test_hard_cases.csv    — samples where confidence < hard_conf_thresh
                                   OR prediction is wrong
  orthrus_test_metrics.json      — accuracy / precision / recall / F1 / CM
  orthrus_train_embeddings.npy   — (N_train, 512) pooled embeddings
  orthrus_test_embeddings.npy    — (N_test,  512) pooled embeddings
  orthrus_train_ids.txt          — transcript IDs matching train embedding rows
  orthrus_test_ids.txt           — transcript IDs matching test embedding rows
  [optional, --save_unpooled]
  orthrus_test_unpooled.h5       — per-position hidden states streamed to disk
                                   batch by batch (no full-dataset RAM spike).
                                   Layout: /embeddings/{transcript_id} (L_i, 512)
                                   float16 lzf-compressed; /lengths; /transcript_ids
                                   Load: h5py.File(...)["embeddings/ENST00000123"][:]

Usage
-----
# Basic
python src/evaluate_orthrus.py \
    --train_lnc_fasta  data/split_gencode_49/lnc_trainval.fa \
    --train_pc_fasta   data/split_gencode_49/pc_trainval.fa \
    --test_lnc_fasta   data/split_gencode_49/lnc_test.fa \
    --test_pc_fasta    data/split_gencode_49/pc_test.fa \
    --output_dir       gencode_v49_experiments/orthrus_evaluation

# With unpooled positional embeddings (uses a separate smaller batch size):
python src/evaluate_orthrus.py \
    ... \
    --save_unpooled \
    --unpooled_batch_size 16

Notes
-----
* Orthrus uses T in place of U. seq_to_oh handles this — no manual conversion.
* Sequences longer than --max_length are truncated at the 3' end.
* The logistic regression is intentionally minimal (L2, liblinear, max_iter=1000,
  C=1.0) to match the linear probing setup in the Orthrus paper.
* Unpooled embeddings are saved as float16 to keep disk footprint manageable.
  Cast back to float32 before downstream arithmetic.
* Memory note: at max_length=15000, the unpooled tensor per batch is
  (B, 15000, 512) float16. At B=128 this is ~23 GB. --unpooled_batch_size
  (default 16, ~3 GB per batch) controls only the unpooled pass; --batch_size
  controls the pooled pass and can remain large.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel


# ---------------------------------------------------------------------------
# FASTA loading
# ---------------------------------------------------------------------------

def load_fasta_pair(
    lnc_fasta: str,
    pc_fasta:  str,
) -> Tuple[List[SeqRecord], np.ndarray]:
    lnc_seqs = list(SeqIO.parse(lnc_fasta, "fasta"))
    pc_seqs  = list(SeqIO.parse(pc_fasta,  "fasta"))
    print(f"  lncRNA  : {len(lnc_seqs):,}")
    print(f"  pcRNA   : {len(pc_seqs):,}")
    sequences = lnc_seqs + pc_seqs
    labels    = np.array([0] * len(lnc_seqs) + [1] * len(pc_seqs), dtype=np.int64)
    return sequences, labels


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OrthrusSequenceDataset(Dataset):
    def __init__(
        self,
        sequences:  List[SeqRecord],
        labels:     np.ndarray,
        model,
        max_length: int = 10_000,
    ) -> None:
        self.sequences  = sequences
        self.labels     = labels
        self.model      = model
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq_str = str(self.sequences[idx].seq).upper()[: self.max_length]
        tid     = self.sequences[idx].id.split("|")[0]
        length  = len(seq_str)

        oh = self.model.seq_to_oh(seq_str)                       # (L, 4)
        if length < self.max_length:
            pad = torch.zeros(self.max_length - length, 4, dtype=oh.dtype)
            oh  = torch.cat([oh, pad], dim=0)

        return {
            "one_hot":       oh,
            "length":        length,
            "label":         int(self.labels[idx]),
            "transcript_id": tid,
        }


def collate_fn(batch):
    one_hots = torch.stack([b["one_hot"]  for b in batch])
    lengths  = torch.tensor([b["length"]  for b in batch])
    labels   = torch.tensor([b["label"]   for b in batch])
    ids      = [b["transcript_id"]        for b in batch]
    return one_hots, lengths, labels, ids


# ---------------------------------------------------------------------------
# Embedding extraction — pooled and unpooled are separate passes
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_pooled(
    model,
    loader: DataLoader,
    device: torch.device,
    desc:   str = "Pooled",
) -> dict:
    """
    Extract mean-pooled embeddings. Fast; batch_size can be large.

    Returns
    -------
    pooled        : np.ndarray (N, D)   float32
    labels        : np.ndarray (N,)     int64
    transcript_ids: list[str]
    lengths       : np.ndarray (N,)     int
    """
    model.eval()
    all_pooled, all_labels, all_ids, all_lengths = [], [], [], []

    for one_hots, lengths, labels, ids in tqdm(loader, desc=f"  {desc}"):
        one_hots = one_hots.to(device)
        lengths  = lengths.to(device)
        pooled   = model.representation(one_hots, lengths, channel_last=True)
        all_pooled.append(pooled.cpu().float().numpy())
        all_labels.append(labels.numpy())
        all_ids.extend(ids)
        all_lengths.append(lengths.cpu().numpy())

    return dict(
        pooled         = np.vstack(all_pooled),
        labels         = np.concatenate(all_labels),
        transcript_ids = all_ids,
        lengths        = np.concatenate(all_lengths),
    )


@torch.no_grad()
def extract_unpooled_to_hdf5(
    model,
    loader:      DataLoader,
    device:      torch.device,
    output_path: Path,
    transcript_ids: List[str],
    lengths:     np.ndarray,
    desc:        str = "Unpooled",
) -> None:
    """
    Extract per-position hidden states and stream directly to an HDF5 file.

    Nothing is accumulated in RAM — each batch is written to disk immediately
    after moving to CPU, so peak memory is a single batch rather than the full
    dataset. The resulting file uses variable-length datasets (one per
    transcript) so no padding is stored on disk either.

    Output HDF5 layout
    ------------------
      /embeddings/{transcript_id}   float16  (L_i, D)
      /lengths                      int64    (N,)
      /transcript_ids               bytes    (N,)

    To load a single transcript later:
        import h5py
        with h5py.File("orthrus_test_unpooled.h5", "r") as f:
            emb = f["embeddings/ENST00000123456"][:]   # (L_i, 512) float16
    """
    import h5py

    model.eval()
    sample_idx = 0

    with h5py.File(output_path, "w") as hf:
        emb_grp = hf.create_group("embeddings")
        hf.create_dataset("lengths",        data=lengths,
                          dtype="int64")
        hf.create_dataset("transcript_ids", data=np.array(transcript_ids,
                          dtype=h5py.special_dtype(vlen=str)))

        for one_hots, batch_lengths, _, _ in tqdm(loader, desc=f"  {desc}"):
            one_hots  = one_hots.to(device)
            hidden    = model.representation_unpooled(one_hots, channel_last=True)
            hidden_np = hidden.cpu().to(torch.float16).numpy()   # (B, L, D) fp16

            for i, length in enumerate(batch_lengths.numpy()):
                tid = transcript_ids[sample_idx]
                emb_grp.create_dataset(
                    tid,
                    data  = hidden_np[i, :length, :],            # (L_i, D) fp16
                    compression = "lzf",                         # fast, lossless
                )
                sample_idx += 1

    size_gb = output_path.stat().st_size / 1e9
    print(f"  Saved unpooled: {output_path.name}  "
          f"({len(transcript_ids):,} transcripts, {size_gb:.2f} GB on disk)")


# ---------------------------------------------------------------------------
# Metrics + per-sample CSV
# ---------------------------------------------------------------------------

def compute_metrics_and_records(
    labels:         np.ndarray,
    predictions:    np.ndarray,
    probabilities:  np.ndarray,
    transcript_ids: List[str],
    sequences:      List[SeqRecord],
    hard_conf_thresh: float = 0.6,
) -> Tuple[dict, pd.DataFrame]:
    idx_to_label = {0: "lnc", 1: "pc"}

    acc                       = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0,
    )
    cm = confusion_matrix(labels, predictions)

    metrics = dict(
        accuracy         = float(acc),
        precision        = float(precision),
        recall           = float(recall),
        f1               = float(f1),
        confusion_matrix = cm.tolist(),
        n_samples        = int(len(labels)),
    )

    seq_len_map = {
        rec.id.split("|")[0]: len(str(rec.seq)) for rec in sequences
    }

    records = []
    for i, tid in enumerate(transcript_ids):
        pred       = int(predictions[i])
        true       = int(labels[i])
        confidence = float(probabilities[i].max())
        error_rate = float(pred != true)
        is_hard    = (error_rate > 0) or (confidence < hard_conf_thresh)

        records.append(dict(
            transcript_id   = tid,
            true_label      = idx_to_label[true],
            prediction      = idx_to_label[pred],
            confidence      = confidence,
            prob_lnc        = float(probabilities[i, 0]),
            prob_pc         = float(probabilities[i, 1]),
            sequence_length = seq_len_map.get(tid, -1),
            error_rate      = error_rate,
            is_hard_case    = bool(is_hard),
        ))

    return metrics, pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Orthrus 4-track linear probing evaluation for β-LNC benchmark"
    )
    parser.add_argument("--train_lnc_fasta",     required=True)
    parser.add_argument("--train_pc_fasta",      required=True)
    parser.add_argument("--test_lnc_fasta",      required=True)
    parser.add_argument("--test_pc_fasta",       required=True)
    parser.add_argument("--output_dir",          required=True)
    parser.add_argument("--model_name",          default="antichronology/orthrus-4-track")
    parser.add_argument("--device",
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size",          type=int,   default=64,
                        help="Batch size for pooled embedding extraction (default 64)")
    parser.add_argument("--unpooled_batch_size", type=int,   default=8,
                        help="Batch size for unpooled extraction — keep small to "
                             "avoid OOM with long sequences (default 8)")
    parser.add_argument("--max_length",          type=int,   default=10_000)
    parser.add_argument("--num_workers",         type=int,   default=0)
    parser.add_argument("--lr_C",                type=float, default=1.0)
    parser.add_argument("--hard_conf_thresh",    type=float, default=0.6)
    parser.add_argument("--save_unpooled",       action="store_true",
                        help="Extract and save per-position hidden states for test set")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 65)
    print("Orthrus 4-track — Linear Probing Evaluation")
    print("=" * 65)
    print(f"Model              : {args.model_name}")
    print(f"Device             : {device}")
    print(f"Max length         : {args.max_length:,}")
    print(f"Batch size (pooled): {args.batch_size}")
    if args.save_unpooled:
        print(f"Batch size (unpld) : {args.unpooled_batch_size}")
    print(f"LR C               : {args.lr_C}")
    print(f"Save unpooled      : {args.save_unpooled}")
    print(f"Output             : {output_dir}")
    print("=" * 65)

    # ── Load model ────────────────────────────────────────────────────────────
    print("\nLoading Orthrus model...")
    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    ).to(device).eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Load sequences ────────────────────────────────────────────────────────
    print("\nLoading training sequences...")
    train_seqs, train_labels = load_fasta_pair(
        args.train_lnc_fasta, args.train_pc_fasta
    )
    print(f"  Total train: {len(train_seqs):,}")

    print("\nLoading test sequences...")
    test_seqs, test_labels = load_fasta_pair(
        args.test_lnc_fasta, args.test_pc_fasta
    )
    print(f"  Total test : {len(test_seqs):,}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = OrthrusSequenceDataset(
        train_seqs, train_labels, model, max_length=args.max_length
    )
    test_dataset = OrthrusSequenceDataset(
        test_seqs, test_labels, model, max_length=args.max_length
    )

    def make_loader(dataset, batch_size):
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(args.num_workers > 0),
            collate_fn=collate_fn,
        )

    # ── Pooled embeddings — both train and test ───────────────────────────────
    print("\n" + "=" * 65)
    print("Extracting training embeddings (pooled)")
    print("=" * 65)
    train_out = extract_pooled(
        model, make_loader(train_dataset, args.batch_size), device, desc="Train"
    )
    print(f"  Shape: {train_out['pooled'].shape}")

    print("\n" + "=" * 65)
    print("Extracting test embeddings (pooled)")
    print("=" * 65)
    test_out = extract_pooled(
        model, make_loader(test_dataset, args.batch_size), device, desc="Test"
    )
    print(f"  Shape: {test_out['pooled'].shape}")

    # ── Save pooled ───────────────────────────────────────────────────────────
    np.save(output_dir / "orthrus_train_embeddings.npy", train_out["pooled"])
    np.save(output_dir / "orthrus_test_embeddings.npy",  test_out["pooled"])
    (output_dir / "orthrus_train_ids.txt").write_text(
        "\n".join(train_out["transcript_ids"])
    )
    (output_dir / "orthrus_test_ids.txt").write_text(
        "\n".join(test_out["transcript_ids"])
    )
    print(f"\n  Pooled embeddings saved to {output_dir}")

    # ── Linear probe ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Fitting linear probe (LogisticRegression, L2, liblinear)")
    print("=" * 65)
    clf = LogisticRegression(
        C            = args.lr_C,
        solver       = "liblinear",
        max_iter     = 1000,
        random_state = 42,
    )
    clf.fit(train_out["pooled"], train_out["labels"])
    print("  Probe fitted.")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Test set evaluation")
    print("=" * 65)
    probabilities = clf.predict_proba(test_out["pooled"])
    predictions   = probabilities.argmax(axis=1)

    metrics, df = compute_metrics_and_records(
        labels           = test_out["labels"],
        predictions      = predictions,
        probabilities    = probabilities,
        transcript_ids   = test_out["transcript_ids"],
        sequences        = test_seqs,
        hard_conf_thresh = args.hard_conf_thresh,
    )

    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  Confusion matrix:\n    {np.array(metrics['confusion_matrix'])}")

    hard_df = df[df["is_hard_case"]]
    df.to_csv(     output_dir / "orthrus_test_predictions.csv", index=False)
    hard_df.to_csv(output_dir / "orthrus_test_hard_cases.csv",  index=False)
    with open(output_dir / "orthrus_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    n_hard = len(hard_df)
    n_tot  = len(df)
    print(f"\n  test_predictions.csv : {n_tot:,} samples")
    print(f"  test_hard_cases.csv  : {n_hard:,} hard cases "
          f"({100 * n_hard / n_tot:.1f}%)")
    for lab in ["lnc", "pc"]:
        n_c   = (df["true_label"] == lab).sum()
        n_c_h = ((df["true_label"] == lab) & df["is_hard_case"]).sum()
        print(f"    {lab}: {n_c_h:,} / {n_c:,} hard ({100 * n_c_h / n_c:.1f}%)")
    
    # ── Unpooled embeddings (separate pass, small batch, streamed to HDF5) ───
    if args.save_unpooled:
        print("\n" + "=" * 65)
        print(f"Extracting test unpooled embeddings → HDF5 "
              f"(batch_size={args.unpooled_batch_size})")
        print("  Peak RAM = one batch only — streamed directly to disk")
        print("=" * 65)
        h5_path = output_dir / "orthrus_test_unpooled.h5"
        extract_unpooled_to_hdf5(
            model          = model,
            loader         = make_loader(test_dataset, args.unpooled_batch_size),
            device         = device,
            output_path    = h5_path,
            transcript_ids = test_out["transcript_ids"],
            lengths        = test_out["lengths"],
            desc           = "Unpooled",
        )

    print("\n" + "=" * 65)
    print("Done.")
    print("=" * 65)


if __name__ == "__main__":
    main()