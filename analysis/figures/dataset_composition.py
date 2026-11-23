"""Dataset composition table: transcripts, genes and length by split and class."""
import json
from pathlib import Path

import numpy as np

from . import tables

CLASSES = (("lnc", "lncRNA"), ("pc", "mRNA"))
SPLITS = (("trainval", "Training and validation"), ("test", "Test"))
HEADER = ["Split", "Class", "Transcripts", "Genes", "Median length (nt)", "5th to 95th percentile (nt)"]


def read_fasta(path):
    """[(gene, length)] per record: gene = versionless ENSG of header field 2 (as data.cv_utils), length from the sequence."""
    if not Path(path).exists():
        raise FileNotFoundError(path)
    recs, gene, n = [], None, 0
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if gene is not None:
                    recs.append((gene, n))
                fields = line[1:].rstrip().split("|")
                if len(fields) < 2 or not fields[1].startswith("ENSG"):
                    raise ValueError(f"{path}: header without a gene field: {line.strip()[:80]}")
                gene, n = fields[1].split(".")[0], 0
            else:
                n += len(line.strip())
    if gene is not None:
        recs.append((gene, n))
    return recs


def _row(split, cls, recs):
    lengths = np.array([n for _, n in recs])
    return [split, cls, f"{len(recs):,}", f"{len({g for g, _ in recs}):,}", f"{np.median(lengths):,.0f}",
            f"{np.percentile(lengths, 5):,.0f} to {np.percentile(lengths, 95):,.0f}"]


def build(split_dir, out_dir):
    """Write dataset_composition.csv and .tex from the split FASTAs; return (inputs, outputs).

    Stops if a file is missing, a record count differs from split_manifest.json, or a gene is in both splits."""
    d = Path(split_dir)
    manifest_path = d / "split_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    inputs, rows, genes = [manifest_path], [], {}
    for split, split_label in SPLITS:
        pooled = []
        for cls, cls_label in CLASSES:
            path = d / manifest[split][f"{cls}_file"]
            inputs.append(path)
            recs = read_fasta(path)
            if len(recs) != manifest[split][cls]:
                raise ValueError(f"{path}: {len(recs)} records, split_manifest.json says {manifest[split][cls]}")
            pooled += recs
            rows.append(_row(split_label, cls_label, recs))
        rows.append(_row(split_label, "All", pooled))
        genes[split] = {g for g, _ in pooled}
    shared = genes["test"] & genes["trainval"]
    if shared:
        raise ValueError(f"{len(shared)} genes occur in both the test and the training/validation FASTAs")
    return inputs, tables.write(out_dir, "dataset_composition", HEADER, rows)