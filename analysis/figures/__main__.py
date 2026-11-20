"""Render the paper figures and tables, and record their inputs in figures_manifest.json.

python -m analysis.figures --release 49 --exp_dir EXP --benchmark_dir B1 --out figures/g49 [--only NAME ...]
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from . import benchmark_supplement, benchmark_table, dataset_composition, frechet, patching, ablations, interpretability_main, style  # noqa: E402
ROOT = Path(__file__).resolve().parents[2]


def _need(args, *names):
    missing = [f"--{n}" for n in names if getattr(args, n) is None]
    if missing:
        raise ValueError(f"needs {' '.join(missing)}")


def _benchmark_table(a):
    _need(a, "benchmark_dir")
    b1, cut = Path(a.benchmark_dir), a.unseen_cutoff
    return benchmark_table.build(b1, b1.parent / f"benchmark_unseen_v{cut}",
                                 b1.parent / f"benchmark_seen_v{cut}", a.out, cut)


def _benchmark_supplement(a):
    _need(a, "benchmark_dir")
    b1, cut = Path(a.benchmark_dir), a.unseen_cutoff
    return benchmark_supplement.build(b1, b1.parent / f"benchmark_unseen_v{cut}",
                                      b1.parent / f"benchmark_seen_v{cut}",
                                      b1.parent / "unseen_subset", a.out, cut)

def _dataset_composition(a):
    return dataset_composition.build(a.split_dir or f"data/split_gencode_{a.release}", a.out)

def _frechet(a):
    _need(a, "exp_dir")
    return frechet.build(a.exp_dir, a.out)

def _patching(a):
    _need(a, "exp_dir")
    return patching.build(a.exp_dir, a.out)

def _ablations(a):
    _need(a, "exp_dir")
    return ablations.build(a.exp_dir, a.out)

def _interpretability_main(a):
    _need(a, "exp_dir")
    return interpretability_main.build(a.exp_dir, a.out)



# name -> function(args) returning (input paths, output paths)
FIGURES = {"benchmark_table": _benchmark_table, "benchmark_supplement": _benchmark_supplement,
           "dataset_composition": _dataset_composition, "frechet": _frechet, 
           "patching": _patching, "ablations": _ablations, "interpretability_main": _interpretability_main}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state():
    """(HEAD commit, whether tracked files differ from it); (None, None) outside git."""
    def run(*cmd):
        return subprocess.run(["git", "-C", str(ROOT), *cmd], capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        return run("rev-parse", "HEAD"), bool(run("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        return None, None


def record(out, release, name, inputs, outputs):
    """Add one figure (input sha256, outputs, commit, dirty flag) to out/figures_manifest.json."""
    path = Path(out) / "figures_manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {"figures": {}}
    commit, dirty = git_state()
    manifest.update(release=release, font=style.resolved_font(), matplotlib=matplotlib.__version__)
    manifest["figures"][name] = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit, "git_dirty": dirty,
        "inputs": {str(p): sha256(p) for p in inputs},
        "outputs": [str(p) for p in outputs],
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m analysis.figures", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--release", type=int, required=True)
    p.add_argument("--exp_dir", help="beta_vae_subgroup_base_gNN folder (interpretability figures)")
    p.add_argument("--benchmark_dir",
                   help="full-test benchmark folder; the unseen and seen folders are its siblings")
    p.add_argument("--split_dir", help="split FASTA folder (default: data/split_gencode_<release>)")
    p.add_argument("--unseen_cutoff", type=int, default=46)
    p.add_argument("--out", required=True)
    p.add_argument("--only", nargs="+", choices=sorted(FIGURES), help="default: all figures")
    a = p.parse_args(argv)
    for name in a.only or FIGURES:
        try:
            inputs, outputs = FIGURES[name](a)
        except (FileNotFoundError, ValueError) as e:
            sys.exit(f"figure {name}: {e}")
        record(a.out, a.release, name, inputs, outputs)
        print(f"{name}: {len(outputs)} files in {a.out}")


if __name__ == "__main__":
    main()