"""compare_with_triframe.py — does triframe reproduce a released β-LNC run?

Reference = released β-LNC outputs (never modified):
  --probing_dir       1.4_latent_probing/  (per-fold redundancy / pattern CSVs; Layer 1)
  --ablation_reference 1.2_ablation/all_folds_ablation.csv      (1.2 ablation, Layer 1)
  --frechet_reference frechet_analysis/frechet_ranking.csv    (2.1 Fréchet, Layer 2)
  --reference         all_folds_patching.csv        (3.1 patching rows)
  --greedy_reference  joint_patching_greedy_all_folds.csv   (3.3 greedy search)
Any subset can be given; each part is reported and gated separately.

Probing (1.4, model-independent): per fold, the saved 1.1 representations
(--repr_dir fold_*_repr.npz) go through triframe.layer1 (representation_redundancy
with alpha=1, clip_per_dim, in-sample; pattern_alignment raw and with z regressed
out) and are compared with fold_*/redundancy.csv, pattern_raw.csv,
pattern_residual.csv. Permutation p-values are not covered (no null in triframe).

Ablation (1.2): triframe.layer1.run_ablation driven through an adapter with its own
token_zero (projector row) and feature_zero (raw columns) hooks, on each fold's
validation set; baseline and per-token acc_drop / f1_drop vs all_folds_ablation.csv.

Fréchet (Layer 2, model-independent): features/labels come from β-LNC's own
loader (analyze_frechet.load_features, config trainval files), so only the
computation is under test: triframe.layer2.frechet_by_token vs
analyze_frechet.frechet_analysis. Per sub-group frechet_dist, mean_term, cov_term
within rtol/atol; NaN must match NaN; same sub-groups; same ranking.

Patching: every (mode, scope, direction) group of the chosen folds is re-run
through triframe on the SAME base/partner transcripts, compared per pair
(logits) and in aggregate (IIA, mean|delta|, symmetry_score, causal_effect).

Greedy: β-LNC's pair selection (select_pairs, split_pairs) is reused; the
search itself is triframe.layer3.greedy_token_search driven by triframe's IIA.
Compared step by step: token order, IIA (selection + held-out), additive null,
symmetry. Columns built from logit_interaction_stats are not covered.

Not covered: noise-control values (random vectors differ by RNG).

Usage (repo root):
  python -m analysis.triframe.compare_with_triframe \
      --experiment_dir EXP --config CFG \
      --frechet_reference EXP/../frechet_analysis/frechet_ranking.csv \
      --reference EXP/interpretability/3.1_patching/all_folds_patching.csv \
      --greedy_reference EXP/interpretability/3.3_joint_search/joint_patching_greedy_all_folds.csv \
      --folds all --max_pairs 0 --batch_size 1 --atol_logit 1e-5 \
      --output_dir analysis/triframe/reports/g47
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from configs.load_config import load_config
from models.model_builder import create_model_builder
from data.cv_utils import get_cv_splits_from_config, load_sequences_in_order
from data.gated_feature_dataset import SequenceFeatureDataset
from data.feature_registry import REGISTRY

from triframe.layer1 import pattern_alignment, representation_redundancy, run_ablation
from triframe.layer2 import frechet_by_token
from triframe.layer3 import (check_self_patch_identity, greedy_token_search,
                             interchange_intervention_accuracy,
                             net_effect_symmetry, patch_and_forward,
                             patch_deltas, symmetry_score)

try:  # run as module or as a script
    from .blnc_triframe_adapter import BlncAdapter, batch_from_indices
except ImportError:
    from blnc_triframe_adapter import BlncAdapter, batch_from_indices

NEAR_THRESHOLD = 1e-4   # |prob - 0.5| below this: a decision flip is float noise


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------
def build_dataset(config):
    g = lambda *k, **kw: config.get(*k, **kw)
    return SequenceFeatureDataset(
        lnc_fasta=g("data", "lnc_fasta"), pc_fasta=g("data", "pc_fasta"),
        te_genomic_csv=g("data", "te_genomic_csv"),
        te_processed_csv=g("data", "te_processed_csv", default=None),
        nonb_genomic_csv=g("data", "nonb_genomic_csv"),
        nonb_processed_csv=g("data", "nonb_processed_csv", default=None),
        nonb2_csv=g("data", "nonb2_csv", default=None),
        te_scaler_bank_path=g("data", "te_scaler_bank_path"),
        nonb_scaler_bank_path=g("data", "nonb_scaler_bank_path"),
        nonb2_scaler_bank_path=g("data", "nonb2_scaler_bank_path", default=None),
        max_length=g("model", "max_length"))


def load_model(config, exp_dir, fold, device):
    ckpt = torch.load(Path(exp_dir) / "models" / f"fold_{fold}_best.pt",
                      map_location=device)
    model = create_model_builder(config)()
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    return model.to(device).eval()


def positions_for(level, scope, token_names, block_map):
    if level == "single":
        return [token_names.index(scope)]
    if level == "block":
        return [i for i, t in enumerate(token_names) if block_map[t] == scope]
    return list(range(len(token_names)))


def chunked(dataset, idx, size, has_nonb2):
    return [batch_from_indices(dataset, idx[s:s + size], has_nonb2)
            for s in range(0, len(idx), size)]


def make_fns(adapter):
    """Per-pair success / validity for a 2-class model (class 1 = mRNA)."""
    L = adapter.get_output_value
    succ = lambda b, p, e: L(p).argmax(-1) == e
    valid = lambda b, e: L(b).argmax(-1) != e
    return succ, valid


# --------------------------------------------------------------------------
# 1.4 probing equivalence (Layer 1, model-independent)
# --------------------------------------------------------------------------
def probing_equivalence(repr_dir, probing_dir, args):
    """One result dict per fold; tokens named as in the npz, compared to the CSVs."""
    results = []
    files = sorted(Path(repr_dir).glob("fold_*_repr.npz"))
    if not files:
        sys.exit(f"no fold_*_repr.npz in {repr_dir}")
    for f in files:
        fold_name = f.stem.replace("_repr", "")
        d = np.load(f, allow_pickle=True)
        z, T = d["z"].astype(np.float32), d["tokens"].astype(np.float32)
        y = d["labels"].astype(int)
        names = [str(t) for t in d["token_names"]] if "token_names" in d \
            else list(REGISTRY.all_subgroups)
        ref_dir = Path(probing_dir) / fold_name
        red, raw, res = (pd.read_csv(ref_dir / n).set_index("subgroup") for n in
                         ("redundancy.csv", "pattern_raw.csv", "pattern_residual.csv"))
        missing = [n for n in names if n not in red.index]
        gaps = {"r2": [], "align_raw": [], "align_resid": []}
        for i, sg in enumerate(names):
            if sg in missing:
                continue
            t = T[:, i, :]
            gaps["r2"].append(abs(red.at[sg, "r2_mean"] - representation_redundancy(
                t, z, test_frac=0, alpha=1.0, clip_per_dim=True)))
            gaps["align_raw"].append(abs(raw.at[sg, "alignment"] - pattern_alignment(t, y)))
            gaps["align_resid"].append(abs(res.at[sg, "alignment"] - pattern_alignment(
                t, y, controlling_for=z)))
        mx = {k: (max(v) if v else float("nan")) for k, v in gaps.items()}
        ok = not missing and all(v <= args.probing_atol for v in mx.values())
        results.append(dict(part="probing", fold=fold_name, passed=ok, n_tokens=len(names),
                            n_missing=len(missing), max_gap_r2=mx["r2"],
                            max_gap_align_raw=mx["align_raw"],
                            max_gap_align_resid=mx["align_resid"]))
        print(results[-1])
    return results


# --------------------------------------------------------------------------
# 1.2 ablation equivalence (Layer 1, needs the model)
# --------------------------------------------------------------------------
class Progress:
    """tqdm bar on a terminal; one log line per step otherwise (sbatch logs)."""

    def __init__(self, total, desc, unit="step"):
        self.total, self.desc, self.n, self.t0 = total, desc, 0, time.time()
        self.bar = None
        if sys.stderr.isatty():
            try:
                from tqdm.auto import tqdm
                self.bar = tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)
            except ImportError:
                pass

    def step(self, label=""):
        self.n += 1
        if self.bar is not None:
            self.bar.set_postfix_str(label)
            self.bar.update(1)
        else:
            el = time.time() - self.t0
            eta = el / self.n * (self.total - self.n)
            print(f"[{self.desc}] {self.n}/{self.total} {label}  "
                  f"elapsed {el / 60:.1f} min, eta {eta / 60:.1f} min", flush=True)

    def close(self):
        if self.bar is not None:
            self.bar.close()


class AblationAdapter(BlncAdapter):
    """run_forward takes {"chunks": [batch, ...], "zero_token": name | None}
    and returns the logits of all chunks; token_zero zeroes one projector row."""

    def run_forward(self, batch):
        zero = batch.get("zero_token")
        handles = []
        if zero is not None:
            block = REGISTRY.token_block(zero)
            row = REGISTRY.block_subgroups(block).index(zero)

            def hook(module, inp, out):
                out = out.clone()
                out[:, row, :] = 0.0
                return out
            handles.append(self.model.projectors[block].register_forward_hook(hook))
        try:
            logits = [self._forward(c)["logits"].cpu() for c in batch["chunks"]]  # streams
        finally:
            for h in handles:
                h.remove()
        return {"logits": torch.cat(logits)}


class Chunks:
    """Re-iterable stream of batches (DataLoader), with an optional per-batch transform."""

    def __init__(self, source, transform=None):
        self.source, self.transform = source, transform

    def __iter__(self):
        for b in self.source:
            yield self.transform(b) if self.transform else b

    def mapped(self, fn):
        prev = self.transform
        return Chunks(self.source, (lambda b: fn(prev(b))) if prev else fn)


def feature_zero(batch, token_list, registry):
    """Same stream with the token's raw feature columns set to 0 (per batch, lazily)."""
    (sg,) = token_list
    block = registry.token_block(sg)
    idx = registry.indices_for_subgroup(sg)
    keys = ([f"{block}_genomic", f"{block}_processed"]
            if registry.block_source(block) == "both" else [block])

    def zero(c):
        c = dict(c)
        for k in keys:
            if k in c:
                c[k] = c[k].clone()
                c[k][:, idx] = 0.0
        return c
    return {"chunks": batch["chunks"].mapped(zero), "zero_token": None}


def token_zero(batch, token_list, registry):
    return {"chunks": batch["chunks"], "zero_token": token_list[0]}


def ablation_equivalence(fold, ref, adapter, dataset, labels, val_idx, has_nonb2, args, out):
    from sklearn.metrics import f1_score
    y = np.array([0 if l == "lnc" else 1 for l in labels])[val_idx]
    from torch.utils.data import DataLoader, Subset
    loader = DataLoader(Subset(dataset, [int(i) for i in val_idx]),
                        batch_size=args.forward_batch_size, shuffle=False,
                        num_workers=args.num_workers)
    batch = {"chunks": Chunks(loader), "zero_token": None}
    f1s, accs = [], []

    names = list(REGISTRY.all_subgroups)
    prog = [None]

    def score_fn(logits, yy):  # acc is the score; acc and f1 recorded in call order
        pred = np.asarray(logits.argmax(-1))
        f1s.append(float(f1_score(yy, pred, average="binary", zero_division=0)))
        accs.append(float((pred == yy).mean()))
        k = len(accs) - 1   # 0 = baseline, then registry order
        prog[0].step("baseline" if k == 0 else names[k - 1])
        return accs[-1]

    rows, base_gap = [], 0.0
    for mode, zfn in (("token_zero", token_zero), ("feature_zero", feature_zero)):
        if mode not in args.ablation_modes.split(","):
            continue
        f1s.clear(); accs.clear()
        prog[0] = Progress(1 + len(names), f"1.2 fold {fold} {mode}", "pass")
        try:
            r = run_ablation(adapter, REGISTRY, [batch], [y], zfn, mode=mode, score_fn=score_fn)
        finally:
            prog[0].close()
        assert len(f1s) == 1 + len(names), "score_fn call order changed"
        refm = ref[(ref["mode"] == mode)].set_index("subgroup")
        base_gap = max(base_gap, abs(accs[0] - float(refm["baseline_acc"].iloc[0])),
                       abs(f1s[0] - float(refm["baseline_f1"].iloc[0])))
        for k, sg in enumerate(names):
            acc_d, f1_d = r.raw[sg][0], f1s[0] - f1s[k + 1]
            rows.append(dict(mode=mode, subgroup=sg,
                             acc_drop_ref=float(refm.at[sg, "acc_drop"]), acc_drop_tri=acc_d,
                             f1_drop_ref=float(refm.at[sg, "f1_drop"]), f1_drop_tri=f1_d))
    tab = pd.DataFrame(rows)
    tab["acc_gap"] = (tab["acc_drop_ref"] - tab["acc_drop_tri"]).abs()
    tab["f1_gap"] = (tab["f1_drop_ref"] - tab["f1_drop_tri"]).abs()
    tab.to_csv(out / "ablation_comparison.csv", index=False)
    worst = float(max(tab["acc_gap"].max(), tab["f1_gap"].max()))
    return dict(part="ablation", fold=fold, passed=worst <= args.ablation_atol
                and base_gap <= args.ablation_atol, n_rows=len(tab),
                max_gap_acc=float(tab["acc_gap"].max()), max_gap_f1=float(tab["f1_gap"].max()),
                max_gap_baseline=base_gap, n_nonzero_ref=int((tab["acc_drop_ref"] != 0).sum()))


# --------------------------------------------------------------------------
# 2.1 Frechet equivalence (Layer 2)
# --------------------------------------------------------------------------
FRECHET_COLS = ("frechet_dist", "mean_term", "cov_term")


def compare_frechet(ref, tri, rtol, atol):
    """(per-sub-group table, failures). ref has 'subgroup', tri has 'token'."""
    ref = ref.set_index("subgroup")
    tri = tri.rename(columns={"token": "subgroup"}).set_index("subgroup")
    fails = []
    only_ref, only_tri = set(ref.index) - set(tri.index), set(tri.index) - set(ref.index)
    if only_ref or only_tri:
        fails.append(f"sub-group sets differ: only reference {sorted(only_ref)}, "
                     f"only triframe {sorted(only_tri)}")
    rank = lambda df: df["frechet_dist"].rank(ascending=False, method="first")
    r_ref, r_tri = rank(ref), rank(tri)
    rows = []
    for sg in ref.index.intersection(tri.index):
        row = {"subgroup": sg}
        for c in FRECHET_COLS:
            a, b = float(ref.at[sg, c]), float(tri.at[sg, c])
            row[f"{c}_ref"], row[f"{c}_tri"] = a, b
            if np.isnan(a) != np.isnan(b):
                fails.append(f"{sg}.{c}: NaN mismatch (ref {a}, triframe {b})")
                row[f"{c}_absdiff"] = np.nan
                continue
            d = 0.0 if np.isnan(a) else abs(a - b)
            row[f"{c}_absdiff"] = d
            if d > atol + rtol * (0.0 if np.isnan(a) else abs(a)):
                fails.append(f"{sg}.{c}: |ref - triframe| = {d:.3e} (ref {a:.6g})")
        row["rank_ref"], row["rank_tri"] = r_ref[sg], r_tri[sg]
        rows.append(row)
    table = pd.DataFrame(rows)
    bad = table[(table["rank_ref"] != table["rank_tri"])
                & ~(table["rank_ref"].isna() & table["rank_tri"].isna())]
    if len(bad):
        fails.append("ranking differs: " + ", ".join(bad["subgroup"]))
    return table, fails


def frechet_equivalence(config, ref_csv, args, out):
    from analysis.pattern.analyze_frechet import load_features
    block_dfs, labels = load_features(config)
    block_data = {b: df.values.astype(np.float64) for b, df in block_dfs.items()}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tri = frechet_by_token(REGISTRY, block_data, labels, group_a=0, group_b=1,
                               standardize=True, reg=1e-4)
    for w in caught:
        print(f"[triframe warning] {w.message}", file=sys.stderr)
    table, fails = compare_frechet(pd.read_csv(ref_csv), tri,
                                   args.frechet_rtol, args.frechet_atol)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "frechet_comparison.csv", index=False)
    for f in fails:
        print(f"  FAIL {f}")
    return dict(part="frechet", passed=not fails, n_subgroups=len(table),
                n_fail=len(fails),
                max_absdiff=max(table[f"{c}_absdiff"].max() for c in FRECHET_COLS),
                n_lnc=int((labels == 0).sum()), n_mrna=int((labels == 1).sum()))


# --------------------------------------------------------------------------
# 3.1 patching equivalence
# --------------------------------------------------------------------------
def patching_equivalence(fold, ref, adapter, dataset, has_nonb2, token_names,
                         block_map, args, out):
    succ, valid = make_fns(adapter)
    L = adapter.get_output_value

    first = ref.iloc[0]
    check_self_patch_identity(
        adapter, batch_from_indices(dataset, [int(first["base_idx"])], has_nonb2),
        atol=args.atol_logit)

    sym_scopes = {"ALL" if x == "all" else x for x in args.aggregate_scopes.split(",") if x}
    per_rows, agg = [], []
    groups = ref.groupby(["patch_mode", "patch_scope", "scope_level", "direction"], sort=False)
    for (mode, scope, level, direction), g in groups:
        pos = positions_for(level, scope, token_names, block_map)
        base, part = g["base_idx"].to_numpy(), g["partner_dataset_idx"].to_numpy()
        tbs = chunked(dataset, base, args.batch_size, has_nonb2)
        sbs = chunked(dataset, part, args.batch_size, has_nonb2)
        b_l, p_l = [], []
        for tb, sb in zip(tbs, sbs):
            bo, po = patch_and_forward(adapter, tb, sb, pos)
            b_l.append(L(bo)[:, 1].cpu().numpy())
            p_l.append(L(po)[:, 1].cpu().numpy())
        t = g.assign(tf_baseline=np.concatenate(b_l), tf_patched=np.concatenate(p_l))
        t["d_base"] = (t["tf_baseline"] - t["baseline_logit"]).abs()
        t["d_patch"] = (t["tf_patched"] - t["patched_logit"]).abs()
        t["d_delta"] = ((t["tf_patched"] - t["tf_baseline"]) - t["delta_logit"]).abs()
        per_rows.append(t)

        # IIA through triframe's own function (all groups)
        cls = 0 if direction == "mrna_to_lnc" else 1
        iia_tf = interchange_intervention_accuracy(
            adapter, tbs, sbs, pos, [cls] * len(tbs), succ, valid_fn=valid)
        v = g[g["iia_baseline_valid"].astype(bool)]
        n_valid = len(v)
        iia_ref = float(v["iia_success"].astype(bool).mean()) if n_valid else float("nan")
        near = int((np.abs(v["patched_prob"] - 0.5) < NEAR_THRESHOLD).sum()) if n_valid else 0
        gap = abs(iia_ref - iia_tf) if n_valid else 0.0
        row = dict(patch_mode=mode, patch_scope=scope, direction=direction,
                   n_valid=n_valid, iia_ref=iia_ref, iia_triframe=iia_tf,
                   iia_gap=gap, near_threshold=near,
                   iia_explained=bool(gap * n_valid <= near + 1e-9))
        if scope in sym_scopes:
            row["mean_abs_delta_ref"] = float(g["delta_logit"].abs().mean())
            row["mean_abs_delta_triframe"] = symmetry_score(
                adapter, tbs, sbs, pos, value_fn=lambda o: L(o)[:, 1])
        agg.append(row)

    per_pair = pd.concat(per_rows, ignore_index=True)
    agg = pd.DataFrame(agg)

    net = []
    for (mode, scope), g in per_pair.groupby(["patch_mode", "patch_scope"]):
        a_, b_ = g[g["direction"] == "lnc_to_mrna"], g[g["direction"] == "mrna_to_lnc"]
        if a_.empty or b_.empty:
            continue
        ma, mb = a_["delta_logit"].mean(), b_["delta_logit"].mean()
        tf = net_effect_symmetry(a_["tf_patched"] - a_["tf_baseline"],
                                 b_["tf_patched"] - b_["tf_baseline"])
        net.append(dict(patch_mode=mode, patch_scope=scope,
                        symmetry_ref=(abs(ma) + abs(mb)) / 2,
                        symmetry_triframe=tf["symmetry_score"],
                        causal_ref=(ma - mb) / 2, causal_triframe=tf["causal_effect"]))
    net = pd.DataFrame(net)

    summary = (per_pair.groupby(["patch_mode", "patch_scope", "direction"])
               .agg(n=("pair_i", "count"), max_d_base=("d_base", "max"),
                    max_d_patch=("d_patch", "max"), max_d_delta=("d_delta", "max"))
               .reset_index())
    per_pair.to_csv(out / "equivalence_per_pair.csv", index=False)
    summary.to_csv(out / "equivalence_by_group.csv", index=False)
    agg.to_csv(out / "equivalence_aggregate.csv", index=False)
    net.to_csv(out / "equivalence_net_effect.csv", index=False)

    worst = float(summary[["max_d_base", "max_d_patch", "max_d_delta"]].max().max())
    floor = float(summary["max_d_base"].max())
    unexplained = int((~agg["iia_explained"]).sum())
    sym_gap = max(
        float((agg["mean_abs_delta_ref"] - agg["mean_abs_delta_triframe"]).abs().max())
        if "mean_abs_delta_ref" in agg and agg["mean_abs_delta_ref"].notna().any() else 0.0,
        float((net["symmetry_ref"] - net["symmetry_triframe"]).abs().max()) if len(net) else 0.0,
        float((net["causal_ref"] - net["causal_triframe"]).abs().max()) if len(net) else 0.0)
    ok = worst <= args.atol_logit and unexplained == 0 and sym_gap <= args.atol_logit
    return dict(part="patching", fold=fold, passed=ok, groups=len(summary),
                pairs=len(per_pair), max_dlogit=worst, noise_floor=floor,
                iia_groups=len(agg), iia_exact=int((agg["iia_gap"] == 0).sum()),
                iia_unexplained=unexplained, stat_gap=sym_gap)


# --------------------------------------------------------------------------
# 3.3 greedy equivalence
# --------------------------------------------------------------------------
def reconstruct_pairs(fold, val_idx, labels, adapter, dataset, has_nonb2, args):
    """β-LNC's own pair selection (select_pairs), fed by a metadata pass."""
    try:
        from analysis.pattern.patch_tokens import select_pairs
    except ImportError:
        from ..pattern.patch_tokens import select_pairs
    confs, lengths, gcs = [], [], []
    for s in range(0, len(val_idx), args.forward_batch_size):
        b = batch_from_indices(dataset, val_idx[s:s + args.forward_batch_size], has_nonb2)
        logits = adapter.get_output_value(adapter.run_forward(b)).float()
        confs.append(torch.softmax(logits, 1).max(1).values.cpu().numpy())
        seq = b["sequence"].float()
        lengths.append(seq[:, :4, :].sum(dim=(1, 2)).numpy().astype(np.float32))
        gcs.append(((seq[:, 1, :] + seq[:, 2, :]).sum(1)
                    / seq[:, :4, :].sum(dim=(1, 2)).clamp(min=1)).numpy().astype(np.float32))
    confs, lengths, gc = np.concatenate(confs), np.concatenate(lengths), np.concatenate(gcs)
    rng = np.random.default_rng(seed=42 + fold)
    lnc_l, mrna_l = select_pairs(
        val_indices=val_idx, labels=labels, lengths=lengths, confidences=confs,
        n_pairs=args.greedy_n_pairs, min_confidence=args.min_confidence,
        max_length_diff=args.max_length_diff, max_gc_diff=args.max_gc_diff,
        gc_content=gc, rng=rng)
    return val_idx[lnc_l], val_idx[mrna_l]


def greedy_equivalence(fold, ref_g, ref_p, adapter, dataset, labels, val_idx,
                       has_nonb2, token_names, args, out):
    try:
        from analysis.pattern.joint_patching_search import split_pairs
    except ImportError:
        from ..pattern.joint_patching_search import split_pairs

    heldout = float(ref_g["heldout_frac"].iloc[0])
    lnc_all, mrna_all = reconstruct_pairs(fold, val_idx, labels, adapter,
                                          dataset, has_nonb2, args)
    # consistency with the 3.1 reference: the 3.1 pairs are a prefix of these
    prefix_ok = None
    if ref_p is not None:
        r = ref_p[(ref_p["patch_mode"] == "all") & (ref_p["direction"] == "lnc_to_mrna")
                  ].sort_values("pair_i")
        k = min(len(r), len(lnc_all))
        prefix_ok = bool((r["lnc_dataset_idx"].to_numpy()[:k] == lnc_all[:k]).all()
                         and (r["mrna_dataset_idx"].to_numpy()[:k] == mrna_all[:k]).all())

    sel, ev = split_pairs(len(lnc_all), heldout, fold)
    succ, valid = make_fns(adapter)
    L = adapter.get_output_value
    size = args.forward_batch_size

    def half(pidx):
        lnc, mrna = lnc_all[pidx], mrna_all[pidx]
        return chunked(dataset, lnc, size, has_nonb2), chunked(dataset, mrna, size, has_nonb2)

    halves = {"sel": half(sel), "eval": half(ev)}

    def iia_set(h, pos):
        lc, mc = halves[h]
        a = interchange_intervention_accuracy(adapter, lc, mc, pos, [1] * len(lc), succ, valid_fn=valid)
        b = interchange_intervention_accuracy(adapter, mc, lc, pos, [0] * len(mc), succ, valid_fn=valid)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return float(np.nanmean([a, b]))

    def sym_set(h, pos):
        lc, mc = halves[h]
        vals = L

        def one(tbs, sbs, cls):
            ds = []
            for tb, sb in zip(tbs, sbs):
                d = patch_deltas(adapter, tb, sb, pos, value_fn=lambda o: vals(o)[:, 1])
                ok = (vals(adapter.run_forward(tb)).argmax(-1) != cls).cpu().numpy()
                ds.append(d[ok])
            ds = np.concatenate(ds)
            return ds if ds.size else np.array([0.0])
        return net_effect_symmetry(one(lc, mc, 1), one(mc, lc, 0))["symmetry_score"]

    def single(h):
        res = {}
        for sg in token_names:
            v = iia_set(h, [token_names.index(sg)])
            res[sg] = v if np.isfinite(v) else 0.0
        return res

    single_sel, single_eval = single("sel"), single("eval")
    rows = greedy_token_search(
        token_names, REGISTRY, single_sel, lambda pos: iia_set("sel", pos),
        args.max_search_depth, eval_iia_fn=lambda pos: iia_set("eval", pos),
        eval_single_token_iia=single_eval)
    tf = pd.DataFrame(rows)
    for i, r in tf.iterrows():
        pos = [token_names.index(t) for t in r["current_set"].split("+")]
        tf.loc[i, "symmetry_joint"] = sym_set("sel", pos)
        tf.loc[i, "symmetry_joint_heldout"] = sym_set("eval", pos)

    rg = ref_g.sort_values("step").reset_index(drop=True)
    n = min(len(rg), len(tf))
    same_len = len(rg) == len(tf)
    same_tokens = bool(same_len and (rg["added_subgroup"].to_numpy()
                                     == tf["added_token"].to_numpy()).all())
    cols = [("iia_joint", "iia_joint"), ("iia_joint_heldout", "iia_joint_heldout"),
            ("iia_additive_expected", "iia_additive_expected"),
            ("delta_vs_additive", "delta_vs_additive"),
            ("iia_additive_expected_heldout", "iia_additive_expected_heldout"),
            ("delta_vs_additive_heldout", "delta_vs_additive_heldout")]
    gaps = {rc: float(np.max(np.abs(rg[rc].to_numpy()[:n] - tf[tc].to_numpy()[:n]))) if n else np.nan
            for rc, tc in cols}
    sym_gap = max(float(np.max(np.abs(rg["symmetry_joint"].to_numpy()[:n] - tf["symmetry_joint"].to_numpy()[:n]))),
                  float(np.max(np.abs(rg["symmetry_joint_heldout"].to_numpy()[:n]
                                      - tf["symmetry_joint_heldout"].to_numpy()[:n])))) if n else np.nan
    cmp = rg[["step", "added_subgroup"]].head(n).copy()
    cmp["triframe_token"] = tf["added_token"].head(n).to_numpy()
    for rc, tc in cols:
        cmp[f"{rc}_ref"] = rg[rc].head(n).to_numpy()
        cmp[f"{rc}_triframe"] = tf[tc].head(n).to_numpy()
    cmp["symmetry_ref"] = rg["symmetry_joint"].head(n).to_numpy()
    cmp["symmetry_triframe"] = tf["symmetry_joint"].head(n).to_numpy()
    cmp.to_csv(out / "greedy_equivalence.csv", index=False)

    ok = (same_len and same_tokens and max(gaps.values()) <= args.atol_iia
          and sym_gap <= args.atol_greedy_sym and prefix_ok is not False)
    return dict(part="greedy", fold=fold, passed=bool(ok), steps_ref=len(rg),
                steps_triframe=len(tf), same_tokens=same_tokens,
                max_iia_gap=max(gaps.values()) if gaps else np.nan,
                sym_gap=sym_gap, pairs=len(lnc_all), n_sel=len(sel), n_eval=len(ev),
                pairs_prefix_matches_3_1=prefix_ok)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_dir", help="needed for --reference / --greedy_reference")
    ap.add_argument("--config", required=True)
    ap.add_argument("--probing_dir", help="1.4_latent_probing (needs --repr_dir)")
    ap.add_argument("--repr_dir", help="1.1_representations (fold_*_repr.npz)")
    ap.add_argument("--probing_atol", type=float, default=1e-5)
    ap.add_argument("--ablation_reference", help="1.2_ablation/all_folds_ablation.csv")
    ap.add_argument("--ablation_modes", default="token_zero,feature_zero")
    ap.add_argument("--num_workers", type=int, default=1, help="DataLoader workers (1.2)")
    ap.add_argument("--ablation_atol", type=float, default=1e-9,
                    help="accuracy/F1 are counts, so equality is expected")
    ap.add_argument("--frechet_reference", help="frechet_analysis/frechet_ranking.csv")
    ap.add_argument("--frechet_rtol", type=float, default=1e-6)
    ap.add_argument("--frechet_atol", type=float, default=1e-8)
    ap.add_argument("--reference", help="all_folds_patching.csv")
    ap.add_argument("--greedy_reference", help="joint_patching_greedy_all_folds.csv")
    ap.add_argument("--folds", default="all", help="'all' or comma list")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--modes", default="single,block,all,shuffled,sameclass")
    ap.add_argument("--max_pairs", type=int, default=0, help="pairs per group (0 = all)")
    ap.add_argument("--batch_size", type=int, default=1,
                    help="patching stage; 1 removes batch-vs-single float noise")
    ap.add_argument("--atol_logit", type=float, default=1e-5)
    ap.add_argument("--aggregate_scopes", default="all",
                    help="scopes also checked with triframe's symmetry_score")
    # greedy: must match the settings of the released joint search
    ap.add_argument("--greedy_n_pairs", type=int, default=1000)
    ap.add_argument("--min_confidence", type=float, default=0.7)
    ap.add_argument("--max_length_diff", type=float, default=0.2)
    ap.add_argument("--max_gc_diff", type=float, default=0.05)
    ap.add_argument("--max_search_depth", type=int, default=10)
    ap.add_argument("--forward_batch_size", type=int, default=128)
    ap.add_argument("--atol_iia", type=float, default=1e-9)
    ap.add_argument("--atol_greedy_sym", type=float, default=1e-4)
    args = ap.parse_args()
    needs_model = args.reference or args.greedy_reference or args.ablation_reference
    if not (needs_model or args.frechet_reference or args.probing_dir):
        sys.exit("give at least one of --probing_dir, --ablation_reference, "
                 "--frechet_reference, --reference, --greedy_reference")
    if needs_model and not args.experiment_dir:
        sys.exit("--experiment_dir is required for ablation / patching / greedy references")
    if bool(args.probing_dir) != bool(args.repr_dir):
        sys.exit("--probing_dir and --repr_dir go together")

    out_root = Path(args.output_dir); out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    config = load_config(args.config)
    results = []
    if args.frechet_reference:
        print("\n=== 2.1 Frechet ===")
        results.append(frechet_equivalence(config, args.frechet_reference, args,
                                           out_root / "frechet"))
        print(results[-1])
    if args.probing_dir:
        print("\n=== 1.4 probing ===")
        results += probing_equivalence(args.repr_dir, args.probing_dir, args)
    if not needs_model:
        return finish(results, [], args, out_root)
    dataset = build_dataset(config)
    has_nonb2 = config.get("data", "nonb2_csv", default=None) is not None
    token_names = list(REGISTRY.all_subgroups)
    block_map = {t: REGISTRY.token_block(t) for t in token_names}

    ref = ref_g = ref_a = None
    avail = None
    if args.ablation_reference:
        ref_a = pd.read_csv(args.ablation_reference)
        avail = set(ref_a["fold"].unique())
    if args.reference:
        ref = pd.read_csv(args.reference)
        if "patch_site" in ref.columns and set(ref["patch_site"].unique()) != {"projector"}:
            sys.exit("reference was not produced with patch_site='projector'")
        avail = set(ref["fold"].unique()) if avail is None else avail & set(ref["fold"].unique())
    if args.greedy_reference:
        ref_g = pd.read_csv(args.greedy_reference)
        if set(ref_g["patch_site"].unique()) != {"projector"}:
            sys.exit("greedy reference was not produced with patch_site='projector'")
        avail = set(ref_g["fold"].unique()) if avail is None else avail & set(ref_g["fold"].unique())
    folds = (sorted(avail) if args.folds == "all"
             else [int(x) for x in args.folds.split(",")])

    labels = splits = None
    if ref_g is not None or ref_a is not None:
        all_seq, labels = load_sequences_in_order(
            lnc_fasta=config.get("data", "lnc_fasta"), pc_fasta=config.get("data", "pc_fasta"))
        splits = get_cv_splits_from_config(config, all_seq, labels)

    for fold in folds:
        print(f"\n=== fold {fold} ===")
        out = out_root / f"fold_{fold}"; out.mkdir(exist_ok=True)
        model = load_model(config, args.experiment_dir, fold, device)
        adapter = BlncAdapter(model, REGISTRY, device, has_nonb2)

        if ref_a is not None:
            results.append(ablation_equivalence(
                fold, ref_a[ref_a["fold"] == fold], AblationAdapter(model, REGISTRY, device, has_nonb2),
                dataset, labels, splits[fold][1], has_nonb2, args, out))
            print(results[-1])
        if ref is not None:
            r = ref[(ref["fold"] == fold) & ref["patch_mode"].isin(args.modes.split(","))].copy()
            if args.max_pairs:
                r = r[r["pair_i"] < args.max_pairs]
            r["base_idx"] = np.where(r["direction"] == "lnc_to_mrna",
                                     r["lnc_dataset_idx"], r["mrna_dataset_idx"])
            results.append(patching_equivalence(
                fold, r, adapter, dataset, has_nonb2, token_names, block_map, args, out))
            print(results[-1])
        if ref_g is not None:
            rp = ref[ref["fold"] == fold] if ref is not None else None
            results.append(greedy_equivalence(
                fold, ref_g[ref_g["fold"] == fold], rp, adapter, dataset, labels,
                splits[fold][1], has_nonb2, token_names, args, out))
            print(results[-1])
        del model, adapter
        torch.cuda.empty_cache()

    finish(results, folds, args, out_root)


def finish(results, folds, args, out_root):
    res = pd.DataFrame(results)
    res.to_csv(out_root / "equivalence_summary.csv", index=False)
    passed = bool(res["passed"].all())
    lines = ["# triframe vs β-LNC — equivalence report",
             f"folds: {folds or 'n/a'}  max_pairs: {args.max_pairs or 'all'}  "
             f"batch_size: {args.batch_size}  atol_logit: {args.atol_logit:g}",
             f"OVERALL: {'PASS' if passed else 'FAIL'}"]
    for part, title in (("probing", "1.4 probing / pattern (Layer 1)"),
                        ("ablation", "1.2 ablation (Layer 1)"),
                        ("frechet", "2.1 Fréchet (Layer 2)"),
                        ("patching", "3.1 patching"), ("greedy", "3.3 greedy search")):
        sub = res[res["part"] == part].dropna(axis=1, how="all")
        if len(sub):
            lines += ["", f"## {title}", sub.to_string(index=False)]
    (out_root / "equivalence_report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()