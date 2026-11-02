"""
tests/test_joint_stats.py — held-out split, output-space interaction,
Benjamini-Hochberg, stopping rule and pairwise search in
analysis/pattern/joint_patching_search.py.
"""
import numpy as np
import pandas as pd
import pytest

from test_patch_site import (_make_model, ToyDataset, TOKEN_NAMES, BLOCK_MAP,
                             BLOCK_NAMES, DEV)
from analysis.pattern import joint_patching_search as J


def _toy_splits(n, n_folds=2):
    """Deterministic (train_idx, val_idx) per fold; both classes in every val set."""
    idx = np.arange(n)
    return [(idx[(idx // 2) % n_folds != k], idx[(idx // 2) % n_folds == k])
            for k in range(n_folds)]

# ── pure statistics ─────────────────────────────────────────────────────────

def test_benjamini_hochberg_hand_values():
    q = J.benjamini_hochberg([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(q, [0.02, 0.04, 0.04, 0.02])


def test_benjamini_hochberg_nan_and_empty():
    q = J.benjamini_hochberg([0.01, np.nan, 0.04])
    assert np.isnan(q[1]) and np.allclose(q[[0, 2]], [0.02, 0.04])
    assert J.benjamini_hochberg([]).size == 0


def test_split_pairs_disjoint_complete_deterministic():
    sel, ev = J.split_pairs(100, 0.5, fold=3)
    assert set(sel).isdisjoint(ev) and len(sel) == len(ev) == 50
    assert sorted(np.concatenate([sel, ev])) == list(range(100))
    sel2, ev2 = J.split_pairs(100, 0.5, fold=3)
    assert np.array_equal(sel, sel2) and np.array_equal(ev, ev2)
    assert not np.array_equal(ev, J.split_pairs(100, 0.5, fold=4)[1])


def test_split_pairs_zero_frac_shares_pairs():
    sel, ev = J.split_pairs(20, 0.0, fold=0)
    assert np.array_equal(sel, ev) and len(sel) == 20


def _arrays(base_l, patched_l, base_m, patched_m):
    base_l, patched_l = np.asarray(base_l, float), np.asarray(patched_l, float)
    base_m, patched_m = np.asarray(base_m, float), np.asarray(patched_m, float)
    return {"diff_base_l2m": base_l, "diff_patched_l2m": patched_l,
            "diff_base_m2l": base_m, "diff_patched_m2l": patched_m,
            "valid_l2m": base_l <= 0, "valid_m2l": base_m > 0}


def test_logit_interaction_zero_for_additive_shifts():
    base_l, base_m = [-1.0, -2.0, -0.5], [1.0, 2.0, 0.5]
    a = _arrays(base_l, np.add(base_l, [0.6, 0.4, 0.2]), base_m, np.subtract(base_m, [0.3, 0.2, 0.1]))
    b = _arrays(base_l, np.add(base_l, [0.5, 0.7, 0.6]), base_m, np.subtract(base_m, [0.4, 0.1, 0.2]))
    joint = _arrays(base_l, np.add(base_l, [1.1, 1.1, 0.8]), base_m,
                    np.subtract(base_m, [0.7, 0.3, 0.3]))
    st = J.logit_interaction_stats(joint, [a, b])
    assert abs(st["logit_interaction"]) < 1e-9
    # additive prediction crosses 0 for pairs 0 and 1 (-1+1.1>0, -2+1.1<0)
    assert 0.0 <= st["iia_logit_additive"] <= 1.0


def test_logit_interaction_sign_positive_for_synergy_both_directions():
    base_l, base_m = [-1.0, -1.0], [1.0, 1.0]
    a = _arrays(base_l, np.add(base_l, 0.2), base_m, np.subtract(base_m, 0.2))
    b = _arrays(base_l, np.add(base_l, 0.2), base_m, np.subtract(base_m, 0.2))
    joint = _arrays(base_l, np.add(base_l, 1.5), base_m, np.subtract(base_m, 1.5))
    st = J.logit_interaction_stats(joint, [a, b])
    assert st["logit_interaction"] == pytest.approx(1.1)
    # additive prediction: -1+0.4 < 0 -> no flip; joint flips -> excess over null
    assert st["iia_logit_additive"] == 0.0


def test_logit_interaction_negative_for_redundancy():
    base_l, base_m = [-1.0], [1.0]
    a = _arrays(base_l, np.add(base_l, 0.8), base_m, np.subtract(base_m, 0.8))
    b = _arrays(base_l, np.add(base_l, 0.8), base_m, np.subtract(base_m, 0.8))
    joint = _arrays(base_l, np.add(base_l, 0.9), base_m, np.subtract(base_m, 0.9))
    assert J.logit_interaction_stats(joint, [a, b])["logit_interaction"] < 0


def test_invalid_baselines_are_excluded():
    # first l2m item already on the mRNA side -> invalid, must not count
    base_l = [0.5, -1.0]
    a = _arrays(base_l, np.add(base_l, 0.1), [1.0, 1.0], [0.9, 0.9])
    joint = _arrays(base_l, np.add(base_l, 5.0), [1.0, 1.0], [0.9, 0.9])
    st = J.logit_interaction_stats(joint, [a])
    assert st["n_units"] == 1 + 2


# ── stopping rule and held-out wiring ───────────────────────────────────────

def _stub_patch_fn(iia_by_size):
    def fn(subgroup_set):
        return {"iia": iia_by_size[len(subgroup_set)], "symmetry_score": 0.0}
    return fn


def test_greedy_continues_on_plateau_and_stops_after_strict_decrease():
    fn = _stub_patch_fn({1: 0.2, 2: 0.2, 3: 0.1, 4: 0.5})
    df = J.greedy_forward_search(list("abcde"), {s: 0.0 for s in "abcde"}, 5, fn)
    assert list(df.step) == [1, 2, 3]          # plateau at 2 continues; drop at 3 stops


def test_greedy_requires_all_eval_args_together():
    fn = _stub_patch_fn({1: 0.1, 2: 0.2})
    with pytest.raises(ValueError):
        J.greedy_forward_search(list("ab"), {"a": 0.0, "b": 0.0}, 2, fn,
                                eval_patch_fn=fn)


# ── on the toy model ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def setup():
    model = _make_model(seed=4)
    dataset = ToyDataset(n=20, seed=8)
    lnc = np.arange(0, 20, 2)[:6]
    mrna = np.arange(1, 20, 2)[:6]
    idx = sorted(set(lnc.tolist()) | set(mrna.tolist()))
    tm = J.cache_tokens_batched(model, dataset, idx, DEV, True, batch_size=4,
                                patch_site="projector")
    layout = J.build_token_layout(model, TOKEN_NAMES, BLOCK_MAP, BLOCK_NAMES)

    def fn(sgs, **kw):
        return J.run_joint_patch(model, dataset, lnc, mrna, tm, sgs, TOKEN_NAMES,
                                 DEV, True, chunk_size=4, layout=layout, **kw)
    return fn, (model, dataset, lnc, mrna)


def test_run_joint_patch_arrays_and_reused_baseline(setup):
    fn, (model, dataset, lnc, mrna) = setup
    plain = fn(["GQ", "STR"])
    base = J.compute_baselines(model, dataset, lnc, mrna, DEV, True, 4)
    with_arr = fn(["GQ", "STR"], baseline=base, return_arrays=True)
    for k in ("iia", "symmetry_score", "iia_l2m", "iia_m2l"):
        assert plain[k] == pytest.approx(with_arr[k], nan_ok=True)
    assert with_arr["diff_patched_l2m"].shape == (len(lnc),)
    assert with_arr["valid_m2l"].dtype == bool


def test_single_member_interaction_is_zero(setup):
    fn, _ = setup
    r = fn(["GQ"], return_arrays=True)
    st = J.logit_interaction_stats(r, [r])
    assert abs(st["logit_interaction"]) < 1e-9
    assert st["iia_logit_additive"] == pytest.approx(r["iia"], nan_ok=True)


def test_pairwise_search_on_toy_model(setup):
    fn, _ = setup
    subs = ["GQ", "STR", "IR", "TE_CORE"]
    single = {s: fn([s], return_arrays=True) for s in subs}
    iia = {s: (0.0 if np.isnan(r["iia"]) else r["iia"]) for s, r in single.items()}
    df = J.pairwise_interaction_search(
        subs, lambda sg: fn(sg, return_arrays=True), iia, single)
    assert len(df) == 6
    assert df.p.between(0, 1).all() and (df.q >= df.p - 1e-12).all()
    assert df.q.is_monotonic_increasing
    assert {"logit_interaction_heldout", "iia_logit_additive_heldout"} <= set(df.columns)


def test_summarize_pairwise_folds():
    rows = []
    for fold in range(3):
        rows.append({"token_a": "A", "token_b": "B", "fold": fold, "p": 1e-4,
                     "q": 0.01, "logit_interaction_heldout": 0.5,
                     "delta_vs_logit_additive_heldout": 0.1})
        rows.append({"token_a": "A", "token_b": "C", "fold": fold, "p": 0.6,
                     "q": 0.9, "logit_interaction_heldout": 0.0,
                     "delta_vs_logit_additive_heldout": 0.0})
    out = J.summarize_pairwise_folds(pd.DataFrame(rows))
    top = out.iloc[0]
    assert (top.token_a, top.token_b) == ("A", "B")
    assert top["n_folds_q_lt_0.05"] == 3 and top.fisher_q < 0.05
    assert out.iloc[1].fisher_q > 0.05


# ── run_one_fold wiring ─────────────────────────────────────────────────────

def test_run_one_fold_end_to_end(tmp_path, monkeypatch):
    import torch
    from types import SimpleNamespace

    N = 60
    dataset = ToyDataset(n=N, seed=1)
    labels = ["lnc" if i % 2 == 0 else "pc" for i in range(N)]
    (tmp_path / "models").mkdir()
    torch.save({"model_state_dict": _make_model(seed=6).state_dict(),
                "epoch": 1, "val_acc": 0.9}, tmp_path / "models" / "fold_0_best.pt")
    monkeypatch.setattr(J, "create_model_builder", lambda cfg: (lambda: _make_model(seed=99)))
    monkeypatch.setattr(J, "load_config", lambda path: {})
    args = SimpleNamespace(
        config="unused", forward_batch_size=8, n_pairs=12, min_confidence=0.0,
        max_length_diff=10.0, max_gc_diff=10.0, patch_site="projector",
        heldout_frac=0.5, max_search_depth=3, mode="all", topk=3,
        output_dir=str(tmp_path))

    greedy, topk, pairs, ctrl = J.run_one_fold(
        fold=0, exp_dir=tmp_path, dataset=dataset, all_sequences=list(range(N)),
        labels=labels, splits=_toy_splits(N, 2), n_folds_config=2,
        random_state=0, has_nonb2=True, device=DEV, args=args,
        all_subgroups=TOKEN_NAMES)

    assert 1 <= len(greedy) <= 3
    for col in ("iia_joint_heldout", "iia_additive_expected_heldout",
                "iia_logit_additive_heldout", "logit_interaction_heldout",
                "heldout_frac", "patch_site", "fold"):
        assert col in greedy.columns
    first = greedy.iloc[0]
    if np.isfinite(first.iia_joint):          # null and joint share pairs
        assert first.delta_vs_additive == pytest.approx(0.0, abs=1e-12)
    if np.isfinite(first.iia_joint_heldout):
        assert first.delta_vs_additive_heldout == pytest.approx(0.0, abs=1e-12)
    assert first.logit_interaction_heldout == pytest.approx(0.0, abs=1e-6)

    assert topk.empty                      # mode="all" no longer runs top-k
    assert len(pairs) == 190 and pairs.q.between(0, 1).all()
    assert ctrl.empty
    for name in ("greedy", "pairs"):
        assert (tmp_path / f"joint_patching_{name}_fold0.csv").exists()
    assert not (tmp_path / "joint_patching_topk_fold0.csv").exists()
    _, tk, _, _ = J.run_one_fold(fold=0, exp_dir=tmp_path, dataset=dataset,
        all_sequences=list(range(N)), labels=labels,
        splits=_toy_splits(N, 2), n_folds_config=2, random_state=0,
        has_nonb2=True, device=DEV, all_subgroups=TOKEN_NAMES,
        args=SimpleNamespace(**{**vars(args), "mode": "topk"}))
    assert len(tk) == 4 and "delta_vs_logit_additive_heldout" in tk.columns
    bal = pd.read_csv(tmp_path / "pair_balance_fold0.csv")
    assert set(bal.group) == {"matched", "random_opposite", "all_highconf_unmatched"}


# ── joint-set controls and pair balance ─────────────────────────────────────

def test_pair_balance_matched_beats_random():
    rng = np.random.default_rng(0)
    n = 200
    lengths = np.concatenate([rng.uniform(1000, 5000, n), rng.uniform(1000, 5000, n)])
    lengths[n:] = lengths[:n] * rng.uniform(0.95, 1.05, n)      # mRNA ~ matched lnc
    gc = np.concatenate([rng.uniform(0.3, 0.6, n), rng.uniform(0.3, 0.6, n)])
    gc[n:] = gc[:n] + rng.uniform(-0.01, 0.01, n)
    conf = np.full(2 * n, 0.9)
    labels = np.array(["lnc"] * n + ["pc"] * n)
    out = J.pair_balance_table(lengths, gc, conf, labels, np.arange(n),
                               np.arange(n, 2 * n), 0.7, np.random.default_rng(1)
                               ).set_index("group")
    assert out.loc["matched", "mean_abs_rel_length_diff"] < 0.06
    assert (out.loc["random_opposite", "mean_abs_rel_length_diff"]
            > 5 * out.loc["matched", "mean_abs_rel_length_diff"])
    assert out.loc["matched", "mean_abs_gc_diff"] < 0.011
    assert out.loc["all_highconf_unmatched", "n_lnc"] == 200


def test_smd_known_value():
    a, b = np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0])
    assert J._smd(a, b) == pytest.approx(-1.0)
    assert np.isnan(J._smd(a[:1], b))


def test_run_joint_patch_vecs_override_equals_matched():
    import torch
    from types import SimpleNamespace  # noqa: F401
    N = 24
    dataset = ToyDataset(n=N, seed=2)
    model = _make_model(seed=3).to(DEV).eval()
    layout = J.build_token_layout(model, TOKEN_NAMES, BLOCK_MAP, BLOCK_NAMES)
    lnc, mrna = np.arange(0, 8), np.arange(8, 16)
    cache = J.cache_tokens_batched(model, dataset, list(range(16)), DEV, True,
                                   batch_size=8, patch_site="projector")
    kw = dict(token_names=TOKEN_NAMES, device=DEV, has_nonb2=True, chunk_size=8,
              patch_site="projector", layout=layout)
    members = TOKEN_NAMES[:2]
    pos = [TOKEN_NAMES.index(m) for m in members]
    ref = J.run_joint_patch(model, dataset, lnc, mrna, cache, members, **kw)
    ov = {"l2m": torch.stack([cache[int(i)][pos] for i in mrna]),
          "m2l": torch.stack([cache[int(i)][pos] for i in lnc])}
    got = J.run_joint_patch(model, dataset, lnc, mrna, cache, members,
                            vecs_override=ov, **kw)
    assert got["symmetry_score"] == pytest.approx(ref["symmetry_score"], abs=1e-6)


def test_control_vectors_norm_and_sameclass():
    import torch
    d = 6
    cache = {i: torch.randn(5, d) for i in range(12)}
    lnc, mrna = np.arange(0, 6), np.arange(6, 12)
    pos = [1, 3]
    gen, rng = torch.Generator().manual_seed(0), np.random.default_rng(0)
    nz = J.control_patch_vectors("noise", cache, lnc, mrna, pos, gen, rng)
    partner = torch.stack([cache[int(i)][pos] for i in mrna])
    assert torch.allclose(nz["l2m"].norm(dim=-1), partner.norm(dim=-1), atol=1e-5)
    assert not torch.allclose(nz["l2m"], partner)
    sc = J.control_patch_vectors("sameclass", cache, lnc, mrna, pos, gen, rng)
    own = torch.stack([cache[int(i)][pos] for i in lnc])
    assert not torch.allclose(sc["l2m"], own)
    src = {tuple(v.flatten().tolist()) for v in sc["l2m"]}
    lnc_vecs = {tuple(cache[int(i)][pos].flatten().tolist()) for i in lnc}
    assert src <= lnc_vecs                      # drawn from the baseline's own class
    with pytest.raises(ValueError):
        J.control_patch_vectors("bogus", cache, lnc, mrna, pos, gen, rng)


def test_controls_mode_end_to_end(tmp_path, monkeypatch):
    import torch
    from types import SimpleNamespace
    N = 60
    dataset = ToyDataset(n=N, seed=1)
    labels = ["lnc" if i % 2 == 0 else "pc" for i in range(N)]
    (tmp_path / "models").mkdir()
    torch.save({"model_state_dict": _make_model(seed=6).state_dict(),
                "epoch": 1, "val_acc": 0.9}, tmp_path / "models" / "fold_0_best.pt")
    monkeypatch.setattr(J, "create_model_builder", lambda cfg: (lambda: _make_model(seed=99)))
    monkeypatch.setattr(J, "load_config", lambda path: {})
    common = dict(config="unused", forward_batch_size=8, n_pairs=12, min_confidence=0.0,
                  max_length_diff=10.0, max_gc_diff=10.0, patch_site="projector",
                  heldout_frac=0.5, max_search_depth=3, topk=3,
                  output_dir=str(tmp_path))
    kw = dict(fold=0, exp_dir=tmp_path, dataset=dataset, all_sequences=list(range(N)),
              labels=labels, splits=_toy_splits(N, 2), n_folds_config=2,
              random_state=0, has_nonb2=True, device=DEV, all_subgroups=TOKEN_NAMES)
    greedy, *_ = J.run_one_fold(args=SimpleNamespace(mode="greedy", **common), **kw)
    gpath = tmp_path / "greedy_all.csv"
    greedy.to_csv(gpath, index=False)

    args = SimpleNamespace(mode="controls", greedy_csv=str(gpath),
                           control_scope="both", n_control_draws=2, **common)
    g, t, p, ctrl = J.run_one_fold(args=args, **kw)
    assert g.empty and t.empty and p.empty
    assert (ctrl.kind == "greedy_step").sum() == len(greedy)
    assert (ctrl.kind == "pair").sum() == 190
    for col in ("iia_matched", "iia_noise_mean", "iia_sameclass_mean",
                "excess_iia_over_noise", "symmetry_noise_mean", "n_draws"):
        assert col in ctrl.columns
    # same held-out pairs and set as the greedy run -> same matched IIA
    step1 = ctrl[(ctrl.kind == "greedy_step") & (ctrl.step == 1)].iloc[0]
    gs1 = greedy[greedy.step == 1].iloc[0]
    if np.isfinite(gs1.iia_joint_heldout):
        assert step1.iia_matched == pytest.approx(gs1.iia_joint_heldout, abs=1e-9)
    assert (tmp_path / "joint_patching_controls_fold0.csv").exists()

    args.control_scope = "greedy"
    args.greedy_csv = None
    with pytest.raises(ValueError):
        J.run_one_fold(args=args, **kw)