"""
tests/test_null_modes.py — control modes in analysis/pattern/patch_tokens.py:
shuffled (now single/block/all scopes), sameclass, noise.
"""
import numpy as np
import pytest
import torch

from test_patch_site import (_make_model, ToyDataset, TOKEN_NAMES, BLOCK_MAP,
                             BLOCK_NAMES, DEV)
from analysis.pattern.patch_tokens import (
    patch_fold, build_summary, make_noise_vectors,
)

N = 40
LABELS = ["lnc" if i % 2 == 0 else "pc" for i in range(N)]


@pytest.fixture(scope="module")
def model():   return _make_model(seed=2)
@pytest.fixture(scope="module")
def dataset(): return ToyDataset(n=N, seed=5)


def _fold(model, dataset, modes, null_scopes=("single", "block", "all")):
    return patch_fold(
        model=model, dataset=dataset, val_indices=np.arange(N), labels=LABELS,
        device=DEV, has_nonb2=True, n_pairs=4, min_confidence=0.0,
        max_length_diff=10.0, max_gc_diff=10.0, fold_idx=0,
        token_names=TOKEN_NAMES, block_map=BLOCK_MAP, block_names=BLOCK_NAMES,
        patch_modes=modes, n_shuffled=3, patch_site="projector",
        null_scopes=null_scopes)


@pytest.fixture(scope="module")
def df(model, dataset):
    return _fold(model, dataset, ["single", "shuffled", "sameclass", "noise"])


@pytest.mark.parametrize("mode", ["shuffled", "sameclass", "noise"])
def test_controls_cover_single_block_all(df, mode):
    sub = df[df.patch_mode == mode]
    by_level = {lv: set(g.patch_scope) for lv, g in sub.groupby("scope_level")}
    assert by_level["single"] == set(TOKEN_NAMES)
    assert by_level["block"] == set(BLOCK_NAMES)
    assert by_level["all"] == {"ALL"}
    assert set(sub.direction) == {"lnc_to_mrna", "mrna_to_lnc"}
    assert np.isfinite(sub.delta_logit).all()


def test_matched_rows_scope_level_equals_patch_mode(df):
    m = df[df.patch_mode == "single"]
    assert (m.scope_level == "single").all()


def test_null_scopes_restricts_scopes(model, dataset):
    d = _fold(model, dataset, ["sameclass"], null_scopes=("all",))
    assert set(d.scope_level) == {"all"} and set(d.patch_scope) == {"ALL"}


def test_sameclass_partner_has_baseline_class_and_differs(df):
    sub = df[df.patch_mode == "sameclass"]
    base_col = np.where(sub.direction == "lnc_to_mrna",
                        sub.lnc_dataset_idx, sub.mrna_dataset_idx)
    expect = np.where(sub.direction == "lnc_to_mrna", "lnc", "pc")
    for b, p, e in zip(base_col, sub.partner_dataset_idx, expect):
        assert LABELS[int(b)] == e and LABELS[int(p)] == e
        assert int(b) != int(p)


def test_shuffled_partner_is_opposite_class(df):
    sub = df[df.patch_mode == "shuffled"]
    for _, r in sub.iterrows():
        if r.direction == "lnc_to_mrna":
            assert LABELS[int(r.lnc_dataset_idx)] == "lnc"
            assert LABELS[int(r.partner_dataset_idx)] == "pc"
        else:
            assert LABELS[int(r.mrna_dataset_idx)] == "pc"
            assert LABELS[int(r.partner_dataset_idx)] == "lnc"


def test_noise_rows_pair_with_shuffled_rows(df):
    key = ["patch_scope", "direction", "pair_i"]
    a = df[df.patch_mode == "shuffled"].set_index(key).sort_index()
    b = df[df.patch_mode == "noise"].set_index(key).sort_index()
    assert a.index.equals(b.index)
    assert np.allclose(a.baseline_logit, b.baseline_logit)
    assert not np.allclose(a.patched_logit, b.patched_logit)


def test_make_noise_vectors_norm_direction_determinism():
    src = torch.randn(5, 16) * torch.tensor([[1.], [2.], [3.], [4.], [5.]])
    g1 = torch.Generator().manual_seed(1); g2 = torch.Generator().manual_seed(1)
    a, b = make_noise_vectors(src, g1), make_noise_vectors(src, g2)
    assert a.shape == src.shape and a.dtype == src.dtype
    assert torch.equal(a, b)
    assert torch.allclose(a.norm(dim=-1), src.norm(dim=-1), atol=1e-5)
    cos = torch.nn.functional.cosine_similarity(a, src, dim=-1)
    assert (cos.abs() < 0.95).all()


def test_build_summary_includes_control_modes(df):
    s = build_summary(df)
    for mode in ("shuffled", "sameclass", "noise"):
        sub = s[s.patch_mode == mode]
        assert len(sub) > 0 and "iia_overall" in sub.columns
        assert {"ALL"} <= set(sub.patch_scope)