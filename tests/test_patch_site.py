"""
tests/test_patch_site.py

Regression tests for the Layer-3 patch SITE in analysis/pattern/patch_tokens.py
(and the batched copy in joint_patching_search.py).

Background: BetaVAESubgroup.encode() builds each block's raw tokens
(self._block_tokens), concatenates them, applies token_norm, and runs
cross-attention on the result — but ALSO feeds the classifier one
attention-weighted summary per block built from the RAW _block_tokens.
Hooking model.token_norm therefore patches only one of two downstream paths.
These tests pin the intended behaviour of patch_site="projector" and
document what patch_site="token_norm" (legacy) does NOT do.

Runs on CPU with a small randomly initialised model; no checkpoint or real
data needed.
"""
import numpy as np
import pytest
import torch

from data.feature_registry import REGISTRY
from models.beta_vae_subgroup import BetaVAESubgroup
from analysis.pattern.patch_tokens import (
    ProjectorTokenCache, TokenCache, build_token_layout, build_fwd_kwargs,
    get_tokens_single, get_logits_single, get_patched_logits,
    self_patch_sanity_check, patch_fold,
)

INPUT_DIM, INPUT_LENGTH, D_PROJ = 5, 1024, 16
TOKEN_NAMES = list(REGISTRY.all_subgroups)
BLOCK_MAP   = {sg: REGISTRY.token_block(sg) for sg in TOKEN_NAMES}
BLOCK_NAMES = list(REGISTRY.block_names)


def _make_model(seed=0):
    torch.manual_seed(seed)
    m = BetaVAESubgroup(
        num_classes=2, input_dim=INPUT_DIM, input_length=INPUT_LENGTH,
        latent_dim=16, cse_d_model=32, d_proj=D_PROJ, attn_heads=4,
        classifier_hidden=[16],
    )
    # Random init leaves BatchNorm running stats at identity; perturb the
    # classifier so the summary path carries real weight.
    with torch.no_grad():
        for p in m.classifier.parameters():
            p.add_(0.3 * torch.randn_like(p))
    return m.eval()


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, n=48, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.items = []
        for i in range(n):
            seq = torch.zeros(INPUT_DIM, INPUT_LENGTH)
            seq[torch.randint(0, 4, (INPUT_LENGTH,), generator=g),
                torch.arange(INPUT_LENGTH)] = 1.0
            self.items.append({
                "sequence":       seq,
                "te_genomic":     torch.randn(REGISTRY.te_dim, generator=g),
                "te_processed":   torch.randn(REGISTRY.te_dim, generator=g),
                "nonb_genomic":   torch.randn(REGISTRY.nonb_dim, generator=g),
                "nonb_processed": torch.randn(REGISTRY.nonb_dim, generator=g),
                "nonb2":          torch.randn(REGISTRY.nonb2_dim, generator=g),
            })
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


@pytest.fixture(scope="module")
def model():
    return _make_model()

@pytest.fixture(scope="module")
def dataset():
    return ToyDataset()

@pytest.fixture(scope="module")
def layout(model):
    return build_token_layout(model, TOKEN_NAMES, BLOCK_MAP, BLOCK_NAMES)

DEV = torch.device("cpu")


def _forward(model, dataset, idx):
    s = dataset[idx]
    return model(s["sequence"].unsqueeze(0),
                 **build_fwd_kwargs(s, DEV, True, batched=False))


# ── layout / cache ───────────────────────────────────────────────────────────

def test_layout_matches_encode_concat_order(model, dataset, layout):
    _forward(model, dataset, 0)
    cat = torch.cat([model._block_tokens[b] for b in BLOCK_NAMES], dim=1)[0]
    for pos, (block, row) in enumerate(layout):
        assert torch.equal(model._block_tokens[block][0, row], cat[pos])


def test_layout_rejects_misordered_token_names(model):
    bad = TOKEN_NAMES[1:] + TOKEN_NAMES[:1]
    with pytest.raises(ValueError):
        build_token_layout(model, bad, BLOCK_MAP, BLOCK_NAMES)


def test_projector_cache_is_pre_norm_and_norm_recovers_legacy_cache(model, dataset):
    raw  = get_tokens_single(model, dataset, 3, DEV, True, patch_site="projector")
    post = get_tokens_single(model, dataset, 3, DEV, True, patch_site="token_norm")
    assert raw.shape == post.shape == (REGISTRY.total_tokens, D_PROJ)
    assert not torch.allclose(raw, post)                       # genuinely pre-norm
    assert torch.allclose(model.token_norm(raw), post, atol=1e-6)


# ── hook correctness ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("site", ["projector", "token_norm"])
def test_self_patch_is_identity(model, dataset, layout, site):
    self_patch_sanity_check(model, dataset, [0, 1, 2], DEV, True, TOKEN_NAMES,
                            site, layout if site == "projector" else None)


def test_projector_patch_replaces_raw_token_in_summary_path(model, dataset, layout):
    """THE regression: after a projector-site patch, the raw token that feeds
    the per-block summary is the SOURCE's; after a legacy token_norm patch it
    is still the TARGET's own."""
    tgt, src = 0, 1
    block = REGISTRY.token_block("REP_CORE" if "REP_CORE" in TOKEN_NAMES else "TE_CORE")
    sg    = "REP_CORE" if "REP_CORE" in TOKEN_NAMES else "TE_CORE"
    pos   = TOKEN_NAMES.index(sg)
    row   = layout[pos][1]

    _forward(model, dataset, tgt); tgt_raw = model._block_tokens[block][0, row].clone()
    _forward(model, dataset, src); src_raw = model._block_tokens[block][0, row].clone()
    assert not torch.allclose(tgt_raw, src_raw)

    # projector site
    src_vec = get_tokens_single(model, dataset, src, DEV, True, "projector")[[pos]]
    get_patched_logits(model, dataset, tgt, DEV, True, [pos], src_vec,
                       patch_site="projector", layout=layout)
    assert torch.allclose(model._block_tokens[block][0, row], src_raw)

    # legacy site: attention path patched, summary path still sees TARGET raw
    src_vec_n = get_tokens_single(model, dataset, src, DEV, True, "token_norm")[[pos]]
    get_patched_logits(model, dataset, tgt, DEV, True, [pos], src_vec_n,
                       patch_site="token_norm")
    assert torch.allclose(model._block_tokens[block][0, row], tgt_raw)


def test_attention_path_sees_identical_normalised_token_under_both_sites(model, dataset, layout):
    """token_norm is per-token, so patching the raw row at the projector must
    hand cross-attention exactly the source's normalised token — the
    attention-path half of the intervention is unchanged by the fix."""
    tgt, src, pos = 0, 1, TOKEN_NAMES.index("GQ")
    seen = {}

    def grab(tag):
        def _pre(module, args):
            seen[tag] = args[1].detach().clone()      # feature_tokens (B,N,d)
        return _pre

    for site in ("projector", "token_norm"):
        h = model.cross_attn.register_forward_pre_hook(grab(site))
        vec = get_tokens_single(model, dataset, src, DEV, True, site)[[pos]]
        get_patched_logits(model, dataset, tgt, DEV, True, [pos], vec,
                           patch_site=site, layout=layout if site == "projector" else None)
        h.remove()
    assert torch.allclose(seen["projector"], seen["token_norm"], atol=1e-5)


def test_sites_differ_in_logits_when_summary_path_matters(model, dataset, layout):
    tgt, src, pos = 0, 1, TOKEN_NAMES.index("GQ")
    base, _ = get_logits_single(model, dataset, tgt, DEV, True)
    out = {}
    for site in ("projector", "token_norm"):
        vec = get_tokens_single(model, dataset, src, DEV, True, site)[[pos]]
        out[site], _ = get_patched_logits(
            model, dataset, tgt, DEV, True, [pos], vec, patch_site=site,
            layout=layout if site == "projector" else None)
    assert abs(out["projector"] - out["token_norm"]) > 1e-6
    assert abs(out["projector"] - base) > 1e-6


def test_all_scope_projector_patch_replaces_every_block_token(model, dataset, layout):
    tgt, src = 0, 1
    allpos = list(range(len(TOKEN_NAMES)))
    _forward(model, dataset, src)
    src_blocks = {b: t.clone() for b, t in model._block_tokens.items()}
    vec = get_tokens_single(model, dataset, src, DEV, True, "projector")
    get_patched_logits(model, dataset, tgt, DEV, True, allpos, vec,
                       patch_site="projector", layout=layout)
    for b in BLOCK_NAMES:
        assert torch.allclose(model._block_tokens[b], src_blocks[b])


def test_projector_hook_requires_layout(model, dataset):
    vec = get_tokens_single(model, dataset, 1, DEV, True, "projector")[[0]]
    with pytest.raises(ValueError):
        get_patched_logits(model, dataset, 0, DEV, True, [0], vec,
                           patch_site="projector", layout=None)


def test_hooks_are_removed_after_patching(model, dataset, layout):
    vec = get_tokens_single(model, dataset, 1, DEV, True, "projector")[[0]]
    get_patched_logits(model, dataset, 0, DEV, True, [0], vec,
                       patch_site="projector", layout=layout)
    for b in BLOCK_NAMES:
        assert len(model.projectors[b]._forward_hooks) == 0
    assert len(model.token_norm._forward_hooks) == 0


# ── end-to-end: patch_fold on a toy fold ─────────────────────────────────────

@pytest.mark.parametrize("site", ["projector", "token_norm"])
def test_patch_fold_runs_and_is_self_describing(model, dataset, site):
    n = len(dataset)
    labels = ["lnc" if i % 2 == 0 else "pc" for i in range(n)]
    df = patch_fold(
        model=model, dataset=dataset, val_indices=np.arange(n), labels=labels,
        device=DEV, has_nonb2=True, n_pairs=4, min_confidence=0.0,
        max_length_diff=10.0, max_gc_diff=10.0, fold_idx=0,
        token_names=TOKEN_NAMES, block_map=BLOCK_MAP, block_names=BLOCK_NAMES,
        patch_modes=["single", "block", "all"], n_shuffled=None, patch_site=site,
    )
    assert set(df["patch_site"]) == {site}
    assert {"single", "block", "all"} == set(df["patch_mode"])
    assert len(df) > 0 and np.isfinite(df["delta_logit"]).all()


def test_patch_fold_sites_give_different_deltas(model, dataset):
    n = len(dataset)
    labels = ["lnc" if i % 2 == 0 else "pc" for i in range(n)]
    kw = dict(model=model, dataset=dataset, val_indices=np.arange(n), labels=labels,
              device=DEV, has_nonb2=True, n_pairs=4, min_confidence=0.0,
              max_length_diff=10.0, max_gc_diff=10.0, fold_idx=0,
              token_names=TOKEN_NAMES, block_map=BLOCK_MAP, block_names=BLOCK_NAMES,
              patch_modes=["single"], n_shuffled=None)
    a = patch_fold(patch_site="projector", **kw)
    b = patch_fold(patch_site="token_norm", **kw)
    assert np.allclose(a["baseline_logit"], b["baseline_logit"])      # same pairs
    assert not np.allclose(a["delta_logit"], b["delta_logit"])        # different intervention