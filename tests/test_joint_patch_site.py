"""
tests/test_joint_patch_site.py — batched hook in joint_patching_search.py.

Pins: (1) batched self-patch identity at both sites, (2) the batched
projector-site hook agrees with patch_tokens.py's per-sample hook,
(3) the --single_summary patch-site guard, (4) greedy search runs end to end
at the projector site, (5) patch_site is written to the build_summary output.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from test_patch_site import (_make_model, ToyDataset, TOKEN_NAMES, BLOCK_MAP,
                             BLOCK_NAMES, DEV)
from analysis.pattern import joint_patching_search as J
from analysis.pattern import patch_tokens as V


@pytest.fixture(scope="module")
def model():   return _make_model(seed=1)
@pytest.fixture(scope="module")
def dataset(): return ToyDataset(n=24, seed=3)
@pytest.fixture(scope="module")
def layout(model): return J.build_token_layout(model, TOKEN_NAMES, BLOCK_MAP, BLOCK_NAMES)

IDX = np.arange(8)


@pytest.mark.parametrize("site", ["projector", "token_norm"])
@pytest.mark.parametrize("positions", [[0], list(range(20))])
def test_batched_self_patch_identity(model, dataset, layout, site, positions):
    tm = J.cache_tokens_batched(model, dataset, IDX.tolist(), DEV, True,
                                batch_size=4, patch_site=site)
    J.batched_self_patch_check(model, dataset, IDX, tm, positions, DEV, True,
                               chunk_size=4, patch_site=site,
                               layout=layout if site == "projector" else None)


@pytest.mark.parametrize("site", ["projector", "token_norm"])
def test_cache_tokens_batched_handles_partial_last_batch(model, dataset, site):
    """10 indices at batch_size=4 -> batches of 4, 4, 2; the cache must match
    the per-sample tokens (regression: stale per-block tensors from the
    previous batch crashed the final, smaller batch)."""
    idx = list(range(10))
    tm = J.cache_tokens_batched(model, dataset, idx, DEV, True,
                                batch_size=4, patch_site=site)
    for i in idx:
        ref = V.get_tokens_single(model, dataset, i, DEV, True, site)
        assert torch.allclose(tm[i], ref, atol=1e-5)


def test_batched_projector_hook_matches_per_sample_hook(model, dataset, layout):
    tgt, src = IDX[:4], IDX[4:8]
    positions = [TOKEN_NAMES.index("GQ"), TOKEN_NAMES.index("TE_CORE"),
                 TOKEN_NAMES.index("SS_BINNED")]
    tm = J.cache_tokens_batched(model, dataset, IDX.tolist(), DEV, True,
                                batch_size=4, patch_site="projector")
    vecs = torch.stack([tm[int(i)][positions] for i in src])
    b_logit, _ = J.batched_patched_forward(
        model, dataset, tgt, DEV, True, positions, vecs, chunk_size=3,
        patch_site="projector", layout=layout)
    for k, (t, s) in enumerate(zip(tgt, src)):
        ps, _ = V.get_patched_logits(
            model, dataset, int(t), DEV, True, positions,
            V.get_tokens_single(model, dataset, int(s), DEV, True, "projector")[positions],
            patch_site="projector", layout=layout)
        assert abs(ps - b_logit[k]) < 1e-4


def test_batched_cache_matches_per_sample_cache(model, dataset):
    tm = J.cache_tokens_batched(model, dataset, IDX.tolist(), DEV, True,
                                batch_size=4, patch_site="projector")
    for i in IDX[:3]:
        ref = V.get_tokens_single(model, dataset, int(i), DEV, True, "projector")
        assert torch.allclose(tm[int(i)], ref, atol=1e-5)


def test_single_summary_guard():
    legacy = pd.DataFrame({"patch_scope": ["GQ"], "iia_overall": [0.1]})
    with pytest.raises(RuntimeError):
        J.check_single_summary_site(legacy, "projector", allow_legacy=False)
    J.check_single_summary_site(legacy, "projector", allow_legacy=True)   # warns only
    ok = legacy.assign(patch_site="projector")
    J.check_single_summary_site(ok, "projector", allow_legacy=False)
    with pytest.raises(RuntimeError):
        J.check_single_summary_site(ok.assign(patch_site="token_norm"),
                                    "projector", allow_legacy=True)


def test_greedy_search_runs_end_to_end_at_projector_site(model, dataset, layout):
    lnc, mrna = IDX[:4], IDX[4:8]
    tm = J.cache_tokens_batched(model, dataset, IDX.tolist(), DEV, True,
                                batch_size=4, patch_site="projector")
    def patch_fn(sgs):
        return J.run_joint_patch(model, dataset, lnc, mrna, tm, sgs, TOKEN_NAMES,
                                 DEV, True, chunk_size=4, patch_site="projector",
                                 layout=layout)
    r = patch_fn(["GQ", "TE_CORE"])
    assert {"iia", "symmetry_score"} <= set(r)
    df = J.greedy_forward_search(TOKEN_NAMES, {s: 0.0 for s in TOKEN_NAMES}, 2, patch_fn)
    assert len(df) >= 1


def test_build_summary_keeps_patch_site():
    rows = []
    for d, ps in (("lnc_to_mrna", 0.1), ("mrna_to_lnc", -0.2)):
        for pair in range(3):
            rows.append(dict(patch_mode="single", patch_scope="GQ", block="nonb",
                             direction=d, pair_i=pair, delta_logit=ps, delta_prob=ps,
                             iia_baseline_valid=True, iia_success=bool(pair % 2),
                             fold=0, patch_site="projector"))
    out = V.build_summary(pd.DataFrame(rows))
    assert set(out["patch_site"]) == {"projector"}