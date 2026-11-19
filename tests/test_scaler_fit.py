"""Feature scalers are fitted on the training/validation partition only.

Runs data.prep_features.prepare_features on synthetic tables that carry the
registry's columns; the test transcripts get extreme values, so any use of
them in the fit would change the scaler.
"""
import numpy as np
import pandas as pd
import pytest

from data.feature_registry import REGISTRY
from data.feature_scaler import FeatureScalerBank
from data.prep_features import prepare_features

N_TRAINVAL, N_TEST = 60, 20


def _ids(n, start):
    return [f"ENST{start + i:011d}" for i in range(n)]


def _table(block, trainval_ids, test_ids, seed):
    rng = np.random.default_rng(seed)
    cols = [m.name for m in REGISTRY.block_features(block)]
    values = rng.uniform(0.0, 1.0, size=(len(trainval_ids) + len(test_ids), len(cols)))
    values[len(trainval_ids):, :] = 1000.0            # test rows: far outside trainval
    df = pd.DataFrame(values, columns=cols, index=trainval_ids + test_ids)
    df.index.name = "transcript_id"
    return df


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("scaler")
    tv_ids, te_ids = _ids(N_TRAINVAL, 0), _ids(N_TEST, 10000)
    split = tmp / "split"
    split.mkdir()
    (split / "lnc_trainval.fa").write_text(
        "".join(f">{i}.1|ENSG{k:011d}.1|x\nACGT\n" for k, i in enumerate(tv_ids[:30])))
    (split / "pc_trainval.fa").write_text(
        "".join(f">{i}.1|ENSG{k + 500:011d}.1|x\nACGT\n" for k, i in enumerate(tv_ids[30:])))
    _table("te", tv_ids, te_ids, 1).to_csv(tmp / "te.csv")
    _table("nonb", tv_ids, te_ids, 2).to_csv(tmp / "nonb.csv")
    out = tmp / "out"
    te, nonb, te_bank, nonb_bank = prepare_features(
        tmp / "te.csv", tmp / "nonb.csv", split, out, release="g00")
    return dict(tv=tv_ids, te_ids=te_ids, out=out, te=te, nonb=nonb,
                te_bank=te_bank, nonb_bank=nonb_bank)


@pytest.mark.parametrize("block", ["te", "nonb"])
def test_bank_equals_a_trainval_only_fit(prepared, block):
    df, bank = prepared[block], prepared[f"{block}_bank"]
    X_all = df.to_numpy(np.float64)
    trainval_only = FeatureScalerBank(block).fit(df.loc[prepared["tv"]].to_numpy(np.float64))
    np.testing.assert_allclose(bank.transform(X_all), trainval_only.transform(X_all))


@pytest.mark.parametrize("block", ["te", "nonb"])
def test_fit_on_all_rows_would_differ(prepared, block):
    """Negative control: the test above can fail, so it is sensitive to leakage."""
    df, bank = prepared[block], prepared[f"{block}_bank"]
    X_all = df.to_numpy(np.float64)
    leaky = FeatureScalerBank(block).fit(X_all)
    assert not np.allclose(bank.transform(X_all), leaky.transform(X_all))


@pytest.mark.parametrize("block", ["te", "nonb"])
def test_saved_bank_reloads_identically(prepared, block):
    df, bank = prepared[block], prepared[f"{block}_bank"]
    loaded = FeatureScalerBank.load(prepared["out"] / f"g00_{block}_scaler_bank.pkl")
    X = df.to_numpy(np.float64)
    np.testing.assert_allclose(loaded.transform(X), bank.transform(X))
