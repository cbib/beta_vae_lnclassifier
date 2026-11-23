"""pipeline_audit.py --path_map reads results written under old folder names.

Protects: OLD=NEW rewrites strings and dict keys everywhere the audit reads them
(config, checkpoint config, layers manifest), applies the mappings in order, rejects a
malformed spec, and warns when a mapping matches nothing.
"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pipeline_audit", ROOT / "scripts" / "pipeline_audit.py")
pa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pa)

CFG = {
    "data": {"lnc_fasta": "data/split_gencode_49_genegrouped/lnc_trainval.fa",
             "nonb2_csv": "data/processed_features_genomic/g49_nonb2_features_clean.csv"},
    "output": {"experiment_name": "gencode_v49_experiments_genegrouped/beta_vae_subgroup_base_g49"},
    "training": {"n_folds": 5},
}


def _ctx(tmp_path, path_map):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(CFG))
    return pa.Ctx(SimpleNamespace(config=str(cfg), exp_root=str(tmp_path), path_map=path_map))


def test_rewrites_config_paths_and_experiment_dir(tmp_path):
    c = _ctx(tmp_path, ["_genegrouped="])
    assert c.d["lnc_fasta"] == "data/split_gencode_49/lnc_trainval.fa"
    assert c.d["nonb2_csv"] == CFG["data"]["nonb2_csv"]                      # untouched
    assert c.exp_dir == tmp_path / "gencode_v49_experiments/beta_vae_subgroup_base_g49"
    assert c.path_map_hits["_genegrouped"] == 2


def test_rewrites_dict_keys_and_nested_values(tmp_path):
    c = _ctx(tmp_path, ["gencode_v49_experiments_genegrouped=gencode_v49_experiments"])
    man = {"checkpoints": {"/r/gencode_v49_experiments_genegrouped/m/fold_0_best.pt": "abc"},
           "config": "/r/gencode_v49_experiments_genegrouped/cfg.json", "n": 3, "l": ["x_genegrouped"]}
    out = c.remap(man)
    assert list(out["checkpoints"]) == ["/r/gencode_v49_experiments/m/fold_0_best.pt"]
    assert out["config"] == "/r/gencode_v49_experiments/cfg.json"
    assert out["n"] == 3 and out["l"] == ["x_genegrouped"]


def test_mappings_apply_in_order(tmp_path):
    c = _ctx(tmp_path, ["a=b", "b=c"])
    assert c.remap("a") == "c"


def test_no_path_map_changes_nothing(tmp_path):
    c = _ctx(tmp_path, [])
    assert c.cfg == CFG and c.path_map == []


@pytest.mark.parametrize("bad", ["nodelimiter", "=new"])
def test_malformed_spec_exits(tmp_path, bad):
    with pytest.raises(SystemExit):
        _ctx(tmp_path, [bad])


def test_unmatched_mapping_is_a_warning(tmp_path):
    c = _ctx(tmp_path, ["_genegrouped=", "typo_xyz=q"])
    c.remap(CFG)
    pa.RES.clear()
    pa.report_path_map(c)
    lv = {r["msg"].split(" ->")[0]: r["level"] for r in pa.RES}
    assert lv == {"_genegrouped": "INFO", "typo_xyz": "WARN"}


def test_manifest_config_not_found_is_a_warning_and_the_map_fixes_it(tmp_path):
    exp = tmp_path / "gencode_v49_experiments" / "beta_vae_subgroup_base_g49"
    (exp / "interpretability").mkdir(parents=True)
    real = tmp_path / "cfg.json"
    real.write_text(json.dumps(CFG))
    old = str(tmp_path / "old_name" / "cfg.json")
    man = {"checkpoints": {}, "config": old, "config_sha": pa.sha16(str(real))}
    (exp / "interpretability" / "run_manifest_1.json").write_text(json.dumps(man))
    cfg = dict(CFG, output={"experiment_name": "gencode_v49_experiments/beta_vae_subgroup_base_g49"})
    (tmp_path / "c.json").write_text(json.dumps(cfg))
    for pm, level in (([], "WARN"), ([f"{tmp_path}/old_name={tmp_path}"], "PASS")):
        c = pa.Ctx(SimpleNamespace(config=str(tmp_path / "c.json"), exp_root=str(tmp_path), path_map=pm))
        pa.RES.clear()
        pa.stage_layers(c)
        got = [r["level"] for r in pa.RES if r["check"] == "manifest_config"]
        assert got == [level], (pm, pa.RES)


def _layers_dir(tmp_path, files):
    exp = tmp_path / "gencode_v49_experiments" / "beta_vae_subgroup_base_g49"
    (exp / "interpretability").mkdir(parents=True)
    for name, df in files.items():
        df.to_csv(exp / "interpretability" / name, index=False)
    (tmp_path / "c.json").write_text(json.dumps(dict(
        CFG, output={"experiment_name": "gencode_v49_experiments/beta_vae_subgroup_base_g49"})))
    c = pa.Ctx(SimpleNamespace(config=str(tmp_path / "c.json"), exp_root=str(tmp_path), path_map=[]))
    pa.RES.clear()
    pa.stage_layers(c)
    return {r["check"]: r for r in pa.RES}


def test_iia_differences_may_be_negative_but_iia_itself_may_not(tmp_path):
    import pandas as pd
    ok = _layers_dir(tmp_path / "a", {"x.csv": pd.DataFrame(
        {"iia": [0.2, 0.9], "excess_iia_over_noise": [-0.3, 0.4], "n_iia_sucess": [12, 340]})})
    assert ok["iia_range"]["level"] == "PASS"
    bad = _layers_dir(tmp_path / "b", {"x.csv": pd.DataFrame(
        {"iia": [0.2, 1.3], "excess_iia_over_noise": [-1.4, 0.4]})})
    assert bad["iia_range"]["level"] == "FAIL"
    assert "x.csv:iia" in bad["iia_range"]["msg"] and "x.csv:excess_iia_over_noise" in bad["iia_range"]["msg"]


def test_non_finite_message_names_the_columns(tmp_path):
    import numpy as np
    import pandas as pd
    r = _layers_dir(tmp_path, {"pair.csv": pd.DataFrame({"a": [1.0, np.nan, 2.0], "b": [1.0, 2.0, np.inf], "c": [1, 2, 3]})})
    assert r["finite"]["level"] == "FAIL"
    assert "pair.csv:a(1),b(1)" in r["finite"]["msg"]

def test_unmatched_baseline_may_have_undefined_pair_means(tmp_path):
    import numpy as np
    import pandas as pd
    df = pd.DataFrame({"group": ["all_highconf_unmatched", "matched"],
                       "smd_gc": [-0.4, 0.1], "mean_abs_gc_diff": [np.nan, 0.05]})
    assert _layers_dir(tmp_path / "a", {"pair_balance_fold0.csv": df})["finite"]["level"] == "PASS"
    df.loc[1, "mean_abs_gc_diff"] = np.nan                       # a matched group must have it
    assert _layers_dir(tmp_path / "b", {"pair_balance_fold0.csv": df})["finite"]["level"] == "FAIL"
    df.loc[1, "mean_abs_gc_diff"] = 0.05
    df.loc[0, "smd_gc"] = np.nan                                  # other columns still checked
    assert _layers_dir(tmp_path / "c", {"pair_balance_fold0.csv": df})["finite"]["level"] == "FAIL"