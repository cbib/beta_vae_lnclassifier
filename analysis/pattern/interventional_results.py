"""Collect Step 2/3 results for one genome release into a single text file.

Usage:
  python interventional_results.py --joint_dir <joint_dir> --controls_dir <controls_dir> \
      --out g49_results.txt [--log <joint_log>] [--top 15]
"""
import argparse
import re
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_colwidth", 60)


def section(f, title):
    f.write(f"\n{'=' * 100}\n{title}\n{'=' * 100}\n")


def dump(f, path, n=None, sort=None, query=None):
    p = Path(path)
    if not p.exists():
        f.write(f"[missing] {p}\n")
        return None
    df = pd.read_csv(p)
    f.write(f"[{p.name}] shape={df.shape}\ncolumns: {list(df.columns)}\n")
    if query:
        try:
            df = df.query(query)
        except Exception as e:  # noqa: BLE001
            f.write(f"[query failed: {e}]\n")
    if sort and sort in df.columns:
        df = df.sort_values(sort)
    f.write((df.head(n) if n else df).round(4).to_string(index=False) + "\n")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint_dir", required=True)
    ap.add_argument("--controls_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--matched_summary", default=None,
                    help="patching_summary.csv containing the matched rows")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--joint_controls_dir", default=None,
                    help="output dir of the --mode controls run, if separate")
    a = ap.parse_args()
    jd, cd = Path(a.joint_dir), Path(a.controls_dir)

    with open(a.out, "w") as f:
        section(f, "1. GREEDY SUMMARY (cross-fold mean/std per step)")
        dump(f, jd / "joint_patching_greedy_summary.csv")

        section(f, "2. TOP-K SUMMARY (best rows)")
        dump(f, jd / "joint_patching_topk_summary.csv", n=a.top)

        section(f, "3. PAIRWISE INTERACTION SUMMARY (top by fisher_q)")
        pr = dump(f, jd / "joint_patching_pairs_summary.csv", n=a.top, sort="fisher_q")
        if pr is not None and "fisher_q" in pr.columns:
            f.write(f"\nn pairs: {len(pr)}\n"
                    f"fisher_q < 0.05: {(pr['fisher_q'] < 0.05).sum()}\n")
            if "mean_logit_interaction" in pr.columns:
                sig = pr[pr["fisher_q"] < 0.05]
                f.write(f"significant, interaction > 0: {(sig['mean_logit_interaction'] > 0).sum()}\n"
                        f"significant, interaction < 0: {(sig['mean_logit_interaction'] < 0).sum()}\n")

        section(f, "4. MATCHED vs CONTROLS: pivot by (scope, block); one row per scope")
        frames = []
        for pth in (a.matched_summary, cd / "patching_summary.csv"):
            if pth and Path(pth).exists():
                frames.append(pd.read_csv(pth))
            elif pth:
                f.write(f"[missing] {pth}\n")
        if frames:
            df = pd.concat(frames, ignore_index=True)
            df["patch_mode"] = df["patch_mode"].replace({"single": "matched", "block": "matched", "all": "matched"})
            df = df[df["patch_mode"].isin(["matched", "shuffled", "sameclass", "noise"])]
            f.write(f"modes: {sorted(df['patch_mode'].unique())}\n")
            for val in ("symmetry_score", "iia_overall"):
                pv = (df.groupby(["patch_scope", "block", "patch_mode"])[val].first()
                        .unstack("patch_mode"))
                order = [m for m in ("matched", "shuffled", "sameclass", "noise") if m in pv.columns]
                pv = pv[order].sort_values(order[0], ascending=False)
                f.write(f"\n-- {val} --\n" + pv.round(4).to_string() + "\n")
            dirs = (df.groupby(["patch_scope", "block", "patch_mode", "direction"])["iia"].first()
                      .unstack(["patch_mode", "direction"]))
            f.write("\n-- iia by direction --\n" + dirs.round(4).to_string() + "\n")

        section(f, "5. JOINT-SET CONTROLS: greedy sets (matched vs noise vs same-class)")
        cs = jd.parent / "joint_patching_controls_summary.csv"
        for cand in (jd / "joint_patching_controls_summary.csv",
                     Path(a.joint_controls_dir) / "joint_patching_controls_summary.csv"
                     if a.joint_controls_dir else cs):
            if cand.exists():
                c = pd.read_csv(cand)
                f.write(f"[{cand}] shape={c.shape}\n")
                keep = [x for x in ("set", "set_size", "iia_matched", "iia_noise_mean",
                                    "iia_sameclass_mean", "excess_iia_over_noise",
                                    "symmetry_matched", "symmetry_noise_mean",
                                    "excess_symmetry_over_noise", "n_folds") if x in c.columns]
                g = c[c["kind"] == "greedy_step"].sort_values("set_size")
                f.write("\n-- greedy steps --\n" + g[keep].round(4).to_string(index=False) + "\n")
                pr_ = c[c["kind"] == "pair"].sort_values("excess_iia_over_noise", ascending=False)
                f.write(f"\n-- pairs: top {a.top} by excess over noise --\n"
                        + pr_[keep].head(a.top).round(4).to_string(index=False) + "\n")
                f.write(f"pairs with excess_iia_over_noise > 0.01: {(pr_['excess_iia_over_noise'] > 0.01).sum()} / {len(pr_)}\n")
                break
        else:
            f.write("[missing] joint_patching_controls_summary.csv\n")

        section(f, "6. PAIR BALANCE (confound balance of the matched pairs)")
        for cand in (jd, Path(a.joint_controls_dir) if a.joint_controls_dir else jd):
            if (cand / "pair_balance_summary.csv").exists():
                dump(f, cand / "pair_balance_summary.csv")
                break
        else:
            f.write("[missing] pair_balance_summary.csv\n")

        if a.log and Path(a.log).exists():
            section(f, "7. LOG LINES (selection / split / warnings / errors)")
            pat = re.compile(r"Selected|held-out|heldout|selection|WARN|Warning|Error|Traceback|patch_site", re.I)
            for line in Path(a.log).read_text(errors="ignore").splitlines():
                if pat.search(line):
                    f.write(line + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()