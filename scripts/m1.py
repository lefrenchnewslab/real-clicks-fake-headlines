# Block E M1 (pre-registered) + exploratory label analyses. CPU. Labels: LLM-drafted, author-approved (see README).
import json, os, sys
import numpy as np, pandas as pd
from common import W
G = f"{W}/runs/stage3"; rng = np.random.RandomState(0); B = 2000
K = pd.read_csv(f"{W}/handlabel/key.csv"); L = pd.read_csv(f"{W}/handlabel/labels.csv")
cats = ["faithful", "fabricated_fact", "misleading_exaggeration", "teaser_no_claim", "other"]
M = K.merge(L[["item"] + cats + ["claude_label", "agreed"]], on="item", validate="one_to_one")
assert len(M) == 100 and (M[cats].fillna(0).sum(1) == 1).all()
M["label"] = M[cats].fillna(0).idxmax(axis=1); M["y"] = (M.label == "fabricated_fact").astype(int)
feats = []
for s in M.source.unique():
    d = pd.read_parquet(f"{G}/final_{s}.parquet")
    for suf in ("metrics", "scores", "posthoc"):
        f = f"{G}/final_{s}.{suf}.parquet"
        if os.path.exists(f): d = d.join(pd.read_parquet(f), rsuffix=f"_{suf}")
    d["source"] = s; feats.append(d)
F = pd.concat(feats)[["source", "id", "sample", "headline", "fabricated", "numbers_only", "fab_zipf", "react", "offarticle", "probe_pyes", "eval_nli_ent"]]
M = M.merge(F, on=["source", "id", "sample"], how="left", suffixes=("", "_gen"), validate="one_to_one")
assert M.fabricated.notna().all() and (M.headline == M.headline_gen).all(), "key/output join mismatch"
assert ((M.fabricated == M.flagged)).all(), "flag in key differs from metric"
M["probe"] = (M.probe_pyes > 0.5).astype(int)
def pr(df, col):
    f = df[col].astype(int); tp = (f & df.y); prec = tp.sum() / f.sum() if f.sum() else np.nan
    rec = (tp * df.ipw).sum() / (df.y * df.ipw).sum() if df.y.sum() else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and not np.isnan(prec) and not np.isnan(rec) else 0.0
    return prec, rec, f1
def withci(df, col):
    p, r, f = pr(df, col); bs = []
    for _ in range(B):
        x = df.iloc[rng.randint(0, len(df), len(df))]; bs.append(pr(x, col)[:2])
    bs = np.array(bs, dtype=float)
    return {"precision": p, "precision_ci": np.nanpercentile(bs[:, 0], [2.5, 97.5]).tolist(), "recall_ipw": r,
            "recall_ci": np.nanpercentile(bs[:, 1], [2.5, 97.5]).tolist(), "f1": f, "n_items": len(df), "n_flagged": int(df[col].sum()), "n_label_fab": int(df.y.sum())}
sel, test = M[M.item <= 50], M[M.item > 50]
# tau for eval NLI on items 1-50 by F1
taus = np.round(np.arange(0.05, 0.96, 0.05), 2); f1s = {}
for t in taus:
    sel_ = sel.assign(nli=(sel.eval_nli_ent < t).astype(int)); f1s[float(t)] = pr(sel_, "nli")[2]
tau = max(f1s, key=lambda t: (f1s[t], -abs(t - 0.5)))
M["nli"] = (M.eval_nli_ent < tau).astype(int); M["det_or_nli"] = (M.fabricated.astype(int) | M.nli).astype(int)
sel, test = M[M.item <= 50], M[M.item > 50]
res = {"provenance": "labels drafted by Claude (LLM) from headline + article only, blind to source/flags/key; Amr reviewed all 100, accepted all (0 overrides)",
       "eval_nli_tau": tau, "tau_f1_grid_selection": f1s}
res["M1"] = withci(test, "fabricated"); res["M1"]["met"] = bool(res["M1"]["precision"] >= 0.7 and res["M1"]["recall_ipw"] >= 0.6)
pa = pr(test[test.source == "main_a_s0"], "fabricated")[0]; pb = pr(test[test.source == "base"], "fabricated")[0]
res["precision_arm_a_test"] = pa; res["precision_base_test"] = pb
res["flag_H1_style_driven"] = bool(not np.isnan(pa) and not np.isnan(pb) and pb - pa > 0.2)
res["n_test_items_by_source"] = test.source.value_counts().to_dict()
res["switch_rule_selection_f1"] = {c: pr(sel, c)[2] for c in ("fabricated", "nli", "det_or_nli")}
res["switch_choice_if_needed"] = max(res["switch_rule_selection_f1"], key=res["switch_rule_selection_f1"].get)
res["nli_on_test"] = withci(test, "nli"); res["det_or_nli_on_test"] = withci(test, "det_or_nli")
res["exploratory"] = {c: {"test_half": withci(test, c), "all_100": withci(M, c)} for c in ("numbers_only", "fab_zipf", "probe", "react", "fabricated", "nli")}
res["exploratory"]["confusion_by_source_all100"] = {s: pd.crosstab(g.fabricated, g.label).to_dict() for s, g in M.groupby("source")}
res["exploratory"]["label_counts_by_source"] = {s: g.label.value_counts().to_dict() for s, g in M.groupby("source")}
res["exploratory"]["precision_by_source_all100"] = {s: pr(g, "fabricated")[0] for s, g in M.groupby("source")}
M.drop(columns=["headline_gen"]).to_csv(f"{W}/handlabel/m1_joined.csv", index=False)
json.dump(res, open(f"{W}/m1_E.json", "w"), indent=1, default=float)
print("M1", json.dumps({k: res[k] for k in ("M1", "eval_nli_tau", "precision_arm_a_test", "precision_base_test", "flag_H1_style_driven", "switch_choice_if_needed")}, default=float))
