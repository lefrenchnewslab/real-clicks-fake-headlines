# Block E post-hoc (EXPLORATORY), CPU: (1) inclusion-weighted detector validation, (2) training-trajectory table,
# (3) blinded intermediate-checkpoint label sheet.
import glob, json, os, re
import numpy as np, pandas as pd
from common import W
G = f"{W}/runs/stage3"; S = f"{W}/runs/scale"; Z = json.load(open(f"{W}/data/zstats.json")); B = 2000
# ---------------- (1) detector validation with inclusion weights ----------------
M = pd.read_csv(f"{W}/handlabel/m1_joined.csv")
DET = {"entity check (main deterministic)": "fabricated", "eval NLI (tau 0.55)": "nli", "entity OR NLI": "det_or_nli",
       "invented numbers": "numbers_only", "invented reactions": "react", "OLMo probe": "probe"}
def wpr(df, col):
    f, y, w = df[col].astype(float).values, df.y.astype(float).values, df.ipw.astype(float).values
    tp = (w * f * y).sum(); pf = (w * f).sum(); py = (w * y).sum()
    return (tp / pf if pf else np.nan), (tp / py if py else np.nan)
def boot(df, col, rng):
    groups = [g for _, g in df.groupby(["source", "flagged"])]            # resample within the sampling strata
    bs = []
    for _ in range(B):
        x = pd.concat([g.iloc[rng.randint(0, len(g), len(g))] for g in groups]); bs.append(wpr(x, col))
    bs = np.array(bs, dtype=float); return np.nanpercentile(bs[:, 0], [2.5, 97.5]).tolist(), np.nanpercentile(bs[:, 1], [2.5, 97.5]).tolist()
rows = []
for subset, df in (("all 100", M), ("items 51-100", M[M.item > 50])):
    for name, col in DET.items():
        rng = np.random.RandomState(0); p, r = wpr(df, col); pci, rci = boot(df, col, rng)
        rows.append({"subset": subset, "detector": name, "group": "mixture (4 sources equally weighted)", "w_precision": p, "w_precision_lo": pci[0],
                     "w_precision_hi": pci[1], "w_recall": r, "w_recall_lo": rci[0], "w_recall_hi": rci[1], "n_items": len(df), "n_label_fab": int(df.y.sum())})
        for s, g in df.groupby("source"):
            p, r = wpr(g, col)
            rows.append({"subset": subset, "detector": name, "group": s, "w_precision": p, "w_recall": r, "n_items": len(g), "n_label_fab": int(g.y.sum())})
V = pd.DataFrame(rows); V.to_csv(f"{W}/detector_validation_weighted.csv", index=False)
# ---------------- (2) training trajectories ----------------
def load(path):
    d = pd.read_parquet(path)
    for suf in ("metrics", "scores", "posthoc"):
        f = path.replace(".parquet", f".{suf}.parquet")
        if os.path.exists(f): d = d.join(pd.read_parquet(f), rsuffix=f"_{suf}")
    return d
def own_reward(d, arm):
    valid = d.valid.astype(bool); rmz = (d.rm - Z["rm"]["mean"]) / Z["rm"]["sd"]
    if arm == "b": r = (d.judge - Z["judge"]["mean"]) / Z["judge"]["sd"]
    elif arm == "c": r = rmz - 2 * (1 - d.train_nli_ent)
    else: r = rmz
    return float(np.where(valid, r, -3.0).mean())
traj = []
def add(run, arm, step, d, kl):
    traj.append({"run": run, "step": step, "kl_seq": kl, "own_reward_z": own_reward(d, arm) if ("judge" in d or arm != "b") else None,
                 "click_model_z": float(((d.rm - Z["rm"]["mean"]) / Z["rm"]["sd"]).mean()),
                 "audience_z": float(((d.judge - Z["judge"]["mean"]) / Z["judge"]["sd"]).mean()) if "judge" in d else None,
                 "distinct_per_600": int(d.headline.nunique()), "invented_reactions_pct": 100 * float(d.react.mean()),
                 "invented_numbers_pct": 100 * float(d.numbers_only.mean()), "off_article_pct": 100 * float(d.offarticle.mean()),
                 "entity_check_pct": 100 * float(d.fabricated.mean()), "n": len(d)})
base0 = load(f"{G}/curve_base_0.parquet")
for arm in "abc":
    run = f"main_{arm}_s0"; KL = json.load(open(f"{W}/runs/grpo/{run}/kl.json"))
    add(run, arm, 0, base0, 0.0)
    for f in glob.glob(f"{G}/curve_{run}_*.parquet"):
        m = re.search(r"_(\d+)\.parquet$", f)
        if m and not re.search(r"\.(scores|metrics|posthoc)\.", f): add(run, arm, int(m.group(1)), load(f), KL[m.group(1)]["kl_seq_mean"])
for run in ("scale_E1", "scale_E2"):
    KL = json.load(open(f"{W}/runs/grpo/{run}/kl.json"))
    for f in glob.glob(f"{S}/{run}_*.parquet"):
        m = re.search(r"_(\d+)\.parquet$", f)
        if m and not re.search(r"\.(scores|metrics|posthoc)\.", f):
            s = int(m.group(1)); add(run, "a", s, load(f), 0.0 if s == 0 else KL[str(s)]["kl_seq_mean"])
T = pd.DataFrame(traj).sort_values(["run", "step"]); T.to_csv(f"{W}/trajectories_seed0.csv", index=False)
# ---------------- (3) blinded intermediate-checkpoint sheet ----------------
AP = f"{W}/data/articles.parquet"   # release: the labelling sheet needs article text, rebuilt by build_articles.py
pick = [("main_a_s0", 100), ("main_a_s0", 150), ("main_b_s0", 250), ("main_b_s0", 350), ("main_c_s0", 350)]
rng = np.random.RandomState(0); ks = []
for run, step in pick:
    d = load(f"{G}/curve_{run}_{step}.parquet").reset_index(drop=True)
    idx = rng.choice(len(d), 12, replace=False)
    for i in idx:
        r = d.iloc[i]
        ks.append({"source": run, "step": step, "seed": 0, "id": r.id, "sample": int(r["sample"]), "headline": r.headline,
                   "entity_check": int(r.fabricated), "invented_numbers": int(r.numbers_only), "invented_reaction": int(r.react),
                   "off_article": int(r.offarticle), "eval_nli_ent": float(r.eval_nli_ent), "nli_tau055": int(r.eval_nli_ent < 0.55)})
K = pd.DataFrame(ks).sample(frac=1, random_state=0).reset_index(drop=True); K.insert(0, "item", np.arange(1, len(K) + 1))
K.to_csv(f"{W}/handlabel/intermediate_key.csv", index=False)
if os.path.exists(AP):
    arts = pd.read_parquet(AP).set_index("id").article_trunc
    pd.DataFrame({"item": K.item, "headline": K.headline, "article": [arts[i] for i in K.id], "faithful": "", "fabricated_fact": "",
              "misleading_exaggeration": "", "teaser_no_claim": "", "other": "", "note": ""}).to_csv(f"{W}/handlabel/intermediate_sheet.csv", index=False)
print("REANALYSIS_DONE", len(V), len(T), len(K))
