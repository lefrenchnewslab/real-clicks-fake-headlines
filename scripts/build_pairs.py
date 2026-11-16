# Block E Stage 0: Upworthy within-test, same-image headline pairs. Login node, CPU.
import glob, json, re, sys, math, itertools, random
import pandas as pd
import os
W = os.environ.get("RLW", os.path.expandvars("$WORK_ROOT/rlhead"))   # set RLW, or WORK_ROOT (parent of rlhead/)
EXPECT = {"exploratory": (4873, 22666), "confirmatory": (22743, 105551), "holdout": (4871, 22600)}
log = {}
def norm(h): return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(h).lower())).strip()
def load(split):
    f = glob.glob(f"{W}/data/upworthy/upworthy-archive-{split}-packages-*.csv"); assert len(f) == 1, f
    d = pd.read_csv(f[0], low_memory=False)
    n_t, n_p = d.clickability_test_id.nunique(), len(d)
    log[split] = {"file": f[0].split("/")[-1], "columns": list(d.columns), "tests": n_t, "packages": n_p,
                  "expected": EXPECT[split], "match": (n_t, n_p) == EXPECT[split]}
    print(split, log[split], flush=True)
    return d
def truthy(s): return s.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes"})
def pairs(d, split):
    L = log[split]; L["rows_in"] = len(d)
    if "problem" in d: d = d[~truthy(d.problem)]; L["after_problem"] = len(d)
    else:
        # Deviation (logged): released CSVs have no `problem` column. Use the window from the 2024 update page
        # (https://upworthy.natematias.com/2024-06-upworthy-archive-update.html): tests from 2013-06-25 to 2014-01-10.
        t0 = pd.to_datetime(d.created_at, format="ISO8601").groupby(d.clickability_test_id).transform("min")
        bad = (t0 >= "2013-06-25") & (t0 < "2014-01-11")
        L["problem_window_tests_dropped"] = int(d.loc[bad, "clickability_test_id"].nunique())
        d = d[~bad]; L["after_problem_window"] = len(d)
    rel = [c for c in d.columns if re.search(r"reliab|unreliab|exclude", c, re.I)]
    L["reliability_cols"] = rel
    for c in rel:
        vals = d[c].astype(str).str.lower().value_counts().to_dict(); L[f"values_{c}"] = vals
        bad = d[c].astype(str).str.lower().isin({"0", "false", "unreliable", "no"}) if not re.search("unreliab|exclude", c, re.I) else truthy(d[c])
        d = d[~bad]; L[f"after_{c}"] = len(d)
    d = d.dropna(subset=["headline", "eyecatcher_id", "impressions", "clicks"])
    d = d[d.impressions > 0]
    g = d.groupby(["clickability_test_id", "eyecatcher_id", "headline"], as_index=False)[["clicks", "impressions"]].sum()
    out = []
    for (t, img), grp in g.groupby(["clickability_test_id", "eyecatcher_id"]):
        rows = grp.to_dict("records")
        for a, b in itertools.combinations(rows, 2):
            if norm(a["headline"]) == norm(b["headline"]): continue
            pa, pb = a["clicks"] / a["impressions"], b["clicks"] / b["impressions"]
            if pa == pb: continue
            p = (a["clicks"] + b["clicks"]) / (a["impressions"] + b["impressions"])
            se = math.sqrt(p * (1 - p) * (1 / a["impressions"] + 1 / b["impressions"])) if 0 < p < 1 else float("nan")
            z = (pa - pb) / se if se and se > 0 else 0.0
            w, l = (a, b) if pa > pb else (b, a)
            out.append({"test": t, "eyecatcher_id": img, "winner": w["headline"], "loser": l["headline"], "ctr_w": max(pa, pb), "ctr_l": min(pa, pb),
                        "imp_w": w["impressions"], "imp_l": l["impressions"], "absz": abs(z)})
    P = pd.DataFrame(out); L["pairs_all"] = len(P); return P
D = {s: load(s) for s in EXPECT}
P = {s: pairs(D[s], s) for s in EXPECT}
train = P["confirmatory"][P["confirmatory"].absz >= 1.0].reset_index(drop=True)
seen = set(map(norm, train.winner)) | set(map(norm, train.loser))
log["train_pairs_absz_ge_1"] = len(train)
res = {"train": train}
for s, name in [("exploratory", "selection"), ("holdout", "test")]:
    q = P[s]; before = len(q)
    q = q[~(q.winner.map(norm).isin(seen) | q.loser.map(norm).isin(seen))].reset_index(drop=True)
    q["clear"] = q.absz >= 1.96
    log[f"{name}_pairs_all"] = len(q); log[f"{name}_dropped_train_overlap"] = before - len(q); log[f"{name}_pairs_clear"] = int(q.clear.sum())
    res[name] = q
rng = random.Random(0)
for k, q in res.items():
    q["swap"] = [rng.random() < 0.5 for _ in range(len(q))]     # fixed random orientation for feature baselines
    q.to_parquet(f"{W}/data/pairs_{k}.parquet")
json.dump(log, open(f"{W}/data/pairs_log.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in log.items() if not isinstance(v, dict)}, indent=1))
