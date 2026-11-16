# Block E Stage 3 (CPU): RM hold-out accuracy (G1 on test, confirmatory) + baselines on test + EXPLORATORY memorisation filter (Addendum A2).
import glob, json, re, time, os
import numpy as np, pandas as pd, torch
from rapidfuzz import process, fuzz
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from common import W
torch.set_num_threads(int(os.environ.get("NPROC", "38")))
t0 = time.time()
tr = pd.read_parquet(f"{W}/data/pairs_train.parquet"); se = pd.read_parquet(f"{W}/data/pairs_selection.parquet"); te = pd.read_parquet(f"{W}/data/pairs_test.parquet")
def raw(split): return pd.read_csv(glob.glob(f"{W}/data/upworthy/upworthy-archive-{split}-packages-*.csv")[0], low_memory=False)
def norm(x): return re.sub(r"\s+", " ", re.sub(r"<[^>]+>|[^a-z0-9 ]", " ", str(x).lower())).strip()
R = {s: raw(s) for s in ("confirmatory", "exploratory", "holdout")}
def story(df, split):
    r = R[split]; key = r.groupby(["clickability_test_id", "headline"]).first()[["slug", "excerpt", "lede"]]
    out = []
    for t, w, l in zip(df.test, df.winner, df.loser):
        s = set(); ex = set(); le = set()
        for h in (w, l):
            if (t, h) in key.index:
                k = key.loc[(t, h)]; s.add(str(k.slug)); ex.add(norm(k.excerpt)); le.add(norm(k.lede))
        out.append((s, ex - {"", "nan"}, le - {"", "nan"}))
    return out
TS = story(tr, "confirmatory")
tr_slug = set().union(*[x[0] for x in TS]); tr_ex = set().union(*[x[1] for x in TS]); tr_le = set().union(*[x[2] for x in TS])
tr_heads = sorted(set(map(norm, list(tr.winner) + list(tr.loser))))
print("train stories: slugs", len(tr_slug), "excerpts", len(tr_ex), "ledes", len(tr_le), "headlines", len(tr_heads), flush=True)
def flags(df, split):
    S = story(df, split)
    exact = np.array([bool(s & tr_slug) or bool(e & tr_ex) or bool(l & tr_le) for s, e, l in S])
    heads = sorted(set(map(norm, list(df.winner) + list(df.loser))))
    best = {}
    for i in range(0, len(heads), 2000):
        M = process.cdist(heads[i:i + 2000], tr_heads, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.uint8)
        for h, m in zip(heads[i:i + 2000], M.max(1)): best[h] = int(m)
    fz = np.array([best[norm(w)] >= 85 or best[norm(l)] >= 85 for w, l in zip(df.winner, df.loser)])
    return exact, fz
p = f"{W}/runs/rm/best"; tok = AutoTokenizer.from_pretrained(p); m = AutoModelForSequenceClassification.from_pretrained(p, num_labels=1, dtype=torch.float32).eval()
@torch.no_grad()
def score(hs):
    out = []
    for i in range(0, len(hs), 256):
        b = tok(hs[i:i + 256], truncation=True, max_length=48, padding=True, return_tensors="pt"); out += m(**b).logits[:, 0].tolist()
    return np.array(out)
res = {}
for name, df, split in (("test", te, "holdout"), ("selection", se, "exploratory")):
    c = df[df.clear].reset_index(drop=True)
    sw, sl = score(list(c.winner)), score(list(c.loser)); corr = sw > sl
    ex, fz = flags(c, split); keep = ~(ex | fz)
    res[name] = {"clear_n": int(len(c)), "clear_acc": float(corr.mean()),
                 "filtered_n": int(keep.sum()), "filtered_acc": float(corr[keep].mean()) if keep.any() else None,
                 "removed_exact_story": int(ex.sum()), "removed_fuzzy_headline": int(fz.sum()), "removed_any": int((ex | fz).sum()),
                 "acc_on_removed": float(corr[~keep].mean()) if (~keep).any() else None}
    if name == "test":
        sa_w, sa_l = score(list(df.winner)), score(list(df.loser)); res["test"]["all_n"] = len(df); res["test"]["all_acc"] = float((sa_w > sa_l).mean())
    print(name, res[name], f"{time.time()-t0:.0f}s", flush=True)
res["G1_on_test_pass"] = bool(res["test"]["clear_acc"] >= 0.60)
json.dump(res, open(f"{W}/runs/rm/rm_test_results.json", "w"), indent=1); print("RMTEST", json.dumps(res, indent=1))
