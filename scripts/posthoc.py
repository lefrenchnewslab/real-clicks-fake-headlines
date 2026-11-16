# Block E post-hoc (EXPLORATORY), CPU. P2: fabrication under the ORIGINAL spec Title-Case rule (zipf >= 4.5, whole-span filter,
# no casing lexicon, no truecasing). P3: invented-reaction and off-article rates (fixed lists in RESULTS_E.md).
import glob, json, os, re, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from common import W
G = f"{W}/runs/stage3"
REACT = {"controversy", "debate", "backlash", "outrage", "uproar", "slam", "blast", "spark", "fury", "criticism", "viral"}
def init():
    global EM, ART, LEM
    import eval_metrics as EM
    ART = pd.read_parquet(f"{W}/data/articles.parquet").set_index("id").article_trunc.to_dict(); LEM = {}
def art_lemmas(i):
    if i not in LEM:
        d = EM.nlp()(ART[i]); LEM[i] = ({t.lemma_.lower() for t in d}, {t.lemma_.lower() for t in d if t.is_alpha and not t.is_stop})
    return LEM[i]
def zipf_entities(h, a):
    out = []
    for e in EM.nlp()(h).ents:
        if e.label_ not in EM.ENT_TYPES: continue
        text = re.sub(r"^(the|a|an)\s+", "", e.text, flags=re.I).strip(" '’"); text = re.sub(r"['’]s$", "", text)
        toks = re.findall(r"[A-Za-z][A-Za-z.'’]*", text)
        if not toks: continue
        acr = any(len(t.replace(".", "")) >= 2 and t.replace(".", "").isupper() for t in toks)
        if not acr and all(EM.zipf_frequency(t.lower(), "en") >= 4.5 and not EM._cap_midsentence(t, a) for t in toks): continue
        out.append((text, e.label_))
    return out
def work(row):
    i, h = row; a = ART[i]
    if not h: return {"fab_zipf": 0, "react": 0, "offarticle": 1, "react_words": "[]"}
    an = EM._norm(a); avals = [v for _, v in EM.numbers(a)]
    ue = [t for t, l in zipf_entities(h, a) if not EM.entity_supported(t, l, an)]
    un = [s for s, v in EM.numbers(h, headline=True) if not EM.number_supported(v, avals)]
    alem, acont = art_lemmas(i); hd = EM.nlp()(h)
    hl = [t.lemma_.lower() for t in hd]; rw = sorted({l for l in hl if l in REACT and l not in alem})
    hc = {t.lemma_.lower() for t in hd if t.is_alpha and not t.is_stop}
    return {"fab_zipf": int(bool(ue or un)), "react": int(bool(rw)), "offarticle": int(len(hc & acont) == 0), "react_words": json.dumps(rw)}
if __name__ == "__main__" and sys.argv[1] == "rows":
    pats = sys.argv[2:] or [f"{G}/final_*.parquet", f"{G}/curve_*.parquet"]
    files = [f for f in sorted(sum((glob.glob(q) for q in pats), [])) if not re.search(r"\.(scores|metrics|posthoc)\.parquet$", f)]
    with Pool(int(os.environ.get("NPROC", "38")), initializer=init) as pool:
        for f in files:
            d = pd.read_parquet(f); t = time.time()
            order = d.sort_values("id").index                          # group rows by article for the lemma cache
            r = pd.DataFrame(pool.map(work, list(zip(d.loc[order, "id"], d.loc[order, "headline"].fillna(""))), chunksize=32), index=order).loc[d.index]
            r.to_parquet(f.replace(".parquet", ".posthoc.parquet")); print(f"POSTHOC {os.path.basename(f)} fab_zipf={r.fab_zipf.mean():.3f} react={r.react.mean():.3f} off={r.offarticle.mean():.3f} {time.time()-t:.0f}s", flush=True)
if __name__ == "__main__" and sys.argv[1] == "stats":
    rng = np.random.RandomState(0); B = 2000
    def load(stem):
        d = pd.read_parquet(f"{G}/{stem}.parquet").join(pd.read_parquet(f"{G}/{stem}.metrics.parquet")[["fabricated", "numbers_only"]]).join(pd.read_parquet(f"{G}/{stem}.posthoc.parquet"))
        return d
    RUNS = [f"main_{a}_s{s}" for a in "abc" for s in (0, 1)]
    F = {s: load(f"final_{s}") for s in ["base", "human"] + RUNS}
    ids = sorted(F["base"].id.unique()); ix = {i: k for k, i in enumerate(ids)}
    def pa(d, col, ids=ids, ix=ix):
        g = d.groupby("id")[col].mean(); v = np.full(len(ids), np.nan); v[[ix[i] for i in g.index]] = g.values; return v
    def boot(x):
        x = x[~np.isnan(x)]; n = len(x); bs = np.array([x[rng.randint(0, n, n)].mean() for _ in range(B)])
        return {"est": float(x.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "p": float(max(min(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1), 1 / B))}
    pool_ = lambda rs: pd.concat([F[r] for r in rs])
    res = {"per_source": {s: {c: float(d[c].mean()) for c in ("fabricated", "fab_zipf", "numbers_only", "react", "offarticle")} for s, d in F.items()}}
    # release: pooled per-arm rates (both seeds; the paper's Table 2 columns "Invented reactions" and "Off-article")
    res["per_arm_pooled"] = {f"arm_{a}": {c: float(pool_([f"main_{a}_s0", f"main_{a}_s1"])[c].mean()) for c in ("fabricated", "fab_zipf", "numbers_only", "react", "offarticle")} for a in "abc"}
    for c in ("fabricated", "fab_zipf"):
        res[f"H2_{c}"] = boot(pa(pool_(["main_b_s0", "main_b_s1"]), c) - pa(F["base"], c))
        res[f"H1_{c}"] = boot(pa(pool_(["main_a_s0", "main_a_s1"]), c) - pa(F["base"], c))
    # matched KL (seed 0), same procedure as analyze.py (first crossing) + alternative (arm b final checkpoint)
    KL = {a: {0: 0.0, **{int(k): v["kl_seq_mean"] for k, v in json.load(open(f"{W}/runs/grpo/main_{a}_s0/kl.json")).items()}} for a in "ab"}
    def curve(a):
        out = {0: load("curve_base_0")}
        for f in glob.glob(f"{G}/curve_main_{a}_s0_*.parquet"):
            m = re.search(r"_(\d+)\.parquet$", f)
            if m and not re.search(r"\.(scores|metrics|posthoc)\.", f): out[int(m.group(1))] = load(f"curve_main_{a}_s0_{m.group(1)}")
        return out
    C = {a: curve(a) for a in "abc"}; cids = sorted(C["a"][0].id.unique()); cix = {i: k for k, i in enumerate(cids)}
    def at(a, tgt, col, first=True):
        kl = KL[a]; st = sorted(s for s in kl if s in C[a]); xs = [kl[s] for s in st]
        if not first and abs(xs[-1] - tgt) < 1e-9: return pa(C[a][st[-1]], col, cids, cix), (st[-1], st[-1], 1.0)
        for k in range(len(st) - 1):
            if xs[k] <= tgt <= xs[k + 1]:
                w = (tgt - xs[k]) / (xs[k + 1] - xs[k]); return (1 - w) * pa(C[a][st[k]], col, cids, cix) + w * pa(C[a][st[k + 1]], col, cids, cix), (st[k], st[k + 1], w)
    tgt = min(KL["a"][max(KL["a"])], KL["b"][max(KL["b"])]); res["kl_target"] = tgt
    for c in ("fabricated", "fab_zipf", "numbers_only", "react", "offarticle"):
        va, ba = at("a", tgt, c); vb, bb = at("b", tgt, c); vb2, bb2 = at("b", tgt, c, first=False)
        res[f"H3m_{c}_firstcross"] = {**boot(va - vb), "bracket_a": ba, "bracket_b": bb}
        res[f"H3m_{c}_bfinal"] = {**boot(va - vb2), "bracket_b": bb2}
    res["curves"] = {a: {s: {c: float(d[c].mean()) for c in ("fabricated", "fab_zipf", "react", "offarticle")} for s, d in sorted(C[a].items())} for a in "abc"}
    json.dump(res, open(f"{W}/posthoc_E.json", "w"), indent=1); print("POSTHOC_STATS_DONE")
