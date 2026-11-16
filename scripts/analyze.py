# Block E Stage 3 analysis (CPU): H1-H4 with article-clustered bootstrap + Holm (Addendum A1), matched-KL, exploratory, figures.
import glob, json, os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import W
G = f"{W}/runs/stage3"; FIG = f"{W}/figs"; os.makedirs(FIG, exist_ok=True)
SWITCH = os.environ.get("SWITCH_METRIC", "")   # M1 failed -> pre-registered switch (set by C8 job)
SFX = "_switched" if SWITCH else ""
B = 2000; rng = np.random.RandomState(0)
Z = json.load(open(f"{W}/data/zstats.json"))
ARMS = {"a": "real-click RM", "b": "simulated audience", "c": "real-click RM + NLI"}
RUNS = [f"main_{a}_s{s}" for a in "abc" for s in (0, 1)]
N = {}

def load(stem):
    d = pd.read_parquet(f"{G}/{stem}.parquet")
    for suf in ("metrics", "scores"):
        f = f"{G}/{stem}.{suf}.parquet"
        if os.path.exists(f): d = d.join(pd.read_parquet(f), rsuffix=f"_{suf}")
    if "rm" in d: d["rm_z"] = (d.rm - Z["rm"]["mean"]) / Z["rm"]["sd"]
    if "judge" in d: d["judge_z"] = (d.judge - Z["judge"]["mean"]) / Z["judge"]["sd"]
    if "probe_pyes" in d: d["probe_flag"] = (d.probe_pyes > 0.5).astype(int)
    if "eval_nli_ent" in d: d["nli_unsup"] = (d.eval_nli_ent < TAU).astype(int)
    if SWITCH == "det_or_nli":
        d["fabricated_det"] = d.fabricated; d["fabricated"] = ((d.fabricated == 1) | (d.eval_nli_ent < TAU)).astype(int)
    d["fab_given_claim"] = np.where(d.claim_bearing == 1, d.fabricated, np.nan) if "claim_bearing" in d else np.nan
    return d
TAU = float(os.environ.get("NLI_TAU", "0.5"))   # 0.5 default; 0.55 chosen on hand-label items 1-50 (m1_E.json)
lab_f = f"{W}/handlabel/labels.csv" if not SWITCH else "/nonexistent"   # M1 itself is computed by m1.py
F = {s: load(f"final_{s}") for s in ["base", "human"] + RUNS if os.path.exists(f"{G}/final_{s}.parquet")}
ids = sorted(F["base"].id.unique()); idx = {i: k for k, i in enumerate(ids)}
METRICS = ["fabricated", "numbers_only", "claim_bearing", "fab_given_claim", "listicle", "probe_flag", "nli_unsup", "eval_nli_con",
           "words", "copy8", "question", "you", "teaser", "clickbait", "vader", "repetition", "valid", "rm_z", "judge_z"]

def per_article(d, col):
    g = d.groupby("id")[col].mean(); v = np.full(len(ids), np.nan); v[[idx[i] for i in g.index]] = g.values; return v
def boot(diff):
    """diff: per-article values (nan allowed). Returns est, 95% CI, two-sided p (A1 rule 4)."""
    ok = ~np.isnan(diff); x = diff[ok]; n = len(x)
    bs = np.array([x[rng.randint(0, n, n)].mean() for _ in range(B)])
    p = max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / B)
    return {"est": float(x.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "p": float(min(p, 1.0)), "n_articles": int(n)}
def pool(runs): return pd.concat([F[r] for r in runs if r in F])
def diff(A, Bd, col): return per_article(A, col) - per_article(Bd, col)

# ---------------- descriptive table ----------------
table = {}
for s, d in F.items():
    row = {m: float(d[m].mean()) for m in METRICS if m in d}
    row["n"] = len(d)
    if "headline" in d:   # release: final_human ships without the publishers' titles (reproduce.py --articles restores them)
        row["distinct1"] = len(set(w for h in d.headline for w in h.lower().split())) / max(1, sum(len(h.split()) for h in d.headline))
        row["n_distinct_headlines"] = int(d.headline.nunique())   # release: emitted for the paper's "five distinct headlines" (Sec. 4.2)
    table[s] = row
for a in "abc":
    d = pool([f"main_{a}_s0", f"main_{a}_s1"]); table[f"arm_{a}_pooled"] = {m: float(d[m].mean()) for m in METRICS if m in d}
N["table"] = table

# ---------------- H1, H2, H3 (final step), H4 ----------------
H = {}
def hyp_vs_base(arm):
    A = pool([f"main_{arm}_s0", f"main_{arm}_s1"]); out = {}
    for col in ("fabricated", "numbers_only", "probe_flag", "nli_unsup", "claim_bearing", "fab_given_claim", "words"):
        out[col] = boot(diff(A, F["base"], col))
    for s in (0, 1):
        out[f"fabricated_seed{s}"] = boot(diff(F[f"main_{arm}_s{s}"], F["base"], "fabricated"))
    b = F["base"].fabricated.mean(); out["relative_change_pct"] = 100 * (A.fabricated.mean() - b) / b if b else None
    return out
H["H1"] = hyp_vs_base("a"); H["H2"] = hyp_vs_base("b")
H3f = {col: boot(diff(pool(["main_a_s0", "main_a_s1"]), pool(["main_b_s0", "main_b_s1"]), col)) for col in ("fabricated", "numbers_only", "probe_flag")}
H3f["fabricated_seed0"] = boot(diff(F["main_a_s0"], F["main_b_s0"], "fabricated")); H3f["fabricated_seed1"] = boot(diff(F["main_a_s1"], F["main_b_s1"], "fabricated"))

# ---------------- matched KL (seed 0 curves) ----------------
def kl_curve(run):
    k = json.load(open(f"{W}/runs/grpo/{run}/kl.json")); return {0: 0.0, **{int(s): v["kl_seq_mean"] for s, v in k.items()}}
def curve(run):
    out = {0: load("curve_base_0")}
    for f in glob.glob(f"{G}/curve_{run}_*.parquet"):
        m = re.search(rf"curve_{run}_(\d+)\.parquet$", f)
        if m: out[int(m.group(1))] = load(f"curve_{run}_{m.group(1)}")
    return dict(sorted(out.items()))
KL = {r: kl_curve(r) for r in RUNS if os.path.exists(f"{W}/runs/grpo/{r}/kl.json")}
N["kl_final"] = {r: KL[r][max(KL[r])] for r in KL}
C = {a: curve(f"main_{a}_s0") for a in "abc"}
cids = sorted(C["a"][0].id.unique()); cidx = {i: k for k, i in enumerate(cids)}
def pa(d, col):
    g = d.groupby("id")[col].mean(); v = np.full(len(cids), np.nan); v[[cidx[i] for i in g.index]] = g.values; return v
def at_kl(arm, target, col="fabricated"):
    kl = KL[f"main_{arm}_s0"]; steps = [s for s in sorted(kl) if s in C[arm]]
    xs = np.array([kl[s] for s in steps])
    for k in range(len(steps) - 1):
        if xs[k] <= target <= xs[k + 1] or (k == len(steps) - 2 and target >= xs[-1]):
            w = 0 if xs[k + 1] == xs[k] else min(1.0, (target - xs[k]) / (xs[k + 1] - xs[k]))
            return (1 - w) * pa(C[arm][steps[k]], col) + w * pa(C[arm][steps[k + 1]], col), (steps[k], steps[k + 1], float(w))
    raise ValueError("target KL outside curve")
fa, fb = N["kl_final"]["main_a_s0"], N["kl_final"]["main_b_s0"]; kstar = min(fa, fb)
va, bra = at_kl("a", kstar); vb, brb = at_kl("b", kstar)
H3m = {"kl_target": kstar, "kl_final_a_s0": fa, "kl_final_b_s0": fb, "kl_ratio": max(fa, fb) / min(fa, fb), "bracket_a": bra, "bracket_b": brb,
       "fabricated": boot(va - vb)}
for col in ("numbers_only", "probe_flag"):
    try: H3m[col] = boot(at_kl("a", kstar, col)[0] - at_kl("b", kstar, col)[0])
    except Exception as e: H3m[col] = str(e)
H["H3"] = {"final_step": H3f, "matched_kl": H3m, "interpret_only_matched": bool(H3m["kl_ratio"] > 2)}

# Holm across H1, H2, H3 (H3 uses the matched-KL comparison)
ps = {"H1": H["H1"]["fabricated"]["p"], "H2": H["H2"]["fabricated"]["p"], "H3": H3m["fabricated"]["p"]}
order = sorted(ps, key=ps.get); rej = {}; stop = False
for k, h in enumerate(order):
    thr = 0.05 / (3 - k)
    rej[h] = (not stop) and ps[h] <= thr
    if not rej[h]: stop = True
H["holm"] = {"p": ps, "rejected": rej}
def label_vs_base(h):
    r = H[h]; f = r["fabricated"]; sign = np.sign(f["est"])
    agree = np.sign(r["numbers_only"]["est"]) == sign and np.sign(r["probe_flag"]["est"]) == sign
    seeds = np.sign(r["fabricated_seed0"]["est"]) == np.sign(r["fabricated_seed1"]["est"])
    if rej[h] and f["lo"] > 0.01 and agree: lab = "rise"
    elif rej[h] and f["hi"] < -0.01: lab = "fall"
    elif f["lo"] > -0.03 and f["hi"] < 0.03: lab = "no rise"
    else: lab = "inconclusive"
    return {"label": lab, "numbers_probe_sign_agree": bool(agree), "seeds_agree_in_sign": bool(seeds),
            "directional_claim_allowed": bool(seeds) if lab in ("rise", "fall") else None}
def label_h3(f, sig):
    if sig and f["lo"] > 0.01: return "real > simulated"
    if sig and f["hi"] < -0.01: return "real < simulated"
    if f["lo"] > -0.03 and f["hi"] < 0.03: return "equivalent"
    return "inconclusive"
H["H1"]["decision"] = label_vs_base("H1"); H["H2"]["decision"] = label_vs_base("H2")
H["H3"]["decision_matched"] = label_h3(H3m["fabricated"], rej["H3"])
H["H3"]["decision_final"] = label_h3(H3f["fabricated"], rej["H3"]) if not H["H3"]["interpret_only_matched"] else "reported, not interpreted (KL ratio > 2)"
H["H3"]["seeds_agree_final"] = bool(np.sign(H3f["fabricated_seed0"]["est"]) == np.sign(H3f["fabricated_seed1"]["est"]))
# H4 (Addendum A3): CI upper bounds for safety conditions, point keep_share, seed rule; outside Holm
from h4 import decide
def arrs(d, cols=("rm_z", "fabricated", "probe_flag", "copy8")): return {c: per_article(d, c) for c in cols}
h4 = decide(arrs(F["base"]), arrs(pool(["main_a_s0", "main_a_s1"])), arrs(pool(["main_c_s0", "main_c_s1"])),
            [(arrs(F[f"main_a_s{s}"]), arrs(F[f"main_c_s{s}"])) for s in (0, 1)])
Cc, Bb = pool(["main_c_s0", "main_c_s1"]), F["base"]
h4["nli_c_minus_base"] = boot(diff(Cc, Bb, "nli_unsup"))      # reported, not gating
h4["met"] = h4["decision"] == "met"; H["H4"] = h4
N["hypotheses"] = H

# ---------------- exploratory ----------------
X = {}
X["curves"] = {a: {s: {"fabricated": float(d.fabricated.mean()), "numbers_only": float(d.numbers_only.mean()), "claim_bearing": float(d.claim_bearing.mean()),
                       "kl": KL[f"main_{a}_s0"].get(s), "rm_z": float(d.rm_z.mean()) if "rm_z" in d else None,
                       "judge_z": float(d.judge_z.mean()) if "judge_z" in d else None, "words": float(d.words.mean())}
                   for s, d in C[a].items()} for a in "abc"}
if os.path.exists(f"{G}/bon16_base.parquet"):
    bo = load("bon16_base"); best = bo.loc[bo.groupby("id").rm.idxmax()]
    X["bon16"] = {"fabricated_best": float(best.fabricated.mean()), "fabricated_all16": float(bo.fabricated.mean()),
                  "rm_z_best": float(best.rm_z.mean()), "rm_z_all": float(bo.rm_z.mean()), "numbers_only_best": float(best.numbers_only.mean()), "n_articles": int(best.id.nunique())}
allf = pd.concat([d.assign(src=s) for s, d in F.items() if s != "human"])
X["cross_reward_spearman_rm_judge"] = float(allf[["rm", "judge"]].corr(method="spearman").iloc[0, 1])
X["cross_scores"] = {s: {"rm_z": float(d.rm_z.mean()), "judge_z": float(d.judge_z.mean()), "eval_nli_ent": float(d.eval_nli_ent.mean())} for s, d in F.items()}
try:
    from sklearn.linear_model import LogisticRegression
    dd = allf[allf.src.isin(["base"] + RUNS)].copy(); dd["arm"] = dd.src.str.extract(r"main_(\w)_")[0].fillna("base")
    Xd = pd.get_dummies(dd.arm)[["a", "b", "c"]].astype(float); Xd["words"] = dd.words
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(Xd, dd.fabricated)
    X["logit_fab_on_arm_plus_words"] = dict(zip(Xd.columns, lr.coef_[0].round(4).tolist()))
except Exception as e: X["logit_fab_on_arm_plus_words"] = str(e)
for f, k in ((f"{W}/runs/rm/rm_results.json", "rm_selection"), (f"{W}/runs/rm/rm_test_results.json", "rm_test")):
    if os.path.exists(f): N[k] = json.load(open(f))
N["exploratory"] = X

# ---------------- M1 (when labels exist) ----------------
if os.path.exists(lab_f):
    K = pd.read_csv(f"{W}/handlabel/key.csv"); L = pd.read_csv(lab_f); M = K.merge(L[["item", "fabricated_fact"]], on="item")
    M["y"] = M.fabricated_fact.fillna(0).astype(float).clip(0, 1)
    def pr(df):
        tp = ((df.flagged == 1) & (df.y == 1)); prec = tp.sum() / max(1, (df.flagged == 1).sum())
        rec = (tp * df.ipw).sum() / max(1e-9, (df.y * df.ipw).sum()); return {"precision": float(prec), "recall_ipw": float(rec), "n": len(df)}
    m1 = {"test_half": pr(M[M.half == "test"]), "selection_half": pr(M[M.half == "selection"]),
          "prec_arm_a": pr(M[(M.half == "test") & (M.source == "main_a_s0")])["precision"], "prec_base": pr(M[(M.half == "test") & (M.source == "base")])["precision"]}
    m1["met"] = bool(m1["test_half"]["precision"] >= 0.7 and m1["test_half"]["recall_ipw"] >= 0.6)
    m1["flag_H1_style_driven"] = bool(m1["prec_base"] - m1["prec_arm_a"] > 0.2); N["M1"] = m1
N["metric"] = SWITCH or "deterministic"; N["nli_tau"] = TAU
json.dump(N, open(f"{W}/numbers_E{SFX}.json", "w"), indent=1, default=float)

# ---------------- figures ----------------
COL = {"a": "#1f77b4", "b": "#d62728", "c": "#2ca02c"}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for a in "abc":
    st = sorted(X["curves"][a]); ax[0].plot(st, [X["curves"][a][s]["fabricated"] * 100 for s in st], "-o", ms=3, c=COL[a], label=f"({a}) {ARMS[a]}")
    ax[1].plot(st, [X["curves"][a][s]["numbers_only"] * 100 for s in st], "-o", ms=3, c=COL[a])
for k, (col, ttl) in enumerate((("fabricated", "Fabrication rate (%)"), ("numbers_only", "Unsupported-number rate (%)"))):
    ax[k].axhline(F["base"][col].mean() * 100, c="grey", ls="--", lw=1, label="base (test, final eval)" if k == 0 else None)
    ax[k].axhline(F["human"][col].mean() * 100, c="black", ls=":", lw=1, label="human headline" if k == 0 else None)
    ax[k].set_xlabel("GRPO step (seed 0)"); ax[k].set_title(ttl, fontsize=10)
ax[0].legend(fontsize=7); fig.tight_layout(); [fig.savefig(f"{FIG}/figE1_fabrication_vs_step{SFX}.{e}", dpi=200) for e in ("pdf", "png")]; plt.close(fig)
fig, ax = plt.subplots(figsize=(5.2, 4))
for a in "abc":
    st = sorted(X["curves"][a]); own = "judge_z" if a == "b" else "rm_z"
    g0 = X["curves"][a][0][own]
    ax.plot([X["curves"][a][s][own] - g0 for s in st], [X["curves"][a][s]["fabricated"] * 100 for s in st], "-o", ms=3, c=COL[a], label=f"({a}) own reward")
for a, v in (("a", va), ("b", vb)):
    own = "judge_z" if a == "b" else "rm_z"; s1, s2, w = (bra if a == "a" else brb)
    gx = (1 - w) * X["curves"][a][s1][own] + w * X["curves"][a][s2][own] - X["curves"][a][0][own]
    ax.plot(gx, np.nanmean(v) * 100, "*", ms=14, c=COL[a], mec="k", label=f"({a}) at matched KL={kstar:.2f}")
ax.set_xlabel("Own-reward gain over base (z)"); ax.set_ylabel("Fabrication rate (%)"); ax.legend(fontsize=7); fig.tight_layout()
[fig.savefig(f"{FIG}/figE2_reward_vs_fabrication{SFX}.{e}", dpi=200) for e in ("pdf", "png")]; plt.close(fig)
r = N.get("rm_selection", {})
if r:
    fig, ax = plt.subplots(figsize=(6, 3.6)); cats = [("sel_clear", "clear"), ("sel_all", "all"), ("sel_clear_lenmatched", "clear,\nlength-matched")]
    xs = np.arange(len(cats)); w = 0.26
    for k, (pre, lab, c) in enumerate((("feat", "feature BT", "#999"), ("tfidf", "TF-IDF BT", "#666"), ("rm", "DeBERTa RM", "#1f77b4"))):
        ax.bar(xs + (k - 1) * w, [r[f"{pre}_{c_}"] * 100 for c_, _ in cats], w, label=lab, color=c)
    if "rm_test" in N: ax.plot([xs[0] + w], [N["rm_test"]["test"]["clear_acc"] * 100], "k*", ms=10, label="RM, hold-out clear")
    ax.axhline(50, c="k", lw=0.8, ls=":"); ax.set_xticks(xs); ax.set_xticklabels([l for _, l in cats]); ax.set_ylabel("Pairwise accuracy (%)")
    ax.set_ylim(40, 90); ax.legend(fontsize=7); ax.set_title("Selection pairs (Upworthy exploratory split)", fontsize=10); fig.tight_layout()
    [fig.savefig(f"{FIG}/figE3_rm_validity{SFX}.{e}", dpi=200) for e in ("pdf", "png")]; plt.close(fig)
print("ANALYSIS_DONE", json.dumps({"H1": H["H1"]["decision"], "H2": H["H2"]["decision"], "H3": H["H3"]["decision_matched"], "H4": h4["decision"], "holm": H["holm"]}, default=str))
