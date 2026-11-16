# Block E post-hoc (EXPLORATORY): seed-1 trajectories, exploratory H3 at matched KL with seed 1, figE1 with seed 1 thin lines.
import glob, json, os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import W
G = f"{W}/runs/stage3"; S = f"{W}/runs/scale"; Z = json.load(open(f"{W}/data/zstats.json")); B = 2000
def load(path):
    d = pd.read_parquet(path)
    for suf in ("metrics", "scores", "posthoc"):
        f = path.replace(".parquet", f".{suf}.parquet")
        if os.path.exists(f): d = d.join(pd.read_parquet(f), rsuffix=f"_{suf}")
    return d
def curves(run):
    out = {0: load(f"{G}/curve_base_0.parquet")}
    for f in glob.glob(f"{G}/curve_{run}_*.parquet"):
        m = re.search(rf"curve_{run}_(\d+)\.parquet$", f)
        if m: out[int(m.group(1))] = load(f)
    return dict(sorted(out.items()))
def own(d, arm):
    v = d.valid.astype(bool); rz = (d.rm - Z["rm"]["mean"]) / Z["rm"]["sd"]
    r = (d.judge - Z["judge"]["mean"]) / Z["judge"]["sd"] if arm == "b" else (rz - 2 * (1 - d.train_nli_ent) if arm == "c" else rz)
    return float(np.where(v, r, -3.0).mean())
rows, C, KL = [], {}, {}
for arm in "abc":
    for seed in (0, 1):
        run = f"main_{arm}_s{seed}"; C[run] = curves(run)
        KL[run] = {0: 0.0, **{int(k): v["kl_seq_mean"] for k, v in json.load(open(f"{W}/runs/grpo/{run}/kl.json")).items()}}
        for s, d in C[run].items():
            rows.append({"run": run, "arm": arm, "seed": seed, "step": s, "kl_seq": KL[run].get(s), "own_reward_z": own(d, arm),
                         "click_model_z": float(((d.rm - Z["rm"]["mean"]) / Z["rm"]["sd"]).mean()), "audience_z": float(((d.judge - Z["judge"]["mean"]) / Z["judge"]["sd"]).mean()),
                         "distinct_per_600": int(d.headline.nunique()), "mean_words": float(d.words.mean()), "invented_reactions_pct": 100 * float(d.react.mean()),
                         "invented_numbers_pct": 100 * float(d.numbers_only.mean()), "off_article_pct": 100 * float(d.offarticle.mean()),
                         "entity_check_pct": 100 * float(d.fabricated.mean()), "n": len(d)})
old = pd.read_csv(f"{W}/trajectories_seed0.csv"); sc = old[old.run.str.startswith("scale")].assign(seed=0, arm="a")
T = pd.concat([pd.DataFrame(rows), sc], ignore_index=True).sort_values(["run", "step"]); T.to_csv(f"{W}/trajectories.csv", index=False)
# exploratory H3 at matched KL, seed 1 (same first-crossing interpolation as the confirmatory seed-0 analysis)
ids = sorted(C["main_a_s1"][0].id.unique()); ix = {i: k for k, i in enumerate(ids)}
def pa(d, col):
    g = d.groupby("id")[col].mean(); v = np.full(len(ids), np.nan); v[[ix[i] for i in g.index]] = g.values; return v
def at(run, tgt, col):
    kl = KL[run]; st = [s for s in sorted(kl) if s in C[run]]; xs = [kl[s] for s in st]
    for k in range(len(st) - 1):
        if xs[k] <= tgt <= xs[k + 1] or (k == len(st) - 2 and tgt >= xs[-1]):
            w = 0 if xs[k + 1] == xs[k] else min(1.0, (tgt - xs[k]) / (xs[k + 1] - xs[k]))
            return (1 - w) * pa(C[run][st[k]], col) + w * pa(C[run][st[k + 1]], col), (st[k], st[k + 1], round(w, 3))
rng = np.random.RandomState(0)
def boot(x):
    x = x[~np.isnan(x)]; n = len(x); bs = np.array([x[rng.randint(0, n, n)].mean() for _ in range(B)])
    return {"est": float(x.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "p": float(max(min(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1), 1 / B))}
H3 = {}
for seed in (0, 1):
    a, b = f"main_a_s{seed}", f"main_b_s{seed}"; tgt = min(KL[a][max(KL[a])], KL[b][max(KL[b])])
    H3[f"seed{seed}"] = {"kl_target": tgt, "kl_final_a": KL[a][max(KL[a])], "kl_final_b": KL[b][max(KL[b])]}
    for col in ("fabricated", "numbers_only", "react"):
        va, ba = at(a, tgt, col); vb, bb = at(b, tgt, col)
        H3[f"seed{seed}"][col] = {**boot(va - vb), "a_rate": float(np.nanmean(va)), "b_rate": float(np.nanmean(vb)), "bracket_a": ba, "bracket_b": bb}
json.dump(H3, open(f"{W}/h3_matched_by_seed.json", "w"), indent=1)
# figure + plotting data
N = json.load(open(f"{W}/numbers_E.json")); TB = N["table"]
COL = {"a": "#1f77b4", "b": "#d62728", "c": "#2ca02c"}; ARMS = {"a": "real-click RM", "b": "simulated audience", "c": "real-click RM + NLI"}
SC = {"scale_E1": ("#9467bd", "E1 Llama-3.1-8B (a)"), "scale_E2": ("#ff7f0e", "E2 Qwen2.5-14B (a)")}
P = T[["run", "arm", "seed", "step", "entity_check_pct", "invented_numbers_pct", "invented_reactions_pct"]].copy()
P["base_entity_check_pct"] = 100 * TB["base"]["fabricated"]; P["human_entity_check_pct"] = 100 * TB["human"]["fabricated"]
P["base_invented_numbers_pct"] = 100 * TB["base"]["numbers_only"]; P["human_invented_numbers_pct"] = 100 * TB["human"]["numbers_only"]
P.to_csv(f"{W}/figs/figE1_data.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for (run, arm, seed), g in T.groupby(["run", "arm", "seed"]):
    g = g.sort_values("step")
    if run.startswith("scale"):
        c, lab = SC[run]; kw = dict(ls="--", marker="s", ms=3, lw=1.5, c=c, label=lab + " (exploratory)")
    else:
        kw = dict(ls="-", marker="o", ms=3 if seed == 0 else 2, lw=1.8 if seed == 0 else 0.8, alpha=1 if seed == 0 else 0.7, c=COL[arm],
                  label=f"({arm}) {ARMS[arm]}, Qwen2.5-3B, seed {seed}")
    ax[0].plot(g.step, g.entity_check_pct, **kw); kw.pop("label"); ax[1].plot(g.step, g.invented_numbers_pct, **kw)
for k, (col, ttl) in enumerate((("fabricated", "Fabrication rate, entity check (%)"), ("numbers_only", "Unsupported-number rate (%)"))):
    ax[k].axhline(TB["base"][col] * 100, c="grey", ls="--", lw=1, label="base Qwen2.5-3B (test, final eval)" if k == 0 else None)
    ax[k].axhline(TB["human"][col] * 100, c="black", ls=":", lw=1, label="human headline" if k == 0 else None)
    ax[k].set_xlabel("GRPO step"); ax[k].set_title(ttl, fontsize=10)
ax[0].legend(fontsize=6); fig.tight_layout()
for e in ("pdf", "png"): fig.savefig(f"{W}/figs/figE1_fabrication_vs_step.{e}", dpi=200)
print("SEED1_DONE", json.dumps({k: {c: (round(v[c]["est"] * 100, 1), round(v[c]["lo"] * 100, 1), round(v[c]["hi"] * 100, 1)) for c in ("fabricated", "numbers_only", "react")} for k, v in H3.items()}))
