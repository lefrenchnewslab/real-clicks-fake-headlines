# Block E exploratory scale/family check: per-checkpoint table, pre-stated prediction check, figE1 with dashed scale lines.
import glob, json, os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import W
S = f"{W}/runs/scale"; Z = json.load(open(f"{W}/data/zstats.json"))
LABEL = {"scale_E1": "E1 Llama-3.1-8B (a)", "scale_E2": "E2 Qwen2.5-14B (a)"}
res = {}
for run in ("scale_E1", "scale_E2"):
    KL = json.load(open(f"{W}/runs/grpo/{run}/kl.json")) if os.path.exists(f"{W}/runs/grpo/{run}/kl.json") else {}
    rows = {}
    for f in glob.glob(f"{S}/{run}_*.parquet"):
        m = re.search(rf"{run}_(\d+)\.parquet$", f)
        if not m: continue
        s = int(m.group(1)); d = pd.read_parquet(f)
        for suf in ("metrics", "posthoc", "scores"):
            g = f.replace(".parquet", f".{suf}.parquet")
            if os.path.exists(g): d = d.join(pd.read_parquet(g), rsuffix=f"_{suf}")
        rows[s] = {"numbers_only": float(d.numbers_only.mean()), "fabricated": float(d.fabricated.mean()), "offarticle": float(d.offarticle.mean()),
                   "react": float(d.react.mean()), "distinct_per_600": int(d.headline.nunique()), "words": float(d.words.mean()),
                   "rm_z": float(((d.rm - Z["rm"]["mean"]) / Z["rm"]["sd"]).mean()), "valid": float(d.valid.mean()),
                   "kl": 0.0 if s == 0 else KL.get(str(s), {}).get("kl_seq_mean"), "examples": list(d.headline.iloc[[0, 2, 4]])}
    rows = dict(sorted(rows.items())); b = rows.get(0, {})
    i_ = any(rows[s]["react"] > b.get("react", 1) for s in rows if 0 < s <= 150)
    ii_ = any(rows[s]["numbers_only"] > 0.5 and rows[s]["distinct_per_600"] < 50 for s in rows if 0 < s <= 250)
    collapsed = os.path.exists(f"{W}/runs/grpo/{run}/COLLAPSED")
    res[run] = {"curve": rows, "pred_i_reaction_above_base_by_150": bool(i_), "pred_ii_numbers_gt50_and_distinct_lt50_by_250": bool(ii_),
                "prediction": "met (both)" if (i_ and ii_) else ("failed (neither)" if not (i_ or ii_) else "partly (one of two)"), "collapsed": collapsed}
json.dump(res, open(f"{W}/scale_E.json", "w"), indent=1)
# figE1 with dashed scale lines
N = json.load(open(f"{W}/numbers_E.json")); X = N["exploratory"]["curves"]; T = N["table"]
COL = {"a": "#1f77b4", "b": "#d62728", "c": "#2ca02c"}; ARMS = {"a": "real-click RM", "b": "simulated audience", "c": "real-click RM + NLI"}
SC = {"scale_E1": "#9467bd", "scale_E2": "#ff7f0e"}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for a in "abc":
    st = sorted(X[a], key=int)
    ax[0].plot([int(s) for s in st], [X[a][s]["fabricated"] * 100 for s in st], "-o", ms=3, c=COL[a], label=f"({a}) {ARMS[a]}, Qwen2.5-3B")
    ax[1].plot([int(s) for s in st], [X[a][s]["numbers_only"] * 100 for s in st], "-o", ms=3, c=COL[a])
for run, r in res.items():
    st = sorted(r["curve"])
    ax[0].plot(st, [r["curve"][s]["fabricated"] * 100 for s in st], "--s", ms=3, c=SC[run], label=LABEL[run] + " (exploratory)")
    ax[1].plot(st, [r["curve"][s]["numbers_only"] * 100 for s in st], "--s", ms=3, c=SC[run])
for k, (col, ttl) in enumerate((("fabricated", "Fabrication rate (%)"), ("numbers_only", "Unsupported-number rate (%)"))):
    ax[k].axhline(T["base"][col] * 100, c="grey", ls="--", lw=1, label="base Qwen2.5-3B (test, final eval)" if k == 0 else None)
    ax[k].axhline(T["human"][col] * 100, c="black", ls=":", lw=1, label="human headline" if k == 0 else None)
    ax[k].set_xlabel("GRPO step (seed 0)"); ax[k].set_title(ttl, fontsize=10)
ax[0].legend(fontsize=6.5); fig.tight_layout()
for e in ("pdf", "png"): fig.savefig(f"{W}/figs/figE1_fabrication_vs_step.{e}", dpi=200)
print("SCALE_SUMMARY", json.dumps({r: (v["prediction"], v["collapsed"]) for r, v in res.items()}))
