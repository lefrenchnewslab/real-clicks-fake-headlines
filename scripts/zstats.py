# Block E J1b: base-policy headlines on the 300 selection articles x 2 (T=0.7) -> RM and judge z-stats, frozen.
import json, os, numpy as np, pandas as pd, torch
from common import W, assert_a100
import gen
from rewards import RMScorer, JudgeScorer
assert_a100()
f = f"{W}/data/zstats.json"
if os.path.exists(f): raise SystemExit("zstats.json exists (frozen); refusing to overwrite")
os.makedirs(f"{W}/runs/gen", exist_ok=True)
df = gen.run("selection", 2, 0.7, None, f"{W}/runs/gen/base_selection_t07.parquet", seed=0, tag="base")
v = df[df.valid]; hs = list(v.headline)
rm = RMScorer(); s_rm = np.array(rm(hs)); del rm; torch.cuda.empty_cache()
jd = JudgeScorer(); s_j = np.array(jd(hs)); del jd; torch.cuda.empty_cache()
z = {"rm": {"mean": float(s_rm.mean()), "sd": float(s_rm.std())}, "judge": {"mean": float(s_j.mean()), "sd": float(s_j.std())},
     "n_valid": len(hs), "n": len(df), "source": "runs/gen/base_selection_t07.parquet (valid only)",
     "judge_min": float(s_j.min()), "judge_max": float(s_j.max()), "judge_q": np.quantile(s_j, [.05, .25, .5, .75, .95]).tolist()}
json.dump(z, open(f, "w"), indent=1); print("ZSTATS", json.dumps(z, indent=1))
