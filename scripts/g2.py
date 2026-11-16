# Block E: G2 check. 300 selection articles x 2 at T=0.7 from the adapter vs the base, scored by the arm's own reward (z units).
import argparse, json, numpy as np, pandas as pd, torch
from common import W, assert_a100
import gen
from rewards import ArmReward
ap = argparse.ArgumentParser(); ap.add_argument("--arm"); ap.add_argument("--adapter"); ap.add_argument("--out"); a = ap.parse_args()
assert_a100()
df = gen.run("selection", 2, 0.7, a.adapter, a.out + ".gen.parquet", seed=0, tag=f"arm{a.arm}")
base = pd.read_parquet(f"{W}/runs/gen/base_selection_t07.parquet")
arts = pd.read_parquet(f"{W}/data/articles.parquet").set_index("id").article_trunc
R = ArmReward(a.arm, a.out + ".rewardlog.jsonl")
assert not R.zs.get("_smoke"), "G2 must use the frozen J1 z-stats, not smoke defaults"
def score(d):
    r, valid, _ = R.score(list(d.headline), [arts[i] for i in d.id]); return np.array(r)
rb, ra = score(base), score(df)
dist = df.groupby("id").headline.nunique().mean()
res = {"arm": a.arm, "adapter": a.adapter, "base_reward": float(rb.mean()), "adapter_reward": float(ra.mean()), "gain": float(ra.mean() - rb.mean()),
       "adapter_valid": float(df.valid.mean()), "base_valid": float(base.valid.mean()), "adapter_distinct_per_article_of2": float(dist),
       "base_reward_validonly": float(rb[base.valid.values].mean()), "adapter_reward_validonly": float(ra[df.valid.values].mean()),
       "examples": list(df.headline.iloc[:12]), "zstats": R.zs}
json.dump(res, open(a.out, "w"), indent=1); print("G2", json.dumps(res, indent=1))
