# Block E Stage 3 generation. Modes: final (test 1000x4, T=0.7), curve (seed-0 checkpoints, 300 test x 2), bon (base 300 test x 16), human.
import argparse, glob, os, pandas as pd
from common import W, assert_a100, postprocess, is_valid
import gen
ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--runs", default=""); a = ap.parse_args(); assert_a100()
G = f"{W}/runs/stage3"; os.makedirs(G, exist_ok=True)
def ck(run, step="final"):
    cks = sorted(glob.glob(f"{W}/runs/grpo/{run}/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
    return cks if step == "all" else cks[-1]
if a.mode == "human":
    arts = pd.read_parquet(f"{W}/data/articles.parquet"); t = arts[arts.split == "test"]
    pd.DataFrame({"id": t.id, "sample": 0, "raw": t.title, "headline": t.title.map(postprocess), "valid": t.title.map(postprocess).map(is_valid),
                  "source": "human"}).to_parquet(f"{G}/final_human.parquet")
if a.mode == "final":
    for r in a.runs.split(","):
        out = f"{G}/final_{r}.parquet"
        if os.path.exists(out): continue
        gen.run("test", 4, 0.7, None if r == "base" else ck(r), out, seed=0, tag=r)
if a.mode == "curve":
    for r in a.runs.split(","):
        for c in ck(r, "all"):
            s = c.rsplit("-", 1)[1]; out = f"{G}/curve_{r}_{s}.parquet"
            if not os.path.exists(out): gen.run("test", 2, 0.7, c, out, seed=0, limit=300, tag=f"{r}@{s}")
    if not os.path.exists(f"{G}/curve_base_0.parquet"): gen.run("test", 2, 0.7, None, f"{G}/curve_base_0.parquet", seed=0, limit=300, tag="base@0")
if a.mode == "bon":
    if not os.path.exists(f"{G}/bon16_base.parquet"): gen.run("test", 16, 0.7, None, f"{G}/bon16_base.parquet", seed=0, limit=300, tag="bon16")
print("GEN3_DONE")
