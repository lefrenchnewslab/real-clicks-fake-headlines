# Block E Stage 3 (CPU): deterministic fabrication + style metrics for generated-headline parquets. Multiprocessing.
import sys, os, glob, json, time
import pandas as pd
from multiprocessing import Pool
from common import W
def init():
    global fabrication, style, ART
    from eval_metrics import fabrication, style
    ART = pd.read_parquet(f"{W}/data/articles.parquet").set_index("id").article_trunc.to_dict()
def work(row):
    i, h = row; a = ART[i]
    f = fabrication(h, a) if h else {"fabricated": 0, "numbers_only": 0, "claim_bearing": 0, "listicle": 0, "n_ent": 0, "n_num": 0, "unsup_ent": [], "unsup_num": []}
    s = style(h or "", a)
    f["unsup_ent"] = json.dumps(f["unsup_ent"]); f["unsup_num"] = json.dumps(f["unsup_num"]); return {**f, **s}
if __name__ == "__main__":
    files = [f for pat in sys.argv[1:] for f in sorted(glob.glob(pat)) if not f.endswith((".scores.parquet", ".metrics.parquet"))]
    with Pool(int(os.environ.get("NPROC", "38")), initializer=init) as pool:
        for f in files:
            out = f.replace(".parquet", ".metrics.parquet")
            if os.path.exists(out) and os.path.getmtime(out) > os.path.getmtime(f): continue
            d = pd.read_parquet(f); t = time.time()
            m = pd.DataFrame(pool.map(work, list(zip(d.id, d.headline.fillna(""))), chunksize=16), index=d.index)
            m.to_parquet(out); print(f"METRICS {os.path.basename(f)} n={len(d)} fab={m.fabricated.mean():.3f} {time.time()-t:.0f}s", flush=True)
    print("METRICS_DONE")
