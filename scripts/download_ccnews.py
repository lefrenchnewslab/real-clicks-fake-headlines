# Block E Stage 0 (login node). cc_news parquet into HF cache; record the revision.
import json, time
from huggingface_hub import snapshot_download, HfApi
rid = "vblagoje/cc_news"; info = HfApi().dataset_info(rid)
for a in range(6):
    try:
        p = snapshot_download(rid, repo_type="dataset", revision=info.sha); break
    except Exception as e: print("retry", e); time.sleep(20)
print("OK", rid, info.sha, p)
json.dump({"repo": rid, "sha": info.sha, "path": p, "date": time.strftime("%F %T")}, open("data/ccnews_source.json", "w"))
