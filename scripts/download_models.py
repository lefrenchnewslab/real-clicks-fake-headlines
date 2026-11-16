# Block E Stage 0 (login node). Downloads models into HF_HOME and records revision hashes.
import json, os, time, sys
from huggingface_hub import snapshot_download, HfApi
api = HfApi()
M = {"policy": "Qwen/Qwen2.5-3B-Instruct", "judge": "Qwen/Qwen2.5-7B-Instruct", "rm_backbone": "microsoft/deberta-v3-base",
     "train_nli": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "eval_nli": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
     "spacy": "spacy/en_core_web_sm", "probe": "allenai/OLMo-2-1124-7B-Instruct"}
out = {}
for role, rid in M.items():
    for attempt in range(6):
        try:
            info = api.model_info(rid)
            pats = None if role == "spacy" else ["*.json", "*.safetensors", "*.model", "*.txt", "*.py", "*.spm", "*.tiktoken", "merges.txt", "vocab*"]
            p = snapshot_download(rid, revision=info.sha, allow_patterns=pats)
            out[role] = {"repo": rid, "sha": info.sha, "path": p, "date": time.strftime("%F %T")}
            print("OK", role, rid, info.sha, p, flush=True); break
        except Exception as e:
            print("RETRY", role, rid, attempt, repr(e)[:300], flush=True); time.sleep(20 * (attempt + 1))
    else:
        print("FAILED", role, rid, flush=True)
json.dump(out, open("data/models.json", "w"), indent=1)
