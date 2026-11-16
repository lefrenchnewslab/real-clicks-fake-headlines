# Block E: batched HF generation of headlines (base or LoRA adapter). Also used for Stage 3.
import argparse, json, os, time
import pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import W, model_path, messages, postprocess, is_valid, assert_a100, fix_pad

def load_policy(adapter=None):
    p = model_path("policy")
    tok = AutoTokenizer.from_pretrained(p); tok.padding_side = "left"; fix_pad(tok)
    m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16).cuda().eval()
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter).merge_and_unload().eval()
    return tok, m

@torch.no_grad()
def generate(tok, m, arts, n, temp, seed=0, bs=64, max_new=32):
    torch.manual_seed(seed)
    prompts = [tok.apply_chat_template(messages(a), tokenize=False, add_generation_prompt=True) for a in arts]
    out = []
    for i in range(0, len(prompts), bs):
        b = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = m.generate(**b, do_sample=temp > 0, temperature=temp if temp > 0 else None, top_p=1.0, top_k=0, max_new_tokens=max_new,
                       num_return_sequences=n, use_cache=True, pad_token_id=tok.pad_token_id)
        txt = tok.batch_decode(g[:, b["input_ids"].shape[1]:], skip_special_tokens=True)
        out += txt
    return out   # len = len(arts) * n, grouped per article

def run(split, n, temp, adapter, out, seed=0, limit=None, tag=""):
    arts = pd.read_parquet(f"{W}/data/articles.parquet"); arts = arts[arts.split == split]
    if limit: arts = arts.iloc[:limit]
    tok, m = load_policy(adapter); t0 = time.time()
    raw = generate(tok, m, list(arts.article_trunc), n, temp, seed)
    rows = []
    for j, (aid, art) in enumerate(zip(arts.id, arts.article_trunc)):
        for k in range(n):
            r = raw[j * n + k]; h = postprocess(r)
            rows.append({"id": aid, "sample": k, "raw": r, "headline": h, "valid": is_valid(h), "source": tag or (adapter or "base")})
    df = pd.DataFrame(rows); df.to_parquet(out)
    print(f"GEN {out} n={len(df)} valid={df.valid.mean():.3f} {time.time()-t0:.0f}s", flush=True)
    del m; torch.cuda.empty_cache(); return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--split"); ap.add_argument("--n", type=int); ap.add_argument("--temp", type=float)
    ap.add_argument("--adapter", default=None); ap.add_argument("--out"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(); assert_a100(); run(a.split, a.n, a.temp, a.adapter, a.out, a.seed, a.limit, a.tag)
