# Block E post-hoc P1: dump the OLMo probe pipeline exactly as score_gpu.Probe ran it, on 20 items.
import json, pandas as pd, torch
from common import W, assert_a100
import ast
PROBE_Q = next(ast.literal_eval(n.value) for n in ast.parse(open(__file__.replace("probe_debug.py", "score_gpu.py")).read()).body
               if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "PROBE_Q")   # identical string to score_gpu.py
assert_a100()
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import model_path
G = f"{W}/runs/stage3"
arts = pd.read_parquet(f"{W}/data/articles.parquet").set_index("id").article_trunc
a = pd.read_parquet(f"{G}/final_main_a_s1.parquet").join(pd.read_parquet(f"{G}/final_main_a_s1.metrics.parquet")[["fabricated"]])
b = pd.read_parquet(f"{G}/final_base.parquet").join(pd.read_parquet(f"{G}/final_base.metrics.parquet")[["fabricated"]])
items = [("arm_a_template", r) for _, r in a.sample(10, random_state=0).iterrows()] + \
        [("base_unflagged", r) for _, r in b[b.fabricated == 0].sample(10, random_state=0).iterrows()]
p = model_path("probe"); tok = AutoTokenizer.from_pretrained(p); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16).cuda().eval()
ids = lambda ws: sorted({t[0] for w in ws for t in [tok.encode(w, add_special_tokens=False)] if len(t) == 1})
Y, N = ids(["Yes", " Yes", "yes", " yes"]), ids(["No", " No", "no", " no"])
info = {"yes_ids": Y, "no_ids": N, "yes_dec": [tok.decode([i]) for i in Y], "no_dec": [tok.decode([i]) for i in N],
        "bos_token": tok.bos_token, "bos_id": tok.bos_token_id, "eos_token": tok.eos_token, "pad_token": tok.pad_token, "pad_id": tok.pad_token_id,
        "chat_template_head": (tok.chat_template or "")[:400], "model_max_len": tok.model_max_length, "config_max_pos": m.config.max_position_embeddings}
print(json.dumps(info, indent=1), flush=True)
P = [tok.apply_chat_template([{"role": "user", "content": PROBE_Q.format(a=arts[r.id], h=r.headline)}], tokenize=False, add_generation_prompt=True) for _, r in items]
out = []
with torch.no_grad():
    b_ = tok(P, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")   # exactly as in score_gpu (batch of 20)
    lo = m(**b_).logits[:, -1].float(); lp = torch.log_softmax(lo, -1)
    for k, (src, r) in enumerate(items):
        am = b_["attention_mask"][k]; npad = int((am == 0).sum()); seq = b_["input_ids"][k][am.bool()]
        top = torch.topk(lp[k].exp(), 5)
        single = tok(P[k], return_tensors="pt", add_special_tokens=False).to("cuda")
        g = m.generate(**single, max_new_tokens=10, do_sample=False)
        lp1 = torch.log_softmax(m(**single).logits[0, -1].float(), -1)
        withbos = tok(P[k], return_tensors="pt", add_special_tokens=True).to("cuda")
        lpb = torch.log_softmax(m(**withbos).logits[0, -1].float(), -1)
        py = lambda L: float(torch.sigmoid(torch.logsumexp(L[Y], -1) - torch.logsumexp(L[N], -1)))
        out.append({"src": src, "headline": r.headline, "article_chars": len(arts[r.id]), "prompt_tokens": int(seq.numel()), "left_pad_tokens": npad,
                    "first_tokens": tok.decode(seq[:8]), "last_tokens": tok.decode(seq[-25:]), "question_present_in_ids": "Answer Yes or No" in tok.decode(seq),
                    "article_end_present": arts[r.id][-40:].strip()[:30] in tok.decode(seq),
                    "top5": [(tok.decode([int(i)]), round(float(v), 4)) for v, i in zip(top.values, top.indices)],
                    "p_yes_batched": py(lp[k]), "p_yes_single": py(lp1), "p_yes_with_add_special_tokens": py(lpb),
                    "greedy10": tok.decode(g[0, single["input_ids"].shape[1]:]), "starts_with_bos": bool(seq[0].item() == tok.bos_token_id) if tok.bos_token_id is not None else None,
                    "ids_with_special_differ": bool(withbos["input_ids"].shape[1] != single["input_ids"].shape[1])})
        print(json.dumps(out[-1]), flush=True)
json.dump({"info": info, "items": out, "rendered_prompt_example": P[0], "rendered_prompt_example_base": P[10]}, open(f"{W}/runs/stage3/probe_debug.json", "w"), indent=1)
print("PROBE_DEBUG_DONE")
