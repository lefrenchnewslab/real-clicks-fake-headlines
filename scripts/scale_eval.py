# Block E exploratory scale/family check: per checkpoint (0 = own base, 50..250) generate 300 TEST articles x 2 at T=0.7 and
# compute offline seq-KL on 300 SELECTION articles x 2 at T=1.0 (same procedure as kl.py); then RM-score the eval outputs.
# Resumable; stops cleanly (exit 0) when the job is close to its time limit (JOB_T0 from rl.slurm).
import argparse, glob, json, os, time
import pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from common import W, model_path, messages, postprocess, is_valid, assert_a100, fix_pad
ap = argparse.ArgumentParser(); ap.add_argument("--run"); ap.add_argument("--budget_min", type=float, default=108); a = ap.parse_args()
assert_a100(); T0 = float(os.environ.get("JOB_T0", time.time()))
RUN = f"{W}/runs/grpo/{a.run}"; OUT = f"{W}/runs/scale"; os.makedirs(OUT, exist_ok=True)
if not (os.path.exists(f"{RUN}/DONE") or os.path.exists(f"{RUN}/COLLAPSED")): print("training not finished; eval skipped"); raise SystemExit(0)
def late(): return (time.time() - T0) / 60 > a.budget_min
arts = pd.read_parquet(f"{W}/data/articles.parquet")
te = arts[arts.split == "test"].iloc[:300]; se = arts[arts.split == "selection"]
p = model_path("policy"); tok = AutoTokenizer.from_pretrained(p); tok.padding_side = "left"; fix_pad(tok)
base = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16).cuda().eval()
P_te = [tok.apply_chat_template(messages(x), tokenize=False, add_generation_prompt=True) for x in te.article_trunc]
P_se = [tok.apply_chat_template(messages(x), tokenize=False, add_generation_prompt=True) for x in se.article_trunc]
klf = f"{RUN}/kl.json"; KL = json.load(open(klf)) if os.path.exists(klf) else {}
@torch.no_grad()
def gen(m, P, temp, seed, n=2, bs=50):
    torch.manual_seed(seed); outs, seqs = [], []
    for i in range(0, len(P), bs):
        b = tok(P[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = m.generate(**b, do_sample=True, temperature=temp, top_p=1.0, top_k=0, max_new_tokens=32, num_return_sequences=n, use_cache=True,
                       pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, b["input_ids"].shape[1]:], skip_special_tokens=True); seqs.append((b, g))
    return outs, seqs
@torch.no_grad()
def seq_kl(m, seqs):
    kls = []
    for b, g in seqs:
        L = b["input_ids"].shape[1]; comp = g[:, L:]; mask = torch.ones_like(comp, dtype=torch.float)
        eos = set([tok.eos_token_id, tok.pad_token_id] + ([*m.generation_config.eos_token_id] if isinstance(m.generation_config.eos_token_id, list) else [m.generation_config.eos_token_id]))
        for r in range(comp.shape[0]):
            e = [k for k, t in enumerate(comp[r].tolist()) if t in eos]
            if e: mask[r, e[0] + 1:] = 0
        am = torch.cat([b["attention_mask"].repeat_interleave(2, 0), (mask > 0).long()], 1)
        def lp(mm):
            lo = mm(input_ids=g, attention_mask=am).logits[:, L - 1:-1].float()
            return torch.log_softmax(lo, -1).gather(-1, comp.unsqueeze(-1)).squeeze(-1)
        lpi = lp(m)
        with m.disable_adapter(): lref = lp(m)
        kls += ((lpi - lref) * mask).sum(1).tolist()
    return kls
def save_eval(step, raw):
    rows = [{"id": i, "sample": k, "raw": raw[j * 2 + k], "headline": postprocess(raw[j * 2 + k]), "valid": is_valid(postprocess(raw[j * 2 + k])), "source": f"{a.run}@{step}"}
            for j, i in enumerate(te.id) for k in range(2)]
    pd.DataFrame(rows).to_parquet(f"{OUT}/{a.run}_{step}.parquet")
if not os.path.exists(f"{OUT}/{a.run}_0.parquet"):
    t = time.time(); raw, _ = gen(base, P_te, 0.7, 0); save_eval(0, raw); print(f"EVAL step 0 (base) {time.time()-t:.0f}s", flush=True)
cks = sorted(glob.glob(f"{RUN}/checkpoint-*"), key=lambda q: int(q.rsplit("-", 1)[1])); model = None
cks = [c for c in cks if int(c.rsplit("-", 1)[1]) % 50 == 0 or c == cks[-1]]   # pre-stated steps + final; skip guard-stop checkpoints
for c in cks:
    s = c.rsplit("-", 1)[1]
    if os.path.exists(f"{OUT}/{a.run}_{s}.parquet") and s in KL: continue
    if late(): print("TIME_GUARD: stopping eval, will resume in next job", flush=True); raise SystemExit(0)
    t = time.time()
    if model is None: model = PeftModel.from_pretrained(base, c, adapter_name=f"s{s}")
    else: model.load_adapter(c, adapter_name=f"s{s}")
    model.set_adapter(f"s{s}"); model.eval()
    if not os.path.exists(f"{OUT}/{a.run}_{s}.parquet"): raw, _ = gen(model, P_te, 0.7, 0); save_eval(s, raw)
    if s not in KL:
        _, seqs = gen(model, P_se, 1.0, 1234); k = seq_kl(model, seqs)
        KL[s] = {"kl_seq_mean": sum(k) / len(k), "n": len(k)}; json.dump(KL, open(klf, "w"))
    print(f"EVAL step {s} KL {KL[s]['kl_seq_mean']:.2f} {time.time()-t:.0f}s", flush=True)
    model.delete_adapter(f"s{s}") if len(model.peft_config) > 1 else None
del base, model; torch.cuda.empty_cache()
from rewards import RMScorer
rm = RMScorer()
for f in sorted(glob.glob(f"{OUT}/{a.run}_*.parquet")):
    if f.endswith((".scores.parquet", ".metrics.parquet", ".posthoc.parquet")): continue
    d = pd.read_parquet(f); pd.DataFrame({"rm": rm(list(d.headline.fillna("")))}, index=d.index).to_parquet(f.replace(".parquet", ".scores.parquet"))
open(f"{RUN}/EVAL_DONE", "w").write("ok\n"); print("SCALE_EVAL_DONE", flush=True)
