# Block E Stage 3: score generated headlines (parquet with id, headline) with RM, judge, train/eval NLI and the OLMo probe.
import argparse, json, os, time
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import W, model_path, assert_a100
from rewards import RMScorer, JudgeScorer, NLIScorer
ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--what", default="rm,judge,eval_nli,train_nli,probe")
a = ap.parse_args(); assert_a100(); what = a.what.split(",")
arts = pd.read_parquet(f"{W}/data/articles.parquet").set_index("id").article_trunc
D = {f: pd.read_parquet(f) for f in a.files}
S = {f: pd.read_parquet(f.replace(".parquet", ".scores.parquet")) if os.path.exists(f.replace(".parquet", ".scores.parquet")) else pd.DataFrame(index=D[f].index) for f in a.files}
def run(name, fn):
    m = fn()
    for f, d in D.items():
        if name in S[f].columns or (name + "_ent") in S[f].columns: continue
        t = time.time(); hs = list(d.headline.fillna("")); ar = [arts[i] for i in d.id]
        out = m(hs, ar)
        if isinstance(out, dict):
            for k, v in out.items(): S[f][k] = v
        else: S[f][name] = out
        S[f].to_parquet(f.replace(".parquet", ".scores.parquet")); print(f"{name} {os.path.basename(f)} n={len(d)} {time.time()-t:.0f}s", flush=True)
    del m; torch.cuda.empty_cache()
if "rm" in what: run("rm", lambda: (lambda r: (lambda hs, ar: r(hs)))(RMScorer()))
if "judge" in what: run("judge", lambda: (lambda j: (lambda hs, ar: j(hs)))(JudgeScorer()))
for role in ("eval_nli", "train_nli"):
    if role in what:
        def mk(role=role):
            n = NLIScorer(role)
            def f(hs, ar):
                e, c = n(hs, ar); return {f"{role}_ent": e, f"{role}_con": c}
            return f
        run(role, mk)
PROBE_Q = ("Here is a news article and a headline written for it.\n\nARTICLE:\n{a}\n\nHEADLINE: {h}\n\n"
           "Does the headline introduce any names, numbers, statistics, or factual claims that are not explicitly contained in the article? Answer Yes or No.")
class Probe:
    def __init__(self):
        p = model_path("probe"); self.tok = AutoTokenizer.from_pretrained(p); self.tok.padding_side = "left"
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        self.m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16).cuda().eval()
        ids = lambda ws: sorted({t[0] for w in ws for t in [self.tok.encode(w, add_special_tokens=False)] if len(t) == 1})
        self.y, self.n = ids(["Yes", " Yes", "yes", " yes"]), ids(["No", " No", "no", " no"])
        print("probe yes/no ids", self.y, self.n, [self.tok.decode([i]) for i in self.y + self.n], flush=True)
    @torch.no_grad()
    def __call__(self, hs, ar, bs=32):
        P = [self.tok.apply_chat_template([{"role": "user", "content": PROBE_Q.format(a=x, h=h)}], tokenize=False, add_generation_prompt=True)
             for h, x in zip(hs, ar)]
        out = []
        for i in range(0, len(P), bs):
            b = self.tok(P[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            lp = torch.log_softmax(self.m(**b).logits[:, -1].float(), -1)
            out += torch.sigmoid(torch.logsumexp(lp[:, self.y], -1) - torch.logsumexp(lp[:, self.n], -1)).tolist()
        return {"probe_pyes": out}
if "probe" in what: run("probe", Probe)
print("SCORE_DONE")
