# Block E: offline sequence-level KL to the base for every saved checkpoint of a run.
# 300 fixed selection articles x 2 samples from the policy (T=1.0); KL_seq = sum over completion tokens of log pi - log pi_ref.
import argparse, glob, json, os, re, time
import pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from common import fix_pad, W, model_path, messages, assert_a100
ap = argparse.ArgumentParser(); ap.add_argument("--run"); ap.add_argument("--force", action="store_true"); a = ap.parse_args()
assert_a100()
OUT = f"{W}/runs/grpo/{a.run}"
if not os.path.exists(f"{OUT}/DONE") and not os.path.exists(f"{OUT}/COLLAPSED") and not a.force:
    print("run not finished; KL skipped"); raise SystemExit(0)
f = f"{OUT}/kl.json"; res = json.load(open(f)) if os.path.exists(f) else {}
cks = sorted(glob.glob(f"{OUT}/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
todo = [c for c in cks if c.rsplit("-", 1)[1] not in res]
print("checkpoints to do", [c.rsplit("/", 1)[1] for c in todo], flush=True)
if not todo: raise SystemExit(0)
arts = pd.read_parquet(f"{W}/data/articles.parquet"); arts = arts[arts.split == "selection"]
p = model_path("policy"); tok = AutoTokenizer.from_pretrained(p); tok.padding_side = "left"; fix_pad(tok)
base = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16).cuda().eval()
prompts = [tok.apply_chat_template(messages(x), tokenize=False, add_generation_prompt=True) for x in arts.article_trunc]
model = None
for c in todo:
    step = c.rsplit("-", 1)[1]; t0 = time.time()
    if model is None: model = PeftModel.from_pretrained(base, c, adapter_name=f"s{step}")
    else: model.load_adapter(c, adapter_name=f"s{step}")
    model.set_adapter(f"s{step}"); model.eval()
    torch.manual_seed(1234); kls, lens = [], []
    with torch.no_grad():
        for i in range(0, len(prompts), 50):
            b = tok(prompts[i:i + 50], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            g = model.generate(**b, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=32, num_return_sequences=2,
                               use_cache=True, pad_token_id=tok.pad_token_id)
            L = b["input_ids"].shape[1]; comp = g[:, L:]
            mask = torch.ones_like(comp, dtype=torch.float)
            for r in range(comp.shape[0]):                       # keep tokens up to and incl. the first EOS
                e = (comp[r] == tok.eos_token_id).nonzero()
                if len(e): mask[r, e[0, 0] + 1:] = 0
            am = torch.cat([b["attention_mask"].repeat_interleave(2, 0), torch.ones_like(comp)], 1)
            def lp(m):
                lo = m(input_ids=g, attention_mask=am).logits[:, L - 1:-1].float()
                return torch.log_softmax(lo, -1).gather(-1, comp.unsqueeze(-1)).squeeze(-1)
            lpi = lp(model)
            with model.disable_adapter(): lref = lp(model)
            kls += ((lpi - lref) * mask).sum(1).tolist(); lens += mask.sum(1).tolist()
    res[step] = {"kl_seq_mean": sum(kls) / len(kls), "n": len(kls), "mean_len": sum(lens) / len(lens), "kl_seq": kls}
    json.dump(res, open(f, "w")); print(f"KL {a.run} step {step}: {res[step]['kl_seq_mean']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    model.delete_adapter(f"s{step}") if len(model.peft_config) > 1 else None
