# Block E reward models. All headline-only except the arm (c) NLI penalty. bf16, eval(), no_grad, batched.
import json, os, math, time
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from common import W, model_path, postprocess, is_valid

DEV = "cuda"
INVALID = -3.0

class RMScorer:
    """DeBERTa-v3-base + scalar head (Bradley-Terry). Headline only, max 48 tokens."""
    def __init__(self, path=None):
        path = path or os.environ.get("RM_PATH", f"{W}/runs/rm/best")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.m = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1, dtype=torch.bfloat16).to(DEV).eval()
    @torch.no_grad()
    def __call__(self, hs, bs=256):
        out = []
        for i in range(0, len(hs), bs):
            b = self.tok(hs[i:i + bs], truncation=True, max_length=48, padding=True, return_tensors="pt").to(DEV)
            out += self.m(**b).logits[:, 0].float().tolist()
        return out

class JudgeScorer:
    """Qwen2.5-7B-Instruct simulated audience: mean over 4 personas of P(Yes)/(P(Yes)+P(No)) on the next token."""
    def __init__(self):
        p = model_path("judge")
        self.tok = AutoTokenizer.from_pretrained(p); self.tok.padding_side = "left"
        self.m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16).to(DEV).eval()
        cfg = json.load(open(f"{W}/scripts/personas.json")); self.tpl = cfg["template"]; self.personas = cfg["personas"]
        def ids(words):
            s = set()
            for w in words:
                t = self.tok.encode(w, add_special_tokens=False)
                if len(t) == 1: s.add(t[0])
            return sorted(s)
        self.yes, self.no = ids(["Yes", " Yes", "yes", " yes", "YES"]), ids(["No", " No", "no", " no", "NO"])
        print("judge yes/no token ids", self.yes, self.no, [self.tok.decode([i]) for i in self.yes + self.no], flush=True)
    def prompts(self, h):
        return [self.tok.apply_chat_template([{"role": "user", "content": self.tpl.format(persona=p, h=h)}], tokenize=False,
                                             add_generation_prompt=True) for p in self.personas]
    @torch.no_grad()
    def __call__(self, hs, bs=256):
        P = [p for h in hs for p in self.prompts(h)]
        sc = []
        for i in range(0, len(P), bs):
            b = self.tok(P[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
            lp = torch.log_softmax(self.m(**b).logits[:, -1].float(), -1)
            y = torch.logsumexp(lp[:, self.yes], -1); n = torch.logsumexp(lp[:, self.no], -1)
            sc += torch.sigmoid(y - n).tolist()
        k = len(self.personas)
        return [sum(sc[j * k:(j + 1) * k]) / k for j in range(len(hs))]

class NLIScorer:
    """Max over 400-token article chunks of P(entailment) (premise=chunk, hypothesis=headline); also max P(contradiction)."""
    def __init__(self, role):
        p = model_path(role)
        self.tok = AutoTokenizer.from_pretrained(p)
        self.m = AutoModelForSequenceClassification.from_pretrained(p, dtype=torch.bfloat16).to(DEV).eval()
        lab = {v.lower(): int(k) for k, v in self.m.config.id2label.items()}
        self.ent = lab["entailment"]; self.con = lab["contradiction"]
        print(role, "labels", self.m.config.id2label, flush=True)
        self._chunks = {}
    def chunks(self, art):
        if art not in self._chunks:
            ids = self.tok(art, add_special_tokens=False)["input_ids"]
            self._chunks[art] = [self.tok.decode(ids[i:i + 400]) for i in range(0, max(1, len(ids)), 400)] or [art]
            if len(self._chunks) > 20000: self._chunks.clear()
        return self._chunks[art]
    @torch.no_grad()
    def __call__(self, hs, arts, bs=64):
        pairs, owner = [], []
        for j, (h, a) in enumerate(zip(hs, arts)):
            for c in self.chunks(a): pairs.append((c, h)); owner.append(j)
        ent, con = [0.0] * len(hs), [0.0] * len(hs)
        for i in range(0, len(pairs), bs):
            b = self.tok([p[0] for p in pairs[i:i + bs]], [p[1] for p in pairs[i:i + bs]], truncation="only_first", max_length=512,
                         padding=True, return_tensors="pt").to(DEV)
            pr = torch.softmax(self.m(**b).logits.float(), -1)
            for k, j in enumerate(owner[i:i + bs]):
                ent[j] = max(ent[j], pr[k, self.ent].item()); con[j] = max(con[j], pr[k, self.con].item())
        return ent, con

SMOKE_DEFAULTS = {"rm": {"mean": 0.0, "sd": 1.0}, "judge": {"mean": 0.0, "sd": 1.0}, "_smoke": True}
def zstats():
    f = f"{W}/data/zstats.json"
    if os.path.exists(f): z = json.load(open(f))
    elif os.environ.get("BLOCKE_SMOKE") == "1": z = SMOKE_DEFAULTS
    else: raise SystemExit(f"ABORT: {f} missing (frozen z-stats from J1 required; only BLOCKE_SMOKE=1 may use defaults)")
    print(f"ZSTATS_LOADED rm mean={z['rm']['mean']:.4f} sd={z['rm']['sd']:.4f} judge mean={z['judge']['mean']:.4f} sd={z['judge']['sd']:.4f}"
          f" smoke={bool(z.get('_smoke'))}", flush=True)
    return z

class ArmReward:
    """Callable reward for TRL GRPO + side logging (valid rate, distinct per group, raw stats, timing)."""
    def __init__(self, arm, log_path, lam=2.0):
        self.arm = arm; self.log = open(log_path, "a"); self.lam = lam; self.t = 0.0
        self.__name__ = f"reward_{arm}"
        zs = zstats(); self.zs = zs
        self.rm = RMScorer() if arm in ("a", "c") else None
        self.judge = JudgeScorer() if arm == "b" else None
        self.nli = NLIScorer("train_nli") if arm == "c" else None
        self.calls = 0
    def score(self, hs, arts):
        """Reward per postprocessed headline (shaping applied)."""
        valid = [is_valid(h) for h in hs]
        idx = [i for i, v in enumerate(valid) if v]
        r = [INVALID] * len(hs); extra = {}
        if idx:
            vh = [hs[i] for i in idx]
            if self.arm in ("a", "c"):
                raw = self.rm(vh); z = [(x - self.zs["rm"]["mean"]) / self.zs["rm"]["sd"] for x in raw]
            else:
                raw = self.judge(vh); z = [(x - self.zs["judge"]["mean"]) / self.zs["judge"]["sd"] for x in raw]
            if self.arm == "c":
                ent, _ = self.nli(vh, [arts[i] for i in idx])
                z = [zz - self.lam * (1 - e) for zz, e in zip(z, ent)]
                extra["p_entail_mean"] = sum(ent) / len(ent)
            extra["raw_mean"] = sum(raw) / len(raw)
            for i, v in zip(idx, z): r[i] = v
        return r, valid, extra
    def __call__(self, prompts=None, completions=None, article=None, **kw):
        t0 = time.time()
        hs = [postprocess(c[0]["content"] if isinstance(c, list) else c) for c in completions]
        r, valid, extra = self.score(hs, article)
        # groups: consecutive completions sharing the same article
        groups = {}
        for h, a, rr in zip(hs, article, r): groups.setdefault(a, []).append((h, rr))
        distinct = [len(set(x[0] for x in g)) for g in groups.values()]
        sds = []
        for g in groups.values():
            v = [x[1] for x in g]; m = sum(v) / len(v); sds.append(math.sqrt(sum((x - m) ** 2 for x in v) / len(v)))
        step = getattr(self, "cur_step", None)
        self.t = time.time() - t0; self.calls += 1
        rec = {"call": self.calls, "step": step, "n": len(hs), "n_groups": len(groups), "valid_rate": sum(valid) / len(valid),
               "distinct_per_group": sum(distinct) / len(distinct), "distinct_per_group_min": min(distinct), "reward_mean": sum(r) / len(r),
               "reward_sd_within": sum(sds) / len(sds), "mean_words": sum(len(h.split()) for h in hs) / len(hs),
               "reward_time_s": round(self.t, 3), "examples": hs[:3], **extra}
        self.last = rec; self.log.write(json.dumps(rec) + "\n"); self.log.flush()
        return r
