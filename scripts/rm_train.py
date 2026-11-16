# Block E J1a: real-click RM (DeBERTa-v3-base, Bradley-Terry) + G1 baselines. Selection data only for choices.
import json, os, re, sys, time, random, math
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from common import W, model_path, assert_a100
from eval_metrics import vader, clickbait_count

gpu = assert_a100()
torch.manual_seed(0); random.seed(0); np.random.seed(0)
OUT = f"{W}/runs/rm"; os.makedirs(OUT, exist_ok=True)
tr = pd.read_parquet(f"{W}/data/pairs_train.parquet"); se = pd.read_parquet(f"{W}/data/pairs_selection.parquet")
print("train", len(tr), "selection", len(se), "clear", int(se.clear.sum()), flush=True)
res = {"gpu": gpu, "n_train": len(tr), "n_sel": len(se), "n_sel_clear": int(se.clear.sum())}

# ---------- feature + TF-IDF BT baselines (fit on train pairs, randomised orientation) ----------
def feats(h):
    return [len(h), len(h.split()), int(bool(re.search(r"\d", h))), int("?" in h), int("!" in h),
            int(bool(re.search(r"\byou(r|'re)?\b", h, re.I))), int(bool(re.match(r"\s*(this|these|that|those|here)\b", h, re.I))),
            clickbait_count(h), vader(h)]
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
def orient(df):
    a = np.where(df.swap, df.loser, df.winner); b = np.where(df.swap, df.winner, df.loser); y = (~df.swap.values).astype(int)
    return list(a), list(b), y
t0 = time.time()
A, B, y = orient(tr); Fa = np.array([feats(h) for h in A]); Fb = np.array([feats(h) for h in B])
sc = StandardScaler().fit(np.vstack([Fa, Fb]))
lr = LogisticRegression(max_iter=2000, fit_intercept=False).fit(sc.transform(Fa) - sc.transform(Fb), y)
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(A + B)
lt = LogisticRegression(max_iter=2000, C=1.0, fit_intercept=False).fit(tf.transform(A) - tf.transform(B), y)
def base_acc(df, kind):
    a, b, yy = orient(df)
    if kind == "feat": X = sc.transform(np.array([feats(h) for h in a])) - sc.transform(np.array([feats(h) for h in b])); p = lr.predict(X)
    else: p = lt.predict(tf.transform(a) - tf.transform(b))
    return float((p == yy).mean())
lenm = se[(se.winner.str.split().str.len() - se.loser.str.split().str.len()).abs() <= 1]
for name, df in [("sel_all", se), ("sel_clear", se[se.clear]), ("sel_lenmatched", lenm), ("sel_clear_lenmatched", lenm[lenm.clear])]:
    res[f"feat_{name}"] = base_acc(df, "feat"); res[f"tfidf_{name}"] = base_acc(df, "tfidf"); res[f"n_{name}"] = len(df)
res["feature_coefs"] = dict(zip(["chars", "words", "has_number", "question", "exclaim", "you", "demonstrative", "clickbait", "vader"], lr.coef_[0].round(4).tolist()))
print("baselines", {k: v for k, v in res.items() if k.startswith(("feat_", "tfidf_"))}, f"{time.time()-t0:.0f}s", flush=True)

# ---------- RM ----------
p = model_path("rm_backbone")
tok = AutoTokenizer.from_pretrained(p)
m = AutoModelForSequenceClassification.from_pretrained(p, num_labels=1, dtype=torch.float32).cuda()   # explicit fp32: transformers 5 loads this .bin as fp16 by default -> NaN (job 189616)
opt = torch.optim.AdamW(m.parameters(), lr=2e-5, weight_decay=0.01)
BS, EPOCHS = 64, 2
steps_total = EPOCHS * math.ceil(len(tr) / BS)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / (0.06 * steps_total)) * max(0.0, (steps_total - s) / (steps_total * 0.94)))
@torch.no_grad()
def score(hs, bs=512):
    m.eval(); out = []
    for i in range(0, len(hs), bs):
        b = tok(hs[i:i + bs], truncation=True, max_length=48, padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16): out += m(**b).logits[:, 0].float().tolist()
    m.train(); return np.array(out)
def rm_acc(df):
    if len(df) == 0: return float("nan")
    return float((score(list(df.winner)) > score(list(df.loser))).mean())
best, hist, step = -1, [], 0
evals = set(int(steps_total * f) for f in (0.25, 0.5, 0.75, 1.0)) | {steps_total}
m.train(); t0 = time.time()
for ep in range(EPOCHS):
    idx = np.random.RandomState(ep).permutation(len(tr))
    for i in range(0, len(idx), BS):
        b = tr.iloc[idx[i:i + BS]]
        enc = tok(list(b.winner) + list(b.loser), truncation=True, max_length=48, padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16): s = m(**enc).logits[:, 0].float()
        n = len(b); loss = -F.logsigmoid(s[:n] - s[n:]).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(); step += 1
        if not torch.isfinite(loss): raise SystemExit(f"ABORT: non-finite RM loss at step {step}")
        if step % 100 == 0: print(f"step {step}/{steps_total} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
        if step in evals:
            a_clear, a_all = rm_acc(se[se.clear]), rm_acc(se)
            hist.append({"step": step, "epoch": ep, "sel_clear": a_clear, "sel_all": a_all}); print("EVAL", hist[-1], flush=True)
            if a_clear > best:
                best = a_clear; m.save_pretrained(f"{OUT}/best"); tok.save_pretrained(f"{OUT}/best"); res["best_step"] = step
res["rm_history"] = hist; res["rm_train_s"] = time.time() - t0
# final selection numbers from the chosen checkpoint
m = AutoModelForSequenceClassification.from_pretrained(f"{OUT}/best", num_labels=1, dtype=torch.float32).cuda()
for name, df in [("sel_all", se), ("sel_clear", se[se.clear]), ("sel_lenmatched", lenm), ("sel_clear_lenmatched", lenm[lenm.clear])]:
    res[f"rm_{name}"] = rm_acc(df)
bestbase = max(res["feat_sel_clear"], 0)
res["G1_pass"] = bool(res["rm_sel_clear"] >= 0.60 and res["rm_sel_clear"] >= bestbase + 0.03)
res["G1_rule"] = "rm_sel_clear >= 0.60 AND >= feat_sel_clear + 0.03"
# domain-shift check (exploratory): RM score vs length on cc_news human titles (selection articles only)
arts = pd.read_parquet(f"{W}/data/articles.parquet"); sel = arts[arts.split == "selection"]
hs = list(sel.title); s = score(hs); L = np.array([len(h.split()) for h in hs])
res["ccnews_title_rm_mean"] = float(s.mean()); res["ccnews_title_rm_sd"] = float(s.std()); res["ccnews_title_rm_len_corr"] = float(np.corrcoef(s, L)[0, 1])
sU = score(list(se.winner) + list(se.loser)); LU = np.array([len(h.split()) for h in list(se.winner) + list(se.loser)])
res["upworthy_sel_rm_mean"] = float(sU.mean()); res["upworthy_sel_rm_len_corr"] = float(np.corrcoef(sU, LU)[0, 1])
json.dump(res, open(f"{OUT}/rm_results.json", "w"), indent=1)
print("RESULT", json.dumps(res, indent=1), flush=True)
