# Block E Stage 0: cc_news articles, truncated to 450 Qwen tokens; plus a casing lexicon from NON-sampled articles.
# Streams parquet files (the login node killed an all-in-memory version). Login node, CPU.
import glob, json, re, random, collections
import pandas as pd, pyarrow.parquet as pq
from transformers import AutoTokenizer
import os
W = os.environ.get("RLW", os.path.expandvars("$WORK_ROOT/rlhead"))   # set RLW, or WORK_ROOT (parent of rlhead/)
src = json.load(open(f"{W}/data/ccnews_source.json"))
SRC = os.path.expandvars(src["path"])
if not os.path.isdir(SRC):   # release: fetch the same revision of the public dataset from the Hugging Face Hub
    from huggingface_hub import snapshot_download
    SRC = snapshot_download(src["repo"], repo_type="dataset", revision=src["sha"])
files = sorted(glob.glob(SRC + "/**/*.parquet", recursive=True)); print(len(files), "parquet files", flush=True)
EN = {"the", "and", "of", "to", "in", "is", "that", "for", "it", "with"}
def english(t):
    t = t[:2000]; w = re.findall(r"[a-z]+", t.lower()); asc = sum(ord(c) < 128 for c in t) / max(1, len(t))
    return asc > 0.97 and sum(x in EN for x in w) / max(1, len(w)) > 0.08
log = {"rows_in": 0, "after_length": 0, "after_english": 0}
meta = []
for fi, f in enumerate(files):
    pf = pq.ParquetFile(f)
    for rg in range(pf.num_row_groups):
        d = pf.read_row_group(rg, columns=["title", "text", "domain"]).to_pandas()
        log["rows_in"] += len(d)
        d["title"] = d.title.fillna("").str.strip(); d["text"] = d.text.fillna("").str.strip()
        tw = d.title.str.split().str.len(); aw = d.text.str.split().str.len()
        d = d[(tw >= 4) & (tw <= 20) & (aw >= 150) & (aw <= 700)]; log["after_length"] += len(d)
        ok = [english(a) and sum(ord(c) < 128 for c in t) / max(1, len(t)) > 0.95 for a, t in zip(d.text, d.title)]
        d = d[ok]
        log["after_english"] += len(d)
        for i, r in d.iterrows():
            meta.append((fi, rg, i, r.domain, re.sub(r"\W+", " ", r.title.lower()).strip(), r.text[:200].lower()))
    print("file", fi, len(meta), flush=True)
M = pd.DataFrame(meta, columns=["fi", "rg", "row", "domain", "tkey", "bkey"])
M = M.drop_duplicates("tkey").drop_duplicates("bkey").reset_index(drop=True); log["after_dedup"] = len(M)
N = 4300; cap = int(0.03 * N)
idx = list(M.index); random.Random(0).shuffle(idx)
cnt = collections.Counter(); pick = []
for i in idx:
    dom = M.at[i, "domain"]
    if cnt[dom] >= cap: continue
    cnt[dom] += 1; pick.append(i)
    if len(pick) == N: break
lex_pool = [i for i in idx if i not in set(pick)][:40000]       # casing lexicon: non-sampled articles only
need = collections.defaultdict(list)
for k, i in enumerate(pick): need[(M.at[i, "fi"], M.at[i, "rg"])].append(("pick", k, M.at[i, "row"]))
for i in lex_pool: need[(M.at[i, "fi"], M.at[i, "rg"])].append(("lex", -1, M.at[i, "row"]))
rows = [None] * N; cap_mid = collections.Counter(); tot_mid = collections.Counter()
WORD = re.compile(r"(?<=[a-z,;] )([A-Za-z][a-z]+)\b")              # word preceded by a lowercase word/comma = mid-sentence
for (fi, rg), items in need.items():
    d = pq.ParquetFile(files[fi]).read_row_group(rg, columns=["title", "text", "domain", "date", "url"]).to_pandas()
    for kind, k, row in items:
        r = d.loc[row]
        if kind == "pick": rows[k] = {"title": r.title.strip(), "text": r.text.strip(), "domain": r.domain, "date": str(r.date), "url": r.url}
        else:
            for m in WORD.finditer(r.text):
                w = m.group(1); lw = w.lower(); tot_mid[lw] += 1
                if w[0].isupper(): cap_mid[lw] += 1
s = pd.DataFrame(rows)
pol = json.load(open(f"{W}/data/models.json"))["policy"]; pp = os.path.expandvars(pol["path"])
tok = AutoTokenizer.from_pretrained(pp) if os.path.isdir(pp) else AutoTokenizer.from_pretrained(pol["repo"], revision=pol["sha"])
def trunc(t):
    ids = tok(t, add_special_tokens=False)["input_ids"]; return tok.decode(ids[:450]).strip(), len(ids)
tt = s.text.map(trunc); s["article_trunc"] = [x[0] for x in tt]; s["n_tok_full"] = [x[1] for x in tt]
s["id"] = ["cc%05d" % k for k in range(N)]
s["split"] = ["train"] * 3000 + ["selection"] * 300 + ["test"] * 1000
s = s[["id", "split", "title", "article_trunc", "text", "domain", "date", "url", "n_tok_full"]]
s.to_parquet(f"{W}/data/articles.parquet")
for k in ["train", "selection", "test"]: open(f"{W}/data/ids_{k}.txt", "w").write("\n".join(s[s.split == k].id))
lex = {w: [cap_mid[w], n] for w, n in tot_mid.items() if n >= 5}
json.dump(lex, open(f"{W}/data/casing_lexicon.json", "w"))
log.update({"sampled": len(s), "domains": s.domain.nunique(), "max_domain_share": max(cnt.values()) / N,
            "truncated_share": float((s.n_tok_full > 450).mean()), "lexicon_articles": len(lex_pool), "lexicon_words": len(lex), "source": src})
json.dump(log, open(f"{W}/data/articles_log.json", "w"), indent=1, default=str); print(json.dumps(log, indent=1, default=str))
