"""Write data/manifest.json (counts + SHA-256 of every committed data file) and, from the cluster's articles.parquet,
data/articles_index.csv (per-article split, domain, date and SHA-256 of the cut article text and of the title; no text).
Written for the code release (16 Nov 2026).

    python3 scripts/make_manifest.py [--articles PATH]      # PATH: articles.parquet written by build_articles.py
    python3 scripts/make_manifest.py --verify-articles PATH  # check a rebuilt articles.parquet against data/articles_index.csv

cc_news article text is not redistributed here. scripts/build_articles.py rebuilds it deterministically from the public
Hugging Face dataset (vblagoje/cc_news, revision in data/ccnews_source.json); --verify-articles then confirms that every
rebuilt article and title is byte-identical to the one used in the paper.
"""
import hashlib, json, os, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
sha = lambda b: hashlib.sha256(b).hexdigest()


def index(path):
    a = pd.read_parquet(path)
    return pd.DataFrame({"id": a.id, "split": a.split, "domain": a.domain, "date": a.date, "n_tok_full": a.n_tok_full,
                         "article_trunc_sha256": [sha(t.encode()) for t in a.article_trunc], "title_sha256": [sha(t.encode()) for t in a.title]})


if "--verify-articles" in sys.argv:
    new = index(sys.argv[sys.argv.index("--verify-articles") + 1]).set_index("id")
    old = pd.read_csv(os.path.join(D, "articles_index.csv")).set_index("id")
    bad = {c: int((new.reindex(old.index)[c].astype(str) != old[c].astype(str)).sum()) for c in ("split", "article_trunc_sha256", "title_sha256")}
    print(f"verify-articles: {len(old)} articles; mismatches {bad}")
    sys.exit(1 if any(bad.values()) else 0)

if "--articles" in sys.argv:
    index(sys.argv[sys.argv.index("--articles") + 1]).to_csv(os.path.join(D, "articles_index.csv"), index=False)

files = sorted(f for f in os.listdir(D) if f != "manifest.json" and not f.startswith("."))
idx = pd.read_csv(os.path.join(D, "articles_index.csv"))
man = {
    "note": "SHA-256 of every file in data/. cc_news article text is not included; articles_index.csv holds per-article SHA-256 so a rebuild "
            "with scripts/build_articles.py can be checked (scripts/make_manifest.py --verify-articles).",
    "counts": {
        "articles": {k: int(v) for k, v in idx.split.value_counts().items()},
        "ids": {k: sum(1 for l in open(os.path.join(D, f"ids_{k}.txt")) if l.strip()) for k in ("train", "selection", "test")},
        "upworthy_pairs": {k: len(pd.read_parquet(os.path.join(D, f"pairs_{k}.parquet"))) for k in ("train", "selection", "test")},
        "upworthy_pairs_clear": {k: int(pd.read_parquet(os.path.join(D, f"pairs_{k}.parquet"))["clear"].sum()) for k in ("selection", "test")},
    },
    "articles_trunc_sha256_all": sha("\n".join(f"{i}\t{h}" for i, h in zip(idx.id, idx.article_trunc_sha256)).encode()),
    "files": {f"data/{f}": {"bytes": os.path.getsize(os.path.join(D, f)), "sha256": sha(open(os.path.join(D, f), "rb").read())} for f in files},
}
json.dump(man, open(os.path.join(D, "manifest.json"), "w"), indent=1)
print(json.dumps(man["counts"]), len(man["files"]), "files")
