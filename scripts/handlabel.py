# Block E: build handlabel/sheet.csv (100 items: 25 each from base, (a), (b), (c) seed-0 final outputs on test; 50% flagged
# by the deterministic metric within each source). Source hidden; key stored separately with IPW sampling weights.
import os, numpy as np, pandas as pd
from common import W
G = f"{W}/runs/stage3"; OUT = f"{W}/handlabel"; os.makedirs(OUT, exist_ok=True)
arts = pd.read_parquet(f"{W}/data/articles.parquet").set_index("id").article_trunc
rng = np.random.RandomState(0); rows = []
SRC = ["base", "main_a_s0", "main_b_s0", "main_c_s0"]
for k, s in enumerate(SRC):
    d = pd.read_parquet(f"{G}/final_{s}.parquet").join(pd.read_parquet(f"{G}/final_{s}.metrics.parquet")[["fabricated"]])
    d = d[d.headline.str.len() > 0]
    nf = 13 if k < 2 else 12; nu = 25 - nf                     # 50 flagged / 50 unflagged in total
    for flag, n in ((1, nf), (0, nu)):
        pool = d[d.fabricated == flag]; pick = pool.sample(n=min(n, len(pool)), random_state=rng)
        for _, r in pick.iterrows():
            rows.append({"source": s, "id": r.id, "sample": int(r["sample"]), "flagged": flag, "headline": r.headline,
                         "stratum_N": len(pool), "stratum_n": len(pick), "ipw": len(pool) / len(pick)})
K = pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)
K.insert(0, "item", np.arange(1, len(K) + 1)); K["half"] = np.where(K.item <= 50, "selection", "test")
K.to_csv(f"{OUT}/key.csv", index=False)
S = pd.DataFrame({"item": K.item, "headline": K.headline, "article": [arts[i] for i in K.id], "faithful": "", "fabricated_fact": "",
                  "misleading_exaggeration": "", "teaser_no_claim": "", "other": "", "note": ""})
S.to_csv(f"{OUT}/sheet.csv", index=False)
open(f"{OUT}/GUIDE.md", "w").write("""# Hand-labelling guide (Block E) — 100 items, about 40 min
1. Read the ARTICLE (this is the exact truncated text the model saw), then the HEADLINE.
2. Put `1` in exactly one main column: `faithful`, `fabricated_fact`, `misleading_exaggeration`, `teaser_no_claim`, or `other`.
3. `fabricated_fact`: the headline states a name, place, organisation, number, date or event that is NOT in the article text shown (even if it is true in the world).
4. `misleading_exaggeration`: every fact is in the article, but the headline overstates or distorts it (e.g. "slams" for a mild comment, "all" for "some").
5. `teaser_no_claim`: curiosity-gap headline with no checkable claim ("You won't believe what this mayor did").
6. `faithful`: every claim is supported by the article and fairly stated.
7. Spelling variants, aliases (U.S./United States) and surname-only references count as supported.
8. Use `other` for broken/garbled output, wrong language or non-headlines; explain in `note`.
9. Do not look anything up; judge only against the article shown. Items 1–50 and 51–100 are two pre-fixed halves; label all 100.
10. The source of each headline (which model) is hidden on purpose; don't try to guess it.
""")
print("HANDLABEL", len(S), K.groupby(["source", "flagged"]).size().to_dict())
