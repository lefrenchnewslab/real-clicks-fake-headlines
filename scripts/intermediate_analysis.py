# Block E post-hoc (EXPLORATORY): intermediate-checkpoint reference labels vs automated flags. Usage: python3 intermediate_analysis.py KEY LABELS OUTCSV
import sys, pandas as pd
K = pd.read_csv(sys.argv[1]); L = pd.read_csv(sys.argv[2])
M = K.merge(L, on="item", validate="one_to_one", suffixes=("_flag", "")); assert len(M) == 60
M["cell"] = M.source.str.replace("main_", "arm ").str.replace("_s0", "") + " @" + M.step.astype(str)
r, o = M.invented_reaction == 1, M.other_invented_fact == 1
M["cat"] = (r & ~o).map({True: "reaction only"}).fillna("")
M.loc[~r & o, "cat"] = "other fact only"; M.loc[r & o, "cat"] = "both"; M.loc[~r & ~o, "cat"] = "neither"
rows = []
for c, g in M.groupby("cell", sort=False):
    n = len(g); d = {"cell": c, "n": n}
    for k in ("reaction only", "other fact only", "both", "neither"): d[k] = int((g.cat == k).sum())
    d["viral template style"] = int(g.viral_template_style.sum())
    d["any invented (label)"] = int((g.invented_reaction | g.other_invented_fact).sum())
    for f in ("invented_numbers", "entity_check", "invented_reaction_flag", "off_article"): d[f"flag: {f.replace('_flag','')}"] = int(g[f].sum())
    d["agree: reaction flag vs reaction label"] = int((g.invented_reaction_flag == g.invented_reaction).sum())
    d["agree: entity check vs any invented label"] = int((g.entity_check == (g.invented_reaction | g.other_invented_fact)).sum())
    d["agree: entity check vs other-fact label"] = int((g.entity_check == g.other_invented_fact).sum())
    d["agree: invented numbers vs other-fact label"] = int((g.invented_numbers == g.other_invented_fact).sum())
    rows.append(d)
T = pd.DataFrame(rows); order = ["arm a @100", "arm a @150", "arm b @250", "arm b @350", "arm c @350"]
T = T.set_index("cell").loc[order].reset_index(); T.to_csv(sys.argv[3], index=False)
M.to_csv(sys.argv[3].replace(".csv", "_joined.csv"), index=False)
pd.set_option("display.width", 250); print(T.T.to_string())
print("\nother-fact items (cell | headline | reason):")
for _, x in M[o].iterrows(): print(f"  {x.cell} | {x.headline_flag if 'headline_flag' in M else x.headline} | {x.reason}")
