"""Check every hand-typed number in the paper (paper/*.tex: abstract, main text, appendix tables and captions) against the
numbers files and committed outputs. Written for the code release (16 Nov 2026) on the final paper text.
  python3 scripts/number_check.py [--out FILE]   -> NUMBER_CHECK.md (or FILE); exit code 1 if any check fails

Each claim is (where, quote, printed, source, value). A claim PASSES only if
  (1) `quote` occurs verbatim in the paper text (whitespace-normalised, \\input files expanded), so a claim about stale
      text cannot pass, and
  (2) the source value, rounded to the printed precision, equals the printed value (|value - printed| <= half a unit in
      the last printed digit), or, for range and yes/no claims, satisfies the stated condition.
Paths are relative to the repository root (the parent of scripts/), so reproduce.py can run it on regenerated outputs.
Fig. 2 is drawn by paper/make_fig_stages.py from values typed into that script; each typed value is checked against
trajectories.csv here.
"""
import ast, json, os, re, sys
import pandas as pd

S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else f"{S}/NUMBER_CHECK.md"
J = lambda p: json.load(open(f"{S}/{p}"))
N, NS, P, M, SC, H3S = (J(f) for f in ("numbers_E.json", "numbers_E_switched.json", "posthoc_E.json", "m1_E.json", "scale_E.json", "h3_matched_by_seed.json"))
V = pd.read_csv(f"{S}/detector_validation_weighted.csv")
T = pd.read_csv(f"{S}/trajectories.csv")
IT = pd.read_csv(f"{S}/labels/intermediate_label_table.csv").set_index("cell")
IJ = pd.read_csv(f"{S}/labels/intermediate_label_table_joined.csv")
Z = J("results/training/pilot/zstats.json"); RM, RT = J("results/training/pilot/rm_results.json"), J("results/training/pilot/rm_test_results.json")
PL, AL = J("data/pairs_log.json"), J("data/articles_log.json")
PD = J("results/final/probe_debug.json")
KLB0 = J("results/training/main/main_b_s0/kl.json")


def tex(f):
    s = open(f"{S}/paper/{f}").read()
    s = re.sub(r"(?<!\\)%.*", "", s)
    return re.sub(r"\\input\{(\w+)\}", lambda m: tex(m.group(1) + ".tex"), s)


TEX = re.sub(r"\s+", " ", tex("main.tex"))
TB, H = N["table"], N["hypotheses"]
C = []


def add(where, quote, printed, src, val):
    C.append(("num", where, quote, str(printed), src, val))


def cond(where, quote, printed, src, ok, val=None):
    C.append(("cond", where, quote, str(printed), src, (bool(ok), val)))


def pct(x): return 100 * x


def sgn(v, d):   # LaTeX-style signed number as typed in the tables; a value that rounds to zero keeps the sign of the typed text
    s = f"{v:+.{d}f}"
    return s


# Known errors in the paper text, found by this check. A claim listed here is reported as ERRATUM (not hidden, not counted
# as a pass); the printed total names them. Key: (where, source) as in the claim.
ERRATA = {
    ("Table 2 (final, pooled)", "arm_c_pooled.posthoc react"):
        "typed 13.6; the pooled rate is 13.54% (per seed 4.175% and 22.9%), which rounds to 13.5. 13.6 is the mean of the already-rounded 4.2 and 22.9.",
    ("Table 5 caption", "numbers_E.nli_tau == 0.55"):
        "caption says tau = 0.55, but the column's values (22.2, 24.6, 95.8, ...) are the tau = 0.5 rates in numbers_E.json; the tau = 0.55 rates "
        "(23.7, 26.8, 96.4, ...) are in numbers_E_switched.json. Either the caption or the column must change.",
}
TYPED = {("Table 2 (final, pooled)", "arm_c_pooled.posthoc react"): "13.6"}   # value as typed in the paper, so the row quote still matches


# ---------------- Abstract ----------------
A = "Abstract"
add(A, r"invented-number flags rose from 1.7\% to 98\%", "1.7", "numbers_E table.base.numbers_only", pct(TB["base"]["numbers_only"]))
add(A, r"invented-number flags rose from 1.7\% to 98\%", "98", "numbers_E table.arm_a_pooled.numbers_only", pct(TB["arm_a_pooled"]["numbers_only"]))
for arm in "bc":
    v = pct(TB[f"arm_{arm}_pooled"]["numbers_only"])
    cond(A, r"against about 2\% under the other rewards", "about 2", f"numbers_E table.arm_{arm}_pooled.numbers_only in [1.5, 2.5]", 1.5 <= v <= 2.5, v)
r0, r1 = N["kl_final"]["main_a_s0"] / N["kl_final"]["main_b_s0"], N["kl_final"]["main_a_s1"] / N["kl_final"]["main_b_s1"]
add(A, "moved the writer four to six times further", "4", "kl_final a_s1 / b_s1 (smaller ratio)", min(r0, r1))
add(A, "moved the writer four to six times further", "6", "kl_final a_s0 / b_s0 (larger ratio)", max(r0, r1))
cond(A, "Best-of-16 selection with the same predictor did not produce this story", "no", "exploratory.bon16.numbers_only_best < 0.05",
     N["exploratory"]["bon16"]["numbers_only_best"] < 0.05, N["exploratory"]["bon16"]["numbers_only_best"])
cond(A, "An LLM probe missed almost every fabrication", "recall < .1", "m1_E exploratory.probe.test_half.recall_ipw < 0.1",
     M["exploratory"]["probe"]["test_half"]["recall_ipw"] < 0.1, M["exploratory"]["probe"]["test_half"]["recall_ipw"])

# ---------------- Methods ----------------
W = "§3"
drop = sum(PL[k]["problem_window_tests_dropped"] for k in ("exploratory", "confirmatory", "holdout"))
tests = sum(PL[k]["tests"] for k in ("exploratory", "confirmatory", "holdout"))
add(W, r"not properly randomised (about 22\% of tests)", "22", "pairs_log: tests dropped by the date window / all tests", pct(drop / tests))
add(W, r"$|z|\ge 1$ (47,932 pairs)", "47932", "pairs_log.train_pairs_absz_ge_1", PL["train_pairs_absz_ge_1"])
add(W, r"cap any single site at 3\% of the sample", "3", "articles_log.max_domain_share (<= 3%)", min(3.0, pct(AL["max_domain_share"])))
n_ids = {k: sum(1 for l in open(f"{S}/data/ids_{k}.txt") if l.strip()) for k in ("train", "selection", "test")}
for k, v in (("train", 3000), ("selection", 300), ("test", 1000)):
    add(W, "We sample 3,000 training, 300 selection and 1,000 test articles", v, f"data/ids_{k}.txt lines", n_ids[k])

# ---------------- §4.1 + Table: click predictor ----------------
W = "§4.1"
q = r"The click predictor passed its gate (80.4\% on clear held-out pairs, 81.3\% on test pairs)"
add(W, q, "80.4", "rm_results.rm_sel_clear", pct(RM["rm_sel_clear"]))
add(W, q, "81.3", "rm_test_results.test.clear_acc", pct(RT["test"]["clear_acc"]))
W = "Table 3 (verdicts)"
add(W, r"met (80.4\%; test 81.3\%)", "80.4", "rm_results.rm_sel_clear", pct(RM["rm_sel_clear"]))
add(W, r"met (80.4\%; test 81.3\%)", "81.3", "rm_test_results.test.clear_acc", pct(RT["test"]["clear_acc"]))
cond(W, r"H1 fabrication rises under the click predictor & rise", "rise", "hypotheses.H1.decision.label", H["H1"]["decision"]["label"] == "rise", H["H1"]["decision"]["label"])
cond(W, r"H2 fabrication rises under the simulated audience & inconclusive", "inconclusive", "hypotheses.H2.decision.label", H["H2"]["decision"]["label"] == "inconclusive", H["H2"]["decision"]["label"])
cond(W, r"real $<$ simulated (not robust across measures)", "real < simulated", "hypotheses.H3.decision_matched", H["H3"]["decision_matched"] == "real < simulated", H["H3"]["decision_matched"])
cond(W, r"with fabrication $\le$ untrained $+1$ point & failed", "failed", "hypotheses.H4.decision", H["H4"]["decision"] == "failed", H["H4"]["decision"])
add(W, r"failed (precision 0.64)", "0.64", "m1_E M1.precision", M["M1"]["precision"])
cond(W, r"failed (precision 0.64)", "failed", "m1_E M1.met is false", not M["M1"]["met"], M["M1"]["met"])
cond(W, "all three entity-check differences stay significant", "all", "hypotheses.holm.rejected (H1, H2, H3)", all(H["holm"]["rejected"].values()), H["holm"]["rejected"])
W = "Table 4 (click predictor)"
for name, n, rm, feat, tf in (("Selection, clear", RM["n_sel_clear"], RM["rm_sel_clear"], RM["feat_sel_clear"], RM["tfidf_sel_clear"]),
                              ("Selection, clear, equal length", RM["n_sel_clear_lenmatched"], RM["rm_sel_clear_lenmatched"], RM["feat_sel_clear_lenmatched"], RM["tfidf_sel_clear_lenmatched"])):
    row = rf"{name} & {n:,} & \textbf{{{pct(rm):.1f}}} & {pct(feat):.1f} & {pct(tf):.1f} \\"
    for p, src, v in ((f"{n:,}", "n", n), (f"{pct(rm):.1f}", "click predictor", pct(rm)), (f"{pct(feat):.1f}", "surface features", pct(feat)), (f"{pct(tf):.1f}", "bag of words", pct(tf))):
        add(W, row, p, f"rm_results ({name}: {src})", v)
for name, n, acc in (("Test, clear", RT["test"]["clear_n"], RT["test"]["clear_acc"]), ("Test, clear, no story overlap", RT["test"]["filtered_n"], RT["test"]["filtered_acc"])):
    row = rf"{name} & {n:,} & \textbf{{{pct(acc):.1f}}} & -- & -- \\"
    add(W, row, f"{n:,}", f"rm_test_results.test ({name}: n)", n); add(W, row, f"{pct(acc):.1f}", f"rm_test_results.test ({name}: accuracy)", pct(acc))

# ---------------- §4.2 + Table: intermediate labels + Fig. 2 ----------------
W = "§4.2"
a100, a150 = IT.loc["arm a @100"], IT.loc["arm a @150"]
cond(W, "At step 100, most sampled headlines already claimed something the article does not say", "most", "intermediate_label_table arm a @100: any invented > 6/12",
     a100["any invented (label)"] > a100["n"] / 2, int(a100["any invented (label)"]))
add(W, "and a third invented a role, event or detail", "4", "intermediate_label_table arm a @100: other fact only + both (= 12/3)", a100["other fact only"] + a100["both"])
cond(W, "At step 150, every sampled headline invented something", "every", "intermediate_label_table arm a @150: neither == 0", a100 is not None and a150["neither"] == 0, int(a150["neither"]))
add(W, "Seed 0 produced only five distinct headlines on the test articles", "5", "numbers_E table.main_a_s0.n_distinct_headlines", TB["main_a_s0"]["n_distinct_headlines"])
add(W, "ran to the end with 200 variants of the same structure", "200", "numbers_E table.main_a_s1.n_distinct_headlines", TB["main_a_s1"]["n_distinct_headlines"])
o = IJ[IJ.other_invented_fact == 1]
q = "Of the 27 labelled headlines with an invented fact beyond a reaction alone, the entity check flagged 8, the invented-numbers check flagged 5 and the NLI check flagged 17"
add(W, q, "27", "intermediate_label_table_joined: other_invented_fact == 1", len(o))
add(W, q, "8", "  ... of which entity_check == 1", o.entity_check.sum())
add(W, q, "5", "  ... of which invented_numbers == 1", o.invented_numbers.sum())
add(W, q, "17", "  ... of which nli_tau055 == 1", o.nli_tau055.sum())
cond(W, "did not increase invented-number flags above the untrained level", "CI covers 0", "hypotheses.H2.numbers_only CI contains 0",
     H["H2"]["numbers_only"]["lo"] <= 0 <= H["H2"]["numbers_only"]["hi"], (H["H2"]["numbers_only"]["lo"], H["H2"]["numbers_only"]["hi"]))
W = "Table 1 (checkpoint labels)"
for cell, lab in (("arm a @100", r"(a) Click predictor, step 100"), ("arm a @150", r"(a) Click predictor, step 150"), ("arm b @250", r"(b) Simulated audience, step 250"),
                  ("arm b @350", r"(b) Simulated audience, step 350"), ("arm c @350", r"(c) Faithfulness penalty, step 350")):
    r = IT.loc[cell]; vals = (r["reaction only"], r["other fact only"] + r["both"], r["neither"], r["viral template style"])
    row = lab + " & " + " & ".join(str(int(v)) for v in vals) + r" \\"
    for v, c in zip(vals, ("reaction only", "other fact only + both", "neither", "viral template style")):
        add(W, row, int(v), f"intermediate_label_table [{cell}] {c}", v)
W = "Fig. 2 (paper/make_fig_stages.py)"
src = open(f"{S}/paper/make_fig_stages.py").read()
lit = {}
for node in ast.walk(ast.parse(src)):
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("num", "rea", "off", "num1", "rea1", "off1", "S"):
        try:
            lit[node.targets[0].id] = ast.literal_eval(node.value)
        except ValueError:   # off['c'] is a comprehension
            lit[node.targets[0].id] = eval(compile(ast.Expression(node.value), "fig", "eval"))
RUN = {"a": "main_a_s0", "b": "main_b_s0", "c": "main_c_s0", "L": "scale_E1", "Q": "scale_E2"}
COLS = {"num": "invented_numbers_pct", "rea": "invented_reactions_pct", "off": "off_article_pct"}
TT = T.set_index(["run", "step"])
for var, col in COLS.items():
    for k, pts in lit[var].items():
        for s, v in pts:
            C.append(("fig", W, f"{var}['{k}'] step {s}", repr(v), f"trajectories.csv {RUN[k]} step {s} {col}", TT.loc[(RUN[k], s), col]))
    for k, vals in lit[var + "1"].items():
        for s, v in zip(lit["S"], vals):
            C.append(("fig", W, f"{var}1['{k}'] step {s}", repr(v), f"trajectories.csv main_{k}_s1 step {s} {col}", TT.loc[(f"main_{k}_s1", s), col]))

# ---------------- §4.3 + final tables ----------------
W = "Table 2 (final, pooled)"
srcs = (("Real titles", "human", None), ("Untrained writer", "base", None), ("(a) Click predictor", "arm_a_pooled", "a"),
        ("(b) Simulated audience", "arm_b_pooled", "b"), ("(c) Click predictor + faithfulness penalty", "arm_c_pooled", "c"))
for lab, key, arm in srcs:
    t = TB[key]
    react = pct(P["per_source"][key]["react"] if arm is None else P["per_arm_pooled"][key.replace("_pooled", "")]["react"])
    off = pct(P["per_source"][key]["offarticle"] if arm is None else P["per_arm_pooled"][key.replace("_pooled", "")]["offarticle"])
    cells = [(f"{pct(t['numbers_only']):.1f}", "numbers_only", pct(t["numbers_only"])), (TYPED.get((W, f"{key}.posthoc react"), f"{react:.1f}"), "posthoc react", react), (f"{off:.1f}", "posthoc offarticle", off),
             (f"{pct(t['fabricated']):.1f}", "fabricated", pct(t["fabricated"])), (f"{t['words']:.1f}", "words", t["words"]),
             (f"${sgn(t['rm_z'], 2)}$", "rm_z", t["rm_z"]), (f"${sgn(t['judge_z'], 2)}$", "judge_z", t["judge_z"])]
    row = lab + " & " + " & ".join(c[0] for c in cells) + r" \\"
    row = row.replace("$-0.00$", "$+0.00$")
    for p, c, v in cells:
        add(W, row, p.strip("$").replace("+", ""), f"{key}.{c}", v)
W = "Table 5 (final, per seed)"
LAB = {"human": "Real titles", "base": "Untrained writer", **{f"main_{a}_s{s}": f"({a}) seed {s}" for a in "abc" for s in (0, 1)}}
for key, lab in LAB.items():
    t = TB[key]
    kl = "--" if key == "human" else ("0" if key == "base" else f"{N['kl_final'][key]:.1f}")
    cells = [(f"{pct(t['fabricated']):.1f}", "fabricated", pct(t["fabricated"])), (f"{pct(t['numbers_only']):.1f}", "numbers_only", pct(t["numbers_only"])),
             (f"{pct(P['per_source'][key]['react']):.1f}", "posthoc react", pct(P["per_source"][key]["react"])),
             (f"{pct(P['per_source'][key]['offarticle']):.1f}", "posthoc offarticle", pct(P["per_source"][key]["offarticle"])),
             (f"{pct(t['nli_unsup']):.1f}", "numbers_E nli_unsup (tau 0.5; see erratum)", pct(t["nli_unsup"])), (f"{pct(t['copy8']):.1f}", "copy8", pct(t["copy8"])),
             (f"{t['words']:.1f}", "words", t["words"]), (f"{pct(t['valid']):.1f}", "valid", pct(t["valid"])),
             (kl, "kl_final", None if key in ("human", "base") else N["kl_final"][key]),
             (f"${sgn(t['rm_z'], 2)}$", "rm_z", t["rm_z"]), (f"${sgn(t['judge_z'], 2)}$", "judge_z", t["judge_z"])]
    row = (lab + " & " + " & ".join(c[0] for c in cells) + r" \\").replace("$-0.00$", "$+0.00$")
    for p, c, v in cells:
        if v is not None:
            add(W, row, p.strip("$").replace("+", ""), f"{key}.{c}", v)
cond("Table 5 caption", r"``NLI: not supported'' is the evaluation NLI check at $\tau=0.55$", "0.55", "numbers_E.nli_tau == 0.55", N.get("nli_tau") == 0.55, N.get("nli_tau"))
add("Table 5 caption", r"Seed 0 of (a) is its checkpoint at the collapse stop (step 240)", "240", "trajectories.csv main_a_s0 last step",
    T[T.run == "main_a_s0"].step.max())
add("Table 6 caption", r"(entity check: click predictor 8.5\% and 7.6\%", "8.5", "h3_matched_by_seed seed0.fabricated.a_rate", pct(H3S["seed0"]["fabricated"]["a_rate"]))
add("Table 6 caption", r"(entity check: click predictor 8.5\% and 7.6\%", "7.6", "h3_matched_by_seed seed1.fabricated.a_rate", pct(H3S["seed1"]["fabricated"]["a_rate"]))
add("Table 6 caption", r"simulated audience 21.5\% and 46.3\%)", "21.5", "h3_matched_by_seed seed0.fabricated.b_rate", pct(H3S["seed0"]["fabricated"]["b_rate"]))
add("Table 6 caption", r"simulated audience 21.5\% and 46.3\%)", "46.3", "h3_matched_by_seed seed1.fabricated.b_rate", pct(H3S["seed1"]["fabricated"]["b_rate"]))
q = "(26.7 nats for seed 0, 40.1 for seed 1)"
add("Table 6 caption", q, "26.7", "h3_matched_by_seed seed0.kl_target", H3S["seed0"]["kl_target"])
add("Table 6 caption", q, "40.1", "h3_matched_by_seed seed1.kl_target", H3S["seed1"]["kl_target"])
for s in (0, 1):
    b = H3S[f"seed{s}"]["fabricated"]["bracket_a"]
    cond("Table 6 caption", "Click-predictor rates are interpolated between steps 50 and 100 in both seeds", "50-100", f"h3_matched_by_seed seed{s}.bracket_a", b[:2] == [50, 100], b)
q = "(33.0 at step 250, 26.7 at step 350), so its rates come from the first crossing, between steps 200 and 250"
add("Table 6 caption", q, "33.0", "runs/grpo/main_b_s0/kl.json step 250", KLB0["250"]["kl_seq_mean"])
add("Table 6 caption", q, "26.7", "runs/grpo/main_b_s0/kl.json step 350", KLB0["350"]["kl_seq_mean"])
cond("Table 6 caption", q, "200-250", "numbers_E hypotheses.H3.matched_kl.bracket_b", H["H3"]["matched_kl"]["bracket_b"][:2] == [200, 250], H["H3"]["matched_kl"]["bracket_b"])
cond("Table 6 caption", "In seed 1 they come from its final checkpoint", "350", "h3_matched_by_seed seed1.fabricated.bracket_b at step 350 with weight 1",
     H3S["seed1"]["fabricated"]["bracket_b"][1] == 350 and H3S["seed1"]["fabricated"]["bracket_b"][2] == 1.0, H3S["seed1"]["fabricated"]["bracket_b"])
bf = P["H3m_fabricated_bfinal"]
q = rf"its final checkpoint instead gives ${pct(bf['est']):+.1f}$ [${pct(bf['lo']):+.1f}$, ${pct(bf['hi']):+.1f}$] on the entity check"
for p, k in (("-26.3", "est"), ("-31.9", "lo"), ("-21.4", "hi")):
    add("Table 6 caption", q, p, f"posthoc_E H3m_fabricated_bfinal.{k}", pct(bf[k]))

# ---------------- Table 5: hypothesis tests ----------------
W = "Table 6 (tests)"
def ci(d): return rf"${pct(d['est']):+.1f}$ [${pct(d['lo']):+.1f}$, ${pct(d['hi']):+.1f}$]"
rows = [(r"H1: (a) $-$ untrained & Entity check & ", H["H1"]["fabricated"], "H1.fabricated"),
        (r" & Invented numbers & ", H["H1"]["numbers_only"], "H1.numbers_only"),
        (r"H2: (b) $-$ untrained & Entity check & ", H["H2"]["fabricated"], "H2.fabricated"),
        (r" & Invented numbers & ", H["H2"]["numbers_only"], "H2.numbers_only"),
        (r"H3: (a) $-$ (b), equal KL & Entity check & ", H["H3"]["matched_kl"]["fabricated"], "H3.matched_kl.fabricated"),
        (r" & Invented numbers & ", H["H3"]["matched_kl"]["numbers_only"], "H3.matched_kl.numbers_only"),
        (r" & Invented reactions (post hoc) & ", P["H3m_react_firstcross"], "posthoc_E H3m_react_firstcross"),
        (r"H3, seed 1 (post hoc) & Entity check & ", H3S["seed1"]["fabricated"], "h3_matched_by_seed seed1.fabricated"),
        (r" & Invented numbers & ", H3S["seed1"]["numbers_only"], "h3_matched_by_seed seed1.numbers_only"),
        (r" & Invented reactions & ", H3S["seed1"]["react"], "h3_matched_by_seed seed1.react")]
for pre, d, key in rows:
    row = pre + ci(d) + r" \\"
    for k in ("est", "lo", "hi"):
        add(W, row, f"{pct(d[k]):+.1f}".replace("+", ""), f"{key}.{k} (pp)", pct(d[k]))
h4 = H["H4"]
row = r"H4: (c) $-$ untrained & Entity check & " + rf"${pct(h4['point']['fab']):+.1f}$ [${pct(h4['ci']['fab'][0]):+.1f}$, ${pct(h4['ci']['fab'][1]):+.1f}$] \\"
for p, v, k in ((f"{pct(h4['point']['fab']):.1f}", pct(h4["point"]["fab"]), "point.fab"), (f"{pct(h4['ci']['fab'][0]):.1f}", pct(h4["ci"]["fab"][0]), "ci.fab[0]"), (f"{pct(h4['ci']['fab'][1]):.1f}", pct(h4["ci"]["fab"][1]), "ci.fab[1]")):
    add(W, row, p, f"H4.{k} (pp)", v)
ps = h4["per_seed"]
row = rf"Share of click-predictor gain kept & {h4['point']['keep']:.2f} [{h4['ci']['keep'][0]:.2f}, {h4['ci']['keep'][1]:.2f}]; seed 0: {ps[0]['keep']:.2f}, seed 1: {ps[1]['keep']:.2f} \\"
for p, v, k in ((f"{h4['point']['keep']:.2f}", h4["point"]["keep"], "point.keep"), (f"{h4['ci']['keep'][0]:.2f}", h4["ci"]["keep"][0], "ci.keep[0]"),
                (f"{h4['ci']['keep'][1]:.2f}", h4["ci"]["keep"][1], "ci.keep[1]"), (f"{ps[0]['keep']:.2f}", ps[0]["keep"], "per_seed[0].keep"), (f"{ps[1]['keep']:.2f}", ps[1]["keep"], "per_seed[1].keep")):
    add(W, row, p, f"H4.{k}", v)

# ---------------- §4.3-4.5 prose ----------------
W = "§4.4-4.5"
cond(W, "H4 failed on its fabrication condition: the entity check rose by more than one point", "> 1 pt", "H4 cond_fab_det false and point.fab > 0.01",
     (not h4["cond_fab_det"]) and h4["point"]["fab"] > 0.01, h4["point"]["fab"])
cond(W, "just at the pre-registered threshold for the pooled seeds and below it for seed 0", "0.50 / <0.50", "H4 point.keep rounds to 0.50; per_seed[0].keep < 0.5",
     round(h4["point"]["keep"], 2) == 0.50 and ps[0]["keep"] < 0.5, (h4["point"]["keep"], ps[0]["keep"]))
q = r"Seed 1 had more invented reactions than seed 0 (22.9\% against 4.2\%)"
add(W, q, "22.9", "posthoc_E per_source.main_c_s1.react", pct(P["per_source"]["main_c_s1"]["react"]))
add(W, q, "4.2", "posthoc_E per_source.main_c_s0.react", pct(P["per_source"]["main_c_s0"]["react"]))
cond(W, "yet passed NLI more often", "c_s1 < c_s0", "numbers_E_switched nli_unsup main_c_s1 < main_c_s0",
     NS["table"]["main_c_s1"]["nli_unsup"] < NS["table"]["main_c_s0"]["nli_unsup"], (NS["table"]["main_c_s1"]["nli_unsup"], NS["table"]["main_c_s0"]["nli_unsup"]))
cond("Discussion", "at the cost of half the click-predictor gain", "half", "H4 point.keep rounds to 0.50", round(h4["point"]["keep"], 2) == 0.50, h4["point"]["keep"])
W = "§4.6 (best-of-16)"
bo = N["exploratory"]["bon16"]; c50 = N["exploratory"]["curves"]["a"]
q = r"raised the click-predictor score from $-0.03$ to $+0.56$"
add(W, q, "-0.03", "exploratory.bon16.rm_z_all", bo["rm_z_all"])
add(W, q, "0.56", "exploratory.bon16.rm_z_best", bo["rm_z_best"])
frac = (bo["rm_z_best"] - bo["rm_z_all"]) / (c50["50"]["rm_z"] - c50["0"]["rm_z"])
cond(W, "two thirds of the gain after 50 RL steps", "~0.67", "bon16 gain / seed-0 arm (a) gain at step 50 (exploratory.curves) in [0.60, 0.73]", 0.60 <= frac <= 0.73, frac)
q = r"(2.0\% against 1.7\%)"
add(W, q, "2.0", "exploratory.bon16.numbers_only_best", pct(bo["numbers_only_best"]))
add(W, q, "1.7", "table.base.numbers_only (untrained writer, final evaluation)", pct(TB["base"]["numbers_only"]))
q = r"(12.7\% against 10.4\% for all 16 candidates; 300 test articles; exploratory)"
add(W, q, "12.7", "exploratory.bon16.fabricated_best", pct(bo["fabricated_best"]))
add(W, q, "10.4", "exploratory.bon16.fabricated_all16", pct(bo["fabricated_all16"]))
add(W, q, "300", "exploratory.bon16.n_articles", bo["n_articles"])

# ---------------- §4.7 measurement validity + appendix label tables ----------------
W = "§4.7"
mx = max(TB[k]["probe_flag"] for k in ("human", "base", *(f"main_{a}_s{s}" for a in "abc" for s in (0, 1))))
cond(W, r"flagged under 1\% of headlines from every source", "< 1%", "max table.<source>.probe_flag < 0.01", mx < 0.01, mx)
tmpl = [i for i in PD["items"] if i["src"] == "arm_a_template"]
add(W, "answered ``No'' for all ten invented templates we inspected", "10", "results/final/probe_debug.json: arm_a_template items with greedy answer No",
    sum(i["greedy10"].strip().startswith("No") for i in tmpl))
add(W, "precision 0.64 on the 50 held-out labels against a bar of 0.7", "0.64", "m1_E M1.precision", M["M1"]["precision"])
add(W, "precision 0.64 on the 50 held-out labels against a bar of 0.7", "50", "m1_E M1.n_items", M["M1"]["n_items"])
vv = lambda sub, det, grp, c: float(V[(V.subset == sub) & (V.detector == det) & (V.group == grp)][c].iloc[0])
MIX = "mixture (4 sources equally weighted)"
q = "Weighted to the mix of generated outputs it is 0.89 (0.86 on all 100)"
add(W, q, "0.89", "detector_validation_weighted items 51-100, entity check, mixture, w_precision", vv("items 51-100", "entity check (main deterministic)", MIX, "w_precision"))
add(W, q, "0.86", "detector_validation_weighted all 100, entity check, mixture, w_precision", vv("all 100", "entity check (main deterministic)", MIX, "w_precision"))
for c, k in (("nli_on_test", "NLI alone"), ):
    d = M[c]; cond(W, "NLI alone and the invented-numbers check cleared both bars", "cleared", f"m1_E {c}: precision >= 0.7 and recall >= 0.6", d["precision"] >= 0.7 and d["recall_ipw"] >= 0.6, (d["precision"], d["recall_ipw"]))
d = M["exploratory"]["numbers_only"]["test_half"]
cond(W, "NLI alone and the invented-numbers check cleared both bars", "cleared", "m1_E exploratory.numbers_only.test_half: precision >= 0.7 and recall >= 0.6", d["precision"] >= 0.7 and d["recall_ipw"] >= 0.6, (d["precision"], d["recall_ipw"]))
q = "NLI, however, fell to precision .07 and recall .09 on the faithfulness-penalty runs"
add(W, q, "0.07", "detector_validation_weighted all 100, eval NLI, main_c_s0, w_precision", vv("all 100", "eval NLI (tau 0.55)", "main_c_s0", "w_precision"))
add(W, q, "0.09", "detector_validation_weighted all 100, eval NLI, main_c_s0, w_recall", vv("all 100", "eval NLI (tau 0.55)", "main_c_s0", "w_recall"))
Hs = NS["hypotheses"]
cond(W, "the entity check combined with NLI, changed no conclusion for H1--H3", "same labels", "numbers_E_switched H1/H2/H3 labels equal numbers_E",
     (Hs["H1"]["decision"]["label"], Hs["H2"]["decision"]["label"], Hs["H3"]["decision_matched"]) == (H["H1"]["decision"]["label"], H["H2"]["decision"]["label"], H["H3"]["decision_matched"]),
     (Hs["H1"]["decision"]["label"], Hs["H2"]["decision"]["label"], Hs["H3"]["decision_matched"]))
cond(W, "For H4 it gives ``inconclusive''", "inconclusive", "numbers_E_switched hypotheses.H4.decision", Hs["H4"]["decision"] == "inconclusive", Hs["H4"]["decision"])
W = "Table 7 (M1)"
def pr_cells(d, ci=True):
    p = f"{d['precision']:.2f}" + (f" [{d['precision_ci'][0]:.2f}, {d['precision_ci'][1]:.2f}]" if ci else "")
    r = f"{d['recall_ipw']:.2f} [{d['recall_ci'][0]:.2f}, {d['recall_ci'][1]:.2f}]"
    return p, r
EX = M["exploratory"]
for lab, d, key, pci in ((r"Entity check", M["M1"], "M1", True), (r"Entity check or NLI ($\tau=0.55$)", M["det_or_nli_on_test"], "det_or_nli_on_test", True),
                         (r"NLI alone ($\tau=0.55$)", M["nli_on_test"], "nli_on_test", True), (r"Invented numbers (exploratory)", EX["numbers_only"]["test_half"], "exploratory.numbers_only.test_half", True),
                         (r"Invented reactions (exploratory)", EX["react"]["test_half"], "exploratory.react.test_half", False), (r"LLM probe", EX["probe"]["test_half"], "exploratory.probe.test_half", False)):
    p, r = pr_cells(d, pci); row = f"{lab} & {p} & {r} &"
    vals = [(f"{d['precision']:.2f}", d["precision"], "precision"), (f"{d['recall_ipw']:.2f}", d["recall_ipw"], "recall_ipw"),
            (f"{d['recall_ci'][0]:.2f}", d["recall_ci"][0], "recall_ci[0]"), (f"{d['recall_ci'][1]:.2f}", d["recall_ci"][1], "recall_ci[1]")]
    if pci:
        vals += [(f"{d['precision_ci'][0]:.2f}", d["precision_ci"][0], "precision_ci[0]"), (f"{d['precision_ci'][1]:.2f}", d["precision_ci"][1], "precision_ci[1]")]
    for pv, v, k in vals:
        add(W, row, pv, f"m1_E {key}.{k}", v)
W = "Table 8 (label counts)"
LC = EX["label_counts_by_source"]
for lab, s in (("Untrained writer", "base"), ("(a) Click predictor", "main_a_s0"), ("(b) Simulated audience", "main_b_s0"), ("(c) Click predictor + faithfulness penalty", "main_c_s0")):
    vals = [LC[s].get(k, 0) for k in ("fabricated_fact", "faithful", "misleading_exaggeration", "teaser_no_claim", "other")]
    row = lab + " & " + " & ".join(str(v) for v in vals) + r" \\"
    for v, k in zip(vals, ("fabricated_fact", "faithful", "misleading_exaggeration", "teaser_no_claim", "other")):
        add(W, row, v, f"m1_E exploratory.label_counts_by_source.{s}.{k}", v)
W = "Table 9 (by source)"
def f2(x): return "--" if pd.isna(x) else (f"{x:.2f}" if x >= 0.995 else f"{x:.2f}".lstrip("0"))
for lab, det in (("Entity check", "entity check (main deterministic)"), (r"NLI ($\tau=0.55$)", "eval NLI (tau 0.55)"), ("Invented numbers", "invented numbers"),
                 ("Invented reactions", "invented reactions"), ("LLM probe", "OLMo probe")):
    cells = []
    for g in ("base", "main_a_s0", "main_b_s0", "main_c_s0", MIX):
        p, r = vv("all 100", det, g, "w_precision"), vv("all 100", det, g, "w_recall"); cells.append((p, r, g))
    row = lab + " & " + " & ".join(f"{f2(p)} / {f2(r)}" for p, r, _ in cells) + r" \\"
    for p, r, g in cells:
        if not pd.isna(p):
            add(W, row, f"{p:.2f}", f"detector_validation_weighted all 100 / {det} / {g} / w_precision", p)
        add(W, row, f"{r:.2f}", f"detector_validation_weighted all 100 / {det} / {g} / w_recall", r)
W = "App. C (labels)"
K = pd.read_csv(f"{S}/labels/key.csv")
for s, fl in (("base", 13), ("main_a_s0", 13), ("main_b_s0", 12), ("main_c_s0", 12)):
    g = K[K.source == s]
    add(W, "Each source gave 13 flagged and 12 unflagged items (untrained writer, click predictor) or 12 and 13 (the other two)", fl, f"labels/key.csv {s}: flagged items", g.flagged.sum())
add(W, "The author reviewed every item and accepted all 100 drafts", "100", "labels/labels.csv agreed == 1", int(pd.read_csv(f"{S}/labels/labels.csv").agreed.sum()))
IL = pd.read_csv(f"{S}/labels/intermediate_labels.csv")
add("Limitations", "were drafted by an LLM and reviewed by the author, who changed one", "1", "labels/intermediate_labels.csv claude_draft_agreed == 0", int((IL.claude_draft_agreed == 0).sum()))

# ---------------- §4.8 scale ----------------
W = "§4.8 (scale)"
E1, E2 = SC["scale_E1"], SC["scale_E2"]
cond(W, "Llama-8B collapsed faster than the 3B model", "collapsed", "scale_E.scale_E1.collapsed and collapse step < main_a_s0 collapse step",
     E1["collapsed"] and max(map(int, E1["curve"])) < T[T.run == "main_a_s0"].step.max(), max(map(int, E1["curve"])))
add(W, "writing one headline for every article from step 100", "1", "trajectories.csv scale_E1 step 100 distinct_per_600", TT.loc[("scale_E1", 100), "distinct_per_600"])
cond(W, "Qwen-14B did not collapse", "not collapsed", "scale_E.scale_E2.collapsed is false", not E2["collapsed"], E2["collapsed"])
cond(W, "most headlines added invented reactions such as ``goes viral''", "> 50%", "scale_E2 react at its last two checkpoints > 0.5",
     all(E2["curve"][s]["react"] > 0.5 for s in ("200", "250")), [E2["curve"][s]["react"] for s in ("200", "250")])
cond(W, "and invented numbers stayed low", "<= 6%", "scale_E2 max numbers_only <= 0.06", max(v["numbers_only"] for v in E2["curve"].values()) <= 0.06,
     max(v["numbers_only"] for v in E2["curve"].values()))

# ---------------- appendix prompts / limitations ----------------
W = "App. D (implementation)"
q = r"Its standardisation constants are mean 0.346 and SD 0.339 (click predictor: $-3.24$ and 1.37)"
add(W, q, "0.346", "zstats.judge.mean", Z["judge"]["mean"]); add(W, q, "0.339", "zstats.judge.sd", Z["judge"]["sd"])
add(W, q, "-3.24", "zstats.rm.mean", Z["rm"]["mean"]); add(W, q, "1.37", "zstats.rm.sd", Z["rm"]["sd"])
W = "Limitations"
q = r"the simulated audience cannot exceed $+1.93\,z$ because it is a probability, while the click predictor's score has no ceiling (it reached $+5.44\,z$)"
add(W, q, "1.93", "(1 - zstats.judge.mean) / zstats.judge.sd", (1 - Z["judge"]["mean"]) / Z["judge"]["sd"])
add(W, q, "5.44", "table.arm_a_pooled.rm_z", TB["arm_a_pooled"]["rm_z"])


# ---------------- evaluate ----------------
def norm(s): return re.sub(r"\s+", " ", s)


def check(kind, where, quote, printed, src, val):
    if kind == "fig":
        d = len(printed.split(".")[1]) if "." in printed else 0
        return abs(float(printed) - float(val)) <= 0.5 * 10 ** -d + 1e-9, True
    found = norm(quote) in TEX
    if kind == "cond":
        return found and val[0], found
    p = float(printed.replace(",", "").replace("−", "-"))
    d = len(printed.split(".")[1]) if "." in printed else 0
    return found and abs(float(val) - p) <= 0.5 * 10 ** -d + 1e-9, found


rows, nfail, nerr, used = [], 0, 0, set()
for kind, where, quote, printed, src, val in C:
    ok, found = check(kind, where, quote, printed, src, val)
    err = (not ok) and found and (where, src) in ERRATA
    used |= {(where, src)} if err else set()
    nfail += not ok and not err; nerr += err
    shown = val[1] if kind == "cond" else val
    shown = f"{shown:.4f}" if isinstance(shown, float) else str(shown)
    status = "PASS" if ok else ("ERRATUM: " + ERRATA[(where, src)] if err else ("FAIL" if found else "FAIL (quote not in paper)"))
    rows.append(f"| {where} | {status} | {printed} | {shown} | {src} | `{quote[:90]}` |")
stale = set(ERRATA) - used   # an erratum that no longer fires (text fixed) must be removed from the list
nfail += len(stale)
rows += [f"| {w} | FAIL (stale erratum entry: this claim now passes or no longer exists; remove it from ERRATA) | | | {s} | |" for w, s in sorted(stale)]
with open(OUT, "w") as f:
    f.write("# Number check\n\nEvery hand-typed number in the paper against its source (written by `scripts/number_check.py`).\n\n")
    f.write(f"**{len(C)} claims; {len(C) - nfail - nerr} pass; {nfail} fail; {nerr} known errata in the paper text (listed below and in ERRATA).**\n\n| Where | Result | Printed | Source value | Source | Quote |\n|---|---|---|---|---|---|\n")
    f.write("\n".join(rows) + "\n")
for r in rows:
    if "FAIL" in r:
        print(r)
for (w, s), note in ERRATA.items():
    print(f"  erratum [{w}] {note}")
print(f"number_check: {len(C)} claims; {len(C) - nfail - nerr} pass; {nfail} fail; {nerr} known errata in the paper text")
sys.exit(1 if nfail else 0)
