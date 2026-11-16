#!/usr/bin/env python3
"""Rebuild every number, table and figure in the paper from the committed outputs (offline, CPU, under a minute).

    python3 reproduce.py                      # writes reproduced/ and prints the headline numbers
    python3 reproduce.py --articles FILE      # also restores the publishers' titles (final_human) from an articles.parquet
                                              # rebuilt by scripts/build_articles.py, so the one title-based key is checked too
    python3 reproduce.py --keep               # keep the temporary working directory

What it does:
 1. checks the SHA-256 of every file listed in data/manifest.json;
 2. lays the committed outputs out in a temporary directory with the cluster layout the scripts expect
    ($RLW/runs/stage3, runs/scale, runs/grpo/<run>/kl.json, runs/rm, handlabel/, data/zstats.json) and decompresses *.gz;
    the committed numbers files are NOT copied, so they are regenerated;
 3. runs the CPU analysis chain in the order the cluster jobs ran it (docs/JOBS.md, jobs C4-C10):
    analyze (C4) -> posthoc stats (C5) -> scale_summary (C6) -> m1 (C7) -> analyze with the switched metric (C8)
    -> reanalysis + intermediate_analysis (C9) -> seed1_analysis (C10) -> paper/make_fig_stages.py -> number_check;
 4. compares every regenerated numbers file and table with the committed one, and copies them to reproduced/.
The deterministic metrics (scripts/metrics_cpu.py, posthoc.py rows) need the cc_news article text, which is not
redistributed; this script therefore starts from the committed *.metrics.parquet and *.posthoc.parquet files.
Exit code 0 only if every step ran, every regenerated value equals the committed one and number_check passes.
Requirements: Python >= 3.10 with numpy, pandas, pyarrow, scikit-learn, matplotlib (requirements.txt). No GPU, no network.
"""
import gzip, hashlib, json, math, os, shutil, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "reproduced")
JSONS = ("numbers_E.json", "numbers_E_switched.json", "posthoc_E.json", "m1_E.json", "scale_E.json", "h3_matched_by_seed.json")
CSVS = (("detector_validation_weighted.csv", "detector_validation_weighted.csv"), ("trajectories.csv", "trajectories.csv"),
        ("handlabel/m1_joined.csv", "labels/m1_joined.csv"), ("handlabel/intermediate_key.csv", "labels/intermediate_key.csv"),
        ("handlabel/intermediate_label_table.csv", "labels/intermediate_label_table.csv"),
        ("handlabel/intermediate_label_table_joined.csv", "labels/intermediate_label_table_joined.csv"), ("figs/figE1_data.csv", "figs/figE1_data.csv"))
NEEDS_TITLES = {f: {"table.human.distinct1", "table.human.n_distinct_headlines"} for f in ("numbers_E.json", "numbers_E_switched.json")}


def arg(name):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


def sh(tmp, *args, cwd=None, env=None):
    t = time.time()
    r = subprocess.run([sys.executable, *args], cwd=cwd or os.path.join(tmp, "scripts"), capture_output=True, text=True,
                       env={**os.environ, "RLW": tmp, "MPLBACKEND": "Agg", "MPLCONFIGDIR": os.path.join(tmp, ".mpl"), "NPROC": "1", **(env or {})})
    tag = " ".join(f"{k}={v}" for k, v in (env or {}).items())
    print(f"  {(tag + ' ' if tag else '') + ' '.join(os.path.basename(a) if a.endswith('.py') else os.path.relpath(a, tmp) if a.startswith(tmp) else a for a in args):<78} {time.time() - t:5.1f} s", flush=True)
    if r.returncode != 0 and "number_check" not in args[0]:
        print(r.stdout[-3000:], r.stderr[-3000:], sep="\n"); sys.exit(f"step failed: {' '.join(args)}")
    return r


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if src.endswith(".gz"):
        with gzip.open(src, "rb") as fi, open(dst[:-3], "wb") as fo:
            shutil.copyfileobj(fi, fo)
    else:
        shutil.copy2(src, dst)


def copydir(src, dst, skip=()):
    for d, _, files in os.walk(src):
        for f in files:
            if f not in skip and f != ".DS_Store":
                copy(os.path.join(d, f), os.path.join(dst, os.path.relpath(d, src), f))


def stage(tmp):
    R = lambda *p: os.path.join(ROOT, *p)
    T = lambda *p: os.path.join(tmp, *p)
    copydir(R("scripts"), T("scripts"))
    copy(R("results", "training", "pilot", "zstats.json"), T("data", "zstats.json"))
    copydir(R("results", "final"), T("runs", "stage3"))
    for f in os.listdir(R("results", "checkpoints")):
        if os.path.isfile(R("results", "checkpoints", f)):
            copy(R("results", "checkpoints", f), T("runs", "stage3", f))
    copydir(R("results", "checkpoints", "scale"), T("runs", "scale"))
    for grp in ("main", "scale"):
        copydir(R("results", "training", grp), T("runs", "grpo"))
    for f in ("rm_results.json", "rm_test_results.json"):
        copy(R("results", "training", "pilot", f), T("runs", "rm", f))
    # labels: only the inputs (keys, labels); every derived table is regenerated. labels.csv is added after the first analyze.py
    # run, as on the cluster (job C4 ran before the labels existed; M1 is computed by m1.py, job C7).
    for f in ("key.csv", "intermediate_labels.csv"):
        copy(R("labels", f), T("handlabel", f))
    os.makedirs(T("figs"), exist_ok=True)
    os.makedirs(T("paper"), exist_ok=True)
    shutil.copy2(R("paper", "make_fig_stages.py"), T("paper"))
    for f in sorted(os.listdir(R("paper"))):
        if f.endswith(".tex"):
            shutil.copy2(R("paper", f), T("paper"))
    for f in ("data/pairs_log.json", "data/articles_log.json", "data/ids_train.txt", "data/ids_selection.txt", "data/ids_test.txt"):
        copy(R(f), T(f))
    titles = arg("--articles")
    if titles:   # restore the human (publisher) titles exactly as scripts/gen_stage3.py "human" mode wrote them
        import pandas as pd
        sys.path.insert(0, T("scripts")); from common import postprocess
        a = pd.read_parquet(titles).set_index("id").title
        h = pd.read_parquet(T("runs", "stage3", "final_human.parquet"))
        h.insert(2, "raw", h.id.map(a)); h.insert(3, "headline", h.raw.map(postprocess))
        assert h.headline.notna().all(), "articles file is missing test ids"
        h.to_parquet(T("runs", "stage3", "final_human.parquet")); shutil.copy2(titles, T("data", "articles.parquet"))


def check_manifest():
    man = json.load(open(os.path.join(ROOT, "data", "manifest.json")))
    bad = [f for f, m in man["files"].items() if hashlib.sha256(open(os.path.join(ROOT, f), "rb").read()).hexdigest() != m["sha256"]]
    print(f"  data/manifest.json: {len(man['files'])} files, {len(bad)} SHA-256 mismatches" + (f" {bad}" if bad else ""))
    return len(bad)


def flat(x, p=""):
    if isinstance(x, dict):
        o = {}
        for k, v in x.items():
            o.update(flat(v, f"{p}.{k}" if p else str(k)))
        return o
    if isinstance(x, list) and any(isinstance(v, (dict, list)) for v in x):
        o = {}
        for i, v in enumerate(x):
            o.update(flat(v, f"{p}[{i}]"))
        return o
    return {p: x}


def same(a, b):
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return abs(a - b) <= 1e-9 * max(1.0, abs(b))
    return a == b


def compare(tmp):
    import pandas as pd
    bad = 0
    for f in JSONS:
        new, old = flat(json.load(open(os.path.join(tmp, f)))), flat(json.load(open(os.path.join(ROOT, f))))
        skip = NEEDS_TITLES.get(f, set()) - set(new)
        diff = sorted((set(new) ^ set(old)) - skip) + sorted(k for k in set(new) & set(old) if not same(new[k], old[k]))
        bad += len(diff)
        note = f"; {len(skip)} title-based keys not checked: they need the publishers' titles (run with --articles): {sorted(skip)}" if skip else ""
        print(f"  {f}: {len(new)} keys regenerated, {len(diff)} differ from the committed file" + (f" (e.g. {diff[:3]})" if diff else "") + note)
    for new_p, old_p in CSVS:
        a, b = pd.read_csv(os.path.join(tmp, new_p)), pd.read_csv(os.path.join(ROOT, old_p))
        nd = 0
        if list(a.columns) != list(b.columns) or len(a) != len(b):
            nd = max(a.size, b.size)
        else:
            for c in a.columns:
                x, y = a[c], b[c]
                if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
                    nd += int((~(((x - y).abs() <= 1e-9 * y.abs().clip(lower=1)) | (x.isna() & y.isna()))).sum())
                else:
                    nd += int((x.astype(str) != y.astype(str)).sum())
        bad += nd > 0
        print(f"  {old_p}: {len(a)} rows x {a.shape[1]} columns regenerated, {nd} cells differ from the committed file")
    return bad


def headline(tmp):
    J = lambda f: json.load(open(os.path.join(tmp, f)))
    N, M, P, H3 = J("numbers_E.json"), J("m1_E.json"), J("posthoc_E.json"), J("h3_matched_by_seed.json")
    T, H = N["table"], N["hypotheses"]
    pp = lambda d: f"{100 * d['est']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]"
    rows = [
        ("Abstract, Tab. 2", "Invented numbers: untrained writer (%)", f"{100 * T['base']['numbers_only']:.1f}"),
        ("Abstract, Tab. 2", "Invented numbers: (a) click predictor, seeds pooled (%)", f"{100 * T['arm_a_pooled']['numbers_only']:.1f}"),
        ("Tab. 2", "Invented numbers: (b) simulated audience / (c) + faithfulness penalty (%)",
         f"{100 * T['arm_b_pooled']['numbers_only']:.1f} / {100 * T['arm_c_pooled']['numbers_only']:.1f}"),
        ("Tab. 6 (H1)", "Entity check, (a) - untrained (pp) [95% CI]", pp(H["H1"]["fabricated"])),
        ("Tab. 3 (H2)", "Pre-registered label", H["H2"]["decision"]["label"]),
        ("Tab. 6 (H3)", "Entity check, (a) - (b) at equal KL, seed 0 (pp)", pp(H["H3"]["matched_kl"]["fabricated"])),
        ("Tab. 6 (H4)", "Share of click-predictor gain kept", f"{H['H4']['point']['keep']:.2f} [{H['H4']['ci']['keep'][0]:.2f}, {H['H4']['ci']['keep'][1]:.2f}]; {H['H4']['decision']}"),
        ("Tab. 7 (M1)", "Entity-check precision, items 51-100", f"{M['M1']['precision']:.2f} (met: {M['M1']['met']})"),
        ("Tab. 5", "KL to the untrained writer, seed 0: (a) / (b) (nats)", f"{N['kl_final']['main_a_s0']:.1f} / {N['kl_final']['main_b_s0']:.1f}"),
        ("Sec. 4.6", "Best-of-16: invented numbers of the chosen headline (%)", f"{100 * N['exploratory']['bon16']['numbers_only_best']:.1f}"),
    ]
    print("\nHeadline numbers (should match README 'Quickstart'):")
    for where, what, val in rows:
        print(f"  {where:<18} {what:<72} {val}")


def main():
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="rcfh_reproduce_")
    print("[1/5] checking data/manifest.json")
    bad = check_manifest()
    print(f"[2/5] staging committed outputs in {tmp}")
    stage(tmp)
    print("[3/5] running the analysis chain (cluster jobs C4-C10)")
    sh(tmp, "analyze.py")                                                         # C4 -> numbers_E.json
    sh(tmp, "posthoc.py", "stats")                                                # C5 -> posthoc_E.json
    sh(tmp, "scale_summary.py")                                                   # C6 -> scale_E.json
    copy(os.path.join(ROOT, "labels", "labels.csv"), os.path.join(tmp, "handlabel", "labels.csv"))
    sh(tmp, "m1.py")                                                              # C7 -> m1_E.json, handlabel/m1_joined.csv
    sh(tmp, "analyze.py", env={"SWITCH_METRIC": "det_or_nli", "NLI_TAU": "0.55"})  # C8 -> numbers_E_switched.json
    sh(tmp, "reanalysis.py")                                                      # C9 -> detector validation, trajectories_seed0, intermediate key
    sh(tmp, "intermediate_analysis.py", os.path.join(tmp, "handlabel", "intermediate_key.csv"),
       os.path.join(tmp, "handlabel", "intermediate_labels.csv"), os.path.join(tmp, "handlabel", "intermediate_label_table.csv"))
    sh(tmp, "seed1_analysis.py")                                                  # C10 -> trajectories.csv, h3_matched_by_seed.json
    sh(tmp, os.path.join(tmp, "paper", "make_fig_stages.py"), cwd=os.path.join(tmp, "paper"))   # paper Fig. 2
    shutil.copytree(os.path.join(ROOT, "scripts"), os.path.join(tmp, "nc", "scripts"))
    for f in JSONS + ("detector_validation_weighted.csv", "trajectories.csv"):
        shutil.copy2(os.path.join(tmp, f), os.path.join(tmp, "nc", f))
    shutil.copytree(os.path.join(tmp, "paper"), os.path.join(tmp, "nc", "paper"))
    for d, src in (("labels", os.path.join(tmp, "handlabel")), ("results", os.path.join(ROOT, "results")), ("data", os.path.join(ROOT, "data"))):
        shutil.copytree(src, os.path.join(tmp, "nc", d))
    nc = sh(tmp, os.path.join(tmp, "nc", "scripts", "number_check.py"), "--out", os.path.join(tmp, "NUMBER_CHECK.md"), cwd=os.path.join(tmp, "nc"))
    print("[4/5] comparing with the committed outputs")
    bad += compare(tmp)
    print(f"[5/5] writing {os.path.relpath(OUT, ROOT)}/")
    shutil.rmtree(OUT, ignore_errors=True)
    for sub in ("figs", "labels"):
        os.makedirs(os.path.join(OUT, sub))
    for f in JSONS + ("detector_validation_weighted.csv", "trajectories.csv", "NUMBER_CHECK.md"):
        shutil.copy2(os.path.join(tmp, f), OUT)
    for new_p, old_p in CSVS[2:]:
        shutil.copy2(os.path.join(tmp, new_p), os.path.join(OUT, old_p))
    for f in os.listdir(os.path.join(tmp, "figs")):
        shutil.copy2(os.path.join(tmp, "figs", f), os.path.join(OUT, "figs"))
    for f in ("fig_stages.pdf", "fig_stages.png"):
        shutil.copy2(os.path.join(tmp, "paper", f), os.path.join(OUT, "figs"))
    headline(tmp)
    if "--keep" not in sys.argv:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n" + nc.stdout.strip().splitlines()[-1])
    ok = bad == 0 and nc.returncode == 0
    print(f"\nreproduce.py: {'OK' if ok else 'FAILED'} in {time.time() - t0:.0f} s "
          f"({'all regenerated values equal the committed ones' if bad == 0 else f'{bad} differences'}; "
          f"number_check {'passed' if nc.returncode == 0 else 'FAILED'})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
