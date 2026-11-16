# Real Clicks, Fake Headlines: AI That Chases Clicks Invents the News

Code, generated headlines, reference labels and every number for the paper by Amr Sobhy (Le French News Lab), written at the Apart Research AI x Epistemics Research Sprint, November 2026. Paper: [`paper.pdf`](paper.pdf). Models: [click predictor](https://huggingface.co/amrsobhy/real-clicks-fake-headlines-click-predictor) and [six final adapters](https://huggingface.co/amrsobhy/real-clicks-fake-headlines-adapters) on the Hugging Face Hub.

## Summary
We use reinforcement learning (GRPO, LoRA) to train a 3B headline writer (Qwen2.5-3B-Instruct) for news articles from cc_news under three rewards: (a) a click predictor fitted to randomised headline tests from the Upworthy Research Archive, (b) a simulated audience of four LLM personas (Qwen2.5-7B-Instruct), and (c) the click predictor minus a faithfulness penalty from an NLI model. The study was pre-registered (hypotheses H1-H4, a measurement criterion M1, two seeds per reward), every headline is checked against its article for invented numbers, unsupported entities and invented reactions, and five automated checks are compared with 100 reference labels. **Headline:** under the click predictor the writer first invented reactions and events, then repeated one fabricated viral story for every article: invented-number flags rose from 1.7% to 97.6%, against 1.7% under the simulated audience and 2.5% with the faithfulness penalty. Best-of-16 selection with the same predictor did not produce the story.

| Result (1,000 test articles × 4 headlines unless stated) | Value [95% CI] | Paper | Source key |
|---|---|---|---|
| Invented numbers: untrained writer → (a) click predictor; (b) simulated audience, (c) + faithfulness penalty | 1.7% → 97.6%; 1.7%, 2.5% | Abstract, Table 2 | `numbers_E.json` `table.base.numbers_only`, `table.arm_{a,b,c}_pooled.numbers_only` |
| H1: entity check, (a) − untrained, seeds pooled (pre-registered "rise") | +85.5 pp [+84.0, +87.0] | Tables 3, 6 | `hypotheses.H1.fabricated`, `hypotheses.H1.decision.label` |
| H2: simulated audience vs untrained | inconclusive (entity check rose, invented numbers did not) | Tables 3, 6 | `hypotheses.H2.decision.label` |
| H3: entity check, (a) − (b) at equal KL, seed 0 (pre-registered "real < simulated", not robust across measures) | −13.0 pp [−16.8, −9.0] | Tables 3, 6 | `hypotheses.H3.matched_kl.fabricated` |
| H4: share of the click-predictor gain kept by the faithfulness penalty | 0.50 [0.49, 0.51]; H4 failed (entity check +3.6 pp) | Tables 3, 6 | `hypotheses.H4.point.keep`, `hypotheses.H4.ci.keep`, `hypotheses.H4.decision` |
| M1: entity-check precision on the 50 held-out labels (bar 0.70) | 0.64 [0.44, 0.82]; failed | Tables 3, 7 | `m1_E.json` `M1.precision`, `M1.precision_ci` |
| KL to the untrained writer, seed 0: (a) vs (b) | 152.5 vs 26.7 nats | Table 5, §4.4 | `numbers_E.json` `kl_final.main_a_s0`, `kl_final.main_b_s0` |
| Best-of-16 with the click predictor, untrained writer (300 test articles): invented numbers | 2.0% | §4.6 | `exploratory.bon16.numbers_only_best` |

## Repository map
Start with `reproduce.py`.

| Path | Purpose |
|---|---|
| `reproduce.py` | **Start here.** Reruns the CPU analysis chain on the committed outputs (offline, < 1 min), checks every regenerated number and table against the committed files and runs the number check |
| `paper.pdf` | The paper |
| `paper/*.tex`, `paper/refs.bib`, `paper/fig_stages.pdf` | Paper source; `number_check.py` reads the `.tex` files to check every hand-typed number |
| `paper/make_fig_stages.py` | Draws Fig. 2 from values typed in from `trajectories.csv` (`number_check.py` checks each typed value against that file) |
| `numbers_E.json` | Every main result: descriptive table per source (`table`), H1-H4 with bootstrap CIs and Holm (`hypotheses`), KL (`kl_final`), click-predictor validity (`rm_selection`, `rm_test`), curves and best-of-16 (`exploratory`) |
| `numbers_E_switched.json` | The same with the pre-registered fallback measure (entity check OR evaluation NLI at τ = 0.55) after M1 failed |
| `posthoc_E.json` | Post-hoc rates: invented reactions, off-article headlines, original title-case rule, H3 alternatives (labelled post hoc in `docs/RESULTS_E.md`) |
| `m1_E.json` | M1 and the validity of every automated check against the 100 reference labels |
| `detector_validation_weighted.csv` | Precision/recall of each check weighted by inverse sampling probability, per source (Table 9) |
| `h3_matched_by_seed.json` | H3 at equal KL for seed 0 and (post hoc) seed 1 |
| `scale_E.json` | Exploratory larger writers (Llama-3.1-8B, Qwen2.5-14B) |
| `trajectories.csv` | Per-checkpoint rates and rewards for every run and seed (Fig. 2) |
| `results/final/` | Final test outputs: `final_<source>.parquet` (headlines), `.scores.parquet` (click predictor, simulated audience, NLI, LLM probe), `.metrics.parquet` (entity and number checks, style), `.posthoc.parquet` (reactions, off-article). Sources: `base` (untrained writer), `main_{a,b,c}_s{0,1}`, `human` (scores and flags only: the publishers' titles are not redistributed). `probe_debug.json`: the post-hoc LLM-probe check |
| `results/checkpoints/` | The same four files for every saved checkpoint, 300 test articles × 2 (`curve_main_<run>_<step>`, seeds 0 and 1; `curve_base_0`), best-of-16 (`bon16_base`: all 16 candidates per article; the chosen one is the highest `rm`), and the larger writers (`scale/scale_E{1,2}_<step>`) |
| `results/training/main/<run>/`, `results/training/scale/<run>/` | Training logs: `config_*.json`, `reward_log.jsonl`, `timing.jsonl`, offline KL per checkpoint (`kl.json`), `DONE`/`COLLAPSED` markers. No weights |
| `results/training/pilot/` | Pilot runs (`pilot_*/g2.json`), `pilot_summary.json`, click-predictor results on selection (`rm_results.json`) and test (`rm_test_results.json`), frozen z constants (`zstats.json`) and the untrained writer's selection headlines they came from |
| `labels/` | Reference labels: `key.csv` (source, sampling weights), `labels.csv` (labels, article text removed), `claude_prelabels.csv`, `GUIDE.md`; checkpoint labels `intermediate_key.csv`, `intermediate_labels.csv`, `claude_intermediate_prelabels.csv`; derived `m1_joined.csv`, `intermediate_label_table*.csv` |
| `data/` | Article ids per split, `articles_index.csv` (split, domain, date, SHA-256 of each cut article and title; no text), Upworthy pairs `pairs_{train,selection,test}.parquet`, logs of how they were built, model revisions (`models.json`), personas, `manifest.json` |
| `scripts/` | All code that produced a reported number, figure or dataset (below), the Slurm wrappers, and `cmds/<job>.txt`, the exact command list of each job in `docs/JOBS.md` |
| `docs/` | `DESIGN_E.md` (design), `RESULTS_E.md` (pre-registration, addenda A1-A3, results), `FAILURE_MODES.md` (written before the code), `JOBS.md` + `usage.csv` (every job and its A100-hours) |
| `site/` | Source of the interactive companion (Svelte 5): `npm ci && npm run build` writes one self-contained `dist/index.html`; its data file `src/data.json` is built by `site/build/make_data.py` from the project's results (paths there follow the cluster/project layout, not `results/`). AGPL-3.0-or-later because it bundles the Ledger design system |
| `figs/` | `fig_stages` (paper Fig. 2) and the analysis figures `figE*` regenerated by `reproduce.py` (not in the paper) |
| `requirements.txt`, `requirements-gpu.txt` | Pinned packages for the analysis and for a full rerun |
| `LICENSE`, `LICENSE-DATA` | MIT (code); CC BY 4.0 (data, outputs, documents), with the Upworthy credit |
| Hugging Face: `amrsobhy/real-clicks-fake-headlines-click-predictor` | The click predictor: DeBERTa-v3-base, Bradley-Terry scalar head, checkpoint at step 749 |
| Hugging Face: `amrsobhy/real-clicks-fake-headlines-adapters` | The six final LoRA adapters (r 32, Qwen2.5-3B-Instruct) in subfolders `a_s0` … `c_s1`; `a_s0` is the step-240 collapse-stop checkpoint |

## Requirements
- **Quickstart (analysis only):** Python ≥ 3.10 (tested with 3.14.6), `pip install -r requirements.txt` (numpy, pandas, pyarrow, scikit-learn, matplotlib, pinned). No GPU, no network, no API keys.
- **Full rerun:** `pip install -r requirements-gpu.txt` (torch 2.8.0, transformers 5.23.0, trl 1.16.0, peft 0.21.3, accelerate 1.15.0, datasets 5.2.0, spacy 3.8.18 with `spacy/en_core_web_sm` from the Hub; we used an isolated uv venv with Python 3.12.15, `scripts/setup_venv.sh`), one NVIDIA A100 80 GB, Slurm, and the Hugging Face models in `data/models.json` (Qwen2.5-3B/7B-Instruct, DeBERTa-v3-base, two DeBERTa NLI models, OLMo-2-1124-7B-Instruct; Llama-3.1-8B-Instruct and Qwen2.5-14B-Instruct for the scale check). No API models are used.

## Compute
| Stage | Hardware | A100-hours | Notes (`docs/JOBS.md`, `docs/usage.csv`) |
|---|---|---|---|
| Stage 0-1: environment tests, click predictor, pilot gates G1-G2 | 1× A100 80 GB | 1.281 | jobs E_J0-E_J3r, including two pre-planned learning-rate retries |
| Stage 2: six main runs (3 rewards × 2 seeds, 350 steps) + offline KL | 1× A100 80 GB each | 6.930 | run in parallel |
| Stage 3: final and checkpoint generations, best-of-16, scoring | 1× A100 80 GB | 1.349 | |
| Post-hoc LLM-probe check | 1× A100 80 GB | 0.024 | |
| Exploratory scale check (Llama-3.1-8B, Qwen2.5-14B) | 1× A100 80 GB | 3.567 | |
| Seed-1 trajectories (post hoc) | 1× A100 80 GB | 0.301 | |
| **Total** | | **13.451** | per-stage values rounded from the ledger |
| CPU jobs (metrics, analysis) | 40-core CPU node | 0 | 10 jobs, 1.7 h wall, ≈ 69 core-hours allocated |
| Analysis (`reproduce.py`) | laptop CPU | — | < 1 min |

## Quickstart: reproduce the paper (CPU, < 1 min, offline)
```bash
git clone https://github.com/lefrenchnewslab/real-clicks-fake-headlines && cd real-clicks-fake-headlines
pip install -r requirements.txt
python3 reproduce.py
```
It checks the SHA-256 of every data file, lays the committed outputs out in a temporary directory with the cluster layout, reruns the CPU analysis chain in the order the cluster jobs ran it (C4-C10 in `docs/JOBS.md`), compares every regenerated numbers file and table with the committed one, runs `scripts/number_check.py` on the paper source, and writes `reproduced/` (numbers, tables, figures, `NUMBER_CHECK.md`). It took about 1 min on a laptop. You should see:

```
  data/manifest.json: 13 files, 0 SHA-256 mismatches
  numbers_E.json: 694 keys regenerated, 0 differ from the committed file; 2 title-based keys not checked: they need the publishers' titles (run with --articles): ['table.human.distinct1', 'table.human.n_distinct_headlines']
  numbers_E_switched.json: 694 keys regenerated, 0 differ from the committed file; 2 title-based keys not checked: they need the publishers' titles (run with --articles): ['table.human.distinct1', 'table.human.n_distinct_headlines']
  posthoc_E.json: 215 keys regenerated, 0 differ from the committed file
  m1_E.json: 196 keys regenerated, 0 differ from the committed file
  scale_E.json: 108 keys regenerated, 0 differ from the committed file
  h3_matched_by_seed.json: 54 keys regenerated, 0 differ from the committed file
  detector_validation_weighted.csv: 60 rows x 11 columns regenerated, 0 cells differ from the committed file
  trajectories.csv: 56 rows x 15 columns regenerated, 0 cells differ from the committed file
  labels/m1_joined.csv: 100 rows x 29 columns regenerated, 0 cells differ from the committed file
  labels/intermediate_key.csv: 60 rows x 13 columns regenerated, 0 cells differ from the committed file
  labels/intermediate_label_table.csv: 5 rows x 16 columns regenerated, 0 cells differ from the committed file
  labels/intermediate_label_table_joined.csv: 60 rows x 20 columns regenerated, 0 cells differ from the committed file
  figs/figE1_data.csv: 56 rows x 11 columns regenerated, 0 cells differ from the committed file

Headline numbers (should match README 'Quickstart'):
  Abstract, Tab. 2   Invented numbers: untrained writer (%)                                   1.7
  Abstract, Tab. 2   Invented numbers: (a) click predictor, seeds pooled (%)                  97.6
  Tab. 2             Invented numbers: (b) simulated audience / (c) + faithfulness penalty (%) 1.7 / 2.5
  Tab. 6 (H1)        Entity check, (a) - untrained (pp) [95% CI]                              +85.5 [+84.0, +87.0]
  Tab. 3 (H2)        Pre-registered label                                                     inconclusive
  Tab. 6 (H3)        Entity check, (a) - (b) at equal KL, seed 0 (pp)                         -13.0 [-16.8, -9.0]
  Tab. 6 (H4)        Share of click-predictor gain kept                                       0.50 [0.49, 0.51]; failed
  Tab. 7 (M1)        Entity-check precision, items 51-100                                     0.64 (met: False)
  Tab. 5             KL to the untrained writer, seed 0: (a) / (b) (nats)                     152.5 / 26.7
  Sec. 4.6           Best-of-16: invented numbers of the chosen headline (%)                  2.0

number_check: 552 claims; 552 pass; 0 fail; 0 known errata in the paper text

reproduce.py: OK in 63 s (all regenerated values equal the committed ones; number_check passed)
```

Every hand-typed number in the paper matches its source. `scripts/number_check.py` keeps an `ERRATA` list for known errors in the paper text; it is empty.

| Paper item | Script (run by `reproduce.py` unless marked) | Output |
|---|---|---|
| Table 1 (checkpoint labels), §4.2 counts | `reanalysis.py` (draws the 60 items), `intermediate_analysis.py` | `labels/intermediate_key.csv`, `labels/intermediate_label_table*.csv` |
| Fig. 2 | `reanalysis.py` + `seed1_analysis.py` (rates), `paper/make_fig_stages.py` (drawing) | `trajectories.csv`, `paper/fig_stages.pdf` |
| Table 2, Table 5, §4.3 | `analyze.py` (rates, rewards, KL), `posthoc.py stats` (invented reactions, off-article), `analyze.py` (NLI column at the default τ = 0.5; the τ = 0.55 rates of Appendix C are in `numbers_E_switched.json`) | `numbers_E.json`, `posthoc_E.json`, `numbers_E_switched.json` |
| Table 3 (verdicts), Table 6 (tests), §4.4-4.5 | `analyze.py` + `h4.py` (H1-H4, Holm), `posthoc.py stats` (reactions, final-checkpoint alternative), `seed1_analysis.py` (seed 1) | `numbers_E.json`, `posthoc_E.json`, `h3_matched_by_seed.json` |
| Table 4 (click predictor) | `rm_train.py` (GPU, job E_J1), `rm_test.py` (job E_C1); committed outputs | `results/training/pilot/rm_results.json`, `rm_test_results.json` |
| §4.6 best-of-16 | `analyze.py` | `numbers_E.json` `exploratory.bon16` |
| Tables 7-9, §4.7 | `m1.py`, `reanalysis.py` | `m1_E.json`, `labels/m1_joined.csv`, `detector_validation_weighted.csv` |
| §4.8 scale | `scale_summary.py` | `scale_E.json` |
| Appendix D constants | `zstats.py` (GPU, job E_J1); committed output | `results/training/pilot/zstats.json` |
| Every hand-typed number in the paper | `number_check.py` | `reproduced/NUMBER_CHECK.md` |

Two inputs of the chain need the cc_news article text and therefore start from committed files: the deterministic checks (`metrics_cpu.py` → `*.metrics.parquet`; `posthoc.py rows` → `*.posthoc.parquet`). To rerun them, rebuild the articles (below), then `RLW=<dir> python3 scripts/metrics_cpu.py "<dir>/runs/stage3/*.parquet"` and `posthoc.py rows`. With a rebuilt `articles.parquet`, `python3 reproduce.py --articles <file>` also restores the publishers' titles in `final_human` and checks the two title-based keys.

## Full rerun from scratch (GPU, ≈ 13.5 A100-hours)
1. **Environment and downloads** (login node with internet, 0 GPU-hours): `bash scripts/setup_venv.sh`, `bash scripts/pack_env.sh`, `bash scripts/stage0_downloads.sh` (Upworthy files listed from OSF by `download_upworthy.py`, cc_news and the models at the revisions in `data/ccnews_source.json` and `data/models.json`). Working directory: `$WORK_ROOT/rlhead` with `scripts/`, `data/`, `logs/`, `runs/`; set `RLW=$WORK_ROOT/rlhead`.
2. **Data** (CPU, deterministic): `python3 scripts/build_pairs.py` (Upworthy pairs; compare with `data/pairs_*.parquet`) and `python3 scripts/build_articles.py` (4,300 cc_news articles, seed 0; fetches the recorded dataset revision from the Hub if it is not cached). Check the rebuild with `python3 scripts/make_manifest.py --verify-articles $RLW/data/articles.parquet` (expects 0 mismatches against `data/articles_index.csv`). `python3 scripts/test_metrics.py` runs the 37 pre-written metric cases.
3. **GPU and CPU jobs**, in the order of `docs/JOBS.md`: `cd $RLW/scripts && CMDS=cmds/<file>.txt sbatch --job-name=E_<name> rl.slurm` (GPU) or `cpu.slurm` (CPU); set your account in both wrappers. Pilot: `J1_rm`, `J2r_arm_a_lr5e5`, `J3r_arm_b_lr5e5` (the 2e-5 runs `J2_arm_a`, `J3_arm_b` failed gate G2 and are kept for the record). Main: `S2_main_{a,b,c}_s{0,1}` (each ≈ 1.2 A100-h; `grpo_train.py` stops cleanly before the job's time limit and resumes from its last checkpoint when the same command file is resubmitted; on the cluster a login-node monitor did the resubmission, not included). `C1` (click-predictor test). Stage 3: `scripts/stage3_submit.sh` submits `S3_gen_final`, `S3_gen_curve`, `C2`, `C3`, `S3_score_final`, `S3_score_curve` and `C4` with dependencies; then `P1`, `C5`, `S2_scale_E1`, `S2_scale_E2`, `C6`-`C9`, `S10_gen_s1`, `S10_score_s1`, `C10`.
4. **Analysis:** copy `runs/stage3`, `runs/scale` and the run logs into the `results/` layout and run `python3 reproduce.py`.

Every selection step (click-predictor checkpoint, z constants, NLI threshold) uses the Upworthy exploratory split, the 300 selection articles or reference items 1-50; every reported test number uses the hold-out split, the 1,000 test articles or items 51-100. Bootstrap CIs resample articles with `numpy.random.RandomState(0)`, 2,000 resamples. Generation is sampled (temperature 1.0 in training, 0.7 at evaluation, seed 0), so a rerun on other GPUs or library versions gives different headlines; the analysis of the committed outputs is exact.

**Not in the repository:** model weights (the click predictor and the six final adapters are on the Hugging Face Hub; intermediate checkpoints are not released, a dual-use decision explained in the paper's ethics appendix), cc_news article text and titles, the Upworthy CSVs (public on OSF), the casing lexicon built from cc_news (`build_articles.py` rebuilds it), the labelling sheets that embed article text, Slurm logs, and debug scripts.

## Data
- **Upworthy Research Archive** (Matias et al., 2021, CC BY 4.0, <https://osf.io/jd64p/>), package files of 03.12.2020. Tests in the maintainers' date window for unreliable randomisation (2013-06-25 to 2014-01-10) are removed, since the released files have no reliability column (`data/pairs_log.json`). Pairs are two headlines from the same test with the same image; training uses the confirmatory split with |z| ≥ 1 (47,932 pairs), selection the exploratory split, test the hold-out split (pairs sharing a headline with training removed). `data/pairs_*.parquet`: test id, image id, winner and loser headline, click rates, impressions, |z|, `clear` (|z| ≥ 1.96).
- **cc_news** (Hugging Face `vblagoje/cc_news`, revision `81eb2ce0`): English articles from 2017-19 with 150-700 words and a 4-20 word title, deduplicated, at most 3% per site, 3,000 train / 300 selection / 1,000 test, cut to the first 450 Qwen tokens. The text is **not** redistributed. `data/ids_*.txt` and `data/articles_index.csv` list our ids (`cc00000` …) with split, domain, date and the SHA-256 of each cut article and title; `scripts/build_articles.py` rebuilds them deterministically and `scripts/make_manifest.py --verify-articles` checks the result. The `final_human` files hold the scores and flags of the publishers' titles without the titles.
- **Generated headlines** (`results/`): one row per headline with `id` (article), `sample`, the raw output `raw`, the post-processed `headline`, `valid` (4-20 words, no line break, mostly ASCII) and `source`. Scores and flags are row-aligned in the sibling `.scores`, `.metrics` and `.posthoc` files.
- **Reference labels** (`labels/`): drafted by an LLM (Claude) from the headline and the cut article only, blind to source, flags and sampling key, and reviewed by the author, who accepted all 100 drafts (`agreed`) and changed one of the 60 checkpoint labels (`claude_draft_agreed`). They measure agreement with LLM-drafted, author-approved judgements, not independent human annotation. Article text columns are removed.
- **Models** (Hugging Face Hub, private until submission): `amrsobhy/real-clicks-fake-headlines-click-predictor` and `amrsobhy/real-clicks-fake-headlines-adapters`. The model cards give the training data, the z-standardisation constants (click predictor mean −3.2443, SD 1.3713; simulated audience mean 0.3457, SD 0.3391) and the dual-use note.

## Provenance
All code, data and results in this repository were designed and produced during the sprint. The design document and pre-registration were written on 14 Nov 2026 before any data were built (`docs/RESULTS_E.md`, addendum A1 also before any data; A2 and A3 before the test outputs they govern), and every job in `docs/JOBS.md` ran on 14-15 Nov 2026. No code or data predates 13 Nov.

| File | Status |
|---|---|
| `scripts/*` | Written 14-15 Nov 2026 and run as is on the cluster, except release edits: cluster paths and account names replaced by `$WORK_ROOT` and `<account>`; `analyze.py` also emits `n_distinct_headlines` and skips title-based statistics when titles are absent; `posthoc.py` also emits `per_arm_pooled`; `reanalysis.py` writes the labelling sheet only when article text is present; `build_articles.py` can fetch the recorded cc_news revision from the Hub |
| `scripts/number_check.py`, `scripts/make_manifest.py`, `reproduce.py` | New, written for this release (16 Nov 2026) |
| `numbers_E.json`, `numbers_E_switched.json`, `posthoc_E.json` | Regenerated by the release scripts. Every value equals the file written on the cluster; the release adds the keys `metric` and `nli_tau` (written by the final `analyze.py` but absent from the earlier cluster run), `table.*.n_distinct_headlines` and `per_arm_pooled` |
| All other numbers files, tables, results and labels | As written on the cluster (14-15 Nov 2026), with article text removed |

## Licence
Code (`scripts/`, `reproduce.py`, `paper/make_fig_stages.py`): MIT, see `LICENSE`. The companion site (`site/`): AGPL-3.0-or-later, see `site/LICENSE`. Data, model outputs, labels, figures, documents and the paper: CC BY 4.0, see `LICENSE-DATA`, which also credits the Upworthy Research Archive (CC BY 4.0). cc_news text is not included. The models used keep their own licences.

## Citation
```bibtex
@misc{sobhy2026realclicks,
  title  = {Real Clicks, Fake Headlines: AI That Chases Clicks Invents the News},
  author = {Sobhy, Amr},
  year   = {2026},
  note   = {AI x Epistemics Research Sprint, Apart Research},
  url    = {https://github.com/lefrenchnewslab/real-clicks-fake-headlines}
}
```

## Contact
Amr Sobhy, Le French News Lab · GitHub [@amrsobhy](https://github.com/amrsobhy) · please report errors as GitHub issues.

## Acknowledgements
This work was granted access to the HPC resources of IDRIS under the allocation 2026-AD011018247 made by GENCI.
