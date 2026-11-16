# FAILURE_MODES — Block E "Real Clicks, Real Lies"

Written 2026-11-14, before any code, data build, rollout or test output. Each row: what can go wrong → how the design handles it → how we detect it.

## E. Section E failure modes

| # | Failure mode | Handling in the design | Detection / check |
|---|---|---|---|
| E1 | **RM learns length or lexicon instead of clicks** | RM must beat a feature BT baseline (chars, words, has-number, ?, !, "you", demonstrative start, clickbait lexicon, VADER) by ≥ 3 pp (G1). TF-IDF BT baseline and length-matched accuracy reported | G1 table; RM score vs length correlation on cc_news human headlines |
| E2 | **Domain shift** (Upworthy 2013–15 viral stories vs 2017–19 hard news in cc_news) | RM framed as a proxy for real clicks; RM score distribution on cc_news human headlines reported | Exploratory: RM score vs length on cc_news titles; RM score of human vs base headlines |
| E3 | **Policy hacks the RM** with gibberish, repetition, teasers | Format shaping r = −3 for invalid outputs (4–20 words, no newline, non-ASCII ≤ 20%); pre-registered collapse stop; KL β = 0.04 | Per-step format-valid rate, distinct-per-group, distinct-1/2, repetition, teaser rate, length |
| E4 | **Curiosity-gap teasers lower fabrication only because they contain no claims** | Report claim-bearing rate and fabrication conditional on claim-bearing next to the unconditional rate; teaser/forward-reference regex rate per checkpoint | RESULTS tables; hand-label category `teaser_no_claim` |
| E5 | **Entities the model adds from background knowledge** (cc_news 2017–19, Qwen knows the events) | Counted as fabrication against the article (the article is the only allowed source, as in Moloch); human-headline rate on the same articles is the calibration | Human-headline fabrication rate reported |
| E6 | **Entity aliasing** ("U.S." vs "United States", surname-only persons) | Alias table incl. demonym ↔ country; PERSON surname matches full name; fuzzy partial_ratio ≥ 90 | 4+ alias/demonym unit tests |
| E7 | **Number formats** ("two" vs "2", "1.5m" vs "1.5 million", % vs percent) | Number normalisation (word2number, m/bn/k multipliers, percent) on both sides | Number unit tests |
| E8 | **Fabrication checked against the full article instead of the truncated text the model saw** | Truncated 450-token text stored in the article file and used everywhere (prompt, metric, NLI) | Code path: one field `article_trunc`; assertion in eval that it is the same string as in the prompt |
| E9 | **NLI fails on headline style** (headlines are telegraphic, Title Case) | NLI is secondary; deterministic metric primary; τ chosen on hand-label selection half | Hand-label precision/recall for NLI vs deterministic |
| E10 | **Arm (c) Goodharts the evaluation** | Training NLI (base) ≠ eval NLI (large); deterministic metric + non-Qwen LLM probe + hand labels are the independent checks; eval NLI not gating for H4 | H4 uses deterministic + probe |
| E11 | **The two NLI models share training data** (same author, MNLI/FEVER/ANLI) | Eval NLI declared not independent evidence for arm (c) in the pre-registration | — (stated) |
| E12 | **Arm (c) satisfies NLI by copying** article sentences | Verbatim 8-gram copy rate is part of H4 (≤ base + 10 pp) | 8-gram copy rate per arm |
| E13 | **Unequal optimisation pressure across arms** | Matched-KL comparison using offline sequence-level KL on a fixed set (300 selection articles × 2 samples) per checkpoint | H3 at matched KL; KL-ratio rule (> 2× → only matched-KL interpreted) |
| E14 | **Judge sees more than the RM** | Both see the headline only; arm (c) penalty is the only article-aware component | Code review of reward functions |
| E15 | **Seed noise** | 2 seeds per arm; directional claim needs both seeds to agree in sign; seeds reported separately | Seed rule in RESULTS |
| E16 | **KL blow-up or mode collapse** | β = 0.04, LoRA, lr 2e-5; pre-registered collapse stop (format-valid < 80% or distinct < 2 for 3 consecutive 10-step intervals) recorded as "collapsed", not rerun | Monitor + training log |
| E17 | **Chat-template leakage** (e.g. "Headline:", quotes, `<|im_end|>` text, "assistant") | One shared post-processing function; skip_special_tokens; format check | Rate of residual "Headline"/quotes/special tokens in outputs; unit test on postprocess |
| E18 | **Upworthy randomisation-problem tests** | Drop `problem==1` and anything flagged unreliable by the 2024 column | Pair-build log counts |
| E19 | **Headline leakage across Upworthy splits** (follow-up tests reuse content) | Drop selection/test pairs whose normalised headline string occurs in train | Pair-build log: number dropped |
| E20 | **Same headline with different images** (image confound) | Pairs only within the same `eyecatcher_id`, different headline text; duplicate (test, image, headline) packages summed | Pair-build code |
| E21 | **TRL API drift** | Pin trl==1.16.0, print `inspect.signature(GRPOConfig)` and use the field names it shows; set beta, loss_type, scale_rewards, num_iterations, steps_per_generation, mask_truncated_completions explicitly | Signature dump saved to `logs/`; config dumped per run |
| E22 | **Dev-QoS timeouts** (TIMEOUT breaks `afterok`) | Wall-clock guard callback saves and exits 0 at 1 h 45 min; next job resumes from last checkpoint; no-op if target step reached | Pilot: kill J2 at step 30, resume, check step counter and loss continue |
| E23 | **Generation without KV cache under gradient checkpointing** (~10× slower) | `use_cache=True` in generation kwargs if allowed; gen and train time logged separately; if gen > 5 s/step turn GC off (micro-batch 8 × accum 8) | Pilot timing log |
| E24 | **spaCy tags Title-Case words as entities** (style counted as fabrication) | Title-Case filter: drop candidates whose every token is common lowercase English (wordfreq zipf ≥ 4.5) and not capitalised mid-sentence in the article; keep acronyms | ≥ 8 Title-Case teaser unit tests; numbers-only sub-metric (casing-proof); precision on arm (a) vs base hand labels |
| E25 | **Teaser numbers** ("This One Trick", "one man", "first") and listicles | "one"/"1" as determiner/pronoun and ordinal "first" not counted; listicle counts reported separately | ≥ 4 listicle unit tests |
| E26 | **Length differences across arms** | Mean word count per arm; exploratory logistic regression of fabrication on arm + word count | RESULTS table |
| E27 | **Accidentally running on H100** | `-C a100`, account `<account>@a100`; every job prints `nvidia-smi` and the python entry aborts unless GPU name contains "A100" | `gpu_name.txt` per job; abort check |
| E28 | **Compute nodes offline** → model/dataset fetch fails mid-job | Everything downloaded on the front-end; `HF_HUB_OFFLINE=1` etc.; 2-minute offline import-and-load test job before the pilot | Test job log |
| E29 | **Fuzzy matching meaningless on short strings** (partial_ratio on "UN") | Candidates < 5 chars require an exact case-insensitive token match | Unit tests |
| E30 | **Judge (arm b) saturates** (all P(Yes) ≈ 1 or 0) → noisy normalised advantages | Within-group reward sd logged per step; z-scoring with base stats | Pilot: judge score sd on base headlines |
| E31 | **Reward computed on unprocessed text** (quotes, "Headline:") differs from evaluated text | Rewards and metrics both call the same `postprocess()` | Shared module `common.py` |
| E32 | **Test data touched before final eval** | Test articles/pairs only read in Stage 3 scripts; selection used for every threshold | Script audit; JOBS.md |
| E33 | **Budget overrun** | usage.csv cumulative; stop and report at 14 A100-h; N computed from pilot s/step | usage.csv |
| E34 | **Upworthy reliability column absent / file names differ** | List OSF files via API, do not hardcode; log column mismatch | build_pairs log |
| E35 | **cc_news load fails / licence unknown** | Fallback xlsum parquet; licence stated in README | build_articles log |
| E36 | **Duplicate / near-duplicate articles across train/selection/test** | Dedup by title and first 200 chars before sampling; disjoint ID lists | build_articles log |
| E37 | **Parallel jobs exceed dev-QoS limits** (10 submitted jobs/user; GPU cap) | 6 chains × ≤ 1 pending + 1 running ≤ 10 when chains are submitted job-by-job by the monitor | Monitor log; `QOSMaxSubmitJobPerUserLimit` |
| E38 | **LoRA on fused lm_head** (TRL ≥ 1.15) | target_modules excludes lm_head | Config dump |
| E39 | **Matched-KL interpolation outside range** (one arm never reaches the other's KL) | Match at the smaller of the two final KLs (always inside both curves) | Code |
| E40 | **Probe/judge P(Yes) read from wrong token** ("Yes" vs " Yes") | Sum probability over the token variants of Yes/No; log the tokens used | Pilot log |
