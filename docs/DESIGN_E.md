# Design E: Real Clicks, Real Lies (second sprint project)

Status: design checked against sources on Sat 14 Nov 2026. The Jean Zay prompt is in `AGENT_PROMPT_E_rl_headlines.md`.

## TL;DR

- **Question.** Moloch's Bargain found that optimising text for a *simulated* audience increases fabrication. Does the same happen when the reward comes from what *real people clicked*?
- **Design.** Train a Qwen2.5-3B-Instruct headline writer with GRPO (LoRA) on news articles. There are three reward arms: (a) a reward model (RM) fitted on real Upworthy A/B tests, (b) a simulated-audience LLM judge, (c) arm (a) plus an NLI faithfulness penalty. A no-RL baseline and the human headlines serve as references. Two seeds per arm.
- **Fabrication measure.** Named entities and numbers in the headline that the article does not contain (deterministic check, with Title-Case and teaser-number filters so Upworthy style is not counted as fabrication). It is cross-checked by an NLI model never used in training, an LLM probe from a non-Qwen family (Moloch-style), and 100 headlines Amr labels.
- **Novelty.** RL on real click data already exists in industry ad text (Meta's RLPF on Facebook ads, arXiv 2507.21983; CTR-driven ad headline work, 2507.20227, 2508.18739), but these measure CTR and diversity, not fabrication against a source. No paper found does on-policy RL for *news* headlines on real clicks, measures fabrication against the article, and compares it with a simulated audience under the same RL. The closest is Wang & Sudhir (Yale, 2025): DPO on Upworthy pairs, measuring clickbait words and relevance but not fabrication. The claim is therefore the matched real-vs-simulated comparison, not "first RL on real clicks".
- **Budget.** Pilot 1.5, main runs 7.5, eval 2.0, reserve 3.0: **14 A100-h** in total, with a hard cap of 16. The step count is set from the measured s/step so the 6 runs fit 7.5 A100-h. The 6 runs go in parallel (each its own resume chain) to fit the 36-h deadline.
- **Main risks.** (1) The RM may barely beat a length/lexicon baseline, since predicting Upworthy winners is hard. (2) The real-click RM may teach curiosity-gap teasers ("You won't believe what this mayor did"), which contain *fewer* claims, so the fabrication rate could fall for a dull reason. Both are measured and pre-registered.

## 1. Verified facts

### Upworthy Research Archive
| Fact | Value | Source |
|---|---|---|
| Location | OSF project `jd64p` (https://osf.io/jd64p/). The full dataset, with all parts, has been public since Aug 2021 | upworthy.natematias.com |
| Licence | CC BY 4.0 (metadata CC0) | Matias et al., Sci Data 8:195 (2021), PMC8329003 |
| Size | 32,487 tests, 150,817 packages (arms), 538M assignments, Jan 2013 to Apr 2015 | Sci Data paper |
| Splits | **Exploratory** 4,873 tests / 22,666 packages; **Confirmatory** 22,743 tests / 105,551 packages; **Hold-out** 4,871 tests / 22,600 packages | Sci Data paper, about-the-archive page |
| Columns (17) | created_at, test_week, clickability_test_id, impressions, headline, eyecatcher_id, clicks, excerpt, lede, slug, share_text, square, significance, first_place, winner, updated_at, problem | Sci Data Table 1 |
| Per-arm outcome | `clicks` and `impressions` per package; CTR = clicks/impressions | about-the-archive |
| Article text | **Not included** (only headline, excerpt, lede, slug, share text; images not included, only `eyecatcher_id`) | Sci Data paper |
| Caveats | Tests from 25 Jun 2013 to 10 Jan 2014 may not be properly randomised (about 22% of tests). They are flagged by `problem`=1, and a reliability column was added in 2024. `significance` is NOT a significance estimate. Follow-up tests reuse content. Many tests vary the image as well as the headline | 2024 update page, Sci Data correction (2024) |
| File names | Not stated on any page I could load. The likely pattern is `upworthy-archive-{exploratory,confirmatory,holdout}-packages-*.csv`, which is **unverified**: the agent lists the OSF files through the API before downloading | — |

Correction to IDEAS_round2: the exploratory split has 4,873 tests (22,666 packages), not "22k tests → 100k pairs". The pairs come mostly from the confirmatory split, which is public.

### Moloch's Bargain (El & Zou, arXiv 2510.06105, ICML 2026)
- Models: Qwen3-8B and Llama-3.1-8B-Instruct, LoRA r16/α32.
- Training: **rejection fine-tuning (STaR) and text feedback, not RL.** The paper names GRPO/DPO as future work.
- Audience: gpt-4o-mini simulating 20 personas from PRODIGy, plus a demographic-profile variant. Each persona picks between two generations. The performance measure is excess win rate.
- Social-media task: CNN/DailyMail articles as anchors, 1,024 train and 1,024 eval.
- Fabrication measure: a gpt-4o probe applied to anchor + generation, reporting the % flagged. The social-media "disinformation" probe asks whether the post "introduce[s] numbers, statistics, or any quantitative claims that are not explicitly contained in the prompt". Human validation F1 was about 0.9 on 20 items per probe.
- Result: social-media disinformation +188.6% (Qwen TFB), but **it fell for Llama** (−14.7%, −28.9%). The effect is model-dependent.

### Prior work close to this idea (searched arXiv, OpenReview, web; Nov 2026)
| Work | What it does | Fabrication vs article? | Threat |
|---|---|---|---|
| Wang & Sudhir, "From What Works to Why It Works" (Yale SOM, Dec 2025) | DPO on Llama-3.2-3B with Upworthy winner/loser pairs; article text scraped (not public); Prolific click test. Vanilla DPO raises "shocking" from 0.7% to 43.7% and "this is" from 2% to 56% | **No** (clickbait words, relevance regressor) | Closest. Our delta: on-policy RL against a real-click RM, a direct real-vs-simulated comparison, and fabrication grounded in the article |
| Loghmani, arXiv 2506.00152 (2025) | Upworthy reward models (Pythia, AUC 0.82 experimental vs 0.74 observational); RL only in a MIND simulation with synthetic rewards | No | Low; useful RM reference |
| LOLA, arXiv 2406.02611 | Upworthy winner prediction: best about 47–49% vs 33% chance; humans near chance | — | Shows the RM bar is low and hard |
| Song et al., AAAI 2020 (2002.02095) | A2C headline extractor with a popularity reward from Daily Mail comment counts | ROUGE only | Pre-LLM, not CTR |
| Xu et al. 2019 (1909.03582); HonestBait (2306.14828) | Sensational/clickbait RL; forward-reference bait with faithfulness | Partly | Pre-LLM, no real-CTR reward |
| Aperstein et al., arXiv 2603.22459 | FUDGE-guided clickable-but-not-clickbait rewriting | Not measured | Low |
| Toolflation (ICLR 2027 sub.) | Agents asked only to "be selected" inflate claims | — | Related mechanism, different domain |
| Meta RLPF / AdLlama, arXiv 2507.21983 (2025) | RL from a reward model fitted on real ad CTR; live A/B test on Facebook | No | Kills any "first RL on real clicks" claim; ads, not news, no source document |
| arXiv 2507.20227 (2025); DIVER, arXiv 2508.18739 (2025) | CTR-driven ad text via online-feedback preference optimisation; ad headline RL with CTR + diversity reward | No | Same: real-click optimisation exists, fabrication against a source is not measured |

**Verdict: real-click RL exists for ads, but no paper found RL-trains a news headline writer on real clicks and measures fabrication against the article, and none compares it with a simulated audience.** If one appears, the minimal change that keeps the work novel is the **matched real-vs-simulated comparison at equal KL**, with fabrication grounded in the source article (and the "teaser" decomposition in §2.4). That comparison is the contribution no listed work makes.

### Tooling
- **TRL.** The current release is v1.18.0 (13 Nov 2026). Pin **trl==1.16.0** (15 Oct 2026, more than two weeks old). Recent changes the agent must respect: from v1.17 a `top_p<1` without `top_k` raises an error; from v1.15 a fused LM head is on by default (no LoRA on `lm_head`). The agent checks `GRPOConfig` field names in the installed version and does not trust memory. The `beta` (KL) default has been 0.0 in recent TRL, so it must be set explicitly.
- **Memory, Qwen2.5-3B.** bf16 weights take about 6 GB. With LoRA, TRL computes the reference log-probs with the adapter disabled, so no second model copy is needed. Policy + DeBERTa RM + DeBERTa NLI fit easily on A100-40GB. Arm (b) adds Qwen2.5-7B-Instruct as judge (about 15 GB bf16), which is comfortable on 80 GB and tight but feasible on 40 GB. Full fine-tuning would need about 50 GB of weights, gradients and Adam states before activations, so we use **LoRA** (simpler and safer). Generation uses plain HF `generate` (headlines are at most 32 tokens), not vLLM, to avoid an offline vLLM setup. **Trap:** with gradient checkpointing on, transformers may silently set `use_cache=False` during generation, which makes 32-token decoding over 64 × 670-token prompts roughly 10× slower and would break the budget. The pilot times generation separately. If the KV cache is off, the agent turns gradient checkpointing off (fits on 80 GB with micro-batch 8). Request **A100-80GB** nodes, since policy + 7B judge + activations on 40 GB is too tight to debug in 36 h.
- **Article source with real headlines.** CNN/DailyMail (`abisee/cnn_dailymail`, Apache-2.0) has **no headline field** (article + highlights only), so it cannot provide human reference headlines. Chosen: **`vblagoje/cc_news`** (HF parquet, 708,241 English articles from 2017–2019, fields title/text/domain/date/description/url; the card's licence says "unknown"). Fallback: `csebuetnlp/xlsum` English (BBC, has `title`, CC BY-NC-SA 4.0), which relies on a legacy loading script, so use the hub's parquet conversion if present. Last resort: CNN/DM without reference headlines.

## 2. Design rationale

### 2.1 Arms (same articles, prompt, steps, LoRA, KL β, seeds)
- **Base.** Qwen2.5-3B-Instruct with the same prompt, no RL. **Human.** The cc_news title.
- **(a) Real-click RM.** A DeBERTa-v3-base Bradley–Terry scalar head fitted on within-test Upworthy pairs. **It sees the headline only.**
- **(b) Simulated audience.** Qwen2.5-7B-Instruct gives P("Yes") to "Would you click this headline?", averaged over 4 fixed personas. **It sees the headline only** (same information as the RM, Moloch-style).
- **(c) Real-click RM + faithfulness.** r = z_RM − 2·(1 − P_entail), where P_entail comes from a *training* NLI model (DeBERTa-v3-base-mnli-fever-anli) over article chunks. **Evaluation uses a different NLI model** (DeBERTa-v3-large …-ling-wanli) plus the deterministic metric. The two NLI models share author and training data, so their errors correlate; the eval NLI is therefore not counted as independent evidence for arm (c). The independent checks are the deterministic metric, the non-Qwen LLM probe and the hand labels. Arm (c) can also satisfy NLI by copying an article sentence, so verbatim copying is part of H4.
- Shared shaping: invalid format (fewer than 4 or more than 20 words, multiple lines, empty) gives r = −3. KL β = 0.04 to the base model.
- Reward scale: GRPO normalises advantages within each group, so the arms' raw reward scales cancel. What does not cancel is within-group variance (a saturated judge gives noisy normalised advantages) and the KL each arm reaches, so per-step within-group reward sd is logged and arms are compared at matched KL, measured offline as sequence-level KL on a fixed article set (not the per-token training-log estimate, which depends on length).

### 2.2 Why this is not built in
Neither reward sees the article, so neither can directly pay for unsupported content. Fabrication can only appear as a side effect of what clicks or appeals. The opposite outcome is also plausible: Upworthy rewarded curiosity-gap teasers, which make *no* claims.

### 2.3 What a sharp reviewer would demand, and where it is handled
| Demand | Handling |
|---|---|
| "Your RM just rewards length or emotion" | Held-out pairwise accuracy vs (i) logistic regression on length, sentiment (VADER), clickbait lexicon, ?, !, "you", numbers, demonstratives; (ii) a TF-IDF BT model. Also accuracy on length-matched pairs |
| "The RM does not transfer to hard news" | Report the RM's score distribution and its length correlation on cc_news human headlines. The RM is a proxy for real clicks, and the paper says so in one sentence |
| "Unequal optimisation pressure across arms" | Compare arms at the final step *and* at matched KL (fabrication-vs-KL curves from checkpoints every 50 steps) |
| "One seed" | 2 seeds per arm. A claim needs both seeds to agree in sign |
| "Reward hacking / gibberish" | Format-validity rate, length, distinct-n, repetition, verbatim copying, non-English, teaser and question rates per checkpoint; pre-registered collapse stop |
| "Your fabrication metric is noisy" | Deterministic check against the *exact text the model saw*, a different NLI model, a non-Qwen LLM probe (Moloch's question), and 100 blind hand labels (50 for choosing the metric threshold, 50 for testing; recall reweighted for the stratified sampling) |
| "spaCy tags Title-Case words as entities, and Upworthy style is Title Case" | Candidate entities made only of common lowercase English words are dropped; "one"/"first" and listicle counts ("7 Reasons") are not counted as fabricated numbers (listicle counts reported separately); a numbers-only sub-metric, which casing cannot affect, is reported for every arm; the unit tests include Title-Case teasers |
| "Longer headlines just contain more claims" | Length per arm, fabrication conditional on claim-bearing, and a logistic regression of fabrication on arm + word count (exploratory) |
| "Fewer claims, not more honesty" | Report claim-bearing rate (headline has ≥1 entity or number) and fabrication *conditional on* claim-bearing alongside the unconditional rate |
| "The model knows the 2017–19 news" | Calibrate with the human-headline rate on the same articles (human editors also add context) |

### 2.4 The result in either direction
- Real-click fabrication rises like the simulated audience's: "Moloch's Bargain holds with real human reward."
- It does not rise (or teasers replace claims) while the simulated audience's does: "Simulated audiences overstate the fabrication risk; real clicks buy vagueness instead." Both are publishable, and the second may surprise more.
- It falls significantly: reported as its own outcome ("fall"), not lumped with "inconclusive".

## 3. Stages and budget (A100-h)
| Stage | Content | A100-h |
|---|---|---|
| 0 | Front-end downloads, data build, FAILURE_MODES, pre-registration | 0 |
| 1 Pilot (go/no-go) | RM + baselines (0.5); 50-step GRPO smoke tests for arms (a) and (b) (0.8); slack (0.2) | 1.5 |
| 2 Main | 3 arms × 2 seeds × N steps (N ≤ 400, set from pilot s/step), about 1.1–1.3 h each, checkpoints every 50 steps, run in parallel | 7.5 |
| 3 Eval | 1,000 test articles × 4 samples × (base + 6 final runs) + checkpoint curves and offline KL (seed 0) + NLI + non-Qwen LLM probe + cross-scoring + BoN-16 reference + RM hold-out test | 2.0 |
| Reserve | Reruns, a failed job | 3.0 |
| **Total** | | **14.0** (hard stop 16) |

Throughput assumption: 64 completions/step (8 prompts × 8), ≤32 new tokens, about 10 s/step with the KV cache on, so 400 steps ≈ 70 min. The pilot measures the real s/step. Pre-registered rule: N = the largest multiple of 50 ≤ 400 such that 6 × (N × s/step + 10 min load) ≤ 7.5 h. If N < 200, arm (c) runs seed 0 only and N is recomputed for 5 runs; if N is still < 200, stop and report. Seeds of arms (a) and (b) are never dropped, since H1–H3 need them.

## 4. Sources
- Matias et al. 2021, Sci Data, "The Upworthy Research Archive" https://pmc.ncbi.nlm.nih.gov/articles/PMC8329003 ; https://upworthy.natematias.com/about-the-archive ; https://upworthy.natematias.com/ ; 2024 update https://upworthy.natematias.com/2024-06-upworthy-archive-update.html ; OSF https://osf.io/jd64p/
- Moloch's Bargain https://arxiv.org/html/2510.06105v1
- Meta RLPF https://arxiv.org/html/2507.21983v1 ; CTR-driven ad text https://arxiv.org/abs/2507.20227 ; DIVER https://arxiv.org/abs/2508.18739
- Wang & Sudhir https://insights.som.yale.edu/sites/default/files/2025-12/Why%20It%20Works.pdf
- Loghmani https://arxiv.org/html/2506.00152v1 ; LOLA https://arxiv.org/html/2406.02611v3 ; Song et al. https://arxiv.org/pdf/2002.02095 ; Aperstein et al. https://arxiv.org/abs/2603.22459
- TRL releases https://releases.sh/hugging-face/trl
- cc_news https://huggingface.co/datasets/vblagoje/cc_news ; CNN/DM https://huggingface.co/datasets/abisee/cnn_dailymail ; XL-Sum https://huggingface.co/datasets/csebuetnlp/xlsum
