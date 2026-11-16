# JOBS — Block E (budget ≤ 16 A100-h; stop and report at 14)
All jobs: account `<account>@a100`, `-C a100`, 1 GPU, 8 CPUs, wrapper `scripts/rl.slurm` (aborts if the GPU is not an A100). A100-h ledger: `usage.csv`.

| date | job id | name | purpose | QoS | GPU | wall | A100-h | outcome |
|---|---|---|---|---|---|---|---|---|
| 2026-11-14 | 188138 | E_J0 | offline import-and-load test (7 models, judge/NLI/probe/spaCy) + 2-step GRPO smoke | a100-dev | A100-SXM4-80GB | 00:20:00 | 0.333 | load test OK (795 s, Lustre imports); GRPO smoke not reached → TIMEOUT |
| 2026-11-14 | 189389 | E_J0b | smoke re-test with env staged to /dev/shm | a100-dev | A100-SXM4-80GB | <1 min | ~0.01 | FAILED: /dev/shm is noexec |
| 2026-11-14 | 189495 | E_J0b | smoke re-test, env staged to node-local /tmp | a100-dev | A100-SXM4-80GB | ~2 min | ~0.03 | imports 4.4 s; GRPO smoke FAILED: deberta-v3-base has only pytorch_model.bin (not fetched) |
| 2026-11-14 | 189541 | E_J0b | smoke re-test after fetching deberta .bin; 2 GRPO steps arm (c), untrained RM head, BLOCKE_SMOKE=1 | a100-dev | A100-SXM4-80GB | ~3 min | ~0.05 | OK: 12–14 s/step (gen 5.5–7.1 s, train 6–6.7 s), peak 13.8 GB → gen > 5 s rule: pilot runs with GC off (8×8) |
| 2026-11-14 | 189616 | E_J1 | RM + baselines (first try) | a100-dev | A100-SXM4-80GB | 00:01:41 | 0.028 | CANCELLED: RM loss NaN (transformers 5 loaded deberta .bin as fp16); J2/J3 189617/189619 cancelled before start |
| 2026-11-14 | 190697 | E_J1 | RM (fp32 + bf16 autocast) + G1 baselines + base headlines + frozen z-stats | a100-dev | A100-SXM4-80GB | 00:05:30 | 0.092 | COMPLETED: G1 met (0.804 vs 0.606 / 0.715) |
| 2026-11-14 | 190700 | E_J2 | arm (a) lr 2e-5, guard-stop at 30 + resume to 50, G2 (afterok 190697) | a100-dev | A100-SXM4-80GB | 00:10:34 | 0.176 | COMPLETED: resume OK; G2 gain 0.144 (failed) |
| 2026-11-14 | 190701 | E_J3 | arm (b) lr 2e-5, 50 steps, G2 (afterok 190697) | a100-dev | A100-SXM4-80GB | 00:12:50 | 0.214 | COMPLETED: G2 gain 0.077 (failed) |
| 2026-11-14 | 191159 | E_J2r | arm (a) pre-planned retry lr 5e-5, G2 | a100-dev | A100-SXM4-80GB | 00:10:23 | 0.173 | COMPLETED: G2 gain 0.482 (met) |
| 2026-11-14 | 191160 | E_J3r | arm (b) pre-planned retry lr 5e-5 (Addendum A1), G2 | a100-dev | A100-SXM4-80GB | 00:12:24 | 0.207 | COMPLETED: G2 gain 0.322 (met) |

| 2026-11-14 | 191984 | E_C1 | CPU: RM hold-out test + memorisation filter + metrics on pilot outputs | cpu-dev | CPU (40 cores) | 00:14:02 | 0 | COMPLETED |
| 2026-11-14 | 191920 | E_S2_main_a_s0 | arm (a) seed 0, lr 5e-5, N 350 + offline KL | a100-dev | A100-SXM4-80GB | 00:47:xx | see usage.csv | COMPLETED: COLLAPSED at step 240 (pre-registered stop) |
| 2026-11-14 | 191921 | E_S2_main_a_s1 | arm (a) seed 1 + KL | a100-dev | A100-SXM4-80GB | 01:08:11 | 1.136 | COMPLETED (350) |
| 2026-11-14 | 191922 | E_S2_main_b_s0 | arm (b) seed 0 + KL | a100-dev | A100-SXM4-80GB | 01:20:45 | 1.346 | COMPLETED (350) |
| 2026-11-14 | 191923 | E_S2_main_b_s1 | arm (b) seed 1 + KL | a100-dev | A100-SXM4-80GB | 01:20:52 | 1.348 | COMPLETED (350) |
| 2026-11-14 | 191924 | E_S2_main_c_s0 | arm (c) seed 0 + KL | a100-dev | A100-SXM4-80GB | 01:10:09 | 1.169 | COMPLETED (350) |
| 2026-11-14 | 191925 | E_S2_main_c_s1 | arm (c) seed 1 + KL | a100-dev | A100-SXM4-80GB | 01:09:06 | 1.152 | COMPLETED (350) |
| 2026-11-14 | 194056 | E_S3gf | final generations (base + 6 runs, 1,000 test × 4) + human | a100-dev | A100-SXM4-80GB | 00:11:33 | 0.193 | COMPLETED |
| 2026-11-14 | 194057 | E_S3gc | checkpoint curves (seed 0, 300 test × 2) + best-of-16 | a100-dev | A100-SXM4-80GB | 00:10:53 | 0.181 | COMPLETED |
| 2026-11-14 | 194058 | E_C2 | CPU: deterministic metrics on finals + hand-label sheet | cpu-dev | CPU | 00:14:12 | 0 | COMPLETED |
| 2026-11-14 | 194059 | E_S3sf | scoring finals: RM, judge, eval/train NLI, OLMo probe | a100-dev | A100-SXM4-80GB | 00:44:27 | 0.741 | COMPLETED |
| 2026-11-14 | 194060 | E_C3 | CPU: metrics on curves + BoN | cpu-dev | CPU | 00:18:48 | 0 | COMPLETED |
| 2026-11-14 | 194061 | E_S3sc | scoring curves + BoN: RM, judge, NLI | a100-dev | A100-SXM4-80GB | 00:14:02 | 0.234 | COMPLETED |
| 2026-11-14 | 194062 | E_C4 | CPU: analysis + figures | cpu-dev | CPU | 00:06:02 | 0 | COMPLETED |

**Block E total: 9.559 A100-h** (pilot 1.281, Stage 2 6.930, Stage 3 1.349; `usage.csv`).
| 2026-11-14 | 195669 | E_P1 | probe bug dump (first try) | a100-dev | — | 0 | 0 | CANCELLED before start (import bug in dump script) |
| 2026-11-14 | 195680 | E_P1 | probe bug dump, 20 items (post hoc) | a100-dev | A100-SXM4-80GB | 00:01:26 | 0.024 | COMPLETED: no bug |
| 2026-11-14 | 195697 | E_C5 | CPU: spec-zipf sensitivity + invented reaction / off-article (post hoc) | cpu-dev | CPU | — | 0 | COMPLETED |
| 2026-11-14 | 198109 | E_S2_scale_E1 | exploratory E1 Llama-3.1-8B arm (a) seed 0, 250 steps + eval/KL | a100-dev | A100-SXM4-80GB | 00:41:59 | 0.700 | COMPLETED: COLLAPSED at step 110 |
| 2026-11-14 | 198110 | E_S2_scale_E2 | exploratory E2 Qwen2.5-14B arm (a) seed 0 (part 1) | a100-dev | A100-SXM4-80GB | 01:40:38 | 1.677 | COMPLETED: guard stop at step 172 |
| 2026-11-14 | 202027 | E_S2_scale_E2 | E2 resume from checkpoint-172 → 250 + eval/KL | a100-dev | A100-SXM4-80GB | 01:11:23 | 1.190 | COMPLETED (DONE, EVAL_DONE) |
| 2026-11-14 | 204077 | E_C6 | CPU: scale metrics + summary + figE1 | cpu-dev | CPU | 00:11:45 | 0 | COMPLETED |
| 2026-11-15 | 204657 | E_C7 | CPU: M1 hand-label analysis | cpu-dev | CPU | — | 0 | COMPLETED |
| 2026-11-15 | 204831 | E_C8 | CPU: H1–H4 with switched metric | cpu-dev | CPU | — | 0 | COMPLETED |
| 2026-11-15 | 207102 | E_C9 | CPU: weighted detector validation, trajectories, intermediate label sheet | cpu-dev | CPU | — | 0 | COMPLETED |
| 2026-11-15 | 211832 | E_S10g | seed-1 curve generation (21 checkpoints × 300 test × 2) | a100-dev | A100-SXM4-80GB | 00:09:32 | 0.159 | COMPLETED |
| 2026-11-15 | 211996 | E_S10s | seed-1 curve scoring (click model, judge, training NLI) | a100-dev | A100-SXM4-80GB | 00:08:31 | 0.142 | COMPLETED |
| 2026-11-15 | 211997 | E_C10 | CPU: seed-1 metrics, trajectories, matched-KL H3 by seed, figE1 | cpu-dev | CPU | — | 0 | COMPLETED |

**Block E total: 13.451 A100-h** (main study 9.559, post-hoc probe check 0.024, exploratory scale check 3.567, seed-1 trajectories 0.301; `usage.csv`).

