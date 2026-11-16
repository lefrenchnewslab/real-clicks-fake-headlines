<script lang="ts">
  import { MorphText, Segmented, Swap } from "@stdlogits/ledger";
  import { onMount } from "svelte";
  import { D, ARM_COLOR } from "../lib";
  import Mark from "./Mark.svelte";

  // 4,000 test headlines (1,000 held-out articles × 4) per writer, drawn as 400 tiles of 10.
  // Tiles share a colour only when they hold the same headline.
  type K = "base" | "main_a_s0" | "main_b_s0" | "main_c_s0";
  const STATES: { value: K; label: string; hint: string; inv: number; color: string }[] = [
    { value: "base", label: "Untrained", hint: "Qwen2.5-3B-Instruct before any RL", inv: 0.01725, color: ARM_COLOR.base },
    { value: "main_a_s0", label: "Clicks", hint: "After RL on the click predictor (seed 0, stopped at step 240 when it collapsed)", inv: 0.95325, color: ARM_COLOR.a },
    { value: "main_b_s0", label: "Audience", hint: "After 350 RL steps on the simulated audience (seed 0)", inv: 0.0185, color: ARM_COLOR.b },
    { value: "main_c_s0", label: "Penalty", hint: "After 350 RL steps on clicks with a faithfulness penalty (seed 0)", inv: 0.01325, color: ARM_COLOR.c },
  ];
  let k = $state<K>("base");
  const st = $derived(STATES.find((s) => s.value === k)!);
  const C = $derived(D.collapse[k]);
  const N = 400;
  // Assign tiles: each of the top headlines gets tiles in proportion to its count; the rest are unique.
  const tiles = $derived.by(() => {
    const out: { c: number; top: number }[] = [];
    let i = 0;
    C.top.forEach(([, n]: [string, number], j: number) => {
      const t = Math.round((n / C.n) * N);
      if (t >= 2) for (let q = 0; q < t && i < N; q++, i++) out.push({ c: j, top: 1 });
    });
    for (; i < N; i++) out.push({ c: 100 + i, top: 0 });
    return out;
  });
  const hash = (x: number) => ((Math.sin(x * 127.1) * 43758.5453) % 1 + 1) % 1;
  let shown = $state(false);
  onMount(() => requestAnimationFrame(() => (shown = true)));
</script>

<section class="hero part">
  <p class="kicker">Apart AI x Epistemics sprint · pre-registered · RL on public click data</p>
  <h1>Real clicks, fake headlines</h1>
  <p class="byline"><span class="au">Amr Sobhy</span><span class="sep" aria-hidden="true">·</span><span>Le French News Lab</span></p>
  <p class="lede">
    We trained a headline writer with reinforcement learning to win clicks, as predicted from real
    A/B tests. It began inventing claims within 100 steps, then dropped the news altogether and wrote
    the same fabricated viral story for every article.
  </p>

  <div class="reader-route">
    <a class="primary-link" href="#train">Explore the training <span aria-hidden="true">↓</span></a>
    <a href="#final">Jump to the results →</a>
    <span>Interactive research companion · about 8 minutes</span>
  </div>
  <p class="note">Here, “clicks” means a model’s prediction learned from real A/B tests. No live readers saw these generated headlines.</p>
  <div class="surface field" style:--tint={st.color}>
    <p class="label">Start here: switch the reward and see what happens to headline diversity.</p>
    <div class="controls">
      <Segmented label="Writer" value={k} onchange={(v) => (k = v)} options={STATES} />
    </div>
    <div class="body">
      <div class="grid" role="img" aria-label={`${C.distinct} distinct headlines among 4,000; coloured blocks show frequent repeated headlines`}>
        {#each tiles as t, i}
          <i
            style:background={!shown
              ? "var(--wash2)"
              : t.top
                ? `color-mix(in oklch, var(--tint) ${100 - t.c * 18}%, var(--wash))`
                : `color-mix(in oklch, var(--ink3) ${10 + Math.round(hash(i + 1) * 26)}%, var(--wash))`}
            style:transition-delay="{(i % 40) * 6 + Math.floor(i / 40) * 10}ms"
          ></i>
        {/each}
      </div>
      <div class="stats">
        <div class="stat">
          <span class="big num"><MorphText text={C.distinct.toLocaleString("en")} /></span>
          <span class="lab">different headlines among 4,000</span>
        </div>
        <div class="stat">
          <span class="big num"><MorphText text={`${(st.inv * 100).toFixed(1)}%`} /></span>
          <span class="lab">contain a number not in the article</span>
        </div>
        <div class="sample">
          <span class="lab">Most common headline</span>
          <Swap key={k}><p class="h">“<Mark h={C.top[0][0]} tint={st.color} />”</p><p class="lab num">{C.top[0][1].toLocaleString("en")} of 4,000</p></Swap>
        </div>
      </div>
    </div>
  </div>
  <p class="note">{st.hint}.</p>
  <p class="note">Seed 0 only: 4,000 headlines for 1,000 held-out articles. Each tile represents approximately 10 outputs. Coloured blocks show frequent repeats; grey tiles group the remaining outputs. The results below pool both seeds.</p>
</section>

<style>
  .hero { padding-top: var(--s7); }
  h1 { font: var(--type-title); letter-spacing: -.045em; max-width: 14ch; text-wrap: balance; }
  .byline { display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--s2); margin-top: calc(var(--s3) * -1); color: var(--ink2); font: var(--type-lead); }
  .byline .au { color: var(--ink); font-weight: 600; }
  .byline .sep { color: var(--ink3); }
  .field { display: grid; gap: var(--s5); margin-top: var(--s2); }
  .body { display: grid; gap: var(--s5); }
  .grid { display: grid; grid-template-columns: repeat(40, minmax(0, 1fr)); gap: 3px; }
  .grid i { display: block; aspect-ratio: 1; border-radius: 3px; transition: background-color 700ms var(--ease); }
  .stats { display: grid; grid-template-columns: auto auto minmax(0, 1fr); gap: var(--s6); align-items: start; }
  .stat { display: grid; gap: var(--s1); }
  .big { font: 600 44px/1 var(--display); letter-spacing: -.03em; }
  .lab { color: var(--ink3); font: var(--type-small); }
  .sample { display: grid; gap: var(--s2); }
  .h { font: var(--type-label); color: var(--ink); max-width: 40ch; }
  @media (max-width: 760px) { .stats { grid-template-columns: 1fr; } .grid { grid-template-columns: repeat(20, minmax(0, 1fr)); } h1 { font: 600 40px/1.05 var(--display); } .big { font-size: 34px; } }
</style>
