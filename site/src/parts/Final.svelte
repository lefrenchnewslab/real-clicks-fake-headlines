<script lang="ts">
  import { Segmented, Swap } from "@stdlogits/ledger";
  import { D, ARM_COLOR } from "../lib";
  import Mark from "./Mark.svelte";

  const WRITERS = [
    { k: "base", name: "Untrained", tint: ARM_COLOR.base },
    { k: "main_a_s0", name: "Clicks · seed 0", tint: ARM_COLOR.a },
    { k: "main_a_s1", name: "Clicks · seed 1", tint: ARM_COLOR.a },
    { k: "main_b_s0", name: "Audience · seed 0", tint: ARM_COLOR.b },
    { k: "main_b_s1", name: "Audience · seed 1", tint: ARM_COLOR.b },
    { k: "main_c_s0", name: "Penalty · seed 0", tint: ARM_COLOR.c },
    { k: "main_c_s1", name: "Penalty · seed 1", tint: ARM_COLOR.c },
  ];
  let art = $state(0);
  const A = $derived(D.same[art]);

  // Paper Tables 2 and 5: final checkpoints, 1,000 test articles × 4 headlines; [pooled, seed 0, seed 1].
  const MEAS = [
    { value: "num", label: "Invented numbers", hint: "A number whose value does not appear in the article" },
    { value: "react", label: "Invented reactions", hint: "A reaction word the article does not use" },
    { value: "ent", label: "Entity check", hint: "A name or number not found in the article; also counts words such as “Sparks”" },
    { value: "nli", label: "NLI: not supported", hint: "An NLI model (DeBERTa-v3-large) finds the headline unsupported, τ = 0.5" },
  ] as const;
  type MKey = (typeof MEAS)[number]["value"];
  const ROWS: { name: string; tint: string; v: Record<MKey, number[]> }[] = [
    { name: "Real titles", tint: "var(--ink2)", v: { num: [3.6], react: [0.3], ent: [14.6], nli: [22.2] } },
    { name: "Untrained writer", tint: ARM_COLOR.base, v: { num: [1.7], react: [2.2], ent: [12.1], nli: [24.6] } },
    { name: "Click predictor", tint: ARM_COLOR.a, v: { num: [97.6, 95.3, 99.8], react: [99.1, 99.1, 99.1], ent: [97.6, 95.3, 99.8], nli: [96.0, 95.8, 96.2] } },
    { name: "Simulated audience", tint: ARM_COLOR.b, v: { num: [1.7, 1.8, 1.5], react: [27.0, 23.7, 30.2], ent: [42.5, 38.0, 47.1], nli: [52.9, 48.6, 57.2] } },
    { name: "Clicks + penalty", tint: ARM_COLOR.c, v: { num: [2.5, 1.3, 3.6], react: [13.5, 4.2, 22.9], ent: [15.8, 15.6, 15.9], nli: [19.6, 23.1, 16.1] } },
  ];
  let m = $state<MKey>("num");
</script>

<section class="part" id="final">
  <p class="kicker">The finished writers</p>
  <h2>Same article, seven writers</h2>
  <p class="lede">Pick a held-out article and compare what each finished writer wrote for it. Numbers the article does not contain are marked solid; reaction words are marked lightly.</p>

  <div class="controls">
    <span class="label">Article</span>
    <div class="pick" role="group" aria-label="Article">
      {#each D.same as s, i}
        <button type="button" aria-pressed={i === art} onclick={() => (art = i)} data-tip={`Untrained headline: ${s.base.h}`}>{i + 1}</button>
      {/each}
    </div>
  </div>

  <Swap key={art}>
    <div class="surface rows">
      {#each WRITERS as w, i}
        <div class="row rise" style:--i={i}>
          <span class="who"><i style:background={w.tint}></i>{w.name}</span>
          <span class="h"><Mark h={A[w.k].h} nums={A[w.k].num} tint={w.tint} /></span>
        </div>
      {/each}
    </div>
  </Swap>

  <div class="score">
    <div class="controls">
      <div class="ctl"><span class="label">Share of 4,000 test headlines with</span><Segmented label="Measure" value={m} onchange={(v) => (m = v)} options={MEAS.map((q) => ({ value: q.value, label: q.label, hint: q.hint }))} /></div>
    </div>
    <div class="surface bars">
      {#each ROWS as r}
        <div class="bar">
          <span class="bn">{r.name}</span>
          <span class="track">
            <span class="fill" style:width="{r.v[m][0]}%" style:background={r.tint}></span>
            {#each r.v[m].slice(1) as sv, j}
              <span class="seed" style:left="{sv}%" data-tip="Seed {j}: {sv.toFixed(1)}%"></span>
            {/each}
          </span>
          <span class="bv num">{r.v[m][0].toFixed(1)}%</span>
        </div>
      {/each}
      <p class="note">Bars pool both seeds; ticks mark each seed. The entity check and NLI also flag faithful headlines, so their rate on real titles is not zero.</p>
    </div>
  </div>
</section>

<style>
  .pick { display: inline-flex; gap: 4px; padding: var(--s1); background: var(--wash); border-radius: var(--r-pill); flex-wrap: wrap; }
  .pick button { width: 32px; height: 32px; border: 0; border-radius: 999px; background: transparent; color: var(--ink2); font: var(--type-control); cursor: pointer; transition: background-color var(--fast) var(--ease), color var(--fast) var(--ease), scale var(--fast) var(--ease); }
  .pick button[aria-pressed="true"] { background: var(--ink); color: var(--paper); }
  .pick button:active { scale: .94; }
  @media (hover: hover) { .pick button[aria-pressed="false"]:hover { color: var(--ink); background: var(--wash2); } }
  .rows { display: grid; gap: 2px; }
  .row { display: grid; grid-template-columns: 170px minmax(0, 1fr); gap: var(--s4); align-items: baseline; padding: var(--s2) var(--s3); border-radius: var(--r-col); }
  .row:nth-child(odd) { background: var(--paper); }
  .who { display: inline-flex; align-items: center; gap: 8px; color: var(--ink3); font: var(--type-small); white-space: nowrap; }
  .who i { width: 8px; height: 8px; border-radius: 999px; }
  .h { font: var(--type-body); color: var(--ink); }
  .score { display: grid; gap: var(--s4); }
  .bars { display: grid; gap: var(--s3); }
  .bar { display: grid; grid-template-columns: 150px minmax(0, 1fr) 64px; gap: var(--s3); align-items: center; }
  .bn { color: var(--ink2); font: var(--type-small); }
  .track { position: relative; height: 18px; background: var(--paper); border-radius: var(--r-chip); }
  .fill { position: absolute; inset: 0 auto 0 0; border-radius: var(--r-chip); transition: width var(--glide) var(--ease); }
  .seed { position: absolute; top: -3px; width: 2px; height: 24px; margin-left: -1px; background: var(--ink); border-radius: 1px; transition: left var(--glide) var(--ease); }
  .bv { text-align: right; font: 600 16px/1 var(--display); }
  @media (max-width: 700px) { .row { grid-template-columns: 1fr; gap: 2px; } .bar { grid-template-columns: 100px minmax(0, 1fr) 54px; } }
</style>
