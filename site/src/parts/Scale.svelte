<script lang="ts">
  import { Segmented } from "@stdlogits/ledger";
  import { D, ARM_COLOR, traj } from "../lib";
  import Mark from "./Mark.svelte";

  const RUNS = [
    { k: "main_a_s0", name: "Qwen2.5-3B (main run, seed 0)", w: 2.8, o: 1 },
    { k: "scale_E1", name: "Llama-3.1-8B", w: 2, o: 0.75 },
    { k: "scale_E2", name: "Qwen2.5-14B", w: 2, o: 0.45 },
  ];
  const curve = (k: string) =>
    k.startsWith("main")
      ? traj(k).map((t) => ({ step: t.step, num: t.invented_numbers_pct / 100, react: t.invented_reactions_pct / 100, off: t.off_article_pct / 100, ex: [] as string[] }))
      : (D.scale[k] as any[]);
  const M = [
    { value: "num", label: "Invented numbers" },
    { value: "react", label: "Invented reactions" },
    { value: "off", label: "Off-article" },
  ] as const;
  let m = $state<(typeof M)[number]["value"]>("num");
  let sel = $state<{ k: string; step: number } | null>({ k: "scale_E1", step: 100 });
  const W = 1040, H = 280, L = 56, R = 200, T = 16, B = 34;
  const x = (s: number) => L + (s / 350) * (W - L - R);
  const y = (v: number) => T + (1 - v) * (H - T - B);
  const ex = $derived(sel ? (curve(sel.k).find((p: any) => p.step === sel!.step)?.ex ?? []) : []);
</script>

<section class="part" id="scale">
  <p class="kicker">Larger writers</p>
  <h2>Larger writers fail in different ways</h2>
  <p class="lede">One exploratory click-predictor run each. Llama-8B collapsed faster than the 3B writer. Qwen-14B never collapsed within 250 steps, but most of its headlines gained invented reactions.</p>
  <div class="controls"><Segmented label="Measure" value={m} onchange={(v) => (m = v)} options={M.map((q) => ({ ...q }))} /></div>
  <div class="controls">
    <label class="ctl">Inspect a saved checkpoint
      <select aria-label="Larger writer checkpoint" value={`${sel?.k}:${sel?.step}`} onchange={(e) => { const [k, step] = e.currentTarget.value.split(":"); sel = { k, step: Number(step) }; }}>
        {#each RUNS as r}{#each curve(r.k) as p}<option value={`${r.k}:${p.step}`}>{r.name} · step {p.step}</option>{/each}{/each}
      </select>
    </label>
  </div>
  <div class="surface figure">
    <svg viewBox="0 0 {W} {H}" role="group" aria-label="Larger writers under the click predictor">
      {#each [0, 0.5, 1] as g}<line class="gridline" x1={L} x2={W - R} y1={y(g)} y2={y(g)} /><text class="axis" x={L - 8} y={y(g) + 4} text-anchor="end">{g * 100}%</text>{/each}
      {#each [0, 100, 200, 300] as s}<text class="axis" x={x(s)} y={H - 10} text-anchor="middle">{s}</text>{/each}
      <text class="axis" x={W - R} y={H - 10} text-anchor="end">RL step</text>
      {#each RUNS as r}
        {@const c = curve(r.k)}
        <path d={c.map((p: any, i: number) => `${i ? "L" : "M"}${x(p.step)},${y(p[m])}`).join("")} fill="none" stroke={`color-mix(in oklch, ${ARM_COLOR.a} ${r.o * 100}%, var(--wash2))`} stroke-width={r.w} stroke-dasharray={r.k.startsWith("scale") ? "6 5" : ""} style="transition: d var(--glide) var(--ease)" />
        {#each c as p}
          <circle cx={x(p.step)} cy={y(p[m])} r={sel && sel.k === r.k && sel.step === p.step ? 7 : 4.5} fill={`color-mix(in oklch, ${ARM_COLOR.a} ${r.o * 100}%, var(--wash2))`} style="transition: cy var(--glide) var(--ease), r var(--fast) var(--ease); cursor: pointer"
            role="button" tabindex="0" aria-label={`${r.name}, step ${p.step}: ${(p[m] * 100).toFixed(1)} percent`} aria-pressed={sel?.k === r.k && sel?.step === p.step} onclick={() => (sel = { k: r.k, step: p.step })} onkeydown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); sel = { k: r.k, step: p.step }; } }}
            data-tip-title="{r.name}, step {p.step}" data-tip="{(p[m] * 100).toFixed(1)}%{p.ex?.length ? ' · click for example headlines' : ''}" />
        {/each}
        {@const lp = c[c.length - 1]}
        <text x={x(lp.step) + 10} y={y(lp[m]) + 4} fill="var(--ink2)" font-size="13">{r.name}</text>
      {/each}
    </svg>
    {#if sel}
      <div class="ex">
        <span class="label">{RUNS.find((r) => r.k === sel?.k)?.name}, step {sel?.step}</span>
        {#if !ex.length}<p class="note">No example headlines are included for this checkpoint. The plotted measurements are available; use the training explorer for the main 3B run’s drafts.</p>{/if}
        {#each ex as h}<p class="h"><Mark {h} nums={[]} tint={ARM_COLOR.a} /></p>{/each}
      </div>
    {/if}
  </div>
  <p class="note">Click a point to read three of its headlines. Llama's repeated story invents a reaction (“leaves her speechless”) that the reaction word list does not include, so its reaction line stays low.</p>
</section>

<style>
  svg { min-width: 640px; display: block; }
  .ex { display: grid; gap: var(--s2); margin-top: var(--s4); }
  .h { background: var(--paper); border-radius: var(--r-col); padding: var(--s2) var(--s3); font: var(--type-small); }
  .label { color: var(--ink3); font: var(--type-meta); }
</style>
