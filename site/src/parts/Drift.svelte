<script lang="ts">
  import { Segmented, Swap } from "@stdlogits/ledger";
  import { D, ARMS, ARM_COLOR, traj, type Arm } from "../lib";
  import Mark from "./Mark.svelte";

  const MEASURES = [
    { key: "invented_reactions_pct", label: "Invented reactions", hint: "A reaction word (viral, controversy, backlash…) that the article does not use" },
    { key: "invented_numbers_pct", label: "Invented numbers", hint: "A number whose value does not appear in the article" },
    { key: "off_article_pct", label: "Off-article", hint: "Shares no content word with the article" },
    { key: "entity_check_pct", label: "Entity check", hint: "A name or number not found in the article (the pre-registered check; confused by title case)" },
  ] as const;
  type MK = (typeof MEASURES)[number]["key"];

  const CHECKS: { run: string; step: number; arm: Arm }[] = [
    { run: "main_a_s0", step: 100, arm: "a" },
    { run: "main_a_s0", step: 150, arm: "a" },
    { run: "main_b_s0", step: 250, arm: "b" },
    { run: "main_b_s0", step: 350, arm: "b" },
    { run: "main_c_s0", step: 350, arm: "c" },
  ];
  let pick = $state(0);
  const cur = $derived(CHECKS[pick]);
  const items = $derived((D.ckpt_labels as any[]).filter((r) => r.run === cur.run && r.step === cur.step));
  const counts = $derived({
    react: items.filter((r) => r.react && !r.other).length,
    other: items.filter((r) => r.other).length,
    none: items.filter((r) => !r.react && !r.other).length,
    viral: items.filter((r) => r.viral).length,
  });

  const W = 330, H = 190, L = 34, R = 10, T = 12, B = 26;
  const x = (s: number) => L + (s / 350) * (W - L - R);
  const y = (v: number) => T + (1 - v / 100) * (H - T - B);
  const series = (r: string, m: MK) => traj(r).map((t, i) => `${i ? "L" : "M"}${x(t.step).toFixed(1)},${y((t as any)[m]).toFixed(1)}`).join("");

</script>

<section class="part" id="drift">
  <p class="kicker">Checkpoints</p>
  <h2>Invented claims first, the template later</h2>
  <p class="lede">Every 50 steps we wrote headlines for 300 held-out articles. Under the click predictor, invented reactions rose by step 100 and invented numbers jumped at step 200. The simulated audience picked up reactions late and never the template.</p>

  <div class="surface figure">
    <div class="panels">
      {#each MEASURES as m}
        <div class="panel">
          <p class="ptitle" data-tip={m.hint}>{m.label}</p>
          <p class="measure-definition">{m.hint}</p>
          <svg viewBox="0 0 {W} {H}" role="img" aria-label="{m.label} by training step">
            {#each [0, 50, 100] as g}
              <line class="gridline" x1={L} x2={W - R} y1={y(g)} y2={y(g)} />
              <text class="axis" x={L - 6} y={y(g) + 4} text-anchor="end">{g}%</text>
            {/each}
            {#each [0, 100, 200, 300] as s}<text class="axis" x={x(s)} y={H - 6} text-anchor="middle">{s}</text>{/each}
            {#each ARMS as a}
              {#each [1, 0] as sd}
                {@const r = `main_${a.key}_s${sd}`}
                <path d={series(r, m.key)} fill="none" stroke={ARM_COLOR[a.key]} stroke-width={sd === 0 ? 2.6 : 1.2} opacity={sd === 0 ? 1 : 0.7} stroke-linejoin="round" />
                {#each traj(r) as t}
                  <circle cx={x(t.step)} cy={y((t as any)[m.key])} r={sd === 0 ? 3.5 : 2.5} fill={ARM_COLOR[a.key]}
                    data-tip-title="{a.name}, seed {sd}, step {t.step}"
                    data-tip="{m.label}: {((t as any)[m.key]).toFixed(1)}% of 600 test headlines · KL {t.kl_seq.toFixed(1)} nats" />
                {/each}
              {/each}
            {/each}
            {#if m.key === "invented_numbers_pct" || m.key === "invented_reactions_pct"}
              {#each CHECKS as c, i}
                {@const t = traj(c.run).find((q) => q.step === c.step)}
                {#if t}
                  <circle class="ring" class:on={i === pick} cx={x(c.step)} cy={y((t as any)[m.key])} r="8" fill="none" stroke="var(--ink)" stroke-width={i === pick ? 2 : 0} />
                {/if}
              {/each}
            {/if}
          </svg>
        </div>
      {/each}
    </div>
    <p class="legendline">
      {#each ARMS as a}<span class="lg"><i style:background={ARM_COLOR[a.key]}></i>{a.name}</span>{/each}
      <span class="lg muted">thick: seed 0 · thin: seed 1 · step 0 is the untrained writer</span>
    </p>
  </div>

  <p class="note">These are automated flags, not verified fabrication rates. Select a labelled checkpoint below to see what the article actually supports; its location is ringed on the first two charts.</p>
  <div class="labels">
    <div class="controls">
      <span class="label">Read 12 labelled headlines from</span>
      <Segmented label="Checkpoint" value={String(pick)} onchange={(v) => (pick = Number(v))}
        options={CHECKS.map((c, i) => ({ value: String(i), label: `${ARMS.find((a) => a.key === c.arm)!.short} · ${c.step}`, hint: `${ARMS.find((a) => a.key === c.arm)!.name}, seed 0, step ${c.step}` }))} />
    </div>
    <Swap key={pick}>
      <div class="surface lab" style:--tint={ARM_COLOR[cur.arm]}>
        <div class="tally">
          <span><b class="num">{counts.other}</b> invent a role, event or detail</span>
          <span><b class="num">{counts.react}</b> invent only a reaction</span>
          <span><b class="num">{counts.none}</b> invent nothing</span>
          {#if counts.viral}<span><b class="num">{counts.viral}</b> in the viral-story form</span>{/if}
        </div>
        <ul class="items">
          {#each items as r, i}
            <li class="rise" style:--i={i}>
              <span class="chip" class:other={r.other} class:react={r.react && !r.other}>{r.other ? "Invented fact" : r.react ? "Invented reaction" : "Nothing invented"}</span>
              <span class="h"><Mark h={r.h} nums={[]} tint={ARM_COLOR[cur.arm]} /></span>
              <span class="why">{r.why}</span>
            </li>
          {/each}
        </ul>
      </div>
    </Swap>
    <p class="note">Labels were drafted by an LLM that saw only the headline and its article, then reviewed by the author, who changed one of 60. The reason column summarises what the article says.</p>
  </div>
</section>

<style>
  .panels { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: var(--s4); }
  .panel { min-width: 0; }
  .ptitle { font: var(--type-control); color: var(--ink2); margin-bottom: var(--s1); }
  svg { width: 100%; display: block; }
  circle { transition: r var(--fast) var(--ease); }
  .ring { pointer-events: none; transition: stroke-width var(--glide) var(--ease); }
  .legendline { display: flex; flex-wrap: wrap; gap: var(--s2) var(--s4); margin-top: var(--s3); color: var(--ink2); font: var(--type-meta); }
  .lg { display: inline-flex; align-items: center; gap: 6px; }
  .lg i { width: 14px; height: 4px; border-radius: 2px; display: inline-block; }
  .muted { color: var(--ink3); }
  .labels { display: grid; gap: var(--s4); }
  .lab { display: grid; gap: var(--s4); }
  .tally { display: flex; flex-wrap: wrap; gap: var(--s2) var(--s5); color: var(--ink2); font: var(--type-small); }
  .tally b { color: var(--ink); font: 600 20px/1 var(--display); margin-right: 4px; }
  .items { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--s2); list-style: none; padding: 0; }
  .items li { display: grid; gap: 6px; background: var(--paper); border-radius: var(--r-col); padding: var(--s3); }
  .chip { justify-self: start; padding: 1px 8px; border-radius: var(--r-pill); font: var(--type-meta); background: var(--wash2); color: var(--ink2); }
  .chip.other { background: var(--tint); color: var(--paper); }
  .chip.react { background: color-mix(in oklch, var(--tint) 30%, var(--wash)); color: var(--ink); }
  .h { font: var(--type-small); color: var(--ink); }
  .why { font: var(--type-meta); color: var(--ink3); }
  @media (max-width: 980px) { .panels { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  @media (max-width: 700px) { .items { grid-template-columns: 1fr; } }
</style>
