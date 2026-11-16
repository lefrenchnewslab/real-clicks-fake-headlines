<script lang="ts">
  import { IconButton, MorphText, Segmented } from "@stdlogits/ledger";
  import { onDestroy } from "svelte";
  import { ARMS, ARM_COLOR, train, type Arm, type Seed } from "../lib";
  import Mark from "./Mark.svelte";

  let seed = $state<Seed>(0);
  let step = $state(100);
  let playing = $state(false);
  let timer: ReturnType<typeof setInterval> | undefined;

  const MAX = 350;
  const rows = $derived(Object.fromEntries(ARMS.map((a) => [a.key, train(a.key, seed)])) as Record<Arm, ReturnType<typeof train>>);
  const at = (a: Arm) => rows[a].find((r) => r[0] === step) ?? null;
  const last = (a: Arm) => rows[a][rows[a].length - 1][0];

  function pause() { clearInterval(timer); playing = false; }
  function selectStep(value: number) { pause(); step = value; }
  function toggle() {
    if (playing) { clearInterval(timer); playing = false; return; }
    if (step >= MAX) step = 1;
    playing = true;
    timer = setInterval(() => { step = Math.min(MAX, step + 1); if (step >= MAX) { clearInterval(timer); playing = false; } }, 900);
  }
  onDestroy(() => clearInterval(timer));

  // Chart geometry.
  const W = 1040, H = 230, L = 44, R = 16, T = 14, B = 26, H2 = 70;
  const x = (s: number) => L + ((s - 1) / (MAX - 1)) * (W - L - R);
  const ymin = -1, ymax = 6;
  const y = (v: number) => T + (1 - (Math.max(ymin, Math.min(ymax, v)) - ymin) / (ymax - ymin)) * (H - T - B);
  const y2 = (d: number) => H + 10 + (1 - (d - 1) / 7) * (H2 - 20);
  const smooth = (r: ReturnType<typeof train>, i: number, k = 4) => {
    const lo = Math.max(0, i - k), hi = Math.min(r.length - 1, i + k);
    let s = 0; for (let j = lo; j <= hi; j++) s += r[j][1]; return s / (hi - lo + 1);
  };
  const path = (a: Arm) => rows[a].map((r, i) => `${i ? "L" : "M"}${x(r[0]).toFixed(1)},${y(smooth(rows[a], i)).toFixed(1)}`).join("");
  const path2 = (a: Arm) => rows[a].map((r, i) => `${i ? "L" : "M"}${x(r[0]).toFixed(1)},${y2(r[2]).toFixed(1)}`).join("");

  let svg: SVGSVGElement;
  let dragging = false;
  function seek(e: PointerEvent) {
    pause();
    const b = svg.getBoundingClientRect();
    const sx = ((e.clientX - b.left) / b.width) * W;
    step = Math.max(1, Math.min(MAX, Math.round(1 + ((sx - L) / (W - L - R)) * (MAX - 1))));
  }
  function key(e: KeyboardEvent) {
    pause();
    if (e.key === "Home") { step = 1; e.preventDefault(); }
    if (e.key === "End") { step = MAX; e.preventDefault(); }
    if (e.key === "ArrowRight") { step = Math.min(MAX, step + (e.shiftKey ? 10 : 1)); e.preventDefault(); }
    if (e.key === "ArrowLeft") { step = Math.max(1, step - (e.shiftKey ? 10 : 1)); e.preventDefault(); }
  }
</script>

<section class="part" id="train">
  <p class="kicker">Training, step by step</p>
  <h2>Watch the writers learn, step by step</h2>
  <p class="lede">Each step uses a new batch of articles. Compare the three rewards at the same step, then move forward to see when the click-trained writer starts repeating itself.</p>

  <div class="controls">
    <IconButton icon={playing ? "pause" : "play"} label={playing ? "Pause" : "Play training"} onclick={toggle} />
    <div class="ctl"><span class="label">Seed</span><Segmented label="Seed" value={String(seed)} onchange={(v) => { pause(); seed = Number(v) as Seed; }} options={[{ value: "0", label: "Seed 0" }, { value: "1", label: "Seed 1" }]} /></div>
    <div class="readout"><span class="label">Step</span><span class="stepnum num"><MorphText text={String(step)} /></span></div>
  </div>

  <div class="scrubber">
    <label for="training-step">Training step <strong class="num">{step} / {MAX}</strong></label>
    <input id="training-step" type="range" min="1" max={MAX} value={step} oninput={(e) => selectStep(Number(e.currentTarget.value))} />
    <div class="milestones" role="group" aria-label="Training milestones">
      {#each [[1, "Start"], [100, "Early inventions"], [200, "Template emerges"], [350, "Final step"]] as [n, label]}
        <button class="quiet-button" aria-pressed={step === n} onclick={() => selectStep(Number(n))}>{n} · {label}</button>
      {/each}
    </div>
  </div>
  <div class="surface figure">
    <svg bind:this={svg} viewBox="0 0 {W} {H + H2 + 16}" role="slider" tabindex="0" aria-label="Training step" aria-valuemin="1" aria-valuemax={MAX} aria-valuenow={step}
      onpointerdown={(e) => { dragging = true; svg.setPointerCapture(e.pointerId); seek(e); }}
      onpointermove={(e) => dragging && seek(e)}
      onpointerup={() => (dragging = false)} onpointercancel={() => (dragging = false)}
      onkeydown={key}>
      {#each [0, 2, 4, 6] as g}
        <line class="gridline" x1={L} x2={W - R} y1={y(g)} y2={y(g)} />
        <text class="axis" x={L - 8} y={y(g) + 4} text-anchor="end">{g}</text>
      {/each}
      <text class="axis" x={L} y={T - 2}>Reward, standardised on the untrained writer (z)</text>
      {#each ARMS as a}
        <path d={path(a.key)} fill="none" stroke={ARM_COLOR[a.key]} stroke-width="2.5" stroke-linejoin="round" opacity={step >= 1 ? 1 : 0.4} />
      {/each}
      <!-- diversity strip -->
      <text class="axis" x={L} y={H + 4}>Different headlines among 8 drafts for the article</text>
      {#each [1, 8] as g}<text class="axis" x={L - 8} y={y2(g) + 4} text-anchor="end">{g}</text>{/each}
      {#each ARMS as a}
        <path d={path2(a.key)} fill="none" stroke={ARM_COLOR[a.key]} stroke-width="1.5" opacity=".85" />
      {/each}
      {#each [50, 100, 150, 200, 250, 300, 350] as s}<text class="axis" x={x(s)} y={H + H2 + 12} text-anchor="middle">{s}</text>{/each}
      <!-- cursor -->
      <rect x={x(step) - 1} y={T} width="2" height={H + H2 - T - 12} fill="var(--ink)" rx="1" />
      {#each ARMS as a}
        {@const r = at(a.key)}
        {#if r}<circle cx={x(step)} cy={y(r[1])} r="5" fill={ARM_COLOR[a.key]} stroke="var(--wash)" stroke-width="2" />{/if}
      {/each}
      {#if seed === 0}
        <text class="axis" x={x(240) + 6} y={y(5.6)} fill="var(--ink2)">collapse stop, step 240</text>
      {/if}
    </svg>
  </div>

  <div class="cols">
    {#each ARMS as a}
      {@const r = at(a.key)}
      <div class="col surface" style:--tint={ARM_COLOR[a.key]}>
        <div class="head">
          <span class="dot"></span><span class="name" data-tip={a.hint}>{a.name}</span>
          {#if r}<span class="rv num">{r[1] >= 0 ? "+" : "−"}{Math.abs(r[1]).toFixed(2)}</span>{/if}
        </div>
        {#if r}
          <ol class="drafts">
            {#each r[4] as h, i (seed + ":" + step + ":" + i)}
              <li><Mark {h} nums={[]} tint={ARM_COLOR[a.key]} /></li>
            {/each}
          </ol>
          <div class="meta"><span class="num">{r[2].toFixed(1)}</span> of 8 drafts differ</div>
        {:else}
          <p class="stopped">Stopped at step {last(a.key)}: fewer than 2 of 8 drafts differed for 30 steps in a row.</p>
        {/if}
      </div>
    {/each}
  </div>
  <p class="note">Each curve shows its own training reward; equal heights do not mean equal predicted clicks. Lines use a nine-step moving average; dots and cards show the selected step’s raw value.</p>
  <p class="note">Three of the eight training drafts per step, sampled at temperature 1. Reaction words from the paper's list are marked. The articles themselves are not shown.</p>
</section>

<style>
  svg { min-width: 640px; touch-action: none; cursor: ew-resize; display: block; }
  svg:focus-visible { outline: 2px solid var(--ink); outline-offset: 4px; border-radius: 8px; }
  .readout { display: flex; align-items: baseline; gap: var(--s2); margin-left: auto; }
  .stepnum { font: 600 28px/1 var(--display); letter-spacing: -.02em; min-width: 3ch; }
  .cols { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: var(--s4); }
  .col { display: grid; gap: var(--s3); align-content: start; }
  .head { display: flex; align-items: center; gap: var(--s2); }
  .dot { width: 10px; height: 10px; border-radius: 999px; background: var(--tint); }
  .name { font: var(--type-label); }
  .rv { margin-left: auto; font: 600 18px/1 var(--display); }
  .drafts { display: grid; gap: var(--s2); padding: 0; list-style: none; min-height: 168px; }
  .drafts li { background: var(--paper); border-radius: var(--r-col); padding: var(--s2) var(--s3); font: var(--type-small); color: var(--ink); }
  .meta, .stopped { color: var(--ink3); font: var(--type-meta); }
  .stopped { font: var(--type-small); min-height: 168px; }
  @media (max-width: 860px) { .cols { grid-template-columns: 1fr; } .drafts { min-height: 0; } }
</style>
