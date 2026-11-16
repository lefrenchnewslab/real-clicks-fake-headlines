<script lang="ts">
  import { Segmented } from "@stdlogits/ledger";
  import { ARMS, ARM_COLOR, traj, type Traj } from "../lib";

  const YS = [
    { value: "click_model_z", label: "Click score", hint: "The click predictor's score, standardised on the untrained writer", z: true, lo: -1.5, hi: 6 },
    { value: "audience_z", label: "Audience score", hint: "The simulated audience's score, standardised on the untrained writer", z: true, lo: -1.5, hi: 2 },
    { value: "invented_numbers_pct", label: "Invented numbers", hint: "Share of headlines with a number not in the article", z: false, lo: 0, hi: 100 },
    { value: "invented_reactions_pct", label: "Invented reactions", hint: "Share of headlines with a reaction word not in the article", z: false, lo: 0, hi: 100 },
  ] as const;
  type YK = (typeof YS)[number]["value"];
  let yk = $state<YK>("click_model_z");
  const Y = $derived(YS.find((q) => q.value === yk)!);

  const W = 1040, H = 380, L = 56, R = 24, T = 20, B = 44;
  const KMAX = 160;
  const x = (k: number) => L + (Math.sqrt(Math.max(0, k)) / Math.sqrt(KMAX)) * (W - L - R);
  const y = (v: number) => T + (1 - (v - Y.lo) / (Y.hi - Y.lo)) * (H - T - B);
  const val = (t: Traj) => (t as any)[yk] as number;
  const line = (r: string) => traj(r).map((t, i) => `${i ? "L" : "M"}${x(t.kl_seq).toFixed(1)},${y(val(t)).toFixed(1)}`).join("");
  const ticks = $derived(Y.z ? (Y.hi > 3 ? [-1, 0, 2, 4, 6] : [-1, 0, 1, 2]) : [0, 25, 50, 75, 100]);
  // Best-of-16 from the untrained writer: KL is bounded by log n − (n−1)/n (Gao et al., 2023).
  const BON_KL = Math.log(16) - 15 / 16;
</script>

<section class="part" id="distance">
  <p class="kicker">How far each reward pushed</p>
  <h2>The click predictor pulled the writer four to six times further</h2>
  <p class="lede">KL measures how much the writer’s output distribution changed from its starting point: further right means a larger change. The click-trained writer moved much further. Comparing writers at the same distance changes the story.</p>
  <div class="controls">
    <div class="ctl"><span class="label">Vertical axis</span><Segmented label="Vertical axis" value={yk} onchange={(v) => (yk = v)} options={YS.map((q) => ({ value: q.value, label: q.label, hint: q.hint }))} /></div>
  </div>
  <div class="surface figure">
    <svg viewBox="0 0 {W} {H}" role="img" aria-label="{Y.label} against KL to the untrained writer">
      {#each ticks as g}
        <line class="gridline" x1={L} x2={W - R} y1={y(g)} y2={y(g)} />
        <text class="axis" x={L - 10} y={y(g) + 4} text-anchor="end">{Y.z ? (g > 0 ? "+" : "") + g : g + "%"}</text>
      {/each}
      {#each [0, 5, 25, 50, 100, 150] as k}<text class="axis" x={x(k)} y={H - 18} text-anchor="middle">{k}</text>{/each}
      <text class="axis" x={W - R} y={H - 2} text-anchor="end">KL to the untrained writer (nats per headline, square-root scale)</text>
      <text class="axis" x={L} y={12}>{Y.label}{Y.z ? " (z)" : ""}</text>

      {#each [{ k: 26.7, s: 0 }, { k: 40.1, s: 1 }] as m}
        <line x1={x(m.k)} x2={x(m.k)} y1={T} y2={H - B} stroke="var(--ink3)" stroke-dasharray="3 4" />
        <text class="axis" x={x(m.k) + 6} y={T + 12 + m.s * 14} data-tip="H3 compares the click predictor with the simulated audience at the audience run's final KL, interpolating between saved checkpoints. Seed 0 was pre-registered; seed 1 is post hoc.">equal KL, seed {m.s}</text>
      {/each}

      {#each ARMS as a}
        {#each [1, 0] as sd}
          {@const r = `main_${a.key}_s${sd}`}
          <path d={line(r)} fill="none" stroke={ARM_COLOR[a.key]} stroke-width={sd ? 1.4 : 2.8} opacity={sd ? 0.7 : 1} stroke-linejoin="round" style="transition: d var(--glide) var(--ease)" />
          {#each traj(r) as t}
            <circle cx={x(t.kl_seq)} cy={y(val(t))} r={sd ? 3 : 4.5} fill={ARM_COLOR[a.key]} style="transition: cy var(--glide) var(--ease)"
              data-tip-title="{a.name}, seed {sd}, step {t.step}"
              data-tip="KL {t.kl_seq.toFixed(1)} · {Y.label} {Y.z ? (val(t) >= 0 ? '+' : '') + val(t).toFixed(2) : val(t).toFixed(1) + '%'}" />
          {/each}
          {@const lastT = traj(r)[traj(r).length - 1]}
          {#if sd === 0}<text x={x(lastT.kl_seq) + 8} y={y(val(lastT)) + 4} fill="var(--ink2)" font-size="13">{a.short}</text>{/if}
        {/each}
      {/each}
      {#if yk === "click_model_z" || yk === "invented_numbers_pct"}
        {@const v = yk === "click_model_z" ? 0.556 : 2.0}
        <rect x={x(BON_KL) - 6} y={y(v) - 6} width="12" height="12" rx="3" fill="var(--ink)"
          data-tip-title="Best of 16, no training"
          data-tip="The click predictor picks the best of 16 untrained headlines (300 test articles): click score +0.56, invented numbers 2.0%. Its KL is at most log 16 − 15/16 ≈ 1.8 nats." />
        <text x={x(BON_KL) + 10} y={y(v) - 8} fill="var(--ink2)" font-size="13">best of 16</text>
      {/if}
    </svg>
    <p class="legendline">
      {#each ARMS as a}<span class="lg"><i style:background={ARM_COLOR[a.key]}></i>{a.name}</span>{/each}
      <span class="lg muted">one dot per saved checkpoint (every 50 steps) · thick seed 0, thin seed 1 · square: best of 16 without training</span>
    </p>
  </div>
  <p class="takeaway">At equal distance, the click-trained writer had fewer invented reactions but slightly more invented numbers than the simulated-audience writer. The result depends on what you measure; it does not establish that real clicks are safer.</p>
  <p class="note">The best-of-16 square is selection without training. Its horizontal position is a theoretical KL upper bound (about 1.8 nats), not a measured checkpoint. Scores are predictions, not observed click rates.</p>
</section>

<style>
  svg { min-width: 640px; display: block; }
  .legendline { display: flex; flex-wrap: wrap; gap: var(--s2) var(--s4); margin-top: var(--s3); color: var(--ink2); font: var(--type-meta); }
  .lg { display: inline-flex; align-items: center; gap: 6px; }
  .lg i { width: 14px; height: 4px; border-radius: 2px; display: inline-block; }
  .muted { color: var(--ink3); }
</style>
