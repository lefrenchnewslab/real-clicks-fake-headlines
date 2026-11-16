<script lang="ts">
  import { MorphText, Segmented } from "@stdlogits/ledger";
  import { D, ARM_COLOR } from "../lib";

  // 27 checkpoint headlines that invent a role, event or detail (Table 1): how many each check flags.
  const CATCH = [
    { value: "nli", label: "NLI", n: 17, hint: "DeBERTa-v3-large NLI, unsupported below τ = 0.55" },
    { value: "ent", label: "Entity check", n: 8, hint: "A name or number not found in the article" },
    { value: "num", label: "Invented numbers", n: 5, hint: "A number whose value is not in the article" },
  ] as const;
  let c = $state<(typeof CATCH)[number]["value"]>("nli");
  const cur = $derived(CATCH.find((q) => q.value === c)!);

  // Precision / recall against all 100 reference labels, weighted by inverse sampling probability (Table 9).
  const SRC = [
    { k: "base", name: "Untrained", tint: ARM_COLOR.base },
    { k: "a", name: "Clicks", tint: ARM_COLOR.a },
    { k: "b", name: "Audience", tint: ARM_COLOR.b },
    { k: "c", name: "Penalty", tint: ARM_COLOR.c },
  ];
  const PR: { name: string; v: ([number | null, number] )[]; mix: [number, number] }[] = [
    { name: "Entity check", v: [[0.31, 0.2], [1, 0.95], [0.83, 0.52], [0.5, 0.55]], mix: [0.86, 0.72] },
    { name: "NLI", v: [[0.47, 0.9], [1, 1], [0.94, 0.76], [0.07, 0.09]], mix: [0.81, 0.85] },
    { name: "Invented numbers", v: [[0.75, 0.15], [1, 0.95], [null, 0], [0, 0]], mix: [0.98, 0.51] },
    { name: "Invented reactions", v: [[1, 0.4], [1, 1], [1, 0.31], [null, 0]], mix: [1, 0.65] },
    { name: "LLM probe", v: [[null, 0], [1, 0.07], [null, 0], [null, 0]], mix: [1, 0.04] },
  ];
  const cell = (v: number | null) => (v == null ? "var(--paper)" : `color-mix(in oklch, var(--deep) ${Math.round(v * 70)}%, var(--wash))`);
  const txt = (v: number | null) => (v == null ? "–" : v.toFixed(2).replace(/^0/, ""));

  // The 100 reference labels, 25 per source; half of each source sampled among entity-check flags.
  const LABEL_TINT: Record<string, string> = {
    fabricated_fact: "var(--ink)", misleading_exaggeration: "var(--ink2)", teaser_no_claim: "var(--ink3)", other: "var(--ink3)", faithful: "var(--wash2)",
  };
  const LABEL_NAME: Record<string, string> = { fabricated_fact: "Fabricated", misleading_exaggeration: "Exaggerated", teaser_no_claim: "Teaser", other: "Other", faithful: "Faithful" };
  let selectedLabel = $state<any>(null);
  const bySrc = (k: string) => (D.labels as any[]).filter((l) => l.src === (k === "base" ? "base" : `main_${k}_s0`)).sort((x, y) => (x.label === y.label ? 0 : x.label === "fabricated_fact" ? -1 : y.label === "fabricated_fact" ? 1 : x.label < y.label ? -1 : 1));
</script>

<section class="part" id="checks">
  <p class="kicker">Measuring fabrication</p>
  <h2>Most checks miss the invented events</h2>
  <p class="lede">The template is easy to catch. The quieter inventions are not: entity and number checks flag few of them, an LLM probe almost none, and NLI loses them on the runs trained against a similar NLI model.</p>

  <div class="two">
    <div class="surface catch">
      <div class="controls"><Segmented label="Check" value={c} onchange={(v) => (c = v)} options={CATCH.map((q) => ({ value: q.value, label: q.label, hint: q.hint }))} /></div>
      <div class="waffle" role="img" aria-label="{cur.n} of 27 caught">
        {#each Array(27) as _, i}<i class:on={i < cur.n} style:transition-delay="{i * 14}ms"></i>{/each}
      </div>
      <p class="big"><span class="num"><MorphText text={String(cur.n)} /></span><span class="of">of 27 headlines with an invented fact caught</span></p>
      <p class="note">{27 - cur.n} missed. From the 60 labelled checkpoint headlines (Table 1 of the paper).</p>
    </div>

    <div class="surface matrix">
      <p class="note">Each cell reads precision / recall. Precision: how often a flag is right. Recall: how much fabrication the check catches. Both should be high.</p>
      <div class="mt" role="table" aria-label="Detector precision and recall by writer">
        <div class="mr th" role="row"><span role="columnheader">Precision / recall</span>{#each SRC as s}<span role="columnheader"><i style:background={s.tint}></i>{s.name}</span>{/each}<span role="columnheader">Output mix</span></div>
        {#each PR as r}
          <div class="mr" role="row">
            <span role="rowheader" class="rn">{r.name}</span>
            {#each r.v as [p, rc], j}
              <span role="cell" class="pair" class:hot={r.name === "NLI" && j === 3}>
                <b style:background={cell(p)} data-tip="Precision {txt(p)} on 25 {SRC[j].name.toLowerCase()} headlines">{txt(p)}</b>
                <b style:background={cell(rc)} data-tip="Recall {txt(rc)}, weighted by inverse sampling probability">{txt(rc)}</b>
              </span>
            {/each}
            <span role="cell" class="pair"><b style:background={cell(r.mix[0])}>{txt(r.mix[0])}</b><b style:background={cell(r.mix[1])}>{txt(r.mix[1])}</b></span>
          </div>
        {/each}
      </div>
      <p class="note">Against all 100 reference labels, weighted for the sampling design (exploratory). Darker is higher. “–”: the check flagged nothing there. Outlined: NLI on the penalty runs, whose reward used a similar NLI model.</p>
    </div>
  </div>

  <div class="surface strip">
    <p class="ptitle">The 100 reference labels, one square per headline</p>
    <p class="note">Select a square to read the headline and its reference label.</p>
    {#each SRC as s}
      <div class="lrow">
        <span class="ln"><i style:background={s.tint}></i>{s.name}</span>
        <div class="sq">
          {#each bySrc(s.k) as l}
            <button type="button" class="label-square" style:background={LABEL_TINT[l.label] ?? "var(--ink3)"} aria-label={`${s.name}: ${LABEL_NAME[l.label] ?? l.label}. ${l.h}`} aria-pressed={selectedLabel === l} onclick={() => (selectedLabel = l)}></button>
          {/each}
        </div>
      </div>
    {/each}
    {#if selectedLabel}<div class="label-detail" aria-live="polite"><strong>{LABEL_NAME[selectedLabel.label]}</strong><p>“{selectedLabel.h}”</p>{#if selectedLabel.note}<p class="note">{selectedLabel.note}</p>{/if}</div>{/if}
    <p class="legendline">{#each Object.entries(LABEL_NAME) as [k, n]}<span class="lg"><i style:background={LABEL_TINT[k]}></i>{n}</span>{/each}</p>
    <p class="note">Half of each source's 25 headlines were sampled among those the entity check flagged, so the counts are not rates. Labels were drafted by an LLM and reviewed by the author, who accepted all 100.</p>
  </div>
</section>

<style>
  .two { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.6fr); gap: var(--s4); }
  .catch { display: grid; gap: var(--s4); align-content: start; }
  .waffle { display: grid; grid-template-columns: repeat(9, 1fr); gap: 5px; max-width: 360px; }
  .waffle i { aspect-ratio: 1; border-radius: 4px; background: var(--wash2); transition: background-color 500ms var(--ease); }
  .waffle i.on { background: var(--deep); }
  .big { display: flex; align-items: baseline; gap: var(--s3); flex-wrap: wrap; }
  .big .num { font: 600 44px/1 var(--display); letter-spacing: -.03em; }
  .of { color: var(--ink2); font: var(--type-small); max-width: 24ch; }
  .matrix { overflow-x: auto; }
  .mt { display: grid; gap: 4px; min-width: 560px; }
  .mr { display: grid; grid-template-columns: 1.3fr repeat(5, minmax(0, 1fr)); gap: 6px; align-items: center; }
  .th { color: var(--ink3); font: var(--type-meta); }
  .th span { display: inline-flex; align-items: center; gap: 6px; }
  .th i { width: 8px; height: 8px; border-radius: 99px; display: inline-block; }
  .rn { color: var(--ink2); font: var(--type-small); }
  .pair { display: grid; grid-template-columns: 1fr 1fr; gap: 2px; border-radius: 8px; padding: 2px; }
  .pair.hot { background: var(--ink); }
  .pair b { font: 500 12px/26px var(--text); text-align: center; border-radius: 6px; color: var(--ink); font-variant-numeric: tabular-nums; }
  .label-square { width: 100%; aspect-ratio: 1; min-height: 20px; border: 0; border-radius: 3px; cursor: pointer; }
  .label-square[aria-pressed="true"] { outline: 2px solid var(--deep); outline-offset: 2px; }
  .label-detail { display: grid; gap: 8px; background: var(--paper); padding: 16px; border-radius: 10px; font: var(--type-small); }
  .strip { display: grid; gap: var(--s3); }
  .ptitle { font: var(--type-control); color: var(--ink2); }
  .lrow { display: grid; grid-template-columns: 110px minmax(0, 1fr); gap: var(--s3); align-items: center; }
  .ln { display: inline-flex; align-items: center; gap: 6px; color: var(--ink2); font: var(--type-small); }
  .ln i { width: 8px; height: 8px; border-radius: 99px; }
  .sq { display: grid; grid-template-columns: repeat(25, minmax(0, 1fr)); gap: 4px; max-width: 760px; }
  .sq button { aspect-ratio: 1; border-radius: 4px; transition: scale var(--fast) var(--ease); }
  @media (hover: hover) { .sq button:hover { scale: 1.25; } }
  .legendline { display: flex; flex-wrap: wrap; gap: var(--s2) var(--s4); color: var(--ink2); font: var(--type-meta); }
  .lg { display: inline-flex; align-items: center; gap: 6px; }
  .lg i { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
  @media (max-width: 600px) { .lrow { grid-template-columns: 1fr; } .sq { grid-template-columns: repeat(10, minmax(0, 1fr)); } .label-square { min-height: 28px; } }
  @media (max-width: 900px) { .two { grid-template-columns: 1fr; } }
</style>
