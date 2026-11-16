<script lang="ts">
  import { Expander } from "@stdlogits/ledger";
  const ROWS: { id: string; name: string; v: string; ok: "met" | "no" | "mixed" }[] = [
    { id: "G1", name: "Click predictor ≥ 60% on clear pairs and 3 points above surface features", v: "met · 80.4%", ok: "met" },
    { id: "G2", name: "Reward gain ≥ 0.3 z after 50 steps without collapse", v: "met on the pre-planned retry", ok: "met" },
    { id: "H1", name: "Fabrication rises under the click predictor", v: "rise · +85.5 points", ok: "met" },
    { id: "H2", name: "Fabrication rises under the simulated audience", v: "inconclusive", ok: "mixed" },
    { id: "H3", name: "Click predictor vs simulated audience at equal KL", v: "real < simulated, depends on the measure", ok: "mixed" },
    { id: "H4", name: "Penalty keeps ≥ 50% of the click gain, fabrication ≤ untrained + 1 point", v: "failed", ok: "no" },
    { id: "M1", name: "Entity check precision ≥ 0.70, recall ≥ 0.60 on held-out labels", v: "failed · precision 0.64", ok: "no" },
  ];
  let open = $state(true);
</script>

<section class="part" id="record">
  <p class="takeaway">The practical lesson: compare each claim with its source article. A click score can improve while the headline becomes less true, and a faithfulness penalty helps without eliminating the problem.</p>
  <p class="kicker">The record</p>
  <h2>Written down before the data existed, including what would fail</h2>
  <p class="lede">Every criterion was fixed in advance with the result that would count against it. Three dated addenda were each written before the outputs they govern.</p>
  <div class="surface figure">
    <div class="tbl" role="table">
      {#each ROWS as r, i}
        <div class="tr rise" role="row" style:--i={i}>
          <span role="cell" class="crit"><b>{r.id}</b> {r.name}</span>
          <span role="cell"><i class="pill {r.ok}">{r.v}</i></span>
        </div>
      {/each}
    </div>
  </div>
  <Expander label="What this does not show" openLabel="Hide" bind:open>
    <ul class="lim">
      <li>The main runs use one 3B writer; the larger writers are single exploratory runs.</li>
      <li>The click predictor reflects one site's readers in 2013–15, and the articles are 2017–19 news. Live readers never saw a generated headline.</li>
      <li>The reference labels were drafted by an LLM and reviewed by the author; they are not independent human annotation, and the samples are small.</li>
      <li>The click predictor's score has no ceiling while the audience's is a probability, so the two rewards could push the writer different distances even at the same starting scale.</li>
      <li>The reaction word list was written after outputs were seen; its rates are descriptive.</li>
    </ul>
  </Expander>
</section>

<style>
  .tbl { display: grid; min-width: 560px; }
  .tr { display: grid; grid-template-columns: minmax(0, 2.2fr) minmax(0, 1fr); gap: var(--s3); align-items: center; padding: var(--s2) var(--s3); border-radius: var(--r-col); }
  .tr:nth-child(even) { background: var(--paper); }
  .crit { font: var(--type-small); color: var(--ink2); }
  .crit b { color: var(--ink); font-weight: 600; margin-right: var(--s1); }
  .pill { display: inline-block; padding: 2px 10px; border-radius: var(--r-pill); font: var(--type-meta); font-style: normal; }
  .pill.met { background: var(--deep); color: var(--paper); font-weight: 600; }
  .pill.mixed { background: color-mix(in oklch, var(--deep) 22%, var(--wash)); color: var(--ink); }
  .pill.no { background: var(--wash2); color: var(--ink2); }
  .figure { overflow-x: auto; }
  .lim { display: grid; gap: var(--s2); padding-left: var(--s5); color: var(--ink2); font: var(--type-body); max-width: 72ch; margin-top: var(--s3); }
</style>
