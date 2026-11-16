<script lang="ts">
  import { ARMS, ARM_COLOR } from "../lib";
  const FACTS: Record<string, string[]> = {
    a: ["DeBERTa-v3, pairwise loss", "47,932 Upworthy A/B pairs", "80.4% on clear held-out pairs"],
    b: ["Qwen2.5-7B as four readers", "“Would you click it?”", "P(Yes), averaged"],
    c: ["Click score − 2 × (1 − entailment)", "NLI against the article", "weight fixed in advance"],
  };
</script>

<section class="part" id="setup">
  <p class="kicker">The study</p>
  <h2>One writer, three rewards, the same articles</h2>
  <p class="lede">The writer always sees the article. We change only what earns it a reward: a predicted click, a simulated reader’s approval, or a predicted click with a penalty for unsupported claims.</p>
  <div class="flow">
    <div class="surface box writer">
      <p class="t">Headline writer</p>
      <p class="d">Qwen2.5-3B-Instruct + LoRA</p>
      <p class="d">Up to 350 training steps, 2 seeds per reward</p>
      <p class="d">3,000 cc_news training articles</p>
    </div>
    <div class="arms">
      {#each ARMS as a}
        <div class="surface box arm" style:--tint={ARM_COLOR[a.key]} data-tip={a.hint}>
          <p class="t"><i></i>({a.key}) {a.name}</p>
          {#each FACTS[a.key] as f}<p class="d">{f}</p>{/each}
        </div>
      {/each}
    </div>
    <div class="surface box test">
      <p class="t">Tested on 1,000 held-out articles</p>
      <p class="d">4 headlines each, checked against the article</p>
      <p class="d">invented numbers · reactions · entities · NLI</p>
      <p class="d">100 + 60 reference labels</p>
    </div>
  </div>
  <p class="note">A seed is an independent training run with different randomness. NLI (natural language inference) estimates whether an article supports a headline. RL means learning by reward.</p>
  <p class="note">Hypotheses and the result that would count as failure were written before any data existed. Only the penalty in (c) sees the article; the other rewards score the headline alone.</p>
</section>

<style>
  .flow { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.4fr) minmax(0, 1fr); gap: var(--s4); align-items: stretch; }
  .arms { display: grid; gap: var(--s3); }
  .box { display: grid; gap: 4px; align-content: center; }
  .t { font: var(--type-label); display: flex; align-items: center; gap: 8px; }
  .t i { width: 10px; height: 10px; border-radius: 99px; background: var(--tint); }
  .d { color: var(--ink2); font: var(--type-small); }
  .arm { padding: var(--s4) var(--s5); }
  @media (max-width: 860px) { .flow { grid-template-columns: 1fr; } }
</style>
