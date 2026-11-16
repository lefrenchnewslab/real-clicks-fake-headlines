<script lang="ts">
  import { marks } from "../lib";
  // A headline with its numbers and reaction words marked. `nums` limits number marks to those the
  // invented-numbers check found missing from the article; null marks every number.
  let { h, nums = null, tint = "var(--deep)" }: { h: string; nums?: string[] | null; tint?: string } = $props();
  const segs = $derived(marks(h, nums));
</script>

<span class="hl" style:--mk={tint}>{#each segs as s}{#if s.kind === "num"}<mark class="num" data-tip={nums ? "Number not found in the article" : "Number"}>{s.t}</mark>{:else if s.kind === "react"}<mark class="react" data-tip="Reaction word from the paper's list">{s.t}</mark>{:else}{s.t}{/if}{/each}</span>

<style>
  .hl { text-wrap: pretty; }
  mark { color: inherit; border-radius: 4px; padding: 0 2px; margin: 0 -1px; }
  .num { background: color-mix(in oklch, var(--mk) 26%, transparent); font-weight: 600; }
  .react { background: color-mix(in oklch, var(--mk) 13%, transparent); }
</style>
