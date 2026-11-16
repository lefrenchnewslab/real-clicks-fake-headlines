<script lang="ts">
  const BIB = `@misc{sobhy2026realclicks,
  title  = {Real Clicks, Fake Headlines: AI That Chases Clicks Invents the News},
  author = {Sobhy, Amr},
  year   = {2026},
  note   = {AI x Epistemics Research Sprint, Apart Research},
  url    = {https://github.com/lefrenchnewslab/real-clicks-fake-headlines}
}`;
  const TEXT = "Amr Sobhy. 2026. Real Clicks, Fake Headlines: AI That Chases Clicks Invents the News. AI x Epistemics Research Sprint, Apart Research.";
  let copied = $state<"" | "bib" | "text">("");
  let fallback = $state("");
  let status = $state("");
  async function copy(which: "bib" | "text") {
    fallback = ""; status = "";
    try { await navigator.clipboard.writeText(which === "bib" ? BIB : TEXT); copied = which; status = "Copied to clipboard."; setTimeout(() => (copied = ""), 1600); }
    catch { fallback = which === "bib" ? BIB : TEXT; status = "Clipboard unavailable. Select and copy the text below."; }
  }
</script>

<section class="part" id="cite">
  <p class="kicker">Cite</p>
  <h2>Cite this work</h2>
  <p class="cite-text">{TEXT}</p>
  <div class="surface box">
    <pre>{BIB}</pre>
    <div class="actions">
      <button type="button" onclick={() => copy("bib")}>{copied === "bib" ? "Copied" : "Copy BibTeX"}</button>
      <button type="button" onclick={() => copy("text")}>{copied === "text" ? "Copied" : "Copy citation"}</button>
    </div>
    <p class="note" role="status">{status}</p>
    {#if fallback}<textarea aria-label="Citation to copy manually" readonly value={fallback} onclick={(e) => e.currentTarget.select()}></textarea>{/if}
  </div>
</section>

<style>
  .cite-text { color: var(--ink2); font: var(--type-body); max-width: 72ch; }
  .box { display: grid; gap: var(--s4); }
  pre { margin: 0; overflow-x: auto; white-space: pre; color: var(--ink2); font: 400 13px/20px var(--text); font-variant-numeric: tabular-nums; }
  .actions { display: flex; flex-wrap: wrap; gap: var(--s2); }
  button { height: 36px; padding: 0 var(--s4); border: 0; border-radius: var(--r-pill); background: var(--paper); color: var(--ink); font: var(--type-control); cursor: pointer; transition: background-color var(--fast) var(--ease), scale var(--fast) var(--ease); }
  button:active { scale: .97; }
  @media (hover: hover) { button:hover { background: var(--wash2); } }
  button:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }
</style>
