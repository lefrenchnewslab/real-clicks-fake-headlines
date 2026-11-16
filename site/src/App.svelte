<script lang="ts">
  import { onMount } from "svelte";
  import { ThemeButton, createTheme, tips } from "@stdlogits/ledger";
  import Hero from "./parts/Hero.svelte";
  import Setup from "./parts/Setup.svelte";
  import Train from "./parts/Train.svelte";
  import Drift from "./parts/Drift.svelte";
  import Distance from "./parts/Distance.svelte";
  import Final from "./parts/Final.svelte";
  import Checks from "./parts/Checks.svelte";
  import Scale from "./parts/Scale.svelte";
  import Record from "./parts/Record.svelte";
  import Cite from "./parts/Cite.svelte";

  const theme = createTheme({ key: "rcfh.theme" });
  let active = $state("top");
  const chapters = [
    ["setup", "Study"], ["train", "Training"], ["drift", "Evidence"],
    ["final", "Results"], ["distance", "Why it differs"], ["checks", "Checks"],
    ["scale", "Scale"], ["record", "Limits"], ["cite", "Cite"],
  ];
  onMount(() => {
    const stopTheme = theme.start();
    const update = () => {
      const sections = [...document.querySelectorAll<HTMLElement>("main > section[id]")];
      active = sections.filter((s) => s.getBoundingClientRect().top <= 160).at(-1)?.id ?? "top";
    };
    window.addEventListener("scroll", update, { passive: true });
    update();
    return () => { window.removeEventListener("scroll", update); if (typeof stopTheme === "function") stopTheme(); };
  });
</script>

<div class="site" use:tips>
  <a class="skip-link" href="#setup">Skip to the study</a>
  <header>
    <a class="mark" href="#top">Real Clicks, Fake Headlines</a>
    <nav aria-label="Chapters">
      {#each chapters as [id, label]}
        <a href={`#${id}`} aria-current={active === id ? "location" : undefined}>{label}</a>
      {/each}
    </nav>
    <ThemeButton {theme} />
  </header>

  <main id="top">
    <Hero />
    <Setup />
    <Train />
    <Drift />
    <Final />
    <Distance />
    <Checks />
    <Scale />
    <Record />
    <Cite />
  </main>

  <footer>
    <span>Amr Sobhy · Le French News Lab · Apart AI x Epistemics sprint, November 2026</span>
    <a href="https://github.com/lefrenchnewslab/real-clicks-fake-headlines">Code and data</a>
    <span data-tip="Built on Ledger, the design language of neuron-lens.com">Drawn in Ledger</span>
  </footer>
</div>
