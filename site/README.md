# Real Clicks, Fake Headlines: interactive companion

A one-page Svelte 5 site drawn in Ledger (`@stdlogits/ledger`, the design system of neuron-lens.com), like the companion of the first sprint paper.
Every number comes from `src/data.json`, built by `build/make_data.py` from the project's saved results: training reward logs (`runs_main/*/reward_log.jsonl`), checkpoint evaluations (`trajectories.csv`), final test outputs (`runs_stage3/`), the reference labels (`handlabel/`) and `scale_E.json`. Values typed into the components (Tables 2, 5 and 9 of the paper, the verdicts) are copied from the paper.
No article text is included; headlines are the writers' own outputs.

    npm ci
    npm run dev      # http://localhost:5199
    npm run build    # dist/index.html, one self-contained file

This folder is AGPL-3.0-or-later because it bundles @stdlogits/ledger; the rest of the repo keeps its own licence.
