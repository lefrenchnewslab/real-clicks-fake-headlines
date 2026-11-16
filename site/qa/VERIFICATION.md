# Companion-page verification — 2026-11-15

## Result

Production build passes without warnings. Core desktop journeys were exercised in Helium through native browser controls. This is a manual E2E record, not an automated test suite.

Standalone artifact: `../dist/index.html` (generated, ignored by git).
SHA-256: `09cd9c28a29909a9f22989449c64a6ecb92a0343934f5b0c868c697e7c802891`

## Repeat the checks

Run `npm ci`, `npm run build`, then `npm run dev -- --host 127.0.0.1` from `project2/site/`. Open the printed URL in a fresh browser tab.

| Journey | Expected result | Observed |
| --- | --- | --- |
| Open page | Hero, navigation and writer selector render | Pass; desktop visual inspection |
| Results → Article 2 | Article ID changes to cc04066; all seven rows update to EU–China headlines | Pass |
| Training → 350 · Final step, seed 0 | Step reads 350; click run explains its step-240 collapse stop | Pass |
| Switch to seed 1 | Click-run drafts become available | Pass |
| Focus training range and press Right | Step increments by one and all three draft cards update | Pass; 175 → 176 |
| Production build | A self-contained HTML file is generated | Pass |
| Mobile at 390px | Chapter navigation remains available; charts scroll within their panels | CSS reviewed; browser emulation could not be verified reliably |
| Theme, clipboard fallback, labelled-square selection, all chart selectors | State and visible result agree | Implemented and source reviewed; not comprehensively exercised in browser |

The native browser's device controls failed to apply dimensions consistently. Do not interpret the build or desktop checks as full mobile, screen-reader, or cross-browser certification. The browser console was not reliably inspected.

## Content checks

Compared the explanations with the manuscript's abstract, introduction, methods, results, discussion and limitations. Preserved numerical results. Clarified predicted clicks versus live readership, seed-0 hero versus pooled final results, automated flags versus reference labels, the best-of-16 KL bound, and reward-specific chart values. No manuscript or experimental-data changes.
