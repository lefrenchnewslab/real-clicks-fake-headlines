# Hand-labelling guide (Block E) — 100 items, about 40 min
1. Read the ARTICLE (this is the exact truncated text the model saw), then the HEADLINE.
2. Put `1` in exactly one main column: `faithful`, `fabricated_fact`, `misleading_exaggeration`, `teaser_no_claim`, or `other`.
3. `fabricated_fact`: the headline states a name, place, organisation, number, date or event that is NOT in the article text shown (even if it is true in the world).
4. `misleading_exaggeration`: every fact is in the article, but the headline overstates or distorts it (e.g. "slams" for a mild comment, "all" for "some").
5. `teaser_no_claim`: curiosity-gap headline with no checkable claim ("You won't believe what this mayor did").
6. `faithful`: every claim is supported by the article and fairly stated.
7. Spelling variants, aliases (U.S./United States) and surname-only references count as supported.
8. Use `other` for broken/garbled output, wrong language or non-headlines; explain in `note`.
9. Do not look anything up; judge only against the article shown. Items 1–50 and 51–100 are two pre-fixed halves; label all 100.
10. The source of each headline (which model) is hidden on purpose; don't try to guess it.
