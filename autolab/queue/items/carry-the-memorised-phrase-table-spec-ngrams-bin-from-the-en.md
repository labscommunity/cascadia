---
id: carry-the-memorised-phrase-table-spec-ngrams-bin-from-the-en
title: carry the memorised-phrase table (spec-ngrams.bin) from the entry box to the box that plays rank 0
status: ready
outcome: 
priority: 6
target: single stream
exact: yes
needs: overrides only: the entry box copies the file into model/.role-swap/ (served by role_sync.py), the other box fetches it once before its worker starts
proposed_by: autolab session 2026-09-20
owner: 
experiment: 
created: 2026-09-20
updated: 2026-09-20
---

## Hypothesis
Since the role swap the table of phrases learned across requests stayed on the old box: same prompts, true/false
9.4 -> 4.3 tok/s, rewrite 7.1 -> 5.6. It re-learns with traffic; carrying it over restores it at once.

## Kill
A gate failure (guesses never change tokens, so this can only cost speed).

## Result
