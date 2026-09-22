---
id: carry-the-memorised-phrase-table-spec-ngrams-bin-from-the-en
title: carry the memorised-phrase table (spec-ngrams.bin) from the entry box to the box that plays rank 0
status: done
outcome: kept
priority: 6
target: single stream
exact: yes
needs: verified transfer complete; 24453 -> 194299 contexts; both gates pass
proposed_by: autolab session 2026-09-20
owner: autolab-continuation-20260921
experiment: 038_phrase_transfer
created: 2026-09-20
updated: 2026-09-21
---

## Hypothesis
Since the role swap the table of phrases learned across requests stayed on the old box: same prompts, true/false
9.4 -> 4.3 tok/s, rewrite 7.1 -> 5.6. It re-learns with traffic; carrying it over restores it at once.

## Kill
A gate failure (guesses never change tokens, so this can only cost speed).

## Result
