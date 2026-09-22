---
id: fleet-wiring-for-mtp-drafts-final-state-on-the-reply-link-mt
title: fleet wiring for MTP drafts: final state on the reply link, MTP blocks on rank 0, guess rows for many streams in the scheduler
status: blocked
outcome: 
priority: 5
target: 15 streams, interactive speed at 3-8 streams
exact: yes
needs: 039 original a1 0.6681 and deployment grids 0.6436 miss 0.70 qualification bar; depends on 042 fleet-trained candidate before runtime deployment
proposed_by: teammate (spec.md E5) + autolab
owner: 
experiment: 
created: 2026-09-20
updated: 2026-09-21
---

## Method
spec.md E5a/E5b, plus what the 15-stream goal adds: today speculation is a lone-stream mode; guess rows for many
streams need per-stream rewind bookkeeping in the group scheduler, and the head's cost lands on rank 0 (fixed by
027: it has ~4 ms of slack against rank 10).
