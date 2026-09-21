---
id: fleet-wiring-for-mtp-drafts-final-state-on-the-reply-link-mt
title: fleet wiring for MTP drafts: final state on the reply link, MTP blocks on rank 0, guess rows for many streams in the scheduler
status: ready
outcome: 
priority: 5
target: 15 streams, interactive speed at 3-8 streams
exact: yes
needs: multi-day build: export the MTP blocks (int4 MLP, int8 attention), run them on the box that plays rank 0 with their own KV + conv state, final state on the reply link, guess rows for MANY streams in the scheduler, 65k-row draft unembed; start with module 0 only; its first step IS the fleet state capture item (final state written on the last rank / carried on the reply link)
proposed_by: teammate (spec.md E5) + autolab
owner: 
experiment: 
created: 2026-09-20
updated: 2026-09-20
---

## Method
spec.md E5a/E5b, plus what the 15-stream goal adds: today speculation is a lone-stream mode; guess rows for many
streams need per-stream rewind bookkeeping in the group scheduler, and the head's cost lands on rank 0 (fixed by
027: it has ~4 ms of slack against rank 10).
