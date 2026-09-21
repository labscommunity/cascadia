---
id: fleet-wiring-for-mtp-drafts-final-state-on-the-reply-link-mt
title: fleet wiring for MTP drafts: final state on the reply link, MTP blocks on rank 0, guess rows for many streams in the scheduler
status: blocked
outcome: 
priority: 20
target: 15 streams, interactive speed at 3-8 streams
exact: yes
needs: blocked on the offline MTP acceptance result; binary + export of the MTP blocks
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
