---
id: expert-usage-skew-per-layer-are-the-top-64-experts-60-of-sel
title: expert-usage skew per layer (are the top 64 experts > 60 % of selections?) to price hot-expert replicas
status: running
outcome: 
priority: 30
target: single stream
exact: n/a
needs: binary (env-gated counters on every rank, default off) + overrides; measured on the fleet under real traffic, no offline machine
proposed_by: teammate (spec.md appendix)
owner: autolab-continuation-20260921
experiment: 040_expert_usage
created: 2026-09-20
updated: 2026-09-21
---

## Why on the fleet
The routers run on every rank for every row. Counting there sees real traffic (any prompt mix, any number of tokens)
on the fleet's own numerics; a dump from another machine would see 36 prompts of a CPU path whose text departs from
the fleet's.

## Method
`CASCADIA_INKLING_EXPERT_COUNTS=1`: each MoE layer keeps a [256] counter of routed selections. Every N frames the rank
prints one "stage profile" line per layer with a CHANGING tag (the relay keeps one record per tag):
`EU<layer>_<n> probe stage profile layer= rows= top16_ppm= top32_ppm= top64_ppm= max_ppm= distinct=`. Run the twelve
prompt families at 15 streams for ten minutes; read with `probe_read.py EU`.

## Decision
Top 64 experts above 60 % of selections at most layers: price hot-expert replicas (MOONSHOTS, single stream).
Otherwise close that idea with the numbers.

## Result
