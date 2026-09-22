---
id: analyze-py-lab-py-map-telemetry-rows-to-pipeline-roles-under
title: analyze.py / lab.py: map telemetry rows to pipeline roles under a role swap; stop double-counting the last rank (its clock is a day ahead)
status: done
outcome: kept
priority: 10
target: tooling
exact: n/a
needs: harness only, no fleet change
proposed_by: autolab session 2026-09-20
owner: autolab-continuation-20260921
experiment: 036_telemetry_roles
created: 2026-09-20
updated: 2026-09-21
---

## Hypothesis

## Method

## Prediction

## Kill

## Result

036: kept. Roles come from worker profiles while physical box identity stays
explicit; duplicate windows and probe tags are excluded, and receive times
place windows on the operator clock. Synthetic skew/backlog tests passed,
and 032 was re-analyzed with role 0 correctly mapped to installed box 8.
