---
id: fleet-state-capture-final-and-rank-boundary-residuals-writte
title: fleet state capture: final and rank-boundary residuals written on the boxes, fetched through the entry box's relay (replaces the Mac Pro dump)
status: running
outcome: 
priority: 5
target: offline studies on the fleet's own numerics; first step of the MTP wiring
exact: yes
needs: binary (env-gated, default off) + overrides (relay route); no offline machine
proposed_by: autolab session 2026-09-20 (user: use the fleet, not the Mac Pro)
owner: autolab-continuation-20260921
experiment: 037_fleet_state_capture
created: 2026-09-20
updated: 2026-09-21
---

## Why
033 measured the MTP head on hidden states dumped from the Mac Pro's CPU path, whose text departs from the fleet's
after 7-70 tokens (int4 CPU kernels against f16 fused experts + int8 attention). The heads will see the FLEET's
states. Every rank already holds what these studies need: the last rank sees the final residual of every position
(prompt rows pass through it too), every rank's output IS a rank-boundary residual. Nothing about this needs another
machine; it needs a way to write the states down and one read-only route out.

## Method
1. Engine, all behind environment variables that default to off:
   * last rank: `CASCADIA_STREAMS_CAPTURE_FINAL=<dir>`: per stream, append `(position, final residual before the
     final norm as f16 [6144], the token it sampled)` for prompt rows and decode rows alike (12 kB a position);
   * any rank: `CASCADIA_STREAMS_CAPTURE_BOUNDARY=<dir>`: the same for that rank's output residual (logit-lens work);
   * a byte budget (`..._CAPTURE_MAX_MB`, default 2048) after which capture stops, so a forgotten switch cannot fill a disk.
2. The door, overrides only: the capturing box serves its capture folder read-only on a LAN port while the switch is on
   (the pattern of `fleet/role_sync.py`: no listing, no writes, that folder only); `fleet/api_relay.py` on the entry box
   answers `GET /api/fleet/capture/<installed rank>/<file>` by streaming from that box. It reaches the operator through
   the existing tunnel: no new port, no key, no shell.
3. Prompt token ids come from the client side (the API rendering reproduces `usage.prompt_tokens` exactly: 033 checked
   it); generated ids are in the capture.
4. Sizes: 36 prompts x ~200 positions = 86 MB; 3 M positions for training a head = 36 GB (hours through the tunnel).

## Prediction
No measurable cost with capture off; with it on, < 1 ms a frame on the last rank (a 12 kB append per row).

## Kill
A gate failure with capture off; first-token or decode times moving by more than 1 % with capture off.

## Result
