# 029 — matias 2-box K2.6 revival (SSH-tunnel-via-Mac workaround)

Iter 004 parked matias because Tailscale broke on the matias boxes
(`tailscale up --reset` flushed creds, no authkey available to re-auth).
This experiment unblocks the 2-box demo without Tailscale by chaining
the inter-rank wire through SSH tunnels that pass through the controller
Mac.

## Operational summary

**Blocker.** Both matias-02 and matias-03 are stuck in `unexpected state: NoState`
with `Tailscale is starting / You are logged out`. `Restart-Service Tailscale`
does not reconnect — `tailscale up` errors with `requires mentioning all
non-default flags. To proceed, either re-run your command with --reset`,
and `--reset` triggers a re-auth flow that needs either a browser SSO
loop or a pre-generated `tskey-auth-*` we do not have in the loop's
environment.

**Pivot 1: local LAN fleet** (`beta` 192.168.86.31, `charlie` 192.168.86.39).
Both are SSH-reachable from the controller Mac, but neither has the
K2.6 model. Each box has ~360 GB free C: — not enough for a full 275 GB
shard. Transfer from miner (only K2.6 source on the home LAN) would
take ~14 hr/box at the previously-observed bastion-mediated ~5 MB/s.
Not viable inside the iteration budget.

**Pivot 2: SSH-tunnel chain via Mac.** Bridge matias-02 and matias-03
by chaining two SSH port-forwards through the controller:
```
matias-02:9100  --(ssh -R)-->  Mac:19100  --(ssh -L)-->  matias-03:9100
```
Plus an API tunnel `Mac:18000 -> matias-02:8000` for the bench harness.

End-to-end RTT measured at 117 ms median (20 frames, 8 bytes each)
vs ~22 ms over direct Tailscale DERP — but still <2% of K2.6's
~9 s per-token decode budget, so transport is not the bottleneck.

## Mechanism

1. From Mac:
   ```
   ssh -f -N -L 19100:127.0.0.1:9100 cascadia-matias-03  # forward leg
   ssh -f -N -R 9100:127.0.0.1:19100 cascadia-matias-02  # reverse leg
   ssh -f -N -L 18000:127.0.0.1:8000 cascadia-matias-02  # API tunnel
   ```
   Critical: **use `127.0.0.1` not `localhost`** as both the bind and
   target. With `localhost`, ssh's `direct-tcpip` channel resolves to
   `::1` on the remote and the forward gets `channel_free`d immediately
   — the listener never sees data.

2. Launch rank-1 on matias-03 with `--listen :9100`.
3. Launch rank-0 on matias-02 with `--next 127.0.0.1:9100`. From rank-0's
   POV it just connects to `localhost:9100`; the kernel routes via the
   `ssh -R` reverse forward to Mac:19100, where `ssh -L` forwards on to
   matias-03:9100.

## WMI-detached spawn

Both ranks must be launched via `Invoke-WmiMethod -Class Win32_Process
-Name Create` (see `bench/spawn_rank{0,1}_wmi.ps1`). `Start-Process
-WindowStyle Hidden -PassThru` inherits the OpenSSH job object and gets
killed the moment the SSH session closes — silently losing the worker.
WMI Win32_Process is the only reliable Windows-OpenSSH detachment path,
matching the pattern documented in tahoma-demo's `win_spawn_wmi.ps1`.

## Result

10-prompt eval @ K=8, mt=32, temp=0 (raw JSONL: `bench_k8_mt32_10p.jsonl`):

- **Aggregate: 0.0770 tok/s, 9/10 quality**
- Total wall: 4156 s for 320 tokens
- Single quality fail is the documented "kilometers vs km" substring artifact (model answered correctly, substring matcher mismatched)
- Per-prompt tok/s range: 0.0745 - 0.0805 (tight spread; tunnel adds negligible variance)

vs baseline iter 000 (3-prompt mt=8): 0.0553 tok/s, 3/3. The +39%
delta is methodology (mt=32 vs mt=8 prefill amortization), not pipeline
improvement. The implied per-token decode rate is unchanged at ~0.11
tok/s (per iter 003 instrumentation). Transport is not the bottleneck
on either path; tunnel chain works.

## Reproduce

```bash
# From Mac, with cascadia-matias-{02,03} SSH config aliases set up:
./autolab/bench/start_workers_tunnel.sh
# Wait ~1-40 min for shells to load (cold = 40 min; warm cache ~1 min).
# Poll: curl http://127.0.0.1:18000/health
./autolab/bench/k26_bench_matias_10.sh http://127.0.0.1:18000 /tmp/k26-bench.jsonl 32
```

## Files

- `bench_k8_mt32_10p.jsonl` — 10-prompt aggregate
- `rank0_stage_timings.txt` — extracted stage_timing lines from rank-0 (one per token)
- `rank1_stage_timings.txt` — extracted stage_timing lines from rank-1
