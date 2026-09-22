# Discoveries

1. **A pipeline's admission order can starve it.** Rounds start at group 0 and requests arrive a
   few per round, so "admit into the group whose turn it is" fills the first groups only
   (15/10/8/6/4/4/2/0/0/0/0 at 48 streams). Every rank then idles two thirds of the time.
2. **Rows do not share expert reads.** MoE cost is 33-34 ms per row per rank at 1 and at 7 rows
   per frame, although 2 of each row's 8 experts are the same shared experts.
3. **`EXPERT_CACHE_MIB` is per layer**: 8000 MiB x 6 layers is the resident copy of the rank's
   experts (anon memory), next to a partial second copy in the page cache.
4. **Prompts over 256 tokens take the fleet down** (one StreamOpen frame over MAX_STREAM_ROWS).
5. **The fleet is two kinds of box**: ranks 0-7 are held at 25-31 W, ranks 8-10 run at 60 W and
   are 1.4x faster per stage.
6. **A frame costs `16 + 19 r` ms per stage and the device-call side does not give.** Decode kernels, a second thread of device calls, reused input tensors, a sleeping completion wait: all measured, none moves the 16 (026, 020, 023, 029).
7. **The GPU plugin's MoE decode kernels refuse int4 group 32 on Xe2 and newer, silently.** The assert is swallowed at compile time and `infer()` throws "Unable to cast reference from base to derived type". The engine's host fallback then filled the box and the OOM killer looked like a crash (021, 022, 025). Six compiled `32`s turned into `16`s (the pre-Xe2 configuration) make them accept group 32 exactly; they are no faster at one row (029).
8. **A dense MLP is an all-experts-active MoE layer.** Cut into eight slices and sent through the fused-experts op, rank 0's dense layers read their int4 weights twice as fast as three compressed MatMuls (4.5 vs 8.1 ms), bit-compatible to 6e-4 between two f16 paths (027).
9. **A closed ring punishes late replies more than it rewards saved work.** Sharing the last rank's head calls saved its bus 5 GB a round and lost 2.5 %, because a deferred reply is a deferred next frame (028). Balance a ring by making a stage cheaper, never by making a frame wait.
10. **A power loss on the entry box blocks every release.** Its clock resets to the firmware date, `publish.py` stamps the manifest with it, and updaters refuse a manifest older than the one they applied. Self-healed in the overrides since 032 (HTTPS `Date` header through the proxy, forward only).
11. **"11/11 serving" is not settled.** Requests that reach a chain still assembling wedge it until the next release (028's first rollout). `steady 3/3` or nothing; `lab.py publish` now says `SETTLED` / `NOT SETTLED`.
12. **The telemetry relay keeps one record per probe tag, the first.** A progress probe under a constant tag looks frozen for ever (032: 25 minutes of doubt while 51 GB moved). The entry box's log had the truth.
13. **A role is not a box.** Two boxes can exchange pipeline roles (rank, layers, API) while everything that identifies a box stays (beacon rank, names, updater source, tunnel, keys): `run.sh` `ROLE_SWAP`, `role_sync.py`, `api_relay.py` (032). Telemetry and `status.sh` list boxes by installed rank.
14. **The model ships its own drafter.** `mtp.safetensors` (eight chained dense MTP modules, dropped by the exporter) names the next token 0.73 of the time on this model's text, against 0.54 for a 0.6B draft model and 0.30-0.38 for n-gram tables; its deeper modules need their own attention and conv state kept current (0.11 without). A deeper rank's state cannot guess the next token (logit lens 0.00 through rank 4) (033).
15. **One of eight identical boxes is not identical.** At the same work the entry box is 8-10 % slower and draws 1-2 W more than its seven siblings, and it is the one that loses power (032).
