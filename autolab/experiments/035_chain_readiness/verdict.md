# 035: keep the full-chain readiness gate

Both fleet output gates passed after the release settled. With the same
034 prompts, fifteen-stream phases completed at 24.212 / 24.631 tokens/s
steady, against 24.580 / 24.679 before the ten-frame experiment. All thirty
requests completed, first token averaged 6.61 / 6.10 seconds. No claim of a
throughput gain; the idle handshake adds no per-frame traffic.

The three-worker integration fixture tests the unsafe startup condition:
all TCP sockets connected, last worker deliberately held. Admission stays
closed until the last worker acknowledges the stateless probe; recovery
requires no generation traffic. Generated tokens then match the reference.
Nine related pipeline tests passed. The fleet itself was never deliberately
sent requests while unsettled. Keep CASCADIA_STREAMS_READY_GATE=1 on role 0.

Binary: cascadia-5b090e55. Overrides: 035_chain_readiness.env, based on the
032 role swap, with eleven frames restored. Capture remains off in this
comparison. The first gate request included lazy initialization (10.47 s
to first token); later single requests were 2.1–2.2 s.
