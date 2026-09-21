# 035: refuse admission until all workers can receive frames

An open rank-1 socket does not establish that the other nine workers have
loaded and connected. The existing idle link keeper can perform a stateless
handshake down the ordinary chain, with an acknowledgement relayed back
from the last rank. CASCADIA_STREAMS_READY_GATE=1 makes rank 0 refuse
submit with NotConnected (HTTP 503) until that succeeds. Default is off.

Prediction: requests during startup are refused, readiness recovers without
client traffic, and after the handshake tokens and throughput are unchanged.
An integration test delays the final worker after all TCP links connect,
checks repeated rejection, then starts it and checks exact generated tokens
against a single-stage reference. Existing pipeline and speculation tests
must pass. Fleet rollout follows normal settle and both output gates;
no deliberate generation traffic during an unsettled release.

Kill: test or fleet output failure; return to cascadia-8baebd3c and the
last kept overrides. The handshake must not touch KV, samplers or weights.
