# 002a direct reply link (+ telemetry door)

Binary 6bad6fd8 (sha 12f998b8), beacon 7e3012f2 (aggregate file + file-server supervision),
overrides = 001 + CASCADIA_STREAMS_RETURN_PORT/HOST. Speculation compiled in, switched off.

Result of 001 to beat (warm, same phases): single 1.43-1.6, 11 streams 8.9 tok/s steady
(5.45 aggregate), 48 streams 14.7 steady (8.98 aggregate).

Hypothesis: rank 0's group turn is about 1.7x a stage time because every reply waits at the busy
ranks it is relayed through. With the direct link the turn approaches the slowest stage's time:
11 streams toward 11 / (11 x 0.055 s) = ~16-18 tok/s steady, 48 streams toward 4.4 rows / 0.18 s
= ~24 tok/s. Single stream: -9 relay hops x ~1.7 ms = about -15 ms of 590 (+3 %).
If the numbers do not move, the relay was not the wait, and the stage profiles (now readable
through /api/fleet/telemetry) must say where rank 0's wait comes from.
