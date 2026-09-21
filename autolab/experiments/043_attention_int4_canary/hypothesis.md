# 043: prepared int4 attention quality canary, awaiting owner decision

041 measured a 39–43% projection-kernel saving, with 9.6–9.7% relative RMS
output deviation on random inputs. Prepare a temporary role-5-only canary
(layers 30–35), separate attn_ov_int4_043 directory, original int8 IRs retained.
No binary or library changes. Export runs before worker load, bounded to
600 seconds. Twelve-start cutoff and signal-death detection disable the
candidate; a partial export is never selected. Docker tests exercise role
selection, success, idempotence, start cutoff and partial-export fallback.

Prediction: role-5 frame time improves about 3 ms (roughly 8%); fleet rate
may barely change because the other ranks retain int8. Changed text is
expected; the required owner decision must explicitly permit that temporary
numerical difference for quality evaluation, overriding the ordinary exact
gate rule only for this trial. Without that decision, publish nothing.

If authorized: finish current capture, save baseline quality results, publish
candidate, wait settled, compare the ordinary gate outputs and run the same
12-prompt quality test, then restore original int8 overrides and verify the
exact gates. Stop immediately for crashes, nonfinite output, failed requests
or obvious answer degradation. Regardless of outcome, this trial does not
authorize a permanent or fleet-wide numerical change. The original int8
rollback is already prepared as 043_attention_int4_rollback.env.
