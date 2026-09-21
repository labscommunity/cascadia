# 038: keep the transferred phrase history

The one-time verified merge succeeded: 24,453 contexts on the current head
became 194,299. The source table was 5,265,261 bytes. Both serial and
concurrent gates passed; all six paired requests completed.

| Original 031 prompt | Before, tok/s | After, tok/s |
|---|---:|---:|
| Explain | 4.116 | 7.730 |
| Story | 4.260 | 7.219 |
| Code | 5.225 | 10.600 |
| Arithmetic | 9.865 | 10.024 |
| Rewrite | 5.869 | 8.234 |
| True/false | 4.909 | 10.767 |

These are repeated, already seen prompts. The before run itself also trained
the current table, and capture was disabled with the transfer release.
Therefore this is evidence of working retained phrase history, not an
isolated causal estimate or a gain on unseen prose. Multi-stream speculation
is still disabled. Preserve the merged table and its backup; the transfer
marker makes subsequent starts idempotent. No binary change in this release.
