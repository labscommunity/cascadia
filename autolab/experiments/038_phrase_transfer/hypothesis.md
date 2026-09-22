# 038: retain the old entry box's learned phrases

The role swap left the original phrase table on the entry box. Merge it
once into the current role-0 table, preserving both histories, with file
size and SHA-256 checks, an atomic replacement, a backup, and a completion
marker. Only the table and its manifest are exposed by a bounded read-only
server in a dedicated directory; no install-directory server or keys.

Prediction: more learned contexts after transfer, exact gate outputs, no
fifteen-stream throughput change (speculation is single-stream only).
Compare the same previously memorized single-stream prompts before/after,
while recognizing repeated prompts also train the current table. Context
counts and successful verified transfer establish the main outcome; do not
attribute every repeated-prompt gain to the transfer.

Kill: failed validation, missing completion probe, or changed gate outputs.
Keep the old table backup for reversal. The binary itself is unchanged.
Merge unit tests, actual HTTP transfer/idempotence checks, and a Docker
shell rig under set -u / set -a passed before deployment.
