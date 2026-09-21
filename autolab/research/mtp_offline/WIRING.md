# MTP runtime design notes (pending the fleet re-score)

The measured target is interactive service up to fifteen streams. Module 0
alone supplies one guess after a verified next token; it cannot fill an
eleven-stage single-stream pipeline without deeper modules or another
drafter. Do not apply the eight-module 033 projection to a module-0 runtime.

## State and arithmetic

037 found raw final residuals beyond f16 range. Capture now uses exact f32
(INKCAP02). A compact f16 reply must carry **post-final-RMSNorm** states;
otherwise carry f32. Never cast the raw final residual to f16.

At anchor position t, module 0 consumes the target's post-final-norm state
and the backbone-normalized embedding of the verified token x[t+1]. Apply
the MTP hidden norm and MTP embedding norm separately, concatenate hidden
first, project, run one dense block, then unembed h/mup **without another
output norm**. It predicts x[t+2]. Its KV and all four convolution histories
must advance for every accepted anchor, including prompt context.

`export_module.py` writes a one-layer manifest and files compatible with
`loader::load_stage(first=false,last=false)`: original shell, group-32 dense
bin, int8 attention, fused all-active dense IR, input projection and 65k
unembed IRs, and three norm vectors. Reuse the role-0 backbone embedding.
`OvHead::logits_rows` can execute the input projection as well as the head.
No synthetic or zero fallback weights are needed. Backend failure must
disable drafting and preserve target inference.

## One guess per stream, batched verification

For the fifteen-stream trial, use an advance frame with one segment per
slot, containing the known next token and optionally one draft token. The
existing `prefill_streams` / `Layer::forward_prefill_slots` machinery already
supports consecutive positions per slot while batching linear/device work
across segments. Ordinary `decode_streams` expects one position per slot
and must not be fed duplicate slots.

The last rank samples every row in segment order using the existing sampler
state. If the first sampled token equals the draft input, retain both rows;
otherwise retain just the first and rewind the target KV, conv and sampler
state. Never emit the prediction from a rejected input row. Handle EOS,
stop conditions, token budgets, cancellation and slot reuse before reuse of
either target or draft state. Test non-greedy sampling and repetition
penalties as well as the greedy fleet gate.

Replies must carry each row's final state as well as actual sampled IDs.
The driver advances MTP only over accepted target rows, using each row's
verified successor embedding; rejected target states never enter MTP.
Batch MTP upkeep across the frame's slots, and unembed only the final row
needed per slot. Calling an 8 ms drafter independently for fifteen streams
would consume more than the available rank-0 slack.

Prompt windows need all their final states for context. One possible first
implementation buffers a bounded number of prompt states on the last rank
and attaches them to the final prompt reply; larger prompts decline MTP
for that stream. Explicit bounds are required for a burst of long prompts.
The default-off wire feature must be consistent across rank 0 and the last
rank, with shape/length validation and an ordinary-path fallback.

## Delivery and validation

The only fleet mutation path is the existing signed six-file release.
No updater, tunnel, key or file-server changes. An appended archive in an
ELF executable is a possible asset carrier, but size, all-eleven download
time against the updater's 120-second timeout, disk headroom and extraction
must be checked before choosing it. Keep a separate plain binary for quick
rollback. Once assets are installed and verified, later binaries need not
carry them again. Do not add an unreferenced blob to the publisher store:
its manifest-based garbage collector can remove it.

The full module export is 1,515,813,614 bytes; tar+xz -3 gives
1,106,436,660 bytes. Eleven downloads through one 1 GbE source are too
close to the timeout to assume a single capsule is suitable. Splitting IRs
and raw fallback files across two existing planned releases is an option;
neither part may enable the runtime until both are verified. These assets
and archives stay outside the repository.

Before any fleet enablement: PyTorch/weight-grid parity, stream-context and
rewind tests, bounded reply parsing, one/many-stream exact reference tests,
then a one-rank cost measurement. Fleet gates and quality checks follow
settling. Only measured acceptance and runtime cost justify enabling
speculation at a given concurrency; the renewal projections are hypotheses.
