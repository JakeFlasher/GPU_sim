# SM86 store acknowledgement frontier assay

## Goal and completion bar

Build and execute one `sm_86` CUDA binary that separates three observations
for the same scalar-global STORE stream:

1. scheduler-visible issue work before any fence;
2. explicit device-scope `MEMBAR` visibility; and
3. the implicit task-boundary handoff to a normal same-stream sentinel kernel.

This is an identification assay for the simulator's write-completion frontier,
not authorization to fit a terminal constant or clear dirty-writeback and
coherence tails. Completion requires correct live execution, static SASS,
deterministic raw observations, an exact NVBit trace/replay bridge for selected
cells, and either a transferable mechanism discriminator or a bounded no-go.
NCU may be collected later as an independent comparison target, but no NCU
metric may choose this assay's mechanism or parameters.

## Single-binary contract

One runtime-selected kernel contains all three arms:

- `control`: address/checksum work, no payload STORE, no explicit fence;
- `store_none`: identical payload STORE path, no explicit fence; and
- `store_device`: identical payload STORE path followed by `membar.gl`.

Each CTA is one warp. Lane zero records `%globaltimer` immediately before the
payload loop, after the issue loop, and after the optional fence. Every CTA
writes its own triplet. A second ordinary kernel is enqueued in the same CUDA
stream and records `%globaltimer` at entry before consuming one payload word.
CUDA stream ordering makes this a post-completion handoff; no programmatic
dependent launch is used. A timer-only 250-us queue gate is enqueued before
the measured pair so both producer and sentinel are already queued when the
producer begins; the host enqueue duration must fit inside the observed gate.
The gate is excluded from every reported pair interval. The host reports:

- per-CTA issue and explicit-fence intervals;
- grid span from earliest start through latest post-fence timestamp;
- `sentinel_start - max(post_fence)` as the terminal-handoff observation; and
- an ordinary CUDA-event duration for the probe-plus-sentinel pair.

`%globaltimer` is a target-specific NVIDIA-tools register. Its values are used
only on the pinned RTX 3090 Ti/sm_86 host and are never generalized across
architectures. The control arm carries the same timestamp stores and sentinel
launch, so their common overhead is removed only through matched comparisons.

## Identification and holdout

Identification matrix:

- blocks: 1 and 84;
- depth: 1, 4, 16, 64;
- active lanes: 32;
- layout: unique 128-byte line per lane/operation;
- arms: control, store-none, store-device;
- a store-device active-lanes-zero calibration for the empty-fence cost; and
- at least 15 independent process repeats with an odd number of in-process
  trials per cell.

Open holdout only after correctness and identification structure pass:

- active lanes 1 and 8;
- repeated-line reuse at depth 32;
- unique-line depth 32 with a dependent-IMAD gap; and
- one deeper B84 unique-line cell.

The selected NVBit bridge is B1/D4 plus B84/D16 for all three arms. Trace
cardinality, empty unsupported/unmapped/cross-allocation sidecars, static SASS,
and current simulator twin determinism are mandatory.

## Exact gates

- All timestamp triplets are nonzero and ordered.
- The sentinel global timestamp is not earlier than any CTA post-fence time.
- The queue gate outlives host enqueue of the complete producer/sentinel pair.
- STORE arms produce the exact expected payload; control leaves it unchanged.
- The sentinel reads the expected final payload value for every arm.
- The two STORE arms have identical payload instruction/request/sector work.
- `store_device` contains exactly one dynamic device-scope fence per CTA;
  `control` and `store_none` contain none.
- Loaded device-fence work is reported only after subtracting the matched
  active-lanes-zero store-device fence interval.
- Repeats retain the raw trial observations; summaries never replace them.
- The same arm/cell is deterministic in structural counts. Timing variability
  is reported rather than hidden.
- A positive terminal residual must separate store-none from its matched
  control and contract after explicit device fencing across both B1 and B84.
- Any proposed simulator timing must transfer to untouched depth/layout/lane
  holdouts without reading their target values during selection.
- Default simulator replay remains byte-compatible. Timing-active replay keeps
  dirty-writeback/coherence tails and changes only its declared write-ack
  lifecycle and terminal lower bound.

## No-go and claim ceiling

Return a complete no-go if the sentinel handoff is dominated by launch jitter,
if explicit fencing does not consistently contract the matched residual, if
the effect reverses across B1/B84, or if the holdout cannot discriminate a
mechanism. Do not repair a no-go by fitting a kernel-end constant, using NCU
targets as features, pooling unrelated maxima, or clearing a typed tail.

A positive result identifies only a host-specific STORE visibility/terminal
relation for the tested scopes and layouts. Dirty eviction/writeback,
coherence propagation, RED read-modify-write, system scope, and cross-device
visibility remain separate work.
