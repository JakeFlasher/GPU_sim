# SM86 selected asynchronous-copy instrumentation-order milestone

## Goal and completion bar

Carry a timing-free, globally comparable per-SM order witness for selected
asynchronous-copy instructions from the NVBit injected callback through the
protobuf trace and into an observe-only simulator diagnostic. Prove that the
witness distinguishes the exact queued/drained W4 manipulation that per-warp
streams currently collapse.

Completion requires producer, wire, reader, and replay fidelity plus an
untouched W8 transfer. It does **not** authorize simulator timing, queue
capacity, service rate, architecture-card, or typed-tail changes. The evidence
root is
`results/captures/sm86_async_selected_order_trace_20260831_v1`. JADE/HETR,
CPU-GPU co-simulation, `sm_120`, and unrelated candidates remain out of scope.

## Prospective trace contract

Add opt-in `SIMBENCH_NVBIT_ASYNC_ORDER=1` instrumentation for exactly:

- `LDGSTS*`: async-copy issue anchor;
- `LDGDEPBAR`: nonempty-group commit anchor; and
- `DEPBAR.LE SB0,*`: asynchronous wait anchor.

The static classifier may inspect canonical SASS text to distinguish SB0 from
ordinary SB1--SB5 waits. Every selected dynamic instruction receives exactly
one 64-bit `instrumentation_order` token from the existing main-channel packet
order when the first injected operand creates its protobuf instruction. The
accepted launch's host receiver numbers only selected instruction creations
from one, so multi-operand LDGSTS allocates once and no additional device
instruction, atomic, clock read, or channel push is introduced. The token names
callback/channel order, not a silicon issue cycle, source reply, destination
service event, or completion time.

Add optional protobuf instruction field tag 14. Default-disabled captures omit
the field; existing tags and meanings do not change. Producer records retain
launch/CTA/SM/warp/function/VPC identity in their existing hierarchy; the new
scalar joins that exact instruction. Independent live captures are not required
to have identical raw protobuf bytes because SM/warp placement and unconstrained
poll-loop interleaving are capture observations. Compatibility is instead
established by field absence, a retained legacy replay byte golden, and exact
ordered-versus-tag-stripped replay normalization of the same capture.

Within every admitted launch:

- tokens are globally unique and exactly dense from one through the selected
  record count;
- per-warp tokens strictly increase with dynamic instruction order; and
- only statically joined LDGSTS, LDGDEPBAR, or SB0-wait records carry tokens.

Missing, duplicate, wrong-SM, wrong-PC, non-selected, multi-operand duplicate,
partial-flush, or non-dense order evidence fails closed. Legacy traces with no
field remain admissible and observation-free.

## Observe-only replay integration

Retain selected order records in a cold sparse warp sidecar; do not enlarge the
hot replay instruction. At SM construction, sort the selected records by token
and replay only the async group *state machine for observation*:

- an effective LDGSTS opens a nonempty group;
- LDGDEPBAR commits that open group;
- `wait_group N` consumes the committed prefix older than the newest N; and
- before every LDGSTS, count committed groups owned by other warps.

Expose readiness schema 21 with:

- selected, issue, commit, and wait event counts;
- issue events seeing other-warp committed groups;
- summed and maximum other-warp committed-group population; and
- any contradiction between the order witness and the per-warp async protocol.

This scan is diagnostic only. It must not constrain picks, warp eligibility,
memory requests, group completion, destination service, event time, or tails.
With the diagnostic removed and schema normalized, every replay byte and
timing/counter field must equal the same trace replayed without the optional
order field.

## Producer and wire canaries

Before the campaign, run focused synthetic/unit checks:

1. One warp executes straight-line `LDGSTS*N -> commit -> wait`; every selected
   instruction joins once and tokens are dense/ordered.
2. Two warps use forced A-before-B and B-before-A handshakes; token relations
   reverse exactly while each warp remains program-ordered.
3. Multi-operand LDGSTS still emits one token, not one per MREF.
4. Default-disabled capture of a retained one-warp probe emits zero tag-14
   fields; a retained legacy replay remains byte-identical, and stripping tag
   14 from an order-enabled capture leaves normalized replay output unchanged.
5. Python and Rust wire preflights bound the new optional field and reject
   malformed type/range/cardinality cases before protobuf allocation.

## W4 identification smoke

Capture two independent repeats of the existing final exact-work queue-overlap
endpoints:

- L1 queued and drained;
- L2 queued and drained.

Use the unchanged W4/G64 binary and already validated NCU work/cardinality.
Each endpoint must contain exactly:

- 2,048 LDGSTS issues;
- 832 commits;
- 256 waits; and
- 3,136 selected order records.

For every one of 64 epochs and both repeats:

- queued: all 12 background commits precede the target's first LDGSTS, and the
  target wait precedes all three background waits;
- drained: all three background waits (and their commits) precede the target's
  first LDGSTS; and
- a classifier using only these token relations identifies the arm in 64/64
  epochs for both routes.

Do not require raw token values or unconstrained interleavings to hash equally
across repeats. Require only the exact causal predicates.

Two simulator replays per cell must be byte-identical. Schema-21 counts and
causal populations must distinguish queued from drained, while normalized
production timing, traffic, schema-20 diagnostics, and typed tails remain
equal to the same trace with order observation removed.

## Untouched W8 holdout

Open only after every producer, wire, W4, and replay gate passes. Capture L2
W8/G64 queued and drained with the unchanged binary and two repeats. Each
endpoint must contain exactly:

- 4,096 LDGSTS issues;
- 1,856 commits;
- 512 waits; and
- 6,464 selected records.

For all 64 epochs and both repeats, queued has all 28 background commits before
target issue and target wait before seven background waits; drained has all
seven background waits before target issue. Classification must be 64/64 and
normalized replay timing/counters must remain unchanged.

## Scientific boundary

The existing injected callback and channel reservation already perturb
unconstrained warp interleaving. Numbering their accepted receive order adds no
device-side perturbation, but the field remains valid only as a witness for
relations enforced by the workload's synchronization. No `%clock64`,
`%globaltimer`, device atomic token, NCU cycle metric, target live-clock value,
or token delta may enter a gate, simulator feature, or timing calculation.

## Minimum executable evidence

1. Implement and regenerate pinned C++/Python/Rust protobuf bindings.
2. Build/test the tracer and both wire preflights.
3. Run default-disabled byte compatibility and forced-order canaries.
4. Capture/reduce all eight W4 order-enabled endpoints and replay twins.
5. Open and run the four W8 holdout endpoints only if W4 passes.
6. Run affected workspace tests, deterministic report regeneration, update the
   durable handoff/status, and commit a clean result.
