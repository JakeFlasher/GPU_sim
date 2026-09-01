# SM86 asynchronous-copy cross-warp queue-overlap milestone

## Goal and completion bar

Test whether the simulator's current SM-shared asynchronous-copy ordering
transfers when an otherwise identical target group starts behind outstanding
groups from sibling warps. Compare an exact-work `queued` arm with a `drained`
arm, keeping source vectors, destination vectors, route, LDGSTS/group counts,
CTA resources, and target instructions fixed. Cache-all is the characterized
positive control; repeated L2-hit bypass is the unresolved target.

Completion is either:

- zero-fit identification and untouched W8 holdout transfer of the current
  source/destination ordering; or
- a reproducible control, identification, or holdout no-go at the first frozen
  gate.

The milestone may retain a timing-neutral cross-warp overlap observation. It
may not restore a rejected issue queue, fit latency, change an architecture
card, or remove `async_copy_shared_visibility`. The evidence root is
`results/captures/sm86_async_cross_warp_queue_overlap_20260831_v1`.
JADE/HETR, CPU-GPU co-simulation, `sm_120`, and unrelated candidates remain
out of scope.

## Independent mechanism

The retained 2026-08-17 async-cross-warp campaign already consumed a
homogeneous cache-all W1--W8 sweep and selected the production two-wavefront/
three-cycle destination service. Homogeneous L1/L2/DRAM width and
async-plus-CTA-barrier cells are also consumed. This assay changes neither
width nor destination work inside a comparison. Its only discriminator is
whether background groups remain outstanding when the target begins.

One fixed 256-thread CTA allocates a 32 KiB destination arena plus bounded
shared timestamp/result state (33,104 static bytes total). Identification uses
four active warps; inactive warps still participate in untimed CTA barriers.
The highest active warp is the measured target and all lower active warps are
backgrounds. Every active warp owns a private repeated 512-byte source and a
disjoint 4 KiB shared destination slot. Each group issues eight contiguous
full-warp 16-byte copies, commits once, and executes `wait_group 0`.

- `queued`: background warps issue and commit, then rendezvous. After release,
  backgrounds wait while the target clocks its own issue/commit and wait.
- `drained`: backgrounds issue, commit, and wait before the same rendezvous.
  After release, the target executes the identical clocked group.

A final rendezvous closes the epoch. Shared timestamps count how many
background `[commit, wait-end]` intervals overlap the target
`[issue-begin, wait-end]`; those checks occur outside the target timing
interval. Direct mode executes both arms in one launch with alternating AB/BA
order. Endpoint mode executes one arm for NCU, NVBit, and replay. Both call the
same route/group body.

Routes are cache-all `.ca` (`l1`) and bypass `.cg` over a repeated prewarmed
footprint (`l2`). Raw device DRAM is non-gating on this display-active host.

## Timing-neutral schema-20 observation

Observe the already selected SMSP picks before any issue mutation. When at
least two effective `GlobalAsyncCopy` records issue together, record:

- coissue cycles and coissued copy count;
- cache-all and bypass coissued copies;
- mixed-policy coissue cycles;
- structural single-acceptor excess `sum(n-1)`; and
- maximum coissue width.

For every effective async-copy pick, also observe other warps' committed
groups present before the transaction:

- copies issued with at least one other-warp group committed;
- summed and maximum other-warp committed-group population.

This observation changes no pick, readiness, cache request, group,
destination service, event time, or typed tail. Allocate schema 20 only when a
nonzero observation exists. A focused test compares four serialized
same-SMSP copies with one four-SMSP collision, checks mixed `.ca`/`.cg`
accounting, transactional overflow, and unchanged timing.

## Prospective structural gates

- Pin device 0 to RTX 3090 Ti UUID
  `GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, compute capability 8.6. Record
  driver, CUDA, NCU, NVBit runtime, binary, source, Makefile, cards, traces,
  control sidecar, simulator, and output identities.
- Queued/drained arms share one target function, fixed CTA/shared/register
  resources, source and destination vectors, route, active-warp set, copy and
  group counts, and target clock/control sequence. Only background wait order
  around the rendezvous changes.
- Target result stores and overlap reductions occur after measured clocks.
  Every epoch is token-correct and internally additive.
- For smoke groups `G=64`, copies/group `D=8`, and active warps `W=4`, each
  endpoint has exactly:
  - `G*D*W = 2,048` LDGSTS warp instructions;
  - `16*G*D*W = 32,768` source sectors; and
  - `4*G*D*W = 8,192` shared LDGSTS wavefronts.
- Queued and drained NCU work/source metrics are exact and equal within each
  route. The L1 route has at least a 95% L1-hit ratio. The L2 route has at least
  95% L1 misses and 95% LTS hits after prewarm, with no material paired LTS
  drift. NCU is a validation target, never a simulator input.
- Each NVBit endpoint has one complete CTA, exact instruction reporting, four
  active LDGSTS warps, and matching per-warp masks/source/destination vectors.
- Release `sim-run` consumes each endpoint twice byte-identically. Cache-all
  retains no async visibility tail; bypass retains it. Schema-20 observations
  must be internally consistent with the queued/drained ordering.
- The live manipulation succeeds only when all background commits precede the
  target issue in queued, at least two of three background intervals overlap
  the target in at least 95% of epochs, and drained has zero overlapping
  backgrounds.

## Smoke and identification

Run a 64-group smoke for route `{l1,l2}` and arm `{queued,drained}` through
static inspection, correctness, NCU, NVBit, and release replay before full
labels. The full identification contains ten alternating AB/BA paired direct
launches at 8,192 groups for each route. A process repeat is the independent
unit.

For repeat `j`, define target-warp contrasts without combining phases:

```text
dI[j,route] = (issue_queued  - issue_drained)  / groups
dW[j,route] = (wait_queued   - wait_drained)   / groups
```

Before live wait labels, derive the zero-fit simulator contrasts from endpoint
replays. The issue prediction is zero because the current model has no LDGSTS
admission resource. The wait prediction is the queued-minus-drained modeled
launch frontier normalized by groups; report source, local destination, and
SM destination-service diagnostics separately.

Read correctness/manipulation first, issue labels second, cache-all wait labels
third, and bypass wait labels last. Identification requires:

1. For each route, at least 8/10 live issue deltas have the simulator sign
   (zero admits either sign within tolerance), with median absolute issue error
   no greater than four cycles/group.
2. The cache-all simulator wait prediction is material (`abs(prediction)>=16`),
   at least 8/10 live wait deltas have its sign, and median absolute wait error
   is no greater than `max(16 cycles, 20% of abs(prediction))`.
3. The L2 target has at least 8/10 live wait deltas with the simulator sign and
   median absolute wait error no greater than the same frozen tolerance.
4. Schema-20 coissue and background-committed observations agree with trace
   work and the manipulation; never collapse them with wait cycles into one
   score.

Directional agreement alone does not pass absolute transfer. A failure is a
useful mechanism no-go, not permission to fit the residual.

## Untouched W8 holdout

Do not capture W8 timing, NCU, NVBit, or replay before every identification
gate passes. Holdout uses the same 256-thread function, target warp 7, seven
background warps, and unchanged group geometry. It requires:

- at least five of seven queued background intervals overlapping the target in
  at least 95% of epochs, and zero drained overlap;
- exact doubled work and complete trace/replay evidence;
- the unchanged issue and wait transfer tolerances for both routes; and
- schema-20 counts consistent with eight active warps.

If identification fails, the W8 root remains absent and unread.

## Minimum executable evidence

1. Build and statically inspect the dedicated binary.
2. Run the complete 64-group W4 smoke through direct, NCU, NVBit, and
   deterministic replay.
3. Retain schema-20 timing-neutral overlap observation and prove no-collision
   predecessor output remains byte-compatible.
4. If smoke passes, run and reduce ten identification pairs per route under the
   frozen rules.
5. Capture W8 only if identification authorizes it.
6. Run focused Rust checks, affected workspace library tests, deterministic
   report regeneration, update the durable status/handoff, and commit a clean
   result.
