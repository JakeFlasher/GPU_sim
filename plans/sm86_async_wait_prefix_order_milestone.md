# SM86 asynchronous-copy wait-prefix ordering milestone

## Goal and completion bar

Validate the simulator's existing Ampere `cp.async` committed-group FIFO and
`DEPBAR.LE SB0, N` prefix-readiness semantics against a fresh live-GPU assay.
The assay must distinguish the oldest committed group from the retained newest
group without fitting a latency, reusing a consumed source-line criterion, or
changing production timing. Completion is either:

- a zero-fit identification and untouched-holdout transfer of the existing
  FIFO mechanism, with a timing-neutral per-wait diagnostic retained; or
- a reproducible mechanism-level no-go that identifies the first failed gate
  while leaving production timing and `async_copy_shared_visibility` intact.

The evidence root is
`results/captures/sm86_async_wait_prefix_order_20260831_v1`. JADE/HETR,
CPU-GPU co-simulation, `sm_120`, and unrelated simulator candidates are out of
scope.

## Prospective mechanism

Each measured epoch commits disjoint compact one-copy groups with fixed source
and destination identities. Source A is always issued as cache-all `.ca` and
source B is always issued as bypass `.cg`; only their committed-group age is
permuted:

- `l2_old`: `.cg` B is committed first, then `.ca` A.
- `l1_old`: `.ca` A is committed first, then `.cg` B.

The two-group identification executes `cp.async.wait_group 1`, consumes only
the oldest group's disjoint shared destination, then executes
`cp.async.wait_group 0` and consumes the remaining destination. Five ordered
clock frontiers keep issue/commit, prefix wait, oldest consume, remainder wait,
and total time separate. All global result stores occur after the measured
interval.

The untouched holdout uses three groups and `wait_group 2`: one `.cg` group is
oldest in `l2_old` and newest in `l1_old`, while the other two groups are
identical `.ca` work. It is not captured or decoded unless identification
passes.

## Frozen structural gates

- The host is device 0, RTX 3090 Ti UUID
  `GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, compute capability 8.6. Record
  driver, CUDA, NCU, binary, source, Makefile, architecture-card, trace, and
  simulator identities.
- Each identification epoch has exactly two effective LDGSTS instructions,
  two nonempty commits, one `DEPBAR.LE SB0, 0x1`, and one
  `DEPBAR.LE SB0, 0x0`. The holdout changes those exact values to three LDGSTS
  instructions, three commits, `SB0, 0x2`, then `SB0, 0x0`.
- Each copy has 32 active 16-byte lane transfers, 32 source sectors, compact
  lane geometry, eight shared LDGSTS wavefronts, and a disjoint 4 KiB shared
  destination slot. The two variants have identical aggregate source vectors,
  destination vectors, `.ca`/`.cg` counts, registers, shared memory, and static
  control shape.
- Direct output is finite, internally additive, token-correct, and records all
  five frontiers. NVBit preserves the exact per-lane source/destination and
  ordered `.ca`/`.cg` identities. NCU is a separate execution and may validate
  aggregate work only; no NCU value enters simulator inputs.
- Raw device DRAM remains non-gating on this display-active host. The `.cg`
  source population must close at attributable L1TEX/LTS demand with positive
  L2 hits and no material paired population drift.
- Release `sim-run` consumes both traces twice with byte-identical outputs.
  Existing issue, traffic, and typed-tail invariants must hold.

## Frozen label order and identification rule

Run a 64-epoch smoke through static inspection, direct correctness, NCU,
NVBit, and release replay before any full timing matrix. For the full
identification, collect ten alternating-order paired repeats at 8,192 epochs.
Read structural labels first, then issue/consume spans, then the two wait spans.

For paired deltas `l2_old - l1_old`, identification requires all of:

- issue/commit median absolute delta no greater than 1 cycle/epoch;
- oldest-consume median absolute delta no greater than 1 cycle/epoch;
- prefix-wait median delta at least +64 cycles/epoch with at least 8/10
  positive repeats;
- remainder-wait median delta at most -64 cycles/epoch with at least 8/10
  negative repeats; and
- in at least 8/10 pairs, the absolute total delta is smaller than both the
  positive prefix delta and the magnitude of the negative remainder delta.

The simulator passes identification only if its existing zero-fit FIFO model
reproduces both phase signs and cancellation, and each phase-delta error is at
most `max(16 cycles, 15% of the live phase magnitude)`. Direction alone does
not authorize a timing change or tail removal.

Only after every identification gate passes may the three-group
`wait_group 2` holdout be captured and read. Apply the same sign, magnitude,
cancellation, traffic, correctness, and zero-fit transfer rules without
retuning.

## Simulator change boundary

Add only the smallest timing-neutral observation needed to report partial and
draining waits separately: wait count, immediate sum, committed groups before,
consumed prefix groups, groups remaining, and exposure cycles. Emit it only
for programs containing a nonzero wait-group immediate so predecessor
wait-zero JSON remains byte-compatible. This observation may advance readiness
schema 19; it must never feed readiness or event time.

Do not add a bypass latency, destination coefficient, queue width, or fitted
constant. Do not clear `async_copy_shared_visibility` in this milestone.

## Minimum executable evidence

1. Build the dedicated binary and inspect its SASS/resource shape.
2. Run both 64-epoch variants directly and verify the ordered clocks/token.
3. Capture both variants with NVBit and replay each twice with release
   `sim-run`.
4. Run bounded NCU work/source attribution for the smoke pair.
5. If smoke passes, run the ten-pair identification and reduce it under the
   frozen rule. Run the holdout only if identification authorizes it.
6. Run focused Rust checks plus the affected workspace library tests, update
   the durable report/status, and commit a clean reproducible result.
