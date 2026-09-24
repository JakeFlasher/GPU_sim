# Historical transition soundness audit — 2026-09-17

> **Archive-only notice (2026-09-24).** This is a dated technical record, not
> an active audit process or admission gate. Its prior-art checkouts,
> manifests, hashes, and receipts are retained as inert data. Current work is
> limited to software trace-driven architectural simulation, one declared NCU
> comparison boundary, and simulator wall-time/design evidence; do not revive
> RTL/Verilator/FPGA or unavailable-AMD requirements from this record.

This bounded review checked the typed transformer against a fresh shallow
MGPUSim checkout and its primary paper.

The checkout is `/home/jakeshea/mgpusim_transition_audit_20260917`, origin
`https://github.com/sarchlab/mgpusim.git`, at shallow commit
`d8f4c0a7d7b75498a91525cde37f3c5d2d6b70b6` (2026-08-29). The Akita timing
dependency is `github.com/sarchlab/akita/v5@v5.0.0-beta.10` in the Go module
cache.

The bounded Accel-Sim source used for the companion review is
`/data1/tmp/gpu_simulator_research/accel-sim-framework` at shallow commit
`d930ad6d02c09bb56867132583735aba0389cff4`; its `accel-sim.cc` loop performs a
detailed cycle and deadlock check for each active iteration.

Relevant source observations:

* `timing/eventqueue.go` assigns a private insertion sequence and orders events
  by timestamp plus sequence. `timing/serialengine.go` additionally gives the
  primary queue precedence over the secondary queue at equal timestamps.
* `modeling/eventdriven.go` deduplicates wakeups by the earliest pending time.
* `amd/timing/cu/computeunit.go` executes a fixed phase order within each tick
  and schedules retries at a future tick. A timestamp alone therefore does not
  identify a complete transition.
* `timing/parallelengine.go` gives each worker its own sequence space and runs
  same-time events concurrently under an independence assumption. Shared-state
  events therefore require an explicit dependency/phase closure before they
  can be summarized.

Sun et al., “MGPUSim: Enabling Multi-GPU Performance Modeling and Optimization,”
ISCA 2019, provides the primary design rationale:
[author PDF](https://people.bu.edu/joshi/files/mgpusim-isca2019.pdf). The paper
describes chronological event ordering, explicit same-time independence
assumptions, 300-cycle DRAM fast-forwarding, and notification-based retry
handling.

The consequence for this project is strict. A native summary proof must carry
event class/phase, sequence reservations (including hidden positions reserved
by an `L2AccessBatch`), retry/dependency effects, and complete mutable state.
The checker must also receive the independently observed entry boundary and
external frontier; accepting a proof's embedded entry would allow a forged
zero-length identity to certify an unrelated run. `check_proof` now requires
both independent values and rejects those substitutions.

The same rule applies to emitted transitions. The checker accepts an
independently observed transition slice and compares it with the proof before
certification; equal endpoints cannot establish omitted or reordered same-time
events. Shared-L2, queue, and power mutations therefore remain reified until a
future transformer supplies explicit phase and dependency closure.
The queue boundary now exposes each event's actual `sequence_span` and the next
unused sequence position, so a batch cannot hide logical insertion positions.
The payload schema also uses explicit enums for memory origin, L2 source,
operation, and response kind rather than formatting native values with
`Debug`.
The typed identity transformer derives its output from the entry and certifies
only a zero-length identity; non-empty intervals reify. The causal-frontier
analyzer independently compares intermediate issue/completion observations and
returns `StateMismatch` or `ObservationMismatch` on any divergence.

The follow-up SM86 audit found that CTA residency and the active SMSP issue
roster are different state dimensions: a warp can retire from the issue roster
while its CTA still reserves the full warp residency until block completion.
Typed core validation now checks the reservation against active block state and
checks unfinished warp states against the current roster separately. A rebuilt
flash-attention replay produced 5,664 identical event/dense typed boundaries,
confirming the correction does not change native scheduling or event skipping.

A native-state follow-up checked lazy memory/scalar pipeline pruning: releases
are removed at the top of each `SmCore::advance`, and native probes observed no
release strictly before the typed core cycle. The L1 MSHR follow-up was
different: access-local cacheable-LDGSTS floors intentionally admit more live
fills than the base configuration, so typed validation was relaxed for that
specific distinction and covered by an async-copy pressure test.

The 2026-09-18 adversarial pass found two additional state-boundary hazards.
An outstanding L1 response owner could be serialized without a matching live
allocation and the shared owner list was not joined to each SM's response
roster. Complete boundaries now reject orphan owners and incomplete boundaries
carry the missing-generation/evicted-allocation reason; completeness also
requires the canonical owner sets to match. Resident virtual lines are checked
against every registered SM86 slice-hash formula and serialized set/slice.

Direct Warp/AsyncCopy replies retain a response-arena index after warp
retirement. The typed projection now recovers the line and one-sector mask from
the retained sector arena, while preserving the dynamic arena index so aliased
sectors remain distinct. An async-copy sidecar may legitimately omit cohort
details while retaining aggregate topology; the typed validator accepts that
native form and rejects one-sided or inconsistent sidecars. The release canary
still produced 13 equal event/dense typed boundaries (schema 11), and the
retained eight-small/seven-large replay sweep remained equal for every row.

A follow-up L1 audit found that a forged typed cache could carry sector bits or
allocation generations for a line absent from both the resident LRU roster and
pending line allocations. The validator now rejects those orphan entries,
checks the miss-counter decomposition and lookup-batch frontier, and retains
the legal access-local MSHR-floor exception. Native stale-fill,
write-supersession, small/large replay, and full reference-workspace checks
remain green after the change.

The same boundary audit found that `write_observer_present` exposed only a
boolean while omitting the observer's tag-window state. Exact typed contracts
now reify when that observer is active instead of certifying from the flag
alone.

The subsequent front-end audit found forged cache-set placement, duplicate LRU
timestamps, demanded stream tails, non-monotonic stream ready times, divergent
L0 geometry, invalid prefetch priority, stale request/reply timers, and
unsupported SM86 front-end widths. Typed validation now rejects these states;
front-end counter conservation now ties L0/L1 access sequences to fetch and
hit/miss counters; the release canary and front-end-heavy embedding replay
remain event/dense identical afterward.

The current SM86 canary also exposes one stable optional-surface profile at
all 13 boundaries: L1, instruction-front-end, and memory-pipeline state are
present; the scalar pipeline is intentionally unconfigured; and async-copy
slots are present. This confirms that the remaining `Option` fields reflect
model configuration on the native path rather than boundary-to-boundary drift.

A power audit found that a sampled boundary could advance its domain clock
past `next_sample` while omitting all elapsed samples. Validation now requires
the next sample frontier to lie strictly beyond the common domain clock and
preserves the exact period-aligned sample cadence, so skipped or off-phase
samples cannot certify. The native sample schedule remains unchanged.

The pipeline follow-up also checks the shared acceptance frontier against the
latest live memory-queue release, preventing a forged queue from advancing with
an earlier service frontier.

Shared-L2 validation also requires the root and geometry architecture
identities to agree before any address or ownership proof can certify.
Replacement-policy tokens are constrained to the model's registered LRU,
SRRIP, and BIP implementations as well.
Observer lookup hits are bounded by observer lookup requests as well as the
model's primary lookup counters.
L2 miss and write-tag counters are also checked against their component
decompositions before an exact boundary can certify.

Core validation additionally binds the next-block admission frontier to the
serialized block-state roster, preventing a forged admission count from being
treated as a resumable scheduler state.

Event-boundary validation now rejects power-domain timestamps that do not match
the event timestamp, preserving the causal relationship between snapshots and
queue events. Core clocks intentionally may run ahead of queue timestamps in
the dense reference lane, so that relation remains model-dependent; event
targets are nevertheless checked against the available core snapshot domain.
Core snapshots are also required to carry one canonical, strictly ordered SM
identity list before cross-core ownership joins are considered.

The scheduler audit found that a bounded idle return left `SmCore::t` at the
previous local timestamp. A later wake could therefore repeat front-end work
at an already-consumed cycle, and the dense reference could omit idle cycles
below an external response horizon. Bounded returns now park the clock,
dense mode steps each locally idle cycle below the horizon, and a regression
covers both repeated front-end wakeups and bounded response waits. The same
park transition now prunes expired memory/scalar pipeline releases and
completes fixed-clock cache entries; a typed flash-attention replay exposed and
validated this requirement.

The L2 projection review found that unsigned sorting of signed virtual line
IDs could reject otherwise valid negative-address traces, while sorting native
way vectors would change replacement tie identity. Allocation serialization now
uses signed canonical key order and leaves native way order untouched. A
projection-level ABA regression also confirms that an evicted fill token cannot
change the newer same-line generation or its typed ownership state.

The overflow audit found one release-only diagnostic masking path in the
async-copy cohort projection: a reversed first/last source-response pair was
guarded only by `debug_assert!` and saturating subtraction. The projection now
returns a typed invalid-order error and the malformed-state regression passes
in both default and reference builds.

The trace-storage review then replaced changed-length array replacement with an
exact schema-3 splice operation (common prefix/suffix, ordered edits, strict
bounds) while retaining schema-1/full decoding. A bounded 512-record
convolution prefix shrank from 1.17 GB to 650 MB, and the full retained
flash-attention trace reconstructed 5,664 equal event/dense boundaries. This is
storage overhead reduction only; it does not omit modeled transitions or alter
cycle timestamps.

The follow-up Accel-Sim audit rechecked the pinned integrated GPGPU-Sim source
at `/home/jakeshea/GPU_sims_notes/upstream_research_20260901/accel-sim-gpgpu-sim_distribution`,
commit `91880c53383d5a6a6742bfb1be2c5f34e39c7871`. Its
`src/gpgpu-sim/gpu-sim.cc` advances the minimum tied clock domain and then
applies explicit ICNT/L2/DRAM/core phases; `shader.cc` orders writeback,
execute, operand-read, issue, decode, and fetch. The Accel-Sim ISCA 2020 paper
also describes simulation-gating as skipping inactive component code while
still ticking each cycle. Our phase-0 typed queue is therefore an explicit
single-serialized-phase scope guard, not evidence of hardware phase
equivalence; future same-time memory-domain classes must carry phase/dependency
closure or reify.

That review also found a typed-boundary ordering hole: the validator compared a
pending event to the current event using `(time, phase, sequence)`, while the
native queue orders `(time, sequence)`. A forged nonzero phase could therefore
make a lower sequence appear later and pass structural validation. Schema 11 is
currently single-phase, so boundary validation now rejects any nonzero event or
pending phase; a future multi-phase engine must replace this guard with an
explicit phase/dependency contract. The adversarial regression passes.

The retired-response follow-up found that signed `warp_slot` identities were
validated for core rosters but not for shared-L2 response-owner keys. A forged
owner with `warp_slot=-1` could otherwise pass the ownership join when its line
and allocation generation were valid. Shared-L2 and core owner validation now
reject negative response slots; the regression preserves native retired-warp
behavior. The same serialized identity also rejects a duplicate dynamic
`(SM, warp, slot, instruction, sector)` owner even when the forged records use
different response kinds, because the native private response-owner token is
not available to disambiguate that collision.

The cycle-frontier follow-up found that standalone L1 validation did not relate
timed outstanding fills to the owning SM's current clock. A forged fill due
strictly before that clock could therefore survive typed structural validation
and be treated as resumable state. `L1BoundaryState::validate_at_cycle` now
retains cycle-independent validation for standalone snapshots but rejects this
past-due case when called from `TypedCoreBoundaryState`. The regression and a
rebuilt SM86 address-carry replay pass; event and dense outputs remain byte
identical.

The same pass found that block validation bounded unfinished warps by the
admitted roster but not by the block's live-warp residency count. A forged
active block could therefore understate `resident_warps` while retaining an
unfinished warp. Typed core validation now rejects `unfinished > warp_count`,
and the rebuilt canary still reports identical issue and terminal frontiers in
event and dense modes.

Finally, response-owner due cycles were checked for presence but not against
the serialized boundary timestamp. Root validation now rejects a due cycle
strictly before the current event while accepting same-cycle queued responses
and explicitly unknown due times. The bounded typed canary reconstructed 13
equal schema-11 boundaries in both modes after this guard.

The residency follow-up also checked physical warp-slot ownership across
active CTAs. Native admission retains every slot in a block until that block's
last unfinished warp retires, so a retired warp can remain outside the issue
roster while its slot is still reserved. Typed validation now rejects duplicate
slots across unfinished blocks but continues to allow reuse across fully
retired blocks. The SM86 typed canary remains equal after this check.

The block-arena follow-up now validates each `first_warp..first_warp+
admitted_warps` range against the serialized warp vector, block identity, and
declared slot roster. It also requires contiguous ranges across block history;
this protects barrier-release indexing from a forged range that points into a
different CTA. Retired-block slot reuse remains supported.

The hold-state follow-up binds every held warp to the matching SMSP roster, not
just the SM-wide resident set. A cross-SMSP hold would otherwise change CGGTY
selection after resume while passing global residency checks. The typed
regression and event/dense canary remain equal.

The CTA-barrier follow-up now binds `block.waiting` to the per-warp
`at_block_barrier` flags, requires dispatch/arrival timestamps for each waiter,
rejects stale timestamps on non-waiters, and checks the block's first-dispatch
and first-arrival frontiers against the observed minima. This protects the
barrier release and re-arm path from aggregate-only forged state.

The block-range audit also requires the final block range to end exactly at the
serialized warp-arena length; unclaimed tail warp states cannot shift native
arena indices on resume. The typed canary remains equal after this check.

The shared-L2 temporal audit also rejects a pending sector whose known ready
time is before the boundary event timestamp, while preserving same-cycle
pending fills for queue-order replay.

Typed queue payloads now bind single and batched L2 request issue times, plus
write-ack visibility times, to their event timestamps. This mirrors native
event-loop admission checks and prevents a structurally valid queue from
resuming a request at the wrong cycle.
Pending event targets are also checked against the available core roster.

The write-ack projection also enforces monotonic issue, acceptance, service,
and visibility timestamps before a queue boundary can certify.
Acknowledgement operations and request witnesses are also constrained to the
native Store/Reduction and four-SMSP index/count domains.

Single and batched typed L2 requests now enforce native sector-mask, Local-mask,
source/operation/reply, and admission-witness contracts before replay can
resume them. Within a batch, admission witnesses remain contiguous within an
active group; after a group reaches its declared request count, the native
observer permits the next index-zero group in the same timestamped batch.
Constant-load and deep-constant markers are restricted to the native L1-fill
read path.
Sector-fill reply identities are restricted to single-sector masks, while line
fill identities retain their native multi-sector form.

Fill identities reject empty/out-of-range sector masks, and response identities
reject the unsupported `None` response kind.
Memory-response events also require a returning reply identity; replyless fills
remain separate `L2Fill` events as in the native event loop.
Responses carrying fill tokens accept the native L1-fill, direct-Warp, and
AsyncCopy reply kinds; the `None` sentinel remains invalid.

The follow-up native review found both rules needed to match the executable
path. `schedule_emitted_requests` batches consecutive replyless Store/Reduction
requests by timestamp without requiring a shared admission identity, and the
observer closes one complete group before accepting the next. Cold L1-bypass
reads also attach a new L2 fill to direct Warp/AsyncCopy responses. The typed
validator now mirrors those transitions while retaining per-group sequence,
operation, and timestamp checks. The native multi-group regression, cold
AsyncCopy reference replay, 58 semantics tests, the focused 431 reference SM
tests, the full-workspace 461 reference SM tests, 135 memory tests, and the 13-boundary SM86
event/dense canary all pass.

The sectored-MSHR follow-up found a cycle-underestimation path in the SM/L1
projection. A line-fold map scoped to an entire actor advance could merge a
sector issued at a later local timestamp into an earlier request, and the
cross-batch path attached that late sector's waiter to the parent line fill.
Pinned Accel-Sim/GPGPU-Sim source splits sectored requests into 32-byte child
fetches and releases merged state per returned child. The implementation now
keys folds by `(line, issue_time)`, reserves a generation-bound L1 sector fill
for late children, emits an `L1SectorFill` response, and retains an early child
completion until the parent line allocation becomes resident. Response-owner
projection uses the exact mask carried by each response rather than the
aggregate per-line traffic mask. Default and reference system suites, 187
reference issue tests, 135 memory tests, and the SM86 event/dense canary pass;
the causal late-sector fixture is 847 cycles versus 458 before the fix.
Retry draining is also gated on an actual independent MSHR release, so a child
completion that shares its parent line slot cannot create a speculative retry
or extra stall event.

The adversarial lifecycle review then found generation-ABA accounting: an old
sector child surviving same-tag eviction was being treated as a child of the
new parent line because occupancy and release checks keyed only on the line.
Those checks now require matching `(line, generation)` identity. The finite
MSHR regression demonstrates that the zombie child still consumes capacity,
while completion of the new parent releases its own slot; the 135-test memory
suite and both SM profiles pass afterward.
The reservation's fill-sequence allocation is included in the issue-cycle
shadow preflight, keyed by the same `(line, issue_time)` fold boundary, so a
near-`u64` sequence failure cannot partially publish the late merge.
The typed L1 occupancy diagnostic uses the same generation-aware parent/child
sharing rule while retaining the access-local MSHR floor as a non-certifying
capability detail.

The subsequent constant-L1 retry audit found a separate transactional edge:
when finite MSHR occupancy is already full, one constant-mode access may pass
through multiple retry timestamps. The ordinary fast preflight budgeted only
one possible stall, so a later counter or timestamp error could apply an
earlier due fill before returning. Constant-mode wrappers now force the
existing shadow-cache transaction whenever the effective MSHR is full,
including access-local LDGSTS floors. Two multi-retry rollback regressions,
the 137-test memory suite, the 187 reference issue tests, and both workspace
profiles pass after the fix.

The follow-up typed-L2 audit found that `ownership_complete` did not cover a
resident event-mode way whose generation was absent: the way was omitted from
the allocation set, so an exact transition could not distinguish an old fill
from a later same-line allocation. Complete event-mode ways and all pending
ways now fail closed without a generation identity, while immediate-mode ways
remain generation-free. The validator also rejects negative L2 clocks, which
the native checked transition cannot produce. The focused regressions and all
62 semantics tests pass.

The core-state follow-up found that serialized per-warp instruction buffers
were not validated at all. Complete boundaries now reject flat-mode buffers,
done-warps with buffered work, capacity overflow, cursor/order drift, and
entries before the next instruction. Reference SM tests and the retained
SM86 canary remain green.

The memory-pipeline audit then found a post-drain frontier gap: once all live
queue entries had retired, `shared_accept_ready` was no longer checked. The
typed projection now binds it to `terminal_shared_accept_cycle` plus the
configured interval, preserving the next-issue timing after the queue empties.

Root boundary validation now preserves independent local and global frontiers:
an SM may be run ahead to its own response horizon while the queue records a
different global event timestamp. Local core invariants still validate the
run-ahead state, but the root no longer imposes an invalid cross-frontier
ordering.

A bounded typed-trace canary exercises this relation in both event and dense
modes: 13 reconstructed boundaries per lane match exactly, including the
run-ahead core clock.

The retained embedding workload was then checked with a bounded 6,000-record
typed trace: 2,448 event calls reconstructed 5,016 records, and both outputs
and boundary streams matched between event and dense modes.

The same probe exposed the prior invalid core/global timestamp comparison; it
was removed so typed validation matches the native actor contract while
retaining all local pipeline, ownership, and queue checks.

The L2 validator now rejects an incomplete ownership flag with an empty
unknown-state reason list, preserving explicit diagnostic provenance for
incomplete projections.

The same reason list is required to be canonically sorted, matching the
native join's sort/deduplicate step and preventing metadata-only replay drift.

Barrier-state validation also rejects a nonzero CTA re-arm frontier before the
first barrier generation, preventing a forged initial barrier delay.

Async-copy validation also binds committed-group timestamps to the owning core
cycle while retaining future completion timestamps for in-flight sources.

The core projection now checks the issued counter against the sum of all
serialized warp instruction cursors.

The typed-L1 follow-up closes two structural gaps: an empty architecture
identity is rejected, and a line cannot be represented simultaneously as a
resident LRU entry and an in-flight line allocation. Both states are
unreachable in the native cache transition and are now rejected before any
typed transform can reason about them.

The async-copy follow-up binds retained bypass observations to the aggregate
state of their live group: pending sources cannot exceed the group count,
source completion cannot exceed the group frontier, and delivery work cannot
exceed the checked group sum. The new regression and SM86 event/dense canary
preserve identical cycle outputs.

The memory-pipeline follow-up closes diagnostic-only drift: maximum delay
fields are bounded by their totals, admitted work records at least one queue
occupancy sample, and an empty pipeline cannot carry nonzero frontiers or
delay counters. Native timing and event/dense outputs remain unchanged.

The optional scalar-pipeline projection now rejects the analogous forged
history: impossible first service frontiers, occupancy exceeding admissions,
and nonzero diagnostics with no admissions. Native SM86 event/dense timing is
unchanged.

The event-boundary projection was also tightened for throughput: it sorts
heap-entry references before constructing typed payloads, avoiding repeated
moves of large request batches while retaining canonical callback and output
order.

Typed L2 request owners now reject the sentinel `none` response kind even
when ownership is incomplete. Incomplete joins may remain unknown, but they
cannot carry an owner identity that the native response path never emits.

Complete pending-sector ownership now rejects consumers whose request state is
already completed, stale, or non-returning. A pending fill can only wait on an
outstanding response owner; incomplete joins remain diagnostic until that
state is known.

The contract classifier now performs the same power and core structural
validation as the root boundary before returning completeness. A direct
classifier call cannot promote malformed core state to `Complete`.

The retained current-tree replay sweep was rerun across all eight small and
seven large SM86 command directories. Every event/dense JSON output matched;
the accelerated lane retained exact cycle frontiers while reducing local idle
work, with no typed boundary trace enabled.

Both complete workspace profiles also pass at this revision: 455 default SM
tests, 461 reference SM tests, and 65 semantics tests, with only the known
machine-local capture tests ignored.

The focused Clippy audit cleared the actionable warnings in the changed
typed-request and generation-aware L1 paths; remaining warnings are confined
to existing test unwraps and unrelated utility style code.

The SM memory-service candidate observation loop was also converted to direct
bounded iteration, removing needless index traffic without changing candidate
ordering or timing; reference tests and the canary remain identical.

The async-copy sector-response loop received the same bounded-slice treatment;
absolute response identities and ordering are unchanged.

The complete retained small/large replay sweep was rerun afterward and again
reported equal event/dense outputs for all 15 workloads.

The same 15-workload sweep was repeated after canonical unknown-state ordering
was added, with equal event/dense outputs in every row.

A DRAM aggregation regression now pins the distinction between additive
traffic and maximum terminal-residency gauges across launch composition.

Admission-observer control flow was simplified only after preserving its
short-circuit and error propagation order; the reference suite and canary
remain cycle-identical.

The full retained replay sweep was rerun after the change and again matched
event and dense outputs for all 15 workloads.

The ownership-unknown diagnostic path now formats canonical line keys in one
step; its output text and ordering remain unchanged.

The shared-L2 counter helper uses direct error construction for its unbound
architecture case, preserving the same typed failure path without a closure.

The complete default and reference workspace profiles were rerun after the
same change and remained green, including 65 semantics tests and 455/461 SM
tests.
