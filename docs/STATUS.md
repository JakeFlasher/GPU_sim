# Current status

Updated 2026-09-23.

The post-slice adversarial review is recorded in
[adversarial_code_review_20260923.md](adversarial_code_review_20260923.md).
It closed the material custom-SIMT, semantic-roundtrip, ELF-adapter, and
overflow seams without changing the SM86 evidence boundary; the full
workspace test suite is green, while broad all-target clippy's historical
test/style findings remain explicitly bounded rather than hidden.

### General-architecture continuation (2026-09-23)

The user authorized the
[general GPU simulator roadmap](../plans/cross_architecture_gpu_simulator_research_plan_20260923.md).
SM86 remains the hardware-validation anchor; a separately scoped custom-model
specification and executable pilot are now active. This extends the former
SM86-only development scope without starting MI300X hardware-dependent work.
Historical comparisons below remain dated observations, not cross-vendor or
held-out elapsed-cycle accuracy claims. The current binding-cycle diagnostic
is model-relative, not a validated silicon duration.

The first custom-architecture slice is runnable: `sim-arch::spec` validates a
bounded opaque architecture/ISA definition, `arch_profiles/specs/simt_u32_minimal.json`
is a hypothetical eight-lane example, and `architecture_spec` prints the
validated representation. This path is not consumed by production SM86 replay
yet. After refreshing the card-schema-induced canary seal, the full Rust
workspace test set passes; machine-local trace/sidecar tests remain explicitly
ignored.

The functional companion slice is now executable as well: `sim-arch::simt`
loads a matching `simt_u32_v1` program, executes one bounded SIMT group, and
emits per-lane registers plus per-lane stores. Unsupported ISAs and identity
mismatches fail closed. It is a functional feasibility oracle, not a timing or
silicon model. The retained four-instruction probe emits eight lane-specific
stores; the negative identity probe exits nonzero. Workspace tests remain green.

The Vortex comparison is now stronger than a hand-authored vector check:
native regression `vecadd` passed through SimX and rtlsim with deterministic
112-instruction results (455 vs 397 cycles across three repeats), and a narrow
real-ELF `lw/lw/add/sw` adapter passed the independent custom differential.
The cycle mismatch is retained as a model-parity finding, not silicon truth.
The same executor's `async_store`/`wait_group` probe preserves eight pending
lane stores and reaches the declared memory completion cycle before the wait
returns; this is a hypothetical resource timing result only.
The retained integer Vortex ELF now also crosses the Rust shared semantic-event
path through `tools/vortex_elf_shared_semantic_probe.py`, producing six
validated events. Its synthetic address binding and narrow pattern scope are
recorded explicitly; general RISC-V pointer/dataflow decoding remains open.

The end to end SM86 path is operational: NVBit traces and control sidecars are
read by `sim-trace`, replayed by the SM and memory models, and emitted through
`sim-run`. The current working tree contains active scheduler, cache, memory
completion, trace, and result-schema changes; those sources remain the live
implementation.

### Active scope (2026-09-23)

MI300X/gfx942 timing and all hardware-dependent follow-up work are explicitly
out of scope until that device is available. The active implementation slice is
SM86/model-relative: carded issue-port width and dual-issue transaction slots,
the separate uniform/SFU cadence, and a checked binding-cycle aggregation rule.
Carded issue resources are consumed when present (legacy pre-resource fixtures
remain readable through an explicit compatibility path); no MI300X timing
claim is implied.
The carded scheduler-resource values are source-bound architecture inputs; the
barrier-latency and occupancy entries retain reviewed-hypothesis provenance and
are not silently promoted to independent silicon measurements.

Binding-cycle emission is model-relative and deterministic: for each exact
launch, take the maximum of the issue-cycle count, terminal-event timestamp
plus one, and the modeled completion lower bound; across launches take the
maximum again. Launches are never summed, and unmodeled retirement tails stay
separate diagnostics.

The current SM86 comparison set is:

| Evidence | Role | Result |
| --- | --- | --- |
| `results/captures/sm86_adversarial_current_campaign_20260904_v11` | Current simulator campaign | 15 workloads; maximum scored error 9.704% |
| `results/captures/sm86_fresh_locked_ncu_validation_20260904_v84` | Best locked-clock comparison | strict median maximum 8.671%; raw SM-domain maximum 9.428% |
| `results/captures/sm86_fresh_locked_ncu_validation_20260904_v85` | Latest locked-clock comparison | strict median maximum 9.147%; raw SM-domain maximum 9.378% |
| `results/captures/sm86_fresh_batch_sensitivity_20260904_v33` | Repeat comparison | zero strict pass/fail flips between batches |
| `results/captures/sm86_fresh_locked_ncu_small_20260904_v10` | Small NCU input set | 8 workload shapes, seven repeats |
| `results/captures/sm86_fresh_locked_ncu_large_20260904_v10` | Large NCU input set | 7 workload shapes, seven repeats |
| `results/captures/sm86_regenerated_workloads_small_20260905_v1` | Fresh full small replay | 8 correctness, NCU, NVBit, static, and simulator stages completed |
| `results/captures/sm86_regenerated_workloads_large_20260905_v1` | Fresh full large replay | 7 correctness, NCU, NVBit, static, and simulator stages completed |
| `results/captures/sm86_postmerge_campaign_small_20260909_v1` | Post-merge clock-qualified small campaign | 8/8 all stages; NCU observation joined; causal probe recorded |
| `results/captures/sm86_postmerge_campaign_large_20260909_v1` | Post-merge clock-qualified large campaign | 7/7 all stages; NCU observation joined; causal probe recorded |

The simulator reports separate issue, modeled-event, completion, and launch
frontiers. An incomplete retirement tail is represented as typed state rather
than an invented fixed delay. Physical cache counters remain diagnostics until
there is an independent NCU mapping.

The active capture tree retains current/best comparison inputs and the runnable
small/large regenerated cohorts. Superseded capture roots are preserved under
`/data1/archive/gpu_captures_20260908`; non-runnable debug and failed staging
trees are under `/data1/archive/results_misc_20260908`. After this cleanup, all
15 regenerated NVBit captures still pass admission, and the complete release
8-small/7-large replay sweep at its retained campaign revision was zero-exit,
silent, and byte-identical. The current-source event/dense audit is recorded
below.

Campaign snapshots bind dynamic traces and threadblock payloads before
materialization. The release campaign runner and direct replay path both pass
their focused package checks, and the retained canary fixture remains runnable.

The post-merge clock-qualified full campaigns at revision `9df50c47` are
retained under `results/captures/sm86_postmerge_campaign_small_20260909_v1`
and `results/captures/sm86_postmerge_campaign_large_20260909_v1`. All 8 small
and 7 large rows completed correctness, seven-repeat NCU, NVBit, static SASS,
and simulator stages with zero clock-contract mismatches. Simulator commands
now attach the matching raw NCU CSV through the observation-only `--ncu-csv`
surface; 15/15 outputs contain the parsed launch observation. Each manifest
also records the native causal-frontier probe (26 exact events, 19 causal,
one collapsed region, two reified regions, unchanged final cycle 9).

The same-input post-merge differential replay at revision `9df50c47`
preserved every scored counter and raw issue/terminal frontier for all 8 small
and 7 large retained traces.
Fresh NCU medians are diagnostic because capture/profile sessions are not the
frozen scorer inputs; no per-workload timing or traffic constants were fitted.

The current-source replay audit reproduces event/dense output equality for all
15 retained workloads. A few retained `replay-1.json` files differ in their
front-end-heavy issue or terminal frontier because `fec7e2f2` added an explicit
demanded-stream maintenance wake; a clean `fec7e2f2` build shows the same
delta. The maintenance wake has a dense/event parity test, so these are
model-revision deltas rather than event-skipping mismatches; retained files
were not overwritten. The comparator accepts repeated `--command-dir` options
and prefixes trace filenames with a deterministic run index, so combined
small/large sweeps cannot silently omit a cohort or overwrite same-named rows.

The typed trace is deliberately diagnostic and opt-in. On the retained
embedding workload, no-trace event/dense replay took approximately 57/108 ms,
while full typed boundary tracing took approximately 9.8 s per lane. The
production replay path does not serialize typed boundaries, so this overhead
does not trade speed for cycle accuracy. A bounded full-trace probe of the
retained large convolution reached the configured 900 s wall-time limit after
about 20 GB of diagnostic JSON; the first 1,000 event and dense records passed
typed validation. Large typed traces therefore remain explicitly bounded
diagnostics; the default diagnostic cap is now 8,192 records, while no-trace
replay is the representative production path.
Reference trace deltas now use schema version 3 with ordered lossless
`replace`/array-`splice` operations; v1/full readers remain supported. On a
512-record convolution prefix, this reduced the serialized event trace from
1.17 GB to 650 MB without changing reconstructed boundaries. The complete
5,664-boundary flash-attention event/dense trace still compares exactly. The
terminal footer is written outside the data-record cap, including at the exact
limit; failed and timed-out comparator runs remove their invalid partial trace.

The SM86 novelty slice now includes a pure `sim-semantics` event and
partial-order certificate crate, an SM86 NVBit adapter, and the
`semantic-probe` executable. The adapter preserves lane-byte addresses, alias
multiplicity, widths, optional masks, allocation IDs, cache operators, and
placement identity. `tools/semantic_events/sm86_roundtrip_compare.py` runs a
canonical event round-trip and then the existing cycle-accurate replay. All
fifteen retained rows round-trip with identical counters, timing, and coverage;
the sidecar classifier leaves no opcode unresolved. Static constant-cache
`LDC/ULDC` records still have explicit dynamic memory-class gaps because the
producer omits that tag; the adapter preserves them as static-only events and
never routes them through global memory. This is a metadata gap, not a timing
default.

The first exact-acceleration implementation slice is now present: a versioned
`sim-semantics::ObservationContract` distinguishes boundary and per-cycle
claims, `sim-engine::QueueStats` reports virtual clock gaps separately from
handler events, and schedule-derived causal inputs reject barriers, memory
roles, and uncharacterized dispatch-only operations before certification.
The optional dense reference differential now matches all 8 retained small
SM86 workloads. The current event lane took 1,246,619 local steps with
219,112 idle jumps covering 3,771,866 virtual cycles across that set; dense
mode took 5,018,485 local steps. The comparison is reproducible with
`tools/semantic_events/sm86_reference_compare.py`. The retained 7-workload
large cohort also matches 7/7, with 2,069,235 event steps, 202,550 idle jumps,
and 3,164,825 skipped cycles versus 5,234,060 dense steps. Across all 15 rows,
the event lane took 3,315,854 local steps and skipped 6,936,691 virtual cycles,
while dense mode took 10,252,545 local steps.

The descriptor-model causal probe now requires an explicit complete resident
warp roster for any collapse. Unknown residency and an extra resident warp both
reify, so SMSP exclusivity is never inferred from the candidate instruction
list alone; native SM replay remains the cycle-accurate timing path.

The optional `SIM_SM_REFERENCE_TRACE` emits post-event ordering plus one
canonical typed warp/block boundary root. The overall projection remains
deliberately marked incomplete for legacy consumers, while L1, shared L2
ownership, async-group state, memory/scalar pipeline state,
instruction-front-end state, and fixed-clock activity power are now typed. Each
event carries versioned cache geometry/set/way state, canonical SM response
owners, pending-fill ready times, typed pending-event payloads, and an
activity-only power state. The trace stores one full event followed by
lossless deltas; the schema-revision-13 typed root matches between event and
dense replay, and explicitly reports incomplete ownership/tail state whenever
an outstanding response lacks an exact due time or the workload contains an
unmodeled completion tail. Exact scheduled response cycles are normalized into
each typed L2 owner after the L2 event is processed; the initial pre-pricing
boundary stays incomplete by design. The old aggregate projection flag is
retained as a conservative legacy field for V1 consumers.

The typed core now marks active immediate-constant-cache state as
non-certifying: its tags, pending lines, and service frontier are not yet
serialized field-by-field, so cycle-exact contracts reify instead of claiming
that a constant-cache transition can be resumed from omitted state. The
address-carry canary remains event/dense-identical; its completeness is also
conditioned on the independent tail/frontier markers below.
The same schema-15 marker covers live async L1-hit service frontiers, result
writeback state, memory-service candidates, and admission-feedback queues;
these optional timing sources are now non-certifying until projected.
The typed-core validator also rejects duplicate arrival stamps among resident
warps. Arrival stamps are the scheduler's stable tie-break key, so accepting a
collision would allow a forged exact boundary to resume with a different issue
order; retired warp records may still retain historical stamps.
Nonzero dependency-barrier pending counts are also non-certifying: response
owners currently preserve requester and allocation identity but not the barrier
index that a response clears, so the contract derives an explicit
`dependency_barrier_ownership` incompleteness reason even if a producer omits
the marker.
Resident warp rosters now also enforce the SM86 `warp_slot % 4` to SMSP mapping;
moving a live warp between scheduler rosters is rejected before an exact claim
can be made.
Historical CTA-barrier dispatch timestamps are checked against each local core
clock as well; future barrier-service arrival frontiers remain legal because
the shared barrier server can complete them after dispatch.
Response owners are now bound to the serialized warp arena: their dynamic warp
index must exist, carry the declared physical slot, and have issued the
referenced instruction. This prevents an owner identity from certifying a
response to nonexistent or not-yet-issued work.
Ownership-complete boundaries also cross-check every known-due owner against
the queued `MemoryResponse` identity and timestamp; orphan owners or response
events now fail structural validation instead of relying on separate local
shape checks.
Pending L2 sectors receive the same cross-check: their allocation/generation,
sector mask, and ready cycle must be represented by a queued `L2Fill` or a
response carrying that fill. Stale pending-ready entries are marked
incomplete during projection rather than rejected as impossible live state.
Live L1 line/sector fills remain non-certifying until the response-owner schema
also carries the private L1 fill-key generation consumed by the response
handler; typed fill timing alone does not establish that owner edge.
CTA-barrier release history likewise marks exact boundaries incomplete while
the cumulative modeled local-completion frontier is not serialized; current
barrier generation state alone cannot reconstruct an earlier maximum release.
Replay capability tails (RMW, dirty writeback, coherence, or unsupported
LDGSTS destination visibility) now also make exact typed contracts incomplete;
their independent tail frontier remains separate from issue and modeled-event
cycles.
Shared-L2 transient retry/fill-waiter/mask/fold state is also non-certifying
until those response-drain maps are represented explicitly; typed L1 tags and
response owners alone do not identify the parked sector-range transition.
Schema 15 replaces the former carded-DRAM incompleteness marker with a lossless
typed server state: mode, rate, clock, cycle quotient, and fractional remainder
are serialized and validated without rounding the next service frontier.
Latency-only and carded canary traces remain event/dense-identical. The strict
SM86 carded-issue lane consumes arbitrary positive issue-port width and the
dual-issue capability alongside the fixed four-SMSP/barrier topology;
occupancy/barrier mismatches still fail closed rather than silently borrowing
defaults.
The second dual-issue transaction is limited to an independently ready warp;
same-warp control/dependency stalls are never cleared by the extra slot.

Typed event payloads use explicit origin, source, operation, and response-kind
enums. Queue boundaries preserve the engine's phase, batch sequence span, and
next unused sequence position, so future summary transformers cannot certify
from debug formatting or an incomplete ordering projection. Live bypass
asynchronous-copy observations preserve producer identity, sector-to-cohort
membership, and response frontiers; count-only or unknown bypass state remains
incomplete for exact contracts. The current single-phase engine also rejects
nonzero serialized phases at the typed boundary, preventing a forged phase
from reversing same-cycle sequence order until a future multi-phase dependency
contract exists. Partially captured bypass projections now validate every
retained observation before reporting diagnostic incompleteness; omitted
observations remain explicitly non-certifying. The SM adapter also marks
aggregate-only bypass captures incomplete when cohort descriptors or
sector-membership masks are absent, so exact contracts cannot certify from
counts alone. The semantic validator rejects a forged complete marker when
those exact topology fields are absent as well.
Power samples also carry per-domain clocked/gated integrals; forged aggregate
activity totals, elapsed-integral overflow, divergent domain clocks, and
duplicate domain identities are rejected before a power-aware proof can
certify. Sample schedules also cannot leave `next_sample` at or before the
elapsed domain clock, skip a declared period, or shift off the native
period-aligned cadence. Power event recording now applies the same schema guard
as clock advancement and leaves an unsupported-schema state unchanged.
Modeled `memory_write_ack` events are also counted as completion activity,
matching the event queue's completion classification without changing timing.
`atomic_completion` is likewise counted in both memory and completion activity,
so returning atomic work is not lost from the typed activity totals.

The typed L2 validator now cross-checks every allocation generation against its
resident way and derives sector ownership from the resident pending/dirty/valid
masks. Consumer and producer edges are checked against canonical request
identities, line/sector masks, and pending ready times. The validator also
checks the generation frontier and requires resident/allocation line identities
to use the declared address domain. Missing resident ways, mask/status drift,
forged completion edges, or frontier/domain mismatches reify as typed failures;
this check does not alter the replay scheduler or event timestamps. Producer
edges now retain the exact L2 fill generation that created a pending sector;
stale or replyless generation ambiguity remains incomplete instead of choosing a
canonical requester. Known SM86 slice-hash formulas are also checked against
every resident virtual line's serialized slice/set placement, and unknown
pre-pricing or evicted allocations remain incomplete. Shared L2 owners are
joined against the per-core response roster before completeness can be claimed.
Shared-L2 response-owner keys also reject negative physical warp slots before
ownership joins, preventing an opaque sentinel from being treated as a valid
retired-warp identity. They also reject duplicate serialized dynamic response
identities even when a forged record changes only the response kind; the
native private owner token is not present in the typed key and cannot disambiguate
that collision.

A cycle-frontier audit also found that a standalone typed L1 validator did not
compare timed outstanding fills with the owning SM clock. Typed core
validation now rejects a fill due strictly before that clock while preserving
untimed event-mode fills and standalone cache validation. The new regression,
all 58 semantics tests, all 430 SM reference tests, and a rebuilt SM86
address-carry event/dense replay pass; the two replay outputs remain byte
identical.

The same audit now bounds each active block's unfinished warp count by its live
warp residency count. This prevents a typed boundary from understating
`resident_warps` while retaining unfinished work; the full focused suites and
the rebuilt event/dense canary remain identical.

Root typed-boundary validation also rejects an outstanding response owner whose
known due cycle is before the boundary event timestamp. Same-cycle due times
remain valid because queue sequence order can leave that response pending;
unknown due times remain diagnostic-incomplete. The bounded canary still
reconstructs 13 equal schema-15 event/dense boundaries after the check.

Typed core validation now also enforces physical warp-slot disjointness across
unfinished CTAs. Retired slots may remain in an active block's reservation
until its final warp retires, so only fully retired blocks permit slot reuse.
The new adversarial regression and rebuilt typed event/dense canary pass.

Block projections now validate the contiguous `first_warp` arena ranges,
block identities, and slot rosters used by native barrier-release indexing.
Forged ranges cannot point into another CTA while retired-block history and
legal slot reuse remain representable.

Hold state is now checked against the owning SMSP roster as well as global
residency, preventing a held warp from being reassigned to another SMSP in a
typed resume.

CTA barrier projections now cross-check aggregate waiting counts with each
warp's barrier flag and dispatch/arrival timestamps, including the block's
first-dispatch and first-arrival frontiers. Aggregate-only barrier state can no
longer certify a different release transition.

Pending shared-L2 sectors now also require a known ready time no earlier than
the boundary event timestamp; same-cycle pending fills remain legal.

Typed queue payloads now bind L2 access issue times and write-ack visibility
times to their serialized event timestamps, matching native event admission
checks for single and batched accesses.
Pending event targets are range-checked against the serialized core roster,
matching current-event target validation.

Write-ack payload validation also enforces the native monotonic chain
`issue_at <= accepted_at <= service_done_at <= fence_visible_at`.
It also restricts acknowledgements to Store/Reduction operations and validates
the four-SMSP request witness index/count domain.

Typed L2 requests now validate native sector-mask/anchor-bit rules, Local-mask
provenance, source/operation/reply combinations, and admission-witness
constraints before a queue boundary can certify. Witnesses inside one L2
access batch must be contiguous within each active instruction group; a
completed group may be followed by the next instruction's index-zero witness
in the same timestamped batch, matching the native admission observer.
An unwitnessed request cannot silently split an incomplete group: the validator
retains that interruption and rejects the next witness rather than accepting a
forged index restart. The focused semantics/trace suites and all 15 retained
SM86 dense/event replays remain equal after this guard.
Constant-load markers are restricted to L1-fill reads, with deep markers
requiring the broad constant marker.
Sector-fill response identities now also require a single-sector request mask;
line-fill replies retain multi-sector masks.

Typed fill identities now require a non-empty four-sector mask, and serialized
response identities cannot use the sentinel `None` response kind.
Memory-response events must carry a returning reply identity; replyless fills
remain represented by `L2Fill` events.
When a response carries a fill token, its reply identity must be a returning
L1-fill, direct-Warp, or AsyncCopy kind, matching native mixed-response
scheduling; the `None` sentinel remains invalid.

The native batch scheduler and reference trace then exposed two boundary false
positives. The typed batch validator now accepts multiple completed admission
groups in one batch while rejecting interleaving and skipped indices. Cold
L1-bypass reads can attach a new L2 fill to a direct Warp or AsyncCopy
response, so those typed response kinds are accepted alongside L1 fill kinds.
The native multi-group regression and cold AsyncCopy reference replay pass;
the address-carry canary still produces 13 equal event/dense boundaries.

The sectored-MSHR audit found a real cycle error in late same-line merges.
`line_folds` previously covered an entire `SmCore::advance`, so a request issued
at a later local cycle could inherit the first request's response time; a
cross-batch follow-up also cleared its waiter on the parent line response.
Timestamp-scoped folding now permits coalescing only for same-cycle packets.
Late sectors reserve a generation-bound L1 sector token, emit a returning
`L1SectorFill` response, and become resident only at that child's completion.
The L1 projection preserves a child that returns before its parent line, and
the typed L2 owner uses each response's exact carried sector mask. Native
source review of pinned Accel-Sim/GPGPU-Sim confirms per-32-byte child
completion for sectored MSHRs. The late-sector regression changes the canary
shape from the old 458-cycle early release to the causal 847-cycle release;
the direct timestamp regression rejects `(0,40,3)` and requires separate
`(0,40,1)` and `(15,41,2)` requests. Retry draining is gated on an independent
MSHR release so a child completion sharing its parent slot cannot introduce a
speculative retry.

The follow-up lifecycle review found a generation-ABA MSHR accounting hole:
zombie sector children from an evicted same-tag allocation were previously
treated as sharing a newly allocated parent line. Occupancy and retry-release
checks now compare `(line, generation)`, preserving finite-MSHR pressure and
allowing the new parent to release its own slot without waiting on an old child.
The new finite-MSHR regression passes in the 137-test memory suite.
The numeric-boundary preflight also models late-sector token allocation on its
shadow L1, so fill-sequence exhaustion fails before the real issue cycle can
mutate counters or waiters.
Typed L1 occupancy diagnostics use the same generation-aware sharing rule.

The bounded constant-L1 retry audit found a transactional edge case when a
finite MSHR set is already full: one access can cross multiple retry points,
but the fast preflight budgeted only one stall. A later counter or timestamp
failure could therefore apply an earlier due fill before returning the error.
Constant-mode wrappers now force the existing shadow-cache transaction whenever
the effective MSHR is full, including access-local LDGSTS floors. The two
multi-retry regressions and the full 137-test `sim-mem` suite pass; ordinary
SM86 issue tests remain green.

The serialized block ranges must also cover the warp arena exactly; an
unclaimed tail warp state is rejected before resume indexing can drift.
Typed allocation lists now use the signed `CanonicalLineKey` order required by
the validator, while each set retains the native way-vector order used by
replacement tie-breaking. A stale same-line fill completion is covered at the
typed boundary and leaves the current allocation generation unchanged.
The latest focused reference-enabled package run passed 431
`sim-sm` tests and 58
`sim-semantics` tests, and the address-carry canary still matched all 13 typed
event boundaries and the dense terminal cycle. Its schema-15 completeness is
contract-relative: pre-pricing ownership, live unmodeled tails, and omitted
timing owners remain explicitly incomplete rather than being counted as exact.
The subsequent full workspace reference sweep passed every runnable target,
including 461 `sim-sm`, 137 `sim-mem`, and 58 `sim-semantics` tests; only the explicitly
machine-local replay and sidecar tests remained ignored for missing host data.
The final default-profile workspace sweep also passed 455 `sim-sm`, 137
`sim-mem`, and 58 `sim-semantics` tests, with the same two host-dependent test
classes ignored.

Bounded SM advancement now commits the local clock to the parked external
horizon before returning. The dense reference also walks idle cycles strictly
below a bounded horizon, while event mode jumps directly to it; a repeated wake
therefore cannot replay front-end maintenance or prefetch work at the old
timestamp. Expired memory/scalar pipeline releases and fixed-clock cache
completions are retired at the parked horizon as well, keeping typed state
valid when an event-mode jump crosses their release times.

Async-copy cohort projection now rejects reversed first/last response times in
release builds instead of relying on a debug assertion and saturating the
reported duration to zero. Valid event and dense replays remain unchanged.

The typed L1 validator now requires a retained constant/event fill-timing latch
whenever a resident line, sector map, or allocation generation is live. An
empty store-only cache may still omit the latch; a populated read cache without
it is ambiguous and cannot certify a cycle-exact resume.

The shared-L2 boundary validator now checks replacement metadata that affects
the next victim: SM86 SRRIP RRPVs stay within range, LRU/BIP recencies stay
within the serialized clock domain, and LRU/BIP live ways cannot share a
recency stamp. Empty resident ways are rejected as well. Forged replacement
state now reifies instead of changing a future eviction while retaining an
apparently complete boundary. Allocation generations are also required to be
globally unique across live lines and serialized allocation records, matching
the native ABA-protection allocator.
The retained FlashAttention SM86 replay passed this guard across 5,664 typed
event boundaries: event and dense outputs and boundary traces were identical,
with terminal cycle 15,231. The subsequent global-generation guard passed the
same 5,664-boundary differential unchanged.

Typed core boundaries validate four subcore rosters, CTA residency reservations,
holds, block rosters/counts, unfinished/shared-memory accounting, response
counts, response state, unfinished warp-state/resident agreement, and
canonical ownership order. A live
canary exposed and removed an invalid assumption that every outstanding
response must still have a resident warp slot.
Historical CTA/warp-slot reuse is keyed by block identity in the typed
projection, so retired blocks do not collide with later admissions.
Attached typed L1 projections now validate stamp inverses, resident capacity,
MSHR shape, latency/fill-timing discipline, sector masks, pending-fill
ordering, line-fill generation identity, minimum due time, and sequence bounds
before a
boundary can claim completeness.
Sector/generation entries must also name a resident line or pending line
allocation; miss decomposition, bounded lookup-batch state, and legal stale
sector-fill tails are checked explicitly. A stateful write-lookup observer is
also explicitly incomplete until its tag-window state is serialized.
The typed L1 projection deliberately does not compare live fill count with the
base MSHR number: SM86 cacheable LDGSTS uses an access-local admission floor.
That distinction is covered by a native async-copy pressure regression.
Typed memory and scalar pipeline projections also validate SM86 queue shape,
capacity, configuration intervals, release ordering, admitted-count and
occupancy conservation, service-frontier consistency, and stale-release
rejection relative to the core cycle.
Typed instruction front-end projections validate fixed SM86 subcore shape,
cache-set occupancy, LRU access-sequence bounds, consecutive stream lines,
cache-set placement, unique LRU stamps, readiness ordering, timer/frontier
consistency, L0 geometry, SM86 width constraints, consecutive stream lines,
stream-tail frontiers, and stream-buffer capacity.
Async-copy projections preserve next group sequence state, commit-time order,
and source-before-destination service ordering; they reify whenever
bypass-copy identity/provenance is explicitly unknown or omitted.

Outstanding response owners preserve the shared-L2 allocation generation that
priced each response separately from the L1 fill generation. Stale same-line
responses therefore remain explicit retirement-tail state and cannot be joined
to a newer pending allocation.
The current schema-15 embedding replay produced 2,456 event boundaries with
identical event/dense typed traces and outputs. Completeness is now derived
per contract from ownership, tail, and optional timing markers; pre-pricing L1
response-owner allocation-generation ambiguity remains explicit, and the
projection never guesses that join.
The front-end-heavy flash-attention replay also produced 5,664 identical typed
event/dense boundaries, exercising the partial-CTA-residency validator.
Two independent event-mode and two independent dense-mode address-carry runs
also produced byte-identical stdout and stderr, covering process-level replay
determinism in addition to lane differential equality.

`sim-semantics::transform` now provides a contract-aware typed boundary proof
and checker interface. Its identity transformer certifies only a complete,
closed zero-length entry/exit pair; non-empty work, open frontiers, and
same-cycle external arrivals reify until a transition rule derives their full
effects. Its checker binds independent entry, exit, frontier, and transition
evidence before accepting a proof. Native GPU summary regions remain guarded
by the existing detailed fallback.

The typed-L2 ownership audit found one certification hole: a complete
event-discipline boundary could carry a resident way with no allocation
generation, causing that way to disappear from the allocation projection and
making a later same-line fill ambiguous. Complete event-mode ways, and all
pending ways, now require a generation identity; immediate-mode resident ways
retain their valid generation-free representation. The new regressions and the
62-test semantics suite pass, along with the 137 memory tests and both SM
profiles. The same validator now rejects negative L2 clocks, which cannot be
produced by the checked native clock transition and would otherwise seed an
impossible replacement timestamp.

The adjacent SM-core audit found that per-warp instruction-buffer contents
were serialized but never structurally checked. Complete boundaries now reject
buffered instructions on flat cores, done warps retaining buffered work,
over-capacity buffers, cursor/order violations, and buffers that precede the
warp's next instruction. The native SM86 replay path remains unchanged.

The memory-pipeline follow-up found that a drained queue stopped validating the
shared acceptance frontier even though that frontier gates the next issue.
Validation now ties it to the terminal historical acceptance cycle, including
the zero-admission case, so a forged post-drain delay cannot certify.

The same history-consistency guard now covers the memory issue pipeline: a
nonzero admitted count with drained queues and four zero local-completion
frontiers is rejected because native admission always advances one SMSP
frontier. The full 76-test semantics suite and SM86 canary remain green.

The scalar-pipeline validator now also rejects a nonzero admitted-history
counter when all release queues are drained and every per-SMSP service
frontier is still zero. Native scalar admission always advances one frontier;
the regression prevents an aggregate diagnostic from hiding an impossible
resume state.

Root boundary validation keeps each SM's local clock independent from the
global queue event timestamp. Event-mode run-ahead intentionally records a
core after it has advanced to its own response horizon while the shared event
queue still contains cross-component work; the typed core invariants validate
that local state without imposing a false cross-frontier ordering.

The bounded typed-trace canary now passes in both event and dense modes with
13 reconstructed boundaries each; outputs and typed boundary traces are equal.

A bounded typed comparison of the retained embedding workload also passes:
2,448 event calls reconstructed 5,016 lossless records, with equal event and
dense outputs and equal boundary traces. The 6,000-record cap kept this
diagnostic run finite.

This probe exposed and closed a real contract defect: root validation had
incorrectly compared each local SM clock with the global queue timestamp.
Native run-ahead legitimately records a core at its response horizon while
the queue still contains cross-component events; local core invariants now
validate that state without imposing a false cross-frontier order.

Incomplete L2 ownership projections now also require at least one explicit
`unknown_state` reason. Native event projections already provide this reason;
an unexplained incomplete boundary cannot silently pass structural validation.
Those reasons are now required in canonical sorted order as well, preserving
deterministic completeness metadata across serialized boundaries.

CTA barrier projections now also reject a nonzero re-arm frontier before the
first barrier generation. That frontier gates future barrier issues and cannot
exist in the native initial block state.

Async-copy boundaries now reject committed groups timestamped after their
owning core cycle. Future source and destination completion times remain legal
for in-flight work; the commit itself cannot be from the future.

Core projections also conserve the issued-instruction counter against the sum
of per-warp instruction cursors, preventing forged throughput totals from
certifying as complete state.

The typed-L1 boundary follow-up now rejects empty architecture provenance and
the impossible overlap of a resident line with an in-flight line allocation.
This keeps cross-architecture or impossible fill/residency snapshots from
entering transition reasoning; the focused semantics suite remains green.

The async-copy follow-up now binds each retained bypass observation to its
owning group's aggregate pending-source count, source-completion frontier,
and delivery-work total. An observation that exceeds any aggregate is
rejected before a complete boundary can certify it; 63 semantics tests, both
SM profiles, and the byte-identical SM86 event/dense canary pass.

The memory-pipeline audit now also checks diagnostic conservation: maximum
operand/shared delay cannot exceed its accumulated total, admitted work must
have recorded nonzero occupancy, and zero-admission state must retain zero
frontiers and delay counters. These checks are projection-only and preserve
the native SM86 timing; both reference modes and the canary remain identical.

The optional scalar-pipeline projection now applies the same history checks:
admitted work must record occupancy, occupancy cannot exceed admissions,
service frontiers cannot precede the first latency-plus-initiation service,
and zero-admission state cannot carry diagnostic history. The 64-test
semantics suite and both SM profiles remain green.

The event queue projection now orders lightweight event references before
materializing typed payloads. This avoids moving large batched request
vectors during diagnostic sorting and makes projection callbacks observe the
same canonical `(time, sequence)` order as the serialized boundary; the
SM86 event/dense canary remains byte-identical.

The typed L2 audit now rejects `response_kind: none` for every request owner,
including incomplete ownership projections. `none` has no executable response
identity and is distinct from an explicitly unknown allocation join; the 64
semantics tests and both SM86 profiles remain green.

Complete typed L2 pending sectors now also require every referenced consumer
owner to remain outstanding. Retired, stale, or non-returning owners cannot
continue to represent a live pending fill; incomplete ownership joins retain
their existing diagnostic latitude.

The completeness contract now validates power and every supplied core before
classifying a record. Direct contract callers can no longer obtain `Complete`
from malformed core state that the root boundary would later reject; the 65
semantics tests and SM86 event/dense replay remain green.

At the current tree, the retained eight-small/seven-large SM86 comparator was
rerun without boundary tracing. All 15 event/dense output pairs matched
exactly; event mode skipped virtual idle cycles in every applicable workload
while preserving issue and terminal frontiers. This is the production replay
path, independent of typed diagnostic trace serialization.

The same revision now passes both full workspace profiles: 455 default SM
tests and 461 reference SM tests, with 65 typed-semantics tests and all other
workspace targets green. Only the documented machine-local capture tests stay
ignored because their host artifacts are unavailable.

After the maximum-cycle memory-pipeline regression was added, the latest
reference-enabled workspace sweep passes 462 `sim-sm`, 138 `sim-mem`, and 65
`sim-semantics` tests, with all other runnable targets green. Earlier 455/461
counts above remain historical checkpoints.

The focused static pass also cleared the actionable Clippy warnings in the
typed memory-request validator and generation-aware L1 release predicate. The
remaining workspace warnings are legacy test-only unwraps or unrelated
utility-code style warnings; they do not affect the validated replay path.

The same pass removed index-based loops from the SM memory-service candidate
hot path and simplified its contract/divisibility checks. Reference SM tests,
the release build, and the event/dense canary retain identical outputs.

The asynchronous-copy response loop now walks its validated sector slice
directly while preserving absolute response indices, removing another
indexing-only hot-path warning without changing request order or cycle
frontiers.

The retained 15-workload event/dense comparator was rerun after this change;
all serialized outputs still match exactly, including async-heavy and large
cohort workloads.

That 15-workload production sweep was rerun again after canonicalizing
incomplete-L2 reason ordering; all event/dense outputs still match exactly.

The DRAM aggregate regression now pins traffic counters as additive and
terminal dirty-residency gauges as maxima across launch composition, avoiding
double-counted residency diagnostics.

The event processor's admission-observer branches were also collapsed after
checking their short-circuit/error ordering. Clippy no longer reports those
branches, and default/reference SM tests plus the canary remain unchanged.

The complete 15-workload comparator was rerun after that event-processor
change; all event/dense outputs remained equal across both retained cohorts.

Ownership diagnostic strings now format canonical line keys directly with
their `Debug` representation, avoiding nested temporary allocations without
changing the serialized unknown-state text.

The shared-L2 counter provenance helper also now constructs its unbound error
without a lazy closure; typed error precedence and replay outputs are
unchanged.

Both full workspace profiles were rerun afterward as well: 455 default SM
tests and 461 reference SM tests, with 65 semantics tests and all other
targets passing.

Process-level determinism was also checked on the retained large
flash-attention backward command: two event-mode runs and two dense-mode runs
were byte-identical in both stdout and stderr.

The semantic round-trip audit found that static `LDC/ULDC` events retained
their native cache operator and declared width in the canonical event but
`to_instruction` restored those fields only for dynamically tagged memory.
The adapter now restores both fields while preserving the intentional missing
dynamic class; the focused trace suite (132 tests) and a retained convolution
round-trip with 2,176 explicit constant-class gaps pass with identical replay
counters, timing, and coverage.

The Python L1-walker boundary now rejects a sector tuple whose global
`sector_id` belongs to a different declared `line_id`. Previously that malformed
input could create an impossible per-line sector and contaminate the filtered
L2 stream; both fully-associative and warp-pool walkers fail closed, with the
focused predictor suite passing 38 tests. The same boundary rejects malformed
float/string/negative line and eligibility fields instead of silently coercing
them into a different access stream; carveout percentages are also required to
be integer manifest values rather than silently truncated. Carveout byte-domain
inputs are now likewise required to be nonnegative integers, preventing a
negative shared-memory request from being hidden by the L1 cap; the focused
predictor suite now passes 43 tests.

The public trace-record adapter now validates dictionary and typed-record
fields before normalization: block/warp/SM identities, widths, byte addresses,
and PCs cannot be silently truncated or string-coerced. A retained v3
embedding capture (3,080 records, 2,080 walker accesses) produced identical
grouped streams and L1/L2 results before and after the guard; a fractional PC
now emits the structured `predictor_l1_sass_provenance_missing` diagnostic.
The separate trace-derived CTA-to-SM partition helper now applies the same
strict identity checks, preventing float/string/boolean IDs from changing the
real-SM partition; the focused predictor suite passes 59 tests.
The same adapter now rejects boolean/float `sm_count`, `grid_dim`, and
`block_dim` values rather than relying on truthiness; the focused suite now
passes 65 tests.
Manifest joins also reject duplicate normalized PC entries instead of letting
the last token silently overwrite the first; the full cache-predictor suite
passes 90 tests under the repository-pinned requirements. The classifier and
the standalone manifest-join helper both reject duplicate or malformed PC
identities before an ambiguous manifest can be written.
The warp-pool card loader now also converts malformed JSON and non-object card
roots into the typed `WalkerConfigError` boundary instead of leaking parser or
attribute exceptions; its focused suite passes 78 tests.
The adjacent static-card table loader now maps malformed JSON and non-object
roots to `L1StaticCardEntryInvalid` as documented; its two regression cases are
kept behind the module's optional `jsonschema` dependency, which is absent from
the lightweight predictor venv. The repository-pinned `uv run
--with-requirements requirements.txt` path runs all 90 cache-predictor tests;
duplicate architecture rows are rejected instead of silently selecting the
last row by file order. Corrupted bundled schemas also remain inside the typed
invalid-entry error boundary.

Typed memory-pipeline validation also now bounds each serialized local
completion frontier by the native terminal shared-accept frontier plus one
local initiation interval. This rejects forged run-ahead state while accepting
the rebuilt release canary: all 13 bounded event/dense typed boundaries match,
and the Conv2D event/dense replay remains byte-identical. Live release queues
also now require an initialized local frontier, matching the native plan's
atomic queue/frontier update.

A maximum-cycle adversarial probe then found that this validation-only upper
bound could itself overflow at `u64::MAX`, rejecting a legal native pipeline
projection even though every scheduled timestamp remained representable. The
bound comparison now uses `u128`; actual cycle transitions remain checked in
the native `u64` domain. The new maximum-frontier regression passes, and the
typed canary plus embedding/FlashAttention differential replays remain equal.

The same arithmetic audit found an empty scalar-pipeline projection was
computing `latency + initiation` even though no service frontier existed yet.
The origin check now runs only after a nonzero service frontier is present, so
high-range empty configurations remain representable while admitted frontiers
retain checked arithmetic.

The causal-frontier admission boundary now rejects non-SM86 schedules before
joining PCs or constructing the recurrence. Its four-SMSP placement model is
architecture-specific; accepting an `sm_120` schedule would otherwise make an
Ampere topology look cycle-equivalent. A targeted `sm_120` regression passes,
and the SM86 event/dense canary remains byte-identical (62 event steps versus
378 dense steps, 316 cycles skipped).
The adjacent native replay and issue-model constructors were also checked:
their architecture/card gates already reject unsupported `sm_120` execution
before entering the SM86 arrays, so no additional compatibility shim was
needed.

The production trace-to-cache adapter audit found two silent fidelity losses.
The protobuf walker was dropping instruction `sm_id`, which made traces with
valid per-CTA placement metadata appear to have no real partition; it now
preserves that optional field. It also joined classifier manifests with the
widened `vpc` and collapsed all address operands to the first operand's width.
The join now uses the contract's function-relative `pc`, and each address
operand becomes its own `TraceRecord` with its own width, matching the Rust
reader's per-address projection. A focused fake-proto regression covers the
high-bit `vpc`, two operand widths, and SM identity.

The SASS classifier now asks `cuobjdump` for the requested architecture and
selects the first exact kernel symbol section, preventing same-named sections
from other fatbin slices or substring-colliding helpers from entering an L1
manifest. The focused cache-predictor suite passes 76 tests; the classifier
also matched all 10 retained SM86 scalar-copy binaries. The bounded SM86
event/dense canary still reports identical boundaries and serialized outputs
(62 event steps versus 378 dense steps, 316 cycles skipped).

The public L1 miss-stream helper now applies the same strict warmup-count
validation as the sibling walkers; booleans and fractional values no longer
become implicit launch counts or leak a raw `range()` error. Three focused
regressions cover those coercion cases.

The production adapter was also exercised with the repository-pinned
`.venv` protobuf runtime (7.34.1), using the retained SM86 address-carry
canary rather than a synthetic proto. The real walk decoded 5 memory records,
preserved the observed SM identity, derived `real_sm_partition`, and produced
13 L1-miss-stream entries; the filtered replay accounted for 4 L1 hits, 13
L1 misses, and 13 L2 accesses without a contract error.
The same production path was then exercised on a retained
`sm86_addr_carry_ldgsts_dst_v3` capture: it decoded 4,112 records across four
warps with mixed per-operand widths `{4, 8, 16}`, preserved the real SM
partition, and completed the filtered replay with 65,672 modeled L1 accesses,
594 L1 misses, and 594 L2 inputs.
Replacing the synthetic manifest with the matching live `cuobjdump` SASS
classifier produced 8 `LDGSTS` entries, correctly modeled all 65,536 accesses
as L1 bypass traffic, and yielded 65,008 L2 hits; the real-SM partition and
v3 width handling remained unchanged.
The adversarial pass then found three global `STG.E` PCs in the same capture
that the classifier candidate regex had silently omitted. Global stores now
enter the closed `OTHER` token as L1-bypass traffic; the live manifest grows
to 11 entries and the v3 replay accounts for 65,608 L2 inputs and 65,014 hits.
Shared-memory stores remain excluded from this global L2 predictor lane.

The typed instruction front-end validator now also requires positive
`function_unique_id` values for cache entries, stream lines, and stream
frontiers. Zero or negative identities could previously pass set/order checks
and contaminate a supposedly complete boundary with an invalid trace join; the
new typed rejection has a direct regression.

The contract classifier also now refuses to certify `CycleExactMemoryV2` or
`CycleExactPowerV2` when the typed SM-core roster is empty. Diagnostic
boundaries may still intentionally carry only shared-L2 state, but an exact
contract without per-SM issue/frontier/resource state cannot reconstruct the
next transition. The new `core_state_missing` regression preserves existing
incomplete-L2 and response-owner error precedence.

Typed core and L2-owner validation also reject `u32::MAX`, the native
conversion sentinel used when an observed `sm_id` is negative or otherwise
invalid. This prevents malformed placement metadata from masquerading as a
real SM identity in an otherwise complete boundary; focused regressions cover
both core and requester-owner paths.

The retained regenerated small/large command artifacts were then replayed
through an explicit path rebase: their historical `/tmp/gpu_full_small` and
`/tmp/gpu_full_large` roots no longer exist, so an unmodified command fails
before simulation. `sm86_reference_compare.py` now accepts repeatable literal
`--path-rewrite OLD=NEW` substitutions instead of guessing roots. With the
rebased commands, all 8 small and 7 large SM86 event/dense JSON outputs are
byte-identical; embedding and FlashAttention boundary traces also match
exactly. The trace-heavy boundary export is intentionally run on `/data1`
because the historical `/tmp` workspace is near capacity.
The final direct run against the original retained command directories also
reports `all_outputs_equal: true` for all 15 rows; its current terminal-cycle
map is recorded by the machine-readable comparator output, while the bounded
canary still reports both output and boundary equality at terminal cycle 785.
After the exact-horizon pruning change, the rebuilt release binary was
rerun against all 8 retained small and 7 retained large simulator commands;
all 15 event/dense outputs remained equal, with event mode skipping virtual
idle cycles in every workload.
The current focused simulator runs remain green as well: 426 default `sim-sm`
tests and 434 `--features reference` tests, including the maximum-cycle
memory-pipeline regression, plus 139 `sim-mem` and 67 `sim-semantics` tests.
Replay termination now also fails closed if any response, wake, or unpriced
response-bound ledger remains after the event queue drains, or if an SM still
owns a response after retirement. The release binary was rebuilt with these
guards and the SM86 event/dense canary remained byte-identical.
The already-parked horizon path now also prunes release queues at the exact
external timestamp before exposing the boundary; a regression covers a
release equal to the current horizon.
The L1 event path now keeps the independent u64 fill-order sequence separate
from the compact u32 parent-generation identity: a resident-line sector fill
continues past the u32 generation boundary when its parent generation is
valid, while new line allocations still fail closed at the representational
limit. The boundary regression passes with the full L1 suite.
The comparator now resolves trace paths before launching a child with a
different working directory, so relative trace directories are cleaned and
compared consistently after failures or successful runs. Its focused Python
coverage is 86 tests. The warp-pool predictor also materializes its outer
access collection before duplicate-key validation, preserving one-shot
iterator workloads through both measured and warmup launches.

A bounded primary-source refresh added three current comparison points. The
[ACALSim paper](https://arxiv.org/abs/2605.22936) presents fast-forwarding and
deterministic two-phase parallel execution as framework mechanisms while
leaving timing-model accuracy to each simulator. [Sim-FA](https://arxiv.org/abs/2605.00555)
retains only timing-visible asynchronous-pipeline state and explicitly limits
validity to fixed software schedules and its modeled critical path. The
bounded [FlashGPU-Sim checkout](https://github.com/FlashGPU-Sim/FlashGPU-Sim)
keeps execution-driven cycle-level modeling and parallelizes independent work
inside each cycle. These sources reinforce the current SM86 rule: event
skipping requires typed state/phase closure and dense differential evidence;
framework fast-forwarding alone is not a cycle-accuracy proof.
FlashGPU-Sim's development notes further constrain parallel acceleration: only
cluster-scoped work is dispatched concurrently, while shared counters, memory
maps, and cross-SM state use explicit atomics, locks, or completion-boundary
aggregation. This supports retaining the simulator's total `(time, sequence)`
ordering for shared event state rather than treating equal timestamps as
implicitly independent.

The latest adversarial schema pass found and closed one typed-boundary hole:
`L1BoundaryState::validate` now rejects an unknown latched `fill_timing` value
even when the cache has no live fills. Without this check, a malformed empty
snapshot could claim completeness and fail only when its next access consulted
the executable timing latch. The regression is covered by the 67 semantic
tests; the affected SM86 memory and scheduler suites remain green at 139 and
434 tests, and the rebuilt retained 8-small/7-large event/dense replay still
reports byte-identical outputs for all 15 workloads.
The complete Cargo workspace/all-targets run also passed, including the
456-test SM suite, direct-run and campaign targets, and the trace reader
targets; machine-local hardware tests remain explicitly ignored when their
capture paths are absent.

The next typed-state audit found a cross-field L2 discipline hole. Complete
boundaries now reject generation-bearing resident ways under `Immediate` or
unknown discipline, and reject generation-free resident ways under `Event`;
the check runs after mask, placement, and ownership validation so existing
typed error precedence remains stable. Incomplete diagnostic projections may
still retain an unknown discipline and reify. The new regression passes with
68 semantic tests, the 139-memory/434-reference-SM suites remain green, and
the rebuilt SM86 canary still has identical dense/event outputs at terminal
cycle 785.

The follow-up audit also rejects a complete active L2 snapshot whose access
discipline is absent, including generation-free resident ways that could
otherwise masquerade as an immediate cache. Fresh empty snapshots remain
valid, and incomplete diagnostic snapshots still reify. The 68-test semantic
suite, 139-memory/434-reference-SM suites, release canary, and all 15 retained
small/large dense-event replays pass after this check.

The exact-contract architecture audit now binds `CycleExactMemoryV2` and
`CycleExactPowerV2` to the current `sm_86` typed topology. SM120 or unknown
L2 identities remain valid diagnostic projections but cannot borrow an SM86
cycle-exact claim. The regression passes with 69 semantic tests; native SM86
memory/SM suites, release build, canary, and all 15 retained dense/event
replays remain green.

The direct typed-contract audit also rejects duplicate SM identities before an
exact claim can be returned. Root boundary validation already enforced the
canonical roster, but direct contract callers could previously pass two valid
cores with the same identity. The new regression passes with 69 semantic
tests, and the rebuilt SM86 canary remains dense/event-identical at cycle 785.

Direct contract validation now also requires canonical increasing core identity
order, matching root event-target indexing. Reversed but individually valid
core vectors can no longer obtain an exact result through the direct API; the
69-test semantic suite and rebuilt SM86 canary remain green.

The semantic round-trip audit also closed a schema/adapter mismatch: a
non-constant memory family could previously claim that its dynamic memory
class was absent, even though `to_instruction` refused to serialize that
combination. `SemanticEvent::validate` now enforces the static-only exception
for constant-cache events; the semantic/trace suites pass at 70 and 132 tests,
and the rebuilt SM86 canary remains dense/event-identical at cycle 785.
The retained Conv2D capture then completed a native-versus-semantic-round-trip
comparison with identical counters, timing, and coverage (219,008 instruction
events; 2,176 explicit static constant-class gaps).
The larger SwiGLU capture also round-tripped identically at 1,835,008
instructions with 98,304 explicit static constant-class gaps.

The deterministic causal-frontier sweep now explicitly exercises both sides of
the optimization boundary: 256 seeded descriptor cases produced 190 safe
summary regions and 439 reified regions under complete residency, while
unknown or fully contended residency was required to execute zero summaries.
This closes a vacuous test-coverage gap without changing native replay timing.
The release `causal-frontier-probe` now demonstrates the same path: 1 of 3
candidate regions collapses (4 instructions), while unknown and extra-resident
cases reify all regions; every case preserves final cycle 9 and terminal state.

Static constant-cache semantic events now also require an explicit load
direction, matching the serializer's omission of dynamic direction metadata
and the adapter's reconstruction rule. The new regression preserves
round-trip meaning for constant events without constraining dynamic families.
They also now require the explicit `dynamic_memory_class_unavailable` loss
reason, preventing a forged static-only event from claiming complete
classification after serialization.
Loss reasons are now required to be strictly sorted and unique as well, so
equivalent semantic events have deterministic serialized evidence.
Dynamic events with missing direction or active-lane address payload now also
require their explicit `memory_direction_unavailable` or
`address_payload_unavailable` loss reason, preventing incomplete metadata from
being silently treated as complete classification.
Dynamic operands with absent address-space metadata likewise require the
explicit `address_space_unavailable` loss reason; the retained Conv2D and
SwiGLU round trips continue to match native replay outputs.
The retained Conv2D round trip still matches native counters, timing, and
coverage across 219,008 instructions; its only loss category remains the
2,176 explicit static constant-class gaps.
Lane-address evidence is likewise required in ascending lane order; duplicate
lanes were already rejected, and the new order guard prevents equivalent
address payloads from producing different canonical event bytes.
The L2 discipline guard deliberately does not infer activity from the L2 clock
alone: boundary fixtures may carry the global boundary timestamp while no L2
mutation has occurred.

The latest power-accounting audit includes `memory_write_ack` and
`atomic_completion` in the typed completion/memory activity totals. The 77-test
semantics suite, full workspace targets, release canary, and all 15 retained
SM86 dense/event replays remain green and cycle-identical.

The async-state audit now applies the core-cycle temporal check to open groups
as well as committed groups. A forged open group cannot carry a future
`committed_at` timestamp while claiming a complete boundary; the 70-test
semantic suite covers this case.

A primary-source refresh of [Accel-Sim 2.0](https://arxiv.org/abs/2608.22602)
adds an important adversarial constraint: its evaluation reports large cycle
and memory-hierarchy errors when legacy simulation substitutes for unsupported
TMA/mbarrier behavior, and notes that trace-time asynchronous polling loops can
be inflated and unrepresentative. This supports the current rule that
cycle-preserving skips must retain asynchronous state and reify timing-sensitive
polling or synchronization regions; aggregate trace compression cannot justify
dropping those transitions.

The September 2026 [FlashGPU-sim preprint](https://arxiv.org/abs/2609.15311)
provides a complementary current baseline: it combines cycle-accurate modern
async/TMA/mbarrier modeling with deterministic host multithreading and reports
5.24% cycle MAPE across 131 configurations. Its host-parallel execution is a
separate acceleration axis from this project's conservative SM86 event
skipping; neither permits skipping a frontier without retaining the state that
can affect issue, completion, or synchronization order.

## Run the live path

```bash
cargo build --release --manifest-path sim/Cargo.toml -p sim-run
```

Use the retained NVBit trace and control sidecars from a current capture with
`sim/target/release/sim-run`; the exact command form is in
[runbook_host.md](runbook_host.md). For a quick numerical comparison:

```bash
python -m eval.direct_score --observed observed.json --predicted predicted.json
```

## Next engineering slice

1. Keep the current SM86 scheduler and memory changes small and measurable.
2. Replay one representative current trace after each change.
3. Profile event throughput and allocation cost when a replay exposes a real
   bottleneck.
4. Keep new behavior SM86-scoped until a separate architecture scope is
   explicitly reopened.
