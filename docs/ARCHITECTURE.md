# Architecture

## Objective

Predict GPU execution behavior from program structure and dynamic NVBit traces
without feeding measured NCU target values into the simulator.

## Components

```text
dynamic_trace.pb
      |
      v
 sim-trace ------ static SASS/control data
      |                     |
      +----------+----------+
                 v
              sim-sm <---- arch profile
                 |
                 v
              sim-mem
                 |
                 v
             sim-engine
                 |
                 v
              sim-run ----> metric predictions
```

- `sim-engine` provides deterministic time-ordered events.
- `sim-trace` decodes the repository's enhanced NVBit protobuf.
- `sim-sm` models warp issue, compiler dependencies, exact forward-branch
  resolution, typed three-cycle uniform backward-target resolution, CTA
  rendezvous, the architecture-local SM-shared barrier-arrival
  service, replay, and instruction progress. Scalar math pipes remain
  dispatch-only: a 99-cell live campaign identified high-occupancy service
  capacity but rejected direct scheduler-port and queue placements. Static
  fixed-latency instructions now carry trusted exact opcode, register file,
  register span, unary modifier, source position, reuse, and sm_86 bank-demand
  projection. It is diagnostic only: two exact-bank campaigns and five staged
  completion candidates failed complete held-out/scheduler transfer, so
  Control/Allocate timing remains inactive. Exact RFC treatments support a
  shared bank/source-position structure tagged by warp/register, but active RFC
  timing fails predecessor transfer and is also inactive. Exact HADD2
  projection and three completion campaigns are retained. Balanced order
  proves an opposite-bank null; a compiler-native same-bank pair independently
  identifies timing-neutral accommodation/bypass for two same-cycle results at
  W4/W8. A 7/8/9 reservation campaign exposes a W8 ninth-reservation canary,
  but the reference capacity-8/pop-1 implementation has no replay effect and
  is rejected. Queue entries and pop width remain unknown. A subsequent exact
  load/HADD2 campaign promotes one narrow result relation: a same-cycle,
  same-SMSP, same-bank HADD2 writeback wins over a barrier-observed scalar
  sm_86 `LDG.E.STRONG.SM` L1 hit and defers load visibility by one cycle.
  Readiness schema 4 reports those collisions; W8+ generic shift timing remains
  separate. A direct correctness campaign also identifies fixed-consumer
  bypass at distance 4 versus global-variable register availability at distance
  5. Exact `IMAD.IADD` projection is retained, but no scoreboard/timing repair
  is applied to invalid compiler schedules because replay does not execute
  register values.
  A dense W4–W16 follow-up localizes the remaining one-cycle producer-shift
  residual to a per-SMSP instruction-frontier phase beginning at W5. Ten
  reference-inspired fetch/decode/IBuffer candidates improve treatment MAE but
  fail cross-cohort/family transfer. A subsequent 96-cell U1/U2/U4/U8
  factorial directly measures three branch-resolving cycles per uniform
  backward target and promotes that architecture-local readiness fact. The
  compiler's five-cycle BRA control still binds issue, so the new state is
  diagnostic on the measured loops. Direct `no_instruction` counters reject a
  transferable IBuffer capacity; fetch/decode queue timing remains inactive
  after a branch-free independent-instruction successor. That successor's
  absolute and treatment objectives both prefer capacity two once in-flight
  requests reserve entries, but require incompatible instruction-cache
  parameters and reproduce fewer than 15% of exact CLOCK event order. Capacity
  two remains a research candidate, not active architecture state.
- `sim-trace` retains exact-address alias multiplicity for address-carrying
  global ATOMG/RED records while compacting lane addresses into 32-byte atomic
  transaction sectors. Same-word lane aliases therefore remain 32 serialized
  sectors instead of collapsing to one unique cache sector. `sim-sm` keeps
  returning ATOMG response completion distinct from non-returning RED and
  reconstructs exact L2 request topology from the compact sector stream.
  Compact global-atomic records also retain their validated allocation ID in a
  route-disjoint word, allowing checked allocation-relative sector projection
  without changing the 24-byte memory record or assigning an LTS class.
  Readiness schema 5 reports first-to-last CLOCKLO issue intervals. Two live
  adversarial campaigns reject operation-only sector rates, a per-SM-only
  atomic queue, portable spacing rules, and all retained ordinary-cache hashes
  as atomic-unit maps. A 544-cell pairwise oracle promotes one timing-free sm_86
  fact: cross-warp/CTA atomics contend exactly within an aligned 256-byte
  region. Distributed many-region service timing remains inactive because its
  shared scheduling network is still unidentified. A successor now provides
  exact weighted and generation-coded 48-class LTS collision oracles, but a
  65,535-line fit and 32,768-line high-address holdout reject every virtual-
  address hash. A CUDA-VMM successor maps one physical handle through seven
  aliases and shows exact sequential-RED class invariance across all proxies:
  absolute class follows physical placement. Its full 2 MiB relative-offset
  map transfers across VMM and `cuMemAlloc` up to class permutation. Extended
  evidence shows one permutation per physical granule, not one universal
  allocation-relative function: eight distinct handles produce eight stable
  maps. Allocation ID is retained as identity, but no LTS hash or atomic timing
  service is active without physical phase.
- `sim-mem` models sectored L1, shared L2, MSHRs, and DRAM timing. It preserves
  literal cache-operation tokens while routing the CUDA-13.2 sm_86
  `LDG.E.STRONG.SM` (`ld.global.ca`) form through L1; Blackwell keeps its
  architecture-specific strong-scope behavior.
- `sim-arch` loads device-specific geometry, timing assumptions, and the
  timing-free sm_86 aligned-256-byte cross-warp atomic contention topology.
- `sim-run` from `sim/direct_run/` composes the libraries into the default
  direct application and emits predictions.
- `sim-run-campaign` in `sim/run/` retains the historical manifest/evidence
  interface and is invoked only for an explicit campaign task.

The implementation is event-driven: it should not introduce a global
per-cycle scan or one queued event per instruction when a local actor loop can
advance multiple instructions.

The shared-system driver similarly compresses only physical representation:
consecutive same-SM/same-time reply-less STORE/RED requests emitted by one actor
advance share one queue entry, then execute their existing L2 transactions in
original order. Logical event diagnostics remain per request. Loads, responses,
fills, ATOMG, mixed timestamps, and timing-active feedback retain individual
entries, so the `(time, insertion-sequence)` relation across actors is unchanged.

Direct replay is cold per launch by default. An explicit same-device,
same-stream session may retain only fully drained L2 replacement/sector state;
per-SM L1, MSHR, pipeline, CTA, scheduler, event, and DRAM-server state is
rebuilt at every launch boundary.

Direct replay also releases immutable raw threadblock protobuf bytes after the
complete validation/compaction pass. Replay consumes only compact validated
blocks. Raw-retaining direct APIs and campaign snapshots remain distinct and
never reopen a released path implicitly.

Validation-proven function and threadblock-warp identity is stored once per
compact warp rather than on every dynamic record. Per-record SM and physical
warp identity remains an exact signed i32 plus explicit presence bits, reducing
`ValidatedInstruction` from 32 to 20 bytes without a sentinel domain hole. A
bounded descriptor-relative roster can alternatively read and compact one
exact file generation at a time in performance builds; its dominant-case gain
was insufficient for default use.

Timing has a separate completion contract. Replay retains the maximum local
issue frontier and the last processed shared-system event, then reports their
maximum through `ModeledCompletionFrontier`. When dirty writeback, coherence,
reduction RMW, or legacy async visibility remains outside the event stream, the
frontier is typed as an incomplete lower bound. It is not converted to
`gpc__cycles_elapsed.max`: live GA102 measurements show an external per-grid
profiler envelope, and multi-launch range time additionally includes untraced
host/front-end gaps.

Readiness schema 6 separately reports scalar-global active warp instructions,
alias-aware sectors, ideal L1TEX wavefronts, unique 128-byte lines, exact
atomic L2 requests, and issue spans. “Ideal” is
intentional: live saturation can emit additional output wavefronts from
internal replay, and system-fence epilogues can generate hidden traffic not
represented by source STG addresses. These diagnostics do not yet gate issue
or create retirement events.

Readiness schema 7 is an explicit research-only extension. A caller may install
at most 16 fixed-capacity streaming candidates, each filtered by global-memory
operation and evaluated on either one SM-shared or four SMSP-local service
timelines. Updates occur at the real issue-cycle transaction boundary and
retain only aggregate work, busy/queue exposure, and terminal maxima. They add
no event, readiness edge, issue delay, or completion claim. Ordinary replay has
an empty candidate set and exits this path before address-topology projection.

Readiness schema 8 separates compiler operand-consumption barriers from the
generic fixed read-dependence bucket. Scalar-global LOAD, STORE, ATOMIC, and
RED producers are counted by operation; any consumer delay on their encoded
read barrier is attributed to a distinct cause. This is not a new latency:
ordinary replay still uses the existing fixed clear. The distinction exists so
future bounded SM-shared/per-SMSP admission candidates can replace only the
identified relation.

Full-hierarchy direct JSON also exposes the already-validated `L2SourceSplit`
(L1 fill, bypass, untagged, and store lanes), each with sectors and hits. This
adds no cache observer; it makes exact cold/tag-hit provenance available to
admission-service research.

Readiness schema 9 evaluates up to 16 finite STORE/RED admission hypotheses.
Each candidate chooses an SM or per-SMSP waiting queue, an instruction
capacity, a fixed operand-visibility floor, and rational L2 hit/miss sector
rates feeding one SM-shared FIFO server. Queue slots release at service start;
same-cycle dequeue/admission is allowed. Exact per-instruction outcomes are
reconstructed from ordered STORE/RED L2 requests. This lane preserves observed
issue/cache order and reports admission and service frontiers only; it does not
park a warp or change a cache outcome.

Readiness schema 10 activates exactly one finite candidate. Candidate issue is
capacity-filtered across the four SMSP picks, reserves a queue entry, and yields
the SM so its emitted L2 requests are priced before the next wake. A full queue
extends warp readiness to the oldest known service-start release; an unpriced
front entry blocks until its L2 group completes. Later issue and shared-L2 order
therefore change executable state. A pick denied by shared capacity repicks
other ready work on the same SMSP. Candidate operand visibility must equal the
active read-barrier latency, and the full timing-active candidate participates
in replay-configuration identity. The lane is launch-scoped, full-coverage,
and mutually exclusive with observational candidate modes.

Admission candidates may additionally declare a fixed
`l2_miss_service_width_sectors`. Zero preserves fluid rational service; a
positive width charges complete miss-sector quanta, including a partial final
round. The width changes only candidate service work and remains explicit
opt-in. Rate/width transfer did not identify an architecture default.

Readiness schema 11 is a separate opt-in completion relation. Queue slots still
release at service start, but a warp's `MEMBAR.SC.GPU` waits for that warp's own
latest matching write-service completion. The completion frontier is per warp,
not the SM-global server tail. Three CLOCKLO reads expose issue and fence phases;
their uncommon timestamps remain in boxed feedback state.

Readiness schema 12 optionally adds a non-occupying
`write_completion_latency_cycles` after service done. It changes the per-warp
MEMBAR frontier but neither server occupancy nor queue release. Service and
write-completion terminal diagnostics remain distinct. A nonzero tail requires
schema-11 MEMBAR feedback and is rejected otherwise.

Readiness schema 13 optionally adds work-dependent write completion. Exact L2
request count and unique 256-byte-region count are retained per priced
instruction. Candidate request/region cycle rates feed one to 32 independent
completion lanes; each job starts after admission service and its selected
lane are both ready. This server changes only the per-warp MEMBAR completion
frontier: admission service, queue release, and cache order remain separate.
The fixed schema-12 tail, when present, follows lane service without occupying
the lane. Work without MEMBAR feedback, work without a valid lane count, and a
lane count without work are typed configuration errors.

## Measurement boundary

NCU is run separately from NVBit. NCU results are observed values used by
`eval/`; NVBit traces and static program data are simulator inputs. A measured
target must never be copied into an architecture card or replay configuration.

## Architecture scope

The first complete path is `sm_86`. Types for `sm_120` are useful for
portability work, but unknown Blackwell behavior should remain an explicit
missing model rather than silently inheriting Ampere constants.

## Files and paths

Commands accept ordinary filesystem paths. Missing, malformed, or inconsistent
inputs return useful errors. Inputs do not need workflow signatures, ownership
tokens, cryptographic identities, special file descriptors, or publication
state.
