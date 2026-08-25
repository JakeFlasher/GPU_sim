# GPU simulator notes corpus: research synthesis and milestone decision

**Study date:** 2026-08-17  
**Research corpus:** `/home/jakeshea/GPU_sims_notes`  
**Implementation compared:** `/home/jakeshea/GPU_simulator_rebase`

## Executive result

The notes corpus and the current live campaign point to the same next
engineering problem: preserve and expose the causal handoff from memory
completion to warp readiness, scheduler selection, and CTA synchronization.

The simulator already models the individual ingredients needed for this work:

- compiler-encoded stall and dependence-counter controls;
- asynchronous-copy groups and their source-completion timestamps;
- exact cache-all LDGSTS destination wavefront work;
- one aggregate per-SM LDGSTS destination-service queue;
- warp eligibility and tensor-pipeline availability; and
- CTA barrier arrival and release.

The scheduler currently reduces those ingredients to one optional earliest
issue time. That is sufficient to advance the event-driven actor, but it loses
which relation was limiting, how much latency was hidden by other work, how
widely CTA warps arrived at a barrier, and whether a ready warp merely lost a
scheduler choice. The loss of this decomposition is now the most important
obstacle to diagnosing the remaining cycle residual.

The configured next milestone is therefore **sm_86 causal warp readiness and
async-copy/CTA-barrier handoff**. Its implementation and live experiment are
specified in [sm86_causal_readiness_milestone.md](sm86_causal_readiness_milestone.md).

This decision deliberately does not introduce a fitted per-workload constant
or a generic hardware register scoreboard. Modern NVIDIA scheduling in the
studied Ampere path uses compiler-controlled dependence counters. A CTA barrier
is also a rendezvous among warps, not another name for a dependence-counter
latency. The controlled experiment must distinguish these mechanisms before a
new architecture parameter can be justified.

### Post-study execution result

That selected milestone is now complete. The simulator gained the causal
readiness decomposition, and the configured 36-cell experiment ran in full:
112 NCU profiles, seven SourceCounters profiles, 36 separate NVBit captures,
and deterministic replay of all cells. The result falsified a fixed CTA
barrier latency. Barrier-only cost rises from 14.00 cycles/group at W2 to 18.00
at W4 and 28.00 at W8; the DRAM async/barrier interaction rises from +1.78 to
+72.97 to +230.53 cycles/group, while the L1 interaction changes sign.

The study's component ordering is therefore refined, not reversed. The next
milestone is
[barrier pipeline and launch-chain cache handoff](sm86_barrier_pipeline_cache_handoff_milestone.md):
carry traced prewarm L2 state across a declared sequential launch boundary,
then characterize barrier issue throughput separately from rendezvous before
multi-CTA and broader issue-resource work.

### Post-successor execution result

That successor is also complete. Transactional stream-L2 replay now separates
traced L2 and DRAM target routes. A 33-cell barrier-only campaign selects one
SM-shared barrier-arrival service at two cycles per warp with 0.091
cycles/generation held-out width MAE. Coupled W2/W4/W8 route tests and
fixed-eight-warp multi-CTA shapes transfer without a fixed CTA release,
generation re-arm, route, or CTA-count constant. The production cohort remains
fully covered at 33.394% active-SM MAPE.

The corpus recommendation and live residual now meet at the next layer:
[sm_86 execution-port and control-pipeline closure](sm86_issue_port_control_pipeline_milestone.md).
Dependent ALU latency, port throughput, branch/control service, and compatible
dual issue are isolated before revisiting the retained b20, skew, multi-CTA,
and async magnitude residuals.

### Post-execution-port result

That campaign is complete as well. Compiler controls already predict dependent
scalar length slopes, every NCU issue ratio remains 1.0, and simple ALU/FMA
initiation candidates fail cross-protocol transfer. No scalar interval or dual
issue is promoted. One narrow branch relation survives: the exact predicated
forward-BRA form needs a ten-cycle minimum next-issue frontier, reducing branch
slope MAE from 5.033 to 0.033 and production MAPE to 33.128% without changing
traffic.

The residual is now localized to the scheduler: only W8 places a second warp on
each modeled SMSP, and only W8 exposes matched math-pipe-throttle and
not-selected delay. The configured successor is
[SMSP scheduler and math-pipe contention](sm86_scheduler_math_pipe_contention_milestone.md).

### Post-scheduler-contention result

That 99-cell successor is complete. At W12/W16, active time approaches a
two-cycle half-warp executed-pipe bound within 1.26 cycles/generation for ALU
and 4.41 for bare IMAD. Ten direct-port, FIFO/oldest, rational, queue, and
service-drain candidates were tested. The best fits W8 same-family and phase
response, but changes W4 no-contention behavior, misses W12 magnitude, and
reduces reversed-family sign transfer from 13/18 to 10/18. No scheduler/port
relation enters production, and restored replay is byte-identical.

The failed coupling matches the more detailed mechanism in the local Huerta
et al. paper: execution availability is not an immediate issue timestamp;
fixed-latency work crosses Control and Allocate stages, and Allocate reserves
register-file reads plus future execution capacity. The configured successor
is therefore
[fixed-latency Control/Allocate and register-file contention](sm86_control_allocate_register_file_milestone.md),
beginning with a trusted SASS operand/register-width/reuse projection.

## Corpus scope and study method

The full notes tree occupies about 519 MB on disk. It contains an imported
Vortex source/toolchain checkout as well as authored research material. To
avoid counting the imported checkout as original notes, the primary census
excluded `gpu_sim_breakthrough_research/toolchains/vortex_v3_source`:

| Material | Count |
|---|---:|
| Authored/indexed research files | 170 |
| Markdown research documents | 23 documents, 6,223 lines |
| PDF papers | 127 documents, 2,616 pages |
| Unreadable PDFs in the structural scan | 0 |
| Text files | 15 |
| CSV catalogs | 3 |
| Other retained archive | 1 ZIP and 1 extensionless file |

All 23 authored Markdown documents were structurally indexed. The central
reports, their conclusions, comparison tables, proposed representations,
evaluation designs, and limitations were read directly. All 127 PDFs were
checked for readable metadata/page structure; the papers most relevant to the
current implementation boundary were then read directly. Source catalogs and
annotated bibliographies were used to route the remainder. This is a corpus
study, not a claim that every one of the 2,616 paper pages was read linearly.

The imported Vortex checkout was treated as a possible future execution oracle,
not as evidence about the current simulator and not as a target for changes.

## Research tracks and what they contribute now

### 1. Current-simulator presentation and novelty analysis

The two presentation reports accurately describe the foundation that led to
the present simulator:

- enhanced NVBit records are reconstructed into true dynamic warp
  instructions;
- lane addresses are coalesced into hierarchy transactions;
- static SASS control information is joined to dynamic records;
- actor-local event execution preserves causal ordering while skipping idle
  time;
- private L1 state feeds a shared L2/DRAM system; and
- exact instruction cardinality is a trace-integrity result, not timing
  validation.

Their strongest defensible contribution is the integration of lane-level
dynamic data, compiler controls, event-driven memory completion, and wake-up.
Their strongest methodological recommendation is to isolate one mechanism at a
time and validate its direction and sensitivity rather than tuning bundles of
constants.

Those reports are now a historical snapshot in one material respect. They say
that hardware-equivalent cycles are not yet emitted. The current repository has
since implemented and live-tested a per-SM issue-frontier cycle predictor,
async-copy timing, tensor initiation, L1/L2/DRAM response timing, and exact
destination service. The 2026-08-18 launch-cycle successor now resolves the
remaining semantic distinction: per-kernel GPC elapsed wraps active work in a
profiler envelope, final store/RED drain is not yet represented, and NCU range
time includes host/front-end gaps absent from SASS traces. Replay therefore
exports a typed complete/incomplete retirement frontier while numerical
`gpc__cycles_elapsed.max` remains unavailable pending real drain and envelope
models. Two immediate successors then reject fixed retirement and L1TEX rates
across depth, concurrency, CTA shape, arrival skew, and line topology. Schema 6
retains exact instructions, sectors, ideal wavefronts, unique 128-byte lines,
atomic requests, and issue spans; production timing remains unchanged until a
candidate evaluated at actual arrival timestamps transfers. That bounded
schema-7 evaluator is now implemented and retrospectively identifies an
SM-shared line-service relation, while rejecting independent SMSP scope. The
local notes' warning about finite lower-memory queues and scheduler/memory
feedback remains decisive: most live STORE/RED cost is SM-active, so the next
prospective milestone must identify admission capacity and issue backpressure
before the candidate can become a completion event.

That prospective milestone is now complete. Static controls and SourceCounters
identify STG/RED read-dependence barriers as operand-consumption/admission
frontiers, while returning ATOMG uses a result write barrier. A 100-cell reuse
discriminator proves that identical dynamic L1TEX/LTS work can differ by more
than 25,000 cycles when L2 tags change from cold to resident. Schema 8 and
direct L2 source hit/miss output retain the exact causal inputs; timing remains
inactive pending bounded SM-shared versus per-SMSP admission candidates feeding
an SM service.

### 2. Locality and trace-synthesis research

The synthesis archive's most useful conclusion for this project is not a new
trace generator. It is the warning that timing is a relation-preservation
problem.

Reuse distance, footprints, phase models, and transition models can reproduce
selected marginal statistics while omitting the dependence and feedback that
control execution. On a GPU, a completion changes warp readiness; readiness
changes scheduler order; scheduler order changes memory interleaving and CTA
rendezvous. A pre-merged trace can silently freeze those decisions.

Applied to the present simulator, this says:

1. retain actor-local order and compiler dependence relations;
2. keep scheduler and coalescer decisions in the executable model;
3. represent completion conditions explicitly rather than only their final
   maximum; and
4. validate a mechanism with held-out consumers, not only with a matching
   trace statistic.

This directly supports a causal readiness layer. It does not justify pivoting
the current milestone into synthetic-trace generation.

### 3. Counterfactual replay research

The CounterTrace-GPU material is a proposal, not an implemented result. Its
central warning is nevertheless relevant: a trace records one realized path,
set of masks, addresses, placement, and schedule; it does not automatically
describe what would have happened under a counterfactual architecture.

The proposal identifies the same dangerous cross-layer chain seen in the live
residual:

```text
memory completion -> warp readiness -> scheduler order
                  -> synchronization/atomic ordering -> later execution
```

That supports keeping the present milestone narrowly within measured sm_86
behavior. It also limits the claim: the new diagnostics will explain the
simulator's causal timing decomposition for a fixed trace. They will not prove
that arbitrary architecture changes leave the trace valid.

The current frontier-study documents explicitly exclude trace-driven,
event-driven, broad async coverage, or exact trace accounting as standalone
novelty claims. Its fresh survey, candidate portfolio, and pilot remain empty
scaffolds, so they add no newer experimental result that supersedes the active
simulator roadmap.

### 4. Primary GPU-simulation and characterization papers

The most consequential primary-source findings are:

- **Huerta et al., MICRO 2025:** modern NVIDIA dependencies are represented by
  compiler-controlled stall and dependence counters rather than a traditional
  hardware register scoreboard. The described model has six per-warp
  dependence counters, one-cycle visibility details, subcore issue behavior,
  and a measured accuracy benefit from modeling true dependencies.
- **NVArchSim:** crude load blocking and omitted dependency tracking produce
  large timing errors; true dependency tracking is both more accurate and can
  expose more independent progress.
- **Sim-FA:** omitting asynchronous barrier and wake-up interactions destroys
  the overlap among memory movement, compute, and activation work.
- **Accel-Sim and later memory-model work:** correlation against silicon needs
  architecture-local timing plus workload breadth; agreement on instruction or
  traffic counts is necessary but not sufficient for cycle accuracy.
- **FlashGPU-sim (MICRO 2026 draft):** modern AI timing depends on the
  interaction among asynchronous transfer, hardware synchronization, and
  tensor execution. Its reported end-to-end accuracy makes this interaction a
  direct contemporary comparison point, while its execution-driven front end
  is a different design choice from this repository's enhanced trace replay.
- **CuLifter (MICRO 2026 draft):** `LDGSTS`, `LDGDEPBAR`, `DEPBAR`, and
  `BAR.SYNC` form a semantic completion chain; treating the instructions as
  isolated operations loses structure. Typed binary lifting is promising
  future front-end work, but the current trace/control join already preserves
  enough of this chain to implement the readiness milestone.
- **MMAscope and bit-accurate MMA modeling (MICRO 2026 drafts):** Tensor Core
  arithmetic has important hidden precision and reduction semantics. That is a
  future functional-fidelity component, distinct from the present cycle and
  readiness residual.

The corpus therefore supports a component order: first make completion and
readiness causality explicit, then calibrate issue-resource competition, then
extend functional arithmetic or architecture scope.

## Reconciliation with the current implementation

The previous notes and current code do not disagree; the implementation has
advanced beyond the notes' timestamp. The resulting gap map is:

| Mechanism | Current state | Evidence or limitation |
|---|---|---|
| Dynamic instruction and lane-address reconstruction | Implemented | Exact replay cardinality and fresh full NVBit captures |
| Compiler stall and dependence controls | Implemented | Six indexed counters in the replay path; no generic scoreboard |
| Event-driven actor progression | Implemented | SM wake events and local issue advancement |
| L1/L2/DRAM source completion | Implemented | Architecture-card latencies and shared-system replies |
| Async-copy group source completion | Implemented | Pending-source count and maximum source timestamp |
| Exact LDGSTS destination work | Implemented | Destination wavefront projection matches live NCU traffic |
| Aggregate LDGSTS destination service | Implemented for sm_86 | Independent W1-W8 probe selected 2 wavefronts / 3 cycles |
| Tensor initiation resource | Implemented | Per-subcore tensor pipe readiness |
| CTA barrier rendezvous | Structurally implemented | Arrival flag/count and release at the final arrival plus one cycle |
| Causal readiness decomposition | Missing | All readiness arms collapse to `max(...)` or `None` |
| Barrier-generation/arrival-spread diagnostics | Missing | Repeated barriers have no public generation or latency breakdown |
| Eligible-but-not-selected accounting | Missing | Scheduler picks a warp but does not retain losing-ready exposure |
| Production-card issue resources | Incomplete | Typed schema exists; `sm_86.json` still lacks the issue-resource block |
| Dual issue and broader pipe contention | Missing/typed gap | Must follow readiness attribution to avoid confounding |
| Counterfactual trace validity | Out of current scope | Fixed-trace prediction cannot certify arbitrary architecture changes |
| sm_120 timing | Explicit gap | Ampere constants must not be inherited |

The precise implementation boundary is visible in `sim/sm/src/issue.rs`:

- `AsyncCopyGroup` retains source, destination, and final completion times;
- `WarpState` retains `eligible`, dependence-counter clear/pending state, and
  `at_block_barrier`;
- `issue_ready_at` takes the maximum of control/dependence readiness, async wait
  completion, and tensor-pipeline availability; and
- CTA release clears every waiting warp when `waiting == unfinished`.

That code has enough state to expose the missing causal decomposition without
replacing the event engine or changing trace format.

## Live evidence selecting the milestone

The 2026-08-17 full campaign contains six fresh NCU/NVBit workloads. The final
aggregate-destination model predicts 1,092 / 8,906 / 8,906 / 5,846 / 2,365 /
12,448 active-SM cycles against live medians of 2,245 / 12,265 / 12,434 / 9,722 /
3,891 / 14,510. Its six-workload active-SM MAPE is 33.403%, improved from
35.145% before aggregate LDGSTS service. Traffic and instruction cardinality
remain exact for the modeled surfaces.

The SourceCounters exports localize long-scoreboard samples as follows:

| Workload | All long-scoreboard samples | Samples on post-wait `BAR.SYNC` |
|---|---:|---:|
| BF16 HMMA | 349 | 349 |
| FP16 HMMA | 320 | 320 |
| INT8 IMMA | 231 | 219 |
| Sparse BF16 HMMA | 62 | 62 |
| BF16 attention | 53 | 53 |

The remaining 12 INT8 samples occur on dependent integer additions. The
canary is too short for useful source sampling.

NCU defines long-scoreboard stalls as waits on an L1TEX dependency and barrier
stalls as waits for sibling warps at a CTA barrier. Source attribution can land
on the active dependent instruction rather than the producer. Therefore the
table is consistent with an L1TEX dependency becoming visible at the
post-async-wait barrier instruction; it does **not** prove that the CTA barrier
itself has a large fixed execution latency. That ambiguity is exactly what the
next probe must resolve.

## Priority ranking

The candidates were ranked by direct live evidence, number of later components
they unblock, workload breadth, and whether the current implementation already
contains the required state.

| Candidate component | Evidence | Downstream leverage | Implementation readiness | Decision |
|---|---:|---:|---:|---|
| Causal readiness and async-to-CTA handoff | High | Very high | High | **Next milestone** |
| Issue ports, pipe competition, and dual issue | Medium | High | Medium | Immediately after readiness |
| CTA occupancy and block-overlap validation | Medium | High | Medium | After issue attribution |
| Atomic serialization and returning-atomic completion | Medium-low in current cohort | Medium | Medium | Add targeted cohort after barrier work |
| Tensor Core bit-accurate arithmetic | Strong external evidence, different target | Medium | Low | Future functional milestone |
| Synthetic/counterfactual trace contracts | Research proposal, not current residual | Long-term | Low | Separate research program |
| sm_120 extension | No architecture-local live evidence yet | High eventually | Low | After sm_86 closure |

Readiness attribution must precede dual issue. Otherwise an idle cycle caused by
an unresolved dependency can be mistaken for missing issue bandwidth, while a
ready warp that loses arbitration can be mistaken for a memory stall.

## Selected hypothesis and falsification conditions

The milestone tests four concrete hypotheses:

1. Increasing async source latency while destination work and CTA shape remain
   fixed increases the source/dependence component of readiness delay.
2. Adding a CTA barrier after the async wait creates a non-additive delay only
   when warp arrival spread or producer/consumer asymmetry matters.
3. Independent work between commit and wait hides a measurable portion of
   source latency; a model that charges all source latency serially will miss
   this slope.
4. Once source, destination, rendezvous, and scheduler exposure are separated,
   at least one dominant component of the six-workload residual becomes
   mechanically explainable without a per-workload fit.

Possible falsification is useful:

- If the async-plus-barrier interaction is additive and the modeled arrival
  spread already matches hardware, no new CTA latency parameter is added.
- If locality changes NCU traffic but not cycles, the source arm is already
  hidden by overlap and the next target becomes scheduler/pipe competition.
- If the live source hierarchy does not produce stable L1/L2/DRAM separation,
  locality labels are rejected and the probe is redesigned from measured
  traffic rather than assumed cache residency.
- If causal diagnostics explain simulator time but do not improve held-out
  ordering, they remain diagnostics; no speculative timing correction is
  promoted.

## Consequences for the following milestones

After causal readiness is implemented and measured, the recommended order is:

1. **Issue-resource closure:** populate only independently supported sm_86
   issue-resource fields, then model eligible warp competition, issue ports,
   and any demonstrated dual-issue behavior.
2. **CTA residency and synchronization breadth:** validate block overlap,
   repeated barrier generations, divergent completion, and shared-memory
   capacity effects across CTA shapes.
3. **Atomic completion and serialization:** add returning-atomic dependencies,
   hot-line contention, and ordering probes to the live cohort.
4. **Performance engineering:** profile the now-stable event/state paths and
   optimize allocations, scans, and queue operations without changing results.
5. **Blackwell and advanced arithmetic:** capture sm_120 evidence independently;
   treat typed SASS IR and bit-accurate MMA behavior as separate front-end and
   functional-fidelity projects.

## Later front-end evidence addendum

The later `fixed_latency_front_end_footprint` campaign resolves one item that
was only a literature hypothesis in the initial synthesis. Across 96 live
U1/U2/U4/U8 cells, direct Nsight counters measure 3.013–3.167
branch-resolving warp-cycles per uniform backward target, while exact target
count scales as `1024/U × warps` and divergent targets remain zero. This
supports a typed three-cycle sm_86 backward-target capability. It does not
resolve IBuffer depth: the best direct-counter candidate selects one entry,
conflicting with Huerta et al.'s three-entry reference and failing several
timing phases. A later branch-free CLOCK reconstruction improves the candidate
to capacity two once in-flight fetches reserve entries, but the absolute and
treatment fits require incompatible instruction-cache parameters and reproduce
less than 15% of exact event order. The reference capacity therefore remains
useful prior art rather than an imported architecture constant.

## Later atomic evidence addendum

Three live atomic campaigns resolve the initial medium-low evidence ranking.
Exact ATOMG/RED sector formation and L2 request topology are now replay inputs,
and readiness schema 5 exposes exact CLOCKLO issue intervals. Partial alias
degrees prove that sector count alone is insufficient; in one SM, RED separates
into L2-sector and L2-request work plus a full-broadcast fast path. Multi-SM
traffic then rejects a per-SM queue and every retained ordinary-cache hash as
the distributed atomic-unit map. The best cache-hash candidate misses both its
fine-spacing training and block-count transfer. Atomic timing remains inactive
after a pairwise graph identifies an exact 256-byte contention region but
falsifies the premise that many-region service is one retained fixed hash. This
is the same research discipline recommended by the notes: preserve trace-visible
contracts, promote exact counts, and keep counterfactual timing explicit when
an address-dependent hidden resource is not identifiable.

## Principal local sources

- `/home/jakeshea/GPU_sims_notes/gpu_simulator_presentation/final_research_report.md`
- `/home/jakeshea/GPU_sims_notes/gpu_simulator_presentation/novelty_and_results_analysis.md`
- `/home/jakeshea/GPU_sims_notes/trace_synthesis_research/trace_synthesis_research_notes.md`
- `/home/jakeshea/GPU_sims_notes/gpu_sim_breakthrough_research/final_research_direction.md`
- `/home/jakeshea/GPU_sims_notes/gpu_sim_breakthrough_research/candidate_portfolio.md`
- `/home/jakeshea/GPU_sims_notes/gpu_sim_breakthrough_research/source_catalog.md`
- `/home/jakeshea/GPU_sims_notes/gpu_simulation_frontier_study/prior_art/existing_work_exclusions.md`
- `/home/jakeshea/GPU_sims_notes/research_papers/05_huerta_modern_gpu_cores_micro2025.pdf`
- `/home/jakeshea/GPU_sims_notes/micro26/micro2026ae-paper261.pdf`
- `/home/jakeshea/GPU_sims_notes/micro26/micro2026ae-paper2617.pdf`
- `/home/jakeshea/GPU_sims_notes/micro26/micro2026ae-paper3220.pdf`
- `/home/jakeshea/GPU_sims_notes/micro26/micro2026ae-paper776.pdf`
- `results/captures/sm86_cycle_residual_20260817_v1/analysis/README.md`

## Official semantic references

- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [Nsight Compute documentation](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
- [CUDA asynchronous-copy API](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html)
- [CUDA cooperative groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
