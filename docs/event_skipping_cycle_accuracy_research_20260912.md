> **Historical research note (2026-09-24).** Retained prior-art and event-skipping analysis. It is not an active implementation plan, comparison gate, or claim of silicon cycle accuracy. Current scope and decisions live in `docs/STATUS.md` and `plans/cross_architecture_gpu_simulator_research_plan_20260923.md`.

# Event Skipping and Cycle Accuracy in GPU Simulators

**Scope.** This report evaluates whether event skipping is a sound and useful
design for cycle-accurate GPU architectural simulation. It covers the two local
trees, fresh source checkouts of MGPUSim, Accel-Sim and GPGPU-Sim, and a
representative set of recent papers and projects. It treats MI300X support as
out of scope until hardware or retained ROCm captures are available.

## Answer first

Event skipping makes sense, but the phrase describes several different
techniques that have very different accuracy claims.

1. **Conservative next-event time advancement is cycle-accurate in a modeled
   system.** If a component has no state transition, arrival, deadline,
   arbitration opportunity, or periodic action between cycles *t* and *t+k*,
   advancing the logical clock directly to the next known event at *t+k* is
   equivalent to executing those idle cycles. MGPUSim/Akita and NVArchSim use
   this idea. MGPUSim’s ISCA paper explicitly requires “no busy ticking” and
   gives a DRAM request that schedules a response 300 cycles later instead of
   updating the controller every intervening cycle.^1

2. **Selective smart ticking is a hybrid implementation choice.** Components
   that are idle sleep until a message or scheduled wake-up; active compute
   pipelines still execute their per-cycle phases. This is the practical shape
   of MGPUSim. It preserves the detailed model’s event order while avoiding
   polling of inert memory/interconnect components.

3. **Region summarization, temporal decoupling, and warp/kernel sampling are
   approximations unless they emit a state-complete equivalent and prove their
   observation contract.** Replacing many instructions with one predicted
   duration, or replaying a sampled wavefront, can be very useful for speed,
   but it is not automatically cycle-accurate. Photon’s sampled MGPUSim path
   predicts wavefront completion after a stable behavior window and bypasses
   detailed execution; its paper reports non-zero error and treats sampling as
   a fidelity trade-off.^2

The local proposal is therefore sensible only when framed as a **conservative
event-driven kernel with certified, mechanism-local summaries and exact
fallback**. “Skip arbitrary GPU events without changing cycle accuracy” is not
sound, and “event-driven GPU simulation” is established prior art. A publishable
contribution would need a state-complete certificate, a refusal path, measured
end-to-end speed including planning, and independent accuracy evidence on real
SM86 traces.

## Evidence and availability

| Evidence | Location or revision | Result |
|---|---|---|
| Local prototype | `/home/jakeshea/GPU_prototypes`, commit `40a331d` (`update`), clean worktree | Small Rust event queue, scoreboard, SM86 coalescer, exact and narrow causal modes |
| Local research notes | `/home/jakeshea/GPU_sims_notes` | Exists and was inspected; contains prior-art exclusions, MGPUSim source memo, papers and experiment notes |
| Requested notes path | `/home/jakeshea/GPU_sims_nots` | **Unavailable**; no directory or files were found. The similarly named `GPU_sims_notes` tree was used instead. |
| MGPUSim pinned checkout | `/home/jakeshea/GPU_sims_notes/upstream_research_20260901/mgpusim`, `d8f4c0a7d7b75498a91525cde37f3c5d2d6b70b6` (2026-08-29) | Retained upstream source and Akita event engine inspected |
| Accel-Sim framework fresh clone | `/data1/tmp/gpu_simulator_research/accel-sim-framework`, `d930ad6d02c09bb56867132583735aba0389cff4` (2026-08-26) | Main performance loop and trace frontend inspected |
| Accel-Sim integrated GPGPU-Sim checkout | `/home/jakeshea/GPU_sims_notes/upstream_research_20260901/accel-sim-gpgpu-sim_distribution`, `91880c53383d5a6a6742bfb1be2c5f34e39c7871` | Exact framework-integrated core/cache/interconnect/DRAM baseline |
| Vanilla GPGPU-Sim pinned checkout | `/home/jakeshea/GPU_sims_notes/upstream_research_20260901/gpgpu-sim_distribution`, `a4ce3feac901c97a4b4601f679e43cf3589c79de` (2025-02-15) | Independent public core/cache/interconnect/DRAM comparison |
| Additional local checkouts | `/home/jakeshea/GPU_sims_notes/upstream_research_20260901` | FlashGPU-Sim, NVArchSim-related material, HyFiSS, GPU-Tile-Sim, modern GPU simulator and others |

The local prototype’s own assessment is appropriately cautious: it has
component-level NCU sector agreement, but no end-to-end prototype-versus-silicon
cycle correlation because a matching public dynamic trace is unavailable. Its
20,000-descriptor result is a synthetic scale probe, not evidence of a
production simulator speedup. The current rebase contains a separate exact and
summary recurrence, but its certificate models only warp readiness, dispatch
readiness, pipeline reservations, and completion; it does not model caches,
MSHRs, memory responses, barriers, or interconnect state.

The two GPGPU-Sim revisions are intentionally distinct: `a4ce3fe...` is the
vanilla public comparison, while `91880c533...` is the tree actually integrated
by the pinned Accel-Sim framework. Interoperability claims must name both and
use the integrated revision for framework runs; the framework setup script’s
mutable branch defaults are not an acceptable provenance pin.

The subsequent SM86 differential exposed and fixed two concrete local wake
gaps: an instruction-stream line can become ready before its warp is eligible,
and a bounded actor wake could leave the local clock at the prior timestamp.
`SmCore::advance` now includes demanded-line readiness, commits bounded clocks
to the external horizon, and makes the dense reference walk idle cycles below
that horizon. Across the retained eight-workload small set, event and
dense-reference outputs are byte-identical; event mode performed 1,246,619
local steps and 219,112 idle jumps covering 3,771,866 virtual cycles, while
dense mode performed 5,018,485 local steps. This is an exactness result for
the declared model, not a silicon validation result. The retained seven-workload
large cohort also matches 7/7: event mode performed 2,069,235 local steps and
skipped 3,164,825 virtual cycles while dense mode performed 5,234,060 local
steps. Across all 15 rows, event mode skipped 6,936,691 virtual cycles while
preserving serialized output.

The rebase’s production SM path already applies the safe form of event
advancement. `sim/sm/src/issue.rs::SmCore::advance` runs issue cycles strictly
before a typed external-event horizon, jumps only to a known warp/resource or
front-end timer, and parks at the horizon instead of crossing an event that may
make a warp ready. Requests emitted during the call tighten the horizon to a
known lower bound; outstanding responses with no computable due time return an
idle wake rather than permitting a speculative jump. `sim/sm/src/system.rs`
tracks these unpriced response bounds between engine calls. This is exactly the
kind of typed horizon and quiescence guard required for sound next-event
simulation. It should be treated as the baseline to measure against, not as
evidence that the broader causal summary is already generally exact.

The 2026-09-18 trace-storage review also inspected the bounded Akita checkout
`/data1/tmp/gpu_simulator_research/lossless_trace_20260918/akita` at commit
`011edf0ecc0d0640d80309af97485a4a76fe2ea3`. Its `modeling/ticker.go` Smart
Ticking path reschedules only after progress and wakes on messages or a newly
free output port; this supports conservative horizon advancement, not omitted
state transitions. Its `tracing/dbtracer.go` stores task lifecycle rows plus
milestones/tags rather than full mutable-state snapshots. The related Akita
paper, [arXiv:2604.28073](https://arxiv.org/abs/2604.28073), reports Smart
Ticking as a measured speed/virtual-time trade-off rather than a cycle-exact
proof. Our typed trace therefore keeps exact state projection and uses array
splices only as a lossless storage encoding.

The restored Accel-Sim checkout was independently rechecked on 2026-09-18.
At `gpu-simulator/accel-sim.cc:150-170`, `simulate()` calls
`gpgpu_sim->cycle()` followed by `deadlock_check()` on every active iteration
and exits only on kernel completion, the configured cycle/instruction cap, or
an inactive simulator. This is direct source evidence that its detailed lane
does not blanket-skip idle-looking cycles; the relevant primary source remains
the Accel-Sim 2.0 paper ([arXiv:2608.22602](https://arxiv.org/abs/2608.22602)).

## What “cycle-accurate” must mean

Let a simulator state be *S* and its deterministic transition at cycle *t* be
`F(S, t)`. A dense cycle loop computes `S(t+k) = F(...F(S(t), t), ..., t+k-1)`.
An exact skip may replace that sequence with a summary transformer `G_k` only if

```text
G_k(S(t), external_inputs[t:t+k]) = S(t+k)
```

and if `G_k` produces every observation that the study promises: instruction
issue and completion times, queue occupancy, cache/MSHR state, memory traffic,
barrier and asynchronous tokens, counters, power samples, and deterministic
same-cycle ordering. A scalar duration is not a state transformer.

The admissibility predicate must be conservative. It has to establish that no
unmodeled input can arrive inside the interval, no state-dependent arbitration
can change, no periodic event is due, and no observable internal transition is
being discarded. If any premise is unknown, the simulator must reify the
interval in detailed mode. This is the same distinction as partial-order
reduction: independent transitions may be commuted, but a transition can be
removed only after its independence and observation effects are proved.

## Findings from the local trees

### `GPU_prototypes`

The prototype is a useful, intentionally small seam rather than a full GPU
model. `EventQueue` orders `(time, sequence)` deterministically. The scoreboard
tracks lane-aware hazards, and the mini-SM has four partition-local resource
ledgers and cross-warp signals. Exact mode emits child completion events and
retains resource reservations. Causal mode collapses only contiguous,
operand-free, control-clean, side-effect-free intervals when one partition has
no competing warp and no in-flight work. Memory roles, barriers, waits/signals,
and multi-child operations force reification.

That is a reasonable safety envelope for a prototype. It is much narrower than
an NVIDIA SM: there is no L0/L1/L2 hierarchy, MSHR merging, interconnect,
DRAM, TLB, global barrier, asynchronous copy, tensor-memory path, or dynamic
warp scheduler. The README and `report-source.md` correctly label the causal
mechanism an engineering hypothesis and state that the synthetic event counts
are not silicon or simulator speed evidence.

### Current rebase implementation

`sim/sm/src/causal_frontier.rs` executes independent exact and summary
recurrences and compares terminal state. The certificate in
`sim/semantics/src/certificate.rs` assumes one homogeneous pipeline class, an
empty entry pipeline, no incoming work, no dependencies or side effects, and a
producer-supplied completion fact. The implementation’s `pipeline_index` maps
dispatch, branch, and barrier to one index; summary application mutates only
warp/dispatch/pipeline readiness and completion vectors. It is therefore a
model-relative proof of a small recurrence, not a proof about a complete SM or
silicon.

The analyzer also sorts programs by sequence. That gives deterministic replay,
but it is not NVIDIA SMSP scheduler arbitration; if sequence numbers are only
per-warp, equal-sequence instructions are forced into an arbitrary serial tie
order. `active_per_smsp` is a static occupancy count, and planner work is at
least linear in the descriptors. A
reported reduction in nominal event accounting is not the same as fewer
production queue pops, and it can consequently be smaller than, or even slower
than, the cost of building the plan. The right metric is total
wall-clock time for planning plus execution, together with the number and
reason of refused opportunities.

## What the mainstream simulators actually do

| Project | Execution mechanism observed | Implication |
|---|---|---|
| **MGPUSim/Akita** | Priority event queue advances to the earliest timestamp. Active CU `Tick` runs branch, scalar, SIMD, LDS, vector memory and scheduler phases. Scoreboard counters decrement each busy cycle; issue/fetch observe port backpressure. Components can stop ticking and wake on messages or scheduled events. | Exact next-event advancement is established, but detailed compute state is still ticked. Sampling is a separate approximate mode. |
| **Accel-Sim 2.0** | `accel-sim.cc::simulate()` calls `m_gpgpu_sim->cycle()` and deadlock checks while the kernel is active. The frontend preserves SASS-to-IR correspondence, masks, registers, addresses, TMA/WGMMA/mbarrier and replay markers. | The main detailed path remains cycle-driven because scheduler, synchronization and counters are dynamic. Trace compression and GPUVision PM sampling attack data/host cost, not arbitrary cycle deletion. |
| **GPGPU-Sim** | `gpgpu_sim::cycle()` advances independent clock domains, clusters, interconnect, L2 and DRAM. Shader-core writeback, execute, operand read, issue, decode and fetch are ordered every core tick. Cache and DRAM queues decrement and arbitrate each cycle. | A dense loop is simple and exposes all queue/credit/tie state needed for architectural studies. |
| **NVArchSim** | Event-based components tick only when work exists; future packet release and instruction completion are scheduled. The design adds fidelity only where a bottleneck requires it and uses faster abstractions elsewhere. | Strong evidence for selective event simulation, not blanket exact skipping. |
| **FlashGPU-Sim** | Execution-driven, cycle-accurate Hopper/Blackwell model with TMA, mbarrier, MMA/WGMMA and OpenMP SM simulation. Its retained README reports close NCU comparisons for vector add, Triton GEMM and FlashAttention. | Modern validation work emphasizes architectural coverage and independent cycle checks rather than an event-skipping headline. |
| **HyFiSS** | Hybrid/statistical memory model with a detailed private SM loop and explicit per-cycle issue, operand collection, barriers and functional-unit slots. | Fidelity is allocated by component; detailed core state is not removed by a generic skip. |
| **LPGSim** | Removes low-impact metadata, coalesces compute instructions and parallelizes local clocks. It reports large speedups with material error (about 21–23% in the published evaluation). | A useful speed/fidelity baseline; demonstrates that compression must be measured against accuracy. |
| **GPU-Tile-Sim / Sim-FA** | Tile/dependency-graph and asynchronous pipeline abstractions target modern GPUs and report explicit error bounds. | Dependency-aware abstraction is active prior art; the novelty burden is GPU-specific state and validation. |
| **ACALSim** | Event-driven framework fast-forwards to the next pending simulator tick, processes same-cycle phases first, and uses deterministic parallel execution. | A recent independent implementation confirms that safe fast-forward is a quiescence/pending-event operation, not deletion of detailed transitions. |
| **GATSim** | Abstract timing keeps occupancy vectors and advances directly to the next remaining latency when no warp is ready. | Large speedups are possible in a reduced model, but its authors report accuracy limits when data-dependent timing, synchronization and detailed memory effects are omitted. |

MGPUSim’s current README still describes AMD GCN3 as the stable timing target
and NVIDIA support as unfinished. Its new MI300X configuration is therefore a
modeling target, not independent MI300X validation; no MI300X machine or ROCm
capture was available for this study.

### MGPUSim: the direct precedent

MGPUSim’s ISCA paper defines “no busy ticking” as a design requirement. Its
event engine triggers state updates chronologically, permits independent events
at the same time, and avoids updating cache, interconnect, and memory-controller
state every cycle. A DRAM request that is known to complete after 300 cycles can
schedule a completion event at that time. A blocked connection wakes when its
producer or consumer changes the availability condition rather than retrying
blindly every cycle.^1

This works because the skipped component has a closed, explicit contract: its
future event time and required post-state are known, and no hidden event can
arrive between now and then. The CU is different. Its scheduler checks
completion, decode, issue and fetch every tick; fetch can be denied by port
backpressure; scoreboards decrement readiness counters; and barriers and
`waitcnt` can alter which warp is eligible. MGPUSim therefore combines an event
engine with cycle-like ticks for active units.

The separate Photon sampled path is instructive. It waits for stable basic-block
behavior, predicts a wavefront interval, stops admitting new code, and schedules
predicted completion events. Running wavefronts and in-flight state are drained
before sampling takes effect; barriers are handled at explicit boundaries. This
is a valid performance mode, but the predictor does not preserve all cache,
network, and arbitration state, so its output is an estimate rather than an
exact replacement for the detailed path.^2

The recent Akita Smart Ticking study makes the same boundary explicit. Pure
event-driven retries can turn back into one event per cycle when a blocked port
is polled; Smart Ticking instead sleeps a component after a no-progress tick and
wakes it on a message or port-availability notification. The reported average
host-time improvement is 2.68x with less than 1% virtual-time change, so this is
an effective framework optimization with a measured residual difference, not a
claim of zero-error removal of all ticks.^17

The pinned primary-source check reinforces the local run-ahead distinction.
Accel-Sim/GPGPU-Sim at commit `91880c53383d5a6a6742bfb1be2c5f34e39c7871`'s
`gpgpu_sim::next_clock_domain` selects the minimum among
independent core, interconnect, L2, and DRAM clocks and advances only the
selected domains (`upstream_research_20260901/accel-sim-gpgpu-sim_distribution/
src/gpgpu-sim/gpu-sim.cc`). MGPUSim's compute-unit `Tick` likewise advances
local fetch/issue/execute phases while separately scheduling completion events
at commit `d8f4c0a7d7b75498a91525cde37f3c5d2d6b70b6`
(`upstream_research_20260901/mgpusim/amd/timing/cu/computeunit.go`). These
references support keeping the typed SM-local clock separate from the global
event-queue timestamp; forcing one scalar frontier would reject valid
multi-domain progress.

### Accel-Sim and GPGPU-Sim: why the core still ticks

Accel-Sim’s frontend intentionally carries dynamic masks, register values,
memory addresses, replay markers, and modern asynchronous instructions. Its
trace documentation notes that data-dependent control flow and global
synchronization limit what a fixed trace can represent. The performance loop
continues to call the GPGPU-Sim cycle function until kernel completion and runs
deadlock checks on each iteration.

GPGPU-Sim’s source makes the hidden state visible: independent clock domains,
issue and arbitration limits, instruction-buffer and barrier checks, cache
latency queues, MSHR/port state, interconnect credits, DRAM ready cycles and
row/bank service state. A jump over *k* cycles is exact only if it reproduces
every one of those state transitions and the same-cycle priority order. A
single “warp completes at t+k” event cannot do that.

## Why blanket GPU event skipping fails

| Mechanism | What can change inside a proposed interval | Required exact-summary content |
|---|---|---|
| Warp scheduling | Another warp becomes eligible; round-robin/oldest priority changes; issue width is consumed | Ready set, arbitration pointer, issue-port reservations and deterministic tie order |
| Scoreboards and pipelines | Register countdown reaches zero; execution-unit slot frees; writeback competes with issue | Per-register readiness, pipeline stage occupancy, release events and phase ordering |
| Caches and MSHRs | Hit/miss classification, replacement state, merged miss, victim lock, fill/writeback | Tags, recency, dirty/sector masks, MSHR ownership and all downstream requests |
| Memory system | Queue head, bank/row policy, refresh, credit or partition availability changes | Queue contents, service deadlines, credits, row state and arbitration history |
| SIMT control | Divergence/reconvergence, `bar.sync`, `waitcnt`, mbarrier, async-copy or TMA completion wakes code | Masks, PCs, barrier generations, async groups, tokens and wake-up conditions |
| Atomics and side effects | A returned value changes a later predicate; ordering affects another warp or CTA | Data values, conflict order, visibility scope and all dependent control flow |
| Interconnect/collectives | Packet arrival changes a consumer; backpressure changes a producer’s issue time | In-flight packets, virtual channels, credits, routing and ordering |
| External/periodic work | Host launch, DMA, TLB/page walk, preemption, clock/power event or refresh arrives | Complete external event stream and periodic deadlines |
| Observability | PM sample, power integral, occupancy or debug trace records an internal transition | Every promised counter/sample or a mathematically equivalent integral |

These effects are non-local. A memory response can arrive while an apparently
independent arithmetic region is being summarized; a cache fill can alter the
latency of a later instruction; and a same-cycle producer can win or lose an
arbitration tie depending on whether an internal event was represented. “No
visible instruction dependency” is therefore weaker than “no state transition
can affect the observation contract.”

The trace itself adds a second problem. A captured SASS trace freezes one path,
mask and address stream. Changing scheduler, cache, latency, synchronization or
placement can change a branch predicate, memory address, atomic value or global
barrier order. A simulator can be internally deterministic while answering only
“what if this observed path ran on the new machine?” rather than “what would the
program do on the new machine?” Accel-Sim’s paper and NVArchSim both discuss
this trace/execution trade-off.^3,^4

Finally, event queues are not free. A priority heap costs `O(log E)` per
scheduled event, allocations and pointer-rich messages hurt locality, and highly
active GPU regions may have an event at nearly every cycle. A planner that scans
all descriptors and constructs safety metadata can erase the queue reduction.
This is why modern systems also pursue trace compression, metadata removal,
parallel local clocks, kernel sampling, and selectively detailed components.

## Why modern projects do not use a generic skip everywhere

The absence of a blanket optimization is not evidence that event simulation is
useless. It reflects five engineering constraints:

1. **The safe interval is difficult to know.** The simulator must prove that no
   hidden arrival, countdown, queue transition, periodic event or tie can occur.
   In a busy GPU this predicate is usually false or expensive to establish.

2. **The post-state is larger than the endpoint time.** Architectural studies
   inspect stalls, cache traffic, occupancy, queue depth, power and phase
   behavior. Preserving only final completion cycles can hide a wrong mechanism.

3. **Architectural interventions invalidate shortcuts.** A summary derived for
   one issue width, cache policy or memory topology may not remain valid when
   those parameters change. A general simulator must either regenerate and
   certify summaries for every candidate or fall back to the detailed path.

4. **Dense loops are easier to maintain and parallelize.** A cycle epoch gives
   deterministic synchronization between SMs, interconnect and memory. A
   priority event engine introduces message ordering, same-time phase rules and
   cross-thread synchronization that can cost more than it saves.

5. **Established alternatives already target the dominant cost.** NVArchSim
   chooses fidelity per bottleneck; Accel-Sim compresses traces and samples
   hardware counters; Photon, SIEVE, PKA and GCL-Sampler reduce representative
   work; LPGSim coalesces metadata and parallelizes simulation. These techniques
   make explicit accuracy/speed trade-offs instead of claiming that arbitrary
   internal events are irrelevant.^2,^5,^6

## Assessment of the local proposal

The proposal is technically meaningful as three separable optimizations:

* **Idle-time next-event advancement:** adopt or strengthen a global event
  engine so inactive components jump to known future messages. This is safe when
  the component contract is closed and is already established prior art.
* **Certified region summarization:** add a state-complete summary transformer
  for a deliberately restricted SM mechanism, with exact fallback and explicit
  refusal reasons. This is the plausible local research seam.
* **Sampling or trace reduction:** predict representative wavefronts, basic
  blocks or tiles and report error bounds. This can deliver large speedups, but
  it should be labeled approximate and compared against Photon/SIEVE/LPGSim and
  related methods.

The current causal certificate should not be presented as a full cycle-accurate
GPU proof. Its correct claim is: *for the descriptor recurrence and declared
boundary state, the summary and exact executors reach the same modeled terminal
state under the certificate’s guards*. It does not establish NVIDIA scheduler,
cache, DRAM, synchronization or silicon equivalence.

The strongest paper direction is a GPU-specific **state-complete validity and
certification layer** rather than another event queue. A candidate summary would
need to carry, at minimum:

* warp PCs, active/predicate masks, register and scoreboard readiness;
* issue-port and execution-unit reservations, arbitration pointers and release
  multiplicities;
* cache tags/recency/dirty sectors, MSHRs, coalescer children and writebacks;
* interconnect packets/credits and DRAM bank, row, refresh and queue state;
* barriers, `waitcnt`, mbarrier/TMA/async groups, atomics and completion tokens;
* counters and any PM/power observations included in the study; and
* a deterministic same-cycle ordering rule.

The certificate should return `CERTIFY`, `REFUSE`, or `UNKNOWN`. For every
`CERTIFY`, an exact reference run should compare the complete boundary snapshot
at the next external interaction. For `REFUSE`, the report should identify the
first missing condition and execute the detailed path. A scalar event-count
reduction without planner-inclusive wall time and boundary-state comparison is
not sufficient.

## Evaluation that could support a paper

Use the retained SM86 traces and independent NCU measurements, keeping NCU
values as validation targets rather than simulator inputs. Compare:

* the existing detailed simulator;
* exact event-driven execution;
* certified summaries with fallback;
* a representative sampling/metadata-compression baseline; and, where the ISA
  and scope permit, MGPUSim, Accel-Sim and FlashGPU-Sim.

Cover tensor MMA/IMMA, FlashAttention forward/backward, contiguous and strided
memory, reuse and partition camping, atomics, barriers, divergent control,
async-copy/TMA-like waits, and multiple resident warps. Include negative
controls where every shortcut must refuse: full queues, out-of-order sibling
responses, aliasing, cross-warp dependencies, barrier generations and
asynchronous producer/consumer chains.

Report total host time (planner plus execution), event count, summary rate,
reification rate, peak memory, and deterministic output. Compare per-instruction
issue/completion, cache/MSHR/DRAM traffic, queue occupancy, stall categories and
kernel time. If PM time-series data are available, compare phase-level samples;
endpoint agreement alone is insufficient. Measure architecture-decision
stability as well: does the optimization preserve the ranking of cache, issue,
warp and memory designs, and can it expose a ranking that a coarse model gets
wrong?

The minimum credible result is zero false certificates over adversarial tests,
positive certificates across several independent mechanisms, a measured
end-to-end speedup after planning overhead, and a clear error bound or exact
equivalence statement. If most real traces refuse, that is still a useful
result: it identifies where event summaries are safe and where a detailed GPU
model is unavoidable.

## Recommendation

Proceed with event-driven execution as an implementation technique, but do not
claim it as the contribution. Keep active SM scheduling, memory queues and
synchronization detailed until a state-complete certificate exists. Extend the
local prototype first with an explicit boundary-state schema and exact-fallback
reasons, then run the SM86 adversarial matrix above. Treat the resulting
certification/validity contract, measured speed/fidelity trade-off and any new
architecture insight as the possible contribution. Defer MI300X claims until a
machine or retained ROCm/HSACO capture is available.

## Sources

1. **MGPUSim authors**, “**MGPUSim: Enabling Multi-GPU Performance Modeling and Optimization**,” ISCA 2019 author PDF. Event-driven engine, conservative ordering, “no busy ticking,” and the 300-cycle DRAM example. [https://people.bu.edu/joshi/files/mgpusim-isca2019.pdf](https://people.bu.edu/joshi/files/mgpusim-isca2019.pdf)
2. **Photon authors**, “**Photon: A Fine-grained Sampled Simulation Methodology for GPU Workloads**,” MICRO 2023. Kernel/warp/basic-block sampling and predicted wavefront completion. [https://www.comp.nus.edu.sg/~tcarlson/pdfs/liu2023pafssmfgw.pdf](https://www.comp.nus.edu.sg/~tcarlson/pdfs/liu2023pafssmfgw.pdf)
3. **Accel-Sim authors**, “**Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling**,” ISCA 2020. SASS-to-IR trace correspondence, counter validation and trace limitations. [https://people.ece.ubc.ca/aamodt/papers/accelsim.isca2020.pdf](https://people.ece.ubc.ca/aamodt/papers/accelsim.isca2020.pdf)
4. **A. Jog et al.**, “**Need for Speed: Experiences Building a Trustworthy System-Level GPU Simulator**,” HPCA 2021 (NVArchSim). Variable fidelity and event-based components. [https://d1qx31qr3h6wln.cloudfront.net/publications/HPCA_2021_NVArchSim.pdf](https://d1qx31qr3h6wln.cloudfront.net/publications/HPCA_2021_NVArchSim.pdf)
5. **SIEVE**, ISPASS 2023. Invocation grouping and trace reduction with reported hardware error bounds. [https://users.elis.ugent.be/~leeckhou/papers/ispass-2023.pdf](https://users.elis.ugent.be/~leeckhou/papers/ispass-2023.pdf)
6. **LPGSim**, SIGMETRICS 2026. Metadata removal, compute coalescing and parallel local clocks with measured speed/fidelity trade-off. [https://doi.org/10.1145/3805642](https://doi.org/10.1145/3805642)
7. **Accel-Sim 2.0 authors**, “**Accel-Sim 2.0**,” arXiv 2608.22602. Modern asynchronous GPU modeling, temporal merge and trace limitations. [https://arxiv.org/abs/2608.22602](https://arxiv.org/abs/2608.22602)
8. **Sim-FA**, arXiv 2605.00555. Asynchronous TMA/WGMMA timing model, scope and explicit limitations. [https://arxiv.org/abs/2605.00555](https://arxiv.org/abs/2605.00555)
9. **GPU-Tile-Sim**, arXiv 2607.11262. Tile/dependency-centric modern GPU simulation. [https://arxiv.org/abs/2607.11262](https://arxiv.org/abs/2607.11262)
10. **ACALSim**, arXiv 2605.22936. Event-driven fast-forward and deterministic parallel simulation. [https://arxiv.org/abs/2605.22936](https://arxiv.org/abs/2605.22936)
11. **GATSim**, DATE 2017. Abstract GPU timing with occupancy-vector fast-forwarding and stated fidelity limits. [https://slam.ece.utexas.edu/pubs/date17.GATSim.pdf](https://slam.ece.utexas.edu/pubs/date17.GATSim.pdf)
12. **MGPUSim project**, current source and timing-mode documentation. [https://github.com/sarchlab/mgpusim](https://github.com/sarchlab/mgpusim)
13. **Accel-Sim framework**, current source, trace frontend and GPUVision documentation. [https://github.com/accel-sim/accel-sim-framework](https://github.com/accel-sim/accel-sim-framework)
14. **GPGPU-Sim distribution**, current cycle-level source. [https://github.com/gpgpu-sim/gpgpu-sim_distribution](https://github.com/gpgpu-sim/gpgpu-sim_distribution)
15. **FlashGPU-Sim**, execution-driven Hopper/Blackwell simulator and validation README. [https://github.com/FlashGPU-Sim/FlashGPU-Sim](https://github.com/FlashGPU-Sim/FlashGPU-Sim)
16. Local source evidence: `/home/jakeshea/GPU_prototypes/{README.md,report-source.md,research-log.md}` at commit `40a331d`; `/home/jakeshea/GPU_sims_notes/gpu_simulation_frontier_study/prior_art/existing_work_exclusions.md`; `/home/jakeshea/GPU_sims_notes/upstream_research_20260901/mgpusim_mechanism_memo.md`; current rebase files `sim/sm/src/causal_frontier.rs` and `sim/semantics/src/certificate.rs`.
17. **Akita authors**, “**Akita: A Versatile Framework for Architectural Simulation**,” arXiv 2604.28073. Smart Ticking wake rules and measured speed/virtual-time trade-off. [https://arxiv.org/abs/2604.28073](https://arxiv.org/abs/2604.28073)
