# RTX 3090 Ti / SM86 full-replay modeling plan

**Status:** design baseline after repository, corpus, and prior-research audit
**Date:** 2026-10-16
**Branch:** `3090ti_full_replay`
**Target:** GeForce RTX 3090 Ti / GA102 / compute capability 8.6 / `sm_86`
**Corpus:** `../sm86_rtx_3090_capture/`

This is the active implementation plan for this branch. The branch is
3090-Ti-only: SM120/RTX-5090 compatibility, merge choreography, and preserving
5090-specific CLI or model assumptions are not design constraints. Existing
SM120-oriented seams may be refactored in focused SM86 work, while unrelated
behavior remains preserved until its replacement path is proven. The target is
an effective, conditional, uncertainty-aware model rather than a claim to
recover proprietary RTL.

## 1. Outcome and completion bar

### 1.1 Requested outcome

Build a default **full replay** path that executes every admitted real SM86
trace through one coherent model:

```text
enhanced NVBit trace + SASS/control sidecars
  -> typed dynamic transaction stream
  -> per-SMSP issue/dependence/occupancy model
  -> sectored L1TEX
  -> six-slice effective L2
  -> finite interconnect/partition queues
  -> banked effective GDDR6X/DRAM service
  -> fills, acknowledgements, writeback, visibility, and warp readiness
  -> separate issue/event/launch/tail frontiers and IPC projections
```

The full mode is the default and composes all **supported and evidence-backed**
capabilities at once; it must not silently fall back to flat timing when a
cache, queue, or IPC component is selected. The landed SM86 path currently
uses the integrated L1/L2 hierarchy, the aggregate carded read-service arm
with the independently measured effective-rate scenario, the bounded
store/fence admission candidate, and observer-only partition/route pressure.
This is an effective conditional service model, not a physical row/bank claim.
Finite per-partition/bank timing, resident dirty writeback, coherence,
payload forwarding, and physical row/bank mapping remain future or explicit
scenario arms until their independent evidence and typed inputs exist. Full
mode still emits typed demand/lower-bound/tail fields for those gaps and never
invents a timing constant or silently falls back to flat timing.

### 1.2 Practical completion bar

The plan is complete when all of the following are true:

1. The runner discovers the complete real corpus (currently 356 capture roots,
   of which 354 are contract-clean and two have typed-invalid trace-contract
   dispositions; 15,165 discovered kernel launches, 107,533,357 instructions
   and 118,262 blocks), and emits a per-root/per-launch coverage matrix recording
   dynamic launches reported, launches selected/replayed, records replayed,
   omissions, and typed input dispositions. “Complete corpus” means every
   discovered launch has a row or explicit disposition; it does not assume that
   every trace record is replayable.
2. Every `status=ok` full replay row contains instruction, L1, L2,
   partition/queue, DRAM demand/service, completion-tail, and IPC/frontier
   fields, or a typed unavailable/open-tail reason scoped to the
   mechanism/configuration. `input_error`, `timeout`, and `replay_error` rows
   carry a failure taxonomy and never fabricate metrics.
3. L1/L2 traffic and hit/lookup projections are produced for all launches whose
   trace carries the required address/control information; unsupported address
   classes remain separate rather than being counted as hits.
4. DRAM read demand, writeback demand, read/write service, and service-delay
   projections are produced independently. Completed DRAM writes are not
   inferred from a terminal dirty gauge.
5. IPC projections use SASS-level instruction identity, compiler control bits,
   scoreboard readiness, resource/operand pressure, memory response events,
   and occupancy. PTX instruction counts are never substituted for SASS issue
   work.
6. NCU values are joined only after replay. Existing profiles are treated as
   one selected launch per root unless an exact launch identity exists; the
   plan does not claim 15,165-launch NCU accuracy from 356 single-launch
   profiles.
7. Results are reported by workload family and mechanism residual. A single
   scalar score is not used to select topology, queue, or service parameters.
8. The complete-corpus run uses exactly twelve simulator workers and writes new
   generated JSONL as `.jsonl.zst`, with isolated per-root logs/results.

## 2. Research baseline and current state

### 2.1 Repository state found in the audit

The current repository is a Rust event-driven simulator with this native path:

```text
NVBit/SASS + enhanced protobuf + sidecars
  -> sim/trace
  -> sim/sm + sim/mem + sim/engine
  -> sim/direct_run (sim-run)
  -> post-replay NCU comparison
```

The relevant implementation seams are already substantial but are currently
framed around the former SM120 alpha lane:

| Area | Existing implementation | SM86 work required |
|---|---|---|
| Trace ingestion | `sim/trace` reads protobuf, dynamic memory metadata, control and async sidecars | Make architecture/contract names generic; preserve SM86 identity even when stale metadata says `sm120` |
| Issue model | `sim/sm/src/issue.rs` and `issue/config.rs` model warp readiness, barriers, compiler control fields, four SMSPs and SM placement | Refactor `Sm120AlphaConfig`/SM120 branches into an SM86 full configuration; add SASS/resource/operand pressure needed for IPC |
| L1 | `sim/mem/src/l1.rs`: 32-B sectors, 128-B lines, lazy line/sector fills, merges, finite MSHR, LRU line pool, stores/reductions and diagnostics | Replace the fully-associative default with a configurable SM86 effective set/way/hash/policy model; add queue/port/backpressure and explicit write path |
| L2 | `sim/mem/src/l2.rs`: sectored set-associative state, six hash candidates, LRU/SRRIP/BIP, event fill tokens, generation checks | Make six-slice GA102 geometry first-class; add slice/partition request queues, bank conflicts, prefetch/operator state, and source-qualified timing |
| DRAM | `sim/sm/src/system.rs`: aggregate fixed-point G/D/1 read server; optional `WritebackQueue` | Add finite partition/bank queues, read/write arbitration, row/bank service, atomics, dirty writeback and service uncertainty |
| Metrics | `sim/direct_run/src/metrics.rs` and `emit.rs` expose conditional L1/L2/DRAM traffic and model-relative IPC/frontiers | Promote the SM86 metric schema; keep unavailable families compact and keep frontiers separate |
| Runner | `tools/sm120_corpus_replay.py` and `sm120_replay_bench.py` are SM120-specific; only three legacy flat replay JSON files exist in the SM86 corpus | Build a generic/SM86 corpus runner, static-sidecar adapter, full-mode default and compressed artifact path |
| Architecture card | `arch_profiles/cards/sm_86.json` is already a 3090 Ti card; `data/pchase/sm_86_static_card.json` supplies measured latencies and typed geometry gaps | Expand the card with effective interval/configuration fields and provenance; do not import SM120 constants |

The plan-audit working-tree evidence includes `/tmp/sm86_vertical.json`, a
freshly rebuilt replay of
`fp32_fma_dep_chain` with `--mode l1-l2`, the SM86 card, the p-chase card, and
`--carded-dram`: 8,704 L1 accesses (864 hits, 7,840 misses, 6,816 merges),
1,792 L2 lookups (248 hits), 128 DRAM read sectors, and 32 terminal dirty L2
sectors. It terminated with an explicit dirty-writeback/coherence tail and no
DRAM-service congestion. The `/tmp` output and release binary are ephemeral;
P1 must rebuild and preserve the actual command, card, control sidecar and
output in the research-results directory before treating this as a durable
reproducibility baseline. It is not corpus accuracy or physical completion.

### 2.2 RTX 3090 Ti facts and carded inputs

The current SM86 card and local host evidence provide:

- GA102, compute capability 8.6, 84 SMs, four SMSPs per SM, one issue slot
  per SMSP, no assumed dual issue.
- 48 resident warps/SM and 16 resident blocks/SM in the current card.
- L1TEX/shared structure: 131,072-B combined capacity declaration, default
  L1 32 KiB, maximum shared allocation 102,400 B, 128-B lines and 32-B sectors.
- L2: 6,291,456 B, six slices, 128-B lines, 32-B sectors,
  `write_allocate_non_inclusive` policy.
- Locked p-chase reference point: 1,350 MHz SM and 10,501 MHz memory clock,
  driver 610.43.02 in the retained p-chase card. The capture host later used
  driver 615.71.09; every new report must record the actual host identity.
- Counter-free dependent-chain anchors from
  `data/pchase/sm_86_static_card.json`: L1 33.0016 cycles (dispersion 1.6386),
  L2 243.6152 cycles (7.0492), DRAM 422.6486 cycles (0.04), shared 23.0016
  cycles (1.4288).
- DRAM peak hypothesis: 1,008 GB/s from 21 Gbps GDDR6X on a 384-bit bus at
  the locked memory point. This is a documented peak/service hypothesis, not
  an achieved service rate.
- The p-chase card leaves L1 associativity/replacement, exact L2 capacity
  transition, and MSHR saturation typed unresolved. These become explicit
  effective parameter intervals, not silently resolved physical facts.
- The current resolver uses the p-chase `l1_capacity_bytes=98,304` base point
  (with a transition bracket toward 163,840 B), while the architecture card
  separately documents 131,072 B combined L1TEX and 32 KiB default L1. These
  are not interchangeable fields: full replay must record which capacity
  provenance is active and run the alternatives as sensitivity scenarios.

### 2.3 Counter-free and NCU-derived prior observations

The prior SM86 research notes provide useful mechanism anchors:

- Aligned 32/64/128-bit global loads generate 64/128/256 sectors; a 4-byte
  misaligned load generated 69 sectors, ruling out a naive fixed 80-sector
  formula.
- `.ca` route: 65,280 L1 lookup hits, 552 DRAM read sectors, 17,952 ns;
  `.cg` route: 0 L1 hits, 712 DRAM read sectors, 78,688 ns.
- Dependent chain: 202,240 ns and 87.22% long-scoreboard; ILP4: 74,368 ns and
  90.43%; ILP16: 72,192 ns and 57.07%.
- Conflict versus distinct atomics: 4,224 versus 2,016 ns; synchronous barrier
  2,080 ns; async copy plus wait 2,176 ns.
- Historical three-workload SM86 comparisons showed promising traffic/cycle
  residuals (for example BF16 GEMM L2 error about 0.17% and DRAM-read error
  about 1.88%), but the later L1 predictor holdout failed and the old values
  are development evidence, not closure.
- Large-store campaigns showed the unresolved tail directly: a 4-MiB-class
  stream had simulator lower bound 10,899 cycles versus NCU median 13,785;
  a 16-MiB-class stream had 16,281 versus 29,567, with large NCU DRAM-write
  counts. The current simulator only exposes dirty gauges/eviction demand in
  the default path.

NCU observations constrain and score the model after replay. They must not be
 used as features, topology selectors, or same-workload fitting labels.

### 2.4 Corpus facts

`../sm86_rtx_3090_capture/` is approximately 58 GiB and currently contains:

- 356 capture roots (plus `_diagnostics`), each with a dynamic trace and a
  corresponding profile directory; the contract audit currently reports 354
  clean roots and two typed-invalid roots. Nested trace-file counts are higher
  and must be reconciled by root identity rather than blindly counted.
- 15,165 kernel launches, 107,533,357 instructions, approximately 118k blocks;
  median root instruction count 84,395 and maximum 7,177,728.
- 327 logical catalog rows: 235 official SOL rows (94 L1, 82 L2, 33 Quant,
  26 FlashInfer) plus 92 SimBench rows.
- Current disposition counts: 70 native SM86, 7 official controls, 2 reduced
  shapes, 53 snapshot controls, 142 portable SM86 controls, and 53 explicitly
  unsupported SM120-ISA rows.
- 58 paired semantic substitute controls in
  `metadata/sm86_substitute_complete.tsv`; they remain controls, not native
  FP8/NVFP4/TMA/cluster evidence.
- Current NCU raw CSVs contain one profiled kernel row per root (the wrappers
  use launch-count 1), not one row for each of the 15,165 launches. Useful
  GA102 fields include L1TEX/LTS lookup and hit counters, DRAM sectors/bytes,
  active/elapsed cycles, and active/elapsed SM IPC.

The NCU identity inventory is asymmetric: 109 roots used profiler-from-start
and have 101 unique exact kernel-name/grid/block candidates, while five roots
have two duplicate-shape candidates and three have five; 247 profiler-from-start
off roots have zero exact-name matches because names are demangled/truncated
and only shape candidates remain. The report schema must preserve these counts
and classify duplicate/shape-only rows as ambiguous fallback, never as exact
identity.

The current family population is highly unbalanced: adapted roots contribute
115 roots/5,921 launches/40.9M instructions, SimBench 70/101/26.2M,
snapshot controls 54/1,340/14.8M, substitute controls 53/7,114/19.3M, FIB 26/
372/2.1M, LCD 16/16/0.1M, SOL 7/92/1.0M, official controls 7/167/0.6M,
candidate controls 5/39/2.5M, and the three vertical roots 3/3/4k. Reports must
show both root-balanced and launch/instruction-weighted aggregates so the large
substitute/control families cannot hide native SM86 residuals.

Important input defects to repair in the generic adapter:

- All 356 current `native_static_metadata.json` files carry the stale
  `metadata_kind=native_static_sm120_v3` producer label even though the roots
  execute on SM86. A current inventory finds 62 roots containing 5,784
  `architecture_version=80` kernel entries: 5 roots are only-version-80 and
  57 are mixed with version-86 entries. These require explicit
  `sm80_code_on_sm86_runtime` disposition rather than relabeling them as native
  SM86. Those 62 roots account for 2,597 launches and 14,247,706 instructions.
  If their decoded controls and memory routes are usable, they may still
  receive an SM86 runtime replay row as portable controls; otherwise they
  receive a typed input row.
- Some dynamic protobuf contract strings retain `sm120_*` names although the
  capture is SM86. Contract-string spelling must not override the explicit
  device/architecture fields.
- Three roots contain 60 typed `unsupported_memory_path` exclusions (all
  `UNSPECIFIED` operations in group-normalization kernels); one root contains
  an empty unmapped-range sidecar and no cross-allocation lane-span entries
  were found. M0 must carry these counts into coverage rather than silently
  dropping affected instructions.

### 2.5 Modern modeling literature used as mechanism guidance

The plan uses mechanism ideas, not transplanted constants, from:

- *Demystifying the Nvidia Ampere Architecture through Microbenchmarking and
  Instruction-level Analysis* ([arXiv:2208.11174](https://arxiv.org/abs/2208.11174)), which motivates dependent
  chains, cache-operator route probes, SASS-level latency/throughput, and
  datatype-specific instruction handling.
- *Analyzing Modern NVIDIA GPU cores* ([arXiv:2503.20481](https://arxiv.org/abs/2503.20481)), which motivates
  compiler-guided scheduling, four sub-cores/SMSPs, register-file cache and
  reuse-bit pressure, instruction-stream buffering, pending-request-table
  coalescing, and memory-pipeline resource modeling. Its reported accuracy is
  an external prior, not evidence for this 3090 Ti configuration.
- NVIDIA's [GA102 Ampere architecture whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)
  for public capacity/bus/SM facts only; effective queue, bank, MSHR and
  replacement parameters still need target-scope probes.

The local SOTA matrix in `/home/jakeshea/GPU_sims_notes/research_papers/`
also supplies mechanism ideas from source inspection of Accel-Sim,
NVArchSim/MGPUSim, sector/cache
models, Sim-FA, PipeWeave, TileSim/TileSight, GCL-style sampling, modern
NoC/L2 mapping, and DRAM/NUMA isolation studies. This plan transfers only
mechanism hypotheses (finite resources, address-aware overlap, nonuniform
routing, offered-ILP/queue probes, and stage-centric critical paths); it does
not transfer constants, use NCU fitting, or substitute another GPU's topology.

### 2.6 Prototype and microbenchmark workspace

`/home/jakeshea/GPU_sims_notes/` is the main research workspace for this
branch's SM86 prototype probes and microbenchmark analysis. Use it for:

- counter-free `clock64` latency/throughput probes and locked-clock metadata;
- real CUDA/NVBit/NCU capture prototypes for L1 route/capacity, L2 mapping,
  partition/DRAM service, writeback, atomics, async copies, scheduler and IPC;
- analysis notebooks/scripts, interval-fitting inputs, and mechanism reports
  that are not part of the simulator runtime;
- retained command/environment/shape records needed to reproduce a probe.

Keep raw capture/profiler data under the existing capture workspace and keep
generated simulator artifacts outside raw captures. Prototype results enter the
repository card or simulator only after independent repetition, explicit
architecture identity, and a mechanism-specific comparison; an NCU value from
the scored corpus cannot promote a prototype parameter.

## 3. Scientific and modeling contracts

1. **Target identity:** every replay records the actual GPU name, UUID, compute
   capability, driver, device index, capture root, card, binary, and sidecars.
   A self-consistent manifest is not host proof.
2. **NCU independence:** NCU is post-replay observation only. Never tune the
   same scored launch from its NCU value, use it to select a cache hash, or use
   it as a replay input.
3. **Separate frontiers:** retain issue, modeled-event, SM-active, launch
   envelope, visibility, writeback/service, and typed-tail frontiers as
   separate fields. IPC projections must identify their denominator and
   endpoint; no scalar hides unresolved tails.
4. **Effective topology:** L1 associativity, L2 hash, partition count, DRAM
   banks, queue depth, and service efficiency are effective/conditional model
   parameters unless independently measured. Report intervals and sensitivity.
5. **Traffic conservation:** maintain typed ledgers: lookups = hits + fresh
   misses + pending merges; DRAM read demand = L2 fetch + write-allocate read
   + atomic/reduction RMW read + explicit prefetch/decompression terms; DRAM
   write demand = dirty eviction + terminal flush + atomic/RMW write + modeled
   store write. Coalescing is only within each source/direction, and any
   residual is emitted rather than hidden. Demand is never a scaled load
   counter.
6. **Completion correctness:** data-ready, store acknowledgement, visibility,
   atomic serialization, async-group completion, and terminal drain are
   distinct events. A kernel exit is not automatically a DRAM-drained point.
7. **Ordering:** `(time, sequence, event-class)` is the simulator's deterministic
   event order, not proof of hardware cross-SM temporal order. When trace
   witnesses do not establish cross-SM order, preserve a partial-order/unknown
   domain and run bounded interleaving perturbations rather than imposing a
   false total order.
8. **Physical mapping:** virtual/GPU addresses do not reveal physical DRAM
   partition, bank, row, or controller mapping. Emit effective service-class
   IDs only for an explicit effective mapping scenario; physical IDs and
   achieved throughput remain unavailable/open without a physical witness.
9. **Input safety:** malformed sidecars, stale architecture metadata, invalid
   widths/addresses, impossible masks, arithmetic overflow, and unsupported
   operation classes fail closed per launch with a useful typed reason.
10. **Corpus evidence:** no synthetic fixture acceptance path is introduced.
   Mechanism probes are real CUDA/NVBit/NCU or retained real sidecar traces.
11. **Full-mode semantics:** full mode enables all implemented effective
   components together. A missing physical fact produces an assumption/interval
   and residual, not a hidden mode switch.

The serialized metric union is explicit rather than prose-only:

```text
Exact(value, unit, endpoint, clock, denominator)
FiniteInterval(lo, hi, unit, endpoint, clock, denominator)
LowerBound(value, unit, endpoint, clock, open_tail)
Unavailable(reason, scope)
Falsified(reason, observed_mismatch)
```

Initial cache state, cache reset/warm state, launch gaps, stream scope and
carry-in/carry-out state are explicit inputs; a single-launch trace cannot claim
stream-L2 persistence.

## 4. Target full-replay model

### 4.1 Typed transaction and address stream

Introduce a generic SM86 transaction adapter (in `sim/mem`, shared with
`sim/sm`) around the existing `AccessKind`, `MemRequest`, and sector-expansion
records. Migrate incrementally rather than creating a parallel object graph:

```text
Transaction {
  launch/cta/warp/lane/pc/occurrence identity,
  issue_cycle and producer frontier,
  address_space and cache operator,
  operation class: load/store/atomic/reduction/async/texture/surface/local,
  memory_scope/order, return-value-use, alias class,
  virtual line, allocation_id/generation/status, sector mask, byte mask,
  logical data width and transfer width, predicate/active mask,
  source/destination route, requester SMSP/SM,
  parent/child request identity and generation,
  completion requirements: data_ready, ack, visible, group/barrier,
}
```

Formation must preserve per-lane widths and addresses until sector expansion is
complete. Coalesce only at a declared hierarchy boundary; retain both the
logical instruction count and lower-level sector/request count. This is needed
for the 69-sector misalignment anchor, local-wide accesses, RED RMW traffic,
and operator-specific L1 routing.

Local addresses require thread-local stack/frame canonicalization before any
line comparison; shared addresses require bank/cluster scope; allocation reuse
and cross-allocation records invalidate physical-owner assumptions. Missing
`cache_op`, `memory_op_access_kind`, lane addresses, allocation identity or
transfer width is typed unknown, not a legacy default. Texture/surface remains
unavailable until a distinct tracer/classifier route exists.

### 4.2 SM86 issue, scheduler, and IPC model

Keep `IssueModelParams::from_card` and `IssueResourceSource::Carded` as the
carded SM86 source of truth. Add a thin `Sm86FullConfig` wrapper/extension for
new queue, operand, frontend and IPC fields instead of wholesale-renaming the
existing issue configuration:

- four SMSPs per SM, one issue transaction per SMSP per cycle by default, no
  dual issue unless an explicitly measured probe changes the card;
- compiler-declared stall/yield/read-barrier/write-barrier fields from SASS;
- per-warp scoreboard and register readiness, variable-latency producers, and
  memory response tokens;
- CTA residency from traced block geometry, registers, shared memory, 48-warp
  and 16-block ceilings, with explicit placement/co-residency assumptions;
- SASS-level pipeline classes (FP/INT, SFU, LSU, tensor HMMA/IMMA/TF32,
  control/barrier) with measured latency and initiation intervals;
- a configurable warp-selection policy (initially current deterministic policy,
  then oldest/youngest/greedy-eligible candidates) scored against independent
  issue/throughput probes, never selected from a workload's NCU result;
- optional register-bank/reuse-bit pressure and instruction-stream buffer
  envelopes. The mechanism is enabled only when static operands/control data
  are present; otherwise it emits a typed unavailable contribution.

Compute IPC from several denominators:

```text
ipc_issue_window       = issued_instructions / issue_frontier_cycles
ipc_sm_active          = issued_instructions / SM-active frontier
ipc_sm_elapsed         = issued_instructions / launch/envelope frontier
ipc_smsP_active        = per-SMSP issued / per-SMSP active frontier
ipc_memory_aware       = issued_instructions / max(issue, memory, writeback, visibility)
```

Retain separate instruction numerators: issue attempts, successfully executed
instructions, replayed instructions, and records retired. NCU
`smsp__inst_issued.sum`, `smsp__inst_executed.sum`, and their SM rollups are
not interchangeable; if the trace cannot identify replay/redo causes, the
corresponding IPC comparison is unavailable rather than silently using issue
attempts as executed work.

The report must retain raw frontiers and the causal readiness histogram so a
low IPC can be attributed to long scoreboard, short scoreboard, LSU throttle,
tensor/resource throttle, barrier, front-end, occupancy, DRAM service, or tail.

Do not revive the rejected composite LDGSTS front-end constants from the August
2026 experiment. The measured experiment worsened transfer MAE and interaction
signs. If LDGSTS issue-cliff behavior is revisited, run the declared direct
40-cell barrier-free assay first and keep it an independent candidate.

### 4.3 L1TEX model

Replace the current SM120/effective-L1 framing with `Sm86L1Config`:

- 32-B sectors and 128-B lines;
- cache-operator route: `.ca`/eligible `LDG.E`/local eligible traffic to L1,
  `.cg` to the measured L2-only route, and `.cs`, atomics, texture/surface and
  LDGSTS kept as explicit probe-dependent routes rather than assumed to share
  one bypass path;
- configurable effective capacity per shared-memory carveout, using the
  counter-free p-chase intervals (base approximately 98,304 B with an upper
  transition bracket near 163,840 B; carveout-specific values remain separate);
- configurable sets/ways/index hash/replacement (current four-way/256-set
  card as the starting declaration, p-chase associativity/replacement as
  unresolved intervals);
- line allocation and resident-sector fills, sector-valid masks, MSHR merge,
  finite fill/request/response queues, hit-under-miss, retry, and eviction;
- L1 lookup/sector counters split by Global/Local/Store/Atomic/Reduction and
  operator route; write-through/no-allocate remains the explicit baseline until
  dirty resident/writeback probes justify the resident-hit policy;
- shared-memory wavefront/bank accounting remains outside L1 traffic but feeds
  the shared pipeline and IPC readiness.

Texture/surface records are not currently represented as distinct `AccessKind`
variants. They remain typed unavailable until tracer/classifier and hierarchy
scopes are admitted; they must never be silently folded into Global traffic.

The current `L1Cache` state machine is a good semantic base. The required
change is to separate state semantics from geometry policy and to make service
queues explicit instead of letting latency alone stand in for admission.

### 4.4 Six-slice L2 model

Refactor `L2Model` into a GA102 configuration with:

- 6,291,456 B capacity, six slices, 128-B lines, 32-B sectors;
- configurable slice hash and set mapping (the existing six hash candidates
  become measured/effective candidates, not an arbitrary `LineMod` default);
- set-associative replacement candidates (LRU, SRRIP, BIP/PLRU where observed);
- non-inclusive state, write-allocate stores, partial-sector validity, dirty
  masks, generation-safe pending fills, and per-source lookup accounting;
- per-slice request queues, MSHR/pending-request merge, response queues,
  cross-SM arbitration, and bounded retry/backpressure;
- optional stream/persistent L2 scope only when capture device/stream identity
  proves a single ordered stream; launch-cold remains the comparison lane;
- L2 prefetch/operator hints represented as explicit transactions, not as an
  unexplained hit-rate multiplier.

The old L2 predictor's historical CV and same-cell correction factors are
starting diagnostics only. Future geometry selection must use independent
address/capacity/hash probes and family-disjoint replay residuals.

### 4.5 Fabric, partition, and DRAM service

Add an effective lower-memory service layer between L2 and DRAM:

1. **L2-to-fabric packetization:** map each fetched/writeback sector to a
   request packet with source SM/slice, operation, priority, and byte count.
2. **Partition mapping:** implement a parameterized address-to-partition/bank
   mapping. Start with a small predeclared candidate set and infer bounds from
   stride/partition-throughput probes; do not copy constants from another SKU.
3. **Finite queues (planned timing arm):** request, return, writeback, atomic,
   and maintenance queues must eventually have explicit capacities/occupancy,
   with admission and service as separate frontiers. H30–H37 currently expose
   route/slice/fill/pressure observers only; they do not impose a hidden GA102
   queue capacity or timing delay.
4. **DRAM controller (current versus planned):** current full replay uses the
   aggregate fixed-point carded read server and its declared effective-rate
   interval. A future timing-active arm may model partition/queue service with
   bounded FR-FCFS-like/open/closed-row candidates, burst service, and
   read/write high-watermark arbitration only after an independent locked-clock
   probe and typed mapping witness. Never derive physical row IDs from virtual
   address bits. Use the 1,008 GB/s card peak only as an upper/service
   hypothesis; infer effective efficiency from independent streaming and mixed
   read/write probes. Physical effective service-class IDs remain unavailable
   unless an explicit effective mapping arm is declared; physical
   partition/bank/row IDs remain unavailable without a physical-page/controller
   witness.
5. **Overlap:** issue independent reads to different banks/partitions; serialize
   only where the configured resource or inferred conflict requires it. Preserve
   fixed-point arithmetic and never charge the same latency twice.
6. **Writes:** distinguish store acknowledgement, L2 dirty eviction, terminal
   dirty flush, DRAM write service, and profiler-visible write counters.
7. **Atomics/reductions:** account for RMW fetch sectors, line/partition
   serialization and returned-value readiness independently from ordinary loads.

The existing `DramBw::Carded` aggregate server is the **current** full-mode
read-service implementation, with H30–H37 partition/route counters retained as
observer covariates. The finite partition/bank controller is a future
timing-active implementation slice; until it has independent evidence and a
real-trace differential, it must not be described as active or substituted by
an imported topology constant. `DramBw::CardedWriteback` remains an explicit
scenario arm rather than a default completed writeback model.

### 4.6 Visibility, coherence, and asynchronous operations

Wire existing opt-in primitives into the SM86 full path with explicit scopes:

- L1 visibility observations and invalidation queue for cross-SM global writes;
- L2 directory-like effective ownership (read/read sharing, write/atomic
  invalidation/owner transfer) with sector granularity;
- `MEMBAR`, CCTL invalidation, store acknowledgement, and fence readiness as
  distinct event classes;
- `cp.async`/LDGSTS source visibility, group commit/wait, destination shared
  wavefronts and transport as separate dependencies;
- constant/immediate cache and shared-bank paths as independent resources;
- payload/value ledgers only when the sidecar supplies a complete generation and
  identity witness; otherwise preserve a typed unknown tail.
- virtual-address aliases, page translation/TLB/page-walk effects, cache
  compression/ECC effects, and physical-page identity remain explicit unknowns
  unless a sidecar witness exists; alias atomics cannot be treated as ordinary
  coalesced requests.
- Multiple CUDA streams, ordered CUDA events, concurrent kernels, copy-engine
  work, context switches and host launch gaps are separate dependencies. When
  the trace lacks a total cross-stream order, do not serialize them by
  simulator sequence; mark launch-envelope/cache carry-in as partial-order
  unknown and report the omitted overlap domain.

These mechanisms are model-local effective semantics. Full replay enables every
implemented, evidence-backed observer/model by default, including the current
aggregate read service and store/fence feedback candidate. Bounded dirty
writeback, acknowledgement extensions, and visibility/coherence arms activate
only when typed trace events make the mechanism applicable and their explicit
scenario configuration is supplied; the default runner does not synthesize
those optional flags. Otherwise it retains typed
unavailable tails and interval uncertainty rather than silently inventing
completion. Payload forwarding and physical ownership remain unavailable
without witnesses. None of these arms claims proprietary ownership, router,
or engine state.

### 4.7 Frontiers and reports

Every launch report must include:

- exact instruction/record/sector/request counts;
- issue attempts, executed instructions, replayed instructions and retired
  records with explicit semantic domains;
- L1 lookup/hit/miss/merge/stall counters and route splits;
- L2 lookup/hit/miss/merge/eviction counters by source and effective slice;
- DRAM read demand, atomic/reduction RMW demand, dirty/writeback demand,
  effective partition/service-class demand, queue delay, and completion counts;
  row/bank hits and physical partition IDs are
  `Unavailable(physical_mapping_absent)` unless their explicit effective arm is
  active;
- issue, modeled event, SM active, launch envelope, visibility, writeback,
  terminal tail, and typed unknown frontiers;
- IPC projections with numerator/denominator/endpoint metadata;
- readiness-cause cycles and critical-path contributors;
- configuration identity and uncertainty intervals.

For every L1/L2/fabric/partition/DRAM queue, expose enqueue, dequeue-to-service,
service-complete, in-flight, queued occupancy (excluding in-flight), and maximum
occupancy. The invariant is
`enqueued - dequeued_to_service - in_flight = queued_occupancy` and
`dequeued_to_service - service_complete = in_flight` after accounting for
cancelled/stale generations. Reject negative/overflow states, duplicate parent
completion, unbounded retry, and deadlock/nontermination.
When finite queues are disabled, the aggregate G/D/1 differential arm must
match the latency-only/reference result within declared rounding semantics.

If a metric is absent in both retained replay modes, the NCU report must place
it in one compact family-grouped “currently unavailable” section with exact
names/counts in machine-readable output. Do not print repeated `N/A` tables or
raw NCU distributions for unavailable metrics.

### 4.8 Required composition ablations

The all-capabilities full mode is already the default for the implemented
evidence-backed components. Run the same real roots through this declared
differential ladder whenever a **future timing-active** finite-service,
writeback, atomic, or visibility arm is proposed for composition:

```text
flat frontier
  -> L1 only
  -> L1 + L2 state
  -> L1 + L2 + aggregate read service
  -> + partition/queue service
  -> + writeback/atomic service
  -> + visibility/coherence arm
  -> + IPC/frontend/operand projections
```

Each step must preserve instruction/sector conservation, expose its incremental
frontier and avoid charging a latency twice. A mechanism is composed into the
default only after its real-trace arm terminates, its counters are bounded, and
the differential residual is understood. This is an implementation comparison
matrix, not an authority or admission workflow and not a reason to use NCU as a
same-workload parameter fitter.

## 5. Implementation map and refactor seams

### 5.1 Architecture and configuration

- Extend `arch_profiles/cards/sm_86.json` with explicit SM86 full-model blocks:
  L1 capacity/carveout intervals, L1 policy candidates, L2 hash/policy
  candidates, partition/service interval candidates, DRAM efficiency interval,
  queue capacities, and front-end/issue resource declarations.
- Preserve measured values and typed gaps separately. Add a new card schema
  version rather than mutating a value into a false measurement.
- In `sim/arch/src/card.rs` and `sim/arch/src/lib.rs`, expose a typed SM86
  card projection and reject cross-architecture binding. Unknown card blocks
  are ignored until projected, so add explicit schema/version handling before
  consuming new interval blocks; keep experimental scenarios in a separate
  configuration document until that migration exists.
- Replace SM120-only defaults in `sim/direct_run/src/lib.rs` with an SM86 full
  configuration while preserving compatibility wrappers until all callers are
  migrated. Remove SM120-only full-replay restrictions from the SM86 default,
  but retain typed errors for unsupported or unevidenced capability arms.
- Demote the current `sim_arch::l2_defaults(Sm86)` 16-way/LRU/LineMod values
  to an explicit candidate world until independent mapping/replacement probes
  support them; do not let a capability-table default silently choose an
  unresolved topology. Add a native SM86 L1 constructor rather than routing
  through `L1Config::effective_sm120`.

### 5.2 Trace and sidecar adapter

- Generalize `tools/sm120_replay_bench.py` into
  `tools/sm86_replay_bench.py` or a generic `replay_bench.py`.
- Accept `architecture_version` per kernel as authoritative; treat the stale
  producer label as legacy metadata only after checking runtime device/CC. The
  current checkout has no usable `tools/sass_control/emit_control_bits_sidecar.py`
  path, so the adapter must convert `extra_info/native_static_metadata.json`
  directly and cross-check PCs/function IDs against
  `extra_info/enhanced_execution_info.json`.
- Emit a generic static control schema with runtime `arch=sm_86`, source
  instruction architecture, control-layout version, normalized SASS, operands,
  cache operator, and instruction identity.
- Retain per-kernel `sm80_code_on_sm86_runtime` metadata for the current
  version-80 inventory and preserve `code_arch=80` alongside `runtime_arch=86`.
  Do not silently relabel it to native SM86; mixed roots need per-function
  disposition rather than root-wide rejection.
- Emit the exact parser contract expected by `ControlSidecar`: normalized
  function names without `___digits`, `0x`-prefixed high/low words, control
  layout shift 41, decoded stall/yield/read/write/wait fields, and summary
  counts. Reconcile the producer's wire `yield_bit`/`control_fields.is_yield`
  semantics explicitly; do not copy an inverted-looking field blindly.
- Keep dynamic contract identifiers as producer provenance; route behavior from
  decoded memory space/opcode/architecture, not a string containing `sm120`.

### 5.3 Memory crates

- `sim/mem/src/access.rs`: add the transaction adapter/operation metadata and
  share sector/byte-mask formation across L1/L2/DRAM without duplicating the
  existing `AccessKind`/`MemRequest` state machine in one large rewrite.
- `sim/mem/src/l1.rs`: split geometry/policy/service from tag state; add set/way
  indexing, finite request/response queues and route-specific counters.
- `sim/mem/src/l2.rs`: make six-slice GA102 geometry and effective hash/policy
  configurable; add slice queue and partition handoff metadata.
- `sim/mem/src/writeback.rs`, `coherence.rs`, `l1_visibility.rs`: wire the
  existing bounded primitives into full mode with typed event ownership,
  generation checks, and separate completion frontiers.
- Add `sim/mem/src/partition.rs` and `sim/mem/src/dram_controller.rs` for
  effective partition/bank/row service. Keep fixed-point arithmetic and checked
  overflow; expose a simple aggregate ablation for differential diagnosis.
- `sim/semantics/src/dram.rs` and related boundary-state types must serialize
  the new queue/bank state without making a digest/receipt an execution gate.

### 5.4 SM issue and system event engine

- `sim/sm/src/issue/config.rs`: extend the carded SM86 issue path with a thin
  full-mode wrapper, per-pipeline intervals, scheduler policy,
  register/operand pressure, and frontend model.
- `sim/sm/src/issue.rs`: connect transaction issue to cache admission and
  queue backpressure; preserve compiler barriers and per-warp readiness.
- `sim/sm/src/system.rs`: retain `DramBw::Carded` as the current full-mode
  aggregate read-service arm. A future evidence-backed slice may replace it
  with the effective partition/DRAM controller only after differential replay;
  until then H30–H37 partition candidates remain observer-only.
- `sim/sm/src/replay.rs`: use one shared L1/L2/system instance per launch or
  explicit stream scope. Preserve simulator event ordering while marking
  hardware cross-SM order partial when the trace lacks a witness.
- Add partition/service model state and new `Ev` variants in
  `sim/sm/src/system.rs` first, preserving current event phase order and typed
  snapshots. Defer a generic `sim/engine` phase API until real-trace
  differential replay proves it does not perturb existing event boundaries.

### 5.5 Direct runner, metrics, and tools

- `sim/direct_run/src/main.rs`: replace SM120 flags with an SM86 full-mode
  default; keep a flat/fast mode only as a complementary frontier.
- `sim/direct_run/src/metrics.rs` and `emit.rs`: emit all SM86 families,
  typed intervals and separate frontiers; preserve exact unavailable reasons.
- Refactor `tools/sm120_corpus_replay.py` into the architecture-parameterized
  complete-corpus runner required by the repository contract; an optional
  `tools/sm86_corpus_replay.py` may be a thin entrypoint. Preserve exactly
  twelve workers, per-root logs, timeout/output bounds, cleaned tracer
  environment, compressed `.jsonl.zst`, and full-before-fast ordering.
- Add `tools/sm86_corpus_compare.py`/`sm86_metric_report.py` to join exact
  selected NCU launches and report unmatched/full-corpus coverage separately.
- Add post-replay reports for family residuals, L1/L2 sensitivity, DRAM service,
  IPC causes, and frontier decomposition. These are descriptive reports, not
  replay gates.

## 6. Measurement and probe program

All probes below are real CUDA/NVBit/NCU captures or retained real sidecars.
They are used to constrain independent model parameters and to expose
mechanism-local residuals; scored corpus NCU rows are not fed back into the
same launch's configuration.

### M0 — host/corpus and trace contract

- First ingest/qualify retained real sidecars and probe reports already under
  `/home/jakeshea/GPU_sims_notes` (L1 admission/arrival, L2 queue frontier,
  memory-retirement/store-ack matrices, scheduler contention, DRAM streams,
  atomics, cp.async, prefetch/persist and carveout studies). Record each
  artifact's independent/development status and rerun only missing mechanism
  gaps; do not treat historical snapshot labels as authority.
- Independently query GPU name, UUID, CC, device index, driver, CUDA, clocks,
  memory clock, and P-state on the 3090 Ti host.
- Reconcile the 356 roots (354 clean + 2 typed-invalid), nested trace count,
  327 catalog rows, 356 profiles, the version-80 inventory, stale metadata
  labels, and diagnostics.
- Generalize static sidecar generation and replay the three verticals plus one
  native SimBench L1/L2/DRAM root in flat and full modes.
- Differentially compare the new metadata converter against the three retained
  real control sidecars (FP32, HMMA, cp.async), comparing PCs/function IDs/
  control/opcode fields; any mismatch fails closed for that root.
- Completion evidence: every row records explicit architecture disposition,
  sidecar source, selected launch identity, dynamic-reported versus
  records-replayed counts, and unsupported/unmapped address counts; no
  generated plain JSONL.

### M1 — L1 route, capacity, and admission

Use qualified retained real SimBench/capture roots first, then fresh independent
probes only for gaps, for:

- `.ca`, `.cg`, `.cs`, `.cv`, bare `LDG.E`, local, atom, RED, texture/surface;
- aligned 32/64/128-bit, misaligned, partial warp and variable-width accesses;
- working-set/capacity sweeps under carveouts 0/25/50/75/100%;
- set-conflict stride grids and replacement/re-reference patterns;
- dependent, ILP4, ILP16 and many-warp miss streams for MSHR/admission;
- L1 request/response queue and fill-under-miss saturation.

Record L1 sector lookups, hit/miss, line allocations, sector fills, merges,
MSHR stalls, queue delay and NCU lookup/sector comparison fields. Produce an
effective interval card only from predeclared counter-free/public/independent
probe evidence; NCU values remain post-replay diagnostics and cannot select or
fit the card.

### M2 — L2 slices, hash, replacement, and persistence

- Ingest qualified retained L2 queue/prefetch/persist/thrash sidecars before
  scheduling new GPU runs; preserve falsifier and unknown-order dispositions.
- Sweep footprints around 1/2/3/4/6/8/12/16 MiB with dependent and parallel
  streams to distinguish L2 hit latency from capacity.
- Use stride and base-offset families to infer slice/partition mapping and
  conflict sets; keep candidate hashes as a partial order when not identifiable.
- Contrast LRU/SRRIP/BIP/PLRU candidates with re-reference distance and
  thrashing streams.
- Exercise `.cg`, atomics, RED, stores, prefetch hints, persisting windows,
  local memory and multi-SM sharing.
- Measure launch-cold versus same-stream persistence only where launch identity
  proves a valid scope.

Output slice/hash likelihood bands, effective set/way candidates, request
coalescing/PRT behavior, L2 queue bounds, and family-specific residuals.

### M3 — partition/fabric/DRAM and writeback

- Ingest retained memory-retirement, store-ack, DRAM-stream and atomic service
  matrices first; keep instruction-mismatch and lower-bound-falsifier cells out
  of parameter selection while retaining their residuals.
- Streaming read, write, read/write mix, random, row-hit/row-conflict and
  bank-conflict probes at multiple strides and working sets.
- Sweep active SM/CTA/wrap counts and independent request depth to expose finite
  partition and DRAM queue throughput.
- Exercise 4-MiB and 16-MiB dirty stores, L2 evictions, terminal dirty
  residency, atomics and RED/RMW.
- Measure effective bandwidth under the locked operating point and no-clock
  control first; later clock sweeps are separate calibration evidence.
- Record read/write arbitration, service delay, row hit/miss, queue occupancy,
  writeback completion, DRAM sector/byte counts and profiler-visible tails.

The output is an effective DRAM service interval and queue/bank candidate set,
not a claim of exact GDDR6X controller topology. Without an independently
supported mapping/service arm, row-hit/miss and achieved-bandwidth fields stay
typed unavailable while demand and lower-bound service remain reportable.

### M4 — IPC and scheduler mechanisms

- Ingest retained scheduler/issue-port/frontend/constant-load reports first;
  quarantine known falsifier cells (including issue-port, SFU, constant and
  result-queue mismatches) rather than clipping their residuals.
- SASS-level dependent/independent FP32, INT, SFU, HMMA FP16/BF16, TF32,
  IMMA INT8, sparse HMMA, ldmatrix/stmatrix, shared-bank,
  branch/reconvergence, barrier and mixed-memory roots.
- Vary warp count, CTA count, register footprint, shared-memory carveout,
  compiler stall/yield/dependency fields, operand reuse bits and source/dest
  register-bank patterns.
- Separate issue, active, memory-ready, barrier, front-end and terminal tails.
- Add a dedicated control-flow arm for SSY/BRA/SYNC/reconvergence stacks,
  predicated-off issue, divergent path occupancy and fetch-loop effects; do not
  hide these under ordinary scoreboard or barrier cycles.
- Measure SASS instruction throughput/latency, scheduler policy, register-file
  pressure, instruction-stream footprint and memory-pipeline contention.

Select bounds only from predeclared independent probe observations, public
facts, or counter-free clock measurements. NCU counters remain post-replay
comparison targets even for probe workloads. Predeclare family partitions
(native SimBench/SOL, adapted controls, substitutes, snapshots) and leave one
partition out when reporting generalization; use full-corpus residuals only for
post-hoc diagnosis, not iterative parameter selection.

### M5 — full-corpus replay and post-replay comparison

- Run full mode on all valid roots with twelve workers, then optional fast mode.
- Retain per-root compressed rows and bounded logs; never mix outputs with raw
  captures.
- Join the one selected NCU launch per existing root only when kernel name,
  grid/block, device/stream and a unique identity match; classify duplicate
  candidates and all profiler-from-start-off shape-only joins separately as
  ambiguous ordered/shape fallback.
- Add a second NCU campaign for representative multi-launch roots if launch-
  level IPC/DRAM claims are required. Until then, label comparison coverage as
  selected-launch/root-level rather than full-launch ground truth.
- Report aggregate, family, native/control/unsupported, and mechanism-stratum
  residuals for L1, L2, DRAM, cycles, stalls and IPC.

## 7. Full replay runner contract

The desired command shape is:

```bash
cd /home/jakeshea/5090_branch/GPU_simulator
(cd sim && cargo build -p sim-trace --release && cargo build -p sim-run --release)
python3 tools/sm120_corpus_replay.py \
  --capture-root ../sm86_rtx_3090_capture \
  --arch-card arch_profiles/cards/sm_86.json \
  --pchase-card data/pchase/sm_86_static_card.json \
  --output ../sm86_replay_results/sm86_full_YYYYMMDD \
  --mode full \
  --timeout 1200 \
  --max-sim-output-bytes 536870912 \
  --ncu-shaped-metrics \
  --workers 12
```

Required runner behavior:

1. Discover only valid roots with `dynamic_trace.pb`; report diagnostics and
   duplicate/nested traces without silently treating them as independent roots.
2. Synthesize or load one generic SM86 control sidecar per root, preserving
   per-kernel architecture and static identity.
3. Translate runner `--mode full` to the binary's SM86 full hierarchy
   (`--mode l1-l2 --pchase-card ...`) and auto-enable NCU-shaped metric keys.
   `--cache-scope launch|stream-l2` remains explicit. Full mode includes L1,
   L2, aggregate effective DRAM read service, observer-only partition/route
   candidates, atomic/RMW accounting, IPC projections, and the defaulted
   evidence-backed store/fence feedback candidate. The SM86 runner explicitly
   enables `--carded-dram` and records the default 825 GB/s effective-rate
   hypothesis; it also auto-selects the launch-scoped store/fence candidate.
   Resident dirty writeback, coherence/visibility, and payload forwarding
   remain typed unavailable in the complete-corpus runner: their numeric/flag
   scenarios are currently direct `sim-run`/prototype-only and are not silently
   synthesized by the corpus driver. Do not relabel observer-only partition
   candidates as physical service. Persistent `stream-l2` scope suppresses the
   launch-scoped store/fence candidate and records that scoped gap rather than
   emitting replay errors.
4. Keep optional fast/flat replay explicitly complementary and never use it as
   the full-mode result.
5. Enforce exactly twelve concurrent simulator processes, bounded stdout,
   timeout, cleaned `LD_PRELOAD`/NVBit environment, isolated stderr and
   per-root status.
6. Write `full.jsonl.zst`, `fast.jsonl.zst` (if requested), summary JSON, and
   post-replay joins through `tools/jsonl_io.py`. Every JSONL reader must also
   use `jsonl_io.py`, transparently accepting legacy plain JSONL plus new
   `.jsonl.zst`; the three historical `replay/*.json` single-document files
   require a separate one-document adapter and are not treated as JSONL.
7. Preserve successful rows even if another root fails, but return a nonzero
   summary status and list every failed root/reason.
8. Record actual binary/card/trace/control/config paths and host identity as
   descriptive run fields; none is an approval or provenance gate.

## 8. Evaluation and reporting design

### 8.1 Accuracy surfaces

Report separate rows for:

- L1 lookup sectors, L1 hit rate and route-specific sectors;
- L2 lookup sectors, L2 hit rate, source/slice request counts;
- DRAM read sectors/bytes, atomic/reduction RMW read sectors;
- DRAM write sectors/bytes, dirty eviction/writeback service;
- active/elapsed cycles and each simulator frontier;
- SM and SMSP IPC projections, active warps/occupancy and stall causes.

The IPC metric registry keeps hardware and model domains distinct:

| Model projection | NCU comparison label | Required denominator |
|---|---|---|
| SM active IPC | `sm__inst_executed.avg.per_cycle_active` | summed SM-active cycles |
| SM elapsed IPC | `sm__inst_executed.avg.per_cycle_elapsed` | elapsed/envelope endpoint |
| SMSP instruction total | `smsp__inst_executed.sum` | exact SMSP execution domain |
| SMSP rollup | `smsp__inst_executed.avg` | per-SMSP average, never SM total |
| Issue-attempt diagnostic | `smsp__inst_issued.sum` | issue-attempt numerator, not executed work |

Model-relative rows retain their own names and endpoint descriptors. If the
frontier or semantic numerator is unavailable, the row is unavailable rather
than substituted across domains.

Use median, P90, mean absolute percentage/absolute error, signed bias and
family stratification, with paired bootstrap/quantile intervals where repeats
exist. Report root-balanced and launch/instruction-weighted views, repeat
dispersion and censor/open-tail status. For rates, report percentage-point
error and denominator coverage. For interval predictions, report coverage and
interval width rather than hiding uncertainty in a point score; a single NCU
row is not treated as precise ground truth.

### 8.2 NCU identity and coverage

Existing root profiles have one selected kernel row. The report therefore has
three explicit statuses and records the expected inventory: 356 NCU rows,
101 unique exact-name+shape candidates, eight duplicate-name candidates (five
roots with two and three roots with five), and 247 profiler-from-start-off
demangled rows with no exact-name candidate:

- `exact_selected_launch`: exact identity/shape join;
- `ordered_or_shape_fallback`: useful descriptive join, not independent proof;
- `no_ncu_target`: simulator row retained, comparison unavailable.

The full 15,165-launch simulator coverage is reported independently of NCU
coverage. A future multi-launch NCU campaign may expand exact target coverage;
until then, no per-launch accuracy claim is made for missing targets.

### 8.3 Residual diagnosis

Every residual report must identify the likely mechanism stratum without
pretending it is proven:

- L1 route/capacity/associativity/MSHR;
- L2 slice/hash/replacement/persistence/coalescing;
- partition/fabric queue or bank conflict;
- DRAM read/write arbitration/row service;
- writeback/visibility/atomic completion tail;
- scheduler/operand/register/front-end/tensor pipeline;
- launch envelope/clock/measurement scope;
- unsupported or stale sidecar metadata.

No global parameter sweep is accepted merely because one aggregate score moves;
future sweeps must be tied to a mechanism and a real trace/probe that exercises
it.

## 9. Phased implementation sequence

### P0 — branch and input reset

- Land the SM86-only memory/plan override in long-term docs (done by this
  plan's companion memory update).
- Inventory every SM120-specific symbol/flag and classify as remove, rename,
  or retain as generic behavior.
- Reconcile the 356 roots, the version-80 inventory, stale metadata labels and
  existing three legacy replay files.

**Exit:** a real-root inventory and a generic sidecar design, with no change to
raw captures.

### P1 — generic SM86 static/dynamic contract

- Implement generic static metadata parsing and SM86 control synthesis.
- Preserve exact per-kernel architecture and stale-producer diagnostics.
- Build `sim-trace`/`sim-run` and replay the three verticals plus one memory
  root in flat and full skeleton modes.

**Exit:** all selected real roots either produce a sidecar or a typed input
reason; no `sm120` string causes a false SM120 route.

### P2 — transaction/sector accounting

- Introduce the transaction record and common sector/byte-mask builder.
- Port aligned/misaligned/local/RED/atomic anchors into real captured traces.
- Emit exact instruction, request, sector, merge and operation-class counters.

**Exit:** sector totals agree with retained independent anchors on the real
  probe roots, and no width/address shortcut is used.

### P3 — L1 physical/effective topology

- Refactor `L1Cache` geometry/policy/service.
- Add carveout-specific capacity, set/way/hash candidates, finite queues and
  MSHR/merge accounting.
- Run M1 probes and write the SM86 effective L1 card.

**Exit:** `.ca`/`.cg`, capacity, conflict, MSHR and ILP behavior are visible in
  full replay; `.cs`/`.cv`/atomic/texture/surface/LDGSTS routes remain
  probe-dependent; unresolved geometry remains interval-labeled.

### P4 — six-slice L2 and L2 queues

- Make six slices, effective hash, replacement, pending-read merge and queue
  state first-class.
- Add source/slice counters and launch/stream scope handling.
- Run M2 probes and retain family residuals.

**Exit:** L2 traffic/hit projections are emitted across the real corpus. Any
  parameter selection is justified by independent probe evidence only; corpus
  and post-replay NCU residuals remain diagnostic/holdout evidence and cannot
  select the scored configuration.

### P5 — partition/fabric and DRAM controller

- Add partition/bank/row service and finite queues.
- Add read/write arbitration, atomics/RMW, dirty writeback and terminal flush.
- Run M3 probes and produce effective service intervals.

**Exit:** DRAM read/write demand and service are distinct; large-store residuals
  no longer disappear into an unknown terminal gauge when the required sidecar
  evidence is present.

### P6 — integrated completion and visibility

- Wire L1 visibility, L2 directory-like service, MEMBAR/CCTL, async-copy
  groups, write acknowledgements and atomic completion into the event engine.
- Keep generation/ownership/response identities checked and separate.

**Exit:** full mode can explain readiness blocking by data, ack, visibility,
  group, atomic or terminal service event without charging a duplicate latency.

### P7 — SM86 IPC and scheduler accuracy

- Refactor issue config and scheduler to SASS-level SM86 resources.
- Add tensor/SFU/LSU pipeline intervals, operand/register/reuse pressure,
  instruction-stream envelope and readiness cause counters.
- Run M4 and compare independent cycles/IPC only after replay.

**Exit:** IPC rows are endpoint-specific and mechanism-attributed; no PTX-only
  or single terminal-cycle shortcut remains.

### P8 — corpus runner and compressed artifacts

- Implement the SM86 mode of `sm120_corpus_replay.py` (and optional thin
  `sm86_corpus_replay.py` entrypoint) plus generic comparison/report tools.
- Full-first, exactly twelve workers, `.jsonl.zst`, isolated logs, bounded
  child output and explicit failure rows.

**Exit:** complete corpus full replay terminates and produces consumable output
  for every valid root or a precise typed failure list.

### P9 — corpus residual program

- Run full mode over all roots.
- Produce L1/L2/DRAM/IPC family reports and selected-launch NCU joins.
- Add independent NCU multi-launch campaign only if the remaining question is
  launch-level rather than mechanism coverage.

**Exit:** all 15,165 discovered launches have either a full simulator row or a
  typed input/disposition row; exact NCU coverage and all unavailable families
  are stated without overclaiming.

### P10 — closure and thesis-quality evidence

- Document/finalize the selected effective SM86 card values as
  intervals/conditional assumptions,
  not proprietary topology claims.
- Report full-corpus generalization, family residuals, uncertainty, simulator
  cost and bare-metal relative performance where available.
- Document remaining unidentifiable physical gaps as scoped residuals.

**Exit:** the 3090 Ti branch meets the project PhD closure criterion: complete
  corpus replay, effective/conditional cache/DRAM/IPC model, and independent
  NCU comparison with explicit residuals.

## 10. Risks and explicit non-goals

### Risks

- The current capture metadata contract is stale and mixed-architecture; the
  generic adapter must repair interpretation without rewriting raw evidence.
- NCU coverage is one selected launch per root; expanding it may dominate the
  measurement schedule.
- L1 associativity, L2 hash/capacity transition, MSHR depth, partition mapping,
  DRAM bank count and writeback completion are only partially identifiable.
- Full event-driven queues may increase replay cost and memory use on the
  largest 7.2M-instruction roots; use bounded diagnostics and twelve workers,
  not silent truncation.
- Clock/P-state/driver variation can move cycle/wall-clock observations; retain
  cycle-domain and duration-domain results separately.

### Non-goals

- Recovering proprietary RTL, exact physical cache tags, MSHR counts, scheduler
  ports, router topology, or DRAM command traces.
- Relabeling SM120-only features as native SM86.
- Using Accel-Sim, NCU fitting, same-cell correction factors, or labels as
  implementation-selection criteria.
- Reintroducing authority/receipt/hash/approval/sealing/admission workflows.
- Adding synthetic fixture acceptance tests or treating a passing self-authored
  test as independent hardware evidence.
- Claiming that an effective queue or IPC projection is physically exact merely
  because it improves one aggregate score.

## 11. Immediate next three implementation slices

1. **Generic SM86 adapter/runner:** repair stale static metadata handling,
   generate SM86 control sidecars, reconcile roots, and replay all valid roots
   in a bounded full skeleton with `.jsonl.zst` output.
2. **L1/L2 transaction seam:** introduce the common transaction/sector record,
   make the existing L1/L2 state machines consume it, and emit route/source/
   merge counters on the real corpus before adding new timing constants.
3. **DRAM service skeleton:** the current full mode retains aggregate G/D/1
   admission plus observer-only partition candidates. Replace it only in a
   future, independently probed slice with a parameterized partition/queue
   controller that first reproduces the aggregate/card-rate result, then add
   measured read/write arbitration and dirty-writeback arms.

These slices are deliberately vertical: each must run a representative real
trace and inspect counters/frontiers before the next mechanism is widened.

## 12. Slice S1 implementation evidence — 2026-10-15

The first implementation slice is complete: the architecture-neutral static
metadata adapter and architecture-parameterized corpus runner are operational.

- `tools/sm120_replay_bench.py` now derives deployment architecture from
  explicit metadata, accepts the stale SM86 `native_static_sm120_v3` producer
  label, emits an SM86 control sidecar, and rejects mixed/unsupported roots with
  typed control dispositions.
- `tools/sm120_corpus_replay.py` now defaults to the SM86 corpus/card/p-chase
  card, translates full mode to integrated `l1-l2`, enables carded DRAM for
  SM86 by default, preserves `--no-carded-dram`, enforces twelve workers,
  bounded child execution, full-first ordering and `.jsonl.zst` output, and
  excludes `_diagnostics` from unfiltered discovery.
- Real vertical batch: full 3/3 `ok`, fast 3/3 `ok` for FP32 FMA, FP16 HMMA,
  and `cp.async`; full/fast artifacts are retained outside the raw corpus.
- Complete-corpus full replay: 356 roots discovered, 286 `ok`, 62 explicit
  mixed-code control dispositions, 8 replay/input failures, 12,392 launches
  replayed in successful rows, and 32.845 s driver wall time. Output is
  `../sm86_replay_results/full_corpus_20261015_v2/full.jsonl.zst`.
- The 62 mixed-code roots contain SM80 code objects that the current simulator
  rejects at schedule binding; they remain explicit controls rather than being
  relabeled as native SM86. The eight replay failures are retained with their
  real causes (unsupported sparse IMMA pipeline, resource-size limits,
  local-memory opcode mismatch, grid-range mismatch, or unresolved allocation
  identity).

This slice proves runner/sidecar/corpus termination and coverage accounting,
not sub-10% NCU accuracy. The next slice is metric reduction and mechanism
coverage over the retained compressed full artifact, followed by an evidence-
backed L1/L2/service refinement; no NCU value is fed into replay selection.

## 13. Slice S2 implementation evidence — SM86 L2 write-observer policy

An evidence-backed metric-only refinement is now implemented in
`sim/sm/src/system.rs` and `sim/direct_run/src/metrics.rs`:

- Ordinary SM86 store sectors use a conditional all-write-hit lookup observer;
  RED retains separate RMW semantics. Physical L2 state, DRAM demand, and
  timing are unchanged.
- The observer sums operation-qualified read/store/atomic/reduction lanes with
  checked arithmetic and emits explicit L2 lookup hit and miss metrics.
- Independent Ampere write-validate/write-allocate research supports the arm;
  no scored NCU value selected this policy.

Evidence:

- Representative real full replay: FP32, FP16 HMMA and cp.async completed 3/3.
  cp.async reached an exact 256-sector / 128-hit / 128-miss L2 observer on its
  selected NCU row; FP32 and store-heavy SimBench controls also matched their
  write-split anchors, while FP16 retains a 64-sector conditional hit residual.
- Complete real full replay after the refinement: 356 roots, 286 `ok`, 62
  mixed-code control dispositions, 8 replay/input failures, 12,392 successful
  launches, 33.625 s wall time, 12 workers. Artifact:
  `../sm86_replay_results/full_corpus_refinement_l2_write_20261015_v1/full.jsonl.zst`.
- Exact-identity NCU comparison remains 107 rows; ordered/shape fallback is
  retained but not scored. L2 lookup-hit mean APE improved from 54.52% to
  35.04% and lookup-miss mean APE from 84.20% to 27.00%; both medians are 0%.
  Source-total L2 APE remains 28.24%, DRAM-read APE 40.91%, SM-active IPC APE
  14.57%, and elapsed IPC APE 12.41%.

The slice improves the L2 observer but does not meet the sub-10% target. Local
address expansion, constant/GCC demand, DRAM service, and IPC denominator
residuals remain open for subsequent mechanism slices.

## 14. Slice S3 implementation evidence — H2 constant-domain demand

H2 was adversarially corrected after reviewing the producer and retained
independent notes. The wire semantics are important: field 15 contains
active/predicate-on bank-local effective CBANK offsets, field 16 is the bank,
and field 17 is a scalar metadata offset that can be present on a predicated-
off record. The simulator now keeps the bank as `Option<u32>`, counts
metadata-only records explicitly, and never uses field 17 as a lane-demand
fallback.

The full default path emits checked source-domain observations:

- active offset-reference and active-offset address counts;
- per-instruction distinct 32-B offset packets;
- launch-wide distinct 32-B offset-domain sectors and 64-B candidate lines;
- field-17-only metadata records;
- the existing L2-missing plus compulsory-instruction demand term, kept
  separate from constant-domain observations.

None of these counters mutates L1/L2 state or DRAM timing, and none is
silently equated with a GCC request, DRAM sector, cache fill, or service
completion. Static `LDG.*.CONSTANT` source attribution remains open.

Evidence:

- Four predeclared real roots completed integrated full replay 4/4; artifact
  `../sm86_replay_results/h2_constant_demand_20261015_v5/full.jsonl.zst`.
- Corrected complete-corpus full replay completed 356 roots (286 `ok`, 62
  mixed-code controls, 8 replay/input failures), 12,392 successful launches,
  34.263 s, and exactly 12 workers. Artifact:
  `../sm86_replay_results/full_corpus_h2_constant_20261015_v2/full.jsonl.zst`.
- Across 107 exact-identity rows, active vectors cover 2,845,702 offset
  references; 3,840 metadata-only records remain explicitly unavailable.
- The post-replay report is
  `../sm86_replay_results/metric_report_h2_constant_20261015_v3/`; L2/DRAM/IPC
  scored residuals are identical to S2 (35.04%, 27.00%, 28.24%, 40.91%,
  14.57%, and 12.41% mean APE respectively for L2 hit, L2 miss, source-total,
  DRAM read, SM active IPC, and elapsed IPC).

H2 is therefore accepted as bounded trace/source coverage and semantic
protection, but rejected as an accuracy-closing DRAM/GCC model. The next
implementation slice is H3: a conditional SM86 fixed-latency register-read
and compiler-managed reuse-cache envelope. It must use SASS operand/reuse
metadata, maintain typed projected/gap/not-applicable frontiers, reserve
resources transactionally per SMSP, and leave cache/DRAM state untouched.


## 15. Slice S4 implementation evidence — H3 fixed register-read/RFC arm

H3 is now implemented at the SM86 issue/readiness seam and is active in the
default integrated full path only. Its assumptions are explicit and bounded:
two effective regular-register banks, one read port per bank, a three-cycle
window, and a deterministic three-entry-per-bank RFC queue. The queue key
retains block/warp slot, bank, register, and source operand position; reuse
hits consume entries under a one-shot effective assumption. Vector/multi-cell
shapes and typed projection gaps do not reserve ports or mutate the queue.

The model is transactionally planned per SMSP before issue-state mutation,
while cache/L1/L2/DRAM state and memory request ordering remain untouched.
Diagnostics separate the overlapping `applied_wait_cycles` readiness exposure
from residual same-cycle `port_conflict_cycles`; neither is an NCU stall count.

Evidence:

- Five predeclared real full roots completed 5/5.
- Corrected complete-corpus full replay completed 356 roots (286 `ok`, 62
  mixed-code controls, 8 replay/input failures), 12,392 successful launches,
  34.431 s, exactly 12 workers. Artifact:
  `../sm86_replay_results/full_corpus_h3_rf_20261015_v2/full.jsonl.zst`.
- Across successful launches: 27,167,662 projected reservations, 3,622,126
  RFC hits, 6,908,102 misses, and 133,572 typed unsupported-shape records.
- Report: `../sm86_replay_results/metric_report_h3_rf_20261015_v4/`.
  L2 hit/miss/source and DRAM-read mean APE remain 35.04%, 27.00%, 28.24%,
  and 40.91%. `sm__cycles_active.avg` rises 18.08%→18.26% mean APE and
  elapsed IPC rises 12.41%→12.53% on 107 exact identities.

H3 is therefore retained as a real conditional mechanism and causal coverage,
but not promoted as an accuracy improvement. The next implementation slice is
H4: separate persistent fetch-last-issued state from issue hold across Yield
instructions, evaluate it full-first with H3 present, and retain whichever
mechanism-local residual is supported by the complete corpus.

## 16. Slice S5 implementation evidence — H4 fetch-last-issued / Yield

H4 corrects the fetch/decode state seam without conflating it with issue
arbitration. Full replay now retains a persistent per-SMSP
`fetch_last_issued` warp; a Yield bit clears issue hold but fetch can continue
that warp while buffer capacity remains. Retirement clears both states.

Evidence:

- Five predeclared real roots completed full replay 5/5 and exposed nonzero
  continuation/separation observations.
- Complete full-first replay completed 356 roots (286 `ok`, 62 mixed-code
  controls, 8 replay/input failures), 12,392 launches, 34.826 s, exactly 12
  workers. Artifact:
  `../sm86_replay_results/full_corpus_h4_fetch_yield_20261015_v1/full.jsonl.zst`.
- Exact-identity cache/DRAM/IPC rows are unchanged from H3 (DRAM 40.91%,
  active-cycle 18.26%, elapsed IPC 12.53% mean APE). Report:
  `../sm86_replay_results/metric_report_h4_fetch_yield_20261015_v1/`.

H4 is accepted as front-end semantic coverage but not as an accuracy gain.
The next slice must be timing-active and independently evidenced: static
`LDG.CONSTANT`/GCC source classification or a separate IPC denominator/front-
end envelope. NCU values remain post-replay comparison targets and cannot
select the next parameter.

## 17. Slice S6 implementation evidence — H5 static LDG.CONSTANT source class

H5 adds a static source-qualified observer for `LDG.*.CONSTANT` global loads.
This is distinct from H2 CBANK field-15 offsets and from the existing
constant-return timing arm. It emits checked instruction/sector counts,
per-instruction 128-B line candidates, and CTA/warp-scoped unique candidates,
without changing L1/L2/DRAM state or equating source candidates with GCC truth.

Evidence:

- Five predeclared real full roots completed 5/5.
- Complete full-first replay completed 356 roots (286 `ok`, 62 mixed-code
  controls, 8 replay/input failures), 12,392 launches, 34.915 s, exactly 12
  workers. Artifact:
  `../sm86_replay_results/full_corpus_h5_ldg_constant_20261015_v1/full.jsonl.zst`.
- Successful rows emitted 6,807,596 static source-class instructions,
  10,817,035 sectors, 7,142,629 request candidates, and 1,039,257 unique
  candidates.
- Report: `../sm86_replay_results/metric_report_h5_ldg_constant_20261015_v1/`;
  exact DRAM and IPC mean APE remain 40.91%, 18.26%, and 12.53%.

H5 is accepted as source attribution but rejected as an accuracy gain. The
next implementation slice must add an independently evidenced GCC/L1.5
response-scope discriminator or a bounded DRAM service relation, with NCU
values remaining post-replay targets rather than tuning inputs.

## 18. Slice S7 implementation evidence — H6 GCC/L1.5 scope candidate

H6 adds a CTA-scoped line-domain candidate to the static `LDG.*.CONSTANT`
observer while retaining the warp-qualified candidate. The two domains expose
replication scope without equating source observations with GCC requests,
L2 sectors, or DRAM service. No timing/cache state changed.

Evidence:

- Four predeclared real roots completed full replay 4/4.
- Complete full-first replay completed 356 roots (286 `ok`, 62 mixed-code
  controls, 8 replay/input failures), 12,392 launches, 35.194 s, exactly 12
  workers. Artifact:
  `../sm86_replay_results/full_corpus_h6_gcc_scope_20261015_v1/full.jsonl.zst`.
- Successful rows emitted 1,039,257 warp candidates and 982,844 CTA
  candidates. Report:
  `../sm86_replay_results/metric_report_h6_gcc_scope_20261015_v1/`.
- DRAM/IPC exact-identity mean APE remains 40.91%, 18.26%, and 12.53%.

No universal GCC scope rule is promoted. H6 is source-domain evidence only;
the next implementation slice must be a bounded service or IPC denominator
model with independent mechanism evidence.

## 19. Negative H7 IPC denominator probe — 2026-10-15

A four-root real full probe tested the mechanism-domain hypothesis that SM86
elapsed IPC should divide by the explicit SM-active observation frontier rather
than the GPC launch envelope. The substitute raised mean APE to 51.81% versus
12.53% for the retained GPC endpoint; it was reverted before complete-corpus
promotion. This is a falsification result, not an NCU fitting step. The next
arm must preserve the two endpoint domains and target a different service or
front-end relation.

## 20. Slice S8 implementation evidence — H8 DRAM partition pressure

H8 adds source-derived observer-only pressure arrays for documented GA102
controller/slice counts: line-ID modulo-12, XOR-fold modulo-12, and modulo-6
candidate domains. The aggregate G/D/1 byte server remains unchanged. The
NVIDIA GA102 whitepaper is used only for controller/slice count evidence;
physical swizzling remains typed unknown, and public generic decoders are not
imported.

Evidence:

- Five predeclared real roots completed full replay 5/5.
- Complete full-first replay completed 356 roots (286 `ok`, 62 mixed-code
  controls, 8 replay/input failures), 12,392 launches, 35.409 s, 12 workers.
  Artifact: `../sm86_replay_results/full_corpus_h8_dram_partition_20261015_v1/full.jsonl.zst`.
- Exact report:
  `../sm86_replay_results/metric_report_h8_dram_partition_20261015_v1/`;
  DRAM/L1/L2/IPC residuals are unchanged.
- Independent real-trace research finds irregular candidate concentration but
  weak/unstable correlation with NCU FBPA max/min ratios; no hash is selected.

H8 is accepted as the current DRAM topology observability envelope, not a
service timing model. Continue to the next plan slice only when a mechanism
has independent service evidence; do not turn NCU residuals into a partition
selector.

## 21. Slice S9 implementation evidence — H9 near/far pressure uncertainty

H9 adds a typed near/far pressure envelope over H8 modulo/XOR candidates. The
six-versus-six half-controller split is a candidate domain only; no physical
proximity, hash, latency, or service rate is inferred. GA102 whitepaper counts
and Luo et al. partitioned-L2 evidence are kept separate from SM86-specific
truth. Independent unlocked-clock 3090 Ti memtest repeats support only a future
[810,850] GB/s scenario interval.

Evidence:

- Five predeclared real roots completed full replay 5/5; near/far candidate
  intervals are emitted as source-domain sectors.
- Complete full-first replay completed 356 roots (286 `ok`, 62 mixed-code
  controls, 8 replay/input failures), 12,392 launches, 35.059 s, 12 workers.
  Artifact:
  `../sm86_replay_results/full_corpus_h9_near_far_20261015_v1/full.jsonl.zst`.
- Report:
  `../sm86_replay_results/metric_report_h9_near_far_20261015_v1/`; exact
  residuals remain unchanged from H8.

H9 is accepted as a map-uncertainty envelope. Timing-active partition/row
service requires a locked local 3090 Ti probe and remains the next research
condition, not an invented topology constant.

## 22. Slice S10 implementation evidence — H10 effective DRAM service rate

H10 adds an explicit `--dram-service-rate-bytes-per-sec` path and promotes
825,000,000,000 B/s as the SM86 full carded default. The value comes from
independent repeated RTX 3090 Ti memtest evidence under an unlocked protocol
(~810–850 GB/s forced-DRAM range). It is an absolute effective-rate hypothesis
mapped onto the simulator's locked-card SM clock for cycle pricing, not a
hardware clock crosswalk; the architecture-card vendor peak 1008 GB/s remains
reference context. The scenario is recorded, not silently treated as physical
truth; partition/row mapping stays unknown.

Evidence:

- Four real representative roots completed full replay at 825 GB/s.
- Complete full-first replay completed 356 roots (286 `ok`, 62 mixed-code
  controls, 8 replay/input failures), 12,392 launches, 35.094 s, 12 workers.
  Artifact:
  `../sm86_replay_results/full_corpus_h10_dram_rate_825_20261015_v1/full.jsonl.zst`.
- Report:
  `../sm86_replay_results/metric_report_h10_dram_rate_825_20261015_v2/`.
  Exact GPC/SM timing mean APE improves 13.14→13.05%, 18.26→18.19%, and
  18.11→18.00%; DRAM traffic remains 40.91%.

H10 is promoted as an effective service-rate scenario with explicit uncertainty,
not a physical DRAM claim. The next slice should target remaining DRAM demand /
writeback or IPC residuals without hiding service-rate uncertainty.

## 23. Slice S11 implementation evidence — H11 source-composed DRAM demand

envelope

H11 emits a typed source-composed demand envelope whose lower edge is modeled
L2-missing data plus compulsory instruction sectors and whose candidate high
edge adds static LDG.CONSTANT and H2 CBANK source-domain candidates. It is
explicitly not achieved DRAM traffic, a mathematical upper bound, or GCC truth;
H10's independently measured 825 GB/s scenario remains separate.

Evidence:

- Representative real full replay placed three of four independent NCU DRAM
  observations inside the source envelope.
- Complete full-first replay completed 356 roots (286 `ok`, 62 mixed-code
  controls, 8 replay/input failures), 12,392 launches, 35.846 s, 12 workers.
  Artifact:
  `../sm86_replay_results/full_corpus_h11_dram_demand_20261015_v1/full.jsonl.zst`.
- Report:
  `../sm86_replay_results/metric_report_h11_dram_demand_20261015_v1/`; DRAM
  mean APE remains 40.91%. Independent residual audit finds no useful global
  correlation with static constant/GCC candidates.

H11 improves demand observability but does not close DRAM accuracy. Continue to
conditional writeback/transport or IPC mechanism slices; do not fit a global
traffic multiplier.

## 24. Slice S12 implementation evidence — H12 atomic conflict exposure

H12 adds exact traced same-sector atomic alias exposure while retaining the
existing transaction vector. It emits conflict-bearing/distinct instruction
counts and alias sectors only; no per-alias latency, DRAM multiplier, or
serialization is invented. Independent kernel-level conflict anchors are not a
safe per-transaction timing source.

Evidence:

- Four predeclared real atomic roots completed full replay 4/4.
- Complete full-first replay completed 356 roots (286 `ok`, 62 mixed-code
  controls, 8 replay/input failures), 12,392 launches, 35.578 s, 12 workers.
  Artifact:
  `../sm86_replay_results/full_corpus_h12_atomic_conflict_20261015_v1/full.jsonl.zst`.
- Successful rows observed 58 conflict-bearing atomic instructions, 622
  distinct instructions, and 198 alias sectors. Report:
  `../sm86_replay_results/metric_report_h12_atomic_conflict_20261015_v1/`;
  exact DRAM/L1/L2/IPC residuals are unchanged.

H12 is accepted as conflict observability, not timing closure. A timing-active
atomic arm requires a locked local microbenchmark and independent holdout.

## 25. Negative H13 atomic service promotion — 2026-10-15

Independent local warm-L2 atomic microbenchmarks show same-word conflict
overhead around 10–18% versus distinct addresses. The result is retained as a
mechanism prior only: it does not identify a per-transaction DRAM/queue
constant, and the current request seam does not preserve lane multiplicity
through the shared system. H12 conflict observability remains active; H13 does
not add timing. A future locked probe must establish lossless conflict identity
before promotion.

## 26. Slice S13 implementation evidence — H14 local-memory width observer

H14 retains the producer's instruction data-width field through `sim-trace` and
emits local-memory width records, unknown-width dispositions, summed bytes, and
predicate-lane×width 32-B candidates. The validated local sector stream,
cache state, DRAM service, and timing remain unchanged.

Evidence:

- Five local/vector real roots completed full replay 5/5.
- Complete full-first replay completed 356 roots (286 `ok`, 62 mixed-code
  controls, 8 replay/input failures), 12,392 launches, 35.671 s, 12 workers.
  Artifact:
  `../sm86_replay_results/full_corpus_h14_local_width_20261015_v1/full.jsonl.zst`.
- 4,892 width records, zero unknown widths, 19,824 source bytes/candidates.
  Report:
  `../sm86_replay_results/metric_report_h14_local_width_20261015_v1/`;
  exact residuals remain unchanged.

H14 closes a trace-source gap but does not improve corpus accuracy. Continue to
the next L1/IPC mechanism rather than inventing local traffic multipliers.

## 27. Slice S14 implementation evidence — H15 RF Control/Allocate decoupling

H15 corrects H3's scheduler coupling. RF/RFC reservations remain transactional
and capacity-bounded, but RF port conflict no longer delays CGGTY issue
eligibility; the reservation is a Control/Allocate downstream diagnostic,
consistent with Huerta MICRO'25 Ampere semantics. Cache/DRAM state is unchanged.

Evidence:

- Four high-RF real roots completed full replay 4/4.
- Complete full-first replay completed 356 roots (286 `ok`, 62 mixed-code
  controls, 8 replay/input failures), 12,392 launches, 35.109 s, 12 workers.
  Artifact:
  `../sm86_replay_results/full_corpus_h15_rf_allocate_20261015_v1/full.jsonl.zst`.
- Report:
  `../sm86_replay_results/metric_report_h15_rf_allocate_20261015_v1/`;
  mean APE improves GPC 13.05→12.91%, SM active 18.19→18.01%, SMSP
  17.69→17.50%, elapsed IPC 12.42→12.31%, while DRAM remains 40.91%.

H15 is promoted as a bounded effective IPC improvement; remaining residuals
require subsequent mechanism slices, not proprietary RF claims.

## 28. Slice S15 implementation evidence — H16 L1 tag-lookup observer

H16 fixes a primary metric-boundary defect: L1 lookup rows now use resident
hits plus merge observations minus one fresh-line tag miss per line allocation,
clamped to the source lookup domain. Cache state, replacement, MSHRs, response
timing, L2, DRAM, and IPC are unchanged.

Evidence:

- Four predeclared real roots completed full and fast replay 4/4 each.
- Complete full-first replay completed 356 roots (286 `ok`, 62 mixed-code
  controls, 8 replay/input failures), 12,392 launches, 36.054 s, 12 workers.
  Artifact: `../sm86_replay_results/full_corpus_h16_l1_lookup_20261015_v1/full.jsonl.zst`.
- Exact report: `../sm86_replay_results/metric_report_h16_l1_lookup_20261015_v1/`.
  L1 global-load lookup-hit APE improved 33.43%→30.71%; lookup-miss APE
  improved 206.41%→58.44%. L2/DRAM/IPC remained 28.24%/40.91%/12.31%.

H16 is promoted as a bounded observer correction, not physical L1 closure. The
next L1 implementation is a card-backed 256-set × 4-way effective state model;
undocumented hash behavior must remain typed unknown/sensitivity, and evaluation
must remain full-first on real working-set, thrash, vector-copy, and SGEMM roots.

## 29. Slice S16 implementation evidence — H17 card-backed L1 set/way state

H17 adds optional set-local LRU state to `sim-mem::L1Cache`. SM86 full mode now
uses the architecture card's 256 sets × 4 ways with a transparent modulo index;
the p-chase active-capacity point remains the global capacity input, and the
undocumented GA102 address hash is explicitly unknown. Fully associative state
remains available as a sensitivity arm. Pending fills, generation guards, MSHR
capacity, dirty protection, and observer semantics remain transactional.

Evidence:

- Five predeclared real roots completed full and fast replay 5/5 each.
- Complete full-first replay completed 356 roots (286 `ok`, 62 mixed-code
  controls, 8 replay/input failures), 12,392 launches, 36.517 s, 12 workers.
  Artifact: `../sm86_replay_results/full_corpus_h17_l1_setway_20261015_v1/full.jsonl.zst`.
- Report: `../sm86_replay_results/metric_report_h17_l1_setway_20261015_v1/`.
  Versus H16, L1 hit/miss APE improved 30.71%→30.39% and 58.44%→57.74%;
  GPC/SM/SMSP/elapsed-IPC improved 12.91/18.01/17.50/12.31% to
  12.75/17.86/17.36/12.12%. L2 source APE regressed 28.24%→32.43%; DRAM
  remained 40.91%.

H17 is promoted as an effective set/way scenario, not physical hash closure.
The next slice must isolate the L2 regression and implement the independent
source-qualified device-scope store/fence completion tail without NCU fitting.

## 30. Slice S17 implementation evidence — H18 L2/hash uncertainty

H18 adversarially isolates H17's L2 regression. Fixed modulo indexing aliases
high-stride real streaming lines, increasing L2 lookup sectors 32,896→82,048
without changing DRAM read sectors (32,768). The default SM86 L1 scenario now
uses a pressure-bounded unknown-hash arm: modulo is retained until a set is full,
then an XOR-fold candidate is used only when it has strictly more room. This is a
source-topology discriminator, not NCU fitting or physical hash recovery.

Evidence:

- Five real representative roots completed full replay 5/5; streaming L2
  sectors returned to 32,896.
- Complete full-first replay completed 356 roots (286 `ok`, 62 mixed-code
  controls, 8 replay/input failures), 12,392 launches, 36.751 s, 12 workers.
  Artifact: `../sm86_replay_results/full_corpus_h18_l1_hashfallback_20261015_v2/full.jsonl.zst`.
- Report: `../sm86_replay_results/metric_report_h18_l1_hashfallback_20261015_v2/`.
  H17→H18 exact mean APE: L2 32.43%→28.24%, DRAM unchanged at 40.91%; L1
  hit/miss 30.39/57.74%→30.71/58.44%; elapsed IPC 12.12%→12.31%.
- The final fallback code separately passed the working-set, L2-thrash,
  vector-copy, and both vendor-SGEMM roots 5/5:
  `../sm86_replay_results/h18_l1_hashfallback_20261015_v2_representative/full.jsonl.zst`.

H18 is promoted as the default effective unknown-hash scenario; fixed modulo and
fixed XOR remain sensitivity arms. The next implementation slice is the
source-qualified device-scope store/fence completion tail for IPC/cycles.

## 31. Slice S18 implementation evidence — H19 source-qualified store/fence tail

H19 promotes the existing transactional `MemoryAdmissionFeedback` and write-ack
seam into default SM86 full replay. The static recognizer covers device-scope
`MEMBAR.GL/GPU/SC.GPU`; feedback is gated on global STORE/RED work plus a device
fence or ERRBAR/CCTL marker. The candidate uses independently sourced Ampere
queue/service priors and a bounded region completion endpoint. It does not claim
physical writeback/coherence completion and does not consume NCU values.

Evidence:

- Five real store/fence/memory roots completed full replay 5/5.
- Complete default full-first replay completed 356 roots (286 `ok`, 62 mixed
  code controls, 8 replay/input failures), 12,392 launches, 36.577 s, 12
  workers. Artifact:
  `../sm86_replay_results/full_corpus_h19_store_fence_default_20261015_v2/full.jsonl.zst`.
- Default summary records the candidate as defaulted and NCU as unused. The
  replay observed 1,069 qualified launches, 103,737 acknowledgements, 378,742
  write-completion requests, and 104,805 busy cycles.
- Report:
  `../sm86_replay_results/metric_report_h19_store_fence_default_20261015_v2/`;
  exact L1/L2/DRAM/elapsed-IPC residuals are unchanged from H18.

H19 closes the source-qualified completion plumbing but is a neutral accuracy
slice because the new frontier is not usually critical on scored launches. The
next slice must add a mechanism-local critical-path discriminator or another
independent IPC source; no global NCU-fitted delay is allowed.

## 32. Slice S19 implementation evidence — H20 dependent-SFU frontier

H20 adds a source-qualified 30-cycle dependent-SFU result frontier for static
same-register `MUFU.*`/special-function forms on SM86. It preserves the
four-cycle SFU initiation cadence and leaves all other forms unchanged. The
coefficient comes from an independent clock64 dependent-chain probe, not NCU.

Evidence:

- Five real SFU-heavy roots completed full and fast replay 5/5.
- Complete default full-first replay completed 356 roots (286 `ok`, 62 mixed
  code controls, 8 replay/input failures), 12,392 launches, 37.530 s, 12
  workers. Artifact:
  `../sm86_replay_results/full_corpus_h20_sfu_dependent_20261015_v1/full.jsonl.zst`.
- The corpus observed 269 launches with nonzero SFU exposure, 970,993 limiting
  cycles, and 1,345,369 exposure cycles. Report:
  `../sm86_replay_results/metric_report_h20_sfu_dependent_20261015_v1/`.
  L1/L2/DRAM/IPC residuals remained neutral at 30.71/58.44/28.24/40.91%
  and 12.31% elapsed IPC.

H20 is retained as a bounded effective mechanism but does not close IPC. Continue
to a stronger source-qualified critical-path model; do not broaden SFU timing or
fit NCU residuals.

## 33. Slice S20 implementation evidence — H21 causal IPC frontier decomposition

H21 adds native source-derived issue/terminal causal frontier and interval-width
rows. They expose the exact simulator uncertainty boundary without changing
scheduler, cache, DRAM, or NCU endpoint selection.

Evidence:

- Five real critical-path roots completed full replay 5/5.
- Complete default full-first replay completed 356 roots (286 `ok`, 62 mixed
  code controls, 8 replay/input failures), 12,392 launches, 38.560 s, 12
  workers. Artifact:
  `../sm86_replay_results/full_corpus_h21_critical_frontier_20261015_v1/full.jsonl.zst`.
- Report:
  `../sm86_replay_results/metric_report_h21_critical_frontier_20261015_v1/`;
  exact L1/L2/DRAM/SM/elapsed-IPC residuals are unchanged from H20.

H21 is a neutral observability slice. The next slice must promote a
source-qualified critical-path timing arm or retain a negative result with a
falsifiable residual; no NCU fitting is permitted.

## 34. Slice S21 implementation evidence — H22 L0C miss-switch scheduling

H22 activates the public Ampere/SM86 four-cycle immediate-constant-cache
miss-switch window. The per-SMSP countdown is independent of the IMC line
response frontier: ordinary control, dependency, front-end, resource, and
memory-admission readiness remain required before the held greedy warp is
forced. Every effective issue reloads the window; each parked greedy miss
consumes one opportunity. The default SM86 full replay path uses this arm.

Evidence:

- Representative real full-first replay passed all four discovered roots (485
  full launches): `../sm86_replay_results/h22_l0c_switch_20261015_v2/full.jsonl.zst`.
- Complete full-first replay then complementary fast replay passed 356 roots
  (286 `ok`, 62 mixed-code controls, 8 replay/input failures), 12,392 full
  launches in 37.195 s with exactly 12 workers. Artifacts:
  `../sm86_replay_results/full_corpus_h22_l0c_switch_20261015_v2/full.jsonl.zst`
  and `fast.jsonl.zst`.
- Post-replay report:
  `../sm86_replay_results/metric_report_h22_l0c_switch_20261015_v2/`.
- IMC tag misses remain 1,174,632 and L1/L2/DRAM counters are unchanged from
  H20. Miss-wait events are 3,125,368; the new bounded greedy-switch-block
  diagnostic is 1,489,412. The readiness schema is 27.
- Mean APE is neutral versus H20: L1 lookup hit/miss 30.71%/58.44%, L2
  28.24%, DRAM 40.91%, SM active 18.01%, elapsed IPC 12.31%, and GPC 12.91%.
  Only six terminal/GPC frontier rows move by one or two cycles (aggregate
  −6 cycles), so this is a live bounded mechanism but not IPC closure.

The public 46-cycle ERRBAR/post-EXIT special remains deferred because no
dynamic `ERRBAR` executes in the complete corpus. A known scheduler residual
also remains: once the L0C countdown expires, a non-greedy candidate that misses
in IMC is parked without scanning the next ready sibling. The next slice should
implement that candidate scan transactionally, with explicit overlap diagnostics
and the same full-first/no-NCU evaluation contract.

## 35. Slice S22 implementation evidence — H23 non-greedy L0C candidate scan

H23 completes the source-backed Huerta issue loop after H22. An expired held
greedy warp is probed first; ordinary-ready non-greedy siblings are then
probed in deterministic youngest order after each L0C miss. Every miss stages a
line request in the cloned cache and parks its wait frontier; only a final hit
is an effective issue pick. Memory-admission feedback and instruction counts
are charged only for that final pick. This is the default SM86 full-replay
composition and does not consume NCU values.

Evidence:

- Post-ordering representative full-first replay passed 10/10 real roots (2,728
  full launches):
  `../sm86_replay_results/h23_l0c_scan_20261015_v2/full.jsonl.zst`.
- Complete full-first replay followed by complementary fast replay passed 356
  roots (286 `ok`, 62 mixed-code controls, 8 replay/input failures), 12,392
  full launches in 40.715 s with exactly 12 workers:
  `../sm86_replay_results/full_corpus_h23_l0c_scan_20261015_v2/full.jsonl.zst`
  and adjacent `fast.jsonl.zst`.
- Report:
  `../sm86_replay_results/metric_report_h23_l0c_scan_20261015_v2/`.
- Full scan diagnostics: 21,574,287 attempts, 21,348,688 misses, 225,599
  successes, and 111,823,998 overlapping IMC wait events. Rows with scan
  activity use readiness schema 28; untouched IMC rows remain schema 27.
- Exact H22→H23 traffic deltas: L1 hits −104, misses +104, merges +234;
  L2 and DRAM counters unchanged. Only 74 common launch frontiers move,
  aggregate GPC/terminal +392 cycles (range −147…+505).
- Post-replay APE is neutral at reported precision: L1 lookup hit/miss
  30.71%/58.44%, L2 28.24%, DRAM 40.91%, SM active 18.01%, elapsed IPC
  12.31%, and GPC 12.91%.

H23 is source-aligned scheduler plumbing rather than IPC closure. The next
slice must target typed terminal/writeback/coherence or issue-resource residuals;
probe counts must remain diagnostics and never become a fitted delay. The public
46-cycle ERRBAR/post-EXIT special remains deferred because no dynamic ERRBAR
executes in the corpus.

## 36. Slice S23 implementation evidence — H24 consecutive-BAR generation re-arm

H24 binds the SM86 CTA-barrier generation re-arm to 28 cycles, scoped only to a
later `BAR.SYNC` generation of the same CTA. A fresh local RTX 3090 Ti clock64
probe from the retained barrier-pipeline source measured 688,134 cycles for
16,384 one-barrier generations and 1,146,885 for two consecutive barriers,
establishing the 28-cycle differential without using NCU as a replay input.
The two-cycle shared barrier-arrival server remains unchanged.

Evidence:

- Five ordinary barrier/control and five high-BAR roots passed representative
  full-first replay.
- Complete full-first replay then complementary fast replay passed 356 roots
  (286 `ok`, 62 mixed-code controls, 8 replay/input failures), 12,392 full
  launches in 40.694 s with exactly 12 workers:
  `../sm86_replay_results/full_corpus_h24_barrier_rearm_20261015_v1/full.jsonl.zst`
  and adjacent `fast.jsonl.zst`.
- Report:
  `../sm86_replay_results/metric_report_h24_barrier_rearm_20261015_v1/`.
- Only 29 launches move versus H23; GPC/terminal frontier sum increases by
  1,847 cycles (maximum per-launch increase 482). L1/L2/DRAM counters remain
  unchanged.
- Mean APE moves slightly: GPC 12.90596→12.90078%, SM active
  18.00767→18.02024%, elapsed IPC 12.31350→12.30731%; cache/DRAM APE remains
  30.71/58.44/28.24/40.91%.

H24 is a promoted conditional barrier mechanism, not accuracy closure. Keep it
  restricted to consecutive CTA generations. The next slice should pursue the
  dominant eligible-not-selected/front-end/L1TEX dependency residuals or land a
  typed fixed-result-queue observer; do not invent execution-to-writeback queue
release timing from the Huerta 8-entry configuration alone.

## 37. Slice S24 implementation evidence — H25 dynamic-youngest priority audit

H25 retains block-local dynamic warp identity in `WarpReplay`/`WarpState` and
uses it as the deterministic tie-break for the existing arrival-order youngest
selection. A within-CTA inversion counter audits the Huerta
greedy-then-highest-dynamic-ID policy without changing timing. All 12,392 full
launches report zero inversions.

The independent RTX 3090 Ti fixed-result-capacity clock64 probe (8,192
generations, 11 trials) reproduces a multiwarp W8 threshold (+0.163 then
+1.118 cycles/generation), but W1/W4 are null and an HADD2 collision/separated
control is non-monotonic. Physical queue capacity and execution-to-writeback
release remain unidentified; no timing arm is enabled.

Evidence:

- Five high-BAR roots passed representative full-first replay.
- Complete full-first replay then complementary fast replay passed 356 roots
  (286 `ok`, 62 mixed-code controls, 8 replay/input failures), 12,392 full
  launches in 40.456 s with exactly 12 workers:
  `../sm86_replay_results/full_corpus_h25_priority_audit_20261015_v1/full.jsonl.zst`
  and adjacent `fast.jsonl.zst`.
- Report:
  `../sm86_replay_results/metric_report_h25_priority_audit_20261015_v1/`.
- Readiness schema is 29. H25 is bit-for-bit neutral versus H24 for all
  L1/L2/DRAM/IPC/frontier outputs.

Promote the explicit priority audit and retain the queue probe as negative
mechanism evidence. The next slice must attack eligible-not-selected,
front-end, or L1TEX dependency residuals with an independent prior; do not
invent fixed-result execution-to-writeback release timing.

## 38. Slice S25 implementation evidence — H26 branch-stream redirect (rejected)

H26 tested a source-motivated front-end hypothesis: flush per-SMSP
fall-through stream-buffer lines on dynamic non-fall-through branches before
fetching the traced target. Five branch-heavy real roots passed representative
replay; complete full-first A/B replay then passed 356 roots (286 `ok`, 62
mixed-code controls, 8 replay/input failures), 12,392 launches with exactly 12
workers:
`../sm86_replay_results/full_corpus_h26_branch_redirect_20261015_v1/full.jsonl.zst`.

The arm is rejected: mean GPC APE worsens 12.90%→14.37%, SM active worsens
18.02%→22.03%, elapsed IPC worsens 12.31%→12.73%, and L1/L2/DRAM do not
improve. The timing change was removed from the active path; negative report:
`docs/sm86_branch_redirect_slice_20261015.md`.

Future front-end work must distinguish static branch-target prediction/path
state instead of globally flushing on every dynamic transfer. This falsifier
does not block subsequent issue/front-end research and preserves the H25/H24
default timing baseline.

## 39. Slice S26 implementation evidence — H27 L1TEX/front-end redirect observer

H27 retains the rejected H26 branch-flush timing experiment and adds only a
native observer for dynamic non-fall-through control transfers. It does not
mutate instruction streams or timing; readiness schema 30 carries the count.

Evidence:

- Five branch-heavy roots passed representative full-first replay.
- Complete full-first replay then complementary fast replay passed 356 roots
  (286 `ok`, 62 mixed-code controls, 8 replay/input failures), 12,392 full
  launches in 40.452 s with exactly 12 workers:
  `../sm86_replay_results/full_corpus_h27_l1tex_observer_20261015_v1/full.jsonl.zst`
  and adjacent `fast.jsonl.zst`.
- Report:
  `../sm86_replay_results/metric_report_h27_l1tex_observer_20261015_v1/`.
- The corpus observed 1,882,174 non-fall-through transfers and 85,461,673
  front-end fetches. L1/L2/DRAM/IPC/frontier outputs are bit-for-bit neutral
  versus H25.

Use this observer to partition future front-end/L1TEX models by static target
shape; never convert transfer counts into a universal delay.

## 40. Slice S27 implementation evidence — H28 constant-pressure L1TEX observer

H28 makes the existing static SASS constant-pressure classifier dynamically
observable in memory-issue readiness. It counts streams beyond the released
five-entry dispatch window (including the narrow U16/write-barrier shape) while
retaining the established 11+39 effective frontier and no new timing/cache
mutation.

Evidence:

- One direct streaming root and four high-IMC/pressure roots passed
  representative full-first replay.
- Complete full-first replay then complementary fast replay passed 356 roots
  (286 `ok`, 62 mixed-code controls, 8 replay/input failures), 12,392 full
  launches in 40.672 s with exactly 12 workers:
  `../sm86_replay_results/full_corpus_h28_constant_pressure_20261015_v1/full.jsonl.zst`
  and adjacent `fast.jsonl.zst`.
- Report:
  `../sm86_replay_results/metric_report_h28_constant_pressure_20261015_v1/`.
- The corpus observed 6,336,451 constant-pressure instructions and 1,882,174
  non-fall-through transfers. Readiness schema is 31. L1/L2/DRAM/IPC/frontier
  outputs are bit-for-bit neutral versus H27.

Keep the pressure count as a residual partition. A timing arm requires fresh
clock64 load-return/queue evidence rather than a static threshold assumption.

## 41. Slice S28 implementation evidence — H29 constant-pressure clock64 differential (negative)

Fresh no-NCU RTX 3090 Ti clock64 runs of the retained constant-load probe used
dependent chains versus four independent reads at 128 iterations and 1/4/16
blocks; every functional oracle passed. Dependent runs were approximately 9.7k
cycles and independent pressure runs 11.6–12.9k cycles. The differential mixes
issue throughput, overlap, and operand return, so it cannot identify a new
L1TEX delay or justify a generic multiplier.

No timing arm was promoted; H28's source-qualified pressure observer remains
the active default. Compact report:
`docs/sm86_constant_dependency_probe_slice_20261015.md`.

## 42. Slice S29 implementation evidence — H30 six-slice L2 routing observer

H30 adds transactionally preserved per-slice L2 request-sector counts using the
active six-slice effective geometry/hash. It does not claim a proprietary GA102
hash, add queue timing, or use NCU routing labels.

Evidence:

- Five high-stride/L2-thrash/working-set/SGEMM roots passed representative
  full-first replay.
- Complete full-first replay then complementary fast replay passed 356 roots
  (286 `ok`, 62 mixed-code controls, 8 replay/input failures), 12,392 full
  launches in 40.089 s with exactly 12 workers:
  `../sm86_replay_results/full_corpus_h30_l2_slice_observer_20261015_v1/full.jsonl.zst`
  and adjacent `fast.jsonl.zst`.
- Report:
  `../sm86_replay_results/metric_report_h30_l2_slice_observer_20261015_v1/`.
- Slice sectors: 1,718,579 / 1,653,401 / 1,717,493 / 1,652,955 / 1,717,655 /
  1,653,918. L1/L2/DRAM/IPC/frontier outputs are bit-for-bit neutral versus
  H29.

The observer is promoted as a routing/queue research surface only. The next L2
timing arm requires independent slice-service evidence, not a hidden hash or
NCU fit.

## 43. Slice S30 implementation evidence — H31 L2 slice-pressure observer

H31 extends H30 with per-launch effective L2 max-slice, min-slice, spread, and
max/min imbalance observers. The rebuilt local RTX 3090 Ti clock64 L2 queue
probe passed unique/reuse smoke points across 1/2/4/8 warps and 1/8/32-load
bursts; pressure growth is stable as an observer but does not identify a fixed
service latency.

Evidence:

- Five high-stride/L2 roots passed representative full-first replay.
- Complete full-first replay then complementary fast replay passed 356 roots
  (286 `ok`, 62 mixed-code controls, 8 replay/input failures), 12,392 full
  launches in 39.887 s with exactly 12 workers:
  `../sm86_replay_results/full_corpus_h31_l2_pressure_observer_20261015_v1/full.jsonl.zst`
  and adjacent `fast.jsonl.zst`.
- Report:
  `../sm86_replay_results/metric_report_h31_l2_pressure_observer_20261015_v1/`.
- Per-launch imbalance ratio: minimum 1.0, median 1.133, p90 3.0, maximum
  38.0. L1/L2/DRAM/IPC/frontier outputs are bit-for-bit neutral versus H30.

No queue timing arm is promoted; further L2 timing requires independent service
evidence beyond pressure observation.

## 44. Slice S31 implementation evidence — H32 L2 pressure strata

H32 extends H31 with active-slice count and maximum-slice demand fraction,
alongside max/min/spread/imbalance observers. It does not recover hidden GA102
hash state or introduce queue timing.

Evidence:

- Complete full-first replay then complementary fast replay passed 356 roots
  (286 `ok`, 62 mixed-code controls, 8 replay/input failures), 12,392 full
  launches in 40.152 s with exactly 12 workers:
  `../sm86_replay_results/full_corpus_h32_l2_pressure_strata_20261015_v1/full.jsonl.zst`
  and adjacent `fast.jsonl.zst`.
- Report:
  `../sm86_replay_results/metric_report_h32_l2_pressure_strata_20261015_v1/`.
- Imbalance ratio: min 1.0, median 1.133, p90 3.0, max 38.0. Max-slice
  fraction: min 16.67%, median 18.75%, p90 50%, max 100%. Active slices:
  6/2/3/4/1/5 for 9,186/1,658/488/487/358/215 launches.
- L1/L2/DRAM/IPC/frontiers are bit-for-bit neutral versus H31.

The observer is promoted as route-local pressure stratification only. Timing
remains unchanged pending independent L2 service evidence.

## 45. Slice S32 implementation evidence — H33 L2 route concentration observer

H33 adds normalized sum-of-squares concentration over effective six-slice
request demand. It distinguishes balanced traffic from single-slice
concentration without recovering a proprietary hash or queue timing.

Evidence:

- Five high-stride/L2 roots passed representative full-first replay.
- Complete full-first replay then complementary fast replay passed 356 roots
  (286 `ok`, 62 mixed-code controls, 8 replay/input failures), 12,392 full
  launches in 39.868 s with exactly 12 workers:
  `../sm86_replay_results/full_corpus_h33_l2_concentration_20261015_v1/full.jsonl.zst`
  and adjacent `fast.jsonl.zst`.
- Report:
  `../sm86_replay_results/metric_report_h33_l2_concentration_20261015_v1/`.
- Concentration: min 166,667 ppm, median 168,405 ppm, p90 500,000 ppm,
  max 1,000,000 ppm; imbalance remains min 1.0, median 1.133, p90 3.0,
  max 38.0. L1/L2/DRAM/IPC/frontiers are bit-for-bit neutral versus H32.

This observer is promoted as route-local pressure evidence only. No stable
service differential has been identified, so timing remains unchanged.

## 46. Slice S33 implementation evidence — H34 L2 route strata

H34 classifies effective L2 routes as incomplete (<6 active slices), balanced
(all six and concentration below 250,000 ppm), or concentrated (all six and at
least 250,000 ppm). The strata are descriptive observers only.

Evidence:

- Five high-stride/L2 roots passed representative full-first replay.
- Complete full-first replay then complementary fast replay passed 356 roots
  (286 `ok`, 62 mixed-code controls, 8 replay/input failures), 12,392 full
  launches in 40.574 s with exactly 12 workers:
  `../sm86_replay_results/full_corpus_h34_route_strata_20261015_v1/full.jsonl.zst`
  and adjacent `fast.jsonl.zst`.
- Report:
  `../sm86_replay_results/metric_report_h34_route_strata_20261015_v1/`.
- Counts: 3,206 incomplete, 9,135 balanced, 51 concentrated. L1/L2/DRAM/
  IPC/frontier outputs are bit-for-bit neutral versus H33.

No stable route-local service differential is identified; timing remains
unchanged pending stronger independent evidence.

## 47. Slice S34 implementation evidence — H35 route-strata correlation

H35 joined H34 incomplete/balanced/concentrated route strata to independent
post-replay NCU comparison rows without using NCU as a model input. Concentrated
rows show higher cycle residuals but only 4–8 joins per metric; incomplete rows
show high DRAM residuals on a small subset while L2-sector residuals are near
zero. The correlation is sparse/confounded and does not identify service timing.

Evidence:

- Complete full-first replay then complementary fast replay passed 356 roots
  (286 `ok`, 62 mixed-code controls, 8 replay/input failures), 12,392 full
  launches in 40.529 s with exactly 12 workers:
  `../sm86_replay_results/full_corpus_h35_route_correlation_20261015_v1/full.jsonl.zst`
  and adjacent `fast.jsonl.zst`.
- Report:
  `../sm86_replay_results/metric_report_h35_route_correlation_20261015_v1/`.
- H35 preserves the typed route-strata ledger and leaves L2 timing unchanged.

No timing arm is justified; route correlation remains a residual-analysis
surface and the next SOTA cache mechanism must be independently sourced.

## 48. Slice S35 implementation evidence — H36 sectored L2 fill ownership

H36 exposes the existing event-mode pending-read coalescing decision as a
checked, cumulative observer. For every read sector, the model records whether
the sector owns a new fill capability or merges onto the matching pending
allocation/sector. A merge remains a lookup miss and reuses the older ready
frontier; these counters never change cache state, service time, DRAM demand,
or IPC. Full replay emits the native rows
`l2_pending_merge_sectors`, `l2_new_fill_sectors`, and
`l2_pending_merge_fraction_pct`.

Evidence:

- Five L2-thrash/high-stride/working-set/SGEMM roots passed representative
  full-first replay (13 launches):
  `../sm86_replay_results/h36_fill_ownership_observer_20261015_v1/full.jsonl.zst`.
- Complete full-first replay then complementary fast replay passed 356 roots
  (286 `ok`, 62 mixed-code controls, 8 replay/input failures), 12,392 full
  launches in 40.379 s with exactly twelve workers:
  `../sm86_replay_results/full_corpus_h36_fill_ownership_observer_20261015_v1/full.jsonl.zst`
  and adjacent `fast.jsonl.zst`.
- Post-replay report:
  `../sm86_replay_results/metric_report_h36_fill_ownership_observer_20261015_v1/`.
- Totals are 1,727,757 pending-merge sectors and 4,823,652 new-fill sectors;
  pending merges occur in 1,822 launches. The nonempty-launch merge fraction
  is 0% minimum, 0% median, 2.273% p90, and 97.196% maximum.
- A keyed adversarial comparison against H35 found zero differences in every
  prior counter, metric prediction, DRAM field, timing/frontier value, and
  trace field after excluding the expected `sidecar_join_wall_ns` measurement
  and the three new observer rows. H36 is timing/traffic/IPC neutral.

The observer is promoted as a source-side fill-ownership ledger. It does not
claim proprietary GA102 pending-table size, fill-buffer count, or queue
latency. A future timing arm must be bounded by independent service evidence
and must use this observer only as a mechanism-local covariate, never as an NCU
fit or scalar global selector.

## 49. Slice S36 implementation evidence — H37 source-qualified PRT/fill pressure

Huerta's Ampere memory-queue/PRT semantics and modern GPGPU-Sim's finite MSHR,
waiter-merge, and fill/response queues motivate a pressure ledger. The local
RTX 3090 Ti clock64 queue-frontier probe, however, shows no stable transferable
latency after occupancy and dirty/coherence tails are separated. A bounded
one-cycle-per-merged-sector drain ablation (cap four sectors) changed 1,670
launch frontiers, improved GPC mean APE by only 0.004 percentage points, and
worsened SM-active APE; it was removed before promotion.

H37 instead adds a source-qualified observer. Immediately before each shared
L2 event request, the model records the number of pending fill sectors by
`L2Source::{L1Fill,Bypass,Untagged,Store}`. Full replay emits sample count,
pressure sum, and bounded mean rows. Pending merges remain lookup misses, new
fills alone own lower-level fetches, and no PRT capacity, hash, timing delay,
or NCU fit is introduced.

Evidence:

- Five L2-thrash/high-stride/working-set/SGEMM roots passed representative
  full-first replay (13 launches):
  `../sm86_replay_results/h37_prt_pressure_observer_20261015_v1/full.jsonl.zst`.
- Complete full-first replay then complementary fast replay passed 356 roots
  (286 `ok`, 62 mixed-code controls, 8 replay/input failures), 12,392 full
  launches in 40.638 s with exactly twelve workers:
  `../sm86_replay_results/full_corpus_h37_prt_pressure_observer_20261015_v1/full.jsonl.zst`
  and adjacent `fast.jsonl.zst`.
- Report:
  `../sm86_replay_results/metric_report_h37_prt_pressure_observer_20261015_v1/`.
- Pressure totals are L1-fill 2,727,211 samples/3,253,767,441 sector-samples,
  bypass 31,433/1,933,553, untagged 32/4,666, and store 3,279,482/198,743,493.
- A keyed H36 comparison found zero old-field differences after excluding the
  new pressure rows and nondeterministic `sidecar_join_wall_ns`; L1/L2/DRAM,
  IPC, timing, and frontiers are unchanged.

The pressure ledger is promoted as a residual-stratification surface only. A
future timing arm requires a fresh locked probe varying one queue factor at a
time and subtracting occupancy, writeback, and coherence tails before any
bounded service delay can enter default full replay.

## 50. Slice S37 implementation evidence — H38 L1 MSHR-full pressure

H38 instruments the existing finite L1 MSHR admission seam. Whenever a miss
finds the configured MSHR set full, the cache records a checked occupancy
sample and the number of entries in use. The fields are unified pressure
observers, separate from global/local traffic partitions; no delay, retry
change, cache mutation, or capacity inference is introduced.

Evidence:

- Five high-MSHR/high-stride/L2 roots passed representative full-first replay:
  `../sm86_replay_results/h38_l1_mshr_pressure_20261015_v1/full.jsonl.zst`.
- Complete full-first replay then complementary fast replay passed 356 roots
  (286 `ok`, 62 mixed-code controls, 8 replay/input failures), 12,392 full
  launches in 40.241 s with exactly twelve workers:
  `../sm86_replay_results/full_corpus_h38_l1_mshr_pressure_20261015_v1/full.jsonl.zst`
  and adjacent `fast.jsonl.zst`.
- Report:
  `../sm86_replay_results/metric_report_h38_l1_mshr_pressure_20261015_v1/`.
- Full-corpus pressure is 953,785 samples in 1,120 launches and 30,978,868
  occupancy entries (maximum sample count 124,766; maximum sum 3,992,512).
  Streaming/DSMEM roots each record 10,924 samples at occupancy 32; vendor
  SGEMM records 36; pointer-chase/working-set roots record none.
- A keyed H37 comparison found zero old-field differences after excluding the
  new MSHR rows and `sidecar_join_wall_ns`; all old L1/L2/DRAM/IPC/frontier
  values and APE remain unchanged.

The observer is promoted as an L1 service-residual partition only. A timing arm
requires an independent locked probe that separates occupancy from queue
service and dirty/coherence retirement tails; do not import a PRT capacity from
another Ampere implementation.

## 51. Slice S38 research evidence — H39 L1 lookup-retry correction rejected

H39 tested a concrete accuracy hypothesis: modern finite MSHR/PRT retry
semantics might mean that MSHR-full miss attempts are absent from the replay's
L1 lookup-sector denominator. The H38 full-first corpus supplied 107 exact
independent NCU joins. Stall-sample correlations with APE are −0.015 for
global-load sectors, −0.041 for lookup hits, −0.018 for lookup misses, 0.046
for GPC cycles, and −0.032 for SM-active cycles. High-stall quartiles show no
consistent lookup residual direction.

No denominator correction, timing arm, or NCU-derived adjustment is promoted.
The H38 pressure observer remains the active mechanism-local surface. A future
retry model requires a dedicated locked request-level probe that separates
retries from occupancy and unmodeled retirement tails.

Negative report: `docs/sm86_l1_lookup_retry_negative_20261015.md`.

## 52. Slice S39 implementation evidence — H40 L1 global-load scope audit

H40 adds a post-replay diagnostic that flags gross (>4x) disagreement between
trace-side logical memory-instruction counts and independent NCU opcode metrics.
The warning is machine-readable but never changes replay, prediction selection,
APE scoring, or NCU model inputs.

Evidence:

- Complete full-first replay then complementary fast replay passed 356 roots
  (286 `ok`, 62 mixed-code controls, 8 replay/input failures), 12,392 full
  launches in 40.199 s with exactly twelve workers:
  `../sm86_replay_results/full_corpus_h40_l1_scope_audit_20261015_v1/full.jsonl.zst`
  and adjacent `fast.jsonl.zst`.
- Report:
  `../sm86_replay_results/metric_report_h40_l1_scope_audit_20261015_v2/`.
- Exactly one full exact-identity join is flagged:
  `sm86_adapted_l2_074_sam_iou_backward_sm120` has 1,024 trace global-load
  instructions/4,352 sectors versus 16 NCU opcode instructions/40 sectors,
  a 64x scope ratio, while warp instructions agree exactly at 32,736.
- A keyed H38 comparison found zero differences in all replay fields after
  excluding the post-replay warning and `sidecar_join_wall_ns`; old APE values
  remain unchanged.

No L1 sector/lookup correction is promoted. The trace-derived sector domain and
independent NCU target remain visible with an explicit scope warning. Future
corrections require scope-matched NCU metrics, not a corpus-wide multiplier.

Compact report: `docs/sm86_l1_scope_audit_slice_20261015.md`.

## 53. Slice S40 implementation evidence — H41 L2/DRAM scope audit

H41 extends H40's fixed-ratio post-replay scope diagnostic to simulator L2
source-sector and DRAM read-sector counters versus independent NCU operation
metrics. Warnings are descriptive only; replay, prediction selection, APE, and
NCU model inputs remain unchanged.

Evidence:

- Complete full-first replay then complementary fast replay passed 356 roots
  (286 `ok`, 62 mixed-code controls, 8 replay/input failures), 12,392 full
  launches in 39.934 s with exactly twelve workers:
  `../sm86_replay_results/full_corpus_h41_l2_dram_scope_audit_20261015_v1/full.jsonl.zst`
  and adjacent `fast.jsonl.zst`.
- Report:
  `../sm86_replay_results/metric_report_h41_l2_dram_scope_audit_20261015_v1/`.
- Exact joins contain 121 DRAM read-sector warnings, 3 L2 source-sector
  warnings, and the one H40 global-load opcode warning. The SAM-IoU join has
  2,464 modeled L2 source sectors versus 168 NCU sectors and 1,076 modeled
  DRAM reads versus 564 NCU sectors.
- A keyed H40 comparison found zero old replay-field differences after
  excluding only new post-replay warnings and `sidecar_join_wall_ns`; old APE
  values remain unchanged.

No L2/DRAM multiplier or timing correction is promoted. Future accuracy work
requires scope-matched NCU metric sets or independently justified operation
projections, not global balancing. Compact report:
`docs/sm86_l2_dram_scope_audit_slice_20261015.md`.

## 54. Slice S41 implementation evidence — H42 scope-aware residual reporting

H42 makes H40/H41 scope warnings operational in the post-replay reducer. A
prediction covered by an explicit gross trace-vs-NCU scope warning remains in
the compressed row and warning inventory with status `scope_mismatch`, but is
excluded from raw/all-row comparison APE. No replay input, timing model,
prediction selector, or NCU-derived parameter changes.

Evidence:

- H42 reduces the H41 complete full-first artifacts:
  `../sm86_replay_results/full_corpus_h41_l2_dram_scope_audit_20261015_v1/full.jsonl.zst`
  and adjacent `fast.jsonl.zst`.
- Report:
  `../sm86_replay_results/metric_report_h42_scope_aware_20261015_v1/`.
- Full mode has 125 scope warnings and 67 excluded predictions across 107
  exact joins. Secondary like-for-like mean APE is L1 global-load sectors
  100.96%→0.21%, L2 source sectors 28.24%→6.76%, DRAM read sectors
  40.91%→26.84%, while GPC cycles remains 12.90%. Raw/all-row H41 means
  (100.96%/28.24%/40.91%) remain authoritative; exclusions are unresolved
  scope/model gaps, not completed mechanisms.

Scope-aware APE is a secondary like-for-like view with excluded rows retained
separately. The remaining scope-compatible DRAM residual is not below 10% and
remains an active modeling target; no overall closure claim is made. Compact report:
`docs/sm86_scope_aware_report_slice_20261015.md`.

## 55. Slice S42 research evidence — H43 scope-compatible DRAM residual ledger

H43 joins H42 scope-aware DRAM rows to H37 L2 route/fill-pressure and H38 L1
MSHR pressure observers. The retained corpus provides 58 scope-compatible exact
DRAM joins at 26.84% mean APE. All are balanced-route rows; concentrated and
incomplete route classes have no exact DRAM joins.

Pressure partitions are non-monotonic: low/high pending-merge quartiles are
28.69%/12.47% APE, low/high new-fill quartiles 40.92%/8.59%, and low/high MSHR
groups 28.59%/23.25%. Family, shape, and demand volume confound these shifts;
they do not identify a transferable partition/service law.

No queue delay, partition/hash selection, bandwidth multiplier, or NCU fit is
promoted. A new locked probe must vary one demand factor while independently
accounting for dirty writeback/coherence tails. Compact report:
`docs/sm86_dram_residual_ledger_slice_20261015.md`.

## 56. Slice S43 research evidence — H44 DRAM dirty-tail ledger

H44 joins the 58 scope-compatible exact DRAM rows with dirty residency,
dirty-eviction, and typed tail flags. Every row carries both
`dirty_writeback` and `coherence` tails; no row has dirty L2 eviction sectors,
so terminal L2 dirty residency is the varying demand.

Dirty-residency APE is non-monotonic: low/middle/high quartiles are 21.29%,
35.82%, and 12.10%. The relation is confounded by read demand, family, and
shape, so no writeback bandwidth, queue capacity, or coherence service law is
identified. Default writeback timing remains disabled; the optional scenario
and typed tails remain explicit.

Compact report: `docs/sm86_dram_dirty_tail_ledger_slice_20261015.md`.

## 57. Slice S44 research evidence — H45 IPC/cycle residual ledger

H45 joins exact cycle predictions to causal issue/terminal frontiers, H37/H38
pressure, route strata, and typed tails. On 107 exact joins, GPC mean APE is
12.90% and SM-active mean APE 18.02%. Balanced/incomplete/concentrated route
classes are 11.67%/16.08%/28.03% GPC APE (n=88/15/4) and
15.07%/31.76%/31.32% SM-active APE. Frontier gaps and pressure partitions are
non-monotonic, and sparse route classes cannot identify a timing law.

No IPC/cycle correction is promoted. Separate issue, SM-active, GPC, and typed
tail frontiers remain active; any timing arm needs a larger independent
route/queue probe. Compact report:
`docs/sm86_ipc_residual_ledger_slice_20261015.md`.

## 58. Slice S45 research evidence — H46 DRAM demand interval audit

The existing trace-derived DRAM demand interval brackets 34/58 (58.6%) H42
scope-compatible exact NCU DRAM read rows. Median low/high bounds are 262/392
sectors and target median is 392; high residuals often lie above the interval,
but workload family, demand, writeback/coherence, and operation scope remain
confounded.

No demand-bound expansion, multiplier, or timing change is promoted. The
interval remains a falsifiable observer; a new source-domain capture must
separate DRAM reads from writeback/coherence and profiler operation scope.
Compact report: `docs/sm86_dram_demand_interval_slice_20261015.md`.

## 59. Slice S46 implementation evidence — H47 dirty-writeback scenario A/B

H47 exercised the existing bounded `CardedWriteback` path at a predeclared
4,096-sector capacity on five high-terminal-dirty real traces. The capacity was
not selected from NCU values. GPC APE improved for qkv-rope (29.19%→24.92%),
RMSNorm (12.34%→11.51%), and Hyena FFT (8.38%→7.26%), was unchanged for Flash
attention (23.72%), and worsened for fused linear/GELU/GRN (24.69%→26.16%).
SM-active frontiers never changed because writeback is a post-issue tail.

The mixed outcome does not identify a transferable capacity/service law; no
default writeback timing promotion is made. Optional scenario and typed tails
remain explicit. Compact report:
`docs/sm86_writeback_scenario_negative_20261015.md`.

## 60. Slice S47 implementation evidence — H48 LDG constant-form observer

H48 adds static modifier-form counters for `LDG.*.CONSTANT` global loads:
`LTC128B`, `U16`, and other/plain. Public Nsight/source semantics distinguish
global LDG traffic from explicit constant-space LDC/CBANK paths, so no L0C/GCC
reroute is made. Form counters are source-qualified observers only.

Evidence:

- Complete full-first replay then complementary fast replay passed 356 roots
  (286 `ok`, 62 mixed-code controls, 8 replay/input failures), 12,392 full
  launches in 40.899 s with exactly twelve workers:
  `../sm86_replay_results/full_corpus_h48_constant_form_observer_20261016_v1/full.jsonl.zst`
  and adjacent `fast.jsonl.zst`.
- Report:
  `../sm86_replay_results/metric_report_h48_constant_form_observer_20261016_v1/`.
- Totals: LTC128B 199,446 instructions/828,751 sectors (141 launches), U16
  5,515,663/5,612,737 (71), other/plain 1,092,487/4,375,547 (688).
- SAM-IoU's global-load outlier is LTC128B; RMSNorm/qkv forms are U16, and
  streaming roots are other/plain. Old traffic, timing, IPC, frontiers, and
  APE remain unchanged.

No form-specific correction is promoted. A future timing/lookup arm requires a
locked form-specific probe varying reuse and working set; static form identity
alone does not prove a physical constant-cache client. Compact report:
`docs/sm86_constant_form_observer_slice_20261016.md`.

## 61. Slice S48 implementation evidence — H49 constant-form microprobe

The designated notes project probe
`GPU_simulator_snapshot/workloads/research_probes/constant_form_scope` builds
two identical-work SM86 kernels. Real executions pass identical functional
checksums, and cuobjdump confirms `LDG.E.CONSTANT` for 32-bit `__ldg` versus
`LDG.E.U16.CONSTANT` for uint16 `__ldg`.

This validates static form identity as an observer discriminator, not an L0C or
constant-space route; modern source/Nsight semantics keep LDG global traffic
distinct from LDC/CBANK. No lookup/timing correction or NCU fit is promoted.
The next form-specific probe should vary reuse distance and warp count while
retaining functional and SASS checks. Compact report:
`docs/sm86_constant_form_probe_slice_20261016.md`.

## 62. Slice S49 implementation evidence — H50 constant-form reuse/warp sweep

The notes probe swept 24 real RTX 3090 Ti points: plain versus uint16 `__ldg`,
1/4/8 warps, and reuse distances 1/4/32/256 lines. All points pass identical
functional checksums per pair; SASS remains `LDG.E.CONSTANT` versus
`LDG.E.U16.CONSTANT`. Clock64 grows with reuse/pressure but form differences
are small and non-monotonic (W1/reuse1 9,257 vs 9,258; W4/reuse32 23,089 vs
21,574; W8/reuse256 39,123 vs 39,108).

No form-specific timing or lookup correction is promoted. The matrix remains
independent form-scope evidence for a future NCU campaign. Compact report:
`docs/sm86_constant_form_sweep_slice_20261016.md`.

## 63. Slice S50 research evidence — H51 constant-form L1 lookup ledger

H51 joins H48 static form counters to exact independent NCU L1 lookup rows.
Other/plain forms show 5.85% hit and 31.53% miss mean APE (32/43 joins), U16
shows 51.36% hit and 34.90% miss (42/50), LTC128B has no positive-target hit
joins and 26.68% miss (3), and no-form rows show 6.89% hit/25.25% miss. Forms
remain global traffic observers; no correction or fitting is promoted.

Compact report: `docs/sm86_constant_form_residual_ledger_slice_20261016.md`.

## 64. Slice S51 audit evidence — H52 integrated non-regression

H48 has 12,392 launch keys, exactly matching H38; an adversarial comparison
finds zero old-field differences after removing only new form rows and
`sidecar_join_wall_ns`. H42/H48 report inventories retain compressed comparison
JSONL, summaries, 125 scope warnings, and 67 explicit exclusions. Workspace
Cargo/Python/diff checks pass. Scope exclusions remain unresolved and are not
closure claims. Their summary headings predate H63's raw-versus-secondary
correction; the regenerated H63 report is the active machine-readable scope
view. Compact report:
`docs/sm86_integrated_audit_slice_20261016.md`.

## 65. Slice S52 research evidence — H53 L2 line/sector residual ledger

On 104 H42 scope-compatible exact joins, L2 lookup-miss APE is exactly 0%.
Lookup-hit mean APE is 11.77% overall: balanced 12.50% (85 rows), incomplete
1.02% (15), concentrated 36.77% (4). Line/sector miss counts and pending/new
pressure are non-monotonic; no L2 multiplier or route timing arm is promoted.

Current modeled/effective L2 line/sector state semantics and source observers
remain active; the sparse concentrated hit residual needs independent
scope-matched probes.
Compact report: `docs/sm86_l2_semantic_residual_ledger_slice_20261016.md`.

## 66. Slice S53 implementation evidence — H54 independent NCU constant-form probe

NCU independently profiled plain 32-bit and uint16 `__ldg` controls with
identical work. At 1 warp/64 iterations/reuse1 both report 64 global-load
instructions/requests, 256 sectors, and 252 hits/4 misses; at 8 warps/64
iterations/reuse256 both report 512 instructions/requests, 2,048 sectors, and
1,764 hits/284 misses. No form-specific lookup differential appears in these
controls; results were not replay inputs or fitted parameters.

Retain H48 form observers and defer form-specific lookup/timing correction until
a broader controlled sweep including LTC128B. Compact report:
`docs/sm86_constant_form_ncu_probe_slice_20261016.md`.

## 67. Slice S54 evidence — H55 consolidated accuracy dashboard

`docs/sm86_accuracy_dashboard_20261016.md` consolidates raw/all-row H41 APE and
secondary H42 scope-compatible APE, excluded counts, full-corpus coverage, and
unresolved tails. Raw values remain authoritative; secondary values are not
closure claims. The dashboard records 356 roots, 12,392 launches, 107 identity
joins, and current L1/L2/DRAM/IPC residuals.

## 68. Slice S55 research evidence — H56 scope-warning threshold sensitivity

On H41 exact joins, L2 scope exclusions are 6/3/3 and DRAM exclusions 70/49/29
at 2×/4×/8× thresholds; DRAM secondary APE is 19.15%/26.84%/32.81%. The 4×
threshold remains a fixed gross-scope diagnostic, not an optimization knob;
raw/all-row APE remains authoritative. Compact report:
`docs/sm86_scope_threshold_sensitivity_slice_20261016.md`.

## 69. Slice S56 research evidence — H57 SOTA DRAM queue prior

Source-inspection notes from GPGPU-Sim/Accel-Sim describe per-partition
private/shared credits, finite L2↔DRAM queues, and round-robin arbitration;
Huerta adds finite memory-subcore/PRT/coalescer structures. No Accel-Sim
binary/runtime or imported constant is used. These generic priors do not identify
GA102 partition hashes/capacities or completion ownership. H43/H47 evidence is
confounded/mixed, so no partitioned timing arm or imported capacity is enabled.

Observer-only candidate pressure and typed writeback/coherence tails remain the
scientifically valid endpoint until a locked SM86 probe isolates one demand
factor. Compact report:
`docs/sm86_dram_sota_queue_prior_slice_20261016.md`.

## 70. Slice S57 audit evidence — H58 branch integration

Release `sim-trace`/`sim-run` builds, workspace all-features checks, Python
compilation, and diff checks pass. Five current real traces replay full-first
5/5 (99 launches) with compressed output and the twelve-worker runner. H48/H42
artifacts/reports and H58 smoke output are present; active maps contain no stale
SM120 default/corpus paths. Compact report:
`docs/sm86_integration_audit_slice_20261016.md`.

## 71. Slice S58 research evidence — H59 GPC observation-envelope residual

The measured SM86 envelope (`1776 + 8·floor(log2(grid))`) is non-monotonic
against 107 exact NCU joins: log₂ 0/2/3/8/10 groups show 12.11%/9.56%/17.51%/
21.76%/29.19% APE with sparse high-grid samples; active-SM partitions are also
non-monotonic. No grid/SM multiplier or NCU correction is promoted. Compact
report: `docs/sm86_gpc_envelope_residual_slice_20261016.md`.

## 72. Slice S59 evidence — H60 unresolved mechanism gap tracker

The explicit tracker records raw/secondary L1/L2/DRAM/IPC residuals, typed
writeback/coherence tails, 125 scope warnings/67 exclusions, sparse route/grid
strata, and four concrete probe requirements. No gap is renamed, silently
excluded, or declared physically recovered. Report:
`docs/sm86_unresolved_gap_tracker_20261016.md`.

## 73. Slice S60 evidence — H61 LTC128B probe boundary

Direct `ld.global.LTC128B.u32` PTX is rejected by CUDA 13.2 `ptxas` with
`Unknown modifier '.LTC128B'`. LTC128B remains an observer from H48 traces and
cuobjdump; no invalid inline opcode or binary patch is attempted. A supported
SASS-generation/injection path is required before an independent LTC128B NCU
sweep. Compact report:
`docs/sm86_ltc128b_probe_blocker_slice_20261016.md`.

## 74. Slice S61 audit evidence — H62 full-mode capability contract

The active full runner enables all implemented/evidence-backed SM86 observers
and models by default: integrated L1/L2, aggregate carded DRAM/read service,
launch-scoped store/fence feedback, source/route pressure, MSHR pressure, form
observers, IPC/frontier projections, and scope diagnostics. Partition
candidates remain observer-only. Typed trace events gate optional direct
`sim-run` writeback/coherence/visibility scenarios; the complete-corpus driver
does not synthesize those flags. `stream-l2` intentionally suppresses the
launch-scoped store/fence arm and records that scoped gap rather than emitting
replay errors. Plan wording was reconciled with this contract; runtime is
unchanged.

## 75. Slice S62 adversarial consistency evidence — H63 — 2026-10-16

The H30–H62 source/plan review found and corrected two semantic defects. First,
H42 prose and the reducer text called scope-compatible rows “primary” even
though explicit scope-mismatch rows are excluded from raw/all-row comparison
APE. The reducer/report now labels the table a secondary scope-compatible view;
rows remain compressed and raw/all-row H41 remains authoritative. Second, the
DRAM section described a finite partition/bank controller as active although the
landed full path uses `DramBw::Carded` aggregate read service, H30–H37
observer-only partition/route covariates, and the H19 store/fence feedback
candidate. Finite partition/bank timing is now explicitly future and
independent-evidence gated; dirty writeback/coherence/visibility/payload arms
remain optional or typed unavailable by default.

H57–H62 entries were reordered chronologically and H62 names aggregate carded
DRAM plus observer-only partition candidates. Additional source consistency
fixes qualify modeled cache-state/trace line-domain observers that were called
“physical,” document source-only GPGPU-Sim/Accel-Sim inspection, and identify
the 825 GB/s value as an unlocked absolute-rate hypothesis mapped onto the
locked simulation clock. The runner now exposes an explicit
`--no-memory-admission-feedback` ablation, defaults the launch-scoped H19
candidate only for launch scope, records stream-L2 suppression, and emits the
configured service rate in each DRAM counter report.

The real H41 full/fast artifacts were reduced with the corrected reporter into
`../sm86_replay_results/metric_report_h63_scope_label_audit_20261016_v1/`:
125 full warnings, 67 full exclusions, and 0/0 fast warnings/exclusions. No
cache/DRAM timing law, NCU fitting, physical-topology claim, or accuracy
closure claim changed. A real stream-L2 full replay of one SM86 capture
completed `ok` with candidate suppression. Release builds, Cargo workspace
checking, Python compilation, and `git diff --check` pass. Compact report:
`docs/sm86_plan_consistency_audit_slice_20261016.md`.

## 76. Slice S63 research evidence — H64 L2 queue-frontier negative — 2026-10-16

The independent notes probe `l2_queue_frontier` was rebuilt and swept on the
identified RTX 3090 Ti host. Its 288-point unique/reuse, warp, burst, stride,
and reuse matrix passed every correctness/timestamp/checksum check. Twenty
repeats show a robust clock64 stride differential at eight warps/64 loads:
median maximum cycles 20,765/28,137/32,334 for strides 1/8/32. However,
selected NCU points show only about a 4% `lts__cycles_active.sum` change and
zero selected L1TEX request-stalled cycles; the differential is not an isolated
L2 service frontier.

No finite queue capacity, hash, stride delay, or merged-sector drain law is
promoted. H30–H37 route/slice/fill/pressure observers remain timing-neutral.
The full-corpus H37 timing arm's 1,670 frontier changes, ~0.004-point GPC gain,
and SM-active regression remain the rollback precedent. Next evidence is a
locked one-factor DRAM demand/completion probe that separates read demand from
dirty writeback/coherence before any service change. Compact report:
`docs/sm86_l2_queue_frontier_negative_20261016.md`.

## 77. Slice S64 research evidence — H65 P1 wavefront lookup negative — 2026-10-16

The rebuilt same-binary `l1tex_arrival_replay_service` NCU probe confirms cold
line-layout loads have zero L1 lookup hits (21,504 sectors, 21,504 misses),
while output-wavefront counts are a separate endpoint. An experimental
replacement of the H16 projection with `hits + pending_lookup_hits` was then
run full-first on all real roots (12 workers, compressed output). It reduced
zero-target hit false positives from 22/29 to 1/29 but worsened mean APE from
30.71% to 33.43% for lookup hits and 58.44% to 110.05% for lookup misses.

The unqualified wavefront formula is rejected and the source is restored to the
H16/H63 projection. Any future correction must be packet/client/form-qualified
and retain sector conservation; wavefront counters remain separate observers.
Compact report: `docs/sm86_l1_wavefront_lookup_negative_20261016.md`.

## 78. Slice S65 research evidence — H65 supported LTC128B probe — 2026-10-16

CUDA 13.2 `ptxas -arch=sm_86` accepts `ld.global.ca.L2::128B.u32` and emits
`LDG.E.LTC128B.STRONG.SM`; a same-binary probe executed successfully on the
identified RTX 3090 Ti. At 84 blocks, eight warps, 64 iterations, and eight
active lanes, plain `.ca` produced 43,008 L2 TEX read-miss sectors, zero
promoted sectors, and 43,176 DRAM read sectors. `.L2::128B` produced 172,032
L2 TEX read-miss sectors, 129,024 promoted sectors, and 172,672 DRAM read
sectors, with the same global-load sectors and zero L1 lookup hits.

This validates a supported frontend and the promotion observation, but no
simulator timing arm is enabled: a complete form/working-set matrix, NVBit
capture/holdout, source scope accounting, and typed promoted-sector observer
are still required. The H40/H41 64× scope outlier is not attributed to LTC128B
by this control alone. Compact report:
`docs/sm86_ltc128b_ptx_probe_slice_20261016.md`.

## 79. Slice S66 implementation evidence — H66 GCC/L1.5 demand observer — 2026-10-16

An independent zero-global-data/constant-load probe on the RTX 3090 Ti confirms
GCC/L1.5 traffic distinct from TEX data: arithmetic control reports 1,764 GCC
requests, 952 GCC L2 sectors, 125 GCC misses, and 3,100 DRAM reads; a constant
load arm reports 562,919 GCC requests, 1,960 GCC L2 sectors, 269 GCC misses, and
276 DRAM reads. NCU values remain probe observations only.

Full replay now emits observer-only `dram_gcc_instruction_sector_candidate`,
`dram_gcc_constant_sector_candidate`, and
`dram_read_demand_multiclient_candidate_low` rows from existing trace-domain
information. The complete real replay passed 356 roots (286 `ok`, 62 controls,
8 failures), 12,392 launches, exactly 12 workers, and compressed artifacts at
`../sm86_replay_results/full_corpus_h66_gcc_demand_observer_20261016_v1/`.
No L2 state, `DramStats.read_sectors`, service rate, or timing frontier changed.
A same-binary NVBit GCC/TEX scope crosswalk is required before any multi-client
timing model. Compact report:
`docs/sm86_gcc_demand_observer_slice_20261016.md`.

## 80. Slice S67 research evidence — H67 L2 write-lookup split negative — 2026-10-16

A same-binary store probe reported cold line stores as zero lookup-hit sectors,
but replacing the corpus observer's conditional write-tag arm with
`lookup.present_sectors` was falsified by full-first real replay. On 104
scope-compatible joins, source-sector APE remained 6.76%, while lookup-hit APE
worsened 11.77%→33.67% and lookup-miss APE worsened 0.0%→53.16%.

The experiment was reverted; no L2 state, DRAM, or timing changed. The store
probe and scored source-unit endpoints are not yet like-for-like. Operation-
split NCU/TEX scope and NVBit crosswalk evidence are required before any write
coalescing or lookup split. Compact report:
`docs/sm86_l2_write_lookup_negative_20261016.md`.

## 81. Slice S68 research evidence — H68 no-fence store-feedback negative — 2026-10-16

The existing independent store-ack matrix shows no-fence store tails above
control (roughly 3,500 cycles at 84-block/depth-16 and 11,400 at depth-64),
but merely broadening H19 candidate qualification from explicit fences to all
STORE/RED kernels changed no full-replay frontiers or admission exposure on
five real roots (20 launches). The candidate is therefore not an implicit
CTA-EXIT drain model; the qualification was restored. A true EXIT-gate arm
requires independent no-fence/fence evidence and a full-corpus A/B. Compact
report: `docs/sm86_store_feedback_no_fence_negative_20261016.md`.

## 82. Slice S69 research evidence — H69 DRAM demand/completion negative — 2026-10-16

The retained same-binary RTX 3090 Ti probe matrix was audited: memory-retirement
has 102 cells (94 instruction-crosswalk matches, 8 explicit exclusions),
store-ack has 30 crosswalk-valid cells, and launch-cycle/DRAM streams provide
matched lower-bound observations. All retained memory-retirement cells are
lower-bound consistent but have no hardware completion token or global event
order; store-tail deltas vary strongly with blocks/depth; one DRAM stream is a
lower-bound falsifier while another is consistent. Dirty writeback, coherence,
EXIT drain, and profiler envelope cannot be separated by a transferable law.

No timing-active DRAM/writeback/coherence model is promoted. Aggregate carded
read service, typed tails, H66 GCC observers, and raw/all-row H41 authority
remain unchanged. A future timing arm requires same-binary NVBit plus locked
one-factor endpoint-matched evidence. Compact report:
`docs/sm86_dram_demand_completion_negative_20261016.md`.

## 83. Slice S70 implementation evidence — H70 per-request L2 frontier observer — 2026-10-16

The shared-L2 event path now emits bounded observer-only per-request issue-to-
fill, issue-to-response, and service-delay sums/maxima plus sample counts. It
uses existing `req.issue_time`, `fill_ready`, `response.ready`, and service-plan
frontiers; it does not alter L2/DRAM state, response ordering, or timing.

Release builds and full-first corpus replay passed 356 roots (286 `ok`, 62
controls, 8 failures), 12,392 launches, exactly 12 workers, with compressed
artifacts at `../sm86_replay_results/full_corpus_h70_l2_frontier_observer_20261016_v1/`.
No NCU alias or tuning input is attached. Compact report:
`docs/sm86_l2_frontier_observer_slice_20261016.md`.

## 84. Slice S71 implementation evidence — H71 terminal-EXIT store drain — 2026-10-16

### Evidence

A matched same-binary locked RTX 3090 Ti probe isolated the independent stall
domain required by the remaining SM-active residual. The no-fence store arm
reported `sm__cycles_active.max=10,345`, `smsp__cycles_active.max=10,327`,
253,415 drain-stall cycles, and zero MEMBAR-stall cycles. The explicit
`MEMBAR.GL` arm reported 11,730/11,712 cycles, 1,915 drain-stall cycles, and
488,950 MEMBAR-stall cycles. This is a real endpoint-domain observation, not an
NCU-fitted replay input.

### Implemented seam

`MemoryAdmissionCandidate` accepts an explicit
`exit_waits_for_service_completion` flag. When enabled, a terminal dynamic
`EXIT` waits only while the owning warp has an unpriced STORE admission or a
pending typed write acknowledgement. The card keeps this store-only arm
explicitly disabled by default after the corpus result. Feedback ACK service
start is clamped to the L2 pricing event, and stale actor-local wakes are
rearmed at the current shared event-clock boundary to preserve causality.

### Full-first result and decision

Enabled-arm artifact: `../sm86_replay_results/full_corpus_h71_exit_drain_20261016_v2/`;
default artifact: `../sm86_replay_results/full_corpus_h71_exit_drain_20261016_v3/`.
Both complete full-first replay with 356 roots, 286 `ok`, 62 controls, 8
replay errors, 12,392 full launches, exactly 12 workers, and compressed output.
The enabled arm changes secondary mean APE from H70 as follows: GPC
12.9008%→12.8836%, SM-active.avg 18.0202%→18.1389%, SM-active.max
17.8377%→17.8763%, SMSP-active.avg 17.5137%→17.6580%, and IPC projection
14.4865%→14.5529%. The default v3 report is bit-for-bit equivalent to H70 on
these surfaces. Therefore H71 is a falsifiable negative for promotion: retain
the bounded implementation as disabled experimental infrastructure, keep the
aggregate DRAM/read service and typed dirty/coherence tails unchanged, and do
not claim an EXIT timing law.

### Next evidence gate

Promotion requires a same-binary NVBit crosswalk of terminal EXIT events to the
exact STORE acknowledgement population, explicit STORE-vs-RED coverage, and a
held-out no-fence/fence endpoint-matched matrix. NCU remains post-replay ground
truth only. Compact report: `docs/sm86_exit_drain_slice_20261016.md`.

## 85. Slice S72 implementation evidence — H72 LDGSTS/LTC128B scope crosswalk — 2026-10-16

### Implemented observer

Native `TraceMetrics` now keeps static-SASS form partitions separate from the
broad logical memory ledger: `global_load_ldg`, `global_load_constant`,
`global_load_ltc128b`, `global_load_strong_gpu`, and `ldgsts_async_copy`.
SM86 native metrics expose plain-LDG and LDGSTS opcode candidates. The
post-replay reporter uses these partitions only for endpoint scope diagnostics:
plain opcode mismatches invalidate the opcode metric, while a gross LTC128B
mismatch additionally excludes the L1 global sector/lookup endpoints from the
secondary like-for-like view. Broad L1/L2/DRAM traffic and timing are not
relocated.

### Evidence

`cp_async_shared_tile_copy` independently confirms 32 LDGSTS records/128 L1
sectors, NCU LDGSTS opcode count 32, and global-LD opcode count zero; LDGSTS
therefore remains in L1TEX global-op sector traffic. The SAM H74 control has
1,024 `LDG.E.LTC128B.CONSTANT` records but NCU global-LD requests/sectors 16/40,
TEX read sectors 40, and separate GCC constant/instruction demand. Vendor and
snapshot controls show active LTC128B forms can still be counted in TEX global
traffic, so a universal subtraction rule is rejected.

### Full-first result and decision

Full-first replay artifact:
`../sm86_replay_results/full_corpus_h72_ldgsts_scope_20261016_v1/`.
Coverage remains 356 roots (286 `ok`, 62 controls, 8 replay errors), 12,392
full launches, exactly 12 workers, compressed output. Report:
`../sm86_replay_results/metric_report_h72_ldgsts_scope_20261016_v2/`.
The scope diagnostic contains 124 operation warnings, seven plain opcode
warnings, and one LTC128B-specific warning. Secondary global-load sector APE
is 0.21% (105/106 exact), while LDGSTS opcode APE is 0% (107/107 exact).
The raw/all-row authority remains retained and is not claimed improved by
scope labelling. H72 promotes only the observer/crosswalk; no LTC128B DRAM
promotion, GCC/TEX relocation, or timing change is active.

### Next evidence gate

A timing or counter relocation requires a same-binary NVBit/SASS crosswalk over
LTC128B.CONSTANT, STRONG.GPU, vendor kernels, and LDGSTS controls. NCU remains
post-replay ground truth only. Compact report:
`docs/sm86_ldgsts_ltc128b_scope_slice_20261016.md`.

## 86. Slice S73 research evidence — H73 GCC/TEX source-client crosswalk — 2026-10-16

The same-binary CUDA13.2 GCC probe confirms independent GCC/L1.5 traffic, but
fresh NVBit capture attempts with both CUDA13.4 and CUDA13.2/g++-14 binaries
segfault during tracer context initialization before dynamic protobuf output.
The post-replay reporter now emits observer-only GCC candidate crosswalk
warnings; on the full H72 artifact, 286 instruction and 200 constant candidate
rows are grossly mismatched to NCU GCC endpoints. This falsifies the existing
trace-domain candidates as direct GCC request/sector predictions.

No `DramStats.read_sectors`, L2 state, aggregate service, or timing frontier is
changed. H73 therefore remains a negative crosswalk: repair the NVBit build and
capture dynamic source-client identity, then establish completion ownership on
held-out controls before promoting P4 multi-client DRAM demand. Compact report:
`docs/sm86_gcc_tex_crosswalk_slice_20261016.md`.

## 87. Slice S74 implementation evidence — H74 L2 operation-split observers — 2026-10-16

The full replay now publishes read/write/atom-dot-alu/reduction L2 TEX source
sectors, lookup hits, and checked derived misses from `L2SourceSplit`. This
observer surface is active in default full mode and changes no lookup policy,
cache state, DRAM service, or timing frontier.

Full-first H74 replay at `../sm86_replay_results/full_corpus_h74_l2_operation_split_20261016_v1/`
retains 356 roots, 286 `ok`, 62 controls, 8 replay errors, and 12,392 full
launches with twelve workers. Secondary APE: read sectors 1.90%, read hits
5.44%, read misses 0%; write sectors/hits ~39.9%, write misses exact/zero-target;
atom-dot-alu and RED rows exact. The write residual falsifies promotion of a
P3 write-coalescing/timing arm without a same-binary L1TEX-to-XBAR crosswalk.
Compact report: `docs/sm86_l2_operation_split_slice_20261016.md`.

## 88. Slice S75 research evidence — H75 L1TEX/XBAR write coalescing — 2026-10-16

A locked-clock CUDA13.2 SM86 probe sweeps distinct-line, same-sector
cross-warp, and spaced same-sector stores. NCU L2 `op_write` and XBAR writes
are 86,016, 5,699, and 6,651 sectors respectively; same-sector lookup hits are
6,213 and 5,888 with zero misses while L1 store sectors remain 21,504. The
mechanism is real, but lookup hits exceed packet sectors and spacing is not
monotonic. No P3 timing or lookup policy is promoted. Require dynamic NVBit
write-request identity and cold/warm endpoint crosswalk. Report:
`docs/sm86_l1tex_write_coalescing_probe_slice_20261016.md`.

## 89. Slice S76 implementation evidence — H76 constant-client interval observer — 2026-10-16

Native TraceMetrics now counts dynamic static-SASS LDC/ULDC records separately
from CBANK address-domain data. Rebase-tracer captures provide 16,128 arithmetic
and 360,192 constant-mode CBANK-tagged records, but NCU GCC requests are
1,008/791 and 562,212/742; the domains are not equivalent. Full-first replay
at `../sm86_replay_results/full_corpus_h76_constant_client_interval_20261016_v1/`
retains 286/62/8 coverage and 12,392 launches. Promote only the observer; no
GCC/DRAM timing or L2 state change. Report:
`docs/sm86_constant_client_interval_slice_20261016.md`.

## 90. Slice S77 implementation evidence — H77 dynamic CBANK tracer repair prototype — 2026-10-16

The rebase tracer was repaired to instrument immediate LDC/ULDC CBANK forms and
emit additive dynamic constant fields (addresses/bank/offset tags 15–17). The
replay-bench adapter accepts enhanced execution-info metadata, derives checked
control words, and narrowly falls back to `enhanced_execution_info.json` when
the native sidecar is absent. Real mode-1 and mode-0 capture/replay succeeds:
mode 1 reports 344,064 LDC, 2,688 ULDC, 11,526,144 addresses, and 147 unique
sectors; mode 0 is a valid no-LDC control with 2,688 ULDC records. The required
full-first replay after this loader repair remains 286/62/8 over 12,392 full
launches with twelve workers at
`../sm86_replay_results/full_corpus_h77_adapter_20261016_v2/`; report:
`../sm86_replay_results/metric_report_h77_adapter_20261016_v2/`. The active
branch contains no tracer source, so this remains a notes/rebase prototype; no
GCC request/sector collapse, L2 state, or DRAM timing is enabled. Next:
checked-in transport integration or a bounded dynamic source-client crosswalk.
Report: `docs/sm86_dynamic_cbank_tracer_slice_20261016.md`.

## 91. Slice S78 implementation evidence — H78 metric-unit crosswalk — 2026-10-16

The post-replay reporter now accepts native SM86 units (`%`, `byte`, `request`,
`thread/block`, `block/SM`, and `register/thread`) and applies explicit
name-qualified Kbyte/Mbyte scaling for DRAM/shared-memory rows. H77's
full-first artifact was re-reported without replay mutation: identity joins
remain 107, coverage remains 286/62/8 over 12,392 full launches with twelve
workers, and compared metrics increase from 40 to 63. Only
`gpu__time_duration.sum` remains a unit mismatch because no clock crosswalk is
available. This is a reporting-dimension repair, not an IPC/L1/L2/DRAM timing
claim. Compact report: `docs/sm86_metric_unit_crosswalk_slice_20261016.md`.

## 92. Slice S79 implementation evidence — H79 packet-aware L1 lookup observer — 2026-10-16

The L1 request-boundary audit now distinguishes ordinary-global same-request
sector merges from older pending/LDGSTS merges. Full replay emits bounded
packet-aware hit/miss candidates and interval edges without changing cache
state, MSHR ownership, L2/DRAM service, or timing. Full-first v2 replay:
`../sm86_replay_results/full_corpus_h79_packet_observer_20261016_v2/`; report:
`../sm86_replay_results/metric_report_h79_packet_observer_20261016_v2/`;
coverage 286/62/8 and 12,392 launches with twelve workers. On 106
scope-compatible joins, hit APE improves 30.71%→17.50% and miss APE
32.39%→19.53%; false-positive zero-target hits fall 22/29→8/29. Because the
authoritative raw/all-row LTC128B scope row regresses under the candidate, the
baseline remains the default selected prediction. Promotion requires a
form-qualified same-binary endpoint witness. Compact report:
`docs/sm86_l1_packet_aware_lookup_slice_20261016.md`.

## 93. Slice S80 implementation evidence — H80 L2 write-packet coalescing — 2026-10-16

The default full replay now selects a bounded per-SM `(line, sector)` write
packet candidate within an 8-cycle issue window only for
`lts__t_sectors_srcunit_tex_op_write.sum`; the request-sector baseline remains
diagnostic. No L2 state, lookup-hit/miss policy, DRAM demand, writeback, or
timing frontier changes. Full-first v2 replay:
`../sm86_replay_results/full_corpus_h80_write_packet_observer_20261016_v2/`;
report:
`../sm86_replay_results/metric_report_h80_write_packet_observer_20261016_v2/`;
coverage 286/62/8, 12,392 launches, twelve workers. Write-sector mean APE
improves 39.86%→9.37%, P90 100%→0%, exact nonzero joins 77→100. This is an
effective endpoint model, not a physical XBAR claim. Compact report:
`docs/sm86_l2_write_packet_observer_slice_20261016.md`.

## 94. Slice S81 implementation evidence — H81 L2 write lookup packet composition — 2026-10-16

The H80 write-packet candidate now composes with operation-split write
lookup-hit/miss rows: candidate hits are bounded by packet sectors and misses
are the checked remainder, while request-domain baselines remain diagnostic.
Full-first replay:
`../sm86_replay_results/full_corpus_h81_write_lookup_packet_20261016_v1/`;
report:
`../sm86_replay_results/metric_report_h81_write_lookup_packet_20261016_v1/`;
coverage 286/62/8, 12,392 launches, twelve workers. Write-sector APE is 9.37%,
write lookup-hit APE 9.36%, and misses remain exact zero-target rows. No
aggregate L2/DRAM/timing/IPC change or physical topology claim. Compact report:
`docs/sm86_l2_write_lookup_packet_slice_20261016.md`.

## 95. Slice S82 implementation/review evidence — H82 DRAM route-concentration diagnostic — 2026-10-16

Full replay emits normalized maximum candidate partition/slice demand fractions
and a joint GPC-frontier observer without selecting a hash or changing
`DramBw::Carded`, L2, DRAM, or timing. Full-first replay:
`../sm86_replay_results/full_corpus_h82_route_concentration_20261016_v1/`;
report:
`../sm86_replay_results/metric_report_h82_route_concentration_20261016_v1/`;
coverage 286/62/8, 12,392 launches, twelve workers. Median concentration is
175,631 ppm, P90 500,000, 23 launches exceed 500,000, and GPC/SM APE is
unchanged. Independent review rejects finite partition/queue timing promotion:
H43 compatible DRAM joins are balanced, route/pressure residuals are sparse or
non-monotonic, and H64 request-stall movement is ~4% and confounded. Compact
report: `docs/sm86_dram_route_concentration_diagnostic_slice_20261016.md`.

## 96. Slice S83 implementation/review evidence — H83 IPC readiness concentration observer — 2026-10-16

Full replay emits dominant exclusive readiness-cause fraction and total
limiting-cycle observers beside the existing issue/SM-active frontier interval;
no scheduler, cache, DRAM, or timing change. Full-first replay:
`../sm86_replay_results/full_corpus_h83_ipc_readiness_20261016_v1/`;
report:
`../sm86_replay_results/metric_report_h83_ipc_readiness_20261016_v1/`;
coverage 286/62/8, 12,392 launches, twelve workers. Median concentration is
425,114 ppm, P90 613,776, 103 launches exceed 500,000, and GPC/SM APE remains
unchanged. No IPC timing promotion without held-out endpoint-matched evidence.
Compact report: `docs/sm86_ipc_readiness_concentration_slice_20261016.md`.

## 97. Slice S84 integration evidence — H84 integrated SM86 accuracy dashboard — 2026-10-16

The latest H83 full-first artifact is consolidated in
`docs/sm86_integrated_accuracy_dashboard_20261016.md`. Coverage remains
286/62/8 over 12,392 full launches with twelve workers. Current mechanism-local
mean APE: L1 sectors 0.21%, L2 TEX sectors 6.76%, H80/H81 write sectors/hits
9.37%/9.36%, DRAM reads 26.84%, GPC 12.90%, and SM-active 18.02%. H80/H81 are
the only newly selected effective operation endpoints; route/readiness,
partition, GCC, and speculative timing arms remain bounded observers or
unresolved gaps. Next work must target L1 lookup scope, multi-client DRAM, or
IPC retirement timing with independent evidence.
## 98. Slice S85 research evidence — H85 independent GCC/constant demand matrix — 2026-10-16

Locked RTX 3090 Ti CUDA13.2/g++-14 matrix varies arithmetic/one-LDC/two-LDC
forms, repetitions, and constant footprint. GCC requests scale 7,143→590,880
while L2 GCC sectors move 2,128→2,240 and DRAM reads 296→316; two-LDC reaches
2,127,353 requests but only 3,248 sectors and 452 DRAM reads. This falsifies a
request-count-to-DRAM rule and blocks naive P4 timing promotion. Keep dynamic
GCC/constant observers and aggregate carded DRAM; require source-client
completion ownership and held-out matrix evidence. Compact report:
`docs/sm86_gcc_demand_matrix_slice_20261016.md`.

## 99. Slice S86 review evidence — H86 integrated adversarial review — 2026-10-16

Workspace Cargo checks/release build, Python compilation, diff checks, zstd
artifact tests, replay coverage, and all 12,392 promoted write conservation
rows pass on the H83 artifact. NCU remains post-replay only; H80/H81 change
only operation-write endpoint predictions; H82/H83 are observers; H85 blocks
request-count-to-DRAM fitting. Remaining gates are L1 lookup scope,
GCC/multi-client DRAM completion ownership, and IPC retirement timing. Compact
report: `docs/sm86_integrated_adversarial_review_slice_20261016.md`.

## 100. Slice S87 research evidence — H87 L1 lookup form-scope review — 2026-10-16

H79 candidate stratification over H83 traces/NCU joins improves the dominant
`global_load_constant` stratum (hit 30.86%→16.95%, miss 31.34%→17.53%) but
worsens four LTC128B rows (miss 725.01%→1,683.10%); mixed LDGSTS forms differ.
Static aggregate scope labels do not justify universal subtraction. Keep the
baseline selected and require per-request client/form identity before an L1
lookup promotion. Compact report:
`docs/sm86_l1_lookup_form_scope_review_slice_20261016.md`.

## 101. Slice S88 implementation evidence — H88 constant-client reuse observers — 2026-10-16

Full replay emits dynamic constant metadata coverage and reference/request
reuse ratios without entering L1/L2/DRAM state or timing. Full-first replay:
`../sm86_replay_results/full_corpus_h88_constant_reuse_20261016_v1/`;
report:
`../sm86_replay_results/metric_report_h88_constant_reuse_20261016_v1/`;
coverage 286/62/8, 12,392 launches, twelve workers. GPC and DRAM-read APE are
bit-equivalent to H83. H85's negative request-to-DRAM result remains the P4
promotion boundary. Compact report:
`docs/sm86_constant_reuse_observer_slice_20261016.md`.

## 102. Slice S89 implementation/research evidence — H89 repaired dynamic-CBANK H85 crosswalk — 2026-10-16

Repaired external CUDA13.2 NVBit captures of H85 mode 0/1 carry dynamic CBANK
fields 15–17 and replay successfully through the active adapter. Mode 1 reports
344,064 LDC, 5,376 ULDC, 712,320 references, 22,794,240 addresses, 147 unique
sectors, versus NCU 590,041 GCC constant requests and 2,240 GCC L2 sectors.
Transport is validated, but domains are not equivalent; no P4 request/sector
conversion or timing is promoted. Compact report:
`docs/sm86_repaired_cbank_crosswalk_slice_20261016.md`.

## 103. Slice S90 reporting evidence — H90 complete simulator-vs-NCU accuracy comparison — 2026-10-16

The complete H88 comparison is published at
`docs/sm86_accuracy_comparison_report_20261016.md`: all 63 finite comparable
metric names, mean/median/P90 APE, compared/exact/scope counts, zero-target
semantics, unavailable inventory, 107 exact identity joins, 179 fallback joins
excluded from APE, and 12,392 replay launches. This report is the authoritative
accuracy summary for the current branch state.

## 104. Slice S91 reporting evidence — H91 expanded ordered-fallback accuracy — 2026-10-16

Per explicit user authorization, same-workload ordered-fallback pairs are
included in a separate expanded simulator-vs-NCU report. The expanded report
scores 286 successful root pairs (107 exact + 179 fallback), while the strict
exact-only report remains unchanged. Complete table:
`docs/sm86_accuracy_comparison_ordered_fallback_20261016.md`.
