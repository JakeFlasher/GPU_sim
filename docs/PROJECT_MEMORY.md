# Project memory

Updated 2026-09-26. A compact continuation note for Codex and other agents;
root [AGENTS.md](../AGENTS.md) remains the project execution profile.

## Resume here

- [STATUS](STATUS.md): implemented paths, measured evidence, open boundaries.
- [Runbook](runbook_host.md): commands from the repository root.
- [Profile contracts](archprofile.md) and [custom programs](../arch_profiles/programs/README.md).
- [Roadmap](../plans/cross_architecture_gpu_simulator_research_plan_20260923.md):
  research choices, not an endless implementation queue.

## Durable decisions

- Software architectural simulation only. Native NVBit/SASS replay is anchored
  on SM86. Custom `simt_u32_v1` has a separate model-relative pipeline;
  `simt_its_v1` is a single-group functional example.
- NCU is an independent comparison target. No RTL, silicon internals, universal
  binary decoder, or arbitrary compiler/ISA generator is required.
- Fair numerical Accel-Sim/MGPUSim comparison remains **deferred**. Existing
  overlap probes are non-scoring. Do not resume M6 just because a historical
  plan calls it the next step.
- A count match is not timing accuracy. The 197,120-instruction mapping is
  demonstrated for one GEMM launch; GPC/SM cycle mapping, unmodeled tails,
  broader evaluation, and publication novelty remain open.
- The custom ISA supports shared load/store, lane-ordered atomic add,
  broadcast/reduction, and bounded barriers. It is not a general memory model.
  Matching barrier position/length/ID and concurrent residency are required.
  Custom-program results now expose per-resource task/work/service/in-flight
  diagnostics in addition to final cycle, strengthening declarative
  architecture studies without turning them into hardware counters. They also
  retain separate issue/completion frontiers, dependency-path and serialized
  resource-work bounds, a conservative lower-bound gap, and the declared
  bottleneck resource; serialized bounds use the final issued beat rather than
  an unused initiation interval, and first-declared resources win ties. These
  values remain model-relative.
- Scalar/profile issue-width overrides are mutually exclusive. The profile
  bridge accepts only its documented fields/defaults; native card identity and
  captured semantics do not change.
- Native direct reports now include trace/activity/native metric projections;
  launch registers/shared-memory totals are captured when present, while the
  runtime L1/shared carveout can be joined from an independent CUPTI resource
  probe. The join now checks CUPTI static/dynamic/register metadata against the
  trace and rejects ambiguous or mismatched records rather than emitting a
  plausible but wrong launch row (the tracer also preserves an explicit CUDA
  carveout preference when one is exposed). Cold-sector ATOM/RED RMW reads are charged by the shared-L2
  DRAM observer and reported separately. Dirty eviction/terminal residency,
  writeback service demand, and LDGSTS destination work remain separate
  observers. The store-only SM86 probe confirms that eventual dirty demand is
  not equivalent to NCU's kernel-interval DRAM-write counter, so physical
  writeback/coherence completion and value-dependent ordering remain unmodeled.
  Native output now separates dirty-eviction demand, terminal dirty residency,
  and eventual-demand interval width rather than collapsing them into a fake
  DRAM-write counter.
  Per-lane first-successor branch events now match NCU divergent/uniform target
  and CBU branch counts on a focused probe; successor lane masks now expose
  minority-path and uniform active-thread populations, while general
  value-dependent CFG coverage remains unavailable. Lane-consistent dynamic
  CFG edges and disjoint path switches are retained; disjoint serialized paths
  are not fabricated into edges.
  Non-memory records with absent active/predicate masks are now counted as
  mask-unknown rather than all-lanes-inactive; branch/CFG/coherence observers
  skip those records and native activity exposes the missing-mask populations.
  NCU scheduler-stall samples are now parsed into an independent map and
  retained on the three-workload holdout, including optional PC/warp-ID/PM
  families when present; they are not collapsed into the simulator's
  exclusive causal readiness counters. Native reports also expose explicitly
  labeled scheduler proxy rows for causal diagnosis.
  Logical-pipeline warp populations retain predicated-off issue records to
  match NCU's instruction convention; FMA is exact on the retained cohort,
  while ALU/LSU mappings remain approximate on RMSNorm/SiLU. Unknown logical-
  pipeline opcode populations and classification coverage are emitted rather
  than silently dropped because NVIDIA documents opcode-pipeline assignment as
  estimated when instructions can use multiple pipelines.
  Independent SM86 NCU physical-pipe counters (ADU/CBU/FP64/Uniform/XU) are
  now retained separately; native rows stay typed unavailable where the trace
  lacks dynamic physical-pipe attribution.
  Trace-local physical-pipe candidate domains now report unique/ambiguous/
  unknown attribution coverage; `EXIT` is conservatively retained in the
  ADU/CBU control domain and `MUFU` in the XU domain, while overlapping
  candidate counts are confidence bounds, never substituted for NCU physical
  counters. FCHK/FSETP/UIMAD/DEPBAR/NOP remain unresolved.
  SM86 occupancy projection includes NVIDIA's documented 1 KiB per-block
  shared-memory system reservation and uses CUPTI's joined carveout capacity
  when available; no runtime carveout is inferred from NCU scored values.
  Decoded memory classes now retain distinct-sector, distinct-line, and
  repeated-sector-alias topology alongside request/sector counts; flat replay
  marks the topology unavailable instead of manufacturing zeros.  See
  `docs/sm86_memory_topology_20260925.md`.
  The rebuilt NVBit producer was re-exercised on fresh one-warp and 168-block
  read/ATOM cases with independent NCU request/sector checks; see
  `docs/sm86_rebuilt_tracer_followup_20260925.md`.
  The producer now resolves accepted launch handles to their exact CUDA
  module/library owner, avoiding same-name guesses.  Fresh GEMM and RMSNorm
  captures resolve to stable hashed executable modules; unresolved
  JIT/deleted/opaque origins remain typed rather than guessed.  See
  `docs/sm86_module_origin_followup_20260926.md`.
  The NCU/HyFiSS profiler crosswalk also keeps canonical unified L1 requests,
  L2 requests, M-stage transactions, throughput, clock, IPC/MIPS, and duration
  observations separate from replay. Rust/Python ingestion also preserves
  GA102 `dram__bytes_read.sum.per_second` and
  `dram__throughput.avg.pct_of_peak_sustained_elapsed` labels; see
  `docs/nvprof_hyfiss_crosswalk_20260926.md`.
  A fresh three-workload NCU profiler crosswalk is retained in
  `docs/sm86_profiler_crosswalk_holdout_20260926.csv`; these values are labels,
  never replay inputs.
  Native metrics now also expose read-only modeled DRAM byte-rate and
  card-peak percentage estimates when the replay has a DRAM observer and GPC
  frontier.  They carry no NCU metric key and do not replace the unavailable
  total `dram__bytes.sum.per_second` row; see
  `docs/sm86_read_bandwidth_projection_20260926.md`.
  The CUPTI launch-resource join now prefers an exact native symbol before
  stripping an NVBit instance suffix, so legitimate names ending in
  `___<digits>` remain joinable.  The Rust NCU CSV parser rejects unterminated
  quoted fields instead of silently accepting malformed observations.
  Immutable memory/constant PC tables in the issue hot loop now use sorted
  compact slices instead of randomized hash lookups; the retained GEMM event
  stream is byte-equivalent and replay is about 1--2% faster in the bounded
  three-run probe.

## Practical continuation facts

- Cargo package `sim-run` is in `sim/direct_run`; its library is
  `sim_direct_run`. Package `sim-run-campaign` is in `sim/run`; its library is
  `sim_run`. There is no package named `sim-direct-run`.
- Use `--manifest-path sim/Cargo.toml` from the repository root; use
  `--manifest-path Cargo.toml` or no manifest flag from `sim/`.
- L1/L1-L2 replay requires `--pchase-card`; flat replay does not.
- Machine-local captures and build outputs are ignored by Git. Dated reports
  identify their local paths; a fresh checkout does not contain those captures.
- The 2026-09-24 host-profile driver/UUID/clocks are dated observations, not a
  claim about the machine forever. Recheck hardware identity for new claims.
- Current changes were made on `GPUSim_main`; inspect the branch rather than
  assuming its name is `alpha`.
- `.claude/knowledge/` is not present in this checkout. Optional old card links
  must not cause repeated failed reads; use tracked docs, source, and
  `references/`. SOL/campaign material remains intact.

[Historical records](archive/README.md) retain older findings and measurements.
Do not interpret their workflow language or earlier blanket “complete” messages
as current scientific evidence. This note replaces stale continuation prose,
not Git history, session transcripts, or raw experiment data.
