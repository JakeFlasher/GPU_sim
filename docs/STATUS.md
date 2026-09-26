# Current status

Updated 2026-09-26. This is the current implementation/evidence summary, not a
claim that every future research task or a SOTA submission is complete.

## Implemented paths

| Path | What is runnable | Boundary |
| --- | --- | --- |
| Native SM86 | NVBit/SASS trace -> `sim-trace` -> SM/cache/memory/event replay -> `sim-run` JSON | Fixed instructions, addresses, predicates, and participants; not value-dependent trace regeneration |
| Trace-safe resource profile | `--architecture-spec` or `--hypothetical-issue-ports` changes issue width in ordinary/prepared replay | Hypothetical override, not a new vendor card or ISA |
| Custom `simt_u32_v1` | `custom-program-run` schedules bounded resident groups through `sim-engine::pipeline` | Fixed functional ISA and resource model; not native SASS or `sim-mem` cache timing |
| Custom `simt_its_v1` | Per-lane PCs, predicate branches, BSSY/BSYNC/BREAK/EXIT, stores | Single-group functional example; no timing/compiler claim |

Custom resources have operation bindings, width, latency, initiation interval,
and in-flight limits. Register/memory dependencies permit independent overlap.
Global/shared address spaces, deterministic shared atomic add, lane-0 broadcast,
u32 reduction, and bounded whole-CTA barriers are implemented. See
[specs](../arch_profiles/specs/README.md) and
[programs](../arch_profiles/programs/README.md) for precise limits.

Barriers require equal program lengths, matching IDs at matching instruction
positions, one use per ID, and all participating groups resident. Same-phase
shared conflicts and all inter-group global read/write conflicts reject.
Initial global/shared words are visible to all groups; shared final state is
common, while final global maps remain per-group. This is not a general GPU
memory-consistency model.
Custom JSON results now include per-resource task/work/service/peak-in-flight
diagnostics and model-relative utilization for architecture-design studies.
They also retain separate issue/completion frontiers, an explicit dependency
path bound, a serialized resource-work bound, a conservative lower-bound gap,
and the declared bottleneck resource; these are model-relative schedule
diagnostics rather than hardware counters.

Direct replay now consumes carded scheduler resources; resource-less legacy
fixtures retain an explicit fallback. Issue-width overrides disable independent
dual-issue widening. The ITS executor rejects unarmed `BSYNC`.

Native direct reports now also carry observation-only `trace_metrics`,
`sm_activity`, and unit-explicit `native_metrics`: instruction/predicate and
memory topology, warp-lifetime occupancy/IPC projections, per-pipeline and
causal-readiness counts, L1/L2/DRAM observer counters, captured launch
resources/theoretical occupancy, and typed unavailable rows for
eligible-warps/scoreboard samples, branch-target divergence, writeback/bandwidth,
power/energy, and composite throughput. NCU values remain independent and are
never replay inputs. See [metric expansion](sm86_metric_expansion_20260924.md) and the
[paired comparison CSV](sm86_metric_comparison_20260924.csv).
The metric surface now also emits NCU-shaped block/grid geometry, SM86 register
and shared allocation projections, register/shared/warp/block occupancy limits,
SMSP instruction averages, classified-pipeline coverage, unclassified opcode
population, and causal scheduler proxy rows. These are labeled model-relative
or approximate; they do not claim to recreate NCU's sampled warp-state
counters.
Physical-pipeline candidate domains now quantify unique, ambiguous, and
unknown attribution instead of collapsing this boundary into a single scalar;
`EXIT` is retained in the conservative ADU/CBU control domain rather than
being silently unknown, and `MUFU` is retained in the dedicated XU candidate
domain. Remaining FCHK/FSETP/UIMAD/DEPBAR/NOP assignments stay unresolved.
Branch target lane masks now expose minority-path and uniform active-thread
populations as trace-derived control-flow diagnostics.
Legacy non-memory records that omit lane masks are no longer interpreted as
all-lanes-inactive: native activity exposes missing-mask instruction counts,
and CFG/branch/coherence observers skip those unknown-lane records.
The trace also emits lane-consistent dynamic CFG edges and disjoint-path
switch counts; it does not claim value-dependent CFG regeneration.
The shared-L2 path also charges cold-sector ATOM/RED read-modify-write demand
and emits separate `atomic_rmw_read_sectors`, `reduction_rmw_read_sectors`,
and terminal-dirty-residency rows; value ordering, physical writeback, and
coherence remain typed completion tails.
The NVBit protobuf now also preserves an explicit CUDA shared-memory carveout
preference when a launch/function exposes one; no preference remains absent.
Decoded memory rows now also separate distinct sectors, distinct 128-byte
lines, and repeated sector-span aliases from the ordinary request/sector
projection.  These are trace-topology rows, not cache/protocol counters; see
[memory topology](sm86_memory_topology_20260925.md).
The NVBit producer/toolchain blocker is resolved with the pinned CUDA 13.2,
GCC 14, and protobuf 34 environment; see the [build record](nvbit_build_blocker_resolution_20260925.md).
Accepted-launch provenance now resolves the exact module/library owner of the
launch handle instead of using same-name guesses.  Fresh GEMM and RMSNorm
captures resolve to hashed regular-file modules; JIT, deleted, opaque, and
graph-only origins remain explicitly unavailable.  See the [module-origin
follow-up](sm86_module_origin_followup_20260926.md).
The NCU/HyFiSS profiler crosswalk is recorded separately; Rust and Python NCU
surfaces now retain the missing throughput, clock, unified-request, and
transaction labels, including GA102 DRAM read-rate and DRAM-throughput
percentage rows, without feeding them into replay.
The fresh profiler crosswalk holdout records those observations for GEMM,
RMSNorm, and SiLU in `sm86_profiler_crosswalk_holdout_20260926.csv`; native
throughput rows remain typed unavailable rather than inferred from NCU values.
Native replay additionally exposes read-only modeled byte-rate and peak-rate
diagnostics when a DRAM observer and GPC frontier exist; total
`dram__bytes.sum.per_second` remains unavailable.  See the [read-bandwidth
projection](sm86_read_bandwidth_projection_20260926.md).
The CUPTI resource join prefers exact native symbols before instance-suffix
fallback, and malformed unterminated NCU CSV quotes now fail closed.
The GA102 NCU crosswalk also round-trips direct DRAM read-rate and throughput
percentage observations; see [the capture record](sm86_dram_rate_capture_20260926.md).

## Evidence and its limits

| Evidence | Observation | What it does not establish |
| --- | --- | --- |
| [2026-09-24 NCU/NVBit pair](sm86_measurement_boundary_20260924.md) | Three NCU `smsp__inst_executed.sum` rows and simulator `instructions_issued` all equal 197,120 | Elapsed-cycle accuracy, general workload accuracy, or a speed/accuracy frontier |
| [Timing crosswalk](sm86_measurement_boundary_20260924.md) | NCU GPC 13,475–13,529; raw BindingCycles 11,102; model GPC frontier 12,949 | Complete retirement/coherence/writeback timing |
| [CUPTI launch-resource join](sm86_metric_expansion_20260924.md) | Driver-selected shared-memory configuration 32 KiB, occupancy shared-memory limit 18 blocks/SM, and driver occupancy limit 16 blocks/SM for fresh GEMM; mismatched trace/CUPTI resource fields now fail closed | Complete cache-carveout/coherence timing or NCU independence |
| [SM86 metric-domain probe](sm86_metric_domain_experiment_20260924.csv) | 100 small/168-block rows: instruction, load, ATOM, and RED sector counts are exact; cold-RMW read demand is within 1.8--23.6% of scaled NCU data-read totals | Launch/flush baseline, writeback scheduling, or complete RED/ATOM ordering |
| [SM86 writeback probe](sm86_writeback_experiment_20260924.csv) | Store requests/sectors exact; eventual dirty-demand bound is 8.0% above NCU writes for 16 MiB and 181.0% above for 4 MiB | Kernel-interval DRAM writes, terminal flush visibility, and coherence |
| [SM86 coherence domain](sm86_coherence_domain_20260925.md) | Bounded trace-only cross-SM global-line/read-write/write-write sharing domain plus exact 32-byte sector-overlap/read-write/write-write subdomains, with one-million-line fail-closed arena | Invalidation ordering, coherence protocol traffic, and physical writeback |
| [SM86 memory topology](sm86_memory_topology_20260925.md) | Distinct 32-byte sectors, 128-byte lines, and repeated span aliases retained for the three-workload holdout and small ATOM probe | Cache allocation/replay, protocol traffic, and temporal memory ordering |
| [Rebuilt tracer follow-up](sm86_rebuilt_tracer_followup_20260925.md) | Pinned NVBit producer rebuild plus fresh read/ATOM traces; instruction/request/sector counts match NCU on one small read, one small ATOM, and a 168-block ATOM launch; 304-column profiler crosswalk round-tripped | This is a small-domain regression, not general elapsed-time or writeback accuracy |
| [Module-origin follow-up](sm86_module_origin_followup_20260926.md) | Same-name module/library handles, unload/reload, context-independent libraries, and anonymous cubin produce exact hashed owners or typed unavailable rows | JIT/graph/opaque paths without a stable regular file remain unavailable |
| [Read-bandwidth projection](sm86_read_bandwidth_projection_20260926.md) | Read-only DRAM demand rate and card-peak percentage are emitted as approximate model diagnostics; total DRAM throughput remains typed unavailable | Does not model writeback/coherence or NCU's kernel/flush interval |
| [SM86 trace-structure holdout](sm86_trace_structure_holdout_20260925.csv) | GEMM/RMSNorm/SiLU CFG-edge, cross-SM sharing, and physical-pipeline-confidence summaries retained together | No value-dependent CFG regeneration or dynamic physical-pipe counter claim |
| [SM86 branch-domain probe](sm86_branch_domain_experiment_20260925.csv) | Per-lane successor observer matches NCU divergent targets, uniform targets, and CBU branch-instruction counts (8/49/74, 8/49/74, 0/9/10) | Per-thread target populations and broader value-dependent CFG coverage |
| [SM86 branch holdout](sm86_branch_holdout_20260925.csv) | GEMM, RMSNorm, and SiLU branch/CBU counts all match NCU (0 divergent targets; exact uniform and control counts) | Divergent control-flow holdout beyond these workloads |
| [SM86 scheduler-stall holdout](sm86_scheduler_stall_holdout_20260925.csv) | 17 available NCU scheduler-sample families retained for GEMM/RMSNorm/SiLU; optional PC/warp-ID/PM families parse separately; simulator causal readiness and proxy rows remain separate | Hardware-equivalent eligible/overlapping stall reconstruction |
| [SM86 full metric holdout](sm86_full_metric_holdout_20260925.csv) | 51 original pipeline/memory/L2/DRAM/branch rows across GEMM/RMSNorm/SiLU; expanded output also includes launch/occupancy and scheduler-proxy rows; request/sector/CBU/FMA counts are exact, read demand is within 1.2--2.5%; SHF/SHFL and control-pipe classification is now explicit | Complete writeback, overlapping scheduler samples, and dynamic physical-pipe assignment |
| [SM86 physical-pipeline probe](sm86_physical_pipeline_probe_20260925.csv) | Independent NCU GA102 physical-pipe rows retained; ALU/FMA/Tensor are covered by logical projections on the GEMM cohort, while ADU/CBU/FP64/Uniform/XU remain explicit native attribution gaps | Dynamic physical-pipe assignment from fixed NVBit records |
| [2026-09-23 speed probe](sm86_speed_probe_20260923.md) | Event replay faster than dense on one GEMM; static-PC optimization reduces the core replay path, while the current confidence-domain surface runs about 746 ms replay | Universal speedup or independent hardware correctness |
| [Issue-width study](sm86_issue_width_design_study_20260923.md) | Widths 2/4 change model frontiers but are slower in the retained study | Hardware speedup; retain the negative result |
| [Feature inventory](sm86_feature_coverage_20260923.md) | 18 rows; explicit producer gaps and typed replay tails | Arbitrary opcode support; older cohorts retain unresolved static origins |
| [Custom pipeline](custom_program_pipeline_20260924.md) / [barrier](custom_group_barrier_20260924.md) | Two-group cycles 45; producer/consumer cycles 90 and value 100 | A validated commercial GPU or arbitrary user-defined ISA |

The older approximately 9.7% maximum is a development counter diagnostic,
not held-out elapsed-time MAPE. Python/Rust differentials and focused tests are
implementation evidence against the declared model, not independent silicon
validation. Dated wall times refer to the recorded builds and hosts, not every
subsequent revision.

## Remaining work versus deliberate exclusions

- **Open research:** cycle-observable mapping and missing modeled tails;
  broader static/JIT provenance coverage; broader independent accuracy/speed evaluation; a
  consequential architecture study with a demonstrated novel mechanism.
- **Deferred by request:** fair numerical Accel-Sim/MGPUSim comparison. Existing
  [baseline cells](baseline_matrix_20260924.json) are non-scoring; executable
  overlap alone does not make them comparable.
- **Unsupported in the custom slice:** arbitrary instruction definitions,
  compiler/ABI generation, native trace regeneration, partial barriers,
  inter-group atomic ordering, weak memory, and general collectives. These are
  not automatically queued implementation obligations.
- **Outside scope:** RTL/Verilator/FPGA, physical design, internal-silicon
  reconstruction, and unavailable-AMD validation. SM120 work needs matching
  local hardware/inputs. Universal binary decoding and silicon-equivalent
  timing are not academic simulator prerequisites.

NCU labels remain independent. Issue, modeled-event, completion, launch, and
unmodeled-tail frontiers stay separate. Available profiling can support useful
cycle prediction without exposing proprietary internals; no exact-equivalence
promise follows.

Use the [roadmap](../plans/cross_architecture_gpu_simulator_research_plan_20260923.md)
for future experiments and [project memory](PROJECT_MEMORY.md) to resume local
work. [Historical records](archive/README.md) and SOL/campaign trees remain
available without becoming active review or approval machinery.
