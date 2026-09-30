# SM86 metric expansion — 2026-09-24

## Implemented slice

The native SM86 replay now exposes three observation-only groups in `sim-run`:

1. `trace_metrics`: dynamic warp/thread predicate populations, opcode-head
   instruction mix, and memory-class request/sector counts.
2. `sm_activity`: per-SM active-warp integral, SM-active interval, derived
   achieved-occupancy ratio, per-SMSP instruction counts, issue-active observations,
   and warp admission/retirement intervals.
3. `native_metrics`: unit-explicit ratios for achieved occupancy, active-warp
   and IPC projections, instruction/branch populations, memory request and
   sector/coalescing counts, captured launch resources and theoretical
   register/shared-memory occupancy limits, L2/DRAM traffic, and typed unavailable rows for
   power, writeback traffic, and composite NCU throughput metrics that the
   native model cannot yet define.

The shared-L2 event path now also charges the cold-sector read arm of returning
`ATOM` and non-returning `RED/REDG` operations through the carded DRAM server.
The report exposes `atomic_rmw_read_sectors` and
`reduction_rmw_read_sectors` separately from ordinary read demand. This is a
   traffic/timing observer only: value-dependent atomic serialization,
   reduction ordering, dirty writeback, and inter-SM coherence remain explicit
   completion tails.

Native metrics now distinguish `dirty_writeback_demand_sectors` (an
eventual-demand topology bound) from `dram_write_sectors` (still unavailable
as a profiler-interval prediction), and expose a separate terminal-flush
service estimate with the same caveat.
The writeback surface now separates observed dirty-eviction demand, terminal
dirty-residency demand, and the resulting eventual-demand interval width. This
is a bounded memory-completion statement, not a claim that either endpoint is
the profiler's kernel-interval DRAM-write counter.

The branch-domain follow-up now resolves per-lane first successors from the
trace's active masks, function identity, and PCs. On two divergent and one
uniform SM86 modes, divergent target events, uniform target events, and the
broader CBU branch-instruction count all match NCU (8/49/74, 8/49/74, and
0/9/10 respectively). The native per-thread target-population metric remains
separate; the observer is not promoted to a general CFG or value-dependent
control-flow model.

The same observer was then applied without retuning to the retained GEMM,
RMSNorm, and SiLU traces. Branch-control totals and uniform-target totals
matched independent NCU rows exactly (4,096/3,584; 23,552/22,016; and
40,960/16,384), with zero divergent targets in all three workloads. The
holdout is recorded in `sm86_branch_holdout_20260925.csv`.

The retained workload holdout also collects the available NCU scheduler-stall
sample family (`barrier`, `branch_resolving`, `dispatch_stall`, `drain`,
`imc_miss`, throttle categories, scoreboard, selected/not-selected, and
wait). These 17 categories are exposed under `ncu_observation.scheduler_stalls`
and recorded in `sm86_scheduler_stall_holdout_20260925.csv`. They remain
comparison targets only: NCU documents them as cumulative warp samples, while
the simulator reports exclusive causal readiness buckets, so the two domains
are not numerically substituted for one another.

The full metric holdout (`sm86_full_metric_holdout_20260925.csv`) extends this
to pipeline, global request/sector, shared-wavefront, L2, DRAM, and branch
rows. GEMM/RMSNorm/SiLU global request/sector and CBU branch counts are exact;
shared-wavefront and L2 rows are close observer projections, and modeled DRAM
read demand is 2.5%, 1.5%, and 1.2% below the fresh independent NCU rows.
After correcting the logical-pipeline population to retain predicated-off
warp instructions (matching NCU's warp-level instruction convention), FMA
counts become exact on all three workloads. ALU and LSU remain architecture-
mapping gaps on RMSNorm/SiLU and are retained as approximate rather than
being fitted to those rows.

These fields are derived only from the NVBit trace, control sidecar, and the
declared model. They never consume NCU values and never affect scheduling,
cache state, or architecture parameters.

The NCU observation parser also now retains:

* GPC maximum/average cycles and SM-active cycles;
* achieved active warps per SM, achieved occupancy, and SMSP active warps;
* launch register/shared-memory values and NCU's occupancy-limit breakdown;
* L1 global-load lookup hits/misses and L1/L2 sector hit rates;
* global request/sector classes, shared wavefronts, pipeline instruction
  counters, L2 TEX-source sectors/hits, and DRAM read/write bytes/sectors;
* NVIDIA logical-pipeline projections: GEMM FMA (17,920), ALU (21,504), and
  Tensor (8,192) populations match the expanded NCU capture; LSU remains
  explicitly approximate because NCU includes additional special/control work;
* model-relative asynchronous-copy destination group/wavefront work, kept
  separate from ordinary shared-store wavefronts;
* DRAM read sectors;
* the existing instruction, traffic, and scoreboard observations.

## Fresh independent NCU captures

Workloads: `gemm_hmma_bf16_sm80`, `rmsnorm_fwd_bf16_sm80`, and
`silu_fwd_bf16_sm80`, each with the timing shape, warmup 1, and reps 1. The
paired simulator traces are the existing warmup-0/reps-1 NVBit captures for
the same binaries and launch shapes. The new expanded captures are retained
under `results/captures/sm86_metrics_expansion_20260924_v1/ncu/` (GEMM) and
`results/captures/sm86_metrics_expansion_20260924_v2/ncu/` (RMSNorm/SiLU).
An additional GEMM parser/traffic smoke capture with request, wavefront,
pipeline, L2-byte, and DRAM-byte metrics is retained under
`results/captures/sm86_metrics_expansion_20260924_v3/ncu/`; its repeated NCU
values are kept out of the v1/v2 cohort table so replay-to-replay variation is
not mixed into one row.
The rebuilt tracer schema was exercised on fresh GEMM captures under
`results/captures/sm86_carveout_capture_20260924_v1/` and `_v3/`; the launch had no
explicit carveout preference, so the new optional field is absent as designed.
The latest `_v3` trace also records the CUDA-driver occupancy limit of 16
blocks/SM.

The absence is now resolvable without NCU leakage: the retained CUPTI
resource-only probe reports `sharedMemoryExecuted=32768` bytes for the same
GEMM launch. Joining `results/captures/sm86_carveout_capture_20260924_v3/cupti_resources.jsonl`
with `--launch-resources` emits `external_shared_memory_config_bytes` with
status `observed_external`. CUPTI distinguishes the requested carveout from
the driver-selected configuration; CUDA documents the carveout itself as only
a hint ([CUDA carveout documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html),
[CUPTI kernel activity fields](https://docs.nvidia.com/cupti/api/structCUpti__ActivityKernel9.html)).

## Adversarial review replay — 2026-09-25

The current release simulator was replayed against the retained GEMM, RMSNorm,
and SiLU traces and independent branch CSVs after the final review fixes. All
three instruction totals, CBU branch totals, uniform-target totals, and zero
divergent-target totals remained equal to the NCU rows (197,120/4,096/3,584,
264,704/23,552/22,016, and 417,792/40,960/16,384 respectively). A forged CUPTI
record with a one-register mismatch is now rejected before report emission;
the matching 32 KiB capture still joins as `observed_external`. The replay
used simulator binary SHA-256
`138e6a47436da00ecadfcee6c9aa3af19ac3a0f0aabe9b04cb6ec1e5eb25933e`.

The review also corrected NCU scaled-integer parsing (`Kbyte/block`, `Kbyte`,
and `block/SM`), made branch-target rows use the NCU `branches` unit, bounded
the successor observer, and rejected malformed launch metadata before payload
join. These are contract/integrity fixes; they do not fit simulator outputs to
NCU values.

## Metric-semantic follow-up

Nsight Compute distinguishes cumulative counters from sampled warp-state
families: scheduler sampling selects a random active warp at a hardware-defined
interval, and `selected`/`not_selected` are scheduler states rather than true
stalls ([Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)).
Its opcode-pipeline source metric is explicitly estimated when instructions can
execute through multiple pipelines. The simulator therefore retains optional
`pcsamp`, `warpidsamp`, and PM sampling families separately, emits causal
`scheduler_proxy_*` rows without relabeling them as NCU samples, and reports
`ncu_pipeline_classified_pct` plus unclassified opcode instructions. This
resolves the schema/coverage gap while preserving uncertainty; it does not
manufacture a hardware sample from model cycles.

The same expansion adds NCU-shaped `block_size`, `grid_size`, register/shared
allocation projections, occupancy-limit components, `warp_instructions_avg`,
SM-active/GPC frontier aliases, and a classified-pipeline coverage percentage.
Allocation and occupancy rows remain model-relative because legacy traces may
not carry the runtime carveout or driver-selected per-block resource values.
The SM86 occupancy projection now includes NVIDIA's documented 1 KiB per-block
system shared-memory reservation and uses a joined CUPTI carveout capacity when
one is supplied; absent that join it keeps the card maximum as a clearly
approximate capacity boundary ([Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)).

An independent SM86 physical-pipeline probe is retained in
`sm86_physical_pipeline_probe_20260925.csv`. It records NCU GA102 ADU/CBU,
ALU/FMA/FP64/LSU/Tensor/Uniform/XU counters. The parser now preserves these
families without conflating them with logical pipeline counters; native ADU,
CBU, FP64, Uniform, and XU rows remain typed unavailable because NVBit records
static SASS identity and execution masks, not the dynamic physical pipe chosen
for each instruction.

The trace also emits conservative candidate domains: on the retained GEMM,
65.19% of warp instructions have one candidate physical pipe, 30.39% have
multiple candidates, and 4.42% have no candidate domain. The bounded update
places `EXIT` in the ADU/CBU control candidate domain, consistent with its
control-instruction identity while preserving dynamic assignment uncertainty.
Candidate membership may overlap (for example ADU/CBU branch candidates), so
these rows are confidence bounds rather than reconstructed NCU counters.

The logical opcode projection now distinguishes Ampere `SHF` (funnel shift)
from `SHFL` (warp shuffle), and keeps branch/barrier heads out of the generic
LSU bucket when the trace identifies them as control operations. This follows
NVIDIA's documented distinction between logical/physical pipes and the SASS
instruction families; ALU/LSU rows remain approximate where dynamic selection
or workload-specific NCU attribution differs.

The confidence domain also recognizes the dedicated `MUFU` special-function
family as an `xu` candidate.  On the retained RMSNorm and SiLU holdouts this
reduces unknown physical-domain populations from 0.77% to 0.39% and from
7.84% to 3.92%, respectively.  `FCHK`, `FSETP`, `UIMAD`, `DEPBAR`, and
`NOP` remain unknown or unresolved rather than being assigned by opcode-name
guessing.

Branch successors now retain lane masks per target. Native reports therefore
include successor-active-thread totals, divergent minority-path populations,
uniform active-thread totals, and unresolved successor lanes. These are useful
for control-flow diagnosis and future value-dependent CFG work, but remain
trace-derived rather than NCU branch counters.

The replay now also reconstructs a lane-consistent dynamic CFG: GEMM exposes
139 unique observed edges and 196,608 transition observations. Consecutive
records with disjoint active masks are counted as serialized path switches and
are not fabricated into edges.

The three-workload structure table (`sm86_trace_structure_holdout_20260925.csv`)
retains these CFG rows beside cross-SM coherence-domain rows and physical-pipe
confidence percentages, making the non-NCU structural evidence reproducible.

Host identity was the dated local RTX 3090 Ti profile: compute capability 8.6,
driver 615.71.09, SM clock 1350 MHz, memory clock 10501 MHz. NCU used
application replay, cache-control all, clock-control none, and the expanded
metric list.

## Model versus NCU observations

| Workload / observable | Simulator | NCU | Difference | Status |
|---|---:|---:|---:|---|
| GEMM / GPC elapsed-cycle maximum | 12,949 frontier | 13,494 | -4.04% | Diagnostic; completion tails remain incomplete |
| GEMM / SM-active cycles (84-SM mean) | 11,030.38 | 11,402.23 | -3.26% | Model-relative active interval |
| GEMM / active warps per SM | 6.0214 | 5.89 | +2.23% | Model activity integral vs independent NCU |
| GEMM / achieved occupancy | 12.5446% | 12.27% | +0.2746 pp | Model activity integral vs independent NCU |
| GEMM / theoretical resource occupancy | 33.33% | 12.27% achieved | separate resource upper bound | Register/shared/block limits, not achieved timing |
| GEMM / L1 unified sector hit | 18.6298% | 19.05% | -0.4202 pp | Same broad observer family |
| GEMM / L2 TEX-source hit | 95.2733% | 95.11% | +0.1633 pp | Close source/roll-up correspondence |
| GEMM / DRAM read sectors | 8,262 | 8,420 | -1.88% | Demand vs hardware observation |
| RMSNorm / GPC elapsed-cycle maximum | 18,944 | 19,123 | -0.94% | Diagnostic; tails remain incomplete |
| RMSNorm / SM-active cycles (84-SM mean) | 13,039.24 | 13,062.64 | -0.18% | Model-relative active interval |
| RMSNorm / active warps per SM | 7.9923 | 7.85 | +1.81% | Model activity integral vs independent NCU |
| RMSNorm / achieved occupancy | 16.6506% | 16.35% | +0.3006 pp | Model activity integral vs independent NCU |
| RMSNorm / theoretical resource occupancy | 33.33% | 16.35% achieved | separate resource upper bound | Register/shared/block limits, not achieved timing |
| RMSNorm / L1 unified sector hit | 37.50% | 36.61% | +0.89 pp | Same broad observer family |
| RMSNorm / L2 TEX-source hit | 66.1458% | 68.86% | -2.71 pp | Source-domain correspondence |
| RMSNorm / DRAM read sectors | 16,677 | 16,872 | -1.16% | Demand vs hardware observation |
| SiLU / GPC elapsed-cycle maximum | 5,642 | 5,815 | -2.98% | Diagnostic; tails remain incomplete |
| SiLU / SM-active cycles (84-SM mean) | 3,010.30 | 3,616.08 | -16.75% | Active-lifetime/launch-wave limitation |
| SiLU / active warps per SM | 42.2249 | 36.94 | +14.31% | Model occupancy limitation |
| SiLU / achieved occupancy | 87.9686% | 76.97% | +10.9986 pp | Model occupancy limitation |
| SiLU / theoretical resource occupancy | 100% | 76.97% achieved | separate resource upper bound | Register/shared/block limits, not achieved timing |
| SiLU / L1 unified sector hit | 23.5718% | 19.82% | +3.75 pp | Cache-state mapping gap remains |
| SiLU / L2 TEX-source hit | 50.00% | 55.04% | -5.04 pp | Source-domain correspondence |
| SiLU / DRAM read sectors | 16,410 | 16,568 | -0.95% | Demand vs hardware observation |
| All three / warp instruction count | exact trace count | exact NCU count | 0% | Strongest current boundary |

The small address-carry canary is intentionally harsher but now has exact
instruction, global-load, ATOM, RED, and aggregate returning-write sectors
after the fresh trace/control recapture. Its remaining DRAM-byte difference is
a completion/flush boundary, not a reason to fit a scalar correction. The
shared-L2 observer attributes one cold ATOM RMW sector and four cold RED RMW
sectors in this launch; the typed tail still prevents treating those reads as
complete value/coherence timing.

The new domain experiment (`sm86_metric_domain_experiment_20260924.csv`) uses
one-warp and 168-block read/ATOM/RED/control kernels before scaling the same
patterns. The 100-row table demonstrates exact instruction, load, ATOM, and
RED-sector counts for every pattern. On the scaled 168-block cohort, the new
cold-RMW observer reduces the data-read gap to 1.8--23.6% (the residual is a
roughly 100-sector control/launch baseline); without that arm RED previously
reported zero modeled reads. DRAM-sector discrepancies on the tiny launches
are intentionally not fitted: Nsight documents that small kernels and
multi-pass/application replay can be affected by asynchronous GPU activity and
replay precision, especially below steady-state durations.[Nsight Compute
replay/precision guidance](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)

The retained domain replay sidecars were regenerated after this RMW update;
their `small.inputs.json`/`large.inputs.json` records bind the simulator
binary, workload binary, architecture cards, and tracer bytes used for the
rows. The NCU CSVs were not used as replay inputs.

The store-only follow-up (`sm86_writeback_experiment_20260924.csv`) is the
writeback boundary probe. Store requests and sectors match NCU exactly. For a
16 MiB stream, the simulator's eventual dirty-demand bound is 528,384 sectors
versus 489,144 NCU DRAM-write sectors (8.0% higher); for a 4 MiB stream the
bound is 132,096 versus 47,004 (181.0% higher). The latter difference shows
that NCU's kernel interval does not flush all terminal dirty L2 residency.
The bound is therefore emitted as an approximate topology metric, while
`dram__sectors_write.sum` remains typed-unavailable. NVIDIA's architecture
documentation describes VRAM traffic as either L2 misses or L2 writeback and
notes that DRAM is half-duplex ([GPU Trace memory-system guide](https://docs.nvidia.com/nsight-graphics/UserGuide/gpu-trace-system-architecture.html)).

NCU's lookup and hit-rate quantities follow the profiler's request/sector
definitions; sectors, requests, wavefronts, and cache-line tags are not
interchangeable. NVIDIA documents these distinctions in the [Nsight Compute
Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).

The native report intentionally retains typed unavailable rows for hardware
eligible-warp populations, overlapping long/short scoreboard warp-cycle
samples, branch-target divergence, DRAM writes/bandwidth, power/energy, and
composite throughput. Those are missing simulator producers, not zero
measurements; launch resources are captured separately when present in the
trace contract.

Launch-resource metadata is preserved from the validated trace root
(registers/thread, block/grid geometry, and static/dynamic/shared-memory
components when the trace contract provides them). The theoretical occupancy
projection uses SM86 allocation rules and card limits; it is kept separate
from achieved occupancy derived from modeled warp lifetimes.
The rebuilt tracer also records the CUDA-driver occupancy limit for the launch
shape; the fresh GEMM trace records 16 blocks/SM, matching the corresponding
NCU limit without using NCU as a replay input.
The runtime L1/shared-memory carveout (`launch__shared_mem_config_size`) is
typed-unavailable in trace-only replay, but can now be supplied by the
independent CUPTI resource-only join; this explains why a trace-only model's
shared-memory resource limit can differ from NCU's launch occupancy-limit
breakdown.
The tracer schema now preserves an exposed CUDA function/launch carveout
preference when one exists; the current GEMM launch had no explicit preference,
so this capture remains a valid absence rather than an inferred default.

Dirty eviction sectors/bytes and terminal dirty residency are reported as
demand topology only. They are not promoted to `dram__sectors_write.sum`
because the replay still lacks the physical writeback scheduling and
post-kernel coherence/flush boundary.
An additional `writeback_service_cycles_estimate` is only the serialized byte
service demand under the documented card bandwidth hypothesis; it is not a
completion-cycle prediction.

Async global-to-shared destination work is a separate observer. It must not be
compared directly with NCU's ordinary shared-store row: Nsight counts L1TEX
access classes, while the replay's destination work belongs to the LDGSTS
completion protocol.

## Remaining timing boundary

The mandatory `gpc__cycles_elapsed.max` descriptor now refuses to emit a
binding-cycle prediction when the replay reports dirty-writeback or coherence
tails. It emits a typed `required_unimplemented` outcome instead of presenting
the lower bound as complete timing. The architecture-scoped GPC frontier remains
available as a separate diagnostic projection. (The frozen descriptor therefore
does not silently turn the current 12,949-cycle GEMM frontier into a hardware
timing claim.)

This is intentional: existing task-done/store probes show that a fixed post-
issue constant cannot be transferred across write footprint and visibility
conditions. Closing the final timing metric requires an explicit SM86 model for
the remaining writeback/coherence/launch envelope, followed by an independent
holdout capture; the current patch does not fit a correction to the GEMM row.

## Reproduction commands

Build:

```bash
cargo build --release --manifest-path sim/Cargo.toml -p sim-run
```

Replay with the expanded trace metrics:

```bash
sim/target/release/sim-run \
  --trace results/captures/sm86_measurement_boundary_20260924_v1/evidence/raw/nvbit/gemm_hmma_bf16_sm80 \
  --control results/captures/sm86_measurement_boundary_20260924_v1/evidence/raw/static/gemm_hmma_bf16_sm80/gemm_hmma_bf16_sm80.control_bits.json \
  --arch-card arch_profiles/cards/sm_86.json \
  --pchase-card data/pchase/sm_86_static_card.json \
  --mode l1-l2 --cache-scope launch --ncu-shaped-metrics \
  --ncu-csv results/captures/sm86_metrics_expansion_20260924_v1/ncu/ncu.csv
```

The next implementation slice should either complete the explicit writeback /
coherence timing model or keep the mandatory GPC descriptor typed-unimplemented.
The SiLU row also shows why achieved occupancy cannot be treated as complete:
the model overestimates occupancy while the GPC frontier is only 2.98% low.
This is a diagnostic target, not a correction fitted from NCU rows.
