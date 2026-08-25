# Completed milestone: sm_86 memory-retirement service boundary

**Status:** complete 2026-08-18; work diagnostics promoted, service timing inactive  
**Architecture:** `sm_86` / RTX 3090 Ti  
**Predecessor:**
[launch-cycle, completion, and range semantics](sm86_launch_cycle_semantics_milestone.md)  
**Evidence:**
`results/captures/sm86_memory_retirement_service_20260818_v1/analysis/README.md`

## Goal

Determine whether the missing kernel-retirement time can be represented by a
fixed load/store/RED completion delay, and identify the smallest trace-derived
work unit needed by a future service model. The experiment must transfer across
tail depth, active lanes, coalescing, same-word RED aliases, address reuse,
one-versus-84 CTA footprint, and explicit device/system fences.

## Why a real retirement mechanism is required

The CUDA Programming Guide states that task boundaries can implicitly flush
writes to satisfy downstream synchronization. Nsight Compute ends a profiled
workload with `WaitForIdle` and a semaphore timestamp. Consequently, final
memory work can remain relevant to launch completion after its issuing
instruction. This motivates a real completion mechanism rather than a fitted
kernel-end constant. The separately reduced GPC and SM `.max` metrics used
below do not, by themselves, locate that work on a post-active timeline.

Primary vendor references:

- [CUDA memory synchronization domains](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/memory-sync-domains.html)
  describes implicit task-boundary flushes and fence interference.
- [Nsight Compute workload durations](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#workload-durations)
  documents the profiler's start/end WaitForIdle envelope.

## Probe and matrix

`workloads/research_probes/memory_retirement_service` uses one warp per CTA and
256 common dependent IMAD iterations, followed by 1/4/16/64 tail operations.
The tail is compile-time specialized as control, discarded load, store, or
non-returning RED. Runtime factors are:

- 1 or 84 CTAs;
- 1, 8, or 32 active lanes;
- 128-byte-per-lane line streams, one-line word streams, same-word RED/load
  streams, or repeated-line reuse; and
- no fence, `MEMBAR.SC.GPU`, or `MEMBAR.SC.SYS`.

Device and system fence variants include ptxas's exact `ERRBAR; CCTL.IVALL`
epilogue. Matched control-fence kernels distinguish that epilogue from memory
work.

## Complete evidence

- 102 cells and 306 correct executions;
- 348 NCU profiles after expanding 11 variable cells to five repeats and ten
  persistent phase cells to seven repeats;
- 12 SourceCounters profiles;
- 102 NVBit captures with 19,385,695 records;
- exact total/reported trace cardinality and empty unsupported, unmapped, and
  cross-allocation sidecars in every capture;
- 12 spill-free kernel specializations, 960 decoded instructions, and zero
  unknown encodings; and
- 102 ordinary plus 102 carded-DRAM byte-identical replay pairs.

Maximum final GPC/SM-active CV is 7.229%/6.506%. The remaining high-CV cells
are the deep B84 line-stream and system-fence phase cells; their multimodality
is retained rather than averaged away.

## Fixed extrema-difference corrections are rejected

Constants trained only on B1 depth-1/4 full-lane line streams were evaluated
on every other CTA, depth, lane, layout, alias, and reuse cell:

| tail | trained correction | held-out zero-correction MAE | fixed-correction MAE | max error | held-out range |
|---|---:|---:|---:|---:|---:|
| load | 125.5 | 165.9 | 93.7 | 362.5 | -38…488 |
| store | 431.5 | 219.4 | 352.1 | 1,069.5 | 44…1,501 |
| RED | 396.0 | 389.5 | 477.0 | 3,992.0 | 83…4,388 |

These values use a matched `gpc__cycles_elapsed.max -
sm__cycles_active.max` difference. Because each maximum can come from a
different hardware instance, this is not a temporal drain interval. The
store/RED constants are worse than a zero correction. Load improves in
aggregate but has a negative-to-488-cycle range and does not justify an added
residual beyond the already modeled response event.

Depth exposes the failure. The B1 store/RED extrema difference falls near
176/158 cycles after depth four, while B84 full-line streams are nonlinear:
depth-16 reaches 1,501/4,388 cycles and depth-64 changes to 44/2,223. This is
consistent with service overlap, but does not temporally prove it. In either
interpretation, a transferable kernel-end correction is rejected.

## Topology result and schema 6

Replay now records `GlobalMemoryIssueDiagnostics` in readiness schema 6:

- active scalar-global load/store/ATOMG/RED warp instructions;
- exact alias-aware sectors;
- trace-derived **ideal** L1TEX wavefronts;
- summed first/last issue timestamps and total/maximum per-SM issue spans.

Ordinary traffic packs four sectors per ideal wavefront. ATOMG/RED uses the
maximum of that packed count and exact same-sector alias multiplicity. This
matches all instruction/request and sector counts in the 90 memory cells.

The distinction between ideal and output wavefronts is essential. All ordinary
cells match NCU output except B84 depth-64 line-stream store, which emits 1,391
additional output wavefronts from internal replay/conflict. Lane addresses
cannot predict that dynamic replay. `MEMBAR.SC.SYS` also adds one hidden store
request per CTA through its compiler epilogue: +1 at B1 and +84 at B84.

Schema 6 is diagnostic only. It changes no issue timestamp, event, cache
counter, or terminal lower bound. Carded peak DRAM bandwidth changes none of
the 102 timing results, so the missing behavior is not the existing aggregate
DRAM server.

## Fence and SourceCounters controls

Device fences change both store/RED GPC elapsed and SM-active work. This is
consistent with outstanding work moving into active execution, but the
extrema subtraction cannot locate it in time. System fences add much broader
work and are not interchangeable with the implicit kernel boundary.
SourceCounters assigns device-fence membar samples predominantly to
the following `ERRBAR` (84 store and 168 RED samples at B84), while system
cells can sample `MEMBAR.SC.SYS`, `ERRBAR`, or `CCTL.IVALL`. As with earlier
dependency sampling, this is active-PC attribution rather than proof that the
sampled instruction alone created the latency.

The matched GPC and SM-active metrics are complementary launch-level
observables, not a direct post-active timeline. NCU's multi-pass application
replay also means their rows can come from different executions, so their
association is interpreted across replicated cells rather than as
co-temporal mediation.

## Promotion decision

Promoted:

- exact scalar-global instruction and sector topology;
- ideal-wavefront projection with atomic alias preservation;
- bounded issue-span diagnostics; and
- readiness schema 6 serialization and checked aggregation.

Not promoted:

- a fixed load, store, RED, fence, or kernel-end delay;
- a per-SM-only or GPU-global queue inferred from one saturated cell;
- output-wavefront replay inferred from ideal addresses;
- RED many-region scheduling (still the atomic-network gap); or
- complete retirement status for store/RED launches.

The next bounded target was the completed
[L1TEX arrival/output-replay service milestone](sm86_l1tex_arrival_replay_service_milestone.md).
It directly varied concurrent warps per SM while keeping ideal wavefront work
fixed. It promotes exact unique-line and atomic-request work, but rejects
work-only and output-replay timing. Its completed
[admission/backpressure successor](sm86_l1tex_admission_backpressure_milestone.md)
then adds exact arrival sequencing, CLOCK/fence separation, reuse, and compiler
operand-read attribution; completion timing remains inactive.

No privileged command was required.
