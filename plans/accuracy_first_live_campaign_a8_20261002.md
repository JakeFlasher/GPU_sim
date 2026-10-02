# A8 — accuracy-first live-campaign foundation

**Revision:** 2026-10-02
**Branch:** `GPUSim_main`
**Input:** latest second-round review, current SM86 protocol, live-host audit
**Status:** executing before A9; no goal-manager workflow is used.

## Purpose

A8 makes the live experiment path safe and executable before any broader
campaign claim is made. It covers the smallest complete slice:

`workload correctness → NVBit/SASS trace → simulator replay → prediction → NCU observation → evaluation`.

NCU values remain independent observations. They are never inserted into the
prediction, used to choose an implementation, or used to tune the same scored
workload. Simulator wall-clock runtime is reported separately and is not
called compression or hardware speedup.

The supported claim after A8 is only that the declared pipeline can execute a
bounded, reproducible workload and preserve explicit pass, refusal, OOM, OOD,
timeout, and open-interval outcomes. A8 does not by itself establish native
selective dispatch or finite hardware accuracy.

## Research constraints carried forward

1. Traces, controls, architecture cards, and p-chase cards are inputs.
2. NCU is an independent target, not a model feature or calibration label for
   the same workload.
3. Native issue, completion, cache, launch, and typed-tail frontiers remain
   separate; an unresolved frontier stays open or falls back to detailed
   replay.
4. Every workload failure is retained with its typed cause. A smaller shape is
   a new workload, not a silent recovery from OOM/OOD.
5. No authority, receipt, approval, freeze, sealing, provenance, or audit
   workflow is introduced. Run documents are ordinary bounded experiment
   results and do not block unrelated work.

## Stage ledger

| Stage | Focus | Completion evidence |
|---|---|---|
| A8.0 | Plan and contract synchronization | This file, protocol, roster, and runbook agree on scope and claims | **complete** |
| A8.1 | Bounded workload runner | Unit tests cover pass, OOM, OOD, timeout, memory preflight, and launcher failure; output is schema-valid | **complete** |
| A8.2 | Source/binary cleanup | Retired CUPTI flags are absent from current sources and rebuilt binaries; all active rows build from source | **complete** |
| A8.3 | Roster smoke campaign | Every active row runs correctness mode or records a typed, reproducible failure; no silent substitution | **complete** |
| A8.4 | Vertical live slice | One row completes correctness, NVBit trace, simulator replay, prediction-before-NCU, and five NCU repetitions | **active** |
| A8.5 | Cross-artifact validation | Trace, control, simulator, NCU, identity, and evaluation artifacts validate together without NCU leakage |
| A8.6 | A8 handoff | Results, remaining failures, and next A9 work are synchronized in status/memory/history; raw data is preserved |

## A8.1 — bounded workload execution

`tools/run_sm86_workload.py` is the common launcher. It must:

- run the child in a separate process group;
- capture stdout/stderr to files instead of unbounded Python buffers;
- terminate the whole group after a fixed wall-time limit;
- query device free memory before and after a run when a GPU is required;
- reject a run before launch when expected allocation plus the configured
  reserve exceeds currently free memory;
- classify output as `passed`, `oom`, `out_of_domain`, `timeout`,
  `runtime_error`, `preflight_memory_low`, `host_query_error`, or
  `launcher_error`;
- write one small `sm86_workload_run_v1` result document; and
- return a nonzero process status for every non-passing class.

The runner is diagnostic and bounded; it is not a permission or admission
mechanism. The workload itself remains responsible for correctness checks and
shape-specific validation.

## A8.2 — stale interface and source/binary cleanup

The current portable-LCD sources already describe NCU-only timing, while some
ignored prebuilt binaries were compiled from older source and still advertise
retired CUPTI options. The cleanup sequence is:

1. Search tracked source, Makefiles, runbooks, and binary help strings for
   `enable-cupti-activity` and retired live-CUPTI collection paths.
2. Remove only stale tracked source/documentation references; preserve raw
   historical measurements as historical data.
3. Rebuild each active portable-LCD binary from its current source with the
   current CUDA toolchain.
4. Check that help output and runner behavior agree with `runner_common.h`.
5. Run the source contract verifier and SASS/architecture checks where the
   local toolchain supports them.

No binary is treated as current merely because it exists on disk.

## A8.3 — roster smoke campaign

The active roster is the 15 portable-LCD rows in `workloads/portable_lcd/`.
Each row receives:

- correctness shape, fixed seed, one warmup, and two repetitions;
- bounded runner output under `results/alpha/a8_smoke_<timestamp>/`;
- manifest and stdout/stderr inspection; and
- a typed disposition.

Timing-shape or larger dimensions are attempted only after correctness passes.
If a timing shape exceeds the memory budget, it is reported as
`preflight_memory_low` or `oom`; it is not silently replaced by a correctness
shape and called a timing result.

## A8.4 — vertical live slice

The first slice uses `gemm_hmma_bf16_sm80` because it has a small deterministic
correctness shape, an instruction-rich SASS kernel, and an existing simulator
trace/control path.

### Native correctness

Run the rebuilt binary through the bounded runner. Require process success and
the workload's own `correctness_ok=true` result.

### NVBit capture

Run a separate process with the repository NVBit tracer and a fresh trace
directory. Require nonempty protobuf payloads, launch metadata, instruction
statistics, and a clean tracer termination. NCU and NVBit are separate runs;
their matching fields are workload, binary, arguments, device, and launch
shape—not profiler output.

### Simulator replay

Run `sim/target/release/sim-run` with the captured trace, control sidecar,
SM86 architecture card, and p-chase card. Record detailed replay termination,
record counts, frontiers, typed tails, and any exact-fallback reason.

### Prediction before observation

Write a prediction artifact from simulator output before launching NCU. The
prediction must contain no NCU value and must bind the declared metric,
endpoint, cache policy, clock policy, launch identity, and source paths.

### NCU observation

Collect five fresh application-replay repetitions with:

- `gpc__cycles_elapsed.max` as the target;
- mechanism counters `gpu__time_duration`, `sm__cycles_active`, and
  `sm__inst_executed.sum`;
- `--launch-skip 0`;
- `--replay-mode application`;
- `--cache-control all`;
- `--clock-control none` when the host clock contract is unchanged; and
- the independently captured device identity.

Nsight Compute documents that application replay requires deterministic
execution, cache flushing improves repeatability, and compared runs should
keep capture settings stable ([CLI documentation](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html),
[Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)).

## A8.5 — validation and interpretation

Validation must be performed in this order:

1. workload result status and correctness output;
2. NVBit output structure and trace invariants;
3. simulator replay and frontier consistency;
4. prediction schema and absence of NCU fields;
5. independent device/tool identity;
6. NCU repetition termination and dispersion;
7. post-observation join and split evaluation.

Report separately:

- finite interval coverage;
- open-upper and lower-only rows;
- repetition median/IQR/all samples;
- mechanism-counter values;
- OOM/OOD/timeout/refusal counts; and
- simulator wall-clock and memory costs.

An open upper is not a failed numeric prediction; it is not finite interval
coverage. A passing self-authored test is implementation evidence, not an
independent hardware measurement.

## A8.6 — completion bar and handoff

A8 is complete only when:

- the plan and runbook are updated;
- the bounded-runner tests and schema checks pass;
- all active rows have a smoke disposition;
- one row completes the full live vertical slice;
- raw NVBit/NCU/simulator outputs remain available;
- no prediction consumed NCU values;
- stale CUPTI interfaces are removed from current artifacts; and
- remaining native-witness, finite-accuracy, OOM/OOD, and workload failures
  are listed explicitly for A9.

If any item fails, A8 remains open and the failure is fixed or recorded before
moving to A9.

## Milestone log

### A8.0 — plan and contract synchronization — complete

The staged program is recorded here. The existing protocol keeps NCU separate
from predictions, and the portable-LCD roster identifies the 15 active SM86
workloads. The live host has an RTX 3090 Ti (SM86), CUDA 13.4, NCU 2026.1,
and a built NVBit tracer. No workload data was changed in this milestone.

### A8.1 — bounded workload runner — complete

Implemented `tools/run_sm86_workload.py`, its schema, and focused tests. The
tests pass (`4 passed`), the protocol/preflight regression tests also pass
(`16 passed` in the combined run), and the new JSON Schema validates under
Draft 2020-12. The first implementation probe found and fixed a missing
stderr file binding before any live run.

### A8.2 — source/binary cleanup — complete

The 15 active portable-LCD rows were rebuilt from source with the current
CUDA 13.4 toolchain and base `sm_86`, `sm_90`, `sm_120`, plus `compute_80`
PTX slices. Binary help strings no longer contain the retired
`--enable-cupti-activity` option. The standalone verifier reports all 15 rows
and suite SASS coverage as passing; trace checks were intentionally deferred
to A8.4's fresh NVBit capture. Build warnings from the vendored CUTLASS
headers are non-fatal and remain visible for later cleanup.

A8.3 is now the active milestone: bounded correctness smoke execution for the
full roster.

### A8.3 — roster smoke campaign — complete

The rebuilt 15-row roster ran sequentially through the bounded launcher with a
512 MiB expected-device-memory budget, a 1 GiB free-memory reserve, one
warmup, two repetitions, and a 120-second timeout. All 15 rows returned
`passed`; every row emitted its workload manifest and had empty stderr. No
OOM, OOD, timeout, or silent shape substitution occurred. The raw run bundle
is retained under `results/alpha/a8_smoke_20261002/` (ignored runtime data,
not a tracked claim artifact).

A8.4 is now the active milestone: one fresh live trace/replay/profile slice.
