# GPU simulator

This repository contains the Rust replay engine, architecture inputs, NVBit
trace producer, CUDA workloads, and measurement tools for two explicit lanes:

- **SM86 native baseline:** the established RTX 3090 Ti trace/replay lane.
- **SM120 active alpha:** the RTX 5090 hardware-in-loop lane driven by the
  external fast-model plan and retained evidence under `/dev/shm`.

```text
NVBit/SASS -> enhanced protobuf + sidecars -> sim-trace
           -> sim-sm / sim-mem / sim-engine -> sim-run
NCU/CUPTI/CUDA events ---------------------> independent comparison
```

## Start here

1. Read [`AGENTS.md`](AGENTS.md) for execution boundaries.
2. Read [`docs/PROJECT_MEMORY.md`](docs/PROJECT_MEMORY.md) for the short
   continuation snapshot.
3. Read [`docs/STATUS.md`](docs/STATUS.md) for current capability limits and
   the next SM120 goal.
4. Follow the active plan at
   `/root/gpu_cache_memory_modeling_proposal/PLAN.md`.
5. Use [`docs/README.md`](docs/README.md) to route to the narrow contract or
   source seam required by the task.

## Permanent alpha evidence policy

The live alpha branch does not use fixture-driven/synthetic simulator tests or
hashing/checking/validation/verification/admission/schema test gates. Keep only
real-trace/real-sidecar functionality probes. Generalization and performance
are judged on the full SOLBench plus modern/SOTA capture suites in
`/root/sm120_rtx_5090_capture` against independent NCU ground truth; full
replay is primary and fast replay is complementary.

## Active SM120 paths

- `sim-run --sm120-alpha --mode flat`: primary full replay.
- `sim-run --sm120-alpha --fast-frontier-only`: complementary fast frontier.
- `tools/sm120_replay_bench.py`: retained-corpus replay and metric join.
- `tools/sm120_envelope_model.py`: stage/feature projection.
- `tools/sm120_trace_features.py`: direct trace/address/locality features.
- `tools/sm120_memory_service_model.py`: logical route/service envelope.
- `tools/sm120_async_state_model.py`: typed TMA/UTMA/mbarrier ledger.
- `tools/sm120_dependency_features.py`: conservative dependency/front-end
  features.

The full replay is the primary performance path; the fast path is diagnostic.
CUDA events and NCU observations remain independent of simulator inputs. The
current alpha reports behavioral frontiers and typed gaps, not proprietary
SM120 issue-port wiring, cache geometry, DRAM policy, or native async
completion timestamps. External detailed cycle-simulator comparisons are not
part of this lane.

## Repository map

| Path | Role |
| --- | --- |
| `sim/` | Rust trace frontends, scheduler, memory model, event engine, runners |
| `arch_profiles/` | Native cards, custom specs/programs, schemas |
| `tools/nvbit_tracer/` | Native dynamic trace and sidecar producer |
| `tools/sm120_*.py` | SM120 feature, replay, service, and evidence tools |
| `workloads/`, `experiments/` | CUDA inputs and bounded probes |
| `eval/` | Small comparison utilities |
| `docs/`, `plans/` | Live contracts and plan pointer |
| `data/`, `references/` | Retained datasets, manuals, papers, and prior art |

Build/run commands and measurement scope are described by the active status and
runbook, not by generated result files.
