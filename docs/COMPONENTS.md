# Component inventory

| Component | Keep and develop | Cleanup direction |
|---|---|---|
| `sim/engine` | Deterministic event scheduling plus ordered same-time reply-less write batches | Retain logical/physical counter separation; profile before broader queue changes |
| `sim/trace` | Protobuf decode, trace semantics, 20-byte compact validated instructions, exact-generation rostered streaming research path, raw-retaining campaign snapshots, alias-aware global-atomic 32-byte transaction projection, and timing-neutral allocation-relative atomic identity | Keep streaming opt-in; profile remaining memory/sector arenas before another representation change |
| `sim/sm` | Issue, causal readiness schemas 1–13, typed completion frontiers, scalar-global work, operation-specific operand-read barriers, streaming services, finite SM/per-SMSP queues, feedback-coupled queue/readiness, fixed miss-sector service quanta, per-warp MEMBAR completion, fixed and request/region-work completion lanes, phase CLOCK diagnostics, ATOMG/RED topology, branch resolution, async copy, CTA rendezvous, replay, control bits, operand projection, and sm_86 writeback priority | Benchmark the opt-in feedback/completion path; atomic timing still requires the many-region scheduling network |
| `sim/mem` | Sectored L1/shared L2/DRAM behavior, transactional snapshots, checked cumulative deltas, drained-state inspection, and architecture-local sm_86 STRONG.SM L1 routing | Benchmark snapshot cost on longer launch chains |
| `sim/direct_run` | Default end-to-end simulator application, readiness schemas 1–13, strict opt-in service/admission/feedback candidate documents, direct L2 per-source sector/hit JSON, typed completion-frontier JSON, and explicit cold-launch/stream-L2 cache scopes | Add a timed command-stream scope only when traces supply launch gaps, not from launch order alone |
| `sim/run` | Optional historical campaign runner | Retain behind explicit package name |
| `tools/nvbit_tracer` | Tracer sources, schemas, reader, focused tests | Remove build/capture permission wrappers |
| `tools/local_labels` | NCU command driver, parser, metric catalog | Remove lifecycle and approval machinery |
| `workloads` | Kernels, ordinary experiment scripts, exact weighted/generation-coded GA102 LTS collision probes, and a CUDA-VMM physical/virtual alias discriminator | Keep hardware counters as evaluation labels, never replay inputs |
| `eval` | Numerical prediction-versus-observation scoring | Add cohort import as needed |
| `arch_profiles` | Explicit architecture/SKU parameters plus typed sm_86 branch, barrier, writeback, and 256-byte atomic-contention topology | Keep unknown values visible |
| `tools/sol_execbench` | Optional SOL-ExecBench source, environment, and dataset | Retain in full; invoke explicitly |
| `tools/sol_baselines_harness` and campaign tools | Optional historical campaign capability | Keep scoped; never make it a root gate |

Generated build products, session worktrees, prompt packages, review receipts,
and duplicated historical plans are not project components. The retained SOL
environment, downloaded SOL dataset, and campaign corpus are explicit
exceptions because they are preserved research capabilities.

## Current construction order

Causal readiness, launch-chain L2 state, the SM-shared barrier-arrival service,
exact forward-branch resolution, and three-cycle uniform backward-target
resolution are implemented and live-validated. The
barrier service transfers across held-out CTA widths and fixed-eight-warp
multi-CTA shapes; warm replay separates the formerly collapsed L2 and DRAM
inputs. The 99-cell SMSP campaign identifies a two-cycle high-occupancy
ALU/bare-IMAD service bound but rejects direct scheduler-port, FIFO, queue, and
service-drain candidates. The exact-bank and completion-distance campaigns
then reject five Control/Allocate/completion compositions while retaining
timing-neutral operand projection. No fixed CTA release, re-arm, route,
CTA-count, math port, bank penalty, or warp-count constant was added. The next
components are:

1. evaluate whether streaming direct validation can reduce the remaining
   admission-time peak without changing wire-error precedence;
2. retain the W8 ninth-reservation cell as a future capacity canary without
   importing the rejected capacity-8/pop-1 rule; and
3. retain the semantic d4 fixed-bypass/d5 variable frontier without adding a
   hardware scoreboard that repairs invalid compiler schedules.

See the [adversarial atomic milestone](sm86_atomic_adversarial_milestone.md).
Atomic transaction/request topology and CLOCK diagnostics are exact; service
timing is withheld because the distributed atomic-unit hash is unidentified.
The retained LTS probes recover 48 exact classes and reject all virtual-address
hashes on a high-address holdout. The VMM successor confirms that sequential
RED class follows physical allocation plus offset across seven aliases, while
the complete 2 MiB relative collision partition transfers across allocation
mechanisms. Extended and cross-allocation tests reject a universal relative
map: each physical granule has hidden phase. Require phase evidence or a
validated statistical phase model before adding atomic timing.
