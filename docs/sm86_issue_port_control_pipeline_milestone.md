# Completed milestone: sm_86 execution-port and control-pipeline closure

**Status:** complete 2026-08-17; successor is SMSP scheduler/math-pipe contention  
**Architecture:** `sm_86` / RTX 3090 Ti  
**Predecessor:**
[barrier pipeline and launch-chain cache handoff](sm86_barrier_pipeline_cache_handoff_milestone.md)  
**Evidence roots:**
`results/captures/sm86_barrier_pipeline_20260817_v1`,
`results/captures/sm86_barrier_pipeline_coupled_20260817_v1`, and
`results/captures/sm86_barrier_pipeline_multicta_20260817_v1`

**Final evidence:**
`results/captures/sm86_issue_port_control_20260817_v1/analysis/README.md`

## Completion summary

The completed matrix contains 68 cells, seven correctness trials per cell, 204
base-clock application-replay NCU profiles, 14 SourceCounters profiles, 68
independent NVBit captures, two fully decoded 5,136-instruction sidecars, and
deterministic dispatch/final replay pairs. No cell exceeds 2% active-cycle CV;
all kernels are spill-free and every trace has full coverage and exact
cardinality.

The primary findings are:

- compiler control bits already predict dependent carry-IADD, IMAD, and LOP3
  length slopes to within about 0.02 cycles/logical step;
- every NCU profile reports `inst_issued_per_issue_active == 1.0`, so no dual
  issue pair is promoted;
- broad ALU/FMA, all-IADD, and carry-only two-/three-cycle initiation candidates
  improve some W8 identification cells but fail independent/mixed or prior
  multi-CTA transfer, so all scalar initiation candidates are rejected;
- predicated forward `BRA` with target greater than function-relative PC,
  encoded stall 5, and yield set has a ten-cycle minimum next-issue interval;
  this campaign left backward loops compiler-controlled; a later U1/U2/U4/U8
  direct-counter factorial independently identifies three uniform
  backward-target resolution cycles without changing the five-cycle
  compiler-control issue frontier;
- branch slope MAE improves from 5.033 to 0.033 cycles/branch across
  identification and held-out lengths;
- barrier traffic is unchanged while b11 skew MAE improves from 16.228 to
  5.718 cycles/generation;
- coupled traffic and c11−c10 deltas are unchanged, and multi-CTA output remains
  byte-identical; and
- production retains 6/6 coverage with MAE 2,559.00, RMSE 2,768.93, and MAPE
  33.128%, improving the 33.394% predecessor baseline.

Readiness schema 3 adds checked branch issue/resolution counters and exclusive
plus overlapping branch-readiness cycles. The branch relation delays only the
issuing warp's next frontier and creates neither a shared server nor a register
scoreboard. `sm_120` remains a typed gap.

The unresolved evidence is localized rather than fitted away. W1, W3, and W4
place at most one warp on each modeled SMSP and have no math-pipe-throttle or
not-selected samples. W8 places two warps per SMSP and exposes both, with live
penalties from 4 to 92.5 cycles/generation depending on instruction family.
Simple ready-timestamp resources do not transfer. The configured successor is
[sm_86 SMSP scheduler and math-pipe contention](sm86_scheduler_math_pipe_contention_milestone.md).

Final validation passes the complete feature-unified workspace (36
`sim-arch`, 8 `sim-engine`, 113 `sim-mem`, 326 `sim-sm`, 138 `sim-trace`, and
all runner/integration targets), formatting, diff integrity, workspace Clippy
with its documented advisory warnings, 12 focused Python tests, every campaign
reducer, a final release replay of all retained cohorts, and a live branch-probe
CUDA smoke. Representative async host replay changes by −0.20%; no privileged
command was required.

## Goal

Identify the smallest architecture-local execution-port model needed beyond
compiler control bits and the two already characterized resources (Tensor Core
initiation and CTA-barrier arrivals). The primary targets are dependent integer
latency, ALU/control throughput, branch service, and compatible-pair dual issue.

Promote a resource only when a no-memory, no-barrier identification probe
predicts held-out instruction mixes, warp counts, and CTA partitions. Then test
the unchanged resource against the retained b20, skew, multi-CTA, async, and
six-workload production cohorts.

## Why this is the nearest essential component

The completed campaigns isolate three residuals that share one missing
boundary:

- b20's shared barrier service predicts the complete width slope, while a
  constant 19.016 cycles/generation remains across four ptxas-scheduled
  `IMAD`/`IADD3`/`LOP3`/control instructions between BARs;
- the dependent nominal-32/64 skew chains are underpredicted by roughly 13-23
  cycles/generation even though their ordering and arrival-spread direction
  transfer; and
- fixed-eight-warp multi-CTA shape slopes transfer with 0.032
  cycles/generation MAE but retain a 4.004-cycle common control offset.

All non-Tensor and non-BAR execution classes are currently
`PipelineClass::DispatchOnly`. The scheduler permits at most one selected
instruction per SMSP per modeled issue step and has no instruction-port
compatibility or dual-issue relation. Compiler stall counts and dependence
counters remain authoritative readiness inputs, but they do not prove that
every selected opcode has zero execution-unit contention.

These residuals must not be folded into barrier release, cache route, CTA
count, or workload-specific constants. They instead motivate a clean
execution-only campaign.

## Ordered stages

1. **Static pipe census:** enumerate executed `sm_86` SASS heads and map only
   experimentally distinguishable pipe families.
2. **Dependent latency versus independent throughput:** measure each candidate
   family with matched instruction counts and register-dependence structure.
3. **Port compatibility and dual issue:** test predeclared same-pipe and
   cross-pipe pairs at one and several resident warps.
4. **Held-out transfer:** replay odd chain lengths, unseen mixes, CTA
   partitions, retained barrier/async cells, and production traces.

## Semantic boundary

Compiler relations, dispatch, and execution resources remain separate:

```text
control/dependence ready
          |
          v
SMSP scheduler selects warp
          |
          v
dispatch/port compatibility ----> execution-resource availability
          |                                  |
          +-------------------- next readiness/completion
```

A new pipe may delay dispatch or completion according to measured behavior,
but it must not create a generic register scoreboard. The sidecar's
write/read/wait barriers remain the producer-consumer relation. A dual-issue
model, if supported, names exact compatible pipe pairs and preserves one
traced instruction record per issued operation.

## Identification probe

Add `workloads/research_probes/issue_port_control/` with compile-time-specialized
one-SM kernels. The measured interval contains no global/shared/local memory,
no CTA barrier, no async copy, no atomics, and no spills.

| Protocol | Core construction | Identifies |
|---|---|---|
| `p00` | loop/control baseline | dispatch and loop subtraction |
| `p10` | dependent `IADD3` chain | integer-add latency |
| `p11` | dependent `IMAD` chain | integer multiply-add latency |
| `p12` | dependent `LOP3` chain | logic latency |
| `p13` | branch/control chain with fixed outcomes | CBU/control service |
| `p20` | independent same-family streams | initiation throughput |
| `p30` | predeclared same-pipe instruction pairs | port collision |
| `p31` | predeclared cross-pipe pairs | compatible dual issue |

Chain lengths are 4, 8, 16, 32, and 64 instructions. Identification uses
lengths 4/16/64 and W1/W4/W8; lengths 8/32, W2/W3/W5/W7, and reordered mixed
pairs are held out. W12/W16 are added only where register and occupancy reports
remain spill-free.

The independent-stream variants use distinct registers and the same executed
opcode count as their dependent siblings. The branch variant uses compile-time
fixed direction and checks executed SASS so divergence or reconvergence does
not become a hidden factor. A separate clock64 calibration reports the actual
chain cost; source labels such as “32” never substitute for measurement.

## Live measurements

For every retained cell:

1. run seven ordinary correctness/timing trials;
2. collect three NCU application-replay profiles at base clocks;
3. expand to five repeats above 2% `sm__cycles_active.max` CV;
4. retain `smsp__inst_issued`, `smsp__inst_executed`, issue-active,
   active/eligible warps, not-selected, dispatch-stall, math-pipe throttle,
   branch-resolving, and per-pipe instruction/cycle counters;
5. run SourceCounters on W1/W4/W8 dependent and mixed sentinels;
6. capture bounded 64-iteration NVBit traces and complete control-bit sidecars;
   and
7. replay every trace twice with exact instruction cardinality.

NCU and NVBit remain separate foreground processes. No NCU metric or protocol
identity enters replay.

## Candidate model shapes

Evaluate, without presupposing, these nested candidates:

- dispatch-only baseline;
- one initiation timeline per SMSP for one exact pipe family;
- separate SMSP-local timelines for distinguishable ALU/control families;
- a shared port between exact families; and
- compatible two-instruction issue from one SMSP in one cycle.

If latency and initiation interval cannot be separated, retain the simpler
single frontier and label the ambiguity. If a pair is not observed to issue
together, dual issue remains absent. Do not populate the dormant generic card
issue-resource block merely to make it complete.

## Diagnostics

Readiness schema 3, only if a resource survives, adds bounded checked counters
for:

- accepted instructions by pipe family;
- pipe/port busy work;
- pipe-limiting and overlapping exposure cycles;
- dual-issue pairs by exact compatibility class;
- eligible instructions denied by a port conflict; and
- maximum/aggregate queue span without a per-cycle log.

Schemas 1 and 2 remain readable and byte-compatible when the new resource is
absent.

## Promotion rules

A resource enters `sm_86` timing only when:

- dependent and independent probes distinguish its latency/throughput role;
- identification values predict every held-out chain length and warp width
  without sign or ordering error;
- a claimed dual-issue pair transfers to a held-out instruction order;
- exact executed SASS and instruction counts match the intended construction;
- the b20, skew, and multi-CTA residual slopes improve without a barrier,
  route, or CTA-count branch;
- all 36 async predecessor traces and the 18 warm-route traces rerun with
  unchanged traffic;
- the six-workload production score retains full keyed coverage and reports
  MAE/RMSE/MAPE against the 33.394% baseline; and
- `sm_120` returns a typed missing capability.

If no resource transfers, publish the pipe measurements and retain
dispatch-only behavior. A failed port hypothesis is more informative than a
per-opcode fitted delay.

## Explicit exclusions

- a generic register scoreboard;
- a constant added to all ALU instructions;
- changing compiler stall or dependence-counter meanings;
- fitting the 19.016-, 13-23-, or 4.004-cycle residuals directly;
- a workload/protocol/kernel-name timing branch;
- a fixed barrier release or generation re-arm latency;
- cache-route-dependent execution ports;
- assuming `warpid_sm % 4` proves physical scheduler identity beyond the
  modeled mapping; and
- copying any `sm_86` value into `sm_120`.

## Acceptance criteria

- Every probe is correct, spill-free, fully decoded, and fully captured.
- Three/five-repeat policy and all variance are published.
- Candidate selection uses predeclared identification cells only.
- Held-out chain, mix, width, and CTA-shape ordering transfers.
- Replay remains event-driven with no global cycle scan or per-instruction
  event allocation.
- Default outputs without a selected resource retain their prior schema.
- Focused and workspace Rust tests, Python checks, CUDA correctness, formatting,
  and diff checks pass.
- Representative host replay regression stays below 3%, or any larger absolute
  millisecond change is removed or explicitly retained.

## Expected artifacts

```text
workloads/research_probes/issue_port_control/
results/captures/sm86_issue_port_control_<date>_v1/
  metadata/
  correctness/
  ncu/
  ncu-source/
  nvbit/
  sass/
  simulator/
  analysis/README.md
```

No privileged command is currently required. If a later NCU or device step
requires operator action, provide the exact `sudo ...` command for the user to
run rather than executing it.
