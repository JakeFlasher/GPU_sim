# Completed milestone: sm_86 SMSP scheduler and math-pipe contention

**Status:** completed on 2026-08-17; no scheduler/port candidate promoted  
**Architecture:** `sm_86` / RTX 3090 Ti  
**Predecessor:**
[execution-port and control-pipeline closure](sm86_issue_port_control_pipeline_milestone.md)  
**Evidence:**
`results/captures/sm86_issue_port_control_20260817_v1/analysis/README.md`

## Outcome

The completed campaign contains 99 cells, 693 correctness executions, 297
base-clock NCU profiles, 21 SourceCounters profiles, and 99 independent NVBit
captures with 3,322,744 replayed warp instructions. All NCU repeats have at
most 2% active-cycle CV and issue ratio exactly 1.0. Both binaries decode all
6,624 SASS instructions across 52 specializations with zero unknown controls,
spills, or local memory.

At W12/W16, live active time approaches a two-cycle half-warp service bound:
mean absolute distance from `2 × executed pipe instructions/SMSP` is 1.26
cycles/generation for ALU-dominant cells and 4.41 for bare-IMAD-dominant cells.
This identifies service capacity but does not identify scheduler backpressure.

Ten nested candidates were replayed twice over all 99 cells. The best cyclic
ALU-burst/FMA candidate reduces W8 same-family MAE from 29.14 to 2.28 and
held-out same-family MAE from 58.15 to 19.91 cycles/generation, but changes W4
no-contention timing, misses W12 magnitude, and reduces reversed-family order
sign transfer from the production scheduler's 13/18 to 10/18. Strict ports,
oldest fallback, bounded queues, rational FMA service, and an unbounded service
drain also fail at least one transfer surface. None is promoted. After removing
the experiments, all 198 restored replay outputs are byte-identical to the
original production baseline; readiness remains schemas 1–3.

Complete reduction:
`results/captures/sm86_smsp_scheduler_contention_20260817_v1/analysis/README.md`.
The successor is
[fixed-latency Control/Allocate and register-file contention](sm86_control_allocate_register_file_milestone.md).

## Goal

Explain the extra delay that appears when a second warp occupies each modeled
SMSP, separating math-pipe throttle from scheduler not-selected time. Select a
scheduler/port relation only if it predicts held-out warp counts, instruction
families, phase offsets, and CTA partitions without altering the already
validated compiler, branch, barrier, cache, or launch-chain semantics.

## Why this is next

The completed issue campaign removes several competing explanations:

- W1, W3, and W4 place at most one warp on each `warpid_sm % 4` modeled SMSP;
  their dependent IADD/IMAD/LOP3 length slopes are already predicted by control
  bits and their NCU math-throttle/not-selected samples are zero.
- W8 places two warps on every modeled SMSP. Its control-subtracted penalties
  are 82.0 cycles/generation for dependent carry-add, 4.0 for dependent IMAD,
  12.0 for dependent LOP3, 92.5/16.0/45.0 for their independent variants, and
  70.0 for mixed IADD/FFMA.
- Those W8 cells expose matched math-pipe-throttle and not-selected samples.
  The largest examples are carry-add (0.872/0.872), independent carry-add
  (0.801/0.792), independent LOP3 (0.733/0.717), and mixed IADD/FFMA
  (0.531/0.562) warps per issue-active cycle.
- Every profile has issue ratio 1.0. Same-cycle dual issue is not observed.
- Simple SMSP-local ALU/FMA or carry-only initiation timestamps improve their
  own cells but fail independent/mixed or prior multi-CTA transfer.

The remaining relation therefore involves arbitration among multiple eligible
warps and a busy execution port, not single-warp dependency latency.

## Semantic boundary

```text
warp control/dependence ready
             |
             v
candidate set on one SMSP
             |
     +-------+-------+
     |               |
port can accept   port unavailable
     |               |
issue winner      choose alternate ready warp or idle
     |               |
     +-------+-------+
             v
      hold/yield update
```

Port availability may affect which ready warp is selected. It must not become a
generic scoreboard, per-opcode delay, or NCU stall-ratio input. Compiler stall,
dependence counters, exact forward-branch resolution, Tensor initiation, and
barrier arrival service retain their present meanings.

## Probe family

Add `workloads/research_probes/smsp_scheduler_contention/` with no memory,
barriers, async copy, atomics, or spills in the measured interval.

The primary construction uses physical trace placement and varies warps per
modeled SMSP:

| Shape | Total warps | Modeled warps/SMSP | Role |
|---|---:|---:|---|
| `s1` | 4 | 1 | no-contention control |
| `s2` | 8 | 2 | identification |
| `s3` | 12 | 3 | held-out depth |
| `s4` | 16 | 4 | held-out saturation |

Instruction bodies use the already validated carry-IADD, IMAD, LOP3, branch,
and mixed ALU/FMA forms. Each has dependent and independent variants. A phase
factor inserts 0/4/8/16 control-predicted cycles before the second warp on each
SMSP reaches its hot sequence, distinguishing a port queue from lockstep warp
selection. A complementary construction pairs different families on the two
warps of one SMSP.

Identification uses W4/W8, phase 0/8, and same-family pairs. W12/W16, phase
4/16, and cross-family order reversals are held out. All source-level labels
are replaced by executed SASS and measured standalone phase cost in analysis.

## Candidate scheduler shapes

Evaluate these nested candidates:

- current held-warp/youngest-ready scheduler with dispatch-only scalar pipes;
- a port-aware picker that falls back to another ready warp when the held warp's
  exact pipe cannot accept;
- one FIFO or oldest-ready queue per SMSP pipe family;
- a bounded rational-throughput port with scheduler-visible availability; and
- a shared ALU/FMA arbitration point only if cross-family probes require it.

Do not add a queue depth that W12/W16 cannot distinguish. Do not select a
candidate from sampled NCU stall ratios; use cycle/ordering transfer, with stall
samples only as causal support.

## Live campaign

For every retained cell:

1. run seven correctness trials;
2. collect three base-clock application-replay NCU profiles and expand to five
   above 2% active-cycle CV;
3. retain issue-active, active/eligible warps, math-pipe-throttle,
   not-selected, selected, dispatch-stall, instruction count, and per-pipe
   counters;
4. run SourceCounters on W4/W8/W12 same- and cross-family sentinels;
5. capture a bounded 64-generation NVBit trace with exact `warpid_sm` slots;
6. replay every trace twice from complete control sidecars; and
7. publish mapping sensitivity separately from claims about physical SMSP
   identity.

NCU and NVBit remain separate processes. No NCU value enters replay.

## Diagnostics

If a scheduler relation survives, readiness schema 4 adds checked bounded
fields for:

- candidates ready per SMSP decision;
- held-warp accepted, alternate-warp accepted, and idle-despite-ready counts;
- port-unavailable exposure and limiting cycles by exact family;
- port queue accepted work/busy cycles;
- ready warp denied by scheduler versus denied by port; and
- maximum queue depth observed without a per-cycle event log.

Schemas 1-3 remain byte-compatible where the relation is absent.

## Promotion rules

A relation enters `sm_86` timing only when:

- it predicts W8 identification and W12/W16 held-out cycle ordering;
- phase-offset sign and magnitude transfer without a protocol branch;
- same-family and reversed cross-family pairs use one unchanged policy;
- the completed branch, barrier-width, b11-skew, warm-route, and multi-CTA
  results retain their direction and traffic;
- the 36 predecessor traces retain exact counters;
- production retains 6/6 coverage and reports MAE/RMSE/MAPE against 33.128%;
- representative host replay remains within 3% or a larger millisecond change
  is explicitly justified; and
- `sm_120` returns a typed missing capability.

If no scheduler shape transfers, publish the W8/W12/W16 evidence and retain the
current scheduler. Do not substitute a fitted ALU interval.

## Explicit exclusions

- generic register scoreboards;
- scalar delays fitted from W8 penalties;
- importing math-throttle or not-selected ratios into the simulator;
- assuming modulo warp-slot mapping proves physical scheduler assignment;
- dual issue without an issue ratio above one;
- changing the exact forward-branch rule;
- barrier, route, CTA-count, or workload-name corrections; and
- `sm_86` defaults copied to `sm_120`.

## Expected artifacts

```text
workloads/research_probes/smsp_scheduler_contention/
results/captures/sm86_smsp_scheduler_contention_<date>_v1/
  metadata/
  correctness/
  ncu/
  ncu-source/
  nvbit/
  sass/
  simulator/
  analysis/README.md
```

No privileged command is currently required. If a later tool invocation needs
operator action, provide the exact `sudo ...` command for the user to run.
