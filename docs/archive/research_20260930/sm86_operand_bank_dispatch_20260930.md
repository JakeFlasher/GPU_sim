# SM86 operand-bank dispatch microbenchmark — RTX 3090 Ti

**Date:** 2026-09-30  
**GPU:** RTX 3090 Ti / GA102 / SM86, driver 615.71.09, clocks 1350/10501 MHz  
**Scope:** performance-only local microbenchmark; not an exact pipe/slot witness.

## Construction

The public reproduction bundle for the current SM86 operand-collector
discussion was compiled with CUDA 13.2 for sm_86 and patched into three
SASS-order variants:

* 1e1o: alternating even/odd source-register bank groups;
* 2e2o: two even, two odd;
* 4e4o: four even, four odd.

The 32 FFMA bodies and their control words were reordered by the supplied
binary patcher. The timed region contains independent FFMA operations and
compiler control words remain attached to each instruction. The external
bundle's bundled CPU result predicate was executed by its launcher; the
observations below are timing and Nsight source-counter fields, not that
predicate.

Artifacts are retained under
results/alpha/sm86_operand_bank_dispatch_goal_20260930/:

* source and patcher snapshots;
* three NCU reports;
* source-page CSV exports with per-PC warp-stall columns;
* launcher stdout and hardware context.

## Local timing

With 20,000 loop iterations and 1,000 warm-up iterations:

| order | total timed-region cycles | cycles / 32 FFMA |
|---|---:|---:|
| 1e1o | 52 | 1.625 |
| 2e2o | 67 | 2.094 |
| 4e4o | 83 | 2.594 |

The ordering-dependent slope reproduces the current public SM86 discussion's
qualitative result (1.5/2.0/2.5 cycles per FFMA on another GA102), while the
3090 Ti values are offset by the surrounding body and clock/toolchain regime.

## NCU source stall observations

Nsight Compute 2026.1.0 SourceCounters profiles used aggregate source metrics
and per-PC warp-stall sampling. All three profiles had 32 FFMA source rows.
The rows below list the first nonzero dispatch-stall PC and the sampled
dispatch-stall counts:

| order | FFMA rows with nonzero dispatch stall | first row | total dispatch-stall samples | all FFMA samples |
|---|---:|---:|---:|---:|
| 1e1o | 13/32 | #8 | 128 | 478 |
| 2e2o | 13/32 | #7 | 253 | 607 |
| 4e4o | 20/32 | #7 | 390 | 742 |

The nonzero row patterns are:

* 1e1o: #8, #10, #12, ... #32;
* 2e2o: #7, #9, #11, ... #31;
* 4e4o: groups beginning at #7, #11, #15, ... with three consecutive
  nonzero rows in each group.

The sampler reports zero not-issued dispatch samples for these FFMA rows.
This is a source-correlated stall pattern, not a receipt of the cycle in
which each FFMA issued.

## Interpretation

The local result strengthens a fixed resource-reservation / register-bank
pressure model:

* source-bank order changes steady-state throughput while static control words
  are unchanged;
* the first dispatch-stall onset stays near FFMA 7--8;
* NCU's per-PC stall samples preserve the ordering signature.

This evidence can tighten the existing opt-in SM86 operand-bank/admission
candidate envelope. It must not be promoted to:

* a physical ALU/FMA sub-pipe label;
* a per-instruction issue ordinal or issue cycle;
* a global SMSP identity;
* a proof that the hidden mechanism is literally an operand collector queue.

The current Nsight documentation states that warp-stall samples are sampled
scheduler states, and that the sampler can select a different warp from the
one issued in the same cycle
([Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)).
The current SM86 discussion also reports that no single public model explains
both onset and steady-state effects
([forum reproduction](https://forums.developer.nvidia.com/t/how-does-the-operand-collector-gate-ffma-issue-on-ampere-sm-86/381274)).
Thus this is a new local timing/resource witness, not closure of the exact
hardware pipe/issue witness.

## Closure update

**Resolved layer:** source-bank-order-dependent dispatch pressure is now
observed on the target RTX 3090 Ti and can be represented by a bounded,
opt-in operand-admission/resource envelope.

**Still unresolved:** dynamic physical pipe, sub-unit, issue-slot, and exact
per-instruction issue-cycle identity. Aggregate and sampled source metrics do
not expose that tuple.
