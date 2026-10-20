# Academic Style Review

This memo records the terminology review that precedes the release rewrite.
The manuscript checkout was read-only; no source paper files were changed.

## Sources consulted

1. Chen Ding and Xiaoya Xiang, “A Higher Order Theory of Locality,” MSPC
   2012, and Xiang et al., “HOTL: A Higher Order Theory of Locality,”
   ASPLOS 2013.
2. Rodrigo Huerta et al., “Dissecting and Modeling the Architecture of
   Modern GPU Cores,” MICRO 2025, including its open-source simulator
   documentation.
3. The current `gpu_simulator_arxiv/main.tex`, bibliography, figures, and
   internal style/jargon audits.

## Language patterns adopted

The HOTL papers define a small set of related locality functions—footprint,
lifetime, miss ratio, inter-miss interval, and reuse distance—and state the
correctness condition for converting a footprint into a miss-ratio curve. The
release therefore uses those terms with explicit domains and does not call an
interval estimate a hardware MRC.

The MICRO'25 paper uses conventional GPU terms: compiler-guided
greedy-then-youngest scheduling, sub-core pipelines, static control fields,
scoreboard/dependence management, register-file cache, instruction
prefetching, hardware validation, and MAPE. The release adopts those names
instead of internal labels such as “tap-point projection,” “causal horizon,”
or “unpriced request.”

## Claims that require qualification

- “Cycle-level” describes the modeled architectural state and timestamps; it
  does not mean that the implementation executes one host iteration per GPU
  cycle.
- “Report-only” describes an attribution observer that cannot change timing or
  cache state. It must not be presented as a post-hoc correction.
- “Effective” and “conditional” identify assumptions that are explicit in the
  architecture card or trace contract; they are not claims of recovered NVIDIA
  RTL.
- NCU rows are independent targets. A join is evidence only at the stated
  identity or fallback confidence level.
- HOTL WAPE is an estimator-vs-reference metric, not simulator-vs-silicon
  accuracy.

## Release vocabulary decisions

The controlled replacements are maintained in [`TERMINOLOGY.md`](TERMINOLOGY.md)
and the implementation sequence is in
[`ACADEMIC_RELEASE_PLAN.md`](ACADEMIC_RELEASE_PLAN.md). In particular:

- “hardware-counter attribution” replaces “counter projection”;
- “conservative lookahead horizon” replaces “causal horizon”;
- “in-flight L1 miss awaiting arbitration” replaces “unpriced miss”;
- “core-only/fixed-latency diagnostic mode” replaces “fast frontier mode”;
- “completion timestamp/stage” replaces an unqualified “frontier”;
- “end-to-end hardware-counter validation” replaces “closed-loop
  calibration.”

## Metric presentation standard

Every result table must state the unit of observation (launch, workload,
capture, or exact join), target population, zero-target handling, and
aggregation. The standard set is:

- Mean APE, median APE, and P90 APE for per-row error;
- WAPE only with its aggregate formula;
- Pearson correlation when dynamic range matters;
- geometric-mean speedup plus total-suite wall time;
- exact, fallback, and unmatched NCU joins as separate populations.

This prevents the release from mixing analytical locality validation,
simulator-to-NCU accuracy, and runtime measurements under one headline.
