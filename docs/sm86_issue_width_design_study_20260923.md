# SM86 issue-width design study

Updated 2026-09-23. This is a model-relative architecture exploration result,
not a silicon prediction.

## Question

Does increasing the modeled per-SMSP issue transaction width from the retained
baseline of one change replay timing or improve simulator wall time on the
eight-row SM86 small cohort?

The intervention is classified trace_safe for these traces: it changes only
the scheduler/resource service while preserving the captured instruction
stream, addresses, predicates, synchronization participants, and values.

## Method

* Same eight retained NVBit traces, control sidecars, SM86 card, pchase card,
  L1+shared-L2 mode, launch cache scope, and optimized sim-run binary.
* Issue widths: 1 (baseline), 2, and 4 (hypothetical override).
* Two process runs per workload/configuration.
* Wall time includes direct trace loading and replay; the detailed speed probe
  separately reports replay-only phase time.
* NCU rows are labels only; no NCU value entered the override or replay.

## Results

| Workload | Width 1 wall | Width 2 wall | Width 4 wall | Width 1 frontier | Width 2 frontier | Width 4 frontier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| embedding_gather_int32_sm80 | 0.064 s | 0.071 s | 0.073 s | 13,738 | 13,664 | 13,664 |
| flash_attn_fwd_bf16_sm80 | 0.088 s | 0.109 s | 0.111 s | 17,039 | 17,039 | 17,039 |
| gemm_hmma_bf16_sm80 | 0.794 s | 0.908 s | 0.922 s | 12,949 | 12,959 | 12,959 |
| gemm_imma_int8_sm80 | 0.581 s | 0.696 s | 0.702 s | 10,918 | 10,912 | 10,912 |
| gemm_sparse_2to4_bf16_sm80 | 0.083 s | 0.097 s | 0.099 s | 5,941 | 5,941 | 5,941 |
| layernorm_fwd_bf16_sm80 | 0.661 s | 0.885 s | 0.945 s | 22,310 | 22,298 | 22,298 |
| rmsnorm_fwd_bf16_sm80 | 0.367 s | 0.455 s | 0.480 s | 18,944 | 18,929 | 18,929 |
| silu_fwd_bf16_sm80 | 0.448 s | 0.564 s | 0.643 s | 5,642 | 5,535 | 5,543 |

Across rows, median process wall time was 0.408 s at width 1, 0.510 s at
width 2, and 0.562 s at width 4. On the fresh GEMM trace, replay-only phase
times were 797.9 ms, 897.9 ms, and 1,422.2 ms for event baseline, width 2,
and dense reference respectively.

## Interpretation

The override is real: issue width changes the production replay frontier on
some workloads and is visible in the JSON output. It does not improve wall
time in this implementation. Width 2/4 can add scheduler transaction work
without enough independently ready warp work to amortize it.

This is a useful negative design result:

* do not claim wider issue is automatically faster;
* retain width as a configurable architecture-study axis;
* profile the issue loop before attempting a speed optimization;
* keep the event/dense speed result separate from the architectural width result;
* do not promote the frontier projection to a physical NCU timing metric.

## Next decision

Use this study as the first architecture-design artifact. The next mechanism
candidate is a demand-driven issue-loop optimization, evaluated against the
width-1 baseline and the dense reference. If profiling shows no safe
optimization opportunity, keep this negative result and choose a different
trace-safe resource axis rather than tuning a width constant.
