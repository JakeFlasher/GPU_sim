# SM86 speed and override probe

Updated 2026-09-23. This is engineering evidence for M2/M3, not a SOTA
comparison or hardware-speed claim.

## Input

Trace: retained full-coverage \`gemm_hmma_bf16_sm80\` from
\`results/captures/sm86_ncu_accuracy_recovery_20260901_final_v1\`.

Release build:

    cargo build --release --manifest-path sim/Cargo.toml -p sim-run \
      --features reference,performance-counters

The replay command uses the SM86 card, pchase card, \`l1-l2\` mode, launch cache
scope, and NCU-shaped output. \`SIM_RUN_PERFORMANCE_COUNTERS=1\` reports setup,
trace-load, schedule, replay, serialization, and total process phases.

## Event versus dense reference

| Mode | Replay phase | Total process | Queue events / local steps | Frontier |
| --- | ---: | ---: | ---: | ---: |
| event-driven | 764.5 ms | 887.3 ms | 344,533 queue events | 12,949 GPC observation cycles |
| dense reference | 1,395.2 ms | 1,511.5 ms | 924,756 local cycle steps | 12,949 GPC observation cycles |

The event lane is 1.82x faster for replay and 1.70x faster end to end on this
input. The two lanes share model state and are therefore a model-relative
engineering comparison, not an independent hardware oracle. The matching
frontier is a consistency control.

## Hypothetical issue-width override

The new direct-run option is:

    --hypothetical-issue-ports N

\`N\` must be 1 through 4. It changes only the issue transaction width; card
identity, latency inputs, occupancy, and memory mechanisms remain unchanged.
The JSON report includes \`hypothetical_issue_ports\` when present, and the issue
source is labeled \`hypothetical_issue_ports\` in campaign-compatible output.

Across the eight retained small-cohort traces, two process runs per cell gave:

| Issue width | Median replay wall time across rows | GEMM issue cycles | GEMM GPC frontier |
| ---: | ---: | ---: | ---: |
| 1 | 0.408 s | 11,090 | 12,949 |
| 2 | 0.510 s | 11,100 | 12,959 |
| 4 | 0.562 s | 11,100 | 12,959 |

The override changes timing/resource state but does not improve wall time in this
first implementation. The negative speed result is retained; the likely
mechanism is extra scheduler transaction work without enough independent
eligible warps for this trace. No speedup claim is made.

## Next step

Profile the issue loop and inspect a second workload family before changing the
implementation. If issue width remains slower, keep it as a configurable
design-study axis and choose a separate demand-driven optimization for M3.



## Fresh live wall/RSS measurement

On the freshly captured locked-clock GEMM trace, separate child-process
measurements reported:

| Mode | Wall time | Replay phase | Peak RSS |
| --- | ---: | ---: | ---: |
| event-driven | 0.910 s | 797.9 ms | 57,648 KB |
| dense reference | 1.541 s | 1,422.2 ms | 58,128 KB |
| hypothetical issue width 2 | 1.015 s | 897.9 ms | 58,144 KB |

The event lane is 1.69x faster end to end and 1.78x faster in replay-only
time than dense on this fresh trace. The issue-width override remains slower
than baseline (about 11.5% end-to-end on this row), so it is a real design
axis but not the speed mechanism.
