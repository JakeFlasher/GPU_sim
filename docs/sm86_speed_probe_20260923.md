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

## M3 bounded optimization: aggregate L1 preflight budgets

The first `perf record` identified `L1Cache::preflight_mutation_budgets` as a
6.18% top-level cycle consumer. Its input is a batch of small mutation budgets;
the old path copied and checked the complete counter snapshot once per budget.
The production semantics do not require those intermediate snapshots because
the operation is a preflight with no cache-state mutation. The implementation
now aggregates the 17 checked counter deltas per memory origin, then performs
at most one checked projection for each origin. Recency, fill-sequence, origin
partition, and overflow checks remain explicit.

The changed release binary was replayed on the same fresh locked-clock GEMM
trace three times. The JSON result was equivalent for the reported
counters/frontiers in the comparison run; event totals remained
`events_processed=362424`, `queue_events_processed=348088`, and
`l2_access=185344`.

| Build | Median replay phase | Median process wall | GPC frontier | SM terminal frontier |
| --- | ---: | ---: | ---: | ---: |
| before aggregation (3 runs) | 782.036 ms | 894.583 ms | 12,949 | 11,101 |
| after aggregation (3 runs) | 755.453 ms | 872.462 ms | 12,949 | 11,101 |

This is a 3.4% replay-phase and 2.5% process-wall reduction on this host and
input. `perf report` after the change no longer places
`preflight_mutation_budgets` in the top-level 2% list; remaining hot paths
include queue cloning, issue-cycle work, cache-origin accounting, and trace
cohort handling. This is simulator engineering evidence, not a hardware-speed
claim and not a universal workload speedup.

As a bounded cross-workload smoke check after the change, the same release
binary completed the fresh RMSNorm and SiLU traces with full block coverage:

| Workload | Median process wall | Median replay phase | GPC frontier | SM terminal frontier |
| --- | ---: | ---: | ---: | ---: |
| BF16 RMSNorm (64 blocks) | 0.375 s | 298.227 ms | 18,944 | 17,120 |
| BF16 SiLU (1,024 blocks) | 0.451 s | 358.085 ms | 5,576 | 3,720 |

All six control replays exited successfully. These are post-change smoke
measurements, not before/after speed claims for those two kernels.
