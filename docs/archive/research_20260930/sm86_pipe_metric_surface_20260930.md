# NCU source-pipeline metric boundary

The installed Nsight Compute 2026.1 source-metric query was captured at
`results/alpha/sm86_pipe_witness_source_metric_20260930/query_metrics.txt`.
For `sass__inst_executed_per_opcode_pipeline`, NCU describes the value as an
**estimated** warp-instruction count by SASS opcode pipeline and explicitly
states that dynamic assignment is not taken into account. This is a direct
public-tool boundary, not an inference from the simulator.

The same source surface exposes `smsp__pcsamp_sample_count` and sampled warp
stall states, but those are periodic PC/state samples, not dynamic instruction
receipts. The retained direct NAK PMA field sweep found no additional route
field in 2,145 decoded records. Thus aggregate pipeline counters, NCU source
estimates, CUPTI/PMA samples, and simulator issue frontiers remain separate
observables; none supplies the required `(instruction, warp, issue ordinal,
physical sub-unit, issue cycle)` witness.

I also requested dynamic `smsp__inst_executed_pipe_alu/lsu/tensor` metrics with
`--page source --print-source sass` on a bridged NAK family body. NCU produced
the source address table but no per-source pipe columns; those counters remain
launch-level aggregates. Isolated bodies can constrain known operation
populations, but this interface does not join dynamic assignment to a
PC/warp/issue ordinal in a mixed body.


A follow-up source profile requested the newer physical-activity signals
smsp__pipe_fmaheavy_cycles_active, smsp__pipe_fmalite_cycles_active,
smsp__pipe_alu_cycles_active, and smsp__pipe_fma_cycles_active on the local
FFMA bank-order body. The NCU source page again emitted SASS rows but no
per-source pipe-cycle columns; the values remained launch aggregate metrics.
This rules out another plausible public-tool seam for an exact PC/warp/slot
join on this GA102 stack. Artifacts:
results/alpha/sm86_pipe_witness_source_metric_20260930/pipe_cycle_source_20260930/.

## 2026-09-30 latest CLI query

The installed command-line query
`ncu --query-metrics --query-metrics-mode all --query-metrics-collection
warpsampling --chip ga102` returns `smsp__pcsamp_warps_issue_stalled_*` and
`_not_issued` metrics. It returns no `warpidsamp` instance metrics and no
per-instruction route/issue ordinal. Raw query output is retained under
`results/alpha/sm86_pipe_metric_surface_20260930/latest_warpsampling_20260930/`.
