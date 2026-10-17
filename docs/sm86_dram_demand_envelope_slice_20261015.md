# SM86 source-composed DRAM-demand envelope — 2026-10-15

## Scope

H11 emits a typed source-composed demand envelope:

- lower edge: modeled L2-missing data plus compulsory instruction sectors;
- candidate high edge: lower edge plus static `LDG.*.CONSTANT` CTA/warp line
  candidates and H2 CBANK offset-domain candidate lines.

This is an uncertainty envelope, not achieved DRAM traffic, a mathematical
upper bound, or a service completion. The 825 GB/s effective service-rate
scenario remains explicit and physical partition/mapping remains unknown.

## Representative full replay

| Root | Candidate low | Candidate high | NCU DRAM sectors |
|---|---:|---:|---:|
| FP16 HMMA | 243 | 343 | 340 |
| FP32 dependency chain | 189 | 705 | 300 |
| local/register spill | 797 | 817 | 912 |
| RMSNorm BF16 | 1,093 | 2,120 | 1,220 |

Three of four NCU observations fall inside the source candidate envelope; the
comparison is post-replay diagnostic evidence only and does not select a
multiplier or topology.

## Complete-corpus evidence

Full-first replay completed 356 roots (286 `ok`, 62 mixed-code controls, 8
replay/input failures), 12,392 launches in 35.846 s with 12 workers. Artifact:
`../sm86_replay_results/full_corpus_h11_dram_demand_20261015_v1/full.jsonl.zst`.

The post-replay report is
`../sm86_replay_results/metric_report_h11_dram_demand_20261015_v1/`.
Exact DRAM mean APE remains 40.91%; cache and IPC residuals remain mechanism
class dependent. Independent audit finds essentially no correlation between
DRAM residual and static constant-load/cbank candidate counts, so no global
GCC/constant multiplier is promoted.

## Verdict

H11 improves typed demand accounting and reports useful intervals but does not
close DRAM accuracy. Remaining residuals are workload/source-class specific;
next work should target typed writeback/transport tails or a conditional IPC
arm rather than a universal traffic scale.
