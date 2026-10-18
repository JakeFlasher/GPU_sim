# SM86 dependent-SFU effective latency — 2026-10-15

## Mechanism

H20 adds a narrow source-qualified SFU dependency arm for static same-register
forms such as `MUFU.RCP R5,R5` and `MUFU.EX2 R31,R31`. On SM86, the warp's next
matching SFU waits on an effective 30-cycle result frontier from an independent
RTX 3090 Ti clock64 dependent-chain probe; the existing four-cycle SMSP
initiation cadence remains unchanged. Other SFU forms retain the prior model.
This is an effective dependent-chain scenario, not a universal MUFU latency or
proprietary pipeline claim.

## Full-first evidence

Five real SFU-bearing roots completed full and fast replay 5/5:
`lcd_cross_entropy_fwd_fp32_sm80`, `lcd_flash_attn_bwd_bf16_sm80`,
`lcd_conv2d_implicit_gemm_bf16_sm80`, `simbench_l1_dram_stream_stride_mix`,
and `simbench_l1_l2_thrash_pointer_chase`.
Artifact: `../sm86_replay_results/h20_sfu_dependent_20261015_v2/full.jsonl.zst`.
The flash-attention backward frontier increased by 268 cycles and its SFU
readiness exposure rose to 592 cycles, demonstrating that the gate is live.

Complete default full-first replay completed 356 roots (286 `ok`, 62 mixed-code
controls, 8 replay/input failures), 12,392 launches in 37.530 s with 12
workers:
`../sm86_replay_results/full_corpus_h20_sfu_dependent_20261015_v1/full.jsonl.zst`.
Report: `../sm86_replay_results/metric_report_h20_sfu_dependent_20261015_v1/`.
The corpus observed 269 launches with nonzero SFU readiness exposure,
970,993 limiting SFU cycles, and 1,345,369 exposure cycles.

## Exact residuals versus H19

| Metric | H19 mean APE | H20 mean APE | Delta |
|---|---:|---:|---:|
| L2 TEX source sectors | 28.24% | 28.24% | 0.00 pp |
| DRAM read sectors | 40.91% | 40.91% | 0.00 pp |
| SM active cycles | 18.01% | 18.01% | +0.00 pp |
| elapsed IPC | 12.31% | 12.31% | +0.00 pp |
| GPC elapsed cycles | 12.91% | 12.91% | +0.00 pp |

The full corpus shows a neutral/slightly adverse aggregate effect despite live
local frontiers. Keep the narrow gate and source evidence as a promoted
mechanism, but do not claim IPC closure; the next slice must target the
remaining critical-path coverage rather than widen the SFU rule globally.
