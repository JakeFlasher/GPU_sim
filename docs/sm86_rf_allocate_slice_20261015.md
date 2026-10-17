# SM86 RF Control/Allocate decoupling — 2026-10-15

## Mechanism

H15 retains H3's transactional two-bank/one-port/three-slot RFC reservation
model, but removes RF-port conflict from scheduler eligibility. Huerta et al.
MICRO'25's Ampere model places fixed-operation RF reservations in Control /
Allocate; CGGTY issue may continue while the downstream read window is
reserved. H3 had incorrectly delayed issue directly, overcharging IPC.

Typed gaps, unsupported shapes, RFC capacity, and cache/DRAM state remain
unchanged. RF reservations are now timing-neutral at issue but remain visible
as causal diagnostics.

## Representative full replay

High-RF real roots completed full replay 4/4:
`simbench_l1_vendor_sgemm_cublas`,
`simbench_l1_vendor_sgemm_strided_batched_cublas`,
`simbench_l2_swiglu_ffn_fused_bf16`, and `fp32_fma_dep_chain`.
Vendor SGEMM issue frontier returned from 10,316 cycles under H3 to 10,070
cycles; RF reservations remain recorded without scheduler delay.

## Complete-corpus evidence

Full-first replay completed 356 roots (286 `ok`, 62 mixed-code controls, 8
replay/input failures), 12,392 launches in 35.109 s with 12 workers:
`../sm86_replay_results/full_corpus_h15_rf_allocate_20261015_v1/full.jsonl.zst`.

## Exact-identity residual deltas

| Metric | H14 mean APE | H15 mean APE | Delta |
|---|---:|---:|---:|
| GPC elapsed cycles | 13.05% | 12.91% | −0.15 pp |
| SM active cycles | 18.19% | 18.01% | −0.19 pp |
| SM active max cycles | 18.00% | 17.82% | −0.17 pp |
| SMSP active cycles | 17.69% | 17.50% | −0.19 pp |
| elapsed IPC | 12.42% | 12.31% | −0.11 pp |
| DRAM read sectors | 40.91% | 40.91% | 0.00 pp |

Report: `../sm86_replay_results/metric_report_h15_rf_allocate_20261015_v1/`.

H15 is promoted as a bounded IPC refinement, not a physical RF claim. The
remaining IPC residual is still above the sub-10% target; continue to the next
mechanism slice.
