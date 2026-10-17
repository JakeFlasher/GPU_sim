# SM86 fixed register-read/RFC slice — 2026-10-15

## Scope

H3 adds a bounded, full-mode-only SM86 register-read envelope at the existing
SASS operand projection seam. It is not a proprietary RF topology claim. The
model uses two effective regular-register banks, one read port per bank, a
three-cycle read window, and a three-entry-per-bank compiler reuse queue. RFC
keys retain block/warp slot, bank, register, and source-operand position;
entries are deterministically evicted FIFO within each bank. Reuse hits consume
an entry under the explicit one-shot effective assumption. Cache/L1/L2/DRAM
state and memory request ordering are untouched.

Projected scalar fixed-latency operands reserve ports transactionally. Gap,
not-applicable, vector/multi-cell, and unsupported shapes do not reserve a
port or mutate RFC state; they remain typed coverage gaps. `applied_wait_cycles`
is readiness exposure charged before issue; `port_conflict_cycles` is only a
residual same-cycle reservation conflict and is not a total stall count.

## Representative full replay

Five real SM86 roots completed integrated full replay **5/5**:
`fp32_fma_dep_chain`, `fp16_hmma_gemm_traceable`,
`simbench_l1_register_spill_local_mem`,
`simbench_l1_warp_divergence_reconvergence`, and
`simbench_l1_vendor_sgemm_cublas`.

The arm reserved projected operands and observed RFC activity on real traces;
for example, vendor SGEMM had 5,024 reservations, 1,595 reuse hits, and 4,189
misses. No replay deadlock or input disposition changed.

## Complete-corpus full-first evidence

| Field | Result |
|---|---:|
| Roots discovered | 356 |
| `ok` roots | 286 |
| Mixed-code controls | 62 |
| Replay/input failures | 8 |
| Successful launches | 12,392 |
| Driver wall time | 34.431 s |
| Workers | 12 |
| NCU used as replay input | No |

Artifact: `../sm86_replay_results/full_corpus_h3_rf_20261015_v2/full.jsonl.zst`.

Across successful launches, the bounded arm recorded 27,167,662 projected
reservations, 27,299,746 bank-0 reads, 26,596,489 bank-1 reads, 3,622,126
RFC hits, 6,908,102 RFC misses, and 133,572 typed unsupported-shape records.
The aggregate applied-wait counter is an overlapping causal exposure sum, not
an elapsed-cycle total; residual same-cycle conflict was zero.

## Exact-identity residual deltas

The H3 arm does not improve the scored cache/DRAM metrics and slightly worsens
the two IPC endpoints on the complete exact-identity set. Values below are
mean APE; the H2 comparison is from the corrected H2 full corpus.

| Metric | H2 mean APE | H3 mean APE | Delta |
|---|---:|---:|---:|
| L2 lookup-hit sectors | 35.04% | 35.04% | 0.00 pp |
| L2 lookup-miss sectors | 27.00% | 27.00% | 0.00 pp |
| L2 source-total sectors | 28.24% | 28.24% | 0.00 pp |
| DRAM read sectors | 40.91% | 40.91% | 0.00 pp |
| `sm__cycles_active.avg` | 18.08% | 18.26% | +0.18 pp |
| `sm__inst_executed.avg.per_cycle_elapsed` | 12.41% | 12.53% | +0.12 pp |

Report artifact: `../sm86_replay_results/metric_report_h3_rf_20261015_v4/`.

## Verdict and next arm

H3 is implemented as a real conditional mechanism and retained for causal
coverage, but it is **not promoted as an accuracy improvement**: its complete-
corpus IPC residual is slightly worse and remains far above the sub-10% goal.
The next bounded arm is H4, separating persistent fetch-last-issued state from
issue hold across Yield instructions, using real SM86 control sidecars and
Huerta MICRO'25 fetch/decode evidence. H4 must be evaluated full-first with
H3 still present, then the combined residual decides whether the RF timing arm
remains in the default full composition.
