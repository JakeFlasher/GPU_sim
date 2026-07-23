# sm_120 frozen-roster L1/L2 collection pilot

Live NCU L1/L2 sector-hit-rate collection on the RTX 5060 Laptop GPU (sm_120) for
the **frozen tri-label roster cells** at their exact `(slug, shape_point)`
identity — growing AC-3's frozen-roster L1/L2 coverage (Codex Path A). NCU was
invoked only through `tools/local_labels/label_driver.sh` into
`data/local_labels/` (the Rule-1 local-label exception).

**Scope:** the 15 uncollected frozen-roster cells whose workload-arg convention
is known (portable/l1 tree: `--correctness_shape` / `--timing_shape` /
`--shape-point trace_l2_accuracy_lg|xl`). The other 30 uncollected roster cells
are the new Blackwell cluster_*/conformer_* `_00`-shape slugs whose arg
convention is undocumented in the roster — a per-slug arg archaeology deferred.

**Live provenance:** RTX 5060 Laptop GPU (sm_120), driver 595.71.05, CUDA 13.2,
NCU 2026.1.1.0, `--clock-control none` (Rule 2), N=5 repeats. Two configs tried:
cold (`warmup=0 cache-control=all`) and warm (`warmup=5 cache-control=none`).

## Result: 3 of 15 admitted; 12 fail repeatability

ADMITTED (`label_admitted.json`, schema `local_label_admitted.schema.json`,
stddev_l2 <= 0.03, n=5, `evidence_kind=live`) — all `trace_l2_accuracy_lg`:

| cell | L2 sector hit % | L1TEX % | stddev_l2 | config |
|------|-----------------|---------|-----------|--------|
| conv2d_implicit_gemm_bf16_sm80 / trace_l2_accuracy_lg | 69.62 | 50.06 | 0.0223 | cold |
| cross_entropy_fwd_fp32_sm80 / trace_l2_accuracy_lg | 3.42 | 49.97 | 0.0157 | cold (warm even tighter: 2.33, stddev 0.0019) |
| embedding_gather_int32_sm80 / trace_l2_accuracy_lg | 21.30 | 24.89 | 0.0151 | cold |

These are NEW frozen-roster cells → frozen-roster sm_120 L1/L2 coverage **9 → 12**.

EXCLUDED (`label_excluded.json`, `label_repeatability_failed`, 12 cells): the
`trace_l2_accuracy_xl`, `correctness_shape`, and `timing_shape` cells of
bf16_hmma_gemm_traceable, conv2d, cp_async_shared_tile_copy, dram_stream_stride_mix,
embedding_gather, flash_attn_bwd, layernorm — their L2 sector hit rate exceeds the
0.03 stddev gate at BOTH the cold and warm configs (a genuine sm_120 L2
label-stability issue, not a collection error). The per-repeat raw sidecars for
the excluded cells were pruned; their results are retained in
`label_excluded.json` + `label_repeatability.json`.

## Implication for AC-3 (see the round summary's Goal Tracker Update Request)

Of the 45 uncollected frozen-roster cells: 15 are known-convention (this pilot
admitted 3, 12 fail repeatability) and 30 need per-slug arg archaeology + unstaged
traces. Reaching >=50 admitted frozen-roster cells looks infeasible at the frozen
grain — which favors the AC-3 engineering-closure path (Path B) that preserves
the sm_120 tri-label grain and records the roster as not satisfied, over a
from-scratch >=50 re-collection. The raw `run_*.ncu-rep` reports are gitignored;
only parsed sidecars + aggregates are committed.
