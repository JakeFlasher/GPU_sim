# SM86 LDGSTS/LTC128B scope crosswalk — H72 — 2026-10-16

## Implemented observer seam

The native trace metrics now retain bounded static-SASS scope partitions in
`trace_metrics.memory_scope` without changing any cache, DRAM, response, or
issue state:

* `global_load_ldg` — plain `LDG.E` width/form records;
* `global_load_constant` — `LDG.*.CONSTANT` records;
* `global_load_ltc128b` — any `LDG` form carrying `.LTC128B`;
* `global_load_strong_gpu` — explicit STRONG.GPU LDG route;
* `ldgsts_async_copy` — dynamic LDGSTS source records.

SM86 native metrics expose plain-LDG and LDGSTS opcode-count candidates. The
post-replay scope reporter compares those narrow candidates to their matching
NCU opcode endpoints. A plain opcode mismatch invalidates only the opcode
metric. A large LTC128B scope mismatch additionally invalidates the L1 global
sector/lookup endpoints for the secondary like-for-like view, while broad
hierarchy traffic remains unchanged. No NCU value is used as a replay input or
prediction selector.

## Independent crosswalk evidence

Two real sidecar/NCU controls bound the interpretation:

* `cp_async_shared_tile_copy`: 32 LDGSTS records, 128 sectors; NCU reports
  `smsp__inst_executed_op_ldgsts.sum=32`, global-LD opcode count zero, and L1
  global-load requests/sectors 32/128. Therefore LDGSTS must remain in the
  documented L1TEX global-op sector surface even though its opcode count is
  separate.
* `sm86_adapted_l2_074_sam_iou_backward_sm120`: 1,024 effective
  `LDG.E.LTC128B.CONSTANT` records; NCU reports global-LD requests/sectors
  16/40, L1 lookup misses 40, TEX read sectors 40, and separate GCC constant /
  instruction demand (160/928 requests; 3,808 GCC sectors). This is the
  previously observed 64× scope outlier and is classified as an LTC128B scope
  mismatch only for post-replay comparison.

Vendor-SGEMM and fused snapshot controls contain active LTC128B forms whose NCU
TEX global requests/sectors remain comparable (for example 64/256 and
9,216/36,864). A universal “subtract CONSTANT/LTC128B” rule is therefore not
valid; the implementation keeps form partitions observer-only.

## Full-first corpus evidence

Release `sim-run` was rebuilt and the complete corpus replayed full before fast
with exactly twelve workers and compressed artifacts:

`../sm86_replay_results/full_corpus_h72_ldgsts_scope_20261016_v1/`

Coverage is unchanged at 356 roots (286 `ok`, 62 controls, 8 replay errors) and
12,392 full launches. The scope report is:

`../sm86_replay_results/metric_report_h72_ldgsts_scope_20261016_v2/`

The report retains 132 post-replay scope warnings (124 operation-scope, seven
plain opcode, one LTC128B scope) and 68 secondary exclusions. The H74 LTC128B
outlier is the sole new LTC-specific exclusion. Secondary L1 metrics remain
like-for-like at approximately:

* global-load sectors: 0.21% mean APE (105/106 exact);
* global-load lookup hits: 30.71% mean APE;
* global-load lookup misses: 32.39% mean APE;
* LDGSTS opcode count: 0.0% mean APE (107/107 exact).

The authoritative raw/all-row view remains retained separately and is not
relabelled as improved by this scope diagnostic.

## Decision

H72 promotes the static form/LDGSTS scope observer and the post-replay
crosswalk only. It does **not** promote LTC128B 128-byte DRAM-fill timing,
relocate GCC/TEX demand, or alter broad L1/L2/DRAM counters. Future timing or
counter relocation requires an endpoint-matched NVBit/SASS crosswalk that
covers LTC128B.CONSTANT, STRONG.GPU, vendor kernels, and LDGSTS controls in the
same binary family.
