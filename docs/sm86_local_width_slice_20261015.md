# SM86 local-memory width observer — 2026-10-15

## Scope

H14 retains producer `data_width` for local-memory records and emits a
width-aware source candidate: record count, unknown-width count, summed bytes,
and predicate-lane × width converted to 32-B candidate sectors. It does not
replace the validated coalesced sector stream, alter L1/L2 state, or invent a
local-memory timing constant.

## Real-trace evidence

Five local/vector roots completed full replay 5/5. Register-spill,
local-volatile, and local-dynamic-index roots all carried producer widths;
vectorized global roots correctly emitted zero local records. The complete
full-first corpus completed 356 roots (286 `ok`, 62 controls, 8 failures),
12,392 launches in 35.671 s with 12 workers:
`../sm86_replay_results/full_corpus_h14_local_width_20261015_v1/full.jsonl.zst`.

Across successful rows H14 observed 4,892 local-width records, zero unknown
widths, 19,824 source bytes, and 19,824 candidate sectors.

## Residuals and verdict

Exact DRAM/L1/L2/IPC residuals are unchanged from H10/H11 (DRAM 40.91%, active
IPC 18.19%, elapsed IPC 12.42%; cache rows unchanged). Report:
`../sm86_replay_results/metric_report_h14_local_width_20261015_v1/`.

The width observer closes a trace-contract gap but is not an accuracy gain;
local-memory residuals are sparse in exact NCU joins and do not explain the
corpus-wide DRAM error. Keep the source metrics and continue to the next
mechanism-local L1/IPC arm.
