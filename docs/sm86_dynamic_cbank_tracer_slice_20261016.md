# SM86 dynamic CBANK tracer repair — H77 — 2026-10-16

## Root cause

The rebase tracer's instrumentation only emitted `CONSTANT_MEM` records for
CBANK operands with a register/uniform-register offset. Plain immediate LDC/
ULDC forms were skipped, and the older enhanced protobuf lacked dynamic
constant payload fields. This left static SASS CBANK operands visible but no
runtime offset/bank data on the wire.

## Prototype repair

A local rebase-tracer prototype adds additive protobuf fields
`constant_memory_addresses` (tag 15), `constant_memory_bank` (tag 16), and
`constant_memory_offset` (tag 17); instruments immediate CBANK offsets;
preserves the static bank id; and emits active-lane constant offsets without
routing them through GLOBAL allocation identity. The replay-bench sidecar
adapter now accepts enhanced-execution-info metadata, derives checked control
words, and keeps the source provenance explicit.

The repaired tracer was rebuilt with CUDA 13.2/g++-14. Real mode-1 and mode-0
captures completed successfully at `/tmp/h77_gcc_capture_operands_v2/` and
`/tmp/h77_gcc_capture_mode0_v2/`; dynamic protobuf records contain fields
15–17. Replay through the adapter completed one full launch for each mode at
`/home/jakeshea/GPU_sims_notes/GPU_simulator_snapshot/workloads/research_probes/gcc_demand_h73/replay_h77_v4/full.jsonl.zst`.
The mode-1 launch reports 344,064 dynamic LDC records, 2,688 ULDC records,
11,526,144 constant addresses, 147 unique constant sectors, and zero missing
metadata records. The mode-0 launch is a valid no-LDC control and reports
2,688 ULDC records, 10,752 store sectors, and no metadata rejection. Its
previous control-error was caused solely by the runner looking for the absent
`native_static_metadata.json`; the adapter now falls back narrowly to
`enhanced_execution_info.json`.

The required full-first corpus replay after this loader repair completed 356
roots (286 `ok`, 62 `control_error`, 8 `replay_error`), 12,392 full launches,
and exactly 12 workers at
`../sm86_replay_results/full_corpus_h77_adapter_20261016_v2/`. The compressed
report is
`../sm86_replay_results/metric_report_h77_adapter_20261016_v2/`; coverage and
all H76 scored residuals are unchanged (GPC max 12.9008%, SM-active average
18.0202%, L1 lookup-hit 30.7137%, DRAM read 26.8371% secondary APE).

## Boundary and decision

This repair is a prototype in the rebase tracer/notes workspace; the active
3090Ti branch has no checked-in NVBit tracer tree, so no current-branch tracer
binary or full-corpus simulator behavior was changed. The new fields are
observer-only and do not become GCC request counts, L2 sectors, or DRAM timing.
A checked-in tracer transport integration is still required before landing the
producer change in the branch; the active branch intentionally has no NVBit
source tree, so the rebase implementation remains an external prototype.
