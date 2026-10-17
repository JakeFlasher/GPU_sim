# SM86 constant-client dynamic interval observer — H76 — 2026-10-16

## Implementation

The native trace metrics now retain dynamic static-SASS populations for
`LDC` and `ULDC` (`constant_client_ldc_instructions` and
`constant_client_uldc_instructions`). These counters are intentionally
separate from CBANK address-domain fields (`constant_memory_unique_sectors`,
metadata-missing records, and constant `LDG` form observers). They do not
pretend that dynamic instruction count equals GCC requests, GCC sectors, or a
DRAM fill.

## Rebase-tracer evidence

The successful rebase-tracer captures in
`/home/jakeshea/GPU_sims_notes/GPU_simulator_snapshot/workloads/research_probes/gcc_demand_h73/`
contain 84 threadblock protobufs per mode. Dynamic records expose only STG.E
memory classes and no constant-bank addresses; joining dynamic PCs to static
SASS recovers 16,128 CBANK-tagged records in the arithmetic mode and 360,192
in the constant-load mode (344,064 user-LDC records plus setup). NCU reports
GCC constant/instruction requests 1,008/791 and 562,212/742 respectively. The
populations are not endpoint-equivalent, so they remain interval evidence only.

## Full-first result

The simulator source was rebuilt and replayed full before fast with exactly 12
workers and compressed output:

`../sm86_replay_results/full_corpus_h76_constant_client_interval_20261016_v1/`

Coverage remains 356 roots (286 `ok`, 62 controls, 8 replay errors) and 12,392
full launches. The post-replay report is
`../sm86_replay_results/metric_report_h76_constant_client_interval_20261016_v1/`.
The observer rows are trace-native diagnostics and intentionally have no NCU
aliases or scoring entries; existing DRAM/GCC candidates, L2 state, service,
and timing remain unchanged.

## Decision

Promote only the dynamic LDC/ULDC population observer. Do not collapse it into
GCC request/sector counts or admit it to DRAM timing. Future promotion requires
repaired dynamic CBANK address fields and a held-out source-client crosswalk.
