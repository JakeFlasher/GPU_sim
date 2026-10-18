# SM86 GCC/L1.5 demand observer — H66 — 2026-10-16

## Independent source-client probe

A zero-global-data/constant-load CUDA probe was rebuilt for the identified RTX
3090 Ti and profiled with NCU. At 84 blocks, 1,024 threads, and 128 loop
iterations, the no-constant control still exposed GCC instruction/constant
requests and DRAM reads:

| mode | GCC requests | constant requests | instruction requests | GCC L2 sectors | GCC L2 misses | DRAM reads |
|---|---:|---:|---:|---:|---:|---:|
| arithmetic control | 1,764 | 1,008 | 791 | 952 | 125 | 3,100 |
| constant load | 562,919 | 562,212 | 742 | 1,960 | 269 | 276 |

This proves that `dram__sectors_read.sum` has a GCC/L1.5 client distinct from
TEX data traffic, but the probe does not provide a trace-to-NCU launch crosswalk
for the scored corpus and therefore does not justify timing admission.

## Simulator observer slice

Full mode now emits three observer-only native rows from existing trace-domain
information:

- `dram_gcc_instruction_sector_candidate`
- `dram_gcc_constant_sector_candidate`
- `dram_read_demand_multiclient_candidate_low`

They combine compulsory instruction-sector and constant-domain candidates with
TEX L2-miss demand but never enter L2 state, DRAM service, or frontiers. A real
full-first corpus replay completed 356 roots (286 `ok`, 62 controls, 8
failures), 12,392 launches with 12 workers and compressed artifacts:

`../sm86_replay_results/full_corpus_h66_gcc_demand_observer_20261016_v1/`

Post-replay report:
`../sm86_replay_results/metric_report_h66_gcc_demand_observer_20261016_v1/`.
The observer is not NCU-scored and raw H41 APE remains authoritative.

## Decision

Retain the GCC candidates as source-domain evidence. Before any multi-client
DRAM timing or source-client L2 state is enabled, capture a same-binary NVBit
trace with static/dynamic GCC scope identity and compare it to the independent
NCU GCC/TEX surfaces. Do not add GCC candidates to `DramStats.read_sectors` or
alter aggregate `DramBw::Carded` timing from this observer alone.
