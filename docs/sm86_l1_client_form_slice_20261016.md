# SM86 H94 per-client L1 lookup provenance — 2026-10-16

## Mechanism

H94 extends the existing L1 merge audit with trace-native same-request merge
counts partitioned by `OrdinaryGlobal`, `Constant`, and `AsyncCopy` client. The
selector uses the constant-client count for mixed constant/plain launches;
LTC128B and async mixed launches remain on the established H92 baseline.

This is observer/endpoint-only. It does not alter L1 tags, sector validity,
MSHR ownership, fill timing, L2 state, DRAM, or IPC.

## Evidence

A real four-root probe (including mixed constant/plain and constant/LDGSTS
launches) passed. Full-first replay then completed 356 roots (286 `ok`, 62
controls, 8 errors), 12,392 launches, twelve workers, and compressed artifacts:

* replay: `../sm86_replay_results/full_corpus_h94_l1_client_lookup_20261016_125135/`
* report: `../sm86_replay_results/metric_report_h94_l1_client_lookup_20261016_v1/`

Compared with H93, only one mixed-form launch changed and no other scored metric
changed. Scope-compatible exact-join APE moved:

| metric | H93 | H94 |
|---|---:|---:|
| L1 global lookup hit | 18.0702% | **17.7036%** |
| L1 global lookup miss | 18.8472% | **18.7117%** |

The gain is small but non-regressing. The independent U16 half-line probe
showed both plain and U16 `LDG.E.*.CONSTANT` forms produced 0 lookup hits and 8
miss sectors in the tested two-warp cold shape, so a universal U16 multiplier
was rejected. A full per-sector-validity selector still requires a broader
same-binary warm/cold probe matrix.
