# SM86 write lookup packet composition — H81 — 2026-10-16

## Mechanism

H81 composes the validated H80 packet-sector candidate with the operation-split
write lookup rows. For the effective write operation endpoint, candidate
lookup hits are bounded by `min(baseline_write_hits, packet_sectors)` and
candidate misses are the checked packet-sector remainder. The original
request-domain write hits/misses remain emitted under diagnostic baseline keys.
Read, atomic, reduction, aggregate L2, DRAM, cache state, and timing paths are
unchanged. This is an effective source-operation model, not physical tag or
XBAR recovery.

## Full-first evidence

* replay: `../sm86_replay_results/full_corpus_h81_write_lookup_packet_20261016_v1/`
* report: `../sm86_replay_results/metric_report_h81_write_lookup_packet_20261016_v1/`
* coverage: 286 `ok`, 62 controls, 8 replay errors; 12,392 full launches with
  twelve workers.

The 107 identity joins report:

| endpoint | mean APE | P90 | exact nonzero/rows |
|---|---:|---:|---:|
| `op_write.sum` | 9.37% | 0.00% | 100/107 |
| `op_write_lookup_hit.sum` | **9.36%** | 0.00% | **100/107** |
| `op_write_lookup_miss.sum` | 0% (all zero-target) | 0% | 107/107 |

The prior H74 write-hit residual was ~39.9%; H81 reduces it below the 10%
mechanism target. Aggregate `lts__t_sectors_srcunit_tex` and all timing/IPC
metrics are bit-equivalent to H80.

## Boundary

The composition is enabled only for the NCU-shaped operation-split write
endpoint. It does not claim that packet count determines all hardware lookup
events; the H75 evidence against that generalization is retained. Any future
timing or physical write-queue model still requires a same-binary endpoint
witness.
