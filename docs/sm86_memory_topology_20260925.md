# SM86 decoded memory-topology domain — 2026-09-25

The native replay now retains three topology counters for every decoded global
or local request class:

* `unique_sectors`: distinct 32-byte sector IDs in the request spans;
* `unique_lines`: distinct 128-byte line IDs (four sectors per line); and
* `sector_aliases`: span entries beyond the distinct-sector set.  This is
  useful for same-word ATOM/RED aliases, where the issue model intentionally
  retains serialized multiplicity in the ordinary `sectors` count.

The existing `requests` and `sectors` fields are unchanged.  `sectors` is the
transaction-oriented span used by the replay, while the new fields are a
trace-topology view.  They must not be read as NCU L1/L2 transaction or
coherence counters: cache allocation, replay, protocol actions, and temporal
ordering can cause hardware counts to differ.  Flat replay exposes these rows
as `unavailable`, rather than treating the absent address arena as zero.

## Retained SM86 holdout

The current release binary was replayed against the retained NVBit traces for
three workloads, plus the existing small ATOM probe.  The values below are
trace-derived and reproducible from the local replay artifacts under
`results/captures/sm86_metric_review_20260925_v1/coalescing_current/`.

| workload/class | requests | sectors | unique sectors | unique lines | aliases |
| --- | ---: | ---: | ---: | ---: | ---: |
| GEMM async global load | 16,384 | 196,608 | 196,608 | 196,608 | 0 |
| GEMM global store | 2,048 | 16,384 | 16,384 | 16,384 | 0 |
| RMSNorm global load | 24,576 | 49,152 | 49,152 | 24,576 | 0 |
| RMSNorm global store | 8,192 | 16,384 | 16,384 | 8,192 | 0 |
| SiLU global load | 8,192 | 16,384 | 16,384 | 8,192 | 0 |
| SiLU global store | 8,192 | 16,384 | 16,384 | 8,192 | 0 |
| small ATOM global atomic | 8 | 32 | 32 | 8 | 0 |

The zero alias values are an observation about these particular traces, not a
claim that all atomics are alias-free.  The unit test exercises a repeated
sector span and reports one alias.  No NCU value is used to derive or tune the
new counters.

## Boundary and next use

The domain is intentionally narrower than a complete memory-system model.  It
is sufficient to compare address coalescing, line fan-out, and alias exposure
across architecture cards or custom traces, and it supplies a bounded input to
future cache/coherence mechanisms.  It does not reconstruct dynamic physical
pipeline assignment, invalidations, writeback timing, DRAM interval traffic,
or value-dependent address regeneration.
