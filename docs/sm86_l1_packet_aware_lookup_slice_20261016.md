# SM86 packet-aware L1 lookup observer — H79 — 2026-10-16

## Mechanism

The L1 cache already records whether a pending merge belongs to the current
logical request or to an older request. H79 preserves the existing NCU-shaped
baseline (`hits + merges - line_alloc_misses`) and separately exports the
ordinary-global same-request merge population. A bounded candidate subtracts
only those ordinary same-request sector merges from the baseline hit projection:

```text
candidate_hits = max(0, baseline_hits - ordinary_global_same_request_merges)
candidate_misses = lookup_denominator - candidate_hits
interval_hits  = [candidate_hits, baseline_hits]
interval_misses = [baseline_misses, candidate_misses]
```

LDGSTS, constant-client, local, cache state, MSHR ownership, fill timing, L2,
DRAM service, and all frontiers remain unchanged. The candidate is emitted with
no NCU alias, so it cannot silently replace the baseline prediction.

## Implementation

`TraceMetrics` now retains ordinary-versus-async same-request merge counters
from `L1MergeAudit`. Native SM86 metrics expose candidate and interval rows for
global-load lookup hit/miss sectors. Checked arithmetic bounds the candidate by
the existing lookup denominator and preserves sector conservation.

## Full-first evidence

Release build and full-first replay completed on all 356 roots with twelve
workers:

* replay: `../sm86_replay_results/full_corpus_h79_packet_observer_20261016_v2/`
* report: `../sm86_replay_results/metric_report_h79_packet_observer_20261016_v2/`
* coverage: 286 `ok`, 62 controls, 8 replay errors; 12,392 full launches.

Using the independent NCU rows only for post-replay comparison, the 106
scope-compatible global-load joins show:

| endpoint | baseline mean APE | packet-aware candidate mean APE |
|---|---:|---:|
| L1 lookup hit (77 nonzero targets) | 30.71% | 17.50% |
| L1 lookup miss (106 targets) | 32.39% | 19.53% |

Zero-target false-positive hits fall from 22/29 to 8/29. The interval brackets
60/107 hit rows and 61/107 miss rows; this is useful uncertainty metadata but
not a high-confidence physical interval. On the authoritative raw/all-row
view, the scope-mismatched LTC128B row makes the miss candidate regress, so the
candidate is **not promoted** into the default NCU-shaped prediction. The
baseline remains the selected full-replay row, while the candidate remains an
advanced observer for future form-qualified crosswalks.

## Decision and next gate

This is a positive mechanism-local observer result, not overall accuracy
closure. Promotion requires a form-qualified ordinary-global mapping (or a
same-binary endpoint witness) that preserves raw/all-row behavior as well as
the secondary improvement. No NCU value was used as replay input, parameter
fit, or prediction selector.
