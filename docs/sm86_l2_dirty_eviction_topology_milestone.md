# Completed milestone: sm_86 L2 dirty-eviction topology

**Status:** complete 2026-09-01 as
`complete_exact_dirty_topology_no_write_service`  
**Implementation:** `0efdc981d3ba82931fe9588ceba55428da77987a`  
**Evidence:** `results/captures/sm86_l2_dirty_eviction_topology_20260901_v1`  
**Tracked summary:** `data/local_labels/sm86_l2_dirty_eviction_topology_report.json`

## Outcome

The shared L2 now carries exact per-sector dirty state and exposes the actual
replacement victim through its existing transactional pre-commit API. STORE,
RED, and ATOM write-class accesses make only their touched sectors valid and
dirty; read allocations and fills remain clean. A write that supersedes a
pending read becomes valid-and-dirty, and an old fill cannot resurrect or clean
an evicted allocation.

The system reports timing-neutral lower-level topology:

- evicted lines and valid sectors;
- evicted pending-read lines and sectors;
- dirty-eviction lines and sectors; and
- maximum terminal dirty-resident lines and sectors.

The terminal fields are gauges. Launch-set aggregation takes their maximum and
never sums them as traffic. Dirty victims are candidate writeback work, not
scheduled or completed DRAM requests.

No event, latency, bandwidth, queue, response, event horizon, issue timestamp,
cache lookup result, or completion frontier changed. STORE launches retain
`dirty_writeback=true` and `coherence=true`.

## Transactional state contract

Every L2 way maintains:

```text
dirty_mask subset-of valid_mask
dirty_mask disjoint-from pending_mask
```

`L2EventMaskPreview::eviction()` exposes at most one immutable victim with its
line id and valid/dirty/pending masks. The callback runs before cache mutation;
a rejected callback leaves validity, dirty state, recency, generation,
counters, and the caller-owned observer unchanged. Snapshots retain dirty state
exactly.

Focused tests cover cold/resident writes, partial dirty lines, clean fills,
pending-read supersession, stale completions, snapshot identity, callback
rollback, exact single-way victim accounting, terminal residency, and unchanged
dirty/coherence tails. The full simulator workspace passes 896 tests with three
machine-dependent tests ignored.

## Capacity/reuse evidence

The primary discriminator reuses four exact full traces from the completed
memory-retirement campaign. Unique-line and reuse cells use the same STORE
kernel specialization, grid, depth, lanes, allocation size, and H2D
initialization; only runtime address layout differs.

The carded L2 is 6,291,456 bytes: exactly 49,152 128-byte lines. Therefore the
empty-cache structural expectations are label-free.

| Cell | STORE inst | STORE sectors | Dirty victims | Terminal dirty sectors | Conservation |
|---|---:|---:|---:|---:|---:|
| B84/D16 unique, 5.25 MiB | 1,344 | 43,008 | 0 | 43,008 | 43,008 |
| B84/D64 unique, 21 MiB | 5,376 | 172,032 | 122,880 | 49,152 | 172,032 |
| B84/D16 reuse, 0.328 MiB | 1,344 | 43,008 | 0 | 2,688 | 2,688 |
| B84/D64 reuse, 0.328 MiB | 5,376 | 172,032 | 0 | 2,688 | 2,688 |

Every value matches exactly. All trace total/reported instruction counts match,
and unsupported, unmapped, and cross-allocation sidecars are empty. Each cell
has byte-identical current replay twins. Removing only the eight new topology
fields reproduces its retained predecessor JSON exactly, including issue and
terminal timestamps, cache traffic, and readiness diagnostics.

## Independent NCU comparison

Retained NCU `dram__sectors_write.sum` medians are directionally consistent:

| Cell | Median DRAM-write sectors | Repeat range |
|---|---:|---:|
| D16 unique | 80,780 | 79,764–87,684 |
| D64 unique | 208,028 | 157,248–209,288 |
| D16 reuse | 1,368 | 0–1,380 |
| D64 reuse | 1,232 | 0–1,604 |

These measurements were not used to define dirty state or expected victims.
They are not an exact oracle: the below-capacity D16 unique cell already writes
more DRAM sectors than target STORE sectors, reuse sometimes reports zero, and
NCU application replay/cache control plus implicit task-boundary flushing are
composite. This is precisely why no write latency or rate is fitted here.

## Reproduction

```bash
cargo build --manifest-path sim/Cargo.toml -p sim-run --release

.venv/bin/python \
  workloads/research_probes/memory_retirement_service/\
analyze_dirty_eviction_topology.py

.venv/bin/python \
  workloads/research_probes/memory_retirement_service/\
analyze_dirty_eviction_topology.py --validate-only

cargo test --manifest-path sim/Cargo.toml --workspace -q
```

The detailed report is SHA-256
`70fbfedf90ee56f7fc50489feed795e99b1405cd29dc3b28de201127d1617a97`;
the release `sim-run` is
`d923d0acf2f270e17efcd90fd5319f38937e62ee2240a1ff356950058949485e`.

## Scientific boundary and next action

The architecture card's `write_allocate_non_inclusive` token establishes the
implemented allocation rule; it does not independently prove a silicon
write-back policy. A dirty eviction names exact simulator work, not a DRAM
completion.

Retain the topology. The next bounded mechanism is a finite below-L2 write
service that consumes these exact dirty victims and terminal dirty residency.
Its rate, width, arbitration with reads, and implicit flush policy require an
independent discriminator; this milestone supplies none of those values and
authorizes no tail clearing.
