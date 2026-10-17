# SM86 H93 aggregate L2 TEX packet conservation — 2026-10-16

## Mechanism

H93 wires the already promoted H80/H81 per-SM `(line, sector)` write-packet
candidate into the aggregate SM86 TEX sector and lookup-hit endpoints. The
aggregate miss row remains the checked subtraction `sectors - hits`.

For SM86 only:

```text
TEX sectors = read requests + packetized write sectors + atom requests + RED requests
TEX hits    = read hits + min(write hits, packetized write sectors) + atom hits + RED hits
TEX misses  = TEX sectors - TEX hits
```

The uncoalesced store request/hit totals remain in diagnostic native rows. L2
tags, dirty state, DRAM demand/service, write acknowledgements, cycles, and IPC
are unchanged.

## Evidence

Release build and a four-root real full replay passed. Complete full-first replay
then completed 356 roots (286 `ok`, 62 controls, 8 replay errors), 12,392 full
launches, exactly twelve workers, and compressed full/fast artifacts:

* replay: `../sm86_replay_results/full_corpus_h93_l2_packet_conservation_20261016_123337/`
* report: `../sm86_replay_results/metric_report_h93_l2_packet_conservation_20261016_v1/`

The H93 report retains the H92 coverage and changes only the aggregate L2 TEX
sector/hit projections. Strict scope-compatible exact-join APE changed:

| metric | H92 | H93 |
|---|---:|---:|
| `lts__t_sectors_srcunit_tex.sum` | 6.7551% | **2.9180%** |
| `lts__t_sectors_srcunit_tex_lookup_hit.sum` | 11.7747% | **3.8618%** |
| `lts__t_sectors_srcunit_tex_lookup_miss.sum` | 0.0000% | **0.0000%** |

No other scored metric changed. This is an accepted observer-only promotion:
both aggregate L2 side metrics are below 10%, with exact miss conservation and
no timing or IPC regression.

## Boundary

H93 does not claim that the effective write packet is physical XBAR topology.
The next L1 slice requires a same-binary U16/partial-line probe and per-request
form identity; no launch-level heuristic is extended to LTC128B or mixed forms.
