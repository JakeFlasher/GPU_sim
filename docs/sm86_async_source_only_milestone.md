# SM86 store-free async-source discriminator

**Status:** diagnostic complete 2026-08-30; no simulator timing change
**Architecture:** `sm_86` / NVIDIA GeForce RTX 3090 Ti (device 0)
**Predecessor:** [SM86 simulator live baseline and core closure](../plans/sm86_simulator_live_baseline_core_closure.md)
**Evidence root:** `results/captures/sm86_ldgsts_burst_release_20260830_v4/`

## Purpose

The current simulator reports dirty-writeback/coherence tails for ordinary
global stores and a separate `async_copy_shared_visibility` boundary for some
LDGSTS bypass traces. This probe removes the target-kernel global-store and
CTA-barrier confounds while retaining real global-to-shared async copies and a
post-wait volatile shared read. It asks whether the remaining completion
frontier is attributable to async source/destination handling alone.

## Construction and commands

`workloads/research_probes/async_source_only/async_source_only.cu` builds ten
SM86 depth/route specializations. The target is one 32-thread warp, 4 KiB
shared destination, eight possible `cp.async` positions, one commit and
`wait_group 0`, and no target-kernel global store, atomic, reduction, or CTA
barrier. Host source initialization is outside the profiler marker.

```bash
make -C workloads/research_probes/async_source_only clean all
CUDA_VISIBLE_DEVICES=0 workloads/research_probes/async_source_only/async_source_only \
  --groups 64 --depth 4 --route l1 --markers on
CUDA_VISIBLE_DEVICES=0 workloads/research_probes/async_source_only/async_source_only \
  --groups 64 --depth 4 --route dram --markers on
```

The static sidecar reports 10 functions, 464 decoded instructions, and zero
unknown encodings; the target functions contain no `STG`, `RED`, `ATOM`, or
`BAR` instructions. Marker-scoped retained-runtime captures were made for
depths 1, 4, and 8 in both routes. Each complete root has one payload and one
accepted launch. Three NCU application-replay profiles were collected for each
of the six cells with `--clock-control none --cache-control none`; NCU remains
a comparison target and is not passed to `sim-run`.

## Observed frontiers

The table reports the median of the three live NCU profiles and the terminal
modeled event from one deterministic simulator replay. These are separate
frontiers, not a summed score.

| route | depth | NCU SM active | NCU GPC elapsed | simulator terminal | simulator frontier | typed tails |
|---|---:|---:|---:|---:|---|---|
| L1 (`.ca`) | 1 | 6,844 | 8,671 | 5,020 | complete | none |
| L1 (`.ca`) | 4 | 8,410 | 10,238 | 7,450 | complete | none |
| L1 (`.ca`) | 8 | 10,573 | 12,397 | 10,527 | complete | none |
| DRAM/bypass (`.cg`) | 1 | 19,119 | 20,940 | 30,740 | incomplete lower bound | async-copy shared visibility |
| DRAM/bypass (`.cg`) | 4 | 34,266 | 36,137 | 34,642 | incomplete lower bound | async-copy shared visibility |
| DRAM/bypass (`.cg`) | 8 | 37,964 | 39,837 | 39,827 | incomplete lower bound | async-copy shared visibility |

Traffic and instruction observations are internally consistent across the
pair: depth 1 has 64 LDGSTS warp instructions, 1,024 source sectors, and 256
shared-destination wavefronts; depth 4 has 256 instructions, 4,096 source
sectors, and 1,024 wavefronts; depth 8 doubles the depth-4 quantities. L1 route
labels show 4,032 hits/64 misses at depth 4; DRAM labels show 4,096 L1 misses
and 4,096 modeled DRAM read sectors. Replay twins are byte-identical for every
captured cell.

## Decision

The L1 source-only traces reach a complete simulator frontier with no
dirty-writeback, coherence, or async-visibility tail. The bypass/DRAM traces
retain the existing conservative `async_copy_shared_visibility` flag even
though the wire payload carries an ordered global source and shared destination;
the current replay code intentionally does not apply the cache-all destination
service rule to bypass copies. Across depths, the bypass lower bound is already
at or above the live SM-active value, so applying that cache-all service as a
new default is not supported by this evidence. The NCU spread (about 2.8–3.0%
for the DRAM cells) is another reason not to fit a latency.

This is a reproducible **no-go for a simulator timing patch**. The next useful
experiment is a route-controlled bypass destination/visibility witness (or a
tracer contract that independently timestamps shared delivery), with fixed
clocks and enough repeats to separate source response from destination
visibility. Until that exists, keep the current typed lower-bound behavior and
do not promote a bypass destination constant. JADE/HETR, CPU-GPU co-simulation,
`sm_120`, and rejected candidates remain unchanged.

## Reproducibility identities

- Probe binary SHA-256: `d196d98e76007716dc2c998c1ff2e422c7e5a627457457c1cd2798036d3cc88b`
- Source-only control sidecar SHA-256: `d7770c9a0cbe8175d69a8b42cb03263b87cd2e1c442d7ae0bdfb062e9c09ca78`
- L1 depth-1 trace SHA-256: `5b2386b3f477415efc2b1c10ebb5ef384f8d68cecca85aa735d61543b7cc6457`
- DRAM depth-1 trace SHA-256: `1824ff3ab3c3ba1bd1cb4e615dd3c13f8694d97413edef4904fde5658323e635`
- L1 depth-4 trace SHA-256: `eb60c6bc17e66d700c22001cfe840042cbcdbba5371753e0e451958e314f665e`
- DRAM depth-4 trace SHA-256: `aafae75d5c522fff7a45604222b12040dd831f185ebcbc0fad62c03d08b9bf3d`
- L1 depth-8 trace SHA-256: `e60f7ec6692877514d89f7d6d19b6bcf40903b18d26478b340e8f7f8f8f2f03d`
- DRAM depth-8 trace SHA-256: `17306c0601dcf8cd969d3e2d269e0da7ce14b8b84f3accd6b72cf51ef1deb11b`
- Release `sim-run` SHA-256: `6cb0d05f0d447c0cd6ee2e30353c0f1e89d464462458912478c826860f15f9a4`
