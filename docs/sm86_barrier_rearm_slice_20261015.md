# SM86 consecutive-BAR generation re-arm — 2026-10-15

## Mechanism and independent prior

H24 promotes a narrow SM86 CTA-barrier generation re-arm. The existing model
already admits barrier arrivals through one SM-shared server at two cycles per
warp; H24 adds a 28-cycle delay only before a later `BAR.SYNC` generation of
the same CTA. Ordinary post-barrier instructions and the first generation are
unchanged.

The bound comes from a fresh clock64 probe compiled and run on the local RTX
3090 Ti from the retained barrier-pipeline source in
`/home/jakeshea/GPU_sims_notes/GPU_simulator_snapshot/workloads/research_probes/barrier_pipeline/barrier_pipeline.cu`:

```text
b10, 4 warps, 16,384 generations: 688,134 cycles
b20, 4 warps, 16,384 generations: 1,146,885 cycles
difference: 458,751 cycles ≈ 28.000 cycles/generation
```

The probe passed its device-side checksum. This is a counter-free local
clock64 observation, not an NCU target or a fit to scored corpus workloads.

## Implementation

`sim-arch` now binds `CtaBarrierIssue.generation_rearm_cycles = 28` for SM86.
The existing `SmCore` barrier state applies the value only when all CTA warps
release one generation and the next generation is re-armed. Cache state, DRAM
traffic, memory admission, and L0C/RF state are untouched.

## Real-trace evidence

Two representative full-first batches completed without failures:

- Five ordinary barrier/control roots: `../sm86_replay_results/h24_barrier_rearm_20261015_v1/full.jsonl.zst`.
- Five high-BAR roots (`groupnorm`, `snapshot_059`, adapted group-topk,
  snapshot-SAM, and layernorm):
  `../sm86_replay_results/h24_barrier_rearm_20261015_v2/full.jsonl.zst`.

The high-BAR holdouts show live frontier deltas of 10–73 cycles; roots without
consecutive generations remain unchanged.

## Complete-corpus evidence

Full-first replay completed before fast replay with exactly twelve workers:

- 356 roots; 286 `ok`, 62 mixed-code controls, 8 replay/input failures.
- 12,392 full launches; full wall time 40.694 s.
- Artifact:
  `../sm86_replay_results/full_corpus_h24_barrier_rearm_20261015_v1/full.jsonl.zst`.
- Report:
  `../sm86_replay_results/metric_report_h24_barrier_rearm_20261015_v1/`.
- NCU was not a replay input; output JSONL is compressed.

Exact H23→H24 full-corpus deltas:

| Quantity | H23 | H24 | Delta |
|---|---:|---:|---:|
| L1 hits | 6,240,088 | 6,240,088 | 0 |
| L1 misses | 8,795,851 | 8,795,851 | 0 |
| L2 hits | 2,793,478 | 2,793,478 | 0 |
| DRAM read sectors | 4,824,390 | 4,824,390 | 0 |
| GPC elapsed frontier sum | 86,122,355 | 86,124,202 | +1,847 |
| terminal frontier sum | 64,014,636 | 64,016,483 | +1,847 |

Only 29 of 12,392 common launches move, with a maximum +482 cycles. The
post-replay mean APE changes are small but directional: GPC 12.90596%→12.90078%,
SM active 18.00767%→18.02024%, and elapsed IPC 12.31350%→12.30731%; L1/L2/
DRAM APE remains exactly unchanged (30.71/58.44/28.24/40.91%).

## Verdict and residual

H24 is promoted as a bounded source-backed barrier mechanism, with a small
elapsed-IPC improvement but a slight SM-active regression. It does not close the
sub-10% target. Keep the 28-cycle arm scoped to consecutive CTA generations;
do not turn it into a generic barrier or control stall. Remaining work should
address the dominant eligible-not-selected/front-end/L1TEX dependency residuals
or add a typed observer for the Huerta fixed-result queue without inventing
execution-to-writeback release timing.
