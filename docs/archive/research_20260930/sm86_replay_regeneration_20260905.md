# SM86 workload replay regeneration

Updated 2026-09-08.

The runnable regenerated cohorts are:

- `results/captures/sm86_regenerated_workloads_small_20260905_v1`
- `results/captures/sm86_regenerated_workloads_large_20260905_v1`

They contain the current static/dynamic split required by the simulator. The
active capture tree keeps the current/best comparison inputs.

The release command for each workload is:

```
sim/target/release/sim-run --trace <nvbit-root> --control <static-control> \
  --arch-card arch_profiles/cards/sm_86.json \
  --pchase-card data/pchase/sm_86_static_card.json \
  --mode l1-l2 --cache-scope launch --ncu-shaped-metrics
```

The large cohort additionally uses
`workloads/research_probes/l1tex_admission_backpressure/memory_write_completion_feedback_store_201.json`.

Validation:

- All 15 regenerated NVBit directories pass `trace_admission.py` with zero
  exit status and empty stderr with the active-tree layout.
- The complete retained 8-small/7-large release sweep remains 15/15 successful
  and deterministic. The only response differences versus the current comparison
  index are the two expected corrected unmodeled-DRAM
  `bandwidth_delayed_responses` gauges; primary instruction, cache, memory,
  issue, and terminal frontiers remain unchanged.
- A second 8-small/7-large sweep against the retained paths matched the
  retained response set with no changes.
- HMMA small replay twins exit zero with empty stderr and emit identical
  15,042-byte JSON. SwiGLU large feedback-enabled twins emit identical
  15,173-byte JSON with empty stderr.
- A fresh pinned-host native HMMA capture admits 422,400 instructions after
  the rebuilt tracer and replays twice with identical 15,034-byte JSON. Its
  capture warning is the known NVBit initialization/fallback diagnostic; the
  admission and replay stages are silent.
- C++ metadata probes accept all 15 regenerated execution documents plus the
  fresh capture and reject malformed fields, duplicate members, invalid shapes,
  invalid guards, and release-only missing lookups.
- The final recursive DOM string guard is included in that rebuilt parser pass;
  zero-length key storage cannot become a null-pointer string construction.
- Campaign-enabled `sim-trace` now passes 169 tests, including the default
  snapshot dynamic-trace and threadblock size boundaries; the valid replay set
  is unchanged.
- The release campaign runner also builds successfully against the bounded
  snapshot reader.
- Its retained address-carrying canary fixture completes with zero exit and
  empty stderr.

The native capture host is the expected RTX 3090 Ti, compute capability 8.6,
with CUDA 13.2 and the pinned tracer/protobuf toolchain. Fresh dynamic payloads
may vary while fixed-trace twin replays remain byte-identical. The NCU
regeneration remains a separate measurement boundary from simulator inputs.
