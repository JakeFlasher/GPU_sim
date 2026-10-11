# F0.1 — Per-launch feature attribution

**Status:** complete (alpha driver seam)  
**Date:** 2026-10-10  
**Driver:** `tools/sm120_replay_bench.py`

## Change

`async_sidecar_observation()` now streams `tma_trace.jsonl` and
`sync_trace.jsonl` instead of materializing every row.  Sidecar rows are
aggregated both capture-wide and by their tracer `kernel_id`.  The simulator
JSONL record retains the capture aggregate for provenance, while each
`simulator.per_launch[]` row receives only the matching kernel bucket under
`feature_attribution.sidecar`.

Each per-launch feature row also carries launch-local trace instruction/opcode,
memory-request, decoded-sector and NCU metric surfaces.  `fast_envelope_rows()`
uses the kernel-specific sidecar bucket rather than repeating capture totals.
Output writing was already line-buffered; this path continues to flush one
JSONL capture record at a time.

## Bounded real alpha replay

Command (one two-launch capture, existing control sidecar):

```text
timeout 120s python3 tools/sm120_replay_bench.py \
  --work-root /dev/shm/sm120_rtx_5090_work \
  --sim-bin sim/target/release/sim-run \
  --arch-card arch_profiles/cards/sm_120-rtx-5090.json \
  --control-root /dev/shm/sm120_rtx_5090_work/m11_sidecars \
  --capture cutlass_fp8_bf16_blockwise_gemm_sm120 \
  --sm120-alpha --run \
  --out /dev/shm/sm120_rtx_5090_work/f0_1_per_launch.jsonl \
  --scorecard-out /dev/shm/sm120_rtx_5090_work/f0_1_per_launch.scorecard.json
```

Result: exit status `0`, two simulator launches, and one streamed JSONL
capture record.  The capture aggregate contains 20 TMA records, 2,208 sync
records and 131,072 TMA bytes.  Per-launch attribution is distinct:

| kernel ID | trace warp instructions | TMA records | sync records | TMA bytes |
|---:|---:|---:|---:|---:|
| 1 | 10,501 | 10 | 1,155 | 65,536 |
| 2 | 10,477 | 10 | 1,053 | 65,536 |

The generated scorecard's two fast-envelope rows carry the same split (10/10
TMA records and 65,536 bytes each) rather than repeating 20/2,208/131,072 in
both rows.

The same bounded driver was then run over all 12 retained captures (29
simulator launches) and emitted
`/dev/shm/sm120_rtx_5090_work/f0_1_all.jsonl`.  Summing the launch-local
features exactly reconstructs the capture aggregates: 2,216 TMA records,
536,954 sync records, and 8,542,208 TMA bytes; no attribution mismatches were
observed in the artifact.  The companion scorecard is
`/dev/shm/sm120_rtx_5090_work/f0_1_all.scorecard.json`.

## Limitations

* Sidecars expose launch/kernel IDs, but no native completion timestamp or
  destination-visibility witness; those remain explicitly unobserved.
* The NCU row is still a profiler-perturbed observation and ordered fallback
  joins remain diagnostic when kernel names are mangled differently.
* This fixes attribution and memory behavior only; it does not change the
  provisional SM120 alpha timing model or infer physical cache/TMA topology.
