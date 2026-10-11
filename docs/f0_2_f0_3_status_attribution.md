# F0.2/F0.3 — Typed envelope status and PC-sampling attribution

**Status:** alpha implementation complete (2026-10-10)

## Change

`tools/sm120_envelope_model.py` now:

* preserves each PC-sampling source row's capture, launch, device, stream,
  kernel, module/function, function index, correlation, PC, stall reason,
  collection mode, sample count, drop count, period, source line, raw columns,
  and parse errors;
* accepts lightweight CSV rows and CUPTI decoder JSON summaries
  (`pc_records`/`buffers`), retaining declared versus decoded sample/drop
  totals separately;
* refuses to infer launch `0` when the source row omits launch identity;
* reports unmatched/partial-identity rows and aggregate sample/drop counts;
* consumes F0.1 `simulator.per_launch[].feature_attribution` before any older
  by-kernel fallback, preventing capture-wide TMA totals from being copied to
  each launch;
* emits explicit statuses: `unavailable`, `not_modelled`, `predicated_off`,
  and `predicted_zero` for issue, memory-service, barrier, async-completion,
  visibility, and feature surfaces while retaining legacy scalar fields;
* retains both compact `pc_stall_samples` and lossless `pc_stall_records`.

## Bounded real alpha paths

1. Single FP32 launch replay:

```text
python3 tools/sm120_replay_bench.py \
  --work-root /dev/shm/sm120_rtx_5090_work \
  --capture postbuild_sm120_tracer_smoke_fp32 \
  --control-root /dev/shm/sm120_rtx_5090_work/p6_controls \
  --sm120-alpha --run \
  --out /dev/shm/sm120_rtx_5090_work/f0_alpha_replay.jsonl \
  --synthesize-control /dev/shm/sm120_rtx_5090_work/f0_sidecars
```

Completed with return code 0. Envelope output:
`/dev/shm/sm120_rtx_5090_work/f0_alpha_envelope.json`.

2. F0.1 two-launch TMA attribution:

```text
python3 tools/sm120_envelope_model.py \
  --input /dev/shm/sm120_rtx_5090_work/f0_1_per_launch.jsonl \
  --out /dev/shm/sm120_rtx_5090_work/f0_2_envelope.json
```

Each launch retains 10 TMA records and 65,536 bytes, proving that the
capture aggregate (20 records / 131,072 bytes) is not duplicated. Async and
visibility remain `not_modelled` because no native completion/consumer witness
exists. Missing PC sampling is explicitly `unavailable`, with zero rows and no
synthetic launch-zero join.

The same parser over the F0.1 all-capture campaign
`/dev/shm/sm120_rtx_5090_work/f0_1_all.jsonl` emits 29 launch rows. Launch-local
features sum to 2,216 TMA records and 8,542,208 TMA bytes. Async/visibility
rows are `not_modelled` for 24 launches with TMA records and `predicted_zero`
for five launches with an explicitly observed zero-TMA sidecar bucket; barrier
rows are `modeled` for 28 launches and `unavailable` for the one launch with no
barrier readiness surface.

## Limitations

* The alpha traces do not contain native TMA transaction-completion timestamps
  or destination-visibility witnesses; `not_modelled` is intentional.
* Existing p6 JSONL produced before F0.1 lacks launch-local sidecar buckets;
  the parser reports async evidence as `unavailable` rather than reusing
  capture-wide totals.
* No PC-sampling CSV was available in the retained work root, so only the
  schema/provenance path was exercised on real alpha data; no stall identity
  claim is made.
* No test suite or hashing/check workflow was run.
