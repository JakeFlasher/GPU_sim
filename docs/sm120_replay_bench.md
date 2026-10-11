# SM120 replay and fast-envelope driver

The active plan is
`/root/gpu_cache_memory_modeling_proposal/PLAN.md`.

## Inputs

```text
/dev/shm/sm120_rtx_5090_work/captures/<capture>/dynamic_trace.pb
/dev/shm/sm120_rtx_5090_work/captures/<capture>/extra_info/native_static_metadata.json
/dev/shm/sm120_rtx_5090_work/captures/<capture>/extra_info/tma_trace.jsonl
/dev/shm/sm120_rtx_5090_work/captures/<capture>/extra_info/sync_trace.jsonl
/dev/shm/sm120_rtx_5090_work/profiles/<capture>/raw.csv
/dev/shm/sm120_rtx_5090_work/m11_hardware_timing.json
```

## Alpha replay

```bash
python3 tools/sm120_replay_bench.py \
  --work-root /dev/shm/sm120_rtx_5090_work \
  --sim-bin sim/target/release/sim-run \
  --arch-card arch_profiles/cards/sm_120-rtx-5090.json \
  --hardware-timing /dev/shm/sm120_rtx_5090_work/m11_hardware_timing.json \
  --synthesize-control /dev/shm/sm120_rtx_5090_work/p6_controls \
  --run --sm120-alpha \
  --out /dev/shm/sm120_rtx_5090_work/p6_alpha_results.jsonl \
  --scorecard-out /dev/shm/sm120_rtx_5090_work/p6_fast_scorecard.json
```

The driver reports structural evidence, raw NCU rows, uninstrumented timing,
simulator phase timers, issue/service/completion ledgers, and grouped feature
accounting. The scorecard is descriptive development-corpus accounting: the
retained captures are not alphabetically split into calibration/holdout groups.
Replay speed is emitted only by this replay driver from its declared hardware
and simulator timers; the feature evaluator does not copy or aggregate it.
Sidecar descriptor byte counts are metadata observations, never destination
payload or visibility witnesses. External detailed cycle-simulator comparison
is permanently banned from alpha.

The executable fast stage path is:

```bash
python3 tools/sm120_replay_bench.py \
  --work-root /dev/shm/sm120_rtx_5090_work \
  --control-root /dev/shm/sm120_rtx_5090_work/p6_controls \
  --hardware-timing /dev/shm/sm120_rtx_5090_work/m11_hardware_timing.json \
  --run --sm120-alpha --fast-frontier \
  --out /dev/shm/sm120_rtx_5090_work/f9_final/fast_results.jsonl
```

Direct feature/service/async tools are `sm120_trace_features.py`,
`sm120_memory_service_model.py`, `sm120_dependency_features.py`, and
`sm120_async_state_model.py`; their retained outputs and the professional
closeout are linked from `reports/FINAL_RESULTS.md` in the active proposal.

## Fast evaluator

```bash
python3 tools/sm120_envelope_model.py \
  --input /dev/shm/sm120_rtx_5090_work/p6_alpha_results.jsonl \
  --out /dev/shm/sm120_rtx_5090_work/p6_fast_stage_envelope.json
```

Optional CUPTI PC-stall CSV rows use `capture_name`, `launch_index`,
`stall_reason`, and `samples` columns. Missing PC sampling is reported as a
typed feature gap; it is not replaced with NCU duration or simulator cycles.
Alpha issue fields are the exact `Sm120AlphaConfig` values consumed by the
executor, not confidence intervals or physical scheduler labels.
