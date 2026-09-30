# Matched start/end timer around a raw SM86 QMD launch

The raw-RM continuation now has a device-side interval endpoint for the
bounded NAK body. This is a performance observation, not a test, hash, or
verification workflow.

## Construction

The external `libmc` compute method stream was extended with:

1. a four-word report-semaphore write at `sema+16` immediately before QMD
   scheduling;
2. `SEND_PCAS_A` / QMD dispatch;
3. explicit `WAIT_FOR_IDLE`;
4. a four-word report-semaphore write at `sema` after the QMD completes.

The four-word report format supplies a payload and the documented **nanosecond
GPU timestamp** (snapped from `NV_PTIMER_TIME_1/0`) as described by
the open [Ampere compute method definitions](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/615.71.09/src/common/sdk/nvidia/inc/class/clc7c0.h)
and [Ampere PBDMA semaphore documentation](https://github.com/NVIDIA/open-gpu-doc/blob/main/manuals/ampere/ga100/dev_pbdma.ref.txt).
The body was
the independent Mesa NAK SM86 shader launched by the adapted Ampere QMD V03.
The host also recorded `CLOCK_MONOTONIC_RAW` around the call.

## Observation

Thirty-two repeated raw-QMD launches completed. Excluding the first setup
launch:

| endpoint | min | median | max |
|---|---:|---:|---:|
| raw GPU-timer delta | 3,072 ns | 3,072 ns | 4,096 ns |
| host monotonic interval | 15,117 ns | 15,370 ns | 23,197 ns |

The GPU timestamp is documented in nanoseconds, but it is still not equated to
NCU/CUPTI/CUDA-event endpoint semantics. The host interval includes
direct-command construction and host polling; the GPU interval is bounded by
the explicit report/WFI method stream. The post-setup medians are 3.072 us GPU
versus 15.370 us host, a ratio of about 4.95x attributable to the host-side
submission/polling envelope in this small body.
The result is a new matched endpoint **within this raw-RM path**, not a
cross-tool unified timing endpoint.

Artifacts: `results/alpha/sm86_nak_direct_qmd_goal_20260930/timestamp_matched_wfi/`
(`summary.json` schema v2 records the documented nanosecond semantics).

## Boundary update

This narrows the timing blocker by separating command-submission overhead from
the raw GPU interval for one independently generated body. It still lacks a
public relation from this endpoint to NCU WaitForIdle, CUPTI activity, CUDA
events, simulator completion, and campaign wall time. It therefore cannot be
used as a drift-free universal timing conversion.
