# SM86 long-body clock-lock timing crosswalk

**Date:** 2026-09-30  
**Hardware:** RTX 3090 Ti / SM86, driver 615.71.09, external clocks locked
1350/10501 MHz.

## Experiment

The existing common-device-timer body was lengthened from 135,000 to 1,350,000
`%clock64` ticks. A semantic-equivalent timer body was run through; the raw-QMD extracted
header differs from the Driver text at control words in the store/tail region.
The operation sequence and launch envelope are matched, but byte identity is
not claimed.

The body was run through:

* independent raw Ampere QMD V03/libmc;
* CUDA Driver API;
* CUPTI concurrent-kernel activity and CUDA events;
* Nsight Compute with `--clock-control none`;
* Nsight Compute with default `--clock-control boost`.

The long interval makes launch/profiler tails a small residual rather than the
majority of the span.

| endpoint | median span |
|---|---:|
| raw QMD `%globaltimer` | 1,000,448 ns |
| CUDA Driver `%globaltimer` | 1,000,448 ns |
| CUPTI activity body | 1,000,032 ns |
| CUDA event | 1,004,528 ns |
| NCU `--clock-control none` | 1,001,600 ns |
| NCU default boost control | 715,824 ns |

Artifacts:
`results/alpha/sm86_common_timer_crosswalk_20260930/long_clock_locked_20260930/`.

## Interpretation

With external `nvidia-smi` locks and explicit `ncu --clock-control none`, raw
QMD, Driver, CUPTI, CUDA-event, and NCU durations agree within about 0.4% for
this long body. This is a materially stronger drift-control result than the
short-kernel crosswalk and gives a practical endpoint transform for a declared
clock policy.

The default NCU boost policy changes the measured body to about 715.8 us,
roughly 28.5% below the locked-device span, despite the external lock command.
Profiler clock policy is therefore a first-class timing factor; mixing default
NCU runs with locked Driver/raw runs is invalid.

This does not prove simulator completion equivalence or eliminate all launch,
replay, and clock-domain tails. It closes the drift-free timing subcase for a
long, matched body under an explicit clock-control contract.

NVIDIA's current Nsight Compute guidance explicitly recommends locking clocks
externally with `nvidia-smi` and using `--clock-control none` when comparing
profiling results to externally timed runs; Nsight Compute 2026.1 also changed
the default clock-control policy to `boost`.
