# SM86 common globaltimer crosswalk (2026-09-30)

## Question

Can the same instruction body provide a drift-free endpoint across an
independent raw Ampere QMD V03 launch and the CUDA Driver launch path, without
using NCU/CUPTI labels to calibrate the comparison?

## Method

A fresh CUDA `sm_86` cubin (`results/alpha/sm86_common_timer_crosswalk_20260930/cuda_driver/common_timer_raw_sm86.cubin`) contains one active lane that:

1. reads `%globaltimer`;
2. spins until 135,000 `%clock64` ticks have elapsed;
3. reads `%globaltimer` again;
4. stores the two 64-bit values with `STG.E.64.STRONG.SYS`.

The timer operation sequence and launch shape were matched at the semantic
body level:

- CUDA Driver API loads the cubin and launches one `32x1x1` block.
- An independently built external `libmc` GA10x path embeds the extracted
  SM86 operation body in an Ampere QMD V03 and launches the same `32x1x1`
  workload through raw RM/QMD.

The retained raw-QMD header is **not byte-identical** to the Driver cubin:
comparison shows control-word differences at PCs `0x100` and `0x110` in the
unreachable/store-tail region. The crosswalk is therefore semantic/body-level,
not an exact SASS-byte identity claim.

The host remained on the locked RTX 3090 Ti regime (SM 1350 MHz, memory
10501 MHz, driver 615.71.09). No NCU/CUPTI value was used as an input or
conversion factor.

## Result

| Endpoint | launches | globaltimer span (ns) | median |
|---|---:|---:|---:|
| raw QMD, post-setup | 32 | 99,328--100,352 | 100,352 |
| CUDA Driver | 16 | 99,328--100,352 | 100,352 |

The quantized span sets are identical and the median difference is zero. The
raw records are in `results/alpha/sm86_common_timer_crosswalk_20260930/raw_qmd/raw_qmd_output.txt`; the CUDA records are in `.../cuda_driver/driver_output.txt`; the machine-readable summary is `.../summary.json`.

## Interpretation

This closes a common **device `%globaltimer` endpoint for a matched semantic
body, block size, and clock regime**, while retaining the control-word caveat
above. It removes the earlier
1.62x raw-QMD/NCU discrepancy as a generic timer conversion issue for this
controlled body; that discrepancy remains an unmatched launch-envelope/NCU
comparison.

It does not establish that NCU `gpu__time_duration`, CUPTI sample timestamps,
CUDA-event timestamps, host monotonic time, or simulator cycles share this
endpoint. It also does not expose cache-line owner/read-from/dirty-sector or
DRAM acknowledgement. The PTX `discard.global.L2` probes retained beside the
artifact are deliberately not correctness or ownership oracles: NVIDIA
specifies discarded data as unstable/indeterminate ([PTX ISA 9.4](https://docs.nvidia.com/cuda/parallel-thread-execution/)).

## Source anchors

- NVIDIA PTX `%globaltimer` and memory instruction semantics: [PTX ISA 9.4](https://docs.nvidia.com/cuda/parallel-thread-execution/).
- NVIDIA Ampere SASS/cache-control instruction inventory: [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/).
- Raw QMD/QMD-V03 implementation and launch artifacts: `results/alpha/sm86_nak_direct_qmd_goal_20260930/` and `results/alpha/sm86_common_timer_crosswalk_20260930/raw_qmd/`.

## CUDA-event and CUPTI extension

The same CUDA Driver process then recorded eight launches with CUDA events and
CUPTI concurrent-kernel activity (no NCU). In-body `%globaltimer` spans were
100,032 ns for all eight launches. CUDA-event elapsed durations were
104,320--112,224 ns (median 104,592 ns, ratio 1.0456), including a larger
stream/event envelope. CUPTI kernel activity spans were 101,218--101,634 ns
(median 101,234 ns, ratio 1.0120), close but measurably above the in-body timer.
CUPTI activity timestamps were about one second later than the in-body timer
because the record buffer was flushed after the launches; the duration, not
absolute epoch, is the comparable quantity.

This extension resolves an important ambiguity: the matched `%globaltimer`
endpoint is stable across raw QMD and CUDA Driver for the body, but CUPTI and
CUDA events are **not identical endpoints** even on the same launch path.
They must remain separate fields with an explicit transform/uncertainty rather
than being silently collapsed into one timer.


## Same-body endpoint extension

The same common_timer_raw cubin was then launched through a CUDA Driver
launcher that recorded CUDA events and CUPTI concurrent-kernel activity around
the same body. The in-body globaltimer spans were 100,032--100,064 ns
(median 100,032 ns). After the first setup launch, CUDA-event spans were
104,224--112,320 ns (median 104,448 ns); CUPTI activity spans were
101,215--101,791 ns (median 101,231.5 ns). The first CUDA-event span was a
launch/setup outlier.

A separate Nsight Compute profile of the same 16-launch driver process
reported 15 profiled rows with gpu__time_duration 101,600--101,632 ns
(median 101,600 ns), alongside gpu__time_start/gpu__time_end frame-relative
values and GPC-cycle counters. This is the first same-body crosswalk that
includes NCU duration, CUPTI activity, CUDA events, and the in-body globaltimer
without fitting any endpoint to a campaign label.

The relation is informative but not equality:

* globaltimer is the body interval;
* CUPTI is about 1.2 us wider;
* NCU is about 1.6 us wider;
* CUDA events are about 4.4 us wider after setup.

These fixed envelopes are now measurable for this body and clock regime, but
they remain path-specific offsets/tails. They do not establish simulator
completion equivalence or a universal transform for arbitrary kernels.
Artifacts are under cuda_driver_event_cupti/ and ncu/; summary:
summary_v2.json.

## Long-body clock-lock extension

A 1,350,000-clock-tick semantic-equivalent version was run across raw QMD,
Driver, CUPTI, CUDA events, and NCU. With external clocks locked and explicit
`ncu --clock-control none`, medians were 1,000,448 ns (raw/Driver), 1,000,032
ns (CUPTI), 1,004,528 ns (event), and 1,001,600 ns (NCU). Default NCU boost
control instead produced 715,824 ns, demonstrating a profiler clock-policy
confound. Full artifact/report:
`sm86_long_timer_clock_lock_20260930.md`.
