# SM86 final locked-campaign performance handoff

**Hardware:** NVIDIA GeForce RTX 3090 Ti, UUID
`GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, SM86, driver 615.71.09.  
**Clocks:** 1350 MHz graphics/SM, 10501 MHz memory, P0.  
**NCU policy:** cold `--clock-control none --cache-control all`; warm
`--clock-control none --cache-control none`.

## Full campaigns

| campaign | completed | NCU median duration | NCU median GPC cycles | native median | replay median |
|---|---:|---:|---:|---:|---:|
| cold/cache-all | 15/15 | 10,080 ns | 13,572 | 0.009448 ms | 0.629196 s |
| warm/cache-none | 15/15 | 8,704 ns | 11,709 | 0.009435 ms | 0.644798 s |

Elapsed wall time was 487.072 s cold and 487.560 s warm. These are final
performance observations for the declared campaign contract, not universal
accuracy claims and not a cache-owner proof.

## Timing contract

The long semantic timer body crosswalk under the same external locks measured:

* raw QMD/Driver in-body globaltimer: 1,000,448 ns median;
* CUPTI: 1,000,032 ns;
* CUDA event: 1,004,528 ns;
* NCU with `--clock-control none`: 1,001,600 ns.

Default NCU boost control measured 715,824 ns for the same long body, so profiler
clock policy must remain explicit. Raw-QMD and Driver text also differ in tail
control words; the crosswalk is semantic-body-level, not byte-identical SASS.

## Remaining physical boundaries

1. No exact dynamic physical pipe/subunit/issue-slot witness.
2. Independent generation remains finite operation-family/ABI/control coverage.
3. Cache owner/read-from/dirty-sector/complete DRAM semantics remain hidden.
4. Simulator completion remains distinct from NCU/CUPTI/event/device-timer
   endpoints.

The full results are retained under:

* `results/alpha/full_3090ti_20260930_rm_cache_cold_live/`
* `results/alpha/full_3090ti_20260930_rm_cache_warm_live/`
* `results/alpha/sm86_common_timer_crosswalk_20260930/long_clock_locked_20260930/`
