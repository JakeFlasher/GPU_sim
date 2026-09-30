# Full RTX 3090 Ti campaigns with explicit NCU cache regimes

Both campaigns ran the 15-workload base/large set at graphics/SM 1350 MHz and
memory 10501 MHz on the local RTX 3090 Ti (SM86, driver 615.71.09). Each row
used five native repeats, three simulator replays, and seven NCU repeats. No
stage errors were recorded; these are performance observations, not tests or
verification workflows.

| campaign | NCU cache control | rows | elapsed | median native ms | median NCU ns | median NCU GPC cycles | median replay wall s |
|---|---|---:|---:|---:|---:|---:|---:|
| `full_3090ti_20260930_rm_cache_cold_live` | `all` | 15/15 | 487.072 s | 0.009448 | 10,080 | 13,572 | 0.629196 |
| `full_3090ti_20260930_rm_cache_warm_live` | `none` | 15/15 | 487.560 s | 0.009435 | 8,704 | 11,709 | 0.644798 |

Machine-readable outputs:

* `results/alpha/full_3090ti_20260930_rm_cache_cold_live/{summary.json,performance.csv,final_performance.md}`
* `results/alpha/full_3090ti_20260930_rm_cache_warm_live/{summary.json,performance.csv,final_performance.md}`

The per-workload warm/cold median ratio is approximately 0.826 for both NCU
duration and NCU GPC cycles (range 0.685--1.025), while native CUDA-event
medians and simulator frontiers remain near 1.0 (native median ratio 0.9995).
This is new evidence for endpoint/cache-domain separation, not a drift-free
timing conversion or a cache-ownership proof.

Although `nvidia-smi` remained at 1350/10501 MHz, the per-kernel NCU clock
fields varied: a representative cold row reports `gpc__cycles_elapsed` at
1.346052 GHz and `dram__cycles_elapsed` at 10.465409 GHz. Those fields are
useful clock-domain context, but they do not make NCU, CUDA events, CUPTI, and
simulator cycles one synchronized endpoint.
