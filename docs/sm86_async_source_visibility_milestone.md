# SM86 async-source destination-visibility discriminator

**Status:** diagnostic complete 2026-08-30; bypass effect identified, no timing promotion

**Architecture:** `sm_86` / NVIDIA GeForce RTX 3090 Ti (device 0)

**Evidence root:** `results/captures/sm86_async_source_visibility_20260830_v2/`

**Machine report:** `analysis/report.json`

## Question and construction

The predecessor's store-free `.cg` traces retained only the simulator's
`async_copy_shared_visibility` tail, but changing depth alone did not identify
the missing destination mechanism. This slice asks whether shared-destination
topology changes live execution while source addresses and source traffic stay
fixed, and whether the current cache-all destination model transfers to that
effect.

The target is one 32-thread warp and 64 serial copy groups. Source lanes are
16-byte copies spaced 32 bytes apart; positions are 1 KiB apart. Both layouts
reserve 32 KiB of shared memory and use 4 KiB position spacing:

- `compact`: 32-byte destination lane stride;
- `conflict`: 64-byte destination lane stride.

The same warp-uniform shared broadcast follows `wait_group 0` in both kernels,
so the consumer load adds no layout-dependent bank conflict. Depth 4 is the
primary slice and depth 8 is the held-out transfer. `.ca` supplies the
cache-all control; `.cg` supplies the bypass/DRAM route. The target contains no
global store, atomic, reduction, or CTA barrier.

The static sidecar contains 20 functions, 912 decoded instructions, and zero
unknown encodings. Within every route/depth pair, compact and conflict have the
same instruction count, static stall sum, registers, 32 KiB shared allocation,
zero stack/local allocation, and no forbidden target opcode. Depth-4 uses 13
registers and depth-8 uses 15; the retained build log reports no spills.

## Live result

Five NCU application-replay profiles per cell were interleaved by layout. NCU
used `--clock-control none --cache-control none`; the host clock sample was
1350 MHz SM / 10501 MHz memory in P0. These observations are comparison data,
never simulator inputs.

| route | depth | compact SM active | conflict SM active | paired median delta | compact→conflict destination wavefronts | simulator terminal compact→conflict |
|---|---:|---:|---:|---:|---:|---:|
| L1 `.ca` | 4 | 10,559 | 14,608 | +4,055 (+38.44%) | 2,048→4,096 | 9,500→13,596 |
| L1 `.ca` | 8 | 14,730 | 22,856 | +8,114 (+55.07%) | 4,096→8,192 | 14,623→22,815 |
| bypass `.cg` | 4 | 37,915 | 39,724 | +1,737 (+4.58%) | 2,048→8,192 | 34,257→34,257 |
| bypass `.cg` | 8 | 45,657 | 48,574 | +2,917 (+6.39%) | 4,096→16,384 | 39,635→39,635 |

Every paired SM-active and GPC-elapsed delta is positive across all five
repeats. The maximum within-cell active-cycle relative range is 4.42%, in the
unlocked DRAM depth-8 compact cell.

The source invariants are exact in every repeat and across each layout pair:

- depth 4: 256 LDGSTS warp instructions and 8,192 source sectors; L1 has
  8,064 hits/128 misses, bypass has 0 hits/8,192 misses;
- depth 8: 512 instructions and 16,384 source sectors; L1 has 16,128 hits/256
  misses, bypass has 0 hits/16,384 misses.

The current cache-all model transfers independently: its terminal layout
delta is 4,096 cycles at depth 4 and 8,192 at depth 8, only 1.01% and 0.96%
from the corresponding live paired medians. Replay twins are byte-identical.
Bypass replays instead remain layout-identical and correctly expose the typed
`async_copy_shared_visibility` incomplete frontier.

## Decision and causal limit

This is positive evidence for a bypass-specific source/destination interaction,
not evidence for importing the cache-all rule. The trace-derived cache-all
bank projection undercounts the observed `.cg` conflict wavefront multiplier,
and whole-kernel active time cannot determine whether the remaining bypass
delta is caused by LDGSTS issue replay, coupled source/destination transfer, or
post-source shared visibility. Those mechanisms have different event ordering;
fitting any one of them to the two observed deltas would overstate what the run
identifies.

Keep the current bypass lower bound and typed tail. The next discriminator is
a three-frontier `.cg` layout assay that observes pre-issue, post-issue, and
post-wait clocks while placing any result store after all measured intervals.
That split can decide whether to change issue service, completion work, or only
the tail. JADE/HETR, CPU-GPU co-simulation, `sm_120`, and scored workload
defaults remain untouched.

## Reproduction

```bash
make -C workloads/research_probes/async_source_only clean all

CUDA_VISIBLE_DEVICES=0 ncu --force-overwrite --export /tmp/visibility \
  --profile-from-start off --launch-count 1 --replay-mode application \
  --clock-control none --cache-control none --metrics \
gpc__cycles_elapsed.max,sm__cycles_active.max,smsp__inst_executed_op_ldgsts.sum,sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts.sum,sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum,lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum,dram__sectors_read.sum \
  workloads/research_probes/async_source_only/async_source_only \
  --groups 64 --depth 4 --route dram --layout compact --markers on

PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/ldgsts_burst_release/capture_nvbit.py \
  --output /tmp/visibility-trace --cell visibility_dram_compact_d4 \
  --binary workloads/research_probes/async_source_only/async_source_only \
  --mode ncu --gpu 0 --timeout 300 --target-args \
  --groups 64 --depth 4 --route dram --layout compact --markers on

sim/target/release/sim-run --trace /tmp/visibility-trace \
  --control results/captures/sm86_async_source_visibility_20260830_v2/static/control_bits.json \
  --arch-card arch_profiles/cards/sm_86.json \
  --pchase-card data/pchase/sm_86_static_card.json --mode l1-l2

PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/analyze_visibility.py \
  --root results/captures/sm86_async_source_visibility_20260830_v2 \
  --output results/captures/sm86_async_source_visibility_20260830_v2/analysis/report.json
```

## Content identities

- Probe binary: `12f79df2ed22becc493fca1590f1d173d84bf1e1e4ae436f5fc6d8070d609169`
- Release `sim-run`: `6cb0d05f0d447c0cd6ee2e30353c0f1e89d464462458912478c826860f15f9a4`
- Static control sidecar: `d9e335fde7d7dbe19f4002f1db1fe0f9731df97ca50ca414b4246b5b46a3d30b`
- Machine report: `8b49b821d5d313f4225fc70bd78c8638e6aa45f56614c5a8883ee63041216e63`
- Threadblock payloads:
  - DRAM compact d4: `b7af565eb2b2ad3177f311efc56f941b1bdb8e3c17c5516daf9dfb439703f26d`
  - DRAM compact d8: `255eec23aaba6840d597e7bc3513e5ba2813ed1b6e2bda37fe3dfdd5db0b1add`
  - DRAM conflict d4: `5e92f89740fcf75a45c1f5dacf78e0bfea23d91d46e7d2faa59ee8f5e062dcd0`
  - DRAM conflict d8: `a6339aa511c1a41143133e0789ede2badc9fe17ef1a80a4875b14f9f8e162913`
  - L1 compact d4: `b8bee8d56a052249e277e2a6b59c246b0fa53ac3bd47abc02de2faf5e35b9dfb`
  - L1 compact d8: `43b92bf65d7e4c69d09ac6dbec87179d314f0fc4a06fcd32e82ba3cc212c24fd`
  - L1 conflict d4: `e49bfd207c4ee5c428a977695541665bb0aac59e578cb3de625f67cc20727231`
  - L1 conflict d8: `b24ec03fa0a84dae7368440ea4b0df1e4f0057a348fe96e6ba42c5ae0653455e`
