# Blackwell/SM120 reverse-engineering evidence

This is the 2026-10-19 Gemini research record for the accuracy goal. It is
evidence and model-boundary documentation, not a new execution plan. The
source checkout remains read-only; no new microbenchmark was run.

## Sources and provenance

### Local evidence

- SM120 NCU attributes and replay policy are in the permitted capture tree:
  `/home/jakeshea/5090_branch/sm120_rtx_5090_capture/profiles/*/raw.csv` and
  `replay_cache_policy.txt` (`--replay-mode kernel --cache-control all`).
- SM86 attributes and policy are in
  `/home/jakeshea/5090_branch/sm86_rtx_3090_capture/`.
- Blackwell manuals and metric definitions are retained under
  `references/NVIDIA_Manuals/blackwell_tuning/`,
  `references/NVIDIA_Manuals/cuda_binary_utility/`, and
  `references/NVIDIA_Manuals/ncu_guide/`.
- The captured TMA sidecar is `extra_info/tma_trace.jsonl`, schema version 2,
  capability `sm120_tma_sidecar_v1`. A representative record includes
  `opcode`, `transfer_count`, `transfer_size`, `byte_count`, runtime `coords`,
  `tensor_map` (rank, data type, dimensions, strides, box dimensions, element
  strides, swizzle/interleave), `raw_handle_hex`, and expanded `src_addresses`
  / `dst_addresses`. The reader is
  `sim/trace/src/async_sidecars.rs`.
- Existing probes and crosswalks are
  `sm86_ltc128b_ptx_probe_slice_20261016.md`,
  `sm86_metric_unit_crosswalk_slice_20261016.md`, and
  `sm86_l1_setway_slice_20261015.md` in the permitted read-only source tree.

### External/public evidence

- [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA Driver TMA API / `CUtensorMap`](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [NVIDIA CUTLASS/CuTe](https://github.com/NVIDIA/cutlass)
- [Colfax TMA tutorial](https://research.colfax-intl.com/tutorial-hopper-tma/)
- [Blackwell microbenchmark study, arXiv:2512.02189](https://arxiv.org/abs/2512.02189)
- [Blackwell microbenchmark study, arXiv:2507.10789](https://arxiv.org/abs/2507.10789)
- NVIDIA patents summarized locally in
  `references/NVIDIA_patents/NVIDIA_patentd.md` (sectored promotion,
  `LDGSTS`, and TMA).

## High-confidence facts

The captured RTX 5090 attributes report 170 SMs, 85 TPCs, 8 FBPs, 96 L2
slices, a 96 MiB L2, a 512-bit memory bus, 102,400 B maximum shared memory per
SM, and 1,024 B reserved shared-memory header per block. The captured RTX 3090
attributes report the corresponding GA102 values. Both lanes use 128-byte
lines and 32-byte sectors in the simulator contract.

The TMA sidecar already carries substantially more than opcode counts: runtime
coordinates, decoded descriptor fields, transfer sizes, barrier identity,
instrumentation order, and expanded valid addresses. The simulator reduces
these to address-derived 32-byte source/destination sectors and keeps a
diagnostic L2 shadow. This is sufficient for provenance and shadow accounting;
it is not proof of hardware arrival order, MSHR coalescing, eviction, or DRAM
acknowledgement.

The NCU families have distinct scopes. `l1tex__t_sectors_*_lookup_*` are
L1TEX tag-stage lookup endpoints; `lts__t_sectors_*_lookup_*` are client- and
operation-qualified L2 lookup endpoints; `dram__sectors_*` are profiler DRAM
sector endpoints. Physical cache response counters, lookup observers, and
payload-volume terms must not be conflated. The active event path deliberately
keeps these distinctions in `sim/mem/src/l2.rs` and
`sim/sm/src/system.rs`.

## Architecture/model boundaries

The physical 96-slice attribute is real, but the current SM120 effective cache
is an explicit conditional scenario. With `SliceHash::LineMod`, 1 slice with
49,152 sets and 96 slices with 512 sets produce the same line equivalence
relation (`line mod 49,152`); changing only `num_slices` is a mathematical
no-op for hits/evictions and can overflow the current per-slice diagnostic
array. A proprietary slice hash, replacement policy, fill-buffer horizon, and
cross-SM chronology remain unknown.

Shared-memory admission is modeled, but a trace does not prove the physical
SM120 L1/SRAM carveout floor, set/way hash, or per-launch hardware capacity.
Changing the default effective L1 geometry would therefore be an uncalibrated
timing/cache experiment, not a recovered fact.

TMA descriptors are available, but sidecar ordering is not engine chronology.
The address-derived TMA L2 shadow omits concurrent ordinary traffic and
cross-CTA fill/eviction state. Replacing raw TMA payload demand with shadow L2
misses was tested on the full SM120 corpus and worsened DRAM APE (39.56% to
67.62% on the gated rows; 49.28% to 49.37% overall), so raw payload demand is
retained for the scored DRAM endpoint.

## Adjudicated candidate changes

The following proposed changes were independently reviewed against code and
the complete replay evidence:

1. **TMA payload → shadow misses:** rejected by the full-corpus experiment and
   by the missing chronology/fill-buffer semantics.
2. **Unconditional L2 write-allocate lookup hits:** rejected. In the active
   SM120 event path, `counted_hit` (physical write/tag response) is distinct
   from `lookup_hit = sector_present`; the validated resident store-packet plus
   cold TMA-store observer reduced generic L2 hit APE from about 57% to 24.3%.
3. **Active LTC128B sibling promotion or SM86 H65 transfer:** rejected. PTX
   describes a promotion/eviction hint, not an unconditional SM120 four-sector
   demand rule. The full LTC shadow audit contains about 300 zero-target rows
   and two false positives; the resident-plus-older-pending L1 candidate is
   shadow-only because sector-unit conservation is not proven.
4. **Default `num_slices=96` or dynamic SM120 carveout:** no immediate active
   change. The former is a `LineMod` no-op; the latter lacks physical SRAM
   floor/way evidence and may be retained only as an explicitly opt-in,
   rollback-gated sensitivity experiment.
5. **Post-hoc IPC scaling, fixed coalescing windows, cache-hint subtraction,
   or constant-pipeline re-scoping:** rejected as workload-shape fitting or
   unobserved scheduler/cache behavior.

## Current disposition

Gemini's latest counter-identity audit identified one safe reporting fix: the
SM86 native unified-L1 percentage now uses the already validated
form-qualified global lookup endpoint. It passed full replay status gates and
reduced joined SM86 unified-L1 mean APE from 29.04% to 12.15%; SM120's alpha
path is unchanged. No other active projection was promoted. The full replay
artifacts remain:

- SM120 DRAM-enabled: `full_replay_runs/l2_scope3_sm120_enabled/`
- SM86: `full_replay_runs/l2_scope3_sm86/`
- comparison: `FULL_NUMERIC_COMPARISON_L2_SHADOW3.json`

The residuals are not hidden or fit away. Resuming active accuracy work needs
external metadata such as TMA descriptor/runtime persistence and launch-cache
state, disaggregated tag/MSHR/fill/stall PMU counters, or existing isolated
LTC128B/TMA hardware evidence that is independently cross-walked to these
endpoints. No fast replay, new SM120-disabled replay, NCU fitting, count
balancing, or new microbenchmark was used here.

## Accepted SM86 private-observer correction

The later SM86 accuracy review found a structural topology bug in the
report-only write-tag observer: one table had been shared by all SM actors even
though L1TEX tags are private per SM. The observer is now allocated as a fixed
vector aligned with the sorted traced-SM actor roster, one 256-set/2-slot table
per SM, and is recreated for each launch. Native projection records whether the
observer was actually enabled and falls back to physical line-tag store hits
when it was not. No SM120 path is instantiated.

Full SM86 replay with twelve workers preserved 286 `ok`, 62 `control_error`,
8 `replay_error`, 12,392 launches, and bit-identical IPC/L2/DRAM/timing. Mean
unified-L1 APE improved from 43.71% to 24.66% over all 286 successful rows,
12.15% to 9.87% on exact joins, and 88.42% to 45.61% on ordered fallback
joins. A repeat full replay was identical apart from wall clocks. The required
SM120 DRAM-enabled lane was rerun at 339/339 `ok` and 11,558 launches and was
unchanged. This is accepted as an SM86 report-only topology correction, not as
closure of the remaining IPC, DRAM, fallback-L1, or SM120 accuracy gaps.

## Follow-up correction

The novel report also proposed reconciling SM120 native-L2 metrics with cache
lookup predictions. Code review found that `metrics::project` returns from the
SM120 alpha branch before the later SM86-only L2 native block; current SM120
rows do not contain those native L2 keys. There is therefore no SM120 rank-0
native-L2 override to repair. Directly adding TMA-shadow hits to retained
SM120 L2 estimates worsened mean APE, so this candidate was not promoted.
