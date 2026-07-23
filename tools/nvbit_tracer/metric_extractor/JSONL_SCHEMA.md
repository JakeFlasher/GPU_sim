# `metric_extractor` per-launch JSON-L schema

This document specifies the wire format of the per-launch records the
`metric_extractor` NVBit tool writes to its launches file (default
`./metric_extractor_launches.jsonl`; override via the
`METRIC_EXTRACTOR_LAUNCHES_PATH` env var). One JSON object per line,
followed by a single `\n`, no trailing whitespace.

The strict reference parser at
`tools/sol_baselines_harness/metric_extractor/launches_parser.py` is the
canonical consumer; any change to this schema MUST be accompanied by a
parser update + a `schema_version` bump.

## Versioning

- `schema_version` (integer) — bumped on ANY wire-incompatible change:
  field added with no default, field removed, field renamed, field
  retyped, or interpretation of an existing field changed. Adding an
  optional field with a documented default is NOT a wire-incompatible
  change. Current value: **2**. The Round-8 bump from 1 to 2 records
  the Round-7 addition of `filter_stats_snapshot.representative_skipped_paths`
  as a required field (no default) — Codex's Round-7 review caught
  that the Round-7 release left the slot at 1, violating the
  wire-discipline rule documented above. The parser is version-aware:
  schema-v2 records require the new field; schema-v1 historical
  records (pre-Round-7 preserved evidence bundles) default the field
  to `[]` so the round trip on captured artifacts still succeeds.
- `tool_version` (string) — semantic-version-like string identifying
  the `metric_extractor` build that produced the record. Current
  value: `"0.0.14-schema-v2"`. Bumped on any tool-side change
  that affects emitted values, even when the schema itself stays at the
  same `schema_version`. Recent bumps:
  - `0.0.9-global-ordinal`: `launch_index` now counts every observed
    kernel-launch CBID (including filter-skipped framework kernels);
    see the `launch_index` row below.
  - `0.0.10-cuda-stream-id`: `stream_id` / `cross_surface_join_key.stream_ordinal`
    now carry the raw CUDA stream ID from `cuStreamGetId()` — the
    same identity CUPTI Activity emits in its `streamId` field; see
    the `cross_surface_join_key` row below.
  - `0.0.11-saturating-counters`: three bounded-memory field-content
    transitions landed together. (a) Every per-launch counter
    `atomicAdd` (histogram buckets, `tma_count`, sector bytes, branch
    histogram, bank conflicts) now routes through a saturating helper
    that clamps at `UINT64_MAX`; `substatus.saturated_counters`
    carries the real per-launch overflow count instead of a literal
    `0`. (b) When the full record would exceed
    `METRIC_EXTRACTOR_PER_LAUNCH_ARTIFACT_CAP_BYTES` (10 MiB), the
    writer DISCARDS the oversize payload and emits a bounded
    replacement record with identity fields + null metrics +
    `substatus.truncated = true` +
    `waiver_reason = "per_launch_artifact_cap_exceeded"`. The
    pre-Round-5 behavior of flipping the flag on the oversize line is
    gone. (c) A new `launch_kind = "per_process_budget_exceeded"`
    is emitted when an accepted kernel lands after the slot pool is
    fully consumed; the record carries `substatus.skipped = true` +
    `waiver_reason = "per_process_budget_exceeded"` and the kernel
    runs uninstrumented. Pre-Round-5 the producer reused slot indices
    via modulo and silently shrank the instrumented set. See the
    `substatus` and `filter_stats_snapshot` rows below for the
    new sub-fields.
  - `0.0.12-calibrated-confidence`: three wire-content changes
    landed together. (a) The Round-5 bounded-replacement record
    emitted `bank_conflicts:null` and
    `branch_active_lanes_hist:[]` which the strict parser rejected;
    the writer now emits `bank_conflicts:0` (integer) and a
    33-element zero array so every truncation / budget-skip record
    round-trips cleanly through the surface parser. (b) The
    terminal `filter_stats.json` writer now merges via
    `flock(LOCK_EX)` + `max()` of existing-vs-in-process counters +
    union of `representative_skipped_paths` capped at 10, then
    atomically renames a per-PID temp file into place. A
    late-arriving zero-counter process can no longer overwrite a
    nonzero file. (c) The four Tier-1 confidence labels
    (`sector_efficiency_global_load`,
    `sector_efficiency_global_store`, `bank_conflicts`,
    `branch_active_lanes_hist`) are promoted from `"placeholder"`
    to `"calibrated"` after the T24 synthetic calibration corpus
    passed every AC-4 verdict end-to-end on sm_86. The
    `instruction_class_histogram` confidence stays `"high"` (direct
    measurement, no calibration step); `static_sidecar` stays
    `"placeholder"` (the harness-side sidecar join populates its
    sub-fields without a calibration corpus of its own).
  - `0.0.13-record-skipped-paths`: per-record
    `filter_stats_snapshot` now embeds
    `representative_skipped_paths` as a strings array (capped at 10,
    matching the in-process producer cap). The harness-side
    reconciler reads this list when `nvbit_at_term` misses the
    terminal `filter_stats.json` write — the previous shape gave
    the reconciler counters only, so the standalone audit artifact
    lost the AC-6 framework module attribution (`libtorch.so`,
    `libcublas.so`, etc.). Codex's Round-6 review caught this as a
    blocking AC-6 gap. The Round-7 release left `schema_version`
    at 1; Round 8's review caught that as a wire-discipline violation
    (a required field with no default IS wire-incompatible) and
    bumps the slot to 2.
  - `0.0.14-schema-v2`: three wire changes landed together.
    (a) `schema_version` bumps from 1 to 2 to correctly record that
    `filter_stats_snapshot.representative_skipped_paths` is a
    required field on every emitted record. The parser is
    version-aware: schema-v2 records require the field; schema-v1
    historical records default the field to `[]` so the pre-Round-7
    preserved evidence bundles still parse. (b) The terminal
    `filter_stats.json` merge writer now covers three counters that
    Round 7's `take_max()` merge omitted: `graph_launches_seen`,
    `graph_launches_not_observed`, and `unknown_opcode_count`.
    Codex's Round-7 review caught the partial coverage — those
    three counters could drift between the in-record state and the
    standalone audit artifact when `nvbit_at_term` missed. (c) The
    Python reconciler's contradiction-detector + rewrite loops
    cover the same three counters.
- Both fields are **mandatory** on every line. The parser rejects a
  line missing either.

## Truncation

- Each record is a single JSON object terminated by a literal `}`
  followed by `\n`. The parser rejects any line that does not end
  with `}\n` as mid-line truncation.
- Bundle-level integrity: when truncation is detected, the surface
  reports `status: failed` (per the per-surface manifest contract);
  the bundle is NEVER reported as `ok` with partial JSON-L.

## Top-level fields

| field | type | units | presence | notes |
|-------|------|-------|----------|-------|
| `schema_version` | integer | — | required | wire-version pin (see above). |
| `tool_version` | string | — | required | tool-build pin (see above). |
| `launch_kind` | string | — | required | one of `regular`, `cooperative`, `graph_node_traced`, `graph_launch_not_observed`, `per_process_budget_exceeded`. The `per_process_budget_exceeded` kind is emitted when an accepted kernel lands after the slot pool is fully consumed: the kernel runs uninstrumented and the record carries `substatus.skipped = true` + `waiver_reason = "per_process_budget_exceeded"` so the operator sees the over-budget cohort explicitly. |
| `kernel_name` | string | — | required for `regular` / `cooperative` / `graph_node_traced`; absent for `graph_launch_not_observed` | demangled C++ symbol from `nvbit_get_func_name(..., /*mangled=*/false)`. |
| `kernel_name_mangled` | string | — | same presence as `kernel_name` | mangled symbol from `nvbit_get_func_name(..., /*mangled=*/true)`. Used as the cross-surface CUPTI join fallback. |
| `function_uid` | integer | — | same presence as `kernel_name` | NVBit-process-local function uid; NOT cross-surface — see `kernel_ordinal_in_workload` for the CUPTI join. |
| `ctx_id` | integer | — | required | `(uintptr_t)CUcontext` cast to uint64; opaque process-local id. |
| `stream_id` | integer | — | required | raw CUDA stream ID from `cuStreamGetId(CUstream, &id)`. This is the same identity the matched `cupti_activity` Surface 2 run's `streamId` field carries, so both surfaces use a common namespace for the AC-8 cross-surface stream comparison. A negative value (`-1`) is the `kStreamIdUnresolved` sentinel returned when the driver call fails — the AC-8 joiner treats that as `unjoinable` because `stream_ordinal` is a required composite-key field. |
| `launch_index` | integer | — | required | global launch order within the matched run; monotonic, 0-based, incremented on **every observed kernel-launch CBID** (including launches the user-module filter skipped). This is the AC-8 `kernel_ordinal_in_workload` field — its value MUST align with what the matched `cupti_activity` Surface 2 run sees over the same launch sequence, including filtered framework (torch / CUB / cuBLAS / cuDNN) kernels CUPTI counts but NVBit drops. Pre-Round-2 the producer incremented only on accepted launches; the Round-2 → Round-3 `tool_version` bump records the semantic transition. |
| `grid_dim` | array[3] of integers | counts | required | `[gridDimX, gridDimY, gridDimZ]`. |
| `block_dim` | array[3] of integers | counts | required | `[blockDimX, blockDimY, blockDimZ]`. |
| `cluster_dim` | array[3] of integers | counts | required | `[clusterDimX, clusterDimY, clusterDimZ]` from `cuLaunchKernelEx` cluster attribute when present, else `[1, 1, 1]`. |
| `dynamic_smem` | integer | bytes | required | `sharedMemBytes` from the launch params. |
| `instruction_class_histogram` | object{string→integer} | counts | required | sparse map from instruction-class name to per-warp count. Class names: `ffma`, `mufu`, `ldg`, `stg`, `lds`, `sts`, `hmma`, `wgmma`, `tcgen05`, `tma`, `dsmem`, `bra`, `barrier`, `warpsync`, `unknown`. One increment per warp per instrumented SASS instruction observed (leader-lane atomicAdd; non-active warps and predicate-off instructions are not counted). Buckets with zero counts are omitted from the sparse map. `graph_launch_not_observed` records carry the empty object `{}`. |
| `tma_descriptor` | object\|null | mixed | required | first-sample decoded TMA descriptor for the launch, or `null` when no TMA instruction fired (the common case for non-TMA kernels). Populated for instructions where `Instr::isTMAMem()` returns true: the device callback `atomicCAS`-captures the first descriptor handle (bounded to 128 B), and on launch-exit the host decodes it via `nvbit_parse_tma_transfer_info(ctx, opcode, handle, size)` and emits the compact summary. Sub-fields when non-null: `sample_count` (integer; total TMA instructions observed this launch — the `tma_count` parallel counter that should equal the `tma` bucket in `instruction_class_histogram`), `sample_opcode` (string; SASS mnemonic of the sampled instruction, e.g. `UTMACCTL` / `UBLKCP`), `sample_handle_size` (integer bytes; size of the captured handle, clamped to 128), `is_tensor` / `is_bulk` / `is_prefetch` / `is_multicast` (booleans from `TMATransferInfo_t`), `src_memspace` / `dst_memspace` (string; one of `none`, `local`, `generic`, `global`, `shared`, `constant`, `global_to_shared`, `surface`, `texture`, `distributed_shared`, `unknown`), `transfer_count` / `transfer_size` / `byte_count` (integers; from `TMATransferInfo_t`), `multicast_cta_mask` (integer; CTA mask for multicast transfers). `graph_launch_not_observed` records carry `null`. |
| `sector_efficiency_global_load` | number\|null | ratio (0.0–1.0) | required | per-warp distinct-32B-sector efficiency for global LDG: ratio `useful_bytes / transferred_bytes` aggregated across every instrumented LDG warp-instance in the launch, where `useful_bytes = active_lanes * bytes_per_access` and `transferred_bytes = distinct_32B_sectors * 32`. Each instrumented LDG warp-instance contributes via leader-lane atomicAdd into the per-launch accumulator. `1.0` means perfectly coalesced (every loaded byte was useful), values below `1.0` mean wasted bandwidth from stride / divergent address shapes. `null` when no LDG was instrumented on the launch (true negative observation, distinct from `0.0` which would mean transferred bytes existed but useful did not — impossible by construction). DSMEM ops are excluded by the `getMemorySpace() == GLOBAL` filter in the instrument pass, so distributed-shared traffic does not contribute. **Confidence label**: `"calibrated"` under `tool_version >= 0.0.12-calibrated-confidence` (Round-6 T25 promoted it after the T24 synthetic-corpus tolerance gate landed `sector_efficiency_global_load = 1.0` for the coalesced fixture and `≈ 0.5` for the stride-2 fixture, both within tolerance). |
| `sector_efficiency_global_store` | number\|null | ratio (0.0–1.0) | required | parallel to `sector_efficiency_global_load` but for global STG. Same aggregation, same `null` semantics. **Confidence label**: `"calibrated"` under `tool_version >= 0.0.12-calibrated-confidence` (T25 promoted it alongside the load side). |
| `bank_conflicts` | integer | counts | required | aggregate per-launch sum of per-warp 32-bank shared-memory conflicts for instrumented plain shared LDS / STS instructions. Algorithm: for each predicate-true lane, compute `word_idx = addr >> 2` and `bank = word_idx & 31`; dedup `(bank, word_idx)` tuples (broadcast within a bank is not a conflict); per-bank conflicts = `max(0, distinct_words_in_bank - 1)`; warp total = sum across banks; launch total = sum across instrumented warp-instances. DSMEM (memory space `DISTRIBUTED_SHARED`) is EXCLUDED by the producer's `getMemorySpace() == SHARED` filter (AC-7 negative-test invariant: DSMEM has different banking semantics and must not be processed by the 32-bank model). The value `0` is a true observation meaning either no bank conflicts occurred or no plain-shared LDS / STS instructions fired — NOT a placeholder. The strict parser rejects any non-non-negative-integer value as a `ParseError`. **Confidence label**: `"calibrated"` under `tool_version >= 0.0.12-calibrated-confidence` (T25 promoted it after the T24 32-way-conflict fixture verdicted `bank_conflicts = 31` exactly and the broadcast fixture verdicted `0`). |
| `branch_active_lanes_hist` | array[33] of integers | counts | required | per-launch histogram of branch-warp-instance divergence. For every instrumented instruction the classifier routes to `INSTR_CLASS_BRA` (BRA / JMP / EXIT / BRX / CALL / RET, etc. — the same bucket set the instruction-class histogram's `bra` field counts), the injected `count_branch_active_mask_cb` device callback computes the popcount of the warp-collective predicate mask (`__ballot_sync(__activemask(), predicate)`) and atomicAdds 1 into the bucket at that index. The array length is exactly 33 because a 32-lane warp's predicate-true count is bounded to `[0, 32]` inclusive. Bucket 0 counts warp-instances where every active lane evaluated the predicate to false (no lane took the branch this warp-instance); bucket 32 counts fully-active warp-instances (unconditional / always-true branches); bucket 16 is the AC-4 canonical "predicated branch with 16 active lanes" landing zone. A bucket value of `0` is a TRUE observation meaning "no branch warp-instances landed in that bucket this launch" — NOT a placeholder. **Confidence label**: `"calibrated"` under `tool_version >= 0.0.12-calibrated-confidence` (T25 promoted it after the T24 predicated-branch fixture verdicted `branch_active_lanes_hist[16] >= 1`). |
| `confidence` | object{string→string} | — | required | per-field confidence label; values: `"placeholder"` (default while metric counters are absent OR while the synthetic-calibration tolerance gate hasn't run yet), `"low"`, `"calibrated"`, `"high"`, `"static_only"`, plus `"absent"` for fields whose absence is itself a true negative observation rather than an unwired pathway. Keys mirror the metric-value field names. **Current per-field labels under `tool_version >= 0.0.12-calibrated-confidence`**: `instruction_class_histogram` → `"high"` (instrumentation directly counts buckets; no calibration needed); `tma_descriptor` → `"high"` when the launch sampled at least one TMA instruction, `"absent"` otherwise; `sector_efficiency_global_load` / `sector_efficiency_global_store` → `"calibrated"` (T24 corpus passed every AC-4 verdict end-to-end on sm_86); `bank_conflicts` → `"calibrated"` (T24 32-way-conflict + broadcast fixtures both passed); `branch_active_lanes_hist` → `"calibrated"` (T24 16-active predicated-branch fixture passed); `static_sidecar` → `"placeholder"` (the harness-side static-sidecar join populates its sub-fields without a calibration corpus of its own). |
| `static_sidecar` | object\|null | — | required | placeholder shape with sub-fields `registers_per_thread` (integer\|null), `dynamic_smem_per_block` (integer\|null), `theoretical_occupancy_warps` (integer\|null), `local_memory_spill_bytes` (integer\|null). **Placeholder**: `null` (sidecar join lands with the harness surface plugin). |
| `cross_surface_join_key` | object | — | required | composite key for post-hoc CUPTI Activity join. Sub-fields: `workload_index` (integer\|null), `evaluator_config_hash` (string\|null), `demangled_kernel_name` (string\|null), `kernel_ordinal_in_workload` (integer; equals top-level `launch_index`. Post-Round-2 the value counts every observed kernel-launch CBID — including filter-skipped framework kernels — so the matched `cupti_activity` Surface 2 run's view aligns by construction), `grid_xyz` ([3] of integers), `block_xyz` ([3] of integers), `dynamic_smem` (integer), `stream_ordinal` (integer; raw CUDA stream ID from `cuStreamGetId()` — same value the top-level `stream_id` carries. Post-Round-4 both NVBit and CUPTI Activity use this identity, so the AC-8 stream-slot comparison is direct. A negative value (`-1`) is the `kStreamIdUnresolved` sentinel and lands the record in `unjoinable`). |
| `substatus` | object | — | required | run-time status flags. Sub-fields: `truncated` (bool; `true` only on a record where the writer emitted a bounded replacement because the full record would have exceeded `METRIC_EXTRACTOR_PER_LAUNCH_ARTIFACT_CAP_BYTES` — when `true`, the metric values are null/empty and `waiver_reason = "per_launch_artifact_cap_exceeded"`), `skipped` (bool; `true` on per-process-budget skip records — when `true`, the kernel ran uninstrumented because the slot pool was fully consumed, the metric values are null/empty, and `waiver_reason = "per_process_budget_exceeded"`), `saturated_counters` (integer; count of per-launch counter overflows clamped at `UINT64_MAX` — incremented by the device-side saturating helper exactly once per saturation event), `is_graph_node` (bool), `is_cooperative` (bool), `waiver_reason` (string\|null; non-null when status is degraded — values include `"per_launch_artifact_cap_exceeded"`, `"per_process_budget_exceeded"`, `"graph launches not enumerated by tool"`). |
| `filter_stats_snapshot` | object | — | required | the running `filter_stats.json` counter snapshot at the moment the record was emitted. Sub-fields: `accepted_launches`, `skipped_launches`, `module_origin_miss_count`, `cooperative_launches`, `graph_node_launches`, `graph_node_accepted`, `graph_node_skipped`, `graph_launches_seen`, `graph_launches_not_observed`, `per_process_budget_skipped` (count of accepted kernels that landed AFTER the slot pool was fully consumed; non-zero means at least one `launch_kind = "per_process_budget_exceeded"` record exists in the JSON-L), `unknown_opcode_count`, `representative_skipped_paths` (array of strings, capped at 10; the framework module paths the user-module filter rejected — torch / CUB / cuBLAS / cuDNN / etc. — captured here per-record so the harness-side reconciler can recover the AC-6 module attribution when `nvbit_at_term` misses the terminal `filter_stats.json` write). The snapshot is included so a downstream consumer can reconstruct the running state from any single record without separately ingesting `filter_stats.json`. |
| `host` | object | — | required | host snapshot at first emit. Sub-fields: `compute_capability` (string; e.g. `"8.6"`, `"10.0"`, `"12.0"`), `cuda_runtime_version` (integer), `nvbit_tool_version` (matches top-level `tool_version`), `arch_label` (string; `"local_sm_<CC>"` or `"b200_sm_100a"` per CLAUDE.md evidence labels). |

### `launch_kind = graph_launch_not_observed` shape

When a `cuGraphLaunch` completes without any `nvbit_at_graph_node_launch`
hook firing during it, the tool emits a single record with:

- `launch_kind = "graph_launch_not_observed"`.
- `kernel_name`, `kernel_name_mangled`, `function_uid` absent (the
  individual nodes are exactly what the tool failed to observe).
- `grid_dim` / `block_dim` / `cluster_dim` / `dynamic_smem`: `[0,0,0]` /
  `[0,0,0]` / `[0,0,0]` / `0` (sentinel; per-node values are not
  reconstructible from the graph-launch boundary alone).
- All metric fields at placeholder values.
- `substatus.is_graph_node = false`, `substatus.is_cooperative = false`.
- `substatus.waiver_reason = "graph launches not enumerated by tool"`
  (the canonical waiver string the harness manifest expects).
- `filter_stats_snapshot` populated as for any other record.

This shape lets downstream consumers count and waive missed graph
launches without confusing them with regular kernel-launch records.

## Run-end summary

The launches JSON-L file does **not** carry a synthetic run-end
record. The tool's `nvbit_at_term()` flushes the running summary into
`filter_stats.json` (path resolved from
`METRIC_EXTRACTOR_FILTER_STATS_PATH`, default
`./metric_extractor_filter_stats.json`); per-record consumers can also
read the running snapshot from any record's `filter_stats_snapshot`
field. Keeping per-launch records and the run-end rollup in separate
artifacts means the launches file is exactly "one record per accepted
launch + one `graph_launch_not_observed` record per missed graph
boundary" — consumers do not have to special-case a non-launch line.

## What is NOT emitted

- **No `duration_us`, `wall_clock_*`, `start_ns`, `end_ns` field** —
  the on-device atomic aggregation perturbs instrumented-run timing
  by construction, so any wall-clock the tool could report would be
  misleading. Wall-clock attribution is owned by the
  `cupti_activity` Surface 2 via post-hoc join on the
  `cross_surface_join_key` on a separate uninstrumented matched run.
- **No raw per-instruction address records.** Address samples for
  Tier-2 calibration (when that lands) are emitted in a separate
  opt-in debug-mode binary file, never in the per-launch JSON-L.
- **No per-thread or per-warp lane addresses.** The aggregated
  histograms above are the only metric surface.

## Cache-identity inputs (for the harness surface plugin to wire)

Per `[[BL-20260515-cache-key-causal-coverage]]`, every operator-tunable
input that affects produced behavior must enter the harness cache-key
allowlist (`NVBIT_METRICS_USER_ENV_VARS`) when the surface plugin
lands (T17-T22). Currently:

- `METRIC_EXTRACTOR_USER_MODULE_REGEX` (default `.*sol_execbench.*`)
  — module-origin regex applied to the resolved `/proc/self/maps`
  entry of each accepted launch.
- `METRIC_EXTRACTOR_KERNEL_NAME_REGEX` (default unset — admit all)
  — per-launch kernel-name regex applied to the demangled kernel
  name BEFORE slot allocation. When set, non-matching launches are
  rejected without consuming a Tier-2 buffer slot, preserving the
  per-process slot budget for the target kernel (use case: pinning
  the harvest to one custom-`__global__` kernel while a PyTorch
  curand-init kernel runs first and would otherwise grab the only
  slot under the default 512 MiB per-process budget cap).
- `METRIC_EXTRACTOR_FILTER_STATS_PATH` (default
  `./metric_extractor_filter_stats.json`).
- `METRIC_EXTRACTOR_MAX_LAUNCH_SLOTS` (default `4096`).
- `METRIC_EXTRACTOR_LAUNCHES_PATH` (default
  `./metric_extractor_launches.jsonl`).
