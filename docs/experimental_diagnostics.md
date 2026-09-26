# Experimental diagnostics and ITS trace contract

Updated 2026-09-26. Consolidates the separate writeback, coherence and pipe
notes. These paths execute, but **do not close the native timing mechanisms**.
Source: `sim/direct_run/src/metrics.rs`, `sim/sm/src/metrics.rs`,
`sim/arch/src/its.rs`.

## Native diagnostic surface

Outer `schema_version=1`, additive `metric_surface_version=2`. Consumers should
select known keys and inspect `status`, units and semantics, not assume a fixed
metric list. There is no complete native-output JSON schema yet. No diagnostic
below has an NCU counter identity or alters native scheduling/cache state.

| Prefix / policy | Formula or algorithm | Limits |
| --- | --- | --- |
| `dirty_writeback_completion_*` | Frontier `F`; width `ceil(32 * (dirty_evicted + terminal_dirty) * card_SM_Hz / card_peak_Bps)`; scenario end `F + width` | Optimistic service arithmetic, no scheduled write queue, interference, ordering or flush visibility; not an upper bound |
| `coherence_completion_*` | Frontier `F`; width = distinct decoded cross-SM overlapping sectors; scenario end `F + width` | Arbitrary one-cycle coefficient; includes read/read sharing and ignores repeated ownership transitions; not coherence latency or a bound |
| `min_selected_then_lexical_v1` | For each extraction record choose the candidate with fewest previous selections, lexical tie break; retain unknowns | Cumulative counts, not SMSP/issue-time availability; no dependency/latency/queue or routing effect |

Completion suffixes are `frontier_cycles`, `service_width_cycles`, and
`scenario_end_cycles`. Flat mode lacks decoded topology. Coherence scenarios
are unavailable when `unknown_sm_or_line_records != 0`; this counter is not
proof that every protocol-relevant fact was captured. Writeback rows need a
DRAM observer and the relevant card rate/frontier. Native `TailFlags` and
`timing.completion_frontier` are not cleared by these diagnostics.

Selected populations remain under `trace_metrics.physical_pipeline` and flat
`physical_pipeline_selected_*` names for compatibility; the serialized
`selection_policy` identifies the heuristic. Assignment percentage is candidate
coverage, not hardware assignment accuracy.

Earlier GEMM observations (development data, not universal timing): issue/event
frontier 11,101, writeback scenario end 11,453, coherence scenario end 19,293;
selector coverage 95.58% with 8,704 unknown records. These cannot identify a
bottleneck or rank coherence work against writeback. The [scenario table](sm86_tail_sensitivity_20260926.csv)
retains arithmetic only; [history](HISTORY.md) records the correction.

## Custom ITS trace

`custom_simt_its SPEC.json PROGRAM.json [--trace]` executes the fixed
`simt_its_v1` ISA, one resident group, 1–32 lanes. Input words are aligned u32
byte-addressed values. New `mul_imm`, `load_indirect`, and `store_indirect`
compute addresses from registers; missing reads and simultaneous aliased stores
reject. This is neither CUDA/PTX compilation nor regenerated native SASS.

Without `--trace` the new trace fields are omitted. With it, output adds
`dynamic_trace_schema_version=1`, `dynamic_trace_contract=custom_its_dynamic_v1`
and `dynamic_trace` records (sequence, PC, active mask, instruction, lane
addresses, immediate unblocked successors). Addresses are sampled before an
instruction overwrites registers. Successors for lanes blocked at reconvergence
are omitted; the current output is not a complete per-lane CFG edge oracle.

Capture stops at 100,000 records or the 32 MiB **estimated** record budget
(`128 + 24 * address_entries + 24 * active_lanes` per step). This is not a
hard process-RSS or serialized-byte guarantee. Input/step limits also apply.
The [trace schema](../arch_profiles/schema/simt_its_trace.schema.json) validates
the envelope, not complete operand/state semantics. Tightening these contracts
and adding independent input-varying branch/reconvergence tests remains open.

Runnable smoke commands and schema validation are in [testing](testing.md).
