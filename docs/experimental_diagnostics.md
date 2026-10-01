# Experimental diagnostics and ITS trace contract

Updated 2026-09-27. Consolidates the separate writeback, coherence and pipe
notes. These paths execute as model-local diagnostics; vendor-internal timing
fields remain black-box inputs to the analytical/statistical model.
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
coverage, not hardware assignment accuracy. A separate
`candidate_earliest_completion_v1_model_only` scheduler reports a model-local
frontier and selected route counts using nominal latency and initiation
interval. It also reports per-candidate model-local earliest/latest issue and
completion bounds plus an interval-record count. These are feasible ranges
under the declared model, not physical-pipe measurements or silicon route
labels; unknown/ambiguous domains remain visible.

`tools/nvbit_tracer/physical_pipe_identifiability.py` makes this boundary
executable. Given dynamic candidate sets and aggregate pipe counts it returns
`aggregate_unique`, `aggregate_ambiguous` with the feasible assignment set and
entropy, `route_contract_mismatch`, or `route_unobservable`. Even
`aggregate_unique` means only uniqueness under the supplied aggregate
constraints; it is never a native route witness, and no NCU aggregate is
consumed as a dynamic label.

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
`dynamic_trace_schema_version=3`, `dynamic_trace_contract=custom_its_dynamic_v3`
and `dynamic_trace` records (sequence, PC, active mask, instruction, explicit
per-lane register/predicate state before and after execution, lane
addresses, grouped `successor_masks`, immediate unblocked successors, an
explicit `lane_blocked_by` barrier map, and bounded
`reconvergence_join_pc`/`reconvergence_resume_pc`
metadata). Addresses are sampled before an instruction overwrites registers.
The successor map is only for lanes that remain runnable after the instruction;
the blocked map records lanes still waiting at a barrier. This remains an
execution trace for the custom frontend, not a complete per-lane CFG edge oracle
or a native trace regeneration path. `successor_masks` are the executed edge
partition; reconvergence PCs are populated only from the explicit bounded
BSSY/BSYNC barrier table, not inferred as a universal IPDOM.

Capture stops at 100,000 records or when the accumulated UTF-8 JSON byte lengths
of the individual records exceed 32 MiB. The reported
`dynamic_trace_serialized_bytes` is that record-byte sum; it intentionally
excludes the enclosing result object and JSON array punctuation, so it is a
deterministic record-resource bound rather than a process-RSS or whole-output
size guarantee. Input/step limits also apply.
The [trace schema](../arch_profiles/schema/simt_its_trace.schema.json) validates
the envelope, not complete operand/state semantics. Tightening these contracts
and adding more independent input-varying branch/reconvergence cases remains
open.

The bounded NVBit `record_reg_vals` experiment can be normalized with
`tools/nvbit_tracer/register_value_adapter.py`. The adapter emits
`nvbit_register_value_capture_v1` records with CTA/warp/opcode and lane
register values, but marks identity as `partial` because the text capture has
no function id/PC join, destination writeback values, memory returns, atomic
results, or fence state. An optional `--static-index` JSON document can provide
opcode-to-PC/function candidates; unique joins are retained and ambiguous
matches are explicitly marked rather than guessed. A separate exact
`--dynamic-index` sidecar keyed by capture sequence upgrades the output to
`nvbit_value_trace_capture_v1` and proves a content-bound dynamic
`(function_unique_id, pc, opcode, sequence)` join. This is a finite
value-dependent execution-trace seam, not static SASS regeneration or timing
input; destination/load/atomic/fence capture remains required for a complete
state trace.

The enhanced protobuf tracer's legacy `GATHER_REGISTERS` knob is fail-closed:
it emits a warning and keeps `is_gathered_registers_values=false` because no
register snapshots are serialized into the protobuf instruction stream.

The finite native compiler lane `tools/nvbit_tracer/sm86_compile_manifest.py`
invokes pinned `ptxas -arch=sm_86`, `nvdisasm`, and `cuobjdump --dump-sass`
and emits `sm86_static_sass_artifact_v1` with source/tool/cubin/disassembly
digests. It
is an external NVIDIA-toolchain integration, not arbitrary ISA/compiler
generation and not value-dependent SASS rewriting; target mismatch and tool
failures refuse.

`tools/nvbit_tracer/sm86_value_trace.py` is the first finite execution-driven
consumer of that static artifact lane. It supports scalar `S2R` lane ID,
`MOV`, `IMAD.MOV.U32`, `IADD3`, `ISETP` predicates, predicated arithmetic,
numeric `BRA`,
`LDG.E`, `STG.E`, `NOP`, and explicit `EXIT`; it emits `sm86_value_trace_v1`
with before/after registers, guards, initialized load returns, and stores. An
optional `sm86_static_sass_artifact_v1` manifest must match the SASS SHA-256
before execution, binding dynamic values to the static compiler artifact.
`LOP3.LUT` is admitted for three GPR/immediate/constant inputs, an 8-bit
truth table, and the SM86 data-only or predicate-plus-register destination
forms when the trailing predicate is `!PT`, plus bounded `SHF.L/R.U32`
funnel shifts, immediate-multiplier `IMAD.SHL.U32`, and multiplier-1
`IMAD.IADD`; input-varying lanes, addresses,
predicates, and Boolean results produce different dynamic values, while calls,
special registers, atomics, fences, and unsupported memory forms refuse. This
closes a small functional dynamic-trace mechanism, not hardware-validated
SASS regeneration.

The executor regression suite now runs the same program with all-true and
mixed loaded predicates and checks the emitted successor masks and lane values.

## Custom alternative-resource routing

The custom architecture schema accepts either a scalar `operation_resources`
binding or an array of allowed resource names. The shared event pipeline
canonicalizes alternatives by resource index and selects the candidate with
the earliest modeled completion, considering issue width, latency, initiation
interval and in-flight capacity; equal candidates use the lowest resource
index. Timing events and resource reports expose the selected name. This is a
bounded executable scheduling experiment. It has no NVBit field for physical
SM86 pipe identity and must not be read as a native routing claim.

## Bounded PTX-like frontend

`ptx_subset_compile` admits only `ptx_subset_v2`: lane ID, integer
add/multiply, signed less-than predicates, labels/branches, indirect aligned
u32 global load/store, deterministic functional indirect atomic-add, and exit.
It bounds source bytes, lines, instructions, memory words and generated ITS
steps, rejects unsupported directives/opcodes, and atomically writes a
`simt_its_v1` functional program. It is deliberately not a CUDA compiler,
PTX ABI/linker, complete fence/memory-model frontend, assembler for SASS, or
timing frontend.

`sim-mem::writeback::WritebackQueue` is the corresponding bounded memory
mechanism seam. It accepts exact `L2Eviction` masks, rejects dirty/pending
overlap, preflights finite dirty-sector capacity, prices a deterministic
fixed-point byte service, tracks same-line fetch hazards, and requires a
tokenized completion acknowledgement. `sim-run --mode l1-l2 --carded-dram
--dirty-writeback-capacity-sectors N` enables this queue for one launch and
uses the same opt-in byte server for read fetches and dirty writebacks. The
default carded path is unchanged. Completion counters/frontier are reported
separately from read traffic. This remains a bounded model, not a coherence
directory or a claim about native ownership visibility.

In the same explicit CardedWriteback mode, after all SM/L2 events drain, the
system snapshots resident dirty L2 sectors, transactionally admits them to the
writeback queue in capacity-bounded batches, completes their service tokens,
and commits clean state back to the L2 model. A single line wider than the
queue or a stale-state failure refuses the run. This closes the bounded
resident-L2 flush mechanism, but not native L1 dirty write policy, coherence
visibility, or hardware launch/host flush semantics.

The explicit `--l1-resident-writeback` experiment selects a typed
resident-hit-only L1 policy and now requires the explicit
`--l1-visibility-protocol`: a global store marks a valid L1 sector dirty and
does not emit the ordinary lower L2 mutation. At the terminal boundary the
dirty L1 plan is admitted to the same capacity-bounded writeback queue and is
committed only after service completion. Cold stores still do not allocate or
invent payload state. L1 completion counters and maximum terminal L1 dirty
gauges are reported separately from L2 counters. This is an executable
bounded policy, not proof of native GA102 write-allocate, eviction, fence, or
host-visible flush semantics.

`sim-mem::SectorPayloadV1` is the next opt-in value seam. It carries one
32-byte sector's allocation identity, known-byte mask, sequence, and bytes;
stores and merges are transactional, unknown baselines and identity/order
mismatches refuse, and a stable digest supports conservation tests. The native
replay path still emits only dirty masks, so this sidecar primitive is not yet
evidence that a hardware resident flush wrote the correct DRAM bytes. Until
that payload merge is wired, terminal L1 writeback refuses any overlap between
an L1 dirty mask and an L2 dirty mask on the same line instead of
double-counting mask-only traffic.

The shared `WritebackQueue` also has a payload-aware admission plan requiring
exact known payload coverage for every dirty sector. This is a commit/coverage
mechanism primitive; the native replay path still needs a producer that can
populate it with real store bytes.

`L1Cache::mark_dirty_payload` and `preview_dirty_payloads` now retain and
validate complete known-byte payloads for resident sectors. The
`PayloadCommitLedgerV1` test primitive enforces L1→L2→DRAM commit order and
final-payload digest conservation. The legacy native terminal path remains
mask-only; a mask-only dirty line deliberately refuses a value-bearing payload
preview rather than inventing bytes.

`--coherence-service-cycles N --coherence-capacity-sectors N` additionally
enables a small directory-style service on the ordered L2-request stream. A
read/read share is free; cross-SM owner transfer or write invalidation charges
one configured service interval per affected sector. The model refuses runs
with flat/unknown SM or sector topology and reports completion counters
separately. It does not see L1-hidden accesses, fences, protocol messages,
resident dirty flushes, or hardware ownership state, so it is not a coherence
protocol or silicon timing claim. In particular, a same-launch resident L1 hit
after a remote writer has no modeled L1 invalidation callback and remains
outside this directory seam.

`sim-mem::L1VisibilityQueue` is now an explicitly separate opt-in stateful
seam for that missing callback contract. It consumes one observation for every
eligible L1 access (including resident hits), tracks per-sector sharers/owners,
returns targeted invalidation masks, holds same-line transitions pending until
tokenized completion, and refuses stale/time-regressing/capacity-invalid
transactions. Synthetic read/read and remote-write invalidation tests cover
the queue. The default coherence path remains unchanged. The explicit
visibility-protocol experiment captures only the currently admitted eligible
global L1 observations (ordinary loads/stores). ATOM/RMW, RED/reduction, and
asynchronous-copy paths are refused. Cacheable-miss fill ownership is now
generation-checked, and dropped stale
sector fills refuse rather than publish visibility. Access/fence/fill events
enter the global engine queue; actor batch-ahead can still pre-generate
observations before a remote invalidation is discovered, so
`batch_ahead_visibility_unsupported` remains an explicit bounded limitation.
Unsupported stale/future callbacks or past-time observations remain explicit
refusals. Peer invalidations are applied only for that admitted subset.
Fence/scope, launch-boundary,
initial-resident-state, and completion scheduling remain required before it can
be treated as native timing or clear the coherence tail.

For a bounded event-path experiment, pass
`--l1-visibility-protocol` together with the existing
`--coherence-service-cycles N --coherence-capacity-sectors N` options. The
typed replay configuration records this choice and the system then schedules
tokenized L1-visibility completions, applies targeted `L1Cache::invalidate`
calls, and wakes affected SM actors at the completion frontier. The default
path remains unchanged. Missing fence or initial-state provenance, dirty
conflicts, pending-fill conflicts, and wake ordering conflicts fail closed
rather than being approximated.

The shared SM issue path also recognizes static `CCTL.IVALL` sidecar text in
the opt-in L1-shared mode and invokes the all-L1 invalidation operation. The
visibility directory now receives an ordered requester-scoped invalidate-all
event, removing that requester from owner/sharer state while retaining peer
state. It refuses pending fills or dirty lines instead of silently discarding
state; this is an executable cache-control seam, not proof that every GA102
cache control semantic or fence scope is reconstructed.

The same opt-in path retains typed static `MEMBAR` observations (block/device/
system scope plus sequentially-consistent ordering) alongside L1 accesses.
Accesses and fences now carry one monotonic per-SM sequence, so equal-cycle
events cannot be reordered merely because they were collected in separate
vectors. The visibility queue consumes fence frontiers and refuses per-SM
sequence regressions. These observations are currently evidence/inputs for
the bounded protocol experiment; they do not yet implement full PTX
release/acquire matching, synchronization-object association, or host-visible
flush semantics.

`sim-mem::L1Cache` also exposes explicit `mark_dirty`, transactional
`preview_dirty_flush`/`commit_dirty_flush`, and targeted `invalidate` APIs.
They refuse pending fills, dirty invalidation, and dirty eviction, so a future
SM event layer cannot silently drop dirty data or invalidate an in-flight
response. These APIs
are unit-tested state machinery only; the native replay path does not call
them until store policy, fence scope, and lower-level completion are wired.

Runnable smoke commands and schema validation are in [testing](testing.md).
