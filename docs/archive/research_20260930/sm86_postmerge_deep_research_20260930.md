# SM86 post-merge deep research and blocker-readiness assessment

**Date:** 2026-09-30  
**Repository tip reviewed:** `3c2ff2fa` (`GPUSim_main`)  
**Hardware anchor:** GeForce RTX 3090 Ti / SM86, UUID
`GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, driver 615.71.09, locked
1350 MHz graphics and 10501 MHz memory.

## Executive decision

The post-merge work is sufficient to proceed with a **constrained, evidence-led
resolution pass** for all four blockers and to use the completed cold/warm
full-campaign results as the performance deliverable. It is not sufficient to
claim that any of the four exact physical boundaries has disappeared.

The correct status is:

* **Pipe/issue:** a bounded four-path scheduler-partition and operand-bank
  pressure model is now justified; the dynamic physical pipe/sub-unit/slot
  tuple remains unobserved.
* **Compiler/ISA/ABI:** independent no-template legacy and current ABI-v8
  bodies now execute, including pointer/scalar and corrected semantically
  consumed vector-16 parameter layouts; unrestricted SM86 encoding,
  scheduler/control, relocation, and linker generation remain outside the
  admitted capability.
* **Cache:** public RM flush controls, PTX cache operators, host WT visibility,
  reuse-latency, and cross-SM latency provide intervention and phase priors;
  line owner/read-from/dirty writeback/DRAM completion remain hidden.
* **Timing:** raw-QMD and CUDA-Driver execution of the same body share a
  quantized `%globaltimer` endpoint; CUPTI, NCU, CUDA events, host time, and
  simulator completion still have distinct tails.

Thus the work is ready for model integration and conservative campaign
interpretation, not for renaming latent mechanisms as solved.

## What the committed changes materially add

The latest commits consolidate the simulator contract and the alpha evidence
without changing the native replay contract. The retained observations now
cover independent launch paths, NAK operation families, PMA packet decoding,
CUPTI PC sampling, public RM flush semantics, current-ABI ELF metadata,
operand-bank pressure, scheduler contention, common-device timer spans, cache
latency/reuse, and cold/warm campaign endpoints. The worktree reports preserve
all negative results and explicitly separate NCU score targets from model
features.

The full campaigns are already complete at the locked-clock regime:

* `full_3090ti_20260930_rm_cache_cold_live`: 15/15 workloads, five native
  repeats, three replay repeats, seven NCU repeats, no stage errors; median NCU
  duration 10.080 us, median native 0.009448 ms, median replay 0.629196 s.
* `full_3090ti_20260930_rm_cache_warm_live`: 15/15 workloads, no stage errors;
  median NCU duration 8.704 us, median native 0.009435 ms, median replay
  0.644798 s.

These are the final campaign observations for this clock/cache pair. They are
not evidence that the simulator has acquired the hidden physical state.

## Public research synthesis

### 1. Pipe and issue identity

Nsight Compute documents `sass__inst_executed_per_opcode_pipeline` as an
**estimated** opcode-pipeline count and states that dynamic assignment is not
taken into account. Source instances can correlate metrics with instruction
locations, but this does not expose a per-instruction route/issue ordinal.
The local PMA sweep found no additional route entropy in 2,145 decoded samples.
The scheduler-contention campaign nevertheless supplies a useful intermediate
constraint: under one CTA per enabled SM, logical warp groups with the same
index modulo four serialize at about twice the time of one group, while four
stride-four groups proceed in parallel. The FFMA operand-bank campaign adds
52/67/83-cycle envelopes for 1e1o/2e2o/4e4o source-bank orderings and sampled
per-PC dispatch-stall shifts.

The proper abstraction is therefore a latent route variable
`R=(scheduler_path, execution_class, issue_slot)` with a set-valued domain.
The new experiments constrain only the scheduler-path occupancy and a
bank-pressure term; they do not identify `execution_class` selection for an
individual instruction or a physical issue slot.

### 2. Independent generation and the CUDA ABI

OpenPTXas' public architecture is useful as a decomposition reference: PTX
parsing, register allocation, instruction selection, scheduling, scoreboard
control-word generation, and ELF emission are separate stages. Its current
public implementation targets newer SMs (SM89/SM120 references), so its
pipeline organization is informative but cannot be transplanted as SM86
encoding truth. The independent SM86 work now demonstrates a real no-template
legacy path and a narrow current ABI-v8 path.

The current ABI shape matrix gives a deterministic first layer: reverse
KPARAM records, natural alignment/padding, 8-byte pointers, 4-byte scalars,
and 16-byte vector alignment. The corrected uint4 run showed why this layer
must be represented as typed size/alignment classes rather than a pointer-only
rule: the vector's `0x41f000` size word is distinct from the pointer `0x21f000`
word. A conservative generator can therefore use a capability ledger keyed by
(operation family, parameter-shape class, resource envelope, control-word
recipe) and reject/widen any request outside the ledger.

A field-level repair then fixed the concrete image defects: the text section
symbol was added, `.nv.info` and `.nv.callgraph` now link to `.symtab`, and
`.nv.constant0` carries `SHF_INFO_LINK`. The repaired pointer/uint4/pointer
image is accepted by `cuobjdump --dump-elf` and `--dump-sass`, and the Driver
API still returns `0,3,6,9`. The three-program-header layout is retained
because it is the layout emitted by the observed CUDA 13.x SM86 images; a
six-header layout is not universal. A subsequent minimal DWARF64 CIE/FDE
variant with no debug relocation makes `cuobjdump --dump-elf`,
`cuobjdump --dump-sass`, and `nvdisasm` all complete for the admitted image.
That is image-format/tool-readability closure for one body, not general debug,
relocation, linker, or resource generation. See
`sm86_abi_v8_toolreadable_20260930.md`.

This remains bounded generation. It does not supply general SM86 control words,
all modifiers/opcodes, dynamic shared-memory/resource metadata, relocations,
device functions, cooperative launches, or linker semantics.

A separate branch/control image extends the admitted body family to
pointer/pointer/u32, `ISETP` and predicated `EXIT`, `LOP3`, `IADD3`, descriptor
`LDG`/`STG`, and multiple exit offsets. All three installed binary tools parse
it; five locked-clock Driver runs have a 1.936384-ms median and return the
branch-sensitive prefix `7,4,9,10,11,16,13,22`. Ptxas control words remain
retained for this bridge, so it is ABI/operation-family evidence rather than
independent scheduler/control closure. Report:
`sm86_abi_v8_branch_codegen_20260930.md`.

### 3. Cache ownership and completion

The public PTX contract supplies useful path semantics but not live line state:
`st.wt` to system memory is a write-through intervention, while ordinary
`st.wb`/`st.cg` and stale-L1 behavior are distinct. Public RM source and probes
show accepted invalidate/evict/writeback combinations and pending/outstanding
protocol states, but accepted success can be a no-op for a requested domain;
the GA102 write-through enum is a static policy. The host WT probe therefore
proves only a system-memory visibility path.

Public cache patents describe invalidate/writeback masters and selective dirty
writeback based on downstream occupancy. The cited US20180181488A1 and
US11562459B2 records are ATI/AMD assignee records, not NVIDIA GA102 telemetry.
They establish plausible protocol states, not a GA102 per-line telemetry
interface. Accel-Sim/GPGPU-Sim expose
sectored valid/modified masks, L1 write-through and L2 write-back policies,
MSHRs, and explicit writeback events. Those are excellent *model structures*,
not measurements of the 3090 Ti's owner or read-from source.

The safe cache state is a set-valued per-sector state
`(valid, dirty_mask, owner_domain, source_class, compression, pending_flush)`.
The observed interventions constrain equivalence classes of this state. The
Sattolo sweep (about 39--49 cycles small, 218--231 cycles L2-like,
236--241 cycles large) and corrected cross-SM matrix (286--361 cycles) become
priors on `source_class` and topology, not owner labels.

### 4. Timing and endpoint drift

A semantically matched CUDA-generated body, including in-body `%globaltimer`, ran through
independent raw QMD V03 and CUDA Driver paths with identical quantized spans
(99,328--100,352 ns; median 100,352 ns in the raw/Driver crosswalk). The
raw-QMD extracted header is not byte-identical to the Driver cubin: control
words differ at PCs 0x100 and 0x110 in the store/tail region. The endpoint is
therefore semantic-body-level, not an exact instruction-image identity. The
same-body extension then measured a median globaltimer span of 100,032 ns,
CUPTI activity of about 101,234 ns, NCU duration of 101,600 ns, and CUDA-event
elapsed time about 104,592 ns after setup. These offsets are small but real and
path-specific; CUDA events include a launch/stream envelope.

The correct timing model is an endpoint vector, not a scalar conversion:
`T=(T_globaltimer, T_CUPTI, T_NCU, T_event, T_host, T_replay)` with a
body/envelope-specific transform and an interval tail. A simulator completion
frontier must remain distinct from all acquisition endpoints until a common
completion witness exists.

## Factorized model and novel predictions

The evidence supports a factor graph with four latent factors:

```
route_factor      -> issue/event frontier and aggregate pipe counters
abi_factor        -> accepted image, parameter reads, resource envelope
cache_factor      -> reuse phase, source class, flush pending state
endpoint_factor   -> timer-domain and launch-envelope transforms
```

Each factor emits observations to the corresponding campaign surface; factors
are coupled only through explicit interventions (occupancy, parameter shape,
cache operator, clock regime, or launch path). This prevents a cache-induced
NCU shift from being silently learned as a pipe route or a timer offset from
being learned as simulator cycles.

Falsifiable predictions from the combined evidence:

1. Repeating a matched raw-QMD/Driver timer body at the same clocks should keep
   the two `%globaltimer` distributions within one 1,024-ns quantization bin.
   Changing only QMD metadata or cache intervention should move the raw launch
   envelope without changing the in-body Driver baseline.
2. Repeating the one-CTA/SM contention pattern with a different block shape or
   occupancy should weaken or move the modulo-four serialization. Therefore
   `warpid & 3` is an admitted local pressure feature, not a global physical
   SMSP identifier.
3. A tiny-kernel CUPTI span should remain roughly 1--2% above the in-body
   globaltimer and a post-setup CUDA-event span roughly 4--5% above it. A
   same-body holdout can falsify this endpoint-tail prior.
4. For ABI-v8 vectors, changing only the KPARAM size class from `0x41f000` to
   the pointer class should reproduce a semantic parameter misread rather than
   a new valid layout; the generator should reject that mismatch instead of
   silently widening the capability ledger.
5. Cross-SM `ld.global.cg` latency clusters should remain stable by topology
   group while absolute values move with clock/cache regime. The model should
   widen the source-class interval when the clusters disappear, not assign a
   new owner state.

## Readiness matrix

| Blocker | Current closure level | Proceed now? | Required boundary for exact closure |
|---|---|---|---|
| physical pipe/issue | constrained four-path/bank-pressure prior | **Yes, model integration** | privileged/proprietary per-instruction pipe/slot trace |
| unrestricted compiler/ISA | independent finite v7/v8 operation/ABI family | **Yes, ledger + reject/widen** | general SM86 encoder, scheduler/control, ABI/linker/relocation path |
| cache ownership/flush | protocol/phase/topology intervals | **Yes, scenario model** | RM/firmware line owner/read-from/DRAM completion telemetry |
| unified timing | common `%globaltimer` subcase + endpoint offsets | **Yes, endpoint transforms** | synchronized NCU/CUPTI/event/host/replay/simulator completion |

## Final assessment

The progress is enough to continue productively and to publish the locked-clock
full-campaign performance results. It is not enough to claim full completion of
the four exact blockers. The remaining gaps are not ordinary implementation
bugs exposed by the new evidence; they are observability/authority boundaries.
The next scientifically valid move is to integrate the factorized priors and
intervals into the existing native replay/projection lane while retaining
explicit unsupported cases. Exact closure would require a new privileged
RM/firmware/telemetry endpoint or an external hardware trace that is not
present in the current public/tool-accessible interfaces.

## Public references consulted

* NVIDIA Nsight Compute Profiling Guide: <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>
* Abdelkhalik et al., *Demystifying the Nvidia Ampere Architecture through
  Microbenchmarking and Instruction-level Analysis*: <https://arxiv.org/abs/2208.11174>
* Huerta et al., *CGGTY*: <https://arxiv.org/abs/2503.20481>
* OpenPTXas architecture and SM reference: <https://github.com/garrick99/openptxas/blob/main/ARCHITECTURE.md>
* GPGPU-Sim: <https://github.com/gpgpu-sim/gpgpu-sim_distribution>
* Accel-Sim cache configuration/model notes: <https://github.com/accel-sim/accel-sim-framework/blob/dev/gpu-simulator/gpgpu-sim4.md>
* ATI/AMD high-speed selective cache invalidates/write-backs patent record:
  <https://patents.google.com/patent/US20180181488A1/en>
* ATI/AMD selective dirty writeback patent record:
  <https://patents.google.com/patent/US11562459B2/en>
* CUDA PTX cache operators:
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators>
* Yan et al., *Revealing NVIDIA Closed-Source Driver Command Streams*:
  <https://arxiv.org/abs/2604.26889>
* Pan et al., *Architecting the Next Generation of Asynchronous, Distributed
  GPUs*: <https://arxiv.org/abs/2608.22602>

## RM PTE-information seam (2026-09-30)

A fresh public `NV0080_CTRL_CMD_DMA_GET_PTE_INFO` probe queried local cached,
host-mapped, and system-uncached allocations. Production RM 615.71.09 returns
`NV_ERR_TEST_ONLY_CODE_NOT_ENABLED` for all three, so the documented aperture/
GPU-cached/shader-access/PTE fields are not available as a user-observable
cache-owner route. This strengthens the external cache boundary without
changing the model. Report:
`sm86_rm_pte_info_probe_20260930.md`.

The follow-up patent scan found no new GA102 telemetry seam: newer NVIDIA
sector-level cache patents describe valid/dirty state as implementation detail,
while older flush patents describe conceptual completion protocols. Open RM
headers expose no Ampere LTC/L2 line-state register interface. References:
<https://patents.google.com/patent/US20240330195A1/en>,
<https://patents.google.com/patent/US9218289B2/en>,
<https://patents.google.com/patent/US7631152B1/en>.

A structured-control current-ABI-v8 image now exercises 56 rows including
`BSSY`/`BSYNC`, predicated `BRA` loops, `PLOP3`, and descriptor memory forms.
It parses through all three binary tools and runs through the Driver API at a
2.033664-ms median per 1,000 launches, returning `0,2,5,9,4,6,9,13`. Ptxas
control words remain retained, so this expands the operation/CFG ledger but
not independent scheduler/control generation. Report:
`sm86_abi_v8_loop_codegen_20260930.md`.

## Latest NCU PM-sampling seam (2026-09-30)

The installed Nsight Compute 2026.1 SM86 PM-sampling sections were inspected
again after the branch/loop expansion. The CC86 path exposes timeline aggregates
for `smsp__warps_issue_stalled_*` and aggregate
`smsp__inst_executed_pipe_fmalite/fmaheavy`; the hidden warp-state group is not
currently command-line collectible. `smsp__warpidsamp_*` instances identify
`<SMSP ID>:<Warp ID>` without a physical SM coordinate, while `smsp__pcsamp_*`
correlates sampled state with function PCs. None supplies a dynamic SASS
route/issue ordinal plus physical SM/SMSP identity. This is a finite negative
for the exact-pipe blocker, not a reason to promote aggregate PM sampling.

A local `ncu --query-metrics` check confirms that GA102 warpsampling exposes
only `smsp__pcsamp_warps_issue_stalled_*` (and `_not_issued`) metrics; no
`warpidsamp` instances are present in the command-line metric surface. The
installed UI-only PM-warp group therefore does not supply a CLI physical
SMSP/route/issue endpoint.

The installed NVIDIA 615.71.09 control ABI was also scanned for topology and
profiler-monitor queries. It exposes GPC/TPC/PPC/ROP/SMC/syspipe masks, but no
SM/SMSP identity, warp scheduler, dynamic pipe, or issue-slot fields. This is
an ABI-level negative independent of the Nsight metric query.

The same control-ABI scan found `NV2080_CTRL_CMD_GR_GET_SM_ISSUE_RATE_MODIFIER_V2`
and `NV2080_CTRL_CMD_GR_GET_SM_ISSUE_THROTTLE_CTRL`, with static categories
(FMLA16/DP/FMLA32/FFMA/IMLA/FP16/FP32/DFMA/DMLA/CTRL). These are useful
instruction-class throughput priors and throttle controls, but they are not
dynamic per-instruction route, scheduler, or issue-slot receipts.

## Long-body timing closure subcase (2026-09-30)

A 1,350,000-clock version of the matched timer body was run through raw QMD,
Driver, CUPTI, CUDA events, and NCU. With external clocks locked and explicit
`ncu --clock-control none`, medians were 1,000,448/1,000,448/1,000,032/
1,004,528/1,001,600 ns respectively. Default NCU boost control instead
produced 715,824 ns. This closes a practical drift-control subcase and proves
profiler clock policy must be modeled explicitly; simulator completion remains a
separate endpoint.

## RM issue-modifier control probe (2026-09-30)

The initial device-handle result was corrected: GR controls require
`NV20_SUBDEVICE_0`. Correct subdevice reads return static rate modifiers
(`fmla32=1`, `imla4=1`; v2 entries `0,0,1,0,0,0,0,0,1`). Public throttle and
internal/static KGR variants return `NV_ERR_NOT_SUPPORTED`. The static values
are instruction-class priors only; they do not provide dynamic route,
subunit, scheduler, issue ordinal, or physical SMSP identity. Report:
`sm86_rm_issue_modifier_subdevice_20260930.md`.

The disposable loop control mutations are retained as hardware sensitivity
observations, with corrected SM86 field labels documented below. They show that
reuse, dependency-field, and delay-field changes affect function and/or timing;
they do not establish a physical pipe or issue-slot witness.

## Causal corrections (2026-09-30)

A field-layout audit corrected the SM86 control-code interpretation. SM70+
fields are delay bits 0..3, yield bit 4, write barrier bits 5..7, read barrier
bits 8..10, wait-mask bits 11..16, and reuse bits 17..20. The earlier disposable
mutations were therefore reuse-field, isolated LDG wait-mask, and delay-field
mutations; their observations remain hardware results, but the earlier
“stall/misc” labels are withdrawn. The temporary rule-driven synthesizer
used the wrong SM120 packing and is not promoted.

The RM issue-control probe also required the `NV20_SUBDEVICE_0` handle. Correct
subdevice reads return static rate modifiers (`fmla32=1`, `imla4=1`, v2
`0,0,1,0,0,0,0,0,1`); throttle and internal/static KGR variants remain
unsupported. The prior device-handle `NOT_SUPPORTED` result is retained only as
a handle-targeting negative.

The `0x27ed -> 0x7ed` mutation is now classified precisely: it clears wait-mask
bit 2 (`0x2000`) while leaving read-barrier bits unchanged. Its deterministic
stale/early output is therefore a direct LDG scoreboard wait-mask witness,
though still not a physical pipe/issue-slot witness.

## Independent control-word generation point (2026-09-30)

After correcting the SM86 field layout, a rule-driven generator emitted all 56
structured-loop control words without copying ptxas control values. It used
opcode-class templates plus explicit S2R/LDG wait-mask and loop-reuse dataflow
rules. The image runs through Driver/nvdisasm with correct output and a
2.092032-ms median versus 2.025472 ms for the ptxas-control image. This is a
real finite scheduler/control capability expansion, not unrestricted SM86
compiler closure. Report:
`sm86_abi_v8_independent_control_codegen_20260930.md`.

The corrected independent control synthesizer now also emits the branch/predicate
body without copying ptxas control words. The Driver/nvdisasm image returns
`7,4,9,10,11,16,13,22` with a 1.949344-ms median versus 1.936384 ms for the
ptxas-control image. This broadens finite operation-family generation while
leaving unrestricted SM86 scheduling and linker/resource semantics open. Report:
`sm86_abi_v8_independent_control_branch_20260930.md`.

A descriptor global-memory body (ptxas lowering of an inline cache-operator
sequence) was independently re-encoded and control-scheduled with corrected
SM86 wait fields. It runs without ptxas control words and returns
`1,2,3,4,5,6,7,8` at a 1.892352-ms median. This broadens finite
cache-sensitive codegen coverage but does not expose cache ownership or close
unrestricted generation. Report:
`sm86_abi_v8_independent_control_cache_operator_20260930.md`.

A reduction/atomic current-ABI-v8 body was independently re-encoded and control-
scheduled, covering VOTEU, UFLO, REDUX, and predicated RED.E.ADD.STRONG.GPU.
The Driver image returns `32896000` with a 1.885184-ms median over five runs.
This broadens finite control/codegen coverage to atomic/reduction operations,
not unrestricted compiler or cache-owner closure. Report:
`sm86_abi_v8_independent_control_atomic_20260930.md`.
