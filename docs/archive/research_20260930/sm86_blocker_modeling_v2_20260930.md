# SM86 blocker modeling v2 — factorized latent mechanisms and campaign predictions

**Date:** 2026-09-30  
**Hardware scope:** GeForce RTX 3090 Ti / GA102 / SM86, driver 615.71.09,
graphics/SM 1350 MHz, memory 10501 MHz.  
**Purpose:** research synthesis and a conservative model boundary after the
committed SM86 changes and the resumed deep-reading pass. This document is
not a claim that the four physical blockers are solved, and it does not
replace the native replay, NCU projection, or campaign contracts.

## Executive decision

The committed work is sufficient to **proceed with the useful approximation
and performance lane** without regressing accuracy: native replay, independent
NAK bodies, the bounded CUDA-ELF bridge, RM protocol observations, endpoint
descriptors, interval committees, and complete cold/warm campaigns are all
usable. It is **not sufficient to claim closure of any of the four exact
mechanism blockers**:

1. physical pipe/sub-unit/issue-slot identity;
2. unrestricted independent SM86 ISA/control-word/ELF/ABI generation;
3. cache-line ownership/read-from/dirty-writeback/DRAM completion;
4. one drift-free endpoint shared by NCU, CUPTI, CUDA events, device timers,
   simulator completion, and replay wall time.

The correct next step is therefore a factorized, set-valued model that can
make predictions and carry the performance campaigns forward while preserving
unknown fields. A single guessed pipe, owner, compiler ABI, or time conversion
would reduce scientific validity.

## 1. What the recent changes establish

### 1.1 Native and independent code paths

* The existing NVBit/SASS -> trace -> SM/memory/event replay remains the
  reference path. Its issue, modeled-event, launch-envelope, typed-tail, and
  completion frontiers remain separate.
* Mesa NAK emits independent SM86 operation-family bodies. An independently
  built Ampere QMD V03 can launch a NAK body without CUDA ELF/libcuda; the
  direct body returns the expected values. This proves bounded native body and
  launch control, not a general CUDA ABI.
* The NAK-to-CUDA-ELF bridge executes the same bounded bodies through the CUDA
  Driver API. The bridge reuses NVIDIA-generated metadata and a hand-covered
  parameter layout, so it is a container/bridge certificate, not a relocatable
  ABI certificate.
* Public NVIDIA PTX Compiler/nvJitLink paths provide complete PTX-to-cubin
  lowering, but the lowering, scheduling decisions, control words, metadata,
  relocations, and ABI are proprietary.

### 1.2 Pipe and PMA surface

The installed Nsight Compute guide explicitly defines
sass__inst_executed_per_opcode_pipeline as an *estimated* opcode-pipeline
count and states that dynamic assignment is not taken into account
([metric documentation](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)).
The source page therefore cannot join a dynamic PC/warp to a physical pipe or
issue ordinal.

The retained PMA blobs contain PC/stall/wave/active information. Reassembly
of 2,145 decoded samples found no additional high-bit route/slot entropy
(bit 63 was always zero, bits 61:60 were always 11, and the top nibble was
only 0x3/0x7 under the active/not-issued split). This is a negative result
for this decoder and stimulus family, not proof that all future PMA modes lack
such a field.

### 1.3 Cache/RM surface

The 615.71.09 open RM source maps public flags to
FB_CACHE_INVALIDATE, FB_CACHE_WRITEBACK, and FB_CACHE_EVICT, and routes
full-cache requests through the architecture cache-op HAL
([RM control source](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/615.71.09/src/nvidia/src/kernel/gpu/mem_sys/kern_mem_sys_ctrl.c)).
The GA102 generated HAL classifies the L2 write mode as WRITETHROUGH
([generated GA102 policy source](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/615.71.09/src/nvidia/generated/g_kern_mem_sys_nvoc.c)).
Those are driver policy/protocol facts, not live per-line state.

On this card, full-cache invalidate/evict combinations and dirty/system
combinations are accepted in the public RM surface, while video-memory
writeback-only and user-client internal controls have explicit unsupported
cases. The per-surface probe accepts invalidate/evict, rejects writeback-only,
and does not expose physical cache attributes. A successful control return
can therefore establish a protocol result for a supported path, but not a
line owner, read-from source, or DRAM acknowledgement.

The PTX model independently requires candidate-write reasoning: a read may
return a write from a constrained set, including initialization, rather than
naming a physical cache owner
([PTX memory consistency model](https://docs.nvidia.com/cuda/archive/13.0.1/hopper-tuning-guide/parallel-thread-execution/index.html#memory-consistency-model);
[formal analysis](https://research.nvidia.com/publication/2019-04_formal-analysis-nvidia-ptx-memory-consistency-model)).
CUDA cache-policy hints are explicitly performance hints and do not change
memory consistency semantics
([L2/cache-policy documentation](https://docs.nvidia.com/cuda/archive/13.0.1/hopper-tuning-guide/parallel-thread-execution/index.html#cache-hint)).

### 1.4 Timing and campaign surface

A raw-RM method stream now writes four-word report semaphores before and after
a QMD launch, with explicit WFI. NVIDIA's open semaphore definitions and
modeset source interpret the report timer as a device nanosecond timestamp
([semaphore definitions](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/615.71.09/src/common/inc/nvSemaphoreCommon.h);
[modeset timestamp extraction](https://fossies.org/dox/NVIDIA-open-gpu-kernel-modules-615.71.09/nvkms-headsurface-3d_8c_source.html)).
After setup, 31 intervals have a median raw GPU delta of 3.072 us and host
monotonic delta of 15.370 us. This is a valid raw-RM endpoint for that method
stream, not an NCU/CUPTI equivalence.

The body-matched crosswalk used the same 14-instruction NAK body and static
10-ALU/3-LSU population. It was not initially launch-equivalent: raw QMD used
an 8x1x1 local size without profiler cache intervention, while NCU used a
32x1x1 block and --cache-control all. Raw GPU median was 4.048 us; NCU
median was 2.496 us / 3,310 GPC cycles. A bounded 8x1x1 NCU launch returned
2.496 us / 3,296 cycles with unchanged instruction populations, making block
size an unlikely dominant explanation. Cache intervention, QMD/metadata,
and endpoint envelopes still differ, so 1.62 is not a conversion.

The full 15-row campaigns are performance evidence:

| campaign | NCU cache mode | rows | elapsed | median native | median NCU duration | median NCU GPC | median replay |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_3090ti_20260930_rm_cache_cold_live | all | 15/15 | 487.072 s | 0.009448 ms | 10.080 us | 13,572 | 0.629196 s |
| full_3090ti_20260930_rm_cache_warm_live | none | 15/15 | 487.560 s | 0.009435 ms | 8.704 us | 11,709 | 0.644798 s |

Per-row warm/cold NCU duration and GPC ratios are nearly identical and range
about 0.685--1.025; native CUDA-event ratios are about 0.994--1.011 (median
0.9995), and simulator SM/GPC frontiers are exactly 1.0 for 13/15 rows. This
is strong cache/endpoint-domain separation, not evidence of clock drift.
Within NCU, duration is internally coherent with GPC cycles and the measured
per-row GPC frequency (median duration/cycle-derived ratio approximately
0.9997). The campaign remains clock-locked, but clock locking does not make
the endpoints identical.

Nsight Compute documents the cause: it inserts front-end WaitForIdle and
semaphore timestamps around the grid, flushes caches between replay passes by
default, and uses a different start/end mechanism from pre-Blackwell
Nsight Systems/CUPTI
([NCU duration semantics](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)).
CUPTI also converts activity timestamps to CPU time through interpolation,
and CUDA events report stream-front timestamps; these are different endpoint
contracts.

## 2. Research synthesis

### 2.1 Modern Ampere issue modeling is a useful prior, not a pipe witness

Huerta et al., **Analyzing Modern NVIDIA GPU Cores** (arXiv:2503.20481),
reverse-engineer compiler-guided dependence handling and a
Compiler-Guided-Greedy-Then-Youngest (CGGTY) issue policy
([paper](https://arxiv.org/abs/2503.20481)). Their controlled SASS/CLOCK
experiments support:

* one issue per sub-core per cycle as a lower-bound model;
* compiler-provided stall/yield and dependence-counter bits;
* fixed-latency readiness versus variable-latency dependence counters;
* greedy same-warp issue while eligible, then youngest ready warp on switch;
* register-file/cache and memory-pipeline constraints.

The paper says the model fits almost all measured cases, not every case, and
the scheduler observation was established for warps within the same CTA.
It validates aggregate timing on RTX 3090 (reported about 17.93% MAPE) and
other Ampere devices, but it does not emit a physical per-instruction
pipe/sub-unit/issue-slot tag. This is exactly the role of a mechanism-local
prior in this project: it can constrain readiness and capacity without
claiming route identity.

Abdelkhalik et al., **Demystifying the Nvidia Ampere Architecture through
Microbenchmarking and Instruction-level Analysis** (arXiv:2208.11174),
shows that PTX-to-SASS mapping is context-sensitive: dependency structure,
operand initialization, and compiler choices can change the selected SASS form
and latency
([paper](https://arxiv.org/abs/2208.11174)). Its measured memory and
instruction latencies are useful priors, but the study used A100 rather than
this GA102 card and does not expose dynamic pipe receipts.

NVIDIA's GA102 whitepaper gives structural capacity bounds—four SM partitions,
each with a warp scheduler/dispatcher and FP32 datapaths
([GA102 whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)).
It does not disclose dynamic issue tags.

### 2.2 Independent encoding tools stop short of unrestricted SM86 generation

The 2026 nvsass-disassembler work provides a machine-readable
encoding/disassembly specification and large cross-generation decode corpus
([paper](https://arxiv.org/abs/2609.18662);
[repository](https://github.com/reoLantern/nvsass-disassembler)). Its public
README does not list SM86 among the supported/tested architectures and it does
not generate a relocatable CUDA ABI, QMD, or scheduler-control stream.

CuAssembler is valuable for controlled binary patching, but its own README and
UserGuide label SM86 support as early/poor and require a compiler-generated
template; local inspection of DefaultInsAsmRepos.sm_86.txt found 425
instruction-family keys and 20 known re-encode error records, including S2R,
ULDC, STG, BRA, MUFU, RET, SHFL, LDG, ATOM, LDGSTS, and related forms
([repository](https://github.com/cloudcores/CuAssembler)). This is not an
unrestricted encoder. The bare-metal cuasmR line explicitly patches sibling
instruction/control words rather than assembling arbitrary text
([repository](https://github.com/pjt222/bare-metal)).

Xtrace/G-Watch is a promising binary-level splice/tracing system for SM86
([arXiv:2609.28769](https://arxiv.org/abs/2609.28769);
[project](https://g-watch.github.io)). Its trace schema supplies PC/site,
timestamp, value, and SMID/clock anchors, but no physical pipe, sub-unit,
issue-slot, or dynamic issue ordinal. The local SM86 splice probe parsed an
image but failed to launch with CUDA error 715, so it contributed no local
hardware trace.

CUTracer's current public source similarly records dynamic PC/opcode,
register/memory values, active masks, and host-received timestamps, but no
physical route tag
([trace record source](https://github.com/facebookexperimental/CUTracer/blob/main/include/common.h)).
It improves semantic trace capture and delay-injection experiments; it does
not close the pipe witness.

## 3. Factorized set-valued model

### 3.1 Latent factors

For every dynamic event e, retain four independent latent factors:

    Z(e) = (R_e, A_e, C_e, T_e)

R_e : physical route/sub-unit/issue-slot assignment
A_e : compiler/control/ELF/ABI realization
C_e : cache owner, residency, read-from, dirty/writeback state
T_e : endpoint start/end and clock-domain realization

The simulator must never collapse these into one scalar “hardware truth.”
A model world w is an assignment of all latent factors consistent with the
observed facts, intervention controls, and architecture-card bounds. Evidence
shrinks the admissible world set W; it cannot create a singleton by convention.

For a field f, expose:

    must(f) = intersection { f(w) : w in W }
    may(f)  = union        { f(w) : w in W }

Report an exact value only when must(f) == may(f) and the provenance is direct
or mechanically derived. Otherwise report an interval, candidate set, partial
order, or unknown. A singleton hypothesis remains a hypothesis.

### 3.2 Event and frontier identities

Use a typed key independent of array order:

    EventKey = {
      launch_uid, function_uid, cta_uid, warp_uid, instruction_ordinal,
      pc, lane_or_sector, sm_id, producer
    }

Preserve the current frontier vector:

    F = {
      issue, modeled_event, launch_envelope, typed_tail,
      visibility, writeback_service, endpoint_start, endpoint_end
    }

Only add an edge when the producer contract establishes it. For example,
issue <= modeled_event <= typed_tail is a simulator ordering, while
NCU_start <= first_scheduled_CTA follows from NCU's documented front-end
semaphore placement. There is no default edge NCU == CUPTI,
globaltimer == NCU, or typed_tail == DRAM.

### 3.3 Intervention matrix

Treat each campaign choice as an explicit intervention:

| intervention | changed factor | held fixed | admissible observation |
|---|---|---|---|
| raw QMD vs CUDA-ELF launch | A,T and launch envelope | body semantics | endpoint/ABI comparison only |
| NCU cache all vs none | cache/replay state and NCU envelope | binary and campaign shape | phase response, not owner |
| NCU/CUPTI/event/raw timer | T | as much launch state as possible | endpoint descriptor/crosswalk |
| clock lock/profile mode | clock-control regime | workload and binary | drift interval, not identity |
| independent NAK body | A producer | operation-family semantics | bounded codegen capability |
| PMA/PC sampling | observation surface | launch | statistical state evidence |

A comparison can support a causal claim only about factors that were held
constant by construction. The original 1.62 comparison had block/cache
confounds; the 8x1x1 NCU control removes the block-size confound but not the
other two.

## 4. Blocker-specific model

### 4.1 Physical pipe/issue witness

For each instruction i, define an exact semantic demand class and a candidate
resource domain:

    D_i subset of { ALU, FMA-heavy, FMA-lite, LSU, XU/SFU, tensor, control }
    x[i,r] in {0,1}
    sum_r x[i,r] = 1
    x[i,r] = 0 when r is not in D_i

Use GA102 partition capacity, compiler control bits, dependency barriers,
memory issue queue constraints, and PM/NCU aggregate counts as bounds. Since
the NCU pipeline count is estimated, an aggregate observation gives a count
interval [C_r-delta_r, C_r+delta_r], not a per-PC assignment.

Schedule with the existing max-plus frontiers:

    issue_i >= max(
      prior_warp_ready,
      stall_i,
      dependence_counter_ready_i,
      resource_free_i,
      memory_queue_ready_i
    )
    completion_i >= issue_i + latency_i

If several route assignments satisfy all bounds, emit the candidate set and
keep simulator timing on the current logical/resource path. Do not pick the
assignment that merely balances counts. A future exact witness must carry at
least (PC, warp/subpartition, issue ordinal or cycle, physical pipe/sub-unit)
from hardware/RM/firmware; aggregate NCU, CUPTI, PMA, or Xtrace records cannot
satisfy this contract.

**Prediction:** mixed FP32/INT32 demand can share the GA10x dual FP32/INT32
capacity, while all-FP32 demand can use both FP32 datapaths. The model should
predict capacity pressure and intervals, but should never label the route of a
particular dynamic PC.

### 4.2 Independent compiler/ISA/ABI generation

Represent generation as four separately certifiable edges:

    Semantics -> instruction word -> control/scheduler metadata
               -> ELF/relocations/constants -> QMD/launch ABI

An artifact carries a capability set:

    Coverage = {
      architecture, opcode/forms, modifiers, register/address modes,
      control fields, relocation kinds, ELF sections, parameter layout,
      QMD fields, launch producer, unsupported cases
    }

Current capability tiers:

1. **Official PTX:** complete for the exposed PTX/toolchain contract, but
   proprietary and not an independent SASS/ABI witness.
2. **NAK/raw QMD:** independent native body and bounded launch contract;
   operation-family and QMD coverage is explicit.
3. **NAK/CUDA-ELF bridge:** proven for its template and parameter layout only.
4. **CuAssembler/template patching:** useful controlled edits, but 425 SM86
   family keys include 20 known re-encode mismatches and require a
   compiler-generated template.
5. **nvsass-disassembler:** decode/re-encode research input; no SM86-complete
   ABI/control/QMD generator.

Unknown forms, modifiers, relocation kinds, or metadata are an explicit
unsupported/fallback outcome. Closure requires a complete independent SM86
encoder, scheduler/control-word generator, relocatable ELF and kernel ABI,
QMD/resource descriptors, and native launch acceptance across ABI shapes.

### 4.3 Cache ownership and complete writeback

Keep architectural candidate semantics separate from physical residency.

For each line/sector:

    RF(read) = candidate writes allowed by PTX scope/order/causality
    owner_must, owner_may
    resident_must, resident_may
    dirty_must, dirty_may
    read_from_must, read_from_may
    writeback_service_must, writeback_service_may
    dram_persisted_must, dram_persisted_may

Transitions:

* A GPU store adds an architectural write and a modeled dirty candidate only
  when the selected cache policy admits it; it does not assert DRAM visibility.
* A fill publishes L1/L2 residency only at the observed fill callback and only
  for its allocation generation.
* A fence adds scope/order edges; it does not name a physical owner.
* A public RM flush records the aperture, operation (invalidate/writeback/
  evict), acceptance, and source-derived completion semantics. It does not
  set dram_persisted_must unless a lower-level acknowledgement and an
  independent consumer are available.
* A CPU/BAR1/DMA consumer reduces candidate read-from worlds only when its
  mapping and bypass/ordering path are independently classified.
* A value match before/after flush proves value equality under the experiment,
  not the source of that value.

The model should expose FlushProtocolComplete,
WritebackServiceComplete, and DRAMPersisted as distinct typed facts. On the
present evidence DRAMPersisted remains unknown for a generic line. NVIDIA's
public CUDA docs also state that cache-policy hints need not be respected and
do not alter memory-consistency behavior, so a hint cannot be promoted to an
ownership fact.

**Prediction:** cold/warm NCU phase changes should alter DRAM/L2 sector
counters and the NCU endpoint while leaving architectural value and simulator
issue frontiers unchanged. This is exactly the pattern in the 15-row
campaigns; it is not a line-owner proof.

### 4.4 Drift-free matched timing

Give every source an endpoint descriptor:

    Endpoint = {
      clock_domain, start_trigger, end_trigger, included_work,
      excluded_work, serialization, cache_state, clock_control, uncertainty
    }

Current descriptors include simulator issue/event/tail, NCU
WaitForIdle+semaphore, CUPTI activity, CUDA events, raw RM report timer,
globaltimer, host monotonic, and replay wall time.

For a deliberately matched pair, permit a local interval transform:

    t_B = a * t_A + b + epsilon
    epsilon in [epsilon_lo, epsilon_hi]

Bind a,b,epsilon to binary identity, launch geometry, driver/tool version,
clock regime, cache intervention, and endpoint descriptors. Never use a
crosswalk learned from a scored NCU row to choose that row's simulator
parameters.

The full campaign predicts:

* NCU duration/cycle ratios remain near one internally, because both are
  measured in the NCU envelope.
* warm/cold NCU ratios can change by approximately 17% while native event and
  simulator frontiers remain near one, because cache/replay state and endpoint
  inclusion differ.
* raw GPU-timer versus host time has a large submission/polling envelope for
  very small launches.
* no scalar conversion can map raw QMD, NCU, CUPTI, CUDA-event, and simulator
  time without holding trigger and included-work semantics constant.

Unified timing closure requires one public endpoint whose start/end triggers
and included work are proven identical to the target endpoint. Correlation,
clock locking, or an affine fit alone is not such a proof.

## 5. Non-regression contract for implementation

The model can be added beside the current replay without changing promoted
outputs:

1. **Opt-in only:** route, compiler capability, cache-world, and endpoint
   envelopes are diagnostics/interval outputs unless explicitly selected.
2. **Baseline freeze:** disabling the layer preserves current instruction
   counts, frontiers, cache transitions, campaign summaries, and performance
   results.
3. **Unknown-preserving arithmetic:** every new lower/upper bound uses checked
   arithmetic; an empty feasible set is an error/fallback, never a repaired
   route.
4. **No label leakage:** NCU labels from a scored workload cannot select its
   route, cache state, or timing parameters.
5. **Scope binding:** every artifact and crosswalk records GPU identity,
   driver/tool version, clock regime, cache intervention, operation family,
   ABI shape, and endpoint descriptor.
6. **Typed completion:** protocol retirement, modeled writeback service,
   visibility, and DRAM persistence remain different result fields.
7. **Provenance:** facts, derived bounds, hypotheses, and unobservable fields
   are distinct in serialized output.
8. **Fallback over invention:** unsupported SASS forms, missing dynamic joins,
   unknown owner state, and unmatched endpoint pairs emit a widened
   candidate_fallback or unknown, never a nearest-neighbor fact.

## 6. Closure conditions and stopping rule

The current research has exhausted the public local seams needed for the
approximation lane. Further progress on exact blockers requires an external
state change:

* a hardware/RM/firmware stream with dynamic pipe and issue tags;
* a complete independent SM86 assembler/linker/ABI/QMD generator;
* owner/read-from/dirty-writeback/DRAM telemetry or a guaranteed bypass
  consumer plus completion acknowledgement;
* a public endpoint contract with identical start/end and included work across
  NCU/CUPTI/device timer/simulator.

Absent one of those, the model should continue to produce useful campaign
predictions and final performance results, but the four exact fields must
remain set-valued/unknown. Claiming otherwise would regress scientific
accuracy even if a point estimate looks closer on the current 15 rows.

## References and retained local artifacts

* Huerta et al., arXiv:2503.20481 — modern NVIDIA issue/resource model:
  <https://arxiv.org/abs/2503.20481>
* Abdelkhalik et al., arXiv:2208.11174 — Ampere PTX/SASS
  microbenchmarks: <https://arxiv.org/abs/2208.11174>
* Automated SASS encoding synthesis, arXiv:2609.18662:
  <https://arxiv.org/abs/2609.18662>
* Xtrace, arXiv:2609.28769:
  <https://arxiv.org/abs/2609.28769>
* CUDA Programming Guide L2 control:
  <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html>
* PTX 8.8 memory consistency:
  <https://docs.nvidia.com/cuda/archive/13.0.1/hopper-tuning-guide/parallel-thread-execution/index.html#memory-consistency-model>
* NVIDIA open RM 615.71.09:
  <https://github.com/NVIDIA/open-gpu-kernel-modules/tree/615.71.09>
* Mesa NAK QMD/source:
  <https://github.com/chaotic-cx/mesa-mirror/tree/main/src/nouveau/compiler/nak>
* CuAssembler SM86:
  <https://github.com/cloudcores/CuAssembler>
* CUTracer trace schema:
  <https://github.com/facebookexperimental/CUTracer>
* Local evidence:
  docs/archive/research_20260930/sm86_blocker_modeling_20260929.md,
  sm86_same_body_endpoint_crosswalk_20260930.md,
  sm86_open_rm_cache_semantics_20260930.md,
  sm86_pipe_metric_surface_20260930.md,
  sm86_full_campaign_rm_cache_20260930.md, and the
  results/alpha/full_3090ti_20260930_rm_cache_{cold,warm}_live/ trees.


## 7. Additional 2026-09-30 source and microbenchmark findings

### 7.1 DocumentSASS SM86 resource vocabulary

The public DocumentSASS extraction path embeds SM86 operation and latency
descriptions in the CUDA toolchain's nvdisasm output
([DocumentSASS](https://github.com/0xD0GF00D/DocumentSASS)). An external
intercept on the local CUDA 13.2 toolchain recovered SM86 semantic operation
sets and a pipeline-resource block. The recovered classes include
int_pipe, mio_pipe, fe_pipe, fmalighter_pipe, fp16_pipe, cbu_pipe,
fma64lite_pipe, fma64heavy_pipe, and udp_pipe. The resource block expresses
occupancy-style capacities such as FXU/FMAI/FMALITE occupancy 1, operation
occupancies of 2, HMMA using FMALITE, IMMA using FXU, DMMA using FMALITE, and
BAR using a control/barrier resource.

This is a useful independent **semantic candidate-domain and capacity prior**
for the route envelope. It is not a dynamic issue receipt: the source is
compiler/disassembler metadata, the extraction marker changed in CUDA 13.2,
and the project notes caution that latency/resource interpretations need
microbenchmark validation. Keep the classes as D_i candidates and keep the
physical sub-unit and issue ordinal latent.

### 7.2 Explicit reject-on-unsupported generation

F2Asm (arXiv:2608.20532) is a current state-of-the-art example of a safer
generation contract
([paper](https://arxiv.org/abs/2608.20532);
[project](https://github.com/JiadingGai/f2asm)). Its encoder reduces a query
against an architecture-specific basis and rejects nonzero residuals instead
of emitting an unsupported instruction. Its onboarding requirements include
a target profile, a verified CUBIN corpus, a pinned disassembler, control
fields, relocation/metadata rules, and round-trip constraints. The paper
explicitly separates scheduler controls and preserves opaque encoding bits;
it does not synthesize or optimize scheduling controls.

F2Asm currently targets SM90/90a, SM100/103, and SM107, not SM86. The safe
lesson for this repository is the rejection policy, not a claim of portability:
an SM86 generator should emit only forms whose instruction fields, hidden
control bits, relocations, ELF metadata, and ABI are in its admitted corpus.
Unknown residuals must become unsupported/fallback.

CuAsmRL (arXiv:2501.08071) provides a related schedule-reordering research
line, but it evaluates SM80/A100, simplifies or ignores control codes in
parts of its learned representation, and leaves manual kernel verification
to users ([paper](https://arxiv.org/abs/2501.08071);
[artifact](https://github.com/hgl71964/cuasmrl/tree/reproduce)). Its result is
a dependency-aware optimization prior, not an exact SM86 pipe witness or
unrestricted generator.

### 7.3 Cache-discard and forwarding observation

A bounded external run of the public Invalidate+Compare source
([repository](https://github.com/0x5ec1ab/invalidate-compare)) on the locked
RTX 3090 Ti showed an important negative: a store/fence/discard/load sequence
returned different values when a final printf/control path was present versus
removed; a single-block variant could return a stale sentinel after
discard, while other repetitions returned a different value. The observation
is not promoted as a deterministic cache result. It demonstrates that
compiler control dependence, LSU forwarding, cache-hint semantics, and launch
population can affect the read path.

The USENIX Security 2024 Invalidate+Compare study reports a non-inclusive,
incoherent-L1/L2 behavior hypothesis, L1 write-through/L2 write-back behavior,
write allocation, context-switch L1 flushing, and sectored dirty state on
tested RTX3060/3080/A2/A10/RTX4060 devices
([paper](https://www.usenix.org/system/files/usenixsecurity24-zhang-zhenkai.pdf)).
Those cards are not the local 3090 Ti, so these are cross-generation priors,
not card-specific facts. They nevertheless justify preserving L1 noncoherence,
sector masks, forwarding candidates, and an unknown read-from source.

The public PTX 8.5 semantics also distinguish cache operators: default global
loads may allocate L1/L2; cg accesses L2; wt targets system memory; and
discard.L2 invalidates a 128-byte range without writeback, leaving subsequent
data undetermined. These are architectural semantics, not an owner/DRAM
telemetry stream.

### 7.4 Patent-informed hidden cache state

NVIDIA patent US9639466B2 describes dirty-line thresholds, per-sector dirty
bits, clean notifications, read-fill arbitration, and a spectrum between
write-through-like and write-back-like behavior
([patent](https://patents.google.com/patent/US9639466B2/en)). US12314175B2
describes sector-level residency controls in a sectored shared cache
([patent](https://patents.google.com/patent/US12314175B2/en)). Patents are
design disclosures rather than proof of the GA102 implementation, but they
make the latent model more realistic: the RM write-through enum cannot
eliminate hidden dirty thresholds, compression/residency state, sector-level
validity, or DRAM arbitration.

The cache state should therefore include:

    L1[SM] : valid/data/dirty/epoch
    L2     : valid sector mask, dirty sector mask, owner/sharer candidates,
             compression/residency priority, pending fill/clean/writeback
    FB/DRAM: backing value plus persistence candidate
    E      : endpoint/flush epoch and protocol completion token

### 7.5 Fixed-frequency CUPTI PM sampling

A local build of the public Accel-Sim CUPTI PM-sampling wrapper used the
CUPTI GPU_TIME_INTERVAL trigger (500 ns) on a repeated 14-instruction,
8-thread NAK bridge launch. Post-setup PM windows measured 479--513 ns
(median 512 ns), and 786 of 4,096 windows contained nonzero
sm__inst_executed samples. This is stronger evidence for a fixed-frequency
sampling clock domain than a host timer, and it can be retained as an
endpoint descriptor. It still has no kernel start/end ownership or issue
ordinal: windows mix setup and repeated launches and split the body across
bins. CUPTI documentation therefore supports a **sampling-domain** endpoint,
not a unified kernel-duration endpoint
([CUPTI PM sampling documentation](https://docs.nvidia.com/cupti/main/main.html)).

### 7.6 Simulator crosswalk

Accel-Sim/GPGPU-Sim's documented Volta+ configuration models a private,
noncoherent L1 with write-allocate/write-through behavior and a shared,
write-allocate/write-back L2 with 32-byte sectors in 128-byte lines
([framework model](https://github.com/accel-sim/accel-sim-framework/blob/dev/gpu-simulator/gpgpu-sim4.md);
[cache discussion](https://github.com/gpgpu-sim/gpgpu-sim_distribution/wiki/test1)).
This is a valuable simulator abstraction, not a GA102 oracle: simulator
configuration defaults vary, and the local card's RM policy/control surface
does not reveal the hidden owner/read-from state. The safe adaptation is a
policy-branch model with explicit L1 noncoherence, L2 sector masks, forwarding
candidates, and unknown DRAM completion.

## 8. Updated predictions

The combined evidence makes the following predictions falsifiable without
promoting hidden fields:

1. **Route-envelope stability:** adding DocumentSASS semantic classes and
   Huerta control-bit constraints should tighten resource intervals and
   preserve current logical issue cycles; it should not produce a unique
   per-PC route.
2. **Unsupported generation monotonicity:** an F2Asm-style residual check
   should reject new SM86 forms until a target-specific corpus admits every
   field/control/relocation; fallback counts may increase while native
   replay outputs stay unchanged.
3. **Cache phase separation:** changing NCU cache intervention should move
   L2/DRAM sector signatures and NCU endpoints while leaving value semantics
   and simulator issue frontiers unchanged. The cold/warm campaigns already
   exhibit this pattern.
4. **Forwarding ambiguity:** discard/fence sequences should be reported as
   read-from candidate sets unless the consumer path is independently
   bypass-classified. A matching value after flush is insufficient to mark
   DRAM persisted.
5. **Sampling clock relation:** fixed CUPTI PM bins should remain near the
   configured interval under changing SM frequency, but bin boundaries will
   not align to NCU start/end or raw QMD report timestamps without an
   explicit launch marker.
6. **Timer-domain non-transitivity:** even if NCU duration and GPC cycles are
   internally coherent, that relation does not imply equality with CUPTI,
   raw QMD, CUDA events, or simulator completion.

These predictions are compatible with the existing promoted campaign results
and cannot regress them because the new factors are opt-in, label-free, and
unknown-preserving.


### 7.7 Timing-source correction and future visibility endpoints

CUPTI's GPU_TIME_INTERVAL selects a fixed-frequency trigger on supported GA10x
devices, but its PM sample Start/End timestamps are CPU-based after CUPTI
conversion; the documentation also warns that the first sample may be an
outlier. The local 479--513 ns widths therefore establish fixed-cadence PM
windows and resource occupancy, not a shared NV_PTIMER epoch with raw QMD.
The apparent UTC-scale alignment between retained PM CSVs and raw reports is
not promoted as a clock identity.

Two newer patent leads reinforce the typed-frontier model:

* US12417181B1 describes aperture-specific cache operations that select a
  subset of lines, write between cache levels, and invalidate a selected
  subset ([patent](https://patents.google.com/patent/US12417181B1/en)).
  This is consistent with the RM API's aperture distinctions but is not
  evidence that the GA102 user client exposes line-level completion.
* US20240403048A1 describes categorized barriers that wait for selected writes
  to become visible in host memory while unrelated operations may remain
  pending ([patent](https://patents.google.com/patent/US20240403048A1/en)).
  This is a useful design hypothesis: visibility, writeback service, and
  unrelated completion are category-scoped frontiers, not one DRAM-complete
  timestamp. It is not a current SM86 API contract.


### 7.8 DocumentSASS broad-kernel negative

A broader SM86 CUDA cubin did not unlock a full form catalog under CUDA 13.2:
the intercepted nvdisasm stream exposed only a handful of IMAD descriptor forms
alongside the repeated operation-set/resource block. The older DocumentSASS
extraction funnel expects a CUDA 11.6 marker. Thus operation-set metadata is
available as a semantic prior, but a current-toolchain extraction does not
become an unrestricted SM86 encoder simply by adding more kernels.

### 7.9 Endpoint equation

For endpoint j in {raw QMD, NCU, CUDA event, CUPTI PM, host}, use:

    D_j = T_launch_j + T_core + T_tail_j + T_cache_j + epsilon_j

where T_core is the simulator issue-to-last-instruction lower bound,
T_tail_j is endpoint-specific retirement/writeback/coherence inclusion,
T_launch_j is FE/prolog/stream setup, T_cache_j is cache/replay intervention,
and epsilon_j is bounded noise. The current evidence supports:

* NCU duration is internally coherent with local GPC cycles
  (D_N approximately GPC_cycles_N / f_GPC_local, residual below about 0.2%);
* raw QMD D_R is a documented NV_PTIMER nanosecond interval from the
  pre-dispatch report through post-WFI report;
* CUDA-event D_E is a stream-front timestamp;
* CUPTI PM D_P is a fixed-cadence sampling window with CPU-normalized
  timestamps;
* host D_H includes submission and polling.

Only a separately matched intervention may fit an offset or affine relation
between two D_j values. A scored NCU label may not choose T_core, T_tail, or
T_cache for that same row.


### 7.10 SOTA simulator boundary

Accel-Sim 2.0 (arXiv:2608.22602) reports strong cycle correlation on large
H100/B200 datasets, and the 2024 modern-NVIDIA analysis (arXiv:2401.10082)
separates GA102 sub-cores, front-end/result/memory stages
([Accel-Sim 2.0](https://arxiv.org/abs/2608.22602);
[modern analysis](https://arxiv.org/abs/2401.10082)). These are valuable
architecture-model and calibration references, not a unified timer or
per-instruction physical route trace. Earlier memory-system work
(arXiv:1810.07269) shows that cache/memory detail materially changes cycle
error ([memory-system study](https://arxiv.org/abs/1810.07269)), supporting
the explicit T_cache and T_tail terms rather than clock-only scaling.


## 9. Latest public negative/positive seams (2026-09-30)

### 9.1 Public issue tracker confirms the physical-coordinate gap

A current NVIDIA open-kernel issue explicitly asks for time-resolved
<physical SM ID, SMSP ID, timestamp, activity> samples. The report states
that CUPTI/PM sampling has a timeline but no supported globally identifiable
physical SM/SMSP coordinate for every sample, and that PTX exposes %smid and
%warpid but no %smspid
([NVIDIA issue #1351](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/1351)).
This is a new independent negative against treating PMA/PC sampling as an
exact pipe witness. It also separates an SMSP coordinate from a physical pipe:
even an SMSP ID would not identify the execution sub-unit or issue slot.

### 9.2 New SM86 operand-bank microbenchmark

A current RTX 3080/SM86 forum study keeps 32 independent FFMA instructions
and their compiler control words fixed while permuting even/odd source-register
bank order. Nsight source stall vectors change from roughly 1.5 to 2.0 to
2.5 cycles per instruction, with first dispatch-stall onset remaining near
instruction 7--8
([study and SASS listings](https://forums.developer.nvidia.com/t/how-does-the-operand-collector-gate-ffma-issue-on-ampere-sm-86/381274)).
The later discussion converges on a fixed-latency resource reservation/
operand-read model rather than a freely growing queue, but does not expose a
pipe or slot tag.

This is actionable for the simulator: the already-present SM86 register-bank
projection and opt-in operand-admission schemas can carry a fixed two-bank
read-reservation candidate. It must remain opt-in and mechanism-local; the
microbenchmark cannot justify a unique dynamic physical route. The current
default replay should not be altered solely from this external discussion.

### 9.3 Command-stream recovery is a launch/timing lead, not physical closure

A 2026 command-stream study recovers complete NVIDIA userspace pushbuffer/
GPFIFO submissions by observing the doorbell commit boundary and uses direct
command issuance to separate raw engine time from driver overhead
([arXiv:2604.26889](https://arxiv.org/abs/2604.26889)). The public
libcuda-research implementation documents that its SM-authored doorbell path
is Hopper-specific; pre-Hopper/Ampere USERD placement and doorbell behavior
differ
([libcuda-research](https://github.com/maxime-peim/libcuda-research)).
Therefore this work offers a useful future launch-envelope/timing capture
method, but it does not expose SM86 physical pipe, cache owner, or DRAM state.
The existing raw QMD path remains the strongest local Ampere command-level
endpoint.

### 9.4 Updated closure interpretation

The newest public seams make the approximation model more actionable:

* route modeling may add fixed register-bank/operand-admission pressure as an
  opt-in interval, while exact route identity stays unknown;
* PMA/NCU/CUPTI can carry timestamps, PC, stall state, and aggregate pipeline
  activity, but no supported dynamic physical SMSP/pipe/slot tuple;
* command-stream capture can separate driver submission from engine time on
  supported architectures, but it does not make NCU, CUPTI, raw PTIMER, and
  simulator completion one endpoint;
* no new public evidence closes cache ownership/read-from/DRAM persistence or
  unrestricted SM86 ABI/ISA generation.


## 10. Integrated local resource-envelope update

The two new target-card campaigns permit a strictly bounded model refinement:

    PathKey(warp) in {0,1,2,3}
    AdmissionPressure(i) = f(source_bank_reads_i,
                              destination_bank_writes_i,
                              reuse_bits_i,
                              path_occupancy,
                              fixed_resource_reservation)

PathKey is valid only under the observed one-CTA-per-SM/warp-layout envelope.
AdmissionPressure is an interval/resource demand, not an exact physical route.
The local FFMA orderings show that changing source-bank sequence changes
dispatch-stall samples and elapsed cycles while the static SASS control words
remain attached; the scheduler-contention campaign shows four parallel path
groups under controlled occupancy.

These results improve the causal partition of the model:

* register-bank/admission pressure belongs before execution-pipe completion;
* four scheduler-path occupancy belongs between warp selection and shared
  execution capacity;
* ALU/FMA/LSU sub-pipe assignment and issue slot remain latent;
* endpoint and cache factors remain independent.

Default native replay must retain its current output unless this resource
envelope is explicitly selected. Promoting the new constraints globally would
be a mechanism change without a family-disjoint accuracy result.


## 11. Bounded common-device-timer closure

A new crosswalk uses a semantically matched SM86 timer sequence whose body
reads %globaltimer, spins for 135,000 %clock64 ticks, reads %globaltimer again,
and publishes both values with STG.E.64.STRONG.SYS. The raw-QMD extracted
header is not byte-identical to the Driver cubin: control words differ at PCs
0x100 and 0x110 in the store/tail region. The operation sequence and launch
shape are matched semantically through:

1. the independent raw Ampere QMD V03/libmc path; and
2. the CUDA Driver API path.

At locked 1350/10501 MHz on the RTX 3090 Ti, raw QMD produced 32 post-setup
spans in 99,328--100,352 ns (median 100,352 ns), and the CUDA Driver path
produced 16 spans in the identical range (median 100,352 ns). The median
difference is zero. Artifacts:
results/alpha/sm86_common_timer_crosswalk_20260930/ and its
README/summary.json.

This closes a **bounded common device-globaltimer endpoint** for a fixed
semantic body, timer sequence, launch envelope, and raw/CUDA-Driver paths. It
is materially stronger than the earlier raw-report/NCU ratio, but it is not an
exact SASS-byte identity claim. It still does not prove
equivalence to Nsight Compute WaitForIdle duration, CUPTI activity timestamps,
CUDA events, host time, or simulator completion; those domains remain separate
until a same-body crosswalk includes them. It also does not close cache owner or
DRAM persistence semantics.


### 11.1 Same-body NCU/CUPTI/event extension

The common-timer body was additionally launched through a CUDA Driver process
that recorded CUDA-event and CUPTI activity spans, and was profiled by NCU in a
separate no-CUPTI process. Results for the same one-block body:

| endpoint | post-setup span |
|---|---:|
| in-body globaltimer | 100,032--100,064 ns; median 100,032 |
| CUPTI activity | 101,215--101,791 ns; median 101,231.5 |
| NCU gpu__time_duration | 101,600--101,632 ns; median 101,600 |
| CUDA event | 104,224--112,320 ns; median 104,448 |

The first CUDA-event launch was a setup outlier. These observations establish
path-specific, same-body envelopes rather than equality: CUPTI is about 1.2 us
wider than globaltimer, NCU about 1.6 us wider, and CUDA events about 4.4 us
wider after setup. The simulator completion/tail remains unobserved in this
crosswalk, so the full unified-endpoint blocker is narrowed but not closed.


## 12. Architectural cache-operator narrowing

The current PTX ISA 9.4 documentation provides a precise architectural
subcase: default global stores are write-back across coherent levels, cg stores
bypass L1 and cache at L2, and wt stores to a global **system-memory** address
write through the L2 to system memory. The same section explicitly warns that
different-SM L1 data can remain stale and that the driver must invalidate L1
between dependent grids
([PTX cache operators](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators)).
This narrows the cache model for a declared system-memory/wt intervention:

* architectural visibility/order may be marked by the PTX scope/fence contract;
* L1 stale/resident state remains a separate candidate;
* L2-to-system write-through is a path property, not a line-owner telemetry
  result;
* complete host/DRAM persistence still requires a completion/consumer witness.

The current RTX 3090 Ti campaigns use ordinary device-memory workloads, so this
subcase cannot be promoted as the native cache semantics for those rows.


### 11. Current-ABI parameter-family expansion

The current-ABI-v8 no-template emitter now has bounded launches for:

* two 64-bit pointers;
* three 64-bit pointers (24-byte PARAM_CBANK);
* four 64-bit pointers (32-byte PARAM_CBANK);
* pointer/u32/pointer, with independently re-encoded address instructions and
  ptxas-matched 64/32/64 KPARAM_INFO flags.

The observed 1,000-launch CUDA-event medians are 1.715040 ms, 1.722368 ms,
1.705952 ms, and 1.725152 ms respectively. These are capability observations,
not scored NCU labels or general compiler validation. They show that the
current ABI boundary is not limited to the original two-pointer template, but
they still leave arbitrary parameter types, relocations, control words,
resource metadata, and all SM86 forms outside the admitted capability set.


A follow-up mixed body actually consumes the 32-bit parameter as an IMAD
multiplier while using the third pointer for input. This prevents a false
capability claim based only on unused metadata: the current v8 pointer/u32/
pointer path is exercised semantically for the admitted body. The capability
boundary remains finite and operation-specific.


A temporary target-card system-memory intervention used inline st.global.wt.u32
to a cudaHostAllocMapped pointer followed by membar.sys. Sixteen launches
published distinct values visible to the host after stream synchronization;
post-setup in-body globaltimer spans were 1,024--2,048 ns and host
synchronization was about 6.1--7.4 us. This confirms a bounded
write-through/system-scope visibility path, but not device-memory ownership,
per-line DRAM acknowledgement, or ordinary st.wb/st.cg behavior.


### 13. Target-card cache-latency envelope

A Sattolo pointer-chase sweep on the locked RTX 3090 Ti measured approximately
39--49 cycles at the smallest working sets, 218--231 cycles across the
L2-like region, and 236--241 cycles at 64--512 MiB working sets. The 3-MiB
point was an isolated 246.89-cycle outlier and is retained as observed.

This supplies a local latency/reuse prior for the cache-phase model. It does
not identify cache owner, read-from source, partition, replacement state, or
DRAM completion. DGNA's A100/H100 NUMA conclusions are used only as
methodology, not transferred to GA102
([DGNA](https://arxiv.org/abs/2607.19922)).


### 14. Cross-SM cache-latency topology

A corrected cooperative 84-block experiment warms one 128-byte-spaced target
line from block 0, synchronizes all blocks, and measures that line with
ld.global.cg from every enabled SM. Across 32 target addresses, per-SM median
clock spans range **286--361 cycles** with stable groups. This is strong local
evidence for nonuniform L2/NoC/DRAM path latency, but it does not identify
which level supplied a read, a line owner, read-from source, compression state,
or DRAM completion. It is retained as a topology/interval prior only.


### 15. ptxas current-ABI shape matrix

A temporary CUDA 13.2 sm_86 cubin containing scalar, pointer, mixed, and
vector parameter kernels exposes current-v8 KPARAM_INFO shape rules:
reverse records, scalar 32-bit size flags, 8-byte pointers, 16-byte vectors,
natural alignment/padding, PARAM_CBANK base c0[0x160], and exit-offset lists.
This expands the metadata ledger for the hand emitter; it still does not
provide general relocation/linker/control synthesis or all-form SM86 coverage.

## Post-merge ABI field repair (2026-09-30)

A field-level comparison against an actual CUDA 13.x SM86 image isolated three
metadata defects in the hand-emitted ABI-v8 vector body: the `.text` section
symbol was absent from `.symtab`, `.nv.info`/`.nv.callgraph` did not link to the
symbol table, and `.nv.constant0` lacked `SHF_INFO_LINK`. The repaired image is
accepted by `cuobjdump --dump-elf` and `--dump-sass`, and five locked-clock
Driver API runs return `0,3,6,9` at a 1.767424-ms median per 1,000 launches.
The three-PHDR layout is retained because it is normal for the observed SM86
CUDA 13.x final images; header count is not a portable ABI invariant.

A subsequent minimal DWARF64 CIE/FDE variant with no debug relocation makes
`cuobjdump --dump-elf`, `cuobjdump --dump-sass`, and `nvdisasm` all complete for
the admitted image. The repair therefore upgrades the narrow ABI point from
“driver-accepted but cuobjdump-invalid” to “driver-accepted and tool-readable,”
while leaving full relocation/debug/linker generation and arbitrary SM86 forms
unresolved. This is a capability-ledger refinement, not unrestricted compiler
closure. See `sm86_abi_v8_toolreadable_20260930.md`.

## Structured-control ABI extension and PTE negative (2026-09-30)

The current ABI-v8 ledger now includes a branch image and a structured-loop
image. The latter re-encodes 56 rows covering BSSY/BSYNC, predicated BRA,
PLOP3, descriptor LDG/STG, and two exit offsets; it runs at a 2.033664-ms
median per 1,000 launches and returns `0,2,5,9,4,6,9,13`. Ptxas control words
remain retained, so independent scheduler/control generation remains outside
the ledger. A later field-layout audit corrected the SM86 control-code decode;
see `sm86_control_word_decode_correction_20260930.md`.

A separate public RM PTE-information probe returned
`NV_ERR_TEST_ONLY_CODE_NOT_ENABLED` for local, host-mapped, and system-uncached
allocations. The documented aperture/GPU-cached/PTE fields therefore do not
supply an owner/read-from route on production RM 615.71.09.

## Long-body clock-policy crosswalk (2026-09-30)

A 1,350,000-clock extension of the common timer body crosses raw QMD, Driver,
CUPTI, CUDA events, and NCU. With external clocks locked and `ncu
--clock-control none`, medians are 1,000,448/1,000,448/1,000,032/1,004,528/
1,001,600 ns. Default NCU boost control yields 715,824 ns on the same body.
This supplies an explicit clock-policy transform and a stronger drift-control
subcase; it does not identify simulator completion or justify mixing default
NCU boost results with locked-clock endpoints.
