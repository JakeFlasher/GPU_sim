# SM86 external-baseline contract audit

Updated 2026-09-27. This is a source audit and a bounded experiment design, not
an external-simulator accuracy result. It uses pinned source checkouts in
`/tmp/gpu-simulator-sota-20260927` and the retained local overlap records. No
large external workload was built or executed for this report.

## Decision in one paragraph

Accel-Sim/GPGPU-Sim is the only retained external path that is close enough to
attempt a **tiny common-input structural baseline**: its trace-driven frontend
consumes a legacy NVBit `.traceg` stream and its tree contains an SM86 profile.
Our enhanced protobuf can be lowered to that legacy subset, but only after a
strict join with the SASS/control sidecar and after refusing fields that the
legacy format cannot represent. The resulting one-CTA/one-warp experiment can
check parser/semantic identity and optionally one small cycle run; it cannot
be presented as an NCU or SOTA accuracy cell. MGPUSim's NVIDIA frontend reads
the same legacy `.traceg` grammar, but its public platform only exposes H100
and A100 and explicitly warns that timings are not validated, so it is a
parser/capability reference rather than an SM86 numerical baseline. FlashGPU-Sim
and Vortex are not same-input SM86 timing baselines: the former's maintained
workflow targets SM90/SM120 PTX/Triton and the latter is a RISC-V GPGPU stack.

## Pinned sources and revisions

| Source | Revision | What the source actually establishes |
| --- | --- | --- |
| [Accel-Sim framework](https://github.com/accel-sim/accel-sim-framework/tree/d930ad6d02c09bb56867132583735aba0389cff4) | `d930ad6` | NVBit SASS tracing, trace-driven GPGPU-Sim/Accel-Sim frontend, and a legacy SM86 profile. |
| [GPGPU-Sim distribution](https://github.com/gpgpu-sim/gpgpu-sim_distribution/tree/a4ce3fe) | `a4ce3fe` | The `SM86_RTX3070` configuration used by Accel-Sim. |
| [FlashGPU-Sim](https://github.com/FlashGPU-Sim/FlashGPU-Sim/tree/d20ce45) | `d20ce45` | Execution-driven/PTX/Triton workflow and maintained SM90/SM120 configurations. |
| [MGPUSim](https://github.com/sarchlab/mgpusim/tree/adf7c16) | `adf7c16` | Experimental NVIDIA H100/A100 trace reader/platform and legacy `.traceg` grammar. |
| [Vortex](https://github.com/vortexgpgpu/vortex/tree/ed4098d) | `ed4098d` | Full-stack RISC-V GPGPU ISA/toolchain/simulator, not NVIDIA SASS. |

Line references below are pinned to these revisions (or to the local checkout
when the contract is ours).

## Contract crosswalk: enhanced protobuf versus `.traceg`

### What the enhanced trace contains

The local wire contract has explicit active and predicate masks,
function/virtual PCs, optional SM/warp identity, typed memory class/access
kind/cache operator, and one or more typed address operands:

- `tools/nvbit_tracer/traces_enhanced/dynamic_trace/instruction.proto:L56-L156`
  defines `pc`, `active_mask`, `predicate_mask`, addresses, memory class, width,
  warp/SM identity, `vpc`, access kind, and cache operator.
- `tools/nvbit_tracer/traces_enhanced/dynamic_trace/address.proto:L24-L84`
  defines compressed addresses, allocation identity, effective address space,
  and an operand-specific lane mask.
- `tools/nvbit_tracer/traces_enhanced/dynamic_trace/kernel.proto:L7-L37`
  carries launch geometry and register/shared-memory metadata; some fields are
  optional and must be checked before lowering.
- `sim/trace/src/semantic.rs:L145-L225` computes the execution mask as
  `active_mask & predicate_mask`, reconstructs lane addresses, and preserves
  per-operand masks/spaces. A converter must not silently collapse a
  multi-operand instruction into one address stream.
- Static opcode/control evidence is intentionally outside the protobuf. The
  loader documents that embedded NVBit `op_code` and control bits are not
  trustworthy and that the sidecar's `opcode_text` is authoritative:
  `sim/trace/src/exec_info.rs:L20-L29`. The adapter therefore needs a
  content-bound `(function_unique_id, pc) -> opcode/control` sidecar join.

### What legacy Accel-Sim `.traceg` represents

The upstream tracer documents the line grammar as
`[line] PC mask dest_num [dest_regs] opcode src_num [src_regs] mem_width
[address-compression] [addresses]` at
[`util/tracer_nvbit/README.md:L159-L183`](https://github.com/accel-sim/accel-sim-framework/blob/d930ad6d02c09bb56867132583735aba0389cff4/util/tracer_nvbit/README.md#L159-L183).
The parser's in-memory record has one 32-bit mask/PC, at most one destination
and four sources, a single memory-width/address payload, and optional immediate
/register-value fields (`gpu-simulator/trace-parser/trace_parser.h:L48-L129`).
The producer writes the **effective** mask (`active_mask & predicate_mask`),
regular GPR operands, and compressed addresses in
`util/tracer_nvbit/tracer_tool/tracer_tool.cu:L1345-L1481`.
Consequently, the following projections are safe only under explicit guards:

| Dimension | Enhanced protobuf | Legacy `.traceg` | Baseline rule |
| --- | --- | --- | --- |
| ISA/opcode | No opcode text in the instruction message; static sidecar has full SASS text. | Opcode token/text is required and mapped through an architecture-specific opcode table. | Join every dynamic `(function_unique_id, vpc)` to sidecar text. Refuse missing/duplicate joins, unknown opcode heads, and sidecar/trace PC disagreement. Do not use NVBit's normalized `op_code`. |
| PC | `pc` plus optional 64-bit `vpc`; local SM86 PCs are function-relative. | `unsigned` PC parsed from hexadecimal text. | Require `vpc <= 0xffffffff` and preserve the function-relative value. A high-address VPC or a lossy truncation is a refusal, not a normalization. |
| Lane mask | Active and predicate masks are separate; operand may carry a narrower `lane_mask`. | One 32-lane mask only. | Emit `effective = active & predicate`. For the common subset require every operand's lane mask to equal `effective`; retain separate-mask cases only for a semantic comparator, not `.traceg` timing. |
| Register operands | No register list in protobuf; SASS sidecar text includes operands. | One destination and up to four source register names; legacy model primarily tracks regular GPRs. Its parser expects a numeric suffix (`RZ` is not directly parseable). | Parse sidecar SASS and require `<=1` regular destination, `<=4` regular sources, no unresolved UR/predicate/implicit dependency, and no special register-value instruction. Canonicalize SASS `RZ` to the legacy tracer's `R255` zero-register spelling only under a tested, documented rule; otherwise refuse. |
| Addresses | Per-operand compressed vectors, allocation IDs, spaces, widths, and masks. | One memory width and one list/base+stride/base+delta stream; no allocation/space identity. | Reconstruct and round-trip exactly for one ordinary global/shared/local operand. Refuse multi-operand `LDGSTS`, TMA, texture/surface, cross-allocation, or lane-mask cases that cannot be encoded. |
| Direction/cache semantics | Typed `MemoryOpAccessKind`, `MemoryOpClass`, and cache operator. | Inferred from opcode text/modifiers; no typed field. | Restrict the first baseline to ordinary `LDG`/`STG` (optionally `LDS`/`STS`) with a sidecar opcode accepted by both models. Do not compare typed atomic/async/TEX paths. |
| Warp/SM identity | Optional `warpid_tb`, `warpid_sm`, and `sm_id`. | CTA/warp grouping only; physical SM assignment is selected by the simulator. | Compare per-CTA/per-warp sequence, not physical placement. A sparse or missing `sm_id` cannot be “recovered” from `.traceg`. |
| Control bits | Local sidecar has stall/yield/barrier/wait and high-word evidence. | Trace-driven Accel-Sim schedules using its config and decoded opcode; it does not consume our sidecar control words. | Treat sidecar control as local evidence only. A cycle difference is a model difference, not a trace mismatch. |
| Launch metadata | Protobuf kernel carries grid/block and optional resource metadata. | Header carries grid/block, shared memory, register count, binary/tracer versions. | Require an explicit `nregs`, shared-memory size, and binary version. Do not default missing occupancy inputs to zero. |
| Termination | Threadblock protobuf is a finite dynamic list; no generic `EXIT` field. | Native `.traceg` commonly contains an `EXIT` record (mask zero), and trace-driven EOF also completes a warp after pipeline/store/scoreboard drain (`trace_driven.cc:L1034-L1058`). | Prefer an explicit sidecar/protobuf `EXIT` in the tiny fixture. If relying on EOF, prove both endpoints use the same completion rule; never silently append an `EXIT` to an arbitrary capture. |
| Metric endpoint | Local reports issue/event/completion/launch frontiers and keeps NCU labels separate. | Accel-Sim reports `gpu_tot_sim_cycle`/simulator counters. | No direct NCU comparison until endpoint and domain are formally aligned. `gpu_tot_sim_cycle` is not `gpc__cycles_elapsed.max`. |

The legacy frontend also has architecture-specific parsing and will assert on an
undefined opcode (`trace_driven.cc:L293-L313`). This is why “the text looks
similar” is not enough for arbitrary SASS.

## SM86 card and metric mismatch

The closest card is `SM86_RTX3070`, not the local RTX 3090 Ti card. Its pinned
configuration declares compute capability 8.6, 46 clusters/SMs, 16 memory
channels, 1132 MHz core/interconnect/L2 and 3500.5 MHz DRAM, 1536 threads per
SM, four schedulers, and Ampere cache/DRAM geometry
[`gpgpusim.config:L15-L118`](https://github.com/gpgpu-sim/gpgpu-sim_distribution/blob/a4ce3fe/configs/tested-cfgs/SM86_RTX3070/gpgpusim.config#L15-L118).
The local anchor is an RTX 3090 Ti (SM86, 84 SMs and a different clock/memory
card) recorded in `docs/baseline_matrix_20260924.json`; changing only cluster
count/clock/memory fields does not validate cache, scheduler, latency, or
address-mapping assumptions. The previously retained derived-profile replay
therefore remains `derived_profile_non_scoring`.

The current local overlap record is useful as **representation evidence only**:
`docs/accelsim_executable_overlap_probe_20260923.md:L29-L77` records 512 matching
warp sequences/197,120 records, identical effective PC/mask sequences, zero
normalized memory-payload mismatches, and 16,384 enhanced LDGSTS shared-
destination records absent from the legacy stream. It also records Accel-Sim
14,174 cycles and 5,621,216 simulator instructions, but explicitly refuses an
accuracy percentage because the card, trace contract, and endpoint differ.
Those numbers must remain non-scoring.

The metric domains are independently named: the local NCU target is
`gpc__cycles_elapsed.max` (see `docs/baseline_matrix_20260924.json`), whereas
FlashGPU-Sim's maintained validation compares `gpu_tot_sim_cycle` against
`sm__cycles_elapsed.avg` on RTX 5090 (`FlashGPU-Sim/README.md:L102-L119`). A
number copied between these columns would be a domain error, not a result.

### Bounded executable smoke (not a score)

To test whether the mismatch is a hard parser/build barrier rather than an
architecture excuse, a one-CTA/one-warp prefix was hand-lowered from the
retained enhanced canary: `vpc={0,16,32}`, full effective mask, and sidecar
 `IADD3 R0, R1, R2, RZ` at each PC. The legacy spelling `RZ -> R255` was required
(leaving `RZ` in the `.traceg` causes the upstream parser's numeric register
conversion at `gpu-simulator/trace-parser/trace_parser.cc:L190-L203` to throw);
the three records then terminated by trace EOF, not by a fabricated `EXIT`.
With the already-built Accel-Sim `d930ad6` executable (linked against the
retained GPGPU-Sim `91880c5` checkout) and the exact `SM86_RTX3070` card, the
bounded run completed in under one second
with exit status 0:

The temporary legacy trace and launch list were content-bound by SHA-256
`4e55adbb5a103bc0fbcba352de991d227bfbed8b237a04bb494560bbcd0d4075` and
`beb03d46bd1f5a6c20fdb34aaaeed37de3fc31e5010ff2c5657a7778e3e9a40c`,
respectively. They are reproducible from the three source records and the
command below; they are not retained as a scored workload artifact.

| Input | `gpu_tot_sim_cycle` | `gpu_tot_sim_insn` | CTAs | Interpretation |
| --- | ---: | ---: | ---: | --- |
| 3 enhanced-prefix records, one warp, EOF completion | 5,024 | 96 | 1 | Legacy frontend can execute this common prefix after explicit register-token normalization. It is not a full protobuf adapter, not an NCU endpoint, and not a 3090 Ti timing cell. |

This smoke closes the narrow feasibility question: a common-subset numeric
endpoint is executable without a large workload. It does **not** close the
contract: the converter still needs an automated sidecar join, address/mask
round-trip checks, explicit `nregs`/termination admission, and typed refusals
for the unsupported paths listed below.

The replay used the normal trace-driven invocation shape (all inputs were in a
temporary directory):

```text
accel-sim.out \
  -config <gpgpu-sim>/configs/tested-cfgs/SM86_RTX3070/gpgpusim.config \
  -trace <tmp>/kernelslist.g \
  -specialized_unit_1 1,4,4,4,4,BRA \
  -specialized_unit_2 1,4,200,4,4,TEX \
  -specialized_unit_3 1,4,32,4,4,TENSOR \
  -specialized_unit_4 1,4,4,4,4,UDP \
  -trace_opcode_latency_initiation_int 2,2 \
  -trace_opcode_latency_initiation_sp 2,1 \
  -trace_opcode_latency_initiation_dp 64,64 \
  -trace_opcode_latency_initiation_sfu 21,8 \
  -trace_opcode_latency_initiation_tensor 32,32
```

## Other retained simulators: precise mismatch decisions

### MGPUSim NVIDIA (closest parser, no SM86 timing cell)

MGPUSim's NVIDIA README says it replays Accel-Sim SASS traces through an SM/L1/
L2/DRAM model, but labels the path experimental and warns that timings are not
validated (`nvidia/README.md:L1-L14`). Its parser stores exactly the legacy
single-mask/register/address fields (`nvidia/trace/trace.go:L82-L100`) and the
workflow explicitly asks for version-5/6 `.traceg` output from Accel-Sim
(`nvidia/README.md:L90-L125`). That makes it a useful **round-trip parser test**
for an adapter-produced file.

It is not an SM86 numeric baseline: the public platform exposes only H100 and
A100 (`nvidia/platform/platform.go:L32-L81`), and the CLI accepts only those
names (`nvidia/README.md:L167-L196`). Unknown pipeline opcodes are mapped to a
similar opcode or one-cycle default (`nvidia/README.md:L195-L196`), which would
violate a fail-closed common-subset experiment. Run only parser tests unless a
new, independently validated SM86 platform is supplied.

### FlashGPU-Sim (modern feature reference, not an SM86 common input)

The maintained README lists Hopper/SM90 and Blackwell/SM120 configurations and
TMA/mbarrier/MMA/WGMMA features, not SM86 (`README.md:L23-L32`). Its validation
numbers are RTX 5090 `sm__cycles_elapsed.avg` versus `gpu_tot_sim_cycle`
(`README.md:L102-L125`), and its Triton workflow captures PTX/arguments and runs a
standalone harness (`README.md:L221-L279`). The checkout contains an inherited
`configs/tested-cfgs/SM86_RTX3070` directory, but the configuration guide calls
those legacy configurations unmaintained (`configs/README.md:L36-L40`); its
presence is not SM86 validation. There is no enhanced-protobuf input adapter in
this workflow. Mark FlashGPU-Sim as a feature/architecture reference only.

### Vortex (ISA/compiler reference, not NVIDIA timing)

Vortex is explicitly a full-stack **RISC-V** GPGPU with a configurable simulator,
RTL, FPGA targets, and OpenCL/OpenGL software (`README.md:L5-L7`); it supports
RISC-V RV32/RV64 and configurable GPU resources (`README.md:L33-L49`). Its VOLT
LLVM-based SIMT compiler is a useful comparison for the compiler/ISA gap
(`README.md:L132-L134`), but NVIDIA SM86 SASS, NVBit protobuf, cache topology, and
NCU domains are absent. No numeric cross-architecture timing comparison is
valid.

## Recommended smallest same-input experiment (E10A--E10C)

This experiment is deliberately smaller than a workload run and is intended to
close the *capability decision*, not manufacture a score.

### E10A — strict protobuf-to-`.traceg` adapter and round trip

The bounded probe is implemented outside the simulator core at
`tools/traceg_adapter_probe/adapter.py` (schema
`sm86_traceg_adapter_probe_v1`) with these admission conditions:

1. One exact-full SM86 launch, one CTA, one warp, and at most 64 dynamic records.
2. Every dynamic `(function_unique_id, vpc)` joins exactly once to the
   content-bound SM86 SASS sidecar; all PCs fit 32 bits; no unknown opcode.
3. Only ordinary `IADD3`/`FADD`/`LDG.E`/`STG.E` (or a similarly agreed pair) with
   regular GPR operands, one memory operand, and no barrier/texture/atomic/TMA/
   async-copy/LDGSTS operation. Canonicalize `RZ` to `R255` as an explicit
   legacy-token conversion, and test that conversion independently.
4. `lane_mask == active_mask & predicate_mask` for each memory operand and one
   address space per record. Reconstruct compressed vectors and then encode
   list/base+stride/base+delta; compare addresses after decompression.
5. Launch metadata includes grid/block, `nregs`, shared-memory size, binary
   version, and a documented termination record. If the source trace only ends
   at EOF (as the retained canary does), mark it **termination-incomplete**
   rather than inventing `EXIT`.

Emit a `.traceg`/`kernelslist.g`, parse it with an independent legacy reader
(Accel-Sim parser or MGPUSim's Go parser), and compare a canonical tuple per
record:

```
(block, warp, pc, effective_mask, opcode, dest_regs, src_regs,
 mem_width, decompressed_lane_addresses, immediate, terminal_kind)
```

Acceptance is byte-stable canonical equality and zero refused records after the
preconditions. This is a T0/T1 adapter result; it says nothing about timing
accuracy.

The synthetic common-subset fixture passed this round trip for three full-mask
`IADD3` records plus an explicit `EXIT`:

```text
schema: sm86_traceg_adapter_probe_v1
record_count: 3
termination: explicit_exit
traceg_sha256: 9c29c7b623e51c6b7e87b5c455883bd2eec628a55e66c836412b2ff16f7bc136
kernelslist_sha256: beb03d46bd1f5a6c20fdb34aaaeed37de3fc31e5010ff2c5657a7778e3e9a40c
```

Running the same probe against the first real SM86 holdout records refused at
record 0 with `unsupported_opcode` (`IMAD.MOV.U32`), as required by the strict
common-subset contract. This is a typed refusal, not silent opcode mapping.

### E10B — MGPUSim parser-only check

Feed the generated tiny `.traceg` to the MGPUSim trace reader/tests. Require
that the header, one block/warp, register names, compressed addresses, and
terminal record round-trip without unknown-opcode fallback. Do **not** select
H100/A100 timing or extrapolate it to SM86.

### E10C — bounded one-CTA Accel-Sim run

The already-built Accel-Sim executable was replayed on the three-record
canary-prefix smoke above. It returned `gpu_tot_sim_cycle=5024`,
`gpu_tot_sim_insn=96`, and `gpu_tot_issued_cta=1` with a 30-second timeout and
no stderr. Record these as a **T1 executable compatibility datum** only. A
future automated adapter should report wall time, peak RSS, and the same
counters separately. Compare only a predeclared common endpoint (for example,
final modeled event after trace EOF); do not divide by NCU cycles or rank
against the local simulator. If a future fixture needs an explicit `EXIT` and
cannot prove one, the correct result is
`capability_mismatch: termination-not-represented`.

### Required refusal matrix

The converter must refuse (with typed reasons) sampled/incomplete captures,
missing masks, missing sidecar opcode/control joins, high PCs, missing `nregs`,
UR/predicate/special-register dependencies, multi-operand memory, divergent
operand masks, excluded memory classes, absent termination evidence, and any
legacy parser unknown opcode. A refusal is evidence that the contract is
honest; it is not a failed accuracy run.

## Published claims versus local evidence

The papers are useful prior art, but their reported validation populations and
hardware are not silently transferable to this SM86 project:

| Prior work | Published scope | What the retained local source/run proves |
| --- | --- | --- |
| [Accel-Sim, ISCA'20](https://engineering.purdue.edu/tgrogers/papers/khairy.isca2020.pdf) | Extensible NVBit/SASS trace-driven validated GPU modeling on the workloads/cards reported by the paper. | The pinned tree parses legacy `.traceg`; the local overlap and 5,024-cycle one-warp smoke are representation/feasibility evidence, not a new accuracy result. |
| [MGPUSim, ISCA'19](https://people.bu.edu/joshi/files/mgpusim-isca2019.pdf) | Modular GPU simulation framework and the paper's reported validation scope. | The current NVIDIA extension is explicitly experimental and H100/A100-only; its parser is useful for E10B, not an SM86 score. |
| [FlashGPU-Sim](https://arxiv.org/abs/2609.15311) | Modern execution-driven/PTX/Triton features and the paper's reported newer-GPU studies. | The checkout's maintained cycle table is RTX 5090/H100-oriented; inherited SM86 files are legacy and no enhanced-protobuf adapter is present. |
| [Vortex, MICRO'21](https://doi.org/10.1145/3466752.3480128) | Open RISC-V GPGPU ISA/software/hardware stack. | It is a compiler/ISA extensibility reference; no NVIDIA SASS, SM86 cache model, or NCU-compatible endpoint is established. |

Thus “validated” in a paper or upstream README is a prior-work claim with a
declared scope, not independent evidence for this repository's RTX 3090 Ti
anchor. Only content-bound same-domain captures and a common endpoint could
upgrade E10 from feasibility to a numeric baseline.

## Publication/evidence boundary

| Claim | Supported now? | Evidence |
| --- | --- | --- |
| Enhanced and legacy captures can share a represented PC/effective-mask/global-address prefix. | Yes, limited. | Retained 512-warp/197,120-record comparator in `docs/accelsim_executable_overlap_probe_20260923.md:L53-L70`. |
| Our model is faster or more accurate than Accel-Sim on SM86/NCU. | No. | Different card, trace observables, and endpoints; no common numeric cell. |
| MGPUSim is an SM86 baseline. | No. | Public NVIDIA platform is H100/A100 only and timings are explicitly experimental. |
| FlashGPU-Sim's RTX5090/H100 results validate SM86. | No. | Different maintained architectures and NCU metric domain. |
| Vortex/OpenVADL demonstrate arbitrary NVIDIA ISA/compiler generation. | No. | Vortex is RISC-V; compiler/ISA extensibility is a reference direction, not an NVIDIA adapter. |
| A tiny, common-subset parser/trace experiment is feasible. | Yes, T1 parser/round-trip probe implemented; real holdout outside subset refuses. | One-CTA/one-warp contract above; no large simulation required. |

E10A is complete as a parser/round-trip capability result; E10B parser
integration and any common-endpoint timing comparison remain open. Keep `E10`
non-scoring and never replace it with a synthetic cycle score.
