---
Use when: decoding SASS mnemonics from `cuobjdump --dump-sass` or `nvdisasm` output; looking up a SASS opcode in NVBit-tracer output; cross-checking which architectures support a given instruction (HMMA, OMMA, QMMA, IMMA, DMMA, BMMA, WMMA, BSSY, BSYNC, UTCCP, etc.); inferring the minimum compute capability from a mnemonic; deciding whether the NVBit tracer needs a new opcode entry.
Primary raw source: references/NVIDIA_Manuals/cuda_binary_utility/04_instruction-set-reference.md §4.4 "Blackwell Instruction Set" (primary cloud target).
Supporting sources: §4.1 Turing (sm_75), §4.2 Ampere+Ada (sm_80/86/89), §4.3 Hopper (sm_90) of the same file; `tools/nvbit_tracer/blackwell_opcode.h` (repo tracer opcode table).
Repo touchpoints: tools/nvbit_tracer/isa_def/blackwell_opcode.h, tools/nvbit_tracer/, workloads/simbench_bw/.
Out of scope: per-mnemonic bit-layout / encoding (in the raw guide); PTX semantics; cuobjdump / nvdisasm invocation.
Version family: CUDA 13.x; covers SM70–SM120 SASS. Mirror snapshot date: 2026-04-13. Not authoritative; recheck raw source on CUDA upgrade.
---

# SASS Instruction Reference — Per-Arch Support Matrix

This card is a **routing index**, not a per-mnemonic mirror of the raw
catalog. When you hit an unknown SASS opcode in tracer output, use this
card to (a) confirm which architecture the mnemonic belongs to and (b)
land in the right section of `04_instruction-set-reference.md` for the
full operand grammar.

## Architectures the raw reference covers

| Arch | Compute cap | Raw doc section | Repo relevance |
|------|-------------|-----------------|----------------|
| Turing | 7.5 | §4.1 | Background and older papers |
| Ampere + Ada | 8.0 / 8.6 / 8.7 / 8.9 | §4.2 | Local sm_86 smoke host and older paper comparisons |
| Hopper | 9.0 | §4.3 | Cluster / TMA / `wgmma` debut; many 2024-era papers target this |
| Blackwell | 10.0 (B200) / 12.0 (consumer) | §4.4 | Primary cloud target: SM120 GB202 / RTX PRO 6000 Blackwell |

**Repo observation:** the cloud validation target is SM120 (Blackwell
workstation, RTX PRO 6000 Blackwell / GB202). Unless a trace came from
a pre-Blackwell microbench archive — including from the local RTX 3090 Ti
sm_86 smoke-test host on this branch, which traces sm_86 SASS — the SASS
the tracer sees should match §4.4. Local-host traces are sm_86 Ampere
SASS and won't match Blackwell §4.4 mnemonic tables.

### How the raw per-arch tables are organized

Each §4.x table groups mnemonics under headings: Floating-point,
Integer, Conversion, Movement, Predicate, Load/Store, Atomics, Cache
control, Uniform datapath, TMA (Hopper+), Tensor Core Memory (Blackwell),
Texture, Surface, Control, Miscellaneous. The grouping is the same
across arches — comparing a mnemonic's presence across §4.1/§4.2/§4.3/§4.4
tells you when it first shipped.

## Operand notation (Blackwell)

| Notation | Meaning |
|----------|---------|
| `Rx` | General-purpose register |
| `URx` | Uniform register (warp-uniform datapath) |
| `SRx` | Special / system register |
| `Px` | Predicate register |
| `UPx` | Uniform predicate register |
| `c[X][Y]` | Constant memory (bank X, offset Y) |
| `desc[URx][Ry]` | Memory descriptor |
| `gdesc[URx]` | Global memory descriptor |
| `tmem[URx]` | **Tensor memory (Blackwell only)** — distinct from `.shared` / `.global` |

## Mnemonic families (Blackwell §4.4)

Grouped by instruction family. Counts are approximate; use the raw guide for exact lists.

| Family | Representative mnemonics | Notes |
|--------|--------------------------|-------|
| Floating-point | `FADD`, `FADD2`, `FFMA`, `FHADD`, `FHFMA`, `FMNMX`, `FMUL`, `FSEL`, `MUFU`, `HADD2`, `HFMA2`, `HMMA`, `DFMA`, `DMMA`, `OMMA`, `QMMA` | **Blackwell-new: `OMMA` (FP4), `QMMA` (FP8)** matrix-multiply-accumulate. `FADD2`/`FFMA2` paired-lane variants also debuted on Blackwell. |
| Integer | `IADD`, `IADD3`, `IMAD`, `IMMA`, `IMNMX`, `LEA`, `LOP3`, `POPC`, `SHF`, `VABSDIFF4`, `VIMNMX3`, `VHMNMX`, `VIADDMNMX` | `VHMNMX`, `VIMNMX3`, `VIADDMNMX` first appear on **Hopper (§4.3)**, carry over to Blackwell. |
| Conversion | `F2F`, `F2I`, `I2F`, `I2I`, `I2IP`, `I2FP`, `F2IP`, `FRND` | `I2FP` / `F2IP` FP32↔int pack/unpack first appear on **Hopper §4.3** (lines 500-501) and carry over unchanged to Blackwell — **not** Blackwell-new. |
| Movement | `MOV`, `MOV32I`, `MOVM`, `PRMT`, `SEL`, `SGXT`, `SHFL` | `MOVM` transposes/expands matrices (tensor-core feeder). |
| Predicate | `PLOP3`, `PSETP`, `P2R`, `R2P` | |
| Load/Store (generic) | `LD`, `ST`, `LDC`, `LDG`/`STG`, `LDL`/`STL`, `LDS`/`STS`, `LDGDEPBAR`, `LDGMC` (reducing load) | `LDGDEPBAR` gates dependent loads without a full fence. |
| Async copy | `LDGSTS` (Ampere+, async global→shared), `LDSM`, `STSM`, `STAS` (async → DSMEM), `REDAS` (async reduce → DSMEM), `SYNCS` | `STAS`, `REDAS`, `SYNCS` first appear on **Hopper** (cluster / DSMEM async-store family), carry over unchanged to Blackwell. |
| Atomics | `ATOM`, `ATOMS`, `ATOMG`, `REDG`, `MATCH`, `QSPC` | |
| Cache control / memory ordering | `CCTL`, `CCTLL`, `CCTLT` (texture), `MEMBAR`, `ERRBAR`, `FENCE` | `FENCE` first appears on **Hopper**. Blackwell retains `MEMBAR` and `ERRBAR` alongside `FENCE` — the three coexist, they are not merged. |
| Uniform datapath | `UIADD3`, `UIMAD`, `ULDC`, `UFFMA`, `UREDUX`, `UMEMSETS`, plus `UCGABAR_*`, `USTGR`, `UREDGR` (global stores/reductions from the uniform path) | Blackwell expands the uniform-datapath surface heavily. |
| TMA (Tensor Memory Access) | `UBLKCP`, `UBLKPF`, `UBLKRED`, `UTMALDG`, `UTMASTG`, `UTMAREDG`, `UTMACCTL`, `UTMACMDFLUSH`, `UTMAPF` | Blackwell first-class TMA path; all live on the uniform datapath (`U*`). |
| Tensor Core Memory | `LDT`, `LDTM`, `STT`, `STTM`, `UTCCP`, `UTCHMMA`, `UTCIMMA`, `UTCOMMA`, `UTCQMMA`, `UTCBAR`, `UTCSHIFT`, `UTCATOMSWS` | **Blackwell-new: the tcgen05 tensor-core pathway** (TMEM loads/stores + MMA variants). |
| Texture | `TEX`, `TLD`, `TLD4`, `TMML`, `TXD`, `TXQ` | Keep as static feature context; do not infer NCU texture metrics. |
| Surface | `SUATOM`, `SULD`, `SURED`, `SUST` | Same status as texture. |
| Control flow | `BRA`, `BRX`, `BRXU`, `BSSY`, `BSYNC`, `BREAK`, `CALL`, `JMP`, `JMX`, `WARPSYNC`, `YIELD`, `KILL`, `EXIT`, `RET`, `NANOSLEEP`, `PREEXIT`, `ELECT`, `CGAERRBAR`, `ENDCOLLECTIVE`, `ACQBULK`, `ACQSHMINIT`, `RPCMOV` | `ACQBULK`/`ACQSHMINIT`/`CGAERRBAR` are cluster-era control-flow additions. `PREEXIT` hints dependent-task launch. |
| Misc | `BAR`, `B2R`, `CS2R`, `S2R`, `DEPBAR`, `LEPC`, `GETLMEMBASE`, `SETLMEMBASE`, `SETCTAID`, `PMTRIG`, `NOP`, `VOTE` | `SETCTAID` is the cluster-aware launch-time CTA index setter. |

## Blackwell-new opcodes that most often bite the NVBit tracer

**Repo observation:** the opcodes below are the ones that have previously shown up as "unknown opcode" in tracer runs and required additions to `tools/nvbit_tracer/blackwell_opcode.h`. Provenance is confirmed against the raw §4.4 Blackwell table + absence from §4.3 Hopper table:

| Opcode | Why it's Blackwell-new |
|--------|------------------------|
| `OMMA`, `QMMA` | FP4 / FP8 matrix-multiply-accumulate across a warp. Not in Hopper §4.3. |
| `LDT`, `LDTM`, `STT`, `STTM` | Tensor Memory (TMEM) register-file ↔ TMEM movement. Blackwell-only path. |
| `UTCCP`, `UTCHMMA`, `UTCIMMA`, `UTCOMMA`, `UTCQMMA`, `UTCBAR`, `UTCSHIFT`, `UTCATOMSWS` | Uniform-datapath tcgen05 path (TMEM async copy, MMA variants, TC barrier). Blackwell-only. |
| `FADD2`, `FFMA2`, `FMUL2`, `FHADD`, `FHFMA` | Paired-lane / half-rate FP variants. Not in Hopper §4.3. |
| `FMNMX3` | FP32 3-input min/max. Blackwell-new (Hopper §4.3 has `FMNMX` only). |

Opcodes that look Blackwell but are actually **Hopper-era** (already in §4.3 and so should already be in the tracer's Hopper opcode tables, but callers sometimes misclassify them):

- `STAS`, `REDAS`, `SYNCS` — cluster / DSMEM async store / reduce; first shipped on Hopper.
- `FENCE` — first shipped on Hopper.
- `VHMNMX`, `VIMNMX3`, `VIADDMNMX` — SIMD min/max/add; first shipped on Hopper.
- `HMNMX2` — FP16 min/max; first shipped on **Ampere** (raw §4.2), unchanged through Hopper and Blackwell.
- `I2FP`, `F2IP` — FP32 ↔ integer convert-and-pack primitives; first shipped on **Hopper** (raw §4.3 lines 500-501), present in Blackwell §4.4 (lines 749-750). The uniform-datapath siblings `UI2FP` / `UF2IP` are Blackwell-new.
- `DMMA` — double-precision MMA; debuted on **Ampere** (§4.2), present through Hopper and Blackwell.

Pair this card with `sass-tooling.md` (the diagnosis workflow).

## Generation-to-generation deltas worth knowing

The §4.x tables show *per-arch*, not deltas. A few patterns hold:

- **Turing → Ampere**: adds `IMMA` variants, `LDGSTS` (async copy), wider `LDSM` shapes.
- **Ampere → Hopper**: adds cluster barrier ops (`ACQBULK`, `CGAERRBAR`), `ELECT`, `STSM`, first-generation `UTMA*` TMA family, async-bulk shared↔global, `STAS`/`REDAS`/`SYNCS`, `FENCE`, the `V*MNMX*` SIMD min/max family.
- **Hopper → Blackwell**: tcgen05 TMEM family (`LDT`/`STT`/`UTC*MMA`), FP4/FP8 MMA (`OMMA`/`QMMA`), paired-lane FP (`FADD2`/`FFMA2`/`FMUL2`), FP32 3-input min-max (`FMNMX3`), uniform-datapath convert-pack (`UI2FP`/`UF2IP`). Note: `I2FP`/`F2IP` themselves debuted on Hopper, not Blackwell.
- **Long-standing**: `BMOV`, `SETCTAID`, `CCTL`, `ATOM`, `MEMBAR`, `ERRBAR`, `BAR`, `SHFL` exist on every arch from Turing through Blackwell.

## Fast grep workflows

```bash
# "When did mnemonic X first ship?" Walk the raw §4.x tables in order.
grep -n '^| OMMA ' references/NVIDIA_Manuals/cuda_binary_utility/04_instruction-set-reference.md
# (line > 649 → Blackwell-only; lines 408-648 → Hopper; etc.)

# "Show me every Blackwell tensor-core memory instruction."
sed -n '/^## \[4\.4\. Blackwell/,/^## /p' \
    references/NVIDIA_Manuals/cuda_binary_utility/04_instruction-set-reference.md \
  | awk '/Tensor Core Memory Instructions/,/Texture Instructions/'

# "Does this opcode exist on Hopper?"
sed -n '/^## \[4\.3\. Hopper/,/^## \[4\.4\. Blackwell/p' \
    references/NVIDIA_Manuals/cuda_binary_utility/04_instruction-set-reference.md \
  | grep -c '^| OMMA '   # expect 0
```

**Repo observation:** these one-liners are the most honest way to figure out "is this opcode Blackwell-new?" — the per-arch tables are source of truth; any card is a stale snapshot by comparison.

## Modeling context (papers)
- [Dissecting and Modeling the Architecture of Modern GPU Cores](references/papers/knowledge_base/active/2025__dissecting_and_modeling_the_architecture_of_modern_gpu_cores.pdf) — maps SASS opcodes to pipeline units and measures per-instruction latency/throughput on modern NVIDIA cores.
- [Analyzing and Improving Hardware Modeling of Accel-Sim](references/papers/knowledge_base/active/2024__analyzing_and_improving_hardware_modeling_of_accel_sim.pdf) — prior art showing why SASS opcode classification affects model quality.

## See also

- `.claude/knowledge/sass/sass-tooling.md` — how to disassemble and inspect what the compiler emitted (`cuobjdump`, `nvdisasm`, demangle, prune).
- `.claude/knowledge/ptx/ptx-isa-sm100-blackwell.md` — PTX-level counterpart to Blackwell SASS deltas (`tcgen05.*`, `cp.async.bulk*`).
- `.claude/knowledge/ptx/ptx-isa-tensor-cores.md` — mma/wgmma PTX.
- `.claude/knowledge/ptx/ptx-isa-async-copy.md` — `cp.async.*` PTX family that lowers to `LDGSTS` / `UBLKCP` / `UTMA*`.
- `tools/nvbit_tracer/blackwell_opcode.h` — repo opcode table that consumes these mnemonics.
- `.claude/knowledge/INDEX.md` — full agent KB router.

## Escalate to raw when

- You need the exact operand grammar / per-mnemonic field ordering for one opcode — the raw §4.x table has one description line per mnemonic; for full encoding detail you need `nvdisasm -hex` on an actual cubin.
- You need the Turing / Ampere / Hopper mnemonic list (anything pre-Blackwell) — §4.1–§4.3 in the raw file.
- A mnemonic appears in tracer output that is **not** in the Blackwell §4.4 table — it may be an undocumented vendor-internal opcode; consult `nvdisasm -hex` encoding + `tools/nvbit_tracer/blackwell_opcode.h` to see whether it was added by an NVBit patch rather than the public ISA.
- You need per-arch SM capability flags (e.g. "which sm_XX first shipped `LDGSTS`?") — that's distributed across §4.x intro paragraphs and PTX ISA §13 (release notes).
