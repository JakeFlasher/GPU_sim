---
Use when: explaining numerical-format evolution; placing FP16 / BF16 / TF32 / FP8 / FP6 / FP4 on a generation timeline; reasoning about the difference between a format and a datapath; understanding the "block-scaled MMA" vs plain low-bit MMA distinction.
Primary raw source: references/NVIDIA_patents/NVIDIA_patentd.md §3.3.
Supporting sources: per-arch cards (Pascal through Blackwell); references/NVIDIA_Manuals/ptx_isa/09_instruction-set.md (mma.sync, wgmma.mma_async, tcgen05.mma).
Patents anchored here: US10338919B2 (HMMA datapath, Volta) — every later format extends the dedicated path this patent established.
Out of scope: data-supply chain (`evolution-data-supply.md`); execution model (`evolution-execution-model.md`).
Version family: Pascal (FP16 entry, 2016) through Blackwell (FP4/MX, 2024). Mirror snapshot: 2026-05.
---

# Precision — from general-purpose floating-point to format-by-compute-semantics

The numerical-format line of NVIDIA architectures is not "more
formats per year." It's a **bifurcation**: at Volta, formats start to
fork according to **compute semantics** (training vs inference, dense
vs structured-sparse, payload vs scale). The general FPU and the
matrix datapath diverge from each other.

## The original problem

When computation was still mainly organized around the general
floating-point path, format change looked like "the same pipeline
supports more modes." That's not how today's Tensor Core precision
line works.

## Pascal — FP16 precursor (2016)

Pascal adds FP16 in the **standard FP32 path** — dual-FP16 fits in
one FP32 lane, so on GP100 peak FP16 is 2× FP32. But this is **not** a
Tensor Core: the dot-product still happens in the general FPU.

Pascal's FP16 is the **format**; Volta's HMMA is the **datapath**.
They land at different times.

## Volta — HMMA datapath turns format into a real fork (2017)

US10338919B2 makes the matrix path its own thing. Once that happens,
formats can fork according to dedicated path needs:

- Volta first-gen Tensor Core: **FP16 input + FP32 accumulate**
  (training-favored).

This is the **first stable Tensor Core precision**. Every later format
extends this dedicated path.

## Turing — INT8, INT4, Binary (2018)

Tensor Core 2nd gen adds:

- **INT8** (with INT32 accumulator): m8n8k16, m16n8k16, m16n8k32 shapes.
- **INT4**: m8n8k32, m16n8k32, m16n8k64.
- **Binary** (1-bit `xor.popc` / `and.popc`): m8n8k128, m16n8k128,
  m16n8k256.

Inference-favored. This is when the precision line forks by
**workload** — Volta = training, Turing = +inference.

SASS instructions: `IMMA` (integer MMA), `BMMA` (binary MMA).

## Ampere — BF16, TF32, FP64 Tensor Core, structured sparsity (2020)

Tensor Core 3rd gen adds:

- **BF16** (Brain Float, 8-bit exponent / 7-bit mantissa). Same range
  as FP32, half the bits. Drop-in for FP32 in many training settings.
- **TF32** (TensorFloat-32). 19-bit total: 8-bit exponent + 10-bit
  mantissa. **Range of FP32, precision of FP16.** Default Tensor Core
  type for FP32 inputs on A100+.
- **FP64 Tensor Core**: m8n8k4, m16n8k4, m16n8k8, m16n8k16. Datacenter
  GA100 first to expose this.
- **Structured 2:4 sparsity**: every 4 contiguous K elements have ≤2
  nonzeros; Tensor Core skips zeros at hardware speed → 2× peak.

TF32 is the trick worth remembering: NOT IEEE; range of FP32 with
FP16-class precision; transparent acceleration when user passed FP32.

## Hopper — FP8 + Transformer Engine (2022)

Tensor Core 4th gen adds:

- **FP8 E4M3** — 1 sign / 4 exponent / 3 mantissa. More precision,
  less range. Used for forward-pass activations and weights.
- **FP8 E5M2** — 1 sign / 5 exponent / 2 mantissa. More range, less
  precision. Used for gradients (which span larger dynamic range).
- **Transformer Engine**: hardware/software co-design that **dynamically
  selects** between FP8 and FP16 per layer to preserve training
  accuracy. The "numerical contract" surface that binds FP8 format,
  scale management, and matrix ops together.

FP8 is the first time the Tensor Core has formats that **need scale
factors** to be useful — payload alone can't represent a typical
training distribution.

PTX: `mma.sync` and `wgmma.mma_async` with `.e4m3` / `.e5m2` types.

## Blackwell — FP4, FP6, NVFP4, MX (2024)

Tensor Core 5th gen adds:

- **FP4 E2M1** — 1 sign / 2 exponent / 1 mantissa.
- **FP6 E3M2** — 1 sign / 3 exponent / 2 mantissa.
- **FP6 E2M3** — 1 sign / 2 exponent / 3 mantissa.
- **NVFP4** — NVIDIA's specific FP4 recipe combining:
  - Micro-block scaling
  - E4M3 scale factor (NOT FP4 — the scale factor itself is FP8)
  - Hadamard reshape
  - 2D block quantization
  - Stochastic rounding
- **MX (Microscaling) formats** — block-scaled formats per the OCP MX
  specification: groups of N values share one scale factor.

This is the first time the Tensor Core consumes **a format definition
plus a block-scaling description**, not just "an N-bit value."

PTX block-scaled MMA: `.kind::mxf4` / `.kind::mxf8f6f4` (sm_120a).

## The full timeline as a table

| Gen | Year | Tensor Core gen | New formats |
|---|---|---|---|
| Pascal | 2016 | — (FP16 in FPU) | FP16 (format only, not datapath) |
| Volta | 2017 | 1st | FP16 input + FP32 accumulate |
| Turing | 2018 | 2nd | INT8, INT4, Binary |
| Ampere | 2020 | 3rd | BF16, TF32, FP64 Tensor Core, 2:4 sparsity |
| Hopper | 2022 | 4th | FP8 (E4M3, E5M2) + Transformer Engine |
| Ada | 2022 | 4th (consumer) | FP8 (E4M3, E5M2) — without WGMMA |
| Blackwell | 2024 | 5th | FP4 (E2M1), FP6 (E3M2 / E2M3), NVFP4, MX |

## The real theme

It's not "more formats per year." It's that **formats start serving
different compute semantics and different throughput paths**.

```
Volta:      FP16 = training
Turing:     INT8/INT4/Binary = inference
Ampere:     BF16/TF32 = friendlier training; 2:4 = sparse compute
Hopper:     FP8 = mixed-precision training (with TE) + low-bit inference
Blackwell:  FP4/FP6/MX = block-scaled training and inference
```

Each new format is added **because** it lets a specific workload run
faster on the dedicated matrix path. The general FPU doesn't get
these — they live on the matrix datapath that Volta peeled out.

## Format ≠ datapath

A common error: "Pascal supports FP16 so it has Tensor Cores." No.

- **Pascal**: FP16 format, general FPU path, no Tensor Core.
- **Volta**: FP16 format **and** dedicated HMMA datapath.

Always distinguish. A format support claim alone tells you nothing
about Tensor Core presence.

## Compute capability floors per format

| Format | Min compute capability |
|---|---|
| FP16 (FPU) | 6.0 (Pascal) |
| FP16 Tensor Core | 7.0 (Volta) |
| INT8 / INT4 / Binary Tensor Core | 7.5 (Turing) |
| BF16 / TF32 / FP64 Tensor Core | 8.0 (A100) |
| 2:4 structured sparsity | 8.0 (A100) |
| FP8 (E4M3, E5M2) Tensor Core | 8.9 (Ada) / 9.0 (Hopper) |
| FP4 (E2M1), FP6 (E3M2, E2M3) Tensor Core | 10.0 (B200) / 12.0 (SM120) |
| Block-scaled MMA (`.kind::mxf4` / `.kind::mxf8f6f4`) | 12.0a (sm_120a, conditional) |

## Why a future agent cares

- "FP16 supported" without further qualification is ambiguous —
  "FP16 in FPU" (Pascal+) or "FP16 Tensor Core" (Volta+)?
- "FP8 inference" can mean Ada (sm_89) or Hopper (sm_90). They differ
  in WGMMA / TMA support, not in the Tensor Core itself.
- TF32 is the **default** Tensor Core dtype for FP32 inputs on
  A100+. Programs expecting bit-exact FP32 must explicitly opt out.
- NVFP4 is **not just FP4**. Without the scale factors, Hadamard
  reshape, etc., it's a different (worse) format.
- Block-scaled MMA shapes (`mxf4`, `mxf8f6f4`) are **architecture-
  conditional** (sm_120a). Not all Blackwell parts have them.

## See also

- `arch-pascal.md` — FP16 in the FPU.
- `arch-volta.md` — first dedicated Tensor Core path.
- `arch-turing.md` — INT8/4 / Binary inference fork.
- `arch-ampere.md` — BF16/TF32/structured sparsity.
- `arch-hopper.md` — FP8 + Transformer Engine.
- `arch-ada.md` — consumer FP8.
- `arch-blackwell.md` — FP4/FP6/NVFP4/MX.
- `evolution-data-supply.md` — formats that need scale factors are
  also a data-supply problem.
- `patent-family-map.md` — full patent index.
- `.claude/knowledge/ptx/ptx-isa-data-types.md` — current PTX
  type table.
- `.claude/knowledge/ptx/ptx-isa-tensor-cores.md` — `mma.sync`
  / `wgmma.mma_async` / `tcgen05.mma` shape × type tables.

## Escalate to raw when

- You need bit-exact NVFP4 specification — NVIDIA's NVFP4 blog and
  the Transformer Engine 2 documentation.
- You need exact OCP MX format spec — Open Compute Project MX
  specification.
- You need TF32 numerical-fidelity tradeoffs — NVIDIA's "TF32 in the
  A100" technical blog.
- You need exact Tensor Core throughput per precision per generation —
  `.claude/knowledge/blackwell/blackwell-tensor-perf.md` for
  Blackwell; NVIDIA's per-arch tuning guides for older.
