---
Use when: explaining `cp.async`; placing BF16 / TF32; reasoning about structured sparsity; understanding MIG (Multi-Instance GPU); placing Ampere as the "post-Volta machine matures and stabilizes" phase; routing GA100 vs GA102 differences.
Primary raw source: references/NVIDIA_patents/NVIDIA_patentd.md §2.3 (Ampere as the maturation phase after Volta).
Supporting sources: references/NVIDIA_patents/gpu-arch-wiki/user/4.ampere.md (GA100 SM card); references/NVIDIA_patents/gpu-arch-wiki/user/4.ga102.md (GA102 SM card); references/NVIDIA_patents/gpu-arch-wiki/user/NV微架构梳理.md (NV microarchitecture overview, Chinese) (memory hierarchy row).
Patents anchored here: no defining new architectural patent in the corpus; "shard scheduling" is mentioned as a patent-side inference but not a public main feature.
Out of scope: Hopper's cluster / TMA / WGMMA (`arch-hopper.md`); ITS / HMMA (Volta-era); RT Core 2nd gen (parallel to compute line, not in this corpus).
Version family: Ampere architecture, 2020. Representative products: GA100 (A100, 2020-05) datacenter; GA102 (RTX 3080/3090, 2020-09) consumer. Compute capability 8.0 (GA100), 8.6 (GA102), 8.7 (mobile).
---

# Ampere (2020, GA100 / GA102) — post-Volta machine stabilizes

Ampere is **not a breakpoint** — it is the maturation phase that takes
Volta's split datapaths and Turing's precision-fork and adds the
machinery to feed and stabilize them.

The headline additions:

1. **`cp.async`**: async copy from global to shared memory; first time
   data staging is peeled off the LSU as an explicit protocol.
2. **BF16 + TF32**: training-friendly precisions on the Tensor Core
   path, sparing programmers the FP16 dynamic-range problems.
3. **Structured 2:4 sparsity**: a 2× Tensor Core throughput boost on
   weights with mandated sparsity pattern.
4. **MIG (Multi-Instance GPU)**: a single A100 partitions into up to
   7 isolated GPUs at the hardware level.

There are two distinct Ampere SKU families: **GA100** (datacenter,
A100) and **GA102** (consumer, RTX 30-series). The cards differ
materially.

## SM organization — GA100 (A100)

| Component | Count |
|---|---|
| Warp Scheduler | 4 |
| Dispatch Unit | 4 |
| FP32 Core | 64 |
| INT32 Core | 64 |
| FP64 Core | 32 |
| **Tensor Core (3rd gen)** | **4** (per SM) |
| LD/ST Unit | 32 |
| SFU | 16 |
| Register file / SM | 64 KB |
| Shared memory + L1 (combined) | 192 KB / SM (max 164 KB usable) |
| L2 cache (whole chip) | **40 MB**, 2 partitions |
| Memory controllers | 12 × 512-bit (6 × HBM2 stacks) |
| SM count (A100) | 108 (out of 128 in full GA100) |

## SM organization — GA102 (RTX 30-series)

| Component | Count |
|---|---|
| Warp Scheduler | 4 |
| Dispatch Unit | 4 |
| FP32 Core | 64 |
| INT32/FP32 Core | **64** (mixed: can do either) |
| Tensor Core (3rd gen) | 4 |
| LD/ST Unit | 16 |
| SFU | 16 |
| Memory controllers | 12 × 32-bit GDDR6X |
| L2 cache (whole chip, GA102) | 6 MB (12 × 512 KB) |

**Key GA102 difference**: introduces the **mixed FP32/INT32 core** —
one of the two CUDA-core lanes can do either FP32 or INT32 each cycle.
This nominally doubles peak FP32 throughput vs A100, but the peak is
harder to hit (achieving it requires no INT32 in the critical path,
which is unusual for real code).

**A100 vs GA102** — totally different memory subsystems:
- A100: HBM2 with 1.555 TB/s bandwidth and 40 MB L2.
- GA102: GDDR6X and 6 MB L2.

GA102 (RTX 3080/3090) targets gaming + light compute; GA100 (A100)
targets data-center training. Don't conflate them.

## Tensor Core 3.0 (count drops, throughput per-core 4× up)

Per-SM Tensor Core count drops from **8 → 4**, but **per-Tensor-Core
throughput is 4× Volta**, so per-SM matrix throughput **doubles**.

Type expansion vs Volta/Turing:

| Type | Volta (1st) | Turing (2nd) | Ampere (3rd) |
|---|---|---|---|
| FP16 | ✓ | ✓ | ✓ |
| INT8 | — | ✓ | ✓ |
| INT4 | — | ✓ | ✓ |
| Binary | — | ✓ | ✓ |
| **BF16** | — | — | **✓** |
| **TF32** | — | — | **✓** |
| **FP64** | — | — | **✓ (Tensor Core)** |
| **2:4 sparsity** | — | — | **✓** |

### TF32 — the trick
TF32 is **NOT** an IEEE format. It uses the **range of FP32**
(8-bit exponent) with the **precision of FP16** (10-bit mantissa).
Idea: drop into TF32 inside the Tensor Core when the user passed
FP32 — get FP16-class throughput while preserving FP32 dynamic range.
For most ML workloads this is good enough for training, and the
programmer doesn't have to do anything.

### BF16 — same range as FP32, half the bits
Bfloat16: 8-bit exponent + 7-bit mantissa. Drop-in replacement for
FP32 in many training settings; less prone to underflow than FP16.

### Structured 2:4 sparsity
- For every 4 contiguous elements in the K dimension of a weight
  matrix, **at most 2 may be nonzero**.
- The Tensor Core skips the zeros at hardware speed → 2× peak
  throughput on sparse-pattern weights.
- Requires offline weight pruning + retraining to satisfy the 2:4
  pattern.

## `cp.async` — the data-supply protocol peels off

Ampere's biggest architectural addition is `cp.async`:

```ptx
cp.async.cg.shared.global  [smem_dst], [global_src], 16;
cp.async.commit_group;
cp.async.wait_group  N;
```

What changes vs the prior LSU-only path:

- **Async**: the load doesn't block the issuing thread. The thread
  continues; the LSU pipeline schedules the global→shared copy in
  the background.
- **Group-aware**: `commit_group` and `wait_group` track a batch of
  outstanding copies; one wait can release multiple copies.
- **No register stage**: data goes directly global→shared, never
  through a thread-private register.

This is the first peeling-off of staging from the LSU. Hopper
**generalizes** this to bulk multi-dimensional tile copies via TMA
(see `arch-hopper.md`); Blackwell **extends** it further to TMEM
(see `arch-blackwell.md`).

SASS-level: `LDGSTS` (asynchronous global-to-shared memcopy), with
`LDGDEPBAR` for fence/wait.

## MIG — Multi-Instance GPU

A100 (GA100) introduces hardware partitioning: split a single GPU
into up to **7 GPU instances**, each with isolated SMs, L2 slices,
and memory controllers. Each instance appears as a complete CUDA
device to its tenant.

Use cases:

- Cloud serving (one A100 backs multiple unrelated inference
  workloads, isolated for QoS).
- Multi-user research clusters.

GA102 (consumer) does **not** support MIG. MIG is data-center-only.

## "Shard scheduling" — patent-side inference, not a public feature

NVIDIA_patentd.md notes that "patent material describes a `shard
scheduling` idea: interleave forward progress by thread subset within
a warp post-divergence." This is consistent with ITS continuing to
descend in scheduling granularity. **It is not in the corpus's
verified-patent list, and it is not in NVIDIA's public Ampere
feature list.** Treat any claim about Ampere "shard scheduling" as
patent-side inference rather than a public main feature.

## Reference reading

- "NVIDIA cp.async patent analysis" —
  https://zhuanlan.zhihu.com/p/1991959111676471249

## Why a future agent cares

- `cp.async` is the floor for any modern double-buffered GEMM kernel
  (CUTLASS Hopper / Blackwell mainloops use it via TMA but the base
  primitive lands here). If a kernel needs `cp.async`, sm_80+.
- BF16 and TF32 are Ampere+. FP16 fallbacks for pre-Ampere matter.
- TF32 is the **default** Tensor Core dtype for FP32 inputs on A100+.
  Programs expecting bit-exact FP32 must explicitly disable it.
- Structured 2:4 sparsity is hardware-supported but requires the
  weights to be pre-pruned. It's not free; it's a hardware feature
  that needs corresponding software pipeline work.
- A100 has **40 MB L2** — much larger than Volta's 6 MB and Turing's
  5.5 MB. L2-resident working sets that work on A100 may not fit on
  consumer Ampere (GA102, 6 MB).
- MIG is GA100-only. Don't assume MIG semantics apply to RTX 30-series.

## See also

- `arch-volta.md` — Volta's HMMA datapath that Ampere matures.
- `arch-turing.md` — Turing's inference-side fork.
- `arch-hopper.md` — `cp.async` becomes `cp.async.bulk.tensor` (TMA).
- `evolution-precision.md` — BF16 / TF32 in the format timeline.
- `evolution-data-supply.md` — `cp.async` as the first staging peel.
- `evolution-execution-model.md` — shard scheduling as patent-side
  inference.
- `evolution-interconnect.md` — NVLink 3.0 + NVSwitch 1 land here.
- `patent-family-map.md` — full patent index (Ampere has no defining
  entry; this card explains why).

## Escalate to raw when

- You need exact A100 SM count vs full GA100 — A100 ships 108/128 SMs.
- You need GA102 vs GA104 vs GA106 deltas — gpu-arch-wiki has only
  GA102; consult NVIDIA's Ampere whitepapers (one for GA100, one for
  GA102).
- You need precise `cp.async` PTX semantics — see
  `.claude/knowledge/ptx/ptx-isa-async-copy.md`.
- You need MIG configuration / API surface — NVIDIA's MIG user guide
  is the canonical source; not in this corpus.
- You need TF32 numerical-fidelity tradeoffs vs FP32 — NVIDIA's
  "TF32 in the A100" technical blog is the canonical source.
