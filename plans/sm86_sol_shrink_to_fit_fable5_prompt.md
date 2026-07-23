# Fable-5 design prompt #2 — SOL-ExecBench shrink-to-fit representative-shape protocol

> Copy everything below the line and send it to Fable 5. Self-contained. MEDIUM priority.

---

You are a GPU-workload and performance-measurement expert. Design a **per-cell "shrink-to-fit" representative-shape decision rubric** for capturing NCU metrics on an **RTX 3090 Ti (GA102, sm_86, 24 GB GDDR6X, 6 MiB L2)** from SOL-ExecBench kernels whose largest native workload shape OOMs on a 24 GB card. I need a concrete, per-kernel rubric — which dimension(s) to shrink and how far — not general advice.

## The problem

SOL-ExecBench ships fused AI kernels (multi-head attention forward/backward, MoE grouped-GEMM/expert-parallel, decoder-block full-block, layernorm/rmsnorm fused, etc.) each with a `workload.jsonl` of shapes. **23 in-scope kernels pass correctness but OOM at their LARGEST shape on 24 GB** (they fit on an 80 GB B200). Examples: an attention-backward that materializes a 5.10 GiB score-gradient; a decoder kernel that materializes a 16 GiB `[B, heads, S, S]` tensor. These are NOT arch defects — purely host-memory shape-gating.

Decision already made: **shrink each to a representative fitting shape and capture it as runnable** (stamped `shape_adjusted` with a native-shape reference) rather than deferring it. **What is OPEN is HOW to choose each reduced shape** so the captured NCU metric stays scientifically useful — i.e. in the same microarchitectural regime as the native shape.

## Why the shape choice is not trivial (the regime-preservation tension)

For these workloads the "shape" is a tuple of problem dims (batch B, sequence S, heads H, hidden/head-dim D, num-experts E, tokens, …), not a launch grid you can grid-stride freely. Shrinking different dims distorts different things:
- **Occupancy / tile structure:** shrinking a dim below one tile/wave changes occupancy and the scheduler regime.
- **L2 working-set regime (the sharpest hazard):** GA102's L2 is **6 MiB**. If the native shape's working set exceeds 6 MiB (L2-thrashing regime) but the shrunk one fits under 6 MiB, the L2 sector-hit rate silently INFLATES and the captured metric describes a regime the native shape never exhibits. Staying on the same side of the 6 MiB boundary as the native shape is critical (or, if impossible, must be flagged).
- **DRAM-pressure / roofline position:** shrinking can move the kernel from memory-bound to compute-bound (or vice versa), changing SOL% and the whole metric vector.
- **Attention/MoE structure:** the O(S²) attention score matrix, the expert routing/imbalance in MoE, and causal-vs-full masking each interact with which dim you cut.

A companion trace-sampling design (`plans/sm86_trace_sampling_strategy_fable5_design.md`) already distinguishes two reduced-shape variants — **`grid_stride_footprint_preserving`** (preserve total footprint/DRAM working set, change concurrency) vs **`dataset_shrunk`** (preserve per-block behavior, change working-set-to-L2 ratio) — and FORBIDS the dataset-shrunk variant when it drops the working set under the 6 MiB L2. Apply that lens here.

## What I need (concrete, per-cell)

1. **A decision rubric**: given a fused kernel's native shape tuple and its per-dim memory-footprint model, which dim(s) to shrink and by how much to (a) fit comfortably under ~22 GB (leave headroom under 24 GB), (b) preserve the compute:memory roofline regime, (c) keep the L2 working-set on the SAME side of the 6 MiB boundary as the native shape (or flag when impossible), (d) preserve occupancy / at least one full tile-wave, (e) preserve the kernel's algorithmic structure/correctness semantics (causal mask, expert count, etc.).
2. **Per-family guidance**: for attention-fwd/bwd (the O(S²) score/grad matrices), MoE grouped-GEMM/expert-parallel (routing imbalance), and decoder-block full-block — which dim is the safe one to cut (batch? a partial-sequence tile? heads? experts?) and which dims must be preserved to keep the regime.
3. **A detection rule for "no fitting shape preserves the regime"**: when every shape that fits 24 GB crosses the L2 boundary or flips the roofline regime → mark the cell `reduced_representative` (a different, still-valid measurement point) or typed `unavailable`, rather than pretending it predicts the native shape.
4. **Provenance to stamp**: native shape, chosen reduced shape, which dim(s) shrunk + factor, the memory-footprint model used, the L2-boundary verdict, the roofline-regime verdict, and the `shape_adjusted` vs `reduced_representative` classification.
5. **A cheap pre-check** to estimate a candidate shape's working-set vs 6 MiB and its roofline position BEFORE committing an NCU capture, so shape selection is a quick offline calculation per cell.

Deliver: the decision rubric, the per-family dim-shrink guidance, the regime-preservation detection rule, the provenance fields, and the cheap pre-check. Be explicit about which shrink choices are safe vs which silently corrupt the L2/roofline regime.
