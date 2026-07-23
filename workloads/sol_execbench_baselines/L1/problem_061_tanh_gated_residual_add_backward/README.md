# problem_061_tanh_gated_residual_add_backward

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | BF16 backward pass for tanh-gated residual addition. Computes `grad_residual` (identity), `grad_hidden_states` (per-element), and a scalar `grad_gate` via whole-tensor reduction scaled by `sech²(gate)`. Three DPS outputs with mixed shapes (two `[B, S, H]` plus a zero-dim scalar). |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_061_tanh_gated_residual_add_backward.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/061_tanh_gated_residual_add_backward/`                      |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

CUDA C++ baseline that implements the backward pass for
`output = residual + tanh(gate) * hidden_states * mask` as
two kernels driven by one DPS entry plus an in-host
identity copy.

Algorithm:
- `tanh_g = tanh(gate)` and `sech²(gate) = 1 - tanh²(gate)`
  evaluated ONCE on the host (gate is a kernel-invocation
  constant scalar).
- `grad_residual = grad_output` (identity), implemented as
  `grad_residual.copy_(grad_output)` in `main.cpp` —
  matches the reference's `grad_output.clone()` semantics
  while satisfying the DPS contract.
- `grad_hidden_states[b, s, h] = bf16(grad_output[b, s, h].f32 * tanh_g * mask[b, s, 0].f32)` (per-element transform).
- `grad_gate = bf16(sech²(gate) * Σ_{b,s,h}(grad_output * hidden_states * mask).f32)` (whole-tensor reduction).

`hidden_size = 4096` is pinned; `batch_size` and `seq_len`
are workload-variable (`B*S ∈ [131, 8192]` across 16
workloads).

Two-kernel implementation:

### Kernel A — `problem_061_grad_hidden_kernel`

Per-(b, s) row CTA. Grid: `(B*S,)`. `BLOCK_THREADS = 256`
(8 warps). Each thread maintains a fp32 `partial_sum`
accumulator and stride-loops over the row's
`hidden_size / 2 = 2048` BF16x2 pairs via `__ldg`:

- Reads `mask_val = bf16_to_f32(mask[b, s, 0])` once
  (broadcast across all threads).
- For each pair `i`:
  - Read `grad_output[b, s, ...]` and `hidden_states[b, s, ...]` as BF16x2; cast to fp32.
  - Write `grad_hidden_states[b, s, ...] = bf16x2(g * tanh_g * mask_val, g * tanh_g * mask_val)`.
  - `partial_sum += g * h * mask_val` over the two fp32 components.
- Warp-shuffle reduces `partial_sum` within each warp; the
  8 warp leaders cross-warp-reduce via shared memory;
  thread 0 writes `row_partials[b*S + s] = total_sum`.

### Kernel B — `problem_061_grad_gate_reduce_kernel`

Single-CTA reduction of the `row_partials[B*S]` fp32
buffer (max 8192 elements per the workload sweep).
`BLOCK_THREADS = 256`. Each thread stride-loops over the
buffer accumulating in a fp32 register; warp-shuffle +
8-way cross-warp tree reduces; thread 0 multiplies by
`sech²(gate)`, casts to BF16, writes to the zero-dim
`grad_gate` output tensor.

The `row_partials[B*S]` fp32 buffer is allocated in
`main.cpp` via `at::empty` and discarded after the
launches.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was
compile-checked and natively-correctness-validated on.
Final SOL scoring is on B200 (sm_100a). Differences worth
noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5060):
  Kernel A is bandwidth-bound. The two BF16x2 streams
  (`grad_output`, `hidden_states`) plus the BF16x2 write
  (`grad_hidden_states`) dominate runtime. B200's HBM3e
  (~8 TB/s) doubles the achievable throughput.
- DSMEM and cluster-cooperative reduction (sm_100a-only)
  are out of scope; both kernels use only CTA-local
  shared memory.
- L2 carveout / partition: Kernel A's per-row tile is
  `hidden_size * 2 B = 8 KiB` per stream, fitting in L1
  trivially. Kernel B's input is the `row_partials[B*S] *
  4 B = up to 32 KiB` buffer, also L1-resident.
- TCGen05 / TMEM divergence: 5th-gen tensor cores and TMEM
  register surface only exist on sm_100a; this kernel
  uses no tensor cores.
