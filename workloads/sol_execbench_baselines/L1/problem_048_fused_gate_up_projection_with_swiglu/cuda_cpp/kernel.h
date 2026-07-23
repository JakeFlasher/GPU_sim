#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Host launcher for the fused per-element GELU-tanh + multiply
// portion of the Gemma3 gate + up projection chain. The two bf16
// matmuls are dispatched by main.cpp via at::matmul (cuBLAS), so
// the kernel here only handles the elementwise tail:
//
//   for each (b, s, j) in [B, S, intermediate_size]:
//     g_fp32   = (float) gate_output[b, s, j]
//     arg      = sqrt(2/pi) * (g_fp32 + 0.044715 * g_fp32^3)
//     act_fp32 = 0.5 * g_fp32 * (1 + tanh(arg))
//     act_bf16 = (bf16) act_fp32
//     output[b, s, j] = act_bf16 * up_output[b, s, j]   (bf16 product)
//
// Mirrors the PyTorch reference's exact dtype contract:
//   activated_gate = activated_gate.to(gate_output.dtype)   # to bf16
//   output         = activated_gate * up_output             # bf16 * bf16
//
// Tensors (all bf16):
//   gate_output : [B, S, intermediate_size = 24576]
//   up_output   : [B, S, intermediate_size = 24576]
//   output      : [B, S, intermediate_size = 24576] (DPS;
//                 pre-allocated by the eval driver)
//
// `intermediate_size = 24576` is pinned by the problem definition;
// `batch_size` and `seq_len` vary.
void problem_048_fused_gelu_tanh_mul_launcher(
    const torch::Tensor& gate_output,
    const torch::Tensor& up_output,
    torch::Tensor& output,
    cudaStream_t stream);
