#include "kernel.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Grid: dim3(updated_seq_len, num_kv_heads, batch_size). One CTA per
// (batch, head, output_seq_position). One thread per head_dim element.
//
// For output positions s < current_seq_len, the CTA copies from
// key_cache / value_cache. For positions s >= current_seq_len, the CTA
// pulls the new state at index (s - current_seq_len): raw copy for V;
// RoPE rotation for K, computed in fp32 to match the reference's
// pytorch bf16-mul + bf16-add semantics (both stages round-trip
// through fp32).
//
// `rotate_half(k)` per the reference is: split head_dim in half along
// the last axis, negate the second half, swap. For tid < half_D, the
// rotated element is -k[tid + half_D]; for tid >= half_D, it is
// +k[tid - half_D]. The kernel reads BOTH the local tid and the
// paired index per thread (one extra global load), which keeps the
// data flow purely sequential per thread and avoids inter-thread
// shuffles for a one-pass write.

namespace {

__global__ void problem_071_kv_cache_update_with_rope_kernel(
    const __nv_bfloat16* __restrict__ key_states,
    const __nv_bfloat16* __restrict__ value_states,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    const __nv_bfloat16* __restrict__ key_cache,
    const __nv_bfloat16* __restrict__ value_cache,
    __nv_bfloat16* __restrict__ updated_key_cache,
    __nv_bfloat16* __restrict__ updated_value_cache,
    int B, int H, int current_seq_len, int new_seq_len, int D, int half_D) {

    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int s = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= D) return;
    if (b >= B || h >= H) return;

    const long updated_seq_len = current_seq_len + new_seq_len;
    if (s >= updated_seq_len) return;

    const long dst_idx =
        ((static_cast<long>(b) * H + h) * updated_seq_len + s) * D + tid;

    if (s < current_seq_len) {
        const long src_idx =
            ((static_cast<long>(b) * H + h) * current_seq_len + s) * D + tid;
        updated_key_cache[dst_idx] = key_cache[src_idx];
        updated_value_cache[dst_idx] = value_cache[src_idx];
        return;
    }

    const int ns = s - current_seq_len;
    const long new_src_idx =
        ((static_cast<long>(b) * H + h) * new_seq_len + ns) * D + tid;
    const long cos_sin_idx =
        (static_cast<long>(b) * new_seq_len + ns) * D + tid;

    updated_value_cache[dst_idx] = value_states[new_src_idx];

    long rotate_pair_idx;
    float rotate_sign;
    if (tid < half_D) {
        rotate_pair_idx =
            ((static_cast<long>(b) * H + h) * new_seq_len + ns) * D
            + (tid + half_D);
        rotate_sign = -1.0f;
    } else {
        rotate_pair_idx =
            ((static_cast<long>(b) * H + h) * new_seq_len + ns) * D
            + (tid - half_D);
        rotate_sign = 1.0f;
    }

    // PyTorch's bf16 multiply lowers to: cast operands to fp32, multiply,
    // cast result back to bf16. The bf16 add then upcasts again, adds in
    // fp32, casts to bf16. To bit-match the reference's
    // `(key_states * cos) + (rotate_half * sin)`, each multiply result
    // is rounded to bf16 before the add (otherwise the fp32-accumulated
    // path is more accurate than the reference and elements drift outside
    // the rel-err tolerance on small workloads where the matched-ratio
    // tolerance has little slack).
    const float k = __bfloat162float(key_states[new_src_idx]);
    const float c = __bfloat162float(cos[cos_sin_idx]);
    const float si = __bfloat162float(sin[cos_sin_idx]);
    const float rotate_val =
        rotate_sign * __bfloat162float(key_states[rotate_pair_idx]);

    const __nv_bfloat16 term1_bf = __float2bfloat16(k * c);
    const __nv_bfloat16 term2_bf = __float2bfloat16(rotate_val * si);
    const float result =
        __bfloat162float(term1_bf) + __bfloat162float(term2_bf);
    updated_key_cache[dst_idx] = __float2bfloat16(result);
}

}  // namespace

void problem_071_kv_cache_update_with_rope_launcher(
    const torch::Tensor& key_states,
    const torch::Tensor& value_states,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    torch::Tensor& updated_key_cache,
    torch::Tensor& updated_value_cache,
    cudaStream_t stream) {

    const int B = static_cast<int>(key_states.size(0));
    const int H = static_cast<int>(key_states.size(1));
    const int new_seq_len = static_cast<int>(key_states.size(2));
    const int D = static_cast<int>(key_states.size(3));
    const int current_seq_len = static_cast<int>(key_cache.size(2));
    const int half_D = D / 2;
    const int updated_seq_len = current_seq_len + new_seq_len;

    if (updated_seq_len == 0) {
        return;
    }

    dim3 grid(updated_seq_len, H, B);
    dim3 block(D);

    problem_071_kv_cache_update_with_rope_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            key_states.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            value_states.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(sin.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            key_cache.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            value_cache.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(
            updated_key_cache.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(
            updated_value_cache.data_ptr<at::BFloat16>()),
        B, H, current_seq_len, new_seq_len, D, half_D);
}
