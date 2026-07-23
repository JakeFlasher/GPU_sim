// slug: chunk_gated_delta_rule_bf16
// tier: P
// arch_gate: sm_90a sm_120
//
// Chunked gated delta-rule recurrence (FlashInfer-Bench GDN variant).
// Matches references/ssm.py::_chunk_gated_delta_rule.
//
// Per timestep, per channel c:
//   err[c]       = x[t, c] - sum_s state[s, c] * key[t, s]
//   state[s, c]  = a[t, s] * state[s, c] + b[t, s] * err[c] * key[t, s]
//   g[c]         = sigmoid(gate[t, c])
//   out[t, c]    = g[c] * mean_s(state[s, c]) + (1 - g[c]) * x[t, c]
//
// Distinct from row 57 (mamba2_discretize_chunk): row 57 evolves a
// 1D state vector via (Abar, Bbar) discretization and projects through
// C; this row evolves a 2D state[s, c] matrix via a key-driven rank-1
// delta-rule update and gates the output with a per-position sigmoid.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

extern "C" __global__ void chunk_gated_delta_rule_bf16_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ a,
    const __nv_bfloat16 *__restrict__ b,
    const __nv_bfloat16 *__restrict__ key,
    const __nv_bfloat16 *__restrict__ gate,
    __nv_bfloat16 *__restrict__ out,
    int32_t seq, int32_t state_dim, int32_t channels)
{
    const int ba = blockIdx.x;
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;
    extern __shared__ float smem[];
    // state[s][c] flat layout: state[s * channels + c].
    float *state = smem;
    const int N = state_dim * channels;

    for (int i = tid; i < N; i += lanes) state[i] = 0.0f;
    __syncthreads();

    for (int t = 0; t < seq; ++t) {
        const size_t base_state = ((size_t)ba * seq + t) * state_dim;
        const size_t base_x     = ((size_t)ba * seq + t) * channels;
        // Per-channel state update.
        for (int c = tid; c < channels; c += lanes) {
            float x_c = __bfloat162float(x[base_x + c]);
            // err = x_c - sum_s state[s, c] * key[t, s]
            float err = x_c;
            for (int s = 0; s < state_dim; ++s) {
                float k_s = __bfloat162float(key[base_state + s]);
                err -= state[s * channels + c] * k_s;
            }
            // state[s, c] = a[t, s] * state[s, c] + b[t, s] * err * key[t, s]
            for (int s = 0; s < state_dim; ++s) {
                float a_s = __bfloat162float(a  [base_state + s]);
                float b_s = __bfloat162float(b  [base_state + s]);
                float k_s = __bfloat162float(key[base_state + s]);
                state[s * channels + c] = a_s * state[s * channels + c] + b_s * err * k_s;
            }
            // Gated output.
            float m = 0.0f;
            for (int s = 0; s < state_dim; ++s) m += state[s * channels + c];
            m /= (float)state_dim;
            float gate_c = __bfloat162float(gate[base_x + c]);
            float g = 1.0f / (1.0f + __expf(-gate_c));
            float outv = g * m + (1.0f - g) * x_c;
            out[base_x + c] = __float2bfloat16(outv);
        }
        __syncthreads();
    }
}

extern "C" void chunk_gated_delta_rule_bf16_launch(
    const __nv_bfloat16 *d_x, const __nv_bfloat16 *d_a, const __nv_bfloat16 *d_b,
    const __nv_bfloat16 *d_key, const __nv_bfloat16 *d_gate,
    __nv_bfloat16 *d_out,
    int B, int seq, int state_dim, int channels,
    cudaStream_t stream)
{
    int threads = 64;
    if (channels > threads) threads = (channels < 256) ? channels : 256;
    dim3 grid(B);
    dim3 block(threads);
    size_t shmem = (size_t)state_dim * channels * sizeof(float);
    chunk_gated_delta_rule_bf16_kernel<<<grid, block, shmem, stream>>>(
        d_x, d_a, d_b, d_key, d_gate, d_out, seq, state_dim, channels);
}
