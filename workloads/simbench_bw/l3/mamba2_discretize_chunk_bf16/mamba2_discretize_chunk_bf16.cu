// slug: mamba2_discretize_chunk_bf16
// tier: P
// arch_gate: sm_90a sm_120
//
// Mamba-2 discretized state-space chunk recurrence.
// Matches references/ssm.py::_mamba2_discretize_chunk.
//
// Per timestep:
//   Abar[t, s] = exp(dt[t, s] * a[t, s])
//   Bbar[t, s] = (Abar[t, s] - 1) / a[t, s] * b[t, s]
//   state[s]   = Abar[t, s] * state[s] + Bbar[t, s] * x[t, s % channels]
// Output:
//   out[t, c]  = sum_s c_proj[s, c] * state[s]            (real C-projection)

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

extern "C" __global__ void mamba2_discretize_chunk_bf16_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ a,
    const __nv_bfloat16 *__restrict__ b,
    const __nv_bfloat16 *__restrict__ dt,
    const __nv_bfloat16 *__restrict__ c_proj,
    __nv_bfloat16 *__restrict__ out,
    int32_t seq, int32_t state_dim, int32_t channels)
{
    const int ba = blockIdx.x;
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;
    extern __shared__ float smem[];
    float *state = smem;

    for (int s = tid; s < state_dim; s += lanes) state[s] = 0.0f;
    __syncthreads();

    for (int t = 0; t < seq; ++t) {
        const size_t base_state = ((size_t)ba * seq + t) * state_dim;
        const size_t base_x     = ((size_t)ba * seq + t) * channels;
        for (int s = tid; s < state_dim; s += lanes) {
            float a_v  = __bfloat162float(a [base_state + s]);
            float b_v  = __bfloat162float(b [base_state + s]);
            float dt_v = __bfloat162float(dt[base_state + s]);
            float x_v  = __bfloat162float(x [base_x + (s % channels)]);
            float Abar = __expf(dt_v * a_v);
            float Bbar = (Abar - 1.0f) / a_v * b_v;
            state[s] = Abar * state[s] + Bbar * x_v;
        }
        __syncthreads();
        const size_t base_out = ((size_t)ba * seq + t) * channels;
        for (int c = tid; c < channels; c += lanes) {
            float acc = 0.0f;
            for (int s = 0; s < state_dim; ++s) {
                acc += __bfloat162float(c_proj[s * channels + c]) * state[s];
            }
            out[base_out + c] = __float2bfloat16(acc);
        }
        __syncthreads();
    }
}

extern "C" void mamba2_discretize_chunk_bf16_launch(
    const __nv_bfloat16 *d_x, const __nv_bfloat16 *d_a, const __nv_bfloat16 *d_b,
    const __nv_bfloat16 *d_dt, const __nv_bfloat16 *d_c_proj,
    __nv_bfloat16 *d_out,
    int B, int seq, int state_dim, int channels,
    cudaStream_t stream)
{
    int threads = 64;
    if (channels > threads) threads = (channels < 256) ? channels : 256;
    if (state_dim > threads) threads = (state_dim < 256) ? state_dim : 256;
    dim3 grid(B);
    dim3 block(threads);
    size_t shmem = (size_t)state_dim * sizeof(float);
    mamba2_discretize_chunk_bf16_kernel<<<grid, block, shmem, stream>>>(
        d_x, d_a, d_b, d_dt, d_c_proj, d_out, seq, state_dim, channels);
}
