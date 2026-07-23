// slug: mamba_selective_scan_bf16
// tier: P
// arch_gate: sm_90a sm_120
//
// Mamba selective-scan forward pass. Simplified reference-shape-matching
// version: state_dim scalar state per batch; the recurrence
//   state[t] = a[t] * state[t-1] + b[t] * x[t]
// is run in FP32 for numerical stability. Output is the mean of state
// replicated across channels (matches references/ssm.py).
//
// Grid: one block per (batch, channels-tile). Threads within a block each
// maintain their own state_dim element and synchronize via shared memory
// for the mean reduction.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void mamba_selective_scan_bf16_kernel(
    const __nv_bfloat16 *__restrict__ x,       // [batch, seq, channels]
    const __nv_bfloat16 *__restrict__ a,       // [batch, seq, state_dim]
    const __nv_bfloat16 *__restrict__ b,       // [batch, seq, state_dim]
    __nv_bfloat16 *__restrict__ out,           // [batch, seq, channels]
    int32_t seq,
    int32_t channels,
    int32_t state_dim)
{
    const int ba = blockIdx.x;
    const int s = threadIdx.x;
    if (s >= state_dim) return;

    extern __shared__ float shared_state[];
    shared_state[s] = 0.0f;
    __syncthreads();

    for (int t = 0; t < seq; ++t) {
        float av = __bfloat162float(a[((ba * seq) + t) * state_dim + s]);
        float bv = __bfloat162float(b[((ba * seq) + t) * state_dim + s]);
        // Broadcast x[t] across state dim via channel-modulo
        int x_c = s % channels;
        float xv = __bfloat162float(x[((ba * seq) + t) * channels + x_c]);
        shared_state[s] = av * shared_state[s] + bv * xv;
        __syncthreads();

        // Per-time-step mean reduction
        if (s == 0) {
            float sum = 0.0f;
            for (int k = 0; k < state_dim; ++k) sum += shared_state[k];
            float m = sum / (float)state_dim;
            for (int c = 0; c < channels; ++c) {
                out[((ba * seq) + t) * channels + c] = __float2bfloat16(m);
            }
        }
        __syncthreads();
    }
}

extern "C" void mamba_selective_scan_bf16_launch(
    const __nv_bfloat16 *d_x,
    const __nv_bfloat16 *d_a,
    const __nv_bfloat16 *d_b,
    __nv_bfloat16 *d_out,
    int batch, int seq, int channels, int state_dim, cudaStream_t stream)
{
    dim3 grid(batch);
    dim3 block(state_dim);
    size_t shmem = state_dim * sizeof(float);
    mamba_selective_scan_bf16_kernel<<<grid, block, shmem, stream>>>(
        d_x, d_a, d_b, d_out, seq, channels, state_dim);
}
