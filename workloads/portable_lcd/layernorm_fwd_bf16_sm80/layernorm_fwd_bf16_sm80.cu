// slug: layernorm_fwd_bf16_sm80
// tier: P (LCD-portable)
// arch_gate: sm_86 sm_90 sm_120
//
// BF16 LayerNorm forward via single-pass Welford reduction:
//
//   mean[r] = (1/H) * sum_c x[r][c]
//   var[r]  = (1/H) * sum_c (x[r][c] - mean[r])^2
//   y[r][c] = ((x[r][c] - mean[r]) / sqrt(var[r] + eps)) * scale[c] + bias[c]
//
// One CTA per row. Each thread accumulates a per-thread Welford
// triple (cnt, mean, M2) over its strided slice of the row, then
// the per-thread triples are combined pairwise in shared memory
// using Chan / Welford parallel accumulation:
//
//   n        = n_a + n_b
//   delta    = mean_b - mean_a
//   mean     = (n_a * mean_a + n_b * mean_b) / n
//   M2       = M2_a + M2_b + delta^2 * n_a * n_b / n
//
// Reduction is FP32 (numerical stability across H up to ~4096).
// `__syncthreads()` between strides lowers to BAR.SYNC on every
// LCD-target arch — same regression-material discipline as the
// existing rmsnorm_fwd_bf16_sm80 sister kernel.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void layernorm_fwd_bf16_sm80_kernel(
    const __nv_bfloat16 *__restrict__ x,     // [rows, hidden]
    const __nv_bfloat16 *__restrict__ scale, // [hidden]
    const __nv_bfloat16 *__restrict__ bias,  // [hidden]
    __nv_bfloat16       *__restrict__ out,   // [rows, hidden]
    int32_t hidden,
    float eps)
{
    const int row   = blockIdx.x;
    const int tid   = threadIdx.x;
    const int lanes = blockDim.x;

    extern __shared__ float shared[];
    float *cnts  = shared;
    float *means = shared + lanes;
    float *M2s   = shared + 2 * lanes;

    float cnt_t = 0.0f, mean_t = 0.0f, M2_t = 0.0f;
    for (int c = tid; c < hidden; c += lanes) {
        float xv     = __bfloat162float(x[row * hidden + c]);
        cnt_t       += 1.0f;
        float delta  = xv - mean_t;
        mean_t      += delta / cnt_t;
        M2_t        += delta * (xv - mean_t);
    }
    cnts[tid]  = cnt_t;
    means[tid] = mean_t;
    M2s[tid]   = M2_t;
    __syncthreads();

    for (int stride = lanes / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float n_a = cnts[tid],            m_a  = means[tid],            M2_a = M2s[tid];
            float n_b = cnts[tid + stride],   m_b  = means[tid + stride],   M2_b = M2s[tid + stride];
            float n   = n_a + n_b;
            if (n > 0.0f) {
                float delta = m_b - m_a;
                float new_mean = (n_a * m_a + n_b * m_b) / n;
                float new_M2   = M2_a + M2_b + delta * delta * n_a * n_b / n;
                cnts[tid]  = n;
                means[tid] = new_mean;
                M2s[tid]   = new_M2;
            }
        }
        __syncthreads();
    }

    float total_cnt = cnts[0];
    float mean      = means[0];
    float var       = M2s[0] / total_cnt;
    float inv_std   = rsqrtf(var + eps);

    for (int c = tid; c < hidden; c += lanes) {
        float xv = __bfloat162float(x[row * hidden + c]);
        float sv = __bfloat162float(scale[c]);
        float bv = __bfloat162float(bias[c]);
        float y  = (xv - mean) * inv_std * sv + bv;
        out[row * hidden + c] = __float2bfloat16(y);
    }
}

extern "C" void layernorm_fwd_bf16_sm80_launch(
    const __nv_bfloat16 *d_x,
    const __nv_bfloat16 *d_scale,
    const __nv_bfloat16 *d_bias,
    __nv_bfloat16 *d_out,
    int rows, int hidden, float eps, cudaStream_t stream)
{
    dim3 grid(rows);
    dim3 block(256);
    size_t shmem = 3 * block.x * sizeof(float);
    layernorm_fwd_bf16_sm80_kernel<<<grid, block, shmem, stream>>>(
        d_x, d_scale, d_bias, d_out, hidden, eps);
}
