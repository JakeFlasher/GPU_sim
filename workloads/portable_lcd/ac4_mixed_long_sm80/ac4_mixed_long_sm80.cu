// One-binary long timing runner for mixed-family timing coverage.
//
// The kernels are deliberately simple and deterministic. The purpose of this
// runner is to provide one current binary identity that dispatches GEMM,
// conv/attention, memory-streaming, and irregular atomic long-duration cases
// by --case_id, while keeping all label capture in the local-label quarantine.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace {

constexpr int GEMM_K_TILE = 16;

__device__ inline uint32_t cp_async_smem_addr(const void *ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

static inline uint32_t next_u32(uint32_t *s) {
    *s = (*s) * 1103515245u + 12345u;
    return *s;
}

static void seed_f32(float *flat, size_t n, uint32_t seed, float lo, float hi) {
    uint32_t st = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) {
        float r = (float)((double)next_u32(&st) / 4294967296.0);
        flat[i] = lo + (hi - lo) * r;
    }
}

static void seed_bf16(__nv_bfloat16 *out, size_t n, uint32_t seed) {
    float *tmp = (float *)malloc(n * sizeof(float));
    if (!tmp) {
        fprintf(stderr, "ac4_mixed_long_sm80_run: host allocation failed\n");
        exit(1);
    }
    seed_f32(tmp, n, seed, -0.5f, 0.5f);
    for (size_t i = 0; i < n; ++i) out[i] = __float2bfloat16(tmp[i]);
    free(tmp);
}

static void fail_cuda(const char *what, cudaError_t rc) {
    fprintf(stderr, "ac4_mixed_long_sm80_run: %s failed (%d: %s)\n",
            what, (int)rc, cudaGetErrorString(rc));
    exit(1);
}

static int parse_int_arg(int argc, char **argv, const char *name, int fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], name) == 0) return atoi(argv[i + 1]);
    }
    return fallback;
}

static const char *parse_str_arg(
    int argc, char **argv, const char *name, const char *fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    return fallback;
}

static bool arg_present(int argc, char **argv, const char *name) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], name) == 0) return true;
    }
    return false;
}

template <typename LaunchFn>
static double time_one_or_more(int warmup, int reps, LaunchFn launch) {
    for (int w = 0; w < warmup; ++w) launch();
    cudaError_t rc = cudaDeviceSynchronize();
    if (rc) fail_cuda("warmup synchronize", rc);

    cudaEvent_t start, stop;
    rc = cudaEventCreate(&start);
    if (rc) fail_cuda("event create start", rc);
    rc = cudaEventCreate(&stop);
    if (rc) fail_cuda("event create stop", rc);
    rc = cudaEventRecord(start, 0);
    if (rc) fail_cuda("event record start", rc);
    for (int r = 0; r < reps; ++r) launch();
    rc = cudaEventRecord(stop, 0);
    if (rc) fail_cuda("event record stop", rc);
    rc = cudaEventSynchronize(stop);
    if (rc) fail_cuda("event synchronize", rc);
    float elapsed_ms = 0.0f;
    rc = cudaEventElapsedTime(&elapsed_ms, start, stop);
    if (rc) fail_cuda("event elapsed", rc);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (!isfinite((double)elapsed_ms) || elapsed_ms <= 0.0f) {
        fprintf(stderr, "ac4_mixed_long_sm80_run: non-positive event time\n");
        exit(1);
    }
    return (double)elapsed_ms;
}

}  // namespace

extern "C" __global__ void gemm_hmma_bf16_sm80_kernel(
    const __nv_bfloat16 *__restrict__ A,
    const __nv_bfloat16 *__restrict__ B,
    float *__restrict__ C,
    int32_t M, int32_t N, int32_t K)
{
    const int tile_m  = blockIdx.x;
    const int tile_n  = blockIdx.y;
    const int lane    = threadIdx.x;
    const int groupID = lane >> 2;
    const int tidg    = lane & 3;

    __shared__ __nv_bfloat16 sA[16 * GEMM_K_TILE];
    __shared__ __nv_bfloat16 sB[8 * GEMM_K_TILE];

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    const int gA_row_base = tile_m * 16;
    const int gB_row_base = tile_n * 8;

    for (int k0 = 0; k0 < K; k0 += GEMM_K_TILE) {
        const int a_row = lane >> 1;
        const int a_col = (lane & 1) << 3;
        const int a_gmem = (gA_row_base + a_row) * K + k0 + a_col;
        const int a_smem = a_row * GEMM_K_TILE + a_col;
        uint32_t a_saddr = cp_async_smem_addr(&sA[a_smem]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                     :: "r"(a_saddr), "l"(A + a_gmem));

        if (lane < 16) {
            const int b_row = lane >> 1;
            const int b_col = (lane & 1) << 3;
            const int b_gmem = (gB_row_base + b_row) * K + k0 + b_col;
            const int b_smem = b_row * GEMM_K_TILE + b_col;
            uint32_t b_saddr = cp_async_smem_addr(&sB[b_smem]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                         :: "r"(b_saddr), "l"(B + b_gmem));
        }

        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_all;\n" ::);
        __syncthreads();

        const int a_row0 = groupID;
        const int a_row1 = groupID + 8;
        const int b_row = groupID;
        const uint32_t a0 = *reinterpret_cast<const uint32_t *>(
            &sA[a_row0 * GEMM_K_TILE + tidg * 2 + 0]);
        const uint32_t a1 = *reinterpret_cast<const uint32_t *>(
            &sA[a_row1 * GEMM_K_TILE + tidg * 2 + 0]);
        const uint32_t a2 = *reinterpret_cast<const uint32_t *>(
            &sA[a_row0 * GEMM_K_TILE + tidg * 2 + 8]);
        const uint32_t a3 = *reinterpret_cast<const uint32_t *>(
            &sA[a_row1 * GEMM_K_TILE + tidg * 2 + 8]);
        const uint32_t b0 = *reinterpret_cast<const uint32_t *>(
            &sB[b_row * GEMM_K_TILE + tidg * 2 + 0]);
        const uint32_t b1 = *reinterpret_cast<const uint32_t *>(
            &sB[b_row * GEMM_K_TILE + tidg * 2 + 8]);

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));

        __syncthreads();
    }

    const int c_row0 = tile_m * 16 + groupID;
    const int c_row1 = tile_m * 16 + groupID + 8;
    const int c_col = tile_n * 8 + tidg * 2;
    C[c_row0 * N + c_col + 0] = d0;
    C[c_row0 * N + c_col + 1] = d1;
    C[c_row1 * N + c_col + 0] = d2;
    C[c_row1 * N + c_col + 1] = d3;
}

extern "C" __global__ void conv2d_implicit_gemm_bf16_sm80_kernel(
    const __nv_bfloat16 *__restrict__ act,
    const __nv_bfloat16 *__restrict__ filt,
    __nv_bfloat16 *__restrict__ out,
    int32_t batch, int32_t h, int32_t w, int32_t channels,
    int32_t filters, int32_t r, int32_t s, int32_t repeats)
{
    const int p_extent = h;
    const int q_extent = w;
    const int64_t total = (int64_t)batch * p_extent * q_extent * filters;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        int64_t t = idx;
        const int k = (int)(t % filters); t /= filters;
        const int q = (int)(t % q_extent); t /= q_extent;
        const int p = (int)(t % p_extent); t /= p_extent;
        const int n = (int)t;
        float acc = 0.0f;
        for (int rep = 0; rep < repeats; ++rep) {
            for (int rr = 0; rr < r; ++rr) {
                const int ih = p + rr - r / 2;
                if (ih < 0 || ih >= h) continue;
                for (int ss = 0; ss < s; ++ss) {
                    const int iw = q + ss - s / 2;
                    if (iw < 0 || iw >= w) continue;
                    for (int c = 0; c < channels; ++c) {
                        const int64_t a_idx =
                            ((int64_t)n * h * w + ih * w + iw) * channels + c;
                        const int64_t f_idx =
                            (((int64_t)k * r + rr) * s + ss) * channels + c;
                        acc += __bfloat162float(act[a_idx])
                             * __bfloat162float(filt[f_idx]);
                    }
                }
            }
        }
        out[idx] = __float2bfloat16(acc * (1.0f / (float)repeats));
    }
}

extern "C" __global__ void flash_attn_fwd_bf16_sm80_kernel(
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v,
    __nv_bfloat16 *__restrict__ out,
    int32_t bh, int32_t sq, int32_t sk, int32_t d, int32_t repeats)
{
    const int64_t total = (int64_t)bh * sq * d;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        int64_t t = idx;
        const int dd = (int)(t % d); t /= d;
        const int row = (int)(t % sq); t /= sq;
        const int b = (int)t;
        const float qv = __bfloat162float(q[((int64_t)b * sq + row) * d + dd]);
        float acc = 0.0f;
        for (int rep = 0; rep < repeats; ++rep) {
            for (int col = 0; col < sk; ++col) {
                const float kval = __bfloat162float(k[((int64_t)b * sk + col) * d + dd]);
                const float vval = __bfloat162float(v[((int64_t)b * sk + col) * d + dd]);
                acc += (qv * kval) * vval;
            }
        }
        out[idx] = __float2bfloat16(acc / (float)(sk * repeats));
    }
}

extern "C" __global__ void dram_streaming_dot_product_kernel(
    const float *__restrict__ a,
    const float *__restrict__ b,
    float *__restrict__ partial,
    int32_t n, int32_t passes)
{
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    float acc = 0.0f;
    for (int p = 0; p < passes; ++p) {
        for (int i = gid; i < n; i += stride) {
            acc += a[i] * b[i];
        }
    }
    sdata[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

extern "C" __global__ void global_atomic_histogram_kernel(
    int32_t *__restrict__ histo,
    float *__restrict__ out,
    int32_t updates_per_block,
    int32_t bins)
{
    const int tid = threadIdx.x;
    const int base = blockIdx.x * bins;
    for (int i = tid; i < updates_per_block; i += blockDim.x) {
        const int bin = (i * 2654435769u + blockIdx.x) % bins;
        atomicAdd(&histo[base + bin], 1);
    }
    __syncthreads();
    for (int bin = tid; bin < bins; bin += blockDim.x) {
        out[base + bin] = (float)histo[base + bin];
    }
}

static double run_gemm(int argc, char **argv) {
    const int m = parse_int_arg(argc, argv, "--m", 2048);
    const int n = parse_int_arg(argc, argv, "--n", 2048);
    const int k = parse_int_arg(argc, argv, "--k", 2048);
    const int warmup = parse_int_arg(argc, argv, "--warmup", 0);
    const int reps = parse_int_arg(argc, argv, "--reps", 1);
    const uint32_t seed = (uint32_t)parse_int_arg(argc, argv, "--seed", 42);
    if (m <= 0 || n <= 0 || k <= 0 || (m % 16) || (n % 8) || (k % 16)) {
        fprintf(stderr, "gemm_hmma_bf16_sm80: dimensions must be tile aligned\n");
        return -1.0;
    }
    const size_t n_a = (size_t)m * k;
    const size_t n_b = (size_t)n * k;
    const size_t n_c = (size_t)m * n;
    __nv_bfloat16 *h_a = (__nv_bfloat16 *)malloc(n_a * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_b = (__nv_bfloat16 *)malloc(n_b * sizeof(__nv_bfloat16));
    seed_bf16(h_a, n_a, seed);
    seed_bf16(h_b, n_b, seed + 1);
    __nv_bfloat16 *d_a = nullptr, *d_b = nullptr;
    float *d_c = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_a, n_a * sizeof(__nv_bfloat16)); if (rc) fail_cuda("malloc A", rc);
    rc = cudaMalloc(&d_b, n_b * sizeof(__nv_bfloat16)); if (rc) fail_cuda("malloc B", rc);
    rc = cudaMalloc(&d_c, n_c * sizeof(float)); if (rc) fail_cuda("malloc C", rc);
    rc = cudaMemcpy(d_a, h_a, n_a * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail_cuda("copy A", rc);
    rc = cudaMemcpy(d_b, h_b, n_b * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail_cuda("copy B", rc);
    dim3 grid((m + 15) / 16, (n + 7) / 8, 1);
    dim3 block(32, 1, 1);
    double ms = time_one_or_more(warmup, reps, [&]() {
        gemm_hmma_bf16_sm80_kernel<<<grid, block, 0>>>(d_a, d_b, d_c, m, n, k);
    });
    rc = cudaGetLastError(); if (rc) fail_cuda("gemm launch", rc);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b);
    return ms;
}

static double run_conv(int argc, char **argv) {
    const int batch = parse_int_arg(argc, argv, "--batch", 8);
    const int h = parse_int_arg(argc, argv, "--h", 32);
    const int w = parse_int_arg(argc, argv, "--w", 32);
    const int channels = parse_int_arg(argc, argv, "--channels", 32);
    const int filters = parse_int_arg(argc, argv, "--filters", 32);
    const int r = parse_int_arg(argc, argv, "--r", 3);
    const int s = parse_int_arg(argc, argv, "--s", 3);
    const int inner_repeats = parse_int_arg(argc, argv, "--inner-repeats", 1);
    const int blocks = parse_int_arg(argc, argv, "--blocks", 512);
    const int threads = parse_int_arg(argc, argv, "--threads", 256);
    const int warmup = parse_int_arg(argc, argv, "--warmup", 0);
    const int reps = parse_int_arg(argc, argv, "--reps", 1);
    const uint32_t seed = (uint32_t)parse_int_arg(argc, argv, "--seed", 42);
    const size_t n_act = (size_t)batch * h * w * channels;
    const size_t n_filt = (size_t)filters * r * s * channels;
    const size_t n_out = (size_t)batch * h * w * filters;
    __nv_bfloat16 *h_act = (__nv_bfloat16 *)malloc(n_act * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_filt = (__nv_bfloat16 *)malloc(n_filt * sizeof(__nv_bfloat16));
    seed_bf16(h_act, n_act, seed);
    seed_bf16(h_filt, n_filt, seed + 1);
    __nv_bfloat16 *d_act = nullptr, *d_filt = nullptr, *d_out = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_act, n_act * sizeof(__nv_bfloat16)); if (rc) fail_cuda("malloc act", rc);
    rc = cudaMalloc(&d_filt, n_filt * sizeof(__nv_bfloat16)); if (rc) fail_cuda("malloc filt", rc);
    rc = cudaMalloc(&d_out, n_out * sizeof(__nv_bfloat16)); if (rc) fail_cuda("malloc out", rc);
    rc = cudaMemcpy(d_act, h_act, n_act * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail_cuda("copy act", rc);
    rc = cudaMemcpy(d_filt, h_filt, n_filt * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail_cuda("copy filt", rc);
    double ms = time_one_or_more(warmup, reps, [&]() {
        conv2d_implicit_gemm_bf16_sm80_kernel<<<blocks, threads>>>(
            d_act, d_filt, d_out, batch, h, w, channels, filters, r, s,
            inner_repeats);
    });
    rc = cudaGetLastError(); if (rc) fail_cuda("conv launch", rc);
    cudaFree(d_act); cudaFree(d_filt); cudaFree(d_out);
    free(h_act); free(h_filt);
    return ms;
}

static double run_flash(int argc, char **argv) {
    const int bh = parse_int_arg(argc, argv, "--bh", 8);
    const int sq = parse_int_arg(argc, argv, "--sq", 128);
    const int sk = parse_int_arg(argc, argv, "--sk", 128);
    const int d = parse_int_arg(argc, argv, "--d", 16);
    const int inner_repeats = parse_int_arg(argc, argv, "--inner-repeats", 1);
    const int blocks = parse_int_arg(argc, argv, "--blocks", 512);
    const int threads = parse_int_arg(argc, argv, "--threads", 256);
    const int warmup = parse_int_arg(argc, argv, "--warmup", 0);
    const int reps = parse_int_arg(argc, argv, "--reps", 1);
    const uint32_t seed = (uint32_t)parse_int_arg(argc, argv, "--seed", 42);
    const size_t n_q = (size_t)bh * sq * d;
    const size_t n_kv = (size_t)bh * sk * d;
    __nv_bfloat16 *h_q = (__nv_bfloat16 *)malloc(n_q * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_k = (__nv_bfloat16 *)malloc(n_kv * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_v = (__nv_bfloat16 *)malloc(n_kv * sizeof(__nv_bfloat16));
    seed_bf16(h_q, n_q, seed);
    seed_bf16(h_k, n_kv, seed + 1);
    seed_bf16(h_v, n_kv, seed + 2);
    __nv_bfloat16 *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_out = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_q, n_q * sizeof(__nv_bfloat16)); if (rc) fail_cuda("malloc q", rc);
    rc = cudaMalloc(&d_k, n_kv * sizeof(__nv_bfloat16)); if (rc) fail_cuda("malloc k", rc);
    rc = cudaMalloc(&d_v, n_kv * sizeof(__nv_bfloat16)); if (rc) fail_cuda("malloc v", rc);
    rc = cudaMalloc(&d_out, n_q * sizeof(__nv_bfloat16)); if (rc) fail_cuda("malloc out", rc);
    rc = cudaMemcpy(d_q, h_q, n_q * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail_cuda("copy q", rc);
    rc = cudaMemcpy(d_k, h_k, n_kv * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail_cuda("copy k", rc);
    rc = cudaMemcpy(d_v, h_v, n_kv * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail_cuda("copy v", rc);
    double ms = time_one_or_more(warmup, reps, [&]() {
        flash_attn_fwd_bf16_sm80_kernel<<<blocks, threads>>>(
            d_q, d_k, d_v, d_out, bh, sq, sk, d, inner_repeats);
    });
    rc = cudaGetLastError(); if (rc) fail_cuda("flash launch", rc);
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
    free(h_q); free(h_k); free(h_v);
    return ms;
}

static double run_dot(int argc, char **argv) {
    const int n = parse_int_arg(argc, argv, "--n", 1 << 24);
    const int passes = parse_int_arg(argc, argv, "--passes", 8);
    const int blocks = parse_int_arg(argc, argv, "--blocks", 512);
    const int threads = parse_int_arg(argc, argv, "--threads", 256);
    const int warmup = parse_int_arg(argc, argv, "--warmup", 0);
    const int reps = parse_int_arg(argc, argv, "--reps", 1);
    const uint32_t seed = (uint32_t)parse_int_arg(argc, argv, "--seed", 42);
    float *h_a = (float *)malloc((size_t)n * sizeof(float));
    float *h_b = (float *)malloc((size_t)n * sizeof(float));
    seed_f32(h_a, n, seed, -1.0f, 1.0f);
    seed_f32(h_b, n, seed + 1, -1.0f, 1.0f);
    float *d_a = nullptr, *d_b = nullptr, *d_partial = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_a, (size_t)n * sizeof(float)); if (rc) fail_cuda("malloc a", rc);
    rc = cudaMalloc(&d_b, (size_t)n * sizeof(float)); if (rc) fail_cuda("malloc b", rc);
    rc = cudaMalloc(&d_partial, (size_t)blocks * sizeof(float)); if (rc) fail_cuda("malloc partial", rc);
    rc = cudaMemcpy(d_a, h_a, (size_t)n * sizeof(float), cudaMemcpyHostToDevice); if (rc) fail_cuda("copy a", rc);
    rc = cudaMemcpy(d_b, h_b, (size_t)n * sizeof(float), cudaMemcpyHostToDevice); if (rc) fail_cuda("copy b", rc);
    const size_t shmem = (size_t)threads * sizeof(float);
    double ms = time_one_or_more(warmup, reps, [&]() {
        dram_streaming_dot_product_kernel<<<blocks, threads, shmem>>>(
            d_a, d_b, d_partial, n, passes);
    });
    rc = cudaGetLastError(); if (rc) fail_cuda("dot launch", rc);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_partial);
    free(h_a); free(h_b);
    return ms;
}

static double run_atomic(int argc, char **argv) {
    const int updates = parse_int_arg(argc, argv, "--updates", 1 << 20);
    const int bins = parse_int_arg(argc, argv, "--bins", 512);
    const int blocks = parse_int_arg(argc, argv, "--blocks", 256);
    const int threads = parse_int_arg(argc, argv, "--threads", 256);
    const int warmup = parse_int_arg(argc, argv, "--warmup", 0);
    const int reps = parse_int_arg(argc, argv, "--reps", 1);
    int32_t *d_histo = nullptr;
    float *d_out = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_histo, (size_t)blocks * bins * sizeof(int32_t)); if (rc) fail_cuda("malloc histo", rc);
    rc = cudaMalloc(&d_out, (size_t)blocks * bins * sizeof(float)); if (rc) fail_cuda("malloc out", rc);
    rc = cudaMemset(d_histo, 0, (size_t)blocks * bins * sizeof(int32_t)); if (rc) fail_cuda("memset histo", rc);
    double ms = time_one_or_more(warmup, reps, [&]() {
        global_atomic_histogram_kernel<<<blocks, threads>>>(d_histo, d_out, updates, bins);
    });
    rc = cudaGetLastError(); if (rc) fail_cuda("atomic launch", rc);
    cudaFree(d_histo); cudaFree(d_out);
    return ms;
}

int main(int argc, char **argv) {
    if (arg_present(argc, argv, "--help")) {
        printf("usage: ac4_mixed_long_sm80_run --case_id <case> [shape args]\n");
        return 0;
    }
    const char *case_id = parse_str_arg(argc, argv, "--case_id", "");
    const char *shape_point = parse_str_arg(argc, argv, "--shape-point", "manual");
    double ms = -1.0;
    const char *slug = case_id;
    if (strcmp(case_id, "gemm_hmma_bf16_sm80") == 0) {
        slug = "gemm_hmma_bf16_sm80";
        ms = run_gemm(argc, argv);
    } else if (strcmp(case_id, "conv2d_implicit_gemm_bf16_sm80") == 0) {
        slug = "conv2d_implicit_gemm_bf16_sm80";
        ms = run_conv(argc, argv);
    } else if (strcmp(case_id, "flash_attn_fwd_bf16_sm80") == 0) {
        slug = "flash_attn_fwd_bf16_sm80";
        ms = run_flash(argc, argv);
    } else if (strcmp(case_id, "dram_streaming_dot_product") == 0) {
        slug = "dram_streaming_dot_product";
        ms = run_dot(argc, argv);
    } else if (strcmp(case_id, "global_atomic_histogram") == 0) {
        slug = "global_atomic_histogram";
        ms = run_atomic(argc, argv);
    } else {
        fprintf(stderr, "ac4_mixed_long_sm80_run: unknown --case_id '%s'\n", case_id);
        return 2;
    }
    if (ms <= 0.0) return 2;
    printf("{\n");
    printf("  \"schema_version\": 1,\n");
    printf("  \"binary_slug\": \"ac4_mixed_long_sm80\",\n");
    printf("  \"case_id\": \"%s\",\n", case_id);
    printf("  \"slug\": \"%s\",\n", slug);
    printf("  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"mean_ms\": %.9f\n", ms);
    printf("}\n");
    return 0;
}
