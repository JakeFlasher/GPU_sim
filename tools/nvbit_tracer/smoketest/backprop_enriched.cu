// backprop_enriched.cu — self-contained CUDA smoke test for the host-built
// NVBit tracer. The "backprop" name refers to the backpropagation training
// algorithm exercised by the kernel, not to any external source tree:
// this is a single .cu file with a standalone Makefile and no external
// source pulls. It is a richer-than-vectoradd trace exerciser intended
// to give the tracer concurrent kernels, divergence, atomics, and SFU
// usage in one binary.
//
// Topology:
//   3-layer MLP, IN_DIM=64 → HID_DIM=64 → OUT_DIM=32, batch 16, 2 SGD steps,
//   fixed seed for reproducibility. The goal is instruction coverage, not
//   training accuracy.
//
// Instruction mix exercised (per step, 5 kernel launches):
//   1. matmul_tiled_kernel  (forward L1)  — shared mem tile, __syncthreads, FMA
//   2. matmul_tiled_kernel  (forward L2)  — same, smaller M dim
//   3. relu_softmax_kernel  (fused)       — divergent branch (ReLU), shmem tree
//                                            reduction, __expf (SFU)
//   4. backward_delta_kernel              — mask-gated branch, atomicAdd on
//                                            global scalar, __logf (SFU)
//   5. weight_update_kernel (SGD+momentum) — FMA optimizer step on W2
//
// End-of-run sanity check: every softmax row must be finite and sum to 1.0
// within 1e-3. If the forward pass is broken (NaN/Inf, wrong reduction, etc.)
// the harness returns non-zero and the smoke test fails loudly.
//
// Build:  nvcc -O3 -arch=sm_120 -std=c++17 backprop_enriched.cu -o backprop_enriched
// Run:    ./backprop_enriched

#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---------- hyperparameters ----------

constexpr int IN_DIM  = 64;
constexpr int HID_DIM = 64;
constexpr int OUT_DIM = 32;
constexpr int BATCH   = 16;
constexpr int STEPS   = 2;
constexpr float LR       = 0.01f;
constexpr float MOMENTUM = 0.9f;
constexpr int TILE       = 16;

static_assert((OUT_DIM & (OUT_DIM - 1)) == 0,
              "OUT_DIM must be a power of two for the tree reduction in "
              "relu_softmax_kernel to work.");
static_assert(HID_DIM % TILE == 0,
              "HID_DIM must be a multiple of TILE for matmul_tiled_kernel.");
static_assert(IN_DIM % TILE == 0,
              "IN_DIM must be a multiple of TILE for matmul_tiled_kernel.");

// ---------- error check macro ----------

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _err = (call);                                           \
        if (_err != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                         __FILE__, __LINE__, cudaGetErrorString(_err));      \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// ============================================================================
// Kernel 1: tiled GEMM (row-major, single-precision)
//
// Computes C[M][N] = A[M][K] * B[K][N] using 16x16 shared-memory tiles.
// Used for both forward passes (L1: BATCH x IN → BATCH x HID, L2: BATCH x HID
// → BATCH x OUT). Exercises __syncthreads (twice per outer iter), shared
// memory loads/stores, and FMA via fmaf(). Unrolled inner loop gives the SASS
// a long sequence of HFMA/FMA ops that the tracer will see as a rich burst.
// ============================================================================
__global__ void matmul_tiled_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float*       __restrict__ C,
                                    int M, int K, int N) {
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t * TILE < K; ++t) {
        const int aCol = t * TILE + threadIdx.x;
        const int bRow = t * TILE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc = fmaf(tileA[threadIdx.y][k], tileB[k][threadIdx.x], acc);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ============================================================================
// Kernel 2: fused ReLU + row-wise softmax with numerical stability
//
// One block per batch row; blockDim.x == OUT_DIM == 32 (one full warp on
// sm_120). Exercises:
//   - divergent branch (v > 0 ? v : 0) with mask capture
//   - shared-memory max reduction (log2(OUT_DIM) = 5 iters of __syncthreads)
//   - __expf (SFU)
//   - shared-memory sum reduction (another log2(OUT_DIM) iters)
//   - final elementwise divide and store
//
// Shared memory layout (passed via dynamic shmem size):
//   [0 .. OUT_DIM-1]             = per-element activated value
//   [OUT_DIM .. 2*OUT_DIM-1]     = reduction scratch (reused for max and sum)
// ============================================================================
__global__ void relu_softmax_kernel(float* __restrict__ logits,
                                    float* __restrict__ relu_mask,
                                    int out_dim) {
    extern __shared__ float shmem[];
    float* activated = shmem;
    float* scratch   = shmem + out_dim;

    const int batch_idx = blockIdx.x;
    const int tid       = threadIdx.x;

    // ReLU forward with mask. Divergent branch — under warp-wide shuffle,
    // half the lanes take one path and half the other depending on data.
    const float v = logits[batch_idx * out_dim + tid];
    const float a = (v > 0.0f) ? v : 0.0f;
    activated[tid] = a;
    relu_mask[batch_idx * out_dim + tid] = (v > 0.0f) ? 1.0f : 0.0f;
    __syncthreads();

    // Row max via shared-memory tree reduction.
    scratch[tid] = a;
    __syncthreads();
    for (int stride = out_dim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float x = scratch[tid];
            const float y = scratch[tid + stride];
            scratch[tid]  = (x > y) ? x : y;
        }
        __syncthreads();
    }
    const float row_max = scratch[0];
    __syncthreads();

    // exp(activated - row_max) for numerical stability; SFU op.
    const float e = __expf(a - row_max);
    scratch[tid]  = e;
    __syncthreads();

    // Row sum via shared-memory tree reduction.
    for (int stride = out_dim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    const float row_sum = scratch[0];
    __syncthreads();

    // Normalize and write back. Softmax output now sums to 1.
    logits[batch_idx * out_dim + tid] = e / row_sum;
}

// ============================================================================
// Kernel 3: backward delta + cross-entropy loss accumulation
//
// For each element (b, o):
//   t[o]      = 1 if o == target[b] else 0
//   delta[o]  = mask[o] > 0 ? (y[o] - t[o]) : 0       // ReLU-gated gradient
//
// For each row: atomicAdd(loss_accum, -log(y[target[b]])) — one atomic per row.
//
// Exercises divergent control flow (mask branch), __logf (SFU), and atomicAdd
// on a global float scalar (the loss accumulator).
// ============================================================================
__global__ void backward_delta_kernel(const float* __restrict__ softmax_out,
                                      const int*   __restrict__ targets,
                                      float*       __restrict__ deltas,
                                      const float* __restrict__ relu_mask,
                                      float*       __restrict__ loss_accum,
                                      int out_dim) {
    const int batch_idx = blockIdx.x;
    const int tid       = threadIdx.x;

    if (tid >= out_dim) return;

    const int   target = targets[batch_idx];
    const float y      = softmax_out[batch_idx * out_dim + tid];
    const float t      = (tid == target) ? 1.0f : 0.0f;
    const float mask   = relu_mask[batch_idx * out_dim + tid];

    // Divergent mask-gated gradient: warp splits based on which outputs
    // survived ReLU in the forward pass.
    float grad;
    if (mask > 0.5f) {
        grad = y - t;
    } else {
        grad = 0.0f;
    }
    deltas[batch_idx * out_dim + tid] = grad;

    // Only the target lane accumulates the cross-entropy contribution.
    if (tid == target) {
        const float p = (y > 1e-10f) ? y : 1e-10f;  // clamp for numerical safety
        atomicAdd(loss_accum, -__logf(p));
    }
}

// ============================================================================
// Kernel 4: W2 gradient reduction across the batch
//
// Computes dW2 = hidden^T @ delta, i.e. for each (h, o) pair:
//   gradW2[h][o] = sum over b in [0, BATCH) of hidden[b][h] * delta[b][o]
//
// This is the mathematically correct gradient of the output-layer weights
// w.r.t. the cross-entropy loss under the softmax+ReLU forward. Written as a
// standalone kernel so the backprop smoke test actually performs a real
// backward pass (previously a stand-in used d_deltas directly, which OOB-read
// into d_gradW2's shape because d_deltas is only BATCH*OUT_DIM = 512 floats,
// while the weight matrix has HID_DIM*OUT_DIM = 2048 elements — caught by
// Codex's review).
//
// Exercises: fmaf, compile-time-unrolled batch reduction loop, elementwise
// global loads/stores, 2D thread indexing. Grid/block shapes (8,1)/(16,16).
// ============================================================================
__global__ void compute_grad_W2_kernel(const float* __restrict__ hidden,
                                       const float* __restrict__ delta,
                                       float*       __restrict__ gradW2) {
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= HID_DIM || o >= OUT_DIM) return;

    float sum = 0.0f;
    #pragma unroll
    for (int b = 0; b < BATCH; ++b) {
        sum = fmaf(hidden[b * HID_DIM + h], delta[b * OUT_DIM + o], sum);
    }
    gradW2[h * OUT_DIM + o] = sum;
}

// ============================================================================
// Kernel 5: SGD with momentum — elementwise weight update
//
//   v_new    = μ * v - η * g
//   v[i]     = v_new
//   w[i]    += v_new
//
// Exercises fmaf() for the v update, elementwise global loads/stores, and
// simple 1D thread indexing — no shared memory, no barriers. This is the
// "optimizer step" equivalent that modern training loops spend real time in.
// ============================================================================
__global__ void weight_update_kernel(float*       __restrict__ w,
                                     float*       __restrict__ v,
                                     const float* __restrict__ grad,
                                     float lr, float momentum,
                                     int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // v = momentum * v + (-lr) * grad  →  fmaf(-lr, grad, momentum*v)
    const float v_new = fmaf(-lr, grad[idx], momentum * v[idx]);
    v[idx] = v_new;
    w[idx] = w[idx] + v_new;
}

// ============================================================================
// Host-side driver
// ============================================================================

// Deterministic LCG for reproducible init — no OS entropy, no libstdc++ RNG.
static void fill_uniform(float* h, int n, float lo, float hi,
                         std::uint32_t& seed) {
    for (int i = 0; i < n; ++i) {
        seed = seed * 1103515245u + 12345u;
        const float r = static_cast<float>((seed >> 16) & 0x7FFF) / 32767.0f;
        h[i] = lo + r * (hi - lo);
    }
}

int main(int /*argc*/, char** /*argv*/) {
    std::printf(
        "backprop_enriched: IN=%d HID=%d OUT=%d BATCH=%d STEPS=%d LR=%g "
        "MOM=%g TILE=%d\n",
        IN_DIM, HID_DIM, OUT_DIM, BATCH, STEPS, LR, MOMENTUM, TILE);

    std::uint32_t seed = 42u;

    std::vector<float> h_input(BATCH * IN_DIM);
    std::vector<float> h_W1(IN_DIM * HID_DIM);
    std::vector<float> h_W2(HID_DIM * OUT_DIM);
    std::vector<int>   h_targets(BATCH);

    fill_uniform(h_input.data(), BATCH * IN_DIM, -1.0f, 1.0f, seed);
    fill_uniform(h_W1.data(),    IN_DIM * HID_DIM, -0.1f, 0.1f, seed);
    fill_uniform(h_W2.data(),    HID_DIM * OUT_DIM, -0.1f, 0.1f, seed);
    for (int i = 0; i < BATCH; ++i) h_targets[i] = i % OUT_DIM;

    float *d_input     = nullptr;
    float *d_W1        = nullptr;
    float *d_W2        = nullptr;
    float *d_hidden    = nullptr;
    float *d_logits    = nullptr;
    float *d_relu_mask = nullptr;
    float *d_deltas    = nullptr;
    float *d_gradW2    = nullptr;  // dL/dW2, shape [HID_DIM, OUT_DIM]
    float *d_loss      = nullptr;
    float *d_vW2       = nullptr;
    int   *d_targets   = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input,     sizeof(float) * BATCH  * IN_DIM));
    CUDA_CHECK(cudaMalloc(&d_W1,        sizeof(float) * IN_DIM * HID_DIM));
    CUDA_CHECK(cudaMalloc(&d_W2,        sizeof(float) * HID_DIM* OUT_DIM));
    CUDA_CHECK(cudaMalloc(&d_hidden,    sizeof(float) * BATCH  * HID_DIM));
    CUDA_CHECK(cudaMalloc(&d_logits,    sizeof(float) * BATCH  * OUT_DIM));
    CUDA_CHECK(cudaMalloc(&d_relu_mask, sizeof(float) * BATCH  * OUT_DIM));
    CUDA_CHECK(cudaMalloc(&d_deltas,    sizeof(float) * BATCH  * OUT_DIM));
    CUDA_CHECK(cudaMalloc(&d_gradW2,    sizeof(float) * HID_DIM* OUT_DIM));
    CUDA_CHECK(cudaMalloc(&d_loss,      sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vW2,       sizeof(float) * HID_DIM* OUT_DIM));
    CUDA_CHECK(cudaMalloc(&d_targets,   sizeof(int)   * BATCH));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                          sizeof(float) * BATCH * IN_DIM,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W1, h_W1.data(),
                          sizeof(float) * IN_DIM * HID_DIM,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, h_W2.data(),
                          sizeof(float) * HID_DIM * OUT_DIM,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, h_targets.data(),
                          sizeof(int) * BATCH,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_vW2, 0, sizeof(float) * HID_DIM * OUT_DIM));

    // Snapshot W2 BEFORE training so we can verify at the end that the
    // backward+update path actually mutated it. The earlier sanity check
    // only looked at d_logits (last-written by relu_softmax_kernel), which
    // proves the forward pass stayed sane but says nothing about whether
    // backward_delta_kernel, compute_grad_W2_kernel, and weight_update_kernel
    // actually ran to completion. Caught in review.
    std::vector<float> h_W2_initial(HID_DIM * OUT_DIM);
    CUDA_CHECK(cudaMemcpy(h_W2_initial.data(), d_W2,
                          sizeof(float) * HID_DIM * OUT_DIM,
                          cudaMemcpyDeviceToHost));
    double initial_W2_abs_sum = 0.0;
    for (float w : h_W2_initial) initial_W2_abs_sum += std::fabs(w);

    float loss_h = 0.0f;

    for (int step = 0; step < STEPS; ++step) {
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));

        // --- Forward L1: hidden = input @ W1 ---
        {
            dim3 block(TILE, TILE);
            dim3 grid((HID_DIM + TILE - 1) / TILE,
                      (BATCH   + TILE - 1) / TILE);
            matmul_tiled_kernel<<<grid, block>>>(d_input, d_W1, d_hidden,
                                                 BATCH, IN_DIM, HID_DIM);
        }

        // --- Forward L2: logits = hidden @ W2 ---
        {
            dim3 block(TILE, TILE);
            dim3 grid((OUT_DIM + TILE - 1) / TILE,
                      (BATCH   + TILE - 1) / TILE);
            matmul_tiled_kernel<<<grid, block>>>(d_hidden, d_W2, d_logits,
                                                 BATCH, HID_DIM, OUT_DIM);
        }

        // --- Fused ReLU + softmax ---
        {
            dim3 block(OUT_DIM);
            dim3 grid(BATCH);
            const size_t shmem_bytes = 2 * OUT_DIM * sizeof(float);
            relu_softmax_kernel<<<grid, block, shmem_bytes>>>(
                d_logits, d_relu_mask, OUT_DIM);
        }

        // --- Backward delta + loss ---
        {
            dim3 block(OUT_DIM);
            dim3 grid(BATCH);
            backward_delta_kernel<<<grid, block>>>(
                d_logits, d_targets, d_deltas, d_relu_mask, d_loss, OUT_DIM);
        }

        // --- Compute real dW2 = hidden^T @ delta (batch reduction) ---
        // This replaces the earlier stand-in that reused d_deltas as the
        // gradient buffer; d_deltas is BATCH*OUT_DIM (512 floats) while W2 is
        // HID_DIM*OUT_DIM (2048 floats), so the old call read past the end
        // of d_deltas — caught in review.
        {
            dim3 block(16, 16);
            dim3 grid((OUT_DIM + 15) / 16, (HID_DIM + 15) / 16);
            compute_grad_W2_kernel<<<grid, block>>>(d_hidden, d_deltas,
                                                    d_gradW2);
        }

        // --- SGD+momentum weight update on W2 with the proper gradient ---
        {
            const int n = HID_DIM * OUT_DIM;
            dim3 block(256);
            dim3 grid((n + 255) / 256);
            weight_update_kernel<<<grid, block>>>(d_W2, d_vW2, d_gradW2,
                                                  LR, MOMENTUM, n);
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&loss_h, d_loss, sizeof(float),
                              cudaMemcpyDeviceToHost));
        std::printf("step %d  loss=%.6f\n", step, loss_h / BATCH);
    }

    // --- Host-side sanity check: forward output ---
    std::vector<float> h_out(BATCH * OUT_DIM);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_logits,
                          sizeof(float) * BATCH * OUT_DIM,
                          cudaMemcpyDeviceToHost));

    int failures = 0;
    for (int b = 0; b < BATCH; ++b) {
        float sum = 0.0f;
        for (int o = 0; o < OUT_DIM; ++o) {
            const float v = h_out[b * OUT_DIM + o];
            if (!std::isfinite(v)) {
                std::printf("FAIL: NaN/Inf at row %d col %d\n", b, o);
                ++failures;
            }
            sum += v;
        }
        if (std::fabs(sum - 1.0f) > 1e-3f) {
            std::printf("FAIL: row %d softmax sum = %.6f (expected 1.0)\n",
                        b, sum);
            ++failures;
        }
    }

    // --- Host-side sanity check: optimizer actually mutated W2 ---
    // Prove the backward+update path ran by comparing the current W2 to the
    // snapshot taken before the training loop. We compute the sum of
    // absolute differences and also verify that no entry is NaN/Inf. If the
    // tracer's concurrency fix had broken state publication, we would see
    // either (a) identical W2 (update never ran) or (b) garbage.
    std::vector<float> h_W2_final(HID_DIM * OUT_DIM);
    CUDA_CHECK(cudaMemcpy(h_W2_final.data(), d_W2,
                          sizeof(float) * HID_DIM * OUT_DIM,
                          cudaMemcpyDeviceToHost));

    double W2_abs_delta_sum = 0.0;
    double final_W2_abs_sum = 0.0;
    int    W2_nonfinite   = 0;
    int    W2_changed     = 0;
    for (size_t i = 0; i < h_W2_final.size(); ++i) {
        const float wi = h_W2_final[i];
        if (!std::isfinite(wi)) ++W2_nonfinite;
        final_W2_abs_sum += std::fabs(wi);
        const float diff = wi - h_W2_initial[i];
        W2_abs_delta_sum += std::fabs(diff);
        if (std::fabs(diff) > 0.0f) ++W2_changed;
    }

    std::printf("W2 sanity: |W2_init|_1=%.6f  |W2_final|_1=%.6f  "
                "||ΔW2||_1=%.6f  nonfinite=%d  changed=%d/%zu\n",
                initial_W2_abs_sum, final_W2_abs_sum,
                W2_abs_delta_sum, W2_nonfinite, W2_changed,
                h_W2_final.size());

    if (W2_nonfinite > 0) {
        std::printf("FAIL: %d non-finite weights in W2 after training\n",
                    W2_nonfinite);
        ++failures;
    }
    if (W2_abs_delta_sum <= 0.0) {
        std::printf("FAIL: W2 unchanged after %d training steps "
                    "(backward/update path did not run)\n", STEPS);
        ++failures;
    }

    if (failures == 0) {
        std::printf("PASS: all %d softmax rows finite and sum to 1.0 "
                    "within 1e-3, and W2 was meaningfully updated "
                    "(%d/%zu weights changed)\n",
                    BATCH, W2_changed, h_W2_final.size());
    } else {
        std::printf("FAIL: %d sanity checks failed\n", failures);
    }

    cudaFree(d_input);
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_hidden);
    cudaFree(d_logits);
    cudaFree(d_relu_mask);
    cudaFree(d_deltas);
    cudaFree(d_gradW2);
    cudaFree(d_loss);
    cudaFree(d_vW2);
    cudaFree(d_targets);

    return failures == 0 ? 0 : 1;
}
