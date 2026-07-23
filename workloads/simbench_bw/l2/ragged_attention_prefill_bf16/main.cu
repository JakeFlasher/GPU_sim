// Host driver for ragged_attention_prefill_bf16. Seeds Q / K / V via the
// same MWC PRNG as references/_common.py::deterministic_seeded_tensor
// and derives cu_seqlens from --num-seqs by the SAME even-partition
// rule the Python reference uses, so the GPU output matches the
// reference segment-for-segment under the (BF16, tree_sum, attention)
// tolerance.
//
// Matches the segmented branch of references/attention.py for slug
// "ragged_attention_prefill_bf16" (non-causal, num_seqs > 1 triggers
// per-segment attention). Seeds (batch default = 1):
//   q = seeded((heads, total_tokens, head_dim), seed,       [-0.5, 0.5))
//   k = seeded((heads, total_tokens, head_dim), seed + 100, [-0.5, 0.5))
//   v = seeded((heads, total_tokens, head_dim), seed + 200, [-0.5, 0.5))
// cu_seqlens = even_partition(total_tokens, num_seqs): segments are
// base = total_tokens / num_seqs tokens each, first (total_tokens %
// num_seqs) segments get one extra token. This matches
// references/attention.py::_even_cu_seqlens.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void ragged_attention_prefill_bf16_launch(
    const __nv_bfloat16 *d_q,
    const __nv_bfloat16 *d_k,
    const __nv_bfloat16 *d_v,
    __nv_bfloat16 *d_out,
    const int32_t *d_cu_seqlens,
    int heads, int total_tokens, int head_dim, int num_seqs,
    float scale, cudaStream_t stream);


static inline uint32_t next_u32(uint32_t *state) {
    *state = (*state) * 1103515245u + 12345u;
    return *state;
}

static inline float next_float(uint32_t *state, double low, double high) {
    double r = (double)next_u32(state) / 4294967296.0;
    return (float)(low + (high - low) * r);
}

static void seed_tensor(float *flat, size_t n, uint32_t seed, double low, double high) {
    uint32_t state = (seed * 2654435761u + 1u);
    for (size_t i = 0; i < n; ++i) flat[i] = next_float(&state, low, high);
}

static void fail(const char *msg, int rc) {
    fprintf(stderr, "ragged_attention_prefill_bf16_run: %s (cuda rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}


int main(int argc, char **argv) {
    int total_tokens = 1024;
    int head_dim = 64;
    int heads = 8;
    int num_seqs = 4;   // triggers the per-segment-isolated path in the reference
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    const char *slug = "ragged_attention_prefill_bf16";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--total-tokens") == 0 && i + 1 < argc) total_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--head-dim") == 0 && i + 1 < argc) head_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--heads") == 0 && i + 1 < argc) heads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--num-seqs") == 0 && i + 1 < argc) num_seqs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else {
            fprintf(stderr, "ragged_attention_prefill_bf16_run: unknown arg %s\n", argv[i]);
            return 2;
        }
    }
    if (num_seqs < 1 || num_seqs > total_tokens) {
        fprintf(stderr, "ragged_attention_prefill_bf16_run: num_seqs must be in [1, total_tokens]\n");
        return 2;
    }

    const size_t n = (size_t)heads * total_tokens * head_dim;
    const float scale = 1.0f / sqrtf((float)head_dim);

    // Derive cu_seqlens via the same even-partition rule as
    // references/attention.py::_even_cu_seqlens. Length: num_seqs + 1.
    int32_t *h_cu = (int32_t *)malloc((size_t)(num_seqs + 1) * sizeof(int32_t));
    {
        int base  = total_tokens / num_seqs;
        int extra = total_tokens % num_seqs;
        int off = 0;
        h_cu[0] = 0;
        for (int s = 0; s < num_seqs; ++s) {
            int seg_len = base + (s < extra ? 1 : 0);
            off += seg_len;
            h_cu[s + 1] = off;
        }
        if (h_cu[num_seqs] != total_tokens) {
            fprintf(stderr, "ragged_attention_prefill_bf16_run: cu_seqlens partition bug\n");
            return 1;
        }
    }

    float *h_q_f32 = (float *)malloc(n * sizeof(float));
    float *h_k_f32 = (float *)malloc(n * sizeof(float));
    float *h_v_f32 = (float *)malloc(n * sizeof(float));
    seed_tensor(h_q_f32, n, seed,       -0.5, 0.5);
    seed_tensor(h_k_f32, n, seed + 100, -0.5, 0.5);
    seed_tensor(h_v_f32, n, seed + 200, -0.5, 0.5);

    __nv_bfloat16 *h_q = (__nv_bfloat16 *)malloc(n * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_k = (__nv_bfloat16 *)malloc(n * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_v = (__nv_bfloat16 *)malloc(n * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_out = (__nv_bfloat16 *)malloc(n * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n; ++i) h_q[i] = __float2bfloat16(h_q_f32[i]);
    for (size_t i = 0; i < n; ++i) h_k[i] = __float2bfloat16(h_k_f32[i]);
    for (size_t i = 0; i < n; ++i) h_v[i] = __float2bfloat16(h_v_f32[i]);

    __nv_bfloat16 *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_out = nullptr;
    int32_t *d_cu = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_q,   n * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_q",   rc);
    rc = cudaMalloc(&d_k,   n * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_k",   rc);
    rc = cudaMalloc(&d_v,   n * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_v",   rc);
    rc = cudaMalloc(&d_out, n * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_out", rc);
    rc = cudaMalloc(&d_cu,  (size_t)(num_seqs + 1) * sizeof(int32_t)); if (rc) fail("cudaMalloc d_cu", rc);
    rc = cudaMemcpy(d_q, h_q, n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D q", rc);
    rc = cudaMemcpy(d_k, h_k, n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D k", rc);
    rc = cudaMemcpy(d_v, h_v, n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D v", rc);
    rc = cudaMemcpy(d_cu, h_cu, (size_t)(num_seqs + 1) * sizeof(int32_t), cudaMemcpyHostToDevice); if (rc) fail("H2D cu_seqlens", rc);

    ragged_attention_prefill_bf16_launch(d_q, d_k, d_v, d_out, d_cu, heads, total_tokens, head_dim, num_seqs, scale, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);

    rc = cudaMemcpy(h_out, d_out, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost); if (rc) fail("D2H out", rc);

    // Reference returns a [batch=1][heads][seq=total_tokens][head_dim] nested list.
    printf("{\n");
    printf("  \"schema_version\": 1,\n");
    printf("  \"slug\": \"%s\",\n", slug);
    printf("  \"shape_set\": \"correctness_shapes\",\n");
    printf("  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [[");  // outer [batch=1] then [heads]
    for (int h = 0; h < heads; ++h) {
        if (h > 0) printf(",");
        printf("[");
        for (int i = 0; i < total_tokens; ++i) {
            if (i > 0) printf(",");
            printf("[");
            for (int d = 0; d < head_dim; ++d) {
                if (d > 0) printf(",");
                float v = __bfloat162float(h_out[(h * total_tokens + i) * head_dim + d]);
                printf("%.6g", v);
            }
            printf("]");
        }
        printf("]");
    }
    printf("]]\n}\n");

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out); cudaFree(d_cu);
    free(h_cu);
    free(h_q_f32); free(h_k_f32); free(h_v_f32);
    free(h_q); free(h_k); free(h_v); free(h_out);
    return 0;
}
