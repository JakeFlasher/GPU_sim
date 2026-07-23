// Host driver for moe_router_topk_softmax. Seeds logits via the same PRNG
// as references/_common.py::deterministic_seeded_tensor and emits a JSON
// object shaped like the reference:
//
//     {"expert_ids": [[int, ...] * top_k] * tokens,
//      "routing_weights": [[float, ...] * top_k] * tokens}
//
// Matches references/grouped_gemm.py::build_reference for
// slug == "moe_router_topk_softmax":
//   logits = seeded((tokens, experts), seed, [-2.0, 2.0))

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void moe_router_topk_softmax_launch(
    const __nv_bfloat16 *d_logits,
    int32_t *d_idx,
    __nv_bfloat16 *d_weights,
    int tokens, int experts, int top_k, cudaStream_t stream);


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
    fprintf(stderr, "moe_router_topk_softmax_run: %s (cuda rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}


int main(int argc, char **argv) {
    int tokens = 128;
    int experts = 8;
    int top_k = 2;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    const char *slug = "moe_router_topk_softmax";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--experts") == 0 && i + 1 < argc) experts = atoi(argv[++i]);
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else {
            fprintf(stderr, "moe_router_topk_softmax_run: unknown arg %s\n", argv[i]);
            return 2;
        }
    }

    const size_t n_logits = (size_t)tokens * experts;
    const size_t n_idx = (size_t)tokens * top_k;
    const size_t n_w = (size_t)tokens * top_k;

    float *h_l_f32 = (float *)malloc(n_logits * sizeof(float));
    seed_tensor(h_l_f32, n_logits, seed, -2.0, 2.0);

    __nv_bfloat16 *h_logits = (__nv_bfloat16 *)malloc(n_logits * sizeof(__nv_bfloat16));
    int32_t *h_idx = (int32_t *)malloc(n_idx * sizeof(int32_t));
    __nv_bfloat16 *h_weights = (__nv_bfloat16 *)malloc(n_w * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_logits; ++i) h_logits[i] = __float2bfloat16(h_l_f32[i]);

    __nv_bfloat16 *d_logits = nullptr, *d_weights = nullptr;
    int32_t *d_idx = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_logits, n_logits * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_logits", rc);
    rc = cudaMalloc(&d_idx, n_idx * sizeof(int32_t));             if (rc) fail("cudaMalloc d_idx", rc);
    rc = cudaMalloc(&d_weights, n_w * sizeof(__nv_bfloat16));     if (rc) fail("cudaMalloc d_weights", rc);
    rc = cudaMemcpy(d_logits, h_logits, n_logits * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D logits", rc);

    moe_router_topk_softmax_launch(d_logits, d_idx, d_weights, tokens, experts, top_k, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);

    rc = cudaMemcpy(h_idx, d_idx, n_idx * sizeof(int32_t), cudaMemcpyDeviceToHost);             if (rc) fail("D2H idx", rc);
    rc = cudaMemcpy(h_weights, d_weights, n_w * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost); if (rc) fail("D2H weights", rc);

    printf("{\n");
    printf("  \"schema_version\": 1,\n");
    printf("  \"slug\": \"%s\",\n", slug);
    printf("  \"shape_set\": \"correctness_shapes\",\n");
    printf("  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": {\n");
    printf("    \"expert_ids\": [");
    for (int t = 0; t < tokens; ++t) {
        if (t > 0) printf(",");
        printf("[");
        for (int k = 0; k < top_k; ++k) {
            if (k > 0) printf(",");
            printf("%" PRId32, h_idx[t * top_k + k]);
        }
        printf("]");
    }
    printf("],\n");
    printf("    \"routing_weights\": [");
    for (int t = 0; t < tokens; ++t) {
        if (t > 0) printf(",");
        printf("[");
        for (int k = 0; k < top_k; ++k) {
            if (k > 0) printf(",");
            float v = __bfloat162float(h_weights[t * top_k + k]);
            printf("%.6g", v);
        }
        printf("]");
    }
    printf("]\n  }\n}\n");

    cudaFree(d_logits); cudaFree(d_idx); cudaFree(d_weights);
    free(h_l_f32); free(h_logits); free(h_idx); free(h_weights);
    return 0;
}
