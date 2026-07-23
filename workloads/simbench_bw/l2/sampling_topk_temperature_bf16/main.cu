// Host driver for sampling_topk_temperature_bf16.
//
// Seeds a [batch, vocab] BF16 logits tensor deterministically, runs the
// kernel to emit top-k indices + temperature-scaled softmax probabilities,
// and serializes the result as {"indices": [[...]], "probabilities": [[...]]}
// matching references/sampling.py's output dict. The JSON wraps that under
// the candidate envelope update_goldens.py expects.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void sampling_topk_temperature_bf16_launch(
    const __nv_bfloat16 *d_logits, int32_t *d_idx, __nv_bfloat16 *d_probs,
    int batch, int vocab, int top_k, float temperature, cudaStream_t stream);


static inline uint32_t next_u32(uint32_t *state) {
    *state = (*state) * 1103515245u + 12345u;
    return *state;
}


static void seed_tensor(float *flat, size_t n, uint32_t seed, double low, double high) {
    uint32_t state = (seed * 2654435761u + 1u);
    for (size_t i = 0; i < n; ++i) {
        double r = (double)next_u32(&state) / 4294967296.0;
        flat[i] = (float)(low + (high - low) * r);
    }
}


static void fail(const char *msg, int rc) {
    fprintf(stderr, "sampling_topk_temperature_bf16_run: %s (cuda rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}


int main(int argc, char **argv) {
    int batch = 1;
    int vocab = 4096;
    int top_k = 10;
    float temperature = 1.0f;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    const char *slug = "sampling_topk_temperature_bf16";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) batch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) vocab = atoi(argv[++i]);
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) temperature = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else {
            fprintf(stderr, "sampling_topk_temperature_bf16_run: unknown arg %s\n", argv[i]);
            return 2;
        }
    }

    const size_t n_logits = (size_t)batch * vocab;
    float *h_logits_f32 = (float *)malloc(n_logits * sizeof(float));
    seed_tensor(h_logits_f32, n_logits, seed, -4.0, 4.0);

    __nv_bfloat16 *h_logits = (__nv_bfloat16 *)malloc(n_logits * sizeof(__nv_bfloat16));
    int32_t *h_idx = (int32_t *)malloc((size_t)batch * top_k * sizeof(int32_t));
    __nv_bfloat16 *h_probs = (__nv_bfloat16 *)malloc((size_t)batch * top_k * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_logits; ++i) h_logits[i] = __float2bfloat16(h_logits_f32[i]);

    __nv_bfloat16 *d_logits = nullptr, *d_probs = nullptr;
    int32_t *d_idx = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_logits, n_logits * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_logits", rc);
    rc = cudaMalloc(&d_idx, (size_t)batch * top_k * sizeof(int32_t)); if (rc) fail("cudaMalloc d_idx", rc);
    rc = cudaMalloc(&d_probs, (size_t)batch * top_k * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_probs", rc);
    rc = cudaMemcpy(d_logits, h_logits, n_logits * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D logits", rc);

    sampling_topk_temperature_bf16_launch(d_logits, d_idx, d_probs, batch, vocab, top_k, temperature, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_idx, d_idx, (size_t)batch * top_k * sizeof(int32_t), cudaMemcpyDeviceToHost); if (rc) fail("D2H idx", rc);
    rc = cudaMemcpy(h_probs, d_probs, (size_t)batch * top_k * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost); if (rc) fail("D2H probs", rc);

    // Emit JSON with output = {indices: [[...]], probabilities: [[...]]}
    // matching references/sampling.py.
    printf("{\n");
    printf("  \"schema_version\": 1,\n");
    printf("  \"slug\": \"%s\",\n", slug);
    printf("  \"shape_set\": \"correctness_shapes\",\n");
    printf("  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": {\n");
    printf("    \"indices\": [");
    for (int b = 0; b < batch; ++b) {
        if (b > 0) printf(",");
        printf("[");
        for (int k = 0; k < top_k; ++k) {
            if (k > 0) printf(",");
            printf("%" PRId32, h_idx[b * top_k + k]);
        }
        printf("]");
    }
    printf("],\n    \"probabilities\": [");
    for (int b = 0; b < batch; ++b) {
        if (b > 0) printf(",");
        printf("[");
        for (int k = 0; k < top_k; ++k) {
            if (k > 0) printf(",");
            float v = __bfloat162float(h_probs[b * top_k + k]);
            printf("%.6g", v);
        }
        printf("]");
    }
    printf("]\n  }\n}\n");

    cudaFree(d_logits); cudaFree(d_idx); cudaFree(d_probs);
    free(h_logits_f32); free(h_logits); free(h_idx); free(h_probs);
    return 0;
}
