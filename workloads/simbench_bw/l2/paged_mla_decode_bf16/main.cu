// Host driver for paged_mla_decode_bf16. Seeds q / k / v via the same PRNG
// as references/_common.py::deterministic_seeded_tensor so the GPU output
// can be compared against the Python reference under the BF16 attention
// tolerance.
//
// Matches references/paged_attention.py::build_reference for
// slug == "paged_mla_decode_bf16":
//   q = seeded((batch, q_heads, head_dim),              seed,       [-0.5, 0.5))
//   k = seeded((batch, kv_heads, context, head_dim),    seed + 100, [-0.5, 0.5))
//   v = seeded((batch, kv_heads, context, head_dim),    seed + 200, [-0.5, 0.5))
//
// The correctness shape declares `head_dim_ckv` (the LoRA-compressed
// key/value head dim under which MLA decodes); the simplified reference
// treats that as the single head_dim across q / k / v.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void paged_mla_decode_bf16_launch(
    const __nv_bfloat16 *d_q,
    const __nv_bfloat16 *d_k,
    const __nv_bfloat16 *d_v,
    __nv_bfloat16 *d_out,
    int batch, int q_heads, int kv_heads, int context, int head_dim,
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
    fprintf(stderr, "paged_mla_decode_bf16_run: %s (cuda rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}


int main(int argc, char **argv) {
    int batch = 1;
    int context = 512;
    int head_dim = 256;
    int q_heads = 8;
    int kv_heads = 1;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    const char *slug = "paged_mla_decode_bf16";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) batch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--context") == 0 && i + 1 < argc) context = atoi(argv[++i]);
        else if (strcmp(argv[i], "--head-dim") == 0 && i + 1 < argc) head_dim = atoi(argv[++i]);
        // Shape axes name the MLA-compressed KV head dim as head_dim_ckv;
        // accept that too so the orchestrator's auto-generated CLI flags
        // from shape axes work without a translation table.
        else if (strcmp(argv[i], "--head-dim-ckv") == 0 && i + 1 < argc) head_dim = atoi(argv[++i]);
        // The position-encoding head dim (head_dim_kpe) is not used by the
        // simplified reference implementation; accept and discard.
        else if (strcmp(argv[i], "--head-dim-kpe") == 0 && i + 1 < argc) { ++i; }
        else if (strcmp(argv[i], "--q-heads") == 0 && i + 1 < argc) q_heads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--kv-heads") == 0 && i + 1 < argc) kv_heads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else {
            fprintf(stderr, "paged_mla_decode_bf16_run: unknown arg %s\n", argv[i]);
            return 2;
        }
    }

    const size_t n_q = (size_t)batch * q_heads * head_dim;
    const size_t n_kv = (size_t)batch * kv_heads * context * head_dim;
    const size_t n_out = (size_t)batch * q_heads * head_dim;
    const float scale = 1.0f / sqrtf((float)head_dim);

    float *h_q_f32 = (float *)malloc(n_q  * sizeof(float));
    float *h_k_f32 = (float *)malloc(n_kv * sizeof(float));
    float *h_v_f32 = (float *)malloc(n_kv * sizeof(float));
    seed_tensor(h_q_f32, n_q,  seed,       -0.5, 0.5);
    seed_tensor(h_k_f32, n_kv, seed + 100, -0.5, 0.5);
    seed_tensor(h_v_f32, n_kv, seed + 200, -0.5, 0.5);

    __nv_bfloat16 *h_q = (__nv_bfloat16 *)malloc(n_q * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_k = (__nv_bfloat16 *)malloc(n_kv * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_v = (__nv_bfloat16 *)malloc(n_kv * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_out = (__nv_bfloat16 *)malloc(n_out * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_q;  ++i) h_q[i] = __float2bfloat16(h_q_f32[i]);
    for (size_t i = 0; i < n_kv; ++i) h_k[i] = __float2bfloat16(h_k_f32[i]);
    for (size_t i = 0; i < n_kv; ++i) h_v[i] = __float2bfloat16(h_v_f32[i]);

    __nv_bfloat16 *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_out = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_q,  n_q  * sizeof(__nv_bfloat16));   if (rc) fail("cudaMalloc d_q", rc);
    rc = cudaMalloc(&d_k,  n_kv * sizeof(__nv_bfloat16));   if (rc) fail("cudaMalloc d_k", rc);
    rc = cudaMalloc(&d_v,  n_kv * sizeof(__nv_bfloat16));   if (rc) fail("cudaMalloc d_v", rc);
    rc = cudaMalloc(&d_out, n_out * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_out", rc);
    rc = cudaMemcpy(d_q, h_q, n_q  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D q", rc);
    rc = cudaMemcpy(d_k, h_k, n_kv * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D k", rc);
    rc = cudaMemcpy(d_v, h_v, n_kv * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D v", rc);

    paged_mla_decode_bf16_launch(d_q, d_k, d_v, d_out, batch, q_heads, kv_heads, context, head_dim, scale, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);

    rc = cudaMemcpy(h_out, d_out, n_out * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost); if (rc) fail("D2H out", rc);

    printf("{\n");
    printf("  \"schema_version\": 1,\n");
    printf("  \"slug\": \"%s\",\n", slug);
    printf("  \"shape_set\": \"correctness_shapes\",\n");
    printf("  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int ba = 0; ba < batch; ++ba) {
        if (ba > 0) printf(",");
        printf("[");
        for (int h = 0; h < q_heads; ++h) {
            if (h > 0) printf(",");
            printf("[");
            for (int d = 0; d < head_dim; ++d) {
                if (d > 0) printf(",");
                float v = __bfloat162float(h_out[((ba * q_heads) + h) * head_dim + d]);
                printf("%.6g", v);
            }
            printf("]");
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
    free(h_q_f32); free(h_k_f32); free(h_v_f32);
    free(h_q); free(h_k); free(h_v); free(h_out);
    return 0;
}
