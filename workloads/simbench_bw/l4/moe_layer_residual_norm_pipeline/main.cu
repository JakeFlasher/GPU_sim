// Host driver for moe_layer_residual_norm_pipeline. Mirrors
// `references/grouped_gemm.py::build_reference` for slug
// "moe_layer_residual_norm_pipeline":
//
//   logits     = seeded((tokens, experts),                 seed,        [-2.0, 2.0))
//   x          = seeded((tokens, hidden),                  seed + 1,    [-0.5, 0.5))
//   w_experts  = seeded((hidden, intermediate),            seed + 100 + e, [-0.1, 0.1))   per e in [0, experts)
//   gamma      = seeded((hidden,),                         seed + 200,  [-0.5, 0.5))
//
// Output JSON envelope's `output` field is the [tokens, hidden]
// post-RMSNorm tensor, matching the reference under (MIXED, atomic_add,
// grouped_gemm) tolerance.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <math.h>
#include <openssl/sha.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


extern "C" void moe_layer_residual_norm_pipeline_launch(
    const __nv_bfloat16 *d_logits,
    const __nv_bfloat16 *d_x,
    const __nv_bfloat16 *d_w_experts,
    const __nv_bfloat16 *d_gamma,
    __nv_bfloat16 *d_out,
    int tokens, int experts, int top_k, int hidden, int intermediate,
    cudaStream_t stream);


// MWC PRNG matching references/_common.py::deterministic_seeded_tensor.
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

// Double-precision MWC PRNG for byte-exact L4 checkpoint digests.
static void seed_tensor_double(double *flat, size_t n, uint32_t seed, double low, double high) {
    uint32_t state = (seed * 2654435761u + 1u);
    for (size_t i = 0; i < n; ++i) {
        state = state * 1103515245u + 12345u;
        double r = (double)state / 4294967296.0;
        flat[i] = low + (high - low) * r;
    }
}

static inline void fp32_to_bf16_bytes_from_double(double v, unsigned char out[2]) {
    union { float f; unsigned char b[4]; } u;
    u.f = (float)v;
    out[0] = u.b[2];
    out[1] = u.b[3];
}

static void sha256_hex_finalize(SHA256_CTX *ctx, char out[65]) {
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256_Final(digest, ctx);
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        out[2 * i]     = hex[digest[i] >> 4];
        out[2 * i + 1] = hex[digest[i] & 0xf];
    }
    out[64] = '\0';
}

static void sha256_bf16_flat_d(const double *flat, size_t n, char out[65]) {
    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    for (size_t i = 0; i < n; ++i) {
        unsigned char bf[2];
        fp32_to_bf16_bytes_from_double(flat[i], bf);
        SHA256_Update(&ctx, bf, 2);
    }
    sha256_hex_finalize(&ctx, out);
}

static void sha256_fp32_flat_d(const double *flat, size_t n, char out[65]) {
    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    for (size_t i = 0; i < n; ++i) {
        float fv = (float)flat[i];
        SHA256_Update(&ctx, (const unsigned char *)&fv, sizeof(float));
    }
    sha256_hex_finalize(&ctx, out);
}

static void sha256_int32_flat(const int *flat, size_t n, char out[65]) {
    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    SHA256_Update(&ctx, (const unsigned char *)flat, n * sizeof(int));
    sha256_hex_finalize(&ctx, out);
}


static void fail(const char *msg, int rc) {
    fprintf(stderr, "moe_layer_residual_norm_pipeline_run: %s (cuda rc=%d: %s)\n",
            msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}


int main(int argc, char **argv) {
    int tokens = 16;
    int experts = 4;
    int top_k = 2;
    int hidden = 32;
    int intermediate = 32;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    const char *slug = "moe_layer_residual_norm_pipeline";
    const char *emit_checkpoint = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--experts") == 0 && i + 1 < argc) experts = atoi(argv[++i]);
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) hidden = atoi(argv[++i]);
        else if (strcmp(argv[i], "--intermediate") == 0 && i + 1 < argc) intermediate = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else if (strcmp(argv[i], "--emit-l4-checkpoint") == 0 && i + 1 < argc) emit_checkpoint = argv[++i];
        else {
            fprintf(stderr, "moe_layer_residual_norm_pipeline_run: unknown arg %s\n", argv[i]);
            return 2;
        }
    }
    if (hidden != intermediate) {
        fprintf(stderr, "moe_layer_residual_norm_pipeline_run: hidden (%d) must equal intermediate (%d) for the residual add\n",
                hidden, intermediate);
        return 2;
    }

    // -------------------------------------------------------------------------
    // L4 checkpoint emission (R37). Mirrors
    // `references/grouped_gemm.py::build_l4_checkpoint`:
    //   - logits seeded at seed, [-2.0, 2.0)
    //   - x      seeded at seed + 1, [-0.5, 0.5)
    //   - per-token softmax + top-k routing with stable tie-breaking
    //     (lower index wins); weights re-normalized to sum to 1
    //   - permuted_tokens: stable sort of (eid, t, k_slot) triples;
    //     permuted_tokens[i] = x[t] of the i-th sorted triple
    //   - expert_offsets: prefix sum of per-expert counts (length experts+1)
    // All compute is FP64 to match the Python reference; FP32 cast happens
    // only at digest serialization time.
    // -------------------------------------------------------------------------
    if (emit_checkpoint != nullptr) {
        const size_t n_logits = (size_t)tokens * experts;
        const size_t n_x      = (size_t)tokens * hidden;

        double *logits = (double *)malloc(n_logits * sizeof(double));
        double *x      = (double *)malloc(n_x      * sizeof(double));
        seed_tensor_double(logits, n_logits, seed,     -2.0, 2.0);
        seed_tensor_double(x,      n_x,      seed + 1, -0.5, 0.5);

        // Per-token: softmax + top-k expert IDs + re-normalized weights.
        int    *expert_ids       = (int *)   malloc((size_t)tokens * top_k * sizeof(int));
        double *routing_weights  = (double *)malloc((size_t)tokens * top_k * sizeof(double));
        for (int t = 0; t < tokens; ++t) {
            const double *row = logits + (size_t)t * experts;
            double m = row[0];
            for (int e = 1; e < experts; ++e) if (row[e] > m) m = row[e];
            double total = 0.0;
            double *probs = (double *)malloc(experts * sizeof(double));
            for (int e = 0; e < experts; ++e) {
                probs[e] = exp(row[e] - m);
                total += probs[e];
            }
            for (int e = 0; e < experts; ++e) probs[e] /= total;

            // Top-k: pick highest unselected prob; tie-break to lower index.
            int    *idx = expert_ids       + (size_t)t * top_k;
            double *w   = routing_weights  + (size_t)t * top_k;
            for (int k = 0; k < top_k; ++k) idx[k] = -1;
            for (int k = 0; k < top_k; ++k) {
                int    best_i = -1;
                double best_p = -1.0;
                for (int e = 0; e < experts; ++e) {
                    bool already = false;
                    for (int j = 0; j < k; ++j) {
                        if (idx[j] == e) { already = true; break; }
                    }
                    if (already) continue;
                    if (probs[e] > best_p || (probs[e] == best_p && (best_i == -1 || e < best_i))) {
                        best_p = probs[e];
                        best_i = e;
                    }
                }
                idx[k] = best_i;
                w[k]   = best_p;
            }
            double wsum = 0.0;
            for (int k = 0; k < top_k; ++k) wsum += w[k];
            if (wsum > 0.0) for (int k = 0; k < top_k; ++k) w[k] /= wsum;
            free(probs);
        }

        // Per-expert counts + permuted_tokens via stable sort of
        // (eid, t, k_slot) triples (matches Python's sorted with key
        // (eid, t, k_slot) over the (t, k_slot) emission order).
        const int N_pairs = tokens * top_k;
        int *pair_eid = (int *)malloc(N_pairs * sizeof(int));
        int *pair_t   = (int *)malloc(N_pairs * sizeof(int));
        int *pair_k   = (int *)malloc(N_pairs * sizeof(int));
        int *counts   = (int *)calloc(experts, sizeof(int));
        for (int t = 0; t < tokens; ++t) {
            for (int k = 0; k < top_k; ++k) {
                int p = t * top_k + k;
                int e = expert_ids[t * top_k + k];
                pair_eid[p] = e;
                pair_t[p]   = t;
                pair_k[p]   = k;
                counts[e]++;
            }
        }
        // Stable sort by (eid, t, k_slot). Insertion sort: O(N^2) but N is
        // small at correctness scale (32) and acceptable at timing scale
        // (~16k for tokens=2048,top_k=8). For larger N a stable merge
        // sort would be needed; keep simple for now.
        for (int i = 1; i < N_pairs; ++i) {
            int e = pair_eid[i], tt = pair_t[i], kk = pair_k[i];
            int j = i - 1;
            while (j >= 0 && (
                pair_eid[j] > e ||
                (pair_eid[j] == e && pair_t[j] > tt) ||
                (pair_eid[j] == e && pair_t[j] == tt && pair_k[j] > kk)
            )) {
                pair_eid[j+1] = pair_eid[j];
                pair_t[j+1]   = pair_t[j];
                pair_k[j+1]   = pair_k[j];
                j--;
            }
            pair_eid[j+1] = e;
            pair_t[j+1]   = tt;
            pair_k[j+1]   = kk;
        }
        // permuted_tokens[i, :] = x[pair_t[i], :]
        double *permuted_tokens = (double *)malloc((size_t)N_pairs * hidden * sizeof(double));
        for (int i = 0; i < N_pairs; ++i) {
            const double *xrow = x + (size_t)pair_t[i] * hidden;
            double       *prow = permuted_tokens + (size_t)i * hidden;
            for (int d = 0; d < hidden; ++d) prow[d] = xrow[d];
        }
        // expert_offsets: prefix sum of counts; length experts + 1.
        int *expert_offsets = (int *)malloc((size_t)(experts + 1) * sizeof(int));
        expert_offsets[0] = 0;
        for (int e = 0; e < experts; ++e) expert_offsets[e + 1] = expert_offsets[e] + counts[e];

        char ids_digest[65], permuted_digest[65], weights_digest[65];
        sha256_int32_flat(expert_ids, (size_t)tokens * top_k, ids_digest);
        sha256_bf16_flat_d(permuted_tokens, (size_t)N_pairs * hidden, permuted_digest);
        sha256_fp32_flat_d(routing_weights, (size_t)tokens * top_k, weights_digest);

        FILE *fp = fopen(emit_checkpoint, "w");
        if (fp == NULL) {
            fprintf(stderr, "moe_layer_residual_norm_pipeline_run: cannot open %s for write\n", emit_checkpoint);
            return 1;
        }
        fprintf(fp, "{\n");
        fprintf(fp, "  \"schema_version\": 1,\n");
        fprintf(fp, "  \"slug\": \"moe_layer_residual_norm_pipeline\",\n");
        fprintf(fp, "  \"checkpoint_name\": \"router_to_grouped_gemm_boundary\",\n");
        fprintf(fp, "  \"expert_ids_per_token\": {\n");
        fprintf(fp, "    \"shape\": [%d, %d],\n", tokens, top_k);
        fprintf(fp, "    \"dtype\": \"int32\",\n");
        fprintf(fp, "    \"digest\": \"%s\"\n  },\n", ids_digest);
        fprintf(fp, "  \"permuted_tokens\": {\n");
        fprintf(fp, "    \"shape\": [%d, %d],\n", N_pairs, hidden);
        fprintf(fp, "    \"dtype\": \"bfloat16\",\n");
        fprintf(fp, "    \"digest\": \"%s\"\n  },\n", permuted_digest);
        fprintf(fp, "  \"expert_offsets\": {\n");
        fprintf(fp, "    \"shape\": [%d],\n", experts + 1);
        fprintf(fp, "    \"dtype\": \"int32\",\n");
        fprintf(fp, "    \"values\": [");
        for (int i = 0; i <= experts; ++i) {
            if (i > 0) fprintf(fp, ",");
            fprintf(fp, "%d", expert_offsets[i]);
        }
        fprintf(fp, "]\n  },\n");
        fprintf(fp, "  \"routing_weights\": {\n");
        fprintf(fp, "    \"shape\": [%d, %d],\n", tokens, top_k);
        fprintf(fp, "    \"dtype\": \"float32\",\n");
        fprintf(fp, "    \"digest\": \"%s\"\n  }\n", weights_digest);
        fprintf(fp, "}\n");
        fclose(fp);

        free(logits); free(x);
        free(expert_ids); free(routing_weights);
        free(pair_eid); free(pair_t); free(pair_k); free(counts);
        free(permuted_tokens); free(expert_offsets);
        return 0;
    }

    const size_t n_logits  = (size_t)tokens * experts;
    const size_t n_x       = (size_t)tokens * hidden;
    const size_t n_w       = (size_t)experts * hidden * intermediate;
    const size_t n_gamma   = (size_t)hidden;
    const size_t n_out     = (size_t)tokens * intermediate;

    float *h_logits_f32 = (float *)malloc(n_logits * sizeof(float));
    float *h_x_f32      = (float *)malloc(n_x      * sizeof(float));
    float *h_w_f32      = (float *)malloc(n_w      * sizeof(float));
    float *h_gamma_f32  = (float *)malloc(n_gamma  * sizeof(float));

    seed_tensor(h_logits_f32, n_logits, seed,     -2.0, 2.0);
    seed_tensor(h_x_f32,      n_x,      seed + 1, -0.5, 0.5);
    for (int e = 0; e < experts; ++e) {
        seed_tensor(h_w_f32 + (size_t)e * hidden * intermediate,
                    (size_t)hidden * intermediate,
                    seed + 100 + (uint32_t)e, -0.1, 0.1);
    }
    seed_tensor(h_gamma_f32,  n_gamma,  seed + 200, -0.5, 0.5);

    __nv_bfloat16 *h_logits_bf16 = (__nv_bfloat16 *)malloc(n_logits * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_x_bf16      = (__nv_bfloat16 *)malloc(n_x      * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_w_bf16      = (__nv_bfloat16 *)malloc(n_w      * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_gamma_bf16  = (__nv_bfloat16 *)malloc(n_gamma  * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_out_bf16    = (__nv_bfloat16 *)malloc(n_out    * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_logits; ++i) h_logits_bf16[i] = __float2bfloat16(h_logits_f32[i]);
    for (size_t i = 0; i < n_x;      ++i) h_x_bf16[i]      = __float2bfloat16(h_x_f32[i]);
    for (size_t i = 0; i < n_w;      ++i) h_w_bf16[i]      = __float2bfloat16(h_w_f32[i]);
    for (size_t i = 0; i < n_gamma;  ++i) h_gamma_bf16[i]  = __float2bfloat16(h_gamma_f32[i]);

    __nv_bfloat16 *d_logits = nullptr, *d_x = nullptr, *d_w = nullptr, *d_gamma = nullptr, *d_out = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_logits, n_logits * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_logits", rc);
    rc = cudaMalloc(&d_x,      n_x      * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_x",      rc);
    rc = cudaMalloc(&d_w,      n_w      * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_w",      rc);
    rc = cudaMalloc(&d_gamma,  n_gamma  * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_gamma",  rc);
    rc = cudaMalloc(&d_out,    n_out    * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_out",    rc);

    rc = cudaMemcpy(d_logits, h_logits_bf16, n_logits * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D logits", rc);
    rc = cudaMemcpy(d_x,      h_x_bf16,      n_x      * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D x",      rc);
    rc = cudaMemcpy(d_w,      h_w_bf16,      n_w      * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D w",      rc);
    rc = cudaMemcpy(d_gamma,  h_gamma_bf16,  n_gamma  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D gamma",  rc);

    moe_layer_residual_norm_pipeline_launch(d_logits, d_x, d_w, d_gamma, d_out,
                                             tokens, experts, top_k, hidden, intermediate, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("warm sync", rc);

    cudaEvent_t e_start, e_stop;
    cudaEventCreate(&e_start);
    cudaEventCreate(&e_stop);
    cudaEventRecord(e_start, 0);
    moe_layer_residual_norm_pipeline_launch(d_logits, d_x, d_w, d_gamma, d_out,
                                             tokens, experts, top_k, hidden, intermediate, 0);
    cudaEventRecord(e_stop, 0);
    rc = cudaEventSynchronize(e_stop); if (rc) fail("event sync", rc);
    float wall_ms = 0.0f;
    cudaEventElapsedTime(&wall_ms, e_start, e_stop);

    rc = cudaMemcpy(h_out_bf16, d_out, n_out * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost); if (rc) fail("D2H out", rc);

    printf("{\n");
    printf("  \"schema_version\": 1,\n");
    printf("  \"slug\": \"%s\",\n", slug);
    printf("  \"shape_set\": \"correctness_shapes\",\n");
    printf("  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"wall_time_ms\": %.6f,\n", wall_ms);
    printf("  \"output\": [");
    for (int t = 0; t < tokens; ++t) {
        if (t > 0) printf(",");
        printf("[");
        for (int c = 0; c < intermediate; ++c) {
            if (c > 0) printf(",");
            float v = __bfloat162float(h_out_bf16[(size_t)t * intermediate + c]);
            printf("%.6g", v);
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaEventDestroy(e_start);
    cudaEventDestroy(e_stop);
    cudaFree(d_logits); cudaFree(d_x); cudaFree(d_w); cudaFree(d_gamma); cudaFree(d_out);
    free(h_logits_f32); free(h_x_f32); free(h_w_f32); free(h_gamma_f32);
    free(h_logits_bf16); free(h_x_bf16); free(h_w_bf16); free(h_gamma_bf16); free(h_out_bf16);
    return 0;
}
