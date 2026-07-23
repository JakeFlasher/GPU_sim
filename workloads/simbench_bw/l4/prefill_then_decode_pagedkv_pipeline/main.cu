// Host driver for prefill_then_decode_pagedkv_pipeline. Mirrors
// `references/paged_attention.py::_prefill_then_decode_pipeline_reference`
// + `_prefill_then_decode_boundary_state`:
//
//   max_blocks_per_sequence = ceil(context / block_size)
//   num_blocks              = batch * max_blocks_per_sequence
//   block_table[s][b]       = s * max_blocks_per_sequence + b
//   kv_data_blocks          = seeded((num_blocks, block_size, kv_heads, head_dim),
//                                    seed + 500, [-0.5, 0.5))
//   q                       = seeded((batch, q_heads, head_dim),
//                                    seed,        [-0.5, 0.5))
//
// Output JSON `output` field is the [batch, q_heads, head_dim] decode
// SDPA result under GQA (kv_h = q_h % kv_heads).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <math.h>
#include <openssl/sha.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


extern "C" void prefill_then_decode_pagedkv_pipeline_launch(
    const __nv_bfloat16 *d_q,
    const __nv_bfloat16 *d_kv_data_blocks,
    const int32_t       *d_block_table,
    __nv_bfloat16 *d_out,
    int batch, int q_heads, int kv_heads,
    int context, int head_dim, int block_size,
    int max_blocks_per_sequence,
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

// Same MWC PRNG, but stores as `double` for byte-exact checkpoint
// digest match with the Python reference. See speculative_decode
// main.cu for the rationale comment.
static void seed_tensor_double(double *flat, size_t n, uint32_t seed, double low, double high) {
    uint32_t state = (seed * 2654435761u + 1u);
    for (size_t i = 0; i < n; ++i) {
        state = state * 1103515245u + 12345u;
        double r = (double)state / 4294967296.0;
        flat[i] = low + (high - low) * r;
    }
}

// R37 L4 checkpoint digest helpers (OpenSSL SHA-256).
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

// Build the canonical RoPE cos/sin cache for prefill_then_decode's
// boundary state. Mirrors `references/paged_attention.py::_rope_cos_sin_cache`.
// Output shape: [2, max_position, head_dim] (axis 0 = cos / sin).
static void build_rope_cos_sin_cache(double *out, int max_position, int head_dim, double base) {
    const int half = head_dim / 2;
    // First zero the buffer.
    for (size_t i = 0; i < (size_t)2 * max_position * head_dim; ++i) out[i] = 0.0;
    for (int k = 0; k < half; ++k) {
        double inv_freq = 1.0 / pow(base, 2.0 * k / head_dim);
        for (int p = 0; p < max_position; ++p) {
            double angle = (double)p * inv_freq;
            double cos_v = cos(angle);
            double sin_v = sin(angle);
            // axis 0 = cos, axis 1 = sin
            out[((size_t)0 * max_position + p) * head_dim + k]        = cos_v;
            out[((size_t)0 * max_position + p) * head_dim + k + half] = cos_v;
            out[((size_t)1 * max_position + p) * head_dim + k]        = sin_v;
            out[((size_t)1 * max_position + p) * head_dim + k + half] = sin_v;
        }
    }
}


static void fail(const char *msg, int rc) {
    fprintf(stderr, "prefill_then_decode_pagedkv_pipeline_run: %s (cuda rc=%d: %s)\n",
            msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}


int main(int argc, char **argv) {
    int batch = 2;
    int context = 32;
    int head_dim = 32;
    int q_heads = 4;
    int kv_heads = 2;
    int block_size = 16;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    const char *slug = "prefill_then_decode_pagedkv_pipeline";
    const char *emit_checkpoint = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) batch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--context") == 0 && i + 1 < argc) context = atoi(argv[++i]);
        else if (strcmp(argv[i], "--head-dim") == 0 && i + 1 < argc) head_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--q-heads") == 0 && i + 1 < argc) q_heads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--kv-heads") == 0 && i + 1 < argc) kv_heads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) block_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else if (strcmp(argv[i], "--emit-l4-checkpoint") == 0 && i + 1 < argc) emit_checkpoint = argv[++i];
        else {
            fprintf(stderr, "prefill_then_decode_pagedkv_pipeline_run: unknown arg %s\n", argv[i]);
            return 2;
        }
    }
    if (head_dim % 2 != 0) {
        fprintf(stderr, "prefill_then_decode_pagedkv_pipeline_run: head_dim must be even (got %d)\n", head_dim);
        return 2;
    }

    const int max_blocks_per_sequence = (context + block_size - 1) / block_size;
    const int num_blocks = batch * max_blocks_per_sequence;

    // -------------------------------------------------------------------------
    // L4 checkpoint emission (R37). The boundary state for
    // prefill_then_decode is the input to the decode phase: kv_data_blocks,
    // kv_block_table, seq_lengths, rope_cos_sin_cache. The binary owns all
    // of these (block table built deterministically; KV blocks seeded at
    // seed+500; seq_lengths uniform context; RoPE cache deterministic from
    // head_dim+context). Writes the schema-conforming JSON checkpoint and
    // exits, no kernel launch required.
    // -------------------------------------------------------------------------
    if (emit_checkpoint != nullptr) {
        const size_t n_kv_d = (size_t)num_blocks * block_size * kv_heads * head_dim;
        const size_t n_rope = (size_t)2 * context * head_dim;
        const size_t n_bt   = (size_t)batch * max_blocks_per_sequence;

        double *kv_data = (double *)malloc(n_kv_d * sizeof(double));
        double *rope    = (double *)malloc(n_rope * sizeof(double));
        int    *block_table = (int *)malloc(n_bt * sizeof(int));
        int    *seq_lengths = (int *)malloc((size_t)batch * sizeof(int));

        seed_tensor_double(kv_data, n_kv_d, seed + 500u, -0.5, 0.5);
        for (int s = 0; s < batch; ++s) {
            seq_lengths[s] = context;
            for (int b = 0; b < max_blocks_per_sequence; ++b) {
                block_table[s * max_blocks_per_sequence + b] = s * max_blocks_per_sequence + b;
            }
        }
        build_rope_cos_sin_cache(rope, context, head_dim, 10000.0);

        char bt_digest[65], kv_digest[65], rope_digest[65];
        sha256_int32_flat(block_table, n_bt, bt_digest);
        sha256_bf16_flat_d(kv_data, n_kv_d, kv_digest);
        sha256_fp32_flat_d(rope,    n_rope, rope_digest);

        FILE *fp = fopen(emit_checkpoint, "w");
        if (fp == NULL) {
            fprintf(stderr, "prefill_then_decode_pagedkv_pipeline_run: cannot open %s for write\n", emit_checkpoint);
            return 1;
        }
        fprintf(fp, "{\n");
        fprintf(fp, "  \"schema_version\": 1,\n");
        fprintf(fp, "  \"slug\": \"prefill_then_decode_pagedkv_pipeline\",\n");
        fprintf(fp, "  \"checkpoint_name\": \"prefill_to_decode_kv_boundary\",\n");
        fprintf(fp, "  \"kv_block_table\": {\n");
        fprintf(fp, "    \"shape\": [%d, %d],\n", batch, max_blocks_per_sequence);
        fprintf(fp, "    \"dtype\": \"int32\",\n");
        fprintf(fp, "    \"digest\": \"%s\"\n  },\n", bt_digest);
        fprintf(fp, "  \"kv_data_blocks\": {\n");
        fprintf(fp, "    \"shape\": [%d, %d, %d, %d],\n", num_blocks, block_size, kv_heads, head_dim);
        fprintf(fp, "    \"dtype\": \"bfloat16\",\n");
        fprintf(fp, "    \"digest\": \"%s\"\n  },\n", kv_digest);
        fprintf(fp, "  \"seq_lengths\": {\n");
        fprintf(fp, "    \"shape\": [%d],\n", batch);
        fprintf(fp, "    \"dtype\": \"int32\",\n");
        fprintf(fp, "    \"values\": [");
        for (int b = 0; b < batch; ++b) {
            if (b > 0) fprintf(fp, ",");
            fprintf(fp, "%d", seq_lengths[b]);
        }
        fprintf(fp, "]\n  },\n");
        fprintf(fp, "  \"rope_cos_sin_cache\": {\n");
        fprintf(fp, "    \"shape\": [2, %d, %d],\n", context, head_dim);
        fprintf(fp, "    \"dtype\": \"float32\",\n");
        fprintf(fp, "    \"rope_kind\": \"standard\",\n");
        fprintf(fp, "    \"digest\": \"%s\"\n  }\n", rope_digest);
        fprintf(fp, "}\n");
        fclose(fp);

        free(kv_data); free(rope); free(block_table); free(seq_lengths);
        return 0;
    }

    const size_t n_q  = (size_t)batch * q_heads * head_dim;
    const size_t n_kv = (size_t)num_blocks * block_size * kv_heads * head_dim;
    const size_t n_bt = (size_t)batch * max_blocks_per_sequence;
    const size_t n_out = n_q;  // same shape as q

    float *h_q_f32  = (float *)malloc(n_q  * sizeof(float));
    float *h_kv_f32 = (float *)malloc(n_kv * sizeof(float));
    seed_tensor(h_q_f32,  n_q,  seed,        -0.5, 0.5);
    seed_tensor(h_kv_f32, n_kv, seed + 500u, -0.5, 0.5);

    __nv_bfloat16 *h_q_bf16  = (__nv_bfloat16 *)malloc(n_q  * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_kv_bf16 = (__nv_bfloat16 *)malloc(n_kv * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_out_bf16 = (__nv_bfloat16 *)malloc(n_out * sizeof(__nv_bfloat16));
    int32_t *h_bt = (int32_t *)malloc(n_bt * sizeof(int32_t));
    for (size_t i = 0; i < n_q;  ++i) h_q_bf16[i]  = __float2bfloat16(h_q_f32[i]);
    for (size_t i = 0; i < n_kv; ++i) h_kv_bf16[i] = __float2bfloat16(h_kv_f32[i]);
    for (int s = 0; s < batch; ++s) {
        for (int b = 0; b < max_blocks_per_sequence; ++b) {
            h_bt[s * max_blocks_per_sequence + b] = s * max_blocks_per_sequence + b;
        }
    }

    __nv_bfloat16 *d_q = nullptr, *d_kv = nullptr, *d_out = nullptr;
    int32_t *d_bt = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_q,  n_q  * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_q",  rc);
    rc = cudaMalloc(&d_kv, n_kv * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_kv", rc);
    rc = cudaMalloc(&d_bt, n_bt * sizeof(int32_t));       if (rc) fail("cudaMalloc d_bt", rc);
    rc = cudaMalloc(&d_out, n_out * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_out", rc);

    rc = cudaMemcpy(d_q,  h_q_bf16,  n_q  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D q",  rc);
    rc = cudaMemcpy(d_kv, h_kv_bf16, n_kv * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D kv", rc);
    rc = cudaMemcpy(d_bt, h_bt,      n_bt * sizeof(int32_t),       cudaMemcpyHostToDevice); if (rc) fail("H2D bt", rc);

    const float scale = 1.0f / sqrtf((float)head_dim);

    prefill_then_decode_pagedkv_pipeline_launch(d_q, d_kv, d_bt, d_out,
        batch, q_heads, kv_heads, context, head_dim, block_size,
        max_blocks_per_sequence, scale, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("warm sync", rc);

    cudaEvent_t e_start, e_stop;
    cudaEventCreate(&e_start);
    cudaEventCreate(&e_stop);
    cudaEventRecord(e_start, 0);
    prefill_then_decode_pagedkv_pipeline_launch(d_q, d_kv, d_bt, d_out,
        batch, q_heads, kv_heads, context, head_dim, block_size,
        max_blocks_per_sequence, scale, 0);
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
    for (int b = 0; b < batch; ++b) {
        if (b > 0) printf(",");
        printf("[");
        for (int h = 0; h < q_heads; ++h) {
            if (h > 0) printf(",");
            printf("[");
            for (int d = 0; d < head_dim; ++d) {
                if (d > 0) printf(",");
                float v = __bfloat162float(h_out_bf16[((size_t)b * q_heads + h) * head_dim + d]);
                printf("%.6g", v);
            }
            printf("]");
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaEventDestroy(e_start);
    cudaEventDestroy(e_stop);
    cudaFree(d_q); cudaFree(d_kv); cudaFree(d_bt); cudaFree(d_out);
    free(h_q_f32); free(h_kv_f32); free(h_q_bf16); free(h_kv_bf16); free(h_out_bf16); free(h_bt);
    return 0;
}
