// Host driver for speculative_decode_draft_verify_pipeline. Seeds
// draft_logits at seed+700 and target_logits at seed+900 to mirror
// references/attention.py::_speculative_decode_pipeline_reference,
// invokes the kernel, and emits a JSON envelope shaped like the
// reference output: [batch, draft_length, vocab].
//
// Matches references/attention.py for slug ==
// "speculative_decode_draft_verify_pipeline":
//   draft_logits  = seeded((B, S, V), seed + 700, [-2.0, 2.0))
//   target_logits = seeded((B, S, V), seed + 900, [-2.0, 2.0))
//   out[b][s][v]  = target_probs[v] * min(1, target_probs[v] / max(draft_probs[v], 1e-12))

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <math.h>
#include <openssl/sha.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


extern "C" void speculative_decode_draft_verify_pipeline_launch(
    const __nv_bfloat16 *d_draft_logits,
    const __nv_bfloat16 *d_target_logits,
    __nv_bfloat16 *d_out,
    int batch, int dlen, int vocab,
    cudaStream_t stream);


// -----------------------------------------------------------------------------
// L4 checkpoint emission helpers (R37). Mirror the canonical byte
// representation in references/_common.py: BF16 = high 16 bits of FP32
// little-endian, FP32 = struct.pack('<f'), INT32 = struct.pack('<i'). The
// digests are computed via OpenSSL SHA-256 (linked through -lcrypto) so the
// binary's checkpoint emission is self-contained and matches the Python
// reference's `build_l4_checkpoint()` byte-for-byte.
// -----------------------------------------------------------------------------

static inline void fp32_to_bf16_bytes(float v, unsigned char out[2]) {
    union { float f; unsigned char b[4]; } u;
    u.f = v;
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

static void sha256_bf16_flat(const float *flat, size_t n, char out[65]) {
    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    for (size_t i = 0; i < n; ++i) {
        unsigned char bf[2];
        fp32_to_bf16_bytes(flat[i], bf);
        SHA256_Update(&ctx, bf, 2);
    }
    sha256_hex_finalize(&ctx, out);
}

static void sha256_fp32_flat(const float *flat, size_t n, char out[65]) {
    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    SHA256_Update(&ctx, (const unsigned char *)flat, n * sizeof(float));
    sha256_hex_finalize(&ctx, out);
}


// MWC PRNG matching references/_common.py::deterministic_seeded_tensor.
// init: state = seed * 2654435761 + 1; step: state = state*1103515245 + 12345.
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

// Same MWC PRNG, but stores as `double` so the checkpoint-emission path
// can match the Python reference's double-precision computation. The
// reference's `deterministic_seeded_tensor` returns Python floats
// (FP64); only the final `struct.pack('<f', float(v))` casts to FP32
// for digest serialization. Mirroring this is required for byte-exact
// digest equality of any FIELD that depends on multi-step double math
// (softmax, matmul, RMSNorm) — single-step casts (BF16-of-seeded-input)
// are byte-exact regardless because BF16 = high 16 bits of FP32 and
// `(double) -> FP32` and `(double) -> FP32 -> (double) -> FP32` give
// the same FP32.
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


static void fail(const char *msg, int rc) {
    fprintf(stderr, "speculative_decode_draft_verify_pipeline_run: %s (cuda rc=%d: %s)\n",
            msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}


int main(int argc, char **argv) {
    int batch = 2;
    int draft_length = 4;
    int vocab = 32;
    int num_layers = 2;
    int num_kv_heads = 2;
    int head_dim = 16;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    const char *slug = "speculative_decode_draft_verify_pipeline";
    // R37: when set, write the canonical L4 checkpoint
    // (matching schemas/l4_checkpoints/<slug>.schema.json) to this
    // path, then exit. Mutually exclusive with the kernel correctness
    // path: the binary either runs the verifier kernel and emits the
    // output JSON (default), or emits the checkpoint and exits 0.
    const char *emit_checkpoint = nullptr;

    // Axes consumed by the checkpoint-emission path: batch /
    // draft_length / vocab / num_layers / num_kv_heads / head_dim.
    // The kernel itself only depends on (batch, draft_length, vocab),
    // but the schema requires shape arrays for the draft attention
    // state. heads / seq are accepted for shape-file CLI compatibility.
    int discard;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) batch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--draft-length") == 0 && i + 1 < argc) draft_length = atoi(argv[++i]);
        else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) vocab = atoi(argv[++i]);
        else if (strcmp(argv[i], "--num-layers") == 0 && i + 1 < argc) num_layers = atoi(argv[++i]);
        else if (strcmp(argv[i], "--num-kv-heads") == 0 && i + 1 < argc) num_kv_heads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--head-dim") == 0 && i + 1 < argc) head_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--heads") == 0 && i + 1 < argc) discard = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc) discard = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else if (strcmp(argv[i], "--emit-l4-checkpoint") == 0 && i + 1 < argc) emit_checkpoint = argv[++i];
        else {
            fprintf(stderr, "speculative_decode_draft_verify_pipeline_run: unknown arg %s\n", argv[i]);
            return 2;
        }
    }
    (void)discard;

    // -------------------------------------------------------------------------
    // L4 checkpoint emission path (R37). When --emit-l4-checkpoint is
    // set, the binary computes the canonical draft-to-verifier boundary state
    // from its own seeded inputs (no kernel run required because the verifier
    // checkpoint depends only on the draft side) and writes a schema-conforming
    // JSON. `_emit_l4_component_strict()` invokes the binary with this flag
    // before the correctness kernel run; the Python reference's
    // `build_l4_checkpoint()` is the comparison oracle, not the artifact source.
    // -------------------------------------------------------------------------
    if (emit_checkpoint != nullptr) {
        const size_t n_logits = (size_t)batch * draft_length * vocab;
        const size_t n_att    = (size_t)num_layers * num_kv_heads * draft_length * head_dim;

        // Use double precision throughout the emit path so the multi-step
        // softmax computation reproduces the Python reference's
        // `_softmax(row)` byte-for-byte. The final cast to FP32 happens
        // inside `sha256_*_flat_d` only at digest serialization time
        // (matching `struct.pack('<f', float(v))`).
        double *draft_logits = (double *)malloc(n_logits * sizeof(double));
        double *draft_probs  = (double *)malloc(n_logits * sizeof(double));
        int    *draft_tokens = (int *)   malloc((size_t)batch * draft_length * sizeof(int));
        double *draft_att    = (double *)malloc(n_att    * sizeof(double));
        seed_tensor_double(draft_logits, n_logits, seed + 700, -2.0, 2.0);
        seed_tensor_double(draft_att,    n_att,    seed + 800, -0.5, 0.5);

        // Per-row stable softmax + argmax over vocab. Mirrors
        // references/attention.py::_speculative_decode_boundary_state's
        // _softmax + argmax exactly: FP64 max, FP64 exp(x - m), FP64 sum,
        // FP64 division.
        for (int b = 0; b < batch; ++b) {
            for (int s = 0; s < draft_length; ++s) {
                double *row = draft_logits + ((size_t)b * draft_length + s) * vocab;
                double m = row[0];
                int    am = 0;
                for (int v = 1; v < vocab; ++v) {
                    if (row[v] > m) { m = row[v]; am = v; }
                }
                draft_tokens[b * draft_length + s] = am;
                double total = 0.0;
                double *prob_row = draft_probs + ((size_t)b * draft_length + s) * vocab;
                for (int v = 0; v < vocab; ++v) {
                    prob_row[v] = exp(row[v] - m);
                    total += prob_row[v];
                }
                for (int v = 0; v < vocab; ++v) {
                    prob_row[v] = prob_row[v] / total;
                }
            }
        }

        char d_logits_digest[65], d_probs_digest[65], d_att_digest[65];
        sha256_bf16_flat_d(draft_logits, n_logits, d_logits_digest);
        sha256_fp32_flat_d(draft_probs,  n_logits, d_probs_digest);
        sha256_bf16_flat_d(draft_att,    n_att,    d_att_digest);

        FILE *fp = fopen(emit_checkpoint, "w");
        if (fp == NULL) {
            fprintf(stderr, "speculative_decode_draft_verify_pipeline_run: cannot open %s for write\n", emit_checkpoint);
            return 1;
        }
        fprintf(fp, "{\n");
        fprintf(fp, "  \"schema_version\": 1,\n");
        fprintf(fp, "  \"slug\": \"speculative_decode_draft_verify_pipeline\",\n");
        fprintf(fp, "  \"checkpoint_name\": \"draft_to_verifier_boundary\",\n");
        fprintf(fp, "  \"draft_tokens\": {\n");
        fprintf(fp, "    \"shape\": [%d, %d],\n", batch, draft_length);
        fprintf(fp, "    \"dtype\": \"int32\",\n");
        fprintf(fp, "    \"values\": [");
        for (int i = 0; i < batch * draft_length; ++i) {
            if (i > 0) fprintf(fp, ",");
            fprintf(fp, "%d", draft_tokens[i]);
        }
        fprintf(fp, "]\n  },\n");
        fprintf(fp, "  \"draft_logits\": {\n");
        fprintf(fp, "    \"shape\": [%d, %d, %d],\n", batch, draft_length, vocab);
        fprintf(fp, "    \"dtype\": \"bfloat16\",\n");
        fprintf(fp, "    \"digest\": \"%s\"\n  },\n", d_logits_digest);
        fprintf(fp, "  \"draft_probabilities\": {\n");
        fprintf(fp, "    \"shape\": [%d, %d, %d],\n", batch, draft_length, vocab);
        fprintf(fp, "    \"dtype\": \"float32\",\n");
        fprintf(fp, "    \"digest\": \"%s\"\n  },\n", d_probs_digest);
        fprintf(fp, "  \"draft_attention_state_digest\": {\n");
        fprintf(fp, "    \"shape\": [%d, %d, %d, %d],\n", num_layers, num_kv_heads, draft_length, head_dim);
        fprintf(fp, "    \"dtype\": \"bfloat16\",\n");
        fprintf(fp, "    \"digest\": \"%s\"\n  }\n", d_att_digest);
        fprintf(fp, "}\n");
        fclose(fp);

        free(draft_logits); free(draft_probs); free(draft_tokens); free(draft_att);
        return 0;
    }

    const size_t n = (size_t)batch * draft_length * vocab;

    float *h_d_f32 = (float *)malloc(n * sizeof(float));
    float *h_t_f32 = (float *)malloc(n * sizeof(float));
    seed_tensor(h_d_f32, n, seed + 700, -2.0, 2.0);
    seed_tensor(h_t_f32, n, seed + 900, -2.0, 2.0);

    __nv_bfloat16 *h_draft_bf16  = (__nv_bfloat16 *)malloc(n * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_target_bf16 = (__nv_bfloat16 *)malloc(n * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_out_bf16    = (__nv_bfloat16 *)malloc(n * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n; ++i) {
        h_draft_bf16[i]  = __float2bfloat16(h_d_f32[i]);
        h_target_bf16[i] = __float2bfloat16(h_t_f32[i]);
    }

    __nv_bfloat16 *d_draft = nullptr, *d_target = nullptr, *d_out = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_draft,  n * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_draft",  rc);
    rc = cudaMalloc(&d_target, n * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_target", rc);
    rc = cudaMalloc(&d_out,    n * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_out",    rc);

    rc = cudaMemcpy(d_draft,  h_draft_bf16,  n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D draft",  rc);
    rc = cudaMemcpy(d_target, h_target_bf16, n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D target", rc);

    // Warm dispatch + timed dispatch.
    speculative_decode_draft_verify_pipeline_launch(d_draft, d_target, d_out, batch, draft_length, vocab, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("warm sync", rc);

    cudaEvent_t e_start, e_stop;
    cudaEventCreate(&e_start);
    cudaEventCreate(&e_stop);
    cudaEventRecord(e_start, 0);
    speculative_decode_draft_verify_pipeline_launch(d_draft, d_target, d_out, batch, draft_length, vocab, 0);
    cudaEventRecord(e_stop, 0);
    rc = cudaEventSynchronize(e_stop); if (rc) fail("event sync", rc);
    float wall_ms = 0.0f;
    cudaEventElapsedTime(&wall_ms, e_start, e_stop);

    rc = cudaMemcpy(h_out_bf16, d_out, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost); if (rc) fail("D2H out", rc);

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
        for (int s = 0; s < draft_length; ++s) {
            if (s > 0) printf(",");
            printf("[");
            for (int v = 0; v < vocab; ++v) {
                if (v > 0) printf(",");
                float val = __bfloat162float(h_out_bf16[((size_t)b * draft_length + s) * vocab + v]);
                printf("%.6g", val);
            }
            printf("]");
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaEventDestroy(e_start);
    cudaEventDestroy(e_stop);
    cudaFree(d_draft); cudaFree(d_target); cudaFree(d_out);
    free(h_d_f32); free(h_t_f32);
    free(h_draft_bf16); free(h_target_bf16); free(h_out_bf16);
    return 0;
}
