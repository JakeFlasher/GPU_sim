// Host driver for dram_stream_stride_mix.

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void dram_stream_stride_mix_launch(const float *d_in, float *d_out, float *d_sink,
                                               int warps, int accesses, int stride_classes,
                                               cudaStream_t stream);

static inline uint32_t next_u32(uint32_t *s) { *s = (*s) * 1103515245u + 12345u; return *s; }
static void seed_tensor(float *flat, size_t n, uint32_t seed, double lo, double hi) {
    uint32_t st = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) {
        double r = (double)next_u32(&st) / 4294967296.0;
        flat[i] = (float)(lo + (hi - lo) * r);
    }
}
static void fail(const char *msg, int rc) {
    fprintf(stderr, "dram_stream_stride_mix_run: %s (rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}

// ---------------------------------------------------------------------
// Inline SHA-256 implementation (public-domain reference; FIPS 180-4).
// Used by `--output-mode sha256_f32le` to compute the digest of the
// FP32 little-endian byte stream of the output tensor in row-major
// (warps, accesses, stride_classes) order. The implementation processes
// the byte stream via a streaming Update / Final API so multi-MiB
// outputs do not need an intermediate buffer.
// ---------------------------------------------------------------------

typedef struct {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t buf[64];
    size_t buflen;
} sha256_ctx_t;

static const uint32_t SHA256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

static inline uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

static void sha256_compress(sha256_ctx_t *c, const uint8_t *block) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = ((uint32_t)block[i*4] << 24) | ((uint32_t)block[i*4+1] << 16) |
               ((uint32_t)block[i*4+2] << 8) | (uint32_t)block[i*4+3];
    }
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a = c->state[0], b = c->state[1], cc = c->state[2], d = c->state[3];
    uint32_t e = c->state[4], f = c->state[5], g = c->state[6], h = c->state[7];
    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + SHA256_K[i] + w[i];
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & cc) ^ (b & cc);
        uint32_t t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = cc; cc = b; b = a; a = t1 + t2;
    }
    c->state[0] += a; c->state[1] += b; c->state[2] += cc; c->state[3] += d;
    c->state[4] += e; c->state[5] += f; c->state[6] += g; c->state[7] += h;
}

static void sha256_init(sha256_ctx_t *c) {
    c->state[0] = 0x6a09e667u; c->state[1] = 0xbb67ae85u;
    c->state[2] = 0x3c6ef372u; c->state[3] = 0xa54ff53au;
    c->state[4] = 0x510e527fu; c->state[5] = 0x9b05688cu;
    c->state[6] = 0x1f83d9abu; c->state[7] = 0x5be0cd19u;
    c->bitlen = 0;
    c->buflen = 0;
}

static void sha256_update(sha256_ctx_t *c, const uint8_t *data, size_t len) {
    c->bitlen += (uint64_t)len * 8u;
    while (len > 0) {
        size_t take = 64 - c->buflen;
        if (take > len) take = len;
        memcpy(c->buf + c->buflen, data, take);
        c->buflen += take;
        data += take;
        len -= take;
        if (c->buflen == 64) {
            sha256_compress(c, c->buf);
            c->buflen = 0;
        }
    }
}

static void sha256_final(sha256_ctx_t *c, uint8_t out[32]) {
    c->buf[c->buflen++] = 0x80u;
    if (c->buflen > 56) {
        while (c->buflen < 64) c->buf[c->buflen++] = 0u;
        sha256_compress(c, c->buf);
        c->buflen = 0;
    }
    while (c->buflen < 56) c->buf[c->buflen++] = 0u;
    for (int i = 7; i >= 0; --i) c->buf[c->buflen++] = (uint8_t)(c->bitlen >> (i * 8));
    sha256_compress(c, c->buf);
    for (int i = 0; i < 8; ++i) {
        out[i*4]   = (uint8_t)(c->state[i] >> 24);
        out[i*4+1] = (uint8_t)(c->state[i] >> 16);
        out[i*4+2] = (uint8_t)(c->state[i] >> 8);
        out[i*4+3] = (uint8_t)(c->state[i]);
    }
}

static void hex_encode(const uint8_t in[32], char out[65]) {
    static const char *hex = "0123456789abcdef";
    for (int i = 0; i < 32; ++i) {
        out[i*2]     = hex[(in[i] >> 4) & 0xf];
        out[i*2 + 1] = hex[in[i] & 0xf];
    }
    out[64] = '\0';
}

int main(int argc, char **argv) {
    int warps = 4, accesses = 16, stride_classes = 2;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    const char *output_mode = "json";
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--warps")          == 0 && i + 1 < argc) warps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--accesses")       == 0 && i + 1 < argc) accesses = atoi(argv[++i]);
        else if (strcmp(argv[i], "--stride-classes") == 0 && i + 1 < argc) stride_classes = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed")           == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point")    == 0 && i + 1 < argc) shape_point = argv[++i];
        else if (strcmp(argv[i], "--output-mode")    == 0 && i + 1 < argc) output_mode = argv[++i];
        else { fprintf(stderr, "dram_stream_stride_mix_run: unknown arg %s\n", argv[i]); return 2; }
    }
    if (strcmp(output_mode, "json") != 0 && strcmp(output_mode, "sha256_f32le") != 0) {
        fprintf(stderr, "dram_stream_stride_mix_run: --output-mode must be 'json' or 'sha256_f32le'\n");
        return 2;
    }
    const size_t n = (size_t)warps * accesses * stride_classes;
    float *h_in = (float *)malloc(n * sizeof(float));
    float *h_out = (float *)malloc(n * sizeof(float));
    seed_tensor(h_in, n, seed, -1.0, 1.0);

    float *d_in = nullptr, *d_out = nullptr, *d_sink = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_in,   n * sizeof(float)); if (rc) fail("mc in",   rc);
    rc = cudaMalloc(&d_out,  n * sizeof(float)); if (rc) fail("mc out",  rc);
    rc = cudaMalloc(&d_sink, sizeof(float));     if (rc) fail("mc sink", rc);
    rc = cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice); if (rc) fail("H2D", rc);

    // Bracket the kernel launch with cudaEvents so the runner reports a
    // cudaEventElapsedTime-derived wall-clock measurement. This is the only
    // live timing observable for this workload. The historical CUPTI
    // Activity dual-source reconciliation contract was retired by D8 in
    // S4-C1; it survives only in frozen campaign evidence, interpreted via
    // the build witness and this directory's PRESERVED_BINARY.md.
    cudaEvent_t kernel_start, kernel_stop;
    rc = cudaEventCreate(&kernel_start); if (rc) fail("eventCreate start", rc);
    rc = cudaEventCreate(&kernel_stop);  if (rc) fail("eventCreate stop", rc);
    rc = cudaEventRecord(kernel_start, 0); if (rc) fail("eventRecord start", rc);
    dram_stream_stride_mix_launch(d_in, d_out, d_sink, warps, accesses, stride_classes, 0);
    rc = cudaEventRecord(kernel_stop, 0); if (rc) fail("eventRecord stop", rc);
    rc = cudaEventSynchronize(kernel_stop); if (rc) fail("eventSynchronize", rc);
    float elapsed_ms = 0.0f;
    rc = cudaEventElapsedTime(&elapsed_ms, kernel_start, kernel_stop);
    if (rc) fail("eventElapsedTime", rc);
    double wall_time_ms = (double)elapsed_ms;
    if (!isfinite(wall_time_ms) || wall_time_ms <= 0.0) fail("non-finite-or-non-positive wall_time_ms", 0);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);

    rc = cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    if (strcmp(output_mode, "sha256_f32le") == 0) {
        // The perturbation-harness digest mode hashes the FP32
        // little-endian byte stream of the logical output tensor in
        // row-major (warps, accesses, stride_classes) order. The
        // ordering matches `flatten` in
        // tools/perturbation_harness/correctness_check.py.
        sha256_ctx_t ctx;
        sha256_init(&ctx);
        // host floats are already IEEE-754 FP32; on a little-endian
        // host (Linux x86_64) the in-memory byte order matches the
        // wire format. The digest input mirrors fp32_le_digest in
        // correctness_check.py.
        sha256_update(&ctx, (const uint8_t *)h_out, n * sizeof(float));
        uint8_t digest[32];
        sha256_final(&ctx, digest);
        char hex[65];
        hex_encode(digest, hex);
        printf("{\n  \"schema_version\": 1,\n  \"slug\": \"dram_stream_stride_mix\",\n");
        printf("  \"shape_set\": \"correctness_shapes\",\n  \"shape_point\": \"%s\",\n", shape_point);
        printf("  \"wall_time_ms\": %.9f,\n", wall_time_ms);
        printf("  \"output_mode\": \"sha256_f32le\",\n");
        printf("  \"output_sha256\": \"%s\",\n", hex);
        printf("  \"element_count\": %zu\n}\n", n);
    } else {
        printf("{\n  \"schema_version\": 1,\n  \"slug\": \"dram_stream_stride_mix\",\n");
        printf("  \"shape_set\": \"correctness_shapes\",\n  \"shape_point\": \"%s\",\n", shape_point);
        printf("  \"wall_time_ms\": %.9f,\n", wall_time_ms);
        printf("  \"output\": [");
        for (int w = 0; w < warps; ++w) {
            if (w > 0) printf(",");
            printf("[");
            for (int a = 0; a < accesses; ++a) {
                if (a > 0) printf(",");
                printf("[");
                for (int s = 0; s < stride_classes; ++s) {
                    if (s > 0) printf(",");
                    printf("%.6g", h_out[(w * accesses + a) * stride_classes + s]);
                }
                printf("]");
            }
            printf("]");
        }
        printf("]\n}\n");
    }

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_sink);
    free(h_in); free(h_out);
    return 0;
}
