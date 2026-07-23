// Host driver for l2_persisting_window_sweep.
//   in_fp32 = seeded(allocation_floats, seed, [-1.0, 1.0))
// Reference: identity round-trip; correctness shape's hash
// digest verifies determinism (the persistence hint does not
// affect the output).

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void l2_persisting_window_sweep_launch(
    const float *d_in, float *d_out, float *d_sink,
    int allocation_floats, int accesses, int warps,
    cudaStream_t stream);

static inline uint32_t next_u32(uint32_t *s) {
    *s = (*s) * 1103515245u + 12345u;
    return *s;
}

static void seed_tensor(float *flat, size_t n, uint32_t seed,
                        double lo, double hi) {
    uint32_t st = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) {
        double r = (double)next_u32(&st) / 4294967296.0;
        flat[i] = (float)(lo + (hi - lo) * r);
    }
}

static void fail(const char *msg, int rc) {
    fprintf(stderr, "l2_persisting_window_sweep_run: %s "
            "(rc=%d: %s)\n", msg, rc,
            cudaGetErrorString((cudaError_t)rc));
    exit(1);
}

int main(int argc, char **argv) {
    int allocation_bytes = 4096;
    int persisting_window_bytes = 1024;
    int accesses = 64;
    int warps = 4;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--allocation-bytes") == 0 && i + 1 < argc)
            allocation_bytes = atoi(argv[++i]);
        else if (strcmp(argv[i], "--persisting-window-bytes") == 0 && i + 1 < argc)
            persisting_window_bytes = atoi(argv[++i]);
        else if (strcmp(argv[i], "--accesses") == 0 && i + 1 < argc)
            accesses = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warps") == 0 && i + 1 < argc)
            warps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc)
            shape_point = argv[++i];
        else {
            fprintf(stderr, "l2_persisting_window_sweep_run: "
                    "unknown arg %s\n", argv[i]);
            return 2;
        }
    }
    int allocation_floats = allocation_bytes / 4;
    if (allocation_floats < 32) allocation_floats = 32;
    int output_count = warps * 32;

    float *h_in = (float *)malloc(allocation_floats * sizeof(float));
    float *h_out = (float *)malloc(output_count * sizeof(float));
    seed_tensor(h_in, allocation_floats, seed, -1.0, 1.0);

    float *d_in = nullptr, *d_out = nullptr, *d_sink = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_in, allocation_floats * sizeof(float));
    if (rc) fail("mc in", rc);
    rc = cudaMalloc(&d_out, output_count * sizeof(float));
    if (rc) fail("mc out", rc);
    rc = cudaMalloc(&d_sink, sizeof(float));
    if (rc) fail("mc sink", rc);
    rc = cudaMemcpy(d_in, h_in, allocation_floats * sizeof(float),
                    cudaMemcpyHostToDevice);
    if (rc) fail("H2D", rc);

    // Set up the L2 persistence access-policy window over the
    // input allocation. Supported on CC >= 8.0; the persistence
    // hint is hardware-advisory and the runtime may legally clip
    // the requested window to the device's actual L2 persisting
    // capacity (queryable via cudaDeviceGetAttribute with
    // cudaDevAttrMaxAccessPolicyWindowSize). On devices that do
    // not advertise the attribute we emit a structured
    // typed-unavailable JSON document to stdout (no `output`
    // field, `result_kind == "typed_unavailable"`,
    // `typed_reason == "arch_feature_unavailable_l2_persisting"`)
    // and exit cleanly without running the kernel. The simbench
    // harness's correctness path detects the typed-unavailable
    // result_kind and persists it as an unsupported typed outcome
    // rather than comparing to the reference.
    cudaStream_t stream = nullptr;
    int max_window = 0;
    rc = cudaDeviceGetAttribute(
        &max_window,
        cudaDevAttrMaxAccessPolicyWindowSize,
        0);
    if (rc != cudaSuccess || max_window <= 0) {
        printf("{\n  \"schema_version\": 1,\n");
        printf("  \"slug\": \"l2_persisting_window_sweep\",\n");
        printf("  \"shape_set\": \"correctness_shapes\",\n");
        printf("  \"shape_point\": \"%s\",\n", shape_point);
        printf("  \"result_kind\": \"typed_unavailable\",\n");
        printf("  \"typed_reason\": "
               "\"arch_feature_unavailable_l2_persisting\"\n}\n");
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_sink);
        free(h_in); free(h_out);
        return 0;
    }
    rc = cudaStreamCreate(&stream);
    if (rc) fail("stream create", rc);
    cudaStreamAttrValue attr = {};
    attr.accessPolicyWindow.base_ptr = (void *)d_in;
    size_t requested = (size_t)persisting_window_bytes;
    size_t cap = (size_t)max_window;
    attr.accessPolicyWindow.num_bytes =
        requested < cap ? requested : cap;
    attr.accessPolicyWindow.hitRatio = 1.0f;
    attr.accessPolicyWindow.hitProp =
        cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp =
        cudaAccessPropertyStreaming;
    rc = cudaStreamSetAttribute(
        stream,
        cudaStreamAttributeAccessPolicyWindow,
        &attr);
    if (rc) fail("set persistence window", rc);

    l2_persisting_window_sweep_launch(
        d_in, d_out, d_sink,
        allocation_floats, accesses, warps, stream);
    rc = cudaDeviceSynchronize();
    if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_out, d_out, output_count * sizeof(float),
                    cudaMemcpyDeviceToHost);
    if (rc) fail("D2H", rc);

    if (stream) cudaStreamDestroy(stream);

    printf("{\n  \"schema_version\": 1,\n");
    printf("  \"slug\": \"l2_persisting_window_sweep\",\n");
    printf("  \"shape_set\": \"correctness_shapes\",\n");
    printf("  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int w = 0; w < warps; ++w) {
        if (w > 0) printf(",");
        printf("[");
        for (int l = 0; l < 32; ++l) {
            if (l > 0) printf(",");
            printf("%.6g", h_out[w * 32 + l]);
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_sink);
    free(h_in); free(h_out);
    return 0;
}
