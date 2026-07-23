// Host driver for carveout_sensitivity_probe.
//   in_fp32 = seeded(allocation_floats, seed, [-1.0, 1.0))
// Reference: identity round-trip; correctness shape's hash
// digest verifies determinism (the carveout preference does not
// affect the deterministic identity output).

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void carveout_sensitivity_probe_launch(
    const float *d_in, float *d_out, float *d_sink,
    int allocation_floats, int accesses, int warps,
    int shared_floats,
    cudaStream_t stream);

extern "C" __global__ void carveout_sensitivity_probe_kernel(
    const float *__restrict__ in, float *__restrict__ out,
    float *__restrict__ sink,
    int32_t allocation_floats, int32_t accesses, int32_t warps,
    int32_t shared_floats);

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
    fprintf(stderr, "carveout_sensitivity_probe_run: %s "
            "(rc=%d: %s)\n", msg, rc,
            cudaGetErrorString((cudaError_t)rc));
    exit(1);
}

int main(int argc, char **argv) {
    int allocation_bytes = 4096;
    int shared_bytes = 16384;
    int carveout_percent = 50;
    int accesses = 64;
    int warps = 4;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--allocation-bytes") == 0 && i + 1 < argc)
            allocation_bytes = atoi(argv[++i]);
        else if (strcmp(argv[i], "--shared-bytes") == 0 && i + 1 < argc)
            shared_bytes = atoi(argv[++i]);
        else if (strcmp(argv[i], "--carveout-percent") == 0 && i + 1 < argc)
            carveout_percent = atoi(argv[++i]);
        else if (strcmp(argv[i], "--accesses") == 0 && i + 1 < argc)
            accesses = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warps") == 0 && i + 1 < argc)
            warps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc)
            shape_point = argv[++i];
        else {
            fprintf(stderr, "carveout_sensitivity_probe_run: "
                    "unknown arg %s\n", argv[i]);
            return 2;
        }
    }
    int allocation_floats = allocation_bytes / 4;
    if (allocation_floats < 32) allocation_floats = 32;
    int shared_floats = shared_bytes / 4;
    if (shared_floats < 32) shared_floats = 32;
    int output_count = warps * 32;

    // Probe support up-front. cudaDevAttrMaxSharedMemoryPerBlockOptin
    // is the canonical query for whether the runtime admits
    // per-kernel carveout selection (CC >= 7.0). On devices that
    // do not advertise the attribute we emit a structured typed-
    // unavailable JSON document on stdout (no `output` field,
    // result_kind == "typed_unavailable",
    // typed_reason == "arch_feature_unavailable_carveout") and
    // exit cleanly. The simbench harness's correctness path
    // detects the typed-unavailable result_kind and persists it
    // as an unsupported typed outcome rather than comparing to
    // the reference.
    int max_optin = 0;
    cudaError_t rc;
    rc = cudaDeviceGetAttribute(
        &max_optin,
        cudaDevAttrMaxSharedMemoryPerBlockOptin,
        0);
    if (rc != cudaSuccess || max_optin <= 0) {
        printf("{\n  \"schema_version\": 1,\n");
        printf("  \"slug\": \"carveout_sensitivity_probe\",\n");
        printf("  \"shape_set\": \"correctness_shapes\",\n");
        printf("  \"shape_point\": \"%s\",\n", shape_point);
        printf("  \"result_kind\": \"typed_unavailable\",\n");
        printf("  \"typed_reason\": "
               "\"arch_feature_unavailable_carveout\"\n}\n");
        return 0;
    }

    float *h_in = (float *)malloc(allocation_floats * sizeof(float));
    float *h_out = (float *)malloc(output_count * sizeof(float));
    seed_tensor(h_in, allocation_floats, seed, -1.0, 1.0);

    float *d_in = nullptr, *d_out = nullptr, *d_sink = nullptr;
    rc = cudaMalloc(&d_in, allocation_floats * sizeof(float));
    if (rc) fail("mc in", rc);
    rc = cudaMalloc(&d_out, output_count * sizeof(float));
    if (rc) fail("mc out", rc);
    rc = cudaMalloc(&d_sink, sizeof(float));
    if (rc) fail("mc sink", rc);
    rc = cudaMemcpy(d_in, h_in, allocation_floats * sizeof(float),
                    cudaMemcpyHostToDevice);
    if (rc) fail("H2D", rc);

    // Set the per-kernel preferred shared-carveout split. The
    // runtime is hardware-advisory and may legally clip the
    // requested percentage to the device's supported partition
    // points (0, 25, 50, 75, 100 on most architectures).
    if (carveout_percent < 0) carveout_percent = 0;
    if (carveout_percent > 100) carveout_percent = 100;
    rc = cudaFuncSetAttribute(
        (const void *)carveout_sensitivity_probe_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        carveout_percent);
    if (rc) fail("set carveout", rc);
    // Opt the kernel into the requested dynamic-shared budget so
    // the runtime grants the sweepable shared tile (default 16 KiB
    // per launch). cudaFuncAttributeMaxDynamicSharedMemorySize is
    // the canonical opt-in for >48 KiB; for the default 16 KiB
    // budget the call is a no-op admission. The call is hardware-
    // advisory: the runtime clips to the carveout-bounded budget
    // already advertised via cudaDevAttrMaxSharedMemoryPerBlockOptin.
    rc = cudaFuncSetAttribute(
        (const void *)carveout_sensitivity_probe_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_bytes);
    if (rc) fail("set shared bytes", rc);

    carveout_sensitivity_probe_launch(
        d_in, d_out, d_sink,
        allocation_floats, accesses, warps, shared_floats,
        /*stream=*/0);
    rc = cudaDeviceSynchronize();
    if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_out, d_out, output_count * sizeof(float),
                    cudaMemcpyDeviceToHost);
    if (rc) fail("D2H", rc);

    printf("{\n  \"schema_version\": 1,\n");
    printf("  \"slug\": \"carveout_sensitivity_probe\",\n");
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
