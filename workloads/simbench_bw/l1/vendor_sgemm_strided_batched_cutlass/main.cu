// Host driver for the project-owned CUTLASS FP32 strided-batched successor.
// The executable is intentionally independent of cuBLAS/cuBLASLt.

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using Element = float;

extern "C" int vendor_sgemm_strided_batched_cutlass_launch(
    const Element *d_a,
    const Element *d_b_column_major,
    float *d_c,
    int m,
    int n,
    int k,
    int batch_count,
    cudaStream_t stream);

static uint32_t next_u32(uint32_t *state) {
    *state = *state * 1103515245u + 12345u;
    return *state;
}

static float sample_value(uint32_t *state) {
    const double unit = static_cast<double>(next_u32(state)) / 4294967296.0;
    return static_cast<float>(-0.5 + unit);
}

static void fail(const char *message, int code = 1) {
    std::fprintf(
        stderr, "vendor_sgemm_strided_batched_cutlass_run: %s (code=%d)\n",
        message, code);
    std::exit(1);
}

static bool parse_int(
    int argc, char **argv, int *index, const char *name, int *value) {
    if (std::strcmp(argv[*index], name) != 0 || *index + 1 >= argc) {
        return false;
    }
    *value = std::atoi(argv[++*index]);
    return true;
}

int main(int argc, char **argv) {
    int m = 32;
    int n = 32;
    int k = 32;
    int batch_count = 4;
    int warmup = 0;
    int reps = 1;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    const char *shape_role = "named";

    for (int i = 1; i < argc; ++i) {
        if (parse_int(argc, argv, &i, "--m", &m)) {
        } else if (parse_int(argc, argv, &i, "--n", &n)) {
        } else if (parse_int(argc, argv, &i, "--k", &k)) {
        } else if (parse_int(argc, argv, &i, "--batch-count", &batch_count)) {
        } else if (parse_int(argc, argv, &i, "--warmup", &warmup)) {
        } else if (parse_int(argc, argv, &i, "--reps", &reps)) {
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (
            std::strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) {
            shape_point = argv[++i];
            shape_role = std::strncmp(shape_point, "trace_", 6) == 0
                ? "trace" : "named";
        } else if (std::strcmp(argv[i], "--timing_shape") == 0) {
            shape_point = "timing_shape";
            shape_role = "timing";
        } else if (std::strcmp(argv[i], "--correctness_shape") == 0) {
            shape_point = "correctness_shape";
            shape_role = "correctness";
        } else {
            std::fprintf(
                stderr,
                "vendor_sgemm_strided_batched_cutlass_run: unknown arg %s\n",
                argv[i]);
            return 2;
        }
    }

    if (
        m <= 0 || n <= 0 || k <= 0 || batch_count <= 0 || warmup < 0
        || reps <= 0) {
        fail("dimensions, batch_count, warmup, and reps are invalid");
    }

    const std::size_t matrix_a = static_cast<std::size_t>(m) * k;
    const std::size_t matrix_b = static_cast<std::size_t>(k) * n;
    const std::size_t matrix_c = static_cast<std::size_t>(m) * n;
    const std::size_t a_count = matrix_a * batch_count;
    const std::size_t b_count = matrix_b * batch_count;
    const std::size_t c_count = matrix_c * batch_count;
    std::vector<Element> h_a(a_count);
    std::vector<Element> h_b_column_major(b_count);
    std::vector<float> h_c(c_count, 0.0f);

    uint32_t state_a = seed * 2654435761u + 1u;
    uint32_t state_b = (seed + 1u) * 2654435761u + 1u;
    for (std::size_t index = 0; index < a_count; ++index) {
        h_a[index] = sample_value(&state_a);
    }
    for (int batch = 0; batch < batch_count; ++batch) {
        const std::size_t base = static_cast<std::size_t>(batch) * matrix_b;
        for (int row = 0; row < k; ++row) {
            for (int column = 0; column < n; ++column) {
                h_b_column_major[base + static_cast<std::size_t>(column) * k + row] =
                    sample_value(&state_b);
            }
        }
    }

    Element *d_a = nullptr;
    Element *d_b = nullptr;
    float *d_c = nullptr;
    cudaError_t cuda_status = cudaMalloc(&d_a, a_count * sizeof(Element));
    if (cuda_status != cudaSuccess) fail("cudaMalloc A");
    cuda_status = cudaMalloc(&d_b, b_count * sizeof(Element));
    if (cuda_status != cudaSuccess) fail("cudaMalloc B");
    cuda_status = cudaMalloc(&d_c, c_count * sizeof(float));
    if (cuda_status != cudaSuccess) fail("cudaMalloc C");

    cuda_status = cudaMemcpy(
        d_a, h_a.data(), a_count * sizeof(Element), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) fail("cudaMemcpy A");
    cuda_status = cudaMemcpy(
        d_b, h_b_column_major.data(), b_count * sizeof(Element),
        cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) fail("cudaMemcpy B");

    for (int iteration = 0; iteration < warmup; ++iteration) {
        if (vendor_sgemm_strided_batched_cutlass_launch(
                d_a, d_b, d_c, m, n, k, batch_count, nullptr) != 0) {
            fail("CUTLASS warmup");
        }
    }
    if (warmup > 0) {
        cuda_status = cudaDeviceSynchronize();
        if (cuda_status != cudaSuccess) fail("warmup synchronize");
    }

    for (int iteration = 0; iteration < reps; ++iteration) {
        if (vendor_sgemm_strided_batched_cutlass_launch(
                d_a, d_b, d_c, m, n, k, batch_count, nullptr) != 0) {
            fail("CUTLASS launch");
        }
    }
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) fail("synchronize");
    cuda_status = cudaMemcpy(
        h_c.data(), d_c, c_count * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) fail("cudaMemcpy C");

    std::printf(
        "{\n  \"schema_version\": 1,\n"
        "  \"slug\": \"vendor_sgemm_strided_batched_cutlass\",\n"
        "  \"shape_set\": \"successor_shapes\",\n"
        "  \"shape_role\": \"%s\",\n"
        "  \"shape_point\": \"%s\",\n"
        "  \"shape_axes\": {\"m\": %d, \"n\": %d, \"k\": %d, \"batch_count\": %d},\n"
        "  \"batch_count\": %d,\n"
        "  \"output_scope\": \"batch_zero_only\",\n"
        "  \"output\": [",
        shape_role, shape_point, m, n, k, batch_count, batch_count);
    // Emit batch zero for the bounded correctness payload. The explicit
    // output_scope field prevents this 2-D payload from being mistaken for a
    // full-batch tensor; batch_count and the strided launch contract remain
    // sealed in the surrounding fields.
    for (int row = 0; row < m; ++row) {
        if (row != 0) std::printf(",");
        std::printf("[");
        for (int column = 0; column < n; ++column) {
            if (column != 0) std::printf(",");
            std::printf("%.9g", h_c[static_cast<std::size_t>(row) * n + column]);
        }
        std::printf("]");
    }
    std::printf("]\n}\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
