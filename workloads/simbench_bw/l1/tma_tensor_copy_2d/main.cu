// Host driver for tma_tensor_copy_2d. Seeds a [rows, cols] BF16 tensor via
// the same PRNG as references/_common.py::deterministic_seeded_tensor,
// constructs two TMA tensor-maps (input + output) via `cuTensorMapEncodeTiled`
// (driver API), invokes the TMA identity-copy kernel, and emits the output
// as a 2-D JSON list matching `references/tma.py::build_reference`.
//
// Requires linking against `-lcuda` because `cuTensorMapEncodeTiled` lives
// in the driver library (libcuda.so), not the runtime library (libcudart).

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void tma_tensor_copy_2d_launch(
    const CUtensorMap *tensor_map_in,
    const CUtensorMap *tensor_map_out,
    int rows, int cols,
    cudaStream_t stream);


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
    fprintf(stderr, "tma_tensor_copy_2d_run: %s (cuda rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}

static void fail_cu(const char *msg, CUresult rc) {
    const char *err = "<unknown>";
    cuGetErrorString(rc, &err);
    fprintf(stderr, "tma_tensor_copy_2d_run: %s (CUresult=%d: %s)\n", msg, (int)rc, err);
    exit(1);
}


// Build a TMA tensor-map for a [rows, cols] BF16 tensor. The `box` dims
// are the per-TMA-issue transfer size — always the kernel's 64×64 tile;
// the grid of tile coords is computed from the globalDim at launch. A
// single `cp.async.bulk.tensor` loads one 64×64 box at the coord passed
// in the PTX.
static constexpr int kTileBoxRows = 64;
static constexpr int kTileBoxCols = 64;

static void make_tensor_map(CUtensorMap *out, void *global_address, int rows, int cols) {
    cuuint64_t size[2]       = { (cuuint64_t)cols, (cuuint64_t)rows };
    // Strides table holds (tensorRank - 1) entries. For row-major
    // [rows, cols] this is the byte stride between successive rows.
    // Must be a multiple of 16 bytes.
    cuuint64_t stride[1]     = { (cuuint64_t)cols * sizeof(__nv_bfloat16) };
    cuuint32_t box[2]        = { (cuuint32_t)kTileBoxCols, (cuuint32_t)kTileBoxRows };
    cuuint32_t elem_stride[2] = { 1, 1 };

    CUresult rc = cuTensorMapEncodeTiled(
        out,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        /*tensorRank=*/2,
        /*globalAddress=*/global_address,
        /*globalDim=*/size,
        /*globalStrides=*/stride,
        /*boxDim=*/box,
        /*elementStrides=*/elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (rc != CUDA_SUCCESS) fail_cu("cuTensorMapEncodeTiled", rc);
}


int main(int argc, char **argv) {
    int rows = 64;
    int cols = 64;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    const char *slug = "tma_tensor_copy_2d";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--rows") == 0 && i + 1 < argc) rows = atoi(argv[++i]);
        else if (strcmp(argv[i], "--cols") == 0 && i + 1 < argc) cols = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else {
            fprintf(stderr, "tma_tensor_copy_2d_run: unknown arg %s\n", argv[i]);
            return 2;
        }
    }
    // The TMA box dims are 64×64; the full tensor must be a multiple of
    // that in each dimension. One block per 64×64 tile handles the copy.
    if ((rows % 64) != 0 || (cols % 64) != 0 || rows <= 0 || cols <= 0) {
        fprintf(stderr,
                "tma_tensor_copy_2d_run: rows (%d) and cols (%d) must be positive "
                "multiples of 64 (the TMA box size).\n",
                rows, cols);
        return 2;
    }

    const size_t n = (size_t)rows * cols;
    float *h_in_f32 = (float *)malloc(n * sizeof(float));
    seed_tensor(h_in_f32, n, seed, -1.0, 1.0);

    __nv_bfloat16 *h_in = (__nv_bfloat16 *)malloc(n * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_out = (__nv_bfloat16 *)malloc(n * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n; ++i) h_in[i] = __float2bfloat16(h_in_f32[i]);

    __nv_bfloat16 *d_in = nullptr, *d_out = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_in, n * sizeof(__nv_bfloat16));  if (rc) fail("cudaMalloc d_in", rc);
    rc = cudaMalloc(&d_out, n * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_out", rc);
    rc = cudaMemcpy(d_in, h_in, n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D in", rc);

    CUtensorMap tensor_map_in, tensor_map_out;
    make_tensor_map(&tensor_map_in, d_in, rows, cols);
    make_tensor_map(&tensor_map_out, d_out, rows, cols);

    tma_tensor_copy_2d_launch(&tensor_map_in, &tensor_map_out, rows, cols, 0);
    rc = cudaDeviceSynchronize();
    if (rc) fail("sync (TMA kernel)", rc);

    rc = cudaMemcpy(h_out, d_out, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost); if (rc) fail("D2H out", rc);

    printf("{\n");
    printf("  \"schema_version\": 1,\n");
    printf("  \"slug\": \"%s\",\n", slug);
    printf("  \"shape_set\": \"correctness_shapes\",\n");
    printf("  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int r = 0; r < rows; ++r) {
        if (r > 0) printf(",");
        printf("[");
        for (int c = 0; c < cols; ++c) {
            if (c > 0) printf(",");
            float v = __bfloat162float(h_out[r * cols + c]);
            printf("%.6g", v);
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(d_in); cudaFree(d_out);
    free(h_in_f32); free(h_in); free(h_out);
    return 0;
}
