// Observation-contract microfixtures (sm_86 quarantine).
//
// One tiny binary, one kernel per fixture, selected by argv[1]; each
// kernel's memory behavior is an exact function of the launch geometry
// and the runtime parameter W, so every counted quantity has a designed
// closed-form expectation. Runtime parameters (never template values)
// keep the SASS byte-identical across semantic probes: a fixture pair
// that differs ONLY in W isolates the counting semantic (issued-warp vs
// active-lane) from code-generation noise.
//
// Geometry contract (mirrored in the Python driver): BLOCKS x TPB
// threads, TPB a multiple of 32. Every kernel launches exactly once.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define BLOCKS 4
#define TPB 128
#define N (BLOCKS * TPB)
#define INST_UNROLL 64

#define CHECK(x) do { \
    cudaError_t err__ = (x); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s at %s:%d\n", \
                     cudaGetErrorName(err__), __FILE__, __LINE__); \
        std::exit(2); \
    } \
} while (0)

// --- instruction-semantics fixture -----------------------------------------
// Body of INST_UNROLL inline-PTX adds under `lane < W`. W is a runtime
// argument: the SASS is identical for every W; only the active mask of
// the guarded body changes. Issued-warp counting predicts IDENTICAL
// whole-kernel inst_executed across W; active-lane counting predicts a
// delta proportional to the lane count.
__global__ void fx_inst_pred(const float *in, float *out, int w) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    float acc = in[gid];
    if (lane < w) {
#pragma unroll
        for (int i = 0; i < INST_UNROLL; ++i) {
            asm volatile("add.f32 %0, %0, 0f3F800000;" : "+f"(acc));
        }
    }
    out[gid] = acc;
}

// --- request-formation / sectoring fixtures ---------------------------------
// Fully coalesced loads+stores at three vector widths. Per warp: one
// load instruction and one store instruction; the warp touches
// 32*width bytes = width 32B-sectors per access.
__global__ void fx_ld32(const float *__restrict__ in, float *__restrict__ out) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    out[gid] = in[gid] + 1.0f;
}

__global__ void fx_ld64(const float2 *__restrict__ in, float2 *__restrict__ out) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float2 v = in[gid];
    v.x += 1.0f; v.y += 1.0f;
    out[gid] = v;
}

__global__ void fx_ld128(const float4 *__restrict__ in, float4 *__restrict__ out) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float4 v = in[gid];
    v.x += 1.0f; v.y += 1.0f; v.z += 1.0f; v.w += 1.0f;
    out[gid] = v;
}

// Misaligned window: each warp reads 128B starting 4B past a sector
// boundary -> the warp's load spans 5 sectors instead of 4. The store
// stays aligned (4 sectors) so the two directions separate cleanly.
__global__ void fx_misaligned(const float *__restrict__ in,
                              float *__restrict__ out, int off) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    out[gid] = in[gid + off] + 1.0f;
}

// Lane-subset access: only lanes < W load+store. Pins inactive-lane
// handling for request formation (a warp with ANY active lane forms
// one request) and for sector counts (only touched sectors count).
__global__ void fx_lane_subset(const float *__restrict__ in,
                               float *__restrict__ out, int w) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    if (lane < w) {
        out[gid] = in[gid] + 1.0f;
    }
}

// --- ATOM vs RED attribution -------------------------------------------------
// Same arithmetic atomic, per-thread distinct addresses; the ONLY
// difference is whether the return value is consumed. NVCC lowers the
// unconsumed form to RED (reduction, no return path) and the consumed
// form to ATOMG. The driver verifies the SASS witness before trusting
// either fixture's attribution.
__global__ void fx_atom_return(float *out, float *sink) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float old = atomicAdd(&out[gid], 1.0f);
    sink[gid] = old;
}

__global__ void fx_red_noreturn(float *out) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(&out[gid], 1.0f);
}

// --- local-memory routing ----------------------------------------------------
// A per-thread array indexed dynamically (defeats register promotion)
// forces STL/LDL: one local store and one local load per thread, plus
// the bracketing global load/store. SASS witness: STL+LDL present.
__global__ void fx_local_rw(const float *__restrict__ in,
                            float *__restrict__ out, int rot) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    float arr[32];
    arr[lane] = in[gid];
    float v = arr[(lane + rot) & 31];
    out[gid] = v + 1.0f;
}

// Wide local accesses: 8B and 16B per-thread local array elements
// (STL.64/LDL.64 and STL.128/LDL.128). Pins whether the local request
// pipe splits wide accesses into multiple requests (the 32b fixture
// alone cannot see this).
__global__ void fx_local_wide64(const float2 *__restrict__ in,
                                float2 *__restrict__ out, int rot) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    float2 arr[32];
    arr[lane] = in[gid];
    float2 v = arr[(lane + rot) & 31];
    v.x += 1.0f;
    out[gid] = v;
}

__global__ void fx_local_wide128(const float4 *__restrict__ in,
                                 float4 *__restrict__ out, int rot) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    float4 arr[32];
    arr[lane] = in[gid];
    float4 v = arr[(lane + rot) & 31];
    v.x += 1.0f;
    out[gid] = v;
}

// --- texture / surface routing ----------------------------------------------
__global__ void fx_tex(cudaTextureObject_t t, float *__restrict__ out) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    out[gid] = tex1Dfetch<float>(t, gid) + 1.0f;
}

__global__ void fx_surf_ld(cudaSurfaceObject_t s, float *__restrict__ out) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float v = surf1Dread<float>(s, gid * (int)sizeof(float));
    out[gid] = v + 1.0f;
}

__global__ void fx_surf_st(const float *__restrict__ in, cudaSurfaceObject_t s) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    surf1Dwrite(in[gid] + 1.0f, s, gid * (int)sizeof(float));
}

// --- host -------------------------------------------------------------------

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <fixture> [w]\n"
                     "fixtures: fx_inst_pred fx_ld32 fx_ld64 fx_ld128 "
                     "fx_misaligned fx_lane_subset fx_atom_return "
                     "fx_red_noreturn fx_local_rw fx_tex fx_surf_ld "
                     "fx_surf_st\n",
                     argv[0]);
        return 2;
    }
    const char *fx = argv[1];
    int w = (argc > 2) ? std::atoi(argv[2]) : 32;

    // +64 floats of slack so fx_misaligned's off=1 window stays in
    // bounds; buffers sized for the widest (float4) fixture.
    size_t slack = 64;
    float *in = nullptr, *out = nullptr, *sink = nullptr;
    CHECK(cudaMalloc(&in, (N + slack) * 4 * sizeof(float)));
    CHECK(cudaMalloc(&out, (N + slack) * 4 * sizeof(float)));
    CHECK(cudaMalloc(&sink, N * sizeof(float)));
    CHECK(cudaMemset(in, 0, (N + slack) * 4 * sizeof(float)));
    CHECK(cudaMemset(out, 0, (N + slack) * 4 * sizeof(float)));
    CHECK(cudaMemset(sink, 0, N * sizeof(float)));

    if (std::strcmp(fx, "fx_inst_pred") == 0) {
        fx_inst_pred<<<BLOCKS, TPB>>>(in, out, w);
    } else if (std::strcmp(fx, "fx_ld32") == 0) {
        fx_ld32<<<BLOCKS, TPB>>>(in, out);
    } else if (std::strcmp(fx, "fx_ld64") == 0) {
        fx_ld64<<<BLOCKS, TPB>>>(reinterpret_cast<const float2 *>(in),
                                 reinterpret_cast<float2 *>(out));
    } else if (std::strcmp(fx, "fx_ld128") == 0) {
        fx_ld128<<<BLOCKS, TPB>>>(reinterpret_cast<const float4 *>(in),
                                  reinterpret_cast<float4 *>(out));
    } else if (std::strcmp(fx, "fx_misaligned") == 0) {
        fx_misaligned<<<BLOCKS, TPB>>>(in, out, 1);
    } else if (std::strcmp(fx, "fx_lane_subset") == 0) {
        fx_lane_subset<<<BLOCKS, TPB>>>(in, out, w);
    } else if (std::strcmp(fx, "fx_atom_return") == 0) {
        fx_atom_return<<<BLOCKS, TPB>>>(out, sink);
    } else if (std::strcmp(fx, "fx_red_noreturn") == 0) {
        fx_red_noreturn<<<BLOCKS, TPB>>>(out);
    } else if (std::strcmp(fx, "fx_local_rw") == 0) {
        fx_local_rw<<<BLOCKS, TPB>>>(in, out, 1);
    } else if (std::strcmp(fx, "fx_local_wide64") == 0) {
        fx_local_wide64<<<BLOCKS, TPB>>>(
            reinterpret_cast<const float2 *>(in),
            reinterpret_cast<float2 *>(out), 1);
    } else if (std::strcmp(fx, "fx_local_wide128") == 0) {
        fx_local_wide128<<<BLOCKS, TPB>>>(
            reinterpret_cast<const float4 *>(in),
            reinterpret_cast<float4 *>(out), 1);
    } else if (std::strcmp(fx, "fx_tex") == 0) {
        cudaResourceDesc rd{};
        rd.resType = cudaResourceTypeLinear;
        rd.res.linear.devPtr = in;
        rd.res.linear.desc = cudaCreateChannelDesc<float>();
        rd.res.linear.sizeInBytes = N * sizeof(float);
        cudaTextureDesc td{};
        td.readMode = cudaReadModeElementType;
        cudaTextureObject_t t = 0;
        CHECK(cudaCreateTextureObject(&t, &rd, &td, nullptr));
        fx_tex<<<BLOCKS, TPB>>>(t, out);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaDestroyTextureObject(t));
    } else if (std::strcmp(fx, "fx_surf_ld") == 0
               || std::strcmp(fx, "fx_surf_st") == 0) {
        cudaChannelFormatDesc cd = cudaCreateChannelDesc<float>();
        cudaArray_t arr = nullptr;
        CHECK(cudaMallocArray(&arr, &cd, N, 1, cudaArraySurfaceLoadStore));
        cudaResourceDesc rd{};
        rd.resType = cudaResourceTypeArray;
        rd.res.array.array = arr;
        cudaSurfaceObject_t s = 0;
        CHECK(cudaCreateSurfaceObject(&s, &rd));
        if (std::strcmp(fx, "fx_surf_ld") == 0) {
            fx_surf_ld<<<BLOCKS, TPB>>>(s, out);
        } else {
            fx_surf_st<<<BLOCKS, TPB>>>(in, s);
        }
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaDestroySurfaceObject(s));
        CHECK(cudaFreeArray(arr));
    } else {
        std::fprintf(stderr, "unknown fixture: %s\n", fx);
        return 2;
    }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaFree(in));
    CHECK(cudaFree(out));
    CHECK(cudaFree(sink));
    std::printf("ok %s w=%d\n", fx, w);
    return 0;
}
