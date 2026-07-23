// Four small sm_86-portable kernels exercising the four phase-injection
// boundary classes that the tracer's phase-classifier predicates target,
// plus a constant-bank witness kernel for the trace-contract live gate.
// Driven by `workloads/simbench_bw/tests/test_phase_injection_smoke.py`
// under `CUDA_INJECTION64_PATH=tracer_tool.so` +
// `SIMBENCH_NVBIT_PHASE_INJECT=1`. Each kernel is launched once; the
// resulting `phase_records_0_<launch>.bin` files are decoded by the
// pytest and the ledger invariants asserted.
//
// Boundary-class coverage:
//   * cp_async_no_sync  — cp.async with no `__syncthreads()`. SASS
//     contains LDGSTS (and the cp.async wait emits LDGDEPBAR) but
//     zero `BAR.SYNC` warp barriers. The tracer must NOT emit any
//     `phase_kind=3` records for this kernel.
//   * syncthreads_only — a single `__syncthreads()` between two
//     trivial loads. Compiles to one `BAR.SYNC` warp barrier per
//     thread per launch. The tracer must emit at least one paired
//     `phase_kind=3` open/close per warp.
//   * divergent_branch — a tid-parity branch produces compiler-emitted
//     BSSY/BSYNC (sm_86 divergent control flow). The tracer must emit
//     at least one `phase_kind=2` (basic_block_entry) record per warp.
//   * constant_load — reads a `__constant__` array at a runtime-derived
//     index. Forces ptxas to emit LDC instructions so the trace-
//     contract live gate has a SASS witness for the producer's
//     CONSTANT_MEM path; downstream the gate asserts the protobuf
//     records at LDC PCs carry no memory payload (no addresses,
//     no `data_width`, no `memory_op_class`) under the producer's
//     STANDARD_MEM-only policy.
//
// Every launched kernel ends via EXIT (the SASS-level terminator
// emitted by ptxas at the bottom of every kernel function), which is
// the function_terminator predicate's positive test surface for the
// shared RET/EXIT terminator path. The CALL/RET function-call witness
// for the trace-contract live gate is supplied by `divergent_branch`
// (its two `__noinline__` helpers compile to CALL.REL.NOINC +
// RET.REL.NODEC SASS that ptxas emits at sm_86).

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

extern "C" __global__ void kernel_cp_async_no_sync(
    const float *__restrict__ in, float *__restrict__ out, int n)
{
  extern __shared__ float smem[];
  int tid = (int)threadIdx.x;
  if (tid < n) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :
        : "r"((unsigned)__cvta_generic_to_shared(&smem[tid])),
          "l"(&in[tid]));
  }
  asm volatile("cp.async.commit_group;\n");
  asm volatile("cp.async.wait_group 0;\n");
  // NO __syncthreads() — the cp.async no-CTA-barrier case requires
  // zero BAR.SYNC instructions so the warp-barrier predicate has
  // nothing to fire on. cp.async.wait_group emits LDGDEPBAR but not
  // BAR.SYNC.
  if (tid < n) {
    out[tid] = smem[tid];
  }
}

extern "C" __global__ void kernel_syncthreads_only(
    const float *__restrict__ in, float *__restrict__ out, int n)
{
  extern __shared__ float smem[];
  int tid = (int)threadIdx.x;
  if (tid < n) {
    smem[tid] = in[tid];
  }
  __syncthreads();
  if (tid < n) {
    out[tid] = smem[tid];
  }
}

// Two `__noinline__` device functions that ptxas cannot inline at
// the call site, so neither branch arm of `kernel_divergent_branch`
// can collapse via predication. Combined with a data-dependent test
// expression below, this forces the compiler to emit BSSY/BSYNC on
// sm_86 to mark the divergent region (Volta+ ITS).
extern "C" __device__ __noinline__ float divergent_helper_even(
    const float *__restrict__ in, int tid, int n)
{
  float v = 0.0f;
  for (int i = 0; i < 8; ++i) {
    int idx = (tid + i * 3) & (n - 1);
    v = v * 1.01f + in[idx];
  }
  return v;
}

extern "C" __device__ __noinline__ float divergent_helper_odd(
    const float *__restrict__ in, int tid, int n)
{
  float w = 1.0f;
  for (int i = 0; i < 4; ++i) {
    int idx = (tid * (i + 1)) & (n - 1);
    w = w + in[idx] * 0.5f;
  }
  return w;
}

// Constant-bank reference table. Sized to 64 floats so the kernel's
// runtime-computed index can mask with 0x3F and remain in-bounds. The
// host populates it via cudaMemcpyToSymbol before launch so its
// values are not known at compile time and the compiler cannot
// constant-fold the loads away.
__constant__ float kConstTable[64];

extern "C" __global__ void kernel_constant_load(
    const int *__restrict__ idx_in, float *__restrict__ out, int n)
{
  int tid = (int)threadIdx.x;
  if (tid >= n) return;
  // Two loads at a runtime-computed index defeat constant-propagation
  // and force ptxas to emit LDC at sm_86. We add the two loaded
  // values so the compiler cannot collapse one of them via
  // common-subexpression elimination across launches.
  int idx = idx_in[tid] & 63;
  float a = kConstTable[idx];
  float b = kConstTable[(idx + 1) & 63];
  out[tid] = a + b;
}

extern "C" __global__ void kernel_divergent_branch(
    const float *__restrict__ in, float *__restrict__ out, int n)
{
  int tid = (int)threadIdx.x;
  if (tid >= n) return;
  // Data-dependent test expression: the loaded value gates the
  // branch so ptxas cannot fold the test at compile time. Each arm
  // calls a __noinline__ helper to defeat predication. The two arms
  // converge on a single register `result`, so the compiler must
  // mark the divergent region with BSSY/BSYNC for warp
  // reconvergence (Volta+ ITS) before the join.
  float pivot = in[tid];
  float result;
  if (pivot > (float)tid * 0.5f) {
    result = divergent_helper_even(in, tid, n);
  } else {
    result = divergent_helper_odd(in, tid, n);
  }
  // __syncwarp() ensures the compiler emits a reconvergence point
  // (BSYNC) for the divergent region above. Together with BSSY,
  // this provides the divergent-branch SASS witness for the branch
  // allow-list smoke test.
  __syncwarp();
  out[tid] = result;
}

static void check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(e));
    std::exit(1);
  }
}

int main() {
  constexpr int N = 64;
  constexpr int BYTES = N * sizeof(float);

  float *h_in = (float *)std::malloc(BYTES);
  for (int i = 0; i < N; i++) h_in[i] = (float)i;

  float *d_in = nullptr, *d_out = nullptr;
  check(cudaMalloc(&d_in, BYTES), "cudaMalloc(d_in)");
  check(cudaMalloc(&d_out, BYTES), "cudaMalloc(d_out)");
  check(cudaMemcpy(d_in, h_in, BYTES, cudaMemcpyHostToDevice),
        "cudaMemcpy H2D");

  dim3 grid(1, 1, 1);
  dim3 block(N, 1, 1);
  size_t shmem = (size_t)BYTES;

  // Allocate an int index buffer for kernel_constant_load and seed
  // it with the thread id pattern so the constant-table lookup is a
  // runtime-computed index (defeats constant-propagation).
  int *h_idx = (int *)std::malloc(N * sizeof(int));
  for (int i = 0; i < N; i++) h_idx[i] = i;
  int *d_idx = nullptr;
  check(cudaMalloc(&d_idx, N * sizeof(int)), "cudaMalloc(d_idx)");
  check(cudaMemcpy(d_idx, h_idx, N * sizeof(int),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy H2D idx");

  // Populate the constant-bank reference table.
  float h_const[64];
  for (int i = 0; i < 64; i++) h_const[i] = (float)(i + 1) * 0.125f;
  check(cudaMemcpyToSymbol(kConstTable, h_const, sizeof(h_const)),
        "cudaMemcpyToSymbol(kConstTable)");

  // Launch order matches the test's expected launch_index assignments
  // (the tracer's per-context kernel_id counter is 1-indexed by the
  // time the per-launch flush sentinel runs, so the produced files
  // are phase_records_0_1.bin, _2.bin, _3.bin, _4.bin):
  //   launch_index=1  -> kernel_cp_async_no_sync
  //   launch_index=2  -> kernel_syncthreads_only
  //   launch_index=3  -> kernel_divergent_branch
  //   launch_index=4  -> kernel_constant_load
  kernel_cp_async_no_sync<<<grid, block, shmem>>>(d_in, d_out, N);
  check(cudaDeviceSynchronize(), "sync after cp_async_no_sync");

  kernel_syncthreads_only<<<grid, block, shmem>>>(d_in, d_out, N);
  check(cudaDeviceSynchronize(), "sync after syncthreads_only");

  kernel_divergent_branch<<<grid, block>>>(d_in, d_out, N);
  check(cudaDeviceSynchronize(), "sync after divergent_branch");

  kernel_constant_load<<<grid, block>>>(d_idx, d_out, N);
  check(cudaDeviceSynchronize(), "sync after constant_load");

  cudaFree(d_idx);
  cudaFree(d_in);
  cudaFree(d_out);
  std::free(h_idx);
  std::free(h_in);
  std::printf("phase_injection_microbench: launched 4 kernels, 0 errors.\n");
  return 0;
}
