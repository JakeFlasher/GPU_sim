// New-contract (ASYNC_COPY+ATOM address-carrying) canary for sm_86 — the new-contract
// canary and the end-to-end confirmation of the LDGSTS operand-ordering
// adjudication (metric_extractor MREF id=1 = global source).
//
// It exercises, over KNOWN device buffers, exactly the memory-op classes the
// address-carrying flip un-excludes, so the captured trace's admitted records
// can be checked:
//   (1) ordinary global load           -> LDG.E            (class GLOBAL)
//   (2) global atomic (return used)     -> ATOM.E.ADD       (class ATOMIC, GLOBAL)
//   (3) global reduction (return unused)-> RED.E.ADD        (class ATOMIC, GLOBAL)
//   (4) shared atomic                   -> ATOMS.ADD        (class ATOMIC, SHARED — exempt)
//   (5) cp.async global->shared         -> LDGSTS           (class ASYNC_COPY, global source)
//   (6) predicated-off global store     -> STG under flag==0 (whole warp masked off)
//
// Launched <<<1,32>>> with flag==0 so (6) is fully predicated off. No NCU metric
// is read here (Rule 1); this is a trace-input generator only.
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#include <cstdio>

__global__ void addr_carry_canary(const int* __restrict__ g_in,
                                  int* __restrict__ g_atom,
                                  int* __restrict__ g_red,
                                  int* __restrict__ g_out, int flag) {
  __shared__ int s_tile[32];
  __shared__ int s_atom[32];
  int t = threadIdx.x;

  // (1) ordinary global load.
  int v = g_in[t];

  // (2) global atomic, return value USED -> ATOM.E.
  int old = atomicAdd(&g_atom[0], v);

  // (3) global reduction, return value UNUSED -> RED.E.
  atomicAdd(&g_red[t & 31], v);

  // (4) shared atomic -> ATOMS (identity-exempt shared space).
  atomicAdd(&s_atom[t & 31], old);
  __syncthreads();

  // (5) cp.async / LDGSTS: global source -> shared destination.
  __pipeline_memcpy_async(&s_tile[t], &g_in[(t + 1) & 31], sizeof(int));
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncthreads();

  // (6) predicated-off global store (flag==0 at launch).
  if (flag) g_out[t] = s_tile[t] + s_atom[t & 31];
}

static int* alloc_filled(int n, int byte) {
  int* p = nullptr;
  if (cudaMalloc(&p, n * sizeof(int)) != cudaSuccess) return nullptr;
  cudaMemset(p, byte, n * sizeof(int));
  return p;
}

int main() {
  const int N = 32;
  int *g_in = alloc_filled(N, 1), *g_atom = alloc_filled(N, 0),
      *g_red = alloc_filled(N, 0), *g_out = alloc_filled(N, 0);
  if (!g_in || !g_atom || !g_red || !g_out) {
    fprintf(stderr, "canary: cudaMalloc failed\n");
    return 2;
  }
  addr_carry_canary<<<1, 32>>>(g_in, g_atom, g_red, g_out, 0);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "canary: launch error: %s\n", cudaGetErrorString(err));
    return 1;
  }
  cudaFree(g_in);
  cudaFree(g_atom);
  cudaFree(g_red);
  cudaFree(g_out);
  printf("addr_carry_canary: ok\n");
  return 0;
}
