// Scorability witness (op_ld surface): a bf16 tensor-core (HMMA) GEMM whose operand tiles
// are staged with cp.async (LDGSTS). Pre-flip the LDGSTS operand-load records were
// stripped to `unsupported_memory_path`, so the sim's op_ld surface for the
// tensor-core operand loads was `na` (starved). Post-flip they are admitted with a
// resolved allocation_id, so the op_ld surface becomes scorable.
//
// One warp, a single 16x16x16 wmma tile. cp.async stages A/B as 16-byte (int4)
// chunks — the size that lowers to LDGSTS on sm_86; wmma::mma_sync is the HMMA.
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdio>

using namespace nvcuda;

__global__ void gemm_hmma_bf16(const __nv_bfloat16* __restrict__ A,
                               const __nv_bfloat16* __restrict__ B,
                               float* __restrict__ C) {
  __shared__ __nv_bfloat16 sA[16 * 16];
  __shared__ __nv_bfloat16 sB[16 * 16];
  int t = threadIdx.x;

  // 16x16 bf16 = 512 bytes = 32 int4 chunks per tile; cp.async each (LDGSTS).
  const int4* A4 = reinterpret_cast<const int4*>(A);
  const int4* B4 = reinterpret_cast<const int4*>(B);
  int4* sA4 = reinterpret_cast<int4*>(sA);
  int4* sB4 = reinterpret_cast<int4*>(sB);
  for (int i = t; i < 32; i += blockDim.x) {
    __pipeline_memcpy_async(&sA4[i], &A4[i], sizeof(int4));
    __pipeline_memcpy_async(&sB4[i], &B4[i], sizeof(int4));
  }
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncthreads();

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
  wmma::fill_fragment(fc, 0.0f);
  wmma::load_matrix_sync(fa, sA, 16);
  wmma::load_matrix_sync(fb, sB, 16);
  wmma::mma_sync(fc, fa, fb, fc);  // HMMA
  wmma::store_matrix_sync(C, fc, 16, wmma::mem_row_major);
}

int main() {
  const int NA = 16 * 16;
  __nv_bfloat16 *A, *B;
  float* C;
  if (cudaMalloc(&A, NA * sizeof(__nv_bfloat16)) != cudaSuccess ||
      cudaMalloc(&B, NA * sizeof(__nv_bfloat16)) != cudaSuccess ||
      cudaMalloc(&C, NA * sizeof(float)) != cudaSuccess) {
    fprintf(stderr, "gemm_hmma_bf16: cudaMalloc failed\n");
    return 2;
  }
  cudaMemset(A, 0, NA * sizeof(__nv_bfloat16));
  cudaMemset(B, 0, NA * sizeof(__nv_bfloat16));
  cudaMemset(C, 0, NA * sizeof(float));
  gemm_hmma_bf16<<<1, 32>>>(A, B, C);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "gemm_hmma_bf16: launch error: %s\n", cudaGetErrorString(err));
    return 1;
  }
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  printf("gemm_hmma_bf16: ok\n");
  return 0;
}
