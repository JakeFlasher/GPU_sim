// Independent opcode-category characterization, not application fitting.
// Compile-time unrolling makes the 8 -> 32 operation delta unambiguous in SASS.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>

#define CUDA_OK(call) do { const auto e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); return 2; } } while (0)

template<int Op, int Repeats>
__global__ void pipe_probe(unsigned* out, unsigned selector) {
  unsigned x = threadIdx.x + 1;
  float f = static_cast<float>(threadIdx.x + 1);
#pragma unroll
  for (int i = 0; i < Repeats; ++i) {
    if constexpr (Op == 0) // SHF, not SHFL
      asm volatile("shf.r.wrap.b32 %0, %0, %0, 5;" : "+r"(x));
    if constexpr (Op == 1)
      asm volatile("prmt.b32 %0, %0, %0, %1;" : "+r"(x) : "r"(selector));
    if constexpr (Op == 2)
      asm volatile("{ .reg .pred p; setp.lt.u32 p, %0, 16; selp.u32 %0, %0, 7, p; }" : "+r"(x));
    if constexpr (Op == 3)
      asm volatile("cvt.rn.bf16x2.f32 %0, %1, %1;" : "=r"(x) : "f"(f));
    if constexpr (Op == 4)
      asm volatile("shfl.sync.bfly.b32 %0, %0, 1, 31, -1;" : "+r"(x));
    if constexpr (Op == 5)
      asm volatile("mov.u32 %0, %%tid.x;" : "=r"(x));
    if constexpr (Op == 6)
      asm volatile("bar.sync 0;" ::: "memory");
    if constexpr (Op == 7)
      asm volatile("cp.async.commit_group;" ::: "memory");
  }
  out[threadIdx.x] = x;
}

template<int Op, int Repeats>
int check(unsigned* out) {
  pipe_probe<Op, Repeats><<<1, 32>>>(out, 0x0123);
  CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaDeviceSynchronize());
  unsigned values[32];
  CUDA_OK(cudaMemcpy(values, out, sizeof(values), cudaMemcpyDeviceToHost));
  for (unsigned t = 0; t < 32; ++t) {
    unsigned expected = t + 1;
    if constexpr (Op == 0) for (int i = 0; i < Repeats; ++i)
      expected = (expected >> 5) | (expected << 27);
    // PRMT with 0x0123 reverses bytes; even repetition restores the input.
    if constexpr (Op == 2) if (expected >= 16) expected = 7;
    if constexpr (Op == 3) {
      const float f = static_cast<float>(t + 1);
      unsigned bits;
      std::memcpy(&bits, &f, sizeof(bits));
      const unsigned bf16 = bits >> 16; // 1..32 are exact in BF16
      expected = bf16 | (bf16 << 16);
    }
    if constexpr (Op == 5) expected = t;
    if (values[t] != expected) {
      std::fprintf(stderr, "FAIL op=%d repeats=%d lane=%u got=%u expected=%u\n",
                   Op, Repeats, t, values[t], expected);
      return 1;
    }
  }
  std::printf("PASS op=%d repeats=%d lanes=32\n", Op, Repeats);
  return 0;
}

template<int Op> int pair(unsigned* out) {
  if (const int rc = check<Op, 8>(out)) return rc;
  return check<Op, 32>(out);
}

int main() {
  CUDA_OK(cudaSetDevice(0));
  unsigned* out = nullptr;
  CUDA_OK(cudaMalloc(&out, 32 * sizeof(unsigned)));
  const int results[] = {pair<0>(out), pair<1>(out), pair<2>(out), pair<3>(out),
                         pair<4>(out), pair<5>(out), pair<6>(out), pair<7>(out)};
  CUDA_OK(cudaFree(out));
  for (const int rc : results) if (rc) return rc;
  return 0;
}
