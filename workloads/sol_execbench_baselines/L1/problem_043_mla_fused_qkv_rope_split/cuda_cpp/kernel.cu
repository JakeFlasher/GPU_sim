// Stub translation unit: this ATen-delegated port has no custom __global__
// kernels. A bare .cu is still needed so torch's cpp_extension builds a CUDA
// extension (adding CUDA include paths); a .cpp-only extension fails with
// "cuda_runtime_api.h: No such file or directory".
#include <cuda_runtime.h>
