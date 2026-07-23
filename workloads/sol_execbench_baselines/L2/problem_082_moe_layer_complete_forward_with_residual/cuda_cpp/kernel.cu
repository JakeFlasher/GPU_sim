// Stub translation unit: this ATen-delegated port has no custom device kernels.
// A bare .cu is still needed so torch's cpp_extension builds a CUDA extension
// (adding CUDA include paths); a .cpp-only extension fails to find
// cuda_runtime_api.h.
#include <cuda_runtime.h>
