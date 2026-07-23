---
title: "4.11.2.2. Using TMA to transfer multi-dimensional arrays"
section: "4.11.2.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#using-tma-to-transfer-multi-dimensional-arrays"
---

### [4.11.2.2. Using TMA to transfer multi-dimensional arrays](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#using-tma-to-transfer-multi-dimensional-arrays)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#using-tma-to-transfer-multi-dimensional-arrays "Permalink to this headline")

In this section, we will focus on multi-dimensional TMA copies.
The primary difference between the one-dimensional and multi-dimensional case is
that a tensor map must be created on the host and passed to the CUDA kernel.

The following table summarizes the possible source and destination memory spaces and completion mechanisms for bulk-tensor asynchronous TMA along with the API that exposes it in device code.

| Source | Destination | Completion Mechanism | API |
| --- | --- | --- | --- |
| global | global |  |  |
| shared::cta | global | bulk async-group | [cuda::ptx::cp_async_bulk_tensor](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/cp_async_bulk_tensor.html) |
| global | shared::cta | shared memory barrier | [cuda::ptx::cp_async_bulk_tensor](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/cp_async_bulk_tensor.html) |
| global | shared::cluster | shared memory barrier | [cuda::ptx::cp_async_bulk_tensor](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/cp_async_bulk_tensor.html) |
| shared::cta | shared::cluster | shared memory barrier | [cuda::ptx::cp_async_bulk_tensor](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/cp_async_bulk_tensor.html) |
| shared::cta | shared::cta |  |  |

All functionality requires inline PTX that is currently made available through the `cuda::ptx` namespace in the [CUDA Standard C++](https://nvidia.github.io/cccl/libcudacxx/ptx_api.html) library.

In the following, we describe how to create a tensor map using the CUDA driver API, how
to pass it to the device, and how to use it on the device.

**Driver API**. A tensor map is created using the [cuTensorMapEncodeTiled](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
driver API. This API can be accessed by linking to the driver directly
(`-lcuda`) or by using the [cudaGetDriverEntryPointByVersion](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DRIVER__ENTRY__POINT.html)
API. Below, we show how to get a pointer to the `cuTensorMapEncodeTiled` API.
For more information, refer to [Driver Entry Point Access](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/driver-entry-point-access.html#driver-entry-point-access).

```c++
#include <cudaTypedefs.h> // PFN_cuTensorMapEncodeTiled, CUtensorMap

PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
  // Get pointer to cuTensorMapEncodeTiled
  cudaDriverEntryPointQueryResult driver_status;
  void* cuTensorMapEncodeTiled_ptr = nullptr;
  CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000, cudaEnableDefault, &driver_status));
  assert(driver_status == cudaDriverEntryPointSuccess);

  return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
}
```

**Creation**. Creating a tensor map requires many parameters. Among
them are the base pointer to an array in global memory, the size of the array
(in number of elements), the stride from one row to the next (in bytes), the
size of the shared memory buffer (in number of elements). The code below creates
a tensor map to describe a two-dimensional row-major array of size `GMEM_HEIGHT
x GMEM_WIDTH`. Note the order of the parameters: the fastest moving dimension
comes first.

```c++
  CUtensorMap tensor_map{};
  // rank is the number of dimensions of the array.
  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {GMEM_WIDTH, GMEM_HEIGHT};
  // The stride is the number of bytes to traverse from the first element of one row to the next.
  // It must be a multiple of 16.
  uint64_t stride[rank - 1] = {GMEM_WIDTH * sizeof(int)};
  // The box_size is the size of the shared memory buffer that is used as the
  // destination of a TMA transfer.
  uint32_t box_size[rank] = {SMEM_WIDTH, SMEM_HEIGHT};
  // The distance between elements in units of sizeof(element). A stride of 2
  // can be used to load only the real component of a complex-valued tensor, for instance.
  uint32_t elem_stride[rank] = {1, 1};

  // Get a function pointer to the cuTensorMapEncodeTiled driver API.
  auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

  // Create the tensor descriptor.
  CUresult res = cuTensorMapEncodeTiled(
    &tensor_map,                // CUtensorMap *tensorMap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
    rank,                       // cuuint32_t tensorRank,
    tensor_ptr,                 // void *globalAddress,
    size,                       // const cuuint64_t *globalDim,
    stride,                     // const cuuint64_t *globalStrides,
    box_size,                   // const cuuint32_t *boxDim,
    elem_stride,                // const cuuint32_t *elementStrides,
    // Interleave patterns can be used to accelerate loading of values that
    // are less than 4 bytes long.
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    // Swizzling can be used to avoid shared memory bank conflicts.
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    // L2 Promotion can be used to widen the effect of a cache-policy to a wider
    // set of L2 cache lines.
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    // Any element that is outside of bounds will be set to zero by the TMA transfer.
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );
```

**Host-to-device transfer**. There are three ways to make a tensor map accessible to
device code. The recommended approach is to pass the tensor map as a const `__grid_constant__`
parameter to a kernel. The other possibilities are copying the tensor map into device `__constant__`
memory using `cudaMemcpyToSymbol` or accessing it via global memory. When passing the tensor map as a parameter, some versions of the
GCC C++ compiler issue the warning “the ABI for passing parameters with 64-byte
alignment has changed in GCC 4.6”. This warning can be ignored.

```cuda
#include <cuda.h>

__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map)
{
   // Use tensor_map here.
}
int main() {
  CUtensorMap map;
  // [ ..Initialize map.. ]
  kernel<<<1, 1>>>(map);
}
```

As an alternative to the `__grid_constant__` kernel parameter, a global
`__constant__` variable can be used. An example is included
below.

```cuda
#include <cuda.h>

__constant__ CUtensorMap global_tensor_map;
__global__ void kernel()
{
  // Use global_tensor_map here.
}
int main() {
  CUtensorMap local_tensor_map;
  // [ ..Initialize map.. ]
  cudaMemcpyToSymbol(global_tensor_map, &local_tensor_map, sizeof(CUtensorMap));
  kernel<<<1, 1>>>();
}
```

Finally, it is possible to copy the tensor map to global memory. Using a pointer to a
tensor map in global device memory requires a fence in each thread block before any thread
in the block uses the updated tensor map. Further uses of the tensor map by that thread block
do not need to be fenced unless the tensor map is modified again. Note that this mechanism
may be slower than the two mechanisms described above.

```cuda
#include <cuda.h>
#include <cuda/ptx>
namespace ptx = cuda::ptx;

__device__ CUtensorMap global_tensor_map;
__global__ void kernel(CUtensorMap *tensor_map)
{
  // Fence acquire tensor map:
  ptx::n32_t<128> size_bytes;
  // Since the tensor map was modified from the host using cudaMemcpy,
  // the scope should be .sys.
  ptx::fence_proxy_tensormap_generic(
     ptx::sem_acquire, ptx::scope_sys, tensor_map, size_bytes
 );
 // Safe to use tensor_map after fence inside this thread.
}
int main() {
  CUtensorMap local_tensor_map;
  // [ ..Initialize map.. ]
  cudaMemcpy(&global_tensor_map, &local_tensor_map, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  kernel<<<1, 1>>>(global_tensor_map);
}
```

**Use**. The kernel below loads a 2D tile of size `SMEM_HEIGHT x SMEM_WIDTH`
from a larger 2D array. The top-left corner of the tile is indicated by the
indices `x` and `y`. The tile is loaded into shared memory, modified, and
written back to global memory.

```cuda
#include <cuda.h>         // CUtensormap
#include <cuda/barrier>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

__device__ inline bool is_elected()
{
    unsigned int tid = threadIdx.x;
    unsigned int warp_id = tid / 32;
    unsigned int uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0); // Broadcast from lane 0.
    return (uniform_warp_id == 0 && ptx::elect_sync(0xFFFFFFFF)); // Elect a leader thread among warp 0.
}

__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map, int x, int y) {
  // The destination shared memory buffer of a bulk tensor operation should be
  // 128 byte aligned.
  __shared__ alignas(128) int smem_buffer[SMEM_HEIGHT][SMEM_WIDTH];

  // Initialize shared memory barrier with the number of threads participating in the barrier.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;

  if (threadIdx.x == 0) {
    // Initialize barrier. All `blockDim.x` threads in block participate.
    init(&bar, blockDim.x);
  }
  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();

  barrier::arrival_token token;
  if (is_elected()) {
    // Initiate bulk tensor copy.
    int32_t tensor_coords[2] = { x, y };
    ptx::cp_async_bulk_tensor(
      ptx::space_shared, ptx::space_global,
      &smem_buffer, &tensor_map, tensor_coords,
      cuda::device::barrier_native_handle(bar));
    // Arrive on the barrier and tell how many bytes are expected to come in.
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
  } else {
    // Other threads just arrive.
    token = bar.arrive();
  }
  // Wait for the data to have arrived.
  bar.wait(std::move(token));

  // Symbolically modify a value in shared memory.
  smem_buffer[0][threadIdx.x] += threadIdx.x;

  // Wait for shared memory writes to be visible to TMA engine.
  ptx::fence_proxy_async(ptx::space_shared);
  __syncthreads();
  // After syncthreads, writes by all threads are visible to TMA engine.

  // Initiate TMA transfer to copy shared memory to global memory
  if (is_elected()) {
    int32_t tensor_coords[2] = { x, y };
    ptx::cp_async_bulk_tensor(
      ptx::space_global, ptx::space_shared,
      &tensor_map, tensor_coords, &smem_buffer);
    // Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
    ptx::cp_async_bulk_commit_group();
    // Wait for the group to have completed reading from shared memory.
    ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
  }

  // Destroy barrier. This invalidates the memory region of the barrier. If
  // further computations were to take place in the kernel, this allows the
  // memory location of the shared memory barrier to be reused.
  if (threadIdx.x == 0) {
    (&bar)->~barrier();
  }
}
```

**Negative indices and out of bounds**. When part of the tile that is being
_read_ from global to shared memory is out of bounds, the shared memory that
corresponds to the out of bounds area is zero-filled. The top-left corner
indices of the tile may also be negative. When _writing_ from shared to global
memory, parts of the tile may be out of bounds, but the top left corner cannot
have any negative indices.

**Size and stride**. The size of a tensor is the number of elements along one
dimension. All sizes must be greater than one. The stride is the number of bytes
between elements of the same dimension. For instance, a *4 x 4* matrix of
integers has sizes 4 and 4. Since it has 4 bytes per element, the strides are 4
and 16 bytes. Due to alignment requirements, a *4 x 3* row-major matrix of
integers must have strides of 4 and 16 bytes as well. Each row is padded with 4
extra bytes to ensure that the start of the next row is aligned to 16 bytes.
More information about alignment requirements can be found in the table below.

| Address / Size | Alignment |
| --- | --- |
| Global memory address | Must be 16 byte aligned. |
| Global memory sizes | Must be greater than or equal to one. Does not have to be a multiple of 16 bytes. |
| Global memory strides | Must be multiples of 16 bytes. |
| Shared memory address | Must be 128 byte aligned. |
| Shared memory barrier address | Must be 8 byte aligned (this is guaranteed by `cuda::barrier`). |
| Size of transfer | Must be a multiple of 16 bytes. |

#### [4.11.2.2.1. Encoding a Tensor Map on Device](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#encoding-a-tensor-map-on-device)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#encoding-a-tensor-map-on-device "Permalink to this headline")

Previous sections have described how to create a tensor map on the host using the CUDA driver API.

This section explains how to encode a tiled-type tensor map on device. This is useful in situations where the typical
way of transferring the tensor map (using `const __grid_constant__` kernel parameters) is undesirable, for instance,
when processing a batch of tensors of various sizes in a single kernel launch.

The recommended pattern is as follows:

1. Create a tensor map “template”, `template_tensor_map`, using the Driver API on the host.
2. In a device kernel, copy the `template_tensor_map`, modify the copy, store in global memory, and appropriately fence.
3. Use the tensor map in a kernel with appropriate fencing.

The high-level code structure is as follows:

```cuda
// Initialize device context:
CUDA_CHECK(cudaDeviceSynchronize());

// Create a tensor map template using the cuTensorMapEncodeTiled driver function
CUtensorMap template_tensor_map = make_tensormap_template();

// Allocate tensor map and tensor in global memory
CUtensorMap* global_tensor_map;
CUDA_CHECK(cudaMalloc(&global_tensor_map, sizeof(CUtensorMap)));
char* global_buf;
CUDA_CHECK(cudaMalloc(&global_buf, 8 * 256));

// Fill global buffer with data.
fill_global_buf<<<1, 1>>>(global_buf);

// Define the parameters of the tensor map that will be created on device.
tensormap_params p{};
p.global_address    = global_buf;
p.rank              = 2;
p.box_dim[0]        = 128; // The box in shared memory has half the width of the full buffer
p.box_dim[1]        = 4;   // The box in shared memory has half the height of the full buffer
p.global_dim[0]     = 256; //
p.global_dim[1]     = 8;   //
p.global_stride[0]  = 256; //
p.element_stride[0] = 1;   //
p.element_stride[1] = 1;   //

// Encode global_tensor_map on device:
encode_tensor_map<<<1, 32>>>(template_tensor_map, p, global_tensor_map);

// Use it from another kernel:
consume_tensor_map<<<1, 1>>>(global_tensor_map);

// Check for errors:
CUDA_CHECK(cudaDeviceSynchronize());
```

The following sections  describe the high-level steps. Throughout the examples, the following `tensormap_params`
struct contains the new values of the fields to be updated. It is included here to reference when reading the examples.

```cuda
struct tensormap_params {
  void* global_address;
  int rank;
  uint32_t box_dim[5];
  uint64_t global_dim[5];
  size_t global_stride[4];
  uint32_t element_stride[5];
};
```

#### [4.11.2.2.2. Device-side Encoding and Modification of a Tensor Map](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#device-side-encoding-and-modification-of-a-tensor-map)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#device-side-encoding-and-modification-of-a-tensor-map "Permalink to this headline")

The recommended process of encoding a tensor map in global memory proceeds as follows.

1. Pass an existing tensor map, the `template_tensor_map`, to the kernel. In contrast to kernels that use
the tensor map in a `cp.async.bulk.tensor` instruction, this may be done in any way: a pointer to global
memory, kernel parameter, a `__const___` variable, and so on.
2. Copy-initialize a tensor map in shared memory with the template_tensor_map value.
3. Modify the tensor map in shared memory using the [cuda::ptx::tensormap_replace](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/tensormap_replace.html)
functions. These functions wrap the [tensormap.replace](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-tensormap-replace)
PTX instruction, which can be used to modify any field of a tiled-type tensor map, including the
base address, size, stride, and so on.
4. Using the [cuda::ptx::tensormap_copy_fenceproxy](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/tensormap_cp_fenceproxy.html#tensormap-cp-fenceproxy)
function, copy the modified tensor map from shared memory to global memory and perform any necessary fencing.

The following code contains a kernel that follows these steps. For completeness, it modifies all the fields
of the tensor map. Typically, a kernel will modify just a few fields.

In this kernel, `template_tensor_map` is passed as a kernel parameter. This is the preferred way of moving `template_tensor_map`
from the host to the device. If the kernel is intended to update an existing tensor map in device memory, it can take a
pointer to the existing tensor map to modify.

> **Note**
>
> The format of the tensor map may change over time. Therefore, the [cuda::ptx::tensormap_replace](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/tensormap_replace.html)
> functions and corresponding [tensormap.replace.tile](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-tensormap-replace)
> PTX instructions are marked as specific to sm_90a. To use them, compile using `nvcc -arch sm_90a ....`.

> **Tip**
>
> On sm_90a, a zero-initialized buffer in shared memory may also be used as the initial tensor map value. This
> enables encoding a tensor map purely on device, without using the driver API to encode the `template_tensor_map value`.

> **Note**
>
> On-device modification is only supported for tiled-type tensor maps; other tensor map types cannot be modified on device. For more
> information on the tensor map types, refer to the [Driver API reference](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY).

```cuda
#include <cuda/ptx>

namespace ptx = cuda::ptx;

// launch with 1 warp.
__launch_bounds__(32)
__global__ void encode_tensor_map(const __grid_constant__ CUtensorMap template_tensor_map, tensormap_params p, CUtensorMap* out) {
   __shared__ alignas(128) CUtensorMap smem_tmap;
   if (threadIdx.x == 0) {
      // Copy template to shared memory:
      smem_tmap = template_tensor_map;

      const auto space_shared = ptx::space_shared;
      ptx::tensormap_replace_global_address(space_shared, &smem_tmap, p.global_address);
      // For field .rank, the operand new_val must be ones less than the desired
      // tensor rank as this field uses zero-based numbering.
      ptx::tensormap_replace_rank(space_shared, &smem_tmap, p.rank - 1);

      // Set box dimensions:
      if (0 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<0>{}, p.box_dim[0]); }
      if (1 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<1>{}, p.box_dim[1]); }
      if (2 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<2>{}, p.box_dim[2]); }
      if (3 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<3>{}, p.box_dim[3]); }
      if (4 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<4>{}, p.box_dim[4]); }
      // Set global dimensions:
      if (0 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<0>{}, (uint32_t) p.global_dim[0]); }
      if (1 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<1>{}, (uint32_t) p.global_dim[1]); }
      if (2 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<2>{}, (uint32_t) p.global_dim[2]); }
      if (3 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<3>{}, (uint32_t) p.global_dim[3]); }
      if (4 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<4>{}, (uint32_t) p.global_dim[4]); }
      // Set global stride:
      if (1 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<0>{}, p.global_stride[0]); }
      if (2 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<1>{}, p.global_stride[1]); }
      if (3 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<2>{}, p.global_stride[2]); }
      if (4 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<3>{}, p.global_stride[3]); }
      // Set element stride:
      if (0 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<0>{}, p.element_stride[0]); }
      if (1 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<1>{}, p.element_stride[1]); }
      if (2 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<2>{}, p.element_stride[2]); }
      if (3 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<3>{}, p.element_stride[3]); }
      if (4 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<4>{}, p.element_stride[4]); }

      // These constants are documented in this table:
      // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensormap-new-val-validity
      auto u8_elem_type = ptx::n32_t<0>{};
      ptx::tensormap_replace_elemtype(space_shared, &smem_tmap, u8_elem_type);
      auto no_interleave = ptx::n32_t<0>{};
      ptx::tensormap_replace_interleave_layout(space_shared, &smem_tmap, no_interleave);
      auto no_swizzle = ptx::n32_t<0>{};
      ptx::tensormap_replace_swizzle_mode(space_shared, &smem_tmap, no_swizzle);
      auto zero_fill = ptx::n32_t<0>{};
      ptx::tensormap_replace_fill_mode(space_shared, &smem_tmap, zero_fill);
   }
   // Synchronize the modifications with other threads in warp
   __syncwarp();
   // Copy the tensor map to global memory collectively with threads in the warp.
   // In addition: make the updated tensor map visible to other threads on device that
   // for use with cp.async.bulk.
   ptx::n32_t<128> bytes_128;
   ptx::tensormap_cp_fenceproxy(ptx::sem_release, ptx::scope_gpu, out, &smem_tmap, bytes_128);
}
```

#### [4.11.2.2.3. Usage of a Modified Tensor Map](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#usage-of-a-modified-tensor-map)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#usage-of-a-modified-tensor-map "Permalink to this headline")

In contrast to using a tensor map that is passed as a `const __grid_constant__` kernel parameter, using a tensor map in
global memory requires explicitly establishing a release-acquire pattern in the tensor map proxy between the threads
that modify the tensor map and the threads that use it.

The release part of the pattern was shown in the previous section. It is accomplished using
the [cuda::ptx::tensormap.cp_fenceproxy](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/tensormap_cp_fenceproxy.html) function.

The acquire part is accomplished using the [cuda::ptx::fence_proxy_tensormap_generic](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/fence.html)
function that wraps the [fence.proxy.tensormap::generic.acquire](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar-fence)
instruction. If the two threads participating in the release-acquire pattern are on the same device, the `.gpu` scope suffices. If the threads are on
different devices, the `.sys` scope must be used. Once a tensor map has been acquired by one thread, it can be used by other threads in the block
after sufficient synchronization, for example, using `__syncthreads()`. The thread that uses the tensor map and the thread that performs the fence
must be in the same block. That is, if the threads are in, for example, two different thread blocks of the same cluster, the same grid, or a
different kernel, synchronization APIs such as `cooperative_groups::cluster` or `grid_group::sync()` or stream-order synchronization do not
suffice to establish ordering for tensor map updates, that is, threads in these other thread blocks still need to acquire the tensor map proxy
at the right scope before using the updated tensor map. If there are no intermediate modifications, the fence does not have to be repeated
before each `cp.async.bulk.tensor` instruction.

The `fence` and subsequent use of the tensor map is shown in the following example.

```cuda
// Consumer of tensor map in global memory:
__global__ void consume_tensor_map(CUtensorMap* tensor_map) {
  // Fence acquire tensor map:
  ptx::n32_t<128> size_bytes;
  ptx::fence_proxy_tensormap_generic(ptx::sem_acquire, ptx::scope_sys, tensor_map, size_bytes);
  // Safe to use tensor_map after fence.

  __shared__ uint64_t bar;
  __shared__ alignas(128) char smem_buf[4][128];

  if (threadIdx.x == 0) {
    // Initialize barrier
    ptx::mbarrier_init(&bar, 1);
    // Issue TMA request
    ptx::cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global, smem_buf, tensor_map, {0, 0}, &bar);
    // Arrive on barrier. Expect 4 * 128 bytes.
    ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar, sizeof(smem_buf));
  }
  const int parity = 0;
  // Wait for load to have completed
  while (!ptx::mbarrier_try_wait_parity(&bar, parity)) {}

  // print items:
  printf("Got:\n\n");
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 128; ++i) {
      printf("%3d ", smem_buf[j][i]);
      if (i % 32 == 31) { printf("\n"); };
    }
    printf("\n");
  }
}
```

#### [4.11.2.2.4. Creating a Template Tensor Map Value Using the Driver API](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#creating-a-template-tensor-map-value-using-the-driver-api)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#creating-a-template-tensor-map-value-using-the-driver-api "Permalink to this headline")

The following code creates a minimal tiled-type tensor map that can be subsequently modified on device.

```cuda
CUtensorMap make_tensormap_template() {
  CUtensorMap template_tensor_map{};
  auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

  uint32_t dims_32         = 16;
  uint64_t dims_strides_64 = 16;
  uint32_t elem_strides    = 1;

  // Create the tensor descriptor.
  CUresult res = cuTensorMapEncodeTiled(
    &template_tensor_map, // CUtensorMap *tensorMap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
    1,                // cuuint32_t tensorRank,
    nullptr,          // void *globalAddress,
    &dims_strides_64, // const cuuint64_t *globalDim,
    &dims_strides_64, // const cuuint64_t *globalStrides,
    &dims_32,         // const cuuint32_t *boxDim,
    &elem_strides,    // const cuuint32_t *elementStrides,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  CU_CHECK(res);
  return template_tensor_map;
}
```

#### [4.11.2.2.5. Shared-Memory Bank Swizzling](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#shared-memory-bank-swizzling)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#shared-memory-bank-swizzling "Permalink to this headline")

By default, the TMA engine loads data to shared memory in the same order as it is laid out in global memory. However, this
layout may not be optimal for certain shared memory access patterns, as it could cause shared memory bank conflicts. To
improve performance and reduce bank conflicts, we can change the shared memory layout by applying a ‘swizzle pattern’.

Shared memory has 32 banks that are organized such that successive 32-bit words map to successive banks. Each bank has a
bandwidth of 32 bits per clock cycle. When loading and storing shared memory, bank conflicts arise if the same bank is
used multiple times within a transaction, resulting in reduced bandwidth. See [Shared Memory Access Patterns](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-shared-memory-access-patterns).

To ensure that data is laid out in shared memory in such a way that user code can avoid shared memory bank conflicts,
the TMA engine can be instructed to ‘swizzle’ the data before storing it in shared memory and ‘unswizzle’ it when copying
the data back from shared memory to global memory. The tensor map encodes the ‘swizzle mode’ indicating which swizzle pattern is used.

**Example: Matrix Transpose**

An example is the transpose of a matrix where data is mapped from row to column first access. The data is stored row major in
global memory, but we want to also access it column wise in shared memory, which leads to bank conflicts. However, by using
the 128 bytes ‘swizzle’ mode and new shared memory indices, they are eliminated.

In the example, we load an 8x8 matrix of type `int4`, stored as row major in global memory to shared memory. Then, each set of eight
threads loads a row from the shared memory buffer and stores it to a column in a separate transpose shared memory buffer. This
results in an eight-way bank conflict when storing. Finally, the transpose buffer is written back to global memory.

To avoid bank conflicts, the `CU_TENSOR_MAP_SWIZZLE_128B` layout can be used. This layout matches the 128 bytes row length and
changes the shared memory layout in a way that both the column wise and row wise access don’t require the same banks per transaction.

The two tables,  [Figure 48](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#figure-swizzle-example1) and  [Figure 49](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#figure-swizzle-example2),  below show the normal and the swizzled shared memory layout of the 8x8 matrix of type `int4` and
its transpose matrix. The colors indicate which of the eight groups of four banks the matrix element is mapped to, and
the margin row and margin column list the global memory row and column indices. The entries show the shared memory
indices of the 16-byte matrix elements.

![The shared memory data layout without swizzle](images/_____-___-__-________-_____-___________-_______1.png)

Figure 48 In the shared memory data layout without swizzle, the shared memory indices are equivalent to the global memory indices.
Per load instruction, one row is read and stored in a column of the transpose buffer. Since all matrix elements of the
column in the transpose fall in the same bank, the store must be serialized, resulting in eight store transactions, giving
an eight-way bank conflict per stored column.[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#id2 "Link to this image")

![The shared memory data layout with CU_TENSOR_MAP_SWIZZLE_128B swizzle.](images/_____-___-__-________-_____-___________-_______2.png)

Figure 49 The shared memory data layout with `CU_TENSOR_MAP_SWIZZLE_128B` swizzle. One row is stored in a column, each matrix
element is from a different bank for both the rows and columns, and so without any bank conflicts.[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#id3 "Link to this image")

```c++
__global__ void kernel_tma(const __grid_constant__ CUtensorMap tensor_map) {
   // The destination shared memory buffer of a bulk tensor operation
   // with the 128-byte swizzle mode, it should be 1024 bytes aligned.
   __shared__ alignas(1024) int4 smem_buffer[8][8];
   __shared__ alignas(1024) int4 smem_buffer_tr[8][8];

   // Initialize shared memory barrier
   #pragma nv_diag_suppress static_var_with_dynamic_init
   __shared__ barrier bar;

   if (threadIdx.x == 0) {
     init(&bar, blockDim.x);
   }
   __syncthreads();

   barrier::arrival_token token;
   if (is_elected()) {
     // Initiate bulk tensor copy from global to shared memory,
     // in the same way as without swizzle.
     int32_t tensor_coords[2] = { 0, 0 };
     ptx::cp_async_bulk_tensor(
       ptx::space_shared, ptx::space_global,
       &smem_buffer, &tensor_map, tensor_coords,
       cuda::device::barrier_native_handle(bar));
     token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
   } else {
     token = bar.arrive();
   }

   bar.wait(std::move(token));

   /* Matrix transpose
    *  When using the normal shared memory layout, there are eight
    *  8-way shared memory bank conflict when storing to the transpose.
    *  When enabling the 128-byte swizzle pattern and using the according access pattern,
    *  they are eliminated both for load and store. */
   for(int sidx_j =threadIdx.x; sidx_j < 8; sidx_j+= blockDim.x){
      for(int sidx_i = 0; sidx_i < 8; ++sidx_i){
         const int swiz_j_idx = (sidx_i % 8) ^ sidx_j;
         const int swiz_i_idx_tr = (sidx_j % 8) ^ sidx_i;
         smem_buffer_tr[sidx_j][swiz_i_idx_tr] = smem_buffer[sidx_i][swiz_j_idx];
      }
   }

   // Wait for shared memory writes to be visible to TMA engine.
   ptx::fence_proxy_async(ptx::space_shared);
   __syncthreads();

   /* Initiate TMA transfer to copy the transposed shared memory buffer back to global memory,
    * it will 'unswizzle' the data. */
   if (is_elected()) {
       int32_t tensor_coords[2] = { x, y };
       ptx::cp_async_bulk_tensor(
         ptx::space_global, ptx::space_shared,
         &tensor_map, tensor_coords, &smem_buffer_tr);
      ptx::cp_async_bulk_commit_group();
      ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
   }

   // Destroy barrier
   if (threadIdx.x == 0) {
     (&bar)->~barrier();
   }
}

// --------------------------------- main ----------------------------------------

int main(){

...
   void* tensor_ptr = d_data;

   CUtensorMap tensor_map{};
   // rank is the number of dimensions of the array.
   constexpr uint32_t rank = 2;
   // global memory size
   uint64_t size[rank] = {4*8, 8};
   // global memory stride, must be a multiple of 16.
   uint64_t stride[rank - 1] = {8 * sizeof(int4)};
   // The inner shared memory box dimension in bytes, equal to the swizzle span.
   uint32_t box_size[rank] = {4*8, 8};

   uint32_t elem_stride[rank] = {1, 1};

   // Create the tensor descriptor.
   CUresult res = cuTensorMapEncodeTiled(
       &tensor_map,                // CUtensorMap *tensorMap,
       CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
       rank,                       // cuuint32_t tensorRank,
       tensor_ptr,                 // void *globalAddress,
       size,                       // const cuuint64_t *globalDim,
       stride,                     // const cuuint64_t *globalStrides,
       box_size,                   // const cuuint32_t *boxDim,
       elem_stride,                // const cuuint32_t *elementStrides,
       CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
       // Using a swizzle pattern of 128 bytes.
       CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
       CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
       CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
   );

   kernel_tma<<<1, 8>>>(tensor_map);
 ...
}
```

**Remark.** This example is supposed to show the use of swizzle and ‘as-is’ is not performant nor does it scale beyond the given dimensions.

**Explanation.** During data transfer, the TMA engine shuffles the data according to the swizzle pattern, as described in the following
tables. These swizzle patterns define the mapping of the 16-byte chunks along the swizzle width to subgroups of four banks.
It is of type `CUtensorMapSwizzle` and has four options: none, 32 bytes, 64 bytes and 128 bytes. Note that the shared memory box’s
inner dimension must be less or equal to the span of the swizzle pattern.

**The Swizzle Modes**

As previously mentioned, there are four swizzle modes. The following tables show the different swizzle patterns, including the relation of the new
shared memory indices. The tables define the mapping of the 16-byte chunks along the 128 bytes to eight subgroups of four banks.

![An Overview of TMA Swizzle Patterns](images/_____-___-__-________-_____-___________-_______3.png)

Figure 50 An Overview of TMA Swizzle Patterns[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#id4 "Link to this image")

**Considerations.**  When applying a TMA swizzle pattern, it is crucial to adhere to specific memory requirements:

- **Global memory alignment:**
Global memory must be aligned to 128 bytes.
- **Shared memory alignment:**
For simplicity shared memory should be aligned according to the number of bytes after which the swizzle pattern repeats. When the shared memory buffer is not aligned by the number of bytes by which the swizzle pattern repeats itself, there is an offset between the swizzle pattern and the shared memory.
See [comment](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#swizzle-pattern-pointer-offset-computation), below.
- **Inner dimension:**
The inner dimension of the shared memory block must meet the size requirements specified in [Table 25](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#table-swizzle-pattern-properties-and-requirements). If these
requirements are not met, the instruction is considered invalid. Additionally, if the swizzle width exceeds the inner dimension,
ensure that the shared memory is allocated to accommodate the full swizzle width.
- **Granularity:**
The granularity of swizzle mapping is fixed at 16 bytes. This means that data is organized and accessed in chunks
of 16 bytes, which must be considered when planning memory layout and access patterns.

**Swizzle Pattern Pointer Offset Computation**. Here, we describe how to determine the offset between the swizzle pattern and the shared memory, when the shared memory buffer is not aligned by the number of bytes by which the swizzle pattern repeats itself.
When using TMA, the shared memory is required to be aligned to 128 bytes. To find how many times the shared memory buffer relative to the swizzle pattern is shifted by that, apply the corresponding offset formula.

| Swizzle Mode | Offset Formula | Index Relation |
| --- | --- | --- |
| CU_TENSOR_MAP_SWIZZLE_128B | `(reinterpret_cast <uintptr_t>(smem_ptr)/128)%8` | `smem[y][x] <-> smem[y][((y+offset)%8)^x]` |
| CU_TENSOR_MAP_SWIZZLE_64B | `(reinterpret_cast <uintptr_t>(smem_ptr)/128)%4` | `smem[y][x] <-> smem[y][((y+offset)%4)^x]` |
| CU_TENSOR_MAP_SWIZZLE_32B | `(reinterpret_cast <uintptr_t>(smem_ptr)/128)%2` | `smem[y][x] <-> smem[y][((y+offset)%2)^x]` |

In [Figure 50](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#figure-swizzle-overview), this offset represents the initial row offset, thus, in the swizzle index calculation, it is added to the row index `y`.
The following snippet shows how to access the swizzled shared memory in the `CU_TENSOR_MAP_SWIZZLE_128B` mode.

```c++
data_t* smem_ptr = &smem[0][0];
int offset = (reinterpret_cast<uintptr_t>(smem_ptr)/128)%8;
smem[y][((y+offset)%8)^x] = ...
```

**Summary.** The following [Table 25](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#table-swizzle-pattern-properties-and-requirements) summarizes the requirements and properties of the different swizzle patterns for Compute Capability 9.

| Pattern | Swizzle width | Shared box’s inner dimension | Repeats after | Shared memory alignment | Global memory alignment |
| --- | --- | --- | --- | --- | --- |
| CU_TENSOR_MAP_SWIZZLE_128B | 128 bytes | <=128 bytes | 1024 bytes | 128 bytes | 128 bytes |
| CU_TENSOR_MAP_SWIZZLE_64B | 64 bytes | <=64 bytes | 512 bytes | 128 bytes | 128 bytes |
| CU_TENSOR_MAP_SWIZZLE_32B | 32 bytes | <=32 bytes | 256 bytes | 128 bytes | 128 bytes |
| CU_TENSOR_MAP_SWIZZLE_NONE (default) |  |  |  | 128 bytes | 16 bytes |
