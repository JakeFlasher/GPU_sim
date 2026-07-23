---
title: "5.6.3.3. cooperative_groups/partition.h"
section: "5.6.3.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#cooperative-groups-partition-h"
---

### [5.6.3.3. cooperative_groups/partition.h](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cooperative-groups-partition-h)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cooperative-groups-partition-h "Permalink to this headline")

#### [5.6.3.3.1. tiled_partition](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#tiled-partition)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#tiled-partition "Permalink to this headline")

```c++
template <unsigned int Size, typename ParentT>
thread_block_tile<Size, ParentT> tiled_partition(const ParentT& g);
```

```c++
thread_group tiled_partition(const thread_group& parent, unsigned int tilesz);
```

The `tiled_partition` method is a collective operation that partitions the parent group into a one-dimensional, row-major, tiling of subgroups. A total of ((size(parent)/tilesz) subgroups will be created, therefore the parent group size must be evenly divisible by the `Size`. The allowed parent groups are `thread_block` or `thread_block_tile`.

The implementation may cause the calling thread to wait until all the members of the parent group have invoked the operation before resuming execution. Functionality is limited to native hardware sizes, 1/2/4/8/16/32 and the `cg::size(parent)` must be greater than the `Size` parameter. The templated version of `tiled_partition` supports 64/128/256/512 sizes as well, but some additional steps are required on Compute Capability 7.5 or lower, refer to [class thread_block_tile](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cg-api-thread-block-tile) for details.

**Codegen Requirements:** Compute Capability 5.0 minimum, C++11 for sizes larger than 32

#### [5.6.3.3.2. labeled_partition](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#labeled-partition)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#labeled-partition "Permalink to this headline")

```c++
template <typename Label>
coalesced_group labeled_partition(const coalesced_group& g, Label label);
```

```c++
template <unsigned int Size, typename Label>
coalesced_group labeled_partition(const thread_block_tile<Size>& g, Label label);
```

The `labeled_partition` method is a collective operation that partitions the parent group into one-dimensional subgroups within which the threads are coalesced. The implementation will evaluate a condition label and assign threads that have the same value for label into the same group.

`Label` can be any integral type.

The implementation may cause the calling thread to wait until all the members of the parent group have invoked the operation before resuming execution.

**Note:** This functionality is still being evaluated and may slightly change in the future.

**Codegen Requirements:** Compute Capability 7.0 minimum, C++11

#### [5.6.3.3.3. binary_partition](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#binary-partition)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#binary-partition "Permalink to this headline")

```c++
coalesced_group binary_partition(const coalesced_group& g, bool pred);
```

```c++
template <unsigned int Size>
coalesced_group binary_partition(const thread_block_tile<Size>& g, bool pred);
```

The `binary_partition()` method is a collective operation that partitions the parent group into one-dimensional subgroups within which the threads are coalesced. The implementation will evaluate a predicate and assign threads that have the same value into the same group. This is a specialized form of `labeled_partition()`, where the label can only be 0 or 1.

The implementation may cause the calling thread to wait until all the members of the parent group have invoked the operation before resuming execution.

**Example:**

```c++
/// This example divides a 32-sized tile into a group with odd
/// numbers and a group with even numbers
_global__ void oddEven(int *inputArr) {
    auto block = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<32>(block);

    // inputArr contains random integers
    int elem = inputArr[block.thread_rank()];
    // after this, tile32 is split into 2 groups,
    // a subtile where elem&1 is true and one where its false
    auto subtile = cg::binary_partition(tile32, (elem & 1));
}
```
