---
title: "Pseudo Code"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html#pseudo-code"
---

### [Pseudo Code](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#pseudo-code)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#pseudo-code "Permalink to this headline")

#### [Non-persistent kernel](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#non-persistent-kernel)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#non-persistent-kernel "Permalink to this headline")

```c++
// Non-persistent kernel
__device__ non_persistent_kernel(...) {
  setup_common_data_structures();
  dim3 workCoordinates = blockIdx;
  coordinate_specific_compute(workCoordinates);
}
```

#### [Static Persistent Kernel](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#static-persistent-kernel)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#static-persistent-kernel "Permalink to this headline")

```c++
// Static Persistent Kernel
__device__ static_persistent_kernel(...) {
  setup_common_data_structures(...);
  dim3 workCoordinates = blockIdx;
  bool isValidId;
  do {
    coordinate_specific_compute(workCoordinates);
    std::tie(isValidId, workCoordinates) = staticTileScheduler.fetch_next_work();
  } while (isValidId);
}
```

#### [Blackwell Dynamic Persistent Kernel](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#blackwell-dynamic-persistent-kernel)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#blackwell-dynamic-persistent-kernel "Permalink to this headline")

```c++
// Dynamic Persistent Kernel
__device__ clc_dynamic_persistent_kernel(...) {
  setup_common_data_structures(...);
  dim3 workCoordinates = blockIdx;
  dim3 newClcID;
  bool isValidId;
  do {
    coordinate_specific_compute(workCoordinates);
    std::tie(isValidId, newClcID) = clcTileScheduler.fetch_next_work();
    workCoordinates = newClcID;
  } while (isValidId);
}
```
