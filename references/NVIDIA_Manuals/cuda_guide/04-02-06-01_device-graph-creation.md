---
title: "4.2.6.1. Device Graph Creation"
section: "4.2.6.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#device-graph-creation"
---

### [4.2.6.1. Device Graph Creation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#device-graph-creation)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#device-graph-creation "Permalink to this headline")

In order for a graph to be launched from the device, it must be instantiated explicitly for device launch. This is achieved by passing the `cudaGraphInstantiateFlagDeviceLaunch` flag to the `cudaGraphInstantiate()` call. As is the case for host graphs, device graph structure is fixed at time of instantiation and cannot be updated without re-instantiation, and instantiation can only be performed on the host. In order for a graph to be able to be instantiated for device launch, it must adhere to various requirements.

#### [4.2.6.1.1. Device Graph Requirements](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#device-graph-requirements)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#device-graph-requirements "Permalink to this headline")

General requirements:

- The graph’s nodes must all reside on a single device.
- The graph can only contain kernel nodes, memcpy nodes, memset nodes, and child graph nodes.

Kernel nodes:

- Use of CUDA Dynamic Parallelism by kernels in the graph is not permitted.
- Cooperative launches are permitted so long as MPS is not in use.

Memcpy nodes:

- Only copies involving device memory and/or pinned device-mapped host memory are permitted.
- Copies involving CUDA arrays are not permitted.
- Both operands must be accessible from the current device at time of instantiation. Note that the copy operation will be performed from the device on which the graph resides, even if it is targeting memory on another device.

#### [4.2.6.1.2. Device Graph Upload](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#device-graph-upload)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#device-graph-upload "Permalink to this headline")

In order to launch a graph on the device, it must first be uploaded to the device to populate the necessary device resources. This can be achieved in one of two ways.

Firstly, the graph can be uploaded explicitly, either via `cudaGraphUpload()` or by requesting an upload as part of instantiation via `cudaGraphInstantiateWithParams()`.

Alternatively, the graph can first be launched from the host, which will perform this upload step implicitly as part of the launch.

Examples of all three methods can be seen below:

```cuda
// Explicit upload after instantiation
cudaGraphInstantiate(&deviceGraphExec1, deviceGraph1, cudaGraphInstantiateFlagDeviceLaunch);
cudaGraphUpload(deviceGraphExec1, stream);

// Explicit upload as part of instantiation
cudaGraphInstantiateParams instantiateParams = {0};
instantiateParams.flags = cudaGraphInstantiateFlagDeviceLaunch | cudaGraphInstantiateFlagUpload;
instantiateParams.uploadStream = stream;
cudaGraphInstantiateWithParams(&deviceGraphExec2, deviceGraph2, &instantiateParams);

// Implicit upload via host launch
cudaGraphInstantiate(&deviceGraphExec3, deviceGraph3, cudaGraphInstantiateFlagDeviceLaunch);
cudaGraphLaunch(deviceGraphExec3, stream);
```

#### [4.2.6.1.3. Device Graph Update](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#device-graph-update)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#device-graph-update "Permalink to this headline")

Device graphs can only be updated from the host, and must be re-uploaded to the device upon executable graph update in order for the changes to take effect. This can be achieved using the same methods outlined in Section [device-graph-upload](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cuda-graphs-device-graph-upload). Unlike host graphs, launching a device graph from the device while an update is being applied will result in undefined behavior.
