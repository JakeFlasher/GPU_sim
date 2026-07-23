---
title: "4.3.3.4. Enabling Memory Pools for IPC"
section: "4.3.3.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#enabling-memory-pools-for-ipc"
---

### [4.3.3.4. Enabling Memory Pools for IPC](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#enabling-memory-pools-for-ipc)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#enabling-memory-pools-for-ipc "Permalink to this headline")

Memory pools can be enabled for interprocess communication (IPC) to
allow easy, efficient and secure sharing of GPU
memory between processes. CUDA’s IPC memory pools provide the same security
benefits as CUDA’s [virtual memory management APIs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#virtual-memory-management).

There are two steps to sharing memory between processes with memory pools:
the processes first needs to share access to the pool, then share specific
allocations from that pool. The first step establishes and enforces security.
The second step coordinates what virtual addresses are used in each process
and when mappings need to be valid in the importing process.

#### [4.3.3.4.1. Creating and Sharing IPC Memory Pools](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#creating-and-sharing-ipc-memory-pools)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#creating-and-sharing-ipc-memory-pools "Permalink to this headline")

Sharing access to a pool involves retrieving an OS-native handle to the pool
with  `cudaMemPoolExportToShareableHandle()`, transferring the
handle to the importing process using OS-native IPC mechanisms, and then
creating an imported memory pool with the
`cudaMemPoolImportFromShareableHandle()` API. For
`cudaMemPoolExportToShareableHandle` to succeed, the memory pool must have been
created with the requested handle type specified in the pool properties
structure.

Please reference [samples](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/streamOrderedAllocationIPC)
for the appropriate IPC mechanisms to
transfer the OS-native handle between processes. The rest of the procedure can
be found in the following code snippets.

```c++
// in exporting process
// create an exportable IPC capable pool on device 0
cudaMemPoolProps poolProps = { };
poolProps.allocType = cudaMemAllocationTypePinned;
poolProps.location.id = 0;
poolProps.location.type = cudaMemLocationTypeDevice;

// Setting handleTypes to a non zero value will make the pool exportable (IPC capable)
poolProps.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

cudaMemPoolCreate(&memPool, &poolProps));

// FD based handles are integer types
int fdHandle = 0;

// Retrieve an OS native handle to the pool.
// Note that a pointer to the handle memory is passed in here.
cudaMemPoolExportToShareableHandle(&fdHandle,
             memPool,
             CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
             0);

// The handle must be sent to the importing process with the appropriate
// OS-specific APIs.
```

```c++
// in importing process
 int fdHandle;
// The handle needs to be retrieved from the exporting process with the
// appropriate OS-specific APIs.
// Create an imported pool from the shareable handle.
// Note that the handle is passed by value here.
cudaMemPoolImportFromShareableHandle(&importedMemPool,
          (void*)fdHandle,
          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
          0);
```

#### [4.3.3.4.2. Set Access in the Importing Process](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#set-access-in-the-importing-process)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#set-access-in-the-importing-process "Permalink to this headline")

Imported memory pools are initially only accessible from their resident
device. The imported memory pool does not inherit any accessibility set by the
exporting process. The importing process needs to enable access with
`cudaMemPoolSetAccess` from any GPU it plans to access the memory from.

If the imported memory pool belongs to a device that is not visible to importing
process, the user must use the `cudaMemPoolSetAccess` API to enable access
from the GPUs the allocations will be used on.
(See [Device Accessibility for Multi-GPU Support](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#stream-ordered-deviceaccessibility))

#### [4.3.3.4.3. Creating and Sharing Allocations from an Exported Pool](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#creating-and-sharing-allocations-from-an-exported-pool)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#creating-and-sharing-allocations-from-an-exported-pool "Permalink to this headline")

Once the pool has been shared, allocations made with `cudaMallocAsync()`
from the pool in the exporting process can be shared with processes that
have imported the pool. Since the pool’s security policy is established and
verified at the pool level, the OS does not need extra bookkeeping to provide
security for specific pool allocations. In other words, the opaque
`cudaMemPoolPtrExportData` required to import a pool allocation may be sent
to the importing process using any mechanism.

While allocations may be exported and imported without synchronizing with
the allocating stream in any way, the importing process must follow the same
rules as the exporting process when accessing the allocation. Specifically, access
to the allocation must happen after the allocation
operation in the allocating stream executes. The two following code snippets show
`cudaMemPoolExportPointer()` and `cudaMemPoolImportPointer()` sharing the
allocation with an IPC event used to guarantee that the allocation isn’t
accessed in the importing process before the allocation is ready.

```c++
// preparing an allocation in the exporting process
cudaMemPoolPtrExportData exportData;
cudaEvent_t readyIpcEvent;
cudaIpcEventHandle_t readyIpcEventHandle;

// ipc event for coordinating between processes
// cudaEventInterprocess flag makes the event an ipc event
// cudaEventDisableTiming  is set for performance reasons

cudaEventCreate(&readyIpcEvent, cudaEventDisableTiming | cudaEventInterprocess)

// allocate from the exporting mem pool
cudaMallocAsync(&ptr, size,exportMemPool, stream);

// event for sharing when the allocation is ready.
cudaEventRecord(readyIpcEvent, stream);
cudaMemPoolExportPointer(&exportData, ptr);
cudaIpcGetEventHandle(&readyIpcEventHandle, readyIpcEvent);

// Share IPC event and pointer export data with the importing process using
//  any mechanism. Here we copy the data into shared memory
shmem->ptrData = exportData;
shmem->readyIpcEventHandle = readyIpcEventHandle;
// signal consumers data is ready
```

```c++
// Importing an allocation
cudaMemPoolPtrExportData *importData = &shmem->prtData;
cudaEvent_t readyIpcEvent;
cudaIpcEventHandle_t *readyIpcEventHandle = &shmem->readyIpcEventHandle;

// Need to retrieve the ipc event handle and the export data from the
// exporting process using any mechanism.  Here we are using shmem and just
// need synchronization to make sure the shared memory is filled in.

cudaIpcOpenEventHandle(&readyIpcEvent, readyIpcEventHandle);

// import the allocation. The operation does not block on the allocation being ready.
cudaMemPoolImportPointer(&ptr, importedMemPool, importData);

// Wait for the prior stream operations in the allocating stream to complete before
// using the allocation in the importing process.
cudaStreamWaitEvent(stream, readyIpcEvent);
kernel<<<..., stream>>>(ptr, ...);
```

When freeing the allocation, the allocation must be freed in the importing
process before it is freed in the exporting process. The following code
snippet demonstrates the use of CUDA IPC events to provide the required
synchronization between the `cudaFreeAsync` operations in both processes.
Access to the allocation from the importing process is obviously restricted by
the free operation in the importing process side. It is worth noting that
`cudaFree` can be used to free the allocation in both processes and that
other stream synchronization APIs may be used instead of CUDA IPC events.

```c++
// The free must happen in importing process before the exporting process
kernel<<<..., stream>>>(ptr, ...);

// Last access in importing process
cudaFreeAsync(ptr, stream);

// Access not allowed in the importing process after the free
cudaIpcEventRecord(finishedIpcEvent, stream);
```

```c++
// Exporting process
// The exporting process needs to coordinate its free with the stream order
// of the importing process’s free.
cudaStreamWaitEvent(stream, finishedIpcEvent);
kernel<<<..., stream>>>(ptrInExportingProcess, ...);

// The free in the importing process doesn’t stop the exporting process
// from using the allocation.
cudFreeAsync(ptrInExportingProcess,stream);
```

#### [4.3.3.4.4. IPC Export Pool Limitations](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#ipc-export-pool-limitations)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#ipc-export-pool-limitations "Permalink to this headline")

IPC pools currently do not support releasing physical blocks back to the OS.
As a result the `cudaMemPoolTrimTo` API has no effect and the
`cudaMemPoolAttrReleaseThreshold` is effectively ignored. This behavior is
controlled by the driver, not the runtime and may change in a future driver
update.

#### [4.3.3.4.5. IPC Import Pool Limitations](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#ipc-import-pool-limitations)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#ipc-import-pool-limitations "Permalink to this headline")

Allocating from an import pool is not allowed; specifically, import pools
cannot be set current and cannot be used in the `cudaMallocFromPoolAsync`
API. As such, the allocation reuse policy attributes do not have meaning for these
pools.

IPC Import pools, like IPC export pools, currently do not support releasing physical blocks back to the OS.

The resource usage stat attribute queries only reflect the allocations
imported into the process and the associated physical memory.
