---
title: "4.1.1.1. Unified Memory: In-Depth Examples"
section: "4.1.1.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#unified-memory-in-depth-examples"
---

### [4.1.1.1. Unified Memory: In-Depth Examples](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#unified-memory-in-depth-examples)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#unified-memory-in-depth-examples "Permalink to this headline")

Systems with full CUDA unified memory support, see table [Overview of Unified Memory Paradigms](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#table-unified-memory-levels),
allow the device to access any memory owned by the host process interacting with the device.

This section shows a few advanced use-cases, using a kernel that simply prints
the first 8 characters of an input character array to the standard output stream:

```cuda
__global__ void kernel(const char* type, const char* data) {
  static const int n_char = 8;
  printf("%s - first %d characters: '", type, n_char);
  for (int i = 0; i < n_char; ++i) printf("%c", data[i]);
  printf("'\n");
}
```

The following tabs show various ways of how this kernel may be called with system-allocated memory:

**Malloc**

```cuda
void test_malloc() {
  const char test_string[] = "Hello World";
  char* heap_data = (char*)malloc(sizeof(test_string));
  strncpy(heap_data, test_string, sizeof(test_string));
  kernel<<<1, 1>>>("malloc", heap_data);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
  free(heap_data);
}
```

**Managed**

```cuda
void test_managed() {
  const char test_string[] = "Hello World";
  char* data;
  cudaMallocManaged(&data, sizeof(test_string));
  strncpy(data, test_string, sizeof(test_string));
  kernel<<<1, 1>>>("managed", data);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
  cudaFree(data);
}
```

**Stack variable**

```cuda
void test_stack() {
  const char test_string[] = "Hello World";
  kernel<<<1, 1>>>("stack", test_string);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
}
```

**File-scope static variable**

```cuda
void test_static() {
  static const char test_string[] = "Hello World";
  kernel<<<1, 1>>>("static", test_string);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
}
```

**Global-scope variable**

```cuda
const char global_string[] = "Hello World";

void test_global() {
  kernel<<<1, 1>>>("global", global_string);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
}
```

**Global-scope extern variable**

```cuda
// declared in separate file, see below
extern char* ext_data;

void test_extern() {
  kernel<<<1, 1>>>("extern", ext_data);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
}
```

```c++
/** This may be a non-CUDA file */
char* ext_data;
static const char global_string[] = "Hello World";

void __attribute__ ((constructor)) setup(void) {
  ext_data = (char*)malloc(sizeof(global_string));
  strncpy(ext_data, global_string, sizeof(global_string));
}

void __attribute__ ((destructor)) tear_down(void) {
  free(ext_data);
}
```

Note that for the extern variable, it could be declared and its memory owned and managed by a third-party library, which does not interact with CUDA at all.

Also note that stack variables as well as file-scope and global-scope variables can only be accessed through a pointer by the GPU. In this specific example, this is convenient because the character array is already declared as a pointer: `const char*`. However, consider the following example with a global-scope integer:

```c++
// this variable is declared at global scope
int global_variable;

__global__ void kernel_uncompilable() {
  // this causes a compilation error: global (__host__) variables must not
  // be accessed from __device__ / __global__ code
  printf("%d\n", global_variable);
}

// On systems with pageableMemoryAccess set to 1, we can access the address
// of a global variable. The below kernel takes that address as an argument
__global__ void kernel(int* global_variable_addr) {
  printf("%d\n", *global_variable_addr);
}
int main() {
  kernel<<<1, 1>>>(&global_variable);
  ...
  return 0;
}
```

In the example above, we need to ensure to pass a _pointer_ to the global variable to the kernel instead of directly accessing the global variable in the kernel. This is because global variables without the `__managed__` specifier are declared as `__host__`-only by default, thus most compilers won’t allow using these variables directly in device code as of now.

#### [4.1.1.1.1. File-backed Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#file-backed-unified-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#file-backed-unified-memory "Permalink to this headline")

Since systems with full CUDA unified memory support allow the device to access any memory owned by the host process, they can directly access file-backed memory.

Here, we show a modified version of the initial example shown in the previous section to use file-backed memory in order to print a string from the GPU, read directly from an input file. In the following example, the memory is backed by a physical file, but the example applies to memory-backed files too.

```cuda
__global__ void kernel(const char* type, const char* data) {
  static const int n_char = 8;
  printf("%s - first %d characters: '", type, n_char);
  for (int i = 0; i < n_char; ++i) printf("%c", data[i]);
  printf("'\n");
}
```

```cuda
void test_file_backed() {
  int fd = open(INPUT_FILE_NAME, O_RDONLY);
  ASSERT(fd >= 0, "Invalid file handle");
  struct stat file_stat;
  int status = fstat(fd, &file_stat);
  ASSERT(status >= 0, "Invalid file stats");
  char* mapped = (char*)mmap(0, file_stat.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  ASSERT(mapped != MAP_FAILED, "Cannot map file into memory");
  kernel<<<1, 1>>>("file-backed", mapped);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
  ASSERT(munmap(mapped, file_stat.st_size) == 0, "Cannot unmap file");
  ASSERT(close(fd) == 0, "Cannot close file");
}
```

Note that on systems without the `hostNativeAtomicSupported` property (see [Host Native Atomics](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#um-host-native-atomics)) including systems with Linux HMM enabled, atomic accesses to file-backed memory are not supported.

#### [4.1.1.1.2. Inter-Process Communication (IPC) with Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#inter-process-communication-ipc-with-unified-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#inter-process-communication-ipc-with-unified-memory "Permalink to this headline")

> **Note**
>
> As of now, using IPC with unified memory can have significant performance implications.

Many applications prefer to manage one GPU per process, but still need to use unified memory,
for example for over-subscription, and access it from multiple GPUs.

CUDA IPC ( see [Interprocess Communication](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html#interprocess-communication) ) does not support managed memory: handles to this type of memory may not be shared through
any of the mechanisms discussed in this section.
On systems with full CUDA unified memory support, system-allocated memory is IPC capable.
Once access to system-allocated memory has been shared with other processes,
the same programming model applies, similar to [File-backed Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#um-sam-file-backed).

See the following references for more information on various ways of creating
IPC-capable system-allocated memory under Linux:

- [mmap with MAP_SHARED](https://man7.org/linux/man-pages/man2/mmap.2.html)
- [POSIX IPC APIs](https://pubs.opengroup.org/onlinepubs/007904875/functions/shm_open.html)
- [Linux memfd_create](https://man7.org/linux/man-pages/man2/memfd_create.2.html) .

Note that it is not possible to share memory between different hosts and their devices using this technique.
