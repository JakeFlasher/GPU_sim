---
title: "Asynchronous pipelines"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/pipeline.html#asynchronous-pipelines"
---

### [Asynchronous pipelines](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#asynchronous-pipelines)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#asynchronous-pipelines "Permalink to this headline")

In order to write a performant GEMM Kernel,
software pipelining is critical to hide the latency of global memory loads.
(Please refer to the
[Efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html#pipelining) document.)
Different threads or groups of threads
may have different roles in the pipeline.
Some are “producers” that load data or perform computations
to satisfy other threads’ input data dependencies.
The same or different threads may be “consumers”
that do other work with those input data dependencies,
once they are satisfied.
Starting with the Hopper architecture,
the presence of hardware-accelerated synchronization instructions
make it possible for “producer” and “consumer” threads
to communicate with each other efficiently
about their data dependencies.

Implementing a persistent GEMM algorithm calls for managing
dozens of different kinds of asynchronously executing operations
that synchronize using multiple barriers organized as a circular list.
This complexity is too much for human programmers to manage by hand.
As a result, we have developed
[asynchronous Pipeline classes](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/pipeline/).
These classes help developers orchestrate a pipeline
of asynchronous producer and consumer threads,
without needing to worry about lower-level hardware details.
These classes serve a similar function as the various
[pipeline abstractions](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/pipeline.html)
in libcu++.

#### [Pipeline methods](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#pipeline-methods)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#pipeline-methods "Permalink to this headline")

##### [Producer acquire](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#producer-acquire)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#producer-acquire "Permalink to this headline")

The `producer_acquire` method is to be used by asynchronous producer threads
before issuing other instructions associated with a particular pipeline stage
(e.g., copy or write).

This is a blocking instruction
which blocks further execution of producer threads
unless the particular stage waiting to be acquired
is released by a consumer.

We say that a pipeline at its start is “empty” if producer threads are free to produce and do not need to wait for a consumer release – that is, if an acquire operation is expected to succeed.  If the pipeline at its start is empty, then we can either skip performing producer acquire operations during the first pass through the pipeline stages, or use the `make_producer_start_state` method.  The latter ensures that the acquire operation will succeed at the start of a pipeline.

##### [Producer commit](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#producer-commit)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#producer-commit "Permalink to this headline")

The `producer_commit` method is to be issued by asynchronous producer threads
after the instructions associated with a particular stage
(e.g., shared memory writes) have completed,
in order to notify the waiting asynchronous consumer threads.
This is a nonblocking instruction.

This API may result in a No-Op in some cases,
if the producer instructions also update the barrier stage associated automatically
(e.g., TMA_based producer threads using the  `PipelineTmaAsync ` class).

##### [Consumer wait](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#consumer-wait)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#consumer-wait "Permalink to this headline")

The `consumer_wait` method is to be used by consumer threads
before consuming data from a particular pipeline stage
which is expected to be produced by producer threads.

This is a blocking instruction.  That is,
until the producer threads have committed to a particular stage,
this instruction is expected to block further execution of consumer threads.

##### [Consumer release](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#consumer-release)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#consumer-release "Permalink to this headline")

The `consumer_release` method is to be used by consumer threads
to signal waiting producer threads that they have finished consuming data
associated with a particular stage of the pipeline.
This is a nonblocking instruction.

#### [Pipeline example](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#pipeline-example)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#pipeline-example "Permalink to this headline")

```c++
// 4-stage Pipeline
static constexpr int NumStages = 4;
using MainloopPipeline = typename cutlass::PipelineAsync<NumStages>;
using PipelineState = typename cutlass::PipelineState<NumStages>;

// 2 producer threads and 1 consumer thread
typename MainloopPipeline::Params params;
params.producer_arv_count = 2;
params.consumer_arv_count = 1;
MainloopPipeline pipeline(shared_storage.storage, params);

// Producer threads
if (thread_idx == 0 or thread_idx == 1) {
  PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
  for ( ; iter > 0; --iter) {
    pipeline.producer_acquire(smem_pipe_write);

    // Producer ops
    // If any memory operations are involved, then we also need
    // to guarantee that writes are completed and visible to consumer(s).

    pipeline.producer_commit(smem_pipe_write);
    ++smem_pipe_write;
  }
}
else if (thread_idx == 2) {
  PipelineState smem_pipe_read;
  for (; iter > 0; --iter) {
    pipeline.consumer_wait(smem_pipe_read);

    // Consumer ops

    pipeline.consumer_release(smem_pipe_read);
    ++smem_pipe_read;
  }
}
```

In this example, we create an instance of the asynchronous pipeline class `PipelineSync`,
and then synchronize among 3 asynchronously executing threads:
2 producer threads and 1 consumer thread.

Please note that this is a basic example.
There are different versions possible,
depending on what the producer and consumer threads are doing.
Please refer to our [unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/pipeline)
and the other [pipeline classes](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/pipeline/pipeline.hpp)
for more details.
