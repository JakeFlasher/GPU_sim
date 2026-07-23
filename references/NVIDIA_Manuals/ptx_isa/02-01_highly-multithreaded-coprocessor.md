---
title: "2.1. A Highly Multithreaded Coprocessor"
section: "2.1"
source: "https://docs.nvidia.com/cuda/parallel-thread-execution/#highly-multithreaded-coprocessor"
---

## [2.1. A Highly Multithreaded Coprocessor](https://docs.nvidia.com/cuda/parallel-thread-execution#highly-multithreaded-coprocessor)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#highly-multithreaded-coprocessor "Permalink to this headline")

The GPU is a compute device capable of executing a very large number of threads in parallel. It
operates as a coprocessor to the main CPU, or host: In other words, data-parallel, compute-intensive
portions of applications running on the host are off-loaded onto the device.

More precisely, a portion of an application that is executed many times, but independently on
different data, can be isolated into a kernel function that is executed on the GPU as many different
threads. To that effect, such a function is compiled to the PTX instruction set and the resulting
kernel is translated at install time to the target GPU instruction set.
