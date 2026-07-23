---
title: "5.4. Virtual Architectures"
section: "5.4"
source: "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#virtual-architectures"
---

## [5.4. Virtual Architectures](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#virtual-architectures)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#virtual-architectures "Permalink to this headline")

GPU compilation is performed via an intermediate representation, PTX, which can be considered as assembly for a virtual GPU architecture. Contrary to an actual graphics processor, such a virtual GPU is defined entirely by the set of capabilities, or features, that it provides to the application. In particular, a virtual GPU architecture provides a (largely) generic instruction set, and binary instruction encoding is a non-issue because PTX programs are always represented in text format.

Hence, a `nvcc` compilation command always uses two architectures: a _virtual_ intermediate architecture, plus a _real_ GPU architecture to specify the intended processor to execute on. For such an `nvcc` command to be valid, the _real_ architecture must be an implementation of the _virtual_ architecture. This is further explained below.

The chosen virtual architecture is more of a statement on the GPU capabilities that the application requires: using a _smaller_ virtual architecture still allows a _wider_ range of actual architectures for the second `nvcc` stage. Conversely, specifying a virtual architecture that provides features unused by the application unnecessarily restricts the set of possible GPUs that can be specified in the second `nvcc` stage.

From this it follows that the virtual architecture should always be chosen as _low_ as possible, thereby maximizing the actual GPUs to run on. The _real_ architecture should be chosen as _high_ as possible (assuming that this always generates better code), but this is only possible with knowledge of the actual GPUs on which the application is expected to run on. As we will see later in the situation of just-in-time compilation, where the driver has this exact knowledge, the runtime GPU is the one on which the program is about to be launched/executed.

![Two-Staged Compilation with Virtual and Real Architectures](images/virtual-architectures_1.png)

Two-Staged Compilation with Virtual and Real Architectures
