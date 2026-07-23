---
title: "5.3. Application Compatibility"
section: "5.3"
source: "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#application-compatibility"
---

## [5.3. Application Compatibility](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#application-compatibility)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#application-compatibility "Permalink to this headline")

Binary code compatibility over CPU generations, together with a published instruction set architecture is the usual mechanism for ensuring that distributed applications _out there in the field_ will continue to run on newer versions of the CPU when these become mainstream.

This situation is different for GPUs, because NVIDIA cannot guarantee binary compatibility without sacrificing regular opportunities for GPU improvements. Rather, as is already conventional in the graphics programming domain, `nvcc` relies on a two stage compilation model for ensuring application compatibility with future GPU generations.
