---
title: "2.2. NVCC Phases"
section: "2.2"
source: "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvcc-phases"
---

## [2.2. NVCC Phases](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#nvcc-phases)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvcc-phases "Permalink to this headline")

A compilation phase is a logical translation step that can be selected by command line options to `nvcc`. A single compilation phase can still be broken up by `nvcc` into smaller steps, but these smaller steps are just implementations of the phase: they depend on seemingly arbitrary capabilities of the internal tools that `nvcc` uses, and all of these internals may change with a new release of the CUDA Toolkit. Hence, only compilation phases are stable across releases, and although `nvcc` provides options to display the compilation steps that it executes, these are for debugging purposes only and must not be copied and used in build scripts.

`nvcc` phases are selected by a combination of command line options and input file name suffixes, and the execution of these phases may be modified by other command line options. In phase selection, the input file suffix defines the phase input, while the command line option defines the required output of the phase.

The following paragraphs list the recognized file name suffixes and the supported compilation phases. A full explanation of the `nvcc` command line options can be found in [NVCC Command Options](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvcc-command-options).
