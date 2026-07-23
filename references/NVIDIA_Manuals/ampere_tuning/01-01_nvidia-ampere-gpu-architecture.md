---
title: "1.1. NVIDIA Ampere GPU Architecture"
section: "1.1"
source: "https://docs.nvidia.com/cuda/ampere-tuning-guide/#nvidia-ampere-gpu-architecture"
---

## [1.1. NVIDIA Ampere GPU Architecture](https://docs.nvidia.com/cuda/ampere-tuning-guide#nvidia-ampere-gpu-architecture)[](https://docs.nvidia.com/cuda/ampere-tuning-guide/#nvidia-ampere-gpu-architecture "Permalink to this headline")

The NVIDIA Ampere GPU architecture is NVIDIA’s latest architecture for CUDA compute applications. The NVIDIA Ampere GPU architecture retains and extends the same CUDA programming model provided by previous NVIDIA GPU architectures such as Turing and Volta, and applications that follow the best practices for those architectures should typically see speedups on the NVIDIA A100 GPU without any code changes. This guide summarizes the ways that an application can be fine-tuned to gain additional speedups by leveraging the NVIDIA Ampere GPU architecture’s features.[^1]

For further details on the programming features discussed in this guide, please refer to the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).
