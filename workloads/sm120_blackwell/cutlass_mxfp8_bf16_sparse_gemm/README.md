# CUTLASS SM120 sparse MXFP8→BF16 GEMM

Copied from NVIDIA CUTLASS 4.8 example 80a at commit
`0b55a2f691d69981583568fd9eb69687b1f0de8a`. This is a separate SM120-only
sparse block-scaled MMA workload; it must not be compiled or relabeled as SM86.
Build with CUDA 13.2, `/usr/bin/g++-14`, `-arch=sm_120a`, and the repository
CUTLASS include tree.
