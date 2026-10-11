# CUTLASS SM120 blockwise FP8→BF16 GEMM

NVIDIA CUTLASS 4.8 example 87a, commit
`0b55a2f691d69981583568fd9eb69687b1f0de8a`. This is an SM120-only blockwise
FP8 control. Build with CUDA 13.2, G++14, and explicit
`-gencode arch=compute_120a,code=sm_120a`.
