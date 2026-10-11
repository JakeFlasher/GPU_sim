# CUTLASS SM120 sparse NVFP4→NVFP4 GEMM

GeForce-only CUTLASS 4.8 example 80b. This probes sparse block-scaled NVFP4
MMA and TMA paths on SM120; it is not an SM100 `tcgen05`/TMEM workload. Build
with CUDA 13.2/G++14 and collect with `--m=128 --n=128 --k=128 --iterations=1`.
