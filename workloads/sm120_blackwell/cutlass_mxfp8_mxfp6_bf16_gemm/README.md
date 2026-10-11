# CUTLASS SM120 mixed MXFP8/MXFP6 to BF16 GEMM

Directly derived from CUTLASS 4.8 example 79c (`79_blackwell_geforce_gemm`).
This is a GeForce RTX 50-series-only block-scaled Tensor Core workload and is
kept separate from the SM100 `tcgen05`/TMEM paths. Build with CUDA 13.2 and
G++14; collection uses `--m=128 --n=128 --k=128 --iterations=1`.
