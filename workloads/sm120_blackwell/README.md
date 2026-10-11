# SM120 GeForce-only workloads

These workloads are separate from SM86 and SM100. They use CUDA 13.2, G++14,
and CUTLASS GeForce examples 79/80/87 with `Sm120` warp-level/block-scaled
MMA and cluster shape 1×1×1. The CUTLASS controls are:

* `cutlass_nvfp4_bf16_gemm` — example 79a NVFP4→BF16;
* `cutlass_mxfp8_bf16_sparse_gemm` — example 80a sparse MXFP8→BF16;
* `cutlass_fp8_bf16_blockwise_gemm` — example 87a blockwise FP8→BF16;
* `cutlass_mxfp8_mxfp6_bf16_gemm` — example 79c mixed MXFP8/MXFP6→BF16.
* `cutlass_nvfp4_nvfp4_sparse_gemm` — example 80b sparse NVFP4→NVFP4.
* `cutlass_nvfp4_grouped_gemm` — example 79d grouped NVFP4 with cooperative
  and pingpong schedules; heterogeneous benchmark files exercise device-side
  TMA descriptor updates between groups.

`sol_adaptations/` contains tracked SM120 source copies for bounded SOL rows
whose original fixed geometry does not fit an instrumented 5090 launch.  The
adapters retain the original operation (including 3-D RoPE, gated Mamba
convolution, and attention backward) and record their actual axes beside each
capture.

`flashinfer_adaptations/runner.py` is a vectorized ATen reference control for
the SOL/FlashInfer rows whose Python reference emits unbounded per-head loops.
Its reduced axes are explicit command-line adaptations; its traces must not be
described as native FlashInfer fused-kernel traces.

`native_fp8/` provides the CUTLASS87b SM120A groupwise FP8 bridge for
Quant/001–007, 010–013, 015–017, and 021–023. Activations use 1x128 scales;
weights use 128x128 scales. The bridge transposes scale buffers into MN-major
CUTLASS layout and executes QMMA with TMA, not FP32 dequantize/matmul.
Older fallback reference files remain distinct from these native adapters.
NVFP4 adapters retain E2M1 packed operands and block16 E4M3 scales through
PyTorch `_scaled_mm`; run metadata records the actual prebuilt Torch/CUDA
library version separately from the CUDA13.2/GCC14 tools used here.

Current collection history is in the cloud workspace's
`metadata/sm120_adapted_rows_20261007.tsv` and summary JSON. Counts include
retained reference controls and superseded captures; they are not a claim
that every source family or shape is finished. Generic descriptor loads use
the current runtime state-space resolver; the earlier L1/003 failure is not
the current capture.

The SOL reference runner propagates `--axis` overrides into scalar inputs as
well as tensor shapes. This is required for scalar-driven references such as
L1/028 hybrid masks, where stale scalar values otherwise make the trace's
recorded geometry differ from the actual CUDA launch.

The SimBench routed NVFP4 grouped-MoE replacement now emits native SM120A
`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` from the block16 warp MMA path. The
former BF16 stand-in remains in retained artifacts but is not native evidence.

Additional tracked geometry adapters now cover L1/079 and L1/082 fixed GEMM
surfaces and L2/042 FFN backward; each reduces only declared geometry while
retaining the operation graph and records the reduced axes beside the pair.

Quant/004 has a native CUTLASS 87b groupwise bridge under
`native_fp8/`; the bridge uses SM120 TMA warp-specialized blockwise scaling,
not the former FP32 dequantize fallback. Both operands-off and operands-on
captures are retained.


## ISA research materials

The user's `/root/nvidia_gpu_ISA_reverse_engineering.md` is the local research
index. `isa_tools/collect_nvdisasm_metadata.sh OUTPUT_DIR` compiles a tiny
SM120 cubin with CUDA13.2/G++14, compiles the retained DocumentSASS shim with
GCC14, and captures **binary** memcpy payloads from nvdisasm. The accompanying
parser reconstructs scheduling tables without assuming an ARCHITECTURE header.
It neither provides a complete SM120 assembler nor measures hardware latency.
