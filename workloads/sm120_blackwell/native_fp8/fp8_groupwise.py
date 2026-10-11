"""Native SM120 E4M3 groupwise GEMM for retained SOL quantized workloads.

Only the SOL 1x128 activation / 128x128 weight recipe is supported here.
No dequantize-to-FP32 GEMM fallback. CUTLASS accumulates/scales in FP32 and
the adapter adds bias before the requested output cast.
"""
import ctypes
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
LIBRARY = ROOT / "libsm120_fp8_groupwise.so"
NATIVE_SOURCE_FILES = [ROOT / "fp8_groupwise.cu", ROOT / "Makefile", Path(__file__).resolve(), LIBRARY]
_lib = ctypes.CDLL(str(LIBRARY))
_lib.sm120_fp8_groupwise.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
_lib.sm120_fp8_groupwise.restype = ctypes.c_int
_lib.sm120_fp8_error.restype = ctypes.c_char_p


class NativeGroupwiseGemm:
    def scaled_mm(self, mat_a, mat_b, scale_a, scale_recipe_a, scale_b,
                  scale_recipe_b, bias=None, output_dtype=torch.bfloat16,
                  use_fast_accum=True):
        if str(scale_recipe_a) != "BlockWise1x128" or str(scale_recipe_b) != "BlockWise128x128":
            raise ValueError("native SM120 bridge requires 1x128 A and 128x128 B scale blocks")
        if mat_a.ndim != 2 or mat_b.ndim != 2:
            raise ValueError("expected A[M,K] and B[N,K]")
        m, k = mat_a.shape
        n, kb = mat_b.shape
        if kb != k or min(m, n, k) <= 0 or k % 128 or n % 128:
            raise ValueError("require matching K and N/K multiples of 128")
        if tuple(scale_a.shape) != (m, k // 128) or tuple(scale_b.shape) != (n // 128, k // 128):
            raise ValueError("scale tensor shape does not match the declared block recipe")
        if mat_a.dtype != torch.float8_e4m3fn or mat_b.dtype != torch.float8_e4m3fn:
            raise TypeError("native SM120 bridge requires E4M3 input tensors")
        tensors = (mat_a, mat_b, scale_a, scale_b)
        if not mat_a.is_cuda or any(t.device != mat_a.device for t in tensors):
            raise ValueError("all operands must be on the same CUDA device")
        if torch.cuda.get_device_capability(mat_a.device) != (12, 0):
            raise RuntimeError("this bridge is compiled specifically for SM120")
        with torch.cuda.device(mat_a.device):
            a, b = mat_a.contiguous(), mat_b.contiguous()
            # CUTLASS's MN-major scale layout has outer M/N contiguous and
            # K-block as the slow dimension, unlike the SOL row-major scales.
            sa = scale_a.float().T.contiguous()
            sb = scale_b.float().T.contiguous()
            out = torch.empty((m, n), device=a.device, dtype=torch.float32)
            stream = torch.cuda.current_stream(a.device).cuda_stream
            rc = _lib.sm120_fp8_groupwise(a.data_ptr(), b.data_ptr(), sa.data_ptr(),
                                         sb.data_ptr(), out.data_ptr(), m, n, k, stream)
            if rc:
                raise RuntimeError(_lib.sm120_fp8_error().decode())
            if bias is not None and bias.numel():
                out = out + bias
            return out.to(output_dtype)
