import torch
# --- inlined fp8_reference ---
import torch

from enum import StrEnum


class ScalingType(StrEnum):
    """
    Enum for different FP8 scaling strategies.

    Scaling types:
    - TensorWise: Global per-tensor scaling (no blocks)
    - RowWise: Per-row scaling (1 scale per row)
    - BlockWise1x16: 1x16 blocks (per-tensor in M, 16-sized blocks in K)
    - BlockWise1x32: 1x32 blocks (per-tensor in M, 32-sized blocks in K)
    - BlockWise1x128: 1x128 blocks (per-tensor in M, 128-sized blocks in K)
    - BlockWise128x128: 128x128 blocks (blockwise in both dimensions)
    """

    TensorWise = "TensorWise"
    RowWise = "RowWise"
    BlockWise1x16 = "BlockWise1x16"
    BlockWise1x32 = "BlockWise1x32"
    BlockWise1x128 = "BlockWise1x128"
    BlockWise128x128 = "BlockWise128x128"

    @property
    def shape(self) -> tuple[int, int]:
        return {
            ScalingType.TensorWise: (None, None),
            ScalingType.RowWise: (1, None),
            ScalingType.BlockWise1x16: (1, 16),
            ScalingType.BlockWise1x32: (1, 32),
            ScalingType.BlockWise1x128: (1, 128),
            ScalingType.BlockWise128x128: (128, 128),
        }[self]


class BlockwiseScaler:
    """
    Compute and apply scales for FP8 tensors.

    Supports various scaling strategies via ScalingType enum:
    - TensorWise: Global per-tensor scaling
    - RowWise: Per-row scaling
    - BlockWise1x16/32/128: Rectangular blocks
    - BlockWise128x128: Square blocks
    """

    E4M3_MAX = 448.0  # Maximum representable value in E4M3

    def __init__(self, scaling_type: ScalingType):
        """
        Initialize BlockwiseScaler with a specific scaling strategy.

        Args:
            scaling_type: ScalingType enum value
                Examples:
                - ScalingType.TensorWise -> global per-tensor scaling
                - ScalingType.RowWise -> per-row scaling (1 scale per row)
                - ScalingType.BlockWise1x128 -> 1x128 blocks
                - ScalingType.BlockWise128x128 -> 128x128 blocks
        """
        self.scaling_type = scaling_type
        self.shape = self.scaling_type.shape

        # Map enum to block dimensions (M, K)
        scaling_map = {
            ScalingType.TensorWise: (None, None),  # No blocking
            ScalingType.RowWise: (1, None),  # Per-row, full K dimension
            ScalingType.BlockWise1x16: (1, 16),
            ScalingType.BlockWise1x32: (1, 32),
            ScalingType.BlockWise1x128: (1, 128),
            ScalingType.BlockWise128x128: (128, 128),
        }

        self.block_size_m, self.block_size_k = scaling_map[scaling_type]

        # Keep for backward compatibility (use first dimension if available)
        self.block_size = self.block_size_m if self.block_size_m else None

    def compute_scales(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute scale factors based on the scaling type.

        This is a unified method that handles all scaling types:
        - TensorWise: Returns scalar tensor
        - RowWise: Returns (M,) tensor
        - BlockWise*: Returns (M//block_size_m, K//block_size_k) tensor

        Args:
            tensor: Input tensor (typically M, K for 2D)

        Returns:
            Scale tensor with shape depending on scaling type.
            These are inverse scales (amax / dtype_max) used for dequantization.
        """
        if self.scaling_type == ScalingType.TensorWise:
            # Global per-tensor scaling
            amax = torch.max(torch.abs(tensor)).clamp(min=1e-12)
            return amax / self.E4M3_MAX

        M, K = tensor.shape

        if self.scaling_type == ScalingType.RowWise:
            # Per-row scaling: (M, K) -> (M,)
            row_max = tensor.abs().amax(dim=1)
            scales = row_max / self.E4M3_MAX
            return torch.clamp(scales, min=1e-12)

        # BlockWise scaling
        assert M % self.block_size_m == 0, (
            f"M={M} must be a multiple of {self.block_size_m}"
        )
        assert K % self.block_size_k == 0, (
            f"K={K} must be a multiple of {self.block_size_k}"
        )

        # Reshape (M, K) -> (M//block_size_m, block_size_m, K//block_size_k, block_size_k)
        new_shape = (
            M // self.block_size_m,
            self.block_size_m,
            K // self.block_size_k,
            self.block_size_k,
        )
        tensor_blocked = tensor.reshape(new_shape)

        # Compute max over the block dimensions (dims 1 and 3)
        block_max = tensor_blocked.abs().amax(dim=3).amax(dim=1)

        # Compute inverse scales
        scales = block_max / self.E4M3_MAX
        return torch.clamp(scales, min=1e-12)

    def apply_scaling(
        self,
        tensor: torch.Tensor,
        scales: torch.Tensor,
        inverse: bool = False,
        clamp_to_fp8_range: bool = False,
    ) -> torch.Tensor:
        """
        Apply scaling to tensor based on the scaling type.

        This is a unified method that handles all scaling types:
        - TensorWise: Uses scalar scale
        - RowWise: Uses per-row scales (M,)
        - BlockWise*: Uses blockwise scales (M//block_size_m, K//block_size_k)

        Args:
            tensor: Input tensor (typically M, K for 2D)
            scales: Scale tensor with shape depending on scaling type
                   These are inverse scales (amax / dtype_max)
            inverse: If True, multiply by scales (dequantization)
                    If False, divide by scales (quantization)
            clamp_to_fp8_range: If True, clamp to FP8 range before returning

        Returns:
            Scaled tensor (same shape as input)
        """
        old_shape = tensor.shape
        if self.scaling_type == ScalingType.RowWise:
            # expand (M,) -> (M, 1)
            scales = scales.unsqueeze(1)
        elif self.scaling_type != ScalingType.TensorWise:
            # blockwise scaling
            M, K = tensor.shape
            new_shape = (
                M // self.block_size_m,
                self.block_size_m,
                K // self.block_size_k,
                self.block_size_k,
            )
            tensor = tensor.reshape(new_shape)
            scales = scales.unsqueeze(1).unsqueeze(3)

        if inverse:
            tensor_scaled = tensor * scales
        else:
            tensor_scaled = tensor / scales
            if clamp_to_fp8_range:
                tensor_scaled = torch.clamp(
                    tensor_scaled, min=-self.E4M3_MAX, max=self.E4M3_MAX
                )

        return tensor_scaled.reshape(*old_shape)


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "native_fp8"))
from fp8_groupwise import NativeGroupwiseGemm as CuBLASRefBlockwiseGemm, NATIVE_SOURCE_FILES
SM120_NATIVE_BACKEND = "CUTLASS87b_sm120a_E4M3_A1x128_B128x128_FP32_accumulate"

# --- end inlined fp8_reference ---



def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """
    Generate random inputs and compute scales for FP8 vision attention output projection.
    
    Returns:
        Dict of (attn_output, weight, scale_attn, scale_weight)
    """
    seq_len = axes_and_scalars["seq_len"]
    hidden_size = axes_and_scalars["hidden_size"]
    
    attn_output = torch.randn(seq_len, hidden_size, dtype=torch.bfloat16, device=device)
    weight = torch.randn(hidden_size, hidden_size, dtype=torch.bfloat16, device=device)
    
    # Convert to FP32 for scale computation
    attn_output_fp32 = attn_output.to(torch.float32)
    weight_fp32 = weight.to(torch.float32)
    
    # Transpose weight for blockwise scale computation
    weight_fp32_t = weight_fp32.T  # (K, N)
    
    # Compute scales using unified interface
    activation_scaler = BlockwiseScaler(ScalingType.BlockWise1x128)
    scale_attn = activation_scaler.compute_scales(attn_output_fp32)
    
    weight_scaler = BlockwiseScaler(ScalingType.BlockWise128x128)
    scale_weight = weight_scaler.compute_scales(weight_fp32_t)
    
    return {
        "attn_output": attn_output,
        "weight": weight,
        "scale_attn": scale_attn,
        "scale_weight": scale_weight,
    }


@torch.no_grad()
def run(
    attn_output: torch.Tensor,
    weight: torch.Tensor,
    scale_attn: torch.Tensor,
    scale_weight: torch.Tensor,
):
    """
    FP8 vision attention output projection with blockwise scaling.
    
    Scaling approach:
    - attn_output (input): BlockWise1x128 scaling
    - weight: BlockWise128x128 scaling
    
    Args:
        attn_output: Input tensor (seq_len, hidden_size) in BF16
        weight: Weight tensor (hidden_size, hidden_size) in BF16
        scale_attn: Blockwise scales for attn_output (seq_len, hidden_size//128)
        scale_weight: Blockwise scales for weight (hidden_size//128, hidden_size//128)
    
    Returns:
        Output tensor (seq_len, hidden_size)
    """
    M, K = attn_output.shape
    N, K_w = weight.shape
    assert K == K_w, f"K dimension mismatch: {K} vs {K_w}"
    
    # Initialize scalers and GEMM reference
    activation_scaler = BlockwiseScaler(ScalingType.BlockWise1x128)
    weight_scaler = BlockwiseScaler(ScalingType.BlockWise128x128)
    gemm_ref = CuBLASRefBlockwiseGemm()
    
    # Convert to FP32 for scaling operations
    x_fp32 = attn_output.to(torch.float32)
    w_fp32 = weight.to(torch.float32)
    
    # Step 1: Apply scaling before quantization
    x_scaled = activation_scaler.apply_scaling(
        x_fp32, scale_attn, inverse=False, clamp_to_fp8_range=True
    )
    
    # Transpose w to (K, N) for blockwise scaling
    w_fp32_t = w_fp32.T  # (K, N)
    w_scaled = weight_scaler.apply_scaling(
        w_fp32_t, scale_weight, inverse=False, clamp_to_fp8_range=True
    )
    
    # Step 2: Quantize to FP8
    qx = x_scaled.to(torch.float8_e4m3fn)  # (M, K)
    qw = w_scaled.T.to(torch.float8_e4m3fn)  # (N, K) - transpose back
    
    # Step 3: Transpose weight scales to match CuBLAS format
    # Scales from compute_scales are (K//128, N//128), need (N//128, K//128)
    scale_w_cublas = scale_weight.T.contiguous()  # (N//128, K//128)
    
    # Step 4: Call CuBLAS reference implementation
    output = gemm_ref.scaled_mm(
        mat_a=qx,
        mat_b=qw,
        scale_a=scale_attn,
        scale_recipe_a=ScalingType.BlockWise1x128,
        scale_b=scale_w_cublas,
        scale_recipe_b=ScalingType.BlockWise128x128,
        bias=None,
        output_dtype=attn_output.dtype,
        use_fast_accum=True,
    )
    
    return output
