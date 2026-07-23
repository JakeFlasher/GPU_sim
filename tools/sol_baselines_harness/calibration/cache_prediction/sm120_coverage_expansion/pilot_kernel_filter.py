"""Per-cell pilot kernel-name-filter source of truth.

NCU's ``--kernel-name <name>`` filter selects launches
whose kernel name matches the given value (with the
basis controlled by NCU's ``--kernel-name-base`` flag,
which defaults to ``function``). For sol-execbench-
wrapped pilots, the workload runs many non-pilot
launches first (torch CUDA init, curand RNG input
randomization, cuBLAS / cuDNN library kernels), so
without an explicit filter NCU captures the wrong
kernel.

This module pins the per-cell kernel function name for
every pilot cell whose solution carries a custom
``__global__`` kernel. ATen-only pilots (whose actual
GPU work runs through cuBLAS / cuDNN library kernels
with shape- and dtype-dependent kernel selection) are
NOT in this map — callers must demote those cells to
the typed ``ncu_kernel_name_not_pinned`` failure rather
than attempt to capture an arbitrary library kernel.

The mapping is anchored to the kernel function name as
it appears in each pilot's ``kernel.cu`` source; the
function-basis match is sufficient because the C++
name-mangling preserves the function basename in NCU's
demangled output.
"""

from __future__ import annotations

from typing import Final


# Per-cell kernel function name used as NCU's
# ``--kernel-name`` filter (with the default
# ``--kernel-name-base function`` basis). Keys are
# (slug, language) tuples. Cells NOT in this map are
# ATen-dispatched and do not have a single stable
# kernel name to pin — callers must route them to the
# ``ncu_kernel_name_not_pinned`` typed failure.
PILOT_KERNEL_NAME_BASE: Final[dict[tuple[str, str], str]] = {
    ("problem_002_fused_add_rmsnorm_h4096", "cuda_cpp"): (
        "problem_002_fused_add_rmsnorm_h4096_kernel"
    ),
    ("problem_024_rmsnorm_h2048", "cuda_cpp"): (
        "problem_024_rmsnorm_h2048_kernel"
    ),
    ("problem_025_video_latent_gelu_activation", "cuda_cpp"): (
        # Two kernels in this cell: gelu_vec4 (main, hot
        # vectorized path) + gelu_tail (small remainder).
        # Pin the main vec4 kernel; the tail kernel runs
        # at most once per workload on the remainder
        # elements and would skew the 3-run repeatability
        # if captured alongside.
        "problem_025_gelu_vec4_kernel"
    ),
    ("problem_069_rms_norm", "cuda_cpp"): (
        "problem_069_rms_norm_kernel"
    ),
    # ---- Cross-arch plan the calibration workstream (sm_86 5-cell pilot) ----
    # Each pilot's primary computational kernel; the secondary
    # kernels (cast/initialisation/tail helpers) are intentionally
    # NOT pinned so NCU's --kernel-name filter captures only the
    # main read-heavy kernel for 3-run repeatability.
    ("problem_006_hyena_depthwise_conv1d_split_gate", "cuda_cpp"): (
        # Primary depthwise conv1d kernel; the secondary
        # `_v_gate_kernel` runs once per workload on the tail.
        "problem_006_hyena_depthwise_conv1d_split_gate_kernel"
    ),
    ("problem_040_conv2d_residual_block", "cuda_cpp"): (
        # Primary conv-in kernel; the secondary
        # `_conv_out_plus_residual_kernel` runs after on the same
        # cell but pinning the first kernel keeps repeatability
        # tight.
        "problem_040_conv2d_residual_block_conv_in_kernel"
    ),
    ("problem_066_resnet_block_with_time_embedding", "cuda_cpp"): (
        # Primary groupnorm + SiLU kernel.
        "problem_066_resnet_block_with_time_embedding_groupnorm_silu_kernel"
    ),
    (
        "problem_073_fused_encoder_final_norm_to_decoder_cross_attention_kv_projection",
        "cuda_cpp",
    ): (
        # R11 refactor: only the RMSNorm kernel remains (matmul
        # delegated to at::matmul); pin the RMSNorm kernel.
        "problem_073_rmsnorm_kernel"
    ),
    ("problem_087_embedding_with_initial_layernorm_backward", "cuda_cpp"): (
        # Primary LayerNorm-backward kernel; the secondary
        # `cast_fp32_to_bf16_kernel` is a tail conversion.
        "problem_087_embedding_layernorm_backward_kernel"
    ),
}


def lookup_pilot_kernel_name_base(
    slug: str, language: str,
) -> str | None:
    """Return the NCU ``--kernel-name`` function-basis value
    for the given pilot cell, or ``None`` when the cell is
    ATen-dispatched (no custom kernel) and cannot have a
    stable per-kernel filter.
    """
    return PILOT_KERNEL_NAME_BASE.get((slug, language))
