# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Correctness computation utilities."""

from __future__ import annotations

import math
import random
from typing import Optional, Tuple

import torch

from sol_execbench.core.data import Correctness
from sol_execbench.core.data.workload import ToleranceSpec


# Default chunk size for the bounded correctness path. 64 Mi fp32 elements
# = 256 MiB working set per chunk before fanout (abs_error, tol_bound,
# exceeds_mask, rel_error materialise as same-shape tensors so the peak
# is roughly 1 GiB per chunk regardless of the input tensor size). On a
# 24 GiB host this leaves substantial headroom for the rest of the
# evaluation pipeline. Operators can override via the
# `chunk_elements` keyword on `compute_error_stats`.
_DEFAULT_CHUNK_ELEMENTS: int = 64 * 1024 * 1024


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, PyTorch CPU and CUDA."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_tensor_sanity(
    sol_tensor: torch.Tensor,
    ref_tensor: torch.Tensor,
    allow_negative_inf: bool = False,
) -> Optional[Correctness]:
    """Check for non-finite values and all-zeros output.

    Returns a ``Correctness`` describing the failure when either tensor
    contains inf/nan values, or the solution is all zeros while the
    reference is not.  Returns ``None`` when both tensors look sane.

    Inf and NaN are always treated as incorrect, even when both tensors
    have the same non-finite value at a given position — unless
    *allow_negative_inf* is True, in which case positions where **both**
    tensors are -inf are tolerated.
    """
    # Non-finite values in either tensor are always wrong.
    # Even matching infinities (e.g. both -inf) are rejected because
    # non-finite values indicate degenerate numerics, not correct output.
    ref_nonfinite = ~torch.isfinite(ref_tensor)
    sol_nonfinite = ~torch.isfinite(sol_tensor)

    if allow_negative_inf:
        # Exclude positions where both tensors are -inf.
        both_neg_inf = (ref_tensor == float("-inf")) & (sol_tensor == float("-inf"))
        ref_nonfinite = ref_nonfinite & ~both_neg_inf
        sol_nonfinite = sol_nonfinite & ~both_neg_inf

    has_nonfinite = ref_nonfinite.any().item() or sol_nonfinite.any().item()
    if has_nonfinite:
        has_nan = (
            torch.isnan(sol_tensor).any().item() or torch.isnan(ref_tensor).any().item()
        )
        # has_inf is True only when there are Inf values but NO NaN values.
        # NaN takes priority because it's a stricter failure mode (Inf can
        # result from overflow, but NaN indicates undefined computation).
        return Correctness(has_nan=has_nan, has_inf=not has_nan)

    # Non-zero output check: if reference has non-trivial values
    # but solution is all zeros, fail immediately.
    ref_norm = torch.linalg.vector_norm(ref_tensor.to(torch.float32))
    if (
        ref_norm.item() > 0
        and torch.linalg.vector_norm(sol_tensor.to(torch.float32)).item() == 0
    ):
        abs_err = float(ref_norm.item())
        return Correctness(
            max_absolute_error=abs_err,
            max_relative_error=abs_err,
        )

    return None


def _check_sanity_chunked(
    sol_tensor: torch.Tensor,
    ref_tensor: torch.Tensor,
    allow_negative_inf: bool,
    chunk_elements: int,
) -> Optional[Correctness]:
    """Bounded variant of :func:`check_tensor_sanity`.

    Streams the input tensors in flat-index chunks of at most
    ``chunk_elements`` so the peak working set is bounded by chunk size
    rather than the full tensor footprint. Semantically identical to
    the whole-tensor implementation: every non-finite element (other
    than the optional matching -inf carve-out) flags the failure;
    all-zero solution against a non-zero reference flags the failure;
    otherwise returns ``None``.
    """
    sol_flat = sol_tensor.reshape(-1)
    ref_flat = ref_tensor.reshape(-1)
    total = sol_flat.numel()
    if total == 0:
        return None

    has_nan = False
    has_inf = False
    sol_sq_sum = 0.0
    ref_sq_sum = 0.0

    for start in range(0, total, chunk_elements):
        end = min(start + chunk_elements, total)
        sol_chunk = sol_flat[start:end].to(torch.float32)
        ref_chunk = ref_flat[start:end].to(torch.float32)

        sol_nf = ~torch.isfinite(sol_chunk)
        ref_nf = ~torch.isfinite(ref_chunk)

        if allow_negative_inf:
            both_neg_inf = (sol_chunk == float("-inf")) & (
                ref_chunk == float("-inf")
            )
            sol_nf = sol_nf & ~both_neg_inf
            ref_nf = ref_nf & ~both_neg_inf

        if sol_nf.any().item() or ref_nf.any().item():
            chunk_has_nan = (
                torch.isnan(sol_chunk).any().item()
                or torch.isnan(ref_chunk).any().item()
            )
            has_nan = has_nan or chunk_has_nan
            has_inf = has_inf or not chunk_has_nan

        sol_sq_sum += float((sol_chunk * sol_chunk).sum().item())
        ref_sq_sum += float((ref_chunk * ref_chunk).sum().item())

        del sol_chunk, ref_chunk, sol_nf, ref_nf

    if has_nan or has_inf:
        # NaN takes priority over Inf when both are present in the
        # streamed view.
        return Correctness(has_nan=has_nan, has_inf=(not has_nan) and has_inf)

    if ref_sq_sum > 0.0 and sol_sq_sum == 0.0:
        ref_norm = math.sqrt(ref_sq_sum)
        return Correctness(
            max_absolute_error=ref_norm, max_relative_error=ref_norm
        )
    return None


def compute_error_stats(
    output: torch.Tensor,
    reference: torch.Tensor,
    tolerance: ToleranceSpec,
    *,
    chunk_elements: int = _DEFAULT_CHUNK_ELEMENTS,
) -> Tuple[Correctness, bool]:
    """Compute numerical error between *output* and *reference*.

    Returns ``(correctness, exceeds)`` where *correctness* is a
    :class:`Correctness` carrying error metrics (and ``has_nan`` /
    ``has_inf`` flags when non-finite values are detected), and *exceeds*
    is ``True`` when the tolerance is violated.

    Memory profile: processes the input tensors in flat-index chunks of
    at most ``chunk_elements`` so the peak intermediate set is bounded
    by chunk size rather than the full tensor footprint. The aggregated
    result (max_abs / max_rel / exceeds_count / matched_ratio) matches
    the whole-tensor implementation bit-for-bit on small inputs (≤
    chunk_elements) and within FP32 round-off elsewhere.
    """
    if chunk_elements <= 0:
        raise ValueError(
            f"chunk_elements must be positive, got {chunk_elements!r}"
        )

    allow_neg_inf = tolerance.allow_negative_inf

    sanity = _check_sanity_chunked(
        output, reference, allow_negative_inf=allow_neg_inf,
        chunk_elements=chunk_elements,
    )
    if sanity is not None:
        return sanity, True

    sol_flat = output.reshape(-1)
    ref_flat = reference.reshape(-1)
    total_input = sol_flat.numel()
    if total_input == 0:
        return Correctness(), False

    max_abs = 0.0
    max_rel = 0.0
    exceeds_count = 0
    matched_count = 0
    counted_elements = 0

    atol = float(tolerance.max_atol)
    rtol = float(tolerance.max_rtol)

    for start in range(0, total_input, chunk_elements):
        end = min(start + chunk_elements, total_input)
        x_chunk = sol_flat[start:end].to(torch.float32)
        y_chunk = ref_flat[start:end].to(torch.float32)

        if allow_neg_inf:
            both_neg_inf = (x_chunk == float("-inf")) & (
                y_chunk == float("-inf")
            )
            finite_mask = ~both_neg_inf
            if not bool(finite_mask.all().item()):
                x_chunk = x_chunk[finite_mask]
                y_chunk = y_chunk[finite_mask]
            del both_neg_inf, finite_mask

        if x_chunk.numel() == 0:
            del x_chunk, y_chunk
            continue

        abs_chunk = torch.abs(x_chunk - y_chunk)
        counted_elements += int(abs_chunk.numel())

        chunk_max_abs = float(abs_chunk.max().item())
        if chunk_max_abs > max_abs:
            max_abs = chunk_max_abs

        abs_y_chunk = torch.abs(y_chunk)
        tol_bound = atol + rtol * abs_y_chunk
        exceeds_mask = (abs_chunk > tol_bound) | ~torch.isfinite(abs_chunk)
        chunk_exceeds = int(exceeds_mask.sum().item())
        exceeds_count += chunk_exceeds
        matched_count += int(abs_chunk.numel() - chunk_exceeds)
        del tol_bound, exceeds_mask

        denom = torch.clamp(abs_y_chunk, min=atol)
        rel_chunk = abs_chunk / denom
        chunk_max_rel = float(rel_chunk.max().item())
        if chunk_max_rel > max_rel:
            max_rel = chunk_max_rel
        del abs_chunk, abs_y_chunk, denom, rel_chunk, x_chunk, y_chunk

    if counted_elements == 0:
        return Correctness(), False

    matched_ratio = matched_count / float(counted_elements)
    matched_ratio = max(0.0, min(1.0, matched_ratio))

    exceeds_tol = matched_ratio < tolerance.required_matched_ratio
    if tolerance.max_error_cap is not None and max_abs > tolerance.max_error_cap:
        exceeds_tol = True

    return Correctness(
        max_absolute_error=max_abs, max_relative_error=max_rel
    ), exceeds_tol
