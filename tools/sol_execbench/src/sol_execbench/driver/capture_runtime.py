# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, self-contained CUDA profiler ROI used by SOL capture mode.

The module deliberately does not import torch.  The evaluation driver supplies
``torch.cuda`` plus pre-created CUDA events, which makes the ordering contract
unit-testable without loading CUDA or running a kernel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

if __package__:
    # Normal package import.  The standalone staged copy intentionally lacks
    # this sibling and only uses the ROI symbols below.
    from .certified_env import (
        CertifiedEnvironmentError,
        PINNED_NVCC_CCBIN,
        build_certified_env,
        environment_contract_sha256,
    )


def _status_code(value: Any) -> int:
    """Normalize CUDA runtime status values exposed by different torch builds."""
    if isinstance(value, tuple):
        if not value:
            raise TypeError("CUDA runtime returned an empty status tuple")
        value = value[0]
    if hasattr(value, "value"):
        value = value.value
    return int(value)


@dataclass(frozen=True)
class RoiOutcome:
    """Auditable call counts and CUDA-event timing for one candidate call."""

    profiler_start_calls: int
    candidate_user_fn_calls: int
    profiler_stop_calls: int
    profiler_start_status: int | None
    profiler_stop_status: int | None
    latency_ms: float | None
    timing_methodology: str = "cuda_events"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProfiledCandidateError(RuntimeError):
    """A profiler API or candidate failure carrying the exact partial outcome."""

    def __init__(self, message: str, outcome: RoiOutcome):
        super().__init__(message)
        self.outcome = outcome


def profile_candidate_once(
    *,
    cuda: Any,
    candidate: Callable[[], Any],
    start_event: Any,
    end_event: Any,
) -> RoiOutcome:
    """Run exactly one candidate call inside one profiler start/stop pair.

    The start event is recorded and completed before the profiler range opens.
    The end event is recorded after the candidate returns.  The device is then
    synchronized before ``cudaProfilerStop`` is called from ``finally`` exactly
    once.  No CUPTI timing path is reachable from this function.
    """

    start_calls = 0
    candidate_calls = 0
    stop_calls = 0
    start_status: int | None = None
    stop_status: int | None = None
    latency_ms: float | None = None
    failure: BaseException | None = None

    # Establish a completed event immediately before the active profiler range.
    cuda.synchronize()
    start_event.record()
    cuda.synchronize()
    cudart = cuda.cudart()

    try:
        start_calls += 1
        start_status = _status_code(cudart.cudaProfilerStart())
        if start_status != 0:
            raise RuntimeError(f"cudaProfilerStart failed with status {start_status}")

        candidate_calls += 1
        candidate()
        end_event.record()
    except BaseException as exc:  # preserve the candidate exception after cleanup
        failure = exc
    finally:
        # Complete every launch issued by the candidate before closing the ROI.
        try:
            cuda.synchronize()
        except BaseException as exc:
            if failure is None:
                failure = exc
        stop_calls += 1
        try:
            stop_status = _status_code(cudart.cudaProfilerStop())
            if stop_status != 0 and failure is None:
                failure = RuntimeError(
                    f"cudaProfilerStop failed with status {stop_status}"
                )
        except BaseException as exc:
            if failure is None:
                failure = exc

    if failure is None:
        try:
            latency_ms = float(start_event.elapsed_time(end_event))
        except BaseException as exc:
            failure = exc

    outcome = RoiOutcome(
        profiler_start_calls=start_calls,
        candidate_user_fn_calls=candidate_calls,
        profiler_stop_calls=stop_calls,
        profiler_start_status=start_status,
        profiler_stop_status=stop_status,
        latency_ms=latency_ms,
    )
    if failure is not None:
        raise ProfiledCandidateError(str(failure), outcome) from failure
    return outcome


__all__ = [
    "ProfiledCandidateError",
    "RoiOutcome",
    "profile_candidate_once",
]
if __package__:
    __all__ += [
        "CertifiedEnvironmentError",
        "PINNED_NVCC_CCBIN",
        "build_certified_env",
        "environment_contract_sha256",
    ]
