#!/usr/bin/env python3
"""Closed CUDA/NCU contracts for live local-label capture.

The historical campaign pins encode ``nvcc_release`` as a short string.  That
representation is retained as an explicit compatibility input, but live
capture code consumes and records the structured ``{major, minor}`` form
defined here.  Keeping the conversion in one place prevents substring checks
such as ``"13.2" in banner`` from admitting a different compiler release.

The publication scope is deliberately separate from the NCU collection
surface.  ADR 0002 binds publication to elapsed GPC cycles while retaining the
registry superset as diagnostic evidence.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


SOL_CUDA_RELEASE = {"major": 13, "minor": 2}
"""Exact nvcc release selected by the live-v2 SOL CUDA environment."""

CYCLES_ONLY_PUBLICATION_SCOPE = {
    "decision": "CYCLES_ONLY_BINDING",
    "binding_metrics": ["gpc__cycles_elapsed.max"],
    "collection_scope": "registry_superset",
}
"""Closed publication claim boundary; collection remains the superset."""

_RELEASE_TEXT = re.compile(r"^(0|[1-9][0-9]*)[.](0|[1-9][0-9]*)$")
_NVCC_BANNER_RELEASE = re.compile(
    r"(?<![A-Za-z0-9_])release[ \t]+"
    r"(0|[1-9][0-9]*)[.](0|[1-9][0-9]*)(?=[,\r\n \t]|$)",
    re.IGNORECASE,
)
_NCU_VERSION_PREFIX = re.compile(
    r"^(0|[1-9][0-9]*)(?:[.](0|[1-9][0-9]*)){1,3}$"
)


class CaptureToolchainContractError(ValueError):
    """A malformed or mismatched live capture toolchain contract."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def structured_nvcc_release(value: Any) -> dict[str, int]:
    """Return the exact two-integer release form.

    A canonical ``X.Y`` string is accepted solely for byte-compatible loading
    of the existing pins.  Every returned/recorded value is structured.  No
    patch component, suffix, leading zero, float, boolean, or extra object key
    is accepted.
    """

    if isinstance(value, str):
        match = _RELEASE_TEXT.fullmatch(value)
        if match is None:
            raise CaptureToolchainContractError(
                "NVCC_RELEASE_INVALID", repr(value)
            )
        return {"major": int(match.group(1)), "minor": int(match.group(2))}
    if not isinstance(value, Mapping) or set(value) != {"major", "minor"}:
        raise CaptureToolchainContractError(
            "NVCC_RELEASE_INVALID", repr(value)
        )
    major = value.get("major")
    minor = value.get("minor")
    if (
        type(major) is not int
        or type(minor) is not int
        or major < 0
        or minor < 0
    ):
        raise CaptureToolchainContractError(
            "NVCC_RELEASE_INVALID", repr(value)
        )
    return {"major": major, "minor": minor}


def nvcc_release_text(value: Any) -> str:
    """Render a validated structured/legacy release as canonical ``X.Y``."""

    release = structured_nvcc_release(value)
    return f"{release['major']}.{release['minor']}"


def parse_nvcc_release_banner(banner: str) -> dict[str, int]:
    """Parse exactly one unambiguous nvcc ``release X.Y`` banner token."""

    if not isinstance(banner, str) or not banner:
        raise CaptureToolchainContractError(
            "NVCC_RELEASE_UNREADABLE", repr(banner)
        )
    matches = {
        (int(match.group(1)), int(match.group(2)))
        for match in _NVCC_BANNER_RELEASE.finditer(banner)
    }
    if len(matches) != 1:
        raise CaptureToolchainContractError(
            "NVCC_RELEASE_UNREADABLE",
            f"expected one distinct release token, observed={sorted(matches)!r}",
        )
    major, minor = next(iter(matches))
    return {"major": major, "minor": minor}


def require_nvcc_release(
    observed: Any, expected: Any, *, code: str = "NVCC_RELEASE_MISMATCH",
) -> dict[str, int]:
    """Require exact structured equality and return the normalized release."""

    observed_release = structured_nvcc_release(observed)
    expected_release = structured_nvcc_release(expected)
    if observed_release != expected_release:
        raise CaptureToolchainContractError(
            code,
            f"observed={observed_release!r}; expected={expected_release!r}",
        )
    return observed_release


def require_sol_cuda_release(value: Any) -> dict[str, int]:
    """Require the live-v2 SOL compiler to be the certified CUDA 13.2 nvcc."""

    return require_nvcc_release(
        value, SOL_CUDA_RELEASE, code="SOL_ENV_NVCC_RELEASE_MISMATCH"
    )


def normalized_ncu_version_prefix(value: Any) -> str:
    """Validate an exact dotted numeric NCU release prefix."""

    if not isinstance(value, str) or _NCU_VERSION_PREFIX.fullmatch(value) is None:
        raise CaptureToolchainContractError(
            "NCU_VERSION_PREFIX_INVALID", repr(value)
        )
    return value


def ncu_version_matches(version_line: Any, expected_prefix: Any) -> bool:
    """Match ``Version X...`` on component boundaries, never by substring."""

    expected = normalized_ncu_version_prefix(expected_prefix)
    if not isinstance(version_line, str):
        return False
    match = re.fullmatch(
        r"Version[ \t]+([0-9]+(?:[.][0-9]+){1,3})(?:[ \t].*)?",
        version_line.strip(),
    )
    if match is None:
        return False
    observed = match.group(1)
    return observed == expected or observed.startswith(expected + ".")


def cycles_only_publication_scope(
    value: Any = CYCLES_ONLY_PUBLICATION_SCOPE,
) -> dict[str, Any]:
    """Validate and copy the exact cycles-only publication contract."""

    if value != CYCLES_ONLY_PUBLICATION_SCOPE:
        raise CaptureToolchainContractError(
            "NCU_PUBLICATION_SCOPE_MISMATCH", repr(value)
        )
    return {
        "decision": CYCLES_ONLY_PUBLICATION_SCOPE["decision"],
        "binding_metrics": list(
            CYCLES_ONLY_PUBLICATION_SCOPE["binding_metrics"]
        ),
        "collection_scope": CYCLES_ONLY_PUBLICATION_SCOPE[
            "collection_scope"
        ],
    }


def require_cycles_metric_in_collection(metrics: Sequence[Any]) -> None:
    """Require the binding cycles metric without narrowing the collection."""

    if (
        isinstance(metrics, (str, bytes))
        or not isinstance(metrics, Sequence)
        or any(not isinstance(metric, str) for metric in metrics)
        or list(metrics).count("gpc__cycles_elapsed.max") != 1
    ):
        raise CaptureToolchainContractError(
            "NCU_PUBLICATION_METRIC_INVALID", repr(metrics)
        )


__all__ = [
    "CYCLES_ONLY_PUBLICATION_SCOPE",
    "CaptureToolchainContractError",
    "SOL_CUDA_RELEASE",
    "cycles_only_publication_scope",
    "ncu_version_matches",
    "normalized_ncu_version_prefix",
    "nvcc_release_text",
    "parse_nvcc_release_banner",
    "require_cycles_metric_in_collection",
    "require_nvcc_release",
    "require_sol_cuda_release",
    "structured_nvcc_release",
]
