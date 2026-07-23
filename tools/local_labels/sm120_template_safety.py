"""Fail-closed boundary for the unvalidated sm_120 template lane.

The repository retains sm_120 labels, traces, bundles, and calibration files as
historical evidence. Those artifacts are not a production predictor contract:
the local Blackwell card records research gaps and every sm_120 predictor claim
is refuted. Production local-label paths therefore use this small, stdlib-only
boundary to make the distinction explicit without deleting or rewriting the
evidence.

``sm_86`` is deliberately *not* rejected here. Callers should invoke
:func:`template_unavailable` before reading inputs that could lead to a
prediction, training a head, or writing a derived artifact.
"""

from __future__ import annotations

from typing import Any

SM86_ARCH = "sm_86"
SM120_ARCH = "sm_120"

# This token is intentionally local-label/predictor scoped. It is not added to
# the closed cache/comparison wire schemas because the safety gate does not
# emit a predictor or comparison bundle for sm_120.
SM120_TEMPLATE_UNAVAILABLE_REASON = "sm120_template_unavailable"
SM120_TEMPLATE_SAFETY_DIAGNOSTIC = "sm120_template_safety_gate"


class Sm120TemplateUnavailable(RuntimeError):
    """Typed failure raised when a production sm_120 path is requested."""

    typed_reason = SM120_TEMPLATE_UNAVAILABLE_REASON
    diagnostic = SM120_TEMPLATE_SAFETY_DIAGNOSTIC
    typed_diagnostic = SM120_TEMPLATE_SAFETY_DIAGNOSTIC

    def __init__(self, *, operation: str, detail: str | None = None) -> None:
        self.operation = operation
        self.detail = detail or (
            "sm_120 is retained as a historical/template lane; live predictor "
            "training, bundle emission, and comparison scoring are disabled"
        )
        super().__init__(
            f"{self.diagnostic}: {self.typed_reason}: "
            f"operation={operation}: {self.detail}"
        )


def canonical_arch(arch: Any) -> str:
    """Normalize common deploy/build spellings to a label arch token.

    The label contract uses ``sm_86`` and ``sm_120``. The aliases are accepted
    at this boundary so a device/path token cannot accidentally bypass the
    safety gate by spelling Blackwell as ``sm120-laptop`` or ``sm_120a``.
    Unknown values are returned in normalized form and remain unsupported by
    the caller's own architecture checks.
    """
    token = str(arch).strip().lower().replace("-", "_")
    compact = token.replace("_", "")
    if token in {"sm_86", "sm86"} or compact == "sm86":
        return SM86_ARCH
    if token in {
        "sm_120",
        "sm120",
        "sm_120_laptop",
        "sm120_laptop",
        "sm_120a",
        "sm120a",
        "sm_120f",
        "sm120f",
    }:
        return SM120_ARCH
    return token


def is_sm120_template_arch(arch: Any) -> bool:
    """Return whether ``arch`` denotes the blocked sm_120 template lane."""
    return canonical_arch(arch) == SM120_ARCH


def raise_if_sm120_template(*, arch: Any, operation: str) -> None:
    """Raise the typed safety failure for sm_120, otherwise return normally."""
    if is_sm120_template_arch(arch):
        raise Sm120TemplateUnavailable(operation=operation)


def unavailable_result(
    *,
    operation: str,
    arch: Any = SM120_ARCH,
    cell_key: str | None = None,
    detail: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a stable, JSON-serializable typed-unavailable result.

    This shape is deliberately a *report*, not a predictor/comparison bundle;
    callers must not feed it to a scorer or write it over historical evidence.
    ``extra`` is additive metadata for the caller (for example stage reports).
    """
    result: dict[str, Any] = {
        "status": "typed_unavailable",
        "result_kind": "typed_unavailable",
        "typed_reason": SM120_TEMPLATE_UNAVAILABLE_REASON,
        "typed_diagnostic": SM120_TEMPLATE_SAFETY_DIAGNOSTIC,
        "operation": operation,
        "arch": canonical_arch(arch),
        "detail": detail
        or (
            "sm_120 template lane is disabled until the per-architecture "
            "research gaps are closed with live evidence"
        ),
    }
    if cell_key is not None:
        result["cell_key"] = cell_key
    result.update(extra)
    return result


__all__ = [
    "SM86_ARCH",
    "SM120_ARCH",
    "SM120_TEMPLATE_UNAVAILABLE_REASON",
    "SM120_TEMPLATE_SAFETY_DIAGNOSTIC",
    "Sm120TemplateUnavailable",
    "canonical_arch",
    "is_sm120_template_arch",
    "raise_if_sm120_template",
    "unavailable_result",
]
