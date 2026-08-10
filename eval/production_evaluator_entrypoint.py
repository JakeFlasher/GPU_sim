"""Source-pinned current-M1 production-evaluator preflight launcher.

This executable deliberately contains no scoring, report, receipt, label-opening,
or production-request implementation.  Current immutable production authority
registries are empty, so loading those later-stage capabilities would create a
monkeypatchable admission surface before M1 exists.  The launcher authenticates
its own source and reports the typed authority blocker only.
"""

from __future__ import annotations

if __name__ != "__main__":
    raise RuntimeError(
        "production evaluator entrypoint is executable only as a source-pinned CLI"
    )

import re
import sys
from pathlib import Path
from typing import Any, Mapping

from eval.contracts._common import EvaluatorAuthorityUnavailable
from tools.local_labels.immutable_artifact import (
    ImmutableArtifactError,
    canonical_json_bytes,
    open_retained_regular,
)

_SOURCE_LIMIT_BYTES = 2 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SUPPORTED_PROBES = frozenset({"evaluate", "evaluate_production"})


class EntrypointError(ImmutableArtifactError):
    """Typed failure in the minimal source-pinned launcher."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _write_cli_model(model: Mapping[str, Any]) -> None:
    sys.stdout.buffer.write(canonical_json_bytes(dict(model)) + b"\n")
    sys.stdout.buffer.flush()


def _require_entrypoint_source_pin(expected_sha256: str) -> None:
    if (
        type(expected_sha256) is not str
        or _SHA256_RE.fullmatch(expected_sha256) is None
    ):
        raise EntrypointError(
            "EVALUATOR_ENTRYPOINT_SOURCE_PIN_INVALID",
            "entrypoint source SHA-256 must be lowercase 64-hex",
        )
    with open_retained_regular(Path(__file__), max_bytes=_SOURCE_LIMIT_BYTES) as source:
        if source.sha256 != expected_sha256:
            raise EntrypointError(
                "EVALUATOR_ENTRYPOINT_SOURCE_PIN_MISMATCH",
                "standalone evaluator source differs from the external pin",
            )
        source.verify_unchanged()


def _require_current_m1_authorities() -> None:
    """Fail before any future production request or caller-controlled path exists."""

    from eval.contracts.launch_crosswalk import _require_production_v2_authorities

    _require_production_v2_authorities()
    raise EntrypointError(
        "PRODUCTION_AUTHORITY_PREFLIGHT_BYPASSED",
        "production authority preflight returned without its required blocker",
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[1] not in _SUPPORTED_PROBES:
        _write_cli_model(
            {
                "code": "CLI_USAGE_FAILED",
                "detail": (
                    "expected: EXPECTED_SOURCE_SHA256 "
                    "evaluate|evaluate_production"
                ),
            }
        )
        return 2
    try:
        _require_entrypoint_source_pin(args[0])
        _require_current_m1_authorities()
    except EvaluatorAuthorityUnavailable as exc:
        _write_cli_model(
            {
                "code": exc.code.value,
                "detail": str(exc).partition(": ")[2],
            }
        )
        return 0
    except (ImmutableArtifactError, ValueError) as exc:
        _write_cli_model(
            {
                "code": getattr(exc, "code", "EVALUATOR_ENTRYPOINT_FAILED"),
                "detail": getattr(exc, "detail", str(exc)),
            }
        )
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
