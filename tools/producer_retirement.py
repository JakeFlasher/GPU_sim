"""Shared typed guard for retired producers of immutable evidence.

Retired producer entry points use one machine-readable reason and refuse before
reading inputs, launching subprocesses, building artifacts, or writing outputs.
"""

from __future__ import annotations

import sys
from typing import NoReturn, TextIO

LEGACY_PRODUCER_RETIRED_IMMUTABLE_EVIDENCE = (
    "legacy_producer_retired_immutable_evidence"
)
RETIRED_PRODUCER_EXIT_CODE = 2


class LegacyProducerRetiredError(RuntimeError):
    """Typed refusal raised before a legacy producer can execute."""

    def __init__(self, *, surface: str, successor_requirement: str):
        self.reason = LEGACY_PRODUCER_RETIRED_IMMUTABLE_EVIDENCE
        self.surface = surface
        self.successor_requirement = successor_requirement
        super().__init__(
            f"{self.reason}: {surface}: {successor_requirement}"
        )


def refuse_legacy_producer(
    surface: str, *, successor_requirement: str
) -> NoReturn:
    """Raise the stable typed refusal without touching producer inputs."""

    raise LegacyProducerRetiredError(
        surface=surface,
        successor_requirement=successor_requirement,
    )


def report_legacy_producer_retirement(
    surface: str,
    *,
    successor_requirement: str,
    stream: TextIO | None = None,
) -> int:
    """Emit one typed CLI diagnostic and return the stable retired exit code."""

    try:
        refuse_legacy_producer(
            surface,
            successor_requirement=successor_requirement,
        )
    except LegacyProducerRetiredError as exc:
        (stream if stream is not None else sys.stderr).write(f"{exc}\n")
    return RETIRED_PRODUCER_EXIT_CODE
