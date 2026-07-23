"""Fail-closed retirement contract for label-trained legacy workflows."""

from __future__ import annotations

from inspect import signature
from typing import Callable, NoReturn, ParamSpec, TypeVar


LEGACY_NCU_CALIBRATION_RETIRED = (
    "legacy_ncu_calibration_retired_targets_not_features"
)


class LegacyCalibrationRetiredError(RuntimeError):
    """Raised before a retired label-trained workflow can execute."""

    def __init__(self, surface: str):
        self.reason = LEGACY_NCU_CALIBRATION_RETIRED
        self.surface = surface
        super().__init__(f"{self.reason}: {surface}")


def refuse_legacy_ncu_calibration(surface: str) -> NoReturn:
    """Refuse a retired workflow with one stable machine-readable reason."""

    raise LegacyCalibrationRetiredError(surface)


_P = ParamSpec("_P")
_R = TypeVar("_R")


def retired_legacy_ncu_calibration(
    surface: str,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Guard an action while preserving its import and help-time metadata."""

    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            refuse_legacy_ncu_calibration(surface)

        # Preserve the metadata needed by help and signature inspection without
        # publishing functools.wraps' ``__wrapped__`` escape hatch. Decorated
        # originals carry the same refusal as their first executable statement,
        # but the retired body does not need to remain discoverable at all.
        guarded.__module__ = function.__module__
        guarded.__name__ = function.__name__
        guarded.__qualname__ = function.__qualname__
        guarded.__doc__ = function.__doc__
        guarded.__annotations__ = dict(function.__annotations__)
        setattr(guarded, "__signature__", signature(function))
        return guarded

    return decorate
