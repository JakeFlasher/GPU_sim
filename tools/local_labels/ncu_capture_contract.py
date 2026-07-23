"""Shared semantic checks for explicit Nsight Compute launch controls."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

MAX_NCU_SELECTOR = 2_147_483_647

_SELECTOR_TRIGGER_FIELDS = frozenset({
    "ncu_target_processes",
    "ncu_filter_mode",
    "ncu_profile_from_start",
    "ncu_launch_skip",
    "ncu_launch_count",
})
_EXPLICIT_NCU_FIELDS = frozenset({
    "ncu_clock_control",
    "ncu_cache_control",
    "ncu_replay_mode",
    *_SELECTOR_TRIGGER_FIELDS,
})


def validate_measurement_protocol(protocol: Any) -> None:
    """Reject partial or overflowing NCU launch-selector provenance.

    Historical protocol blocks that contain no selector-trigger field remain
    representable. Once any selector field is present, the complete explicit
    NCU control tuple is mandatory.
    """
    if protocol is None:
        return
    if not isinstance(protocol, Mapping):
        raise ValueError("measurement_protocol must be an object")

    present_triggers = _SELECTOR_TRIGGER_FIELDS.intersection(protocol)
    if not present_triggers:
        return
    missing = sorted(_EXPLICIT_NCU_FIELDS.difference(protocol))
    if missing:
        raise ValueError(
            "measurement_protocol has partial NCU launch controls; missing "
            f"{missing!r}"
        )

    launch_skip = protocol["ncu_launch_skip"]
    launch_count = protocol["ncu_launch_count"]
    if (
        type(launch_skip) is not int
        or type(launch_count) is not int
        or not 0 <= launch_skip <= MAX_NCU_SELECTOR
        or not 1 <= launch_count <= MAX_NCU_SELECTOR
    ):
        raise ValueError(
            "measurement_protocol NCU launch selector is outside the signed "
            "32-bit range"
        )
    if launch_skip > MAX_NCU_SELECTOR - launch_count:
        raise ValueError(
            "measurement_protocol NCU launch skip plus count exceeds the "
            "signed 32-bit selector range"
        )
    if type(protocol["ncu_profile_from_start"]) is not bool:
        raise ValueError("measurement_protocol ncu_profile_from_start must be boolean")
