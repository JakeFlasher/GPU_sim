"""Shared semantic checks for explicit Nsight Compute launch controls."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import re
import sys
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
_ENUM_FIELDS = {
    "ncu_target_processes": frozenset({"all"}),
    "ncu_filter_mode": frozenset({"global"}),
    "ncu_replay_mode": frozenset({"kernel"}),
    "ncu_cache_control": frozenset({"all", "none"}),
    "ncu_clock_control": frozenset({"base", "boost", "none"}),
}
_CANONICAL_UINT = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_CLI_OPTION_NAMES = frozenset({
    "--target-processes",
    "--filter-mode",
    "--replay-mode",
    "--cache-control",
    "--clock-control",
    "--profile-from-start",
    "--launch-skip",
    "--launch-count",
})


def _require_complete_explicit_protocol(protocol: Mapping[str, Any]) -> None:
    missing = sorted(_EXPLICIT_NCU_FIELDS.difference(protocol))
    if missing:
        raise ValueError(
            "measurement_protocol has partial NCU launch controls; missing "
            f"{missing!r}"
        )


def validate_measurement_protocol(protocol: Any) -> None:
    """Reject partial, incompatible, or overflowing NCU launch controls.

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
    _require_complete_explicit_protocol(protocol)

    for field, allowed in _ENUM_FIELDS.items():
        value = protocol[field]
        if not isinstance(value, str) or value not in allowed:
            raise ValueError(
                f"measurement_protocol {field} must be one of "
                f"{sorted(allowed)!r}"
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
        raise ValueError(
            "measurement_protocol ncu_profile_from_start must be boolean"
        )


def _canonical_selector(value: str, *, option: str) -> int:
    if (
        not _CANONICAL_UINT.fullmatch(value)
        or len(value) > len(str(MAX_NCU_SELECTOR))
    ):
        raise ValueError(f"{option} must be a canonical unsigned decimal")
    return int(value)


def _require_unique_cli_options(argv: Sequence[str]) -> None:
    seen: set[str] = set()
    for token in argv:
        option = token.partition("=")[0]
        if option not in _CLI_OPTION_NAMES:
            continue
        if option in seen:
            raise ValueError(f"duplicate explicit launch control: {option}")
        seen.add(option)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate explicit Nsight Compute launch controls.",
    )
    parser.add_argument("--target-processes")
    parser.add_argument("--filter-mode")
    parser.add_argument("--replay-mode")
    parser.add_argument("--cache-control")
    parser.add_argument("--clock-control")
    parser.add_argument("--profile-from-start")
    parser.add_argument("--launch-skip")
    parser.add_argument("--launch-count")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate only explicitly supplied shell controls for label_driver."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        _require_unique_cli_options(arguments)
    except ValueError as exc:
        print(f"NCU_LAUNCH_CONTROL_INVALID: {exc}", file=sys.stderr)
        return 2
    args = _argument_parser().parse_args(arguments)
    protocol: dict[str, Any] = {}
    option_fields = {
        "target_processes": "ncu_target_processes",
        "filter_mode": "ncu_filter_mode",
        "replay_mode": "ncu_replay_mode",
        "cache_control": "ncu_cache_control",
        "clock_control": "ncu_clock_control",
    }
    for option, field in option_fields.items():
        value = getattr(args, option)
        if value is not None:
            protocol[field] = value

    try:
        if args.profile_from_start is not None:
            if args.profile_from_start not in {"on", "off"}:
                raise ValueError(
                    "--profile-from-start must be either 'on' or 'off'"
                )
            protocol["ncu_profile_from_start"] = (
                args.profile_from_start == "on"
            )
        if args.launch_skip is not None:
            protocol["ncu_launch_skip"] = _canonical_selector(
                args.launch_skip, option="--launch-skip",
            )
        if args.launch_count is not None:
            protocol["ncu_launch_count"] = _canonical_selector(
                args.launch_count, option="--launch-count",
            )
        explicit_driver_trigger = any(
            getattr(args, option) is not None
            for option in (
                "target_processes", "filter_mode", "replay_mode",
                "profile_from_start", "launch_skip", "launch_count",
            )
        )
        if explicit_driver_trigger:
            _require_complete_explicit_protocol(protocol)
        validate_measurement_protocol(protocol)
    except ValueError as exc:
        print(f"NCU_LAUNCH_CONTROL_INVALID: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
