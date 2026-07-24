"""sm_120 NCU capability probe loader + harvester preflight gate.

The plan's the design contract requires every sm_120 harvest entry point to
validate the capability probe artifact before invoking NCU.
This module provides:

* ``load_capability_probe(path)`` — parse + validate the probe
  TOML, returning a frozen view of the host/metric inventory.
* ``CapabilityProbeError`` — typed-reason exception raised on
  any failure (missing field, forbidden metric family,
  unavailable required metric, schema malformed).
* ``preflight_for_arch(arch_tag, probe_path)`` — convenience
  entry point that loads the probe AND asserts the expected
  arch and go/no-go decision before any harvest argv is
  constructed.

The loader lives under the calibration subtree (Critical Rule
1's offline carve-out). The deployed framework path does NOT
import this module.
"""

from __future__ import annotations

from sol_baselines_harness.calibration.cache_prediction.retirement import (
    refuse_legacy_ncu_calibration,
    retired_legacy_ncu_calibration,
)

import tomllib
from dataclasses import dataclass
from pathlib import Path


_CAPABILITY_PROBE_TYPED_REASONS: frozenset = frozenset({
    "capability_probe_missing",
    "capability_probe_malformed_toml",
    "capability_probe_missing_required_field",
    "capability_probe_arch_mismatch",
    "capability_probe_no_go",
    "capability_probe_required_metric_unavailable",
    "capability_probe_forbidden_metric_family",
})


class CapabilityProbeError(Exception):
    """Raised by ``load_capability_probe`` / ``preflight_for_arch``
    on any failure to validate the sm_120 NCU capability probe.

    Carries a typed reason from
    ``_CAPABILITY_PROBE_TYPED_REASONS``.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        if typed_reason not in _CAPABILITY_PROBE_TYPED_REASONS:
            raise ValueError(
                f"unknown typed_reason {typed_reason!r}; expected "
                f"one of {sorted(_CAPABILITY_PROBE_TYPED_REASONS)}",
            )
        self.typed_reason = typed_reason
        self.message = message


# Required top-level [meta] fields.
_REQUIRED_META_FIELDS: tuple = (
    "schema_version",
    "captured_at_utc",
    "host_arch",
    "host_descriptor_hash",
    "overall_go_no_go",
)

# Required [host] fields naming the probe environment.
_REQUIRED_HOST_FIELDS: tuple = (
    "ncu_version",
    "ncu_cli_path",
    "driver_version",
    "compute_capability",
    "os_kernel",
    "perfmon_counter_availability",
    "clock_control_observed",
    "cache_control_observed",
)

# Store-conflated metric families that MUST NOT appear in a
# probe's required_metrics_for_predictor list (per CLAUDE.md
# Critical Rule 1 / the design contract metric-family discipline).
_FORBIDDEN_METRIC_FAMILIES: tuple = (
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_sector_hit_rate.pct",
    "syslts__t_sector_hit_rate.pct",
)


@dataclass(frozen=True)
class CapabilityProbe:
    """Frozen view of a validated NCU capability probe artifact."""

    schema_version: str
    captured_at_utc: str
    host_arch: str
    overall_go_no_go: str
    ncu_version: str
    driver_version: str
    compute_capability: str
    clock_control_observed: str
    accepted_metric_selectors: tuple
    raw_data: dict


def load_capability_probe(path) -> CapabilityProbe:
    """Parse and validate the capability probe TOML at ``path``.

    Raises ``CapabilityProbeError`` with a typed reason on:
      * Missing file.
      * Malformed TOML.
      * Missing required ``[meta]`` or ``[host]`` field.
      * Any required metric reported as not AVAILABLE.
      * Any accepted-metric selector matching a store-conflated
        forbidden family.

    The validator does NOT check the probe's ``overall_go_no_go``
    against an expected value — that is the
    ``preflight_for_arch`` caller's responsibility.
    """
    p = Path(path) if not isinstance(path, Path) else path
    if not p.exists():
        raise CapabilityProbeError(
            "capability_probe_missing",
            f"capability probe TOML at {p} does not exist",
        )
    try:
        data = tomllib.loads(p.read_text("utf-8"))
    except Exception as exc:
        raise CapabilityProbeError(
            "capability_probe_malformed_toml",
            f"capability probe at {p} does not parse as TOML: {exc!r}",
        ) from exc

    meta = data.get("meta") or {}
    for field in _REQUIRED_META_FIELDS:
        if field not in meta:
            raise CapabilityProbeError(
                "capability_probe_missing_required_field",
                f"capability probe [meta] missing required field "
                f"{field!r}",
            )

    host = data.get("host") or {}
    for field in _REQUIRED_HOST_FIELDS:
        if field not in host:
            raise CapabilityProbeError(
                "capability_probe_missing_required_field",
                f"capability probe [host] missing required field "
                f"{field!r}",
            )

    required_metrics_block = data.get(
        "required_metrics_for_predictor"
    ) or {}
    entries = required_metrics_block.get("entries") or []
    if not entries:
        raise CapabilityProbeError(
            "capability_probe_missing_required_field",
            "capability probe lacks "
            "[[required_metrics_for_predictor.entries]] entries",
        )
    accepted_metrics: list = []
    for entry in entries:
        name = entry.get("name")
        status = entry.get("status")
        if name is None or status is None:
            raise CapabilityProbeError(
                "capability_probe_missing_required_field",
                f"required-metrics entry missing name or status: "
                f"{entry!r}",
            )
        if status != "available":
            raise CapabilityProbeError(
                "capability_probe_required_metric_unavailable",
                f"required metric {name!r} reports status "
                f"{status!r} (expected 'available')",
            )
        for forbidden in _FORBIDDEN_METRIC_FAMILIES:
            if name == forbidden:
                raise CapabilityProbeError(
                    "capability_probe_forbidden_metric_family",
                    f"accepted metric selector {name!r} is in the "
                    f"forbidden store-conflated family",
                )
        accepted_metrics.append(name)

    return CapabilityProbe(
        schema_version=meta["schema_version"],
        captured_at_utc=meta["captured_at_utc"],
        host_arch=meta["host_arch"],
        overall_go_no_go=meta["overall_go_no_go"],
        ncu_version=host["ncu_version"],
        driver_version=host["driver_version"],
        compute_capability=host["compute_capability"],
        clock_control_observed=host["clock_control_observed"],
        accepted_metric_selectors=tuple(accepted_metrics),
        raw_data=data,
    )


def harvest_sm120_with_preflight(
    binary,
    *,
    launch_skip_count: int,
    probe_path,
    binary_args: tuple = (),
    ncu_metric_version: str = "ncu-2026.1",
    binary_sha256: str = "",
    warmup_protocol: str = "ncu_launch_skip_n_count_1",
):
    """sm_120 harvest entry point that gates on the capability
    probe BEFORE invoking NCU.

    Calls ``preflight_for_arch("sm_120", probe_path)``; on
    success delegates to
    ``harvest_ncu_label_with_provenance``. On preflight failure
    raises ``CapabilityProbeError`` directly so the caller sees
    the typed reason without spawning an NCU subprocess.
    """
    refuse_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.ncu_capability_probe.harvest_sm120_with_preflight')
    preflight_for_arch("sm_120", probe_path)
    from .ncu_harvester import harvest_ncu_label_with_provenance
    return harvest_ncu_label_with_provenance(
        binary,
        launch_skip_count=launch_skip_count,
        binary_args=binary_args,
        ncu_metric_version=ncu_metric_version,
        binary_sha256=binary_sha256,
        warmup_protocol=warmup_protocol,
    )


def preflight_for_arch(
    arch_tag: str, probe_path,
) -> CapabilityProbe:
    """Load + validate the capability probe AND assert
    ``host_arch == arch_tag`` AND
    ``overall_go_no_go == "go"`` before any harvest argv is
    constructed.

    Raises ``CapabilityProbeError`` on any mismatch. Returns
    the frozen probe view on success.
    """
    probe = load_capability_probe(probe_path)
    if probe.host_arch != arch_tag:
        raise CapabilityProbeError(
            "capability_probe_arch_mismatch",
            f"capability probe [meta].host_arch = "
            f"{probe.host_arch!r}; expected {arch_tag!r}",
        )
    if probe.overall_go_no_go != "go":
        raise CapabilityProbeError(
            "capability_probe_no_go",
            f"capability probe [meta].overall_go_no_go = "
            f"{probe.overall_go_no_go!r}; refusing to start NCU "
            f"harvest until the probe reports 'go'",
        )
    return probe


CAPABILITY_PROBE_TYPED_REASONS: frozenset = _CAPABILITY_PROBE_TYPED_REASONS


__all__ = [
    "CAPABILITY_PROBE_TYPED_REASONS",
    "CapabilityProbe",
    "CapabilityProbeError",
    "harvest_sm120_with_preflight",
    "load_capability_probe",
    "preflight_for_arch",
]
