"""Per-launch label sidecar JSON emitter.

`tools/local_labels/label_driver.sh` invokes this module to build the
sidecar JSON dict after the parser CLI returns. Factoring the emission
out of the heredoc lets the sidecar shape (including the optional
`parser_diagnostics` field) be unit-tested without launching NCU.

Per-launch input arrives as environment variables (`SIDECAR_*`) so the
shell side does not have to interpolate dynamic values into Python
source. The optional `SIDECAR_PARSER_DIAGNOSTICS_PATH` env var (or the
`diagnostics_path` kwarg on `build_sidecar`) points to a JSON file
written by `parse_ncu_metrics.py --diagnostics-json`; when present and
non-empty, its contents merge into the sidecar's `parser_diagnostics`
field. When the path is absent / unreadable / empty, the sidecar omits
the field — consistent with the schema's optional-field semantics.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

# Reuse the SINGLE canonical build-variant identity normalizer (rather than
# re-deriving it here, which could diverge — the failure class that normalizer
# was hardened against). It lives in a dependency-free module so this emitter
# stays stdlib-only: label_driver.sh invokes this script under a bare system
# python3 that may not have jsonschema, so importing the jsonschema-bearing
# repeatability_aggregator here would crash the live capture. Path-robust
# sibling import so this works both as a script and when imported as a library.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from ncu_capture_contract import validate_measurement_protocol  # noqa: E402
from variant_identity import variant_identity as _variant_identity  # noqa: E402


def build_sidecar(
    env: Mapping[str, str],
    *,
    diagnostics_path: str | None = None,
    unit_tagged_metrics_path: str | None = None,
) -> dict[str, Any]:
    """Build the per-launch sidecar dict from `env`.

    Required env vars: `SIDECAR_DEVICE`, `SIDECAR_SLUG`,
    `SIDECAR_SHAPE_POINT`, `SIDECAR_REPEAT_INDEX`,
    `SIDECAR_LAUNCH_INDEX`, `SIDECAR_BINARY_SHA256`,
    `SIDECAR_PROFILER_VERSION`, `SIDECAR_CUDA_VERSION`,
    `SIDECAR_DRIVER_VERSION`, `SIDECAR_GPU_UUID_SHA256`,
    `SIDECAR_AGGREGATION`, `SIDECAR_UNITS`,
    `SIDECAR_CONFIDENCE_LEVEL`, `SIDECAR_WARMUP_COUNT`,
    `SIDECAR_HIT_RATE_L2`, `SIDECAR_HIT_RATE_L1TEX`.

    When `diagnostics_path` is a readable JSON file whose root is a
    non-empty object, its contents merge into `sidecar["parser_diagnostics"]`.
    Any other shape (missing file, unparseable JSON, empty object,
    non-object root) omits the field, matching the schema's optional
    semantics.
    """

    def _env_str(name: str) -> str:
        value = env.get(name)
        if value is None:
            raise ValueError(f"missing required env var {name}")
        return value

    def _env_int(name: str) -> int:
        return int(_env_str(name))

    def _env_float(name: str) -> float:
        return float(_env_str(name))

    hit_rate_l2 = _env_float("SIDECAR_HIT_RATE_L2")
    hit_rate_l1tex = _env_float("SIDECAR_HIT_RATE_L1TEX")
    confidence_level = _env_float("SIDECAR_CONFIDENCE_LEVEL")
    units = _env_str("SIDECAR_UNITS")
    aggregation = _env_str("SIDECAR_AGGREGATION")

    sidecar: dict[str, Any] = {
        "schema_version": 5,
        "device": _env_str("SIDECAR_DEVICE"),
        "slug": _env_str("SIDECAR_SLUG"),
        "shape_point": _env_str("SIDECAR_SHAPE_POINT"),
        "repeat_index": _env_int("SIDECAR_REPEAT_INDEX"),
        "launch_index": _env_int("SIDECAR_LAUNCH_INDEX"),
        "binary_sha256": _env_str("SIDECAR_BINARY_SHA256"),
        "profiler_family": "nsight_compute",
        "profiler_version": _env_str("SIDECAR_PROFILER_VERSION"),
        "cuda_version": _env_str("SIDECAR_CUDA_VERSION"),
        "driver_version": _env_str("SIDECAR_DRIVER_VERSION"),
        "gpu_uuid_sha256": _env_str("SIDECAR_GPU_UUID_SHA256"),
        "replay_mode": "kernel",
        "aggregation": aggregation,
        "warmup_count": _env_int("SIDECAR_WARMUP_COUNT"),
        "cache_hit_rate_l2_sector": hit_rate_l2,
        "confidence_interval_l2_sector": {
            "lower": hit_rate_l2,
            "upper": hit_rate_l2,
            "confidence_level": confidence_level,
            "units": units,
            "aggregation": aggregation,
        },
        "cache_hit_rate_l1tex_sector": hit_rate_l1tex,
        "confidence_interval_l1tex_sector": {
            "lower": hit_rate_l1tex,
            "upper": hit_rate_l1tex,
            "confidence_level": confidence_level,
            "units": units,
            "aggregation": aggregation,
        },
    }

    # v2.1 protocol (CLAUDE.md Rule 2, 2026-07-06): optional
    # measurement-protocol provenance — the explicit NCU clock-control /
    # cache-control / replay-mode plus the observed host clock, persistence,
    # power and temperature state at emission time (one POST-capture sample;
    # see label_driver.sh for the semantics caveat). Emitted only when the
    # driver provides SIDECAR_NCU_CLOCK_CONTROL. NOTE: the driver now always
    # provides it (default "none"), so every driver-mediated sidecar carries
    # this block; only direct build_sidecar() callers without the env remain
    # byte-compatible with pre-v2 output.
    ncu_clock_control = env.get("SIDECAR_NCU_CLOCK_CONTROL", "")
    explicit_ncu_env_names = (
        "SIDECAR_NCU_CACHE_CONTROL",
        "SIDECAR_NCU_REPLAY_MODE",
        "SIDECAR_NCU_TARGET_PROCESSES",
        "SIDECAR_NCU_FILTER_MODE",
        "SIDECAR_NCU_PROFILE_FROM_START",
        "SIDECAR_NCU_LAUNCH_SKIP",
        "SIDECAR_NCU_LAUNCH_COUNT",
    )
    if any(env.get(name, "") for name in explicit_ncu_env_names) and not ncu_clock_control:
        raise ValueError(
            "explicit NCU sidecar controls require SIDECAR_NCU_CLOCK_CONTROL"
        )
    if ncu_clock_control:
        if ncu_clock_control not in ("base", "boost", "none"):
            raise ValueError(
                "SIDECAR_NCU_CLOCK_CONTROL must be base|boost|none, got "
                f"{ncu_clock_control!r}"
            )
        protocol: dict[str, Any] = {"ncu_clock_control": ncu_clock_control}
        for env_name, field in (
            ("SIDECAR_NCU_CACHE_CONTROL", "ncu_cache_control"),
            ("SIDECAR_NCU_REPLAY_MODE", "ncu_replay_mode"),
            ("SIDECAR_HOST_PERSISTENCE_MODE", "host_persistence_mode"),
            ("SIDECAR_HOST_THROTTLE_REASONS", "host_throttle_reasons_active"),
        ):
            raw = env.get(env_name, "")
            if raw:
                protocol[field] = raw

        selector_values = {
            "ncu_target_processes": env.get("SIDECAR_NCU_TARGET_PROCESSES", ""),
            "ncu_filter_mode": env.get("SIDECAR_NCU_FILTER_MODE", ""),
            "ncu_profile_from_start": env.get("SIDECAR_NCU_PROFILE_FROM_START", ""),
            "ncu_launch_skip": env.get("SIDECAR_NCU_LAUNCH_SKIP", ""),
            "ncu_launch_count": env.get("SIDECAR_NCU_LAUNCH_COUNT", ""),
        }
        if any(selector_values.values()):
            missing = sorted(
                field for field, raw in selector_values.items() if not raw
            )
            if missing:
                raise ValueError(
                    "partial explicit NCU sidecar controls; missing "
                    f"{missing!r}"
                )
            protocol["ncu_target_processes"] = selector_values[
                "ncu_target_processes"
            ]
            protocol["ncu_filter_mode"] = selector_values["ncu_filter_mode"]
            profile_from_start = selector_values["ncu_profile_from_start"]
            if profile_from_start not in {"on", "off"}:
                raise ValueError(
                    "SIDECAR_NCU_PROFILE_FROM_START must be on|off"
                )
            protocol["ncu_profile_from_start"] = profile_from_start == "on"
            for field in ("ncu_launch_skip", "ncu_launch_count"):
                raw = selector_values[field]
                if not raw.isascii() or not raw.isdecimal() or (
                    len(raw) > 1 and raw.startswith("0")
                ):
                    raise ValueError(
                        f"SIDECAR_{field.upper()} must be a canonical unsigned decimal"
                    )
                protocol[field] = int(raw)

        for env_name, field in (
            ("SIDECAR_HOST_GPU_INDEX", "host_gpu_index"),
            ("SIDECAR_HOST_SM_CLOCK_MHZ", "host_sm_clock_mhz"),
            ("SIDECAR_HOST_MAX_SM_CLOCK_MHZ", "host_max_sm_clock_mhz"),
            ("SIDECAR_HOST_MEM_CLOCK_MHZ", "host_mem_clock_mhz"),
            ("SIDECAR_HOST_TEMP_C", "host_temperature_c"),
        ):
            raw = env.get(env_name, "")
            if raw:
                protocol[field] = int(raw)
        raw = env.get("SIDECAR_HOST_POWER_W", "")
        if raw:
            protocol["host_power_draw_w"] = float(raw)
        validate_measurement_protocol(protocol)
        sidecar["measurement_protocol"] = protocol

    diagnostics_obj = _load_diagnostics(diagnostics_path)
    if diagnostics_obj:
        sidecar["parser_diagnostics"] = diagnostics_obj

    unit_tagged = _load_unit_tagged_metrics(unit_tagged_metrics_path)
    if unit_tagged:
        sidecar["unit_tagged_metrics"] = unit_tagged

    # Optional build-variant identity. Present ONLY for a register-cap build
    # variant (both fields omitted for a base cell ⇒ byte-identical sidecar).
    # A noncanonical alias / inconsistent pair raises ValueError so the caller
    # fails closed (no contradictory sidecar is written).
    sidecar.update(_variant_fields(env))

    return sidecar


def _load_unit_tagged_metrics(path: str | None) -> list[Any]:
    """Return the parsed unit-tagged-metrics array at `path`.

    When `path` is None / empty the producer did not request unit-tagged
    metrics; return `[]` so `build_sidecar` omits the optional field (the
    legacy L2/L1-only path stays byte-compatible apart from the
    schema_version bump). When `path` IS explicitly set, fail CLOSED: an
    unreadable file, malformed JSON, or a non-array root raises `ValueError`
    rather than silently dropping the metrics the producer asked for — a
    silent drop would let a capture run emit an L2/L1-only sidecar while
    claiming it carried timing/traffic labels."""
    if not path:
        return []
    try:
        with Path(path).open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"SIDECAR_UNIT_TAGGED_METRICS_PATH={path!r} is set but unreadable "
            f"or malformed: {exc}"
        ) from exc
    if not isinstance(data, list):
        raise ValueError(
            f"SIDECAR_UNIT_TAGGED_METRICS_PATH={path!r} content root is "
            f"{type(data).__name__}, expected a JSON array of unit-tagged "
            f"metric records"
        )
    return data


def _variant_fields(env: Mapping[str, str]) -> dict[str, Any]:
    """Return the OPTIONAL build-variant identity fields for the sidecar.

    Reads ``SIDECAR_REGISTER_CAP`` (integer string) and ``SIDECAR_VARIANT_ID``
    (string). Both absent/empty ⇒ base cell ⇒ ``{}`` (the sidecar omits both,
    byte-identical to the pre-variant shape). When EITHER is present the pair is
    normalized through the SINGLE canonical normalizer
    (``repeatability_aggregator._variant_identity``): register_cap-only derives
    the canonical ``variant_id``; variant_id-only requires the canonical decimal
    form (a leading-zero alias like ``register_cap_064`` is rejected); both
    present require exact canonical string equality. A noncanonical alias or a
    ``register_cap``/``variant_id`` disagreement raises ``ValueError`` so the
    producer fails closed and never writes a contradictory sidecar.
    """
    raw_cap = (env.get("SIDECAR_REGISTER_CAP") or "").strip()
    raw_variant_id = (env.get("SIDECAR_VARIANT_ID") or "").strip()
    if not raw_cap and not raw_variant_id:
        return {}
    record: dict[str, Any] = {}
    if raw_cap:
        try:
            record["register_cap"] = int(raw_cap)
        except ValueError as exc:
            raise ValueError(
                f"SIDECAR_REGISTER_CAP={raw_cap!r} is not an integer"
            ) from exc
    if raw_variant_id:
        record["variant_id"] = raw_variant_id
    variant_id, register_cap, error = _variant_identity(record)
    if error is not None:
        raise ValueError(f"build-variant identity inconsistent: {error}")
    fields: dict[str, Any] = {}
    if variant_id:
        fields["variant_id"] = variant_id
        if register_cap is not None:
            fields["register_cap"] = register_cap
    return fields


def _load_diagnostics(path: str | None) -> dict[str, Any]:
    """Return the parsed diagnostics object at `path`, or `{}` for any
    failure mode (missing path, unreadable file, malformed JSON,
    non-object root). The empty-dict return causes `build_sidecar` to
    omit the `parser_diagnostics` field, matching the schema's
    optional-field semantics."""
    if not path:
        return {}
    try:
        with Path(path).open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def main(argv: list[str] | None = None) -> int:
    """CLI entry point used by `label_driver.sh`: read `SIDECAR_*` env
    vars + optional `SIDECAR_PARSER_DIAGNOSTICS_PATH`, write the
    sidecar JSON to stdout."""
    diagnostics_path = os.environ.get("SIDECAR_PARSER_DIAGNOSTICS_PATH")
    unit_tagged_metrics_path = os.environ.get("SIDECAR_UNIT_TAGGED_METRICS_PATH")
    try:
        sidecar = build_sidecar(
            os.environ,
            diagnostics_path=diagnostics_path,
            unit_tagged_metrics_path=unit_tagged_metrics_path,
        )
    except ValueError as exc:
        sys.stderr.write(f"sidecar_emitter: {exc}\n")
        return 1
    json.dump(sidecar, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
