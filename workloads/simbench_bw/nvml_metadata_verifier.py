"""NVML covariate metadata-pin verifier.

The reviewer's task20 handoff prescribed a verifier that detects
drift between a timing-mode `measurement_manifest.json`'s cached
NVML provenance fields and the on-disk
`nvml_covariate_<repeat>.json` sidecars. This module lands the
verifier as an offline (host-only, no GPU) helper that's reusable
by `verify_artifacts.py`, future CI gates, and the sm_120 evidence
sweep.

The cached fields R60 emits in the manifest:

* ``nvml_polling_interval_ns``: the cadence the collector targeted
  (default ``NVML_DEFAULT_POLLING_INTERVAL_NS = 5_000_000`` ns).
* ``nvml_claim_scope``: the cross-die claim-scope label
  (``"measured_local"`` for Loop 2).
* ``nvml_sidecar_paths``: relative filenames of the per-repeat
  sidecars (``["nvml_covariate_0.json", "nvml_covariate_1.json", ...]``).
* ``nvml_collector_status``: union of per-repeat collector
  statuses; one of the four sidecar enum values
  (``"ok"``, ``"nvml_covariate_unavailable"``,
  ``"nvml_alignment_gap"``,
  ``"nvml_covariate_regression_ineffective"``) or
  ``"nvml_covariate_mixed"`` for a heterogeneous union.

The verifier walks each cached sidecar, schema-validates it, and
asserts:

1. The cached ``nvml_polling_interval_ns`` matches every sidecar's
   ``polling_interval_ns`` field.
2. The cached ``nvml_claim_scope`` matches every sidecar's
   ``claim_scope`` field.
3. The cached ``nvml_collector_status`` matches the recomputed
   union of per-repeat ``nvml_collector_status`` values.

A disagreement raises ``NvmlMetadataDriftError`` carrying the
specific field, the cached value, and the actual value. Manifests
without ANY of the cached NVML fields are admitted unchanged because historical
manifests may predate NVML-sidecar capture. The retired live CUPTI producer is
not an emission dependency of this replay verifier.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from validate import ValidationError, validate  # type: ignore[import-not-found]  # noqa: E402

_NVML_SCHEMA_PATH = _HERE / "schemas" / "nvml_covariate.schema.json"


def _validate_nvml_sidecar(sidecar_path: Path) -> list[str]:
    """Strict-validate a per-repeat NVML covariate sidecar
    against ``schemas/nvml_covariate.schema.json``. Returns a list
    of error strings (empty list on admission)."""
    try:
        schema = json.loads(_NVML_SCHEMA_PATH.read_text())
        instance = json.loads(sidecar_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"load_failure: {exc}"]
    try:
        validate(
            instance, schema,
            root_schema=schema,
            root_path=_NVML_SCHEMA_PATH,
            path="$",
        )
    except ValidationError as exc:
        return [str(exc)]
    return []

# The four sidecar collector-status enum values plus the manifest-
# only union value. Mirrors the
# ``nvml_covariate.schema.json::nvml_collector_status_enum`` plus
# the ``measurement_manifest.schema.json::nvml_collector_status``
# enum's extra ``nvml_covariate_mixed`` entry.
_SIDECAR_COLLECTOR_STATUSES = frozenset({
    "ok",
    "nvml_covariate_unavailable",
    "nvml_alignment_gap",
    "nvml_covariate_regression_ineffective",
})


class NvmlMetadataDriftError(Exception):
    """A timing-mode manifest's cached NVML provenance field
    disagrees with the on-disk sidecar(s).

    The error names the field that drifted plus the cached and
    actual values so the operator can decide whether the manifest
    is stale (re-run the timing sweep to refresh) or the sidecar
    was tampered with (investigate).
    """

    TOKEN = "nvml_metadata_pin_drift"

    def __init__(
        self,
        field: str,
        expected: object,
        actual: object,
        manifest_path: Path | None = None,
    ) -> None:
        loc = f" manifest={manifest_path}" if manifest_path else ""
        super().__init__(
            f"{self.TOKEN}: field={field!r} expected={expected!r} "
            f"actual={actual!r}{loc}"
        )
        self.field = field
        self.expected = expected
        self.actual = actual
        self.manifest_path = manifest_path


class NvmlSidecarMissingError(Exception):
    """A sidecar named in the manifest's
    ``nvml_sidecar_paths`` list does not exist on disk."""

    TOKEN = "nvml_metadata_pin_sidecar_missing"

    def __init__(
        self, sidecar_path: Path, manifest_path: Path | None = None,
    ) -> None:
        loc = f" manifest={manifest_path}" if manifest_path else ""
        super().__init__(
            f"{self.TOKEN}: sidecar {sidecar_path} listed in "
            f"manifest's nvml_sidecar_paths but does not exist{loc}"
        )
        self.sidecar_path = sidecar_path
        self.manifest_path = manifest_path


class NvmlSidecarSchemaInvalidError(Exception):
    """A sidecar named in the manifest's
    ``nvml_sidecar_paths`` list fails strict schema
    validation."""

    TOKEN = "nvml_metadata_pin_sidecar_schema_invalid"

    def __init__(
        self,
        sidecar_path: Path,
        errors: list[str],
        manifest_path: Path | None = None,
    ) -> None:
        loc = f" manifest={manifest_path}" if manifest_path else ""
        super().__init__(
            f"{self.TOKEN}: sidecar {sidecar_path} fails strict "
            f"schema validation: {errors}{loc}"
        )
        self.sidecar_path = sidecar_path
        self.errors = errors
        self.manifest_path = manifest_path


def _summarize_collector_statuses(statuses: list[str]) -> str:
    """Recompute the historical manifest union locally.

    Single-distinct → that value; heterogeneous →
    ``nvml_covariate_mixed``; empty → ``nvml_covariate_unavailable``.
    No living `run_suite` producer helper owns this replay rule.
    """
    if not statuses:
        return "nvml_covariate_unavailable"
    distinct = set(statuses)
    if len(distinct) == 1:
        return distinct.pop()
    return "nvml_covariate_mixed"


def verify_nvml_metadata_pin(
    manifest_path: Path,
    sidecar_dir: Path | None = None,
) -> None:
    """Verify that ``manifest_path``'s cached NVML provenance
    fields agree with the on-disk
    ``nvml_covariate_<repeat>.json`` sidecars.

    Returns ``None`` on admission. Raises
    ``NvmlMetadataDriftError``, ``NvmlSidecarMissingError``, or
    ``NvmlSidecarSchemaInvalidError`` on failure.

    A manifest without ANY cached NVML field is admitted cleanly as historical
    replay evidence; this verifier does not enable or invoke a live collector.

    ``sidecar_dir`` defaults to the manifest's parent
    directory (the canonical layout produced by
    ``run_suite.py::handle_timing``).
    """
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise NvmlMetadataDriftError(
            "manifest", "object", type(manifest).__name__,
            manifest_path=manifest_path,
        )
    cached_paths = manifest.get("nvml_sidecar_paths")
    cached_interval = manifest.get("nvml_polling_interval_ns")
    cached_scope = manifest.get("nvml_claim_scope")
    cached_status = manifest.get("nvml_collector_status")
    cached_fields = (
        cached_paths,
        cached_interval,
        cached_scope,
        cached_status,
    )
    if all(v is None for v in cached_fields):
        # No NVML fields cached → emission was opt-out. Admit.
        return
    if any(v is None for v in cached_fields):
        # Partial cache → manifest is malformed; reject the
        # specific missing field.
        for name, value in zip(
            (
                "nvml_sidecar_paths",
                "nvml_polling_interval_ns",
                "nvml_claim_scope",
                "nvml_collector_status",
            ),
            cached_fields,
        ):
            if value is None:
                raise NvmlMetadataDriftError(
                    name, "<present>", None,
                    manifest_path=manifest_path,
                )
    if not isinstance(cached_paths, list) or not cached_paths:
        raise NvmlMetadataDriftError(
            "nvml_sidecar_paths",
            "non-empty list of strings",
            cached_paths,
            manifest_path=manifest_path,
        )
    base = sidecar_dir if sidecar_dir is not None else manifest_path.parent
    statuses: list[str] = []
    for relpath in cached_paths:
        sidecar_path = base / relpath
        if not sidecar_path.is_file():
            raise NvmlSidecarMissingError(
                sidecar_path, manifest_path=manifest_path,
            )
        errors = _validate_nvml_sidecar(sidecar_path)
        if errors:
            raise NvmlSidecarSchemaInvalidError(
                sidecar_path, errors,
                manifest_path=manifest_path,
            )
        sidecar = json.loads(sidecar_path.read_text())
        sc_interval = sidecar.get("polling_interval_ns")
        if sc_interval != cached_interval:
            raise NvmlMetadataDriftError(
                "polling_interval_ns",
                cached_interval, sc_interval,
                manifest_path=manifest_path,
            )
        sc_scope = sidecar.get("claim_scope")
        if sc_scope != cached_scope:
            raise NvmlMetadataDriftError(
                "claim_scope",
                cached_scope, sc_scope,
                manifest_path=manifest_path,
            )
        sc_status = sidecar.get("nvml_collector_status")
        if sc_status not in _SIDECAR_COLLECTOR_STATUSES:
            raise NvmlMetadataDriftError(
                "nvml_collector_status",
                "valid sidecar status enum value",
                sc_status,
                manifest_path=manifest_path,
            )
        statuses.append(sc_status)
    recomputed = _summarize_collector_statuses(statuses)
    if recomputed != cached_status:
        raise NvmlMetadataDriftError(
            "nvml_collector_status",
            cached_status, recomputed,
            manifest_path=manifest_path,
        )


__all__ = (
    "NvmlMetadataDriftError",
    "NvmlSidecarMissingError",
    "NvmlSidecarSchemaInvalidError",
    "verify_nvml_metadata_pin",
)
