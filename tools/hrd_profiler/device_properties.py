"""Device-properties manifest loader for the Maeda 2017 HRD
second-layer cache model.

Loads `tools/hrd_profiler/device_properties.json` and resolves
an entry by the 5-tuple `(arch, sku, driver_version,
cuda_toolkit_version, nvbit_build)`. Computes the manifest
file sha256 so downstream bundles can record provenance binding
under the `cache_model_geometry_provenance` bundle field.

Required entry fields:
  - ``l2_capacity_bytes`` (int)
  - ``sector_size_bytes`` (int; canonical 32)
  - ``num_l2_slices`` (int)
  - ``slice_hash_model`` (string; e.g. ``sector_id_mod_num_l2_slices``)
  - ``write_policy_model`` (string; e.g. ``write_allocate_non_inclusive``)
  - ``cta_interleave_model`` (string; e.g. ``sequential_per_tb`` /
    ``round_robin_per_sm``)
  - ``geometry_confidence`` (string; e.g. ``vendor_documented``,
    ``microbench_calibrated``, ``inferred``).

Required tuple-key fields:
  - ``arch`` (string; e.g. ``sm_86``)
  - ``sku`` (string; e.g. ``rtx_3090_ti``)
  - ``driver_version`` (string)
  - ``cuda_toolkit_version`` (string)
  - ``nvbit_build`` (string)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST_PATH = _HERE / "device_properties.json"

REQUIRED_TUPLE_KEYS: tuple[str, ...] = (
    "arch",
    "sku",
    "driver_version",
    "cuda_toolkit_version",
    "nvbit_build",
)

REQUIRED_GEOMETRY_FIELDS: tuple[str, ...] = (
    "l2_capacity_bytes",
    "sector_size_bytes",
    "num_l2_slices",
    "slice_hash_model",
    "write_policy_model",
    "cta_interleave_model",
    "geometry_confidence",
)

PLACEHOLDER_SOURCE_TOKENS: frozenset[str] = frozenset({
    "",
    "n/a",
    "na",
    "none",
    "not applicable",
    "unavailable",
    "unknown",
    "tbd",
})


class DevicePropertiesManifestError(Exception):
    """Raised when the manifest is malformed or an entry is
    missing required fields."""


class DevicePropertiesEntryNotFound(Exception):
    """Raised when no entry matches the requested 5-tuple."""


@dataclass(frozen=True)
class DevicePropertiesEntry:
    arch: str
    sku: str
    driver_version: str
    cuda_toolkit_version: str
    nvbit_build: str
    l2_capacity_bytes: int
    sector_size_bytes: int
    num_l2_slices: int
    slice_hash_model: str
    write_policy_model: str
    cta_interleave_model: str
    geometry_confidence: str
    manifest_sha256: str

    def to_geometry_provenance(
        self,
        *,
        num_l2_slices_source: str = "device_properties_manifest",
    ) -> dict[str, Any]:
        """Return the 9-field provenance dict the
        `cache_model_geometry_provenance` bundle field expects.
        """
        return {
            "l2_capacity_bytes": self.l2_capacity_bytes,
            "sector_size_bytes": self.sector_size_bytes,
            "num_l2_slices": self.num_l2_slices,
            "num_l2_slices_source": num_l2_slices_source,
            "slice_hash_model": self.slice_hash_model,
            "write_policy_model": self.write_policy_model,
            "cta_interleave_model": self.cta_interleave_model,
            "geometry_confidence": self.geometry_confidence,
            "manifest_sha256": self.manifest_sha256,
        }


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(
    manifest_path: Path | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Load and validate the manifest file.

    Returns ``(entries, manifest_sha256)``. Raises
    ``DevicePropertiesManifestError`` on structural defects.
    """
    if manifest_path is None:
        manifest_path = DEFAULT_MANIFEST_PATH
    if not manifest_path.is_file():
        raise DevicePropertiesManifestError(
            f"device-properties manifest not found: {manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DevicePropertiesManifestError(
            f"device-properties manifest is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise DevicePropertiesManifestError(
            f"manifest top-level must be a JSON object; got {type(payload).__name__}"
        )
    schema_version = payload.get("schema_version")
    if schema_version != "1.0":
        raise DevicePropertiesManifestError(
            f"unknown manifest schema_version: {schema_version!r}; "
            f"expected '1.0'"
        )
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise DevicePropertiesManifestError(
            f"manifest 'entries' must be a list; got {type(entries).__name__}"
        )
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise DevicePropertiesManifestError(
                f"entry {idx} is not a JSON object; got {type(entry).__name__}"
            )
        for key in REQUIRED_TUPLE_KEYS:
            if key not in entry:
                raise DevicePropertiesManifestError(
                    f"entry {idx} missing required tuple key: {key!r}"
                )
        for field in REQUIRED_GEOMETRY_FIELDS:
            if field not in entry:
                raise DevicePropertiesManifestError(
                    f"entry {idx} ({entry.get('arch')!r}/"
                    f"{entry.get('sku')!r}) missing required "
                    f"geometry field: {field!r}"
                )
    return entries, _sha256_of_file(manifest_path)


def resolve_entry(
    *,
    arch: str,
    sku: str,
    driver_version: str,
    cuda_toolkit_version: str,
    nvbit_build: str,
    manifest_path: Path | None = None,
) -> DevicePropertiesEntry:
    """Resolve the 5-tuple to a ``DevicePropertiesEntry``.

    Raises ``DevicePropertiesEntryNotFound`` if no entry matches.
    """
    entries, manifest_sha256 = load_manifest(manifest_path=manifest_path)
    for entry in entries:
        if (
            entry["arch"] == arch
            and entry["sku"] == sku
            and entry["driver_version"] == driver_version
            and entry["cuda_toolkit_version"] == cuda_toolkit_version
            and entry["nvbit_build"] == nvbit_build
        ):
            return DevicePropertiesEntry(
                arch=entry["arch"],
                sku=entry["sku"],
                driver_version=entry["driver_version"],
                cuda_toolkit_version=entry["cuda_toolkit_version"],
                nvbit_build=entry["nvbit_build"],
                l2_capacity_bytes=int(entry["l2_capacity_bytes"]),
                sector_size_bytes=int(entry["sector_size_bytes"]),
                num_l2_slices=int(entry["num_l2_slices"]),
                slice_hash_model=str(entry["slice_hash_model"]),
                write_policy_model=str(entry["write_policy_model"]),
                cta_interleave_model=str(entry["cta_interleave_model"]),
                geometry_confidence=str(entry["geometry_confidence"]),
                manifest_sha256=manifest_sha256,
            )
    raise DevicePropertiesEntryNotFound(
        f"no manifest entry for (arch={arch!r}, sku={sku!r}, "
        f"driver_version={driver_version!r}, "
        f"cuda_toolkit_version={cuda_toolkit_version!r}, "
        f"nvbit_build={nvbit_build!r})"
    )


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "DevicePropertiesEntry",
    "DevicePropertiesEntryNotFound",
    "DevicePropertiesManifestError",
    "PLACEHOLDER_SOURCE_TOKENS",
    "REQUIRED_GEOMETRY_FIELDS",
    "REQUIRED_TUPLE_KEYS",
    "load_manifest",
    "resolve_entry",
]
