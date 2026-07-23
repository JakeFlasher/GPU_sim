"""Host-resource availability manifest loader (Phase-1 sm_86 host).

Parallel companion to `host_arch_availability_v1.toml`. Where the
host-arch manifest records `host_arch_unsupported` carve-outs at the
whole-problem (slug) granularity, this manifest records per-workload
carve-outs for workloads that exceed the host's resource envelope.

The most common case is "PyTorch reference OOMs because intermediate
fp32 tensors exceed host VRAM". The kernel-side implementation is
provably correct; only the reference computation cannot run on this
host class.

This module is import-only inside calibration / diagnostic paths; the
NCU quarantine allow-list already accepts every module that lives
under `scripts/baselines_harness/calibration/cache_prediction/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

try:
    import tomllib
except ImportError:  # pragma: no cover - Python < 3.11 fallback path
    import tomli as tomllib  # type: ignore[no-redef]


HOST_RESOURCE_UNAVAILABLE_TYPED_REASONS: frozenset[str] = frozenset({
    "reference_oom_exceeds_local_vram",
    "reference_timeout_exceeds_local_eval_window",
    "correctness_check_oom_exceeds_local_vram",
    # Timing-phase OOM: the kernel-side correctness rounds pass (with
    # the chunked correctness path bounding peak intermediate memory),
    # but the timing phase that follows cannot fit a fresh user-kernel
    # invocation into the caching allocator's leftover free space.
    # The fragmentation is host-VRAM bound; on a 192 GiB B200 host the
    # same allocations have orders-of-magnitude headroom.
    "timing_oom_exceeds_local_vram",
})


HOST_RESOURCE_MANIFEST_TYPED_REASONS: frozenset[str] = frozenset({
    "host_resource_manifest_schema_version_unsupported",
    "host_resource_manifest_missing_meta_field",
    "host_resource_manifest_missing_workload_field",
    "host_resource_manifest_unknown_typed_reason",
    "host_resource_manifest_duplicate_workload_uuid",
    "host_resource_manifest_missing_slug_field",
    "host_resource_manifest_duplicate_slug",
    "host_resource_manifest_unknown_host_class",
})


# Closed mapping from a host_class (the TOML manifest's
# ``[manifest_meta].host_class`` field) to the sibling-file name
# that carries that host's carve-outs. Adding a new host requires
# adding both the sibling TOML and an entry here.
_HOST_CLASS_TO_MANIFEST_FILENAME: dict[str, str] = {
    "local_sm_86_3090_ti_24gib": "host_resource_availability_v1.toml",
    "local_sm_120_rtx_5060_laptop_8gib": (
        "host_resource_availability_v1_sm120_rtx_5060_laptop_8gib.toml"
    ),
}


class HostResourceAvailabilityError(Exception):
    """Manifest load / validation failure with a typed reason."""

    def __init__(self, typed_reason: str, detail: str) -> None:
        if typed_reason not in HOST_RESOURCE_MANIFEST_TYPED_REASONS:
            raise AssertionError(
                f"unknown typed reason {typed_reason!r} not in closed set"
            )
        super().__init__(f"{typed_reason}: {detail}")
        self.typed_reason = typed_reason
        self.detail = detail


@dataclass(frozen=True)
class HostResourceUnavailableEntry:
    slug: str
    workload_uuid: str
    workload_axes: str
    typed_reason: str
    peak_intermediate_gib: int
    host_vram_gib: int
    would_pass_on: str
    notes: str


@dataclass(frozen=True)
class HostResourceUnavailableSlugEntry:
    slug: str
    typed_reason: str
    peak_resident_gib: int
    host_vram_gib: int
    would_pass_on: str
    notes: str


@dataclass(frozen=True)
class HostResourceAvailabilityManifest:
    schema_version: int
    host_class: str
    host_total_vram_gib: int
    created_at_utc: str
    updated_at_utc: str
    unavailable_workloads: Tuple[HostResourceUnavailableEntry, ...] = field(
        default_factory=tuple
    )
    unavailable_slugs: Tuple[HostResourceUnavailableSlugEntry, ...] = field(
        default_factory=tuple
    )

    def lookup(self, workload_uuid: str) -> HostResourceUnavailableEntry | None:
        for entry in self.unavailable_workloads:
            if entry.workload_uuid == workload_uuid:
                return entry
        return None

    def lookup_slug(self, slug: str) -> HostResourceUnavailableSlugEntry | None:
        for entry in self.unavailable_slugs:
            if entry.slug == slug:
                return entry
        return None


_REQUIRED_META_FIELDS = (
    "schema_version",
    "host_class",
    "host_total_vram_gib",
    "created_at_utc",
    "updated_at_utc",
)

_REQUIRED_WORKLOAD_FIELDS = (
    "slug",
    "workload_uuid",
    "workload_axes",
    "typed_reason",
    "peak_intermediate_gib",
    "host_vram_gib",
    "would_pass_on",
    "notes",
)

_REQUIRED_SLUG_FIELDS = (
    "slug",
    "typed_reason",
    "peak_resident_gib",
    "host_vram_gib",
    "would_pass_on",
    "notes",
)


def load_host_resource_availability_manifest(
    path: Path,
) -> HostResourceAvailabilityManifest:
    """Parse a host-resource availability manifest, fail closed.

    Raises ``HostResourceAvailabilityError`` with a typed reason from
    ``HOST_RESOURCE_MANIFEST_TYPED_REASONS`` on any schema breach.
    """
    raw_bytes = path.read_bytes()
    data = tomllib.loads(raw_bytes.decode("utf-8"))

    meta = data.get("manifest_meta")
    if not isinstance(meta, dict):
        raise HostResourceAvailabilityError(
            "host_resource_manifest_missing_meta_field",
            "manifest_meta table missing or not a table",
        )
    for field_name in _REQUIRED_META_FIELDS:
        if field_name not in meta:
            raise HostResourceAvailabilityError(
                "host_resource_manifest_missing_meta_field",
                f"manifest_meta.{field_name} missing",
            )

    if meta["schema_version"] != 1:
        raise HostResourceAvailabilityError(
            "host_resource_manifest_schema_version_unsupported",
            f"unsupported schema_version: {meta['schema_version']!r}",
        )

    workloads_raw = data.get("unavailable_workloads", [])
    if not isinstance(workloads_raw, list):
        raise HostResourceAvailabilityError(
            "host_resource_manifest_missing_workload_field",
            "unavailable_workloads must be an array",
        )

    seen_uuids: set[str] = set()
    workloads: list[HostResourceUnavailableEntry] = []
    for index, entry in enumerate(workloads_raw):
        if not isinstance(entry, dict):
            raise HostResourceAvailabilityError(
                "host_resource_manifest_missing_workload_field",
                f"unavailable_workloads[{index}] is not a table",
            )
        for field_name in _REQUIRED_WORKLOAD_FIELDS:
            if field_name not in entry:
                raise HostResourceAvailabilityError(
                    "host_resource_manifest_missing_workload_field",
                    f"unavailable_workloads[{index}].{field_name} missing",
                )
        if entry["typed_reason"] not in HOST_RESOURCE_UNAVAILABLE_TYPED_REASONS:
            raise HostResourceAvailabilityError(
                "host_resource_manifest_unknown_typed_reason",
                f"unavailable_workloads[{index}].typed_reason="
                f"{entry['typed_reason']!r} not in closed set",
            )
        uuid_val = entry["workload_uuid"]
        if uuid_val in seen_uuids:
            raise HostResourceAvailabilityError(
                "host_resource_manifest_duplicate_workload_uuid",
                f"workload_uuid={uuid_val!r} appears multiple times",
            )
        seen_uuids.add(uuid_val)
        workloads.append(
            HostResourceUnavailableEntry(
                slug=str(entry["slug"]),
                workload_uuid=str(uuid_val),
                workload_axes=str(entry["workload_axes"]),
                typed_reason=str(entry["typed_reason"]),
                peak_intermediate_gib=int(entry["peak_intermediate_gib"]),
                host_vram_gib=int(entry["host_vram_gib"]),
                would_pass_on=str(entry["would_pass_on"]),
                notes=str(entry["notes"]),
            )
        )

    slugs_raw = data.get("unavailable_slugs", [])
    if not isinstance(slugs_raw, list):
        raise HostResourceAvailabilityError(
            "host_resource_manifest_missing_slug_field",
            "unavailable_slugs must be an array",
        )
    seen_slugs: set[str] = set()
    slugs: list[HostResourceUnavailableSlugEntry] = []
    for index, entry in enumerate(slugs_raw):
        if not isinstance(entry, dict):
            raise HostResourceAvailabilityError(
                "host_resource_manifest_missing_slug_field",
                f"unavailable_slugs[{index}] is not a table",
            )
        for field_name in _REQUIRED_SLUG_FIELDS:
            if field_name not in entry:
                raise HostResourceAvailabilityError(
                    "host_resource_manifest_missing_slug_field",
                    f"unavailable_slugs[{index}].{field_name} missing",
                )
        if entry["typed_reason"] not in HOST_RESOURCE_UNAVAILABLE_TYPED_REASONS:
            raise HostResourceAvailabilityError(
                "host_resource_manifest_unknown_typed_reason",
                f"unavailable_slugs[{index}].typed_reason="
                f"{entry['typed_reason']!r} not in closed set",
            )
        slug_val = entry["slug"]
        if slug_val in seen_slugs:
            raise HostResourceAvailabilityError(
                "host_resource_manifest_duplicate_slug",
                f"slug={slug_val!r} appears multiple times",
            )
        seen_slugs.add(slug_val)
        slugs.append(
            HostResourceUnavailableSlugEntry(
                slug=str(slug_val),
                typed_reason=str(entry["typed_reason"]),
                peak_resident_gib=int(entry["peak_resident_gib"]),
                host_vram_gib=int(entry["host_vram_gib"]),
                would_pass_on=str(entry["would_pass_on"]),
                notes=str(entry["notes"]),
            )
        )

    return HostResourceAvailabilityManifest(
        schema_version=int(meta["schema_version"]),
        host_class=str(meta["host_class"]),
        host_total_vram_gib=int(meta["host_total_vram_gib"]),
        created_at_utc=str(meta["created_at_utc"]),
        updated_at_utc=str(meta["updated_at_utc"]),
        unavailable_workloads=tuple(workloads),
        unavailable_slugs=tuple(slugs),
    )


def discover_manifest_for_host_class(
    host_class: str,
    manifests_dir: Path | None = None,
) -> Path:
    """Resolve the sibling manifest path for a given host_class.

    ``manifests_dir`` defaults to the directory this module lives
    in, where the in-tree host-resource sibling files are kept.
    """
    if host_class not in _HOST_CLASS_TO_MANIFEST_FILENAME:
        raise HostResourceAvailabilityError(
            "host_resource_manifest_unknown_host_class",
            f"no host-resource manifest registered for host_class="
            f"{host_class!r}; registered host classes: "
            f"{sorted(_HOST_CLASS_TO_MANIFEST_FILENAME)!r}",
        )
    if manifests_dir is None:
        manifests_dir = Path(__file__).resolve().parent
    return manifests_dir / _HOST_CLASS_TO_MANIFEST_FILENAME[host_class]


def load_manifest_for_host_class(
    host_class: str,
    manifests_dir: Path | None = None,
) -> HostResourceAvailabilityManifest:
    """Discover and load the host-resource manifest for a host_class.

    Convenience wrapper around ``discover_manifest_for_host_class``
    plus ``load_host_resource_availability_manifest``. Verifies that
    the loaded manifest's own ``[manifest_meta].host_class`` matches
    the requested one (the lookup table is the truth source; a
    sibling file whose body claims a different host_class is
    rejected as a registration mismatch).
    """
    path = discover_manifest_for_host_class(host_class, manifests_dir)
    manifest = load_host_resource_availability_manifest(path)
    if manifest.host_class != host_class:
        raise HostResourceAvailabilityError(
            "host_resource_manifest_unknown_host_class",
            f"file {path!s} declares host_class={manifest.host_class!r} "
            f"but was looked up under {host_class!r}",
        )
    return manifest


def verify_host_resource_availability_manifest(path: Path) -> dict:
    """Operator-facing smoke verifier.

    Returns a small summary dict; raises ``HostResourceAvailabilityError``
    on schema failure.
    """
    manifest = load_host_resource_availability_manifest(path)
    return {
        "schema_version": manifest.schema_version,
        "host_class": manifest.host_class,
        "host_total_vram_gib": manifest.host_total_vram_gib,
        "unavailable_workload_count": len(manifest.unavailable_workloads),
        "unavailable_slug_count": len(manifest.unavailable_slugs),
        "typed_reason_histogram": {
            reason: sum(
                1
                for entry in manifest.unavailable_workloads
                if entry.typed_reason == reason
            )
            + sum(
                1
                for entry in manifest.unavailable_slugs
                if entry.typed_reason == reason
            )
            for reason in HOST_RESOURCE_UNAVAILABLE_TYPED_REASONS
        },
    }


if __name__ == "__main__":  # pragma: no cover
    import sys
    import json as _json

    if len(sys.argv) != 2:
        print(
            "usage: host_resource_availability.py <manifest.toml>",
            file=sys.stderr,
        )
        sys.exit(2)
    summary = verify_host_resource_availability_manifest(Path(sys.argv[1]))
    print(_json.dumps(summary, indent=2))
