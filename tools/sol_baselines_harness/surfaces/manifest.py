"""Manifest builder + lint for the per-bundle ``manifest.json``.

Each ``.harness_traces/<slug>__<lang>/manifest.json`` file carries:

* **Host fingerprint** — ``kind``, ``device_name``, ``compute_capability``,
  ``driver_version``, ``cuda_runtime_version``, ``captured_at_utc``,
  ``nvtx_header_present``, ``nvtx_library_present``.
* **Reproducibility fields** — ``evaluator_tree_hash``, ``native_env_hash``,
  ``native_env_components``, ``build_mode``, ``surface_set``,
  ``workload_subset``, ``workload_selection_policy``, ``config_overrides``,
  ``eval_timeout_s``, ``reference_timing_enabled``, ``dispatch_command``.
* **Per-surface entries** (one slot per surface in ``VALID_SURFACES``)
  — each entry carries ``status``, ``tool_version``, ``schema_version``,
  ``impl_version``, ``command``, ``exit_code``, ``stdout_tail``,
  ``stderr_tail``, ``artifact_path``, ``waiver_reason``, ``start_utc``,
  ``end_utc``, ``size_bytes``, ``compressed_size_bytes``, ``substatus``.
* **Surface invocation order** (``surface_invocation_order: list[str]``)
  — the runtime order surfaces actually ran in; NVBit and nsys MUST
  appear in separate dispatch groups (collision-safety negative test).

Manifest lint invariants:

1. Every surface entry's ``status`` MUST be one of
   ``{ok, unavailable, blocked, failed, skipped}``.
2. ``status: ok`` MUST be paired with a non-empty ``artifact_path``.
3. ``status ∈ {unavailable, blocked, skipped}`` MUST be paired with a
   non-empty ``waiver_reason``.
4. ``status: failed`` MUST carry ``command``, ``exit_code``, AND a
   non-empty ``stderr_tail`` OR ``waiver_reason``.
5. The 13 top-level reproducibility fields MUST be present (any can
   be ``null`` when not applicable, but the slot MUST exist so the
   schema is self-describing).
6. ``workload_subset`` MUST be a sorted list of unique non-negative
   integers; lint pre-write checks bounds against the actual workload
   set when ``workload_subset_max`` is supplied.
7. The v2 schema rejects any of the dropped fields
   (``docker_image_digest``, ``profiling_image_digest``,
   ``dispatch_mode``) and the rejected ``dispatch_kind`` positive-marker
   if present in the manifest dict.
"""

from __future__ import annotations

import datetime
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from sol_baselines_harness.surfaces.base import (
    HostInfo,
    SURFACE_STATUSES,
    SURFACE_STATUS_FAILED,
    SURFACE_STATUS_OK,
    VALID_SURFACES,
)

MANIFEST_SCHEMA_VERSION = 2


def _is_digest_text(value: Any, *, lengths: tuple[int, ...] = (64,)) -> bool:
    """Return whether ``value`` is lowercase hexadecimal digest text."""
    return (
        isinstance(value, str)
        and len(value) in lengths
        and all(character in "0123456789abcdef" for character in value)
    )


class ManifestLintError(ValueError):
    """Raised when ``lint_manifest`` rejects a manifest pre-write."""


_REQUIRED_TOP_LEVEL_FIELDS = (
    "evaluator_tree_hash",
    "native_env_hash",
    "native_env_components",
    "build_mode",
    "surface_set",
    "workload_subset",
    "workload_selection_policy",
    "config_overrides",
    "eval_timeout_s",
    "reference_timing_enabled",
    "dispatch_command",
    "nvtx_header_present",
    "nvtx_library_present",
)

_FORBIDDEN_TOP_LEVEL_FIELDS = (
    "docker_image_digest",
    "profiling_image_digest",
    "dispatch_mode",
    "dispatch_kind",
)


@dataclass
class SurfaceManifestEntry:
    """Per-surface entry in ``manifest.surfaces[<name>]``.

    Mirrors the union of ``SurfaceProbeResult`` / ``SurfaceRunResult``
    / ``SurfaceParseResult`` plus the cache-identity fields the
    orchestrator carries.
    """

    status: str = ""
    tool_version: Optional[str] = None
    schema_version: Optional[str] = None
    impl_version: Optional[str] = None
    command: List[str] = field(default_factory=list)
    exit_code: int = 0
    stdout_tail: str = ""
    stderr_tail: str = ""
    artifact_path: Optional[str] = None
    waiver_reason: Optional[str] = None
    start_utc: str = ""
    end_utc: str = ""
    size_bytes: int = 0
    compressed_size_bytes: int = 0
    substatus: Dict[str, Any] = field(default_factory=dict)
    cache_identity: Dict[str, Any] = field(default_factory=dict)


class ManifestBuilder:
    """Accumulates per-surface entries + top-level reproducibility fields.

    The orchestrator instantiates one ``ManifestBuilder`` per
    ``(slug, language)`` bundle and feeds it surface results as they
    land. ``lint_and_write(path)`` runs the manifest lint invariants and
    writes the manifest JSON; a lint failure raises
    ``ManifestLintError`` BEFORE any write so a malformed manifest
    cannot land on disk.
    """

    def __init__(
        self,
        *,
        slug: str,
        language: str,
        host_info: HostInfo,
        build_mode: str,
        surface_set: Set[str],
    ) -> None:
        self.slug = slug
        self.language = language
        self.host_info = host_info
        self.build_mode = build_mode
        self.surface_set = set(surface_set)
        self.surfaces: Dict[str, SurfaceManifestEntry] = {
            name: SurfaceManifestEntry() for name in VALID_SURFACES
        }
        self.surface_invocation_order: List[str] = []
        self.evaluator_tree_hash: Optional[str] = None
        self.native_env_hash: Optional[str] = None
        self.native_env_components: Dict[str, str] = {}
        self.workload_subset: List[int] = []
        self.workload_selection_policy: str = "lowest-index-default"
        self.config_overrides: Dict[str, Any] = {}
        self.eval_timeout_s: Optional[float] = None
        self.reference_timing_enabled: Optional[bool] = None
        self.dispatch_command: Dict[str, Any] = {"argv": [], "env_overrides": {}}

    def record_surface(
        self,
        name: str,
        entry: SurfaceManifestEntry,
    ) -> None:
        """Stamp one surface's manifest entry.

        Records the invocation order so the orchestrator's NVBit / nsys
        mutual-exclusion can be audited from the manifest alone.
        """
        if name not in VALID_SURFACES:
            raise ValueError(
                f"unknown surface: {name!r}; "
                f"valid: {sorted(VALID_SURFACES)}"
            )
        self.surfaces[name] = entry
        if name not in self.surface_invocation_order:
            self.surface_invocation_order.append(name)

    def evidence_label(self) -> str:
        """Compute the bundle's evidence label.

        * ``Profiled-<host>-complete`` when every surface is ``ok``
          AND the orchestrator-side ``_all_passed_trace`` bit is True.
        * ``Profiled-<host>-<arch_tag>`` when no surface is ``failed``
          AND ``_all_passed_trace`` is True.
        * ``Ran-local-<arch_tag>`` (correctness-only) when
          ``_all_passed_trace`` is True but the framework hasn't
          reached the no-failed-surface bar.
        * ``Compile-checked-<arch_tag>`` fallback when no PASS
          evidence exists.

        ``_all_passed_trace`` is set by the orchestrator immediately
        before ``lint_and_write``; it defaults to ``False`` when the
        builder is constructed in isolation (e.g., unit tests), in
        which case the label stays at ``Compile-checked-<arch_tag>``.

        The arch tag is resolved dynamically from ``host_info.kind``
        via ``arch_resolver.arch_tag_from_host_kind`` — this is the
        single source of truth for the architecture half of every
        evidence label and report row.
        """
        all_passed_trace = getattr(self, "_all_passed_trace", False)
        statuses = {n: e.status for n, e in self.surfaces.items()}
        arch_tag = _host_arch_token(self.host_info.kind)
        if not all_passed_trace:
            return f"Compile-checked-{arch_tag}"
        if any(s == SURFACE_STATUS_FAILED for s in statuses.values()):
            return f"Ran-local-{arch_tag}"
        host_segment = _host_segment(self.host_info.kind)
        if all(s == SURFACE_STATUS_OK for s in statuses.values()):
            return f"Profiled-{host_segment}-complete"
        return f"Profiled-{host_segment}-{arch_tag}"

    def as_dict(self) -> Dict[str, Any]:
        """Serialize the manifest into the on-disk dict shape."""
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "slug": self.slug,
            "language": self.language,
            "host": {
                "kind": self.host_info.kind,
                "device_name": self.host_info.device_name,
                "compute_capability": self.host_info.compute_capability,
                "driver_version": self.host_info.driver_version,
                "cuda_runtime_version": self.host_info.cuda_runtime_version,
                "captured_at_utc": self.host_info.captured_at_utc,
            },
            "arch_tag": _host_arch_token(self.host_info.kind),
            "evaluator_tree_hash": self.evaluator_tree_hash,
            "native_env_hash": self.native_env_hash,
            "native_env_components": dict(self.native_env_components),
            "build_mode": self.build_mode,
            "surface_set": sorted(self.surface_set),
            "workload_subset": list(self.workload_subset),
            "workload_selection_policy": self.workload_selection_policy,
            "config_overrides": dict(self.config_overrides),
            "eval_timeout_s": self.eval_timeout_s,
            "reference_timing_enabled": self.reference_timing_enabled,
            "dispatch_command": dict(self.dispatch_command),
            "nvtx_header_present": self.host_info.nvtx_header_present,
            "nvtx_library_present": self.host_info.nvtx_library_present,
            "surfaces": {
                name: asdict(entry) for name, entry in self.surfaces.items()
            },
            "surface_invocation_order": list(self.surface_invocation_order),
            "evidence_label": self.evidence_label(),
        }

    def lint_and_write(
        self,
        path: Path,
        *,
        workload_subset_max: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Validate then atomically write ``manifest.json``.

        Raises ``ManifestLintError`` BEFORE any write when the
        manifest violates an invariant. The caller is expected
        to surface the lint error and abort rather than fall back to
        a partial write.
        """
        manifest = self.as_dict()
        lint_manifest(manifest, workload_subset_max=workload_subset_max)
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        temporary_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = temporary.name
                temporary.write(serialized)
                temporary.flush()
            os.replace(temporary_path, path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass
        return manifest


def _host_segment(kind: str) -> str:
    """Translate a ``host.kind`` into the evidence-label host segment."""
    if kind.startswith("b200"):
        return "B200"
    return "local"


def _host_arch_token(kind: str) -> str:
    """Translate a ``host.kind`` into the evidence-label arch token.

    Delegates to ``arch_resolver.arch_tag_from_host_kind`` so the
    manifest / report / scoring layers share the same arch-tag
    vocabulary. Covers (at minimum) ``sm_86`` (3090 Ti), ``sm_90``
    (H100), ``sm_100a`` (B200), ``sm_120`` (5060), and the
    generic ``local_sm_X`` / ``local_sm_X_Y`` fallback for any
    other compute capability the host probe classifies; unknown
    inputs return ``sm_unknown``.
    """
    from sol_baselines_harness.arch_resolver import arch_tag_from_host_kind
    return arch_tag_from_host_kind(kind)


def lint_manifest(
    manifest: Dict[str, Any],
    *,
    workload_subset_max: Optional[int] = None,
) -> None:
    """Apply the manifest lint invariants.

    Raises ``ManifestLintError`` with an actionable message at the
    first invariant violation. Callers ``lint`` then ``write``; lint
    runs BEFORE write so a partial / malformed manifest cannot land
    on disk.
    """
    schema_version = manifest.get("schema_version")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ManifestLintError(
            f"manifest.schema_version must be {MANIFEST_SCHEMA_VERSION}; "
            f"got {schema_version!r}"
        )
    for forbidden in _FORBIDDEN_TOP_LEVEL_FIELDS:
        if forbidden in manifest:
            raise ManifestLintError(
                f"manifest carries forbidden field {forbidden!r}; the v2 "
                f"schema rejects this key (dispatch identity is conveyed "
                f"by native_env_hash presence)."
            )
    host = manifest.get("host")
    if not isinstance(host, dict):
        raise ManifestLintError(
            f"manifest.host must be a dict; got {type(host).__name__}"
        )
    host_kind = host.get("kind")
    if not isinstance(host_kind, str) or not host_kind:
        raise ManifestLintError(
            f"manifest.host.kind must be a non-empty string; got {host_kind!r}"
        )
    for field_name in (
        "device_name",
        "compute_capability",
        "driver_version",
        "cuda_runtime_version",
        "captured_at_utc",
    ):
        value = host.get(field_name)
        if not isinstance(value, str) or not value:
            raise ManifestLintError(
                f"manifest.host.{field_name} must be a non-empty string; "
                f"got {value!r}"
            )
    slug = manifest.get("slug")
    if (
        not isinstance(slug, str)
        or not slug
        or "/" in slug
        or "\\" in slug
    ):
        raise ManifestLintError(
            f"manifest.slug must be a non-empty path-safe string; got {slug!r}"
        )
    if manifest.get("language") not in ("cuda_cpp", "cutlass"):
        raise ManifestLintError(
            "manifest.language must be 'cuda_cpp' or 'cutlass'; got "
            f"{manifest.get('language')!r}"
        )
    if "arch_tag" not in manifest:
        raise ManifestLintError(
            "manifest is missing top-level field 'arch_tag'; the v2 "
            "schema requires it to be derived from host.kind via "
            "arch_tag_from_host_kind so report.md / scoring rows have a "
            "single source of truth for the architecture token."
        )
    arch_tag_value = manifest["arch_tag"]
    expected_arch_tag = _host_arch_token(host_kind)
    if arch_tag_value != expected_arch_tag:
        raise ManifestLintError(
            f"manifest.arch_tag = {arch_tag_value!r} does not match "
            f"arch_tag_from_host_kind({host_kind!r}) = "
            f"{expected_arch_tag!r}; the manifest's host-identity tokens "
            f"are inconsistent."
        )
    for field_name in _REQUIRED_TOP_LEVEL_FIELDS:
        if field_name not in manifest:
            raise ManifestLintError(
                f"manifest is missing top-level field {field_name!r}; "
                f"all 13 reproducibility fields must be present "
                f"(set to null when not applicable)."
            )
    native_env_components_value = manifest.get("native_env_components")
    if not isinstance(native_env_components_value, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in native_env_components_value.items()
    ):
        raise ManifestLintError(
            "native_env_components must be a dict of string keys and values; "
            f"got {native_env_components_value!r}"
        )
    surface_set = manifest.get("surface_set")
    if (
        not isinstance(surface_set, list)
        or any(not isinstance(name, str) for name in surface_set)
        or surface_set != sorted(surface_set)
        or len(set(surface_set)) != len(surface_set)
        or not set(surface_set).issubset(VALID_SURFACES)
    ):
        raise ManifestLintError(
            "surface_set must be a sorted list of unique names from "
            f"{sorted(VALID_SURFACES)}; got {surface_set!r}"
        )
    if manifest.get("build_mode") not in ("judge", "profiling"):
        raise ManifestLintError(
            "build_mode must be 'judge' or 'profiling'; got "
            f"{manifest.get('build_mode')!r}"
        )
    if manifest.get("workload_selection_policy") not in (
        "lowest-index-default",
        "full-fanout",
        "explicit",
    ):
        raise ManifestLintError(
            "workload_selection_policy has an unsupported value: "
            f"{manifest.get('workload_selection_policy')!r}"
        )
    for field_name in ("evaluator_tree_hash", "native_env_hash"):
        value = manifest.get(field_name)
        digest_lengths = (40, 64) if field_name == "evaluator_tree_hash" else (64,)
        if value is not None and not _is_digest_text(value, lengths=digest_lengths):
            raise ManifestLintError(
                f"{field_name} must be null or lowercase hexadecimal digest "
                f"text with length {digest_lengths}; got "
                f"{value!r}"
            )
    config_overrides = manifest.get("config_overrides")
    if not isinstance(config_overrides, dict):
        raise ManifestLintError(
            f"config_overrides must be a dict; got "
            f"{type(config_overrides).__name__}"
        )
    eval_timeout = manifest.get("eval_timeout_s")
    if eval_timeout is not None and (
        isinstance(eval_timeout, bool)
        or not isinstance(eval_timeout, (int, float))
        or eval_timeout < 0
    ):
        raise ManifestLintError(
            f"eval_timeout_s must be null or a non-negative number; got "
            f"{eval_timeout!r}"
        )
    reference_timing = manifest.get("reference_timing_enabled")
    if reference_timing is not None and not isinstance(reference_timing, bool):
        raise ManifestLintError(
            "reference_timing_enabled must be null or bool; got "
            f"{reference_timing!r}"
        )
    dispatch_command = manifest.get("dispatch_command")
    if not isinstance(dispatch_command, dict):
        raise ManifestLintError(
            f"dispatch_command must be a dict; got "
            f"{type(dispatch_command).__name__}"
        )
    if "argv" in dispatch_command and (
        not isinstance(dispatch_command["argv"], list)
        or any(not isinstance(arg, str) for arg in dispatch_command["argv"])
    ):
        raise ManifestLintError("dispatch_command.argv must be a list of strings")
    if "env_overrides" in dispatch_command and not isinstance(
        dispatch_command["env_overrides"], dict
    ):
        raise ManifestLintError("dispatch_command.env_overrides must be a dict")
    for field_name in ("nvtx_header_present", "nvtx_library_present"):
        if not isinstance(manifest.get(field_name), bool):
            raise ManifestLintError(
                f"{field_name} must be bool; got {manifest.get(field_name)!r}"
            )
    invocation_order = manifest.get("surface_invocation_order")
    if (
        not isinstance(invocation_order, list)
        or any(not isinstance(name, str) for name in invocation_order)
        or len(set(invocation_order)) != len(invocation_order)
        or not set(invocation_order).issubset(VALID_SURFACES)
    ):
        raise ManifestLintError(
            "surface_invocation_order must be a list of unique surface names"
        )
    ws = manifest.get("workload_subset")
    if not isinstance(ws, list):
        raise ManifestLintError(
            f"workload_subset must be a list of integers; got {type(ws).__name__}"
        )
    seen: Set[int] = set()
    for idx in ws:
        if type(idx) is not int or idx < 0:
            raise ManifestLintError(
                f"workload_subset entries must be non-negative integers; "
                f"got {idx!r}"
            )
        if idx in seen:
            raise ManifestLintError(
                f"workload_subset entries must be unique; duplicate {idx}"
            )
        seen.add(idx)
        if workload_subset_max is not None and idx >= workload_subset_max:
            raise ManifestLintError(
                f"workload_subset entry {idx} is out of bounds for a "
                f"problem with {workload_subset_max} workloads."
            )
    if list(sorted(ws)) != list(ws):
        raise ManifestLintError(
            "workload_subset must be sorted ascending"
        )
    surfaces = manifest.get("surfaces", {})
    if not isinstance(surfaces, dict):
        raise ManifestLintError("manifest.surfaces must be a dict")
    expected_names = set(VALID_SURFACES)
    for name in expected_names:
        if name not in surfaces:
            raise ManifestLintError(
                f"manifest.surfaces is missing entry for {name!r}"
            )
    for name, entry in surfaces.items():
        if not isinstance(entry, dict):
            raise ManifestLintError(
                f"manifest.surfaces[{name!r}] must be a dict"
            )
        for field_name in (
            "tool_version",
            "schema_version",
            "impl_version",
            "artifact_path",
            "waiver_reason",
            "start_utc",
            "end_utc",
            "stdout_tail",
            "stderr_tail",
        ):
            value = entry.get(field_name)
            if value is not None and not isinstance(value, str):
                raise ManifestLintError(
                    f"manifest.surfaces[{name!r}].{field_name} must be "
                    f"a string or null; got {value!r}"
                )
        command = entry.get("command")
        if not isinstance(command, list) or any(
            not isinstance(arg, str) for arg in command
        ):
            raise ManifestLintError(
                f"manifest.surfaces[{name!r}].command must be a list of strings"
            )
        exit_code = entry.get("exit_code")
        if type(exit_code) is not int:
            raise ManifestLintError(
                f"manifest.surfaces[{name!r}].exit_code must be an int; "
                f"got {exit_code!r}"
            )
        for field_name in ("size_bytes", "compressed_size_bytes"):
            value = entry.get(field_name)
            if type(value) is not int or value < 0:
                raise ManifestLintError(
                    f"manifest.surfaces[{name!r}].{field_name} must be a "
                    f"non-negative int; got {value!r}"
                )
        for field_name in ("substatus", "cache_identity"):
            if not isinstance(entry.get(field_name), dict):
                raise ManifestLintError(
                    f"manifest.surfaces[{name!r}].{field_name} must be a dict"
                )
        status = entry.get("status")
        if status not in SURFACE_STATUSES:
            raise ManifestLintError(
                f"manifest.surfaces[{name!r}].status = {status!r} is "
                f"not in the locked vocabulary "
                f"{sorted(SURFACE_STATUSES)}"
            )
        if status == SURFACE_STATUS_OK:
            artifact_path = entry.get("artifact_path")
            if not isinstance(artifact_path, str) or not artifact_path:
                raise ManifestLintError(
                    f"manifest.surfaces[{name!r}].status=ok requires a "
                    f"non-empty artifact_path"
                )
        elif status in ("unavailable", "blocked", "skipped"):
            waiver = entry.get("waiver_reason")
            if not isinstance(waiver, str) or not waiver:
                raise ManifestLintError(
                    f"manifest.surfaces[{name!r}].status={status} requires "
                    f"a non-empty waiver_reason"
                )
        elif status == SURFACE_STATUS_FAILED:
            command = entry.get("command")
            if not command:
                raise ManifestLintError(
                    f"manifest.surfaces[{name!r}].status=failed requires "
                    f"a non-empty command list"
                )
            exit_code = entry.get("exit_code")
            if not (
                isinstance(entry.get("stderr_tail"), str)
                and entry.get("stderr_tail")
            ) and not (
                isinstance(entry.get("waiver_reason"), str)
                and entry.get("waiver_reason")
            ):
                raise ManifestLintError(
                    f"manifest.surfaces[{name!r}].status=failed requires "
                    f"stderr_tail or waiver_reason to be non-empty"
                )


def now_utc_iso() -> str:
    """Convenience: current UTC time as an ISO-8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
