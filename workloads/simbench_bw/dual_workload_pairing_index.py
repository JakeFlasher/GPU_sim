"""Combined dual-workload manifest-pairing index builder.

Implements the cross-workload identity-validation gate: a single
manifest-pairing index keyed by
``(slug, shape_point, binary_sha256, source_commit,
kernel_signature_sha256)`` that aggregates artifact identities
from BOTH ``workloads/simbench_bw`` and
``workloads/portable_lcd``.

Identity discipline (R57 + R58):

1. **Roster-backed (R57)**: both readers consume the
   workload rosters (`workloads/simbench_bw/roster.toml`,
   `workloads/portable_lcd/roster.toml`) via ``tomllib``.
   portable_lcd selects rows where ``status == "active"``;
   simbench rejects any artifact slug not present in the
   roster. **R58 fail-closed**: a missing or unparseable
   roster file raises ``MissingRosterError`` (R57 silently
   degraded to an empty-selection contract when the roster
   path was wrong, which defeated the point of being
   roster-backed).
2. **Schema-validating (R57)**: portable_lcd manifests are
   validated through
   ``workloads/simbench_bw/validate.py::
   validate_manifest_files`` (against
   ``measurement_manifest.schema.json``) BEFORE identity
   fields are read. Schema rejection raises
   ``SchemaInvalidManifestError``; invalid JSON or non-dict
   manifest raises ``MalformedManifestError``.
3. **Content-derived artifact-stable identity (R58
   replaces R57)**: portable_lcd
   ``kernel_signature_sha256`` is now
   ``sha256(slug | actual_binary_sha256)`` where
   ``actual_binary_sha256`` is the sha256 of the binary's
   bytes (read from
   ``<portable_lcd_root>/<slug>/<slug>_run``). The
   manifest's schema-valid ``binary_sha256`` field MUST
   equal the actual binary digest; mismatch raises
   ``BinaryDigestMismatchError``. Path is no longer in the
   hash input, so the 5-tuple is genuinely invariant under
   tree copy/move (R57's path-string hash broke under the
   simplest stability test).

The two workload directories produce slightly different
artifact shapes:

* **simbench_bw** writes per-shape run summaries
  ``results/<slug>/<sku>/<arch>/trace_<shape>.json`` carrying
  the full 5-tuple identity. R57 reads these directly when
  the slug is present in
  ``workloads/simbench_bw/roster.toml``.

* **portable_lcd** writes per-slug measurement manifests
  ``<slug>/results/<slug>/manifest.json`` validated against
  the shared ``measurement_manifest.schema.json``. The two
  identity fields the schema does NOT carry
  (``source_commit``, ``kernel_signature_sha256``) are
  derived deterministically:

    - ``source_commit``: best-effort read from
      ``<repo_root>/.git/HEAD``.
    - ``kernel_signature_sha256``:
      ``sha256(slug | actual_binary_sha256)`` where
      ``actual_binary_sha256`` is the content digest of
      the active row's ``<slug>/<slug>_run`` binary. The
      digest is cross-checked against the manifest's
      schema-valid ``binary_sha256`` field; mismatch
      raises ``BinaryDigestMismatchError``.

The combined index is a ``dict[PairingKey,
WorkloadIdentity]`` with one entry per artifact root. Key
collisions across workloads raise
``DualWorkloadKeyCollision``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import validate as _simbench_validate  # type: ignore[import-not-found]  # noqa: E402

# 5-tuple identity key shape used across the project's HRD,
# order_projections, sharing_locality, divergence_proxies, and
# perturbation_run_ledger sidecars.
PairingKey = tuple[str, str, str, str, str]


@dataclass(frozen=True)
class WorkloadIdentity:
    """One artifact-root identity in the combined index."""

    slug: str
    shape_point: str
    binary_sha256: str
    source_commit: str
    kernel_signature_sha256: str
    workload_registry: str  # "simbench_bw" or "portable_lcd"
    artifact_path: Path

    def key(self) -> PairingKey:
        return (
            self.slug,
            self.shape_point,
            self.binary_sha256,
            self.source_commit,
            self.kernel_signature_sha256,
        )


class DualWorkloadKeyCollision(Exception):
    """Two artifact roots produced the same 5-tuple key."""

    TOKEN = "dual_workload_pairing_key_collision"

    def __init__(self, key: PairingKey, first: Path, second: Path) -> None:
        super().__init__(
            f"{self.TOKEN}: key {key!r} appears at both "
            f"{first} and {second}; the combined "
            f"manifest-pairing index requires unique 5-tuples"
        )
        self.key = key
        self.first = first
        self.second = second


class MalformedManifestError(Exception):
    """A manifest is missing a required identity field, or
    cannot be parsed at all (invalid JSON, non-dict body)."""

    TOKEN = "dual_workload_pairing_manifest_malformed"

    def __init__(self, manifest_path: Path, missing_field: str) -> None:
        super().__init__(
            f"{self.TOKEN}: manifest {manifest_path}: "
            f"{missing_field}"
        )
        self.manifest_path = manifest_path
        self.missing_field = missing_field


class SchemaInvalidManifestError(Exception):
    """Manifest fails strict-schema validation against
    ``measurement_manifest.schema.json``."""

    TOKEN = "dual_workload_pairing_manifest_schema_invalid"

    def __init__(self, manifest_path: Path, errors: list[str]) -> None:
        super().__init__(
            f"{self.TOKEN}: manifest {manifest_path} fails "
            f"strict schema validation: {errors}"
        )
        self.manifest_path = manifest_path
        self.errors = errors


class MissingActiveManifestError(Exception):
    """An active portable_lcd roster row has no manifest on
    disk."""

    TOKEN = "dual_workload_pairing_active_manifest_missing"

    def __init__(self, slug: str, expected_path: Path) -> None:
        super().__init__(
            f"{self.TOKEN}: portable_lcd roster slug {slug!r} is "
            f"`status=\"active\"` but {expected_path} does not "
            f"exist; active rows MUST have a manifest"
        )
        self.slug = slug
        self.expected_path = expected_path


class MissingActiveBinaryError(Exception):
    """An active portable_lcd roster row has no built
    ``<slug>_run`` binary."""

    TOKEN = "dual_workload_pairing_active_binary_missing"

    def __init__(self, slug: str, expected_path: Path) -> None:
        super().__init__(
            f"{self.TOKEN}: portable_lcd roster slug {slug!r} is "
            f"`status=\"active\"` but {expected_path} does not "
            f"exist; the binary is required to derive an "
            f"artifact-stable kernel_signature_sha256"
        )
        self.slug = slug
        self.expected_path = expected_path


class SlugNotInRosterError(Exception):
    """A simbench artifact references a slug not present in
    `simbench_bw/roster.toml`."""

    TOKEN = "dual_workload_pairing_slug_not_in_roster"

    def __init__(self, slug: str, summary_path: Path) -> None:
        super().__init__(
            f"{self.TOKEN}: simbench summary {summary_path} "
            f"references slug {slug!r} but that slug is not in "
            f"`workloads/simbench_bw/roster.toml`"
        )
        self.slug = slug
        self.summary_path = summary_path


class MissingRosterError(Exception):
    """A required roster file is absent or unparseable.

    Distinct from a present-but-empty roster (``rows = []``),
    which is admitted as an empty selection (intentional
    contract — a roster with all-pending rows yields no
    active gate). R57's silent degradation (return
    ``set()`` / ``[]`` on a missing file) defeated the
    "roster-backed" claim because a typo in the caller's
    roster path admitted arbitrary summaries.
    """

    TOKEN = "dual_workload_pairing_roster_missing_or_malformed"

    def __init__(self, roster_path: Path, reason: str) -> None:
        super().__init__(
            f"{self.TOKEN}: roster {roster_path}: {reason}"
        )
        self.roster_path = roster_path
        self.reason = reason


class BinaryDigestMismatchError(Exception):
    """A portable_lcd active row's manifest ``binary_sha256``
    field disagrees with the sha256 of the binary's actual
    bytes.

    The schema-valid ``binary_sha256`` is the producer's
    declared digest at manifest-write time; the actual
    binary content's sha256 is what the gate computes when
    deriving the artifact-stable
    ``kernel_signature_sha256``. The two MUST agree —
    disagreement means the manifest is stale relative to
    the binary (operator rebuilt without refreshing the
    manifest, or vice versa).
    """

    TOKEN = "dual_workload_pairing_binary_digest_mismatch"

    def __init__(
        self,
        slug: str,
        binary_path: Path,
        manifest_digest: str,
        actual_digest: str,
    ) -> None:
        super().__init__(
            f"{self.TOKEN}: portable_lcd slug {slug!r} binary "
            f"{binary_path}: manifest binary_sha256={manifest_digest!r} "
            f"!= sha256(binary bytes)={actual_digest!r}"
        )
        self.slug = slug
        self.binary_path = binary_path
        self.manifest_digest = manifest_digest
        self.actual_digest = actual_digest


def _resolve_source_commit(repo_root: Path) -> str:
    """Best-effort read of `.git/HEAD` resolving to the underlying
    commit SHA. Mirrors `run_suite.py::_resolve_source_commit`."""
    try:
        head_path = repo_root / ".git" / "HEAD"
        head = head_path.read_text().strip()
        if head.startswith("ref: "):
            ref_path = head[len("ref: "):]
            ref_file = repo_root / ".git" / ref_path
            return ref_file.read_text().strip()
        return head
    except OSError:
        return "0" * 64


def _portable_lcd_kernel_signature_sha256(
    slug: str, binary_content_sha256: str,
) -> str:
    """Content-derived kernel-signature for portable_lcd
    entries. Returns
    ``sha256(slug | actual_binary_sha256)`` where the second
    argument is the sha256 of the binary's actual bytes. The
    output is invariant under tree copy/move (the binary's
    content is stable, the path is not), so the combined
    pairing index's 5-tuple key is genuinely artifact-stable.

    R58 fix (Codex's R57 review): R57 hashed
    ``slug | binary_path`` where ``binary_path`` includes
    the parent directory the caller passes in. Copying the
    same artifact tree from ``portable_lcd_a`` to
    ``portable_lcd_b`` changed the fifth key component, so
    the combined index's 5-tuple was not actually stable.
    The R57 stability test punted on this assertion in a
    docstring; R58 asserts ``ids_a[0].key() ==
    ids_b[0].key()``.
    """
    return hashlib.sha256(
        f"{slug}|{binary_content_sha256}".encode("utf-8")
    ).hexdigest()


def _load_simbench_roster(
    simbench_roster_path: Path,
) -> set[str]:
    """Load `workloads/simbench_bw/roster.toml` and return the
    slug set.

    R58: a missing/unparseable roster file raises
    ``MissingRosterError``. A present-but-empty roster
    (``rows = []``) returns ``set()`` cleanly — that's an
    intentional contract (mirrors a roster with all rows in
    a non-active state).
    """
    if not simbench_roster_path.is_file():
        raise MissingRosterError(
            simbench_roster_path, "file_not_found",
        )
    try:
        with simbench_roster_path.open("rb") as f:
            roster = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise MissingRosterError(
            simbench_roster_path, f"toml_decode_error:{exc}",
        )
    rows = roster.get("rows") or []
    return {
        row["slug"] for row in rows
        if isinstance(row, dict) and isinstance(row.get("slug"), str)
    }


def _load_portable_lcd_active_slugs(
    portable_lcd_roster_path: Path,
) -> list[str]:
    """Load `workloads/portable_lcd/roster.toml` and return the
    sorted slug list for rows whose ``status == "active"``.

    R58: a missing/unparseable roster file raises
    ``MissingRosterError``. A present-but-empty roster (or
    one with no active rows) returns an empty list
    cleanly — that's an intentional contract.
    """
    if not portable_lcd_roster_path.is_file():
        raise MissingRosterError(
            portable_lcd_roster_path, "file_not_found",
        )
    try:
        with portable_lcd_roster_path.open("rb") as f:
            roster = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise MissingRosterError(
            portable_lcd_roster_path, f"toml_decode_error:{exc}",
        )
    rows = roster.get("rows") or []
    return sorted(
        row["slug"] for row in rows
        if isinstance(row, dict)
        and row.get("status") == "active"
        and isinstance(row.get("slug"), str)
    )


def collect_simbench_identities(
    simbench_results_root: Path,
    *,
    simbench_roster_path: Path,
) -> list[WorkloadIdentity]:
    """Walk simbench's canonical results tree and return one
    `WorkloadIdentity` per per-shape run summary
    (`trace_<shape>.json`) whose slug appears in the simbench
    roster.

    A summary referencing a slug NOT in the roster raises
    ``SlugNotInRosterError`` — the canonical capture pipeline
    only writes summaries for roster-listed slugs, so a
    missing-roster slug indicates contract drift (or stray
    manifests from a removed slug).

    A summary that does not carry the full 5-tuple identity
    (e.g., a typed-unavailable summary that fired before any
    sidecar emission) is silently skipped — those summaries
    are admitted by the per-shape verifier via the
    typed-skip-reason path.
    """
    # R59 (Codex's R58 review): load the roster BEFORE the
    # results-root short-circuit so a missing/typo'd roster
    # path always raises ``MissingRosterError``, regardless of
    # whether the results tree happens to be empty or absent.
    # R58 returned ``[]`` on a missing results root and never
    # reached the roster load, which let
    # ``build_combined_pairing_index`` silently admit ``{}``
    # under a misconfigured simbench roster path.
    roster_slugs = _load_simbench_roster(simbench_roster_path)
    out: list[WorkloadIdentity] = []
    if not simbench_results_root.is_dir():
        return out
    for summary_path in sorted(
        simbench_results_root.rglob("trace_*.json")
    ):
        if not summary_path.is_file():
            continue
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except json.JSONDecodeError as exc:
            raise MalformedManifestError(
                summary_path, f"invalid_json:{exc}",
            )
        if not isinstance(summary, dict):
            raise MalformedManifestError(
                summary_path, "non_object_json",
            )
        slug = summary.get("slug")
        if isinstance(slug, str) and slug not in roster_slugs:
            raise SlugNotInRosterError(slug, summary_path)
        ident_fields: dict[str, Any] = {
            "slug": summary.get("slug"),
            "shape_point": summary.get("shape_point"),
            "binary_sha256": summary.get("binary_sha256"),
            "source_commit": summary.get("source_commit"),
            "kernel_signature_sha256": summary.get(
                "kernel_signature_sha256"
            ),
        }
        if not all(
            isinstance(v, str) and v for v in ident_fields.values()
        ):
            # Typed-unavailable summary (incomplete identity
            # tuple). Per-shape verifier admits these via the
            # typed-skip-reason path; skip without raising.
            continue
        out.append(WorkloadIdentity(
            slug=str(ident_fields["slug"]),
            shape_point=str(ident_fields["shape_point"]),
            binary_sha256=str(ident_fields["binary_sha256"]),
            source_commit=str(ident_fields["source_commit"]),
            kernel_signature_sha256=str(
                ident_fields["kernel_signature_sha256"]
            ),
            workload_registry="simbench_bw",
            artifact_path=summary_path,
        ))
    return out


def collect_portable_lcd_identities(
    portable_lcd_root: Path,
    *,
    repo_root: Path,
    portable_lcd_roster_path: Path,
) -> list[WorkloadIdentity]:
    """Walk portable_lcd's per-slug results tree, driven by
    the roster's ``status == "active"`` rows. Each active row
    MUST have:

      * ``<portable_lcd_root>/<slug>/results/<slug>/manifest.json``
        that schema-validates against
        ``measurement_manifest.schema.json``.
      * ``<portable_lcd_root>/<slug>/<slug>_run`` binary
        (used to derive an artifact-stable
        ``kernel_signature_sha256``).

    Raises typed exceptions for missing manifest, missing
    binary, schema-invalid manifest, invalid JSON, or
    missing identity field. Stray (non-roster) manifests in
    the filesystem are NOT enumerated.
    """
    out: list[WorkloadIdentity] = []
    active_slugs = _load_portable_lcd_active_slugs(
        portable_lcd_roster_path,
    )
    if not active_slugs:
        return out
    source_commit = _resolve_source_commit(repo_root)
    for slug in active_slugs:
        slug_dir = portable_lcd_root / slug
        manifest_path = slug_dir / "results" / slug / "manifest.json"
        binary_path = slug_dir / f"{slug}_run"
        if not manifest_path.is_file():
            raise MissingActiveManifestError(slug, manifest_path)
        if not binary_path.is_file():
            raise MissingActiveBinaryError(slug, binary_path)
        # Schema validation BEFORE reading identity fields.
        # `validate_manifest_files` returns the list of errors
        # (empty list = admitted), but raises if the manifest
        # is unparseable JSON. Wrap the JSON-parse error as a
        # MalformedManifestError so the caller gets a typed
        # diagnostic rather than a raw JSONDecodeError.
        try:
            schema_errors = (
                _simbench_validate.validate_manifest_files(
                    [manifest_path]
                )
            )
        except json.JSONDecodeError as exc:
            raise MalformedManifestError(
                manifest_path, f"invalid_json:{exc}",
            )
        if schema_errors:
            raise SchemaInvalidManifestError(manifest_path, schema_errors)
        # Re-read after schema validation (single I/O cost; the
        # schema validator's _load_json is internal).
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except json.JSONDecodeError as exc:
            raise MalformedManifestError(
                manifest_path, f"invalid_json:{exc}",
            )
        if not isinstance(manifest, dict):
            raise MalformedManifestError(
                manifest_path, "non_object_json",
            )
        # Identity fields. Schema validation guarantees
        # binary_sha256 + slug + shape_point are present and of
        # the right type, but defensively re-check.
        for required in ("slug", "shape_point", "binary_sha256"):
            v = manifest.get(required)
            if not isinstance(v, str) or not v:
                raise MalformedManifestError(manifest_path, required)
        # R59 (Codex's R58 review): the active roster row's
        # ``slug`` MUST match the manifest body's ``slug``
        # field. Without this check, an active row for
        # ``roster_slug`` could admit a manifest keyed as
        # ``different_slug``, producing an incoherent identity
        # whose tuple component 1 (slug) comes from the
        # manifest body but whose tuple component 5
        # (kernel_signature_sha256) was derived from the
        # roster slug.
        if manifest["slug"] != slug:
            raise MalformedManifestError(
                manifest_path,
                f"slug_mismatch: roster={slug} "
                f"manifest={manifest['slug']}",
            )
        # R58: hash the binary's actual bytes and verify the
        # digest matches the schema-valid manifest field. The
        # actual digest feeds the content-derived
        # kernel_signature_sha256, so the 5-tuple key is
        # invariant under tree copy/move.
        actual_binary_sha256 = hashlib.sha256(
            binary_path.read_bytes()
        ).hexdigest()
        manifest_binary_sha256 = manifest["binary_sha256"]
        if actual_binary_sha256 != manifest_binary_sha256:
            raise BinaryDigestMismatchError(
                slug,
                binary_path,
                manifest_binary_sha256,
                actual_binary_sha256,
            )
        kernel_signature_sha256 = (
            _portable_lcd_kernel_signature_sha256(
                slug, actual_binary_sha256,
            )
        )
        out.append(WorkloadIdentity(
            slug=manifest["slug"],
            shape_point=manifest["shape_point"],
            binary_sha256=manifest_binary_sha256,
            source_commit=source_commit,
            kernel_signature_sha256=kernel_signature_sha256,
            workload_registry="portable_lcd",
            artifact_path=manifest_path,
        ))
    return out


def build_combined_pairing_index(
    *,
    simbench_results_root: Path,
    portable_lcd_root: Path,
    repo_root: Path,
    simbench_roster_path: Path,
    portable_lcd_roster_path: Path,
) -> dict[PairingKey, WorkloadIdentity]:
    """Build the combined manifest-pairing index from both
    workload registries, gated by their rosters. Walks each
    workload's active row set, schema-validates portable_lcd
    manifests, derives an artifact-stable 5-tuple, and rejects
    key collisions.

    Returns ``dict[PairingKey, WorkloadIdentity]`` with one
    entry per unique artifact root.
    """
    index: dict[PairingKey, WorkloadIdentity] = {}
    for source in (
        collect_simbench_identities(
            simbench_results_root,
            simbench_roster_path=simbench_roster_path,
        ),
        collect_portable_lcd_identities(
            portable_lcd_root,
            repo_root=repo_root,
            portable_lcd_roster_path=portable_lcd_roster_path,
        ),
    ):
        for ident in source:
            key = ident.key()
            existing = index.get(key)
            if existing is not None:
                raise DualWorkloadKeyCollision(
                    key, existing.artifact_path, ident.artifact_path,
                )
            index[key] = ident
    return index


__all__ = (
    "PairingKey",
    "WorkloadIdentity",
    "DualWorkloadKeyCollision",
    "MalformedManifestError",
    "SchemaInvalidManifestError",
    "MissingActiveManifestError",
    "MissingActiveBinaryError",
    "SlugNotInRosterError",
    "MissingRosterError",
    "BinaryDigestMismatchError",
    "collect_simbench_identities",
    "collect_portable_lcd_identities",
    "build_combined_pairing_index",
)
