"""`cache_prediction` per-launch sidecar artifact contract.

The `cache_prediction` surface is registered against
the locked vocabulary AND emits a separate sidecar
artifact carrying the per-launch prediction payload.
This module defines the deployed-scope writer / reader
boundary for that sidecar: one JSONL row per
`PredictorBundle`, under
`<bundle_dir>/cache_prediction/predictions.jsonl`.

The writer validates every bundle through
`validate_predictor_bundle` BEFORE serialising, so an
invalid bundle cannot be persisted to disk. The reader
parses each row back into a `PredictorBundle`,
validates it through the same composite validator at
the read boundary, and refuses to return on any
schema failure. Together these two boundaries are
the acceptance gate the cache-prediction surface
entry consumes: a manifest entry cannot promote to
`status="ok"` without a sidecar that exists,
validates clean, and round-trips its
`manifest_pairing_index` against the surface entry's
own pairing index.

Imports only stdlib +
`baselines_harness.metrics.framework_contract` +
`baselines_harness.metrics.predictor_bundle_schema` +
`baselines_harness.surfaces.cache_prediction`. NO
calibration-subtree imports; the NCU quarantine
boundary is preserved.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from baselines_harness.metrics.framework_contract import (
    InputSidecarDigest,
)
from baselines_harness.metrics.predictor_bundle_schema import (
    ManifestPairingIndex,
    PredictorBundle,
    PredictorBundleSchemaError,
    validate_predictor_bundle,
)
from baselines_harness.surfaces.cache_prediction import (
    CachePredictionSurfaceEntry,
    _validate_cache_prediction_surface_entry_pre_attach,
)


CACHE_PREDICTION_SIDECAR_SUBDIR: str = "cache_prediction"
CACHE_PREDICTION_SIDECAR_BASENAME: str = "predictions.jsonl"
CACHE_PREDICTION_SIDECAR_SCHEMA_VERSION: int = 1


CACHE_PREDICTION_SIDECAR_TYPED_REASONS: frozenset[str] = frozenset({
    "cache_prediction_sidecar_path_invalid",
    "cache_prediction_sidecar_write_failed",
    "cache_prediction_sidecar_read_failed",
    "cache_prediction_sidecar_entry_invalid",
    "cache_prediction_sidecar_empty",
    "cache_prediction_sidecar_bundle_invalid",
})


class CachePredictionSidecarError(Exception):
    """Raised on sidecar writer / reader / entry-attach
    failures.

    Carries the typed-reason string so downstream
    callers can compose typed-unavailable manifests or
    report-table errors without parsing message
    strings.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        if typed_reason not in CACHE_PREDICTION_SIDECAR_TYPED_REASONS:
            # A bug in the caller — fail loud rather
            # than silently widening the typed-reason
            # vocabulary.
            raise ValueError(
                f"typed_reason {typed_reason!r} is not in the "
                f"closed enum "
                f"{sorted(CACHE_PREDICTION_SIDECAR_TYPED_REASONS)}"
            )
        super().__init__(f"{typed_reason}: {message}")
        self.typed_reason = typed_reason
        self.message = message


def _bundle_to_json_obj(bundle: PredictorBundle) -> dict:
    """Serialise a `PredictorBundle` to a plain dict.

    Frozen dataclasses round-trip through
    `dataclasses.asdict` so the writer + reader stay
    in lockstep. `arm_payload` is a `Mapping` so it
    serialises as-is; the reader rebuilds the nested
    dataclasses (`ManifestPairingIndex`,
    `InputSidecarDigest`) explicitly at the read
    boundary.
    """
    return {
        "result_kind": bundle.result_kind,
        "predictor_design_coverage": bundle.predictor_design_coverage,
        "manifest_pairing_index": dataclasses.asdict(
            bundle.manifest_pairing_index,
        ),
        "algorithm_version": bundle.algorithm_version,
        "arm_payload": dict(bundle.arm_payload),
        "input_sidecar_digests": [
            dataclasses.asdict(d) for d in bundle.input_sidecar_digests
        ],
    }


def _json_obj_to_bundle(obj: object) -> PredictorBundle:
    """Rebuild a `PredictorBundle` from a parsed JSON
    dict at the read boundary.

    Raises `CachePredictionSidecarError` with
    `cache_prediction_sidecar_bundle_invalid` on any
    shape mismatch BEFORE the composite validator gets
    a chance — so a malformed-but-loadable JSON dict
    is rejected with the sidecar's typed reason rather
    than a less-specific schema reason.
    """
    if not isinstance(obj, dict):
        raise CachePredictionSidecarError(
            "cache_prediction_sidecar_bundle_invalid",
            f"sidecar row must be a JSON object; got "
            f"{type(obj).__name__}",
        )
    required = {
        "result_kind", "predictor_design_coverage",
        "manifest_pairing_index", "algorithm_version",
        "arm_payload", "input_sidecar_digests",
        "sidecar_schema_version",
    }
    missing = required - obj.keys()
    if missing:
        raise CachePredictionSidecarError(
            "cache_prediction_sidecar_bundle_invalid",
            f"sidecar row missing required fields "
            f"{sorted(missing)}",
        )
    sidecar_version = obj["sidecar_schema_version"]
    # Pin to the integer constant, not just any truthy
    # value: a row carrying e.g. `"1"` (string) or `2`
    # (newer version this code does not know) must fail
    # closed so a downstream consumer cannot quietly
    # accept an incompatible payload shape. Reject
    # `bool` explicitly because `isinstance(True, int)`
    # is True in Python and would otherwise accept
    # `True` as a stand-in for `1`.
    if (
        isinstance(sidecar_version, bool)
        or not isinstance(sidecar_version, int)
        or sidecar_version != CACHE_PREDICTION_SIDECAR_SCHEMA_VERSION
    ):
        raise CachePredictionSidecarError(
            "cache_prediction_sidecar_bundle_invalid",
            f"sidecar_schema_version must be the integer "
            f"{CACHE_PREDICTION_SIDECAR_SCHEMA_VERSION!r}; "
            f"got {sidecar_version!r} "
            f"({type(sidecar_version).__name__})",
        )
    pairing = obj["manifest_pairing_index"]
    if not isinstance(pairing, dict):
        raise CachePredictionSidecarError(
            "cache_prediction_sidecar_bundle_invalid",
            f"manifest_pairing_index must be a JSON object; "
            f"got {type(pairing).__name__}",
        )
    try:
        pairing_index = ManifestPairingIndex(**pairing)
    except TypeError as exc:
        raise CachePredictionSidecarError(
            "cache_prediction_sidecar_bundle_invalid",
            f"manifest_pairing_index has unexpected "
            f"fields: {exc}",
        ) from exc
    digests_raw = obj["input_sidecar_digests"]
    if not isinstance(digests_raw, list):
        raise CachePredictionSidecarError(
            "cache_prediction_sidecar_bundle_invalid",
            f"input_sidecar_digests must be a JSON list; "
            f"got {type(digests_raw).__name__}",
        )
    digests: list[InputSidecarDigest] = []
    for i, entry in enumerate(digests_raw):
        if not isinstance(entry, dict):
            raise CachePredictionSidecarError(
                "cache_prediction_sidecar_bundle_invalid",
                f"input_sidecar_digests[{i}] must be a JSON "
                f"object; got {type(entry).__name__}",
            )
        try:
            digests.append(InputSidecarDigest(**entry))
        except TypeError as exc:
            raise CachePredictionSidecarError(
                "cache_prediction_sidecar_bundle_invalid",
                f"input_sidecar_digests[{i}] has unexpected "
                f"fields: {exc}",
            ) from exc
    arm_payload = obj["arm_payload"]
    if not isinstance(arm_payload, dict):
        raise CachePredictionSidecarError(
            "cache_prediction_sidecar_bundle_invalid",
            f"arm_payload must be a JSON object; got "
            f"{type(arm_payload).__name__}",
        )
    return PredictorBundle(
        result_kind=obj["result_kind"],
        predictor_design_coverage=obj["predictor_design_coverage"],
        manifest_pairing_index=pairing_index,
        algorithm_version=obj["algorithm_version"],
        arm_payload=arm_payload,
        input_sidecar_digests=tuple(digests),
    )


def _sidecar_path(bundle_dir: Path) -> Path:
    """Canonical sidecar path under the bundle dir."""
    return (
        bundle_dir
        / CACHE_PREDICTION_SIDECAR_SUBDIR
        / CACHE_PREDICTION_SIDECAR_BASENAME
    )


def _check_bundle_dir(bundle_dir: Path) -> None:
    if not isinstance(bundle_dir, Path):
        raise CachePredictionSidecarError(
            "cache_prediction_sidecar_path_invalid",
            f"bundle_dir must be a pathlib.Path; got "
            f"{type(bundle_dir).__name__}",
        )


def _sha256_hex(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_cache_prediction_sidecar(
    bundle_dir: Path,
    bundles: Sequence[PredictorBundle],
) -> Tuple[str, str]:
    """Serialise `bundles` to the canonical sidecar
    path and return `(relative_path, sha256_hex)`.

    Every bundle is routed through
    `validate_predictor_bundle` BEFORE the JSONL row
    is written, so an invalid bundle cannot be
    persisted. The composite validator's typed
    reason is preserved verbatim in the
    `cache_prediction_sidecar_bundle_invalid` typed
    reason raised by this helper, so callers can
    trace the failure back to a specific schema gap.

    `bundles` must be a non-empty sequence; an empty
    sidecar is rejected with
    `cache_prediction_sidecar_empty` (the surface
    entry cannot promote to `status="ok"` without
    at least one accepted prediction launch).

    The returned relative path is suitable for
    landing on `CachePredictionSurfaceEntry.artifact_path`;
    the SHA-256 digest is the canonical
    content-addressed identifier of the file's
    bytes, used by
    `attach_sidecar_to_surface_entry` to thread the
    digest onto the entry.
    """
    _check_bundle_dir(bundle_dir)
    if not isinstance(bundles, (list, tuple)) or len(bundles) == 0:
        raise CachePredictionSidecarError(
            "cache_prediction_sidecar_empty",
            "bundles must be a non-empty sequence; an "
            "empty sidecar would let a manifest entry "
            "promote to status=ok with no accepted "
            "prediction launches.",
        )
    for i, bundle in enumerate(bundles):
        if not isinstance(bundle, PredictorBundle):
            raise CachePredictionSidecarError(
                "cache_prediction_sidecar_bundle_invalid",
                f"bundles[{i}] must be a PredictorBundle; "
                f"got {type(bundle).__name__}",
            )
        try:
            validate_predictor_bundle(bundle)
        except PredictorBundleSchemaError as exc:
            raise CachePredictionSidecarError(
                "cache_prediction_sidecar_bundle_invalid",
                f"bundles[{i}] failed schema validation "
                f"({exc.typed_reason}: {exc.message})",
            ) from exc
    target = _sidecar_path(bundle_dir)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CachePredictionSidecarError(
            "cache_prediction_sidecar_write_failed",
            f"failed to create sidecar parent dir "
            f"{target.parent}: {exc}",
        ) from exc
    payload_lines: List[str] = []
    for bundle in bundles:
        obj = _bundle_to_json_obj(bundle)
        obj["sidecar_schema_version"] = (
            CACHE_PREDICTION_SIDECAR_SCHEMA_VERSION
        )
        payload_lines.append(
            json.dumps(obj, sort_keys=True, separators=(",", ":"))
        )
    try:
        target.write_text(
            "\n".join(payload_lines) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise CachePredictionSidecarError(
            "cache_prediction_sidecar_write_failed",
            f"failed to write sidecar at {target}: {exc}",
        ) from exc
    sha256_hex = _sha256_hex(target)
    relative_path = (
        f"{CACHE_PREDICTION_SIDECAR_SUBDIR}/"
        f"{CACHE_PREDICTION_SIDECAR_BASENAME}"
    )
    return relative_path, sha256_hex


def read_cache_prediction_sidecar(
    bundle_dir: Path,
) -> List[PredictorBundle]:
    """Read the sidecar under `bundle_dir` and return
    the validated `PredictorBundle` list.

    Every row is routed through
    `validate_predictor_bundle` at the read boundary,
    so a sidecar that was written through a different
    code path (or hand-edited between write and read)
    cannot smuggle an invalid bundle past the schema.

    Typed reasons raised on failure:
      * `cache_prediction_sidecar_read_failed` —
        file missing or unreadable.
      * `cache_prediction_sidecar_empty` — file
        present but zero non-empty rows.
      * `cache_prediction_sidecar_bundle_invalid` —
        a row parses as JSON but fails the bundle
        shape / schema check.
    """
    _check_bundle_dir(bundle_dir)
    target = _sidecar_path(bundle_dir)
    if not target.is_file():
        raise CachePredictionSidecarError(
            "cache_prediction_sidecar_read_failed",
            f"sidecar not found at {target}",
        )
    try:
        raw = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise CachePredictionSidecarError(
            "cache_prediction_sidecar_read_failed",
            f"failed to read sidecar at {target}: {exc}",
        ) from exc
    bundles: List[PredictorBundle] = []
    for line_no, raw_line in enumerate(raw.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CachePredictionSidecarError(
                "cache_prediction_sidecar_bundle_invalid",
                f"sidecar line {line_no} is not valid JSON: "
                f"{exc}",
            ) from exc
        try:
            bundle = _json_obj_to_bundle(obj)
        except CachePredictionSidecarError as exc:
            # Preserve the helper's typed reason
            # (currently `bundle_invalid` for every
            # shape failure including the new
            # `sidecar_schema_version` checks) and
            # prefix the message with the row's line
            # number so an operator can locate the
            # offending row without scanning the file.
            raise CachePredictionSidecarError(
                exc.typed_reason,
                f"sidecar line {line_no}: {exc.message}",
            ) from exc
        try:
            validate_predictor_bundle(bundle)
        except PredictorBundleSchemaError as exc:
            raise CachePredictionSidecarError(
                "cache_prediction_sidecar_bundle_invalid",
                f"sidecar line {line_no} bundle failed "
                f"schema validation "
                f"({exc.typed_reason}: {exc.message})",
            ) from exc
        bundles.append(bundle)
    if not bundles:
        raise CachePredictionSidecarError(
            "cache_prediction_sidecar_empty",
            f"sidecar at {target} contained zero "
            f"non-empty rows; a sidecar must carry at "
            f"least one validated bundle.",
        )
    return bundles


def attach_sidecar_to_surface_entry(
    entry: CachePredictionSurfaceEntry,
    bundle_dir: Path,
) -> CachePredictionSurfaceEntry:
    """Return a new `CachePredictionSurfaceEntry` with
    the sidecar artifact path + SHA-256 digest
    threaded on.

    The attacher refuses to promote the surface entry
    unless ALL of these hold:
      * The sidecar file exists at the canonical
        path.
      * Every bundle in the sidecar validates clean
        via `validate_predictor_bundle` (re-checked
        at this boundary even though the writer
        already validated — the on-disk artifact
        could have been mutated between write and
        attach).
      * Every bundle's `manifest_pairing_index`
        matches the surface entry's own pairing
        index. A mismatch means the sidecar belongs
        to a different baseline / shape / kernel and
        the surface entry cannot truthfully report
        `ok` against it.

    On any failure raises
    `CachePredictionSidecarError`. On success returns
    a new immutable entry; the input entry is not
    mutated.
    """
    if not isinstance(entry, CachePredictionSurfaceEntry):
        raise CachePredictionSidecarError(
            "cache_prediction_sidecar_entry_invalid",
            f"entry must be a CachePredictionSurfaceEntry; "
            f"got {type(entry).__name__}",
        )
    # Round-trip the entry through the pre-attach
    # base validator before we attach. The full
    # final-entry validator requires the canonical
    # `artifact_path` + `command` — those are exactly
    # what THIS function adds, so the unattached input
    # entry MUST not be routed through the final-shape
    # check here.
    _validate_cache_prediction_surface_entry_pre_attach(entry)
    # The reader re-routes every bundle through the
    # composite validator; on failure the typed reason
    # surfaces directly to the caller.
    bundles = read_cache_prediction_sidecar(bundle_dir)
    target = _sidecar_path(bundle_dir)
    sha256_hex = _sha256_hex(target)
    relative_path = (
        f"{CACHE_PREDICTION_SIDECAR_SUBDIR}/"
        f"{CACHE_PREDICTION_SIDECAR_BASENAME}"
    )
    # Pairing-index cross-check: every bundle's
    # pairing index MUST match the surface entry's.
    for i, bundle in enumerate(bundles):
        if bundle.manifest_pairing_index != entry.manifest_pairing_index:
            raise CachePredictionSidecarError(
                "cache_prediction_sidecar_entry_invalid",
                f"bundles[{i}].manifest_pairing_index does "
                f"not match the surface entry's pairing "
                f"index; sidecar belongs to a different "
                f"baseline / shape / kernel.",
            )
    return dataclasses.replace(
        entry,
        artifact_path=relative_path,
        command=f"sidecar_sha256:{sha256_hex}",
    )


__all__ = [
    "CACHE_PREDICTION_SIDECAR_BASENAME",
    "CACHE_PREDICTION_SIDECAR_SCHEMA_VERSION",
    "CACHE_PREDICTION_SIDECAR_SUBDIR",
    "CACHE_PREDICTION_SIDECAR_TYPED_REASONS",
    "CachePredictionSidecarError",
    "attach_sidecar_to_surface_entry",
    "read_cache_prediction_sidecar",
    "write_cache_prediction_sidecar",
]
