"""Operator-facing measurement-manifest pre-flight validator.

The operator runs this CLI on the local host (typically the
RTX 5060 laptop after a cross-host transfer) BEFORE running the
live sweep, to verify each operator-supplied per-cell
measurement-manifest JSON is well-formed and (optionally) that
the manifest's ``binary_sha256`` matches the actual binary file
the sweep will profile.

Manifest schema (per the cross-host pivot runbook):

  {
    "schema_version": 2,
    "slug": "<slug>",
    "shape_point": "<shape>",
    "binary_sha256": "<64-char lowercase hex sha256>",
    "kernel_signature_sha256": "<64-char lowercase hex sha256>"
  }

The label_driver.sh + sweep_repeatability_matrix.py harness
already fails closed on malformed manifests at live-sweep time,
but only AFTER NCU is invoked. This pre-flight tool surfaces the
error instantly so the operator does not burn wall-clock budget
on a misconfigured run.

CLI:

  python3 tools/local_labels/validate_measurement_manifest.py \
      --manifest <path>[,...] \
      [--binary <cell>=<path>[,...]]

``--manifest`` is repeatable; each manifest is validated
independently and the tool reports a per-manifest verdict on
the first failure (or aggregate `<N> manifest(s) valid` on
success).

``--binary`` is an optional cell-to-binary map (matching the
sweep harness's flag shape; cell key is
``<arch>/<slug>/<shape>``, optionally suffixed with a
register-cap build-variant token
``.../register_cap_<cap>``). When provided, the tool computes
the binary file's sha256 and verifies the matching manifest's
``binary_sha256`` field. Manifests and binaries are associated
on a mixed-arity key: ``(slug, shape_point)`` for a base
manifest/binary, or ``(slug, shape_point, variant_id)`` for a
register-cap build-variant, so sibling caps of the same
slug/shape match distinct binary keys without collapsing. A
manifest's optional ``register_cap`` / ``variant_id`` fields are
validated against the same canonical contract as the staging
producer and ``label_driver.sh``.

Exit codes:
- 0: all manifests schema-valid AND (if ``--binary`` set) every
  manifest matches exactly one binary entry, every binary entry
  matches exactly one manifest, and every cross-checked sha256
  matches.
- 1: schema validation failure (missing field, wrong type, bad
  hex, wrong length), binary sha256 mismatch, or ``--binary``
  cell-to-manifest coverage mismatch (unmatched binary cell
  key, unmatched manifest, or either side matching multiple
  entries on the other side).
- 2: io error (manifest file not found, manifest read error,
  malformed JSON, invalid UTF-8 manifest content, binary file
  not found, binary file unreadable when ``--binary`` is set).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# The shared build-variant identity normalizer is a dependency-free leaf
# module living beside this script; a sibling import works both when this tool
# is run as a script (sys.path[0] is its own dir) and when imported under the
# tools/local_labels test harness. Using the ONE normalizer keeps this
# operator-preflight consumer in lockstep with the staging producer + the
# label_driver.sh gate (no re-inlined second copy that could drift).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from variant_identity import VARIANT_PREFIX, variant_identity  # noqa: E402


_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_STRING_FIELDS = (
    "slug",
    "shape_point",
    "binary_sha256",
    "kernel_signature_sha256",
)
_SHA256_FIELDS = ("binary_sha256", "kernel_signature_sha256")

# The on-disk per-shape result directory follows the convention
# ``trace_<shape_point>``; the join-key value is the bare
# shape_point token. A manifest declaring
# ``shape_point: "trace_trace_00"`` has pasted the directory name
# into the value field, which would silently break the 7-field
# comparison-runner join. We surface it explicitly so the
# operator catches the directory-vs-value confusion BEFORE NCU is
# invoked.
_SHAPE_POINT_DIRECTORY_NAME_DRIFT_RE = re.compile(r"^trace_trace_")
_SHAPE_POINT_DIRECTORY_NAME_DRIFT_TOKEN = (
    "shape_point_directory_name_drift"
)


def _diag(msg: str) -> None:
    sys.stderr.write(f"validate_measurement_manifest: {msg}\n")


def _parse_binary_map(arg_values: list[str]) -> dict[str, Path]:
    """Parse the comma-separated ``<cell>=<path>`` map (the same
    shape the sweep harness uses for ``--binaries``). Duplicate
    cell keys (the same cell appearing twice via repeated
    ``--binary`` flags OR within a single comma-separated
    argument) are rejected at parse time so a stale + current
    pair cannot silently collapse to the last-supplied path."""
    out: dict[str, Path] = {}
    for arg in arg_values or []:
        for entry in arg.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if "=" not in entry:
                raise ValueError(
                    f"--binary entry {entry!r} is not in <cell>=<path> form"
                )
            cell, _, path_str = entry.partition("=")
            cell = cell.strip()
            if cell in out:
                raise ValueError(
                    f"--binary cell key {cell!r} appears more than "
                    f"once; supply each cell exactly once"
                )
            out[cell] = Path(path_str.strip())
    return out


def _validate_schema(payload: dict, manifest_path: Path) -> str | None:
    """Return None on success; on failure return a one-line
    diagnostic identifying the offending field/value."""
    schema_version = payload.get("schema_version")
    if schema_version != 2:
        return (
            f"manifest at {manifest_path!s}: invalid schema_version "
            f"{schema_version!r} (expected 2)"
        )
    for field in _REQUIRED_STRING_FIELDS:
        value = payload.get(field)
        if value is None:
            return (
                f"manifest at {manifest_path!s}: missing required "
                f"field {field!r}"
            )
        if not isinstance(value, str):
            return (
                f"manifest at {manifest_path!s}: field {field!r} "
                f"must be a string; got {type(value).__name__}"
            )
    for field in _SHA256_FIELDS:
        value = payload[field]
        if len(value) != 64:
            return (
                f"manifest at {manifest_path!s}: field {field!r} "
                f"must be 64 lowercase hex chars; got "
                f"{len(value)} chars"
            )
        if not _HEX_RE.match(value):
            return (
                f"manifest at {manifest_path!s}: field {field!r} "
                f"must be 64 lowercase hex chars; value {value!r} "
                f"contains non-hex characters"
            )
    shape_point_value = payload["shape_point"]
    if _SHAPE_POINT_DIRECTORY_NAME_DRIFT_RE.match(shape_point_value):
        return (
            f"manifest at {manifest_path!s}: "
            f"{_SHAPE_POINT_DIRECTORY_NAME_DRIFT_TOKEN}: "
            f"shape_point={shape_point_value!r} appears to be the "
            f"per-shape result directory name "
            f"('trace_<shape_point>'); the join-key value should "
            f"be the bare shape_point token (e.g. "
            f"'{shape_point_value[len('trace_'):]}')"
        )
    # Optional register-cap build-variant identity (additive schema fields,
    # validated in LOCKSTEP with measurement_manifest.schema.json + the driver
    # gate). The manifest is an operator-input boundary, so a STRICT type
    # pre-check fronts the tolerate-absence normalizer: a present-but-malformed
    # register_cap (JSON string / bool) or variant_id (non-string / empty) is a
    # declared-but-INVALID value that must ERROR here, not pass preflight and be
    # rejected only at live-sweep time. A noncanonical alias (register_cap_064)
    # or an inconsistent register_cap/variant_id pair likewise fails closed.
    if "register_cap" in payload and (
        not isinstance(payload["register_cap"], int)
        or isinstance(payload["register_cap"], bool)
    ):
        return (
            f"manifest at {manifest_path!s}: field 'register_cap' is present "
            f"but not an integer; got "
            f"{type(payload['register_cap']).__name__}"
        )
    if "variant_id" in payload and (
        not isinstance(payload["variant_id"], str)
        or not payload["variant_id"]
    ):
        return (
            f"manifest at {manifest_path!s}: field 'variant_id' is present "
            f"but not a non-empty string"
        )
    variant_record = {
        k: payload[k] for k in ("register_cap", "variant_id") if k in payload
    }
    _vid, _cap, variant_err = variant_identity(variant_record)
    if variant_err is not None:
        return (
            f"manifest at {manifest_path!s}: inconsistent build-variant "
            f"identity: {variant_err}"
        )
    return None


def _compute_binary_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _manifest_variant_id(payload: dict) -> str:
    """Return the canonical build-variant id (``register_cap_<cap>``) declared
    by a manifest, or ``""`` for a base manifest. Assumes the manifest already
    passed ``_validate_schema`` (so the optional pair is type-checked +
    consistent)."""
    record = {
        k: payload[k] for k in ("register_cap", "variant_id") if k in payload
    }
    variant_id, _cap, _err = variant_identity(record)
    return variant_id


def _manifest_match_key(payload: dict) -> tuple:
    """Mixed-arity manifest match key: the historical ``(slug, shape_point)``
    2-tuple for a base manifest (BYTE-IDENTICAL to the pre-variant behavior),
    widened to ``(slug, shape_point, variant_id)`` only when the manifest
    declares a build variant. Mirrors the aggregator's mixed-arity cell key so a
    base cell's matching + diagnostics are unchanged."""
    base = (payload["slug"], payload["shape_point"])
    variant_id = _manifest_variant_id(payload)
    return base if not variant_id else base + (variant_id,)


def _binary_cell_match_key(cell: str) -> tuple[tuple | None, str | None]:
    """Parse a ``--binary`` cell key into a mixed-arity match key, accepting the
    historical ``<slug>/<shape>`` and ``<arch>/<slug>/<shape>`` forms PLUS the
    variant forms ``<slug>/<shape>/register_cap_<cap>`` and
    ``<arch>/<slug>/<shape>/register_cap_<cap>``. Returns ``(key, None)`` where
    key is ``(slug, shape)`` (base) or ``(slug, shape, variant_id)`` (variant),
    or ``(None, error)`` on a noncanonical variant suffix or a too-short key.

    A trailing ``register_cap_*`` component is the build-variant suffix (a base
    shape_point never starts with that prefix); it is normalized canonically via
    the shared ``variant_identity`` so a noncanonical alias (``register_cap_064``)
    is rejected here, mirroring the schema + driver gate."""
    parts = cell.split("/")
    variant_id = ""
    if parts and parts[-1].startswith(VARIANT_PREFIX):
        vid, _cap, err = variant_identity({"variant_id": parts[-1]})
        if err is not None:
            return (
                None,
                f"--binary cell key {cell!r} has a noncanonical build-variant "
                f"suffix: {err}",
            )
        variant_id = vid
        parts = parts[:-1]
    if len(parts) < 2:
        return (
            None,
            f"--binary cell key {cell!r} must include at least "
            f"slug/shape_point",
        )
    base = (parts[-2], parts[-1])
    return (base if not variant_id else base + (variant_id,), None)


def _check_binary_map_coverage(
    manifests: list[tuple[Path, dict]],
    binary_map: dict[str, Path],
) -> str | None:
    """Bidirectional ``--binary``↔manifest coverage check. When the
    operator supplies ``--binary``, every manifest must match
    exactly one binary entry AND every binary entry must match
    exactly one manifest. Cell-key matching uses the last two
    path components, so both ``<arch>/<slug>/<shape>`` (canonical
    sweep-harness form) and bare ``<slug>/<shape>`` forms are
    accepted. Returns ``None`` on success; on failure returns a
    one-line typed diagnostic naming every unmatched or
    over-matched cell or manifest."""
    manifest_keys: list[tuple] = [
        _manifest_match_key(p) for _, p in manifests
    ]
    manifest_to_binaries: dict[tuple, list[str]] = {
        k: [] for k in manifest_keys
    }
    binary_to_manifests: dict[str, list[tuple]] = {
        cell: [] for cell in binary_map
    }
    for cell in binary_map:
        cell_key, key_err = _binary_cell_match_key(cell)
        if key_err is not None:
            return key_err
        for mkey in manifest_keys:
            if cell_key == mkey:
                manifest_to_binaries[mkey].append(cell)
                binary_to_manifests[cell].append(mkey)

    unmatched_binaries = sorted(
        cell for cell, mkeys in binary_to_manifests.items()
        if not mkeys
    )
    unmatched_manifests = sorted(
        mkey for mkey, cells in manifest_to_binaries.items()
        if not cells
    )
    over_matched_binaries = sorted(
        cell for cell, mkeys in binary_to_manifests.items()
        if len(mkeys) > 1
    )
    over_matched_manifests = sorted(
        mkey for mkey, cells in manifest_to_binaries.items()
        if len(cells) > 1
    )

    if not (unmatched_binaries or unmatched_manifests
            or over_matched_binaries or over_matched_manifests):
        return None

    parts = ["--binary cell-to-manifest coverage mismatch"]
    if unmatched_binaries:
        parts.append(
            f"unmatched binary cell keys: {unmatched_binaries!r}"
        )
    if unmatched_manifests:
        parts.append(
            f"unmatched manifests (slug, shape_point): "
            f"{unmatched_manifests!r}"
        )
    if over_matched_binaries:
        parts.append(
            f"binary cell keys matching multiple manifests: "
            f"{over_matched_binaries!r}"
        )
    if over_matched_manifests:
        parts.append(
            f"manifests (slug, shape_point) matched by multiple "
            f"binary cells: {over_matched_manifests!r}"
        )
    return "; ".join(parts)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate operator-supplied per-cell "
            "measurement-manifest JSON files (pre-flight before "
            "the cross-host live sweep)."
        ),
    )
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        required=True,
        help=(
            "Path to a measurement-manifest JSON file. May be "
            "specified multiple times to validate multiple "
            "manifests in one invocation."
        ),
    )
    parser.add_argument(
        "--binary",
        action="append",
        default=None,
        help=(
            "Optional cell-to-binary map in <cell>=<path> form "
            "(comma-separated and/or repeatable). When set, the "
            "tool computes each binary's sha256 and verifies the "
            "matching manifest's `binary_sha256` field. Cell key "
            "is matched to manifest by slug + shape_point, plus the "
            "register-cap build-variant token when present "
            "(<slug>/<shape>/register_cap_<cap>)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        binary_map = _parse_binary_map(args.binary)
    except ValueError as exc:
        _diag(str(exc))
        return 1

    # Pass 1: load + schema-validate every manifest. We need every
    # manifest's (slug, shape_point) before we can run the
    # bidirectional --binary coverage check, so collect the
    # validated payloads here and defer hashing to pass 3.
    manifests: list[tuple[Path, dict]] = []
    for manifest_path in args.manifest:
        if not manifest_path.is_file():
            _diag(f"manifest file not found at {manifest_path!s}")
            return 2
        try:
            with manifest_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except json.JSONDecodeError as exc:
            _diag(
                f"manifest at {manifest_path!s} did not parse as "
                f"JSON: {exc}"
            )
            return 2
        except UnicodeDecodeError as exc:
            _diag(
                f"manifest at {manifest_path!s} is not valid UTF-8: "
                f"{exc}"
            )
            return 2
        except OSError as exc:
            _diag(
                f"manifest at {manifest_path!s} could not be read: "
                f"{exc}"
            )
            return 2
        if not isinstance(payload, dict):
            _diag(
                f"manifest at {manifest_path!s} root must be a "
                f"JSON object; got {type(payload).__name__}"
            )
            return 1

        schema_err = _validate_schema(payload, manifest_path)
        if schema_err is not None:
            _diag(schema_err)
            return 1

        manifests.append((manifest_path, payload))

    # Pass 2: bidirectional ``--binary``↔manifest coverage check.
    # A typoed cell key (no matching manifest) or a manifest with
    # no matching binary entry must fail closed BEFORE any binary
    # is hashed; otherwise the operator can pass an intended
    # binary map, mistype one cell token, and receive a success
    # verdict without the binary ever being read.
    if binary_map:
        coverage_err = _check_binary_map_coverage(
            manifests, binary_map,
        )
        if coverage_err is not None:
            _diag(coverage_err)
            return 1

    # Pass 3: hash each binary and cross-check against the
    # matching manifest's claimed sha256. Pass 2 has already
    # ensured every binary entry has exactly one matching
    # manifest (and vice versa).
    for manifest_path, payload in manifests:
        if binary_map:
            mkey = _manifest_match_key(payload)
            # Variant-aware (mixed-arity) cell-key matching: a base manifest
            # matches base binary keys (<slug>/<shape> or
            # <arch>/<slug>/<shape>); a build-variant manifest matches only its
            # own .../register_cap_<cap> key. Pass 2 already rejected any
            # noncanonical binary suffix, so a parse error here is treated as a
            # non-match defensively.
            matching_cells = []
            for cell, binary_path in binary_map.items():
                cell_key, key_err = _binary_cell_match_key(cell)
                if key_err is None and cell_key == mkey:
                    matching_cells.append((cell, binary_path))
            for cell, binary_path in matching_cells:
                if not binary_path.is_file():
                    _diag(
                        f"binary at {binary_path!s} (cell {cell!r}) "
                        f"not found"
                    )
                    return 2
                try:
                    actual_sha = _compute_binary_sha256(binary_path)
                except OSError as exc:
                    _diag(
                        f"binary at {binary_path!s} (cell {cell!r}) "
                        f"could not be hashed: {exc}"
                    )
                    return 2
                claimed_sha = payload["binary_sha256"]
                if actual_sha != claimed_sha:
                    _diag(
                        f"binary_sha256 mismatch for cell {cell!r} "
                        f"(manifest {manifest_path!s}): manifest "
                        f"claims {claimed_sha}, computed "
                        f"{actual_sha}"
                    )
                    return 1

    valid_count = len(manifests)
    sys.stdout.write(
        f"validate_measurement_manifest: {valid_count} manifest(s) "
        f"valid\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
