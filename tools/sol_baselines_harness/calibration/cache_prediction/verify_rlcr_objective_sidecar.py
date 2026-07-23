"""Verifier for RLCR-loop objective sidecar JSON files.

Given a sidecar JSON path + its companion manifest JSON path,
validate every cross-field invariant the verdict engine
relies on plus the project's local discipline:

* ``manifest_hash`` matches the SHA-256 of the manifest bytes.
* ``objective_hash`` matches the SHA-256 of the canonical
  objective definition (the ``objective_id`` string).
* ``generated_at`` is a UTC ISO-8601 timestamp in the past
  relative to ``now`` (clamps timestamps that drift into the
  future as misconfigured).
* Identity fields are present and well-typed
  (``sidecar_schema_version``, ``loop_id``, ``round``,
  ``adapter``, ``objective_id``, ``manifest_path``).
* ``correctness.passed`` is a boolean and, when the
  companion smoke artifact is supplied via ``--artifact``,
  the sidecar's claimed
  ``correctness.smoke_artifact_overall_passed`` matches the
  artifact's ``[tiered_gate_report].overall_passed``.

Exit code 0 on clean, 1 on any violation. The runtime emits a
JSON report on stdout with per-check status, so a downstream
consumer can parse the result programmatically.

The verifier lives in the calibration carve-out (NCU-quarantine
allowed import direction) and has no NCU dependency itself; it
is a pure file-bytes + hash + schema validator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


SIDECAR_SCHEMA_VERSION: str = "1.0"
SIDECAR_IDENTITY_FIELDS: tuple = (
    "sidecar_schema_version",
    "loop_id",
    "round",
    "adapter",
    "objective_id",
    "objective_hash",
    "manifest_path",
    "manifest_hash",
    "generated_at",
)


@dataclass(frozen=True)
class VerifierFinding:
    """One verifier finding."""

    check_name: str
    status: str  # "ok" | "fail"
    detail: str


class SidecarVerifierError(Exception):
    """Raised on a malformed sidecar / manifest path that
    cannot be opened or parsed. Distinct from per-check
    failure (which is reported via :class:`VerifierFinding`).
    """


def _file_sha256_hex(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_objective_hash(objective_id: str) -> str:
    """Canonical hash of the objective definition.

    The canonical form is the ``objective_id`` string itself
    (a kebab-cased descriptive slug); future schema extensions
    can canonicalize over additional fields by joining them
    with a stable separator and bumping the schema version.
    """
    return hashlib.sha256(objective_id.encode("utf-8")).hexdigest()


def _read_sidecar(sidecar_path: Path) -> dict[str, Any]:
    try:
        with sidecar_path.open("rb") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise SidecarVerifierError(
            f"cannot read sidecar at {sidecar_path}: {exc}"
        ) from exc


def _read_artifact_toml(
    artifact_path: Optional[Path],
) -> Optional[dict[str, Any]]:
    if artifact_path is None:
        return None
    try:
        with artifact_path.open("rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise SidecarVerifierError(
            f"cannot read artifact at {artifact_path}: {exc}"
        ) from exc


def _check_identity_fields_present(
    sidecar: dict[str, Any],
) -> VerifierFinding:
    missing = [
        f for f in SIDECAR_IDENTITY_FIELDS if f not in sidecar
    ]
    if missing:
        return VerifierFinding(
            check_name="identity_fields_present",
            status="fail",
            detail=f"missing required fields: {missing}",
        )
    return VerifierFinding(
        check_name="identity_fields_present",
        status="ok",
        detail=f"all {len(SIDECAR_IDENTITY_FIELDS)} required fields present",
    )


def _check_schema_version(sidecar: dict[str, Any]) -> VerifierFinding:
    declared = sidecar.get("sidecar_schema_version")
    if declared != SIDECAR_SCHEMA_VERSION:
        return VerifierFinding(
            check_name="schema_version",
            status="fail",
            detail=(
                f"sidecar_schema_version={declared!r} does not match "
                f"expected {SIDECAR_SCHEMA_VERSION!r}"
            ),
        )
    return VerifierFinding(
        check_name="schema_version",
        status="ok",
        detail=f"sidecar_schema_version={declared!r}",
    )


def _check_manifest_hash(
    sidecar: dict[str, Any], sidecar_path: Path,
) -> VerifierFinding:
    declared = sidecar.get("manifest_hash")
    manifest_rel = sidecar.get("manifest_path")
    if not isinstance(declared, str) or not isinstance(manifest_rel, str):
        return VerifierFinding(
            check_name="manifest_hash",
            status="fail",
            detail=(
                f"manifest_hash or manifest_path missing/wrong type "
                f"(manifest_hash={declared!r}, manifest_path={manifest_rel!r})"
            ),
        )
    manifest_full = (sidecar_path.parent.parent.parent / manifest_rel).resolve()
    if not manifest_full.exists():
        manifest_candidate = sidecar_path.parent / Path(manifest_rel).name
        if manifest_candidate.exists():
            manifest_full = manifest_candidate
    if not manifest_full.exists():
        return VerifierFinding(
            check_name="manifest_hash",
            status="fail",
            detail=f"manifest not found at {manifest_full}",
        )
    actual = _file_sha256_hex(manifest_full)
    if actual != declared:
        return VerifierFinding(
            check_name="manifest_hash",
            status="fail",
            detail=(
                f"manifest_hash mismatch: declared {declared!r} vs "
                f"computed {actual!r}"
            ),
        )
    return VerifierFinding(
        check_name="manifest_hash",
        status="ok",
        detail=f"matches SHA-256 of {manifest_full}",
    )


def _check_objective_hash(sidecar: dict[str, Any]) -> VerifierFinding:
    declared = sidecar.get("objective_hash")
    objective_id = sidecar.get("objective_id")
    if not isinstance(declared, str) or not isinstance(objective_id, str):
        return VerifierFinding(
            check_name="objective_hash",
            status="fail",
            detail=(
                f"objective_hash or objective_id missing/wrong type "
                f"(objective_hash={declared!r}, objective_id={objective_id!r})"
            ),
        )
    expected = _canonical_objective_hash(objective_id)
    if declared != expected:
        return VerifierFinding(
            check_name="objective_hash",
            status="fail",
            detail=(
                f"objective_hash mismatch: declared {declared!r} vs "
                f"canonical SHA-256(objective_id)={expected!r}"
            ),
        )
    return VerifierFinding(
        check_name="objective_hash",
        status="ok",
        detail=f"matches SHA-256 of canonical objective_id",
    )


def _check_generated_at(
    sidecar: dict[str, Any],
    now: Optional[datetime] = None,
) -> VerifierFinding:
    declared = sidecar.get("generated_at")
    if not isinstance(declared, str):
        return VerifierFinding(
            check_name="generated_at",
            status="fail",
            detail=f"generated_at missing/wrong type: {declared!r}",
        )
    try:
        ts = datetime.fromisoformat(declared.replace("Z", "+00:00"))
    except ValueError as exc:
        return VerifierFinding(
            check_name="generated_at",
            status="fail",
            detail=f"generated_at is not ISO-8601: {declared!r} ({exc})",
        )
    if ts.tzinfo is None:
        return VerifierFinding(
            check_name="generated_at",
            status="fail",
            detail=(
                f"generated_at lacks timezone info: {declared!r}; "
                f"expected explicit UTC (+00:00 or Z suffix)"
            ),
        )
    reference = now or datetime.now(timezone.utc)
    if ts > reference:
        return VerifierFinding(
            check_name="generated_at",
            status="fail",
            detail=(
                f"generated_at={declared!r} is in the future relative "
                f"to reference time {reference.isoformat()}"
            ),
        )
    return VerifierFinding(
        check_name="generated_at",
        status="ok",
        detail=f"generated_at={declared!r} is in the past",
    )


def _check_correctness_passed(
    sidecar: dict[str, Any],
    artifact: Optional[dict[str, Any]],
) -> VerifierFinding:
    correctness = sidecar.get("correctness")
    if not isinstance(correctness, dict):
        return VerifierFinding(
            check_name="correctness_passed",
            status="fail",
            detail="correctness block missing or wrong type",
        )
    passed = correctness.get("passed")
    if not isinstance(passed, bool):
        return VerifierFinding(
            check_name="correctness_passed",
            status="fail",
            detail=f"correctness.passed missing/wrong type: {passed!r}",
        )
    if artifact is None:
        return VerifierFinding(
            check_name="correctness_passed",
            status="ok",
            detail=(
                f"correctness.passed={passed} (no artifact provided "
                f"for cross-check)"
            ),
        )
    tgr = artifact.get("tiered_gate_report") or {}
    artifact_overall_passed = tgr.get("overall_passed")
    if isinstance(artifact_overall_passed, str):
        artifact_overall_passed = (
            artifact_overall_passed.lower() == "true"
        )
    declared_artifact_passed = (
        correctness.get("smoke_artifact_overall_passed")
    )
    if (
        declared_artifact_passed is not None
        and artifact_overall_passed is not None
        and bool(declared_artifact_passed) != bool(artifact_overall_passed)
    ):
        return VerifierFinding(
            check_name="correctness_passed",
            status="fail",
            detail=(
                f"sidecar declares "
                f"smoke_artifact_overall_passed={declared_artifact_passed!r} "
                f"but artifact records "
                f"tiered_gate_report.overall_passed={artifact_overall_passed!r}"
            ),
        )
    return VerifierFinding(
        check_name="correctness_passed",
        status="ok",
        detail=(
            f"correctness.passed={passed}; artifact-claim is consistent "
            f"with tiered_gate_report.overall_passed="
            f"{artifact_overall_passed!r}"
        ),
    )


_SMOKE_META_COMMIT_SHA_PREFIX: str = (
    "smoke_artifact_meta_commit_sha_"
)


def _check_smoke_artifact_meta_commit_sha(
    sidecar: dict[str, Any],
    artifact: Optional[dict[str, Any]],
) -> VerifierFinding:
    """When a sidecar carries a
    ``correctness.smoke_artifact_meta_commit_sha_<sha>...``
    key set to ``True``, the named SHA MUST match the
    smoke artifact's ``[meta].commit_sha``.

    An earlier review surfaced this exact failure surface:
    a prior sidecar asserted
    ``smoke_artifact_meta_commit_sha_<one-hex>_identifies_<context>=True``
    while the TOML artifact recorded a different
    ``meta.commit_sha=<other-hex>``. The earlier
    verifier did not catch the overclaim because it
    did not parse the per-claim SHA suffix.

    Returns ``ok`` when:
      * no claim key matches the prefix (the field is
        optional), OR
      * no artifact is supplied via ``--artifact`` so a
        cross-check is impossible (records a no-artifact
        OK finding so the caller can audit it).

    Returns ``fail`` when:
      * the claim names a SHA AND the artifact records
        a different ``meta.commit_sha``, OR
      * the artifact's ``meta.commit_sha`` is missing /
        empty / wrong type and a claim was made.

    The first 7+ hex chars suffice for the SHA prefix
    match (matching the project's commit-naming
    convention of full + short SHAs in commit messages).
    """
    correctness = sidecar.get("correctness")
    if not isinstance(correctness, dict):
        return VerifierFinding(
            check_name="smoke_artifact_meta_commit_sha",
            status="ok",
            detail=(
                "no correctness block to inspect for "
                "smoke_artifact_meta_commit_sha_<sha> claim"
            ),
        )
    claim_keys = [
        k for k, v in correctness.items()
        if isinstance(k, str)
        and k.startswith(_SMOKE_META_COMMIT_SHA_PREFIX)
        and v is True
    ]
    if not claim_keys:
        return VerifierFinding(
            check_name="smoke_artifact_meta_commit_sha",
            status="ok",
            detail=(
                "no smoke_artifact_meta_commit_sha_<sha>... "
                "claim found in correctness block"
            ),
        )
    if artifact is None:
        return VerifierFinding(
            check_name="smoke_artifact_meta_commit_sha",
            status="ok",
            detail=(
                f"sidecar claims commit_sha but no artifact "
                f"provided to cross-check ({len(claim_keys)} "
                f"claim(s) recorded)"
            ),
        )
    meta = artifact.get("meta") or {}
    actual_sha = meta.get("commit_sha")
    if not isinstance(actual_sha, str) or not actual_sha:
        return VerifierFinding(
            check_name="smoke_artifact_meta_commit_sha",
            status="fail",
            detail=(
                f"sidecar claims smoke_artifact_meta_commit_sha "
                f"but artifact's [meta].commit_sha is missing/"
                f"wrong type ({actual_sha!r})"
            ),
        )
    actual_sha_lower = actual_sha.lower()
    bad_claims: list[str] = []
    for key in claim_keys:
        named_sha = key[len(_SMOKE_META_COMMIT_SHA_PREFIX):]
        # Extract the leading hex run from the key suffix.
        hex_chars: list[str] = []
        for ch in named_sha:
            if ch in "0123456789abcdef":
                hex_chars.append(ch)
            else:
                break
        named_sha_hex = "".join(hex_chars).lower()
        if len(named_sha_hex) < 7:
            bad_claims.append(
                f"{key!r} (named SHA too short: "
                f"{named_sha_hex!r})"
            )
            continue
        if not actual_sha_lower.startswith(named_sha_hex):
            bad_claims.append(
                f"{key!r} names {named_sha_hex!r} but "
                f"artifact records {actual_sha_lower!r}"
            )
    if bad_claims:
        return VerifierFinding(
            check_name="smoke_artifact_meta_commit_sha",
            status="fail",
            detail=(
                "sidecar commit_sha claim(s) disagree with "
                "artifact: " + "; ".join(bad_claims)
            ),
        )
    return VerifierFinding(
        check_name="smoke_artifact_meta_commit_sha",
        status="ok",
        detail=(
            f"{len(claim_keys)} commit_sha claim(s) match "
            f"artifact's [meta].commit_sha={actual_sha_lower!r}"
        ),
    )


_AC_DELTAS_COMMIT_SHA_TOKEN_PREFIXES: tuple[str, ...] = (
    "artifact_meta_commit_sha_",
    "per_cell_meta_commit_sha_",
)


def _extract_hex_run_after_token(
    text: str, token: str,
) -> list[str]:
    """Return every leading hex run that appears in ``text``
    immediately after each occurrence of ``token``.

    The hex run is the longest contiguous prefix of
    ``[0-9a-f]`` characters (case-insensitive); the run
    stops at the first non-hex character or end-of-string.
    Returns one entry per occurrence of the token, in
    left-to-right order. Empty-hex runs are returned as
    empty strings so the caller can flag too-short-suffix
    cases separately.
    """
    if not text or not token:
        return []
    lower = text.lower()
    needle = token.lower()
    results: list[str] = []
    start = 0
    while True:
        idx = lower.find(needle, start)
        if idx < 0:
            break
        suffix_start = idx + len(needle)
        hex_chars: list[str] = []
        for ch in lower[suffix_start:]:
            if ch in "0123456789abcdef":
                hex_chars.append(ch)
            else:
                break
        results.append("".join(hex_chars))
        start = suffix_start + max(len(hex_chars), 1)
    return results


def _check_ac_deltas_artifact_meta_commit_sha(
    sidecar: dict[str, Any],
    artifact: Optional[dict[str, Any]],
) -> VerifierFinding:
    """When any ``ac_deltas[*].from`` or ``ac_deltas[*].to``
    string carries an ``artifact_meta_commit_sha_<hex>`` or
    ``per_cell_meta_commit_sha_<hex>`` token, the named hex
    prefix MUST match the smoke artifact's
    ``[meta].commit_sha``.

    Companion to
    :func:`_check_smoke_artifact_meta_commit_sha`. That
    check guards the prefixed correctness key shape; this
    check guards the same provenance pattern when it
    appears embedded inside the descriptive transition
    strings of ``ac_deltas``. A sidecar that names a
    commit-SHA-shaped hex run after either token in any
    ``ac_deltas`` value is asserting that the named hex
    matches the artifact's commit, and the artifact is
    the ground truth.

    Returns ``ok`` when:
      * no ``ac_deltas`` block exists (the field is
        optional), OR
      * no token of either prefix appears in any
        ``ac_deltas[*].from`` / ``ac_deltas[*].to`` value,
        OR
      * every occurrence of an ``artifact_meta_commit_sha_``
        / ``per_cell_meta_commit_sha_`` substring is
        followed by fewer than 7 hex chars (treated as
        INCIDENTAL substring overlap — see the divergence
        note below; the artifact's meta validity is NOT
        consulted in this case because there is no
        canonical claim to cross-check), OR
      * at least one canonical claim (≥ 7 hex chars) is
        present but no artifact is supplied via
        ``--artifact`` so a cross-check is impossible
        (records a no-artifact OK finding so the caller
        can audit it).

    Returns ``fail`` when:
      * at least one canonical claim is present AND the
        artifact's ``meta.commit_sha`` is missing / empty /
        wrong type, OR
      * any extracted hex prefix (≥ 7 chars) does not
        match the artifact's recorded ``meta.commit_sha``.

    Deliberate divergence from
    :func:`_check_smoke_artifact_meta_commit_sha`: the
    companion check operates on STRUCTURED KEY SUFFIXES,
    where the entire suffix MUST be a SHA — any
    ``smoke_artifact_meta_commit_sha_<too-short-suffix>``
    key is a typo and fails closed. This check operates
    on FREE-TEXT VALUES, where incidental substring
    overlap is expected (e.g., a sidecar value mentioning
    the verifier-check name
    ``smoke_artifact_meta_commit_sha_cross_check`` has
    a 1-char hex run ``c`` followed by ``_``). Treating
    those short runs as fail would false-positive on
    legitimate function-name references; treating them
    as incidental ok preserves the check's signal on
    real ≥ 7-char SHA claims while admitting the textual
    references that natural sidecar prose carries.
    """
    ac_deltas = sidecar.get("ac_deltas")
    if not isinstance(ac_deltas, dict):
        return VerifierFinding(
            check_name="ac_deltas_artifact_meta_commit_sha",
            status="ok",
            detail=(
                "no ac_deltas block to inspect for "
                "artifact_meta_commit_sha_<hex> / "
                "per_cell_meta_commit_sha_<hex> tokens"
            ),
        )
    token_occurrences: list[tuple[str, str, str, str]] = []
    # (ac_key, side, token, extracted_hex)
    for ac_key, delta in ac_deltas.items():
        if not isinstance(delta, dict):
            continue
        for side in ("from", "to"):
            value = delta.get(side)
            if not isinstance(value, str):
                continue
            for token in _AC_DELTAS_COMMIT_SHA_TOKEN_PREFIXES:
                hex_runs = _extract_hex_run_after_token(
                    value, token,
                )
                for hex_run in hex_runs:
                    token_occurrences.append(
                        (ac_key, side, token, hex_run),
                    )
    if not token_occurrences:
        return VerifierFinding(
            check_name="ac_deltas_artifact_meta_commit_sha",
            status="ok",
            detail=(
                "no artifact_meta_commit_sha_<hex> or "
                "per_cell_meta_commit_sha_<hex> tokens "
                "found in ac_deltas[*].from / "
                "ac_deltas[*].to strings"
            ),
        )
    # Split into canonical (>= 7 hex) claims and incidental
    # (< 7 hex) substring overlaps BEFORE consulting the
    # artifact. The artifact-meta validity check only gates
    # on the canonical claims: when the sidecar carries no
    # canonical claim, there's nothing to cross-check and
    # the artifact's meta is irrelevant to the verdict.
    # The companion :func:`_check_smoke_artifact_meta_commit_sha`
    # uses the stricter "any suffix < 7 is a typo" semantic
    # because it operates on structured key suffixes; this
    # check operates on free-text values where incidental
    # overlap (e.g., a sidecar mentioning the verifier-check
    # name ``smoke_artifact_meta_commit_sha_cross_check``)
    # is expected and benign.
    canonical_claims = [
        (ac_key, side, token, hex_run)
        for (ac_key, side, token, hex_run) in token_occurrences
        if len(hex_run) >= 7
    ]
    if not canonical_claims:
        return VerifierFinding(
            check_name="ac_deltas_artifact_meta_commit_sha",
            status="ok",
            detail=(
                "no ac_deltas[*].from / ac_deltas[*].to "
                "string carries a ≥ 7-hex-char SHA suffix "
                "after artifact_meta_commit_sha_ / "
                "per_cell_meta_commit_sha_ "
                f"(incidental substring overlaps seen: "
                f"{len(token_occurrences)})"
            ),
        )
    if artifact is None:
        return VerifierFinding(
            check_name="ac_deltas_artifact_meta_commit_sha",
            status="ok",
            detail=(
                f"ac_deltas carries commit_sha-shaped tokens "
                f"but no artifact provided to cross-check "
                f"({len(canonical_claims)} canonical claim(s) "
                f"seen)"
            ),
        )
    meta = artifact.get("meta") or {}
    actual_sha = meta.get("commit_sha")
    if not isinstance(actual_sha, str) or not actual_sha:
        return VerifierFinding(
            check_name="ac_deltas_artifact_meta_commit_sha",
            status="fail",
            detail=(
                f"ac_deltas carries {len(canonical_claims)} "
                f"canonical commit_sha claim(s) but artifact's "
                f"[meta].commit_sha is missing/wrong type "
                f"({actual_sha!r})"
            ),
        )
    actual_sha_lower = actual_sha.lower()
    bad: list[str] = []
    for ac_key, side, token, hex_run in canonical_claims:
        if not actual_sha_lower.startswith(hex_run):
            bad.append(
                f"ac_deltas[{ac_key!r}][{side!r}] contains "
                f"{token}{hex_run} but artifact records "
                f"{actual_sha_lower!r}"
            )
    if bad:
        return VerifierFinding(
            check_name="ac_deltas_artifact_meta_commit_sha",
            status="fail",
            detail=(
                "ac_deltas commit_sha-shaped token(s) "
                "disagree with artifact: "
                + "; ".join(bad)
            ),
        )
    return VerifierFinding(
        check_name="ac_deltas_artifact_meta_commit_sha",
        status="ok",
        detail=(
            f"{len(canonical_claims)} ac_deltas commit_sha "
            f"token(s) match artifact's [meta].commit_sha="
            f"{actual_sha_lower!r}"
        ),
    )


def verify_sidecar(
    sidecar_path: Path,
    *,
    artifact_path: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> tuple[VerifierFinding, ...]:
    """Run every check against the sidecar; return the
    findings tuple.

    Raises :class:`SidecarVerifierError` only when the sidecar
    or artifact cannot be opened/parsed — every other failure
    surfaces as a ``VerifierFinding`` with ``status="fail"``.
    """
    sidecar = _read_sidecar(sidecar_path)
    artifact = _read_artifact_toml(artifact_path)
    return (
        _check_identity_fields_present(sidecar),
        _check_schema_version(sidecar),
        _check_manifest_hash(sidecar, sidecar_path),
        _check_objective_hash(sidecar),
        _check_generated_at(sidecar, now=now),
        _check_correctness_passed(sidecar, artifact),
        _check_smoke_artifact_meta_commit_sha(sidecar, artifact),
        _check_ac_deltas_artifact_meta_commit_sha(sidecar, artifact),
    )


def canonical_objective_hash(objective_id: str) -> str:
    """Public helper: compute the canonical objective_hash
    for a given objective_id (SHA-256 of the UTF-8 bytes).
    Sidecar authors call this to populate the
    ``objective_hash`` field deterministically.
    """
    return _canonical_objective_hash(objective_id)


def _main_cli() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the cross-field invariants of an RLCR-loop "
            "objective sidecar JSON file."
        ),
    )
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help=(
            "Optional smoke artifact TOML for cross-checking "
            "correctness.smoke_artifact_overall_passed against the "
            "artifact's tiered_gate_report.overall_passed."
        ),
    )
    args = parser.parse_args()
    try:
        findings = verify_sidecar(
            args.sidecar, artifact_path=args.artifact,
        )
    except SidecarVerifierError as exc:
        print(
            json.dumps({"error": str(exc)}, indent=2),
            file=sys.stderr,
        )
        return 2
    payload = {
        "sidecar": str(args.sidecar),
        "artifact": str(args.artifact) if args.artifact else None,
        "findings": [
            {
                "check": f.check_name,
                "status": f.status,
                "detail": f.detail,
            }
            for f in findings
        ],
        "overall_status": (
            "ok"
            if all(f.status == "ok" for f in findings)
            else "fail"
        ),
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["overall_status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(_main_cli())


__all__ = [
    "SIDECAR_SCHEMA_VERSION",
    "SIDECAR_IDENTITY_FIELDS",
    "VerifierFinding",
    "SidecarVerifierError",
    "verify_sidecar",
    "canonical_objective_hash",
]
