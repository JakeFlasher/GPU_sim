"""Refresh ``predictor_build_commit`` / ``regeneration_build_commit``
on committed sweep-dir artifacts so the field resolves inside the
audit's {HEAD, HEAD^} accept window after follow-up commits advance
HEAD.

Companion to
``tools/local_labels/refresh_predictor_trace_source_commit.py``
(which targets the trace JSON sidecars under
``workloads/simbench_bw/results/<slug>/<arch>/trace_<shape>.json``
and the ``source_commit`` field). The trace-sidecar helper does
not cover the per-sweep ``predictor_bundles/<arch>__<slug>__<shape>.json``
artifacts or the sweep-level ``regeneration_delta.json`` because
those files do NOT carry a ``binary_sha256`` field whose contract
the trace-sidecar helper enforces as a safety check; instead these
artifacts carry their producing-code's commit SHA in
``predictor_build_commit`` (per-bundle) or
``regeneration_build_commit`` (per-delta).

Per ``BL-20260508-post-commit-recapture-discipline``, the canonical
maintenance pattern for sweep-build-commit drift is: rewrite the
target field to current ``git rev-parse HEAD`` while preserving
the prior value in a sibling audit-annotation field
(``<field>_override_from``) so grep / verifier output continues to
expose the drift trail. The rewrite is safe because
``predictor_build_commit`` is a producer-HEAD-derived provenance
annotation, not a content-derived hash; the underlying scalar /
typed-unavailable bundle content was emitted by the producer
identified by the prior commit and remains byte-identical.

CLI:

    python3 tools/local_labels/refresh_sweep_build_commit.py \
        --paths path/to/file1.json,path/to/file2.json \
        --field predictor_build_commit

The ``--field`` flag accepts the exact field name to refresh
(``predictor_build_commit`` for ``sweep_*/predictor_bundles/*.json``;
``regeneration_build_commit`` for ``sweep_*/regeneration_delta.json``).

Idempotency: re-running the script at the same HEAD produces
byte-identical output. ``<field>_override_from`` records the
value PRIOR to the rewrite; a no-op rewrite (when the target
already equals HEAD) preserves the existing override-from value
unchanged so it continues to point at the original drift origin.

Exit codes:
  0 -- every named file was refreshed (or already at HEAD).
  2 -- one or more files failed validation (missing path, missing
       field, malformed JSON).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_HEX_SHA_RE = re.compile(r"[0-9a-f]{40}")


def _git_head_sha(repo_root: Path) -> str:
    """Return ``git rev-parse HEAD`` in ``repo_root`` as a 40-hex SHA."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def refresh_file_in_place(
    *,
    path: Path,
    field: str,
    new_commit: str,
) -> tuple[bool, str]:
    """Rewrite ``path``'s ``field`` to ``new_commit``; preserve the
    prior value under ``<field>_override_from`` (unless it already
    equals ``new_commit``, in which case the file is byte-identical
    after the no-op).

    Returns ``(rewrote, message)`` where ``rewrote`` is True when
    the file changed and ``message`` is a short status string for
    the CLI output.
    """
    if not path.is_file():
        return False, f"absent: {path}"
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError as exc:
        return False, f"malformed-json: {path}: {exc}"
    if field not in payload:
        return False, f"missing-field {field!r}: {path}"
    prior = payload[field]
    if prior == new_commit:
        return False, f"no-op (already at HEAD): {path}"
    if not isinstance(prior, str) or not _HEX_SHA_RE.fullmatch(prior):
        return False, f"prior {field}={prior!r} is not a 40-hex SHA: {path}"
    override_key = f"{field}_override_from"
    # Idempotency: when override_from already exists, do NOT overwrite —
    # keep it pointing at the ORIGINAL drift origin even across
    # multiple refresh runs.
    if override_key not in payload:
        payload[override_key] = prior
    payload[field] = new_commit
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return True, f"refreshed {field}: {prior[:12]} -> {new_commit[:12]}: {path}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths", required=True,
        help="Comma-separated list of file paths (relative to "
             "--repo-root or absolute) to refresh.",
    )
    parser.add_argument(
        "--field", required=True,
        help="Field name to refresh (e.g. predictor_build_commit, "
             "regeneration_build_commit).",
    )
    parser.add_argument(
        "--repo-root", default=str(_REPO_ROOT),
        help="Repository root (defaults to derived path).",
    )
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    new_commit = _git_head_sha(repo_root)
    exit_code = 0
    for raw_path in args.paths.split(","):
        raw_path = raw_path.strip()
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        _rewrote, message = refresh_file_in_place(
            path=path, field=args.field, new_commit=new_commit,
        )
        print(f"[refresh_sweep_build_commit] {message}")
        if (
            "missing-field" in message
            or "is not a 40-hex SHA" in message
            or "absent:" in message
            or "malformed-json:" in message
        ):
            exit_code = 2
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
