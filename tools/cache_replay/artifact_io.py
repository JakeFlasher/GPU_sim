"""Shared artifact-IO helpers: atomic JSON write + git HEAD provenance.

Mirrors the repo idioms: every emitted artifact stamps ``source_commit`` (the
git HEAD, resolved through gitdir indirection so it works in worktrees) and is
written atomically (temp file then replace) so concurrent readers never observe
a half-written artifact.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent.parent
_SCHEMA_DIR = _HERE / "schemas"


def validate_against_schema(payload: Any, schema_filename: str) -> None:
    """Validate ``payload`` against a closed draft-2020-12 schema under
    ``tools/cache_replay/schemas/``. Raises on any validation error. Called by
    every gate-artifact emitter BEFORE the artifact is written."""
    import jsonschema
    schema = json.loads((_SCHEMA_DIR / schema_filename).read_text())
    jsonschema.Draft202012Validator(schema).validate(payload)


def git_head(repo_root: Path = REPO_ROOT) -> str:
    """Resolve the current git HEAD commit SHA (gitdir-indirection safe)."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp.replace(path)
