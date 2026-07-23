"""Stage a Pydantic-compatible copy of a benchmark directory.

The upstream ``sol-execbench`` evaluator constructs ``Definition`` from
``definition.json`` via plain ``Definition(**dict)``. The downloaded
``sol-execbench/data/benchmark/*/*/definition.json`` files set
``"hf_id": ""`` (and similarly for ``op_type`` /
``custom_inputs_entrypoint``) for problems without a HuggingFace model
lineage, but ``sol_execbench/core/data/definition.py`` declares those
fields as ``Optional[NonEmptyString]`` — empty string is rejected with
``String should have at least 1 character``.

This module shims the divergence by copying the canonical benchmark
directory into a project-local staging area, normalising the affected
fields to ``None``, and returning the staged path so the harness can
run the evaluator against the normalised copy.

Conventions:

- Source of truth for the canonical benchmark path remains
  ``sol-execbench/data/benchmark/<subset>/<NNN_name>/``. Parent
  ``README.md`` files cite this path, NOT the staged one (architecture-
  portability discipline; staging is harness-internal).
- Staging root defaults to
  ``<repo_root>/.tmp_staged_benchmarks/<subset>/<NNN_name>/`` and is
  gitignored.
- Each staged directory is byte-identical to the canonical one except
  for the normalised optional fields in ``definition.json``. The
  ``workload.jsonl`` and ``reference.py`` files are copied verbatim.
- The stager is idempotent: re-running with the same source produces
  the same output.

Reference for the bug: ``sol-execbench/src/sol_execbench/cli/main.py``
calls ``Definition(**json.loads(path.read_text()))`` without a
normaliser; the downloaded data is incompatible.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent
REPO_ROOT = _SCRIPTS_DIR.parent
DEFAULT_STAGING_ROOT = REPO_ROOT / ".tmp_staged_benchmarks"

# Definition fields the upstream Pydantic model declares as
# ``Optional[NonEmptyString]`` (or similar) but the downloaded JSON
# carries as ``""`` for problems without that lineage. Adding to this
# list is cheap; missing one would surface as a fresh upstream error
# the next time a pilot is run.
_OPTIONAL_NON_EMPTY_FIELDS: Tuple[str, ...] = (
    "hf_id",
    "op_type",
    "custom_inputs_entrypoint",
)


def normalise_definition_dict(defn: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``defn`` with empty optional strings → ``None``.

    Does not mutate the input. Fields the upstream schema treats as
    ``Optional[NonEmptyString]`` get their empty-string value coerced
    to ``None`` so ``Definition(**result)`` succeeds.
    """
    out = dict(defn)
    for key in _OPTIONAL_NON_EMPTY_FIELDS:
        if out.get(key) == "":
            out[key] = None
    return out


def stage_benchmark(
    canonical_dir: Path,
    staging_root: Path = DEFAULT_STAGING_ROOT,
) -> Path:
    """Stage a normalised copy of ``canonical_dir``; return the staged path.

    Layout:
        ``<staging_root>/<subset>/<NNN_name>/{definition.json, workload.jsonl, reference.py}``

    The staged directory is removed and re-created on every call (so
    the staging is deterministic regardless of prior state). Subprocess
    failures from corrupt JSON propagate as ``ValueError`` /
    ``OSError``; callers may surface them as actionable errors.
    """
    if not canonical_dir.is_dir():
        raise FileNotFoundError(
            f"Canonical benchmark directory not found at {canonical_dir}. "
            f"Run ./sol-execbench/scripts/download_data.sh to populate the "
            f"benchmark tree."
        )
    canonical_dir = canonical_dir.resolve()
    # subset name is the parent dir name (e.g. "FlashInfer-Bench").
    subset = canonical_dir.parent.name
    bench_name = canonical_dir.name
    staged_dir = staging_root / subset / bench_name
    if staged_dir.exists():
        shutil.rmtree(staged_dir)
    staged_dir.mkdir(parents=True)

    # Read + normalise + write definition.json.
    src_def = canonical_dir / "definition.json"
    if not src_def.is_file():
        raise FileNotFoundError(
            f"definition.json missing under {canonical_dir}; cannot stage."
        )
    defn = json.loads(src_def.read_text(encoding="utf-8"))
    defn = normalise_definition_dict(defn)
    (staged_dir / "definition.json").write_text(
        json.dumps(defn, indent=2) + "\n",
        encoding="utf-8",
    )

    # Copy workload.jsonl and reference.py verbatim — those are not
    # touched by the upstream Pydantic divergence.
    for fname in ("workload.jsonl", "reference.py"):
        src = canonical_dir / fname
        if src.is_file():
            shutil.copyfile(src, staged_dir / fname)

    return staged_dir


def needs_staging(canonical_dir: Path) -> bool:
    """Return True when ``canonical_dir/definition.json`` has empty optional fields.

    Used by the harness to skip the staging copy when the canonical
    definition is already Pydantic-compatible — avoids unnecessary
    disk I/O on problems whose downloaded data happens to have
    ``hf_id`` populated.
    """
    src = canonical_dir / "definition.json"
    if not src.is_file():
        return False
    try:
        defn = json.loads(src.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return any(
        defn.get(key) == "" for key in _OPTIONAL_NON_EMPTY_FIELDS
    )
