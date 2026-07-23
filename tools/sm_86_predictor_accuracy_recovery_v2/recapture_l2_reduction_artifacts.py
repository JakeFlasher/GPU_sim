"""Post-commit recapture for the L2 reduction-kernel-family
correction's tracked artifacts under
``data/sm_86_predictor_accuracy_recovery_v2/`` per
BL-20260508.

Discipline (BL-20260508):

  1. Make the predictor-touching code edits.
  2. COMMIT the edits (fix-commit F).
  3. Run this script: read ``git rev-parse HEAD`` (= F) and
     stamp F onto each tracked artifact's ``source_commit``
     field.
  4. Commit the regenerated artifacts in a SEPARATE
     chore-recapture commit C. The artifacts' ``source_commit``
     references F (the predictor-touching fix-commit), NOT C
     (the chore-recapture commit cannot reference its own
     SHA).

Tracked artifacts:

  - data/sm_86_predictor_accuracy_recovery_v2/l2_reduction_correction_calibration.json
  - data/sm_86_predictor_accuracy_recovery_v2/l2_reduction_correction_shadow_report.json

Usage:

  python3 tools/sm_86_predictor_accuracy_recovery_v2/recapture_l2_reduction_artifacts.py

Optional kwarg ``--source-commit <sha>`` overrides the
discovered HEAD (useful for testing or for the chore-recapture
flow when the operator needs to pin a specific fix-commit).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRACKED_ARTIFACTS: tuple[Path, ...] = (
    _REPO_ROOT
    / "data" / "sm_86_predictor_accuracy_recovery_v2"
    / "l2_reduction_correction_calibration.json",
    _REPO_ROOT
    / "data" / "sm_86_predictor_accuracy_recovery_v2"
    / "l2_reduction_correction_shadow_report.json",
)


def _git_head(repo_root: Path) -> str:
    """Read ``git rev-parse HEAD`` and return the full SHA.
    Signature mirrors `tools/local_labels/refresh_*.py::_git_head`
    so a future shared helper can absorb all three call sites."""
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _validate_sha(sha: str) -> str:
    """Verify a string is a 40-character lowercase hex SHA."""
    if (
        len(sha) != 40
        or not all(c in "0123456789abcdef" for c in sha)
    ):
        raise ValueError(
            f"invalid commit SHA {sha!r}: expected 40-char hex"
        )
    return sha


def stamp_artifacts(target_sha: str) -> list[Path]:
    """Stamp ``target_sha`` as ``source_commit`` on each
    tracked artifact. Returns the list of paths that were
    modified."""
    _validate_sha(target_sha)
    modified: list[Path] = []
    for artifact_path in _TRACKED_ARTIFACTS:
        if not artifact_path.exists():
            print(
                f"warning: artifact missing at {artifact_path}; "
                "skipping",
                file=sys.stderr,
            )
            continue
        with artifact_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("source_commit") == target_sha:
            continue
        data["source_commit"] = target_sha
        with artifact_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
        modified.append(artifact_path)
    return modified


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stamp the predictor-touching fix-commit SHA "
            "onto the L2 reduction-correction tracked "
            "artifacts per BL-20260508."
        ),
    )
    parser.add_argument(
        "--source-commit",
        default=None,
        help=(
            "Override the discovered HEAD with a specific SHA. "
            "When omitted, reads `git rev-parse HEAD`."
        ),
    )
    args = parser.parse_args(argv)
    target_sha = (
        args.source_commit
        if args.source_commit is not None
        else _git_head(_REPO_ROOT)
    )
    target_sha = _validate_sha(target_sha)
    modified = stamp_artifacts(target_sha)
    print(
        f"Stamped source_commit={target_sha} onto "
        f"{len(modified)} artifact(s):"
    )
    for p in modified:
        print(f"  {p.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
