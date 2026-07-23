"""Keep legacy SOL-KB router copies byte-identical to the wired router."""

from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
ROUTER_FILES = (
    "build_manifest.py",
    "check_kb_usage.py",
    "_relevance.py",
    "_relevance_evals.yaml",
    "route_query.py",
    "run_evals.py",
    "run_relevance_evals.py",
)


@pytest.mark.parametrize("name", ROUTER_FILES)
def test_sol_kb_router_copy_matches_canonical_wired_router(name: str) -> None:
    canonical = REPO_ROOT / "scripts" / "kb" / name
    compatibility = REPO_ROOT / "tools" / "sol_kb" / name
    assert compatibility.read_bytes() == canonical.read_bytes()


def test_generated_fallback_gates_match_canonical_classifier() -> None:
    generated = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "kb" / "_relevance.py"),
            "--emit-bash",
        ],
        check=True,
        capture_output=True,
    ).stdout
    for name in ("_gate_regexes.sh", "sol_gate_regexes.sh"):
        assert (REPO_ROOT / ".claude" / "hooks" / name).read_bytes() == generated
