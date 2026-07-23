from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys

import pytest


_SOL_KB_ROOT = Path(__file__).resolve().parent
if str(_SOL_KB_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOL_KB_ROOT))

import build_manifest  # noqa: E402
import lint_cards  # noqa: E402


_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXPECTED_SOL_LINT_COUNTS = {
    "cutlass_constructs": 6,
    "patterns": 27,
    "problems": 235,
    "sources": 7,
}
_EXPECTED_SOL_LINT_TOTAL = 275
_EXPECTED_MANIFEST_ROWS = 355


def test_future_baseline_exemption_is_submission_local() -> None:
    prefix = (
        "workloads/sol_execbench_baselines/FIB/"
        "problem_001_fused_add_rmsnorm_h2048/"
    )

    assert lint_cards._is_documented_future_baseline_path(prefix + "cutlass/")
    assert lint_cards._is_documented_future_baseline_path(
        prefix + "cuda_cpp/kernel.cu"
    )
    assert not lint_cards._is_documented_future_baseline_path(
        prefix + "third_party/cutlass/"
    )
    assert not lint_cards._is_documented_future_baseline_path(
        prefix + "unsupported_language/"
    )


def _read_sqlite_table(
    database: Path, table: str
) -> tuple[tuple[str, ...], list[tuple[object, ...]]]:
    uri = f"{database.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        columns = tuple(
            row[1]
            for row in connection.execute(f"PRAGMA table_info({table})")
        )
        rows = list(connection.execute(f"SELECT * FROM {table} ORDER BY path"))
    return columns, rows


def test_canonical_sol_kb_lint_and_manifest_regression(tmp_path: Path) -> None:
    knowledge_root = _REPO_ROOT / ".claude" / "knowledge"
    template = knowledge_root / "_template.md"
    assert template.is_file()

    scoped_cards = lint_cards._load_cards(knowledge_root, _REPO_ROOT)
    scoped_counts = Counter(card.subcorpus for card in scoped_cards)
    assert dict(sorted(scoped_counts.items())) == _EXPECTED_SOL_LINT_COUNTS
    assert len(scoped_cards) == _EXPECTED_SOL_LINT_TOTAL
    assert all(not card.path.name.startswith("_") for card in scoped_cards)

    lint_errors: list[str] = []
    lint_cards._run_rules(scoped_cards, "all", lint_errors, _REPO_ROOT)
    assert lint_errors == []

    # Manifest generation is intentionally after the hard lint assertion.
    # Build only into pytest's temporary directory; repository artifacts are
    # opened read-only below and must remain byte-for-byte untouched.
    expected_rows = build_manifest._scan_cards(knowledge_root)
    assert len(expected_rows) == _EXPECTED_MANIFEST_ROWS
    assert all(
        not Path(row["path"]).name.startswith("_") for row in expected_rows
    )
    assert ".claude/knowledge/_template.md" not in {
        row["path"] for row in expected_rows
    }

    generated_json = tmp_path / "kb_manifest.json"
    generated_sqlite = tmp_path / "kb_manifest.sqlite"
    build_manifest._build_json(expected_rows, generated_json)
    build_manifest._build_sqlite(expected_rows, generated_sqlite)

    repository_json = _REPO_ROOT / ".claude" / "kb_manifest.json"
    repository_sqlite = _REPO_ROOT / ".claude" / "kb_manifest.sqlite"
    assert repository_json.is_file()
    assert repository_sqlite.is_file()
    assert repository_json.read_bytes() == generated_json.read_bytes()

    json_rows = json.loads(repository_json.read_text(encoding="utf-8"))
    assert json_rows == expected_rows
    assert len(json_rows) == _EXPECTED_MANIFEST_ROWS
    json_paths = sorted(row["path"] for row in json_rows)

    for table in ("cards", "cards_fts"):
        generated_columns, generated_rows = _read_sqlite_table(
            generated_sqlite, table
        )
        repository_columns, repository_rows = _read_sqlite_table(
            repository_sqlite, table
        )
        assert repository_columns == generated_columns
        assert repository_rows == generated_rows
        assert len(repository_rows) == _EXPECTED_MANIFEST_ROWS
        assert [row[0] for row in repository_rows] == json_paths


def test_manifest_lint_failure_precedes_scan_and_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repo"
    (repository / ".claude" / "knowledge").mkdir(parents=True)
    stale_json = repository / ".claude" / "kb_manifest.json"
    stale_sqlite = repository / ".claude" / "kb_manifest.sqlite"
    stale_json.write_bytes(b"stale json\n")
    stale_sqlite.write_bytes(b"stale sqlite\n")
    calls: list[str] = []

    def fail_lint(repo_root: Path, quiet: bool) -> int:
        assert repo_root == repository.resolve()
        assert quiet
        calls.append("lint")
        return 1

    def forbidden_scan(_knowledge_root: Path) -> list[dict]:
        calls.append("scan")
        raise AssertionError("manifest scan ran after lint failure")

    monkeypatch.setattr(build_manifest, "_run_lint_precondition", fail_lint)
    monkeypatch.setattr(build_manifest, "_scan_cards", forbidden_scan)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_manifest.py",
            "--root",
            str(repository),
            "--quiet",
        ],
    )

    assert build_manifest.main() == 1
    assert calls == ["lint"]
    assert not stale_json.exists()
    assert not stale_sqlite.exists()
