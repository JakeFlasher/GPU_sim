from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys

import pytest


_SOL_KB_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_KB_ROOT = _REPO_ROOT / "scripts" / "kb"
for import_root in (_SOL_KB_ROOT, _CANONICAL_KB_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import build_manifest  # noqa: E402
import lint_cards  # noqa: E402


_EXPECTED_SOL_LINT_COUNTS = {
    "cutlass_constructs": 6,
    "patterns": 27,
    "problems": 235,
    "sources": 16,
}
_EXPECTED_SOL_LINT_TOTAL = 284
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

    # Manifest generation is intentionally after the hard lint assertion and
    # stays entirely in pytest's temporary directory.
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

    json_rows = json.loads(generated_json.read_text(encoding="utf-8"))
    assert json_rows == expected_rows
    assert len(json_rows) == _EXPECTED_MANIFEST_ROWS
    json_paths = sorted(row["path"] for row in json_rows)

    for table in ("cards", "cards_fts"):
        columns, rows = _read_sqlite_table(generated_sqlite, table)
        assert columns
        assert len(rows) == _EXPECTED_MANIFEST_ROWS
        assert [row[0] for row in rows] == json_paths


def test_manifest_cli_replaces_only_its_generated_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repo"
    knowledge = repository / ".claude" / "knowledge"
    knowledge.mkdir(parents=True)
    (knowledge / "INDEX.md").write_text(
        "# Temporary knowledge index\n", encoding="utf-8"
    )
    stale_json = repository / ".claude" / "kb_manifest.json"
    stale_sqlite = repository / ".claude" / "kb_manifest.sqlite"
    stale_json.write_bytes(b"stale json\n")
    stale_sqlite.write_bytes(b"stale sqlite\n")
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

    assert build_manifest.main() == 0
    assert json.loads(stale_json.read_text(encoding="utf-8"))[0]["path"] == (
        ".claude/knowledge/INDEX.md"
    )
    columns, rows = _read_sqlite_table(stale_sqlite, "cards")
    assert columns
    assert len(rows) == 1
