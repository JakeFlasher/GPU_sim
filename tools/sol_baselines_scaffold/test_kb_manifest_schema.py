"""Compatibility and FTS-shape tests for the canonical KB manifest router."""

from __future__ import annotations

import importlib.util
import re
import sqlite3
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
KB = ROOT / "scripts/kb"
pytestmark = pytest.mark.skipif(
    not all(
        path.is_file()
        for path in (KB / "route_query.py", KB / "build_manifest.py")
    ),
    reason="optional KB manifest sources are unavailable",
)


def _load_router():
    if str(KB) not in sys.path:
        sys.path.insert(0, str(KB))
    spec = importlib.util.spec_from_file_location("canonical_route_query", KB / "route_query.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "canonical_build_manifest", KB / "build_manifest.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _connection(columns: tuple[str, ...], version: int | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE VIRTUAL TABLE cards_fts USING fts5(" + ", ".join(columns) + ")"
    )
    if version is not None:
        conn.execute("CREATE TABLE manifest_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute(
            "INSERT INTO manifest_meta(key, value) VALUES ('schema_version', ?)",
            (str(version),),
        )
    return conn


def test_legacy_v1_shape_without_metadata_is_accepted() -> None:
    router = _load_router()
    columns = tuple(router._REQUIRED_FTS_COLUMNS_V2[:-1])
    conn = _connection(columns)
    try:
        assert router._manifest_bm25_weights(conn) == "0.0, 4.0, 3.0, 6.0, 1.5, 0.5, 1.0"
    finally:
        conn.close()


def test_legacy_tag_aware_shape_without_metadata_is_accepted() -> None:
    router = _load_router()
    columns = tuple(router._REQUIRED_FTS_COLUMNS_V2)
    conn = _connection(columns)
    try:
        assert router._manifest_bm25_weights(conn) == router._BM25_WEIGHTS
    finally:
        conn.close()


def test_explicit_schema_version_must_match_shape() -> None:
    router = _load_router()
    v1 = _connection(tuple(router._REQUIRED_FTS_COLUMNS_V2[:-1]), version=1)
    v2 = _connection(tuple(router._REQUIRED_FTS_COLUMNS_V2), version=2)
    mismatch = _connection(tuple(router._REQUIRED_FTS_COLUMNS_V2), version=1)
    unknown = _connection(tuple(router._REQUIRED_FTS_COLUMNS_V2), version=99)
    try:
        assert router._manifest_bm25_weights(v1) is not None
        assert router._manifest_bm25_weights(v2) == router._BM25_WEIGHTS
        assert router._manifest_bm25_weights(mismatch) is None
        assert router._manifest_bm25_weights(unknown) is None
    finally:
        for conn in (v1, v2, mismatch, unknown):
            conn.close()


def test_builder_emits_independent_v2_fts_ddl(tmp_path: Path) -> None:
    """Pin the generated FTS shape independently of router constants."""
    builder = _load_builder()
    rows = [
        {
            "path": ".claude/knowledge/patterns/example.md",
            "name": "example",
            "subcorpus": "patterns",
            "is_index": False,
            "title": "Example",
            "line_count": 3,
            "frontmatter": {"Routing tags": "gemm; cutlass"},
            "headings": ["Use when"],
            "referenced_paths": [],
        }
    ]
    manifest = tmp_path / "manifest.sqlite"
    builder._build_sqlite(rows, manifest)

    expected_columns = (
        "path",
        "name",
        "title",
        "use_when",
        "primary_raw_source",
        "supporting_sources",
        "headings",
        "routing_tags",
    )
    with sqlite3.connect(manifest) as conn:
        columns = tuple(row[1] for row in conn.execute("PRAGMA table_info(cards_fts)"))
        assert columns == expected_columns
        ddl = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'cards_fts'"
        ).fetchone()[0]
        assert re.search(r"\bpath\s+UNINDEXED\b", ddl, re.IGNORECASE)
        assert re.search(
            r"tokenize\s*=\s*['\"]unicode61\s+remove_diacritics\s+1['\"]",
            ddl,
            re.IGNORECASE,
        )
        assert conn.execute(
            "SELECT value FROM manifest_meta WHERE key = 'schema_version'"
        ).fetchone() == ("2",)
        assert conn.execute(
            "SELECT routing_tags FROM cards_fts WHERE path = ?",
            (rows[0]["path"],),
        ).fetchone() == ("gemm; cutlass",)


def test_routing_tags_are_searchable_in_fts(tmp_path: Path) -> None:
    router = _load_router()
    manifest = tmp_path / "manifest.sqlite"
    conn = sqlite3.connect(manifest)
    columns = tuple(router._REQUIRED_FTS_COLUMNS_V2)
    conn.execute("CREATE TABLE cards (path TEXT PRIMARY KEY, default_skill TEXT)")
    conn.execute(
        "CREATE VIRTUAL TABLE cards_fts USING fts5(" + ", ".join(columns) + ")"
    )
    path = ".claude/knowledge/cutlass_constructs/block_scaled_mma.md"
    conn.execute("INSERT INTO cards(path, default_skill) VALUES (?, ?)", (path, ""))
    values = (path, "block_scaled_mma", "Block scaled MMA", "CUTLASS block scale", "", "", "", "cutlass block-scale tcgen05")
    conn.execute(
        "INSERT INTO cards_fts(path, name, title, use_when, primary_raw_source, supporting_sources, headings, routing_tags) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        values,
    )
    conn.commit()
    conn.close()

    rows = router._fts_query("CUTLASS block-scale tcgen05", manifest, 5)
    assert rows
    assert rows[0]["path"] == path
