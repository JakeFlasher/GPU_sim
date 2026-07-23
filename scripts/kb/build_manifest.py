#!/usr/bin/env python3
"""
build_manifest.py — extract knowledge-base card metadata into a SQLite FTS5 index.

Reads every `.claude/knowledge/**/*.md` card, parses its YAML-ish frontmatter
(seven required fields — `Use when`, `Primary raw source`, `Supporting sources`,
`Repo touchpoints`, `Patents anchored here`, `Out of scope`, `Version family` —
plus optional ownership, routing-tag, and latency-authority fields) and all
markdown headings (`##` / `###`). It writes the canonical v2 artifacts:

  .claude/kb_manifest.sqlite     — FTS5 + metadata table (primary)
  .claude/kb_manifest.json       — parallel JSON dump for non-SQLite consumers

Both files are gitignored regenerable artifacts; rebuild after any KB edit
via:  python3 scripts/kb/build_manifest.py

Usage: python3 scripts/kb/build_manifest.py [--root REPO_ROOT] [--quiet]

Exit codes:
  0 — success
  1 — script error (no manifest written)
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Optional


# The SQLite/FTS layout is part of the router contract.  Bump only when the
# generated schema changes; readers accept the previous layout explicitly and
# reject unknown versions rather than silently scoring a different table.
MANIFEST_SCHEMA_VERSION = 2


# Frontmatter fields we recognize. Order matters for output stability.
# Codex-review addition: optional ownership-metadata fields. When present
# they let the card declare which Skill owns it (vs the subcorpus
# default) and which other Skills also legitimately reach it. The router
# reads these fields to resolve split-personality cards without needing
# Python-side CARD_OVERRIDE_RULES for the simple cases.
FRONTMATTER_FIELDS = [
    "Use when",
    "Primary raw source",
    "Supporting sources",
    "Repo touchpoints",
    "Patents anchored here",
    "Out of scope",
    "Version family",
    "Default skill",       # optional — one canonical skill name
    "Also relevant skills",  # optional — comma/semicolon list
    "Routing tags",        # optional — comma/semicolon list of free-form tags
    "Latency authority",   # optional — paper|leaderboard|local-checkout
]


def _split_frontmatter(text: str) -> tuple[Optional[dict], str]:
    """Parse `---\\n...\\n---\\n` YAML-ish frontmatter at top of card.

    Returns (fields_dict_or_None, body_after_frontmatter). Fields with
    multi-line values are joined intelligently:
      - Continuation lines starting with `- ` or `* ` (bulleted list) are
        treated as separate items and joined with `; `.
      - Plain wrapped-prose continuation lines are joined with a space.
    We do NOT use a YAML parser because the cards use prose values that
    contain colons, which would upset strict YAML.

    phase 2-rev fix: previously joined bulleted lists with bare space,
    producing strings like `'- foo.md - bar.md'` in `Supporting sources`
    of 18 cards.
    """
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return None, text
    raw = text[4:end]
    body = text[end + 5 :]

    fields: dict[str, str] = {}
    current_key: Optional[str] = None
    for line in raw.splitlines():
        m = re.match(r"^([A-Za-z][A-Za-z ]+):\s*(.*)$", line)
        if m and m.group(1) in FRONTMATTER_FIELDS:
            current_key = m.group(1)
            fields[current_key] = m.group(2).strip()
        elif current_key is not None:
            cont = line.strip()
            if not cont:
                continue
            # Bulleted continuation → strip marker, join with semicolon.
            bullet = re.match(r"^[-*]\s+(.*)$", cont)
            if bullet:
                item = bullet.group(1).strip()
                cur = fields[current_key]
                if cur:
                    fields[current_key] = f"{cur}; {item}"
                else:
                    fields[current_key] = item
            else:
                # Plain wrapped prose continuation.
                fields[current_key] = (fields[current_key] + " " + cont).strip()
    return fields, body


# phase 2-rev fix: scan only ## and deeper headings; previously the regex
# captured `# Title` too, polluting the FTS5 `headings` column with the
# title text on 65 of 68 cards.
_HEADING_RE = re.compile(r"^(#{2,6})\s+(.+?)\s*$", re.MULTILINE)


def _extract_headings(body: str, max_headings: int = 50) -> list[str]:
    out = []
    for m in _HEADING_RE.finditer(body):
        text = m.group(2).strip()
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        out.append(text)
        if len(out) >= max_headings:
            break
    return out


def _extract_referenced_paths(text: str) -> list[str]:
    """Pull out repo-relative paths the card cites.

    Looks for `path/to/file.md` or `path/to/file.cu` style backticked refs
    and plain `references/...` / `.claude/knowledge/...` / `plans/...` /
    `workloads/...` / `tools/...` / `scripts/...` / `simulator/...` /
    `tests/...` / `configs/...` / `results/...` mentions in prose.

    phase 2-rev fix: previously called on `body` only — frontmatter-cited
    paths (`Primary raw source`, `Supporting sources`) were missed.
    Now called with full text so frontmatter paths are included.
    """
    paths: set[str] = set()
    for m in re.finditer(
        r"`([\w./_-]+\.(?:md|cu|cuh|h|cpp|hpp|c|py|sh|toml|json|ptx|sass|"
        r"pdf|rs|txt))`",
        text,
    ):
        paths.add(m.group(1))
    for m in re.finditer(
        r"\b((?:references|\.claude/knowledge|\.claude/skills|\.claude/hooks|"
        r"plans|workloads|tools|scripts|simulator|tests|configs|results)/"
        r"[\w./_-]+\.\w{1,5})\b",
        text,
    ):
        paths.add(m.group(1))
    return sorted(paths)


def _scan_cards(knowledge_root: Path) -> list[dict]:
    rows = []
    for md_path in sorted(knowledge_root.rglob("*.md")):
        # Templates and scaffolding are authoring inputs, not routable cards.
        # Keep the manifest and linter's corpus boundary identical.
        if md_path.name.startswith("_"):
            continue
        relpath = md_path.relative_to(knowledge_root.parent.parent)
        # phase 2-rev fix: previously only flagged `is_index` for the root
        # INDEX.md; sub-corpus INDEX.md files (e.g., patents/INDEX.md) were
        # mis-classified as content cards.
        is_index = md_path.name == "INDEX.md"
        rows.append(_card_row(md_path, relpath, is_index=is_index))
    return rows


def _card_row(md_path: Path, relpath: Path, *, is_index: bool) -> dict:
    text = md_path.read_text(encoding="utf-8")
    fm, body = _split_frontmatter(text)
    title_match = re.search(r"^#\s+(.+?)\s*$", body, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else md_path.stem
    title = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", title)

    parts = relpath.parts
    if len(parts) >= 3 and parts[0] == ".claude" and parts[1] == "knowledge":
        subcorpus = parts[2] if len(parts) >= 4 else "_root"
    else:
        subcorpus = "_root"

    headings = _extract_headings(body)
    # phase 2-rev fix: extract paths from the full card text (frontmatter
    # + body), not just the body. Frontmatter `Primary raw source:` and
    # `Supporting sources:` paths were previously missed.
    referenced = _extract_referenced_paths(text)

    line_count = text.count("\n") + 1
    return {
        "path": str(relpath).replace("\\", "/"),
        "name": md_path.stem,
        "subcorpus": subcorpus,
        "is_index": is_index,
        "title": title,
        "line_count": line_count,
        "frontmatter": fm or {},
        "headings": headings,
        "referenced_paths": referenced,
    }


def _build_sqlite(rows: list[dict], db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE cards (
            path TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            subcorpus TEXT NOT NULL,
            is_index INTEGER NOT NULL,
            title TEXT,
            line_count INTEGER,
            use_when TEXT,
            primary_raw_source TEXT,
            supporting_sources TEXT,
            repo_touchpoints TEXT,
            patents_anchored_here TEXT,
            out_of_scope TEXT,
            version_family TEXT,
            default_skill TEXT,
            also_relevant_skills TEXT,
            routing_tags TEXT,
            latency_authority TEXT,
            headings TEXT,
            referenced_paths TEXT
        );
        CREATE TABLE manifest_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE cards_fts USING fts5(
            path UNINDEXED,
            name,
            title,
            use_when,
            primary_raw_source,
            supporting_sources,
            headings,
            routing_tags,
            tokenize = "unicode61 remove_diacritics 1"
        );
        """
    )
    conn.execute(
        "INSERT INTO manifest_meta(key, value) VALUES (?, ?)",
        ("schema_version", str(MANIFEST_SCHEMA_VERSION)),
    )
    conn.execute(
        "INSERT INTO manifest_meta(key, value) VALUES (?, ?)",
        (
            "fts_columns",
            "path,name,title,use_when,primary_raw_source,supporting_sources,headings,routing_tags",
        ),
    )
    for r in rows:
        fm = r["frontmatter"]
        conn.execute(
            """INSERT INTO cards (
                path, name, subcorpus, is_index, title, line_count,
                use_when, primary_raw_source, supporting_sources,
                repo_touchpoints, patents_anchored_here, out_of_scope,
                version_family, default_skill, also_relevant_skills,
                routing_tags, latency_authority, headings, referenced_paths
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                r["path"],
                r["name"],
                r["subcorpus"],
                int(r["is_index"]),
                r["title"],
                r["line_count"],
                fm.get("Use when", ""),
                fm.get("Primary raw source", ""),
                fm.get("Supporting sources", ""),
                fm.get("Repo touchpoints", ""),
                fm.get("Patents anchored here", ""),
                fm.get("Out of scope", ""),
                fm.get("Version family", ""),
                fm.get("Default skill", ""),
                fm.get("Also relevant skills", ""),
                fm.get("Routing tags", ""),
                fm.get("Latency authority", ""),
                "\n".join(r["headings"]),
                "\n".join(r["referenced_paths"]),
            ),
        )
        conn.execute(
            """INSERT INTO cards_fts (
                path, name, title, use_when, primary_raw_source,
                supporting_sources, headings, routing_tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                r["path"],
                r["name"],
                r["title"],
                fm.get("Use when", ""),
                fm.get("Primary raw source", ""),
                fm.get("Supporting sources", ""),
                " ".join(r["headings"]),
                fm.get("Routing tags", ""),
            ),
        )
    conn.commit()
    conn.close()


def _build_json(rows: list[dict], json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repo root (default: cwd)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.root).resolve()
    knowledge_root = repo_root / ".claude" / "knowledge"
    if not knowledge_root.is_dir():
        print(f"error: {knowledge_root} not found", file=sys.stderr)
        return 1

    rows = _scan_cards(knowledge_root)
    db_path = repo_root / ".claude" / "kb_manifest.sqlite"
    json_path = repo_root / ".claude" / "kb_manifest.json"
    _build_sqlite(rows, db_path)
    _build_json(rows, json_path)

    if not args.quiet:
        n_with_fm = sum(1 for r in rows if r["frontmatter"])
        n_total = len(rows)
        by_subcorpus: dict[str, int] = {}
        for r in rows:
            by_subcorpus[r["subcorpus"]] = by_subcorpus.get(r["subcorpus"], 0) + 1
        print(f"Wrote {db_path.relative_to(repo_root)} ({db_path.stat().st_size} bytes)")
        print(f"Wrote {json_path.relative_to(repo_root)} ({json_path.stat().st_size} bytes)")
        print(f"Cards: {n_total}  with-frontmatter: {n_with_fm}  missing-frontmatter: {n_total - n_with_fm}")
        for s, n in sorted(by_subcorpus.items()):
            print(f"  {s:24s} {n:3d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
