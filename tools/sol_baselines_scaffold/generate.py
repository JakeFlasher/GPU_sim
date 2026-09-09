#!/usr/bin/env python3
"""Manifest + slug-resolution primitives for the scaffold toolchain.

This module provides:

- ``parse_mapping`` / ``MappingRow`` for the tracked
  ``tools/sol_baselines_scaffold/sol_pattern_mapping.md`` contract.
- ``derive_benchmark_name`` for slug → ``NNN_<rest>`` resolution
  (strips ``^problem_``).
- ``resolve_benchmark_dir`` / ``load_definition`` for benchmark lookup.
- ``SUBSET_MAP`` (internal category → downloader subset).
- ``SUPPORTED_LANGUAGES`` and other module-level constants the
  downstream scaffolder / validator import.
- A ``--manifest-only`` CLI that prints a JSON inventory of every
  mapping row's readiness (dossier / spec / pattern card / benchmark
  ``definition.json`` presence).

Usage:
    python tools/sol_baselines_scaffold/generate.py --manifest-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Bootstrap so ``baselines_scaffold.*`` is importable when this file is run
# directly (e.g. ``python tools/sol_baselines_scaffold/generate.py ...``).
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# --- Repo-anchored paths --------------------------------------------------

REPO_ROOT = _SCRIPTS_DIR.parent
MAPPING_PATH = _THIS_DIR / "sol_pattern_mapping.md"
DOSSIER_DIR = REPO_ROOT / ".claude" / "knowledge" / "problems"
PATTERNS_DIR = REPO_ROOT / ".claude" / "knowledge" / "patterns"
SPEC_DIR = REPO_ROOT / "references" / "NVIDIA_Manuals" / "solbench_problems"
DEFAULT_BENCHMARK_ROOT = REPO_ROOT / "tools" / "sol_execbench" / "data" / "benchmark"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "baselines"

# Internal-category → downloader-subset name.
SUBSET_MAP: Dict[str, str] = {
    "L1": "L1",
    "L2": "L2",
    "Quant": "Quant",
    "FIB": "FlashInfer-Bench",
}

# Languages the scaffolder is willing to emit. Vendor-library labels
# live in ``compile_options.ld_flags`` on a ``cuda_cpp`` source per
# CLAUDE.md Rule 5 — they are NOT valid ``--language`` values.
SUPPORTED_LANGUAGES = ("cutlass", "cuda_cpp")


_SLUG_PREFIX_RE = re.compile(r"^problem_")
_MAPPING_ROW_RE = re.compile(
    r"^\|\s*(?P<slug>problem_\d+_[A-Za-z0-9_]+)\s*"
    r"\|\s*(?P<category>L1|L2|Quant|FIB)\s*"
    r"\|\s*(?P<primary>[a-z0-9_]+)\s*"
    r"\|\s*(?P<secondary>[a-z0-9_,; ]+|none)\s*"
    r"\|\s*(?P<raw_source>[^|]+)\s*"
    r"\|\s*(?P<notes>[^|]*)\s*\|\s*$"
)


# --- Mapping parser -------------------------------------------------------


@dataclass(frozen=True)
class MappingRow:
    slug: str
    category: str
    primary_pattern: str
    secondary_patterns: List[str]
    raw_source: str
    notes: str


def _normalise_pattern(name: str) -> str:
    """Strip the legacy ``tbd_`` prefix from a pattern name."""
    return name[len("tbd_"):] if name.startswith("tbd_") else name


def _normalise_secondaries(raw: str) -> List[str]:
    raw = raw.strip()
    if raw == "none" or not raw:
        return []
    parts = re.split(r"[,;]", raw)
    return [_normalise_pattern(p.strip()) for p in parts if p.strip()]


def parse_mapping(mapping_path: Path) -> Dict[str, MappingRow]:
    """Parse the slug → pattern mapping at ``mapping_path``.

    Returns a ``{slug: MappingRow}`` mapping. Raises ``ValueError`` if
    no rows are recognised (catches table-format drift).
    """
    rows: Dict[str, MappingRow] = {}
    text = mapping_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        match = _MAPPING_ROW_RE.match(line)
        if match is None:
            continue
        slug = match.group("slug").strip()
        primary = _normalise_pattern(match.group("primary").strip())
        secondary = _normalise_secondaries(match.group("secondary"))
        row = MappingRow(
            slug=slug,
            category=match.group("category").strip(),
            primary_pattern=primary,
            secondary_patterns=secondary,
            raw_source=match.group("raw_source").strip(),
            notes=match.group("notes").strip(),
        )
        rows[slug] = row
    if not rows:
        raise ValueError(
            f"No mapping rows parsed from {mapping_path}. Did the table format change?"
        )
    return rows


# --- Slug → benchmark-name resolution -------------------------------------


def derive_benchmark_name(slug: str) -> str:
    """Strip the ``^problem_`` prefix to get the downloader's short name.

    Internal slug format is ``problem_NNN_<rest>`` (full markdown stem from
    ``references/NVIDIA_Manuals/solbench_problems/``); the downloader places
    benchmarks under ``tools/sol_execbench/data/benchmark/<subset>/NNN_<rest>/``.
    Stripping ``^problem_`` (keeping the digit prefix) gives ``NNN_<rest>``.
    """
    candidate = _SLUG_PREFIX_RE.sub("", slug)
    if not candidate or candidate == slug:
        raise ValueError(
            f"Cannot derive benchmark name from slug {slug!r}: expected "
            f"'problem_NNN_<rest>' format (e.g. "
            f"'problem_001_fused_add_rmsnorm_h2048' -> "
            f"'001_fused_add_rmsnorm_h2048')."
        )
    return candidate


def resolve_benchmark_dir(row: MappingRow, benchmark_root: Path) -> Path:
    """Return the canonical ``benchmark/<subset>/<name>/`` dir for a row."""
    subset = SUBSET_MAP.get(row.category)
    if subset is None:
        raise ValueError(
            f"Unknown category {row.category!r} for slug {row.slug!r}. "
            f"Expected one of {sorted(SUBSET_MAP)}."
        )
    name = derive_benchmark_name(row.slug)
    return benchmark_root / subset / name


def load_definition(benchmark_dir: Path, slug: str) -> Dict[str, Any]:
    """Load and JSON-parse ``definition.json`` from ``benchmark_dir``.

    Fails closed with an actionable error pointing at ``download_data.sh``
    when the file is missing.
    """
    definition = benchmark_dir / "definition.json"
    if not definition.is_file():
        raise FileNotFoundError(
            f"Benchmark definition.json missing for slug {slug!r} at "
            f"{definition}. Run ./tools/sol_execbench/scripts/download_data.sh "
            f"to populate, or check the slug-to-subset mapping."
        )
    return json.loads(definition.read_text(encoding="utf-8"))


# --- Manifest mode --------------------------------------------------------


def manifest_for_mapping(
    rows: Dict[str, MappingRow],
    benchmark_root: Path,
) -> Dict[str, Any]:
    """JSON-friendly inventory of every mapping row's readiness."""
    entries: List[Dict[str, Any]] = []
    pattern_counts: Dict[str, int] = {}
    for slug, row in sorted(rows.items()):
        try:
            benchmark_name = derive_benchmark_name(slug)
        except ValueError:
            benchmark_name = ""
        subset = SUBSET_MAP.get(row.category, "")
        benchmark_dir = benchmark_root / subset / benchmark_name if subset else None
        definition_present = bool(
            benchmark_dir and (benchmark_dir / "definition.json").exists()
        )
        dossier_path = DOSSIER_DIR / f"{slug}.md"
        spec_path = SPEC_DIR / f"{slug}.md"
        pattern_card_path = PATTERNS_DIR / f"{row.primary_pattern}.md"
        pattern_counts[row.primary_pattern] = pattern_counts.get(
            row.primary_pattern, 0
        ) + 1
        entries.append(
            {
                "slug": slug,
                "category": row.category,
                "subset": subset,
                "primary_pattern": row.primary_pattern,
                "secondary_patterns": row.secondary_patterns,
                "benchmark_name": benchmark_name,
                "definition_present": definition_present,
                "dossier_present": dossier_path.exists(),
                "spec_present": spec_path.exists(),
                "pattern_card_present": pattern_card_path.exists(),
            }
        )
    return {
        "total_problems": len(entries),
        "primary_pattern_counts": dict(sorted(pattern_counts.items())),
        "entries": entries,
    }


# --- CLI ------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/sol_baselines_scaffold/generate.py",
        description=(
            "Print a JSON manifest of every problem row in "
            "tools/sol_baselines_scaffold/sol_pattern_mapping.md (slug, category, "
            "primary pattern, "
            "benchmark-data presence, dossier / spec / pattern-card presence). "
            "Per-baseline kernel materialisation moved to scaffold.py."
        ),
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help=(
            "Print the manifest. Retained as an explicit flag for "
            "backwards compatibility; this is the only mode now."
        ),
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT,
        help=(
            f"Root of the downloaded benchmark dataset (default {DEFAULT_BENCHMARK_ROOT}). "
            "Use a custom path for unit tests with synthesised definitions."
        ),
    )
    parser.add_argument(
        "--mapping-path",
        type=Path,
        default=MAPPING_PATH,
        help=(
            "Path to tools/sol_baselines_scaffold/sol_pattern_mapping.md "
            "(override for testing)."
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = parse_mapping(args.mapping_path)
    manifest = manifest_for_mapping(rows, args.benchmark_root)
    json.dump(manifest, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
