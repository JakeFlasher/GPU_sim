"""sm_86 SOL-ExecBench port-scope manifest.

Single source of truth for which SOL-ExecBench problems the sm_86
coverage loop ports, and which it excludes.

* **201 in-scope** — all 94 L1 + all 82 L2 + 25 of 26
  FlashInfer-Bench problems.
* **34 excluded** — the 33 ``Quant`` problems plus the single fp8
  FlashInfer-Bench problem (``problem_020_moe_fp8_block_scale_…``).
  Split 19 ``fp8_e4m3_contract`` + 15 ``nvfp4_e2m1_contract``. These
  cannot honor their dtype contract on Ampere sm_86; a bf16/fp16
  downgrade is a reward-hack risk, not a port (CLAUDE.md Rule 8).
  They defer to Phase 2 (sm_100 / B200) where the hardware supports
  the formats.

Canonical slug form is ``problem_<NNN>_<name>``, derived from the
benchmark directory ``sol-execbench/data/benchmark/<cat>/<NNN>_<name>``
(the authoritative evaluator identifier), then **truncated at the
first character outside ``[A-Za-z0-9_]``**. Three L1/L2 benchmark dirs
carry ``.`` / ``-`` in their names (``079_ImageNet_83.6_…``,
``051_seqlen-finetuned-…``, ``052_seqlen-…``); the canonical slug
truncates these to ``problem_079_ImageNet_83`` / ``problem_051_seqlen``
/ ``problem_052_seqlen`` — the SAME forms the predictor host-arch
availability manifest + the trace-layout bundle-name contract use, so
one slug is consistent across scope / trace layout / host-arch
preflight / baseline+bundle paths. (A ported solution keeps the FULL
benchmark-dir name in ``solution.json.definition`` so ``run.py``
resolves the real data dir; only the harness-facing slug is canonical.)
The benchmark dir is gitignored, so the disk cross-check is
skip-if-absent; the committed TOML is the checked-in source of truth.

This module is import-only inside calibration / diagnostic paths and
contains no NCU / clock-policy surface.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Sequence

try:
    import tomllib
except ImportError:  # pragma: no cover - Python < 3.11 fallback path
    import tomli as tomllib  # type: ignore[no-redef]


# --- Closed vocabularies + expected partition (the contract) ----------

EXCLUSION_REASONS: tuple[str, str] = (
    "fp8_e4m3_contract",
    "nvfp4_e2m1_contract",
)

#: In-scope categories, in canonical order. ``Quant`` is excluded
#: wholesale this loop, so it never appears in ``in_scope``.
IN_SCOPE_CATEGORIES: tuple[str, ...] = ("L1", "L2", "FIB")

#: Categories an excluded row may carry (Quant wholesale + the one
#: fp8 FlashInfer-Bench problem).
EXCLUDED_CATEGORIES: tuple[str, ...] = ("Quant", "FIB")

#: Benchmark directory name -> short category token.
BENCHMARK_DIR_TO_CATEGORY: dict[str, str] = {
    "L1": "L1",
    "L2": "L2",
    "Quant": "Quant",
    "FlashInfer-Bench": "FIB",
}

EXPECTED_TOTAL = 235
EXPECTED_IN_SCOPE = 201
EXPECTED_EXCLUDED = 34
EXPECTED_IN_SCOPE_BY_CATEGORY: dict[str, int] = {"L1": 94, "L2": 82, "FIB": 25}
EXPECTED_EXCLUDED_BY_CATEGORY: dict[str, int] = {"Quant": 33, "FIB": 1}
EXPECTED_EXCLUDED_BY_REASON: dict[str, int] = {
    "fp8_e4m3_contract": 19,
    "nvfp4_e2m1_contract": 15,
}

SCHEMA_VERSION = 1
PLAN_STEM = "sm86-solbench-coverage-predictor-calibration-20260529-023541"

_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SCOPE_PATH = _THIS_DIR / "sm86_port_scope_v1.toml"
# repo-root/sol-execbench/data/benchmark ; calibration/ -> baselines_harness/
# -> scripts/ -> repo-root == parents[2] of this file.
_REPO_ROOT = _THIS_DIR.parents[2]
DEFAULT_BENCHMARK_DIR = _REPO_ROOT / "sol-execbench" / "data" / "benchmark"


class Sm86PortScopeError(ValueError):
    """Raised when the scope manifest violates the port-scope contract."""


# --- Classification + generation (benchmark-dir authority) ------------

#: Slug-identifier characters. The canonical harness-facing slug is the
#: benchmark-dir-derived ``problem_<NNN>_<name>`` truncated at the first
#: character outside this class (so ``problem_079_ImageNet_83.6_…`` ->
#: ``problem_079_ImageNet_83``). A no-op for the 232 already-canonical names.
_CANONICAL_SLUG_RE = re.compile(r"problem_[A-Za-z0-9_]+")


def canonical_slug(full_slug: str) -> str:
    """Return the canonical harness-facing slug for a benchmark-derived slug.

    Truncates ``full_slug`` at the first character outside ``[A-Za-z0-9_]``
    (matching the predictor host-arch manifest + the trace-layout bundle-name
    contract). Classification (fp8/nvfp4) must run on the FULL slug before
    canonicalization, since a marker could in principle sit past a ``.``/``-``.
    """
    match = _CANONICAL_SLUG_RE.match(full_slug)
    if match is None:
        raise Sm86PortScopeError(f"slug {full_slug!r} does not start with problem_<id>")
    return match.group(0)


def classify_exclusion_reason(slug: str) -> str | None:
    """Return the exclusion reason for a slug, or ``None`` if in-scope.

    nvfp4 is checked first because ``nvfp4`` slugs do not contain the
    substring ``fp8`` but the precedence must be unambiguous regardless.
    """
    low = slug.lower()
    if "nvfp4" in low:
        return "nvfp4_e2m1_contract"
    if "fp8" in low:
        return "fp8_e4m3_contract"
    return None


def build_scope_from_benchmark(benchmark_dir: Path) -> dict:
    """Build the canonical scope doc by enumerating the benchmark dir.

    A ``Quant`` problem is always excluded (whole category); a
    FlashInfer-Bench (``FIB``) problem is excluded iff its slug is an
    fp8/nvfp4 problem. L1/L2 are always in-scope (and must contain no
    fp8/nvfp4 leak — enforced by :func:`validate_sm86_port_scope`).
    """
    benchmark_dir = Path(benchmark_dir)
    by_category: dict[str, list[str]] = {c: [] for c in BENCHMARK_DIR_TO_CATEGORY.values()}
    for bench_name, short in BENCHMARK_DIR_TO_CATEGORY.items():
        cat_dir = benchmark_dir / bench_name
        if not cat_dir.is_dir():
            raise Sm86PortScopeError(
                f"benchmark category dir missing: {cat_dir}"
            )
        for sub in sorted(p.name for p in cat_dir.iterdir() if p.is_dir()):
            by_category[short].append("problem_" + sub)

    excluded: list[dict[str, str]] = []
    in_scope: dict[str, list[str]] = {c: [] for c in IN_SCOPE_CATEGORIES}
    for short in ("L1", "L2", "Quant", "FIB"):
        for slug in by_category[short]:
            # Classify on the FULL benchmark-derived slug (preserves the
            # fp8/nvfp4 markers), then store the canonical (truncated) slug.
            reason = classify_exclusion_reason(slug)
            canonical = canonical_slug(slug)
            if short == "Quant":
                # Whole Quant category excluded; reason from the slug.
                excluded.append(
                    {"slug": canonical, "category": "Quant", "reason": reason or "fp8_e4m3_contract"}
                )
            elif short == "FIB" and reason is not None:
                excluded.append({"slug": canonical, "category": "FIB", "reason": reason})
            else:
                in_scope[short].append(canonical)

    excluded.sort(key=lambda r: (EXCLUDED_CATEGORIES.index(r["category"]), r["slug"]))
    for cat in in_scope:
        in_scope[cat].sort()

    excluded_by_category: dict[str, int] = {}
    excluded_by_reason: dict[str, int] = {}
    for row in excluded:
        excluded_by_category[row["category"]] = excluded_by_category.get(row["category"], 0) + 1
        excluded_by_reason[row["reason"]] = excluded_by_reason.get(row["reason"], 0) + 1

    return {
        "schema_version": SCHEMA_VERSION,
        "exclusion_reason_vocabulary": list(EXCLUSION_REASONS),
        "meta": {
            "plan": PLAN_STEM,
            "host_arch": "sm_86",
            "host_kind": "local_sm_8_6",
            "slug_form": "problem_<NNN>_<name>",
            "slug_authority": "sol-execbench/data/benchmark/<category>/<NNN>_<name> (canonical: truncated at first non-[A-Za-z0-9_])",
            "benchmark_source": "sol-execbench/data/benchmark",
            "total_problems": sum(len(v) for v in by_category.values()),
            "in_scope_count": sum(len(v) for v in in_scope.values()),
            "excluded_count": len(excluded),
            "in_scope_by_category": {c: len(in_scope[c]) for c in IN_SCOPE_CATEGORIES},
            "excluded_by_category": excluded_by_category,
            "excluded_by_reason": excluded_by_reason,
        },
        "excluded": excluded,
        "in_scope": in_scope,
    }


def _toml_str(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_scope_toml(doc: Mapping) -> str:
    """Render the canonical doc to deterministic TOML text."""
    meta = doc["meta"]
    lines: list[str] = []
    lines.append(
        "# sm_86 SOL-ExecBench port-scope manifest. GENERATED by"
    )
    lines.append(
        "# scripts/baselines_harness/calibration/sm86_port_scope.py"
    )
    lines.append(
        "# (build_scope_from_benchmark). Slug authority: the benchmark dir,"
    )
    lines.append(
        "# canonicalized (truncated at the first non-[A-Za-z0-9_] character)."
    )
    lines.append("")
    lines.append(f"schema_version = {doc['schema_version']}")
    vocab = ", ".join(_toml_str(r) for r in doc["exclusion_reason_vocabulary"])
    lines.append(f"exclusion_reason_vocabulary = [{vocab}]")
    lines.append("")
    lines.append("[meta]")
    for key in (
        "plan", "host_arch", "host_kind", "slug_form", "slug_authority",
        "benchmark_source",
    ):
        lines.append(f"{key} = {_toml_str(meta[key])}")
    for key in ("total_problems", "in_scope_count", "excluded_count"):
        lines.append(f"{key} = {meta[key]}")
    lines.append("")
    lines.append("[meta.in_scope_by_category]")
    for cat in IN_SCOPE_CATEGORIES:
        lines.append(f"{cat} = {meta['in_scope_by_category'][cat]}")
    lines.append("")
    lines.append("[meta.excluded_by_category]")
    for cat in EXCLUDED_CATEGORIES:
        if cat in meta["excluded_by_category"]:
            lines.append(f"{cat} = {meta['excluded_by_category'][cat]}")
    lines.append("")
    lines.append("[meta.excluded_by_reason]")
    for reason in EXCLUSION_REASONS:
        if reason in meta["excluded_by_reason"]:
            lines.append(f"{reason} = {meta['excluded_by_reason'][reason]}")
    lines.append("")
    for row in doc["excluded"]:
        lines.append("[[excluded]]")
        lines.append(f"slug = {_toml_str(row['slug'])}")
        lines.append(f"category = {_toml_str(row['category'])}")
        lines.append(f"reason = {_toml_str(row['reason'])}")
        lines.append("")
    lines.append("[in_scope]")
    for cat in IN_SCOPE_CATEGORIES:
        slugs = doc["in_scope"][cat]
        lines.append(f"{cat} = [")
        for slug in slugs:
            lines.append(f"  {_toml_str(slug)},")
        lines.append("]")
    return "\n".join(lines) + "\n"


# --- Loading + validation (the always-on contract gate) ---------------

def load_sm86_port_scope(path: Path = DEFAULT_SCOPE_PATH) -> dict:
    """Parse the committed scope TOML into a plain dict."""
    path = Path(path)
    if not path.is_file():
        raise Sm86PortScopeError(f"scope manifest missing: {path}")
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _all_in_scope_slugs(doc: Mapping) -> list[str]:
    out: list[str] = []
    for cat in IN_SCOPE_CATEGORIES:
        out.extend(doc.get("in_scope", {}).get(cat, []))
    return out


def validate_sm86_port_scope(
    doc: Mapping,
    *,
    benchmark_dir: Path | None = None,
) -> None:
    """Validate a scope doc against the port-scope contract; raise on any
    inconsistency.

    The internal-consistency checks (counts, category sums, closed
    reason vocabulary, disjointness, declared-vs-actual drift) always
    run. The slug+category *identity* cross-check against the benchmark
    dir runs only when ``benchmark_dir`` is provided and present on
    disk (it is gitignored, so it is skip-if-absent).
    """
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise Sm86PortScopeError(
            f"schema_version must be {SCHEMA_VERSION}; got {doc.get('schema_version')!r}"
        )
    vocab = doc.get("exclusion_reason_vocabulary")
    if list(vocab or []) != list(EXCLUSION_REASONS):
        raise Sm86PortScopeError(
            f"exclusion_reason_vocabulary must equal {list(EXCLUSION_REASONS)}; got {vocab!r}"
        )

    meta = doc.get("meta")
    if not isinstance(meta, Mapping):
        raise Sm86PortScopeError("missing [meta] table")

    # --- Excluded rows ------------------------------------------------
    excluded = doc.get("excluded")
    if not isinstance(excluded, Sequence) or isinstance(excluded, (str, bytes)):
        raise Sm86PortScopeError("excluded must be an array of tables")
    if len(excluded) != EXPECTED_EXCLUDED:
        raise Sm86PortScopeError(
            f"excluded count must be {EXPECTED_EXCLUDED}; got {len(excluded)}"
        )
    excluded_slugs: list[str] = []
    by_cat: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    for row in excluded:
        slug = row.get("slug")
        category = row.get("category")
        reason = row.get("reason")
        if not isinstance(slug, str) or not slug:
            raise Sm86PortScopeError(f"excluded row has empty slug: {row!r}")
        if category not in EXCLUDED_CATEGORIES:
            raise Sm86PortScopeError(
                f"excluded row category must be one of {EXCLUDED_CATEGORIES}; got {category!r} ({slug})"
            )
        if reason not in EXCLUSION_REASONS:
            raise Sm86PortScopeError(
                f"excluded row reason {reason!r} not in closed vocabulary {EXCLUSION_REASONS} ({slug})"
            )
        excluded_slugs.append(slug)
        by_cat[category] = by_cat.get(category, 0) + 1
        by_reason[reason] = by_reason.get(reason, 0) + 1
    if len(set(excluded_slugs)) != len(excluded_slugs):
        raise Sm86PortScopeError("duplicate slug in excluded set")
    if by_cat != EXPECTED_EXCLUDED_BY_CATEGORY:
        raise Sm86PortScopeError(
            f"excluded-by-category must be {EXPECTED_EXCLUDED_BY_CATEGORY}; got {by_cat}"
        )
    if by_reason != EXPECTED_EXCLUDED_BY_REASON:
        raise Sm86PortScopeError(
            f"excluded-by-reason must be {EXPECTED_EXCLUDED_BY_REASON}; got {by_reason}"
        )

    # --- In-scope lists ----------------------------------------------
    in_scope = doc.get("in_scope")
    if not isinstance(in_scope, Mapping):
        raise Sm86PortScopeError("missing [in_scope] table")
    extra_cats = set(in_scope) - set(IN_SCOPE_CATEGORIES)
    if extra_cats:
        raise Sm86PortScopeError(
            f"in_scope carries non-in-scope category keys {sorted(extra_cats)} "
            f"(Quant is wholesale-excluded this loop)"
        )
    in_scope_by_cat: dict[str, int] = {}
    all_in_scope: list[str] = []
    for cat in IN_SCOPE_CATEGORIES:
        slugs = in_scope.get(cat, [])
        if not isinstance(slugs, Sequence) or isinstance(slugs, (str, bytes)):
            raise Sm86PortScopeError(f"in_scope.{cat} must be an array")
        for slug in slugs:
            if not isinstance(slug, str) or not slug:
                raise Sm86PortScopeError(f"in_scope.{cat} has empty slug")
        in_scope_by_cat[cat] = len(slugs)
        all_in_scope.extend(slugs)
    if len(set(all_in_scope)) != len(all_in_scope):
        raise Sm86PortScopeError("duplicate slug in in_scope set")
    if in_scope_by_cat != EXPECTED_IN_SCOPE_BY_CATEGORY:
        raise Sm86PortScopeError(
            f"in_scope-by-category must be {EXPECTED_IN_SCOPE_BY_CATEGORY}; got {in_scope_by_cat}"
        )
    if len(all_in_scope) != EXPECTED_IN_SCOPE:
        raise Sm86PortScopeError(
            f"in_scope count must be {EXPECTED_IN_SCOPE}; got {len(all_in_scope)}"
        )

    # --- Disjointness + grand total ----------------------------------
    overlap = set(excluded_slugs) & set(all_in_scope)
    if overlap:
        raise Sm86PortScopeError(f"slug(s) in BOTH excluded and in_scope: {sorted(overlap)}")
    if len(excluded_slugs) + len(all_in_scope) != EXPECTED_TOTAL:
        raise Sm86PortScopeError(
            f"excluded + in_scope must equal {EXPECTED_TOTAL}; got "
            f"{len(excluded_slugs)} + {len(all_in_scope)}"
        )

    # --- Declared meta counts must match actuals (drift guard) -------
    declared = {
        "total_problems": meta.get("total_problems"),
        "in_scope_count": meta.get("in_scope_count"),
        "excluded_count": meta.get("excluded_count"),
    }
    actual = {
        "total_problems": EXPECTED_TOTAL,
        "in_scope_count": len(all_in_scope),
        "excluded_count": len(excluded_slugs),
    }
    for key in declared:
        if declared[key] != actual[key]:
            raise Sm86PortScopeError(
                f"meta.{key} declared {declared[key]!r} != actual {actual[key]!r}"
            )
    if dict(meta.get("in_scope_by_category", {})) != EXPECTED_IN_SCOPE_BY_CATEGORY:
        raise Sm86PortScopeError(
            "meta.in_scope_by_category does not match the canonical partition"
        )
    if dict(meta.get("excluded_by_category", {})) != EXPECTED_EXCLUDED_BY_CATEGORY:
        raise Sm86PortScopeError(
            "meta.excluded_by_category does not match the canonical partition"
        )
    if dict(meta.get("excluded_by_reason", {})) != EXPECTED_EXCLUDED_BY_REASON:
        raise Sm86PortScopeError(
            "meta.excluded_by_reason does not match the canonical partition"
        )

    # --- Optional slug+category identity vs the benchmark dir ---------
    if benchmark_dir is not None and Path(benchmark_dir).is_dir():
        built = build_scope_from_benchmark(Path(benchmark_dir))
        built_excluded = {(r["slug"], r["category"], r["reason"]) for r in built["excluded"]}
        doc_excluded = {(r["slug"], r["category"], r["reason"]) for r in excluded}
        if built_excluded != doc_excluded:
            raise Sm86PortScopeError(
                "excluded set diverges from the benchmark-dir partition: "
                f"only-in-disk={sorted(built_excluded - doc_excluded)} "
                f"only-in-toml={sorted(doc_excluded - built_excluded)}"
            )
        for cat in IN_SCOPE_CATEGORIES:
            if set(built["in_scope"][cat]) != set(in_scope.get(cat, [])):
                disk = set(built["in_scope"][cat])
                toml_set = set(in_scope.get(cat, []))
                raise Sm86PortScopeError(
                    f"in_scope.{cat} diverges from benchmark dir: "
                    f"only-in-disk={sorted(disk - toml_set)} "
                    f"only-in-toml={sorted(toml_set - disk)}"
                )


def _main() -> int:  # pragma: no cover - manual regeneration entry point
    import sys

    doc = build_scope_from_benchmark(DEFAULT_BENCHMARK_DIR)
    validate_sm86_port_scope(doc, benchmark_dir=DEFAULT_BENCHMARK_DIR)
    DEFAULT_SCOPE_PATH.write_text(render_scope_toml(doc), encoding="utf-8")
    sys.stdout.write(f"wrote {DEFAULT_SCOPE_PATH}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
