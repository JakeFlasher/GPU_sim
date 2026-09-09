"""Mirror each landed sm_86 coverage bundle's evidence label to its two
human-readable cross-check sites.

The five counter-free analytic surfaces emit one ``<slug>__<lang>__sm86``
bundle per landed sm_86 coverage cell, and that bundle's ``manifest.json``
carries the authoritative ``evidence_label``. The evidence-label consistency
surface (``tests/test_evidence_label_consistency.py``) requires the same label
to appear at two further sites:

  * ``tools/sol_baselines_harness/report.md`` — one aggregate table row per
    bundle, whose trace column points at the committed
    ``.harness_traces/<slug>__<lang>__sm86/trace.json``.
  * ``baselines/<category>/<slug>/README.md`` — an
    ``| Evidence (sm_86) | `<label>` |`` table row.

``report.py`` cannot itself emit these per-arch rows: it walks one row per
``(slug, language)`` and routes the trace dir by the *live* host kind, and its
bundle-trace integrity gate hard-codes the un-suffixed ``<slug>__<lang>``
path — so on a non-sm_86-routed report it would never reference the
``__sm86`` sibling, and on the sm_86 host it would reject the suffixed path.
This module therefore *augments* the report.py-owned ``report.md`` (and the
per-problem READMEs) for exactly the sm_86 coverage bundles that the
one-row-per-(slug,language) model leaves out.

It is read-only over the committed bundles (no evaluator, no device work), and
both deterministic and idempotent: a re-run reproduces byte-identical
``report.md`` and READMEs. The managed set is derived from disk
(``.harness_traces/*__{cuda_cpp,cutlass}__sm86`` with a manifest, minus the
five frozen pilot-seed slugs whose curated rows are owned elsewhere), so it is
independent of the exclusion ledger that this work retires.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Final, List, Optional, Tuple

_THIS_DIR: Final[Path] = Path(__file__).resolve().parent
_REPO_ROOT: Final[Path] = _THIS_DIR.parent.parent
_TRACES_DIR: Final[Path] = _THIS_DIR / ".harness_traces"
_REPORT_MD: Final[Path] = _THIS_DIR / "report.md"
_BASELINES_DIR: Final[Path] = _REPO_ROOT / "baselines"

# The five frozen pilot-seed slugs carry curated ``__sm86`` rows (their
# bundles ran the full workload sweep, not the single-workload coverage
# profile) and are pinned read-only by the immutable seed freeze. Their rows
# are owned by the curated report.md, not by this augmenter.
_PILOT_SEED_SLUGS: Final[frozenset[str]] = frozenset(
    {
        "problem_006_hyena_depthwise_conv1d_split_gate",
        "problem_040_conv2d_residual_block",
        "problem_066_resnet_block_with_time_embedding",
        "problem_073_fused_encoder_final_norm_to_decoder_cross_attention_kv_projection",
        "problem_087_embedding_with_initial_layernorm_backward",
    }
)

_SUPPORTED_LANGS: Final[frozenset[str]] = frozenset({"cuda_cpp", "cutlass"})

# Mirrors the sm_86 rows already curated into report.md: the sm_86 host has no
# SOLAR T_SOL builder/arch-yaml match, so the score column honestly emits this
# sentinel and the t_sol=0 diagnostic column is 0.000.
_UNKNOWN_SOL_SCORE: Final[str] = "unknown_t_sol"


def _slug_of(bundle_key: str) -> str:
    return "__".join(bundle_key.split("__")[:-2])


def _lang_of(bundle_key: str) -> str:
    return bundle_key.split("__")[-2]


def managed_bundle_keys(traces_dir: Path = _TRACES_DIR) -> List[str]:
    """Return the sorted ``slug__lang__sm86`` keys this module owns."""
    keys: List[str] = []
    for bundle in traces_dir.glob("problem_*__sm86"):
        if not bundle.is_dir() or not (bundle / "manifest.json").is_file():
            continue
        parts = bundle.name.split("__")
        if parts[-1] != "sm86" or parts[-2] not in _SUPPORTED_LANGS:
            continue
        if _slug_of(bundle.name) in _PILOT_SEED_SLUGS:
            continue
        keys.append(bundle.name)
    return sorted(keys)


def _category_for_slug(slug: str, baselines_dir: Path) -> Optional[str]:
    # ``p`` is ``baselines/<category>/<slug>/README.md`` -> the category dir
    # is ``p.parent.parent``.
    matches = sorted(
        p.parent.parent.name for p in baselines_dir.glob(f"*/{slug}/README.md")
    )
    return matches[0] if len(matches) == 1 else None


def _readme_for_slug(slug: str, baselines_dir: Path) -> Optional[Path]:
    matches = list(baselines_dir.glob(f"*/{slug}/README.md"))
    return matches[0] if len(matches) == 1 else None


def _bundle_row(
    bundle_key: str, traces_dir: Path, baselines_dir: Path
) -> Dict[str, object]:
    """Compute the report-row fields for one bundle from its committed
    ``trace.json`` + ``manifest.json`` (no device work)."""
    bundle = traces_dir / bundle_key
    trace = json.loads((bundle / "trace.json").read_text(encoding="utf-8"))
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    rows = trace.get("rows", []) or []
    expected = trace.get("expected_workload_count")
    workload_count = expected if expected is not None else len(rows)
    all_passed = (
        bool(rows)
        and all(r.get("status") == "PASSED" for r in rows)
        and (expected is None or len(rows) == expected)
        and not trace.get("parse_errors")
    )
    speedups = [
        float(r["speedup_factor"])
        for r in rows
        if isinstance(r.get("speedup_factor"), (int, float))
    ]
    slowest = min(speedups) if speedups else 0.0
    slug = _slug_of(bundle_key)
    category = _category_for_slug(slug, baselines_dir)
    if category is None:
        raise ValueError(f"{bundle_key}: no unique baselines/<cat>/{slug}/README.md")
    trace_rel = (
        f"tools/sol_baselines_harness/.harness_traces/{bundle_key}/trace.json"
    )
    return {
        "category": category,
        "slug": slug,
        "language": _lang_of(bundle_key),
        "workload_count": workload_count,
        "all_passed": all_passed,
        "meets_speed_gate": slowest > 1.0,
        "slowest_speedup": slowest,
        "evidence_label": manifest.get("evidence_label", ""),
        "trace": trace_rel,
    }


def _render_report_row(row: Dict[str, object]) -> str:
    return (
        f"| {row['category']} | `{row['slug']}` | `{row['language']}` | "
        f"{row['workload_count']} | {row['all_passed']} | "
        f"{row['meets_speed_gate']} | "
        f"{float(row['slowest_speedup']):.3f} | "
        f"`{_UNKNOWN_SOL_SCORE}` | "
        f"0.000 | "
        f"`{row['evidence_label']}` | `{row['trace']}` |"
    )


_MANAGED_TRACE_RE_TEMPLATE = (
    r"\.harness_traces/(?P<key>problem_[^/`]+__(?:cuda_cpp|cutlass)__sm86)/trace\.json"
)


def augment_report_md(
    text: str, rows: List[Dict[str, object]], managed_keys: set[str]
) -> str:
    """Return ``report.md`` text with the managed bundles' rows present.

    Idempotent: any existing table row whose trace path is one of the managed
    keys is removed, then the freshly-rendered managed rows are inserted in
    sorted-key order immediately after the last remaining landed-pilot row.
    Rows for non-managed bundles (e.g. the curated pilot-seed rows) are left
    untouched.
    """
    lines = text.split("\n")
    trace_re = re.compile(_MANAGED_TRACE_RE_TEMPLATE)

    def _row_managed_key(line: str) -> Optional[str]:
        m = trace_re.search(line)
        if m is None:
            return None
        key = m.group("key")
        return key if key in managed_keys else None

    # Drop any existing managed rows (idempotent re-emit).
    kept = [ln for ln in lines if _row_managed_key(ln) is None]

    rendered = [_render_report_row(r) for r in rows]

    # Find the contiguous landed-pilot table: the header separator line, then
    # every following ``| ... |`` row. Insert after the last such row.
    sep_idx = None
    for i, ln in enumerate(kept):
        if ln.strip().startswith("|---") and ln.strip().endswith("|"):
            sep_idx = i
            break
    if sep_idx is None:
        raise ValueError("report.md: landed-pilot table header separator not found")
    last_row_idx = sep_idx
    for i in range(sep_idx + 1, len(kept)):
        if kept[i].lstrip().startswith("|"):
            last_row_idx = i
        else:
            break
    out = kept[: last_row_idx + 1] + rendered + kept[last_row_idx + 1 :]
    return "\n".join(out)


_LEGACY_EVIDENCE_RE = re.compile(r"^\|\s*Evidence\s*\|")
_SM86_EVIDENCE_RE = re.compile(r"^\|\s*Evidence \(sm_86\)\s*\|")


def _build_sm86_readme_row(legacy_line: str, label: str) -> str:
    """Build an ``| Evidence (sm_86) | `<label>` |`` row aligned to the
    legacy ``| Evidence | ... |`` row's column widths."""
    cells = legacy_line.split("|")
    # cells: ['', ' Evidence... ', ' `<value>`... ', ''] for a 2-col table.
    field_w = len(cells[1]) if len(cells) > 1 else len(" Evidence (sm_86) ")
    value_w = len(cells[2]) if len(cells) > 2 else len(f" `{label}` ")
    field = " Evidence (sm_86)"
    field = field + " " * max(1, field_w - len(field))
    value = f" `{label}`"
    value = value + " " * max(1, value_w - len(value))
    return f"|{field}|{value}|"


def augment_readme(text: str, label: str) -> Tuple[str, bool]:
    """Return (new_text, changed). Insert/update the ``| Evidence (sm_86) |``
    row right after the legacy ``| Evidence |`` row. Idempotent."""
    lines = text.split("\n")
    legacy_idx = next(
        (i for i, ln in enumerate(lines) if _LEGACY_EVIDENCE_RE.match(ln)), None
    )
    if legacy_idx is None:
        raise ValueError("README has no `| Evidence | ... |` row to anchor under")
    new_row = _build_sm86_readme_row(lines[legacy_idx], label)
    existing_idx = next(
        (i for i, ln in enumerate(lines) if _SM86_EVIDENCE_RE.match(ln)), None
    )
    if existing_idx is not None:
        if lines[existing_idx] == new_row:
            return text, False
        lines[existing_idx] = new_row
        return "\n".join(lines), True
    lines.insert(legacy_idx + 1, new_row)
    return "\n".join(lines), True


def emit(
    traces_dir: Path = _TRACES_DIR,
    report_md: Path = _REPORT_MD,
    baselines_dir: Path = _BASELINES_DIR,
) -> Dict[str, int]:
    """Augment report.md + the per-problem READMEs in place. Returns a small
    summary dict."""
    keys = managed_bundle_keys(traces_dir)
    managed_set = set(keys)
    rows = [_bundle_row(k, traces_dir, baselines_dir) for k in keys]

    if not report_md.is_file():
        raise SystemExit(
            f"{report_md} is missing; run report.py first to create the "
            "aggregate report before adding sm_86 coverage rows."
        )

    new_report = augment_report_md(
        report_md.read_text(encoding="utf-8"), rows, managed_set
    )
    report_md.write_text(new_report, encoding="utf-8")

    readmes_changed = 0
    for row in rows:
        readme = _readme_for_slug(str(row["slug"]), baselines_dir)
        if readme is None:
            raise ValueError(f"{row['slug']}: no unique README to update")
        new_text, changed = augment_readme(
            readme.read_text(encoding="utf-8"), str(row["evidence_label"])
        )
        if changed:
            readme.write_text(new_text, encoding="utf-8")
            readmes_changed += 1
    return {
        "managed_bundles": len(keys),
        "report_rows_emitted": len(rows),
        "readmes_changed": readmes_changed,
    }


if __name__ == "__main__":
    summary = emit()
    print(json.dumps(summary, indent=2))
