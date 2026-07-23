"""sm_120 cohort candidate inventory.

Walks workloads/simbench_bw/roster.toml (filtered to arch_gate includes
sm_120 AND level == "L2") and workloads/portable_lcd/roster.toml,
expanding each row to (slug, shape_point) candidates via
workloads/simbench_bw/shapes/<slug>.toml when present, or via the
portable_lcd shape conventions (correctness_shape, timing_shape, plus
the trace_l2_accuracy_lg / trace_l2_accuracy_xl shapes admitted in the
historical sm_86 cohort).

Emits a JSON materialization table for downstream cohort freeze and
buildability preflight.

The classifier maps raw family values to family_bucket per the
discipline pinned in data/local_labels/sm120_cohort_family_buckets.toml
(authored alongside this inventory).

Buildability is NOT verified here; that is the preflight tool's job.
This script's output may include candidates whose binary or trace path
will not resolve at preflight time.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

__version__ = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[2]
SIMBENCH_DIR = REPO_ROOT / "workloads" / "simbench_bw"
PORTABLE_LCD_DIR = REPO_ROOT / "workloads" / "portable_lcd"
SIMBENCH_SHAPES_DIR = SIMBENCH_DIR / "shapes"

# Shape_points to enumerate for portable_lcd rows. The first two are
# the canonical portable_lcd workload flags (`--correctness_shape`,
# `--timing_shape`); the latter two were historically admitted on sm_86
# for the gemm and attention slugs and are kept as candidate shapes
# (the preflight will reject combinations whose workload main.cu does
# not handle --shape-point trace_l2_accuracy_* / does not emit the
# expected trace_json_path).
PORTABLE_LCD_SHAPE_POINTS = (
    "correctness_shape",
    "timing_shape",
    "trace_l2_accuracy_lg",
    "trace_l2_accuracy_xl",
)


@dataclass(frozen=True)
class Candidate:
    slug: str
    shape_point: str
    family: str
    family_bucket: str
    level: str  # always "L2" for cohort rows
    arch_gate: tuple[str, ...]
    binary_path: str
    manifest_path: str
    trace_json_path: str
    roster_source: str  # "simbench_bw" | "portable_lcd"
    source_row_level: str  # echo of the source roster's level (or empty for portable_lcd)


def _load_toml(p: Path) -> dict:
    with p.open("rb") as f:
        return tomllib.load(f)


def _enumerate_simbench_shapes(slug: str) -> list[str]:
    """Read workloads/simbench_bw/shapes/<slug>.toml and return all
    shape names from timing_shapes / trace_shapes / correctness_shapes.
    """
    toml_path = SIMBENCH_SHAPES_DIR / f"{slug}.toml"
    if not toml_path.is_file():
        return []
    data = _load_toml(toml_path)
    shapes: list[str] = []
    for section in ("timing_shapes", "trace_shapes", "correctness_shapes"):
        for entry in data.get(section, []):
            name = entry.get("name")
            if isinstance(name, str):
                shapes.append(name)
    return shapes


def _classify_family_bucket(family: str, family_bucket_map: dict[str, str]) -> str:
    """Look up the derived family_bucket for a raw family value.

    The mapping is committed in
    data/local_labels/sm120_cohort_family_buckets.toml. If the raw value
    is not present, returns "other" as a safe fallback; the schema
    validator that consumes the frozen roster will reject any candidate
    whose family is not in the committed mapping (separately from this
    inventory).
    """
    return family_bucket_map.get(family, "other")


def _candidate_for_simbench(
    row: dict,
    shape_point: str,
    family_bucket_map: dict[str, str],
) -> Candidate:
    slug = row["slug"]
    family = row.get("family", "")
    level = row.get("level", "")
    arch_gate = tuple(row.get("arch_gate", []))
    # binary lives under l1/ or l2/ per the source row's level
    bin_dir = "l1" if level == "L1" else "l2"
    binary_path = f"workloads/simbench_bw/{bin_dir}/{slug}/{slug}_run"
    manifest_path = (
        f"workloads/simbench_bw/{bin_dir}/{slug}/results/{slug}/{shape_point}/manifest.json"
    )
    trace_json_path = (
        f"workloads/simbench_bw/results/{slug}/rtx_5060_laptop/sm_120/"
        f"trace_{shape_point}/dynamic_trace.pb"
    )
    return Candidate(
        slug=slug,
        shape_point=shape_point,
        family=family,
        family_bucket=_classify_family_bucket(family, family_bucket_map),
        level="L2",  # cohort-row level is always L2 (the cohort IS L2);
                    # source row's level is preserved in source_row_level
        arch_gate=arch_gate,
        binary_path=binary_path,
        manifest_path=manifest_path,
        trace_json_path=trace_json_path,
        roster_source="simbench_bw",
        source_row_level=level,
    )


def _candidate_for_portable_lcd(
    row: dict,
    shape_point: str,
    family_bucket_map: dict[str, str],
) -> Candidate:
    slug = row["slug"]
    family = row.get("family", "")
    arch_gate = tuple(row.get("arch_gate", []))
    binary_path = f"workloads/portable_lcd/{slug}/{slug}_run"
    manifest_path = (
        f"workloads/portable_lcd/{slug}/results/{slug}/{shape_point}/manifest.json"
    )
    trace_json_path = (
        f"workloads/portable_lcd/results/{slug}/rtx_5060_laptop/sm_120/"
        f"trace_{shape_point}/dynamic_trace.pb"
    )
    return Candidate(
        slug=slug,
        shape_point=shape_point,
        family=family,
        family_bucket=_classify_family_bucket(family, family_bucket_map),
        level="L2",  # cohort-roster override per the plan
        arch_gate=arch_gate,
        binary_path=binary_path,
        manifest_path=manifest_path,
        trace_json_path=trace_json_path,
        roster_source="portable_lcd",
        source_row_level="",
    )


def materialize(family_bucket_map: dict[str, str]) -> list[Candidate]:
    """Return the full candidate inventory.

    Candidates from simbench_bw are filtered to rows whose arch_gate
    includes "sm_120" AND whose source level is "L2". Candidates from
    portable_lcd are taken from all rows whose arch_gate includes
    "sm_120"; shape_points are enumerated over PORTABLE_LCD_SHAPE_POINTS.
    """
    candidates: list[Candidate] = []

    sb = _load_toml(SIMBENCH_DIR / "roster.toml")
    for row in sb.get("rows", []):
        if "sm_120" not in row.get("arch_gate", []):
            continue
        level = row.get("level")
        # The cohort is L2. Source rows admitted into the cohort:
        #   - level==L2 rows: contribute all their shape-catalog shapes.
        #   - level==L1 rows: contribute ONLY their trace_l2_accuracy_*
        #     shapes (these are the L2-cache-layer measurements that the
        #     existing sm_86 cohort historically uses; the cohort-row
        #     level is set to L2 per the AC-2.1 cohort-row-level reading).
        if level not in ("L1", "L2"):
            continue
        shapes = _enumerate_simbench_shapes(row["slug"])
        if level == "L1":
            shapes = [s for s in shapes if s.startswith("trace_l2_accuracy_")]
            if not shapes:
                continue
        for sp in shapes:
            candidates.append(
                _candidate_for_simbench(row, sp, family_bucket_map)
            )

    pl = _load_toml(PORTABLE_LCD_DIR / "roster.toml")
    for row in pl.get("rows", []):
        if "sm_120" not in row.get("arch_gate", []):
            continue
        for sp in PORTABLE_LCD_SHAPE_POINTS:
            candidates.append(
                _candidate_for_portable_lcd(row, sp, family_bucket_map)
            )

    return candidates


def _candidate_to_dict(c: Candidate) -> dict:
    d = asdict(c)
    d["arch_gate"] = list(d["arch_gate"])
    return d


def _summarize_buckets(candidates: list[Candidate]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in candidates:
        out[c.family_bucket] = out.get(c.family_bucket, 0) + 1
    return out


def _summarize_buckets_distinct_slug(candidates: list[Candidate]) -> dict[str, int]:
    """Count distinct slugs per bucket (a single slug appearing under
    multiple shape_points counts once).
    """
    seen: dict[str, set[str]] = {}
    for c in candidates:
        seen.setdefault(c.family_bucket, set()).add(c.slug)
    return {k: len(v) for k, v in seen.items()}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--family-buckets",
        required=True,
        type=Path,
        help="Path to data/local_labels/sm120_cohort_family_buckets.toml",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path for the materialization table.",
    )
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip the per-candidate list; print only the per-bucket summary.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    fbm_data = _load_toml(args.family_buckets)
    family_bucket_map = dict(fbm_data.get("family_to_bucket", {}))
    if not family_bucket_map:
        print(
            f"ERROR: {args.family_buckets} has no [family_to_bucket] mapping",
            file=sys.stderr,
        )
        return 2

    candidates = materialize(family_bucket_map)
    summary = {
        "tool_version": __version__,
        "total_candidates": len(candidates),
        "candidates_per_bucket": _summarize_buckets(candidates),
        "distinct_slugs_per_bucket": _summarize_buckets_distinct_slug(candidates),
        "candidates_per_roster_source": {
            "simbench_bw": sum(
                1 for c in candidates if c.roster_source == "simbench_bw"
            ),
            "portable_lcd": sum(
                1 for c in candidates if c.roster_source == "portable_lcd"
            ),
        },
    }

    if args.summary_only:
        out_doc: dict = summary
    else:
        out_doc = {
            **summary,
            "candidates": [_candidate_to_dict(c) for c in candidates],
        }

    if args.output:
        args.output.write_text(json.dumps(out_doc, indent=2, sort_keys=True))
    else:
        json.dump(out_doc, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
