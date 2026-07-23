#!/usr/bin/env python3
"""Golden-update gate for SimBench-BW.

The only supported writer of workloads/simbench_bw/golden/<slug>/<platform>/<arch>/results.json
in the repository. Running this script:

  * demands --update-golden (by design; bare invocation is refused),
  * enforces that the recorded outputs come from `correctness_shapes`, not
    `timing_shapes` / `trace_shapes`,
  * refuses to write if the aggregate golden/ tree exceeds 64 MiB after the
    write (the repo-size budget enforced by verify_coverage.py
    --check-golden-size),
  * records the invocation rationale in the commit message that the caller
    must supply via --reason.

The script does NOT run any kernel itself. Kernel runs emit a candidate
golden JSON under a temporary location; this script validates + promotes it
into the canonical `golden/` tree. This separation means the write path stays
auditable by `run_suite.py --mode regression --strict-golden-origin` later
without coupling kernel execution to the writer.

Exit codes:
    0  golden(s) updated successfully.
    1  promotion rejected (shape policy violation, size budget, mismatched slug).
    2  usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any


SUITE_ROOT = Path(__file__).resolve().parent
GOLDEN_DIR = SUITE_ROOT / "golden"
MAX_GOLDEN_BYTES = 64 * 1024 * 1024  # 64 MiB per the plan


def _load_candidate(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _load_roster_slug(slug: str) -> dict[str, Any]:
    with open(SUITE_ROOT / "roster.toml", "rb") as f:
        rows = tomllib.load(f)["rows"]
    for r in rows:
        if r["slug"] == slug:
            return r
    raise KeyError(slug)


def _total_golden_bytes() -> int:
    if not GOLDEN_DIR.exists():
        return 0
    total = 0
    for p in GOLDEN_DIR.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def _enforce_correctness_shapes_only(candidate: dict[str, Any], slug: str) -> None:
    """Reject any candidate produced from timing / trace shapes."""
    origin_source = candidate.get("shape_set")
    if origin_source != "correctness_shapes":
        raise SystemExit(f"golden_shape_policy_violation: slug={slug} golden candidate shape_set={origin_source!r}, expected 'correctness_shapes'")
    shape_point = candidate.get("shape_point")
    if not shape_point:
        raise SystemExit(f"golden_shape_policy_violation: slug={slug} candidate missing shape_point name")
    # Cross-check shape_point exists in the row's correctness_shapes
    shape_file = SUITE_ROOT / "shapes" / f"{slug}.toml"
    if not shape_file.exists():
        raise SystemExit(f"golden_shape_policy_violation: slug={slug} shapes/{slug}.toml not found")
    with open(shape_file, "rb") as f:
        shapes = tomllib.load(f)
    correctness_names = {p["name"] for p in shapes.get("correctness_shapes", [])}
    if shape_point not in correctness_names:
        raise SystemExit(f"golden_shape_policy_violation: slug={slug} shape_point {shape_point!r} not in correctness_shapes")


def promote(candidate_path: Path, slug: str, platform: str, arch: str, reason: str) -> Path:
    """Validate + move the candidate into golden/<slug>/<platform>/<arch>/results.json.

    Returns the destination path on success.
    """
    candidate = _load_candidate(candidate_path)
    if candidate.get("slug") != slug:
        raise SystemExit(f"golden_slug_mismatch: candidate slug={candidate.get('slug')!r}, expected {slug!r}")
    _enforce_correctness_shapes_only(candidate, slug)

    dest_dir = GOLDEN_DIR / slug / platform / arch
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "results.json"

    # Simulate the write in-memory and check the aggregate budget
    candidate_bytes = candidate_path.stat().st_size
    existing_bytes = dest.stat().st_size if dest.exists() else 0
    projected_total = _total_golden_bytes() - existing_bytes + candidate_bytes
    if projected_total > MAX_GOLDEN_BYTES:
        raise SystemExit(
            f"golden_repo_budget_exceeded: projected total {projected_total} > {MAX_GOLDEN_BYTES} "
            f"(diff {projected_total - MAX_GOLDEN_BYTES} bytes)"
        )

    # Annotate the rationale into the promoted file itself so the commit-message
    # narrative stays queryable at the file level.
    candidate["_golden_update_rationale"] = reason
    with open(dest, "w") as f:
        json.dump(candidate, f, indent=2, sort_keys=False)
    return dest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SimBench-BW golden update gate")
    p.add_argument("--update-golden", action="store_true", required=True,
                   help="Required flag: no write path exists without it")
    p.add_argument("--candidate", metavar="PATH", required=True,
                   help="Path to the candidate results.json produced by a correctness run")
    p.add_argument("--slug", required=True)
    p.add_argument("--platform", required=True, choices=["h100", "rtx_pro_6000_blackwell", "rtx_5060_laptop", "rtx_3090_ti"])
    p.add_argument(
        "--arch",
        required=True,
        help=(
            "Architecture string. During Loop 2 only `sm_86` and "
            "`sm_120-laptop` are admitted by the manifest-backed "
            "arch-capabilities check at "
            "`tools/arch_capabilities.json`; server arches "
            "(`sm_80`, `sm_89`, `sm_90a`, server `sm_120`) reject "
            "with typed `arch_unsupported` JSON on stderr."
        ),
    )
    p.add_argument("--reason", required=True, help="Rationale recorded in both the promoted file and the commit message")
    args = p.parse_args(argv)

    # Architecture capability gate (C-AC-E). The check runs BEFORE
    # the candidate-file existence check below so an unsupported
    # arch fails through the typed-arch_unsupported path rather
    # than through the legacy `candidate_missing` path. Lazy import
    # keeps the helper out of the module-load chain when the
    # script is imported (rather than executed as __main__).
    _REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    _TOOLS_DIR = _REPO_ROOT / "tools"
    if str(_TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(_TOOLS_DIR))
    import arch_capabilities  # type: ignore[import-not-found]

    typed = arch_capabilities.assert_arch_supported(
        path="workloads/simbench_bw/update_goldens.py",
        arch=args.arch,
    )
    if typed is not None:
        sys.stderr.write(
            json.dumps(typed, sort_keys=True, ensure_ascii=False)
            + "\n"
        )
        return 1

    cand = Path(args.candidate)
    if not cand.exists():
        print(f"candidate_missing: {cand}", file=sys.stderr)
        return 2

    try:
        _load_roster_slug(args.slug)
    except KeyError:
        print(f"unknown_slug: {args.slug}", file=sys.stderr)
        return 2

    try:
        dest = promote(cand, args.slug, args.platform, args.arch, args.reason)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"ok: wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
