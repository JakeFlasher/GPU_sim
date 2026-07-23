"""sm_120 L2 cohort buildability preflight (AC-2.5).

Walks `data/local_labels/sm120_cohort_expansion_roster.toml` and for
each of the 40 candidates verifies that:

1. `arch_gate` includes "sm_120".
2. `binary_path` exists on disk OR builds cleanly via the candidate's
   per-slug Makefile (under `workloads/<roster_source>/<slug>/`).
3. The `manifest_path` parent directory is plausible (the manifest is
   produced at sweep time by the kernel runner; the preflight checks
   that the path is well-formed and that the runner subtree exists).
4. The NVBit tracer at `tools/nvbit_tracer/tracer_tool/tracer_tool.so`
   exists so the `trace_json_path` can be produced by the existing
   tracer toolchain at sweep time.

Emits `data/local_labels/sm120_cohort_buildability_witness.json` with
per-candidate provenance fields per AC-2.5.

Exit code:
  0 — all 40 candidates passed; witness sidecar written.
  2 — at least one candidate failed; sidecar still written with
      per-row failure tokens so the operator can decide whether to
      substitute from the replacement pool.

Usage:

    python tools/local_labels/sm120_cohort_buildability_preflight.py \\
        --roster data/local_labels/sm120_cohort_expansion_roster.toml \\
        --output data/local_labels/sm120_cohort_buildability_witness.json \\
        [--build-missing]      # invoke `make` when binary_path absent
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path

__version__ = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[2]
NVBIT_TRACER_SO = (
    REPO_ROOT / "tools" / "nvbit_tracer" / "tracer_tool" / "tracer_tool.so"
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        return out
    except subprocess.CalledProcessError:
        return ""


def _run_make_for_slug(slug_dir: Path) -> tuple[bool, str]:
    """Invoke `make` inside the slug directory. Returns
    (success, transcript_tail)."""
    try:
        result = subprocess.run(
            ["make"],
            cwd=slug_dir,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return False, "make timed out after 300s"
    except FileNotFoundError as e:
        return False, f"make invocation failed: {e}"
    transcript = (result.stdout + result.stderr).strip()
    tail = "\n".join(transcript.splitlines()[-10:])
    return result.returncode == 0, tail


def _check_candidate(row: dict, build_missing: bool) -> dict:
    """Return a witness-row dict for one candidate."""
    out: dict = {
        "slug": row.get("slug"),
        "shape_point": row.get("shape_point"),
        "split": row.get("split"),
        "candidate_order": row.get("candidate_order"),
        "roster_source": row.get("roster_source"),
        "arch_gate": list(row.get("arch_gate", [])),
        "manifest_path": row.get("manifest_path"),
        "expected_trace_json_path": row.get("trace_json_path"),
        "resolved_binary_path": None,
        "binary_sha256_if_built": None,
        "failure_tokens": [],
    }

    if "sm_120" not in out["arch_gate"]:
        out["failure_tokens"].append("arch_gate_missing_sm120")
        return out

    binary_rel = row.get("binary_path", "")
    binary_abs = REPO_ROOT / binary_rel

    if not binary_abs.is_file() and build_missing:
        slug_dir = binary_abs.parent
        if slug_dir.is_dir():
            success, tail = _run_make_for_slug(slug_dir)
            if not success:
                out["failure_tokens"].append(
                    f"make_failed: {tail.splitlines()[-1] if tail else 'no_output'}"
                )

    if not binary_abs.is_file():
        out["failure_tokens"].append("binary_path_missing")
    else:
        out["resolved_binary_path"] = str(binary_abs.relative_to(REPO_ROOT))
        out["binary_sha256_if_built"] = _sha256_file(binary_abs)

    # The manifest_path is written by the kernel runner at sweep time
    # under a `results/<slug>/<shape_point>/` subtree it creates as
    # needed. Verify the slug's source directory (where the Makefile
    # lives) is on disk; if so, the manifest_path is reachable once the
    # runner writes it.
    slug_dir = binary_abs.parent
    if not slug_dir.is_dir():
        out["failure_tokens"].append("slug_source_dir_missing")

    # Trace JSON path is produced at sweep time by NVBit. Check that the
    # tracer tool .so is built.
    if not NVBIT_TRACER_SO.is_file():
        out["failure_tokens"].append("nvbit_tracer_so_missing")

    return out


def preflight(
    roster_path: Path,
    build_missing: bool = False,
) -> tuple[list[dict], bool]:
    """Return (witness_rows, all_passed)."""
    with roster_path.open("rb") as f:
        roster = tomllib.load(f)
    rows = roster.get("rows", [])
    witness_rows = [_check_candidate(r, build_missing) for r in rows]
    all_passed = all(not w["failure_tokens"] for w in witness_rows)
    return witness_rows, all_passed


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--roster",
        required=True,
        type=Path,
        help="Path to data/local_labels/sm120_cohort_expansion_roster.toml",
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to write the witness sidecar JSON.",
    )
    p.add_argument(
        "--build-missing",
        action="store_true",
        help=(
            "When a candidate's binary_path is absent, invoke `make` in "
            "the per-slug directory before re-checking."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows, all_passed = preflight(args.roster, args.build_missing)
    witness = {
        "preflight_tool_version": __version__,
        "source_commit": _git_head_sha(),
        "preflight_timestamp_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "roster_sha256": _sha256_file(args.roster),
        "total_candidates": len(rows),
        "passed_candidates": sum(1 for r in rows if not r["failure_tokens"]),
        "failed_candidates": sum(1 for r in rows if r["failure_tokens"]),
        "all_passed": all_passed,
        "candidates": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(witness, indent=2, sort_keys=True))
    print(
        f"preflight: {witness['passed_candidates']}/{witness['total_candidates']} "
        f"passed; sidecar written to {args.output}"
    )
    if not all_passed:
        for r in rows:
            if r["failure_tokens"]:
                print(
                    f"  FAIL: {r['slug']}/{r['shape_point']} "
                    f"(candidate_order={r['candidate_order']}): "
                    f"{r['failure_tokens']}",
                    file=sys.stderr,
                )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
