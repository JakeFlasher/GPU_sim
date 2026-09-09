"""Regenerate `tools/sol_baselines_harness/deferred.json` from the authoritative
sm_86 coverage ledger (final-closeout milestone).

The coverage ledger `sm86_coverage_ledger_v1.toml` is the single source of truth
for per-problem landing state: a row with `deferred_correctness = true` carries a
`deferred_reason`. This emitter projects every such row into a `deferred.json`
entry (`status = deferred_correctness`). `deferred_reward_hack` is EMPTY by
construction — no row is ever a reward-hack (we never ship evaluator-state
exploits, CLAUDE.md Rule 6) — and that invariant is recorded explicitly.

Deterministic: entries sorted by (slug, language); `harness_commit` is the
coverage ledger's content SHA256 (the authority's identity), not a live git SHA,
so re-emission is byte-stable. A drift test fails if `deferred.json` diverges from
the coverage ledger's `deferred_correctness` set.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_COVERAGE_LEDGER: Final[Path] = (
    _REPO_ROOT / "tools" / "sol_baselines_harness" / "calibration"
    / "sm86_coverage_ledger_v1.toml"
)
_DEFERRED_JSON: Final[Path] = _REPO_ROOT / "tools" / "sol_baselines_harness" / "deferred.json"

SCHEMA_VERSION: Final[int] = 4  # v4: regenerated from the sm_86 coverage ledger (final-closeout milestone)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_deferred(repo_root: Path = _REPO_ROOT) -> dict:
    import tomllib
    ledger_path = repo_root / _COVERAGE_LEDGER.relative_to(_REPO_ROOT)
    cov = tomllib.loads(ledger_path.read_text("utf-8"))
    ledger_sha = _sha256(ledger_path)
    rows = cov["rows"]
    deferred = sorted(
        (r for r in rows if r.get("deferred_correctness")),
        key=lambda r: (r["slug"], r.get("language", "")),
    )
    entries = [{
        "slug": r["slug"],
        "language": r.get("language", ""),
        "status": "deferred_correctness",
        "category": r.get("category", ""),
        "workloads_passed": int(r.get("workloads_passed", 0)),
        "workloads_total": int(r.get("workloads_total", 0)),
        "justification": r.get("deferred_reason", ""),
        "evidence": (
            f"sm86_coverage_ledger_v1.toml row (baseline_dir={r.get('baseline_dir', '')})"
        ),
        "harness_commit": f"sm86_coverage_ledger_v1@{ledger_sha[:12]}",
    } for r in deferred]
    reward_hack = [r for r in rows if r.get("deferred_reward_hack")]
    return {
        "schema": {
            "version": SCHEMA_VERSION,
            "source_of_truth": "tools/sol_baselines_harness/calibration/sm86_coverage_ledger_v1.toml",
            "source_of_truth_sha256": ledger_sha,
            "fields": {
                "slug": "Internal problem slug.",
                "language": "Submission language: cuda_cpp or cutlass.",
                "status": "deferred_correctness (any non-PASSED status other than REWARD_HACK).",
                "category": "L1 | L2 | Quant | FIB.",
                "workloads_passed": "Workloads that PASS correctness.",
                "workloads_total": "Total workloads in workload.jsonl.",
                "justification": "One-line explanation of the failing workload(s) and why.",
                "evidence": "Pointer to the authoritative coverage-ledger row.",
                "harness_commit": "Coverage-ledger content identity (sm86_coverage_ledger_v1@<sha12>).",
            },
            "valid_statuses": ["deferred_correctness", "deferred_reward_hack"],
            "retired_statuses": ["deferred_slow"],
        },
        "deferred_correctness_count": len(entries),
        "deferred_reward_hack_count": len(reward_hack),
        "entries": entries,
    }


def emit(repo_root: Path = _REPO_ROOT) -> Path:
    out = repo_root / _DEFERRED_JSON.relative_to(_REPO_ROOT)
    out.write_text(json.dumps(build_deferred(repo_root), indent=2) + "\n", encoding="utf-8")
    data = build_deferred(repo_root)
    print(f"wrote {out} ({data['deferred_correctness_count']} deferred_correctness, "
          f"{data['deferred_reward_hack_count']} deferred_reward_hack)")
    return out


if __name__ == "__main__":
    emit(_REPO_ROOT)
