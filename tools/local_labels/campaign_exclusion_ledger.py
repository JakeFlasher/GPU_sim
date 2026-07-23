#!/usr/bin/env python3
"""Typed-exclusion authority ledger for the sm_86 live-capture corpus.

A single authority that types every non-captured cell into one closed reason and
defends the campaign denominator by an accounting identity:

  universe == runnable + pending_build + pending_reconstitution + Σ(typed exclusions)

It unifies the exclusion vocabularies scattered across the producers (simbench
arch_gate, the SOL fp8/nvfp4 dtype-contract gate, the NVBit trace-path
exclusions, the label repeatability/supersession set, and the SOL
correctness/host-capacity deferrals) into one seven-value taxonomy. The
unification map is recorded in the ledger itself so the routing is auditable.

This skeleton is populated from the roster authority's excluded rows — the
stable arch/capability gates knowable before any capture. The
repeatability/correctness/calibration/host-capacity/trace-unsupported reasons
are valid taxonomy members with empty populations here; later capture and
calibration work fills them, and the accounting identity re-checks in lockstep.

Usage:
  .venv/bin/python tools/local_labels/campaign_exclusion_ledger.py \
      --roster-authority data/local_labels/campaign/capture_roster_authority_sm86.json \
      --out data/local_labels/campaign/campaign_exclusion_ledger_sm86.json
Exit: 0 identity holds; 1 identity violated or an excluded row is untyped;
      2 emission failed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_ID = "sm86_capture_exclusion_ledger_v1"

# The closed taxonomy. A non-captured cell carries exactly one of these.
REASON_VOCABULARY = (
    "arch_gate",
    "semantic_mismatch",
    "trace_unsupported",
    "repeatability_failed",
    "correctness_failed",
    "calibration_inconclusive",
    "host_capacity_deferred",
)

# How the scattered producer vocabularies fold into the taxonomy. Recorded in
# the ledger for auditability; the entries below whose source is not yet
# populated are filled by the capture/calibration work that produces them.
SOURCE_VOCABULARY_MAP = {
    "arch_gate": [
        "simbench_bw:roster.toml arch_gate excludes sm_86",
        "sol_execbench:sm86_port_scope_v1.toml fp8_e4m3_contract",
        "sol_execbench:sm86_port_scope_v1.toml nvfp4_e2m1_contract",
    ],
    "trace_unsupported": [
        "nvbit:unsupported_memory_path",
        "nvbit:unmapped_memory_range",
        "nvbit:cross_allocation_lane_span",
        "nvbit:trace_volume_exceeded",
        "nvbit:slowdown_budget_exceeded",
        "nvbit:drop_rate_exceeded",
    ],
    "repeatability_failed": [
        "label_supersession:label_repeatability_failed",
        "label_supersession:label_repeatability_missing",
        "label_supersession:label_schema_admission_failed",
        "label:label_provenance_duplicate",
    ],
    "correctness_failed": [
        "sol_deferred:deferred_correctness (true wrong-answer)",
    ],
    "host_capacity_deferred": [
        "sol_deferred:deferred_correctness (24 GiB OOM, expected-fits-on-B200) "
        "— only for cells still unfittable after shrink-to-fit (DEC-2)",
    ],
    "calibration_inconclusive": [
        "calibration:inconclusive_undersampled / typed bounded gap",
    ],
    "semantic_mismatch": [
        "reserved: op-class semantically unrepresentable/untraceable on sm_86",
    ],
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head(repo_root: Path) -> str:
    import subprocess
    proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=str(repo_root))
    return proc.stdout.strip() if proc.returncode == 0 else ("0" * 40)


def build_ledger(authority: dict, authority_sha256: str,
                 repo_root: Path = REPO_ROOT) -> dict:
    entries = []
    untyped: list[str] = []
    for c in authority.get("cells", []):
        if c.get("resolution_status") != "excluded":
            continue
        reason = c.get("exclusion_reason")
        if reason not in REASON_VOCABULARY:
            untyped.append(f"{c.get('roster_source')}/{c.get('slug')}: "
                           f"reason={reason!r}")
            continue
        entries.append({
            "roster_source": c.get("roster_source"),
            "slug": c.get("slug"),
            "typed_reason": reason,
            "detail": c.get("exclusion_detail"),
        })

    counts_src = authority.get("counts", {})
    universe = counts_src.get("universe")
    runnable = counts_src.get("runnable")
    pending_build = counts_src.get("pending_build")
    pending_recon = counts_src.get("pending_reconstitution")
    pending_shrink = counts_src.get("pending_shrink", 0)
    typed_exclusions = len(entries)

    by_reason: dict[str, int] = {r: 0 for r in REASON_VOCABULARY}
    for e in entries:
        by_reason[e["typed_reason"]] += 1

    identity_holds = (
        isinstance(universe, int)
        and universe == (runnable or 0) + (pending_build or 0)
        + (pending_recon or 0) + (pending_shrink or 0) + typed_exclusions
        and not untyped
        # cross-check against the authority's own excluded tally
        and typed_exclusions == counts_src.get("excluded")
    )

    return {
        "schema_version": 1,
        "ledger_id": LEDGER_ID,
        "arch": authority.get("arch"),
        "sku": authority.get("sku"),
        "source_commit": _git_head(repo_root),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "roster_authority_id": authority.get("roster_id"),
        "roster_authority_sha256": authority_sha256,
        "reason_vocabulary": list(REASON_VOCABULARY),
        "source_vocabulary_map": SOURCE_VOCABULARY_MAP,
        "counts": {
            "total_exclusions": typed_exclusions,
            "by_reason": by_reason,
        },
        "coverage": {
            "universe": universe,
            "runnable": runnable,
            "pending_build": pending_build,
            "pending_reconstitution": pending_recon,
            "pending_shrink": pending_shrink,
            "typed_exclusions": typed_exclusions,
            "identity_holds": identity_holds,
            "untyped_excluded_rows": untyped,
        },
        "entries": sorted(entries, key=lambda e: (e["roster_source"], e["slug"])),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roster-authority", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    authority_path = Path(args.roster_authority)
    if not authority_path.is_absolute():
        authority_path = REPO_ROOT / authority_path
    if not authority_path.is_file():
        print(f"campaign_exclusion_ledger: roster authority absent: "
              f"{authority_path}", file=sys.stderr)
        return 2
    try:
        authority = json.loads(authority_path.read_text())
        ledger = build_ledger(authority, _sha256_file(authority_path))
    except Exception as exc:  # noqa: BLE001
        print(f"campaign_exclusion_ledger: build failed: {exc}", file=sys.stderr)
        return 2

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    except OSError as exc:
        print(f"campaign_exclusion_ledger: write failed: {exc}", file=sys.stderr)
        return 2

    cov = ledger["coverage"]
    print(f"exclusion ledger: typed_exclusions={cov['typed_exclusions']} "
          f"by_reason={ledger['counts']['by_reason']} -> {out}")
    if not cov["identity_holds"]:
        print("campaign_exclusion_ledger: COVERAGE IDENTITY VIOLATED "
              f"(untyped={cov['untyped_excluded_rows']})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
