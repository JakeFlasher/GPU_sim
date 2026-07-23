#!/usr/bin/env python3
"""Read-only regression validator for the frozen M1 sm_86 label+trace bundle.

Re-derives, from the machine-local evidence roots the bundle names, every pinned
identity in ``eval/frozen_labels/sm86_m1_bundle_v21.json`` — the per-cell label
sidecar sha256s, the trace payload identity (dynamic_trace.pb + stats.csv +
threadblocks manifest), the admitted headline means, the L1TEX sector-hit label,
and the admitted/typed-excluded counts — and asserts they still reproduce
exactly. This is the pre-flip regression anchor: any silent re-capture, tampered
pin, or premature contract change trips it.

Derivation mirrors ``freeze_label_bundle.py`` byte-for-byte (same streamed
sha256, same sidecar glob, same threadblocks (relpath, size) manifest hash), so
a validated run means the committed bundle and the evidence are the same object
the freezer sealed. It never writes and never touches the GPU.

The evidence lives in the gitignored annex, so it is machine-local: on a host
without it the validator reports ``evidence_absent`` (a distinct non-zero exit),
never a false pass.

Usage:
  .venv/bin/python tools/local_labels/validate_m1_anchor.py
Exit: 0 anchor reproduces; 1 anchor DRIFTED (a pin/count mismatch); 2 read
      error; 3 machine-local evidence absent (cannot validate on this host).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = REPO_ROOT / "eval" / "frozen_labels" / "sm86_m1_bundle_v21.json"
REF_CFG = "cfg=clock_none__cache_all"
# The full headline tuple the freezer pins (mean + dispersion + repeatability
# rule each). All three are re-derived and checked — cycles and duration are the
# primary NCU timing targets, so leaving them pinned-but-unverified would let a
# re-aggregated mean drift past the anchor silently.
HEADLINE = (
    "gpc__cycles_elapsed.max",
    "gpu__time_duration.sum",
    "smsp__inst_executed.sum",
)
HEADLINE_EXACT = "smsp__inst_executed.sum"
CACHE_LABELS = (
    "cache_hit_rate_l1tex_sector_pct",
    "cache_hit_rate_l2_sector_pct",
)
EXPECTED_ADMITTED = 15
EXPECTED_EXCLUDED = 2
EXPECTED_SIDECARS = 5


class EvidenceAbsent(Exception):
    """Raised when a machine-local evidence root the bundle names is missing."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _threadblocks_manifest(trace_folder: Path) -> tuple[str, int]:
    tb_root = trace_folder / "threadblocks"
    tb_entries = sorted(
        (str(f.relative_to(tb_root)), f.stat().st_size)
        for f in tb_root.rglob("*.pb")
    ) if tb_root.is_dir() else []
    sha = hashlib.sha256(json.dumps(tb_entries).encode()).hexdigest()
    return sha, len(tb_entries)


def validate_anchor(bundle_path: Path = DEFAULT_BUNDLE,
                    repo_root: Path = REPO_ROOT) -> list[str]:
    """Return a list of drift findings (empty == anchor reproduces).

    Raises ``EvidenceAbsent`` if a named machine-local evidence root is missing.
    """
    bundle = json.loads(bundle_path.read_text())
    findings: list[str] = []

    cells = bundle.get("cells", [])
    if len(cells) != EXPECTED_ADMITTED or bundle.get("n_cells") != EXPECTED_ADMITTED:
        findings.append(
            f"admitted count {len(cells)}/n_cells={bundle.get('n_cells')} != "
            f"{EXPECTED_ADMITTED}")
    if len(bundle.get("excluded_cells", [])) != EXPECTED_EXCLUDED:
        findings.append(
            f"excluded count {len(bundle.get('excluded_cells', []))} != "
            f"{EXPECTED_EXCLUDED}")
    # The freezer writes the bundle even when it flagged problems (and returns
    # non-zero); a green anchor requires the sealed bundle carried none.
    if bundle.get("problems"):
        findings.append(
            f"bundle carries {len(bundle['problems'])} unresolved problems: "
            f"{bundle['problems'][:3]}")

    labels_root = repo_root / bundle["labels_root"]
    trace_manifest_path = repo_root / bundle["trace_manifest"]
    if not labels_root.is_dir():
        raise EvidenceAbsent(f"labels_root absent: {labels_root}")
    if not trace_manifest_path.is_file():
        raise EvidenceAbsent(f"trace_manifest absent: {trace_manifest_path}")

    admitted_path = labels_root / REF_CFG / "agg" / "label_admitted.json"
    excluded_path = labels_root / REF_CFG / "agg" / "label_excluded.json"
    if not admitted_path.is_file() or not excluded_path.is_file():
        raise EvidenceAbsent(
            f"aggregated label evidence absent under {labels_root / REF_CFG}")
    admitted = json.loads(admitted_path.read_text())
    excluded = json.loads(excluded_path.read_text())
    trace_manifest = json.loads(trace_manifest_path.read_text())
    admitted_by_slug = {c["slug"]: c for c in admitted.get("cells", [])}
    excluded_by_slug = {c["slug"]: c for c in excluded.get("cells", [])}
    trace_by_slug = {c["slug"]: c for c in trace_manifest.get("cells", [])}

    exact_headline_ok = 0
    l1tex_ok = 0
    for cell in cells:
        slug = cell["slug"]
        src = admitted_by_slug.get(slug)
        if src is None:
            findings.append(f"{slug}: absent from machine-local label_admitted")
            continue

        # Label sidecar sha256 pins — the full v2.1 repeat set, not an empty dict.
        label_dir = (labels_root / REF_CFG / "labels" / cell["arch"] / slug
                     / cell["shape_point"])
        sidecars = [p for p in sorted(label_dir.glob("label_*_*.json"))
                    if "run_manifest" not in p.name]
        rederived = {p.name: _sha256(p) for p in sidecars}
        pinned_sidecars = cell.get("label_sidecar_sha256s") or {}
        if len(pinned_sidecars) != EXPECTED_SIDECARS:
            findings.append(
                f"{slug}: pins {len(pinned_sidecars)} label sidecars != "
                f"{EXPECTED_SIDECARS}")
        if rederived != pinned_sidecars:
            findings.append(f"{slug}: label sidecar sha256s drifted")

        # Headline tuple — all three metrics, mean + dispersion + rule each.
        by_metric = {m.get("ncu_metric"): m
                     for m in (src.get("unit_tagged_metrics") or [])}
        for metric in HEADLINE:
            pinned = cell.get("headline_admitted", {}).get(metric)
            m = by_metric.get(metric)
            rederived_m = None if m is None else {
                "mean": m.get("value"),
                "dispersion": m.get("dispersion"),
                "repeatability_rule": m.get("repeatability_rule"),
            }
            if pinned is not None and pinned == rederived_m:
                if metric == HEADLINE_EXACT:
                    exact_headline_ok += 1
            else:
                findings.append(
                    f"{slug}: headline {metric} rederived={rederived_m} != "
                    f"pinned={pinned}")

        # Cache sector-hit labels — both L1TEX and L2.
        for label in CACHE_LABELS:
            if src.get(label) == cell.get(label):
                if label == "cache_hit_rate_l1tex_sector_pct":
                    l1tex_ok += 1
            else:
                findings.append(
                    f"{slug}: {label} drifted "
                    f"({src.get(label)} != {cell.get(label)})")

        # Trace payload identity — an admitted cell must carry a trace pin.
        tr_src = trace_by_slug.get(slug)
        tr_pin = cell.get("trace") or {}
        if not tr_pin:
            findings.append(f"{slug}: admitted cell carries no trace identity pin")
        if tr_src is None:
            findings.append(f"{slug}: absent from machine-local trace manifest")
        elif tr_pin:
            trace_folder = repo_root / tr_src["traces_folder"] \
                if tr_src.get("traces_folder") and not Path(tr_src["traces_folder"]).is_absolute() \
                else Path(tr_src.get("traces_folder", ""))
            if not trace_folder.is_dir():
                findings.append(f"{slug}: trace folder absent: {trace_folder}")
            else:
                pb = trace_folder / "dynamic_trace.pb"
                st = trace_folder / "stats.csv"
                pb_sha = _sha256(pb) if pb.is_file() else None
                st_sha = _sha256(st) if st.is_file() else None
                tb_sha, tb_n = _threadblocks_manifest(trace_folder)
                if pb_sha != tr_pin.get("dynamic_trace_pb_sha256"):
                    findings.append(f"{slug}: dynamic_trace.pb sha256 drifted")
                if st_sha != tr_pin.get("stats_csv_sha256"):
                    findings.append(f"{slug}: stats.csv sha256 drifted")
                if tb_sha != tr_pin.get("threadblocks_manifest_sha256"):
                    findings.append(f"{slug}: threadblocks manifest sha256 drifted")
                if tb_n != tr_pin.get("threadblocks_file_count"):
                    findings.append(
                        f"{slug}: threadblocks_file_count {tb_n} != "
                        f"{tr_pin.get('threadblocks_file_count')}")
            # Label<->trace binary identity (equality, not substring).
            if tr_src.get("binary_sha256") and cell.get("binary_sha256") and \
                    tr_src["binary_sha256"] != cell["binary_sha256"]:
                findings.append(f"{slug}: trace binary_sha256 != label binary_sha256")

    if exact_headline_ok != EXPECTED_ADMITTED:
        findings.append(
            f"{HEADLINE_EXACT} exact only {exact_headline_ok}/{EXPECTED_ADMITTED}")
    if l1tex_ok != EXPECTED_ADMITTED:
        findings.append(
            f"L1TEX sector-hit label exact only {l1tex_ok}/{EXPECTED_ADMITTED}")

    # Typed-excluded set reproduces (slug + typed_reason).
    for ex in bundle.get("excluded_cells", []):
        src = excluded_by_slug.get(ex.get("slug"))
        if src is None:
            findings.append(f"{ex.get('slug')}: excluded cell absent from evidence")
        elif src.get("typed_reason") != ex.get("typed_reason"):
            findings.append(
                f"{ex.get('slug')}: typed_reason {src.get('typed_reason')!r} != "
                f"pinned {ex.get('typed_reason')!r}")

    return findings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    args = ap.parse_args(argv)
    bundle_path = Path(args.bundle)
    if not bundle_path.is_absolute():
        bundle_path = REPO_ROOT / bundle_path
    if not bundle_path.is_file():
        print(f"validate_m1_anchor: bundle absent: {bundle_path}", file=sys.stderr)
        return 2

    try:
        findings = validate_anchor(bundle_path)
    except EvidenceAbsent as exc:
        print(f"validate_m1_anchor: evidence_absent — {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001
        print(f"validate_m1_anchor: read error: {exc}", file=sys.stderr)
        return 2

    if not findings:
        print(f"validate_m1_anchor: PASS — anchor reproduces "
              f"({EXPECTED_ADMITTED} admitted / {EXPECTED_EXCLUDED} excluded; "
              f"{HEADLINE_EXACT} exact {EXPECTED_ADMITTED}/{EXPECTED_ADMITTED}; "
              f"all sha256 pins reproduce)")
        return 0
    print(f"validate_m1_anchor: DRIFT — {len(findings)} finding(s):", file=sys.stderr)
    for f in findings:
        print(f"  {f}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
