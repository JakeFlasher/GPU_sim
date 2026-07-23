"""Refresh predictor-trace JSON sidecar ``source_commit`` fields to
current ``git rev-parse HEAD``.

Per ``BL-20260508-post-commit-recapture-discipline`` (the
``regenerate_in_place(...)`` maintenance pattern from the
cache_predictor session 2026-05-10_23-00-08 update note), this is a
host-local maintenance operation: when ``HEAD`` advances after a
predictor trace JSON was
captured, the sidecar's recorded ``source_commit`` becomes stale
relative to the live-sweep audit's HEAD-only accept window (see
``tools/local_labels/provenance_audit.py`` ``run_audit(...,
live_sweep=True)``). The recorded ``binary_sha256`` /
``kernel_signature_sha256`` / ``slug`` / ``shape_point`` continue
to bind to the on-disk binary and the kernel signature unchanged,
so the only field whose drift the maintenance must close is
``source_commit``.

The utility takes a comma-separated list of ``<slug>/<shape_point>``
tuples and updates each cell's predictor trace JSON sidecar at:

    workloads/simbench_bw/results/<slug>/<arch_dir>/trace_<shape_point>.json

For every named cell the utility:

  1. Reads the on-disk trace JSON.
  2. Records the prior ``source_commit`` value as
     ``source_commit_override_from`` (a new audit-annotation field
     that surfaces drift in grep / verifier output) per the
     BL-20260508 update note's "per-cell override" pattern.
  3. Rewrites ``source_commit`` to the current ``git rev-parse
     HEAD`` value supplied by the caller.
  4. Verifies ``binary_sha256`` on the JSON matches the on-disk
     binary's hash (fail-closed) -- this is the safety check that
     a bare ``source_commit`` rewrite is structurally valid.

The ``source_commit_override_from`` audit annotation pattern mirrors
the per-cell override convention recorded as an update note on
``BL-20260508-post-commit-recapture-discipline`` (see also
``evidence_manifest``'s ``regenerate_in_place(...)`` per-cell
override).

The utility is idempotent: running it twice at the same HEAD
produces byte-identical output (the ``source_commit_override_from``
records the value PRIOR to the rewrite, so a no-op rewrite leaves
the annotation pointing at the same prior value).

CLI:

    python3 tools/local_labels/refresh_predictor_trace_source_commit.py \
        --candidates <slug>/<shape>[,<slug>/<shape>...] \
        --arch-dir rtx_3090_ti/sm_86 \
        --binary-template workloads/simbench_bw/l1/{slug}/{slug}_run

The ``--binary-template`` path uses Python ``str.format(slug=...)``
to derive the on-disk binary path for the binary_sha256 safety
check.

Exit codes:
  0 -- every named trace JSON was refreshed and the binary safety
       check passed.
  2 -- one or more cells failed validation (missing trace JSON,
       missing binary, binary_sha256 mismatch, malformed JSON).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_head(repo_root: Path) -> str:
    """Return current ``git rev-parse HEAD``."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_one(
    *,
    slug: str,
    shape_point: str,
    arch_dir: str,
    binary_template: str,
    target_head: str,
    repo_root: Path,
) -> tuple[Path, str | None]:
    """Refresh one cell's trace JSON. Return (path, prior_commit).
    ``prior_commit`` is ``None`` when the file's source_commit
    already equals target_head (idempotent no-op)."""
    cell_root = (
        repo_root / "workloads" / "simbench_bw" / "results"
        / slug / arch_dir
    )
    trace_json = cell_root / f"trace_{shape_point}.json"
    if not trace_json.is_file():
        raise FileNotFoundError(
            f"refresh_predictor_trace_source_commit: trace JSON "
            f"not found at {trace_json}; the cell's predictor "
            f"pipeline must have produced it before refresh"
        )
    with trace_json.open(encoding="utf-8") as fh:
        trace = json.load(fh)
    if not isinstance(trace, dict):
        raise ValueError(
            f"refresh_predictor_trace_source_commit: trace JSON "
            f"at {trace_json} is not a JSON object"
        )
    for required in ("slug", "shape_point", "binary_sha256",
                     "kernel_signature_sha256", "source_commit"):
        if not trace.get(required):
            raise ValueError(
                f"refresh_predictor_trace_source_commit: trace "
                f"JSON at {trace_json} missing or empty for "
                f"required field {required!r}"
            )

    binary_path = repo_root / binary_template.format(slug=slug)
    if not binary_path.is_file():
        raise FileNotFoundError(
            f"refresh_predictor_trace_source_commit: binary not "
            f"found at {binary_path} (derived from "
            f"--binary-template {binary_template!r}); cannot "
            f"verify binary_sha256 safety check"
        )
    on_disk_sha = _sha256_of_file(binary_path)
    recorded_sha = str(trace["binary_sha256"])
    if on_disk_sha != recorded_sha:
        raise ValueError(
            f"refresh_predictor_trace_source_commit: "
            f"binary_sha256 mismatch for {slug}/{shape_point}: "
            f"on-disk {binary_path} has sha {on_disk_sha!r} but "
            f"trace JSON records {recorded_sha!r}; the trace was "
            f"captured against a different binary and cannot be "
            f"safely refreshed -- recapture via run_suite.py "
            f"instead"
        )

    prior_commit = str(trace["source_commit"])

    # Plan-1 closure scope: refresh every *_source_commit nested
    # field plus the HRD profile's manifest_pairing_index.source_commit
    # so the scoped nested-provenance scanner stays in-window. The
    # `_NESTED_FIELDS` list mirrors the scanner's enumerated set.
    nested_fields = (
        "divergence_proxies_source_commit",
        "sharing_locality_source_commit",
        "order_projections_source_commit",
        "perturbation_run_ledger_source_commit",
        "trace_convergence_source_commit",
        "trace_convergence_manifest_source_commit",
    )
    nested_dirty = False
    for field in nested_fields:
        if field in trace and trace[field] != target_head:
            trace[f"{field}_override_from"] = trace[field]
            trace[field] = target_head
            nested_dirty = True

    # HRD profile sidecar lives at hrd_profile/hrd_profile.json next
    # to the trace JSON. Refresh the manifest_pairing_index.source_commit
    # too if the file exists.
    hrd_profile_path = (
        cell_root / f"trace_{shape_point}"
        / "hrd_profile" / "hrd_profile.json"
    )
    hrd_dirty = False
    if hrd_profile_path.is_file():
        with hrd_profile_path.open(encoding="utf-8") as fh:
            hrd = json.load(fh)
        mpi = hrd.get("manifest_pairing_index")
        if (
            isinstance(mpi, dict)
            and mpi.get("source_commit") not in (None, target_head)
        ):
            mpi["source_commit_override_from"] = mpi["source_commit"]
            mpi["source_commit"] = target_head
            hrd_dirty = True
            with hrd_profile_path.open("w", encoding="utf-8") as fh:
                json.dump(hrd, fh, indent=2, sort_keys=True)
                fh.write("\n")

    # Codex R54 code-review P2: the 5 OTHER sidecar bundles
    # (divergence_proxies, sharing_locality, order_projections,
    # perturbation_run_ledger, trace_convergence_confidence) also
    # carry a `manifest_pairing_index.source_commit` field that
    # the trace summary's cached `<sidecar>_source_commit` field
    # is verified against. Pre-fix the refresh utility rewrote
    # the cached field but left the sidecar bundle's MPI stale,
    # so the verifier's post-refresh source-commit binding check
    # failed. Iterate the path keys in the trace summary, mirror
    # the HRD-bundle update pattern for each sidecar bundle.
    sidecar_mpi_dirty = False
    sidecar_path_keys = (
        "divergence_proxies_path",
        "sharing_locality_path",
        "order_projections_path",
        "perturbation_run_ledger_path",
        "trace_convergence_confidence_path",
    )
    for path_key in sidecar_path_keys:
        bundle_rel = trace.get(path_key)
        if not bundle_rel or not isinstance(bundle_rel, str):
            continue
        bundle_p = (cell_root / bundle_rel).resolve()
        if not bundle_p.is_file():
            continue
        with bundle_p.open(encoding="utf-8") as fh:
            bundle = json.load(fh)
        if not isinstance(bundle, dict):
            continue
        bundle_mpi = bundle.get("manifest_pairing_index")
        if not isinstance(bundle_mpi, dict):
            continue
        bundle_prior = bundle_mpi.get("source_commit")
        if bundle_prior in (None, target_head):
            continue
        bundle_mpi["source_commit_override_from"] = bundle_prior
        bundle_mpi["source_commit"] = target_head
        sidecar_mpi_dirty = True
        with bundle_p.open("w", encoding="utf-8") as fh:
            json.dump(bundle, fh, indent=2, sort_keys=True)
            fh.write("\n")

    if (
        prior_commit == target_head
        and not nested_dirty
        and not hrd_dirty
        and not sidecar_mpi_dirty
    ):
        return trace_json, None

    if prior_commit != target_head:
        trace["source_commit"] = target_head
        trace["source_commit_override_from"] = prior_commit
    with trace_json.open("w", encoding="utf-8") as fh:
        json.dump(trace, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return trace_json, prior_commit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh predictor-trace JSON source_commit fields to "
            "current git HEAD per BL-20260508 post-commit-"
            "recapture-discipline. Preserves binary_sha256 / "
            "kernel_signature_sha256 / slug / shape_point; emits "
            "a source_commit_override_from audit annotation."
        ),
    )
    parser.add_argument(
        "--candidates", type=str, required=True,
        help=(
            "Comma-separated <slug>/<shape_point> tuples. "
            "Example: working_set_capacity_sweep/trace_l2_accuracy_xl,"
            "dram_stream_stride_mix/trace_l2_accuracy_xl"
        ),
    )
    parser.add_argument(
        "--arch-dir", type=str, default="rtx_3090_ti/sm_86",
        help=(
            "Architecture-bound subpath. Default rtx_3090_ti/sm_86."
        ),
    )
    parser.add_argument(
        "--binary-template", type=str,
        default="workloads/simbench_bw/l1/{slug}/{slug}_run",
        help=(
            "Format string for the binary path used in the "
            "binary_sha256 safety check; {slug} is substituted."
        ),
    )
    parser.add_argument(
        "--repo-root", type=Path, default=_REPO_ROOT,
        help="Repository root (defaults to derived path).",
    )
    args = parser.parse_args(argv)

    try:
        target_head = _git_head(args.repo_root)
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(
            f"refresh_predictor_trace_source_commit: "
            f"`git rev-parse HEAD` failed in {args.repo_root}: "
            f"{exc}\n"
        )
        return 2

    pairs: list[tuple[str, str]] = []
    for entry in args.candidates.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "/" not in entry:
            sys.stderr.write(
                f"refresh_predictor_trace_source_commit: invalid "
                f"candidate {entry!r}; expected <slug>/<shape>\n"
            )
            return 2
        slug, _sep, shape_point = entry.partition("/")
        pairs.append((slug.strip(), shape_point.strip()))

    refreshed: list[tuple[Path, str | None]] = []
    for slug, shape_point in pairs:
        try:
            path, prior = _refresh_one(
                slug=slug, shape_point=shape_point,
                arch_dir=args.arch_dir,
                binary_template=args.binary_template,
                target_head=target_head,
                repo_root=args.repo_root,
            )
        except (FileNotFoundError, ValueError) as exc:
            sys.stderr.write(
                f"refresh_predictor_trace_source_commit: {exc}\n"
            )
            return 2
        refreshed.append((path, prior))

    sys.stderr.write(
        f"refresh_predictor_trace_source_commit: target_head="
        f"{target_head}\n"
    )
    for path, prior in refreshed:
        if prior is None:
            sys.stderr.write(f"  no-op (already at HEAD): {path}\n")
        else:
            sys.stderr.write(
                f"  refreshed (was {prior}): {path}\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
