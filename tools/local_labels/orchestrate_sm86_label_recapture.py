"""Per-cell sm_86 NCU label re-capture orchestrator.

The cross-arch same-74 closure plan's CRITICAL admission blocker
is that every binary on disk for the 44 sm_86 cohort cells has a
sha256 that does NOT match the `binary_sha256` field in
``data/local_labels/label_admitted.json``. The directed
resolution path (confirmed across multiple loop reviews) is to
RE-CAPTURE the sm_86 labels against the current binary set:
preserve the plan's strict-equality identity contract for
predictor bundles; refresh the ``binary_sha256`` field in
``label_admitted.json`` so the downstream cascade can admit
cells.

The capture workflow authorizes Nsight Compute (``ncu``) through the
 descriptor-bound label driver on the selected device and retains ground-truth
 labels only in the declared evidence namespace
 ``data/local_labels/``. The driver script that wraps ``ncu`` is
 ``tools/local_labels/label_driver.sh``.

This orchestrator is the per-cell surface called by the bulk
recapture sweep (a sibling driver to land in a subsequent round
of the loop). It:

1. Resolves the binary path for a `(slug, shape_point)` cell.
2. Resolves the per-shape app-arg vector via workload-specific
   conventions.
3. Invokes ``label_driver.sh`` ``n_repeats`` times (default 5)
   with ``launch_index=0`` each (the per-shape app-arg vector
   pins one launch per binary invocation; the repeat axis
   handles run-to-run variance for the repeatability gate).
4. Reports per-cell summary statistics (sidecars emitted,
   typed failures, wall-clock duration).

### Per-shape app-arg conventions

Simbench_bw l1 slugs (`workloads/simbench_bw/l1/<slug>/<slug>_run`):
the binary parses ``--shape-point <name>`` and dispatches to the
shape's correctness or timing flow based on the shape TOML. All
44 sm_86 cohort cells use ``trace_l2_accuracy_xl`` for these
slugs (per ``data/local_labels/label_admitted.json``).

Portable_lcd slugs (`workloads/portable_lcd/<slug>/<slug>_run`):
the binary's `runner_common::parse_args` accepts the suffixed
shape-mode flags ``--correctness_shape`` / ``--timing_shape``
(per `workloads/portable_lcd/runner_common.h` line 124-125),
plus the generic ``--shape-point <name>`` alias for
``trace_l2_accuracy_xl`` / ``trace_l2_accuracy_lg`` runs (per
line 137-141). The mapping per the cohort:

* ``timing_shape``         -> ``--timing_shape``
* ``correctness_shape``    -> ``--correctness_shape``
* ``trace_l2_accuracy_xl`` -> ``--shape-point trace_l2_accuracy_xl``
* ``trace_l2_accuracy_lg`` -> ``--shape-point trace_l2_accuracy_lg``

The ``--shape-point`` form does NOT take an additional
``--timing_shape`` companion flag; the parser routes the
generic shape directly into Timing mode internally.

The mapping is encoded in ``_app_args_for_cell`` below. A future
bulk-sweep driver iterates ``data/local_labels/sm86_cohort_preflight.json``
and calls ``recapture_cell`` per cell.

This module is read-only against the repository's tracked files;
it writes only to the declared evidence namespace
``data/local_labels/sm_86/<slug>/<shape_point>/`` (per
``label_driver.sh``'s ``--out-dir`` contract).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LABEL_DRIVER = _REPO_ROOT / "tools" / "local_labels" / "label_driver.sh"
_OUT_ROOT = _REPO_ROOT / "data" / "local_labels"

_SIMBENCH_L1_BINARY_ROOT = _REPO_ROOT / "workloads" / "simbench_bw" / "l1"
# Dual-workload l2 sibling tree (e.g. rmsnorm_fwd_bf16): same binary naming +
# the same --shape-point app-arg convention as the l1 tree.
_SIMBENCH_L2_BINARY_ROOT = _REPO_ROOT / "workloads" / "simbench_bw" / "l2"
_PORTABLE_LCD_BINARY_ROOT = _REPO_ROOT / "workloads" / "portable_lcd"


def _classify_workload(slug: str) -> str | None:
    """Return the owning workload tree for ``slug``: either
    ``"simbench_bw"`` (source under ``workloads/simbench_bw/l1/<slug>/``
    or the dual-workload l2 sibling ``workloads/simbench_bw/l2/<slug>/``,
    e.g. rmsnorm_fwd_bf16) or ``"portable_lcd"`` (source under
    ``workloads/portable_lcd/<slug>/``). Returns ``None`` if the slug
    source dir lives under none of the roots (a defect that should
    surface to the caller).
    """
    if (_SIMBENCH_L1_BINARY_ROOT / slug).is_dir():
        return "simbench_bw"
    if (_SIMBENCH_L2_BINARY_ROOT / slug).is_dir():
        return "simbench_bw"
    if (_PORTABLE_LCD_BINARY_ROOT / slug).is_dir():
        return "portable_lcd"
    return None


def _binary_path_for_cell(*, slug: str) -> Path | None:
    """Return the on-disk binary path for the slug, or ``None``
    when the slug source dir is not under either workload root.
    Binary naming is the same in both workloads:
    ``<slug>_run``.
    """
    workload = _classify_workload(slug)
    if workload is None:
        return None
    if workload == "simbench_bw":
        l1 = _SIMBENCH_L1_BINARY_ROOT / slug / f"{slug}_run"
        if l1.is_file() or (_SIMBENCH_L1_BINARY_ROOT / slug).is_dir():
            return l1
        return _SIMBENCH_L2_BINARY_ROOT / slug / f"{slug}_run"
    return _PORTABLE_LCD_BINARY_ROOT / slug / f"{slug}_run"


def _app_args_for_cell(*, slug: str, shape_point: str) -> list[str]:
    """Resolve the workload-specific app-arg vector forwarded to
    the profiled binary after ``label_driver.sh``'s ``--``
    sentinel.

    See the module docstring for the per-shape mapping rationale.
    Raises ``ValueError`` when the slug-shape pair is not
    recognized so the caller fails closed rather than running NCU
    against a binary that will silently miss the requested shape.
    """
    workload = _classify_workload(slug)
    if workload is None:
        raise ValueError(
            f"unknown workload for slug={slug!r}; expected source "
            f"under workloads/simbench_bw/l1/ or "
            f"workloads/portable_lcd/"
        )
    # label_driver.sh's shape-mode validation rejects the generic
    # `--shape-point <name>` form when ``--shape-point`` would name
    # one of the two special shapes (`timing_shape` /
    # `correctness_shape`). Both workloads' binaries accept the
    # suffixed flag form (`--timing_shape` / `--correctness_shape`)
    # and the driver requires it for the special shapes. The
    # generic ``--shape-point <name>`` form remains the right
    # surface for non-special shapes
    # (`trace_l2_accuracy_xl`, etc.).
    if shape_point == "timing_shape":
        return ["--timing_shape"]
    if shape_point == "correctness_shape":
        return ["--correctness_shape"]
    if workload == "simbench_bw":
        # SimBench binaries do NOT derive their launch axes from the
        # ``--shape-point`` token: the token is a label-only convention,
        # and the geometry comes from explicit per-axis flags
        # (``--warps``, ``--allocation-bytes``, ``--m/--n/--k``, ...).
        # Forwarding only ``--shape-point`` profiles the binary's
        # COMPILED DEFAULTS and mis-attributes the resulting NCU ground
        # truth to the canonical shape (the carveout_sensitivity_probe
        # mislabel: default 4-warp/4KiB launch labeled as the 64-warp/
        # 16MiB trace_l2_accuracy_xl cell). Resolve the canonical axes
        # from workloads/simbench_bw/shapes/<slug>.toml exactly as the
        # full-matrix sweep driver does — the SAME helpers, so the two
        # capture paths cannot drift. A slug without a SimBench TOML
        # falls back to label-only forwarding; a TOML without the
        # requested shape entry fails closed (ValueError) rather than
        # silently profiling defaults.
        from sweep_repeatability_matrix import (
            _axes_to_workload_cli_args,
            _load_simbench_shape_axes,
        )
        axes = _load_simbench_shape_axes(slug, shape_point)
        args = ["--shape-point", shape_point]
        if axes:
            args += _axes_to_workload_cli_args(axes)
        return args
    if shape_point.startswith("trace_l2_accuracy_"):
        # The generic --shape-point alias routes directly into
        # Timing mode inside runner_common::parse_args; a
        # companion --timing_shape flag is NOT required and
        # would over-set the mode discriminator.
        return ["--shape-point", shape_point]
    raise ValueError(
        f"unrecognized portable_lcd shape_point={shape_point!r} "
        f"for slug={slug!r}; expected timing_shape, correctness_shape, "
        f"or trace_l2_accuracy_*"
    )


def _binary_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _label_sidecar_paths(
    *, slug: str, shape_point: str, n_repeats: int,
) -> list[Path]:
    """Return the expected on-disk paths for the per-launch
    sidecars ``label_driver.sh`` should write for this cell.
    The driver writes ``label_<repeat>_<launch>.json``; each
    invocation forwards ``--launch 0``.
    """
    base = _OUT_ROOT / "sm_86" / slug / shape_point
    return [base / f"label_{repeat}_0.json" for repeat in range(n_repeats)]


def recapture_cell(
    *,
    slug: str,
    shape_point: str,
    n_repeats: int = 5,
    warmup: int = 1,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Drive ``label_driver.sh`` against one cell.

    Returns a per-cell report dict with:
    * ``cell_key``: ``"<slug>/<shape_point>"``.
    * ``binary_path``: resolved binary path (relative to repo root).
    * ``binary_sha256_observed``: sha256 of the on-disk binary.
    * ``repeats_attempted`` / ``repeats_succeeded`` / ``repeats_failed``.
    * ``sidecar_paths``: list of per-launch sidecar paths
      (whether or not the driver actually wrote them).
    * ``wall_clock_seconds``: total elapsed time across all repeats.
    * ``typed_failure_reason``: optional, populated when the cell
      could not be invoked at all (e.g., binary missing, app-arg
      resolution failed).
    * ``per_repeat``: list of per-repeat status dicts
      (``repeat``, ``returncode``, ``stderr_tail``).

    ``dry_run`` skips the actual ``label_driver.sh`` invocation
    and returns a report shape with ``per_repeat[*].status='dry_run_skipped'``.
    Useful for testing app-arg resolution without invoking NCU.
    """
    cell_key = f"{slug}/{shape_point}"
    binary_path = _binary_path_for_cell(slug=slug)
    if binary_path is None or not binary_path.is_file():
        return {
            "cell_key": cell_key,
            "binary_path": str(binary_path) if binary_path else None,
            "binary_sha256_observed": None,
            "repeats_attempted": 0,
            "repeats_succeeded": 0,
            "repeats_failed": 0,
            "sidecar_paths": [],
            "wall_clock_seconds": 0.0,
            "typed_failure_reason": (
                "binary_missing_on_disk"
                if binary_path is None
                else f"binary_missing_on_disk:{binary_path}"
            ),
            "per_repeat": [],
        }
    try:
        app_args = _app_args_for_cell(slug=slug, shape_point=shape_point)
    except ValueError as exc:
        return {
            "cell_key": cell_key,
            "binary_path": str(binary_path.relative_to(_REPO_ROOT)),
            "binary_sha256_observed": _binary_sha256(binary_path),
            "repeats_attempted": 0,
            "repeats_succeeded": 0,
            "repeats_failed": 0,
            "sidecar_paths": [],
            "wall_clock_seconds": 0.0,
            "typed_failure_reason": f"app_arg_resolution_failed:{exc!s}",
            "per_repeat": [],
        }
    sidecar_paths = _label_sidecar_paths(
        slug=slug, shape_point=shape_point, n_repeats=n_repeats,
    )
    binary_sha = _binary_sha256(binary_path)
    per_repeat: list[dict[str, Any]] = []
    start = time.monotonic()
    succeeded = 0
    failed = 0
    for repeat in range(n_repeats):
        if dry_run:
            per_repeat.append({
                "repeat": repeat,
                "status": "dry_run_skipped",
                "returncode": None,
                "stderr_tail": None,
            })
            continue
        # Pre-clean any stale NCU export from a prior invocation
        # so the driver's ``ncu --export <path>`` does not fail
        # closed on "File already exists" when this orchestrator
        # is invoked repeatedly (e.g., a bulk-sweep re-run).
        # The driver writes the report path as
        # ``<out-dir>/sm_86/<slug>/<shape>/run_<repeat>_<launch>.ncu-rep``.
        cell_dir = _OUT_ROOT / "sm_86" / slug / shape_point
        stale_report = cell_dir / f"run_{repeat}_0.ncu-rep"
        if stale_report.is_file():
            try:
                stale_report.unlink()
            except OSError:
                pass
        cmd = [
            str(_LABEL_DRIVER),
            "--device", "sm_86",
            "--slug", slug,
            "--shape-point", shape_point,
            "--repeat", str(repeat),
            "--launch", "0",
            "--binary", str(binary_path),
            "--warmup", str(warmup),
            "--out-dir", str(_OUT_ROOT),
            "--",
            *app_args,
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=str(_REPO_ROOT), timeout=600,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            per_repeat.append({
                "repeat": repeat,
                "status": "subprocess_failed",
                "returncode": None,
                "stderr_tail": f"{type(exc).__name__}:{exc!s}",
            })
            failed += 1
            continue
        if proc.returncode == 0:
            per_repeat.append({
                "repeat": repeat,
                "status": "ok",
                "returncode": 0,
                "stderr_tail": proc.stderr[-400:] if proc.stderr else "",
            })
            succeeded += 1
        else:
            per_repeat.append({
                "repeat": repeat,
                "status": "driver_returncode_nonzero",
                "returncode": proc.returncode,
                "stderr_tail": proc.stderr[-400:] if proc.stderr else "",
            })
            failed += 1
    wall_clock = time.monotonic() - start
    # Derive a typed failure reason when every attempted repeat
    # failed. The per-repeat stderr tails carry the root cause;
    # surface a compact token + a short failure evidence string
    # so reviewers do not need to inspect the declared evidence namespace's
    # retained .ncu-rep files directly while triaging the outcome. Recognized
    # error families:
    #
    #   - "CPU ISA level is lower than required" -> the binary
    #     was built on a newer-CPU host and won't launch on
    #     this CPU. Classifies as
    #     ``binary_cpu_isa_incompatible``; the operator
    #     remediation is to rebuild the binary on this host
    #     with the project's per-slug Makefile.
    #
    # Any other unrecognized error pattern falls through to a
    # generic ``all_repeats_failed:<first_stderr_tail>`` token
    # so the audit ledger never carries ``null`` for a row that
    # actually failed.
    typed_failure_reason: str | None = None
    failure_evidence: str | None = None
    if succeeded == 0 and failed > 0:
        first_stderr = ""
        for entry in per_repeat:
            stderr = entry.get("stderr_tail") or ""
            if stderr.strip():
                first_stderr = stderr.strip()
                break
        failure_evidence = first_stderr[:256] if first_stderr else None
        if "CPU ISA level is lower than required" in first_stderr:
            typed_failure_reason = "binary_cpu_isa_incompatible"
        elif first_stderr:
            typed_failure_reason = (
                "all_repeats_failed:" + first_stderr[:128]
            )
        else:
            typed_failure_reason = "all_repeats_failed:no_stderr_captured"
    return {
        "cell_key": cell_key,
        "binary_path": str(binary_path.relative_to(_REPO_ROOT)),
        "binary_sha256_observed": binary_sha,
        "repeats_attempted": n_repeats,
        "repeats_succeeded": succeeded,
        "repeats_failed": failed,
        "sidecar_paths": [
            str(p.relative_to(_REPO_ROOT))
            if str(p).startswith(str(_REPO_ROOT)) else str(p)
            for p in sidecar_paths
        ],
        "wall_clock_seconds": round(wall_clock, 3),
        "typed_failure_reason": typed_failure_reason,
        "failure_evidence": failure_evidence,
        "per_repeat": per_repeat,
        "attempted_at_utc": _dt.datetime.now(tz=_dt.timezone.utc)
            .isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slug", required=True,
        help="Cell slug from data/local_labels/sm86_cohort_preflight.json.",
    )
    parser.add_argument(
        "--shape-point", required=True,
        help="Shape point (e.g., trace_l2_accuracy_xl, timing_shape).",
    )
    parser.add_argument(
        "--n-repeats", type=int, default=5,
        help="Number of NCU repeats per cell (default 5).",
    )
    parser.add_argument(
        "--warmup", type=int, default=1,
        help="Warmup launches before measurement (default 1).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Resolve the binary path + app args + emit the report "
            "shape WITHOUT invoking NCU. Useful for verifying the "
            "per-shape app-arg resolution path."
        ),
    )
    args = parser.parse_args(argv)
    report = recapture_cell(
        slug=args.slug,
        shape_point=args.shape_point,
        n_repeats=args.n_repeats,
        warmup=args.warmup,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if (
        report["typed_failure_reason"] is None
        and report["repeats_failed"] == 0
    ) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
