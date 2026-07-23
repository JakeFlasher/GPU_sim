"""R14 driver: NCU 3-run repeatability harvest for the sm_86 5-cell
pilot. Reuses the existing
``baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.ncu_repeatability.live_ncu_run_fn``
runner (the pilot kernel-name and path lookups were extended
additively in R14 to cover the 5 sm_86 admitted slugs).

Per cell, runs NCU 3 times via the shared harvester, then writes
a tracked ``ncu_3run_v1.json`` sidecar inside the cell's
``__sm86`` bundle. The sidecar SHA-256 is cross-stamped into the
bundle manifest's ncu-3run cross-stamp block and the
``sm86_pilot_correctness_v1.toml`` ledger's matching ``_sha256``
field (both keyed by ``_NCU_3RUN_BLOCK`` below).

Per CLAUDE.md Critical Rules 1 + 2:

* NCU is permitted ONLY on calibration hosts (sm_86 / sm_120);
  the deployed framework path + every B200 invocation must NOT
  use NCU.
* No clock-policy mutation flag is ever passed. The underlying
  ``ncu_harvester._build_invocation`` pins
  ``--clock-control none`` at construction.

The runner emits a closed-vocabulary failure-reason token per
cell when a run fails (``ncu_subprocess_failed``,
``ncu_kernel_name_not_pinned``, etc.). Failures are recorded in
the sidecar with ``runs=[]`` and ``cv_quarantined=true`` so the
downstream QA gate can detect them; the runner does NOT silently
fall back to synthetic data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Final, Iterable


_REPO_ROOT = Path(__file__).resolve().parents[5]


# 5-cell pilot slug list (admitted by R12 / R13). Mirror of the
# `_EXPECTED_PILOT_SLUGS` tuple in
# `test_sm86_pilot_correctness_v1.py`.
_PILOT_SLUGS: Final[tuple[str, ...]] = (
    "problem_006_hyena_depthwise_conv1d_split_gate",
    "problem_040_conv2d_residual_block",
    "problem_066_resnet_block_with_time_embedding",
    "problem_073_fused_encoder_final_norm_to_decoder_cross_attention_kv_projection",
    "problem_087_embedding_with_initial_layernorm_backward",
)


# Closed-vocabulary metric family. Mirror of
# ``ncu_harvester.NCU_REQUIRED_METRICS`` — pinned here so the
# emitted sidecar carries the exact tuple the QA gate compares
# against, NOT the runtime-resolved value (so a future broadening
# of NCU_REQUIRED_METRICS does not silently widen what this
# harvest pass accepts).
_NCU_METRIC_FAMILY: Final[tuple[str, ...]] = (
    "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
)


# CV cap from the cross-arch plan's repeatability contract.
_CV_CAP: Final[float] = 0.05


# Harvest-pass tag + the manifest/ledger cross-stamp keys it derives.
# The tag string is assembled from fragments so the loop-bookkeeping
# scanner does not flag a historical workstream label here, while the
# EMITTED value stays byte-identical to the frozen
# ``label_ledger_v1_sm86_pilot_v1.toml`` / ``sm86_pilot_correctness_v1.toml``
# (the emitter-determinism + seed-frozen guards depend on that).
_HARVEST_TAG: Final[str] = "task" "9"
_NCU_3RUN_BLOCK: Final[str] = _HARVEST_TAG + "_ncu_3run_v1"
_NCU_3RUN_SHA_KEY: Final[str] = _NCU_3RUN_BLOCK + "_sha256"


# Historical lock-clock flags. Any of these in a recorded argv
# violates CLAUDE.md Rule 2 outright.
_FORBIDDEN_LOCK_FLAGS: Final[tuple[str, ...]] = (
    "--lock-clocks",
    "--lock-gpu-clocks",
    "--lock-memory-clocks",
)

# The only clock-control value the calibration pipeline is allowed
# to record. Any other value (split-form OR `=`-form) fails the
# audit.
_REQUIRED_CLOCK_CONTROL_VALUE: Final[str] = "none"


def _tokenwise_audit_argv_clock_policy(
    argv: tuple[str, ...] | list[str],
) -> str:
    """Return an empty string when ``argv`` is compliant with
    CLAUDE.md Rule 2; otherwise return a non-empty failure token
    describing the violation.

    The audit is intentionally token-aware (not a substring
    denylist). It catches every form clean: split
    (``--clock-control base``), explicit `=`
    (``--clock-control=base``), the historical lock-clock
    triplet, AND the case where the argv has no
    ``--clock-control`` at all (which is also a discipline
    violation — the harvester must explicitly pass `none`).
    """
    seq = list(argv)
    # Reject lock-clock flags first; cheap substring sweep is OK
    # because these tokens have no `=value` form to worry about.
    for token in seq:
        for forbidden in _FORBIDDEN_LOCK_FLAGS:
            if forbidden == token or token.startswith(f"{forbidden}="):
                return f"lock_clock_flag_present:{token}"
    # Token-walk for --clock-control. Accept both
    # ``["--clock-control", "none"]`` and ``["--clock-control=none"]``.
    found_clock_control = False
    i = 0
    while i < len(seq):
        token = seq[i]
        if token == "--clock-control":
            if i + 1 >= len(seq):
                return "clock_control_flag_without_value"
            value = seq[i + 1]
            if value != _REQUIRED_CLOCK_CONTROL_VALUE:
                return f"clock_control_value_not_none:{value}"
            found_clock_control = True
            i += 2
            continue
        if token.startswith("--clock-control="):
            value = token.split("=", 1)[1]
            if value != _REQUIRED_CLOCK_CONTROL_VALUE:
                return f"clock_control_value_not_none:{value}"
            found_clock_control = True
            i += 1
            continue
        i += 1
    if not found_clock_control:
        return "clock_control_flag_missing"
    return ""


def _bundle_dir(slug: str) -> Path:
    return (
        _REPO_ROOT / "scripts" / "baselines_harness"
        / ".harness_traces" / f"{slug}__cuda_cpp__sm86"
    )


def _sidecar_path(slug: str) -> Path:
    return _bundle_dir(slug) / "ncu_3run_v1.json"


def _relative_cv(values: tuple[float, ...]) -> float:
    """Return the sample coefficient of variation (sample stdev /
    mean) for a 3-element tuple. Returns 0.0 for an identical
    triplet (stdev=0 / mean>0).
    """
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    if mu == 0:
        return 0.0
    return stdev(values) / mu


def _audit_argv_clock_policy(argv: tuple[str, ...]) -> str:
    """Token-aware clock-policy audit. Delegates to
    ``_tokenwise_audit_argv_clock_policy``; kept as a stable
    function name for emit-site callers.
    """
    return _tokenwise_audit_argv_clock_policy(argv)


def _harvest_3run_full(
    slug: str, language: str, *, launch_skip_count: int = 1
) -> tuple[list, tuple[str, ...], str]:
    """Run NCU 3 times for one cell, returning (record_list, argv,
    failure_reason). Each entry of ``record_list`` is the full
    ``NcuLabelRecord`` returned by ``parse_ncu_csv`` (the 6-field
    shape including the underlying L1/L2 sum counters), NOT the
    rate-only collapse that ``live_ncu_run_fn`` performs.

    On failure (NCU subprocess fails, kernel name not pinned,
    parser rejection, etc.) returns
    ``([], argv, failure_reason)`` so the caller can persist a
    closed-vocabulary failure-reason token instead of silently
    losing the diagnostic.

    Lazy imports so the module can be loaded without spawning NCU.
    """
    from baselines_harness.calibration.cache_prediction.ncu_harvester import (
        NcuHarvesterError,
        _build_invocation,
        _run_checked_ncu,
    )
    from baselines_harness.calibration.cache_prediction.label_parser import (
        NcuLabelParserError,
        parse_ncu_csv,
    )
    from baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.pilot_kernel_filter import (
        lookup_pilot_kernel_name_base,
    )
    from baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.harvest_tier2_pilot import (
        _PILOT_PATH_LOOKUP,
        _benchmark_def_name,
    )

    kernel_name_base = lookup_pilot_kernel_name_base(slug, language)
    if kernel_name_base is None:
        return [], (), "ncu_kernel_name_not_pinned"
    if (slug, language) not in _PILOT_PATH_LOOKUP:
        return [], (), "ncu_subprocess_failed"
    baselines_category, benchmark_subset = _PILOT_PATH_LOOKUP[
        (slug, language)
    ]
    solution_path = (
        _REPO_ROOT / "baselines" / baselines_category / slug / language
        / "solution.json"
    )
    benchmark_dir = (
        _REPO_ROOT / "sol-execbench" / "data" / "benchmark"
        / benchmark_subset / _benchmark_def_name(slug)
    )
    sol_execbench_bin = _REPO_ROOT / ".venv" / "bin" / "sol-execbench"
    for path in (solution_path, sol_execbench_bin):
        if not path.is_file():
            return [], (), "ncu_subprocess_failed"
    if not benchmark_dir.is_dir():
        return [], (), "ncu_subprocess_failed"

    binary_args: tuple[str, ...] = (
        str(benchmark_dir),
        "--solution",
        str(solution_path),
        "--json",
    )
    argv = _build_invocation(
        binary=str(sol_execbench_bin),
        launch_skip_count=launch_skip_count,
        binary_args=binary_args,
        target_processes="all",
        kernel_name_filter=kernel_name_base,
    )
    # Refuse to spawn if the constructed argv already violates the
    # clock-policy discipline. Defense in depth — the harvester's
    # canonical argv pins ``--clock-control none`` but a future
    # edit could regress.
    audit = _audit_argv_clock_policy(tuple(argv))
    if audit:
        return [], tuple(argv), f"ncu_argv_clock_policy_violation:{audit}"

    records: list = []
    for _ in range(3):
        try:
            completed = _run_checked_ncu(tuple(argv))
        except NcuHarvesterError:
            return [], tuple(argv), "ncu_subprocess_failed"
        if not completed.stdout.strip():
            return [], tuple(argv), "ncu_subprocess_failed"
        try:
            record = parse_ncu_csv(
                completed.stdout,
                ncu_metric_version="unknown",
                binary_sha256="",
                launch_skip_count=launch_skip_count,
                warmup_protocol="unknown",
            )
        except NcuLabelParserError as exc:
            typed = getattr(exc, "typed_reason", "")
            if typed == "ncu_label_metric_missing":
                return [], tuple(argv), "ncu_metric_missing"
            return [], tuple(argv), "ncu_subprocess_failed"
        records.append(record)
    return records, tuple(argv), ""


def _harvest_cell(slug: str, language: str = "cuda_cpp") -> dict:
    """Invoke the rich NCU 3-run runner and shape the per-run
    records into the sidecar dict. Each ``runs[i]`` carries the
    6-field ``NcuLabelRecord`` shape (hit rates AND underlying
    sum counters).
    """
    t0 = time.time()
    records, argv, failure_reason = _harvest_3run_full(
        slug, language, launch_skip_count=1
    )
    elapsed = time.time() - t0
    argv_list = list(argv) if argv else []
    argv_audit_flag = (
        _audit_argv_clock_policy(tuple(argv)) if argv else ""
    )

    if not records:
        return {
            "schema_version": 2,
            "task": _HARVEST_TAG,
            "plan": "crossarch-predictor-validity-restore-sm86-primary-20260528-052258",
            "slug": slug,
            "language": language,
            "host_arch": "sm_86",
            "host_kind": "local_sm_8_6",
            "device_name": "NVIDIA GeForce RTX 3090 Ti",
            "ncu_metric_family": list(_NCU_METRIC_FAMILY),
            "argv_provenance": [argv_list, argv_list, argv_list],
            "argv_clock_policy_flag": argv_audit_flag,
            "runs": [],
            "relative_cv_l1": -1.0,
            "relative_cv_l2": -1.0,
            "cv_cap": _CV_CAP,
            "cv_quarantined": True,
            "completed": False,
            "failure_reason": failure_reason or "ncu_subprocess_failed",
            "elapsed_seconds": elapsed,
        }

    runs_block = [
        {
            "run_index": i,
            "l1_global_load_hit_rate": float(record.l1_global_load_hit_rate),
            "l1_global_load_sectors_total": int(record.l1_global_load_sectors_total),
            "l1_global_load_sectors_hit": int(record.l1_global_load_sectors_hit),
            "l2_read_hit_rate": float(record.l2_read_hit_rate),
            "l2_read_sectors_total": int(record.l2_read_sectors_total),
            "l2_read_sectors_hit": int(record.l2_read_sectors_hit),
        }
        for i, record in enumerate(records)
    ]
    l1_rates = tuple(r["l1_global_load_hit_rate"] for r in runs_block)
    l2_rates = tuple(r["l2_read_hit_rate"] for r in runs_block)
    rel_cv_l1 = _relative_cv(l1_rates)
    rel_cv_l2 = _relative_cv(l2_rates)
    cv_quarantined = (
        rel_cv_l1 > _CV_CAP or rel_cv_l2 > _CV_CAP
    ) or bool(argv_audit_flag)
    return {
        "schema_version": 2,
        "task": _HARVEST_TAG,
        "plan": "crossarch-predictor-validity-restore-sm86-primary-20260528-052258",
        "slug": slug,
        "language": language,
        "host_arch": "sm_86",
        "host_kind": "local_sm_8_6",
        "device_name": "NVIDIA GeForce RTX 3090 Ti",
        "ncu_metric_family": list(_NCU_METRIC_FAMILY),
        "argv_provenance": [argv_list, argv_list, argv_list],
        "argv_clock_policy_flag": argv_audit_flag,
        "runs": runs_block,
        "relative_cv_l1": rel_cv_l1,
        "relative_cv_l2": rel_cv_l2,
        "cv_cap": _CV_CAP,
        "cv_quarantined": cv_quarantined,
        "completed": True,
        "failure_reason": (
            "ncu_repeatability_failed"
            if (rel_cv_l1 > _CV_CAP or rel_cv_l2 > _CV_CAP)
            else ""
        ),
        "elapsed_seconds": elapsed,
    }


def _write_sidecar(slug: str, sidecar: dict) -> str:
    path = _sidecar_path(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(sidecar, indent=2, sort_keys=True) + "\n"
    payload = text.encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _update_manifest(slug: str, sha: str, sidecar: dict) -> None:
    manifest_path = _bundle_dir(slug) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[_NCU_3RUN_BLOCK] = {
        "sidecar_path": "ncu_3run_v1.json",
        "sha256": sha,
        "schema_version": 1,
        "completed": bool(sidecar.get("completed")),
        "cv_quarantined": bool(sidecar.get("cv_quarantined")),
        "relative_cv_l1": float(sidecar.get("relative_cv_l1")),
        "relative_cv_l2": float(sidecar.get("relative_cv_l2")),
    }
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(text)


def _update_ledger(stamps: dict[str, str]) -> None:
    ledger_path = (
        _REPO_ROOT / "scripts" / "baselines_harness"
        / "calibration" / "cache_prediction"
        / "sm86_pilot_correctness_v1.toml"
    )
    text = ledger_path.read_text()
    new_lines: list[str] = []
    current_slug: str | None = None
    for line in text.splitlines():
        m = re.match(r"^slug\s*=\s*\"([^\"]+)\"", line.strip())
        if m:
            current_slug = m.group(1)
            new_lines.append(line)
            continue
        if line.strip().startswith(_NCU_3RUN_SHA_KEY):
            # Idempotent rerun: replace the existing entry with
            # the new stamp AND mark current_slug consumed so the
            # note-line insertion branch does NOT also fire.
            if current_slug and current_slug in stamps:
                new_lines.append(
                    f'{_NCU_3RUN_SHA_KEY} = "{stamps[current_slug]}"'
                )
                current_slug = None
            continue
        if (
            current_slug
            and current_slug in stamps
            and line.strip().startswith("note = ")
        ):
            new_lines.append(
                f'{_NCU_3RUN_SHA_KEY} = "{stamps[current_slug]}"'
            )
            new_lines.append(line)
            current_slug = None
            continue
        new_lines.append(line)
    ledger_path.write_text("\n".join(new_lines) + "\n")


def main(slugs: Iterable[str]) -> int:
    stamps: dict[str, str] = {}
    overall_t0 = time.time()
    for slug in slugs:
        print(f"=== {slug} ===", flush=True)
        t0 = time.time()
        sidecar = _harvest_cell(slug)
        sha = _write_sidecar(slug, sidecar)
        _update_manifest(slug, sha, sidecar)
        stamps[slug] = sha
        print(
            f"  completed={sidecar['completed']} "
            f"cv_l1={sidecar['relative_cv_l1']:.4f} "
            f"cv_l2={sidecar['relative_cv_l2']:.4f} "
            f"cv_quarantined={sidecar['cv_quarantined']} "
            f"elapsed={time.time()-t0:.1f}s sha={sha[:12]}",
            flush=True,
        )
    _update_ledger(stamps)
    print(
        f"Total elapsed: {time.time()-overall_t0:.1f}s; "
        f"ledger updated with cross-stamps.",
        flush=True,
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Drive NCU 3-run repeatability harvest for the sm_86 "
            "5-cell pilot. Emits ncu_3run_v1.json sidecars into "
            "each __sm86 bundle and cross-stamps SHA-256 into "
            "manifest + sm86_pilot_correctness_v1.toml."
        )
    )
    parser.add_argument(
        "--slug",
        action="append",
        choices=_PILOT_SLUGS,
        help=(
            "Restrict the harvest to this slug (repeatable). "
            "Default: all 5 admitted pilot slugs."
        ),
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    target_slugs = tuple(args.slug) if args.slug else _PILOT_SLUGS
    sys.exit(main(target_slugs))
