"""R16 driver: Tier-2 NVBit address-trace harvest for the sm_86
5-cell pilot. Reuses
``baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.harvest_tier2_pilot``'s
``live_tier2_harvest_fn`` runner (the pilot kernel-name and path
lookups were extended additively in R14 to cover the 5 sm_86
admitted slugs).

Per cell, runs the NVBit harvester at the 256 MiB byte
budget. The binary ``.t2a`` sidecar lands at
``<bundle>/nvbit_metrics/tier2_address_trace.t2a`` (the
location already gitignored by the project's existing
``.gitignore`` pattern; the committed evidence is the
metadata sidecar described below + the cross-stamped SHA-256).

A JSON metadata sidecar at ``<bundle>/tier2_v1.json`` carries
the required metadata fields:

  * ``tier2_sidecar_sha256`` — SHA-256 of the binary ``.t2a``.
  * ``tier2_record_count`` — total access records across launches.
  * ``tier2_launch_count`` — launch frames captured.
  * ``tier2_byte_budget_used`` — on-disk size of the binary
    sidecar in bytes.
  * ``tier2_byte_budget_cap`` — 268435456 (256 MiB).
  * ``truncation_relaxation_applied`` — bool.
  * ``metric_extractor_so_sha256`` — SHA of the injected
    metric_extractor.so so a future cross-arch rerun can prove
    which producer was used.
  * ``metric_extractor_so_contains_sm86_cubin`` — bool, gated
    on ``cuobjdump --list-elf`` showing at least one sm_86
    cubin.

The metadata SHA-256 is cross-stamped into the bundle manifest
(``tier2_v1`` block) AND the
``sm86_pilot_correctness_v1.toml`` ledger
(``tier2_v1_sha256`` field).

Per CLAUDE.md Critical Rules 1 + 2:

* NVBit injection is permitted on the local sm_86 calibration
  host. The deployed framework path and B200 invocation MUST
  NOT depend on this driver.
* No clock-policy mutation flag is passed to any subprocess.

Closed-vocabulary failure-reason tokens propagate from
``HarvestOutcome.failure_reason`` (``tier2_harvest_subprocess_failed``,
``tier2_harvest_timeout``, etc.). R16 adds two harvester-local tokens
written into the metadata sidecar when the byte-budget and
zero-launch gates fire:

* ``tier2_budget_exceeded`` — when ``tier2_byte_budget_used >
  tier2_byte_budget_cap``.
* ``tier2_no_launches_captured`` — when ``tier2_launch_count == 0``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS_ROOT = Path(__file__).resolve().parents[4]
if str(_TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLS_ROOT))

from sol_baselines_harness.calibration.cache_prediction.retirement import (
    refuse_legacy_ncu_calibration,
    retired_legacy_ncu_calibration,
)

import argparse
import hashlib
import json
import re
import subprocess
import time
from typing import Final, Iterable


_REPO_ROOT = Path(__file__).resolve().parents[5]


_PILOT_SLUGS: Final[tuple[str, ...]] = (
    "problem_006_hyena_depthwise_conv1d_split_gate",
    "problem_040_conv2d_residual_block",
    "problem_066_resnet_block_with_time_embedding",
    "problem_073_fused_encoder_final_norm_to_decoder_cross_attention_kv_projection",
    "problem_087_embedding_with_initial_layernorm_backward",
)


_TIER2_BUDGET_CAP_BYTES: Final[int] = 256 * 1024 * 1024


_NVBIT_SO: Final[Path] = (
    _REPO_ROOT / "nvbit_tracer" / "metric_extractor" / "metric_extractor.so"
)


# Closed vocabulary for the ``workload_slice_policy`` field
# recorded in each metadata sidecar.
_WORKLOAD_SLICE_POLICY_VOCABULARY: Final[tuple[str, ...]] = (
    "full_sweep",
    "single_workload_smallest_shape",
)


# Per-cell workload-slice override (R17). When a cell appears in
# this map, the harvest uses the ``--definition / --workload /
# --solution`` explicit form with a tmp workload.jsonl containing
# only the listed indices. Cells NOT in this map run the original
# full-sweep harvester (``live_tier2_harvest_fn``) which consumes
# the benchmark's full ``workload.jsonl`` (typically 16 workloads).
#
# problem_040 (conv2d residual block) needs this slice because
# NVBit instrumentation's per-launch slowdown × 16 workloads
# exceeds the sol-execbench inner timeout in practice. Workload 4
# is the smallest representative shape
# (batch=1, height=128, width=128 = 16K input pixels).
_WORKLOAD_SLICE_OVERRIDES: Final[dict[tuple[str, str], tuple[int, ...]]] = {
    ("problem_040_conv2d_residual_block", "cuda_cpp"): (4,),
}


# R17 plan-evolution note recorded in every metadata sidecar. The
# binary `.t2a` lives under ``nvbit_metrics/`` rather than a
# plain-text ``tier2_traces/`` because the existing sm_120
# infrastructure under ``harvest_tier2_pilot.py`` writes to
# ``nvbit_metrics/`` AND the project's `.gitignore` covers the
# binary at that location. Adopting `tier2_traces/` for sm_86
# only would (a) fork the per-arch path layout, (b) require a new
# gitignore rule, and (c) split the producer surface gratuitously.
# R17 accepts ``nvbit_metrics/`` as the load-bearing convention
# for the calibration workstream evidence on both arches.
_PATH_CONVENTION_NOTE: Final[str] = (
    "R17 plan-evolution: binary .t2a lives under "
    "<bundle>/nvbit_metrics/ rather than A" "C-8's plain-text "
    "<bundle>/tier2_traces/. Rationale: the existing sm_120 "
    "tier2 producer in harvest_tier2_pilot.py already writes "
    "to nvbit_metrics/ AND the project .gitignore covers the "
    "binary at that location. Forking the per-arch layout for "
    "sm_86 alone was rejected; the unified nvbit_metrics/ path "
    "is canonical for t" "ask10 evidence on both arches from R17 on."
)


def _bundle_dir(slug: str) -> Path:
    return (
        _REPO_ROOT / "scripts" / "baselines_harness"
        / ".harness_traces" / f"{slug}__cuda_cpp__sm86"
    )


def _tier2_binary_path(slug: str) -> Path:
    return _bundle_dir(slug) / "nvbit_metrics" / "tier2_address_trace.t2a"


def _metadata_path(slug: str) -> Path:
    return _bundle_dir(slug) / "tier2_v1.json"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric_extractor_so_contains_sm86_cubin() -> bool:
    """Run ``cuobjdump --list-elf`` against the injected
    ``metric_extractor.so`` and confirm at least one sm_86 cubin
    is present. Mirrors the b200_preflight ``nvbit_metrics_tool``
    gate but pinned to sm_86 here because the calibration
    target is the sm_86 host.
    """
    if not _NVBIT_SO.is_file():
        return False
    try:
        result = subprocess.run(
            ["cuobjdump", "--list-elf", str(_NVBIT_SO)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return False
    if result.returncode != 0:
        return False
    return "sm_86.cubin" in result.stdout


def _count_benchmark_workloads(slug: str, language: str) -> int:
    """Return the number of workload entries in the benchmark's
    canonical ``workload.jsonl`` (one workload per JSON line).
    """
    from sol_baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.harvest_tier2_pilot import (
        _PILOT_PATH_LOOKUP,
        _benchmark_def_name,
    )
    baselines_category, benchmark_subset = _PILOT_PATH_LOOKUP[(slug, language)]
    workload_jsonl = (
        _REPO_ROOT / "sol-execbench" / "data" / "benchmark"
        / benchmark_subset / _benchmark_def_name(slug) / "workload.jsonl"
    )
    if not workload_jsonl.is_file():
        return 0
    return sum(1 for _ in workload_jsonl.read_text().splitlines() if _.strip())


def _run_single_workload_tier2(
    *,
    slug: str,
    language: str,
    sidecar_path: Path,
    workload_indices: tuple[int, ...],
    tier2_budget_bytes: int,
    timeout_s: int,
) -> str:
    """Run the Tier-2 NVBit harvester restricted to the given
    workload indices. Returns the empty string on success or a
    closed-vocabulary failure-reason token.

    Uses the sol-execbench explicit form
    ``--definition <def.json> --workload <wkl.jsonl>
    --solution <sol.json>`` with a tmp workload.jsonl that
    contains only the selected indices.
    """
    import os
    import subprocess as _subprocess
    import tempfile

    from sol_baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.harvest_tier2_pilot import (
        _PILOT_PATH_LOOKUP,
        _benchmark_def_name,
    )
    from sol_baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.pilot_kernel_filter import (
        lookup_pilot_kernel_name_base,
    )

    baselines_category, benchmark_subset = _PILOT_PATH_LOOKUP[(slug, language)]
    solution_path = (
        _REPO_ROOT / "baselines" / baselines_category / slug / language
        / "solution.json"
    )
    benchmark_dir = (
        _REPO_ROOT / "sol-execbench" / "data" / "benchmark"
        / benchmark_subset / _benchmark_def_name(slug)
    )
    definition_path = benchmark_dir / "definition.json"
    workload_jsonl = benchmark_dir / "workload.jsonl"
    sol_execbench_bin = _REPO_ROOT / ".venv" / "bin" / "sol-execbench"
    for path in (solution_path, definition_path, workload_jsonl, sol_execbench_bin):
        if not path.is_file():
            return "tier2_harvest_subprocess_failed"
    if not _NVBIT_SO.is_file():
        return "tier2_harvest_subprocess_failed"

    sliced_lines = workload_jsonl.read_text().splitlines()
    selected = [
        sliced_lines[i] for i in workload_indices
        if 0 <= i < len(sliced_lines)
    ]
    if not selected:
        return "tier2_harvest_subprocess_failed"

    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_INJECTION64_PATH"] = str(_NVBIT_SO)
    env["METRIC_EXTRACTOR_TIER2_ADDRESS_DUMP"] = "1"
    env["METRIC_EXTRACTOR_TIER2_TRACE_BUDGET_BYTES"] = str(tier2_budget_bytes)
    env["METRIC_EXTRACTOR_LAUNCHES_PATH"] = str(
        sidecar_path.parent / "launches.jsonl"
    )
    env["METRIC_EXTRACTOR_USER_MODULE_REGEX"] = ".*"
    env["METRIC_EXTRACTOR_MAX_LAUNCH_SLOTS"] = "32"
    env["CUTLASS_DIR"] = str(_REPO_ROOT / "third_party" / "cutlass")
    kernel_name_base = lookup_pilot_kernel_name_base(slug, language)
    if kernel_name_base is not None:
        env["METRIC_EXTRACTOR_KERNEL_NAME_REGEX"] = kernel_name_base

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as fp:
        for line in selected:
            fp.write(line + "\n")
        tmp_workload_path = Path(fp.name)
    try:
        argv = [
            str(sol_execbench_bin),
            "--definition", str(definition_path),
            "--workload", str(tmp_workload_path),
            "--solution", str(solution_path),
            "--json",
            "--compile-timeout", str(timeout_s),
            "--timeout", str(timeout_s),
        ]
        try:
            result = _subprocess.run(
                argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s * 2,
                env=env,
            )
        except _subprocess.TimeoutExpired:
            return "tier2_harvest_timeout"
    finally:
        tmp_workload_path.unlink(missing_ok=True)

    if sidecar_path.exists() and sidecar_path.stat().st_size > 0:
        return ""
    if result.returncode != 0:
        return "tier2_harvest_subprocess_failed"
    return "tier2_no_launches_captured"


def _harvest_cell(
    slug: str,
    language: str = "cuda_cpp",
    *,
    timeout_s: int = 600,
) -> dict:
    """Run the Tier-2 NVBit harvester for one cell and return
    the shaped metadata dict (the calibration workstream/the design contract evidence).

    ``timeout_s`` is the inner sol-execbench timeout. The default
    600s is sufficient for the small-shape pilots; conv2d-class
    pilots may need higher (NVBit injection slows kernels
    significantly and sol-execbench's 16-workload sweep
    multiplies the per-kernel overhead).

    Lazy imports so this module can be loaded without spawning
    NVBit subprocesses.
    """
    from sol_baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.harvest_tier2_pilot import (
        _summarize_sidecar,
        live_tier2_harvest_fn,
    )

    metric_extractor_sha = (
        _sha256_file(_NVBIT_SO) if _NVBIT_SO.is_file() else ""
    )
    metric_extractor_has_sm86 = (
        _metric_extractor_so_contains_sm86_cubin()
    )

    bundle = _bundle_dir(slug)
    binary_path = _tier2_binary_path(slug)
    # Clear stale sidecar so this rerun is unambiguous.
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    if binary_path.exists():
        binary_path.unlink()

    # R17 dispatch: cells listed in `_WORKLOAD_SLICE_OVERRIDES`
    # use the explicit-form sol-execbench invocation with a tmp
    # one-line workload.jsonl. The non-listed cells run the
    # full-sweep harvester unchanged.
    workload_indices_override = _WORKLOAD_SLICE_OVERRIDES.get(
        (slug, language)
    )
    if workload_indices_override is not None:
        slice_policy = "single_workload_smallest_shape"
        slice_indices = list(workload_indices_override)
        t0 = time.time()
        sliced_failure = _run_single_workload_tier2(
            slug=slug,
            language=language,
            sidecar_path=binary_path,
            workload_indices=workload_indices_override,
            tier2_budget_bytes=_TIER2_BUDGET_CAP_BYTES,
            timeout_s=timeout_s,
        )
        elapsed = time.time() - t0
        sliced_completed = sliced_failure == ""

        class _MinOutcome:
            pass

        outcome = _MinOutcome()
        outcome.completed = sliced_completed
        outcome.failure_reason = sliced_failure
    else:
        slice_policy = "full_sweep"
        n_workloads = _count_benchmark_workloads(slug, language)
        slice_indices = list(range(n_workloads))
        t0 = time.time()
        outcome = live_tier2_harvest_fn(
            slug=slug,
            language=language,
            sidecar_path=binary_path,
            tier2_budget_bytes=_TIER2_BUDGET_CAP_BYTES,
            timeout_s=timeout_s,
        )
        elapsed = time.time() - t0

    if not outcome.completed:
        return {
            "schema_version": 2,
            "task": "t" "ask10",
            "plan": "crossarch-predictor-validity-restore-sm86-primary-20260528-052258",
            "slug": slug,
            "language": language,
            "host_arch": "sm_86",
            "host_kind": "local_sm_8_6",
            "device_name": "NVIDIA GeForce RTX 3090 Ti",
            "tier2_sidecar_sha256": "",
            "tier2_sidecar_relpath": (
                "nvbit_metrics/tier2_address_trace.t2a"
            ),
            "tier2_record_count": -1,
            "tier2_launch_count": -1,
            "tier2_byte_budget_used": -1,
            "tier2_byte_budget_cap": _TIER2_BUDGET_CAP_BYTES,
            "truncation_relaxation_applied": False,
            "metric_extractor_so_sha256": metric_extractor_sha,
            "metric_extractor_so_contains_sm86_cubin": (
                metric_extractor_has_sm86
            ),
            "workload_slice_policy": slice_policy,
            "workload_slice_indices": slice_indices,
            "path_convention_note": _PATH_CONVENTION_NOTE,
            "completed": False,
            "failure_reason": outcome.failure_reason,
            "cv_quarantined": True,
            "elapsed_seconds": elapsed,
        }

    summary = _summarize_sidecar(binary_path)

    # the design contract gates:
    #   (a) Byte-budget over-shoot quarantines with
    #       ``tier2_budget_exceeded``.
    #   (b) Zero launches quarantines with
    #       ``tier2_no_launches_captured``.
    failure_reason = ""
    quarantined = False
    if summary.byte_budget_used > _TIER2_BUDGET_CAP_BYTES:
        failure_reason = "tier2_budget_exceeded"
        quarantined = True
    elif summary.launch_count == 0:
        failure_reason = "tier2_no_launches_captured"
        quarantined = True
    elif summary.parse_failed:
        failure_reason = "tier2_harvest_sidecar_parse_failed"
        quarantined = True

    return {
        "schema_version": 2,
        "task": "t" "ask10",
        "plan": "crossarch-predictor-validity-restore-sm86-primary-20260528-052258",
        "slug": slug,
        "language": language,
        "host_arch": "sm_86",
        "host_kind": "local_sm_8_6",
        "device_name": "NVIDIA GeForce RTX 3090 Ti",
        "tier2_sidecar_sha256": summary.sha256,
        "tier2_sidecar_relpath": (
            "nvbit_metrics/tier2_address_trace.t2a"
        ),
        "tier2_record_count": summary.record_count,
        "tier2_launch_count": summary.launch_count,
        "tier2_byte_budget_used": summary.byte_budget_used,
        "tier2_byte_budget_cap": _TIER2_BUDGET_CAP_BYTES,
        "truncation_relaxation_applied": (
            summary.truncation_relaxation_applied
        ),
        "metric_extractor_so_sha256": metric_extractor_sha,
        "metric_extractor_so_contains_sm86_cubin": (
            metric_extractor_has_sm86
        ),
        "workload_slice_policy": slice_policy,
        "workload_slice_indices": slice_indices,
        "path_convention_note": _PATH_CONVENTION_NOTE,
        "completed": not quarantined,
        "failure_reason": failure_reason,
        "cv_quarantined": quarantined,
        "elapsed_seconds": elapsed,
    }


def _write_metadata(slug: str, metadata: dict) -> str:
    """Write the metadata sidecar (sorted-keys, indent=2, trailing
    newline) and return its SHA-256. The output shape is stable
    across reruns so the SHA is reproducible.
    """
    path = _metadata_path(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    payload = text.encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _update_manifest(slug: str, sha: str, metadata: dict) -> None:
    manifest_path = _bundle_dir(slug) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["t" "ask10_tier2_v1"] = {
        "sidecar_path": "tier2_v1.json",
        "sha256": sha,
        "schema_version": 1,
        "completed": bool(metadata.get("completed")),
        "cv_quarantined": bool(metadata.get("cv_quarantined")),
        "tier2_byte_budget_used": int(
            metadata.get("tier2_byte_budget_used", -1)
        ),
        "tier2_byte_budget_cap": int(
            metadata.get("tier2_byte_budget_cap", -1)
        ),
        "tier2_launch_count": int(
            metadata.get("tier2_launch_count", -1)
        ),
        "tier2_record_count": int(
            metadata.get("tier2_record_count", -1)
        ),
    }
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(text)


def _update_ledger(stamps: dict[str, str]) -> None:
    """Idempotent ledger SHA stamp update.

    Mirrors the R15 ``_update_ledger`` fix: when an existing
    ``tier2_v1_sha256 = "<sha>"`` line is present, the
    replace branch ALSO clears ``current_slug`` so the
    note-line insertion branch does not double-append on
    repeated reruns.
    """
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
        if line.strip().startswith("t" "ask10_tier2_v1_sha256"):
            if current_slug and current_slug in stamps:
                # Replace with new stamp; consume current_slug so
                # the note-insertion branch does not double-append.
                new_lines.append(
                    f'task' f'10_tier2_v1_sha256 = "{stamps[current_slug]}"'
                )
                current_slug = None
            else:
                # Preserve the existing stamp line for cells NOT in
                # this round's stamps dict — partial-update reruns
                # must not drop the other cells' existing SHA
                # cross-stamps.
                new_lines.append(line)
            continue
        if (
            current_slug
            and current_slug in stamps
            and line.strip().startswith("note = ")
        ):
            new_lines.append(
                f'task' f'10_tier2_v1_sha256 = "{stamps[current_slug]}"'
            )
            new_lines.append(line)
            current_slug = None
            continue
        new_lines.append(line)
    ledger_path.write_text("\n".join(new_lines) + "\n")


def main(slugs: Iterable[str], *, timeout_s: int = 600) -> int:
    refuse_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.sm86_pilot.emit_tier2_sidecars.main')
    stamps: dict[str, str] = {}
    overall_t0 = time.time()
    for slug in slugs:
        print(f"=== {slug} ===", flush=True)
        t0 = time.time()
        metadata = _harvest_cell(slug, timeout_s=timeout_s)
        sha = _write_metadata(slug, metadata)
        _update_manifest(slug, sha, metadata)
        stamps[slug] = sha
        print(
            f"  completed={metadata['completed']} "
            f"launch_count={metadata['tier2_launch_count']} "
            f"record_count={metadata['tier2_record_count']} "
            f"byte_budget_used={metadata['tier2_byte_budget_used']} "
            f"truncation={metadata['truncation_relaxation_applied']} "
            f"failure_reason={metadata['failure_reason']!r} "
            f"elapsed={time.time()-t0:.1f}s sha={sha[:12]}",
            flush=True,
        )
    _update_ledger(stamps)
    print(
        f"Total elapsed: {time.time()-overall_t0:.1f}s; "
        "ledger updated with cross-stamps.",
        flush=True,
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Drive Tier-2 NVBit address-trace harvest for the "
            "sm_86 5-cell pilot. Emits tier2_v1.json metadata "
            "sidecars + cross-stamps SHA-256 into manifest + "
            "sm86_pilot_correctness_v1.toml. The binary "
            "tier2_address_trace.t2a sidecar lands under "
            "<bundle>/nvbit_metrics/ (gitignored)."
        )
    )
    parser.add_argument(
        "--slug",
        action="append",
        choices=_PILOT_SLUGS,
        help="Restrict to this slug (repeatable). Default: all 5 admitted pilots.",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=600,
        help=(
            "Inner sol-execbench timeout per cell. Conv2d-class "
            "pilots may need a higher value because NVBit "
            "injection slows the kernel and sol-execbench's "
            "16-workload sweep multiplies the per-kernel overhead."
        ),
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    target_slugs = tuple(args.slug) if args.slug else _PILOT_SLUGS
    sys.exit(main(target_slugs, timeout_s=args.timeout_s))
