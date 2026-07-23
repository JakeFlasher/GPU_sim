"""Two-pass trace-sampling orchestration entry point.

Drives one budget-breaching cell through the ladder using the tracer's capture
modes: pass 1 (counting-only) -> admissibility gate -> segment selection -> pass 2
(selective-dump of the chosen whole blocks) -> compress + validate -> the
closed-schema `trace_sampling.json`. Gate-forbidden cells are routed to a typed
`unavailable` (a reduced-shape needs a separate shape-adjust wrapper). The
per-segment HT estimate itself is finalized later at scoring time (the sim yields
only per-SM aggregates, so per-segment measurement is a sub-trace-replay seam);
this entry point records the sampling plan + the exact captured record count.

The roster capture driver calls `orchestrate_cell`; the tracer modes + this local
orchestration are the trace-sampling machinery.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from admissibility_gate import MIN_SEGMENTS, M_SUBSET, classify
from compression import compress_trace_dir
from counting_pass import cell_stats_from_capture
from selection import SelectionPlanError, build_segment_population, plan_from_stats
from sidecar import build_sidecar, write_sidecar

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.nvbit_tracer.runtime_env import (  # noqa: E402
    TracerRuntimeError,
    tracer_runtime_env,
)

DEFAULT_TRACER_SO = _REPO / "tools" / "nvbit_tracer" / "tracer_tool" / "tracer_tool.so"
DEFAULT_PB_LIB = _REPO / ".toolchain" / "pb" / "lib"
CAPTURE_SIDECAR = Path("extra_info") / "trace_sampling_capture.json"


class SamplingRefusal(RuntimeError):
    """Typed refusal: no sampled sidecar or estimator point may be emitted."""

    def __init__(self, reason: str, detail: str | None = None):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


def _create_fresh_directory(path: Path, reason: str) -> None:
    """Atomically reserve a new output directory without deleting user data."""
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise SamplingRefusal(reason, f"output path already exists: {path}") from exc
    except OSError as exc:
        raise SamplingRefusal(reason, f"cannot create {path}: {exc}") from exc


def _refuse_existing_output(path: Path, reason: str) -> None:
    if path.exists() or path.is_symlink():
        raise SamplingRefusal(reason, f"output path already exists: {path}")


def run_pass(binary: Path, app_args: list[str], out_dir: Path, mode: int, *,
             selected_blocks: list[int] | None = None,
             tracer_so: Path = DEFAULT_TRACER_SO,
             pb_lib: Path = DEFAULT_PB_LIB) -> None:
    """Run one instrumented pass in the given trace-sampling mode. Fail-closed if
    no `dynamic_trace.pb` is produced (silent NVBit injection failure)."""
    _create_fresh_directory(out_dir, "capture_output_exists")
    try:
        env = tracer_runtime_env(tracer_so, pb_lib=pb_lib)
    except TracerRuntimeError as exc:
        raise SamplingRefusal("tracer_runtime_unavailable", str(exc)) from exc
    env["CUDA_INJECTION64_PATH"] = str(tracer_so)
    env["USER_DEFINED_FOLDERS"] = "1"
    env["TRACES_FOLDER"] = str(out_dir)
    env["TRACE_SAMPLING_MODE"] = str(mode)
    if mode == 2:
        env["TRACE_SAMPLING_SELECTED_BLOCKS"] = ",".join(str(b) for b in (selected_blocks or []))
    result = subprocess.run([str(binary), *app_args], env=env, cwd=str(binary.parent),
                            capture_output=True)
    # Fail closed on a nonzero target exit: a benchmark that fails a correctness
    # check while still writing a partial trace must NOT be routed as valid input.
    if result.returncode != 0:
        raise SamplingRefusal(
            "target_process_failed",
            f"pass mode={mode}: target exited {result.returncode}\n"
            f"stdout: {result.stdout.decode(errors='replace')[-2000:]}\n"
            f"stderr: {result.stderr.decode(errors='replace')[-2000:]}")
    if not (out_dir / "dynamic_trace.pb").is_file():
        raise SamplingRefusal(
            "capture_trace_missing",
            f"pass mode={mode}: no dynamic_trace.pb (injection failed?)")


def segments_from_stats(stats) -> dict[int, list[int]]:
    """Return the complete first-launch population, including its folded tail."""
    population = build_segment_population(stats)
    return {segment.id: list(segment.block_ids) for segment in population.segments}


def select_segments(stats, *, seed: int = 0,
                    sample_size: int = MIN_SEGMENTS) -> list[dict]:
    """Return certainty units followed by one deterministic seeded SRS draw."""
    return plan_from_stats(
        stats, seed=seed, sample_size=sample_size).sidecar_segments()


def verify_pass2_payload(
        pass2_dir: Path, grid: tuple[int, int, int], selected: list[int],
        expected_serialized_bytes: dict[int, int],
        ) -> tuple[dict[int, int], dict[int, int]]:
    """Verify the first-launch pass-2 roster and each raw protobuf byte size.

    Returns ``(records_by_block, raw_bytes_by_block)``.  The per-file byte check
    binds the producer's mode-2 ``serialized_bytes`` evidence to the actual raw
    payload, so certainty bytes cannot be redistributed between selected blocks.
    """
    import threadblock_pb2  # type: ignore
    from replay_count_producer import first_launch_dir  # type: ignore

    if (not isinstance(expected_serialized_bytes, dict)
            or any(type(block_id) is not int or block_id < 0
                   or type(size) is not int or size <= 0
                   for block_id, size in expected_serialized_bytes.items())):
        raise SamplingRefusal("capture_serialized_bytes_malformed")
    if (len(selected) != len(set(selected))
            or sorted(expected_serialized_bytes) != sorted(selected)):
        raise SamplingRefusal(
            "capture_serialized_roster_mismatch",
            f"sizes={sorted(expected_serialized_bytes)}, selected={sorted(selected)}")
    try:
        first_dir, _ = first_launch_dir(pass2_dir)
    except Exception as exc:
        raise SamplingRefusal("pass2_payload_unreadable", str(exc)) from exc
    gx, gy, gz = grid
    grid_total = gx * gy * gz
    records_by_block: dict[int, int] = {}
    raw_bytes_by_block: dict[int, int] = {}
    for p in sorted(first_dir.rglob("*.pb")):
        tb = threadblock_pb2.threadblock()
        try:
            raw = p.read_bytes()
            tb.ParseFromString(raw)
        except Exception as exc:
            raise SamplingRefusal(
                "pass2_payload_unreadable", f"{p}: {exc}") from exc
        if not tb.HasField("block_id"):
            raise SamplingRefusal(
                "pass2_block_id_missing", f"payload={p}")
        cx, cy, cz = int(tb.block_id.x), int(tb.block_id.y), int(tb.block_id.z)
        if not (0 <= cx < gx and 0 <= cy < gy and 0 <= cz < gz):
            raise SamplingRefusal(
                "pass2_block_out_of_grid",
                f"CTA ({cx},{cy},{cz}) outside grid ({gx},{gy},{gz})")
        fx = cx + cy * gx + cz * gx * gy
        if not (0 <= fx < grid_total):
            raise SamplingRefusal(
                "pass2_block_out_of_grid",
                f"block_id {fx} outside [0,{grid_total})")
        if fx in records_by_block:
            raise SamplingRefusal("pass2_duplicate_block", str(fx))
        expected_size = expected_serialized_bytes.get(fx)
        if expected_size is None:
            raise SamplingRefusal(
                "pass2_payload_membership_mismatch",
                f"dumped unexpected block_id {fx}")
        if len(raw) != expected_size:
            raise SamplingRefusal(
                "pass2_serialized_bytes_mismatch",
                f"block={fx}, payload={len(raw)}, capture={expected_size}")
        records_by_block[fx] = sum(
            len(tb.warps[wi].instructions) for wi in tb.warps)
        raw_bytes_by_block[fx] = len(raw)
    if sorted(records_by_block) != sorted(selected):
        raise SamplingRefusal(
            "pass2_payload_membership_mismatch",
            f"dumped={sorted(records_by_block)}, selected={sorted(selected)}")
    return records_by_block, raw_bytes_by_block


def verify_capture_selection(pass2_dir: Path, selected: list[int]) -> str:
    """Bind the producer's selected-ID declaration and return its raw SHA-256."""
    path = pass2_dir / CAPTURE_SIDECAR
    try:
        raw = path.read_bytes()
        capture = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise SamplingRefusal("capture_sidecar_unreadable", str(exc)) from exc
    if type(capture.get("mode")) is not int or capture["mode"] != 2:
        raise SamplingRefusal(
            "capture_mode_mismatch", f"expected mode 2, got {capture.get('mode')!r}")
    declared = capture.get("selected_block_ids")
    if not isinstance(declared, list) or any(type(value) is not int or value < 0
                                             for value in declared):
        raise SamplingRefusal("capture_selected_ids_malformed")
    if len(declared) != len(set(declared)):
        raise SamplingRefusal("capture_selected_ids_duplicate")
    if sorted(declared) != sorted(selected):
        raise SamplingRefusal(
            "capture_selected_ids_mismatch",
            f"producer={sorted(declared)}, candidate={sorted(selected)}")
    return hashlib.sha256(raw).hexdigest()


def _selected_block_ids(plan) -> list[int]:
    return sorted({block_id for segment in plan.selected_segments
                   for block_id in segment.block_ids})


def _same_launch_identity(bootstrap, observed) -> None:
    fields = ("kernel_name", "grid", "block", "n_blocks")
    mismatch = {field: (getattr(bootstrap, field), getattr(observed, field))
                for field in fields
                if getattr(bootstrap, field) != getattr(observed, field)}
    bootstrap_ids = sorted(block.block_id for block in bootstrap.blocks)
    observed_ids = sorted(block.block_id for block in observed.blocks)
    if bootstrap_ids != observed_ids:
        mismatch["block_ids"] = (bootstrap_ids, observed_ids)
    if mismatch:
        raise SamplingRefusal("pass2_launch_identity_drift", repr(mismatch))


def _certainty_serialized_bytes(stats, plan) -> int:
    by_id = {block.block_id: block for block in stats.blocks}
    total = 0
    for segment in plan.certainty_segments:
        for block_id in segment.block_ids:
            block = by_id[block_id]
            value = getattr(block, "serialized_bytes", None)
            if type(value) is not int or value <= 0:
                raise SamplingRefusal(
                    "missing_exact_certainty_bytes",
                    f"block={block_id}, serialized_bytes={value!r}")
            total += value
    return total


def _selection_auxiliary(stats, selected_blocks: list[int]) -> dict:
    selected = set(selected_blocks)
    expected_full = 0
    selected_total = 0
    for block in stats.blocks:
        value = getattr(block, "sector_accesses", None)
        if type(value) is not int or value < 0:
            raise SamplingRefusal(
                "malformed_sector_auxiliary",
                f"block={block.block_id}, sector_accesses={value!r}")
        expected_full += value
        if block.block_id in selected:
            selected_total += value
    return {
        "name": "l2_sector_transactions_32b",
        "sector_bytes": 32,
        "scope": "pass2_all_blocks",
        "expected_full": expected_full,
        "selected": selected_total,
    }


def orchestrate_cell(binary: Path, app_args: list[str], work_dir: Path, *,
                     args_hash: str = "0" * 8, cooperative: bool = False,
                     seed: int = 0, sample_size: int = MIN_SEGMENTS,
                     max_trace_bytes: int | None = None,
                     max_reconcile_attempts: int = 3,
                     tracer_so: Path = DEFAULT_TRACER_SO,
                     pb_lib: Path = DEFAULT_PB_LIB) -> dict:
    """Run pass1 -> gate -> (subset pass2 | typed unavailable) -> sidecar. Returns
    the validated `trace_sampling.json` dict (also written under work_dir)."""
    cell_id = binary.stem
    out = work_dir / "trace_sampling.json"
    compressed_out = work_dir / "pass2_zst"
    _refuse_existing_output(out, "sidecar_output_exists")
    _refuse_existing_output(compressed_out, "compression_output_exists")
    pass1 = work_dir / "pass1"
    run_pass(binary, app_args, pass1, mode=1, tracer_so=tracer_so, pb_lib=pb_lib)
    stats = cell_stats_from_capture(pass1, cell_id, cooperative=cooperative)
    gate = classify(stats, sample_size=sample_size)

    if gate.method == M_SUBSET:
        if type(max_trace_bytes) is not int or max_trace_bytes <= 0:
            raise SamplingRefusal(
                "missing_cell_trace_budget",
                "subset capture requires an explicit positive max_trace_bytes")
        if type(max_reconcile_attempts) is not int or max_reconcile_attempts <= 0:
            raise SamplingRefusal(
                "invalid_reconcile_attempts", repr(max_reconcile_attempts))
        bootstrap_force = tuple(gate.force_include_blocks)
        try:
            candidate_plan = plan_from_stats(
                stats, seed=seed, sample_size=sample_size,
                force_include_block_ids=bootstrap_force)
        except SelectionPlanError as exc:
            raise SamplingRefusal("bootstrap_selection_refused", str(exc)) from exc
        candidate = _selected_block_ids(candidate_plan)

        accepted = None
        for attempt in range(1, max_reconcile_attempts + 1):
            pass2 = work_dir / f"pass2_attempt_{attempt}"
            run_pass(binary, app_args, pass2, mode=2, selected_blocks=candidate,
                     tracer_so=tracer_so, pb_lib=pb_lib)
            pass2_stats = cell_stats_from_capture(
                pass2, cell_id, cooperative=cooperative)
            _same_launch_identity(stats, pass2_stats)
            capture_stats_sha256 = verify_capture_selection(pass2, candidate)
            candidate_set = set(candidate)
            candidate_serialized_bytes = {
                block.block_id: block.serialized_bytes
                for block in pass2_stats.blocks
                if block.block_id in candidate_set
            }
            actual_records, actual_raw_bytes = verify_pass2_payload(
                pass2, pass2_stats.grid, candidate,
                candidate_serialized_bytes)

            provisional_gate = classify(
                pass2_stats, sample_size=sample_size,
                force_include_block_ids=bootstrap_force)
            if provisional_gate.method != M_SUBSET:
                raise SamplingRefusal(
                    "pass2_gate_forced",
                    f"attempt={attempt}, verdict={provisional_gate.verdict}")
            final_force = tuple(provisional_gate.force_include_blocks)
            try:
                desired_plan = plan_from_stats(
                    pass2_stats, seed=seed, sample_size=sample_size,
                    force_include_block_ids=final_force)
            except SelectionPlanError as exc:
                raise SamplingRefusal(
                    "pass2_selection_refused",
                    f"attempt={attempt}, {exc}") from exc
            desired = _selected_block_ids(desired_plan)
            certainty_bytes = _certainty_serialized_bytes(
                pass2_stats, desired_plan)
            final_gate = classify(
                pass2_stats, sample_size=sample_size,
                force_include_block_ids=final_force,
                certainty_bytes=certainty_bytes,
                max_trace_bytes=max_trace_bytes)
            if final_gate.method != M_SUBSET:
                # The final, exact raw-PB certainty budget is a real gate.  A
                # reduced-shape label does not exist here, so emit no sampled
                # point and return the established typed unavailable method.
                sidecar = build_sidecar(
                    cell_id=cell_id, kernel_name=pass2_stats.kernel_name,
                    args_hash=args_hash, stats=pass2_stats, gate=final_gate,
                    compression=None, records_captured=0, tracer_version=5,
                    stats_pass_version=pass2_stats.stats_pass_version,
                    method="unavailable", reason="gate_forced")
                write_sidecar(sidecar, out)
                return sidecar
            if desired != candidate:
                candidate = desired
                bootstrap_force = final_force
                continue
            accepted = (attempt, pass2, pass2_stats, desired_plan, final_gate,
                        certainty_bytes, capture_stats_sha256, actual_records,
                        actual_raw_bytes)
            break

        if accepted is None:
            raise SamplingRefusal(
                "placement_reconciliation_exhausted",
                f"max_attempts={max_reconcile_attempts}")

        (attempts, pass2, final_stats, plan, gate, certainty_bytes,
         capture_stats_sha256, actual_records, actual_raw_bytes) = accepted
        segments = plan.sidecar_segments()
        selected = _selected_block_ids(plan)
        selected_set = set(selected)
        selected_payload_bytes = sum(actual_raw_bytes.values())
        if selected_payload_bytes > max_trace_bytes:
            raise SamplingRefusal(
                "selected_payload_over_budget",
                f"raw={selected_payload_bytes}, max={max_trace_bytes}")
        _create_fresh_directory(compressed_out, "compression_output_exists")
        comp = compress_trace_dir(pass2, compressed_out)
        selected_serialized_bytes = sum(
            block.serialized_bytes for block in final_stats.blocks
            if block.block_id in selected_set)
        if (comp.raw_bytes != selected_payload_bytes
                or selected_payload_bytes != selected_serialized_bytes):
            raise SamplingRefusal(
                "serialized_payload_size_mismatch",
                f"compressed_input={comp.raw_bytes}, payload={selected_payload_bytes}, "
                f"counting={selected_serialized_bytes}")
        captured = sum(actual_records.values())
        strata = plan.sidecar_strata()
        estimator = {
            "mode": "weighted_full_shape_estimate",
            "targets": ["sectors", "l2_sector_hit_rate"],
            "not_recoverable": ["cycles", "sol_pct_timing"],
            # Finalized at scoring time (per-segment sub-trace replay seam).
            "convergence": {"status": "budget_limited_wide_ci"},
        }
        sidecar = build_sidecar(
            cell_id=cell_id, kernel_name=final_stats.kernel_name, args_hash=args_hash,
            stats=final_stats, gate=gate, compression=comp, records_captured=captured,
            tracer_version=5,
            stats_pass_version=final_stats.stats_pass_version,
            method=M_SUBSET,
            reason="byte_budget", segments=segments, strata=strata,
            estimator=estimator, decompressed_payload_sha256=comp.decompressed_sha256,
            seed=seed, rng_algo="pcg64",
            selected_segment_ids=[segment.id for segment in plan.selected_segments],
            certainty_segments=[segment.id for segment in plan.certainty_segments],
            force_include_blocks=list(plan.population.force_include_block_ids),
            selection_reconciliation={
                "status": "fixed_point", "attempts": attempts,
                "max_attempts": max_reconcile_attempts,
                "capture_stats_sha256": capture_stats_sha256,
            },
            selection_budget={
                "max_trace_bytes": max_trace_bytes,
                "basis": "raw_threadblock_pb",
                "certainty_fraction_numerator": 2,
                "certainty_fraction_denominator": 5,
                "certainty_bytes": certainty_bytes,
            },
            selection_auxiliary=_selection_auxiliary(final_stats, selected))
    else:
        # Gate-forbidden -> typed unavailable (a reduced shape needs a separate
        # shape-adjust wrapper; never silently subset a forbidden cell).
        sidecar = build_sidecar(
            cell_id=cell_id, kernel_name=stats.kernel_name, args_hash=args_hash,
            stats=stats, gate=gate, compression=None, records_captured=0,
            tracer_version=5,
            stats_pass_version=stats.stats_pass_version,
            method="unavailable",
            reason="gate_forced")

    write_sidecar(sidecar, out)
    return sidecar
