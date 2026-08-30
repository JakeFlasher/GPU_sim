#!/usr/bin/env python3
"""Run the sm_86 LDGSTS burst-release measurement campaign.

The implemented live phases are deliberately serial: one process owns the RTX
3090 Ti at a time, and a direct or NCU profile becomes resumable only after all
raw outputs are complete.  NCU profiles use the trusted Nsight Compute path,
application replay, and a bounded capture/export pair.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results"
    / "captures"
    / "sm86_ldgsts_burst_release_20260830_v4"
)
BINARY = HERE / "ldgsts_burst_release"
CUDA_BIN = os.environ.get("CUDA_BIN", "/usr/local/cuda-13.2/bin")
NVCC = Path(os.environ.get("NVCC", str(Path(CUDA_BIN) / "nvcc")))
CUOBJDUMP = Path(
    os.environ.get("CUOBJDUMP", str(Path(CUDA_BIN) / "cuobjdump"))
)
TOOLKIT_BINDIR = Path(
    os.environ.get("CUDA_TOOLKIT_BINDIR", str(CUOBJDUMP.parent))
)
CONTROL_SIDECAR = REPO_ROOT / "tools" / "sass_control" / "emit_control_bits_sidecar.py"
TRUSTED_NCU = "/usr/local/cuda-13.2/nsight-compute-2026.1.0/ncu"
NCU = os.environ.get("NCU", TRUSTED_NCU)
NVIDIA_SMI = os.environ.get("NVIDIA_SMI", "nvidia-smi")

MEASURED_BURSTS = 16_384
FIXED_WARMUP = 64
TIMING_SCHEMA = 2
DIRECT_REPETITIONS = 7
NCU_BASE_REPETITIONS = 3
NCU_EXTENDED_REPETITIONS = 5
ACTIVE_CYCLE_CV_LIMIT = 0.02
FIXED_CORE_MHZ = 1350
FIXED_MEMORY_MHZ = 10501

# Frozen profiler request.  The first metric drives the predeclared 3->5 repeat
# variance rule; the remainder admit static work, traffic, route, and SMSP
# placement.
NCU_METRICS = (
    "sm__cycles_active.max",
    "smsp__inst_executed_op_ldgsts.sum",
    "smsp__sass_inst_executed_op_global_ld.sum",
    "sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts.sum",
    "sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
    "dram__sectors_read.sum",
)

ORDER_SCHEDULE = (
    "forward",
    "reverse",
    "forward",
    "reverse",
    "forward",
    "reverse",
    "forward",
)


class CampaignError(RuntimeError):
    """A bounded campaign command or evidence check failed."""


class CampaignInterrupted(BaseException):
    """A termination signal requested a clean clock restoration."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"interrupted by signal {signum}")
        self.signum = signum


@dataclass(frozen=True, slots=True)
class Cell:
    role: str
    assay: str
    warps: int
    route: str
    depth: int
    gap: int

    @property
    def key(self) -> str:
        return (
            f"{self.assay}_w{self.warps}_{self.route}_"
            f"d{self.depth}_g{self.gap}"
        )

    def manifest_value(self) -> dict[str, object]:
        return {
            **asdict(self),
            "key": self.key,
            "timing_schema": TIMING_SCHEMA,
            "arms": ["enabled", "off"],
            "bursts": MEASURED_BURSTS,
            "warmup": FIXED_WARMUP,
            "direct_repetitions": DIRECT_REPETITIONS,
            "ncu_repeats_min": NCU_BASE_REPETITIONS,
            "ncu_repeats_max": NCU_EXTENDED_REPETITIONS,
        }


@dataclass(frozen=True, slots=True)
class DirectBundle:
    role: str
    key: str
    cells: tuple[Cell, ...]

    def ordered_keys(self, order: str) -> list[str]:
        keys = [cell.key for cell in self.cells]
        if self.role == "placement":
            # Keep the W1 reference phases first even on reverse repetitions;
            # only the order within each warp-count block is reversed.  The
            # final W1 phase's wrapper barrier then remains the explicit
            # rendezvous before W4 enters.
            w1 = [cell.key for cell in self.cells if cell.warps == 1]
            w4 = [cell.key for cell in self.cells if cell.warps == 4]
            if order == "reverse":
                w1.reverse()
                w4.reverse()
            elif order != "forward":
                raise CampaignError(f"unknown order {order!r}")
            return w1 + w4
        if self.role == "route_zero_gap":
            # Keep the unopened L2 holdout last in both directions.  Reverse
            # only the already-open L1/DRAM phases; the requested order is
            # recorded separately in order.json.
            open_routes = [
                cell.key for cell in self.cells if cell.route != "l2"
            ]
            holdout = [cell.key for cell in self.cells if cell.route == "l2"]
            if order == "reverse":
                open_routes.reverse()
            elif order != "forward":
                raise CampaignError(f"unknown order {order!r}")
            return open_routes + holdout
        if order == "reverse":
            keys.reverse()
        elif order != "forward":
            raise CampaignError(f"unknown order {order!r}")
        return keys

    @property
    def canonical_keys(self) -> list[str]:
        if self.role == "placement":
            return [item.key for item in self.cells if item.warps == 4]
        return [item.key for item in self.cells]

    @property
    def reference_keys(self) -> list[str]:
        if self.role == "placement":
            return [item.key for item in self.cells if item.warps == 1]
        return []


def configured_cells() -> list[Cell]:
    """Return the exact frozen 40-configuration logical matrix."""

    cells: list[Cell] = []

    # Identification: W1 x L1/DRAM x depth 1/4/5/6/8 at gap zero.
    for route in ("l1", "dram"):
        for depth in (1, 4, 5, 6, 8):
            cells.append(Cell("identification", "ldgsts", 1, route, depth, 0))

    # Spacing sentinels: W1 x L1/DRAM x depth 4/5/6 x gap 16/64.
    for route in ("l1", "dram"):
        for depth in (4, 5, 6):
            for gap in (16, 64):
                cells.append(Cell("spacing", "ldgsts", 1, route, depth, gap))

    # Sole unopened route holdout.
    for depth in (1, 4, 5, 6, 8):
        cells.append(Cell("holdout", "ldgsts", 1, "l2", depth, 0))

    # Predeclared placement diagnostic, not an identification holdout.
    for route in ("l1", "l2", "dram"):
        for depth in (4, 5, 6):
            cells.append(Cell("placement", "ldgsts", 4, route, depth, 0))

    # Independent-LDG clock assay.
    for depth in (4, 5, 6, 8):
        cells.append(Cell("clock_assay", "ldg", 1, "l1", depth, 0))

    keys = [cell.key for cell in cells]
    if len(cells) != 40 or len(set(keys)) != 40:
        raise CampaignError("burst-release matrix must contain 40 unique cells")
    return cells


def cell_index() -> dict[str, Cell]:
    return {cell.key: cell for cell in configured_cells()}


def cell(assay: str, warps: int, route: str, depth: int, gap: int) -> Cell:
    key = f"{assay}_w{warps}_{route}_d{depth}_g{gap}"
    try:
        return cell_index()[key]
    except KeyError as error:
        raise CampaignError(f"logical cell is not in the frozen matrix: {key}") from error


def direct_bundles() -> list[DirectBundle]:
    """Return matched-CTA bundles and their canonical forward order."""

    bundles: list[DirectBundle] = []

    # Identification and L2 holdout share one all-depth route launch.  This
    # makes every W1 P/S/K and adjacent-route effect a same-CTA contrast.
    bundles.append(
        DirectBundle(
            "route_zero_gap",
            "route_zero_gap",
            tuple(
                cell("ldgsts", 1, route, depth, 0)
                for route in ("l1", "dram", "l2")
                for depth in (1, 4, 5, 6, 8)
            ),
        )
    )

    # Each spacing gap retains both routes and all three knee depths together.
    for gap in (16, 64):
        bundles.append(
            DirectBundle(
                "spacing",
                f"spacing_g{gap}",
                tuple(
                    cell("ldgsts", 1, route, depth, gap)
                    for route in ("l1", "dram")
                    for depth in (4, 5, 6)
                ),
            )
        )

    # The W1 phases are diagnostic references; all nine W4 route/depth cells
    # are the canonical placement evidence.  Keeping all 18 phases in one CTA
    # closes both depth-knee and adjacent-route matched-SM contrasts.
    bundles.append(
        DirectBundle(
            "placement",
            "placement",
            tuple(
                cell("ldgsts", warps, route, depth, 0)
                for warps in (1, 4)
                for route in ("l1", "l2", "dram")
                for depth in (4, 5, 6)
            ),
        )
    )

    bundles.append(
        DirectBundle(
            "clock_assay",
            "clock_assay",
            tuple(cell("ldg", 1, "l1", depth, 0) for depth in (4, 5, 6, 8)),
        )
    )

    if len(bundles) != 5 or len({bundle.key for bundle in bundles}) != 5:
        raise CampaignError("direct campaign must contain five unique matched bundles")
    covered = {item.key for bundle in bundles for item in bundle.cells}
    expected = set(cell_index())
    if covered != expected:
        missing = sorted(expected - covered)
        extra = sorted(covered - expected)
        raise CampaignError(f"direct bundle coverage mismatch: missing={missing}, extra={extra}")
    return bundles


def target_cells_for_bundle(
    bundle: DirectBundle, order: str
) -> tuple[tuple[Cell, ...], str]:
    """Map a requested phase order to the target's simple forward/reverse CLI."""

    keys = bundle.ordered_keys(order)
    canonical = tuple(item.key for item in bundle.cells)
    if tuple(keys) == canonical:
        return bundle.cells, order
    # The placement reverse schedule keeps W1 references before W4.  Materialize
    # that phase order and ask the CUDA target to consume it in forward order;
    # reversing the whole 18-cell list would put W4 ahead of its rendezvous.
    return tuple(cell_index()[key] for key in keys), "forward"


def preflight_cells() -> list[Cell]:
    return [
        cell("ldgsts", warps, route, 5, 0)
        for warps in (1, 4)
        for route in ("l1", "dram", "l2")
    ]


def compact_predicate_positions(depth: int) -> list[int]:
    """Mirror the seven-predicate SM86 depth map for local contract checks."""

    predicates = (
        depth > 3,
        depth >= 4,
        not depth < 4,
        depth >= 5,
        depth > 5,
        depth > 6,
        depth >= 8,
    )
    return [0] + [index + 1 for index, enabled in enumerate(predicates) if enabled]


EXPECTED_RUNTIME_PREDICATES: tuple[str | None, ...] = (
    None,
    "@P0",
    "@P1",
    "@P2",
    "@P3",
    "@P4",
    "@P5",
    "@P6",
)


def normalize_runtime_predicate_shape(
    predicates: object,
) -> tuple[str | None, ...] | None:
    """Normalize only the intentional third-position polarity spelling."""

    if not isinstance(predicates, (list, tuple)) or len(predicates) != 8:
        return None
    values = tuple(predicates)
    if values[3] == "@!P2":
        values = values[:3] + ("@P2",) + values[4:]
    if values != EXPECTED_RUNTIME_PREDICATES:
        return None
    return values


def selected_cells(selector: str | None) -> list[Cell]:
    cells = configured_cells()
    if selector is None:
        return cells
    raw_values = selector.split(",")
    if not selector.strip() or any(not part.strip() for part in raw_values):
        raise CampaignError("--cells must contain one or more non-empty keys")
    values = [part.strip() for part in raw_values]
    if len(values) != len(set(values)):
        raise CampaignError("duplicate logical cell key in --cells")
    requested = set(values)
    selected = [item for item in cells if item.key in requested]
    missing = requested - {item.key for item in selected}
    if missing:
        raise CampaignError(f"unknown cell key(s): {', '.join(sorted(missing))}")
    return selected


def selected_bundles(selector: str | None) -> list[DirectBundle]:
    bundles = direct_bundles()
    if selector is None:
        return bundles
    raw_values = selector.split(",")
    if not selector.strip() or any(not part.strip() for part in raw_values):
        raise CampaignError("--bundles must contain one or more non-empty keys")
    values = [part.strip() for part in raw_values]
    if len(values) != len(set(values)):
        raise CampaignError("duplicate direct bundle key in --bundles")
    requested = set(values)
    selected = [item for item in bundles if item.key in requested]
    missing = requested - {item.key for item in selected}
    if missing:
        raise CampaignError(f"unknown bundle key(s): {', '.join(sorted(missing))}")
    return selected


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    """Atomically replace one small JSON metadata file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def command_record(command: Sequence[str]) -> dict[str, object]:
    return {"argv": list(command), "shell": shlex.join(command)}


def terminate_process_group(process: subprocess.Popen[object], *, grace: int = 25) -> bool:
    """Terminate and reap a bounded child even when descendants own pipes."""

    clean = True
    if os.name == "nt":
        try:
            taskkill = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=grace,
            )
            if taskkill.returncode not in (0, 128, 255) and process.poll() is None:
                clean = False
        except (OSError, subprocess.TimeoutExpired):
            clean = False
        try:
            process.terminate()
        except OSError:
            pass
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            clean = False
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        clean = False
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            clean = False
    else:
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        clean = False
    return clean and process.poll() is not None


def run_streamed(
    command: Sequence[str],
    destination: Path,
    *,
    timeout: int,
    stdout_name: str = "stdout.txt",
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[None]:
    """Run a bounded command without retaining large raw streams in memory."""

    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "command.json", command_record(command))
    started = now()
    with (destination / stdout_name).open("w", encoding="utf-8") as stdout, (
        destination / "stderr.txt"
    ).open("w", encoding="utf-8") as stderr:
        popen_kwargs: dict[str, object] = {
            "cwd": REPO_ROOT,
            "env": environment,
            "text": True,
            "stdout": stdout,
            "stderr": stderr,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        else:
            popen_kwargs["start_new_session"] = True
        process = subprocess.Popen(list(command), **popen_kwargs)
        timed_out = False
        cleanup_complete = True
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            cleanup_complete = terminate_process_group(process)
            return_code = process.returncode
        except BaseException:
            terminate_process_group(process)
            raise
    write_json(
        destination / "run.json",
        {
            "started_at": started,
            "finished_at": now(),
            "return_code": return_code,
            "timed_out": timed_out,
            "cleanup_complete": cleanup_complete,
            "timeout_seconds": timeout,
        },
    )
    if timed_out:
        raise CampaignError(f"command timed out after {timeout}s: {shlex.join(command)}")
    if return_code != 0:
        raise CampaignError(
            f"command failed with status {return_code}; see {destination / 'stderr.txt'}"
        )
    return subprocess.CompletedProcess(list(command), return_code)


def run_small(command: Sequence[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise CampaignError(f"command timed out: {shlex.join(command)}") from error
    if completed.returncode != 0:
        raise CampaignError(
            f"command failed with status {completed.returncode}: {shlex.join(command)}\n"
            f"{completed.stderr.strip()}"
        )
    return completed


def binary_stamp() -> dict[str, object] | None:
    """Return a lightweight build identity for resumable capture directories."""

    try:
        stat = BINARY.stat()
    except OSError:
        return None
    if not BINARY.is_file() or stat.st_size <= 0:
        return None
    return {
        "binary_path": str(BINARY),
        "binary_mtime_ns": stat.st_mtime_ns,
        "binary_size_bytes": stat.st_size,
    }


def complete_directory(
    path: Path, *, kind: str, key: str, current_binary: bool = False
) -> bool:
    marker = path / "complete.json"
    if not path.is_dir() or not marker.is_file():
        return False
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(value, dict):
        return False
    if value.get("complete") is not True or value.get("kind") != kind or value.get("key") != key:
        return False
    required = value.get("required_files")
    if not isinstance(required, list) or not required:
        return False
    if current_binary:
        stamp = binary_stamp()
        if stamp is None or any(value.get(field) != expected for field, expected in stamp.items()):
            return False
    return all(
        isinstance(relative, str)
        and (path / relative).is_file()
        and (path / relative).stat().st_size > 0
        for relative in required
    )


class AtomicDirectory(AbstractContextManager[Path]):
    """Publish one final directory only after its complete marker exists."""

    def __init__(self, final: Path, *, kind: str, key: str, force: bool) -> None:
        self.final = final
        self.kind = kind
        self.key = key
        self.force = force
        self.stage: Path | None = None
        self.required_files: list[str] = []

    def __enter__(self) -> Path:
        self.final.parent.mkdir(parents=True, exist_ok=True)
        if self.final.exists():
            if not self.force:
                raise CampaignError(
                    f"existing directory is not being replaced: {self.final}; "
                    "resume skips only complete directories, or pass --force"
                )
            if self.final.is_dir():
                shutil.rmtree(self.final)
            else:
                self.final.unlink()
        self.stage = Path(
            tempfile.mkdtemp(prefix=f".{self.key}.partial-", dir=self.final.parent)
        )
        return self.stage

    def publish(self, required_files: Iterable[Path | str], **metadata: object) -> None:
        if self.stage is None:
            raise CampaignError("atomic directory has not been entered")
        relative = [str(Path(item)) for item in required_files]
        for item in relative:
            path = self.stage / item
            if not path.is_file() or path.stat().st_size == 0:
                raise CampaignError(f"cannot publish incomplete artifact: {path}")
        self.required_files = relative
        write_json(
            self.stage / "complete.json",
            {
                "complete": True,
                "completed_at": now(),
                "kind": self.kind,
                "key": self.key,
                "required_files": relative,
                **metadata,
            },
        )
        os.rename(self.stage, self.final)
        self.stage = None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.stage is not None:
            if exc_type is None:
                shutil.rmtree(self.stage, ignore_errors=True)
            else:
                # Keep failed scientific output available for diagnosis rather
                # than silently deleting a large raw stream.  Failed siblings
                # are hidden from normal resume discovery and carry no
                # completion marker.
                candidate = self.final.with_name(f"{self.final.name}.failed")
                suffix = 1
                while candidate.exists():
                    candidate = self.final.with_name(
                        f"{self.final.name}.failed-{suffix}"
                    )
                    suffix += 1
                try:
                    os.rename(self.stage, candidate)
                    write_json(
                        candidate / "failure.json",
                        {
                            "failed_at": now(),
                            "kind": self.kind,
                            "key": self.key,
                            "exception_type": getattr(exc_type, "__name__", str(exc_type)),
                            "exception": str(exc),
                        },
                    )
                except OSError:
                    shutil.rmtree(self.stage, ignore_errors=True)
            self.stage = None


def target_command(
    *,
    mode: str,
    cells: Sequence[Cell],
    repetition: int,
    order: str,
    bursts: int = MEASURED_BURSTS,
    warmup: int = FIXED_WARMUP,
) -> list[str]:
    if not cells:
        raise CampaignError("target invocation requires at least one logical cell")
    assays = {item.assay for item in cells}
    if len(assays) != 1:
        raise CampaignError("one target invocation cannot mix LDGSTS and LDG assays")
    return [
        str(BINARY),
        "--mode",
        mode,
        "--assay",
        next(iter(assays)),
        "--cells",
        ",".join(item.key for item in cells),
        "--bursts",
        str(bursts),
        "--warmup",
        str(warmup),
        "--repetition",
        str(repetition),
        "--order",
        order,
    ]


def sass_pc(row: Mapping[str, object]) -> int:
    """Sort malformed sidecar rows without raising from the report builder."""

    value = row.get("pc")
    return value if isinstance(value, int) and not isinstance(value, bool) else -1


def first_drain_clock(
    rows: Sequence[dict[str, object]],
    lower_pc: int,
) -> dict[str, object] | None:
    """Find the clock immediately following this body's first dependency wait.

    A function can contain several loop-carried ``DEPBAR`` instructions.  The
    timed body ends at the first dependency-barrier sequence after its issue
    endpoint; using the last barrier in the function would silently report a
    later loop clock as the drain sample.
    """

    ordered_rows = sorted(rows, key=sass_pc)
    depbars = [
        row
        for row in ordered_rows
        if isinstance(row.get("pc"), int)
        and row["pc"] > lower_pc
        and "DEPBAR" in str(row.get("opcode_text", ""))
    ]
    first_depbar = next(
        (row for row in depbars if "DEPBAR.LE" in str(row.get("opcode_text", ""))),
        depbars[0] if depbars else None,
    )
    if not isinstance(first_depbar, dict) or not isinstance(first_depbar.get("pc"), int):
        return None
    return next(
        (
            row
            for row in ordered_rows
            if isinstance(row.get("pc"), int)
            and row["pc"] > first_depbar["pc"]
            and "SR_CLOCKLO" in str(row.get("opcode_text", ""))
        ),
        None,
    )


def ncu_profile_command(item: Cell, repeat: int, report_base: Path) -> list[str]:
    return [
        NCU,
        "--force-overwrite",
        "--export",
        str(report_base),
        "--profile-from-start",
        "off",
        "--launch-count",
        "1",
        "--replay-mode",
        "application",
        "--clock-control",
        "none",
        "--cache-control",
        "none",
        "--metrics",
        ",".join(NCU_METRICS),
        *target_command(mode="ncu", cells=[item], repetition=repeat, order="forward"),
    ]


def ncu_export_command(report: Path, export: Path) -> list[str]:
    return [
        NCU,
        "--import",
        str(report),
        "--csv",
        "--page",
        "raw",
        "--print-metric-instances",
        "details",
        "--log-file",
        str(export),
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ncu_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = [
                {
                    str(key).strip(): str(value).strip()
                    for key, value in row.items()
                }
                for row in csv.DictReader(handle)
            ]
    except (OSError, csv.Error) as error:
        raise CampaignError(f"cannot parse NCU CSV: {path}") from error
    return [row for row in rows if row.get("Kernel Name")]


def _ncu_number(row: Mapping[str, str], name: str) -> int:
    value = row.get(name)
    if value is None:
        raise CampaignError(f"NCU CSV is missing metric {name}")
    try:
        return int(round(float(value.replace(",", ""))))
    except ValueError as error:
        raise CampaignError(f"NCU metric {name} is not numeric: {value!r}") from error


def _published_command(command: Sequence[str], stage: Path, final: Path) -> list[str]:
    """Normalize temporary atomic-stage paths to the published output path."""

    stage_text = str(stage)
    final_text = str(final)
    return [value.replace(stage_text, final_text) for value in command]


def _rewrite_command_record(path: Path, stage: Path, final: Path) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("argv"), list):
        raise CampaignError(f"NCU command record is malformed: {path}")
    published = _published_command(document["argv"], stage, final)
    document["argv"] = published
    document["shell"] = shlex.join(published)
    write_json(path, document)


def run_ncu(
    output: Path,
    cells: Sequence[Cell],
    *,
    gpu: str,
    force: bool,
) -> None:
    """Collect the frozen base NCU repeats, retaining complete provenance."""

    if not Path(NCU).is_file():
        raise CampaignError(f"trusted NCU executable is missing: {NCU}")
    environment = {
        "HOME": os.environ.get("HOME", "/home/jakeshea"),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/local/cuda-13.2/bin:/opt/cuda/bin:/usr/bin:/bin",
        "LD_LIBRARY_PATH": "/opt/cuda/lib64",
        "CUDA_VISIBLE_DEVICES": gpu,
    }
    for index, item in enumerate(cells, 1):
        for repeat in range(1, NCU_BASE_REPETITIONS + 1):
            final = output / "ncu" / item.key / f"repeat-{repeat}"
            if complete_directory(final, kind="ncu", key=f"{item.key}:repeat-{repeat}", current_binary=True) and not force:
                print(f"ncu {index:02d}/{len(cells):02d} {item.key} r{repeat} (existing)")
                continue
            manager = AtomicDirectory(
                final,
                kind="ncu",
                key=f"{item.key}:repeat-{repeat}",
                force=force,
            )
            with manager as stage:
                report = stage / "ncu-report.ncu-rep"
                profile_command = ncu_profile_command(item, repeat, stage / "ncu-report")
                run_streamed(
                    profile_command,
                    stage,
                    timeout=900,
                    stdout_name="profile.stdout.txt",
                    environment=environment,
                )
                if not report.is_file() or report.stat().st_size == 0:
                    raise CampaignError(f"NCU did not produce a report: {report}")
                export_dir = stage / "export"
                export_command = ncu_export_command(report, stage / "ncu.csv")
                run_streamed(
                    export_command,
                    export_dir,
                    timeout=900,
                    stdout_name="stdout.txt",
                    environment=environment,
                )
                csv_path = stage / "ncu.csv"
                rows = _ncu_rows(csv_path)
                if len(rows) != 1:
                    raise CampaignError(
                        f"{item.key} r{repeat}: expected one NCU kernel row, got {len(rows)}"
                    )
                row = rows[0]
                observed = {
                    metric: _ncu_number(row, metric) for metric in NCU_METRICS
                }
                published_profile_command = _published_command(
                    profile_command, stage, final
                )
                published_export_command = _published_command(
                    export_command, stage, final
                )
                _rewrite_command_record(stage / "command.json", stage, final)
                _rewrite_command_record(stage / "export" / "command.json", stage, final)
                metadata = {
                    "schema_version": 1,
                    "cell": item.key,
                    "repetition": repeat,
                    "command": published_profile_command,
                    "export_command": published_export_command,
                    "environment": environment,
                    "timeout_seconds": 900,
                    "timed_out": False,
                    "export_timed_out": False,
                    "export_return_code": 0,
                    "return_code": 0,
                    "report": str(report.relative_to(stage)),
                    "csv": str(csv_path.relative_to(stage)),
                    "report_sha256": _sha256_file(report),
                    "csv_sha256": _sha256_file(csv_path),
                    "observed": observed,
                    "identity": {
                        key: row.get(key)
                        for key in (
                            "Process Name",
                            "Kernel Name",
                            "Device",
                            "CC",
                            "Grid Size",
                            "Block Size",
                            "launch__thread_count",
                        )
                    },
                    "comparison_only": True,
                }
                write_json(stage / "metadata.json", metadata)
                manager.publish(
                    (
                        "profile.stdout.txt",
                        "run.json",
                        "ncu-report.ncu-rep",
                        "ncu.csv",
                        "export/command.json",
                        "export/run.json",
                        "metadata.json",
                    ),
                    cell=item.key,
                    repetition=repeat,
                    tool_path=NCU,
                    report="ncu-report.ncu-rep",
                    csv="ncu.csv",
                    report_sha256=metadata["report_sha256"],
                    csv_sha256=metadata["csv_sha256"],
                    observed=observed,
                    environment=environment,
                    timeout_seconds=900,
                    comparison_only=True,
                    **(binary_stamp() or {"binary_path": str(BINARY)}),
                )
            print(f"ncu {index:02d}/{len(cells):02d} {item.key} r{repeat}", flush=True)


def last_json(path: Path) -> dict[str, object]:
    # Direct captures can contain millions of paired rows.  Keep only the
    # latest parseable object instead of materializing the whole file.
    latest: dict[str, object] | None = None
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as error:
        raise CampaignError(f"cannot read target output: {path}") from error
    with handle:
        for line in handle:
            if not line.lstrip().startswith("{"):
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                latest = value
    if latest is not None:
        return latest
    raise CampaignError(f"target emitted no JSON object: {path}")


def validate_summary(
    path: Path,
    expected_cells: Sequence[Cell],
    *,
    mode: str,
    repetition: int,
    order: str,
    bursts: int,
    warmup: int,
) -> dict[str, object]:
    summary = last_json(path)
    if summary.get("type") != "summary" or summary.get("correct") is not True:
        raise CampaignError(f"target summary is incomplete or incorrect: {path}")
    observed = summary.get("cell_keys")
    expected = [item.key for item in expected_cells]
    if observed != expected:
        raise CampaignError(
            f"target summary cell order mismatch: expected {expected}, observed {observed}"
        )
    assays = {item.assay for item in expected_cells}
    if len(assays) != 1:
        raise CampaignError("target summary expectation must contain one assay")
    expected_states = sum(item.warps for item in expected_cells)
    expected_records = bursts * expected_states if mode == "direct" else 0
    bindings = {
        "mode": mode,
        "assay": next(iter(assays)),
        "timing_schema": TIMING_SCHEMA,
        "bursts": bursts,
        "warmup": warmup,
        "repetition": repetition,
        "order": order,
        "record_count": expected_records,
    }
    for field, value in bindings.items():
        if summary.get(field) != value:
            raise CampaignError(
                f"target summary {field} mismatch: expected {value!r}, "
                f"observed {summary.get(field)!r}"
            )
    states = summary.get("cell_states")
    if not isinstance(states, list) or len(states) != expected_states:
        raise CampaignError("target summary state cardinality mismatch")
    expected_pairs = {
        (item.key, warp) for item in expected_cells for warp in range(item.warps)
    }
    observed_pairs: set[tuple[str, int]] = set()
    for state in states:
        if not isinstance(state, dict):
            raise CampaignError("target summary contains a malformed state")
        state_cell = state.get("cell")
        state_warp = state.get("warp")
        if (
            not isinstance(state_cell, str)
            or not isinstance(state_warp, int)
            or isinstance(state_warp, bool)
        ):
            raise CampaignError("target summary state identity is malformed")
        pair = (state_cell, state_warp)
        if pair in observed_pairs or pair not in expected_pairs:
            raise CampaignError("target summary contains a duplicate/unexpected state")
        observed_pairs.add(pair)
        if (
            state.get("errors") != 0
            or state.get("smid_before") != state.get("smid_after")
        ):
            raise CampaignError("target summary state failed correctness/SM stability")
    if observed_pairs != expected_pairs:
        raise CampaignError("target summary is missing a state")
    return summary


def emit_manifest(output: Path, *, gpu: str = "0") -> None:
    matrix = {
        "schema_version": 1,
        "cells": [item.manifest_value() for item in configured_cells()],
        "roles": {
            "identification": 10,
            "spacing": 12,
            "holdout": 5,
            "placement": 9,
            "clock_assay": 4,
        },
        "direct_bundles": [
            {
                "key": bundle.key,
                "role": bundle.role,
                "canonical_cells": [item.key for item in bundle.cells],
                "analysis_cells": bundle.canonical_keys,
                "reference_cells": bundle.reference_keys,
                "orders": list(ORDER_SCHEDULE),
            }
            for bundle in direct_bundles()
        ],
        "ncu": {
            "metrics": list(NCU_METRICS),
            "replay_mode": "application",
            "clock_control": "none",
            "cache_control": "none",
            "metric_instances": "details",
            "base_repetitions": NCU_BASE_REPETITIONS,
            "extended_repetitions": NCU_EXTENDED_REPETITIONS,
            "active_cycle_cv_extension_threshold": ACTIVE_CYCLE_CV_LIMIT,
            "repeat_1_preflight_cells": [item.key for item in preflight_cells()],
        },
        "clock_policy": {
            "core_mhz": FIXED_CORE_MHZ,
            "memory_mhz": FIXED_MEMORY_MHZ,
            "host_lock": True,
            "restore": "reset locks by default; observed-frequency re-lock is explicit",
        },
        "raw_effects": {
            "timing_schema": TIMING_SCHEMA,
            "issue": "issue_on - issue_off",
            "completion": "(drain_on - issue_on) - (drain_off - issue_off)",
            "drain_field": "endpoint duration from each arm start",
            "independent_unit": "process repetition",
        },
        "execution_surfaces": {
            "compact_zero_gap": "eight-position seven-predicate body",
            "compact_predicate_depth_domain": {
                str(depth): compact_predicate_positions(depth)
                for depth in (1, 4, 5, 6, 8)
            },
            "direct_zero_gap": "same eight-position seven-predicate body",
            "gapped_target": "one inline-PTX copy/clock region per depth",
            "timing_work_equivalence": "structural zero-gap check is recorded in static report",
        },
    }
    write_json(output / "metadata" / "matrix.json", matrix)

    environment: dict[str, object] = {"captured_at": now()}
    probes = {
        "nvidia_smi": [
            NVIDIA_SMI,
            "-i",
            gpu,
            "--query-gpu=name,uuid,compute_cap,driver_version,persistence_mode,clocks.current.sm,clocks.current.memory",
            "--format=csv,noheader,nounits",
        ],
        "nvcc": [str(NVCC), "--version"],
        "ncu": [NCU, "--version"],
    }
    for name, command in probes.items():
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
                check=False,
            )
            environment[name] = {
                "command": command,
                "return_code": completed.returncode,
                "output": completed.stdout.strip(),
            }
        except subprocess.TimeoutExpired as error:
            environment[name] = {
                "command": command,
                "return_code": None,
                "timed_out": True,
                "output": str(error),
            }
    write_json(output / "metadata" / "environment.json", environment)


class FixedClocks(AbstractContextManager["FixedClocks"]):
    """Apply the project operating point and restore the prior clock policy."""

    QUERY = (
        "--query-gpu=index,name,uuid,persistence_mode,clocks.current.sm,"
        "clocks.current.memory,clocks.max.sm,clocks.max.memory"
    )

    def __init__(
        self,
        *,
        gpu: str,
        record: Path,
        restore_core: str,
        restore_memory: str,
    ) -> None:
        self.gpu = gpu
        self.record = record
        self.restore_core = restore_core
        self.restore_memory = restore_memory
        self.before: dict[str, object] = {}
        self.commands: list[dict[str, object]] = []
        self.old_handlers: dict[int, object] = {}
        self.lock_errors: list[str] = []
        self.lock_applied = False
        self.applied_core = False
        self.applied_memory = False
        self.closed = False

    def _query(self) -> dict[str, object]:
        command = [NVIDIA_SMI, "-i", self.gpu, self.QUERY, "--format=csv,noheader,nounits"]
        completed = run_small(command)
        rows = [part.strip() for part in completed.stdout.strip().split(",")]
        if len(rows) != 8:
            raise CampaignError(f"unexpected nvidia-smi clock row: {completed.stdout!r}")
        try:
            sm = int(rows[4])
            memory = int(rows[5])
            max_sm = int(rows[6])
            max_memory = int(rows[7])
        except ValueError as error:
            raise CampaignError(f"non-numeric nvidia-smi clock row: {completed.stdout!r}") from error
        return {
            "command": command,
            "raw": completed.stdout.strip(),
            "index": rows[0],
            "name": rows[1],
            "uuid": rows[2],
            "persistence_mode": rows[3],
            "sm_mhz": sm,
            "memory_mhz": memory,
            "max_sm_mhz": max_sm,
            "max_memory_mhz": max_memory,
        }

    def _mutate(self, command: Sequence[str], purpose: str) -> None:
        completed = run_small(command)
        self.commands.append(
            {
                "purpose": purpose,
                "command": list(command),
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
        )

    def _signal(self, signum: int, frame: object) -> None:
        del frame
        raise CampaignInterrupted(signum)

    def __enter__(self) -> "FixedClocks":
        self.before = self._query()
        self.before["captured_at"] = now()
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            self.old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._signal)
        try:
            lock_commands = (
                (
                    ["sudo", "-n", NVIDIA_SMI, "-i", self.gpu, "-lgc", f"{FIXED_CORE_MHZ},{FIXED_CORE_MHZ}"],
                    "lock project core clock",
                    "core",
                ),
                (
                    ["sudo", "-n", NVIDIA_SMI, "-i", self.gpu, "-lmc", f"{FIXED_MEMORY_MHZ},{FIXED_MEMORY_MHZ}"],
                    "lock project memory clock",
                    "memory",
                ),
            )
            for command, purpose, resource in lock_commands:
                try:
                    self._mutate(command, purpose)
                except (OSError, CampaignError) as error:
                    # A staging host may not grant nvidia-smi clock control.
                    # Preserve the typed limitation and continue with the
                    # observed operating point instead of losing the run.
                    detail = f"{purpose}: {error}"
                    self.lock_errors.append(detail)
                    self.commands.append(
                        {
                            "purpose": purpose,
                            "command": list(command),
                            "return_code": None,
                            "error": str(error),
                        }
                    )
                    continue
                if resource == "core":
                    self.applied_core = True
                else:
                    self.applied_memory = True
            if not self.lock_errors:
                locked = self._query()
                if locked["sm_mhz"] != FIXED_CORE_MHZ or locked["memory_mhz"] != FIXED_MEMORY_MHZ:
                    self.lock_errors.append(f"clock lock did not take effect: {locked}")
                else:
                    self.lock_applied = True
            if self.lock_errors:
                # Do not discard a useful live run merely because the optional
                # clock lock is unavailable.  The before/after query and this
                # list remain the reproducibility record.
                self.lock_applied = False
        except BaseException:
            self.close("enter_error")
            raise
        return self

    def close(self, reason: str) -> None:
        if self.closed:
            return
        self.closed = True
        restore_errors: list[str] = []

        def restore(command: list[str], purpose: str) -> None:
            try:
                self._mutate(command, purpose)
            except (OSError, CampaignError) as error:
                restore_errors.append(str(error))

        if self.applied_core and self.restore_core == "reset":
            restore(["sudo", "-n", NVIDIA_SMI, "-i", self.gpu, "-rgc"], "restore prior unlocked core policy")
        elif self.applied_core:
            prior = int(self.before["sm_mhz"])
            restore(
                ["sudo", "-n", NVIDIA_SMI, "-i", self.gpu, "-lgc", f"{prior},{prior}"],
                "restore prior observed core limit",
            )
        if self.applied_memory and self.restore_memory == "reset":
            restore(["sudo", "-n", NVIDIA_SMI, "-i", self.gpu, "-rmc"], "restore prior unlocked memory policy")
        elif self.applied_memory:
            prior = int(self.before["memory_mhz"])
            restore(
                ["sudo", "-n", NVIDIA_SMI, "-i", self.gpu, "-lmc", f"{prior},{prior}"],
                "restore prior observed memory limit",
            )

        try:
            after = self._query()
        except (OSError, CampaignError) as error:
            after = {"query_error": str(error)}
            restore_errors.append(str(error))
        write_json(
            self.record,
            {
                "started_at": self.before.get("captured_at", now()),
                "finished_at": now(),
                "reason": reason,
                "before": self.before,
                "project_lock": {"sm_mhz": FIXED_CORE_MHZ, "memory_mhz": FIXED_MEMORY_MHZ},
                "lock_applied": self.lock_applied,
                "lock_limitation": "; ".join(self.lock_errors) if self.lock_errors else None,
                "restore_core": self.restore_core,
                "restore_memory": self.restore_memory,
                "restore_skipped": not (self.applied_core or self.applied_memory),
                "commands": self.commands,
                "after": after,
                "restore_errors": restore_errors,
            },
        )
        for signum, handler in self.old_handlers.items():
            signal.signal(signum, handler)
        if restore_errors:
            print(
                "warning: fixed-clock restoration was incomplete; see "
                f"{self.record}",
                file=sys.stderr,
            )
        if restore_errors and reason == "success":
            raise CampaignError("clock restoration failed: " + "; ".join(restore_errors))

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        reason = "success" if exc_type is None else "signal" if exc_type is CampaignInterrupted else "error"
        self.close(reason)


def build(output: Path, *, force: bool) -> None:
    final = output / "build"
    if (
        complete_directory(final, kind="build", key="build", current_binary=True)
        and BINARY.is_file()
        and BINARY.stat().st_size > 0
        and not force
    ):
        print(f"build (existing): {final}")
        return
    manager = AtomicDirectory(final, kind="build", key="build", force=force)
    with manager as stage:
        run_streamed(["make", "-C", str(HERE), "clean", "all"], stage, timeout=600)
        if not BINARY.is_file() or BINARY.stat().st_size == 0:
            raise CampaignError(f"build did not produce {BINARY}")
        write_json(stage / "binary.json", {"path": str(BINARY), "size_bytes": BINARY.stat().st_size})
        manager.publish(("run.json", "command.json", "binary.json"), **(binary_stamp() or {}))
    print(f"build: {final}")


def control_sidecar_command(destination: Path) -> list[str]:
    """Return the pinned control-bit extraction command."""

    return [
        sys.executable,
        str(CONTROL_SIDECAR),
        "--binary",
        str(BINARY),
        "--arch",
        "sm_86",
        "--toolkit-bindir",
        str(TOOLKIT_BINDIR),
        "--out",
        str(destination),
    ]


def summarize_target_variants(sidecar: dict[str, object]) -> list[dict[str, object]]:
    """Extract compact, descriptive facts from decoded target functions."""

    functions = sidecar.get("functions")
    if not isinstance(functions, dict):
        return []
    pattern = re.compile(r"_Z9run_cellsILb1ELi([01])ELi(\d+)ELi(\d+)EE")
    variants: list[dict[str, object]] = []
    for name, raw_rows in functions.items():
        if not isinstance(name, str) or not isinstance(raw_rows, list):
            continue
        match = pattern.search(name)
        if match is None:
            continue
        assay_id, depth, gap = (int(value) for value in match.groups())
        target_opcode = "LDGSTS" if assay_id == 0 else "LDG.E"
        rows = [row for row in raw_rows if isinstance(row, dict)]
        loads = [
            row for row in rows if target_opcode in str(row.get("opcode_text", ""))
        ]
        pcs = [row.get("pc") for row in loads if isinstance(row.get("pc"), int)]
        between: list[dict[str, object]] = []
        if pcs:
            low, high = min(pcs), max(pcs)
            between = [
                row
                for row in rows
                if isinstance(row.get("pc"), int) and low <= row["pc"] <= high
            ]
        clocks = [
            row
            for row in rows
            if "SR_CLOCKLO" in str(row.get("opcode_text", ""))
        ]
        clock_pcs = [row.get("pc") for row in clocks if isinstance(row.get("pc"), int)]
        before_target = [pc for pc in clock_pcs if pcs and pc < min(pcs)]
        after_target = [pc for pc in clock_pcs if pcs and pc > max(pcs)]
        issue_clock = next(
            (
                row
                for row in sorted(clocks, key=sass_pc)
                if pcs
                and isinstance(row.get("pc"), int)
                and row["pc"] > max(pcs)
                and "LDGDEPBAR" not in str(row.get("opcode_text", ""))
                and "DEPBAR" not in str(row.get("opcode_text", ""))
            ),
            None,
        )
        drain_clock = first_drain_clock(
            rows,
            issue_clock["pc"]
            if isinstance(issue_clock, dict) and isinstance(issue_clock.get("pc"), int)
            else (max(pcs) if pcs else -1),
        )
        timed_low = max(before_target) if before_target else None
        timed_high = (
            drain_clock.get("pc")
            if isinstance(drain_clock, dict) and isinstance(drain_clock.get("pc"), int)
            else (max(pcs) if pcs else None)
        )
        pre_target_rows = [
            row
            for row in rows
            if before_target
            and isinstance(row.get("pc"), int)
            and max(before_target) < row["pc"] < min(pcs)
        ]
        variants.append(
            {
                "function": name,
                "surface": "compact",
                "body_variant": (
                    "runtime_predicated_8"
                    if assay_id == 0
                    and gap == 0
                    and len(loads) == 8
                    and any(row.get("predicate") for row in loads)
                    else "compile_time_prefix"
                    if assay_id == 0 and gap == 0
                    else "ldg_control"
                    if assay_id == 1
                    else "compile_time_gap_variant"
                ),
                "assay": "ldgsts" if assay_id == 0 else "ldg",
                "depth": depth,
                "gap": gap,
                "target_opcode": target_opcode,
                "target_in_function": len(loads),
                "target_predicates": [row.get("predicate") for row in loads],
                "normalized_predicate_shape": normalize_runtime_predicate_shape(
                    [row.get("predicate") for row in loads]
                ),
                "target_pc_contiguous": bool(
                    pcs
                    and pcs == list(range(pcs[0], pcs[0] + 16 * len(pcs), 16))
                ),
                "branch_or_call_between_targets": any(
                    any(token in str(row.get("opcode_text", "")) for token in ("BRA", "CALL"))
                    for row in between
                ),
                "clock_count": len(clocks),
                "all_clock_wait_masks": [
                    row.get("control", {}).get("wait_mask")
                    for row in clocks
                    if isinstance(row.get("control"), dict)
                    and isinstance(row.get("control", {}).get("wait_mask"), int)
                    and not isinstance(row.get("control", {}).get("wait_mask"), bool)
                ],
                "endpoint_clock_wait_masks": [
                    row.get("control", {}).get("wait_mask")
                    for row in clocks
                    if isinstance(row.get("pc"), int)
                    and (
                        (before_target and row["pc"] == max(before_target))
                        or (after_target and row["pc"] == min(after_target))
                    )
                    and isinstance(row.get("control"), dict)
                ],
                "start_clock_wait_masks": [
                    row.get("control", {}).get("wait_mask")
                    for row in clocks
                    if before_target
                    and isinstance(row.get("pc"), int)
                    and row["pc"] == max(before_target)
                    and isinstance(row.get("control"), dict)
                ],
                "issue_clock_wait_masks": [
                    issue_clock.get("control", {}).get("wait_mask")
                ]
                if isinstance(issue_clock, dict)
                and isinstance(issue_clock.get("control"), dict)
                else [],
                "drain_clock_wait_masks": [
                    drain_clock.get("control", {}).get("wait_mask")
                ]
                if isinstance(drain_clock, dict)
                and isinstance(drain_clock.get("control"), dict)
                else [],
                "barrier_pcs": [
                    row.get("pc")
                    for row in rows
                    if str(row.get("opcode_text", "")).startswith("BAR.")
                    and isinstance(row.get("pc"), int)
                ],
                "cta_barriers_in_timed_window": [
                    row.get("pc")
                    for row in rows
                    if timed_low is not None
                    and timed_high is not None
                    and isinstance(row.get("pc"), int)
                    and timed_low <= row["pc"] <= timed_high
                    and "BAR.SYNC" in str(row.get("opcode_text", ""))
                ],
                "pre_target_predicated_off_rows": [
                    row.get("opcode_text")
                    for row in pre_target_rows
                    if "@!PT" in str(row.get("predicate", ""))
                ],
                "predicate_setup_before_target": [
                    row.get("opcode_text")
                    for row in rows
                    if pcs
                    and isinstance(row.get("pc"), int)
                    and row["pc"] < min(pcs)
                    and "ISETP" in str(row.get("opcode_text", ""))
                ],
            }
        )
    return sorted(variants, key=lambda item: (int(item["gap"]), int(item["depth"])))


def summarize_direct_zero_gap(sidecar: dict[str, object]) -> list[dict[str, object]]:
    """Describe contiguous LDGSTS groups in the specialized direct entry."""

    functions = sidecar.get("functions")
    if not isinstance(functions, dict):
        return []
    name = "_Z9run_cellsILb0ELi0ELi0ELi0"
    raw_rows = next(
        (value for key, value in functions.items() if isinstance(key, str) and name in key),
        None,
    )
    if not isinstance(raw_rows, list):
        return []
    rows = [row for row in raw_rows if isinstance(row, dict)]
    loads = [row for row in rows if "LDGSTS" in str(row.get("opcode_text", ""))]
    pcs = sorted({row["pc"] for row in loads if isinstance(row.get("pc"), int)})
    groups: list[list[int]] = []
    for pc in pcs:
        if not groups or pc != groups[-1][-1] + 16:
            groups.append([pc])
        else:
            groups[-1].append(pc)
    result: list[dict[str, object]] = []
    for group in groups:
        if len(group) not in {1, 4, 5, 6, 8}:
            continue
        window = [
            row
            for row in rows
            if isinstance(row.get("pc"), int)
            and group[0] <= row["pc"] <= group[-1]
        ]
        clocks = [
            row for row in rows if "SR_CLOCKLO" in str(row.get("opcode_text", ""))
        ]
        issue_clock = next(
            (
                row
                for row in sorted(clocks, key=sass_pc)
                if isinstance(row.get("pc"), int)
                and row["pc"] > group[-1]
                and "LDGDEPBAR" not in str(row.get("opcode_text", ""))
                and "DEPBAR" not in str(row.get("opcode_text", ""))
            ),
            None,
        )
        drain_clock = first_drain_clock(
            rows,
            issue_clock["pc"] if isinstance(issue_clock, dict) and isinstance(issue_clock.get("pc"), int) else group[-1],
        )
        result.append(
            {
                "count": len(group),
                "surface": "direct",
                "body_variant": (
                    "runtime_predicated_8"
                    if any(row.get("predicate") for row in window)
                    else "compile_time_prefix"
                ),
                "pc_start": group[0],
                "pc_end": group[-1],
                "contiguous": group == list(range(group[0], group[0] + 16 * len(group), 16)),
                "branch_or_call_between_targets": any(
                    any(token in str(row.get("opcode_text", "")) for token in ("BRA", "CALL"))
                    for row in window
                ),
                "wait_masks_on_targets": [
                    row.get("control", {}).get("wait_mask")
                    for row in window
                    if isinstance(row.get("control"), dict)
                ],
                "target_predicates": [
                    row.get("predicate")
                    for row in window
                    if "LDGSTS" in str(row.get("opcode_text", ""))
                ],
                "normalized_predicate_shape": normalize_runtime_predicate_shape(
                    [
                        row.get("predicate")
                        for row in window
                        if "LDGSTS" in str(row.get("opcode_text", ""))
                    ]
                ),
                "issue_clock_wait_masks": [
                    issue_clock.get("control", {}).get("wait_mask")
                ]
                if isinstance(issue_clock, dict)
                and isinstance(issue_clock.get("control"), dict)
                else [],
                "drain_clock_wait_masks": [
                    drain_clock.get("control", {}).get("wait_mask")
                ]
                if isinstance(drain_clock, dict)
                and isinstance(drain_clock.get("control"), dict)
                else [],
                "pre_target_predicated_off_rows": [
                    row.get("opcode_text")
                    for row in rows
                    if isinstance(row.get("pc"), int)
                    and group[0] - 64 <= row["pc"] < group[0]
                    and "@!PT" in str(row.get("predicate", ""))
                ],
            }
        )
    return result


def run_static(output: Path, *, force: bool) -> None:
    if not BINARY.is_file():
        raise CampaignError(f"build first; binary is missing: {BINARY}")
    final = output / "static"
    if complete_directory(final, kind="static", key="static", current_binary=True) and not force:
        print(f"static (existing): {final}")
        return
    manager = AtomicDirectory(final, kind="static", key="static", force=force)
    with manager as stage:
        for name, option in (("sass.txt", "--dump-sass"), ("resources.txt", "--dump-resource-usage")):
            destination = stage / name.removesuffix(".txt")
            run_streamed([str(CUOBJDUMP), option, str(BINARY)], destination, timeout=120)
            shutil.copyfile(destination / "stdout.txt", stage / name)
        control_path = stage / "control_bits.json"
        control_command = control_sidecar_command(control_path)
        replay_command = list(control_command)
        replay_command[replay_command.index("--out") + 1] = str(
            (output / "static" / "control_bits.json").resolve()
        )
        started = now()
        try:
            completed = subprocess.run(
                control_command,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            write_json(
                stage / "control-sidecar-run.json",
                {
                    "command": command_record(control_command),
                    "replay_command": command_record(replay_command),
                    "started_at": started,
                    "finished_at": now(),
                    "return_code": None,
                    "timed_out": True,
                    "stdout": (error.stdout or "") if isinstance(error.stdout, str) else "",
                    "stderr": (error.stderr or "") if isinstance(error.stderr, str) else "",
                },
            )
            raise CampaignError("control-bit sidecar timed out") from error
        write_json(
            stage / "control-sidecar-run.json",
            {
                "command": command_record(control_command),
                "replay_command": command_record(replay_command),
                "started_at": started,
                "finished_at": now(),
                "return_code": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            },
        )
        if completed.returncode != 0:
            raise CampaignError(
                "control-bit sidecar failed; see "
                f"{stage / 'control-sidecar-run.json'}"
            )
        try:
            sidecar = json.loads(control_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CampaignError(f"control-bit sidecar is malformed: {control_path}") from error
        if not isinstance(sidecar, dict):
            raise CampaignError(f"control-bit sidecar must be an object: {control_path}")
        summary = sidecar.get("summary")
        if not isinstance(summary, dict):
            raise CampaignError(f"control-bit sidecar has no summary: {control_path}")
        functions = summary.get("functions")
        if not isinstance(functions, dict) or not functions:
            raise CampaignError(f"control-bit sidecar has no decoded functions: {control_path}")
        target_variants = summarize_target_variants(sidecar)
        direct_zero_gap = summarize_direct_zero_gap(sidecar)
        direct_symbol_present = any(
            isinstance(name, str)
            and "_Z9run_cellsILb0ELi0ELi0ELi0" in name
            for name in sidecar.get("functions", {})
        ) if isinstance(sidecar.get("functions"), dict) else False
        zero_gap = [
            item
            for item in target_variants
            if item["assay"] == "ldgsts" and item["gap"] == 0
        ]
        gapped = [
            item
            for item in target_variants
            if item["assay"] == "ldgsts" and item["gap"] != 0
        ]
        write_json(
            stage / "static-report.json",
            {
                "schema_version": 1,
                "report_kind": "observational_static_report",
                "generated_at": now(),
                "binary": str(BINARY),
                "binary_size_bytes": BINARY.stat().st_size,
                "binary_mtime_ns": BINARY.stat().st_mtime_ns,
                "arch": "sm_86",
                "sass_file": "sass.txt",
                "resource_file": "resources.txt",
                "control_bits_file": "control_bits.json",
                "control_bits": {
                    "function_count": len(functions),
                    "decoded": summary.get("total_decoded"),
                    "unknown_encoding": summary.get("total_unknown_encoding"),
                    "functions": {
                        name: {
                            "instructions": value.get("instructions"),
                            "decoded": value.get("decoded"),
                            "unknown_encoding": value.get("unknown_encoding"),
                        }
                        for name, value in functions.items()
                        if isinstance(value, dict)
                    },
                },
                "target_variants": target_variants,
                "direct_zero_gap_groups": direct_zero_gap,
                "findings": {
                    "zero_gap_ldgsts_variants": len(zero_gap),
                    "zero_gap_target_contiguous_observed_count": sum(
                        bool(item["target_pc_contiguous"])
                        and not item["branch_or_call_between_targets"]
                        for item in zero_gap
                    ),
                    "zero_gap_predicate_shapes_observed_count": sum(
                        item["normalized_predicate_shape"] is not None
                        for item in zero_gap
                    ),
                    "zero_gap_target_endpoint_clean_observed_count": sum(
                        bool(item["target_pc_contiguous"])
                        and not item["branch_or_call_between_targets"]
                        and not item["cta_barriers_in_timed_window"]
                        and len(item["issue_clock_wait_masks"]) == 1
                        and len(item["drain_clock_wait_masks"]) == 1
                        and all(
                            mask == 0
                            for mask in (
                                item["issue_clock_wait_masks"]
                                + item["drain_clock_wait_masks"]
                            )
                        )
                        for item in zero_gap
                    ),
                    "zero_gap_endpoint_wait_masks_observed": sorted(
                        {
                            int(mask)
                            for item in zero_gap
                            for mask in (
                                item["issue_clock_wait_masks"]
                                + item["drain_clock_wait_masks"]
                            )
                        }
                    ),
                    "zero_gap_start_wait_masks_observed": sorted(
                        {
                            int(mask)
                            for item in zero_gap
                            for mask in item["start_clock_wait_masks"]
                        }
                    ),
                    "gapped_clock_wait_masks_observed": sorted(
                        {
                            int(mask)
                            for item in gapped
                            for mask in item["all_clock_wait_masks"]
                            if int(mask) != 0
                        }
                    ),
                    "direct_zero_gap_groups_observed_count": len(direct_zero_gap),
                    "direct_zero_gap_target_clean_observed_count": sum(
                        bool(item["contiguous"])
                        and not item["branch_or_call_between_targets"]
                        and len(item["issue_clock_wait_masks"]) == 1
                        and len(item["drain_clock_wait_masks"]) == 1
                        and all(mask == 0 for mask in item["wait_masks_on_targets"])
                        for item in direct_zero_gap
                    ),
                    "direct_zero_gap_predicate_shapes_observed_count": sum(
                        item["normalized_predicate_shape"] is not None
                        for item in direct_zero_gap
                    ),
                    "direct_zero_gap_symbol_present": direct_symbol_present,
                    "unknown_control_encodings": summary.get("total_unknown_encoding"),
                    "interpretation": "observational; no pass/fail status is inferred",
                },
                "interpretation": (
                    "Descriptive build evidence only; it is not a permission token "
                    "or a substitute for scientific review of a live cell."
                ),
            },
        )
        manager.publish(
            (
                "sass.txt",
                "resources.txt",
                "control_bits.json",
                "control-sidecar-run.json",
                "static-report.json",
                "sass/run.json",
                "resources/run.json",
            ),
            **(binary_stamp() or {"binary_path": str(BINARY)}),
        )
    print(f"static: {final}")


def run_correctness(output: Path, cells: Sequence[Cell], *, force: bool) -> None:
    for index, item in enumerate(cells, 1):
        final = output / "correctness" / item.key
        if complete_directory(final, kind="correctness", key=item.key, current_binary=True) and not force:
            print(f"correctness {index:02d}/{len(cells):02d} {item.key} (existing)")
            continue
        manager = AtomicDirectory(final, kind="correctness", key=item.key, force=force)
        with manager as stage:
            command = target_command(
                mode="correctness",
                cells=[item],
                repetition=1,
                order="forward",
                bursts=64,
                warmup=4,
            )
            run_streamed(command, stage, timeout=180, stdout_name="stdout.jsonl")
            summary = validate_summary(
                stage / "stdout.jsonl",
                [item],
                mode="correctness",
                repetition=1,
                order="forward",
                bursts=64,
                warmup=4,
            )
            write_json(stage / "summary.json", summary)
            manager.publish(
                ("stdout.jsonl", "run.json", "summary.json"),
                **(binary_stamp() or {"binary_path": str(BINARY)}),
            )
        print(f"correctness {index:02d}/{len(cells):02d} {item.key}", flush=True)


def run_direct(output: Path, bundles: Sequence[DirectBundle], *, force: bool) -> None:
    for index, bundle in enumerate(bundles, 1):
        final = output / "direct" / bundle.key
        if complete_directory(final, kind="direct_bundle", key=bundle.key, current_binary=True) and not force:
            print(f"direct {index:02d}/{len(bundles):02d} {bundle.key} (existing)")
            continue
        manager = AtomicDirectory(final, kind="direct_bundle", key=bundle.key, force=force)
        with manager as stage:
            order_rows: list[dict[str, object]] = []
            required: list[str] = []
            for repeat, order in enumerate(ORDER_SCHEDULE, 1):
                repeat_dir = stage / f"repeat-{repeat}"
                effective_cells, target_order = target_cells_for_bundle(bundle, order)
                command = target_command(
                    mode="direct",
                    cells=effective_cells,
                    repetition=repeat,
                    order=target_order,
                )
                run_streamed(command, repeat_dir, timeout=1800, stdout_name="raw.jsonl")
                expected = list(effective_cells)
                summary = validate_summary(
                    repeat_dir / "raw.jsonl",
                    expected,
                    mode="direct",
                    repetition=repeat,
                    order=target_order,
                    bursts=MEASURED_BURSTS,
                    warmup=FIXED_WARMUP,
                )
                write_json(repeat_dir / "summary.json", summary)
                order_rows.append(
                    {
                        "repetition": repeat,
                        "order": order,
                        "canonical_cells": [item.key for item in bundle.cells],
                        "effective_cells": [item.key for item in effective_cells],
                        "target_order": target_order,
                        "arm_order_seed": repeat,
                    }
                )
                required.extend(
                    f"repeat-{repeat}/{name}"
                    for name in ("raw.jsonl", "run.json", "summary.json")
                )
            write_json(stage / "order.json", order_rows)
            required.append("order.json")
            manager.publish(
                required,
                role=bundle.role,
                body_variant=(
                    "runtime_predicated_8"
                    if bundle.role in {"route_zero_gap", "placement"}
                    else "direct_ldg_control"
                    if bundle.role == "clock_assay"
                    else "direct_gapped_or_control"
                ),
                repetitions=DIRECT_REPETITIONS,
                canonical_cells=[item.key for item in bundle.cells],
                analysis_cells=bundle.canonical_keys,
                reference_cells=bundle.reference_keys,
                **(binary_stamp() or {"binary_path": str(BINARY)}),
            )
        print(f"direct {index:02d}/{len(bundles):02d} {bundle.key}", flush=True)


def dry_run(
    action: str,
    cells: Sequence[Cell],
    bundles: Sequence[DirectBundle],
    *,
    gpu: str,
    restore_core: str,
    restore_memory: str,
) -> None:
    commands: list[list[str]] = []
    if action == "build":
        commands.append(["make", "-C", str(HERE)])
    if action == "static":
        commands.extend(
            [str(CUOBJDUMP), option, str(BINARY)]
            for option in ("--dump-sass", "--dump-resource-usage")
        )
        commands.append(control_sidecar_command(Path("OUTPUT") / "control_bits.json"))
    live = action in {"correctness", "direct", "ncu"}
    if live:
        commands.extend(
            (
                [NVIDIA_SMI, "-i", gpu, FixedClocks.QUERY, "--format=csv,noheader,nounits"],
                ["sudo", "-n", NVIDIA_SMI, "-i", gpu, "-lgc", f"{FIXED_CORE_MHZ},{FIXED_CORE_MHZ}"],
                ["sudo", "-n", NVIDIA_SMI, "-i", gpu, "-lmc", f"{FIXED_MEMORY_MHZ},{FIXED_MEMORY_MHZ}"],
            )
        )
    if action == "correctness":
        commands.extend(
            target_command(mode="correctness", cells=[item], repetition=1, order="forward", bursts=64, warmup=4)
            for item in cells
        )
    if action == "direct":
        commands.extend(
            target_command(
                mode="direct",
                cells=target_cells_for_bundle(bundle, order)[0],
                repetition=repeat,
                order=target_cells_for_bundle(bundle, order)[1],
            )
            for bundle in bundles
            for repeat, order in enumerate(ORDER_SCHEDULE, 1)
        )
    if action == "ncu":
        # Three commands are certain.  The manifest records the conditional
        # fourth/fifth repeats rather than pretending the CV decision is known.
        selected = {item.key: item for item in cells}
        preflight = [item for item in preflight_cells() if item.key in selected]
        remaining = [item for item in cells if item.key not in {entry.key for entry in preflight}]

        def add_profile(item: Cell, repeat: int) -> None:
            placeholder = Path("OUTPUT") / "ncu" / item.key / f"repeat-{repeat}"
            commands.append(ncu_profile_command(item, repeat, placeholder / "ncu-report"))
            commands.append(ncu_export_command(placeholder / "ncu-report.ncu-rep", placeholder / "ncu.csv"))

        # The six detailed repeat-1 smoke cells always precede all remaining
        # profiler commands.
        for item in preflight:
            add_profile(item, 1)
        for item in preflight:
            for repeat in (2, 3):
                add_profile(item, repeat)
        for item in remaining:
            for repeat in range(1, NCU_BASE_REPETITIONS + 1):
                add_profile(item, repeat)
    if live:
        core_restore = (
            ["sudo", "-n", NVIDIA_SMI, "-i", gpu, "-rgc"]
            if restore_core == "reset"
            else ["sudo", "-n", NVIDIA_SMI, "-i", gpu, "-lgc", "<prior-core>,<prior-core>"]
        )
        memory_restore = (
            ["sudo", "-n", NVIDIA_SMI, "-i", gpu, "-rmc"]
            if restore_memory == "reset"
            else ["sudo", "-n", NVIDIA_SMI, "-i", gpu, "-lmc", "<prior-memory>,<prior-memory>"]
        )
        commands.extend(
            (
                core_restore,
                memory_restore,
                [NVIDIA_SMI, "-i", gpu, FixedClocks.QUERY, "--format=csv,noheader,nounits"],
            )
        )
    for command in commands:
        if live and command and command[0] in {str(BINARY), NCU}:
            print(shlex.join(["env", f"CUDA_VISIBLE_DEVICES={gpu}", *command]))
        else:
            print(shlex.join(command))
    print(
        json.dumps(
            {
                "dry_run": True,
                "action": action,
                "commands": len(commands),
                "conditional_ncu_commands_per_high_cv_cell": 4,
                "live_gpu_touched": False,
            },
            sort_keys=True,
        )
    )


def self_test() -> None:
    if sass_pc({"pc": None}) != -1 or sass_pc({"pc": 16}) != 16:
        raise CampaignError("malformed SASS PC sort key is not fail-closed")
    cells = configured_cells()
    roles: dict[str, int] = {}
    for item in cells:
        roles[item.role] = roles.get(item.role, 0) + 1
        if item.assay not in {"ldgsts", "ldg"} or item.route not in {"l1", "l2", "dram"}:
            raise CampaignError(f"invalid frozen cell: {item}")
    expected_roles = {
        "identification": 10,
        "spacing": 12,
        "holdout": 5,
        "placement": 9,
        "clock_assay": 4,
    }
    if roles != expected_roles:
        raise CampaignError(f"role cardinality mismatch: {roles}")
    if len(preflight_cells()) != 6 or {item.warps for item in preflight_cells()} != {1, 4}:
        raise CampaignError("NCU preflight must contain the six W1/W4 depth-5 route cells")
    if ORDER_SCHEDULE.count("forward") != 4 or ORDER_SCHEDULE.count("reverse") != 3:
        raise CampaignError("seven direct repetitions must use balanced frozen order")
    bundles = direct_bundles()
    if [bundle.key for bundle in bundles] != [
        "route_zero_gap",
        "spacing_g16",
        "spacing_g64",
        "placement",
        "clock_assay",
    ]:
        raise CampaignError("five direct bundle keys/order are not frozen")
    if [len(bundle.cells) for bundle in bundles] != [15, 6, 6, 18, 4]:
        raise CampaignError("direct phase counts must be 15/6/6/18/4")
    expected_predicate_positions = {
        1: [0],
        4: [0, 1, 2, 3],
        5: [0, 1, 2, 3, 4],
        6: [0, 1, 2, 3, 4, 5],
        8: list(range(8)),
    }
    if any(
        compact_predicate_positions(depth) != positions
        for depth, positions in expected_predicate_positions.items()
    ):
        raise CampaignError("frozen predicate depth map is inconsistent")
    if normalize_runtime_predicate_shape(list(EXPECTED_RUNTIME_PREDICATES)) != EXPECTED_RUNTIME_PREDICATES:
        raise CampaignError("canonical runtime predicate shape was rejected")
    negated_shape = list(EXPECTED_RUNTIME_PREDICATES)
    negated_shape[3] = "@!P2"
    if normalize_runtime_predicate_shape(negated_shape) != EXPECTED_RUNTIME_PREDICATES:
        raise CampaignError("intentional third-predicate polarity was not normalized")
    for invalid_shape in (
        [*EXPECTED_RUNTIME_PREDICATES[:3], "@!P1", *EXPECTED_RUNTIME_PREDICATES[4:]],
        [*EXPECTED_RUNTIME_PREDICATES[:2], *EXPECTED_RUNTIME_PREDICATES[3:]],
    ):
        if normalize_runtime_predicate_shape(invalid_shape) is not None:
            raise CampaignError("invalid runtime predicate shape was accepted")

    route_expected = [
        cell("ldgsts", 1, route, depth, 0).key
        for route in ("l1", "dram", "l2")
        for depth in (1, 4, 5, 6, 8)
    ]
    if [item.key for item in bundles[0].cells] != route_expected:
        raise CampaignError("zero-gap bundle must retain every route/depth in one CTA")
    reverse_route = bundles[0].ordered_keys("reverse")
    if any(cell_index()[key].route == "l2" for key in reverse_route[:-5]) or not all(
        cell_index()[key].route == "l2" for key in reverse_route[-5:]
    ):
        raise CampaignError("reverse route order must keep the L2 holdout last")
    for bundle, gap in zip(bundles[1:3], (16, 64), strict=True):
        expected = [
            cell("ldgsts", 1, route, depth, gap).key
            for route in ("l1", "dram")
            for depth in (4, 5, 6)
        ]
        if [item.key for item in bundle.cells] != expected:
            raise CampaignError(f"{bundle.key}: spacing route/depth phases are not frozen")
    placement_expected = [
        cell("ldgsts", warps, route, depth, 0).key
        for warps in (1, 4)
        for route in ("l1", "l2", "dram")
        for depth in (4, 5, 6)
    ]
    if [item.key for item in bundles[3].cells] != placement_expected:
        raise CampaignError("placement phases must be frozen W1 references then W4 evidence")
    reverse_placement = bundles[3].ordered_keys("reverse")
    if not all(cell_index()[key].warps == 1 for key in reverse_placement[:9]) or not all(
        cell_index()[key].warps == 4 for key in reverse_placement[9:]
    ):
        raise CampaignError("reverse placement order must keep W1 before W4")
    effective, target_order = target_cells_for_bundle(bundles[3], "reverse")
    if target_order != "forward" or [item.key for item in effective] != reverse_placement:
        raise CampaignError("placement reverse schedule was not materialized safely")
    canonical = [key for bundle in bundles for key in bundle.canonical_keys]
    if len(canonical) != 40 or set(canonical) != set(cell_index()):
        raise CampaignError("each logical cell must have exactly one canonical direct source")
    if len(bundles[3].reference_keys) != 9 or any(
        cell_index()[key].warps != 1 for key in bundles[3].reference_keys
    ):
        raise CampaignError("placement must mark exactly nine W1 phases as references")
    sample = ncu_profile_command(preflight_cells()[0], 1, Path("report"))
    required_fragments = (
        ("--replay-mode", "application"),
        ("--clock-control", "none"),
        ("--cache-control", "none"),
    )
    for flag, value in required_fragments:
        if flag not in sample or sample[sample.index(flag) + 1] != value:
            raise CampaignError(f"NCU command lost {flag} {value}")
    export = ncu_export_command(Path("report.ncu-rep"), Path("ncu.csv"))
    if "--print-metric-instances" not in export or export[export.index("--print-metric-instances") + 1] != "details":
        raise CampaignError("NCU export must retain detailed metric instances")
    synthetic_rows = [
        {"pc": 100, "opcode_text": "LDGSTS.E.128", "control": {"wait_mask": 0}},
        {"pc": 116, "opcode_text": "CS2R R1, SR_CLOCKLO", "control": {"wait_mask": 0}},
        {"pc": 132, "opcode_text": "LDGDEPBAR", "control": {"wait_mask": 0}},
        {"pc": 148, "opcode_text": "DEPBAR.LE SB0, 0x0", "control": {"wait_mask": 0}},
        {"pc": 164, "opcode_text": "CS2R R2, SR_CLOCKLO", "control": {"wait_mask": 0}},
        {"pc": 800, "opcode_text": "DEPBAR.LE SB0, 0x0", "control": {"wait_mask": 0}},
        {"pc": 816, "opcode_text": "CS2R R3, SR_CLOCKLO", "control": {"wait_mask": 8}},
    ]
    drain = first_drain_clock(synthetic_rows, 116)
    if not isinstance(drain, dict) or drain.get("pc") != 164:
        raise CampaignError("drain-clock helper selected a later loop barrier")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        try:
            with AtomicDirectory(root / "failed-cell", kind="smoke", key="failed-cell", force=False) as stage:
                write_json(stage / "raw.jsonl", {"partial": True})
                raise RuntimeError("intentional self-test failure")
        except RuntimeError as error:
            if str(error) != "intentional self-test failure":
                raise
        failed = sorted(root.glob("failed-cell.failed*"))
        if len(failed) != 1 or not (failed[0] / "raw.jsonl").is_file() or not (
            failed[0] / "failure.json"
        ).is_file():
            raise CampaignError("failed campaign output was not retained for diagnosis")
    print(
        json.dumps(
            {
                "self_test": "passed",
                "logical_cells": len(cells),
                "direct_bundles": len(direct_bundles()),
                "direct_processes": len(direct_bundles()) * DIRECT_REPETITIONS,
                "ncu_preflight_cells": len(preflight_cells()),
                "ncu_min_profiles": len(cells) * NCU_BASE_REPETITIONS,
            },
            sort_keys=True,
        )
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "action",
        nargs="?",
        default="manifest",
        choices=("manifest", "build", "static", "correctness", "direct", "ncu"),
    )
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--cells", help="comma-separated exact logical cell keys")
    result.add_argument("--bundles", help="comma-separated exact direct bundle keys")
    result.add_argument("--gpu", default="0")
    result.add_argument("--force", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--self-test", action="store_true")
    result.add_argument(
        "--restore-core",
        choices=("observed", "reset"),
        default="reset",
        help="reset the core lock by default; 'observed' re-locks the pre-run frequency",
    )
    result.add_argument(
        "--restore-memory",
        choices=("observed", "reset"),
        default="reset",
        help="reset the memory lock by default; 'observed' re-locks the pre-run frequency",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.self_test:
            self_test()
            return 0
        if args.action == "direct" and args.cells is not None:
            raise CampaignError("direct selects matched bundles with --bundles, not --cells")
        if args.action in {"correctness", "ncu"} and args.bundles is not None:
            raise CampaignError(f"{args.action} selects logical cells with --cells, not --bundles")
        if args.action in {"manifest", "build", "static"} and (
            args.cells is not None or args.bundles is not None
        ):
            raise CampaignError(f"{args.action} does not accept cell or bundle selectors")
        cells = selected_cells(args.cells)
        bundles = selected_bundles(args.bundles)
        if args.dry_run:
            dry_run(
                args.action,
                cells,
                bundles,
                gpu=args.gpu,
                restore_core=args.restore_core,
                restore_memory=args.restore_memory,
            )
            return 0
        output = args.output_root.resolve()
        emit_manifest(output, gpu=args.gpu)
        if args.action == "manifest":
            print(output / "metadata/matrix.json")
            return 0
        if args.action == "build":
            build(output, force=args.force)
        if args.action == "static":
            run_static(output, force=args.force)
        if args.action in {"correctness", "direct", "ncu"}:
            if not BINARY.is_file() or BINARY.stat().st_size == 0:
                raise CampaignError(f"build first; binary is missing: {BINARY}")
            prior_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
            try:
                with FixedClocks(
                    gpu=args.gpu,
                    record=output / "metadata" / f"clock-session-{args.action}.json",
                    restore_core=args.restore_core,
                    restore_memory=args.restore_memory,
                ):
                    if args.action == "correctness":
                        run_correctness(output, cells, force=args.force)
                    if args.action == "direct":
                        run_direct(output, bundles, force=args.force)
                    if args.action == "ncu":
                        run_ncu(output, cells, gpu=args.gpu, force=args.force)
            finally:
                if prior_visible is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = prior_visible
        return 0
    except CampaignInterrupted as error:
        print(f"burst-release campaign: {error}", file=sys.stderr)
        return 128 + error.signum
    except (CampaignError, OSError, json.JSONDecodeError, ValueError) as error:
        print(f"burst-release campaign: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
