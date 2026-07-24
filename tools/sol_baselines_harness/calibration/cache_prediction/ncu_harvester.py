"""NCU CLI harvester for the cache-predictor calibration head.

Wraps the `ncu --csv ...` subprocess invocation used to gather
read-only sector hit-rate labels on sm_86 / sm_120 hosts per
the CLAUDE.md Rule 1 carveout. The deployed framework path
and every B200 invocation MUST NOT import this module.

`harvest_ncu_label(binary, *, launch_skip_count, ...)` runs:

  ncu --launch-skip N --launch-count 1 --clock-control none
      --replay-mode kernel --section MemoryWorkloadAnalysis
      --csv [binary_args...]

then feeds the captured stdout to the companion `label_parser`
module to produce a typed `NcuLabelRecord`.

The module's `harvest_ncu_label` raises
`NcuHarvesterError("ncu_label_invocation_failed", ...)` when
the `ncu` binary is absent from `$PATH` or the subprocess
exits non-zero. The parser-side typed reasons
(`ncu_label_metric_missing`, `ncu_label_denominator_zero`,
`ncu_label_hit_rate_above_100`, `ncu_label_csv_malformed`)
propagate through unchanged.

Importing this module does NOT invoke NCU. Subprocess
execution happens only when `harvest_ncu_label` is called.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from sol_baselines_harness.calibration.cache_prediction.label_parser import (
    NCU_LABEL_PARSER_TYPED_REASONS,
    NcuLabelParserError,
    NcuLabelRecord,
    parse_ncu_csv,
)


# The invocation form is fixed by the predictor's label
# contract: launch-skip N + count 1; clock-control none (no
# clock-policy mutation per CLAUDE.md Rule 2); replay-mode
# kernel (deterministic per-launch replay); the
# MemoryWorkloadAnalysis section is the one that exports the
# five required read-only sector metrics.
#
# This tuple is a canonical reference constant documenting the
# pinned NCU CLI shape. It is NOT consumed at runtime: the
# `_build_invocation` builder constructs the argv tuple
# independently below, inserting the per-call
# `launch_skip_count`, `--metrics <selector>`, binary path, and
# binary_args. Keep this constant and `_build_invocation` in
# sync conceptually; they encode the same calibration
# discipline.
NCU_INVOCATION_TEMPLATE: tuple[str, ...] = (
    "ncu",
    "--launch-skip",
    "--launch-count",
    "1",
    "--clock-control",
    "none",
    "--replay-mode",
    "kernel",
    "--section",
    "MemoryWorkloadAnalysis",
    "--csv",
)


NCU_PROVENANCE_TYPED_REASONS: frozenset[str] = frozenset({
    "ncu_provenance_missing_field",
    "ncu_provenance_invalid_field_value",
})


NCU_HARVESTER_TYPED_REASONS: frozenset[str] = (
    NCU_LABEL_PARSER_TYPED_REASONS
    | frozenset({"ncu_label_invocation_failed"})
    | NCU_PROVENANCE_TYPED_REASONS
)


# Re-exported convenience: the set of NCU metric columns the
# parser requires. The L1 path filters global loads explicitly
# (`pipe_lsu_mem_global_op_ld`) so store traffic stays out of the
# numerator and denominator; the L2 path keeps the `op_read`
# filter. NCU 2026.1 query-metrics verified these names: the
# earlier `lts__t_sectors_op_read_hit.sum` family is not a
# real metric on this NCU; the canonical hit counter is
# `_lookup_hit`. The combined `..._sector_hit_rate.pct` family
# (no `_op_read` / `_op_ld` qualifier) remains forbidden by the
# parser because it conflates store traffic under L2's
# no-write-allocate write-back policy.
# The L2 metrics filter by the L1TEX source unit (`srcunit_tex`)
# so the denominator counts only LSU-pipe-originated L2 read
# sectors - the traffic class the NVBit Tier-2 producer can
# observe at the SASS-execute-time injection surface. On Ampere
# sm_86 the L1 cache is unified with the texture cache as
# L1TEX; NCU tags LSU global LDG traffic that misses L1TEX and
# reaches L2 as `srcunit_tex`, not `srcunit_l1`. Empirical
# verification via direct `ncu --csv --metrics ...` against
# `calib_smoke_coalesced_loop` confirms `srcunit_tex_op_read.sum`
# returns 4 sectors (matching the L1
# `pipe_lsu_mem_global_op_ld.sum=4`) while `srcunit_l1_op_read.sum`
# returns 0. The previous total-L2-read metric
# (`lts__t_sectors_op_read.sum`) counted every L2 read
# transaction including i-cache cold-launch fetches and
# MOV/IMAD-with-CBANK constant operands, which NVBit cannot
# instrument; that mismatch left ~97% of NCU's denominator
# uncovered by the producer.
#
# NCU 2026.1.0.0 publishes:
#   lts__t_sectors_srcunit_tex_op_read.sum
#     # of LTS sectors from unit L1TEX for reads
#   lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum
#     # of LTS sectors from unit L1TEX for reads that hit
#
# NCU 2026.1.0.0 does NOT publish a corresponding
# `srcunit_tex_op_read_hit_rate.pct` column; the parser falls
# back to deriving the hit rate from the sum counters when the
# .pct column is absent (label_parser._resolve_level_hit_rate).
NCU_REQUIRED_METRICS: tuple[str, ...] = (
    "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
)


class NcuHarvesterError(Exception):
    """Raised when the NCU subprocess invocation fails before
    a CSV can be produced. Parser-side errors propagate as
    `NcuLabelParserError`. Carries the typed-reason string.

    Typed reasons:
      * `ncu_label_invocation_failed` — `ncu` binary absent
        or subprocess exits non-zero / out-of-range
        `launch_skip_count`.
      * `ncu_provenance_missing_field` — a ledger-required
        provenance field is empty / None when
        `validate_ncu_label_provenance()` is called.
      * `ncu_provenance_invalid_field_value` — a provenance
        field carries the wrong type (e.g. non-tuple
        `command_argv`).
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        self.typed_reason = typed_reason
        self.message = message


# Provenance fields the ledger write path requires. Every entry
# is non-empty by validator contract.
_NCU_PROVENANCE_REQUIRED_FIELDS: tuple[str, ...] = (
    "ncu_cli_path",
    "ncu_cli_sha256",
    "ncu_version_stdout",
    "driver_version",
    "gpu_device_name",
    "compute_capability",
    "clock_policy_observed",
    "command_argv",
    "raw_report_sha256",
)


@dataclass(frozen=True)
class NcuLabelProvenance:
    """Full provenance the ledger captures alongside each
    `NcuLabelRecord`. The harvester probes the host once per
    invocation; the result is paired with the parsed label
    record and either persisted to the ledger by the batch
    driver or attached to a downstream report's `[meta]`.

    Field semantics:
      * `ncu_cli_path`: absolute path of the resolved `ncu`
        binary (the `shutil.which("ncu")` answer at harvest
        time).
      * `ncu_cli_sha256`: lowercase hex sha256 of the
        resolved `ncu` binary's bytes. Drift here means a
        different NCU build emitted the labels.
      * `ncu_version_stdout`: the verbatim stdout of
        `ncu --version`. Mostly a free-form human-readable
        string; the operator can cross-reference against
        NVIDIA's NCU release notes.
      * `driver_version` / `gpu_device_name` /
        `compute_capability`: `nvidia-smi --query-gpu=...`
        outputs (driver_version, name, compute_cap). Drift
        here means the host changed.
      * `clock_policy_observed`: the clock-control flag the
        harvester argv pinned (always `"none"` today per
        CLAUDE.md Rule 2 / `--clock-control none`). A drift
        here means a future flag mutation slipped past the
        argv pin.
      * `command_argv`: the literal argv tuple `ncu` was
        invoked with. Lets the ledger reproduce the exact
        invocation.
      * `raw_report_sha256`: lowercase hex sha256 of the NCU
        CSV stdout the parser consumed. Drift here means a
        different CSV produced the label record.
    """

    ncu_cli_path: str
    ncu_cli_sha256: str
    ncu_version_stdout: str
    driver_version: str
    gpu_device_name: str
    compute_capability: str
    clock_policy_observed: str
    command_argv: tuple[str, ...]
    raw_report_sha256: str


def _build_invocation(
    binary: Path | str,
    *,
    launch_skip_count: int,
    binary_args: Sequence[str] = (),
    target_processes: str | None = None,
    kernel_name_filter: str | None = None,
) -> tuple[str, ...]:
    """Construct the full `ncu ...` argv that runs the target
    binary under the calibrated invocation form. Public for
    test inspection - callers can assert the exact argv shape
    without invoking NCU.

    The argv pins:
      * `--launch-skip N --launch-count 1` - measure exactly
        one launch after N warmup launches (N >= 1).
      * `--clock-control none` - Rule 2: no clock-policy
        mutation.
      * `--replay-mode kernel` - deterministic per-launch
        replay.
      * `--section MemoryWorkloadAnalysis` - the predictor
        target section.
      * `--metrics <pinned list>` - explicit selector for
        the five required read-only sector metric names.
        Augments the section's default metric set so the
        pinned columns are always present in the output.
      * `--csv` - CSV output format.

    Optional ``target_processes``: when set, inserts
    ``--target-processes <value>`` before ``--csv`` so NCU
    follows the target binary's child processes. Required
    for wrapping benchmark CLIs that fork a worker
    subprocess (the worker is where the GPU launches
    happen). Default ``None`` preserves byte-stable argv
    for every existing caller that does not wrap such a
    CLI; the calibration smoke binaries do not need it
    because they are standalone CUDA executables.

    Optional ``kernel_name_filter``: when set, inserts
    ``--kernel-name <value>`` before ``--csv`` so NCU
    only captures launches whose kernel name matches the
    given value. Essential for sol-execbench-wrapped
    pilots where the workload runs many init / RNG /
    cuBLAS kernels before the actual pilot kernel; without
    this filter the captured launch lands on a curand
    BFloat16 distribution kernel instead of the pilot
    kernel. Default ``None`` preserves byte-stable argv
    for every existing caller.
    """
    if launch_skip_count < 1:
        raise NcuHarvesterError(
            "ncu_label_invocation_failed",
            f"launch_skip_count must be >= 1; got "
            f"{launch_skip_count}",
        )
    metrics_selector = ",".join(NCU_REQUIRED_METRICS)
    argv: list[str] = [
        "ncu",
        "--launch-skip", str(launch_skip_count),
        "--launch-count", "1",
        "--clock-control", "none",
        "--replay-mode", "kernel",
        "--section", "MemoryWorkloadAnalysis",
        "--metrics", metrics_selector,
    ]
    if target_processes is not None:
        argv.extend(["--target-processes", target_processes])
    if kernel_name_filter is not None:
        argv.extend(["--kernel-name", kernel_name_filter])
    argv.append("--csv")
    argv.append(str(binary))
    argv.extend(binary_args)
    return tuple(argv)


def _run_checked_ncu(argv: tuple[str, ...]):
    """Real-subprocess runner for `harvest_ncu_label` and
    `label_smoke.run_label_smoke`.

    Validates that the `ncu` binary is on `$PATH`, invokes
    `subprocess.run` with a 600 s timeout, and normalizes
    every failure surface into a typed
    `NcuHarvesterError("ncu_label_invocation_failed", ...)`.
    Returns the `subprocess.CompletedProcess` on success.

    Public so the smoke harness (`label_smoke.py`) can use
    the same checked path when no runner injection is
    supplied. Callers that wrap this runner (e.g. for stdout
    capture) MUST preserve the typed-error contract.
    """
    if shutil.which("ncu") is None:
        raise NcuHarvesterError(
            "ncu_label_invocation_failed",
            "ncu binary not found on $PATH; calibration "
            "host must have NCU installed (sm_86 / sm_120 "
            "calibration-only carveout)",
        )
    try:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise NcuHarvesterError(
            "ncu_label_invocation_failed",
            f"ncu subprocess failed: {exc}",
        ) from exc


def _probe_ncu_cli_path() -> str:
    """Resolve `ncu` on $PATH to an absolute filesystem path.

    Raises `ncu_label_invocation_failed` when the binary is not
    found. Mirrors the existing `_run_checked_ncu` precheck so
    the provenance probe and the invocation share one boundary.
    """
    resolved = shutil.which("ncu")
    if resolved is None:
        raise NcuHarvesterError(
            "ncu_label_invocation_failed",
            "ncu binary not found on $PATH; calibration "
            "host must have NCU installed (sm_86 / sm_120 "
            "calibration-only carveout)",
        )
    return str(Path(resolved).resolve())


def _probe_ncu_cli_sha256(
    ncu_cli_path: str,
    *,
    _reader=None,
) -> str:
    """Compute lowercase hex sha256 of the `ncu` binary's bytes.

    `_reader` is an injection seam: when provided, the callable
    receives the path and returns the bytes. The default reads
    from disk via `Path.read_bytes()`. Tests inject a synthetic
    reader so the provenance probe runs without touching the
    real NCU binary.
    """
    reader = _reader if _reader is not None else (
        lambda path: Path(path).read_bytes()
    )
    data = reader(ncu_cli_path)
    return hashlib.sha256(data).hexdigest()


def _probe_ncu_version_stdout(
    ncu_cli_path: str,
    *,
    _runner=None,
) -> str:
    """Run `<ncu_cli_path> --version` and return its stdout
    verbatim.

    `_runner` is an injection seam: when provided, the callable
    receives `(ncu_cli_path,)` and returns a
    `subprocess.CompletedProcess`-like object exposing
    `returncode` and `stdout`. The default uses
    `subprocess.run`.
    """

    def _default_runner(path: str):
        return subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

    runner = _runner if _runner is not None else _default_runner
    try:
        completed = runner(ncu_cli_path)
    except (OSError, subprocess.SubprocessError) as exc:
        raise NcuHarvesterError(
            "ncu_label_invocation_failed",
            f"`ncu --version` failed: {exc}",
        ) from exc
    if completed.returncode != 0:
        raise NcuHarvesterError(
            "ncu_label_invocation_failed",
            f"`ncu --version` exited with code "
            f"{completed.returncode}; stderr: "
            f"{getattr(completed, 'stderr', '')!r}",
        )
    return (completed.stdout or "").strip()


def _probe_nvidia_smi_device_info(
    *,
    _runner=None,
) -> tuple[str, str, str]:
    """Run `nvidia-smi --query-gpu=driver_version,name,compute_cap
    --format=csv,noheader,nounits` and return
    `(driver_version, gpu_device_name, compute_capability)`.

    `_runner` is an injection seam: when provided, the callable
    is invoked with no arguments and returns a
    `subprocess.CompletedProcess`-like object exposing
    `returncode` and `stdout`. The default uses
    `subprocess.run`.
    """

    def _default_runner():
        return subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,name,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

    runner = _runner if _runner is not None else _default_runner
    try:
        completed = runner()
    except (OSError, subprocess.SubprocessError) as exc:
        raise NcuHarvesterError(
            "ncu_label_invocation_failed",
            f"`nvidia-smi --query-gpu=...` failed: {exc}",
        ) from exc
    if completed.returncode != 0:
        raise NcuHarvesterError(
            "ncu_label_invocation_failed",
            f"`nvidia-smi --query-gpu=...` exited with code "
            f"{completed.returncode}; stderr: "
            f"{getattr(completed, 'stderr', '')!r}",
        )
    text = (completed.stdout or "").strip()
    if not text:
        raise NcuHarvesterError(
            "ncu_provenance_missing_field",
            "nvidia-smi returned empty device-info stdout; "
            "cannot capture driver_version / gpu_device_name "
            "/ compute_capability provenance",
        )
    first_row = text.splitlines()[0]
    parts = [p.strip() for p in first_row.split(",")]
    if len(parts) < 3:
        raise NcuHarvesterError(
            "ncu_provenance_invalid_field_value",
            f"nvidia-smi device-info CSV row has "
            f"{len(parts)} fields; expected at least 3 "
            f"(driver_version, name, compute_cap). Raw: "
            f"{first_row!r}",
        )
    driver_version, gpu_device_name, compute_capability = parts[:3]
    return driver_version, gpu_device_name, compute_capability


# `_CLOCK_POLICY_FROM_ARGV` is the answer the harvester records
# for `clock_policy_observed`. The argv pin (`--clock-control
# none`) is fixed by the calibration invocation form (CLAUDE.md
# Rule 2: no clock-policy mutation). If a future operator
# changes the argv to include a different `--clock-control`
# flag, this constant should be updated to track the actual
# pin. The provenance field exists so future drift surfaces in
# the ledger.
_CLOCK_POLICY_FROM_ARGV: str = "none"


def collect_ncu_label_provenance(
    argv: Sequence[str],
    raw_report: str,
    *,
    ncu_cli_path: str | None = None,
    ncu_cli_sha256_reader=None,
    ncu_version_runner=None,
    nvidia_smi_runner=None,
) -> NcuLabelProvenance:
    """Capture the full provenance set the ledger requires.

    `argv` is the literal invocation the harvester passed to
    `subprocess.run`. `raw_report` is the verbatim NCU CSV
    stdout the parser consumed.

    Probe seams let tests construct synthetic provenance
    without touching the real subprocess surface. Each seam
    has a sensible default (resolve via `shutil.which`, hash
    via `Path.read_bytes`, run `subprocess.run`).
    """
    if ncu_cli_path is None:
        ncu_cli_path = _probe_ncu_cli_path()
    ncu_cli_sha256 = _probe_ncu_cli_sha256(
        ncu_cli_path, _reader=ncu_cli_sha256_reader,
    )
    ncu_version_stdout = _probe_ncu_version_stdout(
        ncu_cli_path, _runner=ncu_version_runner,
    )
    driver_version, gpu_device_name, compute_capability = (
        _probe_nvidia_smi_device_info(_runner=nvidia_smi_runner)
    )
    raw_report_sha256 = hashlib.sha256(
        raw_report.encode("utf-8")
    ).hexdigest()
    return NcuLabelProvenance(
        ncu_cli_path=ncu_cli_path,
        ncu_cli_sha256=ncu_cli_sha256,
        ncu_version_stdout=ncu_version_stdout,
        driver_version=driver_version,
        gpu_device_name=gpu_device_name,
        compute_capability=compute_capability,
        clock_policy_observed=_CLOCK_POLICY_FROM_ARGV,
        command_argv=tuple(argv),
        raw_report_sha256=raw_report_sha256,
    )


def validate_ncu_label_provenance(
    provenance: Mapping[str, Any] | NcuLabelProvenance,
) -> NcuLabelProvenance:
    """Validate every ledger-required provenance field is
    present and non-empty. Raises
    `NcuHarvesterError("ncu_provenance_missing_field", ...)`
    for missing / empty fields and
    `NcuHarvesterError("ncu_provenance_invalid_field_value", ...)`
    for type mismatches.

    Accepts both a `NcuLabelProvenance` dataclass and a plain
    dict (the ledger-write path goes through dict → validate →
    dataclass). On a `NcuLabelProvenance` input the validator
    confirms every field is non-empty; on a dict input it
    also reconstructs the dataclass.
    """
    if isinstance(provenance, NcuLabelProvenance):
        as_dict: dict[str, Any] = {
            field: getattr(provenance, field)
            for field in _NCU_PROVENANCE_REQUIRED_FIELDS
        }
    else:
        as_dict = dict(provenance)

    for field in _NCU_PROVENANCE_REQUIRED_FIELDS:
        if field not in as_dict:
            raise NcuHarvesterError(
                "ncu_provenance_missing_field",
                f"ncu label provenance missing required "
                f"field {field!r}",
            )
        value = as_dict[field]
        if field == "command_argv":
            if not isinstance(value, (tuple, list)) or len(value) == 0:
                raise NcuHarvesterError(
                    "ncu_provenance_invalid_field_value",
                    f"ncu label provenance command_argv must "
                    f"be a non-empty tuple/list of argv "
                    f"strings; got {value!r} "
                    f"({type(value).__name__})",
                )
            for index, arg in enumerate(value):
                if not isinstance(arg, str):
                    raise NcuHarvesterError(
                        "ncu_provenance_invalid_field_value",
                        f"ncu label provenance "
                        f"command_argv[{index}] must be a "
                        f"string; got {arg!r} "
                        f"({type(arg).__name__})",
                    )
            continue
        if not isinstance(value, str):
            raise NcuHarvesterError(
                "ncu_provenance_invalid_field_value",
                f"ncu label provenance {field!r} must be a "
                f"string; got {value!r} "
                f"({type(value).__name__})",
            )
        if not value:
            raise NcuHarvesterError(
                "ncu_provenance_missing_field",
                f"ncu label provenance {field!r} is empty; "
                f"every provenance field must carry a "
                f"non-empty value at ledger write time",
            )
    return NcuLabelProvenance(
        ncu_cli_path=as_dict["ncu_cli_path"],
        ncu_cli_sha256=as_dict["ncu_cli_sha256"],
        ncu_version_stdout=as_dict["ncu_version_stdout"],
        driver_version=as_dict["driver_version"],
        gpu_device_name=as_dict["gpu_device_name"],
        compute_capability=as_dict["compute_capability"],
        clock_policy_observed=as_dict["clock_policy_observed"],
        command_argv=tuple(as_dict["command_argv"]),
        raw_report_sha256=as_dict["raw_report_sha256"],
    )


def harvest_ncu_label_with_provenance(
    binary: Path | str,
    *,
    launch_skip_count: int,
    binary_args: Sequence[str] = (),
    ncu_metric_version: str = "unknown",
    binary_sha256: str = "",
    warmup_protocol: str = "unknown",
    target_processes: str | None = None,
    kernel_name_filter: str | None = None,
    _runner=None,
    ncu_cli_path: str | None = None,
    ncu_cli_sha256_reader=None,
    ncu_version_runner=None,
    nvidia_smi_runner=None,
) -> tuple[NcuLabelRecord, NcuLabelProvenance]:
    """Invoke NCU against `binary` and return both the parsed
    label record AND the full provenance the ledger requires.

    Functionally `harvest_ncu_label()` + provenance capture in a
    single call so the harvester probes the host ONCE per
    invocation. Probe injection seams (`ncu_cli_path`,
    `ncu_cli_sha256_reader`, `ncu_version_runner`,
    `nvidia_smi_runner`) let tests construct synthetic
    provenance without spawning real subprocesses.

    ``target_processes`` is forwarded to ``_build_invocation``
    so NCU follows child processes when wrapping a subprocess-
    launching benchmark CLI. Default ``None`` preserves
    existing per-binary invocation shape.

    ``kernel_name_filter`` is forwarded to
    ``_build_invocation`` so NCU only captures launches
    whose kernel name contains the given substring.
    Required for benchmark CLIs that interleave the
    target kernel with framework / RNG / library kernels;
    without it the captured launch can land on the wrong
    kernel.
    """
    argv = _build_invocation(
        binary,
        launch_skip_count=launch_skip_count,
        binary_args=binary_args,
        target_processes=target_processes,
        kernel_name_filter=kernel_name_filter,
    )
    runner = _runner if _runner is not None else _run_checked_ncu
    completed = runner(argv)
    returncode = completed.returncode
    stdout = completed.stdout
    stderr = completed.stderr

    if returncode != 0:
        raise NcuHarvesterError(
            "ncu_label_invocation_failed",
            f"ncu exited with code {returncode}; stderr: "
            f"{stderr!r}",
        )
    record = parse_ncu_csv(
        stdout,
        ncu_metric_version=ncu_metric_version,
        binary_sha256=binary_sha256,
        launch_skip_count=launch_skip_count,
        warmup_protocol=warmup_protocol,
    )
    provenance = collect_ncu_label_provenance(
        argv,
        stdout,
        ncu_cli_path=ncu_cli_path,
        ncu_cli_sha256_reader=ncu_cli_sha256_reader,
        ncu_version_runner=ncu_version_runner,
        nvidia_smi_runner=nvidia_smi_runner,
    )
    return record, provenance


def harvest_ncu_label(
    binary: Path | str,
    *,
    launch_skip_count: int,
    binary_args: Sequence[str] = (),
    ncu_metric_version: str = "unknown",
    binary_sha256: str = "",
    warmup_protocol: str = "unknown",
    target_processes: str | None = None,
    kernel_name_filter: str | None = None,
    _runner=None,
) -> NcuLabelRecord:
    """Invoke NCU against `binary` and return the parsed
    label record.

    `launch_skip_count` is the value of NCU's `--launch-skip
    N` argument; must be >= 1. The provenance fields
    (`ncu_metric_version`, `binary_sha256`,
    `warmup_protocol`) are passed through to the returned
    record so the result is self-describing.

    `_runner` is an injection seam: when supplied, the
    callable receives the constructed argv tuple and returns
    a `subprocess.CompletedProcess`-like object exposing
    `returncode`, `stdout`, and `stderr` (all strings). The
    default `None` uses `subprocess.run`. Tests inject a
    synthetic runner so they exercise the harvester without
    spawning a real NCU process.

    Typed reasons raised:
      `ncu_label_invocation_failed`: NCU binary missing,
        subprocess exits non-zero, or `launch_skip_count`
        is out of range.
      Parser-side typed reasons propagate through.

    ``target_processes``: when set, forwards to
    ``_build_invocation`` so NCU follows the target's
    child processes. Default ``None`` preserves the
    existing per-binary command shape.

    ``kernel_name_filter``: when set, forwards to
    ``_build_invocation`` so NCU only captures launches
    whose kernel name contains the given substring.
    """
    argv = _build_invocation(
        binary,
        launch_skip_count=launch_skip_count,
        binary_args=binary_args,
        target_processes=target_processes,
        kernel_name_filter=kernel_name_filter,
    )
    runner = _runner if _runner is not None else _run_checked_ncu
    completed = runner(argv)
    returncode = completed.returncode
    stdout = completed.stdout
    stderr = completed.stderr

    if returncode != 0:
        raise NcuHarvesterError(
            "ncu_label_invocation_failed",
            f"ncu exited with code {returncode}; stderr: "
            f"{stderr!r}",
        )
    return parse_ncu_csv(
        stdout,
        ncu_metric_version=ncu_metric_version,
        binary_sha256=binary_sha256,
        launch_skip_count=launch_skip_count,
        warmup_protocol=warmup_protocol,
    )


__all__ = [
    "NCU_HARVESTER_TYPED_REASONS",
    "NCU_INVOCATION_TEMPLATE",
    "NCU_PROVENANCE_TYPED_REASONS",
    "NCU_REQUIRED_METRICS",
    "NcuHarvesterError",
    "NcuLabelProvenance",
    "_build_invocation",
    "_run_checked_ncu",
    "collect_ncu_label_provenance",
    "harvest_ncu_label",
    "harvest_ncu_label_with_provenance",
    "validate_ncu_label_provenance",
]
