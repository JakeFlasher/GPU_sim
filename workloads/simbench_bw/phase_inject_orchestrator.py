"""A/B overhead orchestration + per-launch phase-ledger emission for
the non-root measurement NVBit phase-injection contract.

This module wires the GPU-side phase-injection tracer extension
(consolidated under `tools/nvbit_tracer/tracer_tool/`) to the
host-side aggregator + emitter (`phase_cycle_ledger_aggregator.py`)
and the consumer's admission rules (`phase_cycle_ledger_consumer.py`).
The producer-side test surface uses synthetic subprocess outputs;
the production-side surface drives `run_suite.py::handle_trace` with
the `--ab-overhead` flag.

Public API:

  * `run_ab_overhead_pair(...)` — invokes the kernel binary twice
    (no-injection vs with-injection NVBit run); returns an
    `ab_overhead_record` matching the schema's
    `ab_overhead_record` object shape.
  * `emit_per_launch_phase_ledgers(...)` — scans a traces folder
    for `phase_records_<repeat>_<launch>.bin` files (emitted by
    the tracer extension's per-launch flush path), decodes each
    via the aggregator's struct format, builds a schema-valid
    `phase_cycle_ledger_<repeat>_<launch>.json` sidecar, runs the
    consumer's `admit_phase_cycle_ledger`, and reports the per-
    ledger admission verdict.

Both helpers are host-independent in the sense that the tracer
binary + GPU launch behavior is encapsulated behind subprocess
calls; tests can monkeypatch the subprocess to feed synthetic
phase-record bytes without a real Blackwell run.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import phase_cycle_ledger_aggregator as _agg  # type: ignore[import-not-found]
import phase_cycle_ledger_consumer as _consumer  # type: ignore[import-not-found]

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from tools.nvbit_tracer.runtime_env import (  # noqa: E402
    TracerRuntimeError,
    tracer_runtime_env,
)


PHASE_RECORDS_FILENAME_RE = re.compile(
    r"^phase_records_(\d+)_(\d+)\.bin$"
)


@dataclass(frozen=True)
class AbOverheadInputs:
    """Inputs needed for a paired no-injection / with-injection
    overhead run."""

    executable: Path
    args: list[str]
    tracer_so_path: Path
    traces_folder: Path
    repeat_count: int = 3
    base_env: dict[str, str] | None = None


def _median_ns(samples: list[int]) -> int:
    s = sorted(samples)
    return int(s[len(s) // 2])


_DEFAULT_TIMEOUT_PER_RUN_S = 600


def run_ab_overhead_pair(
    inputs: AbOverheadInputs,
    *,
    admission_threshold_pct: float,
    subprocess_runner=subprocess.run,
    timer_ns=time.perf_counter_ns,
    timeout_per_run_s: int | None = _DEFAULT_TIMEOUT_PER_RUN_S,
) -> dict[str, Any]:
    """Run the executable twice (no-injection NVBit + with-injection
    NVBit) `repeat_count` times each. Compute the median wall-clock
    times for each side and the overhead percentage. Returns an
    `ab_overhead_record` dict matching the schema's required fields.

    The `subprocess_runner` and `timer_ns` parameters allow tests to
    monkeypatch behavior without a real GPU run; production callers
    use the defaults.

    Each subprocess invocation is bounded by `timeout_per_run_s` (default
    600 s) so the A/B orchestration cannot hang indefinitely. A timeout
    on either side surfaces as `PhaseInjectAbRunTimeout` with the typed
    reason `phase_inject_ab_run_timeout` (distinct from the return-code-
    driven `PhaseInjectAbRunFailed` which carries `phase_inject_ab_run_failed`).
    Both typed exceptions are caught by the `run_suite.handle_trace`
    boundary and translated to `EXIT_BUILD_FAILED` with a stderr
    diagnostic. Pass `timeout_per_run_s=None` to opt out (production
    runs should keep the bound; the opt-out path exists for tests and
    for explicit-debug operator runs)."""
    try:
        base_env = tracer_runtime_env(
            inputs.tracer_so_path, base_env=inputs.base_env
        )
    except TracerRuntimeError as exc:
        raise PhaseInjectRuntimeUnavailable(str(exc)) from exc
    env_no_inject = dict(base_env)
    env_no_inject["CUDA_INJECTION64_PATH"] = str(inputs.tracer_so_path)
    env_no_inject["TRACES_FOLDER"] = str(inputs.traces_folder)
    # The tracer ignores TRACES_FOLDER unless USER_DEFINED_FOLDERS=1
    # is also set (see tracer_tool.cu's R51 redirect at the
    # `user_defined_folders == 1` gate). Without this, the tracer
    # writes phase_records_<repeat>_<launch>.bin to its hardcoded
    # default `<cwd>/traces/` path, where emit_per_launch_phase_ledgers
    # cannot find them and emits zero ledger sidecars even though
    # phase records ARE produced. handle_trace's regular-trace env
    # builder sets this for the non-phase-inject path; the A/B pair
    # must set it on both sides to keep the producer output
    # co-located with traces_folder.
    env_no_inject["USER_DEFINED_FOLDERS"] = "1"
    # The tracer's env-gated activation switch is OFF for the
    # no-inject side (so the existing instruction-trace pipeline
    # runs unchanged) and ON for the with-inject side.
    env_no_inject.pop("SIMBENCH_NVBIT_PHASE_INJECT", None)

    env_with_inject = dict(env_no_inject)
    env_with_inject["SIMBENCH_NVBIT_PHASE_INJECT"] = "1"

    def _run_side(side: str, env: dict[str, str]) -> tuple[Any, int]:
        t_start = timer_ns()
        runner_kwargs: dict[str, Any] = {
            "capture_output": True,
            "text": True,
            "env": env,
        }
        if timeout_per_run_s is not None:
            runner_kwargs["timeout"] = timeout_per_run_s
        try:
            proc = subprocess_runner(
                [str(inputs.executable), *inputs.args],
                **runner_kwargs,
            )
        except subprocess.TimeoutExpired as e:
            raise PhaseInjectAbRunTimeout(
                side=side,
                timeout_s=timeout_per_run_s or 0,
                stderr=(e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")),
            )
        t_end = timer_ns()
        if proc.returncode != 0:
            raise PhaseInjectAbRunFailed(
                side=side,
                returncode=proc.returncode,
                stderr=proc.stderr,
            )
        return proc, t_end - t_start

    no_inject_samples_ns: list[int] = []
    with_inject_samples_ns: list[int] = []
    for _ in range(inputs.repeat_count):
        _, dt = _run_side("no_inject", env_no_inject)
        no_inject_samples_ns.append(dt)
        _, dt = _run_side("with_inject", env_with_inject)
        with_inject_samples_ns.append(dt)

    median_no = _median_ns(no_inject_samples_ns)
    median_with = _median_ns(with_inject_samples_ns)
    if median_no <= 0:
        raise PhaseInjectAbRunFailed(
            side="no_inject",
            returncode=0,
            stderr=(
                f"median_no_inject_wall_ns={median_no}: zero/negative "
                "median wall time, cannot compute overhead percentage."
            ),
        )
    overhead_pct = max(
        0.0, ((median_with - median_no) / median_no) * 100.0
    )
    return {
        "t_wall_no_inject_ns": int(median_no),
        "t_wall_with_inject_ns": int(median_with),
        "overhead_pct": float(overhead_pct),
        "admission_threshold_pct": float(admission_threshold_pct),
        "admitted": overhead_pct <= admission_threshold_pct,
    }


class PhaseInjectAbRunFailed(Exception):
    """Raised when one side of the A/B overhead pair returned
    non-zero. The `side` payload names which side failed so the
    caller can surface the correct typed diagnostic."""

    TOKEN = "phase_inject_ab_run_failed"

    def __init__(self, *, side: str, returncode: int, stderr: str):
        super().__init__(
            f"{self.TOKEN}: side={side!r} rc={returncode} "
            f"stderr={stderr!r}; the A/B overhead pair must complete "
            "cleanly on both sides."
        )
        self.side = side
        self.returncode = returncode
        self.stderr = stderr

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "side": self.side,
            "returncode": self.returncode,
            "stderr": self.stderr,
        }


class PhaseInjectRuntimeUnavailable(Exception):
    """The tracer cannot be dependency-checked before the A/B pair."""

    TOKEN = "phase_inject_runtime_unavailable"

    def __init__(self, detail: str):
        super().__init__(f"{self.TOKEN}: {detail}")
        self.detail = detail


class PhaseInjectAbRunTimeout(Exception):
    """Raised when one side of the A/B overhead pair exceeded the per-run
    timeout. Distinguishable from `PhaseInjectAbRunFailed` because the
    failure mode is wall-clock-bounded, not return-code-driven. Closes
    the R1-review-recorded blocker that an unbounded subprocess.run let
    a transient pathological GPU state burn 14 h+ of CPU."""

    TOKEN = "phase_inject_ab_run_timeout"

    def __init__(self, *, side: str, timeout_s: int, stderr: str):
        super().__init__(
            f"{self.TOKEN}: side={side!r} timeout_s={timeout_s} "
            f"stderr={stderr!r}; the A/B overhead pair's per-run "
            "subprocess exceeded its bounded timeout. Investigate "
            "tracer phase-injection orchestration before re-running."
        )
        self.side = side
        self.timeout_s = timeout_s
        self.stderr = stderr

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "side": self.side,
            "timeout_s": self.timeout_s,
            "stderr": self.stderr,
        }


class PhaseLedgerEvidenceMissing(Exception):
    """The phase-enabled run emitted no decodable per-launch evidence."""

    TOKEN = "phase_inject_ledger_evidence_missing"

    def __init__(self, traces_folder: Path):
        super().__init__(
            f"{self.TOKEN}: no phase_records_<repeat>_<launch>.bin file under "
            f"{traces_folder} produced a matching nonempty launch ledger"
        )
        self.traces_folder = traces_folder


def emit_per_launch_phase_ledgers(
    traces_folder: Path,
    ledger_output_dir: Path,
    *,
    slug: str,
    shape_point: str,
    kernel_signature_sha256: str,
    binary_sha256: str,
    nvbit_tracer_build_hash: str,
    claim_scope: str,
    ab_overhead_record: dict[str, Any],
    dual_run_manifest_ref: str,
    admission: _consumer.PhaseInjectionAdmission | None = None,
    admission_threshold_pct: float | None = None,
) -> dict[str, Any]:
    """Scan `traces_folder` for `phase_records_<repeat>_<launch>.bin`
    files, decode each, build a schema-valid phase ledger, schema-
    validate, write to `ledger_output_dir`, and admit through the
    consumer's `admit_phase_cycle_ledger`. Returns a summary dict
    naming all emitted ledgers + their admission verdicts."""
    ledger_output_dir.mkdir(parents=True, exist_ok=True)
    emitted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for bin_file in sorted(traces_folder.glob("phase_records_*.bin")):
        m = PHASE_RECORDS_FILENAME_RE.match(bin_file.name)
        if not m:
            continue
        repeat_index = int(m.group(1))
        launch_index = int(m.group(2))
        records = _agg.decode_phase_clock_records(bin_file.read_bytes())
        # Filter records to only those matching this launch_index;
        # mixed-launch files are a producer-side defect that the
        # aggregator's typed diagnostic catches at build time, but
        # we also defensively filter here so a single-launch file
        # is the producer's per-launch contract.
        per_launch_records = [
            r for r in records if r.launch_index == launch_index
        ]
        if not per_launch_records:
            continue
        ledger = _agg.build_phase_cycle_ledger(
            per_launch_records,
            slug=slug,
            shape_point=shape_point,
            repeat_index=repeat_index,
            launch_index=launch_index,
            kernel_signature_sha256=kernel_signature_sha256,
            binary_sha256=binary_sha256,
            nvbit_tracer_build_hash=nvbit_tracer_build_hash,
            claim_scope=claim_scope,
            ab_overhead_record=ab_overhead_record,
            dual_run_manifest_ref=dual_run_manifest_ref,
        )
        output_path = (
            ledger_output_dir
            / f"phase_cycle_ledger_{repeat_index}_{launch_index}.json"
        )
        _agg.emit_phase_cycle_ledger(ledger, output_path)
        try:
            _consumer.admit_phase_cycle_ledger(
                ledger,
                admission=admission,
                admission_threshold_pct=admission_threshold_pct,
            )
            emitted.append(
                {
                    "repeat_index": repeat_index,
                    "launch_index": launch_index,
                    "output_path": str(output_path),
                    "admitted": True,
                }
            )
        except _consumer.NvbitPhaseInjectionOverheadExceeded as e:
            rejected.append(
                {
                    "repeat_index": repeat_index,
                    "launch_index": launch_index,
                    "output_path": str(output_path),
                    "admitted": False,
                    "reason": e.to_json(),
                }
            )
        except _consumer.PhaseBoundaryMappingUnresolved as e:
            rejected.append(
                {
                    "repeat_index": repeat_index,
                    "launch_index": launch_index,
                    "output_path": str(output_path),
                    "admitted": False,
                    "reason": e.to_json(),
                }
            )
    if not emitted and not rejected:
        raise PhaseLedgerEvidenceMissing(traces_folder)
    return {
        "emitted_admitted": emitted,
        "emitted_rejected": rejected,
        "n_admitted": len(emitted),
        "n_rejected": len(rejected),
    }
