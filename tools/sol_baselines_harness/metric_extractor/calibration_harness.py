"""Synthetic Tier-1 calibration harness.

Runs the six calibration binaries built under
``nvbit_tracer/metric_extractor/tests/calibration/`` under the
metric_extractor injection, parses the emitted JSON-L through the
existing strict parser, and compares the per-launch metric values
against the CPU-side expectations declared in
``expected_metrics.json``.

The harness produces a :class:`CalibrationReport` with per-kernel,
per-metric `(observed, expected, delta, verdict)` rows. Verdicts are
``pass`` / ``fail`` / ``skip``; a metric outside its declared tolerance
is a ``fail`` verdict but does NOT raise — that decision is the
Tier-1 tolerance gate, which consumes this report.

The harness DOES raise :class:`CalibrationError` on structural failures
that would invalidate the comparison entirely: a missing binary, an
unparseable JSON-L (the strict parser surfaced an error), the JSON-L
contained zero records, or no expected_metrics entry covers the
kernel the producer actually emitted (so the harness cannot decide
what value to compare against).

The expected_metrics.json schema is intentionally small; the four
check kinds cover every positive-test pattern the corpus exercises:

- ``float_within``: ``abs(observed - value) <= tolerance``. Used for
  sector-efficiency ratios.
- ``exact_int``: ``observed == value``, both must be non-bool integers.
  Used for bank_conflicts and the derived distinct-sector count.
- ``min_int``: ``observed >= value``, both integers. Used for the
  16-active branch bucket which is asserted to have at least 1
  increment.
- ``null``: ``observed is None``. Used for sector_efficiency_global_*
  on shared-memory-only kernels that emit no global LDG/STG.

The ``sector_load_distinct_sectors`` metric is **derived** by the
harness rather than read directly from the JSON-L: the producer emits
the load ratio, not the raw distinct-sector count, but useful_bytes is
trivially computable for the single-warp 32x16B kernel
(``32 lanes * 16 B = 512 B``), so the harness reconstructs
``distinct_sectors = round(useful / ratio) / 32`` from the ratio. This
keeps the vector-load assertion ("expected sector count = 16")
runnable without expanding the wire format.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .launches_parser import LaunchRecord, ParseError, parse_launches_jsonl


class CalibrationError(RuntimeError):
    """Raised when the calibration run cannot decide pass/fail for a
    structural reason (missing binary, unparseable JSON-L, no records,
    no expected_metrics entry for the emitted kernel). A metric
    outside tolerance is NOT a CalibrationError — that is a regular
    ``fail`` verdict on the report and the tolerance gate (T25)
    decides what to do with it."""


@dataclass
class MetricVerdict:
    """One per `(kernel, metric)` row in the report."""

    metric: str
    kind: str
    expected: Any
    observed: Any
    delta: Optional[float]
    verdict: str  # "pass" | "fail" | "skip"
    reason: Optional[str] = None


@dataclass
class KernelResult:
    """Aggregated per-kernel report."""

    binary: str
    kernel_name_demangled: str
    ac4_test: str
    verdicts: List[MetricVerdict] = field(default_factory=list)
    record: Optional[LaunchRecord] = None

    @property
    def all_pass(self) -> bool:
        return all(v.verdict == "pass" for v in self.verdicts)


@dataclass
class CalibrationReport:
    """End-to-end report covering every calibration binary the harness
    ran. The ``kernels`` field preserves the binary-name order from the
    expected_metrics table so a fixed-order report is reproducible.
    """

    corpus_version: str
    kernels: List[KernelResult] = field(default_factory=list)

    @property
    def all_pass(self) -> bool:
        return all(k.all_pass for k in self.kernels)

    def per_metric_error_stats(self) -> Dict[str, Dict[str, float]]:
        """Return per-metric aggregate stats (max delta, count of pass /
        fail) so T25's tolerance gate can decide which confidence
        labels to promote. Only metrics with a numeric ``delta`` are
        included; null / exact_int / min_int checks are aggregated
        only by pass/fail count.
        """
        stats: Dict[str, Dict[str, float]] = {}
        for kresult in self.kernels:
            for v in kresult.verdicts:
                slot = stats.setdefault(
                    v.metric,
                    {"pass_count": 0.0, "fail_count": 0.0, "max_delta": 0.0},
                )
                if v.verdict == "pass":
                    slot["pass_count"] += 1.0
                elif v.verdict == "fail":
                    slot["fail_count"] += 1.0
                if v.delta is not None and v.delta > slot["max_delta"]:
                    slot["max_delta"] = v.delta
        return stats


def _load_expected(expected_json: Path) -> Dict[str, Any]:
    """Read and minimally validate the expectation table."""
    if not expected_json.exists():
        raise CalibrationError(
            f"expected_metrics.json not found at {expected_json}"
        )
    with expected_json.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or "kernels" not in data:
        raise CalibrationError(
            f"expected_metrics.json malformed: missing top-level "
            f"'kernels' mapping (got {type(data).__name__})"
        )
    if not isinstance(data["kernels"], dict):
        raise CalibrationError(
            f"expected_metrics.json malformed: 'kernels' must be an "
            f"object, got {type(data['kernels']).__name__}"
        )
    return data


def _run_binary_and_parse(
    binary: Path,
    metric_extractor_so: Path,
    user_module_regex: str = ".*",
) -> LaunchRecord:
    """Spawn a single calibration binary under metric_extractor
    injection and return the parsed JSON-L record. Raises
    CalibrationError on any structural failure.
    """
    if not binary.exists() or not os.access(binary, os.X_OK):
        raise CalibrationError(f"calibration binary missing: {binary}")
    if not metric_extractor_so.exists():
        raise CalibrationError(
            f"metric_extractor.so missing: {metric_extractor_so}"
        )
    with tempfile.TemporaryDirectory() as tmp:
        jsonl_path = Path(tmp) / f"{binary.name}.jsonl"
        env = {
            **os.environ,
            "CUDA_INJECTION64_PATH": str(metric_extractor_so),
            "METRIC_EXTRACTOR_USER_MODULE_REGEX": user_module_regex,
            "METRIC_EXTRACTOR_LAUNCHES_PATH": str(jsonl_path),
            # Mute the nvbit banner so the harness output stays tight.
            "NOBANNER": "1",
        }
        # cwd is the calibration directory so binaries that emit
        # auxiliary side-files (filter_stats.json) write next to the
        # binary, not the project root.
        proc = subprocess.run(
            [str(binary)],
            env=env,
            cwd=str(binary.parent),
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if proc.returncode != 0:
            raise CalibrationError(
                f"calibration binary {binary.name} exited with "
                f"{proc.returncode}: stdout={proc.stdout!r} "
                f"stderr={proc.stderr!r}"
            )
        records, errors = parse_launches_jsonl(jsonl_path)
        if errors:
            err_msgs = "; ".join(f"line {e.line_number}: {e.reason}"
                                 for e in errors)
            raise CalibrationError(
                f"strict parser rejected JSON-L emitted by "
                f"{binary.name}: {err_msgs}"
            )
        if not records:
            raise CalibrationError(
                f"calibration binary {binary.name} emitted zero JSON-L "
                f"records (no kernel launches observed by the tool)"
            )
        if len(records) > 1:
            raise CalibrationError(
                f"calibration binary {binary.name} emitted "
                f"{len(records)} JSON-L records; expected exactly one "
                f"(single-warp single-launch corpus invariant)"
            )
        return records[0]


def _extract_metric(record: LaunchRecord, metric: str) -> Any:
    """Read a documented metric from the parsed record. The metric
    names mirror the wire-format field names with one synthesised
    exception: ``branch_active_lanes_hist[N]`` indexes into the array,
    and ``sector_load_distinct_sectors`` is a derived count from the
    load ratio + the per-launch useful_bytes.
    """
    raw = record.raw
    if metric == "sector_load_distinct_sectors":
        ratio = raw.get("sector_efficiency_global_load")
        if ratio is None or ratio == 0:
            return None
        # The corpus is single-warp single-launch by construction. The
        # derivation relies on useful_bytes = 32 lanes * 16 B = 512 for
        # the LDG.E.128 kernel; this is the only kernel that asks for
        # this metric (the table only declares the check on
        # calib_vector_32x16B), so the constant is fine. A future
        # version that needs other widths can extend this.
        useful_bytes = 32 * 16
        transferred_bytes = round(useful_bytes / ratio)
        return transferred_bytes // 32
    if metric.startswith("branch_active_lanes_hist["):
        try:
            idx = int(metric[len("branch_active_lanes_hist["):-1])
        except ValueError as exc:
            raise CalibrationError(
                f"malformed metric name in expected_metrics.json: "
                f"{metric!r}"
            ) from exc
        hist = raw.get("branch_active_lanes_hist")
        if not isinstance(hist, list) or idx < 0 or idx >= len(hist):
            raise CalibrationError(
                f"branch_active_lanes_hist[{idx}] out of range or "
                f"missing in emitted record"
            )
        return hist[idx]
    if metric not in raw:
        raise CalibrationError(
            f"expected metric {metric!r} is not in the emitted "
            f"record's top-level fields"
        )
    return raw[metric]


def _evaluate_check(check: Dict[str, Any],
                    observed: Any) -> MetricVerdict:
    """Apply one expected_metrics check against an observed value.
    Always returns a MetricVerdict (never raises) — outside-tolerance
    is a ``fail`` verdict, not a structural failure.
    """
    kind = check.get("kind")
    metric = check.get("metric", "<unknown>")
    expected = check.get("value")
    tolerance = check.get("tolerance")
    if kind == "null":
        ok = (observed is None)
        return MetricVerdict(
            metric=metric, kind=kind, expected=None,
            observed=observed, delta=None,
            verdict="pass" if ok else "fail",
            reason=None if ok else f"expected null, observed {observed!r}",
        )
    if kind == "exact_int":
        if (isinstance(observed, bool) or
                not isinstance(observed, int)):
            return MetricVerdict(
                metric=metric, kind=kind, expected=expected,
                observed=observed, delta=None, verdict="fail",
                reason=(f"observed must be a non-bool integer, got "
                        f"{type(observed).__name__}"),
            )
        delta = float(abs(observed - expected))
        ok = (observed == expected)
        return MetricVerdict(
            metric=metric, kind=kind, expected=expected,
            observed=observed, delta=delta,
            verdict="pass" if ok else "fail",
            reason=None if ok else (
                f"expected exact {expected}, observed {observed}, "
                f"delta {delta}"
            ),
        )
    if kind == "min_int":
        if (isinstance(observed, bool) or
                not isinstance(observed, int)):
            return MetricVerdict(
                metric=metric, kind=kind, expected=expected,
                observed=observed, delta=None, verdict="fail",
                reason=(f"observed must be a non-bool integer, got "
                        f"{type(observed).__name__}"),
            )
        ok = (observed >= expected)
        return MetricVerdict(
            metric=metric, kind=kind, expected=expected,
            observed=observed, delta=None,
            verdict="pass" if ok else "fail",
            reason=None if ok else (
                f"expected >= {expected}, observed {observed}"
            ),
        )
    if kind == "float_within":
        if observed is None:
            return MetricVerdict(
                metric=metric, kind=kind, expected=expected,
                observed=observed, delta=None, verdict="fail",
                reason=("observed is null but expected a float within "
                        f"{expected} +/- {tolerance}"),
            )
        if (isinstance(observed, bool) or
                not isinstance(observed, (int, float))):
            return MetricVerdict(
                metric=metric, kind=kind, expected=expected,
                observed=observed, delta=None, verdict="fail",
                reason=(f"observed must be numeric, got "
                        f"{type(observed).__name__}"),
            )
        delta = abs(float(observed) - float(expected))
        ok = (delta <= float(tolerance))
        return MetricVerdict(
            metric=metric, kind=kind, expected=expected,
            observed=observed, delta=delta,
            verdict="pass" if ok else "fail",
            reason=None if ok else (
                f"observed {observed} outside {expected} +/- "
                f"{tolerance} (delta {delta})"
            ),
        )
    return MetricVerdict(
        metric=metric, kind=kind or "<unknown>",
        expected=expected, observed=observed, delta=None,
        verdict="fail",
        reason=f"unknown check kind {kind!r} in expected_metrics.json",
    )


# Launch-shape axes the harness evaluates BEFORE metric scoring. Each
# axis is sourced from the kernel entry's ``launch`` object in
# expected_metrics.json. A mismatch produces a ``fail`` MetricVerdict
# with ``metric == "launch.<axis>"`` and the report rolls up
# ``all_pass = False``. This is the negative-test surface:
# "mismatched launch parameters do NOT silently match the same
# metrics" (the metric values may still happen to land inside
# tolerance, but the launch-shape verdict locks the report into
# fail-state so the downstream tolerance gate refuses to score
# the run).
_LAUNCH_SHAPE_LIST_AXES = ("grid_dim", "block_dim")
_LAUNCH_SHAPE_SCALAR_AXES = ("dynamic_smem",)


def _evaluate_launch_shape(
    expected_launch: Dict[str, Any],
    record_raw: Dict[str, Any],
) -> List[MetricVerdict]:
    """Compare emitted launch shape against the expectation table's
    per-kernel ``launch`` object. Returns one MetricVerdict per axis
    (``grid_dim``, ``block_dim``, ``dynamic_smem``). Mismatch is a
    ``fail`` verdict; the axes are scored independently so a report
    can carry e.g. ``grid_dim`` pass + ``block_dim`` fail.

    The lists are compared element-wise after coercing to plain
    int-lists, so a producer that emits e.g. ``[32, 1, 1]`` matches
    an expectation of ``[32, 1, 1]`` even if the producer's underlying
    types differ.
    """
    verdicts: List[MetricVerdict] = []
    for axis in _LAUNCH_SHAPE_LIST_AXES:
        expected_value = expected_launch.get(axis)
        if expected_value is None:
            raise CalibrationError(
                f"expected_metrics.json kernel entry's launch object "
                f"missing required axis {axis!r}"
            )
        if (not isinstance(expected_value, list) or
                len(expected_value) != 3 or
                not all(isinstance(x, int) and not isinstance(x, bool)
                        for x in expected_value)):
            raise CalibrationError(
                f"expected_metrics.json launch.{axis} must be a 3-list "
                f"of non-bool integers, got {expected_value!r}"
            )
        observed_value = record_raw.get(axis)
        if (not isinstance(observed_value, list) or
                len(observed_value) != 3):
            verdicts.append(MetricVerdict(
                metric=f"launch.{axis}", kind="exact_list",
                expected=expected_value, observed=observed_value,
                delta=None, verdict="fail",
                reason=(
                    f"emitted record's {axis} must be a 3-element "
                    f"list, got {observed_value!r}"
                ),
            ))
            continue
        # Coerce both to plain int 3-tuples for the equality check so
        # e.g. a JSON `1.0` cannot impersonate an `1`.
        try:
            obs_norm = [int(x) for x in observed_value
                        if isinstance(x, int) and not isinstance(x, bool)]
        except (TypeError, ValueError):
            obs_norm = []
        ok = (len(obs_norm) == 3 and obs_norm == list(expected_value))
        verdicts.append(MetricVerdict(
            metric=f"launch.{axis}", kind="exact_list",
            expected=expected_value, observed=observed_value,
            delta=None, verdict="pass" if ok else "fail",
            reason=None if ok else (
                f"observed {axis} = {observed_value!r}, expected "
                f"{expected_value!r}"
            ),
        ))
    for axis in _LAUNCH_SHAPE_SCALAR_AXES:
        expected_value = expected_launch.get(axis)
        if expected_value is None:
            raise CalibrationError(
                f"expected_metrics.json kernel entry's launch object "
                f"missing required axis {axis!r}"
            )
        if (isinstance(expected_value, bool) or
                not isinstance(expected_value, int)):
            raise CalibrationError(
                f"expected_metrics.json launch.{axis} must be a "
                f"non-bool integer, got {expected_value!r}"
            )
        observed_value = record_raw.get(axis)
        if (isinstance(observed_value, bool) or
                not isinstance(observed_value, int)):
            verdicts.append(MetricVerdict(
                metric=f"launch.{axis}", kind="exact_int",
                expected=expected_value, observed=observed_value,
                delta=None, verdict="fail",
                reason=(
                    f"emitted record's {axis} must be a non-bool "
                    f"integer, got {observed_value!r}"
                ),
            ))
            continue
        ok = (observed_value == expected_value)
        verdicts.append(MetricVerdict(
            metric=f"launch.{axis}", kind="exact_int",
            expected=expected_value, observed=observed_value,
            delta=None, verdict="pass" if ok else "fail",
            reason=None if ok else (
                f"observed {axis} = {observed_value!r}, expected "
                f"{expected_value!r}"
            ),
        ))
    return verdicts


def run_calibration(
    metric_extractor_so: Path,
    calibration_dir: Path,
    expected_json: Path,
    only: Optional[Sequence[str]] = None,
) -> CalibrationReport:
    """Run the full calibration corpus and return a report.

    ``metric_extractor_so``: built ``metric_extractor.so`` to inject.
    ``calibration_dir``: directory containing the built calibration
                         binaries (one per entry in expected_metrics).
    ``expected_json``: path to ``expected_metrics.json``.
    ``only``: optional iterable of binary names to restrict the run.
              Useful for the pytest negative test that only re-runs
              one kernel under a wrapped driver. The default ``None``
              runs every kernel listed in the expected_metrics table.

    Raises ``CalibrationError`` on any structural failure (missing
    binary, unparseable JSON-L, missing expected entry); returns a
    ``CalibrationReport`` with per-kernel per-metric verdicts
    otherwise.
    """
    expected = _load_expected(expected_json)
    corpus_version = expected.get("calibration_corpus_version", "")
    report = CalibrationReport(corpus_version=corpus_version)
    binaries = list(expected["kernels"].keys())
    if only is not None:
        only_set = set(only)
        binaries = [b for b in binaries if b in only_set]
        unknown = only_set - set(expected["kernels"].keys())
        if unknown:
            raise CalibrationError(
                f"`only` specified binaries not in expected_metrics: "
                f"{sorted(unknown)!r}"
            )
    for binary_name in binaries:
        spec = expected["kernels"][binary_name]
        kernel_name_demangled = spec.get("kernel_name", "")
        ac4_test = spec.get("ac4_test", "")
        binary_path = calibration_dir / binary_name
        record = _run_binary_and_parse(binary_path, metric_extractor_so)
        # Validate the demangled kernel name matches what the
        # expectation table promises — otherwise the harness is
        # comparing the wrong baseline.
        observed_kernel = record.raw.get("kernel_name", "")
        if (kernel_name_demangled and
                observed_kernel != kernel_name_demangled):
            raise CalibrationError(
                f"binary {binary_name} emitted kernel "
                f"{observed_kernel!r}, but expected_metrics declares "
                f"{kernel_name_demangled!r}"
            )
        kresult = KernelResult(
            binary=binary_name,
            kernel_name_demangled=observed_kernel,
            ac4_test=ac4_test,
            record=record,
        )
        # Launch-shape verdicts come FIRST so the report carries them
        # even when a downstream metric check raises a defensive
        # CalibrationError (e.g. a malformed branch_active_lanes_hist
        # field). The launch-shape verdicts are the negative-test
        # surface — they catch a mismatched-launch wrapper named to
        # match a canonical binary, regardless of whether the metric
        # value happens to land inside tolerance. Per-axis fail
        # verdicts are recorded as MetricVerdicts on the kernel
        # result; the report's all_pass automatically rolls up.
        expected_launch = spec.get("launch")
        if not isinstance(expected_launch, dict):
            raise CalibrationError(
                f"expected_metrics.json kernel entry "
                f"{binary_name!r} missing required `launch` object"
            )
        kresult.verdicts.extend(
            _evaluate_launch_shape(expected_launch, record.raw)
        )
        # Metric-value checks are scored even when launch-shape
        # verdicts fail, so a triage report carries the diagnostic
        # data side-by-side with the structural fail-state. The
        # tolerance gate (T25) refuses to PROMOTE confidence labels
        # for any kernel whose launch-shape verdicts fail, but the
        # error stats are still computed across what was observed.
        for check in spec.get("checks", []):
            metric = check.get("metric", "")
            observed = _extract_metric(record, metric)
            kresult.verdicts.append(_evaluate_check(check, observed))
        report.kernels.append(kresult)
    return report
