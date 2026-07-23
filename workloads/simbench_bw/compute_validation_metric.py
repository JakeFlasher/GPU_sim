#!/usr/bin/env python3
"""Validation metric framework + schema-check CLI for SimBench-BW.

Owns the input-format contract for every artifact the per-kernel structural
fingerprints, phase ledgers, roofline coordinates, theta-vectors, metamorphic
pair ledgers, and synthetic pilot fixtures emit. Two surfaces:

  1. `simbench_bw_validation --schema-check --all` walks every committed
     artifact in the new schema families and validates each against its
     schema using the custom JSON-Schema-2020-12 validator at
     `workloads/simbench_bw/validate.py::validate`. Exits 0 on full
     success; exits non-zero on the first failure with a typed
     `schema_validation_error` token + the offending JSON path.

  2. The metric registry exposes the primitives the gate scaffold and the
     identifiability ledger consume: per-kernel MAPE; decision-quality
     Top-1 / Top-3; per-bug z-score; distributional KL / Wasserstein /
     Kolmogorov-Smirnov on per-kernel `|gap_ns|` arrays; per-axis
     fingerprint difference. Each metric is callable as
     `(measured, predicted) -> typed-JSON dict`. Registering a duplicate
     metric name raises `MetricNameAlreadyRegistered` at framework-init
     time.

Exit codes:
    0   all validated artifacts conform / metric call succeeded.
    1   at least one artifact failed schema validation
        (typed `schema_validation_error` token in stderr).
    2   usage error (unknown flag, missing file, etc.).
    3   metric-registry init failure
        (typed `metric_name_already_registered` token in stderr).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from validate import ValidationError, validate as _validate_against_schema


SUITE_ROOT = Path(__file__).resolve().parent
SCHEMAS_DIR = SUITE_ROOT / "schemas"
DEFAULT_FIXTURE_DIR = SUITE_ROOT / "tests" / "fixtures" / "simbench_bw_validation"


# Map filename glob -> schema basename. Used by `--schema-check --all` to
# decide which schema each artifact validates against.
_FAMILY_SCHEMA_MAP: tuple[tuple[str, str], ...] = (
    ("kernel_feature_vector_*.json", "kernel_feature_vector.schema.json"),
    ("static_sass_features_*.json", "static_sass_features.schema.json"),
    ("phase_cycle_ledger_*.json", "phase_cycle_ledger.schema.json"),
    ("roofline_coordinate_*.json", "roofline_coordinate.schema.json"),
    ("theta_vector_*.json", "theta_vector.schema.json"),
    ("metamorphic_pair_ledger_*.json", "metamorphic_pair_ledger.schema.json"),
    ("measured_fingerprint_*.json", "measured_fingerprint.schema.json"),
    ("predicted_fingerprint_*.json", "predicted_fingerprint.schema.json"),
    ("synthetic_pillar4_pilot_fixture_*.json", "synthetic_pillar4_pilot_fixture.schema.json"),
    ("bug_detection_ledger_*.json", "bug_detection_ledger.schema.json"),
    ("dual_run_manifest_*.json", "dual_run_manifest.schema.json"),
    ("pre_registration_amendment_*.json", "pre_registration_amendment.schema.json"),
    ("nvml_covariate_*.json", "nvml_covariate.schema.json"),
)


# --- Typed exceptions -------------------------------------------------------


class SchemaValidationError(Exception):
    """Wraps validate.ValidationError to surface the typed token at the CLI
    boundary. The string form starts with `schema_validation_error` so
    operators / tests can grep for the typed token."""

    TOKEN = "schema_validation_error"

    def __init__(self, artifact_path: Path, schema_path: Path, json_path: str, message: str):
        super().__init__(
            f"{self.TOKEN}: {artifact_path} (schema={schema_path.name}) "
            f"at {json_path}: {message}"
        )
        self.artifact_path = artifact_path
        self.schema_path = schema_path
        self.json_path = json_path
        self.message = message

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "artifact_path": str(self.artifact_path),
            "schema_name": self.schema_path.name,
            "json_path": self.json_path,
            "message": self.message,
        }


class MetricNameAlreadyRegistered(Exception):
    """Raised at framework-init time when two metrics try to register
    under the same name. Non-root measurement acceptance criterion: duplicate
    metric registration is rejected at init, not at call time."""

    TOKEN = "metric_name_already_registered"

    def __init__(self, name: str):
        super().__init__(f"{self.TOKEN}: {name!r} is already registered in the metric registry")
        self.name = name


class UnknownArtifactFamily(Exception):
    """Raised when a candidate artifact filename does not match any
    registered schema-family glob. Surfaced fail-closed at
    `--schema-check --all` time so an unrecognized JSON file under the
    schema-check tree cannot bypass validation per the non-root measurement contract's
    'validates every artifact' clause."""

    TOKEN = "unknown_artifact_family"

    def __init__(self, artifact_path: Path):
        super().__init__(
            f"{self.TOKEN}: {artifact_path} does not match any registered "
            f"schema-family glob ({', '.join(g for g, _ in _FAMILY_SCHEMA_MAP)})"
        )
        self.artifact_path = artifact_path

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "artifact_path": str(self.artifact_path),
            "registered_globs": [g for g, _ in _FAMILY_SCHEMA_MAP],
        }


# --- Metric registry --------------------------------------------------------


MetricFn = Callable[[Any, Any], dict[str, Any]]


class MetricRegistry:
    """In-process registry of validation metric primitives.

    Each metric is registered under a unique name and is callable as
    `(measured, predicted) -> dict`. The dict carries a typed JSON
    surface with at least the keys `metric`, `status`, and (when status
    == 'ok') a metric-specific `value` or per-element breakdown.

    The registry rejects duplicate names at registration time so the
    non-root measurement contract's framework-init contract is enforced before any
    artifact is consumed.
    """

    def __init__(self) -> None:
        self._metrics: dict[str, MetricFn] = {}

    def register(self, name: str, fn: MetricFn) -> None:
        if name in self._metrics:
            raise MetricNameAlreadyRegistered(name)
        self._metrics[name] = fn

    def get(self, name: str) -> MetricFn:
        return self._metrics[name]

    def list_metrics(self) -> list[str]:
        return sorted(self._metrics.keys())

    def call(self, name: str, measured: Any, predicted: Any) -> dict[str, Any]:
        return self._metrics[name](measured, predicted)


# --- Metric primitive implementations --------------------------------------


def per_kernel_mape(measured: dict[str, float], predicted: dict[str, float]) -> dict[str, Any]:
    """Mean absolute percentage error per kernel.

    `measured` and `predicted` are dicts keyed by kernel slug with scalar
    values (e.g. wall-clock duration, bandwidth, instruction count). The
    metric returns the mean of `|p - m| / |m|` across kernels with non-zero
    `m`, plus the per-kernel breakdown.
    """
    common = sorted(set(measured.keys()) & set(predicted.keys()))
    per_kernel: dict[str, float] = {}
    errs: list[float] = []
    for k in common:
        m = float(measured[k])
        p = float(predicted[k])
        if m == 0.0:
            continue
        err = abs(p - m) / abs(m)
        per_kernel[k] = err
        errs.append(err)
    if not errs:
        return {
            "metric": "per_kernel_mape",
            "status": "all_zero_or_no_overlap",
            "value": None,
            "n_samples": 0,
        }
    return {
        "metric": "per_kernel_mape",
        "status": "ok",
        "value": sum(errs) / len(errs),
        "per_kernel": per_kernel,
        "n_samples": len(errs),
    }


def decision_quality_top_1(
    measured: dict[str, list[str]], predicted: dict[str, list[str]]
) -> dict[str, Any]:
    """Per-bug top-1 ranking match.

    `measured` and `predicted` are dicts keyed by bug id with values equal
    to a kernel-slug ranking (highest-suspect first). The metric returns
    the fraction of bugs where the predicted top-1 equals the measured
    top-1.
    """
    bugs = sorted(set(measured.keys()) & set(predicted.keys()))
    per_bug: dict[str, bool] = {}
    correct = 0
    for bug in bugs:
        m = measured[bug][0] if measured[bug] else None
        p = predicted[bug][0] if predicted[bug] else None
        match = m == p and m is not None
        per_bug[bug] = match
        if match:
            correct += 1
    if not bugs:
        return {
            "metric": "decision_quality_top_1",
            "status": "no_overlap",
            "value": None,
            "n_bugs": 0,
        }
    return {
        "metric": "decision_quality_top_1",
        "status": "ok",
        "value": correct / len(bugs),
        "per_bug": per_bug,
        "n_bugs": len(bugs),
    }


def decision_quality_top_3(
    measured: dict[str, list[str]], predicted: dict[str, list[str]]
) -> dict[str, Any]:
    """Per-bug top-3 ranking match. Predicted hits when the measured top-1
    appears anywhere in the predicted top 3."""
    bugs = sorted(set(measured.keys()) & set(predicted.keys()))
    per_bug: dict[str, bool] = {}
    correct = 0
    for bug in bugs:
        m_top = measured[bug][0] if measured[bug] else None
        p_top3 = predicted[bug][:3]
        match = m_top is not None and m_top in p_top3
        per_bug[bug] = match
        if match:
            correct += 1
    if not bugs:
        return {
            "metric": "decision_quality_top_3",
            "status": "no_overlap",
            "value": None,
            "n_bugs": 0,
        }
    return {
        "metric": "decision_quality_top_3",
        "status": "ok",
        "value": correct / len(bugs),
        "per_bug": per_bug,
        "n_bugs": len(bugs),
    }


def per_bug_z_score(
    measured: dict[str, dict[str, float]],
    predicted: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Per-bug z-score test statistic.

    `measured[bug] = {value, stddev}` carries the measured detection
    statistic + its standard deviation under the null. `predicted[bug] =
    {value}` carries the model's predicted statistic. The metric
    returns `(predicted - measured) / measured.stddev` per bug.
    """
    bugs = sorted(set(measured.keys()) & set(predicted.keys()))
    per_bug: dict[str, dict[str, Any]] = {}
    for bug in bugs:
        m = measured[bug]
        p = predicted[bug]
        sd = float(m.get("stddev", 0.0))
        if sd == 0.0:
            per_bug[bug] = {"z": None, "status": "measured_stddev_zero"}
        else:
            per_bug[bug] = {
                "z": (float(p["value"]) - float(m["value"])) / sd,
                "status": "ok",
            }
    return {
        "metric": "per_bug_z_score",
        "status": "ok" if bugs else "no_overlap",
        "per_bug": per_bug,
        "n_bugs": len(bugs),
    }


_DISTRIBUTIONAL_MIN_SAMPLES = 8


def _equal_strata(values: list[float], n: int) -> list[float]:
    """Helper: project a sorted sample list to n strata samples."""
    sv = sorted(values)
    out: list[float] = []
    for i in range(n):
        idx = min(int(i * len(sv) / n), len(sv) - 1)
        out.append(sv[idx])
    return out


def distributional_kl(measured: list[float], predicted: list[float]) -> dict[str, Any]:
    """Symmetrized Kullback-Leibler divergence on histogram approximation.

    Both inputs are non-negative scalar arrays (e.g. per-kernel `|gap_ns|`
    distributions). The metric histograms each input into 32 equal-width
    buckets over the joint range, smooths zero-buckets with a tiny prior,
    and returns the mean of `KL(m || p)` and `KL(p || m)`.
    """
    if len(measured) < _DISTRIBUTIONAL_MIN_SAMPLES or len(predicted) < _DISTRIBUTIONAL_MIN_SAMPLES:
        return {
            "metric": "distributional_kl",
            "status": "distributional_undersampled",
            "value": None,
            "n_measured": len(measured),
            "n_predicted": len(predicted),
            "min_samples_required": _DISTRIBUTIONAL_MIN_SAMPLES,
        }
    lo = min(min(measured), min(predicted))
    hi = max(max(measured), max(predicted))
    if hi == lo:
        return {
            "metric": "distributional_kl",
            "status": "ok",
            "value": 0.0,
            "n_measured": len(measured),
            "n_predicted": len(predicted),
        }
    buckets = 32
    width = (hi - lo) / buckets
    epsilon = 1e-9
    hist_m = [epsilon] * buckets
    hist_p = [epsilon] * buckets
    for v in measured:
        idx = min(int((v - lo) / width), buckets - 1)
        hist_m[idx] += 1.0
    for v in predicted:
        idx = min(int((v - lo) / width), buckets - 1)
        hist_p[idx] += 1.0
    sum_m = sum(hist_m)
    sum_p = sum(hist_p)
    p_m = [x / sum_m for x in hist_m]
    p_p = [x / sum_p for x in hist_p]
    kl_mp = sum(a * math.log(a / b) for a, b in zip(p_m, p_p) if a > 0 and b > 0)
    kl_pm = sum(a * math.log(a / b) for a, b in zip(p_p, p_m) if a > 0 and b > 0)
    return {
        "metric": "distributional_kl",
        "status": "ok",
        "value": 0.5 * (kl_mp + kl_pm),
        "kl_measured_to_predicted": kl_mp,
        "kl_predicted_to_measured": kl_pm,
        "n_measured": len(measured),
        "n_predicted": len(predicted),
    }


def distributional_wasserstein(measured: list[float], predicted: list[float]) -> dict[str, Any]:
    """1-Wasserstein distance via equal-probability strata projection."""
    if len(measured) < _DISTRIBUTIONAL_MIN_SAMPLES or len(predicted) < _DISTRIBUTIONAL_MIN_SAMPLES:
        return {
            "metric": "distributional_wasserstein",
            "status": "distributional_undersampled",
            "value": None,
            "n_measured": len(measured),
            "n_predicted": len(predicted),
            "min_samples_required": _DISTRIBUTIONAL_MIN_SAMPLES,
        }
    n = max(len(measured), len(predicted))
    sm = _equal_strata(measured, n)
    sp = _equal_strata(predicted, n)
    return {
        "metric": "distributional_wasserstein",
        "status": "ok",
        "value": sum(abs(a - b) for a, b in zip(sm, sp)) / n,
        "n_measured": len(measured),
        "n_predicted": len(predicted),
    }


def distributional_kolmogorov_smirnov(
    measured: list[float], predicted: list[float]
) -> dict[str, Any]:
    """Two-sample Kolmogorov-Smirnov statistic: max|F_m(x) - F_p(x)|."""
    if len(measured) < _DISTRIBUTIONAL_MIN_SAMPLES or len(predicted) < _DISTRIBUTIONAL_MIN_SAMPLES:
        return {
            "metric": "distributional_kolmogorov_smirnov",
            "status": "distributional_undersampled",
            "value": None,
            "n_measured": len(measured),
            "n_predicted": len(predicted),
            "min_samples_required": _DISTRIBUTIONAL_MIN_SAMPLES,
        }
    sm = sorted(measured)
    sp = sorted(predicted)
    nm = len(sm)
    np_ = len(sp)
    grid = sorted(set(sm) | set(sp))
    max_d = 0.0
    for v in grid:
        cm = sum(1 for x in sm if x <= v) / nm
        cp = sum(1 for x in sp if x <= v) / np_
        d = abs(cm - cp)
        if d > max_d:
            max_d = d
    return {
        "metric": "distributional_kolmogorov_smirnov",
        "status": "ok",
        "value": max_d,
        "n_measured": len(measured),
        "n_predicted": len(predicted),
    }


def _iter_numeric_leaves(obj: Any, prefix: str = "") -> Any:
    """Yield (path, float_value) for every numeric leaf in obj.

    Booleans are skipped (they are subclasses of int but not meaningful
    as fingerprint magnitudes). Strings are skipped — by design, the
    recursive walker treats them as METADATA (claim_class enum values,
    `*_ref` repo-relative paths, citation strings, source-location
    hints) that the per-axis distance must NOT compare. None values are
    skipped. Nested dicts and lists are recursed into; list indices are
    encoded into the path with `[i]` so leaf paths remain stable across
    measured vs predicted.
    """
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield prefix, float(obj)
        return
    if isinstance(obj, dict):
        for k in sorted(obj.keys()):
            sub = f"{prefix}.{k}" if prefix else k
            yield from _iter_numeric_leaves(obj[k], sub)
        return
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            sub = f"{prefix}[{i}]"
            yield from _iter_numeric_leaves(v, sub)
        return
    # Strings, None, and other non-numeric scalars: skip (metadata).


def per_axis_fingerprint_difference(
    measured: dict[str, Any], predicted: dict[str, Any]
) -> dict[str, Any]:
    """Per-axis numeric distance between measured and predicted fingerprint axes.

    The metric walks each axis recursively, ignoring metadata strings
    (claim_class enum values, `*_ref` repo-relative paths, citation
    strings, source-location hints) and computes the L2 distance over the
    set of numeric leaves the two sides have in common. Per-axis output:

      {
        "l2_distance": float,                   # sqrt(sum((m-p)^2) over common leaves)
        "n_common_leaves": int,                 # number of matched numeric leaves
        "n_measured_leaves": int,
        "n_predicted_leaves": int,
        "per_leaf": {leaf_path: signed_diff},   # m - p per leaf
        "measured_only_leaves": [leaf_path...], # leaves present only on measured side
        "predicted_only_leaves": [leaf_path...] # leaves present only on predicted side
      }

    An axis with zero common leaves emits `l2_distance: null` and the
    consumer surfaces that case (e.g. when the static SASS sub-axis is a
    `*_ref` string path on both sides — there are no numeric leaves to
    compare and the walker correctly returns nothing rather than coercing
    a string to float).
    """
    common = sorted(set(measured.keys()) & set(predicted.keys()))
    per_axis: dict[str, Any] = {}
    for axis in common:
        m_leaves = dict(_iter_numeric_leaves(measured[axis]))
        p_leaves = dict(_iter_numeric_leaves(predicted[axis]))
        common_leaves = sorted(set(m_leaves.keys()) & set(p_leaves.keys()))
        measured_only = sorted(set(m_leaves.keys()) - set(p_leaves.keys()))
        predicted_only = sorted(set(p_leaves.keys()) - set(m_leaves.keys()))
        if common_leaves:
            sq_sum = sum(
                (m_leaves[leaf] - p_leaves[leaf]) ** 2 for leaf in common_leaves
            )
            l2 = math.sqrt(sq_sum)
            per_leaf = {leaf: m_leaves[leaf] - p_leaves[leaf] for leaf in common_leaves}
        else:
            l2 = None
            per_leaf = {}
        per_axis[axis] = {
            "l2_distance": l2,
            "n_common_leaves": len(common_leaves),
            "n_measured_leaves": len(m_leaves),
            "n_predicted_leaves": len(p_leaves),
            "per_leaf": per_leaf,
            "measured_only_leaves": measured_only,
            "predicted_only_leaves": predicted_only,
        }
    return {
        "metric": "per_axis_fingerprint_difference",
        "status": "ok" if common else "no_overlap",
        "per_axis": per_axis,
        "n_axes": len(common),
    }


def build_default_registry() -> MetricRegistry:
    """Return a MetricRegistry pre-populated with every metric primitive
    the non-root measurement contract's validation framework requires. Raises
    `MetricNameAlreadyRegistered` if any name collides; the registry is
    fail-closed at framework-init time."""
    reg = MetricRegistry()
    reg.register("per_kernel_mape", per_kernel_mape)
    reg.register("decision_quality_top_1", decision_quality_top_1)
    reg.register("decision_quality_top_3", decision_quality_top_3)
    reg.register("per_bug_z_score", per_bug_z_score)
    reg.register("distributional_kl", distributional_kl)
    reg.register("distributional_wasserstein", distributional_wasserstein)
    reg.register("distributional_kolmogorov_smirnov", distributional_kolmogorov_smirnov)
    reg.register("per_axis_fingerprint_difference", per_axis_fingerprint_difference)
    return reg


# --- Schema-check path ------------------------------------------------------


def schema_for_artifact(artifact_path: Path) -> Path:
    """Return the schema file under SCHEMAS_DIR matching the artifact's
    filename glob. Raises UnknownArtifactFamily for unmatched filenames.

    Uses `fnmatch.fnmatch` for proper Unix-shell-style glob matching so
    that runtime filenames following the non-root measurement contract's verbatim
    convention (`<family>_<repeat>.json` with a single underscore
    separator) and the existing fixture convention (`<family>__<label>.json`
    with a double underscore) both validate against the same family.
    """
    name = artifact_path.name
    for glob, schema in _FAMILY_SCHEMA_MAP:
        if fnmatch.fnmatch(name, glob):
            return SCHEMAS_DIR / schema
    raise UnknownArtifactFamily(artifact_path)


def schema_check_path(artifact_path: Path) -> None:
    """Validate a single artifact against the matching schema. Raises
    SchemaValidationError on failure with the typed token + offending
    JSON path."""
    schema_path = schema_for_artifact(artifact_path)
    schema = json.loads(schema_path.read_text())
    try:
        instance = json.loads(artifact_path.read_text())
    except json.JSONDecodeError as e:
        raise SchemaValidationError(artifact_path, schema_path, "$", f"json_parse_error: {e}")
    try:
        _validate_against_schema(
            instance, schema, root_schema=schema, root_path=schema_path, path="$"
        )
    except ValidationError as e:
        raise SchemaValidationError(artifact_path, schema_path, e.path, e.message)


def discover_fixture_artifacts(fixture_dir: Path) -> list[Path]:
    """Walk fixture_dir for JSON artifacts whose filename matches a
    registered schema-family glob. Returns sorted paths.

    Reserved for diagnostics / counters only. The non-root measurement contract
    requires every artifact under the schema-check tree to be validated
    BEFORE downstream metric or gate computation runs, so the
    schema_check_all path now uses `discover_all_json_artifacts` and
    fail-closes on unrecognized filenames; this helper is retained as a
    convenience for the matched-only count surfaced in `--schema-check
    --all`'s typed-JSON success diagnostic.
    """
    out: list[Path] = []
    if not fixture_dir.is_dir():
        return out
    for p in sorted(fixture_dir.rglob("*.json")):
        try:
            schema_for_artifact(p)
        except UnknownArtifactFamily:
            continue
        out.append(p)
    return out


def discover_all_json_artifacts(fixture_dir: Path) -> list[Path]:
    """Walk fixture_dir for ALL `.json` files (regardless of whether the
    filename matches a registered family glob). Returns sorted paths.
    Used by `schema_check_all` so unrecognized JSON artifacts are
    fail-closed with the typed `unknown_artifact_family` diagnostic
    rather than silently skipped."""
    if not fixture_dir.is_dir():
        return []
    return sorted(fixture_dir.rglob("*.json"))


def schema_check_all(
    fixture_dir: Path,
) -> list[SchemaValidationError | UnknownArtifactFamily]:
    """Validate every JSON artifact under fixture_dir. Returns the list
    of errors encountered; empty list means every artifact resolved to a
    registered schema family AND passed validation. Fail-closed: an
    unrecognized JSON filename is recorded as an `UnknownArtifactFamily`
    diagnostic, not silently skipped."""
    errors: list[SchemaValidationError | UnknownArtifactFamily] = []
    for p in discover_all_json_artifacts(fixture_dir):
        try:
            schema_check_path(p)
        except SchemaValidationError as e:
            errors.append(e)
        except UnknownArtifactFamily as e:
            errors.append(e)
    return errors


# --- CLI dispatch -----------------------------------------------------------


def _emit_json(obj: Any, *, file: Any = None) -> None:
    """Emit obj to the given file (default sys.stdout) using the canonical
    JSON serialization rules from canonical_serialization_spec.md."""
    if file is None:
        file = sys.stdout
    file.write(json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False))
    file.write("\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="simbench_bw_validation",
        description="Validation metric framework + schema-check CLI for SimBench-BW.",
    )
    p.add_argument(
        "--schema-check",
        nargs="?",
        const="__USE_ALL__",
        metavar="PATH",
        help=(
            "Validate a single artifact against its matching schema. "
            "Use --schema-check --all to walk the default fixture dir."
        ),
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="With --schema-check, walk the default fixture dir and validate every artifact.",
    )
    p.add_argument(
        "--fixture-dir",
        type=Path,
        default=DEFAULT_FIXTURE_DIR,
        help="Directory under which --schema-check --all walks for committed artifacts.",
    )
    p.add_argument(
        "--metric-registry-list",
        action="store_true",
        help="Print the registered metric names as canonical-form JSON and exit 0.",
    )
    return p


def _run(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.metric_registry_list:
        try:
            registry = build_default_registry()
        except MetricNameAlreadyRegistered as e:
            sys.stderr.write(f"{e}\n")
            return 3
        _emit_json({"registered_metrics": registry.list_metrics()})
        return 0

    if args.schema_check is not None:
        if args.schema_check == "__USE_ALL__" or args.all:
            fixture_dir = args.fixture_dir
            errors = schema_check_all(fixture_dir)
            if errors:
                _emit_json(
                    {
                        "diagnostic": "schema_check_all_failed",
                        "fixture_dir": str(fixture_dir),
                        "errors": [e.to_json() for e in errors],
                    },
                    file=sys.stderr,
                )
                return 1
            _emit_json(
                {
                    "diagnostic": "schema_check_all_ok",
                    "fixture_dir": str(fixture_dir),
                    "n_artifacts": len(discover_fixture_artifacts(fixture_dir)),
                }
            )
            return 0
        candidate = Path(args.schema_check)
        if not candidate.is_file():
            sys.stderr.write(f"usage_error: not a file: {candidate}\n")
            return 2
        try:
            schema_check_path(candidate)
        except UnknownArtifactFamily as e:
            sys.stderr.write(f"{e}\n")
            return 1
        except SchemaValidationError as e:
            _emit_json(e.to_json(), file=sys.stderr)
            return 1
        _emit_json({"diagnostic": "schema_check_ok", "artifact_path": str(candidate)})
        return 0

    parser.print_help(sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    return _run(argv)


if __name__ == "__main__":
    sys.exit(main())
