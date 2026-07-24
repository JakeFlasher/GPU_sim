"""Lockstep contract tests for the schema-v5 unit-tagged label substrate.

Covers the end-to-end positive round-trip (a v5 fixture carrying L1/L2
percentage hit rates plus cycles / ns / count metrics travels
parser -> producer -> importer -> aggregator and validates against both v5
schemas), the mixed-version rejection (a pre-v5 sidecar/admitted record is
rejected at parse time), the unit-path-mismatch rejection (a count/ns/cycles
value can never travel the percentage path), the forbidden-field denylist, and
a cohort-wide uniform-upgrade scanner that hard-fails if any committed
sidecar / admitted-set file is left at a pre-v5 version.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import jsonschema

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import import_label  # noqa: E402
import metric_registry  # noqa: E402
import parse_ncu_metrics as pnm  # noqa: E402
import repeatability_aggregator as ra  # noqa: E402
import sidecar_emitter  # noqa: E402

_SIDECAR_SCHEMA = json.loads(
    (_REPO_ROOT / "workloads/simbench_bw/schemas/local_label_sidecar.schema.json")
    .read_text(encoding="utf-8")
)
_ADMITTED_SCHEMA = json.loads(
    (_REPO_ROOT / "workloads/simbench_bw/schemas/local_label_admitted.schema.json")
    .read_text(encoding="utf-8")
)

_NCU_BANNER = "Version 2026.1.1.0 (build 37634170) (public-release)"
_SHA = "a" * 64
_KSIG = "c" * 64
_COMMIT = "b" * 40

# A synthetic NCU details-page CSV carrying one pct hit-rate, plus cycles, ns,
# and a count metric for one launch identity.
_CSV = (
    "ID,Process ID,Kernel Name,Metric Name,Metric Value\n"
    "0,1,k,lts__t_sector_hit_rate.pct,90.5\n"
    "0,1,k,l1tex__t_sector_hit_rate.pct,71.0\n"
    "0,1,k,gpc__cycles_elapsed.max,123456\n"
    "0,1,k,gpu__time_duration.sum,250000\n"
    "0,1,k,dram__sectors_read.sum,\"4,096\"\n"
)


def _build_v5_sidecar(repeat: int, unit_tagged: list[dict]) -> dict:
    """Build a v5 sidecar via the producer (sidecar_emitter) carrying the
    flat L1/L2 fraction fields plus the unit_tagged_metrics block."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False
    ) as fh:
        json.dump(unit_tagged, fh)
        ut_path = fh.name
    env = {
        "SIDECAR_DEVICE": "sm_86",
        "SIDECAR_SLUG": "fake_slug",
        "SIDECAR_SHAPE_POINT": "row_0",
        "SIDECAR_REPEAT_INDEX": str(repeat),
        "SIDECAR_LAUNCH_INDEX": "0",
        "SIDECAR_BINARY_SHA256": _SHA,
        "SIDECAR_PROFILER_VERSION": _NCU_BANNER,
        "SIDECAR_CUDA_VERSION": "13.2",
        "SIDECAR_DRIVER_VERSION": "595.71.05",
        "SIDECAR_GPU_UUID_SHA256": "d" * 64,
        "SIDECAR_AGGREGATION": "kernel",
        "SIDECAR_UNITS": "sector_hit_rate_fraction",
        "SIDECAR_CONFIDENCE_LEVEL": "0.95",
        "SIDECAR_WARMUP_COUNT": "0",
        "SIDECAR_HIT_RATE_L2": "0.905",
        "SIDECAR_HIT_RATE_L1TEX": "0.71",
    }
    return sidecar_emitter.build_sidecar(env, unit_tagged_metrics_path=ut_path)


class V5RoundTripTest(unittest.TestCase):
    """A v5 fixture carrying L1/L2 (pct), cycles, ns, and a traffic count
    round-trips parser -> producer -> importer -> aggregator and validates
    against both v5 schemas."""

    def test_parser_extracts_unit_tagged_metrics_in_native_units(self) -> None:
        reg = metric_registry.load_registry()
        res = pnm.extract_unit_tagged_metrics(
            _CSV,
            reg,
            [
                "gpc__cycles_elapsed.max",
                "gpu__time_duration.sum",
                "dram__sectors_read.sum",
            ],
        )
        self.assertTrue(res.ok, res.detail)
        # cycles / ns / count are RAW (never divided by 100).
        self.assertEqual(res.metrics["gpc__cycles_elapsed.max"], {"unit": "cycles", "value": 123456.0})
        self.assertEqual(res.metrics["gpu__time_duration.sum"], {"unit": "ns", "value": 250000.0})
        self.assertEqual(res.metrics["dram__sectors_read.sum"], {"unit": "count", "value": 4096.0})

    def test_producer_emits_v5_sidecar_that_imports(self) -> None:
        sidecar = _build_v5_sidecar(
            0,
            [
                {"ncu_metric": "gpc__cycles_elapsed.max", "unit": "cycles", "value": 123456.0},
                {"ncu_metric": "gpu__time_duration.sum", "unit": "ns", "value": 250000.0},
                {"ncu_metric": "dram__sectors_read.sum", "unit": "count", "value": 4096.0},
            ],
        )
        self.assertEqual(sidecar["schema_version"], 5)
        self.assertIn("unit_tagged_metrics", sidecar)
        # Validates against the v5 sidecar schema.
        jsonschema.validate(instance=sidecar, schema=_SIDECAR_SCHEMA)
        # And passes the importer's full pipeline (denylist + schema + CI +
        # registry consistency).
        result = import_label.validate(sidecar)
        self.assertTrue(result.ok, result.detail)

    def test_aggregator_admits_unit_tagged_metrics_through_to_v5_admitted(self) -> None:
        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            cell_dir = td / "sm_86" / "fake_slug" / "row_0"
            cell_dir.mkdir(parents=True)
            for repeat in range(5):
                # Low across-repeat dispersion so the per-unit gates pass.
                sidecar = _build_v5_sidecar(
                    repeat,
                    [
                        {"ncu_metric": "gpc__cycles_elapsed.max", "unit": "cycles", "value": 123456.0 + repeat},
                        {"ncu_metric": "gpu__time_duration.sum", "unit": "ns", "value": 250000.0 + 10 * repeat},
                        {"ncu_metric": "dram__sectors_read.sum", "unit": "count", "value": 4096.0},
                    ],
                )
                (cell_dir / f"label_{repeat}_0.json").write_text(json.dumps(sidecar))
                manifest = {
                    "schema_version": 1,
                    "source_commit": _COMMIT,
                    "binary_sha256": _SHA,
                    "kernel_signature_sha256": _KSIG,
                    "arch": "sm_86",
                    "slug": "fake_slug",
                    "shape_point": "row_0",
                    "repeat_index": repeat,
                    "launch_index": 0,
                }
                (cell_dir / f"label_run_manifest_{repeat}_0.json").write_text(
                    json.dumps(manifest)
                )
            _rep, admitted, _exc, _rej = ra.aggregate(td)
            self.assertEqual(admitted["schema_version"], 5)
            self.assertEqual(len(admitted["cells"]), 1)
            cell = admitted["cells"][0]
            jsonschema.validate(instance=admitted, schema=_ADMITTED_SCHEMA)
            ncu = {m["ncu_metric"]: m for m in cell.get("unit_tagged_metrics", [])}
            self.assertIn("gpc__cycles_elapsed.max", ncu)
            self.assertIn("gpu__time_duration.sum", ncu)
            self.assertIn("dram__sectors_read.sum", ncu)
            self.assertEqual(ncu["gpc__cycles_elapsed.max"]["unit"], "cycles")
            self.assertEqual(ncu["gpc__cycles_elapsed.max"]["repeatability_rule"], "coefficient_of_variation")
            self.assertEqual(ncu["dram__sectors_read.sum"]["repeatability_rule"], "relative_tolerance")
            # The flat L2/L1 percentage path is unchanged and still present.
            self.assertAlmostEqual(cell["cache_hit_rate_l2_sector_pct"], 90.5, places=3)
            self.assertEqual(cell["l1tex_label_status"], "mean_present")


class MixedVersionRejectionTest(unittest.TestCase):
    """A pre-v5 record fed to a v5 consumer is rejected at parse time."""

    def test_pre_v5_sidecar_rejected_by_importer(self) -> None:
        sidecar = _build_v5_sidecar(0, [])
        sidecar["schema_version"] = 1  # downgrade to the pre-v5 version
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "schema_validation_failed")

    def test_v4_admitted_record_rejected_by_v5_schema(self) -> None:
        cell = {
            "schema_version": 4,
            "slug": "s", "shape_point": "p", "binary_sha256": _SHA,
            "source_commit": _COMMIT, "kernel_signature_sha256": _KSIG,
            "arch": "sm_86", "ncu_metric_version": "ncu_2026.1.x",
            "profiler_version": "ncu_2026.1.x",
            "cache_hit_rate_l2_sector_pct": 90.0,
            "cache_hit_rate_l1tex_sector_pct": 70.0,
            "l1tex_label_status": "mean_present", "n_repeats": 5,
            "stddev_l2_sector": 0.01, "evidence_kind": "live",
        }
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(
                instance={"schema_version": 5, "cells": [cell]},
                schema=_ADMITTED_SCHEMA,
            )


class UnitPathMismatchRejectionTest(unittest.TestCase):
    """A count/ns/cycles value can never travel the percentage path."""

    def test_count_value_through_percentage_path_rejected_when_out_of_range(self) -> None:
        # parse_raw_page_csv is the percentage-only path; a raw count well
        # above the [0,105] window is rejected as out-of-range rather than
        # silently divided by 100.
        csv = (
            "ID,Metric Name,Metric Value\n"
            "0,dram__sectors_read.sum,\"4,096\"\n"
        )
        res = pnm.parse_raw_page_csv(csv, ["dram__sectors_read.sum"])
        self.assertFalse(res.ok)
        self.assertEqual(res.typed_reason, "ncu_csv_metric_out_of_range")

    def test_importer_rejects_unit_tag_disagreeing_with_registry(self) -> None:
        sidecar = _build_v5_sidecar(
            0,
            [{"ncu_metric": "dram__sectors_read.sum", "unit": "pct", "value": 0.5}],
        )
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "unit_tagged_metric_unit_mismatch")

    def test_importer_rejects_unregistered_unit_tagged_metric(self) -> None:
        sidecar = _build_v5_sidecar(
            0,
            [{"ncu_metric": "not__a_real_metric.sum", "unit": "count", "value": 1.0}],
        )
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "unit_tagged_metric_unregistered")


class ForbiddenFieldDenylistTest(unittest.TestCase):
    """A label sidecar carrying a metric_*/counter_* field name trips the
    importer denylist."""

    def test_metric_name_field_rejected(self) -> None:
        sidecar = _build_v5_sidecar(0, [])
        sidecar["metric_name"] = "lts__t_sector_hit_rate.pct"
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "forbidden_field_name")

    def test_counter_id_field_rejected(self) -> None:
        sidecar = _build_v5_sidecar(0, [])
        sidecar["counter_id"] = 42
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "forbidden_field_name")


class RegistryValueRangeEnforcementTest(unittest.TestCase):
    """The importer enforces the registry's declared valid range on every
    unit-tagged value (and CI bound): a pct value > 1 is rejected even though
    the JSON schema only pins minimum 0; a count value above an UNSET valid_max
    is allowed."""

    def test_pct_value_above_one_rejected(self) -> None:
        sidecar = _build_v5_sidecar(
            0,
            [
                {
                    "ncu_metric": "sm__warps_active.avg.pct_of_peak_sustained_active",
                    "unit": "pct",
                    "value": 1.5,
                }
            ],
        )
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "unit_tagged_metric_value_out_of_range")

    def test_pct_value_within_unit_interval_accepted(self) -> None:
        sidecar = _build_v5_sidecar(
            0,
            [
                {
                    "ncu_metric": "sm__warps_active.avg.pct_of_peak_sustained_active",
                    "unit": "pct",
                    "value": 0.83,
                }
            ],
        )
        self.assertTrue(import_label.validate(sidecar).ok)

    def test_count_value_above_unset_valid_max_allowed(self) -> None:
        sidecar = _build_v5_sidecar(
            0,
            [
                {
                    "ncu_metric": "dram__sectors_read.sum",
                    "unit": "count",
                    "value": 9_999_999_999.0,
                }
            ],
        )
        self.assertTrue(import_label.validate(sidecar).ok)

    def test_pct_confidence_interval_bound_out_of_range_rejected(self) -> None:
        sidecar = _build_v5_sidecar(
            0,
            [
                {
                    "ncu_metric": "sm__warps_active.avg.pct_of_peak_sustained_active",
                    "unit": "pct",
                    "value": 0.5,
                    "confidence_interval": {
                        "lower": 0.4,
                        "upper": 1.4,
                        "confidence_level": 0.95,
                        "unit": "pct",
                        "aggregation": "kernel",
                    },
                }
            ],
        )
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "unit_tagged_metric_value_out_of_range")


class SidecarUnitTaggedFailClosedTest(unittest.TestCase):
    """When SIDECAR_UNIT_TAGGED_METRICS_PATH is explicitly set but the file is
    unreadable / malformed / non-array, the producer fails CLOSED (raises)
    rather than silently dropping the metrics the producer asked for."""

    def _env(self) -> dict:
        return {
            "SIDECAR_DEVICE": "sm_86", "SIDECAR_SLUG": "s",
            "SIDECAR_SHAPE_POINT": "p", "SIDECAR_REPEAT_INDEX": "0",
            "SIDECAR_LAUNCH_INDEX": "0", "SIDECAR_BINARY_SHA256": "a" * 64,
            "SIDECAR_PROFILER_VERSION": _NCU_BANNER, "SIDECAR_CUDA_VERSION": "13.2",
            "SIDECAR_DRIVER_VERSION": "595.71.05", "SIDECAR_GPU_UUID_SHA256": "d" * 64,
            "SIDECAR_AGGREGATION": "kernel", "SIDECAR_UNITS": "sector_hit_rate_fraction",
            "SIDECAR_CONFIDENCE_LEVEL": "0.95", "SIDECAR_WARMUP_COUNT": "0",
            "SIDECAR_HIT_RATE_L2": "0.9", "SIDECAR_HIT_RATE_L1TEX": "0.7",
        }

    def test_missing_file_when_path_set_raises(self) -> None:
        with self.assertRaises(ValueError):
            sidecar_emitter.build_sidecar(
                self._env(), unit_tagged_metrics_path="/no/such/file.json"
            )

    def test_non_array_root_when_path_set_raises(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            fh.write('{"not": "an array"}')
            bad = fh.name
        with self.assertRaises(ValueError):
            sidecar_emitter.build_sidecar(self._env(), unit_tagged_metrics_path=bad)

    def test_malformed_json_when_path_set_raises(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            fh.write("{not json")
            bad = fh.name
        with self.assertRaises(ValueError):
            sidecar_emitter.build_sidecar(self._env(), unit_tagged_metrics_path=bad)

    def test_complete_explicit_ncu_controls_emit_and_validate(self) -> None:
        env = self._env()
        env.update({
            "SIDECAR_NCU_CLOCK_CONTROL": "none",
            "SIDECAR_NCU_CACHE_CONTROL": "all",
            "SIDECAR_NCU_REPLAY_MODE": "kernel",
            "SIDECAR_NCU_TARGET_PROCESSES": "all",
            "SIDECAR_NCU_FILTER_MODE": "global",
            "SIDECAR_NCU_PROFILE_FROM_START": "off",
            "SIDECAR_NCU_LAUNCH_SKIP": "0",
            "SIDECAR_NCU_LAUNCH_COUNT": "2",
        })
        sidecar = sidecar_emitter.build_sidecar(env)
        self.assertEqual(sidecar["measurement_protocol"]["ncu_launch_count"], 2)
        result = import_label.validate(sidecar)
        self.assertTrue(result.ok, msg=result.detail)

    def test_explicit_control_without_clock_control_refuses_emission(self) -> None:
        for field, value in (
            ("SIDECAR_NCU_CACHE_CONTROL", "all"),
            ("SIDECAR_NCU_REPLAY_MODE", "kernel"),
            ("SIDECAR_NCU_TARGET_PROCESSES", "all"),
        ):
            with self.subTest(field=field):
                env = self._env()
                env[field] = value
                with self.assertRaisesRegex(
                    ValueError, "require SIDECAR_NCU_CLOCK_CONTROL"
                ):
                    sidecar_emitter.build_sidecar(env)

    def test_partial_explicit_ncu_controls_refuse_emission(self) -> None:
        env = self._env()
        env.update({
            "SIDECAR_NCU_CLOCK_CONTROL": "none",
            "SIDECAR_NCU_CACHE_CONTROL": "all",
            "SIDECAR_NCU_REPLAY_MODE": "kernel",
            "SIDECAR_NCU_TARGET_PROCESSES": "all",
        })
        with self.assertRaisesRegex(ValueError, "partial explicit NCU"):
            sidecar_emitter.build_sidecar(env)

    def test_noncanonical_ncu_selector_refuses_emission(self) -> None:
        env = self._env()
        env.update({
            "SIDECAR_NCU_CLOCK_CONTROL": "none",
            "SIDECAR_NCU_CACHE_CONTROL": "all",
            "SIDECAR_NCU_REPLAY_MODE": "kernel",
            "SIDECAR_NCU_TARGET_PROCESSES": "all",
            "SIDECAR_NCU_FILTER_MODE": "global",
            "SIDECAR_NCU_PROFILE_FROM_START": "off",
            "SIDECAR_NCU_LAUNCH_SKIP": "00",
            "SIDECAR_NCU_LAUNCH_COUNT": "2",
        })
        with self.assertRaisesRegex(ValueError, "canonical unsigned decimal"):
            sidecar_emitter.build_sidecar(env)

    def test_unset_path_omits_field(self) -> None:
        sidecar = sidecar_emitter.build_sidecar(self._env())
        self.assertNotIn("unit_tagged_metrics", sidecar)


class CohortUniformV5UpgradeScannerTest(unittest.TestCase):
    """Hard-fail if any committed ACTIVE-cohort local-label artifact is left
    at a pre-v5 version (no uneven upgrade). Scope is the active store: the
    operator-facing label_admitted.json plus the per-launch sidecars
    (label_<r>_<l>.json) the aggregator's default-root walk consumes. The
    historical per-config snapshots under data/local_labels/sweeps/ are
    EXCLUDED: they are frozen artifacts from prior loops carrying mixed
    pre-v4 versions (v2 / v3) that were never current-schema-conformant and
    are not schema-validated by any consumer (the aggregator excludes the
    sweeps subtree from its default walk). The separate run-manifest /
    excluded / repeatability families stay at v1 and are out of scope.
    The sm_120 FROZEN pilot rosters (sm120_frozen_roster_l1l2_pilot*,
    sm120_l1l2_live_reverification_pilot) are excluded for the same reason
    as sweeps/: byte-stable historical pilot evidence from the pre-v5 era,
    committed as provenance in the 2026-07-06 pivot baseline, never consumed
    by the aggregator's default walk, and deliberately NOT upgraded (an
    upgrade would break their frozen byte-identity)."""

    _SIDECAR_RE = re.compile(r"/label_\d+_\d+\.json$")
    _FROZEN_SUBTREES = (
        "/sweeps/",
        "data/local_labels/sm120_frozen_roster_l1l2_pilot",
        "data/local_labels/sm120_l1l2_live_reverification_pilot",
    )

    def _git_tracked_json(self) -> list[str]:
        out = subprocess.check_output(
            ["git", "ls-files", "data/local_labels"], cwd=_REPO_ROOT
        ).decode()
        return [
            f
            for f in out.split()
            if f.endswith(".json")
            and not any(t in f for t in self._FROZEN_SUBTREES)
        ]

    def test_every_committed_sidecar_is_v5(self) -> None:
        stragglers = []
        for f in self._git_tracked_json():
            if not self._SIDECAR_RE.search(f):
                continue
            v = json.loads((_REPO_ROOT / f).read_text(encoding="utf-8")).get("schema_version")
            if v != 5:
                stragglers.append(f"{f}: schema_version={v}")
        self.assertEqual(stragglers, [], f"non-v5 committed sidecars: {stragglers}")

    def test_every_committed_admitted_file_is_v5(self) -> None:
        stragglers = []
        for f in self._git_tracked_json():
            if not f.endswith("/label_admitted.json"):
                continue
            d = json.loads((_REPO_ROOT / f).read_text(encoding="utf-8"))
            if not (isinstance(d, dict) and "cells" in d):
                continue
            if d.get("schema_version") != 5:
                stragglers.append(f"{f}: file schema_version={d.get('schema_version')}")
            for i, c in enumerate(d.get("cells") or []):
                if isinstance(c, dict) and c.get("schema_version") != 5:
                    stragglers.append(f"{f}[cell {i}]: schema_version={c.get('schema_version')}")
        self.assertEqual(stragglers, [], f"non-v5 committed admitted records: {stragglers}")

    def test_operator_admitted_file_is_v5_and_value_complete(self) -> None:
        d = json.loads(
            (_REPO_ROOT / "data/local_labels/label_admitted.json").read_text(encoding="utf-8")
        )
        self.assertEqual(d["schema_version"], 5)
        jsonschema.validate(instance=d, schema=_ADMITTED_SCHEMA)


if __name__ == "__main__":
    unittest.main()
