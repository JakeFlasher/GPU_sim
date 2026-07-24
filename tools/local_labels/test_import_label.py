"""Tests for the local-label sidecar importer.

Covers every positive and negative contract case for the local-label
sidecar pipeline, exercised offline against synthetic JSON dicts (no
GPU, no Nsight Compute invocation needed). Mirrors the contract from
the local-label specification:

- positive: a fully populated sidecar passes.
- negative: each missing mandatory field is rejected.
- negative: a confidence interval with ``lower > upper`` is rejected.
- negative: a forbidden field name (raw HW-counter event identifiers
  and similar tokens) anywhere in the JSON tree is rejected.
- negative: metric values outside ``[0, 1]`` are rejected.
- negative: malformed ``binary_sha256`` and ``gpu_uuid_sha256`` are
  rejected.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import import_label  # noqa: E402
import sidecar_schema_authority  # noqa: E402


_VALID_SHA256_A = "a" * 64
_VALID_SHA256_B = "b" * 64


def _valid_sidecar() -> dict:
    """Return a fresh, schema-valid sidecar dict with all mandatory
    fields populated. Tests mutate copies of this template.

    The layer-aware admitted-cell migration relaxed L1/TEX evidence
    to OPTIONAL at the importer layer (cache_hit_rate_l1tex_sector
    + confidence_interval_l1tex_sector are a coupled pair via the
    schema's dependentRequired arms: either both present or both
    absent; a half-pair is rejected). The template carries both
    L2 and L1/TEX targets so the happy-path positive control
    exercises the full sidecar shape; tests that need the L1
    absence path drop both L1 fields explicitly."""
    return {
        "schema_version": 5,
        "device": "sm_86",
        "slug": "gemm_hmma_bf16_sm80",
        "shape_point": "row_0",
        "repeat_index": 0,
        "launch_index": 0,
        "binary_sha256": _VALID_SHA256_A,
        "profiler_family": "nsight_compute",
        "profiler_version": "2024.3.0",
        "cuda_version": "13.0",
        "driver_version": "560.35.03",
        "gpu_uuid_sha256": _VALID_SHA256_B,
        "replay_mode": "kernel",
        "aggregation": "kernel",
        "warmup_count": 1,
        "cache_hit_rate_l2_sector": 0.42,
        "confidence_interval_l2_sector": {
            "lower": 0.40,
            "upper": 0.44,
            "confidence_level": 0.95,
            "units": "sector_hit_rate_fraction",
            "aggregation": "kernel",
        },
        "cache_hit_rate_l1tex_sector": 0.71,
        "confidence_interval_l1tex_sector": {
            "lower": 0.69,
            "upper": 0.73,
            "confidence_level": 0.95,
            "units": "sector_hit_rate_fraction",
            "aggregation": "kernel",
        },
    }


def _explicit_ncu_protocol() -> dict:
    return {
        "ncu_target_processes": "all",
        "ncu_filter_mode": "global",
        "ncu_replay_mode": "kernel",
        "ncu_cache_control": "all",
        "ncu_clock_control": "none",
        "ncu_profile_from_start": False,
        "ncu_launch_skip": 0,
        "ncu_launch_count": 2,
    }


class ActiveSchemaAuthorityTest(unittest.TestCase):
    def test_active_schema_bytes_match_pinned_authority(self) -> None:
        raw, schema = sidecar_schema_authority.load_authoritative_schema()
        self.assertEqual(schema["schema_version"], 5)
        self.assertEqual(
            schema["properties"]["measurement_protocol"]["properties"]
            ["ncu_target_processes"]["const"],
            "all",
        )
        with self.assertRaisesRegex(ValueError, "pinned authority"):
            sidecar_schema_authority.parse_authoritative_schema_bytes(raw + b" ")


class ExplicitNcuProtocolTest(unittest.TestCase):
    def test_complete_launch_controls_pass(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["measurement_protocol"] = _explicit_ncu_protocol()
        result = import_label.validate(sidecar)
        self.assertTrue(result.ok, msg=result.detail)

    def test_partial_launch_controls_fail_schema_validation(self) -> None:
        sidecar = _valid_sidecar()
        protocol = _explicit_ncu_protocol()
        del protocol["ncu_launch_count"]
        sidecar["measurement_protocol"] = protocol
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "schema_validation_failed")
        self.assertIn("ncu_launch_count", result.detail)

    def test_skip_count_sum_overflow_fails_semantic_validation(self) -> None:
        sidecar = _valid_sidecar()
        protocol = _explicit_ncu_protocol()
        protocol["ncu_launch_skip"] = 2_147_483_647
        protocol["ncu_launch_count"] = 1
        sidecar["measurement_protocol"] = protocol
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "schema_validation_failed")
        self.assertIn("skip plus count exceeds", result.detail)


class ValidSidecarTest(unittest.TestCase):
    def test_minimum_valid_sidecar_passes(self) -> None:
        result = import_label.validate(_valid_sidecar())
        self.assertTrue(result.ok, msg=result.detail)

    def test_l2_only_sidecar_admitted(self) -> None:
        """The layer-aware migration makes L1/TEX evidence OPTIONAL
        at the per-repeat sidecar layer (the layer-aware comparison
        path's `l1tex_label_status=mean_absent_label_missing` arm
        consumes L2-valid/L1-missing aggregated records). A sidecar
        with only the L2 target is now admitted by the importer.
        The aggregator absorbs the L1-absent sidecars into a v4
        admitted record with null L1 mean +
        `l1tex_label_status=mean_absent_label_missing`."""
        sidecar = _valid_sidecar()
        del sidecar["cache_hit_rate_l1tex_sector"]
        del sidecar["confidence_interval_l1tex_sector"]
        result = import_label.validate(sidecar)
        self.assertTrue(result.ok, msg=result.detail)

    def test_l1_metric_without_ci_rejected(self) -> None:
        """Half-pair sidecar with cache_hit_rate_l1tex_sector but no
        confidence_interval_l1tex_sector remains rejected via
        dependentRequired: the two L1 fields are a coupled pair so
        either-both-or-none is the contract; a half-pair is
        malformed."""
        sidecar = _valid_sidecar()
        del sidecar["confidence_interval_l1tex_sector"]
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.typed_reason, "schema_validation_failed",
        )

    def test_l1_ci_without_metric_rejected(self) -> None:
        """Symmetric half-pair: confidence_interval_l1tex_sector
        without cache_hit_rate_l1tex_sector is also rejected."""
        sidecar = _valid_sidecar()
        del sidecar["cache_hit_rate_l1tex_sector"]
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.typed_reason, "schema_validation_failed",
        )

    def test_lower_equals_upper_passes(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["confidence_interval_l2_sector"]["lower"] = 0.42
        sidecar["confidence_interval_l2_sector"]["upper"] = 0.42
        result = import_label.validate(sidecar)
        self.assertTrue(result.ok, msg=result.detail)


class MandatoryFieldTest(unittest.TestCase):
    def _assert_missing_rejected(self, missing_field: str) -> None:
        sidecar = _valid_sidecar()
        del sidecar[missing_field]
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok, msg=f"expected reject for missing {missing_field}")
        self.assertEqual(
            result.typed_reason,
            "schema_validation_failed",
            msg=f"missing {missing_field}: {result.detail}",
        )
        self.assertIn(
            missing_field,
            result.detail,
            msg=f"detail should mention missing field {missing_field}: {result.detail}",
        )

    def test_each_mandatory_field_rejected_when_missing(self) -> None:
        # The L1/TEX metric + CI fields are NOT in this list because
        # the layer-aware migration relaxed them to optional at the
        # sidecar layer (the aggregator emits
        # `l1tex_label_status=mean_absent_label_missing` when none of
        # the per-repeat sidecars carry L1 values). The dependentRequired
        # half-pair rejection arm is still tested separately via
        # test_l1_metric_without_ci_rejected and
        # test_l1_ci_without_metric_rejected.
        for field in (
            "schema_version",
            "device",
            "slug",
            "shape_point",
            "repeat_index",
            "launch_index",
            "binary_sha256",
            "profiler_family",
            "profiler_version",
            "cuda_version",
            "driver_version",
            "gpu_uuid_sha256",
            "replay_mode",
            "aggregation",
            "warmup_count",
            "cache_hit_rate_l2_sector",
            "confidence_interval_l2_sector",
        ):
            with self.subTest(field=field):
                self._assert_missing_rejected(field)


class ConfidenceIntervalTest(unittest.TestCase):
    def test_lower_greater_than_upper_rejected(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["confidence_interval_l2_sector"]["lower"] = 0.50
        sidecar["confidence_interval_l2_sector"]["upper"] = 0.40
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "confidence_interval_inverted")

    def test_lower_greater_than_upper_rejected_in_l1tex(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["cache_hit_rate_l1tex_sector"] = 0.55
        sidecar["confidence_interval_l1tex_sector"] = {
            "lower": 0.60,
            "upper": 0.50,
            "confidence_level": 0.95,
            "units": "sector_hit_rate_fraction",
            "aggregation": "kernel",
        }
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "confidence_interval_inverted")

    def test_wrong_confidence_level_rejected(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["confidence_interval_l2_sector"]["confidence_level"] = 0.99
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "schema_validation_failed")

    def test_wrong_units_rejected(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["confidence_interval_l2_sector"]["units"] = "percent"
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "schema_validation_failed")

    def test_wrong_aggregation_rejected(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["confidence_interval_l2_sector"]["aggregation"] = "warp"
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "schema_validation_failed")


class ForbiddenFieldNameTest(unittest.TestCase):
    """The schema's ``additionalProperties: false`` already blocks
    unknown top-level fields, but the importer's denylist is the
    contract-enforcing path: it scans the entire JSON tree (so a
    forbidden name nested under a permitted block still trips), and it
    returns the specific typed reason ``forbidden_field_name`` instead
    of the generic ``schema_validation_failed``."""

    def test_event_id_at_top_level_rejected(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["event_id"] = 12345
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "forbidden_field_name")
        self.assertIn("event_id", result.detail)

    def test_hw_counter_event_id_rejected(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["hw_counter_event_id"] = "lts__t_sector_hits"
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "forbidden_field_name")

    def test_pm_sampling_metadata_rejected(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["pm_sampling_block"] = {"frequency_hz": 1000}
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "forbidden_field_name")

    def test_cupti_event_array_rejected(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["cupti_event_records"] = [{"name": "lts__t_sector_hits"}]
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "forbidden_field_name")

    def test_pc_sampling_at_top_level_rejected(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["pc_sampling_records"] = []
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "forbidden_field_name")

    def test_metric_id_rejected(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["metric_id"] = "lts__t_sector_hit_rate.pct"
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "forbidden_field_name")
        self.assertIn("metric_id", result.detail)

    def test_metric_name_rejected(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["metric_name"] = "lts__t_sector_hit_rate"
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "forbidden_field_name")

    def test_counter_id_rejected(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["counter_id"] = 42
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "forbidden_field_name")

    def test_counter_name_rejected(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["counter_name"] = "fb__bytes"
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "forbidden_field_name")

    def test_signal_id_rejected(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["signal_id"] = 1
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "forbidden_field_name")

    def test_signal_name_rejected(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["signal_name"] = "fb_bytes"
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "forbidden_field_name")

    def test_nvperf_block_rejected(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["nvperf_attributes"] = {"chip": "GB202"}
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "forbidden_field_name")

    def test_perfworks_block_rejected(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["perfworks_session"] = "abc"
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "forbidden_field_name")

    def test_nested_forbidden_field_in_unknown_block_rejected(self) -> None:
        # Schema rejects unknown top-level blocks, but the importer's
        # tree walker fires first and returns the forbidden_field_name
        # reason regardless of where the offending name lives.
        sidecar = _valid_sidecar()
        sidecar["provenance_extras"] = {
            "metadata": {
                "raw_hw_counter_table": [{"event_name": "fb__bytes"}],
            }
        }
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "forbidden_field_name")
        # Either `raw_hw_counter_table` or `event_name` may trip first
        # depending on dict iteration order; both are forbidden.
        self.assertTrue(
            "raw_hw_counter" in result.detail or "event_name" in result.detail,
            msg=result.detail,
        )


class MetricRangeTest(unittest.TestCase):
    def test_l2_metric_below_zero_rejected(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["cache_hit_rate_l2_sector"] = -0.01
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "schema_validation_failed")

    def test_l2_metric_above_one_rejected(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["cache_hit_rate_l2_sector"] = 1.5
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "schema_validation_failed")

    def test_l1tex_metric_above_one_rejected(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["cache_hit_rate_l1tex_sector"] = 1.2
        sidecar["confidence_interval_l1tex_sector"] = {
            "lower": 0.7,
            "upper": 0.8,
            "confidence_level": 0.95,
            "units": "sector_hit_rate_fraction",
            "aggregation": "kernel",
        }
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "schema_validation_failed")


class L1TexPairDependencyTest(unittest.TestCase):
    """R4 (AC3) makes both `cache_hit_rate_l1tex_sector` and
    `confidence_interval_l1tex_sector` mandatory at the top-level
    `required` array. The schema's `dependentRequired` clause is now
    redundant but stays as belt-and-suspenders. Either field alone is
    a contract violation; either omitted is a contract violation."""

    def test_metric_without_ci_rejected(self) -> None:
        sidecar = _valid_sidecar()
        # Strip the partner field; only the metric remains.
        del sidecar["confidence_interval_l1tex_sector"]
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "schema_validation_failed")
        self.assertIn("confidence_interval_l1tex_sector", result.detail)

    def test_ci_without_metric_rejected(self) -> None:
        sidecar = _valid_sidecar()
        # Strip the partner field; only the CI remains.
        del sidecar["cache_hit_rate_l1tex_sector"]
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "schema_validation_failed")
        self.assertIn("cache_hit_rate_l1tex_sector", result.detail)


class ProvenanceFormatTest(unittest.TestCase):
    def test_malformed_binary_sha256_rejected(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["binary_sha256"] = "not-a-real-sha"
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "schema_validation_failed")

    def test_malformed_gpu_uuid_sha256_rejected(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["gpu_uuid_sha256"] = "ZZZ"
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "schema_validation_failed")


class ValidatePathTest(unittest.TestCase):
    def test_valid_sidecar_file_passes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "label_0_0.json"
            with path.open("w", encoding="utf-8") as fh:
                json.dump(_valid_sidecar(), fh)
            result = import_label.validate_path(path)
            self.assertTrue(result.ok, msg=result.detail)

    def test_unreadable_file_typed(self) -> None:
        result = import_label.validate_path(Path("/nonexistent/sidecar.json"))
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "sidecar_unreadable")

    def test_malformed_json_typed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "label_0_0.json"
            with path.open("w", encoding="utf-8") as fh:
                fh.write("{not json")
            result = import_label.validate_path(path)
            self.assertFalse(result.ok)
            self.assertEqual(result.typed_reason, "sidecar_unreadable")


class ParserDiagnosticsRoundTripTest(unittest.TestCase):
    """The per-launch sidecar schema carries an optional
    `parser_diagnostics` field that records non-fatal events the metric
    parser wishes the audit trail to retain (e.g. NCU's
    `lts__t_sector_hit_rate.pct` saturation-rounding overshoots clamped
    to fraction 1.0). The importer must (a) admit a sidecar with the
    new shape, (b) admit a sidecar without the field (additive widening
    per BL-20260507), and (c) reject any unknown sub-key or unknown
    record field (closed enum per BL-20260506)."""

    def test_sidecar_with_clamp_diagnostic_admits(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["parser_diagnostics"] = {
            "ncu_sector_hit_rate_clamped_to_100": [
                {
                    "metric_label": "lts__t_sector_hit_rate.pct",
                    "raw_value": 101.45,
                },
            ],
        }
        result = import_label.validate(sidecar)
        self.assertTrue(result.ok, msg=result.detail)

    def test_sidecar_with_multiple_clamp_records_admits(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["parser_diagnostics"] = {
            "ncu_sector_hit_rate_clamped_to_100": [
                {
                    "metric_label": "lts__t_sector_hit_rate.pct",
                    "raw_value": 102.28,
                },
                {
                    "metric_label": "l1tex__t_sector_hit_rate.pct",
                    "raw_value": 100.5,
                },
            ],
        }
        result = import_label.validate(sidecar)
        self.assertTrue(result.ok, msg=result.detail)

    def test_sidecar_without_parser_diagnostics_admits(self) -> None:
        """Additive widening per BL-20260507 producer-side rules: the
        field is optional, so sidecars that predate the producer change
        validate unchanged."""
        sidecar = _valid_sidecar()
        self.assertNotIn("parser_diagnostics", sidecar)
        result = import_label.validate(sidecar)
        self.assertTrue(result.ok, msg=result.detail)

    def test_unknown_diagnostic_key_rejected(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["parser_diagnostics"] = {
            "ncu_unknown_future_diagnostic": [
                {"metric_label": "x", "raw_value": 0.0},
            ],
        }
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "schema_validation_failed")

    def test_unknown_record_field_rejected(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["parser_diagnostics"] = {
            "ncu_sector_hit_rate_clamped_to_100": [
                {
                    "metric_label": "lts__t_sector_hit_rate.pct",
                    "raw_value": 101.0,
                    "unknown_record_field": "extra",
                },
            ],
        }
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "schema_validation_failed")

    def test_missing_record_metric_label_rejected(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["parser_diagnostics"] = {
            "ncu_sector_hit_rate_clamped_to_100": [
                {"raw_value": 101.0},
            ],
        }
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "schema_validation_failed")

    def test_missing_record_raw_value_rejected(self) -> None:
        sidecar = _valid_sidecar()
        sidecar["parser_diagnostics"] = {
            "ncu_sector_hit_rate_clamped_to_100": [
                {"metric_label": "lts__t_sector_hit_rate.pct"},
            ],
        }
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "schema_validation_failed")

    def test_record_with_forbidden_metric_name_key_rejected(self) -> None:
        """If a future producer accidentally serializes the record
        with `metric_name` instead of `metric_label`, the importer's
        forbidden-name denylist catches it before schema validation
        even gets a chance — the typed reason is `forbidden_field_name`,
        not `schema_validation_failed`."""
        sidecar = _valid_sidecar()
        sidecar["parser_diagnostics"] = {
            "ncu_sector_hit_rate_clamped_to_100": [
                {"metric_name": "lts__t_sector_hit_rate.pct", "raw_value": 101.0},
            ],
        }
        result = import_label.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "forbidden_field_name")


class CLITest(unittest.TestCase):
    def test_main_returns_zero_for_valid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "label_0_0.json"
            with path.open("w", encoding="utf-8") as fh:
                json.dump(_valid_sidecar(), fh)
            self.assertEqual(import_label.main([str(path)]), 0)

    def test_main_returns_nonzero_for_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "label_0_0.json"
            invalid = _valid_sidecar()
            del invalid["binary_sha256"]
            with path.open("w", encoding="utf-8") as fh:
                json.dump(invalid, fh)
            self.assertEqual(import_label.main([str(path)]), 1)


class PackagePathImportTest(unittest.TestCase):
    """Paired control for a prior Codex review P2: the documented
    package path (`from tools.local_labels import import_label` with
    ONLY the repo root on sys.path) must work. The bare sibling
    `from metric_registry import ...` used to raise
    ModuleNotFoundError under that convention; the module now inserts
    its own directory first (the emit_ac4_load_bearing_coverage.py
    convention). A clean subprocess is required because this test
    module itself runs with tools/local_labels already on sys.path,
    which would mask the regression in-process."""

    def test_import_via_package_path_with_repo_root_only(self) -> None:
        import subprocess
        repo_root = Path(__file__).resolve().parents[2]
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "from tools.local_labels import import_label; "
                "print(import_label.__name__)",
                str(repo_root),
            ],
            capture_output=True,
            text=True,
            cwd=str(repo_root / "tools"),  # NOT the module dir
        )
        self.assertEqual(
            proc.returncode, 0,
            f"package-path import failed:\n{proc.stderr}",
        )
        self.assertIn("import_label", proc.stdout)


if __name__ == "__main__":
    unittest.main()
