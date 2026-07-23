"""Tests for ``workloads/simbench_bw/schemas/local_label_admitted.schema.json``.

The schema validates each admitted-cell record emitted by
``repeatability_aggregator.py`` into ``data/local_labels/label_admitted.json``.
The contract:

- The cell record carries the runner-joinable 7-field key (slug,
  shape_point, binary_sha256, source_commit, kernel_signature_sha256,
  arch, ncu_metric_version) plus the runner-consumed
  ``cache_hit_rate_l2_sector_pct`` value plus ``n_repeats`` and
  ``stddev_l2_sector``.
- ``additionalProperties: false``: any unknown field is rejected.
- ``schema_version`` is pinned at 5 (a const). The 4->5 bump adds the
  optional per-cell ``unit_tagged_metrics`` block (count/ns/cycles families)
  in lockstep with the sidecar schema and the committed metric registry.
- ``profiler_version`` and ``ncu_metric_version`` must equal the
  pinned form ``ncu_2026.1.x``.

The aggregator validates every emitted admitted-cell record against
this schema and fails-closed if validation fails (the schema is the
producer/consumer wire pin).
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import jsonschema

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_SCHEMA_PATH = (
    _REPO_ROOT
    / "workloads"
    / "simbench_bw"
    / "schemas"
    / "local_label_admitted.schema.json"
)

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import repeatability_aggregator as ra  # noqa: E402


# Canonical valid v4 admitted-cell record. Each negative test starts
# from this record and removes / mutates one field at a time to
# exercise a specific schema clause. The v4 contract requires
# evidence_kind, cache_hit_rate_l1tex_sector_pct, AND l1tex_label_status
# on every cell; the live + mean_present variant is the natural default
# because the aggregator only emits live records and computes the L1
# mean when the per-repeat sidecars carry the L1 field.
_VALID_RECORD = {
    "schema_version": 5,
    "slug": "fake_slug",
    "shape_point": "row_0",
    "binary_sha256": "a" * 64,
    "source_commit": "b" * 40,
    "kernel_signature_sha256": "c" * 64,
    "arch": "sm_86",
    "ncu_metric_version": "ncu_2026.1.x",
    "profiler_version": "ncu_2026.1.x",
    "cache_hit_rate_l2_sector_pct": 73.0,
    "cache_hit_rate_l1tex_sector_pct": 55.5,
    "l1tex_label_status": "mean_present",
    "n_repeats": 5,
    "stddev_l2_sector": 0.005,
    "stddev_l1tex_sector": 0.003,
    "evidence_kind": "live",
}


# Pre-migration v3 record (missing cache_hit_rate_l1tex_sector_pct +
# l1tex_label_status). The v4 strict schema rejects this shape with
# the typed reason local_label_admitted_l1_mean_migration_required;
# negative tests use it as a regression guard against accidental
# acceptance of the old form.
_PRE_MIGRATION_V3_RECORD = {
    "schema_version": 3,
    "slug": "fake_slug",
    "shape_point": "row_0",
    "binary_sha256": "a" * 64,
    "source_commit": "b" * 40,
    "kernel_signature_sha256": "c" * 64,
    "arch": "sm_86",
    "ncu_metric_version": "ncu_2026.1.x",
    "profiler_version": "ncu_2026.1.x",
    "cache_hit_rate_l2_sector_pct": 73.0,
    "n_repeats": 5,
    "stddev_l2_sector": 0.005,
    "stddev_l1tex_sector": 0.003,
    "evidence_kind": "live",
}


_REQUIRED_FIELDS = (
    "schema_version",
    "slug",
    "shape_point",
    "binary_sha256",
    "source_commit",
    "kernel_signature_sha256",
    "arch",
    "ncu_metric_version",
    "profiler_version",
    "cache_hit_rate_l2_sector_pct",
    "cache_hit_rate_l1tex_sector_pct",
    "l1tex_label_status",
    "n_repeats",
    "stddev_l2_sector",
    "evidence_kind",
)


def _load_schema() -> dict:
    with _SCHEMA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _record_schema(schema: dict) -> dict:
    """Return the per-record sub-schema. The top-level schema validates
    the full label_admitted.json file (with its ``cells`` list); the
    per-record sub-schema is reachable at ``$defs/admitted_cell``."""
    return schema["$defs"]["admitted_cell"]


class SchemaFileTest(unittest.TestCase):
    def test_schema_file_exists(self) -> None:
        self.assertTrue(_SCHEMA_PATH.is_file(), msg=str(_SCHEMA_PATH))

    def test_schema_loads_as_valid_json(self) -> None:
        schema = _load_schema()
        self.assertIsInstance(schema, dict)
        self.assertIn("$schema", schema)


class SchemaContractTest(unittest.TestCase):
    """Pin the structural clauses the contract requires."""

    def test_record_schema_has_additional_properties_false(self) -> None:
        schema = _record_schema(_load_schema())
        self.assertFalse(schema.get("additionalProperties", True))

    def test_record_schema_required_lists_every_required_field(self) -> None:
        schema = _record_schema(_load_schema())
        required = set(schema.get("required", []))
        for field in _REQUIRED_FIELDS:
            with self.subTest(field=field):
                self.assertIn(field, required)

    def test_record_schema_pins_ncu_metric_version_constant(self) -> None:
        schema = _record_schema(_load_schema())
        prop = schema.get("properties", {}).get("ncu_metric_version", {})
        self.assertEqual(prop.get("const"), "ncu_2026.1.x")

    def test_record_schema_pins_profiler_version_constant(self) -> None:
        schema = _record_schema(_load_schema())
        prop = schema.get("properties", {}).get("profiler_version", {})
        self.assertEqual(prop.get("const"), "ncu_2026.1.x")


class RecordValidationTest(unittest.TestCase):
    """Positive + negative-control validation against the per-record
    sub-schema."""

    def test_canonical_record_validates(self) -> None:
        schema = _record_schema(_load_schema())
        jsonschema.validate(instance=_VALID_RECORD, schema=schema)

    def test_record_missing_required_field_is_rejected(self) -> None:
        schema = _record_schema(_load_schema())
        for field in _REQUIRED_FIELDS:
            with self.subTest(missing=field):
                broken = copy.deepcopy(_VALID_RECORD)
                broken.pop(field, None)
                with self.assertRaises(jsonschema.ValidationError):
                    jsonschema.validate(instance=broken, schema=schema)

    def test_record_with_extra_unknown_field_is_rejected(self) -> None:
        schema = _record_schema(_load_schema())
        broken = copy.deepcopy(_VALID_RECORD)
        broken["unexpected_extra_key"] = "boom"
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=broken, schema=schema)

    def test_record_with_non_pinned_profiler_version_is_rejected(self) -> None:
        schema = _record_schema(_load_schema())
        broken = copy.deepcopy(_VALID_RECORD)
        broken["profiler_version"] = "Version 2024.3.0"
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=broken, schema=schema)

    def test_record_with_non_pinned_ncu_metric_version_is_rejected(self) -> None:
        schema = _record_schema(_load_schema())
        broken = copy.deepcopy(_VALID_RECORD)
        broken["ncu_metric_version"] = "ncu_2024.3.x"
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=broken, schema=schema)


class AggregatorEmissionValidatesTest(unittest.TestCase):
    """The aggregator validates every emitted admitted-cell record
    against the schema. A record that fails validation aborts emission
    fail-closed."""

    def test_aggregator_emitted_record_validates(self) -> None:
        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            cell_dir = td / "sm_86" / "fake_slug" / "row_0"
            cell_dir.mkdir(parents=True)
            for repeat in range(5):
                sidecar = {
                    "schema_version": 5,
                    "device": "sm_86",
                    "slug": "fake_slug",
                    "shape_point": "row_0",
                    "repeat_index": repeat,
                    "launch_index": 0,
                    "binary_sha256": "a" * 64,
                    "profiler_family": "nsight_compute",
                    "profiler_version": (
                        "Version 2026.1.1.0 (build 37634170) (public-release)"
                    ),
                    "cuda_version": "13.2",
                    "driver_version": "595.71.05",
                    "gpu_uuid_sha256": "d" * 64,
                    "replay_mode": "kernel",
                    "aggregation": "kernel",
                    "warmup_count": 0,
                    "cache_hit_rate_l2_sector": 0.730 + 0.001 * repeat,
                    "confidence_interval_l2_sector": {
                        "lower": 0.730,
                        "upper": 0.731,
                        "confidence_level": 0.95,
                        "units": "sector_hit_rate_fraction",
                        "aggregation": "kernel",
                    },
                    "cache_hit_rate_l1tex_sector": 0.555,
                    "confidence_interval_l1tex_sector": {
                        "lower": 0.555,
                        "upper": 0.555,
                        "confidence_level": 0.95,
                        "units": "sector_hit_rate_fraction",
                        "aggregation": "kernel",
                    },
                }
                with (cell_dir / f"label_{repeat}_0.json").open("w") as fh:
                    json.dump(sidecar, fh)
                manifest = {
                    "schema_version": 1,
                    "source_commit": "b" * 40,
                    "binary_sha256": "a" * 64,
                    "kernel_signature_sha256": "c" * 64,
                    "arch": "sm_86",
                    "slug": "fake_slug",
                    "shape_point": "row_0",
                    "repeat_index": repeat,
                    "launch_index": 0,
                }
                with (cell_dir / f"label_run_manifest_{repeat}_0.json").open("w") as fh:
                    json.dump(manifest, fh)
            _repeatability, admitted, _excluded, _ = ra.aggregate(td)
            self.assertEqual(len(admitted["cells"]), 1)
            schema = _record_schema(_load_schema())
            jsonschema.validate(
                instance=admitted["cells"][0], schema=schema,
            )


class FullFileSchemaTest(unittest.TestCase):
    """The top-level schema validates the full label_admitted.json
    file shape (schema_version + cells list)."""

    def test_full_file_validates(self) -> None:
        schema = _load_schema()
        instance = {
            "schema_version": 5,
            "cells": [_VALID_RECORD],
        }
        jsonschema.validate(instance=instance, schema=schema)

    def test_full_file_with_unknown_top_level_key_rejected(self) -> None:
        schema = _load_schema()
        instance = {
            "schema_version": 5,
            "cells": [_VALID_RECORD],
            "rogue_top_level_field": "boom",
        }
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=instance, schema=schema)


# Synthetic-evidence variant of the canonical v4 record. The live
# variant lives at module scope as `_VALID_RECORD` because the
# aggregator only emits live records; tests that exercise the
# synthetic-evidence path use this record explicitly.
_VALID_V4_RECORD_SYNTHETIC = {
    **_VALID_RECORD,
    "evidence_kind": "synthetic",
}


class V4SchemaContractTest(unittest.TestCase):
    """Pin the schema_version const = 5 (bumped 4->5 alongside the optional
    unit_tagged_metrics block), the nullable cache_hit_rate_l1tex_sector_pct
    field, and the required l1tex_label_status enum on the per-record
    sub-schema. The earlier 3->4 bump marked the layer-aware migration that
    lets the comparison runner compute L1 relative error alongside L2.
    evidence_kind values are exactly {live, synthetic}; any other value is
    rejected."""

    def test_record_schema_version_is_const_five(self) -> None:
        schema = _record_schema(_load_schema())
        prop = schema.get("properties", {}).get("schema_version", {})
        self.assertEqual(prop.get("const"), 5)

    def test_top_level_schema_version_is_const_five(self) -> None:
        schema = _load_schema()
        prop = schema.get("properties", {}).get("schema_version", {})
        self.assertEqual(prop.get("const"), 5)

    def test_record_required_includes_evidence_kind(self) -> None:
        schema = _record_schema(_load_schema())
        self.assertIn("evidence_kind", set(schema.get("required", [])))

    def test_record_required_includes_l1_mean_and_status(self) -> None:
        schema = _record_schema(_load_schema())
        required = set(schema.get("required", []))
        self.assertIn("cache_hit_rate_l1tex_sector_pct", required)
        self.assertIn("l1tex_label_status", required)

    def test_record_evidence_kind_enum_is_live_and_synthetic(self) -> None:
        schema = _record_schema(_load_schema())
        prop = schema.get("properties", {}).get("evidence_kind", {})
        self.assertEqual(set(prop.get("enum", [])), {"live", "synthetic"})

    def test_record_l1_status_enum_is_four_values(self) -> None:
        schema = _record_schema(_load_schema())
        prop = schema.get("properties", {}).get("l1tex_label_status", {})
        self.assertEqual(
            set(prop.get("enum", [])),
            {
                "mean_present",
                "mean_absent_label_missing",
                "mean_absent_provenance_missing",
                "all_zero",
            },
        )


class V4RecordValidationTest(unittest.TestCase):
    """Positive + negative-control validation against the v4
    per-record sub-schema, exercising the new evidence_kind +
    cache_hit_rate_l1tex_sector_pct + l1tex_label_status clauses."""

    def test_v4_live_mean_present_record_validates(self) -> None:
        schema = _record_schema(_load_schema())
        jsonschema.validate(instance=_VALID_RECORD, schema=schema)

    def test_v4_synthetic_record_validates(self) -> None:
        schema = _record_schema(_load_schema())
        jsonschema.validate(
            instance=_VALID_V4_RECORD_SYNTHETIC, schema=schema,
        )

    def test_v4_mean_absent_label_missing_validates(self) -> None:
        schema = _record_schema(_load_schema())
        record = copy.deepcopy(_VALID_RECORD)
        record["cache_hit_rate_l1tex_sector_pct"] = None
        record["l1tex_label_status"] = "mean_absent_label_missing"
        jsonschema.validate(instance=record, schema=schema)

    def test_v4_mean_absent_provenance_missing_validates(self) -> None:
        schema = _record_schema(_load_schema())
        record = copy.deepcopy(_VALID_RECORD)
        record["cache_hit_rate_l1tex_sector_pct"] = None
        record["l1tex_label_status"] = "mean_absent_provenance_missing"
        jsonschema.validate(instance=record, schema=schema)

    def test_v4_mean_present_with_null_value_rejected(self) -> None:
        schema = _record_schema(_load_schema())
        broken = copy.deepcopy(_VALID_RECORD)
        broken["cache_hit_rate_l1tex_sector_pct"] = None
        broken["l1tex_label_status"] = "mean_present"
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=broken, schema=schema)

    def test_v4_absent_status_with_non_null_value_rejected(self) -> None:
        schema = _record_schema(_load_schema())
        broken = copy.deepcopy(_VALID_RECORD)
        broken["l1tex_label_status"] = "mean_absent_label_missing"
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=broken, schema=schema)

    def test_v4_record_missing_l1_mean_field_is_rejected(self) -> None:
        schema = _record_schema(_load_schema())
        broken = copy.deepcopy(_VALID_RECORD)
        broken.pop("cache_hit_rate_l1tex_sector_pct")
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=broken, schema=schema)

    def test_v4_record_missing_l1_status_field_is_rejected(self) -> None:
        schema = _record_schema(_load_schema())
        broken = copy.deepcopy(_VALID_RECORD)
        broken.pop("l1tex_label_status")
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=broken, schema=schema)

    def test_v4_record_missing_evidence_kind_is_rejected(self) -> None:
        schema = _record_schema(_load_schema())
        broken = copy.deepcopy(_VALID_RECORD)
        broken.pop("evidence_kind")
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=broken, schema=schema)

    def test_v4_record_with_unknown_evidence_kind_is_rejected(self) -> None:
        schema = _record_schema(_load_schema())
        broken = copy.deepcopy(_VALID_RECORD)
        broken["evidence_kind"] = "unknown"
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=broken, schema=schema)

    def test_pre_migration_v3_record_fails_v4_schema_validation(self) -> None:
        """A v3 record (schema_version=3, no cache_hit_rate_l1tex_sector_pct
        / l1tex_label_status) MUST fail v4 strict validation. Without
        this regression guard, a downstream consumer could accept a
        v3-shape record that silently skips the L1 layer comparison
        path. This is the schema-side enforcement of the typed reason
        local_label_admitted_l1_mean_migration_required."""
        schema = _record_schema(_load_schema())
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(
                instance=_PRE_MIGRATION_V3_RECORD, schema=schema,
            )


class V4FullFileSchemaTest(unittest.TestCase):
    """Top-level v4 file shape: schema_version=4 + cells list of v4
    records. Mirrors FullFileSchemaTest but for the bumped version."""

    def test_v4_full_file_with_live_cell_validates(self) -> None:
        schema = _load_schema()
        instance = {
            "schema_version": 5,
            "cells": [_VALID_RECORD],
        }
        jsonschema.validate(instance=instance, schema=schema)

    def test_v4_full_file_with_synthetic_cell_validates(self) -> None:
        schema = _load_schema()
        instance = {
            "schema_version": 5,
            "cells": [_VALID_V4_RECORD_SYNTHETIC],
        }
        jsonschema.validate(instance=instance, schema=schema)

    def test_v3_full_file_fails_v4_schema(self) -> None:
        schema = _load_schema()
        instance = {
            "schema_version": 3,
            "cells": [_PRE_MIGRATION_V3_RECORD],
        }
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=instance, schema=schema)


class V5MigrationBoundaryTest(unittest.TestCase):
    """The 4->5 schema_version bump is a hard parse-time boundary: a v4
    record/file (even one that is otherwise structurally complete) is
    rejected by the v5 schema, and the new optional unit_tagged_metrics
    block validates only at v5. This is the schema-side enforcement of the
    mixed-version rejection contract (a v4 record fed to a v5 consumer, or
    vice versa, is rejected with a typed error)."""

    def test_v4_record_fails_v5_record_schema(self) -> None:
        schema = _record_schema(_load_schema())
        v4_record = {**_VALID_RECORD, "schema_version": 4}
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=v4_record, schema=schema)

    def test_v4_full_file_fails_v5_schema(self) -> None:
        schema = _load_schema()
        instance = {
            "schema_version": 4,
            "cells": [{**_VALID_RECORD, "schema_version": 4}],
        }
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=instance, schema=schema)

    def test_v5_record_with_unit_tagged_metrics_validates(self) -> None:
        schema = _record_schema(_load_schema())
        record = copy.deepcopy(_VALID_RECORD)
        record["unit_tagged_metrics"] = [
            {
                "ncu_metric": "gpc__cycles_elapsed.max",
                "unit": "cycles",
                "value": 123456.0,
                "dispersion": 0.012,
                "repeatability_rule": "coefficient_of_variation",
                "n_repeats": 5,
            }
        ]
        jsonschema.validate(instance=record, schema=schema)

    def test_v5_record_unit_tagged_metric_missing_required_field_rejected(
        self,
    ) -> None:
        schema = _record_schema(_load_schema())
        record = copy.deepcopy(_VALID_RECORD)
        # missing "dispersion" / "repeatability_rule" / "n_repeats"
        record["unit_tagged_metrics"] = [
            {"ncu_metric": "gpu__time_duration.sum", "unit": "ns", "value": 250000.0}
        ]
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=record, schema=schema)


_DATA_LOCAL_LABELS_ADMITTED_PATH = (
    _REPO_ROOT / "data" / "local_labels" / "label_admitted.json"
)


class OnDiskAdmittedFileV4MigrationTest(unittest.TestCase):
    """The on-disk ``data/local_labels/label_admitted.json`` carries the
    7 admitted sm_86 cells from the local 3090 Ti NCU sweep. After the
    v3 to v4 migration, the file's top-level schema_version is 4 and
    every cell carries ``evidence_kind: "live"`` plus the new
    ``cache_hit_rate_l1tex_sector_pct`` + ``l1tex_label_status``
    fields. The committed file carries
    ``l1tex_label_status="mean_absent_provenance_missing"`` for the
    pre-existing 7 cells because the original per-repeat provenance
    is no longer reproducible from the canonical sidecar paths; a
    fresh NCU sweep emitting cells via the aggregator produces
    records with ``l1tex_label_status="mean_present"`` and a non-null
    L1 mean."""

    def setUp(self) -> None:
        self.assertTrue(
            _DATA_LOCAL_LABELS_ADMITTED_PATH.is_file(),
            msg=str(_DATA_LOCAL_LABELS_ADMITTED_PATH),
        )

    def test_admitted_file_validates_against_v4_schema(self) -> None:
        with _DATA_LOCAL_LABELS_ADMITTED_PATH.open(encoding="utf-8") as fh:
            instance = json.load(fh)
        schema = _load_schema()
        jsonschema.validate(instance=instance, schema=schema)

    def test_admitted_file_top_level_schema_version_is_five(self) -> None:
        with _DATA_LOCAL_LABELS_ADMITTED_PATH.open(encoding="utf-8") as fh:
            instance = json.load(fh)
        self.assertEqual(instance.get("schema_version"), 5)

    def test_admitted_cells_all_carry_evidence_kind_live(self) -> None:
        with _DATA_LOCAL_LABELS_ADMITTED_PATH.open(encoding="utf-8") as fh:
            instance = json.load(fh)
        cells = instance.get("cells", [])
        self.assertGreater(len(cells), 0)
        for cell in cells:
            with self.subTest(slug=cell.get("slug")):
                self.assertEqual(cell.get("evidence_kind"), "live")

    def test_admitted_cells_all_carry_l1_status_field(self) -> None:
        with _DATA_LOCAL_LABELS_ADMITTED_PATH.open(encoding="utf-8") as fh:
            instance = json.load(fh)
        cells = instance.get("cells", [])
        self.assertGreater(len(cells), 0)
        valid_statuses = {
            "mean_present",
            "mean_absent_label_missing",
            "mean_absent_provenance_missing",
            "all_zero",
        }
        for cell in cells:
            with self.subTest(slug=cell.get("slug")):
                self.assertIn(cell.get("l1tex_label_status"), valid_statuses)


class AggregatorEmitsEvidenceKindLiveTest(unittest.TestCase):
    """The aggregator runs only on actual NCU sweep output, so every
    admitted-cell record it emits is live by construction. The
    aggregator must emit ``evidence_kind: "live"`` on every record
    AND bump the file-level schema_version to 4."""

    def test_aggregator_emitted_record_carries_evidence_kind_live(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            cell_dir = td / "sm_86" / "fake_slug" / "row_0"
            cell_dir.mkdir(parents=True)
            for repeat in range(5):
                sidecar = {
                    "schema_version": 5,
                    "device": "sm_86",
                    "slug": "fake_slug",
                    "shape_point": "row_0",
                    "repeat_index": repeat,
                    "launch_index": 0,
                    "binary_sha256": "a" * 64,
                    "profiler_family": "nsight_compute",
                    "profiler_version": (
                        "Version 2026.1.1.0 (build 37634170) (public-release)"
                    ),
                    "cuda_version": "13.2",
                    "driver_version": "595.71.05",
                    "gpu_uuid_sha256": "d" * 64,
                    "replay_mode": "kernel",
                    "aggregation": "kernel",
                    "warmup_count": 0,
                    "cache_hit_rate_l2_sector": 0.730 + 0.001 * repeat,
                    "confidence_interval_l2_sector": {
                        "lower": 0.730,
                        "upper": 0.731,
                        "confidence_level": 0.95,
                        "units": "sector_hit_rate_fraction",
                        "aggregation": "kernel",
                    },
                    "cache_hit_rate_l1tex_sector": 0.555,
                    "confidence_interval_l1tex_sector": {
                        "lower": 0.555,
                        "upper": 0.555,
                        "confidence_level": 0.95,
                        "units": "sector_hit_rate_fraction",
                        "aggregation": "kernel",
                    },
                }
                with (cell_dir / f"label_{repeat}_0.json").open("w") as fh:
                    json.dump(sidecar, fh)
                manifest = {
                    "schema_version": 1,
                    "source_commit": "b" * 40,
                    "binary_sha256": "a" * 64,
                    "kernel_signature_sha256": "c" * 64,
                    "arch": "sm_86",
                    "slug": "fake_slug",
                    "shape_point": "row_0",
                    "repeat_index": repeat,
                    "launch_index": 0,
                }
                with (cell_dir / f"label_run_manifest_{repeat}_0.json").open("w") as fh:
                    json.dump(manifest, fh)
            _repeatability, admitted, _excluded, _ = ra.aggregate(td)
            self.assertEqual(len(admitted["cells"]), 1)
            self.assertEqual(admitted["cells"][0].get("evidence_kind"), "live")

    def test_aggregator_emitted_file_top_level_schema_version_is_five(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            cell_dir = td / "sm_86" / "fake_slug" / "row_0"
            cell_dir.mkdir(parents=True)
            for repeat in range(5):
                sidecar = {
                    "schema_version": 5,
                    "device": "sm_86",
                    "slug": "fake_slug",
                    "shape_point": "row_0",
                    "repeat_index": repeat,
                    "launch_index": 0,
                    "binary_sha256": "a" * 64,
                    "profiler_family": "nsight_compute",
                    "profiler_version": (
                        "Version 2026.1.1.0 (build 37634170) (public-release)"
                    ),
                    "cuda_version": "13.2",
                    "driver_version": "595.71.05",
                    "gpu_uuid_sha256": "d" * 64,
                    "replay_mode": "kernel",
                    "aggregation": "kernel",
                    "warmup_count": 0,
                    "cache_hit_rate_l2_sector": 0.730 + 0.001 * repeat,
                    "confidence_interval_l2_sector": {
                        "lower": 0.730,
                        "upper": 0.731,
                        "confidence_level": 0.95,
                        "units": "sector_hit_rate_fraction",
                        "aggregation": "kernel",
                    },
                    "cache_hit_rate_l1tex_sector": 0.555,
                    "confidence_interval_l1tex_sector": {
                        "lower": 0.555,
                        "upper": 0.555,
                        "confidence_level": 0.95,
                        "units": "sector_hit_rate_fraction",
                        "aggregation": "kernel",
                    },
                }
                with (cell_dir / f"label_{repeat}_0.json").open("w") as fh:
                    json.dump(sidecar, fh)
                manifest = {
                    "schema_version": 1,
                    "source_commit": "b" * 40,
                    "binary_sha256": "a" * 64,
                    "kernel_signature_sha256": "c" * 64,
                    "arch": "sm_86",
                    "slug": "fake_slug",
                    "shape_point": "row_0",
                    "repeat_index": repeat,
                    "launch_index": 0,
                }
                with (cell_dir / f"label_run_manifest_{repeat}_0.json").open("w") as fh:
                    json.dump(manifest, fh)
            _repeatability, admitted, _excluded, _ = ra.aggregate(td)
            self.assertEqual(admitted.get("schema_version"), 5)


class SourceCommitOverrideFromOptionalFieldTest(unittest.TestCase):
    """The optional ``source_commit_override_from`` audit-annotation
    field carries the pre-refresh source_commit SHA when a cell's
    source_commit was rewritten (e.g., via
    ``refresh_admitted_label_source_commit.py``). The field is purely
    additive: it is NOT in ``required``; legacy cells that never
    needed the override remain valid; admitted cells that DID get
    refreshed pin the original SHA for audit traceability. The
    field's type matches ``source_commit`` (non-empty string).

    The committed evidence at ``data/local_labels/label_admitted.json``
    carries this field on 43 of 44 cells; without admission into the
    v4 schema, the file is rejected by
    ``tools/local_labels/merge_admitted_cells.py``'s strict-validation
    gate and the cross-host merge cannot complete.
    """

    def test_record_with_source_commit_override_from_validates(self) -> None:
        schema = _record_schema(_load_schema())
        record = copy.deepcopy(_VALID_RECORD)
        record["source_commit_override_from"] = "9" * 40
        jsonschema.validate(instance=record, schema=schema)

    def test_record_without_source_commit_override_from_still_validates(self) -> None:
        schema = _record_schema(_load_schema())
        record = copy.deepcopy(_VALID_RECORD)
        record.pop("source_commit_override_from", None)
        jsonschema.validate(instance=record, schema=schema)

    def test_record_with_non_string_source_commit_override_from_rejected(self) -> None:
        schema = _record_schema(_load_schema())
        record = copy.deepcopy(_VALID_RECORD)
        record["source_commit_override_from"] = 42
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=record, schema=schema)

    def test_record_with_empty_source_commit_override_from_rejected(self) -> None:
        schema = _record_schema(_load_schema())
        record = copy.deepcopy(_VALID_RECORD)
        record["source_commit_override_from"] = ""
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=record, schema=schema)

    def test_source_commit_override_from_is_optional_not_required(self) -> None:
        schema = _record_schema(_load_schema())
        required = set(schema.get("required", []))
        self.assertNotIn("source_commit_override_from", required)

    def test_source_commit_override_from_declared_in_properties(self) -> None:
        schema = _record_schema(_load_schema())
        properties = schema.get("properties", {})
        self.assertIn("source_commit_override_from", properties)
        spec = properties["source_commit_override_from"]
        self.assertEqual(spec.get("type"), "string")


class LiveLabelAdmittedFileStrictValidationTest(unittest.TestCase):
    """The committed ``data/local_labels/label_admitted.json`` (44
    sm_86 cells; 43 carrying ``source_commit_override_from``) must
    strict-validate against the v4 schema. This is the unblocker for
    ``tools/local_labels/merge_admitted_cells.py``'s strict-validation
    gate, which would otherwise reject the file as input."""

    def test_live_label_admitted_file_strict_validates(self) -> None:
        schema = _load_schema()
        live_path = _REPO_ROOT / "data" / "local_labels" / "label_admitted.json"
        if not live_path.is_file():
            self.skipTest(f"Live label_admitted.json not present at {live_path}")
        with live_path.open(encoding="utf-8") as fh:
            instance = json.load(fh)
        jsonschema.validate(instance=instance, schema=schema)


if __name__ == "__main__":
    unittest.main()
