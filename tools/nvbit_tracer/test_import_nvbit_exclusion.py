"""Tests for the NVBit tracer exclusion-sidecar importer.

Covers every positive and negative case for the three AC4 typed
exclusion sidecars:

- ``unsupported_memory_path`` — the tracer encountered an op-class it
  cannot represent (TMA / async_copy / cluster_dsmem / atomic /
  texture / surface).
- ``unmapped_memory_range`` — the tracer's address fell outside any
  allocator-tracked range.
- ``cross_allocation_lane_span`` — a single warp instruction's
  active+predicated lanes resolved to two or more distinct non-zero
  allocation ids, so no single ``allocation_id`` correctly
  represents the warp instruction.

All three sidecars share their structural surface; the typed
reason field dispatches between them. Tests exercise the dispatch
path: a wrong literal, a missing literal, and an unknown literal are
all rejected with typed reasons (``schema_validation_failed`` for the
wrong literal because the schema's ``const`` constraint catches it,
``typed_reason_missing`` for the missing field, ``typed_reason_unknown``
for the unknown value).
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
import import_nvbit_exclusion  # noqa: E402


def _valid_unsupported_sidecar() -> dict:
    """Schema-valid `unsupported_memory_path` sidecar template."""
    return {
        "schema_version": 1,
        "kernel_unique_id": 7,
        "device": "sm_120",
        "slug": "gemm_hmma_bf16_sm80",
        "excluded_op_class": "TMA",
        "instruction_pc": 0x1000,
        "count": 42,
        "typed_reason": "unsupported_memory_path",
    }


def _valid_unmapped_sidecar() -> dict:
    """Schema-valid `unmapped_memory_range` sidecar template."""
    return {
        "schema_version": 1,
        "kernel_unique_id": 7,
        "device": "sm_120",
        "slug": "gemm_hmma_bf16_sm80",
        "excluded_op_class": "GLOBAL",
        "instruction_pc": 0x2000,
        "count": 3,
        "typed_reason": "unmapped_memory_range",
    }


def _valid_cross_allocation_sidecar() -> dict:
    """Schema-valid `cross_allocation_lane_span` sidecar template."""
    return {
        "schema_version": 1,
        "kernel_unique_id": 7,
        "device": "sm_120",
        "slug": "gemm_hmma_bf16_sm80",
        "excluded_op_class": "GLOBAL",
        "instruction_pc": 0x3000,
        "count": 2,
        "typed_reason": "cross_allocation_lane_span",
    }


class ValidSidecarTest(unittest.TestCase):
    def test_unsupported_memory_path_template_passes(self) -> None:
        result = import_nvbit_exclusion.validate(_valid_unsupported_sidecar())
        self.assertTrue(result.ok, msg=result.detail)
        self.assertEqual(result.typed_reason, "unsupported_memory_path")

    def test_unmapped_memory_range_template_passes(self) -> None:
        result = import_nvbit_exclusion.validate(_valid_unmapped_sidecar())
        self.assertTrue(result.ok, msg=result.detail)
        self.assertEqual(result.typed_reason, "unmapped_memory_range")

    def test_cross_allocation_lane_span_template_passes(self) -> None:
        """R5 task11b/d coupled slice: the producer aggregates a
        `cross_allocation_lane_span` typed-exclusion entry when a
        single warp instruction's active lanes resolve to two or more
        distinct non-zero allocation ids. The schema mirrors the
        other two typed-reason schemas; the importer dispatches on
        `typed_reason` and validates against the new schema file."""
        result = import_nvbit_exclusion.validate(
            _valid_cross_allocation_sidecar()
        )
        self.assertTrue(result.ok, msg=result.detail)
        self.assertEqual(
            result.typed_reason, "cross_allocation_lane_span"
        )

    def test_every_op_class_admissible(self) -> None:
        """All op-classes from VALID_OP_CLASSES must be accepted as
        `excluded_op_class` values: the 9 named MemoryOpClass enum
        values plus `UNSPECIFIED` (the sentinel for STANDARD_MEM
        opcodes the classifier could not match against any known
        prefix). A typo in the schema's enum list, a missing
        class, or a drift between the schema enum and the
        importer's VALID_OP_CLASSES would fail this test."""
        for op_class in import_nvbit_exclusion.VALID_OP_CLASSES:
            with self.subTest(op_class=op_class):
                sidecar = _valid_unsupported_sidecar()
                sidecar["excluded_op_class"] = op_class
                result = import_nvbit_exclusion.validate(sidecar)
                self.assertTrue(result.ok, msg=result.detail)

    def test_unspecified_op_class_admitted(self) -> None:
        """Targeted regression test for R24: the producer's
        kUnspecified STANDARD_MEM hook routes records through the
        `unsupported_memory_path` sidecar with
        `excluded_op_class="UNSPECIFIED"`. The schema enum and
        importer must admit this value so a future schema check
        (manual or CI) does not silently reject the live sidecar."""
        sidecar = _valid_unsupported_sidecar()
        sidecar["excluded_op_class"] = "UNSPECIFIED"
        result = import_nvbit_exclusion.validate(sidecar)
        self.assertTrue(result.ok, msg=result.detail)
        self.assertEqual(result.typed_reason, "unsupported_memory_path")


class MandatoryFieldTest(unittest.TestCase):
    """Each required field must be enforced by the schema."""

    def _assert_missing_rejected(
        self,
        template_factory,
        missing_field: str,
    ) -> None:
        sidecar = template_factory()
        del sidecar[missing_field]
        result = import_nvbit_exclusion.validate(sidecar)
        self.assertFalse(
            result.ok,
            msg=f"expected reject for missing {missing_field}",
        )
        if missing_field == "typed_reason":
            # Without typed_reason the importer cannot dispatch a
            # schema; it returns the dispatch-level reason rather
            # than the schema-level one.
            self.assertEqual(result.typed_reason, "typed_reason_missing")
        else:
            self.assertEqual(
                result.typed_reason, "schema_validation_failed"
            )
            self.assertIn(missing_field, result.detail)

    def test_unsupported_each_mandatory_field_rejected(self) -> None:
        for field in (
            "schema_version",
            "kernel_unique_id",
            "device",
            "slug",
            "excluded_op_class",
            "instruction_pc",
            "count",
            "typed_reason",
        ):
            with self.subTest(field=field):
                self._assert_missing_rejected(
                    _valid_unsupported_sidecar, field
                )

    def test_unmapped_each_mandatory_field_rejected(self) -> None:
        for field in (
            "schema_version",
            "kernel_unique_id",
            "device",
            "slug",
            "excluded_op_class",
            "instruction_pc",
            "count",
            "typed_reason",
        ):
            with self.subTest(field=field):
                self._assert_missing_rejected(
                    _valid_unmapped_sidecar, field
                )

    def test_cross_allocation_each_mandatory_field_rejected(
        self,
    ) -> None:
        for field in (
            "schema_version",
            "kernel_unique_id",
            "device",
            "slug",
            "excluded_op_class",
            "instruction_pc",
            "count",
            "typed_reason",
        ):
            with self.subTest(field=field):
                self._assert_missing_rejected(
                    _valid_cross_allocation_sidecar, field
                )


class FieldRangeTest(unittest.TestCase):
    def test_invalid_op_class_rejected(self) -> None:
        sidecar = _valid_unsupported_sidecar()
        sidecar["excluded_op_class"] = "VECTORIZED"  # not in enum
        result = import_nvbit_exclusion.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "schema_validation_failed")

    def test_count_zero_rejected(self) -> None:
        """`count` is the number of instances skipped; an exclusion
        sidecar with count=0 is meaningless and is rejected by the
        schema's `minimum: 1`."""
        sidecar = _valid_unsupported_sidecar()
        sidecar["count"] = 0
        result = import_nvbit_exclusion.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "schema_validation_failed")

    def test_negative_kernel_unique_id_rejected(self) -> None:
        sidecar = _valid_unsupported_sidecar()
        sidecar["kernel_unique_id"] = -1
        result = import_nvbit_exclusion.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "schema_validation_failed")

    def test_negative_instruction_pc_rejected(self) -> None:
        sidecar = _valid_unsupported_sidecar()
        sidecar["instruction_pc"] = -16
        result = import_nvbit_exclusion.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "schema_validation_failed")

    def test_empty_device_string_rejected(self) -> None:
        sidecar = _valid_unsupported_sidecar()
        sidecar["device"] = ""
        result = import_nvbit_exclusion.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "schema_validation_failed")


class SchemaVersionPinTest(unittest.TestCase):
    """R9 regression: the sidecar `schema_version` field is
    pinned via JSON Schema `const: 1`. A producer that bumps the
    version without coordinating with consumers (or a malicious
    sidecar that claims a future version) MUST be rejected, not
    silently admitted. Codex's R8 review observed that
    `minimum: 1` was used instead and that
    `validate(... schema_version=999 ...)` returned ok; this test
    locks in the fix."""

    def test_unsupported_schema_version_999_rejected(self) -> None:
        sidecar = _valid_unsupported_sidecar()
        sidecar["schema_version"] = 999
        result = import_nvbit_exclusion.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "schema_validation_failed")

    def test_unmapped_schema_version_999_rejected(self) -> None:
        sidecar = _valid_unmapped_sidecar()
        sidecar["schema_version"] = 999
        result = import_nvbit_exclusion.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "schema_validation_failed")

    def test_unsupported_schema_version_2_rejected(self) -> None:
        """A future producer that bumps to schema_version 2 must
        fail every consumer until the consumer is also updated."""
        sidecar = _valid_unsupported_sidecar()
        sidecar["schema_version"] = 2
        result = import_nvbit_exclusion.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "schema_validation_failed")

    def test_schema_version_zero_rejected(self) -> None:
        sidecar = _valid_unsupported_sidecar()
        sidecar["schema_version"] = 0
        result = import_nvbit_exclusion.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "schema_validation_failed")


class TypedReasonDispatchTest(unittest.TestCase):
    """The importer dispatches on `typed_reason`. A wrong literal
    that names an op-class but not a known schema must be rejected
    cleanly; a value that names neither schema must be rejected with
    a typed `typed_reason_unknown` reason."""

    def test_wrong_literal_in_unsupported_template_rejected(self) -> None:
        """The unsupported_memory_path schema pins typed_reason via
        const. Putting any other string (even one that names the
        OTHER known schema) trips the schema's const constraint.
        Importer dispatches on typed_reason FIRST, so swapping
        typed_reason to 'unmapped_memory_range' while leaving the
        rest of the sidecar valid actually validates against the
        unmapped schema (since the dispatch picks the unmapped
        schema). Verify that a TRULY unknown literal is rejected
        with typed_reason_unknown."""
        sidecar = _valid_unsupported_sidecar()
        sidecar["typed_reason"] = "made_up_reason"
        result = import_nvbit_exclusion.validate(sidecar)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "typed_reason_unknown")
        self.assertIn("made_up_reason", result.detail)

    def test_typed_reason_swap_revalidates_against_other_schema(self) -> None:
        """A sidecar shaped like an unsupported_memory_path sidecar
        but with typed_reason='unmapped_memory_range' validates
        against the unmapped schema (its `const` matches), proving
        the dispatch path is exact-string and the two schemas are
        kept in lockstep on the structural surface."""
        sidecar = _valid_unsupported_sidecar()
        sidecar["typed_reason"] = "unmapped_memory_range"
        result = import_nvbit_exclusion.validate(sidecar)
        self.assertTrue(result.ok, msg=result.detail)
        self.assertEqual(result.typed_reason, "unmapped_memory_range")

    def test_non_string_typed_reason_rejected(self) -> None:
        sidecar = _valid_unsupported_sidecar()
        sidecar["typed_reason"] = 42
        result = import_nvbit_exclusion.validate(sidecar)
        self.assertFalse(result.ok)
        # The importer treats a non-string value as missing, so the
        # dispatch-level error fires before any schema lookup.
        self.assertEqual(result.typed_reason, "typed_reason_missing")


class ValidatePathTest(unittest.TestCase):
    """``validate_path`` validates the producer's array file shape.

    The producer aggregator
    (`tools/nvbit_tracer/tracer_tool/exclusion_aggregator.cc`) emits
    each typed-reason sidecar as a top-level JSON array of per-entry
    dicts. Tests in this class write that shape to a temp file and
    verify ``validate_path`` admits / rejects appropriately.
    """

    def test_valid_unsupported_sidecar_file_passes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "unsupported_memory_path.json"
            with path.open("w", encoding="utf-8") as fh:
                json.dump([_valid_unsupported_sidecar()], fh)
            result = import_nvbit_exclusion.validate_path(path)
            self.assertTrue(result.ok, msg=result.detail)
            self.assertEqual(
                result.typed_reason, "unsupported_memory_path"
            )

    def test_unreadable_file_typed(self) -> None:
        result = import_nvbit_exclusion.validate_path(
            Path("/path/that/does/not/exist.json")
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "sidecar_unreadable")

    def test_malformed_json_typed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "broken.json"
            with path.open("w", encoding="utf-8") as fh:
                fh.write("{ not valid json")
            result = import_nvbit_exclusion.validate_path(path)
            self.assertFalse(result.ok)
            self.assertEqual(result.typed_reason, "sidecar_unreadable")


class ValidatePathFileShapeTest(unittest.TestCase):
    """File-shape coverage matching the producer's actual output.

    Every test in this class writes the array-shaped JSON the
    aggregator emits and exercises one path through ``validate_path``:
    multi-entry, empty array, top-level object (legacy single-entry
    shape no longer accepted), malformed entry with index, and mixed
    typed reasons within a single file.
    """

    def test_array_with_one_unsupported_entry_passes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "unsupported_memory_path.json"
            with path.open("w", encoding="utf-8") as fh:
                json.dump([_valid_unsupported_sidecar()], fh)
            result = import_nvbit_exclusion.validate_path(path)
            self.assertTrue(result.ok, msg=result.detail)
            self.assertEqual(
                result.typed_reason, "unsupported_memory_path"
            )

    def test_array_with_one_unmapped_entry_passes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "unmapped_memory_range.json"
            with path.open("w", encoding="utf-8") as fh:
                json.dump([_valid_unmapped_sidecar()], fh)
            result = import_nvbit_exclusion.validate_path(path)
            self.assertTrue(result.ok, msg=result.detail)
            self.assertEqual(
                result.typed_reason, "unmapped_memory_range"
            )

    def test_array_with_one_cross_allocation_entry_passes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cross_allocation_lane_span.json"
            with path.open("w", encoding="utf-8") as fh:
                json.dump([_valid_cross_allocation_sidecar()], fh)
            result = import_nvbit_exclusion.validate_path(path)
            self.assertTrue(result.ok, msg=result.detail)
            self.assertEqual(
                result.typed_reason, "cross_allocation_lane_span"
            )

    def test_array_with_multiple_same_reason_passes(self) -> None:
        """Multiple entries that all share the same typed_reason
        round-trip through ``validate_path`` and resolve to that
        single typed_reason at file level. ``entries`` carries the
        post-walk count (R7 polish: callers want both the
        success typed_reason and the entry count without re-parsing
        the file)."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "unsupported_memory_path.json"
            entry_a = _valid_unsupported_sidecar()
            entry_a["instruction_pc"] = 0x1000
            entry_b = _valid_unsupported_sidecar()
            entry_b["instruction_pc"] = 0x2000
            entry_c = _valid_unsupported_sidecar()
            entry_c["instruction_pc"] = 0x3000
            with path.open("w", encoding="utf-8") as fh:
                json.dump([entry_a, entry_b, entry_c], fh)
            result = import_nvbit_exclusion.validate_path(path)
            self.assertTrue(result.ok, msg=result.detail)
            self.assertEqual(
                result.typed_reason, "unsupported_memory_path"
            )
            self.assertEqual(result.entries, 3)

    def test_empty_array_admits_with_sidecar_empty_marker(
        self,
    ) -> None:
        """The producer's empty-typed-reason contract: a sidecar with
        zero aggregated entries is still emitted as ``[]\\n``. The
        importer must admit it; ``detail`` carries ``sidecar_empty``
        so callers that need to know the file was empty can
        distinguish from a non-empty homogeneous file."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cross_allocation_lane_span.json"
            with path.open("w", encoding="utf-8") as fh:
                fh.write("[]\n")
            result = import_nvbit_exclusion.validate_path(path)
            self.assertTrue(result.ok, msg=result.detail)
            self.assertEqual(result.typed_reason, "")
            self.assertEqual(result.detail, "sidecar_empty")

    def test_top_level_object_rejected_with_sidecar_not_a_list(
        self,
    ) -> None:
        """Pre-Round-6 callers wrote a single object (no array
        wrapping); the producer never emitted that shape. The
        importer now rejects it with a typed reason so a
        misconfigured caller fails closed instead of silently
        passing on a single-entry file."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "exclusion.json"
            with path.open("w", encoding="utf-8") as fh:
                json.dump(_valid_unsupported_sidecar(), fh)
            result = import_nvbit_exclusion.validate_path(path)
            self.assertFalse(result.ok)
            self.assertEqual(
                result.typed_reason, "sidecar_not_a_list"
            )

    def test_malformed_entry_rejected_with_index(self) -> None:
        """One malformed entry (missing required field) within an
        array fails the file with the entry index in detail."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "unsupported_memory_path.json"
            entry_a = _valid_unsupported_sidecar()
            entry_b = _valid_unsupported_sidecar()
            del entry_b["count"]  # required field
            with path.open("w", encoding="utf-8") as fh:
                json.dump([entry_a, entry_b], fh)
            result = import_nvbit_exclusion.validate_path(path)
            self.assertFalse(result.ok)
            self.assertEqual(
                result.typed_reason, "schema_validation_failed"
            )
            self.assertIn("entry[1]", result.detail)

    def test_non_dict_entry_rejected_with_index(self) -> None:
        """Entry that is not a JSON object (e.g., a string) fails
        with a schema_validation_failed and the entry index."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "unsupported_memory_path.json"
            entry_a = _valid_unsupported_sidecar()
            with path.open("w", encoding="utf-8") as fh:
                json.dump([entry_a, "not a dict"], fh)
            result = import_nvbit_exclusion.validate_path(path)
            self.assertFalse(result.ok)
            self.assertEqual(
                result.typed_reason, "schema_validation_failed"
            )
            self.assertIn("entry[1]", result.detail)

    def test_mixed_typed_reasons_rejected(self) -> None:
        """Each per-typed-reason file must be homogeneous. A file
        containing one unsupported and one unmapped entry fails
        with ``sidecar_typed_reason_mismatch``."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mixed.json"
            with path.open("w", encoding="utf-8") as fh:
                json.dump(
                    [
                        _valid_unsupported_sidecar(),
                        _valid_unmapped_sidecar(),
                    ],
                    fh,
                )
            result = import_nvbit_exclusion.validate_path(path)
            self.assertFalse(result.ok)
            self.assertEqual(
                result.typed_reason, "sidecar_typed_reason_mismatch"
            )


class CLITest(unittest.TestCase):
    def test_main_returns_zero_for_valid_array(self) -> None:
        import io
        from contextlib import redirect_stdout

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "unmapped_memory_range.json"
            entry_a = _valid_unmapped_sidecar()
            entry_a["instruction_pc"] = 0x1000
            entry_b = _valid_unmapped_sidecar()
            entry_b["instruction_pc"] = 0x2000
            with path.open("w", encoding="utf-8") as fh:
                json.dump([entry_a, entry_b], fh)
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = import_nvbit_exclusion.main([str(path)])
            self.assertEqual(rc, 0)
            out = buf.getvalue()
            # R7 polish: CLI success line names typed_reason
            # AND entries count.
            self.assertIn("typed_reason=unmapped_memory_range", out)
            self.assertIn("entries=2", out)

    def test_main_returns_zero_for_empty_array(self) -> None:
        """Empty `[]` files are admitted; CLI exits 0 and reports
        ``entries=0``."""
        import io
        from contextlib import redirect_stdout

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cross_allocation_lane_span.json"
            with path.open("w", encoding="utf-8") as fh:
                fh.write("[]\n")
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = import_nvbit_exclusion.main([str(path)])
            self.assertEqual(rc, 0)
            self.assertIn("entries=0", buf.getvalue())

    def test_main_returns_nonzero_for_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "exclusion.json"
            entry = _valid_unsupported_sidecar()
            del entry["typed_reason"]
            with path.open("w", encoding="utf-8") as fh:
                json.dump([entry], fh)
            self.assertEqual(
                import_nvbit_exclusion.main([str(path)]), 1
            )


if __name__ == "__main__":
    unittest.main()
