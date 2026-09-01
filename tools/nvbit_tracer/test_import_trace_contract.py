"""Tests for the offline trace-contract validator.

Builds small synthetic ``threadblock`` protos in memory using the
generated Python bindings and exercises both the positive-admission
path and each negative-rejection path with the right typed reason.
The fixtures are deliberately minimal — one warp, one instruction —
so each test focuses on a single contract clause without the noise
of a real trace.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from tools.nvbit_tracer.proto_binding_loader import (  # noqa: E402
    load_dynamic_trace_bindings,
)


def _ensure_protobuf_bindings_available() -> tuple[Any, Any, Any]:
    """Load fixtures through the production authenticated authority."""
    bindings = load_dynamic_trace_bindings()
    return (
        bindings.threadblock_pb2,
        bindings.instruction_pb2,
        bindings.address_pb2,
    )


def _import_pb() -> tuple[Any, Any, Any]:
    """Backwards-compatible alias used inside test methods. Always
    routes through `_ensure_protobuf_bindings_available` so tracked-authority
    checks apply uniformly."""
    return _ensure_protobuf_bindings_available()


def _make_threadblock_with_instruction(
    *,
    set_data_width: bool = True,
    data_width_value: int = 4,
    set_warpid_tb: bool = True,
    warpid_tb_value: int = 0,
    add_address: bool = False,
    set_memory_op_class: bool = False,
    memory_op_class_value: int | None = None,
    set_memory_op_access_kind: bool = False,
    memory_op_access_kind_value: int | None = None,
    set_instruction_cache_op: bool = False,
    instruction_cache_op_value: int | None = None,
    address_space_value: int | None = None,
) -> Any:
    """Build a one-instruction threadblock fixture and return it.

    Each kwarg toggles whether the corresponding contract field is
    populated and (where applicable) the value to populate it with.
    The defaults produce a minimal positive-admission fixture: a
    non-memory instruction (no addresses, no memory_op_class) with
    valid data_width=4 and warpid_tb=0.
    """
    threadblock_pb2, _instruction_pb2, address_pb2 = _import_pb()

    tb = threadblock_pb2.threadblock()
    tb.block_id.x = 0
    tb.block_id.y = 0
    tb.block_id.z = 0
    warp = tb.warps[0]
    warp.id = 0
    inst = warp.instructions.add()
    inst.pc = 0x1000
    # Default lane mask: only lane 0 active (active_mask &
    # predicate_mask == 0x1). The fixture's `add_address=True`
    # branch puts the real address at slot 0 and sentinels at
    # slots 1-31, so the lane mask must match for positional
    # identity (R18). Tests that need a different mask shape
    # overwrite both fields and (when `add_address=True`) the
    # corresponding addrs slots after the fixture returns.
    inst.active_mask = 0x1
    inst.predicate_mask = 0x1
    inst.function_unique_id = 1
    inst.vpc = 0x1000

    if set_data_width:
        inst.data_width = data_width_value
    if set_warpid_tb:
        inst.warpid_tb = warpid_tb_value
    if add_address:
        # R17 AC4 32-entry sentinel format: emit one outer
        # `address` record with `compression_format=list_all`
        # (proto value 0, the default) and exactly 32 addrs
        # entries — one slot per lane. The fixture defaults to a
        # single-active-lane shape: lane 0 carries a real
        # address; lanes 1-31 carry the sentinel (-1). Tests that
        # need a different lane shape mutate after the fixture
        # returns. Tests that build the predicated-off
        # all-lanes-inactive case overwrite all 32 slots with
        # the sentinel.
        addr = inst.addresses.add()
        addr.data_width = 4
        addr.addrs.append(0xCAFE0000)
        for _ in range(31):
            addr.addrs.append(-1)
        if address_space_value is not None:
            addr.address_space = address_space_value
    if set_memory_op_class:
        assert memory_op_class_value is not None
        inst.memory_op_class = memory_op_class_value
    if set_memory_op_access_kind:
        assert memory_op_access_kind_value is not None
        inst.memory_op_access_kind = memory_op_access_kind_value
    if set_instruction_cache_op:
        assert instruction_cache_op_value is not None
        inst.instruction_cache_op = instruction_cache_op_value

    return tb


class TraceContractValidatorPositiveTest(unittest.TestCase):
    """Positive-admission cases: a threadblock that satisfies the
    contract on every instruction must validate."""

    def test_non_memory_instruction_admits(self) -> None:
        from import_trace_contract import validate_threadblock

        tb = _make_threadblock_with_instruction()
        result = validate_threadblock(tb)
        self.assertTrue(
            result.ok,
            msg=(
                f"expected admission, got {result.typed_reason}: "
                f"{result.detail}"
            ),
        )

    def test_instrumentation_order_requires_identity_and_strict_warp_order(self) -> None:
        from import_trace_contract import validate_threadblock

        tb = _make_threadblock_with_instruction()
        first = tb.warps[0].instructions[0]
        first.instrumentation_order = 1
        first.sm_id = 0
        first.warpid_sm = 0
        result = validate_threadblock(tb)
        self.assertTrue(result.ok, msg=f"{result.typed_reason}: {result.detail}")

        first.instrumentation_order = 0
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "instrumentation_order_zero")

        first.instrumentation_order = 1
        first.ClearField("sm_id")
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.typed_reason, "instrumentation_order_identity_missing"
        )

        first.sm_id = 0
        second = tb.warps[0].instructions.add()
        second.CopyFrom(first)
        second.pc = 0x1004
        second.vpc = 0x1004
        second.instrumentation_order = 1
        first.instrumentation_order = 2
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.typed_reason, "instrumentation_order_not_increasing"
        )

    def test_memory_instruction_with_class_admits(self) -> None:
        # SHARED is used here (rather than GLOBAL) because this test
        # exercises the generic memory-op admission path, not
        # allocation-identity semantics. SHARED records have no
        # allocation_id requirement, so the test stays focused on
        # data_width / class-presence regardless of the new
        # `allocation_id_missing` gate that fires only on GLOBAL.
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_SHARED,
        )
        result = validate_threadblock(tb)
        self.assertTrue(
            result.ok,
            msg=(
                f"expected admission, got {result.typed_reason}: "
                f"{result.detail}"
            ),
        )

    def test_warpid_tb_zero_admits(self) -> None:
        """warpid_tb=0 is a valid warp id; the validator must
        distinguish "value 0" from "absent" via HasField, not via
        truthiness. Regression case for the proto3 zero-default
        collapse the R8 v1 review caught.
        """
        from import_trace_contract import validate_threadblock

        tb = _make_threadblock_with_instruction(warpid_tb_value=0)
        result = validate_threadblock(tb)
        self.assertTrue(
            result.ok,
            msg=(
                f"expected admission, got {result.typed_reason}: "
                f"{result.detail}"
            ),
        )

    def test_non_memory_without_data_width_admits(self) -> None:
        """Non-memory instructions (no addresses, no memory_op_class)
        legitimately leave data_width absent. The device-side stub
        in inject_funcs.cu reports 0 for non-memory; the producer
        therefore omits data_width on those records to keep the
        wire honest about non-applicability.
        """
        from import_trace_contract import validate_threadblock

        tb = _make_threadblock_with_instruction(set_data_width=False)
        result = validate_threadblock(tb)
        self.assertTrue(
            result.ok,
            msg=(
                f"expected admission, got {result.typed_reason}: "
                f"{result.detail}"
            ),
        )

    def test_explicit_memory_space_compatibility_matrix_admits(self) -> None:
        """Operation families admit only their reviewed explicit spaces."""
        from import_trace_contract import validate_threadblock

        _tb_pb2, instruction_pb2, address_pb2 = _import_pb()
        cases = (
            (
                instruction_pb2.MEMORY_OP_CLASS_GLOBAL,
                address_pb2.MEMORY_SPACE_GLOBAL,
            ),
            (
                instruction_pb2.MEMORY_OP_CLASS_GLOBAL,
                address_pb2.MEMORY_SPACE_GENERIC,
            ),
            (
                instruction_pb2.MEMORY_OP_CLASS_SHARED,
                address_pb2.MEMORY_SPACE_SHARED,
            ),
            (
                instruction_pb2.MEMORY_OP_CLASS_LOCAL,
                address_pb2.MEMORY_SPACE_LOCAL,
            ),
            (
                instruction_pb2.MEMORY_OP_CLASS_ATOMIC,
                address_pb2.MEMORY_SPACE_GLOBAL,
            ),
            (
                instruction_pb2.MEMORY_OP_CLASS_ATOMIC,
                address_pb2.MEMORY_SPACE_GENERIC,
            ),
            (
                instruction_pb2.MEMORY_OP_CLASS_ATOMIC,
                address_pb2.MEMORY_SPACE_SHARED,
            ),
            (
                instruction_pb2.MEMORY_OP_CLASS_ATOMIC,
                address_pb2.MEMORY_SPACE_LOCAL,
            ),
            (
                instruction_pb2.MEMORY_OP_CLASS_ASYNC_COPY,
                address_pb2.MEMORY_SPACE_GLOBAL,
            ),
            (
                instruction_pb2.MEMORY_OP_CLASS_ASYNC_COPY,
                address_pb2.MEMORY_SPACE_GENERIC,
            ),
        )
        for op_class, address_space in cases:
            with self.subTest(op_class=op_class, address_space=address_space):
                tb = _make_threadblock_with_instruction(
                    add_address=True,
                    set_memory_op_class=True,
                    memory_op_class_value=op_class,
                    address_space_value=address_space,
                )
                inst = tb.warps[0].instructions[0]
                inst.predicate_mask = 0
                inst.addresses[0].addrs[0] = -1
                result = validate_threadblock(tb)
                self.assertTrue(result.ok, result.detail)

    def test_missing_address_space_retains_legacy_fallback(self) -> None:
        from import_trace_contract import validate_threadblock

        _tb_pb2, instruction_pb2, _address_pb2 = _import_pb()
        for op_class in (
            instruction_pb2.MEMORY_OP_CLASS_GLOBAL,
            instruction_pb2.MEMORY_OP_CLASS_SHARED,
            instruction_pb2.MEMORY_OP_CLASS_LOCAL,
            instruction_pb2.MEMORY_OP_CLASS_ATOMIC,
            instruction_pb2.MEMORY_OP_CLASS_ASYNC_COPY,
        ):
            with self.subTest(op_class=op_class):
                tb = _make_threadblock_with_instruction(
                    add_address=True,
                    set_memory_op_class=True,
                    memory_op_class_value=op_class,
                )
                inst = tb.warps[0].instructions[0]
                inst.predicate_mask = 0
                inst.addresses[0].addrs[0] = -1
                result = validate_threadblock(tb)
                self.assertTrue(result.ok, result.detail)

    def test_empty_threadblock_admits(self) -> None:
        """A threadblock with no warps is vacuously contract-conformant
        — there are no instructions to violate the contract on. This
        guards the directory walk against a benign empty .pb breaking
        the whole sweep.
        """
        from import_trace_contract import validate_threadblock
        threadblock_pb2, _instruction_pb2, _addr_pb2 = _import_pb()

        tb = threadblock_pb2.threadblock()
        result = validate_threadblock(tb)
        self.assertTrue(
            result.ok,
            msg=(
                f"expected admission, got {result.typed_reason}: "
                f"{result.detail}"
            ),
        )


class TraceContractValidatorNegativeTest(unittest.TestCase):
    """Each of these constructs a threadblock with one targeted
    contract violation and asserts the validator rejects it with the
    right typed_reason."""

    def test_data_width_missing_on_memory_inst_rejected(self) -> None:
        """data_width is required on memory instructions. The
        signal "looks like memory" comes from either an address
        record or a present memory_op_class — this test uses the
        addresses-non-empty arm.
        """
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            set_data_width=False,
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_GLOBAL,
        )
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "data_width_missing")

    def test_data_width_missing_on_class_only_inst_rejected(
        self,
    ) -> None:
        """The "memory_op_class present, addresses empty" arm of
        looks_like_memory also requires data_width. This is the
        predicated-off all-lanes-inactive case the proto comment
        highlights.
        """
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            set_data_width=False,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_GLOBAL,
        )
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "data_width_missing")

    def test_data_width_zero_on_memory_inst_rejected(self) -> None:
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            data_width_value=0,
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_GLOBAL,
        )
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "data_width_invalid")

    def test_data_width_negative_on_memory_inst_rejected(self) -> None:
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            data_width_value=-1,
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_GLOBAL,
        )
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "data_width_invalid")

    def test_warpid_tb_missing_rejected(self) -> None:
        from import_trace_contract import validate_threadblock

        tb = _make_threadblock_with_instruction(set_warpid_tb=False)
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "warpid_tb_missing")

    def test_memory_op_class_missing_on_memory_inst_rejected(
        self,
    ) -> None:
        from import_trace_contract import validate_threadblock

        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=False,
        )
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.typed_reason,
            "memory_op_class_missing_on_memory_instruction",
        )

    def test_memory_op_class_unspecified_on_memory_inst_rejected(
        self,
    ) -> None:
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=(
                instruction_pb2.MEMORY_OP_CLASS_UNSPECIFIED
            ),
        )
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.typed_reason,
            "memory_op_class_unspecified",
        )

    def test_class_only_unspecified_is_rejected(self) -> None:
        from import_trace_contract import validate_threadblock

        _tb_pb2, instruction_pb2, _address_pb2 = _import_pb()
        tb = _make_threadblock_with_instruction(
            add_address=False,
            set_memory_op_class=True,
            memory_op_class_value=(
                instruction_pb2.MEMORY_OP_CLASS_UNSPECIFIED
            ),
        )
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "memory_op_class_unspecified")

    def test_unknown_memory_class_is_schema_skew(self) -> None:
        from import_trace_contract import validate_threadblock

        for add_address in (False, True):
            with self.subTest(add_address=add_address):
                tb = _make_threadblock_with_instruction(
                    add_address=add_address,
                    set_memory_op_class=True,
                    memory_op_class_value=99,
                )
                result = validate_threadblock(tb)
                self.assertFalse(result.ok)
                self.assertEqual(result.typed_reason, "unknown_enum_value")
                self.assertIn("memory_op_class", result.detail)

    def test_unknown_address_space_is_schema_skew_for_unsupported_class(
        self,
    ) -> None:
        from import_trace_contract import validate_threadblock

        _tb_pb2, instruction_pb2, _address_pb2 = _import_pb()
        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_TEXTURE,
            address_space_value=99,
        )
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "unknown_enum_value")
        self.assertIn("address_space", result.detail)

    def test_earlier_unknown_space_precedes_later_address_damage(self) -> None:
        from import_trace_contract import validate_threadblock

        _tb_pb2, instruction_pb2, _address_pb2 = _import_pb()
        for compression_format in (0, 2):
            with self.subTest(compression_format=compression_format):
                tb = _make_threadblock_with_instruction(
                    add_address=True,
                    set_memory_op_class=True,
                    memory_op_class_value=(
                        instruction_pb2.MEMORY_OP_CLASS_SHARED
                    ),
                    address_space_value=99,
                )
                malformed_later = (
                    tb.warps[0].instructions[0].addresses.add()
                )
                malformed_later.compression_format = compression_format
                if compression_format == 2:
                    malformed_later.addrs.append(1)
                result = validate_threadblock(tb)
                self.assertFalse(result.ok)
                self.assertEqual(result.typed_reason, "unknown_enum_value")
                self.assertIn("address_space", result.detail)

    def test_explicit_unspecified_address_space_is_rejected(self) -> None:
        from import_trace_contract import validate_threadblock

        _tb_pb2, instruction_pb2, address_pb2 = _import_pb()
        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_ATOMIC,
            address_space_value=address_pb2.MEMORY_SPACE_UNSPECIFIED,
        )
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "memory_space_unspecified")

    def test_crossed_memory_spaces_are_rejected(self) -> None:
        from import_trace_contract import validate_threadblock

        _tb_pb2, instruction_pb2, address_pb2 = _import_pb()
        cases = (
            (
                instruction_pb2.MEMORY_OP_CLASS_SHARED,
                address_pb2.MEMORY_SPACE_LOCAL,
            ),
            (
                instruction_pb2.MEMORY_OP_CLASS_LOCAL,
                address_pb2.MEMORY_SPACE_SHARED,
            ),
            (
                instruction_pb2.MEMORY_OP_CLASS_GLOBAL,
                address_pb2.MEMORY_SPACE_SHARED,
            ),
            (
                instruction_pb2.MEMORY_OP_CLASS_GLOBAL,
                address_pb2.MEMORY_SPACE_LOCAL,
            ),
        )
        for op_class, address_space in cases:
            with self.subTest(op_class=op_class, address_space=address_space):
                tb = _make_threadblock_with_instruction(
                    add_address=True,
                    set_memory_op_class=True,
                    memory_op_class_value=op_class,
                    address_space_value=address_space,
                )
                result = validate_threadblock(tb)
                self.assertFalse(result.ok)
                self.assertEqual(
                    result.typed_reason,
                    "memory_op_class_address_space_conflict",
                )

    def test_async_copy_shared_operand_is_structurally_compatible(self) -> None:
        from import_trace_contract import validate_threadblock

        _tb_pb2, instruction_pb2, address_pb2 = _import_pb()
        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_ASYNC_COPY,
            address_space_value=address_pb2.MEMORY_SPACE_SHARED,
        )

        result = validate_threadblock(tb)

        self.assertTrue(result.ok, result)

    def test_structural_error_precedes_space_conflict(self) -> None:
        from import_trace_contract import validate_threadblock

        _tb_pb2, instruction_pb2, address_pb2 = _import_pb()
        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_SHARED,
            address_space_value=address_pb2.MEMORY_SPACE_LOCAL,
        )
        malformed_later = tb.warps[0].instructions[0].addresses.add()
        malformed_later.compression_format = 0
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "lane_vector_length_invalid")

    def test_structural_error_precedes_allocation_identity(self) -> None:
        from import_trace_contract import validate_threadblock

        _tb_pb2, instruction_pb2, address_pb2 = _import_pb()
        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_GLOBAL,
            address_space_value=address_pb2.MEMORY_SPACE_GLOBAL,
        )
        malformed_later = tb.warps[0].instructions[0].addresses.add()
        malformed_later.compression_format = 0
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "lane_vector_length_invalid")

    def test_space_conflict_precedes_allocation_identity(self) -> None:
        """A contradictory GLOBAL space cannot reroute a SHARED record."""
        from import_trace_contract import validate_threadblock

        _tb_pb2, instruction_pb2, address_pb2 = _import_pb()
        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_SHARED,
            address_space_value=address_pb2.MEMORY_SPACE_GLOBAL,
        )
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.typed_reason,
            "memory_op_class_address_space_conflict",
        )

    def test_missing_class_precedes_explicit_space(self) -> None:
        from import_trace_contract import validate_threadblock

        _tb_pb2, _instruction_pb2, address_pb2 = _import_pb()
        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=False,
            address_space_value=address_pb2.MEMORY_SPACE_GLOBAL,
        )
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.typed_reason,
            "memory_op_class_missing_on_memory_instruction",
        )

    def test_memory_op_access_kind_absent_on_memory_inst_admitted(
        self,
    ) -> None:
        """Legacy traces emitted before the producer-side access-
        kind classifier landed leave the wire field absent. A
        memory instruction with `memory_op_class` set to a valid
        value (GLOBAL) AND `memory_op_access_kind` absent must
        still admit; the consumer-side reader in
        `tools/hrd_profiler/observation_builder.py` falls back to
        the UNSPECIFIED bucket for absent fields. This pins the
        back-compat contract for the additive proto3 `optional`
        widening at tag 12."""
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=(
                instruction_pb2.MEMORY_OP_CLASS_SHARED
            ),
            set_memory_op_access_kind=False,
        )
        result = validate_threadblock(tb)
        self.assertTrue(
            result.ok,
            msg=f"legacy admission: {result.typed_reason}: {result.detail}",
        )

    def test_memory_op_access_kind_unspecified_on_memory_inst_rejected(
        self,
    ) -> None:
        """A memory instruction whose `memory_op_access_kind` is
        present on the wire AND set to UNSPECIFIED is a producer-
        side fail-closed event: the producer should have routed
        the opcode through the typed-exclusion sidecar instead of
        emitting a kind-tagged-with-UNSPECIFIED wire record.
        Mirrors the `memory_op_class_unspecified` clause."""
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=(
                instruction_pb2.MEMORY_OP_CLASS_SHARED
            ),
            set_memory_op_access_kind=True,
            memory_op_access_kind_value=(
                instruction_pb2.MEMORY_OP_ACCESS_KIND_UNSPECIFIED
            ),
        )
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.typed_reason,
            "memory_op_access_kind_unspecified",
        )

    def test_memory_op_access_kind_load_on_memory_inst_admitted(
        self,
    ) -> None:
        """A memory instruction with `memory_op_access_kind` set
        to LOAD admits. Positive control for the new wire field —
        the producer-classifier-equipped tracer emits this shape
        for every LDG / LDS / LDL / TEX / TLD / SULD / UTMALDG
        opcode."""
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=(
                instruction_pb2.MEMORY_OP_CLASS_SHARED
            ),
            set_memory_op_access_kind=True,
            memory_op_access_kind_value=(
                instruction_pb2.MEMORY_OP_ACCESS_KIND_LOAD
            ),
        )
        result = validate_threadblock(tb)
        self.assertTrue(
            result.ok,
            msg=f"LOAD admission: {result.typed_reason}: {result.detail}",
        )

    def test_known_additive_memory_metadata_on_non_memory_rejected(self) -> None:
        from import_trace_contract import validate_threadblock

        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()
        cases = (
            (
                "memory_op_access_kind",
                instruction_pb2.MEMORY_OP_ACCESS_KIND_LOAD,
            ),
            (
                "instruction_cache_op",
                instruction_pb2.INSTRUCTION_CACHE_OP_LDG_E_CA,
            ),
        )
        for field, value in cases:
            with self.subTest(field=field):
                tb = _make_threadblock_with_instruction()
                setattr(tb.warps[0].instructions[0], field, value)
                result = validate_threadblock(tb)
                self.assertFalse(result.ok)
                self.assertEqual(
                    result.typed_reason,
                    "memory_metadata_on_non_memory_instruction",
                )
                self.assertIn(field, result.detail)

    def test_unspecified_cache_op_on_memory_instruction_rejected(self) -> None:
        from import_trace_contract import validate_threadblock

        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()
        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_SHARED,
            set_instruction_cache_op=True,
            instruction_cache_op_value=(
                instruction_pb2.INSTRUCTION_CACHE_OP_UNSPECIFIED
            ),
        )
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.typed_reason,
            "instruction_cache_op_unspecified",
        )

    def test_unknown_additive_instruction_enums_are_schema_skew(self) -> None:
        from import_trace_contract import validate_threadblock

        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()
        for field in ("memory_op_access_kind", "instruction_cache_op"):
            with self.subTest(field=field):
                tb = _make_threadblock_with_instruction(
                    add_address=True,
                    set_memory_op_class=True,
                    memory_op_class_value=(
                        instruction_pb2.MEMORY_OP_CLASS_SHARED
                    ),
                )
                setattr(tb.warps[0].instructions[0], field, 99)
                result = validate_threadblock(tb)
                self.assertFalse(result.ok)
                self.assertEqual(result.typed_reason, "unknown_enum_value")
                self.assertIn(field, result.detail)

    def test_active_mask_missing_rejected(self) -> None:
        """active_mask is marked `optional` in the proto for
        explicit-presence. A regression that dropped the
        producer's inst->set_active_mask(...) call leaves the
        field absent on the wire (HasField returns false). The
        presence check fires BEFORE the lane-mask consistency
        value check, so a missing field surfaces as a presence
        failure with a precise typed reason rather than as a
        downstream value mismatch."""
        from import_trace_contract import validate_threadblock

        tb = _make_threadblock_with_instruction()
        warp = tb.warps[0]
        warp.instructions[0].ClearField("active_mask")
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.typed_reason, "active_mask_missing",
        )

    def test_active_mask_zero_admits(self) -> None:
        """active_mask=0 is a valid lane mask (predicated-off
        all-lanes-inactive case). The presence check distinguishes
        "tracer did not emit" (HasField false) from "tracer
        emitted value 0" (HasField true), so an explicit zero
        admits."""
        from import_trace_contract import validate_threadblock

        tb = _make_threadblock_with_instruction()
        warp = tb.warps[0]
        warp.instructions[0].active_mask = 0
        warp.instructions[0].predicate_mask = 0
        result = validate_threadblock(tb)
        self.assertTrue(
            result.ok,
            msg=(
                f"expected admission, got {result.typed_reason}: "
                f"{result.detail}"
            ),
        )

    def test_active_mask_missing_precedes_lane_mask_inconsistent(
        self,
    ) -> None:
        """When BOTH `active_mask` is absent AND an inner addrs
        list is non-empty (which would otherwise trip
        `lane_mask_inconsistent`), the validator must surface the
        presence failure first. The presence check is the more
        precise diagnostic — `lane_mask_inconsistent` is a value-
        only invariant that depends on the masks being meaningful,
        and reading inst.active_mask on an absent field returns
        the proto3 default 0 which would otherwise produce a
        misleading 'AND mask zero' diagnostic."""
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_GLOBAL,
        )
        warp = tb.warps[0]
        warp.instructions[0].ClearField("active_mask")
        # Leave predicate_mask present + non-empty inner addrs
        # from the fixture so the lane_mask_inconsistent rule
        # WOULD trip on the value side.
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.typed_reason, "active_mask_missing",
            msg=(
                f"presence check must surface before "
                f"lane_mask_inconsistent; got {result.typed_reason}"
            ),
        )

    def test_predicate_mask_missing_rejected(self) -> None:
        """predicate_mask is marked `optional` for the same
        explicit-presence reason as active_mask. A regression
        that dropped inst->set_predicate_mask(...) surfaces as a
        typed contract failure."""
        from import_trace_contract import validate_threadblock

        tb = _make_threadblock_with_instruction()
        warp = tb.warps[0]
        warp.instructions[0].ClearField("predicate_mask")
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.typed_reason, "predicate_mask_missing",
        )

    def test_vpc_missing_rejected(self) -> None:
        """vpc is marked `optional` so the validator distinguishes
        "tracer did not emit vpc" from "vpc=0" (legitimate value
        when pc=0 for the first instruction in a kernel). A
        regression that dropped inst->set_vpc(...) is rejected
        with vpc_missing rather than passing the bitmask
        comparison silently."""
        from import_trace_contract import validate_threadblock

        tb = _make_threadblock_with_instruction()
        warp = tb.warps[0]
        warp.instructions[0].ClearField("vpc")
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "vpc_missing")

    def test_vpc_zero_with_pc_zero_admits(self) -> None:
        """vpc=0 is legitimate when pc=0 (the first instruction
        in a kernel function lives at function-relative offset
        0). The presence check distinguishes this from "tracer
        did not emit"."""
        from import_trace_contract import validate_threadblock

        tb = _make_threadblock_with_instruction()
        warp = tb.warps[0]
        warp.instructions[0].pc = 0
        warp.instructions[0].vpc = 0
        result = validate_threadblock(tb)
        self.assertTrue(
            result.ok,
            msg=(
                f"expected admission, got {result.typed_reason}: "
                f"{result.detail}"
            ),
        )

    def test_function_unique_id_zero_rejected(self) -> None:
        """The tracer's instrument_function_if_needed assigns the
        first function id 1 (it increments
        next_candidate_unique_function_id from 0 before assigning).
        A wire record carrying function_unique_id=0 is the proto3
        default and means the producer did not classify the
        function — reject so a regression in the function-id
        wiring surfaces as a typed contract failure."""
        from import_trace_contract import validate_threadblock

        tb = _make_threadblock_with_instruction()
        warp = tb.warps[0]
        warp.instructions[0].function_unique_id = 0
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.typed_reason, "function_unique_id_invalid",
        )

    def test_function_unique_id_negative_rejected(self) -> None:
        """function_unique_id values are NVBit-assigned positive
        integers starting at 1; a negative wire value indicates
        either a corrupted record or a synthesized non-conforming
        path."""
        from import_trace_contract import validate_threadblock

        tb = _make_threadblock_with_instruction()
        warp = tb.warps[0]
        warp.instructions[0].function_unique_id = -1
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.typed_reason, "function_unique_id_invalid",
        )

    def test_vpc_diverges_from_pc_low_32_bits_rejected(self) -> None:
        """The producer sets pc and vpc from the same
        per-instruction PC offset (tracer_tool.cu lines 2546 and
        2573). A wire record whose vpc low 32 bits do not match
        pc means the tracer failed to populate vpc (proto3 zero
        default) or corrupted the record."""
        from import_trace_contract import validate_threadblock

        tb = _make_threadblock_with_instruction()
        warp = tb.warps[0]
        # pc=0x1000 from the fixture; set vpc to a different
        # low-32 value so the bitmask comparison rejects it.
        warp.instructions[0].vpc = 0x2000
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.typed_reason, "vpc_inconsistent_with_pc",
        )

    def test_vpc_zero_with_nonzero_pc_rejected(self) -> None:
        """The proto3 zero default for vpc on an instruction with
        a non-zero pc means the tracer did not populate the field.
        This is the most common regression mode the validator
        catches."""
        from import_trace_contract import validate_threadblock

        tb = _make_threadblock_with_instruction()
        warp = tb.warps[0]
        warp.instructions[0].vpc = 0  # proto3 default
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.typed_reason, "vpc_inconsistent_with_pc",
        )

    def test_vpc_with_high_bits_admits(self) -> None:
        """The bitmask comparison preserves forward compatibility
        with the queued channel-side widening of vpc to a true
        64-bit virtual PC: a future emitter that sets the high
        32 bits is admitted as long as the low 32 bits agree
        with pc."""
        from import_trace_contract import validate_threadblock

        tb = _make_threadblock_with_instruction()
        warp = tb.warps[0]
        # pc=0x1000 from the fixture; set vpc with high bits set
        # but low 32 bits matching pc.
        warp.instructions[0].vpc = (0x12345678 << 32) | 0x1000
        result = validate_threadblock(tb)
        self.assertTrue(
            result.ok,
            msg=(
                f"expected admission, got {result.typed_reason}: "
                f"{result.detail}"
            ),
        )

    def test_lane_mask_inconsistent_with_inner_addrs_rejected(
        self,
    ) -> None:
        """A memory instruction whose AND mask is zero but whose
        inner address record's `addrs` list is non-empty violates
        the producer's per-lane emission gate at tracer_tool.cu
        line 2665 (`if (mask.test(s)) addr->add_addrs(...)`).
        Rejecting this catches an internal inconsistency that a
        conforming producer cannot generate."""
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_GLOBAL,
        )
        warp = tb.warps[0]
        # AND mask zero; the fixture's add_address=True populates
        # one address record with one inner addrs entry, which is
        # exactly the inconsistent shape.
        warp.instructions[0].active_mask = 0xAAAAAAAA
        warp.instructions[0].predicate_mask = 0x55555555
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.typed_reason, "lane_mask_inconsistent",
        )

    def test_zero_lane_masks_with_all_sentinel_addrs_admits(
        self,
    ) -> None:
        """Under the AC4 32-entry sentinel format, the predicated-
        off all-lanes-inactive STANDARD_MEM case has one outer
        `addresses` entry whose inner `addrs` list is exactly 32
        `LANE_INACTIVE_SENTINEL` (-1) entries. The validator must
        admit this — the lane-mask consistency check only counts
        non-sentinel addrs entries, so all sentinels means "no
        real address claimed", which is consistent with AND mask
        zero."""
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_GLOBAL,
        )
        warp = tb.warps[0]
        # AND mask zero. Overwrite all 32 addrs slots with the
        # sentinel so no real address is claimed.
        warp.instructions[0].active_mask = 0xAAAAAAAA
        warp.instructions[0].predicate_mask = 0x55555555
        addr_record = warp.instructions[0].addresses[0]
        del addr_record.addrs[:]
        for _ in range(32):
            addr_record.addrs.append(-1)
        result = validate_threadblock(tb)
        self.assertTrue(
            result.ok,
            msg=(
                f"expected admission of predicated-off STANDARD_MEM "
                f"with all-sentinel inner addrs, got "
                f"{result.typed_reason}: {result.detail}"
            ),
        )

    def test_list_all_with_seven_entries_rejected(self) -> None:
        """AC4 requires the `list_all` compression format to
        carry exactly 32 entries (one slot per lane). A `list_all`
        record with 7 entries is malformed — likely a regression
        in the producer's lane-emission gate or a legacy
        pre-Round-17 sparse trace. Reject with
        `lane_vector_length_invalid`."""
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_GLOBAL,
        )
        warp = tb.warps[0]
        addr_record = warp.instructions[0].addresses[0]
        # The fixture default is 32 entries; truncate to 7 to
        # exercise the malformed-length rejection path.
        del addr_record.addrs[7:]
        self.assertEqual(len(addr_record.addrs), 7)
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.typed_reason, "lane_vector_length_invalid",
        )

    def test_list_all_with_thirty_two_entries_admits(self) -> None:
        """The fixture default emits a 32-entry `list_all` record
        with 1 real address and 31 sentinels. The validator must
        admit it — this is the canonical conforming shape under
        AC4's 32-entry sentinel format. Uses SHARED (rather than
        GLOBAL) so the lane-vector contract is exercised
        independently of the allocation-id gate."""
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_SHARED,
        )
        warp = tb.warps[0]
        addr_record = warp.instructions[0].addresses[0]
        # Fixture invariant: 32 addrs entries; lane 0 has the
        # real address, lanes 1-31 are sentinels.
        self.assertEqual(len(addr_record.addrs), 32)
        result = validate_threadblock(tb)
        self.assertTrue(
            result.ok,
            msg=(
                f"expected admission, got {result.typed_reason}: "
                f"{result.detail}"
            ),
        )

    def test_list_all_all_sentinels_with_nonzero_and_mask_rejected(
        self,
    ) -> None:
        """Symmetric direction of `lane_mask_inconsistent`. The
        producer's `list_all` branch (`tracer_tool.cu` lines
        2687-2693) writes a real (non-sentinel) address at every
        lane the AND mask gates true. A `list_all` record with
        non-zero AND mask and ALL 32 entries being
        `LANE_INACTIVE_SENTINEL` (-1) cannot be produced by the
        conforming producer. Reject with `lane_mask_inconsistent`."""
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_GLOBAL,
        )
        warp = tb.warps[0]
        # Non-zero AND mask (lane 0 is active+predicated).
        warp.instructions[0].active_mask = 0x1
        warp.instructions[0].predicate_mask = 0x1
        addr_record = warp.instructions[0].addresses[0]
        # Overwrite all 32 slots with the sentinel.
        del addr_record.addrs[:]
        for _ in range(32):
            addr_record.addrs.append(-1)
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.typed_reason, "lane_mask_inconsistent",
        )

    def test_list_all_with_real_addr_under_nonzero_and_mask_admits(
        self,
    ) -> None:
        """Canonical conforming shape: non-zero AND mask plus a
        `list_all` record whose 32 entries include at least one
        real (non-sentinel) address. The fixture default emits
        exactly this shape (1 real address at slot 0, 31 sentinels);
        verify the symmetric-direction check admits. Uses SHARED so
        the lane-mask check is exercised independently of the
        allocation-id gate."""
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_SHARED,
        )
        warp = tb.warps[0]
        # Fixture default: active_mask / predicate_mask non-zero,
        # 32 addrs entries with a real value at slot 0. Confirm
        # the symmetric-direction check admits this shape.
        self.assertNotEqual(
            warp.instructions[0].active_mask
            & warp.instructions[0].predicate_mask,
            0,
        )
        addr_record = warp.instructions[0].addresses[0]
        self.assertEqual(len(addr_record.addrs), 32)
        self.assertNotEqual(addr_record.addrs[0], -1)
        result = validate_threadblock(tb)
        self.assertTrue(
            result.ok,
            msg=(
                f"expected admission of canonical list_all under "
                f"non-zero AND mask, got {result.typed_reason}: "
                f"{result.detail}"
            ),
        )

    def test_address_lane_mask_narrows_one_operand(self) -> None:
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_SHARED,
        )
        instruction = tb.warps[0].instructions[0]
        instruction.active_mask = 0b11
        instruction.predicate_mask = 0b11
        instruction.addresses[0].lane_mask = 0b01
        result = validate_threadblock(tb)
        self.assertTrue(result.ok, msg=f"{result.typed_reason}: {result.detail}")

    def test_address_lane_mask_cannot_widen_instruction_mask(self) -> None:
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_SHARED,
        )
        tb.warps[0].instructions[0].addresses[0].lane_mask = 0b10
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.typed_reason,
            "address_lane_mask_outside_instruction_mask",
        )

    def test_base_delta_with_correct_delta_count_admits(self) -> None:
        """A `base_delta` record with delta count == popcount-1
        admits. The compressed format encodes 32 lane slots via
        `base_address + cumulative deltas` walked in lane-index
        order over active lanes; the validator's reconstruction
        logic places real addresses at active-lane positions and
        sentinels at inactive-lane positions. Uses SHARED so the
        compressed-format reconstruction is exercised
        independently of the allocation-id gate."""
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_SHARED,
        )
        warp = tb.warps[0]
        # 6 active lanes (popcount=6), so 5 deltas is the
        # producer-faithful count.
        warp.instructions[0].active_mask = 0x3F  # lanes 0..5
        warp.instructions[0].predicate_mask = 0x3F
        addr_record = warp.instructions[0].addresses[0]
        addr_record.compression_format = 2  # base_delta
        addr_record.base_address = 0xDEADBEEF
        del addr_record.addrs[:]
        for delta in (4, 4, 4, 4, 4):
            addr_record.addrs.append(delta)
        result = validate_threadblock(tb)
        self.assertTrue(
            result.ok,
            msg=(
                f"expected admission of base_delta record with "
                f"5 deltas under 6-active-lane mask, got "
                f"{result.typed_reason}: {result.detail}"
            ),
        )

    def test_list_all_real_addr_in_inactive_slot_rejected(
        self,
    ) -> None:
        """R18 positional identity. A `list_all` record where
        a non-sentinel slot's lane bit is clear in
        (active_mask & predicate_mask) is malformed; the producer
        only writes real addresses at AND-mask-true lanes. This is
        exactly the malformed shape Codex flagged in the R17
        review (real address in slot whose mask bit is clear)."""
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_GLOBAL,
        )
        warp = tb.warps[0]
        # AND mask = 0x1 (lane 0 only). Move the real address to
        # slot 31 and put the sentinel in slot 0; positional
        # identity now violates at slot 0 (bit set, sentinel) and
        # slot 31 (bit clear, real address).
        addr_record = warp.instructions[0].addresses[0]
        addr_record.addrs[0] = -1
        addr_record.addrs[31] = 0xCAFE0000
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.typed_reason, "lane_position_mismatch",
        )

    def test_list_all_sentinel_in_active_slot_rejected(self) -> None:
        """R18 positional identity (active-side). A `list_all`
        record where a sentinel slot's lane bit is SET in the AND
        mask is malformed; the producer writes a real address at
        every AND-mask-true lane."""
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_GLOBAL,
        )
        warp = tb.warps[0]
        # AND mask = 0x3 (lanes 0 and 1). Slot 0 has a real
        # address, slot 1 is sentinel — slot 1 violates positional
        # identity (bit set, sentinel).
        warp.instructions[0].active_mask = 0x3
        warp.instructions[0].predicate_mask = 0x3
        addr_record = warp.instructions[0].addresses[0]
        # addrs[0] = 0xCAFE0000 (fixture), addrs[1] = -1 (fixture)
        # — leave as-is. Bit 1 set with sentinel in slot 1
        # violates positional identity.
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.typed_reason, "lane_position_mismatch",
        )

    def test_base_stride_contiguous_lanes_admits(self) -> None:
        """R18 reconstruction (base_stride). Active lanes
        [3..7] with constant stride: validator reconstructs to
        sentinel at [0..2, 8..31] and real addresses
        base + 0, base + stride, ..., base + 4*stride at [3..7].
        Uses SHARED so the compressed-format reconstruction is
        exercised independently of the allocation-id gate."""
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_SHARED,
        )
        warp = tb.warps[0]
        # AND mask = 0xF8 (lanes 3..7 contiguous).
        warp.instructions[0].active_mask = 0xF8
        warp.instructions[0].predicate_mask = 0xF8
        addr_record = warp.instructions[0].addresses[0]
        addr_record.compression_format = 1  # base_stride
        addr_record.base_address = 0x10000000
        addr_record.stride = 8
        del addr_record.addrs[:]  # base_stride leaves addrs empty
        result = validate_threadblock(tb)
        self.assertTrue(
            result.ok,
            msg=(
                f"expected admission of base_stride record with "
                f"contiguous lanes [3..7], got "
                f"{result.typed_reason}: {result.detail}"
            ),
        )

    def test_base_stride_non_contiguous_lanes_rejected(self) -> None:
        """R18 reconstruction (base_stride). The producer's
        base_stride_compress only succeeds when active lanes form
        a contiguous run; a base_stride record with non-contiguous
        active lanes is structurally malformed and cannot
        reconstruct."""
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_GLOBAL,
        )
        warp = tb.warps[0]
        # AND mask = 0x5 (lanes 0 and 2 — non-contiguous).
        warp.instructions[0].active_mask = 0x5
        warp.instructions[0].predicate_mask = 0x5
        addr_record = warp.instructions[0].addresses[0]
        addr_record.compression_format = 1  # base_stride
        addr_record.base_address = 0x10000000
        addr_record.stride = 8
        del addr_record.addrs[:]
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.typed_reason, "compressed_format_malformed",
        )

    def test_base_stride_zero_and_mask_with_nonempty_addrs_rejected(
        self,
    ) -> None:
        """R19 hardening (Codex R18 review). Under
        zero AND mask the producer's `base_stride` branch emits
        an empty `addrs` list (the compress loop completes
        without `first_bit1_found`, so no entries are pushed
        and `addrs` stays empty). A compressed-format record
        with `(active_mask & predicate_mask) == 0` AND any
        `addrs` entry is therefore an internal inconsistency
        that the existing `lane_mask_inconsistent` rule catches
        BEFORE the reconstruction-aware check; this test pins
        that diagnostic path so a future re-ordering or rule
        change does not silently drop the case."""
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_GLOBAL,
        )
        warp = tb.warps[0]
        warp.instructions[0].active_mask = 0
        warp.instructions[0].predicate_mask = 0
        addr_record = warp.instructions[0].addresses[0]
        addr_record.compression_format = 1  # base_stride
        addr_record.base_address = 0xCAFE0000
        addr_record.stride = 8
        del addr_record.addrs[:]
        addr_record.addrs.append(0xDEADBEEF)  # non-empty addrs
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.typed_reason, "lane_mask_inconsistent",
        )

    def test_base_stride_single_active_lane_rejected(self) -> None:
        """R18 producer-exactness. The producer's
        base_stride_compress at tracer_tool.cu:2397-2406 sets
        const_stride=false when the first active lane has no
        immediately-following active lane (s==31 or lane s+1
        inactive), falling through to base_delta. So a
        base_stride record with exactly one active lane is
        structurally impossible from the conforming producer;
        the validator rejects it as compressed_format_malformed
        (Codex R18 review hardening)."""
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_GLOBAL,
        )
        warp = tb.warps[0]
        # AND mask = 0x1 (lane 0 only). A conforming producer
        # would emit base_delta here (with 0 deltas), not
        # base_stride.
        addr_record = warp.instructions[0].addresses[0]
        addr_record.compression_format = 1  # base_stride
        addr_record.base_address = 0xCAFE0000
        addr_record.stride = 8
        del addr_record.addrs[:]
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.typed_reason, "compressed_format_malformed",
        )

    def test_base_delta_wrong_delta_count_rejected(self) -> None:
        """R18 reconstruction (base_delta). The producer's
        base_delta_compress writes one delta per subsequent
        active lane after the lowest, so delta count must equal
        popcount(and_mask) - 1 (or 0 when popcount<=1). A
        base_delta record with the wrong delta count is
        structurally malformed."""
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_GLOBAL,
        )
        warp = tb.warps[0]
        # AND mask = 0x7 (lanes 0, 1, 2 active; popcount=3).
        # Producer-faithful delta count is 2; this record has 1.
        warp.instructions[0].active_mask = 0x7
        warp.instructions[0].predicate_mask = 0x7
        addr_record = warp.instructions[0].addresses[0]
        addr_record.compression_format = 2  # base_delta
        addr_record.base_address = 0xDEADBEEF
        del addr_record.addrs[:]
        addr_record.addrs.append(8)  # only 1 delta, expected 2
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.typed_reason, "compressed_format_malformed",
        )

    def test_base_delta_zero_active_lanes_admits(self) -> None:
        """R18 reconstruction (base_delta edge case). Zero
        active lanes: producer's base_delta_compress emits an
        empty addrs list; reconstruction yields all sentinels;
        positional identity holds (every slot is sentinel and
        every bit is clear)."""
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_GLOBAL,
        )
        warp = tb.warps[0]
        warp.instructions[0].active_mask = 0
        warp.instructions[0].predicate_mask = 0
        addr_record = warp.instructions[0].addresses[0]
        addr_record.compression_format = 2  # base_delta
        addr_record.base_address = 0
        del addr_record.addrs[:]  # zero deltas
        result = validate_threadblock(tb)
        self.assertTrue(
            result.ok,
            msg=(
                f"expected admission of zero-active-lanes "
                f"base_delta, got {result.typed_reason}: "
                f"{result.detail}"
            ),
        )

    def test_base_delta_one_active_lane_admits(self) -> None:
        """R18 reconstruction (base_delta edge case). Single
        active lane: producer writes the lane's address into
        base_address with zero deltas; reconstruction places the
        real address at the active lane and sentinels at the
        other 31. Uses SHARED so the compressed-format
        reconstruction is exercised independently of the
        allocation-id gate."""
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_SHARED,
        )
        warp = tb.warps[0]
        # AND mask = 0x4 (lane 2 only).
        warp.instructions[0].active_mask = 0x4
        warp.instructions[0].predicate_mask = 0x4
        addr_record = warp.instructions[0].addresses[0]
        addr_record.compression_format = 2  # base_delta
        addr_record.base_address = 0xCAFE1234
        del addr_record.addrs[:]  # zero deltas (popcount=1)
        result = validate_threadblock(tb)
        self.assertTrue(
            result.ok,
            msg=(
                f"expected admission of one-active-lane "
                f"base_delta, got {result.typed_reason}: "
                f"{result.detail}"
            ),
        )

    def test_unknown_compression_format_rejected(self) -> None:
        """R18: a record with an unknown compression_format
        value (not 0/1/2) cannot be reconstructed; the validator
        surfaces it as `compressed_format_malformed` so a future
        producer that adds a new format without updating the
        validator gets a typed contract failure rather than
        silent admission."""
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_GLOBAL,
        )
        warp = tb.warps[0]
        addr_record = warp.instructions[0].addresses[0]
        addr_record.compression_format = 99  # unknown
        del addr_record.addrs[:]
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.typed_reason, "compressed_format_malformed",
        )

    def test_zero_lane_masks_without_addresses_admits(self) -> None:
        """A non-memory instruction (no outer addresses, no
        memory_op_class) can legitimately have active_mask=0 or
        predicate_mask=0; the lane-mask consistency check ONLY
        fires when an inner addrs list is non-empty."""
        from import_trace_contract import validate_threadblock

        tb = _make_threadblock_with_instruction()
        warp = tb.warps[0]
        warp.instructions[0].active_mask = 0
        warp.instructions[0].predicate_mask = 0
        result = validate_threadblock(tb)
        self.assertTrue(
            result.ok,
            msg=(
                f"expected admission, got {result.typed_reason}: "
                f"{result.detail}"
            ),
        )


class TraceContractValidatorReadPathTest(unittest.TestCase):
    """The validate_threadblock_path entry point handles missing
    files and non-threadblock bytes with typed reasons."""

    def test_missing_file_typed(self) -> None:
        from import_trace_contract import validate_threadblock_path

        result = validate_threadblock_path(
            Path("/nonexistent/threadblock.pb")
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.typed_reason, "trace_unreadable")

    def test_round_trip_serialized_pb_admits(self) -> None:
        """Serialize a positive fixture to disk, read it back through
        validate_threadblock_path, and confirm admission. This guards
        the bytes-on-the-wire path in addition to the in-memory walk.
        """
        import tempfile

        from import_trace_contract import validate_threadblock_path

        tb = _make_threadblock_with_instruction()
        with tempfile.NamedTemporaryFile(
            "wb", suffix=".pb", delete=False
        ) as fh:
            fh.write(tb.SerializeToString())
            tmp_path = Path(fh.name)
        try:
            result = validate_threadblock_path(tmp_path)
            self.assertTrue(
                result.ok,
                msg=(
                    f"expected admission, got {result.typed_reason}: "
                    f"{result.detail}"
                ),
            )
        finally:
            tmp_path.unlink(missing_ok=True)


class TraceContractValidatorTraceDirTest(unittest.TestCase):
    """The validate_trace_dir entry point walks every *.pb under
    <traces_dir>/threadblocks/ and short-circuits on the first
    failure."""

    def test_trace_dir_with_one_bad_threadblock_rejected(self) -> None:
        import tempfile

        from import_trace_contract import validate_trace_dir
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb_good = _make_threadblock_with_instruction()
        tb_bad = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=False,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            traces_dir = Path(tmp_dir)
            tb_root = traces_dir / "threadblocks" / "d0" / "s0" / "k0"
            tb_root.mkdir(parents=True)
            (tb_root / "good.pb").write_bytes(tb_good.SerializeToString())
            (tb_root / "bad.pb").write_bytes(tb_bad.SerializeToString())

            result = validate_trace_dir(traces_dir)
            self.assertFalse(result.ok)
            self.assertEqual(
                result.typed_reason,
                "memory_op_class_missing_on_memory_instruction",
            )

    def test_trace_dir_missing_threadblocks_subtree_typed(self) -> None:
        import tempfile

        from import_trace_contract import validate_trace_dir

        with tempfile.TemporaryDirectory() as tmp_dir:
            traces_dir = Path(tmp_dir)
            result = validate_trace_dir(traces_dir)
            self.assertFalse(result.ok)
            self.assertEqual(result.typed_reason, "trace_unreadable")

    def test_trace_dir_all_pass_admits(self) -> None:
        import tempfile

        from import_trace_contract import validate_trace_dir

        tb_good = _make_threadblock_with_instruction()

        with tempfile.TemporaryDirectory() as tmp_dir:
            traces_dir = Path(tmp_dir)
            tb_root = traces_dir / "threadblocks"
            tb_root.mkdir()
            (tb_root / "good.pb").write_bytes(tb_good.SerializeToString())

            result = validate_trace_dir(traces_dir)
            self.assertTrue(
                result.ok,
                msg=(
                    f"expected admission, got {result.typed_reason}: "
                    f"{result.detail}"
                ),
            )


class TraceContractValidatorRecordPathTest(unittest.TestCase):
    """The detail string carries the offending record path so an
    operator can locate the bad record without re-parsing the
    threadblock.
    """

    def test_detail_carries_warp_and_instruction_index(self) -> None:
        from import_trace_contract import validate_threadblock
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()

        tb = _make_threadblock_with_instruction(
            set_data_width=False,
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_GLOBAL,
        )
        result = validate_threadblock(tb)
        self.assertFalse(result.ok)
        self.assertIn("warp=0", result.detail)
        self.assertIn("inst=0", result.detail)


class TraceContractValidatorAllocationsTest(unittest.TestCase):
    """task11d allocation/range identity checks.

    Builds a one-instruction threadblock with a GLOBAL memory record
    carrying ``addr.allocation_id``, writes an
    ``extra_info/allocations.json`` sidecar with controlled shape,
    and verifies that ``validate_trace_dir`` produces the right
    typed-reason outcome for each of the 5 new allowlist entries
    plus the positive-admission case.
    """

    @staticmethod
    def _build_global_tb_with_allocation_id(
        allocation_id: int,
        lane0_address: int,
    ) -> Any:
        """Build a threadblock with one warp, one GLOBAL memory
        instruction. Lane 0 is active; lane 0 address is
        ``lane0_address``; the address record carries
        ``addr.allocation_id = allocation_id``.
        """
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()
        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_GLOBAL,
        )
        addr = tb.warps[0].instructions[0].addresses[0]
        addr.addrs[0] = lane0_address
        addr.allocation_id = allocation_id
        return tb

    @staticmethod
    def _write_traces_dir(
        tmp_path: Path, tb: Any, allocations_json: str | None
    ) -> Path:
        """Materialize a trace tree at ``tmp_path``: write the
        threadblock under ``threadblocks/`` and (optionally) the
        allocations.json sidecar under ``extra_info/``. Returns
        ``tmp_path``.
        """
        tb_root = tmp_path / "threadblocks"
        tb_root.mkdir()
        (tb_root / "tb.pb").write_bytes(tb.SerializeToString())
        if allocations_json is not None:
            extra_info = tmp_path / "extra_info"
            extra_info.mkdir()
            (extra_info / "allocations.json").write_text(
                allocations_json, encoding="utf-8"
            )
        return tmp_path

    def test_global_record_with_in_range_lane_admits(self) -> None:
        """Positive admission: the producer stamps allocation_id=7;
        the sidecar lists id=7 with base=0xCAFE0000 size=0x10000;
        the lane 0 address 0xCAFE0000 is in
        ``[0xCAFE0000, 0xCAFF0000)``. The validator must admit."""
        import tempfile

        from import_trace_contract import validate_trace_dir

        tb = self._build_global_tb_with_allocation_id(
            allocation_id=7, lane0_address=0xCAFE0000
        )
        sidecar = (
            "[\n"
            '  {"allocation_id": 7, "context_id": 1, '
            '"device_id": 0, "base_address": "0xcafe0000", '
            '"size_bytes": 65536, "alive": true}\n'
            "]\n"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            traces_dir = self._write_traces_dir(
                Path(tmp_dir), tb, sidecar
            )
            result = validate_trace_dir(traces_dir)
            self.assertTrue(
                result.ok,
                msg=(
                    f"expected admission, got {result.typed_reason}: "
                    f"{result.detail}"
                ),
            )

    def test_record_with_alloc_id_but_missing_sidecar_rejected(
        self,
    ) -> None:
        """If a record carries ``allocation_id != 0`` but the trace
        directory has no allocations.json, the walk surfaces
        ``allocations_sidecar_missing`` at the offending record."""
        import tempfile

        from import_trace_contract import validate_trace_dir

        tb = self._build_global_tb_with_allocation_id(
            allocation_id=3, lane0_address=0xDEAD0000
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            traces_dir = self._write_traces_dir(
                Path(tmp_dir), tb, None
            )
            result = validate_trace_dir(traces_dir)
            self.assertFalse(result.ok)
            self.assertEqual(
                result.typed_reason, "allocations_sidecar_missing"
            )

    def test_malformed_sidecar_rejected(self) -> None:
        """An entry missing a required field is rejected with
        ``allocations_sidecar_malformed``."""
        import tempfile

        from import_trace_contract import validate_trace_dir

        tb = self._build_global_tb_with_allocation_id(
            allocation_id=7, lane0_address=0xCAFE0000
        )
        # Missing `size_bytes` field on the only entry.
        sidecar = (
            "[\n"
            '  {"allocation_id": 7, "context_id": 1, '
            '"device_id": 0, "base_address": "0xcafe0000", '
            '"alive": true}\n'
            "]\n"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            traces_dir = self._write_traces_dir(
                Path(tmp_dir), tb, sidecar
            )
            result = validate_trace_dir(traces_dir)
            self.assertFalse(result.ok)
            self.assertEqual(
                result.typed_reason, "allocations_sidecar_malformed"
            )
            self.assertIn("size_bytes", result.detail)

    def test_malformed_sidecar_extra_field_rejected(self) -> None:
        """An entry with an extra field beyond the producer's
        canonical six-field `AllocationRecord` shape is rejected
        with ``allocations_sidecar_malformed``. The producer never
        emits extra fields; an extra field signals either drift or
        a manual edit and must fail closed so the consumer-side
        shape stays exact."""
        import tempfile

        from import_trace_contract import validate_trace_dir

        tb = self._build_global_tb_with_allocation_id(
            allocation_id=7, lane0_address=0xCAFE0000
        )
        # All six required fields present, plus an extra
        # `unexpected_field` that the producer never writes.
        sidecar = (
            "[\n"
            '  {"allocation_id": 7, "context_id": 1, '
            '"device_id": 0, "base_address": "0xcafe0000", '
            '"size_bytes": 65536, "alive": true, '
            '"unexpected_field": "drift"}\n'
            "]\n"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            traces_dir = self._write_traces_dir(
                Path(tmp_dir), tb, sidecar
            )
            result = validate_trace_dir(traces_dir)
            self.assertFalse(result.ok)
            self.assertEqual(
                result.typed_reason, "allocations_sidecar_malformed"
            )
            self.assertIn("unexpected_field", result.detail)

    def test_malformed_sidecar_bad_hex_base_address_rejected(
        self,
    ) -> None:
        """A ``base_address`` that does not parse as ``0x``-prefixed
        hex is rejected with ``allocations_sidecar_malformed``."""
        import tempfile

        from import_trace_contract import validate_trace_dir

        tb = self._build_global_tb_with_allocation_id(
            allocation_id=7, lane0_address=0xCAFE0000
        )
        sidecar = (
            "[\n"
            '  {"allocation_id": 7, "context_id": 1, '
            '"device_id": 0, "base_address": "deadbeef", '
            '"size_bytes": 65536, "alive": true}\n'
            "]\n"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            traces_dir = self._write_traces_dir(
                Path(tmp_dir), tb, sidecar
            )
            result = validate_trace_dir(traces_dir)
            self.assertFalse(result.ok)
            self.assertEqual(
                result.typed_reason, "allocations_sidecar_malformed"
            )
            self.assertIn("base_address", result.detail)

    def test_duplicate_allocation_id_rejected(self) -> None:
        """Two entries with the same ``allocation_id`` are rejected
        with ``allocation_id_duplicate``."""
        import tempfile

        from import_trace_contract import validate_trace_dir

        tb = self._build_global_tb_with_allocation_id(
            allocation_id=7, lane0_address=0xCAFE0000
        )
        sidecar = (
            "[\n"
            '  {"allocation_id": 7, "context_id": 1, '
            '"device_id": 0, "base_address": "0xcafe0000", '
            '"size_bytes": 65536, "alive": true},\n'
            '  {"allocation_id": 7, "context_id": 1, '
            '"device_id": 0, "base_address": "0xcafe0000", '
            '"size_bytes": 65536, "alive": false}\n'
            "]\n"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            traces_dir = self._write_traces_dir(
                Path(tmp_dir), tb, sidecar
            )
            result = validate_trace_dir(traces_dir)
            self.assertFalse(result.ok)
            self.assertEqual(
                result.typed_reason, "allocation_id_duplicate"
            )

    def test_unknown_allocation_id_rejected(self) -> None:
        """A record carrying an ``allocation_id`` not present in the
        sidecar table is rejected with ``allocation_id_unknown``."""
        import tempfile

        from import_trace_contract import validate_trace_dir

        tb = self._build_global_tb_with_allocation_id(
            allocation_id=42, lane0_address=0xCAFE0000
        )
        sidecar = (
            "[\n"
            '  {"allocation_id": 7, "context_id": 1, '
            '"device_id": 0, "base_address": "0xcafe0000", '
            '"size_bytes": 65536, "alive": true}\n'
            "]\n"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            traces_dir = self._write_traces_dir(
                Path(tmp_dir), tb, sidecar
            )
            result = validate_trace_dir(traces_dir)
            self.assertFalse(result.ok)
            self.assertEqual(
                result.typed_reason, "allocation_id_unknown"
            )

    def test_lane_address_outside_range_rejected(self) -> None:
        """A lane address outside ``[base, base + size_bytes)`` is
        rejected with ``lane_address_outside_allocation_range``.
        Producer-side per-lane resolution (task11b) would have
        emitted a ``cross_allocation_lane_span`` typed exclusion
        instead of stamping an allocation_id; the validator catches
        the case where a tracer regression breaks that contract."""
        import tempfile

        from import_trace_contract import validate_trace_dir

        # base=0xCAFE0000, size=0x10000 => valid range
        # [0xCAFE0000, 0xCAFF0000). Lane 0 address 0xCAFFFFFF is
        # outside this range.
        tb = self._build_global_tb_with_allocation_id(
            allocation_id=7, lane0_address=0xCAFFFFFF
        )
        sidecar = (
            "[\n"
            '  {"allocation_id": 7, "context_id": 1, '
            '"device_id": 0, "base_address": "0xcafe0000", '
            '"size_bytes": 65536, "alive": true}\n'
            "]\n"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            traces_dir = self._write_traces_dir(
                Path(tmp_dir), tb, sidecar
            )
            result = validate_trace_dir(traces_dir)
            self.assertFalse(result.ok)
            self.assertEqual(
                result.typed_reason,
                "lane_address_outside_allocation_range",
            )

    def test_global_record_with_active_lanes_and_no_allocation_id_rejected(
        self,
    ) -> None:
        """A GLOBAL record with ``(active_mask & predicate_mask) != 0``
        and absent ``addr.allocation_id`` is rejected with
        ``allocation_id_missing``. The producer's per-lane resolution
        gate either stamps a non-zero id or routes through one of the
        typed-exclusion sidecars; a wire record with active lanes and
        no allocation_id is a producer-side regression that must fail
        offline validation rather than silently passing as a regular
        memory record."""
        import tempfile

        from import_trace_contract import validate_trace_dir

        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()
        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_GLOBAL,
        )
        # active_mask & predicate_mask = 0x1 (lane 0 active), but
        # addr.allocation_id is absent (proto3 optional default).
        with tempfile.TemporaryDirectory() as tmp_dir:
            traces_dir = self._write_traces_dir(
                Path(tmp_dir), tb, None
            )
            result = validate_trace_dir(traces_dir)
            self.assertFalse(result.ok)
            self.assertEqual(
                result.typed_reason, "allocation_id_missing"
            )
            self.assertIn("absent", result.detail)

    def test_global_record_with_active_lanes_and_zero_allocation_id_rejected(
        self,
    ) -> None:
        """A GLOBAL record where ``addr.allocation_id`` is explicitly
        set to 0 (HasField=True, value=0) with active lanes is also
        rejected with ``allocation_id_missing``. proto3 explicit-zero
        is observationally distinct from absent but must produce the
        same typed failure here because both shapes carry no
        allocation identity for the active lanes."""
        import tempfile

        from import_trace_contract import validate_trace_dir

        tb = self._build_global_tb_with_allocation_id(
            allocation_id=0, lane0_address=0xCAFE0000
        )
        # The fixture sets HasField=True via the explicit assignment,
        # but the value is 0; verify here that HasField is indeed True
        # (defensive — proto3 explicit-presence semantics).
        addr = tb.warps[0].instructions[0].addresses[0]
        self.assertTrue(addr.HasField("allocation_id"))
        self.assertEqual(addr.allocation_id, 0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            traces_dir = self._write_traces_dir(
                Path(tmp_dir), tb, None
            )
            result = validate_trace_dir(traces_dir)
            self.assertFalse(result.ok)
            self.assertEqual(
                result.typed_reason, "allocation_id_missing"
            )
            self.assertIn("zero", result.detail)

    def test_global_atomic_missing_allocation_id_rejected(self) -> None:
        """Address-carrying contract, operand-adjudication negative test (a): a global atomic
        (memory_op_class=ATOMIC, address_space=GLOBAL) with active lanes and no
        allocation_id is REJECTED with ``allocation_id_missing``. The widened
        identity gate keys on the per-address space, closing the pre-transition
        fail-open on ATOMIC records."""
        import tempfile

        from import_trace_contract import validate_trace_dir

        _tb_pb2, instruction_pb2, address_pb2 = _import_pb()
        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_ATOMIC,
            address_space_value=address_pb2.MEMORY_SPACE_GLOBAL,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            traces_dir = self._write_traces_dir(Path(tmp_dir), tb, None)
            result = validate_trace_dir(traces_dir)
            self.assertFalse(result.ok)
            self.assertEqual(result.typed_reason, "allocation_id_missing")

    def test_shared_atomic_admits_without_allocation_id(self) -> None:
        """Address-carrying contract, operand-adjudication negative test (b): a shared atomic
        (ATOMS: memory_op_class=ATOMIC, address_space=SHARED) is EXEMPT from the
        global allocation-id rule — it admits with no allocation_id and is never
        forced through global identity (a shared address legitimately resolves
        to no global allocation)."""
        import tempfile

        from import_trace_contract import validate_trace_dir

        _tb_pb2, instruction_pb2, address_pb2 = _import_pb()
        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_ATOMIC,
            set_memory_op_access_kind=True,
            memory_op_access_kind_value=(
                instruction_pb2.MEMORY_OP_ACCESS_KIND_ATOMIC
            ),
            address_space_value=address_pb2.MEMORY_SPACE_SHARED,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            traces_dir = self._write_traces_dir(Path(tmp_dir), tb, None)
            result = validate_trace_dir(traces_dir)
            self.assertTrue(result.ok, result.detail)

    def test_ldgsts_global_source_missing_allocation_id_rejected(self) -> None:
        """Address-carrying contract, ASYNC_COPY leg: the LDGSTS global-source
        record is a GLOBAL-space record (memory_op_class=ASYNC_COPY,
        address_space=GLOBAL). With active lanes and no allocation_id it is
        REJECTED with allocation_id_missing, exactly like a global atomic — the
        per-address identity gate does not care about the op class."""
        import tempfile

        from import_trace_contract import validate_trace_dir

        _tb_pb2, instruction_pb2, address_pb2 = _import_pb()
        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_ASYNC_COPY,
            address_space_value=address_pb2.MEMORY_SPACE_GLOBAL,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            traces_dir = self._write_traces_dir(Path(tmp_dir), tb, None)
            result = validate_trace_dir(traces_dir)
            self.assertFalse(result.ok)
            self.assertEqual(result.typed_reason, "allocation_id_missing")

    def test_global_record_with_zero_and_mask_admits_without_allocation_id(
        self,
    ) -> None:
        """A predicated-off GLOBAL memory record (active_mask &
        predicate_mask == 0) admits without an allocation_id because
        no lane address requires identity. The fixture's all-sentinel
        addrs vector matches the zero AND-mask, so the existing
        lane-position invariants still hold; only the active-lane
        identity gate is relaxed."""
        import tempfile

        from import_trace_contract import validate_trace_dir

        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()
        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_GLOBAL,
        )
        inst = tb.warps[0].instructions[0]
        # Force AND-mask zero by clearing predicate_mask. The fixture
        # set active_mask=0x1 and predicate_mask=0x1; clearing one
        # bit makes the combined mask zero. With AND-mask zero, the
        # `lane_mask_inconsistent` check requires all addrs to be
        # sentinel — overwrite lane 0 to LANE_INACTIVE_SENTINEL.
        inst.predicate_mask = 0x0
        inst.addresses[0].addrs[0] = -1  # LANE_INACTIVE_SENTINEL

        with tempfile.TemporaryDirectory() as tmp_dir:
            traces_dir = self._write_traces_dir(
                Path(tmp_dir), tb, None
            )
            result = validate_trace_dir(traces_dir)
            self.assertTrue(
                result.ok,
                msg=(
                    f"expected admission, got {result.typed_reason}: "
                    f"{result.detail}"
                ),
            )


class TraceContractValidatorSinglePbSidecarInferenceTest(
    unittest.TestCase,
):
    """``validate_threadblock_path`` auto-loads the allocations
    sidecar when its argument is a ``.pb`` inside a conventional
    trace layout. This is the single-file mirror of
    ``validate_trace_dir`` for traces whose records carry
    ``allocation_id != 0``.
    """

    @staticmethod
    def _build_global_tb_with_allocation_id(
        allocation_id: int,
        lane0_address: int,
    ) -> Any:
        _tb_pb2, instruction_pb2, _addr_pb2 = _import_pb()
        tb = _make_threadblock_with_instruction(
            add_address=True,
            set_memory_op_class=True,
            memory_op_class_value=instruction_pb2.MEMORY_OP_CLASS_GLOBAL,
        )
        addr = tb.warps[0].instructions[0].addresses[0]
        addr.addrs[0] = lane0_address
        addr.allocation_id = allocation_id
        return tb

    @staticmethod
    def _materialize_conventional_layout(
        traces_dir: Path,
        tb: Any,
        allocations_json: str | None,
    ) -> Path:
        """Write ``tb`` to
        ``traces_dir/threadblocks/d0/s0/k0/0_0_0.pb`` (the full
        conventional layout from the docstring) and optionally write
        ``allocations_json`` to
        ``traces_dir/extra_info/allocations.json``. Returns the
        full ``.pb`` path so callers can pass it to
        ``validate_threadblock_path``.
        """
        tb_dir = (
            traces_dir / "threadblocks" / "d0" / "s0" / "k0"
        )
        tb_dir.mkdir(parents=True)
        pb_path = tb_dir / "0_0_0.pb"
        pb_path.write_bytes(tb.SerializeToString())
        if allocations_json is not None:
            extra = traces_dir / "extra_info"
            extra.mkdir()
            (extra / "allocations.json").write_text(
                allocations_json, encoding="utf-8"
            )
        return pb_path

    def test_conventional_layout_loads_sidecar_admits(self) -> None:
        """A ``.pb`` inside a conventional trace layout with a
        well-formed sidecar admits even when the caller does not pass
        ``allocations`` — the validator infers the trace root and
        loads the sidecar."""
        import tempfile

        from import_trace_contract import validate_threadblock_path

        tb = self._build_global_tb_with_allocation_id(
            allocation_id=7, lane0_address=0xCAFE0000
        )
        sidecar = (
            "[\n"
            '  {"allocation_id": 7, "context_id": 1, '
            '"device_id": 0, "base_address": "0xcafe0000", '
            '"size_bytes": 65536, "alive": true}\n'
            "]\n"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            pb_path = self._materialize_conventional_layout(
                Path(tmp_dir), tb, sidecar
            )
            result = validate_threadblock_path(pb_path)
            self.assertTrue(
                result.ok,
                msg=(
                    f"expected admission, got {result.typed_reason}: "
                    f"{result.detail}"
                ),
            )

    def test_conventional_layout_missing_sidecar_rejects(self) -> None:
        """A ``.pb`` inside a conventional trace layout with a
        record carrying ``allocation_id != 0`` and NO sidecar is
        rejected with ``allocations_sidecar_missing`` (the existing
        fail-closed contract is preserved when inference finds no
        sidecar)."""
        import tempfile

        from import_trace_contract import validate_threadblock_path

        tb = self._build_global_tb_with_allocation_id(
            allocation_id=3, lane0_address=0xDEAD0000
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            pb_path = self._materialize_conventional_layout(
                Path(tmp_dir), tb, None
            )
            result = validate_threadblock_path(pb_path)
            self.assertFalse(result.ok)
            self.assertEqual(
                result.typed_reason, "allocations_sidecar_missing"
            )

    def test_conventional_layout_malformed_sidecar_surfaces_typed_error(
        self,
    ) -> None:
        """A ``.pb`` inside a conventional trace layout with a
        malformed sidecar surfaces the typed error (mirrors
        ``validate_trace_dir``'s behavior)."""
        import tempfile

        from import_trace_contract import validate_threadblock_path

        tb = self._build_global_tb_with_allocation_id(
            allocation_id=7, lane0_address=0xCAFE0000
        )
        # Missing required field `size_bytes`.
        sidecar = (
            "[\n"
            '  {"allocation_id": 7, "context_id": 1, '
            '"device_id": 0, "base_address": "0xcafe0000", '
            '"alive": true}\n'
            "]\n"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            pb_path = self._materialize_conventional_layout(
                Path(tmp_dir), tb, sidecar
            )
            result = validate_threadblock_path(pb_path)
            self.assertFalse(result.ok)
            self.assertEqual(
                result.typed_reason, "allocations_sidecar_malformed"
            )

    def test_pb_outside_conventional_layout_does_not_auto_load(
        self,
    ) -> None:
        """A ``.pb`` written to a flat directory (no ``threadblocks``
        ancestor) does NOT trigger inference; records with
        ``allocation_id != 0`` still fail with
        ``allocations_sidecar_missing``."""
        import tempfile

        from import_trace_contract import validate_threadblock_path

        tb = self._build_global_tb_with_allocation_id(
            allocation_id=5, lane0_address=0xBEEF0000
        )
        with tempfile.NamedTemporaryFile(
            "wb", suffix=".pb", delete=False
        ) as fh:
            fh.write(tb.SerializeToString())
            flat_path = Path(fh.name)
        try:
            result = validate_threadblock_path(flat_path)
            self.assertFalse(result.ok)
            self.assertEqual(
                result.typed_reason, "allocations_sidecar_missing"
            )
        finally:
            flat_path.unlink(missing_ok=True)

    def test_explicit_allocations_argument_short_circuits_inference(
        self,
    ) -> None:
        """When the caller passes an explicit ``allocations`` table,
        the validator uses it directly and skips the inference path,
        even when the ``.pb`` sits inside a conventional layout with
        a different sidecar on disk."""
        import tempfile

        from import_trace_contract import (
            AllocationEntry,
            validate_threadblock_path,
        )

        tb = self._build_global_tb_with_allocation_id(
            allocation_id=7, lane0_address=0xCAFE0000
        )
        # On-disk sidecar carries a DIFFERENT id (11) — would reject
        # via allocation_id_unknown if auto-loaded.
        on_disk_sidecar = (
            "[\n"
            '  {"allocation_id": 11, "context_id": 1, '
            '"device_id": 0, "base_address": "0x10000000", '
            '"size_bytes": 4096, "alive": true}\n'
            "]\n"
        )
        explicit_table = {
            7: AllocationEntry(
                allocation_id=7,
                context_id=1,
                device_id=0,
                base_address=0xCAFE0000,
                size_bytes=0x10000,
                alive=True,
            ),
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            pb_path = self._materialize_conventional_layout(
                Path(tmp_dir), tb, on_disk_sidecar
            )
            result = validate_threadblock_path(
                pb_path, allocations=explicit_table
            )
            self.assertTrue(
                result.ok,
                msg=(
                    f"expected admission via explicit table, got "
                    f"{result.typed_reason}: {result.detail}"
                ),
            )


if __name__ == "__main__":
    unittest.main()
