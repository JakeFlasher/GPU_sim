"""Structural lockstep: the dynamic_trace.pb ``InstructionCacheOp``
proto enum mirrors the JSON schema's ``instruction_cache_op`` enum
at ``tools/cache_predictor/cache_op_map.schema.json``.

Per BL-20260507-arch-token-multi-surface-lockstep: when a discrete
dispatch vocabulary is mirrored across two surfaces (in this case
the producer-side proto enum + the consumer-side JSON schema enum),
a structural cross-surface parity test programmatically extracts
the token set from each surface and asserts equality. The test
fires immediately when either surface is widened without the other,
naming both sets in the diagnostic.

Producer-side (proto) tokens carry an ``INSTRUCTION_CACHE_OP_``
prefix to comply with proto3 enum-naming convention (each enum
value must be unique across the file scope). The consumer-side
(JSON schema) tokens drop the prefix because the schema's enum
field name is already ``instruction_cache_op``. The lockstep test
normalizes both surfaces to the 12-token prefix-free vocabulary
before comparing.

The proto's ``UNSPECIFIED`` zero value is the proto3 default and
does NOT appear in the JSON schema enum (the JSON manifest's
producer never emits UNSPECIFIED — the classifier always resolves
to one of the 12 named tokens or fail-closes with a typed_unavailable
status). The lockstep parity strips UNSPECIFIED from the proto side
before comparing.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PB_GENERATED = _REPO_ROOT / "workloads" / "simbench_bw" / "_pb_generated"
_SCHEMA_PATH = (
    _REPO_ROOT / "tools" / "cache_predictor" / "cache_op_map.schema.json"
)

_INSTRUCTION_CACHE_OP_PREFIX = "INSTRUCTION_CACHE_OP_"
_UNSPECIFIED_TOKEN = "INSTRUCTION_CACHE_OP_UNSPECIFIED"


def _load_schema_enum() -> list[str]:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    entry_props = (
        schema["properties"]["per_pc_instruction_cache_op"]
              ["items"]["properties"]
    )
    return list(entry_props["instruction_cache_op"]["enum"])


def _load_proto_enum(
    case: unittest.TestCase,
) -> dict[int, str]:
    """Return the enum from the tracked binding authority.

    Missing tracked bytes are repository corruption; a missing runtime is a
    test prerequisite failure. Neither condition skips lockstep enforcement.
    """
    pb_path = _PB_GENERATED / "instruction_pb2.py"
    if not pb_path.is_file() or pb_path.is_symlink():
        case.fail(
            "tracked protobuf authority is missing; restore byte-exact "
            "committed content with `git restore --source=HEAD --worktree -- "
            "workloads/simbench_bw/_pb_generated/instruction_pb2.py`"
        )
    try:
        import google.protobuf  # noqa: F401
    except ImportError as exc:
        case.fail(
            "protobuf runtime prerequisite is missing; install the exact "
            f"requirements.txt pin: {exc}"
        )
    if str(_PB_GENERATED) not in sys.path:
        sys.path.insert(0, str(_PB_GENERATED))
    import instruction_pb2 as ipb  # type: ignore[import-untyped]
    enum_desc = ipb.InstructionCacheOp.DESCRIPTOR
    return {v.number: v.name for v in enum_desc.values}


class ProtoVsSchemaEnumLockstepTest(unittest.TestCase):
    """Programmatic structural lockstep test."""

    def test_proto_enum_token_set_mirrors_schema_enum(self) -> None:
        proto_enum = _load_proto_enum(self)
        # Strip the proto-side prefix and the UNSPECIFIED zero
        # value before comparing.
        proto_tokens = {
            name[len(_INSTRUCTION_CACHE_OP_PREFIX):]
            for name in proto_enum.values()
            if name != _UNSPECIFIED_TOKEN
        }
        schema_tokens = set(_load_schema_enum())
        self.assertEqual(
            proto_tokens, schema_tokens,
            f"proto / schema enum lockstep drift: "
            f"proto_only={proto_tokens - schema_tokens}; "
            f"schema_only={schema_tokens - proto_tokens}",
        )

    def test_proto_unspecified_is_zero_value(self) -> None:
        proto_enum = _load_proto_enum(self)
        self.assertEqual(proto_enum[0], _UNSPECIFIED_TOKEN)

    def test_proto_enum_has_exactly_thirteen_values(self) -> None:
        # 12 named tokens + UNSPECIFIED zero value.
        proto_enum = _load_proto_enum(self)
        self.assertEqual(len(proto_enum), 13)

    def test_proto_instruction_message_carries_instruction_cache_op_field(
        self,
    ) -> None:
        _load_proto_enum(self)
        import instruction_pb2 as ipb  # type: ignore[import-untyped]
        inst = ipb.instruction()
        field_descriptor = inst.DESCRIPTOR.fields_by_name.get(
            "instruction_cache_op"
        )
        self.assertIsNotNone(
            field_descriptor,
            "instruction message must carry the optional "
            "instruction_cache_op field added by the proto sidecar "
            "proto widening (tag 13).",
        )
        # The field is proto3 optional => it lives in a oneof
        # synthetic group for has-presence support; its tag is 13.
        self.assertEqual(field_descriptor.number, 13)

    def test_proto_instruction_cache_op_field_round_trips_on_wire(
        self,
    ) -> None:
        _load_proto_enum(self)
        import instruction_pb2 as ipb  # type: ignore[import-untyped]
        inst = ipb.instruction()
        inst.instruction_cache_op = (
            ipb.INSTRUCTION_CACHE_OP_LDG_E_CA
        )
        self.assertTrue(inst.HasField("instruction_cache_op"))
        wire = inst.SerializeToString()
        roundtrip = ipb.instruction()
        roundtrip.ParseFromString(wire)
        self.assertTrue(roundtrip.HasField("instruction_cache_op"))
        self.assertEqual(
            roundtrip.instruction_cache_op,
            ipb.INSTRUCTION_CACHE_OP_LDG_E_CA,
        )

    def test_proto_legacy_instruction_without_field_round_trips(
        self,
    ) -> None:
        # Wire-compat with legacy traces: an instruction emitted
        # before this round's proto widening had no
        # instruction_cache_op field. Parsing it must not raise;
        # HasField returns false; the field's value is
        # UNSPECIFIED on the consumer side.
        _load_proto_enum(self)
        import instruction_pb2 as ipb  # type: ignore[import-untyped]
        legacy = ipb.instruction()
        legacy.pc = 0x10
        wire = legacy.SerializeToString()
        roundtrip = ipb.instruction()
        roundtrip.ParseFromString(wire)
        self.assertFalse(
            roundtrip.HasField("instruction_cache_op"),
            "legacy wire (no instruction_cache_op field) must "
            "parse with HasField false on the consumer side; "
            "the L1 predictor head treats absent as "
            "predictor_l1_sass_provenance_missing.",
        )
        self.assertEqual(
            roundtrip.instruction_cache_op,
            ipb.INSTRUCTION_CACHE_OP_UNSPECIFIED,
        )


if __name__ == "__main__":
    unittest.main()
