"""Effective-space admission tests for the sector-stream generator
(address-carrying contract). A GLOBAL/GENERIC-space record is L2 traffic and
admits regardless of its op class (global atomic, LDGSTS global source); a
SHARED/LOCAL-space record is typed-excluded; absent address_space falls back to
the instruction-level op-class check."""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import generator  # noqa: E402

# MemorySpace enum values (address.proto).
_GLOBAL, _SHARED, _LOCAL, _GENERIC = 1, 2, 3, 4


def _call(memory_op_class: str, address_space: int | None):
    # One active lane (lane 0) carrying a real global-looking address; the rest
    # sentinel. data_width 4 is admissible.
    lane_addresses = [0x7000_0000] + [-1] * 31
    return generator.compute_sector_stream(
        lane_addresses=lane_addresses,
        active_mask=0x1,
        predicate_mask=0x1,
        data_width=4,
        memory_op_class=memory_op_class,
        address_space=address_space,
    )


def test_global_space_atomic_admits():
    # op_class ATOMIC but address_space GLOBAL -> L2 traffic -> admits.
    out = _call("ATOMIC", _GLOBAL)
    assert out["result_kind"] != "typed_exclusion", out


def test_ldgsts_global_source_admits():
    out = _call("ASYNC_COPY", _GLOBAL)
    assert out["result_kind"] != "typed_exclusion", out


def test_shared_space_atomic_is_typed_excluded():
    # A shared atomic (ATOMS): address_space SHARED -> not L2 traffic.
    out = _call("ATOMIC", _SHARED)
    assert out["result_kind"] == "typed_exclusion"
    assert out["typed_reason"] == "sector_stream_class_unsupported"


def test_local_space_is_typed_excluded():
    out = _call("ATOMIC", _LOCAL)
    assert out["result_kind"] == "typed_exclusion"


def test_legacy_global_no_address_space_admits():
    # Absent address_space -> fall back to op-class membership (GLOBAL admits).
    out = _call("GLOBAL", None)
    assert out["result_kind"] != "typed_exclusion", out


def test_legacy_atomic_no_address_space_is_excluded():
    # Absent address_space + ATOMIC op class -> legacy typed exclusion.
    out = _call("ATOMIC", None)
    assert out["result_kind"] == "typed_exclusion"
