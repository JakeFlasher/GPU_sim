"""Static launch occupancy-limit calculator (GPU-free, fail-closed).

Computes the four per-SM resident-block limits that NCU reports as the
``launch__occupancy_limit_*`` metrics, from a launch configuration plus the
static per-thread resource usage and the shared arch-keyed device constants.
The arithmetic mirrors the CUDA occupancy calculator's documented Compute
Capability 8.6 rounding rules; nothing here reads the GPU at run time.

Each of the four limits is expressed in BLOCKS resident per SM imposed by one
resource:

- ``launch__occupancy_limit_warps``: the warp/SM ceiling expressed as a block
  cap. ``warps_per_block = ceil(block_dim / warp_size)``;
  ``limit = max_warps_per_sm // warps_per_block``.
- ``launch__occupancy_limit_blocks``: the hardware resident-block cap,
  ``max_blocks_per_sm`` (independent of the launch's resource footprint).
- ``launch__occupancy_limit_registers``: the register-file ceiling. When the
  static per-thread register count is unknown (``None``/0), the register limit
  is typed-unavailable. Otherwise the per-warp register block is
  ``round_up(registers_per_thread * warp_size, register_allocation_unit)``,
  the per-block register footprint is ``regs_per_warp * warps_per_block``, and
  the block limit is ``registers_per_sm // regs_per_block``.
- ``launch__occupancy_limit_shared_mem``: the shared-memory ceiling. NCU
  computes this against the launch's CONFIGURED shared-memory carveout
  (``launch__shared_mem_config_size``) and counts the driver-reserved per-block
  shared memory, so it is exact only when that carveout is supplied
  (``shared_mem_carveout_bytes``): ``shared_per_block = round_up(static +
  dynamic + reserved_shared_memory_per_block_bytes, shared_mem_allocation_unit_bytes)``
  and ``limit = carveout // shared_per_block``. The configured carveout is a
  runtime/driver decision NOT in the pure-static input set, so when it is not
  supplied this limit is ``typed_unavailable`` (reason
  ``requires_runtime_shared_mem_carveout``) rather than a fabricated
  max-carveout value that disagrees with NCU.

Every limit is returned as ``{"value": int|None, "status":
"available"|"typed_unavailable", "reason": str|None}``. The calculator is
FAIL-CLOSED: when a device constant the limit needs is deferred / unavailable
for the architecture (e.g. all sm_120 constants), or when a required input is
missing (``registers_per_thread is None`` for the register limit), the affected
limit is emitted as ``typed_unavailable`` with a ``missing_input:<name>``
reason rather than a fabricated number. Limits that do not depend on the
missing constant still resolve.

The device-constant resolution goes through the fail-closed
``DeviceConstantTable``; a missing constant can never become a number here.

Public surface:

- ``LAUNCH_OCCUPANCY_LIMIT_METRICS`` tuple (the four NCU metric names).
- ``compute_occupancy_limits(...) -> dict[str, dict]`` (the four limits).
- ``OccupancyLimit`` dataclass for one resolved limit.
- ``OccupancyInputError`` raised on a malformed launch configuration (not on a
  deferred constant, which is reported as ``typed_unavailable`` instead).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from device_constants import (
    DeviceConstantError,
    DeviceConstantTable,
    load_device_constants,
)

# The four NCU launch occupancy-limit metric names this calculator produces.
# Names match the metric registry's ``ncu_metric`` keys exactly.
LAUNCH_OCCUPANCY_LIMIT_METRICS: tuple[str, ...] = (
    "launch__occupancy_limit_warps",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
)

# Status tokens for a resolved limit.
_STATUS_AVAILABLE = "available"
_STATUS_UNAVAILABLE = "typed_unavailable"


class OccupancyInputError(ValueError):
    """Raised when the launch configuration itself is malformed.

    A deferred device constant or an absent ``registers_per_thread`` is NOT an
    input error: those are reported per-metric as ``typed_unavailable``. This
    is reserved for a structurally invalid launch (e.g. ``block_dim <= 0``).
    """


@dataclass(frozen=True)
class OccupancyLimit:
    """One resolved occupancy limit (blocks resident per SM by one resource)."""

    value: int | None
    status: str
    reason: str | None

    @classmethod
    def available(cls, value: int) -> "OccupancyLimit":
        return cls(value=int(value), status=_STATUS_AVAILABLE, reason=None)

    @classmethod
    def unavailable(cls, reason: str) -> "OccupancyLimit":
        return cls(value=None, status=_STATUS_UNAVAILABLE, reason=reason)

    def as_dict(self) -> dict:
        return asdict(self)


def _round_up(value: int, unit: int) -> int:
    """Round ``value`` up to the nearest multiple of ``unit`` (unit > 0)."""
    return ((value + unit - 1) // unit) * unit


def _const(table: DeviceConstantTable, arch: str, name: str) -> int | None:
    """Resolve a device constant as an int, or ``None`` if it is unavailable.

    Wraps the fail-closed loader: a deferred (null) constant or an unknown
    arch/name yields ``None`` so the caller can emit a typed ``missing_input``
    limit instead of raising.
    """
    try:
        return int(table.get(arch, name))
    except DeviceConstantError:
        return None


def compute_occupancy_limits(
    arch: str,
    block_dim: int,
    registers_per_thread: int | None,
    static_shared_bytes: int,
    dynamic_shared_bytes: int,
    table: DeviceConstantTable | None = None,
    shared_mem_carveout_bytes: int | None = None,
) -> dict[str, dict]:
    """Compute the four launch occupancy-limit metrics for one launch config.

    Args:
        arch: architecture key into the device-constant table (e.g. ``sm_86``).
        block_dim: threads per block (must be a positive integer).
        registers_per_thread: static registers per thread from
            ``cuobjdump -res-usage``; ``None`` (or 0) means register usage is
            unknown / not binding, which makes the register limit
            ``typed_unavailable`` rather than guessing a number.
        static_shared_bytes: statically declared shared memory per block.
        dynamic_shared_bytes: dynamically requested shared memory per block.
        table: optional pre-loaded device-constant table; defaults to the
            committed table.

    Returns:
        A dict keyed by the four ``launch__occupancy_limit_*`` metric names,
        each mapping to ``{"value", "status", "reason"}``.

    Raises:
        OccupancyInputError: when the launch configuration is structurally
            invalid (non-positive ``block_dim`` or negative shared bytes).
    """
    if not isinstance(block_dim, int) or isinstance(block_dim, bool):
        raise OccupancyInputError(f"block_dim must be an int, got {block_dim!r}")
    if block_dim <= 0:
        raise OccupancyInputError(f"block_dim must be positive, got {block_dim}")
    if registers_per_thread is not None:
        if not isinstance(registers_per_thread, int) or isinstance(
            registers_per_thread, bool
        ):
            raise OccupancyInputError(
                f"registers_per_thread must be int or None, got "
                f"{registers_per_thread!r}"
            )
        if registers_per_thread < 0:
            raise OccupancyInputError(
                f"registers_per_thread must be >= 0, got {registers_per_thread}"
            )
    for nm, val in (
        ("static_shared_bytes", static_shared_bytes),
        ("dynamic_shared_bytes", dynamic_shared_bytes),
    ):
        if not isinstance(val, int) or isinstance(val, bool):
            raise OccupancyInputError(f"{nm} must be an int, got {val!r}")
        if val < 0:
            raise OccupancyInputError(f"{nm} must be >= 0, got {val}")
    if shared_mem_carveout_bytes is not None:
        if not isinstance(shared_mem_carveout_bytes, int) or isinstance(
            shared_mem_carveout_bytes, bool
        ):
            raise OccupancyInputError(
                f"shared_mem_carveout_bytes must be int or None, got "
                f"{shared_mem_carveout_bytes!r}"
            )
        if shared_mem_carveout_bytes <= 0:
            raise OccupancyInputError(
                f"shared_mem_carveout_bytes must be positive, got "
                f"{shared_mem_carveout_bytes}"
            )

    if table is None:
        table = load_device_constants()

    # Shared resources used by more than one limit.
    warp_size = _const(table, arch, "warp_size")
    max_warps_per_sm = _const(table, arch, "max_warps_per_sm")
    max_blocks_per_sm = _const(table, arch, "max_blocks_per_sm")

    warps_per_block: int | None = None
    if warp_size is not None:
        warps_per_block = math.ceil(block_dim / warp_size)

    # ---- launch__occupancy_limit_warps ---------------------------------
    # The warp/SM cap expressed as a block ceiling: how many blocks the
    # resident-warp budget admits. Needs warp_size + max_warps_per_sm.
    if warps_per_block is None:
        warp_limit = OccupancyLimit.unavailable("missing_input:warp_size")
    elif max_warps_per_sm is None:
        warp_limit = OccupancyLimit.unavailable("missing_input:max_warps_per_sm")
    else:
        warp_limit = OccupancyLimit.available(max_warps_per_sm // warps_per_block)

    # ---- launch__occupancy_limit_blocks --------------------------------
    # The hardware resident-block cap, independent of resource footprint.
    if max_blocks_per_sm is None:
        block_limit = OccupancyLimit.unavailable("missing_input:max_blocks_per_sm")
    else:
        block_limit = OccupancyLimit.available(max_blocks_per_sm)

    # ---- launch__occupancy_limit_registers -----------------------------
    # The register-file ceiling expressed as a block cap. When the static
    # register count is unknown, registers are not binding and we DO NOT
    # fabricate a number: registers_per_thread is a required input here.
    registers_per_sm = _const(table, arch, "registers_per_sm")
    register_allocation_unit = _const(table, arch, "register_allocation_unit")
    if registers_per_thread is None or registers_per_thread == 0:
        register_limit = OccupancyLimit.unavailable(
            "missing_input:registers_per_thread"
        )
    elif warps_per_block is None:
        register_limit = OccupancyLimit.unavailable("missing_input:warp_size")
    elif registers_per_sm is None:
        register_limit = OccupancyLimit.unavailable("missing_input:registers_per_sm")
    elif register_allocation_unit is None:
        register_limit = OccupancyLimit.unavailable(
            "missing_input:register_allocation_unit"
        )
    else:
        # Per-warp register block, rounded up to the allocation unit.
        regs_per_warp = _round_up(
            registers_per_thread * warp_size, register_allocation_unit
        )
        regs_per_block = regs_per_warp * warps_per_block
        register_limit = OccupancyLimit.available(registers_per_sm // regs_per_block)

    # ---- launch__occupancy_limit_shared_mem ----------------------------
    # NCU computes the shared-memory block limit against the launch's CONFIGURED
    # shared-memory carveout (launch__shared_mem_config_size, a runtime/driver
    # decision) and counts the driver-reserved per-block shared memory. The
    # pure-static inputs do NOT determine the configured carveout, so without it
    # this limit is typed_unavailable rather than a fabricated max-carveout value
    # that disagrees with NCU (e.g. 16 vs NCU's 8 on a zero-shared kernel). When
    # the carveout IS supplied the limit is exact:
    #   shared_per_block = round_up(static + dynamic + reserved, alloc_unit)
    #   limit            = carveout // shared_per_block
    shared_mem_allocation_unit_bytes = _const(
        table, arch, "shared_mem_allocation_unit_bytes"
    )
    reserved_shared = _const(table, arch, "reserved_shared_memory_per_block_bytes")
    if shared_mem_carveout_bytes is None:
        shared_limit = OccupancyLimit.unavailable(
            "requires_runtime_shared_mem_carveout"
        )
    elif shared_mem_allocation_unit_bytes is None:
        shared_limit = OccupancyLimit.unavailable(
            "missing_input:shared_mem_allocation_unit_bytes"
        )
    elif reserved_shared is None:
        shared_limit = OccupancyLimit.unavailable(
            "missing_input:reserved_shared_memory_per_block_bytes"
        )
    else:
        shared_per_block = _round_up(
            static_shared_bytes + dynamic_shared_bytes + reserved_shared,
            shared_mem_allocation_unit_bytes,
        )
        shared_limit = OccupancyLimit.available(
            shared_mem_carveout_bytes // shared_per_block
        )

    return {
        "launch__occupancy_limit_warps": warp_limit.as_dict(),
        "launch__occupancy_limit_blocks": block_limit.as_dict(),
        "launch__occupancy_limit_registers": register_limit.as_dict(),
        "launch__occupancy_limit_shared_mem": shared_limit.as_dict(),
    }


def _format_example(arch: str, **kwargs) -> str:
    limits = compute_occupancy_limits(arch=arch, **kwargs)
    lines = [f"occupancy limits for arch={arch} {kwargs}:"]
    for name in LAUNCH_OCCUPANCY_LIMIT_METRICS:
        cell = limits[name]
        lines.append(
            f"  {name} = value={cell['value']} status={cell['status']} "
            f"reason={cell['reason']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(
        _format_example(
            "sm_86",
            block_dim=256,
            registers_per_thread=32,
            static_shared_bytes=0,
            dynamic_shared_bytes=0,
        )
    )
