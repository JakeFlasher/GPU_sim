"""Parse ``cuobjdump -res-usage`` static resource-usage text (GPU-free).

``cuobjdump`` is a static CUDA binary-utility (the project's Active Measurement
Stack lists "static SASS inspection through CUDA binary utilities"): it reads a
compiled fatbin and prints per-function resource usage WITHOUT executing any
kernel, reading any hardware counter, or touching driver state. This module
turns that text into the two static inputs the launch occupancy-limit
calculator needs -- the per-thread register count and the static shared-memory
bytes -- so the ``launch__occupancy_limit_*`` metrics can be fed real static
resource usage instead of caller-guessed integers.

``cuobjdump -res-usage`` emits one ``Fatbin elf code:`` block per (architecture,
translation-unit) pair. Each block declares ``arch = sm_XX`` and a
``Resource usage:`` section with a ``Common:`` line and, for blocks that carry
a device function, one ``Function <name>:`` line followed by a resource line::

    Function foo_kernel:
     REG:32 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:388 TEXTURE:0 SURFACE:0 SAMPLER:0

Blocks that carry no device function (the host stub / empty translation units)
have only the ``Common:`` line and contribute no usage row. The same arch can
appear in several blocks; the parser binds each ``Function`` line to the most
recently seen ``arch =`` line.

Units: ``REG`` is 32-bit registers per thread; ``SHARED`` is statically
declared shared-memory bytes per block; ``STACK``/``LOCAL`` are the per-thread
stack-frame / local-memory spill bytes; ``CONSTANT[0]`` is constant-bank bytes.

Public surface:

- ``KernelResourceUsage`` dataclass (one parsed ``Function`` row).
- ``parse_cuobjdump_res_usage(text)`` -> tuple of usage rows.
- ``select_resource_usage(usages, arch, function=None)`` -> one row, fail-closed
  and ambiguity-failing.
- ``occupancy_limits_from_res_usage(...)`` bridges a parsed row into
  ``compute_occupancy_limits`` so the T0 limits derive from real static usage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from occupancy_calculator import compute_occupancy_limits


class CuobjdumpResUsageError(ValueError):
    """Raised when the -res-usage text is unparseable or a selection misses.

    A miss (no row for the requested arch/function) and an ambiguity (several
    conflicting rows for the same arch/function) both fail closed here rather
    than silently returning a guessed or first-wins value.
    """


@dataclass(frozen=True)
class KernelResourceUsage:
    """One ``Function`` resource row from a single fatbin arch block."""

    arch: str
    function: str
    registers_per_thread: int
    stack_frame_bytes: int
    static_shared_bytes: int
    local_bytes: int
    constant_bytes: int
    texture: int
    surface: int
    sampler: int


_ARCH_RE = re.compile(r"^\s*arch\s*=\s*(sm_\w+)\s*$")
_FUNCTION_RE = re.compile(r"^\s*Function\s+(.+?)\s*:\s*$")
# The single resource line that follows a ``Function`` line. cuobjdump always
# emits the eight fields in this fixed order.
_RESOURCE_RE = re.compile(
    r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)\s+"
    r"CONSTANT\[0\]:(\d+)\s+TEXTURE:(\d+)\s+SURFACE:(\d+)\s+SAMPLER:(\d+)"
)


def parse_cuobjdump_res_usage(text: str) -> tuple[KernelResourceUsage, ...]:
    """Parse ``cuobjdump -res-usage`` text into per-function usage rows.

    Lines that match neither the arch header, a ``Function`` header, nor a
    resource line are ignored (``Common:``, ``GLOBAL:``, ``code version``,
    ``identifier``, banners, blank lines, and any PTX ``ptxas info`` chatter),
    so the parser is robust to surrounding noise. A ``Function`` header with no
    following resource line is dropped (no fabricated row).
    """
    if not isinstance(text, str) or not text.strip():
        raise CuobjdumpResUsageError("empty cuobjdump -res-usage text")

    usages: list[KernelResourceUsage] = []
    current_arch: str | None = None
    pending_function: str | None = None

    for raw in text.splitlines():
        arch_m = _ARCH_RE.match(raw)
        if arch_m:
            current_arch = arch_m.group(1)
            pending_function = None
            continue
        func_m = _FUNCTION_RE.match(raw)
        if func_m:
            pending_function = func_m.group(1)
            continue
        res_m = _RESOURCE_RE.search(raw)
        if res_m and pending_function is not None:
            if current_arch is None:
                raise CuobjdumpResUsageError(
                    f"resource line before any 'arch =' header: {raw!r}"
                )
            (reg, stack, shared, local, const, tex, surf, samp) = (
                int(g) for g in res_m.groups()
            )
            usages.append(
                KernelResourceUsage(
                    arch=current_arch,
                    function=pending_function,
                    registers_per_thread=reg,
                    stack_frame_bytes=stack,
                    static_shared_bytes=shared,
                    local_bytes=local,
                    constant_bytes=const,
                    texture=tex,
                    surface=surf,
                    sampler=samp,
                )
            )
            pending_function = None

    if not usages:
        raise CuobjdumpResUsageError(
            "no 'Function ...: REG:... SHARED:...' resource rows found; the text "
            "carries no device-function resource usage"
        )
    return tuple(usages)


def select_resource_usage(
    usages: tuple[KernelResourceUsage, ...] | list[KernelResourceUsage],
    arch: str,
    function: str | None = None,
) -> KernelResourceUsage:
    """Pick the single usage row for ``arch`` (and optional ``function``).

    Fail-closed: raises when no row matches. Ambiguity-failing: when several
    rows match, identical rows collapse to one, but two rows that disagree on
    resource values for the same (arch, function) -- or several distinct
    kernels with no ``function`` disambiguator -- raise rather than guessing.
    """
    candidates = [
        u
        for u in usages
        if u.arch == arch and (function is None or u.function == function)
    ]
    if not candidates:
        raise CuobjdumpResUsageError(
            f"no resource usage for arch={arch!r} function={function!r}"
        )

    distinct: list[KernelResourceUsage] = []
    for u in candidates:
        if u not in distinct:
            distinct.append(u)
    if len(distinct) == 1:
        return distinct[0]

    functions = sorted({u.function for u in distinct})
    if function is None and len(functions) > 1:
        raise CuobjdumpResUsageError(
            f"arch {arch!r} carries multiple kernels {functions}; pass "
            f"function= to disambiguate"
        )
    raise CuobjdumpResUsageError(
        f"conflicting resource usage for arch={arch!r} function={function!r}: "
        f"{distinct}"
    )


def occupancy_limits_from_res_usage(
    res_usage_text: str,
    arch: str,
    block_dim: int,
    function: str | None = None,
    dynamic_shared_bytes: int = 0,
    table=None,
) -> dict[str, dict]:
    """Compute the launch occupancy limits from raw ``cuobjdump -res-usage`` text.

    Bridges the static resource parse into ``compute_occupancy_limits``: the
    register count and static shared bytes come from the parsed row, the launch
    config (block_dim, dynamic shared) from the caller, and the per-SM ceilings
    from the fail-closed device-constant table.
    """
    usages = parse_cuobjdump_res_usage(res_usage_text)
    chosen = select_resource_usage(usages, arch=arch, function=function)
    return compute_occupancy_limits(
        arch=arch,
        block_dim=block_dim,
        registers_per_thread=chosen.registers_per_thread,
        static_shared_bytes=chosen.static_shared_bytes,
        dynamic_shared_bytes=dynamic_shared_bytes,
        table=table,
    )
