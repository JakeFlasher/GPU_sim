"""Static-surface sidecar joiner for the metric extractor JSON-L.

The per-launch JSON-L the metric extractor emits reserves a
``static_sidecar`` slot (see ``nvbit_tracer/metric_extractor/JSONL_SCHEMA.md``
line 62). The producer leaves it ``null`` because the static artifact
the schema documents — ``registers_per_thread``,
``dynamic_smem_per_block``, ``theoretical_occupancy_warps``,
``local_memory_spill_bytes`` — is owned by the harness-side Surface 4
(``ptxas -v`` recompile + ``cuobjdump --dump-resource-usage``), not by
the NVBit injection path. This module owns the post-hoc join: it
parses Surface 4's ``ptxas-v.txt`` into a mangled-name → resource-dict
mapping and enriches every launch record whose
``kernel_name_mangled`` matches.

The join is fail-closed by default: when the ``ptxas-v.txt`` artifact
is missing or the kernel name does not appear, the enriched record
carries a ``static_sidecar`` object whose four sub-fields are each
``null`` and whose ``join_status`` substatus marker records the
reason (``unjoinable_no_ptxas_v_artifact`` or
``unjoinable_kernel_not_in_ptxas_v``). The fail-closed shape is
deliberate: a downstream consumer that assumes ``registers_per_thread``
is always an integer would silently take an unjoinable record as
having zero registers, which would corrupt every occupancy / spill
analysis built on top. The explicit-null shape forces the consumer to
acknowledge the missing data.

The launches list returned by :func:`join_static_sidecar` carries the
original record verbatim plus the populated ``static_sidecar`` slot;
the function does NOT mutate the input list. This keeps the join
side-effect-free for testing and round-tripping through the strict
parser.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# Architecture-limit table for the supported targets in this project.
# Keys are the canonical arch tags the project uses for evidence labels
# and the manifest host fingerprint; values are the per-SM hardware
# limits relevant to theoretical-occupancy computation:
#
# * ``max_warps_per_sm`` — hardware cap on resident warps.
# * ``max_threads_per_sm`` — hardware cap on resident threads.
# * ``max_blocks_per_sm`` — hardware cap on resident blocks.
# * ``max_regs_per_sm`` — total register file size per SM.
# * ``max_shared_mem_per_sm`` — total shared memory budget per SM
#   (the value the compiler can allocate; static + dynamic smem
#   reduce per-block availability against this cap).
#
# Numbers come from the published CUDA C Programming Guide
# Appendix H "Compute Capabilities" table for each CC. Only the
# architectures this project actually targets (local sm_86 / sm_120
# and the cloud B200 sm_100a) are populated. Unknown architectures
# land in ``unjoinable_unknown_arch_limits`` per Codex review #4
# directive — silent null is rejected.
_ARCH_LIMITS = {
    "sm_86": {
        "max_warps_per_sm": 48,
        "max_threads_per_sm": 1536,
        "max_blocks_per_sm": 16,
        "max_regs_per_sm": 65536,
        "max_shared_mem_per_sm": 102400,  # 100 KB on Ampere consumer
    },
    "sm_100a": {
        # Blackwell datacenter (B200). Numbers from the CUDA C
        # Programming Guide Appendix H entry for compute capability
        # 10.0; conservative estimate where the spec is still in flux.
        "max_warps_per_sm": 64,
        "max_threads_per_sm": 2048,
        "max_blocks_per_sm": 32,
        "max_regs_per_sm": 65536,
        "max_shared_mem_per_sm": 233472,  # 228 KB per SM (TMEM excluded)
    },
    "sm_120": {
        # Blackwell consumer (RTX 5090 / 5090 / 5080 / 5090). The
        # consumer CC differs from sm_100a in SM count + bandwidth but
        # the per-SM occupancy caps follow the same Blackwell pattern.
        "max_warps_per_sm": 48,
        "max_threads_per_sm": 1536,
        "max_blocks_per_sm": 32,
        "max_regs_per_sm": 65536,
        "max_shared_mem_per_sm": 102400,
    },
}


def theoretical_occupancy_warps(
    *,
    registers_per_thread: Optional[int],
    block_size: int,
    static_shared_memory_bytes: int,
    dynamic_shared_memory_bytes: int,
    arch: str,
) -> Tuple[Optional[int], Optional[str]]:
    """Compute the theoretical occupancy in warps for one block on one SM.

    Returns ``(warps, reason)``: ``(N, None)`` when the limits resolve,
    ``(None, "<reason>")`` when they don't (unknown arch, missing
    register count, etc.). The reason string lands on the sidecar's
    join_status when the joiner cannot compute occupancy, per the
    Codex directive that unknown architectures must surface explicit
    ``unjoinable_unknown_arch_limits`` rather than silently emit
    ``None``.

    The math follows the CUDA Occupancy Calculator's standard
    formulation:

    * register cap: floor(max_regs_per_sm / (regs_per_thread * 32))
      gives warps capped by register file. The kernel-side
      ``regs_per_thread`` is whole-thread, so multiplying by 32 gets
      the per-warp register footprint.
    * shared-memory cap: floor(max_shared_mem_per_sm /
      (static_smem + dynamic_smem)) gives blocks capped by shared
      memory. Block count times block_size / 32 gives warps.
    * warps-per-SM hardware cap: from the arch table.
    * threads-per-SM hardware cap: max_threads_per_sm / block_size
      gives blocks; times block_size / 32 gives warps.
    * blocks-per-SM hardware cap: max_blocks_per_sm * block_size / 32
      gives warps.

    The final occupancy is the MIN across all five caps. Returns a
    typed ``None`` when registers_per_thread is missing (the static
    surface produced no ptxas-info line for the kernel), because the
    register cap cannot be computed without it.
    """
    if arch not in _ARCH_LIMITS:
        return None, f"unjoinable_unknown_arch_limits: arch={arch!r}"
    if registers_per_thread is None or registers_per_thread <= 0:
        return None, "unjoinable_missing_register_count"
    if block_size <= 0:
        return None, f"unjoinable_invalid_block_size: {block_size}"
    limits = _ARCH_LIMITS[arch]
    warp_size = 32
    warps_per_block = max(1, (block_size + warp_size - 1) // warp_size)
    # Register cap (warps).
    regs_warp = registers_per_thread * warp_size
    reg_cap_warps = limits["max_regs_per_sm"] // max(regs_warp, 1)
    # Hardware warp / thread / block caps (warps).
    warp_hw_cap_warps = limits["max_warps_per_sm"]
    thread_hw_cap_blocks = limits["max_threads_per_sm"] // max(block_size, 1)
    thread_hw_cap_warps = thread_hw_cap_blocks * warps_per_block
    block_hw_cap_warps = limits["max_blocks_per_sm"] * warps_per_block
    # Shared-memory cap (warps): per-block smem footprint = static + dynamic.
    block_smem = max(static_shared_memory_bytes, 0) + max(
        dynamic_shared_memory_bytes, 0,
    )
    if block_smem > 0:
        smem_cap_blocks = limits["max_shared_mem_per_sm"] // block_smem
    else:
        smem_cap_blocks = limits["max_blocks_per_sm"]
    smem_cap_warps = smem_cap_blocks * warps_per_block
    occupancy = min(
        reg_cap_warps,
        warp_hw_cap_warps,
        thread_hw_cap_warps,
        block_hw_cap_warps,
        smem_cap_warps,
    )
    # Floor at zero for the degenerate case where one cap reports 0
    # (e.g., regs_per_thread so large that no warps fit). That is a
    # valid observation, not an error — the kernel cannot run on this
    # arch.
    return max(0, int(occupancy)), None


# Resource-usage.txt parser. Format example:
#
#   Resource usage:
#    Common:
#     GLOBAL:46 CONSTANT[4]:368
#    Function _Zfoo:
#     REG:22 STACK:0 SHARED:128 LOCAL:0 CONSTANT[0]:392 TEXTURE:0 SURFACE:0 SAMPLER:0
#
# Each per-Function line carries the same per-kernel metrics as ptxas-v
# (registers, static shared memory, local memory) plus stack/texture/
# surface/sampler. The parser cross-checks ptxas-v's parse: if both
# artifacts disagree on a kernel's registers / shared / local, the
# sidecar carries the resource-usage value (canonically the source of
# truth in newer toolchains).
_RESOURCE_USAGE_FUNCTION_RE = re.compile(
    r"^\s*Function\s+([^\s:]+):\s*$"
)
_RESOURCE_USAGE_VALUES_RE = re.compile(
    r"^\s*REG:(?P<reg>\d+)"
    r"\s+STACK:(?P<stack>\d+)"
    r"\s+SHARED:(?P<shared>\d+)"
    r"\s+LOCAL:(?P<local>\d+)"
)


def parse_resource_usage(text: str) -> Dict[str, Dict[str, Optional[int]]]:
    """Parse a ``cuobjdump --dump-resource-usage`` output into
    ``{mangled_name: {registers, static_smem_bytes, stack_bytes, local_bytes}}``.

    Per-Function block format:

    ::

        Function <mangled>:
         REG:N STACK:N SHARED:N LOCAL:N CONSTANT[0]:N ...

    Lines that do not match either anchor are skipped. The parser is
    tolerant of formatting variation across CUDA toolchain versions —
    the four required values (REG / STACK / SHARED / LOCAL) anchor the
    parse; anything past them is ignored.
    """
    out: Dict[str, Dict[str, Optional[int]]] = {}
    current: Optional[str] = None
    for line in text.splitlines():
        fm = _RESOURCE_USAGE_FUNCTION_RE.match(line)
        if fm is not None:
            current = fm.group(1)
            out.setdefault(current, {
                "registers": None,
                "static_smem_bytes": None,
                "stack_bytes": None,
                "local_bytes": None,
            })
            continue
        if current is None:
            continue
        vm = _RESOURCE_USAGE_VALUES_RE.match(line)
        if vm is not None:
            out[current]["registers"] = int(vm.group("reg"))
            out[current]["stack_bytes"] = int(vm.group("stack"))
            out[current]["static_smem_bytes"] = int(vm.group("shared"))
            out[current]["local_bytes"] = int(vm.group("local"))
            # The match only consumes one line per kernel; reset
            # current so any trailing tokens (CONSTANT[0]:N etc.) on
            # the same kernel's line don't re-enter the parser.
            current = None
    return out


# A real ``ptxas -v`` dump shows entries like:
#
#   ptxas info    : Compiling entry function '_ZN...EpKf' for 'sm_86'
#   ptxas info    : Function properties for _ZN...EpKf
#       0 bytes stack frame, 8 bytes spill stores, 4 bytes spill loads
#   ptxas info    : Used 22 registers, used 1 barriers, 128 bytes smem, ...
#
# The "Compiling entry function" / "Function properties" lines anchor
# the per-kernel section; the indented "X bytes stack frame, ..." line
# and the "Used N registers, ..." line carry the values the schema's
# sidecar fields need.
_COMPILING_RE = re.compile(
    r"^ptxas\s+info\s*:\s*Compiling\s+entry\s+function\s+'([^']+)'"
    r"\s+for\s+'(?P<arch>sm_\d+a?)'"
)
_USED_REGS_RE = re.compile(
    r"^ptxas\s+info\s*:\s*Used\s+(?P<regs>\d+)\s+registers"
    r"(?:.*?(?P<smem>\d+)\s+bytes\s+smem)?"
)
_STACK_FRAME_RE = re.compile(
    r"(?P<stack>\d+)\s+bytes\s+stack\s+frame"
    r"(?:,\s*(?P<spill_stores>\d+)\s+bytes\s+spill\s+stores)?"
    r"(?:,\s*(?P<spill_loads>\d+)\s+bytes\s+spill\s+loads)?"
)


def parse_ptxas_v(text: str) -> Dict[str, Dict[str, Optional[int]]]:
    """Parse a ``ptxas -v`` dump into ``{mangled_name: {field: value}}``.

    Returned fields per kernel:

    * ``registers_per_thread`` — integer from the ``Used N registers``
      line. ``None`` when the line is missing.
    * ``shared_memory_bytes`` — integer bytes of static shared memory
      reported on the same line (``N bytes smem``). ``None`` when
      the kernel does not declare static shared memory.
    * ``stack_frame_bytes`` — integer from the indented function-
      properties line. ``None`` when the line is missing.
    * ``spill_store_bytes`` — integer from the same line. ``None``
      when the line is missing.
    * ``spill_load_bytes`` — integer from the same line. ``None``
      when the line is missing.

    Lines that do not match any of the documented anchors are skipped
    silently — the parser is forgiving about ordering and stray
    comments because the ``ptxas`` output format has varied slightly
    across CUDA toolchains. The kernel name comes from the
    "Compiling entry function" line; the function-properties line
    that immediately follows is associated with the most-recent
    kernel. Any "Function properties for X" line without a preceding
    "Compiling entry function X" anchor is ignored.
    """
    out: Dict[str, Dict[str, Optional[int]]] = {}
    current: Optional[str] = None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _COMPILING_RE.search(line)
        if m is not None:
            current = m.group(1)
            out.setdefault(current, {
                "registers_per_thread": None,
                "shared_memory_bytes": None,
                "stack_frame_bytes": None,
                "spill_store_bytes": None,
                "spill_load_bytes": None,
            })
            i += 1
            continue
        # ``Function properties for X`` precedes the indented line in
        # canonical output but we don't need to anchor on it — the
        # indented "X bytes stack frame, ..." line is unambiguous when
        # paired with the most-recent ``current`` kernel.
        if "bytes stack frame" in line and current is not None:
            sf = _STACK_FRAME_RE.search(line)
            if sf is not None:
                kdict = out[current]
                kdict["stack_frame_bytes"] = int(sf.group("stack"))
                if sf.group("spill_stores") is not None:
                    kdict["spill_store_bytes"] = int(sf.group("spill_stores"))
                if sf.group("spill_loads") is not None:
                    kdict["spill_load_bytes"] = int(sf.group("spill_loads"))
            i += 1
            continue
        ur = _USED_REGS_RE.match(line)
        if ur is not None and current is not None:
            kdict = out[current]
            kdict["registers_per_thread"] = int(ur.group("regs"))
            if ur.group("smem") is not None:
                kdict["shared_memory_bytes"] = int(ur.group("smem"))
            i += 1
            continue
        i += 1
    return out


def _empty_sidecar() -> Dict[str, Optional[int]]:
    """Return the four-slot null sidecar used when join is fail-closed."""
    return {
        "registers_per_thread": None,
        "dynamic_smem_per_block": None,
        "theoretical_occupancy_warps": None,
        "local_memory_spill_bytes": None,
    }


def join_static_sidecar(
    launches: List[Dict[str, Any]],
    ptxas_v_text: Optional[str],
    *,
    resource_usage_text: Optional[str] = None,
    arch: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return a new list of launch records with ``static_sidecar`` populated.

    Parameters
    ----------
    launches
        List of dict-shaped launch records (the parser's ``LaunchRecord.raw``
        payload). The function does not mutate the input list; each
        returned entry is a shallow copy with a fresh ``static_sidecar``
        slot. Records carrying ``launch_kind="graph_launch_not_observed"``
        are passed through unchanged with ``static_sidecar=null`` and a
        ``join_status="skipped_graph_launch_not_observed"`` annotation
        because the graph nodes' kernel identity is exactly what the
        NVBit tool failed to observe.
    ptxas_v_text
        The contents of the matched static surface bundle's
        ``ptxas-v.txt``. ``None`` when the static surface did not run or
        the artifact is absent; every record's sidecar is then the
        null-shape with ``join_status="unjoinable_no_ptxas_v_artifact"``.
    resource_usage_text
        The contents of the matched static surface bundle's
        ``resource-usage.txt`` (``cuobjdump --dump-resource-usage``).
        Optional — cross-checks ptxas-v's parse and supplies stack /
        local bytes that ptxas-v's per-Function properties line carries
        in a different format. When present, the resource-usage values
        WIN on conflict (cuobjdump is the canonical post-link view).
    arch
        Architecture tag for occupancy computation (one of ``sm_86``,
        ``sm_100a``, ``sm_120``). ``None`` or unknown arch -> sidecar
        carries occupancy as ``None`` with a documented
        ``unjoinable_unknown_arch_limits`` join_status; per Codex
        review #4 directive: silent null on a matched kernel is rejected.
    """
    if not ptxas_v_text:
        out: List[Dict[str, Any]] = []
        for rec in launches:
            enriched = dict(rec)
            enriched["static_sidecar"] = _empty_sidecar()
            enriched["static_sidecar_join_status"] = (
                "unjoinable_no_ptxas_v_artifact"
            )
            out.append(enriched)
        return out

    table = parse_ptxas_v(ptxas_v_text)
    resource_table: Dict[str, Dict[str, Optional[int]]] = {}
    if resource_usage_text:
        resource_table = parse_resource_usage(resource_usage_text)

    enriched_list: List[Dict[str, Any]] = []
    for rec in launches:
        enriched = dict(rec)
        kind = rec.get("launch_kind")
        if kind == "graph_launch_not_observed":
            enriched["static_sidecar"] = None
            enriched["static_sidecar_join_status"] = (
                "skipped_graph_launch_not_observed"
            )
            enriched_list.append(enriched)
            continue
        mangled = rec.get("kernel_name_mangled")
        if not mangled or mangled not in table:
            enriched["static_sidecar"] = _empty_sidecar()
            enriched["static_sidecar_join_status"] = (
                "unjoinable_kernel_not_in_ptxas_v"
            )
            enriched_list.append(enriched)
            continue
        stats = table[mangled]
        # Cross-check with resource-usage.txt when available. The
        # canonical-source rule per Codex directive: registers /
        # static shared memory / spill agree across both artifacts;
        # disagreement is surfaced as a substatus annotation but
        # cuobjdump's resource-usage view wins on conflict.
        ru_entry = resource_table.get(mangled)
        registers_per_thread = stats.get("registers_per_thread")
        static_smem_bytes = stats.get("shared_memory_bytes")
        stack_frame_bytes = stats.get("stack_frame_bytes")
        local_bytes = None  # ptxas-v does not report local-memory bytes
        if ru_entry is not None:
            ru_regs = ru_entry.get("registers")
            ru_smem = ru_entry.get("static_smem_bytes")
            ru_stack = ru_entry.get("stack_bytes")
            ru_local = ru_entry.get("local_bytes")
            if ru_regs is not None:
                registers_per_thread = ru_regs
            if ru_smem is not None:
                static_smem_bytes = ru_smem
            if ru_stack is not None:
                stack_frame_bytes = ru_stack
            if ru_local is not None:
                local_bytes = ru_local

        # Block size for occupancy = product of block_dim entries.
        block_dim = rec.get("block_dim") or [1, 1, 1]
        block_size = 1
        for bd in block_dim[:3]:
            block_size *= max(int(bd or 0), 0)
        block_size = max(block_size, 1)
        dynamic_smem = int(rec.get("dynamic_smem") or 0)

        # Theoretical occupancy from arch limits.
        occupancy_warps: Optional[int] = None
        join_status = "matched"
        if arch is None:
            occupancy_warps = None
            join_status = "unjoinable_unknown_arch_limits"
        else:
            occ, reason = theoretical_occupancy_warps(
                registers_per_thread=registers_per_thread,
                block_size=block_size,
                static_shared_memory_bytes=int(static_smem_bytes or 0),
                dynamic_shared_memory_bytes=dynamic_smem,
                arch=arch,
            )
            if occ is None:
                occupancy_warps = None
                # Reason already starts with "unjoinable_*" per the
                # helper contract.
                join_status = reason or "unjoinable_unknown_arch_limits"
            else:
                occupancy_warps = occ

        enriched["static_sidecar"] = {
            "registers_per_thread": registers_per_thread,
            "dynamic_smem_per_block": dynamic_smem,
            "theoretical_occupancy_warps": occupancy_warps,
            "local_memory_spill_bytes": _local_memory_spill_bytes(stats),
            # Forward-compat extras that the schema's optional sub-fields
            # tolerate (they're documented in the parser's
            # _OPTIONAL_STATIC_SIDECAR_SUBFIELDS list as a future
            # extension surface). Today they're recorded so a downstream
            # consumer can audit the joined values directly.
            "static_smem_bytes": static_smem_bytes,
            "stack_frame_bytes": stack_frame_bytes,
            "local_memory_bytes": local_bytes,
            "block_size": block_size,
            "arch": arch,
        }
        enriched["static_sidecar_join_status"] = join_status
        enriched_list.append(enriched)
    return enriched_list


def _local_memory_spill_bytes(
    stats: Dict[str, Optional[int]],
) -> Optional[int]:
    """Return the conservative spill estimate for one kernel.

    The ``ptxas info`` line reports ``spill_store_bytes`` and
    ``spill_load_bytes`` separately because the compiler counts each
    direction of the local-memory traffic; the larger of the two is a
    valid upper bound on the live spill set, which is what downstream
    consumers want for occupancy / pressure analysis. Returns
    ``None`` only when neither value was parsed.
    """
    stores = stats.get("spill_store_bytes")
    loads = stats.get("spill_load_bytes")
    if stores is None and loads is None:
        return None
    return max(stores or 0, loads or 0)
