#!/usr/bin/env python3
"""Deterministic replay-count producer (sm_86 quarantine).

Reconstructs the T1 replay-count metrics per cell from UNSAMPLED NVBit
dynamic traces, implementing EXACTLY the live-discovered observation
contract semantics — no metric is emitted as scoreable without passing
``observation_contract.assert_metric_scoreable`` first.

Pinned semantics implemented (from the committed contract):

- LSU request formation is per warp instruction: every wire memory
  record with a non-empty effective mask (active & predicate)
  contributes ONE request to its (class, kind) pipe. 128b vector
  accesses are NOT wavefront-split. SCOPE NOTE (task review finding):
  the effective-mask handling is fixture-pinned for the WIRE-MODELED
  classes (GLOBAL/LOCAL) only; ATOMIC/TEXTURE/SURFACE counts come from
  the exclusion sidecar, whose aggregation happens producer-side
  before any zero-mask filter — their predication behavior is
  unfixtured (no corpus contamination observed: non-zero atomic cells
  match NCU exactly and tex/surface scored rows are zero).
- Texture/surface request formation is FOUR wavefronts per warp
  instruction: the tracer routes TEXTURE/SURFACE records to the
  unsupported-memory-path sidecar (recognized-but-excluded,
  path_resolved=true), whose per-PC counts are warp-instruction
  executions; the producer multiplies by 4 per the contract.
- Warp-instruction counting is issued-warp semantics: the wire carries
  EVERY issued warp instruction (memory and non-memory records alike),
  so ``smsp__inst_executed.sum`` is the total record count.
- Sector counting is touched-32B-sector arithmetic, SCOPE-RESTRICTED
  to 128B-aligned warp footprints: any scored global ld/st instruction
  whose footprint is not 32B-boundary-clean types the sector metric
  for the whole cell (``sector_scope_misaligned_footprint``).
- The requests rollup is the sum of the 11 pipe request metrics and is
  scoreable ONLY when every contract-included op class is either
  modeled or absent from the cell's trace; otherwise
  ``rollup_unsupported``.

Wire-truth notes (verified against live labels):

- ATOM vs RED collapse to MemoryOpClass ATOMIC on the wire (DEC-3: the
  pinned tracer tree is untouchable). The producer resolves the split
  STATICALLY: the exclusion sidecar records per-PC counts; cuobjdump
  SASS for the closed-identity-verified cell binary maps each PC to
  its opcode head (ATOMG/ATOM -> op_atom, RED -> op_red). A PC that
  cannot be resolved types both metrics ``atom_red_sass_unresolved``
  (fail-closed) — never guessed.
- Exact support: any sampling/truncation/drop/budget evidence
  (trace_volume_exceeded.json, nvbit_drop_rate_exceeded.json,
  nvbit_slowdown_budget_exceeded.json, non-zero drop counters,
  expected!=admitted record counts) types the WHOLE cell
  ``trace_support_insufficient``.
- Unknown op classes (UNSPECIFIED sidecar entries, path_resolved
  false) are counted in the typed coverage denominator and gate the
  rollup fail-closed.

Closed identity: every emission carries (binary_sha256,
kernel_signature_sha256, shape_point) and is produced ONLY when the
trace sidecar's identity equals the admitted label pin; mismatches are
typed (``trace_binary_stale`` / ``kernel_signature_mismatch``), never
joined loosely.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
_TRACER_DIR = REPO_ROOT / "tools" / "nvbit_tracer"
if str(_TRACER_DIR) not in sys.path:
    sys.path.insert(0, str(_TRACER_DIR))

from tools.nvbit_tracer.proto_binding_loader import (  # noqa: E402
    load_dynamic_trace_bindings,
)

from observation_contract import (  # noqa: E402
    ContractEntryMissingError,
    ContractMismatchError,
    ContractNoExpressibleFixtureError,
    ContractUnderdeterminedError,
    assert_metric_scoreable,
    load_contract,
)
from capture_count_labels import (  # noqa: E402
    QuarantinePathError,
    assert_quarantine_path,
)

_M = {
    "rollup": "l1tex__t_requests.sum",
    "g_atom": "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum",
    "g_ld": "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "g_red": "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum",
    "g_st": "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "l_ld": "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum",
    "l_st": "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum",
    "s_atom": "l1tex__t_requests_pipe_tex_mem_surface_op_atom.sum",
    "s_ld": "l1tex__t_requests_pipe_tex_mem_surface_op_ld.sum",
    "s_red": "l1tex__t_requests_pipe_tex_mem_surface_op_red.sum",
    "s_st": "l1tex__t_requests_pipe_tex_mem_surface_op_st.sum",
    "tex": "l1tex__t_requests_pipe_tex_mem_texture.sum",
    "sec_ld": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "sec_st": "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "inst": "smsp__inst_executed.sum",
}
ROLLUP_PIPE_KEYS = (
    "g_atom", "g_ld", "g_red", "g_st", "l_ld", "l_st",
    "s_atom", "s_ld", "s_red", "s_st", "tex",
)

# proto enum values (instruction.proto) — mirrored, not imported, so the
# synthetic-fixture tests can run without bindings present.
_CLASS_GLOBAL = 1
_CLASS_LOCAL = 3
_KIND_LOAD = 1
_KIND_STORE = 2

_SECTOR_BYTES = 32
_TEX_SURFACE_WAVEFRONTS_PER_INST = 4
_UNSUPPORTED_MEMORY_PATH_MAX_BYTES = 16 * 1024 * 1024

DEFAULT_EMISSIONS_OUT = "data/local_labels/replay_count_emissions_sm_86.json"

# Sidecar evidence of sampling / truncation / budget pressure: presence
# of ANY of these (or non-zero drop counters) types the cell.
_SUPPORT_SIDECARS = (
    "trace_volume_exceeded.json",
    "nvbit_drop_rate_exceeded.json",
    "nvbit_slowdown_budget_exceeded.json",
)
_DROP_COUNTER_FIELDS = (
    "channel_overflow_dropped",
    "ring_buffer_loss_dropped",
    "decoder_gap_dropped",
    "expected_record_mismatch_dropped",
)


class TraceSupportInsufficientError(RuntimeError):
    typed_reason = "trace_support_insufficient"


class ClosedIdentityError(RuntimeError):
    def __init__(self, typed_reason: str, detail: str) -> None:
        super().__init__(detail)
        self.typed_reason = typed_reason


def _reject_duplicate_json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _load_unsupported_memory_path(path: Path) -> list[dict]:
    """Load the exclusion sidecar without duplicate-key or size fallbacks."""
    try:
        with path.open("rb") as stream:
            raw = stream.read(_UNSUPPORTED_MEMORY_PATH_MAX_BYTES + 1)
    except OSError as exc:
        raise TraceSupportInsufficientError(
            f"unsupported-memory sidecar read failed: {exc}"
        ) from exc
    if len(raw) > _UNSUPPORTED_MEMORY_PATH_MAX_BYTES:
        raise TraceSupportInsufficientError(
            f"unsupported-memory sidecar exceeds {_UNSUPPORTED_MEMORY_PATH_MAX_BYTES} bytes"
        )
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_json_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise TraceSupportInsufficientError(
            f"unsupported-memory sidecar is malformed: {exc}"
        ) from exc
    if not isinstance(value, list) or any(not isinstance(entry, dict) for entry in value):
        raise TraceSupportInsufficientError(
            "unsupported-memory sidecar must be a JSON array of objects"
        )
    return value


def _scope_exclusion_entries(
    entries: list[dict], selected_function_id: int | None,
) -> list[dict]:
    """Select exclusions by the tracer's function-unique identity."""
    if not entries:
        return []
    if selected_function_id is None or selected_function_id <= 0:
        raise TraceSupportInsufficientError(
            "unsupported-memory sidecar cannot be scoped to one function"
        )
    selected = selected_function_id
    scoped: list[dict] = []
    for entry in entries:
        raw_id = entry.get("kernel_unique_id")
        if type(raw_id) is not int or raw_id <= 0:
            raise TraceSupportInsufficientError(
                "unsupported-memory sidecar has an invalid kernel_unique_id"
            )
        if type(entry.get("schema_version")) is not int or entry["schema_version"] != 1:
            raise TraceSupportInsufficientError(
                "unsupported-memory sidecar has an unsupported schema_version"
            )
        if entry.get("typed_reason") != "unsupported_memory_path":
            raise TraceSupportInsufficientError(
                "unsupported-memory sidecar has an invalid typed_reason"
            )
        for field in ("device", "slug", "excluded_op_class"):
            if not isinstance(entry.get(field), str) or not entry[field]:
                raise TraceSupportInsufficientError(
                    f"unsupported-memory sidecar has an invalid {field}"
                )
        if "path_resolved" in entry and type(entry["path_resolved"]) is not bool:
            raise TraceSupportInsufficientError(
                "unsupported-memory sidecar has an invalid path_resolved"
            )
        for field, minimum in (("instruction_pc", 0), ("count", 1)):
            value = entry.get(field)
            if type(value) is not int or value < minimum:
                raise TraceSupportInsufficientError(
                    f"unsupported-memory sidecar has an invalid {field}"
                )
        function_id = raw_id
        if function_id == selected:
            scoped.append(entry)
    if not scoped:
        raise TraceSupportInsufficientError(
            f"unsupported-memory sidecar has no entries for function {selected}"
        )
    return scoped


def assert_exact_support(trace_dir: Path, sidecar: dict) -> None:
    """Fail closed on ANY sampling/truncation/budget evidence."""
    for name in _SUPPORT_SIDECARS:
        for candidate in (trace_dir / name,
                          trace_dir / "extra_info" / name):
            if candidate.is_file():
                raise TraceSupportInsufficientError(
                    f"budget/sampling sidecar present: {candidate.name}"
                )
    policy = sidecar.get("sampling_policy")
    if policy not in (None, "off"):
        raise TraceSupportInsufficientError(
            f"sampling_policy={policy!r} is not exact support"
        )
    for field in _DROP_COUNTER_FIELDS:
        if sidecar.get(field, 0) != 0:
            raise TraceSupportInsufficientError(
                f"{field}={sidecar.get(field)} != 0"
            )
    if sidecar.get("actual_drop_rate", 0.0) != 0.0:
        raise TraceSupportInsufficientError(
            f"actual_drop_rate={sidecar.get('actual_drop_rate')}"
        )
    expected = sidecar.get("expected_record_count")
    admitted = sidecar.get("admitted_record_count")
    if expected is not None and admitted is not None and expected != admitted:
        raise TraceSupportInsufficientError(
            f"expected_record_count={expected} != "
            f"admitted_record_count={admitted} (truncation)"
        )


def assert_closed_identity(cell: dict, sidecar: dict) -> None:
    """The trace binds to the admitted cell ONLY on the full closed
    identity tuple; anything weaker is rejected with a typed reason."""
    if sidecar.get("binary_sha256") != cell.get("binary_sha256"):
        raise ClosedIdentityError(
            "trace_binary_stale",
            f"trace sidecar binary {str(sidecar.get('binary_sha256'))[:12]} "
            f"!= label pin {str(cell.get('binary_sha256'))[:12]}",
        )
    if sidecar.get("kernel_signature_sha256") != cell.get(
            "kernel_signature_sha256"):
        raise ClosedIdentityError(
            "kernel_signature_mismatch",
            f"trace kernel signature "
            f"{str(sidecar.get('kernel_signature_sha256'))[:12]} != label "
            f"{str(cell.get('kernel_signature_sha256'))[:12]}",
        )


# ---------------------------------------------------------------------------
# SASS-side ATOM/RED resolution (static, fail-closed)
# ---------------------------------------------------------------------------


def sass_opcode_by_function_pc(
    binary: Path, arch: str = "sm_86",
) -> dict[str, list[dict[int, str]]]:
    """Map function name -> LIST of {function-relative PC -> opcode}
    maps for the sections targeting ``arch`` ONLY (cuobjdump
    --dump-sass). A fatbinary embeds one code section per target arch
    repeating the same function names at different instruction
    layouts; the sidecar's PCs are offsets into the section that
    actually RAN, so other arch sections must never be consulted."""
    proc = subprocess.run(
        ["cuobjdump", "--dump-sass", str(binary)],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"cuobjdump failed: {proc.stderr[-300:]}")
    out: dict[str, list[dict[int, str]]] = {}
    current: dict[int, str] | None = None
    current_arch: str | None = None
    for line in proc.stdout.splitlines():
        am = re.match(r"\s*code for (\S+)", line)
        if am:
            current_arch = am.group(1)
            current = None
            continue
        fm = re.search(r"Function : (\S+)", line)
        if fm:
            if current_arch == arch:
                current = {}
                out.setdefault(fm.group(1), []).append(current)
            else:
                current = None
            continue
        if current is None:
            continue
        m = re.match(
            r"\s*/\*([0-9a-fA-F]+)\*/\s+(?:@!?\S+\s+)?([A-Z][A-Z0-9.@]*)",
            line)
        if m:
            current[int(m.group(1), 16)] = m.group(2)
    return out


def _atomic_family(opcode: str) -> str | None:
    head = opcode.split(".")[0]
    if head.startswith("ATOM"):
        return "atom"
    if head.startswith("RED"):
        return "red"
    return None


def split_atomic_counts(
    atomic_entries: list[dict],
    opcode_by_function_pc: dict[str, list[dict[int, str]]],
) -> tuple[int, int] | None:
    """Split the wire's combined ATOMIC per-(kernel, PC) counts into
    (atom, red) via the static SASS opcode FAMILY at the matching
    function's PC (the sidecar's ``slug`` field carries the kernel
    symbol). When the fatbinary carries several sections with the same
    function, every section defining the PC must agree on the family
    (ATOM* vs RED*); any disagreement or unresolvable PC fails closed
    (caller types ``atom_red_sass_unresolved``)."""
    atom = red = 0
    for entry in atomic_entries:
        kernel = str(entry.get("slug", ""))
        pc = int(entry.get("instruction_pc", -1))
        count = int(entry.get("count", 0))
        if not kernel:
            return None
        # The sidecar's ``slug`` is the exact function symbol used by the
        # tracer.  A substring match is unsafe here: symbols such as
        # ``foo`` and ``foo_variant`` can coexist in one cubin, and a missing
        # ``foo`` entry would otherwise silently borrow ``foo_variant``'s
        # opcode family at the same PC.  That turns an unresolved ATOM/RED
        # attribution into a scored (but wrong) metric.  Require an exact
        # symbol key and fail closed when the symbol is absent.
        candidate_maps = opcode_by_function_pc.get(kernel, [])
        families = {
            _atomic_family(pcs[pc])
            for pcs in candidate_maps if pc in pcs
        }
        if len(families) != 1 or None in families:
            return None
        family = families.pop()
        if family == "atom":
            atom += count
        else:
            red += count
    return atom, red


# ---------------------------------------------------------------------------
# Trace walk (pure, deterministic)
# ---------------------------------------------------------------------------


def walk_threadblock(tb) -> dict:
    """Aggregate one threadblock message: per-(class, kind) request
    counts, global ld/st sector counts + alignment flags, and the
    issued-warp instruction total. Reuses the canonical lane
    reconstruction from the trace import contract."""
    from import_trace_contract import _reconstruct_lane_vector

    agg = {
        "inst_records": 0,
        "requests": defaultdict(int),     # (class, kind) -> count
        "local_word_requests": defaultdict(int),  # kind -> 32b-word count
        "sectors_ld": 0,
        "sectors_st": 0,
        "misaligned_ld": 0,
        "misaligned_st": 0,
        "reconstruction_failures": 0,
    }
    for warp_index in sorted(tb.warps.keys()):
        warp = tb.warps[warp_index]
        for inst in warp.instructions:
            agg["inst_records"] += 1
            if not inst.HasField("memory_op_class"):
                continue
            if not inst.addresses:
                continue
            active = inst.active_mask if inst.HasField("active_mask") else 0
            pred = (inst.predicate_mask
                    if inst.HasField("predicate_mask") else 0xFFFFFFFF)
            and_mask = active & pred
            if and_mask == 0:
                continue
            cls = inst.memory_op_class
            kind = (inst.memory_op_access_kind
                    if inst.HasField("memory_op_access_kind") else 0)
            agg["requests"][(cls, kind)] += 1
            if cls == _CLASS_LOCAL:
                w = inst.data_width if inst.HasField("data_width") else 4
                agg["local_word_requests"][kind] += max(
                    1, (w + 3) // 4)
            if cls != _CLASS_GLOBAL or kind not in (_KIND_LOAD, _KIND_STORE):
                continue
            width = inst.data_width if inst.HasField("data_width") else None
            addr = inst.addresses[0]
            address_mask = (
                int(addr.lane_mask) if addr.HasField("lane_mask") else and_mask
            )
            lanes, reason = _reconstruct_lane_vector(addr, address_mask)
            if lanes is None:
                agg["reconstruction_failures"] += 1
                continue
            sectors: set[int] = set()
            aligned = True
            for lane in range(32):
                if not (address_mask >> lane) & 1:
                    continue
                a = lanes[lane]
                w = width or addr.data_width or 4
                if a % w != 0 or a % _SECTOR_BYTES not in range(
                        0, _SECTOR_BYTES, w):
                    pass  # per-lane natural alignment is the witness below
                first = a // _SECTOR_BYTES
                last = (a + w - 1) // _SECTOR_BYTES
                sectors.update(range(first, last + 1))
                if a % w != 0:
                    aligned = False
            footprint_lanes = [lanes[i] for i in range(32)
                               if (address_mask >> i) & 1]
            w = width or addr.data_width or 4
            start = min(footprint_lanes)
            end = max(footprint_lanes) + w
            if start % _SECTOR_BYTES != 0 or end % _SECTOR_BYTES != 0:
                aligned = False
            if kind == _KIND_LOAD:
                agg["sectors_ld"] += len(sectors)
                if not aligned:
                    agg["misaligned_ld"] += 1
            else:
                agg["sectors_st"] += len(sectors)
                if not aligned:
                    agg["misaligned_st"] += 1
    return agg


def first_launch_dir(trace_dir: Path) -> tuple[Path, int]:
    """Return (first-launch kernel directory, first_kernel_id): the lowest
    (device, stream, kernel) launch under ``threadblocks/``.

    The NCU label profiles ONE launch (``--launch-count 1``, the first) while
    the trace tree carries EVERY launch, so every launch-scoped consumer must
    restrict to this directory; a multi-launch cell walked unscoped over-counts
    by exactly the launch count (verified live: x5 on the 5-launch streaming
    cell). This is the single source of truth for that scoping."""
    tb_root = trace_dir / "threadblocks"
    if not tb_root.is_dir():
        raise TraceSupportInsufficientError(
            f"no threadblocks tree under {trace_dir}"
        )

    def _num(p: Path, prefix: str) -> int:
        try:
            return int(p.name.removeprefix(prefix))
        except ValueError:
            return 1 << 30

    devices = sorted(tb_root.glob("device_*"), key=lambda p: _num(p, "device_"))
    if not devices:
        raise TraceSupportInsufficientError("no device_* under threadblocks")
    streams = sorted(devices[0].glob("stream_*"), key=lambda p: _num(p, "stream_"))
    if not streams:
        raise TraceSupportInsufficientError("no stream_* under device")
    kernels = sorted(streams[0].glob("kernel_*"), key=lambda p: _num(p, "kernel_"))
    if not kernels:
        raise TraceSupportInsufficientError("no kernel_* under stream")
    return kernels[0], _num(kernels[0], "kernel_")


def walk_trace_dir(trace_dir: Path) -> dict:
    """Walk every threadblock .pb under the trace dir and merge the
    aggregates; load the typed exclusion sidecar (per-PC counts for
    the recognized-but-excluded classes + the UNSPECIFIED fallback)."""
    threadblock_pb2 = load_dynamic_trace_bindings().threadblock_pb2

    # Restrict to the lowest (device, stream, kernel) launch — the one the
    # label measured — via the shared launch-scoping helper.
    first_launch_root, _first_kernel_id = first_launch_dir(trace_dir)
    total = {
        "inst_records": 0,
        "requests": defaultdict(int),
        "local_word_requests": defaultdict(int),
        "sectors_ld": 0,
        "sectors_st": 0,
        "misaligned_ld": 0,
        "misaligned_st": 0,
        "reconstruction_failures": 0,
        "threadblock_files": 0,
    }
    selected_function_id: int | None = None
    for pb_path in sorted(first_launch_root.rglob("*.pb")):
        tb = threadblock_pb2.threadblock()
        tb.ParseFromString(pb_path.read_bytes())
        agg = walk_threadblock(tb)
        total["threadblock_files"] += 1
        total["inst_records"] += agg["inst_records"]
        total["sectors_ld"] += agg["sectors_ld"]
        total["sectors_st"] += agg["sectors_st"]
        total["misaligned_ld"] += agg["misaligned_ld"]
        total["misaligned_st"] += agg["misaligned_st"]
        total["reconstruction_failures"] += agg["reconstruction_failures"]
        for warp in tb.warps.values():
            for inst in warp.instructions:
                function_id = int(inst.function_unique_id)
                if function_id > 0:
                    if selected_function_id is None:
                        selected_function_id = function_id
                    elif function_id != selected_function_id:
                        raise TraceSupportInsufficientError(
                            "selected launch carries multiple function_unique_id values"
                        )
        for key, n in agg["requests"].items():
            total["requests"][key] += n
        for kind, n in agg["local_word_requests"].items():
            total["local_word_requests"][kind] += n
    exclusions: dict[str, dict[int, int]] = defaultdict(dict)
    entries_by_class: dict[str, list[dict]] = defaultdict(list)
    ex_path = trace_dir / "extra_info" / "unsupported_memory_path.json"
    if ex_path.is_file():
        all_entries = _load_unsupported_memory_path(ex_path)
        # Exclusion entries identify the static function, not the launch
        # directory.  Scope by the function_unique_id carried by the selected
        # launch's dynamic instructions; comparing against kernel_<id> would
        # accidentally include every entry on the mismatch fallback.
        scoped = _scope_exclusion_entries(all_entries, selected_function_id)
        for entry in scoped:
            cls = entry.get("excluded_op_class")
            pc = int(entry.get("instruction_pc", -1))
            exclusions[cls][pc] = exclusions[cls].get(pc, 0) + int(
                entry.get("count", 0))
            entries_by_class[cls].append(entry)
    total["excluded_op_class_pc_counts"] = {
        k: dict(v) for k, v in exclusions.items()
    }
    total["excluded_op_class_entries"] = dict(entries_by_class)
    return total


# ---------------------------------------------------------------------------
# Emission (contract-gated, typed accounting)
# ---------------------------------------------------------------------------


def _typed(reason: str, detail: str = "") -> dict:
    out = {"status": "typed", "typed_reason": reason}
    if detail:
        out["detail"] = detail[:300]
    return out


def _scored(value: int) -> dict:
    return {"status": "scored", "value": value}


def emit_metrics_for_walk(
    walk: dict,
    contract: dict,
    opcode_by_pc: dict[int, str] | None,
) -> dict[str, dict]:
    """Map the trace walk to per-metric scored/typed emissions, gated
    metric-by-metric through the observation contract."""
    requests = walk["requests"]
    excl = walk.get("excluded_op_class_pc_counts", {})

    def gate(metric: str):
        try:
            return assert_metric_scoreable(metric, contract), None
        except (ContractEntryMissingError, ContractMismatchError,
                ContractUnderdeterminedError,
                ContractNoExpressibleFixtureError) as exc:
            return None, _typed(exc.typed_reason, str(exc))

    out: dict[str, dict] = {}

    # --- LSU global/local ld+st: wire records, per-warp-inst ---
    local_words = walk.get("local_word_requests", {})
    lsu_values = {
        "g_ld": sum(n for (c, k), n in requests.items()
                    if c == _CLASS_GLOBAL and k == _KIND_LOAD),
        "g_st": sum(n for (c, k), n in requests.items()
                    if c == _CLASS_GLOBAL and k == _KIND_STORE),
        "l_ld": sum(n for (c, k), n in requests.items()
                    if c == _CLASS_LOCAL and k == _KIND_LOAD),
        "l_st": sum(n for (c, k), n in requests.items()
                    if c == _CLASS_LOCAL and k == _KIND_STORE),
    }
    local_word_values = {
        "l_ld": local_words.get(_KIND_LOAD, 0),
        "l_st": local_words.get(_KIND_STORE, 0),
    }
    async_copy_count = sum(excl.get("ASYNC_COPY", {}).values())
    for key, value in lsu_values.items():
        metric = _M[key]
        entry, typed = gate(metric)
        if typed:
            out[metric] = typed
            continue
        if key in ("g_ld",) and async_copy_count > 0:
            # NCU attributes LDGSTS (cp.async) loads to the global ld
            # pipe; that attribution is unfixtured in the contract, so
            # the pipe cannot be reconstructed for this cell.
            out[metric] = _typed(
                "async_copy_unfixtured",
                f"{async_copy_count} ASYNC_COPY warp instructions "
                "present; LDGSTS-to-op_ld attribution is not pinned by "
                "any fixture",
            )
            continue
        semantic = (entry or {}).get("semantic")
        if key in local_word_values and semantic == "per_32b_word_split":
            # The contract discovered LOCAL request formation splits
            # per 32-bit word (a 64b access forms 2 requests, 128b
            # forms 4); the wire's data_width drives the split.
            out[metric] = _scored(local_word_values[key])
        else:
            out[metric] = _scored(value)

    # --- ATOM/RED: sidecar counts + static SASS split ---
    atomic_entries = walk.get("excluded_op_class_entries", {}).get(
        "ATOMIC", [])
    atomic_pcs = {int(pc): n for pc, n in excl.get("ATOMIC", {}).items()}
    for key in ("g_atom", "g_red"):
        metric = _M[key]
        entry, typed = gate(metric)
        if typed:
            out[metric] = typed
            continue
        if not atomic_entries and not atomic_pcs:
            out[metric] = _scored(0)
            continue
        if opcode_by_pc is None:
            out[metric] = _typed(
                "atom_red_sass_unresolved",
                "no SASS opcode map available for the cell binary",
            )
            continue
        split = split_atomic_counts(atomic_entries, opcode_by_pc)
        if split is None:
            out[metric] = _typed(
                "atom_red_sass_unresolved",
                f"unresolvable atomic PCs: {sorted(atomic_pcs)}",
            )
            continue
        atom, red = split
        out[metric] = _scored(atom if key == "g_atom" else red)

    # --- texture/surface: sidecar counts x 4 wavefronts ---
    tex_count = sum(excl.get("TEXTURE", {}).values())
    surf_count = sum(excl.get("SURFACE", {}).values())
    for key, warp_insts in (("tex", tex_count),):
        metric = _M[key]
        entry, typed = gate(metric)
        out[metric] = typed if typed else _scored(
            warp_insts * _TEX_SURFACE_WAVEFRONTS_PER_INST)
    # Surface ld/st: the sidecar pools the class without an opcode-kind
    # split; resolvable only when the class is absent (both exactly 0).
    for key in ("s_ld", "s_st"):
        metric = _M[key]
        entry, typed = gate(metric)
        if typed:
            out[metric] = typed
        elif surf_count == 0:
            out[metric] = _scored(0)
        else:
            out[metric] = _typed(
                "surface_kind_wire_unresolved",
                "the exclusion sidecar pools SURFACE without a "
                "load/store split; non-zero surface traffic cannot be "
                "attributed per-kind from this trace surface",
            )
    for key in ("s_atom", "s_red"):
        metric = _M[key]
        entry, typed = gate(metric)
        out[metric] = typed if typed else _scored(0)

    # --- sectors: aligned-scope gate ---
    for key, total_key, mis_key in (
            ("sec_ld", "sectors_ld", "misaligned_ld"),
            ("sec_st", "sectors_st", "misaligned_st")):
        metric = _M[key]
        entry, typed = gate(metric)
        if typed:
            out[metric] = typed
            continue
        if key == "sec_ld" and async_copy_count > 0:
            out[metric] = _typed(
                "async_copy_unfixtured",
                "LDGSTS sectors land in the global ld pipe; the "
                "attribution is not pinned by any fixture",
            )
            continue
        scope = (entry or {}).get("semantic_scope")
        if walk.get(mis_key, 0) > 0 and scope == (
                "aligned_128B_warp_footprints_only"):
            out[metric] = _typed(
                "sector_scope_misaligned_footprint",
                f"{walk[mis_key]} misaligned warp footprints; the "
                "pinned sector semantic is scope-restricted to aligned "
                "footprints",
            )
            continue
        if walk.get("reconstruction_failures", 0) > 0:
            out[metric] = _typed(
                "lane_reconstruction_failed",
                f"{walk['reconstruction_failures']} records failed "
                "lane reconstruction",
            )
            continue
        out[metric] = _scored(walk[total_key])

    # --- inst_executed: issued-warp records ---
    metric = _M["inst"]
    entry, typed = gate(metric)
    out[metric] = typed if typed else _scored(walk["inst_records"])

    # --- rollup: sum of pipes, fail-closed on coverage ---
    metric = _M["rollup"]
    entry, typed = gate(metric)
    if typed:
        out[metric] = typed
    else:
        unspecified = sum(excl.get("UNSPECIFIED", {}).values())
        other_classes = {
            cls: sum(pcs.values()) for cls, pcs in excl.items()
            if cls not in ("ATOMIC", "TEXTURE", "SURFACE", "UNSPECIFIED")
            and sum(pcs.values()) > 0
        }
        # The plan's rollup rule: every included op class MODELED or
        # ABSENT. A pipe whose metric is typed still contributes 0 when
        # its op-class traffic is provably absent from the trace.
        _pipe_traffic = {
            "s_atom": surf_count, "s_ld": surf_count,
            "s_red": surf_count, "s_st": surf_count,
            "tex": tex_count,
            "g_atom": sum(atomic_pcs.values()),
            "g_red": sum(atomic_pcs.values()),
        }

        def _pipe_contribution(key: str) -> int | None:
            emission = out[_M[key]]
            if emission["status"] == "scored":
                return emission["value"]
            if _pipe_traffic.get(key, 1) == 0:
                return 0  # op class absent from the trace
            return None

        contributions = {k: _pipe_contribution(k) for k in ROLLUP_PIPE_KEYS}
        if unspecified > 0:
            out[metric] = _typed(
                "rollup_unsupported",
                f"{unspecified} warp instructions in UNSPECIFIED op "
                "classes (unknown to the classifier); the rollup's "
                "included-pipe coverage cannot be established",
            )
        elif other_classes:
            out[metric] = _typed(
                "rollup_unsupported",
                f"unmodeled excluded op classes present: "
                f"{sorted(other_classes)}",
            )
        elif any(v is None for v in contributions.values()):
            bad = sorted(_M[k] for k, v in contributions.items()
                         if v is None)
            out[metric] = _typed(
                "rollup_unsupported",
                f"included pipe metrics neither scoreable nor "
                f"traffic-absent: {bad}",
            )
        else:
            out[metric] = _scored(sum(contributions.values()))

    # --- typed coverage denominator ---
    out["_coverage"] = {
        "modeled_warp_instructions": sum(requests.values()),
        "excluded_warp_instructions_by_class": {
            cls: sum(pcs.values()) for cls, pcs in excl.items()
        },
        "unspecified_warp_instructions": sum(
            excl.get("UNSPECIFIED", {}).values()),
        "reconstruction_failures": walk.get("reconstruction_failures", 0),
    }
    return out


# ---------------------------------------------------------------------------
# Corpus emission CLI
# ---------------------------------------------------------------------------


def _trace_dir_for_cell(cell: dict) -> Path | None:
    slug = cell["slug"]
    shape = cell["shape_point"]
    variant = cell.get("variant_id")
    for workload_root in ("workloads/simbench_bw/results",
                          "workloads/portable_lcd/results"):
        base = REPO_ROOT / workload_root / slug / "rtx_3090_ti" / "sm_86"
        if variant:
            base = base / "variants" / variant
        d = base / f"trace_{shape}"
        if d.is_dir():
            return d
    return None


def _sidecar_for_trace_dir(trace_dir: Path) -> dict | None:
    sidecar_path = trace_dir.parent / f"{trace_dir.name}.json"
    if not sidecar_path.is_file():
        return None
    return json.loads(sidecar_path.read_text(encoding="utf-8"))


def _binary_for_cell(cell: dict) -> Path | None:
    """Resolve the cell's pinned binary via the proven sha-first
    resolver from the count-label capturer (handles the l1/l2/
    portable_lcd roots and register-cap build variants)."""
    from capture_count_labels import _binary_path_for

    return _binary_path_for(cell["slug"], cell.get("binary_sha256"))


def emit_for_cell(cell: dict, contract: dict) -> dict:
    """One cell's emission record: identity + per-metric emissions or a
    cell-level typed record. Never raises for per-cell conditions."""
    identity = {
        "arch": cell.get("arch"),
        "slug": cell["slug"],
        "shape_point": cell["shape_point"],
        "binary_sha256": cell.get("binary_sha256"),
        "kernel_signature_sha256": cell.get("kernel_signature_sha256"),
        "source_commit": cell.get("source_commit"),
    }
    if cell.get("variant_id"):
        identity["variant_id"] = cell["variant_id"]
    record: dict = {"identity": identity}
    trace_dir = _trace_dir_for_cell(cell)
    if trace_dir is None:
        record["cell_status"] = _typed(
            "trace_absent", "no trace directory on disk for this cell")
        return record
    sidecar = _sidecar_for_trace_dir(trace_dir)
    if sidecar is None:
        record["cell_status"] = _typed(
            "trace_sidecar_absent", f"no sidecar next to {trace_dir.name}")
        return record
    try:
        assert_closed_identity(cell, sidecar)
        assert_exact_support(trace_dir, sidecar)
    except ClosedIdentityError as exc:
        record["cell_status"] = _typed(exc.typed_reason, str(exc))
        return record
    except TraceSupportInsufficientError as exc:
        record["cell_status"] = _typed(exc.typed_reason, str(exc))
        return record
    try:
        walk = walk_trace_dir(trace_dir)
    except TraceSupportInsufficientError as exc:
        record["cell_status"] = _typed(exc.typed_reason, str(exc))
        return record
    opcode_by_pc = None
    if walk.get("excluded_op_class_pc_counts", {}).get("ATOMIC"):
        binary = _binary_for_cell(cell)
        if binary is not None:
            opcode_by_pc = sass_opcode_by_function_pc(binary)
    metrics = emit_metrics_for_walk(walk, contract, opcode_by_pc)
    record["cell_status"] = {"status": "emitted"}
    record["coverage"] = metrics.pop("_coverage")
    record["metrics"] = metrics
    record["trace_provenance"] = {
        "trace_dir": str(trace_dir.relative_to(REPO_ROOT)),
        "threadblock_files": walk["threadblock_files"],
        "trace_pb_sha256": sidecar.get("nvbit_trace_pb_sha256"),
    }
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-admitted",
        default="data/local_labels/label_admitted.json")
    parser.add_argument("--arch", default="sm_86")
    parser.add_argument("--out", default=DEFAULT_EMISSIONS_OUT)
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    try:
        assert_quarantine_path(out_path)
    except QuarantinePathError as exc:
        print(json.dumps({"typed_reason": exc.typed_reason,
                          "detail": str(exc)}), file=sys.stderr)
        return 3

    contract = load_contract()
    admitted = json.loads(
        (REPO_ROOT / args.label_admitted).read_text(encoding="utf-8"))
    cells = [c for c in admitted["cells"] if c.get("arch") == args.arch]
    records = []
    n_emitted = 0
    for cell in cells:
        record = emit_for_cell(cell, contract)
        records.append(record)
        if record["cell_status"].get("status") == "emitted":
            n_emitted += 1
        tag = record["cell_status"].get("status") or record[
            "cell_status"].get("typed_reason")
        print(f"[replay_counts] {cell['slug']}@{cell['shape_point']}"
              f"{('@' + cell['variant_id']) if cell.get('variant_id') else ''}"
              f": {tag}", flush=True)

    artifact = {
        "schema_version": 1,
        "artifact_kind": "replay_count_emissions",
        "arch": args.arch,
        "contract_build_binary_sha256": contract.get(
            "build", {}).get("binary_sha256"),
        "n_cells": len(records),
        "n_emitted": n_emitted,
        "n_typed": len(records) - n_emitted,
        "cells": records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"[replay_counts] wrote {out_path}: {n_emitted}/{len(records)} "
          f"cells emitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
