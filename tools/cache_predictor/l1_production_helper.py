"""Production proto-loader: drive the L1 walker on a real
``dynamic_trace.pb`` trace directory.

This module is the public production caller for
``l1_trace_adapter.build_per_block_l1_accesses(...)`` and
``cache_predictor.predict_l1_hit_rate_from_manifest(...)``.
It reads the kernel's grid / block / static-shared metadata
from ``dynamic_trace.pb``, walks the per-threadblock proto
files to extract per-block access records with active-lane
byte addresses, joins those records through the adapter, and
emits a schema-valid L1 predictor bundle.

The protobuf bindings under ``workloads/simbench_bw/_pb_generated/`` are
tracked authorities. Missing or unusable binding bytes are repository
corruption; a missing/incompatible runtime is a prerequisite failure. Both
raise hard dependency exceptions and are never converted into prediction
evidence. Missing trace metadata (grid /
block dims), adapter eligibility defects, or missing capacity
provenance all route to typed_unavailable bundles with typed
reasons + evidence shapes the L1 head accepts via the
existing ``_emit_l1_scalar_or_walker_fail_close`` plumbing.

The helper is intentionally narrow: it consumes a trace
directory + classifier manifest + provenance kwargs and
returns a bundle. It does not own the dynamic_trace.pb file's
discovery logic (the caller picks the file) and it does not
walk multiple kernels (the caller picks the kernel id). Both
concerns belong to the higher-level comparison-runner ingress
that the layer-aware comparison path will land in a follow-up
slice.

KNOWN DEFECTS (2026-07-06, recorded; fixes tracked in the
label-pipeline backlog): (1) the consumed classifier manifests
are cross-arch contaminated on multi-arch fatbins — see the
KNOWN DEFECT note in ``l1_cache_op_classifier``'s module
docstring. (2) This module joins the manifest by ``vpc`` when
present, but manifest pcs are function-relative, so when the
function base is nonzero the join never matches and records
are silently dropped. (3) It applies the FIRST nonzero
address-record ``data_width`` to ALL of an instruction's
lanes; the wire contract is per-record width. Both (2) and
(3) diverge from the wire contract — the Rust replay uses
function-relative pc + per-record width priority deliberately.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


from l1_trace_adapter import (  # noqa: E402
    L1TraceAdapterDataWidthInvalid,
    L1TraceAdapterEligibilityDefect,
    L1TraceAdapterGridBlockMissing,
    L1TraceAdapterManifestNotMatched,
    TraceRecord,
    build_per_block_l1_accesses,
)


_REPO_ROOT = _HERE.parent.parent
_PB_GENERATED_DIR = (
    _REPO_ROOT / "workloads" / "simbench_bw" / "_pb_generated"
)
_CACHE_PREDICTOR_DIR = (
    _REPO_ROOT / "workloads" / "simbench_bw"
)
_IMPORT_TRACE_CONTRACT_PATH = (
    _REPO_ROOT / "tools" / "nvbit_tracer" / "import_trace_contract.py"
)


class L1ProductionHelperProtobufBindingsMissing(Exception):
    """Compatibility base for hard protobuf dependency failures."""

    TOKEN = "predictor_l1_sass_provenance_missing"


class L1ProductionHelperTrackedBindingsCorrupt(
    L1ProductionHelperProtobufBindingsMissing
):
    """A tracked binding authority is absent, damaged, or misresolved."""


class L1ProductionHelperRuntimePrerequisiteMissing(
    L1ProductionHelperProtobufBindingsMissing
):
    """The project-pinned Python protobuf runtime is unavailable."""


class L1ProductionHelperThreadblockMalformed(Exception):
    """Raised when a per-threadblock proto file has no
    ``block_id`` or carries an out-of-range block_index for
    the kernel's grid. The L1 head catches this to emit
    ``predictor_l1_cta_to_sm_schedule_unknown`` with structured
    evidence — silently skipping the file would falsely route
    to ``predictor_l1_no_modeled_l1_eligible_accesses`` as if
    the kernel had nothing to model."""

    TOKEN = "predictor_l1_cta_to_sm_schedule_unknown"


class L1ProductionHelperAddressMalformed(Exception):
    """Raised when an address record fails structural
    reconstruction via ``_reconstruct_lane_vector``. The L1
    head catches this to emit
    ``predictor_l1_sass_provenance_missing`` with
    ``evidence.address_record_malformed`` naming the format
    and diagnostic — silently dropping the record would
    underreport accesses and bias the prediction."""

    TOKEN = "predictor_l1_sass_provenance_missing"


class L1ProductionHelperInstructionMalformed(Exception):
    """Raised when a memory instruction (i.e. an instruction
    that carries one or more address records) is missing a
    wire-contract field that the trace contract treats as
    invalid: ``memory_op_class`` (Codex R59 P2 finding 1) or
    ``active_mask`` / ``predicate_mask`` (Codex R59 P2
    finding 2). The L1 head catches this to emit
    ``predictor_l1_sass_provenance_missing`` with
    ``evidence.wire_contract_violation`` naming the missing
    field and the offending (warp, instruction index, pc) —
    silently skipping the instruction would underreport L1
    accesses and bias the scalar hit-rate prediction, OR
    falsely route to
    ``predictor_l1_no_modeled_l1_eligible_accesses`` as if
    the kernel had no eligible accesses to model."""

    TOKEN = "predictor_l1_sass_provenance_missing"


def _load_proto_bindings() -> tuple[Any, Any, Any]:
    """Return the (trace_pb2, threadblock_pb2, instruction_pb2)
    tracked bindings. Dependency failures distinguish repository corruption
    from a missing runtime prerequisite and must propagate to the caller."""
    try:
        import google.protobuf  # noqa: F401
    except ImportError as exc:
        raise L1ProductionHelperRuntimePrerequisiteMissing(
            "protobuf runtime prerequisite is missing; install the exact "
            "version pinned in requirements.txt"
        ) from exc
    names = (
        "address_pb2.py", "cuda_stream_pb2.py", "dim3d_pb2.py",
        "gpu_device_pb2.py", "instruction_pb2.py", "kernel_pb2.py",
        "threadblock_pb2.py", "trace_pb2.py", "warp_pb2.py",
    )
    missing = [
        name
        for name in names
        if not (_PB_GENERATED_DIR / name).is_file()
        or (_PB_GENERATED_DIR / name).is_symlink()
    ]
    if missing:
        relative = " ".join(
            f"workloads/simbench_bw/_pb_generated/{name}" for name in missing
        )
        raise L1ProductionHelperTrackedBindingsCorrupt(
            "tracked protobuf authority is missing; restore byte-exact "
            "committed content with `git restore --source=HEAD --worktree -- "
            f"{relative}`; do not regenerate bindings in the canonical checkout"
        )
    if str(_PB_GENERATED_DIR) not in sys.path:
        sys.path.insert(0, str(_PB_GENERATED_DIR))
    try:
        import trace_pb2  # type: ignore[import-not-found]
        import threadblock_pb2  # type: ignore[import-not-found]
        import instruction_pb2  # type: ignore[import-not-found]
    except Exception as exc:
        if exc.__class__.__module__.startswith("google.protobuf"):
            raise L1ProductionHelperRuntimePrerequisiteMissing(
                "protobuf runtime prerequisite is incompatible with the "
                f"tracked bindings: {type(exc).__name__}: {exc}"
            ) from exc
        raise L1ProductionHelperTrackedBindingsCorrupt(
            "tracked protobuf authorities are unusable; restore their "
            "byte-exact committed content with `git restore --source=HEAD "
            "--worktree -- workloads/simbench_bw/_pb_generated`; "
            f"do not regenerate them: {type(exc).__name__}: {exc}"
        ) from exc
    for module, filename in (
        (trace_pb2, "trace_pb2.py"),
        (threadblock_pb2, "threadblock_pb2.py"),
        (instruction_pb2, "instruction_pb2.py"),
    ):
        module_file = getattr(module, "__file__", None)
        if module_file is None or Path(module_file).resolve() != (
            _PB_GENERATED_DIR / filename
        ).resolve():
            raise L1ProductionHelperTrackedBindingsCorrupt(
                f"tracked protobuf authority {module.__name__} resolved "
                f"outside {_PB_GENERATED_DIR}: {getattr(module, '__file__', None)}"
            )
    for filename in names:
        module = sys.modules.get(filename.removesuffix(".py"))
        if module is None:
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None or Path(module_file).resolve() != (
            _PB_GENERATED_DIR / filename
        ).resolve():
            raise L1ProductionHelperTrackedBindingsCorrupt(
                f"tracked protobuf authority {module.__name__} resolved "
                f"outside {_PB_GENERATED_DIR}: {getattr(module, '__file__', None)}"
            )
    return trace_pb2, threadblock_pb2, instruction_pb2


def _load_cache_predictor() -> Any:
    """Lazy-import the simbench cache_predictor module.

    Inserts ``workloads/simbench_bw/`` on sys.path so the import
    resolves to the simbench module (NOT the
    ``tools/cache_predictor/`` directory which shares the bare
    ``cache_predictor`` name). The path is inserted at index 0
    so it takes precedence over any sibling directory that
    happens to be in front of it."""
    if str(_CACHE_PREDICTOR_DIR) not in sys.path:
        sys.path.insert(0, str(_CACHE_PREDICTOR_DIR))
    # If a stale `cache_predictor` module (the tools-side
    # package) is already cached, evict it first.
    cached = sys.modules.get("cache_predictor")
    if cached is not None and not hasattr(cached, "_TYPED_REASONS"):
        del sys.modules["cache_predictor"]
    import cache_predictor  # type: ignore[import-not-found]
    return cache_predictor


def _load_import_trace_contract() -> Any:
    """Lazy-import the authoritative address-reconstruction
    helper at ``tools/nvbit_tracer/import_trace_contract.py``.
    Registers the module in ``sys.modules`` before
    ``exec_module`` so the dataclass-init machinery can
    resolve the module's ``__module__`` attribute back to a
    real module (Python's dataclass loader otherwise fails
    when a module is loaded via spec without a sys.modules
    registration)."""
    module_name = "_import_trace_contract_for_l1_production"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(
        module_name,
        _IMPORT_TRACE_CONTRACT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _find_kernel_record(
    trace_pb2_mod: Any, trace_pb_path: Path, kernel_id: int,
) -> Any | None:
    """Return the kernel record with the requested id, or None
    when the trace contains no matching kernel."""
    trace = trace_pb2_mod.Trace()
    trace.ParseFromString(trace_pb_path.read_bytes())
    for _device_id in sorted(trace.gpu_device.keys()):
        device = trace.gpu_device[_device_id]
        for _stream_id in sorted(device.streams.keys()):
            stream = device.streams[_stream_id]
            for kernel in stream.kernels:
                if int(kernel.id) == kernel_id:
                    return kernel
    return None


def _flatten_block_id(
    tb_x: int, tb_y: int, tb_z: int,
    *,
    grid_x: int, grid_y: int,
) -> int:
    """Map a 3-D threadblock coordinate to a 1-D block_index.

    Uses CUDA's row-major convention: ``block_index = tb_x +
    tb_y * grid_x + tb_z * grid_x * grid_y``. The convention
    matches the walker's per-block SM-assignment policy."""
    return tb_x + tb_y * grid_x + tb_z * grid_x * grid_y


def _reconstruct_lane_addresses(
    address_record: Any,
    and_mask: int,
    import_trace_contract_mod: Any,
) -> list[int]:
    """Reconstruct active-lane byte addresses for one address
    record using the authoritative
    ``_reconstruct_lane_vector`` decoder.

    Computes the lane-vector via the canonical decoder, then
    filters inactive-lane sentinels. Raises
    ``L1ProductionHelperAddressMalformed`` when the decoder
    returns the structural-failure arm (e.g., base_stride
    record with non-empty addrs); silently dropping malformed
    records would underreport accesses and bias the
    prediction.
    """
    vector, reason = (
        import_trace_contract_mod._reconstruct_lane_vector(
            address_record, and_mask,
        )
    )
    if vector is None:
        raise L1ProductionHelperAddressMalformed(
            f"address record malformed: format="
            f"{address_record.compression_format}; "
            f"reason={reason!r}"
        )
    sentinel = (
        import_trace_contract_mod.LANE_INACTIVE_SENTINEL
    )
    return [
        int(addr) for addr in vector
        if int(addr) != sentinel
    ]


def _block_index_from_message(
    tb_message: Any,
    *,
    grid_x: int,
    grid_y: int,
    grid_dim: int,
    tb_path: Path,
) -> int:
    """Return the threadblock's flattened block_index from the
    proto's canonical ``block_id`` field.

    Fail-close paths:
      - The producer must write ``block_id`` per
        ``threadblock.proto`` (it is a required dim3d field on
        the proto3 schema; ``HasField`` reports True when the
        wire carries a non-default message). When the proto
        does not carry block_id at all, raise
        ``L1ProductionHelperThreadblockMalformed`` naming the
        file path.
      - When the flattened index is outside ``[0, grid_dim)``
        raise the same exception with the offending coord +
        computed index in the message.

    The filename (e.g., ``d_0_s_0_k_1_x,y,z.pb`` from the real
    tracer, or any other shape from a synthetic test fixture)
    is treated as diagnostic-only; the proto field is the
    canonical source of truth.
    """
    if not tb_message.HasField("block_id"):
        raise L1ProductionHelperThreadblockMalformed(
            f"threadblock proto at {tb_path.name!r} is missing "
            f"block_id (proto3 explicit-presence reports "
            f"absent); the producer MUST populate block_id per "
            f"the threadblock.proto contract"
        )
    tb_x = int(tb_message.block_id.x)
    tb_y = int(tb_message.block_id.y)
    tb_z = int(tb_message.block_id.z)
    block_index = _flatten_block_id(
        tb_x, tb_y, tb_z,
        grid_x=grid_x, grid_y=grid_y,
    )
    if block_index < 0 or block_index >= grid_dim:
        raise L1ProductionHelperThreadblockMalformed(
            f"threadblock proto at {tb_path.name!r} has "
            f"block_id=({tb_x},{tb_y},{tb_z}) flattened to "
            f"index {block_index}, which is outside the kernel's "
            f"grid range [0, {grid_dim})"
        )
    return block_index


def _walk_threadblock_protos(
    threadblock_pb2_mod: Any,
    threadblock_pb_paths: list[Path],
    import_trace_contract_mod: Any,
    *,
    grid_x: int,
    grid_y: int,
    grid_dim: int,
) -> list[TraceRecord]:
    """Walk all per-threadblock proto files and emit
    TraceRecord entries for each memory instruction.

    Block coordinates come from ``threadblock.block_id``
    (the canonical proto field) — NOT from the filename. An
    unparseable file (missing block_id, or block_index
    out-of-range) raises
    ``L1ProductionHelperThreadblockMalformed``; the L1 head
    catches and emits a typed_unavailable bundle. Silent
    skips would falsely route to
    ``predictor_l1_no_modeled_l1_eligible_accesses`` as if
    the kernel had nothing to model.

    Address records use the authoritative
    ``_reconstruct_lane_vector`` decoder; malformed records
    raise ``L1ProductionHelperAddressMalformed``.
    """
    records: list[TraceRecord] = []
    for tb_path in threadblock_pb_paths:
        tb_message = threadblock_pb2_mod.threadblock()
        tb_message.ParseFromString(tb_path.read_bytes())
        block_index = _block_index_from_message(
            tb_message,
            grid_x=grid_x,
            grid_y=grid_y,
            grid_dim=grid_dim,
            tb_path=tb_path,
        )
        for warp_index in sorted(tb_message.warps.keys()):
            warp = tb_message.warps[warp_index]
            for inst_index, inst in enumerate(warp.instructions):
                # Codex R59 code-review P2: when a memory
                # instruction (one or more `addresses` records)
                # is missing a wire-contract field
                # (memory_op_class, active_mask, predicate_mask),
                # raise a typed wire-contract violation rather
                # than silently skipping the instruction. The
                # prior silent-skip behavior could either
                # underreport L1 accesses (biasing the scalar
                # hit-rate) or falsely route to
                # `predictor_l1_no_modeled_l1_eligible_accesses`
                # as if the kernel had no eligible accesses to
                # model. The trace contract treats these as
                # malformed; the L1 head should fail-close.
                if not inst.addresses:
                    # No addresses -> not a memory instruction;
                    # missing memory_op_class / lane masks is
                    # not a wire-contract violation for non-
                    # memory instructions.
                    continue
                for required_field in (
                    "memory_op_class",
                    "active_mask",
                    "predicate_mask",
                ):
                    if not inst.HasField(required_field):
                        raise L1ProductionHelperInstructionMalformed(
                            f"memory instruction at "
                            f"tb={tb_path.name}/warp={warp_index}"
                            f"/inst_index={inst_index}/pc={inst.pc} "
                            f"has {len(inst.addresses)} address "
                            f"record(s) but is missing the "
                            f"`{required_field}` field; the wire "
                            f"contract treats this as invalid for "
                            f"memory instructions"
                        )
                active_mask = inst.active_mask
                predicate_mask = inst.predicate_mask
                and_mask = active_mask & predicate_mask
                if and_mask == 0:
                    continue
                pc_value = (
                    inst.vpc if inst.HasField("vpc") else inst.pc
                )
                # Resolve the per-lane byte width from the proto.
                # Source priority (per the sector-with-line-
                # allocation contract):
                #   1. Any address_record's ``data_width`` field
                #      when non-zero (proto3 scalar default = 0
                #      is treated as "not emitted").
                #   2. ``inst.data_width`` when explicitly
                #      present.
                # A non-positive width means the tracer emitted
                # a contract violation; the L1 head's typed
                # fail-close arm catches this and rejects the
                # bundle with ``predictor_l1_sass_provenance_missing``
                # rather than silently undercounting the sector
                # denominator.
                data_width: int | None = None
                for address_record in inst.addresses:
                    width = int(address_record.data_width)
                    if width != 0:
                        data_width = width
                        break
                if data_width is None and inst.HasField("data_width"):
                    data_width = int(inst.data_width)
                byte_addresses: list[int] = []
                for address_record in inst.addresses:
                    byte_addresses.extend(
                        _reconstruct_lane_addresses(
                            address_record,
                            and_mask,
                            import_trace_contract_mod,
                        )
                    )
                if not byte_addresses:
                    continue
                records.append(TraceRecord(
                    block_index=block_index,
                    pc=f"0x{int(pc_value):x}",
                    byte_addresses=tuple(byte_addresses),
                    data_width=data_width,
                    warp_index=int(warp_index),
                ))
    return records


def predict_l1_from_trace_dir(
    *,
    trace_pb_path: Path,
    threadblock_pb_paths: list[Path],
    kernel_id: int,
    classifier_manifest: dict,
    sm_count: int,
    l1tex_capacity_source: str,
    l1_shared_carveout_source: str,
    arch_token: str,
    manifest_pairing_index: dict,
    input_sidecar_digests: dict,
    device: str,
    cache_model_capacity_bytes: int,
    bootstrap_seed: int = 0,
    epsilon: float = 0.01,
    cta_to_sm_schedule_policy: str = "round_robin_homogeneous",
    occupancy_derived_assignment: list[int] | None = None,
    carveout_max_l1_bytes: int | None = None,
    default_l1_bytes: int | None = None,
    dynamic_shared_bytes: int = 0,
    warmup_launches: int | None = None,
    carveout_percent: int | None = None,
) -> dict:
    """Drive the L1 walker on a real trace directory and return
    a schema-valid L1 predictor bundle.

    Arguments:
      trace_pb_path: path to the ``dynamic_trace.pb`` file.
      threadblock_pb_paths: ordered list of per-threadblock
        proto files for the target kernel.
      kernel_id: the kernel id within the trace; the helper
        extracts grid/block/static-shared metadata from this
        kernel record.
      classifier_manifest: a matched cache-op manifest.
      sm_count: per-arch SM count.
      l1tex_capacity_source / l1_shared_carveout_source /
      arch_token / manifest_pairing_index /
      input_sidecar_digests / device /
      cache_model_capacity_bytes / bootstrap_seed / epsilon:
      see ``predict_l1_hit_rate_from_manifest``.
      cta_to_sm_schedule_policy: the walker's policy
        (defaults to round_robin_homogeneous).
      occupancy_derived_assignment: per-block sm_id list when
        policy is occupancy_derived.
      carveout_max_l1_bytes / default_l1_bytes: passed through
        to the L1 head; required when l1tex_capacity_source is
        non-static and the carveout source is default.
      dynamic_shared_bytes: per-kernel dynamic shared usage
        (the proto does not record this; the caller must
        supply it from the launch parameters).

    Returns a schema-valid bundle from
    ``predict_l1_hit_rate_from_manifest``."""
    cp = _load_cache_predictor()

    # Pre-adapter short-circuit: when the classifier manifest's
    # sass_provenance_status is ``matched_no_load_class_instructions``
    # the kernel parsed cleanly but has no LDG/LDGSTS/LDL/STL/ATOM/TEX
    # opcodes for the L1 model to walk. Delegate directly to
    # ``predict_l1_hit_rate_from_manifest``, whose documented
    # behavior is to emit ``predictor_l1_no_modeled_l1_eligible_accesses``
    # with ``cache_layer=L1TEX``. Routing this case through the
    # adapter would raise ``L1TraceAdapterManifestNotMatched`` and
    # mis-route the bundle to ``predictor_l1_sass_provenance_missing``.
    if (
        classifier_manifest.get("sass_provenance_status")
        == "matched_no_load_class_instructions"
    ):
        return cp.predict_l1_hit_rate_from_manifest(
            cache_op_manifest=classifier_manifest,
            manifest_pairing_index=manifest_pairing_index,
            input_sidecar_digests=input_sidecar_digests,
            device=device,
            cache_model_capacity_bytes=cache_model_capacity_bytes,
            bootstrap_seed=bootstrap_seed,
            epsilon=epsilon,
        )

    trace_pb2_mod, threadblock_pb2_mod, _instruction_pb2_mod = (
        _load_proto_bindings()
    )

    kernel = _find_kernel_record(
        trace_pb2_mod, trace_pb_path, kernel_id,
    )
    if kernel is None:
        return cp._typed_unavailable_bundle(
            cache_layer="L1TEX",
            typed_reason="predictor_l1_cta_to_sm_schedule_unknown",
            manifest_pairing_index=manifest_pairing_index,
            input_sidecar_digests=input_sidecar_digests,
            device=device,
            cache_model_capacity_bytes=cache_model_capacity_bytes,
            bootstrap_seed=bootstrap_seed,
            epsilon=epsilon,
            evidence={
                "kernel_id_not_found_in_trace": kernel_id,
                "trace_pb_path": str(trace_pb_path),
            },
        )

    grid_x = int(kernel.grid_dim.x) if kernel.HasField("grid_dim") else 0
    grid_y = int(kernel.grid_dim.y) if kernel.HasField("grid_dim") else 0
    grid_z = int(kernel.grid_dim.z) if kernel.HasField("grid_dim") else 0
    block_x = (
        int(kernel.block_dim.x) if kernel.HasField("block_dim") else 0
    )
    block_y = (
        int(kernel.block_dim.y) if kernel.HasField("block_dim") else 0
    )
    block_z = (
        int(kernel.block_dim.z) if kernel.HasField("block_dim") else 0
    )
    grid_dim = grid_x * grid_y * grid_z
    block_dim = block_x * block_y * block_z

    if grid_dim <= 0 or block_dim <= 0:
        return cp._typed_unavailable_bundle(
            cache_layer="L1TEX",
            typed_reason="predictor_l1_cta_to_sm_schedule_unknown",
            manifest_pairing_index=manifest_pairing_index,
            input_sidecar_digests=input_sidecar_digests,
            device=device,
            cache_model_capacity_bytes=cache_model_capacity_bytes,
            bootstrap_seed=bootstrap_seed,
            epsilon=epsilon,
            evidence={
                "grid_dim_total": grid_dim,
                "block_dim_total": block_dim,
                "grid_dim_xyz": [grid_x, grid_y, grid_z],
                "block_dim_xyz": [block_x, block_y, block_z],
            },
        )

    static_shared_bytes = int(kernel.size_shared_memory)

    # Fail-close on missing dynamic evidence: a matched manifest
    # with eligible PCs but ZERO threadblock proto paths means
    # the capture failed (the `threadblocks/` directory is empty
    # or absent), NOT that the kernel has no modeled L1 accesses.
    # Returning an empty trace-record list here would silently
    # route the bundle to
    # `predictor_l1_no_modeled_l1_eligible_accesses` and the
    # downstream walker would compute hit_rate=0.0 over zero
    # accesses, masking the capture failure as a real prediction.
    # Emit a typed_unavailable bundle naming the missing dynamic
    # evidence so reviewers can trace the capture-side defect.
    if not threadblock_pb_paths:
        return cp._typed_unavailable_bundle(
            cache_layer="L1TEX",
            typed_reason="predictor_l1_sass_provenance_missing",
            manifest_pairing_index=manifest_pairing_index,
            input_sidecar_digests=input_sidecar_digests,
            device=device,
            cache_model_capacity_bytes=cache_model_capacity_bytes,
            bootstrap_seed=bootstrap_seed,
            epsilon=epsilon,
            evidence={
                "threadblock_pb_paths_empty": True,
                "grid_dim_total": grid_dim,
                "trace_pb_path": str(trace_pb_path),
                "detail": (
                    "matched manifest with eligible PCs but no "
                    "threadblock_pb_paths supplied; dynamic trace "
                    "evidence is missing. This is a capture-side "
                    "defect, not a kernel-without-loads condition."
                ),
            },
        )

    try:
        import_trace_contract_mod = _load_import_trace_contract()
        trace_records = _walk_threadblock_protos(
            threadblock_pb2_mod,
            threadblock_pb_paths,
            import_trace_contract_mod,
            grid_x=grid_x, grid_y=grid_y, grid_dim=grid_dim,
        )
        per_block_accesses = build_per_block_l1_accesses(
            trace_records,
            classifier_manifest,
            sm_count=sm_count,
            grid_dim=grid_dim,
            block_dim=block_dim,
            # Production traces MUST carry a positive data_width
            # on every memory instruction so the adapter can
            # expand into per-sector walker entries. Synthetic
            # fixtures may still use the legacy line-level path
            # by leaving data_width=None; production cannot.
            require_data_width=True,
        )
    except L1ProductionHelperThreadblockMalformed as exc:
        return cp._typed_unavailable_bundle(
            cache_layer="L1TEX",
            typed_reason="predictor_l1_cta_to_sm_schedule_unknown",
            manifest_pairing_index=manifest_pairing_index,
            input_sidecar_digests=input_sidecar_digests,
            device=device,
            cache_model_capacity_bytes=cache_model_capacity_bytes,
            bootstrap_seed=bootstrap_seed,
            epsilon=epsilon,
            evidence={
                "threadblock_malformed": str(exc),
            },
        )
    except L1ProductionHelperAddressMalformed as exc:
        return cp._typed_unavailable_bundle(
            cache_layer="L1TEX",
            typed_reason="predictor_l1_sass_provenance_missing",
            manifest_pairing_index=manifest_pairing_index,
            input_sidecar_digests=input_sidecar_digests,
            device=device,
            cache_model_capacity_bytes=cache_model_capacity_bytes,
            bootstrap_seed=bootstrap_seed,
            epsilon=epsilon,
            evidence={
                "address_record_malformed": str(exc),
            },
        )
    except L1ProductionHelperInstructionMalformed as exc:
        return cp._typed_unavailable_bundle(
            cache_layer="L1TEX",
            typed_reason="predictor_l1_sass_provenance_missing",
            manifest_pairing_index=manifest_pairing_index,
            input_sidecar_digests=input_sidecar_digests,
            device=device,
            cache_model_capacity_bytes=cache_model_capacity_bytes,
            bootstrap_seed=bootstrap_seed,
            epsilon=epsilon,
            evidence={
                "wire_contract_violation": str(exc),
            },
        )
    except L1TraceAdapterGridBlockMissing as exc:
        return cp._typed_unavailable_bundle(
            cache_layer="L1TEX",
            typed_reason="predictor_l1_cta_to_sm_schedule_unknown",
            manifest_pairing_index=manifest_pairing_index,
            input_sidecar_digests=input_sidecar_digests,
            device=device,
            cache_model_capacity_bytes=cache_model_capacity_bytes,
            bootstrap_seed=bootstrap_seed,
            epsilon=epsilon,
            evidence={
                "adapter_grid_block_missing": str(exc),
            },
        )
    except L1TraceAdapterManifestNotMatched as exc:
        return cp._typed_unavailable_bundle(
            cache_layer="L1TEX",
            typed_reason="predictor_l1_sass_provenance_missing",
            manifest_pairing_index=manifest_pairing_index,
            input_sidecar_digests=input_sidecar_digests,
            device=device,
            cache_model_capacity_bytes=cache_model_capacity_bytes,
            bootstrap_seed=bootstrap_seed,
            epsilon=epsilon,
            evidence={
                "adapter_manifest_not_matched": str(exc),
            },
        )
    except L1TraceAdapterEligibilityDefect as exc:
        return cp._typed_unavailable_bundle(
            cache_layer="L1TEX",
            typed_reason="predictor_l1_sass_provenance_missing",
            manifest_pairing_index=manifest_pairing_index,
            input_sidecar_digests=input_sidecar_digests,
            device=device,
            cache_model_capacity_bytes=cache_model_capacity_bytes,
            bootstrap_seed=bootstrap_seed,
            epsilon=epsilon,
            evidence={
                "adapter_eligibility_defect": exc.to_evidence(),
            },
        )
    except L1TraceAdapterDataWidthInvalid as exc:
        return cp._typed_unavailable_bundle(
            cache_layer="L1TEX",
            typed_reason="predictor_l1_sass_provenance_missing",
            manifest_pairing_index=manifest_pairing_index,
            input_sidecar_digests=input_sidecar_digests,
            device=device,
            cache_model_capacity_bytes=cache_model_capacity_bytes,
            bootstrap_seed=bootstrap_seed,
            epsilon=epsilon,
            evidence={
                "adapter_data_width_invalid": exc.to_evidence(),
            },
        )

    walker_inputs: dict = {
        "per_block_accesses": per_block_accesses,
        "sm_count": sm_count,
        "unified_budget_bytes": cache_model_capacity_bytes,
        "static_shared_bytes": static_shared_bytes,
        "dynamic_shared_bytes": dynamic_shared_bytes,
        "l1tex_capacity_source": l1tex_capacity_source,
        "l1_shared_carveout_source": l1_shared_carveout_source,
        "cta_to_sm_schedule_policy": cta_to_sm_schedule_policy,
        "arch_token": arch_token,
    }
    if occupancy_derived_assignment is not None:
        walker_inputs["occupancy_derived_assignment"] = (
            occupancy_derived_assignment
        )
    if carveout_max_l1_bytes is not None:
        walker_inputs["carveout_max_l1_bytes"] = carveout_max_l1_bytes
    if default_l1_bytes is not None:
        walker_inputs["default_l1_bytes"] = default_l1_bytes
    if warmup_launches is not None:
        walker_inputs["warmup_launches"] = warmup_launches
    if carveout_percent is not None:
        walker_inputs["carveout_percent"] = carveout_percent

    return cp.predict_l1_hit_rate_from_manifest(
        cache_op_manifest=classifier_manifest,
        manifest_pairing_index=manifest_pairing_index,
        input_sidecar_digests=input_sidecar_digests,
        device=device,
        cache_model_capacity_bytes=cache_model_capacity_bytes,
        bootstrap_seed=bootstrap_seed,
        epsilon=epsilon,
        l1_walker_inputs=walker_inputs,
    )


__all__ = [
    "L1ProductionHelperAddressMalformed",
    "L1ProductionHelperInstructionMalformed",
    "L1ProductionHelperProtobufBindingsMissing",
    "L1ProductionHelperRuntimePrerequisiteMissing",
    "L1ProductionHelperThreadblockMalformed",
    "L1ProductionHelperTrackedBindingsCorrupt",
    "predict_l1_from_trace_dir",
]
