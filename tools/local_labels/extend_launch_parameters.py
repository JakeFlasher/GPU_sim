"""Build per-cell launch_parameters_sidecar.json by joining the
admitted-cell label index, the per-cell dynamic_trace.pb
(carrying kernel-level launch geometry: grid, block,
static_shared, regs_per_thread), and the auxiliary covariate
sidecars (CUPTI Activity / NVML covariate / phase cycle ledger)
where they exist.

The sidecar is a per-cell intermediate artifact that the
candidate-feature catalog generator consumes; it is not itself
the load-bearing contract artifact (that is the catalog at
``data/sm_86_predictor_accuracy_recovery_v2/ac6_missing_feature_candidates.json``).

The 5 named candidate covariates the downstream-calibration
plan should evaluate are:

  - per_warp_occupancy:  derived from
      ``block.x * block.y * block.z`` and the architecture's
      warp size; sidecar source is the kernel.proto
      ``block_dim`` field.
  - blockDim:            from kernel.proto ``block_dim``.
  - gridDim:             from kernel.proto ``grid_dim``.
  - dvfs_state:          from nvml_covariate samples
      (sm_clock_mhz / mem_clock_mhz / throttle_reasons);
      label-provenance-only (NEVER a predictor input under
      the no-DVFS-as-predictor-input directive).
  - mshr_pressure_proxy: derived from per-kernel
      memcpy-overlap + register-pressure heuristics
      observed in CUPTI Activity records;
      label-provenance-only when not collected.

Inputs:
  - ``data/local_labels/label_admitted.json`` (the operator-
    facing admitted-cell index; filtered to ``arch=sm_86``).
  - For each admitted cell, the trace directory at
    ``workloads/simbench_bw/results/<slug>/rtx_3090_ti/sm_86/trace_<shape_point>/``
    + the ``dynamic_trace.pb`` it carries.
  - Optional auxiliary sidecars under the same trace directory
    or under the prior live-sweep per-repeat dirs; absent
    sidecars are tolerated and emit
    ``availability_at_prediction_time=not_collected_for_cell``.

Outputs:
  - One ``launch_parameters_sidecar_<slug>__<shape_point>.json``
    per admitted (slug, shape_point) at the documented
    host-local cache directory
    ``data/local_labels/launch_parameters_sidecars/sm_86/``.

Fail-close arms (typed exceptions):
  - ``LaunchParametersTraceMissing`` when an admitted cell's
    dynamic_trace.pb is not found.
  - ``LaunchParametersTrackedBindingsCorrupt`` when a tracked `_pb2.py`
    authority is absent or unusable.
  - ``LaunchParametersRuntimePrerequisiteMissing`` when the project-pinned
    google.protobuf runtime is unavailable or incompatible.
  - ``LaunchParametersKernelIdNotFound`` when the trace
    parses but contains no kernel records (a malformed-trace
    case).
  - ``LaunchParametersAmbiguousJoin`` when multiple
    auxiliary-sidecar candidates exist with conflicting
    join keys (e.g., two CUPTI Activity files whose
    ``start_ns`` disagree).
  - ``LaunchParametersJoinKeyMismatch`` when a discovered
    sidecar's join key does not match the cell's identity
    or the kernel-proto kernel record (e.g., CUPTI
    Activity ``kernel_correlation_id`` does not equal the
    kernel proto ``kernel.id``).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_DEFAULT_LABEL_ADMITTED = (
    _REPO_ROOT / "data" / "local_labels" / "label_admitted.json"
)
_DEFAULT_OUT_DIR = (
    _REPO_ROOT
    / "data"
    / "local_labels"
    / "launch_parameters_sidecars"
    / "sm_86"
)
_DEFAULT_ARCH_DIR = "rtx_3090_ti/sm_86"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.nvbit_tracer.proto_binding_loader import (  # noqa: E402
    ProtoBindingIntegrityError,
    ProtoBindingRuntimePrerequisiteError,
    load_dynamic_trace_bindings,
)

# Arch-token policy boundary mapping for the cohort sidecar emitter.
# Imported lazily inside the helper to keep the module-load-time
# import surface small (the arch_token_policy module is intentionally
# a thin pure-Python pin so this lazy import is safe).
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import arch_token_policy as _atp  # noqa: E402

_ADMITTED_LABEL_ARCHES: tuple[str, ...] = tuple(sorted(_atp.SUPPORTED_LABEL_ARCHES))


def _out_dir_for_label_arch(arch: str) -> Path:
    """Per-arch output directory for the cohort sidecar emitter. Uses
    the shared ``arch_token_policy.sidecar_token_for_label_arch`` map
    so the sm_120 cells route to
    ``data/local_labels/launch_parameters_sidecars/sm_120-laptop/``
    instead of the default sm_86 directory. Raises
    :class:`arch_token_policy.UnsupportedLabelArchError` on unknown
    arch."""
    return (
        _REPO_ROOT
        / "data"
        / "local_labels"
        / "launch_parameters_sidecars"
        / _atp.sidecar_token_for_label_arch(arch)
    )


def _arch_dir_for_label_arch(arch: str) -> str:
    """Per-arch trace/results platform-path segment used to locate the
    per-cell trace directory under
    ``workloads/simbench_bw/results/<slug>/<sku>/<arch>/``. Uses the
    shared ``arch_token_policy.platform_path_for_label_arch`` map."""
    return _atp.platform_path_for_label_arch(arch)
_AMPERE_WARP_SIZE = 32


class LaunchParametersTraceMissing(Exception):
    """Raised when an admitted cell's dynamic_trace.pb is
    missing or cannot be read."""


class LaunchParametersProtobufBindingsMissing(Exception):
    """Compatibility base for hard protobuf dependency failures."""


class LaunchParametersTrackedBindingsCorrupt(
    LaunchParametersProtobufBindingsMissing
):
    """A tracked `_pb2.py` authority is absent, damaged, or misresolved."""


class LaunchParametersRuntimePrerequisiteMissing(
    LaunchParametersProtobufBindingsMissing
):
    """The project-pinned Python protobuf runtime is unavailable."""


class LaunchParametersKernelIdNotFound(Exception):
    """Raised when the parsed trace contains no kernel
    records (a malformed trace surface)."""


class LaunchParametersAmbiguousJoin(Exception):
    """Raised when multiple auxiliary sidecar candidates exist
    for a cell with conflicting join keys (e.g., two CUPTI
    Activity files whose start_ns / kernel_correlation_id
    disagree). The four-surface join cannot pick a winner
    without an explicit operator decision; fail-close instead
    of silently choosing one."""


class LaunchParametersJoinKeyMismatch(Exception):
    """Raised when a discovered auxiliary sidecar's join key
    does not match the cell's kernel-proto kernel record
    (e.g., a CUPTI Activity file whose
    `kernel_correlation_id` does not match the kernel proto
    `kernel.id`, or whose `slug`/`shape_point` disagrees
    with the cell's identity)."""


@dataclass(frozen=True)
class LaunchGeometry:
    """Per-kernel launch geometry extracted from the
    dynamic_trace.pb ``kernel`` message."""

    kernel_id: int
    grid_x: int
    grid_y: int
    grid_z: int
    block_x: int
    block_y: int
    block_z: int
    static_shared_bytes: int
    regs_per_thread: int


def _load_trace_bindings() -> Any:
    """Return the centrally authenticated ``trace_pb2`` authority."""
    try:
        return load_dynamic_trace_bindings().trace_pb2
    except ProtoBindingRuntimePrerequisiteError as exc:
        raise LaunchParametersRuntimePrerequisiteMissing(str(exc)) from exc
    except ProtoBindingIntegrityError as exc:
        raise LaunchParametersTrackedBindingsCorrupt(str(exc)) from exc


def _extract_first_kernel_geometry(
    trace_pb_path: Path,
    trace_pb2_mod: Any,
) -> LaunchGeometry:
    """Parse the dynamic_trace.pb and return the geometry of
    the FIRST kernel record. The 9 admitted cells each have a
    single kernel of interest; if there are multiple kernels,
    the first (by sorted (device_id, stream_id) iteration) is
    chosen — the catalog's per-cell aggregation uses this
    representative kernel's geometry."""
    if not trace_pb_path.is_file():
        raise LaunchParametersTraceMissing(
            f"dynamic_trace.pb missing at {trace_pb_path}"
        )
    trace = trace_pb2_mod.Trace()
    trace.ParseFromString(trace_pb_path.read_bytes())
    for _device_id in sorted(trace.gpu_device.keys()):
        device = trace.gpu_device[_device_id]
        for _stream_id in sorted(device.streams.keys()):
            stream = device.streams[_stream_id]
            for kernel in stream.kernels:
                return LaunchGeometry(
                    kernel_id=int(kernel.id),
                    grid_x=int(kernel.grid_dim.x),
                    grid_y=int(kernel.grid_dim.y),
                    grid_z=int(kernel.grid_dim.z),
                    block_x=int(kernel.block_dim.x),
                    block_y=int(kernel.block_dim.y),
                    block_z=int(kernel.block_dim.z),
                    static_shared_bytes=int(
                        kernel.size_shared_memory
                    ),
                    regs_per_thread=int(
                        kernel.number_of_registers
                    ),
                )
    raise LaunchParametersKernelIdNotFound(
        f"no kernel records found in {trace_pb_path}"
    )


def _per_warp_occupancy(geom: LaunchGeometry) -> float:
    """Compute per-CTA warp count: threads-per-CTA / warpSize.
    The CTA occupancy proxy carries the per-CTA scheduling
    pressure on the warp scheduler; cross-cell variation
    captures kernel-to-kernel scheduling differences."""
    threads_per_cta = (
        geom.block_x * geom.block_y * geom.block_z
    )
    return threads_per_cta / _AMPERE_WARP_SIZE


def _block_volume(geom: LaunchGeometry) -> int:
    return geom.block_x * geom.block_y * geom.block_z


def _grid_volume(geom: LaunchGeometry) -> int:
    return geom.grid_x * geom.grid_y * geom.grid_z


# Candidate search globs for each auxiliary surface, rooted at
# the cell's trace_dir. The four-surface join walks these
# patterns in order; if multiple files match with conflicting
# join keys the discovery fail-closes with
# LaunchParametersAmbiguousJoin. If zero files match the cell's
# observation is tagged `not_collected_for_cell` for that
# surface only after the checked-absence path runs.
_CUPTI_ACTIVITY_GLOBS: tuple[str, ...] = (
    "cupti_activity_*.json",
    "cupti_activity/cupti_activity_*.json",
)
_NVML_COVARIATE_GLOBS: tuple[str, ...] = (
    "nvml_covariate_*.json",
    "nvml_covariate/nvml_covariate_*.json",
)
_PHASE_LEDGER_GLOBS: tuple[str, ...] = (
    "phase_cycle_ledger_*.json",
    "phase_cycle_ledger/phase_cycle_ledger_*.json",
    "phase_cycle_ledgers/phase_cycle_ledger_*.json",
)


def _discover_candidates(
    trace_dir: Path, globs: tuple[str, ...],
) -> list[Path]:
    """Search the trace directory for files matching any of
    ``globs``. Returns a sorted list of resolved paths."""
    seen: set[Path] = set()
    for pattern in globs:
        for match in trace_dir.glob(pattern):
            if match.is_file():
                seen.add(match.resolve())
    return sorted(seen)


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _join_cupti_activity(
    *,
    trace_dir: Path,
    cell: dict,
    kernel_id: int,
) -> dict:
    """Discover + validate CUPTI Activity candidate files for
    a cell. Returns a dict with one of three shapes:

      - `{"availability": "prediction_time_available",
          "records": [...], "start_ns_min": ..., ...}` when a
        single matching candidate joined cleanly.
      - `{"availability": "not_collected_for_cell",
          "searched_globs": [...]}` when zero candidates
        exist (checked-absence path).
      - never returns ambiguous / mismatched results; both
        raise typed exceptions.

    Validation: each candidate must satisfy slug + shape_point
    + kernel_correlation_id == cell.slug + cell.shape_point +
    kernel_id (kernel-proto kernel.id). Multiple candidates
    with conflicting start_ns raise
    LaunchParametersAmbiguousJoin.
    """
    candidates = _discover_candidates(trace_dir, _CUPTI_ACTIVITY_GLOBS)
    if not candidates:
        return {
            "availability": "not_collected_for_cell",
            "searched_globs": list(_CUPTI_ACTIVITY_GLOBS),
        }
    cell_slug = str(cell["slug"])
    cell_shape = str(cell["shape_point"])
    records: list[dict] = []
    seen_start_ns: set[int] = set()
    seen_repeats: set[int] = set()
    # Per-repeat ambiguity tracking: each repeat_index must
    # appear in at most one CUPTI candidate (or in multiple
    # candidates with identical start_ns; conflicting
    # start_ns within the same repeat = ambiguous launch).
    start_ns_by_repeat: dict[int, set[int]] = {}
    for path in candidates:
        doc = _load_json(path)
        rec_slug = str(doc.get("slug", ""))
        rec_shape = str(doc.get("shape_point", ""))
        rec_kid = doc.get("kernel_correlation_id")
        rec_start_ns = doc.get("start_ns")
        rec_repeat = doc.get("repeat_index")
        if rec_slug != cell_slug or rec_shape != cell_shape:
            raise LaunchParametersJoinKeyMismatch(
                f"CUPTI Activity sidecar at {path} declares "
                f"slug={rec_slug!r}/shape={rec_shape!r}, "
                f"but cell identity is "
                f"{cell_slug}/{cell_shape}"
            )
        # CUPTI is the cross-surface repeat-index anchor; every
        # candidate must declare the three plan-pinned join keys
        # (kernel_correlation_id, start_ns, repeat_index). A
        # missing key on any candidate is a schema-required-
        # field gap that fails closed (BL-20260509
        # strict-admission-contract-first).
        if rec_kid is None:
            raise LaunchParametersJoinKeyMismatch(
                f"CUPTI Activity sidecar at {path} is missing "
                f"the plan-pinned `kernel_correlation_id` "
                f"join key"
            )
        if int(rec_kid) != int(kernel_id):
            raise LaunchParametersJoinKeyMismatch(
                f"CUPTI Activity sidecar at {path} declares "
                f"kernel_correlation_id={rec_kid!r}, but the "
                f"kernel-proto record's kernel.id is "
                f"{kernel_id}"
            )
        if rec_start_ns is None:
            raise LaunchParametersJoinKeyMismatch(
                f"CUPTI Activity sidecar at {path} is missing "
                f"the plan-pinned `start_ns` join key"
            )
        if rec_repeat is None:
            raise LaunchParametersJoinKeyMismatch(
                f"CUPTI Activity sidecar at {path} is missing "
                f"the plan-pinned `repeat_index` join key"
            )
        seen_start_ns.add(int(rec_start_ns))
        seen_repeats.add(int(rec_repeat))
        start_ns_by_repeat.setdefault(int(rec_repeat), set()).add(
            int(rec_start_ns)
        )
        records.append({
            "path": str(path),
            "repeat_index": int(rec_repeat),
            "start_ns": int(rec_start_ns),
            "kernel_correlation_id": int(rec_kid),
        })
    # Ambiguity check: within any single repeat_index, all
    # CUPTI candidates must agree on start_ns. Cross-repeat
    # start_ns differences are expected (each repeat is a
    # distinct kernel launch); the original "any conflicting
    # start_ns globally" check was too coarse for
    # multi-repeat cohorts.
    for repeat, ns_set in start_ns_by_repeat.items():
        if len(ns_set) > 1:
            raise LaunchParametersAmbiguousJoin(
                f"CUPTI Activity discovery for "
                f"{cell_slug}/{cell_shape} found multiple "
                f"candidates with conflicting start_ns values "
                f"{sorted(ns_set)} for repeat_index={repeat}; "
                f"refusing to pick one. "
                f"Paths: {[str(p) for p in candidates]}"
            )
    return {
        "availability": "prediction_time_available",
        "records": records,
        "start_ns_min": min(seen_start_ns),
        "repeat_indices": sorted(seen_repeats),
        "candidate_count": len(candidates),
    }


def _join_nvml_covariate(
    *,
    trace_dir: Path,
    cell: dict,
    expected_repeat_indices: list[int] | None = None,
) -> dict:
    """Discover + validate NVML covariate candidate files
    for a cell. NVML samples carry no kernel_id (they are
    a separate stream sampled by host monotonic clock), so
    the within-surface validation is by slug + shape_point
    + repeat_index.

    When ``expected_repeat_indices`` is non-None and
    non-empty (the caller threads in CUPTI's repeat-index
    anchor when CUPTI is present), every discovered
    candidate's ``repeat_index`` MUST belong to the anchor
    set; missing or out-of-set values raise
    ``LaunchParametersJoinKeyMismatch``. When the anchor is
    None or empty, within-surface validation continues as
    before.

    DVFS state is recorded as label-provenance-only on the
    sidecar regardless of whether the NVML sidecar exists;
    the catalog's per-cell DVFS observation always carries
    the forbidden_requires_explicit_reopen prohibition per
    the no-DVFS-as-predictor-input directive."""
    candidates = _discover_candidates(trace_dir, _NVML_COVARIATE_GLOBS)
    if not candidates:
        return {
            "availability": "not_collected_for_cell",
            "searched_globs": list(_NVML_COVARIATE_GLOBS),
        }
    cell_slug = str(cell["slug"])
    cell_shape = str(cell["shape_point"])
    anchor: set[int] | None = (
        set(expected_repeat_indices)
        if expected_repeat_indices else None
    )
    records: list[dict] = []
    seen_repeat: set[int] = set()
    for path in candidates:
        doc = _load_json(path)
        # NVML sidecars in this repository carry slug +
        # shape_point only on some schema versions; admit
        # missing fields but reject mismatched ones.
        rec_slug = doc.get("slug")
        rec_shape = doc.get("shape_point")
        if rec_slug is not None and str(rec_slug) != cell_slug:
            raise LaunchParametersJoinKeyMismatch(
                f"NVML covariate sidecar at {path} declares "
                f"slug={rec_slug!r}, but cell identity is "
                f"{cell_slug}"
            )
        if rec_shape is not None and str(rec_shape) != cell_shape:
            raise LaunchParametersJoinKeyMismatch(
                f"NVML covariate sidecar at {path} declares "
                f"shape_point={rec_shape!r}, but cell identity "
                f"is {cell_shape}"
            )
        repeat_index = doc.get("repeat_index")
        if anchor is not None:
            if repeat_index is None:
                raise LaunchParametersJoinKeyMismatch(
                    f"NVML covariate sidecar at {path} is "
                    f"missing the `repeat_index` join key "
                    f"required for cross-surface JOIN against "
                    f"CUPTI Activity anchor "
                    f"{sorted(anchor)}"
                )
            if int(repeat_index) not in anchor:
                raise LaunchParametersJoinKeyMismatch(
                    f"NVML covariate sidecar at {path} declares "
                    f"repeat_index={repeat_index!r}, but CUPTI "
                    f"Activity anchor set for this cell is "
                    f"{sorted(anchor)}; cross-surface "
                    f"repeat-index disagreement"
                )
        if repeat_index is not None:
            seen_repeat.add(int(repeat_index))
        records.append({
            "path": str(path),
            "repeat_index": (
                int(repeat_index)
                if repeat_index is not None else None
            ),
        })
    # Cross-surface exact-set check: when CUPTI Activity is
    # the anchor and NVML has candidates, the NVML observed
    # repeat-index set MUST equal the CUPTI anchor exactly.
    # Subset coverage (NVML missing repeats CUPTI saw) means
    # the joined record would falsely claim NVML coverage on
    # CUPTI repeats with no NVML data; superset coverage
    # means NVML carries repeats CUPTI did not see. Both
    # fail closed.
    if anchor is not None and seen_repeat != anchor:
        raise LaunchParametersJoinKeyMismatch(
            f"NVML covariate cross-surface join failed: "
            f"observed repeat_indices={sorted(seen_repeat)} "
            f"do not equal CUPTI Activity anchor set "
            f"{sorted(anchor)}; partial / over-covered NVML "
            f"surface cannot be declared joined"
        )
    return {
        "availability": "label-provenance-only",
        "records": records,
        "repeat_indices": sorted(seen_repeat),
        "candidate_count": len(candidates),
    }


def _join_phase_cycle_ledger(
    *,
    trace_dir: Path,
    cell: dict,
    kernel_id: int,
    expected_repeat_indices: list[int] | None = None,
) -> dict:
    """Discover + validate phase_cycle_ledger candidate
    files. The ledger is keyed by slug + shape_point +
    repeat_index + launch_index + kernel_signature_sha256
    (BL-20260507 lockstep). When the cell carries a
    non-empty ``kernel_signature_sha256``, every discovered
    candidate's signature MUST match (missing/empty
    signatures fail closed; disagreement raises).

    When ``expected_repeat_indices`` is non-None and
    non-empty (the caller threads in CUPTI's repeat-index
    anchor when CUPTI is present), every discovered
    candidate's ``repeat_index`` MUST belong to the anchor
    set; missing or out-of-set values raise
    ``LaunchParametersJoinKeyMismatch``."""
    candidates = _discover_candidates(trace_dir, _PHASE_LEDGER_GLOBS)
    # Also search the sibling `phase_cycle_ledgers_<shape>/`
    # directory under trace_dir's parent. `run_suite.py`'s
    # phase-injection path emits ledgers there (alongside
    # `trace_<shape>/`) rather than under
    # `trace_<shape>/phase_cycle_ledger{,s}/`. Without this
    # search, the discovery returns `not_collected_for_cell` for
    # cells where phase injection actually ran, dropping a real
    # feature from the launch-parameter sidecar.
    cell_shape = str(cell["shape_point"])
    if trace_dir.parent != trace_dir:
        sibling_dir = trace_dir.parent / f"phase_cycle_ledgers_{cell_shape}"
        if sibling_dir.is_dir():
            sibling_matches = sorted(
                p.resolve()
                for p in sibling_dir.glob("phase_cycle_ledger_*.json")
                if p.is_file()
            )
            seen = {p for p in candidates}
            for m in sibling_matches:
                if m not in seen:
                    candidates.append(m)
                    seen.add(m)
            candidates.sort()
    searched_globs = list(_PHASE_LEDGER_GLOBS) + [
        f"<parent>/phase_cycle_ledgers_{cell_shape}/phase_cycle_ledger_*.json",
    ]
    if not candidates:
        return {
            "availability": "not_collected_for_cell",
            "searched_globs": searched_globs,
        }
    cell_slug = str(cell["slug"])
    cell_shape = str(cell["shape_point"])
    cell_ksig = str(cell.get("kernel_signature_sha256", ""))
    anchor: set[int] | None = (
        set(expected_repeat_indices)
        if expected_repeat_indices else None
    )
    records: list[dict] = []
    seen_repeat: set[int] = set()
    for path in candidates:
        doc = _load_json(path)
        rec_slug = str(doc.get("slug", ""))
        rec_shape = str(doc.get("shape_point", ""))
        rec_ksig = str(doc.get("kernel_signature_sha256", ""))
        if rec_slug != cell_slug or rec_shape != cell_shape:
            raise LaunchParametersJoinKeyMismatch(
                f"phase_cycle_ledger at {path} declares "
                f"slug={rec_slug!r}/shape={rec_shape!r}, but "
                f"cell identity is {cell_slug}/{cell_shape}"
            )
        # BL-20260507 lockstep: when the cell carries a
        # non-empty kernel_signature_sha256, the phase
        # ledger candidate MUST also carry it AND match
        # exactly. A missing or empty value is a contract
        # gap that fails closed.
        if cell_ksig:
            if not rec_ksig:
                raise LaunchParametersJoinKeyMismatch(
                    f"phase_cycle_ledger at {path} is missing "
                    f"the BL-20260507 lockstep "
                    f"`kernel_signature_sha256` key while the "
                    f"cell carries {cell_ksig!r}"
                )
            if rec_ksig != cell_ksig:
                raise LaunchParametersJoinKeyMismatch(
                    f"phase_cycle_ledger at {path} declares "
                    f"kernel_signature_sha256={rec_ksig!r}, but "
                    f"cell records {cell_ksig!r}"
                )
        rec_repeat = doc.get("repeat_index")
        if anchor is not None:
            if rec_repeat is None:
                raise LaunchParametersJoinKeyMismatch(
                    f"phase_cycle_ledger at {path} is missing "
                    f"the `repeat_index` join key required for "
                    f"cross-surface JOIN against CUPTI "
                    f"Activity anchor {sorted(anchor)}"
                )
            if int(rec_repeat) not in anchor:
                raise LaunchParametersJoinKeyMismatch(
                    f"phase_cycle_ledger at {path} declares "
                    f"repeat_index={rec_repeat!r}, but CUPTI "
                    f"Activity anchor set for this cell is "
                    f"{sorted(anchor)}; cross-surface "
                    f"repeat-index disagreement"
                )
        if rec_repeat is not None:
            seen_repeat.add(int(rec_repeat))
        records.append({
            "path": str(path),
            "repeat_index": (
                int(rec_repeat)
                if rec_repeat is not None else -1
            ),
            "launch_index": int(doc.get("launch_index", 0)),
            "kernel_signature_sha256": rec_ksig,
        })
    # Cross-surface exact-set check: when CUPTI Activity is
    # the anchor and the phase-cycle ledger has candidates,
    # the phase ledger observed repeat-index set MUST equal
    # the CUPTI anchor exactly. Subset OR superset coverage
    # both fail closed (partial joined record).
    if anchor is not None and seen_repeat != anchor:
        raise LaunchParametersJoinKeyMismatch(
            f"phase_cycle_ledger cross-surface join failed: "
            f"observed repeat_indices={sorted(seen_repeat)} "
            f"do not equal CUPTI Activity anchor set "
            f"{sorted(anchor)}; partial / over-covered phase "
            f"surface cannot be declared joined"
        )
    return {
        "availability": "prediction_time_available",
        "records": records,
        "repeat_indices": sorted(seen_repeat),
        "candidate_count": len(candidates),
    }


def build_launch_parameters_sidecar(
    *,
    cell: dict,
    arch_dir: str = _DEFAULT_ARCH_DIR,
    repo_root: Path = _REPO_ROOT,
    trace_pb2_mod: Any | None = None,
) -> dict:
    """Build the launch_parameters_sidecar dict for a single
    admitted cell.

    Pure function: takes the inputs explicitly so tests can
    inject synthetic trace_pb2 modules + synthetic
    label-admitted records.
    """
    if trace_pb2_mod is None:
        trace_pb2_mod = _load_trace_bindings()

    slug = str(cell["slug"])
    shape_point = str(cell["shape_point"])
    trace_dir = (
        repo_root / "workloads" / "simbench_bw" / "results"
        / slug / arch_dir / f"trace_{shape_point}"
    )
    trace_pb = trace_dir / "dynamic_trace.pb"
    geometry = _extract_first_kernel_geometry(
        trace_pb, trace_pb2_mod
    )

    # Four-surface checked-join discovery. Each helper returns
    # availability=prediction_time_available with records when
    # candidates exist and validate; availability=
    # label-provenance-only for NVML (DVFS is never
    # prediction-time available regardless of collection
    # status); availability=not_collected_for_cell after a
    # checked-absence path. Each helper raises typed
    # exceptions on ambiguous candidates / mismatched join
    # keys; the caller surfaces those upstream.
    # CUPTI Activity is the cross-surface repeat-index anchor
    # (the only auxiliary surface that carries start_ns AND a
    # kernel-bound identity). When CUPTI joined cleanly, its
    # repeat_indices list is threaded into the NVML + phase
    # joins so each downstream surface validates its
    # repeat_index against the anchor. When CUPTI is absent
    # for the cell, the downstream joins fall back to their
    # within-surface validation only.
    cupti_join = _join_cupti_activity(
        trace_dir=trace_dir, cell=cell,
        kernel_id=geometry.kernel_id,
    )
    anchor_repeat_indices: list[int] | None = (
        cupti_join.get("repeat_indices")
        if cupti_join.get("availability")
        == "prediction_time_available"
        else None
    )
    nvml_join = _join_nvml_covariate(
        trace_dir=trace_dir, cell=cell,
        expected_repeat_indices=anchor_repeat_indices,
    )
    phase_join = _join_phase_cycle_ledger(
        trace_dir=trace_dir, cell=cell,
        kernel_id=geometry.kernel_id,
        expected_repeat_indices=anchor_repeat_indices,
    )

    sidecar: dict = {
        "schema_version": 1,
        "generator": (
            "tools/local_labels/extend_launch_parameters.py"
        ),
        "arch": str(cell.get("arch", "sm_86")),
        "slug": slug,
        "shape_point": shape_point,
        "binary_sha256": str(cell.get("binary_sha256", "")),
        "kernel_signature_sha256": str(
            cell.get("kernel_signature_sha256", "")
        ),
        "kernel_id": geometry.kernel_id,
        "grid_dim": {
            "x": geometry.grid_x,
            "y": geometry.grid_y,
            "z": geometry.grid_z,
        },
        "block_dim": {
            "x": geometry.block_x,
            "y": geometry.block_y,
            "z": geometry.block_z,
        },
        "static_shared_bytes": geometry.static_shared_bytes,
        "regs_per_thread": geometry.regs_per_thread,
        "derived": {
            "threads_per_cta": _block_volume(geometry),
            "warps_per_cta": _per_warp_occupancy(geometry),
            "grid_volume": _grid_volume(geometry),
        },
        # Four-surface checked-join results. The downstream
        # catalog reads `*_availability` to decide
        # per-feature availability rather than a hard-coded
        # tag; the records arrays document what was found at
        # the cell's trace directory.
        "cupti_activity_join": cupti_join,
        "nvml_covariate_join": nvml_join,
        "phase_cycle_ledger_join": phase_join,
        # DVFS availability is structurally
        # label-provenance-only regardless of whether NVML
        # sidecars were found (the no-DVFS-as-predictor-input
        # directive forbids it as a prediction-time input);
        # the prohibition is independent of the
        # checked-absence path's outcome.
        "dvfs_state_availability": "label-provenance-only",
        # The MSHR-pressure proxy is derived from CUPTI
        # Activity records (memcpy-overlap + concurrent-kernel
        # heuristics). When the CUPTI join landed, the proxy
        # is prediction-time-available; otherwise the
        # checked-absence path emits not_collected_for_cell.
        "mshr_pressure_proxy_availability": (
            "prediction_time_available"
            if cupti_join.get("availability")
            == "prediction_time_available"
            else "not_collected_for_cell"
        ),
    }
    return sidecar


def write_sidecar(
    sidecar: dict, *, out_dir: Path,
) -> Path:
    """Write the sidecar to the documented host-local cache
    directory. Returns the path written.

    Deterministic output: sort_keys=True + indent=2 +
    trailing newline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        f"launch_parameters_sidecar_{sidecar['slug']}__"
        f"{sidecar['shape_point']}.json"
    )
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return out_path


def build_sidecars_for_cohort(
    *,
    label_admitted_path: Path = _DEFAULT_LABEL_ADMITTED,
    out_dir: Path = _DEFAULT_OUT_DIR,
    arch_dir: str = _DEFAULT_ARCH_DIR,
    repo_root: Path = _REPO_ROOT,
    trace_pb2_mod: Any | None = None,
) -> list[Path]:
    """Build the sidecar for every admitted ``arch=sm_86``
    cell in ``label_admitted_path``. Returns the list of
    written paths."""
    if trace_pb2_mod is None:
        trace_pb2_mod = _load_trace_bindings()
    with label_admitted_path.open(encoding="utf-8") as fh:
        admitted = json.load(fh)
    admitted_cells = [
        c for c in admitted.get("cells", [])
        if isinstance(c, dict) and c.get("arch") in _ADMITTED_LABEL_ARCHES
    ]
    if not admitted_cells:
        raise ValueError(
            f"extend_launch_parameters: zero admitted cells in "
            f"{label_admitted_path} for label arches in "
            f"{_ADMITTED_LABEL_ARCHES}"
        )
    admitted_cells.sort(
        key=lambda c: (
            str(c.get("arch")),
            str(c.get("slug")),
            str(c.get("shape_point")),
        ),
    )
    written: list[Path] = []
    for cell in admitted_cells:
        cell_arch = str(cell.get("arch"))
        # Per-arch routing: when the caller passed the default
        # arch_dir / out_dir, route by the cell's label arch using the
        # shared policy mapping; when the caller passed explicit
        # overrides (e.g., a test fixture pinning a non-default
        # location), honor them as the per-cell value.
        cell_arch_dir = (
            arch_dir if arch_dir != _DEFAULT_ARCH_DIR
            else _arch_dir_for_label_arch(cell_arch)
        )
        cell_out_dir = (
            out_dir if out_dir != _DEFAULT_OUT_DIR
            else _out_dir_for_label_arch(cell_arch)
        )
        sidecar = build_launch_parameters_sidecar(
            cell=cell,
            arch_dir=cell_arch_dir,
            repo_root=repo_root,
            trace_pb2_mod=trace_pb2_mod,
        )
        written.append(write_sidecar(sidecar, out_dir=cell_out_dir))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-cell launch_parameters_sidecar.json "
            "for the 9 admitted sm_86 cells by joining the "
            "label index, dynamic_trace.pb launch geometry, "
            "and any available auxiliary covariate sidecars."
        ),
    )
    parser.add_argument(
        "--label-admitted",
        type=Path,
        default=_DEFAULT_LABEL_ADMITTED,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
    )
    parser.add_argument(
        "--arch-dir",
        type=str,
        default=_DEFAULT_ARCH_DIR,
    )
    parser.add_argument(
        "--repo-root", type=Path, default=_REPO_ROOT,
    )
    args = parser.parse_args(argv)

    try:
        written = build_sidecars_for_cohort(
            label_admitted_path=args.label_admitted,
            out_dir=args.out_dir,
            arch_dir=args.arch_dir,
            repo_root=args.repo_root,
        )
    except (
        ValueError,
        LaunchParametersTraceMissing,
        LaunchParametersProtobufBindingsMissing,
        LaunchParametersKernelIdNotFound,
        LaunchParametersAmbiguousJoin,
        LaunchParametersJoinKeyMismatch,
    ) as exc:
        sys.stderr.write(
            f"extend_launch_parameters: {exc}\n"
        )
        return 2

    sys.stderr.write(
        f"extend_launch_parameters: wrote {len(written)} "
        f"sidecars under {args.out_dir}\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
