"""``SurfaceOrchestrator`` — driver for the six counter-free surfaces.

The orchestrator owns the per-bundle workflow:

1. Probe the host once (``HostInfo``) at startup.
2. Resolve ``bundle_dir`` for the ``(slug, language)`` target.
3. For each surface in topological order (NVBit and nsys in separate
   dispatch groups for collision safety):
   * Skip if not in ``enabled_surfaces`` (record ``skipped`` with
     waiver).
   * Check the cache; reuse the stored entry when every cache-identity
     field matches and the stored status is non-failed .
     ``--force-resurface`` bypasses the cache.
   * Call ``probe()``; if not ``ok``, record the probe result and
     continue to the next surface.
   * Call ``run()`` inside an isolation context appropriate for the
     surface's dispatch group.
   * Call ``parse()`` to validate the on-disk artifact.
4. Write ``manifest.json`` to ``bundle_dir`` via
   ``ManifestBuilder.lint_and_write`` — lint refuses to write a
   manifest that violates the manifest lint invariants.

This module hosts the orchestrator skeleton; the six concrete
surface plugins implement their own ``probe`` / ``run`` / ``parse``
logic. The orchestrator wires plugin registration and the manifest
workflow end-to-end; each plugin owns its own artifact emission.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence

# Add the parent scripts/ to sys.path so the orchestrator can import
# from sol_baselines_harness.native_dispatch when invoked as a module from CLI.
_THIS_PARENT = Path(__file__).resolve().parents[2]
if str(_THIS_PARENT) not in sys.path:
    sys.path.insert(0, str(_THIS_PARENT))

from sol_baselines_harness.native_env import (  # noqa: E402
    DegradedNativeEnvError,
    compute_native_env_hash,
    missing_components,
    native_env_components,
)
from sol_baselines_harness.surfaces.base import (  # noqa: E402
    HostInfo,
    Surface,
    SurfaceParseResult,
    SURFACE_STATUS_OK,
    SURFACE_STATUS_SKIPPED,
    VALID_SURFACES,
)
from sol_baselines_harness.surfaces.manifest import (  # noqa: E402
    ManifestBuilder,
    SurfaceManifestEntry,
    now_utc_iso,
)


class UnknownSurfaceError(ValueError):
    """Raised when ``enabled_surfaces`` carries an unrecognized name.

    The exception message lists the locked surface vocabulary so the
    operator can fix the typo without consulting the source.
    """


# Surface groups. Surfaces that inject at the CUDA driver layer MUST
# run in separate dispatch groups because the kernel module allows
# exactly one ``CUDA_INJECTION64_PATH`` per process: ``nvbit_metrics``
# (the counter-free metric extractor) and ``nsys`` (process-level
# ``nsys profile --trace=cuda,nvtx`` wrapper) both claim that slot.
# ``nvbit_metrics`` additionally installs a CUDA-callback CUPTI
# subscriber at injection time, so it cannot co-reside with
# ``cupti_activity`` inside the same process. The orchestrator keeps
# the three in distinct groups so a single bundle can run all three
# as matched sequential dispatch invocations against the same
# baseline. Static introspection is build-time only and shares no
# driver state.
_DISPATCH_GROUPS: Dict[str, str] = {
    "static": "build_time",
    "sanitizer": "sanitizer_eval",
    "nvbit_metrics": "nvbit_metrics_eval",
    "cupti_activity": "cupti_eval",
    "nsys": "nsys_eval",
    # `cache_prediction` is the passive predictor surface: it never
    # drives the evaluator and has no shared driver state with the
    # other groups, so it gets its own group key. The plugin reports
    # `unavailable` and emits no payload while no per-launch
    # prediction sidecar producer is wired into the bundle.
    "cache_prediction": "cache_prediction_passive",
}

# Topological order the orchestrator walks. Build-time surfaces first
# so subsequent surface dispatches see a populated bundle. The
# mutual-exclusion between nvbit_metrics, nsys, and cupti_activity is
# honored by the dispatch-group split, NOT by ordering.
# `cache_prediction` lands last because it is passive (non-eval-
# driving) and post-hoc by construction: its eventual payload joins
# against the other surfaces' artifacts, not the other way around.
_SURFACE_ORDER: List[str] = [
    "static",
    "sanitizer",
    "cupti_activity",
    "nvbit_metrics",
    "nsys",
    "cache_prediction",
]

# Surfaces that require driving the native ``sol-execbench`` evaluator
# subprocess. The passive ``cache_prediction`` surface is deliberately
# excluded — it never spawns the evaluator and exists today only to
# lock the framework vocabulary; an enabled-only-``cache_prediction``
# run takes the empty-rows trace branch in ``_emit_trace_json`` so the
# manifest is still lint-clean. Adding a surface to this set elsewhere
# changes whether the orchestrator runs the evaluator subprocess; the
# ``cache_prediction`` plugin therefore must NOT be promoted into this
# set until it has a real evaluator-driving payload path.
_EVAL_DRIVING_SURFACES: frozenset = frozenset(
    {
        "static",
        "sanitizer",
        "nvbit_metrics",
        "cupti_activity",
        "nsys",
    }
)


# Fields recognised on ``SurfaceManifestEntry`` for cache-reuse
# reconstruction. The list stays in sync with the
# ``SurfaceManifestEntry`` dataclass so a stale manifest entry that
# carries extra keys doesn't break the reuse path.
_MANIFEST_ENTRY_FIELDS = frozenset({
    "status", "tool_version", "schema_version", "impl_version",
    "command", "exit_code", "stdout_tail", "stderr_tail",
    "artifact_path", "waiver_reason", "start_utc", "end_utc",
    "size_bytes", "compressed_size_bytes", "substatus", "cache_identity",
})


def _count_workloads(problem_dir: Path) -> int:
    """Count non-empty / non-comment lines in ``workload.jsonl``.

    Matches ``run.py:_count_expected_workloads``'s rule set so the
    orchestrator's workload subset stays consistent with the upstream
    CLI's row count.
    """
    workload_path = problem_dir / "workload.jsonl"
    if not workload_path.is_file():
        return 1  # Best-effort fallback so the orchestrator can still drive a smoke.
    count = 0
    for raw in workload_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        count += 1
    return max(count, 1)


def _resolve_workload_subset(
    *,
    requested: List[int],
    policy: str,
    workload_count: int,
) -> List[int]:
    """Materialise the workload subset against the problem's count.

    * ``policy == "full-fanout"`` materialises ``[0..N-1]``.
    * ``policy == "explicit"`` validates each requested index is in
      ``[0, N)`` and raises ``ValueError`` on out-of-bounds.
    * ``policy == "lowest-index-default"`` (default) keeps the
      requested list (typically ``[0]``).
    """
    if policy == "full-fanout":
        return list(range(workload_count))
    if policy == "explicit":
        for idx in requested:
            if idx < 0 or idx >= workload_count:
                raise ValueError(
                    f"--representative-workload {idx} out of bounds; "
                    f"problem has {workload_count} workload(s)."
                )
        return list(requested) if requested else [0]
    # Default policy: lowest index. An empty requested list falls
    # back to [0] so the orchestrator always has at least one
    # workload to drive.
    return list(requested) if requested else [0]


def _hash_config_overrides(config_overrides: Dict[str, Any]) -> str:
    """SHA-256 over the canonical-order JSON of ``config_overrides``."""
    import hashlib
    import json
    payload = json.dumps(config_overrides or {}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_prior_manifest_entries(bundle_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Return the prior manifest's ``surfaces`` dict, or ``{}``."""
    import json
    path = bundle_dir / "manifest.json"
    if not path.is_file():
        return {}
    try:
        prior = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    surfaces = prior.get("surfaces")
    if not isinstance(surfaces, dict):
        return {}
    return {k: v for k, v in surfaces.items() if isinstance(v, dict)}


def _prior_trace_is_valid(bundle_dir: Path) -> bool:
    """True when ``bundle_dir/trace.json`` exists, parses, and has rows."""
    import json
    trace_path = bundle_dir / "trace.json"
    if not trace_path.is_file():
        return False
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    rows = trace.get("rows")
    return isinstance(rows, list) and len(rows) > 0


def _prior_trace_all_passed(bundle_dir: Path) -> bool:
    """True when every row in the prior ``trace.json`` is PASSED."""
    import json
    trace_path = bundle_dir / "trace.json"
    if not trace_path.is_file():
        return False
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    rows = trace.get("rows", [])
    return bool(rows) and all(r.get("status") == "PASSED" for r in rows)


def _emit_trace_json(
    *,
    bundle_dir: Path,
    baseline_dir: Path,
    enabled_surfaces: frozenset,
    plan: Any,
) -> bool:
    """Drive the upstream evaluator once (if needed) and persist ``trace.json``.

    Returns ``True`` only when ALL gates pass:
      * Evaluator returncode == 0.
      * Parsed row count == expected workload count (from the plan's
        ``workload_subset`` or the problem's full workload count).
      * Every row has status ``PASSED``.
      * No parse errors.

    Any failure produces a ``False`` return AND records the reason in
    the persisted ``trace.json`` so the manifest cannot over-claim
    ``Profiled-*`` without a clean run.
    """
    import json
    import os
    import subprocess
    import sys
    _SCRIPTS_DIR_HERE = Path(__file__).resolve().parents[1]
    if str(_SCRIPTS_DIR_HERE.parent) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR_HERE.parent))
    if str(_SCRIPTS_DIR_HERE) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR_HERE))
    from sol_baselines_harness.parse_trace import parse_jsonl_lines

    trace_path = bundle_dir / "trace.json"
    if not enabled_surfaces & _EVAL_DRIVING_SURFACES:
        trace_path.write_text(
            json.dumps({
                "baseline": str(baseline_dir),
                "rows": [],
                "note": (
                    "no eval-driving surface enabled; orchestrator did "
                    "not invoke the upstream evaluator."
                ),
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        return False
    # Compose the canonical native eval argv from the plan.
    eval_argv, eval_env, _ = plan.build_eval_argv()
    if not eval_argv:
        trace_path.write_text(
            json.dumps({
                "baseline": str(baseline_dir),
                "rows": [],
                "note": "plan eval argv empty; eval invocation skipped.",
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        return False
    from sol_baselines_harness.surfaces.subprocess_env import (
        BUILD_TOOLCHAIN_ALLOWED,
        build_subprocess_env,
    )
    env = build_subprocess_env(
        plan_env_overrides=eval_env,
        extra_allowed=BUILD_TOOLCHAIN_ALLOWED,
    )
    try:
        proc = subprocess.run(
            eval_argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=plan.eval_timeout_s,
            env=env,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        trace_path.write_text(
            json.dumps({
                "baseline": str(baseline_dir),
                "rows": [],
                "note": f"upstream eval launch failed: {exc!r}",
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        return False
    rows: List[Dict[str, Any]] = []
    parse_errors: List[Dict[str, Any]] = []
    workload_index = 0
    for line_no, trace, raw in parse_jsonl_lines(proc.stdout.splitlines()):
        if trace is None:
            parse_errors.append({"line_no": line_no, "raw": raw[:200]})
            continue
        if trace.is_workload_trace():
            continue
        row: Dict[str, Any] = {
            "slug": baseline_dir.parent.name,
            "language": baseline_dir.name,
            "workload_index": workload_index,
            "status": (
                trace.evaluation.status.value
                if trace.evaluation is not None else None
            ),
            "latency_ms": None,
            "reference_latency_ms": None,
            "speedup_factor": None,
            "max_relative_error": None,
            "max_absolute_error": None,
            "has_nan": None,
            "has_inf": None,
            "hardware": None,
            "definition": trace.definition,
        }
        if trace.evaluation is not None and trace.evaluation.performance is not None:
            perf = trace.evaluation.performance
            row["latency_ms"] = perf.latency_ms
            row["reference_latency_ms"] = perf.reference_latency_ms
            row["speedup_factor"] = perf.speedup_factor
        if trace.evaluation is not None and trace.evaluation.correctness is not None:
            corr = trace.evaluation.correctness
            row["max_relative_error"] = corr.max_relative_error
            row["max_absolute_error"] = corr.max_absolute_error
            row["has_nan"] = corr.has_nan
            row["has_inf"] = corr.has_inf
        if trace.evaluation is not None:
            row["hardware"] = trace.evaluation.environment.hardware
        rows.append(row)
        workload_index += 1
    expected_count = (
        len(plan.workload_subset) if plan.workload_subset else 0
    )
    gate_reasons: List[str] = []
    if proc.returncode != 0:
        gate_reasons.append(f"evaluator exited rc={proc.returncode}")
    # Mirror run.py:_collect_failure_reasons — parse_errors are only
    # gating when ``rows`` is empty (the upstream CLI mixes progress
    # / banner lines into stdout that parse_jsonl_lines correctly
    # rejects without invalidating the actual Trace rows).
    if parse_errors and not rows:
        gate_reasons.append(
            f"{len(parse_errors)} unparseable JSONL line(s) and zero rows"
        )
    if expected_count > 0 and len(rows) != expected_count:
        gate_reasons.append(
            f"row count mismatch: expected {expected_count}, parsed {len(rows)}"
        )
    if not rows:
        gate_reasons.append("zero rows parsed")
    if not gate_reasons and not all(r.get("status") == "PASSED" for r in rows):
        gate_reasons.append("at least one non-PASSED row")
    trace_path.write_text(
        json.dumps({
            "baseline": str(baseline_dir),
            "rows": rows,
            "returncode": proc.returncode,
            "expected_workload_count": expected_count,
            "parse_errors": parse_errors,
            "gate_reasons": gate_reasons,
            "stderr_tail": (proc.stderr or "")[-2000:],
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return not gate_reasons


class SurfaceOrchestrator:
    """Driver for the six counter-free surfaces over one baseline.

    The six registered surfaces are the five eval-driving surfaces
    (``static``, ``sanitizer``, ``nvbit_metrics``, ``cupti_activity``,
    ``nsys``) plus the passive ``cache_prediction`` predictor
    surface, which is registered against the locked vocabulary but
    emits no evaluator-driving payload — its plugin returns
    ``unavailable`` until a per-launch prediction sidecar producer
    is wired into the bundle.

    Parameters
    ----------
    surface_plugins
        Mapping ``{surface_name: Surface_instance}`` covering every
        surface in ``VALID_SURFACES``. The orchestrator accepts
        passive plugin stubs (returning ``unavailable`` from
        ``probe()`` with a non-empty ``waiver_reason``) so a
        registered surface that has not yet shipped a real payload
        still satisfies the orchestrator's coverage invariant.
    enabled_surfaces
        Subset of ``VALID_SURFACES`` the operator wants to run.
        Surfaces outside this set land in the manifest as ``skipped``
        with a ``waiver_reason``.
    build_mode
        ``"judge"`` or ``"profiling"``. Affects which staged
        ``solution.json`` the surfaces dispatch against.
    workload_subset
        List of workload indices to evaluate. Default is the lowest-
        index workload only. ``--full-fanout`` sets this to
        the complete index list; ``--representative-workload <idx>``
        sets it to ``[idx]``.
    workload_selection_policy
        One of ``"lowest-index-default"`` / ``"full-fanout"`` /
        ``"explicit"``. Recorded verbatim in the manifest.
    force_resurface
        When ``True``, bypass the manifest-based cache and re-run
        every selected surface even when a matching entry exists.
    """

    def __init__(
        self,
        *,
        surface_plugins: Dict[str, Surface],
        enabled_surfaces: FrozenSet[str],
        build_mode: str = "judge",
        workload_subset: Optional[Sequence[int]] = None,
        workload_selection_policy: str = "lowest-index-default",
        force_resurface: bool = False,
    ) -> None:
        invalid = set(enabled_surfaces) - VALID_SURFACES
        if invalid:
            raise UnknownSurfaceError(
                f"unknown surface: {sorted(invalid)}; "
                f"valid: {sorted(VALID_SURFACES)}"
            )
        if build_mode not in ("judge", "profiling"):
            raise ValueError(
                f"build_mode must be 'judge' or 'profiling'; got {build_mode!r}"
            )
        missing = VALID_SURFACES - set(surface_plugins.keys())
        if missing:
            raise UnknownSurfaceError(
                f"surface_plugins missing entries for {sorted(missing)}; "
                f"every surface in VALID_SURFACES must register a plugin "
                f"(stubs returning 'unavailable' are acceptable)."
            )
        self.surface_plugins = dict(surface_plugins)
        self.enabled_surfaces = frozenset(enabled_surfaces)
        self.build_mode = build_mode
        self.workload_subset = list(workload_subset) if workload_subset else [0]
        self.workload_selection_policy = workload_selection_policy
        self.force_resurface = force_resurface

    def run_all(
        self,
        *,
        baseline_dir: Path,
        problem_dir: Path,
        bundle_dir: Path,
        host_info: HostInfo,
        eval_timeout_s: float = 1800.0,
    ) -> Dict[str, Any]:
        """Drive every selected surface against one baseline.

        Constructs a ``DispatchPlan`` carrying the native
        ``sol-execbench`` argv, the curated env overrides, the staged
        solution / config paths, and the evaluator tree hash, then
        passes the plan's argv to each selected surface's ``run()`` so
        the surfaces can actually invoke the upstream evaluator
        subprocess (no container wrapping).

        Resume / idempotency: when a prior ``manifest.json`` exists in
        ``bundle_dir`` and ``self.force_resurface`` is False, surfaces
        whose cache identity matches the prior entry are reused
        verbatim. ``failed`` entries are never reused. Cache identity
        is persisted onto every new entry.

        Trace emission: the orchestrator's native ``sol-execbench``
        invocation produces ``trace.json`` under the bundle dir,
        mirroring the row schema ``run.py`` writes so ``report.py``
        consumes the same bundle. When no eval-driving surface is
        selected, the orchestrator writes a marker ``trace.json`` with
        an empty rows list so the bundle dir is consistently shaped.
        """
        from sol_baselines_harness.surfaces.dispatch_plan import (
            assert_landed_solution_bytes_unchanged,
            build_dispatch_plan,
        )
        slug = baseline_dir.parent.name
        language = baseline_dir.name
        bundle_dir.mkdir(parents=True, exist_ok=True)

        # Resolve the workload subset against the problem's actual
        # workload count, so ``--full-fanout`` materialises a real
        # index list and ``--representative-workload`` is bounds-
        # checked.
        workload_count = _count_workloads(problem_dir)
        workload_subset = _resolve_workload_subset(
            requested=list(self.workload_subset),
            policy=self.workload_selection_policy,
            workload_count=workload_count,
        )

        plan = build_dispatch_plan(
            baseline_dir=baseline_dir,
            problem_dir=problem_dir,
            bundle_dir=bundle_dir,
            host_info=host_info,
            build_mode=self.build_mode,
            workload_subset=workload_subset,
            workload_count=workload_count,
            eval_timeout_s=eval_timeout_s,
            capture_static_so="static" in self.enabled_surfaces,
            enabled_surfaces=self.enabled_surfaces,
        )

        components = native_env_components()
        missing = missing_components(components)
        if missing:
            raise DegradedNativeEnvError(missing, components)
        env_hash = compute_native_env_hash(components)

        manifest = ManifestBuilder(
            slug=slug,
            language=language,
            host_info=host_info,
            build_mode=self.build_mode,
            surface_set=set(self.enabled_surfaces),
        )
        manifest.workload_subset = list(workload_subset)
        manifest.workload_selection_policy = self.workload_selection_policy
        manifest.evaluator_tree_hash = plan.evaluator_tree_hash
        manifest.native_env_hash = env_hash
        manifest.native_env_components = dict(components)
        manifest.eval_timeout_s = plan.eval_timeout_s
        manifest.reference_timing_enabled = plan.reference_timing_enabled
        manifest.config_overrides = dict(plan.config_overrides)
        manifest.dispatch_command = plan.as_manifest_dispatch_command()

        prior_entries = _load_prior_manifest_entries(bundle_dir)

        live_identities: Dict[str, Dict[str, Any]] = {}
        for name in _SURFACE_ORDER:
            if name not in self.enabled_surfaces:
                continue
            argv = self._surface_dispatch_argv(name, plan)
            live_identities[name] = self.compute_surface_cache_identity(
                name=name,
                host_info=host_info,
                constructed_command=argv,
                solution_source_hash=plan.landed_solution_sha256,
                evaluator_tree_hash=plan.evaluator_tree_hash,
                native_env_hash=env_hash,
                config_overrides_hash=_hash_config_overrides(plan.config_overrides),
                workload_subset_signature=plan.workload_subset_signature,
            )

        every_surface_cached = (
            not self.force_resurface
            and bool(self.enabled_surfaces)
            and all(
                prior_entries.get(name) is not None
                and self.surface_cache_hit(prior_entries[name], live_identities[name])
                for name in self.enabled_surfaces
            )
            and _prior_trace_is_valid(bundle_dir)
        )

        if every_surface_cached:
            print("all surfaces cached; nothing to do")
            all_passed_trace = _prior_trace_all_passed(bundle_dir)
        else:
            all_passed_trace = _emit_trace_json(
                bundle_dir=bundle_dir,
                baseline_dir=baseline_dir,
                enabled_surfaces=self.enabled_surfaces,
                plan=plan,
            )

        for name in _SURFACE_ORDER:
            entry = self._invoke_surface(
                name=name,
                baseline_dir=baseline_dir,
                problem_dir=problem_dir,
                bundle_dir=bundle_dir,
                host_info=host_info,
                plan=plan,
                workload_subset=workload_subset,
                prior_entry=prior_entries.get(name),
                live_identity=live_identities.get(name),
            )
            manifest.record_surface(name, entry)

        # Enforce landed-baseline byte immutability before lint+write
        # so an inadvertent mutation fails closed.
        assert_landed_solution_bytes_unchanged(
            baseline_dir / "solution.json",
            plan.landed_solution_sha256,
        )

        # Recompute the manifest's evidence_label using trace state +
        # surface statuses so a manifest cannot over-claim
        # ``Profiled-*`` without an all-PASSED trace.
        manifest._all_passed_trace = all_passed_trace  # noqa: SLF001
        return manifest.lint_and_write(bundle_dir / "manifest.json")

    def _surface_dispatch_argv(self, name: str, plan: Any) -> List[str]:
        """Return the surface-specific argv that would dispatch the eval.

        Used by the resume / cache-identity machinery so the hashed
        command actually reflects each surface's wrapper / env /
        mount contributions, not just the orchestrator's base argv.
        """
        if name not in self.enabled_surfaces:
            return list(plan.base_eval_argv)
        plugin = self.surface_plugins[name]
        # Each plugin's class implements ``dispatch_argv(plan)`` when
        # it customises the eval invocation. Plugins that haven't
        # implemented the hook yet fall back to the base argv.
        builder = getattr(plugin, "dispatch_argv", None)
        if builder is None:
            return list(plan.base_eval_argv)
        try:
            argv, _env, _mounts = builder(plan)
        except Exception:  # noqa: BLE001
            return list(plan.base_eval_argv)
        return list(argv)

    def _invoke_surface(
        self,
        *,
        name: str,
        baseline_dir: Path,
        problem_dir: Path,
        bundle_dir: Path,
        host_info: HostInfo,
        plan: Any,
        workload_subset: List[int],
        prior_entry: Optional[Dict[str, Any]] = None,
        live_identity: Optional[Dict[str, Any]] = None,
    ) -> SurfaceManifestEntry:
        """Drive one surface through probe -> run -> parse.

        A failed run records ``status: failed`` with the
        full failure context but does NOT abort the orchestrator;
        the remaining surfaces still get their turn.
        """
        plugin = self.surface_plugins[name]
        if name not in self.enabled_surfaces:
            return SurfaceManifestEntry(
                status=SURFACE_STATUS_SKIPPED,
                schema_version=plugin.schema_version or None,
                impl_version=plugin.impl_version or None,
                waiver_reason=f"surface {name!r} not in enabled_surfaces",
                start_utc=now_utc_iso(),
                end_utc=now_utc_iso(),
            )
        # Use the orchestrator-supplied live identity when available
        # (resume / cache-identity check is done in run_all before
        # the eval runs); fall back to computing it here so the unit
        # tests that call _invoke_surface directly still work.
        if live_identity is None:
            components = native_env_components()
            missing = missing_components(components)
            if missing:
                raise DegradedNativeEnvError(missing, components)
            env_hash = compute_native_env_hash(components)
            live_identity = self.compute_surface_cache_identity(
                name=name,
                host_info=host_info,
                constructed_command=self._surface_dispatch_argv(name, plan),
                solution_source_hash=plan.landed_solution_sha256,
                evaluator_tree_hash=plan.evaluator_tree_hash,
                native_env_hash=env_hash,
                config_overrides_hash=_hash_config_overrides(plan.config_overrides),
                workload_subset_signature=plan.workload_subset_signature,
            )
        if (
            prior_entry is not None
            and not self.force_resurface
            and self.surface_cache_hit(prior_entry, live_identity)
        ):
            reused = SurfaceManifestEntry(**{
                k: v for k, v in prior_entry.items()
                if k in _MANIFEST_ENTRY_FIELDS
            })
            reused.cache_identity = dict(live_identity)
            # Annotate the substatus so the operator can spot reuse.
            reused.substatus = {**reused.substatus, "cache_reused": True}
            return reused
        probe = plugin.probe(host_info)
        if probe.status != SURFACE_STATUS_OK:
            # Non-ok probe: surface cannot run on this host. Record
            # the probe result verbatim; the manifest captures the
            # ``status`` + ``waiver_reason`` so manifest lint passes.
            return SurfaceManifestEntry(
                status=probe.status,
                tool_version=probe.tool_version,
                schema_version=plugin.schema_version or None,
                impl_version=plugin.impl_version or None,
                waiver_reason=(
                    probe.waiver_reason
                    or f"{name!r} probe returned status={probe.status}"
                ),
                start_utc=now_utc_iso(),
                end_utc=now_utc_iso(),
                cache_identity=dict(live_identity),
            )
        start_utc = now_utc_iso()
        try:
            run_result = plugin.run(
                baseline_dir=baseline_dir,
                problem_dir=problem_dir,
                bundle_dir=bundle_dir,
                build_mode=self.build_mode,
                workload_subset=list(workload_subset),
                dispatch_command=self._surface_dispatch_argv(name, plan),
                plan=plan,
            )
        except Exception as exc:  # noqa: BLE001
            # Per the orchestrator's contract: an uncaught exception in a
            # surface plugin is a framework bug, not a routine
            # ``failed`` status. Re-raise so the orchestrator's
            # caller can surface the traceback; do NOT silently
            # absorb as ``status: failed``.
            raise RuntimeError(
                f"surface {name!r} raised an uncaught exception: {exc!r}"
            ) from exc
        end_utc = run_result.end_utc or now_utc_iso()
        if run_result.status == SURFACE_STATUS_OK:
            parse = plugin.parse(bundle_dir)
        else:
            # Non-ok run paths: give the surface a chance to overlay
            # a typed parse result if the on-disk artifact carries
            # operator-actionable diagnostics the run-step couldn't
            # see (NVBit's invalid_regex flag + filter_stats.json
            # atexit fallback are the canonical case). The hook
            # returns None when no overlay applies; we fall back to
            # the legacy mapping that propagates the run's own waiver.
            failure_overlay = plugin.parse_on_failure(bundle_dir, run_result)
            if failure_overlay is not None:
                parse = failure_overlay
            else:
                parse = SurfaceParseResult(
                    status=run_result.status,
                    artifact_path=None,
                    waiver_reason=run_result.waiver_reason,
                )
        substatus = dict(parse.substatus)
        # Preserve the original run-step failure context when the
        # surface plugin overlaid a typed parse result on top of a
        # failed run. This keeps the eval driver's own waiver visible
        # to operators ("the eval failed because X, AND the tracer
        # reported Y") instead of silently swallowing it.
        if (
            run_result.status != SURFACE_STATUS_OK
            and run_result.waiver_reason
            and parse.waiver_reason
            and parse.waiver_reason != run_result.waiver_reason
        ):
            substatus["run_failure_waiver"] = run_result.waiver_reason
            substatus["run_failure_status"] = run_result.status
        # Surface-specific run metadata (env_overrides, config_overrides,
        # per-tool status, wrapper identifier) lives on run_result.extra
        # and lands on the manifest entry's substatus so the operator
        # can audit per-surface dispatch decisions.
        for key in ("env_overrides", "config_overrides", "per_tool_status",
                    "wrapper", "sample_count", "cupti_bypass_shim_path"):
            if key in run_result.extra:
                substatus[key] = run_result.extra[key]
        return SurfaceManifestEntry(
            status=parse.status,
            tool_version=probe.tool_version,
            schema_version=plugin.schema_version or None,
            impl_version=plugin.impl_version or None,
            command=list(run_result.command),
            exit_code=run_result.exit_code,
            stdout_tail=run_result.stdout_tail,
            stderr_tail=run_result.stderr_tail,
            artifact_path=parse.artifact_path,
            waiver_reason=(parse.waiver_reason or run_result.waiver_reason),
            start_utc=start_utc,
            end_utc=end_utc,
            size_bytes=parse.size_bytes,
            compressed_size_bytes=parse.compressed_size_bytes,
            substatus=substatus,
            cache_identity=dict(live_identity),
        )

    # Resume / idempotency: see ``compute_cache_identity`` +
    # ``surface_cache_hit`` below. The orchestrator wraps every
    # surface invocation with a cache-identity check; reuse occurs
    # only when the stored entry status is non-failed AND every
    # cache-identity field matches the live invocation. ``failed``
    # is never cached. ``--force-resurface`` bypasses the cache.

    def compute_surface_cache_identity(
        self,
        name: str,
        host_info: HostInfo,
        constructed_command: List[str],
        *,
        solution_source_hash: str = "",
        evaluator_tree_hash: str = "",
        native_env_hash: str = "",
        config_overrides_hash: str = "",
        workload_subset_signature: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the cache-identity tuple for one surface.

        Eleven fields participate. The first six are the locked-
        vocabulary axes every plugin must agree on; the remaining
        five are extended axes that affect a produced surface
        artifact's contents without showing up in the joined command
        line. All eleven feed cache-hit decisions in
        :func:`surface_cache_hit`.

        Locked axes:
          * ``command_hash`` — sha256 over the joined command string.
          * ``host_fingerprint`` — sha256 over the host fingerprint
            (kind + compute_capability + driver_version +
            cuda_runtime_version + nvtx_header_present +
            nvtx_library_present).
          * ``workload_subset_signature`` — sorted comma-joined index list.
          * ``surface_schema_version`` — plugin schema_version.
          * ``surface_impl_version`` — plugin impl_version.
          * ``surface_tool_version`` — plugin tool_version from probe.

        Extended axes:
          * ``solution_source_hash`` — sha256 of the staged solution
            (judge or profiling).
          * ``evaluator_tree_hash`` — vendored sol-execbench tree hash.
          * ``native_env_hash`` — SHA-256 of the seven-component native
            host-env fingerprint.
          * ``config_overrides_hash`` — sha256 over the staged reduced
            config contents (warmup_runs / iterations /
            benchmark_reference).
          * ``build_mode`` — judge | profiling.
          * ``env_filter_hash`` — SHA-256 over the
            ``NVBIT_METRICS_USER_ENV_VARS`` allowlist of
            ``METRIC_EXTRACTOR_*`` operator-tunable env vars; only
            nonempty for the ``nvbit_metrics`` surface, empty string
            for every other surface (which never reads those vars).
            Drives cache invalidation when the operator changes any
            metric-extractor-tunable env-var.
        """
        import hashlib
        from sol_baselines_harness.surfaces.nvbit_metrics import (
            NVBIT_METRICS_USER_ENV_VARS,
            compute_env_filter_hash as compute_metrics_env_filter_hash,
        )
        plugin = self.surface_plugins[name]
        cmd_text = " ".join(constructed_command)
        host_text = "|".join((
            host_info.kind,
            host_info.compute_capability,
            host_info.driver_version,
            host_info.cuda_runtime_version,
            str(host_info.nvtx_header_present),
            str(host_info.nvtx_library_present),
        ))
        tool_version = ""
        try:
            tool_version = plugin.probe(host_info).tool_version or ""
        except Exception:  # noqa: BLE001
            tool_version = ""
        signature = (
            workload_subset_signature
            if workload_subset_signature is not None
            else ",".join(str(i) for i in sorted(self.workload_subset))
        )
        if name == "nvbit_metrics":
            # The metric extractor surface owns an operator-tunable
            # env-var allowlist (METRIC_EXTRACTOR_*) and hashes a
            # canonical key=value join into the cache key — every
            # operator-tunable input that changes produced behavior
            # must enter the cache key per
            # ``BL-20260515-cache-key-causal-coverage``.
            metrics_env_view = {
                k: os.environ[k]
                for k in NVBIT_METRICS_USER_ENV_VARS
                if k in os.environ
            }
            env_filter_hash = compute_metrics_env_filter_hash(metrics_env_view)
        else:
            # Surfaces other than the NVBit-injecting metric extractor
            # never read the user-module filter env-var set, so the
            # field is intentionally empty and cannot ever invalidate
            # their cache.
            env_filter_hash = ""
        return {
            "command_hash": hashlib.sha256(cmd_text.encode("utf-8")).hexdigest(),
            "host_fingerprint": hashlib.sha256(host_text.encode("utf-8")).hexdigest(),
            "workload_subset_signature": signature,
            "surface_schema_version": plugin.schema_version,
            "surface_impl_version": plugin.impl_version,
            "surface_tool_version": tool_version,
            "solution_source_hash": solution_source_hash,
            "evaluator_tree_hash": evaluator_tree_hash,
            "native_env_hash": native_env_hash,
            "config_overrides_hash": config_overrides_hash,
            "build_mode": self.build_mode,
            "env_filter_hash": env_filter_hash,
        }

    def surface_cache_hit(
        self,
        prior_entry: Dict[str, Any],
        live_identity: Dict[str, Any],
    ) -> bool:
        """Decide whether a prior manifest entry can be reused.

        Reuse requires:
        * Stored ``status`` is non-failed (failures are never cached).
        * Every cache-identity field matches exactly.

        ``--force-resurface`` is enforced at the call site by
        ``run_all`` bypassing this method entirely.
        """
        prior_status = prior_entry.get("status")
        if prior_status not in ("ok", "unavailable", "blocked", "skipped"):
            return False
        prior_cache = prior_entry.get("cache_identity") or {}
        for field_name in (
            "command_hash",
            "host_fingerprint",
            "workload_subset_signature",
            "surface_schema_version",
            "surface_impl_version",
            "surface_tool_version",
            "solution_source_hash",
            "evaluator_tree_hash",
            "native_env_hash",
            "config_overrides_hash",
            "build_mode",
            "env_filter_hash",
        ):
            if prior_cache.get(field_name) != live_identity.get(field_name):
                return False
        return True


def assert_b200_fanout_ready(
    manifest_path: Path,
    *,
    strict_b200_target: bool = False,
) -> None:
    """Refuse to start a fanout when the preflight manifest is unsatisfied.

    Called by orchestration scripts (and the surfaces CLI) BEFORE
    fanning out to a multi-pilot run. Raises ``RuntimeError`` with an
    actionable message when:

    * The manifest file is missing.
    * The manifest reports ``ready_for_phase_2: false``.
    * ``strict_b200_target=True`` AND the manifest's ``device`` probe
      doesn't report ``phase_2_target: true`` (i.e. the host is not a
      compute-capability sm_100 family device).
    """
    import json as _json
    if not manifest_path.is_file():
        raise RuntimeError(
            f"fanout refused: {manifest_path} does not exist. Run "
            f"`python scripts/baselines_harness/b200_preflight.py` first."
        )
    try:
        manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
    except _json.JSONDecodeError as exc:
        raise RuntimeError(
            f"fanout refused: cannot parse {manifest_path}: {exc}"
        ) from exc
    if not manifest.get("ready_for_phase_2"):
        blockers = manifest.get("blockers", [])
        raise RuntimeError(
            f"fanout refused: {manifest_path} reports "
            f"ready_for_phase_2=false. Resolve blockers {blockers} "
            f"and re-probe."
        )
    if strict_b200_target:
        device = manifest.get("checks", {}).get("device", {})
        if not device.get("phase_2_target"):
            raise RuntimeError(
                "fanout refused: strict B200 target requires the device "
                "probe to report phase_2_target=true (compute capability "
                "in the sm_100 family)."
            )
