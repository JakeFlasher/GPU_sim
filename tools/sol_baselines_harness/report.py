#!/usr/bin/env python3
"""Aggregate harness report generator.

Walks every landed baseline under ``baselines/<cat>/<slug>/<lang>/``,
either replays the latest cached harness trace (when one exists) or
runs the baseline fresh, then emits a JSON + Markdown aggregate report
to ``tools/sol_baselines_harness/report.{json,md}``.

Each landed baseline contributes one row containing the slowest-
workload ``speedup_factor``, the per-baseline ``local_sol_score``
(or ``"unknown_t_sol"`` when the resolved arch yaml is absent or
the precision is unsupported), plus the resolved ``arch_tag``,
``arch_yaml_path``, ``arch_resolution_reason`` metadata, and a
pointer to the trace JSON used. The aggregate report carries:

- A local-vs-leaderboard disclaimer for every row.
- The deferred-ledger contents (entry count + per-entry status).
- A lint pre-flight summary (forbidden-surface scan + deferred-ledger
  hygiene); the report refuses to write when either fails.

Usage:
    python tools/sol_baselines_harness/report.py
    python tools/sol_baselines_harness/report.py --output-json /tmp/r.json
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent
REPO_ROOT = _SCRIPTS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from tools.sol_baselines_harness._bootstrap import install_package_aliases

install_package_aliases()

from baselines_scaffold.generate import (  # noqa: E402
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_OUTPUT_ROOT,
    SUBSET_MAP,
)
from baselines_scaffold.validate import _iter_all_baselines  # noqa: E402
from baselines_harness.score import (  # noqa: E402
    UNKNOWN_T_SOL,
    local_arch_config_available,
    local_proxy_disclaimer,
    local_score_for_workloads,
    resolved_arch_yaml,
    slowest_speedup,
    upper_bound_score_t_sol_zero,
)
from baselines_harness.arch_resolver import (  # noqa: E402
    arch_resolution_reason,
    arch_tag_from_host_kind,
    resolve_local_arch_yaml,
)
from baselines_harness.cache import (  # noqa: E402
    cache_record_for_key,
    current_baseline_key,
    is_passed_trace,
    key_matches_record,
    load_cache_index,
    save_cache_index,
    sol_execbench_source_hash,
)
from baselines_harness.lint import (  # noqa: E402
    scan_harness_production_code,
    scan_harness_production_text_for_container_terms,
    scan_harness_production_text_for_workflow_history_terms,
    scan_landed_baselines,
    validate_deferred_ledger,
)
from baselines_harness.native_env import (  # noqa: E402
    DegradedNativeEnvError,
    compute_native_env_hash,
    missing_components,
    native_env_components,
)
from baselines_harness.run import run_baseline  # noqa: E402
from baselines_harness.surfaces.base import (  # noqa: E402
    SURFACE_STATUS_SKIPPED,
    VALID_SURFACES,
)
from baselines_harness.surfaces.host_probe import probe_host  # noqa: E402
from baselines_harness.surfaces.manifest import (  # noqa: E402
    ManifestBuilder,
    ManifestLintError,
    SurfaceManifestEntry,
    lint_manifest,
)


def _baseline_metadata(baseline_dir: Path) -> Dict[str, str]:
    """Extract category / slug / language from the path layout."""
    parents = list(baseline_dir.resolve().parents)
    return {
        "category": parents[1].name,
        "slug": parents[0].name,
        "language": baseline_dir.name,
    }


_DEFAULT_TRACES_DIR = _THIS_DIR / ".harness_traces"


def _bundle_dir(
    baseline_dir: Path, host_kind: Optional[str] = None
) -> Path:
    """Compute the bundle directory for a baseline, routing through
    the cross-arch trace-layout contract.

    ``baselines/<cat>/<slug>/<lang>/`` ->
      ``.harness_traces/<slug>__<lang>/`` when ``host_kind`` is
        ``None`` / ``local_sm_120`` / ``b200_sm_100a`` (legacy
        un-suffixed canonical form; reads of the 86 sm_120 R0-R23
        bundles stay at this path).
      ``.harness_traces/<slug>__<lang>__sm86/`` when ``host_kind ==
        "local_sm_8_6"`` (the active 3090 Ti host's sm_86 sibling
        form per DEC-B).

    Delegates to ``baselines_harness.trace_layout.default_bundle_dir``
    so the surfaces CLI and report walker share one resolver. report.py
    callers that walk EXISTING bundles (i.e. iterating sm_120 R0-R23
    evidence whose host_kind is not yet known to the walker) pass
    ``host_kind=None`` and get the un-suffixed canonical path, matching
    the on-disk layout.

    The bundle directory hosts ``trace.json`` (the existing per-workload
    JSON) plus, once the surfaces orchestrator runs, ``manifest.json``
    and one subdirectory per analytic surface.
    """
    from baselines_harness.trace_layout import default_bundle_dir
    return default_bundle_dir(
        traces_root=_DEFAULT_TRACES_DIR,
        slug=baseline_dir.parent.name,
        language=baseline_dir.name,
        host_kind=host_kind,
    )


def _active_host_kind() -> Optional[str]:
    """Return the live host's ``host.kind`` token (e.g.
    ``"local_sm_8_6"`` on the 3090 Ti) via the cached
    :func:`_cached_host_info` probe. Used by :func:`_active_bundle_dir`
    to route fresh sm_86 writes/reads to the ``__sm86`` arch-qualified
    sibling, leaving the locked sm_120 R0-R23 canonical paths
    untouched. Returns ``None`` only if the host probe itself is
    unable to classify the active host (degraded state)."""
    info = _cached_host_info()
    return getattr(info, "kind", None)


def _active_bundle_dir(baseline_dir: Path) -> Path:
    """Resolve the bundle directory for FRESH writes / post-write
    reads on the live host. Delegates to :func:`_bundle_dir` with
    ``host_kind`` resolved via :func:`_active_host_kind`.

    Inside ``report.py``'s fresh-run path (`_persist_fresh_trace`,
    `_persist_fresh_manifest`, post-write identity reads, replay
    candidate canonical comparison, NVBit metrics summary), every
    bundle reference goes through this helper so the 3090 Ti host
    writes to ``<slug>__<lang>__sm86/`` and not the locked sm_120
    canonical path. Codex review required this to satisfy
    the design contract — see the cross-arch plan + trace-layout contract for the
    DEC-B sibling-dir scheme.

    Legacy un-suffixed reads (walking historical sm_120 R0-R23
    evidence) MUST call :func:`_bundle_dir` directly with
    ``host_kind=None`` so the routing remains explicit, not
    accidental."""
    return _bundle_dir(baseline_dir, host_kind=_active_host_kind())


_JOINED_ROWS_REPORT_CAP = 64
_LAUNCHES_STATIC_JOINED_FILENAME = "launches.static_joined.jsonl"


def _nvbit_metrics_summary_for_bundle(
    bundle_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Return the per-bundle nvbit_metrics summary: static-sidecar
    join + wall-clock cross-surface join.

    Returns ``None`` when the bundle did not run the nvbit_metrics
    surface (no ``nvbit_metrics/launches.jsonl`` artifact). Otherwise
    returns a dict carrying:

    * ``records_parsed`` — count of NVBit records.
    * ``parse_errors`` — count of strict-parser errors (non-zero means
      the surface's manifest entry already records the failure; the
      summary still emits the count for visibility).
    * ``static_sidecar_join`` — dict mapping
      ``{matched, unjoinable_no_ptxas_v_artifact,
      unjoinable_kernel_not_in_ptxas_v,
      unjoinable_unknown_arch_limits,
      skipped_graph_launch_not_observed}`` to count.
    * ``wall_clock_join`` — the wall-clock ``join_status`` bucket
      counts ``{matched, ambiguous, unmatched_nvbit_only,
      unmatched_cupti_only, unjoinable}``.
    * ``joined_rows`` — a bounded (cap = ``_JOINED_ROWS_REPORT_CAP``)
      list of per-launch enriched records carrying
      ``kernel_name_mangled``, ``static_sidecar``,
      ``static_sidecar_join_status``, ``wall_clock_join_status``,
      ``duration_ns`` (when matched), and ``waiver_reason``. Report
      consumers must be able to inspect which launch matched which
      CUPTI duration and which static resources were joined, not
      just bucket counts.
    * ``launches_static_joined_path`` — repo-relative path to the
      bundle-local artifact ``nvbit_metrics/launches.static_joined.jsonl``
      this helper writes (always, when records exist); the artifact
      is the unbounded source of truth for the joined data, while
      ``joined_rows`` is the report-side truncated preview.

    Per Codex review #5 directive: the per-bundle summary carries the
    bucket counts AND a bounded payload of joined rows AND persists the
    unbounded per-launch detail to a bundle-local artifact.
    """
    nvbit_path = bundle_dir / "nvbit_metrics" / "launches.jsonl"
    if not nvbit_path.is_file():
        return None
    from baselines_harness.metric_extractor.launches_parser import (
        parse_launches_jsonl,
    )
    from baselines_harness.metric_extractor.static_sidecar import (
        join_static_sidecar,
    )
    from baselines_harness.metric_extractor.wall_clock_join import (
        join_bundle,
        nvbit_join_key,
    )

    records, errors = parse_launches_jsonl(nvbit_path)
    ptxas_path = bundle_dir / "static" / "ptxas-v.txt"
    ptxas_text: Optional[str] = None
    if ptxas_path.is_file():
        try:
            ptxas_text = ptxas_path.read_text(encoding="utf-8")
        except OSError:
            ptxas_text = None
    resource_usage_path = bundle_dir / "static" / "resource-usage.txt"
    resource_usage_text: Optional[str] = None
    if resource_usage_path.is_file():
        try:
            resource_usage_text = resource_usage_path.read_text(encoding="utf-8")
        except OSError:
            resource_usage_text = None

    # Best-effort arch resolution from the bundle manifest's
    # host.compute_capability, since the static sidecar joiner needs
    # an arch tag to compute theoretical occupancy. Falls back to
    # ``None`` -> the joiner emits unjoinable_unknown_arch_limits, per
    # Codex directive #4.
    arch = _bundle_arch_tag(bundle_dir)

    enriched = join_static_sidecar(
        [r.raw for r in records],
        ptxas_v_text=ptxas_text,
        resource_usage_text=resource_usage_text,
        arch=arch,
    )
    sidecar_buckets: Dict[str, int] = {}
    for rec in enriched:
        status = rec.get("static_sidecar_join_status", "unknown")
        sidecar_buckets[status] = sidecar_buckets.get(status, 0) + 1

    wall_clock_summary = join_bundle(bundle_dir)
    # The wall-clock attribution lookup uses the FULL composite tuple
    # as its key, not the partial (mangled_name, ordinal) spine. The
    # full tuple is the joiner's canonical identifier for a matched
    # launch pair; downstream consumers must be able to audit which
    # exact key the wall-clock duration is associated with.
    wc_status_by_key: Dict[Tuple[Any, ...], str] = {}
    wc_duration_by_key: Dict[Tuple[Any, ...], Optional[int]] = {}
    wc_waiver_by_key: Dict[Tuple[Any, ...], Optional[str]] = {}
    if wall_clock_summary is not None:
        wc_buckets = dict(wall_clock_summary.bucket_counts)
        nvbit_total = wall_clock_summary.nvbit_total
        cupti_total = wall_clock_summary.cupti_total
        for row in wall_clock_summary.rows:
            if row.join_key is None:
                continue
            wc_status_by_key[row.join_key] = row.join_status
            wc_duration_by_key[row.join_key] = row.duration_ns
            wc_waiver_by_key[row.join_key] = row.waiver_reason
    else:
        wc_buckets = {}
        nvbit_total = len(records)
        cupti_total = 0

    joined_artifact_rows: List[Dict[str, Any]] = []
    for rec in enriched:
        cs = rec.get("cross_surface_join_key") or {}
        # Use the same strict key builder as the wall-clock joiner so
        # context/stream namespace rules cannot drift between the two
        # report paths.
        full_key = nvbit_join_key(rec)
        wc_status = (
            wc_status_by_key.get(full_key) if full_key is not None else None
        )
        joined_artifact_rows.append({
            # Carry the full composite tuple as the operator-auditable
            # join key. ``join_key`` is the canonical row identifier;
            # the auxiliary fields below (``kernel_name_mangled``,
            # ``kernel_ordinal_in_workload``, etc.) remain as
            # convenience accessors.
            "join_key": (
                list(full_key) if full_key is not None else None
            ),
            "demangled_kernel_name": cs.get("demangled_kernel_name"),
            "kernel_name_mangled": rec.get("kernel_name_mangled"),
            "kernel_ordinal_in_workload": cs.get("kernel_ordinal_in_workload"),
            "workload_index": cs.get("workload_index"),
            "evaluator_config_hash": cs.get("evaluator_config_hash"),
            "launch_kind": rec.get("launch_kind"),
            "grid_dim": rec.get("grid_dim"),
            "block_dim": rec.get("block_dim"),
            "dynamic_smem": rec.get("dynamic_smem"),
            "stream_ordinal": cs.get("stream_ordinal"),
            "context_id": cs.get("context_id"),
            "static_sidecar": rec.get("static_sidecar"),
            "static_sidecar_join_status": rec.get(
                "static_sidecar_join_status"
            ),
            "wall_clock_join_status": wc_status,
            "duration_ns": (
                wc_duration_by_key.get(full_key)
                if full_key is not None
                else None
            ),
            "waiver_reason": (
                wc_waiver_by_key.get(full_key)
                if full_key is not None
                else None
            ),
        })

    # Persist the unbounded joined-row artifact under the bundle dir
    # so downstream consumers (papers, manual inspection, follow-up
    # rounds) can read the full evidence without re-running the join.
    artifact_path_rel: Optional[str] = None
    if joined_artifact_rows:
        try:
            artifact_full = (
                bundle_dir / "nvbit_metrics"
                / _LAUNCHES_STATIC_JOINED_FILENAME
            )
            artifact_full.parent.mkdir(parents=True, exist_ok=True)
            with artifact_full.open("w", encoding="utf-8") as fh:
                for row in joined_artifact_rows:
                    fh.write(
                        json.dumps(row, separators=(",", ":"),
                                   sort_keys=True) + "\n"
                    )
            try:
                artifact_path_rel = str(
                    artifact_full.resolve().relative_to(REPO_ROOT)
                )
            except ValueError:
                artifact_path_rel = str(artifact_full)
        except OSError:
            # Best-effort write — when the bundle dir is read-only the
            # joined rows still travel in the in-memory summary, just
            # without the on-disk durable copy.
            artifact_path_rel = None

    return {
        "records_parsed": len(records),
        "parse_errors": len(errors),
        "nvbit_total": nvbit_total,
        "cupti_total": cupti_total,
        "static_sidecar_join": sidecar_buckets,
        "wall_clock_join": wc_buckets,
        "joined_rows": joined_artifact_rows[:_JOINED_ROWS_REPORT_CAP],
        "joined_rows_total": len(joined_artifact_rows),
        "joined_rows_truncated": (
            len(joined_artifact_rows) > _JOINED_ROWS_REPORT_CAP
        ),
        "launches_static_joined_path": artifact_path_rel,
        "arch_for_occupancy": arch,
    }


def _bundle_arch_tag(bundle_dir: Path) -> Optional[str]:
    """Return the arch tag for occupancy computation from a bundle's
    manifest, or ``None`` when the manifest is missing / unparseable.

    Reads ``manifest.json::host.compute_capability`` (e.g. ``"8.6"``,
    ``"10.0"``, ``"12.0"``) and maps to the canonical arch tag the
    static sidecar joiner understands (``sm_86``, ``sm_100a``,
    ``sm_120``). Unrecognised compute capabilities return ``None`` so
    the joiner reports ``unjoinable_unknown_arch_limits``.
    """
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    host = payload.get("host", {}) if isinstance(payload, dict) else {}
    cc = host.get("compute_capability") if isinstance(host, dict) else None
    if not isinstance(cc, str):
        return None
    # Map "8.6" -> "sm_86", "10.0" -> "sm_100a" (project shorthand for
    # the datacenter Blackwell target), "12.0" -> "sm_120". Other
    # values fall through to None.
    if cc == "8.6":
        return "sm_86"
    if cc == "10.0":
        return "sm_100a"
    if cc == "12.0":
        return "sm_120"
    return None


def _gather_baseline_row(
    baseline_dir: Path,
    benchmark_root: Path,
    trace_dir: Optional[Path],
    cache_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Produce a single per-baseline report row.

    Strategy:
      1. Look for a candidate trace first under the report's own
         ``.harness_traces/<slug>__<lang>.json`` (which is where this
         function stamps fresh runs), then under any ``--trace-dir``
         the caller supplied (handy for the loop-state traces under
         ``.humanize/rlcr/<loop>/round-*-trace.json``).
      2. A candidate replays only when the persisted ``cache_index``
         record for that trace path matches the freshly derived
         ``CacheKey`` exactly AND every row is ``PASSED``. Speedup
         vs the PyTorch reference is recorded as diagnostic data
         only; a row's ``speedup_factor`` does NOT gate replay or
         landing under the correctness-only landing contract.
      3. After a fresh run, write the trace to the default per-baseline
         path and stamp the cache index so the next report invocation
         can replay.

    Replay never accepts a slug-substring-only match — every component
    of the key (solution source hash, ``sol_execbench_source_hash``
    (tree hash of the vendored ``sol-execbench/`` subdirectory),
    ``native_env_hash`` (SHA-256 over python / torch / nvcc /
    cuda-runtime / cutlass-tree / cudnn / sol_execbench-package /
    nvcc-ccbin / cxx / cc; see :mod:`baselines_harness.native_env`),
    evaluator environment libs) must agree. ``native_env_hash`` uses
    strict equality on every axis — any disagreement drops the cache
    record and re-runs the eval. A degraded host environment that
    left any component as ``"unknown"`` raises
    :class:`DegradedNativeEnvError`; this function catches the
    exception at both the candidate-key derivation and the cache-
    stamp call so a degraded host gets a fresh eval + persisted
    trace but no cache record, instead of aborting the whole report.
    """
    meta = _baseline_metadata(baseline_dir)
    trace_candidate = _find_trace_candidate(baseline_dir, trace_dir)
    rows: List[Dict[str, Any]] = []
    replayed_trace_path: Optional[Path] = None
    # Stash the existing-on-disk trace's rows for use as a fresh-run
    # fallback. When cache replay declines (key mismatch / source
    # change / non-PASSED rows) the fresh run still fires — but if
    # the fresh run produces zero rows (host can't complete the eval;
    # upstream CLI times out before any workload row lands), we fall
    # back to the planted on-disk trace so the report has honest
    # evidence instead of either dropping the baseline or tripping
    # the integrity gate.
    fallback_cand_rows: List[Dict[str, Any]] = []
    fallback_cand_path: Optional[Path] = None
    if trace_candidate is not None:
        cand_path, cand_payload = trace_candidate
        cand_rows = cand_payload.get("rows", [])
        env_libs = _environment_libs_from_trace(cand_payload, cand_rows)
        # A degraded native-env fingerprint (any component reading
        # UNKNOWN) makes the current cache key uncomputable. The
        # right semantic is "this bundle is not replayable under
        # the current identity" — fall through to a fresh eval
        # rather than aborting report generation. The cache layer
        # already refuses to match degraded records by construction;
        # this just keeps the report flow alive on a degraded host.
        try:
            current_key = current_baseline_key(
                baseline_dir, environment_libs=env_libs
            )
        except DegradedNativeEnvError:
            current_key = None
        stored_record = cache_index.get(str(cand_path))
        identity_matches = (
            current_key is not None
            and stored_record is not None
            and key_matches_record(current_key, stored_record)
        )
        if (
            identity_matches
            and is_passed_trace(cand_rows)
            and _trace_is_clean_full(cand_payload)
        ):
            rows = cand_rows
            replayed_trace_path = cand_path
        elif (
            not identity_matches
            and is_passed_trace(cand_rows)
            and _trace_is_clean_full(cand_payload)
            and cand_path == _active_bundle_dir(baseline_dir) / "trace.json"
            and _trace_env_libs_match_current(env_libs)
            and _manifest_identity_matches_current(
                baseline_dir=baseline_dir,
            )
        ):
            # Manifest-anchored replay path:
            # cache_index has no record (e.g., clean checkout — the
            # gitignored ``.harness_cache.json`` was never created),
            # but the bundle's committed v2 manifest stamps an
            # identity that matches current code/env/host AND the
            # on-disk trace is all-PASSED. Treat the bundle as if
            # cache-replay fired — use the existing trace as report
            # evidence, skip the fresh-run path entirely so
            # ``_persist_fresh_manifest`` is never invoked and the
            # committed manifest stays intact. This preserves the
            # canonical pilot's orchestrator-stamped surface
            # evidence across clean checkouts without the preserve
            # guard having to reason about dispatch-config
            # divergence (the orchestrator may have used reduced
            # profiling config; report.py's fresh-run uses default
            # config; the surfaces describe one dispatch and the
            # trace describes another — the right answer is to just
            # keep both together, not stitch them). an earlier iteration dropped
            # the ``current_solution_source_hash`` argument: the
            # helper computes it itself using the manifest scheme
            # (``_solution_file_sha256``) instead of the cache
            # scheme (canonical-JSON hash); Codex review
            # caught the scheme mismatch.
            rows = cand_rows
            replayed_trace_path = cand_path
        elif (
            identity_matches
            and cand_rows
            and not _trace_has_structural_failure(cand_payload)
        ):
            # Fresh-run-failure fallback is only safe when the
            # on-disk trace's identity is verifiably current
            # (cache key still matches the freshly-computed
            # baseline identity). A key-mismatched candidate
            # belongs to a previous solution / evaluator / env
            # generation, so reusing its rows after a fresh-run
            # zero-row failure would promote stale evidence as
            # evidence for changed code. Such candidates are
            # NOT stashed; the report falls through to a
            # legitimate fresh run, and bundles whose fresh run
            # cannot produce rows on this host AND have no
            # cached identity proof are excluded from the report
            # scope via the in-tree exclusion ledger.
            fallback_cand_rows = cand_rows
            fallback_cand_path = cand_path
    fresh_trace_path: Optional[Path] = None
    if not rows:
        outcome = run_baseline(
            baseline_dir=baseline_dir,
            benchmark_root=benchmark_root,
            workers=1,
            timeout_seconds=900.0,
            print_cmd=False,
        )
        rows = outcome.rows
        if not rows and fallback_cand_rows:
            # Fresh-run-failure on-disk trace fallback: the harness
            # didn't produce a single workload row (e.g. the upstream
            # eval timed out before any row landed, as observed for
            # problem_008's MoE-routing kernel under a 900 s budget).
            # Use the planted bundle trace as honest evidence so the
            # report has a row with all_passed=False / non-null
            # trace path instead of either dropping the baseline or
            # tripping the evidence-integrity gate. The cache is NOT
            # stamped on this path — non-PASSED on-disk rows remain
            # non-replayable and the next report run with valid
            # cache material will still trigger a fresh-run
            # replacement.
            rows = fallback_cand_rows
            replayed_trace_path = fallback_cand_path
        if rows and replayed_trace_path is None:
            # Persist the bundle trace.json whenever the native eval
            # produced any rows — including honest non-PASSED rows like
            # INVALID_REFERENCE. Trace persistence is decoupled from
            # cache trust: the rows are evidence regardless of whether
            # they're replay-worthy.
            fresh_env_libs = _environment_libs_from_rows(rows)
            fresh_trace_path = _persist_fresh_trace(
                baseline_dir,
                rows=rows,
                env_libs=fresh_env_libs,
            )
            # Manifest stamping is independent of cache trust. Every
            # bundle that has a trace.json MUST also have a v2-native
            # manifest.json so _find_trace_candidate's acceptance
            # gate succeeds on the next report run. Without this,
            # cache replay is silently broken for every trace-only
            # bundle.
            _persist_fresh_manifest(
                baseline_dir,
                rows=rows,
                eval_timeout_s=900.0,
            )
            # Cache stamping stays gated on cache-trustworthy runs only:
            # outcome.is_clean() AND every row PASSED. Non-PASSED rows
            # surface in the bundle + report but never replay as PASSED.
            #
            # A degraded native-env fingerprint (any component reading
            # UNKNOWN) makes the cache key uncomputable. The trace is
            # already persisted via _persist_fresh_trace + the manifest
            # writer above; the only thing we skip on degraded is the
            # cache record. The report still emits with this row in it,
            # and the next report run will re-execute (correct on a
            # degraded host — its evidence must never serve as replay).
            if outcome.is_clean() and is_passed_trace(rows):
                try:
                    _stamp_cache_record(
                        baseline_dir,
                        fresh_trace_path,
                        env_libs=fresh_env_libs,
                        cache_index=cache_index,
                    )
                except DegradedNativeEnvError as exc:
                    print(
                        f"[report] degraded native env: skipping cache "
                        f"stamp for {baseline_dir} ({exc.missing}); "
                        f"trace persisted at {fresh_trace_path}",
                        file=sys.stderr,
                    )
    trace_path_obj = (
        replayed_trace_path if replayed_trace_path is not None
        else fresh_trace_path
    )
    # Write repo-relative paths to report.json so the artifact stays
    # portable across checkouts; the bundle-trace integrity gate
    # resolves these against REPO_ROOT before checking ``is_file()``.
    if trace_path_obj is None:
        trace_for_report: Optional[str] = None
    else:
        try:
            trace_for_report = str(trace_path_obj.resolve().relative_to(REPO_ROOT))
        except ValueError:
            trace_for_report = str(trace_path_obj)
    all_passed = bool(rows) and all(r.get("status") == "PASSED" for r in rows)
    # `meets_speed_gate` is diagnostic-only: under the correctness-
    # only landing contract, `all_passed` alone determines the
    # evidence label and category-floor membership. The field stays
    # in the row output so operators can still see whether a kernel
    # beat the PyTorch reference on every workload.
    speed_gate = _meets_speed_gate(rows)
    meets_speed_gate = all_passed and speed_gate
    # Per-baseline context the SOLAR Stage 4 score path needs.
    slug = meta["slug"]
    category = meta["category"]
    subset_name = SUBSET_MAP.get(category)
    solution_path = baseline_dir / "solution.json"
    benchmark_name: Optional[str] = None
    if solution_path.is_file():
        try:
            benchmark_name = json.loads(
                solution_path.read_text(encoding="utf-8")
            ).get("definition")
        except json.JSONDecodeError:
            benchmark_name = None

    arch_context = _arch_context_for_bundle(baseline_dir=baseline_dir)
    nvbit_metrics_summary = _nvbit_metrics_summary_for_bundle(
        bundle_dir=_active_bundle_dir(baseline_dir),
    )
    return {
        **meta,
        "trace": trace_for_report,
        "workload_count": len(rows),
        "all_passed": all_passed,
        "meets_speed_gate": meets_speed_gate,
        "slowest_speedup": slowest_speedup(rows),
        # nvbit_metrics cross-surface summary: per-bundle T16 (static
        # sidecar join + wall-clock CUPTI join bucket counts. ``None``
        # when the bundle did not run the nvbit_metrics surface;
        # otherwise a dict carrying ``records_parsed``,
        # ``static_sidecar_join`` (4 buckets), and ``wall_clock_join``
        # (5 buckets). Every launch from either side lands in
        # exactly one ``join_status`` bucket so the operator can
        # audit cross-surface coverage from this row alone.
        "nvbit_metrics_summary": nvbit_metrics_summary,
        # `local_sol_score` is the canonical local SOL Score column.
        # The SOLAR Stage 4 estimator derives a per-workload `t_sol`
        # via `solar_analysis.t_sol_for_baseline_workload`, the
        # precision guard fails closed on missing MAC keys, and the
        # helper returns the median `compute_score` across rows.
        # Pilots whose slug lacks a registered analysis builder OR
        # whose precision is unsupported on the resolved arch emit
        # `unknown_t_sol` honestly.
        "local_sol_score": local_score_for_workloads(
            rows,
            slug=slug,
            benchmark_root=benchmark_root,
            subset=subset_name,
            benchmark_name=benchmark_name,
        ),
        # Arch metadata: identifies the roofline this row was scored
        # against. Downstream consumers (papers, audit tools) can
        # disambiguate runs without re-deriving the host->yaml map.
        "arch_tag": arch_context["arch_tag"],
        "arch_yaml_path": arch_context["arch_yaml_path"],
        "arch_resolution_reason": arch_context["arch_resolution_reason"],
        # `upper_bound_score_t_sol_zero` is a separate diagnostic
        # column: the median of `compute_score(t_k, t_b, 0.0)` across
        # rows. The field name is historical — the SOL Score is
        # monotonic-increasing in t_sol for speedup rows, so this
        # value is a LOWER diagnostic baseline for the canonical
        # column, NOT an upper bound. See
        # score.upper_bound_score_t_sol_zero docstring. The JSON key
        # is retained for backward compatibility.
        "upper_bound_score_t_sol_zero": upper_bound_score_t_sol_zero(rows),
        # Evidence label is the externally meaningful claim: it goes
        # into parent READMEs and the per-row classification.
        #
        # Layered upgrade:
        #   `Compile-checked-<arch>` — compiles but doesn't pass
        #       correctness.
        #   `Ran-local-<arch>` — every workload PASSED correctness
        #       (no manifest required).
        #   `Profiled-<host>-<arch>` — trace all PASSED AND manifest
        #       exists AND no surface in `failed` state.
        #   `Profiled-<host>-complete` — trace all PASSED AND every
        #       surface in `ok` state.
        #
        # The arch tag is resolved from the bundle's manifest host
        # kind when present, otherwise from the live host probe
        # (so a not-yet-profiled row on a 3090 Ti still labels as
        # ``Compile-checked-sm_86`` instead of leaking the legacy
        # sm_120 literal).
        "evidence_label": _evidence_label_for_bundle(
            baseline_dir=baseline_dir,
            all_passed=all_passed,
            arch_context=arch_context,
        ),
    }


def _arch_context_for_bundle(
    *, baseline_dir: Path
) -> Dict[str, Any]:
    """Resolve ``(arch_tag, arch_yaml_path, arch_resolution_reason)``
    for this baseline. Prefers the bundle's ``manifest.json`` host
    fingerprint over the live-host probe so historical bundles keep
    their original arch labels.
    """
    manifest_path = _active_bundle_dir(baseline_dir) / "manifest.json"
    host_info_obj: object = None
    if manifest_path.is_file():
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            host_info_obj = manifest.get("host", {})
        except json.JSONDecodeError:
            host_info_obj = None
    if host_info_obj is None:
        try:
            host_info_obj = probe_host()
        except Exception:
            host_info_obj = None

    kind: str
    if isinstance(host_info_obj, dict):
        kind = str(host_info_obj.get("kind", "unknown"))
    else:
        kind = str(getattr(host_info_obj, "kind", "unknown"))

    arch_tag = arch_tag_from_host_kind(kind)
    arch_yaml = (
        resolve_local_arch_yaml(host_info_obj)
        if host_info_obj is not None
        else None
    )
    if arch_yaml is None:
        arch_yaml_str = "unknown"
    else:
        try:
            arch_yaml_str = str(arch_yaml.resolve().relative_to(REPO_ROOT))
        except ValueError:
            arch_yaml_str = str(arch_yaml)

    return {
        "arch_tag": arch_tag,
        "arch_yaml_path": arch_yaml_str,
        "arch_resolution_reason": (
            arch_resolution_reason(host_info_obj)
            if host_info_obj is not None
            else "host probe unavailable"
        ),
        "host_kind": kind,
    }


def _evidence_label_for_bundle(
    *,
    baseline_dir: Path,
    all_passed: bool,
    arch_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Compute the evidence label for this baseline.

    The label is gated on:

    1. ``trace.json`` rows all-PASSED (correctness gate).
    2. ``manifest.json`` exists in the bundle directory.
    3. No surface in `failed` state -> `Profiled-*-<arch>`.
    4. Every surface in `ok` state -> `Profiled-*-complete`.

    Falls back to ``Ran-local-<arch>`` (correctness gate but no
    manifest) and ``Compile-checked-<arch>`` (no correctness).

    ``arch_context`` is the dict returned by
    ``_arch_context_for_bundle``; if omitted, this function recomputes
    it (callers that already built it should pass it in to avoid the
    duplicate host probe).
    """
    if arch_context is None:
        arch_context = _arch_context_for_bundle(baseline_dir=baseline_dir)
    arch_tag = arch_context.get("arch_tag", "sm_unknown")
    host_kind = arch_context.get("host_kind", "")

    if not all_passed:
        return f"Compile-checked-{arch_tag}"
    manifest_path = _active_bundle_dir(baseline_dir) / "manifest.json"
    if not manifest_path.is_file():
        return f"Ran-local-{arch_tag}"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return f"Ran-local-{arch_tag}"
    surfaces = manifest.get("surfaces", {})
    if not isinstance(surfaces, dict) or not surfaces:
        return f"Ran-local-{arch_tag}"
    statuses = [
        entry.get("status") for entry in surfaces.values()
        if isinstance(entry, dict)
    ]
    if any(s == "failed" for s in statuses):
        return f"Ran-local-{arch_tag}"

    if isinstance(host_kind, str) and host_kind.startswith("b200"):
        host_seg = "B200"
    else:
        host_seg = "local"

    if all(s == "ok" for s in statuses):
        return f"Profiled-{host_seg}-complete"
    return f"Profiled-{host_seg}-{arch_tag}"


def _meets_speed_gate(rows: List[Dict[str, Any]]) -> bool:
    """Diagnostic: True when every row has a numeric ``speedup_factor > 1.0``.

    Under the correctness-only landing contract this function is
    **not** a landing or replay gate. It exists so the row-level
    ``meets_speed_gate`` diagnostic and Markdown column continue to
    report whether a kernel beat the PyTorch reference on every
    workload. Empty / missing speedup fields are treated as
    ``False`` for the diagnostic.
    """
    if not rows:
        return False
    for r in rows:
        s = r.get("speedup_factor")
        if not isinstance(s, (int, float)) or s <= 1.0:
            return False
    return True


def _bundle_manifest_is_v2_native(bundle: Path) -> bool:
    """Verify the bundle's ``manifest.json`` is schema v2 native.

    Required: the file must exist, parse as JSON, lint-pass
    (:func:`lint_manifest`), AND carry every required v2 field
    (``native_env_hash`` non-empty, ``native_env_components``
    present) AND no Docker-era forbidden fields (``docker_image_digest``,
    ``profiling_image_digest``, ``dispatch_mode``, ``dispatch_kind``).

    Any bundle whose manifest is stale or carries Docker-era
    identity material is hard-invalidated: this check returns
    ``False``, the bundle is treated as if the trace were missing,
    and ``_gather_baseline_row`` schedules a fresh native run.
    """
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(manifest, dict):
        return False
    try:
        lint_manifest(manifest)
    except ManifestLintError:
        return False
    if not manifest.get("native_env_hash"):
        return False
    if not isinstance(manifest.get("native_env_components"), dict):
        return False
    forbidden = (
        "docker_image_digest",
        "profiling_image_digest",
        "dispatch_mode",
        "dispatch_kind",
    )
    if any(k in manifest for k in forbidden):
        return False
    return True


def _find_trace_candidate(
    baseline_dir: Path,
    trace_dir: Optional[Path],
) -> Optional[tuple]:
    """Locate the bundle-dir trace for this baseline.

    Replay requires the bundle layout
    ``.harness_traces/<slug>__<lang>/{trace.json,manifest.json}`` AND
    the manifest must be schema v2 native (lint passes,
    ``native_env_hash`` + ``native_env_components`` present, no
    Docker-era forbidden fields). A bundle that fails either gate
    is treated as if the trace were missing and
    ``_gather_baseline_row`` schedules a fresh native run.

    When ``trace_dir`` is supplied (manual bring-up workflow), the
    helper also accepts ``round-*-trace*.json`` files there — that
    path is for operator-driven manual bring-up and does NOT replay
    through the cache index; ``_gather_baseline_row`` only consults
    the cache for the bundle path.

    Returns ``(trace_path, parsed_payload)`` or ``None``.
    """
    bundle = _active_bundle_dir(baseline_dir) / "trace.json"
    if bundle.is_file() and _bundle_manifest_is_v2_native(bundle.parent):
        try:
            payload = json.loads(bundle.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = None
        rows = payload.get("rows") if isinstance(payload, dict) else None
        if isinstance(payload, dict) and isinstance(rows, list) and all(
            isinstance(row, dict) for row in rows
        ):
            return (bundle, payload)
    if trace_dir is not None:
        slug = baseline_dir.parent.name
        for c in sorted(trace_dir.glob("round-*-trace*.json"), reverse=True):
            try:
                payload = json.loads(c.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            rows = payload.get("rows") if isinstance(payload, dict) else None
            if (
                isinstance(payload, dict)
                and isinstance(rows, list)
                and all(isinstance(row, dict) for row in rows)
                and slug in str(payload.get("baseline", ""))
            ):
                return (c, payload)
    return None


def _environment_libs_from_trace(
    payload: Dict[str, Any],
    rows: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Extract the upstream ``evaluation.environment.libs`` from a trace.

    Order of preference: the trace payload's top-level
    ``environment_libs`` (when written by a current-format run); then
    the first row's ``environment_libs``; otherwise ``{}``. Empty
    libs are an honest fallback that still produces a valid (if
    over-strict) cache key.
    """
    libs = payload.get("environment_libs")
    if isinstance(libs, dict):
        return {str(k): str(v) for k, v in libs.items()}
    for row in rows:
        row_libs = row.get("environment_libs")
        if isinstance(row_libs, dict):
            return {str(k): str(v) for k, v in row_libs.items()}
    return {}


def _environment_libs_from_rows(
    rows: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Same as ``_environment_libs_from_trace`` but row-only."""
    for row in rows:
        row_libs = row.get("environment_libs")
        if isinstance(row_libs, dict):
            return {str(k): str(v) for k, v in row_libs.items()}
    return {}


def _persist_fresh_trace(
    baseline_dir: Path,
    rows: List[Dict[str, Any]],
    env_libs: Dict[str, str],
) -> Path:
    """Persist a freshly produced trace to the bundle's ``trace.json``.

    Trace destination is the bundle directory
    ``tools/sol_baselines_harness/.harness_traces/<slug>__<lang>/trace.json``.
    The bundle dir is created on-demand so the counter-free surfaces
    orchestrator can drop ``manifest.json`` and per-surface artifact
    subdirectories alongside it.

    **Trace persistence is independent of cache trust.** A native run
    that produced honest non-PASSED rows (e.g. ``INVALID_REFERENCE``
    when the host can't run the PyTorch reference) MUST still land
    its bundle trace so downstream consumers (the aggregate report,
    the test suite, the operator's debugging session) can read the
    rows. Cache stamping is a separate decision — see
    :func:`_stamp_cache_record` for the cache-trust gate.
    """
    bundle = _active_bundle_dir(baseline_dir)
    bundle.mkdir(parents=True, exist_ok=True)
    trace_path = bundle / "trace.json"
    payload = {
        "baseline": str(baseline_dir),
        "rows": rows,
        "environment_libs": env_libs,
    }
    trace_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return trace_path


def _recompute_preserved_evidence_label(
    *, existing: Dict[str, Any], rows: List[Dict[str, Any]]
) -> Optional[str]:
    """Recompute the evidence label for a manifest preserved by the
    ``_persist_fresh_manifest`` identity guard.

    Mirrors :func:`baselines_harness.surfaces.manifest.ManifestBuilder.evidence_label`
    so the preserve path stays in sync with what the orchestrator
    would emit for the same host kind + surface state + trace
    pass/fail state. Returns ``None`` when the manifest is missing
    the host.kind needed for arch tag derivation; the caller leaves
    the stored label untouched in that degenerate case.

    The four-branch ladder:
      * ``Compile-checked-<arch>`` when not all-PASSED.
      * ``Ran-local-<arch>`` when any surface is ``failed``.
      * ``Profiled-<host>-complete`` when every surface is ``ok``.
      * ``Profiled-<host>-<arch>`` otherwise (no failed, not all-ok).

    The arch tag and host segment derive from the existing
    manifest's ``host.kind`` (the same identity axis the preserve
    guard already verified is current).
    """
    from baselines_harness.arch_resolver import arch_tag_from_host_kind
    from baselines_harness.surfaces.base import (
        SURFACE_STATUS_FAILED,
        SURFACE_STATUS_OK,
    )
    host_kind = (existing.get("host") or {}).get("kind") or ""
    if not host_kind:
        return None
    arch_tag = arch_tag_from_host_kind(host_kind)
    statuses = [
        surface.get("status")
        for surface in (existing.get("surfaces") or {}).values()
        if isinstance(surface, dict)
    ]
    all_passed = bool(rows) and all(
        r.get("status") == "PASSED" for r in rows
    )
    if not all_passed:
        return f"Compile-checked-{arch_tag}"
    if any(s == SURFACE_STATUS_FAILED for s in statuses):
        return f"Ran-local-{arch_tag}"
    host_segment = "B200" if host_kind.startswith("b200") else "local"
    if statuses and all(s == SURFACE_STATUS_OK for s in statuses):
        return f"Profiled-{host_segment}-complete"
    return f"Profiled-{host_segment}-{arch_tag}"


def _solution_file_sha256(path: Path) -> str:
    """SHA-256 hex digest of ``solution.json``'s file bytes.

    Matches the orchestrator's
    :func:`baselines_harness.surfaces.dispatch_plan._sha256_of_file`
    so the manifest-preserve guard can compare the freshly-derived
    solution identity against the value the surfaces orchestrator
    stamped in each surface's ``cache_identity.solution_source_hash``
    slot. The two helpers MUST stay in sync; if the orchestrator
    ever changes its identity scheme, this function must follow.
    """
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_host_fingerprint(host_info: Any) -> str:
    """SHA-256 hex digest of the orchestrator's host-fingerprint
    field set: ``(kind, compute_capability, driver_version,
    cuda_runtime_version, nvtx_header_present, nvtx_library_present)``
    joined with ``"|"``.

    Mirrors :func:`baselines_harness.surfaces.orchestrator.SurfaceOrchestrator.compute_surface_cache_identity`'s
    ``host_fingerprint`` derivation EXACTLY so the manifest-preserve
    guard can compare the freshly-computed value against the
    stored value in each surface's
    ``cache_identity.host_fingerprint`` slot. The two helpers MUST
    stay in sync; if the orchestrator ever changes its host-
    fingerprint recipe, this function must follow.

    Accepts ``host_info`` as a ``HostInfo`` dataclass instance
    (the production path) — the field set is the same shape as
    the manifest's top-level ``host`` dict + ``nvtx_*_present``
    booleans, hashed identically.
    """
    host_text = "|".join((
        host_info.kind,
        host_info.compute_capability,
        host_info.driver_version,
        host_info.cuda_runtime_version,
        str(host_info.nvtx_header_present),
        str(host_info.nvtx_library_present),
    ))
    return hashlib.sha256(host_text.encode("utf-8")).hexdigest()


def _current_environment_libs() -> Dict[str, str]:
    """Return the live ``environment.libs`` dict the eval driver
    would record at run time.

    Mirrors ``sol_execbench.core.utils.env_snapshot`` shape: at
    minimum ``torch`` version, plus ``triton`` and ``cuda`` when
    those imports succeed. Used by the manifest-anchored replay
    branch to verify the candidate trace's stored ``env_libs``
    haven't drifted between the orchestrator's stamp time and
    report.py's run time (Codex review P2a). Triton in
    particular is NOT covered by ``native_env_components`` /
    ``native_env_hash``, so a triton upgrade would otherwise slip
    past the manifest identity check.

    On import failure for ``triton`` or ``torch.version.cuda``,
    the corresponding key is omitted (matching the eval driver's
    fallback so the comparison stays consistent).
    """
    libs: Dict[str, str] = {}
    try:
        import torch
        libs["torch"] = torch.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import triton as _tr
        libs["triton"] = getattr(_tr, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch.version as _tv
        cuda_v = getattr(_tv, "cuda", None)
        if cuda_v:
            libs["cuda"] = cuda_v
    except Exception:  # noqa: BLE001
        pass
    return libs


def _trace_env_libs_match_current(trace_env_libs: Dict[str, str]) -> bool:
    """True iff the trace's stored ``env_libs`` matches the live
    eval-driver env EXACTLY — same keys and same values.

    The cache key's ``environment_libs`` is a sorted tuple of
    ``(name, version)`` pairs (see :class:`cache.CacheKey`); strict
    equality on the tuple means any addition, removal, or value
    change rejects. The an earlier iteration first cut was too lenient — it
    only iterated trace keys and skipped when the live side was
    missing the key (i.e., ``triton`` removed from the eval env
    would still pass). Codex review caught that gap.

    The fix: require dict equality. Both sides must have the same
    keys AND the same values per key. If triton was installed
    between orchestrator stamp and report.py run, the live dict
    gains a ``triton`` key the trace lacks → mismatch → decline.
    If triton was removed, the trace has a ``triton`` key the
    live dict lacks → mismatch → decline.
    """
    live_libs = _current_environment_libs()
    return dict(live_libs) == dict(trace_env_libs)


def _trace_is_clean_full(cand_payload: Dict[str, Any]) -> bool:
    """True iff the candidate trace is a clean, complete orchestrator
    run (or a legacy trace that lacks the gate fields entirely).

    The orchestrator's :func:`surfaces.orchestrator._emit_trace_json`
    writes two integrity fields into the trace JSON:

      * ``gate_reasons``: list of strings. Non-empty means the run
        was rejected by the orchestrator's eval-gate (rc != 0, zero
        rows parsed, row-count mismatch, parse errors, at least one
        non-PASSED row, etc.).
      * ``expected_workload_count``: int. The orchestrator's
        ``len(plan.workload_subset)`` at dispatch time (0 when the
        plan has no explicit subset).

    The manifest-anchored replay path's existing
    :func:`is_passed_trace` check only verifies parsed rows are
    PASSED — it doesn't catch the case where an orchestrator
    full-fanout run timed out after emitting only a subset of
    PASSED rows. The orchestrator would write ``gate_reasons =
    ["row count mismatch: expected 7, parsed 3"]`` in that case;
    this helper rejects on the gate_reasons signal (Codex
    an earlier iteration P2a).

    Legacy traces (e.g., report.py-stamped traces that don't carry
    these fields, or pre-orchestrator-trace-emit traces like the
    canonical pilot's committed trace) lack both fields. They
    return True here — the parsed-rows-PASSED check upstream is
    the only signal available, and that already covers the
    common cases.
    """
    if "gate_reasons" in cand_payload:
        gate_reasons = cand_payload["gate_reasons"]
        if not isinstance(gate_reasons, list) or gate_reasons:
            return False
    if "expected_workload_count" not in cand_payload:
        return True
    expected_count = cand_payload["expected_workload_count"]
    if type(expected_count) is not int or expected_count < 0:
        return False
    rows = cand_payload.get("rows") or []
    if not isinstance(rows, list):
        return False
    if expected_count > 0 and len(rows) != expected_count:
        return False
    if expected_count == 0 and rows:
        return False
    return True


def _trace_has_structural_failure(cand_payload: Dict[str, Any]) -> bool:
    """Return True when gate reasons indicate incomplete evaluator output.

    A complete trace with honest non-PASSED rows remains useful fallback
    evidence. Parse errors, evaluator failures, zero rows, and row-count
    mismatches must never be promoted after a failed fresh run.
    """
    if "gate_reasons" not in cand_payload:
        reasons = []
    else:
        reasons = cand_payload["gate_reasons"]
    if not isinstance(reasons, list):
        return True
    for reason in reasons:
        if not isinstance(reason, str):
            return True
        if (
            reason.startswith("evaluator exited")
            or reason.startswith("row count mismatch")
            or reason == "zero rows parsed"
            or "unparseable JSONL" in reason
        ):
            return True
    if "expected_workload_count" in cand_payload:
        expected_count = cand_payload["expected_workload_count"]
        if type(expected_count) is not int or expected_count < 0:
            return True
        if expected_count > 0:
            rows = cand_payload.get("rows")
            if not isinstance(rows, list) or len(rows) != expected_count:
                return True
        elif cand_payload.get("rows"):
            return True
    return False


def _manifest_identity_matches_current(
    *,
    baseline_dir: Path,
) -> bool:
    """True iff the bundle's v2 ``manifest.json`` stamps an identity
    that matches current code / env / host / evaluator state.

    Used by :func:`_gather_baseline_row`'s manifest-anchored replay
    branch — a clean-checkout report.py run has no
    ``.harness_cache.json`` entries, so the cache-replay path
    declines for every bundle. Without an additional trust anchor,
    every committed orchestrator-stamped manifest would go through
    the preserve guard (which strictly rejects non-default-config
    dispatches) and get stubbed. This helper recovers the
    canonical-pilot workflow: the manifest's stored identity is its
    own integrity anchor when cache_index lacks a record.

    The identity check covers two top-level axes and seven
    per-surface axes report.py can independently re-derive,
    mirroring the per-surface preserve guard's verifiable axes:

      * ``native_env_hash`` (top-level) — compare against
        :func:`compute_native_env_hash` over current
        :func:`native_env_components`.
      * ``evaluator_tree_hash`` (top-level) — compare against
        :func:`sol_execbench_source_hash`.
      * ``solution_source_hash`` (per-surface) — re-hash the
        on-disk ``solution.json`` bytes via
        :func:`_solution_file_sha256`. The cache layer's
        :data:`CacheKey.solution_source_hash` uses a DIFFERENT
        scheme (canonical-JSON-of-solution-dict hash); the
        manifest comparison MUST use the file scheme that the
        orchestrator stamps via ``dispatch_plan._sha256_of_file``.
      * ``host_fingerprint`` (per-surface) — compare against
        :func:`_compute_host_fingerprint` over
        :func:`_cached_host_info`.
      * ``surface_schema_version`` (per-surface) — compare
        against the live plugin's class attribute via
        :func:`default_surface_plugins`.
      * ``surface_impl_version`` (per-surface) — compare
        against the live plugin's class attribute.
      * ``surface_tool_version`` (per-surface) — compare
        against the live ``plugin.probe(host_info).tool_version``.
      * ``workload_subset_signature`` (per-surface) — derive
        from the manifest top-level ``workload_subset`` using
        the orchestrator's exact rule
        (``surfaces/dispatch_plan.py:208-211``). Non-empty
        subset → exact signature; empty subset (full-fanout) →
        accept any ``all-*`` form.
      * ``env_filter_hash`` (per-surface) — for
        ``nvbit_metrics``, compare against the hash of the
        current ``METRIC_EXTRACTOR_*`` env subset via
        :func:`compute_env_filter_hash`; for every other
        surface, must be empty string (matching the
        orchestrator's stamping rule).

    Per-surface discipline:
    EVERY non-skipped surface must carry matching axes. The
    an earlier iteration first-match-wins shape would have replayed a
    multi-surface manifest with one current + one stale surface.

    The 10th per-surface axis (``command_hash``) is intentionally
    NOT checked — the orchestrator's command_hash depends on the
    per-surface dispatch argv produced by
    ``plugin.dispatch_argv(plan)``, which embeds bundle-local
    file paths. Reconstructing it requires instantiating a
    ``DispatchPlan`` plus each plugin's ``dispatch_argv`` hook
    (the remaining piece of an earlier architectural follow-up).
    Trusting the stored value is safe here: the other 7
    per-surface axes already cover the inputs ``command_hash``
    is derived from.

    The ``config_overrides_hash`` axis the preserve guard checks
    is also intentionally NOT in the replay check. The preserve
    guard compares against ``_hash_config_overrides({})``
    (report.py's actual empty-config dispatch), correctly
    rejecting any orchestrator manifest stamped with reduced
    Stage-3 config. The canonical pilot was stamped with reduced
    config (warmup_runs=1, iterations=1, benchmark_reference=
    False); applying the preserve-guard config check here would
    destroy its committed evidence on every clean-checkout run.
    Replay's purpose is to honor the orchestrator's dispatch
    verbatim, accepting that the orchestrator's reduced-config
    timing data is what the bundle reports. The full reconciliation
    of orchestrator vs report.py dispatch is a queued architectural
    follow-up.

    Returns False (decline replay) when:
      * Manifest file missing or unparseable / non-dict.
      * Any top-level axis mismatches.
      * Manifest's surfaces dict has no non-skipped entry (a stub
        manifest from a previous report.py run — stubs don't
        anchor a replay).
      * Any non-skipped surface's per-surface axes is missing,
        non-string, empty, or mismatched on any of the 7 axes
        above.
      * Solution.json missing on disk (degraded bundle state).
    """
    bundle = _active_bundle_dir(baseline_dir)
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(manifest, dict):
        return False
    # Native env: re-derive and compare.
    #
    # ``native_env_components()`` does NOT raise on a degraded host
    # (one or more axes missing); it returns ``"unknown"`` sentinels
    # and ``compute_native_env_hash()`` hashes them deterministically.
    # The cache layer separately rejects degraded fingerprints via
    # ``missing_components`` (see ``cache.derive``). Without an
    # explicit degraded-env rejection here, a manifest stamped on
    # the same degraded host would replay with untrusted evidence
    # (Codex review P2c). Bail before computing the hash.
    components = native_env_components()
    if missing_components(components):
        return False
    try:
        current_native_env_hash = compute_native_env_hash(components)
    except DegradedNativeEnvError:
        return False
    if manifest.get("native_env_hash") != current_native_env_hash:
        return False
    # Evaluator tree.
    if manifest.get("evaluator_tree_hash") != sol_execbench_source_hash():
        return False
    # Workload selection policy: only ``full-fanout`` manifests can
    # anchor a replay. A partial-workload run (the surfaces CLI
    # default ``lowest-index-default``, or
    # ``--representative-workload``) covers fewer workloads than
    # report.py's expected full run; replaying it would publish
    # ``all_passed=true`` based on a subset of workloads and feed
    # category-floor membership with incomplete evidence (Codex
    # an earlier iteration P2).
    if manifest.get("workload_selection_policy") != "full-fanout":
        return False
    # Build mode: only ``judge`` manifests can anchor a replay.
    # A ``--build-mode profiling`` surface run produces a
    # different binary (NVTX-instrumented, reduced eval config)
    # whose rows + surface artifacts are not equivalent to a
    # judge-mode fresh run; replaying it would publish
    # profiling-build evidence as landed judge evidence (Codex
    # an earlier iteration P2b).
    if manifest.get("build_mode") != "judge":
        return False
    # Solution source: re-hash the on-disk solution.json bytes using
    # the SAME scheme the orchestrator stamps into surface
    # ``cache_identity.solution_source_hash`` slots.
    try:
        fresh_solution_file_hash = _solution_file_sha256(
            baseline_dir / "solution.json"
        )
    except OSError:
        return False
    # Host fingerprint: live probe (re-uses LRU cache so this costs
    # at most one host probe per report.py invocation across the
    # entire baseline walk).
    host_info_for_probe = _cached_host_info()
    current_host_fp = _compute_host_fingerprint(host_info_for_probe)
    # Workload subset signature: derive from the manifest's own
    # top-level workload_subset using the orchestrator's exact
    # rule. Non-empty subset → ",".join(sorted indices). Empty
    # subset → accept any "all-*" shape (the orchestrator's
    # signature embeds workload_count which we can't reproduce
    # without parsing workload.jsonl).
    stored_workload_subset = manifest.get("workload_subset") or []
    if isinstance(stored_workload_subset, list) and stored_workload_subset:
        expected_signature: Optional[str] = ",".join(
            str(i) for i in sorted(stored_workload_subset)
        )
    else:
        expected_signature = None
    # Plugin lookup (for surface_schema_version /
    # surface_impl_version / surface_tool_version verification).
    from baselines_harness.surfaces import default_surface_plugins
    from baselines_harness.surfaces.nvbit_metrics import (
        NVBIT_METRICS_USER_ENV_VARS,
        compute_env_filter_hash as _compute_metrics_env_filter_hash,
    )
    try:
        plugins = default_surface_plugins()
    except Exception:  # noqa: BLE001
        plugins = {}
    current_metrics_env_filter_hash = _compute_metrics_env_filter_hash(
        {
            k: os.environ[k]
            for k in NVBIT_METRICS_USER_ENV_VARS
            if k in os.environ
        }
    )
    # Per-surface verification: EVERY non-skipped surface must
    # independently match the 7 axes above. No first-match-wins
    # early-break.
    surfaces = manifest.get("surfaces") or {}
    if not isinstance(surfaces, dict):
        return False
    has_real_surface = False
    for surface_name, entry in surfaces.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("status") == "skipped":
            continue
        has_real_surface = True
        identity = entry.get("cache_identity") or {}
        if not isinstance(identity, dict):
            return False
        stored_solution = identity.get("solution_source_hash")
        if (
            not isinstance(stored_solution, str)
            or not stored_solution
            or stored_solution != fresh_solution_file_hash
        ):
            return False
        stored_host_fp = identity.get("host_fingerprint")
        if (
            not isinstance(stored_host_fp, str)
            or not stored_host_fp
            or stored_host_fp != current_host_fp
        ):
            return False
        # Plugin schema / impl: must match the live plugin's
        # class attributes. If the plugin can't be resolved
        # (e.g., schema changed name) the manifest is stale.
        plugin = plugins.get(surface_name)
        if plugin is None:
            return False
        stored_schema = identity.get("surface_schema_version")
        stored_impl = identity.get("surface_impl_version")
        if (
            stored_schema != plugin.schema_version
            or stored_impl != plugin.impl_version
        ):
            return False
        # Surface tool version: live probe. Returns "" on probe
        # failure, matching the orchestrator's fallback at
        # surfaces/orchestrator.py:893-895.
        #
        # SKIP for ``nvbit_metrics``: its probe at
        # surfaces/nvbit_metrics.py:332 returns
        # ``f"metric_extractor.so@{mtime}"`` where mtime comes from
        # the on-disk ``.so`` file. The ``.so`` is gitignored (must
        # be rebuilt locally via ``make ARCH=sm_120`` per CLAUDE.md),
        # so the mtime is set at build time and differs between
        # orchestrator stamp time and any subsequent clone /
        # rebuild — even when the underlying tool is functionally
        # unchanged. an earlier iteration strict live-probe check rejected
        # canonical pilot's nvbit_metrics surface on every clean
        # checkout, defeating the manifest-anchored replay path
        # for that one surface (Codex review). The
        # ``nvbit_metrics`` surface's identity is otherwise covered
        # by ``surface_schema_version`` + ``surface_impl_version``
        # + ``env_filter_hash`` + the committed artifact files
        # themselves; skipping tool_version here loses no
        # meaningful identity coverage.
        if surface_name != "nvbit_metrics":
            try:
                live_tool_version = (
                    plugin.probe(host_info_for_probe).tool_version or ""
                )
            except Exception:  # noqa: BLE001
                live_tool_version = ""
            stored_tool_version = identity.get("surface_tool_version")
            if (
                not isinstance(stored_tool_version, str)
                or stored_tool_version != live_tool_version
            ):
                return False
        # Workload subset signature.
        stored_signature = identity.get("workload_subset_signature")
        if expected_signature is not None:
            if (
                not isinstance(stored_signature, str)
                or stored_signature != expected_signature
            ):
                return False
        else:
            if (
                not isinstance(stored_signature, str)
                or not stored_signature.startswith("all-")
            ):
                return False
        # env_filter_hash: nvbit_metrics → live env hash; every
        # other surface → empty string (orchestrator stamping rule).
        stored_env_filter_hash = identity.get("env_filter_hash")
        if not isinstance(stored_env_filter_hash, str):
            return False
        if surface_name == "nvbit_metrics":
            if stored_env_filter_hash != current_metrics_env_filter_hash:
                return False
        else:
            if stored_env_filter_hash != "":
                return False
    if not has_real_surface:
        # Stub manifests don't anchor a replay. The caller should
        # fall through to fresh-run.
        return False
    return True


@functools.lru_cache(maxsize=1)
def _cached_host_info():
    """Probe the host once per ``report.py`` invocation.

    ``probe_host`` shells out to ``nvcc --version`` and
    ``nvidia-smi --query-gpu=...``; caching the result keeps the
    per-baseline manifest writer from re-probing the same host
    fingerprint dozens of times in a single report run.

    ``captured_at_utc`` is intentionally captured once at the first
    call — every fresh-trace manifest emitted in a single report run
    shares the same start-of-run timestamp, which is the honest
    reading. The cache survives for the lifetime of the process.
    """
    return probe_host()


_FRESH_MANIFEST_SURFACE_WAIVER = (
    "report.py fresh-trace path did not invoke any analytic surface; "
    "manifest exists so _find_trace_candidate can replay the "
    "cache-keyed trace on the next report run."
)


def _persist_fresh_manifest(
    baseline_dir: Path,
    rows: List[Dict[str, Any]],
    eval_timeout_s: float,
) -> Path:
    """Persist a minimal v2-native ``manifest.json`` for a fresh trace.

    Bundles emitted by :func:`_persist_fresh_trace` need a matching
    schema-v2 ``manifest.json`` so that
    :func:`_find_trace_candidate`'s acceptance gate
    (:func:`_bundle_manifest_is_v2_native`) succeeds on the next
    report run. Without this manifest, every fresh trace is treated
    as if the trace were missing and the evaluator re-runs from
    scratch — defeating the cache and failing closed on hosts that
    cannot re-run the evaluator.

    **Preserve-existing-evidence guard:** if a manifest is already
    on disk for the bundle AND passes :func:`lint_manifest`, this
    function returns its path **without rewriting**. The surfaces
    orchestrator's real surface evidence (e.g. cupti / nvbit_metrics
    / sanitizer / static in ``status=ok`` with artifact paths) is
    durable evidence the report pipeline must not erase. Only when
    no manifest exists (or the existing one fails lint, indicating
    a corrupted bundle) does this writer create a stub.

    The stub manifest captures:

    * Host fingerprint from :func:`_cached_host_info` (a cached
      ``probe_host`` call).
    * ``evaluator_tree_hash`` from :func:`sol_execbench_source_hash`
      — the same git tree hash the cache key uses.
    * ``native_env_hash`` + ``native_env_components`` from the
      single source of truth (:func:`native_env_components` +
      :func:`compute_native_env_hash`).
    * ``surface_set=[]`` — report.py runs the eval driver without
      any analytic surface, matching the default
      ``surface_set=()`` in :func:`current_baseline_key`.
    * Every surface entry marked ``status=skipped`` with a
      :data:`_FRESH_MANIFEST_SURFACE_WAIVER` reason so lint passes.
    * ``_all_passed_trace`` flagged on the builder before
      serialisation so the embedded ``evidence_label`` agrees with
      :func:`_evidence_label_for_bundle`.

    A degraded native-env host still produces a deterministic hash
    over its components (the ``"unknown"`` sentinels), so the
    manifest lands. Replay through this bundle remains gated on
    :func:`cache.derive`, which raises
    :class:`DegradedNativeEnvError` for any UNKNOWN component, so a
    degraded bundle cannot serve a replay.
    """
    # Resolve the active-host probe ONCE per writer invocation. The
    # same object is reused for: the active bundle-dir routing
    # (sm_86 host -> __sm86 sibling), the preserve-path
    # host_fingerprint computation below, the per-surface
    # plugin.probe(host_info) calls, and the stub ManifestBuilder.
    # The "at most one probe per invocation" budget is pinned by
    # ``test_persist_fresh_manifest_preserve_path_probes_host_at_most_once``.
    host_info_for_probe = _cached_host_info()
    host_kind = getattr(host_info_for_probe, "kind", None)
    bundle = _bundle_dir(baseline_dir, host_kind=host_kind)
    bundle.mkdir(parents=True, exist_ok=True)
    manifest_path = bundle / "manifest.json"
    # Compute the fresh identity once. The preserve guard's
    # identity-match check uses the same components / hashes that
    # the stub-write path below would otherwise record, so any
    # discrepancy between an existing manifest and the freshly-
    # observed environment / evaluator tree / solution source is
    # caught right here.
    components = native_env_components()
    fresh_native_env_hash = compute_native_env_hash(components)
    fresh_evaluator_tree_hash = sol_execbench_source_hash()
    # File-byte SHA-256 of solution.json — matches what the surfaces
    # orchestrator records in each surface's ``cache_identity.
    # solution_source_hash`` slot (see
    # ``surfaces/dispatch_plan.py:_sha256_of_file``). The orchestrator
    # uses raw file bytes, NOT the dict-canonical-order hash in
    # ``cache.solution_source_hash``; comparing apples-to-apples here
    # is the only way the preserve guard can verify that the existing
    # surface artifacts were produced from the SAME solution file the
    # fresh trace is being written for.
    try:
        fresh_solution_source_hash: Optional[str] = _solution_file_sha256(
            baseline_dir / "solution.json"
        )
    except OSError:
        fresh_solution_source_hash = None
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                raise ManifestLintError(
                    "manifest top-level must be a dict; non-object "
                    "JSON (e.g. an array) is treated as corrupt and "
                    "regenerated as a stub."
                )
            lint_manifest(existing)
        except (json.JSONDecodeError, ManifestLintError):
            pass
        else:
            # Preserve only manifests that carry REAL surface evidence
            # (at least one surface in a status other than ``skipped``)
            # AND whose stored identity matches the freshly-computed
            # identity. The identity check prevents stale surface
            # evidence from a previous build (different solution /
            # evaluator / native env) being combined with the fresh
            # trace.json the surrounding ``_persist_fresh_trace`` just
            # wrote — that combination could promote a changed
            # solution to ``Profiled-*`` using surface artifacts that
            # belong to the previous build.
            existing_surfaces = existing.get("surfaces", {})
            # ``host_info_for_probe`` was resolved once at the top of
            # this writer invocation; reusing it here satisfies the
            # an earlier iteration "at most one probe per writer invocation"
            # budget the regression test pins.
            # Host identity: the orchestrator stamps
            # ``host_fingerprint = sha256("|".join(kind,
            # compute_capability, driver_version,
            # cuda_runtime_version, nvtx_header_present,
            # nvtx_library_present))`` into each surface's
            # ``cache_identity``. ``native_env_hash`` covers the
            # native software fingerprint (nvcc / torch / cuda
            # runtime / cudnn / sol_execbench package / ...) but
            # NOT the GPU-side identity, so a fresh report run on
            # a different GPU with the same software stack would
            # match every other axis. report.py probes the live
            # host via ``_cached_host_info`` and computes the
            # fingerprint with the same recipe; on mismatch the
            # preserved manifest is regenerated as a stub so a
            # cross-host run cannot promote stale surface evidence.
            fresh_host_fingerprint = _compute_host_fingerprint(
                host_info_for_probe
            )
            # Per-surface identity verification: the orchestrator's
            # ``surface_cache_hit`` (surfaces/orchestrator.py:935)
            # invalidates on 12 cache_identity axes. 2 are top-level
            # (``native_env_hash``, ``evaluator_tree_hash``); 9 of
            # the remaining 10 per-surface axes are independently
            # verifiable from report.py's standpoint and are checked
            # below:
            #
            #   * ``solution_source_hash`` — live: re-hash
            #     ``solution.json`` via :func:`_solution_file_sha256`.
            #   * ``host_fingerprint`` — live: re-hash the host_info
            #     via :func:`_compute_host_fingerprint`.
            #   * ``surface_schema_version`` / ``surface_impl_version``
            #     — live: read the plugin's class attributes via
            #     ``default_surface_plugins()``.
            #   * ``config_overrides_hash`` — internal-consistency:
            #     hash the manifest top-level ``config_overrides`` via
            #     ``_hash_config_overrides`` (same function the
            #     orchestrator uses) and compare. Catches dispatch
            #     config drift (Codex review example).
            #   * ``workload_subset_signature`` — internal-consistency:
            #     derive the signature from the manifest top-level
            #     ``workload_subset`` using the same rule as
            #     ``surfaces/dispatch_plan.py:208-211``. For non-empty
            #     subset the signature is fully derivable; for empty
            #     subset (full-fanout) the signature is ``all-N`` where
            #     ``N`` is the workload count — when ``N`` cannot be
            #     reproduced we accept any ``all-*`` shape (the
            #     workload_subset==[] branch is trusted, since report.py
            #     also implies full-fanout). Catches workload subset
            #     divergence (Codex review example).
            #   * ``surface_tool_version`` — live: invoke
            #     ``plugin.probe(host_info).tool_version``. Catches
            #     tool drift (e.g., nvcc / cuobjdump upgrade between
            #     orchestrator stamp time and report.py run time —
            #     Codex review example).
            #   * ``env_filter_hash`` (nvbit_metrics only) — live:
            #     hash the current ``METRIC_EXTRACTOR_*`` env subset
            #     via :func:`compute_env_filter_hash`. Catches
            #     env-override drift.
            #
            # The 10th per-surface axis (``command_hash``) is NOT
            # independently verifiable from report.py: the orchestrator
            # hashes the full per-surface dispatch argv produced by
            # ``_surface_dispatch_argv``, which includes bundle-local
            # file paths and plugin-specific wrapper construction (e.g.,
            # cupti's env-gated wrapper). Reconstructing this requires
            # instantiating a ``DispatchPlan`` + each plugin's
            # ``dispatch_argv`` hook — that's the architectural refactor
            # an earlier follow-up flagged as out of round scope.
            # Trusting the stored value is safe here because the four
            # axes above already cover the inputs ``command_hash`` is
            # derived from (config_overrides + workload_subset +
            # plugin schema/impl + host).
            #
            # All checked axes are verified PER SURFACE (not first-
            # match-wins): a multi-surface manifest where surface A
            # has current identity and surface B has stale identity
            # must regenerate.
            #
            # History: an earlier iteration rejected any non-empty unverifiable
            # axis (too strict — broke clean checkouts). an earlier iteration
            # dropped that check (too loose — preserved stale
            # evidence). an earlier iteration verifies the 9 axes report.py can
            # independently re-derive and trusts the 1 axis that
            # requires DispatchPlan reconstruction. The canonical
            # pilot survives on the same host; cross-host, tool-drift,
            # config-drift, workload-divergence all get rejected.
            from baselines_harness.surfaces import (
                default_surface_plugins,
            )
            from baselines_harness.surfaces.orchestrator import (
                _hash_config_overrides,
            )
            from baselines_harness.surfaces.nvbit_metrics import (
                NVBIT_METRICS_USER_ENV_VARS,
                compute_env_filter_hash as _compute_metrics_env_filter_hash,
            )
            try:
                plugins = default_surface_plugins()
            except Exception:  # noqa: BLE001
                plugins = {}
            # Pre-compute the values that don't vary per surface.
            #
            # ``config_overrides_hash`` is compared against report.py's
            # ACTUAL expected config — which is empty (report.py's
            # fresh-run path invokes ``run_baseline`` without any
            # ``--config`` flag, so the dispatch uses default eval
            # config). an earlier iteration "compare against the manifest's own
            # top-level config_overrides" check was semantically trivial
            # (stored == itself); Codex review flagged it.
            # The canonical pilot's manifest stamps a reduced-config
            # hash (warmup_runs=1, iterations=1, benchmark_reference=
            # False from Stage-3 profiling); it correctly fails this
            # check. Canonical pilot is preserved via the manifest-
            # anchored replay path in ``_gather_baseline_row``, not
            # via this preserve guard.
            fresh_config_overrides_hash = _hash_config_overrides({})
            stored_workload_subset = existing.get("workload_subset") or []
            if isinstance(stored_workload_subset, list) and stored_workload_subset:
                fresh_workload_subset_signature: Optional[str] = ",".join(
                    str(i) for i in sorted(stored_workload_subset)
                )
            else:
                # Empty workload_subset == full-fanout. The
                # orchestrator's signature is ``all-N`` where N is the
                # workload count. We can't reproduce N from the
                # manifest alone, so accept any ``all-*`` form. Report.py
                # also implies full-fanout, so the dispatch is
                # equivalent.
                fresh_workload_subset_signature = None
            fresh_metrics_env_filter_hash = _compute_metrics_env_filter_hash(
                {
                    k: os.environ[k]
                    for k in NVBIT_METRICS_USER_ENV_VARS
                    if k in os.environ
                }
            )
            real_surface_present = False
            per_surface_identity_matches = True
            for surface_name, surface_entry in existing_surfaces.items():
                if not isinstance(surface_entry, dict):
                    continue
                if (
                    surface_entry.get("status") == SURFACE_STATUS_SKIPPED
                ):
                    continue
                real_surface_present = True
                identity = surface_entry.get("cache_identity") or {}
                if not isinstance(identity, dict):
                    per_surface_identity_matches = False
                    break
                # Solution-source identity, per surface. A non-skipped
                # surface MUST carry a non-empty
                # ``solution_source_hash`` that equals the freshly-
                # computed value; missing / mismatched on any surface
                # regenerates.
                surface_solution_hash = identity.get(
                    "solution_source_hash"
                )
                if (
                    fresh_solution_source_hash is None
                    or not isinstance(surface_solution_hash, str)
                    or not surface_solution_hash
                    or surface_solution_hash
                    != fresh_solution_source_hash
                ):
                    per_surface_identity_matches = False
                    break
                # Host fingerprint, per surface. Same shape: a non-
                # skipped surface MUST carry a non-empty
                # ``host_fingerprint`` matching the freshly-computed
                # value; cross-host or partial-stale rejects.
                surface_host_fingerprint = identity.get(
                    "host_fingerprint"
                )
                if (
                    not isinstance(surface_host_fingerprint, str)
                    or not surface_host_fingerprint
                    or surface_host_fingerprint
                    != fresh_host_fingerprint
                ):
                    per_surface_identity_matches = False
                    break
                plugin = plugins.get(surface_name)
                if plugin is None:
                    per_surface_identity_matches = False
                    break
                stored_schema = identity.get("surface_schema_version")
                stored_impl = identity.get("surface_impl_version")
                if (
                    stored_schema != plugin.schema_version
                    or stored_impl != plugin.impl_version
                ):
                    per_surface_identity_matches = False
                    break
                # config_overrides_hash internal-consistency.
                stored_config_overrides_hash = identity.get(
                    "config_overrides_hash"
                )
                if (
                    not isinstance(stored_config_overrides_hash, str)
                    or stored_config_overrides_hash
                    != fresh_config_overrides_hash
                ):
                    per_surface_identity_matches = False
                    break
                # workload_subset_signature internal-consistency.
                stored_signature = identity.get(
                    "workload_subset_signature"
                )
                if fresh_workload_subset_signature is not None:
                    if (
                        not isinstance(stored_signature, str)
                        or stored_signature
                        != fresh_workload_subset_signature
                    ):
                        per_surface_identity_matches = False
                        break
                else:
                    # Empty-subset case: accept any ``all-*`` shape.
                    if (
                        not isinstance(stored_signature, str)
                        or not stored_signature.startswith("all-")
                    ):
                        per_surface_identity_matches = False
                        break
                # surface_tool_version live re-probe.
                #
                # SKIP for ``nvbit_metrics``: its probe value embeds
                # the metric extractor .so file's mtime (see
                # surfaces/nvbit_metrics.py:332). The .so is
                # gitignored / locally built, so the mtime is set at
                # build time and differs between orchestrator stamp
                # and any subsequent clone or rebuild. an earlier iteration
                # strict live-probe check rejected canonical pilot's
                # nvbit surface on every clean checkout (Codex
                # an earlier iteration P1). The skip mirrors the replay path's
                # same-axis skip in ``_manifest_identity_matches_current``.
                stored_tool_version = identity.get(
                    "surface_tool_version"
                )
                if not isinstance(stored_tool_version, str):
                    per_surface_identity_matches = False
                    break
                if surface_name != "nvbit_metrics":
                    try:
                        live_tool_version = (
                            plugin.probe(host_info_for_probe).tool_version
                            or ""
                        )
                    except Exception:  # noqa: BLE001
                        live_tool_version = ""
                    if live_tool_version != stored_tool_version:
                        per_surface_identity_matches = False
                        break
                # env_filter_hash: for nvbit_metrics, hash current env
                # subset and compare. For every other surface the
                # orchestrator records the empty string (it never
                # reads METRIC_EXTRACTOR_* vars).
                stored_env_filter_hash = identity.get("env_filter_hash")
                if not isinstance(stored_env_filter_hash, str):
                    per_surface_identity_matches = False
                    break
                if surface_name == "nvbit_metrics":
                    if (
                        stored_env_filter_hash
                        != fresh_metrics_env_filter_hash
                    ):
                        per_surface_identity_matches = False
                        break
                else:
                    if stored_env_filter_hash != "":
                        per_surface_identity_matches = False
                        break
            # Dispatch identity: the surfaces orchestrator's
            # ``surface_cache_hit`` treats ``build_mode`` and the
            # workload selection as causal axes. report.py's
            # fresh-run uses ``judge`` mode with all workloads
            # (the orchestrator labels this
            # ``workload_selection_policy == "full-fanout"``).
            # A stored manifest produced via ``--build-mode
            # profiling`` or ``--representative-workload N`` has
            # surface artifacts that are NOT representative of
            # report.py's eval. Preserving them would let
            # ``_evidence_label_for_bundle`` promote fresh
            # judge/all-workload rows to ``Profiled-*`` using
            # mismatched-dispatch surface evidence.
            dispatch_matches = (
                existing.get("build_mode") == "judge"
                and existing.get("workload_selection_policy") == "full-fanout"
            )
            identity_matches = (
                existing.get("native_env_hash") == fresh_native_env_hash
                and existing.get("evaluator_tree_hash") == fresh_evaluator_tree_hash
                and dispatch_matches
                and per_surface_identity_matches
            )
            if real_surface_present and identity_matches:
                # Identity-matched manifest is preserved, BUT its
                # stored ``evidence_label`` reflects the OLD trace's
                # pass/fail state. ``_persist_fresh_trace`` has
                # already overwritten ``trace.json`` with the new
                # rows, so the manifest's label must be recomputed
                # against the fresh ``rows`` + the preserved surface
                # statuses; otherwise a previously-Profiled bundle
                # that now produces RUNTIME_ERROR rows would still
                # claim ``Profiled-*`` while the report.md row
                # correctly says ``Compile-checked-*``. The
                # consistency test would surface that drift as a
                # hard failure.
                recomputed = _recompute_preserved_evidence_label(
                    existing=existing,
                    rows=rows,
                )
                if recomputed is not None and recomputed != existing.get(
                    "evidence_label"
                ):
                    existing["evidence_label"] = recomputed
                    manifest_path.write_text(
                        json.dumps(existing, indent=2, sort_keys=True)
                        + "\n",
                        encoding="utf-8",
                    )
                return manifest_path
    meta = _baseline_metadata(baseline_dir)
    # Reuse the host probe resolved at the top of this writer
    # invocation (single-probe-per-writer contract).
    host_info = host_info_for_probe
    builder = ManifestBuilder(
        slug=meta["slug"],
        language=meta["language"],
        host_info=host_info,
        build_mode="judge",
        surface_set=set(),
    )
    builder.evaluator_tree_hash = fresh_evaluator_tree_hash
    builder.native_env_hash = fresh_native_env_hash
    builder.native_env_components = dict(components)
    builder.workload_subset = []
    builder.workload_selection_policy = "full-fanout"
    builder.config_overrides = {}
    builder.eval_timeout_s = float(eval_timeout_s)
    builder.reference_timing_enabled = True
    builder.dispatch_command = {"argv": [], "env_overrides": {}}
    for name in VALID_SURFACES:
        builder.surfaces[name] = SurfaceManifestEntry(
            status=SURFACE_STATUS_SKIPPED,
            waiver_reason=_FRESH_MANIFEST_SURFACE_WAIVER,
        )
    builder._all_passed_trace = is_passed_trace(rows)
    builder.lint_and_write(manifest_path)
    return manifest_path


def _stamp_cache_record(
    baseline_dir: Path,
    trace_path: Path,
    env_libs: Dict[str, str],
    cache_index: Dict[str, Dict[str, Any]],
) -> None:
    """Stamp the cache index for a cache-trustworthy fresh trace.

    Caller MUST gate this on ``outcome.is_clean() AND
    is_passed_trace(rows)`` — only fully-PASSED runs become replay
    evidence. Non-PASSED traces (``INVALID_REFERENCE``, etc.) still
    land in the bundle via :func:`_persist_fresh_trace` but do NOT
    enter the cache index, so the next report run will re-execute
    them rather than replay a host-gap result.
    """
    key = current_baseline_key(baseline_dir, environment_libs=env_libs)
    cache_index[str(trace_path)] = cache_record_for_key(key)


_REPORT_EXCLUSION_LEDGER_PATH = (
    _THIS_DIR
    / "tests"
    / "fixtures"
    / "evidence_consistency_exclusions.json"
)


def _load_report_exclusion_bundle_keys() -> set:
    """Read the in-tree exclusion ledger and return the set of
    ``slug__lang`` bundle keys the report walker must skip.

    Mirrors the ledger consumed by
    ``test_evidence_label_consistency.py`` so the report scope and
    the consistency-test scope cannot drift apart. Returns an empty
    set when the ledger is missing or malformed (the report falls
    back to walking every baseline).
    """
    if not _REPORT_EXCLUSION_LEDGER_PATH.is_file():
        return set()
    try:
        payload = json.loads(
            _REPORT_EXCLUSION_LEDGER_PATH.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError:
        return set()
    entries = payload.get("entries", {})
    if not isinstance(entries, dict):
        return set()
    return {str(k) for k in entries.keys()}


def _bundle_key_for_baseline_dir(baseline_dir: Path) -> str:
    """Derive ``<slug>__<lang>`` from a baseline_dir path
    ``baselines/<cat>/<slug>/<lang>/``."""
    return f"{baseline_dir.parent.name}__{baseline_dir.name}"


def build_report(
    output_root: Path,
    benchmark_root: Path,
    trace_dir: Optional[Path],
    deferred_path: Path,
    cache_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble the aggregate report payload (lint-gated)."""
    excluded_bundle_keys = _load_report_exclusion_bundle_keys()
    # Lint MUST run over every baseline regardless of exclusion
    # ledger membership — an excluded bundle's solution.json /
    # sources still live under baselines/ and could carry
    # forbidden compile options or surface / profiling terms.
    # Filtering before ``scan_landed_baselines`` would create a
    # lint blind spot exactly where the lint is most needed (a
    # bundle gets excluded BECAUSE something is wrong with it).
    # Only filter at row-generation time below.
    all_baseline_dirs = list(_iter_all_baselines(output_root))
    lint_hits = scan_landed_baselines(all_baseline_dirs)
    lint_hits.extend(scan_harness_production_code())
    lint_hits.extend(scan_harness_production_text_for_container_terms())
    lint_hits.extend(scan_harness_production_text_for_workflow_history_terms())
    ledger_errors = validate_deferred_ledger(deferred_path)

    baseline_dirs = [
        b for b in all_baseline_dirs
        if _bundle_key_for_baseline_dir(b) not in excluded_bundle_keys
    ]
    rows = [
        _gather_baseline_row(b, benchmark_root, trace_dir, cache_index)
        for b in baseline_dirs
    ]

    # Category floor membership requires correctness only (every row
    # PASSED). Speedup vs the PyTorch reference is diagnostic and
    # does NOT gate category-floor membership under the correctness-
    # only landing contract.
    landed_categories = sorted(
        {r["category"] for r in rows if r["all_passed"]}
    )
    floor_categories = sorted(SUBSET_MAP.keys())

    deferred = []
    if deferred_path.is_file():
        try:
            deferred = json.loads(
                deferred_path.read_text(encoding="utf-8")
            ).get("entries", [])
        except json.JSONDecodeError:
            deferred = []

    return {
        "landed_pilots": rows,
        "deferred_pilots": [
            {
                "slug": e.get("slug"),
                "language": e.get("language"),
                "status": e.get("status"),
                "evidence": e.get("evidence"),
                "harness_commit": e.get("harness_commit"),
            }
            for e in deferred
        ],
        "category_floor": {
            "required": floor_categories,
            "landed": landed_categories,
            "complete": sorted(landed_categories) == floor_categories,
        },
        "lint": {
            "forbidden_surface_hits": [
                {"label": label, "rule": rule, "detail": detail}
                for label, rule, detail in lint_hits
            ],
            "deferred_ledger_errors": ledger_errors,
            "clean": not lint_hits and not ledger_errors,
        },
        "local_proxy_disclaimer": local_proxy_disclaimer(_top_level_arch_tag()),
        "local_arch_yaml_present": local_arch_config_available(),
        "local_arch_yaml_path": _top_level_arch_yaml_path(),
        "local_arch_tag": _top_level_arch_tag(),
    }


def _top_level_arch_tag() -> str:
    """Resolve the live host's arch tag once for the report header."""
    try:
        host_info = probe_host()
    except Exception:
        return "sm_unknown"
    return arch_tag_from_host_kind(getattr(host_info, "kind", "unknown"))


def _top_level_arch_yaml_path() -> str:
    """Repo-relative path to the resolved arch yaml; ``"unknown"`` when
    the resolver returns ``None``."""
    yaml_path = resolved_arch_yaml()
    if yaml_path is None:
        return "unknown"
    try:
        return str(yaml_path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(yaml_path)


def _render_markdown(report: Dict[str, Any]) -> str:
    """Render the report payload as a human-readable Markdown table."""
    lines: List[str] = []
    lines.append("# Baselines Harness Aggregate Report")
    lines.append("")
    arch_tag = report.get("local_arch_tag", "sm_unknown")
    arch_yaml_path = report.get("local_arch_yaml_path", "unknown")
    score_column_header = f"local_sol_score ({arch_tag})"
    lines.append(
        f"> Local arch yaml resolved to `{arch_yaml_path}` "
        f"(`arch_tag={arch_tag}`); present on disk: "
        f"**{report['local_arch_yaml_present']}**. The canonical "
        f"`local_sol_score` column is the per-workload SOLAR Stage 4 "
        f"score for slugs covered by a builder in "
        f"`tools/sol_baselines_harness/solar_analysis.py:_REGISTRY` AND "
        f"whose precision token has a matching `MAC_per_cycle_*` key "
        f"in the resolved arch yaml; slugs that fail either check "
        f"honestly emit `{UNKNOWN_T_SOL}`. The "
        f"`upper_bound_score_t_sol_zero` column below is a separate "
        f"`t_sol = 0.0` diagnostic baseline (NOT a SOL Score). The "
        f"field name is historical: the SOL Score formula is "
        f"monotonic-increasing in `t_sol` for speedup rows, so this "
        f"value is a **lower** bound on the canonical column, not an "
        f"upper bound. The JSON key is retained for backward "
        f"compatibility."
    )
    lines.append("")
    lines.append("## Local-vs-leaderboard disclaimer")
    lines.append("")
    lines.append(f"> {report['local_proxy_disclaimer']}")
    lines.append("")

    lines.append("## Landed pilots")
    lines.append("")
    lines.append(
        "| Category | Slug | Language | Workloads | All PASSED | "
        "Speedup > 1.0 (diagnostic) | Slowest speedup | "
        f"{score_column_header} | upper_bound_score_t_sol_zero | "
        "Evidence | Trace |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for r in report["landed_pilots"]:
        trace = r.get("trace") or "—"
        ub = r.get("upper_bound_score_t_sol_zero")
        ub_str = (
            f"{ub:.3f}" if isinstance(ub, (int, float))
            else f"`{ub}`"
        )
        lines.append(
            f"| {r['category']} | `{r['slug']}` | `{r['language']}` | "
            f"{r['workload_count']} | {r['all_passed']} | "
            f"{r['meets_speed_gate']} | "
            f"{r['slowest_speedup']:.3f} | "
            f"`{r['local_sol_score']}` | "
            f"{ub_str} | "
            f"`{r['evidence_label']}` | `{trace}` |"
        )
    lines.append("")

    floor = report["category_floor"]
    lines.append("## Category floor (DEC-1)")
    lines.append("")
    lines.append(
        f"- Required categories: {floor['required']}"
    )
    lines.append(
        f"- Landed (PASSED on all workloads): {floor['landed']}"
    )
    lines.append(
        f"- 4-category floor complete: **{floor['complete']}**"
    )
    lines.append("")

    lines.append("## Deferred pilots")
    lines.append("")
    if not report["deferred_pilots"]:
        lines.append("(none)")
    else:
        lines.append(
            "| Slug | Language | Status | Harness commit | Evidence |"
        )
        lines.append("|---|---|---|---|---|")
        for e in report["deferred_pilots"]:
            lines.append(
                f"| `{e['slug']}` | `{e['language']}` | "
                f"`{e['status']}` | `{e['harness_commit']}` | "
                f"`{e['evidence']}` |"
            )
    lines.append("")

    lint = report["lint"]
    lines.append("## Lint pre-flight")
    lines.append("")
    lines.append(
        f"- Forbidden-surface hits: {len(lint['forbidden_surface_hits'])}"
    )
    for h in lint["forbidden_surface_hits"]:
        lines.append(f"  - `{h['label']}` ({h['rule']}): {h['detail']}")
    lines.append(
        f"- Deferred-ledger errors: {len(lint['deferred_ledger_errors'])}"
    )
    for err in lint["deferred_ledger_errors"]:
        lines.append(f"  - {err}")
    lines.append(f"- Clean: **{lint['clean']}**")
    lines.append("")

    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/sol_baselines_harness/report.py",
        description=(
            "Walk every landed baseline; emit a JSON + Markdown "
            "aggregate report with slowest-workload speedups, "
            "deferred-ledger snapshot, and a lint pre-flight."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root of landed baselines (default: ./baselines).",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT,
        help="Root of the downloaded benchmark dataset.",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory holding round-N-*-trace.json files. "
            "When supplied, the report reads cached traces for each "
            "baseline whose slug matches; otherwise each baseline is "
            "re-run via the harness."
        ),
    )
    parser.add_argument(
        "--deferred-ledger",
        type=Path,
        default=_THIS_DIR / "deferred.json",
        help="Path to the deferred.json ledger.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=_THIS_DIR / "report.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=_THIS_DIR / "report.md",
        help="Markdown output path.",
    )
    parser.add_argument(
        "--fail-on-lint",
        action="store_true",
        default=True,
        help=(
            "Exit non-zero when lint detects any forbidden-surface or "
            "deferred-ledger error (default: enabled)."
        ),
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=_THIS_DIR / ".harness_cache.json",
        help=(
            "Per-trace cache index used to gate replay. Stores a "
            "CacheKey serialisation for each trace path that the "
            "report has previously trusted; replay only fires when "
            "the current key for the baseline matches the stored "
            "key exactly AND the trace's rows are all PASSED."
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cache_index = load_cache_index(args.cache_file)
    report = build_report(
        output_root=args.output_root,
        benchmark_root=args.benchmark_root,
        trace_dir=args.trace_dir,
        deferred_path=args.deferred_ledger,
        cache_index=cache_index,
    )
    # Persist cache index regardless of lint outcome — the index
    # captures legitimate fresh-run evidence that should survive a
    # lint failure (lint can fail on unrelated ledger or surface
    # issues without invalidating the trace itself).
    save_cache_index(args.cache_file, cache_index)
    if args.fail_on_lint and not report["lint"]["clean"]:
        sys.stderr.write(
            "lint gate failed; refusing to write report. Details:\n"
        )
        for hit in report["lint"]["forbidden_surface_hits"]:
            sys.stderr.write(
                f"  - {hit['label']} ({hit['rule']}): {hit['detail']}\n"
            )
        for err in report["lint"]["deferred_ledger_errors"]:
            sys.stderr.write(f"  - {err}\n")
        return 2
    # Evidence-integrity gate: every landed baseline must have at
    # least one workload row, either replayed from a key-matched
    # trace or stamped from a fresh harness run. A zero-row row is
    # almost always a silent harness failure (cache miss + failed
    # fresh run) and would overwrite valid prior evidence with a
    # `Compile-checked-<arch>` placeholder. Fail closed before any
    # output is written.
    zero_row_pilots = [
        r["slug"] for r in report["landed_pilots"]
        if r.get("workload_count", 0) == 0
    ]
    if zero_row_pilots:
        sys.stderr.write(
            "evidence-integrity gate failed; refusing to write "
            "report. The following landed baselines produced zero "
            "workload rows (no cache replay AND no clean fresh "
            "run):\n"
        )
        for slug in zero_row_pilots:
            sys.stderr.write(f"  - {slug}\n")
        sys.stderr.write(
            "Re-run the harness for each affected baseline (e.g. "
            "python tools/sol_baselines_harness/run.py --baseline "
            "baselines/<cat>/<slug>/<lang>) before regenerating "
            "the aggregate report.\n"
        )
        return 2
    # Bundle-trace integrity gate: every landed pilot must have a
    # non-null `trace` field that resolves to the **canonical** bundle
    # path ``.harness_traces/<slug>__<lang>/trace.json``. Mere
    # ``is_file()`` is too permissive — a stale absolute path from a
    # prior dev box, the legacy flat ``<slug>__<lang>.json`` layout,
    # or a ``--trace-dir`` operator path (which is a bring-up surface,
    # not landed-pilot evidence) could all silently satisfy a loose
    # check. The strict invariant: ``Path(reported).resolve() ==
    # (_DEFAULT_TRACES_DIR / f"{slug}__{language}" / "trace.json").resolve()``.
    # If a future rename of the bundle layout occurs, the gate and the
    # ``test_landed_bundles_v2`` regression test must update together
    # — they reference the same canonical-path formula on purpose.
    missing_trace: list[tuple[str, Any, str]] = []
    for r in report["landed_pilots"]:
        slug = r["slug"]
        language = r["language"]
        trace_value = r.get("trace")
        expected = (
            _DEFAULT_TRACES_DIR / f"{slug}__{language}" / "trace.json"
        ).resolve()
        if not trace_value:
            missing_trace.append((slug, trace_value, "null"))
            continue
        p = Path(trace_value)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if not p.is_file():
            missing_trace.append((slug, trace_value, "does-not-resolve"))
            continue
        if p.resolve() != expected:
            missing_trace.append(
                (slug, trace_value, f"not-canonical (expected {expected})")
            )
    if missing_trace:
        sys.stderr.write(
            "bundle-trace integrity gate failed; refusing to write "
            "report. The following landed baselines have a `trace` "
            "field that does not resolve to the canonical bundle "
            "path:\n"
        )
        for slug, trace_value, why in missing_trace:
            sys.stderr.write(
                f"  - {slug}: trace={trace_value!r} ({why})\n"
            )
        sys.stderr.write(
            "Each landed pilot must persist a bundle "
            "`<.harness_traces>/<slug>__<lang>/trace.json` (even when "
            "the rows are INVALID_REFERENCE / non-PASSED — those still "
            "count as honest evidence). Legacy flat traces, absolute "
            "paths outside REPO_ROOT, and operator `--trace-dir` paths "
            "do NOT satisfy this gate.\n"
        )
        return 2
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(_render_markdown(report), encoding="utf-8")
    print(
        f"wrote {args.output_json} ({len(report['landed_pilots'])} landed, "
        f"{len(report['deferred_pilots'])} deferred)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
