"""SOL Score utilities for landed baselines.

Wraps the upstream ``sol_execbench.sol_score`` standalone helper so
the harness can produce a ``local_sol_score`` cell when the live
host's compute capability resolves to a SOLAR arch yaml, and
``"unknown_t_sol"`` otherwise. The local-vs-leaderboard distinction
is intentional: this column is NOT comparable to the public SOL
Score (which uses B200@1.5GHz with the paper §4.4 methodology + the
benchmark-internal scoring baseline ``T_b``). The harness aggregate
report carries the disclaimer alongside every local score it emits.

The arch yaml is resolved dynamically via
``baselines_harness.arch_resolver.resolve_local_arch_yaml`` against
the live ``HostInfo`` (probed from ``nvidia-smi``). Hosts whose CC
is not in the resolver's mapping (or that are MIG-partitioned)
short-circuit to ``"unknown_t_sol"`` rather than fabricating a
roofline against a non-matching arch.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent
REPO_ROOT = _SCRIPTS_DIR.parent

UNKNOWN_T_SOL = "unknown_t_sol"


def local_proxy_disclaimer(arch_tag: str = "sm_unknown") -> str:
    """Arch-parametric disclaimer string for the local proxy score.

    ``arch_tag`` is the manifest arch label (``sm_86``, ``sm_120``,
    etc.). When ``sm_unknown`` is passed, the disclaimer still
    renders but with that literal — callers are responsible for
    feeding the resolved arch tag from
    ``arch_resolver.arch_tag_from_host_kind``.
    """
    return (
        f"local proxy score; computed via sol_score(t_k, t_b := "
        f"local PyTorch reference, t_sol := SOLAR roofline on the "
        f"resolved arch yaml for {arch_tag}). NOT comparable to the "
        f"leaderboard SOL Score (which uses B200@1.5GHz with paper "
        f"section 4.4 methodology + the benchmark-internal scoring "
        f"baseline T_b not present in our data)."
    )


# Legacy export retained for callers that read the disclaimer at
# import time. The arch-parametric form
# ``local_proxy_disclaimer(arch_tag)`` is preferred.
LOCAL_PROXY_DISCLAIMER = local_proxy_disclaimer("sm_unknown")


def _import_upstream_sol_score():
    """Import the standalone ``sol_score`` helper."""
    try:
        sys.path.insert(0, str(REPO_ROOT / "tools" / "sol_execbench" / "src"))
        from sol_execbench.sol_score import sol_score  # noqa: E402
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot import upstream sol_score helper: {exc}."
        ) from exc
    return sol_score


def resolved_arch_yaml() -> Optional[Path]:
    """Return the SOLAR arch yaml resolved against the live host.

    Probes the host via ``surfaces.host_probe.probe_host`` and asks
    ``arch_resolver.resolve_local_arch_yaml`` for the matching yaml.
    Returns ``None`` when the host CC is not in the resolver mapping,
    the device is MIG-partitioned, or the host probe fails. The
    caller short-circuits to ``unknown_t_sol`` in that case.
    """
    try:
        from baselines_harness.arch_resolver import resolve_local_arch_yaml
        from baselines_harness.surfaces.host_probe import probe_host
    except Exception:
        return None
    try:
        host_info = probe_host()
    except Exception:
        return None
    return resolve_local_arch_yaml(host_info)


def local_arch_config_available() -> bool:
    """True when the live host's arch yaml resolves to a real file.

    The function used to check a static ``5060.yaml`` path; it now
    consults the resolver so that the wrong-arch path returns False
    instead of falsely advertising a 5060 yaml on a 3090 Ti host.
    """
    return resolved_arch_yaml() is not None


def compute_score(
    t_k: float,
    t_b: float,
    t_sol: Optional[float],
) -> object:
    """Return the SOL Score, or the literal ``"unknown_t_sol"``.

    ``t_sol`` is the SOLAR-derived roofline lower bound for this
    workload on the local arch; when no local SOLAR config exists yet
    we propagate the ``"unknown_t_sol"`` sentinel string instead of
    fabricating a number.

    ``t_k`` is the kernel-under-test latency (ms) and ``t_b`` is the
    baseline (reference) latency (ms). The formula follows the upstream
    ``sol_score(t_k, t_b, t_sol)`` definition.

    The arch yaml's presence is the caller's concern: a numeric
    ``t_sol`` from ``solar_analysis.predict_t_sol_ms`` already implies
    the resolved arch yaml was usable. The function fails closed only
    when ``t_sol`` itself is the ``unknown_t_sol`` sentinel (or
    ``None``); otherwise it returns the formula value directly.
    """
    if t_sol is None:
        return UNKNOWN_T_SOL
    sol_score = _import_upstream_sol_score()
    return float(sol_score(t_k, t_b, t_sol))


def slowest_speedup(rows: list) -> float:
    """Min ``speedup_factor`` across a list of report rows.

    Used by the aggregate report's per-baseline "slowest workload"
    column. Rows without a numeric ``speedup_factor`` are skipped; if
    every row lacks one (e.g. the baseline failed correctness on every
    workload), the function returns ``0.0``.
    """
    values = [
        r.get("speedup_factor")
        for r in rows
        if isinstance(r.get("speedup_factor"), (int, float))
    ]
    if not values:
        return 0.0
    return float(min(values))


def local_score_for_workloads(
    rows: list,
    *,
    slug: Optional[str] = None,
    benchmark_root: Optional[object] = None,
    subset: Optional[str] = None,
    benchmark_name: Optional[str] = None,
    t_sol_provider: Optional[object] = None,
) -> object:
    """Return the local SOL Score across a baseline's workloads.

    When the SOLAR Stage 4 derivation succeeds for every row in
    ``rows``, returns the median of ``compute_score(t_k, t_b, t_sol)``.
    Otherwise returns the ``unknown_t_sol`` sentinel — no fabricated
    numbers ever ship under this label.

    Per-workload ``t_sol_ms`` derivation:

      - The caller passes the baseline slug, the benchmark subset
        (e.g. ``"FlashInfer-Bench"``), the benchmark name (the
        ``solution.json.definition`` field), and the benchmark root
        path.
      - For each row's ``workload_index``, ``solar_analysis.t_sol_for_baseline_workload``
        looks up the slug's analysis builder, reads the matching
        ``workload.jsonl`` axes, materialises a temporary
        ``analysis.yaml``, and runs
        ``EinsumGraphPerfModel.predict`` against the live host's
        resolved arch yaml (from ``arch_resolver.resolve_local_arch_yaml``).
        The resulting ``fused_prefetched.runtime_ms`` is used as
        ``t_sol_ms``.
      - If the slug has no registered builder, or any row's analysis
        cannot be built, the function returns ``unknown_t_sol``.

    Tests can pass a ``t_sol_provider(workload_index) -> Optional[float]``
    callable that bypasses the SOLAR import — used to monkey-patch
    the estimator without spinning up the real Stage 4 path.
    """
    if not rows:
        return UNKNOWN_T_SOL

    if t_sol_provider is None:
        # The default provider path delegates to
        # ``solar_analysis.t_sol_for_baseline_workload``, which in
        # turn consults the arch resolver. A missing / unresolvable
        # arch yaml short-circuits to ``unknown_t_sol`` inside the
        # provider; this function does not need its own arch check.
        if not local_arch_config_available():
            return UNKNOWN_T_SOL
        if (
            slug is None
            or benchmark_root is None
            or subset is None
            or benchmark_name is None
        ):
            return UNKNOWN_T_SOL
        from baselines_harness.solar_analysis import (
            t_sol_for_baseline_workload,
        )

        def _default_provider(workload_index: int) -> Optional[float]:
            return t_sol_for_baseline_workload(
                slug=slug,
                benchmark_root=benchmark_root,  # type: ignore[arg-type]
                subset=subset,  # type: ignore[arg-type]
                benchmark_name=benchmark_name,  # type: ignore[arg-type]
                workload_index=workload_index,
            )

        provider = _default_provider
    else:
        provider = t_sol_provider  # type: ignore[assignment]

    scores: list = []
    for row in rows:
        t_k = row.get("latency_ms")
        t_b = row.get("reference_latency_ms")
        if not isinstance(t_k, (int, float)) or not isinstance(t_b, (int, float)):
            return UNKNOWN_T_SOL
        if t_k <= 0 or t_b <= 0:
            return UNKNOWN_T_SOL
        workload_index = row.get("workload_index")
        if not isinstance(workload_index, int):
            return UNKNOWN_T_SOL
        t_sol_ms = provider(workload_index)
        if not isinstance(t_sol_ms, (int, float)):
            return UNKNOWN_T_SOL
        s = compute_score(float(t_k), float(t_b), float(t_sol_ms))
        if not isinstance(s, (int, float)):
            return UNKNOWN_T_SOL
        scores.append(float(s))

    if not scores:
        return UNKNOWN_T_SOL
    scores.sort()
    n = len(scores)
    if n % 2:
        return scores[n // 2]
    return (scores[n // 2 - 1] + scores[n // 2]) / 2.0


def upper_bound_score_t_sol_zero(rows: list) -> object:
    """Diagnostic ``t_sol = 0.0`` baseline score.

    Returns the median of ``compute_score(t_k, t_b, 0.0)`` across rows
    when the local-arch YAML is present, otherwise the
    ``unknown_t_sol`` sentinel. The ``t_sol = 0.0`` substitution
    collapses the upstream ``sol_score`` formula
    ``S = (T_b - T_SOL) / ((T_k - T_SOL) + (T_b - T_SOL))`` to
    ``Tb / (Tk + Tb)``, which for ``Tk < Tb`` (the speedup case) is
    strictly > 0.5 and trends toward 1.0 as speedup grows.

    **Historical name caveat**: the field name
    ``upper_bound_score_t_sol_zero`` predates the discovery that the
    SOL Score formula is monotonic-INCREASING in ``T_SOL`` for
    speedup rows. The partial derivative is
    ``dS/dT_SOL = (T_b - T_k) / v^2`` where
    ``v = (T_k - T_SOL) + (T_b - T_SOL)``; for ``T_k < T_b`` this is
    positive, so a real SOLAR-derived ``t_sol > 0`` produces a
    **larger** score than the ``t_sol = 0`` substitution. Therefore
    this value is a **lower** diagnostic baseline for the canonical
    column, NOT an upper bound. The name is retained as the JSON key
    for backward compatibility (cache index records, downstream
    consumers). The canonical local SOL Score is the registry-backed
    ``local_sol_score`` column.
    """
    if not local_arch_config_available():
        return UNKNOWN_T_SOL
    scores: list = []
    for row in rows:
        t_k = row.get("latency_ms")
        t_b = row.get("reference_latency_ms")
        if not isinstance(t_k, (int, float)):
            continue
        if not isinstance(t_b, (int, float)):
            continue
        if t_k <= 0 or t_b <= 0:
            continue
        s = compute_score(float(t_k), float(t_b), 0.0)
        if isinstance(s, (int, float)):
            scores.append(float(s))
    if not scores:
        return UNKNOWN_T_SOL
    scores.sort()
    n = len(scores)
    if n % 2:
        return scores[n // 2]
    return (scores[n // 2 - 1] + scores[n // 2]) / 2.0
