#!/usr/bin/env python3
"""Subsystem-coverage + metadata-consistency verifier for SimBench-BW.

Each check surface corresponds to a specific invariant from the plan's
positive/negative-test acceptance criteria. Invoke with one or more
--check-<name> flags, or --strict to run every check.

Exit codes:
    0  every requested check passed.
    1  at least one check failed (a machine-readable reason token is printed).
    2  usage error.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any, Callable


SUITE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SUITE_ROOT.parent.parent


def _load_toml(path: Path) -> Any:
    with open(path, "rb") as f:
        return tomllib.load(f)


# --- check implementations ----------------------------------------------------


BENCHMARK_ROW_COUNT = 88

# Canonical first-wave slug list from the plan's first-wave ordering.
# The rmsnorm file-pair counts as one first-wave entry; fwd is the anchor.
FIRST_WAVE_SLUGS = frozenset({
    "flash_attn_fwd_bf16_causal",
    "flash_attn_fwd_fp8_causal",
    "flash_attn_bwd_bf16",
    "pagedkv_decode_gqa_bf16",
    "paged_mla_decode_bf16",
    "ragged_attention_prefill_bf16",
    "moe_router_topk_softmax",
    "moe_grouped_gemm_bf16",
    "moe_grouped_gemm_fp8",
    "nvfp4_blockscale_gemm",
    "swiglu_ffn_fused_bf16",
    "rmsnorm_fwd_bf16",
    "qkv_rope_standard_bf16",
    "mamba_selective_scan_bf16",
    "vae_conv_gn_silu_residual_bf16",
    "hyena_fft_conv_block_fp32",
    "sampling_topk_temperature_bf16",
    "sparse_2to4_bf16_gemm",
    "tma_tensor_copy_2d",
    "cluster_remote_reduction",
})


def check_roster() -> list[str]:
    path = SUITE_ROOT / "roster.toml"
    if not path.exists():
        return ["roster_missing: roster.toml not found"]
    data = _load_toml(path)
    rows = data.get("rows", [])
    benchmark_rows = [r for r in rows if r.get("tier") != "control"]
    if len(benchmark_rows) != BENCHMARK_ROW_COUNT:
        return [f"roster_desync: expected {BENCHMARK_ROW_COUNT} benchmark (non-control) rows, got {len(benchmark_rows)}"]
    errors: list[str] = []
    required = {
        "slug", "level", "domain", "family", "primary_precision",
        "primary_subsystems", "secondary_subsystems", "arch_gate", "tier",
        "trace_budget_mib", "walltime_target_ms", "sm_meaningful", "deterministic",
        "reference_family", "reference_type", "seed_policy", "citation",
        "probe_dependencies", "status", "wave", "split",
    }
    slugs: set[str] = set()
    for i, r in enumerate(rows):
        missing = required - set(r.keys())
        if missing:
            errors.append(f"row_schema_violation: row[{i}] slug={r.get('slug', '<?>')} missing {sorted(missing)}")
        slug = r.get("slug")
        if slug in slugs:
            errors.append(f"row_schema_violation: duplicate slug {slug!r}")
        slugs.add(slug)
    return errors


def check_tier_consistency() -> list[str]:
    roster = _load_toml(SUITE_ROOT / "roster.toml")
    pm = _load_toml(SUITE_ROOT / "platform_matrix.toml")
    pm_map = {r["slug"]: r for r in pm["rows"]}
    errors: list[str] = []
    for r in roster["rows"]:
        slug = r["slug"]
        if slug not in pm_map:
            errors.append(f"tier_vs_platform_mismatch: {slug} absent from platform_matrix.toml")
            continue
        platforms = pm_map[slug]["platforms"]
        tier = r["tier"]
        if tier in ("P", "P-strict") and platforms != "both":
            errors.append(f"tier_vs_platform_mismatch: tier={tier} slug={slug} expected platforms=both got {platforms!r}")
        if tier == "H" and platforms not in ("h100", "either"):
            errors.append(f"tier_vs_platform_mismatch: tier=H slug={slug} got platforms={platforms!r}")
        if tier == "B" and platforms not in ("rtx_pro_6000_blackwell", "either"):
            errors.append(f"tier_vs_platform_mismatch: tier=B slug={slug} got platforms={platforms!r}")
        if tier == "control" and platforms != "either":
            errors.append(f"tier_vs_platform_mismatch: tier=control slug={slug} expected platforms=either (vendor control) got {platforms!r}")
    return errors


def check_subsystems() -> list[str]:
    """Every S1-S20 must have >= 3 primary stressors among non-control rows.
    Control rows are permitted to retain their own primary_subsystems metadata
    (for provenance / future comparison); they are silently excluded from the
    counts, not treated as an error."""
    roster = _load_toml(SUITE_ROOT / "roster.toml")
    primary = Counter()
    for r in roster["rows"]:
        if r["tier"] == "control":
            continue
        for s in r["primary_subsystems"]:
            primary[s] += 1
    errors: list[str] = []
    for k in range(1, 21):
        name = f"S{k}"
        count = primary.get(name, 0)
        if count < 3:
            errors.append(f"subsystem_underfilled: {name} has {count} primary stressor(s) (minimum 3)")
    return errors


def check_wave() -> list[str]:
    """First-wave view must be exactly the canonical 20 slugs; union across all
    waves must equal the non-control benchmark-row slug set."""
    roster = _load_toml(SUITE_ROOT / "roster.toml")
    errors: list[str] = []
    benchmark_rows = [r for r in roster["rows"] if r.get("tier") != "control"]
    wave_1_slugs = {r["slug"] for r in benchmark_rows if r.get("wave") == "1"}
    # Exact match against canonical first-wave
    missing = FIRST_WAVE_SLUGS - wave_1_slugs
    extra = wave_1_slugs - FIRST_WAVE_SLUGS
    if missing:
        errors.append(f"wave_first_wave_missing: wave=1 is missing canonical slugs {sorted(missing)}")
    if extra:
        errors.append(f"wave_first_wave_extra: wave=1 contains non-canonical slugs {sorted(extra)}")
    if len(wave_1_slugs) != 20 and not missing and not extra:
        # should be unreachable given the set comparison above, but guards against future drift
        errors.append(f"wave_first_wave_count: wave=1 has {len(wave_1_slugs)} rows (expected 20)")
    # Ensure union covers every non-control benchmark row exactly once
    all_wave_slugs = [r["slug"] for r in benchmark_rows]
    if len(set(all_wave_slugs)) != len(benchmark_rows):
        errors.append("wave_union_duplicate: some benchmark rows appear twice in the wave mapping")
    if len(benchmark_rows) != BENCHMARK_ROW_COUNT:
        errors.append(f"wave_union_mismatch: benchmark-row count {len(benchmark_rows)} != {BENCHMARK_ROW_COUNT}")
    return errors


def check_identity_in_manifest(results_root: Path | None) -> list[str]:
    """Every measurement_manifest.json under results_root must carry the identity fields."""
    errors: list[str] = []
    if results_root is None or not results_root.exists():
        return errors
    required = {
        "device_name", "compute_capability_major", "compute_capability_minor",
        "gpu_uuid", "cuda_version", "driver_version", "sm_count",
        "total_global_mem_bytes",
    }
    import json as _json
    for mp in results_root.rglob("measurement_manifest.json"):
        with open(mp) as f:
            m = _json.load(f)
        missing = required - set(m.keys())
        if missing:
            errors.append(f"identity_fields_incomplete: {mp} missing {sorted(missing)}")
        if m.get("kb_capability_card") != "references/NVIDIA_Manuals/blackwell_compatibility/01_blackwell-architecture-compatibility.md":
            errors.append(f"kb_reference_missing: {mp}")
    return errors


ENFORCEMENT_SCRIPTS = (
    "run_suite.py",
    "validate.py",
    "verify_coverage.py",
    "verify_artifacts.py",
    "verify_kernel_contracts.py",  # R8 — AC41 cubin-level enforcement
    "trace_size_estimator.py",
    "update_goldens.py",
    "merge_evidence.py",
    "dual_run_manifest_consumer.py",
    "dual_run_manifest_emitter.py",
    "nvbit_trace_summary_adapter.py",
    "phase_cycle_ledger_aggregator.py",
    "phase_cycle_ledger_consumer.py",
    "phase_inject_orchestrator.py",
)


def check_enforcement_tooling() -> list[str]:
    """Every non-withdrawn enforcement script named in the plan must exist."""
    errors: list[str] = []
    for name in ENFORCEMENT_SCRIPTS:
        if not (SUITE_ROOT / name).exists():
            errors.append(f"enforcement_script_missing: workloads/simbench_bw/{name}")
    return errors


def check_platform_matrix() -> list[str]:
    pm = _load_toml(SUITE_ROOT / "platform_matrix.toml")
    roster = _load_toml(SUITE_ROOT / "roster.toml")
    roster_map = {r["slug"]: r for r in roster["rows"]}
    errors: list[str] = []
    for r in pm["rows"]:
        slug = r["slug"]
        if slug not in roster_map:
            errors.append(f"platform_matrix_orphan: {slug} absent from roster.toml")
            continue
        tier = roster_map[slug]["tier"]
        if tier in ("P", "P-strict") and r["platforms"] != "both":
            errors.append(f"portable_tier_missing_platform: slug={slug} tier={tier} has platforms={r['platforms']!r}")
    return errors


def check_arch_gate_platform_consistency() -> list[str]:
    roster = _load_toml(SUITE_ROOT / "roster.toml")
    pm = _load_toml(SUITE_ROOT / "platform_matrix.toml")
    pm_map = {r["slug"]: r for r in pm["rows"]}
    errors: list[str] = []
    for r in roster["rows"]:
        slug = r["slug"]
        gates = set(r["arch_gate"])
        platforms = pm_map.get(slug, {}).get("platforms")
        # Blackwell-only gates (sm_120a / sm_120f) require platforms that include rtx_pro_6000_blackwell
        has_hopper = "sm_90a" in gates
        has_blackwell = bool(gates & {"sm_120", "sm_120a", "sm_120f"})
        if platforms == "both":
            if not (has_hopper and has_blackwell):
                errors.append(f"arch_gate_platform_mismatch: slug={slug} platforms=both but arch_gate={sorted(gates)}")
        elif platforms == "h100":
            if not has_hopper:
                errors.append(f"arch_gate_platform_mismatch: slug={slug} platforms=h100 but arch_gate={sorted(gates)}")
        elif platforms == "rtx_pro_6000_blackwell":
            if not has_blackwell:
                errors.append(f"arch_gate_platform_mismatch: slug={slug} platforms=rtx_pro_6000_blackwell but arch_gate={sorted(gates)}")
    return errors


KERNEL_SOURCE_SEARCH_PATHS = ("l1", "l2", "l3", "l4")


def _find_kernel_source(slug: str) -> Path | None:
    """Resolve a kernel's .cu source path in either the legacy suite-root
    layout or the per-level layout (workloads/simbench_bw/l{1,2,3,4}/<slug>/<slug>.cu)."""
    legacy = SUITE_ROOT / f"{slug}.cu"
    if legacy.exists():
        return legacy
    for level in KERNEL_SOURCE_SEARCH_PATHS:
        candidate = SUITE_ROOT / level / slug / f"{slug}.cu"
        if candidate.exists():
            return candidate
    return None


def check_portability() -> list[str]:
    """P-strict rows must have no #ifdef __CUDA_ARCH__ branch in their .cu
    source. Kernels may live either at the legacy suite root or under the
    per-level directory layout (l1/l2/l3/l4/<slug>/<slug>.cu); both are
    searched. P rows are allowed to branch; no static check required beyond
    arch_gate consistency."""
    roster = _load_toml(SUITE_ROOT / "roster.toml")
    errors: list[str] = []
    for r in roster["rows"]:
        if r["tier"] != "P-strict":
            continue
        src = _find_kernel_source(r["slug"])
        if src is None:
            # No source yet; nothing to check (row is scaffold)
            continue
        text = src.read_text()
        if re.search(r"__CUDA_ARCH__", text):
            errors.append(f"strict_portability_violated: {r['slug']} has __CUDA_ARCH__ branch but tier=P-strict (at {src.relative_to(SUITE_ROOT)})")
    return errors


def check_splits() -> list[str]:
    """splits.toml assigns every benchmark (non-control) row to one split with
    frozen 46 / 17 / 16 counts (R44 expanded calibration by 2 to admit
    `working_set_capacity_sweep` and `warm_cold_pair_relaunch`; R46 added
    `l2_prefetch_hint_sweep` and grew calibration to 43; R47 added
    `l2_persisting_window_sweep` and grew calibration to 44; R48 added
    `carveout_sensitivity_probe` and grew calibration to 45; the sm86
    cohort regen loop's task3 added `dram_streaming_dot_product` and
    grew calibration to 46; the atomics ATOM-vs-RED pairing added
    `global_atomic_return_used` and grew validation to 17; subsequent
    atomics + vector-width members grew validation to 24; the local-memory
    family added `local_dynamic_index_array` + `local_volatile_array` and
    grew calibration to 48). Control rows
    are excluded from splits by
    design — controls are not part of the calibration / validation /
    holdout surface."""
    splits = _load_toml(SUITE_ROOT / "splits.toml")
    roster = _load_toml(SUITE_ROOT / "roster.toml")
    errors: list[str] = []
    counts = splits.get("counts", {})
    if counts.get("calibration") != 48:
        errors.append(f"split_counts: calibration != 48 (got {counts.get('calibration')})")
    if counts.get("validation") != 24:
        errors.append(f"split_counts: validation != 24 (got {counts.get('validation')})")
    if counts.get("holdout") != 16:
        errors.append(f"split_counts: holdout != 16 (got {counts.get('holdout')})")
    by_slug = {a["slug"]: a["split"] for a in splits.get("assignments", [])}
    benchmark_slugs = {r["slug"] for r in roster["rows"] if r.get("tier") != "control"}
    for slug in benchmark_slugs:
        if slug not in by_slug:
            errors.append(f"split_membership_violation: {slug} has no split assignment")
    for slug in by_slug:
        if slug not in benchmark_slugs:
            errors.append(f"split_membership_violation: {slug} in splits.toml but absent from benchmark rows in roster.toml")
    # Cross-check with roster.toml per-row split field (benchmarks only)
    for r in roster["rows"]:
        if r.get("tier") == "control":
            continue
        if by_slug.get(r["slug"]) != r["split"]:
            errors.append(f"split_membership_violation: slug={r['slug']} roster split={r['split']!r} but splits.toml says {by_slug.get(r['slug'])!r}")
    actual_counts = Counter(by_slug.values())
    for k in ("calibration", "validation", "holdout"):
        if actual_counts.get(k, 0) != counts.get(k, 0):
            errors.append(f"split_counts: actual {k}={actual_counts.get(k, 0)} != declared {counts.get(k, 0)}")
    return errors


def check_golden_size(max_mib: int = 64) -> list[str]:
    golden = SUITE_ROOT / "golden"
    if not golden.exists():
        return []
    total = 0
    for p in golden.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    mib = total / (1024 * 1024)
    if mib > max_mib:
        return [f"golden_repo_budget_exceeded: golden/ is {mib:.2f} MiB (budget {max_mib} MiB)"]
    return []


def check_xgen_notes() -> list[str]:
    roster = _load_toml(SUITE_ROOT / "roster.toml")
    errors: list[str] = []
    xgen_required_probes = {"tma_bulk_tensor", "tma_multicast", "cluster_mbarrier", "dsmem_remote"}
    for r in roster["rows"]:
        uses_xgen = bool(set(r.get("probe_dependencies", [])) & xgen_required_probes)
        has_note = r.get("cross_gen_comparability_note") is not None
        if uses_xgen and not has_note and r["tier"] in ("P", "P-strict"):
            errors.append(f"xgen_note_missing: slug={r['slug']} uses TMA/cluster/DSMEM but has no cross_gen_comparability_note")
    return errors


def _import_reference_family(family_name: str):
    """Import a reference-family module by name (e.g. "embedding") and
    return the module object or None if missing. Adds `references/` to
    sys.path so intra-family `from _common import ...` works under
    importlib spec loading."""
    import importlib.util
    ref_dir = SUITE_ROOT / "references"
    ref_path = ref_dir / f"{family_name}.py"
    if not ref_path.exists():
        return None
    if str(ref_dir) not in sys.path:
        sys.path.insert(0, str(ref_dir))
    spec = importlib.util.spec_from_file_location(f"simbench_bw_ref_{family_name}", ref_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        # A broken module is already flagged by run_suite.py / pytest;
        # this check only guards roster-vs-module contract alignment.
        return None
    return mod


def check_reference_type_runtime() -> list[str]:
    """R28-review: enforce roster.reference_type agrees with the
    reference module's declared runtime backend when the module opts in
    via a `REFERENCE_BACKEND` module attribute.

    Opt-in design: families that do not expose `REFERENCE_BACKEND` are
    skipped silently. Families with per-slug heterogeneous backends
    (e.g. some rows analytic, some cpu) should not export a single
    `REFERENCE_BACKEND`; their roster rows remain unchecked by this
    hook and must be validated by per-family logic instead.

    Fixes the R27-review blocking side issue where `control`
    family roster rows declared `reference_type = "fp32_gpu"` while
    `references/control.py` was a pure Python CPU FP32 matmul."""
    roster = _load_toml(SUITE_ROOT / "roster.toml")
    # Group covered slugs by reference_family; some slugs share a family.
    errors: list[str] = []
    by_family: dict[str, list[dict]] = {}
    for row in roster["rows"]:
        by_family.setdefault(row["reference_family"], []).append(row)
    for family_name, rows in by_family.items():
        mod = _import_reference_family(family_name)
        if mod is None:
            continue
        backend = getattr(mod, "REFERENCE_BACKEND", None)
        if backend is None:
            continue  # Opt-in: family did not declare a uniform backend.
        covers = set(getattr(mod, "FAMILY_COVERS_SLUGS", ()))
        for row in rows:
            slug = row["slug"]
            if slug not in covers:
                continue
            declared = row.get("reference_type")
            if declared != backend:
                errors.append(
                    f"reference_type_runtime_mismatch: slug={slug} "
                    f"roster.reference_type={declared!r} but "
                    f"references/{family_name}.py::REFERENCE_BACKEND={backend!r}"
                )
    return errors


# --- CLI ----------------------------------------------------------------------


CHECKS: dict[str, Callable[[], list[str]]] = {
    "roster": check_roster,
    "tier-consistency": check_tier_consistency,
    "subsystems": check_subsystems,
    "wave": check_wave,
    "enforcement-tooling": check_enforcement_tooling,
    "platform-matrix": check_platform_matrix,
    "arch-gate-platform-consistency": check_arch_gate_platform_consistency,
    "portability": check_portability,
    "splits": check_splits,
    "golden-size": check_golden_size,
    "xgen-notes": check_xgen_notes,
    "reference-type-runtime": check_reference_type_runtime,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SimBench-BW coverage + consistency verifier")
    for name in sorted(CHECKS):
        flag = f"--check-{name}"
        p.add_argument(flag, action="store_true", help=f"Run {name}")
    p.add_argument("--check-identity-in-manifest", metavar="RESULTS_ROOT", default=None,
                   help="Path under which to scan measurement_manifest.json files for identity fields")
    p.add_argument("--strict", action="store_true",
                   help="Run every check (equivalent to every --check-* flag)")
    args = p.parse_args(argv)

    any_requested = args.strict or any(getattr(args, f"check_{name.replace('-', '_')}") for name in CHECKS) or args.check_identity_in_manifest is not None
    if not any_requested:
        p.error("specify --strict or at least one --check-* flag")  # exits with code 2

    errors: list[str] = []
    for name, fn in CHECKS.items():
        attr = f"check_{name.replace('-', '_')}"
        if args.strict or getattr(args, attr):
            errors.extend(fn())
    if args.check_identity_in_manifest is not None:
        errors.extend(check_identity_in_manifest(Path(args.check_identity_in_manifest)))

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        print(f"{len(errors)} check failure(s)", file=sys.stderr)
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
