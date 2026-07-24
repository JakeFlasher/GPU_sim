"""Emit ``sm120_coverage_expansion_diagnostics_v1.toml``.

Per-cell diagnostic records covering compile / correctness
verification fields, baseline-ledger-derived NCU / Tier-2 /
predictor evidence, validation-only MAPE evidence, simulator-OOD
typed reasons, arch-incompat classification, AND the
quarantine-sidecar surface. The diagnostics sidecar is the
upstream of the coverage-readiness ladder: every ladder state
reduces from a diagnostics row plus the env-set memberships.

Schema (every field is always emitted; defaults document the
"no evidence yet" state for unharvested cells):

    [meta]
    schema_version
    host_arch
    captured_at_utc
    total_problems
    corpus_sha256

    [[entries]]
    slug
    ladder_state             # reduction of this row + env
    compile_verified         # bool: probe says compile succeeded
    sol_execbench_correct    # bool: probe says correctness passed
    compile_failed           # bool: probe says compile attempted + failed
    ncu_label_available      # bool: in baseline ledger
    tier2_trace_available    # bool: in baseline ledger
    predictor_prediction_available  # bool: in baseline ledger
    mape_scored              # bool: in baseline historical split (validation_only)
    arch_incompat_nvfp4_tcgen05    # bool: env-set membership
    ext_a2_pending           # bool: env-set membership
    ext_b_pending            # bool: env-set membership
    simulator_ood_typed_reason     # str | "" — one of
                                   #   _VALID_SIMULATOR_OOD_TYPED_REASONS
                                   #   when the cell is in
                                   #   sm120_simulator_unavailable_cells_v1.toml
                                   #   (empty for cells outside the manifest)
    # Quarantine-sidecar surface (full closed contract).
    ncu_attempted                  # bool; default false
    ncu_cv_observed                # float; NULL_FLOAT (-1.0) when null
    ncu_quarantine_reason          # str; default ""
    tier2_attempted                # bool; default false
    tier2_truncation_record_count  # int; NULL_INT (-1) when null
    tier2_quarantine_reason        # str; default ""
    correctness_failed             # bool; True iff compile_verified
                                   #   but at least one workload failed
                                   #   correctness (derived from probe)
    predictor_design_coverage_missing # bool; default false

The quarantine-sidecar surface fields are emitted with documented
defaults for cells the harvest pipeline has not yet touched. The
schema is stable; future harvest work only populates VALUES, not
shape. ``NULL_FLOAT`` and ``NULL_INT`` are documented sentinels
for fields TOML cannot encode as ``null``; ``is_null_float()`` /
``is_null_int()`` helpers translate them back to ``None``
semantics for QA loaders.

Determinism: with the same disk state and the same explicit
``--captured-at-utc``, re-running the emitter produces a
byte-identical TOML.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

_TOOLS_ROOT = Path(__file__).resolve().parents[4]
if str(_TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLS_ROOT))

from sol_baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.disk_discovery import (
    DiscoveryResult,
    build_diagnostics_and_env,
    probe_disk_state,
)
from sol_baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.ladder_qa import (
    compute_corpus_sha256,
)
from sol_baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.ladder_reducer import (
    emit_ladder_state,
)


_REPO_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_OUTPUT = (
    _REPO_ROOT
    / "scripts"
    / "baselines_harness"
    / "calibration"
    / "cache_prediction"
    / "sm120_coverage_expansion_diagnostics_v1.toml"
)
_SIMULATOR_UNAVAILABLE = (
    _REPO_ROOT
    / "scripts"
    / "baselines_harness"
    / "calibration"
    / "cache_prediction"
    / "sm120_simulator_unavailable_cells_v1.toml"
)

DIAGNOSTICS_SCHEMA_VERSION: str = "1.0"
DIAGNOSTICS_HOST_ARCH: str = "sm_120"
DIAGNOSTICS_TOTAL_PROBLEMS: int = 235

# TOML cannot represent ``null``. The quarantine-sidecar schema
# uses these sentinel values for nullable scalar fields; QA
# loaders translate them back to ``None`` on read. Sentinels are
# documented in-line in the emitter and re-asserted by the
# diagnostics test suite.
#
#   - ``NULL_FLOAT = -1.0`` for ``ncu_cv_observed``. A real
#     coefficient-of-variation is in [0, +inf); -1.0 is
#     unambiguously out of range.
#   - ``NULL_INT = -1`` for ``tier2_truncation_record_count``.
#     A real record count is in [0, +inf); -1 is unambiguously
#     out of range.
NULL_FLOAT: float = -1.0
NULL_INT: int = -1


def is_null_float(value: float) -> bool:
    """Translate the sentinel back to ``None`` semantics for
    QA / lookup loaders."""
    return value == NULL_FLOAT


def is_null_int(value: int) -> bool:
    return value == NULL_INT


def _read_simulator_ood_typed_reasons(
    manifest_path: Path,
) -> dict[str, str]:
    """Returns a slug → simulator-OOD typed-reason map. Slugs not
    in the manifest are absent from the dict."""
    if not manifest_path.is_file():
        return {}
    data = tomllib.loads(manifest_path.read_text("utf-8"))
    out: dict[str, str] = {}
    for entry in data.get("excluded", []):
        slug = entry.get("cell_name", "")
        reason = entry.get("typed_reason", "")
        if (
            isinstance(slug, str)
            and slug.startswith("problem_")
            and isinstance(reason, str)
        ):
            out[slug] = reason
    return out


def _toml_escape(value: str) -> str:
    """TOML string escaping (we only need backslash + quote;
    artifact filenames + slugs don't contain control chars)."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def render_diagnostics_toml(
    *,
    captured_at_utc: str,
    discovery: DiscoveryResult,
    ood_reason_by_slug: dict[str, str],
) -> str:
    """Render the diagnostics sidecar TOML.

    Each entry's ``ladder_state`` is computed via
    ``emit_ladder_state`` from the same diagnostics dict the
    ladder emission uses — the sidecar is by construction
    consistent with the ladder.
    """
    diagnostics, env = build_diagnostics_and_env(discovery)
    lines: list[str] = []
    lines.append(
        "# sm_120 coverage-expansion per-cell diagnostics sidecar."
    )
    lines.append("#")
    lines.append(
        "# One entry per problem slug; the ``ladder_state`` field"
    )
    lines.append(
        "# is the reducer's output over this row's evidence plus"
    )
    lines.append(
        "# the env-set memberships. The sibling coverage-readiness"
    )
    lines.append(
        "# ladder manifest reduces over the same evidence and MUST"
    )
    lines.append(
        "# carry the same per-slug state token (asserted by"
    )
    lines.append(
        "# test_sm120_coverage_expansion_diagnostics.py)."
    )
    lines.append("")
    lines.append("[meta]")
    lines.append(f'schema_version = "{DIAGNOSTICS_SCHEMA_VERSION}"')
    lines.append(f'host_arch = "{DIAGNOSTICS_HOST_ARCH}"')
    lines.append(f'captured_at_utc = "{captured_at_utc}"')
    lines.append(f"total_problems = {DIAGNOSTICS_TOTAL_PROBLEMS}")
    lines.append(
        f'corpus_sha256 = '
        f'"{compute_corpus_sha256(list(discovery.corpus_slugs))}"'
    )
    lines.append("")
    for slug in discovery.corpus_slugs:
        d = diagnostics[slug]
        state = emit_ladder_state(slug, d, env)
        compile_verified = bool(d.get("cu_compiles", False))
        sol_correct = bool(d.get("sol_execbench_correct", False))
        compile_failed = bool(d.get("compile_failed", False))
        ncu_avail = bool(d.get("ncu_label_available", False))
        tier2_avail = bool(d.get("tier2_trace_available", False))
        pred_avail = bool(d.get("predictor_prediction_available", False))
        mape_scored = bool(d.get("mape_scored", False))
        arch_incompat = slug in discovery.arch_incompat_nvfp4_tcgen05
        ext_a2 = slug in discovery.ext_a2_pending
        ext_b = slug in discovery.ext_b_pending
        ood_reason = ood_reason_by_slug.get(slug, "")
        # Quarantine-sidecar surface. Defaults for cells the
        # harvest pipeline has NOT yet touched:
        #   ncu_attempted = False              (no NCU run for this cell yet)
        #   ncu_cv_observed = NULL_FLOAT       (no CV measurement yet)
        #   ncu_quarantine_reason = ""         (no quarantine reason recorded)
        #   tier2_attempted = False            (no Tier-2 harvest yet)
        #   tier2_truncation_record_count = NULL_INT
        #   tier2_quarantine_reason = ""
        #   correctness_failed = derived from probe (compile_verified
        #                                            without all-PASSED)
        #   predictor_design_coverage_missing = False (no lookup gap yet)
        #
        # Post-pilot harvest work populates these from real
        # measurements: the NCU 3-run repeatability harvester
        # sets ``ncu_attempted`` / ``ncu_cv_observed`` /
        # ``ncu_quarantine_reason``; the Tier-2 NVBit harvester
        # sets ``tier2_attempted`` /
        # ``tier2_truncation_record_count`` /
        # ``tier2_quarantine_reason``; the canonical-bucket
        # extension sets ``predictor_design_coverage_missing``.
        # Quarantine-sidecar surface: read from the diagnostics
        # dict that ``build_diagnostics_and_env`` populates.
        # Pilot-cohort cells overlay their pilot NCU / Tier-2 /
        # predictor evidence; cells outside the pilot cohort keep
        # the documented defaults (False / "" / NULL_FLOAT /
        # NULL_INT) because the pilot artifacts only carry the
        # 12 pilot rows.
        ncu_attempted = bool(d.get("ncu_attempted", False))
        ncu_cv_raw = d.get("ncu_cv_observed", None)
        ncu_cv_observed = (
            float(ncu_cv_raw) if ncu_cv_raw is not None else NULL_FLOAT
        )
        ncu_quarantine_reason = str(d.get("ncu_quarantine_reason", ""))
        tier2_attempted = bool(d.get("tier2_attempted", False))
        tier2_truncation_raw = d.get("tier2_truncation_record_count", None)
        tier2_truncation_record_count = (
            int(tier2_truncation_raw)
            if tier2_truncation_raw is not None
            else NULL_INT
        )
        tier2_quarantine_reason = str(d.get("tier2_quarantine_reason", ""))
        correctness_failed_field = slug in discovery.correctness_failed
        predictor_design_coverage_missing = False

        lines.append("[[entries]]")
        lines.append(f'slug = "{_toml_escape(slug)}"')
        lines.append(f'ladder_state = "{state}"')
        lines.append(
            f"compile_verified = {str(compile_verified).lower()}"
        )
        lines.append(
            f"sol_execbench_correct = {str(sol_correct).lower()}"
        )
        lines.append(
            f"compile_failed = {str(compile_failed).lower()}"
        )
        lines.append(
            f"ncu_label_available = {str(ncu_avail).lower()}"
        )
        lines.append(
            f"tier2_trace_available = {str(tier2_avail).lower()}"
        )
        lines.append(
            f"predictor_prediction_available = {str(pred_avail).lower()}"
        )
        lines.append(f"mape_scored = {str(mape_scored).lower()}")
        lines.append(
            f"arch_incompat_nvfp4_tcgen05 = {str(arch_incompat).lower()}"
        )
        lines.append(f"ext_a2_pending = {str(ext_a2).lower()}")
        lines.append(f"ext_b_pending = {str(ext_b).lower()}")
        lines.append(
            f'simulator_ood_typed_reason = "{_toml_escape(ood_reason)}"'
        )
        # Quarantine-sidecar surface (always emitted; defaults for
        # unharvested cells use the documented sentinel values).
        lines.append(
            f"ncu_attempted = {str(ncu_attempted).lower()}"
        )
        lines.append(
            f"ncu_cv_observed = {ncu_cv_observed}"
        )
        lines.append(
            f'ncu_quarantine_reason = "{_toml_escape(ncu_quarantine_reason)}"'
        )
        lines.append(
            f"tier2_attempted = {str(tier2_attempted).lower()}"
        )
        lines.append(
            f"tier2_truncation_record_count = {tier2_truncation_record_count}"
        )
        lines.append(
            f'tier2_quarantine_reason = "{_toml_escape(tier2_quarantine_reason)}"'
        )
        lines.append(
            f"correctness_failed = {str(correctness_failed_field).lower()}"
        )
        lines.append(
            f"predictor_design_coverage_missing = "
            f"{str(predictor_design_coverage_missing).lower()}"
        )
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def emit_diagnostics_sidecar(
    *,
    output: Path,
    captured_at_utc: str,
    discovery: DiscoveryResult | None = None,
    simulator_unavailable_manifest: Path = _SIMULATOR_UNAVAILABLE,
) -> Path:
    """Build the diagnostics TOML and write it to disk."""
    discovery_resolved: DiscoveryResult = (
        discovery if discovery is not None else probe_disk_state()
    )
    ood_map = _read_simulator_ood_typed_reasons(
        simulator_unavailable_manifest
    )
    toml_text = render_diagnostics_toml(
        captured_at_utc=captured_at_utc,
        discovery=discovery_resolved,
        ood_reason_by_slug=ood_map,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(toml_text, encoding="utf-8")
    return output


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Emit sm120_coverage_expansion_diagnostics_v1.toml "
            "from disk evidence."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--captured-at-utc",
        type=str,
        default=None,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    captured = args.captured_at_utc or datetime.now(
        timezone.utc
    ).replace(microsecond=0).isoformat()
    output = emit_diagnostics_sidecar(
        output=args.output,
        captured_at_utc=captured,
    )
    print(f"emitted: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
