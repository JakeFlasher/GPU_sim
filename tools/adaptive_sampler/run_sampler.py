"""Operator CLI for the adaptive-sampler clause.

Reads an upstream `hrd_profile.json` plus a manifest-pairing tuple,
runs the cumulative-window MRC sampler decision tree, and writes a
strict-valid `convergence_confidence.json` bundle. `--arch` admits
both local-development arches structurally (`sm_86` desktop +
`sm_120-laptop`) per CLAUDE.md "Two-Device Local Development
Model"; the Python producer is arch-independent in implementation.
Live v3 sm_86 evidence is recorded in
`tools/BLOCKED_ON_3090TI.md` task17 Section (f) (admission record,
source-commit equality, digest round-trip). The remaining
`Cross-arch-verified` work is a future sm_120 v3 leg replaying the
same producer against a fresh sm_120-laptop HRD bundle on the
5060 laptop. Operator-facing entry point declared in
`tools/arch_capabilities.json` per the C-AC-E manifest discipline.

The CLI mirrors the controller helper
`workloads/simbench_bw/run_suite.py::_emit_trace_convergence_confidence_for_run_summary`
in priority order to keep the two admission surfaces in lockstep:

1. `trace_convergence_disabled` if `--no-trace-convergence` is passed
   (default false; the convergence stopping rule is enabled).
2. `hrd_layer_unavailable` when `bundle_underconfigured == true` or
   either of the sector / line layers has zero usable admitted bins
   per the shared `is_usable_admitted_bin` filter in
   `trace_convergence_ledger.py`.
3. `trace_convergence_window_count_underconfigured` when usable
   admitted bins exist for both layers but only one cumulative window
   of HRD evidence is available (the offline CLI ingests one HRD
   bundle; admitting `result_kind=ok` requires K+1 windows).
4. `result_kind == "ok"` would require iterative cumulative-window
   evidence which the offline CLI does not yet receive; future work
   widens the CLI to ingest a window-table input.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sampler import (  # type: ignore[import-not-found]  # noqa: E402
    disabled_admission,
    hrd_unavailable_admission,
    window_count_underconfigured_admission,
)
from trace_convergence_ledger import (  # type: ignore[import-not-found]  # noqa: E402
    HIERARCHY_LAYERS,
    MRC_QUANTILES,
    TARGET_ARCH,
    TRACE_WINDOW_SIZE_BYTES,
    build_bundle,
    manifest_pairing_index_dict,
    usable_admitted_bins,
)

_SUPPORTED_ARCHES: tuple[str, ...] = ("sm_86", TARGET_ARCH)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_sampler",
        description="The adaptive-sampler clause CLI: compute trace_convergence_confidence.",
    )
    p.add_argument("--hrd-bundle", required=True, type=Path,
                   help="Path to upstream hrd_profile.json")
    p.add_argument("--out", required=True, type=Path,
                   help="Path to write convergence_confidence.json")
    p.add_argument("--arch", required=True, choices=_SUPPORTED_ARCHES,
                   help=(
                       "Target arch (sm_86 + sm_120-laptop admitted "
                       "structurally; live v3 sm_86 evidence "
                       "recorded in tools/BLOCKED_ON_3090TI.md "
                       "task17 Section (f); remaining sm_120 v3 leg "
                       "is a future 5060-laptop replay)"
                   ))
    p.add_argument("--slug", required=True)
    p.add_argument("--shape-point", required=True)
    p.add_argument("--binary-sha256", required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--kernel-signature-sha256", required=True)
    p.add_argument("--device-name", required=True)
    p.add_argument("--cc-major", required=True, type=int)
    p.add_argument("--cc-minor", required=True, type=int)
    p.add_argument("--sm-count", required=True, type=int)
    p.add_argument("--l2-cache-size-bytes", required=True, type=int)
    p.add_argument(
        "--no-trace-convergence",
        action="store_true",
        help=(
            "Disable the convergence stopping rule; the CLI emits "
            "trace_convergence_disabled instead of attempting to admit."
        ),
    )
    return p.parse_args(argv)


def _baseline_window(observation_count: int, trace_bytes: int) -> dict:
    """Synthesize the schema-required baseline window (window_index=0)
    that all typed-unavailable bundles ship with. Per-layer per-quantile
    records are zero-initialized; the single window has
    `distance_to_previous = null` and `converged = false`."""
    return {
        "window_index": 0,
        "observation_count_cumulative": observation_count,
        "trace_bytes_cumulative": trace_bytes,
        "mrc_at_quantile": [
            {
                "hierarchy_layer": layer,
                "quantile": float(q),
                "mrc_hit_rate_fraction": 0.0,
                "quantile_observation_count": 0,
            }
            for layer in HIERARCHY_LAYERS
            for q in MRC_QUANTILES
        ],
        "distance_to_previous": None,
        "converged": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    try:
        hrd_bundle = json.loads(args.hrd_bundle.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(
            f"run_sampler: hrd_bundle not found: {args.hrd_bundle}",
            file=sys.stderr,
        )
        return 2
    except json.JSONDecodeError as exc:
        print(
            f"run_sampler: hrd_bundle JSON decode failed: {exc}",
            file=sys.stderr,
        )
        return 2
    if not isinstance(hrd_bundle, dict):
        print(
            "run_sampler: hrd_bundle must be a JSON object",
            file=sys.stderr,
        )
        return 2

    manifest_pairing_index = manifest_pairing_index_dict(
        slug=args.slug,
        shape_point=args.shape_point,
        binary_sha256=args.binary_sha256,
        source_commit=args.source_commit,
        kernel_signature_sha256=args.kernel_signature_sha256,
    )
    device = {
        "device_name": args.device_name,
        "compute_capability_major": args.cc_major,
        "compute_capability_minor": args.cc_minor,
        "sm_count": args.sm_count,
        "l2_cache_size_bytes": args.l2_cache_size_bytes,
    }

    bundle_underconfigured = bool(hrd_bundle.get("bundle_underconfigured", False))
    raw_bins = hrd_bundle.get("bins", [])
    sector_bins = usable_admitted_bins(raw_bins, layer="sector")
    line_bins = usable_admitted_bins(raw_bins, layer="line")
    cumulative_observation_count = sum(
        int(b["observation_count"]) for b in (sector_bins + line_bins)
    )

    # Decision tree mirroring the controller helper. Disabled wins
    # over all because it expresses an explicit operator opt-out.
    # The baseline window's cumulative bytes default to 0 (the disabled /
    # hrd-unavailable arms carry no usable trace-convergence evidence, and
    # their cumulative_observation_count may be 0); only the
    # underconfigured arm DID ingest one HRD window and sets it to one
    # window so it stays consistent with that arm's actual_trace_bytes.
    baseline_trace_bytes = 0
    if args.no_trace_convergence:
        admission = disabled_admission()
    elif bundle_underconfigured or not sector_bins or not line_bins:
        unavailable = "sector" if not sector_bins else "line"
        admission = hrd_unavailable_admission(
            hrd_profile_path=str(args.hrd_bundle),
            bundle_underconfigured=bundle_underconfigured,
            requested_layers=HIERARCHY_LAYERS,
            unavailable_layer=unavailable,
            usable_bin_count=(
                len(sector_bins) if unavailable == "sector" else len(line_bins)
            ),
        )
    else:
        # Usable admitted bins exist for both layers, but the offline
        # CLI receives one HRD bundle's worth of evidence (one
        # cumulative window). The K=3 stationary stopping rule
        # requires K+1 cumulative windows; the producer cannot admit
        # `result_kind=ok` from a single bundle. Emit the typed-
        # unavailable arm honestly per Codex's R37 review directive
        # rather than fabricating convergence by duplicating the
        # bundle into multiple windows.
        # Report the bytes of the ONE window this CLI actually ingested,
        # NOT (K) windows: `observed_window_count=1`, so `actual_trace_bytes`
        # must be a single `TRACE_WINDOW_SIZE_BYTES` window — otherwise an
        # operator reads ~48 MiB of evidence (close to the 64 MiB =
        # (K+1)*window requirement) for a run that only saw one 16 MiB
        # window. `required_trace_bytes` (set inside the admission to
        # (K+1)*window) carries the real gap.
        admission = window_count_underconfigured_admission(
            observed_window_count=1,
            actual_trace_bytes=TRACE_WINDOW_SIZE_BYTES,
        )
        # This arm DID ingest one HRD window: the baseline window's
        # cumulative bytes must equal the admission's actual_trace_bytes
        # (one window), matching sampler.py's window-0
        # `(k+1)*TRACE_WINDOW_SIZE_BYTES` convention.
        baseline_trace_bytes = TRACE_WINDOW_SIZE_BYTES

    windows = [
        _baseline_window(cumulative_observation_count, baseline_trace_bytes)
    ]

    bundle = build_bundle(
        manifest_pairing_index=manifest_pairing_index,
        device=device,
        hrd_profile_path=str(args.hrd_bundle),
        windows=windows,
        admission=admission,
        source_commit=args.source_commit,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")

    typed_reason = admission.get("typed_reason")
    if typed_reason:
        print(
            f"run_sampler: typed_unavailable={typed_reason}; "
            f"wrote {args.out}",
        )
    else:
        print(
            f"run_sampler: ok windows={len(windows)} "
            f"converged={admission['windows_converged']} "
            f"final_residual={admission['final_residual']:.6f}; "
            f"wrote {args.out}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
