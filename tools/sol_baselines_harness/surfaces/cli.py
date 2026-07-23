"""Command-line front end for the counter-free surfaces orchestrator.

Subcommands:

* ``list`` — print the locked surface vocabulary + the registered
  plugin's ``name``, ``schema_version``, ``impl_version``.
* ``run`` — drive the orchestrator against one baseline. Flags:
  ``--surface`` (alias for ``--enabled-surfaces``), ``--stage``
  (Stage 1 / 2 / 3 presets), ``--full-fanout``,
  ``--representative-workload <idx>``, ``--force-resurface``,
  ``--build-mode``, ``--mode {smoke,fanout}``, ``--target {local,b200}``.
* ``dry-run`` — print the constructed orchestrator config + the
  surfaces that would invoke, without spawning any subprocess.

The CLI is a thin front end over ``SurfaceOrchestrator``: it parses
the surface / stage / workload / build-mode / target flags into the
orchestrator's constructor kwargs and either prints the resolved
configuration (``dry-run``) or drives the orchestrator's ``run_all``
against a landed baseline (``run``). Every subcommand resolves the
enabled surface set against the locked vocabulary in
``VALID_SURFACES``; ``--stage`` is a shorthand for the five
eval-driving surfaces and excludes the passive ``cache_prediction``
predictor surface, which must be requested explicitly with
``--surface cache_prediction``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

_THIS_DIR = Path(__file__).resolve().parent
_HARNESS_DIR = _THIS_DIR.parent
_SCRIPTS_DIR = _HARNESS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from baselines_harness.surfaces import (  # noqa: E402
    SurfaceOrchestrator,
    UnknownSurfaceError,
    VALID_SURFACES,
    default_surface_plugins,
)
from baselines_harness.surfaces.host_probe import probe_host  # noqa: E402

REPO_ROOT = _SCRIPTS_DIR.parent

# Stage presets are escalating subsets of the locked surface vocabulary.
# The operator can always override with explicit ``--surface`` flags.
# Stage 3 enumerates the five eval-driving surfaces explicitly so it
# does not silently include the passive ``cache_prediction`` surface
# (which is registered against ``VALID_SURFACES`` but emits no
# evaluator-driving payload and has no operator value in stage presets).
_STAGE_PRESETS = {
    1: frozenset({"static", "sanitizer"}),
    2: frozenset({"static", "sanitizer", "nvbit_metrics"}),
    3: frozenset({
        "static",
        "sanitizer",
        "nvbit_metrics",
        "cupti_activity",
        "nsys",
    }),
}


def _parse_enabled_surfaces(
    args: argparse.Namespace,
) -> frozenset:
    """Resolve ``--surface`` / ``--stage`` into a concrete set.

    Operator precedence: explicit ``--surface`` flags win over
    ``--stage`` presets (the operator can scope below a stage).
    Default falls back to Stage 3 (the five eval-driving surfaces:
    ``static`` + ``sanitizer`` + ``nvbit_metrics`` +
    ``cupti_activity`` + ``nsys``) so a bare invocation exercises
    the orchestrator's full eval-driving path. The passive
    ``cache_prediction`` surface is registered against the locked
    vocabulary but is excluded from every stage preset; an operator
    who wants it must request it explicitly with
    ``--surface cache_prediction``.
    """
    if args.surface:
        for s in args.surface:
            if s not in VALID_SURFACES:
                raise UnknownSurfaceError(
                    f"unknown surface: {s!r}; "
                    f"valid: {sorted(VALID_SURFACES)}"
                )
        return frozenset(args.surface)
    if args.stage is not None:
        return _STAGE_PRESETS[args.stage]
    return _STAGE_PRESETS[3]


def _resolve_workload_subset(
    args: argparse.Namespace,
) -> tuple:
    """Resolve workload selection flags into ``(subset, policy)``.

    Workload selection contract:
    * ``--full-fanout`` and ``--representative-workload`` are
      mutually exclusive.
    * Default: lowest workload index only (``[0]``).

    The CLI records the *policy* and lets the orchestrator
    materialise the concrete index list against the problem's
    actual workload count (so ``--full-fanout`` resolves to
    ``[0..N-1]`` at dispatch time without the CLI having to read
    ``workload.jsonl``).
    """
    if args.full_fanout and args.representative_workload is not None:
        raise ValueError(
            "--full-fanout and --representative-workload are "
            "mutually exclusive."
        )
    if args.representative_workload is not None:
        if args.representative_workload < 0:
            raise ValueError(
                f"--representative-workload must be non-negative; "
                f"got {args.representative_workload}"
            )
        return [args.representative_workload], "explicit"
    if args.full_fanout:
        # Orchestrator's ``_resolve_workload_subset`` reads the
        # problem's workload count and materialises the full index
        # list at dispatch time. We pass an empty subset here as the
        # signal for "all workloads".
        return [], "full-fanout"
    return [0], "lowest-index-default"


def _cmd_list(args: argparse.Namespace) -> int:
    """Print the locked surface vocabulary + registered plugin metadata."""
    plugins = default_surface_plugins()
    rows = []
    for name in sorted(VALID_SURFACES):
        plugin = plugins[name]
        rows.append({
            "name": name,
            "schema_version": plugin.schema_version,
            "impl_version": plugin.impl_version,
        })
    print(json.dumps({"surfaces": rows}, indent=2))
    return 0


def _build_orchestrator(args: argparse.Namespace) -> SurfaceOrchestrator:
    """Construct an orchestrator from CLI args, sharing logic across run/dry-run."""
    enabled = _parse_enabled_surfaces(args)
    workload_subset, workload_policy = _resolve_workload_subset(args)
    return SurfaceOrchestrator(
        surface_plugins=default_surface_plugins(),
        enabled_surfaces=enabled,
        build_mode=args.build_mode,
        workload_subset=workload_subset,
        workload_selection_policy=workload_policy,
        force_resurface=args.force_resurface,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    """Drive the orchestrator against one baseline."""
    baseline_dir = args.baseline.resolve()
    if not baseline_dir.is_dir():
        sys.stderr.write(
            f"error: baseline directory not found at {baseline_dir}\n"
        )
        return 2
    problem_dir = args.problem_dir.resolve() if args.problem_dir else None
    if problem_dir is None or not problem_dir.is_dir():
        sys.stderr.write(
            "error: --problem-dir must point to the matching benchmark "
            "spec (e.g. sol-execbench/data/benchmark/<subset>/<NNN>/)\n"
        )
        return 2
    if args.mode == "fanout":
        manifest_path = REPO_ROOT / "scripts" / "baselines_harness" / "b200_ready_manifest.json"
        if not manifest_path.is_file():
            sys.stderr.write(
                f"error: --mode fanout requires {manifest_path.relative_to(REPO_ROOT)} "
                f"to exist. Run `python scripts/baselines_harness/b200_preflight.py` "
                f"first.\n"
            )
            return 2
        try:
            preflight = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            sys.stderr.write(
                f"error: cannot parse {manifest_path}: {exc}\n"
            )
            return 2
        if not preflight.get("ready_for_phase_2"):
            sys.stderr.write(
                f"error: --mode fanout refused; "
                f"{manifest_path.relative_to(REPO_ROOT)} reports "
                f"ready_for_phase_2=false. Resolve the listed blockers "
                f"({preflight.get('blockers', [])}) and re-probe.\n"
            )
            return 2
        if args.target == "b200":
            device = preflight.get("checks", {}).get("device", {})
            if not device.get("phase_2_target"):
                sys.stderr.write(
                    "error: --mode fanout --target b200 requires the "
                    "device probe to report phase_2_target=true (i.e. "
                    "compute capability in the sm_100 family).\n"
                )
                return 2
    orchestrator = _build_orchestrator(args)
    host_info = probe_host()
    if args.bundle_dir:
        bundle_dir = args.bundle_dir.resolve()
        # Operator-supplied explicit --bundle-dir bypasses the
        # arch-aware default. Guard against the sm_86-writes-into-
        # locked-sm_120-bundle scenario before the orchestrator
        # touches the path.
        from baselines_harness.trace_layout import (
            assert_no_sm120_overwrite,
        )
        try:
            assert_no_sm120_overwrite(bundle_dir, host_info.kind)
        except Exception as exc:
            sys.stderr.write(f"error: {exc}\n")
            return 2
    else:
        bundle_dir = _default_bundle_dir(
            baseline_dir, host_kind=host_info.kind
        )
    manifest = orchestrator.run_all(
        baseline_dir=baseline_dir,
        problem_dir=problem_dir,
        bundle_dir=bundle_dir,
        host_info=host_info,
    )
    print(json.dumps({"manifest_path": str(bundle_dir / "manifest.json"),
                      "evidence_label": manifest.get("evidence_label"),
                      "surfaces": {n: e["status"]
                                   for n, e in manifest.get("surfaces", {}).items()}},
                     indent=2))
    return 0


def _cmd_dry_run(args: argparse.Namespace) -> int:
    """Print the orchestrator's resolved config without spawning surfaces."""
    orchestrator = _build_orchestrator(args)
    print(json.dumps({
        "enabled_surfaces": sorted(orchestrator.enabled_surfaces),
        "build_mode": orchestrator.build_mode,
        "workload_subset": list(orchestrator.workload_subset),
        "workload_selection_policy": orchestrator.workload_selection_policy,
        "force_resurface": orchestrator.force_resurface,
    }, indent=2))
    return 0


def _default_bundle_dir(
    baseline_dir: Path, host_kind: Optional[str] = None
) -> Path:
    """Compute the default bundle directory for a baseline, routing
    through the cross-arch trace-layout contract.

    ``baselines/<cat>/<slug>/<lang>/`` ->
      ``scripts/baselines_harness/.harness_traces/<slug>__<lang>/``
        when ``host_kind`` is ``None``, ``local_sm_120``, or
        ``b200_sm_100a`` (legacy un-suffixed canonical form).
      ``scripts/baselines_harness/.harness_traces/<slug>__<lang>__sm86/``
        when ``host_kind == "local_sm_8_6"`` (the active 3090 Ti
        host's sm_86 arch-qualified sibling form per DEC-B).

    Delegates to ``baselines_harness.trace_layout.default_bundle_dir``;
    the routing logic lives in that module so ``report.py``, the
    surfaces CLI, and the tests share one source of truth.
    """
    from baselines_harness.trace_layout import default_bundle_dir
    traces_root = (
        REPO_ROOT / "scripts" / "baselines_harness" / ".harness_traces"
    )
    return default_bundle_dir(
        traces_root=traces_root,
        slug=baseline_dir.parent.name,
        language=baseline_dir.name,
        host_kind=host_kind,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/baselines_harness/surfaces/cli.py",
        description=(
            "Drive the counter-free analytics framework's six surfaces "
            "(five eval-driving + the passive cache_prediction "
            "predictor surface) over a landed baseline. Emits one "
            "manifest.json per (slug, language) bundle alongside the "
            "existing per-workload trace.json. No NCU; no clock-"
            "policy mutation; no privileged CUPTI surfaces."
        ),
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # `list` subcommand.
    list_p = subparsers.add_parser(
        "list",
        help="Print the locked surface vocabulary + plugin metadata.",
    )
    list_p.set_defaults(handler=_cmd_list)

    # `run` subcommand.
    run_p = subparsers.add_parser(
        "run",
        help="Drive the orchestrator against one baseline.",
    )
    _add_orchestrator_args(run_p)
    run_p.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to baselines/<cat>/<slug>/<lang>/",
    )
    run_p.add_argument(
        "--problem-dir",
        type=Path,
        default=None,
        help=(
            "Path to the matching benchmark spec "
            "(e.g. sol-execbench/data/benchmark/<subset>/<NNN>/)."
        ),
    )
    run_p.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help=(
            "Output bundle directory. Default: "
            "scripts/baselines_harness/.harness_traces/<slug>__<lang>/"
        ),
    )
    run_p.add_argument(
        "--mode",
        choices=("smoke", "fanout"),
        default="smoke",
        help=(
            "Smoke runs a single baseline locally without preflight "
            "requirements. Fanout requires a fresh b200_ready_manifest.json "
            "with ready_for_phase_2=true ."
        ),
    )
    run_p.add_argument(
        "--target",
        choices=("local", "b200"),
        default="local",
        help=(
            "Target host class. --target b200 in fanout mode additionally "
            "requires the device probe to report phase_2_target=true."
        ),
    )
    run_p.set_defaults(handler=_cmd_run)

    # `dry-run` subcommand.
    dry_p = subparsers.add_parser(
        "dry-run",
        help="Print the resolved orchestrator config without running surfaces.",
    )
    _add_orchestrator_args(dry_p)
    dry_p.set_defaults(handler=_cmd_dry_run)

    return parser


def _add_orchestrator_args(p: argparse.ArgumentParser) -> None:
    """Attach the orchestrator-shared flags to a subparser."""
    p.add_argument(
        "--surface",
        action="append",
        default=[],
        choices=sorted(VALID_SURFACES),
        help=(
            "Enabled surface (repeatable). When omitted, --stage's "
            "default Stage-3 preset (the five eval-driving surfaces) "
            "applies; pass `--surface cache_prediction` to enable "
            "the passive predictor surface explicitly."
        ),
    )
    p.add_argument(
        "--stage",
        type=int,
        choices=(1, 2, 3),
        default=None,
        help=(
            "Stage preset. 1: static + sanitizer. 2: static + sanitizer "
            "+ nvbit_metrics. 3: the five eval-driving surfaces (static "
            "+ sanitizer + nvbit_metrics + cupti_activity + nsys). The "
            "passive cache_prediction surface is registered against the "
            "locked vocabulary but excluded from every stage preset; an "
            "operator who wants it must request it explicitly with "
            "`--surface cache_prediction`."
        ),
    )
    p.add_argument(
        "--full-fanout",
        action="store_true",
        help=(
            "Run every workload (mutually exclusive with "
            "--representative-workload). Default is the lowest "
            "workload index only ."
        ),
    )
    p.add_argument(
        "--representative-workload",
        type=int,
        default=None,
        help=(
            "Run only the supplied workload index (mutually exclusive "
            "with --full-fanout)."
        ),
    )
    p.add_argument(
        "--force-resurface",
        action="store_true",
        help=(
            "Bypass manifest-based cache reuse and re-run every "
            "selected surface ."
        ),
    )
    p.add_argument(
        "--build-mode",
        choices=("judge", "profiling"),
        default="judge",
        help=(
            "judge: byte-identical landed solution.json. profiling: "
            "staged solution.json with -DSOLBENCH_PROFILING always "
            "and -DSOLBENCH_HAS_NVTX conditional on host NVTX probes."
        ),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
