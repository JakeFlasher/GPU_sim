"""Compile / sol-execbench correctness verification probe.

Refreshes ``sm120_compile_verification_probe_v1.toml`` by walking
the existing harness trace bundles at
``scripts/baselines_harness/.harness_traces/<slug>__<lang>/`` and
extracting per-slug compile + correctness evidence from
``manifest.json`` and ``trace.json``.

Why this approach: the harness already invokes ``sol-execbench``
end-to-end for every landed baseline and persists the result.
Re-running ``sol-execbench`` here would duplicate that work; the
probe instead reads the existing evidence and emits a compact,
schema-stamped summary that the ladder reducer consumes.

Per-slug compile evidence:

- ``arch_tag`` in ``manifest.json`` must equal ``sm_120``.
- At least one surface (``static``, ``cupti_activity``,
  ``nvbit_metrics``, ``sanitizer``) recorded ``status: ok``,
  which transitively implies the build completed successfully
  (the harness launches every surface from the same built
  extension).

Per-slug correctness evidence:

- ``trace.json`` must list at least one row.
- Every row's ``status`` field must be ``"PASSED"``.

Cells absent from ``.harness_traces/`` carry no probe evidence;
they are recorded with ``compile_verified=false`` and
``sol_execbench_correct_verified=false``. The ladder reducer's
fall-through assigns ``cu_compile_failed`` to a cu_present slug
without probe evidence — the conservative position.

CLI:

    python -m baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.compile_verification_probe \\
        --output scripts/baselines_harness/calibration/cache_prediction/sm120_compile_verification_probe_v1.toml \\
        [--captured-at-utc 2026-05-27T09:00:00+00:00]

Re-runs against unchanged disk state produce a byte-identical
TOML when the same ``--captured-at-utc`` is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[5]
_TRACES_ROOT = (
    _REPO_ROOT
    / "scripts"
    / "baselines_harness"
    / ".harness_traces"
)
_DEFAULT_OUTPUT = (
    _REPO_ROOT
    / "scripts"
    / "baselines_harness"
    / "calibration"
    / "cache_prediction"
    / "sm120_compile_verification_probe_v1.toml"
)

PROBE_SCHEMA_VERSION: str = "1.0"
PROBE_HOST_ARCH: str = "sm_120"

# Compile-evidence surfaces. Any one of these in ``status: ok``
# proves the harness built and ran the cell's torch extension.
_COMPILE_EVIDENCE_SURFACES: tuple[str, ...] = (
    "cupti_activity",
    "nvbit_metrics",
    "sanitizer",
    "static",
)

# Substring the surface's stderr_tail contains when the build
# step inside sol-execbench printed success. Defensive: not
# every surface emits this string, but its presence is a strong
# positive signal alongside surface status==ok.
_COMPILATION_SUCCEEDED_SUBSTRING = "Compilation succeeded"


@dataclass(frozen=True, slots=True)
class TraceBundle:
    """One per-(slug, language) trace bundle on disk.

    Field ``bundle_path`` is the directory name
    (e.g. ``problem_004_gemm_n128_k2048__cutlass``); slug + language
    are parsed from it.
    """

    slug: str
    language: str
    bundle_dir: Path


@dataclass(frozen=True, slots=True)
class ProbeRow:
    """One per-slug verification row emitted into the TOML.

    Provenance fields:

    - ``dispatch_argv``: the exact subprocess argv the harness
      invoked when capturing the trace bundle. Empty when the
      manifest does not record one (older bundle formats record
      an empty list at ``dispatch_command.argv``).
    - ``solution_path``: the repo-relative path of the
      ``solution.json`` for this (slug, language) pair. Two
      provenance sources, in preference order:
        (1) ``argv_recorded`` — extracted from the manifest's
            ``--solution`` argv flag.
        (2) ``reconstructed_from_bundle`` — when the manifest
            argv is empty, the path is reconstructed from the
            bundle's (slug, language) pair against the canonical
            ``baselines/{L1,L2,Quant,FIB}/<slug>/<language>/solution.json``
            layout (the live solution file MUST exist).
      Empty when neither source resolves.
    - ``solution_source_sha256``: SHA-256 of the live
      ``solution.json`` bytes at the resolved path; empty when
      the path does not exist or could not be read.
    - ``provenance_source``: closed-vocabulary token recording
      where the provenance came from.
    - ``failure_reason``: empty when both compile_verified and
      sol_execbench_correct_verified are True; otherwise a
      single typed token from ``FAILURE_REASON_VOCABULARY``.

    Demotion contract: when ``provenance_source = "unavailable"``,
    the row's ``compile_verified`` and
    ``sol_execbench_correct_verified`` fields are both set to
    False and the row's ``failure_reason`` is set to
    ``"provenance_unavailable"``. This prevents source-identity-
    less probe rows from driving ladder promotion.
    """

    slug: str
    language: str
    arch_tag: str
    captured_at_utc: str
    compile_verified: bool
    sol_execbench_correct_verified: bool
    compile_failed: bool
    workload_row_count: int
    passed_row_count: int
    bundle_relpath: str
    dispatch_argv: tuple[str, ...]
    solution_path: str
    solution_source_sha256: str
    provenance_source: str
    failure_reason: str


# Closed vocabulary of failure-reason tokens. Empty string is the
# implicit "no failure" value when both compile and correctness
# verified.
FAILURE_REASON_VOCABULARY: tuple[str, ...] = (
    "compile_failed",
    "compile_succeeded_correctness_failed",
    "compile_succeeded_correctness_unknown",
    "provenance_unavailable",
)

# Closed vocabulary of provenance-source tokens.
PROVENANCE_SOURCE_VOCABULARY: tuple[str, ...] = (
    "argv_recorded",
    "reconstructed_from_bundle",
    "unavailable",
)

# Canonical category roots under ``baselines/`` for solution.json
# reconstruction. Matches the layout enumerated by
# ``disk_discovery._BASELINE_CATEGORIES`` (kept in sync).
_BASELINE_CATEGORIES_FOR_PROBE: tuple[str, ...] = (
    "L1", "L2", "Quant", "FIB",
)


def _reconstruct_solution_path_from_bundle(
    slug: str, language: str, repo_root: Path
) -> Path | None:
    """Walk the canonical baselines layout to find solution.json
    for (slug, language). Returns the absolute path when found,
    None otherwise. Does NOT validate the resolved JSON's
    spec.languages — that check lives in
    ``_solution_languages_match_bundle``."""
    baselines = repo_root / "baselines"
    for category in _BASELINE_CATEGORIES_FOR_PROBE:
        candidate = (
            baselines / category / slug / language / "solution.json"
        )
        if candidate.is_file():
            return candidate
    return None


def _solution_languages_match_bundle(
    solution_file: Path, bundle_language: str
) -> bool:
    """Parse ``solution.json`` and assert
    ``spec.languages == [bundle_language]``.

    Returns False on any of:
      - file unreadable or unparseable as JSON
      - ``spec.languages`` missing, not a list, or not equal to
        the single-element ``[bundle_language]`` list

    The check is strict: a solution that declares
    ``["cutlass", "cuda_cpp"]`` does NOT match a ``cuda_cpp``
    bundle even though ``cuda_cpp`` appears in the list. Bundle
    directories are 1:1 with a single language; the solution
    must agree."""
    try:
        text = solution_file.read_text("utf-8")
    except OSError:
        return False
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not isinstance(data, dict):
        return False
    spec = data.get("spec", {})
    if not isinstance(spec, dict):
        return False
    languages = spec.get("languages", [])
    if not isinstance(languages, list):
        return False
    return languages == [bundle_language]


def _enumerate_bundles(traces_root: Path) -> list[TraceBundle]:
    bundles: list[TraceBundle] = []
    if not traces_root.is_dir():
        return bundles
    for entry in sorted(traces_root.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if "__" not in name or not name.startswith("problem_"):
            continue
        slug, _, language = name.rpartition("__")
        if not slug or not language:
            continue
        bundles.append(
            TraceBundle(slug=slug, language=language, bundle_dir=entry)
        )
    return bundles


def _read_manifest(bundle_dir: Path) -> dict[str, Any] | None:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        return json.loads(manifest_path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _read_trace(bundle_dir: Path) -> dict[str, Any] | None:
    trace_path = bundle_dir / "trace.json"
    if not trace_path.is_file():
        return None
    try:
        return json.loads(trace_path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _bundle_arch_tag(manifest: dict[str, Any]) -> str:
    arch = manifest.get("arch_tag", "")
    if isinstance(arch, str):
        return arch
    return ""


def _bundle_compile_evidence(manifest: dict[str, Any]) -> bool:
    """Returns True when at least one surface recorded status=ok.

    The surfaces enumerated in ``_COMPILE_EVIDENCE_SURFACES`` all
    invoke ``sol-execbench`` (or wrap it via compute-sanitizer)
    against the same built torch extension. Any surface with
    status=ok implies the build succeeded.
    """
    surfaces = manifest.get("surfaces", {})
    if not isinstance(surfaces, dict):
        return False
    for surface_name in _COMPILE_EVIDENCE_SURFACES:
        surface = surfaces.get(surface_name)
        if not isinstance(surface, dict):
            continue
        status = surface.get("status")
        if status == "ok":
            return True
        stderr_tail = surface.get("stderr_tail", "")
        if (
            isinstance(stderr_tail, str)
            and _COMPILATION_SUCCEEDED_SUBSTRING in stderr_tail
        ):
            return True
    return False


def _bundle_correctness_evidence(
    trace: dict[str, Any] | None,
) -> tuple[int, int, bool]:
    """Returns (workload_row_count, passed_row_count, correctness_ok).

    correctness_ok is True iff there is at least one row AND every
    row's status is PASSED. An empty trace or any non-PASSED row
    defeats this signal.
    """
    if trace is None:
        return 0, 0, False
    rows = trace.get("rows", [])
    if not isinstance(rows, list):
        return 0, 0, False
    if not rows:
        return 0, 0, False
    passed = 0
    for row in rows:
        if isinstance(row, dict) and row.get("status") == "PASSED":
            passed += 1
    return len(rows), passed, passed == len(rows)


def _bundle_captured_at(manifest: dict[str, Any]) -> str:
    host = manifest.get("host", {})
    if isinstance(host, dict):
        ts = host.get("captured_at_utc")
        if isinstance(ts, str):
            return ts
    return ""


def _bundle_dispatch_argv(manifest: dict[str, Any]) -> tuple[str, ...]:
    """Extract the dispatch argv from manifest.dispatch_command.argv.

    Returns the empty tuple when the manifest does not record one
    (older trace bundles, or any structural anomaly)."""
    dispatch = manifest.get("dispatch_command", {})
    if not isinstance(dispatch, dict):
        return ()
    argv = dispatch.get("argv", [])
    if not isinstance(argv, list):
        return ()
    cleaned: list[str] = []
    for token in argv:
        if isinstance(token, str):
            cleaned.append(token)
    return tuple(cleaned)


def _extract_solution_path_from_argv(
    argv: tuple[str, ...],
) -> str:
    """Pull the value following ``--solution`` from the argv."""
    for idx, token in enumerate(argv):
        if token == "--solution" and idx + 1 < len(argv):
            candidate = argv[idx + 1]
            return candidate if isinstance(candidate, str) else ""
    return ""


def _resolve_solution_repo_relative(
    raw_solution_path: str, repo_root: Path
) -> str:
    """Convert an absolute solution path to a repo-relative one.

    Returns ``""`` when the input is empty. Returns the input
    unchanged when it does not live under ``repo_root``."""
    if not raw_solution_path:
        return ""
    try:
        absolute = Path(raw_solution_path).resolve()
        return str(absolute.relative_to(repo_root.resolve()))
    except (ValueError, OSError):
        return raw_solution_path


def _solution_source_sha256(
    raw_solution_path: str,
) -> str:
    """Return SHA-256 of the live solution.json file at the path
    the manifest's dispatch argv recorded.

    Empty when the path is empty, the file does not exist, or it
    could not be read."""
    if not raw_solution_path:
        return ""
    try:
        path = Path(raw_solution_path)
        if not path.is_file():
            return ""
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _compute_failure_reason(
    *,
    compile_verified: bool,
    correctness_ok: bool,
    workload_rows: int,
) -> str:
    """Pick a closed-vocabulary failure reason for the row.

    Empty token when both compile and correctness pass. Other
    tokens (in ``FAILURE_REASON_VOCABULARY``) describe the
    pipeline-stage gap. ``compile_succeeded_correctness_unknown``
    fires when compile is verified but trace.json has zero rows
    (no correctness signal either way)."""
    if compile_verified and correctness_ok:
        return ""
    if not compile_verified:
        return "compile_failed"
    if workload_rows == 0:
        return "compile_succeeded_correctness_unknown"
    return "compile_succeeded_correctness_failed"


def _slug_to_row(
    bundle: TraceBundle, traces_root: Path
) -> ProbeRow | None:
    """Convert one trace bundle into a probe row.

    Returns None when the bundle has no parseable manifest or
    when its arch_tag is not sm_120 (cross-arch traces are
    ignored by this sm_120-specific probe).
    """
    manifest = _read_manifest(bundle.bundle_dir)
    if manifest is None:
        return None
    if _bundle_arch_tag(manifest) != PROBE_HOST_ARCH:
        return None
    trace = _read_trace(bundle.bundle_dir)
    surface_compile_evidence = _bundle_compile_evidence(manifest)
    workload_rows, passed_rows, correctness_ok = (
        _bundle_correctness_evidence(trace)
    )
    # Compile-evidence sources, in increasing strength:
    #   1. Any surface status=ok or stderr_tail contains
    #      "Compilation succeeded" — direct evidence the build
    #      step ran cleanly.
    #   2. trace.json has at least one workload row — sol-execbench
    #      only emits a workload row after a successful build, so
    #      any row's presence is transitive evidence of compile
    #      success.
    # Either is sufficient. The transitive route catches bundles
    # where every surface was skipped but trace.json still shows
    # PASSED rows.
    compile_evidence = (
        surface_compile_evidence or workload_rows > 0
    )
    # Compile-failed evidence: trace.json carries no rows AND no
    # surface succeeded AND the bundle exists (we tried). This
    # approximates "we attempted but the build did not succeed".
    # When a bundle is missing entirely the slug is simply absent
    # from the probe output.
    compile_failed = (
        not surface_compile_evidence and workload_rows == 0
    )
    dispatch_argv = _bundle_dispatch_argv(manifest)
    raw_solution = _extract_solution_path_from_argv(dispatch_argv)

    # Provenance resolution. Preference: argv-recorded over
    # bundle-reconstructed over unavailable. Every promoted row
    # (compile_verified=true) MUST end with a non-empty
    # solution_path and solution_source_sha256, AND the resolved
    # solution.json's spec.languages MUST equal [bundle.language].
    # Rows that fail the language match are demoted with
    # provenance_source="unavailable" (defense in depth: a
    # planted mismatch between the bundle directory's language
    # and the solution's spec.languages cannot drive ladder
    # promotion).
    provenance_source = "unavailable"
    solution_repo_rel = ""
    solution_sha = ""

    if raw_solution:
        resolved_argv = _resolve_solution_repo_relative(
            raw_solution, _REPO_ROOT
        )
        argv_sha = _solution_source_sha256(raw_solution)
        argv_path = Path(raw_solution)
        argv_languages_match = (
            argv_path.is_file()
            and _solution_languages_match_bundle(
                argv_path, bundle.language
            )
        )
        if resolved_argv and argv_sha and argv_languages_match:
            solution_repo_rel = resolved_argv
            solution_sha = argv_sha
            provenance_source = "argv_recorded"

    if provenance_source == "unavailable":
        reconstructed = _reconstruct_solution_path_from_bundle(
            bundle.slug, bundle.language, _REPO_ROOT
        )
        if reconstructed is not None and _solution_languages_match_bundle(
            reconstructed, bundle.language
        ):
            try:
                solution_repo_rel = str(
                    reconstructed.resolve().relative_to(
                        _REPO_ROOT.resolve()
                    )
                )
            except (ValueError, OSError):
                solution_repo_rel = str(reconstructed)
            try:
                solution_sha = hashlib.sha256(
                    reconstructed.read_bytes()
                ).hexdigest()
            except OSError:
                solution_sha = ""
            if solution_repo_rel and solution_sha:
                provenance_source = "reconstructed_from_bundle"
            else:
                solution_repo_rel = ""
                solution_sha = ""

    # Demotion: a row without solution-identity anchoring cannot
    # carry verified compile / correctness. The failure_reason
    # is fixed to "provenance_unavailable" so consumers can
    # distinguish this from a real compile/correctness failure.
    if provenance_source == "unavailable":
        compile_evidence_final = False
        correctness_ok_final = False
        failure_reason = "provenance_unavailable"
    else:
        compile_evidence_final = compile_evidence
        correctness_ok_final = correctness_ok
        failure_reason = _compute_failure_reason(
            compile_verified=compile_evidence,
            correctness_ok=correctness_ok,
            workload_rows=workload_rows,
        )

    return ProbeRow(
        slug=bundle.slug,
        language=bundle.language,
        arch_tag=PROBE_HOST_ARCH,
        captured_at_utc=_bundle_captured_at(manifest),
        compile_verified=compile_evidence_final,
        sol_execbench_correct_verified=correctness_ok_final,
        compile_failed=compile_failed,
        workload_row_count=workload_rows,
        passed_row_count=passed_rows,
        bundle_relpath=str(bundle.bundle_dir.relative_to(traces_root)),
        dispatch_argv=dispatch_argv,
        solution_path=solution_repo_rel,
        solution_source_sha256=solution_sha,
        provenance_source=provenance_source,
        failure_reason=failure_reason,
    )


def collect_probe_rows(
    *, traces_root: Path = _TRACES_ROOT
) -> list[ProbeRow]:
    """Walk the harness trace bundles and return per-slug probe rows.

    Multiple language variants for the same slug each produce
    their own row; downstream consumers can aggregate as needed.
    Rows are sorted ASCII by (slug, language).
    """
    bundles = _enumerate_bundles(traces_root)
    rows: list[ProbeRow] = []
    for bundle in bundles:
        row = _slug_to_row(bundle, traces_root)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda r: (r.slug, r.language))
    return rows


def _render_probe_toml(
    *,
    captured_at_utc: str,
    rows: list[ProbeRow],
) -> str:
    """Render a verification-probe TOML. Output is deterministic
    given the same inputs and the same captured_at_utc."""
    aggregate_compile = sum(1 for r in rows if r.compile_verified)
    aggregate_correct = sum(
        1 for r in rows if r.sol_execbench_correct_verified
    )
    aggregate_failed = sum(1 for r in rows if r.compile_failed)
    payload_sha = hashlib.sha256(
        "\n".join(
            f"{r.slug}|{r.language}|{int(r.compile_verified)}|"
            f"{int(r.sol_execbench_correct_verified)}"
            for r in rows
        ).encode("utf-8")
    ).hexdigest()
    lines: list[str] = []
    lines.append("# sm_120 compile / correctness verification probe.")
    lines.append("#")
    lines.append(
        "# Refreshed by walking the harness trace bundles at"
    )
    lines.append(
        "# ``scripts/baselines_harness/.harness_traces/<slug>__<lang>/``"
    )
    lines.append(
        "# and extracting per-slug compile evidence (any harness"
    )
    lines.append(
        "# surface recording status=ok or the 'Compilation succeeded'"
    )
    lines.append(
        "# stderr substring) plus correctness evidence (every"
    )
    lines.append(
        "# trace.json row status=PASSED)."
    )
    lines.append("")
    lines.append("[meta]")
    lines.append(f'schema_version = "{PROBE_SCHEMA_VERSION}"')
    lines.append(f'host_arch = "{PROBE_HOST_ARCH}"')
    lines.append(f'captured_at_utc = "{captured_at_utc}"')
    lines.append(f"row_count = {len(rows)}")
    lines.append(f"aggregate_compile_verified = {aggregate_compile}")
    lines.append(
        f"aggregate_sol_execbench_correct_verified = {aggregate_correct}"
    )
    lines.append(f"aggregate_compile_failed = {aggregate_failed}")
    lines.append(f'payload_sha256 = "{payload_sha}"')
    lines.append("")
    for row in rows:
        lines.append("[[entries]]")
        lines.append(f'slug = "{row.slug}"')
        lines.append(f'language = "{row.language}"')
        lines.append(f'arch_tag = "{row.arch_tag}"')
        lines.append(f'bundle_captured_at_utc = "{row.captured_at_utc}"')
        lines.append(
            f"compile_verified = {str(row.compile_verified).lower()}"
        )
        lines.append(
            f"sol_execbench_correct_verified = "
            f"{str(row.sol_execbench_correct_verified).lower()}"
        )
        lines.append(
            f"compile_failed = {str(row.compile_failed).lower()}"
        )
        lines.append(f"workload_row_count = {row.workload_row_count}")
        lines.append(f"passed_row_count = {row.passed_row_count}")
        lines.append(f'bundle_relpath = "{row.bundle_relpath}"')
        if row.dispatch_argv:
            argv_repr = ", ".join(
                f'"{tok.replace(chr(92), chr(92)*2).replace(chr(34), chr(92)+chr(34))}"'
                for tok in row.dispatch_argv
            )
            lines.append(f"dispatch_argv = [{argv_repr}]")
        else:
            lines.append("dispatch_argv = []")
        lines.append(f'solution_path = "{row.solution_path}"')
        lines.append(
            f'solution_source_sha256 = "{row.solution_source_sha256}"'
        )
        lines.append(
            f'provenance_source = "{row.provenance_source}"'
        )
        lines.append(f'failure_reason = "{row.failure_reason}"')
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def refresh_probe(
    *,
    output: Path,
    captured_at_utc: str,
    traces_root: Path = _TRACES_ROOT,
) -> Path:
    """Walk the trace bundles and write the probe TOML to disk.

    Returns the output path on success.
    """
    rows = collect_probe_rows(traces_root=traces_root)
    toml_text = _render_probe_toml(
        captured_at_utc=captured_at_utc, rows=rows
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(toml_text, encoding="utf-8")
    return output


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh the sm_120 compile / correctness verification "
            "probe by walking the harness trace bundles."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Output path for the probe TOML.",
    )
    parser.add_argument(
        "--captured-at-utc",
        type=str,
        default=None,
        help=(
            "ISO-8601 UTC timestamp to stamp into meta. Defaults "
            "to the current UTC time at second precision."
        ),
    )
    parser.add_argument(
        "--traces-root",
        type=Path,
        default=_TRACES_ROOT,
        help="Override the harness traces root for testing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    captured = args.captured_at_utc or datetime.now(
        timezone.utc
    ).replace(microsecond=0).isoformat()
    output = refresh_probe(
        output=args.output,
        captured_at_utc=captured,
        traces_root=args.traces_root,
    )
    print(f"emitted: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
