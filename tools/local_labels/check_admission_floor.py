"""Operator-facing admission-floor self-check tool.

The operator runs this CLI on either host (the current 3090 Ti
host pre-pivot or the receiving RTX 5060 laptop post-pivot) to
verify the current ``data/local_labels/label_admitted.json``
against the contract's smoke-test floor. The tool:

1. Loads the admitted file (default
   ``data/local_labels/label_admitted.json``).
2. Validates it against the strict admitted-cell schema
   (``workloads/simbench_bw/schemas/local_label_admitted.schema.json``).
3. Counts admitted cells per arch.
4. Reports the verdict against the requested floor.
5. Exits 0 iff the requested arch slice meets the floor (or, for
   ``--cross-arch``, both ``sm_86`` and ``sm_120`` meet the floor).

The default floor (5) matches the contract's
``coverage_gate.smoke_test_floor_per_arch``. The tool replaces
the ad-hoc ``python3 -c "import json; ..."`` snippets in
SWEEP_RUNBOOK.md and the round summaries with a single bounded
CLI surface so the operator's reading-path is one command, not
a copy-paste recipe.

CLI:

  python3 tools/local_labels/check_admission_floor.py \
      [--arch sm_86|sm_120] [--cross-arch] \
      [--floor N] \
      [--admitted-path <path>] \
      [--schema-path <path>] \
      [--quiet] \
      [--source-commit <SHA> | --all-source-commits]

Exactly one of ``--arch`` or ``--cross-arch`` must be given. The
default ``--floor`` is 5. A non-zero exit code means the floor is
not met OR the admitted file failed schema validation.

Source-commit filtering: when neither ``--source-commit`` nor
``--all-source-commits`` is passed, the helper auto-derives
``git rev-parse HEAD`` and filters live cells to that SHA. This
mirrors the comparison_runner's effective join grain (the runner
joins on the full 7-tuple including source_commit). Pass
``--source-commit <SHA>`` to filter on a specific commit (e.g.,
the pre-sweep HEAD when verifying post-sweep on a host whose
HEAD has advanced). Pass ``--all-source-commits`` for the
legacy behavior of counting unique
``(arch, slug, shape_point)`` identities regardless of
``source_commit`` (use only when the verdict is meant to
ignore the runner's join grain).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import jsonschema


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]

# Reuse the SINGLE canonical build-variant identity normalizer rather than
# re-deriving it here: a second copy could diverge (the exact failure class the
# aggregator's normalizer was hardened against). Path-robust sibling import so
# this works both as a script and when imported as a library.
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from repeatability_aggregator import _variant_identity  # noqa: E402

_DEFAULT_ADMITTED_PATH = _REPO_ROOT / "data" / "local_labels" / "label_admitted.json"
_DEFAULT_SCHEMA_PATH = (
    _REPO_ROOT
    / "workloads"
    / "simbench_bw"
    / "schemas"
    / "local_label_admitted.schema.json"
)


def _resolve_default_source_commit(repo_root: Path) -> str | None:
    """Return ``git rev-parse HEAD`` resolved in ``repo_root``, or
    None if git is unavailable or ``repo_root`` is not a git repo.

    The helper's default source_commit (when neither
    ``--source-commit`` nor ``--all-source-commits`` is passed)
    must mirror the comparison_runner's effective join grain:
    the runner joins on the full 7-tuple including
    ``source_commit`` taken from each predictor bundle's
    manifest_pairing_index. Right after a sweep, that
    source_commit is the pre-sweep HEAD — which equals current
    git HEAD on the same host. Auto-derive HEAD here so the
    documented runbook commands (which omit ``--source-commit``)
    produce a verdict aligned with what the runner would emit.

    For post-sweep verification on a host whose HEAD has
    advanced past the pre-sweep HEAD, the operator passes
    ``--source-commit <SHA>`` explicitly. For the legacy
    pre-fix behavior (count across all source commits),
    the operator passes ``--all-source-commits``.
    """

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    head_sha = result.stdout.strip()
    return head_sha or None


def _load_strict_schema(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_admitted(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"admitted file not found at {path!s}",
        )
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _count_per_arch(
    admitted: dict,
    *,
    source_commit: str | None = None,
) -> Counter[str]:
    """Count admitted cells per arch, restricted to
    ``evidence_kind == "live"`` AND deduplicated by cell identity.

    The cell identity is MIXED-ARITY (mirroring
    ``repeatability_aggregator._cell_key``): a base cell keeps the historical
    3-tuple ``(arch, slug, shape_point)`` verbatim, while a register-cap build
    variant widens to ``(arch, slug, shape_point, "register_cap_<cap>")``. So
    two cap variants of one shape count as TWO distinct admitted cells instead
    of collapsing to one. A cell whose ``register_cap`` / ``variant_id`` pair is
    inconsistent or a noncanonical alias is failed closed (skipped from the
    count) via the same canonical normalizer the aggregator uses.

    The contract's smoke-test floor is live-only:
    ``comparison_runner`` counts only cells with
    ``evidence_kind == "live"`` toward
    ``per_arch.<arch>.coverage_gate_verdict.headline_count_live``,
    and the strict admitted-cell schema documents that
    synthetic cells must NOT satisfy the smoke-test gate.

    The runner ALSO joins on the full 7-tuple — which
    includes ``source_commit`` — so historical/superseded
    live records (preserved by the materializer's
    ``--preserve-historical-source-commit`` discipline)
    surface as ``comparison_runner_join_key_unmatched``
    rather than contributing to the headline live count.
    The self-check must mirror that grain: count UNIQUE
    ``(arch, slug, shape_point)`` cell identities so a
    corpus with 4 distinct cells + 1 historical duplicate
    of one of them does not falsely admit at floor=5.

    The optional ``source_commit`` parameter narrows
    further: only cells whose ``source_commit`` matches the
    given value contribute. Operators who know the
    pre-sweep HEAD pass it explicitly; the default (None)
    counts every live cell identity regardless of
    ``source_commit`` (the dedup alone collapses
    duplicates into one identity).
    """

    cells = admitted.get("cells", []) or []
    seen_identities: set[tuple[str, ...]] = set()
    counts: Counter[str] = Counter()
    for cell in cells:
        if cell.get("evidence_kind") != "live":
            continue
        if (
            source_commit is not None
            and cell.get("source_commit") != source_commit
        ):
            continue
        # Build-variant identity: fail closed on an inconsistent / noncanonical
        # register_cap/variant_id pair (do not silently count it), then widen
        # the key only when a canonical variant is present (base cells keep the
        # exact 3-tuple).
        variant_id, _register_cap, variant_error = _variant_identity(cell)
        if variant_error is not None:
            continue
        base = (
            str(cell.get("arch", "")),
            str(cell.get("slug", "")),
            str(cell.get("shape_point", "")),
        )
        identity = base + (variant_id,) if variant_id else base
        if identity in seen_identities:
            continue
        seen_identities.add(identity)
        counts[identity[0]] += 1
    return counts


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the local-label admission floor against "
            "data/local_labels/label_admitted.json."
        ),
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--arch",
        choices=("sm_86", "sm_120"),
        help=(
            "Check a single arch's admitted-cell count against "
            "the floor."
        ),
    )
    target.add_argument(
        "--cross-arch",
        action="store_true",
        help=(
            "Check that both sm_86 and sm_120 admitted-cell counts "
            "meet the floor (cross-arch closure verdict)."
        ),
    )
    parser.add_argument(
        "--floor",
        type=int,
        default=5,
        help=(
            "Per-arch admitted-cell floor (default: 5, matching "
            "the contract's coverage_gate.smoke_test_floor_per_arch)."
        ),
    )
    parser.add_argument(
        "--admitted-path",
        type=Path,
        default=_DEFAULT_ADMITTED_PATH,
        help=(
            f"Path to the admitted-cells JSON file "
            f"(default: {_DEFAULT_ADMITTED_PATH!s})."
        ),
    )
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=_DEFAULT_SCHEMA_PATH,
        help=(
            f"Path to the strict admitted-cell schema "
            f"(default: {_DEFAULT_SCHEMA_PATH!s})."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Emit only the verdict token (e.g., 'sm_86=admitted' or "
            "'cross_arch=below_floor') on stdout."
        ),
    )
    source_commit_group = parser.add_mutually_exclusive_group()
    source_commit_group.add_argument(
        "--source-commit",
        default=None,
        help=(
            "Optional. When set, count only live cells whose "
            "source_commit equals this SHA. Use when the operator "
            "knows the pre-sweep HEAD and wants to mirror the "
            "comparison runner's full 7-tuple join exactly. "
            "Mutually exclusive with --current-head."
        ),
    )
    source_commit_group.add_argument(
        "--current-head",
        action="store_true",
        help=(
            "Auto-derive `git rev-parse HEAD` and count only "
            "live cells whose source_commit equals it. Use when "
            "the operator runs the self-check immediately after "
            "a fresh sweep and wants to mirror what "
            "comparison_runner would join right now. Mutually "
            "exclusive with --source-commit. The default (when "
            "neither --current-head nor --source-commit is "
            "given) is to count every unique "
            "(arch, slug, shape_point) live cell identity "
            "regardless of source_commit — see "
            "`--all-source-commits` for the legacy explicit "
            "synonym."
        ),
    )
    source_commit_group.add_argument(
        "--all-source-commits",
        action="store_true",
        help=(
            "Explicit synonym for the default: count every "
            "unique (arch, slug, shape_point) live cell "
            "identity regardless of source_commit. Codex "
            "Round-22 review P2 fix made this the default; "
            "the flag is retained for documentation and "
            "back-compat with existing operator runbooks that "
            "pass it explicitly. Mutually exclusive with "
            "--source-commit and --current-head."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.floor < 0:
        sys.stderr.write(
            "check_admission_floor: --floor must be a non-negative integer\n"
        )
        return 2

    try:
        admitted = _load_admitted(args.admitted_path)
    except FileNotFoundError as exc:
        sys.stderr.write(f"check_admission_floor: {exc}\n")
        return 2
    except json.JSONDecodeError as exc:
        sys.stderr.write(
            f"check_admission_floor: admitted file at "
            f"{args.admitted_path!s} did not parse as JSON: {exc}\n"
        )
        return 2

    try:
        schema = _load_strict_schema(args.schema_path)
    except FileNotFoundError:
        sys.stderr.write(
            f"check_admission_floor: schema file not found at "
            f"{args.schema_path!s}\n"
        )
        return 2
    except json.JSONDecodeError as exc:
        sys.stderr.write(
            f"check_admission_floor: schema file at "
            f"{args.schema_path!s} did not parse as JSON: {exc}\n"
        )
        return 2

    try:
        jsonschema.validate(instance=admitted, schema=schema)
    except jsonschema.SchemaError as exc:
        sys.stderr.write(
            f"check_admission_floor: schema at "
            f"{args.schema_path!s} is not a valid JSON Schema: "
            f"{exc.message}\n"
        )
        return 2
    except jsonschema.ValidationError as exc:
        sys.stderr.write(
            f"check_admission_floor: admitted file at "
            f"{args.admitted_path!s} failed strict-schema validation: "
            f"{exc.message}\n"
        )
        return 2

    # Codex Round-22 review P2 fix: the default is now to count
    # across all source_commits (the legacy pre-R33 semantic).
    # The prior default (auto-derive current HEAD) false-negatived
    # on committed `label_admitted.json` corpora whose cells carry
    # historical sweep source_commits — the documented operator
    # command `check_admission_floor.py --arch sm_86 --floor 5`
    # would report below_floor on a file with >5 live cells just
    # because the current HEAD doesn't match any sweep's
    # source_commit. The consumer-runtime-mirror discipline (R33
    # BL-20260509 update) is still accessible via the explicit
    # --current-head flag for operators running the self-check
    # immediately after a fresh sweep.
    effective_source_commit: str | None
    if args.source_commit is not None:
        effective_source_commit = args.source_commit
    elif args.current_head:
        effective_source_commit = _resolve_default_source_commit(_REPO_ROOT)
        if effective_source_commit is None:
            sys.stderr.write(
                "check_admission_floor: --current-head requested but "
                "`git rev-parse HEAD` failed in "
                f"{_REPO_ROOT!s}; pass --source-commit <SHA> "
                "explicitly\n"
            )
            return 2
    else:
        # Default + --all-source-commits both land here.
        effective_source_commit = None

    counts = _count_per_arch(
        admitted, source_commit=effective_source_commit,
    )

    def _verdict_for_arch(arch: str) -> str:
        return "admitted" if counts.get(arch, 0) >= args.floor else "below_floor"

    if args.cross_arch:
        sm_86_verdict = _verdict_for_arch("sm_86")
        sm_120_verdict = _verdict_for_arch("sm_120")
        cross_admitted = (
            sm_86_verdict == "admitted"
            and sm_120_verdict == "admitted"
        )
        verdict_token = (
            "cross_arch=admitted" if cross_admitted
            else "cross_arch=below_floor"
        )
        if args.quiet:
            sys.stdout.write(verdict_token + "\n")
        else:
            sys.stdout.write(
                f"sm_86={counts.get('sm_86', 0)}/{args.floor} ({sm_86_verdict}); "
                f"sm_120={counts.get('sm_120', 0)}/{args.floor} ({sm_120_verdict}); "
                f"{verdict_token}\n"
            )
        return 0 if cross_admitted else 1

    arch = args.arch
    arch_count = counts.get(arch, 0)
    arch_verdict = _verdict_for_arch(arch)
    verdict_token = f"{arch}={arch_verdict}"
    if args.quiet:
        sys.stdout.write(verdict_token + "\n")
    else:
        sys.stdout.write(
            f"{arch}={arch_count}/{args.floor} ({arch_verdict})\n"
        )
    return 0 if arch_verdict == "admitted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
