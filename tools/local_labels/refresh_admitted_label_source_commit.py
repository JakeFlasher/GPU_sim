"""Refresh ``source_commit`` on selected ``label_admitted.json``
records so the cells align with the audit's accept window after a
follow-up commit.

The companion utility
``tools/local_labels/refresh_predictor_trace_source_commit.py``
refreshes predictor-side trace JSON sidecars. This utility is its
label-side counterpart: when a round's selected admitted cells
were captured at HEAD = X, and one or more follow-up commits
advance HEAD past the audit's ``{HEAD, HEAD^}`` accept window,
the label records' recorded ``source_commit = X`` falls outside
the window and the comparison runner's join key drifts.

Per ``BL-20260508-post-commit-recapture-discipline``, the
canonical maintenance operation is to rewrite the ``source_commit``
field to the current HEAD value while preserving the load-bearing
identity fields (``arch``, ``slug``, ``shape_point``,
``binary_sha256``, ``kernel_signature_sha256``,
``ncu_metric_version``). The rewrite is safe because
``source_commit`` is a HEAD-derived provenance annotation, not a
content-derived hash; the underlying labels' L2 / L1 means and
stddevs were measured against the binary identified by
``binary_sha256`` (verifiable independently). The maintenance
must preserve every other field byte-identical so downstream
schema validators still admit.

CLI:

    python3 tools/local_labels/refresh_admitted_label_source_commit.py \
        --label-admitted data/local_labels/label_admitted.json \
        --cells <arch>/<slug>/<shape_point>[,...]

The ``--label-admitted`` flag accepts any file that conforms to
the local-label-admitted schema (the same v4 schema validates
both the operator-facing ``data/local_labels/label_admitted.json``
and any selected-label artifact emitted by
``select_best_config_labels.py`` under a per-round strengthening
directory).

Exit codes:
  0 -- every named cell's source_commit was refreshed (or already
       at the target HEAD).
  2 -- one or more cells failed validation (missing cell, missing
       required field, malformed JSON, schema validation failure
       on the rewritten file).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    import jsonschema  # type: ignore
except ImportError:
    jsonschema = None

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_PATH = (
    _REPO_ROOT / "workloads" / "simbench_bw" / "schemas"
    / "local_label_admitted.schema.json"
)


def _git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _refresh_cells(
    *,
    label_admitted: dict,
    cells_to_refresh: list[tuple[str, str, str]],
    target_head: str,
) -> list[tuple[str, str, str, str | None]]:
    """Mutate ``label_admitted["cells"]`` in place; return a list
    of ``(arch, slug, shape_point, prior_commit_or_None)`` tuples
    where ``prior_commit_or_None`` is ``None`` for a no-op
    (already at target HEAD)."""
    refreshed: list[tuple[str, str, str, str | None]] = []
    cells = label_admitted.get("cells", [])
    if not isinstance(cells, list):
        raise ValueError(
            "refresh_admitted_label_source_commit: "
            "label_admitted.json's `cells` is not a list"
        )
    # Codex R54 code-review P2: a dict keyed only on
    # (arch, slug, shape_point) collapses historical cells with
    # the same identity tuple but different source_commits. The
    # last-write-wins behavior would leave the intended HEAD cell
    # stale OR corrupt historical provenance. Build a multimap +
    # detect ambiguity before mutating; the operator must
    # disambiguate (deduplicate the admitted file first) before
    # rerunning. The fail-closed behavior is per
    # BL-20260509-strict-admission-contract-first: reject
    # ambiguity at the admission boundary rather than silently
    # last-write.
    keyed_multimap: dict[tuple[str, str, str], list[dict]] = {}
    for c in cells:
        if not isinstance(c, dict):
            continue
        key = (
            str(c.get("arch")),
            str(c.get("slug")),
            str(c.get("shape_point")),
        )
        keyed_multimap.setdefault(key, []).append(c)
    for arch, slug, shape_point in cells_to_refresh:
        key = (arch, slug, shape_point)
        if key not in keyed_multimap:
            raise ValueError(
                f"refresh_admitted_label_source_commit: cell "
                f"{arch}/{slug}/{shape_point} not found in "
                f"label_admitted.json"
            )
        matches = keyed_multimap[key]
        if len(matches) > 1:
            conflicting_commits = sorted(
                {
                    str(m.get("source_commit", "<missing>"))
                    for m in matches
                }
            )
            raise ValueError(
                f"refresh_admitted_label_source_commit: cell "
                f"{arch}/{slug}/{shape_point} has "
                f"{len(matches)} records in label_admitted.json "
                f"with conflicting source_commits "
                f"{conflicting_commits!r}; the dict-based lookup "
                f"would collapse them via last-write-wins. "
                f"Disambiguate by deduplicating the admitted "
                f"file first OR by including source_commit / "
                f"binary identity in the cell selector."
            )
        cell = matches[0]
        prior = str(cell.get("source_commit", ""))
        if not prior:
            raise ValueError(
                f"refresh_admitted_label_source_commit: cell "
                f"{arch}/{slug}/{shape_point} has empty "
                f"source_commit"
            )
        if prior == target_head:
            refreshed.append((arch, slug, shape_point, None))
            continue
        # Audit-annotation discipline: every non-no-op rewrite must
        # leave the immediate prior value greppable.
        # `source_commit_override_from` records the FIRST prior
        # value across the cell's lifetime (the original drift
        # origin); set once on the first-ever refresh, preserved
        # across subsequent refreshes so a grep for the original
        # drift origin still succeeds.
        # `source_commit_refresh_history` records the IMMEDIATE prior
        # of each non-no-op refresh as a list; appended on every
        # refresh so the audit trail captures the full sequence of
        # restamps.
        # Validate existing source_commit_refresh_history shape BEFORE
        # any mutation, so a malformed-history ValueError leaves the
        # cell byte-identical to its pre-call state. A library caller
        # that catches the exception and reuses the cell object must
        # not observe a half-mutated cell (e.g., source_commit_override_from
        # injected but the rewrite aborted).
        if "source_commit_refresh_history" in cell:
            existing_history = cell["source_commit_refresh_history"]
            # Fail-closed on malformed existing history rather than
            # silently dropping it: an audit-annotation field whose
            # type does not match the schema (or whose entries are
            # not strings) signals corruption upstream. Refusing to
            # rewrite preserves the original drift trail for an
            # operator to investigate.
            if not isinstance(existing_history, list):
                raise ValueError(
                    f"refresh_admitted_label_source_commit: cell "
                    f"{arch}/{slug}/{shape_point} has malformed "
                    f"source_commit_refresh_history of type "
                    f"{type(existing_history).__name__}; expected list"
                )
            for idx, entry in enumerate(existing_history):
                if not isinstance(entry, str):
                    raise ValueError(
                        f"refresh_admitted_label_source_commit: "
                        f"cell {arch}/{slug}/{shape_point} has "
                        f"source_commit_refresh_history[{idx}] of "
                        f"type {type(entry).__name__}; expected str"
                    )
            history: list[str] = existing_history
        else:
            history = []
        # All validation passed; safe to mutate.
        if "source_commit_override_from" not in cell:
            cell["source_commit_override_from"] = prior
        history.append(prior)
        cell["source_commit_refresh_history"] = history
        cell["source_commit"] = target_head
        refreshed.append((arch, slug, shape_point, prior))
    return refreshed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh source_commit on selected label_admitted.json "
            "records to current git HEAD per BL-20260508 post-"
            "commit-recapture-discipline. Preserves every other "
            "field byte-identical so the schema validator still "
            "admits the rewritten file."
        ),
    )
    parser.add_argument(
        "--label-admitted",
        type=Path,
        default=_REPO_ROOT / "data" / "local_labels" / "label_admitted.json",
        help="Path to label_admitted.json.",
    )
    parser.add_argument(
        "--cells",
        type=str,
        required=True,
        help=(
            "Comma-separated <arch>/<slug>/<shape_point> tuples. "
            "Example: sm_86/carveout_sensitivity_probe/trace_l2_accuracy_xl,"
            "sm_86/working_set_capacity_sweep/trace_l2_accuracy_xl"
        ),
    )
    parser.add_argument(
        "--repo-root", type=Path, default=_REPO_ROOT,
        help="Repository root.",
    )
    args = parser.parse_args(argv)

    try:
        target_head = _git_head(args.repo_root)
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(
            f"refresh_admitted_label_source_commit: "
            f"`git rev-parse HEAD` failed: {exc}\n"
        )
        return 2

    cells: list[tuple[str, str, str]] = []
    for entry in args.cells.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split("/")
        if len(parts) != 3:
            sys.stderr.write(
                f"refresh_admitted_label_source_commit: invalid "
                f"cell {entry!r}; expected <arch>/<slug>/<shape>\n"
            )
            return 2
        cells.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))

    if not args.label_admitted.is_file():
        sys.stderr.write(
            f"refresh_admitted_label_source_commit: "
            f"{args.label_admitted} not found\n"
        )
        return 2
    with args.label_admitted.open(encoding="utf-8") as fh:
        payload = json.load(fh)

    try:
        refreshed = _refresh_cells(
            label_admitted=payload,
            cells_to_refresh=cells,
            target_head=target_head,
        )
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    # Strict re-validation against the schema before writing.
    if jsonschema is not None and _SCHEMA_PATH.is_file():
        with _SCHEMA_PATH.open(encoding="utf-8") as fh:
            schema = json.load(fh)
        try:
            jsonschema.validate(instance=payload, schema=schema)
        except jsonschema.ValidationError as exc:
            sys.stderr.write(
                f"refresh_admitted_label_source_commit: rewritten "
                f"label_admitted.json fails schema validation: "
                f"{exc.message}\n"
            )
            return 2

    with args.label_admitted.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")

    sys.stderr.write(
        f"refresh_admitted_label_source_commit: target_head="
        f"{target_head}\n"
    )
    for arch, slug, shape_point, prior in refreshed:
        if prior is None:
            sys.stderr.write(
                f"  no-op (already at HEAD): {arch}/{slug}/{shape_point}\n"
            )
        else:
            sys.stderr.write(
                f"  refreshed (was {prior}): "
                f"{arch}/{slug}/{shape_point}\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
