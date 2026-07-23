"""Deterministic best-config label selector for the local-label
sweep tree.

Codex's external review of the prior round flagged that the root
``data/local_labels/label_admitted.json`` corpus used by the
comparison-runner orchestrator was NOT a produced artifact of the
committed sweep: the sub-sweep's own top-level
``label_admitted.json`` (the global-stddev path) admitted only a
subset of the cells admitted under at least one per-config
aggregator output. The lifted "best per config" subset that
populated the comparison bundle was hand-consolidated; nothing on
disk recorded which per-cell-per-config record was chosen for
each cell, so the corpus could not be re-derived from the
committed artifacts alone.

This module closes that gap. It walks a sub-sweep tree of the
shape::

    <sub_sweep>/
      verdict-table.csv
      cell=<arch>__<slug>__<shape_point>/
        cfg=warmup<N>_cache_<C>/
          _aggregator_out/
            label_admitted.json
            label_excluded.json
            label_repeatability.json

reads each per-cell-per-config admitted record, and for each
``(arch, slug, shape_point)`` cell selects ONE record using the
pinned tie-break order
``(stddev_l2_sector, stddev_l1tex_sector, warmup_count,
cache_control, source_relpath)`` ascending. It emits two
artifacts:

  - ``selected_label_admitted.json`` — the chosen admitted records
    in the same shape as the per-config aggregator's output (so
    downstream consumers stay byte-compatible).
  - ``selected_config_manifest.json`` — a per-cell record of the
    chosen ``(warmup_count, cache_control)`` config plus the
    relative path AND sha256 of the source
    ``_aggregator_out/label_admitted.json`` file. This is the
    audit trail.

Both outputs are byte-deterministic for the same input tree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]

# The shared build-variant identity normalizer is a dependency-free leaf module
# beside this script; a sibling import works both as a script and under the
# tools/local_labels test harness. Using the ONE normalizer keeps this selector
# in lockstep with the aggregator / driver / staging / validator (no re-inlined
# copy that could drift).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from variant_identity import variant_identity  # noqa: E402


def _canonical_variant_id(*, register_cap, variant_id, context: str) -> str:
    """Normalize an optional ``(register_cap, variant_id)`` pair to its canonical
    ``variant_id`` (``""`` for a base cell) via the shared normalizer.

    ``register_cap`` may be an int (an admitted-cell JSON payload) or a decimal
    string / ``""`` (a verdict-table CSV column); ``variant_id`` is a str / ``""``.
    A strict pre-check fronts the tolerate-absence normalizer: a present-but-
    malformed value ERRORs (it is a declared-but-invalid identity, not an absent
    one). Raises ``ValueError`` on a malformed / noncanonical / inconsistent pair;
    ``context`` names the source for the diagnostic."""
    record: dict = {}
    if register_cap is not None and register_cap != "":
        if isinstance(register_cap, bool):
            raise ValueError(
                f"{context}: register_cap is a bool, not an int: "
                f"{register_cap!r}"
            )
        if isinstance(register_cap, int):
            record["register_cap"] = register_cap
        elif isinstance(register_cap, str):
            try:
                record["register_cap"] = int(register_cap)
            except ValueError:
                raise ValueError(
                    f"{context}: register_cap {register_cap!r} is not an integer"
                )
        else:
            raise ValueError(
                f"{context}: register_cap has unexpected type "
                f"{type(register_cap).__name__}"
            )
    if variant_id is not None and variant_id != "":
        if not isinstance(variant_id, str):
            raise ValueError(
                f"{context}: variant_id is not a string: {variant_id!r}"
            )
        record["variant_id"] = variant_id
    vid, _cap, err = variant_identity(record)
    if err is not None:
        raise ValueError(
            f"{context}: inconsistent build-variant identity: {err}"
        )
    return vid


def _cell_variant_id(cell: dict, context: str) -> str:
    """Canonical variant_id for an admitted-cell payload (``register_cap`` int /
    ``variant_id`` str), or ``""`` for a base cell."""
    return _canonical_variant_id(
        register_cap=cell.get("register_cap"),
        variant_id=cell.get("variant_id"),
        context=context,
    )


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _stddev_or_inf(value) -> float:
    """Return a float suitable for the selector's tie-break key.

    The admitted-cell schema allows ``stddev_l2_sector`` and
    ``stddev_l1tex_sector`` to be JSON null when the underlying
    metric is absent for a cell (e.g., microbench-only kernels
    that don't exercise L1). Treat null as ``float("inf")`` so the
    sort key is total — a cell with a measured stddev always
    wins over a null-stddev cell on the same tie level."""
    if value is None:
        return float("inf")
    return float(value)


def _nullable_float(value) -> float | None:
    """Preserve the JSON null shape on the manifest emission path.

    When the upstream per-config aggregator emitted ``null`` for
    a stddev field, the manifest-side record keeps the same
    null so downstream consumers that aggregate the manifest see
    the same shape they would see in the per-config aggregator
    output. Non-null values are coerced to ``float`` for
    deterministic JSON serialization."""
    if value is None:
        return None
    return float(value)


_VERDICT_REQUIRED_COLUMNS = (
    "cell_id", "arch", "slug", "shape_point",
    "warmup_count", "cache_control", "admitted",
)


def _read_verdict_table_admitted(
    sub_sweep_dir: Path,
) -> dict[tuple[str, int, str], dict]:
    """Read ``verdict-table.csv`` and return a deterministic map
    keyed by ``(cell_id, warmup_count, cache_control)`` whose
    values are the row's identity dict
    ``{"arch": ..., "slug": ..., "shape_point": ...}``.

    The selector REQUIRES this committed source-of-truth: a
    per-config ``_aggregator_out/label_admitted.json`` directory
    that exists on disk but is NOT marked admitted in the verdict
    table is a rogue / stale artifact and MUST NOT influence the
    selection. The returned identity dict is consumed by the
    payload-binding pass in ``_walk_per_config_admissions`` to
    enforce that every emitted cell's
    ``(arch, slug, shape_point)`` matches the admitted row.

    Fail-closed on missing required columns; any malformed row
    raises ``ValueError`` rather than silently skip.
    """
    csv_path = sub_sweep_dir / "verdict-table.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"select_best_config_labels: missing verdict-table.csv "
            f"at {csv_path}; the selector requires the committed "
            f"sweep verdict table to filter out rogue per-config "
            f"aggregator outputs"
        )
    out: dict[tuple[str, int, str], dict] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(
                f"select_best_config_labels: verdict-table.csv "
                f"at {csv_path} has no header row"
            )
        missing = [
            c for c in _VERDICT_REQUIRED_COLUMNS
            if c not in reader.fieldnames
        ]
        if missing:
            raise ValueError(
                f"select_best_config_labels: verdict-table.csv "
                f"at {csv_path} is missing required column(s) "
                f"{missing!r}; cannot bind verdict-row identity "
                f"to per-config payload identity without them"
            )
        for row in reader:
            if str(row.get("admitted", "")).strip().lower() != "true":
                continue
            try:
                wc = int(row["warmup_count"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"select_best_config_labels: verdict-table.csv "
                    f"at {csv_path} has malformed warmup_count "
                    f"in row {row!r}: {exc}"
                )
            cc = str(row.get("cache_control", "")).strip()
            cell_id = str(row.get("cell_id", "")).strip()
            arch = str(row.get("arch", "")).strip()
            slug = str(row.get("slug", "")).strip()
            sp = str(row.get("shape_point", "")).strip()
            if not all([cell_id, cc, arch, slug, sp]):
                raise ValueError(
                    f"select_best_config_labels: verdict-table.csv "
                    f"at {csv_path} has empty identity column(s) "
                    f"in row {row!r}"
                )
            # Optional register-cap build-variant: a variant row carries
            # register_cap / variant_id columns and a 4-part cell_id
            # (arch/slug/shape/register_cap_<cap>). Reconstruct with the
            # canonical variant suffix (via the shared normalizer) so a variant
            # row's cell_id binds correctly; base rows reconstruct the
            # historical 3-part id byte-identically.
            try:
                variant_id = _canonical_variant_id(
                    register_cap=row.get("register_cap", ""),
                    variant_id=row.get("variant_id", ""),
                    context=f"verdict-table.csv at {csv_path} row {row!r}",
                )
            except ValueError as exc:
                raise ValueError(f"select_best_config_labels: {exc}")
            base_cell_id = f"{arch}/{slug}/{sp}"
            row_cell_id = (
                f"{base_cell_id}/{variant_id}" if variant_id else base_cell_id
            )
            if row_cell_id != cell_id:
                raise ValueError(
                    f"select_best_config_labels: verdict-table.csv "
                    f"at {csv_path} has internally inconsistent row: "
                    f"cell_id={cell_id!r} but reconstructed "
                    f"arch/slug/shape_point[/variant_id]="
                    f"{row_cell_id!r}; the verdict row's identity "
                    f"columns must agree with cell_id before any "
                    f"per-config payload can be selected"
                )
            out[(cell_id, wc, cc)] = {
                "arch": arch, "slug": slug, "shape_point": sp,
                "variant_id": variant_id,
            }
    return out


def _walk_per_config_admissions(
    sub_sweep_dir: Path,
) -> list[tuple[dict, Path, int, str]]:
    """Walk the sub-sweep tree and return one tuple per admitted
    per-cell-per-config record that ALSO appears as
    ``admitted=true`` in the sweep's ``verdict-table.csv`` AND
    whose payload's ``(arch, slug, shape_point)`` matches the
    admitted row's identity:
      (cell_record, source_path, warmup_count, cache_control).

    Filters out:
      - per-config ``_aggregator_out/label_admitted.json`` whose
        ``(cell_id, warmup_count, cache_control)`` is not present
        in the verdict-table admitted map (rogue / stale
        directory);
      - per-config payloads whose ``cells[*].(arch, slug,
        shape_point)`` does NOT match the admitted verdict row's
        identity (stale / foreign payload under an admitted
        directory) — raises ``ValueError`` rather than silently
        drop.
    """
    admitted_map = _read_verdict_table_admitted(sub_sweep_dir)
    out: list[tuple[dict, Path, int, str]] = []
    for cell_dir in sorted(sub_sweep_dir.glob("cell=*")):
        if not cell_dir.is_dir():
            continue
        cell_id_from_dir = (
            cell_dir.name[len("cell="):].replace("__", "/")
        )
        for cfg_dir in sorted(cell_dir.glob("cfg=warmup*_cache_*")):
            agg = cfg_dir / "_aggregator_out" / "label_admitted.json"
            if not agg.is_file():
                continue
            cfg_name = cfg_dir.name  # "cfg=warmup5_cache_all"
            tail = cfg_name[len("cfg=warmup"):]
            wc_str, _, cc = tail.partition("_cache_")
            try:
                warmup_count = int(wc_str)
            except ValueError:
                continue
            row_identity = admitted_map.get(
                (cell_id_from_dir, warmup_count, cc)
            )
            if row_identity is None:
                # Rogue / stale per-config aggregator output —
                # the sweep's verdict-table did NOT record this
                # config as admitted for this cell.
                continue
            with open(agg) as f:
                payload = json.load(f)
            # Bind verdict-row identity to per-config payload:
            # every cell's (arch, slug, shape_point) MUST equal
            # the admitted row's identity AND the directory
            # identity. A mismatch is a stale/foreign payload
            # under an otherwise-admitted directory — fail-closed.
            for cell in payload.get("cells", []):
                cell_variant_id = _cell_variant_id(
                    cell, context=f"per-config payload at {agg}"
                )
                cell_identity = (
                    str(cell.get("arch", "")),
                    str(cell.get("slug", "")),
                    str(cell.get("shape_point", "")),
                    cell_variant_id,
                )
                expected_identity = (
                    row_identity["arch"], row_identity["slug"],
                    row_identity["shape_point"],
                    row_identity.get("variant_id", ""),
                )
                if cell_identity != expected_identity:
                    raise ValueError(
                        f"select_best_config_labels: per-config "
                        f"payload at {agg} has cell with "
                        f"(arch, slug, shape_point, variant_id)="
                        f"{cell_identity!r} but verdict-table.csv "
                        f"admits "
                        f"{expected_identity!r} for "
                        f"(cell_id={cell_id_from_dir!r}, "
                        f"warmup_count={warmup_count}, "
                        f"cache_control={cc!r}); the per-config "
                        f"_aggregator_out payload is stale or "
                        f"foreign relative to the committed "
                        f"verdict-table identity"
                    )
                out.append((cell, agg, warmup_count, cc))
    return out


def _materialize_root_corpus(
    *,
    selected_cells: list[dict],
    existing_labels_path: Path,
    historical_source_commit: str,
    expected_historical_count: int,
    out_root_label_admitted_path: Path,
    schema_path: Path | None = None,
) -> dict:
    """Materialize the operator-facing root label corpus by
    concatenating the selected cells with the preserved historical
    cells from ``existing_labels_path``. Validates the result and
    writes it to ``out_root_label_admitted_path``."""
    with open(existing_labels_path) as f:
        existing = json.load(f)
    historical_cells = [
        c for c in existing.get("cells", [])
        if c.get("source_commit") == historical_source_commit
    ]
    if len(historical_cells) != expected_historical_count:
        raise ValueError(
            f"select_best_config_labels: historical-count mismatch "
            f"for source_commit={historical_source_commit!r}: "
            f"--existing-labels {existing_labels_path} contains "
            f"{len(historical_cells)} cells, "
            f"--expected-historical-count={expected_historical_count}"
        )
    # Reject duplicate (arch, slug, shape_point, source_commit)
    # tuples between selected and historical sets. By design,
    # selected cells carry the round's HEAD source_commit and
    # historical cells carry --preserve-historical-source-commit,
    # which the round contract requires to differ (the historical
    # cells are SUPERSEDED-by-history under the supersession-
    # bookkeeping discipline). A same-4-tuple overlap means the
    # operator passed --preserve-historical-source-commit equal to
    # the round's HEAD source_commit, which would silently produce
    # two cells for the same join key.
    def _dup_key(c: dict) -> tuple:
        # Mixed-arity duplicate key: base cells keep the historical 4-tuple
        # (arch, slug, shape_point, source_commit); a build variant widens with
        # its canonical variant_id so sibling caps are NOT false duplicates.
        vid = _cell_variant_id(c, context="materialize root corpus cell")
        base = (c["arch"], c["slug"], c["shape_point"], c["source_commit"])
        return base if not vid else (
            c["arch"], c["slug"], c["shape_point"], vid, c["source_commit"]
        )

    selected_keys = {_dup_key(c) for c in selected_cells}
    historical_keys = {_dup_key(c) for c in historical_cells}
    overlap = selected_keys & historical_keys
    if overlap:
        raise ValueError(
            f"select_best_config_labels: duplicate "
            f"(arch, slug, shape_point, source_commit) tuples "
            f"between selected and historical sets: "
            f"{sorted(overlap)!r}"
        )
    root = {
        "schema_version": 5,
        "cells": list(selected_cells) + list(historical_cells),
    }
    if schema_path is None:
        schema_path = (
            _REPO_ROOT / "workloads" / "simbench_bw" / "schemas"
            / "local_label_admitted.schema.json"
        )
    if schema_path.is_file():
        try:
            import jsonschema  # type: ignore[import-not-found]
        except ImportError:
            jsonschema = None  # type: ignore[assignment]
        if jsonschema is not None:
            with open(schema_path) as f:
                schema = json.load(f)
            try:
                jsonschema.validate(instance=root, schema=schema)
            except jsonschema.ValidationError as exc:
                raise ValueError(
                    f"select_best_config_labels: materialized "
                    f"root corpus failed schema validation against "
                    f"{schema_path}: {exc.message}"
                )
    out_root_label_admitted_path.parent.mkdir(
        parents=True, exist_ok=True,
    )
    with open(out_root_label_admitted_path, "w") as f:
        json.dump(root, f, indent=2, sort_keys=True)
    return root


def select_best_config_labels(
    *,
    sub_sweep_dir: Path,
    out_labels_path: Path,
    out_config_manifest_path: Path,
) -> tuple[dict, dict]:
    """Read every admitted per-cell-per-config record under
    ``sub_sweep_dir`` and select one per ``(arch, slug,
    shape_point)`` by the pinned tie-break order. Writes the two
    output artifacts and also returns them as dicts."""
    rows = _walk_per_config_admissions(sub_sweep_dir)

    grouped: dict[tuple, list] = {}
    for cell, src_path, warmup, cc in rows:
        # Mixed-arity selection key: base cells keep the historical 3-tuple
        # (byte-identical selection + grouping); a build variant widens to a
        # 4-tuple so sibling caps of the same (arch, slug, shape) are selected
        # as DISTINCT cells instead of collapsing to one.
        variant_id = _cell_variant_id(
            cell, context=f"selected cell from {src_path}"
        )
        base_key = (
            str(cell["arch"]),
            str(cell["slug"]),
            str(cell["shape_point"]),
        )
        key = base_key if not variant_id else base_key + (variant_id,)
        try:
            relpath = src_path.resolve().relative_to(
                sub_sweep_dir.resolve()
            )
        except ValueError:
            relpath = src_path
        grouped.setdefault(key, []).append({
            "cell": cell,
            "warmup_count": warmup,
            "cache_control": cc,
            "source_path": src_path,
            "source_relpath": str(relpath),
        })

    selected_per_cell: list[dict] = []
    selected_cells: list[dict] = []
    for key in sorted(grouped.keys()):
        candidates = grouped[key]
        # Tie-break: (stddev_l2, stddev_l1tex, warmup_count,
        # cache_control, source_relpath) ascending. Strings sort
        # lexicographically — "all" < "none" so cache_control is
        # naturally a stable secondary on identical numerics.
        # The stddev fields are schema-allowed to be JSON null when
        # the underlying metric is absent (e.g., microbench-only
        # kernels that don't exercise L1); _stddev_or_inf treats
        # null as float("inf") so a cell with a measured stddev
        # is always preferred over a null-stddev cell on the same
        # tie level.
        candidates.sort(
            key=lambda r: (
                _stddev_or_inf(r["cell"].get("stddev_l2_sector")),
                _stddev_or_inf(r["cell"].get("stddev_l1tex_sector")),
                int(r["warmup_count"]),
                str(r["cache_control"]),
                str(r["source_relpath"]),
            )
        )
        chosen = candidates[0]
        cell_payload = dict(chosen["cell"])
        selected_cells.append(cell_payload)
        per_cell = {
            "arch": str(cell_payload["arch"]),
            "slug": str(cell_payload["slug"]),
            "shape_point": str(cell_payload["shape_point"]),
            "warmup_count": int(chosen["warmup_count"]),
            "cache_control": str(chosen["cache_control"]),
            "source_relpath": str(chosen["source_relpath"]),
            "source_sha256": _file_sha256(chosen["source_path"]),
            # Manifest emission preserves the null shape verbatim
            # when the underlying metric was absent; downstream
            # consumers that aggregate the manifest see the same
            # null they would see in the per-config aggregator
            # output.
            "stddev_l2_sector": _nullable_float(
                cell_payload.get("stddev_l2_sector"),
            ),
            "stddev_l1tex_sector": _nullable_float(
                cell_payload.get("stddev_l1tex_sector"),
            ),
        }
        # Carry the optional build-variant identity into the audit manifest so
        # the chosen-config record is variant-distinct (a base cell omits both).
        if cell_payload.get("register_cap") is not None:
            per_cell["register_cap"] = cell_payload["register_cap"]
        if cell_payload.get("variant_id") is not None:
            per_cell["variant_id"] = cell_payload["variant_id"]
        selected_per_cell.append(per_cell)

    selected_label_admitted = {
        "schema_version": 5,
        "cells": selected_cells,
    }
    selected_config_manifest = {
        "schema_version": 1,
        "sub_sweep_relpath": _try_relpath(sub_sweep_dir),
        "selected_per_cell": selected_per_cell,
    }

    out_labels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_labels_path, "w") as f:
        json.dump(
            selected_label_admitted, f, indent=2, sort_keys=True,
        )
    out_config_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_config_manifest_path, "w") as f:
        json.dump(
            selected_config_manifest, f, indent=2, sort_keys=True,
        )
    return selected_label_admitted, selected_config_manifest


def _try_relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Select one admitted record per (arch, slug, "
            "shape_point) from a sub-sweep tree's per-cell-per-"
            "config aggregator outputs, writing both the chosen "
            "label corpus and an audit-trail config manifest."
        ),
    )
    parser.add_argument(
        "--sub-sweep",
        type=Path,
        required=True,
        help=(
            "Path to the sub-sweep dir (the dir containing "
            "verdict-table.csv + cell=*/cfg=*/_aggregator_out/"
            "label_admitted.json)."
        ),
    )
    parser.add_argument(
        "--out-selected",
        type=Path,
        required=True,
        help="Path to write selected_label_admitted.json.",
    )
    parser.add_argument(
        "--out-config-manifest",
        type=Path,
        required=True,
        help="Path to write selected_config_manifest.json.",
    )
    parser.add_argument(
        "--existing-labels",
        type=Path,
        default=None,
        help=(
            "Optional: existing operator-facing label_admitted.json "
            "containing historical cells to preserve. When set, "
            "must be paired with --preserve-historical-source-commit, "
            "--expected-historical-count, and "
            "--out-root-label-admitted."
        ),
    )
    parser.add_argument(
        "--preserve-historical-source-commit",
        default=None,
        help=(
            "Source-commit SHA whose cells in --existing-labels are "
            "preserved into the materialized root corpus under "
            "SUPERSEDED-by-history semantics."
        ),
    )
    parser.add_argument(
        "--expected-historical-count",
        type=int,
        default=None,
        help=(
            "Expected count of cells at "
            "--preserve-historical-source-commit in --existing-labels. "
            "Mismatch -> nonzero exit with typed diagnostic."
        ),
    )
    parser.add_argument(
        "--out-root-label-admitted",
        type=Path,
        default=None,
        help=(
            "Path to write the materialized root corpus "
            "(selected_cells + preserved_historical_cells). "
            "Required when the historical-merge flags are set."
        ),
    )
    args = parser.parse_args(argv)

    materialize_args = (
        args.existing_labels,
        args.preserve_historical_source_commit,
        args.expected_historical_count,
        args.out_root_label_admitted,
    )
    materialize_present = [a is not None for a in materialize_args]
    if any(materialize_present) and not all(materialize_present):
        print(
            "select_best_config_labels: --existing-labels, "
            "--preserve-historical-source-commit, "
            "--expected-historical-count, and "
            "--out-root-label-admitted MUST be provided together "
            "to materialize the root corpus",
            file=sys.stderr,
        )
        return 1

    try:
        selected, manifest = select_best_config_labels(
            sub_sweep_dir=args.sub_sweep,
            out_labels_path=args.out_selected,
            out_config_manifest_path=args.out_config_manifest,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(
            f"select_best_config_labels: {exc}",
            file=sys.stderr,
        )
        return 1

    if all(materialize_present):
        try:
            root = _materialize_root_corpus(
                selected_cells=selected["cells"],
                existing_labels_path=args.existing_labels,
                historical_source_commit=(
                    args.preserve_historical_source_commit
                ),
                expected_historical_count=(
                    args.expected_historical_count
                ),
                out_root_label_admitted_path=(
                    args.out_root_label_admitted
                ),
            )
        except (FileNotFoundError, ValueError) as exc:
            print(
                f"select_best_config_labels: {exc}",
                file=sys.stderr,
            )
            return 1
        print(
            f"  materialized root corpus: "
            f"{_try_relpath(args.out_root_label_admitted)} "
            f"(cells={len(root['cells'])})",
            file=sys.stderr,
        )

    print(
        f"select_best_config_labels: selected "
        f"{len(selected['cells'])} cells from "
        f"{_try_relpath(args.sub_sweep)}",
        file=sys.stderr,
    )
    print(
        f"  selected_label_admitted: "
        f"{_try_relpath(args.out_selected)}",
        file=sys.stderr,
    )
    print(
        f"  selected_config_manifest: "
        f"{_try_relpath(args.out_config_manifest)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
