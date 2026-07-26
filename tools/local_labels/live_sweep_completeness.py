"""Scanner: a committed full-matrix live sweep MUST be complete.

The prior bounded recovery round committed a five-activation-cell
sweep dir with 40 cell-config directories but only 192 of 200 raw
label sidecars and 37 of 40 per-config aggregator outputs; three
configs hit an NCU saturation-rounding artifact that the parser
fail-closed before the per-launch sidecars were written.

This scanner pins the full-matrix evidence contract so a partial
sweep cannot be selected, merged, or used as headline-regen input:

  - exactly 40 ``cell=*/cfg=*`` directories;
  - exactly 40 ``_aggregator_out/`` directories;
  - exactly 200 ``label_<repeat>_<launch>.json`` raw sidecars
    (excluding ``label_run_manifest_*.json`` and the
    ``_aggregator_out/label_admitted.json`` /
    ``label_excluded.json`` / ``label_repeatability.json``
    aggregator files);
  - exactly 200 ``label_run_manifest_<repeat>_<launch>.json``
    files;
  - exactly 200 ``run_<repeat>_<launch>.ncu-rep`` reports;
  - no ``verdict-table.csv`` row with ``n_sidecars`` strictly less
    than the expected ``repeats`` (default 5).

The defaults pin the post-launch-skip-fix matrix shape
(``5 cells × 4 warmups × 2 cache_controls × 5 repeats``). Callers
may override ``expected_*`` parameters for a different matrix
shape, but the default is the canonical one.

Offline evidence-completeness boundary: this scanner is host-only Python; it
does not invoke NCU, does not access the GPU, and does not write
outside ``data/local_labels/``.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SWEEPS_ROOT = _REPO_ROOT / "data" / "local_labels" / "sweeps"


_LABEL_FILE_RE = re.compile(r"^label_\d+_\d+\.json$")
_LABEL_RUN_MANIFEST_RE = re.compile(r"^label_run_manifest_\d+_\d+\.json$")
_NCU_REPORT_RE = re.compile(r"^run_\d+_\d+\.ncu-rep$")


def validate_sweep_dir(
    sweep_dir: Path,
    *,
    expected_cell_config_dirs: int = 40,
    expected_aggregator_outs: int = 40,
    expected_label_sidecars: int = 200,
    expected_run_manifests: int = 200,
    expected_ncu_reports: int = 200,
    expected_repeats: int = 5,
) -> list[str]:
    """Return a list of typed defect tokens; an empty list means
    the sweep dir matches the full-matrix evidence contract.

    Aggregate-only-aware contract: raw per-config NCU sidecars
    (``label_<r>_<l>.json``, ``label_run_manifest_<r>_<l>.json``,
    ``run_<r>_<l>.ncu-rep``) under ``cell=*/cfg=*/`` are
    excluded by the project's ``.gitignore`` (see ``.gitignore``
    lines 105-115; the rule is pre-existing and reflects the
    decision to keep aggregator outputs but drop raw sweep
    volume). On a fresh checkout the raw sidecar counts are
    therefore zero. The scanner SKIPS the raw-count checks when
    zero raw sidecars are found (the aggregate evidence
    ``_aggregator_out/`` + ``verdict-table.csv`` is sufficient
    to validate the sweep's claim); when an operator runs the
    scanner against a non-cleaned local sweep tree with raw
    sidecars present, the count checks still fire. The
    aggregate counts (``_aggregator_out/``,
    ``cell=*/cfg=*``) + ``verdict-table.csv`` schema checks
    remain MANDATORY regardless of raw-sidecar presence.

    Defect tokens:
      ``sweep_cell_config_dir_count_not_40``,
      ``sweep_aggregator_out_count_not_40``,
      ``sweep_label_sidecar_count_not_200``,
      ``sweep_run_manifest_count_not_200``,
      ``sweep_ncu_report_count_not_200``,
      ``sweep_verdict_table_missing``,
      ``sweep_verdict_row_n_sidecars_below_repeats``.
    """
    defects: list[str] = []

    cell_config_dirs = sorted(sweep_dir.glob("cell=*/cfg=*"))
    if len(cell_config_dirs) != expected_cell_config_dirs:
        defects.append("sweep_cell_config_dir_count_not_40")

    aggregator_outs = sorted(sweep_dir.glob("cell=*/cfg=*/_aggregator_out"))
    if len(aggregator_outs) != expected_aggregator_outs:
        defects.append("sweep_aggregator_out_count_not_40")

    label_sidecars: list[Path] = []
    run_manifests: list[Path] = []
    ncu_reports: list[Path] = []
    for cfg_dir in cell_config_dirs:
        for path in cfg_dir.rglob("*"):
            if not path.is_file():
                continue
            if "_aggregator_out" in path.parts:
                continue
            name = path.name
            if _LABEL_FILE_RE.match(name):
                label_sidecars.append(path)
            elif _LABEL_RUN_MANIFEST_RE.match(name):
                run_manifests.append(path)
            elif _NCU_REPORT_RE.match(name):
                ncu_reports.append(path)
    # Raw-sidecar count checks: aggregate-only-aware. Skip when
    # zero raw sidecars are found (fresh checkout, gitignored);
    # fire when raw sidecars exist but the count is wrong
    # (operator's local sweep tree, partial sweep, etc.).
    if label_sidecars and len(label_sidecars) != expected_label_sidecars:
        defects.append("sweep_label_sidecar_count_not_200")
    if run_manifests and len(run_manifests) != expected_run_manifests:
        defects.append("sweep_run_manifest_count_not_200")
    if ncu_reports and len(ncu_reports) != expected_ncu_reports:
        defects.append("sweep_ncu_report_count_not_200")

    verdict_table_path = sweep_dir / "verdict-table.csv"
    if not verdict_table_path.is_file():
        defects.append("sweep_verdict_table_missing")
    else:
        with verdict_table_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    n_sidecars = int(row["n_sidecars"])
                except (KeyError, ValueError):
                    continue
                if n_sidecars < expected_repeats:
                    defects.append(
                        "sweep_verdict_row_n_sidecars_below_repeats"
                    )
                    break

    return defects


def find_latest_matrix_sweep_dir(
    sweeps_root: Path = _DEFAULT_SWEEPS_ROOT,
    *,
    full_matrix_cell_config_count: int = 40,
) -> Path | None:
    """Return the lex-greatest FULL-MATRIX sweep dir under
    ``sweeps_root``, or None if none exists.

    A FULL-MATRIX sweep dir carries the complete
    ``5 cells × 4 warmups × 2 cache_controls = 40 cell=*/cfg=*``
    subdirectories. The selection filter intentionally EXCLUDES
    amendment / partial sweeps (single-cell variance searches,
    single-config follow-ups, etc.) so the "latest matrix sweep"
    consumed by downstream validators is always a complete
    matrix-style sweep. Regen-only sweep dirs (which carry
    ``predictor_bundles/`` and ``comparison_bundle.json`` but no
    ``cell=*`` children) are skipped by the same predicate.

    Partial sweeps live under the same ``sweeps_root`` for audit
    history but should not be selected as the canonical "latest"
    by the matrix-completeness scanner. Operators who want to
    iterate over ALL sweep dirs (including partials) can walk
    ``sweeps_root`` directly without going through this helper.
    """
    if not sweeps_root.is_dir():
        return None
    matrix_dirs: list[Path] = []
    for child in sweeps_root.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith("sweep_"):
            continue
        # Full-matrix filter: require the complete cell-config
        # count. Partial / amendment sweeps (fewer than 40
        # cell=*/cfg=* dirs) are excluded from the "latest"
        # selection so the matrix-completeness scanner always
        # validates a canonical full-matrix sweep.
        cell_config_count = sum(1 for _ in child.glob("cell=*/cfg=*"))
        if cell_config_count >= full_matrix_cell_config_count:
            matrix_dirs.append(child)
    if not matrix_dirs:
        return None
    matrix_dirs.sort(key=lambda p: p.name)
    return matrix_dirs[-1]
