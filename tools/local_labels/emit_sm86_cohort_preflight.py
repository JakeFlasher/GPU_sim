"""Emit a per-cell preflight matrix for the 44 sm_86 admitted cells.

The matrix is the input data structure that drives the sm_86 cohort
regeneration lane on the originating-arch host (RTX 3090 Ti).
For each cell in ``data/local_labels/label_admitted.json`` whose
``arch == "sm_86"`` it records:

* the cell identity 6-tuple (arch, slug, shape_point, binary_sha256,
  kernel_signature_sha256, source_commit) — directly from the
  committed label
* the owning workload tree (``simbench_bw`` vs ``portable_lcd``)
  classified by source-dir presence at
  ``workloads/<source>/l1/<slug>/`` (simbench_bw) or
  ``workloads/portable_lcd/<slug>/`` (portable_lcd)
* the resolved binary path and whether the binary currently exists
  on disk
* the slug's ``ARCH_GATE`` set parsed from its ``Makefile``; cells
  whose gate excludes sm_86 take a terminal-typed-failure
  classification (no rebuild path available)
* the canonical trace directory under
  ``workloads/<source>/results/<slug>/rtx_3090_ti/sm_86/trace_<shape_point>/``
  and whether the four NVBit-derived artifacts
  (``dynamic_trace.pb``, ``sharing_locality.json``,
  ``hrd_profile/hrd_profile.json``,
  ``sector_stream.jsonl`` + sibling
  ``sector_stream/manifest_pairing_index.json``)
  are present
* any pre-existing predictor bundle under
  ``data/local_labels/sweeps/**/predictor_bundles/*.json`` whose
  ``manifest_pairing_index.{slug, shape_point, binary_sha256,
  source_commit, kernel_signature_sha256}`` 5-tuple binds to the
  label cell
* a closed-enum ``proposed_action`` derived from the missing-piece
  flags and a replayable ``preflight_command`` that the next task in
  the lane will execute

Output: ``data/local_labels/sm86_cohort_preflight.json``.

This module is read-only against the repo; it never invokes NCU,
never modifies driver state, never touches privileged CUPTI
surfaces. It writes a single JSON artifact and exits.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LABEL = _REPO_ROOT / "data" / "local_labels" / "label_admitted.json"
_DEFAULT_OUT = _REPO_ROOT / "data" / "local_labels" / "sm86_cohort_preflight.json"
# Recursive scan root + pattern mirroring the per-cell builder's
# `_enumerate_candidate_bundles` (in
# `tools/local_labels/build_predictor_bundle_for_cell.py`). The
# earlier preflight implementation used two one-level globs which
# silently skipped bundles at depths greater than one; the
# recursive pattern below uses ``Path.rglob`` against the sweeps
# root so any depth of intermediate directory is covered.
_SWEEPS_ROOT = _REPO_ROOT / "data" / "local_labels" / "sweeps"
_BUNDLE_GLOB_PATTERN = "predictor_bundles/*.json"
_PREFLIGHT_SCHEMA_VERSION = 1
_HOST_SKU = "rtx_3090_ti"
_HOST_ARCH = "sm_86"
_ARCH_GATE_PATTERN = re.compile(r"^\s*ARCH_GATE\s*[:?]?=\s*(.+?)\s*$", re.MULTILINE)

_CLOSED_ACTION_ENUM: tuple[str, ...] = (
    "READY",
    "REBUILD_BINARY_LOCAL",
    "REBUILD_TRACE_LOCAL",
    "REGENERATE_SIDECARS_LOCAL",
    "REBUILD_BUNDLE_LOCAL",
    "TERMINAL_TYPED_FAILURE",
)

# Portable-LCD ARCH_GATE alias. The portable_lcd workload tree
# declares ``ARCH_GATE = lcd`` on every slug; the alias expands
# to the LCD base-arch triple ``sm_86 sm_90 sm_120`` per
# ``workloads/portable_lcd/Makefile`` and ``common.mk`` (with a
# compute_80 PTX fallback for forward-compat JIT). Without
# expansion the parser sees a single token ``lcd`` and
# misclassifies every portable_lcd slug as
# ``arch_gate_excludes_sm_86`` — which is the inverse of the
# truth.
_LCD_ALIAS_TOKEN = "lcd"
_LCD_ALIAS_EXPANSION: tuple[str, ...] = ("sm_86", "sm_90", "sm_120")


def _classify_workload(slug: str) -> str | None:
    """Return ``"simbench_bw"`` if the slug source dir lives under
    ``workloads/simbench_bw/l1/<slug>/``, ``"portable_lcd"`` if it
    lives under ``workloads/portable_lcd/<slug>/``, else ``None``.
    """
    simbench = _REPO_ROOT / "workloads" / "simbench_bw" / "l1" / slug
    portable = _REPO_ROOT / "workloads" / "portable_lcd" / slug
    sb = simbench.is_dir()
    pl = portable.is_dir()
    if sb and not pl:
        return "simbench_bw"
    if pl and not sb:
        return "portable_lcd"
    return None


def _slug_source_dir(workload: str, slug: str) -> Path:
    if workload == "simbench_bw":
        return _REPO_ROOT / "workloads" / "simbench_bw" / "l1" / slug
    return _REPO_ROOT / "workloads" / "portable_lcd" / slug


def _binary_path(workload: str, slug: str) -> Path:
    return _slug_source_dir(workload, slug) / f"{slug}_run"


def _trace_dir(workload: str, slug: str, shape_point: str) -> Path:
    return (
        _REPO_ROOT
        / "workloads"
        / workload
        / "results"
        / slug
        / _HOST_SKU
        / _HOST_ARCH
        / f"trace_{shape_point}"
    )


def _read_arch_gate(workload: str, slug: str) -> list[str]:
    """Return the parsed ``ARCH_GATE`` list from the slug's
    ``Makefile``. Empty list when the file is missing or has no
    ARCH_GATE assignment.
    """
    mk = _slug_source_dir(workload, slug) / "Makefile"
    if not mk.is_file():
        return []
    text = mk.read_text()
    match = _ARCH_GATE_PATTERN.search(text)
    if not match:
        return []
    tokens = match.group(1).split()
    cleaned: list[str] = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if tok == _LCD_ALIAS_TOKEN:
            for expansion in _LCD_ALIAS_EXPANSION:
                if expansion not in cleaned:
                    cleaned.append(expansion)
            continue
        if tok not in cleaned:
            cleaned.append(tok)
    return cleaned


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sidecar_paths(trace_dir: Path) -> dict[str, Path]:
    """Producer-side sidecar paths under ``<trace_dir>/``.

    The canonical producers write each sidecar into a per-producer
    SUBDIRECTORY (not the trace root). A naive check that looked
    for root-level ``sharing_locality.json`` /
    ``sector_stream.jsonl`` paths would misreport every
    cell-with-trace as ``sidecars_missing``. The producers actually
    used are:

    * ``tools.sharing_locality.classifier.compute_sharing_locality``
      writes ``<trace_dir>/sharing_locality/sharing_locality.json``.
    * ``tools.hrd_profiler.bundle_from_trace.emit_hrd_bundle``
      writes ``<trace_dir>/hrd_profile/hrd_profile.json``.
    * ``tools.sector_stream.consume_trace`` writes
      ``<trace_dir>/sector_stream/sector_stream.jsonl`` plus the
      sibling identity-binding manifest
      ``<trace_dir>/sector_stream/manifest_pairing_index.json``.

    Keeping the path table in one helper avoids drift if the
    producer convention widens later; downstream callers consume
    the dict.
    """
    return {
        "sharing_locality": (
            trace_dir / "sharing_locality" / "sharing_locality.json"
        ),
        "hrd_profile": trace_dir / "hrd_profile" / "hrd_profile.json",
        "sector_stream": trace_dir / "sector_stream" / "sector_stream.jsonl",
        "sector_stream_mpi": (
            trace_dir / "sector_stream" / "manifest_pairing_index.json"
        ),
    }


def _sidecar_presence(trace_dir: Path) -> dict[str, bool]:
    return {key: path.is_file() for key, path in _sidecar_paths(trace_dir).items()}


def _all_sidecars_present(presence: dict[str, bool]) -> bool:
    return all(presence.values())


def _scan_existing_bundles() -> list[dict[str, Any]]:
    """Walk every committed bundle under
    ``data/local_labels/sweeps/`` recursively
    (``Path.rglob('predictor_bundles/*.json')``) and return their
    parsed payloads with their on-disk paths.

    The recursive enumeration matches the per-cell builder's
    ``_enumerate_candidate_bundles`` (in
    ``tools/local_labels/build_predictor_bundle_for_cell.py``); the
    two surfaces stay in lockstep so a bundle reachable to the
    per-cell cascade is also discoverable by the preflight matrix's
    Stage 1 lookup.
    """
    bundles: list[dict[str, Any]] = []
    if not _SWEEPS_ROOT.is_dir():
        return bundles
    for path in sorted(_SWEEPS_ROOT.rglob(_BUNDLE_GLOB_PATTERN)):
        try:
            with path.open() as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        bundles.append({"path": path, "payload": payload})
    return bundles


def _bundle_identity_5tuple(payload: dict[str, Any]) -> tuple[str, ...] | None:
    mpi = payload.get("manifest_pairing_index")
    if not isinstance(mpi, dict):
        return None
    needed = ("slug", "shape_point", "binary_sha256", "source_commit",
              "kernel_signature_sha256")
    out = []
    for key in needed:
        val = mpi.get(key)
        if not isinstance(val, str) or not val:
            return None
        out.append(val)
    return tuple(out)


def _find_bundle_for_cell(cell: dict[str, Any],
                          bundles: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the first bundle whose MPI 5-tuple matches the cell,
    else ``None``. The match is on
    ``(slug, shape_point, binary_sha256, source_commit,
    kernel_signature_sha256)`` only — producer-side fields
    (``predictor_build_commit``) are out of scope at preflight time.
    """
    cell_identity = (
        cell["slug"], cell["shape_point"], cell["binary_sha256"],
        cell["source_commit"], cell["kernel_signature_sha256"],
    )
    for entry in bundles:
        bundle_id = _bundle_identity_5tuple(entry["payload"])
        if bundle_id is None:
            continue
        if bundle_id == cell_identity:
            return entry
    return None


def _classify_action(
    *,
    arch_gate_excludes: bool,
    binary_present: bool,
    trace_present: bool,
    sidecars_complete: bool,
    bundle_present: bool,
) -> tuple[str, str | None]:
    """Map the missing-piece flags onto the closed
    ``proposed_action`` enum + ``action_typed_reason``.

    The cascade follows the regeneration-lane ordering: rebuild
    binary first, then trace, then sidecars, then bundle. A cell that
    has every piece bound + identity match lands on ``READY``;
    cells whose Makefile gate excludes sm_86 land on
    ``TERMINAL_TYPED_FAILURE`` with reason
    ``arch_gate_excludes_sm_86`` (the rebuild has no replayable
    surface).
    """
    if arch_gate_excludes:
        return "TERMINAL_TYPED_FAILURE", "arch_gate_excludes_sm_86"
    if not binary_present:
        return "REBUILD_BINARY_LOCAL", None
    if not trace_present:
        return "REBUILD_TRACE_LOCAL", None
    if not sidecars_complete:
        return "REGENERATE_SIDECARS_LOCAL", None
    if not bundle_present:
        return "REBUILD_BUNDLE_LOCAL", None
    return "READY", None


def _preflight_command(
    *,
    workload: str,
    slug: str,
    shape_point: str,
    proposed_action: str,
) -> str:
    if proposed_action == "REBUILD_BINARY_LOCAL":
        return (
            f"make -C workloads/{workload}/"
            + ("l1/" if workload == "simbench_bw" else "")
            + f"{slug} NVCC='nvcc -ccbin=/usr/bin/g++-14'"
        )
    if proposed_action == "REBUILD_TRACE_LOCAL":
        # Workload-specific dispatch: simbench_bw l1 slugs go through
        # `run_suite.py --mode trace`; portable_lcd slugs go through
        # `activation_trace_orchestrator.py`. The simbench_bw runner
        # resolves `workloads/simbench_bw/shapes/<slug>.toml`; it
        # cannot find portable_lcd slugs because their shape catalogs
        # live alongside their source dirs under
        # `workloads/portable_lcd/<slug>/`. The orchestrator accepts
        # any portable_lcd slug + shape_point pair via
        # ``--slug`` + ``--shape-point`` flags; the orchestrator's
        # default 5-cell tuple applies only when both flags are
        # omitted.
        if workload == "portable_lcd":
            return (
                "uv run --with-requirements requirements.txt --with protobuf "
                "python tools/local_labels/activation_trace_orchestrator.py "
                f"--slug {slug} --shape-point {shape_point} "
                f"--sku {_HOST_SKU} --arch-dir {_HOST_ARCH}"
            )
        return (
            "uv run --with-requirements requirements.txt --with protobuf "
            f"python workloads/simbench_bw/run_suite.py --mode trace "
            f"--kernel {slug} --shape-point {shape_point}"
        )
    if proposed_action == "REGENERATE_SIDECARS_LOCAL":
        # Per-cell sidecar entry point. The bulk-cohort entry point at
        # `regenerate_v5_trace_sidecars.py` consumes the v5 staging
        # manifest and only accepts (--manifest, --report, --dry-run);
        # the per-cell surface lives at
        # `regenerate_sidecars_for_cell.py` and accepts the per-cell
        # triple (--slug, --shape-point, --arch).
        return (
            "uv run --with-requirements requirements.txt --with protobuf "
            "python -m tools.local_labels.regenerate_sidecars_for_cell "
            f"--slug {slug} --shape-point {shape_point} --arch {_HOST_ARCH}"
        )
    if proposed_action == "REBUILD_BUNDLE_LOCAL":
        # Per-cell two-stage admission cascade (legacy_match then
        # fresh_regen via the walker's in-process API). The
        # bulk-cohort surface is `regenerate_v5_artifacts.py`;
        # this per-cell entry point lives at
        # `build_predictor_bundle_for_cell.py`.
        return (
            "uv run --with-requirements requirements.txt --with protobuf "
            "python -m tools.local_labels.build_predictor_bundle_for_cell "
            f"--slug {slug} --shape-point {shape_point} --arch {_HOST_ARCH}"
        )
    if proposed_action == "TERMINAL_TYPED_FAILURE":
        return "noop  # terminal: arch_gate_excludes_sm_86"
    if proposed_action == "READY":
        return "noop  # all artifacts present + identity bound"
    raise ValueError(f"unknown proposed_action {proposed_action!r}")


def _resolve_current_source_commit() -> str:
    """Return the current ``HEAD`` short-or-full commit hash, or the
    string ``"unknown"`` when ``git`` is unavailable.
    """
    try:
        out = subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def _build_cell_entry(
    cell: dict[str, Any],
    bundles: list[dict[str, Any]],
) -> dict[str, Any]:
    slug = cell["slug"]
    shape_point = cell["shape_point"]
    workload = _classify_workload(slug)
    if workload is None:
        # The slug source dir was not located under either workload.
        # The cell still gets a ledger entry; downstream tasks
        # classify it as terminal with reason
        # ``slug_source_dir_not_found``. The preflight action is
        # ``TERMINAL_TYPED_FAILURE`` so the regen lane skips it.
        return {
            "arch": cell["arch"],
            "slug": slug,
            "shape_point": shape_point,
            "binary_sha256": cell["binary_sha256"],
            "kernel_signature_sha256": cell["kernel_signature_sha256"],
            "source_commit": cell["source_commit"],
            "workload": None,
            "binary_path": None,
            "binary_present": False,
            "binary_arch_gate": [],
            "arch_gate_excludes_sm_86": True,
            "trace_dir": None,
            "trace_present": False,
            "sidecar_paths": {},
            "sidecars_present": {
                "sharing_locality": False,
                "hrd_profile": False,
                "sector_stream": False,
                "sector_stream_mpi": False,
            },
            "bundle_match_path": None,
            "bundle_present": False,
            "proposed_action": "TERMINAL_TYPED_FAILURE",
            "action_typed_reason": "slug_source_dir_not_found",
            "preflight_command": "noop  # terminal: slug_source_dir_not_found",
        }
    binary_path = _binary_path(workload, slug)
    binary_present = binary_path.is_file()
    arch_gate = _read_arch_gate(workload, slug)
    arch_gate_excludes = bool(arch_gate) and _HOST_ARCH not in arch_gate
    trace_dir = _trace_dir(workload, slug, shape_point)
    pb_path = trace_dir / "dynamic_trace.pb"
    trace_present = pb_path.is_file()
    sidecars_present = _sidecar_presence(trace_dir)
    bundle = _find_bundle_for_cell(cell, bundles)
    bundle_match_path = (
        str(bundle["path"].relative_to(_REPO_ROOT)) if bundle else None
    )
    bundle_present = bundle is not None
    proposed_action, typed_reason = _classify_action(
        arch_gate_excludes=arch_gate_excludes,
        binary_present=binary_present,
        trace_present=trace_present,
        sidecars_complete=_all_sidecars_present(sidecars_present),
        bundle_present=bundle_present,
    )
    preflight_command = _preflight_command(
        workload=workload,
        slug=slug,
        shape_point=shape_point,
        proposed_action=proposed_action,
    )
    sidecar_paths = {
        key: str(path.relative_to(_REPO_ROOT))
        for key, path in _sidecar_paths(trace_dir).items()
    }
    return {
        "arch": cell["arch"],
        "slug": slug,
        "shape_point": shape_point,
        "binary_sha256": cell["binary_sha256"],
        "kernel_signature_sha256": cell["kernel_signature_sha256"],
        "source_commit": cell["source_commit"],
        "workload": workload,
        "binary_path": str(binary_path.relative_to(_REPO_ROOT)),
        "binary_present": binary_present,
        "binary_arch_gate": arch_gate,
        "arch_gate_excludes_sm_86": arch_gate_excludes,
        "trace_dir": str(trace_dir.relative_to(_REPO_ROOT)),
        "trace_present": trace_present,
        "sidecar_paths": sidecar_paths,
        "sidecars_present": sidecars_present,
        "bundle_match_path": bundle_match_path,
        "bundle_present": bundle_present,
        "proposed_action": proposed_action,
        "action_typed_reason": typed_reason,
        "preflight_command": preflight_command,
    }


def build_preflight(
    *,
    label_path: Path = _DEFAULT_LABEL,
    out_path: Path = _DEFAULT_OUT,
) -> dict[str, Any]:
    with label_path.open() as fh:
        label = json.load(fh)
    sm86 = [c for c in label["cells"] if c["arch"] == _HOST_ARCH]
    bundles = _scan_existing_bundles()
    entries = [_build_cell_entry(c, bundles) for c in sm86]
    action_summary: dict[str, int] = {a: 0 for a in _CLOSED_ACTION_ENUM}
    for entry in entries:
        action_summary[entry["proposed_action"]] += 1
    matrix = {
        "schema_version": _PREFLIGHT_SCHEMA_VERSION,
        "generated_at_utc": _dt.datetime.now(tz=_dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "source_commit": _resolve_current_source_commit(),
        "host_sku": _HOST_SKU,
        "host_arch": _HOST_ARCH,
        "n_cells": len(entries),
        "n_sm86_cells_in_label": len(sm86),
        "action_summary": action_summary,
        "closed_action_enum": list(_CLOSED_ACTION_ENUM),
        "cells": entries,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(matrix, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return matrix


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-path", type=Path, default=_DEFAULT_LABEL,
        help=f"Source label (default: {_DEFAULT_LABEL.relative_to(_REPO_ROOT)})",
    )
    parser.add_argument(
        "--out-path", type=Path, default=_DEFAULT_OUT,
        help=f"Output JSON (default: {_DEFAULT_OUT.relative_to(_REPO_ROOT)})",
    )
    return parser.parse_args(argv)


def _display_path(path: Path) -> str:
    """Return ``path`` relative to ``_REPO_ROOT`` when it lives
    inside the repo, otherwise return the absolute path. The
    unguarded ``relative_to`` call raises ``ValueError`` when the
    user passes an out-of-tree ``--out-path`` (e.g., a temp file
    in ``/tmp``) and exits the CLI with a confusing traceback
    AFTER the file has already been written. Guarding the display
    keeps the CLI usable with temp outputs.
    """
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    matrix = build_preflight(label_path=args.label_path, out_path=args.out_path)
    print(
        f"sm_86 preflight: {matrix['n_cells']} cells emitted to "
        f"{_display_path(args.out_path)} "
        f"(actions: {matrix['action_summary']})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
