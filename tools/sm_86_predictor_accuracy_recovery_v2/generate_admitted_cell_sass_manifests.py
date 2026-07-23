"""Task10 (Plan-1.5) driver: generate SASS provenance manifests for
the 9 admitted sm_86 cells via the L1 cache-op classifier.

For each admitted (slug, shape_point) in
``data/local_labels/label_admitted.json`` filtered to ``arch=sm_86``,
this driver:

  1. Resolves the binary path. Microbench cells live at
     ``workloads/simbench_bw/l1/<slug>/<slug>_run``; activation cells
     live at ``workloads/portable_lcd/<slug>/<slug>_run``.
  2. Resolves the kernel unmangled name. By the project convention,
     each binary exports a single ``STT_FUNC`` ``STO_ENTRY`` symbol
     named ``<slug>_kernel``. The driver verifies via
     ``cuobjdump --dump-elf-symbols`` that exactly one such entry
     exists; multi-kernel binaries fail-close with a typed
     diagnostic.
  3. Invokes
     ``tools.cache_predictor.l1_cache_op_classifier.classify_binary``
     with the resolved binary path + kernel name + ``arch_token=sm_86``.
  4. Writes the per-cell manifest to
     ``data/local_labels/cache_op_classifier_manifests/sm_86/<slug>__<shape>.json``.

CLI:

    python3 tools/sm_86_predictor_accuracy_recovery_v2/\
        generate_admitted_cell_sass_manifests.py

The driver is idempotent: running it twice produces byte-identical
output unless the binary changed or the classifier changed.

Exit codes:
  0 -- every admitted cell's manifest was written successfully.
  2 -- one or more cells failed (binary missing / cuobjdump
       missing / multiple kernel entries / classifier raised).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "tools" / "cache_predictor"))

import l1_cache_op_classifier as classifier  # noqa: E402


def _is_microbench_slug(slug: str) -> bool:
    """Microbench cells live under ``workloads/simbench_bw/l1/``.
    Activation cells live under ``workloads/portable_lcd/``. The
    distinction is by source-tree layout, not name pattern, so we
    check both candidate paths and return whichever has the binary
    on disk.
    """
    return (
        _REPO_ROOT
        / "workloads" / "simbench_bw" / "l1" / slug / f"{slug}_run"
    ).is_file()


def _binary_path_for(slug: str) -> Path | None:
    """Return the on-disk binary path for ``slug`` or None if no
    candidate path holds a real binary. Covers both simbench_bw
    binary-name conventions (``<slug>`` and ``<slug>_run``), the
    dual-workload l2 sibling tree (e.g. rmsnorm_fwd_bf16), and the
    portable_lcd tree."""
    candidates = (
        _REPO_ROOT / "workloads" / "simbench_bw" / "l1" / slug / slug,
        _REPO_ROOT / "workloads" / "simbench_bw" / "l1" / slug / f"{slug}_run",
        _REPO_ROOT / "workloads" / "simbench_bw" / "l2" / slug / f"{slug}_run",
        _REPO_ROOT / "workloads" / "simbench_bw" / "l2" / slug / slug,
        _REPO_ROOT / "workloads" / "portable_lcd" / slug / f"{slug}_run",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _resolve_kernel_entry_name(binary_path: Path) -> str:
    """Run ``cuobjdump --dump-elf-symbols`` and return the single
    ``STT_FUNC STB_GLOBAL STO_ENTRY`` symbol name. Raises if the
    binary exports zero or multiple such symbols (the project
    convention is one kernel entry per admitted-cell binary)."""
    proc = subprocess.run(
        ["cuobjdump", "--dump-elf-symbols", str(binary_path)],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"cuobjdump --dump-elf-symbols rc={proc.returncode} on "
            f"{binary_path}: stderr_tail={proc.stderr[-256:]!r}"
        )
    entries: set[str] = set()
    for line in proc.stdout.splitlines():
        if "STT_FUNC" in line and "STO_ENTRY" in line:
            parts = line.split()
            if parts:
                entries.add(parts[-1])
    if not entries:
        raise RuntimeError(
            f"no STT_FUNC STO_ENTRY symbols found in {binary_path}; "
            f"cannot resolve kernel_unmangled_name"
        )
    if len(entries) > 1:
        raise RuntimeError(
            f"multiple kernel entries in {binary_path}: "
            f"{sorted(entries)}; admitted-cell binaries must export "
            f"a single kernel per the project convention"
        )
    return next(iter(entries))


def _load_admitted_cells(label_admitted_path: Path) -> list[tuple[str, str]]:
    """Return ``[(slug, shape_point), ...]`` for every admitted
    sm_86 cell. Rows whose ``arch`` is set to a non-sm_86 value
    (e.g., ``sm_120`` after the cross-host merge workflow) are
    skipped: this is the sm_86-only manifest generator, and the
    file header docstring asserts the input is filtered to
    ``arch=sm_86``. Rows whose ``arch`` field is missing (None)
    are preserved because earlier Plan-1 ``label_admitted.json``
    revisions omitted the field and were sm_86-only by
    construction."""
    with label_admitted_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    cells = data.get("cells", [])
    pairs: list[tuple[str, str]] = []
    for cell in cells:
        cell_arch = cell.get("arch")
        if cell_arch is not None and cell_arch != "sm_86":
            continue
        slug = cell.get("slug")
        shape_point = cell.get("shape_point")
        if not slug or not shape_point:
            raise ValueError(
                f"admitted cell missing slug or shape_point: {cell!r}"
            )
        pairs.append((slug, shape_point))
    return sorted(set(pairs))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SASS provenance manifests for all admitted "
            "sm_86 cells in label_admitted.json via the L1 cache-op "
            "classifier."
        ),
    )
    parser.add_argument(
        "--label-admitted-path",
        type=Path,
        default=_REPO_ROOT / "data" / "local_labels" / "label_admitted.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            _REPO_ROOT / "data" / "local_labels"
            / "cache_op_classifier_manifests" / "sm_86"
        ),
        help=(
            "Output directory for per-cell manifests "
            "(default: data/local_labels/cache_op_classifier_manifests/sm_86/)."
        ),
    )
    parser.add_argument(
        "--arch-token", default="sm_86",
        help="Architecture token to record in each manifest (default: sm_86).",
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = _load_admitted_cells(args.label_admitted_path)
    print(
        f"generate_admitted_cell_sass_manifests: {len(pairs)} cells",
        file=sys.stderr,
    )

    failures: list[tuple[str, str, str]] = []
    written: list[Path] = []
    for slug, shape_point in pairs:
        binary = _binary_path_for(slug)
        if binary is None:
            failures.append(
                (slug, shape_point, f"binary not found at either "
                 f"workloads/simbench_bw/l1/{slug}/{slug}_run or "
                 f"workloads/portable_lcd/{slug}/{slug}_run")
            )
            continue
        try:
            kernel_name = _resolve_kernel_entry_name(binary)
        except RuntimeError as exc:
            failures.append((slug, shape_point, str(exc)))
            continue
        try:
            manifest = classifier.classify_binary(
                binary,
                kernel_unmangled_name=kernel_name,
                arch_token=args.arch_token,
            )
        except Exception as exc:  # pylint: disable=broad-except
            failures.append((slug, shape_point, f"classify_binary raised: {exc!r}"))
            continue
        # `binary` is an absolute on-disk path so cuobjdump can read it, but
        # classify_binary serializes that path verbatim into `binary_path`.
        # Record the REPO-RELATIVE form instead so the committed provenance is
        # portable across checkouts/hosts (the schema documents binary_path as
        # repo-relative; an absolute /home/... path is host-specific).
        manifest["binary_path"] = binary.relative_to(_REPO_ROOT).as_posix()
        out_path = args.output_dir / f"{slug}__{shape_point}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
            fh.write("\n")
        written.append(out_path)
        status = manifest.get("sass_provenance_status")
        per_pc = manifest.get("per_pc") or []
        print(
            f"  {slug}/{shape_point}: kernel={kernel_name!r} "
            f"status={status!r} per_pc_entries={len(per_pc)}",
            file=sys.stderr,
        )

    if failures:
        print(
            f"\n=== {len(failures)} failures ===",
            file=sys.stderr,
        )
        for slug, shape_point, reason in failures:
            print(f"  {slug}/{shape_point}: {reason}", file=sys.stderr)
        return 2

    print(
        f"\nwrote {len(written)} manifests to {args.output_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
