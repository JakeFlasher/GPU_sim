"""Authoritative CUTLASS / CuTe vendoring closure tool.

The vendored subset under
`workloads/portable_lcd/cutlass/upstream/` is the include-closure of
a fixed seed list under a parser that handles BOTH `#include "..."`
(quoted) AND `#include <...>` (angle-bracket) directives, with a
local-prefix gate (`cutlass/`, `cute/`) so that CUDA toolkit / system
headers are skipped without needing them on disk.

This module is the single source of truth for that closure. It is
also exposed as a CLI so any future round can:

  python3 workloads/portable_lcd/cutlass/tools/recompute_subset.py \
      --source-include-root <CUTLASS-vYY.Z>/include \
      --dest-upstream workloads/portable_lcd/cutlass/upstream \
      --mode check

to verify the on-disk subset is exactly the closure under the
documented seed list, or run with `--mode sync` to rebuild it after
a tag bump.

The same parser logic backs
`tests/test_portable_lcd_cutlass_upstream_closure.py`, which validates
the on-disk tree without needing a fresh upstream snapshot.

Design rules (mirrored in the closure pytest):

* Quoted include `#include "X"`:
    - Treat as local. Resolve absolute-from-root (`<root>/X`) first;
      fall back to sibling-relative (`<includer.parent>/X`).
    - If `X` starts with one of the recognised external prefixes
      (`cuda`, `cooperative_groups`, `nvrtc.h`, `math_constants.h`,
      libc), skip it as a toolchain header.
* Angle-bracket include `#include <X>`:
    - Treat as local only when `X` starts with `cutlass/` or
      `cute/`. Resolve absolute-from-root.
    - Otherwise treat as a system / CUDA toolkit header and skip.

The seed list reflects the kernels currently in scope:
sm_80 GEMM (BF16/FP16), Conv2dFprop sm_80, plus numeric type wrappers.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

SEEDS: tuple[str, ...] = (
    # Top-level core types
    "cutlass/cutlass.h",
    "cutlass/array.h",
    "cutlass/coord.h",
    "cutlass/numeric_types.h",
    # Warp-level MMA atoms (sm_80)
    "cutlass/arch/mma.h",
    "cutlass/arch/mma_sm80.h",
    # sm_80 GEMM multistage mainloop
    "cutlass/gemm/collective/sm80_mma_multistage.hpp",
    # Numeric type wrappers
    "cutlass/bfloat16.h",
    "cutlass/half.h",
    # Conv2dFprop sm_80 entry points
    "cutlass/conv/device/implicit_gemm_convolution.h",
    "cutlass/conv/kernel/default_conv2d_fprop.h",
)

EXTERNAL_QUOTED_PREFIXES: tuple[str, ...] = (
    "cuda",
    "cooperative_groups",
    "nvrtc.h",
    "math_constants.h",
    "stdint.h",
    "stdlib.h",
    "stdio.h",
    "stddef.h",
    "string.h",
    "assert.h",
)

LOCAL_ANGLE_PREFIXES: tuple[str, ...] = ("cutlass/", "cute/")

VENDORED_SUFFIXES: frozenset[str] = frozenset({".h", ".hpp", ".cuh", ".inl"})

_QUOTED_RE = re.compile(r'^\s*#\s*include\s*"([^"]+)"')
_ANGLE_RE = re.compile(r'^\s*#\s*include\s*<([^>]+)>')


def _includes_of(path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        m = _QUOTED_RE.match(line)
        if m:
            out.append(("quote", m.group(1)))
            continue
        m = _ANGLE_RE.match(line)
        if m:
            out.append(("angle", m.group(1)))
    return out


def _is_external_quoted(inc: str) -> bool:
    return any(inc.startswith(prefix) for prefix in EXTERNAL_QUOTED_PREFIXES)


def _is_local_angle(inc: str) -> bool:
    return any(inc.startswith(prefix) for prefix in LOCAL_ANGLE_PREFIXES)


def compute_closure(root: Path, seeds: tuple[str, ...] = SEEDS) -> set[str]:
    """Return the set of repo-relative paths reachable from `seeds`.

    Walks both quoted and angle-bracket `#include` directives under
    the rules documented in the module docstring. Paths are returned
    as POSIX-style strings relative to `root`.

    Missing files are silently skipped (returned in the visited set
    but cannot contribute further dependencies). Callers that want
    presence enforced should compare the result against the actual
    file set on disk.
    """
    visited: set[str] = set()
    queue: list[str] = list(seeds)
    while queue:
        rel = queue.pop()
        if rel in visited:
            continue
        visited.add(rel)
        path = root / rel
        if not path.is_file():
            continue
        for form, inc in _includes_of(path):
            if form == "quote":
                if _is_external_quoted(inc):
                    continue
                # Try absolute from include root first, then sibling
                # relative to the includer.
                abs_target = (root / inc).resolve()
                rel_target = (path.parent / inc).resolve()
                for candidate in (abs_target, rel_target):
                    try:
                        candidate_rel = candidate.relative_to(root.resolve())
                    except ValueError:
                        continue
                    if candidate.is_file():
                        next_rel = candidate_rel.as_posix()
                        if next_rel not in visited:
                            queue.append(next_rel)
                        break
            else:
                if not _is_local_angle(inc):
                    continue
                # Angle-bracket local: absolute from root only.
                if inc not in visited:
                    queue.append(inc)
    return visited


def discover_actual_files(root: Path) -> set[str]:
    """Set of repo-relative paths to every vendored-suffixed header under `root`.

    Uses the lexical path (does not resolve symlinks) so that staged
    test trees built out of symlinks back to a real source tree are
    reported as if the targets lived under `root`.
    """
    out: set[str] = set()
    if not root.is_dir():
        return out
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix not in VENDORED_SUFFIXES:
            continue
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue
        out.add(rel.as_posix())
    return out


def read_files_manifest(dest_upstream: Path) -> set[str]:
    """Return entries from `<dest>/FILES.txt`, stripping blanks / comments."""
    manifest = dest_upstream / "FILES.txt"
    if not manifest.is_file():
        return set()
    out: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(line)
    return out


def write_files_manifest(dest_upstream: Path, paths: set[str]) -> None:
    manifest = dest_upstream / "FILES.txt"
    sorted_paths = sorted(paths)
    manifest.write_text("\n".join(sorted_paths) + "\n", encoding="utf-8")


def run_check(source_root: Path, dest_upstream: Path) -> int:
    """Compute the closure under `source_root` and compare to dest tree.

    Returns 0 on agreement, non-zero on drift. Prints typed
    diagnostics to stderr on drift.
    """
    closure = compute_closure(source_root)
    closure_files = {p for p in closure if (source_root / p).is_file()}

    actual = discover_actual_files(dest_upstream)
    manifest = read_files_manifest(dest_upstream)

    missing_from_dest = sorted(closure_files - actual)
    extra_in_dest = sorted(actual - closure_files)
    manifest_drift = sorted(closure_files.symmetric_difference(manifest))

    if not (missing_from_dest or extra_in_dest or manifest_drift):
        print(
            f"recompute_subset_check_ok: closure={len(closure_files)} "
            f"actual={len(actual)} manifest={len(manifest)}"
        )
        return 0

    print("recompute_subset_check_failed:", file=sys.stderr)
    if missing_from_dest:
        print(
            f"  closure_files_missing_from_dest_upstream "
            f"({len(missing_from_dest)}):",
            file=sys.stderr,
        )
        for p in missing_from_dest[:20]:
            print(f"    {p}", file=sys.stderr)
        if len(missing_from_dest) > 20:
            print(
                f"    ... and {len(missing_from_dest) - 20} more",
                file=sys.stderr,
            )
    if extra_in_dest:
        print(
            f"  dest_upstream_files_not_in_closure ({len(extra_in_dest)}):",
            file=sys.stderr,
        )
        for p in extra_in_dest[:20]:
            print(f"    {p}", file=sys.stderr)
        if len(extra_in_dest) > 20:
            print(
                f"    ... and {len(extra_in_dest) - 20} more",
                file=sys.stderr,
            )
    if manifest_drift:
        print(
            f"  files_manifest_drift ({len(manifest_drift)}):",
            file=sys.stderr,
        )
        for p in manifest_drift[:20]:
            print(f"    {p}", file=sys.stderr)
        if len(manifest_drift) > 20:
            print(
                f"    ... and {len(manifest_drift) - 20} more",
                file=sys.stderr,
            )
    return 1


def run_sync(source_root: Path, dest_upstream: Path) -> int:
    """Compute the closure and materialise it under `dest_upstream`.

    Copies every closure file from `source_root` into the same
    relative path under `dest_upstream`. Rewrites
    `dest_upstream/FILES.txt`. Does NOT remove pre-existing files
    that fall outside the new closure -- the caller is expected to
    invoke this on a clean dest, or to reconcile separately.
    """
    closure = compute_closure(source_root)
    closure_files = {p for p in closure if (source_root / p).is_file()}
    if not closure_files:
        print(
            "recompute_subset_sync_failed: empty closure; check seeds and "
            "source-include-root",
            file=sys.stderr,
        )
        return 1
    dest_upstream.mkdir(parents=True, exist_ok=True)
    for rel in closure_files:
        src = source_root / rel
        dst = dest_upstream / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    write_files_manifest(dest_upstream, closure_files)
    print(
        f"recompute_subset_sync_ok: copied={len(closure_files)} "
        f"manifest_entries={len(closure_files)}"
    )
    return 0


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute / verify the vendored CUTLASS+CuTe include-closure "
            "subset under workloads/portable_lcd/cutlass/upstream/."
        ),
    )
    parser.add_argument(
        "--source-include-root",
        type=Path,
        required=True,
        help=(
            "Path to a CUTLASS include root that contains both `cutlass/` "
            "and `cute/` subtrees (typically <CUTLASS-tag>/include)."
        ),
    )
    parser.add_argument(
        "--dest-upstream",
        type=Path,
        required=True,
        help=(
            "Path to the vendored subset destination "
            "(workloads/portable_lcd/cutlass/upstream)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("check", "sync"),
        required=True,
        help=(
            "check: compare computed closure against dest-upstream and "
            "FILES.txt (non-zero exit on drift). sync: copy the closure "
            "into dest-upstream and rewrite FILES.txt."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    source_root: Path = args.source_include_root
    dest: Path = args.dest_upstream
    if not source_root.is_dir():
        print(
            f"recompute_subset_failed: --source-include-root not a directory: "
            f"{source_root}",
            file=sys.stderr,
        )
        return 2
    if args.mode == "check":
        return run_check(source_root, dest)
    return run_sync(source_root, dest)


if __name__ == "__main__":
    raise SystemExit(main())
