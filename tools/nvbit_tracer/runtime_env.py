"""Fail-closed runtime environment for loading an NVBit tracer tool.

The custom tracer links the checkout-local protobuf/Abseil runtime. CUDA's
``CUDA_INJECTION64_PATH`` loader can fail silently when one of those DSOs is
missing, leaving the target application to run without instrumentation. Keep
that loader contract in one place for every Python capture entry point.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PB_LIB = REPO_ROOT / ".toolchain" / "pb" / "lib"
_PINNED_DSO_PREFIXES = ("libprotobuf.so", "libabsl_", "libutf8_")


class TracerRuntimeError(RuntimeError):
    """The tracer cannot be loaded reliably in the requested environment."""


def _prepend_once(value: str, current: str) -> str:
    parts = [part for part in current.split(":") if part]
    return ":".join([value, *(part for part in parts if part != value)])


def _configured_pb_lib(pb_lib: Path | None) -> Path:
    """Resolve the runtime prefix without silently accepting relative overrides."""

    if pb_lib is not None:
        selected = Path(pb_lib)
        if not selected.is_absolute():
            raise TracerRuntimeError(
                f"pinned protobuf runtime path must be absolute: {selected}"
            )
        return selected
    override = os.environ.get("NVBIT_PB_PREFIX")
    if override is None:
        return DEFAULT_PB_LIB
    prefix = Path(override)
    if not prefix.is_absolute():
        raise TracerRuntimeError(
            f"NVBIT_PB_PREFIX must be absolute when set: {override}"
        )
    return prefix / "lib"


def _pinned_runtime_mismatches(ldd_output: str, pb_lib: Path) -> list[str]:
    """Return protobuf-family DSOs that resolved outside ``pb_lib``."""

    mismatches: list[str] = []
    protobuf_seen = False
    for raw_line in ldd_output.splitlines():
        fields = raw_line.strip().split()
        if not fields:
            continue
        soname = fields[0]
        if not soname.startswith(_PINNED_DSO_PREFIXES):
            continue
        protobuf_seen = protobuf_seen or soname.startswith("libprotobuf.so")
        if "=>" not in fields:
            mismatches.append(f"{soname} has no resolved path")
            continue
        try:
            resolved = fields[fields.index("=>") + 1]
        except IndexError:
            mismatches.append(f"{soname} has no resolved path")
            continue
        if resolved == "not":
            # The caller reports unresolved DSOs separately with the complete
            # ldd line, which is more actionable than a prefix mismatch.
            continue
        try:
            Path(resolved).resolve().relative_to(pb_lib)
        except ValueError:
            mismatches.append(f"{soname} => {resolved}")
    if not protobuf_seen:
        mismatches.append("libprotobuf.so mapping is absent")
    return mismatches


def tracer_runtime_env(
    tracer_so: Path,
    *,
    base_env: Mapping[str, str] | None = None,
    pb_lib: Path | None = None,
) -> dict[str, str]:
    """Return an environment in which ``tracer_so`` has no missing DSOs.

    ``ldd`` is a necessary preflight, not proof of successful instrumentation;
    callers must still require a non-empty, contract-valid trace.
    """

    tracer_so = Path(tracer_so).resolve()
    selected_pb_lib = _configured_pb_lib(pb_lib)
    cursor = selected_pb_lib
    symlinked = None
    while cursor != cursor.parent:
        if cursor.is_symlink():
            symlinked = cursor
            break
        cursor = cursor.parent
    if symlinked is not None:
        raise TracerRuntimeError(
            f"pinned protobuf runtime directory must not be a symlink: "
            f"{symlinked}"
        )
    pb_lib = selected_pb_lib.resolve()
    if not tracer_so.is_file():
        raise TracerRuntimeError(f"tracer .so is missing: {tracer_so}")
    if not pb_lib.is_dir():
        raise TracerRuntimeError(
            f"pinned protobuf runtime directory is missing: {pb_lib}"
        )

    env = dict(os.environ if base_env is None else base_env)
    env["LD_LIBRARY_PATH"] = _prepend_once(
        str(pb_lib), env.get("LD_LIBRARY_PATH", "")
    )
    try:
        proc = subprocess.run(
            ["ldd", str(tracer_so)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
    except OSError as exc:
        raise TracerRuntimeError(f"could not execute ldd: {exc}") from exc

    ldd_output = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
    missing = [line.strip() for line in ldd_output.splitlines() if "not found" in line]
    if proc.returncode != 0 or missing:
        detail = missing or [proc.stderr.strip() or f"ldd exited {proc.returncode}"]
        raise TracerRuntimeError(
            "tracer dependencies are unresolved; CUDA_INJECTION64_PATH would "
            f"silently run untraced: {detail}"
        )
    wrong_prefix = _pinned_runtime_mismatches(ldd_output, pb_lib)
    if wrong_prefix:
        raise TracerRuntimeError(
            "tracer protobuf dependencies did not resolve from the pinned "
            f"runtime {pb_lib}: {wrong_prefix}"
        )
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracer_so", type=Path)
    parser.add_argument("--pb-lib", type=Path)
    args = parser.parse_args(argv)
    try:
        env = tracer_runtime_env(args.tracer_so, pb_lib=args.pb_lib)
    except TracerRuntimeError as exc:
        print(f"tracer_runtime: FAIL {exc}", file=sys.stderr)
        return 1
    print(f"tracer_runtime: ok LD_LIBRARY_PATH={env['LD_LIBRARY_PATH']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
