"""Differential register-spill verifier for the -maxrregcount sweep (GPU-free).

A register-spill microbench is only a valid factorial cell if its
``-maxrregcount`` sweep ACTUALLY changes spill behavior: at a high register cap
the kernel's live set fits in registers (no spill), and as the cap drops the
live set is forced to spill to local memory. If spill does NOT change across
the sweep (a confounded signal -- e.g. the kernel has too few live values to
ever spill, or spills even at the highest cap so the cap is not the driver),
the family must be REJECTED (the plan's AC negative test for this family).

This is enforced statically and GPU-FREE: the kernel is compiled at a ladder of
``-maxrregcount`` caps and ``cuobjdump -res-usage`` reports each build's spill
(STACK-frame + LOCAL bytes). No kernel runs, no counter is read. Empirically on
sm_86 the committed register-spill kernel gives spill bytes
{255: 0, 64: 112, 32: 424, 24: 472} -- a clean monotonic delta.

Public surface:

- ``LADDER_CAPS`` -- the default -maxrregcount ladder (high -> low).
- ``spill_bytes(usage)`` -- STACK + LOCAL spill bytes for one parsed row.
- ``spill_sweep_verdict(spill_by_cap)`` -- pure verdict (ok + typed reason).
- ``assert_spill_sweep_not_confounded(spill_by_cap)`` -- raise on a bad sweep.
- ``build_spill_ladder(...)`` / ``measure_spill_ladder(...)`` -- nvcc + cuobjdump
  steps (skip-gated on toolchain availability).
- ``toolchain_available()`` -- (ok, reason) for nvcc + cuobjdump.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# Make the bare sibling import work both when run as a script (cwd =
# tools/local_labels, pytest prepend mode) AND as ``python -m
# tools.local_labels.verify_spill_sweep`` (cwd = repo root, this dir not on
# sys.path). Insert this module's own directory before importing the sibling.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from cuobjdump_res_usage import (  # noqa: E402  (path-robust sibling import)
    KernelResourceUsage,
    parse_cuobjdump_res_usage,
    select_resource_usage,
)

# High -> low. The highest cap must NOT spill and the lowest must spill, so the
# sweep demonstrably changes spill. (sm_86 register file is 64K/SM; a 64-wide
# FP32 live set fits uncapped and spills under a tight cap.)
LADDER_CAPS: tuple[int, ...] = (255, 64, 32, 24)


class SpillSweepError(ValueError):
    """Raised when a -maxrregcount sweep does not demonstrably change spill."""


def toolchain_available() -> tuple[bool, str]:
    if shutil.which("nvcc") is None:
        return False, "nvcc not on PATH"
    if shutil.which("cuobjdump") is None:
        return False, "cuobjdump not on PATH"
    return True, ""


def spill_bytes(usage: KernelResourceUsage) -> int:
    """Total per-thread spill bytes: stack-frame + local. Either being > 0 is a
    spill; cuobjdump reports the register-pressure spill as STACK on sm_86."""
    return int(usage.stack_frame_bytes) + int(usage.local_bytes)


def spill_sweep_verdict(spill_by_cap: dict[int, int]) -> dict:
    """Pure verdict over a {register_cap: spill_bytes} map (no GPU, no I/O).

    The sweep is CONFIRMED (ok) only if every condition holds:
      - at least two distinct caps were measured;
      - the highest cap does NOT spill (spill == 0) -- the live set fits when
        registers are plentiful, so the cap (not something else) drives spill;
      - the lowest cap DOES spill (spill > 0) -- a tight cap forces it;
      - spill is non-decreasing as the cap decreases (monotone) -- no inversion;
      - max spill > min spill (a real delta, not a constant/confounded signal).
    Otherwise ok is False with a typed reason.
    """
    if len(spill_by_cap) < 2:
        return {
            "ok": False,
            "reason": "insufficient_caps",
            "detail": f"need >= 2 register caps, got {sorted(spill_by_cap)}",
            "spill_by_cap": dict(spill_by_cap),
        }
    caps_desc = sorted(spill_by_cap, reverse=True)
    spills_desc = [spill_by_cap[c] for c in caps_desc]
    top_cap, min_cap = caps_desc[0], caps_desc[-1]
    top_spill, bottom_spill = spill_by_cap[top_cap], spill_by_cap[min_cap]
    delta = bottom_spill - top_spill

    base = {
        "spill_by_cap": dict(spill_by_cap),
        "top_cap": top_cap,
        "top_spill": top_spill,
        "min_cap": min_cap,
        "bottom_spill": bottom_spill,
        "delta": delta,
    }
    if max(spills_desc) == min(spills_desc):
        return {**base, "ok": False, "reason": "confounded_no_spill_delta",
                "detail": f"spill constant across the sweep ({spills_desc})"}
    if top_spill != 0:
        return {**base, "ok": False, "reason": "spills_even_at_max_register_cap",
                "detail": f"cap {top_cap} already spills {top_spill}B; the cap is "
                          f"not the spill driver"}
    if bottom_spill <= 0:
        return {**base, "ok": False, "reason": "no_spill_at_min_register_cap",
                "detail": f"cap {min_cap} does not spill; the live set is too "
                          f"small to force spill"}
    # Monotone: spill must not decrease as the cap decreases (descending caps).
    for higher, lower in zip(spills_desc, spills_desc[1:]):
        if lower < higher:
            return {**base, "ok": False, "reason": "non_monotonic_spill_sweep",
                    "detail": f"spill decreased as the cap dropped: {spills_desc}"}
    return {**base, "ok": True, "reason": "spill_sweep_confirmed", "detail": ""}


def assert_spill_sweep_not_confounded(spill_by_cap: dict[int, int]) -> dict:
    """Raise SpillSweepError unless the sweep demonstrably changes spill."""
    verdict = spill_sweep_verdict(spill_by_cap)
    if not verdict["ok"]:
        raise SpillSweepError(
            f"register-spill -maxrregcount sweep rejected "
            f"({verdict['reason']}): {verdict['detail']}"
        )
    return verdict


def build_spill_ladder(
    kernel_src: Path,
    main_src: Path,
    arch: str,
    out_dir: Path,
    caps: tuple[int, ...] = LADDER_CAPS,
    nvcc: str = "nvcc",
) -> dict[int, Path]:
    """Compile ``kernel_src`` + ``main_src`` once per cap into ``out_dir``.

    Single-arch (``-arch=<arch>``) executables so ``cuobjdump -res-usage`` emits
    an ``arch =`` header the parser binds to. GPU-free (compile only)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    built: dict[int, Path] = {}
    for cap in caps:
        binary = out_dir / f"spill_cap{cap}_run"
        cmd = [
            nvcc, "-std=c++17", f"-arch={arch}", f"-maxrregcount={cap}",
            str(kernel_src), str(main_src), "-o", str(binary), "-lcudart",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SpillSweepError(
                f"nvcc build failed at -maxrregcount={cap} (rc={proc.returncode}): "
                f"{proc.stderr.strip()[:300]}"
            )
        built[cap] = binary
    return built


def measure_spill_ladder(
    binaries: dict[int, Path],
    arch: str,
    function: str,
    cuobjdump: str = "cuobjdump",
) -> dict[int, int]:
    """Run ``cuobjdump -res-usage`` per ladder binary and return spill bytes."""
    spill_by_cap: dict[int, int] = {}
    for cap, binary in binaries.items():
        proc = subprocess.run(
            [cuobjdump, "-res-usage", str(binary)], capture_output=True, text=True
        )
        if proc.returncode != 0:
            raise SpillSweepError(
                f"cuobjdump -res-usage failed for {binary} (rc={proc.returncode})"
            )
        usage = select_resource_usage(
            parse_cuobjdump_res_usage(proc.stdout), arch=arch, function=function
        )
        spill_by_cap[cap] = spill_bytes(usage)
    return spill_by_cap


def build_and_verify(
    kernel_src: Path,
    main_src: Path,
    arch: str,
    function: str,
    out_dir: Path,
    caps: tuple[int, ...] = LADDER_CAPS,
    nvcc: str = "nvcc",
    cuobjdump: str = "cuobjdump",
) -> dict:
    """Build the ladder, measure spill, and return the verdict (raises on
    toolchain absence -- callers gate on ``toolchain_available()``)."""
    ok, reason = toolchain_available()
    if not ok:
        raise SpillSweepError(f"toolchain unavailable: {reason}")
    binaries = build_spill_ladder(kernel_src, main_src, arch, out_dir, caps, nvcc)
    spill_by_cap = measure_spill_ladder(binaries, arch, function, cuobjdump)
    return spill_sweep_verdict(spill_by_cap)


# The committed register-spill family this verifier defaults to.
_REPO_ROOT = _HERE.parents[1]
_DEFAULT_KERNEL_DIR = (
    _REPO_ROOT / "workloads" / "simbench_bw" / "l1" / "register_spill_local_mem"
)
_DEFAULT_KERNEL_SRC = _DEFAULT_KERNEL_DIR / "register_spill_local_mem.cu"
_DEFAULT_MAIN_SRC = _DEFAULT_KERNEL_DIR / "main.cu"
_DEFAULT_FUNCTION = "register_spill_local_mem_kernel"


def _main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    import tempfile

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kernel-src", default=str(_DEFAULT_KERNEL_SRC))
    ap.add_argument("--main-src", default=str(_DEFAULT_MAIN_SRC))
    ap.add_argument("--arch", default="sm_86")
    ap.add_argument("--function", default=_DEFAULT_FUNCTION)
    ap.add_argument(
        "--caps", default=",".join(str(c) for c in LADDER_CAPS),
        help="comma-separated -maxrregcount ladder (high -> low)",
    )
    ap.add_argument("--out", default=None, help="ladder build dir (default: temp)")
    ap.add_argument("--json", default=None, help="write the verdict JSON here")
    args = ap.parse_args(argv)

    ok, reason = toolchain_available()
    if not ok:
        print(json.dumps({"ok": False, "reason": "toolchain_unavailable",
                          "detail": reason}))
        return 3  # toolchain not on PATH (matches verify_kernel_contracts)

    caps = tuple(int(c) for c in args.caps.split(",") if c.strip())
    tmp_ctx = tempfile.TemporaryDirectory() if args.out is None else None
    out_dir = Path(args.out) if args.out is not None else Path(tmp_ctx.name)
    try:
        verdict = build_and_verify(
            Path(args.kernel_src), Path(args.main_src), args.arch, args.function,
            out_dir, caps=caps,
        )
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    text = json.dumps(verdict, indent=2, sort_keys=True)
    if args.json:
        Path(args.json).write_text(text + "\n", encoding="utf-8")
    print(text)
    # Exit nonzero on any non-confirmed sweep so the gate / CI can rely on it.
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
