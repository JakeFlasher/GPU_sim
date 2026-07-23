"""Emit the register-spill `register_cap` build-variant manifest (GPU-free).

The register-spill family's spill phenomenon is controlled by a BUILD-TIME
per-thread register cap (`-maxrregcount`, the Makefile `SPILL_MAXREG`), NOT a
runtime axis. This emitter turns that cap ladder into an explicit, committed
build-variant manifest so the variants are first-class, labelable cells with
distinct STABLE identity (register_cap + build flag + per-variant binary_sha256
+ kernel_signature_sha256 + regs/thread + spill bytes + source_commit), rather
than prose alone. The collection harness selects/rebuilds the matching per-cap
binary (via `make ... SPILL_MAXREG=<cap>`) before capture.

Each variant's identity is distinct because the cap changes the static register
count (sm_86: cap 255 -> 96 regs / 0 spill; 64 -> 64 / 112; 32 -> 32 / 424;
24 -> 24 / 472), which flows into both the binary and the kernel signature
(`regs_per_thread` is a signature field). The kernel signature is deterministic
(a hash of launch geometry + regs), so it is stable across rebuilds; the binary
hash is a per-build snapshot.

GPU-free: builds via nvcc and inspects via `cuobjdump -res-usage` only (no kernel
run, no counters). Public surface:

- ``build_variant_manifest(...)`` -> the manifest dict (builds the ladder).
- ``emit(...)`` -> write the committed manifest.
- CLI: ``python -m tools.local_labels.emit_register_cap_variants [--out ...]``.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cuobjdump_res_usage import (  # noqa: E402
    parse_cuobjdump_res_usage,
    select_resource_usage,
)
from verify_spill_sweep import (  # noqa: E402
    LADDER_CAPS,
    SpillSweepError,
    build_spill_ladder,
    spill_bytes,
    spill_sweep_verdict,
    toolchain_available,
)
from workloads.simbench_bw.nvbit_trace_summary_adapter import (  # noqa: E402
    compute_kernel_signature_sha256,
)

_MANIFEST_SCHEMA_VERSION = 1
_ARTIFACT_KIND = "register_cap_build_variants"
_SLUG = "register_spill_local_mem"
_KERNEL = "register_spill_local_mem_kernel"
_ARCH = "sm_86"
_KERNEL_DIR = _REPO_ROOT / "workloads" / "simbench_bw" / "l1" / _SLUG
_KERNEL_SRC = _KERNEL_DIR / f"{_SLUG}.cu"
_MAIN_SRC = _KERNEL_DIR / "main.cu"
_DEFAULT_MANIFEST = (
    _REPO_ROOT / "data" / "local_labels" / "build_variants"
    / f"{_SLUG}_{_ARCH}.json"
)
# A fixed canonical launch geometry so each variant's kernel signature isolates
# the register cap (via regs_per_thread); the cap, not the geometry, is the axis.
_CANONICAL_LAUNCH = {
    "grid_x": 8, "grid_y": 1, "grid_z": 1,
    "block_x": 128, "block_y": 1, "block_z": 1,
    "shared_mem_static_bytes": 0, "shared_mem_dynamic_bytes": 0,
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown_source_commit"


def _variant_signature(regs_per_thread: int) -> str:
    return compute_kernel_signature_sha256(
        {"kernel_name": _KERNEL, "regs_per_thread": int(regs_per_thread),
         **_CANONICAL_LAUNCH}
    )


def build_variant_manifest(
    out_dir: Path, caps: tuple[int, ...] = LADDER_CAPS
) -> dict:
    """Build the cap ladder and assemble the build-variant manifest dict.

    Requires the CUDA toolchain (callers gate on ``toolchain_available()``)."""
    ok, reason = toolchain_available()
    if not ok:
        raise SpillSweepError(f"toolchain unavailable: {reason}")
    binaries = build_spill_ladder(_KERNEL_SRC, _MAIN_SRC, _ARCH, out_dir, caps)

    variants: list[dict] = []
    spill_by_cap: dict[int, int] = {}
    for cap in caps:
        binary = binaries[cap]
        proc = subprocess.run(
            ["cuobjdump", "-res-usage", str(binary)], capture_output=True, text=True
        )
        if proc.returncode != 0:
            raise SpillSweepError(
                f"cuobjdump -res-usage failed for cap {cap} (rc={proc.returncode})"
            )
        usage = select_resource_usage(
            parse_cuobjdump_res_usage(proc.stdout), arch=_ARCH, function=_KERNEL
        )
        sb = spill_bytes(usage)
        spill_by_cap[cap] = sb
        variants.append({
            "variant_id": f"register_cap_{cap}",
            "register_cap": cap,
            "nvcc_flag": f"-maxrregcount={cap}",
            "build_via_makefile": f"make run-bin SPILL_MAXREG={cap}",
            "regs_per_thread": int(usage.registers_per_thread),
            "spill_bytes": sb,
            "binary_sha256": _sha256_file(binary),
            "kernel_signature_sha256": _variant_signature(usage.registers_per_thread),
        })

    return {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "artifact_kind": _ARTIFACT_KIND,
        "arch": _ARCH,
        "slug": _SLUG,
        "kernel": _KERNEL,
        "source_commit": _source_commit(),
        "build_flag_template": "-maxrregcount=<register_cap>",
        "canonical_launch_geometry": dict(_CANONICAL_LAUNCH),
        "spill_sweep_verdict": spill_sweep_verdict(spill_by_cap),
        "variants": variants,
    }


def emit(out_path: Path | None = None, caps: tuple[int, ...] = LADDER_CAPS) -> Path:
    import json

    import tempfile

    out_path = out_path if out_path is not None else _DEFAULT_MANIFEST
    with tempfile.TemporaryDirectory() as tmp:
        manifest = build_variant_manifest(Path(tmp), caps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return out_path


def _main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(_DEFAULT_MANIFEST))
    ap.add_argument("--print", action="store_true", help="print the manifest JSON")
    args = ap.parse_args(argv)

    ok, reason = toolchain_available()
    if not ok:
        print(f"toolchain_unavailable: {reason}", file=sys.stderr)
        return 3
    written = emit(Path(args.out))
    if args.print:
        print(json.loads(written.read_text(encoding="utf-8")))
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
