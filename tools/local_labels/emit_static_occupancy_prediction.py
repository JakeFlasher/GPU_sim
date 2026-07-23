"""Produce T0 launch occupancy-limit predictions as unit-tagged static records.

This is the production-side producer for the four ``launch__occupancy_limit_*``
metrics: it turns a built kernel binary's static resource usage (per-thread
registers + static shared memory from ``cuobjdump -res-usage``) plus the launch
block dimension into a schema-valid prediction artifact, one ``{value, status,
reason, unit}`` record per limit. Every value comes verbatim from the
fail-closed static occupancy calculator, so a limit is emitted exact-or-typed-
unavailable: a deferred device constant (e.g. every sm_120 constant) or a
missing resource input yields ``status: typed_unavailable`` with a typed reason
rather than a fabricated number.

GPU-free: ``cuobjdump`` is a static binary utility (it never runs the kernel,
reads a counter, or touches driver state). The agreement of these predictions
with NCU's launch-section values is checked separately by
``validate_static_occupancy_agreement.py``.

Public surface:

- ``build_occupancy_prediction(res_usage_text, arch, block_dim, ...)`` -> the
  prediction artifact dict (the core, GPU-free, text-in entry point).
- ``run_cuobjdump_res_usage(binary, arch)`` -> raw ``-res-usage`` text.
- ``emit(...)`` -> write the artifact to a path.
- CLI: ``--binary ... --arch sm_86 --kernel ... --block-dim 256 [--out ...]``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from cuobjdump_res_usage import (
    CuobjdumpResUsageError,
    parse_cuobjdump_res_usage,
    select_resource_usage,
)
from occupancy_calculator import (
    LAUNCH_OCCUPANCY_LIMIT_METRICS,
    compute_occupancy_limits,
)

_PREDICTION_SCHEMA_VERSION = 1
_ARTIFACT_KIND = "static_occupancy_prediction"
# These predictions are static-arithmetic forecasts, never an admitted NCU
# ground-truth label; the claim_class keeps them out of the label surface.
_CLAIM_CLASS = "static_predicted"


class StaticOccupancyPredictionError(ValueError):
    """Raised when the binary cannot be inspected or the inputs are malformed."""


def run_cuobjdump_res_usage(binary: str | Path, arch: str) -> str:
    """Return ``cuobjdump -res-usage -arch <arch> <binary>`` stdout (GPU-free)."""
    cmd = ["cuobjdump", "-res-usage", "-arch", arch, str(binary)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise StaticOccupancyPredictionError("cuobjdump not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise StaticOccupancyPredictionError(
            f"cuobjdump failed (rc={exc.returncode}): {exc.stderr.strip()[:200]}"
        ) from exc
    return proc.stdout


def build_occupancy_prediction(
    res_usage_text: str,
    arch: str,
    block_dim: int,
    *,
    kernel: str | None = None,
    dynamic_shared_bytes: int = 0,
    shared_mem_carveout_bytes: int | None = None,
    slug: str | None = None,
    shape_point: str | None = None,
    binary_sha256: str | None = None,
    kernel_signature_sha256: str | None = None,
    table=None,
) -> dict:
    """Build the static occupancy-limit prediction artifact from -res-usage text.

    The per-thread register count and static shared bytes are taken from the
    parsed ``cuobjdump -res-usage`` row for ``arch`` (and ``kernel`` when the
    binary carries more than one); the launch block dimension and any dynamic
    shared memory come from the caller. Each of the four limits is emitted
    verbatim from ``compute_occupancy_limits`` (exact-or-typed-unavailable).
    """
    usages = parse_cuobjdump_res_usage(res_usage_text)
    try:
        row = select_resource_usage(usages, arch=arch, function=kernel)
    except CuobjdumpResUsageError as exc:
        raise StaticOccupancyPredictionError(str(exc)) from exc

    limits = compute_occupancy_limits(
        arch=arch,
        block_dim=block_dim,
        registers_per_thread=row.registers_per_thread,
        static_shared_bytes=row.static_shared_bytes,
        dynamic_shared_bytes=dynamic_shared_bytes,
        table=table,
        shared_mem_carveout_bytes=shared_mem_carveout_bytes,
    )

    predictions: dict[str, dict] = {}
    for name in LAUNCH_OCCUPANCY_LIMIT_METRICS:
        cell = limits[name]
        predictions[name] = {
            "value": cell["value"],
            "status": cell["status"],
            "reason": cell["reason"],
            "unit": "count",
        }

    artifact: dict = {
        "schema_version": _PREDICTION_SCHEMA_VERSION,
        "artifact_kind": _ARTIFACT_KIND,
        "claim_class": _CLAIM_CLASS,
        "arch": arch,
        "kernel": row.function,
        "block_dim": block_dim,
        "registers_per_thread": row.registers_per_thread,
        "static_shared_bytes": row.static_shared_bytes,
        "dynamic_shared_bytes": dynamic_shared_bytes,
        "predictions": predictions,
    }
    # Optional identity fields are included only when supplied (the schema marks
    # them optional), so a hermetic text-driven call need not fabricate them.
    for key, val in (
        ("shared_mem_carveout_bytes", shared_mem_carveout_bytes),
        ("slug", slug),
        ("shape_point", shape_point),
        ("binary_sha256", binary_sha256),
        ("kernel_signature_sha256", kernel_signature_sha256),
    ):
        if val is not None:
            artifact[key] = val
    return artifact


def emit(artifact: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return out_path


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--binary", help="built kernel runner binary to inspect")
    src.add_argument(
        "--res-usage-file", help="pre-captured cuobjdump -res-usage text file"
    )
    ap.add_argument("--arch", required=True, help="architecture key, e.g. sm_86")
    ap.add_argument("--block-dim", type=int, required=True, help="threads per block")
    ap.add_argument("--kernel", default=None, help="kernel function (disambiguator)")
    ap.add_argument("--dynamic-shared-bytes", type=int, default=0)
    ap.add_argument(
        "--shared-mem-carveout-bytes",
        type=int,
        default=None,
        help="configured shared-memory carveout (launch__shared_mem_config_size); "
        "without it launch__occupancy_limit_shared_mem is typed_unavailable",
    )
    ap.add_argument("--slug", default=None)
    ap.add_argument("--shape-point", default=None)
    ap.add_argument("--binary-sha256", default=None)
    ap.add_argument("--kernel-signature-sha256", default=None)
    ap.add_argument("--out", required=True, help="output artifact path")
    args = ap.parse_args(argv)

    if args.binary is not None:
        text = run_cuobjdump_res_usage(args.binary, args.arch)
    else:
        text = Path(args.res_usage_file).read_text(encoding="utf-8")

    artifact = build_occupancy_prediction(
        text,
        arch=args.arch,
        block_dim=args.block_dim,
        kernel=args.kernel,
        dynamic_shared_bytes=args.dynamic_shared_bytes,
        shared_mem_carveout_bytes=args.shared_mem_carveout_bytes,
        slug=args.slug,
        shape_point=args.shape_point,
        binary_sha256=args.binary_sha256,
        kernel_signature_sha256=args.kernel_signature_sha256,
    )
    written = emit(artifact, Path(args.out))
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
