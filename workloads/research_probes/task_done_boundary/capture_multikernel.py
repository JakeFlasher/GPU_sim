#!/usr/bin/env python3
"""Capture a bounded multi-kernel task-done probe with the retained NVBit runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workloads.research_probes.capture_sm86_nvbit import (  # noqa: E402
    TRACER_PATH,
    capture_argv,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cell", default="task_done_store_64_multikernel")
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("workloads/research_probes/task_done_boundary/task_done_boundary_trace"),
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--target-args", nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args()
    if not args.target_args:
        parser.error("--target-args must contain the probe command")
    output_root = args.output_root.resolve()
    destination = output_root / "evidence" / "raw" / "nvbit" / args.cell
    logical_environment = {
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "CUDA_INJECTION64_PATH": str(TRACER_PATH),
        "LD_LIBRARY_PATH": ".toolchain/pb/lib",
        "USER_DEFINED_FOLDERS": "1",
        "TRACES_FOLDER": str(destination),
        "ACTIVE_FROM_START": "1",
        "SKIP_ENHANCED_TRACER": "0",
        # The task-done probe has producer, thrash, and consumer kernels.
        # Capture all three; the copy engine is represented by the host-side
        # probe record, not a synthetic CUDA kernel.
        "DYNAMIC_KERNEL_LIMIT_START": "1",
        "DYNAMIC_KERNEL_LIMIT_END": "3",
        "ENHANCED_TRACER_MAX_PARSER_INPUT_KB": "256",
        "ENHANCED_TRACER_PER_CUBIN_TIMEOUT_S": "45",
    }
    try:
        return capture_argv(
            args.cell,
            [str(args.binary.resolve()), *args.target_args],
            logical_environment,
            output_root,
            args.timeout,
            allow_multiple_launches=True,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"multi-kernel NVBit capture: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
