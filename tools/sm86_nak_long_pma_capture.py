#!/usr/bin/env python3
"""Capture a long independently generated NAK body through public PMA.

This produces aggregate PC/stall observations only. The PMA record format has
no physical pipe or issue-slot field; lengthening the body tests sampling
coverage, not hardware-route identifiability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=256)
    parser.add_argument("--direct-pma", action="store_true",
                        help="also enable tinygrad's direct RM/PMA stream")
    args = parser.parse_args()
    if args.rounds <= 0:
        raise SystemExit("--rounds must be positive")
    sys.path.insert(0, "/tmp/tinygrad")
    from tinygrad.tensor import Tensor  # type: ignore
    from tinygrad.helpers import Context  # type: ignore
    from tinygrad.device import Compiled, Device  # type: ignore
    from tinygrad.runtime.ops_nv import ProfilePMAEvent  # type: ignore
    from tinygrad.runtime.support.compiler_mesa import NAKCompiler  # type: ignore

    args.output.mkdir(parents=True, exist_ok=True)
    original_compile = NAKCompiler.compile_cached
    def capture_compile(self, source):
        library = original_compile(self, source)
        (args.output / "nak_shader.bin").write_bytes(library)
        return library
    NAKCompiler.compile_cached = capture_compile
    with Context(DEV="NVK+NV:NAK", DEBUG=0, CACHELEVEL=0, PARALLEL=0, CCACHE=0,
                 PROFILE=1, PMA=1 if args.direct_pma else 0):
        x = Tensor.arange(1024).to("NV", force=True)
        y = x
        for index in range(args.rounds):
            # Keep data-dependent integer operations that NAK cannot fold into
            # one multiply/add. This is a sampling-coverage body, not a
            # correctness workload.
            left = (index % 5) + 1
            right = (index % 3) + 1
            y = (y.maximum(x + index) ^ (y << left)) + (x >> right)
        started = time.perf_counter()
        y.realize()
        Device["NV"].synchronize()
        elapsed = time.perf_counter() - started
        events = []
        for index, event in enumerate(Compiled.profile_events):
            if isinstance(event, ProfilePMAEvent):
                blob = bytes(event.blob)
                path = args.output / f"pma_{index}.bin"
                path.write_bytes(blob)
                events.append({"index": index, "path": str(path), "bytes": len(blob),
                               "records_8byte": len(blob) // 8})
        result = {
            "schema": "sm86_nak_long_pma_capture_v1",
            "rounds": args.rounds,
            "host_elapsed_seconds": elapsed,
            "events": events,
            "event_count": len(events),
            "coverage_contract": "long-body sampling coverage only; no pipe/issue-slot identity",
            "pma_payload_is_observation_only": True,
        }
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
