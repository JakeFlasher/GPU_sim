#!/usr/bin/env python3
"""Compare the custom functional backend with Vortex SimX/RTL vecadd logs."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess


def vortex_values(path: pathlib.Path) -> list[int]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("#0: dst:"):
            return [int(value) for value in line.split("dst:", 1)[1].split()]
    raise RuntimeError(f"no dst line in {path}")


def custom_values(
    binary: pathlib.Path, spec: pathlib.Path, program: pathlib.Path, width: int, dst: int
) -> list[int]:
    completed = subprocess.run(
        [str(binary), str(spec), str(program)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip())
    document = json.loads(completed.stdout)
    return [document["memory"][str(dst + index * 4)] for index in range(width)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom-binary", type=pathlib.Path, required=True)
    parser.add_argument("--spec", type=pathlib.Path, required=True)
    parser.add_argument("--program", type=pathlib.Path, required=True)
    parser.add_argument("--simx-log", type=pathlib.Path, required=True)
    parser.add_argument("--rtlsim-log", type=pathlib.Path, required=True)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--dst", type=int, default=1200)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.width <= 64:
        parser.error("--width must be in 1..=64")
    if not 0 <= args.dst <= 0xFFFF_FFFF or args.dst + 4 * (args.width - 1) + 3 > 0xFFFF_FFFF:
        parser.error("--dst plus the lane word span must fit the u32 byte-address domain")
    custom = custom_values(args.custom_binary, args.spec, args.program, args.width, args.dst)
    simx = vortex_values(args.simx_log)
    rtlsim = vortex_values(args.rtlsim_log)
    if any(len(values) != args.width for values in (custom, simx, rtlsim)):
        parser.error("custom, SimX, and rtlsim outputs must have exactly --width values")
    result = {
        "kind": "custom_vortex_vecadd_functional_differential",
        "custom": custom,
        "vortex_simx": simx,
        "vortex_rtlsim": rtlsim,
        "custom_matches_simx": custom == simx,
        "custom_matches_rtlsim": custom == rtlsim,
        "simx_matches_rtlsim": simx == rtlsim,
        "interpretation": "functional output parity only; no cycle or silicon claim",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not all(
        (result["custom_matches_simx"], result["custom_matches_rtlsim"], result["simx_matches_rtlsim"])
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
