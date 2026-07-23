"""Operator-facing hardware-metadata capture CLI.

The operator runs this CLI on the local host (typically the
RTX 5060 laptop after a cross-host transfer, but it works on
either host) to capture the observed nvidia-smi state required
by the architecture-coupling discipline. The tool:

1. Resolves nvidia-smi (default: PATH lookup; override via
   ``--nvidia-smi <path>``).
2. Runs the canonical query string verbatim:
   ``nvidia-smi --query-gpu=name,driver_version,compute_cap,
   clocks.sm,clocks.mem,clocks_throttle_reasons.active
   --format=csv,noheader``.
3. Parses the CSV output line into structured fields.
4. Resolves nvcc (default: PATH lookup; override via
   ``--nvcc <path>``) and runs ``nvcc --version`` to capture
   the CUDA toolkit ``release X.Y`` substring. nvcc absence /
   non-zero / unparseable output records the literal sentinel
   ``"unavailable"`` (capture continues; the toolkit probe is
   informational and not gating).
5. Captures ``source_commit`` from ``git rev-parse HEAD`` per
   the post-commit recapture discipline.
6. Computes ``host_arch`` from ``compute_cap`` (e.g., ``"8.6"``
   → ``"sm_86"``, ``"12.0"`` → ``"sm_120"``).
7. Writes both a JSON sidecar (the structured capture) and a
   CSV sidecar (the verbatim nvidia-smi output) atomically via
   a tmp + rename pattern (``_atomic_write`` helper) to a
   timestamped path under
   ``data/local_labels/sweeps/clock_throttle_evidence/``.

Optional ``--expected-arch sm_86|sm_120`` flag verifies the
host's reported arch matches the operator's expectation; on
mismatch the tool exits 1 without writing the captures.

CLI:

  python3 tools/local_labels/capture_observed_state.py \
      [--nvidia-smi <path>] \
      [--nvcc <path>] \
      [--out-prefix <path>] \
      [--expected-arch sm_86|sm_120] \
      [--quiet]

Exit codes:
- 0 on successful capture (and matching --expected-arch if set).
- 1 on --expected-arch mismatch (the host reports a different arch).
- 2 on schema/io error (nvidia-smi unavailable, malformed CSV
  output, nvidia-smi non-zero rc, or unparseable compute_cap).

The JSON output schema (frozen):

  {
    "schema_version": 1,
    "captured_at_utc": "<ISO-8601-UTC>",
    "source_commit": "<git rev-parse HEAD>",
    "host_arch": "sm_<XY>",
    "cuda_toolkit_version": "<X.Y>" | "unavailable",
    "captured_command": "<verbatim nvidia-smi command>",
    "captured_output": "<verbatim CSV line>",
    "fields": {
      "name": "<device name>",
      "driver_version": "<driver version>",
      "compute_cap": "<X.Y>",
      "clocks_sm_mhz": <int>,
      "clocks_mem_mhz": <int>,
      "clocks_throttle_reasons_active_hex": "<0xHHH...>"
    }
  }
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_DEFAULT_OUT_PREFIX = (
    _REPO_ROOT / "data" / "local_labels" / "sweeps"
    / "clock_throttle_evidence" / "observed_state"
)
_NVIDIA_SMI_QUERY = (
    "name,driver_version,compute_cap,clocks.sm,clocks.mem,"
    "clocks_throttle_reasons.active"
)


def _diag(msg: str) -> None:
    sys.stderr.write(f"capture_observed_state: {msg}\n")


def _git_head(cwd: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False, capture_output=True, text=True, cwd=str(cwd),
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _arch_from_compute_cap(compute_cap: str) -> str:
    """Convert a nvidia-smi compute_cap string (e.g., ``"8.6"``)
    into the project's arch token (``"sm_86"``)."""
    digits = compute_cap.replace(".", "").strip()
    return f"sm_{digits}"


def _parse_clock_mhz(field: str) -> int:
    """Parse a clock field like ``"210 MHz"`` to the integer 210."""
    return int(field.strip().split()[0])


def _atomic_write(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically: write to a sibling
    ``<path>.tmp`` first, then ``Path.replace(path)`` to the final
    name. The replace is atomic on the same filesystem, so a
    concurrent reader never sees a half-written file with the
    final name.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def _capture_cuda_toolkit_version(nvcc_path: Path | None) -> str:
    """Probe nvcc and return the parsed ``release X.Y`` substring,
    or the literal sentinel ``"unavailable"`` if nvcc is missing,
    fails, or returns unparseable output.

    The probe is read-only (``nvcc --version``) and never gates
    capture: nvcc is not always present on capture hosts (e.g., a
    minimal-deps image with the driver but no toolkit), and the
    operator's hardware-metadata sidecar should still be produced
    in that case with the field clearly marked unavailable."""
    if nvcc_path is None or not nvcc_path.is_file():
        return "unavailable"
    try:
        proc = subprocess.run(
            [str(nvcc_path), "--version"],
            check=False, capture_output=True, text=True, timeout=15,
        )
    except (subprocess.TimeoutExpired, OSError):
        return "unavailable"
    if proc.returncode != 0:
        return "unavailable"
    match = re.search(
        r"release\s+(\d+\.\d+)", proc.stdout, flags=re.IGNORECASE,
    )
    if match is None:
        return "unavailable"
    return match.group(1)


def _parse_csv_line(line: str) -> dict:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 6:
        raise ValueError(
            f"expected 6 comma-separated fields in nvidia-smi output; "
            f"got {len(parts)}: {parts!r}"
        )
    name, driver_version, compute_cap, sm_clk, mem_clk, throttle_hex = parts
    return {
        "name": name,
        "driver_version": driver_version,
        "compute_cap": compute_cap,
        "clocks_sm_mhz": _parse_clock_mhz(sm_clk),
        "clocks_mem_mhz": _parse_clock_mhz(mem_clk),
        "clocks_throttle_reasons_active_hex": throttle_hex,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture the local host's nvidia-smi observed state "
            "(name, driver, compute_cap, clocks, throttle reasons) "
            "to the canonical evidence prefix under "
            "data/local_labels/sweeps/clock_throttle_evidence/."
        ),
    )
    parser.add_argument(
        "--nvidia-smi",
        type=Path,
        default=None,
        help=(
            "Path to nvidia-smi (default: resolve via PATH). "
            "Useful for testing with a synthetic shim."
        ),
    )
    parser.add_argument(
        "--nvcc",
        type=Path,
        default=None,
        help=(
            "Path to nvcc (default: resolve via PATH). The tool "
            "runs `nvcc --version` to capture the CUDA toolkit "
            "release substring; absence / non-zero / unparseable "
            "output is recorded as the literal sentinel "
            "'unavailable' and does NOT gate capture."
        ),
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=_DEFAULT_OUT_PREFIX,
        help=(
            f"Output prefix (default: {_DEFAULT_OUT_PREFIX!s}). "
            f"The tool appends '_<UTC-timestamp>.json' and "
            f"'_<UTC-timestamp>.csv' to this prefix."
        ),
    )
    parser.add_argument(
        "--expected-arch",
        choices=("sm_86", "sm_120"),
        default=None,
        help=(
            "If set, verify the host's reported arch matches this "
            "value. Mismatch exits 1 without writing captures."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Emit only the verdict token on stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.nvidia_smi is not None:
        nvidia_smi: Path | None = args.nvidia_smi
        if not nvidia_smi.is_file():
            _diag(f"nvidia-smi not found at {nvidia_smi!s}")
            return 2
    else:
        resolved = shutil.which("nvidia-smi")
        if resolved is None:
            _diag(
                "nvidia-smi not found on PATH; install the NVIDIA "
                "driver tools or pass --nvidia-smi <path>"
            )
            return 2
        nvidia_smi = Path(resolved)

    cmd = [
        str(nvidia_smi),
        f"--query-gpu={_NVIDIA_SMI_QUERY}",
        "--format=csv,noheader",
    ]
    captured_command = (
        f"nvidia-smi --query-gpu={_NVIDIA_SMI_QUERY} "
        f"--format=csv,noheader"
    )

    proc = subprocess.run(
        cmd, check=False, capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        _diag(
            f"nvidia-smi at {nvidia_smi!s} exited rc={proc.returncode}; "
            f"stderr tail:\n{proc.stderr[-1500:]}"
        )
        return 2

    raw_output = proc.stdout.strip()
    if not raw_output:
        _diag("nvidia-smi produced empty output")
        return 2

    first_line = raw_output.splitlines()[0]
    try:
        fields = _parse_csv_line(first_line)
    except (ValueError, IndexError) as exc:
        _diag(f"unable to parse nvidia-smi CSV output: {exc}")
        return 2

    host_arch = _arch_from_compute_cap(fields["compute_cap"])

    if args.nvcc is not None:
        nvcc_path: Path | None = args.nvcc
    else:
        resolved_nvcc = shutil.which("nvcc")
        nvcc_path = Path(resolved_nvcc) if resolved_nvcc else None
    cuda_toolkit_version = _capture_cuda_toolkit_version(nvcc_path)

    if args.expected_arch is not None and host_arch != args.expected_arch:
        _diag(
            f"host reports {host_arch} (compute_cap="
            f"{fields['compute_cap']}), but --expected-arch="
            f"{args.expected_arch} was requested. Refusing to write "
            f"captures; the operator is on the wrong host."
        )
        return 1

    now = _utc_now()
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    iso_utc = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    payload = {
        "schema_version": 1,
        "captured_at_utc": iso_utc,
        "source_commit": _git_head(_REPO_ROOT),
        "host_arch": host_arch,
        "cuda_toolkit_version": cuda_toolkit_version,
        "captured_command": captured_command,
        "captured_output": first_line,
        "fields": fields,
    }

    out_prefix = args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_name(f"{out_prefix.name}_{timestamp}.json")
    csv_path = out_prefix.with_name(f"{out_prefix.name}_{timestamp}.csv")

    _atomic_write(
        json_path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    csv_header = (
        "name, driver_version, compute_cap, clocks.current.sm [MHz], "
        "clocks.current.memory [MHz], clocks_event_reasons.active"
    )
    _atomic_write(
        csv_path,
        csv_header + "\n" + first_line + "\n",
    )

    arch_match_note = (
        f" (matches {args.expected_arch})" if args.expected_arch else ""
    )
    verdict = (
        f"capture_observed_state: device={fields['name']} "
        f"compute_cap={fields['compute_cap']}{arch_match_note}"
    )
    if args.quiet:
        sys.stdout.write(verdict + "\n")
    else:
        sys.stdout.write(verdict + "\n")
        sys.stdout.write(
            f"capture_observed_state: wrote {json_path!s}\n"
            f"capture_observed_state: wrote {csv_path!s}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
