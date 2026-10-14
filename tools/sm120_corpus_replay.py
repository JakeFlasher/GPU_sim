#!/usr/bin/env python3
"""Replay every captured trace, full first and fast second, without NCU inputs.

Outputs are zstd-compressed native simulator reports (`*.jsonl.zst`), one row
per recursive trace directory, including failures. Static sidecars are derived
from the captured metadata by the same transport adapter used by
sm120_replay_bench.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import tempfile
import time

from sm120_replay_bench import (
    DEFAULT_CARD,
    DEFAULT_SIM,
    discover_captures,
    load_static_metadata,
    synthesize_static_sidecar,
)
from jsonl_io import jsonl_compression_description, open_jsonl_writer


_CONTAMINATING_ENV_PREFIXES = ("NVBIT_", "INJECTION_", "CUDA_INJECTION")


def _clean_replay_environment() -> tuple[dict[str, str], list[str]]:
    """Remove tracer/preload controls from simulator child environments."""
    environment = os.environ.copy()
    removed: list[str] = []
    for key in list(environment):
        if key == "LD_PRELOAD" or key.startswith(_CONTAMINATING_ENV_PREFIXES):
            removed.append(key)
            environment.pop(key, None)
    return environment, sorted(removed)


def _reject_nonfinite_json(value: str):
    raise ValueError(f"non-finite JSON constant is not permitted: {value}")


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    # The session leader can exit while a descendant remains alive.  Always
    # signal the process group; checking poll() first would leak descendants.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _run_simulator(command: list[str], stdout_path: Path, stderr, timeout: float,
                   max_output_bytes: int, environment: dict[str, str]) -> tuple[int | None, bool, bool]:
    """Run one simulator with a streaming, bounded stdout capture.

    The old implementation redirected stdout to a file and only checked its
    size after the child exited.  A malformed child could therefore fill the
    filesystem (and ``read_bytes`` could recreate the same memory spike).  A
    selector drains the pipe incrementally, retains at most ``limit + 1``
    bytes for diagnostics, and kills the entire child session as soon as the
    limit or timeout is reached.
    """
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=stderr,
        env=environment,
        start_new_session=True,
    )
    assert process.stdout is not None
    stream = process.stdout
    selector = selectors.DefaultSelector()
    selector.register(stream, selectors.EVENT_READ)
    total = 0
    exceeded = False
    timed_out = False
    deadline = time.monotonic() + timeout
    try:
        with stdout_path.open("wb") as output:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    _kill_process_group(process)
                    break
                events = selector.select(min(0.1, remaining))
                if events:
                    chunk = os.read(stream.fileno(), 1024 * 1024)
                    if chunk:
                        if total <= max_output_bytes:
                            keep = min(len(chunk), max_output_bytes + 1 - total)
                            if keep:
                                output.write(chunk[:keep])
                        total += len(chunk)
                        if total > max_output_bytes:
                            exceeded = True
                            _kill_process_group(process)
                            # Drain only what is already available, then leave
                            # the group killed.  No unbounded output is kept.
                    else:
                        break
                elif process.poll() is not None:
                    # The selector can briefly report no event at EOF; drain
                    # once more before considering the child complete.
                    chunk = os.read(stream.fileno(), 1024 * 1024)
                    if not chunk:
                        break
                    if total <= max_output_bytes:
                        keep = min(len(chunk), max_output_bytes + 1 - total)
                        if keep:
                            output.write(chunk[:keep])
                    total += len(chunk)
                    if total > max_output_bytes:
                        exceeded = True
                        _kill_process_group(process)
                        break
            if timed_out or exceeded:
                # Ensure the process is reaped even if it exited between the
                # selector event and the containment signal.
                if process.poll() is None:
                    _kill_process_group(process)
            else:
                process.wait()
    except BaseException:
        _kill_process_group(process)
        raise
    finally:
        try:
            selector.unregister(stream)
        except (KeyError, ValueError):
            pass
        selector.close()
        stream.close()
    return (None if timed_out else process.returncode), timed_out, exceeded


def replay(capture: Path, relative: str, control: Path, args, mode: str, logs: Path) -> dict:
    memory_mode = args.memory_mode if mode == "full" else args.fast_memory_mode
    command = [str(args.sim_bin), "--trace", str(capture), "--control", str(control),
               "--arch-card", str(args.arch_card), "--sm120-alpha", "--mode",
               str(memory_mode)]
    if args.sm120_effective_l1 and memory_mode != "flat":
        command.extend(("--sm120-effective-l1", args.sm120_effective_l1))
    if args.sm120_effective_l2 and memory_mode != "flat":
        command.extend(("--sm120-effective-l2", args.sm120_effective_l2))
    if args.carded_dram and mode == "full":
        command.append("--carded-dram")
    if args.ncu_shaped_metrics and mode == "full":
        command.append("--ncu-shaped-metrics")
    if mode == "fast":
        command.append("--fast-frontier-only")
    stderr_path = logs / (relative + ".stderr")
    started = time.perf_counter_ns()
    environment, removed_environment = _clean_replay_environment()
    row: dict = {}
    stdout_path: Path | None = None
    try:
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix="sim.stdout.", suffix=".tmp",
                                    dir=str(stderr_path.parent))
        os.close(fd)
        stdout_path = Path(name)
        with stderr_path.open("x") as stderr:
            returncode, timed_out, exceeded = _run_simulator(
                command, stdout_path, stderr, args.timeout,
                args.max_sim_output_bytes, environment)
        if timed_out:
            row = {"status": "timeout", "returncode": None}
        elif exceeded:
            row = {
                "status": "output_error",
                "detail": f"simulator stdout exceeds {args.max_sim_output_bytes} bytes",
                "returncode": returncode,
            }
        else:
            try:
                with stdout_path.open("r", encoding="utf-8") as stream:
                    row = json.load(stream, parse_constant=_reject_nonfinite_json)
                if not isinstance(row, dict):
                    raise ValueError("simulator output is not an object")
            except (ValueError, UnicodeError, OSError) as error:
                row = {"status": "output_error", "detail": str(error)}
            if returncode:
                row["status"] = "replay_error"
            else:
                row.setdefault("status", "ok")
            row["returncode"] = returncode
        if row["status"] != "ok" and stdout_path.is_file():
            diagnostic = stderr_path.with_suffix(".stdout")
            os.replace(stdout_path, diagnostic)
    except OSError as error:
        row = {"status": "launch_error", "detail": str(error), "returncode": None}
    except Exception as error:
        row = {"status": "runner_error", "detail": str(error), "returncode": None}
    finally:
        if stdout_path is not None:
            try:
                stdout_path.unlink(missing_ok=True)
            except OSError:
                # Preserve the original row; cleanup failure is diagnostic and
                # must not turn one capture into an uncaught pool exception.
                pass
        wall_ns = time.perf_counter_ns() - started
    row.update(canonical_capture=relative, capture_dir=str(capture),
               driver_wall_ms=wall_ns / 1_000_000,
               replay_environment_removed=removed_environment)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, default=Path("/root/sm120_rtx_5090_capture"))
    parser.add_argument("--capture", action="append", default=[],
                        help="select a real capture by relative path or name; repeat for a prototype batch")
    parser.add_argument("--output", type=Path, required=True, help="new output directory outside the corpus")
    parser.add_argument("--sim-bin", type=Path, default=DEFAULT_SIM)
    parser.add_argument("--arch-card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--mode", choices=("both", "full", "fast"), default="both")
    parser.add_argument("--memory-mode", choices=("flat", "l1", "l1-l2"), default="flat",
                        help="simulator memory mode passed to every replay")
    parser.add_argument("--fast-memory-mode", choices=("flat",), default="flat",
                        help="complementary fast-stage mode; fast SM120 stage is flat-only")
    parser.add_argument("--sm120-effective-l1",
                        help="effective L1 profile CAPACITY:MSHR|none:HIT:MISS")
    parser.add_argument("--sm120-effective-l2",
                        help="effective L2 scenario CAPACITY_BYTES:SLICES:ASSOCIATIVITY (opt-in l1-l2 only)")
    parser.add_argument("--carded-dram", action="store_true",
                        help="enable the architecture-card DRAM path (for l1-l2 mode)")
    parser.add_argument("--ncu-shaped-metrics", action="store_true",
                        help="emit typed NCU metric projection for full mode (never fast mode)")
    parser.add_argument("--timeout", type=float, default=1200, help="seconds per complete capture replay")
    parser.add_argument("--max-sim-output-bytes", type=int, default=512 * 1024 * 1024,
                        help="bounded simulator stdout retained by the driver")
    parser.add_argument("--workers", type=int, default=12,
                        help="concurrent simulator processes (must be exactly 12)")
    args = parser.parse_args()
    args.capture_root = args.capture_root.resolve()
    args.output = args.output.resolve()
    args.sim_bin = args.sim_bin.resolve()
    args.arch_card = args.arch_card.resolve()
    if not 0 < args.timeout < float("inf"):
        parser.error("--timeout must be positive and finite")
    if args.max_sim_output_bytes <= 0:
        parser.error("--max-sim-output-bytes must be positive")
    if args.workers != 12:
        parser.error("--workers must be exactly 12 for every corpus replay")
    if args.sm120_effective_l2 and args.mode == "fast":
        parser.error("--sm120-effective-l2 applies only to the full l1-l2 stage")
    if args.sm120_effective_l2 and args.memory_mode != "l1-l2":
        parser.error("--sm120-effective-l2 requires --memory-mode l1-l2")
    if args.carded_dram and args.mode == "fast":
        parser.error("--carded-dram applies only to the full l1-l2 stage")
    if args.carded_dram and args.memory_mode != "l1-l2":
        parser.error("--carded-dram requires --memory-mode l1-l2")
    if args.output.is_relative_to(args.capture_root):
        parser.error("outputs must be outside the raw capture corpus")
    captures = discover_captures(args.capture_root, args.capture)
    if not captures:
        parser.error("no recursive captures found")
    args.output.mkdir(parents=True, exist_ok=False)
    controls = args.output / "controls"
    prepared = []
    for capture in captures:
        relative = capture.relative_to(args.capture_root / "captures").as_posix()
        control = controls / (relative + ".control_bits.json")
        error = None
        try:
            metadata = load_static_metadata(capture)
            if metadata is None:
                raise ValueError("native static metadata missing")
            control.parent.mkdir(parents=True, exist_ok=True)
            synthesize_static_sidecar(capture, metadata, control)
        except Exception as failure:
            error = str(failure)
        prepared.append((capture, relative, control, error))
    modes = ("full", "fast") if args.mode == "both" else (args.mode,)
    summary = {"capture_root": str(args.capture_root), "sim_bin": str(args.sim_bin),
               "arch_card": str(args.arch_card), "captures_discovered": len(captures),
               "scope": ("explicitly selected real-capture prototype batch" if args.capture else
                         "all recursively discovered traces; none selected by replay outcome"),
               "selected_captures": args.capture,
               "memory_mode": args.memory_mode,
               "fast_memory_mode": args.fast_memory_mode,
               "sm120_effective_l1": args.sm120_effective_l1,
               "sm120_effective_l2": args.sm120_effective_l2,
               "carded_dram": bool(args.carded_dram),
               "ncu_shaped_metrics": bool(args.ncu_shaped_metrics and "full" in modes),
               "ncu_shaped_metrics_requested": bool(args.ncu_shaped_metrics),
               "workers": args.workers,
               "ncu_used_as_input": False, "modes": {}}
    for mode in modes:
        started = time.perf_counter_ns()
        counts = Counter()
        launches = 0
        with open_jsonl_writer(args.output / f"{mode}.jsonl") as output:
            def run_one(item):
                capture, relative, control, error = item
                return ({"canonical_capture": relative, "capture_dir": str(capture),
                         "status": "control_error", "detail": error} if error else
                        replay(capture, relative, control, args, mode,
                               args.output / f"{mode}_logs"))

            with ThreadPoolExecutor(max_workers=args.workers,
                                    thread_name_prefix=f"replay-{mode}") as pool:
                # Keep only twelve futures/results live.  ``pool.map`` retains
                # completed out-of-order rows behind the first slow capture;
                # that defeated the stdout bound for large JSON documents.
                pending = {}
                prepared_iter = iter(prepared)
                for _ in range(args.workers):
                    try:
                        item = next(prepared_iter)
                    except StopIteration:
                        break
                    pending[pool.submit(run_one, item)] = item
                index = 0
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        item_for_future = pending.pop(future, (None, "", None, None))
                        index += 1
                        try:
                            row = future.result()
                        except Exception as failure:
                            capture, relative, _control, _error = item_for_future
                            row = {"canonical_capture": relative,
                                   "capture_dir": str(capture) if capture else "",
                                   "status": "runner_error", "detail": str(failure)}
                        try:
                            item = next(prepared_iter)
                        except StopIteration:
                            item = None
                        if item is not None:
                            pending[pool.submit(run_one, item)] = item
                        output.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                        output.flush()
                        counts[row["status"]] += 1
                        mode_rows = row.get("launches" if mode == "full" else "fast_frontier", [])
                        if isinstance(mode_rows, list):
                            launches += len(mode_rows)
                        print(f"{mode} {index}/{len(captures)} {row['canonical_capture']}: {row['status']}", flush=True)
        summary["modes"][mode] = {
            "status_counts": dict(counts),
            "launches": launches,
            "driver_total_wall_ns": time.perf_counter_ns() - started,
            "jsonl_artifact": str(args.output / f"{mode}.jsonl.zst"),
            "jsonl_compression": jsonl_compression_description(),
            "ncu_shaped_metrics_applied": bool(args.ncu_shaped_metrics and mode == "full"),
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return int(any(set(row["status_counts"]) != {"ok"} for row in summary["modes"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
