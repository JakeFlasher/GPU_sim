#!/usr/bin/env python3
"""Replay every captured trace, full first and fast second, without NCU inputs.

Outputs are zstd-compressed native simulator reports (`*.jsonl.zst`), one row
per recursive trace directory, including failures. Static sidecars are derived
from the captured metadata by the architecture-neutral transport adapter used
by ``sm120_replay_bench`` (the historical module name is retained for old
SM120 report tooling).
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
    DEFAULT_SIM,
    REPO_ROOT,
    card_architecture,
    discover_captures,
    load_static_metadata,
    metadata_architecture,
    metadata_architectures,
    synthesize_static_sidecar,
)
from jsonl_io import jsonl_compression_description, open_jsonl_writer


_CONTAMINATING_ENV_PREFIXES = ("NVBIT_", "INJECTION_", "CUDA_INJECTION")

# This checkout is the dedicated SM86/RTX 3090 Ti lane.  Keep the historical
# SM120 constants imported above for old callers, but make the runner's
# parameterized defaults target the active corpus and architecture card.
DEFAULT_SM86_WORK_ROOT = REPO_ROOT.parent / "sm86_rtx_3090_capture"
DEFAULT_SM86_CARD = REPO_ROOT / "arch_profiles/cards/sm_86.json"
DEFAULT_SM86_PCHASE_CARD = REPO_ROOT / "data/pchase/sm_86_static_card.json"
DEFAULT_SM86_MEMORY_ADMISSION_FEEDBACK = (
    REPO_ROOT / "arch_profiles/candidates/sm86_store_fence_candidate_v1.json"
)


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


def replay(capture: Path, relative: str, control: Path, args, mode: str, logs: Path,
           scope_override: str | None = None) -> dict:
    memory_mode = args.memory_mode if mode == "full" else args.fast_memory_mode
    cache_scope = scope_override or args.cache_scope
    command = [str(args.sim_bin), "--trace", str(capture), "--control", str(control),
               "--arch-card", str(args.arch_card)]
    if args.target_arch == "sm_86" and args.pchase_card and memory_mode != "flat":
        command.extend(("--pchase-card", str(args.pchase_card)))
    command.extend(("--mode", str(memory_mode)))
    if args.target_arch == "sm_120":
        command.append("--sm120-alpha")
    if args.sm120_effective_l1 and args.target_arch == "sm_120" and memory_mode != "flat":
        command.extend(("--sm120-effective-l1", args.sm120_effective_l1))
    if args.sm120_effective_l2 and args.target_arch == "sm_120" and memory_mode == "l1-l2":
        command.extend(("--sm120-effective-l2", args.sm120_effective_l2))
    if args.carded_dram_enabled and mode == "full":
        command.append("--carded-dram")
        if args.dram_service_rate_bytes_per_sec is not None:
            command.extend(("--dram-service-rate-bytes-per-sec", str(args.dram_service_rate_bytes_per_sec)))
    if mode == "full":
        command.extend(("--cache-scope", cache_scope))
        command.append("--ncu-shaped-metrics")
        if args.memory_admission_feedback:
            command.extend(("--memory-admission-feedback", str(args.memory_admission_feedback)))
    if mode == "fast":
        command.append("--fast-frontier-only")
    suffix = ".fallback-launch" if scope_override else ""
    stderr_path = logs / (relative + suffix + ".stderr")
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
    if (
        scope_override is None
        and mode == "full"
        and args.cache_scope == "stream-l2"
        and row.get("status") != "ok"
    ):
        try:
            detail = stderr_path.read_text(encoding="utf-8")
        except OSError:
            detail = ""
        if "--cache-scope stream-l2 requires every launch in the trace to belong to one device and stream" in detail:
            fallback = replay(capture, relative, control, args, mode, logs, "launch")
            fallback["requested_cache_scope"] = "stream-l2"
            fallback["cache_scope_fallback"] = "launch"
            fallback["cache_scope_fallback_reason"] = "capture contains multiple device/stream identities"
            return fallback
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_SM86_WORK_ROOT,
                        help="capture corpus root (recursive captures/ tree)")
    parser.add_argument("--capture", action="append", default=[],
                        help="select a real capture by relative path or name; repeat for a prototype batch")
    parser.add_argument("--output", type=Path, required=True, help="new output directory outside the corpus")
    parser.add_argument("--sim-bin", type=Path, default=DEFAULT_SIM)
    parser.add_argument("--arch-card", type=Path, default=DEFAULT_SM86_CARD)
    parser.add_argument("--pchase-card", type=Path, default=DEFAULT_SM86_PCHASE_CARD,
                        help="SM86 P-chase geometry card used by full L1/L2 replay")
    parser.add_argument("--target-arch", choices=("auto", "sm_86", "sm_120"), default="auto",
                        help="architecture deployment token; auto resolves from --arch-card")
    parser.add_argument("--mode", choices=("both", "full", "fast"), default="both")
    parser.add_argument("--memory-mode", choices=("l1-l2", "l1", "flat"), default="l1-l2",
                        help="full replay memory path (full defaults to integrated L1+L2)")
    parser.add_argument("--cache-scope", choices=("launch", "stream-l2"), default="launch",
                        help="full replay L2 lifetime: launch-cold or persistent ordered stream")
    parser.add_argument("--fast-memory-mode", choices=("flat",), default="flat",
                        help="complementary fast-stage mode; fast SM120 stage is flat-only")
    parser.add_argument("--sm120-effective-l1",
                        help="override the default effective L1 parameters: CAPACITY:MSHR|none:HIT:MISS")
    parser.add_argument("--sm120-effective-l2",
                        help="override the default effective L2 parameters: CAPACITY_BYTES:SLICES:ASSOCIATIVITY")
    parser.add_argument("--carded-dram", action=argparse.BooleanOptionalAction, default=None,
                        help="enable/disable the architecture-card DRAM path (full SM86 defaults enabled)")
    parser.add_argument("--dram-service-rate-bytes-per-sec", type=int, default=None,
                        help="explicit effective DRAM service-rate scenario; SM86 full defaults to the independently measured 825 GB/s hypothesis and records the override")
    parser.add_argument("--memory-admission-feedback", type=Path, default=None,
                        help="override the default launch-scoped SM86 store/fence admission+completion candidate; stream-l2 suppresses this arm; static gate remains enforced in sim-sm")
    parser.add_argument("--no-memory-admission-feedback", action="store_true",
                        help="explicit SM86 ablation: do not auto-select the launch-scoped store/fence candidate")
    # Accepted for old command lines; full replay always emits these projections.
    parser.add_argument("--ncu-shaped-metrics", action="store_true", default=True,
                        help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=float, default=1200, help="seconds per complete capture replay")
    parser.add_argument("--max-sim-output-bytes", type=int, default=512 * 1024 * 1024,
                        help="bounded simulator stdout retained by the driver")
    parser.add_argument("--workers", type=int, default=12,
                        help="concurrent simulator processes (must be exactly 12)")
    args = parser.parse_args(argv)
    args.capture_root = args.capture_root.resolve()
    args.output = args.output.resolve()
    args.sim_bin = args.sim_bin.resolve()
    args.arch_card = args.arch_card.resolve()
    args.pchase_card = args.pchase_card.resolve() if args.pchase_card else None
    try:
        card_arch = card_architecture(args.arch_card)
    except ValueError as error:
        parser.error(str(error))
    if args.target_arch != "auto" and args.target_arch != card_arch:
        parser.error(f"--target-arch {args.target_arch} conflicts with architecture card {card_arch}")
    args.target_arch = card_arch
    args.memory_admission_feedback_defaulted = False
    args.memory_admission_feedback_scope_suppressed = False
    args.memory_admission_feedback_disabled = bool(args.no_memory_admission_feedback)
    # Full replay is the comprehensive path.  The separate fast stage is the
    # only flat-memory ablation; do not let a forgotten flag silently turn a
    # full SM86 run into the old flat baseline.
    if args.mode in ("both", "full") and args.memory_mode != "l1-l2":
        parser.error("full replay requires --memory-mode l1-l2; flat/l1 are fast-stage-only ablations")
    args.carded_dram_enabled = args.carded_dram if args.carded_dram is not None else args.target_arch == "sm_86"
    args.dram_service_rate_requested = args.dram_service_rate_bytes_per_sec
    args.dram_service_rate_defaulted = False
    # Independent repeated RTX 3090 Ti memtest runs support an effective
    # forced-DRAM warm-rate scenario near 825 GB/s (roughly [810,850] under
    # the unlocked protocol). Promote the absolute-rate midpoint only for the
    # SM86 full path; the simulator maps it onto the locked-card SM clock for
    # cycle pricing. Retain the vendor 1008 GB/s card value as reference and
    # record that the scenario was defaulted rather than supplied by the caller.
    if (
        args.dram_service_rate_bytes_per_sec is None
        and args.target_arch == "sm_86"
        and args.carded_dram_enabled
        and args.mode in ("both", "full")
    ):
        args.dram_service_rate_bytes_per_sec = 825_000_000_000
        args.dram_service_rate_defaulted = True
    if args.no_memory_admission_feedback and args.memory_admission_feedback is not None:
        parser.error("--no-memory-admission-feedback cannot be combined with --memory-admission-feedback")
    if args.memory_admission_feedback is not None and args.cache_scope == "stream-l2":
        parser.error(
            "--memory-admission-feedback currently requires --cache-scope launch; "
            "stream-l2 has no launch-scoped store/fence feedback arm"
        )
    if (
        args.memory_admission_feedback is None
        and not args.no_memory_admission_feedback
        and args.target_arch == "sm_86"
        and args.mode in ("both", "full")
    ):
        if args.cache_scope == "launch":
            args.memory_admission_feedback = DEFAULT_SM86_MEMORY_ADMISSION_FEEDBACK
            args.memory_admission_feedback_defaulted = True
        else:
            # The direct runner deliberately rejects this timing-active arm
            # for persistent stream-L2 scope.  Keep stream-L2 usable as an
            # explicit cache-lifetime ablation and record the scoped gap
            # instead of emitting replay_error rows for every launch.
            args.memory_admission_feedback_scope_suppressed = True
    if args.memory_admission_feedback is not None and not args.memory_admission_feedback.is_file():
        parser.error(f"memory-admission feedback does not exist: {args.memory_admission_feedback}")
    if not 0 < args.timeout < float("inf"):
        parser.error("--timeout must be positive and finite")
    if args.max_sim_output_bytes <= 0:
        parser.error("--max-sim-output-bytes must be positive")
    if args.workers not in (1, 12):
        parser.error("--workers must be 1 (sequential timing) or 12 (standard corpus replay)")
    if args.sm120_effective_l2 and args.mode == "fast":
        parser.error("--sm120-effective-l2 applies only to the full l1-l2 stage")
    if args.sm120_effective_l2 and args.memory_mode != "l1-l2":
        parser.error("--sm120-effective-l2 requires --memory-mode l1-l2")
    if args.target_arch != "sm_120" and (args.sm120_effective_l1 or args.sm120_effective_l2):
        parser.error("--sm120-effective-l1/--sm120-effective-l2 require an sm_120 architecture card")
    if args.cache_scope == "stream-l2" and args.mode == "fast":
        parser.error("--cache-scope stream-l2 applies only to the full replay stage")
    if args.carded_dram is True and args.mode == "fast":
        parser.error("--carded-dram applies only to the full l1-l2 stage")
    if args.carded_dram_enabled and args.memory_mode != "l1-l2":
        parser.error("--carded-dram requires --memory-mode l1-l2")
    if args.dram_service_rate_bytes_per_sec is not None:
        if args.dram_service_rate_bytes_per_sec <= 0:
            parser.error("--dram-service-rate-bytes-per-sec must be positive")
        if not args.carded_dram_enabled or args.memory_mode != "l1-l2":
            parser.error("--dram-service-rate-bytes-per-sec requires full carded l1-l2 replay")
    if args.target_arch == "sm_86" and args.mode in ("both", "full") and not args.pchase_card:
        parser.error("SM86 full replay requires --pchase-card")
    if args.target_arch == "sm_86" and args.pchase_card and not args.pchase_card.is_file():
        parser.error(f"SM86 P-chase card does not exist: {args.pchase_card}")
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
            metadata_arches = metadata_architectures(metadata)
            if metadata_arches != {args.target_arch}:
                raise ValueError(
                    f"capture metadata source architectures {sorted(metadata_arches)} require an explicit mixed-code disposition; expected homogeneous {args.target_arch}"
                )
            control.parent.mkdir(parents=True, exist_ok=True)
            synthesize_static_sidecar(
                capture, metadata, control, expected_arch=args.target_arch
            )
        except Exception as failure:
            error = str(failure)
        prepared.append((capture, relative, control, error))
    modes = ("full", "fast") if args.mode == "both" else (args.mode,)
    summary = {"capture_root": str(args.capture_root), "sim_bin": str(args.sim_bin),
               "arch_card": str(args.arch_card), "captures_discovered": len(captures),
               "pchase_card": str(args.pchase_card) if args.pchase_card else None,
               "target_arch": args.target_arch,
               "scope": ("explicitly selected real-capture prototype batch" if args.capture else
                         "all recursively discovered traces; none selected by replay outcome"),
               "selected_captures": args.capture,
               "memory_mode": args.memory_mode,
               "fast_memory_mode": args.fast_memory_mode,
               "cache_scope": args.cache_scope,
               "cache_scope_fallbacks": 0,
               "sm120_effective_l1": None,
               "sm120_effective_l2": None,
               "sm120_effective_l1_override": args.sm120_effective_l1,
               "sm120_effective_l2_override": args.sm120_effective_l2,
               "carded_dram": bool(args.carded_dram_enabled),
               "carded_dram_requested": args.carded_dram,
               "dram_service_rate_bytes_per_sec": args.dram_service_rate_bytes_per_sec,
               "dram_service_rate_requested": args.dram_service_rate_requested,
               "dram_service_rate_defaulted": args.dram_service_rate_defaulted,
               "memory_admission_feedback": str(args.memory_admission_feedback) if args.memory_admission_feedback else None,
               "memory_admission_feedback_defaulted": args.memory_admission_feedback_defaulted,
               "memory_admission_feedback_scope_suppressed": args.memory_admission_feedback_scope_suppressed,
               "memory_admission_feedback_disabled": args.memory_admission_feedback_disabled,
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
                        if row.get("cache_scope_fallback"):
                            summary["cache_scope_fallbacks"] += 1
                        if mode == "full" and row["status"] == "ok":
                            # Report the actual binary-resolved defaults, not
                            # absent override flags or duplicated Python constants.
                            summary["sm120_effective_l1"] = row.get("sm120_effective_l1")
                            summary["sm120_effective_l2"] = row.get("sm120_effective_l2")
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
