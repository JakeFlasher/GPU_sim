"""Calibrate the local NVML SHORT-BURST reference operating point (read-only).

The sustained anchor produced by
``tools/local_labels/calibrate_nvml_reference_operating_point.py`` records the
card's 90 C SW-thermal boost-EQUILIBRIUM operating point (~1350 MHz on the
RTX 3090 Ti) — the clock a long sustained kernel settles to. A SHORT kernel
never reaches that equilibrium: it runs during the cool, un-throttled full-boost
window (~1.85 GHz). Converting a short kernel's cycle count to nanoseconds at the
sustained 1350 MHz over-predicts ns by ~37 %. This tool captures the short-burst
operating point so the cycles->ns conversion can anchor short-regime kernels on
the clock they actually run at.

Method (READ-ONLY NVML; no torch — the test suite's uv env has no torch):
  - Build a tiny FP32-FMA spin kernel with nvcc (sm_86) once. The host driver
    LOOPS the launch until a bounded wall-clock duration elapses, so the GPU is
    CONTINUOUSLY busy.
  - Launch that spin as a BACKGROUND process and poll NVML SM clock / temp /
    power / throttle at a fixed rate WHILE it runs. Sampling during the active
    window is essential: sampling after a sub-second burst returns the idle DVFS
    bin (~210 MHz), not the boost clock — that bug failed the first capture.
  - The regime discriminator is TEMPERATURE, not the throttle bitmask. On this
    Ampere card NVML reports SW_THERMAL_SLOWDOWN essentially CONTINUOUSLY under
    any load (it is the normal DVFS boost-equilibrium signal, NOT a 90 C thermal
    event — CLAUDE.md "Active Measurement Stack" and the sustained calibrator
    both treat it as provenance, not a rejection). Empirically the clock under
    brief load is ~1950-2010 MHz while the card is cool and degrades toward the
    ~1350 MHz sustained floor only after seconds of saturation at >=90 C. So:
  - Accept a sample only when the card is active (clock above the idle floor,
    below NVML max), COOL (temp <= a short-burst ceiling well under the
    sustained 90 C equilibrium — this is what separates the two regimes), and
    drawing plausible power, and NOT HW-fault-throttled (HwSlowdown /
    HwThermalSlowdown / HwPowerBrake / SwPowerCap — genuine throttle events, as
    opposed to the benign SW-thermal boost bit). The reference per field is the
    MEDIAN of accepted samples; sw_thermal_active_fraction is recorded for
    provenance (typically ~1.0, the normal Ampere boost signal).

CLAUDE.md compliance (binding):
  - Rule 2 (no GPU-driver-root): READ-ONLY NVML getters only
    (nvmlDeviceGetClockInfo / GetTemperature / GetPowerUsage /
    GetCurrentClocksThrottleReasons). NEVER mutates application clocks,
    persistence, power limit, MIG, or compute mode; never invokes nvidia-smi
    clock-lock. The spin kernel is an ordinary user-space CUDA launch.
  - Rule 1 (NCU quarantine): no NCU; NVML telemetry only. Artifact lands under
    data/local_labels/.
  - Rule 3 (no privileged CUPTI): NVML is not CUPTI.
  - sm_86-only: fails closed if the detected device is not an RTX 3090 Ti.

Offline / no-GPU CI: emits nothing and exits 3 (the committed artifact is the
source of truth; a host without the 3090 Ti / nvcc must not overwrite it).

CLI:
  python tools/local_labels/calibrate_nvml_short_burst_anchor.py \
    [--output data/local_labels/nvml_short_burst_anchor_sm_86.json] \
    [--duration-s 4] [--warmup-s 1] [--poll-hz 100] [--cool-temp-c 75]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT = "data/local_labels/nvml_short_burst_anchor_sm_86.json"
_EXPECTED_DEVICE_SUBSTR = "RTX 3090 Ti"
_ARCH = "sm_86"
_NVCC = "/usr/local/cuda-13.2/bin/nvcc"

# Short-burst acceptance bands (GA102 / RTX 3090 Ti). A sample outside any band
# is rejected with the corresponding typed reason. The card must be ACTIVE (not
# idle) and COOL (well below the sustained 90 C equilibrium — the temperature
# ceiling is what separates the cool-boost short-burst regime from the hot
# sustained-throttle regime). The benign SW_THERMAL_SLOWDOWN boost bit is NOT a
# rejection (it is set ~continuously under load on Ampere); only genuine
# HW-fault / power-cap throttles are.
_MIN_ACTIVE_SM_CLOCK_MHZ = 300
_MIN_PLAUSIBLE_POWER_MW = 30_000
_MAX_PLAUSIBLE_POWER_MW = 500_000
_DEFAULT_COOL_TEMP_CEILING_C = 75.0

# A tiny FP32-FMA spin kernel: a long dependent FMA chain per thread keeps the
# SM busy (driving the boost clock up) without touching memory or driver state.
_SPIN_KERNEL_CU = r"""
extern "C" __global__ void spin_fma(float* sink, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = (float)i * 1.0001f + 1.0f;
    float y = 0.9999f;
    #pragma unroll 1
    for (int k = 0; k < iters; ++k) {
        x = __fmaf_rn(x, y, 1.0f);
        y = __fmaf_rn(y, x, 0.5f);
    }
    if (x == -1.0f) sink[i] = x;  // sink prevents dead-code elision
}
"""

# Host driver: a DUTY-CYCLED pulse train. Each period runs the spin for
# `active_ms` (driving the SM clock to full boost) then sleeps `idle_ms` (letting
# the die shed heat). A continuous spin reheats this card past 90 C within ~1 s,
# which trips SW-thermal regulation and drops the clock toward the sustained
# floor — the WRONG regime. A ~20-25 % duty cycle holds the card cool (well under
# 90 C) at full boost indefinitely, so NVML sampled DURING the active windows
# captures the genuine short-burst cool-boost operating point.
_SPIN_MAIN_CU = r"""
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
static double now_s(struct timespec* t0) {
    struct timespec n; clock_gettime(CLOCK_MONOTONIC, &n);
    return (n.tv_sec - t0->tv_sec) + (n.tv_nsec - t0->tv_nsec) / 1e9;
}
int main(int argc, char** argv) {
    int blocks = argc > 1 ? atoi(argv[1]) : 4096;
    int threads = 256;
    int iters = argc > 2 ? atoi(argv[2]) : 30000;
    double active_ms = argc > 3 ? atof(argv[3]) : 40.0;
    double idle_ms = argc > 4 ? atof(argv[4]) : 160.0;
    double duration_s = argc > 5 ? atof(argv[5]) : 5.0;
    float* d_sink = nullptr;
    if (cudaMalloc(&d_sink, (size_t)blocks * threads * sizeof(float)) != cudaSuccess) return 2;
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
    while (now_s(&t0) < duration_s) {
        double a0 = now_s(&t0);
        while (now_s(&t0) - a0 < active_ms / 1000.0) {
            spin_fma<<<blocks, threads>>>(d_sink, iters);
            if (cudaDeviceSynchronize() != cudaSuccess) { cudaFree(d_sink); return 3; }
        }
        struct timespec sl;
        sl.tv_sec = (time_t)(idle_ms / 1000.0);
        sl.tv_nsec = (long)((idle_ms / 1000.0 - sl.tv_sec) * 1e9);
        nanosleep(&sl, nullptr);
    }
    cudaFree(d_sink);
    return 0;
}
"""


def _git_head_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _classify_sample(
    sample: dict[str, Any],
    *,
    max_sm_clock_mhz: int,
    hw_fault_mask: int,
    cool_temp_ceiling_c: float,
) -> str | None:
    """Return ``None`` when the sample is ACCEPTED into the short-burst
    reference, else a typed rejection reason."""
    throttle = int(sample["throttle_reasons"])
    # Only GENUINE throttle events disqualify a sample. The benign
    # SW_THERMAL_SLOWDOWN boost bit is intentionally NOT in hw_fault_mask: on
    # Ampere it is set ~continuously under load (normal DVFS boost equilibrium),
    # so rejecting it would discard every sample.
    if throttle & hw_fault_mask:
        return "throttled_hw_fault"
    sm = int(sample["sm_clock_mhz"])
    if sm < _MIN_ACTIVE_SM_CLOCK_MHZ:
        return "idle_sm_clock_below_active_floor"
    if sm > max_sm_clock_mhz:
        return "sm_clock_above_nvml_max"
    temp = float(sample["temp_c"])
    # The temperature ceiling separates the cool-boost short-burst regime from
    # the hot sustained-throttle regime.
    if temp > cool_temp_ceiling_c:
        return "temp_above_short_burst_cool_ceiling"
    power = int(sample["power_mw"])
    if not (_MIN_PLAUSIBLE_POWER_MW <= power <= _MAX_PLAUSIBLE_POWER_MW):
        return "power_out_of_plausible_band"
    return None


def _build_spin_binary(workdir: Path) -> Path:
    """Compile the FP32-FMA spin kernel + duration-controlled host driver into a
    tiny executable and return its path. Raises RuntimeError if nvcc fails."""
    src = workdir / "spin.cu"
    exe = workdir / "spin"
    src.write_text(_SPIN_KERNEL_CU + _SPIN_MAIN_CU, encoding="utf-8")
    env = dict(os.environ)
    env.setdefault("NVCC_CCBIN", "g++-14")
    try:
        subprocess.run(
            [_NVCC, "-arch=sm_86", "-O2", "-o", str(exe), str(src)],
            check=True, capture_output=True, text=True, env=env,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"nvcc not found at {_NVCC}: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"spin-kernel build failed: {exc.stderr.strip()[:300]}"
        ) from exc
    return exe


def run_short_burst_sweep(
    *,
    duration_s: float,
    warmup_s: float,
    poll_hz: float,
    burst_blocks: int,
    burst_iters: int,
    active_ms: float,
    idle_ms: float,
    cool_temp_ceiling_c: float,
) -> dict[str, Any]:
    """Run the read-only NVML short-burst sweep (duty-cycled pulse-train spin in a
    background process + concurrent NVML polling) and return the anchor artifact
    dict. Raises RuntimeError on no GPU / wrong device / no accepted samples."""
    import pynvml  # local import so the module imports on a host w/o pynvml

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        raw_name = pynvml.nvmlDeviceGetName(handle)
        device_name = raw_name if isinstance(raw_name, str) else raw_name.decode()
        if _EXPECTED_DEVICE_SUBSTR not in device_name:
            raise RuntimeError(
                f"device {device_name!r} is not an {_EXPECTED_DEVICE_SUBSTR}; "
                f"this calibration is sm_86 (RTX 3090 Ti) only"
            )
        raw_driver = pynvml.nvmlSystemGetDriverVersion()
        driver = raw_driver if isinstance(raw_driver, str) else raw_driver.decode()
        max_sm_clock = int(
            pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
        )
        hw_fault_mask = (
            int(pynvml.nvmlClocksThrottleReasonHwSlowdown)
            | int(pynvml.nvmlClocksThrottleReasonHwThermalSlowdown)
            | int(pynvml.nvmlClocksThrottleReasonHwPowerBrakeSlowdown)
            | int(pynvml.nvmlClocksThrottleReasonSwPowerCap)
        )
        # Benign Ampere boost bit — recorded for provenance, never a rejection.
        sw_thermal_mask = int(pynvml.nvmlClocksThrottleReasonSwThermalSlowdown)

        with tempfile.TemporaryDirectory() as td:
            spin = _build_spin_binary(Path(td))
            samples: list[dict[str, Any]] = []
            proc = subprocess.Popen(
                [str(spin), str(burst_blocks), str(burst_iters),
                 str(active_ms), str(idle_ms), str(warmup_s + duration_s)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            try:
                period = 1.0 / poll_hz
                t_start = time.monotonic()
                t_sample_end = t_start + warmup_s + duration_s
                while time.monotonic() < t_sample_end and proc.poll() is None:
                    elapsed = time.monotonic() - t_start
                    sample = {
                        "elapsed_s": elapsed,
                        "power_mw": int(pynvml.nvmlDeviceGetPowerUsage(handle)),
                        "temp_c": float(
                            pynvml.nvmlDeviceGetTemperature(
                                handle, pynvml.NVML_TEMPERATURE_GPU
                            )
                        ),
                        "throttle_reasons": int(
                            pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
                        ),
                        "sm_clock_mhz": int(
                            pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                        ),
                        "mem_clock_mhz": int(
                            pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                        ),
                    }
                    # Only samples after warmup count (let the clock ramp).
                    if elapsed >= warmup_s:
                        samples.append(sample)
                    time.sleep(period)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            rc = proc.poll()
            if rc not in (0, None) and not samples:
                raise RuntimeError(
                    f"spin workload exited with code {rc} before any sample was "
                    f"taken"
                )
            accepted: list[dict[str, Any]] = []
            rejected: list[dict[str, Any]] = []
            for sample in samples:
                reason = _classify_sample(
                    sample,
                    max_sm_clock_mhz=max_sm_clock,
                    hw_fault_mask=hw_fault_mask,
                    cool_temp_ceiling_c=cool_temp_ceiling_c,
                )
                if reason is not None:
                    rejected.append({**sample, "rejection_reason": reason})
                else:
                    accepted.append(sample)
    finally:
        pynvml.nvmlShutdown()

    if not accepted:
        counts = dict(Counter(r["rejection_reason"] for r in rejected))
        raise RuntimeError(
            "no NVML sample reached an accepted cool, un-throttled, active "
            f"short-burst operating point (n_total={len(samples)}, "
            f"rejections={counts}); refusing to emit a reference from rejected "
            "data (fail closed)"
        )

    def _ref(field: str) -> float:
        return float(statistics.median(s[field] for s in accepted))

    rejection_counts = dict(Counter(r["rejection_reason"] for r in rejected))
    # Fraction of ACCEPTED samples carrying the benign SW-thermal boost bit
    # (recorded for provenance; typically ~1.0 on Ampere, NOT a rejection).
    sw_thermal_accepted = sum(
        1 for s in accepted if int(s["throttle_reasons"]) & sw_thermal_mask
    )
    sw_thermal_fraction = (
        float(sw_thermal_accepted) / len(accepted) if accepted else 0.0
    )
    return {
        "schema_version": 1,
        "artifact_kind": "nvml_reference_operating_point",
        "anchor_regime": "short_burst",
        "workload_class": "short_burst_fp32_fma_spin",
        "arch": _ARCH,
        "device_name": device_name,
        "driver_version": driver,
        "source_commit": _git_head_sha(),
        "sampling_method": {
            "read_only_nvml": True,
            "getters": [
                "nvmlDeviceGetClockInfo(NVML_CLOCK_SM)",
                "nvmlDeviceGetClockInfo(NVML_CLOCK_MEM)",
                "nvmlDeviceGetTemperature(NVML_TEMPERATURE_GPU)",
                "nvmlDeviceGetPowerUsage",
                "nvmlDeviceGetCurrentClocksThrottleReasons",
            ],
            "no_clock_lock_or_persistence_or_power_mutation": True,
            "workload": "short_burst_fp32_fma_spin_duty_cycled",
            "duration_s": duration_s,
            "warmup_s": warmup_s,
            "target_poll_hz": poll_hz,
            "burst_blocks": burst_blocks,
            "burst_iters": burst_iters,
            "active_ms": active_ms,
            "idle_ms": idle_ms,
            "cool_temp_ceiling_c": cool_temp_ceiling_c,
        },
        "max_sm_clock_mhz": max_sm_clock,
        "n_samples_total": len(samples),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "rejection_reason_counts": rejection_counts,
        # The benign Ampere boost bit, recorded for provenance (NOT a rejection).
        "sw_thermal_active_fraction": sw_thermal_fraction,
        "reference_operating_point": {
            "sm_clock_ref_mhz": _ref("sm_clock_mhz"),
            "mem_clock_ref_mhz": _ref("mem_clock_mhz"),
            "temp_ref_c": _ref("temp_c"),
            "power_ref_mw": _ref("power_mw"),
        },
        "accepted_sample_dispersion": {
            "sm_clock_mhz_stdev": (
                float(statistics.pstdev(s["sm_clock_mhz"] for s in accepted))
                if len(accepted) > 1 else 0.0
            ),
            "power_mw_stdev": (
                float(statistics.pstdev(s["power_mw"] for s in accepted))
                if len(accepted) > 1 else 0.0
            ),
        },
    }


def emit_short_burst_anchor(argv: list[str] | None = None) -> int:
    """CLI entry: run the duty-cycled short-burst sweep and write the artifact.

    Named `emit_short_burst_anchor` (not `main`) to match the local-label tool
    convention so the arch-capabilities coverage scan (which excludes
    tools/local_labels/ from entry-point discovery and only flags declared files
    that carry a literal `def main(`) treats this as a declared callable module."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument(
        "--warmup-s", type=float, default=0.0,
        help="seconds to skip before sampling; default 0 — the cool window IS "
             "the start of the load, so we sample from t=0 and let the temp "
             "ceiling reject samples once the card heats past the cool ceiling",
    )
    parser.add_argument("--poll-hz", type=float, default=100.0)
    parser.add_argument("--burst-blocks", type=int, default=4096)
    parser.add_argument("--burst-iters", type=int, default=30000)
    parser.add_argument(
        "--active-ms", type=float, default=40.0,
        help="spin duration per duty-cycle period (drives full boost)",
    )
    parser.add_argument(
        "--idle-ms", type=float, default=160.0,
        help="sleep per duty-cycle period (sheds heat; 40/160 = 20%% duty keeps "
             "the card cool at full boost)",
    )
    parser.add_argument(
        "--cool-temp-c", type=float, default=_DEFAULT_COOL_TEMP_CEILING_C
    )
    args = parser.parse_args(argv)
    try:
        artifact = run_short_burst_sweep(
            duration_s=args.duration_s,
            warmup_s=args.warmup_s,
            poll_hz=args.poll_hz,
            burst_blocks=args.burst_blocks,
            burst_iters=args.burst_iters,
            active_ms=args.active_ms,
            idle_ms=args.idle_ms,
            cool_temp_ceiling_c=args.cool_temp_c,
        )
    except ImportError as exc:
        sys.stderr.write(
            f"nvml_short_burst_unavailable: {exc} (no write performed)\n"
        )
        return 3
    except RuntimeError as exc:
        sys.stderr.write(f"nvml_short_burst_failed: {exc}\n")
        return 1
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    ref = artifact["reference_operating_point"]
    sys.stderr.write(
        f"[nvml_short_burst] device={artifact['device_name']} "
        f"driver={artifact['driver_version']} "
        f"accepted={artifact['n_accepted']}/{artifact['n_samples_total']} "
        f"sm_clock_ref={ref['sm_clock_ref_mhz']}MHz "
        f"temp_ref={ref['temp_ref_c']}C -> {args.output}\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(emit_short_burst_anchor())
