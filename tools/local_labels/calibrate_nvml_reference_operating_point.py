"""Calibrate the local NVML covariate reference operating point (read-only).

Runs a READ-ONLY NVML telemetry sweep on the local RTX 3090 Ti (sm_86) while a
sustained GPU workload holds the card at a steady-state active operating point,
then emits a committed calibration artifact recording the reference SM clock,
temperature, power, and memory clock, the sampling method, and the accepted vs
rejected samples (with rejection reasons). Downstream, the NVML covariate
regression / ns-conversion path centers its deviations on this reference instead
of the retracted ``nan`` / ``-1`` placeholders in
``workloads/simbench_bw/nvml_covariate_acceptance.toml``.

CLAUDE.md compliance (binding):
  - Rule 2 (no GPU-driver-root): READ-ONLY NVML only. This script ONLY reads
    telemetry getters (``nvmlDeviceGetClockInfo`` / ``GetTemperature`` /
    ``GetPowerUsage`` / ``GetCurrentClocksThrottleReasons``). It NEVER mutates
    application clocks, persistence, power limit, MIG, or compute mode, and never
    invokes ``nvidia-smi`` clock-lock. The sustained workload is an ordinary
    user-space CUDA matmul (no driver state change).
  - Rule 1 (NCU quarantine): no NCU is invoked; this is NVML telemetry, not
    profiling. The artifact lands under ``data/local_labels/``.
  - Rule 3 (no privileged CUPTI): NVML is not CUPTI.
  - sm_86-only: the artifact is named/scoped sm_86; the script fails closed if the
    detected device is not an RTX 3090 Ti.

Sample acceptance: a sample is ACCEPTED into the reference computation only when
the GPU is in an active, plausible state — SM clock within a plausible band (not
idle, not above the NVML max clock), temperature/power within plausible bounds,
and the throttle bitmask carries no HARDWARE-fault slowdown (HW slowdown / HW
thermal / power-brake). The SW THERMAL slowdown reason is NOT a rejection: on
Ampere it is the normal DVFS boost-equilibrium signal (the card boosts until it
reaches its thermal/power steady state and reports SW thermal), so the genuine
reference operating point the regression must center on IS that thermally-settled
boost clock. Whether SW-thermal was active throughout is recorded in the
artifact's provenance (``sw_thermal_active_fraction``) for reviewer replay.
Rejected samples are recorded with a typed reason so the reference is computed
only over genuine active steady-state operating points.

The reference value per field is the MEDIAN of accepted samples (robust to the
warmup ramp and transient dips). The artifact is self-describing for reviewer
replay.

CLI:
  python tools/local_labels/calibrate_nvml_reference_operating_point.py \
    [--output data/local_labels/nvml_reference_operating_point_sm_86.json] \
    [--duration-s 20] [--warmup-s 5] [--poll-hz 200] [--matmul-dim 4096]

  # Offline / no-GPU CI: emit nothing and exit 3 (the committed artifact is the
  # source of truth; a host without the 3090 Ti must not overwrite it).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT = "data/local_labels/nvml_reference_operating_point_sm_86.json"
_EXPECTED_DEVICE_SUBSTR = "RTX 3090 Ti"
_ARCH = "sm_86"

# Plausibility bands for sample acceptance (GA102 / RTX 3090 Ti). A sample
# outside any band is rejected with the corresponding typed reason. These bound
# physically-reasonable steady-state operating points; they are NOT clock-control
# (the script never SETS a clock — it only filters which OBSERVED samples anchor
# the reference).
_MIN_ACTIVE_SM_CLOCK_MHZ = 300      # below this the card is effectively idle
_MIN_PLAUSIBLE_TEMP_C = 20.0
_MAX_PLAUSIBLE_TEMP_C = 95.0
_MIN_PLAUSIBLE_POWER_MW = 30_000    # an active GA102 draws well above idle ~25W
_MAX_PLAUSIBLE_POWER_MW = 500_000


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
    slowdown_mask: int,
) -> str | None:
    """Return ``None`` when the sample is ACCEPTED, else a typed rejection reason.

    A sample is accepted only at a genuine active, unthrottled, plausible
    steady-state point so the reference is not skewed by the idle/warmup ramp or
    transient slowdown."""
    throttle = int(sample["throttle_reasons"])
    if throttle & slowdown_mask:
        return "throttled_slowdown"
    sm = int(sample["sm_clock_mhz"])
    if sm < _MIN_ACTIVE_SM_CLOCK_MHZ:
        return "idle_sm_clock_below_active_floor"
    if sm > max_sm_clock_mhz:
        return "sm_clock_above_nvml_max"
    temp = float(sample["temp_c"])
    if not (_MIN_PLAUSIBLE_TEMP_C <= temp <= _MAX_PLAUSIBLE_TEMP_C):
        return "temp_out_of_plausible_band"
    power = int(sample["power_mw"])
    if not (_MIN_PLAUSIBLE_POWER_MW <= power <= _MAX_PLAUSIBLE_POWER_MW):
        return "power_out_of_plausible_band"
    return None


def run_calibration_sweep(
    *,
    duration_s: float,
    warmup_s: float,
    poll_hz: float,
    matmul_dim: int,
) -> dict[str, Any]:
    """Run the read-only NVML sweep under a sustained matmul load and return the
    calibration artifact dict. Raises RuntimeError if no GPU / wrong device / no
    accepted samples (fail closed — never emit a reference from rejected data)."""
    import pynvml  # local import so the module is importable on a host w/o pynvml

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        raw_name = pynvml.nvmlDeviceGetName(handle)
        device_name = raw_name if isinstance(raw_name, str) else raw_name.decode()
        if _EXPECTED_DEVICE_SUBSTR not in device_name:
            raise RuntimeError(
                f"device {device_name!r} is not an {_EXPECTED_DEVICE_SUBSTR}; "
                f"this calibration is sm_86 (RTX 3090 Ti) only — refusing to "
                f"write a reference from a different card"
            )
        raw_driver = pynvml.nvmlSystemGetDriverVersion()
        driver = raw_driver if isinstance(raw_driver, str) else raw_driver.decode()
        max_sm_clock = int(
            pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
        )
        # The HARDWARE-FAULT slowdown mask: any of these throttle reasons means
        # the sample is a faulted, not a clean steady-state, operating point. SW
        # THERMAL slowdown is INTENTIONALLY EXCLUDED: on Ampere it is the normal
        # DVFS boost-equilibrium signal (the card boosts to its thermal/power
        # steady state and reports SW thermal), so the genuine reference IS that
        # thermally-settled boost clock. GPU_IDLE is also not in the mask (idle is
        # rejected by the active-clock floor). SW power cap is included: it caps
        # the card below its natural boost, which is a power-limited (not a clean
        # thermal-equilibrium) operating point.
        slowdown_mask = (
            int(pynvml.nvmlClocksThrottleReasonHwSlowdown)
            | int(pynvml.nvmlClocksThrottleReasonHwThermalSlowdown)
            | int(pynvml.nvmlClocksThrottleReasonHwPowerBrakeSlowdown)
            | int(pynvml.nvmlClocksThrottleReasonSwPowerCap)
        )
        sw_thermal_mask = int(pynvml.nvmlClocksThrottleReasonSwThermalSlowdown)

        # Sustained user-space CUDA workload to hold the card at an active
        # operating point. Ordinary matmul — no driver state mutation.
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch CUDA is not available; cannot warm the GPU")
        dev = torch.device("cuda:0")
        a = torch.randn(matmul_dim, matmul_dim, device=dev, dtype=torch.float32)
        b = torch.randn(matmul_dim, matmul_dim, device=dev, dtype=torch.float32)

        poll_interval_s = 1.0 / poll_hz
        samples: list[dict[str, Any]] = []
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        n_accepted_sw_thermal = 0
        t0 = time.monotonic_ns()
        deadline = time.monotonic() + duration_s
        warmup_end = time.monotonic() + warmup_s
        while time.monotonic() < deadline:
            # Keep the GPU busy between samples.
            c = a @ b
            a = c * 1.0000001  # keep the chain live; prevents dead-code elision
            sample = {
                "timestamp_ns": time.monotonic_ns() - t0,
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
            samples.append(sample)
            in_warmup = time.monotonic() < warmup_end
            reason = _classify_sample(
                sample, max_sm_clock_mhz=max_sm_clock, slowdown_mask=slowdown_mask,
            )
            if in_warmup:
                rejected.append({**sample, "rejection_reason": "warmup_window"})
            elif reason is not None:
                rejected.append({**sample, "rejection_reason": reason})
            else:
                accepted.append(sample)
                if sample["throttle_reasons"] & sw_thermal_mask:
                    n_accepted_sw_thermal += 1
            time.sleep(poll_interval_s)
        # Ensure the workload result is consumed (defensive; the chain above
        # already keeps it live).
        _ = float(a.flatten()[0].item())
    finally:
        pynvml.nvmlShutdown()

    if not accepted:
        raise RuntimeError(
            "no NVML sample reached an accepted steady-state operating point "
            f"(n_total={len(samples)}, n_rejected={len(rejected)}); refusing to "
            "emit a reference from rejected data (fail closed)"
        )

    def _ref(field: str) -> float:
        return float(statistics.median(s[field] for s in accepted))

    from collections import Counter

    rejection_counts = dict(
        Counter(r["rejection_reason"] for r in rejected)
    )
    return {
        "schema_version": 1,
        "artifact_kind": "nvml_reference_operating_point",
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
            "workload": "user_space_cuda_matmul",
            "matmul_dim": matmul_dim,
            "target_poll_hz": poll_hz,
            "duration_s": duration_s,
            "warmup_s": warmup_s,
        },
        "max_sm_clock_mhz": max_sm_clock,
        "n_samples_total": len(samples),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "rejection_reason_counts": rejection_counts,
        "sw_thermal_active_fraction": (
            n_accepted_sw_thermal / len(accepted) if accepted else 0.0
        ),
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--warmup-s", type=float, default=5.0)
    parser.add_argument("--poll-hz", type=float, default=200.0)
    parser.add_argument("--matmul-dim", type=int, default=4096)
    args = parser.parse_args(argv)
    try:
        artifact = run_calibration_sweep(
            duration_s=args.duration_s,
            warmup_s=args.warmup_s,
            poll_hz=args.poll_hz,
            matmul_dim=args.matmul_dim,
        )
    except ImportError as exc:
        # No pynvml / torch on this host -> cannot calibrate. Exit 3 WITHOUT
        # writing, so an offline/CI host never overwrites the committed artifact.
        sys.stderr.write(
            f"nvml_calibration_unavailable: {exc} (no write performed)\n"
        )
        return 3
    except RuntimeError as exc:
        sys.stderr.write(f"nvml_calibration_failed: {exc}\n")
        return 1
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    ref = artifact["reference_operating_point"]
    sys.stderr.write(
        f"[nvml_calibration] device={artifact['device_name']} "
        f"driver={artifact['driver_version']} "
        f"accepted={artifact['n_accepted']}/{artifact['n_samples_total']} "
        f"sm_clock_ref={ref['sm_clock_ref_mhz']}MHz "
        f"temp_ref={ref['temp_ref_c']}C power_ref={ref['power_ref_mw']}mW "
        f"-> {args.output}\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
