#!/usr/bin/env python3
"""B200 Phase-1 → Phase-2 preflight gate.

Emits ``b200_ready_manifest.json`` summarising whether the host (local
sm_120 or a B200 sm_100a provider) can host the framework's five
counter-free analytic surfaces. Per CLAUDE.md §"Counter-Free Analytics
Framework", this is the explicit phase gate the framework MUST clear
before fanning out to all 235 *.cu profiling runs.

The preflight is **read-only** end-to-end. No clock-policy mutation,
no NCU invocation, no host-driver changes. Every probe either reads
device state via ``nvidia-smi`` (read-only device-identity queries)
or invokes a tool with ``--version`` / a stub compile.

Every probe's ``status`` field is drawn from the locked counter-free
framework vocabulary: ``{ok, unavailable, blocked, failed, skipped}``.
A probe that can't find a tool reports ``unavailable`` (not the
pre-2026-05-13 ``missing``) and carries a ``waiver_reason``. The
``storage`` probe always reports ``ok`` and uses a separate ``warn``
boolean to flag low-disk conditions, since "warn" is not a status —
it's a hint to the operator that fanout artifact retention may exhaust
the artifact path mid-run.

Required probes (all must be ``ok`` for ``ready_for_phase_2: true``):

  - ``nvcc``: ``nvcc --version`` succeeds; major/minor recorded.
  - ``driver``: GPU driver version reads cleanly via ``nvidia-smi``.
  - ``cuda_runtime``: ``cudaRuntimeGetVersion`` succeeds via a stub
    compile-and-run (skipped on hosts without an executable GPU; the
    runtime version is still read from nvcc).
  - ``device``: at least one GPU visible, with compute capability
    recorded. For Phase-2 the expected CC is ``sm_100`` or ``sm_100a``.
  - ``cutlass``: CUTLASS headers resolve; major/minor recorded from
    ``cutlass/include/cutlass/version.h``.
  - ``sm_100a_compile_proof``: ``nvcc -arch=compute_100a -code=sm_100a``
    accepts a one-line stub. Proves the local nvcc can codegen for B200
    even when the local card is sm_120.

Recommended probes (record status but do NOT block Phase 2):

  - ``nsys``: ``nsys --version`` (Surface 3).
  - ``compute_sanitizer``: ``compute-sanitizer --version`` (Surface 5).
  - ``cupti``: CUPTI shared lib resolves on the linker path; sample
    activity-enable call OK (Surface 2).
  - ``nvbit_metrics_tool``:
    ``nvbit_tracer/metric_extractor/metric_extractor.so`` exists and
    its ``cuobjdump --list-elf`` output covers the host architecture.
    On a B200 host (compute capability ``10.x``) the probe escalates
    to ``failed`` when ``sm_100a`` is missing from the cubin list —
    rejection path for "the committed `.so` lacks an `sm_100a` cubin
    on a B200 host" (canonical Surface 1). The probe is required
    (so a B200 fanout gates on the canonical Surface 1 producer
    actually being injectable), with the host-conditional rule
    elevating it from a build-existence check to a wrong-arch
    rejection gate on B200 hosts.
  - ``storage``: estimate free disk on the artifact path; warn under
    20 GiB.

Exit codes:

  - 0: manifest emitted; all REQUIRED probes ``ok`` (Phase 2 ready).
  - 1: manifest emitted; one or more REQUIRED probes failed
    (manifest captures the blockers).
  - 2: manifest emission itself failed (couldn't write the output
    file, internal exception, etc.).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

# Mode vocabulary. `phase2-b200-fanout` is the legacy default — the
# probe set the 235-pilot fanout assumes. `phase1-local-smoke` is a
# narrower contract for the local sm_120 / sm_86 dev loop: it does
# NOT require the B200-specific sm_100a probes (`sm_100a_compile_proof`
# and `sm_100a_cubin_disasm`), and it disables the
# `nvbit_metrics_tool` host-conditional escalation that turns a
# wrong-arch `.so` into a blocker on a B200 host. Phase-1 still
# requires the rest of the toolchain — nvcc, driver, cuda_runtime,
# device, cutlass, cache_replay_native_available, nvbit_metrics_tool
# — so a local smoke run cannot silently skip a missing CUTLASS
# checkout or a broken nvcc.
MODE_PHASE_1_LOCAL_SMOKE = "phase1-local-smoke"
MODE_PHASE_2_B200_FANOUT = "phase2-b200-fanout"
_VALID_MODES = (MODE_PHASE_1_LOCAL_SMOKE, MODE_PHASE_2_B200_FANOUT)
_DEFAULT_MODE = MODE_PHASE_2_B200_FANOUT

_REQUIRED_PROBES = (
    "nvcc",
    "driver",
    "cuda_runtime",
    "device",
    "cutlass",
    "sm_100a_compile_proof",
    # Required-for-fanout, refined-scope additions. Each maps to a
    # specific surface gate the 235-pilot fanout depends on. A
    # failure here MUST block fanout — the loop's downstream design
    # assumes these surfaces are reachable.
    "nvbit_metrics_tool",        # Canonical NVBit metric extractor (Surface 1).
    "sm_100a_cubin_disasm",      # nvdisasm consumes sm_100a cubin output (Surface 4 ext).
    "cache_replay_native_available",  # Compiled cache-replay throughput floor (Surface 7).
)

# Probes that the B200-fanout contract requires but that the
# local-smoke contract demotes to recommended. A non-`ok` here in
# phase-1 mode is recorded in the manifest's recommended_probes list
# rather than blocking exit.
_PHASE_2_ONLY_REQUIRED_PROBES = (
    "sm_100a_compile_proof",
    "sm_100a_cubin_disasm",
)

# Probes that report degraded/unavailable WITHOUT blocking the fanout.
# Differs from _RECOMMENDED_PROBES in that the manifest explicitly
# records a dependency / fallback decision rather than just absence.
# A required-prefer failure flips the matching substatus but does
# not populate `blockers`. Currently empty post Scope-B follow-up
# cleanup (the `zstd_dep` probe was retired alongside the legacy
# compact-writer it gated on).
_REQUIRED_PREFER_PROBES: Tuple[str, ...] = ()

_RECOMMENDED_PROBES = (
    "nsys",
    "compute_sanitizer",
    "cupti",
    "storage",
)

# Operator opt-out for cache_replay_native_available — when this env
# is set, that probe degrades from required-for-fanout to
# required-prefer (substatus records the operator opt-out instead of
# blocking). The 235-pilot fanout's < 2 h replay budget is no longer
# guaranteed when the slow Python backend is engaged; the operator
# accepts that tradeoff explicitly.
_PYTHON_REPLAY_FANOUT_OPT_OUT_ENV = "SIMBENCH_ALLOW_PYTHON_REPLAY_FANOUT"


def _run(
    cmd: List[str],
    timeout: float = 30.0,
) -> Tuple[int, str, str]:
    """Run a subprocess and return (exit_code, stdout, stderr)."""
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return completed.returncode, completed.stdout, completed.stderr
    except FileNotFoundError:
        return 127, "", f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"command timed out after {timeout}s: {' '.join(cmd)}"


def probe_nvcc() -> Dict[str, Any]:
    rc, stdout, stderr = _run(["nvcc", "--version"], timeout=10.0)
    if rc != 0:
        return {
            "status": "unavailable",
            "exit_code": rc,
            "stderr": stderr.strip(),
            "waiver_reason": (
                "nvcc not on PATH or failed to launch; CUDA toolchain "
                "absent from this host."
            ),
        }
    match = re.search(r"release (\d+)\.(\d+)", stdout)
    version = ".".join(match.groups()) if match else stdout.split("\n")[0]
    return {
        "status": "ok",
        "version": version,
        "command": "nvcc --version",
        "exit_code": 0,
    }


def probe_driver() -> Dict[str, Any]:
    rc, stdout, _ = _run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        timeout=10.0,
    )
    if rc != 0:
        return {
            "status": "unavailable",
            "exit_code": rc,
            "waiver_reason": (
                "nvidia-smi --query-gpu=driver_version exited non-zero; "
                "no NVIDIA driver visible to this host."
            ),
        }
    version = stdout.strip().splitlines()[0] if stdout.strip() else "unknown"
    return {
        "status": "ok",
        "version": version,
        "command": "nvidia-smi --query-gpu=driver_version",
        "exit_code": 0,
    }


def probe_cuda_runtime() -> Dict[str, Any]:
    nvcc = probe_nvcc()
    if nvcc.get("status") != "ok":
        return {
            "status": "unavailable",
            "reason": "nvcc probe failed; cannot infer runtime version",
        }
    return {
        "status": "ok",
        "version_from_nvcc": nvcc["version"],
        "note": "runtime version inferred from nvcc; for an authoritative reading run cudaRuntimeGetVersion() via a host stub compile.",
    }


def probe_device() -> Dict[str, Any]:
    rc, stdout, _ = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,memory.total,memory.free",
            "--format=csv,noheader",
        ],
        timeout=10.0,
    )
    if rc != 0:
        return {
            "status": "unavailable",
            "exit_code": rc,
            "waiver_reason": (
                "nvidia-smi exited non-zero; no GPU device visible to "
                "this host."
            ),
        }
    line = stdout.strip().splitlines()[0] if stdout.strip() else ""
    if not line:
        return {
            "status": "failed",
            "waiver_reason": "no GPU visible to nvidia-smi (empty output)",
        }
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        return {
            "status": "failed",
            "waiver_reason": f"unexpected nvidia-smi format: {line!r}",
        }
    name, compute_cap, mem_total, mem_free = parts[0], parts[1], parts[2], parts[3]
    is_b200 = compute_cap.startswith("10.")
    is_sm_120 = compute_cap.startswith("12.")
    return {
        "status": "ok",
        "device_name": name,
        "compute_capability": compute_cap,
        "is_sm_100_family": is_b200,
        "is_sm_120_family": is_sm_120,
        "memory_total": mem_total,
        "memory_free": mem_free,
        "phase_2_target": is_b200,
    }


def probe_cutlass() -> Dict[str, Any]:
    version_h = REPO_ROOT / "third_party" / "cutlass" / "include" / "cutlass" / "version.h"
    if not version_h.is_file():
        return {
            "status": "unavailable",
            "expected_path": str(version_h),
            "waiver_reason": (
                "cutlass/include/cutlass/version.h absent — re-clone "
                "CUTLASS or check submodule."
            ),
        }
    try:
        text = version_h.read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "status": "failed",
            "waiver_reason": f"cannot read {version_h}: {exc}",
        }
    major = re.search(r"#define\s+CUTLASS_MAJOR\s+(\d+)", text)
    minor = re.search(r"#define\s+CUTLASS_MINOR\s+(\d+)", text)
    patch = re.search(r"#define\s+CUTLASS_PATCH\s+(\d+)", text)
    if not (major and minor):
        return {
            "status": "failed",
            "waiver_reason": "CUTLASS_MAJOR / CUTLASS_MINOR not found in version.h",
            "expected_path": str(version_h),
        }
    return {
        "status": "ok",
        "version": f"{major.group(1)}.{minor.group(1)}.{patch.group(1) if patch else 0}",
        "header_path": str(version_h),
    }


def probe_sm_100a_compile() -> Dict[str, Any]:
    """Cross-compile proof for the B200 codegen path.

    Two sub-steps run in sequence; the probe's top-level ``status`` is the
    worst of the two:

      1. ``stub_compile``: a one-line ``__global__`` stub goes through nvcc
         with ``-gencode arch=compute_100a,code=sm_100a``. Toolchain-level
         "nvcc accepts B200 codegen at all" check.
      2. ``metric_extractor_compile``: compiles
         ``nvbit_tracer/metric_extractor/{metric_extractor.cu,
         inject_funcs.cu}`` with the same ``-I`` and per-source ``-Xptxas``
         flags the tool's Makefile uses. Tool-specific "the new sources
         survive a B200 cross-compile" check. Skipped (status reported as
         ``skipped``) when the source tree is absent — the stub probe still
         carries the toolchain-level result.

    The substatus dict carries every sub-step's outcome plus the absolute
    paths probed so a non-``ok`` manifest entry has enough context to act
    on.
    """
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return {
            "status": "unavailable",
            "waiver_reason": "nvcc not on PATH",
            "substatus": {
                "stub_compile": "unavailable",
                "metric_extractor_compile": "unavailable",
            },
        }

    substatus: Dict[str, Any] = {}

    with tempfile.TemporaryDirectory() as td:
        # --- Sub-step 1: stub compile (the original toolchain check). ---
        stub_src = Path(td) / "stub.cu"
        stub_src.write_text(
            "extern \"C\" __global__ void stub(int* x) { *x = 0; }\n",
            encoding="utf-8",
        )
        stub_obj = Path(td) / "stub.o"
        stub_cmd = [
            nvcc,
            "-c",
            str(stub_src),
            "-gencode", "arch=compute_100a,code=sm_100a",
            "-o", str(stub_obj),
        ]
        stub_rc, _, stub_stderr = _run(stub_cmd, timeout=60.0)
        if stub_rc != 0:
            substatus["stub_compile"] = {
                "status": "failed",
                "exit_code": stub_rc,
                "stderr_tail": stub_stderr.strip().splitlines()[-5:]
                if stub_stderr else [],
            }
            return {
                "status": "failed",
                "exit_code": stub_rc,
                "stderr_tail": stub_stderr.strip().splitlines()[-5:]
                if stub_stderr else [],
                "waiver_reason": (
                    "nvcc rejects -gencode arch=compute_100a,code=sm_100a; "
                    "toolchain does not codegen for B200."
                ),
                "substatus": substatus,
            }
        substatus["stub_compile"] = {
            "status": "ok",
            "command": (
                "nvcc -c stub.cu "
                "-gencode arch=compute_100a,code=sm_100a"
            ),
            "exit_code": 0,
        }

        # --- Sub-step 2: cross-compile the metric_extractor sources. ---
        me_dir = REPO_ROOT / "nvbit_tracer" / "metric_extractor"
        nvbit_include = REPO_ROOT / "nvbit_tracer" / "nvbit_release" / "core"
        me_main = me_dir / "metric_extractor.cu"
        me_inject = me_dir / "inject_funcs.cu"

        if not me_main.is_file() or not me_inject.is_file():
            missing = [
                str(p)
                for p in (me_main, me_inject)
                if not p.is_file()
            ]
            substatus["metric_extractor_compile"] = {
                "status": "failed",
                "waiver_reason": (
                    "metric_extractor sources absent on this host; the "
                    "toolchain stub still compiles for B200 but the new "
                    "tool's `.cu` files are missing, so the .so cannot "
                    "be rebuilt before fanout."
                ),
                "expected_paths": [str(me_main), str(me_inject)],
                "missing_paths": missing,
            }
            # Top-level escalates to ``failed`` because the source tree
            # is part of the v1 readiness contract; toolchain-only
            # cleanliness is not enough to call B200 codegen ready.
            return {
                "status": "failed",
                "exit_code": 0,
                "waiver_reason": (
                    "metric_extractor source tree missing from "
                    f"{me_dir}; rebuild the repo (or sync the missing "
                    f"files: {missing}) — the source tree is part of "
                    "the new tool's B200 readiness contract."
                ),
                "substatus": substatus,
            }

        # Match the Makefile's host-side recipe for metric_extractor.cu:
        #   -dc -c -std=c++11 -I<core> -Xptxas -cloning=no -Xcompiler -Wall
        #   -gencode arch=compute_100a,code=sm_100a -Xcompiler -fPIC
        me_main_obj = Path(td) / "metric_extractor.o"
        me_main_cmd = [
            nvcc,
            "-dc", "-c",
            "-std=c++11",
            f"-I{nvbit_include}",
            "-Xptxas", "-cloning=no",
            "-Xcompiler", "-Wall",
            "-gencode", "arch=compute_100a,code=sm_100a",
            "-Xcompiler", "-fPIC",
            str(me_main),
            "-o", str(me_main_obj),
        ]
        me_main_rc, _, me_main_stderr = _run(me_main_cmd, timeout=120.0)

        # Match the Makefile's inject_funcs recipe:
        #   -I<core> -Xptxas -astoolspatch --keep-device-functions
        #   -gencode arch=compute_100a,code=sm_100a -Xcompiler -Wall
        #   -Xcompiler -fPIC -c
        # (No -dc — inject_funcs is the patch-target translation unit
        # and must be compiled with -astoolspatch.)
        me_inject_obj = Path(td) / "inject_funcs.o"
        me_inject_cmd = [
            nvcc,
            f"-I{nvbit_include}",
            "-Xptxas", "-astoolspatch",
            "--keep-device-functions",
            "-gencode", "arch=compute_100a,code=sm_100a",
            "-Xcompiler", "-Wall",
            "-Xcompiler", "-fPIC",
            "-c",
            str(me_inject),
            "-o", str(me_inject_obj),
        ]
        me_inject_rc, _, me_inject_stderr = _run(me_inject_cmd, timeout=120.0)

    if me_main_rc != 0 or me_inject_rc != 0:
        which_failed = []
        if me_main_rc != 0:
            which_failed.append("metric_extractor.cu")
        if me_inject_rc != 0:
            which_failed.append("inject_funcs.cu")
        # Pick the worst stderr to surface in the top-level field; the
        # substatus carries both for the operator.
        stderr_tail_main = (
            me_main_stderr.strip().splitlines()[-5:] if me_main_stderr else []
        )
        stderr_tail_inject = (
            me_inject_stderr.strip().splitlines()[-5:]
            if me_inject_stderr else []
        )
        substatus["metric_extractor_compile"] = {
            "status": "failed",
            "failed_sources": which_failed,
            "metric_extractor_cu": {
                "exit_code": me_main_rc,
                "stderr_tail": stderr_tail_main,
            },
            "inject_funcs_cu": {
                "exit_code": me_inject_rc,
                "stderr_tail": stderr_tail_inject,
            },
        }
        return {
            "status": "failed",
            "exit_code": me_main_rc if me_main_rc != 0 else me_inject_rc,
            "stderr_tail": (
                stderr_tail_main if me_main_rc != 0 else stderr_tail_inject
            ),
            "waiver_reason": (
                "nvcc accepted the stub but rejected the metric_extractor "
                f"sources for B200 codegen: {', '.join(which_failed)}. "
                "Both sources must cross-compile for compute_100a,sm_100a "
                "before the new tool is B200-ready."
            ),
            "substatus": substatus,
        }

    substatus["metric_extractor_compile"] = {
        "status": "ok",
        "compiled_sources": ["metric_extractor.cu", "inject_funcs.cu"],
        "command_metric_extractor_cu": (
            f"nvcc -dc -c -std=c++11 -I{nvbit_include} "
            "-Xptxas -cloning=no -Xcompiler -Wall "
            "-gencode arch=compute_100a,code=sm_100a "
            "-Xcompiler -fPIC metric_extractor.cu"
        ),
        "command_inject_funcs_cu": (
            f"nvcc -I{nvbit_include} -Xptxas -astoolspatch "
            "--keep-device-functions "
            "-gencode arch=compute_100a,code=sm_100a "
            "-Xcompiler -Wall -Xcompiler -fPIC -c inject_funcs.cu"
        ),
    }
    return {
        "status": "ok",
        "command": (
            "nvcc -c stub.cu "
            "-gencode arch=compute_100a,code=sm_100a"
            " (+ metric_extractor.cu + inject_funcs.cu)"
        ),
        "exit_code": 0,
        "substatus": substatus,
    }


def probe_nsys() -> Dict[str, Any]:
    rc, stdout, _ = _run(["nsys", "--version"], timeout=10.0)
    if rc != 0:
        return {
            "status": "unavailable",
            "exit_code": rc,
            "waiver_reason": (
                "nsys not on PATH or failed to launch; Nsight Systems "
                "binary absent."
            ),
        }
    return {"status": "ok", "version": stdout.strip().splitlines()[0]}


def probe_compute_sanitizer() -> Dict[str, Any]:
    rc, stdout, _ = _run(["compute-sanitizer", "--version"], timeout=10.0)
    if rc != 0:
        return {
            "status": "unavailable",
            "exit_code": rc,
            "waiver_reason": (
                "compute-sanitizer not on PATH or failed to launch; "
                "binary absent."
            ),
        }
    return {"status": "ok", "version": stdout.strip().splitlines()[0]}


def probe_cupti() -> Dict[str, Any]:
    candidates: List[Path] = []
    for env_var in ("CUDA_HOME", "CUDA_INSTALL_PATH"):
        base = os.environ.get(env_var)
        if base:
            candidates.append(Path(base) / "extras" / "CUPTI" / "lib64" / "libcupti.so")
    nvcc = shutil.which("nvcc")
    if nvcc:
        base = Path(nvcc).resolve().parent.parent
        candidates.append(base / "extras" / "CUPTI" / "lib64" / "libcupti.so")
    candidates.append(Path("/usr/local/cuda/extras/CUPTI/lib64/libcupti.so"))
    for cand in candidates:
        if cand.is_file() or cand.is_symlink():
            return {"status": "ok", "library_path": str(cand)}
    return {
        "status": "unavailable",
        "searched": [str(c) for c in candidates],
        "waiver_reason": "libcupti.so not found in any standard location.",
    }


# Regex carving the cubin tags out of `cuobjdump --list-elf` output.
# A typical line is ``ELF file    1: metric_extractor.1.sm_86.cubin`` —
# the trailing ``sm_<digits>[a]?`` token is what we want.
_CUOBJDUMP_CUBIN_TAG_RE = re.compile(r"\bsm_(\d+a?)\b")


def _list_so_cubins(
    so_path: Path,
    timeout: float = 30.0,
) -> Tuple[Optional[List[str]], int, str]:
    """Return ``(cubin_arch_tags, rc, stderr_tail)`` for a built .so.

    ``cubin_arch_tags`` is None when ``cuobjdump`` is absent or fails,
    and a (possibly empty) list otherwise. The list is the ordered set
    of unique ``sm_*`` tags in the ELF listing — e.g. ``["sm_86"]`` for
    a local-arch tracer build, ``["sm_86", "sm_100a"]`` for a B200-
    ready fat build.
    """
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        return None, 127, "cuobjdump not on PATH"
    rc, stdout, stderr = _run(
        [cuobjdump, "--list-elf", str(so_path)],
        timeout=timeout,
    )
    if rc != 0:
        tail = stderr.strip().splitlines()[-3:] if stderr else []
        return None, rc, "\n".join(tail)
    # Preserve first-seen order so the manifest reports cubins in the
    # order cuobjdump enumerated them.
    seen: List[str] = []
    for line in stdout.splitlines():
        m = _CUOBJDUMP_CUBIN_TAG_RE.search(line)
        if m is None:
            continue
        tag = "sm_" + m.group(1)
        if tag not in seen:
            seen.append(tag)
    return seen, 0, ""


def probe_nvbit_metrics_tool() -> Dict[str, Any]:
    """Required-probe view of the metric extractor `.so` (Surface 1).

    Reports:

    - ``unavailable`` when the `.so` has not been built yet (operator
      should run ``make -C nvbit_tracer/metric_extractor``).
    - ``failed`` when the host is a B200 (compute capability ``10.x``)
      and the built `.so` does not include an ``sm_100a`` cubin. The
      operator must rebuild for the B200 target before fanout — the
      "committed `.so` lacks an `sm_100a` cubin on a B200 host"
      rejection path lives here.
    - ``ok`` otherwise: either the host is not a B200 (local sm_86 /
      sm_120 dev workflow — operator rebuilds on the provider) or the
      `.so` is B200-ready. The substatus carries the cubin list either
      way so the operator can confirm coverage at a glance.

    Registered in ``_RECOMMENDED_PROBES`` so a local development host
    with an sm_86-only `.so` does not get blocked unnecessarily.
    Hardened in ``build_manifest()`` with a host-conditional rule:
    when the device probe reports ``phase_2_target == True`` (the
    host IS a B200), this probe's non-``ok`` status is appended to
    ``blockers`` and ``ready_for_phase_2`` flips to ``False``. The
    "wrong-arch `.so` slipping into a B200 fanout" failure mode is
    therefore caught at the gate, not just recorded as advisory.
    """
    so = REPO_ROOT / "nvbit_tracer" / "metric_extractor" / "metric_extractor.so"
    if not so.is_file():
        return {
            "status": "unavailable",
            "expected_path": str(so),
            "waiver_reason": (
                "metric_extractor.so not built; run "
                "`make -C nvbit_tracer/metric_extractor`."
            ),
        }

    cubins, rc, stderr_tail = _list_so_cubins(so)
    if cubins is None:
        return {
            "status": "failed",
            "expected_path": str(so),
            "exit_code": rc,
            "stderr_tail": stderr_tail,
            "waiver_reason": (
                "cuobjdump --list-elf failed on metric_extractor.so; "
                "cannot verify cubin coverage."
            ),
        }

    has_sm_100a = "sm_100a" in cubins
    device = probe_device()
    phase_2_target = bool(device.get("phase_2_target"))
    compute_cap = device.get("compute_capability")

    base = {
        "library_path": str(so),
        "size_bytes": so.stat().st_size,
        "mtime_utc": datetime.datetime.fromtimestamp(
            so.stat().st_mtime, tz=datetime.timezone.utc
        ).isoformat(),
        "substatus": {
            "cubins_present": cubins,
            "sm_100a_present": has_sm_100a,
            "host_compute_capability": compute_cap,
            "host_phase_2_target": phase_2_target,
        },
    }

    if phase_2_target and not has_sm_100a:
        return {
            "status": "failed",
            "waiver_reason": (
                f"metric_extractor.so on this B200 host (cc={compute_cap}) "
                f"lacks an sm_100a cubin (cubins_present={cubins}); "
                "rebuild with "
                "`make -C nvbit_tracer/metric_extractor ARCH=sm_100a` "
                "(or a multi-arch build covering both sm_86 and sm_100a) "
                "before kicking off the 235-baseline fanout."
            ),
            **base,
        }

    return {"status": "ok", **base}


def probe_storage(artifact_path: Path) -> Dict[str, Any]:
    """Disk-space probe.

    Reports ``status: ok`` always when the disk read succeeds; a
    free-disk reading under the warn threshold flips the separate
    ``warn`` field to ``True`` (locked status vocabulary keeps
    ``status`` strictly in ``{ok, unavailable, blocked, failed,
    skipped}``).
    """
    path = artifact_path
    while not path.exists() and path.parent != path:
        path = path.parent
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        return {
            "status": "failed",
            "waiver_reason": f"disk_usage error on {path}: {exc}",
        }
    free_gb = usage.free / (1024 ** 3)
    return {
        "status": "ok",
        "warn": free_gb < 20.0,
        "path": str(path),
        "free_gb": round(free_gb, 2),
        "total_gb": round(usage.total / (1024 ** 3), 2),
        "warn_threshold_gb": 20.0,
    }




def probe_sm_100a_cubin_disasm() -> Dict[str, Any]:
    """Run the Surface 4 cubin-extraction chain end-to-end for sm_100a.

    Sequence:
      1. Compile a stub .cu to a fatbinary-bearing shared library
         (.so) embedding sm_100a device code.
      2. Run ``cuobjdump -xelf all -arch=sm_100a`` against the .so
         to extract the per-arch cubin into a temp directory.
      3. Run ``nvdisasm`` against the EXTRACTED cubin (NOT the raw
         compile-output, which would skip the cuobjdump step the
         M3/M4 static surface depends on).
      4. Assert the disassembly carries SASS mnemonics.

    Reports ``ok`` only when all four steps succeed. ``failed`` when
    any step exits non-zero, with the offending step + stderr
    recorded. ``unavailable`` when nvcc/cuobjdump/nvdisasm are
    missing from PATH.
    """
    nvdisasm = shutil.which("nvdisasm")
    cuobjdump = shutil.which("cuobjdump")
    nvcc = shutil.which("nvcc")
    if nvdisasm is None or cuobjdump is None or nvcc is None:
        missing = []
        if nvdisasm is None:
            missing.append("nvdisasm")
        if cuobjdump is None:
            missing.append("cuobjdump")
        if nvcc is None:
            missing.append("nvcc")
        return {
            "status": "unavailable",
            "waiver_reason": (
                f"missing on PATH: {missing}; Surface 4 cubin "
                "extraction + disasm chain cannot run."
            ),
        }

    # Non-empty kernel body so the compiler emits actual SASS
    # (an empty body optimizes to a single EXIT and may not
    # round-trip through the cuobjdump extraction with code).
    stub_source = (
        "extern \"C\" __global__ void touch_mem(int* out) {\n"
        "    if (threadIdx.x == 0) out[0] = 42;\n"
        "}\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        stub_cu = tmp_path / "stub_sm_100a.cu"
        stub_cu.write_text(stub_source)
        # Compile to a host shared library carrying an sm_100a
        # fatbinary — this is the artifact Surface 4 will run
        # cuobjdump against in the live fanout.
        stub_so = tmp_path / "stub_sm_100a.so"
        rc, _, stderr = _run(
            [
                nvcc,
                "-O0",
                "-Xcompiler", "-fPIC",
                "-shared",
                "-gencode", "arch=compute_100a,code=sm_100a",
                "-o", str(stub_so),
                str(stub_cu),
            ],
            timeout=60.0,
        )
        if rc != 0:
            return {
                "status": "failed",
                "exit_code": rc,
                "stderr": stderr.strip()[:500],
                "substatus": {"step_failed": "nvcc_compile_so"},
                "waiver_reason": (
                    "nvcc -gencode arch=compute_100a,code=sm_100a "
                    "-shared rejected the stub; toolchain cannot "
                    "codegen the sm_100a fatbinary Surface 4 needs."
                ),
            }

        # Extract the per-arch cubin via cuobjdump scoped to an
        # explicit extraction dir. cuobjdump -xelf all writes
        # <basename>.<arch>.cubin into the CURRENT directory; we
        # run with cwd=extract_dir so files don't leak into the
        # caller's working directory.
        extract_dir = tmp_path / "cubins"
        extract_dir.mkdir()
        rc, stdout, stderr = _run_in(
            extract_dir,
            [
                cuobjdump,
                "-xelf", "all",
                "-arch", "sm_100a",
                str(stub_so),
            ],
            timeout=30.0,
        )
        if rc != 0:
            return {
                "status": "failed",
                "exit_code": rc,
                "stderr": stderr.strip()[:500],
                "substatus": {"step_failed": "cuobjdump_extract"},
                "waiver_reason": (
                    "cuobjdump -xelf all -arch=sm_100a failed to "
                    "extract the cubin from the stub .so; the static "
                    "surface extraction step that Surface 4 depends "
                    "on cannot run."
                ),
            }

        extracted = sorted(extract_dir.glob("*.cubin"))
        if not extracted:
            return {
                "status": "failed",
                "substatus": {
                    "step_failed": "cuobjdump_no_cubins_emitted",
                    "extract_dir": str(extract_dir),
                },
                "waiver_reason": (
                    "cuobjdump exited 0 but emitted no .cubin files "
                    "under the per-arch extraction dir; Surface 4 "
                    "would have nothing to feed to nvdisasm."
                ),
            }

        # nvdisasm each extracted cubin; cuobjdump may emit several
        # (a linker-stub cubin + a per-kernel SASS cubin). Surface 4
        # will need to disassemble all of them, so the probe walks
        # every extracted cubin and asserts at least ONE carries
        # genuine SASS mnemonics.
        sass_found_in: Optional[Path] = None
        last_nvdisasm_text = ""
        last_stderr = ""
        last_rc = 0
        for cubin in extracted:
            rc, stdout, stderr = _run(
                [nvdisasm, str(cubin)],
                timeout=15.0,
            )
            last_nvdisasm_text = stdout
            last_stderr = stderr
            last_rc = rc
            if rc != 0:
                return {
                    "status": "failed",
                    "exit_code": rc,
                    "stderr": stderr.strip()[:500],
                    "substatus": {
                        "step_failed": "nvdisasm_on_extracted_cubin",
                        "cubin_path": str(cubin),
                    },
                    "waiver_reason": (
                        "nvdisasm rejected one of the cuobjdump-extracted "
                        "sm_100a cubins; M3/M4 static extensions cannot "
                        "run on this host."
                    ),
                }
            if any(
                tok in stdout
                for tok in ("EXIT", "RET", "STG", "S2R", "MOV", "BAR")
            ):
                sass_found_in = cubin
                break

        if sass_found_in is None:
            return {
                "status": "failed",
                "substatus": {
                    "step_failed": "no_sass_mnemonic_in_any_cubin",
                    "extracted_cubins_count": len(extracted),
                    "last_disasm_length_bytes": len(last_nvdisasm_text),
                },
                "waiver_reason": (
                    "nvdisasm produced output for each extracted cubin "
                    "but none carried a recognizable SASS mnemonic; "
                    "the toolchain may have stripped the kernel body."
                ),
            }

        return {
            "status": "ok",
            "command": (
                "nvcc -shared -gencode arch=compute_100a,code=sm_100a"
                " + cuobjdump -xelf all -arch sm_100a + nvdisasm"
            ),
            "exit_code": 0,
            "substatus": {
                "extracted_cubins_count": len(extracted),
                "sass_found_in_cubin": sass_found_in.name,
            },
        }


def _run_in(
    cwd: Path,
    cmd: List[str],
    timeout: float = 30.0,
) -> Tuple[int, str, str]:
    """Run a subprocess with an explicit CWD."""
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return completed.returncode, completed.stdout, completed.stderr
    except FileNotFoundError:
        return 127, "", f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"command timed out after {timeout}s: {' '.join(cmd)}"


def probe_cache_replay_native_available() -> Dict[str, Any]:
    """Check the compiled cache-replay extension AND its measured throughput.

    Reports ``ok`` only when ALL of:
      1. ``cache_analytics_replay_native`` is importable.
      2. A 4 M-record synthetic microbench (sector-lookup pattern)
         clears
         ``THROUGHPUT_FLOOR_SECTOR_LOOKUPS_PER_SEC = 20_000_000``.

    Reports ``degraded`` when the extension is importable but
    throughput is below floor — the 235-pilot fanout's < 2 h replay
    budget is no longer guaranteed but a local single-pilot run can
    proceed.

    Reports ``unavailable`` when the extension does not import. The
    operator opt-out env ``SIMBENCH_ALLOW_PYTHON_REPLAY_FANOUT=1``
    downgrades this case to ``degraded`` so the operator can run
    the slow Python fallback explicitly. The orchestrator's
    ``ready_for_phase_2`` logic excludes the ``degraded`` status
    from blockers.

    Import-only success without measured throughput is NOT a
    valid ``ok`` path; the throughput floor must be measured.
    """
    operator_opt_out = bool(os.environ.get(_PYTHON_REPLAY_FANOUT_OPT_OUT_ENV))
    THROUGHPUT_FLOOR_SECTOR_LOOKUPS_PER_SEC = 20_000_000

    try:
        import cache_analytics_replay_native  # type: ignore  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        substatus = {
            "import_error": str(exc),
            "expected_module": "cache_analytics_replay_native",
            "operator_opt_out_env": _PYTHON_REPLAY_FANOUT_OPT_OUT_ENV,
            "operator_opt_out_engaged": operator_opt_out,
            "throughput_floor_sector_lookups_per_sec":
                THROUGHPUT_FLOOR_SECTOR_LOOKUPS_PER_SEC,
        }
        if operator_opt_out:
            return {
                "status": "degraded",
                "substatus": substatus,
                "waiver_reason": (
                    "Compiled cache_analytics_replay_native extension is "
                    f"not built; {_PYTHON_REPLAY_FANOUT_OPT_OUT_ENV}=1 "
                    "downgrades this probe from required-for-fanout to "
                    "required-prefer. The Python backend is the fallback "
                    "and the 235-pilot < 2 h replay budget is no longer "
                    "guaranteed."
                ),
            }
        return {
            "status": "unavailable",
            "substatus": substatus,
            "waiver_reason": (
                "Compiled cache_analytics_replay_native extension is "
                "not importable. Build it via the M4 milestone before "
                f"the 235-pilot fanout, or set "
                f"{_PYTHON_REPLAY_FANOUT_OPT_OUT_ENV}=1 to accept the "
                "slower Python replay path with a 'degraded' substatus."
            ),
        }

    # Extension is importable — run the throughput microbench.
    bench_result = _run_replay_throughput_microbench(
        cache_analytics_replay_native,
        floor_lookups_per_sec=THROUGHPUT_FLOOR_SECTOR_LOOKUPS_PER_SEC,
    )

    substatus = {
        "module_loaded_from": getattr(
            cache_analytics_replay_native, "__file__", "<built-in>"
        ),
        "throughput_floor_sector_lookups_per_sec":
            THROUGHPUT_FLOOR_SECTOR_LOOKUPS_PER_SEC,
        **bench_result,
    }
    if bench_result.get("microbench_status") == "ok":
        return {"status": "ok", "substatus": substatus}
    if bench_result.get("microbench_status") == "below_floor":
        return {
            "status": "degraded",
            "substatus": substatus,
            "waiver_reason": (
                "Compiled cache_analytics_replay_native imported, "
                f"but microbench measured "
                f"{bench_result.get('measured_lookups_per_sec', 0):.0f} "
                f"lookups/sec — below the "
                f"{THROUGHPUT_FLOOR_SECTOR_LOOKUPS_PER_SEC} floor "
                "required for the 235-pilot fanout's < 2 h replay "
                "budget. Local single-pilot runs may proceed."
            ),
        }
    # Microbench errored out (e.g. extension's API changed
    # incompatibly).
    return {
        "status": "failed",
        "substatus": substatus,
        "waiver_reason": (
            "cache_analytics_replay_native imported but the "
            f"throughput microbench failed: "
            f"{bench_result.get('error', 'unknown error')}"
        ),
    }


def _run_replay_throughput_microbench(
    extension: Any,
    *,
    floor_lookups_per_sec: int,
) -> Dict[str, Any]:
    """Run a 4 M-record synthetic sector-lookup microbench against the
    compiled cache-replay extension.

    Returns a dict with ``microbench_status`` in
    {ok, below_floor, error}, plus measured fields when reachable.

    The extension's API surface is not yet fixed (it's M4 work). The
    canonical entry point this probe expects is
    ``extension.replay_microbench(num_records: int) -> float`` that
    returns the elapsed seconds. If the extension does NOT yet
    expose that symbol, the probe records the API mismatch as an
    error and lets the operator decide whether to ship a stub.
    """
    NUM_RECORDS = 4_000_000

    try:
        replay_microbench = getattr(extension, "replay_microbench", None)
        if replay_microbench is None:
            return {
                "microbench_status": "error",
                "error": (
                    "extension does not expose replay_microbench(n) "
                    "entry point; throughput microbench cannot run"
                ),
            }
        elapsed = float(replay_microbench(NUM_RECORDS))
    except Exception as exc:
        return {
            "microbench_status": "error",
            "error": f"replay_microbench raised: {type(exc).__name__}: {exc}",
        }

    if elapsed <= 0:
        return {
            "microbench_status": "error",
            "error": (
                f"replay_microbench returned non-positive elapsed="
                f"{elapsed!r}"
            ),
        }

    lookups_per_sec = NUM_RECORDS / elapsed
    status = "ok" if lookups_per_sec >= floor_lookups_per_sec else "below_floor"
    return {
        "microbench_status": status,
        "microbench_records": NUM_RECORDS,
        "elapsed_sec": elapsed,
        "measured_lookups_per_sec": lookups_per_sec,
    }




def _required_probes_for_mode(mode: str) -> Tuple[str, ...]:
    """Return the required-probe tuple for the given mode.

    Phase-1 (local-smoke) drops the B200-specific cross-compile
    probes; Phase-2 (B200-fanout) keeps the full legacy set.
    """
    if mode == MODE_PHASE_1_LOCAL_SMOKE:
        return tuple(
            name
            for name in _REQUIRED_PROBES
            if name not in _PHASE_2_ONLY_REQUIRED_PROBES
        )
    return tuple(_REQUIRED_PROBES)


def _recommended_probes_for_mode(mode: str) -> Tuple[str, ...]:
    """Return the recommended-probe tuple for the given mode.

    Phase-1 promotes the sm_100a B200-cross-compile probes from
    required to recommended so a local sm_120 host can still report
    their status without that status blocking exit.
    """
    if mode == MODE_PHASE_1_LOCAL_SMOKE:
        return tuple(_PHASE_2_ONLY_REQUIRED_PROBES) + tuple(_RECOMMENDED_PROBES)
    return tuple(_RECOMMENDED_PROBES)


def build_manifest(
    artifact_path: Path,
    mode: str = _DEFAULT_MODE,
) -> Dict[str, Any]:
    """Run every probe and assemble the manifest.

    ``mode`` controls which probes block exit. ``phase2-b200-fanout``
    (default) is the legacy contract used before the 235-pilot
    fanout. ``phase1-local-smoke`` is the narrower contract used for
    the local sm_120 dev loop — see the module docstring and the
    ``_required_probes_for_mode`` helper.
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"invalid preflight mode {mode!r}; expected one of {_VALID_MODES}"
        )

    checks: Dict[str, Dict[str, Any]] = {
        "nvcc": probe_nvcc(),
        "driver": probe_driver(),
        "cuda_runtime": probe_cuda_runtime(),
        "device": probe_device(),
        "cutlass": probe_cutlass(),
        "sm_100a_compile_proof": probe_sm_100a_compile(),
        "sm_100a_cubin_disasm": probe_sm_100a_cubin_disasm(),
        "cache_replay_native_available": probe_cache_replay_native_available(),
        "nsys": probe_nsys(),
        "compute_sanitizer": probe_compute_sanitizer(),
        "cupti": probe_cupti(),
        "nvbit_metrics_tool": probe_nvbit_metrics_tool(),
        "storage": probe_storage(artifact_path),
    }

    required_probes = _required_probes_for_mode(mode)
    recommended_probes = _recommended_probes_for_mode(mode)

    # Required-for-fanout blocker computation. A `cache_replay_native_available`
    # probe with status `degraded` does NOT block (operator opt-out via
    # SIMBENCH_ALLOW_PYTHON_REPLAY_FANOUT=1). Every other required probe
    # must be `ok`.
    blockers: List[str] = []
    for name in required_probes:
        s = checks.get(name, {}).get("status")
        if s == "ok":
            continue
        if name == "cache_replay_native_available" and s == "degraded":
            continue
        blockers.append(name)

    # Host-conditional escalation: ``nvbit_metrics_tool`` is in
    # ``_REQUIRED_PROBES``. A non-`ok` status on a B200 host
    # (compute capability ``10.x``) means the committed `.so`
    # cannot drive the 235-baseline fanout: either the file is
    # missing, ``cuobjdump`` cannot read it, or its cubin coverage
    # lacks ``sm_100a``. In that case the probe escalates from
    # advisory to blocking so a wrong-arch `.so` cannot silently
    # slip through. The escalation only applies in Phase-2 mode —
    # a Phase-1 local-smoke run on any host (including a B200
    # provider) is explicitly opting out of B200-strict gating.
    device_check = checks.get("device", {})
    nvbit_metrics_check = checks.get("nvbit_metrics_tool", {})
    if (
        mode == MODE_PHASE_2_B200_FANOUT
        and device_check.get("phase_2_target") is True
        and nvbit_metrics_check.get("status") != "ok"
        and "nvbit_metrics_tool" not in blockers
    ):
        blockers.append("nvbit_metrics_tool")

    return {
        "schema_version": 2,
        "captured_at_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "mode": mode,
        "host": {
            "kernel": platform.platform(),
            "arch": platform.machine(),
            "python_version": sys.version.split()[0],
            "hostname": platform.node(),
        },
        "required_probes": list(required_probes),
        "required_prefer_probes": list(_REQUIRED_PREFER_PROBES),
        "recommended_probes": list(recommended_probes),
        "checks": checks,
        "blockers": blockers,
        "ready_for_phase_2": not blockers,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/baselines_harness/b200_preflight.py",
        description=(
            "Probe the current host for the toolchain prerequisites of "
            "the six counter-free analytic surfaces; emit "
            "b200_ready_manifest.json. Read-only; never mutates clock "
            "or power state."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "scripts"
        / "baselines_harness"
        / "b200_ready_manifest.json",
        help="Output manifest path.",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=REPO_ROOT / "scripts" / "baselines_harness" / ".harness_traces",
        help=(
            "Path the storage probe checks for free disk space "
            "(default: scripts/baselines_harness/.harness_traces). "
            "Walks upward if the path doesn't exist yet."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=list(_VALID_MODES),
        default=_DEFAULT_MODE,
        help=(
            "Preflight contract. 'phase2-b200-fanout' (default) is the "
            "legacy 235-pilot B200-fanout contract — requires "
            "sm_100a_compile_proof and sm_100a_cubin_disasm, and "
            "escalates nvbit_metrics_tool to blocking on a B200 host. "
            "'phase1-local-smoke' is the narrower local sm_120 / sm_86 "
            "dev-loop contract — sm_100a probes drop to recommended and "
            "the nvbit_metrics_tool host-conditional escalation is "
            "disabled."
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        manifest = build_manifest(args.artifact_path, mode=args.mode)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            f"b200_preflight: failed to emit manifest: {exc}\n"
        )
        return 2
    if manifest["ready_for_phase_2"]:
        print(
            f"b200_preflight: READY ({args.output}). "
            f"Required probes ok; recommended probe statuses recorded."
        )
        return 0
    sys.stderr.write(
        f"b200_preflight: NOT READY ({args.output}). "
        f"Blocked on: {manifest['blockers']}\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
