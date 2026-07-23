#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SOL ExecBench evaluation driver.

Self-contained script written to the GPU staging directory.
Evaluates a user solution and outputs JSONL Trace objects to stdout.
All non-JSON output (library messages, Triton JIT logs) goes to stderr.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import sys
import threading
import traceback
from pathlib import Path
from typing import Optional

# ── Redirect stdout → stderr BEFORE importing torch/triton ──────────────────
# Saves the original stdout fd so we can print JSON to it later.
_real_stdout_fd = os.dup(1)
_real_stdout = os.fdopen(_real_stdout_fd, "w", buffering=1)
os.dup2(2, 1)  # fd 1 now points at stderr
sys.stdout = open(1, "w", buffering=1, closefd=False)

import torch  # noqa: E402 — must come after redirect

# ── Staging directory ────────────────────────────────────────────────────────
STAGING_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(STAGING_DIR))


# ── DPS helper ───────────────────────────────────────────────────────────────
def _call_and_collect_outputs(
    fn,
    inputs,
    dps,
    definition,
    resolved_axes,
    device,
    output_names,
    output_dtypes_torch,
):
    """Call fn in the correct calling convention and return a list of output tensors.

    DPS=True (destination-passing style):
        fn(*inputs, *pre_allocated_outputs)  → outputs written in-place
    DPS=False (return-value style):
        result = fn(*inputs)  → normalize_outputs(result) → list of tensors
    """
    if dps:
        outputs = allocate_outputs(definition, resolved_axes, device)
        fn(*inputs, *outputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        return outputs
    else:
        result = fn(*inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        out_dict = normalize_outputs(
            result,
            device=torch.device(device),
            output_names=output_names,
            output_dtypes=output_dtypes_torch,
        )
        return [out_dict[n] for n in output_names]


# ── Load problem ─────────────────────────────────────────────────────────────
def _load_problem() -> "tuple[dict, list[dict]]":
    """Load definition dict + workload dicts from staging directory."""
    local_def = STAGING_DIR / "definition.json"
    local_wkl = STAGING_DIR / "workload.jsonl"

    if not local_def.exists():
        raise RuntimeError(
            "definition.json not found in staging directory — "
            "client must supply definition and workloads inline"
        )

    definition = json.loads(local_def.read_text())
    workloads: list[dict] = []
    if local_wkl.exists():
        for line in local_wkl.read_text().splitlines():
            line = line.strip()
            if line:
                workloads.append(json.loads(line))
    return definition, workloads


definition_dict, _workload_dicts = _load_problem()


# ── Optional persistent reference-capture contract ───────────────────────────
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value) -> str:
    data = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    return hashlib.sha256(data).hexdigest()


def _capture_entry(entry, label, *, stage_relative=False):
    if not isinstance(entry, dict):
        raise RuntimeError(f"capture contract {label} is not an object")
    raw_path = entry.get("path")
    expected = entry.get("sha256")
    if not isinstance(raw_path, str) or not raw_path or not isinstance(expected, str):
        raise RuntimeError(f"capture contract {label} lacks path/sha256")
    declared = Path(raw_path)
    if stage_relative:
        if declared.is_absolute() or ".." in declared.parts or declared == Path("."):
            raise RuntimeError(f"capture contract {label} is not stage-relative")
        path = (STAGING_DIR / declared).resolve()
        try:
            path.relative_to(STAGING_DIR)
        except ValueError as exc:
            raise RuntimeError(f"capture contract {label} escapes stage") from exc
    else:
        if not declared.is_absolute():
            raise RuntimeError(f"capture contract {label} is not absolute")
        path = declared
    if not path.is_file():
        raise RuntimeError(f"capture contract {label} path is absent: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"capture contract {label} hash mismatch: expected={expected} actual={actual}"
        )
    return path


def _load_capture_contract():
    """Validate the pinned workload and every staged byte before GPU work."""
    if os.environ.get("SOL_EXECBENCH_CAPTURE_MODE") != "1":
        return None
    required_env = (
        "SOL_EXECBENCH_CAPTURE_PREPARE_MANIFEST",
        "SOL_EXECBENCH_CAPTURE_SELECTION_MANIFEST",
        "SOL_EXECBENCH_CAPTURE_SELECTION_SHA256",
        "SOL_EXECBENCH_CAPTURE_RESULT_MANIFEST",
        "SOL_EXECBENCH_CAPTURE_SOURCE_WORKLOAD_INDEX",
    )
    absent = [name for name in required_env if not os.environ.get(name)]
    if absent:
        raise RuntimeError(f"capture contract environment is incomplete: {absent}")
    try:
        source_index = int(os.environ["SOL_EXECBENCH_CAPTURE_SOURCE_WORKLOAD_INDEX"])
    except ValueError as exc:
        raise RuntimeError("capture workload index is not an integer") from exc
    if source_index < 0:
        raise RuntimeError("capture workload index must be nonnegative")
    if len(_workload_dicts) != 1:
        raise RuntimeError(
            f"capture stage must contain exactly one workload, found {len(_workload_dicts)}"
        )

    prepare_path = Path(os.environ["SOL_EXECBENCH_CAPTURE_PREPARE_MANIFEST"])
    selection_path = Path(os.environ["SOL_EXECBENCH_CAPTURE_SELECTION_MANIFEST"])
    result_path = Path(os.environ["SOL_EXECBENCH_CAPTURE_RESULT_MANIFEST"])
    if prepare_path != STAGING_DIR / "capture_prepare_manifest.json":
        raise RuntimeError("capture preparation manifest path does not match staging dir")
    if not selection_path.is_absolute() or not result_path.is_absolute():
        raise RuntimeError("capture selection/result paths must be absolute")
    try:
        selection_relative = selection_path.resolve().relative_to(STAGING_DIR).as_posix()
        result_relative = result_path.resolve().relative_to(STAGING_DIR).as_posix()
    except ValueError as exc:
        raise RuntimeError("capture selection/result must be inside staging dir") from exc
    if not result_path.parent.is_dir():
        raise RuntimeError(f"capture result parent is absent: {result_path.parent}")
    if result_path.exists():
        raise RuntimeError(f"capture result already exists: {result_path}")

    prepare = json.loads(prepare_path.read_text())
    selection = json.loads(selection_path.read_text())
    if not isinstance(prepare, dict) or not isinstance(selection, dict):
        raise RuntimeError("capture preparation/selection manifest must be objects")
    if (
        prepare.get("schema_version") != 1
        or prepare.get("capture_contract_version") != 2
        or prepare.get("manifest_kind") != "sol_execbench_capture_preparation"
        or prepare.get("path_mode") != "stage_relative_v1"
        or prepare.get("stage_root") != "."
        or "stage_dir" in prepare
    ):
        raise RuntimeError("capture preparation manifest identity mismatch")
    environment_contract = prepare.get("environment_contract")
    certified_environment = {
        "CUDA_HOME": "/usr/local/cuda-13.2",
        "NVCC_CCBIN": "g++-14",
        "CC": "gcc-14",
        "CXX": "g++-14",
        "CUDAHOSTCXX": "g++-14",
    }
    if environment_contract != certified_environment:
        raise RuntimeError("capture environment contract is not certified")
    environment_contract_sha256 = _canonical_sha256(environment_contract)
    if (
        prepare.get("environment_contract_sha256")
        != environment_contract_sha256
    ):
        raise RuntimeError("capture environment contract digest mismatch")
    flashinfer_contract = prepare.get("flashinfer_corpus_contract")
    if not isinstance(flashinfer_contract, dict):
        raise RuntimeError("capture FlashInfer corpus contract is absent")
    flashinfer_contract_sha256 = _canonical_sha256(flashinfer_contract)
    if (
        prepare.get("flashinfer_corpus_contract_sha256")
        != flashinfer_contract_sha256
    ):
        raise RuntimeError("capture FlashInfer corpus contract digest mismatch")
    category = flashinfer_contract.get("category")
    trace_dir = os.environ.get("FLASHINFER_TRACE_DIR")
    if category == "FIB":
        if (
            flashinfer_contract.get("injection_policy")
            != "verified_absolute_root"
            or flashinfer_contract.get("root")
            != "data/annex/flashinfer_trace"
            or trace_dir is None
            or not Path(trace_dir).is_absolute()
            or not Path(trace_dir).resolve().is_dir()
        ):
            raise RuntimeError("verified FIB trace corpus was not injected")
    elif (
        category not in {"L1", "L2", "Quant"}
        or flashinfer_contract.get("injection_policy")
        != "forbidden_remove_ambient"
        or flashinfer_contract.get("root") is not None
        or trace_dir is not None
    ):
        raise RuntimeError("non-FIB capture inherited a FlashInfer trace corpus")
    stage_without_digest = dict(prepare)
    claimed_stage_definition_sha256 = stage_without_digest.pop(
        "stage_definition_sha256", None
    )
    if claimed_stage_definition_sha256 != _canonical_sha256(stage_without_digest):
        raise RuntimeError("capture stage definition digest mismatch")
    if any(
        os.environ.get(name) != value
        for name, value in certified_environment.items()
    ):
        raise RuntimeError("capture process compiler environment is not certified")
    expected_nvcc = (
        Path(certified_environment["CUDA_HOME"]) / "bin" / "nvcc"
    ).resolve()
    observed_nvcc = shutil.which("nvcc")
    if (
        observed_nvcc is None
        or Path(observed_nvcc).resolve() != expected_nvcc
        or not expected_nvcc.is_file()
        or not os.access(expected_nvcc, os.X_OK)
    ):
        raise RuntimeError(
            f"capture process nvcc is not certified: {observed_nvcc!r}"
        )
    hazards = [
        name
        for name in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS")
        if name in os.environ
    ]
    if hazards:
        raise RuntimeError(f"capture process contains forbidden NVCC variables: {hazards}")
    selection_sha = _sha256_file(selection_path)
    if selection_sha != os.environ["SOL_EXECBENCH_CAPTURE_SELECTION_SHA256"]:
        raise RuntimeError("capture selection manifest hash mismatch")
    if (
        selection.get("schema_version") != 1
        or selection.get("capture_contract_version") != 2
        or selection.get("manifest_kind") != "sol_execbench_capture_selection"
        or selection.get("path_mode") != "stage_relative_v1"
        or selection.get("stage_root") != "."
        or "stage_dir" in selection
        or selection.get("source_workload_index") != source_index
        or selection.get("staged_workload_index") != 0
        or selection.get("result_manifest") != result_relative
        or selection.get("capture_kind") not in {"label", "trace", "control", "smoke"}
        or selection.get("environment_contract") != environment_contract
        or selection.get("environment_contract_sha256")
        != environment_contract_sha256
        or selection.get("flashinfer_corpus_contract") != flashinfer_contract
        or selection.get("flashinfer_corpus_contract_sha256")
        != flashinfer_contract_sha256
        or selection.get("stage_definition_sha256")
        != claimed_stage_definition_sha256
    ):
        raise RuntimeError("capture selection manifest identity mismatch")
    prepare_pin = selection.get("prepare_manifest")
    if (
        not isinstance(prepare_pin, dict)
        or prepare_pin.get("path") != "capture_prepare_manifest.json"
        or prepare_pin.get("sha256") != _sha256_file(prepare_path)
    ):
        raise RuntimeError("capture preparation manifest pin mismatch")

    prepare_selection = prepare.get("selection")
    if (
        not isinstance(prepare_selection, dict)
        or prepare_selection.get("source_workload_index") != source_index
        or prepare_selection.get("staged_workload_index") != 0
        or selection.get("workload") != prepare_selection
    ):
        raise RuntimeError("capture workload selection mismatch")
    raw_rows = [
        line.strip()
        for line in (STAGING_DIR / "workload.jsonl").read_bytes().splitlines()
        if line.strip()
    ]
    if len(raw_rows) != 1:
        raise RuntimeError("capture staged workload count changed during validation")
    if hashlib.sha256(raw_rows[0]).hexdigest() != prepare_selection.get(
        "source_row_sha256"
    ):
        raise RuntimeError("capture workload row hash mismatch")
    if _workload_dicts[0].get("uuid") != prepare_selection.get("workload_uuid"):
        raise RuntimeError("capture workload UUID mismatch")

    artifacts = prepare.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("capture preparation artifacts are absent")
    required_artifacts = {
        "definition": Path("definition.json"),
        "workload": Path("workload.jsonl"),
        "solution": Path("solution.json"),
        "eval_driver": Path("eval_driver.py"),
        "capture_runtime": Path("_capture_runtime.py"),
        "benchmark_kernel_so": Path("benchmark_kernel.so"),
    }
    for name, expected_relative in required_artifacts.items():
        entry = artifacts.get(name)
        actual_path = _capture_entry(
            entry, f"artifacts.{name}", stage_relative=True
        )
        if Path(entry["path"]) != expected_relative:
            raise RuntimeError(f"capture artifact {name} path mismatch")
        if actual_path != STAGING_DIR / expected_relative:
            raise RuntimeError(f"capture artifact {name} resolution mismatch")
    _capture_entry(artifacts.get("adapter"), "artifacts.adapter")
    _capture_entry(artifacts.get("adapter_source"), "artifacts.adapter_source")
    _capture_entry(
        artifacts.get("certified_env_source"), "artifacts.certified_env_source"
    )
    sources = artifacts.get("solution_sources")
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("capture solution source artifact set is empty")
    for index, source in enumerate(sources):
        _capture_entry(
            source, f"artifacts.solution_sources[{index}]", stage_relative=True
        )
    for key in ("benchmark_kernel_so", "eval_driver", "capture_runtime"):
        if selection.get(key) != artifacts.get(key):
            raise RuntimeError(f"capture selection {key} pin mismatch")

    if prepare.get("solution_contract", {}).get("definition") != definition_dict.get(
        "name"
    ):
        raise RuntimeError("capture definition identity mismatch")
    return {
        "prepare": prepare,
        "prepare_path": prepare_path,
        "selection": selection,
        "selection_path": selection_path,
        "selection_relative": selection_relative,
        "selection_sha256": selection_sha,
        "result_path": result_path,
        "source_workload_index": source_index,
    }


_CAPTURE_CONTRACT = _load_capture_contract()

# ── Imports from sol_execbench.core ─────────────────────────────────────────────
from sol_execbench.core.bench.clock_lock import are_clocks_locked  # noqa: E402
from sol_execbench.core.bench.config import BenchmarkConfig  # noqa: E402
from sol_execbench.core.bench.correctness import (  # noqa: E402
    compute_error_stats,
    set_seed,
)
from sol_execbench.core.bench.io import (  # noqa: E402
    allocate_outputs,
    gen_inputs,
    load_safetensors,
    normalize_outputs,
)
from sol_execbench.core.bench.reward_hack import (  # noqa: E402
    RewardHackDetected,
    check_eval_integrity,
    check_lazy_outputs,
    check_monkey_patch,
    check_thread_injection,
    snapshot_critical_functions,
)
# Loading the timing module imports CUPTI.  The opt-in capture path must coexist
# with an external profiler, so it never imports CUPTI and times its one call
# with CUDA events instead.  The canonical judge imports and uses the unchanged
# timing implementation when capture mode is absent.
if _CAPTURE_CONTRACT is None:
    from sol_execbench.core.bench.timing import time_runnable  # noqa: E402
else:
    time_runnable = None
from sol_execbench.core.bench.utils import (  # noqa: E402
    make_eval,
)
from sol_execbench.core import (  # noqa: E402
    Correctness,
    Definition,
    EvaluationStatus,
    Performance,
    Solution,
    SupportedLanguages,
    Trace,
    Workload,
)
from sol_execbench.core.data.dtypes import dtype_str_to_torch_dtype  # noqa: E402

# ── Load config ───────────────────────────────────────────────────────────────
_config_path = STAGING_DIR / "config.json"
bench_config = (
    BenchmarkConfig(**json.loads(_config_path.read_text()))
    if _config_path.exists()
    else BenchmarkConfig()
)

# ── Parse definition ──────────────────────────────────────────────────────────
definition = Definition(**definition_dict)
workloads = [Workload(**w) for w in _workload_dicts]

# ── Parse solution ────────────────────────────────────────────────────────────
_solution = Solution(**json.loads((STAGING_DIR / "solution.json").read_text()))
_solution_name = _solution.name
_entry_point = _solution.spec.entry_point
_dps = _solution.spec.destination_passing_style

# ── Exec reference code ───────────────────────────────────────────────────────
# Write to a real file so decorators that call inspect.getsourcelines()
# (e.g. @triton.jit) can read the source back.
_ref_file = STAGING_DIR / "_reference.py"
_ref_file.write_text(definition.reference)
try:
    _ref_spec = importlib.util.spec_from_file_location("_reference", _ref_file)
    _ref_module = importlib.util.module_from_spec(_ref_spec)
    _ref_spec.loader.exec_module(_ref_module)
except Exception as _ref_err:
    raise RuntimeError(f"Failed to exec reference code: {_ref_err}") from _ref_err

ref_namespace = vars(_ref_module)
ref_fn = ref_namespace.get("run")
if ref_fn is None:
    raise RuntimeError("Reference code does not define a top-level 'run' function")

# ── Integrity snapshot (before user code import) ─────────────────────────────
# Capture id() of every function that affects measurement or correctness.
# Checked after user code import and after each user_fn() call.
_CRITICAL_NAMES = [
    "compute_error_stats",
    "check_monkey_patch",
    "check_lazy_outputs",
    "check_thread_injection",
    "check_eval_integrity",
    "_call_and_collect_outputs",
    "gen_inputs",
    "allocate_outputs",
    "normalize_outputs",
    "make_eval",
]
# ``time_runnable`` is a canonical-judge dependency only.  Capture mode avoids
# importing its CUPTI-backed module and intentionally binds the name to None, so
# snapshotting that name would make the unchanged capture state look patched to
# ``check_eval_integrity``.  Keep every shared integrity guard active in both
# modes while preserving the canonical judge's original monitored-name set.
if _CAPTURE_CONTRACT is None:
    _CRITICAL_NAMES.insert(0, "time_runnable")
_integrity_snapshot = snapshot_critical_functions(globals(), _CRITICAL_NAMES)
# Keep a local reference so that patching the name in globals() is ineffective.
_check_integrity = check_eval_integrity

# ── Resolve user function ─────────────────────────────────────────────────────
if "::" in _entry_point:
    _entry_module_or_file, _entry_func_name = _entry_point.rsplit("::", 1)
else:
    _entry_module_or_file, _entry_func_name = _entry_point, "run"

_CPP_LANGUAGES = {
    SupportedLanguages.CUDA_CPP,
    SupportedLanguages.CUTLASS,
    SupportedLanguages.CUDNN,
    SupportedLanguages.CUBLAS,
}
if any(lang in _CPP_LANGUAGES for lang in _solution.spec.languages):
    _so_path = STAGING_DIR / "benchmark_kernel.so"
    if not _so_path.exists():
        raise RuntimeError(f"benchmark_kernel.so not found at {_so_path}")
    _spec_obj = importlib.util.spec_from_file_location("benchmark_kernel", _so_path)
    _user_mod = importlib.util.module_from_spec(_spec_obj)
    _spec_obj.loader.exec_module(_user_mod)
    user_fn = getattr(_user_mod, _entry_func_name)
else:
    # Block torch.utils.cpp_extension.load/load_inline on the GPU server.
    # CUDA/C++ compilation must go through Phase 1 on the compile server
    # (use a C++ language like "cuda_cpp" in the solution spec).
    import torch.utils.cpp_extension as _cpp_ext

    def _blocked_cpp_ext_load(*args, **kwargs):
        raise RuntimeError(
            "torch.utils.cpp_extension.load() and load_inline() are not permitted "
            'on the GPU server. Use a C++ language (e.g. "cuda_cpp") in your '
            "solution spec to compile on the compile server."
        )

    _cpp_ext.load = _blocked_cpp_ext_load
    _cpp_ext.load_inline = _blocked_cpp_ext_load

    sys.path.insert(0, str(STAGING_DIR))
    _mod_name = (
        _entry_module_or_file.removesuffix(".py").replace("/", ".").replace(os.sep, ".")
    )
    _user_mod = importlib.import_module(_mod_name)
    user_fn = getattr(_user_mod, _entry_func_name)

# ── Device ────────────────────────────────────────────────────────────────────
_device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ── Safetensors blob roots ────────────────────────────────────────────────────
# Priority: 1) staging dir (client-inlined blobs), 2) flashinfer-trace directory.
_safetensors_roots = [STAGING_DIR]
_benchmark_dir = os.environ.get("FLASHINFER_TRACE_DIR", None)
if _benchmark_dir:
    _safetensors_roots.append(Path(_benchmark_dir))


# ── Output helper ─────────────────────────────────────────────────────────────
def _emit(trace: Trace) -> None:
    """Write one Trace as strictly valid JSON to the real stdout.

    Uses allow_nan=False so that any unexpected NaN/Inf in the trace dict
    raises ValueError immediately instead of producing the non-standard
    JavaScript literal ``NaN`` which strict JSON parsers silently reject.
    """
    print(
        json.dumps(trace.model_dump(mode="json"), allow_nan=False),
        file=_real_stdout,
        flush=True,
    )


def _reward_hack_check(workload, check_fn, *args, suppress_errors=False):
    """Run a reward-hack check; emit REWARD_HACK trace and return True if detected."""
    try:
        check_fn(*args)
    except RewardHackDetected as e:
        _emit(
            Trace(
                definition=definition.name,
                solution=_solution_name,
                workload=workload,
                evaluation=_make_eval(
                    EvaluationStatus.REWARD_HACK, _device, None, extra_msg=str(e)
                ),
            )
        )
        return True
    except Exception:
        if not suppress_errors:
            raise
    return False


# ── Pre-compute output metadata (used by normalize_outputs for DPS=false) ─────
_output_names = list(definition.outputs.keys())
_output_dtypes_torch = {
    k: dtype_str_to_torch_dtype(v.dtype) for k, v in definition.outputs.items()
}


# ── Opt-in one-workload profiler capture ─────────────────────────────────────
def _publish_capture_result(
    *, status, outside_roi, roi, correctness=None, error=None
):
    contract = _CAPTURE_CONTRACT
    result_path = contract["result_path"]
    artifact = contract["prepare"]["artifacts"]["benchmark_kernel_so"]
    result = {
        "schema_version": 1,
        "manifest_kind": "sol_execbench_capture_result",
        "capture_contract_version": 2,
        "path_mode": "stage_relative_v1",
        "stage_root": ".",
        "status": status,
        "capture_kind": contract["selection"]["capture_kind"],
        "selection_manifest": {
            "path": contract["selection_relative"],
            "sha256": contract["selection_sha256"],
        },
        "prepare_manifest": {
            "path": "capture_prepare_manifest.json",
            "sha256": _sha256_file(contract["prepare_path"]),
        },
        "definition": definition.name,
        "solution": _solution_name,
        "source_workload_index": contract["source_workload_index"],
        "staged_workload_index": 0,
        "workload_uuid": workloads[0].uuid,
        "workload_axes": workloads[0].axes,
        "environment_contract": contract["prepare"]["environment_contract"],
        "environment_contract_sha256": contract["prepare"][
            "environment_contract_sha256"
        ],
        "flashinfer_corpus_contract": contract["prepare"][
            "flashinfer_corpus_contract"
        ],
        "flashinfer_corpus_contract_sha256": contract["prepare"][
            "flashinfer_corpus_contract_sha256"
        ],
        "stage_definition_sha256": contract["prepare"][
            "stage_definition_sha256"
        ],
        "benchmark_kernel_so": artifact,
        "execution_order": [
            "correctness_outside_roi",
            "warmup_outside_roi",
            "cuda_synchronize",
            "cudaProfilerStart",
            "candidate_user_fn_once",
            "cuda_synchronize",
            "cudaProfilerStop_finally",
        ],
        "outside_roi": outside_roi,
        "roi": roi,
    }
    if correctness is not None:
        result["correctness"] = correctness
    if error is not None:
        result["error"] = error
    data = (
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    tmp = result_path.parent / f".{result_path.name}.{os.getpid()}.tmp"
    try:
        with tmp.open("xb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.link(tmp, result_path)
    finally:
        tmp.unlink(missing_ok=True)
    return 0 if status == "captured" else 1


def _capture_error(error_type, exc):
    detail = str(exc).replace(str(STAGING_DIR), ".")
    return {
        "type": error_type,
        "exception_type": type(exc).__name__,
        "detail": detail,
    }


def _run_capture_mode():
    """Correctness + warmup outside one exact CUDA profiler candidate call."""
    from _capture_runtime import ProfiledCandidateError, profile_candidate_once

    outside = {
        "correctness_rounds": 0,
        "correctness_user_fn_calls": 0,
        "warmup_user_fn_calls": 0,
        "all_completed_before_profiler_start": False,
    }
    empty_roi = {
        "profiler_start_calls": 0,
        "candidate_user_fn_calls": 0,
        "profiler_stop_calls": 0,
        "profiler_start_status": None,
        "profiler_stop_status": None,
        "latency_ms": None,
        "timing_methodology": "cuda_events",
    }
    if not torch.cuda.is_available():
        return _publish_capture_result(
            status="cuda_unavailable",
            outside_roi=outside,
            roi=empty_roi,
            error={"type": "cuda_unavailable", "detail": "CUDA is not available"},
        )
    if not _dps:
        return _publish_capture_result(
            status="unsupported_calling_convention",
            outside_roi=outside,
            roi=empty_roi,
            error={
                "type": "destination_passing_required",
                "detail": "capture outputs must be allocated before the profiler ROI",
            },
        )

    workload = workloads[0]
    resolved_axes = definition.get_resolved_axes_values(workload.axes)
    custom_inputs_fn = ref_namespace.get(definition.custom_inputs_entrypoint)
    set_seed(bench_config.seed)

    try:
        safe_tensors = {}
        if any(value.type == "safetensors" for value in workload.inputs.values()):
            safe_tensors = load_safetensors(
                definition, workload, _safetensors_roots
            )
        inputs = gen_inputs(
            definition,
            workload,
            device=_device,
            safe_tensors=safe_tensors or None,
            custom_inputs_fn=custom_inputs_fn,
        )
    except Exception as exc:
        return _publish_capture_result(
            status="input_generation_failed",
            outside_roi=outside,
            roi=empty_roi,
            error=_capture_error("input_generation_failed", exc),
        )

    try:
        reference_outputs = _call_and_collect_outputs(
            ref_fn,
            inputs,
            False,
            definition,
            resolved_axes,
            _device,
            _output_names,
            _output_dtypes_torch,
        )
    except Exception as exc:
        return _publish_capture_result(
            status="reference_failed",
            outside_roi=outside,
            roi=empty_roi,
            error=_capture_error("reference_failed", exc),
        )

    try:
        outside["correctness_user_fn_calls"] += 1
        user_outputs = _call_and_collect_outputs(
            user_fn,
            inputs,
            True,
            definition,
            resolved_axes,
            _device,
            _output_names,
            _output_dtypes_torch,
        )
        outside["correctness_rounds"] = 1
        _check_integrity(_integrity_snapshot, globals())
        check_lazy_outputs(user_outputs)
    except RewardHackDetected as exc:
        return _publish_capture_result(
            status="reward_hack_detected",
            outside_roi=outside,
            roi=empty_roi,
            error=_capture_error("reward_hack_detected", exc),
        )
    except Exception as exc:
        return _publish_capture_result(
            status="correctness_execution_failed",
            outside_roi=outside,
            roi=empty_roi,
            error=_capture_error("correctness_execution_failed", exc),
        )

    correctness = Correctness()
    if len(reference_outputs) != len(user_outputs) or not reference_outputs:
        return _publish_capture_result(
            status="correctness_failed",
            outside_roi=outside,
            roi=empty_roi,
            error={
                "type": "output_cardinality_mismatch",
                "detail": (
                    f"reference={len(reference_outputs)} user={len(user_outputs)}"
                ),
            },
        )
    for output_index in range(len(reference_outputs)):
        ref_output = reference_outputs[output_index]
        user_output = user_outputs[output_index]
        if ref_output.shape != user_output.shape:
            return _publish_capture_result(
                status="correctness_failed",
                outside_roi=outside,
                roi=empty_roi,
                error={"type": "incorrect_shape", "detail": "output shape mismatch"},
            )
        if ref_output.dtype != user_output.dtype:
            return _publish_capture_result(
                status="correctness_failed",
                outside_roi=outside,
                roi=empty_roi,
                error={"type": "incorrect_dtype", "detail": "output dtype mismatch"},
            )
        try:
            current, exceeds = compute_error_stats(
                user_output, ref_output, workload.tolerance
            )
        except Exception as exc:
            return _publish_capture_result(
                status="correctness_check_failed",
                outside_roi=outside,
                roi=empty_roi,
                error=_capture_error("correctness_check_failed", exc),
            )
        if current.max_absolute_error > correctness.max_absolute_error:
            correctness = current
        if current.has_nan or current.has_inf:
            correctness = current
        reference_outputs[output_index] = None
        user_outputs[output_index] = None
        del ref_output, user_output
        if exceeds:
            return _publish_capture_result(
                status="correctness_failed",
                outside_roi=outside,
                roi=empty_roi,
                correctness=correctness.model_dump(mode="json"),
                error={
                    "type": "incorrect_numerical",
                    "detail": "candidate exceeds workload tolerance",
                },
            )

    reference_outputs = None
    user_outputs = None
    gc.collect()
    torch.cuda.empty_cache()

    # One explicit warmup remains outside the profiler range.  The production
    # SOL roster is destination-passing, so both warmup and candidate outputs
    # are allocated before cudaProfilerStart.
    try:
        warmup_outputs = allocate_outputs(definition, resolved_axes, _device)
        outside["warmup_user_fn_calls"] += 1
        user_fn(*inputs, *warmup_outputs)
        torch.cuda.synchronize()
        _check_integrity(_integrity_snapshot, globals())
        check_monkey_patch()
    except RewardHackDetected as exc:
        return _publish_capture_result(
            status="reward_hack_detected",
            outside_roi=outside,
            roi=empty_roi,
            correctness=correctness.model_dump(mode="json"),
            error=_capture_error("reward_hack_detected", exc),
        )
    except Exception as exc:
        return _publish_capture_result(
            status="warmup_failed",
            outside_roi=outside,
            roi=empty_roi,
            correctness=correctness.model_dump(mode="json"),
            error=_capture_error("warmup_failed", exc),
        )

    warmup_outputs = None
    gc.collect()
    torch.cuda.empty_cache()

    try:
        roi_outputs = allocate_outputs(definition, resolved_axes, _device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
    except Exception as exc:
        return _publish_capture_result(
            status="roi_preparation_failed",
            outside_roi=outside,
            roi=empty_roi,
            correctness=correctness.model_dump(mode="json"),
            error=_capture_error("roi_preparation_failed", exc),
        )
    outside["all_completed_before_profiler_start"] = True
    threads_before = threading.active_count()

    def candidate_once():
        return user_fn(*inputs, *roi_outputs)

    try:
        roi = profile_candidate_once(
            cuda=torch.cuda,
            candidate=candidate_once,
            start_event=start_event,
            end_event=end_event,
        ).to_dict()
        _check_integrity(_integrity_snapshot, globals())
        check_thread_injection(threads_before, threading.active_count())
    except ProfiledCandidateError as exc:
        return _publish_capture_result(
            status="profiled_candidate_failed",
            outside_roi=outside,
            roi=exc.outcome.to_dict(),
            correctness=correctness.model_dump(mode="json"),
            error=_capture_error("profiled_candidate_failed", exc),
        )
    except RewardHackDetected as exc:
        return _publish_capture_result(
            status="reward_hack_detected",
            outside_roi=outside,
            roi=roi,
            correctness=correctness.model_dump(mode="json"),
            error=_capture_error("reward_hack_detected", exc),
        )
    except Exception as exc:
        return _publish_capture_result(
            status="capture_internal_error",
            outside_roi=outside,
            roi=locals().get("roi", empty_roi),
            correctness=correctness.model_dump(mode="json"),
            error=_capture_error("capture_internal_error", exc),
        )

    return _publish_capture_result(
        status="captured",
        outside_roi=outside,
        roi=roi,
        correctness=correctness.model_dump(mode="json"),
    )


if _CAPTURE_CONTRACT is not None:
    sys.exit(_run_capture_mode())


# ── Evaluate each workload ────────────────────────────────────────────────────
_device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""

# Check whether clocks are locked (env var set by an upstream evaluator
# server's startup script when applicable; never exported from this
# local harness — CLAUDE.md Rule 2).
_clocks_locked = are_clocks_locked()

# Only include clock status in trace logs when the user explicitly requested
# clock locking (lock_clocks=True).  When lock_clocks=False (default), the
# status is noise and misleads users into thinking failures are clock-related.
_clock_status_msg: Optional[str] = None
if bench_config.lock_clocks:
    _clock_status_msg = "Clocks locked: yes" if _clocks_locked else "Clocks locked: no"


def _make_eval(
    status, device, log_path, *, correctness=None, performance=None, extra_msg=None
):
    parts = [p for p in (_clock_status_msg, extra_msg) if p]
    return make_eval(
        status,
        device,
        log_path,
        correctness=correctness,
        performance=performance,
        extra_msg="\n".join(parts) or None,
    )


# ── Integrity check after user code import ───────────────────────────────────
# Catches patches applied at import time (e.g. via sys.modules['__main__']).
# Runs once before any workload processing so patched correctness or timing
# functions are never called.
try:
    _check_integrity(_integrity_snapshot, globals())
except RewardHackDetected as _integrity_err:
    for _wl in workloads:
        _emit(
            Trace(
                definition=definition.name,
                solution=_solution_name,
                workload=_wl,
                evaluation=_make_eval(
                    EvaluationStatus.REWARD_HACK,
                    _device,
                    None,
                    extra_msg=str(_integrity_err),
                ),
            )
        )
    sys.exit(0)

# ── Validate clock lock requirement ───────────────────────────────────────────
# If lock_clocks=True but clocks are not locked, reject all workloads.
if bench_config.lock_clocks and not _clocks_locked:
    _reject_msg = "lock_clocks=True but GPU clocks are not locked on this server"
    for _wl in workloads:
        _emit(
            Trace(
                definition=definition.name,
                solution=_solution_name,
                workload=_wl,
                evaluation=_make_eval(
                    EvaluationStatus.RUNTIME_ERROR,
                    _device,
                    None,
                    extra_msg=_reject_msg,
                ),
            )
        )
    sys.exit(0)

set_seed(bench_config.seed)
_inputs = None
_ref_outputs = None
_user_outputs = None
_timing_outputs = None
for _workload in workloads:
    # Free GPU memory held by previous workload's tensors.
    # PyTorch's CUDA caching allocator retains freed blocks.
    # Without this cleanup, memory accumulates across workloads causing OOM.
    _inputs = None
    _ref_outputs = None
    _user_outputs = None
    _timing_outputs = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _resolved_axes = definition.get_resolved_axes_values(_workload.axes)

    # -- Safetensors inputs --
    _safe_tensors: dict = {}
    if any(v.type == "safetensors" for v in _workload.inputs.values()):
        try:
            _safe_tensors = load_safetensors(definition, _workload, _safetensors_roots)
        except Exception as _e:
            _emit(
                Trace(
                    definition=definition.name,
                    solution=_solution_name,
                    workload=_workload,
                    evaluation=_make_eval(
                        EvaluationStatus.RUNTIME_ERROR,
                        _device,
                        None,
                        extra_msg=f"Failed to load safetensors: {_e}",
                    ),
                )
            )
            continue

    # -- Correctness checking (10 rounds with fresh inputs) --
    # Running multiple rounds catches non-deterministic bugs and input-dependent
    # correctness issues.  Structural checks (lazy outputs, shape/dtype) and
    # defense snapshots run on round 0 only; numerical checks run every round.
    _custom_inputs_fn = ref_namespace.get(definition.custom_inputs_entrypoint)
    _correctness_failed = False
    _threads_before = None
    _correctness = Correctness()

    for _round in range(10):
        # Python evaluates an assignment's right-hand side before replacing its
        # target.  Drop the previous round's input and output containers first
        # so gen_inputs() does not allocate a second immutable input set while
        # the first remains live.  The caching allocator can immediately reuse
        # the released blocks, so a per-round gc/empty_cache cycle is neither
        # necessary nor desirable.  Round 9 is intentionally left live after
        # this loop because its fresh inputs are the inputs used for timing.
        if _round > 0:
            _inputs = None
            _ref_outputs = None
            _user_outputs = None

        # -- Generate inputs --
        try:
            _inputs = gen_inputs(
                definition,
                _workload,
                device=_device,
                safe_tensors=_safe_tensors or None,
                custom_inputs_fn=_custom_inputs_fn,
            )
        except Exception as _e:
            _emit(
                Trace(
                    definition=definition.name,
                    solution=_solution_name,
                    workload=_workload,
                    evaluation=_make_eval(
                        EvaluationStatus.RUNTIME_ERROR,
                        _device,
                        None,
                        extra_msg=f"gen_inputs failed: {_e}",
                    ),
                )
            )
            _correctness_failed = True
            break

        # -- Run reference (always return-value style) to get ground-truth --
        try:
            _ref_outputs = _call_and_collect_outputs(
                ref_fn,
                _inputs,
                False,
                definition,
                _resolved_axes,
                _device,
                _output_names,
                _output_dtypes_torch,
            )
        except Exception as _e:
            _emit(
                Trace(
                    definition=definition.name,
                    solution=_solution_name,
                    workload=_workload,
                    evaluation=_make_eval(
                        EvaluationStatus.INVALID_REFERENCE,
                        _device,
                        None,
                        extra_msg=f"Reference run() failed: {_e}\n{traceback.format_exc()}",
                    ),
                )
            )
            _correctness_failed = True
            break

        # -- Run user function (DPS=True or False per solution spec) --
        # Round 0 also serves as a warmup call: JIT compilers (torch.compile,
        # Triton, CuTe DSL) spawn infrastructure threads on first call.
        try:
            _user_outputs = _call_and_collect_outputs(
                user_fn,
                _inputs,
                _dps,
                definition,
                _resolved_axes,
                _device,
                _output_names,
                _output_dtypes_torch,
            )
        except Exception as _e:
            _emit(
                Trace(
                    definition=definition.name,
                    solution=_solution_name,
                    workload=_workload,
                    evaluation=_make_eval(
                        EvaluationStatus.RUNTIME_ERROR,
                        _device,
                        None,
                        extra_msg=f"User function failed: {_e}\n{traceback.format_exc()}",
                    ),
                )
            )
            _correctness_failed = True
            break

        # -- Round 0 only: integrity, thread snapshot, lazy, shape/dtype --
        if _round == 0:
            if _reward_hack_check(
                _workload, _check_integrity, _integrity_snapshot, globals()
            ):
                _correctness_failed = True
                break

            # Capture stable post-JIT thread count; the check after
            # time_runnable will catch threads injected during timing.
            _threads_before = threading.active_count()

            if _reward_hack_check(_workload, check_lazy_outputs, _user_outputs):
                _correctness_failed = True
                break

            _shape_dtype_issue: Optional[EvaluationStatus] = None
            for _ref_out, _usr_out in zip(_ref_outputs, _user_outputs):
                if _ref_out.shape != _usr_out.shape:
                    _shape_dtype_issue = EvaluationStatus.INCORRECT_SHAPE
                    break
                if _ref_out.dtype != _usr_out.dtype:
                    _shape_dtype_issue = EvaluationStatus.INCORRECT_DTYPE
                    break

            if _shape_dtype_issue is not None:
                _emit(
                    Trace(
                        definition=definition.name,
                        solution=_solution_name,
                        workload=_workload,
                        evaluation=_make_eval(_shape_dtype_issue, _device, None),
                    )
                )
                _correctness_failed = True
                break

        # -- Numerical correctness check (every round) --
        # Release each (ref_out, user_out) pair as soon as the chunked
        # correctness computation completes for it. Without this the
        # eval driver holds every output pair (potentially many GiB of
        # bf16/fp16 tensors) materialised simultaneously while iterating,
        # which OOMs on workloads whose total output footprint exceeds
        # device memory (e.g. problem_005's [I, H]=[53248, 16384] bf16
        # weight gradient tensors). The per-pair `del` is enough — we
        # don't call empty_cache here because the freed blocks stay
        # available to the same allocator for the next pair's
        # compute_error_stats intermediates, and empty_cache forces a
        # device sync that adds significant overhead per workload.
        _numerically_wrong = False
        _n_outputs = len(_ref_outputs)
        for _i in range(_n_outputs):
            _ref_out = _ref_outputs[_i]
            _usr_out = _user_outputs[_i]
            _c, _exceeds = compute_error_stats(_usr_out, _ref_out, _workload.tolerance)
            if _c.max_absolute_error > _correctness.max_absolute_error:
                _correctness = _c
            if _c.has_nan:
                _correctness = _c
            elif _c.has_inf and not _correctness.has_nan:
                _correctness = _c
            _ref_outputs[_i] = None
            _user_outputs[_i] = None
            del _ref_out, _usr_out
            if _exceeds:
                _numerically_wrong = True
                break

        if _numerically_wrong:
            _emit(
                Trace(
                    definition=definition.name,
                    solution=_solution_name,
                    workload=_workload,
                    evaluation=_make_eval(
                        EvaluationStatus.INCORRECT_NUMERICAL,
                        _device,
                        None,
                        correctness=_correctness,
                    ),
                )
            )
            _correctness_failed = True
            break

    if _correctness_failed:
        continue

    # Release the correctness-round tensors before timing starts so the
    # caching allocator has a fresh slate. Without this the timing
    # phase's first allocation often OOMs against a fragmented pool
    # left behind by 10 rounds of ref + user output materialisation —
    # even when total live memory is well under device capacity.
    _ref_outputs = None
    _user_outputs = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- Monkey-patch defense before timing --
    if _reward_hack_check(_workload, check_monkey_patch):
        continue

    # -- User latency measurement --
    try:
        _timing_outputs = (
            allocate_outputs(definition, _resolved_axes, _device) if _dps else []
        )
        _sol_latency_ms = time_runnable(
            user_fn,
            _inputs,
            _timing_outputs,
            _device,
            warmup=bench_config.warmup_runs,
            rep=bench_config.iterations,
        )
    except Exception as _e:
        _emit(
            Trace(
                definition=definition.name,
                solution=_solution_name,
                workload=_workload,
                evaluation=_make_eval(
                    EvaluationStatus.RUNTIME_ERROR,
                    _device,
                    None,
                    extra_msg=f"Timing failed: {_e}",
                ),
            )
        )
        continue

    # -- Thread injection defense check --
    if _reward_hack_check(
        _workload, check_thread_injection, _threads_before, threading.active_count()
    ):
        continue

    # -- Reference latency (for speedup factor) —always return-value style --
    # Inputs are cloned (not regenerated) since the reference cannot cheat.
    _ref_latency_ms = 0.0
    if bench_config.benchmark_reference:
        try:
            _ref_latency_ms = time_runnable(
                ref_fn,
                _inputs,
                [],
                _device,
                warmup=bench_config.warmup_runs,
                rep=bench_config.iterations,
            )
        except Exception:
            pass

    _speedup = _ref_latency_ms / _sol_latency_ms if _sol_latency_ms > 0 else 0.0

    # -- Emit PASSED trace --
    _emit(
        Trace(
            definition=definition.name,
            solution=_solution_name,
            workload=_workload,
            evaluation=_make_eval(
                EvaluationStatus.PASSED,
                _device,
                None,
                correctness=_correctness,
                performance=Performance(
                    latency_ms=_sol_latency_ms,
                    reference_latency_ms=_ref_latency_ms,
                    speedup_factor=_speedup,
                ),
            ),
        )
    )
