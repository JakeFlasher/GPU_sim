#!/bin/bash -p
# Regenerate the real golden traces into guarded active staging.  Historical
# campaign evidence under data/local_labels/campaign is read-only.  Non-plan
# NCU collection requires an independently retained v3 campaign preflight:
#   capture_golden_cells.sh --preflight /abs/preflight.json \
#     --preflight-sha256 <sha256> [cells...]
set -euo pipefail

for name in BASH_ENV ENV LD_AUDIT LD_LIBRARY_PATH LD_PRELOAD \
            PYTHONHOME PYTHONPATH CUDA_INJECTION32_PATH CUDA_INJECTION64_PATH \
            NVBIT_TOOL TRACES_FOLDER USER_DEFINED_FOLDERS \
            CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER NVIDIA_VISIBLE_DEVICES \
            GPU_DEVICE_ORDINAL NV_GPU; do
  if [[ -v "$name" ]]; then
    printf 'capture_golden_cells: refusing ambient startup variable: %s\n' "$name" >&2
    exit 2
  fi
done
export PYTHONNOUSERSITE=1

SCRIPT_SOURCE="${BASH_SOURCE[0]}"
if [[ "$SCRIPT_SOURCE" != */* ]]; then
  printf 'capture_golden_cells: collector must be invoked through an explicit path\n' >&2
  exit 2
fi
REPO="$(cd -- "${SCRIPT_SOURCE%/*}/../../.." && pwd -P)"
SO="$REPO/tools/nvbit_tracer/tracer_tool/tracer_tool.so"
NVCC=/opt/cuda/bin/nvcc
CCBIN=g++-14
CCBIN_SHA256=
NVCC_SHA256=
TRACER_SHA256=
PYTHON="$REPO/.venv/bin/python"
OUTPUT_ROOT="$REPO/results/nvbit_capture_staging/golden_cells"
PLAN_ONLY=0
if [[ "${1:-}" == "--plan-only" ]]; then
  PLAN_ONLY=1
  shift
fi
PREFLIGHT=""
PREFLIGHT_SHA256=""
ORIGINAL_ARGS=("$@")
PARSED_CELLS=()
while (( $# )); do
  case "$1" in
    --preflight)
      (( $# >= 2 )) || { echo "capture_golden_cells: --preflight requires a path" >&2; exit 2; }
      PREFLIGHT="$2"; shift 2 ;;
    --preflight-sha256)
      (( $# >= 2 )) || { echo "capture_golden_cells: --preflight-sha256 requires a value" >&2; exit 2; }
      PREFLIGHT_SHA256="$2"; shift 2 ;;
    --)
      echo "capture_golden_cells: unexpected --" >&2; exit 2 ;;
    *)
      PARSED_CELLS+=("$1"); shift ;;
  esac
done

if [[ ! -x "$PYTHON" ]]; then
  echo "capture_golden_cells: Python environment missing: $PYTHON" >&2
  exit 2
fi

clear_path() {
  "$PYTHON" - "$REPO" "$1" <<'PY'
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve(strict=True)
candidate = Path(sys.argv[2])
sys.path.insert(0, str(repo))
from tools.local_labels.protected_paths import assert_write_allowed

cursor = candidate
while True:
    try:
        if cursor.is_symlink():
            raise SystemExit(
                f"capture_golden_cells: refusing symlinked output component: {cursor}"
            )
    except OSError as exc:
        raise SystemExit(
            f"capture_golden_cells: cannot inspect output component {cursor}: {exc}"
        ) from exc
    if cursor == cursor.parent:
        break
    cursor = cursor.parent

resolved = assert_write_allowed(
    candidate, repo_root=repo, operation="capture golden-cell output",
)
staging = (repo / "results" / "nvbit_capture_staging").resolve(strict=False)
try:
    resolved.relative_to(staging)
except ValueError as exc:
    raise SystemExit(
        f"capture_golden_cells: refusing output outside {staging}: {resolved}"
    ) from exc
print(resolved)
PY
}

reject_lexical_symlink_components() {
  "$PYTHON" - "$1" <<'PY'
import sys
from pathlib import Path

cursor = Path(sys.argv[1])
while True:
    if cursor.is_symlink():
        raise SystemExit(
            f"capture_golden_cells: refusing symlinked output component: {cursor}"
        )
    if cursor == cursor.parent:
        break
    cursor = cursor.parent
PY
}

publish_target_check() {
  "$PYTHON" - "$1" "$2" <<'PY'
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
kind = sys.argv[2]
try:
    metadata = path.lstat()
except FileNotFoundError:
    raise SystemExit(0)
except OSError as exc:
    raise SystemExit(f"capture_golden_cells: cannot inspect publication target {path}: {exc}")
if stat.S_ISLNK(metadata.st_mode):
    raise SystemExit(f"capture_golden_cells: publication target is a symlink: {path}")
if kind == "binary" and stat.S_ISREG(metadata.st_mode) and metadata.st_nlink > 1:
    raise SystemExit(
        f"capture_golden_cells: refusing hardlinked binary destination: {path}"
    )
if kind == "trace":
    raise SystemExit(f"capture_golden_cells: trace publication target already exists: {path}")
raise SystemExit(f"capture_golden_cells: publication target already exists: {path}")
PY
}

reject_ambient_cuda_selectors() {
  local name
  for name in CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER NVIDIA_VISIBLE_DEVICES \
              GPU_DEVICE_ORDINAL NV_GPU; do
    if [[ -v "$name" ]]; then
      echo "capture_golden_cells: refusing ambient CUDA selector: $name" >&2
      exit 2
    fi
  done
}

resolve_admitted_toolchain() {
  "$PYTHON" - "$PREFLIGHT" "$PREFLIGHT_SHA256" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_sha = sys.argv[2]
try:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError("preflight SHA-256 changed while resolving toolchain")
    manifest = json.loads(payload)
    nvcc = manifest["checks"]["nvcc"]["executable"]
    compiler = manifest["checks"]["sol_environment"]["compiler_identity"]
    ccbin = compiler["resolved_realpath"]
    ccbin_sha = compiler["executable_sha256"]
    for label, value in (("nvcc", nvcc), ("compiler", ccbin)):
        candidate = Path(value)
        if (
            not candidate.is_absolute()
            or candidate != Path(os.path.normpath(value))
            or candidate.is_symlink()
            or not candidate.is_file()
            or not os.access(candidate, os.X_OK)
        ):
            raise ValueError(f"admitted {label} executable is not retained: {value}")
    if (
        type(ccbin_sha) is not str
        or len(ccbin_sha) != 64
        or any(character not in "0123456789abcdef" for character in ccbin_sha)
    ):
        raise ValueError("admitted compiler hash is not lowercase 64-hex")
    nvcc_sha = hashlib.sha256(Path(nvcc).read_bytes()).hexdigest()
    tracer_sha = manifest["checks"]["injection_runtime"]["tracer_so_sha256"]
    if (
        type(tracer_sha) is not str
        or len(tracer_sha) != 64
        or any(character not in "0123456789abcdef" for character in tracer_sha)
    ):
        raise ValueError("admitted tracer hash is not lowercase 64-hex")
    print(nvcc)
    print(ccbin)
    print(ccbin_sha)
    print(nvcc_sha)
    print(tracer_sha)
except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"capture_golden_cells: admitted toolchain unavailable: {exc}") from exc
PY
}

verify_admitted_nvcc() {
  "$PYTHON" - "$REPO" "$PREFLIGHT" "$PREFLIGHT_SHA256" "$NVCC" "$NVCC_SHA256" <<'PY'
import hashlib
import sys
from pathlib import Path

repo, preflight, expected_sha, nvcc, expected_nvcc_sha = sys.argv[1:]
sys.path.insert(0, repo)
from tools.local_labels import capture_toolchain_contract
from tools.local_labels import immutable_artifact as immutable
from tools.local_labels import ncu_action_executor
try:
    payload = immutable.read_regular(Path(preflight))
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError("preflight SHA-256 changed while checking nvcc")
    manifest = immutable.strict_json_loads(payload, label="preflight manifest")
    expected = manifest["checks"]["nvcc"]["expected"]
    with immutable.open_retained_executable(Path(nvcc)) as retained_nvcc:
        if retained_nvcc.sha256 != expected_nvcc_sha:
            raise ValueError("NVCC SHA-256 changed after toolchain resolution")
        result = ncu_action_executor.execute_bounded_command(
            (f"/proc/self/fd/{retained_nvcc.descriptor}", "--version"),
            environment={
                "LANG": "C",
                "LANGUAGE": "C",
                "LC_ALL": "C",
                "PATH": "/bin:/usr/bin",
            },
            working_directory="/",
            inherited_descriptors=(retained_nvcc.descriptor,),
            timeout_seconds=30.0,
            max_output_bytes=1024 * 1024,
            timeout_code=ncu_action_executor.NCU_VERSION_PROBE_FAILED,
            launch_code=ncu_action_executor.NCU_VERSION_PROBE_FAILED,
        )
    if result.returncode != 0:
        raise ValueError(f"{nvcc} --version failed: {result.stderr.decode(errors='replace').strip()}")
    observed = capture_toolchain_contract.parse_nvcc_release_banner(
        result.stdout.decode(errors="replace")
    )
    capture_toolchain_contract.require_nvcc_release(observed, expected)
except (OSError, ValueError, KeyError, TypeError,
        immutable.ImmutableArtifactError,
        capture_toolchain_contract.CaptureToolchainContractError,
        ncu_action_executor.NcuActionExecutorError) as exc:
    raise SystemExit(f"capture_golden_cells: nvcc identity changed after preflight: {exc}") from exc
PY
}

fsync_publication_path() {
  "$PYTHON" - "$1" <<'PY'
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
directory = path if path.is_dir() else path.parent
try:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
except OSError as exc:
    raise SystemExit(
        f"capture_golden_cells: cannot fsync publication directory {directory}: {exc}"
    ) from exc
PY
}

publish_same_parent_noreplace() {
  "$PYTHON" - "$REPO" "$1" "$2" <<'PY'
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).resolve(strict=True)))
from tools.local_labels import immutable_artifact as immutable

source = Path(sys.argv[2]).absolute()
destination = Path(sys.argv[3]).absolute()
try:
    if source.parent != destination.parent:
        raise immutable.ImmutableArtifactError(
            "golden publication requires one source and destination parent"
        )
    source_metadata = source.lstat()
    if stat.S_ISLNK(source_metadata.st_mode):
        raise immutable.ImmutableArtifactError(
            f"golden publication source is a symlink: {source}"
        )
    if stat.S_ISDIR(source_metadata.st_mode):
        material = []
        for path in sorted(source.rglob("*")):
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise immutable.ImmutableArtifactError(
                    f"golden publication tree contains a symlink: {path}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise immutable.ImmutableArtifactError(
                    f"golden publication tree contains a special file: {path}"
                )
            material.append(
                immutable.ImmutableFile.from_path(
                    path.relative_to(source).as_posix(), path,
                )
            )
        if not material:
            raise immutable.ImmutableArtifactError(
                f"golden publication tree is empty: {source}"
            )
        immutable.publish_directory(
            destination.parent, destination.name, material,
        )
    elif stat.S_ISREG(source_metadata.st_mode):
        immutable.publish_regular_file(
            destination.parent, destination.name, source,
            executable=bool(source_metadata.st_mode & 0o111),
        )
    else:
        raise immutable.ImmutableArtifactError(
            f"golden publication source is not regular or directory: {source}"
        )
except (OSError, immutable.ImmutableArtifactError) as exc:
    raise SystemExit(
        f"capture_golden_cells: immutable publication failed: {exc}"
    ) from exc
PY
}

verify_published_receipt() {
  "$PYTHON" - "$REPO" "$1" "$PREFLIGHT" "$PREFLIGHT_SHA256" <<'PY'
import hashlib
import stat
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve(strict=True)
receipt_path = Path(sys.argv[2])
expected_preflight_path = Path(sys.argv[3])
expected_preflight_sha256 = sys.argv[4]
sys.path.insert(0, str(repo))
from tools.local_labels import immutable_artifact as immutable
from tools.nvbit_tracer import trace_admission
from tools.trace_sampling.golden_cells import golden_ncu_capture
from tools.trace_sampling.golden_cells import golden_ncu_receipt
try:
    receipt_metadata = receipt_path.lstat()
    receipt_resolved = receipt_path.resolve(strict=False)
    receipt_resolved.relative_to(repo)
except (OSError, ValueError) as exc:
    raise SystemExit(f"capture_golden_cells: published receipt path invalid: {receipt_path}") from exc
if (
    receipt_resolved != receipt_path
    or stat.S_ISLNK(receipt_metadata.st_mode)
    or not stat.S_ISREG(receipt_metadata.st_mode)
    or receipt_metadata.st_mode & 0o222
):
    raise SystemExit(f"capture_golden_cells: published receipt is not retained read-only evidence: {receipt_path}")
try:
    retained_receipt = immutable.open_retained_regular(receipt_path)
    if retained_receipt.snapshot[2] & 0o222:
        raise golden_ncu_receipt.GoldenReceiptError(
            "published receipt is writable during retained read"
        )
    receipt_snapshot = retained_receipt.snapshot
    receipt_sha256 = retained_receipt.sha256
    receipt_payload = retained_receipt.read_bytes()
    document = immutable.strict_json_loads(
        receipt_payload, label="published golden receipt",
    )
    immutable.verify_model_seal(document, "receipt_seal_sha256")
    golden_ncu_receipt.validate_intrinsic_receipt(document)
    preflight_path = Path(document["preflight_path"])
    _preflight_sha256, _expected_ncu, preflight_manifest = (
        golden_ncu_capture._load_preflight(
            expected_preflight_path,
            expected_sha256=expected_preflight_sha256,
            require_fresh=False,
        )
    )
    if (
        preflight_path != expected_preflight_path
        or document["preflight_sha256"] != expected_preflight_sha256
    ):
        raise golden_ncu_receipt.GoldenReceiptError(
            "published receipt preflight authority differs from invocation"
        )
    golden_ncu_receipt.validate_ncu_identity_against_preflight(
        document, preflight_manifest,
    )
    target_model = golden_ncu_receipt._target_binary(document["target_binary"])
    trace_model = golden_ncu_receipt._relative_path(
        document["trace_relative_path"], label="published golden trace",
    )
    artifact_model = golden_ncu_receipt._artifacts(document["artifacts"])
    golden_ncu_receipt._topology(target_model, trace_model, artifact_model)
    golden_ncu_receipt._controls(document["launch_controls"])
    if receipt_path.parent != repo / Path(trace_model).parent:
        raise golden_ncu_receipt.GoldenReceiptError(
            "published receipt is outside its selected golden cell"
        )
    trace_path = repo / trace_model
    trace_metadata = trace_path.lstat()
    if (
        not stat.S_ISDIR(trace_metadata.st_mode)
        or trace_metadata.st_mode & 0o222
        or trace_path.is_symlink()
        or trace_path.resolve(strict=False) != trace_path
    ):
        raise golden_ncu_receipt.GoldenReceiptError(
            "published golden trace is not a retained directory"
        )
    trace_rows = []
    trace_bytes = 0
    for path in sorted(trace_path.rglob("*")):
        if path.is_symlink():
            raise golden_ncu_receipt.GoldenReceiptError(
                f"published golden trace contains a symlink: {path}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise golden_ncu_receipt.GoldenReceiptError(
                f"published golden trace contains a special file: {path}"
            )
        with immutable.open_retained_regular(path) as retained_trace:
            trace_rows.append({
                "relative_path": path.relative_to(trace_path).as_posix(),
                "sha256": retained_trace.sha256,
                "size_bytes": retained_trace.size_bytes,
            })
            trace_bytes += retained_trace.size_bytes
    if not trace_rows:
        raise golden_ncu_receipt.GoldenReceiptError(
            "published golden trace is empty"
        )
    observed_trace_sha256 = hashlib.sha256(
        immutable.canonical_json_bytes(trace_rows)
    ).hexdigest()
    if (
        observed_trace_sha256 != document["trace_tree_sha256"]
        or trace_bytes != document["trace_tree_bytes"]
    ):
        raise golden_ncu_receipt.GoldenReceiptError(
            "published golden trace tree identity differs from receipt"
        )
    immutable.verify_published_files(
        trace_path,
        tuple(
            immutable.ImmutableFile(
                row["relative_path"],
                row["sha256"],
                row["size_bytes"],
                source_path=trace_path / row["relative_path"],
            )
            for row in trace_rows
        ),
    )
    observed_admission = trace_admission.require_trace_admission(trace_path)
    observed_admission_model = {
        "launch_count": observed_admission.launch_count,
        "threadblock_count": observed_admission.threadblock_count,
        "instruction_count": observed_admission.instruction_count,
        "expected_record_count": observed_admission.expected_record_count,
        "admitted_record_count": observed_admission.admitted_record_count,
    }
    if observed_admission_model != document["trace_admission"]:
        raise golden_ncu_receipt.GoldenReceiptError(
            "published golden trace admission differs from receipt"
        )
except (OSError, KeyError, TypeError, immutable.ImmutableArtifactError,
        golden_ncu_capture.GoldenNcuCaptureError,
        trace_admission.TraceAdmissionError,
        golden_ncu_receipt.GoldenReceiptError) as exc:
    raise SystemExit(f"capture_golden_cells: published receipt is invalid: {exc}")
target = document["target_binary"]
relative_target = target["relative_path"]
target_path = repo / relative_target
try:
    target_metadata = target_path.lstat()
    target_resolved = target_path.resolve(strict=False)
    target_resolved.relative_to(repo)
except (OSError, ValueError) as exc:
    raise SystemExit(f"capture_golden_cells: published target path invalid: {target_path}") from exc
if target_resolved != target_path or stat.S_ISLNK(target_metadata.st_mode) or not stat.S_ISREG(target_metadata.st_mode) or not (target_metadata.st_mode & 0o111):
    raise SystemExit(f"capture_golden_cells: published target binary is not retained executable: {target_path}")
try:
    with immutable.open_retained_regular(target_path) as retained_target:
        expected_target_mode = int(target["mode"], 8)
        if (
            stat.S_IMODE(retained_target.snapshot[2]) != expected_target_mode
            or retained_target.snapshot[2] & 0o222
            or not retained_target.snapshot[2] & 0o111
        ):
            raise SystemExit(
                f"capture_golden_cells: published target binary mode differs from receipt: {target_path}"
            )
        if (
            retained_target.sha256 != target["sha256"]
            or retained_target.size_bytes != target["size_bytes"]
        ):
            raise SystemExit(
                f"capture_golden_cells: published target hash/size mismatch: {target_path}"
            )
except immutable.ImmutableArtifactError as exc:
    raise SystemExit(
        f"capture_golden_cells: published target binary is not retained: {target_path}: {exc}"
    ) from exc
for artifact in document.get("artifacts", []):
    relative = artifact["relative_path"]
    path = repo / relative
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=False)
        resolved.relative_to(repo)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"capture_golden_cells: published artifact path invalid: {path}") from exc
    if resolved != path or stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SystemExit(f"capture_golden_cells: published artifact is not retained: {path}")
    try:
        with immutable.open_retained_regular(path) as retained_artifact:
            if retained_artifact.snapshot[2] & 0o222:
                raise SystemExit(
                    f"capture_golden_cells: published artifact is writable: {path}"
                )
            if (
                retained_artifact.sha256 != artifact["sha256"]
                or retained_artifact.size_bytes != artifact["size_bytes"]
            ):
                raise SystemExit(
                    f"capture_golden_cells: published artifact hash/size mismatch: {path}"
                )
    except immutable.ImmutableArtifactError as exc:
        raise SystemExit(
            f"capture_golden_cells: published artifact is not retained: {path}: {exc}"
        ) from exc
try:
    retained_receipt.verify_unchanged()
    if (
        retained_receipt.snapshot != receipt_snapshot
        or retained_receipt.sha256 != receipt_sha256
        or retained_receipt.snapshot[2] & 0o222
    ):
        raise SystemExit(
            f"capture_golden_cells: published receipt changed during verification: {receipt_path}"
        )
except immutable.ImmutableArtifactError as exc:
    raise SystemExit(
        f"capture_golden_cells: published receipt lost retention: {receipt_path}: {exc}"
    ) from exc
finally:
    retained_receipt.close()
PY
}

if (( ${#PARSED_CELLS[@]} )); then
  CELLS=("${PARSED_CELLS[@]}")
else
  CELLS=(gs_stream_copy gs_stream_reuse gs_var_stride ac4_global_atomic_histogram)
fi

HAS_NCU=0
for slug in "${CELLS[@]}"; do
  case "$slug" in
    gs_stream_copy|gs_stream_reuse|gs_var_stride) HAS_NCU=1 ;;
    ac4_global_atomic_histogram) ;;
    *) echo "unknown golden cell: $slug" >&2; exit 2 ;;
  esac
done

if (( ! PLAN_ONLY && HAS_NCU )) && [[ -z "$PREFLIGHT" || -z "$PREFLIGHT_SHA256" ]]; then
  echo "capture_golden_cells: non-plan capture requires --preflight and --preflight-sha256" >&2
  exit 2
fi

if (( ! PLAN_ONLY )); then
  if [[ -z "${GOLDEN_CAPTURE_LOCK_FD:-}" ]]; then
    exec "$PYTHON" -I -S \
      "$REPO/tools/trace_sampling/golden_cells/golden_capture_lock.py" \
      --repo-root "$REPO" --exec-script "$0" -- "${ORIGINAL_ARGS[@]}"
  fi
  "$PYTHON" -I -S \
    "$REPO/tools/trace_sampling/golden_cells/golden_capture_lock.py" \
    --repo-root "$REPO" --verify-fd "$GOLDEN_CAPTURE_LOCK_FD"
fi

# Phase 1: derive and clear the entire batch before runtime probes or mutation.
GOLDEN="$(clear_path "$OUTPUT_ROOT")"
declare -A CU BIN OUT COMPILED STAGED NCU_CSV NCU_STDERR NCU_FINAL NCU_STDERR_FINAL \
                 NCU_OBSERVATION NCU_RUN_MANIFEST NCU_CAPTURE_MANIFEST NCU_RECEIPT \
                 NCU_OBSERVATION_FINAL NCU_RUN_MANIFEST_FINAL NCU_CAPTURE_MANIFEST_FINAL NCU_RECEIPT_FINAL \
                 NEEDS_NCU BINARY_SHA256
for slug in "${CELLS[@]}"; do
  case "$slug" in
    gs_stream_copy|gs_stream_reuse|gs_var_stride)
      cu="$(dirname "$0")/${slug}.cu"; needs_ncu=1 ;;
    ac4_global_atomic_histogram)
      cu="$REPO/tools/local_labels/canary/ac4_global_atomic_histogram.cu"; needs_ncu=0 ;;
    *) echo "unknown golden cell: $slug" >&2; exit 2 ;;
  esac
  cell="$GOLDEN/$slug"
  for candidate in \
      "$cell/${slug}.bin" \
      "$cell/${slug}.bin.staged.$$" \
      "$cell/trace" \
      "$cell/trace.staged.$$" \
      "$cell/ncu_l2.csv.staged.$$" \
      "$cell/ncu_l2.stderr.staged.$$" \
      "$cell/ncu_launch_observation.json.staged.$$" \
      "$cell/ncu_run_manifest.json.staged.$$" \
      "$cell/ncu_capture_manifest.json.staged.$$" \
      "$cell/ncu_collection_receipt.json.staged.$$" \
      "$cell/ncu_l2.csv" \
      "$cell/ncu_l2.stderr" \
      "$cell/ncu_launch_observation.json" \
      "$cell/ncu_run_manifest.json" \
      "$cell/ncu_capture_manifest.json" \
      "$cell/ncu_collection_receipt.json"; do
    reject_lexical_symlink_components "$candidate"
  done
  bin="$(clear_path "$cell/${slug}.bin")"
  compiled_candidate="$(clear_path "$cell/${slug}.bin.staged.$$")"
  out="$(clear_path "$cell/trace")"
  staged_candidate="$(clear_path "$cell/trace.staged.$$")"
  ncu_csv_candidate="$(clear_path "$cell/ncu_l2.csv.staged.$$")"
  ncu_stderr_candidate="$(clear_path "$cell/ncu_l2.stderr.staged.$$")"
  ncu_observation_candidate="$(clear_path "$cell/ncu_launch_observation.json.staged.$$")"
  ncu_run_manifest_candidate="$(clear_path "$cell/ncu_run_manifest.json.staged.$$")"
  ncu_capture_manifest_candidate="$(clear_path "$cell/ncu_capture_manifest.json.staged.$$")"
  ncu_receipt_candidate="$(clear_path "$cell/ncu_collection_receipt.json.staged.$$")"
  ncu_final="$(clear_path "$cell/ncu_l2.csv")"
  ncu_stderr_final="$(clear_path "$cell/ncu_l2.stderr")"
  ncu_observation_final="$(clear_path "$cell/ncu_launch_observation.json")"
  ncu_run_manifest_final="$(clear_path "$cell/ncu_run_manifest.json")"
  ncu_capture_manifest_final="$(clear_path "$cell/ncu_capture_manifest.json")"
  ncu_receipt_final="$(clear_path "$cell/ncu_collection_receipt.json")"
  CU["$slug"]="$cu"; BIN["$slug"]="$bin"; COMPILED["$slug"]="$compiled_candidate"; OUT["$slug"]="$out"
  STAGED["$slug"]="$staged_candidate"; NCU_CSV["$slug"]="$ncu_csv_candidate"
  NCU_STDERR["$slug"]="$ncu_stderr_candidate"; NCU_FINAL["$slug"]="$ncu_final"
  NCU_STDERR_FINAL["$slug"]="$ncu_stderr_final"
  NCU_OBSERVATION["$slug"]="$ncu_observation_candidate"
  NCU_RUN_MANIFEST["$slug"]="$ncu_run_manifest_candidate"
  NCU_CAPTURE_MANIFEST["$slug"]="$ncu_capture_manifest_candidate"
  NCU_RECEIPT["$slug"]="$ncu_receipt_candidate"
  NCU_OBSERVATION_FINAL["$slug"]="$ncu_observation_final"
  NCU_RUN_MANIFEST_FINAL["$slug"]="$ncu_run_manifest_final"
  NCU_CAPTURE_MANIFEST_FINAL["$slug"]="$ncu_capture_manifest_final"
  NCU_RECEIPT_FINAL["$slug"]="$ncu_receipt_final"
  NEEDS_NCU["$slug"]="$needs_ncu"
  # Refuse every final destination before runtime probes, compilation, NVBit,
  # or NCU.  The later checks close the final symlink-substitution interval.
  publish_target_check "$bin" binary
  publish_target_check "$compiled_candidate" artifact
  publish_target_check "$out" trace
  if (( needs_ncu )); then
    publish_target_check "$ncu_final" artifact
    publish_target_check "$ncu_stderr_final" artifact
    publish_target_check "$ncu_observation_final" artifact
    publish_target_check "$ncu_run_manifest_final" artifact
    publish_target_check "$ncu_capture_manifest_final" artifact
    publish_target_check "$ncu_receipt_final" artifact
  fi
done

if (( PLAN_ONLY )); then
  printf 'capture_golden_cells: plan ok (%d cells) root=%s\n' "${#CELLS[@]}" "$GOLDEN"
  exit 0
fi

reject_ambient_cuda_selectors

# Trap-visible deletion targets remain empty until the complete plan passed.
staged=""; compiled=""; ncu_csv=""; ncu_stderr=""; ncu_observation=""
ncu_run_manifest=""; ncu_capture_manifest=""; ncu_receipt=""
cleanup() {
  if [[ -n "$compiled$staged$ncu_csv$ncu_stderr$ncu_observation$ncu_run_manifest$ncu_capture_manifest$ncu_receipt" ]]; then
    echo "capture_golden_cells: incomplete artifacts retained in staging" >&2
  fi
}
trap cleanup EXIT

if (( HAS_NCU )); then
  "$PYTHON" "$REPO/tools/trace_sampling/golden_cells/golden_ncu_capture.py" \
    --validate-only --preflight "$PREFLIGHT" \
    --preflight-sha256 "$PREFLIGHT_SHA256"
  mapfile -t ADMITTED_TOOLCHAIN < <(resolve_admitted_toolchain)
  if (( ${#ADMITTED_TOOLCHAIN[@]} != 5 )); then
    echo "capture_golden_cells: admitted toolchain output is incomplete" >&2
    exit 2
  fi
  NVCC="${ADMITTED_TOOLCHAIN[0]}"
  CCBIN="${ADMITTED_TOOLCHAIN[1]}"
  CCBIN_SHA256="${ADMITTED_TOOLCHAIN[2]}"
  NVCC_SHA256="${ADMITTED_TOOLCHAIN[3]}"
  TRACER_SHA256="${ADMITTED_TOOLCHAIN[4]}"
  verify_admitted_nvcc
else
  NVCC="$(realpath -e -- "$NVCC")"
  CCBIN="$(command -v -- "$CCBIN")"
  CCBIN="$(realpath -e -- "$CCBIN")"
  if [[ ! -x "$NVCC" || ! -x "$CCBIN" ]]; then
    echo "capture_golden_cells: local canary toolchain is unavailable" >&2
    exit 2
  fi
  NVCC_SHA256="$(sha256sum -- "$NVCC" | cut -d' ' -f1)"
  CCBIN_SHA256="$(sha256sum -- "$CCBIN" | cut -d' ' -f1)"
  TRACER_SHA256="$(sha256sum -- "$SO" | cut -d' ' -f1)"
fi

"$PYTHON" "$REPO/tools/nvbit_tracer/runtime_env.py" \
  --pb-lib "$REPO/.toolchain/pb/lib" "$SO" >/dev/null

for slug in "${CELLS[@]}"; do
  cu="${CU[$slug]}"; bin="${BIN[$slug]}"; compiled="${COMPILED[$slug]}"; out="${OUT[$slug]}"
  staged="${STAGED[$slug]}"; needs_ncu="${NEEDS_NCU[$slug]}"
  ncu_csv="${NCU_CSV[$slug]}"; ncu_stderr="${NCU_STDERR[$slug]}"
  ncu_observation="${NCU_OBSERVATION[$slug]}"
  ncu_run_manifest="${NCU_RUN_MANIFEST[$slug]}"
  ncu_capture_manifest="${NCU_CAPTURE_MANIFEST[$slug]}"
  ncu_receipt="${NCU_RECEIPT[$slug]}"
  # Re-check the compiler destination immediately before nvcc so a path
  # created after phase-one planning cannot be overwritten.
  publish_target_check "$bin" binary
  publish_target_check "$compiled" artifact
  compiler_sha256="$(sha256sum -- "$CCBIN" | cut -d' ' -f1)"
  if [[ "$compiler_sha256" != "$CCBIN_SHA256" ]]; then
    echo "capture_golden_cells: compiler identity changed before compile: $CCBIN" >&2
    exit 2
  fi
  if (( needs_ncu )); then
    verify_admitted_nvcc
  fi
  bin_parent="$(clear_path "$(dirname "$bin")")"
  mkdir -p "$bin_parent"
  "$PYTHON" "$REPO/tools/trace_sampling/golden_cells/retained_nvcc_compile.py" \
    --nvcc "$NVCC" --nvcc-sha256 "$NVCC_SHA256" \
    --compiler "$CCBIN" --compiler-sha256 "$CCBIN_SHA256" \
    --source "$cu" --output "$compiled"
  compiler_sha256="$(sha256sum -- "$CCBIN" | cut -d' ' -f1)"
  if [[ "$compiler_sha256" != "$CCBIN_SHA256" ]]; then
    echo "capture_golden_cells: compiler identity changed during compile: $CCBIN" >&2
    exit 2
  fi
  publish_same_parent_noreplace "$compiled" "$bin"
  compiled=""
  fsync_publication_path "$bin"
  binary_sha256="$(sha256sum -- "$bin" | cut -d' ' -f1)"
  [[ "$binary_sha256" =~ ^[0-9a-f]{64}$ ]] || {
    echo "capture_golden_cells: failed to hash compiled binary: $bin" >&2
    exit 2
  }
  BINARY_SHA256["$slug"]="$binary_sha256"
  staged="$(clear_path "$staged")"
  if [[ -e "$staged" || -L "$staged" ]]; then
    echo "capture_golden_cells: staged trace destination already exists: $staged" >&2
    exit 2
  fi
  "$PYTHON" "$REPO/tools/trace_sampling/golden_cells/retained_nvbit_launch.py" \
    --binary "$bin" --binary-sha256 "${BINARY_SHA256[$slug]}" \
    --tracer "$SO" --tracer-sha256 "$TRACER_SHA256" \
    --pb-lib "$REPO/.toolchain/pb/lib" --trace-dir "$staged"
  "$PYTHON" "$REPO/tools/nvbit_tracer/trace_admission.py" "$staged"
  if (( needs_ncu )); then
    ncu_csv="$(clear_path "$ncu_csv")"
    ncu_stderr="$(clear_path "$ncu_stderr")"
    ncu_observation="$(clear_path "$ncu_observation")"
    ncu_run_manifest="$(clear_path "$ncu_run_manifest")"
    ncu_capture_manifest="$(clear_path "$ncu_capture_manifest")"
    ncu_receipt="$(clear_path "$ncu_receipt")"
    "$PYTHON" "$REPO/tools/trace_sampling/golden_cells/golden_ncu_capture.py" \
      --preflight "$PREFLIGHT" --preflight-sha256 "$PREFLIGHT_SHA256" \
      --slug "$slug" --binary "$bin" --binary-sha256 "${BINARY_SHA256[$slug]}" \
        --trace-dir "$staged" \
      --csv-out "$ncu_csv" --stderr-out "$ncu_stderr" \
      --observation-out "$ncu_observation" \
      --run-manifest-out "$ncu_run_manifest" \
      --capture-manifest-out "$ncu_capture_manifest" \
      --receipt-out "$ncu_receipt" \
      --published-trace-dir "$out" \
      --published-csv-out "${NCU_FINAL[$slug]}" \
      --published-stderr-out "${NCU_STDERR_FINAL[$slug]}" \
      --published-observation-out "${NCU_OBSERVATION_FINAL[$slug]}" \
      --published-run-manifest-out "${NCU_RUN_MANIFEST_FINAL[$slug]}" \
      --published-capture-manifest-out "${NCU_CAPTURE_MANIFEST_FINAL[$slug]}" \
      --published-receipt-out "${NCU_RECEIPT_FINAL[$slug]}"
  fi
  # Re-admit immediately before publication so post-admission trace mutation
  # or deletion cannot be published under the earlier admission result.
  "$PYTHON" "$REPO/tools/nvbit_tracer/trace_admission.py" "$staged"
  publish_target_check "$out" trace
  if (( needs_ncu )); then
    publish_target_check "${NCU_FINAL[$slug]}" artifact
    publish_target_check "${NCU_STDERR_FINAL[$slug]}" artifact
    publish_target_check "${NCU_OBSERVATION_FINAL[$slug]}" artifact
    publish_target_check "${NCU_RUN_MANIFEST_FINAL[$slug]}" artifact
    publish_target_check "${NCU_CAPTURE_MANIFEST_FINAL[$slug]}" artifact
    publish_target_check "${NCU_RECEIPT_FINAL[$slug]}" artifact
  fi
  out="$(clear_path "$out")"; staged="$(clear_path "$staged")"
  publish_same_parent_noreplace "$staged" "$out"; staged=""
  fsync_publication_path "$out"
  fsync_publication_path "$(dirname "$out")"
  "$PYTHON" "$REPO/tools/nvbit_tracer/trace_admission.py" "$out"
  echo "$slug: $(find "$out/threadblocks" -name '*.pb' | wc -l) blocks"
  if (( needs_ncu )); then
    ncu_final="$(clear_path "${NCU_FINAL[$slug]}")"
    ncu_csv="$(clear_path "$ncu_csv")"
    publish_same_parent_noreplace "$ncu_csv" "$ncu_final"; ncu_csv=""
    fsync_publication_path "$ncu_final"
    ncu_stderr_final="$(clear_path "${NCU_STDERR_FINAL[$slug]}")"
    ncu_stderr="$(clear_path "$ncu_stderr")"
    publish_same_parent_noreplace "$ncu_stderr" "$ncu_stderr_final"; ncu_stderr=""
    fsync_publication_path "$ncu_stderr_final"
    ncu_observation_final="$(clear_path "${NCU_OBSERVATION_FINAL[$slug]}")"
    ncu_observation="$(clear_path "$ncu_observation")"
    publish_same_parent_noreplace "$ncu_observation" "$ncu_observation_final"; ncu_observation=""
    fsync_publication_path "$ncu_observation_final"
    ncu_run_manifest_final="$(clear_path "${NCU_RUN_MANIFEST_FINAL[$slug]}")"
    ncu_run_manifest="$(clear_path "$ncu_run_manifest")"
    publish_same_parent_noreplace "$ncu_run_manifest" "$ncu_run_manifest_final"; ncu_run_manifest=""
    fsync_publication_path "$ncu_run_manifest_final"
    ncu_capture_manifest_final="$(clear_path "${NCU_CAPTURE_MANIFEST_FINAL[$slug]}")"
    ncu_capture_manifest="$(clear_path "$ncu_capture_manifest")"
    publish_same_parent_noreplace "$ncu_capture_manifest" "$ncu_capture_manifest_final"; ncu_capture_manifest=""
    fsync_publication_path "$ncu_capture_manifest_final"
    ncu_receipt_final="$(clear_path "${NCU_RECEIPT_FINAL[$slug]}")"
    ncu_receipt="$(clear_path "$ncu_receipt")"
    publish_same_parent_noreplace "$ncu_receipt" "$ncu_receipt_final"; ncu_receipt=""
    fsync_publication_path "$ncu_receipt_final"
    verify_published_receipt "$ncu_receipt_final"
    tail -1 "$ncu_final"
  fi
done
