# shellcheck shell=bash
# Verify/export the pinned NVBit protobuf build environment.
#
# Sourcing is strictly read-only: it verifies the exact explicit lock, an
# already-provisioned protobuf prefix, CUDA 13.2, g++-14, protoc 34.1, and
# libprotobuf 34.1.0 before exporting anything. The only provisioning mode is
# executing this file with --ensure (canonical prefix) or --ensure-external
# (an explicit external NVBIT_PB_PREFIX); neither mode provisions over an
# existing path and the script never repairs or replaces a present prefix.
#
# H3 forbids canonical-worktree tracer/metric build, clean, replacement, and
# repin. These exports are also used by the separately guarded non-Git
# temporary-copy verification workflow.

_NVBIT_PB_LOCK_SHA256="d522b3ee1560e98512cd460b213adee5a901f77aef0e4fae3ba94699e3b1e222"
_NVBIT_PB_VERSION="34.1.0"
_NVBIT_PROTOC_VERSION="34.1"

_nvbit_fail() {
    local code="$1"
    shift
    printf '[pinned_build_env] %s\n' "$*" >&2
    return "$code"
}

_nvbit_prepend_value() {
    local required_entry="$1"
    local current_value="$2"
    local rebuilt="$required_entry"
    local entry
    local old_ifs="$IFS"
    IFS=:
    for entry in $current_value; do
        if [[ -n "$entry" && "$entry" != "$required_entry" ]]; then
            rebuilt="$rebuilt:$entry"
        fi
    done
    IFS="$old_ifs"
    printf '%s' "$rebuilt"
}

_nvbit_normalize_external_prefix() {
    local prefix="$1"
    local python_bin
    python_bin="$(command -v python3 || true)"
    if [[ -z "$python_bin" ]]; then
        _nvbit_fail 2 "python3 is required to normalize the external protobuf handoff"
        return $?
    fi
    "$python_bin" - "$prefix" <<'PY'
import os
import stat
import sys
from pathlib import Path

lexical_root = Path(sys.argv[1])
root_metadata = lexical_root.lstat()
if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
    raise SystemExit("external protobuf handoff root is not a real directory")
root = lexical_root.resolve(strict=True)
sequence = 0
for directory, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
    base = Path(directory)
    for name in tuple(dirnames):
        member = base / name
        if member.is_symlink():
            raise SystemExit(f"external protobuf handoff contains a directory symlink: {member}")
    for name in filenames:
        member = base / name
        metadata = member.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            target = member.resolve(strict=True)
            try:
                target.relative_to(root)
            except ValueError:
                raise SystemExit(
                    f"external protobuf handoff symlink escapes prefix: {member} -> {target}"
                )
            target_metadata = target.stat()
            if not stat.S_ISREG(target_metadata.st_mode):
                raise SystemExit(f"external protobuf handoff symlink target is not regular: {member}")
            payload = target.read_bytes()
            mode = stat.S_IMODE(target_metadata.st_mode)
        elif stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink == 1:
                continue
            payload = member.read_bytes()
            mode = stat.S_IMODE(metadata.st_mode)
        else:
            raise SystemExit(f"external protobuf handoff contains a special file: {member}")
        sequence += 1
        temporary = base / f".nvbit-handoff-{os.getpid()}-{sequence}"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError(f"short external protobuf handoff write: {member}")
                view = view[written:]
            os.fchmod(descriptor, mode)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, member)
        parent = os.open(base, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent)
        finally:
            os.close(parent)

for directory, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
    base = Path(directory)
    for name in (*dirnames, *filenames):
        member = base / name
        metadata = member.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise SystemExit(f"external protobuf handoff retained a symlink: {member}")
        if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink != 1:
            raise SystemExit(f"external protobuf handoff retained a hardlink: {member}")
PY
}

_nvbit_validate_external_destination() {
    local prefix="$1"
    local suite_root="$2"
    local python_bin
    python_bin="$(command -v python3 || true)"
    if [[ -z "$python_bin" ]]; then
        _nvbit_fail 2 "python3 is required to validate the external protobuf handoff"
        return $?
    fi
    "$python_bin" - "$prefix" "$suite_root" <<'PY'
import os
import stat
import sys
from pathlib import Path

candidate = Path(sys.argv[1])
repository = Path(sys.argv[2]).resolve(strict=True)
if not candidate.is_absolute() or any(part in {".", ".."} for part in candidate.parts):
    raise SystemExit("external protobuf handoff requires an absolute path without dot components")
if ".git" in candidate.parts:
    raise SystemExit("external protobuf handoff must not name Git metadata")
parent = candidate.parent
cursor = Path(parent.anchor)
for component in parent.parts[1:]:
    cursor /= component
    metadata = os.lstat(cursor)
    if stat.S_ISLNK(metadata.st_mode):
        raise SystemExit(f"external protobuf handoff parent contains a symlink: {cursor}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise SystemExit(f"external protobuf handoff parent is not a directory: {cursor}")
try:
    os.lstat(candidate)
except FileNotFoundError:
    pass
else:
    raise SystemExit(f"external protobuf handoff destination already exists: {candidate}")
resolved = parent.resolve(strict=True) / candidate.name
try:
    resolved.relative_to(repository)
except ValueError:
    pass
else:
    raise SystemExit("external protobuf handoff destination resolves inside the repository")
PY
}

_nvbit_main() {
    local executed=0
    local ensure=0
    local ensure_external=0
    if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
        executed=1
        if [[ "$#" -eq 1 && "$1" == "--ensure" ]]; then
            ensure=1
        elif [[ "$#" -eq 1 && "$1" == "--ensure-external" ]]; then
            ensure=1
            ensure_external=1
        elif [[ "$#" -ne 0 ]]; then
            _nvbit_fail 2 "usage: pinned_build_env.sh [--ensure|--ensure-external]"
            return $?
        fi
    fi

    local suite_root lock_path lock_actual default_prefix pb_prefix
    suite_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || {
        _nvbit_fail 2 "cannot resolve repository root"
        return $?
    }
    lock_path="$suite_root/tools/nvbit_tracer/toolchain/pb_explicit.lock"
    default_prefix="$suite_root/.toolchain/pb"
    pb_prefix="${NVBIT_PB_PREFIX:-$default_prefix}"

    if [[ "$pb_prefix" != /* ]]; then
        _nvbit_fail 1 "NVBIT_PB_PREFIX must be absolute when set: $pb_prefix"
        return $?
    fi
    if [[ "$ensure_external" -eq 1 ]]; then
        if [[ -z "${NVBIT_PB_PREFIX:-}" || "$pb_prefix" == "$default_prefix" ]]; then
            _nvbit_fail 2 "--ensure-external requires an explicit external NVBIT_PB_PREFIX"
            return $?
        fi
        if [[ "$pb_prefix" == "$suite_root" || "$pb_prefix" == "$suite_root/"* ]]; then
            _nvbit_fail 2 "--ensure-external destination must be outside the repository"
            return $?
        fi
        if [[ ! "$pb_prefix" =~ ^/[A-Za-z0-9._+/-]+$ ]] || [[ "$pb_prefix" == *"/.git/"* ]]; then
            _nvbit_fail 2 "--ensure-external destination is not a Make-safe non-Git path: $pb_prefix"
            return $?
        fi
        if [[ ! -d "$(dirname "$pb_prefix")" || -L "$(dirname "$pb_prefix")" ]]; then
            _nvbit_fail 2 "--ensure-external destination parent must be a real existing directory"
            return $?
        fi
        _nvbit_validate_external_destination "$pb_prefix" "$suite_root" || {
            _nvbit_fail 2 "--ensure-external destination failed exact path validation"
            return $?
        }
    fi
    if [[ ! -f "$lock_path" || -L "$lock_path" ]]; then
        _nvbit_fail 2 "exact protobuf lock is missing or not a regular file: $lock_path"
        return $?
    fi
    lock_actual="$(sha256sum "$lock_path" 2>/dev/null | awk '{print $1}')" || {
        _nvbit_fail 2 "cannot hash protobuf lock: $lock_path"
        return $?
    }
    if [[ "$lock_actual" != "$_NVBIT_PB_LOCK_SHA256" ]] \
        || [[ "$(wc -l <"$lock_path")" -ne 8 ]] \
        || [[ "$(head -n 1 "$lock_path")" != "@EXPLICIT" ]]; then
        _nvbit_fail 1 "protobuf lock mismatch: expected $_NVBIT_PB_LOCK_SHA256 got ${lock_actual:-unreadable}"
        return $?
    fi

    if [[ ! -e "$pb_prefix" ]]; then
        if [[ "$ensure" -ne 1 ]]; then
            _nvbit_fail 2 "pinned protobuf prefix is missing: $pb_prefix; execute pinned_build_env.sh --ensure"
            return $?
        fi
        if [[ "$ensure_external" -eq 0 && "$pb_prefix" != "$default_prefix" ]]; then
            _nvbit_fail 2 "--ensure may create only the canonical .toolchain/pb prefix; use --ensure-external for an external handoff"
            return $?
        fi
        local installer
        installer="$(command -v micromamba || command -v mamba || command -v conda || true)"
        if [[ -z "$installer" ]]; then
            _nvbit_fail 2 "micromamba, mamba, or conda is required for --ensure"
            return $?
        fi
        "$installer" create -y --copy -p "$pb_prefix" --file "$lock_path" || {
            _nvbit_fail 2 "exact protobuf-prefix installation failed"
            return $?
        }
        if [[ "$ensure_external" -eq 1 ]]; then
            _nvbit_normalize_external_prefix "$pb_prefix" || {
                _nvbit_fail 2 "external protobuf handoff normalization failed"
                return $?
            }
        fi
    fi

    if [[ ! -d "$pb_prefix" || -L "$pb_prefix" ]]; then
        _nvbit_fail 1 "present protobuf prefix is not a regular directory: $pb_prefix"
        return $?
    fi

    local cuda_root host_cxx protoc pkg_config
    cuda_root="${NVBIT_TRACER_CUDA_ROOT:-/usr/local/cuda-13.2}"
    host_cxx="${NVBIT_TRACER_HOST_CXX:-/usr/bin/g++-14}"
    protoc="$pb_prefix/bin/protoc"
    pkg_config="$(command -v pkg-config || true)"
    if [[ ! -x "$cuda_root/bin/nvcc" ]]; then
        _nvbit_fail 2 "missing CUDA prerequisite: $cuda_root/bin/nvcc"
        return $?
    fi
    if [[ ! -x "$host_cxx" ]]; then
        _nvbit_fail 2 "missing host compiler prerequisite: $host_cxx"
        return $?
    fi
    if [[ ! -x "$protoc" ]]; then
        _nvbit_fail 1 "present protobuf prefix lacks executable protoc: $protoc"
        return $?
    fi
    if [[ -z "$pkg_config" ]]; then
        _nvbit_fail 2 "pkg-config prerequisite is missing"
        return $?
    fi

    local probe_path probe_pkg probe_ld
    probe_path="$(_nvbit_prepend_value "$pb_prefix/bin" "${PATH:-}")"
    probe_path="$(_nvbit_prepend_value "$cuda_root/bin" "$probe_path")"
    probe_pkg="$(_nvbit_prepend_value "$pb_prefix/lib/pkgconfig" "${PKG_CONFIG_PATH:-}")"
    probe_ld="$(_nvbit_prepend_value "$pb_prefix/lib" "${LD_LIBRARY_PATH:-}")"

    local nvcc_release protoc_release host_cxx_major protobuf_prefix protobuf_version
    nvcc_release="$("$cuda_root/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -n 1)"
    protoc_release="$("$protoc" --version 2>/dev/null | awk '{print $2}')"
    host_cxx_major="$("$host_cxx" -dumpversion 2>/dev/null | cut -d. -f1)"
    protobuf_prefix="$(PATH="$probe_path" PKG_CONFIG_PATH="$probe_pkg" "$pkg_config" --define-prefix --variable=prefix protobuf 2>/dev/null)"
    protobuf_version="$(PATH="$probe_path" PKG_CONFIG_PATH="$probe_pkg" "$pkg_config" --define-prefix --modversion protobuf 2>/dev/null)"

    if [[ "$nvcc_release" != "13.2" ]]; then
        _nvbit_fail 1 "expected nvcc release 13.2, got ${nvcc_release:-unknown}"
        return $?
    fi
    if [[ "$protoc_release" != "$_NVBIT_PROTOC_VERSION" ]]; then
        _nvbit_fail 1 "expected libprotoc $_NVBIT_PROTOC_VERSION, got ${protoc_release:-unknown}"
        return $?
    fi
    if [[ "$host_cxx_major" != "14" ]]; then
        _nvbit_fail 1 "expected g++ major 14, got ${host_cxx_major:-unknown}"
        return $?
    fi
    if [[ "$protobuf_prefix" != "$pb_prefix" || "$protobuf_version" != "$_NVBIT_PB_VERSION" ]]; then
        _nvbit_fail 1 "expected protobuf $_NVBIT_PB_VERSION under $pb_prefix; got version=${protobuf_version:-unknown} prefix=${protobuf_prefix:-unknown}"
        return $?
    fi

    export SUITE_ROOT="$suite_root"
    export PB_PREFIX="$pb_prefix"
    export NVBIT_TRACER_CUDA_ROOT="$cuda_root"
    export NVBIT_TRACER_HOST_CXX="$host_cxx"
    export NVBIT_TRACER_PROTOC="$protoc"
    export PATH="$probe_path"
    export PKG_CONFIG_PATH="$probe_pkg"
    export LD_LIBRARY_PATH="$probe_ld"
    export CUDA_INSTALL_PATH="$cuda_root"
    export NVCC_CCBIN="$host_cxx"
    export CXX="$host_cxx"
    export PROTOC="$protoc"
    export PKG_CONFIG_FLAGS="--define-prefix"

    printf '[pinned_build_env] nvcc=%s ccbin=%s protoc=%s protobuf=%s PB_PREFIX=%s\n' \
        "$nvcc_release" "$host_cxx" "$protoc_release" "$protobuf_version" "$pb_prefix"
    [[ "$executed" -eq 0 || "$ensure" -eq 0 || -d "$pb_prefix" ]]
}

_nvbit_main "$@"
_nvbit_status=$?
unset -f _nvbit_fail _nvbit_prepend_value _nvbit_normalize_external_prefix _nvbit_validate_external_destination _nvbit_main
unset _NVBIT_PB_LOCK_SHA256 _NVBIT_PB_VERSION _NVBIT_PROTOC_VERSION
return "$_nvbit_status" 2>/dev/null || exit "$_nvbit_status"
