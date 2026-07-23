# shellcheck shell=bash
# Verify/export the pinned NVBit protobuf build environment.
#
# Sourcing is strictly read-only: it verifies the exact explicit lock, an
# already-provisioned protobuf prefix, CUDA 13.2, g++-14, protoc 34.1, and
# libprotobuf 34.1.0 before exporting anything. The only provisioning mode is
# executing this file with --ensure; it may create an absent prefix from the
# exact lock and never repairs or replaces a present prefix.
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

_nvbit_main() {
    local executed=0
    local ensure=0
    if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
        executed=1
        if [[ "$#" -eq 1 && "$1" == "--ensure" ]]; then
            ensure=1
        elif [[ "$#" -ne 0 ]]; then
            _nvbit_fail 2 "usage: pinned_build_env.sh [--ensure]"
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
        if [[ "$pb_prefix" != "$default_prefix" ]]; then
            _nvbit_fail 2 "--ensure may create only the canonical .toolchain/pb prefix"
            return $?
        fi
        local installer
        installer="$(command -v micromamba || command -v mamba || command -v conda || true)"
        if [[ -z "$installer" ]]; then
            _nvbit_fail 2 "micromamba, mamba, or conda is required for --ensure"
            return $?
        fi
        "$installer" create -y -p "$pb_prefix" --file "$lock_path" || {
            _nvbit_fail 2 "exact protobuf-prefix installation failed"
            return $?
        }
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
unset -f _nvbit_fail _nvbit_prepend_value _nvbit_main
unset _NVBIT_PB_LOCK_SHA256 _NVBIT_PB_VERSION _NVBIT_PROTOC_VERSION
return "$_nvbit_status" 2>/dev/null || exit "$_nvbit_status"
