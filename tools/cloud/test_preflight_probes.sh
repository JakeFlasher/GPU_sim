#!/usr/bin/env bash
# Shell-level regression smokes for tools/cloud/preflight.sh.
# Run with:  bash tools/cloud/test_preflight_probes.sh
#
# Smokes covered:
#   1. NVBit injection probe accepts the repo tracer's `traces/` output shape.
#   2. NVBit injection probe correctly hard-exits NvbitUnavailable when the
#      `traces/` subdir is missing.
#   3. CUPTI Activity probe reports `available` when the mock fixture says so.
#   4. CUPTI Activity probe + manifest validator record `header_missing`
#      and validate clean; the capability is non-gating under D8.
#   5. (AC1, AC1.2) Each committed `mock_inputs/cupti_smoke_<status>/`
#      fixture round-trips through preflight.sh + validator and produces the
#      expected `capabilities.cupti_activity_api_runtime` value AND the
#      expected validator outcome (`OK` for every recognized recorded
#      capability value under D8).
#   6. (AC2 Lower) `deployment_mode_cc_on/` fixture round-trips through
#      preflight.sh + validator and fires
#      `ConfidentialComputingNotSupportedForCollection`.
#   7. (AC2 Lower) `deployment_mode_mig_enabled/` fixture round-trips
#      through preflight.sh + validator and fires
#      `MigSliceNotValidForCollectionEvidence`.
#   8. (AC1.1 status precedence) Real-mode bash-shim test: when nvcc
#      compile fails AND a downstream stage would have failed too, the
#      staged probe records `compile_failed` (compile is reached first).
#
# All smokes use bash-level fake shims (fake `nvidia-smi`, fake `nvcc`) or
# the committed `tools/cloud/mock_inputs/<fixture>/` directories so
# they're fully offline-reproducible.
#
# AC1.1 status precedence (documented for reference):
#   mock_input_missing < compile_failed < link_failed < runtime_failed
#   < insufficient_privilege < cc_unsupported < available
# Enforced by the staged elif chain in preflight.sh's CUPTI runtime
# block: compile is attempted before link, link before run, run before
# the insufficient_privilege / cc_unsupported stderr-grep classification.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFLIGHT="${HERE}/preflight.sh"

FAIL_COUNT=0

log()  { printf '  %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; FAIL_COUNT=$((FAIL_COUNT+1)); }
pass() { printf 'PASS: %s\n' "$*"; }

make_nvidia_smi_shim() {
    local dir="$1"
    cat > "${dir}/nvidia-smi" <<'SHIM'
#!/usr/bin/env bash
argstr="$*"
if [[ "${argstr}" == *"--query-gpu="* ]]; then
    if [[ "${argstr}" == *"name,uuid,driver_version,compute_cap,clocks.max.sm"* ]]; then
        echo "NVIDIA GeForce RTX 5060 Laptop GPU, GPU-fake-uuid-0001, 595.58.03, 12.0, 2700, 14000, 2100, Disabled, Default"
        exit 0
    fi
    echo ""
    exit 0
fi
exit 0
SHIM
    chmod +x "${dir}/nvidia-smi"
}

make_nvcc_shim() {
    local dir="$1"
    cat > "${dir}/nvcc" <<'SHIM'
#!/usr/bin/env bash
args=("$@")
out=""
for i in "${!args[@]}"; do
    if [[ "${args[$i]}" == "-o" ]]; then
        out="${args[$((i+1))]:-}"
    fi
done
if [[ -z "${out}" ]]; then
    echo "nvcc: NVIDIA (R) Cuda compiler driver"
    echo "Cuda compilation tools, release 13.2, V13.2.89"
    exit 0
fi
cat > "${out}" <<'BINARY'
#!/usr/bin/env bash
mkdir -p "./traces/threadblocks/device_0/stream_0/kernel_0"
printf '%s' 'MigIABIkEiIIABIeEhwSCm5vcF9rZXJuZWxCBggBEAEYAUoGCAEQARgBOhJzbTg2X2FkZHJfY2FycnlfdjFCBQECAwYI' | base64 -d > "./traces/dynamic_trace.pb"
printf '%s' 'CgASFggAEhISEAiAIBABGAEgATgEQABYgCA=' | base64 -d > "./traces/threadblocks/device_0/stream_0/kernel_0/0.pb"
printf 'kernel_id,total_insts,total_reported_insts\n0,1,1\n' > "./traces/stats.csv"
exit 0
BINARY
chmod +x "${out}"
exit 0
SHIM
    chmod +x "${dir}/nvcc"
}

stage_fake_tracer_so() {
    local repo_root="$1"
    local bin_dir="$2"
    local so_dir="${repo_root}/tools/nvbit_tracer/tracer_tool"
    # The real preflight intentionally fails closed unless the tracer's pinned
    # protobuf runtime can be placed on LD_LIBRARY_PATH. Model that required
    # checkout layout in every fake-repository injection fixture.
    mkdir -p "${so_dir}" "${repo_root}/.toolchain/pb/lib"
    printf 'fake-tracer-payload' > "${so_dir}/tracer_tool.so"
    cat > "${bin_dir}/ldd" <<'LDD_SHIM'
#!/usr/bin/env bash
target="$1"
repo="${target%/tools/nvbit_tracer/tracer_tool/tracer_tool.so}"
printf 'libprotobuf.so.34.1.0 => %s/.toolchain/pb/lib/libprotobuf.so.34.1.0 (0x1)\n' "$repo"
printf 'libabsl_base.so => %s/.toolchain/pb/lib/libabsl_base.so (0x2)\n' "$repo"
LDD_SHIM
    chmod +x "${bin_dir}/ldd"
}

# ----------------------------------------------------------------------
# Smoke 1: NVBit injection probe accepts traces/ subdir output shape.
# ----------------------------------------------------------------------
smoke_nvbit_injection_accepts_traces_subdir() {
    local tmp; tmp="$(mktemp -d)"
    local bin="${tmp}/bin"
    local fake_repo="${tmp}/repo"
    mkdir -p "${bin}" "${fake_repo}"
    make_nvidia_smi_shim "${bin}"
    make_nvcc_shim "${bin}"
    stage_fake_tracer_so "${fake_repo}" "${bin}"

    local out_json="${tmp}/preflight.json"
    if PATH="${bin}:${PATH}" CLOUD_PREFLIGHT_REPO_ROOT="${fake_repo}" \
            bash "${PREFLIGHT}" --out "${out_json}" >"${tmp}/preflight.log" 2>&1; then
        local hash
        hash="$(python3 -c "
import json
with open('${out_json}') as f:
    m = json.load(f)
print(m['toolchain']['nvbit_build_hash'])
        ")"
        if [[ "${hash}" =~ ^[0-9a-f]{64}$ ]]; then
            pass "smoke_nvbit_injection_accepts_traces_subdir: nvbit_build_hash=${hash:0:12}... is a 64-char sha256"
        else
            fail "smoke_nvbit_injection_accepts_traces_subdir: nvbit_build_hash is not a valid sha256 (got=$(printf '%q' "${hash}"))"
            log "preflight log: $(tail -10 "${tmp}/preflight.log" 2>/dev/null || true)"
        fi
    else
        fail "smoke_nvbit_injection_accepts_traces_subdir: preflight exited non-zero"
        log "preflight log: $(tail -15 "${tmp}/preflight.log" 2>/dev/null || true)"
    fi

    rm -rf "${tmp}"
}

# ----------------------------------------------------------------------
# Smoke 2: NVBit probe rejects missing traces/ subdir.
# ----------------------------------------------------------------------
smoke_nvbit_rejects_missing_traces_subdir() {
    local tmp; tmp="$(mktemp -d)"
    local bin="${tmp}/bin"
    local fake_repo="${tmp}/repo"
    mkdir -p "${bin}" "${fake_repo}"
    make_nvidia_smi_shim "${bin}"
    cat > "${bin}/nvcc" <<'NVCC_NO_TRACES'
#!/usr/bin/env bash
args=("$@")
out=""
for i in "${!args[@]}"; do
    [[ "${args[$i]}" == "-o" ]] && out="${args[$((i+1))]:-}"
done
if [[ -z "${out}" ]]; then
    echo "nvcc: release 13.2, V13.2.89"; exit 0
fi
cat > "${out}" <<'BINARY'
#!/usr/bin/env bash
exit 0
BINARY
chmod +x "${out}"
exit 0
NVCC_NO_TRACES
    chmod +x "${bin}/nvcc"
    stage_fake_tracer_so "${fake_repo}" "${bin}"

    local out_json="${tmp}/preflight.json"
    if PATH="${bin}:${PATH}" CLOUD_PREFLIGHT_REPO_ROOT="${fake_repo}" \
            bash "${PREFLIGHT}" --out "${out_json}" >"${tmp}/preflight.log" 2>&1; then
        fail "smoke_nvbit_rejects_missing_traces_subdir: preflight should have exited non-zero"
        log "preflight log: $(tail -10 "${tmp}/preflight.log" 2>/dev/null || true)"
    else
        if grep -q "NvbitUnavailable" "${tmp}/preflight.log"; then
            pass "smoke_nvbit_rejects_missing_traces_subdir: preflight correctly hard-exits NvbitUnavailable"
        else
            fail "smoke_nvbit_rejects_missing_traces_subdir: preflight exited non-zero but did not emit NvbitUnavailable"
            log "preflight log: $(tail -10 "${tmp}/preflight.log" 2>/dev/null || true)"
        fi
    fi

    rm -rf "${tmp}"
}

# ----------------------------------------------------------------------
# Smoke 3: CUPTI Activity probe reports `available` via mock fixture.
# ----------------------------------------------------------------------
smoke_cupti_activity_available_via_mock() {
    local tmp; tmp="$(mktemp -d)"
    local mock="${tmp}/mock"
    mkdir -p "${mock}"
    printf 'NVIDIA GeForce RTX 5060 Laptop GPU, GPU-fake-uuid-0001, 595.58.03, 12.0, 2700, 14000, 2100, Disabled, Default\n' \
        > "${mock}/nvidia_smi.csv"
    printf '13.2\n' > "${mock}/nvcc_version.txt"
    printf 'a%.0s' {1..64} > "${mock}/nvbit_build_hash.txt"
    printf '\n' >> "${mock}/nvbit_build_hash.txt"
    printf 'available\n' > "${mock}/cluster_attr_probe.txt"
    printf 'available\n' > "${mock}/cupti_activity_probe.txt"
    printf 'ok\n' > "${mock}/nvbit_injection_probe.txt"
    printf '{"nvidia_smi_executable":true,"nvidia_dev_readable":true}\n' > "${mock}/permissions.json"

    local out_json="${tmp}/preflight.json"
    if bash "${PREFLIGHT}" --mock-input "${mock}" --out "${out_json}" >"${tmp}/preflight.log" 2>&1; then
        local cupti
        cupti="$(python3 -c "
import json
with open('${out_json}') as f:
    m = json.load(f)
print(m['capabilities']['cupti_activity_api_available'])
        ")"
        if [[ "${cupti}" == "available" ]]; then
            pass "smoke_cupti_activity_available_via_mock: capabilities.cupti_activity_api_available=available"
        else
            fail "smoke_cupti_activity_available_via_mock: cupti_activity_api_available='${cupti}' (expected 'available')"
        fi
    else
        fail "smoke_cupti_activity_available_via_mock: preflight exited non-zero"
        log "preflight log: $(tail -10 "${tmp}/preflight.log" 2>/dev/null || true)"
    fi

    rm -rf "${tmp}"
}

# ----------------------------------------------------------------------
# Smoke 4: validator records a cupti header_missing sentinel without gating.
# ----------------------------------------------------------------------
smoke_cupti_header_missing_records_ok_via_validator() {
    local tmp; tmp="$(mktemp -d)"
    local mock="${tmp}/mock"
    mkdir -p "${mock}"
    printf 'NVIDIA GeForce RTX 5060 Laptop GPU, GPU-fake-uuid-0001, 595.58.03, 12.0, 2700, 14000, 2100, Disabled, Default\n' \
        > "${mock}/nvidia_smi.csv"
    printf '13.2\n' > "${mock}/nvcc_version.txt"
    printf 'a%.0s' {1..64} > "${mock}/nvbit_build_hash.txt"
    printf '\n' >> "${mock}/nvbit_build_hash.txt"
    printf 'available\n' > "${mock}/cluster_attr_probe.txt"
    printf 'header_missing\n' > "${mock}/cupti_activity_probe.txt"
    # Both CUPTI fields are required recorded provenance. The runtime
    # value remains closed-vocabulary, while neither recognized runtime
    # nor legacy availability value gates admission under D8.
    printf 'available\n' > "${mock}/cupti_runtime_probe.txt"
    printf 'off\n' > "${mock}/cc_mode_probe.txt"
    printf 'full\n' > "${mock}/nvml_capability_matrix_probe.txt"
    printf 'ok\n' > "${mock}/nvbit_injection_probe.txt"
    printf '{"nvidia_smi_executable":true,"nvidia_dev_readable":true}\n' > "${mock}/permissions.json"

    local out_json="${tmp}/preflight.json"
    bash "${PREFLIGHT}" --mock-input "${mock}" --out "${out_json}" >"${tmp}/preflight.log" 2>&1 || true

    if [[ ! -f "${out_json}" ]]; then
        fail "smoke_cupti_header_missing_records_ok_via_validator: preflight did not emit a manifest"
        rm -rf "${tmp}"
        return
    fi

    local kind
    kind="$(python3 -c "
import sys, json
sys.path.insert(0, '${HERE}')
import preflight_manifest as pm
with open('${out_json}') as f:
    m = json.load(f)
result = pm.validate_preflight_manifest(m)
print('OK' if result.ok else result.error.kind.value)
    ")"
    if [[ "${kind}" == "OK" ]]; then
        pass "smoke_cupti_header_missing_records_ok_via_validator: validator records header_missing and returns OK"
    else
        fail "smoke_cupti_header_missing_records_ok_via_validator: expected OK, got '${kind}'"
    fi

    rm -rf "${tmp}"
}

REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
MOCK_INPUTS_ROOT="${REPO_ROOT}/tools/cloud/mock_inputs"

# ----------------------------------------------------------------------
# Helper: run preflight.sh against a committed fixture and validate.
#   $1 = fixture subdir name under mock_inputs/
#   $2 = manifest field path (jq-style: "capabilities.cupti_activity_api_runtime")
#   $3 = expected manifest field value (or "" if not checked)
#   $4 = expected validator outcome ("OK" or PreflightErrorKind value)
# ----------------------------------------------------------------------
run_committed_fixture_smoke() {
    local label="$1"
    local fixture="$2"
    local field_path="$3"
    local expected_field="$4"
    local expected_validator="$5"

    local tmp; tmp="$(mktemp -d)"
    local out_json="${tmp}/preflight.json"
    local fixture_dir="${MOCK_INPUTS_ROOT}/${fixture}"

    if [[ ! -d "${fixture_dir}" ]]; then
        fail "${label}: committed fixture ${fixture_dir} does not exist"
        rm -rf "${tmp}"; return
    fi

    if ! bash "${PREFLIGHT}" --mock-input "${fixture_dir}" --out "${out_json}" >"${tmp}/preflight.log" 2>&1; then
        fail "${label}: preflight exited non-zero against fixture ${fixture}"
        log "preflight log: $(tail -10 "${tmp}/preflight.log" 2>/dev/null || true)"
        rm -rf "${tmp}"; return
    fi

    if [[ -n "${expected_field}" ]]; then
        local actual_field
        actual_field="$(python3 -c "
import json, sys
with open('${out_json}') as f: m = json.load(f)
parts = '${field_path}'.split('.')
v = m
for p in parts: v = v[p]
# Print bools in JSON form (true/false) so callers can write the
# JSON-canonical expected value rather than Python's True/False.
# All other types print via Python's default str().
if isinstance(v, bool):
    print('true' if v else 'false')
else:
    print(v)
")"
        if [[ "${actual_field}" != "${expected_field}" ]]; then
            fail "${label}: expected ${field_path}='${expected_field}' but got '${actual_field}'"
            rm -rf "${tmp}"; return
        fi
    fi

    local validator_outcome
    validator_outcome="$(python3 -c "
import json, sys
sys.path.insert(0, '${HERE}')
import preflight_manifest as pm
with open('${out_json}') as f: m = json.load(f)
result = pm.validate_preflight_manifest(m)
print('OK' if result.ok else result.error.kind.value)
")"
    if [[ "${validator_outcome}" != "${expected_validator}" ]]; then
        fail "${label}: expected validator '${expected_validator}' but got '${validator_outcome}'"
        rm -rf "${tmp}"; return
    fi

    pass "${label}: fixture=${fixture} ${field_path}=${expected_field:-(any)} validator=${validator_outcome}"
    rm -rf "${tmp}"
}

# ----------------------------------------------------------------------
# Smokes 5a-5g: per-status committed cupti_smoke_<status> fixtures.
# Verifies AC1's full enum surface and AC1.2's fixture coverage.
# ----------------------------------------------------------------------
smoke_cupti_runtime_status_per_committed_fixture() {
    run_committed_fixture_smoke \
        "smoke_cupti_runtime_available_via_committed_fixture" \
        "cupti_smoke_available" \
        "capabilities.cupti_activity_api_runtime" \
        "available" \
        "OK"
    run_committed_fixture_smoke \
        "smoke_cupti_runtime_compile_failed_via_committed_fixture" \
        "cupti_smoke_compile_failed" \
        "capabilities.cupti_activity_api_runtime" \
        "compile_failed" \
        "OK"
    run_committed_fixture_smoke \
        "smoke_cupti_runtime_link_failed_via_committed_fixture" \
        "cupti_smoke_link_failed" \
        "capabilities.cupti_activity_api_runtime" \
        "link_failed" \
        "OK"
    run_committed_fixture_smoke \
        "smoke_cupti_runtime_runtime_failed_via_committed_fixture" \
        "cupti_smoke_runtime_failed" \
        "capabilities.cupti_activity_api_runtime" \
        "runtime_failed" \
        "OK"
    run_committed_fixture_smoke \
        "smoke_cupti_runtime_insufficient_privilege_via_committed_fixture" \
        "cupti_smoke_insufficient_privilege" \
        "capabilities.cupti_activity_api_runtime" \
        "insufficient_privilege" \
        "OK"
    run_committed_fixture_smoke \
        "smoke_cupti_runtime_cc_unsupported_via_committed_fixture" \
        "cupti_smoke_cc_unsupported" \
        "capabilities.cupti_activity_api_runtime" \
        "cc_unsupported" \
        "OK"
    run_committed_fixture_smoke \
        "smoke_cupti_runtime_mock_input_missing_via_committed_fixture" \
        "cupti_smoke_mock_input_missing" \
        "capabilities.cupti_activity_api_runtime" \
        "mock_input_missing" \
        "OK"
}

# ----------------------------------------------------------------------
# Smoke 6: deployment_mode_cc_on fixture fires the CC-on validator code.
# ----------------------------------------------------------------------
smoke_cc_on_rejects_via_committed_fixture() {
    run_committed_fixture_smoke \
        "smoke_cc_on_rejects_via_committed_fixture" \
        "deployment_mode_cc_on" \
        "capabilities.cc_mode" \
        "on" \
        "ConfidentialComputingNotSupportedForCollection"
}

# ----------------------------------------------------------------------
# Smoke 7: deployment_mode_mig_enabled fixture fires the MIG validator.
# ----------------------------------------------------------------------
smoke_mig_enabled_rejects_via_committed_fixture() {
    run_committed_fixture_smoke \
        "smoke_mig_enabled_rejects_via_committed_fixture" \
        "deployment_mode_mig_enabled" \
        "gpu.mig_mode" \
        "Enabled" \
        "MigSliceNotValidForCollectionEvidence"
}

# ----------------------------------------------------------------------
# Smoke 7a: deployment_mode_mps_on fixture round-trips through the
# preflight emitter and validator. Detect-and-declare per DEC-2 +
# cupti-activity-api KB: the validator records `mps_active=true` on
# the manifest and validates clean (no typed rejection). Downstream
# artifacts read the field to reason about CUPTI per-client
# attribution semantics.
# ----------------------------------------------------------------------
smoke_mps_on_accepts_via_committed_fixture() {
    run_committed_fixture_smoke \
        "smoke_mps_on_accepts_via_committed_fixture" \
        "deployment_mode_mps_on" \
        "capabilities.mps_active" \
        "true" \
        "OK"
}

# ----------------------------------------------------------------------
# Smoke 7b: deployment_mode_vgpu_on fixture round-trips through the
# preflight emitter and validator. Detect-and-declare per DEC-2 +
# KB: the validator records `vgpu_active=true` on the manifest and
# validates clean. Downstream artifacts use the field to choose
# `_KERNEL` fallback when HES source-correlation is unavailable
# under vGPU.
# ----------------------------------------------------------------------
smoke_vgpu_on_accepts_via_committed_fixture() {
    run_committed_fixture_smoke \
        "smoke_vgpu_on_accepts_via_committed_fixture" \
        "deployment_mode_vgpu_on" \
        "capabilities.vgpu_active" \
        "true" \
        "OK"
}

# ----------------------------------------------------------------------
# Smoke 7c: deployment_mode_mps_and_vgpu_on fixture: both fields
# True. The validator records both fields and validates clean per
# the DEC-2 detect-and-declare contract. We assert the mps_active
# field directly here; smoke 7d below asserts vgpu_active off the
# same fixture so both fields are exercised.
# ----------------------------------------------------------------------
smoke_mps_and_vgpu_on_accepts_via_committed_fixture() {
    run_committed_fixture_smoke \
        "smoke_mps_and_vgpu_on_accepts_via_committed_fixture" \
        "deployment_mode_mps_and_vgpu_on" \
        "capabilities.mps_active" \
        "true" \
        "OK"
}

# ----------------------------------------------------------------------
# Smoke 7d: companion to 7c — asserts the vgpu_active field on the
# combined fixture also round-trips correctly.
# ----------------------------------------------------------------------
smoke_mps_and_vgpu_on_vgpu_field_via_committed_fixture() {
    run_committed_fixture_smoke \
        "smoke_mps_and_vgpu_on_vgpu_field_via_committed_fixture" \
        "deployment_mode_mps_and_vgpu_on" \
        "capabilities.vgpu_active" \
        "true" \
        "OK"
}

# ----------------------------------------------------------------------
# Smoke 8: AC1.1 status precedence is enforced by the staged elif
# chain in preflight.sh's CUPTI runtime block (compile-only nvcc is
# attempted before link-stage nvcc, link before run, run before the
# insufficient-privilege / cc-unsupported stderr-grep classification).
# This smoke verifies (a) every enum value in the AC1.1 precedence
# is reachable somewhere in the source (catches accidental enum drop)
# and (b) the staged Stage-1/2/3 markers are present in source order
# (catches reordering of the staged probe). This is a structural
# invariant test; behavioral coverage (each enum value individually)
# is provided by the seven per-status committed mock fixtures.
#
# A real-mode multi-failure simulation that actually exercises the
# precedence at runtime is NOT attempted here because it requires
# staging a fake CUPTI under one of preflight.sh's hard-coded search
# roots (e.g. /opt/cuda), which needs root in the typical environment.
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Smoke: AC1.1 status precedence behavioral test (real-mode probe).
# Stages a fake CUDA/CUPTI installation + a fake nvcc that fails compile
# while a downstream run stage WOULD have produced
# PROBE_INSUFFICIENT_PRIVILEGES. The staged probe must record
# `compile_failed` (compile is reached first per the AC1.1 precedence
# ordering); if it ever records `available` for a failed compile, fail
# with `cupti_smoke_status_inconsistent`.
#
# Uses the CLOUD_PREFLIGHT_CUPTI_SEARCH_ROOTS env override (added in
# R3 specifically for this test) to inject the fake CUPTI root
# without requiring root privilege.
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Smoke: AC1.2 fixture-coverage cross-check. Derives the required
# status set from preflight_manifest.CUPTI_ACTIVITY_RUNTIME_VALID_VALUES
# and asserts that exactly one committed
# tools/cloud/mock_inputs/cupti_smoke_<status>/ directory exists per
# status (and no extras). If any required status lacks a fixture, fail
# with `cupti_smoke_status_uncovered`. If any extra unknown fixture
# exists, fail with `cupti_smoke_status_unexpected_fixture`.
# ----------------------------------------------------------------------
smoke_cupti_smoke_fixtures_match_validator_enum() {
    local audit
    audit="$(python3 - "${HERE}" "${MOCK_INPUTS_ROOT}" <<'PY'
import sys, os
sys.path.insert(0, sys.argv[1])
import preflight_manifest as pm
mock_root = sys.argv[2]
required = sorted(pm.CUPTI_ACTIVITY_RUNTIME_VALID_VALUES)
present = sorted(
    name[len("cupti_smoke_"):]
    for name in os.listdir(mock_root)
    if name.startswith("cupti_smoke_")
    and os.path.isdir(os.path.join(mock_root, name))
)
missing = sorted(set(required) - set(present))
unexpected = sorted(set(present) - set(required))
print(f"required={required}")
print(f"present={present}")
if missing:
    print(f"MISSING={missing}")
if unexpected:
    print(f"UNEXPECTED={unexpected}")
print("OK" if not missing and not unexpected else "FAIL")
PY
)"
    if [[ "${audit}" == *"OK"* && "${audit}" != *"FAIL"* ]]; then
        local present_summary
        present_summary="$(echo "${audit}" | grep '^present=' | sed 's/^present=//')"
        pass "smoke_cupti_smoke_fixtures_match_validator_enum: every required status has exactly one cupti_smoke_<status> fixture; present=${present_summary}"
    else
        local missing_line unexpected_line
        missing_line="$(echo "${audit}" | grep '^MISSING=' || true)"
        unexpected_line="$(echo "${audit}" | grep '^UNEXPECTED=' || true)"
        if [[ -n "${missing_line}" ]]; then
            fail "smoke_cupti_smoke_fixtures_match_validator_enum: cupti_smoke_status_uncovered — required statuses lack a fixture: ${missing_line}"
        fi
        if [[ -n "${unexpected_line}" ]]; then
            fail "smoke_cupti_smoke_fixtures_match_validator_enum: cupti_smoke_status_unexpected_fixture — fixtures present for statuses not in the validator enum: ${unexpected_line}"
        fi
    fi
}

smoke_status_precedence_compile_failed_via_real_mode_shim() {
    local tmp; tmp="$(mktemp -d)"
    local bin="${tmp}/bin"
    local fake_repo="${tmp}/repo"
    local fake_cuda="${tmp}/cuda"
    mkdir -p "${bin}" "${fake_repo}" \
             "${fake_cuda}/extras/CUPTI/include" \
             "${fake_cuda}/extras/CUPTI/lib64"

    # Stage a fake CUDA + CUPTI install (header + lib both present).
    touch "${fake_cuda}/extras/CUPTI/include/cupti.h"
    : > "${fake_cuda}/extras/CUPTI/lib64/libcupti.so.13.0"

    make_nvidia_smi_shim "${bin}"
    stage_fake_tracer_so "${fake_repo}" "${bin}"

    # Custom nvcc shim:
    #   * Version query (no -o):       returns version banner.
    #   * Compile-only with -I to fake
    #     CUPTI include AND -c flag:    fails (simulates CUPTI compile
    #                                    failure; downstream runtime
    #                                    privilege would have fired).
    #   * Other compile invocations
    #     (cluster_attr, NVBit probe): writes a runnable stub matching
    #                                    the make_nvcc_shim convention
    #                                    so the cluster_attr and NVBit
    #                                    probes still pass.
    cat > "${bin}/nvcc" <<NVCC_SHIM
#!/usr/bin/env bash
args=("\$@")
out=""
has_cupti_include=0
has_compile_only=0
for i in "\${!args[@]}"; do
    [[ "\${args[\$i]}" == "-o" ]] && out="\${args[\$((i+1))]:-}"
    [[ "\${args[\$i]}" == "-I${fake_cuda}/extras/CUPTI/include" ]] && has_cupti_include=1
    [[ "\${args[\$i]}" == "-c" ]] && has_compile_only=1
done
# Simulate CUPTI compile failure for the runtime-probe Stage 1.
if (( has_cupti_include == 1 && has_compile_only == 1 )); then
    echo "fake nvcc: simulated CUPTI compile failure (downstream PROBE_INSUFFICIENT_PRIVILEGES would have fired)" >&2
    exit 1
fi
# Version query.
if [[ -z "\${out}" ]]; then
    echo "nvcc: NVIDIA (R) Cuda compiler driver"
    echo "Cuda compilation tools, release 13.2, V13.2.89"
    exit 0
fi
# Other compiles (cluster_attr / NVBit injection probe): writable stub.
cat > "\${out}" <<'BINARY_EOF'
#!/usr/bin/env bash
mkdir -p "./traces/threadblocks/device_0/stream_0/kernel_0"
printf '%s' 'MigIABIkEiIIABIeEhwSCm5vcF9rZXJuZWxCBggBEAEYAUoGCAEQARgBOhJzbTg2X2FkZHJfY2FycnlfdjFCBQECAwYI' | base64 -d > "./traces/dynamic_trace.pb"
printf '%s' 'CgASFggAEhISEAiAIBABGAEgATgEQABYgCA=' | base64 -d > "./traces/threadblocks/device_0/stream_0/kernel_0/0.pb"
printf 'kernel_id,total_insts,total_reported_insts\n0,1,1\n' > "./traces/stats.csv"
exit 0
BINARY_EOF
chmod +x "\${out}"
exit 0
NVCC_SHIM
    chmod +x "${bin}/nvcc"

    local out_json="${tmp}/preflight.json"
    if PATH="${bin}:${PATH}" \
       CLOUD_PREFLIGHT_REPO_ROOT="${fake_repo}" \
       CLOUD_PREFLIGHT_CUPTI_SEARCH_ROOTS="${fake_cuda}" \
       bash "${PREFLIGHT}" --out "${out_json}" >"${tmp}/preflight.log" 2>&1; then
        local runtime
        runtime="$(python3 -c "
import json
with open('${out_json}') as f:
    print(json.load(f)['capabilities']['cupti_activity_api_runtime'])
")"
        if [[ "${runtime}" == "compile_failed" ]]; then
            pass "smoke_status_precedence_compile_failed_via_real_mode_shim: staged probe correctly recorded 'compile_failed' (compile_failed wins over downstream insufficient_privilege per AC1.1)"
        elif [[ "${runtime}" == "available" ]]; then
            fail "smoke_status_precedence_compile_failed_via_real_mode_shim: cupti_smoke_status_inconsistent — staged probe recorded 'available' for a failed compile"
        else
            fail "smoke_status_precedence_compile_failed_via_real_mode_shim: expected 'compile_failed' but got '${runtime}'"
            log "preflight log: $(tail -10 "${tmp}/preflight.log" 2>/dev/null || true)"
        fi
    else
        fail "smoke_status_precedence_compile_failed_via_real_mode_shim: preflight exited non-zero"
        log "preflight log: $(tail -15 "${tmp}/preflight.log" 2>/dev/null || true)"
    fi

    rm -rf "${tmp}"
}

smoke_status_precedence_invariant_in_source() {
    local src="${PREFLIGHT}"
    local errors=()
    # (a) Every enum value in the AC1.1 precedence MUST appear at
    # least once in the source (catches accidental enum drop).
    local enum_values=(
        '"mock_input_missing"'
        '"compile_failed"'
        '"link_failed"'
        '"runtime_failed"'
        '"insufficient_privilege"'
        '"cc_unsupported"'
        '"available"'
    )
    for v in "${enum_values[@]}"; do
        if ! grep -F -q -- "CUPTI_RUNTIME=${v}" "${src}"; then
            errors+=("AC1.1 precedence enum value not found in source: CUPTI_RUNTIME=${v}")
        fi
    done
    # (b) The staged Stage-1/2/3 markers MUST appear in source order
    # in the CUPTI runtime block (catches reordering of the staged
    # probe regardless of what enum values are assigned).
    local stage_markers=(
        'Stage 1: compile-only.'
        'Stage 2: link.'
        'Stage 3: run.'
    )
    local previous_line=0
    for marker in "${stage_markers[@]}"; do
        local found_line
        found_line="$(grep -n -F -- "${marker}" "${src}" | head -1 | cut -d: -f1)"
        if [[ -z "${found_line}" ]]; then
            errors+=("AC1.1 precedence stage marker not found in source: '${marker}'")
        elif (( found_line < previous_line )); then
            errors+=("AC1.1 precedence staged-probe order broken: '${marker}' at line ${found_line} appears before earlier marker at line ${previous_line}")
        else
            previous_line="${found_line}"
        fi
    done
    if (( ${#errors[@]} > 0 )); then
        fail "smoke_status_precedence_invariant_in_source: AC1.1 precedence invariants broken in preflight.sh:"
        for e in "${errors[@]}"; do log "${e}"; done
    else
        pass "smoke_status_precedence_invariant_in_source: all 7 AC1.1 enum values present in preflight.sh source AND Stage-1/2/3 markers appear in source order"
    fi
}

# ----------------------------------------------------------------------
# Real-probe behavior smokes for `probe_mps_active_real` and
# `probe_vgpu_active_real`. Each smoke builds `${tmp}/fakebin/{pgrep,
# nvidia-smi}` shims, prepends them to PATH, and invokes preflight.sh
# with `CLOUD_PREFLIGHT_PROBE_ONLY=mps_vgpu` to short-circuit the
# script after the two probe functions return. Stdout is parsed for
# the two `mps_active=<bool>` + `vgpu_active=<bool>` lines and
# compared against the expected values.
#
# These behavior smokes complement the source-level static scans in
# `test_preflight_manifest.py::PreflightShellSourceProbeSemanticsTests`
# (token-presence guards). The behavior smokes catch broken regex,
# wrong precedence around `No supported devices in vGPU mode`, and
# daemon/server-only behavioral regressions; the static scans catch
# token re-introduction or removal.
# ----------------------------------------------------------------------

# Helper: stage fakebin scripts, run probe-only preflight, and assert
# the two parsed booleans match expected values.
#
# Args:
#   $1  label                       smoke label for pass/fail messages
#   $2  pgrep_match_token           argument value that fake `pgrep -x`
#                                   should match (i.e. that pgrep returns
#                                   exit 0 for); empty string means no
#                                   pgrep invocation should match
#   $3  nvidia_smi_vgpu_q_stdout    text to print on `nvidia-smi vgpu -q`
#                                   (printed regardless of exit code)
#   $4  nvidia_smi_vgpu_q_exit      exit code for `nvidia-smi vgpu -q`
#                                   (0 = success; non-zero = failure)
#   $5  cuda_mps_pipe_dir_state     one of: "unset" | "set_existing"
#                                   | "set_missing"
#   $6  default_pipe_dir_state      one of: "absent" | "present"
#   $7  expected_mps_active         "true" or "false"
#   $8  expected_vgpu_active        "true" or "false"
_run_probe_only_with_fakebin() {
    local label="$1"
    local pgrep_match_token="$2"
    local nvidia_smi_vgpu_q_stdout="$3"
    local nvidia_smi_vgpu_q_exit="$4"
    local cuda_mps_pipe_dir_state="$5"
    local default_pipe_dir_state="$6"
    local expected_mps_active="$7"
    local expected_vgpu_active="$8"

    local tmp; tmp="$(mktemp -d)"
    local fakebin="${tmp}/fakebin"
    mkdir -p "${fakebin}"

    # Fake pgrep: matches only when invoked as `pgrep -x <token>` for
    # the configured token. Anything else exits 1.
    cat >"${fakebin}/pgrep" <<EOF_PGREP
#!/usr/bin/env bash
# Fake pgrep for behavior smokes. Returns 0 only when called as
# 'pgrep -x ${pgrep_match_token}'.
if [[ "\$#" -eq 2 ]] && [[ "\$1" == "-x" ]] && [[ "\$2" == "${pgrep_match_token}" ]]; then
    echo 1234
    exit 0
fi
exit 1
EOF_PGREP
    chmod +x "${fakebin}/pgrep"

    # Fake nvidia-smi: prints the configured vgpu -q stdout and exits
    # with the configured code only when invoked as `nvidia-smi vgpu -q`.
    # Other invocations exit 0 with empty output (the probe does not
    # call nvidia-smi for any other purpose under PROBE_ONLY=mps_vgpu).
    local stdout_b64
    stdout_b64="$(printf '%s' "${nvidia_smi_vgpu_q_stdout}" | base64 -w0)"
    cat >"${fakebin}/nvidia-smi" <<EOF_NVS
#!/usr/bin/env bash
# Fake nvidia-smi for behavior smokes. Only handles 'vgpu -q'.
if [[ "\$#" -eq 2 ]] && [[ "\$1" == "vgpu" ]] && [[ "\$2" == "-q" ]]; then
    printf '%s' "${stdout_b64}" | base64 -d
    exit ${nvidia_smi_vgpu_q_exit}
fi
exit 0
EOF_NVS
    chmod +x "${fakebin}/nvidia-smi"

    # Configure the MPS pipe-dir env vars for this case.
    local cuda_mps_pipe_dir_arg=""
    case "${cuda_mps_pipe_dir_state}" in
        unset)        unset CUDA_MPS_PIPE_DIRECTORY 2>/dev/null || true ;;
        set_existing) export CUDA_MPS_PIPE_DIRECTORY="${tmp}/mps_pipe_existing"; mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" ;;
        set_missing)  export CUDA_MPS_PIPE_DIRECTORY="${tmp}/mps_pipe_missing" ;;  # do NOT mkdir
        *) fail "${label}: unknown cuda_mps_pipe_dir_state='${cuda_mps_pipe_dir_state}'"; rm -rf "${tmp}"; return ;;
    esac

    local default_pipe_dir
    case "${default_pipe_dir_state}" in
        absent)  default_pipe_dir="${tmp}/default_pipe_absent" ;;  # do NOT mkdir
        present) default_pipe_dir="${tmp}/default_pipe_present"; mkdir -p "${default_pipe_dir}" ;;
        *) fail "${label}: unknown default_pipe_dir_state='${default_pipe_dir_state}'"; rm -rf "${tmp}"; return ;;
    esac

    # Run probe-only preflight with fakebin on PATH.
    local probe_out
    probe_out="$(PATH="${fakebin}:${PATH}" \
                 CLOUD_PREFLIGHT_PROBE_ONLY=mps_vgpu \
                 CLOUD_PREFLIGHT_MPS_DEFAULT_PIPE_DIR="${default_pipe_dir}" \
                 bash "${PREFLIGHT}" 2>&1)" || {
        fail "${label}: preflight exited non-zero under PROBE_ONLY=mps_vgpu"
        log "stdout: ${probe_out}"
        rm -rf "${tmp}"
        unset CUDA_MPS_PIPE_DIRECTORY 2>/dev/null || true
        return
    }

    local actual_mps actual_vgpu
    actual_mps="$(awk -F= '/^mps_active=/ {print $2}' <<<"${probe_out}")"
    actual_vgpu="$(awk -F= '/^vgpu_active=/ {print $2}' <<<"${probe_out}")"

    if [[ "${actual_mps}" != "${expected_mps_active}" ]]; then
        fail "${label}: expected mps_active='${expected_mps_active}' but got '${actual_mps}' (full output: ${probe_out//$'\n'/ \\n })"
        rm -rf "${tmp}"
        unset CUDA_MPS_PIPE_DIRECTORY 2>/dev/null || true
        return
    fi
    if [[ "${actual_vgpu}" != "${expected_vgpu_active}" ]]; then
        fail "${label}: expected vgpu_active='${expected_vgpu_active}' but got '${actual_vgpu}' (full output: ${probe_out//$'\n'/ \\n })"
        rm -rf "${tmp}"
        unset CUDA_MPS_PIPE_DIRECTORY 2>/dev/null || true
        return
    fi

    pass "${label}: mps_active=${actual_mps} vgpu_active=${actual_vgpu} (pgrep_match='${pgrep_match_token}' vgpu_q_exit=${nvidia_smi_vgpu_q_exit} mps_pipe=${cuda_mps_pipe_dir_state} default_pipe=${default_pipe_dir_state})"
    rm -rf "${tmp}"
    unset CUDA_MPS_PIPE_DIRECTORY 2>/dev/null || true
}

# Case A: no MPS daemon/server, no MPS pipe dirs, vGPU query exits
# non-zero. Baseline bare-container case. Expect both false.
smoke_probe_no_mps_no_vgpu_baseline() {
    _run_probe_only_with_fakebin \
        "smoke_probe_no_mps_no_vgpu_baseline" \
        "" \
        "" \
        "1" \
        "unset" \
        "absent" \
        "false" \
        "false"
}

# Case B: only the MPS control daemon is running. Expect mps=true,
# vgpu=false.
smoke_probe_mps_control_daemon_alone() {
    _run_probe_only_with_fakebin \
        "smoke_probe_mps_control_daemon_alone" \
        "nvidia-cuda-mps-control" \
        "" \
        "1" \
        "unset" \
        "absent" \
        "true" \
        "false"
}

# Case C: only the per-user MPS server process is running (the
# control daemon may have spawned it on a previous client connection
# and exited). Expect mps=true, vgpu=false.
smoke_probe_mps_server_only() {
    _run_probe_only_with_fakebin \
        "smoke_probe_mps_server_only" \
        "nvidia-cuda-mps-server" \
        "" \
        "1" \
        "unset" \
        "absent" \
        "true" \
        "false"
}

# Case D: documented active vGPU output (`Active vGPUs : 1` +
# `vGPU ID :` line). pgrep returns nothing. Expect vgpu=true,
# mps=false.
smoke_probe_vgpu_active_via_documented_query() {
    local active_output="GPU 00000000:01:00.0
    Active vGPUs                      : 1
    vGPU ID                           : 3251634178
        VM ID                         : 12345
        VM Name                       : ubuntu-vgpu-guest"
    _run_probe_only_with_fakebin \
        "smoke_probe_vgpu_active_via_documented_query" \
        "" \
        "${active_output}" \
        "0" \
        "unset" \
        "absent" \
        "false" \
        "true"
}

# Case E: documented no-vGPU output (`No supported devices in vGPU
# mode`). vgpu query exits 0 with that text. Expect vgpu=false.
smoke_probe_vgpu_no_supported_devices() {
    _run_probe_only_with_fakebin \
        "smoke_probe_vgpu_no_supported_devices" \
        "" \
        "No supported devices in vGPU mode" \
        "0" \
        "unset" \
        "absent" \
        "false" \
        "false"
}

# Case F: vgpu query command fails (bare-metal host where vGPU
# support is not compiled into the driver). Expect vgpu=false.
smoke_probe_vgpu_query_command_fails() {
    _run_probe_only_with_fakebin \
        "smoke_probe_vgpu_query_command_fails" \
        "" \
        "" \
        "6" \
        "unset" \
        "absent" \
        "false" \
        "false"
}

echo "Running preflight.sh shell-level regression smokes..."
smoke_nvbit_injection_accepts_traces_subdir
smoke_nvbit_rejects_missing_traces_subdir
smoke_cupti_activity_available_via_mock
smoke_cupti_header_missing_records_ok_via_validator
smoke_cupti_runtime_status_per_committed_fixture
smoke_cc_on_rejects_via_committed_fixture
smoke_mig_enabled_rejects_via_committed_fixture
smoke_mps_on_accepts_via_committed_fixture
smoke_vgpu_on_accepts_via_committed_fixture
smoke_mps_and_vgpu_on_accepts_via_committed_fixture
smoke_mps_and_vgpu_on_vgpu_field_via_committed_fixture
smoke_cupti_smoke_fixtures_match_validator_enum
smoke_status_precedence_invariant_in_source
smoke_status_precedence_compile_failed_via_real_mode_shim
smoke_probe_no_mps_no_vgpu_baseline
smoke_probe_mps_control_daemon_alone
smoke_probe_mps_server_only
smoke_probe_vgpu_active_via_documented_query
smoke_probe_vgpu_no_supported_devices
smoke_probe_vgpu_query_command_fails

if (( FAIL_COUNT > 0 )); then
    printf '\n%s\n' "test_preflight_probes: ${FAIL_COUNT} smoke(s) failed" >&2
    exit 1
fi
printf '\n%s\n' "test_preflight_probes: all smokes passed"
exit 0
