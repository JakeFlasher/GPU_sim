"""Regression guards for the pinned NVBit build and runtime contract."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
NVBIT = REPO / "tools" / "nvbit_tracer"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\nset -e\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _fake_pinned_tree(
    tmp_path: Path,
    *,
    nvcc: str = "13.2",
    protoc: str = "34.1",
    cxx: str = "14.3.0",
    protobuf: str = "34.1.0",
) -> tuple[Path, dict[str, str]]:
    repo = tmp_path / "repo"
    script = repo / "tools" / "nvbit_tracer" / "pinned_build_env.sh"
    script.parent.mkdir(parents=True)
    script.write_text(_text(NVBIT / "pinned_build_env.sh"), encoding="utf-8")
    lock = script.parent / "toolchain" / "pb_explicit.lock"
    lock.parent.mkdir()
    lock.write_bytes((NVBIT / "toolchain" / "pb_explicit.lock").read_bytes())
    pb = repo / ".toolchain" / "pb"
    (pb / "lib" / "pkgconfig").mkdir(parents=True)
    _executable(pb / "bin" / "protoc", f"printf 'libprotoc {protoc}\\n'\n")
    cuda = repo / "cuda"
    _executable(
        cuda / "bin" / "nvcc",
        f"printf 'Cuda compilation tools, release {nvcc}, V{nvcc}.0\\n'\n",
    )
    host_cxx = _executable(
        repo / "host" / "g++-14", f"printf '{cxx}\\n'\n"
    )
    fake_bin = repo / "fake-bin"
    _executable(
        fake_bin / "pkg-config",
        """
case " $* " in
  *" --variable=prefix "*) printf '%s\\n' "${FAKE_PB_PREFIX}" ;;
  *" --modversion "*) printf '%s\\n' "${FAKE_PB_VERSION}" ;;
  *) exit 2 ;;
esac
""",
    )
    env = {
        "HOME": str(tmp_path),
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "FAKE_PB_PREFIX": str(pb),
        "FAKE_PB_VERSION": protobuf,
        "NVBIT_TRACER_CUDA_ROOT": str(cuda),
        "NVBIT_TRACER_HOST_CXX": str(host_cxx),
    }
    return script, env


def test_makefiles_keep_recursive_include_paths_private() -> None:
    tracer_make = _text(NVBIT / "tracer_tool" / "Makefile")
    traces_make = _text(NVBIT / "traces_enhanced" / "Makefile")

    assert "PROTOBUF_CFLAGS:=" in tracer_make
    assert "TRACER_INCLUDES=" in tracer_make
    assert "$(TRACER_INCLUDES)" in tracer_make
    assert "$(INCLUDES)" not in tracer_make

    assert "PROTOC?=protoc" in traces_make
    assert "TRACE_PROTO_INCLUDES=" in traces_make
    assert "$(TRACE_PROTO_INCLUDES)" in traces_make
    assert "$(INCLUDES)" not in traces_make
    assert "$(CXX) -c -o $@ $< $(CFLAGS)" in traces_make


def test_pinned_environment_exports_the_complete_contract() -> None:
    env_script = _text(NVBIT / "pinned_build_env.sh")
    assert "NVBIT_TRACER_INCLUDES" not in env_script
    assert '_NVBIT_PB_LOCK_SHA256="d522b3ee1560e98512cd460b213adee5a901f77aef0e4fae3ba94699e3b1e222"' in env_script
    assert 'if [[ "${BASH_SOURCE[0]}" == "$0" ]]' in env_script
    assert 'if [[ "$#" -eq 1 && "$1" == "--ensure" ]]' in env_script
    assert 'elif [[ "$#" -eq 1 && "$1" == "--ensure-external" ]]' in env_script
    assert 'pb_prefix="${NVBIT_PB_PREFIX:-$default_prefix}"' in env_script
    assert '"$installer" create -y --copy -p "$pb_prefix" --file "$lock_path"' in env_script
    assert "_nvbit_normalize_external_prefix" in env_script
    assert "_nvbit_validate_external_destination" in env_script
    assert 'export NVCC_CCBIN="$host_cxx"' in env_script
    assert 'export CXX="$host_cxx"' in env_script
    assert 'export PROTOC="$protoc"' in env_script
    assert 'export PKG_CONFIG_FLAGS="--define-prefix"' in env_script
    assert 'nvcc_release" != "13.2"' in env_script
    assert 'protoc_release" != "$_NVBIT_PROTOC_VERSION"' in env_script


def test_external_pinned_handoff_is_operational_and_dealiases_members(
    tmp_path: Path,
) -> None:
    script, env = _fake_pinned_tree(tmp_path)
    repo = tmp_path / "repo"
    shutil.rmtree(repo / ".toolchain" / "pb")
    external_parent = tmp_path / "external"
    external_parent.mkdir()
    external = external_parent / "protobuf-34.1"
    fake_bin = repo / "fake-bin"
    _executable(
        fake_bin / "micromamba",
        """
prefix=
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "-p" ]]; then prefix="$2"; shift 2; else shift; fi
done
[[ -n "$prefix" ]]
mkdir -p "$prefix/bin" "$prefix/include" "$prefix/lib/pkgconfig"
printf '#!/usr/bin/env bash\nprintf "libprotoc 34.1\\n"\n' > "$prefix/bin/protoc-real"
chmod 755 "$prefix/bin/protoc-real"
ln -s protoc-real "$prefix/bin/protoc"
printf fixture > "$prefix/lib/libprotobuf-real.so"
ln "$prefix/lib/libprotobuf-real.so" "$prefix/lib/libprotobuf-hardlink.so"
printf 'Version: 34.1.0\n' > "$prefix/lib/pkgconfig/protobuf.pc"
""",
    )
    env["NVBIT_PB_PREFIX"] = str(external)
    env["FAKE_PB_PREFIX"] = str(external)

    result = subprocess.run(
        ["bash", str(script), "--ensure-external"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert external.is_dir()
    for path in external.rglob("*"):
        metadata = path.lstat()
        assert not path.is_symlink()
        if path.is_file():
            assert metadata.st_nlink == 1
    assert subprocess.run(
        [str(external / "bin/protoc"), "--version"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip() == "libprotoc 34.1"


def test_external_pinned_handoff_rejects_dot_and_symlink_ancestors(
    tmp_path: Path,
) -> None:
    script, base_env = _fake_pinned_tree(tmp_path)
    external_parent = tmp_path / "external"
    (external_parent / "child").mkdir(parents=True)
    alias = tmp_path / "external-alias"
    alias.symlink_to(external_parent, target_is_directory=True)
    bad_destinations = (
        external_parent / ".." / "repo" / "escaped-prefix",
        alias / "child" / "aliased-prefix",
    )

    for destination in bad_destinations:
        env = base_env | {"NVBIT_PB_PREFIX": str(destination)}
        result = subprocess.run(
            ["bash", str(script), "--ensure-external"],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "exact path validation" in result.stderr
        assert not destination.exists()


def test_top_level_make_default_goal_builds_all() -> None:
    top_make = _text(NVBIT / "Makefile")
    assert ".DEFAULT_GOAL := all" in top_make


def test_sourcing_environment_selects_exact_fake_prefix_tools(
    tmp_path: Path,
) -> None:
    script, env = _fake_pinned_tree(tmp_path)
    command = f'''
source "{script}" >/dev/null
printf '%s\n' "$(command -v nvcc)" "$(command -v protoc)" \
  "$NVCC_CCBIN" "$CXX" "$CUDA_INSTALL_PATH" \
  "$(pkg-config --define-prefix --modversion protobuf)" \
  "$(pkg-config --define-prefix --variable=prefix protobuf)"
'''
    proc = subprocess.run(
        ["bash", "-c", command],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    repo = tmp_path / "repo"
    assert proc.stdout.splitlines() == [
        str(repo / "cuda" / "bin" / "nvcc"),
        str(repo / ".toolchain" / "pb" / "bin" / "protoc"),
        str(repo / "host" / "g++-14"),
        str(repo / "host" / "g++-14"),
        str(repo / "cuda"),
        "34.1.0",
        str(repo / ".toolchain" / "pb"),
    ]


def test_pinned_environment_is_idempotent_with_fake_tools(tmp_path: Path) -> None:
    script, env = _fake_pinned_tree(tmp_path)
    command = f'''
source "{script}" >/dev/null
source "{script}" >/dev/null
printf '%s\n' "$PATH" "$PKG_CONFIG_PATH" "$LD_LIBRARY_PATH" \
  "$NVCC_CCBIN" "$PROTOC" "$CUDA_INSTALL_PATH"
'''
    proc = subprocess.run(
        ["bash", "-c", command], env=env, capture_output=True, text=True, check=True
    )
    path, pkg_path, ld_path, cxx, protoc, cuda = proc.stdout.splitlines()
    repo = tmp_path / "repo"
    assert path.split(":").count(str(repo / "cuda" / "bin")) == 1
    assert path.split(":").count(str(repo / ".toolchain" / "pb" / "bin")) == 1
    assert pkg_path.split(":") == [str(repo / ".toolchain" / "pb" / "lib" / "pkgconfig")]
    assert ld_path.split(":") == [str(repo / ".toolchain" / "pb" / "lib")]
    assert cxx == str(repo / "host" / "g++-14")
    assert protoc == str(repo / ".toolchain" / "pb" / "bin" / "protoc")
    assert cuda == str(repo / "cuda")


@pytest.mark.parametrize(
    ("overrides", "diagnostic"),
    (
        ({"nvcc": "13.3"}, "expected nvcc release 13.2"),
        ({"protoc": "35.1"}, "expected libprotoc 34.1"),
        ({"cxx": "15.2.0"}, "expected g++ major 14"),
        ({"protobuf": "35.1.0"}, "expected protobuf 34.1.0"),
    ),
)
def test_pinned_environment_rejects_version_drift(
    tmp_path: Path, overrides: dict[str, str], diagnostic: str,
) -> None:
    script, env = _fake_pinned_tree(tmp_path, **overrides)
    proc = subprocess.run(
        ["bash", "-c", f'source "{script}"'],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert diagnostic in proc.stderr


def test_active_docs_do_not_restore_broken_build_recipe() -> None:
    active_docs = (
        NVBIT / "README.md",
        NVBIT / "metric_extractor" / "README.md",
        NVBIT / "CROSS_ARCH_RUNBOOK.md",
        REPO / "docs" / "runbook_host.md",
        REPO / "docs" / "nvbit-tracing.md",
        REPO / "workloads" / "portable_lcd" / "README.md",
        REPO / ".claude" / "knowledge" / "measurement" / "nvbit-injection-playbook.md",
    )
    for path in active_docs:
        content = _text(path)
        assert 'INCLUDES="$NVBIT_TRACER_INCLUDES"' not in content, path
        assert "source tools/nvbit_tracer/pinned_build_env.sh" in content, path


def test_active_docs_lock_corrected_compatibility_and_bootstrap_claims(
    tmp_path: Path,
) -> None:
    tracer_readme = _text(NVBIT / "README.md")
    tracing_doc = _text(REPO / "docs" / "nvbit-tracing.md")
    metric_readme = _text(NVBIT / "metric_extractor" / "README.md")
    cross_arch = _text(NVBIT / "CROSS_ARCH_RUNBOOK.md")
    smoke = _text(NVBIT / "smoketest" / "no_op_kernel_idempotency.sh")
    env_script = _text(NVBIT / "pinned_build_env.sh")
    host_smoke = _text(REPO / "tools" / "cloud" / "host_smoke_local_preflight.sh")

    assert "stays **dormant**" not in tracer_readme + tracing_doc
    assert "make -C nvbit_tracer" not in metric_readme
    assert "pip install protobuf`;" not in cross_arch
    assert 'mamba create -y -p "$SUITE_ROOT/.toolchain/pb"' not in env_script
    assert "nvcc -arch=sm_120" not in smoke
    tracer_hash_check = host_smoke.index('actual_nvbit_sha256="$(sha256sum')
    pinned_env_check = host_smoke.index('source "${NVBIT_DIR}/pinned_build_env.sh"')
    binding_check = host_smoke.index("for binding in address cuda_stream")
    assert tracer_hash_check < pinned_env_check < binding_check
    assert "de47b3af6a18ae69287f61ff149dbdb03b4baf6ca8e33f03d2a2221a0482ee5d" in host_smoke
    assert "make -C" not in host_smoke
    assert "nvbit-pb-python" not in host_smoke
    assert "restore the byte-exact" in host_smoke
    assert "canonical rebuild is forbidden" in host_smoke
    active_h3_docs = (
        tracer_readme,
        tracing_doc,
        cross_arch,
        _text(REPO / ".claude" / "knowledge" / "measurement" / "nvbit-injection-playbook.md"),
        _text(REPO / ".claude" / "skills" / "analyzing-sass-nvbit-traces" / "SKILL.md"),
        _text(REPO / "tools" / "local_labels" / "SWEEP_RUNBOOK.md"),
    )
    for content in active_h3_docs:
        assert "NVBIT_TEMPCOPY_BUILD_OK=1" in content
        assert "pinned_build_env.sh --ensure" in content
    for content in (
        tracer_readme,
        tracing_doc,
        _text(REPO / ".claude" / "knowledge" / "measurement" / "nvbit-injection-playbook.md"),
        _text(REPO / ".claude" / "skills" / "analyzing-sass-nvbit-traces" / "SKILL.md"),
    ):
        assert "results/nvbit_capture_staging" in content
        assert 'TRACES_FOLDER="$PWD/out"' not in content
        assert "TRACES_FOLDER=/absolute/output/path" not in content
    assert "policy-quarantined until Stage 7" in tracing_doc
    assert "universal runtime enforcement" in tracing_doc

    verify_lcd = _text(REPO / "tools" / "build" / "verify_lcd_nvbit_trace.sh")
    assert "build it via" not in verify_lcd
    assert "canonical rebuild is forbidden" in verify_lcd
    assert "trace_printer is an ignored host-local artifact, not a tracked file" in verify_lcd

    root_ignore = _text(REPO / ".gitignore")
    metric_patterns = [
        line
        for line in root_ignore.splitlines()
        if line.startswith("/tools/nvbit_tracer/metric_extractor/")
    ]
    assert metric_patterns == [
        "/tools/nvbit_tracer/metric_extractor/metric_extractor.so",
        "/tools/nvbit_tracer/metric_extractor/metric_extractor.o",
        "/tools/nvbit_tracer/metric_extractor/inject_funcs.o",
        "/tools/nvbit_tracer/metric_extractor/test_classifier",
        "/tools/nvbit_tracer/metric_extractor/test_saturating_counter",
        "/tools/nvbit_tracer/metric_extractor/test_tier2_marker_metadata",
        "/tools/nvbit_tracer/metric_extractor/test_tier2_sha256",
        "/tools/nvbit_tracer/metric_extractor/metric_extractor.provenance.json",
    ]

    container_bootstrap = _text(REPO / "tools" / "cloud" / "container_bootstrap.sh")
    assert "protobuf-compiler" not in container_bootstrap
    assert "libprotobuf-dev" not in container_bootstrap
    assert "pinned_build_env.sh --ensure" in container_bootstrap
    assert "verified g++-14; an absent prefix additionally requires" in container_bootstrap
    assert "micromamba, mamba, or" in container_bootstrap
    assert "--out ../../results/nvbit_capture_staging/simbench_bw" in container_bootstrap

    assert "NOOP_SMOKE_CUDA_ROOT+x" in smoke
    assert "${NOOP_SMOKE_CUDA_ROOT}/bin/nvcc" in smoke
    assert "de47b3af6a18ae69287f61ff149dbdb03b4baf6ca8e33f03d2a2221a0482ee5d" in smoke
    assert "make -C tools/nvbit_tracer" not in smoke
    assert "byte-exact committed content" not in smoke
    assert "byte-preserved pinned content" in smoke
    assert "SKIP (advisory nvcc compile" not in smoke

    # A discovered nvcc is advisory only while it is absent. Once present, a
    # compile failure is a validation failure (1), never a successful skip.
    fake_repo = tmp_path / "repo"
    copied_smoke = (
        fake_repo
        / "tools"
        / "nvbit_tracer"
        / "smoketest"
        / "no_op_kernel_idempotency.sh"
    )
    copied_smoke.parent.mkdir(parents=True)
    shutil.copy2(NVBIT / "smoketest" / "no_op_kernel_idempotency.sh", copied_smoke)
    tracer = fake_repo / "tools" / "nvbit_tracer" / "tracer_tool" / "tracer_tool.so"
    tracer.parent.mkdir()
    tracer.write_bytes(b"test-only preserved tracer stand-in")
    (fake_repo / ".toolchain" / "pb" / "lib").mkdir(parents=True)
    fake_python = _executable(fake_repo / ".venv" / "bin" / "python", "exit 0\n")
    fake_bin = tmp_path / "fake-bin"
    _executable(fake_bin / "nvcc", "exit 42\n")
    _executable(
        fake_bin / "sha256sum",
        "printf 'de47b3af6a18ae69287f61ff149dbdb03b4baf6ca8e33f03d2a2221a0482ee5d  %s\\n' \"$1\"\n",
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "NVBIT_SMOKE_ARCH": "sm_86",
            "NVBIT_SMOKE_PYTHON": str(fake_python),
        }
    )
    compile_failure = subprocess.run(
        ["bash", str(copied_smoke)], env=env, capture_output=True, text=True
    )
    assert compile_failure.returncode == 1
    assert "FAIL (present nvcc compile for sm_86 failed" in compile_failure.stderr
    assert "SKIP" not in compile_failure.stderr

    empty_override_env = env | {"NOOP_SMOKE_CUDA_ROOT": ""}
    missing_authoritative = subprocess.run(
        ["bash", str(copied_smoke)],
        env=empty_override_env,
        capture_output=True,
        text=True,
    )
    assert missing_authoritative.returncode == 2
    assert "set but empty" in missing_authoritative.stderr


def test_active_python_injectors_use_fail_closed_runtime_env() -> None:
    injectors = (
        REPO / "workloads" / "simbench_bw" / "run_suite.py",
        REPO / "tools" / "local_labels" / "activation_trace_orchestrator.py",
        REPO / "tools" / "local_labels" / "trace_register_cap_variant_cells.py",
        REPO / "tools" / "local_labels" / "emit_long_timing_feature_artifacts.py",
        REPO / "tools" / "local_labels" / "canary" / "emit_canary_manifest.py",
        REPO / "tools" / "trace_sampling" / "orchestrate.py",
    )
    for path in injectors:
        assert "tracer_runtime_env" in _text(path), path


def test_sm120_bundle_producer_does_not_recapture_admitted_traces() -> None:
    producer = _text(
        REPO / "tools" / "local_labels" / "sm120_activation_bundle_producer.py"
    )
    assert "def _capture_trace(" not in producer
    assert "CUDA_INJECTION64_PATH" not in producer


def test_full_trace_publishers_require_nonvacuous_admission() -> None:
    publishers = (
        REPO / "workloads" / "simbench_bw" / "run_suite.py",
        REPO / "tools" / "local_labels" / "activation_trace_orchestrator.py",
        REPO / "tools" / "local_labels" / "roster_trace_capture.py",
        REPO / "tools" / "local_labels" / "trace_register_cap_variant_cells.py",
        REPO / "tools" / "local_labels" / "canary" / "emit_canary_manifest.py",
        REPO / "tools" / "trace_sampling" / "golden_cells" / "capture_golden_cells.sh",
        REPO / "tools" / "cloud" / "preflight.sh",
    )
    for path in publishers:
        assert "trace_admission" in _text(path), path
