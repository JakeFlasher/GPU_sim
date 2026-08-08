from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import jsonschema
import pytest

from tools.local_labels import immutable_artifact as immutable
from tools.nvbit_tracer import external_tracer_build as build


_REPO = Path(__file__).resolve().parents[2]
_REAL_GIT = Path("/usr/bin/git")
_SCHEMA_ROOT = _REPO / "workloads/simbench_bw/schemas"


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [str(_REAL_GIT), "-C", str(repo), *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _make_writable(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        try:
            if path.is_symlink():
                continue
            path.chmod(0o755 if path.is_dir() else 0o644)
        except FileNotFoundError:
            pass
    root.chmod(0o755)


def _source_files(include: str = '#include "../traces_enhanced/pb_trace/include/trace.pb.h"\n') -> dict[str, bytes]:
    return {
        "tools/nvbit_tracer/Makefile": b"all:\n\t@true\n",
        "tools/nvbit_tracer/isa_def/trace_opcode.h": b"#pragma once\n",
        "tools/nvbit_tracer/metric_extractor/tier2_sha256.h": b"#pragma once\n",
        "tools/nvbit_tracer/nvbit_release/core/nvbit.h": b"#pragma once\n",
        "tools/nvbit_tracer/tracer_tool/Makefile": b"all:\n\t@true\n",
        "tools/nvbit_tracer/tracer_tool/tracer_tool.cu": include.encode("utf-8"),
        "tools/nvbit_tracer/traces_enhanced/Makefile": b"all:\n\t@true\n",
        "tools/nvbit_tracer/traces_enhanced/dynamic_trace/trace.proto": b'syntax = "proto3";\n',
        "tools/nvbit_tracer/traces_enhanced/src/helper.cc": b"int helper() { return 1; }\n",
        "tools/nvbit_tracer/traces_enhanced/pb_trace/include/trace.pb.h": b"#pragma once\n",
        "tools/nvbit_tracer/traces_enhanced/pb_trace/src/trace.pb.cc": b'#include "../include/trace.pb.h"\n',
    }


def _init_repo(repo: Path, *, include: str | None = None) -> None:
    repo.mkdir()
    files = _source_files() if include is None else _source_files(include)
    for relative, payload in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    for schema_name in (
        build.SOURCE_SCHEMA_NAME,
        build.BUILD_SCHEMA_NAME,
        build.ARTIFACT_SCHEMA_NAME,
    ):
        schema_path = repo / "workloads/simbench_bw/schemas" / schema_name
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        schema_path.write_bytes((_SCHEMA_ROOT / schema_name).read_bytes())
    (repo / ".gitignore").write_text(
        "tools/nvbit_tracer/tracer_tool/tracer_tool.so\n"
        "ignored-protobuf/\n",
        encoding="utf-8",
    )
    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.name", "Task 630 Test")
    _run_git(repo, "config", "user.email", "task630@example.invalid")
    _run_git(repo, "add", ".")
    _run_git(repo, "commit", "-qm", "fixture")


def _fake_tracer_elf(
    *,
    cuda_architecture: str = "sm_86",
    machine: int = 62,
) -> bytes:
    text = b"\xc3"
    dynamic = b"\0" * 16
    fatbin = f"FAKE_CUDA_ARCH={cuda_architecture}\0".encode("ascii")
    names = b"\0.shstrtab\0.text\0.dynamic\0.nv_fatbin\0"
    phoff = 64
    phnum = 2
    cursor = phoff + phnum * 56
    text_offset = cursor
    cursor += len(text)
    cursor = (cursor + 7) & ~7
    dynamic_offset = cursor
    cursor += len(dynamic)
    fatbin_offset = cursor
    cursor += len(fatbin)
    names_offset = cursor
    cursor += len(names)
    shoff = (cursor + 7) & ~7
    shnum = 5
    total = shoff + shnum * 64
    payload = bytearray(total)
    ident = b"\x7fELF" + bytes((2, 1, 1, 0)) + b"\0" * 8
    struct.pack_into(
        "<16sHHIQQQIHHHHHH",
        payload,
        0,
        ident,
        3,
        machine,
        1,
        0,
        phoff,
        shoff,
        0,
        64,
        56,
        phnum,
        64,
        shnum,
        1,
    )
    struct.pack_into("<IIQQQQQQ", payload, phoff, 1, 5, 0, 0, 0, total, total, 8)
    struct.pack_into(
        "<IIQQQQQQ",
        payload,
        phoff + 56,
        2,
        6,
        dynamic_offset,
        dynamic_offset,
        dynamic_offset,
        len(dynamic),
        len(dynamic),
        8,
    )
    payload[text_offset : text_offset + len(text)] = text
    payload[dynamic_offset : dynamic_offset + len(dynamic)] = dynamic
    payload[fatbin_offset : fatbin_offset + len(fatbin)] = fatbin
    payload[names_offset : names_offset + len(names)] = names
    section_rows = (
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (names.index(b".shstrtab"), 3, 0, 0, names_offset, len(names), 0, 0, 1, 0),
        (names.index(b".text"), 1, 6, 0, text_offset, len(text), 0, 0, 1, 0),
        (names.index(b".dynamic"), 6, 3, 0, dynamic_offset, len(dynamic), 0, 0, 8, 16),
        (names.index(b".nv_fatbin"), 1, 2, 0, fatbin_offset, len(fatbin), 0, 0, 8, 0),
    )
    for index, row in enumerate(section_rows):
        struct.pack_into("<IIQQQQIIQQ", payload, shoff + index * 64, *row)
    return bytes(payload)


_DEFAULT_TRACER = _fake_tracer_elf()


def _fake_make_source(
    *,
    payload: bytes = _DEFAULT_TRACER,
    stdout: bytes = b"fake build stdout\x00\n",
    stderr: bytes = b"fake build stderr\xff\n",
    body_before_output: str = "",
    body_after_output: str = "",
    returncode: int = 0,
) -> str:
    return f'''#!/usr/bin/python3
import pathlib
import sys
import time
{body_before_output}
args = sys.argv[1:]
root = pathlib.Path(args[args.index("-C") + 1])
out = root / "tracer_tool" / "tracer_tool.so"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_bytes({payload!r})
{body_after_output}
sys.stdout.buffer.write({stdout!r})
sys.stderr.buffer.write({stderr!r})
raise SystemExit({returncode})
'''


class Fixture:
    def __init__(self, root: Path, make_source: str | None = None) -> None:
        self.root = root
        self.repo = root / "repo"
        _init_repo(self.repo)
        self.tools = root / "fake-tools"
        self.tools.mkdir()
        self.protobuf = root / "protobuf-34.1"
        (self.protobuf / "include").mkdir(parents=True)
        (self.protobuf / "include/google-protobuf-fixture.h").write_text(
            "fixture\n", encoding="utf-8"
        )
        (self.protobuf / "lib/pkgconfig").mkdir(parents=True)
        (self.protobuf / "lib/libprotobuf-fixture.so").write_bytes(b"fixture-lib")
        (self.protobuf / "lib/pkgconfig/protobuf.pc").write_text(
            "Version: 34.1.0\n", encoding="utf-8"
        )
        scripts = {
            "bash": "#!/bin/sh\nexit 0\n",
            "nvcc": "#!/bin/sh\nprintf 'Cuda compilation tools, release 13.2, V13.2.0\\n'\n",
            "ptxas": "#!/bin/sh\nprintf 'ptxas release 13.2, V13.2.0\\n'\n",
            "cuobjdump": (
                "#!/usr/bin/python3\n"
                "import pathlib, re, sys\n"
                "if '--version' in sys.argv:\n"
                "    print('cuobjdump release 13.2, V13.2.0')\n"
                "    raise SystemExit(0)\n"
                "payload = pathlib.Path(sys.argv[-1]).read_bytes()\n"
                "match = re.search(rb'FAKE_CUDA_ARCH=(sm_[0-9]+[a-z]?)', payload)\n"
                "if match is None:\n"
                "    raise SystemExit(3)\n"
                "arch = match.group(1).decode('ascii')\n"
                "print(f'ELF file    1: tracer_tool.1.{arch}.cubin')\n"
            ),
            "host-cxx": "#!/bin/sh\nprintf '14.3.1\\n'\n",
            "protoc": "#!/bin/sh\nprintf 'libprotoc 34.1\\n'\n",
            "pkg-config": (
                "#!/bin/sh\n"
                "case \"$*\" in\n"
                "  *--modversion*) printf '34.1.0\\n' ;;\n"
                f"  *--variable=prefix*) printf '%s\\n' {str(self.protobuf)!r} ;;\n"
                "  *) exit 2 ;;\n"
                "esac\n"
            ),
        }
        for role, script in scripts.items():
            _write_executable(self.tools / role, script)
        self.make = self.tools / "make"
        _write_executable(self.make, make_source or _fake_make_source())
        self.output = root / "published"
        self.output.mkdir()
        self.bundle = self.output / "task630-sm86-candidate"
        self.admitted = self.output / "task630-sm86-admitted"
        self.build_roots: list[Path] = []

    def request(self, **changes: object) -> build.BuildRequest:
        build_root = Path("/tmp") / f"task630-test-{uuid.uuid4().hex}"
        self.build_roots.append(build_root)
        request = build.BuildRequest(
            repo_root=self.repo,
            build_root=build_root,
            bundle_dir=self.bundle,
            git=_REAL_GIT,
            bash=self.tools / "bash",
            make=self.make,
            nvcc=self.tools / "nvcc",
            ptxas=self.tools / "ptxas",
            cuobjdump=self.tools / "cuobjdump",
            host_cxx=self.tools / "host-cxx",
            protoc=self.tools / "protoc",
            pkg_config=self.tools / "pkg-config",
            protobuf_prefix=self.protobuf,
            timeout_seconds=5,
            max_output_bytes=1024 * 1024,
        )
        return replace(request, **changes)

    def cleanup(self) -> None:
        _make_writable(self.output)
        for root in self.build_roots:
            if root.exists():
                _make_writable(root)
                shutil.rmtree(root)


@pytest.fixture
def fixture() -> Iterator[Fixture]:
    with tempfile.TemporaryDirectory(prefix="task630-build-test-", dir="/tmp") as raw:
        value = Fixture(Path(raw))
        try:
            yield value
        finally:
            value.cleanup()


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _make_bundle_documents_writable(fixture: Fixture) -> None:
    fixture.bundle.chmod(0o755)
    for name in (
        build.ARTIFACT_IDENTITY_NAME,
        build.BUILD_RECEIPT_NAME,
        build.TRACER_NAME,
    ):
        (fixture.bundle / name).chmod(0o644)


def _rewrite_build_and_artifact(
    fixture: Fixture,
    receipt: dict[str, object],
    artifact: dict[str, object],
) -> None:
    receipt["model_seal_sha256"] = immutable.model_seal(receipt)
    receipt_payload = immutable.pretty_json_bytes(receipt)
    receipt_path = fixture.bundle / build.BUILD_RECEIPT_NAME
    receipt_path.write_bytes(receipt_payload)
    artifact["build_receipt"] = build._file_binding(
        build.BUILD_RECEIPT_NAME, receipt_payload, receipt
    )
    artifact["model_seal_sha256"] = immutable.model_seal(artifact)
    (fixture.bundle / build.ARTIFACT_IDENTITY_NAME).write_bytes(
        immutable.pretty_json_bytes(artifact)
    )


def _admit_fixture(
    fixture: Fixture,
    *,
    expected_sha256: str | None = None,
    timeout_seconds: int = 5,
) -> Path:
    return build.admit_candidate(
        fixture.bundle,
        fixture.admitted,
        repo_root=fixture.repo,
        expected_tracer_sha256=(
            expected_sha256
            if expected_sha256 is not None
            else hashlib.sha256(_DEFAULT_TRACER).hexdigest()
        ),
        cuobjdump=fixture.tools / "cuobjdump",
        timeout_seconds=timeout_seconds,
    )


def _verify_admitted(fixture: Fixture) -> dict[str, object]:
    return build.verify_bundle(
        fixture.admitted,
        repo_root=fixture.repo,
        expected_tracer_sha256=hashlib.sha256(_DEFAULT_TRACER).hexdigest(),
        cuobjdump=fixture.tools / "cuobjdump",
        timeout_seconds=5,
    )


def _verify_retained_documents(
    fixture: Fixture,
    bundle: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> dict[str, object]:
    return build.verify_admitted_bundle_documents(
        (bundle / build.SOURCE_MANIFEST_NAME).read_bytes(),
        (bundle / build.BUILD_RECEIPT_NAME).read_bytes(),
        (bundle / build.ARTIFACT_IDENTITY_NAME).read_bytes(),
        repo_root=fixture.repo,
        expected_tracer_sha256=expected_sha256,
        expected_tracer_size_bytes=expected_size_bytes,
    )


def test_retained_document_replay_accepts_reviewed_bundle(fixture: Fixture) -> None:
    build.build_bundle(fixture.request())
    admitted = _admit_fixture(fixture)
    digest = hashlib.sha256(_DEFAULT_TRACER).hexdigest()

    artifact = _verify_retained_documents(
        fixture,
        admitted,
        expected_sha256=digest,
        expected_size_bytes=len(_DEFAULT_TRACER),
    )

    assert artifact["artifact"]["sha256"] == digest
    assert artifact["artifact"]["size_bytes"] == len(_DEFAULT_TRACER)


def test_retained_document_replay_rejects_candidate_authority(
    fixture: Fixture,
) -> None:
    candidate = build.build_bundle(fixture.request())

    with pytest.raises(
        build.ExternalTracerBuildAuthorityError,
        match="admitted tracer document authority",
    ):
        _verify_retained_documents(
            fixture,
            candidate,
            expected_sha256=hashlib.sha256(_DEFAULT_TRACER).hexdigest(),
            expected_size_bytes=len(_DEFAULT_TRACER),
        )


def test_retained_document_replay_rejects_independent_digest_mismatch(
    fixture: Fixture,
) -> None:
    build.build_bundle(fixture.request())
    admitted = _admit_fixture(fixture)

    with pytest.raises(
        build.ExternalTracerBuildAuthorityError,
        match="admitted tracer document digest",
    ):
        _verify_retained_documents(
            fixture,
            admitted,
            expected_sha256="0" * 64,
            expected_size_bytes=len(_DEFAULT_TRACER),
        )


def test_retained_document_replay_rejects_resealed_source_substitution(
    fixture: Fixture,
) -> None:
    build.build_bundle(fixture.request())
    admitted = _admit_fixture(fixture)
    source = _load(admitted / build.SOURCE_MANIFEST_NAME)
    source["repository"]["head_tree_sha1"] = "0" * 40
    source["model_seal_sha256"] = immutable.model_seal(source)
    source_payload = immutable.pretty_json_bytes(source)

    with pytest.raises(
        build.ExternalTracerBuildAuthorityError,
        match="source manifest file identity",
    ):
        build.verify_admitted_bundle_documents(
            source_payload,
            (admitted / build.BUILD_RECEIPT_NAME).read_bytes(),
            (admitted / build.ARTIFACT_IDENTITY_NAME).read_bytes(),
            repo_root=fixture.repo,
            expected_tracer_sha256=hashlib.sha256(_DEFAULT_TRACER).hexdigest(),
            expected_tracer_size_bytes=len(_DEFAULT_TRACER),
        )


def test_fake_build_publishes_exact_cross_sealed_bundle(fixture: Fixture) -> None:
    published = build.build_bundle(fixture.request())

    assert published == fixture.bundle
    assert sorted(path.name for path in published.iterdir()) == list(build.BUNDLE_FILES)
    candidate_receipt = _load(published / build.BUILD_RECEIPT_NAME)
    assert candidate_receipt["output_authority"] == {
        "kind": build.CANDIDATE_AUTHORITY_KIND,
        "expected_sha256": hashlib.sha256(_DEFAULT_TRACER).hexdigest(),
    }
    admitted = _admit_fixture(fixture)
    assert admitted == fixture.admitted
    assert fixture.bundle.exists()
    artifact = _verify_admitted(fixture)
    source = _load(admitted / build.SOURCE_MANIFEST_NAME)
    receipt = _load(admitted / build.BUILD_RECEIPT_NAME)
    tracer = (admitted / build.TRACER_NAME).read_bytes()

    assert artifact["contract"] == build.CONTRACT
    assert source["contract"] == build.CONTRACT
    assert receipt["contract"] == build.CONTRACT
    assert receipt["process"]["stdout"] == {
        "sha256": hashlib.sha256(b"fake build stdout\x00\n").hexdigest(),
        "size_bytes": len(b"fake build stdout\x00\n"),
    }
    assert receipt["process"]["stderr"] == {
        "sha256": hashlib.sha256(b"fake build stderr\xff\n").hexdigest(),
        "size_bytes": len(b"fake build stderr\xff\n"),
    }
    assert artifact["artifact"]["sha256"] == hashlib.sha256(tracer).hexdigest()
    assert [row["role"] for row in receipt["tools"]] == list(build.TOOL_ROLES)
    assert all("/proc/self/fd/" not in arg for arg in receipt["invocation"]["argv"])
    assert all(
        "/proc/self/fd/" not in row["value"]
        for row in receipt["invocation"]["environment"]
    )
    assert receipt["invocation"]["argv"][1:4] == [
        "-C",
        str(fixture.build_roots[0] / "tools/nvbit_tracer"),
        "all",
    ]
    assert receipt["invocation"]["tool_execution"] == (
        "retained_descriptor_aliases.v1"
    )
    assert len(receipt["invocation"]["argv"]) == 21
    assert len(receipt["invocation"]["environment"]) == 20
    assert [row["name"] for row in receipt["invocation"]["environment"]] == sorted(
        build._build_environment(
            fixture.request(),
            private_bin=fixture.build_roots[0] / ".task630-tool-bin",
            protobuf_prefix=fixture.protobuf,
        )
    )
    assert receipt["output_authority"] == {
        "kind": build.REVIEWED_AUTHORITY_KIND,
        "expected_sha256": hashlib.sha256(_DEFAULT_TRACER).hexdigest(),
    }
    assert receipt["artifact_validation"]["elf"]["machine"] == "EM_X86_64"
    assert receipt["artifact_validation"]["cuda"]["architectures"] == ["sm_86"]
    assert "ARCH=sm_86" in receipt["invocation"]["argv"]
    assert "NVBIT_TEMPCOPY_BUILD_OK=1" in receipt["invocation"]["argv"]
    nvcc_assignment = next(
        value
        for value in receipt["invocation"]["argv"]
        if value.startswith("NVCC=")
    )
    cuda_include = fixture.tools.parent / "include"
    assert f"-I{cuda_include}" in nvcc_assignment
    assert "OBJECTS=tracer_tool.o" in receipt["invocation"]["argv"]
    assert "OBJ=obj/helper.o" in receipt["invocation"]["argv"]
    assert "PROTO_FILES=dynamic_trace/trace.proto" in receipt["invocation"]["argv"]
    assert (
        "TRACES_ENHANCED_OBJ=../traces_enhanced/obj/helper.o"
        in receipt["invocation"]["argv"]
    )
    assert (
        "TRACES_ENHANCED_PROTOCOL_BUFFER_OBJ="
        "../traces_enhanced/pb_trace/obj/trace.pb.o"
        in receipt["invocation"]["argv"]
    )


def test_schemas_are_closed_draft_2020_12() -> None:
    names = (
        build.SOURCE_SCHEMA_NAME,
        build.BUILD_SCHEMA_NAME,
        build.ARTIFACT_SCHEMA_NAME,
    )
    for name in names:
        schema = json.loads((_SCHEMA_ROOT / name).read_text(encoding="utf-8"))
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["additionalProperties"] is False
        jsonschema.Draft202012Validator.check_schema(schema)

    schema = json.loads(
        (_SCHEMA_ROOT / build.SOURCE_SCHEMA_NAME).read_text(encoding="utf-8")
    )
    instance = {
        "schema_version": build.SOURCE_SCHEMA_VERSION,
        "unexpected": True,
    }
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(instance))
    assert any("Additional properties" in error.message for error in errors)

    build_schema = json.loads(
        (_SCHEMA_ROOT / build.BUILD_SCHEMA_NAME).read_text(encoding="utf-8")
    )
    assert build_schema["properties"]["schema_version"]["const"].endswith(".v3")
    invocation = build_schema["properties"]["invocation"]["properties"]
    assert invocation["argv"]["minItems"] == invocation["argv"]["maxItems"] == 21
    assert invocation["argv"]["items"] is False
    assert (
        invocation["environment"]["minItems"]
        == invocation["environment"]["maxItems"]
        == 20
    )
    assert invocation["environment"]["items"] is False
    assert build_schema["properties"]["tools"]["minItems"] == 9
    assert build_schema["properties"]["tools"]["maxItems"] == 9


@pytest.mark.parametrize("dirty_kind", ["tracked", "staged", "untracked"])
def test_dirty_checkout_fails_before_make(
    fixture: Fixture, dirty_kind: str
) -> None:
    marker = fixture.root / "make-ran"
    _write_executable(
        fixture.make,
        _fake_make_source(body_before_output=f"pathlib.Path({str(marker)!r}).write_text('ran')"),
    )
    source = fixture.repo / "tools/nvbit_tracer/tracer_tool/tracer_tool.cu"
    if dirty_kind == "tracked":
        source.write_text("dirty\n", encoding="utf-8")
    elif dirty_kind == "staged":
        source.write_text("staged\n", encoding="utf-8")
        _run_git(fixture.repo, "add", str(source.relative_to(fixture.repo)))
    else:
        (fixture.repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(build.ExternalTracerBuildAuthorityError, match="checkout is dirty"):
        build.build_bundle(fixture.request())
    assert not marker.exists()
    assert not fixture.build_roots[0].exists()


def test_head_drift_during_build_is_rejected(fixture: Fixture) -> None:
    command = (
        f"subprocess.run([{str(_REAL_GIT)!r}, '-C', {str(fixture.repo)!r}, "
        "'-c', 'user.name=Drift', '-c', 'user.email=drift@example.invalid', "
        "'commit', '--allow-empty', '-qm', 'drift'], check=True)"
    )
    source = _fake_make_source(body_after_output="import subprocess\n" + command)
    _write_executable(fixture.make, source)

    with pytest.raises(build.ExternalTracerBuildAuthorityError, match="HEAD drifted"):
        build.build_bundle(fixture.request())
    assert not fixture.bundle.exists()


def test_tool_drift_during_build_is_rejected(fixture: Fixture) -> None:
    victim = fixture.tools / "ptxas"
    source = _fake_make_source(
        body_after_output=f"pathlib.Path({str(victim)!r}).write_text('changed')"
    )
    _write_executable(fixture.make, source)

    with pytest.raises(build.ExternalTracerBuildAuthorityError, match="ptxas tool drifted"):
        build.build_bundle(fixture.request())
    assert not fixture.bundle.exists()


def test_child_tool_alias_consumes_retained_descriptor_after_path_swap(
    fixture: Fixture,
) -> None:
    victim = fixture.tools / "ptxas"
    malicious_marker = fixture.root / "malicious-ptxas-ran"
    observed = fixture.root / "observed-ptxas-version"
    replacement = fixture.root / "replacement-ptxas"
    _write_executable(
        replacement,
        "#!/bin/sh\n"
        f"printf ran > {str(malicious_marker)!r}\n"
        "printf 'malicious ptxas\\n'\n",
    )
    body = (
        "import os, subprocess\n"
        "ptxas_alias = next(value.split('=', 1)[1] for value in args "
        "if value.startswith('PTXAS='))\n"
        f"os.replace({str(replacement)!r}, {str(victim)!r})\n"
        "probe = subprocess.run([ptxas_alias, '--version'], check=True, "
        "stdout=subprocess.PIPE, close_fds=False)\n"
        f"pathlib.Path({str(observed)!r}).write_bytes(probe.stdout)"
    )
    _write_executable(
        fixture.make, _fake_make_source(body_after_output=body)
    )

    with pytest.raises(
        build.ExternalTracerBuildAuthorityError,
        match="ptxas tool drifted",
    ):
        build.build_bundle(fixture.request())
    assert b"release 13.2" in observed.read_bytes()
    assert not malicious_marker.exists()
    assert not fixture.bundle.exists()


def test_top_level_make_exec_uses_retained_descriptor_under_path_swap(
    fixture: Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malicious_marker = fixture.root / "malicious-make-ran"
    replacement = fixture.root / "replacement-make"
    _write_executable(
        replacement,
        "#!/usr/bin/python3\n"
        "import pathlib\n"
        f"pathlib.Path({str(malicious_marker)!r}).write_text('ran')\n",
    )
    original_run = build._run_bounded_process
    swapped = False

    def intercept(argv: object, **kwargs: object):
        nonlocal swapped
        values = tuple(argv)
        if len(values) >= 4 and values[1:4] == (
            "-C",
            str(fixture.build_roots[0] / "tools/nvbit_tracer"),
            "all",
        ):
            swapped = True
            os.replace(replacement, fixture.make)
            assert str(kwargs["executable"]).startswith("/proc/self/fd/")
        return original_run(values, **kwargs)

    monkeypatch.setattr(build, "_run_bounded_process", intercept)
    with pytest.raises(
        build.ExternalTracerBuildAuthorityError,
        match="make tool drifted|retained make tool drifted",
    ):
        build.build_bundle(fixture.request())
    assert swapped
    assert not malicious_marker.exists()
    assert not fixture.bundle.exists()


def test_make_expansion_syntax_in_external_value_is_rejected_before_make(
    fixture: Fixture,
) -> None:
    marker = fixture.root / "injected"
    bad_prefix = fixture.root / "protobuf-$(touch-injected)"
    fixture.protobuf.rename(bad_prefix)
    make_marker = fixture.root / "make-ran"
    _write_executable(
        fixture.make,
        _fake_make_source(
            body_before_output=f"pathlib.Path({str(make_marker)!r}).write_text('ran')"
        ),
    )

    with pytest.raises(
        build.ExternalTracerBuildAuthorityError,
        match="unsafe for exact GNU Make expansion",
    ):
        build.build_bundle(fixture.request(protobuf_prefix=bad_prefix))
    assert not marker.exists()
    assert not make_marker.exists()
    assert not fixture.build_roots[0].exists()


def test_extracted_source_drift_during_build_is_rejected(fixture: Fixture) -> None:
    relative = "tracer_tool/tracer_tool.cu"
    source = _fake_make_source(
        body_after_output=f"(root / {relative!r}).write_text('drifted')"
    )
    _write_executable(fixture.make, source)

    with pytest.raises(build.ExternalTracerBuildAuthorityError, match="tracked source drifted"):
        build.build_bundle(fixture.request())
    assert not fixture.bundle.exists()


def test_transient_extracted_source_modify_restore_is_rejected(
    fixture: Fixture,
) -> None:
    relative = "tracer_tool/tracer_tool.cu"
    body = (
        f"victim = root / {relative!r}\n"
        "original = victim.read_bytes()\n"
        "victim.write_bytes(b'transient source replacement')\n"
        "victim.write_bytes(original)"
    )
    _write_executable(
        fixture.make, _fake_make_source(body_after_output=body)
    )

    with pytest.raises(
        build.ExternalTracerBuildAuthorityError,
        match="extracted tracked source drifted",
    ):
        build.build_bundle(fixture.request())
    assert not fixture.bundle.exists()


def test_transient_protobuf_modify_restore_is_rejected(
    fixture: Fixture,
) -> None:
    victim = fixture.protobuf / "lib/libprotobuf-fixture.so"
    body = (
        f"victim = pathlib.Path({str(victim)!r})\n"
        "original = victim.read_bytes()\n"
        "victim.write_bytes(b'transient protobuf replacement')\n"
        "victim.write_bytes(original)"
    )
    _write_executable(
        fixture.make, _fake_make_source(body_after_output=body)
    )

    with pytest.raises(
        build.ExternalTracerBuildAuthorityError,
        match="pinned protobuf prefix drifted",
    ):
        build.build_bundle(fixture.request())
    assert not fixture.bundle.exists()


def test_checkout_source_drift_during_build_is_rejected(fixture: Fixture) -> None:
    victim = fixture.repo / "tools/nvbit_tracer/tracer_tool/tracer_tool.cu"
    source = _fake_make_source(
        body_after_output=f"pathlib.Path({str(victim)!r}).write_text('checkout drift')"
    )
    _write_executable(fixture.make, source)

    with pytest.raises(build.ExternalTracerBuildAuthorityError, match="checkout is dirty"):
        build.build_bundle(fixture.request())
    assert not fixture.bundle.exists()


def test_generated_protobuf_drift_is_typed_separately(fixture: Fixture) -> None:
    relative = "traces_enhanced/pb_trace/include/trace.pb.h"
    source = _fake_make_source(
        body_after_output=f"(root / {relative!r}).write_text('regenerated drift')"
    )
    _write_executable(fixture.make, source)

    with pytest.raises(build.ExternalTracerBuildAuthorityError, match="generated protobuf drifted"):
        build.build_bundle(fixture.request())


def test_pinned_protobuf_runtime_drift_is_rejected(fixture: Fixture) -> None:
    victim = fixture.protobuf / "lib/libprotobuf-fixture.so"
    source = _fake_make_source(
        body_after_output=f"pathlib.Path({str(victim)!r}).write_bytes(b'drifted-lib')"
    )
    _write_executable(fixture.make, source)

    with pytest.raises(build.ExternalTracerBuildAuthorityError, match="protobuf prefix drifted"):
        build.build_bundle(fixture.request())
    assert not fixture.bundle.exists()


def test_invalid_toolchain_pin_fails_before_make(fixture: Fixture) -> None:
    _write_executable(fixture.tools / "nvcc", "#!/bin/sh\nprintf 'release 13.3\\n'\n")
    marker = fixture.root / "make-ran"
    _write_executable(
        fixture.make,
        _fake_make_source(
            body_before_output=f"pathlib.Path({str(marker)!r}).write_text('ran')"
        ),
    )

    with pytest.raises(build.ExternalTracerBuildAuthorityError, match="release 13.2"):
        build.build_bundle(fixture.request())
    assert not marker.exists()


def test_non_elf_output_cannot_claim_sm86(fixture: Fixture) -> None:
    payload = b"not-elf-but-claims-FAKE_CUDA_ARCH=sm_86" + b"x" * 64
    _write_executable(fixture.make, _fake_make_source(payload=payload))
    request = fixture.request()

    with pytest.raises(build.ExternalTracerBuildAuthorityError, match="not ELF"):
        build.build_bundle(request)
    assert not fixture.bundle.exists()


def test_wrong_host_elf_architecture_is_rejected(fixture: Fixture) -> None:
    payload = _fake_tracer_elf(machine=183)
    _write_executable(fixture.make, _fake_make_source(payload=payload))
    request = fixture.request()

    with pytest.raises(
        build.ExternalTracerBuildAuthorityError,
        match="EM_X86_64",
    ):
        build.build_bundle(request)
    assert not fixture.bundle.exists()


def test_wrong_cuda_architecture_is_rejected(fixture: Fixture) -> None:
    payload = _fake_tracer_elf(cuda_architecture="sm_75")
    _write_executable(fixture.make, _fake_make_source(payload=payload))
    request = fixture.request()

    with pytest.raises(
        build.ExternalTracerBuildAuthorityError,
        match="CUDA architectures violate exact contract",
    ):
        build.build_bundle(request)
    assert not fixture.bundle.exists()


def test_independent_review_digest_mismatch_prevents_admission(
    fixture: Fixture,
) -> None:
    build.build_bundle(fixture.request())
    with pytest.raises(
        build.ExternalTracerBuildAuthorityError,
        match="independent caller tracer digest",
    ):
        _admit_fixture(fixture, expected_sha256="0" * 64)
    assert fixture.bundle.exists()
    assert not fixture.admitted.exists()


def test_admission_collision_preserves_candidate_and_existing_bytes(
    fixture: Fixture,
) -> None:
    build.build_bundle(fixture.request())
    fixture.admitted.mkdir()
    sentinel = fixture.admitted / "do-not-replace"
    sentinel.write_bytes(b"existing-admitted-generation")

    with pytest.raises(build.ExternalTracerBuildAuthorityError, match="already exists"):
        _admit_fixture(fixture)

    assert fixture.bundle.exists()
    assert sentinel.read_bytes() == b"existing-admitted-generation"
    assert tuple(fixture.admitted.iterdir()) == (sentinel,)


@pytest.mark.parametrize(
    ("relative", "pattern"),
    [
        ("tools/nvbit_tracer/tracer_tool/injected.cu", "wildcard CUDA"),
        (
            "tools/nvbit_tracer/traces_enhanced/src/injected.cc",
            "find-based C\\+\\+",
        ),
    ],
)
def test_unmanifested_discovered_source_is_rejected_before_make(
    fixture: Fixture,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
    pattern: str,
) -> None:
    make_marker = fixture.root / "make-ran"
    _write_executable(
        fixture.make,
        _fake_make_source(
            body_before_output=f"pathlib.Path({str(make_marker)!r}).write_text('ran')"
        ),
    )
    original = build._write_extracted_sources

    def inject(root: Path, snapshot: build.SourceSnapshot) -> None:
        original(root, snapshot)
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("int injected;\n", encoding="utf-8")

    monkeypatch.setattr(build, "_write_extracted_sources", inject)
    with pytest.raises(build.ExternalTracerBuildAuthorityError, match=pattern):
        build.build_bundle(fixture.request())
    assert not make_marker.exists()
    assert not fixture.bundle.exists()


def test_end_to_end_deadline_covers_authority_inventory(
    fixture: Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = build._snapshot_protobuf

    def slow_snapshot(prefix: Path) -> build.ProtobufSnapshot:
        time.sleep(3)
        return original(prefix)

    monkeypatch.setattr(build, "_snapshot_protobuf", slow_snapshot)
    started = time.monotonic()
    with pytest.raises(
        build.ExternalTracerBuildProcessError,
        match="end-to-end timeout exceeded",
    ):
        build.build_bundle(fixture.request(timeout_seconds=1))
    assert time.monotonic() - started < 1.5
    assert not fixture.bundle.exists()


def test_end_to_end_deadline_interrupts_and_rolls_back_publication(
    fixture: Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = immutable.publish_directory

    def slow_publish(*args: object, **kwargs: object):
        time.sleep(2)
        return original(*args, **kwargs)

    monkeypatch.setattr(immutable, "publish_directory", slow_publish)
    started = time.monotonic()
    with pytest.raises(
        build.ExternalTracerBuildProcessError,
        match="external candidate build",
    ):
        build.build_bundle(fixture.request(timeout_seconds=1))
    assert time.monotonic() - started < 2
    assert not fixture.bundle.exists()


def test_candidate_is_not_directly_verifiable_as_admitted(
    fixture: Fixture,
) -> None:
    build.build_bundle(fixture.request())
    with pytest.raises(
        build.ExternalTracerBuildAuthorityError,
        match="output authority phase mismatch",
    ):
        build.verify_bundle(
            fixture.bundle,
            repo_root=fixture.repo,
            expected_tracer_sha256=hashlib.sha256(_DEFAULT_TRACER).hexdigest(),
            cuobjdump=fixture.tools / "cuobjdump",
            timeout_seconds=5,
        )


def test_local_include_escape_is_rejected_before_extraction(fixture: Fixture) -> None:
    escaped = '#include "../../../../outside.h"\n'
    source_path = fixture.repo / "tools/nvbit_tracer/tracer_tool/tracer_tool.cu"
    source_path.write_text(escaped, encoding="utf-8")
    _run_git(fixture.repo, "add", str(source_path.relative_to(fixture.repo)))
    _run_git(fixture.repo, "commit", "-qm", "include escape")

    with pytest.raises(build.ExternalTracerBuildAuthorityError, match="include escapes"):
        build.build_bundle(fixture.request())
    assert not fixture.build_roots[0].exists()


def test_ignored_repository_tracer_is_never_opened(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    ignored = fixture.repo / "tools/nvbit_tracer/tracer_tool/tracer_tool.so"
    ignored.write_bytes(b"must never be consumed")
    original_open = os.open
    original_lstat = os.lstat
    original_stat = os.stat

    def is_ignored(path: object) -> bool:
        return isinstance(path, (str, bytes, os.PathLike)) and Path(path) == ignored

    def guarded_open(path: object, *args: object, **kwargs: object) -> int:
        assert not is_ignored(path)
        return original_open(path, *args, **kwargs)

    def guarded_lstat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        assert not is_ignored(path)
        return original_lstat(path, *args, **kwargs)

    def guarded_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        assert not is_ignored(path)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", guarded_open)
    monkeypatch.setattr(os, "lstat", guarded_lstat)
    monkeypatch.setattr(os, "stat", guarded_stat)
    build.build_bundle(fixture.request())
    assert ignored.read_bytes() == b"must never be consumed"


def test_repo_contained_tool_path_has_zero_identity_access(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    ignored = fixture.repo / "tools/nvbit_tracer/tracer_tool/tracer_tool.so"
    ignored.write_bytes(b"repository ignored tracer tool alias")
    ignored.chmod(0o755)
    marker = fixture.root / "make-ran"
    _write_executable(
        fixture.make,
        _fake_make_source(
            body_before_output=f"pathlib.Path({str(marker)!r}).write_text('ran')"
        ),
    )
    original_open = os.open
    original_lstat = os.lstat
    original_stat = os.stat

    def is_ignored(path: object) -> bool:
        return isinstance(path, (str, bytes, os.PathLike)) and Path(path) == ignored

    def guarded_open(path: object, *args: object, **kwargs: object) -> int:
        assert not is_ignored(path)
        return original_open(path, *args, **kwargs)

    def guarded_lstat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        assert not is_ignored(path)
        return original_lstat(path, *args, **kwargs)

    def guarded_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        assert not is_ignored(path)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", guarded_open)
    monkeypatch.setattr(os, "lstat", guarded_lstat)
    monkeypatch.setattr(os, "stat", guarded_stat)
    with pytest.raises(build.ExternalTracerBuildAuthorityError, match="outside the repository"):
        build.build_bundle(fixture.request(nvcc=ignored))
    assert not marker.exists()


def test_repo_contained_protobuf_prefix_has_zero_member_access(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = fixture.repo / "ignored-protobuf"
    member = prefix / "include/repository-member.h"
    member.parent.mkdir(parents=True)
    member.write_text("must not be read\n", encoding="utf-8")
    (prefix / "lib").mkdir()
    marker = fixture.root / "make-ran"
    _write_executable(
        fixture.make,
        _fake_make_source(
            body_before_output=f"pathlib.Path({str(marker)!r}).write_text('ran')"
        ),
    )
    forbidden = {prefix, member, prefix / "include", prefix / "lib"}
    original_open = os.open
    original_lstat = os.lstat
    original_stat = os.stat

    def is_forbidden(path: object) -> bool:
        return isinstance(path, (str, bytes, os.PathLike)) and Path(path) in forbidden

    def guarded_open(path: object, *args: object, **kwargs: object) -> int:
        assert not is_forbidden(path)
        return original_open(path, *args, **kwargs)

    def guarded_lstat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        assert not is_forbidden(path)
        return original_lstat(path, *args, **kwargs)

    def guarded_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        assert not is_forbidden(path)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", guarded_open)
    monkeypatch.setattr(os, "lstat", guarded_lstat)
    monkeypatch.setattr(os, "stat", guarded_stat)
    with pytest.raises(build.ExternalTracerBuildAuthorityError, match="outside the repository"):
        build.build_bundle(fixture.request(protobuf_prefix=prefix))
    assert not marker.exists()


def test_external_hardlink_tool_alias_is_rejected_before_hash(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    ignored = fixture.repo / "tools/nvbit_tracer/tracer_tool/tracer_tool.so"
    ignored.write_bytes(b"hardlink identity")
    ignored.chmod(0o755)
    alias = fixture.root / "external-hardlink-nvcc"
    os.link(ignored, alias)
    original = immutable.open_retained_executable

    def guarded(path: Path, *args: object, **kwargs: object):
        assert Path(path) != alias
        return original(path, *args, **kwargs)

    monkeypatch.setattr(immutable, "open_retained_executable", guarded)
    with pytest.raises(build.ExternalTracerBuildAuthorityError, match="hard-linked alias"):
        build.build_bundle(fixture.request(nvcc=alias))


def test_repo_hardlink_protobuf_member_is_rejected_before_hash(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    ignored = fixture.repo / "tools/nvbit_tracer/tracer_tool/tracer_tool.so"
    ignored.write_bytes(b"protobuf hardlink identity")
    alias = fixture.protobuf / "lib/libprotobuf-fixture.so"
    alias.unlink()
    os.link(ignored, alias)
    original = immutable.open_retained_regular

    def guarded(path: Path, *args: object, **kwargs: object):
        assert Path(path) != alias
        return original(path, *args, **kwargs)

    monkeypatch.setattr(immutable, "open_retained_regular", guarded)
    with pytest.raises(build.ExternalTracerBuildAuthorityError, match="hard-linked alias"):
        build.build_bundle(fixture.request())


def test_protobuf_symlink_to_ignored_tracer_never_accesses_target(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = fixture.repo / "tools/nvbit_tracer/tracer_tool/tracer_tool.so"
    target.write_bytes(b"forbidden protobuf symlink target")
    link = fixture.protobuf / "include/ignored-tracer-link.h"
    link.symlink_to(target)
    original_open = os.open
    original_lstat = os.lstat
    original_stat = os.stat
    original_retained = immutable.open_retained_regular

    def is_target(path: object) -> bool:
        return isinstance(path, (str, bytes, os.PathLike)) and Path(path) == target

    def guarded_open(path: object, *args: object, **kwargs: object) -> int:
        assert not is_target(path)
        return original_open(path, *args, **kwargs)

    def guarded_lstat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        assert not is_target(path)
        return original_lstat(path, *args, **kwargs)

    def guarded_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        assert not is_target(path)
        return original_stat(path, *args, **kwargs)

    def guarded_retained(path: Path, *args: object, **kwargs: object):
        assert Path(path) != target
        return original_retained(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", guarded_open)
    monkeypatch.setattr(os, "lstat", guarded_lstat)
    monkeypatch.setattr(os, "stat", guarded_stat)
    monkeypatch.setattr(immutable, "open_retained_regular", guarded_retained)
    with pytest.raises(build.ExternalTracerBuildAuthorityError, match="symlink member"):
        build.build_bundle(fixture.request())
    assert link.is_symlink()


def test_protobuf_symlink_to_git_repo_never_traverses_alias(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = fixture.root / "other-git-repository"
    _init_repo(target)
    link = fixture.protobuf / "lib/repository-alias"
    link.symlink_to(target, target_is_directory=True)
    original_open = os.open
    original_lstat = os.lstat
    original_stat = os.stat
    original_retained = immutable.open_retained_regular

    def is_target(path: object) -> bool:
        return isinstance(path, (str, bytes, os.PathLike)) and Path(path) == target

    def guarded_open(path: object, *args: object, **kwargs: object) -> int:
        assert not is_target(path)
        return original_open(path, *args, **kwargs)

    def guarded_lstat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        assert not is_target(path)
        return original_lstat(path, *args, **kwargs)

    def guarded_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        assert not is_target(path)
        return original_stat(path, *args, **kwargs)

    def guarded_retained(path: Path, *args: object, **kwargs: object):
        assert Path(path) != target
        return original_retained(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", guarded_open)
    monkeypatch.setattr(os, "lstat", guarded_lstat)
    monkeypatch.setattr(os, "stat", guarded_stat)
    monkeypatch.setattr(immutable, "open_retained_regular", guarded_retained)
    with pytest.raises(build.ExternalTracerBuildAuthorityError, match="symlink member"):
        build.build_bundle(fixture.request())
    assert link.is_symlink()


def test_protobuf_enumerator_never_uses_following_classification() -> None:
    source = (
        _REPO / "tools/nvbit_tracer/external_tracer_build.py"
    ).read_text(encoding="utf-8")
    body = source.split("def _snapshot_protobuf", 1)[1].split(
        "def _reverify_protobuf", 1
    )[0]
    assert "os.walk" not in body
    assert ".is_dir(" not in body
    assert "entry.stat(follow_symlinks=False)" in body
    assert "O_NOFOLLOW" in body
    assert "dir_fd=directory_descriptor" in body


def test_postcommit_verification_failure_retires_active_bundle(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = build._verify_bundle_with_inspector
    calls = 0

    def fail_first(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise build.ExternalTracerBuildAuthorityError(
                "injected final authority failure"
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(build, "_verify_bundle_with_inspector", fail_first)
    with pytest.raises(build.ExternalTracerBuildAuthorityError, match="final authority"):
        build.build_bundle(fixture.request())
    assert calls == 1
    assert not fixture.bundle.exists()
    assert any(
        path.name.startswith(".failed-directory-")
        for path in fixture.output.iterdir()
    )


def test_postcommit_capability_close_failure_retires_active_bundle(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = fixture.request()
    output = request.build_root / "tools/nvbit_tracer/tracer_tool/tracer_tool.so"
    original = immutable.RetainedRegular.close
    injected = False

    def fail_output_close(retained: immutable.RetainedRegular) -> None:
        nonlocal injected
        if retained.path == output and not injected:
            injected = True
            original(retained)
            raise OSError("injected retained output close failure")
        original(retained)

    monkeypatch.setattr(immutable.RetainedRegular, "close", fail_output_close)
    with pytest.raises(
        build.ExternalTracerBuildAuthorityError,
        match="retained output close",
    ):
        build.build_bundle(request)
    assert injected
    assert not fixture.bundle.exists()
    assert any(
        path.name.startswith(".failed-directory-")
        for path in fixture.output.iterdir()
    )


def test_preexisting_bundle_collision_refuses_before_make(fixture: Fixture) -> None:
    fixture.bundle.mkdir()
    sentinel = fixture.bundle / "sentinel"
    sentinel.write_bytes(b"preserve")
    marker = fixture.root / "make-ran"
    _write_executable(
        fixture.make,
        _fake_make_source(body_before_output=f"pathlib.Path({str(marker)!r}).write_text('ran')"),
    )

    with pytest.raises(build.ExternalTracerBuildAuthorityError, match="already exists"):
        build.build_bundle(fixture.request())
    assert sentinel.read_bytes() == b"preserve"
    assert not marker.exists()


def test_process_timeout_is_exit_one_and_publishes_nothing(fixture: Fixture) -> None:
    source = _fake_make_source(body_before_output="time.sleep(2)")
    _write_executable(fixture.make, source)
    request = fixture.request(timeout_seconds=1)

    assert build.main(
        [
            "build-candidate",
            "--repo-root", str(request.repo_root),
            "--build-root", str(request.build_root),
            "--bundle-dir", str(request.bundle_dir),
            "--protobuf-prefix", str(request.protobuf_prefix),
            "--git", str(request.git),
            "--bash", str(request.bash),
            "--make", str(request.make),
            "--nvcc", str(request.nvcc),
            "--ptxas", str(request.ptxas),
            "--cuobjdump", str(request.cuobjdump),
            "--host-cxx", str(request.host_cxx),
            "--protoc", str(request.protoc),
            "--pkg-config", str(request.pkg_config),
            "--timeout-seconds", "1",
        ]
    ) == 1
    assert not fixture.bundle.exists()


def test_closed_pipes_do_not_disable_process_timeout(tmp_path: Path) -> None:
    started = time.monotonic()
    with pytest.raises(build.ExternalTracerBuildProcessError, match="timeout"):
        build._run_bounded_process(
            (
                sys.executable,
                "-c",
                "import os,time; os.close(1); os.close(2); time.sleep(10)",
            ),
            cwd=tmp_path,
            environment={},
            timeout_seconds=1,
            max_output_bytes=1024,
            process_failure=True,
        )

    assert time.monotonic() - started < 2


def test_closed_pipes_process_exit_returns_without_waiting_for_deadline(
    tmp_path: Path,
) -> None:
    started = time.monotonic()
    result = build._run_bounded_process(
        (
            sys.executable,
            "-c",
            "import os; os.close(1); os.close(2)",
        ),
        cwd=tmp_path,
        environment={},
        timeout_seconds=5,
        max_output_bytes=1024,
        process_failure=True,
    )

    assert result == (0, b"", b"")
    assert time.monotonic() - started < 1


def test_process_deadline_starts_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_popen = build.subprocess.Popen

    def delayed_launch(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        time.sleep(0.8)
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(build.subprocess, "Popen", delayed_launch)
    started = time.monotonic()
    with pytest.raises(build.ExternalTracerBuildProcessError, match="timeout"):
        build._run_bounded_process(
            (sys.executable, "-c", "import time; time.sleep(30)"),
            cwd=tmp_path,
            environment={},
            timeout_seconds=1,
            max_output_bytes=1024,
            process_failure=True,
        )

    assert time.monotonic() - started < 1.5


def test_keyboard_interrupt_unconditionally_reaps_running_leader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid_path = tmp_path / "interrupted.pid"
    original_select = build.selectors.DefaultSelector.select
    interrupted = False

    def interrupt_after_launch(
        selector: object,
        timeout: float | None = None,
    ) -> list[tuple[object, int]]:
        nonlocal interrupted
        if not interrupted:
            expires = time.monotonic() + 1
            while not pid_path.exists() and time.monotonic() < expires:
                time.sleep(0.005)
            if pid_path.exists():
                interrupted = True
                raise KeyboardInterrupt
        return original_select(selector, timeout)

    monkeypatch.setattr(
        build.selectors.DefaultSelector,
        "select",
        interrupt_after_launch,
    )
    source = (
        "import os,time; "
        f"open({str(pid_path)!r},'w').write(str(os.getpid())); "
        "time.sleep(30)"
    )

    with pytest.raises(KeyboardInterrupt):
        build._run_bounded_process(
            (sys.executable, "-c", source),
            cwd=tmp_path,
            environment={},
            timeout_seconds=5,
            max_output_bytes=1024,
            process_failure=True,
        )

    pid = int(pid_path.read_text(encoding="ascii"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)


def test_sigterm_cancellation_reaps_running_leader(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "sigterm-child.pid"
    child_source = (
        "import os,time; "
        f"open({str(child_pid_path)!r},'w').write(str(os.getpid())); "
        "time.sleep(30)"
    )
    wrapper_source = (
        "import sys; "
        "from pathlib import Path; "
        "from tools.nvbit_tracer import external_tracer_build as build; "
        "build._run_bounded_process("
        f"(sys.executable,'-c',{child_source!r}),"
        f"cwd=Path({str(tmp_path)!r}),environment={{}},"
        "timeout_seconds=30,max_output_bytes=1024,process_failure=True)"
    )
    wrapper = subprocess.Popen(
        [sys.executable, "-c", wrapper_source],
        cwd=_REPO,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    expires = time.monotonic() + 5
    while not child_pid_path.exists() and time.monotonic() < expires:
        if wrapper.poll() is not None:
            break
        time.sleep(0.01)
    assert child_pid_path.exists(), wrapper.communicate(timeout=1)[1]
    child_pid = int(child_pid_path.read_text(encoding="ascii"))

    wrapper.send_signal(signal.SIGTERM)
    _stdout, stderr = wrapper.communicate(timeout=5)

    assert wrapper.returncode != 0
    assert "interrupted by signal" in stderr
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_timeout_contains_and_reaps_setsid_descendant(tmp_path: Path) -> None:
    descendant_pid_path = tmp_path / "setsid-descendant.pid"
    descendant_source = (
        "import os,time; os.setsid(); "
        f"open({str(descendant_pid_path)!r},'w').write(str(os.getpid())); "
        "os.close(1); os.close(2); time.sleep(30)"
    )
    leader_source = (
        "import os,subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{descendant_source!r}]); "
        "os.close(1); os.close(2); time.sleep(30)"
    )
    started = time.monotonic()

    with pytest.raises(build.ExternalTracerBuildProcessError, match="timeout"):
        build._run_bounded_process(
            (sys.executable, "-c", leader_source),
            cwd=tmp_path,
            environment={},
            timeout_seconds=1,
            max_output_bytes=1024,
            process_failure=True,
        )

    assert time.monotonic() - started < 2
    descendant_pid = int(descendant_pid_path.read_text(encoding="ascii"))
    with pytest.raises(ProcessLookupError):
        os.kill(descendant_pid, 0)
    with pytest.raises(ChildProcessError):
        os.waitpid(descendant_pid, os.WNOHANG)


def test_process_output_bound_is_exit_one(fixture: Fixture) -> None:
    _write_executable(
        fixture.make,
        _fake_make_source(stdout=b"x" * 4096),
    )
    request = fixture.request(max_output_bytes=128)
    with pytest.raises(build.ExternalTracerBuildProcessError, match="output limit"):
        build.build_bundle(request)
    assert not fixture.bundle.exists()


def test_nonzero_build_is_process_failure(fixture: Fixture) -> None:
    _write_executable(fixture.make, _fake_make_source(returncode=9))
    with pytest.raises(build.ExternalTracerBuildProcessError, match="exited 9"):
        build.build_bundle(fixture.request())
    assert not fixture.bundle.exists()


def test_replaced_tracer_and_fully_resealed_bundle_needs_independent_digest(
    fixture: Fixture,
) -> None:
    build.build_bundle(fixture.request())
    _make_bundle_documents_writable(fixture)
    replacement = _fake_tracer_elf() + b"independent-replacement"
    replacement_sha256 = hashlib.sha256(replacement).hexdigest()
    tracer_path = fixture.bundle / build.TRACER_NAME
    tracer_path.write_bytes(replacement)
    receipt = _load(fixture.bundle / build.BUILD_RECEIPT_NAME)
    artifact = _load(fixture.bundle / build.ARTIFACT_IDENTITY_NAME)
    receipt["output_authority"]["expected_sha256"] = replacement_sha256
    artifact["artifact"]["sha256"] = replacement_sha256
    artifact["artifact"]["size_bytes"] = len(replacement)
    _rewrite_build_and_artifact(fixture, receipt, artifact)

    with pytest.raises(
        build.ExternalTracerBuildAuthorityError,
        match="independent caller tracer digest",
    ):
        _admit_fixture(fixture)


@pytest.mark.parametrize("surface", ["argv", "environment"])
def test_resealed_invocation_supersets_are_rejected(
    fixture: Fixture,
    surface: str,
) -> None:
    build.build_bundle(fixture.request())
    _make_bundle_documents_writable(fixture)
    receipt = _load(fixture.bundle / build.BUILD_RECEIPT_NAME)
    artifact = _load(fixture.bundle / build.ARTIFACT_IDENTITY_NAME)
    if surface == "argv":
        receipt["invocation"]["argv"].append("HARMLESS_SUPERSET=1")
    else:
        receipt["invocation"]["environment"].append(
            {"name": "ZZZ_HARMLESS_SUPERSET", "value": "1"}
        )
    _rewrite_build_and_artifact(fixture, receipt, artifact)

    with pytest.raises(
        build.ExternalTracerBuildAuthorityError,
        match="does not satisfy|exact pinned invocation|exact closed contract",
    ):
        _admit_fixture(fixture)


def test_resealed_invocation_without_cuda_include_is_rejected(
    fixture: Fixture,
) -> None:
    build.build_bundle(fixture.request())
    _make_bundle_documents_writable(fixture)
    receipt = _load(fixture.bundle / build.BUILD_RECEIPT_NAME)
    artifact = _load(fixture.bundle / build.ARTIFACT_IDENTITY_NAME)
    argv = receipt["invocation"]["argv"]
    index = next(
        position
        for position, value in enumerate(argv)
        if value.startswith("NVCC=")
    )
    cuda_include = f" -I{fixture.tools.parent / 'include'}"
    assert cuda_include in argv[index]
    argv[index] = argv[index].replace(cuda_include, "")
    _rewrite_build_and_artifact(fixture, receipt, artifact)

    with pytest.raises(
        build.ExternalTracerBuildAuthorityError,
        match="does not satisfy|exact pinned invocation",
    ):
        _admit_fixture(fixture)


def test_output_mutation_is_rejected_by_verify(fixture: Fixture) -> None:
    build.build_bundle(fixture.request())
    tracer = fixture.bundle / build.TRACER_NAME
    tracer.chmod(0o644)
    tracer.write_bytes(b"mutated output")

    with pytest.raises(build.ExternalTracerBuildAuthorityError, match="differs|final authority"):
        _admit_fixture(fixture)


def test_cross_seal_mismatch_is_rejected_even_after_resealing(fixture: Fixture) -> None:
    build.build_bundle(fixture.request())
    artifact_path = fixture.bundle / build.ARTIFACT_IDENTITY_NAME
    fixture.bundle.chmod(0o755)
    artifact_path.chmod(0o644)
    artifact = _load(artifact_path)
    artifact["source_manifest"]["sha256"] = "0" * 64
    artifact["model_seal_sha256"] = immutable.model_seal(artifact)
    artifact_path.write_bytes(immutable.pretty_json_bytes(artifact))

    with pytest.raises(build.ExternalTracerBuildAuthorityError, match="cross-document"):
        _admit_fixture(fixture)


def test_build_root_must_be_fresh_direct_tmp_child(fixture: Fixture) -> None:
    nested_parent = fixture.root / "nested"
    nested_parent.mkdir()
    request = fixture.request(build_root=nested_parent / "build")
    with pytest.raises(build.ExternalTracerBuildAuthorityError, match="direct child"):
        build.build_bundle(request)


def test_authority_failure_maps_to_exit_two(fixture: Fixture) -> None:
    (fixture.repo / "dirty.txt").write_text("dirty", encoding="utf-8")
    request = fixture.request()
    arguments = [
        "build-candidate",
        "--repo-root", str(request.repo_root),
        "--build-root", str(request.build_root),
        "--bundle-dir", str(request.bundle_dir),
        "--protobuf-prefix", str(request.protobuf_prefix),
        "--git", str(request.git),
        "--bash", str(request.bash),
        "--make", str(request.make),
        "--nvcc", str(request.nvcc),
        "--ptxas", str(request.ptxas),
        "--cuobjdump", str(request.cuobjdump),
        "--host-cxx", str(request.host_cxx),
        "--protoc", str(request.protoc),
        "--pkg-config", str(request.pkg_config),
    ]
    assert build.main(arguments) == 2


def test_producer_has_no_repository_build_clean_copy_or_worktree_fallback() -> None:
    source = (_REPO / "tools/nvbit_tracer/external_tracer_build.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "git worktree",
        "git archive",
        "shutil.copy",
        "copytree(",
        "tracer_tool/tracer_tool.so\"" + ", repo_root",
        ".claude/worktrees/",
    )
    for token in forbidden:
        assert token not in source
    assert "cat-file" in source
    assert "ls-tree" in source
    assert "publish_directory" in source
    assert "accept_existing=False" in source
    assert "require_external_tracer_authority" in source


def test_source_manifest_records_mode_oid_sha_size_commit_tree_inventory(
    fixture: Fixture,
) -> None:
    build.build_bundle(fixture.request())
    source = _load(fixture.bundle / build.SOURCE_MANIFEST_NAME)
    assert len(source["repository"]["head_commit_sha1"]) == 40
    assert len(source["repository"]["head_tree_sha1"]) == 40
    assert source["inventory"]["file_count"] == len(source["files"])
    assert all(
        set(row) == {"path", "mode", "blob_oid_sha1", "sha256", "size_bytes"}
        for row in source["files"]
    )
    assert source["inventory"]["sha256"] == hashlib.sha256(
        immutable.canonical_json_bytes(source["files"])
    ).hexdigest()
