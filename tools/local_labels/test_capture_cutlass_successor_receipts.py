from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.local_labels import capture_cutlass_successor_receipts as capture


MATRIX_KEY = "cutlass_successor:vendor_gemm_cublas:vendor_gemm_cutlass"
VARIANT_ID = "vendor_gemm_cutlass_bf16_simt_trace00_v1"
GPU_UUID = "GPU-11111111-1111-1111-1111-111111111111"
_REAL_RETAIN_EXTERNAL_BUILD_AUTHORITY = (
    capture._retain_nvbit_external_build_authority
)


@pytest.fixture(autouse=True)
def _fixture_external_build_authority(monkeypatch):
    def retain(
        _stack,
        *,
        bundle_path,
        reviewed_tracer_sha256,
        tracer_authority,
        tracer_retained,
    ):
        assert tracer_authority.sha256 == reviewed_tracer_sha256
        assert tracer_retained.sha256 == reviewed_tracer_sha256
        bundle = Path(bundle_path)
        return capture.NvbitExternalBuildAuthority(
            bundle_path=bundle,
            expected_tracer_sha256=reviewed_tracer_sha256,
            source_commit_sha1="1" * 40,
            source_tree_sha1="2" * 40,
            source_inventory_sha256="3" * 64,
            source_manifest_payload=(bundle / "source_manifest.json").read_bytes(),
            build_receipt_payload=(bundle / "build_receipt.json").read_bytes(),
            artifact_identity_payload=(bundle / "artifact_identity.json").read_bytes(),
        )

    monkeypatch.setattr(
        capture,
        "_retain_nvbit_external_build_authority",
        retain,
    )



@pytest.fixture
def tmp_path():
    """Use a truly external tree for capture-output policy tests."""

    with tempfile.TemporaryDirectory(
        prefix="cutlass-successor-capture-test-",
        dir="/tmp",
    ) as directory:
        yield Path(directory)


@contextlib.contextmanager
def _retained_runtime(base_env: dict[str, str], retained_descriptor: int):
    runtime_descriptor = capture.os.dup(retained_descriptor)
    environment = capture.nvbit_runtime.sanitized_capture_environment(base_env)
    environment["LD_LIBRARY_PATH"] = f"/proc/self/fd/{runtime_descriptor}"
    try:
        yield SimpleNamespace(
            environment=environment,
            pass_fds=(runtime_descriptor,),
        )
    finally:
        capture.os.close(runtime_descriptor)


def _runtime_manifest_kwargs(tracer: Path) -> dict[str, object]:
    payload = b"fixture protobuf runtime DSO"
    document = capture.immutable.seal_model({
        "authority_kind": capture.nvbit_runtime.RUNTIME_MANIFEST_AUTHORITY_KIND,
        "dsos": [{
            "path": "libprotobuf.so.34.1.0",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "soname": "libprotobuf.so.34.1.0",
        }],
        "runtime_library_path": str(capture.NVBIT_PB_LIB),
        "schema_version": capture.nvbit_runtime.RUNTIME_MANIFEST_SCHEMA_VERSION,
        "tracer": {
            "sha256": hashlib.sha256(tracer.read_bytes()).hexdigest(),
            "size_bytes": tracer.stat().st_size,
        },
    })
    path = tracer.with_name(f"{tracer.name}.runtime-manifest.json")
    path.write_bytes(capture.immutable.pretty_json_bytes(document))
    tracer_digest = hashlib.sha256(tracer.read_bytes()).hexdigest()
    bundle = tracer.with_name(f"{tracer.name}.admitted-build")
    bundle.mkdir(exist_ok=True)
    (bundle / "source_manifest.json").write_bytes(b'{"fixture":"source"}\n')
    (bundle / "build_receipt.json").write_bytes(b'{"fixture":"build"}\n')
    (bundle / "artifact_identity.json").write_bytes(b'{"fixture":"artifact"}\n')
    (bundle / "tracer_tool.so").write_bytes(tracer.read_bytes())
    return {
        "nvbit_runtime_manifest": path,
        "nvbit_runtime_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "nvbit_external_build_bundle": bundle,
        "nvbit_reviewed_tracer_sha256": tracer_digest,
    }


def _containment() -> capture.delegated_cgroup.CgroupContainmentEvidence:
    return capture.delegated_cgroup.CgroupContainmentEvidence(
        kind=capture.delegated_cgroup.CGROUP_V2_KILL_V1,
        parent_cgroup_sha256="c" * 64,
        cgroup_inode=1234,
        terminal_populated_zero=True,
        leaf_removed=True,
    )


class _FailingGenerationPin:
    def __init__(self, observed: dict[str, BaseException | None]) -> None:
        self._observed = observed

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, primary, _traceback):
        self._observed["close_primary"] = primary
        raise capture.immutable.ImmutableArtifactError(
            "injected generation pin close failure"
        )


def test_retains_independently_admitted_external_build_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tracer, tracer_payload = _external_tracer(tmp_path)
    digest = hashlib.sha256(tracer_payload).hexdigest()
    bundle = tmp_path / "reviewed-build-bundle"
    bundle.mkdir()
    payloads = {
        "source_manifest.json": b'{"source":"fixture"}\n',
        "build_receipt.json": b'{"build":"fixture"}\n',
        "artifact_identity.json": b'{"artifact":"fixture"}\n',
        "tracer_tool.so": tracer_payload,
    }
    for name, payload in payloads.items():
        path = bundle / name
        path.write_bytes(payload)
        path.chmod(0o444)
    bundle.chmod(0o555)
    artifact = {
        "source_manifest": {
            "sha256": hashlib.sha256(payloads["source_manifest.json"]).hexdigest(),
            "size_bytes": len(payloads["source_manifest.json"]),
        },
        "build_receipt": {
            "sha256": hashlib.sha256(payloads["build_receipt.json"]).hexdigest(),
            "size_bytes": len(payloads["build_receipt.json"]),
        },
        "source_identity": {
            "head_commit_sha1": "1" * 40,
            "head_tree_sha1": "2" * 40,
            "inventory_sha256": "3" * 64,
        },
        "artifact": {
            "name": "tracer_tool.so",
            "sha256": digest,
            "size_bytes": len(tracer_payload),
            "source_kind": capture.external_build.EXTERNAL_TRACER_SOURCE_KIND,
        },
    }

    def verify(path, **kwargs):
        assert path == bundle
        assert kwargs["expected_tracer_sha256"] == digest
        assert kwargs["cuobjdump"] == capture.CUDA_CUOBJDUMP
        return artifact

    monkeypatch.setattr(capture.external_build, "verify_bundle", verify)
    tracer_authority = capture._require_nvbit_tracer_authority(tracer, digest)
    try:
        with contextlib.ExitStack() as stack:
            retained = stack.enter_context(
                capture._retained_bounded_regular(
                    tracer,
                    "fixture tracer",
                    max_bytes=capture.EXTERNAL_TRACER_MAX_BYTES,
                )
            )
            authority = _REAL_RETAIN_EXTERNAL_BUILD_AUTHORITY(
                stack,
                bundle_path=bundle,
                reviewed_tracer_sha256=digest,
                tracer_authority=tracer_authority,
                tracer_retained=retained,
            )
            assert authority.provenance() == {
                "authority_kind": capture.external_build.REVIEWED_AUTHORITY_KIND,
                "expected_tracer_sha256": digest,
                "source_commit_sha1": "1" * 40,
                "source_tree_sha1": "2" * 40,
                "source_inventory_sha256": "3" * 64,
                "source_manifest_sha256": hashlib.sha256(
                    payloads["source_manifest.json"]
                ).hexdigest(),
                "build_receipt_sha256": hashlib.sha256(
                    payloads["build_receipt.json"]
                ).hexdigest(),
                "artifact_identity_sha256": hashlib.sha256(
                    payloads["artifact_identity.json"]
                ).hexdigest(),
            }
    finally:
        bundle.chmod(0o755)
        for path in bundle.iterdir():
            path.chmod(0o644)


def test_capture_producer_identity_matches_current_source() -> None:
    source, digest = capture._capture_producer_identity()

    assert source == Path(capture.__file__).resolve(strict=True)
    assert digest == hashlib.sha256(source.read_bytes()).hexdigest()


def test_capture_producer_identity_rejects_resealed_named_replacement(
    monkeypatch,
) -> None:
    payload = Path(capture.__file__).read_bytes().replace(
        b"Capture one diagnostic",
        b"Capture old diagnostic",
        1,
    )
    match = capture.admission._CAPTURE_PRODUCER_SELF_SEAL.search(payload)
    assert match is not None
    start, end = match.span(1)
    normalized = payload[:start] + (b"0" * 64) + payload[end:]
    replacement_seal = hashlib.sha256(normalized).hexdigest().encode("ascii")
    replacement = payload[:start] + replacement_seal + payload[end:]

    class RetainedReplacement:
        sha256 = hashlib.sha256(replacement).hexdigest()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read_bytes(self, *, max_bytes=None):
            assert max_bytes == capture.CAPTURE_SOURCE_MAX_BYTES
            return replacement

    monkeypatch.setattr(
        capture.immutable,
        "open_retained_regular",
        lambda _path, **kwargs: (
            pytest.fail("capture source byte ceiling was not supplied")
            if kwargs.get("max_bytes") != capture.CAPTURE_SOURCE_MAX_BYTES
            else RetainedReplacement()
        ),
    )

    with pytest.raises(
        capture.CaptureError,
        match="loaded capture producer disagrees",
    ):
        capture._capture_producer_identity()


def _pinned_preflight(tmp_path: Path) -> Path:
    row, roster_row = capture._row_for(MATRIX_KEY, VARIANT_ID)
    document = capture._expected_preflight(row, roster_row)
    path = tmp_path / "successor_preflight.json"
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
    return path


def test_preflight_clock_refusal_happens_before_live_gpu_queries(tmp_path: Path, monkeypatch):
    path = _pinned_preflight(tmp_path)
    document = json.loads(path.read_text())
    document["sm_clock_mhz"] = 1410
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
    monkeypatch.setattr(
        capture,
        "validate_live_capture_preflight",
        lambda **_kwargs: pytest.fail(
            "live GPU query must not run after static preflight refusal"
        ),
    )
    monkeypatch.setattr(
        capture,
        "_run_capture_command",
        lambda *args, **kwargs: pytest.fail("capture process must not run"),
    )
    with pytest.raises(capture.CaptureError, match="preflight pins mismatch"):
        capture.capture_phase(
            matrix_key=MATRIX_KEY,
            variant_id=VARIANT_ID,
            phase="ncu",
            preflight_path=path,
            output_root=tmp_path / "capture-output",
        )
    assert not list((tmp_path / "capture-output").glob("*.capture_receipt.json"))


def test_preflight_requires_exact_successor_identity(tmp_path: Path):
    path = _pinned_preflight(tmp_path)
    document = json.loads(path.read_text())
    document["binary_sha256"] = "0" * 64
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
    row, roster_row = capture._row_for(MATRIX_KEY, VARIANT_ID)
    with pytest.raises(capture.CaptureError, match="preflight pins mismatch"):
        capture.validate_successor_preflight(path, row, roster_row)


def test_preflight_rejects_extras_and_wrong_toolchain(tmp_path: Path):
    path = _pinned_preflight(tmp_path)
    document = json.loads(path.read_text())
    document["cuda_release"] = {"major": 13, "minor": 3}
    document["unexpected"] = True
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
    row, roster_row = capture._row_for(MATRIX_KEY, VARIANT_ID)
    with pytest.raises(capture.CaptureError, match="preflight pins mismatch"):
        capture.validate_successor_preflight(path, row, roster_row)


def test_operator_json_byte_limit_precedes_materialization(
    tmp_path: Path,
    monkeypatch,
):
    path = tmp_path / "oversized-compact.json"
    path.write_bytes(
        b'{"padding":"'
        + b"x" * capture.OPERATOR_PREFLIGHT_MAX_BYTES
        + b'"}'
    )
    monkeypatch.setattr(
        capture.immutable,
        "strict_json_loads",
        lambda *_args, **_kwargs: pytest.fail(
            "oversized operator JSON reached materialization"
        ),
    )

    with pytest.raises(capture.CaptureError) as exc_info:
        capture._load_object(path, "successor preflight")

    assert exc_info.value.code == "capture_input_limit_exceeded"


def test_compact_high_cardinality_json_fails_before_materialization(
    tmp_path: Path,
    monkeypatch,
):
    payload = (
        "{"
        + ",".join(
            f'"k{index}":0'
            for index in range(capture.JSON_MAX_CONTAINER_ITEMS + 1)
        )
        + "}"
    ).encode()
    assert len(payload) < capture.OPERATOR_PREFLIGHT_MAX_BYTES
    path = tmp_path / "compact-high-cardinality.json"
    path.write_bytes(payload)
    monkeypatch.setattr(
        capture.immutable,
        "strict_json_loads",
        lambda *_args, **_kwargs: pytest.fail(
            "high-cardinality JSON reached materialization"
        ),
    )

    with pytest.raises(capture.CaptureError) as exc_info:
        capture._load_object(path, "successor preflight")

    assert exc_info.value.code == "capture_json_cardinality_limit_exceeded"


@pytest.mark.parametrize(
    "failure",
    (
        MemoryError("synthetic retained-open exhaustion"),
        OSError(capture.errno.EMFILE, "synthetic EMFILE"),
        OSError(capture.errno.ENFILE, "synthetic ENFILE"),
        OSError(capture.errno.ENOMEM, "synthetic ENOMEM"),
    ),
)
def test_operator_reader_resource_failures_are_typed(
    tmp_path: Path,
    monkeypatch,
    failure: BaseException,
):
    path = tmp_path / "operator.json"
    path.write_bytes(b"{}")
    monkeypatch.setattr(
        capture.immutable,
        "open_retained_regular",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(capture.CaptureError) as exc_info:
        capture._load_object(path, "successor preflight")

    assert exc_info.value.code == "capture_input_resource_unavailable"


def test_operator_json_materialization_memoryerror_is_resource_typed(
    tmp_path: Path,
    monkeypatch,
):
    path = tmp_path / "operator.json"
    path.write_bytes(b"{}")
    monkeypatch.setattr(
        capture.immutable,
        "strict_json_loads",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            MemoryError("synthetic JSON materialization exhaustion")
        ),
    )

    with pytest.raises(capture.CaptureError) as exc_info:
        capture._load_object(path, "successor preflight")

    assert exc_info.value.code == "capture_input_resource_unavailable"


def test_operator_reader_post_open_substitution_is_generation_typed(
    tmp_path: Path,
    monkeypatch,
):
    path = tmp_path / "operator.json"
    path.write_bytes(b'{"identity":"retained"}')
    real_open = capture.immutable.open_retained_regular

    def open_then_substitute(open_path: Path, **kwargs):
        retained = real_open(open_path, **kwargs)
        real_read = retained.read_bytes

        def substitute_before_read(*, max_bytes=None):
            replacement = path.with_name("replacement.json")
            replacement.write_bytes(b'{"identity":"replacement"}')
            replacement.replace(path)
            return real_read(max_bytes=max_bytes)

        retained.read_bytes = substitute_before_read
        return retained

    monkeypatch.setattr(
        capture.immutable,
        "open_retained_regular",
        open_then_substitute,
    )

    with pytest.raises(capture.CaptureError) as exc_info:
        capture._load_object(path, "successor preflight")

    assert exc_info.value.code == "capture_generation_changed"


def test_preflight_rejects_non_absolute_input(tmp_path: Path):
    row, roster_row = capture._row_for(MATRIX_KEY, VARIANT_ID)
    with pytest.raises(capture.CaptureError, match="absolute"):
        capture.validate_successor_preflight(Path("relative-preflight.json"), row, roster_row)


def test_external_nvbit_tracer_rejects_relative_repository_and_symlink_paths(
    tmp_path: Path,
):
    external = tmp_path / "tracer_tool.so"
    external.write_bytes(b"external tracer")
    digest = capture.sha256_bytes(external.read_bytes())
    symlink = tmp_path / "tracer_link.so"
    symlink.symlink_to(external)

    with pytest.raises(capture.CaptureError, match="absolute path"):
        capture._require_nvbit_tracer_authority(Path("tracer_tool.so"), digest)
    with pytest.raises(capture.CaptureError, match="external to the repository"):
        capture._require_nvbit_tracer_authority(Path(capture.__file__), digest)
    with pytest.raises(capture.CaptureError, match="symlink"):
        capture._require_nvbit_tracer_authority(symlink, digest)

    checkout = tmp_path / "external-checkout"
    checkout.mkdir()
    (checkout / ".git").mkdir()
    checkout_tracer = checkout / "tracer_tool.so"
    checkout_tracer.write_bytes(b"Git checkout tracer")
    with pytest.raises(capture.CaptureError, match="non-Git build tree"):
        capture._require_nvbit_tracer_authority(checkout_tracer, digest)

    bare = tmp_path / "bare.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    bare_tracer = bare / "tracer_tool.so"
    bare_tracer.write_bytes(b"bare Git tracer")
    with pytest.raises(capture.CaptureError, match="non-Git build tree"):
        capture._require_nvbit_tracer_authority(bare_tracer, digest)


def test_external_nvbit_tracer_rejects_empty_and_hardlinked_sources_before_hardware(
    tmp_path: Path,
    monkeypatch,
):
    empty = tmp_path / "empty_tracer.so"
    empty.touch()
    with pytest.raises(capture.CaptureError, match="nonempty regular file"):
        capture._require_nvbit_tracer_authority(empty, capture.sha256_bytes(b""))

    tracer, payload = _external_tracer(tmp_path)
    hardlink = tracer.with_name("hardlinked_tracer.so")
    os.link(tracer, hardlink)
    monkeypatch.setattr(
        capture,
        "validate_live_capture_preflight",
        lambda **_kwargs: pytest.fail("hardlinked tracer reached hardware preflight"),
    )

    with pytest.raises(capture.CaptureError, match="hard-linked alias"):
        capture.capture_phase(
            matrix_key=MATRIX_KEY,
            variant_id=VARIANT_ID,
            phase="nvbit",
            preflight_path=tmp_path / "unused-preflight.json",
            output_root=tmp_path / "output",
            nvbit_tracer_path=tracer,
            nvbit_tracer_sha256=capture.sha256_bytes(payload),
            **_runtime_manifest_kwargs(tracer),
        )


@pytest.mark.parametrize(
    ("stage", "mutated_member"),
    tuple(
        (stage, member)
        for stage in ("copy", "launch", "admission", "finalization")
        for member in ("source", "copied")
    ),
)
def test_nvbit_tracer_chain_rehashes_source_and_copy_drift_at_every_boundary(
    tmp_path: Path,
    stage: str,
    mutated_member: str,
) -> None:
    tracer, payload = _external_tracer(tmp_path)
    digest = capture.sha256_bytes(payload)
    authority = capture._require_nvbit_tracer_authority(tracer, digest)
    copied_path = tmp_path / "retained_capture_tool.so"
    copied_path.write_bytes(payload)

    source = capture._open_bounded_regular(
        tracer,
        "external NVBit tracer",
        max_bytes=capture.EXTERNAL_TRACER_MAX_BYTES,
    )
    copied = capture._open_bounded_regular(
        copied_path,
        "retained NVBit tracer",
        max_bytes=capture.EXTERNAL_TRACER_MAX_BYTES,
    )
    try:
        capture._verify_external_tracer_checkpoint(
            source,
            authority,
            stage="initial authority",
            initial=True,
        )
        capture._verify_nvbit_tracer_chain(
            authority,
            source,
            copied,
            digest,
            stage=f"pre-{stage}",
        )
        mutated_path = tracer if mutated_member == "source" else copied_path
        mutated_path.write_bytes(
            f"{mutated_member} drift during {stage}".encode()
        )
        with pytest.raises(capture.CaptureError) as exc_info:
            capture._verify_nvbit_tracer_chain(
                authority,
                source,
                copied,
                digest,
                stage=stage,
            )
    finally:
        copied.close()
        source.close()

    assert exc_info.value.code == "capture_generation_changed"
    assert stage in exc_info.value.detail


def test_nvbit_tracer_must_be_outside_capture_output_root(
    tmp_path: Path,
    monkeypatch,
):
    preflight = _pinned_preflight(tmp_path)
    output_root = tmp_path / "capture-output"
    tracer = output_root / "build" / "tracer_tool.so"
    tracer.parent.mkdir(parents=True)
    tracer.write_bytes(b"misplaced tracer")
    monkeypatch.setattr(
        capture,
        "validate_live_capture_preflight",
        lambda *, phase: _nvbit_live()
        if phase == "nvbit"
        else pytest.fail("wrong phase"),
    )
    monkeypatch.setattr(
        capture,
        "_capture_producer_identity",
        lambda: (Path(capture.__file__), "a" * 64),
    )
    monkeypatch.setattr(
        capture,
        "_run_capture_command",
        lambda *_args, **_kwargs: pytest.fail("misplaced tracer must not launch"),
    )

    with pytest.raises(capture.CaptureError, match="outside the capture output root"):
        capture.capture_phase(
            matrix_key=MATRIX_KEY,
            variant_id=VARIANT_ID,
            phase="nvbit",
            preflight_path=preflight,
            output_root=output_root,
            nvbit_tracer_path=tracer,
            nvbit_tracer_sha256=capture.sha256_bytes(tracer.read_bytes()),
            **_runtime_manifest_kwargs(tracer),
        )


def test_nvbit_capture_requires_explicit_external_tracer_pair(tmp_path: Path):
    with pytest.raises(capture.CaptureError, match="requires exactly"):
        capture.capture_phase(
            matrix_key=MATRIX_KEY,
            variant_id=VARIANT_ID,
            phase="nvbit",
            preflight_path=tmp_path / "unused-preflight.json",
            output_root=tmp_path / "capture-output",
        )


def test_cli_requires_nvbit_tracer_for_nvbit_phase(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        capture.main([
            "--matrix-key",
            MATRIX_KEY,
            "--variant-id",
            VARIANT_ID,
            "--phase",
            "nvbit",
            "--preflight",
            "/tmp/preflight.json",
            "--output-root",
            "/tmp/output",
        ])

    assert exc_info.value.code == 2
    assert "requires exactly --nvbit-tracer, --nvbit-tracer-sha256, --nvbit-runtime-manifest" in (
        capsys.readouterr().err
    )


@pytest.mark.parametrize(
    "tracer_args",
    (
        ["--nvbit-tracer", "/tmp/tracer_tool.so"],
        ["--nvbit-tracer-sha256", "0" * 64],
    ),
)
def test_cli_rejects_either_nvbit_authority_argument_alone(
    capsys,
    tracer_args: list[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        capture.main([
            "--matrix-key",
            MATRIX_KEY,
            "--variant-id",
            VARIANT_ID,
            "--phase",
            "nvbit",
            "--preflight",
            "/tmp/preflight.json",
            "--output-root",
            "/tmp/output",
            *tracer_args,
        ])

    assert exc_info.value.code == 2
    assert "requires exactly --nvbit-tracer, --nvbit-tracer-sha256, --nvbit-runtime-manifest" in (
        capsys.readouterr().err
    )


@pytest.mark.parametrize(
    "duplicate_args",
    (
        [
            "--nvbit-tracer",
            "/tmp/first.so",
            "--nvbit-tracer",
            "/tmp/second.so",
            "--nvbit-tracer-sha256",
            "0" * 64,
        ],
        [
            "--nvbit-tracer",
            "/tmp/tracer.so",
            "--nvbit-tracer-sha256",
            "0" * 64,
            "--nvbit-tracer-sha256",
            "1" * 64,
        ],
    ),
)
def test_cli_rejects_duplicate_nvbit_authority_options(
    capsys,
    duplicate_args: list[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        capture.main([
            "--matrix-key",
            MATRIX_KEY,
            "--variant-id",
            VARIANT_ID,
            "--phase",
            "nvbit",
            "--preflight",
            "/tmp/preflight.json",
            "--output-root",
            "/tmp/output",
            *duplicate_args,
        ])

    assert exc_info.value.code == 2
    assert "must be supplied exactly once" in capsys.readouterr().err


@pytest.mark.parametrize("declared_sha256", ("A" * 64, "0" * 63, "g" * 64))
def test_cli_rejects_malformed_nvbit_hash_before_hardware_execution(
    tmp_path: Path,
    monkeypatch,
    capsys,
    declared_sha256: str,
) -> None:
    tracer, _payload = _external_tracer(tmp_path)
    manifest_authority = _runtime_manifest_kwargs(tracer)
    monkeypatch.setattr(
        capture,
        "validate_live_capture_preflight",
        lambda **_kwargs: pytest.fail("malformed hash reached hardware preflight"),
    )

    result = capture.main([
        "--matrix-key",
        MATRIX_KEY,
        "--variant-id",
        VARIANT_ID,
        "--phase",
        "nvbit",
        "--preflight",
        str(tmp_path / "unused-preflight.json"),
        "--output-root",
        str(tmp_path / "output"),
        "--nvbit-tracer",
        str(tracer),
        "--nvbit-tracer-sha256",
        declared_sha256,
        "--nvbit-runtime-manifest",
        str(manifest_authority["nvbit_runtime_manifest"]),
        "--nvbit-runtime-manifest-sha256",
        str(manifest_authority["nvbit_runtime_manifest_sha256"]),
        "--nvbit-external-build-bundle",
        str(manifest_authority["nvbit_external_build_bundle"]),
        "--nvbit-reviewed-tracer-sha256",
        str(manifest_authority["nvbit_reviewed_tracer_sha256"]),
    ])

    assert result == 1
    assert "exact lowercase 64-hex" in capsys.readouterr().err


def test_cli_rejects_relative_nvbit_tracer_before_hardware_execution(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        capture,
        "validate_live_capture_preflight",
        lambda **_kwargs: pytest.fail("relative tracer reached hardware preflight"),
    )

    result = capture.main([
        "--matrix-key",
        MATRIX_KEY,
        "--variant-id",
        VARIANT_ID,
        "--phase",
        "nvbit",
        "--preflight",
        str(tmp_path / "unused-preflight.json"),
        "--output-root",
        str(tmp_path / "output"),
        "--nvbit-tracer",
        "relative-tracer.so",
        "--nvbit-tracer-sha256",
        "0" * 64,
        "--nvbit-runtime-manifest",
        str(tmp_path / "unused-runtime-manifest.json"),
        "--nvbit-runtime-manifest-sha256",
        "0" * 64,
        "--nvbit-external-build-bundle",
        str(tmp_path / "unused-build-bundle"),
        "--nvbit-reviewed-tracer-sha256",
        "0" * 64,
    ])

    assert result == 1
    assert "absolute path" in capsys.readouterr().err


def test_cli_rejects_nvbit_hash_mismatch_before_hardware_execution(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    tracer, _payload = _external_tracer(tmp_path)
    manifest_authority = _runtime_manifest_kwargs(tracer)
    monkeypatch.setattr(
        capture,
        "validate_live_capture_preflight",
        lambda **_kwargs: pytest.fail("hash mismatch reached hardware preflight"),
    )

    result = capture.main([
        "--matrix-key",
        MATRIX_KEY,
        "--variant-id",
        VARIANT_ID,
        "--phase",
        "nvbit",
        "--preflight",
        str(tmp_path / "unused-preflight.json"),
        "--output-root",
        str(tmp_path / "output"),
        "--nvbit-tracer",
        str(tracer),
        "--nvbit-tracer-sha256",
        "0" * 64,
        "--nvbit-runtime-manifest",
        str(manifest_authority["nvbit_runtime_manifest"]),
        "--nvbit-runtime-manifest-sha256",
        str(manifest_authority["nvbit_runtime_manifest_sha256"]),
        "--nvbit-external-build-bundle",
        str(manifest_authority["nvbit_external_build_bundle"]),
        "--nvbit-reviewed-tracer-sha256",
        str(manifest_authority["nvbit_reviewed_tracer_sha256"]),
    ])

    assert result == 1
    assert "digest mismatch" in capsys.readouterr().err


def test_cli_accepts_and_forwards_exact_nvbit_authority_pair(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tracer, payload = _external_tracer(tmp_path)
    digest = capture.sha256_bytes(payload)
    manifest_authority = _runtime_manifest_kwargs(tracer)
    observed = {}
    expected_receipt = tmp_path / "receipt.json"

    def accept(**kwargs):
        observed.update(kwargs)
        return expected_receipt

    monkeypatch.setattr(capture, "capture_phase", accept)

    result = capture.main([
        "--matrix-key",
        MATRIX_KEY,
        "--variant-id",
        VARIANT_ID,
        "--phase",
        "nvbit",
        "--preflight",
        str(tmp_path / "preflight.json"),
        "--output-root",
        str(tmp_path / "output"),
        "--nvbit-tracer",
        str(tracer),
        "--nvbit-tracer-sha256",
        digest,
        "--nvbit-runtime-manifest",
        str(manifest_authority["nvbit_runtime_manifest"]),
        "--nvbit-runtime-manifest-sha256",
        str(manifest_authority["nvbit_runtime_manifest_sha256"]),
        "--nvbit-external-build-bundle",
        str(manifest_authority["nvbit_external_build_bundle"]),
        "--nvbit-reviewed-tracer-sha256",
        str(manifest_authority["nvbit_reviewed_tracer_sha256"]),
    ])

    assert result == 0
    assert observed["nvbit_tracer_path"] == tracer
    assert observed["nvbit_tracer_sha256"] == digest
    assert observed["nvbit_runtime_manifest"] == (
        manifest_authority["nvbit_runtime_manifest"]
    )
    assert observed["nvbit_runtime_manifest_sha256"] == (
        manifest_authority["nvbit_runtime_manifest_sha256"]
    )
    assert observed["nvbit_external_build_bundle"] == (
        manifest_authority["nvbit_external_build_bundle"]
    )
    assert observed["nvbit_reviewed_tracer_sha256"] == (
        manifest_authority["nvbit_reviewed_tracer_sha256"]
    )


@pytest.mark.parametrize(
    "tracer_args",
    (
        ["--nvbit-tracer", "/tmp/tracer_tool.so"],
        ["--nvbit-tracer-sha256", "0" * 64],
        [
            "--nvbit-tracer",
            "/tmp/tracer_tool.so",
            "--nvbit-tracer-sha256",
            "0" * 64,
        ],
    ),
)
def test_cli_forbids_any_nvbit_tracer_authority_on_ncu_phase(
    capsys,
    tracer_args: list[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        capture.main([
            "--matrix-key",
            MATRIX_KEY,
            "--variant-id",
            VARIANT_ID,
            "--phase",
            "ncu",
            "--preflight",
            "/tmp/preflight.json",
            "--output-root",
            "/tmp/output",
            *tracer_args,
        ])

    assert exc_info.value.code == 2
    assert "valid only with --phase nvbit" in capsys.readouterr().err


def test_capture_output_root_uses_protected_path_gate_before_mutation(
    tmp_path: Path,
    monkeypatch,
):
    output_root = tmp_path / "capture-output"
    observed = {}

    def refuse(path, *, repo_root, operation, **_kwargs):
        observed.update({
            "path": path,
            "repo_root": repo_root,
            "operation": operation,
        })
        raise capture.protected_paths.ProtectedPathError("synthetic protected root")

    monkeypatch.setattr(
        capture.protected_paths,
        "assert_external_output",
        refuse,
    )
    monkeypatch.setattr(
        capture,
        "validate_live_capture_preflight",
        lambda **_kwargs: pytest.fail("protected output must fail before live queries"),
    )

    with pytest.raises(capture.CaptureError, match="synthetic protected root"):
        capture.capture_phase(
            matrix_key=MATRIX_KEY,
            variant_id=VARIANT_ID,
            phase="ncu",
            preflight_path=tmp_path / "unused.json",
            output_root=output_root,
        )

    assert observed == {
        "path": output_root,
        "repo_root": capture.ROOT,
        "operation": "write CUTLASS successor capture output",
    }
    assert not output_root.exists()


def test_capture_phase_reservation_blocks_process_and_reuses_stale_owner(
    tmp_path: Path,
):
    reservation = tmp_path / "capture.lock"
    ready = tmp_path / "ready"
    program = r'''
import sys
import time
from pathlib import Path
from tools.local_labels import capture_cutlass_successor_receipts as capture

with capture._capture_phase_reservation(
    matrix_key="child-matrix",
    variant_id="child-variant",
    phase="ncu",
    output_root=Path("/tmp/child-output"),
    reservation_path=Path(sys.argv[1]),
):
    Path(sys.argv[2]).write_text("ready")
    time.sleep(30)
'''
    child = subprocess.Popen(
        [
            capture.sys.executable,
            "-c",
            program,
            str(reservation),
            str(ready),
        ],
        cwd=capture.ROOT,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and child.poll() is None:
            if time.monotonic() >= deadline:
                pytest.fail("reservation child did not acquire its lock")
            time.sleep(0.01)
        assert child.poll() is None
        with pytest.raises(capture.CaptureError) as exc_info:
            with capture._capture_phase_reservation(
                matrix_key=MATRIX_KEY,
                variant_id=VARIANT_ID,
                phase="nvbit",
                output_root=tmp_path / "capture-output",
                reservation_path=reservation,
            ):
                pytest.fail("contending phase acquired the reservation")
        assert exc_info.value.code == "capture_phase_reserved"
        assert "child-variant" in exc_info.value.detail
    finally:
        child.terminate()
        child.wait(timeout=5)

    stale_payload = reservation.read_bytes()
    assert b'"pid"' in stale_payload
    with capture._capture_phase_reservation(
        matrix_key=MATRIX_KEY,
        variant_id=VARIANT_ID,
        phase="nvbit",
        output_root=tmp_path / "capture-output",
        reservation_path=reservation,
    ) as owner:
        assert owner["pid"] == capture.os.getpid()
        assert owner["phase"] == "nvbit"
        assert reservation.read_bytes() != stale_payload


def test_nvbit_live_preflight_never_discovers_or_invokes_ncu(monkeypatch):
    observed_queries = []

    def query(_command, label):
        observed_queries.append(label)
        if label == "nvidia-smi device":
            return (
                f"{capture.GPU_NAME}, 8.6, Enabled, "
                f"{capture.SM_CLOCK_MHZ}, {capture.MEM_CLOCK_MHZ}\n",
                [],
            )
        if label == "nvidia-smi competing-process":
            return "", []
        if label == "nvidia-smi device identity":
            return f"{GPU_UUID}, {capture.DRIVER_VERSION}\n", []
        if label == "CUDA 13.2 nvcc":
            return "synthetic nvcc", []
        pytest.fail(f"unexpected live query: {label}")

    monkeypatch.setattr(capture, "_query", query)
    monkeypatch.setattr(capture, "_nvbit_version", lambda: capture.NVBIT_VERSION)
    monkeypatch.setattr(
        capture,
        "parse_nvcc_release_banner",
        lambda _stdout: {"major": 13, "minor": 2},
    )
    monkeypatch.setattr(capture, "require_sol_cuda_release", lambda _cuda: None)
    monkeypatch.setattr(
        capture,
        "_probe_ncu_authority",
        lambda: pytest.fail("NVBit preflight must not launch or discover NCU"),
    )

    live = capture.validate_live_capture_preflight(phase="nvbit")

    assert observed_queries == [
        "nvidia-smi device",
        "nvidia-smi competing-process",
        "nvidia-smi device identity",
        "CUDA 13.2 nvcc",
    ]
    assert not any(key.startswith("ncu_") for key in live)


@pytest.mark.parametrize(
    "malformed_uuid",
    (
        "GPU-11111111111111111111111111111111",
        "GPU-11111111-1111-1111-1111-11111111111",
        "GPU-11111111-1111-1111-1111-111111111111-",
    ),
)
def test_live_preflight_rejects_malformed_gpu_uuid(
    monkeypatch,
    malformed_uuid: str,
):
    def query(_command, label):
        if label == "nvidia-smi device":
            return (
                f"{capture.GPU_NAME}, 8.6, Enabled, "
                f"{capture.SM_CLOCK_MHZ}, {capture.MEM_CLOCK_MHZ}\n",
                [],
            )
        if label == "nvidia-smi competing-process":
            return "", []
        if label == "nvidia-smi device identity":
            return f"{malformed_uuid}, {capture.DRIVER_VERSION}\n", []
        pytest.fail(f"unexpected live query: {label}")

    monkeypatch.setattr(capture, "_query", query)

    with pytest.raises(capture.CaptureError, match="live GPU identity is not pinned"):
        capture.validate_live_capture_preflight(phase="nvbit")


def test_live_preflight_rejects_retired_driver_authority(monkeypatch):
    def query(_command, label):
        if label == "nvidia-smi device":
            return (
                f"{capture.GPU_NAME}, 8.6, Enabled, "
                f"{capture.SM_CLOCK_MHZ}, {capture.MEM_CLOCK_MHZ}\n",
                [],
            )
        if label == "nvidia-smi competing-process":
            return "", []
        if label == "nvidia-smi device identity":
            return f"{GPU_UUID}, 610.43.02\n", []
        pytest.fail(f"unexpected live query: {label}")

    monkeypatch.setattr(capture, "_query", query)

    with pytest.raises(capture.CaptureError, match="live GPU identity is not pinned"):
        capture.validate_live_capture_preflight(phase="nvbit")


def test_capture_metadata_is_namespaced_by_variant_and_phase(tmp_path: Path):
    first = set(
        capture._capture_metadata_paths(
            tmp_path,
            variant_id="vendor_gemm_cutlass_bf16_simt_trace00_v1",
            phase="ncu",
        )
    )
    second = set(
        capture._capture_metadata_paths(
            tmp_path,
            variant_id="vendor_sgemm_cutlass_fp32_simt_trace00_v1",
            phase="ncu",
        )
    )
    first_nvbit = set(
        capture._capture_metadata_paths(
            tmp_path,
            variant_id="vendor_gemm_cutlass_bf16_simt_trace00_v1",
            phase="nvbit",
        )
    )

    assert first.isdisjoint(second)
    assert first.isdisjoint(first_nvbit)
    assert second.isdisjoint(first_nvbit)


def test_post_capture_tool_identity_rejects_substitution(tmp_path: Path):
    tracer = tmp_path / "tracer_tool.so"
    tracer.write_bytes(b"pinned tracer")
    retained = capture._open_bounded_regular(
        tracer,
        "NVBit tracer",
        max_bytes=capture.EXTERNAL_TRACER_MAX_BYTES,
    )
    expected = retained.sha256
    tracer.write_bytes(b"substituted tracer")

    try:
        with pytest.raises(capture.CaptureError) as exc_info:
            capture._verify_unchanged(retained, expected, "NVBit tracer")
    finally:
        retained.close()

    assert exc_info.value.code == "capture_generation_changed"
    assert "generation verification failed" in exc_info.value.detail


def test_failure_provenance_seals_process_tool_and_stream_identity(
    tmp_path: Path, monkeypatch,
):
    failure_root = tmp_path / "failure"
    failure_root.mkdir()
    (failure_root / "capture_timeout.stdout.bin").write_bytes(b"partial stdout")
    row, _roster_row = capture._row_for(MATRIX_KEY, VARIANT_ID)
    command = ["/fixture/target", "--shape", "128"]
    environment_payload = b'{"environment":{"LANG":"C"}}\n'
    error = capture.CaptureError(
        "synthetic timeout",
        code="capture_timeout",
        stdout=b"partial stdout",
        process_id=4321,
        started_at_utc="2026-08-02T00:00:00Z",
    )
    monkeypatch.setattr(capture, "_boot_id_sha256", lambda: "b" * 64)

    capture._write_failure_provenance(
        failure_root,
        root=tmp_path,
        phase="ncu",
        row=row,
        command=command,
        live={
            "gpu_name": capture.GPU_NAME,
            "smi_stdout_sha256": "c" * 64,
            "gpu_uuid_sha256": "u" * 64,
            "driver_version": capture.DRIVER_VERSION,
            "identity_stdout_sha256": "i" * 64,
        },
        environment_payload=environment_payload,
        default_environment={"LANG": "C"},
        capture_producer_sha256="d" * 64,
        containment_supervisor_sha256="a" * 64,
        phase_tool_sha256="e" * 64,
        runtime_manifest_sha256=None,
        action_executor_sha256="f" * 64,
        error=error,
    )

    document = json.loads((failure_root / "failure_provenance.json").read_text())
    assert document["failure_code"] == "capture_timeout"
    assert document["process"] == {
        "role": "ncu_capture",
        "boot_id_sha256": "b" * 64,
        "command": command,
        "command_sha256": capture.canonical_sha(command),
        "environment": {"LANG": "C"},
        "environment_sha256": capture.canonical_sha({"LANG": "C"}),
        "host_identity_sha256": capture.canonical_sha({
            "gpu": capture.GPU_NAME,
            "smi": "c" * 64,
            "gpu_uuid_sha256": "u" * 64,
            "driver_version": capture.DRIVER_VERSION,
            "identity_stdout_sha256": "i" * 64,
        }),
        "pid": 4321,
        "started_at_utc": "2026-08-02T00:00:00Z",
    }
    assert document["capture_environment_document_sha256"] == (
        capture.sha256_bytes(environment_payload)
    )
    assert document["handoff_artifacts"] == []
    assert document["provenance"] == {
        "action_executor_sha256": "f" * 64,
        "capture_producer_sha256": "d" * 64,
        "containment_supervisor_sha256": "a" * 64,
        "phase_tool_sha256": "e" * 64,
    }
    assert document["streams"] == [
        {
            "path": "capture_timeout.stdout.bin",
            "sha256": capture.sha256_bytes(b"partial stdout"),
            "size_bytes": len(b"partial stdout"),
        }
    ]
    seal = document.pop("failure_seal_sha256")
    assert seal == capture.canonical_sha(document)


def test_failure_provenance_pin_close_failure_preserves_semantic_primary(
    tmp_path: Path,
    monkeypatch,
):
    failure_root = tmp_path / "failure"
    failure_root.mkdir()
    (failure_root / "capture_failure.stderr.bin").write_bytes(b"semantic failure")
    row, _roster_row = capture._row_for(MATRIX_KEY, VARIANT_ID)
    semantic = capture.CaptureError(
        "synthetic semantic capture failure",
        code="capture_failed",
    )
    observed: dict[str, BaseException | None] = {}
    generation_stack = contextlib.ExitStack()
    generation_stack.enter_context(_FailingGenerationPin(observed))
    monkeypatch.setattr(capture, "_boot_id_sha256", lambda: "b" * 64)

    capture._write_failure_provenance(
        failure_root,
        root=tmp_path,
        phase="ncu",
        row=row,
        command=["/fixture/target"],
        live=_nvbit_live(),
        environment_payload=b"{}\n",
        default_environment={"LANG": "C"},
        capture_producer_sha256="c" * 64,
        containment_supervisor_sha256="d" * 64,
        phase_tool_sha256="e" * 64,
        runtime_manifest_sha256=None,
        action_executor_sha256="f" * 64,
        error=semantic,
        generation_stack=generation_stack,
    )

    assert observed["close_primary"] is semantic
    assert any(
        "failure-provenance generation cleanup failed: "
        "injected generation pin close failure" in note
        for note in semantic.__notes__
    )
    document = json.loads(
        (failure_root / "failure_provenance.json").read_text()
    )
    assert document["failure_code"] == "capture_failed"
    assert document["detail"] == "synthetic semantic capture failure"


def _nvbit_live() -> dict:
    smi_stdout = (
        f"{capture.GPU_NAME}, 8.6, Enabled, "
        f"{capture.SM_CLOCK_MHZ}, {capture.MEM_CLOCK_MHZ}\n"
    )
    return {
        "smi_command": ["nvidia-smi"],
        "smi_stdout": smi_stdout,
        "smi_stdout_sha256": capture.sha256_bytes(smi_stdout.encode()),
        "competing_process_command": ["nvidia-smi", "--compute-apps"],
        "competing_process_stdout": "",
        "competing_process_stdout_sha256": capture.sha256_bytes(b""),
        "identity_command": ["nvidia-smi", "--identity"],
        "identity_stdout": f"{GPU_UUID}, {capture.DRIVER_VERSION}\n",
        "identity_stdout_sha256": capture.sha256_bytes(
            f"{GPU_UUID}, {capture.DRIVER_VERSION}\n".encode()
        ),
        "gpu_uuid_sha256": capture.sha256_bytes(GPU_UUID.encode()),
        "driver_version": capture.DRIVER_VERSION,
        "nvcc_command": [str(capture.CUDA_NVCC), "--version"],
        "nvcc_stdout": "synthetic nvcc",
        "nvcc_stdout_sha256": capture.sha256_bytes(b"synthetic nvcc"),
        "cuda_release": dict(capture.CUDA_RELEASE),
        "nvbit_version": capture.NVBIT_VERSION,
        "gpu_name": capture.GPU_NAME,
        "persistence_mode": "Enabled",
        "sm_clock_mhz": capture.SM_CLOCK_MHZ,
        "mem_clock_mhz": capture.MEM_CLOCK_MHZ,
    }


def _external_tracer(tmp_path: Path) -> tuple[Path, bytes]:
    payload = b"external split-capable tracer"
    path = tmp_path / "external_tracer_tool.so"
    path.write_bytes(payload)
    return path, payload


def _assert_failed_nvbit_phase_tool(
    document: dict,
    output_root: Path,
    expected_payload: bytes,
    *,
    expected_name: str = f"{VARIANT_ID}.nvbit_tracer_tool.so",
) -> None:
    artifacts = [
        artifact
        for artifact in document["handoff_artifacts"]
        if artifact["role"] == "phase_tool"
    ]
    expected_path = Path("capture_tools") / expected_name
    assert artifacts == [{
        "role": "phase_tool",
        "path": expected_path.as_posix(),
        "sha256": capture.sha256_bytes(expected_payload),
        "size_bytes": len(expected_payload),
    }]
    assert (output_root / expected_path).read_bytes() == expected_payload
    assert document["provenance"]["phase_tool_sha256"] == (
        capture.sha256_bytes(expected_payload)
    )
    runtime_manifest_artifacts = [
        artifact
        for artifact in document["handoff_artifacts"]
        if artifact["role"] == "runtime_manifest"
    ]
    assert len(runtime_manifest_artifacts) == 1
    runtime_manifest_artifact = runtime_manifest_artifacts[0]
    runtime_manifest_path = output_root / runtime_manifest_artifact["path"]
    runtime_manifest_payload = runtime_manifest_path.read_bytes()
    assert runtime_manifest_artifact["sha256"] == capture.sha256_bytes(
        runtime_manifest_payload
    )
    assert runtime_manifest_artifact["size_bytes"] == len(
        runtime_manifest_payload
    )
    assert document["provenance"]["runtime_manifest_sha256"] == (
        runtime_manifest_artifact["sha256"]
    )


def test_trace_root_is_descriptor_pinned_before_capture_shutdown_gap(
    tmp_path: Path,
    monkeypatch,
):
    preflight = _pinned_preflight(tmp_path)
    output_root = tmp_path / "capture-output"
    tracer, _tracer_payload = _external_tracer(tmp_path)
    row, _roster = capture._row_for(MATRIX_KEY, VARIANT_ID)
    trace_root = output_root / "nvbit" / row["successor_slug"]
    displaced = trace_root.with_name(f"{trace_root.name}.displaced")

    monkeypatch.setattr(
        capture,
        "validate_live_capture_preflight",
        lambda *, phase: _nvbit_live()
        if phase == "nvbit"
        else pytest.fail("wrong phase"),
    )
    monkeypatch.setattr(
        capture.nvbit_runtime,
        "retained_tracer_runtime_env",
        lambda _path, *, base_env, retained_descriptor, **_kwargs: (
            _retained_runtime(base_env, retained_descriptor)
        ),
    )

    def replace_root(_command, environment, _cwd, **kwargs):
        trace_proc_path = kwargs["byte_root"]
        assert trace_proc_path == Path(environment["TRACES_FOLDER"])
        assert trace_proc_path.as_posix().startswith("/proc/self/fd/")
        trace_descriptor = int(trace_proc_path.name)
        assert trace_descriptor in kwargs["inherited_descriptors"]
        assert Path(os.readlink(trace_proc_path)) == trace_root
        trace_root.replace(displaced)
        trace_root.mkdir()
        return (
            0,
            4321,
            "2026-08-02T00:00:00Z",
            b"",
            b"",
            _containment(),
        )

    monkeypatch.setattr(capture, "_run_capture_command", replace_root)
    monkeypatch.setattr(capture, "_boot_id_sha256", lambda: "b" * 64)

    with pytest.raises(capture.CaptureError) as exc_info:
        capture.capture_phase(
            matrix_key=MATRIX_KEY,
            variant_id=VARIANT_ID,
            phase="nvbit",
            preflight_path=preflight,
            output_root=output_root,
            nvbit_tracer_path=tracer,
            nvbit_tracer_sha256=capture.sha256_bytes(tracer.read_bytes()),
            **_runtime_manifest_kwargs(tracer),
        )

    assert exc_info.value.code == "capture_generation_changed"
    assert displaced.is_dir()
    assert trace_root.is_dir()
    assert not (
        output_root / f"{VARIANT_ID}.nvbit.capture_receipt.json"
    ).exists()
    failure = json.loads(
        (
            output_root
            / "failures"
            / VARIANT_ID
            / "nvbit"
            / "failure_provenance.json"
        ).read_text()
    )
    assert failure["failure_code"] == "capture_generation_changed"


def test_trace_root_pin_resource_failure_is_typed_before_nvbit_launch(
    tmp_path: Path,
    monkeypatch,
):
    preflight = _pinned_preflight(tmp_path)
    output_root = tmp_path / "capture-output"
    tracer, tracer_payload = _external_tracer(tmp_path)
    row, _roster = capture._row_for(MATRIX_KEY, VARIANT_ID)
    trace_root = output_root / "nvbit" / row["successor_slug"]
    real_reader = capture.immutable.PinnedDirectoryReader

    class ResourceReader(real_reader):
        def __init__(self, path):
            if Path(path) == trace_root:
                raise capture.immutable.ImmutableArtifactResourceError(
                    "injected trace-root descriptor exhaustion"
                )
            super().__init__(path)

    monkeypatch.setattr(
        capture,
        "validate_live_capture_preflight",
        lambda *, phase: _nvbit_live()
        if phase == "nvbit"
        else pytest.fail("wrong phase"),
    )
    monkeypatch.setattr(
        capture.immutable,
        "PinnedDirectoryReader",
        ResourceReader,
    )
    monkeypatch.setattr(
        capture,
        "_run_capture_command",
        lambda *_args, **_kwargs: pytest.fail("NVBit launched without output pin"),
    )
    monkeypatch.setattr(capture, "_boot_id_sha256", lambda: "b" * 64)

    with pytest.raises(capture.CaptureError) as exc_info:
        capture.capture_phase(
            matrix_key=MATRIX_KEY,
            variant_id=VARIANT_ID,
            phase="nvbit",
            preflight_path=preflight,
            output_root=output_root,
            nvbit_tracer_path=tracer,
            nvbit_tracer_sha256=capture.sha256_bytes(tracer.read_bytes()),
            **_runtime_manifest_kwargs(tracer),
        )

    assert exc_info.value.code == "trace_snapshot_resource_unavailable"
    failure = json.loads(
        (
            output_root
            / "failures"
            / VARIANT_ID
            / "nvbit"
            / "failure_provenance.json"
        ).read_text()
    )
    assert failure["failure_code"] == "trace_snapshot_resource_unavailable"
    _assert_failed_nvbit_phase_tool(failure, output_root, tracer_payload)


def test_external_nvbit_tracer_is_retained_and_executed_by_descriptor(
    tmp_path: Path,
    monkeypatch,
):
    preflight = _pinned_preflight(tmp_path)
    output_root = tmp_path / "capture-output"
    tracer = tmp_path / "external_tracer_tool.so"
    tracer_payload = b"external split-capable tracer"
    tracer.write_bytes(tracer_payload)
    observed = {}

    monkeypatch.setattr(
        capture,
        "validate_live_capture_preflight",
        lambda *, phase: _nvbit_live()
        if phase == "nvbit"
        else pytest.fail("wrong phase"),
    )

    @contextlib.contextmanager
    def retained_runtime(_path, *, base_env, retained_descriptor, **_kwargs):
        observed["runtime_base_environment"] = dict(base_env)
        with _retained_runtime(base_env, retained_descriptor) as runtime:
            yield runtime

    monkeypatch.setattr(
        capture.nvbit_runtime,
        "retained_tracer_runtime_env",
        retained_runtime,
    )

    def run(command, environment, _cwd, **kwargs):
        observed["command"] = command
        observed["environment"] = environment
        observed["descriptors"] = kwargs["inherited_descriptors"]
        observed["byte_root"] = kwargs["byte_root"]
        observed["byte_root_descriptor"] = kwargs[
            "byte_root_descriptor"
        ]
        observed["trace_target"] = Path(
            os.readlink(environment["TRACES_FOLDER"])
        )
        return (
            0,
            4321,
            "2026-08-02T00:00:00Z",
            b"",
            b"",
            _containment(),
        )

    def finalize(**kwargs):
        retained = kwargs["retained_phase_tool_path"]
        assert retained == (
            output_root
            / "capture_tools"
            / f"{VARIANT_ID}.nvbit_tracer_tool.so"
        )
        assert retained.read_bytes() == tracer_payload
        assert kwargs["phase_tool_sha256"] == capture.sha256_bytes(
            tracer_payload
        )
        kwargs["generation_stack"].close()
        return kwargs["receipt_path"]

    monkeypatch.setattr(capture, "_run_capture_command", run)
    monkeypatch.setattr(capture, "_finalize_capture", finalize)

    result = capture.capture_phase(
        matrix_key=MATRIX_KEY,
        variant_id=VARIANT_ID,
        phase="nvbit",
        preflight_path=preflight,
        output_root=output_root,
        nvbit_tracer_path=tracer,
        nvbit_tracer_sha256=capture.sha256_bytes(tracer.read_bytes()),
        **_runtime_manifest_kwargs(tracer),
    )

    assert result == output_root / f"{VARIANT_ID}.nvbit.capture_receipt.json"
    assert {
        "CUDA_INJECTION64_PATH",
        "LD_LIBRARY_PATH",
        "TRACES_FOLDER",
        "USER_DEFINED_FOLDERS",
    }.isdisjoint(observed["runtime_base_environment"])
    assert observed["command"][0].startswith("/proc/self/fd/")
    assert observed["environment"]["CUDA_INJECTION64_PATH"].startswith(
        "/proc/self/fd/"
    )
    assert observed["environment"]["TRACES_FOLDER"].startswith(
        "/proc/self/fd/"
    )
    assert observed["environment"]["USER_DEFINED_FOLDERS"] == "1"
    assert observed["byte_root"] == Path(
        observed["environment"]["TRACES_FOLDER"]
    )
    assert observed["byte_root_descriptor"] in observed["descriptors"]
    assert observed["byte_root"] == Path(
        f"/proc/self/fd/{observed['byte_root_descriptor']}"
    )
    assert observed["trace_target"] == (
        output_root / "nvbit" / "vendor_gemm_cutlass"
    )
    assert len(observed["descriptors"]) == 4
    assert tracer.read_bytes() == tracer_payload


def test_capture_inventory_uses_exact_retained_directory_descriptor(
    tmp_path: Path,
):
    trace_root = tmp_path / "trace"
    trace_root.mkdir()
    (trace_root / "kernelslist").write_bytes(b"kernel\n")
    descriptor = os.open(
        trace_root,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
    )
    try:
        proc_root = Path(f"/proc/self/fd/{descriptor}")
        inventory = capture._capture_tree_inventory(
            proc_root,
            retained_root_descriptor=descriptor,
        )
        assert inventory["file_count"] == 1
        assert inventory["total_bytes"] == len(b"kernel\n")

        with pytest.raises(
            capture.CaptureError,
            match="retained root is not exact",
        ):
            capture._capture_tree_inventory(
                trace_root,
                retained_root_descriptor=descriptor,
            )

        (trace_root / "escape").symlink_to(tmp_path)
        with pytest.raises(
            capture.CaptureError,
            match="capture output descendant is unsafe",
        ) as caught:
            capture._capture_tree_inventory(
                proc_root,
                retained_root_descriptor=descriptor,
            )
        assert caught.value.code == "capture_inventory_unsafe_entry"
    finally:
        os.close(descriptor)


def test_post_copy_source_mutation_retains_authenticated_failed_tool(
    tmp_path: Path,
    monkeypatch,
):
    preflight = _pinned_preflight(tmp_path)
    output_root = tmp_path / "capture-output"
    tracer = tmp_path / "external_tracer_tool.so"
    tracer_payload = b"external split-capable tracer"
    tracer.write_bytes(tracer_payload)
    retained_name = f"{VARIANT_ID}.nvbit_tracer_tool.so"
    real_write_new = capture._write_new

    monkeypatch.setattr(
        capture,
        "validate_live_capture_preflight",
        lambda *, phase: _nvbit_live()
        if phase == "nvbit"
        else pytest.fail("wrong phase"),
    )
    monkeypatch.setattr(
        capture,
        "_run_capture_command",
        lambda *_args, **_kwargs: pytest.fail(
            "post-copy mutation failure must not launch"
        ),
    )
    monkeypatch.setattr(capture, "_boot_id_sha256", lambda: "b" * 64)

    def mutate_after_copy(path: Path, payload: bytes) -> None:
        real_write_new(path, payload)
        if path.name == retained_name:
            tracer.write_bytes(b"mutated external source")

    monkeypatch.setattr(capture, "_write_new", mutate_after_copy)

    with pytest.raises(capture.CaptureError) as exc_info:
        capture.capture_phase(
            matrix_key=MATRIX_KEY,
            variant_id=VARIANT_ID,
            phase="nvbit",
            preflight_path=preflight,
            output_root=output_root,
            nvbit_tracer_path=tracer,
            nvbit_tracer_sha256=capture.sha256_bytes(tracer.read_bytes()),
            **_runtime_manifest_kwargs(tracer),
        )

    assert exc_info.value.code == "capture_generation_changed"
    assert "post-copy" in exc_info.value.detail
    failure_path = (
        output_root
        / "failures"
        / VARIANT_ID
        / "nvbit"
        / "failure_provenance.json"
    )
    failure = json.loads(failure_path.read_text())
    assert failure["failure_code"] == "capture_generation_changed"
    assert failure["process"]["pid"] is None
    _assert_failed_nvbit_phase_tool(failure, output_root, tracer_payload)
    assert tracer.read_bytes() == b"mutated external source"
    assert not (
        output_root / f"{VARIANT_ID}.nvbit.capture_receipt.json"
    ).exists()


def test_post_copy_runtime_manifest_mutation_retains_authenticated_fallback(
    tmp_path: Path,
    monkeypatch,
):
    preflight = _pinned_preflight(tmp_path)
    output_root = tmp_path / "capture-output"
    tracer, tracer_payload = _external_tracer(tmp_path)
    manifest_authority = _runtime_manifest_kwargs(tracer)
    manifest_path = Path(manifest_authority["nvbit_runtime_manifest"])
    manifest_payload = manifest_path.read_bytes()
    retained_name = f"{VARIANT_ID}.nvbit_runtime_manifest.json"
    real_write_new = capture._write_new

    monkeypatch.setattr(
        capture,
        "validate_live_capture_preflight",
        lambda *, phase: _nvbit_live()
        if phase == "nvbit"
        else pytest.fail("wrong phase"),
    )
    monkeypatch.setattr(
        capture,
        "_run_capture_command",
        lambda *_args, **_kwargs: pytest.fail(
            "runtime-manifest copy mutation must not launch"
        ),
    )
    monkeypatch.setattr(capture, "_boot_id_sha256", lambda: "b" * 64)

    def mutate_after_copy(path: Path, payload: bytes) -> None:
        real_write_new(path, payload)
        if path.name == retained_name:
            path.chmod(0o600)
            path.write_bytes(b"mutated copied runtime manifest")

    monkeypatch.setattr(capture, "_write_new", mutate_after_copy)

    with pytest.raises(capture.CaptureError) as exc_info:
        capture.capture_phase(
            matrix_key=MATRIX_KEY,
            variant_id=VARIANT_ID,
            phase="nvbit",
            preflight_path=preflight,
            output_root=output_root,
            nvbit_tracer_path=tracer,
            nvbit_tracer_sha256=capture.sha256_bytes(tracer.read_bytes()),
            **manifest_authority,
        )

    assert exc_info.value.code == "capture_generation_changed", (
        exc_info.value.detail
    )
    failure_path = (
        output_root
        / "failures"
        / VARIANT_ID
        / "nvbit"
        / "failure_provenance.json"
    )
    failure = json.loads(failure_path.read_text())
    _assert_failed_nvbit_phase_tool(failure, output_root, tracer_payload)
    manifest_artifact = next(
        artifact
        for artifact in failure["handoff_artifacts"]
        if artifact["role"] == "runtime_manifest"
    )
    assert manifest_artifact["path"] == (
        f"capture_tools/{VARIANT_ID}.failed_nvbit_runtime_manifest.json"
    )
    assert (output_root / manifest_artifact["path"]).read_bytes() == (
        manifest_payload
    )
    assert not (
        output_root / f"{VARIANT_ID}.nvbit.capture_receipt.json"
    ).exists()


def test_external_tracer_launch_drift_refuses_receipt_and_retains_failure_tool(
    tmp_path: Path,
    monkeypatch,
):
    preflight = _pinned_preflight(tmp_path)
    output_root = tmp_path / "capture-output"
    tracer, tracer_payload = _external_tracer(tmp_path)
    tracer_sha256 = capture.sha256_bytes(tracer_payload)

    monkeypatch.setattr(
        capture,
        "validate_live_capture_preflight",
        lambda *, phase: _nvbit_live()
        if phase == "nvbit"
        else pytest.fail("wrong phase"),
    )
    monkeypatch.setattr(
        capture.nvbit_runtime,
        "retained_tracer_runtime_env",
        lambda _path, *, base_env, retained_descriptor, **_kwargs: (
            _retained_runtime(base_env, retained_descriptor)
        ),
    )

    def mutate_during_launch(*_args, **_kwargs):
        tracer.write_bytes(b"external tracer drifted during launch")
        return (
            0,
            4321,
            "2026-08-02T00:00:00Z",
            b"",
            b"",
            _containment(),
        )

    monkeypatch.setattr(capture, "_run_capture_command", mutate_during_launch)
    monkeypatch.setattr(capture, "_boot_id_sha256", lambda: "b" * 64)

    with pytest.raises(capture.CaptureError) as exc_info:
        capture.capture_phase(
            matrix_key=MATRIX_KEY,
            variant_id=VARIANT_ID,
            phase="nvbit",
            preflight_path=preflight,
            output_root=output_root,
            nvbit_tracer_path=tracer,
            nvbit_tracer_sha256=tracer_sha256,
            **_runtime_manifest_kwargs(tracer),
        )

    assert exc_info.value.code == "capture_generation_changed"
    assert "post-launch" in exc_info.value.detail
    failure_path = (
        output_root
        / "failures"
        / VARIANT_ID
        / "nvbit"
        / "failure_provenance.json"
    )
    failure = json.loads(failure_path.read_text())
    assert failure["failure_code"] == "capture_generation_changed"
    _assert_failed_nvbit_phase_tool(failure, output_root, tracer_payload)
    assert not (
        output_root / f"{VARIANT_ID}.nvbit.capture_receipt.json"
    ).exists()


def test_external_source_descriptor_remains_retained_through_finalization(
    tmp_path: Path,
    monkeypatch,
):
    preflight = _pinned_preflight(tmp_path)
    output_root = tmp_path / "capture-output"
    tracer, tracer_payload = _external_tracer(tmp_path)
    tracer_sha256 = capture.sha256_bytes(tracer_payload)
    real_open = capture.immutable.open_retained_regular
    observed = {"source_exited": False}

    class TrackSourceContext:
        def __init__(self, path, **kwargs):
            self.path = Path(path)
            self.retained = real_open(path, **kwargs)

        def __getattr__(self, name):
            return getattr(self.retained, name)

        def __enter__(self):
            return self.retained.__enter__()

        def __exit__(self, *exc_info):
            result = self.retained.__exit__(*exc_info)
            if self.path == tracer:
                observed["source_exited"] = True
            return result

    monkeypatch.setattr(
        capture,
        "validate_live_capture_preflight",
        lambda *, phase: _nvbit_live()
        if phase == "nvbit"
        else pytest.fail("wrong phase"),
    )
    monkeypatch.setattr(
        capture.immutable,
        "open_retained_regular",
        TrackSourceContext,
    )
    monkeypatch.setattr(
        capture.nvbit_runtime,
        "retained_tracer_runtime_env",
        lambda _path, *, base_env, retained_descriptor, **_kwargs: (
            _retained_runtime(base_env, retained_descriptor)
        ),
    )
    monkeypatch.setattr(
        capture,
        "_run_capture_command",
        lambda *_args, **_kwargs: (
            0,
            4321,
            "2026-08-02T00:00:00Z",
            b"",
            b"",
            _containment(),
        ),
    )

    def finalize(**kwargs):
        assert observed["source_exited"] is False
        source = kwargs["external_tracer_retained"]
        assert source.path == tracer
        capture.os.fstat(source.descriptor)
        assert kwargs["external_tracer_authority"].sha256 == tracer_sha256
        kwargs["generation_stack"].close()
        return kwargs["receipt_path"]

    monkeypatch.setattr(capture, "_finalize_capture", finalize)

    result = capture.capture_phase(
        matrix_key=MATRIX_KEY,
        variant_id=VARIANT_ID,
        phase="nvbit",
        preflight_path=preflight,
        output_root=output_root,
        nvbit_tracer_path=tracer,
        nvbit_tracer_sha256=tracer_sha256,
        **_runtime_manifest_kwargs(tracer),
    )

    assert result == output_root / f"{VARIANT_ID}.nvbit.capture_receipt.json"
    assert observed["source_exited"] is True


def test_failed_tracer_fallback_path_is_reserved_at_startup(
    tmp_path: Path,
    monkeypatch,
):
    preflight = _pinned_preflight(tmp_path)
    output_root = tmp_path / "capture-output"
    fallback = (
        output_root
        / "capture_tools"
        / f"{VARIANT_ID}.failed_nvbit_tracer_tool.so"
    )
    fallback.parent.mkdir(parents=True)
    fallback.write_bytes(b"retained prior failure")
    tracer, _tracer_payload = _external_tracer(tmp_path)
    monkeypatch.setattr(
        capture,
        "validate_live_capture_preflight",
        lambda *, phase: _nvbit_live()
        if phase == "nvbit"
        else pytest.fail("wrong phase"),
    )
    monkeypatch.setattr(
        capture,
        "_capture_producer_identity",
        lambda: (Path(capture.__file__), "a" * 64),
    )
    monkeypatch.setattr(
        capture,
        "_run_capture_command",
        lambda *_args, **_kwargs: pytest.fail(
            "capture with pre-existing fallback must not launch"
        ),
    )

    with pytest.raises(capture.CaptureError, match="pre-existing output artifact"):
        capture.capture_phase(
            matrix_key=MATRIX_KEY,
            variant_id=VARIANT_ID,
            phase="nvbit",
            preflight_path=preflight,
            output_root=output_root,
            nvbit_tracer_path=tracer,
            nvbit_tracer_sha256=capture.sha256_bytes(tracer.read_bytes()),
            **_runtime_manifest_kwargs(tracer),
        )

    assert fallback.read_bytes() == b"retained prior failure"


def test_external_nvbit_tracer_substitution_after_capture_refuses_receipt(
    tmp_path: Path,
    monkeypatch,
):
    preflight = _pinned_preflight(tmp_path)
    output_root = tmp_path / "capture-output"
    tracer = tmp_path / "external_tracer_tool.so"
    tracer.write_bytes(b"external split-capable tracer")
    retained = (
        output_root
        / "capture_tools"
        / f"{VARIANT_ID}.nvbit_tracer_tool.so"
    )

    monkeypatch.setattr(
        capture,
        "validate_live_capture_preflight",
        lambda *, phase: _nvbit_live()
        if phase == "nvbit"
        else pytest.fail("wrong phase"),
    )
    monkeypatch.setattr(
        capture.nvbit_runtime,
        "retained_tracer_runtime_env",
        lambda _path, *, base_env, retained_descriptor, **_kwargs: (
            _retained_runtime(base_env, retained_descriptor)
        ),
    )

    def run(_command, _environment, _cwd, **_kwargs):
        replacement = retained.with_name("replacement_tracer_tool.so")
        replacement.write_bytes(b"substituted after capture")
        replacement.replace(retained)
        return (
            0,
            4321,
            "2026-08-02T00:00:00Z",
            b"",
            b"",
            _containment(),
        )

    monkeypatch.setattr(capture, "_run_capture_command", run)
    monkeypatch.setattr(capture, "_boot_id_sha256", lambda: "b" * 64)

    with pytest.raises(capture.CaptureError, match="post-launch verification"):
        capture.capture_phase(
            matrix_key=MATRIX_KEY,
            variant_id=VARIANT_ID,
            phase="nvbit",
            preflight_path=preflight,
            output_root=output_root,
            nvbit_tracer_path=tracer,
            nvbit_tracer_sha256=capture.sha256_bytes(tracer.read_bytes()),
            **_runtime_manifest_kwargs(tracer),
        )

    assert not (
        output_root / f"{VARIANT_ID}.nvbit.capture_receipt.json"
    ).exists()
    failure_path = (
        output_root
        / "failures"
        / VARIANT_ID
        / "nvbit"
        / "failure_provenance.json"
    )
    assert failure_path.is_file()
    _assert_failed_nvbit_phase_tool(
        json.loads(failure_path.read_text()),
        output_root,
        b"external split-capable tracer",
        expected_name=f"{VARIANT_ID}.failed_nvbit_tracer_tool.so",
    )


@pytest.mark.parametrize(
    ("failure_type", "stream_write_failure"),
    (
        (OSError, False),
        (TypeError, False),
        (KeyboardInterrupt, False),
        (TypeError, True),
    ),
)
def test_nvbit_descriptor_execution_and_post_validation_failure_provenance(
    tmp_path: Path,
    monkeypatch,
    failure_type: type[BaseException],
    stream_write_failure: bool,
):
    preflight = _pinned_preflight(tmp_path)
    output_root = tmp_path / "capture-output"
    tracer, tracer_payload = _external_tracer(tmp_path)
    live = _nvbit_live()
    observed = {}

    def run(command, environment, _cwd, **kwargs):
        observed.update({
            "command": command,
            "environment": environment,
            "descriptors": kwargs["inherited_descriptors"],
            "byte_root": kwargs["byte_root"],
            "trace_target": Path(os.readlink(environment["TRACES_FOLDER"])),
        })
        assert command[0].startswith("/proc/self/fd/")
        assert environment["CUDA_INJECTION64_PATH"].startswith(
            "/proc/self/fd/"
        )
        for descriptor in kwargs["inherited_descriptors"]:
            capture.os.fstat(descriptor)
        return (
            0,
            4321,
            "2026-08-02T00:00:00Z",
            b"validated stdout",
            b"",
            _containment(),
        )

    monkeypatch.setattr(
        capture,
        "validate_live_capture_preflight",
        lambda *, phase: live if phase == "nvbit" else pytest.fail("wrong phase"),
    )
    monkeypatch.setattr(
        capture.nvbit_runtime,
        "retained_tracer_runtime_env",
        lambda _path, *, base_env, retained_descriptor, **_kwargs: (
            _retained_runtime(base_env, retained_descriptor)
        ),
    )
    monkeypatch.setattr(capture, "_run_capture_command", run)

    class GenerationGuard:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            observed["generation_closed"] = True
            return False

    def finalize(**kwargs):
        assert kwargs["capture_stdout"] == b"validated stdout"
        assert kwargs["capture_stderr"] == b""
        kwargs["generation_stack"].enter_context(GenerationGuard())
        raise failure_type("synthetic post-validation failure")

    monkeypatch.setattr(capture, "_finalize_capture", finalize)
    if stream_write_failure:
        real_write_new = capture._write_new

        def write_with_diagnostic_failure(path: Path, payload: bytes) -> None:
            if path.name == "post_validation.stdout.bin":
                raise capture.CaptureError("synthetic diagnostic write failure")
            real_write_new(path, payload)

        monkeypatch.setattr(capture, "_write_new", write_with_diagnostic_failure)
    monkeypatch.setattr(capture, "_boot_id_sha256", lambda: "b" * 64)

    expected_error = (
        KeyboardInterrupt
        if failure_type is KeyboardInterrupt
        else capture.CaptureError
    )
    expected_message = (
        "synthetic post-validation failure"
        if failure_type is KeyboardInterrupt
        else "post-capture validation"
    )
    with pytest.raises(expected_error, match=expected_message):
        capture.capture_phase(
            matrix_key=MATRIX_KEY,
            variant_id=VARIANT_ID,
            phase="nvbit",
            preflight_path=preflight,
            output_root=output_root,
            nvbit_tracer_path=tracer,
            nvbit_tracer_sha256=capture.sha256_bytes(tracer.read_bytes()),
            **_runtime_manifest_kwargs(tracer),
        )

    assert len(observed["descriptors"]) == 4
    assert observed["byte_root"] == Path(
        observed["environment"]["TRACES_FOLDER"]
    )
    assert observed["trace_target"] == (
        output_root / "nvbit" / "vendor_gemm_cutlass"
    )
    assert observed["generation_closed"] is True
    failure_root = output_root / "failures" / VARIANT_ID / "nvbit"
    stdout_path = failure_root / "post_validation.stdout.bin"
    if stream_write_failure:
        assert not stdout_path.exists()
    else:
        assert stdout_path.read_bytes() == b"validated stdout"
    document = json.loads((failure_root / "failure_provenance.json").read_text())
    if stream_write_failure:
        assert "post-validation stream retention failed" in document["detail"]
    _assert_failed_nvbit_phase_tool(
        document,
        output_root,
        tracer_payload,
    )
    assert document["process"]["pid"] == 4321
    assert document["process"]["role"] == "nvbit_postprocess_validation"
    assert document["process"]["command"][0] != observed["command"][0]
    assert document["process"]["environment"]["CUDA_INJECTION64_PATH"] == str(
        output_root
        / "capture_tools"
        / f"{VARIANT_ID}.nvbit_tracer_tool.so"
    )
    assert document["provenance"] == {
        "capture_producer_sha256": capture.sha256_bytes(
            Path(capture.__file__).read_bytes()
        ),
        "containment_supervisor_sha256": capture.sha256_bytes(
            Path(capture.delegated_cgroup.__file__).read_bytes()
        ),
        "phase_tool_sha256": capture.sha256_bytes(tracer_payload),
        "runtime_manifest_sha256": next(
            artifact["sha256"]
            for artifact in document["handoff_artifacts"]
            if artifact["role"] == "runtime_manifest"
        ),
        "external_build_authority": {
            "authority_kind": capture.external_build.REVIEWED_AUTHORITY_KIND,
            "expected_tracer_sha256": capture.sha256_bytes(tracer_payload),
            "source_commit_sha1": "1" * 40,
            "source_tree_sha1": "2" * 40,
            "source_inventory_sha256": "3" * 64,
            "source_manifest_sha256": next(
                artifact["sha256"]
                for artifact in document["handoff_artifacts"]
                if artifact["role"] == "external_build_source_manifest"
            ),
            "build_receipt_sha256": next(
                artifact["sha256"]
                for artifact in document["handoff_artifacts"]
                if artifact["role"] == "external_build_receipt"
            ),
            "artifact_identity_sha256": next(
                artifact["sha256"]
                for artifact in document["handoff_artifacts"]
                if artifact["role"] == "external_build_artifact_identity"
            ),
        },
    }


def test_post_validation_pin_close_failure_preserves_primary_and_provenance(
    tmp_path: Path,
    monkeypatch,
):
    preflight = _pinned_preflight(tmp_path)
    output_root = tmp_path / "capture-output"
    tracer, tracer_payload = _external_tracer(tmp_path)
    observed: dict[str, BaseException | None] = {}

    monkeypatch.setattr(
        capture,
        "validate_live_capture_preflight",
        lambda *, phase: _nvbit_live()
        if phase == "nvbit"
        else pytest.fail("wrong phase"),
    )
    monkeypatch.setattr(
        capture,
        "_validate_capture_producer_identity",
        lambda source, retained: (source, retained.sha256),
    )
    monkeypatch.setattr(
        capture.nvbit_runtime,
        "retained_tracer_runtime_env",
        lambda _path, *, base_env, retained_descriptor, **_kwargs: (
            _retained_runtime(base_env, retained_descriptor)
        ),
    )
    monkeypatch.setattr(
        capture,
        "_run_capture_command",
        lambda *_args, **_kwargs: (
            0,
            4321,
            "2026-08-02T00:00:00Z",
            b"validated stdout",
            b"",
            _containment(),
        ),
    )

    def fail_finalization(**kwargs):
        kwargs["generation_stack"].enter_context(
            _FailingGenerationPin(observed)
        )
        raise capture.CaptureError(
            "synthetic semantic post-validation failure",
            code="post_capture_validation_failed",
        )

    monkeypatch.setattr(capture, "_finalize_capture", fail_finalization)
    real_write_failure_provenance = capture._write_failure_provenance

    def write_with_independent_generation(*args, **kwargs):
        assert kwargs.pop("generation_stack") is not None
        return real_write_failure_provenance(*args, **kwargs)

    monkeypatch.setattr(
        capture,
        "_write_failure_provenance",
        write_with_independent_generation,
    )
    monkeypatch.setattr(capture, "_boot_id_sha256", lambda: "b" * 64)

    with pytest.raises(
        capture.CaptureError,
        match="synthetic semantic post-validation failure",
    ) as exc_info:
        capture.capture_phase(
            matrix_key=MATRIX_KEY,
            variant_id=VARIANT_ID,
            phase="nvbit",
            preflight_path=preflight,
            output_root=output_root,
            nvbit_tracer_path=tracer,
            nvbit_tracer_sha256=capture.sha256_bytes(tracer.read_bytes()),
            **_runtime_manifest_kwargs(tracer),
        )

    semantic = exc_info.value
    assert observed["close_primary"] is semantic
    assert any(
        "post-validation generation cleanup failed: "
        "injected generation pin close failure" in note
        for note in semantic.__notes__
    )
    failure_path = (
        output_root
        / "failures"
        / VARIANT_ID
        / "nvbit"
        / "failure_provenance.json"
    )
    document = json.loads(failure_path.read_text())
    assert document["failure_code"] == "post_capture_validation_failed"
    assert document["detail"] == "synthetic semantic post-validation failure"
    _assert_failed_nvbit_phase_tool(document, output_root, tracer_payload)
    assert not (
        output_root / f"{VARIANT_ID}.nvbit.capture_receipt.json"
    ).exists()


def test_nvbit_runtime_preflight_failure_is_retained(
    tmp_path: Path, monkeypatch,
):
    preflight = _pinned_preflight(tmp_path)
    output_root = tmp_path / "capture-output"
    tracer, tracer_payload = _external_tracer(tmp_path)
    monkeypatch.setattr(
        capture,
        "validate_live_capture_preflight",
        lambda *, phase: _nvbit_live()
        if phase == "nvbit"
        else pytest.fail("wrong phase"),
    )
    monkeypatch.setattr(
        capture.nvbit_runtime,
        "retained_tracer_runtime_env",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            capture.nvbit_runtime.TracerRuntimeError(
                "synthetic runtime dependency refusal"
            )
        ),
    )
    monkeypatch.setattr(
        capture,
        "_run_capture_command",
        lambda *_args, **_kwargs: pytest.fail("runtime refusal must not launch"),
    )
    monkeypatch.setattr(capture, "_boot_id_sha256", lambda: "b" * 64)

    with pytest.raises(capture.CaptureError, match="runtime dependency refusal"):
        capture.capture_phase(
            matrix_key=MATRIX_KEY,
            variant_id=VARIANT_ID,
            phase="nvbit",
            preflight_path=preflight,
            output_root=output_root,
            nvbit_tracer_path=tracer,
            nvbit_tracer_sha256=capture.sha256_bytes(tracer.read_bytes()),
            **_runtime_manifest_kwargs(tracer),
        )

    failure = json.loads(
        (
            output_root
            / "failures"
            / VARIANT_ID
            / "nvbit"
            / "failure_provenance.json"
        ).read_text()
    )
    assert failure["process"]["pid"] is None
    assert failure["process"]["role"] == "nvbit_capture"
    _assert_failed_nvbit_phase_tool(
        failure,
        output_root,
        tracer_payload,
    )
    assert failure["process"]["command"][0].startswith(str(capture.ROOT))
    assert failure["process"]["environment"]["CUDA_INJECTION64_PATH"] == str(
        output_root
        / "capture_tools"
        / f"{VARIANT_ID}.nvbit_tracer_tool.so"
    )


def test_nvbit_process_failure_records_canonical_descriptor_authority(
    tmp_path: Path, monkeypatch,
):
    preflight = _pinned_preflight(tmp_path)
    output_root = tmp_path / "capture-output"
    tracer, tracer_payload = _external_tracer(tmp_path)
    observed = {}
    monkeypatch.setattr(
        capture,
        "validate_live_capture_preflight",
        lambda *, phase: _nvbit_live()
        if phase == "nvbit"
        else pytest.fail("wrong phase"),
    )
    monkeypatch.setattr(
        capture.nvbit_runtime,
        "retained_tracer_runtime_env",
        lambda _path, *, base_env, retained_descriptor, **_kwargs: (
            _retained_runtime(base_env, retained_descriptor)
        ),
    )

    def fail(command, environment, _cwd, **_kwargs):
        observed["command"] = command
        observed["environment"] = environment
        raise capture.CaptureError(
            "synthetic descriptor-bound timeout",
            process_id=4321,
            started_at_utc="2026-08-02T00:00:00Z",
            containment=_containment(),
        )

    monkeypatch.setattr(capture, "_run_capture_command", fail)
    monkeypatch.setattr(capture, "_boot_id_sha256", lambda: "b" * 64)

    with pytest.raises(capture.CaptureError, match="descriptor-bound timeout"):
        capture.capture_phase(
            matrix_key=MATRIX_KEY,
            variant_id=VARIANT_ID,
            phase="nvbit",
            preflight_path=preflight,
            output_root=output_root,
            nvbit_tracer_path=tracer,
            nvbit_tracer_sha256=capture.sha256_bytes(tracer.read_bytes()),
            **_runtime_manifest_kwargs(tracer),
        )

    failure = json.loads(
        (
            output_root
            / "failures"
            / VARIANT_ID
            / "nvbit"
            / "failure_provenance.json"
        ).read_text()
    )
    assert observed["command"][0].startswith("/proc/self/fd/")
    assert observed["environment"]["CUDA_INJECTION64_PATH"].startswith(
        "/proc/self/fd/"
    )
    assert failure["process"]["pid"] == 4321
    _assert_failed_nvbit_phase_tool(
        failure,
        output_root,
        tracer_payload,
    )
    assert failure["process"]["command"][0] != observed["command"][0]
    assert failure["process"]["environment"]["CUDA_INJECTION64_PATH"] == str(
        output_root
        / "capture_tools"
        / f"{VARIANT_ID}.nvbit_tracer_tool.so"
    )


def test_exclusive_writer_never_follows_preexisting_symlink(tmp_path: Path):
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"retained")
    destination = tmp_path / "receipt.json"
    destination.symlink_to(victim)

    with pytest.raises(capture.CaptureError, match="exclusively publish"):
        capture._write_new(destination, b"forged")

    assert victim.read_bytes() == b"retained"
    assert destination.is_symlink()


def test_preexisting_sibling_files_are_allowed_but_current_output_is_safe(tmp_path: Path):
    root = tmp_path / "output"
    root.mkdir()
    (root / "prior-phase").mkdir()
    (root / "prior-phase" / "prior.ncu-rep").write_bytes(b"prior")
    current = root / "nvbit" / "successor"
    capture._reject_preexisting_output_descendants(
        root, output_dir=current, reserved_files=(root / "preflight_nvbit.json",)
    )

    current.parent.mkdir(parents=True)
    current.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(capture.CaptureError, match="unsafe"):
        capture._reject_preexisting_output_descendants(
            root, output_dir=current, reserved_files=()
        )


def test_preexisting_output_files_are_rejected(tmp_path: Path):
    root = tmp_path / "output-file"
    root.mkdir()
    output_dir = root / "ncu"
    output_dir.write_bytes(b"stale")
    with pytest.raises(capture.CaptureError, match="unsafe"):
        capture._reject_preexisting_output_descendants(
            root, output_dir=output_dir, reserved_files=()
        )


def test_receipt_publication_rejects_artifact_generation_mutation(
    tmp_path: Path, monkeypatch,
):
    root = tmp_path / "capture"
    row, _roster_row = capture._row_for(MATRIX_KEY, VARIANT_ID)
    cell = root / capture.ARCH / row["successor_slug"] / "trace_00"
    cell.mkdir(parents=True)
    sidecar = cell / "label_0_0.json"
    sidecar.write_text(json.dumps({
        "unit_tagged_metrics": [
            {"ncu_metric": "gpc__cycles_elapsed.max"},
            {"ncu_metric": "gpu__time_duration.sum"},
        ],
    }))
    (cell / "run_0_launch_set.ncu-rep").write_bytes(b"report")
    run_manifest = {
        "schema_version": 1,
        "source_commit": capture.PINNED_COMMIT,
        "binary_sha256": row["static_launch_inventory"]["binary_sha256"],
        "kernel_signature_sha256": hashlib.sha256(
            (
                f"{row['successor_slug']}|"
                f"{row['static_launch_inventory']['binary_path']}"
            ).encode()
        ).hexdigest(),
        "kernel_signature_source": "fallback_legacy",
        "arch": capture.ARCH,
        "slug": row["successor_slug"],
        "shape_point": row["shape"]["shape_point"],
        "repeat_index": 0,
        "launch_index": 0,
    }
    (cell / "label_run_manifest_0_launch_set.json").write_text(
        json.dumps(run_manifest, sort_keys=True, indent=2) + "\n"
    )
    observation = {
        "schema_version": 1,
        "kind": "ncu_details_csv_launch_observation",
        "kernel_name_base": "mangled",
        "requested_launch_skip": 0,
        "requested_launch_count": 1,
        "report_sha256": capture.sha256_bytes(b"report"),
        "identity_columns": [
            "ID", "Device", "Kernel Name", "Block Size", "Grid Size",
        ],
        "identity": {
            "ID": "0",
            "Device": capture.GPU_NAME,
            "Kernel Name": "fixture",
            "Block Size": "(1, 1, 1)",
            "Grid Size": "(1, 1, 1)",
        },
    }
    observation["observation_sha256"] = capture.sha256_bytes(
        json.dumps(
            observation,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    )
    (cell / "run_0_launch_set.launch_observation.json").write_text(
        json.dumps(observation, sort_keys=True, indent=2) + "\n"
    )
    preflight_payload = b"{\"preflight\":true}\n"
    environment_payload = b"{\"environment\":true}\n"
    preflight = root / f"{VARIANT_ID}.preflight_ncu.json"
    environment = root / f"{VARIANT_ID}.environment_ncu.json"
    preflight.write_bytes(preflight_payload)
    environment.write_bytes(environment_payload)
    receipt = root / f"{VARIANT_ID}.ncu.capture_receipt.json"
    original_validate = capture.admission._validate_registered_receipt_schema

    def mutate_after_validation(*args, **kwargs):
        result = original_validate(*args, **kwargs)
        sidecar.write_text(json.dumps({
            "unit_tagged_metrics": [
                {"ncu_metric": "gpc__cycles_elapsed.max"},
            ],
        }))
        return result

    monkeypatch.setattr(
        capture.admission,
        "_sass_kernel_names",
        lambda _row: {"fixture"},
    )
    monkeypatch.setattr(
        capture.admission,
        "_validate_registered_receipt_schema",
        mutate_after_validation,
    )

    with contextlib.ExitStack() as source_stack:
        capture_source_retained = source_stack.enter_context(
            capture._open_bounded_regular(
                Path(capture.__file__),
                "capture producer",
                max_bytes=capture.CAPTURE_SOURCE_MAX_BYTES,
            )
        )
        containment_supervisor_retained = source_stack.enter_context(
            capture._open_bounded_regular(
                Path(capture.delegated_cgroup.__file__),
                "containment supervisor",
                max_bytes=capture.TOOL_SOURCE_MAX_BYTES,
            )
        )
        action_executor_retained = source_stack.enter_context(
            capture._open_bounded_regular(
                Path(capture.ncu_executor.__file__),
                "NCU action executor",
                max_bytes=capture.TOOL_SOURCE_MAX_BYTES,
            )
        )
        with pytest.raises(
            capture.CaptureError,
            match="transactionally publish capture receipt",
        ):
            capture._finalize_capture(
                root=root,
                phase="ncu",
                row=row,
                live={
                    "gpu_name": capture.GPU_NAME,
                    "smi_stdout_sha256": "a" * 64,
                    "gpu_uuid_sha256": "b" * 64,
                    "driver_version": capture.DRIVER_VERSION,
                    "identity_stdout_sha256": "c" * 64,
                },
                pid=1234,
                started="2026-08-02T00:00:00Z",
                containment=_containment(),
                command=[str(capture.LABEL_DRIVER), "synthetic"],
                capture_source=Path(capture.__file__),
                capture_source_retained=capture_source_retained,
                capture_producer_sha256=capture_source_retained.sha256,
                containment_supervisor_retained=(
                    containment_supervisor_retained
                ),
                containment_supervisor_sha256=(
                    containment_supervisor_retained.sha256
                ),
                phase_tool_sha256="b" * 64,
                action_executor_retained=action_executor_retained,
                action_executor_sha256=action_executor_retained.sha256,
                preflight_out=preflight,
                preflight_payload=preflight_payload,
                environment_out=environment,
                environment_payload=environment_payload,
                receipt_path=receipt,
            )

    assert not receipt.exists()
    retired = [
        path
        for path in receipt.parent.iterdir()
        if path.name.startswith(".failed-regular-")
        and not path.name.endswith(".salvaging")
    ]
    assert len(retired) == 1
    assert json.loads(retired[0].read_bytes())["receipt"]["status"] == "pass"


@pytest.mark.parametrize(
    "mutated_member",
    ("source", "copied", "source_manifest", "copied_manifest"),
)
def test_nvbit_publication_window_tracer_drift_retires_pass_receipt(
    tmp_path: Path,
    monkeypatch,
    mutated_member: str,
) -> None:
    root = tmp_path / "capture"
    root.mkdir()
    tracer, tracer_payload = _external_tracer(tmp_path)
    tracer_sha256 = capture.sha256_bytes(tracer_payload)
    authority = capture._require_nvbit_tracer_authority(
        tracer,
        tracer_sha256,
    )
    copied_path = (
        root
        / "capture_tools"
        / f"{VARIANT_ID}.nvbit_tracer_tool.so"
    )
    copied_path.parent.mkdir()
    copied_path.write_bytes(tracer_payload)
    authority_stack = contextlib.ExitStack()
    tracer_authority_stack = authority_stack.enter_context(
        contextlib.ExitStack()
    )
    source = tracer_authority_stack.enter_context(
        capture._retained_bounded_regular(
            tracer,
            "external NVBit tracer",
            max_bytes=capture.EXTERNAL_TRACER_MAX_BYTES,
        )
    )
    copied = tracer_authority_stack.enter_context(
        capture._retained_bounded_regular(
            copied_path,
            "retained NVBit tracer",
            max_bytes=capture.EXTERNAL_TRACER_MAX_BYTES,
        )
    )
    manifest_authority = _runtime_manifest_kwargs(tracer)
    source_manifest_path = manifest_authority["nvbit_runtime_manifest"]
    source_manifest = authority_stack.enter_context(
        capture._retained_bounded_regular(
            source_manifest_path,
            "external NVBit runtime manifest",
            max_bytes=capture.nvbit_runtime.RUNTIME_MANIFEST_MAX_BYTES,
        )
    )
    copied_manifest_path = (
        root
        / "capture_tools"
        / f"{VARIANT_ID}.nvbit_runtime_manifest.json"
    )
    copied_manifest_path.write_bytes(Path(source_manifest_path).read_bytes())
    copied_manifest = tracer_authority_stack.enter_context(
        capture._retained_bounded_regular(
            copied_manifest_path,
            "retained NVBit runtime manifest",
            max_bytes=capture.nvbit_runtime.RUNTIME_MANIFEST_MAX_BYTES,
        )
    )
    receipt = root / f"{VARIANT_ID}.nvbit.capture_receipt.json"
    receipt_payload = b'{"receipt":{"status":"pass"}}\n'
    generation_stack = contextlib.ExitStack()
    observed = {}
    mutated_path = {
        "source": tracer,
        "copied": copied_path,
        "source_manifest": source_manifest_path,
        "copied_manifest": copied_manifest_path,
    }[mutated_member]

    def mutate_during_descriptor_close():
        observed["descriptor_close_mutation_ran"] = True
        mutated_path.write_bytes(
            f"{mutated_member} drift after final publication rehash".encode()
        )

    mutation_stack = (
        authority_stack
        if mutated_member == "source_manifest"
        else tracer_authority_stack
    )
    mutation_stack.callback(mutate_during_descriptor_close)

    def finalized_generation(**kwargs):
        assert kwargs["generation_stack"] is generation_stack

        def observe_generation_cleanup():
            observed["generation_cleanup_ran"] = True

        generation_stack.callback(observe_generation_cleanup)
        return receipt, receipt_payload

    monkeypatch.setattr(
        capture,
        "_finalize_capture_in_generation",
        finalized_generation,
    )
    real_publish = capture.immutable.publish_regular_file

    def observe_after_commit(*args, **kwargs):
        after_commit = kwargs["after_commit"]

        def observe_primary():
            try:
                after_commit()
            except BaseException as exc:
                observed["after_commit_primary"] = exc
                raise

        kwargs["after_commit"] = observe_primary
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(
        capture.immutable,
        "publish_regular_file",
        observe_after_commit,
    )

    try:
        with pytest.raises(capture.CaptureError) as exc_info:
            capture._finalize_capture(
                generation_stack=generation_stack,
                authority_stack=authority_stack,
                root=root,
                phase="nvbit",
                row={},
                live={},
                pid=4321,
                started="2026-08-02T00:00:00Z",
                containment=_containment(),
                command=["synthetic-nvbit-capture"],
                capture_source=Path(capture.__file__),
                capture_source_retained=source,
                capture_producer_sha256=source.sha256,
                containment_supervisor_retained=source,
                containment_supervisor_sha256=source.sha256,
                phase_tool_sha256=tracer_sha256,
                action_executor_retained=None,
                action_executor_sha256=None,
                preflight_out=root / "preflight.json",
                preflight_payload=b"{}\n",
                environment_out=root / "environment.json",
                environment_payload=b"{}\n",
                receipt_path=receipt,
                retained_phase_tool_path=copied_path,
                retained_runtime_manifest_path=copied_manifest_path,
                runtime_manifest_sha256=copied_manifest.sha256,
                tracer_authority_stack=tracer_authority_stack,
                external_tracer_authority=authority,
                external_tracer_retained=source,
                external_runtime_manifest_retained=source_manifest,
                retained_phase_tool=copied,
                retained_runtime_manifest=copied_manifest,
            )
    finally:
        authority_stack.close()

    assert exc_info.value.code == "capture_generation_changed"
    assert "retained close failed" in exc_info.value.detail
    assert observed["generation_cleanup_ran"] is True
    assert observed["descriptor_close_mutation_ran"] is True
    assert observed["after_commit_primary"] is exc_info.value
    with pytest.raises(capture.immutable.ImmutableArtifactError):
        _ = source.descriptor
    with pytest.raises(capture.immutable.ImmutableArtifactError):
        _ = copied.descriptor
    with pytest.raises(capture.immutable.ImmutableArtifactError):
        _ = source_manifest.descriptor
    assert not receipt.exists()
    retired = [
        path
        for path in receipt.parent.iterdir()
        if path.name.startswith(".failed-regular-")
        and not path.name.endswith(".salvaging")
    ]
    assert len(retired) == 1
    assert retired[0].read_bytes() == receipt_payload


def test_nvbit_publication_stack_preserves_generation_cleanup_primary(
    tmp_path: Path,
) -> None:
    tracer, payload = _external_tracer(tmp_path)
    digest = capture.sha256_bytes(payload)
    authority = capture._require_nvbit_tracer_authority(tracer, digest)
    copied_path = tmp_path / "retained_capture_tool.so"
    copied_path.write_bytes(payload)
    tracer_stack = contextlib.ExitStack()
    source = tracer_stack.enter_context(
        capture._retained_bounded_regular(
            tracer,
            "external NVBit tracer",
            max_bytes=capture.EXTERNAL_TRACER_MAX_BYTES,
        )
    )
    copied = tracer_stack.enter_context(
        capture._retained_bounded_regular(
            copied_path,
            "retained NVBit tracer",
            max_bytes=capture.EXTERNAL_TRACER_MAX_BYTES,
        )
    )
    tracer_stack.callback(
        lambda: copied_path.write_bytes(b"copied drift during descriptor close")
    )
    generation_stack = contextlib.ExitStack()
    primary = capture.CaptureError(
        "synthetic generation cleanup primary",
        code="post_capture_validation_failed",
    )

    def fail_generation_cleanup():
        raise primary

    generation_stack.callback(fail_generation_cleanup)
    try:
        with pytest.raises(capture.CaptureError) as exc_info:
            capture._close_nvbit_publication_generation(
                generation_stack,
                tracer_stack,
                authority,
                source,
                copied,
                digest,
            )
    finally:
        tracer_stack.close()
        generation_stack.close()

    assert exc_info.value is primary
    assert any(
        "publication tracer descriptor close also failed" in note
        for note in primary.__notes__
    )
    with pytest.raises(capture.immutable.ImmutableArtifactError):
        _ = source.descriptor
    with pytest.raises(capture.immutable.ImmutableArtifactError):
        _ = copied.descriptor


def test_zero_size_artifacts_are_rejected(tmp_path: Path):
    root = tmp_path / "output"
    root.mkdir()
    empty = root / "empty.pb"
    empty.touch()
    with pytest.raises(capture.CaptureError, match="zero-sized"):
        capture._artifact(root, empty, "threadblock")


def _write_successor_nvbit_trace(trace_root: Path) -> None:
    from tools.nvbit_tracer.proto_binding_loader import (
        load_dynamic_trace_bindings,
    )
    from tools.nvbit_tracer.test_trace_admission import _write_trace

    trace_root.parent.mkdir(parents=True, exist_ok=True)
    _write_trace(trace_root)
    dynamic_path = trace_root / "dynamic_trace.pb"
    trace = load_dynamic_trace_bindings().trace_pb2.Trace()
    trace.ParseFromString(dynamic_path.read_bytes())
    trace.binary_version = capture.admission.NVBIT_TRACE_BINARY_VERSION
    trace.nvbit_version = capture.NVBIT_VERSION
    trace.trace_contract_id = capture.EXPECTED_TRACE_CONTRACT_ID
    del trace.memory_payload_classes[:]
    trace.memory_payload_classes.extend(
        capture.EXPECTED_MEMORY_PAYLOAD_CLASSES
    )
    kernel = trace.gpu_device[0].streams[0].kernels[0]
    row, _roster_row = capture._row_for(MATRIX_KEY, VARIANT_ID)
    kernel.name = f"{sorted(capture.admission._sass_kernel_names(row))[0]}___0"
    kernel.shared_memory_static_bytes = 0
    kernel.shared_memory_dynamic_bytes = 0
    dynamic_path.write_bytes(trace.SerializeToString())


@pytest.mark.parametrize(
    ("drift_stage", "expected_checkpoint"),
    (
        ("admission", "post-admission"),
        ("finalization", "post-finalization"),
    ),
)
def test_external_tracer_drift_during_admission_or_finalization_refuses_receipt(
    tmp_path: Path,
    monkeypatch,
    drift_stage: str,
    expected_checkpoint: str,
):
    preflight = _pinned_preflight(tmp_path)
    output_root = tmp_path / "capture-output"
    tracer, tracer_payload = _external_tracer(tmp_path)
    tracer_sha256 = capture.sha256_bytes(tracer_payload)

    monkeypatch.setattr(
        capture,
        "validate_live_capture_preflight",
        lambda *, phase: _nvbit_live()
        if phase == "nvbit"
        else pytest.fail("wrong phase"),
    )
    monkeypatch.setattr(
        capture.nvbit_runtime,
        "retained_tracer_runtime_env",
        lambda _path, *, base_env, retained_descriptor, **_kwargs: (
            _retained_runtime(base_env, retained_descriptor)
        ),
    )

    def run_successor_trace(_command, environment, _cwd, **_kwargs):
        trace_root = Path(os.readlink(environment["TRACES_FOLDER"]))
        staging = trace_root.with_name(f"{trace_root.name}.synthetic")
        _write_successor_nvbit_trace(staging)
        for child in staging.iterdir():
            child.replace(trace_root / child.name)
        staging.rmdir()
        return (
            0,
            4321,
            "2026-08-02T00:00:00Z",
            b"",
            b"",
            _containment(),
        )

    monkeypatch.setattr(capture, "_run_capture_command", run_successor_trace)
    monkeypatch.setattr(capture, "_boot_id_sha256", lambda: "b" * 64)

    if drift_stage == "admission":
        real_admit = capture._require_trace_admission_snapshot
        admission_calls = 0

        def admit_then_drift(*args, **kwargs):
            nonlocal admission_calls
            result = real_admit(*args, **kwargs)
            admission_calls += 1
            if admission_calls == 1:
                tracer.write_bytes(b"external tracer drifted during admission")
            return result

        monkeypatch.setattr(
            capture,
            "_require_trace_admission_snapshot",
            admit_then_drift,
        )
    else:
        def validate_receipt_then_drift(_receipt, _label):
            tracer.write_bytes(b"external tracer drifted during finalization")

        monkeypatch.setattr(
            capture.admission,
            "_validate_registered_receipt_schema",
            validate_receipt_then_drift,
        )

    with pytest.raises(capture.CaptureError) as exc_info:
        capture.capture_phase(
            matrix_key=MATRIX_KEY,
            variant_id=VARIANT_ID,
            phase="nvbit",
            preflight_path=preflight,
            output_root=output_root,
            nvbit_tracer_path=tracer,
            nvbit_tracer_sha256=tracer_sha256,
            **_runtime_manifest_kwargs(tracer),
        )

    assert exc_info.value.code == "capture_generation_changed"
    assert expected_checkpoint in exc_info.value.detail
    receipt = output_root / f"{VARIANT_ID}.nvbit.capture_receipt.json"
    assert not receipt.exists()
    failure_path = (
        output_root
        / "failures"
        / VARIANT_ID
        / "nvbit"
        / "failure_provenance.json"
    )
    failure = json.loads(failure_path.read_text())
    assert failure["failure_code"] == "capture_generation_changed"
    _assert_failed_nvbit_phase_tool(failure, output_root, tracer_payload)


def test_trace_summary_and_artifacts_share_retained_admitted_generation(
    tmp_path: Path,
    monkeypatch,
):
    row, _roster_row = capture._row_for(MATRIX_KEY, VARIANT_ID)
    root = tmp_path / "capture"
    trace_root = root / "nvbit" / row["successor_slug"]
    _write_successor_nvbit_trace(trace_root)
    monkeypatch.setattr(
        capture,
        "require_trace_admission",
        lambda *_args, **_kwargs: pytest.fail(
            "retained trace generation reopened its mutable root"
        ),
    )

    with contextlib.ExitStack() as generation_stack:
        (
            reader,
            snapshot,
            admitted,
            producer,
            summary_path,
            summary_payload,
        ) = capture._retain_nvbit_trace_generation(
            generation_stack=generation_stack,
            root=root,
            trace_root=trace_root,
            row=row,
            capture_stdout=b"",
            capture_stderr=b"",
        )
        summary = json.loads(summary_payload)
        assert summary_path.read_bytes() == summary_payload
        assert summary["trace_contract_id"] == producer["trace_contract_id"]
        assert summary["expected_record_count"] == admitted.expected_record_count
        assert summary["admitted_record_count"] == admitted.admitted_record_count
        assert summary["threadblock_count"] == admitted.threadblock_count
        artifact = capture._snapshot_artifact(
            root,
            trace_root,
            reader,
            dict(snapshot.files),
            "dynamic_trace.pb",
            "dynamic_trace",
        )
        assert artifact["sha256"] == admitted.dynamic_trace_sha256
        assert artifact["size_bytes"] == admitted.dynamic_trace_size_bytes


def test_same_validity_replacement_during_first_admission_is_rejected(
    tmp_path: Path,
    monkeypatch,
):
    row, _roster_row = capture._row_for(MATRIX_KEY, VARIANT_ID)
    root = tmp_path / "capture"
    trace_root = root / "nvbit" / row["successor_slug"]
    _write_successor_nvbit_trace(trace_root)
    dynamic_path = trace_root / "dynamic_trace.pb"
    real_admit = capture._require_trace_admission_snapshot
    admission_calls = 0

    def admit_then_replace(path: Path, **kwargs):
        nonlocal admission_calls
        admitted = real_admit(path, **kwargs)
        admission_calls += 1
        if admission_calls == 1:
            replacement = dynamic_path.with_name("same-validity.pb")
            replacement.write_bytes(dynamic_path.read_bytes())
            replacement.replace(dynamic_path)
        return admitted

    monkeypatch.setattr(
        capture,
        "_require_trace_admission_snapshot",
        admit_then_replace,
    )

    with pytest.raises(
        capture.CaptureError,
        match="producer-metadata validation failed.*generation changed",
    ) as exc_info:
        with contextlib.ExitStack() as generation_stack:
            capture._retain_nvbit_trace_generation(
                generation_stack=generation_stack,
                root=root,
                trace_root=trace_root,
                row=row,
                capture_stdout=b"",
                capture_stderr=b"",
            )

    assert exc_info.value.code == "capture_generation_changed"
    assert admission_calls == 1
    assert not (trace_root / "trace_summary.json").exists()
    assert capture.require_trace_admission(trace_root).admitted_record_count > 0


@pytest.mark.parametrize(
    "typed_reason",
    (
        "protobuf_decoded_resource_limit_exceeded",
        "trace_root_unparseable",
        "trace_snapshot_resource_unavailable",
    ),
)
def test_trace_admission_reason_is_preserved_by_capture_validation(
    tmp_path: Path,
    monkeypatch,
    typed_reason: str,
):
    def reject(_root):
        raise capture.TraceAdmissionError(typed_reason, "synthetic trace refusal")

    monkeypatch.setattr(capture, "require_trace_admission", reject)

    with pytest.raises(capture.CaptureError) as exc_info:
        capture._validate_nvbit_trace_generation(
            tmp_path,
            stage="adversarial",
        )

    assert exc_info.value.code == typed_reason
    assert "synthetic trace refusal" in exc_info.value.detail


def test_trace_validation_memoryerror_remains_resource_typed(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        capture,
        "require_trace_admission",
        lambda _root: (_ for _ in ()).throw(MemoryError("synthetic exhaustion")),
    )

    with pytest.raises(capture.CaptureError) as exc_info:
        capture._validate_nvbit_trace_generation(
            tmp_path,
            stage="adversarial",
        )

    assert exc_info.value.code == "trace_snapshot_resource_unavailable"


def test_trace_protobuf_byte_limit_precedes_wire_decode(
    tmp_path: Path,
    monkeypatch,
):
    (tmp_path / "dynamic_trace.pb").write_bytes(b"x" * 9)
    monkeypatch.setattr(capture, "_trace_path_max_bytes", lambda _path: 8)
    monkeypatch.setattr(
        capture,
        "preflight_trace_protobuf",
        lambda _payload: pytest.fail("oversized trace reached wire decode"),
    )

    with pytest.raises(capture.CaptureError) as exc_info:
        capture._validate_trace_producer(tmp_path)

    assert exc_info.value.code == "trace_snapshot_limit_exceeded"


@pytest.mark.parametrize(
    "failure",
    (
        MemoryError("synthetic trace snapshot exhaustion"),
        OSError(capture.errno.EMFILE, "synthetic trace EMFILE"),
        OSError(capture.errno.ENFILE, "synthetic trace ENFILE"),
        OSError(capture.errno.ENOMEM, "synthetic trace ENOMEM"),
    ),
)
def test_trace_protobuf_snapshot_resource_failures_are_typed(
    tmp_path: Path,
    monkeypatch,
    failure: BaseException,
):
    (tmp_path / "dynamic_trace.pb").write_bytes(b"trace")
    real_reader = capture.immutable.PinnedDirectoryReader

    class FailingSnapshotReader(real_reader):
        def read_snapshot(self, *_args, **_kwargs):
            raise failure

    monkeypatch.setattr(
        capture.immutable,
        "PinnedDirectoryReader",
        FailingSnapshotReader,
    )

    with pytest.raises(capture.CaptureError) as exc_info:
        capture._validate_trace_producer(tmp_path)

    assert exc_info.value.code == "trace_snapshot_resource_unavailable"


def test_trace_producer_version_mismatch_is_rejected(tmp_path: Path):
    trace_pb2 = capture._import_trace_pb2()
    trace = trace_pb2.Trace()
    trace.trace_contract_id = capture.EXPECTED_TRACE_CONTRACT_ID
    trace.memory_payload_classes.extend(capture.EXPECTED_MEMORY_PAYLOAD_CLASSES)
    trace.binary_version = 120
    trace.nvbit_version = capture.NVBIT_VERSION
    (tmp_path / "dynamic_trace.pb").write_bytes(trace.SerializeToString())
    with pytest.raises(capture.CaptureError, match="producer version mismatch"):
        capture._validate_trace_producer(tmp_path)


def test_stream_reader_bounds_each_ready_descriptor_to_one_read(monkeypatch):
    key = SimpleNamespace(data="stdout", fd=73, fileobj=object())

    class AlwaysReadySelector:
        def select(self, _timeout):
            return [(key, 1)]

        def unregister(self, _fileobj):
            pytest.fail("nonempty synthetic stream was unregistered")

    reads = 0

    def always_readable(_descriptor, size):
        nonlocal reads
        reads += 1
        if reads > 1:
            pytest.fail("one readiness event drained without a bounded return")
        return b"x" * size

    monkeypatch.setattr(capture.os, "read", always_readable)
    buffers = {"stdout": bytearray(), "stderr": bytearray()}

    exceeded, error, progressed = capture._read_capture_streams(
        AlwaysReadySelector(),
        buffers,
        timeout_s=0,
    )

    assert error is None
    assert progressed is True
    assert exceeded is False
    assert reads == 1
    assert buffers["stdout"] == b"x" * capture.CAPTURE_STREAM_READ_BYTES


def test_capture_timeout_is_bounded(tmp_path: Path):
    with pytest.raises(capture.CaptureError, match="timeout"):
        capture._run_capture_command(
            ["python", "-c", "import time; time.sleep(2)"],
            dict(capture.os.environ),
            tmp_path,
            timeout_s=0.05,
        )


def test_capture_timeout_retains_process_streams(tmp_path: Path):
    failure_root = tmp_path / "failure"
    command = [
        capture.sys.executable,
        "-c",
        (
            "import sys,time; "
            "sys.stdout.write('timeout stdout\\n'); sys.stdout.flush(); "
            "sys.stderr.write('timeout stderr\\n'); sys.stderr.flush(); "
            "time.sleep(2)"
        ),
    ]
    with pytest.raises(capture.CaptureError, match="timeout") as caught:
        capture._run_capture_command(
            command,
            dict(capture.os.environ),
            tmp_path,
            timeout_s=0.05,
            failure_root=failure_root,
        )
    assert caught.value.code == "capture_timeout"
    assert caught.value.stdout == b"timeout stdout\n"
    assert caught.value.stderr == b"timeout stderr\n"
    assert caught.value.process_id is not None
    assert caught.value.process_id > 0
    assert caught.value.started_at_utc is not None
    assert caught.value.started_at_utc.endswith("Z")
    assert (failure_root / "capture_timeout.stdout.bin").read_bytes() == (
        b"timeout stdout\n"
    )
    assert (failure_root / "capture_timeout.stderr.bin").read_bytes() == (
        b"timeout stderr\n"
    )


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_capture_streams_are_bounded_and_retained(
    tmp_path: Path, monkeypatch, stream: str,
):
    failure_root = tmp_path / "failure"
    monkeypatch.setattr(capture, "CAPTURE_STREAM_LIMIT_BYTES", 32)
    expression = (
        "sys.stdout.buffer.write(b'x' * 128)"
        if stream == "stdout"
        else "sys.stderr.buffer.write(b'x' * 128)"
    )
    with pytest.raises(
        capture.CaptureError,
        match="stdout/stderr byte limit",
    ) as caught:
        capture._run_capture_command(
            [capture.sys.executable, "-c", f"import sys; {expression}"],
            dict(capture.os.environ),
            tmp_path,
            timeout_s=2,
            failure_root=failure_root,
        )

    assert (
        failure_root / f"capture_output_limit.{stream}.bin"
    ).read_bytes() == b"x" * 32
    other = "stderr" if stream == "stdout" else "stdout"
    assert caught.value.code == "capture_output_limit"
    assert getattr(caught.value, stream) == b"x" * 32
    assert getattr(caught.value, other) == b""
    assert (
        failure_root / f"capture_output_limit.{other}.bin"
    ).read_bytes() == b""


def test_capture_watchdog_rejects_many_zero_byte_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    failure_root = tmp_path / "failure"
    trace_root = tmp_path / "trace"
    trace_root.mkdir()
    policy = dict(capture._capture_inventory_policy())
    policy["max_files"] = 16
    monkeypatch.setattr(
        capture,
        "_capture_inventory_policy",
        lambda: policy,
    )
    command = [
        capture.sys.executable,
        "-c",
        (
            "from pathlib import Path; import sys,time; "
            f"root=Path({str(trace_root)!r}); "
            "sys.stdout.write('inventory output'); sys.stdout.flush(); "
            "[(root / f'empty-{i:03d}').touch() for i in range(64)]; "
            "time.sleep(2)"
        ),
    ]

    with pytest.raises(
        capture.CaptureError,
        match="file-count limit exceeded",
    ) as caught:
        capture._run_capture_command(
            command,
            dict(capture.os.environ),
            tmp_path,
            timeout_s=2,
            byte_root=trace_root,
            byte_budget=1024,
            failure_root=failure_root,
        )

    assert caught.value.code == "capture_inventory_file_limit_exceeded"
    assert caught.value.stdout == b"inventory output"
    assert caught.value.stderr == b""
    assert (
        failure_root / "capture_inventory_error.stdout.bin"
    ).read_bytes() == b"inventory output"
    assert (
        failure_root / "capture_inventory_error.stderr.bin"
    ).read_bytes() == b""


def test_capture_watchdog_types_memory_exhaustion_and_retains_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    failure_root = tmp_path / "failure"
    trace_root = tmp_path / "trace"
    trace_root.mkdir()
    monkeypatch.setattr(
        capture,
        "_capture_tree_inventory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            MemoryError("injected inventory exhaustion")
        ),
    )

    with pytest.raises(
        capture.CaptureError,
        match="watchdog exhausted memory",
    ) as caught:
        capture._run_capture_command(
            [
                capture.sys.executable,
                "-c",
                (
                    "import sys,time; "
                    "sys.stderr.write('resource output'); sys.stderr.flush(); "
                    "time.sleep(2)"
                ),
            ],
            dict(capture.os.environ),
            tmp_path,
            timeout_s=2,
            byte_root=trace_root,
            byte_budget=1024,
            failure_root=failure_root,
        )

    assert caught.value.code == "capture_inventory_resource_unavailable"
    assert caught.value.stdout == b""
    assert caught.value.stderr == b"resource output"
    assert (
        failure_root / "capture_inventory_error.stdout.bin"
    ).read_bytes() == b""
    assert (
        failure_root / "capture_inventory_error.stderr.bin"
    ).read_bytes() == b"resource output"


def test_capture_owner_rejects_unexpected_watchdog_death(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    failure_root = tmp_path / "failure"

    class DeadWatchdog:
        def __init__(self, *, target, daemon):
            assert callable(target)
            assert daemon is True

        def start(self):
            return None

        def join(self, timeout=None):
            del timeout

        def is_alive(self):
            return False

    monkeypatch.setattr(capture.threading, "Thread", DeadWatchdog)

    with pytest.raises(
        capture.CaptureError,
        match="watchdog terminated unexpectedly",
    ) as caught:
        capture._run_capture_command(
            [
                capture.sys.executable,
                "-c",
                (
                    "import sys,time; "
                    "sys.stdout.write('watchdog evidence'); sys.stdout.flush(); "
                    "time.sleep(2)"
                ),
            ],
            dict(capture.os.environ),
            tmp_path,
            timeout_s=2,
            byte_root=tmp_path,
            byte_budget=1024,
            failure_root=failure_root,
        )

    assert caught.value.code == "capture_inventory_watchdog_terminated"
    assert caught.value.stdout == b"watchdog evidence"
    assert caught.value.stderr == b""
    assert (
        failure_root / "capture_inventory_error.stdout.bin"
    ).read_bytes() == b"watchdog evidence"


def test_transient_trace_budget_growth_remains_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    failure_root = tmp_path / "failure"
    scans = 0

    def transient_scan(_root: Path) -> int:
        nonlocal scans
        scans += 1
        return 64 if scans == 1 else 0

    monkeypatch.setattr(capture, "_capture_tree_bytes", transient_scan)

    with pytest.raises(capture.CaptureError, match="trace byte budget") as caught:
        capture._run_capture_command(
            [
                capture.sys.executable,
                "-c",
                (
                    "import sys,time; "
                    "sys.stderr.write('budget evidence'); sys.stderr.flush(); "
                    "time.sleep(2)"
                ),
            ],
            dict(capture.os.environ),
            tmp_path,
            timeout_s=2,
            byte_root=tmp_path,
            byte_budget=8,
            failure_root=failure_root,
        )

    assert scans >= 2
    assert caught.value.code == "capture_trace_byte_budget_exceeded"
    assert caught.value.stdout == b""
    assert caught.value.stderr == b"budget evidence"
    assert (
        failure_root / "capture_budget.stderr.bin"
    ).read_bytes() == b"budget evidence"


def test_capture_output_materialization_failure_preserves_exact_prefixes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    failure_root = tmp_path / "failure"
    original_copy = capture._copy_capture_stream_prefix
    original_popen = capture.subprocess.Popen
    observed = []

    def popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        observed.append(process)
        return process

    def bounded_copy(buffer: bytearray, length: int) -> bytes:
        if buffer.startswith(b"abcdefgh") and length > 4:
            raise MemoryError("synthetic capture materialization exhaustion")
        return original_copy(buffer, length)

    monkeypatch.setattr(capture.subprocess, "Popen", popen)
    monkeypatch.setattr(capture, "_copy_capture_stream_prefix", bounded_copy)

    with pytest.raises(
        capture.CaptureError,
        match="output resources are unavailable",
    ) as caught:
        capture._run_capture_command(
            [
                capture.sys.executable,
                "-c",
                "import os; os.write(1,b'abcdefgh'); os.write(2,b'stderr-evidence')",
            ],
            dict(capture.os.environ),
            tmp_path,
            timeout_s=2,
            failure_root=failure_root,
        )

    assert len(observed) == 1
    assert observed[0].poll() is not None
    assert observed[0].stdout.closed
    assert observed[0].stderr.closed
    assert caught.value.code == "capture_output_resource_unavailable"
    assert caught.value.stdout == b"abcd"
    assert caught.value.stderr == b"stderr-evidence"
    assert (
        failure_root / "capture_output_resource.stdout.bin"
    ).read_bytes() == b"abcd"
    assert (
        failure_root / "capture_output_resource.stderr.bin"
    ).read_bytes() == b"stderr-evidence"


def test_transient_wait_memoryerror_remains_typed_after_successful_reap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    failure_root = tmp_path / "failure"
    original_popen = capture.subprocess.Popen
    observed = []

    class TransientWaitFailure:
        def __init__(self, process):
            self.process = process
            self.wait_calls = 0

        def __getattr__(self, name):
            return getattr(self.process, name)

        def wait(self, timeout=None):
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise MemoryError("synthetic transient wait exhaustion")
            return self.process.wait(timeout=timeout)

    def popen(*args, **kwargs):
        process = TransientWaitFailure(original_popen(*args, **kwargs))
        observed.append(process)
        return process

    monkeypatch.setattr(capture.subprocess, "Popen", popen)

    with pytest.raises(
        capture.CaptureError,
        match="process wait resources are unavailable",
    ) as caught:
        capture._run_capture_command(
            [
                capture.sys.executable,
                "-c",
                "import os; os.write(1,b'wait evidence')",
            ],
            dict(capture.os.environ),
            tmp_path,
            timeout_s=2,
            failure_root=failure_root,
        )

    assert caught.value.code == "capture_process_resource_unavailable"
    assert caught.value.stdout == b"wait evidence"
    assert caught.value.stderr == b""
    assert observed[0].wait_calls == 2
    assert observed[0].process.poll() is not None
    assert (
        failure_root / "capture_process_resource.stdout.bin"
    ).read_bytes() == b"wait evidence"


def test_watchdog_death_during_join_is_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class JoinRaceWatchdog:
        def __init__(self, *, target, daemon):
            assert callable(target)
            assert daemon is True
            self.joined = False

        def start(self):
            return None

        def join(self, timeout=None):
            del timeout
            self.joined = True

        def is_alive(self):
            return not self.joined

    monkeypatch.setattr(capture.threading, "Thread", JoinRaceWatchdog)

    with pytest.raises(
        capture.CaptureError,
        match="watchdog terminated unexpectedly",
    ) as caught:
        capture._run_capture_command(
            [capture.sys.executable, "-c", "pass"],
            dict(capture.os.environ),
            tmp_path,
            timeout_s=2,
            byte_root=tmp_path,
            byte_budget=1024,
        )

    assert caught.value.code == "capture_inventory_watchdog_terminated"


def test_terminal_budget_failure_supersedes_operator_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    interruption = KeyboardInterrupt("synthetic concurrent interruption")
    monkeypatch.setattr(
        capture,
        "_process_exited_unreaped",
        lambda _process: (_ for _ in ()).throw(interruption),
    )
    monkeypatch.setattr(capture, "_capture_tree_bytes", lambda _root: 64)

    with pytest.raises(capture.CaptureError) as caught:
        capture._run_capture_command(
            [capture.sys.executable, "-c", "import time; time.sleep(2)"],
            dict(capture.os.environ),
            tmp_path,
            timeout_s=2,
            byte_root=tmp_path,
            byte_budget=8,
        )

    assert caught.value.code == "capture_trace_byte_budget_exceeded"
    assert any(
        "synthetic concurrent interruption" in note
        for note in caught.value.__notes__
    )


def test_failure_stream_publication_memoryerror_cannot_mask_resource_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    original_copy = capture._copy_capture_stream_prefix

    def bounded_copy(buffer: bytearray, length: int) -> bytes:
        if buffer.startswith(b"abcdefgh") and length > 4:
            raise MemoryError("synthetic capture materialization exhaustion")
        return original_copy(buffer, length)

    monkeypatch.setattr(capture, "_copy_capture_stream_prefix", bounded_copy)
    monkeypatch.setattr(
        capture,
        "_write_new",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            MemoryError("synthetic failure-stream publication exhaustion")
        ),
    )

    with pytest.raises(capture.CaptureError) as caught:
        capture._run_capture_command(
            [
                capture.sys.executable,
                "-c",
                "import os; os.write(1,b'abcdefgh'); os.write(2,b'stderr-evidence')",
            ],
            dict(capture.os.environ),
            tmp_path,
            timeout_s=2,
            failure_root=tmp_path / "failure",
        )

    assert caught.value.code == "capture_output_resource_unavailable"
    assert caught.value.stdout == b"abcd"
    assert caught.value.stderr == b"stderr-evidence"
    assert sum("failure-stream retention failed" in note for note in caught.value.__notes__) == 2


def test_trace_byte_budget_stops_capture(tmp_path: Path):
    command = [
        capture.sys.executable,
        "-c",
        f"from pathlib import Path; import time; Path({str(tmp_path / 'payload.bin')!r}).write_bytes(b'x' * 64); time.sleep(2)",
    ]
    with pytest.raises(capture.CaptureError, match="byte budget"):
        capture._run_capture_command(
            command,
            dict(capture.os.environ),
            tmp_path,
            timeout_s=2,
            byte_root=tmp_path,
            byte_budget=8,
        )


def test_stalled_successor_containment_creation_uses_capture_deadline(
    tmp_path: Path,
    monkeypatch,
):
    def stalled_create():
        time.sleep(60)
        raise AssertionError("capture deadline did not interrupt cgroup creation")

    monkeypatch.setattr(
        capture.delegated_cgroup.DelegatedCgroupV2,
        "create",
        stalled_create,
    )

    started = time.monotonic()
    with pytest.raises(capture.CaptureError) as caught:
        capture._run_capture_command(
            [capture.sys.executable, "-c", "pass"],
            dict(capture.os.environ),
            tmp_path,
            timeout_s=0.05,
            require_cgroup_containment=True,
        )

    assert caught.value.code == "capture_timeout"
    assert "during launch" in caught.value.detail
    assert time.monotonic() - started < 1


def test_stalled_successor_contained_popen_uses_capture_deadline(
    tmp_path: Path,
    monkeypatch,
):
    class StalledContainment:
        def __init__(self):
            self.closed = False

        def popen(self, _command, **_kwargs):
            time.sleep(60)
            raise AssertionError("capture deadline did not interrupt Popen")

        def close_after_error(self):
            self.closed = True

    containment = StalledContainment()
    monkeypatch.setattr(
        capture.delegated_cgroup.DelegatedCgroupV2,
        "create",
        lambda: containment,
    )

    started = time.monotonic()
    with pytest.raises(capture.CaptureError) as caught:
        capture._run_capture_command(
            [capture.sys.executable, "-c", "pass"],
            dict(capture.os.environ),
            tmp_path,
            timeout_s=0.05,
            require_cgroup_containment=True,
        )

    assert caught.value.code == "capture_timeout"
    assert containment.closed is True
    assert time.monotonic() - started < 1


@pytest.mark.parametrize("exception_type", (KeyboardInterrupt, SystemExit))
def test_successor_popen_interruption_closes_containment(
    tmp_path: Path,
    monkeypatch,
    exception_type,
):
    interruption = exception_type("synthetic successor Popen interruption")

    class Containment:
        def __init__(self):
            self.closed = False

        def popen(self, _command, **_kwargs):
            raise interruption

        def close_after_error(self):
            self.closed = True

    containment = Containment()
    monkeypatch.setattr(
        capture.delegated_cgroup.DelegatedCgroupV2,
        "create",
        lambda: containment,
    )

    with pytest.raises(exception_type) as caught:
        capture._run_capture_command(
            [capture.sys.executable, "-c", "pass"],
            dict(capture.os.environ),
            tmp_path,
            timeout_s=1,
            require_cgroup_containment=True,
        )

    assert caught.value is interruption
    assert containment.closed is True


def test_successor_post_popen_interruption_reaps_and_finishes(
    tmp_path: Path,
    monkeypatch,
):
    interruption = KeyboardInterrupt("synthetic post-Popen interruption")
    processes = []
    calls = []

    class InterruptAfterLaunch:
        def __enter__(self):
            return None

        def __exit__(self, exception_type, _exception, _traceback):
            if exception_type is None:
                raise interruption
            return False

    class Containment:
        def popen(self, command, **kwargs):
            process = capture.subprocess.Popen(command, **kwargs)
            processes.append(process)
            return process

        def kill_all(self):
            calls.append("kill_all")

        def finish(self):
            calls.append("finish")
            return _containment()

    monkeypatch.setattr(
        capture.nvbit_capture_process,
        "bounded_launch",
        lambda _deadline: InterruptAfterLaunch(),
    )
    monkeypatch.setattr(
        capture.delegated_cgroup.DelegatedCgroupV2,
        "create",
        Containment,
    )

    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            capture._run_capture_command(
                [capture.sys.executable, "-c", "import time; time.sleep(60)"],
                dict(capture.os.environ),
                tmp_path,
                timeout_s=1,
                require_cgroup_containment=True,
            )
        assert caught.value is interruption
        assert calls == ["kill_all", "finish"]
        assert len(processes) == 1 and processes[0].poll() is not None
        with pytest.raises(ChildProcessError):
            os.waitpid(processes[0].pid, os.WNOHANG)
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, capture.signal.SIGKILL)
                process.wait(timeout=2)


def test_successor_launch_cleanup_failure_supersedes_interruption(
    tmp_path: Path,
    monkeypatch,
):
    interruption = KeyboardInterrupt("synthetic post-Popen interruption")
    processes = []

    class InterruptAfterLaunch:
        def __enter__(self):
            return None

        def __exit__(self, exception_type, _exception, _traceback):
            if exception_type is None:
                raise interruption
            return False

    class FailingContainment:
        def popen(self, command, **kwargs):
            process = capture.subprocess.Popen(command, **kwargs)
            processes.append(process)
            return process

        def kill_all(self):
            raise RuntimeError("synthetic persistent cgroup.kill failure")

        def finish(self):
            raise RuntimeError("synthetic persistent cgroup finish failure")

    monkeypatch.setattr(
        capture.nvbit_capture_process,
        "bounded_launch",
        lambda _deadline: InterruptAfterLaunch(),
    )
    monkeypatch.setattr(
        capture.delegated_cgroup.DelegatedCgroupV2,
        "create",
        FailingContainment,
    )

    try:
        with pytest.raises(capture.CaptureError) as caught:
            capture._run_capture_command(
                [capture.sys.executable, "-c", "import time; time.sleep(60)"],
                dict(capture.os.environ),
                tmp_path,
                timeout_s=1,
                require_cgroup_containment=True,
            )
        assert caught.value.code == "capture_containment_cleanup_failed"
        assert "persistent cgroup" in caught.value.detail
        assert any(
            "post-Popen interruption" in note
            for note in caught.value.__notes__
        )
        assert len(processes) == 1 and processes[0].poll() is not None
        with pytest.raises(ChildProcessError):
            os.waitpid(processes[0].pid, os.WNOHANG)
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, capture.signal.SIGKILL)
                process.wait(timeout=2)


def test_capture_setup_cancellation_reaps_started_process(
    tmp_path: Path, monkeypatch,
):
    started = []
    original_popen = capture.subprocess.Popen

    def record_popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        started.append(process)
        return process

    interruption = KeyboardInterrupt("synthetic selector setup interruption")
    monkeypatch.setattr(capture.subprocess, "Popen", record_popen)
    monkeypatch.setattr(
        capture.selectors,
        "DefaultSelector",
        lambda: (_ for _ in ()).throw(interruption),
    )

    with pytest.raises(KeyboardInterrupt) as failure:
        capture._run_capture_command(
            [capture.sys.executable, "-c", "import time; time.sleep(2)"],
            dict(capture.os.environ),
            tmp_path,
            timeout_s=3,
        )

    assert failure.value is interruption
    assert len(started) == 1
    assert started[0].poll() is not None


def test_process_group_kill_cancellation_does_not_skip_cgroup_finish(
    tmp_path: Path, monkeypatch,
):
    calls = []
    interruption = KeyboardInterrupt("synthetic process-group kill interruption")

    class Containment:
        def popen(self, command, **kwargs):
            return capture.subprocess.Popen(command, **kwargs)

        def kill_all(self):
            calls.append("kill_all")

        def finish(self):
            calls.append("finish")
            return _containment()

    monkeypatch.setattr(
        capture.delegated_cgroup.DelegatedCgroupV2,
        "create",
        Containment,
    )
    monkeypatch.setattr(
        capture,
        "_kill_capture_process",
        lambda _process: (_ for _ in ()).throw(interruption),
    )

    with pytest.raises(KeyboardInterrupt) as failure:
        capture._run_capture_command(
            [capture.sys.executable, "-c", "pass"],
            dict(capture.os.environ),
            tmp_path,
            timeout_s=2,
            require_cgroup_containment=True,
        )

    assert failure.value is interruption
    assert "kill_all" in calls
    assert calls[-1] == "finish"


def test_process_wait_failure_supersedes_setup_cancellation(
    tmp_path: Path, monkeypatch,
):
    original_popen = capture.subprocess.Popen
    spawned = []

    class FailingWaitProcess:
        def __init__(self, process):
            self.process = process

        def __getattr__(self, name):
            return getattr(self.process, name)

        def wait(self, timeout=None):
            raise capture.subprocess.TimeoutExpired(self.args, timeout)

    def wrap_popen(*args, **kwargs):
        wrapped = FailingWaitProcess(original_popen(*args, **kwargs))
        spawned.append(wrapped)
        return wrapped

    monkeypatch.setattr(capture.subprocess, "Popen", wrap_popen)
    monkeypatch.setattr(
        capture.selectors,
        "DefaultSelector",
        lambda: (_ for _ in ()).throw(
            KeyboardInterrupt("synthetic selector setup interruption")
        ),
    )

    try:
        with pytest.raises(
            capture.CaptureError,
            match="capture process cleanup failed",
        ) as failure:
            capture._run_capture_command(
                [capture.sys.executable, "-c", "import time; time.sleep(2)"],
                dict(capture.os.environ),
                tmp_path,
                timeout_s=3,
            )
        assert any(
            "selector setup interruption" in note
            for note in failure.value.__notes__
        )
    finally:
        if spawned:
            spawned[0].process.wait(timeout=2)


def test_watchdog_join_is_bounded_and_typed(tmp_path: Path, monkeypatch):
    joins = []

    class StuckWatchdog:
        def __init__(self, *, target, daemon):
            assert callable(target)
            assert daemon is True

        def start(self):
            return None

        def join(self, timeout=None):
            joins.append(timeout)

        def is_alive(self):
            return True

    monkeypatch.setattr(capture.threading, "Thread", StuckWatchdog)
    monkeypatch.setattr(capture, "CAPTURE_CLEANUP_TIMEOUT_S", 0.01)

    with pytest.raises(
        capture.CaptureError,
        match="byte-budget watchdog did not stop",
    ) as caught:
        capture._run_capture_command(
            [capture.sys.executable, "-c", "pass"],
            dict(capture.os.environ),
            tmp_path,
            timeout_s=2,
        )

    assert caught.value.code == "capture_process_cleanup_failed"
    assert caught.value.stdout == b""
    assert caught.value.stderr == b""
    assert joins == [0.01]


def test_watchdog_is_joined_before_process_authority_release(
    tmp_path: Path, monkeypatch,
):
    entered = capture.threading.Event()
    release = capture.threading.Event()
    kill_threads: list[str] = []
    result: list[tuple[int, int, str, bytes, bytes]] = []
    errors: list[BaseException] = []
    original_kill = capture._kill_capture_process

    def blocked_scan(_root: Path) -> int:
        entered.set()
        if not release.wait(2):
            raise OSError("synthetic blocked watchdog")
        return 0

    def record_kill(process) -> None:
        kill_threads.append(capture.threading.current_thread().name)
        original_kill(process)

    def run_capture() -> None:
        try:
            result.append(capture._run_capture_command(
                [
                    capture.sys.executable,
                    "-c",
                    "import time; time.sleep(0.15)",
                ],
                dict(capture.os.environ),
                tmp_path,
                timeout_s=2,
                byte_root=tmp_path,
                byte_budget=1024,
            ))
        except BaseException as exc:  # surfaced in the owner test thread
            errors.append(exc)

    monkeypatch.setattr(capture, "_capture_tree_bytes", blocked_scan)
    monkeypatch.setattr(capture, "_kill_capture_process", record_kill)
    monkeypatch.setattr(
        capture.nvbit_capture_process,
        "bounded_launch",
        lambda _deadline: contextlib.nullcontext(),
    )
    owner = capture.threading.Thread(
        target=run_capture, name="capture-owner-test",
    )
    owner.start()
    try:
        assert entered.wait(1)
        time.sleep(0.1)
        assert owner.is_alive()
    finally:
        release.set()
        owner.join(2)

    assert not owner.is_alive()
    assert errors == []
    assert result and result[0][0] == 0
    assert kill_threads
    assert set(kill_threads) == {"capture-owner-test"}


def test_unexpected_capture_monitor_failure_is_not_mistyped_as_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    failure_root = tmp_path / "failure"
    monkeypatch.setattr(
        capture,
        "_process_exited_unreaped",
        lambda _process: (_ for _ in ()).throw(
            RuntimeError("injected process monitor failure")
        ),
    )

    with pytest.raises(
        capture.CaptureError,
        match="capture process monitor failed",
    ) as caught:
        capture._run_capture_command(
            [
                capture.sys.executable,
                "-c",
                (
                    "import sys,time; "
                    "sys.stdout.write('monitor output'); sys.stdout.flush(); "
                    "time.sleep(2)"
                ),
            ],
            dict(capture.os.environ),
            tmp_path,
            timeout_s=2,
            failure_root=failure_root,
        )

    assert caught.value.code == "capture_monitor_failed"
    assert caught.value.stdout == b"monitor output"
    assert caught.value.stderr == b""
    assert (
        failure_root / "capture_monitor_error.stdout.bin"
    ).read_bytes() == b"monitor output"


def test_terminal_trace_byte_budget_rejects_fast_final_burst(tmp_path: Path):
    command = [
        capture.sys.executable,
        "-c",
        (
            "from pathlib import Path; "
            f"Path({str(tmp_path / 'payload.bin')!r}).write_bytes(b'x' * 64)"
        ),
    ]
    with pytest.raises(capture.CaptureError, match="byte budget") as caught:
        capture._run_capture_command(
            command,
            dict(capture.os.environ),
            tmp_path,
            timeout_s=2,
            byte_root=tmp_path,
            byte_budget=8,
        )
    assert caught.value.process_id is not None
    assert caught.value.started_at_utc is not None


def test_post_processed_trace_tree_must_remain_within_budget(tmp_path: Path):
    (tmp_path / "raw-trace.bin").write_bytes(b"12345678")
    capture._require_capture_tree_budget(tmp_path, 8)
    (tmp_path / "trace_summary.json").write_bytes(b"x")

    with pytest.raises(capture.CaptureError, match="post-processed"):
        capture._require_capture_tree_budget(tmp_path, 8)


def test_terminal_trace_scan_error_retains_process_and_streams(
    tmp_path: Path, monkeypatch,
):
    failure_root = tmp_path / "failure"
    monkeypatch.setattr(
        capture,
        "_capture_tree_bytes",
        lambda _root: (_ for _ in ()).throw(OSError("synthetic stat race")),
    )
    with pytest.raises(capture.CaptureError, match="unsafe output") as caught:
        capture._run_capture_command(
            [
                capture.sys.executable,
                "-c",
                "import sys; sys.stdout.write('completed output')",
            ],
            dict(capture.os.environ),
            tmp_path,
            timeout_s=2,
            byte_root=tmp_path,
            byte_budget=1024,
            failure_root=failure_root,
        )

    assert caught.value.code == "capture_inventory_io_failed"
    assert caught.value.stdout == b"completed output"
    assert caught.value.stderr == b""
    assert caught.value.process_id is not None
    assert caught.value.process_id > 0
    assert caught.value.started_at_utc is not None
    assert (
        failure_root / "capture_inventory_error.stdout.bin"
    ).read_bytes() == b"completed output"


def test_containment_cleanup_failure_supersedes_base_exception(
    tmp_path: Path, monkeypatch,
):
    class FailingContainment:
        def popen(self, command, **kwargs):
            return capture.subprocess.Popen(command, **kwargs)

        def kill_all(self):
            raise capture.delegated_cgroup.CgroupContainmentError(
                "synthetic cgroup.kill failure"
            )

        def finish(self):
            raise capture.delegated_cgroup.CgroupContainmentError(
                "synthetic terminal cleanup failure"
            )

    containment = FailingContainment()
    monkeypatch.setattr(
        capture.delegated_cgroup.DelegatedCgroupV2,
        "create",
        lambda: containment,
    )
    calls = 0

    def interrupt_once(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt("synthetic interruption")
        return False, None, False

    monkeypatch.setattr(capture, "_read_capture_streams", interrupt_once)

    with pytest.raises(
        capture.CaptureError,
        match="containment cleanup failed.*synthetic terminal cleanup failure",
    ) as caught:
        capture._run_capture_command(
            [capture.sys.executable, "-c", "import time; time.sleep(2)"],
            dict(capture.os.environ),
            tmp_path,
            timeout_s=3,
            require_cgroup_containment=True,
        )

    assert caught.value.code == "capture_containment_cleanup_failed"
    assert caught.value.stdout == b""
    assert caught.value.stderr == b""
    assert any(
        "synthetic interruption" in note
        for note in getattr(caught.value, "__notes__", ())
    )


@pytest.mark.parametrize("escape_call", ["setsid", "setpgid"])
def test_cgroup_containment_kills_group_escape_descendant(
    tmp_path: Path, escape_call: str,
):
    if capture.os.environ.get("GPU_SIM_DELEGATED_CGROUP_TEST") != "1":
        pytest.skip("requires a systemd-delegated cgroup-v2 test service")
    ready = tmp_path / f"{escape_call}-ready.txt"
    marker = tmp_path / "late-write.bin"
    escape_source = (
        "os.setsid()" if escape_call == "setsid" else "os.setpgid(0, 0)"
    )
    child_source = (
        "import os,time\n"
        "from pathlib import Path\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        f"    {escape_source}\n"
        f"    Path({str(ready)!r}).write_text(f'{{os.getpid()}},{{os.getsid(0)}},{{os.getpgrp()}},{{Path(\"/proc/self/cgroup\").read_text().strip()}}')\n"
        "    os.close(1)\n"
        "    os.close(2)\n"
        "    time.sleep(0.4)\n"
        f"    Path({str(marker)!r}).write_bytes(b'late')\n"
        "    os._exit(0)\n"
        "deadline = time.monotonic() + 2\n"
        f"while not Path({str(ready)!r}).is_file() and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
        f"os._exit(0 if Path({str(ready)!r}).is_file() else 9)\n"
    )

    result = capture._run_capture_command(
        [capture.sys.executable, "-c", child_source],
        dict(capture.os.environ),
        tmp_path,
        timeout_s=3,
        require_cgroup_containment=True,
    )

    assert result[0] == 0
    ready_fields = ready.read_text().split(",", maxsplit=3)
    descendant_pid, descendant_sid, descendant_pgid = [
        int(value) for value in ready_fields[:3]
    ]
    descendant_cgroup = ready_fields[3]
    assert descendant_cgroup.startswith("0::/")
    assert descendant_cgroup.rsplit("/", maxsplit=1)[-1].startswith("gpu-sim-")
    assert descendant_pgid == descendant_pid
    if escape_call == "setsid":
        assert descendant_sid == descendant_pid
    else:
        assert descendant_sid != descendant_pid
    assert result[5] is not None
    assert result[5].as_dict()["terminal_populated_zero"] is True
    time.sleep(0.5)
    assert not marker.exists()


def test_successful_capture_kills_pipe_detached_descendants(tmp_path: Path):
    marker = tmp_path / "late-write.bin"
    command = [
        capture.sys.executable,
        "-c",
        (
            "import os,time\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    os.close(1)\n"
            "    os.close(2)\n"
            "    time.sleep(0.25)\n"
            f"    open({str(marker)!r}, 'wb').write(b'late')\n"
            "    os._exit(0)\n"
            "os._exit(0)\n"
        ),
    ]

    result = capture._run_capture_command(
        command,
        dict(capture.os.environ),
        tmp_path,
        timeout_s=2,
    )

    assert result[0] == 0
    time.sleep(0.4)
    assert not marker.exists()


def _ncu_authority() -> dict:
    return {
        "status": "ok",
        "expected": capture.NCU_VERSION,
        "execution_mode": capture.campaign_preflight.NCU_EXECUTION_MODE,
        "host_machine": capture.campaign_preflight.NCU_HOST_MACHINE,
        "launcher_resolved_path": capture.admission.NCU_LAUNCHER_PATH,
        "launcher_sha256": capture.admission.NCU_LAUNCHER_SHA256,
        "launcher_target_relpath": (
            capture.campaign_preflight.NCU_LAUNCHER_TARGET_RELPATH
        ),
        "resolved_path": capture.admission.NCU_EXECUTABLE_PATH,
        "executable_sha256": capture.admission.NCU_EXECUTABLE_SHA256,
        "version_line": capture.admission.NCU_VERSION_LINE,
    }


def _ncu_command(identity: capture.ncu_executor.ExpectedNcuIdentity) -> list[str]:
    return [
        str(capture.LABEL_DRIVER),
        "--binary", "/fixture/target",
        "--ncu", identity.resolved_path,
        "--ncu-executable-sha256", identity.executable_sha256,
        "--ncu-version-line", identity.version_line,
        "--target-processes", "all",
        "--filter-mode", "global",
        "--replay-mode", "kernel",
        "--cache-control", "all",
        "--clock-control", "none",
        "--profile-from-start", "on",
        "--launch-skip", "0",
        "--launch-count", "1",
        "--",
        "128",
    ]


def _retained_target():
    return SimpleNamespace(
        path=Path("/fixture/target"),
        proc_path="/proc/self/fd/91",
        descriptor=91,
    )


def _ncu_handoff_manifest(row: dict) -> dict:
    return {
        "schema_version": 1,
        "source_commit": capture.PINNED_COMMIT,
        "binary_sha256": row["static_launch_inventory"]["binary_sha256"],
        "kernel_signature_sha256": hashlib.sha256(
            (
                f"{row['successor_slug']}|"
                f"{row['static_launch_inventory']['binary_path']}"
            ).encode()
        ).hexdigest(),
        "kernel_signature_source": "fallback_legacy",
        "arch": capture.ARCH,
        "slug": row["successor_slug"],
        "shape_point": row["shape"]["shape_point"],
        "repeat_index": 0,
        "launch_index": 0,
    }


def _ncu_details_csv(
    row: dict,
    kernel_name: str = "_Zfixture_kernel",
    *,
    block: list[int] | None = None,
    grid: list[int] | None = None,
) -> str:
    launch = row["shape"]["trace_shape"]["launch_config"]
    block_text = ",".join(str(value) for value in (block or launch["block"]))
    grid_text = ",".join(str(value) for value in (grid or launch["grid"]))
    prefix = (
        f'0,{capture.GPU_NAME},{kernel_name},"{block_text}","{grid_text}"'
    )
    return (
        "ID,Device,Kernel Name,Block Size,Grid Size,Metric Name,Metric Value\n"
        f"{prefix},lts__t_sector_hit_rate.pct,50\n"
        f"{prefix},l1tex__t_sector_hit_rate.pct,25\n"
        f"{prefix},gpc__cycles_elapsed.max,123\n"
        f"{prefix},gpu__time_duration.sum,456\n"
    )


def _write_ncu_handoff(root: Path, row: dict) -> tuple[Path, bytes]:
    cell = root / capture.ARCH / row["successor_slug"] / "trace_00"
    cell.mkdir(parents=True)
    report_payload = b"synthetic retained NCU report"
    (cell / "run_0_launch_set.ncu-rep").write_bytes(report_payload)
    (cell / "label_run_manifest_0_launch_set.json").write_text(
        json.dumps(_ncu_handoff_manifest(row), sort_keys=True, indent=2) + "\n"
    )
    return cell, report_payload


def test_ncu_manifest_byte_limit_precedes_report_import(
    tmp_path: Path,
    monkeypatch,
):
    row, _roster = capture._row_for(MATRIX_KEY, VARIANT_ID)
    root = tmp_path / "capture"
    cell, _report_payload = _write_ncu_handoff(root, row)
    manifest = cell / "label_run_manifest_0_launch_set.json"
    manifest.write_bytes(b"{" + b"x" * capture.NCU_MANIFEST_MAX_BYTES + b"}")
    monkeypatch.setattr(
        capture.parse_ncu_metrics,
        "_import_retained_report_csv",
        lambda *_args, **_kwargs: pytest.fail(
            "oversized manifest reached report import"
        ),
    )

    with pytest.raises(capture.CaptureError) as exc_info:
        capture._postprocess_ncu_handoff(
            root=root,
            row=row,
            live=_nvbit_live(),
            expected_ncu_identity=capture._ncu_identity_from_authority(
                _ncu_authority()
            ),
        )

    assert exc_info.value.code == "capture_input_limit_exceeded"


@pytest.mark.parametrize(
    "failure",
    (
        MemoryError("synthetic NCU evidence exhaustion"),
        OSError(capture.errno.EMFILE, "synthetic NCU EMFILE"),
        OSError(capture.errno.ENFILE, "synthetic NCU ENFILE"),
        OSError(capture.errno.ENOMEM, "synthetic NCU ENOMEM"),
    ),
)
def test_ncu_report_retained_open_resource_failures_are_typed(
    tmp_path: Path,
    monkeypatch,
    failure: BaseException,
):
    row, _roster = capture._row_for(MATRIX_KEY, VARIANT_ID)
    root = tmp_path / "capture"
    cell, _report_payload = _write_ncu_handoff(root, row)
    report = cell / "run_0_launch_set.ncu-rep"
    real_open = capture.immutable.open_retained_regular

    def fail_report(path: Path, **kwargs):
        if Path(path) == report:
            raise failure
        return real_open(path, **kwargs)

    monkeypatch.setattr(
        capture.immutable,
        "open_retained_regular",
        fail_report,
    )

    with pytest.raises(capture.CaptureError) as exc_info:
        capture._postprocess_ncu_handoff(
            root=root,
            row=row,
            live=_nvbit_live(),
            expected_ncu_identity=capture._ncu_identity_from_authority(
                _ncu_authority()
            ),
        )

    assert exc_info.value.code == "capture_input_resource_unavailable"


def test_ncu_report_byte_limit_precedes_import(
    tmp_path: Path,
    monkeypatch,
):
    row, _roster = capture._row_for(MATRIX_KEY, VARIANT_ID)
    root = tmp_path / "capture"
    _cell, _report_payload = _write_ncu_handoff(root, row)
    monkeypatch.setattr(capture.parse_ncu_metrics, "NCU_REPORT_MAX_BYTES", 8)
    monkeypatch.setattr(
        capture.parse_ncu_metrics,
        "_import_retained_report_csv",
        lambda *_args, **_kwargs: pytest.fail(
            "oversized report reached descriptor import"
        ),
    )

    with pytest.raises(capture.CaptureError) as exc_info:
        capture._postprocess_ncu_handoff(
            root=root,
            row=row,
            live=_nvbit_live(),
            expected_ncu_identity=capture._ncu_identity_from_authority(
                _ncu_authority()
            ),
        )

    assert exc_info.value.code == "capture_input_limit_exceeded"


@pytest.mark.parametrize(
    "failure",
    (
        MemoryError("synthetic namespace exhaustion"),
        OSError(capture.errno.EMFILE, "synthetic namespace EMFILE"),
        OSError(capture.errno.ENFILE, "synthetic namespace ENFILE"),
        OSError(capture.errno.ENOMEM, "synthetic namespace ENOMEM"),
    ),
)
def test_ncu_namespace_resource_failures_are_typed(
    tmp_path: Path,
    monkeypatch,
    failure: BaseException,
):
    row, _roster = capture._row_for(MATRIX_KEY, VARIANT_ID)
    root = tmp_path / "capture"
    cell, _report_payload = _write_ncu_handoff(root, row)
    with capture.immutable.PinnedDirectoryReader(cell) as parent:
        with monkeypatch.context() as scoped:
            scoped.setattr(
                capture.os,
                "scandir",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
            )
            with pytest.raises(capture.CaptureError) as exc_info:
                capture._require_ncu_cell_namespace(
                    parent,
                    expected_names=capture._NCU_INITIAL_ARTIFACT_NAMES,
                )

    assert exc_info.value.code == "capture_namespace_resource_unavailable"


def test_ncu_report_path_substitution_after_retention_is_generation_typed(
    tmp_path: Path,
    monkeypatch,
):
    row, _roster = capture._row_for(MATRIX_KEY, VARIANT_ID)
    root = tmp_path / "capture"
    cell, report_payload = _write_ncu_handoff(root, row)
    report = cell / "run_0_launch_set.ncu-rep"

    def substitute_after_retention(_expected, retained_report, **_kwargs):
        assert retained_report.path == report
        assert retained_report.sha256 == capture.sha256_bytes(report_payload)
        replacement = report.with_name("replacement.ncu-rep")
        replacement.write_bytes(b"post-open replacement")
        replacement.replace(report)
        return (
            capture.parse_ncu_metrics.ImportedReportCsv(
                text=_ncu_details_csv(row),
                report_sha256=retained_report.sha256,
                report_size_bytes=retained_report.size_bytes,
            ),
            None,
        )

    monkeypatch.setattr(
        capture.parse_ncu_metrics,
        "_import_retained_report_csv",
        substitute_after_retention,
    )
    monkeypatch.setattr(
        capture.admission,
        "_sass_kernel_names",
        lambda _row: {"_Zfixture_kernel"},
    )

    with pytest.raises(capture.CaptureError) as exc_info:
        capture._postprocess_ncu_handoff(
            root=root,
            row=row,
            live=_nvbit_live(),
            expected_ncu_identity=capture._ncu_identity_from_authority(
                _ncu_authority()
            ),
        )

    assert exc_info.value.code == "capture_generation_changed"
    assert not (cell / "run_0_launch_set.launch_observation.json").exists()


def test_ncu_handoff_rejects_duplicate_manifest_keys(
    tmp_path: Path,
    monkeypatch,
):
    row, _roster = capture._row_for(MATRIX_KEY, VARIANT_ID)
    root = tmp_path / "capture"
    cell, _report_payload = _write_ncu_handoff(root, row)
    manifest = cell / "label_run_manifest_0_launch_set.json"
    payload = manifest.read_text()
    manifest.write_text(
        payload.replace(
            '"source_commit":',
            f'"source_commit": "{capture.PINNED_COMMIT}",\n  "source_commit":',
            1,
        )
    )
    monkeypatch.setattr(
        capture.parse_ncu_metrics,
        "_import_retained_report_csv",
        lambda *_args, **_kwargs: pytest.fail(
            "duplicate manifest must fail before report import"
        ),
    )

    with pytest.raises(capture.CaptureError, match="duplicate JSON key"):
        capture._postprocess_ncu_handoff(
            root=root,
            row=row,
            live=_nvbit_live(),
            expected_ncu_identity=capture._ncu_identity_from_authority(
                _ncu_authority()
            ),
        )


def test_failed_ncu_handoff_inventory_binds_retained_artifacts(
    tmp_path: Path,
    monkeypatch,
):
    row, _roster = capture._row_for(MATRIX_KEY, VARIANT_ID)
    root = tmp_path / "capture"
    cell, report_payload = _write_ncu_handoff(root, row)
    failure_root = root / "failure"
    failure_root.mkdir()
    monkeypatch.setattr(capture, "_boot_id_sha256", lambda: "b" * 64)

    capture._write_failure_provenance(
        failure_root,
        root=root,
        phase="ncu",
        row=row,
        command=["/fixture/target"],
        live=_nvbit_live(),
        environment_payload=b"{}\n",
        default_environment={"LANG": "C"},
        capture_producer_sha256="c" * 64,
        containment_supervisor_sha256="d" * 64,
        phase_tool_sha256="e" * 64,
        runtime_manifest_sha256=None,
        action_executor_sha256="f" * 64,
        error=capture.CaptureError("synthetic postprocess failure"),
    )

    document = json.loads((failure_root / "failure_provenance.json").read_text())
    artifacts = {
        artifact["role"]: artifact
        for artifact in document["handoff_artifacts"]
    }
    assert artifacts == {
        "label_run_manifest": {
            "role": "label_run_manifest",
            "path": (
                cell / "label_run_manifest_0_launch_set.json"
            ).relative_to(root).as_posix(),
            "sha256": capture.sha256_bytes(
                (cell / "label_run_manifest_0_launch_set.json").read_bytes()
            ),
            "size_bytes": (
                cell / "label_run_manifest_0_launch_set.json"
            ).stat().st_size,
        },
        "ncu_report": {
            "role": "ncu_report",
            "path": (
                cell / "run_0_launch_set.ncu-rep"
            ).relative_to(root).as_posix(),
            "sha256": capture.sha256_bytes(report_payload),
            "size_bytes": len(report_payload),
        },
    }


def test_failed_ncu_handoff_mutation_retires_failure_provenance(
    tmp_path: Path,
    monkeypatch,
):
    row, _roster = capture._row_for(MATRIX_KEY, VARIANT_ID)
    root = tmp_path / "capture"
    cell, _report_payload = _write_ncu_handoff(root, row)
    failure_root = root / "failure"
    failure_root.mkdir()
    report = cell / "run_0_launch_set.ncu-rep"
    real_publish = capture.immutable.publish_regular_file

    def publish_with_handoff_mutation(*args, **kwargs):
        after_commit = kwargs["after_commit"]

        def mutate_then_close():
            replacement = report.with_name("replacement.ncu-rep")
            replacement.write_bytes(report.read_bytes())
            replacement.replace(report)
            after_commit()

        kwargs["after_commit"] = mutate_then_close
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(
        capture.immutable,
        "publish_regular_file",
        publish_with_handoff_mutation,
    )
    monkeypatch.setattr(capture, "_boot_id_sha256", lambda: "b" * 64)

    with pytest.raises(
        capture.CaptureError,
        match="transactionally publish failure provenance",
    ):
        capture._write_failure_provenance(
            failure_root,
            root=root,
            phase="ncu",
            row=row,
            command=["/fixture/target"],
            live=_nvbit_live(),
            environment_payload=b"{}\n",
            default_environment={"LANG": "C"},
            capture_producer_sha256="c" * 64,
            containment_supervisor_sha256="d" * 64,
            phase_tool_sha256="e" * 64,
            runtime_manifest_sha256=None,
            action_executor_sha256="f" * 64,
            error=capture.CaptureError("synthetic postprocess failure"),
        )

    assert not (failure_root / "failure_provenance.json").exists()
    retired = [
        path for path in failure_root.iterdir()
        if path.name.startswith(".failed-regular-")
    ]
    assert len(retired) == 1


def test_late_failure_stream_retires_failure_provenance(
    tmp_path: Path,
    monkeypatch,
):
    row, _roster = capture._row_for(MATRIX_KEY, VARIANT_ID)
    root = tmp_path / "capture"
    failure_root = root / "failure"
    failure_root.mkdir(parents=True)
    initial_stream = failure_root / "initial.stdout.bin"
    initial_stream.write_bytes(b"initial")
    provenance = failure_root / "failure_provenance.json"
    real_verify = capture.immutable.RetainedRegular.verify_unchanged

    def verify_with_late_stream(retained):
        if retained.path == initial_stream and provenance.exists():
            late_stream = failure_root / "late.stderr.bin"
            if not late_stream.exists():
                late_stream.write_bytes(b"late")
        real_verify(retained)

    monkeypatch.setattr(
        capture.immutable.RetainedRegular,
        "verify_unchanged",
        verify_with_late_stream,
    )
    monkeypatch.setattr(capture, "_boot_id_sha256", lambda: "b" * 64)

    with pytest.raises(
        capture.CaptureError,
        match="transactionally publish failure provenance",
    ):
        capture._write_failure_provenance(
            failure_root,
            root=root,
            phase="ncu",
            row=row,
            command=["/fixture/target"],
            live=_nvbit_live(),
            environment_payload=b"{}\n",
            default_environment={"LANG": "C"},
            capture_producer_sha256="c" * 64,
            containment_supervisor_sha256="d" * 64,
            phase_tool_sha256="e" * 64,
            runtime_manifest_sha256=None,
            action_executor_sha256="f" * 64,
            error=capture.CaptureError("synthetic capture failure"),
        )

    assert not (failure_root / "failure_provenance.json").exists()
    retired = [
        path for path in failure_root.iterdir()
        if path.name.startswith(".failed-regular-")
    ]
    assert len(retired) == 1


def test_ncu_handoff_is_imported_and_published_as_singleton(
    tmp_path: Path,
    monkeypatch,
):
    row, _roster = capture._row_for(MATRIX_KEY, VARIANT_ID)
    root = tmp_path / "capture"
    cell, report_payload = _write_ncu_handoff(root, row)
    identity = capture._ncu_identity_from_authority(_ncu_authority())
    csv_text = _ncu_details_csv(row)
    observed = {}

    def imported(expected, retained_report, **kwargs):
        observed["expected"] = expected
        observed["report_path"] = retained_report.path
        observed["report_descriptor"] = retained_report.descriptor
        observed["report_sha256"] = retained_report.sha256
        observed["containment"] = kwargs["require_cgroup_containment"]
        return (
            capture.parse_ncu_metrics.ImportedReportCsv(
                text=csv_text,
                report_sha256=capture.sha256_bytes(report_payload),
                report_size_bytes=len(report_payload),
            ),
            None,
        )

    monkeypatch.setattr(
        capture.parse_ncu_metrics,
        "_import_retained_report_csv",
        imported,
    )
    monkeypatch.setattr(
        capture.parse_ncu_metrics,
        "_import_report_csv",
        lambda *_args, **_kwargs: pytest.fail(
            "NCU handoff reread the mutable report path"
        ),
    )
    monkeypatch.setattr(
        capture.admission,
        "_sass_kernel_names",
        lambda _row: {"_Zfixture_kernel"},
    )

    capture._postprocess_ncu_handoff(
        root=root,
        row=row,
        live=_nvbit_live(),
        expected_ncu_identity=identity,
    )

    report_descriptor = observed.pop("report_descriptor")
    assert isinstance(report_descriptor, int)
    assert report_descriptor >= 0
    assert observed == {
        "expected": identity,
        "report_path": cell / "run_0_launch_set.ncu-rep",
        "report_sha256": capture.sha256_bytes(report_payload),
        "containment": True,
    }
    assert {path.name for path in cell.iterdir()} == set(
        capture._NCU_COMPLETE_ARTIFACT_NAMES
    )
    sidecar = json.loads((cell / "label_0_0.json").read_text())
    assert sidecar["cache_hit_rate_l2_sector"] == 0.5
    assert sidecar["cache_hit_rate_l1tex_sector"] == 0.25
    assert sidecar["measurement_protocol"] == {
        "ncu_clock_control": "none",
        "ncu_cache_control": "all",
        "ncu_replay_mode": "kernel",
        "ncu_target_processes": "all",
        "ncu_filter_mode": "global",
        "ncu_profile_from_start": True,
        "ncu_launch_skip": 0,
        "ncu_launch_count": 1,
        "host_gpu_index": 0,
        "host_persistence_mode": "Enabled",
        "host_sm_clock_mhz": capture.SM_CLOCK_MHZ,
        "host_mem_clock_mhz": capture.MEM_CLOCK_MHZ,
    }
    assert sidecar["unit_tagged_metrics"] == [
        {
            "ncu_metric": "gpc__cycles_elapsed.max",
            "unit": "cycles",
            "value": 123.0,
        },
        {
            "ncu_metric": "gpu__time_duration.sum",
            "unit": "ns",
            "value": 456.0,
        },
    ]
    observation = json.loads(
        (cell / "run_0_launch_set.launch_observation.json").read_text()
    )
    assert observation["report_sha256"] == capture.sha256_bytes(report_payload)
    assert observation["identity"]["Kernel Name"] == "_Zfixture_kernel"


def test_ncu_handoff_retains_observed_geometry_over_declared_geometry(
    tmp_path: Path,
    monkeypatch,
):
    row, _roster = capture._row_for(MATRIX_KEY, VARIANT_ID)
    root = tmp_path / "capture"
    cell, report_payload = _write_ncu_handoff(root, row)
    identity = capture._ncu_identity_from_authority(_ncu_authority())
    observed_block = [256, 1, 1]
    observed_grid = [1, 1, 1]
    assert observed_block != row["shape"]["trace_shape"]["launch_config"]["block"]
    assert observed_grid != row["shape"]["trace_shape"]["launch_config"]["grid"]
    monkeypatch.setattr(
        capture.parse_ncu_metrics,
        "_import_retained_report_csv",
        lambda *_args, **_kwargs: (
            capture.parse_ncu_metrics.ImportedReportCsv(
                text=_ncu_details_csv(
                    row,
                    block=observed_block,
                    grid=observed_grid,
                ),
                report_sha256=capture.sha256_bytes(report_payload),
                report_size_bytes=len(report_payload),
            ),
            None,
        ),
    )
    monkeypatch.setattr(
        capture.admission,
        "_sass_kernel_names",
        lambda _row: {"_Zfixture_kernel"},
    )

    capture._postprocess_ncu_handoff(
        root=root,
        row=row,
        live=_nvbit_live(),
        expected_ncu_identity=identity,
    )

    observation = json.loads(
        (cell / "run_0_launch_set.launch_observation.json").read_text()
    )
    assert capture._ncu_observed_launch(row, observation) == {
        "kernel_name": "_Zfixture_kernel",
        "grid_dim": observed_grid,
        "block_dim": observed_block,
    }


def test_ncu_handoff_generation_stays_retained_until_receipt_commit(
    tmp_path: Path,
    monkeypatch,
):
    row, _roster = capture._row_for(MATRIX_KEY, VARIANT_ID)
    root = tmp_path / "capture"
    cell, report_payload = _write_ncu_handoff(root, row)
    identity = capture._ncu_identity_from_authority(_ncu_authority())
    monkeypatch.setattr(
        capture.parse_ncu_metrics,
        "_import_retained_report_csv",
        lambda *_args, **_kwargs: (
            capture.parse_ncu_metrics.ImportedReportCsv(
                text=_ncu_details_csv(row),
                report_sha256=capture.sha256_bytes(report_payload),
                report_size_bytes=len(report_payload),
            ),
            None,
        ),
    )
    monkeypatch.setattr(
        capture.admission,
        "_sass_kernel_names",
        lambda _row: {"_Zfixture_kernel"},
    )
    report = cell / "run_0_launch_set.ncu-rep"

    with pytest.raises(
        capture.CaptureError,
        match="retained close failed|generation",
    ):
        with contextlib.ExitStack() as generation_stack:
            capture._postprocess_ncu_handoff(
                root=root,
                row=row,
                live=_nvbit_live(),
                expected_ncu_identity=identity,
                generation_stack=generation_stack,
            )
            replacement = report.with_name("replacement.ncu-rep")
            replacement.write_bytes(report_payload)
            replacement.replace(report)


@pytest.mark.parametrize(
    "artifact_name",
    (
        "run_0_launch_set.launch_observation.json",
        "label_0_0.json",
    ),
)
def test_ncu_generated_artifact_stays_retained_until_receipt_commit(
    tmp_path: Path,
    monkeypatch,
    artifact_name: str,
):
    row, _roster = capture._row_for(MATRIX_KEY, VARIANT_ID)
    root = tmp_path / "capture"
    cell, report_payload = _write_ncu_handoff(root, row)
    identity = capture._ncu_identity_from_authority(_ncu_authority())
    monkeypatch.setattr(
        capture.parse_ncu_metrics,
        "_import_retained_report_csv",
        lambda *_args, **_kwargs: (
            capture.parse_ncu_metrics.ImportedReportCsv(
                text=_ncu_details_csv(row),
                report_sha256=capture.sha256_bytes(report_payload),
                report_size_bytes=len(report_payload),
            ),
            None,
        ),
    )
    monkeypatch.setattr(
        capture.admission,
        "_sass_kernel_names",
        lambda _row: {"_Zfixture_kernel"},
    )

    with pytest.raises(
        capture.CaptureError,
        match="retained close failed|generation",
    ):
        with contextlib.ExitStack() as generation_stack:
            capture._postprocess_ncu_handoff(
                root=root,
                row=row,
                live=_nvbit_live(),
                expected_ncu_identity=identity,
                generation_stack=generation_stack,
            )
            artifact = cell / artifact_name
            replacement = artifact.with_name(f"replacement-{artifact.name}")
            replacement.write_bytes(artifact.read_bytes())
            replacement.replace(artifact)


def test_ncu_handoff_rejects_unexpected_namespace_before_import(
    tmp_path: Path,
    monkeypatch,
):
    row, _roster = capture._row_for(MATRIX_KEY, VARIANT_ID)
    root = tmp_path / "capture"
    cell, _report_payload = _write_ncu_handoff(root, row)
    (cell / "unexpected.sidecar.json").write_text("{}\n")
    monkeypatch.setattr(
        capture.parse_ncu_metrics,
        "_import_retained_report_csv",
        lambda *_args, **_kwargs: pytest.fail(
            "unexpected namespace must fail before report import"
        ),
    )

    with pytest.raises(capture.CaptureError, match="namespace differs"):
        capture._postprocess_ncu_handoff(
            root=root,
            row=row,
            live=_nvbit_live(),
            expected_ncu_identity=capture._ncu_identity_from_authority(
                _ncu_authority()
            ),
        )


def test_ncu_handoff_rejects_imported_report_identity_drift(
    tmp_path: Path,
    monkeypatch,
):
    row, _roster = capture._row_for(MATRIX_KEY, VARIANT_ID)
    root = tmp_path / "capture"
    _cell, report_payload = _write_ncu_handoff(root, row)
    imported = capture.parse_ncu_metrics.ImportedReportCsv(
        text=_ncu_details_csv(row),
        report_sha256="0" * 64,
        report_size_bytes=len(report_payload),
    )
    monkeypatch.setattr(
        capture.parse_ncu_metrics,
        "_import_retained_report_csv",
        lambda *_args, **_kwargs: (imported, None),
    )

    with pytest.raises(capture.CaptureError, match="identity differs"):
        capture._postprocess_ncu_handoff(
            root=root,
            row=row,
            live=_nvbit_live(),
            expected_ncu_identity=capture._ncu_identity_from_authority(
                _ncu_authority()
            ),
        )


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr"),
    [
        (0, b"", capture._NCU_HANDOFF_STDERR),
        (3, b"unexpected", capture._NCU_HANDOFF_STDERR),
        (3, b"", capture._NCU_HANDOFF_STDERR.rstrip(b"\n")),
        (3, b"", capture._NCU_HANDOFF_STDERR + b"extra\n"),
        (4, b"", capture._NCU_HANDOFF_STDERR),
    ],
)
def test_ncu_handoff_accepts_only_exact_driver_result(
    returncode: int,
    stdout: bytes,
    stderr: bytes,
):
    assert not capture._capture_protocol_succeeded(
        "ncu",
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )
    assert capture._capture_protocol_succeeded(
        "ncu",
        returncode=3,
        stdout=b"",
        stderr=capture._NCU_HANDOFF_STDERR,
    )


def _nvbit_startup_stderr(context: bytes = b"555f69dd8fc0") -> bytes:
    return (
        b"WARNING: Do not call CUDA memory allocation in nvbit_at_ctx_init(). "
        b"It will cause deadlocks. Do them in nvbit_tool_init(). If you "
        b"encounter deadlocks, remove CUDA API calls to debug.\n"
        b"INFO: tracer binary_version initialised per-context from device "
        b"compute capability at tool-init: binary_version=86 (ctx=0x"
        + context
        + b")\n"
        b"INFO: cuFuncGetAttribute(BINARY_VERSION) returned attr_bv=0 "
        b"(CUresult=400); falling back to device compute capability 8.6 -> "
        b"binary_version=86\n"
    )


def test_nvbit_protocol_accepts_empty_or_exact_startup_diagnostics():
    for stderr in (b"", _nvbit_startup_stderr()):
        assert capture._nvbit_capture_stderr_matches(stderr)
        assert capture._capture_protocol_succeeded(
            "nvbit",
            returncode=0,
            stdout=b"bounded tracer diagnostics",
            stderr=stderr,
        )
    assert not capture._capture_protocol_succeeded(
        "nvbit",
        returncode=1,
        stdout=b"bounded tracer diagnostics",
        stderr=_nvbit_startup_stderr(),
    )


@pytest.mark.parametrize(
    "stderr",
    [
        _nvbit_startup_stderr() + b"extra\n",
        _nvbit_startup_stderr(b"0"),
        _nvbit_startup_stderr().replace(b"binary_version=86", b"binary_version=120", 1),
        _nvbit_startup_stderr().replace(b"CUresult=400", b"CUresult=0"),
        b"arbitrary warning\n",
    ],
)
def test_nvbit_protocol_rejects_malformed_startup_diagnostics(stderr: bytes):
    assert not capture._nvbit_capture_stderr_matches(stderr)
    assert not capture._capture_protocol_succeeded(
        "nvbit",
        returncode=0,
        stdout=b"bounded tracer diagnostics",
        stderr=stderr,
    )


def test_nvbit_stream_artifacts_bind_exact_captured_buffers(tmp_path: Path):
    trace_root = tmp_path / "nvbit" / "synthetic"
    trace_root.mkdir(parents=True)
    stdout = b"captured stdout"
    stderr = _nvbit_startup_stderr()
    stdout_path = trace_root / capture._NVBIT_CAPTURE_STDOUT_NAME
    stderr_path = trace_root / capture._NVBIT_CAPTURE_STDERR_NAME
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    artifacts = [
        capture._artifact(tmp_path, stdout_path, "trace_auxiliary"),
        capture._artifact(tmp_path, stderr_path, "trace_auxiliary"),
    ]

    capture._require_nvbit_stream_artifact_bindings(
        tmp_path,
        trace_root,
        artifacts,
        capture_stdout=stdout,
        capture_stderr=stderr,
    )


@pytest.mark.parametrize(
    ("captured", "retained"),
    [
        (b"captured stdout", b"substituted stdout"),
        (b"", b"unexpected stdout"),
    ],
)
def test_nvbit_stream_artifacts_reject_substitution_or_empty_vacuity(
    tmp_path: Path,
    captured: bytes,
    retained: bytes,
):
    trace_root = tmp_path / "nvbit" / "synthetic"
    trace_root.mkdir(parents=True)
    stream_path = trace_root / capture._NVBIT_CAPTURE_STDOUT_NAME
    stream_path.write_bytes(retained)
    artifacts = [capture._artifact(
        tmp_path, stream_path, "trace_auxiliary",
    )]

    with pytest.raises(capture.CaptureError, match="capture stream artifact"):
        capture._require_nvbit_stream_artifact_bindings(
            tmp_path,
            trace_root,
            artifacts,
            capture_stdout=captured,
            capture_stderr=b"",
        )


def test_ncu_authority_requires_launcher_inner_path_binding():
    authority = _ncu_authority()
    authority["resolved_path"] = str(
        Path(authority["launcher_resolved_path"]).parent / "target/other/ncu"
    )
    with pytest.raises(capture.CaptureError, match="paths differ"):
        capture._ncu_identity_from_authority(authority)


def test_ncu_probe_records_exact_descriptor_output(monkeypatch):
    authority = _ncu_authority()
    monkeypatch.setattr(
        capture.campaign_preflight,
        "probe_ncu",
        lambda _pins, **_kwargs: dict(authority),
    )
    version_stdout = (
        "NVIDIA (R) Nsight Compute Command Line Profiler\n"
        "Copyright (c) 2018-2026 NVIDIA Corporation\n"
        + authority["version_line"]
        + "\n"
    )
    observed_calls = []

    def execute(*args, **kwargs):
        observed_calls.append((args, kwargs))
        return capture.ncu_executor.NcuActionResult(
            0,
            version_stdout.encode(),
            b"",
        )

    monkeypatch.setattr(
        capture.ncu_executor,
        "execute_ncu_action",
        execute,
    )
    observed_authority, stdout = capture._probe_ncu_authority()
    assert observed_authority == authority
    assert stdout == version_stdout
    assert observed_calls[0][1]["require_cgroup_containment"] is True


def test_ncu_probe_rejects_unrecorded_banner_output(monkeypatch):
    authority = _ncu_authority()
    monkeypatch.setattr(
        capture.campaign_preflight,
        "probe_ncu",
        lambda _pins, **_kwargs: dict(authority),
    )
    version_stdout = (
        "NVIDIA (R) Nsight Compute Command Line Profiler\n"
        "Copyright (c) 2018-2026 NVIDIA Corporation\n"
        + authority["version_line"]
        + "\n"
    )
    monkeypatch.setattr(
        capture.ncu_executor,
        "execute_ncu_action",
        lambda *_args, **_kwargs: capture.ncu_executor.NcuActionResult(
            0,
            (version_stdout + "unrecorded banner\n").encode(),
            b"",
        ),
    )
    with pytest.raises(capture.CaptureError, match="exact authority"):
        capture._probe_ncu_authority()


def test_ncu_capture_uses_descriptor_bound_supervision(
    tmp_path: Path, monkeypatch,
):
    identity = capture._ncu_identity_from_authority(_ncu_authority())
    driver = capture.ncu_executor.ExpectedExecutableIdentity(
        str(capture.LABEL_DRIVER), "d" * 64, 0o555,
    )
    observed = {}

    def execute(expected, expected_driver, command, **kwargs):
        observed.update({
            "expected": expected,
            "driver": expected_driver,
            "command": command,
            "environment": kwargs["environment"].as_dict(),
            "working_directory": kwargs["working_directory"],
            "auxiliary_descriptors": kwargs["auxiliary_descriptors"],
            "require_cgroup_containment": kwargs["require_cgroup_containment"],
        })
        return capture.ncu_executor.NcuActionResult(
            0,
            b"",
            b"",
            process_id=4321,
            started_at_utc="2026-08-02T00:00:00Z",
            containment=_containment(),
        )

    monkeypatch.setattr(
        capture.ncu_executor, "execute_ncu_supervised_executable", execute,
    )
    env = {
        "CUDA_INJECTION64_PATH": "/hostile/tracer.so",
        "LD_PRELOAD": "/hostile/preload.so",
        "LD_SOURCE_COMMIT_OVERRIDE": capture.PINNED_COMMIT,
        "TMPDIR": str(tmp_path),
    }
    collection_environment = capture._label_collection_environment(
        env, cwd=tmp_path
    )
    result = capture._run_ncu_capture_command(
        _ncu_command(identity),
        identity,
        collection_environment,
        driver,
        _retained_target(),
        tmp_path,
    )
    assert result[0:2] == (0, 4321)
    assert observed["expected"] is identity
    assert observed["driver"] is driver
    command = observed["command"]
    separator = command.index("--")
    assert command[separator - 2:separator] == [
        "--ncu-exec-path",
        capture.ncu_executor.AUTHORIZED_NCU_PROC_PATH,
    ]
    assert command[command.index("--binary") + 1] == _retained_target().proc_path
    assert command[separator + 1] == "128"
    assert observed["auxiliary_descriptors"] == (_retained_target().descriptor,)
    assert observed["working_directory"] == tmp_path
    assert observed["require_cgroup_containment"] is True
    collection_env = observed["environment"]
    assert collection_env["LD_SOURCE_COMMIT_OVERRIDE"] == capture.PINNED_COMMIT
    assert collection_env["TMPDIR"] == str(tmp_path)
    assert "CUDA_INJECTION64_PATH" not in collection_env
    assert "LD_PRELOAD" not in collection_env


def test_generated_ncu_command_substitutes_binary_option(
    tmp_path: Path, monkeypatch,
):
    row, _roster = capture._row_for(MATRIX_KEY, VARIANT_ID)
    identity = capture._ncu_identity_from_authority(_ncu_authority())
    binary = capture.ROOT / row["static_launch_inventory"]["binary_path"]
    retained_target = SimpleNamespace(
        path=binary,
        proc_path="/proc/self/fd/91",
        descriptor=91,
    )
    driver = capture.ncu_executor.ExpectedExecutableIdentity(
        str(capture.LABEL_DRIVER), "d" * 64, 0o555,
    )
    observed = {}

    def execute(_expected, _expected_driver, command, **_kwargs):
        observed["command"] = command
        return capture.ncu_executor.NcuActionResult(
            0,
            b"",
            b"",
            process_id=4321,
            started_at_utc="2026-08-02T00:00:00Z",
            containment=_containment(),
        )

    monkeypatch.setattr(
        capture.ncu_executor, "execute_ncu_supervised_executable", execute,
    )
    environment = capture._label_collection_environment({}, cwd=binary.parent)
    capture._run_ncu_capture_command(
        capture.admission._expected_ncu_command(tmp_path, row, identity),
        identity,
        environment,
        driver,
        retained_target,
        binary.parent,
    )

    command = observed["command"]
    assert command[command.index("--binary") + 1] == retained_target.proc_path
    assert command[command.index("--") + 1:] == [
        str(value) for value in row["shape"]["actual_args"]
    ]


def test_ncu_capture_failure_canonicalizes_target_descriptor(
    tmp_path: Path, monkeypatch,
):
    identity = capture._ncu_identity_from_authority(_ncu_authority())
    driver = capture.ncu_executor.ExpectedExecutableIdentity(
        str(capture.LABEL_DRIVER), "d" * 64, 0o555,
    )
    environment = capture._label_collection_environment({}, cwd=tmp_path)

    def fail(_expected, _expected_driver, command, **kwargs):
        error = capture.ncu_executor.NcuActionExecutorError(
            capture.ncu_executor.NCU_ACTION_TIMEOUT,
            "synthetic supervised timeout",
        )
        error.process_id = 4321
        error.started_at_utc = "2026-08-02T00:00:00Z"
        error.process_role = "ncu_supervised_executable"
        error.process_command = tuple(
            identity.resolved_path
            if item is capture.ncu_executor.AUTHORIZED_NCU_PROC_PATH
            else str(item)
            for item in command
        )
        error.process_environment = kwargs["environment"].as_dict()
        error.process_tool_sha256 = driver.executable_sha256
        raise error

    monkeypatch.setattr(
        capture.ncu_executor, "execute_ncu_supervised_executable", fail,
    )
    with pytest.raises(capture.CaptureError, match="descriptor-bound NCU") as caught:
        capture._run_ncu_capture_command(
            _ncu_command(identity),
            identity,
            environment,
            driver,
            _retained_target(),
            tmp_path,
        )

    assert _retained_target().proc_path not in caught.value.process_command
    assert str(_retained_target().path) in caught.value.process_command
    assert caught.value.process_environment == environment.as_dict()


def test_ncu_capture_rejects_command_authority_drift(
    tmp_path: Path, monkeypatch,
):
    identity = capture._ncu_identity_from_authority(_ncu_authority())
    command = _ncu_command(identity)
    command[command.index("--launch-count") + 1] = "2"
    monkeypatch.setattr(
        capture.ncu_executor,
        "execute_ncu_supervised_executable",
        lambda *_args, **_kwargs: pytest.fail("drifted command must not run"),
    )
    environment = capture._label_collection_environment({}, cwd=tmp_path)
    driver = capture.ncu_executor.ExpectedExecutableIdentity(
        str(capture.LABEL_DRIVER), "d" * 64, 0o555
    )
    with pytest.raises(capture.CaptureError, match="--launch-count"):
        capture._run_ncu_capture_command(
            command, identity, environment, driver, _retained_target(), tmp_path
        )


def test_ncu_capture_rejects_profile_from_start_off(
    tmp_path: Path, monkeypatch,
):
    identity = capture._ncu_identity_from_authority(_ncu_authority())
    command = _ncu_command(identity)
    command[command.index("--profile-from-start") + 1] = "off"
    monkeypatch.setattr(
        capture.ncu_executor,
        "execute_ncu_supervised_executable",
        lambda *_args, **_kwargs: pytest.fail("inactive profiler command must not run"),
    )
    environment = capture._label_collection_environment({}, cwd=tmp_path)
    driver = capture.ncu_executor.ExpectedExecutableIdentity(
        str(capture.LABEL_DRIVER), "d" * 64, 0o555
    )
    with pytest.raises(capture.CaptureError, match="--profile-from-start"):
        capture._run_ncu_capture_command(
            command, identity, environment, driver, _retained_target(), tmp_path
        )


def test_label_driver_identity_requires_independent_digest(monkeypatch):
    monkeypatch.setattr(capture.admission, "LABEL_DRIVER_SHA256", "0" * 64)
    with pytest.raises(capture.CaptureError, match="pinned capture authority"):
        capture._label_driver_identity()


def test_descriptor_executed_label_driver_resolves_repository_siblings(
    tmp_path: Path,
):
    ncu = tmp_path / "ncu"
    ncu.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' "
        "'NVIDIA (R) Nsight Compute Command Line Profiler' "
        "'Copyright (c) 2018-2026 NVIDIA Corporation' "
        "'Version 2026.1.0.0 (build synthetic)'\n"
    )
    ncu.chmod(0o555)
    ncu_identity = capture.ncu_executor.ExpectedNcuIdentity(
        str(ncu),
        capture.sha256_bytes(ncu.read_bytes()),
        "Version 2026.1.0.0 (build synthetic)",
    )
    driver_identity = capture._label_driver_identity()
    environment = capture._label_collection_environment(
        {}, cwd=capture.ROOT
    )
    result = capture.ncu_executor.execute_ncu_supervised_executable(
        ncu_identity,
        driver_identity,
        [
            str(capture.LABEL_DRIVER),
            "--ncu-exec-path",
            capture.ncu_executor.AUTHORIZED_NCU_PROC_PATH,
        ],
        timeout_seconds=2,
        environment=environment,
        working_directory=capture.ROOT,
    )
    assert result.returncode == 2
    assert b"SOURCE_PATH_INVALID" not in result.stderr
    assert b"local-label driver" in result.stderr


def test_descriptor_failure_streams_are_retained(
    tmp_path: Path, monkeypatch,
):
    identity = capture._ncu_identity_from_authority(_ncu_authority())
    driver = capture.ncu_executor.ExpectedExecutableIdentity(
        str(capture.LABEL_DRIVER), "d" * 64, 0o555
    )
    environment = capture._label_collection_environment({}, cwd=tmp_path)
    failure = capture.ncu_executor.NcuActionExecutorError(
        capture.ncu_executor.NCU_ACTION_TIMEOUT,
        "synthetic timeout",
    )
    failure.stdout = b"bounded stdout"
    failure.stderr = b"bounded stderr"
    failure.process_id = 4321
    failure.started_at_utc = "2026-08-02T00:00:00Z"

    def execute(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(
        capture.ncu_executor, "execute_ncu_supervised_executable", execute
    )
    failure_root = tmp_path / "retained-failure"
    with pytest.raises(capture.CaptureError, match="descriptor-bound") as caught:
        capture._run_ncu_capture_command(
            _ncu_command(identity),
            identity,
            environment,
            driver,
            _retained_target(),
            tmp_path,
            failure_root=failure_root,
        )
    assert caught.value.process_id == 4321
    assert caught.value.started_at_utc == "2026-08-02T00:00:00Z"
    assert (failure_root / "ncu_executor_failure.stdout.bin").read_bytes() == (
        b"bounded stdout"
    )
    assert (failure_root / "ncu_executor_failure.stderr.bin").read_bytes() == (
        b"bounded stderr"
    )


def test_nvbit_runtime_environment_excludes_hostile_ambient_controls(
    tmp_path: Path, monkeypatch,
):
    row, _roster_row = capture._row_for(MATRIX_KEY, VARIANT_ID)
    expected = capture.admission._expected_nvbit_environment(tmp_path, row)
    tracer = tmp_path / "tracer_tool.so"
    tracer.write_bytes(b"synthetic tracer")
    pb_lib = capture.NVBIT_PB_LIB
    monkeypatch.setenv("LD_PRELOAD", "/tmp/hostile.so")
    monkeypatch.setenv("PYTHONPATH", "/tmp/hostile-python")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setattr(
        capture.nvbit_runtime,
        "_run_ldd",
        lambda _tracer_path, *, env, pass_fds: capture.subprocess.CompletedProcess(
            [],
            0,
            stdout=f"libprotobuf.so.1 => {pb_lib / 'libprotobuf.so.1'} (0x0)\n",
            stderr="",
        ),
    )

    observed = capture.nvbit_runtime.tracer_runtime_env(
        tracer,
        base_env=expected,
        pb_lib=pb_lib,
    )

    sanitized_expected = capture.nvbit_runtime.sanitized_capture_environment(
        expected
    )
    sanitized_expected["LD_LIBRARY_PATH"] = str(pb_lib)
    assert observed == sanitized_expected
    assert set(observed) == {
        "CUDA_VISIBLE_DEVICES",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "PATH",
        "PWD",
    }
    assert observed["CUDA_VISIBLE_DEVICES"] == "0"
    for name in (
        "CUDA_INJECTION64_PATH",
        "TRACES_FOLDER",
        "USER_DEFINED_FOLDERS",
        "LD_PRELOAD",
        "PYTHONPATH",
    ):
        assert name not in observed


def test_environment_provenance_names_trace_folder_controls():
    document = capture._environment_document(
        {"pin": "static"}, {
            "check": "live",
            "smi_stdout": "smi",
            "competing_process_stdout": "",
            "identity_stdout": "identity",
            "nvcc_stdout": "nvcc",
            "ncu_stdout": "ncu",
        },
        {"TRACES_FOLDER": "/capture/traces", "USER_DEFINED_FOLDERS": "1"},
        capture_kind="nvbit",
    )
    assert document["environment"]["TRACES_FOLDER"] == "/capture/traces"
    assert document["environment"]["USER_DEFINED_FOLDERS"] == "1"
