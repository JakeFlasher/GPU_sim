from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from tools.local_labels import activation_trace_orchestrator as activation
from tools.local_labels import roster_trace_capture as roster
from tools.local_labels import trace_register_cap_variant_cells as variants
from tools.local_labels.canary import emit_canary_manifest as canary
from tools.local_labels.immutable_artifact import open_retained_regular
from tools.nvbit_tracer import external_tracer as external_tracer_module
from tools.nvbit_tracer.external_tracer import (
    EXTERNAL_TRACER_MAX_BYTES,
    EXTERNAL_TRACER_PATH_ENV,
    EXTERNAL_TRACER_SHA256_ENV,
    ExternalTracerAuthorityError,
    require_external_tracer_authority,
    verify_retained_tracer_authority,
)
from workloads.simbench_bw import run_suite


_REPO = Path(__file__).resolve().parents[2]
_TARGETS = (
    _REPO / "tools/local_labels/activation_trace_orchestrator.py",
    _REPO / "tools/local_labels/roster_trace_capture.py",
    _REPO / "tools/local_labels/trace_register_cap_variant_cells.py",
    _REPO / "tools/local_labels/canary/emit_canary_manifest.py",
    _REPO / "workloads/simbench_bw/run_suite.py",
)
_TRACER_LIMIT_TARGETS = _TARGETS + (
    _REPO / "tools/local_labels/campaign_preflight.py",
    _REPO / "tools/local_labels/capture_cutlass_successor_receipts.py",
    _REPO / "tools/local_labels/emit_long_timing_feature_artifacts.py",
    _REPO / "tools/local_labels/sm86_capture_chunk.py",
    _REPO / "tools/trace_sampling/orchestrate.py",
    _REPO / "tools/trace_sampling/golden_cells/retained_nvbit_launch.py",
)


@pytest.fixture
def external_authority_fixture() -> Iterator[tuple[Path, str, Path]]:
    with tempfile.TemporaryDirectory(
        prefix="task615-external-tracer-", dir="/tmp"
    ) as raw_root:
        root = Path(raw_root)
        repo = root / "checkout"
        repo.mkdir()
        tracer = root / "external-build" / "tracer_tool.so"
        tracer.parent.mkdir()
        tracer.write_bytes(b"external split-v2 tracer fixture")
        digest = hashlib.sha256(tracer.read_bytes()).hexdigest()
        yield tracer, digest, repo


def test_external_tracer_authority_accepts_exact_external_non_git_file(
    external_authority_fixture: tuple[Path, str, Path],
) -> None:
    tracer, digest, repo = external_authority_fixture

    authority = require_external_tracer_authority(
        tracer,
        digest,
        repo_root=repo,
        environment={},
    )

    assert authority.path == tracer
    assert authority.sha256 == digest
    with open_retained_regular(
        tracer, max_bytes=EXTERNAL_TRACER_MAX_BYTES
    ) as retained:
        verify_retained_tracer_authority(retained, authority)


def test_external_tracer_authority_rejects_file_above_byte_ceiling(
    external_authority_fixture: tuple[Path, str, Path],
) -> None:
    tracer, _digest, repo = external_authority_fixture
    tracer.unlink()
    with tracer.open("wb") as stream:
        stream.truncate(EXTERNAL_TRACER_MAX_BYTES + 1)

    with pytest.raises(ExternalTracerAuthorityError, match="byte ceiling"):
        require_external_tracer_authority(
            tracer,
            "0" * 64,
            repo_root=repo,
            environment={},
        )


def test_external_tracer_authority_accepts_file_at_byte_ceiling(
    external_authority_fixture: tuple[Path, str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer, _digest, repo = external_authority_fixture
    payload = b"exact tracer ceiling"
    tracer.write_bytes(payload)
    monkeypatch.setattr(
        external_tracer_module, "EXTERNAL_TRACER_MAX_BYTES", len(payload)
    )

    authority = require_external_tracer_authority(
        tracer,
        hashlib.sha256(payload).hexdigest(),
        repo_root=repo,
        environment={},
    )

    assert authority.path == tracer


def test_external_tracer_authority_requires_complete_explicit_pair(
    external_authority_fixture: tuple[Path, str, Path],
) -> None:
    tracer, digest, repo = external_authority_fixture

    with pytest.raises(ExternalTracerAuthorityError, match="supplied together"):
        require_external_tracer_authority(
            tracer,
            None,
            repo_root=repo,
            environment={},
        )
    with pytest.raises(ExternalTracerAuthorityError, match="requires"):
        require_external_tracer_authority(repo_root=repo, environment={})
    with pytest.raises(ExternalTracerAuthorityError, match="lowercase 64-hex"):
        require_external_tracer_authority(
            tracer,
            digest.upper(),
            repo_root=repo,
            environment={},
        )


def test_external_tracer_authority_rejects_relative_and_checkout_local_paths(
    external_authority_fixture: tuple[Path, str, Path],
) -> None:
    _tracer, digest, repo = external_authority_fixture
    local_tracer = repo / "ignored" / "tracer_tool.so"
    local_tracer.parent.mkdir()
    local_tracer.write_bytes(b"checkout-local ignored tracer")

    with pytest.raises(ExternalTracerAuthorityError, match="absolute"):
        require_external_tracer_authority(
            Path("tracer_tool.so"),
            digest,
            repo_root=repo,
            environment={},
        )
    with pytest.raises(ExternalTracerAuthorityError, match="external"):
        require_external_tracer_authority(
            local_tracer,
            hashlib.sha256(local_tracer.read_bytes()).hexdigest(),
            repo_root=repo,
            environment={},
        )


def test_external_tracer_authority_rejects_other_git_repository(
    external_authority_fixture: tuple[Path, str, Path],
) -> None:
    tracer, _digest, repo = external_authority_fixture
    git_root = repo.parent / "other-checkout"
    (git_root / ".git").mkdir(parents=True)
    git_tracer = git_root / "build" / "tracer_tool.so"
    git_tracer.parent.mkdir()
    git_tracer.write_bytes(tracer.read_bytes())

    with pytest.raises(ExternalTracerAuthorityError, match="non-Git"):
        require_external_tracer_authority(
            git_tracer,
            hashlib.sha256(git_tracer.read_bytes()).hexdigest(),
            repo_root=repo,
            environment={},
        )


def test_retained_external_tracer_digest_mismatch_is_refused(
    external_authority_fixture: tuple[Path, str, Path],
) -> None:
    tracer, _digest, repo = external_authority_fixture
    authority = require_external_tracer_authority(
        tracer,
        "0" * 64,
        repo_root=repo,
        environment={},
    )

    with open_retained_regular(
        tracer, max_bytes=EXTERNAL_TRACER_MAX_BYTES
    ) as retained:
        with pytest.raises(ExternalTracerAuthorityError, match="digest mismatch"):
            verify_retained_tracer_authority(retained, authority)


def test_external_tracer_authority_rejects_leaf_swap_during_validation(
    external_authority_fixture: tuple[Path, str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer, digest, repo = external_authority_fixture
    payload = tracer.read_bytes()
    original_validate = external_tracer_module.require_external_non_git_artifact
    calls = 0

    def swap_before_revalidation(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            tracer.rename(tracer.with_suffix(".retired"))
            tracer.write_bytes(payload)
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(
        external_tracer_module,
        "require_external_non_git_artifact",
        swap_before_revalidation,
    )

    with pytest.raises(ExternalTracerAuthorityError, match="could not be retained"):
        require_external_tracer_authority(
            tracer,
            digest,
            repo_root=repo,
            environment={},
        )


def test_retained_external_tracer_rejects_authority_without_path_identity(
    external_authority_fixture: tuple[Path, str, Path],
) -> None:
    tracer, digest, _repo = external_authority_fixture
    authority = external_tracer_module.ExternalTracerAuthority(tracer, digest)

    with open_retained_regular(
        tracer, max_bytes=EXTERNAL_TRACER_MAX_BYTES
    ) as retained:
        with pytest.raises(ExternalTracerAuthorityError, match="lacks retained"):
            verify_retained_tracer_authority(retained, authority)


def test_retained_external_tracer_rejects_same_byte_leaf_replacement(
    external_authority_fixture: tuple[Path, str, Path],
) -> None:
    tracer, digest, repo = external_authority_fixture
    authority = require_external_tracer_authority(
        tracer,
        digest,
        repo_root=repo,
        environment={},
    )
    payload = tracer.read_bytes()
    tracer.rename(tracer.with_suffix(".retired"))
    tracer.write_bytes(payload)

    with open_retained_regular(
        tracer, max_bytes=EXTERNAL_TRACER_MAX_BYTES
    ) as retained:
        with pytest.raises(ExternalTracerAuthorityError, match="path identity"):
            verify_retained_tracer_authority(retained, authority)


def test_retained_external_tracer_rejects_same_byte_ancestor_replacement(
    external_authority_fixture: tuple[Path, str, Path],
) -> None:
    tracer, digest, repo = external_authority_fixture
    authority = require_external_tracer_authority(
        tracer,
        digest,
        repo_root=repo,
        environment={},
    )
    payload = tracer.read_bytes()
    build_root = tracer.parent
    build_root.rename(build_root.with_name("external-build-retired"))
    build_root.mkdir()
    tracer.write_bytes(payload)

    with open_retained_regular(
        tracer, max_bytes=EXTERNAL_TRACER_MAX_BYTES
    ) as retained:
        with pytest.raises(ExternalTracerAuthorityError, match="path identity"):
            verify_retained_tracer_authority(retained, authority)


def test_external_tracer_authority_rejects_normalized_claude_worktree_path(
    external_authority_fixture: tuple[Path, str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer, digest, repo = external_authority_fixture
    real_resolve = Path.resolve

    def resolve_into_forbidden_tree(
        candidate: Path, *, strict: bool = False
    ) -> Path:
        if candidate == tracer:
            return Path("/synthetic/.claude/worktrees/fixture/tracer_tool.so")
        return real_resolve(candidate, strict=strict)

    monkeypatch.setattr(Path, "resolve", resolve_into_forbidden_tree)

    with pytest.raises(ExternalTracerAuthorityError, match="Claude worktree"):
        require_external_tracer_authority(
            tracer,
            digest,
            repo_root=repo,
            environment={},
        )


def test_target_sources_have_no_checkout_or_installed_tracer_fallbacks() -> None:
    sources = {path: path.read_text(encoding="utf-8") for path in _TARGETS}

    for path, source in sources.items():
        assert "shutil.which" not in source, path
        assert '"tracer_tool" / "tracer_tool.so"' not in source, path
        assert "require_external_tracer_authority" in source, path
        assert "verify_retained_tracer_authority" in source, path
    assert "\n_TRACER_SO =" not in sources[_TARGETS[0]]
    assert "\n_TRACER_SO =" not in sources[_TARGETS[1]]
    assert "\n_TRACER_SO =" not in sources[_TARGETS[2]]
    assert "\nTRACER_SO =" not in sources[_TARGETS[3]]
    assert "\nNVBIT_TRACER_SO =" not in sources[_TARGETS[4]]


def test_retained_tracer_consumers_use_common_byte_ceiling() -> None:
    for path in _TRACER_LIMIT_TARGETS:
        source = path.read_text(encoding="utf-8")
        assert "EXTERNAL_TRACER_MAX_BYTES" in source, path
        assert "max_bytes=EXTERNAL_TRACER_MAX_BYTES" in source, path


def test_all_new_capture_entry_points_refuse_missing_external_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(EXTERNAL_TRACER_PATH_ENV, raising=False)
    monkeypatch.delenv(EXTERNAL_TRACER_SHA256_ENV, raising=False)
    output = tmp_path / "must-not-exist"

    assert activation.main(["--slug", "fixture", "--shape-point", "timing_shape"]) == 2
    assert roster.main(["--roster", str(tmp_path / "missing-roster.json")]) == 2
    assert variants.main(["--shape-point", "trace_l2_accuracy_xl"]) == 2
    assert canary.main([]) == 2
    assert run_suite.handle_trace({}, run_suite.DeviceProps(), output) == 2
    assert not output.exists()
