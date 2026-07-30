"""Owner-side runner for the three ignored machine-local Rust smokes.

The tracked v2 path-alias authority supplies the 17 repository-relative M1
trace roots.  Historical absolute paths are migration provenance only and are
never passed to Rust.  Ordinary explicit pytest runs report unavailable local
evidence as skips; the authoritative gate sets ``HW_EVIDENCE_GATE=1``, which
turns the same condition into a failure.

The module always collects 21 tests: one trace walk, one sidecar load, two
single-cell replay ablations, and one full-hierarchy replay for each of the 17
authority rows.
"""
from __future__ import annotations

import atexit
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import NoReturn

import pytest

from tools.nvbit_tracer.build_m1_payload_authority import (
    M1AuthorityError,
    validate_path_aliases,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SIM = _REPO_ROOT / "sim"
_PATH_ALIASES = _REPO_ROOT / "evidence/path_aliases.v2.json"
_CONTROL_BITS_DIR = _REPO_ROOT / "data/annex/control_bits_sm86_20260706"
_EXPECTED_ROW_COUNT = 17
_EXPECTED_CURRENT_PREFIXES = (
    "data/annex/m1_trace_capture_direct/",
    "workloads/portable_lcd/results/",
)
_RESULT_RE = re.compile(
    r"test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;"
)

pytestmark = pytest.mark.hardware_evidence


def _inside_repo(path: Path) -> bool:
    try:
        path.relative_to(_REPO_ROOT.resolve())
    except ValueError:
        return False
    return True


def _safe_current_root(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("current_root must be a nonempty string")
    if "\\" in value:
        raise ValueError(f"current_root uses a backslash: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise ValueError(f"current_root is not a safe repository-relative path: {value!r}")
    if not value.startswith(_EXPECTED_CURRENT_PREFIXES):
        raise ValueError(f"current_root is outside the two current M1 prefixes: {value!r}")
    resolved = (_REPO_ROOT / Path(*pure.parts)).resolve(strict=False)
    if not _inside_repo(resolved):
        raise ValueError(f"current_root escapes the repository: {value!r}")
    return value


def _load_current_rows() -> tuple[list[dict[str, str]], str | None]:
    """Load only slug/current_root from the tracked migration authority."""

    try:
        if not _PATH_ALIASES.is_file() or _PATH_ALIASES.is_symlink():
            raise ValueError(f"tracked M1 path authority is unavailable: {_PATH_ALIASES}")
        document = validate_path_aliases(_PATH_ALIASES.read_bytes())
        mappings = document["prefix_mappings"]
        if not isinstance(mappings, list) or len(mappings) != 2:
            raise ValueError("path-alias authority must contain two prefix mappings")
        current_prefixes = tuple(
            mapping.get("current_prefix") if isinstance(mapping, dict) else None
            for mapping in mappings
        )
        if current_prefixes != _EXPECTED_CURRENT_PREFIXES:
            raise ValueError("path-alias authority current prefixes are not canonical")
        attestation = document["attestation"]
        if not isinstance(attestation, dict) or attestation.get("roots") != 17:
            raise ValueError("path-alias authority roots attestation must equal 17")
        rows = document["rows"]
        if not isinstance(rows, list) or len(rows) != _EXPECTED_ROW_COUNT:
            raise ValueError("path-alias authority must contain exactly 17 rows")
        current_rows: list[dict[str, str]] = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {
                "slug",
                "historical_root",
                "current_root",
            }:
                raise ValueError("each path-alias row must have exactly three fields")
            slug = row["slug"]
            if (
                not isinstance(slug, str)
                or not slug
                or not slug.isascii()
                or not all(character.isprintable() and not character.isspace() for character in slug)
            ):
                raise ValueError(
                    "path-alias row slug must be a nonempty ASCII-graphic scope token"
                )
            # Deliberately retain no historical_root value after shape validation.
            current_rows.append(
                {"slug": slug, "current_root": _safe_current_root(row["current_root"])}
            )
        slugs = [row["slug"] for row in current_rows]
        if slugs != sorted(slugs) or len(set(slugs)) != _EXPECTED_ROW_COUNT:
            raise ValueError("path-alias rows must have 17 unique byte-sorted slugs")
        if len({row["current_root"] for row in current_rows}) != _EXPECTED_ROW_COUNT:
            raise ValueError("path-alias current roots must be unique")
        return current_rows, None
    except (
        OSError,
        UnicodeError,
        M1AuthorityError,
        ValueError,
    ) as exc:
        placeholders = [
            {"slug": f"authority_unavailable_{index:02d}", "current_root": ""}
            for index in range(_EXPECTED_ROW_COUNT)
        ]
        return placeholders, str(exc)


_CURRENT_ROWS, _AUTHORITY_ERROR = _load_current_rows()
_OWNED_TARGET_DIR: Path | None = None


def _gate_is_authoritative() -> bool:
    return os.environ.get("HW_EVIDENCE_GATE") == "1"


def _unavailable(reason: str) -> NoReturn:
    if _gate_is_authoritative():
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason)


def _require_common_prerequisites() -> None:
    if _AUTHORITY_ERROR is not None:
        _unavailable(_AUTHORITY_ERROR)
    if shutil.which("cargo") is None:
        _unavailable("cargo is not available on PATH")
    if not _CONTROL_BITS_DIR.is_dir() or _CONTROL_BITS_DIR.is_symlink():
        _unavailable(f"current control-bits directory is unavailable: {_CONTROL_BITS_DIR}")


def _trace_root(row: dict[str, str]) -> Path:
    _require_common_prerequisites()
    value = row["current_root"]
    if not value:
        _unavailable("path-alias authority did not supply a current trace root")
    root = (_REPO_ROOT / value).resolve(strict=False)
    if not _inside_repo(root):
        pytest.fail(f"resolved current trace root escapes the repository: {value}")
    if not root.is_dir() or root.is_symlink():
        _unavailable(f"current trace root is unavailable: {value}")
    trace = root / "dynamic_trace.pb"
    if not trace.is_file() or trace.is_symlink():
        _unavailable(f"current trace payload is unavailable: {value}/dynamic_trace.pb")
    return root


def _cleanup_owned_target() -> None:
    if _OWNED_TARGET_DIR is not None:
        shutil.rmtree(_OWNED_TARGET_DIR, ignore_errors=True)


atexit.register(_cleanup_owned_target)


def _cargo_target_dir() -> Path:
    global _OWNED_TARGET_DIR
    configured = os.environ.get("CARGO_TARGET_DIR")
    if configured:
        target = Path(configured)
        if not target.is_absolute():
            pytest.fail("CARGO_TARGET_DIR must be absolute for the hardware evidence owner")
        target = target.resolve(strict=False)
    else:
        if _OWNED_TARGET_DIR is None:
            _OWNED_TARGET_DIR = Path(tempfile.mkdtemp(prefix="gpu-sim-hw-cargo-", dir="/tmp"))
        target = _OWNED_TARGET_DIR.resolve()
    if _inside_repo(target):
        pytest.fail(f"CARGO_TARGET_DIR must be outside the repository: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _run_rust_test(
    *, package: str, test_binary: str, test_name: str, env_extra: dict[str, str]
) -> str:
    _require_common_prerequisites()
    env = os.environ.copy()
    for flag in (
        "SIM_L1_DISABLE",
        "SIM_L2_DISABLE",
        "SIM_DRAM_BW_DISABLE",
    ):
        env.pop(flag, None)
    env.update(env_extra)
    env["CARGO_TARGET_DIR"] = str(_cargo_target_dir())
    env["CARGO_TERM_COLOR"] = "never"
    command = [
        "cargo",
        "test",
        "--locked",
        "--release",
        "-p",
        package,
        "--test",
        test_binary,
        test_name,
        "--",
        "--ignored",
        "--exact",
        "--nocapture",
    ]
    proc = subprocess.run(
        command,
        cwd=_SIM,
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
        check=False,
    )
    combined = proc.stdout + proc.stderr
    if _gate_is_authoritative():
        print(combined, end="" if combined.endswith("\n") else "\n")
    assert proc.returncode == 0, combined
    assert _RESULT_RE.search(combined), (
        f"{test_name} did not execute as exactly one passing ignored Rust test\n{combined}"
    )
    assert "SKIP" not in combined, f"{test_name} returned a vacuous skip\n{combined}"
    return combined


def test_sim_trace_smoke_walks_real_trace() -> None:
    trace_root = _trace_root(_CURRENT_ROWS[0])
    output = _run_rust_test(
        package="sim-trace",
        test_binary="machine_local_smoke",
        test_name="real_trace_full_walk",
        env_extra={"SIM_TRACE_SMOKE_DIR": str(trace_root)},
    )
    assert "smoke:" in output


def test_sim_sm_sidecars_load_real_evidence() -> None:
    _require_common_prerequisites()
    output = _run_rust_test(
        package="sim-sm",
        test_binary="machine_local_sidecars",
        test_name="real_sidecars_all_load",
        env_extra={"SIM_CONTROL_BITS_DIR": str(_CONTROL_BITS_DIR)},
    )
    assert "smoke:" in output


def _run_replay(row: dict[str, str], **mode: str) -> str:
    trace_root = _trace_root(row)
    return _run_rust_test(
        package="sim-sm",
        test_binary="machine_local_replay",
        test_name="real_trace_replays_under_issue_model",
        env_extra={
            "SIM_TRACE_SMOKE_DIR": str(trace_root),
            "SIM_CONTROL_BITS_DIR": str(_CONTROL_BITS_DIR),
            **mode,
        },
    )


def test_sim_sm_replay_flat_mode_one_cell() -> None:
    output = _run_replay(_CURRENT_ROWS[0], SIM_L1_DISABLE="1")
    assert "smoke: replay" in output
    assert "smoke: l1" not in output
    assert "smoke: l2" not in output


def test_sim_sm_replay_l1_const_mode_one_cell() -> None:
    output = _run_replay(_CURRENT_ROWS[0], SIM_L2_DISABLE="1")
    assert "smoke: replay" in output
    assert "smoke: l1" in output
    assert "smoke: l2" not in output


@pytest.mark.parametrize("row", _CURRENT_ROWS, ids=lambda row: row["slug"])
def test_sim_sm_replay_smoke_per_cell(row: dict[str, str]) -> None:
    output = _run_replay(row)
    assert "smoke: replay" in output
    assert "smoke: l1" in output
    assert "smoke: l2" in output
