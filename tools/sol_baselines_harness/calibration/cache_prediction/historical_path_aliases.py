"""Versioned, hash-enforcing aliases for retained SOL calibration files."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
AUTHORITY_PATH = Path(__file__).with_name("historical_path_aliases.v1.json")
AUTHORITY_SHA256 = (
    "7094628cb7cc4dbb28ee3eb70ace634b91b6f6cfc5b1944682440da7074bc6ee"
)
AUTHORITY_KIND = "sol_calibration_historical_path_aliases_v1"
SM86_HARNESS_TRACES_ALIAS = "sm86_harness_traces_v1"
SM120_CALIBRATION_BINARIES_ALIAS = "sm120_metric_extractor_calibration_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_ALIASES = (
    {
        "alias_id": SM120_CALIBRATION_BINARIES_ALIAS,
        "current_prefix": (
            "data/annex/s5_metric_extractor_divergence/solbench_tests"
        ),
        "historical_prefix": (
            "/home/jakeshea/solbench_baseline/nvbit_tracer/"
            "metric_extractor/tests"
        ),
    },
    {
        "alias_id": SM86_HARNESS_TRACES_ALIAS,
        "current_prefix": "data/annex/harness_traces_sm86",
        "historical_prefix": "scripts/baselines_harness/.harness_traces",
    },
)


class HistoricalPathAliasError(RuntimeError):
    """A retained path cannot be resolved through the sealed alias table."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _invalid(code: str, detail: str) -> None:
    raise HistoricalPathAliasError(code, detail)


def _json_without_duplicates(payload: bytes) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _invalid("historical_path_alias_authority_invalid", f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(payload, object_pairs_hook=object_pairs)
    except (UnicodeError, json.JSONDecodeError) as exc:
        _invalid("historical_path_alias_authority_invalid", str(exc))


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def load_alias_authority() -> dict[str, Any]:
    """Load the exact v1 authority and reject any byte or shape drift."""

    try:
        payload = AUTHORITY_PATH.read_bytes()
    except OSError as exc:
        _invalid("historical_path_alias_authority_unavailable", str(exc))
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != AUTHORITY_SHA256:
        _invalid(
            "historical_path_alias_authority_hash_mismatch",
            f"expected {AUTHORITY_SHA256}, got {observed_sha256}",
        )
    document = _json_without_duplicates(payload)
    if (
        not isinstance(document, dict)
        or set(document) != {"aliases", "kind", "schema_version"}
        or type(document.get("schema_version")) is not int
        or document.get("schema_version") != 1
        or document.get("kind") != AUTHORITY_KIND
        or document.get("aliases") != list(_EXPECTED_ALIASES)
        or payload != _canonical_json_bytes(document)
    ):
        _invalid(
            "historical_path_alias_authority_invalid",
            "authority is not the closed canonical v1 table",
        )
    return document


def _safe_suffix(value: str) -> Path:
    suffix = Path(value)
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or any(part in {"", ".", ".."} for part in suffix.parts)
    ):
        _invalid("historical_path_alias_suffix_invalid", repr(value))
    return suffix


def _read_regular_nofollow(path: Path, *, repo_root: Path) -> bytes:
    lexical_root = repo_root.absolute()
    lexical_path = path.absolute()
    try:
        relative = lexical_path.relative_to(lexical_root)
        root = lexical_root.resolve(strict=True)
    except (OSError, ValueError) as exc:
        _invalid("historical_path_alias_escape", f"{path}: {exc}")
    if not relative.parts:
        _invalid("historical_path_alias_target_not_regular", str(path))

    directory_flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_fd = os.open(root, directory_flags)
    except OSError as exc:
        _invalid("historical_path_alias_target_unavailable", f"{root}: {exc}")
    descriptor: int | None = None
    try:
        for component in relative.parts[:-1]:
            child = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child
        descriptor = os.open(
            relative.parts[-1], file_flags, dir_fd=directory_fd,
        )
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            _invalid("historical_path_alias_target_not_regular", str(path))
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            blocks.append(block)
        return b"".join(blocks)
    except HistoricalPathAliasError:
        raise
    except OSError as exc:
        _invalid("historical_path_alias_target_unavailable", f"{path}: {exc}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)


def resolve_historical_file(
    historical_path: str | Path,
    *,
    alias_id: str,
    expected_sha256: str,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Resolve one historical path and require exact retained bytes."""

    if not isinstance(alias_id, str) or not isinstance(expected_sha256, str):
        _invalid("historical_path_alias_request_invalid", repr(alias_id))
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        _invalid("historical_path_alias_expected_hash_invalid", repr(expected_sha256))
    historical = (
        historical_path.as_posix()
        if isinstance(historical_path, Path)
        else historical_path
    )
    if not isinstance(historical, str) or not historical or "\\" in historical:
        _invalid("historical_path_alias_request_invalid", repr(historical))

    authority = load_alias_authority()
    matches = [
        row for row in authority["aliases"] if row["alias_id"] == alias_id
    ]
    if len(matches) != 1:
        _invalid("historical_path_alias_unknown", alias_id)
    row = matches[0]
    prefix = row["historical_prefix"]
    if not historical.startswith(prefix + "/"):
        _invalid(
            "historical_path_alias_prefix_mismatch",
            f"{historical!r} is not below {prefix!r}",
        )
    suffix = _safe_suffix(historical[len(prefix) + 1 :])
    current_prefix = _safe_suffix(row["current_prefix"])
    candidate = repo_root / current_prefix / suffix
    payload = _read_regular_nofollow(candidate, repo_root=repo_root)
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        _invalid(
            "historical_path_alias_hash_mismatch",
            f"{candidate}: expected {expected_sha256}, got {observed_sha256}",
        )
    return candidate


__all__ = [
    "AUTHORITY_PATH",
    "AUTHORITY_SHA256",
    "HistoricalPathAliasError",
    "SM120_CALIBRATION_BINARIES_ALIAS",
    "SM86_HARNESS_TRACES_ALIAS",
    "load_alias_authority",
    "resolve_historical_file",
]
