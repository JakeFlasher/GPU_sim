#!/usr/bin/env python3
"""Build or verify the frozen M1 path-alias and payload authorities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST_REL = Path(
    "data/annex/m1_trace_capture_20260706/capture_manifest.json"
)
SOURCE_MANIFEST_SHA256 = (
    "1cc49e44a556e618bb130cd9060f98b537d6a97d625facf578839253f8757565"
)
PATH_ALIASES_REL = Path("evidence/path_aliases.v2.json")
PAYLOAD_MANIFEST_REL = Path("evidence/manifests/m1_trace_payloads.sha256")

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_PAYLOAD_LINE_RE = re.compile(r"^([0-9a-f]{64})  (.+)$")


class M1PrerequisiteMissing(RuntimeError):
    """A required path or runtime prerequisite is absent."""


class M1AuthorityError(RuntimeError):
    """Present M1 inputs or authorities violate the closed contract."""


@dataclass(frozen=True)
class PrefixMapping:
    historical_prefix: str
    current_prefix: str

    @property
    def historical_path(self) -> Path:
        return Path(self.historical_prefix.rstrip("/"))


PREFIX_MAPPINGS = (
    PrefixMapping(
        "/home/jakeshea/non_root_profiler/cross_arch_gpu_profiler_suite/"
        "data/annex/m1_trace_capture_direct/",
        "data/annex/m1_trace_capture_direct/",
    ),
    PrefixMapping(
        "/home/jakeshea/non_root_profiler/cross_arch_gpu_profiler_suite/"
        "workloads/portable_lcd/results/",
        "workloads/portable_lcd/results/",
    ),
)


@dataclass(frozen=True)
class AuthorityBytes:
    path_aliases: bytes
    payload_manifest: bytes
    payload_paths: tuple[str, ...]


def _missing(message: str) -> NoReturn:
    raise M1PrerequisiteMissing(message)


def _invalid(message: str) -> NoReturn:
    raise M1AuthorityError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular(path: Path, label: str) -> bytes:
    if not path.exists():
        _missing(f"{label} is missing: {path}")
    if path.is_symlink() or not path.is_file():
        _invalid(f"{label} is not a regular nonsymlink file: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        _missing(f"{label} is unreadable: {path}: {exc}")


def _json_no_duplicates(payload: bytes, label: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _invalid(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=pairs_hook)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _invalid(f"{label} is not valid UTF-8 JSON: {exc}")


def _safe_relative(path: str, label: str) -> PurePosixPath:
    if not isinstance(path, str) or not path:
        _invalid(f"{label} must be a nonempty string")
    if "\\" in path or path.startswith("/") or path.endswith("/"):
        _invalid(f"{label} is not a safe relative POSIX path: {path!r}")
    if any(char.isspace() or ord(char) == 127 for char in path):
        _invalid(f"{label} contains whitespace/control bytes: {path!r}")
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        _invalid(f"{label} contains an empty/dot/dot-dot component: {path!r}")
    candidate = PurePosixPath(path)
    if candidate.is_absolute():
        _invalid(f"{label} is absolute: {path!r}")
    return candidate


def _inventory(base: Path, label: str) -> dict[str, Path]:
    if not base.exists():
        _missing(f"{label} root is missing: {base}")
    if base.is_symlink() or not base.is_dir():
        _invalid(f"{label} root is not a regular nonsymlink directory: {base}")
    files: dict[str, Path] = {}
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        here = Path(dirpath)
        for name in dirnames:
            child = here / name
            if child.is_symlink() or not child.is_dir():
                _invalid(f"{label} contains a nonregular directory: {child}")
        for name in filenames:
            path = here / name
            if path.is_symlink() or not path.is_file():
                _invalid(f"{label} contains a nonregular payload: {path}")
            relative = path.relative_to(base).as_posix()
            _safe_relative(relative, f"{label} payload path")
            if relative in files:
                _invalid(f"{label} contains duplicate payload path: {relative}")
            files[relative] = path
    return files


def _load_source_rows(repo_root: Path) -> list[dict[str, str]]:
    source = repo_root / SOURCE_MANIFEST_REL
    payload = _read_regular(source, "M1 source manifest")
    actual_sha = _sha256_bytes(payload)
    if actual_sha != SOURCE_MANIFEST_SHA256:
        _invalid(
            "M1 source manifest SHA-256 mismatch: "
            f"expected={SOURCE_MANIFEST_SHA256} actual={actual_sha}"
        )
    doc = _json_no_duplicates(payload, "M1 source manifest")
    if not isinstance(doc, dict) or not isinstance(doc.get("cells"), list):
        _invalid("M1 source manifest must be an object with a cells array")
    cells = doc["cells"]
    if len(cells) != 17:
        _invalid(f"M1 source manifest must contain exactly 17 cells, got {len(cells)}")

    rows: list[dict[str, str]] = []
    seen_slugs: set[str] = set()
    counts = [0, 0]
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            _invalid(f"M1 source cell {index} is not an object")
        slug = cell.get("slug")
        historical_root = cell.get("traces_folder")
        if not isinstance(slug, str) or _SLUG_RE.fullmatch(slug) is None:
            _invalid(f"M1 source cell {index} has unsafe slug: {slug!r}")
        if slug in seen_slugs:
            _invalid(f"M1 source manifest contains duplicate slug: {slug}")
        seen_slugs.add(slug)
        if not isinstance(historical_root, str):
            _invalid(f"M1 source cell {slug} has invalid traces_folder")
        matches = [
            mapping
            for mapping in PREFIX_MAPPINGS
            if historical_root.startswith(mapping.historical_prefix)
        ]
        if len(matches) != 1:
            _invalid(
                f"M1 source cell {slug} must match exactly one historical prefix"
            )
        mapping = matches[0]
        suffix = historical_root[len(mapping.historical_prefix) :]
        _safe_relative(suffix, f"M1 source cell {slug} historical suffix")
        current_root = mapping.current_prefix + suffix
        _safe_relative(current_root, f"M1 source cell {slug} current_root")
        mapping_index = PREFIX_MAPPINGS.index(mapping)
        counts[mapping_index] += 1
        rows.append(
            {
                "slug": slug,
                "historical_root": historical_root,
                "current_root": current_root,
            }
        )
    rows.sort(key=lambda row: row["slug"])
    if counts != [2, 15]:
        _invalid(f"M1 source partition must be [2, 15], got {counts}")
    return rows


def _prove_prefix_equality(
    mapping: PrefixMapping, repo_root: Path
) -> tuple[list[str], list[str]]:
    manifest_rows: list[str] = []
    payload_paths: list[str] = []
    historical = _inventory(mapping.historical_path, "historical M1 prefix")
    current_base = repo_root / mapping.current_prefix.rstrip("/")
    current = _inventory(current_base, "current M1 prefix")
    historical_set = set(historical)
    current_set = set(current)
    if historical_set != current_set:
        missing = sorted(historical_set - current_set)
        extra = sorted(current_set - historical_set)
        _invalid(
            "M1 payload relative-set mismatch: "
            f"prefix={mapping.current_prefix} missing={missing} extra={extra}"
        )
    for relative in sorted(historical_set):
        historical_bytes = _read_regular(
            historical[relative], "historical M1 payload"
        )
        current_bytes = _read_regular(current[relative], "current M1 payload")
        historical_sha = _sha256_bytes(historical_bytes)
        current_sha = _sha256_bytes(current_bytes)
        if historical_sha != current_sha:
            _invalid(
                "M1 payload byte mismatch: "
                f"path={mapping.current_prefix + relative} "
                f"historical={historical_sha} current={current_sha}"
            )
        repo_relative = mapping.current_prefix + relative
        _safe_relative(repo_relative, "M1 payload manifest path")
        payload_paths.append(repo_relative)
        manifest_rows.append(f"{current_sha}  {repo_relative}\n")
    return manifest_rows, payload_paths


def _prove_payload_equality(repo_root: Path) -> tuple[list[str], list[str]]:
    manifest_rows: list[str] = []
    payload_paths: list[str] = []
    for mapping in PREFIX_MAPPINGS:
        mapping_rows, mapping_paths = _prove_prefix_equality(mapping, repo_root)
        manifest_rows.extend(mapping_rows)
        payload_paths.extend(mapping_paths)
    if len(payload_paths) != 5033 or len(set(payload_paths)) != 5033:
        _invalid(
            "M1 payload inventory must contain exactly 5,033 unique paths, "
            f"got total={len(payload_paths)} unique={len(set(payload_paths))}"
        )
    ordered = sorted(zip(payload_paths, manifest_rows), key=lambda item: item[0])
    return [row for _path, row in ordered], [path for path, _row in ordered]


def build_authority_bytes(repo_root: Path = REPO_ROOT) -> AuthorityBytes:
    repo_root = repo_root.resolve()
    rows = _load_source_rows(repo_root)
    manifest_rows, payload_paths = _prove_payload_equality(repo_root)
    aliases = {
        "schema_version": 2,
        "source_manifest": {
            "path": SOURCE_MANIFEST_REL.as_posix(),
            "sha256": SOURCE_MANIFEST_SHA256,
        },
        "prefix_mappings": [
            {
                "historical_prefix": mapping.historical_prefix,
                "current_prefix": mapping.current_prefix,
            }
            for mapping in PREFIX_MAPPINGS
        ],
        "rows": rows,
        "attestation": {
            "roots": 17,
            "payloads": 5033,
            "missing": 0,
            "extra": 0,
            "mismatched": 0,
        },
    }
    alias_bytes = (
        json.dumps(aliases, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    payload_bytes = "".join(manifest_rows).encode("ascii")
    validate_path_aliases(alias_bytes)
    validate_payload_manifest(payload_bytes, repo_root)
    return AuthorityBytes(alias_bytes, payload_bytes, tuple(payload_paths))


def validate_path_aliases(payload: bytes) -> dict[str, Any]:
    doc = _json_no_duplicates(payload, "M1 path-alias authority")
    if not isinstance(doc, dict) or set(doc) != {
        "schema_version",
        "source_manifest",
        "prefix_mappings",
        "rows",
        "attestation",
    }:
        _invalid("M1 path-alias authority has an incorrect top-level key set")
    if doc["schema_version"] != 2 or isinstance(doc["schema_version"], bool):
        _invalid("M1 path-alias schema_version must be integer 2")
    source = doc["source_manifest"]
    if source != {
        "path": SOURCE_MANIFEST_REL.as_posix(),
        "sha256": SOURCE_MANIFEST_SHA256,
    }:
        _invalid("M1 path-alias source_manifest is not the pinned authority")
    expected_mappings = [
        {
            "historical_prefix": mapping.historical_prefix,
            "current_prefix": mapping.current_prefix,
        }
        for mapping in PREFIX_MAPPINGS
    ]
    if doc["prefix_mappings"] != expected_mappings:
        _invalid("M1 path-alias prefix_mappings are not exact and ordered")
    rows = doc["rows"]
    if not isinstance(rows, list) or len(rows) != 17:
        _invalid("M1 path-alias rows must contain exactly 17 entries")
    slugs: list[str] = []
    historical_roots: list[str] = []
    current_roots: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {
            "slug",
            "historical_root",
            "current_root",
        }:
            _invalid(f"M1 path-alias row {index} is not closed")
        slug = row["slug"]
        if not isinstance(slug, str) or _SLUG_RE.fullmatch(slug) is None:
            _invalid(f"M1 path-alias row {index} has invalid slug")
        matches = [
            mapping
            for mapping in PREFIX_MAPPINGS
            if isinstance(row["historical_root"], str)
            and row["historical_root"].startswith(mapping.historical_prefix)
        ]
        if len(matches) != 1:
            _invalid(f"M1 path-alias row {slug} has ambiguous historical root")
        mapping = matches[0]
        suffix = row["historical_root"][len(mapping.historical_prefix) :]
        _safe_relative(suffix, f"M1 path-alias row {slug} suffix")
        expected_current = mapping.current_prefix + suffix
        if row["current_root"] != expected_current:
            _invalid(f"M1 path-alias row {slug} has incorrect current_root")
        _safe_relative(row["current_root"], f"M1 path-alias row {slug} current_root")
        slugs.append(slug)
        historical_roots.append(row["historical_root"])
        current_roots.append(row["current_root"])
    if slugs != sorted(slugs) or len(set(slugs)) != 17:
        _invalid("M1 path-alias rows must have unique bytewise-sorted slugs")
    if len(set(historical_roots)) != 17 or len(set(current_roots)) != 17:
        _invalid("M1 path-alias rows must have unique historical and current roots")
    if doc["attestation"] != {
        "roots": 17,
        "payloads": 5033,
        "missing": 0,
        "extra": 0,
        "mismatched": 0,
    }:
        _invalid("M1 path-alias attestation is not exact")
    canonical = (
        json.dumps(doc, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    if payload != canonical:
        _invalid("M1 path-alias authority is not canonically serialized")
    return doc


def parse_payload_manifest(payload: bytes) -> dict[str, str]:
    if not payload.endswith(b"\n") or b"\r" in payload:
        _invalid("M1 payload manifest must contain LF-terminated rows")
    try:
        lines = payload[:-1].decode("ascii").split("\n")
    except UnicodeDecodeError as exc:
        raise M1AuthorityError("M1 payload manifest is not ASCII") from exc
    if len(lines) != 5033:
        _invalid(f"M1 payload manifest must contain 5,033 rows, got {len(lines)}")
    entries: dict[str, str] = {}
    for line in lines:
        match = _PAYLOAD_LINE_RE.fullmatch(line)
        if match is None:
            _invalid(f"malformed M1 payload manifest row: {line!r}")
        digest, path = match.groups()
        _safe_relative(path, "M1 payload manifest path")
        if path in entries:
            _invalid(f"duplicate M1 payload manifest path: {path}")
        entries[path] = digest
    if list(entries) != sorted(entries):
        _invalid("M1 payload manifest paths are not bytewise sorted")
    return entries


def validate_payload_manifest(payload: bytes, repo_root: Path = REPO_ROOT) -> None:
    entries = parse_payload_manifest(payload)
    expected_prefixes = tuple(mapping.current_prefix for mapping in PREFIX_MAPPINGS)
    for relative, expected_sha in entries.items():
        if not relative.startswith(expected_prefixes):
            _invalid(f"M1 payload manifest path is outside declared prefixes: {relative}")
        path = repo_root / relative
        actual = _sha256_bytes(_read_regular(path, "current M1 payload"))
        if actual != expected_sha:
            _invalid(
                f"M1 payload manifest hash mismatch: path={relative} "
                f"expected={expected_sha} actual={actual}"
            )


def _inside_git_worktree(path: Path) -> bool:
    probe = path if path.is_dir() else path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        result = subprocess.run(
            ["git", "-C", str(probe), "rev-parse", "--is-inside-work-tree"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        _missing("git is unavailable for M1 output worktree validation")
    return result.returncode == 0 and result.stdout.strip() == "true"


def _git_tracks_authority(path: Path) -> bool:
    """Return whether the active repository index already owns ``path``."""

    try:
        relative = path.absolute().relative_to(REPO_ROOT.absolute())
    except ValueError:
        return False
    try:
        worktree = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--is-inside-work-tree"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        _missing("git is unavailable for M1 canonical-output validation")
    if worktree.returncode != 0 or worktree.stdout.strip() != "true":
        return False
    tracked = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "ls-files",
            "--error-unmatch",
            "--",
            relative.as_posix(),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if tracked.returncode not in (0, 1):
        _invalid(f"cannot determine tracked state for M1 authority: {relative}")
    return tracked.returncode == 0


def _preflight_outputs(alias_out: Path, payload_out: Path) -> bool:
    alias_out = alias_out.absolute()
    payload_out = payload_out.absolute()
    if alias_out == payload_out:
        _invalid("M1 output destinations must be distinct")
    canonical_alias = (REPO_ROOT / PATH_ALIASES_REL).absolute()
    canonical_payload = (REPO_ROOT / PAYLOAD_MANIFEST_REL).absolute()
    canonical = alias_out == canonical_alias and payload_out == canonical_payload
    if (alias_out == canonical_alias) != (payload_out == canonical_payload):
        _invalid("M1 canonical outputs must be named together")
    if canonical:
        if alias_out.exists() or payload_out.exists():
            _invalid("M1 canonical creation requires both authorities to be absent")
        if _git_tracks_authority(alias_out) or _git_tracks_authority(payload_out):
            _invalid(
                "M1 canonical creation is forbidden after either authority "
                "has entered the repository index"
            )
    else:
        protected = [
            REPO_ROOT.resolve(),
            Path("/home/jakeshea/sm86_reference_capture_r27").resolve(),
        ] + [
            mapping.historical_path.resolve() for mapping in PREFIX_MAPPINGS
        ]
        for output in (alias_out, payload_out):
            parent = output.parent
            if not parent.exists() or parent.is_symlink() or not parent.is_dir():
                _missing(f"M1 external output parent is unavailable: {parent}")
            resolved = parent.resolve() / output.name
            for root in protected:
                try:
                    resolved.relative_to(root)
                except ValueError:
                    continue
                _invalid(f"M1 external output enters a protected root: {output}")
            if _inside_git_worktree(output):
                _invalid(f"M1 external output is inside a Git worktree: {output}")
            if output.exists() and (output.is_symlink() or not output.is_file()):
                _invalid(f"M1 output is not a regular nonsymlink file: {output}")
    return canonical


def _write_temp(destination: Path, payload: bytes) -> Path:
    fd, raw = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temp = Path(raw)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return temp


def _restore_output(destination: Path, original: bytes | None) -> None:
    if original is None:
        destination.unlink(missing_ok=True)
        return
    temp = _write_temp(destination, original)
    os.replace(temp, destination)


def _atomic_pair_write(
    alias_out: Path,
    alias_bytes: bytes,
    payload_out: Path,
    payload_bytes: bytes,
) -> None:
    originals = {
        alias_out: alias_out.read_bytes() if alias_out.exists() else None,
        payload_out: payload_out.read_bytes() if payload_out.exists() else None,
    }
    prepared: list[Path] = []
    try:
        alias_temp = _write_temp(alias_out, alias_bytes)
        prepared.append(alias_temp)
        payload_temp = _write_temp(payload_out, payload_bytes)
        prepared.append(payload_temp)
    except BaseException:
        for temp in prepared:
            temp.unlink(missing_ok=True)
        raise
    replaced: list[Path] = []
    try:
        os.replace(alias_temp, alias_out)
        replaced.append(alias_out)
        os.replace(payload_temp, payload_out)
        replaced.append(payload_out)
    except BaseException:
        alias_temp.unlink(missing_ok=True)
        payload_temp.unlink(missing_ok=True)
        for destination in reversed(replaced):
            _restore_output(destination, originals[destination])
        raise


def emit_authorities(alias_out: Path, payload_out: Path) -> AuthorityBytes:
    alias_out = alias_out.absolute()
    payload_out = payload_out.absolute()
    _preflight_outputs(alias_out, payload_out)
    built = build_authority_bytes(REPO_ROOT)
    _atomic_pair_write(
        alias_out,
        built.path_aliases,
        payload_out,
        built.payload_manifest,
    )
    return built


def verify_tracked_authorities() -> AuthorityBytes:
    built = build_authority_bytes(REPO_ROOT)
    alias_path = REPO_ROOT / PATH_ALIASES_REL
    payload_path = REPO_ROOT / PAYLOAD_MANIFEST_REL
    alias_bytes = _read_regular(alias_path, "tracked M1 path-alias authority")
    payload_bytes = _read_regular(payload_path, "tracked M1 payload authority")
    validate_path_aliases(alias_bytes)
    validate_payload_manifest(payload_bytes, REPO_ROOT)
    if alias_bytes != built.path_aliases:
        _invalid("tracked M1 path-alias authority is not byte-regenerable")
    if payload_bytes != built.payload_manifest:
        _invalid("tracked M1 payload authority is not byte-regenerable")
    return built


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-aliases-out", type=Path)
    parser.add_argument("--payload-manifest-out", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    emit_requested = (
        args.path_aliases_out is not None or args.payload_manifest_out is not None
    )
    if args.verify_only:
        if emit_requested:
            print("m1_payload_authority: --verify-only forbids output options", file=sys.stderr)
            return 2
    elif not (
        args.path_aliases_out is not None and args.payload_manifest_out is not None
    ):
        print(
            "m1_payload_authority: emission requires both output options",
            file=sys.stderr,
        )
        return 2
    try:
        if args.verify_only:
            built = verify_tracked_authorities()
        else:
            built = emit_authorities(
                args.path_aliases_out,
                args.payload_manifest_out,
            )
    except M1PrerequisiteMissing as exc:
        print(f"m1_payload_authority: prerequisite missing: {exc}", file=sys.stderr)
        return 2
    except (M1AuthorityError, OSError) as exc:
        print(f"m1_payload_authority: invalid: {exc}", file=sys.stderr)
        return 1
    print("prefixes=2")
    print("roots=17")
    print(f"payloads={len(built.payload_paths)}")
    print("missing=0")
    print("extra=0")
    print("mismatched=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
