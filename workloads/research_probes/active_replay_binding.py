"""Content-identity helpers for the retained simulator replay corpus."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    """Hash a file tree by relative names and bytes, never by mtimes."""

    if path.is_file():
        return sha256(path)
    if not path.is_dir():
        raise ValueError(f"content binding input is missing: {path}")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"content binding input tree is empty: {path}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        payload_hash = sha256(item).encode("ascii")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload_hash).to_bytes(8, "big"))
        digest.update(payload_hash)
    return digest.hexdigest()


def canonical_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def content_hash(path: Path) -> str:
    return tree_sha256(path)


def pair_digest(label: str, first: Path, second: Path) -> str:
    """Return a stable digest for one named replay pair."""

    if not first.is_file() or not second.is_file():
        raise ValueError(f"replay pair is missing: {first}, {second}")
    try:
        if os.path.samefile(first, second):
            raise ValueError(f"replay pair paths resolve to one file: {label}")
    except OSError as exc:
        raise ValueError(f"replay pair file identity is unreadable: {label}") from exc
    record = {
        "label": label,
        "first": {"path": canonical_path(first), "sha256": sha256(first)},
        "second": {"path": canonical_path(second), "sha256": sha256(second)},
    }
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def root_digest(entries: Sequence[tuple[str, str]]) -> str:
    """Hash sorted ``(label, pair_digest)`` entries as an aggregate root."""

    payload = b"".join(
        label.encode("utf-8")
        + b"\0"
        + digest.encode("ascii")
        + b"\n"
        for label, digest in sorted(entries)
    )
    return hashlib.sha256(payload).hexdigest()


def dependency_records(paths: Sequence[Path]) -> dict[str, str]:
    records: dict[str, str] = {}
    for path in paths:
        key = canonical_path(path)
        value = content_hash(path)
        previous = records.get(key)
        if previous is not None and previous != value:
            raise ValueError(f"dependency path resolved to conflicting content: {key}")
        records[key] = value
    return records


class ContentBinding:
    """Validate input hashes and aggregate replay-pair roots while scanning."""

    def __init__(self, document: Mapping[str, Any], *, binary: Path) -> None:
        if (
            document.get("kind") != "active_replay_content_binding"
            or type(document.get("schema_version")) is not int
            or document.get("schema_version") != 1
        ):
            raise ValueError("active replay content binding schema is unsupported")
        revision = document.get("repository_revision")
        if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError("active replay content binding repository revision is malformed")
        binary_record = document.get("simulator_binary")
        if not isinstance(binary_record, Mapping):
            raise ValueError("active replay content binding simulator identity is missing")
        expected_binary = canonical_path(binary)
        if binary_record.get("path") != expected_binary or binary_record.get("sha256") != sha256(binary):
            raise ValueError("active replay content binding simulator identity differs")
        for name, expected_path in (
            ("architecture_card", "arch_profiles/cards/sm_86.json"),
            ("pchase_card", "data/pchase/sm_86_static_card.json"),
        ):
            card_record = document.get(name)
            card_path = REPO_ROOT / expected_path
            if (
                not isinstance(card_record, Mapping)
                or card_record.get("path") != expected_path
                or card_record.get("sha256") != sha256(card_path)
            ):
                raise ValueError(f"active replay content binding {name} identity differs")
        dependencies = document.get("dependencies")
        groups = document.get("groups")
        if not isinstance(dependencies, Mapping) or not isinstance(groups, Mapping):
            raise ValueError("active replay content binding is incomplete")
        self.document = document
        self.dependencies = dependencies
        self.groups = groups
        self.entries: dict[str, list[tuple[str, str]]] = {}
        self.verified_dependencies: dict[str, str] = {}

    def check_pair(
        self,
        group: str,
        label: str,
        first: Path,
        second: Path,
        dependencies: Sequence[Path],
    ) -> str | None:
        if group not in self.groups:
            return f"content binding has no group {group}"
        for path in dependencies:
            key = canonical_path(path)
            expected = self.dependencies.get(key)
            if not isinstance(expected, str):
                return f"content binding has no dependency identity: {key}"
            observed = self.verified_dependencies.get(key)
            if observed is None:
                observed = content_hash(path)
                self.verified_dependencies[key] = observed
            if observed != expected:
                return f"content binding dependency mismatch: {key}"
        try:
            digest = pair_digest(label, first, second)
        except (OSError, ValueError) as exc:
            return str(exc)
        self.entries.setdefault(group, []).append((label, digest))
        return None

    def finish(self, counts: Mapping[str, int]) -> list[str]:
        failures: list[str] = []
        for group, record in self.groups.items():
            if not isinstance(record, Mapping):
                failures.append(f"content binding group {group} is malformed")
                continue
            entries = self.entries.get(group, [])
            expected_count = record.get("pair_count")
            expected_root = record.get("pair_root")
            if counts.get(group, 0) != expected_count or len(entries) != expected_count:
                failures.append(f"content binding pair count differs for {group}")
            if not isinstance(expected_root, str) or root_digest(entries) != expected_root:
                failures.append(f"content binding pair root differs for {group}")
        expected_total = self.document.get("active_pair_count")
        actual_total = sum(counts.get(group, 0) for group in counts if group != "supplemental")
        if actual_total != expected_total:
            failures.append("content binding active pair count differs")
        missing_dependencies = set(self.dependencies) - set(self.verified_dependencies)
        if missing_dependencies:
            failures.append(
                "content binding did not observe all declared dependencies: "
                + ", ".join(sorted(missing_dependencies)[:3])
            )
        all_entries = [entry for values in self.entries.values() for entry in values]
        corpus_root = self.document.get("corpus_root")
        if not isinstance(corpus_root, str) or root_digest(all_entries) != corpus_root:
            failures.append("content binding corpus root differs")
        return failures
