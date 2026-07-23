#!/usr/bin/env python3
"""Exact-prefix, non-destructive recovery for preserved M6 staging trees."""
from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator, FormatChecker

try:
    from . import immutable_artifact as immutable
    from . import m6_determinism_campaign as campaign
except ImportError:
    import immutable_artifact as immutable  # type: ignore
    import m6_determinism_campaign as campaign  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "workloads/simbench_bw/schemas/m6_recovery_receipt.schema.json"
TOOL_ID = "m6_recover_staging.v1"


class RecoveryError(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> None:
    raise RecoveryError(code, detail)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha(payload: bytes) -> str:
    return campaign.sha256_bytes(payload)


def _safe_relative(path: str) -> str:
    try:
        value = immutable.safe_relative_path(path)
    except immutable.ImmutableArtifactError as exc:
        _fail("M6_RECOVERY_PATH", str(exc))
    return PurePosixPath(*value.parts).as_posix()


def _require_current_owner(root: Path, role: str) -> None:
    if not hasattr(os, "getuid"):
        return
    try:
        metadata = root.lstat()
    except OSError as exc:
        _fail("M6_RECOVERY_INPUT", f"{role}: {root}: {exc}")
    if metadata.st_uid != os.getuid():
        _fail(
            "M6_RECOVERY_OWNER",
            f"{role}: owner uid {metadata.st_uid} differs from current uid {os.getuid()}",
        )


def _inventory(root: Path) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    """Take a pinned, no-follow inventory and retain only verified file payloads."""
    _require_current_owner(root, "M6 recovery staging")
    try:
        transaction = campaign._PinnedTreeTransaction(  # noqa: SLF001
            root, role="M6 recovery staging"
        )
    except campaign.M6Error as exc:
        _fail(exc.code, exc.detail)
    entries: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    try:
        for relative, (kind, snapshot) in sorted(transaction.inventory.items()):
            if not relative:
                continue
            device, inode, mode, nlink, size, _mtime, _ctime = snapshot
            if kind == "file":
                payload = transaction.read(relative, role="M6 recovery staging file")
                if len(payload) != size:
                    _fail("M6_RECOVERY_INVENTORY", relative)
                payloads[relative] = payload
                digest = _sha(payload)
            else:
                # Directory entries are represented by a deterministic empty
                # payload digest; their inode/mode identity remains authoritative.
                digest = _sha(b"")
            entries.append({
                "relative_path": relative,
                "kind": kind,
                "sha256": digest,
                "size_bytes": size,
                "device": device,
                "inode": inode,
                "mode": mode,
                "nlink": nlink,
            })
        transaction.verify()
    except campaign.M6Error as exc:
        _fail(exc.code, exc.detail)
    finally:
        transaction.close()
    return entries, payloads


def _bound_identity(path: Path, role: str) -> tuple[str, bytes]:
    _require_current_owner(path, role)
    try:
        before = path.lstat()
    except OSError as exc:
        _fail("M6_RECOVERY_INPUT", f"{role}: {path}: {exc}")
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        _fail("M6_RECOVERY_INPUT", f"{role}: regular single-link file required")
    try:
        payload = immutable.read_regular(path)
    except immutable.ImmutableArtifactError as exc:
        _fail("M6_RECOVERY_INPUT", f"{role}: {exc}")
    try:
        after = path.lstat()
    except OSError as exc:
        _fail("M6_RECOVERY_INPUT", f"{role}: {path}: {exc}")
    if (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        _fail("M6_RECOVERY_INPUT_MIXED", role)
    return _sha(payload), payload


def _validate_declared_sha(value: str, role: str) -> None:
    if not campaign._sha256_text(value):  # noqa: SLF001
        _fail("M6_RECOVERY_IDENTITY", f"{role} must be lowercase SHA-256")


def _recheck_bound_identity(
    path: Path,
    role: str,
    expected_payload: bytes,
) -> None:
    observed_digest, payload = _bound_identity(path, role)
    if observed_digest != _sha(expected_payload) or payload != expected_payload:
        _fail("M6_RECOVERY_INPUT_MIXED", f"{role}: owner generation changed")


def _bind_declared_identity(
    declared: str,
    path: Path | None,
    role: str,
    seal_field: str,
) -> bytes | None:
    """Bind a stable owner file to its embedded artifact model seal.

    The command-line identities are the schedule/journal/analysis model seals,
    not hashes of their presentation JSON bytes.  Raw bytes are retained and
    rechecked separately to close the pathname generation throughout recovery.
    """
    _validate_declared_sha(declared, role)
    if path is None:
        return None
    _observed_digest, payload = _bound_identity(path, role)
    try:
        value = campaign.loads_strict(payload, role=f"recovery {role}")
    except campaign.M6Error:
        _fail("M6_RECOVERY_IDENTITY_MISMATCH", f"{role}: {path}")
    if not isinstance(value, Mapping) or value.get(seal_field) != declared:
        _fail("M6_RECOVERY_IDENTITY_MISMATCH", f"{role}: {path}")
    return payload


def _validate_semantic_inputs(
    *,
    schedule_payload: bytes,
    journal_payload: bytes,
    analysis_payload: bytes,
    schedule_path: Path,
    journal_path: Path,
    analysis_path: Path,
    evidence_root: Path | None,
    schedule_sha256: str,
    journal_sha256: str,
    analysis_sha256: str,
) -> None:
    """Recompute the owning M6 inputs before accepting recovery material."""
    try:
        schedule = campaign.loads_strict(schedule_payload, role="recovery schedule")
        journal = campaign.loads_strict(journal_payload, role="recovery journal")
        analysis = campaign.loads_strict(analysis_payload, role="recovery analysis")
        if not all(isinstance(value, dict) for value in (schedule, journal, analysis)):
            _fail("M6_RECOVERY_SEMANTIC_INPUT", "schedule/journal/analysis objects required")
        if (
            schedule.get("schedule_sha256") != schedule_sha256
            or journal.get("journal_sha256") != journal_sha256
            or analysis.get("analysis_sha256") != analysis_sha256
        ):
            _fail("M6_RECOVERY_IDENTITY_MISMATCH", "declared artifact identity differs")
        campaign.validate_schedule(
            schedule,
            repo_root=campaign.REPO_ROOT,
            rehash_sources=True,
        )
        campaign.validate_journal(
            schedule,
            journal,
            journal_path=journal_path,
            repo_root=campaign.REPO_ROOT,
            rehash_sidecars=True,
            evidence_root=evidence_root,
        )
        campaign._validate_schema("analysis", analysis, campaign.REPO_ROOT)  # noqa: SLF001
        campaign._validate_seal(analysis, "analysis_sha256", role="recovery analysis")  # noqa: SLF001
        regenerated = campaign.analyze(
            schedule,
            journal,
            journal_path=journal_path,
            generated_at_utc=analysis.get("generated_at_utc"),
            bootstrap_replicates=analysis.get("analysis_policy", {}).get(
                "bootstrap_replicates"
            ),
            bootstrap_seed=analysis.get("analysis_policy", {}).get("bootstrap_seed"),
            evidence_root=evidence_root,
            repo_root=campaign.REPO_ROOT,
        )
        if regenerated != analysis:
            _fail("M6_RECOVERY_SEMANTIC_MISMATCH", "analysis does not rederive")
    except campaign.M6Error as exc:
        _fail(exc.code, exc.detail)
    except RecoveryError:
        raise
    except Exception as exc:
        _fail("M6_RECOVERY_SEMANTIC_INPUT", f"{type(exc).__name__}: {exc}")


def _failure_inventory(root: Path) -> list[dict[str, Any]]:
    """Retain a minimally useful identity even when a tree cannot be scanned."""
    try:
        metadata = root.lstat()
        device = int(metadata.st_dev)
        inode = int(metadata.st_ino)
        mode = int(metadata.st_mode)
        nlink = int(metadata.st_nlink)
        size = int(metadata.st_size)
    except OSError:
        device = inode = size = 0
        mode = stat.S_IFDIR
        nlink = 1
    return [{
        "relative_path": ".root",
        "kind": "directory",
        "sha256": _sha(b""),
        "size_bytes": size,
        "device": device,
        "inode": inode,
        "mode": mode,
        "nlink": max(1, nlink),
    }]


def _expected_material(
    expected_root: Path,
) -> tuple[list[str], dict[str, bytes], list[dict[str, Any]]]:
    entries, payloads = _inventory(expected_root)
    files = sorted(
        entry["relative_path"] for entry in entries
        if entry["kind"] == "file"
    )
    if not files:
        _fail("M6_RECOVERY_EXPECTED_EMPTY", str(expected_root))
    expected_directories: set[str] = set()
    for relative in files:
        parent = PurePosixPath(relative).parent
        while parent != PurePosixPath("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    unsupported_empty = sorted(
        entry["relative_path"]
        for entry in entries
        if entry["kind"] == "directory"
        and entry["relative_path"] not in expected_directories
    )
    if unsupported_empty:
        _fail(
            "M6_RECOVERY_EMPTY_DIRECTORY_UNSUPPORTED",
            ",".join(unsupported_empty),
        )
    return files, payloads, entries


def _compare_staged_topology(
    staged_entries: Sequence[Mapping[str, Any]],
    expected_entries: Sequence[Mapping[str, Any]],
    staged_payloads: Mapping[str, bytes],
    expected_payloads: Mapping[str, bytes],
    expected_files: Sequence[str],
) -> tuple[str | None, list[str]]:
    """Compare every staged node, not only nonempty file leaves.

    A preserved partial may omit a suffix of the expected file sequence and
    its now-empty parent directories, but it may not introduce an extra node,
    change a node kind/mode, or provide a later file while an earlier expected
    file is absent.  The returned list is the exact staged file prefix.
    """
    staged = {str(row["relative_path"]): row for row in staged_entries}
    expected = {str(row["relative_path"]): row for row in expected_entries}
    if set(staged) - set(expected):
        return "unexpected_path", sorted(staged)
    for relative, row in staged.items():
        counterpart = expected[relative]
        if row.get("kind") != counterpart.get("kind"):
            return "type_mismatch", sorted(staged)
        # A staged node comes from the same logical publication and its mode is
        # part of the contradiction boundary.  Inode/device are intentionally
        # not compared: separate expected/staging trees necessarily have
        # distinct inode identities; _PinnedTreeTransaction already proves
        # each tree stable and single-link.
        if row.get("mode") != counterpart.get("mode"):
            return "mode_mismatch", sorted(staged)
        if row.get("kind") == "file":
            if staged_payloads.get(relative) != expected_payloads.get(relative):
                return "byte_mismatch", sorted(staged)
    staged_files = sorted(
        relative for relative, row in staged.items() if row.get("kind") == "file"
    )
    expected_file_list = list(expected_files)
    if staged_files != expected_file_list[: len(staged_files)]:
        return "non_prefix_material", staged_files
    return None, staged_files


def _receipt(
    *,
    decision: str,
    staging_root: Path,
    inventory: Sequence[Mapping[str, Any]],
    schedule_sha256: str,
    journal_sha256: str,
    analysis_sha256: str,
    contradiction_code: str | None = None,
    published_path: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "m6_recovery_receipt",
        "tool_id": TOOL_ID,
        "created_at_utc": _now(),
        "decision": decision,
        "schedule_sha256": schedule_sha256,
        "journal_sha256": journal_sha256,
        "analysis_sha256": analysis_sha256,
        "staging_root": os.fspath(staging_root),
        "inventory": [dict(row) for row in inventory],
        "contradiction_code": contradiction_code,
        "published_path": published_path,
    }
    # The seal is over the unsigned receipt, matching the repository's
    # immutable-artifact model-seal convention.  Never hash the seal field
    # recursively.
    value["receipt_sha256"] = immutable.model_seal(value, "receipt_sha256")
    try:
        schema = json.loads(immutable.read_regular(SCHEMA_PATH))
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(value)
    except Exception as exc:
        _fail("M6_RECOVERY_SCHEMA", str(exc))
    return value


def _contains_path(parent: Path, child: Path) -> bool:
    return parent == child or parent in child.parents


def _resolved_contains(parent: Path, child: Path) -> bool:
    """Compare roots after resolving aliases without trusting a leaf symlink."""
    parent_real = Path(os.path.realpath(os.fspath(parent)))
    child_real = Path(os.path.realpath(os.fspath(child)))
    return _contains_path(parent_real, child_real)


def _output_entry_kind(descriptor: int, name: str) -> str | None:
    try:
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        _fail("M6_RECOVERY_OUTPUT_PARENT", str(exc))
    if stat.S_ISLNK(metadata.st_mode):
        _fail("M6_RECOVERY_OUTPUT_PARENT", f"symlinked output entry: {name}")
    if stat.S_ISDIR(metadata.st_mode):
        return "directory"
    if stat.S_ISREG(metadata.st_mode):
        return "file"
    _fail("M6_RECOVERY_OUTPUT_PARENT", f"non-regular output entry: {name}")


def _output_has_conflicting_stage(
    reader: immutable.PinnedDirectoryReader,
    *,
    final_name: str,
    stage_name: str,
) -> bool:
    final_kind = _output_entry_kind(reader.descriptor, final_name)
    stage_kind = _output_entry_kind(reader.descriptor, stage_name)
    if final_kind is not None and stage_kind is not None:
        return True
    if final_kind is not None and final_kind != "directory":
        _fail("M6_RECOVERY_OUTPUT_PARENT", f"final publication is not a directory: {final_name}")
    if stage_kind is not None and stage_kind != "directory":
        _fail("M6_RECOVERY_OUTPUT_PARENT", f"staging publication is not a directory: {stage_name}")
    return False


def _publish_receipt(path: Path, value: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = campaign._pretty_json_bytes(value)  # noqa: SLF001

    def read_and_compare_existing() -> Mapping[str, Any]:
        try:
            existing = immutable.strict_json_load(path)
            if not isinstance(existing, dict):
                _fail("M6_RECOVERY_RECEIPT_COLLISION", os.fspath(path))
            immutable.verify_model_seal(existing, "receipt_sha256")
        except RecoveryError:
            raise
        except (OSError, immutable.ImmutableArtifactError) as exc:
            _fail("M6_RECOVERY_RECEIPT_COLLISION", f"{path}: {exc}")
        expected_unsigned = dict(value)
        expected_unsigned.pop("created_at_utc", None)
        expected_unsigned.pop("receipt_sha256", None)
        existing_unsigned = dict(existing)
        existing_unsigned.pop("created_at_utc", None)
        existing_unsigned.pop("receipt_sha256", None)
        if existing_unsigned != expected_unsigned:
            _fail("M6_RECOVERY_RECEIPT_COLLISION", os.fspath(path))
        return existing

    try:
        # Recovery receipts are immutable.  A pre-existing valid receipt is an
        # idempotent replay; a concurrent exact winner follows the same path.
        if path.exists() or path.is_symlink():
            if path.is_symlink():
                _fail("M6_RECOVERY_RECEIPT_COLLISION", os.fspath(path))
            return read_and_compare_existing()
        try:
            immutable.publish_regular_file(path.parent, path.name, payload)
        except immutable.ImmutableArtifactError as exc:
            if "publication already exists" not in str(exc):
                raise
            return read_and_compare_existing()
        return value
    except RecoveryError:
        raise
    except (OSError, immutable.ImmutableArtifactError) as exc:
        _fail("M6_RECOVERY_RECEIPT_PUBLICATION", str(exc))


def _emit_receipt(path: Path | None, value: dict[str, Any]) -> dict[str, Any]:
    if path is None:
        if value.get("decision") == "CONTRADICTORY":
            _fail("M6_RECOVERY_RECEIPT_REQUIRED", "contradictory evidence needs an immutable receipt")
        return value
    return dict(_publish_receipt(path, value))


def recover_staging(
    *,
    staging_root: Path,
    expected_root: Path,
    output_root: Path,
    final_name: str,
    schedule_sha256: str,
    journal_sha256: str,
    analysis_sha256: str,
    receipt_path: Path | None = None,
    schedule_path: Path | None = None,
    journal_path: Path | None = None,
    analysis_path: Path | None = None,
    evidence_root: Path | None = None,
) -> dict[str, Any]:
    staging_root = Path(os.path.abspath(os.fspath(staging_root)))
    expected_root = Path(os.path.abspath(os.fspath(expected_root)))
    output_root = Path(os.path.abspath(os.fspath(output_root)))
    schedule_payload = _bind_declared_identity(
        schedule_sha256, schedule_path, "schedule", "schedule_sha256"
    )
    journal_payload = _bind_declared_identity(
        journal_sha256, journal_path, "journal", "journal_sha256"
    )
    analysis_payload = _bind_declared_identity(
        analysis_sha256, analysis_path, "analysis", "analysis_sha256"
    )
    owner_paths = (schedule_path, journal_path, analysis_path)
    if any(path is None for path in owner_paths):
        value = _receipt(
            decision="UNAUTHORIZED",
            staging_root=staging_root,
            inventory=_failure_inventory(staging_root),
            schedule_sha256=schedule_sha256,
            journal_sha256=journal_sha256,
            analysis_sha256=analysis_sha256,
            contradiction_code="M6_RECOVERY_OWNER_PATHS_REQUIRED",
        )
        return _emit_receipt(receipt_path, value)
    assert schedule_path is not None
    assert journal_path is not None
    assert analysis_path is not None
    assert schedule_payload is not None
    assert journal_payload is not None
    assert analysis_payload is not None
    _validate_semantic_inputs(
        schedule_payload=schedule_payload,
        journal_payload=journal_payload,
        analysis_payload=analysis_payload,
        schedule_path=schedule_path,
        journal_path=journal_path,
        analysis_path=analysis_path,
        evidence_root=evidence_root,
        schedule_sha256=schedule_sha256,
        journal_sha256=journal_sha256,
        analysis_sha256=analysis_sha256,
    )
    _recheck_bound_identity(
        schedule_path, "schedule", schedule_payload
    )
    _recheck_bound_identity(
        journal_path, "journal", journal_payload
    )
    _recheck_bound_identity(
        analysis_path, "analysis", analysis_payload
    )
    roots = {
        "staging": staging_root,
        "expected": expected_root,
        "output": output_root,
    }
    for left_name, left in roots.items():
        for right_name, right in roots.items():
            if left_name >= right_name:
                continue
            if _resolved_contains(left, right) or _resolved_contains(right, left):
                _fail("M6_RECOVERY_TOPOLOGY", f"{left_name}/{right_name} roots overlap")
    if receipt_path is not None:
        receipt_absolute = Path(os.path.abspath(os.fspath(receipt_path)))
        if any(_resolved_contains(root, receipt_absolute) for root in roots.values()):
            _fail("M6_RECOVERY_TOPOLOGY", "receipt overlaps a recovery root")
    try:
        immutable._validate_publication_name(final_name)  # noqa: SLF001
    except immutable.ImmutableArtifactError as exc:
        _fail("M6_RECOVERY_PATH", str(exc))
    try:
        inventory, staged_payloads = _inventory(staging_root)
    except RecoveryError as exc:
        # A failed scan is itself contradictory evidence.  Do not attempt to
        # repair or remove the tree; retain a receipt describing the refusal.
        value = _receipt(
            decision="CONTRADICTORY",
            staging_root=staging_root,
            inventory=_failure_inventory(staging_root),
            schedule_sha256=schedule_sha256,
            journal_sha256=journal_sha256,
            analysis_sha256=analysis_sha256,
            contradiction_code=exc.code,
        )
        return _emit_receipt(receipt_path, value)
    try:
        expected_files, expected_payloads, expected_entries = _expected_material(
            expected_root
        )
    except RecoveryError as exc:
        value = _receipt(
            decision="CONTRADICTORY",
            staging_root=staging_root,
            inventory=inventory,
            schedule_sha256=schedule_sha256,
            journal_sha256=journal_sha256,
            analysis_sha256=analysis_sha256,
            contradiction_code=exc.code,
        )
        return _emit_receipt(receipt_path, value)
    contradiction, staged_files = _compare_staged_topology(
        inventory,
        expected_entries,
        staged_payloads,
        expected_payloads,
        expected_files,
    )
    final = output_root / final_name
    stage_name = f".{final_name}.staging"
    if output_root.exists():
        _require_current_owner(output_root, "M6 recovery output")
    try:
        immutable.ensure_directory(output_root)
        output_reader = immutable.PinnedDirectoryReader(output_root)
    except (OSError, immutable.ImmutableArtifactError) as exc:
        _fail("M6_RECOVERY_OUTPUT_PARENT", str(exc))
    try:
        if _output_has_conflicting_stage(
            output_reader, final_name=final_name, stage_name=stage_name
        ):
            contradiction = "final_and_stage_coexist"
    finally:
        output_reader.close()
    if contradiction is not None:
        value = _receipt(
            decision="CONTRADICTORY",
            staging_root=staging_root,
            inventory=inventory,
            schedule_sha256=schedule_sha256,
            journal_sha256=journal_sha256,
            analysis_sha256=analysis_sha256,
            contradiction_code=contradiction,
        )
        return _emit_receipt(receipt_path, value)
    # Reclose both source trees immediately before constructing publication
    # material.  A scan performed earlier is not authority for a later commit.
    try:
        latest_inventory, latest_staged_payloads = _inventory(staging_root)
        latest_expected_files, latest_expected_payloads, latest_expected_entries = (
            _expected_material(expected_root)
        )
    except RecoveryError as exc:
        _fail("M6_RECOVERY_INPUT_MIXED", exc.detail)
    if (
        latest_inventory != inventory
        or latest_expected_files != expected_files
        or latest_expected_entries != expected_entries
        or latest_staged_payloads != staged_payloads
        or latest_expected_payloads != expected_payloads
    ):
        _fail("M6_RECOVERY_INPUT_MIXED", "staging or expected tree changed before publication")
    decision = (
        "EXACT_PREFIX_COMPLETE" if staged_files == expected_files
        else "EXACT_PREFIX_PARTIAL"
    )
    material = [
        immutable.ImmutableFile.from_bytes(
            path,
            staged_payloads[path] if path in staged_payloads else expected_payloads[path],
        )
        for path in expected_files
    ]
    try:
        output_reader = immutable.PinnedDirectoryReader(output_root)
        try:
            if _output_has_conflicting_stage(
                output_reader, final_name=final_name, stage_name=stage_name
            ):
                _fail("M6_RECOVERY_OUTPUT_CONFLICT", f"{final_name}/{stage_name}")

            def before_commit(_root: Path, _descriptor: int) -> None:
                _recheck_bound_identity(
                    schedule_path, "schedule", schedule_payload
                )
                _recheck_bound_identity(
                    journal_path, "journal", journal_payload
                )
                _recheck_bound_identity(
                    analysis_path, "analysis", analysis_payload
                )
                try:
                    current_inventory, current_staged_payloads = _inventory(staging_root)
                    current_expected_files, current_expected_payloads, current_expected_entries = (
                        _expected_material(expected_root)
                    )
                except RecoveryError as exc:
                    _fail("M6_RECOVERY_INPUT_MIXED", exc.detail)
                if (
                    current_inventory != inventory
                    or current_staged_payloads != staged_payloads
                    or current_expected_files != expected_files
                    or current_expected_payloads != expected_payloads
                    or current_expected_entries != expected_entries
                ):
                    _fail("M6_RECOVERY_INPUT_MIXED", "source tree changed during publication")

            published = immutable.publish_directory(
                output_root,
                final_name,
                material,
                before_commit=before_commit,
                pinned_root=output_reader,
            )
        finally:
            output_reader.close()
    except RecoveryError:
        raise
    except (OSError, immutable.ImmutableArtifactError) as exc:
        _fail("M6_RECOVERY_PUBLICATION", str(exc))
    _recheck_bound_identity(
        schedule_path, "schedule", schedule_payload
    )
    _recheck_bound_identity(
        journal_path, "journal", journal_payload
    )
    _recheck_bound_identity(
        analysis_path, "analysis", analysis_payload
    )
    value = _receipt(
        decision=decision,
        staging_root=staging_root,
        inventory=inventory,
        schedule_sha256=schedule_sha256,
        journal_sha256=journal_sha256,
        analysis_sha256=analysis_sha256,
        published_path=os.fspath(published),
    )
    return _emit_receipt(receipt_path, value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", required=True, type=Path)
    parser.add_argument("--expected-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--final-name", required=True)
    parser.add_argument("--schedule-sha256", required=True)
    parser.add_argument("--journal-sha256", required=True)
    parser.add_argument("--analysis-sha256", required=True)
    parser.add_argument("--schedule", dest="schedule_path", type=Path)
    parser.add_argument("--journal", dest="journal_path", type=Path)
    parser.add_argument("--analysis", dest="analysis_path", type=Path)
    parser.add_argument(
        "--evidence-root",
        dest="evidence_root",
        type=Path,
        help="root containing journal-referenced sidecars and launch evidence",
    )
    parser.add_argument("--receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        value = recover_staging(
            staging_root=args.staging_root,
            expected_root=args.expected_root,
            output_root=args.output_root,
            final_name=args.final_name,
            schedule_sha256=args.schedule_sha256,
            journal_sha256=args.journal_sha256,
            analysis_sha256=args.analysis_sha256,
            receipt_path=args.receipt,
            schedule_path=args.schedule_path,
            journal_path=args.journal_path,
            analysis_path=args.analysis_path,
            evidence_root=args.evidence_root,
        )
    except (RecoveryError, campaign.M6Error) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"decision={value['decision']} receipt_sha256={value['receipt_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
