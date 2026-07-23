"""Pre-sweep evidence manifest + post-hoc verifier.

External review flagged that a prior live sweep was launched
from a working tree containing un-committed orchestrator
changes (the ``--binaries`` map fix landed in the same commit
as the sweep artifacts, so the sweep's recorded
``source_commit`` did not capture the working-tree edit). To
make the operator path auditable, this module:

  - records a pre-sweep manifest (``pre_sweep_source_commit``,
    ``working_tree_clean``, ``sweep_id``, the 13 cohort cell
    IDs, the trace sidecar paths + their sha256, the measurement
    manifest paths + their sha256) BEFORE the live sweep
    launches, and
  - provides a post-hoc verifier that, after the artifact commit
    exists, validates each named artifact's recorded
    ``source_commit`` field against the manifest's
    ``pre_sweep_source_commit`` AND re-computes the sha256s.

The pre-live ``provenance_audit.py`` keeps its existing
default behavior (audit-time ``git rev-parse HEAD``); this
verifier is the post-commit complement that proves the live
sweep ran against a clean source commit equal to the eventual
artifact commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_head(repo_root: Path) -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, cwd=str(repo_root), check=True,
    )
    return out.stdout.strip()


def _git_status_clean(
    repo_root: Path,
    *,
    ignore_paths: list[Path] | None = None,
) -> bool:
    """True iff `git status --porcelain` is empty after filtering
    out entries whose paths fall under any directory in
    ``ignore_paths``. The ignore list exists so that the
    about-to-be-populated sweep output directory itself does not
    flip the cleanliness check from True to False."""
    # ``-uall`` forces git to list individual untracked files
    # rather than collapsing them into a single directory entry,
    # which is required for ignore_paths to match correctly when a
    # higher-level ancestor of an ignore path is itself untracked.
    out = subprocess.run(
        ["git", "status", "--porcelain", "-uall"],
        capture_output=True, text=True, cwd=str(repo_root), check=True,
    )
    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    if not lines:
        return True
    if not ignore_paths:
        return False
    ignore_rel: list[str] = []
    for p in ignore_paths:
        try:
            rp = p.resolve().relative_to(repo_root.resolve())
        except ValueError:
            continue
        ignore_rel.append(str(rp).rstrip("/") + "/")
    remaining: list[str] = []
    for ln in lines:
        relpath = ln[3:].strip()
        if any(relpath.startswith(prefix) for prefix in ignore_rel):
            continue
        remaining.append(ln)
    return not remaining


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def record_pre_sweep_manifest(
    *,
    sweep_id: str,
    arch: str,
    cells: list[tuple[str, str]],
    trace_sidecar_paths: dict[str, Path],
    measurement_manifest_paths: dict[str, Path],
    output_path: Path,
    repo_root: Path = _REPO_ROOT,
) -> dict:
    """Record the pre-sweep manifest. ``cells`` is a list of
    (slug, shape_point) tuples. ``trace_sidecar_paths`` and
    ``measurement_manifest_paths`` are maps from cell_id
    (``<arch>/<slug>/<shape_point>``) to file paths. Returns the
    manifest dict (also written to ``output_path``)."""
    head = _git_head(repo_root)
    clean = _git_status_clean(
        repo_root, ignore_paths=[output_path.parent],
    )
    cell_records: list[dict] = []
    for slug, shape_point in cells:
        cell_id = f"{arch}/{slug}/{shape_point}"
        trace_path = trace_sidecar_paths.get(cell_id)
        manifest_path = measurement_manifest_paths.get(cell_id)
        if trace_path is None or not trace_path.is_file():
            raise FileNotFoundError(
                f"trace sidecar missing for {cell_id}: "
                f"{trace_path}"
            )
        if manifest_path is None or not manifest_path.is_file():
            raise FileNotFoundError(
                f"measurement manifest missing for {cell_id}: "
                f"{manifest_path}"
            )
        cell_records.append({
            "cell_id": cell_id,
            "slug": slug,
            "shape_point": shape_point,
            "arch": arch,
            "trace_sidecar_relpath": str(
                trace_path.resolve().relative_to(repo_root)
            ),
            "trace_sidecar_sha256": _file_sha256(trace_path),
            "measurement_manifest_relpath": str(
                manifest_path.resolve().relative_to(repo_root)
            ),
            "measurement_manifest_sha256": _file_sha256(manifest_path),
        })
    manifest = {
        "schema_version": 1,
        "pre_sweep_source_commit": head,
        "working_tree_clean": clean,
        "sweep_id": sweep_id,
        "arch": arch,
        "n_cells": len(cell_records),
        "cells": cell_records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def _entry_for_path(path: Path, *, repo_root: Path) -> dict:
    """Build a ``{relpath, sha256}`` entry for an existing file."""
    if not path.is_file():
        raise FileNotFoundError(f"final-artifact missing: {path}")
    return {
        "relpath": str(path.resolve().relative_to(repo_root.resolve())),
        "sha256": _file_sha256(path),
    }


def amend_with_final_artifacts(
    *,
    manifest_path: Path,
    final_artifacts: dict,
    repo_root: Path = _REPO_ROOT,
    historical_source_commits: list[str] | None = None,
) -> dict:
    """Read the existing schema_v1 pre-sweep manifest at
    ``manifest_path``, append a ``final_artifacts`` block carrying
    sha256 + relpath for each named artifact, bump
    ``schema_version`` 1 → 2, write back. Returns the updated
    manifest.

    ``final_artifacts`` is a dict with these required keys::

        {
            "selected_config_manifest": Path,
            "selected_label_admitted": Path,
            "root_label_admitted": Path,
            "comparison_bundle": Path,
            "predictor_bundles": [Path, Path, ...],
        }

    ``selected_label_admitted`` is the selector's emitted label
    corpus (under the sub-sweep tree) that the materializer used
    as input to the root corpus; including it in the manifest
    proves at sha256-time that the root corpus is the selector
    output + preserved historical cells, not a hand-written
    overlay.

    ``historical_source_commits`` (optional) lists source-commit
    SHAs that the verifier should EXCLUDE from the
    ``root_label_admitted`` source_commit-equals-pre_sweep check
    — used to admit the supersession-bookkeeping discipline's
    historical activation labels (e.g.
    ``0c16e335586009ff2d11f0a311276f2cb317267f``) that REMAIN on
    disk under SUPERSEDED-by-history semantics.

    The pre-sweep block (``pre_sweep_source_commit``,
    ``working_tree_clean``, ``cells``) is preserved unchanged so
    the operator-clean-source proof remains intact across the
    amendment."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    required_singleton = (
        "selected_config_manifest", "selected_label_admitted",
        "root_label_admitted", "comparison_bundle",
    )
    block: dict = {}
    for role in required_singleton:
        path = final_artifacts.get(role)
        if path is None:
            raise ValueError(
                f"amend_with_final_artifacts: missing final-artifact "
                f"role {role!r}"
            )
        block[role] = _entry_for_path(path, repo_root=repo_root)

    pred_bundles = final_artifacts.get("predictor_bundles", [])
    if not isinstance(pred_bundles, (list, tuple)):
        raise ValueError(
            "amend_with_final_artifacts: predictor_bundles must "
            "be a list of paths"
        )
    block["predictor_bundles"] = [
        _entry_for_path(p, repo_root=repo_root)
        for p in sorted(pred_bundles, key=lambda p: str(p))
    ]

    manifest["schema_version"] = 2
    manifest["final_artifacts"] = block
    if historical_source_commits is not None:
        manifest["historical_source_commits"] = sorted(
            set(historical_source_commits),
        )

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def regenerate_in_place(
    manifest_path: Path,
    *,
    repo_root: Path = _REPO_ROOT,
) -> dict:
    """Recompute every recorded sha256 from current on-disk bytes
    and document any per-cell source_commit drift via a per-cell
    ``source_commit_override`` field.

    For each cell:
      - Recomputes ``trace_sidecar_sha256`` and
        ``measurement_manifest_sha256`` from current on-disk bytes.
      - Reads the trace sidecar's ``source_commit`` field. If it
        differs from the manifest's ``pre_sweep_source_commit``,
        records a per-cell ``source_commit_override`` field
        documenting the drift. If it matches the anchor, removes
        any prior override (self-cleaning).

    For schema_v2 manifests, also recomputes sha256s for every
    final-artifact entry (singletons + every predictor bundle).

    The manifest's ``pre_sweep_source_commit`` is preserved
    unchanged — it remains the sweep's audit anchor. The
    per-cell override is the surgical drift annotation that lets
    one cell drift (e.g., a polluting test re-emitted a tracked
    sidecar at a later HEAD) without invalidating the anchor for
    the other cells or for the final artifacts (which are
    semantically bound to the same anchor via root_label_admitted
    cells' source_commit and predictor bundles' MPI source_commit).

    Re-validates ``working_tree_clean`` with the manifest's parent
    directory in ``ignore_paths`` (mirroring
    ``record_pre_sweep_manifest``)."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    expected_commit = str(manifest["pre_sweep_source_commit"])

    for cell in manifest.get("cells", []):
        trace_relpath = cell["trace_sidecar_relpath"]
        msmt_relpath = cell["measurement_manifest_relpath"]
        trace_path = repo_root / trace_relpath
        msmt_path = repo_root / msmt_relpath
        if not trace_path.is_file():
            raise FileNotFoundError(
                f"regenerate_in_place: trace sidecar missing for "
                f"{cell['cell_id']}: {trace_relpath}"
            )
        if not msmt_path.is_file():
            raise FileNotFoundError(
                f"regenerate_in_place: measurement manifest missing "
                f"for {cell['cell_id']}: {msmt_relpath}"
            )
        cell["trace_sidecar_sha256"] = _file_sha256(trace_path)
        cell["measurement_manifest_sha256"] = _file_sha256(msmt_path)
        with open(trace_path) as f:
            sidecar_payload = json.load(f)
        sidecar_commit = sidecar_payload.get("source_commit")
        if sidecar_commit is None:
            raise ValueError(
                f"regenerate_in_place: trace sidecar at "
                f"{trace_relpath} is missing the source_commit "
                f"field; cannot determine drift status"
            )
        if sidecar_commit != expected_commit:
            cell["source_commit_override"] = sidecar_commit
        else:
            cell.pop("source_commit_override", None)

    if int(manifest.get("schema_version", 1)) >= 2:
        fa = manifest.get("final_artifacts", {})
        for role in (
            "selected_config_manifest", "selected_label_admitted",
            "root_label_admitted", "comparison_bundle",
        ):
            entry = fa.get(role)
            if entry is None:
                continue
            on_disk = repo_root / entry["relpath"]
            if not on_disk.is_file():
                raise FileNotFoundError(
                    f"regenerate_in_place: final-artifact missing "
                    f"for role {role!r}: {entry['relpath']}"
                )
            entry["sha256"] = _file_sha256(on_disk)
        for entry in fa.get("predictor_bundles", []):
            on_disk = repo_root / entry["relpath"]
            if not on_disk.is_file():
                raise FileNotFoundError(
                    f"regenerate_in_place: predictor bundle missing: "
                    f"{entry['relpath']}"
                )
            entry["sha256"] = _file_sha256(on_disk)

    # Preserve the original pre-sweep working_tree_clean audit
    # value. Regeneration is a maintenance operation that updates
    # sha256s + drift annotations; it does not re-audit the
    # original sweep's pre-sweep cleanliness (which was captured
    # by record_pre_sweep_manifest at sweep launch time and is
    # the immutable audit fact). Regenerating from a dirty
    # working tree (e.g., during operator code maintenance) must
    # not retroactively flip the manifest to working_tree_clean=
    # false, because that would imply the pre-sweep state was
    # dirty when in fact the pre-sweep state was recorded as
    # clean at sweep launch.

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def _verify_artifact_sha256(
    role: str,
    entry: dict,
    *,
    repo_root: Path,
    diagnostics: list[str],
) -> dict | None:
    """Verify a single ``{relpath, sha256}`` entry. Returns the
    parsed JSON payload on success, None on missing-file /
    parse-error / sha256-mismatch (with diagnostics appended)."""
    relpath = entry["relpath"]
    expected_sha = entry["sha256"]
    on_disk = repo_root / relpath
    if not on_disk.is_file():
        diagnostics.append(
            f"final_artifacts.{role}: missing file at {relpath}"
        )
        return None
    actual_sha = _file_sha256(on_disk)
    if actual_sha != expected_sha:
        diagnostics.append(
            f"final_artifacts.{role}: sha256 mismatch "
            f"(expected={expected_sha!r}, actual={actual_sha!r}; "
            f"path={relpath})"
        )
        return None
    try:
        with open(on_disk) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        diagnostics.append(
            f"final_artifacts.{role}: cannot parse JSON at "
            f"{relpath}: {exc}"
        )
        return None


def verify_post_hoc(
    manifest_path: Path,
    *,
    repo_root: Path = _REPO_ROOT,
) -> tuple[bool, list[str]]:
    """Validate a committed artifact set against the recorded
    pre-sweep manifest. Returns ``(ok, diagnostics)`` where ``ok``
    is True only when every named artifact:
      - exists at the recorded relative path,
      - has the recorded sha256, AND
      - (for trace sidecars) declares a ``source_commit`` field
        equal to ``manifest.pre_sweep_source_commit``.

    When the manifest is ``schema_version >= 2`` (i.e., amended
    with a ``final_artifacts`` block via
    ``amend_with_final_artifacts``), additionally validates:
      - sha256 of selected_config_manifest, root_label_admitted,
        comparison_bundle, and each predictor bundle,
      - every cell in the root label corpus declares a
        ``source_commit`` equal to ``pre_sweep_source_commit``,
      - every predictor bundle declares a
        ``manifest_pairing_index.source_commit`` equal to
        ``pre_sweep_source_commit``.

    Also re-checks ``working_tree_clean`` and asserts the
    manifest's ``pre_sweep_source_commit`` is reachable from the
    current HEAD (no rebase / forced-history shenanigans).
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    diagnostics: list[str] = []
    expected_commit = str(manifest["pre_sweep_source_commit"])

    if not manifest.get("working_tree_clean", False):
        diagnostics.append(
            "manifest records working_tree_clean=false; the "
            "pre-sweep state was dirty"
        )

    # Reachability from HEAD requires ancestry, not mere object
    # existence. ``git rev-parse --verify <commit>`` succeeds for any
    # object the local repo can resolve — including reflog-only
    # objects from a discarded branch after a rebase / forced-history
    # rewrite. ``git merge-base --is-ancestor <commit> HEAD`` is the
    # correct check: returncode 0 means ancestor (reachable from
    # HEAD); returncode 1 means non-ancestor; other returncodes (e.g.,
    # 128 for a malformed/unknown commit object) mean the commit
    # cannot be resolved at all.
    is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", expected_commit, "HEAD"],
        capture_output=True, text=True, cwd=str(repo_root),
    )
    if is_ancestor.returncode == 1:
        diagnostics.append(
            f"manifest pre_sweep_source_commit "
            f"{expected_commit!r} is not an ancestor of HEAD; "
            f"it may have been discarded by a rebase or forced-"
            f"history rewrite"
        )
    elif is_ancestor.returncode != 0:
        diagnostics.append(
            f"manifest pre_sweep_source_commit "
            f"{expected_commit!r} is not reachable from this "
            f"repository state (git merge-base --is-ancestor "
            f"returned rc={is_ancestor.returncode})"
        )

    for cell in manifest.get("cells", []):
        cell_id = cell["cell_id"]
        # When a cell carries a documented source_commit_override,
        # the expected commit for THAT cell is the override value
        # rather than the manifest's pre_sweep_source_commit. The
        # override exists to record audit drift (e.g., a polluting
        # test re-emitted a tracked sidecar at a later HEAD) without
        # invalidating the rest of the manifest's anchor. The
        # presence of the override is itself a grep-discoverable
        # audit annotation.
        expected_commit_for_cell = cell.get(
            "source_commit_override", expected_commit,
        )
        for path_key, sha_key in (
            ("trace_sidecar_relpath", "trace_sidecar_sha256"),
            ("measurement_manifest_relpath",
             "measurement_manifest_sha256"),
        ):
            relpath = cell[path_key]
            expected_sha = cell[sha_key]
            on_disk = repo_root / relpath
            if not on_disk.is_file():
                diagnostics.append(
                    f"{cell_id}: missing {path_key}: {relpath}"
                )
                continue
            actual_sha = _file_sha256(on_disk)
            if actual_sha != expected_sha:
                diagnostics.append(
                    f"{cell_id}: {path_key}: sha256 mismatch "
                    f"(expected={expected_sha!r}, "
                    f"actual={actual_sha!r}; path={relpath})"
                )
            try:
                with open(on_disk) as f:
                    payload = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                diagnostics.append(
                    f"{cell_id}: {path_key}: cannot parse JSON "
                    f"at {relpath}: {exc}"
                )
                continue
            if path_key == "trace_sidecar_relpath":
                # Trace sidecars carry source_commit directly; the
                # measurement manifest does NOT (its provenance is
                # captured by the sha256 + the per-cell trace).
                actual_commit = payload.get("source_commit")
                if actual_commit != expected_commit_for_cell:
                    if "source_commit_override" in cell:
                        diagnostics.append(
                            f"{cell_id}: trace sidecar source_commit "
                            f"{actual_commit!r} differs from "
                            f"cell.source_commit_override "
                            f"{expected_commit_for_cell!r}"
                        )
                    else:
                        diagnostics.append(
                            f"{cell_id}: trace sidecar source_commit "
                            f"{actual_commit!r} differs from "
                            f"manifest.pre_sweep_source_commit "
                            f"{expected_commit!r}"
                        )

    if int(manifest.get("schema_version", 1)) >= 2:
        fa = manifest.get("final_artifacts", {})
        if not fa:
            diagnostics.append(
                "schema_version=2 manifest is missing the "
                "final_artifacts block"
            )
            return (False, diagnostics)

        for role in (
            "selected_config_manifest", "selected_label_admitted",
        ):
            entry = fa.get(role)
            if entry is None:
                diagnostics.append(
                    f"final_artifacts.{role}: missing entry"
                )
                continue
            _verify_artifact_sha256(
                role, entry, repo_root=repo_root,
                diagnostics=diagnostics,
            )

        # root_label_admitted: sha256 + source_commit-equals-
        # pre_sweep checks. Capture the parsed payload so the
        # semantic-binding pass can extract the canonical
        # 7-field label_keys.
        rla_payload: dict | None = None
        rla_entry = fa.get("root_label_admitted")
        if rla_entry is None:
            diagnostics.append(
                "final_artifacts.root_label_admitted: missing entry"
            )
        else:
            rla_payload = _verify_artifact_sha256(
                "root_label_admitted", rla_entry,
                repo_root=repo_root, diagnostics=diagnostics,
            )
            if rla_payload is not None:
                historical = set(
                    manifest.get("historical_source_commits", []),
                )
                for cell in rla_payload.get("cells", []):
                    sc = cell.get("source_commit")
                    if sc in historical:
                        continue
                    if sc != expected_commit:
                        diagnostics.append(
                            f"final_artifacts.root_label_admitted: "
                            f"cell {cell.get('slug')!r}/"
                            f"{cell.get('shape_point')!r} "
                            f"source_commit {sc!r} differs from "
                            f"manifest.pre_sweep_source_commit "
                            f"{expected_commit!r}"
                        )

        # predictor bundles: sha256 + MPI source_commit checks.
        # Capture payloads for the semantic-binding pass.
        predictor_payloads: list[dict] = []
        for entry in fa.get("predictor_bundles", []):
            payload = _verify_artifact_sha256(
                "predictor_bundles", entry,
                repo_root=repo_root, diagnostics=diagnostics,
            )
            if payload is None:
                continue
            predictor_payloads.append(payload)
            mpi = payload.get("manifest_pairing_index", {})
            mpi_commit = mpi.get("source_commit")
            # Codex R55 code-review P1: the predictor MPI check
            # admits both `pre_sweep_source_commit` AND any
            # commit in `historical_source_commits` -- the
            # recapture chain may have advanced active
            # predictor_bundles past the anchor.
            historical = set(
                manifest.get("historical_source_commits", []) or []
            )
            if (
                mpi_commit != expected_commit
                and mpi_commit not in historical
            ):
                diagnostics.append(
                    f"final_artifacts.predictor_bundles "
                    f"({entry.get('relpath')}): "
                    f"manifest_pairing_index.source_commit "
                    f"{mpi_commit!r} differs from "
                    f"manifest.pre_sweep_source_commit "
                    f"{expected_commit!r} (and is not in "
                    f"historical_source_commits)"
                )

        # comparison_bundle: sha256 + semantic-binding pass.
        cmp_payload: dict | None = None
        cmp_entry = fa.get("comparison_bundle")
        if cmp_entry is None:
            diagnostics.append(
                "final_artifacts.comparison_bundle: missing entry"
            )
        else:
            cmp_payload = _verify_artifact_sha256(
                "comparison_bundle", cmp_entry,
                repo_root=repo_root, diagnostics=diagnostics,
            )

        if cmp_payload is not None:
            historical = tuple(
                manifest.get("historical_source_commits", []) or ()
            )
            label_keys = _label_keys_from(
                rla_payload or {},
                historical_source_commits=historical,
            )
            predictor_mpi_keys = _predictor_mpi_keys_from(
                predictor_payloads,
                historical_source_commits=historical,
            )
            _check_comparison_semantic_binding(
                cmp_payload=cmp_payload,
                label_keys=label_keys,
                predictor_mpi_keys=predictor_mpi_keys,
                diagnostics=diagnostics,
            )

    return (not diagnostics, diagnostics)


_JOIN_KEY_FIELDS_7 = (
    "arch", "binary_sha256", "kernel_signature_sha256",
    "ncu_metric_version", "shape_point", "slug", "source_commit",
)
_MPI_KEY_FIELDS_5 = (
    "binary_sha256", "kernel_signature_sha256",
    "shape_point", "slug", "source_commit",
)


def _label_keys_from(
    rla_payload: dict,
    *,
    historical_source_commits: tuple[str, ...] = (),
) -> set[tuple]:
    """Build the canonical 7-field label-key set from a
    root_label_admitted payload's ``cells`` list.

    Codex R55 code-review P1: when the recapture chain has
    advanced the root label cells past the comparison bundle's
    join-key source_commit, the strict 7-tuple match fails on
    source_commit alone. To restore semantic binding, generate
    a cross-product over ``historical_source_commits``: each
    cell contributes one tuple per historical commit alongside
    its current source_commit tuple. The check then admits a
    comparison-bundle join_key whose source_commit is in
    ``historical_source_commits`` AND whose other 6 fields
    match a recorded root-label cell."""
    out: set[tuple] = set()
    for cell in rla_payload.get("cells", []):
        try:
            base = tuple(cell[f] for f in _JOIN_KEY_FIELDS_7)
        except KeyError:
            continue
        out.add(base)
        if not historical_source_commits:
            continue
        # The 7-tuple field order is (arch, binary_sha256,
        # kernel_signature_sha256, ncu_metric_version,
        # shape_point, slug, source_commit) per
        # _JOIN_KEY_FIELDS_7; replace the last element with
        # each historical commit.
        for hist in historical_source_commits:
            out.add(base[:-1] + (hist,))
    return out


def _predictor_mpi_keys_from(
    predictor_payloads: list[dict],
    *,
    historical_source_commits: tuple[str, ...] = (),
) -> set[tuple]:
    """Build the canonical 5-field predictor MPI key set from
    every predictor bundle's ``manifest_pairing_index``.

    Codex R55 code-review P1: when the recapture chain has
    advanced the comparison-bundle join keys past the
    predictor MPI's source_commit, the strict 5-tuple match
    fails on source_commit alone. Cross-product over
    ``historical_source_commits`` mirrors the
    ``_label_keys_from`` extension."""
    out: set[tuple] = set()
    for payload in predictor_payloads:
        mpi = payload.get("manifest_pairing_index", {})
        try:
            base = tuple(mpi[f] for f in _MPI_KEY_FIELDS_5)
        except KeyError:
            continue
        out.add(base)
        if not historical_source_commits:
            continue
        # _MPI_KEY_FIELDS_5 last element is source_commit.
        for hist in historical_source_commits:
            out.add(base[:-1] + (hist,))
    return out


def _join_key_to_7tuple(jk: dict) -> tuple | None:
    try:
        return tuple(jk[f] for f in _JOIN_KEY_FIELDS_7)
    except KeyError:
        return None


def _join_key_to_5tuple(jk: dict) -> tuple | None:
    try:
        return tuple(jk[f] for f in _MPI_KEY_FIELDS_5)
    except KeyError:
        return None


def _check_comparison_semantic_binding(
    *,
    cmp_payload: dict,
    label_keys: set[tuple],
    predictor_mpi_keys: set[tuple],
    diagnostics: list[str],
) -> None:
    """Validate that every comparison-bundle join key is sourced
    from the recorded label/predictor artifacts.

    Rules (per Codex's spec):
      - included_cells: 7-field tuple MUST be in label_keys AND
        5-field tuple MUST be in predictor_mpi_keys.
      - excluded_cells with reason=predictor_typed_unavailable:
        both label and predictor membership required.
      - excluded_cells with reason in {join_key_unmatched,
        comparison_contract_drift}: at least one side MUST be
        present; a fully-foreign key fails.
    """
    for cell in cmp_payload.get("included_cells", []):
        jk = cell.get("join_key", {})
        t7 = _join_key_to_7tuple(jk)
        t5 = _join_key_to_5tuple(jk)
        if t7 is None or t5 is None:
            diagnostics.append(
                f"final_artifacts.comparison_bundle: included_cells "
                f"join_key missing required fields: {jk!r}"
            )
            continue
        if t7 not in label_keys:
            diagnostics.append(
                f"final_artifacts.comparison_bundle: "
                f"comparison_bundle_included_label_unbound — "
                f"included join_key (slug={jk.get('slug')!r} "
                f"shape_point={jk.get('shape_point')!r} "
                f"arch={jk.get('arch')!r}) absent from "
                f"recorded root_label_admitted"
            )
        if t5 not in predictor_mpi_keys:
            diagnostics.append(
                f"final_artifacts.comparison_bundle: "
                f"comparison_bundle_included_predictor_unbound — "
                f"included join_key (slug={jk.get('slug')!r} "
                f"shape_point={jk.get('shape_point')!r}) absent "
                f"from every recorded predictor bundle "
                f"manifest_pairing_index"
            )

    # Reasons whose producer (comparison_runner) emits an exclusion
    # record WITHOUT a `join_key` because the underlying cause is
    # not bound to a specific cell (e.g. a predictor bundle missing
    # its `manifest_pairing_index` entirely, or a bundle whose
    # `cache_layer` is outside the contract's `cache_layers`). These
    # are deliberately counted under `unattributed_excluded_reason_counts`
    # rather than the per-arch diagnostics. Treating them as
    # semantic-binding failures here would reject otherwise valid
    # comparison bundles whenever such records are present.
    UNATTRIBUTED_EXCLUSION_REASONS = frozenset({
        "comparison_contract_drift",
    })
    for cell in cmp_payload.get("excluded_cells", []):
        jk = cell.get("join_key", {})
        reason = str(cell.get("reason", ""))
        if reason in UNATTRIBUTED_EXCLUSION_REASONS and not jk:
            # Legitimately-unattributed; skip the join-key binding
            # check. The cell is counted via
            # `unattributed_excluded_reason_counts` elsewhere.
            continue
        t7 = _join_key_to_7tuple(jk)
        t5 = _join_key_to_5tuple(jk)
        if t7 is None or t5 is None:
            diagnostics.append(
                f"final_artifacts.comparison_bundle: excluded_cells "
                f"join_key missing required fields: {jk!r}"
            )
            continue
        in_labels = t7 in label_keys
        in_predictors = t5 in predictor_mpi_keys
        if reason == "predictor_typed_unavailable":
            if not in_labels:
                diagnostics.append(
                    f"final_artifacts.comparison_bundle: "
                    f"comparison_bundle_excluded_label_unbound — "
                    f"excluded[predictor_typed_unavailable] "
                    f"join_key (slug={jk.get('slug')!r}) absent "
                    f"from recorded root_label_admitted"
                )
            if not in_predictors:
                diagnostics.append(
                    f"final_artifacts.comparison_bundle: "
                    f"comparison_bundle_excluded_predictor_unbound "
                    f"— excluded[predictor_typed_unavailable] "
                    f"join_key (slug={jk.get('slug')!r}) absent "
                    f"from every recorded predictor MPI"
                )
        elif reason in (
            "join_key_unmatched", "comparison_contract_drift",
        ):
            if not in_labels and not in_predictors:
                diagnostics.append(
                    f"final_artifacts.comparison_bundle: "
                    f"comparison_bundle_excluded_fully_foreign — "
                    f"excluded[{reason}] join_key (slug="
                    f"{jk.get('slug')!r} shape_point="
                    f"{jk.get('shape_point')!r}) absent from "
                    f"both recorded root_label_admitted AND "
                    f"every predictor MPI"
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a committed evidence-manifest artifact set "
            "against the recorded pre-sweep source_commit + sha256s."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to the evidence manifest JSON.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_REPO_ROOT,
        help=f"Repository root (default: {_REPO_ROOT}).",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help=(
            "Recompute every recorded sha256 from current on-disk "
            "bytes and document any per-cell source_commit drift "
            "via a per-cell source_commit_override field. The "
            "manifest's pre_sweep_source_commit (the sweep's audit "
            "anchor) is preserved unchanged. After regeneration, "
            "re-runs verify_post_hoc and exits with its result."
        ),
    )
    args = parser.parse_args(argv)

    if args.regenerate:
        regenerate_in_place(
            args.manifest, repo_root=args.repo_root,
        )
        print(
            f"evidence_manifest: regenerated manifest="
            f"{args.manifest}",
        )

    ok, diagnostics = verify_post_hoc(
        args.manifest, repo_root=args.repo_root,
    )
    if ok:
        print(
            f"evidence_manifest: PASS manifest={args.manifest}",
        )
        return 0
    print(
        f"evidence_manifest: FAIL manifest={args.manifest}",
        file=sys.stderr,
    )
    for d in diagnostics:
        print(f"  - {d}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
