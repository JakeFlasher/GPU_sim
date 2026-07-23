"""Seal the holdout-cell truth labels for the regenerated cohort
split into ``sealed_holdout_labels.json``.

The sealed file locks each holdout cell's NCU truth values behind the
operational-blinding contract enforced by ``label_loader`` (the
calibration/validation loader REFUSES to open this path; only the
once-only holdout-evaluation runner reads it under
``holdout_unblind_context``). Sealing it now — bound to the regenerated
cohort's manifest digest — supersedes the stale pre-relabel seal so a
later holdout evaluation cannot silently reuse the old digest.

Each holdout cell's label row is pulled by its 5-field identity from
the label artifact the cohort manifest declares for that cell
(``expected_label_path``): the canonical ``label_admitted.json`` for the
pre-existing admitted cells, and the fresh
``task6_recapture_admitted_labels.json`` for the re-captured cells. No
label is fabricated — a holdout cell with no matching label row is a
fail-closed error.

    python -m tools.sm_86_predictor_accuracy_recovery_v2.seal_holdout_labels
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from generate_cohort_split import _canonical_manifest_digest  # noqa: E402

_DATA_DIR = _REPO_ROOT / "data" / "sm_86_predictor_accuracy_recovery_v2"
_DEFAULT_MANIFEST_PATH = _DATA_DIR / "cohort_manifest.json"
_DEFAULT_SPLIT_PATH = _DATA_DIR / "cohort_split.json"
_DEFAULT_SEALED_PATH = _DATA_DIR / "sealed_holdout_labels.json"

_NOTE = (
    "Holdout-cell truth values. This file is the sealed-labels surface "
    "for the submission-track operational-blinding contract: the "
    "calibration- and validation-mode label loader must REFUSE to open "
    "this path. Only the holdout-evaluation runner (after a passing "
    "validation pass and with the unblind_token flipped via "
    "holdout_unblind_context) may read it, once. Regenerated for the "
    "live-reconciled admitted union recorded in the bound manifest; "
    "the cohort_manifest_digest binds this seal to that exact manifest "
    "so a stale-digest reuse after a model/threshold change is "
    "rejected. Labels are pulled by identity from the manifest's "
    "declared expected_label_path per cell; none are fabricated."
)


def _identity5(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("arch", "")),
        str(row.get("slug", "")),
        str(row.get("shape_point", "")),
        str(row.get("binary_sha256", "")),
        str(row.get("kernel_signature_sha256", "")),
    )


def _load_label_index(path: Path) -> dict[tuple, dict]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {_identity5(row): row for row in payload.get("cells", ())}


def seal(
    manifest_path: Path | None = None,
    split_path: Path | None = None,
    out_path: Path | None = None,
) -> dict:
    manifest_path = manifest_path or _DEFAULT_MANIFEST_PATH
    split_path = split_path or _DEFAULT_SPLIT_PATH
    out_path = out_path or _DEFAULT_SEALED_PATH

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    holdout_cells = split["splits"]["holdout"]["cells"]

    # Index the manifest's admitted cells so we can resolve each holdout
    # cell's declared expected_label_path.
    manifest_by_identity = {
        _identity5(c["identity_tuple"]): c
        for c in manifest["cells"]
        if c["admission_status"] == "admitted"
    }
    # Cache label indices per resolved label file.
    label_indices: dict[Path, dict] = {}

    sealed_cells: list[dict] = []
    for cell in holdout_cells:
        ident = _identity5(cell["identity_tuple"])
        mcell = manifest_by_identity.get(ident)
        if mcell is None:
            raise ValueError(
                f"seal_holdout_labels: holdout cell {ident!r} is not an "
                f"admitted manifest cell"
            )
        label_path = (_REPO_ROOT / mcell["expected_label_path"]).resolve()
        if label_path not in label_indices:
            label_indices[label_path] = _load_label_index(label_path)
        row = label_indices[label_path].get(ident)
        if row is None:
            raise ValueError(
                f"seal_holdout_labels: no label row for holdout cell "
                f"{ident!r} in {mcell['expected_label_path']}"
            )
        sealed_cells.append(row)

    sealed_cells.sort(key=lambda r: (r.get("slug", ""), r.get("shape_point", "")))
    sealed = {
        "schema_version": 1,
        "frozen_at_commit": manifest["frozen_at_commit"],
        "cohort_manifest_digest": _canonical_manifest_digest(manifest),
        "note": _NOTE,
        "cells": sealed_cells,
    }
    out_path.write_text(
        json.dumps(sealed, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sealed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Seal holdout truth labels for the regenerated cohort split."
    )
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST_PATH)
    parser.add_argument("--split", type=Path, default=_DEFAULT_SPLIT_PATH)
    parser.add_argument("--out", type=Path, default=_DEFAULT_SEALED_PATH)
    args = parser.parse_args(argv)
    sealed = seal(
        manifest_path=args.manifest, split_path=args.split, out_path=args.out
    )
    sys.stdout.write(
        f"seal_holdout_labels: wrote {args.out} "
        f"({len(sealed['cells'])} holdout cells; digest "
        f"{sealed['cohort_manifest_digest'][:12]}...)\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
