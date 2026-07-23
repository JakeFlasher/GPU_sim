"""One-shot driver: move pre-R28 activation labels (source_commit
matching a named anchor) from label_admitted.json to
label_excluded.json with typed_reason=superseded_by_launch_skip_fix.

This is a one-shot operator script for the A1 sub-deliverable
(plan task7). It runs once per label-corpus state and is NOT
expected to be re-invoked after the corpus stabilizes.

The script delegates the actual mutation to the
``label_supersession.supersede_cells`` helper (which gates the
typed-reason against the closed enum). The script's job is to
identify WHICH (arch, slug, shape_point, source_commit) rows in
the current corpus match the operator's chosen anchor and to
pass that filtered list to the helper.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "tools" / "local_labels"))

from label_supersession import (  # noqa: E402
    _ALLOWED_EXCLUSION_REASONS,
    supersede_cells,
)


_TARGET_ACTIVATION_IDENTITIES: tuple[tuple[str, str, str], ...] = (
    ("sm_86", "layernorm_fwd_bf16_sm80", "timing_shape"),
    ("sm_86", "rmsnorm_fwd_bf16_sm80", "timing_shape"),
    ("sm_86", "silu_fwd_bf16_sm80", "correctness_shape"),
    ("sm_86", "silu_fwd_bf16_sm80", "timing_shape"),
    ("sm_86", "softmax_safe_bf16_sm80", "timing_shape"),
)


def _find_demote_keys(
    admitted: dict, source_commit_prefix: str,
) -> list[tuple[str, str, str, str]]:
    """Return the source-aware 4-tuple keys for every admitted row
    whose (arch, slug, shape_point) is in the activation identity
    set AND whose source_commit starts with the named prefix."""
    keys: list[tuple[str, str, str, str]] = []
    target_set = set(_TARGET_ACTIVATION_IDENTITIES)
    for cell in admitted["cells"]:
        arch = str(cell.get("arch") or cell.get("device") or "")
        slug = str(cell.get("slug") or "")
        shape_point = str(cell.get("shape_point") or "")
        identity = (arch, slug, shape_point)
        if identity not in target_set:
            continue
        source_commit = str(cell.get("source_commit", ""))
        if not source_commit.startswith(source_commit_prefix):
            continue
        keys.append((arch, slug, shape_point, source_commit))
    return keys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Move pre-launch-skip-fix activation labels from "
            "label_admitted.json to label_excluded.json with "
            "typed_reason=superseded_by_launch_skip_fix. Delegates "
            "the mutation to label_supersession.supersede_cells "
            "with source-aware 4-tuple keys."
        ),
    )
    parser.add_argument(
        "--source-commit-prefix",
        required=True,
        help=(
            "Prefix of source_commit to identify the pre-fix row "
            "set (e.g., '0c16e3355860' or '7189cda8e43f')."
        ),
    )
    parser.add_argument(
        "--detail",
        required=True,
        help="Free-text detail recorded on each demoted record.",
    )
    parser.add_argument(
        "--admitted-path",
        default=str(_REPO_ROOT / "data" / "local_labels" / "label_admitted.json"),
        help="Path to label_admitted.json (default: data/local_labels/label_admitted.json).",
    )
    parser.add_argument(
        "--excluded-path",
        default=str(_REPO_ROOT / "data" / "local_labels" / "label_excluded.json"),
        help="Path to label_excluded.json (default: data/local_labels/label_excluded.json).",
    )
    args = parser.parse_args(argv)

    if "superseded_by_launch_skip_fix" not in _ALLOWED_EXCLUSION_REASONS:
        raise ValueError(
            "superseded_by_launch_skip_fix not in the helper's "
            "_ALLOWED_EXCLUSION_REASONS enum"
        )

    admitted_path = Path(args.admitted_path)
    excluded_path = Path(args.excluded_path)
    with admitted_path.open("r", encoding="utf-8") as fh:
        admitted = json.load(fh)
    with excluded_path.open("r", encoding="utf-8") as fh:
        excluded = json.load(fh)

    demote_keys = _find_demote_keys(admitted, args.source_commit_prefix)
    if not demote_keys:
        print(
            f"no admitted rows match source_commit prefix "
            f"{args.source_commit_prefix!r} in the activation identity set",
            file=sys.stderr,
        )
        return 1

    print(
        f"demoting {len(demote_keys)} admitted rows whose source_commit "
        f"starts with {args.source_commit_prefix!r}:",
        file=sys.stderr,
    )
    for arch, slug, shape_point, source_commit in demote_keys:
        print(
            f"  {arch}/{slug}/{shape_point} src={source_commit[:12]}",
            file=sys.stderr,
        )

    new_admitted, new_excluded = supersede_cells(
        admitted=admitted,
        excluded=excluded,
        cell_keys_to_demote=demote_keys,
        typed_reason="superseded_by_launch_skip_fix",
        detail=args.detail,
    )

    admitted_path.write_text(
        json.dumps(new_admitted, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    excluded_path.write_text(
        json.dumps(new_excluded, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"after: admitted={len(new_admitted['cells'])}, "
        f"excluded={len(new_excluded['cells'])}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
