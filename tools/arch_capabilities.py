"""Architecture-capability helper backed by `tools/arch_capabilities.json`.

Library API:

  - ``load_manifest()`` — read and parse the JSON manifest.
  - ``lookup_entry(path, make_target=None, manifest=None)`` — resolve
    a manifest entry by ``(path, make_target)``. Returns ``None`` if
    the entry is missing.
  - ``assert_arch_supported(path, arch, make_target=None, manifest=None)``
    — return ``None`` when the entry exists and the arch is in
    ``supported_arches``; otherwise return a typed-reason dict
    ``{"typed_reason": "arch_unsupported", "path": ..., "arch": ...,
    ...}`` so callers can emit the typed reason without re-deriving
    it. The dict's ``detail`` field disambiguates between missing
    manifest entries and unsupported arches.

CLI surface:

  - ``python tools/arch_capabilities.py --path PATH --arch ARCH
    [--make-target TARGET]`` — exits 0 when the manifest entry exists
    and admits the arch, prints the entry on stdout. Exits 1 when the
    entry is missing or the arch is unsupported, prints the
    typed-reason JSON on stderr. The CLI is the operator-facing
    surface that scripts wire in to fail through a typed
    ``arch_unsupported`` path rather than silently returning a
    default value on an unsupported architecture.

Wiring guidance:

  Any operator-facing script or collection helper that accepts or
  infers an architecture should call ``assert_arch_supported`` at
  startup with its own ``__file__``-derived path and the operator-
  supplied (or detected) arch. A non-None return value means the
  caller must abort with the typed reason rather than continue. See
  ``tools/test_arch_capabilities_helper.py`` for the synthetic-
  fixture pattern that exercises this contract end-to-end.

Cloud-deferred entries (kind ``cloud_target_deferred``) carry empty
``supported_arches`` lists per DEC-2; ``assert_arch_supported`` rejects
ANY arch passed for such entries with a typed-reason result whose
``detail`` is ``cloud_target_deferred``. This is the same typed-
rejection shape as a true unsupported-arch case so callers do not
need to special-case the deferred-cloud branch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST_PATH = _REPO_ROOT / "tools" / "arch_capabilities.json"


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    """Load `arch_capabilities.json` and return the parsed dict.

    The default path resolves relative to this file's location so the
    helper works from any cwd. Callers that need to test against a
    synthetic manifest pass a ``path`` directly.
    """
    target = path if path is not None else _MANIFEST_PATH
    with target.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def lookup_entry(
    path: str,
    make_target: str | None = None,
    *,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Resolve a manifest entry by ``(path, make_target)``.

    Returns the first matching entry (the manifest enforces uniqueness
    per ``(path, make_target)``) or ``None`` if no entry matches.
    Callers that want to disambiguate a "no entry" case from
    "unsupported arch" should use ``assert_arch_supported``, which
    returns a typed-reason dict in either case.
    """
    if manifest is None:
        manifest = load_manifest()
    for entry in manifest["entries"]:
        if entry["path"] != path:
            continue
        if entry.get("make_target") != make_target:
            continue
        return entry
    return None


def assert_arch_supported(
    path: str,
    arch: str,
    make_target: str | None = None,
    *,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return ``None`` when ``arch`` is supported by the manifest entry
    keyed by ``(path, make_target)``; otherwise return a typed-reason
    dict.

    The typed-reason dict always has:
      - ``typed_reason`` = ``"arch_unsupported"``
      - ``path``: the queried path
      - ``arch``: the queried arch
      - ``make_target``: optional Makefile workflow target

    The dict's ``detail`` differentiates the rejection cause:
      - ``manifest_entry_missing``: no entry matches ``(path,
        make_target)`` in the manifest.
      - ``cloud_target_deferred``: the entry exists with kind
        ``cloud_target_deferred`` and an empty supported_arches list.
      - ``arch_not_in_supported_list``: the entry exists with a
        non-empty supported_arches list, but ``arch`` is not in it.
    """
    if manifest is None:
        manifest = load_manifest()
    entry = lookup_entry(path, make_target=make_target, manifest=manifest)
    if entry is None:
        return {
            "typed_reason": "arch_unsupported",
            "detail": "manifest_entry_missing",
            "path": path,
            "make_target": make_target,
            "arch": arch,
        }
    supported = entry.get("supported_arches", [])
    if not supported:
        # Cloud-deferred (or otherwise scope-empty) entry: nothing
        # is supported during Loop 2.
        return {
            "typed_reason": "arch_unsupported",
            "detail": "cloud_target_deferred",
            "path": path,
            "make_target": make_target,
            "arch": arch,
            "kind": entry.get("kind"),
        }
    if arch not in supported:
        return {
            "typed_reason": "arch_unsupported",
            "detail": "arch_not_in_supported_list",
            "path": path,
            "make_target": make_target,
            "arch": arch,
            "supported_arches": list(supported),
        }
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve a project-owned entry's architecture support "
            "from `tools/arch_capabilities.json`. Exits 0 when the "
            "entry admits the arch; exits 1 with a typed-reason "
            "JSON on stderr when the entry is missing or the arch "
            "is unsupported."
        )
    )
    parser.add_argument(
        "--path",
        required=True,
        help=(
            "Repository-relative path of the entry-point file "
            "(e.g. `tools/local_labels/label_driver.sh`)."
        ),
    )
    parser.add_argument(
        "--arch",
        required=True,
        help=(
            "Architecture string to check against supported_arches "
            "(e.g. `sm_86`, `sm_120-laptop`)."
        ),
    )
    parser.add_argument(
        "--make-target",
        default=None,
        help=(
            "Optional Makefile workflow target name. When set, the "
            "lookup matches against an entry whose `make_target` "
            "field equals this value."
        ),
    )
    args = parser.parse_args(argv)

    typed = assert_arch_supported(
        path=args.path,
        arch=args.arch,
        make_target=args.make_target,
    )
    if typed is None:
        entry = lookup_entry(args.path, make_target=args.make_target)
        # When `assert_arch_supported` returns None the entry must
        # exist and admit the arch; otherwise it would have returned
        # a typed-reason dict above. The assert documents the
        # invariant for readers and silences static-typing warnings
        # about subscripting an Optional.
        assert entry is not None, (
            "lookup_entry returned None after assert_arch_supported "
            "admitted the arch — the manifest entry must exist when "
            "the arch is supported."
        )
        sys.stdout.write(
            f"ok: {args.path}"
            + (f" ({args.make_target})" if args.make_target else "")
            + f" admits arch {args.arch}; "
            + f"supported_arches="
            + json.dumps(entry["supported_arches"])
            + "\n"
        )
        return 0

    sys.stderr.write(
        json.dumps(typed, sort_keys=True, ensure_ascii=False) + "\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
