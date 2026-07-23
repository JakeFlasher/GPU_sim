"""Inventory candidate slugs for the sm_86 submission-track cohort
expansion.

Walks two binary-corpus roots and one trace-corpus root, computes
``binary_sha256`` for each on-disk binary, classifies the slug into
a family bucket via a small lookup table, and emits a JSON catalog
at ``data/sm_86_predictor_accuracy_recovery_v2/expansion_candidates.json``.

The catalog is consumed by the manifest expansion step that lifts
``cohort_manifest.json``'s attempted scope from the seed 13 cells
toward the publication-target gate of >= 31 admitted / >= 30
unique slugs. Each catalog entry binds a candidate slug to:

  - ``binary_path``  : repo-relative path to the on-disk binary
  - ``binary_sha256``: sha256 hex of the binary
  - ``trace_present``: whether a trace exists under
    ``workloads/simbench_bw/results/<slug>/rtx_3090_ti/sm_86/``
  - ``family_bucket``: one of the cohort schema's closed-enum
    family buckets
  - ``proposed_typed_reason``: the typed exclusion reason the
    manifest should adopt when adding this slug as
    ``attempted_excluded`` (``trace_missing`` when no trace is
    present yet; ``label_missing`` when a trace is present but no
    NCU label has been generated).

Per CLAUDE.md Rule 1: this script reads only on-disk files; it
never invokes NCU, the NVBit tracer, or any GPU runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PORTABLE_LCD_ROOT = _REPO_ROOT / "workloads" / "portable_lcd"
_SIMBENCH_L1_ROOT = _REPO_ROOT / "workloads" / "simbench_bw" / "l1"
_TRACE_ROOT = _REPO_ROOT / "workloads" / "simbench_bw" / "results"
_DEFAULT_OUTPUT_PATH = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "expansion_candidates.json"
)

# Family-bucket lookup. Each key is a substring matched against the
# slug name (longest-first ordering ensures specific matches like
# ``flash_attn`` beat generic matches like ``attn``).
_FAMILY_BUCKET_LOOKUP: tuple[tuple[str, str], ...] = (
    # gemm_family
    ("hmma_gemm", "gemm_family"),
    ("imma_gemm", "gemm_family"),
    ("gemm_hmma", "gemm_family"),
    ("gemm_imma", "gemm_family"),
    ("gemm_sparse", "gemm_family"),
    ("blockscale_gemm", "gemm_family"),
    # attention_conv
    ("flash_attn", "attention_conv"),
    ("conv2d", "attention_conv"),
    ("cross_attn", "attention_conv"),
    # memory_stress
    ("dram_stream", "memory_stress"),
    ("l2_thrash", "memory_stress"),
    ("l2_persisting", "memory_stress"),
    ("l2_prefetch", "memory_stress"),
    ("warm_cold", "memory_stress"),
    ("carveout", "memory_stress"),
    ("working_set", "memory_stress"),
    ("cp_async_shared", "memory_stress"),
    ("shared_bank_conflict", "memory_stress"),
    ("bank_conflict", "memory_stress"),
    # irregular_hostile
    ("register_spill", "irregular_hostile"),
    ("global_atomic", "irregular_hostile"),
    ("embedding_bag", "irregular_hostile"),
    ("fma_dep_chain", "irregular_hostile"),
    ("warp_divergence", "irregular_hostile"),
    # dl_primitives (lowest-priority fallbacks for activation-style kernels)
    ("layernorm", "dl_primitives"),
    ("rmsnorm", "dl_primitives"),
    ("silu", "dl_primitives"),
    ("softmax", "dl_primitives"),
    ("swiglu", "dl_primitives"),
    ("cross_entropy", "dl_primitives"),
    ("embedding_gather", "dl_primitives"),
    ("groupnorm", "dl_primitives"),
    ("geglu", "dl_primitives"),
    ("kv_cache", "dl_primitives"),
)


@dataclass(frozen=True)
class ExpansionCandidate:
    """One candidate slug, binding the on-disk binary to a proposed
    cohort-manifest entry."""

    slug: str
    binary_path: str
    binary_sha256: str
    trace_present: bool
    family_bucket: str
    proposed_typed_reason: str


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _classify_family_bucket(slug: str) -> str:
    """Return the family bucket for ``slug`` by substring match.
    Defaults to ``vendor_baselines`` for slugs that match no rule;
    the operator can override in the manifest. The fallback bucket
    matches the schema's closed enum even when the slug name is
    novel."""
    for needle, bucket in _FAMILY_BUCKET_LOOKUP:
        if needle in slug:
            return bucket
    return "vendor_baselines"


def _trace_present_for(slug: str) -> bool:
    """The slug counts as having a trace when
    ``workloads/simbench_bw/results/<slug>/rtx_3090_ti/sm_86/``
    exists on disk."""
    return (_TRACE_ROOT / slug / "rtx_3090_ti" / "sm_86").is_dir()


def _proposed_typed_reason(trace_present: bool) -> str:
    """A candidate that lacks a trace under
    ``workloads/simbench_bw/results/<slug>/rtx_3090_ti/sm_86/`` is
    queued as ``trace_missing``; one whose trace exists but no
    NCU label has yet landed in ``data/local_labels/`` is queued
    as ``label_missing``."""
    return "label_missing" if trace_present else "trace_missing"


def _walk_binary_corpus(root: Path) -> Iterable[tuple[str, Path]]:
    """Yield ``(slug, binary_path)`` pairs for every ``<slug>_run``
    binary directly under ``root/<slug>/``."""
    if not root.is_dir():
        return
    for slug_dir in sorted(root.iterdir()):
        if not slug_dir.is_dir():
            continue
        slug = slug_dir.name
        binary = slug_dir / f"{slug}_run"
        if binary.is_file():
            yield slug, binary


def collect_candidates() -> tuple[ExpansionCandidate, ...]:
    """Walk both binary corpora and return the deduplicated set of
    candidate slugs, sorted by slug for stable output."""
    seen: dict[str, ExpansionCandidate] = {}
    for root in (_PORTABLE_LCD_ROOT, _SIMBENCH_L1_ROOT):
        for slug, binary in _walk_binary_corpus(root):
            if slug in seen:
                continue
            binary_rel = str(binary.relative_to(_REPO_ROOT))
            sha = _sha256_of_file(binary)
            family = _classify_family_bucket(slug)
            trace_present = _trace_present_for(slug)
            seen[slug] = ExpansionCandidate(
                slug=slug,
                binary_path=binary_rel,
                binary_sha256=sha,
                trace_present=trace_present,
                family_bucket=family,
                proposed_typed_reason=_proposed_typed_reason(trace_present),
            )
    return tuple(sorted(seen.values(), key=lambda c: c.slug))


def emit_catalog(
    candidates: Iterable[ExpansionCandidate],
    output_path: Path,
) -> None:
    """Emit the catalog as a JSON object with a stable shape:
    ``schema_version``, ``candidates_count``, and ``candidates`` as
    a list of per-slug records."""
    candidates_list = list(candidates)
    payload = {
        "schema_version": 1,
        "candidates_count": len(candidates_list),
        "candidates": [asdict(c) for c in candidates_list],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")
    tmp_path.replace(output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory sm_86 candidate slugs for cohort expansion. "
            "Walks pre-built binaries under workloads/portable_lcd/ "
            "and workloads/simbench_bw/l1/ and emits a JSON catalog "
            "binding each candidate slug to its real binary_sha256 "
            "+ family bucket + trace presence + proposed typed "
            "exclusion reason."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=_DEFAULT_OUTPUT_PATH,
        help=(
            "Output JSON catalog path. Default: "
            "data/sm_86_predictor_accuracy_recovery_v2/expansion_candidates.json"
        ),
    )
    args = parser.parse_args(argv)
    candidates = collect_candidates()
    emit_catalog(candidates, args.output)
    print(
        f"wrote {len(candidates)} candidate(s) to "
        f"{args.output.relative_to(_REPO_ROOT)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
