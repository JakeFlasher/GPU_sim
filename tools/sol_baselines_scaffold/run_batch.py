"""Operator-callable batch driver for corpus-expansion batches.

Wraps the existing scaffold + pack + validate chain
(`scripts/baselines_scaffold/{scaffold,pack,validate}.py`) so a
single helper call drives the chain per slug in a
corpus-expansion batch manifest. The driver runs in two modes:

  * ``dry_run=True`` (default) — resolves each slug's
    ``(category, language, exemplar, target_dir, bench_dir)``
    tuple and reports per-slug status without writing anything.
    Useful as a smoke / pre-flight against a freshly authored
    batch manifest before execution.

  * ``dry_run=False`` — invokes ``scaffold()`` then ``pack()``
    then ``validate_baseline()`` for each slug. Per-slug
    failures do not abort the batch; each slug's status is
    captured in the returned ``ScaffoldBatchResult`` so the
    operator can review every outcome.

Per-slug native ``sol-execbench`` correctness runs live OUTSIDE
this driver: they require an authored kernel body and a real
GPU build that exceed the batch driver's automation scope. The
driver gates each slug as far as the scaffold + pack + validate
contract layer; the operator then handles per-slug kernel
authoring + correctness separately.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from baselines_scaffold.generate import (  # noqa: E402
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_OUTPUT_ROOT,
)
from baselines_scaffold.scaffold import (  # noqa: E402
    DEFAULT_EXAMPLES_ROOT,
    ScaffoldError,
    resolve_benchmark_dir,
    resolve_category_for_slug,
    resolve_exemplar_dir,
    scaffold,
)
from baselines_scaffold.pack import (  # noqa: E402
    PackError,
    pack,
)
from baselines_scaffold.validate import (  # noqa: E402
    ValidationError,
    validate_baseline,
)
from baselines_harness.calibration.cache_prediction.corpus_expansion_batch import (  # noqa: E402
    CORPUS_EXPANSION_BATCH_SCHEMA_VERSION,
    CorpusExpansionBatchError,
    load_corpus_expansion_batch_manifest,
)
from baselines_harness.calibration.cache_prediction.verified_dispatch_families import (  # noqa: E402
    VerifiedDispatchFamiliesError,
    load_verified_dispatch_families,
)


DEFAULT_MAPPING_PATH: Path = (
    _REPO_ROOT / "tools" / "sol_baselines_scaffold" / "sol_pattern_mapping.md"
)

DEFAULT_VERIFIED_DISPATCH_FAMILIES_PATH: Path = (
    _REPO_ROOT / ".humanize" / "corpus-expansion"
    / "verified_dispatch_families.toml"
)


SCAFFOLD_BATCH_TYPED_REASONS: frozenset[str] = frozenset({
    "scaffold_batch_manifest_load_failed",
    "scaffold_batch_slug_missing_from_mapping",
    "scaffold_batch_exemplar_missing",
    "scaffold_batch_target_dir_exists",
    "scaffold_batch_bench_dir_missing",
    "scaffold_batch_scaffold_failed",
    "scaffold_batch_pack_failed",
    "scaffold_batch_validate_failed",
    "scaffold_batch_verified_dispatch_families_load_failed",
})


STATUS_DRY_RUN_RESOLVED: str = "dry_run_resolved"
STATUS_EXECUTED: str = "executed"
STATUS_FAILED: str = "failed"


class ScaffoldBatchError(Exception):
    """Typed error raised by the batch driver.

    `typed_reason` is drawn from the closed set
    `SCAFFOLD_BATCH_TYPED_REASONS`.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        if typed_reason not in SCAFFOLD_BATCH_TYPED_REASONS:
            raise ValueError(
                f"unknown typed_reason {typed_reason!r}; expected "
                f"one of {sorted(SCAFFOLD_BATCH_TYPED_REASONS)}",
            )
        self.typed_reason = typed_reason
        self.message = message


@dataclass(frozen=True)
class ScaffoldBatchSlugEntry:
    """Per-slug result of the batch driver.

    `status` ∈ {`dry_run_resolved`, `executed`, `failed`}.
    When `status == "failed"`, `typed_reason` is one of the
    closed-set reasons in `SCAFFOLD_BATCH_TYPED_REASONS`;
    otherwise `typed_reason` is `None`. `category`,
    `exemplar`, `target_dir`, `bench_dir` are populated to the
    extent the driver could resolve them before the failure.
    """

    slug: str
    category: str | None
    language: str
    exemplar: str | None
    target_dir: Path | None
    bench_dir: Path | None
    status: str
    typed_reason: str | None
    detail: str | None


@dataclass(frozen=True)
class ScaffoldBatchResult:
    """Aggregate batch-driver result.

    `entries` carries one `ScaffoldBatchSlugEntry` per
    manifest slug, in manifest order.
    """

    manifest_path: Path
    dry_run: bool
    entries: tuple[ScaffoldBatchSlugEntry, ...]


def default_exemplar_picker(
    slug: str, category: str, language: str,
) -> str:
    """Pick an exemplar path for `(slug, category, language)`.

    Heuristic:
      * `language == "cutlass"` → `"cutlass/gemm"`.
      * `language == "cuda_cpp"` AND slug contains `"rmsnorm"`
        or `"_norm_"` or ends in `"_norm"` → `"cuda_cpp/rmsnorm"`.
      * Otherwise → `"cuda_cpp/flux_rope"` (the most general
        elementwise / RoPE starter).
    """
    if language == "cutlass":
        return "cutlass/gemm"
    if language == "cuda_cpp":
        lowered = slug.lower()
        if (
            "rmsnorm" in lowered
            or "_norm_" in lowered
            or lowered.endswith("_norm")
        ):
            return "cuda_cpp/rmsnorm"
    return "cuda_cpp/flux_rope"


def _resolve_per_slug(
    slug: str,
    language: str,
    exemplar_picker: Callable[[str, str, str], str],
    mapping_path: Path,
    benchmark_root: Path,
    examples_root: Path,
    output_root: Path,
) -> tuple[
    str | None, str | None, Path | None, Path | None,
    str | None, str | None,
]:
    """Resolve per-slug `(category, exemplar, target_dir,
    bench_dir, typed_reason, detail)` for `slug`.

    Returns `(typed_reason, detail)` as `(None, None)` on
    success; otherwise the typed reason from the closed set
    plus a free-text detail.
    """
    try:
        category = resolve_category_for_slug(slug, mapping_path)
    except ScaffoldError as exc:
        return (
            None, None, None, None,
            "scaffold_batch_slug_missing_from_mapping",
            str(exc),
        )

    exemplar = exemplar_picker(slug, category, language)
    try:
        resolve_exemplar_dir(exemplar, examples_root)
    except ScaffoldError as exc:
        return (
            category, exemplar, None, None,
            "scaffold_batch_exemplar_missing",
            str(exc),
        )

    try:
        bench_dir = resolve_benchmark_dir(
            slug, category, benchmark_root,
        )
    except ScaffoldError as exc:
        return (
            category, exemplar, None, None,
            "scaffold_batch_bench_dir_missing",
            str(exc),
        )

    target_dir = output_root / category / slug / language
    if target_dir.exists():
        return (
            category, exemplar, target_dir, bench_dir,
            "scaffold_batch_target_dir_exists",
            f"target_dir already exists: {target_dir}",
        )
    return (category, exemplar, target_dir, bench_dir, None, None)


def run_scaffold_batch(
    manifest_path: Path,
    *,
    dry_run: bool = True,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    benchmark_root: Path = DEFAULT_BENCHMARK_ROOT,
    examples_root: Path = DEFAULT_EXAMPLES_ROOT,
    mapping_path: Path = DEFAULT_MAPPING_PATH,
    exemplar_picker: Callable[[str, str, str], str] = default_exemplar_picker,
    canonical_slug_set: Iterable[str] | None = None,
    host_arch_unsupported_slug_set: Iterable[str] | None = None,
    verified_dispatch_families_path: Path | None = None,
    scaffold_fn: Callable[..., Path] = scaffold,
    pack_fn: Callable[..., Path] = pack,
    validate_fn: Callable[..., Any] = validate_baseline,
) -> ScaffoldBatchResult:
    """Drive the scaffold + pack + validate chain for every
    slug in `manifest_path`.

    The first manifest-load failure raises
    `ScaffoldBatchError("scaffold_batch_manifest_load_failed")`;
    per-slug failures are captured in the returned result
    rather than raising, so the operator can review every
    outcome in one pass.

    Cadence-uplift gating: the verified-dispatch-families
    manifest at `verified_dispatch_families_path` (default:
    `.humanize/corpus-expansion/verified_dispatch_families.toml`)
    is loaded and its `family_status_map` is passed to the
    batch loader. v1.1 `friendly_multi_slug` batches with an
    unknown or `pending_verification` `dispatch_family_signature`
    are surfaced as
    `ScaffoldBatchError("scaffold_batch_manifest_load_failed")`
    wrapping the underlying typed `CorpusExpansionBatchError`.
    If the verified-families manifest cannot be loaded AND the
    batch is v1.1, raises
    `ScaffoldBatchError("scaffold_batch_verified_dispatch_families_load_failed")`;
    v1.0 legacy batches continue to scaffold when the
    verified-families manifest is absent or invalid.

    `scaffold_fn`, `pack_fn`, `validate_fn` are injection
    seams for tests; production callers leave them at the
    default values.
    """
    if verified_dispatch_families_path is None:
        verified_dispatch_families_path = (
            DEFAULT_VERIFIED_DISPATCH_FAMILIES_PATH
        )
    family_status_map: dict[str, str] | None = None
    verified_load_error: VerifiedDispatchFamiliesError | None = None
    try:
        vdf_manifest = load_verified_dispatch_families(
            verified_dispatch_families_path,
        )
        family_status_map = dict(vdf_manifest.family_status_map)
    except VerifiedDispatchFamiliesError as exc:
        verified_load_error = exc

    try:
        manifest = load_corpus_expansion_batch_manifest(
            manifest_path,
            canonical_slug_set=canonical_slug_set,
            host_arch_unsupported_slug_set=(
                host_arch_unsupported_slug_set
            ),
            verified_dispatch_families=family_status_map,
        )
    except CorpusExpansionBatchError as exc:
        raise ScaffoldBatchError(
            "scaffold_batch_manifest_load_failed",
            f"failed to load batch manifest at "
            f"{manifest_path!s}: {exc.typed_reason}: "
            f"{exc.message}",
        ) from exc

    if (
        manifest.schema_version
        == CORPUS_EXPANSION_BATCH_SCHEMA_VERSION
        and verified_load_error is not None
    ):
        raise ScaffoldBatchError(
            "scaffold_batch_verified_dispatch_families_load_failed",
            f"verified-dispatch-families manifest at "
            f"{verified_dispatch_families_path!s} could not be "
            f"loaded ({verified_load_error.typed_reason}: "
            f"{verified_load_error.message}); v1.1 batch "
            f"manifests require a valid verified-families "
            f"manifest for cadence-uplift gating",
        ) from verified_load_error

    entries: list[ScaffoldBatchSlugEntry] = []
    for slug_entry in manifest.slugs:
        slug = slug_entry.slug
        language = slug_entry.target_language
        (
            category, exemplar, target_dir, bench_dir,
            typed_reason, detail,
        ) = _resolve_per_slug(
            slug, language, exemplar_picker, mapping_path,
            benchmark_root, examples_root, output_root,
        )
        if typed_reason is not None:
            entries.append(ScaffoldBatchSlugEntry(
                slug=slug,
                category=category,
                language=language,
                exemplar=exemplar,
                target_dir=target_dir,
                bench_dir=bench_dir,
                status=STATUS_FAILED,
                typed_reason=typed_reason,
                detail=detail,
            ))
            continue

        if dry_run:
            entries.append(ScaffoldBatchSlugEntry(
                slug=slug,
                category=category,
                language=language,
                exemplar=exemplar,
                target_dir=target_dir,
                bench_dir=bench_dir,
                status=STATUS_DRY_RUN_RESOLVED,
                typed_reason=None,
                detail=None,
            ))
            continue

        try:
            scaffold_fn(
                slug=slug,
                language=language,
                exemplar=exemplar,
                output_root=output_root,
                benchmark_root=benchmark_root,
                examples_root=examples_root,
                mapping_path=mapping_path,
            )
        except Exception as exc:  # noqa: BLE001
            entries.append(ScaffoldBatchSlugEntry(
                slug=slug, category=category, language=language,
                exemplar=exemplar, target_dir=target_dir,
                bench_dir=bench_dir, status=STATUS_FAILED,
                typed_reason="scaffold_batch_scaffold_failed",
                detail=f"{type(exc).__name__}: {exc}",
            ))
            continue

        try:
            pack_fn(target_dir)
        except Exception as exc:  # noqa: BLE001
            entries.append(ScaffoldBatchSlugEntry(
                slug=slug, category=category, language=language,
                exemplar=exemplar, target_dir=target_dir,
                bench_dir=bench_dir, status=STATUS_FAILED,
                typed_reason="scaffold_batch_pack_failed",
                detail=f"{type(exc).__name__}: {exc}",
            ))
            continue

        try:
            validate_fn(target_dir, benchmark_root=benchmark_root)
        except Exception as exc:  # noqa: BLE001
            entries.append(ScaffoldBatchSlugEntry(
                slug=slug, category=category, language=language,
                exemplar=exemplar, target_dir=target_dir,
                bench_dir=bench_dir, status=STATUS_FAILED,
                typed_reason="scaffold_batch_validate_failed",
                detail=f"{type(exc).__name__}: {exc}",
            ))
            continue

        entries.append(ScaffoldBatchSlugEntry(
            slug=slug, category=category, language=language,
            exemplar=exemplar, target_dir=target_dir,
            bench_dir=bench_dir, status=STATUS_EXECUTED,
            typed_reason=None, detail=None,
        ))

    return ScaffoldBatchResult(
        manifest_path=manifest_path,
        dry_run=dry_run,
        entries=tuple(entries),
    )


__all__ = [
    "DEFAULT_MAPPING_PATH",
    "DEFAULT_VERIFIED_DISPATCH_FAMILIES_PATH",
    "SCAFFOLD_BATCH_TYPED_REASONS",
    "STATUS_DRY_RUN_RESOLVED",
    "STATUS_EXECUTED",
    "STATUS_FAILED",
    "ScaffoldBatchError",
    "ScaffoldBatchResult",
    "ScaffoldBatchSlugEntry",
    "default_exemplar_picker",
    "run_scaffold_batch",
]
