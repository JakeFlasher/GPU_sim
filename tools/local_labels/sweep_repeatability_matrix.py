"""Parameterized NCU repeatability sweep harness.

Wraps ``tools/local_labels/label_driver.sh`` in nested loops over the
matrix axes (warmup_count × cache_control × repeats) for an operator-
chosen candidate cell set. Aggregates per-cell-per-config sidecars via
``repeatability_aggregator.py`` and emits a verdict-table (CSV + JSON)
plus a terminal verdict (``admitting_config_found`` or
``no_admitting_config_found``).

Output layout::

    <out-root>/sweep_<UTC-timestamp>_<digest>/
        verdict-table.csv
        verdict-table.json
        verdict.json
        cell=<arch>__<slug>__<shape>/cfg=warmup<N>_cache_<mode>/
            <arch>/<slug>/<shape>/label_<repeat>_0.json
            ...

A candidate may optionally carry a build-variant suffix
(``arch/slug/shape_point/register_cap_<cap>``); when present the per-cell
output directory and the ``cell_id`` widen to
``cell=<arch>__<slug>__<shape>__register_cap_<cap>`` so two register-cap
build variants of the same shape are distinct, non-collapsing cells. When
absent, every derived value is byte-identical to the historical 3-field cell.

The harness's only output prefix is ``data/local_labels/sweeps/``;
attempting to write outside that declared evidence sub-prefix produces a
typed CLI diagnostic. Matrix iteration is sorted across all axes so
two invocations with the same ``--seed`` produce a byte-identical
verdict-table.

Modes
-----

``--dry-run``
    Prints the planned ``label_driver.sh`` invocation per (cell,
    warmup_count, cache_control, repeat) without invoking either
    label_driver.sh or NCU. The printed line includes the warmup
    count, cache-control mode, and per-repeat indices so an operator
    can audit the matrix before committing wall-clock budget.

``--live``
    Invokes ``label_driver.sh`` for every (cell, warmup_count,
    cache_control, repeat) combination, aggregates the per-cell
    sidecars, records the per-config admission outcome in the
    verdict table, and emits the terminal verdict.

The default (and only valid) value set for ``--cache-controls`` is
``{all, none}`` mirroring NCU's CLI: ``all`` flushes caches between
replays; ``none`` retains them. The harness rejects any other value at
parse time.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


_HERE = Path(__file__).resolve().parent
_DEFAULT_DRIVER = _HERE / "label_driver.sh"
_DEFAULT_AGGREGATOR = _HERE / "repeatability_aggregator.py"
_REPO_ROOT = _HERE.parents[1]
_SIMBENCH_SHAPES_DIR = _REPO_ROOT / "workloads" / "simbench_bw" / "shapes"

# Pre-flight provenance audit module (sibling under
# ``tools/local_labels/``). The harness imports the library API
# rather than spawning a subprocess so the audit's typed
# diagnostics surface inline before any NCU shim runs.
sys.path.insert(0, str(_HERE))
import provenance_audit as _provenance_audit  # noqa: E402
sys.path.pop(0)

# The harness's only output prefix. Any --out-root that is not under
# this declared evidence sub-prefix is rejected at parse time. NCU artifacts
# are confined to the declared evidence namespace, and the sweep harness
# narrows further to ``data/local_labels/sweeps/``.
_QUARANTINE_PREFIX = "data/local_labels/sweeps"
_DEFAULT_OUTPUT_ROOT = Path(_QUARANTINE_PREFIX)

# Allowed values for --cache-controls. Mirrors NCU's CLI vocabulary;
# any other value is rejected as an unsupported NCU CLI option (the
# harness does not silently fall back).
_ALLOWED_CACHE_CONTROLS = ("all", "none")

# stddev threshold for admission. Pinned by
# ``comparison_contract.toml`` and the existing repeatability_aggregator
# default. The harness MUST NOT loosen this; the static-source-scan
# discipline asserts the literal 0.03 stays in the aggregator source.
_STDDEV_ADMISSION_THRESHOLD = 0.03

# Build-variant suffix prefix. A 4-part candidate token's final segment
# (``register_cap_<cap>``) names the register-cap build-variant axis. The
# axis is OPTIONAL: a 3-part token has no variant and behaves exactly as the
# historical harness.
_VARIANT_PREFIX = "register_cap_"

# Required CSV columns. Pinned by the harness contract; tests assert
# every column appears. Adding a column is allowed (the ``register_cap`` /
# ``variant_id`` columns are appended for build-variant cells and are empty
# strings for base cells); removing one is a contract break.
_VERDICT_TABLE_COLUMNS = (
    "cell_id",
    "arch",
    "slug",
    "shape_point",
    "warmup_count",
    "cache_control",
    "repeats",
    "n_sidecars",
    "stddev_l2_sector",
    "stddev_l1tex_sector",
    "mean_l2_sector",
    "admitted",
    "detail",
    "register_cap",
    "variant_id",
)


@dataclass(frozen=True)
class CellId:
    arch: str
    slug: str
    shape_point: str
    # Optional register-cap build-variant axis. When ``None`` the cell
    # identity is the historical 3-field ``(arch, slug, shape_point)`` and
    # every derived value (cell_id, sort key, output directory, verdict row)
    # is byte-identical to the pre-variant harness. When set, the cap widens
    # the identity so two build variants of the same shape are distinct,
    # non-collapsing cells. It is never used as a runtime shape axis.
    register_cap: int | None = None

    @classmethod
    def parse(cls, token: str) -> "CellId":
        parts = token.split("/")
        if len(parts) not in (3, 4):
            raise ValueError(
                f"candidate token must have 3 slash-separated parts "
                f"(arch/slug/shape_point) or 4 with a build-variant suffix "
                f"(arch/slug/shape_point/register_cap_<cap>); got {token!r}"
            )
        arch, slug, shape_point = parts[0], parts[1], parts[2]
        if not (arch and slug and shape_point):
            raise ValueError(
                f"candidate token has empty arch/slug/shape_point: {token!r}"
            )
        register_cap = cls._parse_variant_suffix(parts[3]) if len(parts) == 4 else None
        return cls(
            arch=arch, slug=slug, shape_point=shape_point,
            register_cap=register_cap,
        )

    @staticmethod
    def _parse_variant_suffix(token: str) -> int:
        if not token.startswith(_VARIANT_PREFIX):
            raise ValueError(
                f"build-variant suffix must be 'register_cap_<cap>'; "
                f"got {token!r}"
            )
        cap_str = token[len(_VARIANT_PREFIX):]
        if not cap_str.isdigit():
            raise ValueError(
                f"register_cap value must be a non-negative integer; "
                f"got {cap_str!r} in {token!r}"
            )
        return int(cap_str)

    @property
    def variant_axis(self) -> str | None:
        """The build-variant axis name when this cell carries one, else None."""
        return "register_cap" if self.register_cap is not None else None

    @property
    def variant_id(self) -> str | None:
        """``register_cap_<cap>`` when this cell carries a variant, else None."""
        return (
            f"{_VARIANT_PREFIX}{self.register_cap}"
            if self.register_cap is not None else None
        )

    @property
    def cell_id(self) -> str:
        base = f"{self.arch}/{self.slug}/{self.shape_point}"
        return f"{base}/{self.variant_id}" if self.register_cap is not None else base

    @property
    def output_dir_name(self) -> str:
        """The ``cell=...`` run-directory segment, variant-suffixed when set."""
        base = f"cell={self.arch}__{self.slug}__{self.shape_point}"
        return f"{base}__{self.variant_id}" if self.register_cap is not None else base

    @property
    def sort_key(self) -> tuple[str, str, str, int]:
        """Deterministic sort key. ``-1`` for the absent variant sorts base
        cells before any variant of the same shape AND keeps the relative
        order of a base-only candidate set byte-identical to before."""
        return (
            self.arch, self.slug, self.shape_point,
            self.register_cap if self.register_cap is not None else -1,
        )


@dataclass(frozen=True)
class MatrixConfig:
    warmup_count: int
    cache_control: str
    repeats: int


@dataclass(frozen=True)
class SweepArgs:
    mode: str  # "dry-run" or "live"
    candidates: tuple[CellId, ...]
    binaries: dict[CellId, Path]
    measurement_manifests: dict[CellId, Path]
    unit_metrics: tuple[str, ...]
    warmup_counts: tuple[int, ...]
    cache_controls: tuple[str, ...]
    repeats: int
    seed: int
    ncu: str
    driver: Path
    aggregator: Path
    out_root: Path


def _parse_candidates(raw: str) -> tuple[CellId, ...]:
    tokens = [t for t in raw.split(",") if t]
    if not tokens:
        raise ValueError("--candidates must list at least one cell")
    cells = tuple(sorted(
        (CellId.parse(t) for t in tokens),
        key=lambda c: c.sort_key,
    ))
    return cells


def _parse_binaries(raw: str) -> dict[CellId, Path]:
    out: dict[CellId, Path] = {}
    for entry in raw.split(","):
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(
                f"--binaries entry must be "
                f"'arch/slug/shape_point[/register_cap_<cap>]=path'; "
                f"got {entry!r}"
            )
        token, path_str = entry.split("=", 1)
        cell = CellId.parse(token)
        path = Path(path_str)
        out[cell] = path
    return out


def _parse_measurement_manifests(raw: str) -> dict[CellId, Path]:
    """Parse a ``--measurement-manifests`` map of the form
    ``arch/slug/shape_point[/register_cap_<cap>]=path,...``.

    The format mirrors ``--binaries`` (the optional build-variant suffix
    keys a variant cell distinctly). Each candidate cell must name its
    operator-supplied manifest path; the label_driver consumes the
    manifest's ``kernel_signature_sha256`` instead of the locally-computed
    fallback (the manifest is the operator-pinned source-of-truth for the
    kernel-signature value).
    """
    out: dict[CellId, Path] = {}
    for entry in raw.split(","):
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(
                f"--measurement-manifests entry must be "
                f"'arch/slug/shape_point[/register_cap_<cap>]=path'; "
                f"got {entry!r}"
            )
        token, path_str = entry.split("=", 1)
        cell = CellId.parse(token)
        path = Path(path_str)
        out[cell] = path
    return out


def _parse_predictor_trace_jsons(raw: str) -> dict[CellId, Path]:
    """Parse a ``--predictor-trace-jsons`` map of the form
    ``arch/slug/shape_point=path,arch/slug/shape_point=path,...``.

    The format mirrors ``--measurement-manifests``. Each entry
    names the predictor-side per-shape trace JSON sidecar that
    the pre-flight provenance audit will cross-check against the
    matching cell's measurement manifest.
    """
    out: dict[CellId, Path] = {}
    for entry in raw.split(","):
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(
                f"--predictor-trace-jsons entry must be "
                f"'arch/slug/shape_point[/register_cap_<cap>]=path'; "
                f"got {entry!r}"
            )
        token, path_str = entry.split("=", 1)
        cell = CellId.parse(token)
        path = Path(path_str)
        out[cell] = path
    return out


def _parse_int_list(raw: str, name: str) -> tuple[int, ...]:
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError as exc:
            raise ValueError(
                f"{name} must be a comma-separated list of integers; "
                f"got token {token!r}"
            ) from exc
    if not values:
        raise ValueError(f"{name} must list at least one integer")
    if any(v < 0 for v in values):
        raise ValueError(f"{name} integers must be non-negative")
    return tuple(sorted(set(values)))


def _parse_cache_controls(raw: str) -> tuple[str, ...]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("--cache-controls must list at least one mode")
    bad = [v for v in values if v not in _ALLOWED_CACHE_CONTROLS]
    if bad:
        raise ValueError(
            f"unsupported cache_control value(s) {bad!r}; "
            f"allowed: {list(_ALLOWED_CACHE_CONTROLS)!r}"
        )
    return tuple(sorted(set(values)))


def _parse_unit_metrics(values: Sequence[str]) -> tuple[str, ...]:
    """Parse repeatable ``--unit-metric`` values.

    The shell driver and registry own semantic validation. The sweep
    harness only normalizes repeatable/comma-separated operator input
    into a deterministic, de-duplicated tuple for forwarding and for
    the run-directory digest.
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        for token in str(raw).split(","):
            metric = token.strip()
            if not metric:
                raise ValueError("--unit-metric entries must be non-empty")
            if metric in seen:
                continue
            out.append(metric)
            seen.add(metric)
    return tuple(out)


def _validate_out_root(path: Path) -> Path:
    """Reject any --out-root outside the declared evidence sub-prefix.

    Two acceptable shapes:
      1. The path's resolved form ends with ``data/local_labels/sweeps``
         (an absolute or relative path under the evidence namespace).
      2. The path's resolved form contains
         ``data/local_labels/sweeps`` as an ancestor segment.
    """
    resolved = path.resolve()
    parts = resolved.parts
    # Look for the three-part subsequence ('data', 'local_labels',
    # 'sweeps') anywhere in the resolved path.
    for i in range(len(parts) - 2):
        if (parts[i], parts[i + 1], parts[i + 2]) == (
            "data", "local_labels", "sweeps",
        ):
            return resolved
    raise ValueError(
        f"--out-root must be under the declared evidence prefix "
        f"{_QUARANTINE_PREFIX}/; got {path!s} "
        f"(resolved to {resolved!s})"
    )


def _build_matrix(args: SweepArgs) -> tuple[tuple[CellId, MatrixConfig], ...]:
    """Enumerate the (cell, config) tuples in deterministic sorted order."""
    out: list[tuple[CellId, MatrixConfig]] = []
    for cell in args.candidates:
        for warmup in args.warmup_counts:
            for cache_control in args.cache_controls:
                cfg = MatrixConfig(
                    warmup_count=warmup,
                    cache_control=cache_control,
                    repeats=args.repeats,
                )
                out.append((cell, cfg))
    # Already in deterministic order because the input axes are sorted
    # tuples; tuple-sort the result for paranoia.
    return tuple(sorted(
        out,
        key=lambda item: (
            *item[0].sort_key,
            item[1].warmup_count, item[1].cache_control,
        ),
    ))


def _matrix_digest(
    matrix: Sequence[tuple[CellId, MatrixConfig]],
    seed: int,
    unit_metrics: Sequence[str] = (),
) -> str:
    """Return a deterministic 12-hex-digit digest over the matrix and
    seed. Used as the run-directory suffix so two runs with the same
    matrix and seed share the directory layout."""
    h = hashlib.sha256()
    h.update(str(seed).encode("utf-8"))
    h.update(b"\x00")
    for metric in unit_metrics:
        h.update(b"unit_metric|")
        h.update(metric.encode("utf-8"))
        h.update(b"\x00")
    for cell, cfg in matrix:
        h.update(cell.cell_id.encode("utf-8"))
        h.update(b"|")
        h.update(f"warmup{cfg.warmup_count}".encode("utf-8"))
        h.update(b"|")
        h.update(f"cache_{cfg.cache_control}".encode("utf-8"))
        h.update(b"|")
        h.update(f"repeats{cfg.repeats}".encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:12]


def _load_simbench_shape_axes(
    slug: str, shape_point: str
) -> dict[str, int] | None:
    """Load the axes for a SimBench (slug, shape_point) pair from
    ``workloads/simbench_bw/shapes/<slug>.toml``.

    Returns the axes dict (an ordered mapping of axis name to integer
    value) when the slug has a SimBench shape TOML AND a shape entry
    matching ``shape_point`` exists in any of the
    ``timing_shapes`` / ``trace_shapes`` / ``correctness_shapes``
    arrays. Returns ``None`` when no SimBench TOML exists for the slug
    (the caller falls back to forwarding only ``--shape-point``,
    preserving the prior behavior for non-SimBench slugs such as the
    portable_lcd workloads that handle ``--shape-point`` as a label-
    only convention).

    Raises ``ValueError`` when the TOML exists but no entry matches
    ``shape_point`` — that combination indicates drift between the
    candidate roster (or the operator's ``--candidates`` argument)
    and the per-slug shape file, and silently falling back to
    label-only forwarding would mis-attribute the resulting NCU
    ground truth to the requested shape while actually profiling
    the binary's compiled defaults.
    """
    toml_path = _SIMBENCH_SHAPES_DIR / f"{slug}.toml"
    if not toml_path.is_file():
        return None
    with toml_path.open("rb") as f:
        data = tomllib.load(f)
    for section in ("timing_shapes", "trace_shapes", "correctness_shapes"):
        for entry in data.get(section, []):
            if entry.get("name") == shape_point:
                axes = entry.get("axes", {})
                return dict(axes)
    raise ValueError(
        f"SimBench slug {slug!r} TOML at {toml_path} has no shape "
        f"entry named {shape_point!r}; the candidate roster references "
        f"a shape that does not exist in the per-slug TOML, indicating "
        f"drift between the sweep candidates and "
        f"workloads/simbench_bw/shapes/{slug}.toml"
    )


def _axes_to_workload_cli_args(axes: dict[str, int]) -> list[str]:
    """Convert a TOML axes mapping to the SimBench `main.cu` CLI
    convention: each key with snake_case is rewritten to kebab-case
    and paired with its integer value as ``--<flag-name> <value>``.

    The convention is verified across all SimBench workload runners
    (e.g., ``--working-set-bytes``, ``--allocation-bytes``,
    ``--carveout-percent``, ``--prefetch-distance``,
    ``--chain-length``, ``--accesses``, ``--warps``,
    ``--stride-classes``). Each runner's `else { unknown arg }`
    branch exits non-zero on unrecognized flags, so a future
    typo here would surface as a visible runtime error rather
    than a silent fall-through.
    """
    args: list[str] = []
    for key, value in axes.items():
        flag = "--" + key.replace("_", "-")
        args.extend([flag, str(value)])
    return args


def _label_driver_argv(
    *,
    driver: Path,
    cell: CellId,
    cfg: MatrixConfig,
    repeat_index: int,
    launch_index: int,
    binary: Path,
    ncu: str,
    out_dir: Path,
    measurement_manifest: Path | None = None,
    unit_metrics: Sequence[str] = (),
) -> list[str]:
    """Compose the label_driver.sh argv for one (cell, config, repeat)
    invocation. The argv is deterministic for a given input tuple.

    Two shape-point conventions are supported:

    - ``correctness_shape`` and ``timing_shape`` are special-case
      tokens the portable_lcd workloads accept as bare flags
      (``--correctness_shape``, ``--timing_shape``). The driver's
      shape-binding guard requires the workload args to match the
      shape_point token.
    - Any other shape token is forwarded as a generic
      ``--shape-point <token>`` flag (e.g.,
      ``--shape-point trace_00`` for simbench_bw shapes). For
      SimBench slugs that have a per-slug shape TOML at
      ``workloads/simbench_bw/shapes/<slug>.toml``, the harness
      ALSO forwards each axis as a ``--<kebab-axis> <value>``
      flag pair so the runner profiles the requested shape's
      parameters instead of the binary's compiled defaults
      (the SimBench `main.cu` runners treat ``--shape-point``
      as a label-only field and rely on the explicit axis flags
      to override their compiled defaults).

    When ``measurement_manifest`` is supplied, the path is forwarded
    as ``--measurement-manifest <path>`` so the driver consumes the
    operator-supplied manifest as the source-of-truth for
    ``kernel_signature_sha256``.
    """
    workload_args: list[str]
    if cell.shape_point in ("correctness_shape", "timing_shape"):
        workload_args = [f"--{cell.shape_point}"]
    else:
        workload_args = ["--shape-point", cell.shape_point]
        axes = _load_simbench_shape_axes(cell.slug, cell.shape_point)
        if axes:
            workload_args += _axes_to_workload_cli_args(axes)
    # NCU --launch-skip offset for warmed activation runs.
    #
    # label_driver.sh forwards --launch verbatim to NCU as
    # --launch-skip <LAUNCH_INDEX> at line 685. For activation
    # shapes (correctness_shape / timing_shape), the driver also
    # appends --warmup <count> to the portable_lcd workload at
    # line 395-410, which makes the binary do <count> extra
    # warmup launches BEFORE the measured launches. Without
    # offsetting --launch here, NCU profiles the first warmup
    # launch instead of the first measured launch, and the
    # resulting label is misattributed to the warmed config.
    #
    # SimBench shapes (e.g., trace_00) do NOT accept --warmup —
    # the driver's warmup-forwarding policy only appends it for
    # activation shapes; SimBench runners reject unknown args.
    # For SimBench shapes, warmup is metadata-only (no extra
    # launches) and the launch index MUST NOT be offset.
    is_activation_shape = cell.shape_point in (
        "correctness_shape", "timing_shape",
    )
    effective_launch_index = launch_index
    if is_activation_shape and cfg.warmup_count > 0:
        effective_launch_index = launch_index + cfg.warmup_count
    argv = [
        "bash",
        str(driver),
        "--device", cell.arch,
        "--slug", cell.slug,
        "--shape-point", cell.shape_point,
        "--repeat", str(repeat_index),
        "--launch", str(effective_launch_index),
        "--binary", str(binary),
        "--warmup", str(cfg.warmup_count),
        "--cache-control", cfg.cache_control,
        "--ncu", ncu,
        "--out-dir", str(out_dir),
    ]
    if measurement_manifest is not None:
        argv += ["--measurement-manifest", str(measurement_manifest)]
    # Forward the build-variant identity so the driver validates the manifest's
    # register_cap/variant_id against this expected canonical token (and fails
    # closed before NCU on a mismatch). Base cells (variant_id is None) get no
    # flag, so their argv is unchanged.
    if cell.variant_id is not None:
        argv += ["--expected-variant", cell.variant_id]
    for metric in unit_metrics:
        argv += ["--unit-metric", str(metric)]
    argv += ["--", *workload_args]
    return argv


def _aggregator_argv(
    *,
    aggregator: Path,
    config_subdir: Path,
    out_dir: Path,
    repeats_required: int,
) -> list[str]:
    """Compose the repeatability_aggregator.py argv for one cell-config.

    The aggregator's ``--root`` scans recursively for sidecars named
    ``label_<repeat>_<launch>.json``; ``--out`` is where the three
    indices (``label_repeatability.json`` / ``label_admitted.json`` /
    ``label_excluded.json``) are written. ``repeats_required`` MUST
    match the sweep's configured repeats axis so a 3-repeat sweep is
    not erroneously flagged as missing repeats and a 10-repeat sweep
    is not admitted after only five successful sidecars.
    """
    return [
        sys.executable,
        str(aggregator),
        "--root", str(config_subdir),
        "--out", str(out_dir),
        "--stddev-threshold", str(_STDDEV_ADMISSION_THRESHOLD),
        "--repeats-required", str(int(repeats_required)),
    ]


def _run_dry(args: SweepArgs, matrix: Sequence[tuple[CellId, MatrixConfig]]) -> int:
    """Print the planned per-config invocations to stdout. No
    label_driver.sh or NCU invocation occurs."""
    sys.stdout.write(
        f"# sweep_repeatability_matrix.py dry-run: "
        f"{len(matrix)} (cell, config) tuples × {args.repeats} repeats\n"
    )
    sys.stdout.write(f"# seed={args.seed}\n")
    if args.unit_metrics:
        sys.stdout.write(
            "# unit_metrics=" + ",".join(args.unit_metrics) + "\n"
        )
    for cell, cfg in matrix:
        binary = args.binaries.get(cell)
        if binary is None:
            sys.stdout.write(
                f"# missing --binaries entry for {cell.cell_id}; skipping\n"
            )
            continue
        manifest = args.measurement_manifests.get(cell)
        for repeat_index in range(args.repeats):
            argv = _label_driver_argv(
                driver=args.driver,
                cell=cell,
                cfg=cfg,
                repeat_index=repeat_index,
                launch_index=0,
                binary=binary,
                ncu=args.ncu,
                out_dir=Path("<out-dir-placeholder>"),
                measurement_manifest=manifest,
                unit_metrics=args.unit_metrics,
            )
            # Render --warmup <N> and --cache-control <mode> as adjacent
            # tokens so the test grep can assert the per-axis token shape
            # without parsing the full argv.
            sys.stdout.write(" ".join(argv) + "\n")
    return 0


def _run_live(args: SweepArgs, matrix: Sequence[tuple[CellId, MatrixConfig]]) -> int:
    """Execute the matrix end-to-end: invoke label_driver.sh per (cell,
    config, repeat); aggregate per-cell stddev; emit verdict-table and
    terminal verdict."""
    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = _matrix_digest(matrix, args.seed, args.unit_metrics)
    run_dir = args.out_root / f"sweep_{timestamp}_{digest}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    # Per-config admission tracking. Each unique (warmup_count,
    # cache_control, repeats) tuple maps to the SET of cell_ids
    # observed under it AND the SET of cells that admitted. A
    # config is the canonical "admitting_config" only when its
    # admitted set equals the set of all distinct candidate cells
    # (i.e. EVERY cell admitted under that config). Pre-fix,
    # setting admitting_config as soon as any single (cell, cfg)
    # admitted produced misleading verdicts: a config that
    # admitted exactly ONE stable cell while the other cells
    # failed would still be reported as `admitting_config_found:
    # true`, even though no shared (warmup, cache_control,
    # repeats) tuple satisfied all candidates.
    candidate_cells: set[str] = set()
    per_config_seen: dict[tuple[int, str, int], set[str]] = {}
    per_config_admitted: dict[tuple[int, str, int], set[str]] = {}

    for cell, cfg in matrix:
        candidate_cells.add(cell)
        cfg_key = (cfg.warmup_count, cfg.cache_control, cfg.repeats)
        per_config_seen.setdefault(cfg_key, set()).add(cell)
        per_config_admitted.setdefault(cfg_key, set())
        binary = args.binaries.get(cell)
        if binary is None:
            row = _row_for_missing_binary(cell, cfg)
            rows.append(row)
            continue
        cell_dir, summary_path, admit_outcome = _execute_cell_config(
            args=args, cell=cell, cfg=cfg, binary=binary, run_dir=run_dir,
        )
        rows.append(_row_from_admit_outcome(cell, cfg, admit_outcome))
        if admit_outcome.admitted:
            per_config_admitted[cfg_key].add(cell)

    admitting_config: dict[str, object] | None = None
    for cfg_key in sorted(per_config_admitted.keys()):
        # A config admits only when EVERY candidate cell was
        # exercised under it AND every exercised cell admitted.
        seen_set = per_config_seen.get(cfg_key, set())
        admitted_set = per_config_admitted[cfg_key]
        if seen_set == candidate_cells and admitted_set == candidate_cells:
            warmup_count, cache_control, repeats = cfg_key
            admitting_config = {
                "warmup_count": warmup_count,
                "cache_control": cache_control,
                "repeats": repeats,
            }
            break

    rows = sorted(rows, key=lambda r: (
        r["cell_id"], int(r["warmup_count"]), r["cache_control"],
    ))
    _write_verdict_table_csv(run_dir / "verdict-table.csv", rows)
    _write_verdict_table_json(run_dir / "verdict-table.json", rows)

    if admitting_config is not None:
        verdict = {
            "verdict": "admitting_config_found",
            "config": admitting_config,
            "evidence": (
                f"per-cell stddev <= {_STDDEV_ADMISSION_THRESHOLD} "
                f"under config "
                f"warmup={admitting_config['warmup_count']} "
                f"cache_control={admitting_config['cache_control']} "
                f"repeats={admitting_config['repeats']}"
            ),
        }
    else:
        verdict = {
            "verdict": "no_admitting_config_found",
            "config": None,
            "evidence": (
                f"no (warmup, cache_control, repeats) config produced "
                f"per-cell stddev <= {_STDDEV_ADMISSION_THRESHOLD} "
                f"across the candidate cell set"
            ),
        }
    _write_verdict_json(run_dir / "verdict.json", verdict)
    sys.stdout.write(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    return 0


@dataclass(frozen=True)
class _AdmitOutcome:
    n_sidecars: int
    stddev_l2_sector: float
    stddev_l1tex_sector: float
    mean_l2_sector: float
    admitted: bool
    detail: str


def _write_failed_invocation_sidecar(
    *,
    config_subdir: Path,
    cell: "CellId",
    cfg: "MatrixConfig",
    repeat_index: int,
    launch_index: int,
    returncode: int,
    stderr_excerpt: str,
) -> Path:
    """Emit an explicit failed-invocation record for the planned
    ``(cell, warmup, cache_control, repeat, launch)`` tuple.

    The file name uses the ``failed_invocation_<r>_<l>.json``
    prefix so the aggregator's ``label_*_*.json`` collector does
    NOT pick it up as a measurement sidecar; the file's role is
    audit-trail, not measurement evidence. The committed presence
    of this file is the machine-readable record the
    SWEEP_RUNBOOK requires for every planned NCU invocation so
    the operator's `.ncu-rep` count (gitignored) can be
    independently reconciled against the committed sidecar
    count.
    """
    payload = {
        "schema_version": 1,
        "failed_invocation": True,
        "arch": cell.arch,
        "slug": cell.slug,
        "shape_point": cell.shape_point,
        "warmup_count": int(cfg.warmup_count),
        "cache_control": str(cfg.cache_control),
        "repeat_index": int(repeat_index),
        "launch_index": int(launch_index),
        "label_driver_returncode": int(returncode),
        "label_driver_stderr_excerpt": stderr_excerpt[:240],
    }
    failed_path = (
        config_subdir
        / f"failed_invocation_{repeat_index}_{launch_index}.json"
    )
    with failed_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return failed_path


def _execute_cell_config(
    *, args: SweepArgs, cell: CellId, cfg: MatrixConfig, binary: Path,
    run_dir: Path,
) -> tuple[Path, Path, _AdmitOutcome]:
    """Invoke label_driver.sh ``cfg.repeats`` times for one
    (cell, config) tuple, then aggregate the per-launch sidecars.

    When a per-repeat ``label_driver.sh`` invocation exits
    non-zero (the documented NCU OOR / parser-error / missing-
    metric path), the harness emits an explicit
    ``failed_invocation_<r>_<l>.json`` audit-trail record AND
    continues executing the remaining planned repeats. This
    preserves the SWEEP_RUNBOOK's "machine-readable record for
    every planned invocation" invariant: the committed evidence
    tree accounts for every planned (cell, cfg, repeat) tuple
    independently of the gitignored ``.ncu-rep`` files.
    Aggregation runs over the successful sidecars only; the
    aggregator's ``n_repeats`` reflects the count of successful
    NCU launches, not the planned count.
    """
    config_subdir = (
        run_dir
        / cell.output_dir_name
        / f"cfg=warmup{cfg.warmup_count}_cache_{cfg.cache_control}"
    )
    config_subdir.mkdir(parents=True, exist_ok=True)

    manifest = args.measurement_manifests.get(cell)
    failed_repeats: list[int] = []
    for repeat_index in range(cfg.repeats):
        argv = _label_driver_argv(
            driver=args.driver,
            cell=cell,
            cfg=cfg,
            repeat_index=repeat_index,
            launch_index=0,
            binary=binary,
            ncu=args.ncu,
            out_dir=config_subdir,
            measurement_manifest=manifest,
            unit_metrics=args.unit_metrics,
        )
        proc = subprocess.run(
            argv, capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            _write_failed_invocation_sidecar(
                config_subdir=config_subdir,
                cell=cell,
                cfg=cfg,
                repeat_index=repeat_index,
                launch_index=0,
                returncode=proc.returncode,
                stderr_excerpt=proc.stderr.strip(),
            )
            failed_repeats.append(repeat_index)
            continue

    aggregator_out_dir = config_subdir / "_aggregator_out"
    aggregator_out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = aggregator_out_dir / "label_repeatability.json"
    aggregator_argv = _aggregator_argv(
        aggregator=args.aggregator,
        config_subdir=config_subdir,
        out_dir=aggregator_out_dir,
        repeats_required=cfg.repeats,
    )
    proc = subprocess.run(
        aggregator_argv, capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        return (config_subdir, summary_path, _AdmitOutcome(
            n_sidecars=0,
            stddev_l2_sector=0.0,
            stddev_l1tex_sector=0.0,
            mean_l2_sector=0.0,
            admitted=False,
            detail=(
                f"repeatability_aggregator returncode={proc.returncode} "
                f"stderr={proc.stderr.strip()[:240]!r}"
            ),
        ))

    if not summary_path.is_file():
        return (config_subdir, summary_path, _AdmitOutcome(
            n_sidecars=0,
            stddev_l2_sector=0.0,
            stddev_l1tex_sector=0.0,
            mean_l2_sector=0.0,
            admitted=False,
            detail="aggregator summary not written",
        ))
    with summary_path.open(encoding="utf-8") as fh:
        summary = json.load(fh)
    outcome = _admit_outcome_from_summary(summary, cell)
    return (config_subdir, summary_path, outcome)


def _admit_outcome_from_summary(summary: dict, cell: CellId) -> _AdmitOutcome:
    """Translate the aggregator's ``label_repeatability.json`` to an
    ``_AdmitOutcome``. The aggregator emits a ``cells`` list whose
    entries each carry an ``admission`` dict plus per-cell stats
    (``stddev_l2_sector``, ``stddev_l1tex_sector``, ``mean_l2_sector``,
    ``n_repeats``)."""
    cells = summary.get("cells", [])
    record = None
    for entry in cells:
        if (
            entry.get("device") == cell.arch
            and entry.get("slug") == cell.slug
            and entry.get("shape_point") == cell.shape_point
        ):
            record = entry
            break
    if record is None:
        return _AdmitOutcome(
            n_sidecars=0,
            stddev_l2_sector=0.0,
            stddev_l1tex_sector=0.0,
            mean_l2_sector=0.0,
            admitted=False,
            detail="aggregator did not emit a cell record for this cell",
        )
    admission = record.get("admission", {}) or {}
    admitted = bool(admission.get("admitted"))

    def _coerce_float(value: object, fallback: float = 0.0) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        return fallback

    return _AdmitOutcome(
        n_sidecars=int(record.get("n_repeats", 0)),
        stddev_l2_sector=_coerce_float(record.get("stddev_l2_sector")),
        stddev_l1tex_sector=_coerce_float(record.get("stddev_l1tex_sector")),
        mean_l2_sector=_coerce_float(record.get("mean_l2_sector")),
        admitted=admitted,
        detail=(
            admission.get("detail", "") if not admitted else "admitted"
        ),
    )


def _row_for_missing_binary(cell: CellId, cfg: MatrixConfig) -> dict[str, str]:
    return {
        "cell_id": cell.cell_id,
        "arch": cell.arch,
        "slug": cell.slug,
        "shape_point": cell.shape_point,
        "warmup_count": str(cfg.warmup_count),
        "cache_control": cfg.cache_control,
        "repeats": str(cfg.repeats),
        "n_sidecars": "0",
        "stddev_l2_sector": "0.0",
        "stddev_l1tex_sector": "0.0",
        "mean_l2_sector": "0.0",
        "admitted": "false",
        "detail": "no --binaries entry for this cell",
        "register_cap": "" if cell.register_cap is None else str(cell.register_cap),
        "variant_id": cell.variant_id or "",
    }


def _row_from_admit_outcome(
    cell: CellId, cfg: MatrixConfig, outcome: _AdmitOutcome,
) -> dict[str, str]:
    return {
        "cell_id": cell.cell_id,
        "arch": cell.arch,
        "slug": cell.slug,
        "shape_point": cell.shape_point,
        "warmup_count": str(cfg.warmup_count),
        "cache_control": cfg.cache_control,
        "repeats": str(cfg.repeats),
        "n_sidecars": str(outcome.n_sidecars),
        "stddev_l2_sector": f"{outcome.stddev_l2_sector:.6f}",
        "stddev_l1tex_sector": f"{outcome.stddev_l1tex_sector:.6f}",
        "mean_l2_sector": f"{outcome.mean_l2_sector:.6f}",
        "admitted": "true" if outcome.admitted else "false",
        "detail": outcome.detail,
        "register_cap": "" if cell.register_cap is None else str(cell.register_cap),
        "variant_id": cell.variant_id or "",
    }


def _write_verdict_table_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_VERDICT_TABLE_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in _VERDICT_TABLE_COLUMNS})


def _write_verdict_table_json(path: Path, rows: Sequence[dict[str, str]]) -> None:
    serializable = [
        {col: row.get(col, "") for col in _VERDICT_TABLE_COLUMNS}
        for row in rows
    ]
    with path.open("w", encoding="utf-8") as fh:
        json.dump(serializable, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _write_verdict_json(path: Path, verdict: dict) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _run_provenance_audit_preflight(
    *,
    candidates: Sequence["CellId"],
    measurement_manifests: dict["CellId", Path],
    predictor_trace_jsons: dict["CellId", Path],
) -> str | None:
    """Run the provenance audit per candidate cell. Returns None
    if every cell admits with a PASS verdict; returns a single
    multi-line diagnostic string otherwise (the caller writes it
    to stderr and returns exit 2 BEFORE any NCU shim is spawned).

    A cell missing from ``predictor_trace_jsons`` is itself a
    fail-closed condition: if the operator passes
    ``--predictor-trace-jsons`` at all, every candidate must be
    covered (mirroring the existing ``--measurement-manifests``
    coverage check).
    """

    missing = [
        cell.cell_id for cell in candidates
        if cell not in predictor_trace_jsons
    ]
    if missing:
        return (
            "sweep_repeatability_matrix: "
            "--predictor-trace-jsons does not cover every "
            f"candidate cell; missing: {missing!r}"
        )
    failure_lines: list[str] = []
    for cell in candidates:
        manifest_path = measurement_manifests[cell]
        predictor_json_path = predictor_trace_jsons[cell]
        # Mirror the label_driver.sh:217-224 case-statement: the
        # documented sm_120 candidate token (matching the label
        # schema and SWEEP_RUNBOOK.md) is the legacy local label
        # for the RTX 5060 laptop, while arch_capabilities.json
        # admits the disambiguated `sm_120-laptop` token. Every
        # other arch passes through unchanged so server arches
        # stay forbidden by the audit's admitted-set check.
        cap_arch = "sm_120-laptop" if cell.arch == "sm_120" else cell.arch
        try:
            # live_sweep=True narrows the audit's source_commit
            # accept set to current HEAD only. This preflight is
            # only invoked in --live mode (see the caller's
            # `if not raw.dry_run` block), where the subsequent
            # label_driver.sh records source_commit from
            # `git rev-parse HEAD` only — admitting a HEAD^
            # sidecar here would mint labels that cannot join
            # the predictor bundle in comparison_runner. Codex
            # Round-22 review P1 fix.
            result = _provenance_audit.run_audit(
                predictor_trace_json=predictor_json_path,
                manifest=manifest_path,
                arch=cap_arch,
                live_sweep=True,
            )
        except _provenance_audit._IoFailure as exc:
            failure_lines.append(
                f"sweep_repeatability_matrix: provenance_audit "
                f"io failure for cell {cell.cell_id!r}: "
                f"{exc.message}"
            )
            continue
        if result.typed_reason is not None:
            failure_lines.append(
                f"sweep_repeatability_matrix: provenance_audit "
                f"failed closed for cell {cell.cell_id!r}: "
                f"{result.typed_reason}: {result.detail}"
            )
    if failure_lines:
        return "\n".join(failure_lines)
    return None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true",
                      help="Print planned per-config invocations; do "
                           "not invoke label_driver.sh or NCU.")
    mode.add_argument("--live", action="store_true",
                      help="Execute the matrix; aggregate sidecars; "
                           "emit verdict-table and terminal verdict.")
    parser.add_argument(
        "--candidates", required=True,
        help="Comma-separated candidate cell tokens "
             "(arch/slug/shape_point,...). A token may carry an optional "
             "register-cap build-variant suffix "
             "(arch/slug/shape_point/register_cap_<cap>); two caps of the "
             "same shape are distinct cells.",
    )
    parser.add_argument(
        "--binaries", default="",
        help="Comma-separated cell-id-to-binary map "
             "(arch/slug/shape_point[/register_cap_<cap>]=path,...). A "
             "build-variant cell must map to its matching per-cap binary.",
    )
    parser.add_argument(
        "--measurement-manifests", default="",
        help=(
            "Comma-separated cell-id-to-measurement-manifest map "
            "(arch/slug/shape_point=path,...). REQUIRED in --live "
            "mode: every candidate must name an operator-supplied "
            "measurement_manifest.json carrying "
            "kernel_signature_sha256 (the operator-pinned "
            "source-of-truth for the kernel-signature value). The "
            "harness forwards each entry as label_driver.sh "
            "--measurement-manifest <path>; the driver fails closed "
            "before NCU on any manifest unavailability or mismatch."
        ),
    )
    parser.add_argument(
        "--predictor-trace-jsons", default="",
        help=(
            "Comma-separated cell-id-to-predictor-trace-JSON map "
            "(arch/slug/shape_point=path,...). REQUIRED in --live "
            "mode (mirrors --measurement-manifests): every "
            "candidate must name a predictor-side per-shape trace "
            "JSON sidecar. The harness runs "
            "tools/local_labels/provenance_audit.py per cell "
            "before any NCU shim is spawned; any non-PASS audit "
            "verdict aborts the sweep with exit 2. Dry-run mode "
            "ignores this flag."
        ),
    )
    parser.add_argument(
        "--warmup-counts", default="0,1,5",
        help="Comma-separated warmup counts (default: 0,1,5).",
    )
    parser.add_argument(
        "--cache-controls", default="none",
        help=(
            "Comma-separated cache-control modes; allowed values: "
            f"{','.join(_ALLOWED_CACHE_CONTROLS)} (default: none)."
        ),
    )
    parser.add_argument(
        "--repeats", type=int, default=5,
        help="Repeats per (cell, config) tuple (default: 5).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Deterministic ordering seed (default: 0).",
    )
    parser.add_argument(
        "--ncu", default="ncu",
        help="Path to the Nsight Compute CLI binary (default: 'ncu').",
    )
    parser.add_argument(
        "--unit-metric",
        action="append",
        default=[],
        help=(
            "Canonical non-percentage NCU metric id to collect and "
            "forward into label_driver.sh's unit_tagged_metrics path. "
            "Repeatable; comma-separated values are also accepted."
        ),
    )
    parser.add_argument(
        "--driver", type=Path, default=_DEFAULT_DRIVER,
        help=(
            f"Path to label_driver.sh (default: {_DEFAULT_DRIVER!s})."
        ),
    )
    parser.add_argument(
        "--aggregator", type=Path, default=_DEFAULT_AGGREGATOR,
        help=(
            f"Path to repeatability_aggregator.py "
            f"(default: {_DEFAULT_AGGREGATOR!s})."
        ),
    )
    parser.add_argument(
        "--out-root", type=Path, default=_DEFAULT_OUTPUT_ROOT,
        help=(
            f"Output root; must be under {_QUARANTINE_PREFIX}/ "
            f"(default: {_DEFAULT_OUTPUT_ROOT!s})."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    raw = parser.parse_args(argv)
    try:
        candidates = _parse_candidates(raw.candidates)
        binaries = _parse_binaries(raw.binaries) if raw.binaries else {}
        manifests_raw = getattr(raw, "measurement_manifests", "")
        measurement_manifests = (
            _parse_measurement_manifests(manifests_raw)
            if manifests_raw else {}
        )
        predictor_jsons_raw = getattr(raw, "predictor_trace_jsons", "")
        predictor_trace_jsons = (
            _parse_predictor_trace_jsons(predictor_jsons_raw)
            if predictor_jsons_raw else {}
        )
        warmup_counts = _parse_int_list(raw.warmup_counts, "--warmup-counts")
        cache_controls = _parse_cache_controls(raw.cache_controls)
        unit_metrics = _parse_unit_metrics(raw.unit_metric)
        out_root = _validate_out_root(raw.out_root)
    except ValueError as exc:
        sys.stderr.write(f"sweep_repeatability_matrix: {exc}\n")
        return 2
    if raw.repeats <= 0:
        sys.stderr.write(
            "sweep_repeatability_matrix: --repeats must be a positive integer\n"
        )
        return 2

    # Operator-pinned source-of-truth: live mode REQUIRES a
    # measurement-manifest for every candidate cell. The driver-level
    # fallback exists only to keep unit tests + offline rehearsal
    # hermetic; once the operator commits to an actual NCU sweep, the
    # manifest map is the binding kernel_signature_sha256 source.
    if raw.live:
        if not measurement_manifests:
            sys.stderr.write(
                "sweep_repeatability_matrix: --measurement-manifests is "
                "REQUIRED in --live mode (operator-pinned "
                "kernel_signature_sha256 source-of-truth)\n"
            )
            return 2
        missing = [
            cell.cell_id
            for cell in candidates
            if cell not in measurement_manifests
        ]
        if missing:
            sys.stderr.write(
                "sweep_repeatability_matrix: --measurement-manifests does "
                "not cover every candidate cell; missing: "
                f"{missing!r}\n"
            )
            return 2
        # The pre-live provenance audit is mandatory: the live NCU
        # sweep cannot launch unless every candidate cell admits
        # the seven-field join-key audit. Mirroring the
        # --measurement-manifests contract above, --predictor-trace-jsons
        # must be supplied AND cover every candidate.
        if not predictor_trace_jsons:
            sys.stderr.write(
                "sweep_repeatability_matrix: --predictor-trace-jsons is "
                "REQUIRED in --live mode (the pre-live provenance audit "
                "must run before any NCU shim is spawned)\n"
            )
            return 2
        audit_failure = _run_provenance_audit_preflight(
            candidates=candidates,
            measurement_manifests=measurement_manifests,
            predictor_trace_jsons=predictor_trace_jsons,
        )
        if audit_failure is not None:
            sys.stderr.write(audit_failure + "\n")
            return 2

    args = SweepArgs(
        mode="live" if raw.live else "dry-run",
        candidates=candidates,
        binaries=binaries,
        measurement_manifests=measurement_manifests,
        unit_metrics=unit_metrics,
        warmup_counts=warmup_counts,
        cache_controls=cache_controls,
        repeats=raw.repeats,
        seed=raw.seed,
        ncu=raw.ncu,
        driver=raw.driver,
        aggregator=raw.aggregator,
        out_root=out_root,
    )
    matrix = _build_matrix(args)
    if args.mode == "dry-run":
        return _run_dry(args, matrix)
    return _run_live(args, matrix)


if __name__ == "__main__":
    raise SystemExit(main())
