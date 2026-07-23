#!/usr/bin/env python3
"""Build and seal the 31 sm_86-gated SimBench binaries for task7.

The witness is durable evidence, not a cache of current filesystem presence.
Every selected roster row is rebuilt with an explicit toolchain, smoke-run at
its first trace shape, checked for an sm_86 cubin and unresolved libraries, and
bound to its source/shape/binary hashes. Any missing row or failed check makes
the whole witness non-admissible; callers must not infer success over zero rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jsonschema

REPO_ROOT = Path(__file__).resolve().parents[2]
ROSTER_PATH = REPO_ROOT / "workloads" / "simbench_bw" / "roster.toml"
SHAPES_DIR = REPO_ROOT / "workloads" / "simbench_bw" / "shapes"
COMMON_MK = REPO_ROOT / "workloads" / "simbench_bw" / "common.mk"
PROTOCOL_PINS = REPO_ROOT / "tools" / "local_labels" / "campaign_protocol_pins.json"
ARCH = "sm_86"
EXPECTED_ROWS = 31
WITNESS_ID = "sm86_simbench_build_witness_v1"
DEFAULT_REGISTRY_REL = Path("tools/local_labels/witness_compat_registry.json")
WITNESS_SCHEMA_REL = Path(
    "workloads/simbench_bw/schemas/simbench_build_witness.schema.json")
TOP_INPUTS = {
    "roster_sha256": "workloads/simbench_bw/roster.toml",
    "common_mk_sha256": "workloads/simbench_bw/common.mk",
    "protocol_pins_sha256": "tools/local_labels/campaign_protocol_pins.json",
}
INPUT_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cu", ".h", ".hpp", ".cuh"})
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_SAFE_REL_RE = re.compile(
    r"^(?!/)(?!.*\\)(?!.*//)(?!.*/$)(?!\.\.?(?:/|$))"
    r"(?!.*(?:/\.\.?)(?:/|$))[^\x00-\x1f\x7f]+$")
REGISTERED_TOKEN_FORMS = (
    "witness_registry_missing:{path}",
    "witness_registry_unreadable:{path}",
    "witness_registry_not_regular:{path}",
    "witness_registry_path_escape:{path}",
    "witness_registry_invalid:json_decode",
    "witness_registry_invalid:duplicate_key:{json_pointer}",
    "witness_registry_invalid:noncanonical_bytes",
    "witness_registry_invalid:root_type",
    "witness_registry_invalid:field_type:{json_pointer}",
    "witness_registry_invalid:missing_field:{json_pointer}",
    "witness_registry_invalid:unknown_field:{json_pointer}",
    "witness_registry_invalid:schema_version",
    "witness_registry_invalid:path_syntax:{json_pointer}",
    "witness_registry_invalid:hash_syntax:{json_pointer}",
    "witness_registry_invalid:commit_syntax:{json_pointer}",
    "witness_registry_invalid:witness_cardinality",
    "witness_registry_invalid:migration_cardinality",
    "witness_registry_invalid:witness_order",
    "witness_registry_invalid:migration_order",
    "witness_file_missing:{path}",
    "witness_file_unreadable:{path}",
    "witness_file_not_regular:{path}",
    "witness_file_path_escape:{path}",
    "witness_file_sha256_mismatch:{path}",
    "witness_json_invalid:json_decode",
    "witness_json_invalid:duplicate_key:{json_pointer}",
    "witness_structure_invalid:root_type",
    "witness_structure_invalid:schema:{json_pointer}",
    "witness_structure_invalid:path_syntax:{json_pointer}",
    "witness_structure_invalid:schema_version",
    "witness_structure_invalid:identity",
    "witness_structure_invalid:execution_flags",
    "witness_structure_invalid:row_count",
    "witness_structure_invalid:slug_uniqueness",
    "witness_structure_invalid:row_shape:{index}",
    "witness_structure_invalid:row_status:{slug}",
    "witness_structure_invalid:row_build:{slug}",
    "witness_structure_invalid:row_libraries:{slug}",
    "witness_structure_invalid:row_smoke:{slug}",
    "witness_structure_invalid:row_shape_file:{slug}",
    "witness_structure_invalid:row_binary_path:{slug}",
    "witness_structure_invalid:row_input_set_sha256:{slug}",
    "witness_structure_invalid:row_shape_file_sha256:{slug}",
    "witness_structure_invalid:pass",
    "row_seal_mismatch:{slug}",
    "witness_seal_mismatch",
    "witness_registry_mismatch:witness_seal",
    "witness_registry_mismatch:source_commit",
    "witness_registry_mismatch:migrated_path_not_in_witness:{path}",
    "witness_registry_mismatch:witness_sha256:{path}",
    "witness_commit_unresolvable:{commit}",
    "witness_commit_unresolvable:{commit}:shallow_clone",
    "row_inputs_at_commit:{slug}:{path}:missing_at_commit",
    "row_inputs_at_commit:{slug}:{path}:blob_mismatch",
    "witness_top_input_at_commit:{path}:missing_at_commit",
    "witness_top_input_at_commit:{path}:blob_mismatch",
    "witness_source_at_commit:roster_invalid",
    "witness_source_at_commit:slug_set_mismatch",
    "witness_source_at_commit:input_scan_failed",
    "row_source_at_commit:{slug}:shape_invalid",
    "row_source_at_commit:{slug}:level_mismatch",
    "row_source_at_commit:{slug}:shape_mismatch",
    "row_source_at_commit:{slug}:shape_binding_kind_mismatch",
    "row_source_at_commit:{slug}:app_args_mismatch",
    "row_source_at_commit:{slug}:binary_path_mismatch",
    "row_inputs_at_commit:{slug}:{path}:missing_recorded_input",
    "row_inputs_at_commit:{slug}:{path}:unexpected_recorded_input",
    "missing_current_source:{path}",
    "new_current_source:{path}",
    "unmigrated_source_drift:{path}",
    "migration_pin_mismatch:{path}",
    "migration_pin_stale:{path}",
    "current_source_unreadable:{path}",
    "current_source_not_regular:{path}",
    "current_source_path_escape:{path}",
    "current_source_scan_failed:{path}",
)
assert len(REGISTERED_TOKEN_FORMS) == len(set(REGISTERED_TOKEN_FORMS)) == 76


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return sha256_bytes(payload)


def run_command(
    command: list[str], *, cwd: Path = REPO_ROOT, timeout: int = 300,
) -> tuple[int, bytes]:
    try:
        proc = subprocess.run(
            command, cwd=str(cwd), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=timeout,
        )
        return proc.returncode, proc.stdout
    except FileNotFoundError as exc:
        return 127, str(exc).encode("utf-8", errors="replace")
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or b""
        return 124, output + b"\nTIMEOUT\n"


def command_identity(path: str) -> dict:
    resolved = Path(path).resolve()
    rc, output = run_command([str(resolved), "--version"], timeout=30)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved) if resolved.is_file() else None,
        "version_returncode": rc,
        "version_output": output.decode("utf-8", errors="replace")[:2000],
    }


def selected_rows(repo_root: Path = REPO_ROOT) -> list[dict]:
    rows = tomllib.loads(
        (repo_root / "workloads/simbench_bw/roster.toml").read_text()
    ).get("rows", [])
    selected = [row for row in rows if ARCH in (row.get("arch_gate") or [])]
    slugs = [str(row.get("slug") or "") for row in selected]
    if len(selected) != EXPECTED_ROWS:
        raise ValueError(
            f"expected {EXPECTED_ROWS} {ARCH}-gated rows, found {len(selected)}")
    if any(not slug for slug in slugs) or len(slugs) != len(set(slugs)):
        raise ValueError("selected SimBench rows contain an empty/duplicate slug")
    return selected


def _shape_and_args_docs(row: dict, shape_doc: dict) -> tuple[dict, list[str], str]:
    slug = row["slug"]
    if shape_doc.get("slug") != slug:
        raise ValueError(
            f"shape slug mismatch for {slug}: {shape_doc.get('slug')!r}")
    shapes = shape_doc.get("trace_shapes") or []
    if not isinstance(shapes, list) or not shapes:
        raise ValueError(f"{slug} has zero trace_shapes")
    shape = shapes[0]
    name = shape.get("name")
    axes = shape.get("axes")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{slug} first trace shape has no name")
    if not isinstance(axes, dict) or not axes:
        raise ValueError(f"{slug}/{name} has zero axes")

    args: list[str] = []
    for key, value in axes.items():
        if type(value) is not int:
            raise ValueError(f"{slug}/{name} axis {key} is not an integer")
        args.extend([f"--{key.replace('_', '-')}", str(value)])
    seed = row.get("seed_policy", {}).get("seed", 42)
    if type(seed) is not int:
        raise ValueError(f"{slug} seed is not an integer")
    args.extend(["--seed", str(seed)])
    if name == "timing_shape":
        # label_driver treats this literal as its portable legacy form. The sole
        # SimBench collision (vendor_sgemm_cublas) explicitly supports the alias.
        args.append("--timing_shape")
        binding = "legacy_timing_alias"
    elif name == "correctness_shape":
        args.append("--correctness_shape")
        binding = "legacy_correctness_alias"
    else:
        args.extend(["--shape-point", name])
        binding = "generic_shape_point"
    return shape, args, binding


def shape_and_args(
    row: dict, repo_root: Path = REPO_ROOT,
) -> tuple[dict, list[str], str, Path]:
    slug = row["slug"]
    shape_path = repo_root / "workloads/simbench_bw/shapes" / f"{slug}.toml"
    shape_doc = tomllib.loads(shape_path.read_text())
    shape, args, binding = _shape_and_args_docs(row, shape_doc)
    return shape, args, binding, shape_path


def input_hashes(
    kernel_dir: Path, shape_path: Path, repo_root: Path = REPO_ROOT,
) -> dict[str, str]:
    paths = [repo_root / "workloads/simbench_bw/common.mk", shape_path]
    for path in sorted(kernel_dir.iterdir()):
        if path.name == "Makefile" or path.suffix in {".c", ".cc", ".cpp", ".cu", ".h", ".hpp", ".cuh"}:
            paths.append(path)
    unique = sorted(set(paths))
    if len(unique) < 4:
        raise ValueError(f"too few build inputs under {kernel_dir}")
    return {str(path.relative_to(repo_root)): sha256_file(path) for path in unique}


def inspect_cubins(binary: Path, cuobjdump: str) -> tuple[int, list[str], str]:
    rc, output = run_command([cuobjdump, "--list-elf", str(binary)], timeout=60)
    text = output.decode("utf-8", errors="replace")
    cubins = []
    for line in text.splitlines():
        if ".cubin" in line and ":" in line:
            cubins.append(line.split(":", 1)[1].strip())
    return rc, cubins, sha256_bytes(output)


def inspect_libraries(binary: Path) -> tuple[int, list[dict], list[str], str]:
    rc, output = run_command(["ldd", str(binary)], timeout=60)
    text = output.decode("utf-8", errors="replace")
    resolved: list[dict] = []
    missing: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if "not found" in stripped:
            missing.append(stripped)
        elif "=>" in stripped:
            name, target = stripped.split("=>", 1)
            resolved.append({
                "name": name.strip(),
                "path": target.strip().split(" ", 1)[0],
            })
    return rc, resolved, missing, sha256_bytes(output)


def build_row(
    row: dict, *, nvcc: str, host_cxx: str, cuobjdump: str,
    execute_build: bool, execute_smoke: bool,
) -> dict:
    slug = row["slug"]
    level = str(row["level"]).lower()
    kernel_dir = REPO_ROOT / "workloads" / "simbench_bw" / level / slug
    binary = kernel_dir / f"{slug}_run"
    shape, app_args, binding, shape_path = shape_and_args(row)
    inputs = input_hashes(kernel_dir, shape_path)
    build_command = [
        # Force the recipe even when an ignored host-local binary already exists;
        # an up-to-date no-op is not evidence that this explicit toolchain built it.
        "make", "-B", "-C", str(kernel_dir.relative_to(REPO_ROOT)), "run-bin",
        f"NVCC={nvcc} -ccbin {host_cxx}",
    ]
    if execute_build:
        build_rc, transcript = run_command(build_command, timeout=600)
    else:
        build_rc, transcript = 125, b"BUILD_NOT_EXECUTED\n"

    problems: list[str] = []
    if build_rc != 0:
        problems.append("build_command_failed")
    if not (binary.is_file() and os.access(binary, os.X_OK)):
        problems.append("binary_missing_or_not_executable")

    binary_sha = sha256_file(binary) if binary.is_file() else None
    binary_bytes = binary.stat().st_size if binary.is_file() else None
    cubin_rc, cubins, cubin_transcript_sha = (
        inspect_cubins(binary, cuobjdump) if binary.is_file()
        else (127, [], sha256_bytes(b"binary missing")))
    sm86_cubin_present = any(".sm_86.cubin" in name for name in cubins)
    if cubin_rc != 0 or not sm86_cubin_present:
        problems.append("sm86_cubin_missing")

    ldd_rc, libraries, missing_libraries, ldd_sha = (
        inspect_libraries(binary) if binary.is_file()
        else (127, [], ["binary missing"], sha256_bytes(b"binary missing")))
    if ldd_rc != 0 or missing_libraries:
        problems.append("runtime_dependency_missing")

    smoke: dict[str, Any]
    if execute_smoke and binary.is_file():
        smoke_rc, smoke_output = run_command(
            [str(binary), *app_args], cwd=kernel_dir, timeout=120)
        smoke_text = smoke_output.decode("utf-8", errors="replace")
        try:
            smoke_doc = json.loads(smoke_text)
        except json.JSONDecodeError:
            smoke_doc = None
        smoke_contract_ok = bool(
            isinstance(smoke_doc, dict)
            and smoke_doc.get("slug") == slug
            and smoke_doc.get("shape_point") == shape["name"]
            and "output" in smoke_doc
        )
        if smoke_rc != 0:
            problems.append("smoke_command_failed")
        elif not smoke_contract_ok:
            problems.append("smoke_output_invalid")
        smoke = {
            "executed": True,
            "returncode": smoke_rc,
            "output_sha256": sha256_bytes(smoke_output),
            "output_bytes": len(smoke_output),
            "contract_ok": smoke_contract_ok,
            "reported_slug": smoke_doc.get("slug") if isinstance(smoke_doc, dict) else None,
            "reported_shape_point": (
                smoke_doc.get("shape_point") if isinstance(smoke_doc, dict) else None),
        }
    else:
        problems.append("smoke_not_executed")
        smoke = {"executed": False, "returncode": None, "contract_ok": False}

    row_doc = {
        "slug": slug,
        "level": level,
        "status": "runnable" if not problems else "blocked",
        "problems": problems,
        "build_command": build_command,
        "build_returncode": build_rc,
        "build_transcript_sha256": sha256_bytes(transcript),
        "build_transcript_bytes": len(transcript),
        "build_inputs": inputs,
        "build_input_set_sha256": canonical_sha256(inputs),
        "shape_file": str(shape_path.relative_to(REPO_ROOT)),
        "shape_file_sha256": sha256_file(shape_path),
        "shape_point": shape["name"],
        "shape": shape,
        "shape_binding_kind": binding,
        "app_args": app_args,
        "binary": str(binary.relative_to(REPO_ROOT)),
        "binary_sha256": binary_sha,
        "binary_bytes": binary_bytes,
        "binary_executable": bool(binary.is_file() and os.access(binary, os.X_OK)),
        "cuobjdump_returncode": cubin_rc,
        "cuobjdump_transcript_sha256": cubin_transcript_sha,
        "cubins": cubins,
        "sm86_cubin_present": sm86_cubin_present,
        "ldd_returncode": ldd_rc,
        "ldd_transcript_sha256": ldd_sha,
        "resolved_libraries": libraries,
        "missing_libraries": missing_libraries,
        "smoke": smoke,
    }
    row_doc["row_seal_sha256"] = canonical_sha256(row_doc)
    return row_doc


def git_head() -> str:
    rc, output = run_command(["git", "rev-parse", "HEAD"], timeout=30)
    return output.decode().strip() if rc == 0 else "0" * 40


def build_witness(
    *, nvcc: str, host_cxx: str, cuobjdump: str,
    execute_build: bool, execute_smoke: bool,
) -> dict:
    rows = selected_rows()
    built = [
        build_row(
            row, nvcc=nvcc, host_cxx=host_cxx, cuobjdump=cuobjdump,
            execute_build=execute_build, execute_smoke=execute_smoke,
        )
        for row in rows
    ]
    slugs = [row["slug"] for row in built]
    integrity = {
        "expected_rows": EXPECTED_ROWS,
        "actual_rows": len(built),
        "unique_slugs": len(slugs) == len(set(slugs)),
        "all_rows_runnable": all(row["status"] == "runnable" for row in built),
    }
    witness = {
        "schema_version": 1,
        "witness_id": WITNESS_ID,
        "arch": ARCH,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_commit": git_head(),
        "roster_sha256": sha256_file(ROSTER_PATH),
        "common_mk_sha256": sha256_file(COMMON_MK),
        "protocol_pins_sha256": sha256_file(PROTOCOL_PINS),
        "build_executed": execute_build,
        "smoke_executed": execute_smoke,
        "toolchain": {
            "nvcc": command_identity(nvcc),
            "host_cxx": command_identity(host_cxx),
            "cuobjdump": command_identity(cuobjdump),
            "make": command_identity(str(Path(subprocess.check_output(
                ["which", "make"], text=True).strip()))),
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        },
        "integrity": integrity,
        "rows": sorted(built, key=lambda row: row["slug"]),
        "pass": (
            execute_build and execute_smoke
            and len(built) == EXPECTED_ROWS
            and integrity["unique_slugs"]
            and integrity["all_rows_runnable"]
        ),
    }
    witness["witness_seal_sha256"] = canonical_sha256(witness)
    return witness


def validate_witness(
    witness: dict, *, repo_root: Path = REPO_ROOT,
) -> list[str]:
    problems: list[str] = []
    if witness.get("schema_version") != 1:
        problems.append("schema_version")
    if witness.get("witness_id") != WITNESS_ID or witness.get("arch") != ARCH:
        problems.append("identity")
    if witness.get("roster_sha256") != sha256_file(
            repo_root / "workloads/simbench_bw/roster.toml"):
        problems.append("roster_sha256")
    if witness.get("common_mk_sha256") != sha256_file(
            repo_root / "workloads/simbench_bw/common.mk"):
        problems.append("common_mk_sha256")
    if witness.get("protocol_pins_sha256") != sha256_file(
            repo_root / "tools/local_labels/campaign_protocol_pins.json"):
        problems.append("protocol_pins_sha256")
    if witness.get("build_executed") is not True or witness.get("smoke_executed") is not True:
        problems.append("execution_flags")
    rows = witness.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        problems.append("row_count")
        rows = rows if isinstance(rows, list) else []
    slugs = [row.get("slug") for row in rows if isinstance(row, dict)]
    if len(slugs) != len(set(slugs)) or any(not slug for slug in slugs):
        problems.append("slug_uniqueness")
    source_rows = {row["slug"]: row for row in selected_rows(repo_root)}
    if set(slugs) != set(source_rows):
        problems.append("source_slug_set")
    for row in rows:
        if not isinstance(row, dict):
            problems.append("row_shape")
            continue
        if row.get("status") != "runnable" or row.get("problems") != []:
            problems.append(f"row_status:{row.get('slug')}")
        if row.get("build_returncode") != 0 or row.get("sm86_cubin_present") is not True:
            problems.append(f"row_build:{row.get('slug')}")
        if row.get("missing_libraries") != []:
            problems.append(f"row_libraries:{row.get('slug')}")
        if (row.get("smoke") or {}).get("contract_ok") is not True:
            problems.append(f"row_smoke:{row.get('slug')}")
        source = source_rows.get(row.get("slug"))
        if source is not None:
            expected_level = str(source["level"]).lower()
            kernel_dir = (
                repo_root / "workloads" / "simbench_bw"
                / expected_level / source["slug"])
            try:
                shape, args, binding, shape_path = shape_and_args(source, repo_root)
                inputs = input_hashes(kernel_dir, shape_path, repo_root)
            except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
                problems.append(f"row_source_unreadable:{row.get('slug')}:{exc}")
            else:
                expected_binary = str(
                    (kernel_dir / f"{source['slug']}_run").relative_to(repo_root))
                if row.get("level") != expected_level:
                    problems.append(f"row_level:{row.get('slug')}")
                if row.get("shape") != shape or row.get("shape_point") != shape["name"]:
                    problems.append(f"row_shape:{row.get('slug')}")
                if row.get("app_args") != args or row.get("shape_binding_kind") != binding:
                    problems.append(f"row_app_args:{row.get('slug')}")
                if row.get("shape_file") != str(shape_path.relative_to(repo_root)):
                    problems.append(f"row_shape_file:{row.get('slug')}")
                if row.get("shape_file_sha256") != sha256_file(shape_path):
                    problems.append(f"row_shape_hash:{row.get('slug')}")
                if (row.get("build_inputs") != inputs
                        or row.get("build_input_set_sha256") != canonical_sha256(inputs)):
                    problems.append(f"row_inputs:{row.get('slug')}")
                if row.get("binary") != expected_binary:
                    problems.append(f"row_binary_path:{row.get('slug')}")
                binary_path = repo_root / expected_binary
                if (binary_path.is_file()
                        and row.get("binary_sha256") != sha256_file(binary_path)):
                    problems.append(f"row_binary_hash:{row.get('slug')}")
        if row.get("row_seal_sha256") != canonical_sha256({
                key: value for key, value in row.items() if key != "row_seal_sha256"}):
            problems.append(f"row_seal:{row.get('slug')}")
    expected_seal = canonical_sha256({
        key: value for key, value in witness.items() if key != "witness_seal_sha256"})
    if witness.get("witness_seal_sha256") != expected_seal:
        problems.append("witness_seal")
    if witness.get("pass") is not True:
        problems.append("pass")
    return problems


class _ObjectPairs(list):
    """Marker used to retain object pairs until duplicate keys are audited."""


def _pointer_part(value: object) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _json_pointer(parts: list[object] | tuple[object, ...]) -> str:
    return "".join(f"/{_pointer_part(part)}" for part in parts)


def _decode_json_no_duplicates(payload: bytes) -> tuple[Any, list[str]]:
    """Decode JSON while retaining every duplicate object's RFC 6901 path."""
    text = payload.decode("utf-8")
    raw = json.loads(text, object_pairs_hook=_ObjectPairs)
    duplicates: set[str] = set()

    def convert(value: Any, parts: list[object]) -> Any:
        if isinstance(value, _ObjectPairs):
            result: dict[str, Any] = {}
            seen: set[str] = set()
            for key, child in value:
                if key in seen:
                    duplicates.add(_json_pointer([*parts, key]))
                seen.add(key)
                result[key] = convert(child, [*parts, key])
            return result
        if isinstance(value, list):
            return [convert(child, [*parts, index])
                    for index, child in enumerate(value)]
        return value

    return convert(raw, []), sorted(duplicates)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, ensure_ascii=True, sort_keys=True, indent=2,
    ) + "\n").encode("utf-8")


def _safe_relative_path(value: object) -> bool:
    return isinstance(value, str) and bool(_SAFE_REL_RE.fullmatch(value))


def _normalize_inside(
    value: str | os.PathLike[str], repo_root: Path,
) -> tuple[Path | None, str]:
    root = repo_root.resolve()
    supplied = Path(value)
    lexical = supplied if supplied.is_absolute() else root / supplied
    try:
        resolved = lexical.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None, str(lexical)
    try:
        rel = resolved.relative_to(root)
    except ValueError:
        return None, str(resolved)
    return resolved, rel.as_posix()


def _read_inside_file(
    value: str | os.PathLike[str], repo_root: Path, *, family: str,
) -> tuple[Path | None, str, bytes | None, list[str]]:
    resolved, display = _normalize_inside(value, repo_root)
    if resolved is None:
        return None, display, None, [f"{family}_path_escape:{display}"]
    try:
        exists = resolved.exists()
    except OSError:
        return resolved, display, None, [f"{family}_unreadable:{display}"]
    if not exists:
        return resolved, display, None, [f"{family}_missing:{display}"]
    try:
        if not resolved.is_file():
            return resolved, display, None, [f"{family}_not_regular:{display}"]
        return resolved, display, resolved.read_bytes(), []
    except OSError:
        return resolved, display, None, [f"{family}_unreadable:{display}"]


_REGISTRY_FIELDS = {
    "witness_path", "witness_file_sha256", "witness_seal_sha256",
    "source_commit", "migrated_paths",
}
_MIGRATION_FIELDS = {"path", "witness_sha256", "current_pinned_sha256"}


def _registry_phase(
    registry_path: str | os.PathLike[str], repo_root: Path,
) -> tuple[dict | None, Path | None, list[str]]:
    path, _display, payload, errors = _read_inside_file(
        registry_path, repo_root, family="witness_registry")
    if errors:
        return None, path, sorted(set(errors))
    assert payload is not None
    try:
        registry, duplicates = _decode_json_no_duplicates(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, path, ["witness_registry_invalid:json_decode"]
    if duplicates:
        return None, path, [
            f"witness_registry_invalid:duplicate_key:{pointer}"
            for pointer in duplicates
        ]

    found: set[str] = set()
    if not isinstance(registry, dict):
        return None, path, ["witness_registry_invalid:root_type"]
    if payload != canonical_json_bytes(registry):
        found.add("witness_registry_invalid:noncanonical_bytes")
    required_root = {"schema_version", "witnesses"}
    for key in sorted(required_root - set(registry)):
        found.add(f"witness_registry_invalid:missing_field:/{key}")
    for key in sorted(set(registry) - required_root):
        found.add(f"witness_registry_invalid:unknown_field:/{key}")
    if registry.get("schema_version") != 1:
        found.add("witness_registry_invalid:schema_version")
    witnesses = registry.get("witnesses")
    if not isinstance(witnesses, list):
        found.add("witness_registry_invalid:field_type:/witnesses")
        witnesses = []
    if len(witnesses) != 1:
        found.add("witness_registry_invalid:witness_cardinality")

    witness_paths: list[str] = []
    for index, entry in enumerate(witnesses):
        base = ["witnesses", index]
        if not isinstance(entry, dict):
            found.add(
                f"witness_registry_invalid:field_type:{_json_pointer(base)}")
            continue
        for key in sorted(_REGISTRY_FIELDS - set(entry)):
            found.add(
                f"witness_registry_invalid:missing_field:"
                f"{_json_pointer([*base, key])}")
        for key in sorted(set(entry) - _REGISTRY_FIELDS):
            found.add(
                f"witness_registry_invalid:unknown_field:"
                f"{_json_pointer([*base, key])}")
        witness_path = entry.get("witness_path")
        if "witness_path" in entry and not isinstance(witness_path, str):
            found.add(
                f"witness_registry_invalid:field_type:"
                f"{_json_pointer([*base, 'witness_path'])}")
        elif "witness_path" in entry and not _safe_relative_path(witness_path):
            found.add(
                f"witness_registry_invalid:path_syntax:"
                f"{_json_pointer([*base, 'witness_path'])}")
        elif "witness_path" in entry:
            witness_paths.append(witness_path)
        for key in ("witness_file_sha256", "witness_seal_sha256"):
            value = entry.get(key)
            if key not in entry:
                continue
            if not isinstance(value, str):
                found.add(
                    f"witness_registry_invalid:field_type:"
                    f"{_json_pointer([*base, key])}")
            elif not _HEX64_RE.fullmatch(value):
                found.add(
                    f"witness_registry_invalid:hash_syntax:"
                    f"{_json_pointer([*base, key])}")
        source_commit = entry.get("source_commit")
        if "source_commit" in entry and not isinstance(source_commit, str):
            found.add(
                f"witness_registry_invalid:field_type:"
                f"{_json_pointer([*base, 'source_commit'])}")
        elif "source_commit" in entry and not _HEX40_RE.fullmatch(source_commit):
            found.add(
                f"witness_registry_invalid:commit_syntax:"
                f"{_json_pointer([*base, 'source_commit'])}")

        migrations = entry.get("migrated_paths")
        if "migrated_paths" not in entry:
            migrations = []
        elif not isinstance(migrations, list):
            found.add(
                f"witness_registry_invalid:field_type:"
                f"{_json_pointer([*base, 'migrated_paths'])}")
            migrations = []
        if "migrated_paths" in entry and len(migrations) != 4:
            found.add("witness_registry_invalid:migration_cardinality")
        migration_paths: list[str] = []
        for mindex, migration in enumerate(migrations):
            mbase = [*base, "migrated_paths", mindex]
            if not isinstance(migration, dict):
                found.add(
                    f"witness_registry_invalid:field_type:{_json_pointer(mbase)}")
                continue
            for key in sorted(_MIGRATION_FIELDS - set(migration)):
                found.add(
                    f"witness_registry_invalid:missing_field:"
                    f"{_json_pointer([*mbase, key])}")
            for key in sorted(set(migration) - _MIGRATION_FIELDS):
                found.add(
                    f"witness_registry_invalid:unknown_field:"
                    f"{_json_pointer([*mbase, key])}")
            migration_path = migration.get("path")
            if "path" in migration and not isinstance(migration_path, str):
                found.add(
                    f"witness_registry_invalid:field_type:"
                    f"{_json_pointer([*mbase, 'path'])}")
            elif "path" in migration and not _safe_relative_path(migration_path):
                found.add(
                    f"witness_registry_invalid:path_syntax:"
                    f"{_json_pointer([*mbase, 'path'])}")
            elif "path" in migration:
                migration_paths.append(migration_path)
            for key in ("witness_sha256", "current_pinned_sha256"):
                value = migration.get(key)
                if key not in migration:
                    continue
                if not isinstance(value, str):
                    found.add(
                        f"witness_registry_invalid:field_type:"
                        f"{_json_pointer([*mbase, key])}")
                elif not _HEX64_RE.fullmatch(value):
                    found.add(
                        f"witness_registry_invalid:hash_syntax:"
                        f"{_json_pointer([*mbase, key])}")
        if (migration_paths != sorted(migration_paths)
                or len(migration_paths) != len(set(migration_paths))):
            found.add("witness_registry_invalid:migration_order")
    if (witness_paths != sorted(witness_paths)
            or len(witness_paths) != len(set(witness_paths))):
        found.add("witness_registry_invalid:witness_order")
    return registry, path, sorted(found)


def _schema_error_token(error: jsonschema.ValidationError) -> str:
    pointer = _json_pointer(list(error.absolute_path))
    path_tail = list(error.absolute_path)[-1:] or [None]
    schema_parts = list(error.absolute_schema_path)
    if (error.validator == "propertyNames"
            or path_tail[0] in {"slug", "shape_file", "binary"}
            and error.validator == "pattern"
            or "propertyNames" in schema_parts):
        return f"witness_structure_invalid:path_syntax:{pointer}"
    return f"witness_structure_invalid:schema:{pointer}"


def _witness_phase2(
    payload: bytes, repo_root: Path,
) -> tuple[dict | None, list[str]]:
    try:
        witness, duplicates = _decode_json_no_duplicates(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, ["witness_json_invalid:json_decode"]
    if duplicates:
        return None, [f"witness_json_invalid:duplicate_key:{pointer}"
                      for pointer in duplicates]
    if not isinstance(witness, dict):
        return None, ["witness_structure_invalid:root_type"]
    found: set[str] = set()
    try:
        schema = json.loads((repo_root / WITNESS_SCHEMA_REL).read_text())
        validator = jsonschema.Draft202012Validator(schema)
        for error in validator.iter_errors(witness):
            found.add(_schema_error_token(error))
    except (OSError, json.JSONDecodeError, jsonschema.SchemaError):
        found.add("witness_structure_invalid:schema:")

    if witness.get("schema_version") != 1:
        found.add("witness_structure_invalid:schema_version")
    if witness.get("witness_id") != WITNESS_ID or witness.get("arch") != ARCH:
        found.add("witness_structure_invalid:identity")
    if (witness.get("build_executed") is not True
            or witness.get("smoke_executed") is not True):
        found.add("witness_structure_invalid:execution_flags")
    rows = witness.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        found.add("witness_structure_invalid:row_count")
        rows = rows if isinstance(rows, list) else []
    slugs = [row.get("slug") for row in rows if isinstance(row, dict)]
    if (len(slugs) != len(rows) or len(slugs) != len(set(slugs))
            or any(not isinstance(slug, str) or not slug for slug in slugs)):
        found.add("witness_structure_invalid:slug_uniqueness")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            found.add(f"witness_structure_invalid:row_shape:{index}")
            continue
        slug = row.get("slug")
        slug_label = str(slug)
        if not isinstance(slug, str) or not _SLUG_RE.fullmatch(slug):
            found.add(
                f"witness_structure_invalid:path_syntax:/rows/{index}/slug")
        for key in ("shape_file", "binary"):
            if not _safe_relative_path(row.get(key)):
                found.add(
                    f"witness_structure_invalid:path_syntax:/rows/{index}/{key}")
        build_inputs = row.get("build_inputs")
        if isinstance(build_inputs, dict):
            if any(not _safe_relative_path(path) for path in build_inputs):
                found.add(
                    f"witness_structure_invalid:path_syntax:"
                    f"/rows/{index}/build_inputs")
            if row.get("build_input_set_sha256") != canonical_sha256(build_inputs):
                found.add(
                    f"witness_structure_invalid:row_input_set_sha256:{slug_label}")
            shape_file = row.get("shape_file")
            if (not isinstance(shape_file, str)
                    or build_inputs.get(shape_file) != row.get("shape_file_sha256")):
                found.add(
                    f"witness_structure_invalid:row_shape_file_sha256:{slug_label}")
        if row.get("status") != "runnable" or row.get("problems") != []:
            found.add(f"witness_structure_invalid:row_status:{slug_label}")
        if (row.get("build_returncode") != 0
                or row.get("sm86_cubin_present") is not True):
            found.add(f"witness_structure_invalid:row_build:{slug_label}")
        if row.get("missing_libraries") != []:
            found.add(f"witness_structure_invalid:row_libraries:{slug_label}")
        smoke = row.get("smoke")
        if not isinstance(smoke, dict) or smoke.get("contract_ok") is not True:
            found.add(f"witness_structure_invalid:row_smoke:{slug_label}")
        level = row.get("level")
        if isinstance(slug, str):
            expected_shape = f"workloads/simbench_bw/shapes/{slug}.toml"
            if row.get("shape_file") != expected_shape:
                found.add(f"witness_structure_invalid:row_shape_file:{slug}")
            expected_binary = (
                f"workloads/simbench_bw/{level}/{slug}/{slug}_run")
            if row.get("binary") != expected_binary:
                found.add(f"witness_structure_invalid:row_binary_path:{slug}")
    if witness.get("pass") is not True:
        found.add("witness_structure_invalid:pass")
    return witness, sorted(found)


def _git(
    repo_root: Path, arguments: list[str], *, text: bool = False,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=text,
        check=False,
    )


def _phase3(witness: dict, entry: dict) -> list[str]:
    found: set[str] = set()
    rows = witness.get("rows") if isinstance(witness.get("rows"), list) else []
    input_hashes_by_path: dict[str, set[str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        slug = str(row.get("slug"))
        expected = canonical_sha256({
            key: value for key, value in row.items() if key != "row_seal_sha256"})
        if row.get("row_seal_sha256") != expected:
            found.add(f"row_seal_mismatch:{slug}")
        for path, digest in (row.get("build_inputs") or {}).items():
            input_hashes_by_path.setdefault(path, set()).add(digest)
    expected_witness = canonical_sha256({
        key: value for key, value in witness.items()
        if key != "witness_seal_sha256"})
    if witness.get("witness_seal_sha256") != expected_witness:
        found.add("witness_seal_mismatch")
    if entry.get("witness_seal_sha256") != witness.get("witness_seal_sha256"):
        found.add("witness_registry_mismatch:witness_seal")
    if entry.get("source_commit") != witness.get("source_commit"):
        found.add("witness_registry_mismatch:source_commit")
    for migration in entry.get("migrated_paths", []):
        path = migration["path"]
        if path not in input_hashes_by_path:
            found.add(
                f"witness_registry_mismatch:migrated_path_not_in_witness:{path}")
        elif migration.get("witness_sha256") not in input_hashes_by_path[path]:
            found.add(f"witness_registry_mismatch:witness_sha256:{path}")
    return sorted(found)


def _phase4(witness: dict, repo_root: Path) -> list[str]:
    commit = str(witness.get("source_commit"))
    resolved = _git(repo_root, ["rev-parse", "--verify", f"{commit}^{{commit}}"])
    if resolved.returncode == 0:
        return []
    shallow = _git(repo_root, ["rev-parse", "--is-shallow-repository"], text=True)
    suffix = ":shallow_clone" if (
        shallow.returncode == 0 and shallow.stdout.strip() == "true") else ""
    return [f"witness_commit_unresolvable:{commit}{suffix}"]


def _phase5(witness: dict, repo_root: Path) -> list[str]:
    commit = witness["source_commit"]
    rows = [row for row in witness["rows"] if isinstance(row, dict)]
    cache: dict[str, bytes | None] = {}

    def blob(path: str) -> bytes | None:
        if path not in cache:
            try:
                proc = _git(repo_root, ["cat-file", "blob", f"{commit}:{path}"])
            except (OSError, ValueError):
                cache[path] = None
            else:
                cache[path] = proc.stdout if proc.returncode == 0 else None
        return cache[path]

    found: set[str] = set()
    for row in rows:
        slug = row["slug"]
        for path, expected in row["build_inputs"].items():
            payload = blob(path)
            if payload is None:
                found.add(f"row_inputs_at_commit:{slug}:{path}:missing_at_commit")
            elif sha256_bytes(payload) != expected:
                found.add(f"row_inputs_at_commit:{slug}:{path}:blob_mismatch")
    for field, path in TOP_INPUTS.items():
        payload = blob(path)
        if payload is None:
            found.add(f"witness_top_input_at_commit:{path}:missing_at_commit")
        elif sha256_bytes(payload) != witness.get(field):
            found.add(f"witness_top_input_at_commit:{path}:blob_mismatch")
    if found:
        return sorted(found)

    roster_payload = cache[TOP_INPUTS["roster_sha256"]]
    assert roster_payload is not None
    try:
        roster = tomllib.loads(roster_payload.decode("utf-8"))
        source_rows_list = roster.get("rows")
        if not isinstance(source_rows_list, list):
            raise ValueError("rows is not a list")
        selected: dict[str, dict] = {}
        for source in source_rows_list:
            if not isinstance(source, dict):
                raise ValueError("source row is not an object")
            gate = source.get("arch_gate") or []
            if not isinstance(gate, list):
                raise ValueError("arch_gate is not a list")
            if ARCH not in gate:
                continue
            slug = source.get("slug")
            level = source.get("level")
            if (not isinstance(slug, str) or not _SLUG_RE.fullmatch(slug)
                    or slug in selected or not isinstance(level, str)
                    or level.lower() not in {"l1", "l2", "l3", "l4"}):
                raise ValueError("invalid selected row")
            selected[slug] = source
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError, TypeError):
        return ["witness_source_at_commit:roster_invalid"]

    witness_slugs = {row["slug"] for row in rows}
    if len(selected) != EXPECTED_ROWS or witness_slugs != set(selected):
        found.add("witness_source_at_commit:slug_set_mismatch")

    tree = _git(repo_root, [
        "ls-tree", "-rz", "--name-only", commit, "--", "workloads/simbench_bw",
    ])
    if tree.returncode != 0:
        found.add("witness_source_at_commit:input_scan_failed")
        return sorted(found)
    try:
        tree_paths = {
            item.decode("utf-8") for item in tree.stdout.split(b"\0") if item
        }
    except UnicodeDecodeError:
        found.add("witness_source_at_commit:input_scan_failed")
        return sorted(found)

    for row in rows:
        slug = row["slug"]
        source = selected.get(slug)
        if source is None:
            continue
        source_level = str(source["level"]).lower()
        if row.get("level") != source_level:
            found.add(f"row_source_at_commit:{slug}:level_mismatch")
        expected_binary = (
            f"workloads/simbench_bw/{source_level}/{slug}/{slug}_run")
        if row.get("binary") != expected_binary:
            found.add(f"row_source_at_commit:{slug}:binary_path_mismatch")
        shape_path = f"workloads/simbench_bw/shapes/{slug}.toml"
        shape_payload = blob(shape_path)
        try:
            if shape_payload is None:
                raise ValueError("shape missing")
            shape_doc = tomllib.loads(shape_payload.decode("utf-8"))
            shape, args, binding = _shape_and_args_docs(source, shape_doc)
        except (UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError, TypeError):
            found.add(f"row_source_at_commit:{slug}:shape_invalid")
        else:
            if row.get("shape") != shape or row.get("shape_point") != shape["name"]:
                found.add(f"row_source_at_commit:{slug}:shape_mismatch")
            if row.get("shape_binding_kind") != binding:
                found.add(
                    f"row_source_at_commit:{slug}:shape_binding_kind_mismatch")
            if row.get("app_args") != args:
                found.add(f"row_source_at_commit:{slug}:app_args_mismatch")

        prefix = f"workloads/simbench_bw/{source_level}/{slug}/"
        expected_inputs = {
            "workloads/simbench_bw/common.mk", shape_path,
        }
        for path in tree_paths:
            candidate = Path(path)
            if path.startswith(prefix) and (
                    candidate.name == "Makefile"
                    or candidate.suffix in INPUT_SUFFIXES):
                expected_inputs.add(path)
        recorded_inputs = set(row["build_inputs"])
        for path in sorted(expected_inputs - recorded_inputs):
            found.add(
                f"row_inputs_at_commit:{slug}:{path}:missing_recorded_input")
        for path in sorted(recorded_inputs - expected_inputs):
            found.add(
                f"row_inputs_at_commit:{slug}:{path}:unexpected_recorded_input")
    return sorted(found)


def _phase6(witness: dict, entry: dict, repo_root: Path) -> list[str]:
    rows = [row for row in witness["rows"] if isinstance(row, dict)]
    expected: dict[str, str] = {}
    for row in rows:
        expected.update(row["build_inputs"])
    for field, path in TOP_INPUTS.items():
        expected[path] = witness[field]
    migrations = {item["path"]: item for item in entry["migrated_paths"]}
    found: set[str] = set()

    for path, historical_digest in sorted(expected.items()):
        resolved, display = _normalize_inside(path, repo_root)
        if resolved is None:
            found.add(f"current_source_path_escape:{display}")
            continue
        try:
            exists = resolved.exists()
        except OSError:
            found.add(f"current_source_unreadable:{display}")
            continue
        if not exists:
            found.add(f"missing_current_source:{path}")
            continue
        try:
            if not resolved.is_file():
                found.add(f"current_source_not_regular:{path}")
                continue
            current_digest = sha256_file(resolved)
        except OSError:
            found.add(f"current_source_unreadable:{path}")
            continue
        migration = migrations.get(path)
        if migration is None:
            if current_digest != historical_digest:
                found.add(f"unmigrated_source_drift:{path}")
            continue
        pinned = migration["current_pinned_sha256"]
        witness_digest = migration["witness_sha256"]
        if current_digest != pinned:
            found.add(f"migration_pin_mismatch:{path}")
        if current_digest == witness_digest or pinned == witness_digest:
            found.add(f"migration_pin_stale:{path}")

    recorded = set(expected)
    for row in rows:
        slug = row["slug"]
        directory_rel = f"workloads/simbench_bw/{row['level']}/{slug}"
        directory, display = _normalize_inside(directory_rel, repo_root)
        if directory is None:
            found.add(f"current_source_path_escape:{display}")
            continue
        try:
            children = list(directory.iterdir())
        except OSError:
            found.add(f"current_source_scan_failed:{directory_rel}")
            continue
        for child in children:
            if child.name != "Makefile" and child.suffix not in INPUT_SUFFIXES:
                continue
            resolved_child, child_display = _normalize_inside(child, repo_root)
            if resolved_child is None:
                found.add(f"current_source_path_escape:{child_display}")
                continue
            try:
                if not resolved_child.is_file():
                    found.add(f"current_source_not_regular:{child_display}")
                    continue
            except OSError:
                found.add(f"current_source_unreadable:{child_display}")
                continue
            rel = child_display
            if rel not in recorded:
                found.add(f"new_current_source:{rel}")
    return sorted(found)


def validate_witness_file(
    witness_path: str | os.PathLike[str], *, repo_root: Path,
    registry_path: str | os.PathLike[str] | None = None,
) -> list[str]:
    """Validate a witness against its registered historical source contract."""
    repo_root = Path(repo_root).resolve()
    selected_registry = registry_path if registry_path is not None else DEFAULT_REGISTRY_REL
    registry, _registry_file, errors = _registry_phase(selected_registry, repo_root)
    if errors:
        return errors
    assert registry is not None
    entry = registry["witnesses"][0]

    witness_file, display, payload, errors = _read_inside_file(
        witness_path, repo_root, family="witness_file")
    if errors:
        return errors
    assert witness_file is not None and payload is not None
    normalized_rel = witness_file.relative_to(repo_root).as_posix()
    registered = normalized_rel == entry["witness_path"]
    if registered and sha256_bytes(payload) != entry["witness_file_sha256"]:
        return [f"witness_file_sha256_mismatch:{display}"]

    if not registered:
        try:
            legacy_witness, duplicates = _decode_json_no_duplicates(payload)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return ["witness_json_invalid:json_decode"]
        if duplicates:
            return [f"witness_json_invalid:duplicate_key:{pointer}"
                    for pointer in duplicates]
        if not isinstance(legacy_witness, dict):
            return ["witness_structure_invalid:root_type"]
        return validate_witness(legacy_witness, repo_root=repo_root)

    witness, errors = _witness_phase2(payload, repo_root)
    if errors:
        return errors
    assert witness is not None
    errors = _phase3(witness, entry)
    if errors:
        return errors
    errors = _phase4(witness, repo_root)
    if errors:
        return errors
    errors = _phase5(witness, repo_root)
    if errors:
        return errors
    return _phase6(witness, entry, repo_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--build", action="store_true",
                        help="execute all 31 builds (required for pass=true)")
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--nvcc", default="/opt/cuda/bin/nvcc")
    parser.add_argument("--host-cxx", default="/usr/bin/g++-15")
    parser.add_argument("--cuobjdump", default="/opt/cuda/bin/cuobjdump")
    args = parser.parse_args(argv)

    try:
        witness = build_witness(
            nvcc=str(Path(args.nvcc).resolve()),
            host_cxx=str(Path(args.host_cxx).resolve()),
            cuobjdump=str(Path(args.cuobjdump).resolve()),
            execute_build=args.build,
            execute_smoke=args.build and not args.no_smoke,
        )
        problems = validate_witness(witness)
    except Exception as exc:  # noqa: BLE001 — whole-witness generation blocker
        print(f"build_simbench_build_witness: failed: {exc}", file=sys.stderr)
        return 2

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(witness, indent=2, sort_keys=True) + "\n")
    print(
        f"simbench build witness: rows={len(witness['rows'])} "
        f"pass={witness['pass']} problems={problems} -> {out}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
