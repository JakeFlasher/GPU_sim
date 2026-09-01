from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import importlib.util
import marshal
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from tools.nvbit_tracer.proto_binding_loader import (
    BINDING_PINS,
    ProtoBindingIntegrityError,
    ProtoBindingRuntimePrerequisiteError,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BINDING_ROOT = REPO_ROOT / "workloads" / "simbench_bw" / "_pb_generated"
CANONICAL_MODULES = frozenset(pin.module_name for pin in BINDING_PINS)
CANONICAL_FILENAMES = frozenset(pin.filename for pin in BINDING_PINS)
AUTHORIZED_CPP_BINDING_BUILDER = (
    REPO_ROOT / "tools" / "nvbit_tracer" / "external_tracer_build.py"
)


def _copy_bindings(destination: Path) -> Path:
    destination.mkdir(parents=True)
    for pin in BINDING_PINS:
        shutil.copyfile(BINDING_ROOT / pin.filename, destination / pin.filename)
    return destination


def _run_subprocess(source: str, *args: object) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.fspath(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, "-B", "-c", source, *(os.fspath(arg) for arg in args)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"subprocess failed ({result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


def _expect_integrity_script(extra: str = "") -> str:
    return f"""
import pathlib
import sys
import tools.nvbit_tracer.proto_binding_loader as loader
loader._BINDING_ROOT = pathlib.Path(sys.argv[1])
{extra}
try:
    loader.load_dynamic_trace_bindings()
except loader.ProtoBindingIntegrityError as exc:
    assert "tracked protobuf authority" in str(exc), str(exc)
else:
    raise AssertionError("contaminated authority was accepted")
assert not any(name in sys.modules for name in loader._CANONICAL_MODULE_NAMES)
"""


def test_literal_pin_table_covers_exact_tracked_authorities() -> None:
    expected = {
        "address_pb2.py": (
            2113,
            "d3307a8fb59b5915fdcfe863c4c72a0144d32a001632669385689f30b2baef33",
        ),
        "cuda_stream_pb2.py": (
            1455,
            "7fdcab45b3a9206aba4c24b04fb4359196010dda30f9b5891179a11ce90ab198",
        ),
        "dim3d_pb2.py": (
            1309,
            "6d933c129b3e21c81c60d4d6c2f9093b9c28d1c07ce085954546e01f3447f8d9",
        ),
        "gpu_device_pb2.py": (
            1826,
            "339a921d00e1146472cefad9cf9186a6d5d415053b461826c8e5c91fcfbf2b31",
        ),
        "instruction_pb2.py": (
            4270,
            "b43918eff894e2a87f7dd64bc7d1344fc12f300b92534999d0aa100228daa81f",
        ),
        "kernel_pb2.py": (
            2024,
            "d84c72e4217dd3efe535e04ffb6549137cfdc79e384e07db8bb5fad6e29cf6c0",
        ),
        "threadblock_pb2.py": (
            1885,
            "b3e95c0fe358864226bcb1933cad6b922d4cc30293babac4e2cc872aa1d3da8b",
        ),
        "trace_pb2.py": (
            2187,
            "4e479bc1271a1720733577a08a23bbbdd2ad8b700809ba26d5f0ab047868487b",
        ),
        "warp_pb2.py": (
            1377,
            "842afd34bc76ac5245debbf5e4d389c9a9f9762b4c6c60b23a51dd7b9e05c967",
        ),
    }
    observed = {
        pin.filename: (pin.size_bytes, pin.sha256) for pin in BINDING_PINS
    }
    assert observed == expected
    assert {
        path.name
        for path in BINDING_ROOT.iterdir()
        if path.is_file() and path.name.endswith("_pb2.py")
    } == set(expected)
    for pin in BINDING_PINS:
        payload = (BINDING_ROOT / pin.filename).read_bytes()
        assert len(payload) == pin.size_bytes
        assert hashlib.sha256(payload).hexdigest() == pin.sha256


def test_current_authorities_load_as_one_complete_immutable_bundle() -> None:
    result = _run_subprocess(
        """
from dataclasses import FrozenInstanceError
from tools.nvbit_tracer.proto_binding_loader import (
    BINDING_PINS, load_dynamic_trace_bindings,
)
bundle = load_dynamic_trace_bindings()
assert len(bundle.modules) == 9
assert tuple(module.__name__ for module in bundle.modules) == tuple(
    pin.module_name for pin in BINDING_PINS
)
assert bundle.trace_pb2.DESCRIPTOR.name == "trace.proto"
assert bundle.threadblock_pb2.DESCRIPTOR.name == "threadblock.proto"
assert all(module.__cached__ is None for module in bundle.modules)
try:
    bundle.trace_pb2 = None
except FrozenInstanceError:
    pass
else:
    raise AssertionError("binding bundle is mutable")
print("ok")
"""
    )
    assert result.stdout.strip() == "ok"


def test_instruction_binding_exposes_optional_instrumentation_order_tag14() -> None:
    result = _run_subprocess(
        """
from tools.nvbit_tracer.proto_binding_loader import load_dynamic_trace_bindings
instruction_pb2 = load_dynamic_trace_bindings().instruction_pb2
field = instruction_pb2.instruction.DESCRIPTOR.fields_by_name["instrumentation_order"]
assert field.number == 14
assert field.has_presence is True
instruction = instruction_pb2.instruction()
assert not instruction.HasField("instrumentation_order")
legacy_bytes = instruction.SerializeToString(deterministic=True)
instruction.instrumentation_order = 1
assert instruction.HasField("instrumentation_order")
assert instruction.instrumentation_order == 1
assert instruction.SerializeToString(deterministic=True) != legacy_bytes
print("ok")
"""
    )
    assert result.stdout.strip() == "ok"


def test_load_isolates_descriptors_classes_and_authenticated_consumers() -> None:
    _run_subprocess(
        """
from google.protobuf import descriptor_pool, symbol_database
import tools.nvbit_tracer.proto_binding_loader as loader
import workloads.simbench_bw.nvbit_trace_summary_adapter as adapter

def assert_identity_mapping(observed, expected):
    assert set(observed) == set(expected)
    assert all(observed[key] is value for key, value in expected.items())

def assert_default_pool_clean(pool):
    for pin in loader.BINDING_PINS:
        try:
            pool.FindFileByName(pin.proto_name)
        except KeyError:
            pass
        else:
            raise AssertionError(f"default pool contains {pin.proto_name}")

default_pool = descriptor_pool.Default()
public_database = symbol_database.Default()
public_classes = symbol_database.SymbolDatabase._classes
assert public_database._classes is public_classes
public_snapshot = dict(public_classes)
assert_default_pool_clean(default_pool)

bundle = loader.load_dynamic_trace_bindings()
private_pool = bundle.address_pb2.DESCRIPTOR.pool
assert private_pool is not default_pool
assert all(module.DESCRIPTOR.pool is private_pool for module in bundle.modules)
assert_default_pool_clean(default_pool)
assert_identity_mapping(public_classes, public_snapshot)
for symbol in ("dynamic_trace.address", "dynamic_trace.threadblock", "dynamic_trace.Trace"):
    try:
        public_database.GetSymbol(symbol)
    except KeyError:
        pass
    else:
        raise AssertionError(f"public symbol database contains {symbol}")

trace_pb2, threadblock_pb2 = adapter._import_pb_modules()
assert trace_pb2 is bundle.trace_pb2
assert threadblock_pb2 is bundle.threadblock_pb2
trace = trace_pb2.Trace(name="isolated", binary_version=86)
device = trace.gpu_device[7]
device.id = 7
stream = device.streams[3]
stream.id = 3
kernel = stream.kernels.add(id=11, name="consumer")
kernel.grid_dim.x = 2
kernel.block_dim.x = 32
round_trip = trace_pb2.Trace.FromString(trace.SerializeToString())
assert round_trip == trace
threadblock = threadblock_pb2.threadblock()
threadblock.block_id.x = 4
threadblock.warps[0].id = 0
assert threadblock_pb2.threadblock.FromString(
    threadblock.SerializeToString()
) == threadblock
assert trace_pb2.Trace.__module__ == "trace_pb2"
assert trace_pb2.Trace.GpuDeviceEntry.__qualname__ == "Trace.GpuDeviceEntry"

assert loader.load_dynamic_trace_bindings() is bundle
assert_default_pool_clean(default_pool)
assert_identity_mapping(public_classes, public_snapshot)
"""
    )


def test_concurrent_callers_publish_one_complete_bundle() -> None:
    _run_subprocess(
        """
from concurrent.futures import ThreadPoolExecutor
from google.protobuf import descriptor_pool, symbol_database
from tools.nvbit_tracer.proto_binding_loader import (
    BINDING_PINS, load_dynamic_trace_bindings,
)
default_pool = descriptor_pool.Default()
public_classes = symbol_database.SymbolDatabase._classes
public_snapshot = dict(public_classes)
with ThreadPoolExecutor(max_workers=8) as pool:
    bundles = list(pool.map(lambda _: load_dynamic_trace_bindings(), range(32)))
assert all(bundle is bundles[0] for bundle in bundles)
assert len(bundles[0].modules) == 9
assert set(public_classes) == set(public_snapshot)
assert all(public_classes[key] is value for key, value in public_snapshot.items())
for pin in BINDING_PINS:
    try:
        default_pool.FindFileByName(pin.proto_name)
    except KeyError:
        pass
    else:
        raise AssertionError(f"default pool contains {pin.proto_name}")
"""
    )


@pytest.mark.parametrize("mutation", ("size", "hash", "missing"))
def test_wrong_size_hash_and_missing_member_fail_before_execution(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = _copy_bindings(tmp_path / "bindings")
    source = root / "address_pb2.py"
    if mutation == "size":
        source.write_bytes(source.read_bytes() + b"\n")
    elif mutation == "hash":
        payload = bytearray(source.read_bytes())
        payload[0] ^= 1
        source.write_bytes(payload)
    else:
        source.unlink()
    _run_subprocess(_expect_integrity_script(), root)


@pytest.mark.parametrize("topology", ("leaf_symlink", "hardlink", "fifo"))
def test_leaf_symlink_hardlink_and_fifo_are_rejected_without_execution(
    tmp_path: Path,
    topology: str,
) -> None:
    root = _copy_bindings(tmp_path / "bindings")
    source = root / "address_pb2.py"
    retained = tmp_path / "retained-address.py"
    shutil.copyfile(source, retained)
    source.unlink()
    if topology == "leaf_symlink":
        source.symlink_to(retained)
    elif topology == "hardlink":
        os.link(retained, source)
    else:
        os.mkfifo(source)
    _run_subprocess(_expect_integrity_script(), root)


def test_symlinked_ancestor_is_rejected(tmp_path: Path) -> None:
    actual_parent = tmp_path / "actual"
    actual_root = _copy_bindings(actual_parent / "bindings")
    alias = tmp_path / "alias"
    alias.symlink_to(actual_parent, target_is_directory=True)
    assert actual_root.is_dir()
    _run_subprocess(_expect_integrity_script(), alias / "bindings")


@pytest.mark.parametrize("candidate", ("pyc", "native", "package"))
def test_top_level_same_stem_import_candidates_fail_before_execution(
    tmp_path: Path,
    candidate: str,
) -> None:
    root = _copy_bindings(tmp_path / "bindings")
    if candidate == "pyc":
        (root / "trace_pb2.pyc").write_bytes(b"planted sourceless payload")
        trap = """
import importlib.machinery
candidate_loader_calls = []
def reject_cache(self, fullname):
    candidate_loader_calls.append((self.path, fullname))
    raise AssertionError("sourceless loader was consulted")
importlib.machinery.SourcelessFileLoader.get_code = reject_cache
"""
    elif candidate == "native":
        suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
        (root / f"trace_pb2{suffix}").write_bytes(b"planted native payload")
        trap = """
import importlib.machinery
candidate_loader_calls = []
_original_native_exec = importlib.machinery.ExtensionFileLoader.exec_module
def reject_native(self, module):
    if module.__name__ == "trace_pb2":
        candidate_loader_calls.append((self.path, module.__name__))
        raise AssertionError("trace_pb2 extension loader was consulted")
    return _original_native_exec(self, module)
importlib.machinery.ExtensionFileLoader.exec_module = reject_native
"""
    else:
        (root / "trace_pb2").mkdir()
        trap = "candidate_loader_calls = []"
    _run_subprocess(
        _expect_integrity_script(trap) + "\nassert candidate_loader_calls == []\n",
        root,
    )


def test_timestamp_cache_payload_is_ignored_and_never_executed(tmp_path: Path) -> None:
    root = _copy_bindings(tmp_path / "bindings")
    marker = tmp_path / "pyc-executed"
    source = root / "trace_pb2.py"
    source_stat = source.stat()
    planted_code = compile(
        f"from pathlib import Path\nPath({os.fspath(marker)!r}).write_text('bad')\n",
        os.fspath(source),
        "exec",
    )
    cache_dir = root / "__pycache__"
    cache_dir.mkdir()
    cache_path = cache_dir / f"trace_pb2.{sys.implementation.cache_tag}.pyc"
    header = (
        importlib.util.MAGIC_NUMBER
        + struct.pack("<I", 0)
        + struct.pack("<II", int(source_stat.st_mtime) & 0xFFFFFFFF, source_stat.st_size)
    )
    cache_path.write_bytes(header + marshal.dumps(planted_code))
    _run_subprocess(
        """
import pathlib
import sys
import tools.nvbit_tracer.proto_binding_loader as loader
loader._BINDING_ROOT = pathlib.Path(sys.argv[1])
marker = pathlib.Path(sys.argv[2])
bundle = loader.load_dynamic_trace_bindings()
assert bundle.trace_pb2.DESCRIPTOR.name == "trace.proto"
assert not marker.exists(), "timestamp bytecode cache executed"
""",
        root,
        marker,
    )
    assert not marker.exists()


def test_hostile_sys_path_and_meta_path_are_not_consulted(tmp_path: Path) -> None:
    root = _copy_bindings(tmp_path / "bindings")
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    marker = tmp_path / "hostile-executed"
    (hostile / "trace_pb2.py").write_text(
        f"from pathlib import Path\nPath({os.fspath(marker)!r}).write_text('bad')\n",
        encoding="utf-8",
    )
    _run_subprocess(
        """
import importlib.abc
import importlib.util
import pathlib
import sys
import tools.nvbit_tracer.proto_binding_loader as loader
root, hostile, marker = map(pathlib.Path, sys.argv[1:4])
loader._BINDING_ROOT = root
calls = []
class HostileFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in loader._CANONICAL_MODULE_NAMES:
            calls.append(fullname)
            return importlib.util.spec_from_file_location(
                fullname, hostile / "trace_pb2.py"
            )
        return None
finder = HostileFinder()
sys.meta_path.insert(0, finder)
sys.path.insert(0, str(hostile))
before = tuple(sys.path)
bundle = loader.load_dynamic_trace_bindings()
assert bundle.trace_pb2.DESCRIPTOR.name == "trace.proto"
assert calls == []
assert tuple(sys.path) == before
assert not marker.exists()
""",
        root,
        hostile,
        marker,
    )
    assert not marker.exists()


def test_preloaded_canonical_sys_modules_entry_is_rejected(tmp_path: Path) -> None:
    root = _copy_bindings(tmp_path / "bindings")
    _run_subprocess(
        """
import pathlib
import sys
import types
import tools.nvbit_tracer.proto_binding_loader as loader
loader._BINDING_ROOT = pathlib.Path(sys.argv[1])
fake = types.ModuleType("trace_pb2")
fake.__file__ = str(loader._BINDING_ROOT / "trace_pb2.py")
sys.modules["trace_pb2"] = fake
try:
    loader.load_dynamic_trace_bindings()
except loader.ProtoBindingIntegrityError as exc:
    assert "preloaded" in str(exc)
else:
    raise AssertionError("preloaded module was accepted")
assert sys.modules["trace_pb2"] is fake
""",
        root,
    )


def test_public_descriptor_pool_constructor_is_never_executed() -> None:
    _run_subprocess(
        """
from google.protobuf import descriptor_pool
real_descriptor_pool = descriptor_pool.DescriptorPool
constructor_calls = []
class PlantedDescriptorPool:
    def __new__(cls, *args, **kwargs):
        constructor_calls.append((args, kwargs))
        return real_descriptor_pool(*args, **kwargs)
descriptor_pool.DescriptorPool = PlantedDescriptorPool
import tools.nvbit_tracer.proto_binding_loader as loader
bundle = loader.load_dynamic_trace_bindings()
assert bundle.trace_pb2.DESCRIPTOR.pool is not descriptor_pool.Default()
assert constructor_calls == []
for pin in loader.BINDING_PINS:
    try:
        descriptor_pool.Default().FindFileByName(pin.proto_name)
    except KeyError:
        pass
    else:
        raise AssertionError(f"default pool contains {pin.proto_name}")
"""
    )


@pytest.mark.parametrize("target", ("metaclass", "message", "api_implementation"))
def test_preimport_message_factory_global_contamination_is_rejected(
    target: str,
) -> None:
    _run_subprocess(
        """
import sys
from google.protobuf import descriptor_pool, message_factory, reflection
calls = []
target = sys.argv[1]
if target == "metaclass":
    real_metaclass = message_factory._GENERATED_PROTOCOL_MESSAGE_TYPE
    def planted_metaclass(*args, **kwargs):
        calls.append((args, kwargs))
        return real_metaclass(*args, **kwargs)
    message_factory._GENERATED_PROTOCOL_MESSAGE_TYPE = planted_metaclass
elif target == "message":
    message_factory.message = object()
else:
    class PlantedApiImplementation:
        @staticmethod
        def Type():
            calls.append("Type")
            return "upb"
    message_factory.api_implementation = PlantedApiImplementation
import tools.nvbit_tracer.proto_binding_loader as loader
try:
    loader.load_dynamic_trace_bindings()
except loader.ProtoBindingRuntimePrerequisiteError as exc:
    assert "message factory runtime bindings changed" in str(exc), str(exc)
else:
    raise AssertionError("pre-import message-factory contamination was accepted")
assert calls == []
assert not any(name in sys.modules for name in loader._CANONICAL_MODULE_NAMES)
for pin in loader.BINDING_PINS:
    try:
        descriptor_pool.Default().FindFileByName(pin.proto_name)
    except KeyError:
        pass
    else:
        raise AssertionError(f"default pool contains {pin.proto_name}")
""",
        target,
    )


@pytest.mark.parametrize("mutation", ("new", "namespace"))
def test_preimport_enum_wrapper_class_mutation_is_rejected_before_binding_import(
    mutation: str,
) -> None:
    _run_subprocess(
        """
import sys
from google.protobuf import descriptor_pool
from google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper

constructor_calls = []
mutation = sys.argv[1]
if mutation == "new":
    def planted_new(cls, *args, **kwargs):
        constructor_calls.append((args, kwargs))
        return object.__new__(cls)
    EnumTypeWrapper.__new__ = planted_new
else:
    EnumTypeWrapper.planted_namespace_member = object()

import tools.nvbit_tracer.proto_binding_loader as loader
binding_root_calls = []
def reject_binding_root(*args, **kwargs):
    binding_root_calls.append((args, kwargs))
    raise AssertionError("generated binding root was opened")
loader._open_directory_chain = reject_binding_root
try:
    loader.load_dynamic_trace_bindings()
except loader.ProtoBindingRuntimePrerequisiteError as exc:
    assert "EnumTypeWrapper class" in str(exc), str(exc)
else:
    raise AssertionError("pre-import EnumTypeWrapper mutation was accepted")
assert constructor_calls == []
assert binding_root_calls == []
assert not any(name in sys.modules for name in loader._CANONICAL_MODULE_NAMES)
for pin in loader.BINDING_PINS:
    try:
        descriptor_pool.Default().FindFileByName(pin.proto_name)
    except KeyError:
        pass
    else:
        raise AssertionError(f"default pool contains {pin.proto_name}")
""",
        mutation,
    )


def test_runtime_version_must_match_exact_pin(tmp_path: Path) -> None:
    root = _copy_bindings(tmp_path / "bindings")
    _run_subprocess(
        """
import pathlib
import sys
import google.protobuf
import tools.nvbit_tracer.proto_binding_loader as loader
loader._BINDING_ROOT = pathlib.Path(sys.argv[1])
google.protobuf.__version__ = "7.34.2"
try:
    loader.load_dynamic_trace_bindings()
except loader.ProtoBindingRuntimePrerequisiteError as exc:
    assert "expected exactly '7.34.1'" in str(exc)
else:
    raise AssertionError("wrong protobuf runtime was accepted")
assert not any(name in sys.modules for name in loader._CANONICAL_MODULE_NAMES)
""",
        root,
    )


def test_runtime_components_must_match_exact_pin(tmp_path: Path) -> None:
    root = _copy_bindings(tmp_path / "bindings")
    _run_subprocess(
        """
import pathlib
import sys
from google.protobuf import runtime_version
import tools.nvbit_tracer.proto_binding_loader as loader
loader._BINDING_ROOT = pathlib.Path(sys.argv[1])
runtime_version.PATCH = 2
try:
    loader.load_dynamic_trace_bindings()
except loader.ProtoBindingRuntimePrerequisiteError as exc:
    assert "in-memory version components" in str(exc)
    assert "(7, 34, 1, '')" in str(exc)
else:
    raise AssertionError("wrong protobuf runtime components were accepted")
assert not any(name in sys.modules for name in loader._CANONICAL_MODULE_NAMES)
""",
        root,
    )


def test_runtime_check_performs_no_pathlib_file_discovery(tmp_path: Path) -> None:
    root = _copy_bindings(tmp_path / "bindings")
    _run_subprocess(
        """
import pathlib
import sys
import tools.nvbit_tracer.proto_binding_loader as loader
loader._BINDING_ROOT = pathlib.Path(sys.argv[1])

def reject_pathlib_io(*_args, **_kwargs):
    raise AssertionError("loader attempted pathlib file discovery")

pathlib.Path.open = reject_pathlib_io
pathlib.Path.read_bytes = reject_pathlib_io
bundle = loader.load_dynamic_trace_bindings()
assert len(bundle.modules) == 9
""",
        root,
    )


def test_descriptor_pool_contamination_is_rejected_before_execution(
    tmp_path: Path,
) -> None:
    root = _copy_bindings(tmp_path / "bindings")
    _run_subprocess(
        """
import pathlib
import sys
from google.protobuf import descriptor_pb2, descriptor_pool
import tools.nvbit_tracer.proto_binding_loader as loader
loader._BINDING_ROOT = pathlib.Path(sys.argv[1])
contaminant = descriptor_pb2.FileDescriptorProto(name="address.proto")
descriptor_pool.Default().AddSerializedFile(contaminant.SerializeToString())
try:
    loader.load_dynamic_trace_bindings()
except loader.ProtoBindingIntegrityError as exc:
    assert "already contains" in str(exc)
else:
    raise AssertionError("descriptor-pool contamination was accepted")
assert not any(name in sys.modules for name in loader._CANONICAL_MODULE_NAMES)
""",
        root,
    )


def test_all_nine_are_authenticated_and_compiled_before_execution(
    tmp_path: Path,
) -> None:
    root = _copy_bindings(tmp_path / "bindings")
    _run_subprocess(
        """
import hashlib
import pathlib
import sys
from google.protobuf import descriptor_pool
import tools.nvbit_tracer.proto_binding_loader as loader
loader._BINDING_ROOT = pathlib.Path(sys.argv[1])
called = []
def stop_before_exec(directory, sources, runtime):
    called.append(True)
    assert len(sources) == 9
    assert all(source.code is not None for source in sources)
    assert [source.pin.module_name for source in sources] == [
        pin.module_name for pin in loader.BINDING_PINS
    ]
    assert all(
        hashlib.sha256(source.payload).hexdigest() == source.pin.sha256
        for source in sources
    )
    raise RuntimeError("planted stop at first execution boundary")
loader._execute_compiled_modules = stop_before_exec
try:
    loader.load_dynamic_trace_bindings()
except loader.ProtoBindingIntegrityError:
    pass
else:
    raise AssertionError("execution boundary did not run")
assert called == [True]
for pin in loader.BINDING_PINS:
    try:
        descriptor_pool.Default().FindFileByName(pin.proto_name)
    except KeyError:
        pass
    else:
        raise AssertionError(f"{pin.proto_name} executed before boundary")
""",
        root,
    )


@pytest.mark.parametrize("mutation", ("replace", "in_place"))
def test_source_replacement_or_mutation_before_execution_is_rejected(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = _copy_bindings(tmp_path / "bindings")
    _run_subprocess(
        """
import os
import pathlib
import sys
from google.protobuf import descriptor_pool
import tools.nvbit_tracer.proto_binding_loader as loader
loader._BINDING_ROOT = pathlib.Path(sys.argv[1])
mode = sys.argv[2]
original_compile = loader._compile_sources
def compile_then_mutate(directory, sources):
    original_compile(directory, sources)
    target = directory.path / "address_pb2.py"
    if mode == "replace":
        replacement = directory.path / ".replacement"
        replacement.write_bytes(target.read_bytes())
        os.replace(replacement, target)
    else:
        payload = bytearray(target.read_bytes())
        payload[0] ^= 1
        target.write_bytes(payload)
loader._compile_sources = compile_then_mutate
try:
    loader.load_dynamic_trace_bindings()
except loader.ProtoBindingIntegrityError:
    pass
else:
    raise AssertionError("source race was accepted")
assert not any(name in sys.modules for name in loader._CANONICAL_MODULE_NAMES)
for pin in loader.BINDING_PINS:
    try:
        descriptor_pool.Default().FindFileByName(pin.proto_name)
    except KeyError:
        pass
    else:
        raise AssertionError(f"{pin.proto_name} executed before race check")
""",
        root,
        mutation,
    )


def test_execution_failure_cleans_modules_and_permanently_poisons_loader(
    tmp_path: Path,
) -> None:
    root = _copy_bindings(tmp_path / "bindings")
    _run_subprocess(
        """
import builtins
import pathlib
import sys
from google.protobuf import descriptor_pool, symbol_database
import tools.nvbit_tracer.proto_binding_loader as loader
loader._BINDING_ROOT = pathlib.Path(sys.argv[1])
public_classes = symbol_database.SymbolDatabase._classes
public_snapshot = dict(public_classes)
real_exec = builtins.exec
count = 0
def execute_first_then_fail(code, globals_dict, locals_dict):
    global count
    real_exec(code, globals_dict, locals_dict)
    count += 1
    if count == 1:
        raise RuntimeError("planted failure after first descriptor registration")
loader.exec = execute_first_then_fail
try:
    loader.load_dynamic_trace_bindings()
except loader.ProtoBindingIntegrityError:
    pass
else:
    raise AssertionError("planted execution failure was accepted")
assert count == 1
try:
    descriptor_pool.Default().FindFileByName("address.proto")
except KeyError:
    pass
else:
    raise AssertionError("failed transaction contaminated the default pool")
assert set(public_classes) == set(public_snapshot)
assert all(public_classes[key] is value for key, value in public_snapshot.items())
assert not any(name in sys.modules for name in loader._CANONICAL_MODULE_NAMES)
try:
    loader.load_dynamic_trace_bindings()
except loader.ProtoBindingLoaderPoisonedError as exc:
    assert "permanently poisoned" in str(exc)
else:
    raise AssertionError("poisoned loader retried irreversible execution")
""",
        root,
    )


def test_post_execution_source_mutation_cleans_and_poisons(tmp_path: Path) -> None:
    root = _copy_bindings(tmp_path / "bindings")
    _run_subprocess(
        """
import pathlib
import sys
import tools.nvbit_tracer.proto_binding_loader as loader
loader._BINDING_ROOT = pathlib.Path(sys.argv[1])
original_execute = loader._execute_compiled_modules
def execute_then_mutate(directory, sources, runtime):
    owned = original_execute(directory, sources, runtime)
    target = directory.path / "trace_pb2.py"
    payload = bytearray(target.read_bytes())
    payload[-1] ^= 1
    target.write_bytes(payload)
    return owned
loader._execute_compiled_modules = execute_then_mutate
try:
    loader.load_dynamic_trace_bindings()
except loader.ProtoBindingIntegrityError:
    pass
else:
    raise AssertionError("post-execution source mutation was accepted")
assert not any(name in sys.modules for name in loader._CANONICAL_MODULE_NAMES)
try:
    loader.load_dynamic_trace_bindings()
except loader.ProtoBindingLoaderPoisonedError:
    pass
else:
    raise AssertionError("loader retried after post-execution mutation")
""",
        root,
    )


def test_caller_integrity_taxonomies_remain_hard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.cache_predictor.l1_production_helper as l1_helper
    import tools.local_labels.extend_launch_parameters as launch_parameters
    import tools.nvbit_tracer.import_trace_contract as trace_contract
    import tools.nvbit_tracer.trace_admission as trace_admission
    import workloads.simbench_bw.nvbit_trace_summary_adapter as adapter

    def reject():
        raise ProtoBindingIntegrityError("planted tracked authority corruption")

    cases = (
        (
            trace_contract,
            trace_contract._import_pb_modules,
            trace_contract.TraceContractTrackedBindingsCorrupt,
        ),
        (
            trace_admission,
            trace_admission._load_authenticated_bindings,
            trace_contract.TraceContractTrackedBindingsCorrupt,
        ),
        (
            adapter,
            adapter._import_pb_modules,
            adapter.NvbitTrackedProtobufCorruption,
        ),
        (
            l1_helper,
            l1_helper._load_proto_bindings,
            l1_helper.L1ProductionHelperTrackedBindingsCorrupt,
        ),
        (
            launch_parameters,
            launch_parameters._load_trace_bindings,
            launch_parameters.LaunchParametersTrackedBindingsCorrupt,
        ),
    )
    for module, caller, expected in cases:
        with monkeypatch.context() as context:
            context.setattr(module, "load_dynamic_trace_bindings", reject)
            with pytest.raises(expected):
                caller()


def test_caller_runtime_taxonomies_remain_hard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.cache_predictor.l1_production_helper as l1_helper
    import tools.local_labels.extend_launch_parameters as launch_parameters
    import tools.nvbit_tracer.import_trace_contract as trace_contract
    import tools.nvbit_tracer.trace_admission as trace_admission
    import workloads.simbench_bw.nvbit_trace_summary_adapter as adapter

    def reject():
        raise ProtoBindingRuntimePrerequisiteError("planted runtime prerequisite")

    cases = (
        (
            trace_contract,
            trace_contract._import_pb_modules,
            trace_contract.TraceContractRuntimePrerequisiteMissing,
        ),
        (
            trace_admission,
            trace_admission._load_authenticated_bindings,
            trace_contract.TraceContractRuntimePrerequisiteMissing,
        ),
        (
            adapter,
            adapter._import_pb_modules,
            adapter.NvbitProtobufRuntimePrerequisiteMissing,
        ),
        (
            l1_helper,
            l1_helper._load_proto_bindings,
            l1_helper.L1ProductionHelperRuntimePrerequisiteMissing,
        ),
        (
            launch_parameters,
            launch_parameters._load_trace_bindings,
            launch_parameters.LaunchParametersRuntimePrerequisiteMissing,
        ),
    )
    for module, caller, expected in cases:
        with monkeypatch.context() as context:
            context.setattr(module, "load_dynamic_trace_bindings", reject)
            with pytest.raises(expected):
                caller()


def _production_python_files() -> list[Path]:
    # The two fixed roots contain every audited binding consumer and exclude the
    # prohibited conference/worktree trees by construction.  Never follow a
    # production-file symlink during the routing audit.
    roots = (REPO_ROOT / "tools", REPO_ROOT / "workloads" / "simbench_bw")
    files: list[Path] = []
    for root in roots:
        for path in root.rglob("*.py"):
            if path.is_symlink():
                raise AssertionError(f"production Python route is symlinked: {path}")
            relative = path.relative_to(REPO_ROOT)
            if path == Path(__file__).resolve():
                continue
            if path == REPO_ROOT / "tools/nvbit_tracer/proto_binding_loader.py":
                continue
            if "_pb_generated" in relative.parts:
                continue
            if "tests" in relative.parts or path.name.startswith("test_"):
                continue
            files.append(path)
    return sorted(set(files))


def _attribute_name(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _expression_is_generated_path(node: ast.AST, tainted: set[str]) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            if "_pb_generated" in child.value:
                return True
        if isinstance(child, ast.Name) and child.id in tainted:
            return True
    return False


def _assigned_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            names.add(child.id)
    return names


def _targets_sys_path(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Attribute)
        and _attribute_name(child) == "sys.path"
        for child in ast.walk(node)
    )


def _generated_import_failures(path: Path, tree: ast.AST) -> list[str]:
    failures: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if "_pb_generated" in parts or parts[-1] in CANONICAL_MODULES:
                    failures.append(
                        f"{path}:{node.lineno}: generated import {alias.name}"
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            module_parts = module.split(".") if module else []
            for alias in node.names:
                imported_parts = alias.name.split(".")
                if (
                    "_pb_generated" in module_parts
                    or "_pb_generated" in imported_parts
                    or (module_parts and module_parts[-1] in CANONICAL_MODULES)
                    or imported_parts[-1] in CANONICAL_MODULES
                ):
                    target = f"{module}.{alias.name}" if module else alias.name
                    failures.append(
                        f"{path}:{node.lineno}: generated import-from {target}"
                    )
    return failures


@pytest.mark.parametrize(
    "source",
    (
        "from workloads.simbench_bw._pb_generated import trace_pb2",
        "from workloads.simbench_bw._pb_generated import trace_pb2 as planted",
        "from workloads.simbench_bw import _pb_generated",
        "import workloads.simbench_bw._pb_generated.trace_pb2 as planted",
    ),
)
def test_binding_route_guard_rejects_package_form_generated_imports(
    tmp_path: Path,
    source: str,
) -> None:
    path = tmp_path / "planted_generated_import.py"
    path.write_text(source, encoding="utf-8")
    failures = _binding_route_failures([path])
    assert failures
    assert any("generated import" in failure for failure in failures)


def test_binding_route_guard_allows_central_loader_import(tmp_path: Path) -> None:
    path = tmp_path / "authenticated_consumer.py"
    path.write_text(
        "from tools.nvbit_tracer.proto_binding_loader "
        "import load_dynamic_trace_bindings",
        encoding="utf-8",
    )
    assert _binding_route_failures([path]) == []


def _binding_route_failures(paths: list[Path]) -> list[str]:
    failures: list[str] = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=os.fspath(path))
        failures.extend(_generated_import_failures(path, tree))
        tainted: set[str] = set()
        assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
        ]
        changed = True
        while changed:
            changed = False
            for assignment in assignments:
                value = getattr(assignment, "value", None)
                if value is None or not _expression_is_generated_path(value, tainted):
                    continue
                targets: list[ast.AST]
                if isinstance(assignment, ast.Assign):
                    targets = assignment.targets
                else:
                    targets = [assignment.target]
                for target in targets:
                    for name in _assigned_names(target):
                        if name not in tainted:
                            tainted.add(name)
                            changed = True

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                function = _attribute_name(node.func)
                call_strings = tuple(
                    child.value
                    for child in ast.walk(node)
                    if isinstance(child, ast.Constant)
                    and isinstance(child.value, str)
                )
                python_generation_marker = any(
                    "--python_out" in value
                    or "grpc_tools" in value
                    or "nvbit-pb-python" in value
                    for value in call_strings
                )
                protoc_marker = any(
                    value == "protoc" or value.endswith("/protoc")
                    for value in call_strings
                )
                if python_generation_marker or (
                    protoc_marker and path != AUTHORIZED_CPP_BINDING_BUILDER
                ):
                    failures.append(
                        f"{path}:{node.lineno}: binding generation fallback"
                    )
                if (
                    function.startswith("sys.path.")
                    or function == "site.addsitedir"
                ) and any(
                    _expression_is_generated_path(argument, tainted)
                    for argument in (*node.args, *[kw.value for kw in node.keywords])
                ):
                    failures.append(
                        f"{path}:{node.lineno}: generated-directory sys.path mutation"
                    )
                if function.endswith(
                    (
                        "spec_from_file_location",
                        "SourceFileLoader",
                        "SourcelessFileLoader",
                        "ExtensionFileLoader",
                    )
                ) and any(
                    _expression_is_generated_path(argument, tainted)
                    for argument in (*node.args, *[kw.value for kw in node.keywords])
                ):
                    failures.append(
                        f"{path}:{node.lineno}: direct generated-file loader route"
                    )
                if function.startswith("sys.modules.") and node.args:
                    first = node.args[0]
                    if (
                        isinstance(first, ast.Constant)
                        and isinstance(first.value, str)
                        and first.value in CANONICAL_MODULES
                    ):
                        failures.append(
                            f"{path}:{node.lineno}: canonical sys.modules mutation"
                        )
                if function in {"__import__", "importlib.import_module"}:
                    for argument in node.args[:1]:
                        if isinstance(argument, ast.Constant) and isinstance(
                            argument.value, str
                        ) and argument.value.rsplit(".", 1)[-1] in CANONICAL_MODULES:
                            failures.append(
                                f"{path}:{node.lineno}: dynamic generated import "
                                f"{argument.value}"
                            )
            elif isinstance(node, ast.Subscript):
                if _attribute_name(node.value) == "sys.modules" and isinstance(
                    node.slice, ast.Constant
                ) and node.slice.value in CANONICAL_MODULES:
                    failures.append(
                        f"{path}:{node.lineno}: direct canonical sys.modules access"
                    )
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and node.value is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(_targets_sys_path(target) for target in targets) and (
                    _expression_is_generated_path(node.value, tainted)
                ):
                    failures.append(
                        f"{path}:{node.lineno}: generated-directory sys.path assignment"
                    )
                values = {
                    child.value
                    for child in ast.walk(node.value)
                    if isinstance(child, ast.Constant)
                    and isinstance(child.value, str)
                    and child.value in CANONICAL_FILENAMES
                }
                if len(values) >= 2:
                    failures.append(
                        f"{path}:{node.lineno}: duplicate generated binding table"
                    )
    return failures


def test_active_production_tree_has_one_binding_import_route() -> None:
    """AST-discover all production Python, rather than curating consumers."""
    assert _binding_route_failures(_production_python_files()) == []
