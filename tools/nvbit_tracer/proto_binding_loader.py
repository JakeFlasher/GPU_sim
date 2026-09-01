"""Authenticated loader for the tracked dynamic-trace Python bindings.

All nine tracked ``*_pb2.py`` authorities are authenticated and compiled before
any generated code executes. Execution uses private modules, a private descriptor
pool, immutable runtime captures, and no canonical ``sys.modules`` publication.
"""

from __future__ import annotations

import _imp
import builtins
import enum
import errno
import hashlib
import importlib.machinery
import os
import stat
import sys
import threading
import types
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, NoReturn


EXPECTED_PROTOBUF_RUNTIME_VERSION = "7.34.1"
_EXPECTED_PROTOBUF_RUNTIME_COMPONENTS = (7, 34, 1, "")

# Import once while this module initializes. The public load invocation validates
# only retained in-memory objects and performs no package/path metadata discovery.
_RUNTIME_IMPORT_FAILURE: BaseException | None = None
_IMPORTED_RUNTIME: dict[str, Any] | None
try:
    import google.protobuf as _protobuf_module
    import google.protobuf.descriptor as _descriptor_module
    import google.protobuf.descriptor_pool as _descriptor_pool_module
    import google.protobuf.internal as _protobuf_internal_module
    import google.protobuf.internal.api_implementation as _api_implementation_module
    import google.protobuf.internal.builder as _builder_module
    import google.protobuf.internal.enum_type_wrapper as _enum_wrapper_module
    import google.protobuf.message as _message_module
    import google.protobuf.message_factory as _message_factory_module
    import google.protobuf.reflection as _reflection_module
    import google.protobuf.runtime_version as _runtime_version_module
    import google.protobuf.symbol_database as _symbol_database_module
except Exception as exc:  # pragma: no cover - exercised in isolated missing-runtime tests
    _RUNTIME_IMPORT_FAILURE = exc
    _IMPORTED_RUNTIME = None
else:
    _IMPORTED_RUNTIME = {
        "protobuf": _protobuf_module,
        "descriptor": _descriptor_module,
        "descriptor_pool": _descriptor_pool_module,
        "protobuf_internal": _protobuf_internal_module,
        "api_implementation": _api_implementation_module,
        "builder": _builder_module,
        "enum_wrapper": _enum_wrapper_module,
        "message": _message_module,
        "message_factory": _message_factory_module,
        "reflection": _reflection_module,
        "runtime_version": _runtime_version_module,
        "symbol_database": _symbol_database_module,
    }

_ORIGINAL_OS_CLOSE = os.close


class ProtoBindingError(RuntimeError):
    """Base class for hard dynamic-trace binding dependency failures."""


class ProtoBindingIntegrityError(ProtoBindingError):
    """The tracked binding authority or private execution state is contaminated."""


class ProtoBindingRuntimePrerequisiteError(ProtoBindingError):
    """The exact project-pinned Python protobuf runtime is unavailable."""


class ProtoBindingLoaderPoisonedError(ProtoBindingIntegrityError):
    """A prior generated-code execution failed after irreversible side effects."""


@dataclass(frozen=True, slots=True)
class BindingPin:
    module_name: str
    filename: str
    proto_name: str
    size_bytes: int
    sha256: str
    dependencies: tuple[str, ...] = ()


# Literal authority pins in dependency-safe execution order. Never derive these
# expected values from the checkout or regenerate in the canonical worktree.
BINDING_PINS: tuple[BindingPin, ...] = (
    BindingPin(
        "address_pb2",
        "address_pb2.py",
        "address.proto",
        2042,
        "3a22f1db336cac6df4451d8608fb9d3bb81d806840b78e529edcb8c19340cf73",
    ),
    BindingPin(
        "dim3d_pb2",
        "dim3d_pb2.py",
        "dim3d.proto",
        1309,
        "6d933c129b3e21c81c60d4d6c2f9093b9c28d1c07ce085954546e01f3447f8d9",
    ),
    BindingPin(
        "instruction_pb2",
        "instruction_pb2.py",
        "instruction.proto",
        4270,
        "b43918eff894e2a87f7dd64bc7d1344fc12f300b92534999d0aa100228daa81f",
        ("address_pb2",),
    ),
    BindingPin(
        "kernel_pb2",
        "kernel_pb2.py",
        "kernel.proto",
        2024,
        "d84c72e4217dd3efe535e04ffb6549137cfdc79e384e07db8bb5fad6e29cf6c0",
        ("dim3d_pb2",),
    ),
    BindingPin(
        "cuda_stream_pb2",
        "cuda_stream_pb2.py",
        "cuda_stream.proto",
        1455,
        "7fdcab45b3a9206aba4c24b04fb4359196010dda30f9b5891179a11ce90ab198",
        ("kernel_pb2",),
    ),
    BindingPin(
        "gpu_device_pb2",
        "gpu_device_pb2.py",
        "gpu_device.proto",
        1826,
        "339a921d00e1146472cefad9cf9186a6d5d415053b461826c8e5c91fcfbf2b31",
        ("cuda_stream_pb2",),
    ),
    BindingPin(
        "warp_pb2",
        "warp_pb2.py",
        "warp.proto",
        1377,
        "842afd34bc76ac5245debbf5e4d389c9a9f9762b4c6c60b23a51dd7b9e05c967",
        ("instruction_pb2",),
    ),
    BindingPin(
        "threadblock_pb2",
        "threadblock_pb2.py",
        "threadblock.proto",
        1885,
        "b3e95c0fe358864226bcb1933cad6b922d4cc30293babac4e2cc872aa1d3da8b",
        ("warp_pb2", "dim3d_pb2"),
    ),
    BindingPin(
        "trace_pb2",
        "trace_pb2.py",
        "trace.proto",
        2187,
        "4e479bc1271a1720733577a08a23bbbdd2ad8b700809ba26d5f0ab047868487b",
        ("gpu_device_pb2", "instruction_pb2"),
    ),
)

_PIN_BY_MODULE: Mapping[str, BindingPin] = types.MappingProxyType(
    {pin.module_name: pin for pin in BINDING_PINS}
)
_CANONICAL_MODULE_NAMES = tuple(pin.module_name for pin in BINDING_PINS)
_CANONICAL_FILENAMES = frozenset(pin.filename for pin in BINDING_PINS)

_LOADER_PATH = Path(os.path.abspath(os.fspath(Path(__file__))))
_REPO_ROOT = _LOADER_PATH.parents[2]
_BINDING_ROOT = _REPO_ROOT / "workloads" / "simbench_bw" / "_pb_generated"
_RESTORE_GUIDANCE = (
    "restore byte-exact committed content with `git restore --source=HEAD "
    "--worktree -- workloads/simbench_bw/_pb_generated`; do not regenerate "
    "bindings in the canonical checkout"
)


@dataclass(frozen=True, slots=True)
class _ReadOnlyModuleSpec:
    name: str
    loader: Any
    origin: str
    has_location: bool = False

    @property
    def parent(self) -> str:
        return ""


class _AuthenticatedBindingLoader:
    __slots__ = ()

    def create_module(self, _spec: Any) -> NoReturn:
        raise ImportError("authenticated protobuf capabilities cannot be recreated")

    def exec_module(self, _module: Any) -> NoReturn:
        raise ImportError("authenticated protobuf capabilities cannot be reexecuted")

    def __repr__(self) -> str:
        return "<authenticated-dynamic-trace-binding-loader>"


_OWNED_LOADER = _AuthenticatedBindingLoader()


class _FrozenNamespace:
    __slots__ = ("_label", "_values")

    def __init__(self, label: str, values: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_label", label)
        object.__setattr__(self, "_values", types.MappingProxyType(dict(values)))

    def __getattribute__(self, name: str) -> Any:
        if name in {"_label", "_values", "__class__"}:
            return object.__getattribute__(self, name)
        values = object.__getattribute__(self, "_values")
        if name == "__dict__":
            return values
        if name in values:
            return values[name]
        return object.__getattribute__(self, name)

    def __setattr__(self, _name: str, _value: Any) -> NoReturn:
        raise AttributeError("retained runtime capability is read-only")

    def __delattr__(self, _name: str) -> NoReturn:
        raise AttributeError("retained runtime capability is read-only")

    def __repr__(self) -> str:
        return f"<retained-runtime-capability {self._label}>"


class _FrozenCallable:
    __slots__ = ("_callable", "_label")

    def __init__(self, label: str, function: Any) -> None:
        object.__setattr__(self, "_label", label)
        object.__setattr__(self, "_callable", function)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return object.__getattribute__(self, "_callable")(*args, **kwargs)

    def __setattr__(self, _name: str, _value: Any) -> NoReturn:
        raise AttributeError("retained runtime callable is read-only")

    def __delattr__(self, _name: str) -> NoReturn:
        raise AttributeError("retained runtime callable is read-only")


class _ConstantFactory:
    __slots__ = ("_label", "_value")

    def __init__(self, label: str, value: Any) -> None:
        object.__setattr__(self, "_label", label)
        object.__setattr__(self, "_value", value)

    def __call__(self) -> Any:
        return object.__getattribute__(self, "_value")

    def __setattr__(self, _name: str, _value: Any) -> NoReturn:
        raise AttributeError("private runtime factory is read-only")

    def __delattr__(self, _name: str) -> NoReturn:
        raise AttributeError("private runtime factory is read-only")


class _PinnedRuntimeValidator:
    __slots__ = ("_domain_public", "_proto_names")

    def __init__(
        self, domain_public: Any, proto_names: frozenset[str]
    ) -> None:
        object.__setattr__(self, "_domain_public", domain_public)
        object.__setattr__(self, "_proto_names", proto_names)

    def __call__(
        self,
        domain: Any,
        major: int,
        minor: int,
        patch: int,
        suffix: str,
        location: str,
    ) -> None:
        expected = _EXPECTED_PROTOBUF_RUNTIME_COMPONENTS
        if (
            domain is not object.__getattribute__(self, "_domain_public")
            or (type(major), type(minor), type(patch), type(suffix), type(location))
            != (int, int, int, str, str)
            or (major, minor, patch, suffix) != expected
            or location not in object.__getattribute__(self, "_proto_names")
        ):
            raise ProtoBindingRuntimePrerequisiteError(
                "generated authority requested an unpinned protobuf runtime contract"
            )

    def __setattr__(self, _name: str, _value: Any) -> NoReturn:
        raise AttributeError("pinned runtime validator is read-only")

    def __delattr__(self, _name: str) -> NoReturn:
        raise AttributeError("pinned runtime validator is read-only")


class _PrivateSymbolDatabase:
    __slots__ = ("_classes", "pool")

    def __init__(self, pool: Any) -> None:
        self.pool = pool
        self._classes: dict[Any, type] = {}

    def RegisterMessage(self, message: type) -> type:
        descriptor = message.DESCRIPTOR
        self._classes[descriptor] = message
        return message


class _IsolatedMessageFactory:
    __slots__ = ("_get_message_class", "_symbol_database", "pool")

    def __init__(
        self,
        pool: Any,
        get_message_class: Any,
        symbol_database: _PrivateSymbolDatabase,
    ) -> None:
        object.__setattr__(self, "pool", pool)
        object.__setattr__(self, "_get_message_class", get_message_class)
        object.__setattr__(self, "_symbol_database", symbol_database)

    def build_generated_class(
        self, module_name: str, descriptor: Any, qualname: str
    ) -> type:
        if getattr(descriptor.file, "pool", None) is not self.pool:
            raise _integrity(
                f"message descriptor {descriptor.full_name} is not factory-owned"
            )
        message_class = self._get_message_class(descriptor)
        if (
            not isinstance(message_class, type)
            or getattr(message_class, "DESCRIPTOR", None) is not descriptor
            or getattr(message_class, "__name__", None) != descriptor.name
        ):
            raise _integrity(
                f"message factory returned incompatible class for {descriptor.full_name}"
            )
        message_class.__module__ = module_name
        message_class.__qualname__ = qualname
        for nested_descriptor in descriptor.nested_types:
            nested_qualname = f"{qualname}.{nested_descriptor.name}"
            nested_class = self.build_generated_class(
                module_name, nested_descriptor, nested_qualname
            )
            if getattr(message_class, nested_descriptor.name, None) is not nested_class:
                raise _integrity(
                    f"message factory nested class mismatch for "
                    f"{nested_descriptor.full_name}"
                )
        self._symbol_database.RegisterMessage(message_class)
        return message_class

    def __setattr__(self, _name: str, _value: Any) -> NoReturn:
        raise AttributeError("isolated message factory is read-only")

    def __delattr__(self, _name: str) -> NoReturn:
        raise AttributeError("isolated message factory is read-only")


class _PrivateEnumTypeWrapper:
    """Loader-owned enum capability with no ambient protobuf class dispatch."""

    __slots__ = ("_descriptor", "_enum_type")
    ValueType = int

    def __init__(self, enum_type: Any) -> None:
        object.__setattr__(self, "_enum_type", enum_type)
        object.__setattr__(self, "_descriptor", enum_type)

    @property
    def DESCRIPTOR(self) -> Any:
        return object.__getattribute__(self, "_descriptor")

    def Name(self, number: int) -> str:
        enum_type = object.__getattribute__(self, "_enum_type")
        try:
            return enum_type.values_by_number[number].name
        except KeyError:
            if not isinstance(number, int):
                raise TypeError(
                    f"Enum value for {enum_type.name} must be an int, but got "
                    f"{type(number)} {number!r}."
                ) from None
            raise ValueError(
                f"Enum {enum_type.name} has no name defined for value {number!r}"
            ) from None

    def Value(self, name: str) -> int:
        enum_type = object.__getattribute__(self, "_enum_type")
        try:
            return enum_type.values_by_name[name].number
        except KeyError:
            raise ValueError(
                f"Enum {enum_type.name} has no value defined for name {name!r}"
            ) from None

    def keys(self) -> list[str]:
        return [
            value.name
            for value in object.__getattribute__(self, "_enum_type").values
        ]

    def values(self) -> list[int]:
        return [
            value.number
            for value in object.__getattribute__(self, "_enum_type").values
        ]

    def items(self) -> list[tuple[str, int]]:
        return [
            (value.name, value.number)
            for value in object.__getattribute__(self, "_enum_type").values
        ]

    def __getattr__(self, name: str) -> int:
        enum_type = object.__getattribute__(self, "_enum_type")
        try:
            return enum_type.values_by_name[name].number
        except KeyError:
            raise AttributeError(
                f"Enum {enum_type.name} has no value defined for name {name!r}"
            ) from None

    def __or__(self, other: Any) -> Any:
        return type(self) | other

    def __setattr__(self, _name: str, _value: Any) -> NoReturn:
        raise AttributeError("private enum wrapper is read-only")

    def __delattr__(self, _name: str) -> NoReturn:
        raise AttributeError("private enum wrapper is read-only")


class _PrivateGeneratedBuilder:
    __slots__ = ("_message_factory",)

    def __init__(self, message_factory: _IsolatedMessageFactory) -> None:
        object.__setattr__(self, "_message_factory", message_factory)

    def _build_nested_descriptors(
        self, module: dict[str, Any], descriptor: Any, prefix: str
    ) -> None:
        for name, nested_descriptor in descriptor.nested_types_by_name.items():
            module_name = f"{prefix}{name.upper()}"
            module[module_name] = nested_descriptor
            self._build_nested_descriptors(
                module, nested_descriptor, f"{module_name}_"
            )
        for enum_descriptor in descriptor.enum_types:
            module[f"{prefix}{enum_descriptor.name.upper()}"] = enum_descriptor

    def BuildMessageAndEnumDescriptors(
        self, file_descriptor: Any, module: dict[str, Any]
    ) -> None:
        for name, message_descriptor in file_descriptor.message_types_by_name.items():
            module_name = f"_{name.upper()}"
            module[module_name] = message_descriptor
            self._build_nested_descriptors(
                module, message_descriptor, f"{module_name}_"
            )

    def BuildTopDescriptorsAndMessages(
        self, file_descriptor: Any, module_name: str, module: dict[str, Any]
    ) -> None:
        for name, enum_descriptor in file_descriptor.enum_types_by_name.items():
            module[f"_{name.upper()}"] = enum_descriptor
            module[name] = _PrivateEnumTypeWrapper(enum_descriptor)
            for enum_value in enum_descriptor.values:
                module[enum_value.name] = enum_value.number
        for name, extension_descriptor in file_descriptor.extensions_by_name.items():
            module[f"{name.upper()}_FIELD_NUMBER"] = extension_descriptor.number
            module[name] = extension_descriptor
        for name, service_descriptor in file_descriptor.services_by_name.items():
            module[f"_{name.upper()}"] = service_descriptor
        for name, message_descriptor in file_descriptor.message_types_by_name.items():
            module[name] = self._message_factory.build_generated_class(
                module_name, message_descriptor, message_descriptor.name
            )

    def __setattr__(self, _name: str, _value: Any) -> NoReturn:
        raise AttributeError("private generated builder is read-only")

    def __delattr__(self, _name: str) -> NoReturn:
        raise AttributeError("private generated builder is read-only")


class BindingModuleView:
    """Sealed, read-only capability for one authenticated generated namespace."""

    __slots__ = ("_module_name", "_token", "_values")

    def __init__(self, module_name: str, token: object, values: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_module_name", module_name)
        object.__setattr__(self, "_token", token)
        object.__setattr__(self, "_values", types.MappingProxyType(dict(values)))

    def __getattribute__(self, name: str) -> Any:
        if name in {"_module_name", "_token", "_values", "__class__"}:
            return object.__getattribute__(self, name)
        _validate_bundle_token(object.__getattribute__(self, "_token"))
        values = object.__getattribute__(self, "_values")
        if name == "__dict__":
            return values
        if name in values:
            return values[name]
        return object.__getattribute__(self, name)

    def __setattr__(self, _name: str, _value: Any) -> NoReturn:
        raise AttributeError("authenticated binding capability is read-only")

    def __delattr__(self, _name: str) -> NoReturn:
        raise AttributeError("authenticated binding capability is read-only")

    def __dir__(self) -> list[str]:
        return sorted(object.__getattribute__(self, "_values"))

    def __repr__(self) -> str:
        return f"<authenticated-binding-capability {self._module_name}>"


class DynamicTraceBindings:
    __slots__ = ("_token", "_views")

    def __init__(
        self, token: object, views: tuple[BindingModuleView, ...]
    ) -> None:
        object.__setattr__(self, "_token", token)
        object.__setattr__(self, "_views", views)

    def __setattr__(self, _name: str, _value: Any) -> NoReturn:
        raise FrozenInstanceError("cannot assign to immutable binding bundle")

    def __delattr__(self, _name: str) -> NoReturn:
        raise FrozenInstanceError("cannot delete from immutable binding bundle")

    def _at(self, index: int) -> BindingModuleView:
        _validate_bundle_token(self._token)
        return self._views[index]

    @property
    def address_pb2(self) -> BindingModuleView:
        return self._at(0)

    @property
    def dim3d_pb2(self) -> BindingModuleView:
        return self._at(1)

    @property
    def instruction_pb2(self) -> BindingModuleView:
        return self._at(2)

    @property
    def kernel_pb2(self) -> BindingModuleView:
        return self._at(3)

    @property
    def cuda_stream_pb2(self) -> BindingModuleView:
        return self._at(4)

    @property
    def gpu_device_pb2(self) -> BindingModuleView:
        return self._at(5)

    @property
    def warp_pb2(self) -> BindingModuleView:
        return self._at(6)

    @property
    def threadblock_pb2(self) -> BindingModuleView:
        return self._at(7)

    @property
    def trace_pb2(self) -> BindingModuleView:
        return self._at(8)

    def by_name(self, module_name: str) -> BindingModuleView:
        if module_name not in _PIN_BY_MODULE:
            raise KeyError(module_name)
        return self._at(_CANONICAL_MODULE_NAMES.index(module_name))

    @property
    def modules(self) -> tuple[BindingModuleView, ...]:
        _validate_bundle_token(self._token)
        return self._views


# File metadata excludes atime because an authenticated read may update it.
_FileIdentity = tuple[int, int, int, int, int, int, int]
_DirectoryNode = tuple[int, int, int]


@dataclass(slots=True)
class _RetainedSource:
    pin: BindingPin
    fd: int
    identity: _FileIdentity
    payload: bytes
    code: types.CodeType | None = None


@dataclass(slots=True)
class _PinnedBindingDirectory:
    path: Path
    fd: int
    chain: tuple[tuple[str, _DirectoryNode], ...]
    root_identity: _FileIdentity


@dataclass(frozen=True, slots=True)
class _ModuleCapture:
    name: str
    module: Any
    spec: Any
    loader: Any
    spec_state: tuple[Any, ...]
    loader_state: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _FunctionCapture:
    label: str
    owner: Any
    attribute: str
    function: Any
    code: Any
    globals_dict: Any
    defaults: Any
    kwdefaults: Any
    closure: Any
    fingerprint: str
    global_bindings: tuple[tuple[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _ClassCapture:
    label: str
    owner: Any
    attribute: str
    value: Any
    exact_type: type
    module_name: str
    class_name: str


@dataclass(frozen=True, slots=True)
class _RuntimeAuthority:
    modules: tuple[_ModuleCapture, ...]
    functions: tuple[_FunctionCapture, ...]
    classes: tuple[_ClassCapture, ...]
    protobuf: Any
    protobuf_internal: Any
    api_implementation: Any
    descriptor: Any
    descriptor_pool: Any
    runtime_version: Any
    symbol_database: Any
    builder: Any
    enum_wrapper: Any
    reflection: Any
    message: Any
    message_factory: Any
    default_pool: Any
    default_symbol_database: Any
    pool_runtime_type: type
    domain: Any
    domain_public: Any
    version_error: type
    descriptor_use_c: bool
    descriptor_pool_class: type
    symbol_database_class: type
    enum_wrapper_class: type
    generated_message_meta: type
    message_class: type


@dataclass(slots=True)
class _ExecutionMarker:
    started: bool = False


@dataclass(frozen=True, slots=True)
class _ExecutionRuntime:
    authority: _RuntimeAuthority
    pool: Any
    symbol_database: _PrivateSymbolDatabase
    message_factory: _IsolatedMessageFactory
    protobuf_proxy: _FrozenNamespace
    internal_proxy: _FrozenNamespace
    marker: _ExecutionMarker


@dataclass(frozen=True, slots=True)
class _ObjectSnapshot:
    value: Any
    namespace: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _ModuleSnapshot:
    pin: BindingPin
    module: types.ModuleType
    namespace: Mapping[str, Any]
    spec_state: tuple[Any, ...]
    descriptor: Any
    classes: tuple[_ObjectSnapshot, ...]
    objects: tuple[_ObjectSnapshot, ...]
    view: BindingModuleView
    view_values: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _PrivateBindingState:
    token: object
    modules: Mapping[str, types.ModuleType]
    snapshots: tuple[_ModuleSnapshot, ...]
    views: Mapping[str, BindingModuleView]
    runtime: _ExecutionRuntime
    symbol_classes: Mapping[Any, type]


_MISSING = object()


def _feed_code_fingerprint(digest: Any, value: Any) -> None:
    if isinstance(value, types.CodeType):
        digest.update(b"C")
        for number in (
            value.co_argcount,
            value.co_posonlyargcount,
            value.co_kwonlyargcount,
            value.co_nlocals,
            value.co_stacksize,
            value.co_flags,
        ):
            digest.update(number.to_bytes(8, "big", signed=True))
        for payload in (value.co_code, value.co_exceptiontable):
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        for sequence in (
            value.co_names,
            value.co_varnames,
            value.co_freevars,
            value.co_cellvars,
            value.co_consts,
        ):
            _feed_code_fingerprint(digest, tuple(sequence))
    elif value is None:
        digest.update(b"N")
    elif value is Ellipsis:
        digest.update(b"E")
    elif isinstance(value, tuple):
        digest.update(b"T" + len(value).to_bytes(8, "big"))
        for item in value:
            _feed_code_fingerprint(digest, item)
    elif isinstance(value, frozenset):
        children: list[bytes] = []
        for item in value:
            child = hashlib.sha256()
            _feed_code_fingerprint(child, item)
            children.append(child.digest())
        digest.update(b"F" + len(children).to_bytes(8, "big"))
        for child_digest in sorted(children):
            digest.update(child_digest)
    elif type(value) in (bool, int, float, complex, str, bytes):
        type_name = type(value).__name__.encode("ascii")
        payload = value if isinstance(value, bytes) else repr(value).encode("utf-8")
        digest.update(b"V" + len(type_name).to_bytes(2, "big") + type_name)
        digest.update(len(payload).to_bytes(8, "big") + payload)
    else:
        raise TypeError(f"unsupported code constant {type(value).__name__}")


def _code_fingerprint(code: types.CodeType) -> str:
    digest = hashlib.sha256()
    _feed_code_fingerprint(digest, code)
    return digest.hexdigest()


_ENUM_WRAPPER_METHOD_NAMES = (
    "__init__",
    "Name",
    "Value",
    "keys",
    "values",
    "items",
    "__getattr__",
    "__or__",
)
_ENUM_WRAPPER_NAMESPACE_NAMES = frozenset(
    {
        "__module__",
        "__doc__",
        "DESCRIPTOR",
        "ValueType",
        *_ENUM_WRAPPER_METHOD_NAMES,
        "__dict__",
        "__weakref__",
    }
)
_ENUM_WRAPPER_DOC = "A utility for finding the names of enum values."

_FUNCTION_PINS = {
    "descriptor_pool.Default": "e004443293e7a69fe433a64b03989b15db8c539942a2d6592bab64994a9ae82c",
    "runtime_version.ValidateProtobufRuntimeVersion": "fbd06c76d975f7c1012af0f7ddf3c45b08913949e7a068c61375cce9225d8c13",
    "runtime_version._ReportVersionError": "bacac9504f68114abfa1c0c42cc7873b4aa8016209e7eeeb97cca11802a3a9af",
    "symbol_database.Default": "c86ea973f5b76db479844be0e1d4257722408f2e8b91a1ae7494a2da9ba85a40",
    "builder.BuildMessageAndEnumDescriptors": "2afb30006be0ce3d42adab45dbbd50833565fb9f782c522e1be6c7072cc7826d",
    "builder.BuildTopDescriptorsAndMessages": "59f391fcc20e2cd2bf7c8424ea6e97b152369e5199bb20f3d9e511bb22dbbc54",
    "builder._BuildNestedDescriptors": "4ef8b058d76f71ba60275127d9748f23baa962cd9f826802166edd00c63ae2c3",
    "builder._BuildMessage": "32fa9fdb2ec45a483d8ed9f069503c99af1167da31e7944de9d588867419a14e",
    "message_factory.GetMessageClass": "efd27444fb41f82ac19ce13a22b2f8c848e3e120f317f91e4330cf59e365b814",
    "message_factory._InternalCreateMessageClass": "de4fb5e479e320afa6866dabb94d89202d7273cf22ad39b93229758ee0b2b3ad",
    "EnumTypeWrapper.__init__": "37f452941845e5683a3c23a95d40862aac7132e8fbac6fc9cbe2bfefa93cb1b9",
    "EnumTypeWrapper.Name": "a7623424c80416801d5fdf838e88615431056b25e5368cdf33c104805f249a95",
    "EnumTypeWrapper.Value": "18478f259e7f4d588a2b1cf440f6262ac35bd8e7d5563e684897d31ed379d99c",
    "EnumTypeWrapper.keys": "459e6d30b225f11262255339e47f804309129971339e883ea37e0fa546f8f0a0",
    "EnumTypeWrapper.values": "7d64973148e55405a55f67ca9414ca75d02900632523f6e78c2aa002317515f9",
    "EnumTypeWrapper.items": "eb68ee31faf094e2ccff22807245d7ead8d966cc00631321c0ed82049743b0fc",
    "EnumTypeWrapper.__getattr__": "ed2ce042d147b3198a2cc6daad14fcad1bc7b3ce780e37135dc5a1f68c7e1a6a",
    "EnumTypeWrapper.__or__": "8269d89706b9cf63c6bbea003abc16ded0354f80ef1c6895e46b4b4330831b96",
}


def _spec_state(spec: Any) -> tuple[Any, ...]:
    locations = getattr(spec, "submodule_search_locations", None)
    normalized_locations = None if locations is None else tuple(locations)
    return (
        getattr(spec, "name", _MISSING),
        getattr(spec, "loader", _MISSING),
        getattr(spec, "origin", _MISSING),
        getattr(spec, "cached", _MISSING),
        bool(getattr(spec, "has_location", False)),
        normalized_locations,
    )


def _capture_module(name: str, module: Any) -> _ModuleCapture:
    spec = getattr(module, "__spec__", None)
    loader = getattr(module, "__loader__", None)
    try:
        loader_values = dict(vars(loader))
    except TypeError:
        loader_values = {}
    return _ModuleCapture(
        name=name,
        module=module,
        spec=spec,
        loader=loader,
        spec_state=_spec_state(spec),
        loader_state=types.MappingProxyType(loader_values),
    )


def _capture_function(
    label: str,
    owner: Any,
    attribute: str,
    global_names: tuple[str, ...] = (),
) -> _FunctionCapture:
    function = getattr(owner, attribute, _MISSING)
    globals_dict = getattr(function, "__globals__", _MISSING)
    bindings: list[tuple[str, Any]] = []
    for name in global_names:
        value = (
            globals_dict.get(name, _MISSING)
            if type(globals_dict) is dict
            else _MISSING
        )
        bindings.append((name, value))
    return _FunctionCapture(
        label=label,
        owner=owner,
        attribute=attribute,
        function=function,
        code=getattr(function, "__code__", _MISSING),
        globals_dict=globals_dict,
        defaults=getattr(function, "__defaults__", _MISSING),
        kwdefaults=getattr(function, "__kwdefaults__", _MISSING),
        closure=getattr(function, "__closure__", _MISSING),
        fingerprint=_FUNCTION_PINS[label],
        global_bindings=tuple(bindings),
    )


def _capture_class(label: str, owner: Any, attribute: str) -> _ClassCapture:
    value = getattr(owner, attribute, _MISSING)
    return _ClassCapture(
        label=label,
        owner=owner,
        attribute=attribute,
        value=value,
        exact_type=type(value),
        module_name=getattr(value, "__module__", ""),
        class_name=getattr(value, "__name__", ""),
    )


def _build_runtime_authority(imported: Mapping[str, Any]) -> _RuntimeAuthority:
    protobuf = imported["protobuf"]
    descriptor = imported["descriptor"]
    descriptor_pool = imported["descriptor_pool"]
    protobuf_internal = imported["protobuf_internal"]
    api_implementation = imported["api_implementation"]
    builder = imported["builder"]
    enum_wrapper = imported["enum_wrapper"]
    message = imported["message"]
    message_factory = imported["message_factory"]
    reflection = imported["reflection"]
    runtime_version = imported["runtime_version"]
    symbol_database = imported["symbol_database"]

    modules = tuple(
        _capture_module(name, module)
        for name, module in (
            ("google.protobuf", protobuf),
            ("google.protobuf.descriptor", descriptor),
            ("google.protobuf.descriptor_pool", descriptor_pool),
            ("google.protobuf.internal", protobuf_internal),
            ("google.protobuf.internal.api_implementation", api_implementation),
            ("google.protobuf.internal.builder", builder),
            ("google.protobuf.internal.enum_type_wrapper", enum_wrapper),
            ("google.protobuf.message", message),
            ("google.protobuf.message_factory", message_factory),
            ("google.protobuf.reflection", reflection),
            ("google.protobuf.runtime_version", runtime_version),
            ("google.protobuf.symbol_database", symbol_database),
        )
    )
    enum_wrapper_class = getattr(enum_wrapper, "EnumTypeWrapper", _MISSING)
    functions = (
        _capture_function(
            "descriptor_pool.Default", descriptor_pool, "Default", ("_DEFAULT",)
        ),
        _capture_function(
            "runtime_version.ValidateProtobufRuntimeVersion",
            runtime_version,
            "ValidateProtobufRuntimeVersion",
            (
                "os",
                "MAJOR",
                "MINOR",
                "PATCH",
                "SUFFIX",
                "VersionError",
                "DOMAIN",
                "_ReportVersionError",
            ),
        ),
        _capture_function(
            "runtime_version._ReportVersionError",
            runtime_version,
            "_ReportVersionError",
            ("VersionError",),
        ),
        _capture_function(
            "symbol_database.Default", symbol_database, "Default", ("_DEFAULT",)
        ),
        _capture_function(
            "builder.BuildMessageAndEnumDescriptors",
            builder,
            "BuildMessageAndEnumDescriptors",
            ("_BuildNestedDescriptors",),
        ),
        _capture_function(
            "builder.BuildTopDescriptorsAndMessages",
            builder,
            "BuildTopDescriptorsAndMessages",
            ("enum_type_wrapper", "_BuildMessage"),
        ),
        _capture_function(
            "builder._BuildNestedDescriptors",
            builder,
            "_BuildNestedDescriptors",
            ("_BuildNestedDescriptors",),
        ),
        _capture_function(
            "builder._BuildMessage",
            builder,
            "_BuildMessage",
            ("_BuildMessage", "_reflection", "_message", "_sym_db"),
        ),
        _capture_function(
            "message_factory.GetMessageClass",
            message_factory,
            "GetMessageClass",
            ("_InternalCreateMessageClass",),
        ),
        _capture_function(
            "message_factory._InternalCreateMessageClass",
            message_factory,
            "_InternalCreateMessageClass",
            (
                "_GENERATED_PROTOCOL_MESSAGE_TYPE",
                "message",
                "GetMessageClass",
                "api_implementation",
            ),
        ),
        *(
            _capture_function(
                f"EnumTypeWrapper.{name}", enum_wrapper_class, name
            )
            for name in _ENUM_WRAPPER_METHOD_NAMES
        ),
    )
    classes = (
        _capture_class("runtime_version.Domain", runtime_version, "Domain"),
        _capture_class("runtime_version.VersionError", runtime_version, "VersionError"),
        _capture_class("descriptor_pool.DescriptorPool", descriptor_pool, "DescriptorPool"),
        _capture_class("symbol_database.SymbolDatabase", symbol_database, "SymbolDatabase"),
        _capture_class("enum_type_wrapper.EnumTypeWrapper", enum_wrapper, "EnumTypeWrapper"),
        _capture_class(
            "reflection.GeneratedProtocolMessageType",
            reflection,
            "GeneratedProtocolMessageType",
        ),
        _capture_class("message.Message", message, "Message"),
    )
    default_pool = vars(descriptor_pool).get("_DEFAULT", _MISSING)
    default_symbol_database = vars(symbol_database).get("_DEFAULT", _MISSING)
    domain = getattr(runtime_version, "Domain", _MISSING)
    return _RuntimeAuthority(
        modules=modules,
        functions=functions,
        classes=classes,
        protobuf=protobuf,
        protobuf_internal=protobuf_internal,
        api_implementation=api_implementation,
        descriptor=descriptor,
        descriptor_pool=descriptor_pool,
        runtime_version=runtime_version,
        symbol_database=symbol_database,
        builder=builder,
        enum_wrapper=enum_wrapper,
        reflection=reflection,
        message=message,
        message_factory=message_factory,
        default_pool=default_pool,
        default_symbol_database=default_symbol_database,
        pool_runtime_type=type(default_pool),
        domain=domain,
        domain_public=getattr(domain, "PUBLIC", _MISSING),
        version_error=getattr(runtime_version, "VersionError", _MISSING),
        descriptor_use_c=getattr(descriptor, "_USE_C_DESCRIPTORS", _MISSING),
        descriptor_pool_class=getattr(descriptor_pool, "DescriptorPool", _MISSING),
        symbol_database_class=getattr(symbol_database, "SymbolDatabase", _MISSING),
        enum_wrapper_class=enum_wrapper_class,
        generated_message_meta=getattr(
            reflection, "GeneratedProtocolMessageType", _MISSING
        ),
        message_class=getattr(message, "Message", _MISSING),
    )


_RUNTIME_CAPTURE_FAILURE: BaseException | None = None
_CAPTURED_RUNTIME_AUTHORITY: _RuntimeAuthority | None = None
if _IMPORTED_RUNTIME is not None:
    try:
        _CAPTURED_RUNTIME_AUTHORITY = _build_runtime_authority(_IMPORTED_RUNTIME)
    except Exception as exc:  # pragma: no cover - hostile pre-import state
        _RUNTIME_CAPTURE_FAILURE = exc

_LOAD_LOCK = threading.RLock()
_loaded: DynamicTraceBindings | None = None
_private_state: _PrivateBindingState | None = None
_poisoned_reason: str | None = None


@contextmanager
def _held_import_locks() -> Iterator[None]:
    _imp.acquire_lock()
    try:
        with _LOAD_LOCK:
            yield
    finally:
        _imp.release_lock()


def _runtime_prerequisite(message: str) -> ProtoBindingRuntimePrerequisiteError:
    return ProtoBindingRuntimePrerequisiteError(
        f"protobuf runtime prerequisite {message}; install exactly "
        f"protobuf=={EXPECTED_PROTOBUF_RUNTIME_VERSION} from requirements.txt"
    )


def _same_typed_value(observed: Any, expected: Any) -> bool:
    return type(observed) is type(expected) and observed == expected


def _validate_module_capture(capture: _ModuleCapture) -> None:
    module = capture.module
    if type(module) is not types.ModuleType:
        raise _runtime_prerequisite(f"module {capture.name!r} is forged or subclassed")
    if sys.modules.get(capture.name) is not module:
        raise _runtime_prerequisite(f"module {capture.name!r} identity changed")
    if module.__name__ != capture.name:
        raise _runtime_prerequisite(f"module {capture.name!r} name changed")
    if (
        type(module.__spec__) is not importlib.machinery.ModuleSpec
        or module.__spec__ is not capture.spec
        or module.__loader__ is not capture.loader
        or capture.spec.loader is not capture.loader
        or _spec_state(capture.spec) != capture.spec_state
    ):
        raise _runtime_prerequisite(f"module {capture.name!r} spec/loader is forged")
    if type(capture.loader) is not type(module.__loader__):
        raise _runtime_prerequisite(f"module {capture.name!r} loader type changed")
    try:
        observed_loader_state = vars(capture.loader)
    except TypeError:
        observed_loader_state = {}
    if set(observed_loader_state) != set(capture.loader_state):
        raise _runtime_prerequisite(f"module {capture.name!r} loader state changed")
    for name, expected in capture.loader_state.items():
        if not _same_typed_value(observed_loader_state[name], expected):
            raise _runtime_prerequisite(
                f"module {capture.name!r} loader field {name!r} changed"
            )


def _validate_function_capture(capture: _FunctionCapture) -> None:
    function = capture.function
    if (
        type(function) is not types.FunctionType
        or getattr(capture.owner, capture.attribute, _MISSING) is not function
        or function.__code__ is not capture.code
        or function.__globals__ is not capture.globals_dict
        or function.__defaults__ is not capture.defaults
        or function.__kwdefaults__ is not capture.kwdefaults
        or function.__closure__ is not capture.closure
    ):
        raise _runtime_prerequisite(f"function {capture.label} identity changed")
    try:
        fingerprint = _code_fingerprint(function.__code__)
    except (TypeError, ValueError) as exc:
        raise _runtime_prerequisite(
            f"function {capture.label} code cannot be authenticated"
        ) from exc
    if fingerprint != capture.fingerprint:
        raise _runtime_prerequisite(f"function {capture.label} code fingerprint changed")
    for name, expected in capture.global_bindings:
        if function.__globals__.get(name, _MISSING) is not expected:
            raise _runtime_prerequisite(
                f"function {capture.label} global {name!r} changed"
            )


def _validate_enum_wrapper_class(
    runtime: _RuntimeAuthority,
    function_captures: Mapping[str, _FunctionCapture],
) -> None:
    wrapper_class = runtime.enum_wrapper_class
    if (
        type(wrapper_class) is not type
        or wrapper_class.__module__
        != "google.protobuf.internal.enum_type_wrapper"
        or wrapper_class.__name__ != "EnumTypeWrapper"
        or wrapper_class.__qualname__ != "EnumTypeWrapper"
        or wrapper_class.__bases__ != (object,)
        or wrapper_class.__mro__ != (wrapper_class, object)
        or wrapper_class.__new__ is not object.__new__
    ):
        raise _runtime_prerequisite("EnumTypeWrapper class authority changed")

    namespace = vars(wrapper_class)
    if frozenset(namespace) != _ENUM_WRAPPER_NAMESPACE_NAMES:
        raise _runtime_prerequisite("EnumTypeWrapper class namespace changed")
    if (
        type(namespace["__module__"]) is not str
        or namespace["__module__"]
        != "google.protobuf.internal.enum_type_wrapper"
        or type(namespace["__doc__"]) is not str
        or namespace["__doc__"] != _ENUM_WRAPPER_DOC
        or namespace["DESCRIPTOR"] is not None
        or namespace["ValueType"] is not int
    ):
        raise _runtime_prerequisite("EnumTypeWrapper class constants changed")

    for name in _ENUM_WRAPPER_METHOD_NAMES:
        capture = function_captures[f"EnumTypeWrapper.{name}"]
        function = namespace[name]
        if (
            function is not capture.function
            or type(function.__annotations__) is not dict
            or function.__annotations__
            or function.__defaults__ is not None
            or function.__kwdefaults__ is not None
        ):
            raise _runtime_prerequisite(
                f"EnumTypeWrapper class member {name!r} changed"
            )
    getattr_closure = namespace["__getattr__"].__closure__
    if (
        type(getattr_closure) is not tuple
        or len(getattr_closure) != 1
        or getattr_closure[0].cell_contents is not wrapper_class
    ):
        raise _runtime_prerequisite("EnumTypeWrapper __getattr__ closure changed")
    for name in _ENUM_WRAPPER_METHOD_NAMES:
        if name != "__getattr__" and namespace[name].__closure__ is not None:
            raise _runtime_prerequisite(
                f"EnumTypeWrapper class member {name!r} closure changed"
            )

    for name in ("__dict__", "__weakref__"):
        descriptor = namespace[name]
        if (
            type(descriptor) is not types.GetSetDescriptorType
            or descriptor.__objclass__ is not wrapper_class
            or descriptor.__name__ != name
        ):
            raise _runtime_prerequisite(
                f"EnumTypeWrapper class descriptor {name!r} changed"
            )


def _validate_runtime_authority(runtime: _RuntimeAuthority) -> None:
    version = getattr(runtime.protobuf, "__version__", None)
    if type(version) is not str or version != EXPECTED_PROTOBUF_RUNTIME_VERSION:
        raise _runtime_prerequisite(
            f"has version {version!r}; expected exactly "
            f"{EXPECTED_PROTOBUF_RUNTIME_VERSION!r}"
        )
    components = tuple(
        getattr(runtime.runtime_version, name, None)
        for name in ("MAJOR", "MINOR", "PATCH", "SUFFIX")
    )
    if (
        tuple(type(value) for value in components) != (int, int, int, str)
        or components != _EXPECTED_PROTOBUF_RUNTIME_COMPONENTS
    ):
        raise _runtime_prerequisite(
            f"has in-memory version components {components!r}; expected exactly "
            f"{_EXPECTED_PROTOBUF_RUNTIME_COMPONENTS!r}"
        )

    for capture in runtime.modules:
        _validate_module_capture(capture)
    for capture in runtime.functions:
        _validate_function_capture(capture)
    for capture in runtime.classes:
        observed = getattr(capture.owner, capture.attribute, _MISSING)
        if (
            observed is not capture.value
            or type(observed) is not capture.exact_type
            or getattr(observed, "__module__", "") != capture.module_name
            or getattr(observed, "__name__", "") != capture.class_name
        ):
            raise _runtime_prerequisite(f"class {capture.label} identity changed")

    function_captures = {capture.label: capture for capture in runtime.functions}
    _validate_enum_wrapper_class(runtime, function_captures)
    get_message_class = function_captures["message_factory.GetMessageClass"]
    create_message_class = function_captures[
        "message_factory._InternalCreateMessageClass"
    ]
    get_globals = get_message_class.function.__globals__
    create_globals = create_message_class.function.__globals__
    if (
        get_globals.get("_InternalCreateMessageClass", _MISSING)
        is not create_message_class.function
        or create_globals.get("_GENERATED_PROTOCOL_MESSAGE_TYPE", _MISSING)
        is not runtime.generated_message_meta
        or create_globals.get("message", _MISSING) is not runtime.message
        or create_globals.get("GetMessageClass", _MISSING)
        is not get_message_class.function
        or create_globals.get("api_implementation", _MISSING)
        is not runtime.api_implementation
    ):
        raise _runtime_prerequisite("message factory runtime bindings changed")

    package_bindings = (
        (runtime.protobuf, "descriptor", runtime.descriptor),
        (runtime.protobuf, "descriptor_pool", runtime.descriptor_pool),
        (runtime.protobuf, "message_factory", runtime.message_factory),
        (runtime.protobuf, "runtime_version", runtime.runtime_version),
        (runtime.protobuf, "symbol_database", runtime.symbol_database),
        (
            runtime.protobuf_internal,
            "api_implementation",
            runtime.api_implementation,
        ),
        (runtime.protobuf_internal, "builder", runtime.builder),
    )
    for owner, name, expected in package_bindings:
        if getattr(owner, name, _MISSING) is not expected:
            raise _runtime_prerequisite(f"retained package binding {name!r} changed")

    version = getattr(runtime.protobuf, "__version__", None)
    if type(version) is not str or version != EXPECTED_PROTOBUF_RUNTIME_VERSION:
        raise _runtime_prerequisite(
            f"has version {version!r}; expected exactly "
            f"{EXPECTED_PROTOBUF_RUNTIME_VERSION!r}"
        )
    components = tuple(
        getattr(runtime.runtime_version, name, None)
        for name in ("MAJOR", "MINOR", "PATCH", "SUFFIX")
    )
    if (
        tuple(type(value) for value in components) != (int, int, int, str)
        or components != _EXPECTED_PROTOBUF_RUNTIME_COMPONENTS
    ):
        raise _runtime_prerequisite(
            f"has in-memory version components {components!r}; expected exactly "
            f"{_EXPECTED_PROTOBUF_RUNTIME_COMPONENTS!r}"
        )
    if (
        type(runtime.domain) is not enum.EnumType
        or getattr(runtime.runtime_version, "DOMAIN", _MISSING)
        is not runtime.domain_public
        or tuple((member.name, member.value) for member in runtime.domain)
        != (("GOOGLE_INTERNAL", 1), ("PUBLIC", 2))
        or getattr(runtime.domain, "PUBLIC", _MISSING) is not runtime.domain_public
    ):
        raise _runtime_prerequisite("runtime version domain identity changed")
    if (
        type(runtime.descriptor_use_c) is not bool
        or runtime.descriptor_use_c is not True
        or getattr(runtime.descriptor, "_USE_C_DESCRIPTORS", _MISSING) is not True
    ):
        raise _runtime_prerequisite("descriptor implementation constant changed")
    if (
        vars(runtime.descriptor_pool).get("_DEFAULT", _MISSING)
        is not runtime.default_pool
        or vars(runtime.symbol_database).get("_DEFAULT", _MISSING)
        is not runtime.default_symbol_database
        or type(runtime.default_pool) is not runtime.pool_runtime_type
        or runtime.pool_runtime_type.__module__ != "google._upb._message"
        or runtime.pool_runtime_type.__name__ != "DescriptorPool"
    ):
        raise _runtime_prerequisite("default runtime objects changed")


def _load_runtime_authority() -> _RuntimeAuthority:
    if _RUNTIME_IMPORT_FAILURE is not None:
        raise _runtime_prerequisite("is missing") from _RUNTIME_IMPORT_FAILURE
    if _RUNTIME_CAPTURE_FAILURE is not None or _CAPTURED_RUNTIME_AUTHORITY is None:
        raise _runtime_prerequisite("could not be captured") from _RUNTIME_CAPTURE_FAILURE
    _validate_runtime_authority(_CAPTURED_RUNTIME_AUTHORITY)
    return _CAPTURED_RUNTIME_AUTHORITY


def _file_identity(value: os.stat_result) -> _FileIdentity:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _directory_node(value: os.stat_result) -> _DirectoryNode:
    return (int(value.st_dev), int(value.st_ino), int(value.st_mode))


def _integrity(message: str) -> ProtoBindingIntegrityError:
    return ProtoBindingIntegrityError(
        f"tracked protobuf authority is unusable: {message}; {_RESTORE_GUIDANCE}"
    )


def _lexical_absolute(path: Path) -> Path:
    raw = os.fspath(path)
    if "\x00" in raw:
        raise _integrity("binding root contains a NUL byte")
    supplied = Path(raw)
    if ".." in supplied.parts:
        raise _integrity(f"binding root aliases an ancestor via '..': {path}")
    return Path(os.path.abspath(raw))


def _open_directory_chain(path: Path) -> _PinnedBindingDirectory:
    lexical = _lexical_absolute(path)
    parts = lexical.parts
    if not parts or parts[0] != os.path.sep:
        raise _integrity(f"binding root is not absolute: {lexical}")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    fd = -1
    chain: list[tuple[str, _DirectoryNode]] = []
    try:
        fd = os.open(os.path.sep, flags)
        root_stat = os.fstat(fd)
        chain.append((os.path.sep, _directory_node(root_stat)))
        for component in parts[1:]:
            try:
                child_fd = os.open(component, flags, dir_fd=fd)
            except OSError as exc:
                raise _integrity(
                    f"binding-root component {component!r} is missing, symlinked, "
                    f"or unreadable: {exc}"
                ) from exc
            _ORIGINAL_OS_CLOSE(fd)
            fd = child_fd
            child_stat = os.fstat(fd)
            if not stat.S_ISDIR(child_stat.st_mode):
                raise _integrity(
                    f"binding-root component {component!r} is not a directory"
                )
            chain.append((component, _directory_node(child_stat)))
        final_stat = os.fstat(fd)
        return _PinnedBindingDirectory(
            lexical, fd, tuple(chain), _file_identity(final_stat)
        )
    except BaseException:
        if fd >= 0:
            try:
                _ORIGINAL_OS_CLOSE(fd)
            except OSError:
                pass
        raise


def _read_regular_fd(fd: int, *, maximum: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset <= maximum:
        chunk = os.pread(fd, min(64 * 1024, maximum + 1 - offset), offset)
        if not chunk:
            break
        chunks.append(chunk)
        offset += len(chunk)
        if offset > maximum:
            break
    return b"".join(chunks)


def _reject_top_level_import_candidates(directory: _PinnedBindingDirectory) -> None:
    try:
        entries = os.listdir(directory.fd)
    except OSError as exc:
        raise _integrity(f"cannot inventory binding root: {exc}") from exc
    extension_suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    for entry in entries:
        if entry == "__pycache__" or entry in _CANONICAL_FILENAMES:
            continue
        folded = entry.casefold()
        for pin in BINDING_PINS:
            module = pin.module_name
            if folded == pin.filename.casefold():
                raise _integrity(
                    f"case-alias conflicts with canonical source {pin.filename}: {entry}"
                )
            if entry == module:
                raise _integrity(
                    f"same-stem package/import candidate conflicts with {pin.filename}: "
                    f"{entry}"
                )
            if entry in (f"{module}.pyc", f"{module}.pyo"):
                raise _integrity(
                    f"same-stem sourceless candidate conflicts with {pin.filename}: "
                    f"{entry}"
                )
            if any(entry == module + suffix for suffix in extension_suffixes) or (
                entry.startswith(module + ".")
                and entry.endswith((".so", ".pyd", ".dll", ".dylib"))
            ):
                raise _integrity(
                    f"same-stem native candidate conflicts with {pin.filename}: {entry}"
                )


def _open_source(
    directory: _PinnedBindingDirectory, pin: BindingPin
) -> _RetainedSource:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        fd = os.open(pin.filename, flags, dir_fd=directory.fd)
    except OSError as exc:
        raise _integrity(
            f"{pin.filename} is missing, symlinked, or unreadable: {exc}"
        ) from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise _integrity(f"{pin.filename} is not a regular file")
        if before.st_nlink != 1:
            raise _integrity(
                f"{pin.filename} has link count {before.st_nlink}, expected 1"
            )
        if before.st_size != pin.size_bytes:
            raise _integrity(
                f"{pin.filename} has size {before.st_size}, expected {pin.size_bytes}"
            )
        payload = _read_regular_fd(fd, maximum=pin.size_bytes)
        after = os.fstat(fd)
        if _file_identity(after) != _file_identity(before):
            raise _integrity(f"{pin.filename} changed while it was read")
        digest = hashlib.sha256(payload).hexdigest()
        if len(payload) != pin.size_bytes or digest != pin.sha256:
            raise _integrity(
                f"{pin.filename} authenticated bytes do not match literal pin"
            )
        return _RetainedSource(pin, fd, _file_identity(before), payload)
    except BaseException:
        _ORIGINAL_OS_CLOSE(fd)
        raise


def _reopen_source_identity(
    directory: _PinnedBindingDirectory, source: _RetainedSource
) -> _FileIdentity:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        reopened = os.open(source.pin.filename, flags, dir_fd=directory.fd)
    except OSError as exc:
        raise _integrity(
            f"{source.pin.filename} path binding changed: {exc}"
        ) from exc
    try:
        return _file_identity(os.fstat(reopened))
    finally:
        _ORIGINAL_OS_CLOSE(reopened)


def _verify_directory_chain(directory: _PinnedBindingDirectory) -> None:
    current = _open_directory_chain(directory.path)
    try:
        if current.chain != directory.chain:
            raise _integrity("binding-root lexical ancestor identity changed")
    finally:
        _ORIGINAL_OS_CLOSE(current.fd)
        current.fd = -1


def _verify_retained_sources(
    directory: _PinnedBindingDirectory, sources: tuple[_RetainedSource, ...]
) -> None:
    if _file_identity(os.fstat(directory.fd)) != directory.root_identity:
        raise _integrity("binding-root directory generation changed")
    for source in sources:
        if _file_identity(os.fstat(source.fd)) != source.identity:
            raise _integrity(f"{source.pin.filename} retained identity changed")
        if _reopen_source_identity(directory, source) != source.identity:
            raise _integrity(f"{source.pin.filename} path was rebound")
        if _read_regular_fd(source.fd, maximum=source.pin.size_bytes) != source.payload:
            raise _integrity(f"{source.pin.filename} retained bytes changed")
    _verify_directory_chain(directory)


def _compile_sources(
    directory: _PinnedBindingDirectory, sources: tuple[_RetainedSource, ...]
) -> None:
    for source in sources:
        try:
            source.code = compile(
                source.payload,
                os.fspath(directory.path / source.pin.filename),
                "exec",
                dont_inherit=True,
                optimize=0,
            )
        except (SyntaxError, ValueError, TypeError) as exc:
            raise _integrity(
                f"authenticated {source.pin.filename} cannot compile: "
                f"{type(exc).__name__}: {exc}"
            ) from exc


def _close_fd_once(fd: int) -> None:
    _ORIGINAL_OS_CLOSE(fd)


def _close_retained_resources(
    sources: list[_RetainedSource], directory: _PinnedBindingDirectory | None
) -> tuple[BaseException, ...]:
    failures: list[BaseException] = []
    resources: list[tuple[str, Any]] = [
        (source.pin.filename, source) for source in sources
    ]
    if directory is not None:
        resources.append(("binding root", directory))
    for label, resource in resources:
        fd = resource.fd
        if fd < 0:
            continue
        try:
            _close_fd_once(fd)
        except BaseException as exc:
            failures.append(RuntimeError(f"failed to close {label}: {exc}"))
            try:
                _ORIGINAL_OS_CLOSE(fd)
            except OSError as fallback:
                if fallback.errno != errno.EBADF:
                    failures.append(
                        RuntimeError(f"fallback close failed for {label}: {fallback}")
                    )
            except BaseException as fallback:
                failures.append(
                    RuntimeError(f"fallback close failed for {label}: {fallback}")
                )
        finally:
            resource.fd = -1
    return tuple(failures)


def _canonical_module_entries() -> tuple[str, ...]:
    return tuple(name for name in _CANONICAL_MODULE_NAMES if name in sys.modules)


def _reject_preloaded_modules() -> None:
    contaminated = _canonical_module_entries()
    if contaminated:
        raise _integrity(
            "preloaded canonical binding names exist outside the private transaction: "
            + ", ".join(contaminated)
        )


def _reject_descriptor_contamination(runtime: _RuntimeAuthority) -> None:
    contaminated: list[str] = []
    for pin in BINDING_PINS:
        try:
            runtime.default_pool.FindFileByName(pin.proto_name)
        except KeyError:
            continue
        contaminated.append(pin.proto_name)
    if contaminated:
        raise _integrity(
            "default protobuf descriptor pool already contains tracked project "
            f"descriptors: {', '.join(contaminated)}"
        )


def _function_capture(runtime: _RuntimeAuthority, label: str) -> _FunctionCapture:
    for capture in runtime.functions:
        if capture.label == label:
            return capture
    raise AssertionError(f"missing retained function capture {label}")


def _clone_retained_function(
    capture: _FunctionCapture, globals_dict: dict[str, Any]
) -> types.FunctionType:
    return types.FunctionType(
        capture.code,
        globals_dict,
        getattr(capture.function, "__name__", capture.attribute),
        capture.defaults,
        capture.closure,
    )


def _new_execution_runtime(runtime: _RuntimeAuthority) -> _ExecutionRuntime:
    pool = runtime.pool_runtime_type()
    if type(pool) is not runtime.pool_runtime_type or pool is runtime.default_pool:
        raise _runtime_prerequisite("isolated descriptor pool construction failed")
    symbol_database = _PrivateSymbolDatabase(pool)
    message_factory = _IsolatedMessageFactory(
        pool,
        _function_capture(runtime, "message_factory.GetMessageClass").function,
        symbol_database,
    )
    generated_builder = _PrivateGeneratedBuilder(message_factory)

    descriptor_proxy = _FrozenNamespace(
        "google.protobuf.descriptor",
        {"_USE_C_DESCRIPTORS": runtime.descriptor_use_c},
    )
    descriptor_pool_proxy = _FrozenNamespace(
        "google.protobuf.descriptor_pool",
        {"Default": _ConstantFactory("private descriptor pool", pool)},
    )
    runtime_version_proxy = _FrozenNamespace(
        "google.protobuf.runtime_version",
        {
            "Domain": _FrozenNamespace(
                "google.protobuf.runtime_version.Domain",
                {"PUBLIC": runtime.domain_public},
            ),
            "ValidateProtobufRuntimeVersion": _PinnedRuntimeValidator(
                runtime.domain_public,
                frozenset(pin.proto_name for pin in BINDING_PINS),
            ),
        },
    )
    symbol_database_proxy = _FrozenNamespace(
        "google.protobuf.symbol_database",
        {"Default": _ConstantFactory("private symbol database", symbol_database)},
    )
    builder_proxy = _FrozenNamespace(
        "google.protobuf.internal.builder",
        {
            "BuildMessageAndEnumDescriptors": _FrozenCallable(
                "BuildMessageAndEnumDescriptors",
                generated_builder.BuildMessageAndEnumDescriptors,
            ),
            "BuildTopDescriptorsAndMessages": _FrozenCallable(
                "BuildTopDescriptorsAndMessages",
                generated_builder.BuildTopDescriptorsAndMessages,
            ),
        },
    )
    protobuf_proxy = _FrozenNamespace(
        "google.protobuf",
        {
            "descriptor": descriptor_proxy,
            "descriptor_pool": descriptor_pool_proxy,
            "runtime_version": runtime_version_proxy,
            "symbol_database": symbol_database_proxy,
        },
    )
    internal_proxy = _FrozenNamespace(
        "google.protobuf.internal", {"builder": builder_proxy}
    )
    return _ExecutionRuntime(
        authority=runtime,
        pool=pool,
        symbol_database=symbol_database,
        message_factory=message_factory,
        protobuf_proxy=protobuf_proxy,
        internal_proxy=internal_proxy,
        marker=_ExecutionMarker(),
    )


def _make_safe_import(
    owner: BindingPin,
    owned: Mapping[str, types.ModuleType],
    runtime: _ExecutionRuntime,
):
    def authenticated_import(
        name: str,
        globals: Mapping[str, Any] | None = None,
        locals: Mapping[str, Any] | None = None,
        fromlist: tuple[str, ...] | list[str] = (),
        level: int = 0,
    ) -> Any:
        del globals, locals
        if level != 0:
            raise ImportError(
                f"authenticated binding {owner.module_name} attempted relative import"
            )
        if name in _PIN_BY_MODULE:
            if fromlist or name not in owner.dependencies:
                raise ImportError(
                    f"authenticated binding {owner.module_name} attempted undeclared "
                    f"peer import {name}"
                )
            dependency = owned.get(name)
            if dependency is None:
                raise ImportError(
                    f"authenticated dependency {name} is not private transaction state"
                )
            return dependency
        requested = frozenset(str(item) for item in fromlist)
        if name == "google.protobuf" and requested <= {
            "descriptor",
            "descriptor_pool",
            "runtime_version",
            "symbol_database",
        }:
            return runtime.protobuf_proxy
        if name == "google.protobuf.internal" and requested <= {"builder"}:
            return runtime.internal_proxy
        raise ImportError(
            f"authenticated binding {owner.module_name} attempted unapproved import "
            f"{name!r} fromlist={tuple(fromlist)!r}"
        )

    return authenticated_import


def _new_private_module(
    pin: BindingPin, directory: _PinnedBindingDirectory
) -> types.ModuleType:
    module = types.ModuleType(pin.module_name)
    filename = os.fspath(directory.path / pin.filename)
    spec = importlib.machinery.ModuleSpec(
        pin.module_name,
        _OWNED_LOADER,
        origin=filename,
        is_package=False,
    )
    spec.has_location = False
    module.__file__ = filename
    module.__cached__ = None
    module.__loader__ = _OWNED_LOADER
    module.__package__ = ""
    module.__spec__ = spec
    return module


_RETAINED_MODULE_METADATA = frozenset(
    {
        "__name__",
        "__doc__",
        "__package__",
        "__loader__",
        "__spec__",
        "__file__",
        "__cached__",
    }
)


def _seal_private_module_namespace(module: types.ModuleType) -> None:
    for name in tuple(vars(module)):
        if name == "__builtins__" or (
            name.startswith("_") and name not in _RETAINED_MODULE_METADATA
        ):
            del module.__dict__[name]


def _execute_compiled_modules(
    directory: _PinnedBindingDirectory,
    sources: tuple[_RetainedSource, ...],
    runtime: _ExecutionRuntime,
) -> dict[str, types.ModuleType]:
    owned: dict[str, types.ModuleType] = {}
    published: list[tuple[str, types.ModuleType]] = []
    try:
        for source in sources:
            pin = source.pin
            if source.code is None:
                raise AssertionError(
                    f"{pin.filename} was not compiled before execution"
                )
            contaminated = tuple(
                name
                for name in _CANONICAL_MODULE_NAMES
                if name in sys.modules and name not in owned
            )
            if contaminated:
                raise _integrity(
                    "canonical binding names appeared during private transaction: "
                    + ", ".join(contaminated)
                )
            missing = [name for name in pin.dependencies if name not in owned]
            if missing:
                raise _integrity(
                    f"{pin.module_name} dependencies are not private: "
                    + ", ".join(missing)
                )
            module = _new_private_module(pin, directory)
            safe_builtins = {
                "__import__": _make_safe_import(pin, owned, runtime),
                "globals": builtins.globals,
            }
            module.__dict__["__builtins__"] = safe_builtins
            owned[pin.module_name] = module
            # UPB's generated-message metaclass resolves the module name through
            # sys.modules. Publish only authenticated transaction-owned modules
            # for the execution window, then remove them unconditionally.
            sys.modules[pin.module_name] = module
            published.append((pin.module_name, module))
            runtime.marker.started = True
            exec(source.code, module.__dict__, module.__dict__)
            if sys.modules.get(pin.module_name) is not module:
                raise _integrity(
                    f"{pin.module_name} transaction module binding changed"
                )
            _seal_private_module_namespace(module)
        return owned
    finally:
        for name, module in reversed(published):
            if sys.modules.get(name) is module:
                del sys.modules[name]



def _validate_owned_modules(
    owned: Mapping[str, types.ModuleType],
    runtime: _ExecutionRuntime,
    directory: _PinnedBindingDirectory | None,
) -> None:
    _validate_runtime_authority(runtime.authority)
    if (
        type(runtime.pool) is not runtime.authority.pool_runtime_type
        or runtime.pool is runtime.authority.default_pool
    ):
        raise _integrity("private descriptor pool identity changed")
    if runtime.symbol_database.pool is not runtime.pool:
        raise _integrity("private symbol database pool binding changed")
    if (
        runtime.message_factory.pool is not runtime.pool
        or object.__getattribute__(runtime.message_factory, "_symbol_database")
        is not runtime.symbol_database
        or object.__getattribute__(runtime.message_factory, "_get_message_class")
        is not _function_capture(
            runtime.authority, "message_factory.GetMessageClass"
        ).function
    ):
        raise _integrity("isolated message factory binding changed")
    expected_root = (
        directory.path if directory is not None else _lexical_absolute(_BINDING_ROOT)
    )
    for pin in BINDING_PINS:
        module = owned.get(pin.module_name)
        if type(module) is not types.ModuleType:
            raise _integrity(f"private module {pin.module_name} is missing or forged")
        if module.__name__ != pin.module_name or module.__loader__ is not _OWNED_LOADER:
            raise _integrity(f"{pin.module_name} private provenance changed")
        spec = module.__spec__
        expected_file = os.fspath(expected_root / pin.filename)
        if (
            type(spec) is not importlib.machinery.ModuleSpec
            or spec.name != pin.module_name
            or spec.loader is not _OWNED_LOADER
            or spec.origin != expected_file
            or module.__file__ != expected_file
            or module.__cached__ is not None
        ):
            raise _integrity(f"{pin.module_name} private spec provenance changed")
        descriptor = getattr(module, "DESCRIPTOR", None)
        if descriptor is None or getattr(descriptor, "name", None) != pin.proto_name:
            raise _integrity(f"{pin.module_name} descriptor ownership changed")
        try:
            pooled = runtime.pool.FindFileByName(pin.proto_name)
        except KeyError as exc:
            raise _integrity(
                f"private descriptor pool lacks {pin.proto_name}"
            ) from exc
        if pooled is not descriptor:
            raise _integrity(
                f"private descriptor identity mismatch for {pin.proto_name}"
            )
        observed_dependencies = tuple(dep.name for dep in descriptor.dependencies)
        expected_dependencies = tuple(
            _PIN_BY_MODULE[name].proto_name for name in pin.dependencies
        )
        if observed_dependencies != expected_dependencies:
            raise _integrity(
                f"{pin.proto_name} dependency graph changed: "
                f"{observed_dependencies!r}"
            )
        namespace = vars(module)
        for name, enum_descriptor in descriptor.enum_types_by_name.items():
            wrapper = namespace.get(name, _MISSING)
            if (
                type(wrapper) is not _PrivateEnumTypeWrapper
                or wrapper.DESCRIPTOR is not enum_descriptor
            ):
                raise _integrity(
                    f"{pin.module_name} enum wrapper {name!r} is not loader-owned"
                )
            for enum_value in enum_descriptor.values:
                observed = namespace.get(enum_value.name, _MISSING)
                if not _same_typed_value(observed, enum_value.number):
                    raise _integrity(
                        f"{pin.module_name} enum value {enum_value.name!r} changed"
                    )


def _identity_snapshot(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return types.MappingProxyType(dict(mapping))


def _identity_mapping_matches(
    observed: Mapping[Any, Any], expected: Mapping[Any, Any]
) -> bool:
    return set(observed) == set(expected) and all(
        observed[key] is value for key, value in expected.items()
    )


def _snapshot_object_namespace(value: Any) -> Mapping[str, Any] | None:
    try:
        namespace = vars(value)
    except TypeError:
        return None
    return _identity_snapshot(namespace)


def _build_private_state(
    owned: Mapping[str, types.ModuleType], runtime: _ExecutionRuntime
) -> tuple[_PrivateBindingState, DynamicTraceBindings]:
    token = object()
    views: dict[str, BindingModuleView] = {}
    snapshots: list[_ModuleSnapshot] = []
    module_names = {id(module): name for name, module in owned.items()}
    for pin in BINDING_PINS:
        module = owned[pin.module_name]
        namespace = _identity_snapshot(vars(module))
        view_values: dict[str, Any] = {}
        for name, value in namespace.items():
            dependency_name = module_names.get(id(value))
            if dependency_name is not None:
                view_values[name] = views[dependency_name]
            elif name == "__spec__":
                view_values[name] = _ReadOnlyModuleSpec(
                    pin.module_name,
                    _OWNED_LOADER,
                    os.fspath(_BINDING_ROOT / pin.filename),
                )
            else:
                view_values[name] = value
        frozen_view_values = _identity_snapshot(view_values)
        view = BindingModuleView(pin.module_name, token, frozen_view_values)
        views[pin.module_name] = view

        class_snapshots: list[_ObjectSnapshot] = []
        object_snapshots: list[_ObjectSnapshot] = []
        for value in namespace.values():
            if isinstance(value, type) and getattr(value, "__module__", None) == pin.module_name:
                class_namespace = _snapshot_object_namespace(value)
                if class_namespace is not None:
                    class_snapshots.append(_ObjectSnapshot(value, class_namespace))
            elif type(value) is _PrivateEnumTypeWrapper:
                object_namespace = _snapshot_object_namespace(value)
                if object_namespace is not None:
                    object_snapshots.append(_ObjectSnapshot(value, object_namespace))
        snapshots.append(
            _ModuleSnapshot(
                pin=pin,
                module=module,
                namespace=namespace,
                spec_state=_spec_state(module.__spec__),
                descriptor=module.DESCRIPTOR,
                classes=tuple(class_snapshots),
                objects=tuple(object_snapshots),
                view=view,
                view_values=frozen_view_values,
            )
        )

    state = _PrivateBindingState(
        token=token,
        modules=types.MappingProxyType(dict(owned)),
        snapshots=tuple(snapshots),
        views=types.MappingProxyType(dict(views)),
        runtime=runtime,
        symbol_classes=types.MappingProxyType(
            dict(runtime.symbol_database._classes)
        ),
    )
    bundle = DynamicTraceBindings(
        token,
        tuple(views[name] for name in _CANONICAL_MODULE_NAMES),
    )
    return state, bundle


def _validate_private_state(
    state: _PrivateBindingState, bundle: DynamicTraceBindings | None
) -> None:
    runtime = state.runtime
    _validate_runtime_authority(runtime.authority)
    _reject_descriptor_contamination(runtime.authority)
    _reject_preloaded_modules()
    _validate_owned_modules(state.modules, runtime, None)
    if not _identity_mapping_matches(
        runtime.symbol_database._classes, state.symbol_classes
    ):
        raise _integrity("private symbol database registrations changed")
    for snapshot in state.snapshots:
        module = snapshot.module
        if not _identity_mapping_matches(vars(module), snapshot.namespace):
            raise _integrity(
                f"{snapshot.pin.module_name} complete exported namespace changed"
            )
        if _spec_state(module.__spec__) != snapshot.spec_state:
            raise _integrity(f"{snapshot.pin.module_name} private spec changed")
        if module.DESCRIPTOR is not snapshot.descriptor:
            raise _integrity(f"{snapshot.pin.module_name} descriptor changed")
        for class_snapshot in snapshot.classes:
            namespace = _snapshot_object_namespace(class_snapshot.value)
            if namespace is None or not _identity_mapping_matches(
                namespace, class_snapshot.namespace
            ):
                raise _integrity(
                    f"{snapshot.pin.module_name} generated class namespace changed"
                )
        for object_snapshot in snapshot.objects:
            namespace = _snapshot_object_namespace(object_snapshot.value)
            if namespace is None or not _identity_mapping_matches(
                namespace, object_snapshot.namespace
            ):
                raise _integrity(
                    f"{snapshot.pin.module_name} generated export state changed"
                )
        view_values = object.__getattribute__(snapshot.view, "_values")
        if not _identity_mapping_matches(view_values, snapshot.view_values):
            raise _integrity(f"{snapshot.pin.module_name} sealed view changed")
    if bundle is not None:
        if (
            object.__getattribute__(bundle, "_token") is not state.token
            or object.__getattribute__(bundle, "_views")
            != tuple(state.views[name] for name in _CANONICAL_MODULE_NAMES)
        ):
            raise _integrity("published binding bundle changed")


def _mark_poisoned(exc: BaseException) -> None:
    global _loaded, _poisoned_reason, _private_state
    _loaded = None
    _private_state = None
    _poisoned_reason = f"{type(exc).__name__}: {exc}"


def _raise_poisoned() -> NoReturn:
    raise ProtoBindingLoaderPoisonedError(
        "authenticated protobuf loader is permanently poisoned after a prior "
        f"nonrollbackable failure: {_poisoned_reason}"
    )


def _validate_bundle_token(token: object) -> None:
    with _LOAD_LOCK:
        if _poisoned_reason is not None:
            _raise_poisoned()
        if (
            _private_state is None
            or _loaded is None
            or _private_state.token is not token
        ):
            raise ProtoBindingLoaderPoisonedError(
                "authenticated protobuf capability is no longer published"
            )
        try:
            _validate_private_state(_private_state, _loaded)
        except BaseException as exc:
            _mark_poisoned(exc)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise ProtoBindingLoaderPoisonedError(
                "authenticated protobuf private state changed after publication"
            ) from exc


def load_dynamic_trace_bindings() -> DynamicTraceBindings:
    """Return the complete immutable nine-binding capability bundle."""

    global _loaded, _private_state
    with _held_import_locks():
        if _poisoned_reason is not None:
            _raise_poisoned()
        runtime_authority = _load_runtime_authority()
        if _loaded is not None:
            try:
                if _private_state is None:
                    raise _integrity("published private transaction state is missing")
                _validate_private_state(_private_state, _loaded)
            except BaseException as exc:
                _mark_poisoned(exc)
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                raise ProtoBindingLoaderPoisonedError(
                    "authenticated protobuf private state changed after publication"
                ) from exc
            return _loaded

        _reject_preloaded_modules()
        _reject_descriptor_contamination(runtime_authority)
        directory: _PinnedBindingDirectory | None = None
        sources: list[_RetainedSource] = []
        execution_runtime: _ExecutionRuntime | None = None
        candidate_state: _PrivateBindingState | None = None
        candidate_bundle: DynamicTraceBindings | None = None
        primary_error: BaseException | None = None
        try:
            directory = _open_directory_chain(_BINDING_ROOT)
            _reject_top_level_import_candidates(directory)
            for pin in BINDING_PINS:
                sources.append(_open_source(directory, pin))
            retained = tuple(sources)
            _compile_sources(directory, retained)
            if any(source.code is None for source in retained):
                raise AssertionError("not all binding snapshots were compiled")
            _verify_retained_sources(directory, retained)
            _reject_descriptor_contamination(runtime_authority)
            execution_runtime = _new_execution_runtime(runtime_authority)
            owned = _execute_compiled_modules(
                directory, retained, execution_runtime
            )
            _validate_owned_modules(owned, execution_runtime, directory)
            _verify_retained_sources(directory, retained)
            candidate_state, candidate_bundle = _build_private_state(
                owned, execution_runtime
            )
        except BaseException as exc:
            primary_error = exc

        cleanup_failures = _close_retained_resources(sources, directory)
        execution_started = bool(
            execution_runtime is not None and execution_runtime.marker.started
        )
        if cleanup_failures:
            details = "; ".join(str(failure) for failure in cleanup_failures)
            cleanup_error = _integrity(f"retained descriptor cleanup failed: {details}")
            if execution_started:
                _mark_poisoned(cleanup_error)
            raise cleanup_error from cleanup_failures[0]

        if primary_error is not None:
            if execution_started:
                _mark_poisoned(primary_error)
            if isinstance(primary_error, (KeyboardInterrupt, SystemExit)):
                raise primary_error
            if isinstance(
                primary_error,
                (ProtoBindingRuntimePrerequisiteError, ProtoBindingIntegrityError),
            ):
                raise primary_error
            raise _integrity(
                "authenticated private binding load failed with "
                f"{type(primary_error).__name__}: {primary_error}"
            ) from primary_error

        if candidate_state is None or candidate_bundle is None:
            raise _integrity("private binding transaction produced no complete bundle")
        try:
            _validate_private_state(candidate_state, None)
        except BaseException as exc:
            if execution_started:
                _mark_poisoned(exc)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if isinstance(exc, ProtoBindingError):
                raise
            raise _integrity(
                f"private binding validation failed with {type(exc).__name__}: {exc}"
            ) from exc
        # Publication is deliberately after all retained filesystem resources close.
        _private_state = candidate_state
        _loaded = candidate_bundle
        return candidate_bundle


__all__ = [
    "BINDING_PINS",
    "BindingModuleView",
    "BindingPin",
    "DynamicTraceBindings",
    "EXPECTED_PROTOBUF_RUNTIME_VERSION",
    "ProtoBindingError",
    "ProtoBindingIntegrityError",
    "ProtoBindingLoaderPoisonedError",
    "ProtoBindingRuntimePrerequisiteError",
    "load_dynamic_trace_bindings",
]
