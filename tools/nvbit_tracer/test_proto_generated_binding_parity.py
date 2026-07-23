"""Parity test between `instruction.proto` and the tracked
generated C++ bindings at `pb_trace/include/instruction.pb.h`.

A prior review caught a critical build-breaking gap: the
proto file declared `MEMORY_OP_CLASS_TMEM = 10` and the
tracer C++ static_assert referenced it, but the tracked
generated C++ binding still ended `MemoryOpClass` at
`MEMORY_OP_CLASS_CLUSTER_DSMEM = 9`. A source-tree checkout
could not compile the tracer code as written.

This parity test prevents the regression: it parses the
enum members from both `instruction.proto` and the tracked
generated header, and asserts they match. A drift = test
failure with an actionable diagnostic.

The proto + tracked-generated pair are intentionally checked together. The
generated header is a tracked authority at
`tools/nvbit_tracer/traces_enhanced/pb_trace/include/instruction.pb.h`
and missing or mismatched bytes are restored from the committed tree. Canonical
checkout regeneration is not an error-recovery path while H3 holds.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_PROTO_PATH = (
    _HERE / "traces_enhanced" / "dynamic_trace" / "instruction.proto"
)
_GEN_HEADER_PATH = (
    _HERE / "traces_enhanced" / "pb_trace" / "include" / "instruction.pb.h"
)
_PROTO_DIR = _HERE / "traces_enhanced" / "dynamic_trace"
_GEN_ROOT = _HERE / "traces_enhanced" / "pb_trace"
_PB_PREFIX = _HERE.parents[1] / ".toolchain" / "pb"
_HOST_GATE = "NVBIT_PROTOBUF_HOST_GATE"


_PROTO_MEMBER_RE = re.compile(
    r"^\s*(MEMORY_OP_CLASS_[A-Z_]+)\s*=\s*(\d+)\s*;",
    re.MULTILINE,
)
_CPP_MEMBER_RE = re.compile(
    r"^\s*(MEMORY_OP_CLASS_[A-Z_]+)\s*=\s*(\d+)\s*,?\s*$",
    re.MULTILINE,
)
_CPP_MAX_RE = re.compile(
    r"MemoryOpClass_MAX\s*=\s*static_cast<MemoryOpClass>\((\d+)\)",
)


def _parse_proto_members() -> list[tuple[str, int]]:
    text = _PROTO_PATH.read_text(encoding="utf-8")
    return [(m.group(1), int(m.group(2))) for m in _PROTO_MEMBER_RE.finditer(text)]


def _parse_cpp_members() -> list[tuple[str, int]]:
    text = _GEN_HEADER_PATH.read_text(encoding="utf-8")
    members: list[tuple[str, int]] = []
    seen = set()
    for m in _CPP_MEMBER_RE.finditer(text):
        name = m.group(1)
        if name in seen:
            continue
        seen.add(name)
        members.append((name, int(m.group(2))))
    return members


def _parse_cpp_max() -> int | None:
    text = _GEN_HEADER_PATH.read_text(encoding="utf-8")
    m = _CPP_MAX_RE.search(text)
    return int(m.group(1)) if m else None


class ProtoGeneratedMemoryOpClassParityTest(unittest.TestCase):
    """The tracked generated C++ binding's MemoryOpClass enum
    MUST be an exact mirror of the proto enum (same names,
    same numeric values, same ordering).

    Drift between the two breaks the production tracer build
    silently: the C++ static_assert in tracer_tool.cu
    references the enum member by name, and a missing member
    causes a compile error a developer only sees when
    rebuilding from a clean checkout.
    """

    def test_generated_binding_mirrors_proto_enum(self) -> None:
        proto_members = _parse_proto_members()
        cpp_members = _parse_cpp_members()
        self.assertGreater(
            len(proto_members), 0,
            msg="proto enum extraction yielded zero members",
        )
        self.assertEqual(
            proto_members, cpp_members,
            msg=(
                "instruction.proto MemoryOpClass enum diverges "
                "from tracked generated binding at "
                f"{_GEN_HEADER_PATH.name}:\n"
                f"  proto:  {proto_members!r}\n"
                f"  cpp:    {cpp_members!r}\n"
                "Restore the tracked authority byte-exactly via:\n"
                "  git restore --source=HEAD --worktree -- "
                "tools/nvbit_tracer/traces_enhanced/pb_trace/include/"
                "instruction.pb.h\n"
                "Do not regenerate it in the canonical checkout."
            ),
        )

    def test_generated_binding_max_matches_highest_proto_value(self) -> None:
        proto_members = _parse_proto_members()
        cpp_max = _parse_cpp_max()
        self.assertIsNotNone(
            cpp_max,
            msg=(
                f"MemoryOpClass_MAX not found in "
                f"{_GEN_HEADER_PATH.name}"
            ),
        )
        highest_proto_value = max(value for _, value in proto_members)
        self.assertEqual(
            cpp_max, highest_proto_value,
            msg=(
                "MemoryOpClass_MAX in tracked generated binding "
                f"({cpp_max}) does not match the highest proto "
                f"enum value ({highest_proto_value}); "
                "restore the tracked binding byte-exactly from HEAD."
            ),
        )

    def test_host_only_regeneration_acceptance(self) -> None:
        """Byte-compare protoc output only under the explicit host gate.

        The default branch is a real structural test, not a skipped test: it
        proves that the complete tracked C++ authority set is present. The
        explicit host gate adds regeneration into an external temporary
        directory and never writes beneath the checkout.
        """

        protos = sorted(_PROTO_DIR.glob("*.proto"))
        expected_names = [path.stem for path in protos]
        self.assertEqual(len(expected_names), 9)
        self.assertEqual(
            sorted(
                path.stem.removesuffix(".pb")
                for path in (_GEN_ROOT / "include").glob("*.pb.h")
            ),
            expected_names,
        )
        self.assertEqual(
            sorted(
                path.stem.removesuffix(".pb")
                for path in (_GEN_ROOT / "src").glob("*.pb.cc")
            ),
            expected_names,
        )

        gate = os.environ.get(_HOST_GATE)
        self.assertIn(
            gate,
            (None, "0", "1"),
            f"{_HOST_GATE} must be 0, 1, or unset",
        )
        if gate != "1":
            return

        prefix = Path(os.environ.get("NVBIT_PB_PREFIX", str(_PB_PREFIX)))
        self.assertTrue(prefix.is_absolute(), "NVBIT_PB_PREFIX must be absolute")
        protoc = prefix / "bin" / "protoc"
        self.assertTrue(protoc.is_file() and os.access(protoc, os.X_OK), protoc)
        self.assertTrue(
            protoc.resolve().is_relative_to(prefix.resolve()),
            f"pinned protoc resolves outside its verified prefix: {protoc}",
        )
        version = subprocess.run(
            [str(protoc), "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.assertEqual(version, "libprotoc 34.1")
        pinned_env = _HERE / "pinned_build_env.sh"
        verification_env = os.environ.copy()
        verification_env["NVBIT_PB_PREFIX"] = str(prefix)
        subprocess.run(
            [
                "bash",
                "-c",
                f'source "{pinned_env}" >/dev/null',
            ],
            check=True,
            env=verification_env,
        )

        with tempfile.TemporaryDirectory(
            prefix="nvbit-pb-host-gate-", dir="/tmp"
        ) as raw:
            out = Path(raw)
            subprocess.run(
                [
                    str(protoc),
                    f"--proto_path={_PROTO_DIR}",
                    f"--cpp_out={out}",
                    *(str(path) for path in protos),
                ],
                check=True,
            )
            for name in expected_names:
                for suffix, tracked_dir in (
                    (".pb.h", "include"),
                    (".pb.cc", "src"),
                ):
                    generated = out / f"{name}{suffix}"
                    tracked = _GEN_ROOT / tracked_dir / f"{name}{suffix}"
                    self.assertEqual(
                        generated.read_bytes(),
                        tracked.read_bytes(),
                        f"host-only regeneration differs from tracked authority: {tracked}",
                    )


if __name__ == "__main__":
    unittest.main()
