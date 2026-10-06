#!/usr/bin/env python3
import shutil
import tempfile
import unittest
from pathlib import Path

from sm86_compile_manifest import CompileRefusal, compile_sm86


PTX = r""".version 8.0
.target sm_86
.address_size 64
.visible .entry add_one(.param .u64 out_ptr)
{
    .reg .b32 %r<3>;
    .reg .b64 %rd<2>;
    ld.param.u64 %rd1, [out_ptr];
    mov.u32 %r1, %tid.x;
    add.u32 %r2, %r1, 1;
    st.global.u32 [%rd1], %r2;
    exit;
}
"""


class Sm86CompileManifestTests(unittest.TestCase):
    def test_target_mismatch_refuses_before_toolchain(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "wrong.ptx"
            source.write_text(PTX.replace(".target sm_86", ".target sm_90"))
            with self.assertRaises(CompileRefusal) as raised:
                compile_sm86(source, Path(directory) / "out")
            self.assertEqual(raised.exception.code, "target_mismatch")

    def test_pinned_tools_emit_static_sm86_artifacts_when_available(self):
        ptxas = shutil.which("ptxas")
        nvdisasm = shutil.which("nvdisasm")
        cuobjdump = shutil.which("cuobjdump")
        if not ptxas or not nvdisasm or not cuobjdump:
            self.skipTest("CUDA ptxas/nvdisasm/cuobjdump unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "kernel.ptx"
            source.write_text(PTX)
            result = compile_sm86(
                source,
                root / "out",
                ptxas=ptxas,
                nvdisasm=nvdisasm,
                cuobjdump=cuobjdump,
                alpha=False,
            )
            self.assertEqual(result["schema"], "sm86_static_sass_artifact_v1")
            self.assertTrue(result["claims"]["static_sm86_sass_generation"])
            self.assertFalse(result["claims"]["runtime_value_regeneration"])
            self.assertTrue((root / "out/kernel.sm_86.cubin").stat().st_size > 0)
            self.assertTrue((root / "out/kernel.sm_86.sass.txt").stat().st_size > 0)
            self.assertTrue((root / "out/kernel.sm_86.cuobjdump.sass.txt").stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
