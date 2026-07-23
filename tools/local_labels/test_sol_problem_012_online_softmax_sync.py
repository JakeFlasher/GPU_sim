"""Static contracts for the problem-012 online-softmax synchronization."""
from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE = (
    REPO_ROOT
    / "workloads/sol_execbench_baselines/FIB"
    / "problem_012_gqa_paged_decode_h32_kv4_d128_ps1/cuda_cpp"
)


def _kernel_text() -> str:
    return (BASELINE / "kernel.cu").read_text(encoding="utf-8")


def test_online_softmax_factors_are_published_before_all_warps_use_them():
    kernel = _kernel_text()
    update = re.search(
        r"const float logit = logit_smem;\s+"
        r"// Thread 0 owns.*?"
        r"if \(tid == 0\) \{(?P<owner>.*?)\}\s+"
        r"__syncthreads\(\);\s+"
        r"acc = acc \* rescale_smem \+ exp_logit_smem \* v_val;",
        kernel,
        flags=re.DOTALL,
    )
    assert update is not None
    owner = update.group("owner")
    assert "const float m_prev = m_smem;" in owner
    assert "m_smem = m_new;" in owner
    assert "rescale_smem =" in owner
    assert "exp_logit_smem =" in owner
    assert kernel.count("const float m_prev = m_smem;") == 1
    assert "__shared__ float rescale_smem;" in kernel
    assert "__shared__ float exp_logit_smem;" in kernel


def test_packed_solution_embeds_the_exact_synchronized_kernel():
    kernel = _kernel_text()
    solution = json.loads((BASELINE / "solution.json").read_text(encoding="utf-8"))
    sources = {source["path"]: source["content"] for source in solution["sources"]}
    assert sources["kernel.cu"] == kernel
