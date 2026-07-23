"""
_relevance.py — Single source of truth for the KB-relevance classifier.

This module defines the Group A / B / C gate that decides whether a user
prompt is relevant to the project knowledge base. It is imported by both
`route_query.py` (UserPromptSubmit hook) and `check_kb_usage.py` (Stop
hook); the bash fallback in `.claude/hooks/kb_reminder.sh` is generated from
this module by `python3 scripts/kb/_relevance.py --emit-bash` and delegates
classification back to this same Python implementation.

Group A — fires alone. Specific NVIDIA / project-internal tokens that
cannot collide with everyday English: instruction mnemonics,
NVIDIA-specific acronyms, sm_NN families, SKU codes, CUDA primitive
names, measurement-contract terms, error tokens.

Group B — architecture codenames (Tesla through Rubin). Fires only when
Group C is also present — kills "ada lovelace" / "fermi paradox" /
"pascal triangle" / "tesla model s" / "kepler third law" / "volta the
streamer" / "ampere the unit" false-positive class.

Group C — GPU-context disambiguators. Group C alone NEVER fires; it
only enables Group B codenames. So we can include semi-common words
here that would be unsafe in Group A.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Pattern source strings (kept as plain strings so the bash regex emitter
# can serialize them verbatim).
# ---------------------------------------------------------------------------

GROUP_A_SOURCE = (
    # PTX / SASS mnemonics that can't collide with English
    r"\btcgen05\b|\bwgmma\b|\bwarpgroup\b|\bdsmem\b|\bldmatrix\b|"
    r"\bstmatrix\b|\bldgsts\b|\bmbarrier\b|cp\.async|tensormap|"
    # Privileged-API error and config tokens (forbidden surfaces)
    r"ERR_NVGPUCTRPERM|RestrictProfiling|"
    # Numerical formats (Blackwell-era)
    r"\bnvfp4\b|mxfp[468]?\b|\bfp4\b|\bfp6\b|\bfp8\b|\bbf16\b|\btf32\b|"
    # CUDA tooling binaries
    r"\bcuobjdump\b|\bnvdisasm\b|\bnvprune\b|cu\+\+filt|\bcutlass\b|"
    r"\bnvbit\b|\bcupti\b|cupti\.?activity|\bnvcc\b|\bncu\b|\bnsight\b|"
    r"\bnsys\b|\bptxas\b|\bptx\b|\bsass\b|\bsimt\b|\bcuda\b|"
    # Project-internal infrastructure tokens
    r"\bglobaltimer\b|\bclock64\b|\bnvl72\b|nv-?hbi|\bnvlink\b|"
    r"\bnvswitch\b|\btmem\b|\bcga\b|\btma\b|"
    # Tensor-core SASS family (Volta through Blackwell)
    r"\bhmma\b|\bimma\b|\bdmma\b|\bomma\b|\bqmma\b|\bwmma\b|\bbmma\b|"
    r"\bgmma\b|"
    # SM target families and SKU codes
    r"sm_?[0-9]{2,3}a?\b|"
    r"\bgb[12]0[0-9]\b|\bgh[12]0[0-9]\b|\bga[12]0[0-9]\b|\bad[12]0[0-9]\b|"
    r"\btu1[01][0-9]\b|\bb[12]00\b|\bh[12]00\b|\ba[12]00\b|\bv100\b|"
    r"\bp100\b|\bk80\b|\bk[12]0\b|"
    # CUDA C++ primitives (decorators / sync intrinsics / launch syntax)
    r"__syncthreads|__syncwarp|__shfl_?(sync|up|down|xor)?|"
    r"__ballot_?(sync)?|__any_?(sync)?|__all_?(sync)?|__activemask|"
    r"__nanosleep|__threadfence(_block|_system)?|__launch_bounds__|"
    r"__restrict__|__device__|__global__|__shared__|__constant__|"
    r"__managed__|__forceinline__|__noinline__|"
    # CUDA atomic intrinsics + memory fences
    r"\batomicCAS\b|\batomicAdd\b|\batomicSub\b|\batomicExch\b|"
    r"\batomicOr\b|\batomicXor\b|\batomicAnd\b|\batomicMin\b|\batomicMax\b|"
    r"\batomicInc\b|\batomicDec\b|memory.?fence|fence\.acq_rel|"
    r"fence\.sc|fence\.gpu|fence\.sys|fence\.cluster|fence\.cta|"
    # Bare PTX warp-vote / shuffle mnemonics (no double-underscore prefix)
    r"shfl\.sync|vote\.sync|ballot\.sync|match\.sync|activemask\.sync|"
    r"redux\.sync|"
    # Architecture-history phrasing
    r"warp.?lockstep|single.?thread.?issue|warp.?level\s+execution|"
    r"register.?pressure|register.?spill|register.?budget|"
    r"cooperative_?groups?|cooperative\.?group|cudagraph|cudastream|"
    r"cudamalloc|cudamemcpy|cudalaunchkernel|<<<[^>]+>>>|"
    # Measurement-contract phrasing
    r"wall.?clock|non.?root|counter.?free|counter.?less|"
    # Architecture-history phrasing (specific to this project's KB)
    r"architectural.?evolution|architectural.?rewrite|three.?rewrites|"
    r"three.?(?:architectural)?.?rewrite[sd]?|rtx ?pro ?6000|"
    # Cluster + tensor-core surface
    r"cluster\.?(launch|sync|api|attribute|cga|grid|cooperative|size|"
    r"portable|remote)|tensor.?core|"
    # `bank.?conflict` only when paired with a memory anchor — bare
    # "bank conflict" is a finance phrase too. Require shared/smem/ldg/etc.
    r"(?:shared|smem|ldg|stg|memory|kernel|warp|thread).{0,30}bank.?conflict|"
    r"bank.?conflict.{0,30}(?:shared|smem|ldg|stg|memory|kernel|warp|thread)|"
    r"warp.?(scheduler|divergence|group|specialization|specialized|sync|"
    r"shuffle|vote|primitive|level|reconverg)|"
    # Simulator / modeling research (project-specific high-signal phrases)
    r"\bnvarchsim\b|\bgpgpu.?sim\b|\baccel.?sim\b|\bclustersim\b|"
    r"\bnvasim\b|\bnexosim\b|\bgcom\b|"
    # Bare `mma.sync` / `mma.async` style mentions
    r"\b(?:m|h|i|d|o|q|w|b|g)?mma\.?(?:sync|async)?\b|"
    # Project-context phrases that uniquely identify this repo
    r"validation.?contract|event.?driven\s+(des|sm|simulator|timing)|"
    r"simulator\s+(?:bug|validation|contract|model|architecture|timing|"
    r"baseline|verdict)|gpu\s+simulator|"
    r"\bsynchronization\s+(evolv|rewrite|history|primitive|domain)|"
    # Meta-router prompts (project-internal vocabulary).
    # `specialty\s+skills?` alone fires on "my specialty skill is CSS" or
    # "this candidate has specialty skills" — too loose. Require explicit
    # KB-vocabulary tail (in repo/kb/project/knowledge) OR an enumeration
    # verb prefix (list/show/name/available).
    r"\b(?:available|list|show|name)\s+(?:the\s+)?(?:available\s+)?"
    r"specialty\s+skills?|"
    r"\b(?:repo|kb|project|our)\s+specialty\s+skills?|"
    r"specialty\s+skills?\s+(?:in|of|for|under|inside|within|that|to)\s+"
    r"(?:the\s+|this\s+|our\s+)?(?:repo|kb|project|knowledge\s+base)|"
    r"knowledge\s+base\s+(?:structure|inventory|router|index|"
    r"layout|content|cards?|skills?)|"
    r"(?:in|of|for|under|inside|within|browse|navigate)\s+"
    r"(?:the\s+|this\s+|our\s+|repo\'?s?\s+|project\'?s?\s+)?"
    r"(?:knowledge\s+base|kb)\b|"
    r"\bINDEX\.md\b|\bproject\s+kb\b|\brepo\'?s?\s+kb\b|"
    r"available\s+(?:specialty\s+)?skills?\s+(?:in\s+)?(?:this\s+)?(?:repo|kb)|"
    # Control-flow / predication phrasing (GPU-specific compositions only).
    r"\bpredication\b|control.?flow\s+(?:perf|optimization|tuning|graph|"
    r"analysis|kernel|gpu|warp|divergen)|\bbranch\s+(?:divergence|predication|"
    r"prediction|reconverg)\b|"
    # Meta-router phrasing — "shortest path to the right card", "which card"
    r"shortest\s+path\s+to\s+(?:the\s+)?(?:right\s+)?card|"
    r"\b(?:right|correct|best)\s+(?:card|skill)\s+(?:to|for|that)\b|"
    r"\bwhich\s+card\s+(?:should|do|to|covers|owns|explains)\b|"
    # SOL-ExecBench routing anchors. `problem_NNN` is highly specific
    # (matches the canonical filename stem under
    # references/NVIDIA_Manuals/solbench_problems/). The kernel-family
    # tokens (gemm, rmsnorm, rope, moe, gqa, swiglu, paged-attention,
    # fused-add, fused-rms, mla, ssm, mamba) are also technical enough
    # to fire alone in Group A; false-positive class is small.
    r"\bproblem_[0-9]{3,}|"
    r"\b(?:gemm|rmsnorm|rope|moe|gqa|swiglu|paged.?attention|"
    r"fused.?add|fused.?rms|mla|ssm|mamba|block.?scale|e4m3|"
    r"e5m2|e2m1|geglu|hyena|layernorm|groupnorm)\b|"
    # High-precision source-stub and pattern-family anchors.
    r"\bsol.?execbench\b|"
    r"\bblackwell\s+tuning\b|"
    r"\bflow.?coupling\b|"
    r"\bperceptual.?loss\b|"
    r"\bmultimodal.?(?:sequence.?packing|packing|tokens?)\b|"
    r"\b(?:sequence|token).?packing\b|"
    r"\bmask.?decoder\b|"
    r"\bsam.?(?:hq|mask)\b|"
    # Attention, activation, and dispatch compounds avoid generic-word FPs.
    r"\battention\s+(?:backward|forward|core|mask(?:ing)?|prefill|decode|qkv)\b|"
    r"\b(?:decoder|encoder|transformer)\s+layer\b|"
    r"\b(?:silu|gelu|relu|sigmoid|tanh)\s+activation\b|"
    r"\bactivation\s+(?:silu|gelu|relu|sigmoid|tanh)\b|"
    r"\belementwise\s+activation\b|"
    r"\bsparse\s+activation\b|"
    r"\b(?:top.?k|topk)\s+(?:dispatch|routing|gating|select)\b|"
    r"\bflash\s+attention\b|"
    # CUTLASS/CuTe and SOLAR/evaluator-harness compounds.
    r"\btcgen05\b|\btmem\b|"
    r"\bcollective\s+mainloop\b|\btiled\s+mma\b|\btiledmma\b|"
    r"\bepilogue\s+visitor\b|\btma\s+descriptor\b|"
    r"\bt_sol\b|\borojenesis\b|"
    r"\bsolar\s+(?:framework|deriv\w+|tsol|t_sol|pipeline|roofline|"
    r"einsum|orojenesis|analyz\w+)\b|"
    r"\beval(?:uator)?[\s\-_]?harness\b|"
    r"\b(?:evaluator.?framework|native.?evaluator.?entrypoint|"
    r"solution.?json.?schema|cupti.?timing.?divergence|"
    r"tsol.?deriver|orojenesis.?elements|"
    r"einsum.?graph(?:.?renamed)?|5.?stage.?pipeline|arch.?config.?b200)\b"
)

GROUP_B_SOURCE = (
    r"\b(tesla|fermi|kepler|maxwell|pascal|volta|turing|ampere|ada|"
    r"hopper|blackwell|rubin)\b"
)

GROUP_C_SOURCE = (
    # Hardware-domain noun anchors
    r"\bgpu\b|\bkernel\b|\bwarp\b|\bthread\b|\bblock\b|\bgrid\b|"
    r"\bregister\b|shared.?memory|\bsmem\b|global.?memory|\blaunch\b|"
    r"\bstream\b|\barchitecture\b|\barchitectural\b|microarchitect|"
    # Performance-attribution words
    r"\bspeedup\b|\bthroughput\b|\btflops\b|\bgflops\b|"
    r"\boccupanc(y|ies)\b|\bcoalesc(e|ed|ing|ence)\b|\bdriver\b|"
    r"\bcompute\s+capabilit|\bsm[ \t_]|\binstruction\b|\bnvidia\b|"
    r"\bcompiler\b|\bcubin\b|\bfatbin\b|\bdevice\b|\bhost\b|"
    # Patent / architecture-evolution prose
    r"\bpatent\b|\bcluster\b|\bbroadcast\b|\bbarrier\b|"
    r"transformer.?engine|tensor.?memory|\btile\b|\bdsm\b|"
    r"\bdescriptor\b|\bscoreboard\b|\bdispatch\b|\bpipeline\b|"
    r"execution.?model|\btransaction\b|\binstruction.?set\b|"
    r"\bcoalescing\b|\bdivergen|\bsubcore\b|\bsharp\b|"
    r"all.?reduce|\ball-?reduce\b|\bcollective\b|"
    # Generation-comparison and history words
    r"\bevolve\b|\bevolution\b|\bsynchronization\b|\brewrite[sd]?\b|"
    r"\bgeneration\b|"
    # Simulator / modeling vocabulary (project-internal)
    r"\bsimulator\b|\bsimulation\b|\bvalidation\b|\bmodeling\b|"
    r"event.?driven|\bbenchmark\b|\bmicrobench|\bworkload\b|"
    r"\btrace\b|\btiming\b"
)

# ---------------------------------------------------------------------------
# Compiled patterns (used by route_query.py and check_kb_usage.py).
# ---------------------------------------------------------------------------

GROUP_A = re.compile(GROUP_A_SOURCE, re.IGNORECASE)
GROUP_B = re.compile(GROUP_B_SOURCE, re.IGNORECASE)
GROUP_C = re.compile(GROUP_C_SOURCE, re.IGNORECASE)

# Codename anti-patterns. When the codename appears in clearly non-GPU
# context (Tesla cars, Pascal triangle, Fermi paradox, Ada Lovelace's
# biography, Kepler's laws, Maxwell's equations, Turing test, etc.), the
# Group B codename match is a false positive. We bypass even when
# Group C also fires, because the surrounding text reveals the non-GPU
# domain. This is surgical — list only the highest-frequency collisions.
_CODENAME_FP_SOURCE = (
    r"\btesla\s+(?:model\s+[xyzs3]\b|motors|stock|company|inc\b|cars?\b|"
    r"founded|cybertruck|coil|inventor|nikola)|"
    r"\bfermi\s+(?:paradox|estimate|level|enrico)|"
    r"\bpascal\s+(?:triangle|wager|programming\s+language|line\b|blaise|"
    r"\bunit\b)|"
    r"\bada\s+(?:lovelace|byron|countess|programming\s+language|king)|"
    r"\bkepler\s+(?:third|second|first|law\b|laws\b|orbital|telescope|"
    r"johannes)|"
    r"\bvolta\s+(?:streamer|content|youtube|alessandro|inventor\s+of\s+the\s+"
    r"battery)|"
    r"\bampere\s+(?:unit|of\s+current|measurement|andre.marie|definition)|"
    r"\bturing\s+(?:test|machine|completeness|alan|halting)|"
    r"\bmaxwell\s+(?:equation|electromagnetic|james|clerk|demon)|"
    r"\bhopper\s+(?:grace|admiral|grasshopper|toy)|"
    r"\bgrace\s+hopper\b|"
    r"\badmiral\s+hopper\b|"
    r"\bblackwell\s+(?:college|island|justice|publishing|elizabeth)|"
    r"\bnikola\s+tesla\b|\btesla\s+coil\b"
)
_CODENAME_FP = re.compile(_CODENAME_FP_SOURCE, re.IGNORECASE)


def is_kb_relevant(prompt: str) -> bool:
    """Return True if the prompt should activate the KB router/Stop hook.

    Group A fires alone. Group B (codenames) requires Group C
    (GPU-context word) to fire. This kills the "ada lovelace" /
    "fermi paradox" / "pascal triangle" / "tesla model s" false-positive
    class while still admitting genuinely cross-architecture prompts.
    """
    if not prompt:
        return False
    if GROUP_A.search(prompt):
        return True
    if GROUP_B.search(prompt) and GROUP_C.search(prompt):
        # Reject codename + GPU-context combos when the codename's
        # surrounding text is clearly non-GPU (Tesla model X, Pascal
        # triangle, Fermi paradox, Ada Lovelace, etc.).
        if _CODENAME_FP.search(prompt):
            return False
        return True
    return False


def _emit_bash() -> str:
    """Emit the sourced bash fallback wrapper for the canonical classifier.

    Bash's POSIX ERE engine cannot faithfully evaluate the Python patterns
    (which use constructs such as non-capturing groups and word boundaries).
    The generated wrapper therefore passes the prompt through stdin to this
    module instead of duplicating or approximating the classifier.
    """
    return r'''# Auto-generated from scripts/kb/_relevance.py — DO NOT EDIT BY HAND.
# Regenerate via: python3 scripts/kb/_relevance.py --emit-bash > .claude/hooks/_gate_regexes.sh
#
# This file is meant to be sourced by a bash hook:
#     source .claude/hooks/_gate_regexes.sh
#     if kb_gate_is_relevant "$PROMPT"; then ...; fi
#
# The Python classifier in scripts/kb/_relevance.py is the canonical source of
# truth. Bash delegates to it through stdin so long prompts do not collide with
# the operating system's argv-size limit and no regex dialect can drift.

kb_gate_is_relevant() {
    # Silent predicate. Returns exit status 0 when the prompt is KB-relevant,
    # 1 otherwise (including an empty prompt or unavailable Python helper).
    # No stdout output: UserPromptSubmit hooks show stdout to the agent, so a
    # stray true/false line would pollute every reminder.
    local prompt="${1-}"
    if [ -z "$prompt" ]; then
        return 1
    fi
    # Resolve the Python script path relative to this sourced file:
    # <repo>/.claude/hooks/_gate_regexes.sh ->
    # <repo>/scripts/kb/_relevance.py. Avoids depending on git being available
    # or on a particular working directory.
    local _hook_dir _repo_root _script result
    _hook_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    _repo_root="$(dirname "$(dirname "$_hook_dir")")"
    _script="$_repo_root/scripts/kb/_relevance.py"
    result="$(printf '%s' "$prompt" | python3 "$_script" --classify-stdin 2>/dev/null)"
    [ "$result" = "true" ]
}
'''



if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emit-bash", action="store_true",
                   help="Print the generated bash fallback wrapper.")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--classify", default=None,
                       help="Classify a single prompt (prints 'true'/'false').")
    group.add_argument("--classify-stdin", action="store_true",
                       help="Classify UTF-8 prompt text read from stdin.")
    args = p.parse_args()

    if args.emit_bash:
        print(_emit_bash(), end="")
    elif args.classify is not None:
        print("true" if is_kb_relevant(args.classify) else "false")
    elif args.classify_stdin:
        print("true" if is_kb_relevant(sys.stdin.read()) else "false")
    else:
        p.print_help()
        sys.exit(0)
