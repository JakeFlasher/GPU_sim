#!/usr/bin/env python3
"""
route_query.py — opt-in project knowledge router.

Classifies a prompt against the KB manifest (.claude/kb_manifest.sqlite) and
prints the top matching cards plus a skill destination. It is an ordinary CLI:
it is not installed as a Claude/Codex hook and it never intercepts a session.

Usage:
  python3 scripts/kb/route_query.py --prompt "..." [--top N]
  printf '%s\n' "..." | python3 scripts/kb/route_query.py [--top N]

Skill mapping is loaded from `scripts/kb/routing_contract.py`:
  ptx, inline-ptx        -> interpreting-ptx-isa
  sass                   -> analyzing-sass-nvbit-traces
  cuda, best-practices   -> optimizing-cuda-kernels
  blackwell              -> tuning-blackwell-sm120-kernels
  measurement            -> validating-nonroot-gpu-measurement
  patents                -> reasoning-about-nvidia-architecture-history
  simulator              -> designing-gpu-simulator
  nvcc                   -> compiling-cuda-artifacts
  sources, patterns,
    problems, cutlass_constructs -> implementing-sol-baselines
  _root                  -> navigating-project-kb
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path
from typing import Optional

# Add scripts/kb/ to import path so we can import _relevance.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from _relevance import is_kb_relevant  # noqa: E402
from routing_contract import SUBCORPUS_TO_SKILL  # noqa: E402


# Context-sensitive per-card skill override. Some cards have a primary
# physical subcorpus (where they live on disk) but split-personality
# ownership: depending on the prompt's angle, a different SKILL.md is
# canonically responsible. Each rule is `(pattern, skill)` — first match
# wins; no match falls through to the subcorpus default.
#
# - nvbit-injection-playbook.md lives under measurement/ but is owned by
#   analyzing-sass-nvbit-traces when the failure mode is binary/SASS
#   (version mismatch, opcode, decode, disassembly). For non-SASS angles
#   ("preflight manifest detecting NVBit"), measurement remains the historical
#   subcorpus owner.
# - nvarchsim-vs-this-project.md is owned by designing-gpu-simulator when
#   the prompt asks about NVArchSim as a baseline / for the new
#   simulator. For "wall-clock-primary thesis"-style questions it stays
#   under the retained historical validating-nonroot-gpu-measurement pointer.
CARD_OVERRIDE_RULES: dict[str, list[tuple[re.Pattern, str]]] = {
    ".claude/knowledge/measurement/nvbit-injection-playbook.md": [
        (re.compile(
            r"\b(?:binary.?(?:version|mismatch)|version.?mismatch|"
            r"opcode|decode|disassembl|sass|cuobjdump|nvdisasm|cu\+\+filt|"
            r"nvprune|trace.?replay|instrumentation\s+handler)\b",
            re.IGNORECASE),
         "analyzing-sass-nvbit-traces"),
    ],
    ".claude/knowledge/measurement/nvarchsim-vs-this-project.md": [
        (re.compile(
            r"\bnvarchsim\b.{0,80}\b(?:baseline|design|build|"
            r"simulator|new|future|target|good)\b|"
            r"\b(?:our|new|future|build\s+a|design).{0,40}\bsimulator\b"
            r".{0,80}\bnvarchsim\b|"
            r"\bobjects?\s+(?:must|should|to)\s+(?:i|we)\s+model\b",
            re.IGNORECASE | re.DOTALL),
         "designing-gpu-simulator"),
    ],
}


def _resolve_skill_for_card(card_path: str, subcorpus: str,
                            prompt: str,
                            default_skill_md: str = "") -> Optional[str]:
    """Return the skill that owns this card *for this prompt*.

    Resolution order (Codex-review revision):
      1. CARD_OVERRIDE_RULES — Python-side, prompt-context-aware. For
         split-personality cards where ownership genuinely depends on
         the prompt's angle (e.g., nvbit-injection-playbook.md is owned
         by analyzing-sass-nvbit-traces for binary/SASS questions but
         by the retained historical validating-nonroot-gpu-measurement pointer
         for preflight questions).
      2. `Default skill:` frontmatter — declarative ownership for cards
         whose canonical Skill differs from their physical subcorpus
         (e.g., a card that lives in `measurement/` but is owned by
         `analyzing-sass-nvbit-traces` could declare it in metadata
         instead of needing a Python-side rule).
      3. SUBCORPUS_TO_SKILL fallback.
    """
    rules = CARD_OVERRIDE_RULES.get(card_path)
    if rules:
        for pattern, skill in rules:
            if pattern.search(prompt):
                return skill
    if default_skill_md:
        skill = default_skill_md.strip()
        # Defensive: only honor if it's in the canonical skill list
        # (avoid typos in frontmatter silently routing to nonsense).
        if skill in SUBCORPUS_TO_SKILL.values():
            return skill
    return SUBCORPUS_TO_SKILL.get(subcorpus)

# Meta-router intent: prompts that ask FOR THE INVENTORY (which skill?
# what's in the KB?), not for technical content. When any of these
# patterns fires we force INDEX.md to the top result and return
# navigating-project-kb as the primary skill.
#
# IMPORTANT: every pattern must REQUIRE a KB-vocabulary token (skill,
# card, kb, knowledge base, router, inventory, INDEX, specialty,
# meta-router). Bare "where do I start" without KB vocabulary is too
# loose — it pulls "where do I start with cp.async" into the meta-
# router instead of interpreting-ptx-isa.
_META_ROUTER_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bspecialty\s+skills?\b", re.IGNORECASE),
    re.compile(r"\bavailable\s+(?:specialty\s+)?skills?\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+skill\s+(?:should|do|to|would|can)\b", re.IGNORECASE),
    re.compile(r"\bwhich\s+skill\s+(?:should|do|to|owns|covers)\b", re.IGNORECASE),
    re.compile(r"\bknowledge\s+base\s+structure\b", re.IGNORECASE),
    re.compile(r"\bproject\s+(?:knowledge\s+base|kb)\b", re.IGNORECASE),
    re.compile(r"\boverview\s+of\s+the\s+(?:project\s+)?(?:knowledge\s+base|kb)\b",
               re.IGNORECASE),
    # "unsure whether X or Y" requires X/Y to be a KB/skill token, not
    # a free-form topic word — otherwise "unsure whether to use cp.async
    # or ldmatrix" would force-route to INDEX.
    re.compile(r"\bunsure\s+(?:whether|which|if).{0,40}\b(?:skill|"
               r"card|kb|knowledge|router|specialty|inventory)\b",
               re.IGNORECASE | re.DOTALL),
    re.compile(r"\b(?:list|show|name)\s+(?:the\s+)?(?:available\s+)?"
               r"(?:specialty\s+)?skills?\b", re.IGNORECASE),
    re.compile(r"\bINDEX\.md\b"),
    re.compile(r"\bmeta.?router\b", re.IGNORECASE),
    re.compile(r"\bshortest\s+path\s+to\s+(?:the\s+)?(?:right\s+)?card\b",
               re.IGNORECASE),
    re.compile(r"\b(?:right|correct|best)\s+(?:card|skill)\s+(?:to|for|that)\b",
               re.IGNORECASE),
    re.compile(r"\bwhich\s+card\s+(?:should|do|to|covers|owns|explains)\b",
               re.IGNORECASE),
    # "where do/should I start" only fires meta-routing when paired with
    # KB-vocabulary in the same prompt. Two-stage: this pattern plus
    # any of (skill|card|kb|knowledge\s+base|router|specialty|inventory|
    # ptx\s+or\s+sass) elsewhere in the prompt.
    re.compile(
        r"\bwhere\s+(?:do\s+i|should\s+i|to)\s+start\b"
        r".{0,200}\b(?:skill|card|kb|knowledge\s+base|router|specialty|"
        r"inventory|ptx\s+or\s+sass|sass\s+or\s+ptx)\b|"
        r"\b(?:skill|card|kb|knowledge\s+base|router|specialty|inventory|"
        r"ptx\s+or\s+sass|sass\s+or\s+ptx)\b.{0,200}"
        r"\bwhere\s+(?:do\s+i|should\s+i|to)\s+start\b",
        re.IGNORECASE | re.DOTALL),
]


def _is_meta_router_intent(prompt: str) -> bool:
    return any(p.search(prompt) for p in _META_ROUTER_PATTERNS)


# Skill-anchor tokens. When the prompt contains any of these (case-insensitive,
# word-bounded), boost candidate cards in the listed subcorpora by `boost`.
# Multiple anchors stack multiplicatively (cap at SUBCORPUS_BOOST_CAP).
SUBCORPUS_BOOST_CAP = 6.0
SKILL_ANCHORS: list[tuple[re.Pattern, list[str], float]] = [
    # SASS / NVBit / SASS tooling — utc* enumerated to avoid matching the
    # English word "UTC" (timezone). Mnemonics from sass-instruction-reference.md.
    (re.compile(r"\b(?:sass|cuobjdump|nvdisasm|cu\+\+filt|nvprune|nvbit|"
                r"omma|qmma|warpgroup|wgmma|hgmma|igmma|qgmma|"
                r"bgmma|hmma|imma|dmma|bmma|wmma|ldgsts|ldsm|"
                r"utmacmdflush|utmaldg|utmastg|utmaredg|utmacctl|utmapf|"
                r"ucgabar_arv|ucgabar_wait|"
                r"utccp|utcshift|utcbar|utchmma|utcomma|utcqmma|"
                r"ldt|ldtm|stt|sttm|ublkcp|ublkpf|ublkred)\b",
                re.IGNORECASE),
     ["sass"], 3.0),
    (re.compile(r"\b(?:disassembl|decode|opcode|instruction.?mix|"
                r"unknown.?opcode|trace.?replay|binary.?instrument)\b",
                re.IGNORECASE),
     ["sass"], 2.0),

    # nvcc / compilation — strong, nvcc-only. Add gencode/fallback/opt-in
    # phrasing so prompts like "-gencode for sm_120 with PTX fallback" route
    # to nvcc (even when sm_120 also boosts blackwell).
    (re.compile(r"(?:-gencode|-arch=|-Xptxas|-keep|-lineinfo|-dlto|"
                r"-rdc|-maxrregcount|--?ptx-options)", re.IGNORECASE),
     ["nvcc"], 4.0),
    (re.compile(r"\b(?:nvcc|fatbin|cubin|separate.?compilation|jit\b|"
                r"ptxas|cudafe|cudart|cuobjdump|nvprune|extract.?ptx|"
                r"extract.?cubin|fatbinary|gencode|virtual.?arch|"
                r"ptx.?fallback|arch.?conditional|opt.?in)\b",
                re.IGNORECASE),
     ["nvcc"], 3.0),
    # `compute_NN` — virtual architecture suffix; nvcc-specific. `sm_NN`
    # alone is too generic (could be PTX/blackwell prompt).
    (re.compile(r"\bcompute_[0-9]+a?\b", re.IGNORECASE),
     ["nvcc"], 1.5),
    # `sm_NNa` (architecture-conditional suffix-a) is fundamentally an
    # nvcc opt-in flag (`-gencode arch=compute_NNa,code=sm_NNa`). Push
    # nvcc strongly; do not co-boost blackwell (the `blackwell` /
    # `sm_NN`-family anchors below already cover that axis when relevant).
    (re.compile(r"\bsm_?[0-9]{2,3}a\b", re.IGNORECASE),
     ["nvcc"], 3.5),
    # Compile-artifact extraction phrasing (extract / dump PTX or cubin
    # out of a fatbin) is unambiguously a compilation-pipeline question.
    # Strong nvcc boost so it beats the bare `ptx` anchor.
    (re.compile(r"\b(?:extract|dump|inspect|view|read|get|disassembl)"
                r"\W{0,10}(?:ptx|cubin|sass)\b|"
                r"\bfatbin\b.{0,40}\b(?:contents|inside|inner|layout|"
                r"sections?|slices?|extract|dump|inspect)\b",
                re.IGNORECASE | re.DOTALL),
     ["nvcc"], 5.0),

    # Measurement / non-root contract
    (re.compile(r"\b(?:cupti|cupti.?activity|err_nvgpuctrperm|"
                r"restrictprofiling|nvbit|clock64|globaltimer|preflight|"
                r"non.?root|wall.?clock|trace.?budget|perturbation.?sweep|"
                r"counter.?free|counter.?less|nvarchsim)\b", re.IGNORECASE),
     ["measurement"], 3.0),
    (re.compile(r"\bncu\b", re.IGNORECASE),
     ["measurement"], 4.0),

    # PTX / inline-PTX
    (re.compile(r"\b(?:ptx|inline.?ptx|asm.?volatile|wgmma\.mma_async|"
                r"tcgen05\.mma|cp\.async|cp\.async\.bulk|mbarrier|"
                r"ldmatrix|stmatrix|bar\.sync|fence\.[a-z_]+|setp|"
                r"cvt\.[a-z0-9.]+|prmt|popc|brev|bfe|bfi|shfl\.sync|"
                r"vote\.sync|ballot\.sync|match\.sync|setmaxnreg|"
                r"nanosleep)\b", re.IGNORECASE),
     ["ptx", "inline-ptx"], 3.0),

    # Blackwell / SM120 / SM100 — strong on Blackwell-specific tokens.
    # Add 5090/5080 (consumer Blackwell SKUs commonly referenced in forum FUD).
    (re.compile(r"\b(?:blackwell|rtx ?pro ?6000|gb20[0-9]|gb10[0-9]|"
                r"\bb[12]00\b|tmem|tcgen05|nvfp4|cta.?pair|nv.?hbi|nvl72|"
                r"\b50[89]0\b|rtx ?50[89]0)\b",
                re.IGNORECASE),
     ["blackwell"], 3.0),
    # sm_120 / sm_100 SM family — only modestly boost blackwell (these
    # tokens also legitimately appear in nvcc / compatibility / PTX prompts).
    (re.compile(r"\bsm_?12[0-9]a?\b|\bsm_?10[0-9]a?\b", re.IGNORECASE),
     ["blackwell"], 1.7),
    # FP4 / FP6 / NVFP4 / MX — Blackwell-era precisions; co-boost ptx since
    # those bytes have PTX-level encoding too.
    (re.compile(r"\bfp4\b|\bfp6\b|nvfp4|mxfp[468]?\b", re.IGNORECASE),
     ["blackwell", "ptx"], 2.0),

    # Cluster / DSMEM (CUDA programming model)
    (re.compile(r"\b(?:cluster\.sync|cluster\.launch|dsmem|"
                r"distributed.?shared|cooperative.?group|"
                r"cudaLaunchKernelEx|cudaFuncAttributeNonPortableCluster"
                r"SizeAllowed|barrier\.cluster|cga\b)\b", re.IGNORECASE),
     ["cuda", "blackwell"], 3.0),

    # Generic CUDA tuning
    (re.compile(r"\b(?:occupanc|coalesc|bank.?conflicts?|register.?(?:pressure|"
                r"spill|file)|warp.?(?:scheduler|divergence)|__launch_bounds__|"
                r"__forceinline__|maxrregcount)\b", re.IGNORECASE),
     ["cuda", "best-practices"], 2.5),

    # Architecture history / patents
    (re.compile(r"\b(?:patent|three.?rewrites|fermi.?rewrite|"
                r"volta.?rewrite|hopper.?rewrite|architectural.?evolution|"
                r"historical|TMAU|US[0-9]{7,}|EP[0-9]{7,})\b", re.IGNORECASE),
     ["patents"], 3.0),
    # Glossary / multi-generation definition queries — "define BSSY, DSMEM,
    # HMMA, TMEM and which generation introduced each", "what changed in
    # warp scheduling from Turing to Ampere", "evolution of NVLink".
    (re.compile(
        r"\bwhich\s+generation\s+introduced|"
        r"\b(?:define|definition\s+of|glossary|terminology)\b.{0,80}\b"
        r"(?:bssy|bsync|dsmem|hmma|omma|qmma|tmem|wgmma|tcgen05|cga|nvfp4)\b|"
        r"\bevolution\s+of\s+(?:nvlink|nvswitch|tensor\s+core|sm|warp|"
        r"sync|interconnect|memory|precision|execution)|"
        r"\bwhat\s+changed\b.{0,40}\bfrom\s+(?:tesla|fermi|kepler|maxwell|"
        r"pascal|volta|turing|ampere|ada|hopper|blackwell|rubin)\b.{0,40}\b"
        r"to\s+(?:tesla|fermi|kepler|maxwell|pascal|volta|turing|ampere|"
        r"ada|hopper|blackwell|rubin)\b",
        re.IGNORECASE | re.DOTALL),
     ["patents"], 4.0),

    # Simulator design
    (re.compile(r"\b(?:simulator|nvasim|event.?driven|gpgpu.?sim|"
                r"accel.?sim|clustersim|modeling.?paper|validation.?contract|"
                r"decision.?quality|nexosim|priority.?queue.?des)\b",
                re.IGNORECASE),
     ["simulator"], 3.0),
    # Simulator-design verb-shape: "what objects must I model in a
    # Blackwell simulator", "design an event-driven simulator",
    # "build / architect / model the simulator". Push simulator harder
    # than the bare `\bsimulator\b` anchor so it beats blackwell when
    # both fire.
    (re.compile(
        r"\bsimulator\b.{0,80}\b(?:design|model|build|architect|objects?|"
        r"primitives?|abstraction|hierarch|baseline|new|future|target)\b|"
        r"\b(?:design|model|build|architect|new|future)\b.{0,40}"
        r"\bsimulator\b|"
        r"\bwhat.{0,15}\bobjects?\b.{0,40}\bsimulator\b|"
        r"\bobjects?\s+(?:must|should|to|do)\b.{0,20}\bmodel\b",
        re.IGNORECASE | re.DOTALL),
     ["simulator"], 1.7),

    # --- SOL-ExecBench kernel-pattern anchors (merged from the dissolved
    # solbench router) — route to implementing-sol-baselines via patterns/
    # + problems/. ---
    (re.compile(
        r"\b(?:gemm|rmsnorm|rope|moe|gqa|swiglu|paged.?attention|"
        r"fused.?add|fused.?rms|mla|ssm|mamba)\b",
        re.IGNORECASE),
     ["patterns", "problems"], 3.0),
    # SOL-ExecBench problem-ID anchor — boost problems/ when the prompt names
    # a specific `problem_NNN_<slug>` (no trailing \b: `_` is a word char).
    (re.compile(r"\bproblem_[0-9]{3,}", re.IGNORECASE),
     ["problems"], 4.0),
    # `attention` only boosts when it co-occurs with a technical compose-word.
    (re.compile(
        r"\battention\b.{0,40}\b(?:output|projection|qkv|softmax|dropout|"
        r"rope|paged|kv|forward|backward|fused|multi.?head)\b|"
        r"\b(?:output|projection|qkv|softmax|dropout|rope|paged|kv|"
        r"forward|backward|fused|multi.?head)\b.{0,40}\battention\b",
        re.IGNORECASE | re.DOTALL),
     ["patterns", "problems"], 2.0),
    # FP8 / NVFP4 quantization anchors.
    (re.compile(r"\b(?:fp8|nvfp4|block.?scale|e4m3|e5m2|e2m1)\b",
                re.IGNORECASE),
     ["patterns", "problems"], 2.5),
    # CUTLASS/CuTe layout constructs are a distinct authoring corpus. Keep
    # this compound anchor narrow so generic "layout" questions do not drift
    # away from their normal source routing.
    (re.compile(
        r"\b(?:cutlass|cute)\b.{0,40}\blayout\b|"
        r"\blayout\b.{0,40}\b(?:cutlass|cute)\b|"
        r"(?:^|[_\W])block.?scaled.?layout\b",
        re.IGNORECASE),
     ["cutlass_constructs"], 4.0),
]


# Anti-anchors — suppress a positive boost when the prompt's intent is
# clearly *not* about that subcorpus, even though one of its anchor
# tokens fired. Each rule: `(pattern, subcorpora-to-cap, cap-value)`.
# Applied AFTER positive boosts; cap the per-subcorpus value to no more
# than `cap-value` if the pattern fires.
SKILL_ANTI_ANCHORS: list[tuple[re.Pattern, list[str], float]] = [
    # When the prompt mentions a fatbin / cubin / cuobjdump artifact, the
    # bare `ptx` anchor firing is a false positive — the question is
    # about the compilation pipeline, not PTX semantics. Cap ptx /
    # inline-ptx so nvcc wins for "extract PTX from a fatbin"-style
    # prompts.
    (re.compile(r"\b(?:fatbin|cubin|cuobjdump|nvprune|cu\.o|\.fatbin\b)\b",
                re.IGNORECASE),
     ["ptx", "inline-ptx"], 1.0),
    # When the prompt is fundamentally about simulator design, the
    # blackwell anchor (firing because "Blackwell simulator" is mentioned)
    # is secondary. Cap blackwell so simulator wins.
    (re.compile(
        r"\bsimulator\b.{0,80}\b(?:design|model|build|architect|objects?|"
        r"primitives?|abstraction|hierarch)\b|"
        r"\b(?:design|model|build|architect)\b.{0,40}\bsimulator\b|"
        r"\bwhat.{0,15}\bobjects?\b.{0,40}\bsimulator\b|"
        r"\bobjects?\s+(?:must|should|to|do)\b.{0,20}\bmodel\b",
        re.IGNORECASE | re.DOTALL),
     ["blackwell"], 1.5),
]


# Relevance classification is imported from _relevance.py so the router and
# its standalone classifier share one implementation.


def _compute_subcorpus_boosts(prompt: str) -> dict[str, float]:
    """Multiply boosts for each subcorpus the prompt's anchor tokens hit.

    Then apply anti-anchors that cap a subcorpus boost when the prompt's
    intent clearly belongs to a different subcorpus (e.g., "extract PTX
    from a fatbin" caps ptx so nvcc wins).
    """
    boosts: dict[str, float] = {}
    for pattern, subcorpora, b in SKILL_ANCHORS:
        if pattern.search(prompt):
            for sc in subcorpora:
                cur = boosts.get(sc, 1.0)
                boosts[sc] = min(cur * b, SUBCORPUS_BOOST_CAP)
    for pattern, subcorpora, cap in SKILL_ANTI_ANCHORS:
        if pattern.search(prompt):
            for sc in subcorpora:
                if sc in boosts and boosts[sc] > cap:
                    boosts[sc] = cap
    return boosts


# FTS5 column order in cards_fts (matches build_manifest.py schema v2):
#   0: path (UNINDEXED, weight ignored)
#   1: name
#   2: title
#   3: use_when
#   4: primary_raw_source
#   5: supporting_sources
#   6: headings
#   7: routing_tags
# Weighted bm25() scoring: heaviest weight on `use_when` (the explicit
# trigger frontmatter field), then curated name/title/routing tags, then
# headings and raw-source pointers. Routing tags are intentionally indexed;
# lint_cards.py's anchor rule depends on this contract.
_BM25_WEIGHTS = "0.0, 4.0, 3.0, 6.0, 1.5, 0.5, 1.0, 3.0"

_SUPPORTED_MANIFEST_SCHEMA_VERSIONS = {1, 2}
_REQUIRED_FTS_COLUMNS_V2 = (
    "path",
    "name",
    "title",
    "use_when",
    "primary_raw_source",
    "supporting_sources",
    "headings",
    "routing_tags",
)


def _manifest_bm25_weights(conn: sqlite3.Connection) -> str | None:
    """Return weights for a known manifest schema, or reject it.

    Version 1 is retained for old, regenerable manifests that predate routing
    tags.  Version 2 is the canonical layout.  Metadata is optional on legacy
    artifacts, so an absent/empty ``manifest_meta`` table is resolved from the
    exact FTS column shape; an explicit version must agree with that shape.
    Unknown versions and malformed layouts fail closed instead of guessing SQL
    columns or silently dropping tags.
    """
    try:
        columns = tuple(item[1] for item in conn.execute("PRAGMA table_info(cards_fts)"))
    except sqlite3.Error:
        return None

    v1_columns = _REQUIRED_FTS_COLUMNS_V2[:-1]
    if columns == v1_columns:
        inferred_version = 1
        weights = "0.0, 4.0, 3.0, 6.0, 1.5, 0.5, 1.0"
    elif columns == _REQUIRED_FTS_COLUMNS_V2:
        # Historical tag-aware manifests were emitted before manifest_meta was
        # added; they still have the canonical eight-column BM25 shape.
        inferred_version = 2
        weights = _BM25_WEIGHTS
    else:
        return None

    try:
        row = conn.execute(
            "SELECT value FROM manifest_meta WHERE key = 'schema_version'"
        ).fetchone()
    except sqlite3.Error:
        # No metadata table is a supported legacy state.  The exact FTS shape
        # above is the authority in that case.
        row = None

    if row is not None:
        try:
            version = int(row[0])
        except (TypeError, ValueError):
            return None
        if version not in _SUPPORTED_MANIFEST_SCHEMA_VERSIONS or version != inferred_version:
            return None
    return weights


def _fts_query(prompt: str, manifest: Path, top_n: int) -> list[dict]:
    if not manifest.is_file():
        return []
    try:
        conn = sqlite3.connect(manifest)
    except sqlite3.Error:
        return []
    weights = _manifest_bm25_weights(conn)
    if weights is None:
        conn.close()
        return []

    tokens = _tokenize_for_fts(prompt)
    if not tokens:
        conn.close()
        return []

    fts_query = " OR ".join(tokens)
    # Pull a wider candidate pool (3*top_n + 6) so subcorpus-boost re-ranking
    # has enough candidates to promote the right card from rank-4..10 into
    # top_n. FTS5 bm25() returns LOWER = better.
    candidate_n = top_n * 3 + 6
    # JOIN to cards table so we can read the optional `default_skill`
    # frontmatter field. Cards without that field get the empty string
    # and fall through to subcorpus / per-card-rule resolution.
    sql = (
        f"SELECT cards_fts.path, cards_fts.name, cards_fts.title, "
        f"cards_fts.use_when, bm25(cards_fts, {weights}) AS score, "
        f"COALESCE(cards.default_skill, '') AS default_skill "
        f"FROM cards_fts LEFT JOIN cards ON cards.path = cards_fts.path "
        f"WHERE cards_fts MATCH ? ORDER BY score LIMIT ?"
    )
    try:
        rows = conn.execute(sql, (fts_query, candidate_n)).fetchall()
    except sqlite3.Error:
        rows = []
    conn.close()

    if not rows:
        return []

    boosts = _compute_subcorpus_boosts(prompt)

    candidates = []
    for r in rows:
        path = r[0]
        parts = path.split("/")
        subcorpus = parts[2] if len(parts) >= 4 and parts[1] == "knowledge" else "_root"
        raw_score = r[4] if r[4] is not None else 0.0
        default_skill_md = r[5] if len(r) > 5 else ""
        # bm25() is negative-better in FTS5 (closer to 0 = better). Convert to
        # positive-better and apply subcorpus multiplier.
        positive = -raw_score
        boost = boosts.get(subcorpus, 1.0)
        adjusted = positive * boost
        # Skill resolution order:
        #   1. CARD_OVERRIDE_RULES (Python-side, prompt-context-aware) —
        #      for genuinely contextual ownership changes.
        #   2. `Default skill:` frontmatter on the card — declarative,
        #      lives with the card (Codex-review preference for
        #      cross-skill ownership metadata).
        #   3. SUBCORPUS_TO_SKILL fallback (the subcorpus-default mapping).
        skill = _resolve_skill_for_card(
            path, subcorpus, prompt, default_skill_md=default_skill_md
        )
        candidates.append(
            {
                "path": path,
                "name": r[1],
                "title": r[2],
                "use_when": r[3],
                "subcorpus": subcorpus,
                "skill": skill,
                "_raw_score": positive,
                "_boost": boost,
                "_adjusted": adjusted,
            }
        )

    candidates.sort(key=lambda c: c["_adjusted"], reverse=True)
    return candidates[:top_n]


def _inject_meta_router_top(matches: list[dict], top_n: int,
                            manifest: Path) -> list[dict]:
    """Force INDEX.md to the top of the results list.

    Used when _is_meta_router_intent(prompt) is True. INDEX.md may already
    be in `matches` (in which case we just promote it) or absent (we look
    it up in the manifest and prepend it).
    """
    index_path = ".claude/knowledge/INDEX.md"
    promoted = None
    rest: list[dict] = []
    for m in matches:
        if m["path"] == index_path and promoted is None:
            promoted = m
        else:
            rest.append(m)

    if promoted is None and manifest.is_file():
        try:
            conn = sqlite3.connect(manifest)
            row = conn.execute(
                "SELECT path, name, title, use_when FROM cards "
                "WHERE path = ?", (index_path,)
            ).fetchone()
            conn.close()
            if row:
                promoted = {
                    "path": row[0],
                    "name": row[1],
                    "title": row[2],
                    "use_when": row[3],
                    "subcorpus": "_root",
                    "skill": "navigating-project-kb",
                    "_raw_score": 0.0,
                    "_boost": 1.0,
                    "_adjusted": 0.0,
                }
        except sqlite3.Error:
            pass

    if promoted is None:
        return matches

    promoted["skill"] = "navigating-project-kb"
    return [promoted, *rest][:top_n]


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_FTS_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "be", "what", "how", "why", "when", "where", "do", "does",
    "this", "that", "i", "my", "me", "you", "your", "we", "us", "it",
    "as", "by", "at", "if", "but", "not", "no", "so", "can", "should",
    "explain", "show", "tell", "help", "use", "using", "compare",
}


def _tokenize_for_fts(text: str) -> list[str]:
    """Pull alphanumeric tokens, drop stopwords/short, dedupe, cap to 12.

    Quote tokens with non-alphanumeric edges so FTS5 doesn't choke. Strip
    leading/trailing punctuation. Underscore in `__syncthreads`/`sm_120` is
    preserved.
    """
    raw = _TOKEN_RE.findall(text)
    out: list[str] = []
    seen: set[str] = set()
    for tok in raw:
        low = tok.lower()
        if len(low) < 2:
            continue
        if low in _FTS_STOPWORDS:
            continue
        if low.isdigit():
            continue
        if low in seen:
            continue
        seen.add(low)
        out.append(f'"{low}"')
        if len(out) >= 12:
            break
    return out


def _format_skill_label(skill: str) -> str:
    """Render a skill name for the opt-in routing result."""
    return f"`{skill}`"


def _format_reminder(matches: list[dict], skills: list[str]) -> str:
    """Compact KB-router reminder block.

    Format goal: ~5 lines of stdout per fire (down from ~12). Prints only
    the card paths (skill annotated only when distinct from rank-1) and a
    single citation directive. Titles dropped — `path.md` already encodes
    the topic, and the agent reads the card body anyway.
    """
    primary = skills[0] if skills else "navigating-project-kb"
    lines = [f"[KB] Skill: {_format_skill_label(primary)}. Open + cite under `## Knowledge Consulted`:"]
    for m in matches:
        skill = m.get("skill") or "navigating-project-kb"
        if skill == primary:
            lines.append(f"  - `{m['path']}`")
        else:
            lines.append(f"  - `{m['path']}` ({_format_skill_label(skill)})")
    return "\n".join(lines)


def _explain(prompt: str, manifest: Path, top_n: int) -> str:
    """Verbose decision trace for debugging routing misses.

    Prints, in order: relevance gate result + which group fired; FTS
    tokens after stopword filter; raw BM25 candidate pool with scores;
    subcorpus boosts (positive anchors); anti-anchors that capped any
    boosts; meta-router intent fired (if any); per-card override rules
    that fired (if any); final ranking with adjusted scores. Useful as
    `python3 scripts/kb/route_query.py --explain --prompt 'foo'`.
    """
    out: list[str] = []
    out.append(f"=== EXPLAIN: {prompt!r}")
    out.append("")

    # Stage 1: relevance gate.
    relevant = is_kb_relevant(prompt)
    out.append(f"[1] Relevance gate: {'PASS' if relevant else 'FAIL (silent exit)'}")
    if not relevant:
        # Probe individual groups for diagnostic clarity.
        from _relevance import GROUP_A, GROUP_B, GROUP_C, _CODENAME_FP
        a = bool(GROUP_A.search(prompt))
        b = bool(GROUP_B.search(prompt))
        c = bool(GROUP_C.search(prompt))
        fp = bool(_CODENAME_FP.search(prompt))
        out.append(f"    Group A: {a}, Group B: {b}, Group C: {c}, "
                   f"CODENAME_FP: {fp}")
        out.append("    -> No further stages run.")
        return "\n".join(out)
    out.append("")

    # Stage 2: FTS5 tokens.
    tokens = _tokenize_for_fts(prompt)
    out.append(f"[2] FTS5 tokens (stopwords filtered, ≤12): {tokens}")
    out.append("")

    # Stage 3: raw BM25 candidates (before any boost).
    if not manifest.is_file():
        out.append(f"[3] Manifest missing at {manifest}; skipping.")
        return "\n".join(out)
    fts_query = " OR ".join(tokens) if tokens else ""
    raw_rows: list[tuple] = []
    if fts_query:
        try:
            conn = sqlite3.connect(manifest)
            weights = _manifest_bm25_weights(conn)
            if weights is None:
                conn.close()
                out.append("[3] Manifest schema unsupported; falling back.")
                return "\n".join(out)
            sql = (
                f"SELECT path, bm25(cards_fts, {weights}) "
                f"FROM cards_fts WHERE cards_fts MATCH ? "
                f"ORDER BY 2 LIMIT ?"
            )
            raw_rows = conn.execute(sql, (fts_query, top_n * 3 + 6)).fetchall()
            conn.close()
        except sqlite3.Error as e:
            out.append(f"[3] FTS error: {e}")
            return "\n".join(out)
    out.append(f"[3] Raw BM25 candidates (negative-better; {len(raw_rows)}):")
    for path, score in raw_rows[:10]:
        out.append(f"      bm25={-score:6.2f}   {path}")
    if len(raw_rows) > 10:
        out.append(f"      ... +{len(raw_rows) - 10} more")
    out.append("")

    # Stage 4: subcorpus boosts (positive anchors).
    boosts_pos: dict[str, float] = {}
    for pattern, subcorpora, b in SKILL_ANCHORS:
        if pattern.search(prompt):
            for sc in subcorpora:
                cur = boosts_pos.get(sc, 1.0)
                boosts_pos[sc] = min(cur * b, SUBCORPUS_BOOST_CAP)
            out.append(f"    + anchor {pattern.pattern[:60]!r:65s} -> "
                       f"{subcorpora} ×{b}")
    out.append("[4] Subcorpus boosts (after positive anchors only):")
    for sc, b in sorted(boosts_pos.items()):
        out.append(f"      {sc:14s} = {b:.2f}")
    out.append("")

    # Stage 5: anti-anchors.
    boosts_final = dict(boosts_pos)
    fired_anti: list[str] = []
    for pattern, subcorpora, cap in SKILL_ANTI_ANCHORS:
        if pattern.search(prompt):
            fired_anti.append(f"{pattern.pattern[:60]!r} caps {subcorpora} "
                              f"-> max {cap}")
            for sc in subcorpora:
                if sc in boosts_final and boosts_final[sc] > cap:
                    boosts_final[sc] = cap
    out.append(f"[5] Anti-anchors fired: {len(fired_anti)}")
    for entry in fired_anti:
        out.append(f"      {entry}")
    out.append("    Final boosts:")
    for sc, b in sorted(boosts_final.items()):
        out.append(f"      {sc:14s} = {b:.2f}")
    out.append("")

    # Stage 6: meta-router intent.
    meta_fired = _is_meta_router_intent(prompt)
    out.append(f"[6] Meta-router intent: {meta_fired}")
    out.append("")

    # Stage 7: per-card override rules.
    out.append("[7] Per-card overrides (rules that fired against this prompt):")
    fired_overrides = []
    for card_path, rules in CARD_OVERRIDE_RULES.items():
        for pattern, skill in rules:
            if pattern.search(prompt):
                fired_overrides.append(
                    f"{card_path:60s} -> skill={skill}")
    if fired_overrides:
        for line in fired_overrides:
            out.append(f"      {line}")
    else:
        out.append("      (none)")
    out.append("")

    # Stage 8: final ranking.
    matches = _fts_query(prompt, manifest, top_n)
    if meta_fired:
        matches = _inject_meta_router_top(matches, top_n, manifest)
    out.append(f"[8] Final ranking (top {top_n}):")
    for i, m in enumerate(matches, 1):
        skill = m.get("skill") or "?"
        out.append(
            f"      #{i}  raw={m['_raw_score']:6.2f}  "
            f"boost={m['_boost']:.2f}  adj={m['_adjusted']:6.2f}  "
            f"[{skill}]  {m['path']}"
        )

    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument(
        "--manifest",
        default=str(Path(".claude") / "kb_manifest.sqlite"),
    )
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--explain", action="store_true",
        help="Verbose decision trace mode (read prompt from --prompt or stdin).")
    parser.add_argument(
        "--prompt", default=None,
        help="Prompt to classify (otherwise read raw text from stdin).")
    args = parser.parse_args()

    repo_root = Path(args.root).resolve()
    manifest = (repo_root / args.manifest).resolve() if not Path(args.manifest).is_absolute() else Path(args.manifest)

    # --explain mode: read prompt from --prompt or stdin (raw text, not JSON),
    # print full decision trace, exit 0.
    if args.explain:
        if args.prompt is not None:
            prompt = args.prompt
        else:
            prompt = sys.stdin.read().strip()
        if not prompt:
            print("error: --explain requires --prompt or stdin text",
                  file=sys.stderr)
            return 1
        print(_explain(prompt, manifest, args.top))
        return 0

    prompt = args.prompt if args.prompt is not None else sys.stdin.read().strip()
    if not prompt:
        return 0

    # Relevance gate: silently exit if the prompt is not KB-relevant.
    # Without this, FTS5 fires on prompts like "ada lovelace was a
    # mathematician" because `ada` appears in card content. Shared
    # classifier in scripts/kb/_relevance.py.
    if not is_kb_relevant(prompt):
        return 0

    matches = _fts_query(prompt, manifest, args.top)

    # Meta-router intent: force INDEX.md to top so the meta-router skill
    # wins. Uses a curated pattern list (see _META_ROUTER_PATTERNS).
    if _is_meta_router_intent(prompt):
        matches = _inject_meta_router_top(matches, args.top, manifest)

    if not matches:
        return 0

    skills_seen: list[str] = []
    for m in matches:
        s = m.get("skill")
        if s and s not in skills_seen:
            skills_seen.append(s)

    print(_format_reminder(matches, skills_seen))

    return 0


if __name__ == "__main__":
    sys.exit(main())
