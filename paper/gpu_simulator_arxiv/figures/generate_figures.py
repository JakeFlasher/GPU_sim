#!/usr/bin/env python3
"""Generate mechanism-first, vector SVG figures for the GPU replay paper.

The script deliberately uses only the Python standard library.  It keeps all
labels as SVG text (rather than rasterizing them) and uses redundant color,
shape, and line-style encodings for print accessibility.
"""
from __future__ import annotations

from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parent

INK = "#14233B"
MUTED = "#52657A"
GRID = "#D7E1EC"
BLUE = "#1769AA"
SKY = "#56B4E9"
TEAL = "#008F82"
PURPLE = "#7356A5"
CORAL = "#D95D39"
GOLD = "#E29B22"
GREEN = "#2C9362"
RED = "#B94652"
PALE_BLUE = "#EAF4FC"
PALE_TEAL = "#E7F6F2"
PALE_PURPLE = "#F0ECF8"
PALE_GOLD = "#FFF3D9"
PALE_CORAL = "#FDECE7"
PALE_GRAY = "#F5F8FB"


def esc(s: str) -> str:
    return escape(str(s), quote=True)


def text(x, y, s, size=22, fill=INK, weight=400, anchor="start", family="Arial", italic=False):
    style = f"font-family:{family};font-size:{size}px;font-weight:{weight};fill:{fill};"
    if italic:
        style += "font-style:italic;"
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" style="{style}">{esc(s)}</text>'


def rect(x, y, w, h, fill="white", stroke=GRID, sw=2, r=14, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'


def line(x1, y1, x2, y2, stroke=INK, sw=2, dash=None, marker=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#{marker})"' if marker else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d}{m}/>'


def path(d, stroke=INK, sw=2, fill="none", dash=None, marker=None):
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#{marker})"' if marker else ""
    return f'<path d="{d}" stroke="{stroke}" stroke-width="{sw}" fill="{fill}"{ds}{m}/>'


def circle(cx, cy, r, fill, stroke="none", sw=0):
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def header(title, subtitle):
    return text(72, 65, title, 38, INK, 700, family="Georgia") + text(74, 100, subtitle, 19, MUTED)


def svg_start(title, desc, w=1800, h=1050):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc">
<title id="title">{esc(title)}</title><desc id="desc">{esc(desc)}</desc>
<defs><marker id="arrowBlue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0 L10 5 L0 10z" fill="{BLUE}"/></marker><marker id="arrowCoral" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0 L10 5 L0 10z" fill="{CORAL}"/></marker></defs>
<rect width="{w}" height="{h}" fill="white"/>'''


def finish(caption, source):
    out = rect(55, 930, 1690, 88, "#EEF4FA", "#C9D9EA", 1.5, 12)
    out += text(78, 960, "Caption.", 16, INK, 700)
    out += text(176, 960, caption, 16, INK)
    out += text(78, 988, source, 13, MUTED)
    return out + "</svg>"


def novelty_stack():
    parts = [svg_start("The contribution is a trace-to-frontier evidence contract", "Dynamic evidence, causal replay, and independent NCU authority are connected without collapsing their semantics."), header("The contribution is a trace-to-frontier evidence contract", "Inherited ingredients are explicit; the integration and full-corpus denominator are the contribution.")]

    # left: evidence
    parts += [rect(65, 145, 470, 700, PALE_BLUE, BLUE, 2.5, 20), text(92, 190, "01  OBSERVED EVIDENCE", 22, BLUE, 700), text(92, 222, "what the recorder actually knows", 16, MUTED, 400, italic=True)]
    boxes = [(92, 270, 410, 110, "dynamic_trace.pb", "PC • masks • lane sectors • placement", SKY), (92, 420, 410, 110, "SASS control sidecar", "stall/yield • barriers • wait masks • forms", PURPLE), (92, 570, 410, 110, "architecture card", "geometry • latency • typed assumptions", GOLD)]
    for x, y, w, h, title, sub, color in boxes:
        parts += [rect(x, y, w, h, "white", color, 2, 14), circle(x + 27, y + 28, 11, color), text(x + 50, y + 35, title, 21, INK, 700), text(x + 28, y + 72, sub, 16, MUTED)]
    parts += [text(92, 760, "Identity-preserving join", 18, BLUE, 700), text(92, 790, "function + PC + warp + launch", 16, MUTED), path("M535 500 C585 500 585 500 640 500", BLUE, 4, marker="arrowBlue")]

    # center: replay
    parts += [rect(640, 145, 530, 700, PALE_TEAL, TEAL, 2.5, 20), text(670, 190, "02  CAUSAL REPLAY", 22, TEAL, 700), text(670, 222, "what the simulator contributes", 16, MUTED, 400, italic=True)]
    stages = [(685, 280, 440, 82, "canonicalize", "warp records + static control", BLUE), (685, 390, 440, 82, "issue + wake", "4 SMSP partitions • dependencies", PURPLE), (685, 500, 440, 82, "sector hierarchy", "L1 → shared L2 → service envelope", CORAL), (685, 610, 440, 82, "frontier ledger", "issue ≠ completion ≠ observation ≠ tail", GOLD)]
    for i, (x, y, w, h, title, sub, color) in enumerate(stages, 1):
        parts += [rect(x, y, w, h, "white", color, 2, 14), circle(x + 29, y + 28, 14, color), text(x + 29, y + 34, str(i), 16, "white", 700, "middle"), text(x + 58, y + 32, title, 20, INK, 700), text(x + 58, y + 61, sub, 15, MUTED)]
        if i < len(stages):
            parts += [line(x + 220, y + h, x + 220, y + h + 28, TEAL, 3, marker="arrowBlue")]
    parts += [rect(685, 750, 440, 58, TEAL, TEAL, 1, 12), text(905, 787, "deterministic next-event order", 18, "white", 700, "middle")]

    # right: authority
    parts += [rect(1260, 145, 475, 700, PALE_GOLD, GOLD, 2.5, 20), text(1290, 190, "03  INDEPENDENT AUTHORITY", 22, GOLD, 700), text(1290, 222, "what is measured after replay", 16, MUTED, 400, italic=True)]
    parts += [rect(1290, 280, 415, 100, "white", GREEN, 2, 14), text(1318, 318, "NCU profile directory", 21, INK, 700), text(1318, 350, "same capture identity • renamed kernel safe", 15, MUTED)]
    parts += [rect(1290, 420, 415, 100, "white", GOLD, 2, 14), text(1318, 458, "launch-order fallback", 21, INK, 700), text(1318, 490, "98 exact + 11,460 ordered SM120 joins", 15, MUTED)]
    parts += [rect(1290, 560, 415, 100, "white", RED, 2, 14), text(1318, 598, "metric decision", 21, INK, 700), text(1318, 630, "≤15% headline • >15% future work", 15, MUTED)]
    parts += [path("M1170 470 C1210 470 1210 330 1290 330", CORAL, 4, dash="10 8", marker="arrowCoral"), text(1190, 300, "observation-only", 15, CORAL, 700, anchor="middle")]

    # bottom callout
    parts += [rect(65, 865, 1670, 48, PALE_CORAL, CORAL, 1.5, 10), text(900, 897, "Novelty boundary: not first SASS tracing • not first event simulation • not proprietary RTL recovery", 17, CORAL, 700, "middle")]
    return "".join(parts) + finish("Dynamic records, static controls, sector state, named frontiers, and independent NCU joins form one inspectable contract. Inherited mechanisms are labeled rather than rebranded as new.", "Source: simulator crates, corrected 286/339 full-corpus joins, and SOTA audit in GPU_sims_notes.")


def sota_matrix():
    parts = [svg_start("Scope-aware positioning against modern GPU simulation", "Qualitative capability matrix; checkmarks mean the cited system reports the capability in a comparable scope, not a scalar ranking."), header("Scope-aware positioning against modern GPU simulation", "The project occupies an unusual intersection; headline errors from other papers are not pooled with our launch-level MAPE.")]
    cols = ["native\nSASS", "event /\ncausal", "sectored\nL1/L2", "NCU\npost-join", "full real\ncorpus", "typed\nunknowns"]
    rows = [
        ("GPGPU-Sim", ["—", "~", "✓", "—", "~", "—"], "execution-driven baseline"),
        ("Accel-Sim", ["✓", "~", "✓", "✓", "~", "—"], "validated SASS framework"),
        ("MGPUSim / Akita", ["—", "✓", "✓", "—", "~", "~"], "event/protocol model"),
        ("NVArchSim", ["~", "✓", "~", "~", "—", "~"], "industrial event-driven context"),
        ("Modern GPU core", ["✓", "~", "~", "✓", "~", "~"], "MICRO 2025 core fidelity"),
        ("Sim-FA / GTSim", ["~", "✓", "~", "✓", "~", "~"], "async / tile abstraction"),
        ("PipeWeave / TileSight", ["—", "~", "~", "✓", "✓", "~"], "analytical / learned / tile"),
        ("Our replay stack", ["✓", "✓", "✓", "✓", "✓", "✓"], "trace-to-frontier contract"),
    ]
    x0, y0 = 70, 170
    widths = [285, 150, 150, 150, 150, 150, 150]
    heights = 66
    # Header
    xx = x0
    for i, label in enumerate(["System / model family"] + cols):
        parts += [rect(xx, y0, widths[i], 74, INK if i == 0 else BLUE, INK, 1, 5)]
        lines = label.split("\n")
        for j, ln in enumerate(lines):
            parts += [text(xx + widths[i] / 2, y0 + 31 + j * 20, ln, 16, "white", 700, "middle")]
        xx += widths[i]
    y = y0 + 74
    for ridx, (name, vals, note) in enumerate(rows):
        fill = PALE_GOLD if "Our replay" in name else ("white" if ridx % 2 == 0 else PALE_GRAY)
        xx = x0
        parts += [rect(xx, y, widths[0], heights, fill, GOLD if "Our replay" in name else GRID, 2 if "Our replay" in name else 1, 2), text(xx + 15, y + 27, name, 18, INK, 700), text(xx + 15, y + 51, note, 13, MUTED)]
        xx += widths[0]
        for val, w in zip(vals, widths[1:]):
            parts += [rect(xx, y, w, heights, fill, GOLD if "Our replay" in name else GRID, 2 if "Our replay" in name else 1, 2)]
            color = GREEN if val == "✓" else (GOLD if val == "~" else MUTED)
            parts += [text(xx + w / 2, y + 42, val, 28, color, 700, "middle")]
            xx += w
        y += heights
    parts += [rect(70, 820, 1665, 62, PALE_BLUE, GRID, 1.5, 10), text(95, 846, "Reading rule", 16, BLUE, 700), text(220, 846, "✓ = in-scope evidence; ~ = partial/different abstraction; — = not claimed. The matrix compares capability intersections, not a single quality score.", 15, INK)]
    return "".join(parts) + finish("The strongest positioning is not a generic ‘faster simulator’ claim. It is the intersection of native trace identity, causal sector replay, explicit frontier semantics, independent NCU joins, and complete 286/339-root evidence.", "Sources: Accel-Sim, MGPUSim/Akita, NVArchSim, MICRO 2025 modern-core model, Sim-FA, GPU-Tile-Sim/GTSim, PipeWeave, TileSight, and local SOTA audit.")


def results_dashboard():
    parts = [svg_start("What the complete corpus supports", "Current headline metrics meet a declared threshold; unresolved families stay visible as future work."), header("What the complete corpus supports", "Launch-level MAPE from corrected full-corpus joins; the right panel keeps campaign boundaries honest.")]
    # accuracy panel
    parts += [rect(65, 145, 1030, 700, PALE_BLUE, BLUE, 2.5, 20), text(95, 190, "A  ARCHITECTURAL METRICS", 22, BLUE, 700), text(95, 222, "mean absolute percentage error (lower is better)", 16, MUTED, 400, italic=True)]
    metrics = [("Global load sectors", 0.210, 0.010), ("L1 lookup misses", 9.828, 6.094), ("L2 TEX lookup misses", 0.000, 2.297), ("Shared load wavefronts", 5.623, 7.250), ("Active IPC", 14.487, 9.455), ("Elapsed IPC", 12.307, 13.602)]
    chart_x, chart_y, chart_w = 360, 285, 630
    for tick in [0, 5, 10, 15, 20]:
        x = chart_x + chart_w * tick / 20
        parts += [line(x, chart_y - 22, x, chart_y + 470, GRID, 1, dash="4 7"), text(x, chart_y + 500, f"{tick}%", 14, MUTED, anchor="middle")]
    parts += [line(chart_x + chart_w * 0.5, chart_y - 30, chart_x + chart_w * 0.5, chart_y + 470, GOLD, 3, dash="10 7"), text(chart_x + chart_w * 0.5, chart_y - 43, "10%", 14, GOLD, 700, anchor="middle"), line(chart_x + chart_w * 0.75, chart_y - 30, chart_x + chart_w * 0.75, chart_y + 470, CORAL, 3, dash="10 7"), text(chart_x + chart_w * 0.75, chart_y - 43, "15%", 14, CORAL, 700, anchor="middle")]
    for i, (name, sm86, sm120) in enumerate(metrics):
        y = chart_y + i * 76
        parts += [text(95, y + 28, name, 17, INK, 700)]
        for val, color, marker, dy in [(sm86, BLUE, "square", 10), (sm120, TEAL, "circle", 32)]:
            x = chart_x + chart_w * min(val, 20) / 20
            parts += [line(chart_x, y + dy, x, y + dy, color, 8), (rect(x - 8, y + dy - 8, 16, 16, color, color, 1, 3) if marker == "square" else circle(x, y + dy, 8, color)), text(x + 14, y + dy + 5, f"{val:.3f}%", 14, color, 700)]
    parts += [rect(95, 775, 900, 48, "white", GRID, 1.2, 8), rect(120, 791, 18, 12, BLUE, BLUE, 1, 3), text(150, 803, "SM86 / RTX 3090 Ti", 14, INK), circle(390, 797, 7, TEAL), text(410, 803, "SM120 / RTX 5090", 14, INK)]

    # runtime panel
    parts += [rect(1140, 145, 595, 700, PALE_GOLD, GOLD, 2.5, 20), text(1170, 190, "B  CAMPAIGN COST", 22, GOLD, 700), text(1170, 222, "seconds; boundaries are not interchangeable", 16, MUTED, 400, italic=True)]
    times = [("bare GPU", 242.530, GREEN), ("sim driver", 344.819, BLUE), ("sim external", 509.050, PURPLE), ("NCU command", 902.797, GOLD), ("NVBit capture", 2640.820, CORAL)]
    maxv = 2700
    for i, (name, val, color) in enumerate(times):
        y = 300 + i * 72
        w = 420 * (val / maxv) ** 0.55
        parts += [text(1170, y + 20, name, 16, INK, 700), rect(1300, y + 2, w, 32, color, color, 1, 6), text(1310 + w, y + 25, f"{val:,.1f}s", 15, color, 700)]
    parts += [rect(1170, 690, 500, 105, PALE_CORAL, CORAL, 1.5, 10), text(1195, 722, "Future work", 17, CORAL, 700), text(1195, 750, "DRAM reads • aggregate L2 hits", 15, INK), text(1195, 775, "GPC/SM completion frontiers", 15, INK)]
    parts += [rect(1170, 820, 500, 54, "white", GRID, 1.2, 8), text(1420, 852, "No same-host Accel-Sim speedup claimed", 15, INK, 700, "middle")]
    return "".join(parts) + finish("The dashboard makes the paper's argument visually testable: several architecture-facing surfaces are below 10–15%, complete-corpus replay is practical, and unresolved families are not hidden behind a pooled score.", "Source: H116 SM86 report, corrected 339-directory SM120 reduction, and sequential runtime notes.")


def architecture_pipeline():
    parts = [svg_start("Native replay architecture and dataflow", "Concrete artifacts flow through sim-trace, sim-sm, sim-engine, sim-mem, and sim-run; NCU remains a one-way post-replay branch."), header("Native replay architecture and dataflow", "The simulator is a connected pipeline, not a collection of independent counters.")]
    # Inputs
    parts += [rect(65, 155, 340, 650, PALE_BLUE, BLUE, 2.5, 20), text(92, 198, "INPUT ARTIFACTS", 22, BLUE, 700), text(92, 230, "captured once; never regenerated by replay", 16, MUTED, 400, italic=True)]
    inputs = [(92, 275, "CUDA / SASS workload", "real kernel binary + launch roster", SKY), (92, 395, "dynamic_trace.pb", "PC • masks • lane addresses • placement", TEAL), (92, 515, "SASS control sidecar", "stall • yield • barriers • reuse", PURPLE), (92, 635, "SM86 / SM120 card", "geometry • latency • typed assumptions", GOLD)]
    for x, y, title, sub, col in inputs:
        parts += [rect(x, y, 285, 82, "white", col, 2, 13), circle(x + 23, y + 25, 10, col), text(x + 45, y + 32, title, 18, INK, 700), text(x + 23, y + 60, sub, 14, MUTED)]
    # pipeline
    parts += [rect(465, 155, 960, 650, PALE_TEAL, TEAL, 2.5, 20), text(495, 198, "CONNECTED NATIVE REPLAY", 22, TEAL, 700), text(495, 230, "identity and causal state remain visible across stages", 16, MUTED, 400, italic=True)]
    stage_x = [495, 720, 945, 1170]
    stages = [("sim-trace", "decode + canonical join", BLUE), ("sim-sm", "issue + readiness + barriers", PURPLE), ("sim-engine", "deterministic event queue", CORAL), ("sim-mem", "sectored L1 → L2 → service", GOLD)]
    for i, (title, sub, col) in enumerate(stages):
        x = stage_x[i]
        parts += [rect(x, 330, 195, 140, "white", col, 2.5, 15), circle(x + 28, 362, 14, col), text(x + 28, 368, str(i + 1), 16, "white", 700, "middle"), text(x + 22, 410, title, 20, INK, 700), text(x + 22, 442, sub, 14, MUTED)]
        if i < len(stages) - 1:
            parts += [line(x + 195, 400, stage_x[i + 1], 400, BLUE, 4, marker="arrowBlue")]
    parts += [path("M1030 470 C1030 555 815 555 815 470", TEAL, 3, dash="9 7", marker="arrowBlue"), text(925, 585, "SM wake-up ↔ memory response", 15, TEAL, 700, "middle")]
    parts += [rect(600, 650, 690, 80, TEAL, TEAL, 1, 14), text(945, 685, "warp identity • line/sector masks • event sequence", 18, "white", 700, "middle"), text(945, 712, "carried across the entire replay", 15, "white", 400, "middle")]
    # output
    parts += [rect(1485, 155, 250, 650, PALE_GOLD, GOLD, 2.5, 20), text(1510, 198, "OUTPUTS", 22, GOLD, 700), text(1510, 230, "post-replay evidence", 16, MUTED, 400, italic=True)]
    outs = [(1515, 295, "sim-run JSONL", "counters + predictions", GREEN), (1515, 425, "frontier ledger", "issue • completion • tails", PURPLE), (1515, 555, "NCU comparator", "directory + launch join", CORAL), (1515, 685, "reports", "MAPE + future-work gaps", BLUE)]
    for x, y, title, sub, col in outs:
        parts += [rect(x, y, 190, 82, "white", col, 2, 13), text(x + 15, y + 32, title, 17, INK, 700), text(x + 15, y + 60, sub, 14, MUTED)]
    parts += [path("M1425 400 L1485 400", BLUE, 4, marker="arrowBlue"), path("M1425 610 C1460 610 1460 590 1515 590", CORAL, 3, dash="9 7", marker="arrowCoral"), text(1460, 548, "one-way observation", 14, CORAL, 700, "middle")]
    return "".join(parts) + finish("The native path connects trace identity, scheduling, event ordering, memory state, and report emission. NCU enters only after replay; no profiler value is allowed to alter the modeled path.", "Source: docs/ARCHITECTURE.md, docs/COMPONENTS.md, sim/trace, sim/sm, sim/engine, sim/mem, and sim/direct_run.")


def causal_timeline():
    parts = [svg_start("One instruction through the causal replay pipeline", "A worked timeline separates issue, cache lookup, service, response wake-up, and observation frontiers."), header("One instruction through the causal replay pipeline", "The model advances between meaningful events while preserving ownership of each frontier.")]
    x0, x1 = 250, 1690
    parts += [rect(65, 145, 1670, 700, PALE_GRAY, GRID, 2.0, 18), text(95, 190, "WORKED EVENT SLICE", 22, BLUE, 700), text(95, 220, "one warp instruction • four sectors • one merged fill", 16, MUTED, 400, italic=True)]
    lanes = [("warp / SMSP", BLUE), ("L1TEX", TEAL), ("shared L2", PURPLE), ("service", CORAL), ("wake-up", GREEN)]
    ys = [300, 390, 480, 570, 660]
    for (name, col), y in zip(lanes, ys):
        parts += [text(95, y + 6, name, 17, INK, 700), line(x0, y, x1, y, GRID, 2)]
    # time scale
    for t in [0, 1, 2, 8, 24, 48, 96]:
        x = x0 + (x1 - x0) * t / 96
        parts += [line(x, 250, x, 725, GRID, 1, dash="4 8"), text(x, 755, str(t), 14, MUTED, anchor="middle")]
    parts += [text(965, 790, "relative SM cycles", 15, MUTED, anchor="middle")]
    # Events as rounded bars
    events = [("issue", 0, 8, 300, BLUE), ("L1 lookup", 8, 12, 390, TEAL), ("L2 miss", 20, 20, 480, PURPLE), ("DRAM / service", 40, 32, 570, CORAL), ("fill + wake", 72, 10, 660, GREEN)]
    for label, start, width, y, col in events:
        sx = x0 + (x1 - x0) * start / 96
        ex = x0 + (x1 - x0) * (start + width) / 96
        parts += [rect(sx, y - 23, max(35, ex - sx), 46, col, col, 1, 10), text((sx + max(35, ex - sx) / 2), y + 6, label, 15, "white", 700, "middle")]
    # merge arrow
    parts += [path("M390 390 C500 345 520 480 620 480", TEAL, 3, dash="8 7", marker="arrowBlue"), text(500, 350, "MSHR merge", 14, TEAL, 700, "middle")]
    # frontiers
    frontier = [("issue frontier", 8, BLUE), ("modeled completion", 82, PURPLE), ("observation", 96, GOLD)]
    for label, t, col in frontier:
        x = x0 + (x1 - x0) * t / 96
        parts += [line(x, 245, x, 725, col, 3, dash="10 7"), text(x + 8, 265, label, 14, col, 700)]
    parts += [rect(95, 815, 1590, 48, PALE_CORAL, CORAL, 1.5, 10), text(890, 846, "Frontier rule: issue ≠ completion ≠ profiler observation; dirty writeback/coherence remain typed tails.", 17, CORAL, 700, "middle")]
    return "".join(parts) + finish("A single traced instruction exercises the simulator's core promise: event time jumps from issue to lookup, service, and wake-up, but the report preserves which frontier each event belongs to.", "Source: sim-engine event ordering, sim-sm readiness, sim-mem sector state, and the frontier ledger.")


def simt_interleaving():
    parts = [svg_start("How SIMT is modeled without pretending lanes are CPU threads", "CPU thread interleaving, SIMT warp lockstep, and warp-level scheduling are separated; the replay uses the third model."), header("How SIMT is modeled without pretending lanes are CPU threads", "Lane masks stay inside a warp; scheduling interleaves resident warps across SMSP issue partitions.")]
    panels = [(65, 145, 510, "CPU-style thread interleaving", PALE_BLUE, BLUE), (645, 145, 510, "SIMT warp lockstep", PALE_TEAL, TEAL), (1225, 145, 510, "Our native replay", PALE_GOLD, GOLD)]
    for x, y, w, title, fill, col in panels:
        parts += [rect(x, y, w, 610, fill, col, 2.5, 20), text(x + 28, y + 45, title, 21, col, 700)]
    # CPU rows
    parts += [text(95, 225, "host threads own independent PCs", 15, MUTED, italic=True)]
    for i in range(4):
        y = 280 + i * 68
        parts += [text(95, y + 18, f"T{i}", 16, INK, 700), rect(145, y, 100, 35, BLUE, BLUE, 1, 6), text(195, y + 23, "op", 15, "white", 700, "middle"), rect(265, y, 100, 35, SKY, SKY, 1, 6), text(315, y + 23, "op", 15, "white", 700, "middle"), rect(385, y, 100, 35, BLUE, BLUE, 1, 6), text(435, y + 23, "op", 15, "white", 700, "middle")]
    parts += [rect(95, 595, 420, 92, "white", BLUE, 1.5, 10), text(115, 625, "Tradeoff", 16, BLUE, 700), text(115, 652, "accurate per-thread control; high host/event cost", 15, INK)]
    # lockstep
    parts += [text(675, 225, "one warp instruction, many lanes", 15, MUTED, italic=True)]
    for row in range(4):
        y = 280 + row * 68
        parts += [text(675, y + 18, f"warp {row}", 16, INK, 700)]
        for j in range(8):
            col = TEAL if (j + row) % 3 else PURPLE
            parts += [rect(755 + j * 34, y, 27, 35, col, col, 1, 4)]
    parts += [rect(675, 595, 420, 92, "white", TEAL, 1.5, 10), text(695, 625, "Tradeoff", 16, TEAL, 700), text(695, 652, "excellent lane efficiency; divergence and masks matter", 15, INK)]
    # Our replay
    parts += [text(1255, 225, "recorded warp streams + masks", 15, MUTED, italic=True)]
    # scheduler lanes
    for i, col in enumerate([BLUE, PURPLE, CORAL, GOLD]):
        y = 282 + i * 65
        parts += [rect(1255, y, 100, 35, col, col, 1, 6), text(1305, y + 23, f"SMSP {i}", 15, "white", 700, "middle"), line(1365, y + 18, 1410, y + 18, col, 3, marker="arrowBlue"), rect(1415, y, 105, 35, col, col, 1, 6), text(1467, y + 23, "warp", 15, "white", 700, "middle"), rect(1535, y, 115, 35, "white", col, 1.5, 6), text(1592, y + 23, "mask + sectors", 13, col, 700, "middle")]
    parts += [rect(1255, 595, 420, 92, "white", GOLD, 1.5, 10), text(1275, 625, "Our tradeoff", 16, GOLD, 700), text(1275, 652, "one host actor per SM; lanes stay compact and auditable", 15, INK)]
    # bottom comparison
    parts += [rect(65, 785, 1670, 84, PALE_CORAL, CORAL, 1.5, 12), text(95, 818, "Key distinction", 17, CORAL, 700), text(245, 818, "we do not spawn 32 host threads per warp", 17, INK, 700), text(245, 846, "the trace supplies lane masks/addresses; sim-sm interleaves resident warps, and sim-mem aggregates sectors at cache acceptance", 15, INK)]
    return "".join(parts) + finish("SIMT modeling is a hierarchy: lanes execute one warp instruction together, while resident warps interleave at SMSP issue boundaries. The replay preserves the lane evidence without paying CPU-style per-thread event cost.", "Source: sim/sm/src/issue.rs, sim/sm/src/replay.rs, static SASS masks, placement metadata, and custom SIMT differential evidence.")


def write(name, content):
    (OUT / name).write_text(content, encoding="utf-8")


if __name__ == "__main__":
    write("novelty_stack.svg", novelty_stack())
    write("sota_matrix.svg", sota_matrix())
    write("results_dashboard.svg", results_dashboard())
    write("architecture_pipeline.svg", architecture_pipeline())
    write("causal_timeline.svg", causal_timeline())
    write("simt_interleaving.svg", simt_interleaving())
