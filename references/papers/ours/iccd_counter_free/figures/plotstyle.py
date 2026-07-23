"""Shared matplotlib style for the paper figures.

IEEE two-column friendly: compact, Times-compatible serif (matches the
IEEEtran body font), colour-blind-safe palette. Every figure imports
`apply_style()` and the `PALETTE`/`TIER_COLOR`/`GPU_COLOR` helpers so the
whole report is visually consistent.

Colour code (one semantic code across all figures):
* #0072B2 (blue) and #D55E00 (orange) are RESERVED for the two
  identifiability verdicts (recoverable / non-identifiable).
* GPU identity (RTX 5060 vs RTX 3090 Ti) uses two NEUTRAL hues
  (teal / slate) plus distinct hatching so it survives grayscale printing
  and never collides with the verdict colours.

The RTX 5060 (Blackwell) is the primary measured GPU. The RTX 3090 Ti
(Ampere, sm_86) cross-architecture study is complete: its
recoverable-traffic and timing legs are confirmed, and the large-cache (L2)
identifiability boundary is confirmed structurally (the same erasure
mechanism on both arches; the calibrated sm_86 head remains a fitted
correction, so no fitted per-family Ampere value is drawn). The
identifiability map presents the 11 essential metrics and two tiers
(3 recoverable + 8 non-identifiable). Wall-clock kernel time and the
near-duplicate variants remain artifact-only, and the trivially-exact
counting metrics are out of scope.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Column widths (inches) for IEEEtran. Figures are included at (almost) full
# width in LaTeX, so the PDF size IS the print size -- do not change.
COL_W = 3.35      # single column
DBL_W = 7.0       # full text width (two columns)

# Colour-blind-safe palette (Wong 2011).
PALETTE = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
    "grey":   "#999999",
    "black":  "#000000",
}

# The two identifiability tiers get fixed colours across all figures.
# These two hues are reserved for verdicts and used for nothing else.
TIER_COLOR = {
    "recoverable":      PALETTE["blue"],    # #0072B2
    "non_identifiable": PALETTE["red"],     # #D55E00
}

# GPU identity uses neutral hues (never the verdict colours above).
GPU_COLOR = {
    "rtx5060":  "#2a9d8f",   # teal  -- RTX 5060 (Blackwell, sm_120)
    "rtx3090ti": "#4d6d8c",  # slate -- RTX 3090 Ti (Ampere, sm_86)
}


def apply_style():
    plt.rcParams.update({
        # Typography: Times-compatible serif chain to match the IEEEtran body
        # font; STIX mathtext; Type-42 (TrueType) embedding for camera-ready.
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "STIXGeneral",
                       "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Sizes: nothing in any figure may fall below 6.5 pt.
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        # Axes: open frame, y-only dotted grid drawn below the data.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,
        "grid.color": "#666666",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
