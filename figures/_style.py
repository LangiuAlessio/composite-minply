"""One visual system for every figure in the paper.

The palette is a validated categorical set (lightness band, chroma floor, all-pairs
CVD separation under protanopia/deuteranopia/tritanopia, contrast against a white
page). Identity is carried by hue AND marker shape, so the figures survive
colour-blind readers and a black-and-white printer; the two hues that sit below the
3:1 contrast floor on white are never asked to carry a value alone -- the tables in
the paper hold every number the figures show.

Two conventions hold across the figures, and they are what make them read as one set:
  filled, coloured mark  = this work
  open, grey mark        = the reference we are compared against (a published
                           experiment, a published optimum, a rival prediction)
"""
from __future__ import annotations

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ink and chrome
INK, SECOND, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, SURFACE = '#e1e0d9', '#c3c2b7', '#ffffff'
REF = '#898781'                 # anything that is not ours: open, grey

# categorical slots, in fixed order (the order is the CVD-safety mechanism)
BLUE, GREEN, AMBER = '#2a78d6', '#1baf7a', '#eda100'

MDPI_LINEWIDTH_IN = 5.5         # \linewidth of the MDPI class: draw at print size


def style():
    """Thin marks, hairline solid grid, recessive axes, text never in a data colour."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans'],
        'font.size': 8.5,
        'axes.labelsize': 8.5, 'axes.titlesize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'axes.edgecolor': AXIS, 'axes.linewidth': 0.6,
        'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'grid.color': GRID, 'grid.linewidth': 0.6, 'grid.linestyle': '-',
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'pdf.fonttype': 42, 'savefig.facecolor': SURFACE,
    })
