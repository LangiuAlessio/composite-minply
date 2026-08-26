#!/usr/bin/env python3
"""Figure: three decisive pitfalls in composite-buckling optimisation (Figure 9).

PORTED, 2026-07-20. This figure was the only one in the paper with no source inside this
bundle, and RS-005 had recorded two of its three panels as unrecoverable. Both records were
wrong: the source existed all along in a DIFFERENT repository --
`ingegneria/fe-batch-lab/cases/negative_results_plot.py`, the FE development lab from which
this bundle was extracted -- so every search run from the root of `ateneo/` was blind to it
by construction. This file is that script, brought into the bundle and put on the paper's
visual system, with every number carrying its provenance.

  (A) the weak-material 'chop' yields a layup-independent spurious factor near 1.0, while the
      frequency of the SAME model still matches Abaqus to <1%: the method fails, not the setup.
  (B) the C3D8I solid mis-predicts buckling by ~300x against Abaqus, while frequency and the
      S8R shell agree to <2%.
  (C) the free-edge peel point-maximum is singular (Pipes-Pagano), the Whitney-Nuismer averaged
      criterion stays bounded.

The numbers are those of the published figure, reproduced here exactly; the canary below
refuses to plot if any of them drifts. Their provenance:
  panel (A) -- weak-chop buckling on the 14k-node C3D8I solid (Composite_buckling_3.inp, case 3).
              GENERATED IN-BUNDLE (RS-005, 2026-07-20): experiments/exp15_panelA_weakchop.py
              reruns the four variants on ccx (translator fe/abq2ccx_rr.py) and reproduces the
              published bars to the printed digits -- [0]_60=20.14, [0]_24+36w=0.955,
              [90]_24+36w=0.991, [0]_12+48w=0.985 vs the plotted 20.10/0.95/0.99/0.98, with a
              per-bar canary. Data: code/data/exp15_panelA_weakchop.json. The source deck
              (decks/Composite_buckling_3.inp) is git-ignored: not versioned, coauthor's model.
              (Il vincolo di releasability RR e' CADUTO il 2026-07-28, messaggio di G. Canale:
              ne' geometria ne' dati sono coperti da copyright o restrizioni proprietarie.)
  panel (B) -- Table 7 of the paper for the three healthy ratios; the 298x is ccx ~1.0 against
              Abaqus LE 0.00336 on the 928-node 3-ply crop (FINDINGS:82). Both re-measured:
              exp12 reproduces the Table 7 deck, exp13 reproduces the ccx ~1.0 on the crop.
  panel (C) -- REBUILT ON THE DATA, 2026-07-25. It was the one panel whose numbers no artefact in
              the bundle produced: four hardcoded point-max values [9.8, 21.0, 33.0, 62.0] MPa,
              commented "across layups", against a single averaged point at 0.1, under a dashed
              "naive allowable 'peel < 10'" that three of the four cleared. The generator named
              right here, exp10_peel_mesh_sweep.py, measures something else -- a MESH sweep, which
              is also what the paper's caption and Section 3.8 describe -- and its four rows are
              0.280/0.989/2.636/4.861 MPa point-max against 0.0296/0.0252/0.0802/0.1138 averaged.
              Two consequences, both of which the published panel got wrong: the point-max spread
              is a factor of 17, not 6.3, and NO mesh in the sweep reaches the 10 MPa allowable,
              so the line the panel drew the eye to was not crossed by any datum in the bundle.
              The panel now plots the four rows of code/data/exp10_peel_mesh_sweep.json directly,
              and the canary reads that file instead of trusting a transcription. The "averaged
              criterion (stable)" legend and the "the averaged Q converges" annotation went with
              it: B2 of the 2026-07-22 audit retired exactly that claim from the body ("bounded
              rather than mesh-converged, and deliberately", Section 3.8), and the figure was
              still asserting it.

⚠️ OPEN CAVEAT ON PANEL (B) -- the number is right, the stated cause may not be. The paper
attributes the 298x to element locking. exp13_solid_buckling_spurious.py shows the ccx side of
that ratio is not an eigenvalue of the reference problem at all: it does not scale with the
reference load (the invariant BF*|load| disperses by 94539%, against 0.00% and 0.72% on the
healthy shell and the healthy full solid), it does not relax under through-thickness refinement
at constant thickness (1.0213 -> 1.0173 from one to four elements per ply), and the spectrum is
a dense cluster of local modes between 0.87 and 1.17 with no separated global mode. That is the
signature panel (A) itself uses to prove an eigenvalue is SPURIOUS. The conclusion the figure
draws is unaffected -- the solid is not trustworthy for buckling, use the shell -- but the
mechanism named in the caption and in the body is not the one the data supports. Left as it was
published, deliberately: changing the science is a coauthor decision, not a porting decision.

    python3 code/figures/fig_pitfalls.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import (AMBER, AXIS, BLUE, GREEN, GRID, INK, MDPI_LINEWIDTH_IN, MUTED,
                    REF, SECOND, SURFACE, style)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'code' / 'figures' / '_out' / 'fig_pitfalls.pdf'

# --- panel (A): weak-material chop -------------------------------------------------------
# BF1 of the real 60-ply laminate, then three weak-filler variants. The point is that the
# three weak ones are all ~1.0 AND indistinguishable from each other: [0] == [90] is
# unphysical, so the eigenvalue is not seeing the laminate.
CHOP = [('[0]$_{60}$\nreal',      20.10, True),
        ('[0]$_{24}$\n+36 w',     0.95, False),
        ('[90]$_{24}$\n+36 w',    0.99, False),
        ('[0]$_{12}$\n+48 w',     0.98, False)]

# --- panel (B): |ccx / Abaqus| ------------------------------------------------------------
# Three healthy ratios from Table 7, plus the 298x. Stored as the (ccx, Abaqus) PAIR rather
# than the ratio, so the canary can check the arithmetic instead of trusting a copied number.
# La barra della frequenza veniva dal crop a 928 nodi del modello del coautore, che non e' nel
# bundle e non e' ricalcolabile da nessuno. Sostituita il 2026-08-26 con il confronto modale
# misurato da exp29 sul pannello S8R a 661 nodi che il bundle genera: ccx 532.91 contro Abaqus
# 2026 LE 531.69 Hz. Stesso significato, stessa scala, ma la barra ora si rigenera.
RATIOS = [('freq\nshell',         532.91, 531.69,   True),
          ('buckling\nsolid',        1.00,   0.00336, False),
          ('axial\nshell',         10.84,  10.819,  True),
          ('combo\nshell',          3.992,  3.915,  True)]   # 3.985 -> 3.992, audit A2

# --- panel (C): free-edge peel -------------------------------------------------------------
# Read from the artefact, not transcribed: the mesh sweep of exp10_peel_mesh_sweep.py, which is
# the sweep the caption and Section 3.8 describe. Four in-plane meshes at fixed nz=24.
PEEL_JSON = ROOT / 'code' / 'data' / 'exp10_peel_mesh_sweep.json'
# Ricalibrati il 2026-08-26. I due numeri di prima (15 e 6) codificavano il 17 e il 4,5 del corpo
# del testo, che erano stati prodotti da un parser .frd difettoso: scartava in silenzio ogni riga
# con sigma_xx negativo e lasciava quelle celle a 0.0. Col parser corretto e sul caso di carico
# della campagna (`campaign_S1`, che e' quello che il paragrafo del manoscritto descrive, non il
# caso demo in compressione) i fattori misurati sono 21,7 sul massimo puntuale e 4,2 sulla media.
PEEL_SPREAD_MIN = 15.0     # il corpo del testo rivendica un fattore ~22 sul massimo puntuale
PEEL_AVG_SPREAD_MAX = 6.0  # ... contro ~4,2 sulla media di banda: varia lentamente, non converge

PUBLISHED_RATIOS = [1.002, 298.0, 1.002, 1.020]   # as printed in the published panel (B)
#                                          ^ was 1.018 on the inherited 3.985; the audit of
#                                            2026-07-20 recomputed the combined factor as
#                                            3.9924, so the published bar is 3.9924/3.915.


def load_peel():
    """Panel (C) data, read from the exp10 artefact. Fails loudly rather than plotting a guess."""
    if not PEEL_JSON.exists():
        sys.exit(f'fig_pitfalls: {PEEL_JSON.name} is missing -- panel (C) has no source, not drawn')
    data = json.loads(PEEL_JSON.read_text())
    # Dal 26/08/2026 l'artefatto porta DUE casi di carico e dichiara in `figure_case` quale
    # sostiene questo pannello. Il caso demo in compressione NON lo sostiene: li' la media di
    # banda della sigma_zz e' compressiva ovunque, il criterio la prende in parte positiva e il
    # peel mediato e' identicamente zero -- il contrasto che il pannello mostra non esiste su
    # quel carico. Si legge quindi il caso dichiarato, e si rifiuta di disegnare se il file e'
    # vecchio (un JSON senza `cases` e' pre-correzione, e le sue curve sono quelle del campo
    # azzerato).
    if 'cases' not in data or 'figure_case' not in data:
        sys.exit('fig_pitfalls: exp10_peel_mesh_sweep.json e\' in formato pre-2026-08-26 (un '
                 'solo caso di carico, prodotto dal parser .frd difettoso) -- rigenera exp10, '
                 'pannello (C) non disegnato')
    case = data['figure_case']
    rows = data['cases'][case]['rows']
    if len(rows) < 3:
        sys.exit('fig_pitfalls: the peel sweep has fewer than three meshes -- '
                 'panel (C) could not show a trend under refinement, not drawn')
    if min(r['peel_avg'] for r in rows) <= 0:
        sys.exit(f'fig_pitfalls: sul caso {case} il peel mediato e\' nullo su almeno una mesh: '
                 f'il contrasto del pannello (C) non e\' definito, non disegnato')
    labels = [f"{r['nx']}$\\times${r['ny']}" for r in rows]
    point = [r['peel_point'] for r in rows]
    avg = [r['peel_avg'] for r in rows]
    return labels, point, avg


def canary():
    """Refuse to plot if the arithmetic of panel (B) no longer gives the published bars."""
    got = [ccx / abq for _, ccx, abq, _ in RATIOS]
    for g, p in zip(got, PUBLISHED_RATIOS):
        if abs(g - p) / p > 0.005:
            sys.exit(f'fig_pitfalls: panel (B) does not reproduce the published bar '
                     f'({g:.4f} vs {p}) -- the figure would misstate the paper, not drawn')
    _, point, avg = load_peel()
    spread, avg_spread = max(point) / min(point), max(avg) / min(avg)
    if spread < PEEL_SPREAD_MIN:
        sys.exit(f'fig_pitfalls: the point-max peel spreads by {spread:.1f}x under refinement, '
                 f'below the {PEEL_SPREAD_MIN}x the body claims -- panel (C) would not show the '
                 'singularity it claims, not drawn')
    if avg_spread > PEEL_AVG_SPREAD_MAX:
        sys.exit(f'fig_pitfalls: the band-averaged peel spreads by {avg_spread:.1f}x, above the '
                 f'{PEEL_AVG_SPREAD_MAX}x that still reads as bounded -- the contrast the panel '
                 'is for would not hold, not drawn')
    if avg_spread >= spread:
        sys.exit('fig_pitfalls: the band average is no longer more stable than the point maximum '
                 '-- panel (C) would show no contrast at all, not drawn')
    if not all(v < 1.05 for _, v, ok in CHOP if not ok):
        sys.exit('fig_pitfalls: the weak-chop factors are no longer stuck at ~1.0 -- '
                 'panel (A) would not show the failure it claims, not drawn')
    return got


def main() -> None:
    ratios = canary()
    style()
    fig, (axA, axB, axC) = plt.subplots(
        1, 3, figsize=(MDPI_LINEWIDTH_IN, MDPI_LINEWIDTH_IN / 1.95), layout='constrained')

    # ---------------- (A) ----------------
    labels = [l for l, _, _ in CHOP]
    vals = [v for _, v, _ in CHOP]
    cols = [BLUE if ok else AMBER for _, _, ok in CHOP]
    axA.bar(range(4), vals, color=cols, width=0.62, edgecolor=SURFACE, lw=0.6)
    axA.axhline(1.0, ls=(0, (3, 3)), color=AXIS, lw=0.7)
    axA.set_xticks(range(4))
    axA.set_xticklabels(labels, fontsize=5.8, color=SECOND, linespacing=1.3)
    axA.set_ylabel('1st buckling factor  BF$_1$', color=INK, fontsize=7.2)
    axA.set_ylim(0, 38)
    for i, v in enumerate(vals):
        axA.text(i, v + 0.7, f'{v:.2f}', ha='center', fontsize=6.0, color=SECOND)
    axA.text(0.96, 0.97, 'stuck at ~1.0 for any layup\n([0] $\\equiv$ [90]: unphysical)',
             transform=axA.transAxes, ha='right', va='top', fontsize=5.5, color=AMBER,
             linespacing=1.3)
    axA.text(0.96, 0.79, 'control: frequency on the same\nmodel, <1% vs Abaqus',
             transform=axA.transAxes, ha='right', va='top', fontsize=5.5, color=BLUE,
             linespacing=1.3)
    axA.set_title("(A)  Weak 'chop'", loc='left', color=INK, pad=4, fontsize=7.4)

    # ---------------- (B) ----------------
    btags = [t for t, _, _, _ in RATIOS]
    bcols = [BLUE if ok else AMBER for _, _, _, ok in RATIOS]
    axB.bar(range(4), ratios, color=bcols, width=0.62, edgecolor=SURFACE, lw=0.6)
    axB.set_yscale('log')
    axB.axhline(1.0, ls=(0, (3, 3)), color=AXIS, lw=0.7)
    axB.set_xticks(range(4))
    axB.set_xticklabels(btags, fontsize=5.5, color=SECOND, linespacing=1.3)
    axB.set_ylabel('| ccx / Abaqus |  (1 = perfect)', color=INK, fontsize=7.2)
    axB.set_ylim(0.5, 1500)
    for i, v in enumerate(ratios):
        axB.text(i, v * 1.4, f'{v:.0f}×' if v > 5 else f'{v:.3f}',
                 ha='center', fontsize=6.0, color=SECOND)
    axB.text(0.5, 0.46, 'frequency and shell buckling\nagree to <2%; only the thin\nC3D8I solid is ~300× off',
             transform=axB.transAxes, ha='center', va='top', fontsize=5.8, color=MUTED,
             linespacing=1.35)
    axB.set_title('(B)  Solid vs shell', loc='left', color=INK, pad=4, fontsize=7.4)

    # ---------------- (C) ----------------
    mesh_labels, peel_point, peel_avg = load_peel()
    x = list(range(len(mesh_labels)))
    axC.plot(x, peel_point, '-o', ms=4.0, color=AMBER, zorder=3, lw=0.9,
             markeredgecolor=SURFACE, markeredgewidth=0.5,
             label='point-max $\\sigma_{33}$ (singular)')
    axC.plot(x, peel_avg, '-s', ms=4.0, color=GREEN, zorder=3, lw=0.9,
             markeredgecolor=SURFACE, markeredgewidth=0.5,
             label='band-averaged (bounded)')
    axC.annotate(f'$\\times${max(peel_point) / min(peel_point):.0f}',
                 xy=(x[-1], peel_point[-1]), xytext=(4, 3), textcoords='offset points',
                 fontsize=6.0, color=AMBER, ha='right', va='bottom')
    axC.annotate(f'$\\times${max(peel_avg) / min(peel_avg):.1f}',
                 xy=(x[-1], peel_avg[-1]), xytext=(-1, 5), textcoords='offset points',
                 fontsize=6.0, color=GREEN, ha='right', va='bottom')
    axC.set_xlim(-0.4, len(x) - 0.6)
    axC.set_xticks(x)
    axC.set_xticklabels(mesh_labels, fontsize=5.8, color=SECOND)
    axC.set_xlabel('in-plane mesh ($n_z=24$)', color=SECOND, fontsize=6.0)
    axC.set_ylim(0.006, 40)
    axC.set_yscale('log')
    axC.set_ylabel('interlaminar peel $\\sigma_{33}$ (MPa)', color=INK, fontsize=7.2)
    axC.legend(fontsize=5.4, frameon=False, loc='upper left', handletextpad=0.3,
               borderaxespad=0.15)
    axC.text(0.97, 0.145, 'the point value diverges\nwith mesh (Pipes–Pagano);\n'
                          'the band average does not',
             transform=axC.transAxes, ha='right', va='top', fontsize=5.5, color=MUTED,
             linespacing=1.3)
    axC.set_title('(C)  Free-edge peel', loc='left', color=INK, pad=4, fontsize=7.4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix('.png'), dpi=300)
    print(f'wrote {OUT.relative_to(ROOT)} and .png')


if __name__ == '__main__':
    main()
