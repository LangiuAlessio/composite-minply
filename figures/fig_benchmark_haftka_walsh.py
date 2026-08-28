#!/usr/bin/env python3
"""Figure: our results against the Haftka-Walsh certified global optima.

(a) maximum buckling load factor over twelve load ratios, recovered by the GA;
(b) the minimum-thickness dual over eight transverse loads, recovered by EXHAUSTIVE
    ENUMERATION over the balanced symmetric half-stacks -- not by the search. The two
    panels do not come from the same method, and the legend must say so: until 27/08
    it said 'this search (GA)' for both, contradicting the paper's own caption.

In both panels the published global optimum is the open grey mark (the reference)
and ours is the filled blue one -- the same convention as the experimental
figure, where open grey is always somebody else's number.

Data: code/data/exp6_haftka_walsh.json, written by
      python3 -m experiments.exp6_haftka_walsh   (closed-form CLT, no FE, seconds)

The script refuses to plot unless the data still supports the claim the caption
makes: the global optimum recovered on all twelve ratios and all eight thickness
points, with the CLT evaluator within 1% of the published optima.

    python3 code/figures/fig_benchmark_haftka_walsh.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import AXIS, BLUE, GRID, INK, MDPI_LINEWIDTH_IN, REF, SECOND, SURFACE, style

# ⚠️ I percorsi si calcolano dal BUNDLE, non da due livelli sopra. `parents[2]` funzionava solo
# dentro il monorepo, dove questo repo si chiama `code/` e sta dentro la cartella del paper: in un
# clone di composite-minply puntava alla directory SOPRA il clone, e i tre script delle figure
# fallivano tutti -- cioe' `reproduce.sh` usciva in errore per ogni lettore.
BUNDLE = Path(__file__).resolve().parents[1]
# La figura si scrive accanto al paper se siamo nel monorepo, dentro il clone altrimenti.
_LEAF = BUNDLE.parent
OUTDIR = _LEAF if (_LEAF / 'composite_opt.bib').is_file() else BUNDLE / 'figures' / '_out'
OUTDIR.mkdir(parents=True, exist_ok=True)
DATA = BUNDLE / 'data' / 'exp6_haftka_walsh.json'
OUT = OUTDIR / 'RR_benchmark_haftka_walsh.pdf'


def canary(d):
    """The caption claims recovery on 12 ratios and 8 thickness points. Check it."""
    lam, thk = d['max_buckling'], d['min_thickness']
    recovered = sum(r['ga_recovers_optimum'] for r in lam)
    matched = sum(r['match'] for r in thk)
    worst = d['worst_clt_err_pct']
    problems = []
    if recovered != len(lam):
        problems.append(f'GA recovers the optimum on {recovered}/{len(lam)} load ratios, not all')
    if matched != len(thk):
        problems.append(f'min-thickness matches on {matched}/{len(thk)} points, not all')
    if worst >= 1.0:
        problems.append(f'CLT evaluator is {worst:.2f}% off the published optima (>= 1%)')
    if problems:
        sys.exit('CANARY FAILED - the figure would overstate the paper:\n  ' + '\n  '.join(problems))
    print(f'canary ok: optimum recovered {recovered}/{len(lam)} ratios and {matched}/{len(thk)} '
          f'thickness points; CLT within {worst:.2f}% of the published values')
    return recovered, len(lam), matched, len(thk), worst


def main():
    d = json.loads(DATA.read_text())
    nrec, nlam, nthk_ok, nthk, worst = canary(d)
    lam, thk = d['max_buckling'], d['min_thickness']

    style()
    # same rule as the validation figure: the canvas IS the printed width, so a point
    # drawn here is a point on paper (a tight bbox would come out wider and LaTeX would
    # scale the type down with it)
    fig = plt.figure(figsize=(MDPI_LINEWIDTH_IN, 3.05), layout='constrained')
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.06)
    ax, bx = fig.subplots(1, 2)

    # ---------------- (a) maximum buckling: lambda_cr over the load ratio ----------------
    r = [x['ratio'] for x in lam]
    ax.plot(r, [x['lam_paper'] for x in lam], 'o', ms=6.0, mfc=SURFACE, mec=REF, mew=1.0,
            ls='none', zorder=2)
    ax.plot(r, [x['lam_ga'] for x in lam], 'o', ms=3.6, color=BLUE, mec=SURFACE, mew=0.6,
            ls='none', zorder=3)
    ax.set(xscale='log', xlim=(0.10, 3.0), ylim=(20, 168))
    ax.set_xticks([0.125, 0.25, 0.5, 1.0, 2.0])
    ax.set_xticklabels(['0.125', '0.25', '0.5', '1', '2'])
    ax.set_xlabel('Load ratio  $N_y/N_x$')
    ax.set_ylabel(r'Buckling load factor  $\lambda_{cr}$', labelpad=2)
    ax.grid(True, zorder=0)
    ax.tick_params(length=2.5)
    ax.text(0.96, 0.93, f'recovered on\nall {nlam} ratios',
            transform=ax.transAxes, ha='right', va='top', fontsize=6.6, color=SECOND,
            linespacing=1.4)
    ax.set_title('(a)  Maximum buckling', loc='left', color=INK, pad=5)

    # ---------------- (b) the minimum-thickness dual ----------------
    ny = [x['ny'] for x in thk]
    plies = [x['nply_paper'] for x in thk]
    # the reference is a step function: it is a threshold, so draw it as one
    edges, levels = [], []
    for i, (n, p) in enumerate(zip(ny, plies)):
        edges.append(n)
        levels.append(p)
    bx.step(edges + [82.0], levels + [levels[-1]], where='post', color=AXIS, lw=0.9, zorder=1)
    bx.plot(ny, plies, 'o', ms=6.0, mfc=SURFACE, mec=REF, mew=1.0, ls='none', zorder=2)
    bx.plot(ny, [x['nply_ours'] for x in thk], 'o', ms=3.6, color=BLUE, mec=SURFACE, mew=0.6,
            ls='none', zorder=3)
    bx.set(xlim=(-6, 82), ylim=(9, 17.4))
    bx.set_yticks([10, 12, 14, 16])
    bx.set_xlabel('Transverse load  $N_y$  [lb/in]')
    bx.set_ylabel('Minimum number of plies', labelpad=2)
    bx.grid(True, axis='y', zorder=0)
    bx.tick_params(length=2.5)
    for spine in ('top', 'right'):
        bx.spines[spine].set_visible(False)
    bx.text(0.05, 0.94, f'the same ply count on all {nthk} points',
            transform=bx.transAxes, ha='left', va='top', fontsize=6.6, color=SECOND)
    bx.set_title('(b)  Minimum thickness at $N_x=30$', loc='left', color=INK, pad=5)

    fig.legend(handles=[
        Line2D([], [], marker='o', ls='none', ms=6.0, mfc=SURFACE, mec=REF, mew=1.0,
               label='Haftka-Walsh (1992), certified global optimum'),
        Line2D([], [], marker='o', ls='none', ms=3.6, color=BLUE, mec=SURFACE, mew=0.6,
               label='ours: GA in (a), exhaustive enumeration in (b)')],
        # ⚠️ ncol=1, non 2. Il canvas ha larghezza FISSA (e' la larghezza di stampa, vedi il
        # commento su savefig) e a due colonne l'etichetta allungata il 27/08 -- che distingue il
        # pannello (a) dal (b) -- sbordava: nel PDF consegnato il «(b)» finale risultava TAGLIATO,
        # e il manoscritto includeva la figura cosi'. Impilate, le due voci ci stanno intere.
        loc='outside upper center', ncol=1, frameon=False,
        handletextpad=0.4, labelspacing=0.3, fontsize=7.0)

    fig.savefig(OUT)                      # no tight bbox: the canvas IS the printed width
    fig.savefig(OUT.with_suffix('.png'), dpi=300)
    print(f'wrote {OUT.name} and .png')


if __name__ == '__main__':
    main()
