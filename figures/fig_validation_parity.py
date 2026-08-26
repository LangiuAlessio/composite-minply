#!/usr/bin/env python3
"""Figure: the evaluator against thirteen measured buckling loads (Table 8).

Panel (a) parity plot, predicted vs measured, three sources, three decades of load.
Panel (b) the same thirteen deviations against the prediction each source published
          with its own tests.

Behind both panels sits the envelope by which the two NASA campaigns disagree with
each other on the same nominal plates (5.0-10.1%): no model can be validated more
finely than the experiment can repeat itself.

Every number is read from code/experiments/_out/exp9/exp9_all_sources.json; the
prediction published by each source is recovered exactly from the deviation the
experiment script recorded against it. The script refuses to plot unless it
reproduces the manuscript's Table 8 to the digit it prints (canary check).

    python3 code/figures/fig_validation_parity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import (AMBER, AXIS, BLUE, GREEN, GRID, INK, MDPI_LINEWIDTH_IN, MUTED,
                    REF as RIVAL, SECOND, SURFACE, style)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'code' / 'experiments' / '_out' / 'exp9' / 'exp9_all_sources.json'
OUT = ROOT / 'RR_experimental_validation.pdf'

LB = 4.4482216152605  # N per lbf, as in exp9_experimental_validation.py

# the three sources, in fixed slot order; shape carries identity too, so the figure
# survives colour-blind readers and a black-and-white printer
SOURCES = [
    ('tp3007', 'NASA TP-3007 (1990)', BLUE, 'o'),
    ('tp2528', 'NASA TP-2528 (1986)', GREEN, 's'),
    ('wysmulski', 'Wysmulski (2024)', AMBER, '^'),
]
# specimen labels, in the order the source tabulates them
LABELS = {
    'A1_aluminium': 'A1', 'B1_0_10s': 'B1', 'C1_90_10s': 'C1', 'D1_0_90_5s': 'D1',
    'E1_pm30_6s': 'E1', 'F1_pm45_6s': 'F1', 'G1_pm60_6s': 'G1',
    'N26_A1_0_10s': 'A1', 'N26_B1_90_10s': 'B1', 'N26_C1_0_90_5s': 'C1',
    'W_S1': 'S1', 'W_S2': 'S2', 'W_S3': 'S3',
}
# the two campaigns tested these nominally identical plates four years apart
MATCHED = [('B1_0_10s', 'N26_A1_0_10s'), ('C1_90_10s', 'N26_B1_90_10s'),
           ('D1_0_90_5s', 'N26_C1_0_90_5s')]

# Table 8 of the manuscript: measured, this evaluator, source's own [lb, lb, lb / N]
TABLE8 = {
    # A1 evaluator was 1936 until 2026-07-20; the FE gives 1937.48 lb, and the +3.5% printed
    # beside it is 1937.48/1872, not 1936/1872 (which would be +3.4%). Audit finding B9: the
    # old canary's +-1 lb tolerance is exactly what let this through.
    'A1_aluminium': (1872, 1937, 1773), 'B1_0_10s': (9256, 9172, 9272),
    'C1_90_10s': (2292, 2456, 2473), 'D1_0_90_5s': (6950, 6493, 6544),
    'E1_pm30_6s': (10105, 9487, 9898), 'F1_pm45_6s': (9651, 9945, 10962),
    'G1_pm60_6s': (5790, 5631, 5944), 'N26_A1_0_10s': (8406, 8447, 8519),
    'N26_B1_90_10s': (2183, 2199, 2208), 'N26_C1_0_90_5s': (6484, 6493, 6539),
    'W_S1': (187, 200, 199), 'W_S2': (610, 652, 649), 'W_S3': (639, 696, 694),
}


def load():
    d = json.loads(DATA.read_text())
    rows = []
    for key, _, _, _ in SOURCES:
        for r in d[key]:
            # the source's own prediction, recovered from the deviation of our FE
            # against it: p_rival = p_fe / (1 + dev_vs_rival/100)
            p_rival = r['p_fe_N'] / (1.0 + r['dev_vs_rival_pct'] / 100.0)
            rows.append(dict(
                src=key, key=r['key'], label=LABELS[r['key']],
                p_exp=r['p_exp_N'], p_fe=r['p_fe_N'], p_rival=p_rival,
                dev=r['dev_vs_experiment_pct'],
                dev_rival=(p_rival - r['p_exp_N']) / r['p_exp_N'] * 100.0,
            ))
    return rows


# The percentages the table prints beside each load, (evaluator, source's own), in per cent.
# The loads alone are NOT enough to guard the table: a reader reads the percentages, and before
# 2026-07-20 nothing checked them (audit finding B9). They are recomputed here from the loads
# and compared against what is printed.
TABLE8_PCT = {
    'A1_aluminium': (+3.5, -5.3), 'B1_0_10s': (-0.9, +0.2), 'C1_90_10s': (+7.1, +7.9),
    'D1_0_90_5s': (-6.6, -5.8), 'E1_pm30_6s': (-6.1, -2.0), 'F1_pm45_6s': (+3.0, +13.6),
    'G1_pm60_6s': (-2.7, +2.7), 'N26_A1_0_10s': (+0.5, +1.3), 'N26_B1_90_10s': (+0.7, +1.1),
    'N26_C1_0_90_5s': (+0.1, +0.8), 'W_S1': (+6.9, +6.4), 'W_S2': (+6.9, +6.4),
    'W_S3': (+9.0, +8.6),
}
# means the table prints: all 13 rows, then the 10 rows that carry distinct FE models
PRINTED_MEANS = {'all13_ours': 4.2, 'all13_rival': 4.8, 'ten_ours': 5.3, 'ten_rival': 5.9}
# the three rows of the 1986 campaign re-evaluate the same FE model as their 1990 counterparts
SHARED_MODEL_KEYS = ('N26_A1_0_10s', 'N26_B1_90_10s', 'N26_C1_0_90_5s')


def canary(rows):
    """Refuse to plot a figure that does not reproduce the table it illustrates.

    Guards three things, not one: the loads, the percentages printed beside them, and the
    means in the last two rows. The load check is EXACT to the printed digit -- the old
    `abs(round(got) - want) > 1` tolerance let a 1937-vs-1936 discrepancy through, which is
    precisely the kind of drift a canary exists to catch.
    """
    bad = []
    for r in rows:
        unit = 1.0 if r['src'] == 'wysmulski' else LB   # NASA reports in lbf
        exp, fe, rival = TABLE8[r['key']]
        for name, got, want in (('measured', r['p_exp'] / unit, exp),
                                ('evaluator', r['p_fe'] / unit, fe),
                                ("source's own", r['p_rival'] / unit, rival)):
            if round(got) != want:              # the table prints whole lb / N, exactly
                bad.append(f"{r['key']:>15s} {name:>13s}: figure {got:9.1f} != table {want}")
        # the percentages, recomputed from the loads rather than trusted
        want_ours, want_rival = TABLE8_PCT[r['key']]
        for name, got, want in (('dev evaluator', r['dev'], want_ours),
                                ('dev source', r['dev_rival'], want_rival)):
            if abs(round(got, 1) - want) > 0.05:
                bad.append(f"{r['key']:>15s} {name:>13s}: figure {got:+7.2f}% "
                           f"!= table {want:+.1f}%")

    mean_ours = sum(abs(r['dev']) for r in rows) / len(rows)
    mean_rival = sum(abs(r['dev_rival']) for r in rows) / len(rows)
    ten = [r for r in rows if r['key'] not in SHARED_MODEL_KEYS]
    ten_ours = sum(abs(r['dev']) for r in ten) / len(ten)
    ten_rival = sum(abs(r['dev_rival']) for r in ten) / len(ten)
    for name, got, want in (('mean all 13, ours', mean_ours, PRINTED_MEANS['all13_ours']),
                            ('mean all 13, source', mean_rival, PRINTED_MEANS['all13_rival']),
                            ('mean 10 rows, ours', ten_ours, PRINTED_MEANS['ten_ours']),
                            ('mean 10 rows, source', ten_rival, PRINTED_MEANS['ten_rival'])):
        if abs(round(got, 1) - want) > 0.05:
            bad.append(f"{name:>24s}: figure {got:.2f}% != table {want:.1f}%")

    if len(ten) != 10:
        bad.append(f'expected 10 distinct-model rows, got {len(ten)}')

    if bad:
        sys.exit('CANARY FAILED - figure disagrees with Table 8:\n  ' + '\n  '.join(bad))

    print(f'canary ok: 13/13 loads and 26/26 percentages match Table 8 '
          f'(mean |dev| ours {mean_ours:.1f}%, source\'s own {mean_rival:.1f}%; '
          f'on the 10 distinct-model rows {ten_ours:.1f}% and {ten_rival:.1f}%)')
    return mean_ours, mean_rival


def repeatability(rows):
    """The envelope: how far the two NASA campaigns are from each other."""
    by_key = {r['key']: r for r in rows}
    gaps = [abs(by_key[a]['p_exp'] - by_key[b]['p_exp']) / by_key[b]['p_exp'] * 100.0
            for a, b in MATCHED]
    print('experimental repeatability, same nominal plate, two campaigns: '
          + ', '.join(f'{g:.1f}%' for g in sorted(gaps)))
    return min(gaps), max(gaps)




def main():
    rows = load()
    mean_ours, mean_rival = canary(rows)
    lo, hi = repeatability(rows)          # 5.0%, 10.1%

    style()
    # 5.5 in is \linewidth of the MDPI class. The figure must LEAVE this script at exactly that
    # width: a tight bbox that comes out wider is scaled down by LaTeX, and the type shrinks with
    # it (the first cut of this figure printed its 7.5 pt labels at 5.7 pt). A constrained layout
    # fits every artist inside the 5.5 in canvas instead of growing the canvas around them, so a
    # point here is a point on paper. The width is the scarce resource, so the two panels share
    # ONE legend along the top instead of carrying one each.
    fig = plt.figure(figsize=(MDPI_LINEWIDTH_IN, 3.25), layout='constrained')
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.06, hspace=0.0)
    ax, bx = fig.subplots(1, 2, width_ratios=[1.0, 1.45])

    # ---------------- (a) parity ----------------
    # No tolerance band here on purpose: across three decades a 10% envelope is
    # thinner than the identity line itself, and drawing it would claim a
    # resolution the axis cannot show. The envelope belongs to panel (b).
    span = [140.0, 60000.0]
    ax.plot(span, span, color=AXIS, lw=0.8, zorder=1)

    for key, name, colour, marker in SOURCES:
        pts = [r for r in rows if r['src'] == key]
        ax.plot([r['p_exp'] for r in pts], [r['p_fe'] for r in pts], marker,
                ms=5.6, color=colour, mec=SURFACE, mew=0.9, ls='none', zorder=3, label=name)

    ax.set(xscale='log', yscale='log', xlim=span, ylim=span)
    ax.set_xlabel('Measured buckling load [N]')
    ax.set_ylabel('Predicted buckling load [N]', labelpad=1)
    ax.set_aspect('equal')
    ax.set_anchor('N')          # the square box hangs from the top, so (a) and (b) align
    ax.grid(True, which='major', zorder=0)
    ax.tick_params(length=2.5)
    ax.text(0.95, 0.07, 'Three decades of load,\nno tuned parameter',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=6.6, color=SECOND,
            linespacing=1.4)
    ax.set_title('(a)  Predicted vs measured', loc='left', color=INK, pad=5)

    # ---------------- (b) deviations, ours vs the source's own ----------------
    xs, ticks = [], []
    x = 0.0
    for key, name, _, _ in SOURCES:
        for r in [r for r in rows if r['src'] == key]:
            xs.append((x, r))
            ticks.append((x, r['label']))
            x += 1.0
        x += 1.3                                   # air between the three sources

    lim = [-0.75, x - 1.15]
    for band, alpha in ((hi, 0.55), (lo, 0.75)):
        bx.fill_between(lim, -band, band, color=GRID, alpha=alpha, lw=0, zorder=0)
    bx.plot(lim, [0, 0], color=AXIS, lw=0.8, zorder=1)
    for band in (lo, hi):
        for sign in (-1, 1):
            bx.plot(lim, [sign * band] * 2, color=AXIS, lw=0.5, zorder=1)

    for xi, r in xs:
        bx.plot([xi, xi], [r['dev'], r['dev_rival']], color=AXIS, lw=0.7, zorder=2)
    for xi, r in xs:
        bx.plot(xi, r['dev_rival'], 'o', ms=4.2, mfc=SURFACE, mec=RIVAL, mew=0.9, zorder=3)
    for key, _, colour, marker in SOURCES:
        for xi, r in [(xi, r) for xi, r in xs if r['src'] == key]:
            bx.plot(xi, r['dev'], marker, ms=5.6, color=colour, mec=SURFACE, mew=0.9, zorder=4)

    bx.set(xlim=lim, ylim=(-16.0, 18.5))
    bx.set_xticks([t for t, _ in ticks])
    bx.set_xticklabels([l for _, l in ticks], fontsize=6.4)
    bx.set_yticks([-15, -10, -5, 0, 5, 10, 15])
    bx.set_yticklabels(['-15%', '-10%', '-5%', '0', '+5%', '+10%', '+15%'])
    bx.set_ylabel('Deviation from measurement', labelpad=1)
    bx.grid(True, axis='y', zorder=0)
    bx.tick_params(length=2.5)
    for spine in ('top', 'right'):
        bx.spines[spine].set_visible(False)

    # the two numbers the panel exists to compare, and the band it compares them to
    bx.text(lim[0] + 0.2, 17.6,
            f'mean error {mean_ours:.1f}% (filled)  ·  {mean_rival:.1f}% (open)',
            ha='left', va='top', fontsize=6.8, color=INK)
    bx.text(lim[0] + 0.2, -12.0,
            f'the two campaigns disagree with each other by {lo:.1f}-{hi:.1f}%\n'
            'on the same nominal plates: the resolution of the test',
            ha='left', va='center', fontsize=6.6, color=SECOND, linespacing=1.4)
    bx.set_title("(b)  Ours vs the source's own", loc='left', color=INK, pad=5)

    # one legend for both panels: shape = source (a and b), fill = whose prediction (b)
    handles = [Line2D([], [], marker=m, ls='none', ms=5.6, color=c, mec=SURFACE, mew=0.9,
                      label=name) for _, name, c, m in SOURCES]
    handles.append(Line2D([], [], marker='o', ls='none', ms=4.2, mfc=SURFACE, mec=RIVAL,
                          mew=0.9, label="the source's own prediction"))
    fig.legend(handles=handles, loc='outside upper center', ncol=2,
               frameon=False, handletextpad=0.4, columnspacing=2.2, labelspacing=0.3)

    fig.savefig(OUT)                      # no tight bbox: the canvas IS the printed width
    fig.savefig(OUT.with_suffix('.png'), dpi=300)
    print(f'wrote {OUT.relative_to(ROOT)} and .png')


if __name__ == '__main__':
    main()
