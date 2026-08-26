"""First natural frequency of the six DELIVERED stacking sequences.

Closes a provenance gap found by the 2026-07-21 journal review (rilievo H8(i)): the
frequencies quoted in the caption of tab:feasible (617, 617 and 671 Hz for C1, C2 and C3)
did not come from any script in this bundle. exp3 does not evaluate f1 at all -- see the
WARNING in its own docstring -- and exp3b covers the axial case only. Every number in the
manuscript has to be produced by a script here, so this one produces those.

It reads the delivered half-stacks from the exp3/exp3b artefacts, mirrors each about the
mid-plane and runs the shell modal deck once per design. Nothing is recomputed by hand.

    NPROC=6 python3 experiments/exp17_delivered_frequencies.py

Writes data/exp17_delivered_frequencies.json.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fe.reference_cases as RC

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, 'data')
FREQ_MIN_HZ = 600.0          # Constraint 4 of the manuscript


def _full(half):
    """Mirror a half-stack about the mid-plane: the laminates are symmetric by construction."""
    return list(half) + list(half)[::-1]


def _delivered():
    """Yield (case, orientation_set, source, half_stack) for every delivered design.

    C1 comes from exp3b, not exp3: the exp3 axial design sat at 46 plies and 590 Hz, i.e.
    below the frequency floor it was never checked against (audit A1, 2026-07-20).
    """
    exp3 = json.load(open(os.path.join(DATA, 'exp3_minply_sequences.json')))
    for oset, cases in exp3['delivered'].items():
        for case, rec in cases.items():
            if 'half_stack' not in rec:
                continue
            yield case, oset, 'exp3', rec['half_stack']

    path3b = os.path.join(DATA, 'exp3b_c1_freq_constrained.json')
    if os.path.exists(path3b):
        exp3b = json.load(open(path3b))
        for oset, rec in exp3b['designs'].items():
            yield 'c1_axial', oset, 'exp3b', rec['chosen']['half_stack']


def main() -> None:
    out = {'freq_min_Hz': FREQ_MIN_HZ, 'designs': []}
    for case, oset, source, half in _delivered():
        full = _full(half)
        f1 = RC.first_freq(full)
        row = {
            'case': case,
            'orientation_set': oset,
            'source': source,
            'n_plies': len(full),
            'f1_Hz': round(f1, 4),
            'freq_ok': bool(f1 > FREQ_MIN_HZ),
        }
        out['designs'].append(row)
        print('%-9s %-5s %-6s %2d ply  f1 = %8.2f Hz  %s'
              % (case, oset, source, len(full), f1,
                 'ok' if row['freq_ok'] else 'BELOW THE %.0f Hz FLOOR' % FREQ_MIN_HZ),
              flush=True)

    dest = os.path.join(DATA, 'exp17_delivered_frequencies.json')
    with open(dest, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('written %s' % dest)


if __name__ == '__main__':
    main()
