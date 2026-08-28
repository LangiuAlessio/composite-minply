#!/usr/bin/env python3
"""exp18 -- the reference-load screen: the fourth failure mode, with an artefact behind it.

THE generator of the reference-load failure mode of the verification-layer section ("A fourth failure mode") and
of the percentage the abstract quotes for it. Until this script existed that number lived in a
docstring of exp9_experimental_validation.py and NOWHERE else -- no deck, no .dat, no data
file. The 2026-07-22 audit listed it under "debts in the repo"; this closes it, and in closing
it corrects it.

What the old docstring said, and why it could not be re-run. It reported a scale measured on
"an isotropic strip whose exact Euler load is 530.1 N (60x12 mesh)", giving 1123 N against a
sane-scale 549 N. No strip in this bundle has an Euler load of 530.1 N, and no material,
geometry or thickness was recorded with the number: the case was not reproducible, and
guessing a geometry that lands on 530.1 N would be fitting a story to an arithmetic
coincidence rather than measuring. So this script does not reconstruct that strip. It
reproduces the FAILURE MODE on the strips the manuscript actually names -- the free-edge
Wysmulski specimens of Section 4.2, which are in the bundle with their published loads.

Method, per strip. Solve once at a sane reference load (a tenth of the published critical
load, so the factor lands near 10 and the reference state is far below instability), which
fixes the FE answer. Then re-solve on a scale of reference loads spanning that answer,
including one ABOVE it. A linear eigenvalue analysis must return the same critical load at
every scale: BLF x |F| is invariant. Where it does not, CalculiX has returned a different
mode -- the eigenvalue solver searches a spectral window about unity, and when the first
buckling factor falls well below it the SECOND mode comes back instead, silently.

The signature. The ratio of the reported critical load to the sane-scale one should be the
ratio of the second to the first buckling load of a clamped-clamped column,
(8.9868/2pi)^2 = 2.0457 -- a property of the mode shapes, not of the mesh, which is why it
identifies the mechanism. Two ratios are reported, deliberately kept apart, because the
manuscript conflated them until 2026-07-25:

    ratio_vs_fe     reported / sane-scale FE      the error of the SOLVER alone
    ratio_vs_euler  reported / closed-form Euler  solver error PLUS FE discretisation

Only the first identifies the failure. The second adds the few percent by which a given mesh
misses the closed form, which is discretisation error and has nothing to do with the wrong
mode being returned.

Cost: four single ccx solves per strip on a 60x12 mesh, seconds each. Not a campaign.

Run:  CCX_BIN=ccx_2.21 PYTHONPATH=<repo>/code python3 -m experiments.exp18_reference_load_screen
"""
from __future__ import annotations

import json
import os

import numpy as np

from experiments.exp9_experimental_validation import (
    OUTDIR, WYSMULSKI, buckle_deck)
from fe.plate_model import parse_buckling_factors, run_ccx

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data',
                    'exp18_reference_load_screen.json')
MESH = (60, 12)
# Reference loads as multiples of the FE critical load found at a sane scale. 2.0 is the
# pathological case (reference state past the instability); the rest bracket it from below.
SCALE = [2.0, 1.0, 0.5, 0.25]
SANE_FACTOR_TARGET = 10.0                  # the sane solve aims at BLF ~ 10
MODE_RATIO = (8.9868 / (2 * np.pi)) ** 2   # 2nd/1st buckling load, clamped-clamped column


def solve_at(bm, ref: float, tag: str) -> tuple[float, float]:
    """One buckling solve at reference load `ref`. Returns (factor, factor * ref)."""
    os.makedirs(OUTDIR, exist_ok=True)
    job = os.path.join(OUTDIR, f'refscreen_{bm.key}_{tag}_{MESH[0]}x{MESH[1]}')
    with open(job + '.inp', 'w') as f:
        f.write(buckle_deck(bm, MESH[0], MESH[1], ref))
    run_ccx(job)
    factor = min(parse_buckling_factors(job + '.dat'))
    return factor, factor * ref


def screen(bm) -> dict:
    """The scale sweep on one strip. Returns the record that goes into the artefact."""
    ref_sane = bm.p_exp / SANE_FACTOR_TARGET
    f_sane, p_sane = solve_at(bm, ref_sane, 'sane')
    print(f'\n=== {bm.key}: published {bm.p_exp:7.1f} N')
    print(f'  sane scale   ref {ref_sane:7.1f} N   BLF {f_sane:8.4f}   Pcr = {p_sane:8.1f} N')

    rows, pathological = [], None
    for s in SCALE:
        ref = s * p_sane
        factor, p = solve_at(bm, ref, f's{str(s).replace(".", "p")}')
        dev = (p - p_sane) / p_sane * 100
        rows.append({'scale_of_pcr': s, 'ref_load_N': round(ref, 1),
                     'buckling_factor': round(factor, 4), 'p_critical_N': round(p, 1),
                     'dev_vs_sane_pct': round(dev, 3)})
        mark = ''
        if abs(dev) > 1.0:
            mark = '  <-- NOT the same eigenvalue'
            if pathological is None or abs(dev) > abs(pathological['dev_vs_sane_pct']):
                pathological = rows[-1]
        print(f'  ref {ref:7.1f} N ({s:4.2f} x Pcr)   BLF {factor:8.4f}   '
              f'Pcr = {p:8.1f} N   {dev:+7.2f}%{mark}')

    rec = {'key': bm.key, 'source': bm.source, 'p_published_N': bm.p_exp,
           'mesh': {'nelx': MESH[0], 'nely': MESH[1]},
           'sane': {'ref_load_N': round(ref_sane, 1), 'buckling_factor': round(f_sane, 4),
                    'p_critical_N': round(p_sane, 1)},
           'scale_rows': rows}
    if pathological:
        rec['pathological'] = {
            **pathological,
            'ratio_vs_fe': round(pathological['p_critical_N'] / p_sane, 4),
            'error_vs_fe_pct': round((pathological['p_critical_N'] / p_sane - 1) * 100, 1),
            'caught_by_factor_gt_1': bool(pathological['buckling_factor'] <= 1.0),
        }
    return rec


def main() -> None:
    print(f'=== reference-load screen on the free-edge strips, {MESH[0]}x{MESH[1]} mesh')
    records = [screen(bm) for bm in WYSMULSKI]

    hit = [r for r in records if 'pathological' in r]
    out = {'mesh': {'nelx': MESH[0], 'nely': MESH[1]},
           'mode_ratio_closed_form': round(MODE_RATIO, 4),
           'scale_of_pcr_swept': SCALE,
           'n_strips': len(records), 'n_strips_with_wrong_mode': len(hit),
           'strips': records}
    if hit:
        ratios = [r['pathological']['ratio_vs_fe'] for r in hit]
        out['ratio_vs_fe_mean'] = round(float(np.mean(ratios)), 4)
        out['error_vs_fe_pct_mean'] = round((float(np.mean(ratios)) - 1) * 100, 1)

    with open(DATA, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'\nwrote {os.path.relpath(DATA)}')

    if not hit:
        raise SystemExit('exp18 canary FAILED: no strip returned a different eigenvalue at any '
                         'scale, so the failure mode the verification-layer section reports did not reproduce')

    print(f"\n  wrong mode on {len(hit)}/{len(records)} strips")
    print(f"  ratio vs sane FE: {out['ratio_vs_fe_mean']:.4f} "
          f"(closed-form second/first mode {MODE_RATIO:.4f})")
    print(f"  solver error:     {out['error_vs_fe_pct_mean']:+.1f}%")
    for r in hit:
        p = r['pathological']
        print(f"    {r['key']}: reported factor {p['buckling_factor']:.4f}, "
              f"ratio {p['ratio_vs_fe']:.4f}, "
              f"'factor > 1' check catches it: {p['caught_by_factor_gt_1']}")

    fail = []
    if abs(out['ratio_vs_fe_mean'] - MODE_RATIO) / MODE_RATIO > 0.02:
        fail.append(f"the ratio is {out['ratio_vs_fe_mean']:.4f}, not the second-to-first mode "
                    f"ratio {MODE_RATIO:.4f}: the mechanism named in the verification-layer section -- the second "
                    f"mode returned in place of the first -- is not what these data show")
    if any(r['pathological']['caught_by_factor_gt_1'] for r in hit):
        fail.append('at least one pathological run reports a factor below unity, so the '
                    '"factor > 1" check WOULD have caught it and the section overstates how '
                    'invisible the failure is')
    if fail:
        raise SystemExit('exp18 canary FAILED:\n  - ' + '\n  - '.join(fail))
    print('canaries: ratio = second-to-first mode ratio, and no pathological run is caught by '
          'the "factor > 1" check')


if __name__ == '__main__':
    main()
