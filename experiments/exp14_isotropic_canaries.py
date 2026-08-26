#!/usr/bin/env python3
"""exp14 -- the two isotropic calibration canaries, with a generator instead of orphan decks.

The manuscript states four closed-form calibration checks (Section "Validation against
Published Experiments"). Two of them were reproducible from the bundle -- the simply
supported cross-ply plate against classical laminated-plate theory, and the clamped strip
against the Euler column, both computed by exp9. The other two were NOT: the isotropic
plate at the exact coefficient k=4 and the clamped-loaded NASA-rig configuration against a
Rayleigh-Ritz solution at k=6.743 survived only as .dat files in _out/ with no code that
produced them. The 2026-07-20 pre-submission audit flagged that, and this script closes it.

Reference. For a square isotropic plate of side b and bending stiffness
D = E t^3 / (12 (1 - nu^2)) under a uniform uniaxial edge compression, the critical edge
resultant is N_cr = k pi^2 D / b^2 and the total edge load is P_cr = N_cr b, with the
coefficient k fixed by the boundary conditions: k = 4 for four simply supported edges, and
k = 6.743 for the NASA TP-3007 rig (loaded ends clamped, unloaded edges simply supported).

Result of the reconstruction (2026-07-20, ccx 2.21, 40x40 mesh):

    k = 4.000  SS isotropic                FE 5078.13 N  exact 5150.16 N   -1.398%
    k = 6.743  clamped-loaded (Ritz)       FE 8621.29 N  exact 8681.88 N   -0.698%

The first confirms the 1.4% printed in the manuscript. The second does NOT confirm the 0.8%
that the manuscript used to print: it is 0.7%, and the manuscript has been corrected.

Transverse shear. Both residuals are dominated by shear softening, not by the mesh: the
closed-form coefficients are Kirchhoff results, while the S8R shell is shear-deformable.
Running the k=4 case with G13=G23 scaled by 1000 (the Kirchhoff limit) moves the residual
from 1.40% to 0.89%, so roughly two thirds of it is physics the closed form omits. This is
why the two orphan decks in _out/exp9 (iso_1_40x40 and iso_1000_40x40) differ ONLY in G23:
they are the two ends of that comparison, and reading the wrong one gives the wrong residual.

Run:  CCX_BIN=ccx_2.21 PYTHONPATH=<repo>/code python3 -m experiments.exp14_isotropic_canaries
"""
from __future__ import annotations

import json
import math
import os

from experiments.exp9_experimental_validation import Benchmark, p_critical, al_6061

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_out', 'exp14')

# the isotropic canary plate, as fixed by the decks in _out/exp9
E, NU, T, B = 75842.327, 0.33, 1.64338, 241.3
MESH = (40, 40)

CASES = [
    ('k4_ss', 4.000, 'ss', 1.0, 'simply supported on four edges'),
    ('k6743_clamped', 6.743, 'clamped_loaded', 1.0, 'NASA TP-3007 rig, Rayleigh-Ritz'),
    ('k4_ss_kirchhoff', 4.000, 'ss', 1000.0, 'as k4_ss but G13=G23 x1000 (Kirchhoff limit)'),
]


def iso_mat(g_scale: float) -> dict:
    g = E / (2 * (1 + NU)) * g_scale
    mat = al_6061()
    mat.update(E1=E, E2=E, E3=E, nu12=NU, nu13=NU, nu23=NU,
               G12=E / (2 * (1 + NU)), G13=g, G23=g)
    return mat


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    D = E * T ** 3 / (12 * (1 - NU ** 2))
    results = []
    for key, k, bc, g_scale, note in CASES:
        exact = k * math.pi ** 2 * D / B ** 2 * B
        bm = Benchmark(key=key, source='closed form', a=B, b=B, layup=[0], ply_t=T,
                       mat=iso_mat(g_scale), bc=bc, p_exp=exact, notes=note)
        p_fe = p_critical(bm, *MESH)
        dev = (p_fe - exact) / exact * 100
        print(f'{key:<18} k={k:<6} {note}')
        print(f'   FE {p_fe:9.2f} N   exact {exact:9.2f} N   deviation {dev:+.3f}%\n')
        results.append(dict(key=key, k=k, bc=bc, g_scale=g_scale,
                            p_fe_N=p_fe, p_exact_N=exact, dev_pct=dev, note=note))

    dest = os.path.join(OUTDIR, 'exp14_results.json')
    with open(dest, 'w') as fh:
        json.dump(dict(D_Nmm=D, plate=dict(E=E, nu=NU, t=T, b=B), mesh=list(MESH),
                       results=results), fh, indent=2)
    print(f'written: {dest}')


if __name__ == '__main__':
    main()
