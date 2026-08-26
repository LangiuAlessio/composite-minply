#!/usr/bin/env python3
"""exp9 -- validate the FE evaluator against PUBLISHED EXPERIMENTAL buckling tests.

Every validation in the paper so far is numerical: CalculiX against Abaqus (exp1), against
the Haftka--Walsh integer-programming optima (exp6), against an exhaustive enumeration
(exp7). A referee can accept all of that and still ask the question the Solids editor asked:
does the model agree with a plate that was actually built and crushed?

This experiment answers it in the only way a computational paper can without a laboratory:
it rebuilds, from the numbers printed in the source papers, composite plates whose buckling
loads were MEASURED, and checks what our evaluator predicts for them. Nothing is tuned. The
benchmark data are entered once, in BENCHMARKS, with the citation attached; if a source does
not print E1, E2, G12, nu12, the stacking sequence, the geometry and the measured load, it
cannot be used here and is not listed.

Protocol (the point is that it cannot be gamed):
  1. the model is built from the paper's numbers only -- no fitting parameter, no calibration;
  2. mesh convergence is checked first, and the converged mesh is used for the comparison,
     so a lucky agreement at a coarse mesh cannot be mistaken for physics;
  3. the predicted critical load is compared to the MEASURED one, and the deviation is
     reported as it comes out.

Run:  python3 -m experiments.exp9_experimental_validation
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from fe.plate_model import (build_mesh, _common_model, _corner_node, run_ccx,
                            parse_buckling_factors)

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_out', 'exp9')
REF_LOAD = 1000.0        # N, reference edge resultant; Pcr = BF * REF_LOAD


@dataclass
class Benchmark:
    """One published experiment. Fields are what the source paper prints -- nothing else."""
    key: str
    source: str                  # citation, with the bib key used in the manuscript
    a: float                     # mm, loaded (compression) direction
    b: float                     # mm, transverse
    layup: list                  # full stacking sequence, degrees
    ply_t: float                 # mm per ply
    mat: dict                    # E1,E2,E3,nu12,nu13,nu23,G12,G13,G23 (MPa), rho
    bc: str                      # 'ss' (simply supported 4 edges) | 'clamped_unloaded'
    p_exp: float                 # N, MEASURED critical load
    p_exp_note: str = ''         # how the source defines it (e.g. from strain reversal)
    notes: str = ''
    t_meas: float = 0.0          # mm, TOTAL measured average thickness (p. 2 of the report),
    #                              which for two specimens differs from the H the author used
    #                              in his own calculations (Table 8). Pcr goes as t^3, so this
    #                              is not a detail: 0.107 in vs 0.110 in is 9% of the load.


# ---------------------------------------------------------------------------
# The benchmark: NASA TP-3007 (Nemeth 1990). Square graphite-epoxy plates compressed to
# buckling, plus one aluminium plate. Every number below was read off the report's own
# tables (pp. 2, 14-17 of the PDF on NTRS) and re-checked against the scanned pages, not
# taken from a secondary source: fabricating a benchmark number would invalidate the very
# thing this experiment exists to establish.
#
#   material  (p. 2): AS4/3502, E1 = 18.5e6 psi, E2 = 1.6e6 psi, G12 = 0.832e6 psi,
#                     nu12 = 0.35; aluminium 6061-T6, E = 11.0e6 psi, nu = 0.33
#   geometry  (p. 3): the analysis geometry is the SQUARE 9.5 in x 9.5 in unsupported region
#                     between the knife-edge supports -- not the 10 in specimen
#   BCs       (p. 3): "The loaded edges were assumed to be clamped, and the unloaded edges
#                     were assumed to be simply supported."
#   thickness (Table 8): nominal H used by the author in his own calculations
#   measured  (Tables 1-7, the d/W = 0 rows, i.e. the plates WITHOUT a cutout)
#
# Unit system: the report is imperial, the evaluator is N-mm-MPa.
IN = 25.4                 # mm per inch
PSI = 6.894757e-3         # MPa per psi
LB = 4.4482216            # N per pound-force

# Out-of-plane constants (E3, nu13, nu23, G13, G23) are NOT printed in TP-3007 -- no 1990
# plate-buckling paper prints them. They are set by the standard transversely-isotropic
# assumptions (E3 = E2, nu13 = nu12, G13 = G12, G23 = 0.5 G12). They enter only through the
# shell's transverse-shear stiffness, which for these thin plates shifts the load by ~1%
# (the canary below quantifies exactly that effect against the closed form).
def as4_3502() -> dict:
    e1, e2, g12, nu12 = 18.5e6 * PSI, 1.6e6 * PSI, 0.832e6 * PSI, 0.35
    return dict(E1=e1, E2=e2, E3=e2, nu12=nu12, nu13=nu12, nu23=0.4,
                G12=g12, G13=g12, G23=0.5 * g12, rho=1.58e-9)


def al_6061() -> dict:
    e, nu = 11.0e6 * PSI, 0.33
    g = e / (2 * (1 + nu))
    return dict(E1=e, E2=e, E3=e, nu12=nu, nu13=nu, nu23=nu,
                G12=g, G13=g, G23=g, rho=2.70e-9)


def _sym(half: list) -> list:
    return half + half[::-1]


W = 9.5 * IN              # 241.3 mm, square plate

BENCHMARKS: list[Benchmark] = [
    # aluminium: the second canary. Isotropic, answer known from BOTH the experiment and the
    # author's own analysis -- if the harness cannot do THIS plate, nothing below it counts.
    Benchmark(key='A1_aluminium', source='NASA TP-3007 Table 1 (Nemeth 1990), specimen A1',
              a=W, b=W, layup=[0], ply_t=0.0647 * IN, mat=al_6061(), bc='clamped_loaded',
              p_exp=1872 * LB, p_exp_note='measured; author analytic 1773 lb', t_meas=0.0647 * IN,
              notes='6061-T6, single isotropic layer'),
    Benchmark(key='B1_0_10s', source='NASA TP-3007 Table 2, specimen B1',
              a=W, b=W, layup=[0] * 20, ply_t=0.1100 * IN / 20, mat=as4_3502(),
              bc='clamped_loaded', p_exp=9256 * LB,
              p_exp_note='measured; author analytic 9272 lb', t_meas=0.1070 * IN),
    Benchmark(key='C1_90_10s', source='NASA TP-3007 Table 3, specimen C1',
              a=W, b=W, layup=[90] * 20, ply_t=0.1100 * IN / 20, mat=as4_3502(),
              bc='clamped_loaded', p_exp=2292 * LB,
              p_exp_note='measured; author analytic 2473 lb', t_meas=0.1060 * IN,
              notes='source records TWO axial half-waves for this specimen: the FE must find '
                    'that mode by itself, it is not imposed'),
    Benchmark(key='D1_0_90_5s', source='NASA TP-3007 Table 4, specimen D1',
              a=W, b=W, layup=_sym([0, 90] * 5), ply_t=0.1100 * IN / 20, mat=as4_3502(),
              bc='clamped_loaded', p_exp=6950 * LB,
              p_exp_note='measured; author analytic 6544 lb', t_meas=0.1100 * IN),
    Benchmark(key='E1_pm30_6s', source='NASA TP-3007 Table 5, specimen E1',
              a=W, b=W, layup=_sym([30, -30] * 6), ply_t=0.1176 * IN / 24, mat=as4_3502(),
              bc='clamped_loaded', p_exp=10105 * LB,
              p_exp_note='measured; author analytic 9898 lb', t_meas=0.1176 * IN),
    Benchmark(key='F1_pm45_6s', source='NASA TP-3007 Table 7, specimen F1',
              a=W, b=W, layup=_sym([45, -45] * 6), ply_t=0.1300 * IN / 24, mat=as4_3502(),
              bc='clamped_loaded', p_exp=9651 * LB,
              p_exp_note='measured (from out-of-plane displacements); author analytic 10962 lb', t_meas=0.1307 * IN),
    Benchmark(key='G1_pm60_6s', source='NASA TP-3007 Table 6, specimen G1',
              a=W, b=W, layup=_sym([60, -60] * 6), ply_t=0.1176 * IN / 24, mat=as4_3502(),
              bc='clamped_loaded', p_exp=5790 * LB,
              p_exp_note='measured; author analytic 5944 lb', t_meas=0.1176 * IN),
]

# the author's own analytical predictions (Table 8, lb) -- a second yardstick: our FE and his
# in-house buckling code solve the same idealised problem, so they should agree with EACH
# OTHER more closely than either agrees with the test.
# VERIFICATI sulla fonte primaria il 2026-08-26: NASA TP-3007, Table 8 "Analytic Buckling Loads,
# Critical End-Shortenings, and Nominal Thicknesses for Plates Without Cutouts", pagina 17 della
# scansione pubblica NTRS (19900016761). Tutti e sette coincidono, e coincidono anche gli spessori
# nominali della stessa tabella con quelli usati qui. L'audit del 2026-07-20 li dava per NON
# verificati e "plausibilmente digitalizzati da figura": era sbagliato. Attenzione se qualcuno
# ricontrolla: l'OCR della scansione del 1990 scrive "9 272" con lo spazio delle migliaia e un
# carico misurato come "579O", con la O maiuscola al posto dello zero -- una ricerca testuale
# delle cifre non trova nulla e sembra un'assenza.
ANALYTIC_LB = {'A1_aluminium': 1773, 'B1_0_10s': 9272, 'C1_90_10s': 2473, 'D1_0_90_5s': 6544,
               'E1_pm30_6s': 9898, 'F1_pm45_6s': 10962, 'G1_pm60_6s': 5944}


# ---------------------------------------------------------------------------
# SECOND SOURCE: NASA TP-2528 (Nemeth, Stein & Johnson 1986). Same laboratory, same material,
# same 9.5 in square, same boundary conditions ("The loaded ends of the specimens were clamped
# by fixtures during testing, and the sides were simply supported by restraints", p. 9) and
# THREE OF THE SAME LAMINATES as TP-3007 -- but a DIFFERENT test campaign, four years earlier.
#
# This is the most valuable thing the literature search turned up, and not for the reason one
# would expect. The two campaigns disagree with each other:
#
#     [0_10]s      TP-3007: 9256 lb      TP-2528: 8406 lb      -> 9.2 % apart
#     [90_10]s     TP-3007: 2292 lb      TP-2528: 2183 lb      -> 4.8 % apart
#     [(0/90)_5]s  TP-3007: 6950 lb      TP-2528: 6484 lb      -> 6.7 % apart
#
# Nominally identical plates, same lab. That spread IS the resolution of the experiment, and it
# bounds what any validation against it can honestly claim. The difference tracks the thickness
# each report used (TP-2528 states it used the measured average, TP-3007's table rounds it up):
# Pcr goes as t^3, and 0.107 vs 0.110 in is 9 % of the load. So we model each campaign at the
# thickness that campaign used, and let the reader see both.
TP2528: list[Benchmark] = [
    Benchmark(key='N26_A1_0_10s', source='NASA TP-2528 Table II, specimen A1 (1986 campaign)',
              a=W, b=W, layup=[0] * 20, ply_t=0.107 * IN / 20, mat=as4_3502(),
              bc='clamped_loaded', p_exp=8406 * LB, t_meas=0.107 * IN,
              p_exp_note='measured; report analytic 8519 lb; TP-3007 measured 9256 lb on the '
                         'same nominal plate'),
    Benchmark(key='N26_B1_90_10s', source='NASA TP-2528 Table III, specimen B1 (1986 campaign)',
              a=W, b=W, layup=[90] * 20, ply_t=0.106 * IN / 20, mat=as4_3502(),
              bc='clamped_loaded', p_exp=2183 * LB, t_meas=0.106 * IN,
              p_exp_note='measured; report analytic 2208 lb; TP-3007 measured 2292 lb',
              notes='again two axial half-waves, as in TP-3007'),
    Benchmark(key='N26_C1_0_90_5s', source='NASA TP-2528 Table IV, specimen C1 (1986 campaign)',
              a=W, b=W, layup=_sym([0, 90] * 5), ply_t=0.110 * IN / 20, mat=as4_3502(),
              bc='clamped_loaded', p_exp=6484 * LB, t_meas=0.110 * IN,
              p_exp_note='measured; report analytic 6539 lb; TP-3007 measured 6950 lb'),
]
ANALYTIC_LB.update({'N26_A1_0_10s': 8519, 'N26_B1_90_10s': 2208, 'N26_C1_0_90_5s': 6539})


# ---------------------------------------------------------------------------
# THIRD SOURCE: Wysmulski 2024 (Materials 17(5) 1081, DOI 10.3390/ma17051081). A DIFFERENT
# REGIME, which is the whole point of including it: the plate is gripped top and bottom
# (all six DOF blocked at the loaded ends) and its long edges touch nothing at all. Where the
# NASA rig supports the sides on knife edges, here they are free. A model that only ever
# reproduced one support condition would have been validated at a single point.
#
# The test area is 20 x 100 mm -- the 140 mm coupon has 20 mm in each grip -- so the modelled
# plate is the 100 mm free length, not the coupon.
def hexcel_uts() -> dict:
    e1, e2, g12, nu12 = 131.71e3, 6.36e3, 4.18e3, 0.32     # MPa, Table 2 of the paper
    return dict(E1=e1, E2=e2, E3=e2, nu12=nu12, nu13=nu12, nu23=0.4,
                G12=g12, G13=g12, G23=0.5 * g12, rho=1.55e-9)


WYS = [
    ('W_S1', [45, -45, 90, 0, 0, 90, -45, 45], 187.0),
    ('W_S2', [0, -45, 45, 90, 90, 45, -45, 0], 610.0),
    ('W_S3', [0, 90, 0, 90, 90, 0, 90, 0], 639.0),
]
WYSMULSKI: list[Benchmark] = [
    Benchmark(key=k, source=f'Wysmulski 2024, Materials 17(5) 1081, Table 3 (EXP, no hole), {k[-2:]}',
              a=100.0, b=20.0, layup=seq, ply_t=0.131, mat=hexcel_uts(),
              bc='clamped_loaded_free', p_exp=p, t_meas=8 * 0.131,
              p_exp_note='measured; the paper\'s own FE gives 199/649/694 N for S1/S2/S3',
              notes='critical load read by the straight-line intersection method on the '
                    'load-shortening path, not from a sharp bifurcation: a few percent of '
                    'method-dependent uncertainty is baked into the reference value')
     for k, seq, p in WYS]
WYS_OWN_FE = {'W_S1': 199.0, 'W_S2': 649.0, 'W_S3': 694.0}


# ---------------------------------------------------------------------------
# deck assembly: same model machinery as the paper, only the BCs are new
# ---------------------------------------------------------------------------
def _boundary(bc: str) -> list[str]:
    """Out-of-plane support on the four edges, plus the rotational restraint that
    distinguishes one test rig from another -- which is the single most consequential
    modelling choice here (a clamped edge can carry several times the load of a hinged one).

      'ss'                  simply supported all round (the closed-form canary case)
      'clamped_loaded'      the NASA TP-3007 rig: loaded ends clamped in the fixtures, unloaded
                            edges simply supported on knife edges
      'clamped_loaded_free' loaded ends clamped, unloaded edges COMPLETELY FREE -- the modern
                            testing-machine setup (specimen gripped top and bottom, nothing
                            touching the sides). A different regime from TP-3007, and the
                            reason to include it: a validation that only ever exercises one
                            support condition has only been validated at one point.
    """
    L = ['*BOUNDARY']
    edges = (('EDGE_X0', 'EDGE_X1') if bc == 'clamped_loaded_free'
             else ('EDGE_X0', 'EDGE_X1', 'EDGE_Y0', 'EDGE_Y1'))
    for s in edges:
        L.append(f'{s}, 3, 3')
    if bc in ('clamped_loaded', 'clamped_loaded_free'):
        for s in ('EDGE_X0', 'EDGE_X1'):
            L.append(f'{s}, 4, 6')      # the loaded ends cannot rotate: gripped by the fixture
    elif bc != 'ss':
        raise ValueError(f'unknown bc: {bc}')
    return L


def _compression(m, total: float) -> list[str]:
    """Uniform edge traction on x=a, lumped to the quadratic edge nodes with the
    consistent 1/6-4/6-1/6 weights so the resultant is exactly `total` newtons."""
    edge = sorted(set(m.edge_x1), key=lambda nid: m.nodes[nid][1])
    ys = [m.nodes[nid][1] for nid in edge]
    w = {nid: 0.0 for nid in edge}
    for s in range((len(edge) - 1) // 2):
        n0, n1, n2 = edge[2 * s], edge[2 * s + 1], edge[2 * s + 2]
        seg = ys[2 * s + 2] - ys[2 * s]
        w[n0] += seg / 6.0
        w[n1] += seg * 4.0 / 6.0
        w[n2] += seg / 6.0
    tot = sum(w.values())
    return ['*CLOAD'] + [f'{nid}, 1, {-total * w[nid] / tot:.8f}' for nid in edge]


def buckle_deck(bm: Benchmark, nelx: int, nely: int, ref: float = REF_LOAD) -> str:
    m = build_mesh(nelx, nely, lx=bm.a, ly=bm.b)
    L = _common_model(m, seq=bm.layup, mat=bm.mat, ply_t=bm.ply_t)
    L += _boundary(bm.bc)
    n_origin = _corner_node(m, 0.0, 0.0)
    n_yend = _corner_node(m, 0.0, bm.b)
    L += ['*STEP, PERTURBATION', '*BUCKLE', '4',
          '*BOUNDARY', 'EDGE_X0, 1, 1', f'{n_origin}, 2, 2', f'{n_yend}, 2, 2']
    L += _compression(m, ref)
    L.append('*END STEP')
    return '\n'.join(L) + '\n'


def p_critical(bm: Benchmark, nelx: int, nely: int) -> float:
    """Predicted critical load [N] = lowest buckling factor x reference load.

    A LINEAR eigenvalue buckling analysis is supposed to be independent of the magnitude of
    the reference load. CalculiX's is not, and it fails silently. The geometric stiffness is
    built from the stress state produced by the reference load; if that load EXCEEDS the
    critical one (buckling factor < 1) the reference state is past the instability and the
    eigenvalue comes back wrong -- on the Wysmulski strips, by +104%, converged in the mesh,
    with no warning of any kind. Mesh refinement does not catch this: the wrong answer is
    perfectly converged.

    MEASURED IN-BUNDLE, 2026-07-25: experiments/exp18_reference_load_screen.py sweeps the
    reference load on the three Wysmulski strips and writes data/exp18_reference_load_screen.json.
    Sane scales (a quarter, a half and once the critical load) return the same critical load to
    the digit; at twice the critical load the reported value jumps by a factor of 2.079 / 2.023 /
    2.018 (mean 2.040), against the closed-form second-to-first mode ratio (8.9868/2pi)^2 =
    2.0457 for a clamped-clamped column. The reported factors there are 1.040 / 1.011 / 1.009 --
    all ABOVE unity, which is why the obvious sanity check does not fire.

    This docstring used to carry a five-row table measured "on an isotropic strip whose exact
    Euler load is 530.1 N", reporting 1123 N against a sane-scale 549 N, i.e. +112%. That case
    was not reproducible: no strip in this bundle has that Euler load, and no geometry or
    thickness was recorded with it. The 2026-07-22 audit flagged the number as living in a
    docstring and nowhere else; the 2026-07-25 session replaced it with the measurement above
    and corrected the manuscript, where the same "+112%" had been quoted in the abstract. Note
    that 1123/549 = 2.0455 was itself the SOLVER error and matched the mode ratio: the +112%
    was that error plus the discretisation gap between 549 N and the 530.1 N closed form, and
    the text was calling both of them "the correct one".

    So the rule is: keep the reference load below the critical one. The manuscript's own
    campaign runs at BLF ~ 4 and is therefore in the safe region -- but by luck, not by
    design, and that is exactly the kind of silent FE failure mode Section 3.8 is about.
    The screen is now explicit: pick the reference load from the expected magnitude, and
    assert afterwards that the factor came out above 1.
    """
    os.makedirs(OUTDIR, exist_ok=True)
    ref = bm.p_exp / 10.0 if bm.p_exp else REF_LOAD    # target a factor of ~10, safely > 1
    job = os.path.join(OUTDIR, f'{bm.key}_{nelx}x{nely}')
    # I job persistono in _out/ e run_ccx non controlla il returncode: se ccx fallisce PRIMA di
    # scrivere (binario assente, deck rotto, licenza), il .dat della run precedente -- magari di
    # un altro materiale o di un'altra condizione al contorno -- veniva parsato come fresco. Le
    # difese esistenti (factor<1 -> RuntimeError, .dat assente -> FileNotFoundError) non lo
    # coprono. Si cancellano quindi gli output vecchi PRIMA del lancio, e si verifica che il
    # .dat sia stato riscritto dopo l'.inp.
    for ext in ('.dat', '.frd', '.sta', '.cvg'):
        if os.path.exists(job + ext):
            os.remove(job + ext)
    with open(job + '.inp', 'w') as f:
        f.write(buckle_deck(bm, nelx, nely, ref))
    inp_mtime = os.path.getmtime(job + '.inp')
    run_ccx(job)
    if not os.path.exists(job + '.dat'):
        raise RuntimeError(f'{bm.key}: ccx non ha scritto {job}.dat -- la solve non e\' girata')
    if os.path.getmtime(job + '.dat') < inp_mtime:
        raise RuntimeError(f'{bm.key}: {job}.dat e\' piu\' vecchio del deck -- e\' un residuo '
                           f'di una run precedente, non il risultato di questa')
    factor = min(parse_buckling_factors(job + '.dat'))
    if factor < 1.0:
        raise RuntimeError(f'{bm.key}: buckling factor {factor:.2f} < 1 -- the reference load '
                           f'is above the critical load and CalculiX returns a wrong eigenvalue')
    return factor * ref


def converge(bm: Benchmark, meshes=((20, 10), (30, 15), (40, 20), (60, 30))) -> list:
    """Mesh convergence: the comparison with the experiment is only meaningful on the
    converged mesh, so we show the sequence rather than a single lucky number."""
    rows = []
    for nelx, nely in meshes:
        p = p_critical(bm, nelx, nely)
        prev = rows[-1][2] if rows else None
        drift = abs(p - prev) / prev * 100 if prev else float('nan')
        rows.append((nelx, nely, p, drift))
        print(f'  {nelx:3d}x{nely:<3d}  Pcr = {p:9.1f} N   drift vs previous: {drift:5.2f}%')
    return rows


# ---------------------------------------------------------------------------
# canary: before pointing the harness at an experiment whose answer we do not know,
# point it at a case whose answer is KNOWN in closed form. A specially orthotropic
# symmetric laminate (D16=D26=0), simply supported on four edges under uniaxial
# compression, has the classical laminated-plate-theory critical load
#
#   Nx_cr = pi^2 [ D11 (m/a)^4 + 2(D12+2D66)(m/a)^2 (1/b)^2 + D22 (1/b)^4 ] / (m/a)^2
#
# minimised over the number of half-waves m. If the FE harness (mesh, BCs, load
# lumping, ply orientations) does not reproduce THIS, any agreement with a real
# experiment would be luck, and a disagreement would be uninterpretable.
# ---------------------------------------------------------------------------
def d_matrix(layup: list, mat: dict, ply_t: float):
    import numpy as np
    n = len(layup)
    h = n * ply_t
    z = [-h / 2 + k * ply_t for k in range(n + 1)]
    nu21 = mat['nu12'] * mat['E2'] / mat['E1']
    den = 1 - mat['nu12'] * nu21
    Q = np.array([[mat['E1'] / den, mat['nu12'] * mat['E2'] / den, 0],
                  [mat['nu12'] * mat['E2'] / den, mat['E2'] / den, 0],
                  [0, 0, mat['G12']]])
    D = np.zeros((3, 3))
    for k, ang in enumerate(layup):
        c, s = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
        T = np.array([[c**2, s**2, 2 * c * s],
                      [s**2, c**2, -2 * c * s],
                      [-c * s, c * s, c**2 - s**2]])
        R = np.diag([1., 1., 2.])
        Qbar = np.linalg.inv(T) @ Q @ R @ T @ np.linalg.inv(R)
        D += Qbar * (z[k + 1] ** 3 - z[k] ** 3) / 3.0
    return D


def clpt_pcr(layup: list, mat: dict, ply_t: float, a: float, b: float) -> float:
    """Closed-form critical LOAD [N] (= Nx_cr * b) for the SS orthotropic plate."""
    import numpy as np
    D = d_matrix(layup, mat, ply_t)
    best = float('inf')
    for m in range(1, 6):
        nx = np.pi ** 2 * (D[0, 0] * (m / a) ** 4
                           + 2 * (D[0, 1] + 2 * D[2, 2]) * (m / a) ** 2 * (1 / b) ** 2
                           + D[1, 1] * (1 / b) ** 4) / ((m / a) ** 2)
        best = min(best, nx)
    return best * b


def canary() -> float:
    """Cross-ply [0/90]_4s in the generic T300/epoxy lamina of fe/materials.py: FE vs closed
    form. Returns the deviation in percent. NOTE: this is not the campaign lamina of the
    paper's material table (canale2018, E1 125100, ply 0.1 mm); the check is a
    self-consistency test of the FE model against CLPT, so it is run on whichever lamina the
    plate model defaults to, and its validity does not depend on that choice."""
    from fe.plate_model import MAT
    layup = [0, 90] * 4
    layup = layup + layup[::-1]                 # symmetric, 16 plies
    bm = Benchmark(key='canary_clpt', source='closed-form CLPT (no experiment)',
                   a=200.0, b=100.0, layup=layup, ply_t=0.125, mat=MAT, bc='ss',
                   p_exp=clpt_pcr(layup, MAT, 0.125, 200.0, 100.0),
                   p_exp_note='analytical, not measured')
    print(f'=== canary: SS cross-ply plate, answer known in closed form')
    print(f'  CLPT closed form: {bm.p_exp:9.1f} N')
    rows = converge(bm, meshes=((20, 10), (40, 20), (60, 30), (80, 40)))
    p_fe = rows[-1][2]
    dev = (p_fe - bm.p_exp) / bm.p_exp * 100
    print(f'  FE (converged):   {p_fe:9.1f} N  -> deviation {dev:+.2f}%')
    return dev


def canary_strip() -> float:
    """Known-answer check for the OTHER support regime (clamped ends, free sides).

    A laminated strip clamped at both ends and free along its sides is an Euler column with
    fixed-fixed ends: P = 4 pi^2 (EI) / L^2. The trap is the bending stiffness. For a WIDE
    plate the transverse curvature is suppressed and EI = D11*b; for a strip with FREE sides
    the transverse moment vanishes instead, and the effective stiffness is EI = b / d11, with
    d11 taken from the INVERSE of the D matrix. The two differ by 4.5% on this laminate, and
    using the wide-plate value makes a correct FE look 3.6% wrong -- which is exactly what it
    did here before the formula was fixed.
    """
    import numpy as np
    mat = as4_3502()
    layup = _sym([0, 90, 45, -45] * 2)      # 16 plies, symmetric
    ply_t, a, b = 0.125, 200.0, 40.0
    D = d_matrix(layup, mat, ply_t)
    d11 = np.linalg.inv(D)[0, 0]
    p_euler = 4 * np.pi ** 2 * (b / d11) / a ** 2
    bm = Benchmark(key='canary_strip', source='closed-form Euler column (no experiment)',
                   a=a, b=b, layup=layup, ply_t=ply_t, mat=mat, bc='clamped_loaded_free',
                   p_exp=p_euler, p_exp_note='analytical, not measured')
    print('=== canary 2: clamped-clamped strip, free sides (Euler, answer known)')
    print(f'  Euler fixed-fixed: {p_euler:9.1f} N')
    rows = converge(bm, meshes=((40, 8), (60, 12), (80, 16)))
    p_fe = rows[-1][2]
    dev = (p_fe - p_euler) / p_euler * 100
    print(f'  FE (converged):    {p_fe:9.1f} N  -> deviation {dev:+.2f}%')
    return dev


def run_set(title: str, bms: list, rival: dict, rival_name: str, mesh=(50, 50),
            rival_unit: float = LB) -> list:
    """Run one source and compare with the experiment and with that source's own prediction.

    rival_unit converts the rival's numbers to newtons: the NASA tables are in pounds, the
    modern paper is already in newtons. Guessing this from the magnitude (as a first version
    of this code did) silently divides one of the two comparisons by 4.45."""
    print(f'\n########## {title}\n')
    out = []
    for bm in bms:
        p_fe = p_critical(bm, *mesh)
        d_exp = (p_fe - bm.p_exp) / bm.p_exp * 100
        r = rival.get(bm.key)
        r_N = r * rival_unit if r else None
        d_riv = (p_fe - r_N) / r_N * 100 if r_N else float('nan')
        print(f'  {bm.key:16s} measured {bm.p_exp:9.1f} N | our FE {p_fe:9.1f} N '
              f'-> {d_exp:+6.1f}%   ({rival_name}: {d_riv:+.1f}%)')
        out.append(dict(key=bm.key, source=bm.source, p_exp_N=bm.p_exp, p_fe_N=p_fe,
                        dev_vs_experiment_pct=d_exp, dev_vs_rival_pct=d_riv))
    d = [abs(r['dev_vs_experiment_pct']) for r in out]
    print(f'\n  --> vs experiment: mean |dev| {sum(d) / len(d):.1f}%, worst {max(d):.1f}%')
    return out


def main_all() -> None:
    """The full validation: three independent sources, two support regimes, two canaries."""
    c1, c2 = canary(), canary_strip()

    a = run_set('SOURCE 1 -- NASA TP-3007 (1990): 7 plates, clamped ends / simply supported sides',
                BENCHMARKS, ANALYTIC_LB, 'their analysis')
    b = run_set('SOURCE 2 -- NASA TP-2528 (1986): same lab, same plates, EARLIER campaign',
                TP2528, ANALYTIC_LB, 'their analysis')
    c = run_set('SOURCE 3 -- Wysmulski 2024: clamped ends, FREE sides (the other regime)',
                WYSMULSKI, WYS_OWN_FE, 'their FE', mesh=(60, 12), rival_unit=1.0)

    # What the two NASA campaigns say about EACH OTHER. This is the honest ceiling on any
    # validation claim: nominally identical plates, same laboratory, 5-9% apart.
    pairs = [('[0_10]s', 9256, 8406), ('[90_10]s', 2292, 2183), ('[(0/90)_5]s', 6950, 6484)]
    print('\n########## THE CEILING: the two NASA campaigns vs each other\n')
    for name, p07, p28 in pairs:
        print(f'  {name:14s} TP-3007 {p07:6d} lb   TP-2528 {p28:6d} lb   '
              f'-> {abs(p07 - p28) / p28 * 100:.1f}% apart')
    spread = [abs(x - y) / y * 100 for _, x, y in pairs]
    print(f'\n  Two real experiments on the same nominal plates disagree by '
          f'{min(spread):.1f}-{max(spread):.1f}%.')

    allr = a + b + c
    d = [abs(r['dev_vs_experiment_pct']) for r in allr]
    print(f'\n########## OVERALL: {len(allr)} measured plates, 3 sources, 2 support regimes')
    print(f'  our FE vs experiment: mean |dev| {sum(d) / len(d):.1f}%, worst {max(d):.1f}%')
    print(f'  canaries: closed form {c1:+.2f}%, Euler strip {c2:+.2f}%')

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, 'exp9_all_sources.json'), 'w') as f:
        json.dump(dict(canary_plate_pct=c1, canary_strip_pct=c2, tp3007=a, tp2528=b,
                       wysmulski=c, cross_campaign_spread_pct=spread), f, indent=2)
    print(f'\nwritten: {OUTDIR}/exp9_all_sources.json')


def main() -> None:
    if not BENCHMARKS:
        raise SystemExit('BENCHMARKS is empty: enter the verified experimental cases first.')
    dev_canary = canary()
    results = []
    print()
    for bm in BENCHMARKS:
        print(f'=== {bm.key}: {bm.source}')
        rows = converge(bm, meshes=((30, 30), (50, 50)))
        p_fe = rows[-1][2]

        # the same plate at the thickness the source MEASURED, when it differs from the H the
        # author fed to his own code. Pcr ~ t^3, so this is the dominant experimental
        # uncertainty and it must be shown, not buried.
        p_fe_meas = p_fe
        if abs(bm.t_meas - bm.ply_t * len(bm.layup)) > 1e-6:
            bm2 = Benchmark(**{**bm.__dict__, 'key': bm.key + '_tmeas',
                               'ply_t': bm.t_meas / len(bm.layup)})
            p_fe_meas = p_critical(bm2, 50, 50)

        p_ana = ANALYTIC_LB[bm.key] * LB
        d_exp = (p_fe - bm.p_exp) / bm.p_exp * 100
        d_exp_m = (p_fe_meas - bm.p_exp) / bm.p_exp * 100
        d_ana = (p_fe - p_ana) / p_ana * 100
        band = '' if p_fe_meas == p_fe else f' ... {p_fe_meas / LB:.0f} lb at the measured t'
        print(f'  measured {bm.p_exp / LB:7.0f} lb | author {p_ana / LB:7.0f} lb | '
              f'our FE {p_fe / LB:7.0f} lb{band}')
        print(f'  -> vs experiment {d_exp:+6.1f}% (at measured t: {d_exp_m:+.1f}%)   '
              f'vs author analysis {d_ana:+6.1f}%\n')
        results.append(dict(key=bm.key, source=bm.source,
                            p_exp_N=bm.p_exp, p_analytic_N=p_ana,
                            p_fe_N=p_fe, p_fe_at_measured_t_N=p_fe_meas,
                            dev_vs_experiment_pct=d_exp,
                            dev_vs_experiment_at_measured_t_pct=d_exp_m,
                            dev_vs_analysis_pct=d_ana,
                            convergence=[dict(nelx=r[0], nely=r[1], p_N=r[2]) for r in rows]))

    d = [abs(r['dev_vs_experiment_pct']) for r in results]
    a = [abs(r['dev_vs_analysis_pct']) for r in results]
    # how well the SOURCE's own analysis does against its own experiment: the yardstick that
    # says whether our agreement is good or merely ordinary
    s = [abs((ANALYTIC_LB[b.key] * LB - b.p_exp) / b.p_exp * 100) for b in BENCHMARKS]
    print(f'SUMMARY  our FE vs experiment:       mean |dev| {sum(d) / len(d):.1f}%, '
          f'worst {max(d):.1f}%')
    print(f'         source analysis vs same experiment: mean |dev| {sum(s) / len(s):.1f}%, '
          f'worst {max(s):.1f}%')
    print(f'         our FE vs source analysis:  mean |dev| {sum(a) / len(a):.1f}%, '
          f'worst {max(a):.1f}%')
    print(f'         canary (FE vs closed form): {dev_canary:+.2f}%')

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, 'exp9_results.json'), 'w') as f:
        json.dump(dict(canary_dev_pct=dev_canary, benchmarks=results), f, indent=2)
    print(f'\nwritten: {OUTDIR}/exp9_results.json')


if __name__ == '__main__':
    import sys
    main_all() if '--all' in sys.argv else main()
