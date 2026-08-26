"""Literature benchmark: reproduce the Haftka & Walsh (NASA TM-104033, 1991) integer-programming
GLOBAL optima for buckling of a simply-supported graphite-epoxy plate under biaxial compression,
with our optimiser methodology — validating the search against a provably-optimal reference.

Closed-form CLT buckling (no FE per candidate). Design: 16-ply symmetric + balanced laminate,
angles {0,±45,90}; maximise the critical load multiplier lambda_cr at fixed thickness, swept over
the transverse/axial load ratio Ny/Nx. Reference values: NASA TM-104033 Fig. 2.

Steps:
  1. validate the CLT lambda_cr evaluator against the published optimal lambda_cr values;
  2. EXHAUSTIVE global optimum (4^8 balanced-symmetric half-laminates) -> ground truth == paper;
  3. our GA (manufacturing-aware, symmetry+balance, load-aware seeds, repair crossover) -> recovers it.
"""
from __future__ import annotations
import math, itertools, random

# graphite-epoxy (TM-104033 Results): E1,E2,G12 in psi, nu12; ply thickness t (in)
E1, E2, G12, NU12, T = 18.5e6, 1.89e6, 0.93e6, 0.30, 0.005
A_IN, B_IN = 20.0, 10.0                 # plate a x b (in), aspect ratio 2
ANGLES = (0, 45, -45, 90)

# Tsai-Pagano reduced stiffnesses + material invariants U1..U5 (constant for the material)
_nu21 = NU12 * E2 / E1
_den = 1.0 - NU12 * _nu21
Q11, Q22, Q12, Q66 = E1 / _den, E2 / _den, NU12 * E2 / _den, G12
U1 = (3 * Q11 + 3 * Q22 + 2 * Q12 + 4 * Q66) / 8
U2 = (Q11 - Q22) / 2
U3 = (Q11 + Q22 - 2 * Q12 - 4 * Q66) / 8
U4 = (Q11 + Q22 + 6 * Q12 - 4 * Q66) / 8
U5 = (Q11 + Q22 - 2 * Q12 + 4 * Q66) / 8


def flexural_D(seq):
    """Flexural stiffnesses D11,D12,D22,D66 for the full laminate `seq` (list of angles, deg),
    plies of thickness T stacked from one face to the other. Eqs (2)-(5),(10) of TM-104033."""
    n = len(seq)
    h = n * T
    V0 = V1 = V3 = 0.0
    for k, ang in enumerate(seq):
        z0 = -h / 2 + k * T
        z1 = -h / 2 + (k + 1) * T
        dz3 = (z1 ** 3 - z0 ** 3) / 3.0
        r = math.radians(ang)
        V0 += dz3
        V1 += math.cos(2 * r) * dz3
        V3 += math.cos(4 * r) * dz3
    D11 = U1 * V0 + U2 * V1 + U3 * V3
    D22 = U1 * V0 - U2 * V1 + U3 * V3
    D12 = U4 * V0 - U3 * V3
    D66 = U5 * V0 - U3 * V3
    return D11, D12, D22, D66


def lambda_cr(seq, ny, nx=1.0, a=A_IN, b=B_IN, mmax=4, nmax=4):
    """Critical load multiplier for SS biaxial buckling under (Nx=nx, Ny=ny), minimised over the
    half-wave numbers m,n. Eq (1) of TM-104033. (Max-buckling sweep uses nx=1, ny=ratio.)"""
    D11, D12, D22, D66 = flexural_D(seq)
    Nx, Ny = nx, ny
    best = math.inf
    for m in range(1, mmax + 1):
        for n in range(1, nmax + 1):
            ka, kb = (m / a) ** 2, (n / b) ** 2
            num = D11 * ka ** 2 + 2 * (D12 + 2 * D66) * ka * kb + D22 * kb ** 2
            den = ka * Nx + kb * Ny
            lam = math.pi ** 2 * num / den
            best = min(best, lam)
    return best


def _symm(half):
    """Full symmetric laminate from a half (i=1 nearest midplane ... outside): reverse(half)+half."""
    return list(reversed(half)) + list(half)


def _balanced(half):
    return half.count(45) == half.count(-45)


def _rebalance(h):
    """Make h balanced (#45 == #-45) IN PLACE. If the total ±45 count is odd (balance impossible
    by flipping alone), convert one ±45 ply to 90 first, then equalise the two by flipping the
    surplus. Returns h."""
    pm = [i for i, a in enumerate(h) if a in (45, -45)]
    if len(pm) % 2:
        h[pm[0]] = 90
    p = [i for i, a in enumerate(h) if a == 45]
    m = [i for i, a in enumerate(h) if a == -45]
    while len(p) > len(m):
        i = p.pop(); h[i] = -45; m.append(i)
    while len(m) > len(p):
        i = m.pop(); h[i] = 45; p.append(i)
    return h


def exhaustive_optimum(ny_over_nx, nhalf=8):
    """Global optimum over ALL balanced symmetric laminates (half of nhalf plies in ANGLES)."""
    best_lam, best_seq = -math.inf, None
    for half in itertools.product(ANGLES, repeat=nhalf):
        if not _balanced(half):
            continue
        lam = lambda_cr(_symm(half), ny_over_nx)
        if lam > best_lam:
            best_lam, best_seq = lam, half
    return best_lam, best_seq


# ---- our optimiser methodology (manufacturing-aware GA over the half-laminate) ----
def _gen_half(nhalf, rng, prefer=None, pbias=0.0):
    """Random balanced half-laminate; biased toward `prefer` angles (load-aware seeding)."""
    h = [(rng.choice(prefer) if prefer and rng.random() < pbias else rng.choice(ANGLES))
         for _ in range(nhalf)]
    return _rebalance(h)


def _ga_run(ny_over_nx, nhalf, pop, gens, seed):
    rng = random.Random(seed)
    seeds = [[45, -45], [90], [0], [0, 90], None]
    P = [_gen_half(nhalf, rng, prefer=seeds[i % len(seeds)], pbias=0.7) for i in range(pop)]
    fit = lambda h: lambda_cr(_symm(h), ny_over_nx)
    best = max(P, key=fit)
    for _ in range(gens):
        P.sort(key=fit, reverse=True)
        elite = P[:max(2, pop // 4)]
        if fit(elite[0]) > fit(best):
            best = elite[0]
        kids = []
        while len(kids) < pop - len(elite):
            a, b = rng.choice(elite), rng.choice(elite)
            cut = rng.randint(1, nhalf - 1)
            child = a[:cut] + b[cut:]
            for _ in range(2):                       # up to 2 mutations (richer exploration)
                if rng.random() < 0.4:
                    child[rng.randrange(nhalf)] = rng.choice(ANGLES)
            kids.append(_rebalance(child))
        P = elite + kids
    return fit(best), best


def ga_optimum(ny_over_nx, nhalf=8, pop=60, gens=80, seed=0, restarts=6):
    """Our GA with random restarts (CLT eval is ~microseconds, so we can afford a thorough search).
    Load-aware seeds + repair-based crossover/mutation, balance preserved. Returns best over restarts."""
    best_lam, best_seq = -math.inf, None
    for r in range(restarts):
        lam, seq = _ga_run(ny_over_nx, nhalf, pop, gens, seed + r)
        if lam > best_lam:
            best_lam, best_seq = lam, seq
    return best_lam, best_seq


# published global optima (TM-104033 Fig. 2): Ny/Nx -> lambda_cr
PUBLISHED = {0.125: 154.06, 0.15: 148.46, 0.2: 137.10, 0.24: 129.28, 0.25: 127.49,
             0.5: 94.29, 1.0: 61.99, 1.5: 46.18, 2.0: 36.84, 2.1: 35.40, 2.4: 31.64, 2.45: 31.06}

# minimum-thickness dual (TM-104033 Fig. 5): Nx=30 lb/in, a=20,b=10; Ny (lb/in) -> min #plies
MIN_THICKNESS = {0.0: 10, 7.5: 10, 15.0: 12, 22.5: 12, 30.0: 14, 45.0: 14, 60.0: 16, 75.0: 16}


def min_thickness(ny_abs, nx=30.0, nmax_half=10):
    """Our exact RR-style objective: smallest #plies (even, symmetric, balanced) such that a
    feasible laminate exists (max-over-layup lambda_cr >= 1) under the load (Nx=nx, Ny=ny_abs).
    Exhaustive max-buckling per ply count (design space is small here). Returns (#plies, best_lam)."""
    for nhalf in range(2, nmax_half + 1):
        best = -math.inf
        for half in itertools.product(ANGLES, repeat=nhalf):
            if not _balanced(half):
                continue
            best = max(best, lambda_cr(_symm(half), ny_abs, nx=nx))
        if best >= 1.0:
            return 2 * nhalf, best
    return None, None


if __name__ == "__main__":
    import json, os
    # Step 1+2+3: for each published Ny/Nx, compare exhaustive global opt and our GA to the paper.
    print(f"{'Ny/Nx':>6} {'paper':>8} {'exhaustive':>11} {'GA(ours)':>9} {'err%':>6}  match")
    worst = 0.0
    lam_rows = []
    for r, lam_paper in PUBLISHED.items():
        lam_ex, seq_ex = exhaustive_optimum(r)
        lam_ga, _ = ga_optimum(r, seed=1)
        err = abs(lam_ex - lam_paper) / lam_paper * 100
        worst = max(worst, err)
        ok = "OK" if abs(lam_ga - lam_ex) / lam_ex < 1e-6 else f"GA<{lam_ga:.2f}"
        print(f"{r:>6} {lam_paper:>8.2f} {lam_ex:>11.2f} {lam_ga:>9.2f} {err:>6.2f}  {ok}")
        lam_rows.append(dict(ratio=r, lam_paper=lam_paper, lam_exhaustive=lam_ex, lam_ga=lam_ga,
                             err_pct=err, ga_recovers_optimum=abs(lam_ga - lam_ex) / lam_ex < 1e-6,
                             seq_exhaustive=list(seq_ex)))
    print(f"\nworst |exhaustive - paper| = {worst:.2f}% "
          f"({'CLT VALIDATED' if worst < 1.0 else 'CHECK CLT'})")

    # min-thickness dual (Fig. 5) — our exact "min plies s.t. buckling" objective
    print(f"\nmin-thickness dual (Nx=30 lb/in):  {'Ny':>6} {'paper#ply':>9} {'ours#ply':>8} {'lam>=1':>7}")
    okt = True
    thk_rows = []
    for ny, nply_paper in MIN_THICKNESS.items():
        nply, lam = min_thickness(ny)
        flag = "OK" if nply == nply_paper else "DIFF"
        okt = okt and (nply == nply_paper)
        print(f"{'':>34} {ny:>6} {nply_paper:>9} {nply:>8} {lam:>7.2f}  {flag}")
        thk_rows.append(dict(ny=ny, nx=30.0, nply_paper=nply_paper, nply_ours=nply, lam=lam,
                             match=nply == nply_paper))
    print(f"min-thickness matches paper on all points: {okt}")

    # the numbers the paper's benchmark figure is drawn from: written, not retyped
    out = os.path.join(os.path.dirname(__file__), '..', 'data', 'exp6_haftka_walsh.json')
    with open(os.path.abspath(out), 'w') as fh:
        json.dump(dict(source='Haftka & Walsh, NASA TM-104033 (1991), Fig. 2 and Fig. 5',
                       plate=dict(a_in=A_IN, b_in=B_IN, plies=16, angles=list(ANGLES)),
                       worst_clt_err_pct=worst, min_thickness_all_match=okt,
                       max_buckling=lam_rows, min_thickness=thk_rows), fh, indent=1)
    print('wrote', os.path.relpath(os.path.abspath(out)))
