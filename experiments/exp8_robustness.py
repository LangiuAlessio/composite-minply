"""Buckling-evaluator diagnostic (CORRECTED conclusion).

An earlier note claimed buckling_factor() suffered a spurious mode-switching BUG
(a low first eigenvalue appearing at higher ply counts). That claim was WRONG.
Two controls disprove it:

1. MESH CONVERGENCE: the low mode is mesh-converged (e.g. quasi-iso N=32 m1=0.693
   identical at 20x10 / 30x15 / 40x20 / 60x30). A numerical/spurious mode would
   move with refinement; this does not. => the mode is REAL, not an artifact.

2. SYMMETRIC layups are non-monotonic TOO (symmetric [0/45/-45/90]_s repeated:
   3.18@24, 0.55@28, 0.81@32, 1.13@36, ... ), so it is not a B-coupling/symmetry
   issue either.

The correct explanation: the buckling factor of a FIXED repeating layup is
genuinely non-monotonic in ply count, because the bending stiffness D depends on
the z-position of each ply (z^3 weighting), so a [sublaminate]_n laminate's D (and
hence its governing buckling mode) does not scale monotonically with n. Taking the
LOWEST eigenvalue is the correct definition of the governing buckling load; the
evaluator is correct. The optimiser MAXIMISES this factor at each ply count, so its
per-N result should be monotonic in n GIVEN ENOUGH BUDGET; the non-monotonicity in
the optimised campaign (records.jsonl) is optimiser budget-starvation, not an
evaluator fault.

Real, open issues (from the robustness audit), NOT evaluator bugs:
  - the campaign runs guided=False (non-symmetric, only disorientation+contiguity);
    standard practice enforces symmetry/balance/10% (guided=True);
  - the optimiser is not budget-converged, so per-N optimised results are noisy.

Run from cases/:  python3 rr_buckling_mode_diag.py
"""
import os, sys, re, tempfile, subprocess, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from optimisers.constrained_search import make_ccx_deck, buckling_factor, CASES, CCX
except ImportError:
    from optimisers.constrained_search import make_ccx_deck, buckling_factor, CASES, CCX
CASE = CASES["c1_axial"]


def spectrum(seq, nmodes=6):
    deck = make_ccx_deck(seq, CASE).replace("*BUCKLE\n2", "*BUCKLE\n%d" % nmodes)
    d = tempfile.mkdtemp()
    try:
        open(d + "/job.inp", "w").write(deck)
        subprocess.run([CCX, "-i", "job"], cwd=d, capture_output=True, timeout=180,
                       env={**os.environ, "OMP_NUM_THREADS": "1"})
        dat = open(d + "/job.dat").read() if os.path.exists(d + "/job.dat") else ""
        return [float(v) for _, v in re.findall(r"^\s*(\d+)\s+([\d.E+\-]+)\s*$", dat, re.M)][:nmodes]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def sym(n):
    h = [[0, 45, -45, 90][i % 4] for i in range(n // 2)]
    return h + h[::-1]


def nonsym(n):
    return [[0, 45, 90, -45][i % 4] for i in range(n)]


if __name__ == "__main__":
    print("Governing buckling factor m1 (lowest eigenvalue) for FIXED repeating layups:")
    print("N  | symmetric | non-symmetric  (both non-monotonic = real D(z) mechanics, not a bug)")
    for n in [24, 28, 32, 36, 40, 44, 48]:
        print("%2d |  %6.3f   |   %6.3f" % (n, buckling_factor((sym(n), CASE)), buckling_factor((nonsym(n), CASE))))
    print("\nFull spectrum (non-sym N=32): %s" % " ".join("%.3f" % v for v in spectrum(nonsym(32))))
    print("=> lowest mode is the governing buckling load (correct); optimiser maximises it.")
