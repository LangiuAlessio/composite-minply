"""Giacomo Canale's EXACT reviewed problem ("v1" scheme) — minimum-ply stacking search.

This is NON-DESTRUCTIVE: it imports the validated FE machinery from rr_optimiser /
rr_interlaminar and applies Giacomo's exact problem definition (which DIFFERS from the
"v2" scheme currently hard-coded in rr_optimiser.CASES). In particular:

  * Three DISTINCT buckling targets (not a common BLF>=4):
        C1 pure axial          axial=-1000, side=0,    torsion=0       -> BLF >= 12
        C2 axial+shear         axial=-1000, side=5000, torsion=0       -> BLF >=  8
        C3 axial+shear+torsion axial=-1000, side=5000, torsion=300000  -> BLF >=  5
  * Torsion IS retained (the v2 code dropped it as "inert"; the validation anchor below
    PROVES it is NOT inert in this model: C3 != C2, matching Giacomo's 3.70 -> 2.47 drop).
  * Static case S1 for strength: axial=20000, side=5000, torsion=0.
  * Constraints: sigma_x<700, sigma_y<700, peel sigma_zz<10 (averaged), tip disp<1.1,
    first natural frequency>600 Hz, plus manufacturing rules (<=45 deg disorientation,
    <=3 consecutive same [code's stricter rule; Giacomo says <=4], symmetry, balance,
    10% rule). Max 60 plies.
  * Model split: buckling+frequency on the 2D shell; strength + delamination Q on the 3D
    solid (winner-only).
  * Two ply alphabets: set1 = {0,+-45,90}; set2 = {0,+-30,+-45,+-60,90}.

Buckling/frequency: shell (rr_optimiser.make_ccx_deck). Strength/delamination: solid
(rr_interlaminar.interlaminar). Every number is produced by an actual ccx run.
"""
from __future__ import annotations
import os, re, json, subprocess, tempfile, shutil, random, time
from multiprocessing import Pool

import optimisers.constrained_search as R
from fe.interlaminar import interlaminar

CCX = R.CCX

# --- Giacomo's exact buckling load cases (loads at the MPC control point) -----------
CASES_V1 = {
    "C1": dict(axial=-1000., side=0.,    torsion=0.,      threshold=12.),
    "C2": dict(axial=-1000., side=5000., torsion=0.,      threshold=8.),
    "C3": dict(axial=-1000., side=5000., torsion=300000., threshold=5.),
}
# Giacomo's static case S1 (strength + delamination)
S1 = dict(axial=20000., side=5000., torsion=0., threshold=0.)

# constraints
DISP_MAX, SIGMA_MAX, PEEL_MAX, FREQ_MIN = 1.1, 700.0, 10.0, 600.0
MAX_PLIES = 60
PLY_COUNTS = [n for n in range(8, MAX_PLIES + 1, 2)]   # even (symmetry/balance), ascending

ALPHABETS = R.ALPHABETS   # {"set1":..., "set2":...}


# --- frequency on the SAME cantilever shell mesh (clamped short edge) ----------------
def make_freq_deck(seq, nfreq=6):
    """Frequency step on the rr_optimiser cantilever shell mesh (clamp x=0, free else)."""
    case = dict(axial=0., side=0., torsion=0., threshold=0.)
    d = R.make_ccx_deck(seq, case)
    d = d.replace("*STEP\n*BUCKLE\n2\n", f"*STEP\n*FREQUENCY\n{nfreq}\n")
    out, skip = [], False
    for ln in d.splitlines():
        if ln.strip() == "*CLOAD":
            skip = True
            continue
        if skip and ln.strip() == "*END STEP":
            skip = False
        if not skip:
            out.append(ln)
    return "\n".join(out) + "\n"


def _parse_freqs(dat):
    freqs, inblk = [], False
    for line in dat.splitlines():
        u = line.upper()
        if "MODE NO" in u and "EIGENVALUE" in u:
            inblk = True
            continue
        if inblk:
            p = line.split()
            if len(p) >= 5:
                try:
                    int(p[0]); vals = [float(x) for x in p[1:]]
                except ValueError:
                    if freqs:
                        break
                    continue
                freqs.append(vals[2])   # cycles/time = Hz
            elif freqs and line.strip() == "":
                continue
    return freqs


class FreqSolveFailure(RuntimeError):
    """La solve modale non ha prodotto una frequenza. NON e' una frequenza bassa: un guasto."""


# Valvola per gli esperimenti che provano deliberatamente i guasti. Non usarla in campagna.
_ALLOW_FREQ_SENTINEL = os.environ.get("CCX_ALLOW_FAILURE_SENTINEL", "0") == "1"


def first_freq(seq):
    """Prima frequenza propria, in Hz.

    ⚠️ Fino al 26/08/2026 questa funzione tornava -1.0 su QUALUNQUE guasto, e -1.0 e' un
    numero in Hz: i consumatori lo confrontavano con FREQ_MIN e concludevano "vincolo di
    frequenza violato". Cioe' "mai girato" e "girato e non passa" erano lo stesso valore.
    `reference_cases.search` proseguiva lo sweep ascendente, exp17 avrebbe registrato
    `f1_Hz: -1.0` come una frequenza misurata nel JSON che alimenta la caption di tab:feasible,
    e exp3b/exp2 lo stesso. Ora un guasto alza; la sentinella resta solo dietro opt-in.
    """
    d = tempfile.mkdtemp()
    try:
        open(d + "/job.inp", "w").write(make_freq_deck(seq))
        r = subprocess.run([CCX, "-i", "job"], cwd=d, capture_output=True, timeout=120,
                           env={**os.environ, "OMP_NUM_THREADS": "1"})
        if r.returncode != 0:
            raise FreqSolveFailure("ccx exit code %d" % r.returncode)
        if not os.path.exists(d + "/job.dat"):
            raise FreqSolveFailure("ccx non ha scritto job.dat")
        fr = _parse_freqs(open(d + "/job.dat").read())
        if not fr:
            raise FreqSolveFailure("nessuna frequenza nel .dat")
        return fr[0]
    except Exception as e:
        if _ALLOW_FREQ_SENTINEL:
            return -1.0
        if isinstance(e, FreqSolveFailure):
            raise
        raise FreqSolveFailure("solve modale fallita (%s)" % e) from e
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --- the per-(case, alphabet) minimum-ply search ------------------------------------
def search(case_name, case, alpha_name, pool, rng, ply_counts=PLY_COUNTS):
    """Ascending ply-count GA maximising buckling for `case` (torsion threaded into the
    deck via make_ccx_deck). First N meeting the buckling target AND the static shell
    constraints (sigma_x,sigma_y<700, disp<1.1, freq>600) wins; that winner is then
    verified on the 3D solid for stress + delamination Q (Giacomo's model split)."""
    alpha = ALPHABETS[alpha_name]
    guided = True   # symmetry + balance + 10% rule (Giacomo requires these)
    for n in ply_counts:
        bf, seq = R.ga_best(case, n, alpha, pool, rng, guided=guided)
        if seq is None:
            continue
        if bf < case["threshold"]:
            continue
        # shell strength/disp under Giacomo's S1
        m = R.static_metrics((seq, S1))
        if not (m["disp"] <= DISP_MAX and m["sx"] <= SIGMA_MAX and m["sy"] <= SIGMA_MAX):
            continue
        # frequency (shell, cantilever)
        f1 = first_freq(seq)
        if f1 < FREQ_MIN:
            continue
        # winner: verify on the 3D solid (strength + delamination)
        il = interlaminar(seq, axial=S1["axial"], side=S1["side"], nx=20, ny=10)
        Q = il.get("Q")
        peel = il.get("peel")
        feasible_solid = (Q is not None and Q < 1.0
                          and (peel is None or peel <= PEEL_MAX))
        return dict(case=case_name, alphabet=alpha_name, n_plies=n, seq=list(seq),
                    bf=round(bf, 4), sx=round(m["sx"], 2), sy=round(m["sy"], 2),
                    disp=round(m["disp"], 4), freq=round(f1, 2),
                    Q=(round(Q, 5) if Q is not None else None),
                    solid_peel=(round(peel, 4) if peel is not None else None),
                    solid_disp=(round(il.get("disp"), 4) if il.get("disp") is not None else None),
                    threshold=case["threshold"], feasible=bool(feasible_solid),
                    solid_log=(il.get("log") if "error" in il else None))
    return dict(case=case_name, alphabet=alpha_name, n_plies=None, seq=None,
                threshold=case["threshold"], feasible=False,
                note=f"INFEASIBLE up to {ply_counts[-1]} plies")


def validation_anchor(pool):
    """60-ply cross-ply [0/90]_15s on the shell for C1/C2/C3 with Giacomo's exact loads.
    Giacomo: 15.22 / 3.70 / 2.47. The verdict on torsion hinges on C3 != C2.

    2026-07-22 -- the sequence was [0,90]*30, which is ANTIsymmetric, while
    exp2_crossply_baseline.py and the manuscript use the SYMMETRIC [0/90]_15s. Two different
    laminates were being compared against the same Abaqus reference. Measured side by side:
    [0/90]_15s gives f1 = 627.97 Hz (0.2% from Abaqus' 627), [0,90]*30 gives 614.68 Hz (1.9%).
    That also closes the open question in VALIDATION.md: the 614.7 recorded there as "does NOT
    reproduce" was never a solver-version drift, it was this other cross-ply. Aligned to the
    symmetric stack so the bundle has ONE definition of the anchor.
    """
    half = [0, 90] * 15
    seq = half + half[::-1]                     # [0/90]_15s, 60 plies, mid-plane symmetric
    out = {"seq_desc": "[0/90]_15s (60-ply symmetric cross-ply)",
           "giacomo": {"C1": 15.22, "C2": 3.70, "C3": 2.47}}
    bfs = {}
    for nm in ("C1", "C2", "C3"):
        bfs[nm] = round(R.buckling_factor((seq, CASES_V1[nm])), 4)
    out["computed"] = bfs
    out["freq1_hz"] = round(first_freq(seq), 2)
    out["torsion_works"] = bfs["C3"] < bfs["C2"] - 0.3   # C3 must drop below C2
    return out


def main():
    t0 = time.time()
    nproc = int(os.environ.get("NPROC", str(os.cpu_count())))
    import multiprocessing as _mp
    try:
        ctx = _mp.get_context("fork")
    except ValueError:
        ctx = _mp
    with ctx.Pool(nproc) as pool:
        print("=== VALIDATION ANCHOR (60-ply cross-ply, shell buckling) ===")
        anchor = validation_anchor(pool)
        for nm in ("C1", "C2", "C3"):
            print(f"  {nm}: computed BF={anchor['computed'][nm]:7.4f}   "
                  f"Giacomo={anchor['giacomo'][nm]:.2f}")
        print(f"  first natural freq = {anchor['freq1_hz']} Hz")
        print(f"  torsion works (C3 < C2)? {anchor['torsion_works']}  "
              f"(C2={anchor['computed']['C2']}, C3={anchor['computed']['C3']})")

        results = []
        for alpha_name in ("set1", "set2"):
            for case_name in ("C1", "C2", "C3"):
                rng = random.Random(1)
                tt = time.time()
                r = search(case_name, CASES_V1[case_name], alpha_name, pool, rng)
                r["seconds"] = round(time.time() - tt, 1)
                results.append(r)
                fe = "FEASIBLE" if r["feasible"] else "INFEASIBLE"
                np_ = r["n_plies"]
                print(f"[{case_name} x {alpha_name}] {fe}  n_plies={np_}  ({r['seconds']}s)")

    payload = {"anchor": anchor, "results": results,
               "constraints": dict(disp_max=DISP_MAX, sigma_max=SIGMA_MAX,
                                   peel_max=PEEL_MAX, freq_min=FREQ_MIN, max_plies=MAX_PLIES),
               "static_S1": S1, "cases": CASES_V1, "total_seconds": round(time.time() - t0, 1)}
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, "_giacomo_v1_sequences.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    # summary table
    print("\n=== GIACOMO v1 — 6 minimum-ply sequences ===")
    hdr = f"{'case':4} {'alpha':5} {'n':>3} {'BF':>7} {'tgt':>4} {'sx':>7} {'sy':>7} {'disp':>6} {'Q':>8} {'freq':>7} feas"
    print(hdr)
    for r in results:
        if r["n_plies"] is None:
            print(f"{r['case']:4} {r['alphabet']:5} {'--':>3}  INFEASIBLE up to {MAX_PLIES} plies")
            continue
        print(f"{r['case']:4} {r['alphabet']:5} {r['n_plies']:>3} {r['bf']:>7.3f} "
              f"{r['threshold']:>4.0f} {r['sx']:>7.1f} {r['sy']:>7.1f} {r['disp']:>6.3f} "
              f"{(r['Q'] if r['Q'] is not None else float('nan')):>8.4f} {r['freq']:>7.1f} "
              f"{'Y' if r['feasible'] else 'N'}")
    print("\n--- explicit winning sequences ---")
    for r in results:
        if r["seq"]:
            print(f"{r['case']} x {r['alphabet']} ({r['n_plies']} ply): {r['seq']}")
    print(f"\nJSON -> {out_path}   total {payload['total_seconds']}s")


if __name__ == "__main__":
    main()
