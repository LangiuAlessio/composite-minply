"""exp3_minply_sequences.py -- Delivered minimum-ply stacking sequences (paper "Table 5").

For each in-plane buckling load case (C1 pure axial, C2 pure shear, C3 axial+shear) and
each ply alphabet -- restricted {0,+/-45,90} and extended {0,+/-30,+/-45,+/-60,90} -- this
script finds the THINNEST mid-plane-symmetric, balanced, manufacturing-valid laminate that
satisfies EVERY constraint at the common buckling target (BLF >= 4, load scale 1.0):

    buckling factor      >= 4
    in-plane stress      sigma_x, sigma_y < 700 MPa   (static load S1)
    tip displacement     < 1.1 mm                     (static load S1)
    delamination index   Q = sigma_33 / sigma_33_allow < 1   (sigma_33_allow = 10 MPa)
    manufacturing        symmetric + balanced + 10% rule + disorientation/contiguity

It outputs the six explicit stacking sequences the coauthors asked for, so they can be
validated independently. Each laminate is symmetric, so only the half-stack (outer surface
to mid-plane) is reported; the full laminate is the half followed by its mirror, [...]_s.

Method: budget-converged guided genetic algorithm (population 40, 24 generations, two random
seeds, best taken). Buckling on the validated 2D S8R shell; strength/displacement on the 3D
solid; delamination by equilibrium interlaminar-stress recovery. All FE solves use CalculiX
(ccx). Designed to run on the ailab k3s control node (gw1, 32 cores).

WARNING (2026-07-20 pre-submission audit, finding A1): this script DOES NOT EVALUATE THE
FIRST NATURAL FREQUENCY, although the manuscript declares f1 > 600 Hz as Constraint 4. The
docstring used to claim "Buckling/frequency on the validated 2D S8R shell" -- it was wrong;
the feasibility test below is bf/sigma/disp/Q only. Recomputing f1 on the six delivered
half-stacks showed the two C1 designs at 590.8 (set1) and 590.2 Hz (set2): both INFEASIBLE
against the stated constraint. C2 and C3 pass (617-688 Hz). The corrected C1 design at 48
plies is produced by exp3b_c1_freq_constrained.py, which imposes the frequency gate. Do not
re-run this script to regenerate the C1 row.

NOTE (re-validation finding to confirm): the load magnitudes live in rr_optimiser.CASES
(C1 axial=-2400, C2 side=4900, C3 axial=-1500/side=4900). These differ from the nominal
values printed in the manuscript's load-case table -- reconcile before publication.

Usage (on gw1):
    MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=$HOME/fe-batch-lab CCX_BIN=ccx \\
        $HOME/bin/micromamba run -n fe python experiments/exp3_minply_sequences.py
"""
from __future__ import annotations

import json
import os
import random
import time
import multiprocessing as mp

# FE evaluator + optimiser (synced toolchain). In the cleaned bundle these become
# `from optimisers.constrained_search import ...` / `from fe.interlaminar import interlaminar`.
from optimisers.constrained_search import (
    ga_best, static_metrics, CASES, STATIC_LOAD, ALPHABETS, SIGMA_MAX, DISP_MAX,
)
from fe.interlaminar import interlaminar

# --- search budget (converged; matches the campaign that produced the delivered designs) ---
POP, GENS, SEEDS = 40, 24, (1, 2)

# Ascending ply-count sweeps; the search stops at the first fully-feasible thickness, which is
# the minimum ply count. The restricted alphabet generally needs >= as many plies as the
# extended one, so its sweeps start no lower and run up to the 60-ply manufacturing cap.
SWEEP = {
    "set2": {  # extended {0,+/-30,+/-45,+/-60,90}
        "c1_axial": [40, 42, 44, 46, 48, 50],
        "c2_side":  [46, 48, 50, 52, 54],
        "c3_combo": [48, 50, 52, 54, 56, 58],
    },
    "set1": {  # restricted {0,+/-45,90}
        "c1_axial": [40, 42, 44, 46, 48, 50, 52],
        "c2_side":  [46, 48, 50, 52, 54, 56, 58],
        "c3_combo": [48, 50, 52, 54, 56, 58, 60],
    },
}

CASE_LABEL = {"c1_axial": "C1 pure axial", "c2_side": "C2 pure shear", "c3_combo": "C3 axial+shear"}


def evaluate(case_name: str, n: int, alpha: list[int], pool) -> dict:
    """Run the guided GA at ply count `n`, take the best of `SEEDS`, and check all constraints.

    Returns the design's buckling factor, the static-load stresses/displacement, the
    delamination index Q, the symmetric half-stack, and a feasibility flag.
    """
    case = CASES[case_name]
    best_bf, best_seq = -1.0, None
    for seed in SEEDS:
        bf, seq = ga_best(case, n, alpha, pool, random.Random(seed), pop=POP, gens=GENS, guided=True)
        if seq is not None and bf > best_bf:
            best_bf, best_seq = bf, seq

    metrics = static_metrics((best_seq, STATIC_LOAD))
    shell_ok = (
        best_bf >= case["threshold"]
        and metrics["sx"] < SIGMA_MAX
        and metrics["sy"] < SIGMA_MAX
        and metrics["disp"] < DISP_MAX
    )
    # The delamination check is the last gate, and it used to fail SILENTLY: a bare
    # `except Exception: Q = None` turned an interlaminar solve that never ran into a row
    # indistinguishable from one that ran and passed the shell checks but failed on Q. Both
    # came out `feasible: false`, the ascending sweep never broke, and it ran to the
    # manufacturing cap. That is what produced data/exp3_minply_sequences.json, whose sweep
    # rows all carry `Q: null, feasible: false` while `delivered` reports the same ply counts
    # and buckling factors as feasible with a real Q: the two blocks come from passes in
    # which the solid solver was, respectively, unavailable and available. The error is now
    # recorded in the row and re-raised unless the caller opts out.
    Q, delam_error = None, None
    if shell_ok:
        try:
            Q = interlaminar(best_seq, axial=STATIC_LOAD["axial"], side=STATIC_LOAD["side"],
                             nx=20, ny=10).get("Q")
        except Exception as exc:                      # noqa: BLE001 - recorded, not swallowed
            delam_error = f"{type(exc).__name__}: {exc}"
            print(f"  !! delamination solve FAILED at N={n} ({case_name}): {delam_error}",
                  flush=True)
            if os.environ.get("EXP3_ALLOW_DELAM_FAILURE", "0") != "1":
                raise
    feasible = bool(shell_ok and Q is not None and Q < 1.0)
    half = best_seq[: len(best_seq) // 2] if best_seq else None
    return {
        "n_plies": n,
        "buckling": round(best_bf, 3),
        "sigma_x": round(metrics["sx"], 1),
        "sigma_y": round(metrics["sy"], 1),
        "disp_mm": round(metrics["disp"], 3),
        "Q": (round(Q, 6) if Q is not None else None),
        "symmetric": bool(best_seq == best_seq[::-1]) if best_seq else False,
        "feasible": feasible,
        "delam_error": delam_error,
        "half_stack": half,
        "full_stack": best_seq,
    }


def main() -> None:
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "data", "exp3_minply_sequences.json")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    nproc = int(os.environ.get("NPROC", str(os.cpu_count())))
    pool = mp.get_context("fork").Pool(nproc)
    t_start = time.time()
    results = {
        "config": {"pop": POP, "gens": GENS, "seeds": list(SEEDS), "nproc": nproc,
                   "target_BLF": 4, "load_cases": CASES, "static_load": STATIC_LOAD},
        "delivered": {},  # delivered[alphabet][case] = first feasible (min-ply) design
        "sweeps": {},     # full ascending sweep for transparency
    }
    try:
        for alpha_name in ("set2", "set1"):
            alpha = ALPHABETS[alpha_name]
            results["delivered"][alpha_name] = {}
            results["sweeps"][alpha_name] = {}
            for case_name in ("c1_axial", "c2_side", "c3_combo"):
                results["sweeps"][alpha_name][case_name] = []
                delivered = None
                for n in SWEEP[alpha_name][case_name]:
                    t0 = time.time()
                    r = evaluate(case_name, n, alpha, pool)
                    r["secs"] = round(time.time() - t0)
                    results["sweeps"][alpha_name][case_name].append(r)
                    json.dump(results, open(out_path, "w"), indent=1)  # checkpoint each step
                    print(f"[{alpha_name}] {CASE_LABEL[case_name]} N={n}: BF={r['buckling']:.2f} "
                          f"sx={r['sigma_x']:.0f} sy={r['sigma_y']:.0f} disp={r['disp_mm']:.2f} "
                          f"Q={r['Q']} feasible={r['feasible']} [{r['secs']}s]", flush=True)
                    if r["feasible"]:
                        delivered = r
                        break
                results["delivered"][alpha_name][case_name] = delivered
    finally:
        pool.close()
        pool.join()

    results["total_secs"] = round(time.time() - t_start)
    json.dump(results, open(out_path, "w"), indent=1)

    print("\n=== DELIVERED MINIMUM-PLY STACKING SEQUENCES ===", flush=True)
    for alpha_name in ("set1", "set2"):
        for case_name in ("c1_axial", "c2_side", "c3_combo"):
            d = results["delivered"][alpha_name].get(case_name)
            if d:
                half = "/".join(map(str, d["half_stack"]))
                print(f"{CASE_LABEL[case_name]:16} [{alpha_name}] {d['n_plies']} plies, BF {d['buckling']:.2f}: "
                      f"[{half}]_s", flush=True)
            else:
                print(f"{CASE_LABEL[case_name]:16} [{alpha_name}]: NO feasible design within the swept range",
                      flush=True)
    print(f"\nJSON -> {out_path}   total {results['total_secs']}s", flush=True)


if __name__ == "__main__":
    main()
