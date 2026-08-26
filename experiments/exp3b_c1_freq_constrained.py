"""exp3b -- regenerate the C1 (pure axial) delivered design with Constraint 4 IMPOSED.

Why this script exists
----------------------
`exp3_minply_sequences.py` computes feasibility as

    bf >= threshold AND sigma_x,sigma_y < 700 AND disp < 1.1 AND Q < 1

-- the FIRST NATURAL FREQUENCY IS ABSENT, although the manuscript states it as
Constraint 4 (f1 > 600 Hz) in tab:optsummary, in prose and in the abstract. The
pre-submission audit of 2026-07-20 recomputed f1 on the six delivered half-stacks and
found the two C1 designs (46 plies, both orientation sets) at 590.8 / 590.2 Hz: both
VIOLATE the constraint the paper declares, by 1.6%. C2 and C3 pass with margin.

Consequence: the published minimum for C1 is not a minimum of the stated problem. On the
C1 sweep f1 crosses 600 Hz between 46 and 48 plies, so this script re-runs the delivered-
design search at N=48 with the frequency gate in the feasibility test, for both
orientation sets, at the SAME budget as exp3 (pop 40, 24 generations, best of seeds 1,2)
so the C1 row stays methodologically comparable with the C2/C3 rows it sits beside.

Difference from exp3, stated precisely: the GA objective is unchanged (maximise the
buckling factor); the FEASIBILITY SCREEN now includes f1 > 600 Hz, and it is applied by
walking the final population from best to worst and taking the first design that clears
every constraint, rather than taking the best-buckling design and checking it afterwards.

Usage:
    CCX_BIN=ccx_2.21 PYTHONPATH=<repo>/code NPROC=4 python3 \\
        experiments/exp3b_c1_freq_constrained.py
"""
from __future__ import annotations

import json
import os
import random
import time
import multiprocessing as mp

from optimisers.constrained_search import (
    buckling_factor, gen_guided, guidelines_ok, manufacturing_ok, static_metrics,
    CASES, STATIC_LOAD, ALPHABETS, SIGMA_MAX, DISP_MAX,
)
from fe.interlaminar import interlaminar
from fe.reference_cases import first_freq, FREQ_MIN

# same budget as exp3 (the delivered-design campaign), so C1 stays comparable with C2/C3
POP, GENS, SEEDS = 40, 24, (1, 2)
N_PLIES = 48
CASE_NAME = "c1_axial"
# how many of the ranked final designs get the (expensive) full constraint screen
SCREEN_TOP = 12


def ranked_population(case, n, alpha, pool, rng, pop=POP, gens=GENS):
    """ga_best's search, verbatim in its guided form, but returning the whole final
    population ranked by buckling factor instead of only its argmax."""
    seeds = [[0], [45, -45], [0, 45, -45], None]
    P, tries = [], 0
    while len(P) < pop and tries < pop * 8:
        pr = seeds[len(P) % len(seeds)]
        s = gen_guided(alpha, n, rng, prefer=pr, pbias=0.6 if pr else 0.0)
        tries += 1
        if s:
            P.append(s)
    if not P:
        return []
    fits = pool.map(buckling_factor, [(p, case) for p in P])
    for _ in range(gens):
        ranked = [p for _, p in sorted(zip(fits, P), key=lambda t: -t[0])]
        elite = ranked[:max(2, pop // 3)]
        kids = []
        while len(kids) < pop - len(elite):
            s = gen_guided(alpha, n, rng, prefer=rng.choice(seeds), pbias=0.6)
            if s:
                kids.append(s)
        P = elite + kids
        fits = pool.map(buckling_factor, [(p, case) for p in P])
    return sorted(zip(fits, P), key=lambda t: -t[0])


def full_screen(seq, bf, case):
    """Every constraint the manuscript declares, frequency included."""
    m = static_metrics((seq, STATIC_LOAD))
    row = dict(buckling=bf, sigma_x=m["sx"], sigma_y=m["sy"], disp_mm=m["disp"])
    row["shell_ok"] = (bf >= case["threshold"] and m["sx"] < SIGMA_MAX
                       and m["sy"] < SIGMA_MAX and m["disp"] < DISP_MAX)
    row["f1_Hz"] = first_freq(seq) if row["shell_ok"] else None
    row["freq_ok"] = bool(row["f1_Hz"] is not None and row["f1_Hz"] > FREQ_MIN)
    # Stesso archetipo corretto in exp3_minply_sequences.py il 26/08/2026, e lo stesso rimedio:
    # un `except Exception: Q = None` nudo rendeva una solve interlaminare MAI GIRATA
    # indistinguibile da una girata che fa fallire il vincolo (entrambe `Q: null,
    # feasible: false`), e il walk sulla popolazione passava al design successivo o usciva con
    # "NO FEASIBLE DESIGN in the top 12" -- un risultato scientifico plausibile prodotto da un
    # guasto. Questo modulo e' il generatore della riga C1 pubblicata (48 ply).
    row["Q"], row["delam_error"] = None, None
    if row["shell_ok"] and row["freq_ok"]:
        try:
            il = interlaminar(seq, axial=STATIC_LOAD["axial"],
                              side=STATIC_LOAD["side"], nx=20, ny=10)
            if "error" in il or il.get("Q") is None:
                raise RuntimeError("interlaminar: %s"
                                   % {k: v for k, v in il.items() if k != "log"})
            row["Q"] = il["Q"]
        except Exception as exc:                      # noqa: BLE001 - registrato, non ingoiato
            row["delam_error"] = f"{type(exc).__name__}: {exc}"
            print(f"  !! delamination solve FAILED: {row['delam_error']}", flush=True)
            if os.environ.get("EXP3_ALLOW_DELAM_FAILURE", "0") != "1":
                raise
    row["feasible"] = bool(row["shell_ok"] and row["freq_ok"]
                           and row["Q"] is not None and row["Q"] < 1.0)
    return row


def main() -> None:
    case = CASES[CASE_NAME]
    nproc = int(os.environ.get("NPROC", "4"))
    pool = mp.get_context("fork").Pool(nproc)
    t0 = time.time()
    out = {"n_plies": N_PLIES, "case": CASE_NAME, "budget": {"pop": POP, "gens": GENS,
           "seeds": list(SEEDS)}, "freq_min_Hz": FREQ_MIN, "designs": {}}

    for aname in ("set1", "set2"):
        alpha = ALPHABETS[aname]
        pooled = []
        for seed in SEEDS:
            pooled += ranked_population(case, N_PLIES, alpha, pool, random.Random(seed))
        pooled.sort(key=lambda t: -t[0])
        print(f"[{aname}] best-buckling over both seeds: "
              f"{', '.join('%.4f' % bf for bf, _ in pooled[:5])}", flush=True)

        chosen, rejected, seen = None, [], set()
        for bf, seq in pooled:
            key = tuple(seq)
            if key in seen:
                continue
            seen.add(key)
            if len(rejected) + (1 if chosen else 0) >= SCREEN_TOP:
                break
            row = full_screen(seq, bf, case)
            row["half_stack"] = seq[: len(seq) // 2]
            row["symmetric"] = seq == seq[::-1]
            row["guidelines_ok"] = guidelines_ok(seq, alpha) and manufacturing_ok(seq)
            print(f"[{aname}] bf={bf:.4f} sx={row['sigma_x']:.1f} sy={row['sigma_y']:.1f} "
                  f"disp={row['disp_mm']:.3f} f1={row['f1_Hz']} Q={row['Q']} "
                  f"feasible={row['feasible']}", flush=True)
            if row["feasible"]:
                chosen = row
                break
            rejected.append(row)

        out["designs"][aname] = {"chosen": chosen, "rejected": rejected}

    out["wall_s"] = round(time.time() - t0, 1)
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                        "exp3b_c1_freq_constrained.json")
    dest = os.path.abspath(dest)
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {dest} in {out['wall_s']}s")
    for aname, d in out["designs"].items():
        c = d["chosen"]
        if c:
            print(f"{aname}: N={N_PLIES} BF={c['buckling']:.4f} sx={c['sigma_x']:.1f} "
                  f"sy={c['sigma_y']:.1f} disp={c['disp_mm']:.3f} f1={c['f1_Hz']:.2f} "
                  f"Q={c['Q']:.6f}")
            print(f"  half: {'/'.join(str(a) for a in c['half_stack'])}")
        else:
            print(f"{aname}: NO FEASIBLE DESIGN in the top {SCREEN_TOP}")
    pool.close()


if __name__ == "__main__":
    main()
