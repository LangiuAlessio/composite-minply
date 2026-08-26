"""exp23_basin_transport.py -- il falsificatore della Direzione 2: trasportare il bacino fra spessori.

Lo sweep del lavoro precedente e' NON MONOTONO sul caso assiale: 4,61 a 40 lamine, **3,79 a 44**,
4,93 a 48. Il manoscritto dice che non e' rumore e ne da' il meccanismo: a N=40 il migliore e'
+-45-dominato (12 lamine su 20 di half-stack), a N=44 tutti i semi cadono nel bacino 0-dominato
(11-13 su 22), che a quello spessore e' circa il 20% peggiore. «Adding material does not help a
search that has settled in the wrong basin.»

IPOTESI: il bacino +-45 e' buono anche a N=44 ma **irraggiungibile per restart indipendente**. Se
l'elite si trasporta attraverso lo spessore invece di ripartire da zero, il best risale.

⚠️ Criterio in `IDEA_ALGORITMO_DFA_2026-08-06.md`, «PRE-REGISTRAZIONE #3», scritto e committato
PRIMA di questo file. Primario: TRANSPORT batte COLD sul best, Mann-Whitney a una coda, alpha 0,05.
Falsificata se non lo batte, e allora la Direzione 2 muore qui.

IL PUNTO DELICATO E' L'INSERZIONE, e non e' un dettaglio implementativo. La soglia del 10% e'
`ten_pct_min_half(N)`: **2 a N=40, 3 a N=44**. Un half-stack conforme a 40 puo' quindi essere NON
conforme a 44, e le due lamine da inserire non sono libere: devono coprire il deficit sulle
direzioni sotto soglia, e in coppia (+theta con -theta, oppure due fra 0 e 90) perche' il
bilanciamento resti chiuso. Se nessuna inserzione riesce, si ricade sul campionatore esatto e **lo
si conta**: un trasporto che fallisce in silenzio farebbe sembrare TRANSPORT uguale a COLD per la
ragione sbagliata.

Uso:  PYTHONPATH=$PWD CCX_BIN=/usr/bin/ccx NPROC=30 python3 -m experiments.exp23_basin_transport
"""
from __future__ import annotations

import json
import os
import random
import statistics as st
import sys
import multiprocessing as mp

from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimisers.constrained_search import CASES, ALPHABETS, manufacturing_ok, guidelines_ok
from optimisers.laminate_language import uniform_sampler, ten_pct_min_half, PRINCIPAL

from experiments.exp20_dfa_crossover import _bf_symmetric, POP                    # noqa: E402
from experiments.exp4_optimiser_comparison import sig                             # noqa: E402

ALPHA = ALPHABETS["set2"]          # lo sweep del lavoro precedente gira sull'esteso
CASE = "c1_axial"                  # il caso bimodale, quello con la caduta
N_FROM, N_TO = 40, 44
BUDGET_DISTINCT = 176
MAX_GENS = 40
ELITE = max(2, POP // 3)
SEEDS = list(range(1, 31))
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "_out", "exp23_basin_transport.json")


def compliant(half, n):
    full = list(half) + list(half)[::-1]
    return manufacturing_ok(full) and guidelines_ok(full, ALPHA)


def pm45(half):
    return (half.count(45) + half.count(-45)) / len(half)


def insert_pair(half, rng, tries=400):
    """Porta uno half-stack da N_FROM a N_TO inserendo DUE lamine, o dichiara di non riuscirci.

    Le coppie candidate sono ordinate per quanto coprono il deficit alla soglia NUOVA: prima le
    direzioni sotto soglia, poi il resto. Una coppia e' (+theta, -theta) oppure due fra {0, 90},
    perche' il bilanciamento deve restare chiuso.
    """
    t_new = ten_pct_min_half(N_TO)
    deficit = [a for a in PRINCIPAL if a in ALPHA and half.count(a) < t_new]
    pairs = []
    for a in deficit:                                   # prima chi copre il deficit
        pairs.append((a, -a) if -a in ALPHA and a not in (0, 90) else (a, a))
    for th in sorted({abs(x) for x in ALPHA if x}):     # poi le coppie +-theta
        if th in ALPHA and -th in ALPHA:
            pairs.append((th, -th))
    pairs += [(0, 0), (90, 90), (0, 90)]
    rng.shuffle(pairs)
    m = len(half)
    for _ in range(tries):
        p1, p2 = rng.choice(pairs)
        i, j = sorted((rng.randint(0, m), rng.randint(0, m + 1)))
        cand = list(half)
        cand.insert(i, p1)
        cand.insert(j, p2)
        if len(cand) == N_TO // 2 and compliant(cand, N_TO):
            return cand
    return None


def elitist(case, pool, rng, seed_pop):
    """Lo scheletro di exp22: budget in valutazioni FE DISTINTE, figli per rigenerazione esatta."""
    sample, _t, _b, _s = uniform_sampler(ALPHA, N_TO, rng)
    cache = {}

    def ev(pop):
        fresh = []
        for h in pop:
            k = tuple(h)
            if k not in cache and k not in {tuple(x) for x in fresh}:
                fresh.append(h)
        if fresh:
            for h, v in zip(fresh, pool.map(_bf_symmetric, [(h, case) for h in fresh])):
                cache[tuple(h)] = v
        return [cache[tuple(h)] for h in pop], len(fresh)

    P = list(seed_pop)
    fits, spent = ev(P)
    best, besth, gens = max(fits), P[fits.index(max(fits))], 0
    while spent < BUDGET_DISTINCT and gens < MAX_GENS:
        rank = [p for _, p in sorted(zip(fits, P), key=lambda x: -x[0])]
        P = rank[:ELITE] + [sample() for _ in range(POP - ELITE)]
        fits, s = ev(P)
        spent += s
        if max(fits) > best:
            best, besth = max(fits), P[fits.index(max(fits))]
        gens += 1
    return round(best, 4), besth, spent, gens


def main():
    nproc = int(os.environ.get("NPROC", max(1, (os.cpu_count() or 2) - 2)))
    case40, case44 = CASES[CASE], CASES[CASE]
    print(f"exp23: Direzione 2, caso {CASE}, alfabeto esteso, N {N_FROM} -> {N_TO}, "
          f"{len(SEEDS)} semi, {nproc} processi")
    print(f"  soglia del 10%: t={ten_pct_min_half(N_FROM)} a {N_FROM}, "
          f"t={ten_pct_min_half(N_TO)} a {N_TO} — e' questo che rende l'inserzione vincolata\n")

    out = {"cold": [], "transport": [], "pm45_cold": [], "pm45_transport": [],
           "fallback_transport": [], "best40": [], "pm45_40": []}
    pool = mp.Pool(nproc)
    try:
        for seed in SEEDS:
            # --- COLD: restart indipendente a N=44 -------------------------------------
            rng = random.Random(seed * 10 + 3)
            s44, _t, _b, _s = uniform_sampler(ALPHA, N_TO, rng)
            b, h, _sp, _g = elitist(case44, pool, rng, [s44() for _ in range(POP)])
            out["cold"].append(b)
            out["pm45_cold"].append(round(pm45(h), 4))

            # --- il run a N=40 da cui si trasporta -------------------------------------
            rng40 = random.Random(seed * 10 + 5)
            s40, _t, _b, _s = uniform_sampler(ALPHA, N_FROM, rng40)
            P40 = [s40() for _ in range(POP)]
            f40 = pool.map(_bf_symmetric, [(x, case40) for x in P40])
            for _ in range(10):
                rank40 = [p for _, p in sorted(zip(f40, P40), key=lambda x: -x[0])]
                P40 = rank40[:ELITE] + [s40() for _ in range(POP - ELITE)]
                f40 = pool.map(_bf_symmetric, [(x, case40) for x in P40])
            elite40 = [p for _, p in sorted(zip(f40, P40), key=lambda x: -x[0])][:ELITE]
            out["best40"].append(round(max(f40), 4))
            out["pm45_40"].append(round(st.mean(pm45(x) for x in elite40), 4))

            # --- TRANSPORT: le elite di N=40 portate dentro N=44 -----------------------
            rngT = random.Random(seed * 10 + 7)
            sT, _t, _b, _s = uniform_sampler(ALPHA, N_TO, rngT)
            seeded, fb = [], 0
            for x in elite40:
                c = insert_pair(x, rngT)
                if c is None:
                    fb += 1
                    c = sT()
                seeded.append(c)
            while len(seeded) < POP:
                seeded.append(sT())
            b, h, _sp, _g = elitist(case44, pool, rngT, seeded)
            out["transport"].append(b)
            out["pm45_transport"].append(round(pm45(h), 4))
            out["fallback_transport"].append(fb)
            print(f"  seme {seed:>2}: N40 best {out['best40'][-1]:.3f} (+-45 elite "
                  f"{out['pm45_40'][-1]:.2f}) | N44 COLD {out['cold'][-1]:.3f} "
                  f"TRANSPORT {out['transport'][-1]:.3f} (fallback {fb}/{ELITE})", flush=True)
    finally:
        pool.close()
        pool.join()

    p = float(mannwhitneyu(out["transport"], out["cold"], alternative="greater").pvalue)
    out["VERDICT"] = dict(
        mean_cold=round(st.mean(out["cold"]), 4), mean_transport=round(st.mean(out["transport"]), 4),
        best_cold=max(out["cold"]), best_transport=max(out["transport"]),
        p_one_sided=sig(p), beats=p < 0.05,
        OUTCOME="CONFERMATO" if p < 0.05 else "FALSIFICATO",
        reaches_4_5=max(out["transport"]) >= 4.5,
        fallback_rate=round(st.mean(out["fallback_transport"]) / ELITE, 3))
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    json.dump(out, open(DATA_PATH, "w"), indent=1)

    v = out["VERDICT"]
    print(f"\n=== PRE-REGISTRAZIONE #3, criterio primario ===")
    print(f"  COLD      media {v['mean_cold']:.3f}  best {v['best_cold']:.3f}")
    print(f"  TRANSPORT media {v['mean_transport']:.3f}  best {v['best_transport']:.3f}")
    print(f"  Mann-Whitney a una coda, TRANSPORT > COLD: p = {v['p_one_sided']}")
    print(f"  >>> ESITO: {v['OUTCOME']}")
    print(f"  [second.] il best raggiunge ~4,5: {v['reaches_4_5']}")
    print(f"  [second.] +-45 nel migliore: COLD {st.mean(out['pm45_cold']):.2f}, "
          f"TRANSPORT {st.mean(out['pm45_transport']):.2f}, elite a N=40 {st.mean(out['pm45_40']):.2f}")
    print(f"  [second.] inserzioni fallite e ricadute sul campionatore: {100*v['fallback_rate']:.0f}%")
    print(f"\nscritto {DATA_PATH}")


if __name__ == "__main__":
    main()
