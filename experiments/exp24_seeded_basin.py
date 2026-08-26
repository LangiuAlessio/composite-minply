"""exp24_seeded_basin.py -- il bacino +-45 COSTRUITO per condizionamento, non sperato.

`exp23` non ha potuto testare l'ipotesi dei bacini: l'elite trasportata da N=40 aveva +-45 a 0,290,
cioe' non era nel bacino, e il criterio primario ha finito per misurare l'avvio a caldo. Qui il
bacino si costruisce.

PERCHE' IL RESTART NON CI ARRIVA MAI, misurato: sul linguaggio conforme a N=44 (esteso), gli
half-stack con +-45 >= 12 su 22 sono l'**1,13%**, e quelli con +-45 >= 14 lo **0,0154%**. Una
popolazione uniforme di 16 individui ne contiene 0,18 sopra 12 e 0,0025 sopra 14. Il campionamento
condizionato non e' un lusso: e' l'unico modo di metterci un piede.

IL METODO E' QUELLO DI §3.6 DEL PAPER. Il DP tiene esatti i contatori delle direzioni principali,
quindi condizionare sul vettore di Parikh e' una condizione sulle **celle terminali** — esattamente
come per il criterio di rigidezza di Irisarri. Si filtrano i finali, si rinormalizza, e la
camminata all'indietro resta **uniforme dentro la regione condizionata**: nessun rigetto, nessuna
distorsione.

⚠️ GATE SULLA PREMESSA, ed e' la lezione di `exp23`: prima di spendere una sola valutazione FE si
verifica che ogni individuo seminato soddisfi davvero la condizione. Se anche uno solo non la
soddisfa lo script muore. In `exp23` la premessa era un endpoint descrittivo, e l'abbiamo scoperta
falsa **dopo** 16k solve.

Criterio in `IDEA_ALGORITMO_DFA_2026-08-06.md`, «PRE-REGISTRAZIONE #4», committata prima di questo
file.

Uso:  PYTHONPATH=$PWD CCX_BIN=/usr/bin/ccx NPROC=30 python3 -m experiments.exp24_seeded_basin
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
from optimisers.laminate_language import _prepare, _unstep

from experiments.exp20_dfa_crossover import _bf_symmetric, POP                    # noqa: E402
from experiments.exp4_optimiser_comparison import holm, sig                        # noqa: E402

ALPHA = ALPHABETS["set2"]
CASE = "c1_axial"
N = 44
HALF = N // 2
BUDGET_DISTINCT = 176
MAX_GENS = 40
ELITE = max(2, POP // 3)
SEEDS = list(range(1, 31))
ARMS = ("COLD", "PM45-12", "PM45-14")
THRESHOLD = {"COLD": 0, "PM45-12": 12, "PM45-14": 14}
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "_out", "exp24_seeded_basin.json")


def conditioned_sampler(alpha, n, rng, pm45_min):
    """Campionatore uniforme sul linguaggio conforme CONDIZIONATO a c[+45]+c[-45] >= pm45_min.

    Identico a quello del modulo, tranne che i finali ammessi sono filtrati sui contatori. Poiche'
    il DP tiene +45 e -45 esatti, il filtro e' esatto e la camminata all'indietro resta uniforme
    dentro la regione: non e' campionamento per rigetto travestito.
    """
    P = _prepare(alpha, n)
    L, S, levels, preds = P['L'], P['S'], P['levels'], P['preds']
    i45, i_45 = L.ei[45], L.ei[-45]
    finals = [(k, w) for k, w in P['finals'] if k[1][i45] + k[1][i_45] >= pm45_min]
    total = sum(w for _, w in finals)
    if total == 0:
        sys.exit(f"regione vuota: nessun half-stack conforme con +-45 >= {pm45_min}")

    def draw():
        r = rng.randrange(total)
        for key, w in finals:
            if r < w:
                break
            r -= w
        si, ex, df = key
        seq = [S[si][0]]
        for j in range(L.m - 1, 0, -1):
            b = S[si][0]
            ex_p, df_p = _unstep(L, ex, df, b)
            cands, tot_p = [], 0
            for pi in preds[si]:
                w = levels[j - 1].get((pi, ex_p, df_p))
                if w:
                    cands.append((pi, w))
                    tot_p += w
            r = rng.randrange(tot_p)
            for pi, w in cands:
                if r < w:
                    break
                r -= w
            si, ex, df = pi, ex_p, df_p
            seq.append(S[si][0])
        seq.reverse()
        return seq
    return draw, total


def pm45(h):
    return h.count(45) + h.count(-45)


def elitist(case, pool, rng, seed_pop, refill):
    """Scheletro di exp22/exp23: budget in valutazioni DISTINTE, figli per rigenerazione."""
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
    best = max(fits)
    besth = P[fits.index(best)]
    gens = 0
    while spent < BUDGET_DISTINCT and gens < MAX_GENS:
        rank = [p for _, p in sorted(zip(fits, P), key=lambda x: -x[0])]
        P = rank[:ELITE] + [refill() for _ in range(POP - ELITE)]
        fits, s = ev(P)
        spent += s
        if max(fits) > best:
            best, besth = max(fits), P[fits.index(max(fits))]
        gens += 1
    return round(best, 4), besth


def main():
    nproc = int(os.environ.get("NPROC", max(1, (os.cpu_count() or 2) - 2)))
    case = CASES[CASE]
    print(f"exp24: bacino +-45 seminato per condizionamento, {CASE}, esteso, N={N}, "
          f"{len(SEEDS)} semi, {nproc} processi\n")

    # --- quanto e' rara ciascuna regione, prima di tutto -------------------------------
    r0 = random.Random(0)
    _d, tot_free = conditioned_sampler(ALPHA, N, r0, 0)
    for arm in ARMS[1:]:
        _d, m = conditioned_sampler(ALPHA, N, r0, THRESHOLD[arm])
        print(f"  {arm}: +-45 >= {THRESHOLD[arm]:>2}  ->  {m:,} half-stack, "
              f"{100*m/tot_free:.4f}% del linguaggio")
    print()

    out = {a: {"best": [], "pm45_final": []} for a in ARMS}
    pool = mp.Pool(nproc)
    try:
        for seed in SEEDS:
            for arm in ARMS:
                rng = random.Random(seed * 10 + 3 + 2 * ARMS.index(arm))
                draw, _m = conditioned_sampler(ALPHA, N, rng, THRESHOLD[arm])
                refill, _m2 = conditioned_sampler(ALPHA, N, rng, 0)   # i figli sono sempre liberi
                seeded = [draw() for _ in range(POP)]

                # --- GATE SULLA PREMESSA: prima di spendere un solo solve FE -----------
                for h in seeded:
                    if pm45(h) < THRESHOLD[arm]:
                        sys.exit(f"GATE FALLITO ({arm}, seme {seed}): individuo con +-45="
                                 f"{pm45(h)} < {THRESHOLD[arm]}. La premessa del test non e' "
                                 "soddisfatta e l'esperimento si ferma qui, prima di spendere FE.")
                    full = list(h) + list(h)[::-1]
                    if not (manufacturing_ok(full) and guidelines_ok(full, ALPHA)):
                        sys.exit(f"GATE FALLITO ({arm}, seme {seed}): seme non conforme.")

                b, h = elitist(case, pool, rng, seeded, refill)
                out[arm]["best"].append(b)
                out[arm]["pm45_final"].append(pm45(h))
            print(f"  seme {seed:>2}: " + "  ".join(
                f"{a} {out[a]['best'][-1]:.3f} (+-45 fin. {out[a]['pm45_final'][-1]:>2})"
                for a in ARMS), flush=True)
    finally:
        pool.close()
        pool.join()

    raw = {a: float(mannwhitneyu(out[a]["best"], out["COLD"]["best"],
                                 alternative="greater").pvalue) for a in ARMS[1:]}
    hol = holm(raw)
    beats = {a: hol[a] < 0.05 for a in ARMS[1:]}
    out["VERDICT"] = dict(
        means={a: round(st.mean(out[a]["best"]), 4) for a in ARMS},
        bests={a: max(out[a]["best"]) for a in ARMS},
        holm={a: hol[a] for a in ARMS[1:]}, beats_cold=beats,
        OUTCOME="CONFERMATO" if any(beats.values()) else "FALSIFICATO",
        pm45_final={a: round(st.mean(out[a]["pm45_final"]), 2) for a in ARMS})
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    json.dump(out, open(DATA_PATH, "w"), indent=1)

    v = out["VERDICT"]
    print("\n=== PRE-REGISTRAZIONE #4 ===")
    for a in ARMS:
        print(f"  {a:<8} media {v['means'][a]:.3f}   best {v['bests'][a]:.3f}   "
              f"+-45 nel migliore finale {v['pm45_final'][a]:.1f}/{HALF}")
    for a in ARMS[1:]:
        print(f"  {a} batte COLD: {beats[a]}  (Holm p = {v['holm'][a]})")
    print(f"  >>> ESITO: {v['OUTCOME']}")
    print(f"\nscritto {DATA_PATH}")


if __name__ == "__main__":
    main()
