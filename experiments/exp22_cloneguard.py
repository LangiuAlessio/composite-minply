"""exp22_cloneguard.py -- DFAX* contro REGEN a parita' di DESIGN DISTINTI, non di posti valutati.

Corregge il confound che `exp21` ha misurato: in `exp20` il budget era identico in valutazioni FE
(176) ma REGEN vedeva 126 design distinti e DFAX 29-36. Quell'esperimento confrontava quanti
punti nuovi vede ciascun braccio, non la qualita' dell'ereditarieta'.

QUI IL BUDGET E' 176 VALUTAZIONI FE **DISTINTE**. Le fitness dei duplicati vengono da una cache e
non consumano budget; le generazioni sono variabili e si va avanti finche' il budget di punti
nuovi non e' esaurito (tetto a 40 generazioni). Cosi' ogni braccio vede lo stesso numero di
design mai visti prima, che e' l'unica definizione di «pari budget» che misuri l'operatore.

Tre bracci:
  REGEN   invariato: figli per rigenerazione, nessuna ereditarieta'. E' ricerca casuale ESATTA
          sul linguaggio conforme, e non e' un avversario debole: esiste solo grazie al DP.
  DFAX    invariato: serve a misurare a quanti design distinti SATURA. Se non riesce a spendere
          il budget nemmeno in 40 generazioni, quello e' il risultato.
  DFAX*   DFAX + clone-guard: se il figlio coincide con un genitore o con un individuo gia' in
          popolazione, si ricade sul campionatore uniforme esatto. Preserva l'ereditarieta'
          (i figli restano nella regione ad alto fitness) e ripristina la parita' di punti nuovi.

⚠️ Criterio in `IDEA_ALGORITMO_DFA_2026-08-06.md`, sezione «PRE-REGISTRAZIONE #2», scritto e
committato PRIMA di questo run. Primario: DFAX* batte REGEN su c2_side E c3_combo, Mann-Whitney a
una coda, Holm sulla famiglia per alfabeto, alpha 0,05. Tre esiti.

Uso:  PYTHONPATH=$PWD CCX_BIN=/usr/bin/ccx NPROC=30 python3 -m experiments.exp22_cloneguard
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

from optimisers.constrained_search import (CASES, ALPHABETS, gen_guided,
                                           manufacturing_ok, guidelines_ok)
from optimisers.laminate_language import Layout, uniform_sampler

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "algo_dfa"))
from crossover_states import dfa_parikh_cuts                                    # noqa: E402

from experiments.exp20_dfa_crossover import _bf_symmetric, N, HALF, POP         # noqa: E402
from experiments.exp4_optimiser_comparison import holm, sig                     # noqa: E402

BUDGET_DISTINCT = 176      # gli stessi 176 di exp20, ma contati in DESIGN MAI VISTI
MAX_GENS = 40              # tetto: senza, un braccio che satura girerebbe per sempre
ELITE = max(2, POP // 3)
SEEDS = list(range(1, 31))
CASES_ORDER = ("c1_axial", "c2_side", "c3_combo")
ARMS = ("REGEN", "DFAX", "DFAXSTAR")
OPT_OFFSET = {"REGEN": 7, "DFAX": 9, "DFAXSTAR": 11}
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "_out", "exp22_cloneguard.json")


def run(arm, case, alpha, pool, rng):
    """Un run a budget di DESIGN DISTINTI. Restituisce (best, distinti, generazioni, guardie)."""
    sample, _t, _b, _s = uniform_sampler(alpha, N, rng)
    L = Layout(alpha, N)
    cache = {}                       # half-stack -> fitness. E' qui che si risparmia budget.
    guards = 0                       # quante volte il clone-guard e' intervenuto

    def evaluate(pop):
        """Valuta SOLO i cache miss. Restituisce le fitness e quanti punti nuovi ha speso."""
        fresh = []
        for h in pop:
            k = tuple(h)
            if k not in cache and k not in [tuple(x) for x in fresh]:
                fresh.append(h)
        if fresh:
            vals = pool.map(_bf_symmetric, [(h, case) for h in fresh])
            for h, v in zip(fresh, vals):
                cache[tuple(h)] = v
        return [cache[tuple(h)] for h in pop], len(fresh)

    def make(elite, r, banned):
        nonlocal guards
        if arm == "REGEN":
            full = gen_guided(alpha, N, r)
            return full[:HALF] if full else sample()
        ha, hb = r.choice(elite), r.choice(elite)
        cuts = dfa_parikh_cuts(list(ha), list(hb), L)
        if cuts:
            i = r.choice(cuts)                # UN solo sorteggio: due estrarrebbero tagli diversi
            child = list(ha)[:i] + list(hb)[i:]
            if arm == "DFAX":
                return child
            # clone-guard: un figlio identico a un genitore o a un individuo gia' presente non
            # porta informazione. Non lo si scarta e basta: si sostituisce con un'estrazione
            # uniforme esatta, cosi' il posto in popolazione resta e il budget non si perde.
            if tuple(child) not in banned:
                return child
            guards += 1
            return sample()
        return sample()

    P = [sample() for _ in range(POP)]
    fits, spent = evaluate(P)
    best = max(fits)
    gens = 0
    while spent < BUDGET_DISTINCT and gens < MAX_GENS:
        rank = [p for _, p in sorted(zip(fits, P), key=lambda x: -x[0])]
        elite = rank[:ELITE]
        banned = {tuple(h) for h in P}
        kids = []
        while len(kids) < POP - ELITE:
            c = make(elite, rng, banned | {tuple(x) for x in kids})
            full = list(c) + list(c)[::-1]
            assert manufacturing_ok(full) and guidelines_ok(full, alpha), "figlio non conforme"
            kids.append(c)
        P = elite + kids
        fits, s = evaluate(P)
        spent += s
        best = max(best, max(fits))
        gens += 1
    return round(best, 4), spent, gens, guards


def campaign(alpha_name, pool):
    alpha = ALPHABETS[alpha_name]
    out = {}
    for cname in CASES_ORDER:
        case = CASES[cname]
        vals = {a: [] for a in ARMS}
        dist = {a: [] for a in ARMS}
        gens = {a: [] for a in ARMS}
        grd = {a: [] for a in ARMS}
        for seed in SEEDS:
            for arm in ARMS:
                b, d, g, gu = run(arm, case, alpha, pool, random.Random(seed * 10 + OPT_OFFSET[arm]))
                vals[arm].append(b)
                dist[arm].append(d)
                gens[arm].append(g)
                grd[arm].append(gu)
        raw = {f"DFAXSTAR-{a}": float(mannwhitneyu(vals["DFAXSTAR"], vals[a],
                                                   alternative="greater").pvalue)
               for a in ARMS if a != "DFAXSTAR"}
        out[cname] = {a: dict(mean=round(st.mean(vals[a]), 4), std=round(st.pstdev(vals[a]), 4),
                              best=round(max(vals[a]), 4),
                              distinct=round(st.mean(dist[a]), 1),
                              gens=round(st.mean(gens[a]), 1),
                              guards=round(st.mean(grd[a]), 1)) for a in ARMS}
        out[cname]["_per_seed"] = vals
        out[cname]["MWU_raw"] = {k: sig(v) for k, v in raw.items()}
        print(f"[{alpha_name}] {cname}: " +
              " ".join(f"{a}={out[cname][a]['mean']:.2f}(d={out[cname][a]['distinct']:.0f},"
                       f"g={out[cname][a]['gens']:.0f})" for a in ARMS), flush=True)
    return out


def verdict(res_alpha):
    """Il criterio della PRE-REGISTRAZIONE #2, alla lettera."""
    raw = {}
    for cname in CASES_ORDER:
        v = res_alpha[cname]["_per_seed"]
        for a in ARMS:
            if a == "DFAXSTAR":
                continue
            raw[f"{cname}|DFAXSTAR-{a}"] = float(
                mannwhitneyu(v["DFAXSTAR"], v[a], alternative="greater").pvalue)
    hol = holm(raw)
    beats = {c: hol[f"{c}|DFAXSTAR-REGEN"] < 0.05 for c in ("c2_side", "c3_combo")}
    n = sum(beats.values())
    return dict(family_size=len(raw), holm=hol, beats_regen=beats,
                OUTCOME="CONFERMATO" if n == 2 else "FALSIFICATO" if n == 0 else "INCONCLUSIVO")


def main():
    nproc = int(os.environ.get("NPROC", max(1, (os.cpu_count() or 2) - 2)))
    print(f"exp22: budget = {BUDGET_DISTINCT} valutazioni FE DISTINTE per run, "
          f"tetto {MAX_GENS} generazioni, {nproc} processi")
    res = {}
    pool = mp.Pool(nproc)
    try:
        for aname in ("set1", "set2"):
            res[aname] = campaign(aname, pool)
    finally:
        pool.close()
        pool.join()
    for aname in res:
        res[aname]["VERDICT"] = verdict(res[aname])
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    json.dump(res, open(DATA_PATH, "w"), indent=1)
    for aname in res:
        v = res[aname]["VERDICT"]
        print(f"\n=== {aname}: PRE-REGISTRAZIONE #2, famiglia Holm da {v['family_size']} ===")
        for c, b in v["beats_regen"].items():
            print(f"  DFAX* batte REGEN su {c}: {b}  (Holm p = {v['holm'][c + '|DFAXSTAR-REGEN']})")
        print(f"  >>> ESITO: {v['OUTCOME']}")
        for c in CASES_ORDER:
            print(f"  [esplor.] {c}: distinti " + ", ".join(
                f"{a}={res[aname][c][a]['distinct']:.0f}" for a in ARMS) +
                "  | guardie DFAX* = " + f"{res[aname][c]['DFAXSTAR']['guards']:.0f}")
    print(f"\nscritto {DATA_PATH}")


if __name__ == "__main__":
    main()
