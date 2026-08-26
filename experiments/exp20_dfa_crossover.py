"""exp20_dfa_crossover.py -- fase uno: l'operatore DFA-Parikh contro la rigenerazione, a parita' di budget FE.

E' l'unico esperimento che puo' dire se il linguaggio serve a CERCARE meglio, e non solo a
contare e campionare. Fino a qui i claim su crossover e riparazione sono STRUTTURALI (validita',
deriva, costo, tenuta), misurati a zero solve FE in `algo_dfa/RISULTATI_crossover_2026-08-06.md`.
Qui si spende budget FE e si guarda la qualita' della ricerca.

I CINQUE BRACCI, tutti a budget FE identico (pop=16, gens=10, popolazione intera valutata a ogni
generazione, cioe' 16 + 10x16 = 176 valutazioni per run):

  GA, ACO, PSO   il trio pubblicato, invariato, da `optimisers.metaheuristics`. Cercano nel
                 linguaggio della sola MANIFATTURA e usano `repair`.
  REGEN          GA sul linguaggio CONFORME, figli per RIGENERAZIONE (`gen_guided`): e' cio' che
                 il ramo guidato fa oggi.
  DFAX           GA sul linguaggio CONFORME, figli per TAGLIO DFA-Parikh fra due genitori d'elite,
                 con fallback sul campionatore uniforme esatto quando nessun taglio esiste.
                 Mai `repair`.

Il confronto che interessa a §5 e' DFAX contro REGEN, appaiato per seme. Il confronto col trio
serve al criterio pre-registrato.

⚠️ IL CRITERIO E' PRE-REGISTRATO in `IDEA_ALGORITMO_DFA_2026-08-06.md` e non si tocca dopo aver
visto i risultati: *«30 semi x 3 casi; falsificata se non batte il trio su C2/C3 dopo Holm e non
alza la frazione di semi che raggiunge il bacino +-45 su C1»*. Una cosa il criterio non la
specificava, e va fissata QUI, prima di guardare i numeri: **«bacino +-45» = lo half-stack
migliore ha almeno meta' delle sue lamine a +-45**, cioe' `c(+45)+c(-45) >= m/2`. E' la
definizione simmetrica a quella di «0-dominato» gia' usata nei referti (`k >= m/2`).

Uso:
    PYTHONPATH=$PWD CCX_BIN=/usr/bin/ccx python3 -m experiments.exp20_dfa_crossover
    ... --smoke          3 semi, un caso, un alfabeto: valida il codice, non conclude nulla
    ... --restat         solo statistiche dal record per seme gia' salvato, zero solve FE
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

from optimisers.constrained_search import (buckling_factor as _real_bf, CASES, ALPHABETS,
                                           gen_guided, manufacturing_ok, guidelines_ok)
from optimisers.laminate_language import Layout, uniform_sampler
import optimisers.metaheuristics as mh

# L'operatore vive in algo_dfa/, che non e' un package: si aggiunge al path e si importa.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "algo_dfa"))
from crossover_states import dfa_parikh_cuts                                    # noqa: E402

# Holm e l'arrotondamento a cifre significative si IMPORTANO da exp4, non si riscrivono: quella
# implementazione porta gia' la correzione dell'audit 2026-07-22 (Holm va applicato ai p-value
# GREZZI, non a quelli stampati). Riscriverli qui vorrebbe dire riaprire quel bug.
from experiments.exp4_optimiser_comparison import holm, sig                     # noqa: E402

N = 40
HALF = N // 2
POP, GENS = 16, 10
BUDGET = POP + GENS * POP          # 176 valutazioni FE per run, identiche per ogni braccio
SEEDS = list(range(1, 31))
CASES_ORDER = ("c1_axial", "c2_side", "c3_combo")
ARMS = ("GA", "ACO", "PSO", "REGEN", "DFAX")
OPT_OFFSET = {"GA": 1, "ACO": 3, "PSO": 5, "REGEN": 7, "DFAX": 9}
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "_out", "exp20_dfa_crossover.json")


def _bf_symmetric(args):
    """Buckling factor del laminato simmetrico costruito da uno half-stack."""
    half, case = args
    return _real_bf((list(half) + list(half)[::-1], case))


mh.buckling_factor = _bf_symmetric


def _eval(pop, case, pool):
    return pool.map(_bf_symmetric, [(p, case) for p in pop])


def _elite_ga(case, alpha, pool, rng, make_child, seed_pool):
    """Lo SCHELETRO comune a REGEN e DFAX: identico a `mh.ga` in tutto tranne il figlio.

    Stessa taglia di popolazione, stesso numero di generazioni, stessa elite, e la popolazione
    intera rivalutata a ogni generazione. Cosi' i due bracci differiscono per UNA cosa sola, che
    e' il punto dell'esperimento: se cambiassi anche il budget non saprei a cosa attribuire la
    differenza.
    """
    P = [seed_pool() for _ in range(POP)]
    fits = _eval(P, case, pool)
    used = len(P)
    best = max(zip(fits, P), key=lambda x: x[0])
    fallbacks = 0
    for _ in range(GENS):
        rank = [p for _, p in sorted(zip(fits, P), key=lambda x: -x[0])]
        elite = rank[:max(2, POP // 3)]
        kids = []
        while len(kids) < POP - len(elite):
            child, fell = make_child(elite, rng)
            fallbacks += fell
            full = list(child) + list(child)[::-1]
            assert manufacturing_ok(full) and guidelines_ok(full, alpha), \
                "figlio NON conforme: il braccio si e' rotto, non si continua"
            kids.append(child)
        P = elite + kids
        fits = _eval(P, case, pool)
        used += len(P)
        best = max([best] + list(zip(fits, P)), key=lambda x: x[0])
    assert used == BUDGET, f"budget FE {used} != {BUDGET}: i bracci non sono confrontabili"
    return best[0], best[1], fallbacks


def run_arm(arm, case, alpha, pool, rng):
    """Un run completo di un braccio. Restituisce (fitness, half-stack, fallback)."""
    if arm in ("GA", "ACO", "PSO"):
        mh.ALPHA = alpha
        fn = {"GA": mh.ga, "ACO": mh.aco, "PSO": mh.pso}[arm]
        kw = {"GA": dict(pop=POP, gens=GENS), "ACO": dict(ants=POP, iters=GENS),
              "PSO": dict(swarm=POP, iters=GENS)}[arm]
        fit, half = fn(case, HALF, pool, rng, **kw)
        return fit, half, 0

    sample, _total, _by_k, _st = uniform_sampler(alpha, N, rng)
    L = Layout(alpha, N)

    def seed_pool():
        return sample()

    if arm == "REGEN":
        def make_child(elite, r):
            # La pratica attuale del ramo guidato: si rigenera da zero finche' non e' conforme.
            # A N=40, che e' la taglia di QUESTO esperimento, gen_guided quasi non fallisce: il
            # record della campagna conta 0 fallback su 9.900 chiamate su set1 e 3 su 9.900 su set2
            # (`_fallbacks` nel JSON). Il «~6%» che stava qui e' la cifra di N=44 di §4 del paper, e
            # riportarla a questa taglia e' l'errore corretto il 2026-08-10, RS-REGEN-NON-ESATTO.
            # Quando fallisce si ricade sul campionatore esatto, cosi' il braccio non muore e il
            # budget FE resta identico a quello di DFAX.
            full = gen_guided(alpha, N, r)
            if full:
                return full[:HALF], 0
            return sample(), 1
    else:                                                   # DFAX
        def make_child(elite, r):
            ha, hb = r.choice(elite), r.choice(elite)
            cuts = dfa_parikh_cuts(list(ha), list(hb), L)
            if cuts:
                i = r.choice(cuts)
                return list(ha)[:i] + list(hb)[i:], 0
            return sample(), 1                              # nessun taglio: campionatore esatto

    return _elite_ga(case, alpha, pool, rng, make_child, seed_pool)


def pm45_basin(half, alpha):
    """Lo half-stack e' nel bacino +-45? Definizione fissata PRIMA di vedere i risultati:
    almeno meta' delle lamine a +-45, simmetrica a quella di «0-dominato» dei referti."""
    return (half.count(45) + half.count(-45)) >= len(half) / 2


def is_compliant(half, alpha):
    """Il laminato simmetrico costruito da questo half-stack rispetta le linee guida?

    Serve a sapere se un confronto e' OMOGENEO. Il trio cerca nel linguaggio della sola
    manifattura -- il paper 1 lo dichiara: «balance and the 10% rule [...] are not imposed in
    this campaign [...] does not deliver designs» -- mentre REGEN e DFAX cercano nel linguaggio
    CONFORME, che e' 13x piu' piccolo sul set ristretto. Confrontare un vincolato con un non
    vincolato non misura l'operatore, misura il vincolo.
    """
    full = list(half) + list(half)[::-1]
    return bool(manufacturing_ok(full) and guidelines_ok(full, alpha))


def summarise(vals: dict, basins: dict, compl: dict) -> dict:
    """Sintesi e test per un caso. Separata dalla ricerca cosi' `--restat` la rifa' senza FE."""
    out = {a: dict(mean=round(st.mean(vals[a]), 4), std=round(st.pstdev(vals[a]), 4),
                   best=round(max(vals[a]), 4),
                   basin_pm45=round(sum(basins[a]) / len(basins[a]), 4),
                   compliant=round(sum(compl[a]) / len(compl[a]), 4)) for a in ARMS}
    # p-value GREZZI: Holm si applica a questi, mai a quelli arrotondati (audit 2026-07-22).
    raw = {}
    for a in ARMS:
        if a == "DFAX":
            continue
        raw[f"DFAX-{a}"] = float(mannwhitneyu(vals["DFAX"], vals[a], alternative="greater").pvalue)
    out["MWU_raw"] = {k: sig(v) for k, v in raw.items()}
    out["MWU_holm"] = holm(raw)
    return out


def verdict(res):
    """Il criterio PRE-REGISTRATO, applicato alla lettera E CON LA SUA APPLICABILITA' DICHIARATA.

    Il criterio dice «batte il trio su C2/C3». Ma il trio non e' vincolato alle linee guida e
    DFAX si': se la conformita' del trio e' prossima a zero, quel confronto non e' una gara fra
    operatori, e' un confronto fra due insiemi ammissibili diversi. Lo script NON riscrive il
    criterio -- non si tocca dopo aver visto i dati -- ma dichiara `criterion_applicable`, cosi'
    la decisione resta a chi legge invece di essere presa di nascosto qui dentro.
    """
    beats = {c: all(res[c]["MWU_holm"][f"DFAX-{a}"] < 0.05 for a in ("GA", "ACO", "PSO"))
             for c in ("c2_side", "c3_combo")}
    b_dfax = res["c1_axial"]["DFAX"]["basin_pm45"]
    b_trio = max(res["c1_axial"][a]["basin_pm45"] for a in ("GA", "ACO", "PSO"))
    raises = b_dfax > b_trio
    falsified = not (beats["c2_side"] and beats["c3_combo"]) and not raises
    trio_compl = max(res[c][a]["compliant"] for c in CASES_ORDER for a in ("GA", "ACO", "PSO"))
    # Confronto appaiato che invece E' omogeneo: stesso spazio, stesso budget, una differenza.
    head2head = {c: sig(float(mannwhitneyu(res[c]["_per_seed"]["DFAX"],
                                           res[c]["_per_seed"]["REGEN"],
                                           alternative="greater").pvalue))
                 for c in CASES_ORDER}
    return dict(beats_trio_c2=beats["c2_side"], beats_trio_c3=beats["c3_combo"],
                basin_dfax_c1=b_dfax, basin_trio_c1=b_trio, raises_basin=raises,
                FALSIFIED=falsified,
                trio_max_compliance=trio_compl,
                criterion_applicable=trio_compl > 0.0,
                DFAX_vs_REGEN_p=head2head)


def verdict_amended(res_alpha):
    """Il criterio EMENDATO (2026-08-06 17:17 UTC), implementato prima che i risultati esistessero.

    Famiglia di Holm: TUTTI i confronti di DFAX dentro un alfabeto, cioe' 3 casi x 4 bracci = 12
    p-value. E' piu' severa della correzione per singolo caso, quindi lavora CONTRO la nostra
    ipotesi: e' la direzione giusta in cui sbagliare.

    Tre esiti e non due, perche' «non confermato» e «falsificato» non sono la stessa cosa:
      CONFERMATO   DFAX batte REGEN su C2 E su C3
      INCONCLUSIVO su esattamente uno dei due
      FALSIFICATO  su nessuno dei due
    L'endpoint sul bacino +-45 di C1 e' ESPLORATIVO e non entra qui: quel caso era gia' stato
    visto quando il criterio e' stato emendato, e si riporta soltanto.
    """
    raw = {}
    for cname in CASES_ORDER:
        vals = res_alpha[cname]["_per_seed"]
        for a in ARMS:
            if a == "DFAX":
                continue
            raw[f"{cname}|DFAX-{a}"] = float(
                mannwhitneyu(vals["DFAX"], vals[a], alternative="greater").pvalue)
    hol = holm(raw)
    beats = {c: hol[f"{c}|DFAX-REGEN"] < 0.05 for c in ("c2_side", "c3_combo")}
    nbeat = sum(beats.values())
    outcome = ("CONFERMATO" if nbeat == 2 else "FALSIFICATO" if nbeat == 0 else "INCONCLUSIVO")
    return dict(family_size=len(raw), holm=hol, beats_regen=beats, OUTCOME=outcome,
                exploratory=dict(
                    basin_pm45_c1_dfax=res_alpha["c1_axial"]["DFAX"]["basin_pm45"],
                    basin_pm45_c1_regen=res_alpha["c1_axial"]["REGEN"]["basin_pm45"],
                    dfax_fallbacks={c: res_alpha[c]["_fallbacks"]["DFAX"] for c in CASES_ORDER},
                    trio_compliance={c: {a: res_alpha[c][a]["compliant"]
                                         for a in ("GA", "ACO", "PSO")} for c in CASES_ORDER}))


def campaign(alpha_name, pool, cases, seeds):
    alpha = ALPHABETS[alpha_name]
    out = {}
    for cname in cases:
        case = CASES[cname]
        vals = {a: [] for a in ARMS}
        basins = {a: [] for a in ARMS}
        compl = {a: [] for a in ARMS}
        falls = {a: 0 for a in ARMS}
        for seed in seeds:
            for arm in ARMS:
                rng = random.Random(seed * 10 + OPT_OFFSET[arm])
                fit, half, fb = run_arm(arm, case, alpha, pool, rng)
                vals[arm].append(round(fit, 4))
                basins[arm].append(pm45_basin(list(half), alpha))
                compl[arm].append(is_compliant(list(half), alpha))
                falls[arm] += fb
        out[cname] = summarise(vals, basins, compl)
        out[cname]["_per_seed"] = vals            # il record per seme, che rende --restat vero
        out[cname]["_basins"] = {a: basins[a] for a in ARMS}
        out[cname]["_compliant"] = {a: compl[a] for a in ARMS}
        out[cname]["_fallbacks"] = falls
        print(f"[{alpha_name}] {cname}: " +
              " ".join(f"{a}={out[cname][a]['mean']:.2f}({out[cname][a]['best']:.2f})"
                       for a in ARMS) +
              f"  fallback DFAX={falls['DFAX']}", flush=True)
    return out


def main():
    smoke = "--smoke" in sys.argv
    cases = ("c1_axial",) if smoke else CASES_ORDER
    seeds = SEEDS[:3] if smoke else SEEDS
    alphas = ("set1",) if smoke else ("set1", "set2")
    nproc = int(os.environ.get("NPROC", max(1, (os.cpu_count() or 2) - 2)))
    total = len(alphas) * len(cases) * len(seeds) * len(ARMS) * BUDGET
    print(f"exp20: {len(alphas)} alfabeti x {len(cases)} casi x {len(seeds)} semi x "
          f"{len(ARMS)} bracci x {BUDGET} valutazioni = {total:,} solve FE, {nproc} processi")

    res = {}
    pool = mp.Pool(nproc)
    try:
        for alpha_name in alphas:
            res[alpha_name] = campaign(alpha_name, pool, cases, seeds)
    finally:
        pool.close()
        pool.join()

    if not smoke:
        for alpha_name in alphas:
            res[alpha_name]["VERDICT"] = verdict(res[alpha_name])
            res[alpha_name]["VERDICT_AMENDED"] = verdict_amended(res[alpha_name])
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        json.dump(res, open(DATA_PATH, "w"), indent=1)
        print(f"\nscritto {DATA_PATH}")
        for alpha_name in alphas:
            v = res[alpha_name]["VERDICT"]
            print(f"\n=== criterio pre-registrato, alfabeto {alpha_name} ===")
            print(f"  batte il trio su C2 (Holm<0.05): {v['beats_trio_c2']}")
            print(f"  batte il trio su C3 (Holm<0.05): {v['beats_trio_c3']}")
            print(f"  bacino +-45 su C1: DFAX {v['basin_dfax_c1']:.2f} contro "
                  f"trio {v['basin_trio_c1']:.2f} -> alza: {v['raises_basin']}")
            print(f"  conformita' massima del trio: {v['trio_max_compliance']:.3f}  ->  "
                  f"criterio applicabile: {v['criterion_applicable']}")
            print(f"  DFAX contro REGEN (omogeneo, appaiato): {v['DFAX_vs_REGEN_p']}")
            print(f"  >>> criterio letterale, FALSIFICATA: {v['FALSIFIED']}")
            va = res[alpha_name]["VERDICT_AMENDED"]
            print(f"  --- criterio EMENDATO (famiglia Holm da {va['family_size']}) ---")
            for c, b in va["beats_regen"].items():
                print(f"      DFAX batte REGEN su {c}: {b}  "
                      f"(Holm p={va['holm'][f'{c}|DFAX-REGEN']})")
            print(f"      >>> ESITO: {va['OUTCOME']}")
    else:
        print("\nsmoke: nessun verdetto, il campione non lo sostiene.")


def restat():
    res = json.load(open(DATA_PATH))
    for alpha_name in [k for k in res if k in ALPHABETS]:
        for cname in CASES_ORDER:
            c = res[alpha_name][cname]
            basins = {a: c["_basins"][a] for a in ARMS}
            res[alpha_name][cname] = {**summarise(c["_per_seed"], basins, c["_compliant"]),
                                      "_per_seed": c["_per_seed"], "_basins": c["_basins"],
                                      "_fallbacks": c.get("_fallbacks", {})}
        res[alpha_name]["VERDICT"] = verdict(res[alpha_name])
        res[alpha_name]["VERDICT_AMENDED"] = verdict_amended(res[alpha_name])
    json.dump(res, open(DATA_PATH, "w"), indent=1)
    print("statistiche rigenerate dal record per seme, zero solve FE")


if __name__ == "__main__":
    restat() if "--restat" in sys.argv else main()
