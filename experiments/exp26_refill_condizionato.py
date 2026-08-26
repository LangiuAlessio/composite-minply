"""exp26_refill_condizionato.py -- l'ultimo test della Direzione 2: condizionare ANCHE i figli.

`exp24` ha falsificato che SEMINARE nel bacino +-45 aiuti, e ha mostrato il meccanismo: la
ricerca esce dal bacino appena puo' (parte a 12-14 lamine +-45 e finisce a ~6,7 su 22), perche'
i figli sono estrazioni libere. La domanda che resta, e che nessuno dei tre test precedenti ha
posto: *il migliore DENTRO la regione +-45-dominata e' migliore o peggiore del migliore su tutto
il linguaggio, a parita' di budget?*

LA MODIFICA E' UNA RIGA, ed e' quella che `exp24` stesso indica: il `refill` estrae dal DP
condizionato a +-45 >= soglia (filtro sulle celle terminali + rinormalizzazione, il metodo di
§3.6 gia' usato per la semina) invece che dal linguaggio intero. Popolazione iniziale
condizionata come in `exp24`. Il campionamento resta uniforme DENTRO la regione: nessun rigetto.

Criterio in `IDEA_ALGORITMO_DFA_2026-08-06.md`, «PRE-REGISTRAZIONE #6» (2026-08-10), scritta e
committata PRIMA di questo file. Tre bracci: COLD (IN ARCHIVIO, exp24, non si ripaga) ·
REFILL-12 · REFILL-14. Primario: REFILL-12 batte COLD sul best, Mann-Whitney a una coda,
Holm sui due confronti condizionati contro COLD. FALSIFICATA se nessuno dei due batte COLD.

DUE GATE, stampati DAL RUN:
  1. PERMANENZA (piu' forte di exp24, che controllava la sola popolazione iniziale): ogni
     individuo di OGNI generazione deve stare nella regione. Se uno esce, il run si ferma
     prima di spendere altri FE: non starebbe testando l'ipotesi.
  2. BUDGET: la regione deve poter spendere 176 design distinti; il conteggio della regione
     si RIFA' e si STAMPA qui, non si riprende dalla pre-registrazione. Un braccio che non
     spende il budget e' un risultato (come DFAX in exp22 e EDA-POS in exp25), non un guasto.

Checkpoint per seme (il run e' ~1 h): `_out/exp26_refill_condizionato.ckpt.json`; al riavvio
i semi gia' fatti non si ripagano. I campioni per seme (best, half-stack migliore, +-45
finale, distinti spesi, generazioni, traccia per generazione) si persistono TUTTI nel JSON.

Uso:  PYTHONPATH=$PWD CCX_BIN=/usr/bin/ccx NPROC=16 python3 -m experiments.exp26_refill_condizionato
      SMOKE=1 python3.12 -m experiments.exp26_refill_condizionato   # 0 FE, fitness surrogata
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
from experiments.exp20_dfa_crossover import _bf_symmetric, POP                    # noqa: E402
from experiments.exp24_seeded_basin import conditioned_sampler, pm45              # noqa: E402
from experiments.exp4_optimiser_comparison import holm                            # noqa: E402

ALPHA = ALPHABETS["set2"]
CASE = "c1_axial"
N = 44
HALF = N // 2
BUDGET_DISTINCT = 176
MAX_GENS = 40
ELITE = max(2, POP // 3)
SEEDS = list(range(1, 31))
ARMS = ("REFILL-12", "REFILL-14")
THRESHOLD = {"REFILL-12": 12, "REFILL-14": 14}
SMOKE = os.environ.get("SMOKE") == "1"

_BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_BASE, "_out", "exp26_refill_condizionato.json")
CKPT_PATH = os.path.join(_BASE, "_out", "exp26_refill_condizionato.ckpt.json")
# COLD e' in archivio: il braccio di exp24, 30 semi, stesso caso/alfabeto/budget/harness.
COLD_PATH = os.path.join(os.path.dirname(os.path.dirname(_BASE)),
                         "algo_dfa", "fase_uno_2026-08-06", "exp24_seeded_basin.json")


def _surrogate(args):
    """Fitness FINTA per lo smoke (0 FE): deterministica, distingue individui, nient'altro."""
    h, _case = args
    r = random.Random(hash(tuple(h)) & 0xFFFFFFFF)
    return 2.0 + 0.5 * pm45(h) / HALF + r.random() * 0.1


def elitist_gated(case, pool, seed_pop, refill, thr, evfun):
    """L'elitista di exp24, con il GATE DI PERMANENZA su ogni individuo di ogni generazione."""
    cache = {}

    def check_region(pop, gen):
        for h in pop:
            if pm45(h) < thr:
                sys.exit(f"GATE PERMANENZA FALLITO (soglia {thr}, gen {gen}): individuo con "
                         f"+-45={pm45(h)}. Il run si ferma PRIMA di spendere altri FE.")
            full = list(h) + list(h)[::-1]
            if not (manufacturing_ok(full) and guidelines_ok(full, ALPHA)):
                sys.exit(f"GATE PERMANENZA FALLITO (gen {gen}): individuo non conforme.")

    def ev(pop):
        fresh = []
        for h in pop:
            k = tuple(h)
            if k not in cache and k not in {tuple(x) for x in fresh}:
                fresh.append(h)
        if fresh:
            for h, v in zip(fresh, pool.map(evfun, [(h, case) for h in fresh])):
                cache[tuple(h)] = v
        return [cache[tuple(h)] for h in pop], len(fresh)

    P = list(seed_pop)
    check_region(P, 0)
    fits, spent = ev(P)
    best = max(fits)
    besth = P[fits.index(best)]
    gens = 0
    trace = [round(best, 4)]
    while spent < BUDGET_DISTINCT and gens < MAX_GENS:
        rank = [p for _, p in sorted(zip(fits, P), key=lambda x: -x[0])]
        P = rank[:ELITE] + [refill() for _ in range(POP - ELITE)]
        gens += 1
        check_region(P, gens)
        fits, s = ev(P)
        spent += s
        if max(fits) > best:
            best, besth = max(fits), P[fits.index(max(fits))]
        trace.append(round(best, 4))
    return round(best, 4), besth, spent, gens, trace


def main():
    seeds = SEEDS[:3] if SMOKE else SEEDS
    nproc = int(os.environ.get("NPROC", max(1, (os.cpu_count() or 2) - 2)))
    case = CASES[CASE]
    evfun = _surrogate if SMOKE else _bf_symmetric
    print(f"exp26: refill CONDIZIONATO, {CASE}, esteso, N={N}, {len(seeds)} semi, "
          f"{nproc} processi{'  [SMOKE, fitness surrogata, 0 FE]' if SMOKE else ''}\n", flush=True)

    # --- GATE 2 (budget): il conteggio della regione, RIFATTO E STAMPATO QUI ------------------
    r0 = random.Random(0)
    _d, tot_free = conditioned_sampler(ALPHA, N, r0, 0)
    for arm in ARMS:
        _d, m = conditioned_sampler(ALPHA, N, r0, THRESHOLD[arm])
        ok = m >= BUDGET_DISTINCT
        print(f"  GATE BUDGET {arm}: +-45 >= {THRESHOLD[arm]:>2} -> {m:,} half-stack "
              f"({100*m/tot_free:.4f}% del linguaggio)  vs budget {BUDGET_DISTINCT}: "
              f"{'OK' if ok else 'INSUFFICIENTE'}")
        if not ok:
            sys.exit("  regione piu' piccola del budget: il disegno non si puo' eseguire.")
    print(flush=True)

    # --- COLD dall'archivio, non si ripaga ----------------------------------------------------
    cold = json.load(open(COLD_PATH))["COLD"]["best"]
    print(f"  COLD (archivio exp24): n={len(cold)}, media {st.mean(cold):.4f}, "
          f"best {max(cold):.4f}\n", flush=True)

    out = {a: {"best": [], "pm45_final": [], "spent": [], "gens": [],
               "best_half": [], "trace": []} for a in ARMS}
    done = set()
    if os.path.exists(CKPT_PATH) and not SMOKE:
        ck = json.load(open(CKPT_PATH))
        out = ck["out"]
        done = {tuple(x) for x in ck["done"]}
        print(f"  checkpoint trovato: {len(done)} run gia' fatti, si riparte da li'\n", flush=True)

    pool = mp.Pool(nproc)
    try:
        for seed in seeds:
            for arm in ARMS:
                if (seed, arm) in [(s, a) for s, a in done]:
                    continue
                thr = THRESHOLD[arm]
                rng = random.Random(90000 + seed * 10 + 2 * ARMS.index(arm))
                draw, _m = conditioned_sampler(ALPHA, N, rng, thr)
                refill = draw                     # LA RIGA: i figli dalla STESSA regione
                seeded = [draw() for _ in range(POP)]
                b, h, spent, gens, trace = elitist_gated(case, pool, seeded, refill, thr, evfun)
                out[arm]["best"].append(b)
                out[arm]["pm45_final"].append(pm45(h))
                out[arm]["spent"].append(spent)
                out[arm]["gens"].append(gens)
                out[arm]["best_half"].append(list(h))
                out[arm]["trace"].append(trace)
                done.add((seed, arm))
                if not SMOKE:
                    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
                    json.dump({"out": out, "done": [list(x) for x in done]},
                              open(CKPT_PATH, "w"))
            line = "  ".join(f"{a} {out[a]['best'][-1]:.3f} (+-45 {out[a]['pm45_final'][-1]:>2}, "
                             f"{out[a]['spent'][-1]} distinti)"
                             for a in ARMS if out[a]["best"])
            print(f"  seme {seed:>2}: {line}", flush=True)
    finally:
        pool.close()
        pool.join()

    print("\n  GATE PERMANENZA: nessun individuo di nessuna generazione e' uscito dalla "
          "regione (il run muore alla prima violazione, quindi arrivare qui E' il gate).",
          flush=True)

    raw = {a: float(mannwhitneyu(out[a]["best"], cold, alternative="greater").pvalue)
           for a in ARMS}
    hol = holm(raw)
    beats = {a: hol[a] < 0.05 for a in ARMS}
    out["COLD_archive"] = dict(best=cold, source="exp24_seeded_basin.json")
    out["VERDICT"] = dict(
        means={a: round(st.mean(out[a]["best"]), 4) for a in ARMS} |
              {"COLD": round(st.mean(cold), 4)},
        bests={a: max(out[a]["best"]) for a in ARMS} | {"COLD": max(cold)},
        holm={a: hol[a] for a in ARMS}, beats_cold=beats,
        spent={a: round(st.mean(out[a]["spent"]), 1) for a in ARMS},
        pm45_final={a: round(st.mean(out[a]["pm45_final"]), 2) for a in ARMS},
        OUTCOME="CONFERMATO" if any(beats.values()) else "FALSIFICATO",
        smoke=SMOKE)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    final_path = (DATA_PATH + ".smoke") if SMOKE else DATA_PATH
    json.dump(out, open(final_path, "w"), indent=1)

    v = out["VERDICT"]
    print("\n=== PRE-REGISTRAZIONE #6 ===")
    for a in ("COLD",) + ARMS:
        extra = ("" if a == "COLD" else
                 f"   +-45 nel migliore finale {v['pm45_final'][a]:.1f}/{HALF}"
                 f"   distinti spesi {v['spent'][a]:.1f}/{BUDGET_DISTINCT}")
        print(f"  {a:<10} media {v['means'][a]:.3f}   best {v['bests'][a]:.3f}{extra}")
    for a in ARMS:
        print(f"  {a} batte COLD: {beats[a]}  (Holm p = {v['holm'][a]:.5f})")
    print(f"  >>> ESITO: {v['OUTCOME']}" + ("  [SMOKE: non e' un risultato]" if SMOKE else ""))
    print(f"\nscritto {final_path}")


if __name__ == "__main__":
    main()
