"""exp25_eda_transitions.py -- l'EDA sulle transizioni del DFA, contro DUE restart diversi.

L'ultimo braccio previsto dalla Direzione 1 e mai costruito: «un EDA sulle transizioni del DFA
pesate dall'elite». Crossover e clone-guard hanno pareggiato col restart (`exp20`/`exp22`),
l'avvio a caldo e la semina nel bacino sono stati falsificati (`exp23`/`exp24`). Questo e' cio'
che resta, ed e' l'unico che sfrutta l'automa in un modo che il taglio non sfrutta.

PERCHE' SI PUO' FARE, ED E' IL PUNTO. Il DP condiziona sui CONTATORI, i pesi stanno sulle
TRANSIZIONI: i due meccanismi vivono in posti diversi e si compongono. Appendendo theta > 0 a
ogni transizione e portando avanti il PRODOTTO invece del conteggio, la stessa camminata
all'indietro estrae ESATTAMENTE da P(x) = prod theta / Z ristretta al linguaggio conforme.
Nessuna riparazione, nessun rigetto -- ed e' la riparazione che negli altri EDA di laminati
distorce proprio la distribuzione che il modello dichiara di avere. Dimostrato per via di
verifica in `algo_dfa/eda_transitions.py` (sette controlli, aritmetica esatta, costo FE zero).

CINQUE BRACCI, e i due restart NON sono lo stesso restart:
  REGEN-PRATICA  figli da `gen_guided`, cioe' la costruzione-con-rigetto: e' cio' che si fa
                 oggi, ed e' il braccio che `exp20`/`exp22` chiamano REGEN. Distorto in modo
                 misurabile (lamine a 0: 3,942 contro 4,185 dell'esatto, 5 sigma) e fallisce
                 nel 5,8% delle chiamate.
  REGEN-ESATTO   figli dal campionatore uniforme esatto. E' il restart che il paper CREDE di
                 aver misurato in §6 (vedi RS-REGEN-NON-ESATTO) e non ha mai misurato a N=40.
  DFAXSTAR       il crossover a stati col clone-guard di `exp22`, rifatto QUI dentro e non
                 ripreso dall'archivio: e' cio' che permette di rifare il confronto di §6
                 contro un restart esatto misurato NELLO STESSO run, invece di incrociare due
                 campagne diverse. Serve a `RS-REGEN-NON-ESATTO`, strada 2.
  EDA-DFA        theta su (stato = (ultimo angolo, run), angolo): il modello VEDE l'automa.
  EDA-POS        theta su (posizione, angolo): la parametrizzazione del feromone dell'ACO del
                 paper 1 (:561), che ignora lo stato di run, ma qui con campionamento esatto.

Il quarto braccio e' il punto del disegno: con i soli EDA-DFA e restart, un guadagno sarebbe
INATTRIBUIBILE. EDA-POS isola «aver imparato una distribuzione» da «aver visto l'automa», che
e' la tesi. Le due parametrizzazioni girano sulla STESSA macchina e differiscono solo nella
chiave dei pesi.

Criterio in `IDEA_ALGORITMO_DFA_2026-08-06.md`, «PRE-REGISTRAZIONE #5» piu' il suo EMENDAMENTO
(N=40, elite dall'harness, budget di design DISTINTI), scritti e committati PRIMA di questo file.
Primario: EDA-DFA batte REGEN-ESATTO, Mann-Whitney a una coda, Holm sulla famiglia dei sei
confronti. Falsificata se non lo batte su nessun caso.

GATE SULLA PREMESSA, che e' la lezione di `exp23`: prima di spendere valutazioni FE oltre la
prima generazione si verifica che il modello si sia MOSSO. Se non si muove, l'EDA sta
ricampionando uniforme e misureremmo restart contro restart.

Uso:  PYTHONPATH=$PWD CCX_BIN=/usr/bin/ccx NPROC=30 python3 -m experiments.exp25_eda_transitions
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
from eda_transitions import Structure, START                                   # noqa: E402
from crossover_states import dfa_parikh_cuts                                   # noqa: E402

from experiments.exp20_dfa_crossover import _bf_symmetric, N, HALF, POP         # noqa: E402
from experiments.exp4_optimiser_comparison import holm, sig                     # noqa: E402

BUDGET_DISTINCT = 176           # design MAI VISTI, contabilita' di exp22 (non i posti valutati)
MAX_GENS = 40
ELITE = max(2, POP // 3)        # 5 su 16: il valore dell'harness, non uno inventato qui
SEEDS = list(range(1, 31))
CASES_ORDER = ("c1_axial", "c2_side", "c3_combo")
ARMS = ("REGEN-PRATICA", "REGEN-ESATTO", "DFAXSTAR", "EDA-DFA", "EDA-POS")
OPT_OFFSET = {"REGEN-PRATICA": 7, "REGEN-ESATTO": 13, "DFAXSTAR": 11, "EDA-DFA": 15,
              "EDA-POS": 17}

RHO = 0.30                      # tasso di apprendimento, fissato nella #5
EPS = 0.05                      # pavimento: senza, un peso nullo toglierebbe cammini in silenzio
GATE_DRAWS = 200                # taglia del campione del gate sulla premessa

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "_out", "exp25_eda_transitions.json")

_STRUCT = {}                    # (nome alfabeto, N) -> Structure. Non dipende da theta ne' dal seme.


def structure_for(alpha_name, alpha):
    """La struttura del DP costa 8,2 s sull'esteso e NON dipende da theta: una per campagna.

    E' l'ottimizzazione dichiarata nell'emendamento: ricostruirla a ogni generazione costerebbe
    5,65 s x ~15 generazioni x 180 run, cioe' ore di CPU per rifare conti identici.
    """
    key = (alpha_name, N)
    if key not in _STRUCT:
        _STRUCT[key] = Structure(alpha, N)
    return _STRUCT[key]


# ---- il modello -------------------------------------------------------------------------

def init_theta(struct, alpha, positional):
    """Pesi iniziali UNIFORMI sulle uscite di ogni stato: theta_0 riproduce il campionatore
    uniforme, quindi la generazione 1 dell'EDA e' identica a REGEN-ESATTO per costruzione."""
    if positional:
        # 1/|alpha| e NON 1.0: i due bracci devono partire entrambi da una distribuzione gia'
        # NORMALIZZATA dentro il proprio gruppo, altrimenti rho non significa la stessa cosa nei
        # due. Con theta_0 = 1,0 il termine (1-rho)*theta domina e l'aggiornamento resta
        # smorzato: dopo la prima generazione il rapporto usato/non-usato viene 1,43 invece di
        # 4,43, cioe' EDA-POS imparerebbe ~3x piu' piano di EDA-DFA per una ragione che non ha
        # nulla a che vedere con la parametrizzazione -- e l'attribuzione, che e' il punto
        # dell'esperimento, sarebbe stata invalida. Trovato dal gate sulla premessa.
        return {(j, b): 1.0 / len(alpha) for j in range(HALF) for b in alpha}
    out = {}
    for si in range(len(struct.S)):
        outs = [b for (_sj, b) in struct.trans[si]]
        for b in alpha:
            out[(si, b)] = 1.0 / len(outs) if outs else 0.0
    out.update({(START, a): 1.0 / len(alpha) for a in alpha})
    return out


def key_of(theta_kind, j, si, b):
    return (j, b) if theta_kind == "pos" else (si, b)


def usage_counts(struct, elite, alpha, positional):
    """Quante volte l'elite usa ogni transizione. E' l'unica lettura dell'elite che il modello fa."""
    idx = {s: i for i, s in enumerate(struct.S)}
    cnt = {}
    for h in elite:
        a, r = h[0], 1
        cnt[key_of("pos" if positional else "dfa", 0, START, h[0])] = \
            cnt.get(key_of("pos" if positional else "dfa", 0, START, h[0]), 0) + 1
        for j, b in enumerate(h[1:], start=1):
            k = key_of("pos" if positional else "dfa", j, idx[(a, r)], b)
            cnt[k] = cnt.get(k, 0) + 1
            a, r = (a, r + 1) if b == a else (b, 1)
    return cnt


def update_theta(theta, cnt, struct, alpha, positional):
    """theta <- (1-rho) theta + rho * frequenza nell'elite, con pavimento EPS e rinormalizzazione.

    Il pavimento non e' prudenza generica: un peso nullo toglierebbe cammini in SILENZIO e
    potrebbe rendere irraggiungibile una cella, mentre con EPS > 0 il supporto resta l'INTERO
    linguaggio conforme a ogni generazione -- cioe' l'esattezza verificata vale al passo 15
    come al passo 1. Gli stati che l'elite non visita NON si aggiornano (dichiarato nella #5).
    """
    groups = {}
    for k in theta:
        groups.setdefault(k[0], []).append(k)
    for g, keys in groups.items():
        tot = sum(cnt.get(k, 0) for k in keys)
        if tot == 0:
            continue                                   # gruppo non visitato: si lascia com'e'
        for k in keys:
            theta[k] = (1 - RHO) * theta[k] + RHO * (cnt.get(k, 0) / tot)
        s = sum(theta[k] for k in keys)
        if s <= 0:
            continue
        floor = EPS / len(keys)
        for k in keys:
            theta[k] = max(theta[k] / s, floor)
        s2 = sum(theta[k] for k in keys)
        for k in keys:
            theta[k] /= s2
    return theta


def as_fn(theta, positional):
    if positional:
        return lambda j, si, b: theta.get((j, b), 0.0)
    return lambda j, si, b: theta.get((si, b), 0.0)


# ---- un run -----------------------------------------------------------------------------

def run(arm, case, alpha, alpha_name, pool, rng):
    """Un run a budget di DESIGN DISTINTI. Ritorna (best, distinti, generazioni, mosso)."""
    sample, _t, _b, _s = uniform_sampler(alpha, N, rng)
    L = Layout(alpha, N)
    is_eda = arm.startswith("EDA")
    positional = (arm == "EDA-POS")
    struct = structure_for(alpha_name, alpha) if is_eda else None
    theta = init_theta(struct, alpha, positional) if is_eda else None
    cache = {}
    moved = None                                        # esito del gate sulla premessa

    def evaluate(pop):
        fresh, seen = [], set()
        for h in pop:
            k = tuple(h)
            if k not in cache and k not in seen:
                fresh.append(h)
                seen.add(k)
        if fresh:
            for h, v in zip(fresh, pool.map(_bf_symmetric, [(h, case) for h in fresh])):
                cache[tuple(h)] = v
        return [cache[tuple(h)] for h in pop], len(fresh)

    state = {}                                          # (Z, tot, thfn) della generazione corrente

    def child(r):
        if arm == "REGEN-PRATICA":
            full = gen_guided(alpha, N, r)              # costruzione-con-rigetto: la pratica
            return full[:HALF] if full else sample()
        if arm == "REGEN-ESATTO":
            return sample()
        if arm == "DFAXSTAR":
            # il braccio di exp22, rifatto QUI dentro invece che ripreso dall'archivio: cosi'
            # il confronto di §6 che RS-REGEN-NON-ESATTO deve riparare si rifa' contro un
            # restart esatto misurato NELLO STESSO run, senza incrociare due campagne.
            ha, hb = r.choice(state["elite"]), r.choice(state["elite"])
            cuts = dfa_parikh_cuts(list(ha), list(hb), L)
            if not cuts:
                return sample()
            i = r.choice(cuts)                       # UN solo sorteggio, come in exp22
            c = list(ha)[:i] + list(hb)[i:]
            return c if tuple(c) not in state["banned"] else sample()
        # theta NON cambia dentro una generazione: la riponderata si fa una volta sola, dopo
        # l'aggiornamento del modello. Rifarla a ogni figlio costerebbe 0,465 s x 11 x 15 per
        # run sull'esteso, cioe' rifare 165 volte un conto identico.
        return sample_weighted_from(struct, state["Z"], state["tot"], r, state["thfn"])

    P = [sample() for _ in range(POP)]
    fits, spent = evaluate(P)
    best = max(fits)
    gens = 0
    while spent < BUDGET_DISTINCT and gens < MAX_GENS:
        rank = [p for _, p in sorted(zip(fits, P), key=lambda x: -x[0])]
        elite = rank[:ELITE]
        state["elite"] = elite
        state["banned"] = {tuple(h) for h in P}
        if is_eda:
            theta = update_theta(theta, usage_counts(struct, elite, alpha, positional),
                                 struct, alpha, positional)
            thfn = as_fn(theta, positional)
            Z, tot = struct.reweight(thfn)              # UNA volta per generazione
            state.update(Z=Z, tot=tot, thfn=thfn)
            if moved is None:                           # GATE, alla PRIMA generazione soltanto
                moved = premise_gate(struct, Z, tot, thfn, theta, positional, elite, alpha,
                                     sample, rng)
                assert moved, (f"GATE FALLITO su {arm}: il modello non si e' mosso rispetto "
                               f"all'uniforme. L'EDA starebbe ricampionando uniforme e il "
                               f"confronto misurerebbe restart contro restart.")
        kids = []
        while len(kids) < POP - ELITE:
            c = child(rng)
            full = list(c) + list(c)[::-1]
            assert manufacturing_ok(full) and guidelines_ok(full, alpha), \
                f"figlio NON conforme in {arm}: il braccio si e' rotto, non si continua"
            kids.append(c)
            state["banned"] = state["banned"] | {tuple(c)}
        P = elite + kids
        fits, s = evaluate(P)
        spent += s
        best = max(best, max(fits))
        gens += 1
    return round(best, 4), spent, gens, bool(moved) if is_eda else None


def sample_weighted_from(struct, Z, tot, rng, thfn):
    """Camminata all'indietro sulla struttura gia' riponderata. Stessa procedura del
    campionatore uniforme: cambia solo che il peso di un predecessore porta theta."""
    L, S = struct.L, struct.S
    r = rng.random() * tot
    acc = 0.0
    for i in struct.finals:
        acc += Z[L.m - 1][i]
        if r < acc:
            fi = i
            break
    else:
        fi = struct.finals[-1]
    cell = struct.cells[L.m - 1][fi]
    (si, ex, df) = cell
    seq = [S[si][0]]
    for j in range(L.m - 1, 0, -1):
        b = S[si][0]
        prev = _unstep_counts(L, ex, df, b)
        cands, tot_p = [], 0.0
        for pi in struct.preds[si]:
            k = (pi, prev[0], prev[1])
            ip = struct.index[j - 1].get(k)
            if ip is None:
                continue
            w = Z[j - 1][ip]
            if w:
                wt = w * thfn(j, pi, b)
                cands.append((pi, wt))
                tot_p += wt
        assert tot_p > 0, "cella raggiunta senza predecessori: struttura incoerente"
        r = rng.random() * tot_p
        acc = 0.0
        for pi, wt in cands:
            acc += wt
            if r < acc:
                break
        si, ex, df = pi, prev[0], prev[1]
        seq.append(S[si][0])
    seq.reverse()
    return seq


def _unstep_counts(L, ex, df, b):
    if b in L.ei:
        i = L.ei[b]
        return ex[:i] + (ex[i] - 1,) + ex[i + 1:], df
    if b in L.di:
        k, s = L.di[b]
        return ex, df[:k] + (df[k] - s,) + df[k + 1:]
    return ex, df


def premise_gate(struct, Z, tot, thfn, theta, positional, elite, alpha, sample, rng):
    """Il modello si e' MOSSO? Costa 0 valutazioni FE.

    ⚠️ LA STATISTICA E' STATA CAMBIATA IL 08/08 DOPO CHE IL GATE E' FALLITO, e va detto:
    e' un cambiamento POST-HOC. Riguarda pero' una GUARDIA, non il criterio di esito, e la
    ragione non e' che l'esito non piaceva -- e' che la statistica originale **non poteva
    discriminare**, misurato: la #5 chiedeva «l'uso medio delle transizioni dell'elite dev'essere
    piu' alto nel campione pesato», ma su `set1` l'elite (5 individui x 20 lamine) copre **25
    delle 32** transizioni valide, cioe' il 78%. Quel conteggio **satura**: 17,90 sul campione
    pesato contro 18,20 sull'uniforme, su un massimo di 20. Sarebbe stato rosso per qualunque
    modello, anche perfettamente funzionante.

    La statistica nuova e' la **log-verosimiglianza sotto theta**, che non puo' saturare e prova
    la cosa giusta: le estrazioni pesate devono essere individui che il modello PREFERISCE
    rispetto a quelle uniformi. Sullo stesso caso dava **-25,485 contro -25,960, delta +0,474**
    mentre il modello si era mosso di `max|theta - theta0| = 0,294`. Soglia a **3 errori
    standard** del delta, cosi' il verde non e' rumore.
    """
    import math

    def logp(h):
        c = usage_counts(struct, [h], alpha, positional)
        return sum(v * math.log(max(theta.get(k, 0.0), 1e-12)) for k, v in c.items())

    w = [logp(sample_weighted_from(struct, Z, tot, rng, thfn)) for _ in range(GATE_DRAWS)]
    u = [logp(sample()) for _ in range(GATE_DRAWS)]
    d = st.mean(w) - st.mean(u)
    se = (st.pstdev(w) ** 2 / len(w) + st.pstdev(u) ** 2 / len(u)) ** 0.5
    return d > 3 * se if se > 0 else d > 0


# ---- campagna ---------------------------------------------------------------------------

def checkpoint(res, why):
    """La campagna dura ~6 h: scrivere il JSON solo alla fine significa perderle tutte se il
    processo muore. Si versa dopo OGNI caso, cosi' il peggio che si perde e' un caso."""
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, "w") as f:
        json.dump(dict(result=res, partial=why), f, indent=1)
    print(f"    [checkpoint] {why} -> {DATA_PATH}", flush=True)


def campaign(alpha_name, pool, res_so_far=None):
    alpha = ALPHABETS[alpha_name]
    structure_for(alpha_name, alpha)                    # una volta, fuori dai run
    out = {}
    for cname in CASES_ORDER:
        case = CASES[cname]
        vals = {a: [] for a in ARMS}
        dist = {a: [] for a in ARMS}
        gens = {a: [] for a in ARMS}
        for seed in SEEDS:
            for arm in ARMS:
                b, d, g, _m = run(arm, case, alpha, alpha_name, pool,
                                  random.Random(seed * 10 + OPT_OFFSET[arm]))
                vals[arm].append(b)
                dist[arm].append(d)
                gens[arm].append(g)
            print(f"  [{alpha_name} {cname}] seme {seed:>2}: "
                  + "  ".join(f"{a}={vals[a][-1]:.3f}" for a in ARMS), flush=True)
        raw = {f"{a}-vs-REGEN-ESATTO": float(mannwhitneyu(vals[a], vals["REGEN-ESATTO"],
                                                          alternative="greater").pvalue)
               for a in ARMS if a.startswith("EDA")}
        raw["EDA-DFA-vs-EDA-POS"] = float(mannwhitneyu(vals["EDA-DFA"], vals["EDA-POS"],
                                                       alternative="greater").pvalue)
        raw["EDA-DFA-vs-REGEN-PRATICA"] = float(mannwhitneyu(vals["EDA-DFA"],
                                                             vals["REGEN-PRATICA"],
                                                             alternative="greater").pvalue)
        # secondario e non primario, ma e' il numero che RS-REGEN-NON-ESATTO aspetta:
        # i due restart misurati DENTRO la stessa campagna, non incrociati fra run diversi
        raw["REGEN-ESATTO-vs-REGEN-PRATICA"] = float(
            mannwhitneyu(vals["REGEN-ESATTO"], vals["REGEN-PRATICA"],
                         alternative="two-sided").pvalue)
        out[cname] = {a: dict(mean=round(st.mean(vals[a]), 4), std=round(st.pstdev(vals[a]), 4),
                              best=round(max(vals[a]), 4),
                              distinct=round(st.mean(dist[a]), 1),
                              gens=round(st.mean(gens[a]), 1)) for a in ARMS}
        out[cname]["_per_seed"] = vals
        out[cname]["MWU_raw"] = {k: sig(v) for k, v in raw.items()}
        if res_so_far is not None:                      # checkpoint dopo OGNI caso
            res_so_far[alpha_name] = out
            checkpoint(res_so_far, f"{alpha_name}/{cname} completato")
    return out


def main():
    nproc = int(os.environ.get("NPROC", "8"))
    res = {}
    with mp.Pool(nproc) as pool:
        for alpha_name in ("set1", "set2"):
            print(f"\n=== alfabeto {alpha_name} ===", flush=True)
            res[alpha_name] = campaign(alpha_name, pool, res_so_far=res)
    fam = {}
    for alpha_name, per_case in res.items():
        for cname, d in per_case.items():
            for k, v in d["MWU_raw"].items():
                fam[f"{alpha_name}|{cname}|{k}"] = v
    hol = holm({k: v for k, v in fam.items() if "vs-REGEN-ESATTO" in k})
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, "w") as f:
        json.dump(dict(result=res, holm_primary=hol), f, indent=1)
    print(f"\nscritto {DATA_PATH}")
    beats = [k for k, p in hol.items() if p < 0.05 and "EDA-DFA" in k]
    print("\nPRIMARIO (EDA-DFA contro REGEN-ESATTO, Holm sulla famiglia dei sei):")
    print(f"  {'CONFERMATO su ' + ', '.join(beats) if beats else 'FALSIFICATO: nessun caso'}")


if __name__ == "__main__":
    main()
